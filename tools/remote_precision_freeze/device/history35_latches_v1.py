"""Default-off, startup-only selected-query capture for the current candidate.

Model arguments/results and cache data are never replaced. GPU selection nodes
are recorded even when a warmup position does not match, so CUDA Graph replay
can reach a later target. A local controller must drain its request before
asking the private Unix socket to reset or export the fixed GPU latches.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import socket
import sys
import threading
from pathlib import Path


_config = None
_buffers = {}
_metadata = {}
_context = threading.local()
_lock = threading.RLock()
_backend_identity = None
_source_identity = {}
_server_pid = None
_installed = False


def validate_config(value):
    for key in ("task_id", "source_commit", "wheel_sha256", "input_ids_sha256"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("missing controller identity: " + key)
    for key in ("wheel_sha256", "input_ids_sha256", "flashmla_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get(key, "")):
            raise ValueError("invalid SHA256: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]):
        raise ValueError("source_commit must be the full candidate commit")
    for name in ("attention_v41", "_v41_mega_attention"):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get("python_sha256", {}).get(name, "")):
            raise ValueError("missing candidate Python fingerprint: " + name)
    for key in ("layers", "positions"):
        items = value.get(key)
        if not isinstance(items, list) or not 1 <= len(items) <= 3:
            raise ValueError("selector needs one to three " + key)
        if len(set(items)) != len(items) or any(type(x) is not int or x < 0 for x in items):
            raise ValueError("invalid selector: " + key)
    if not Path(value["output"]).is_absolute():
        raise ValueError("capture output must be an absolute task directory")
    if not 0 < int(value.get("gpu_budget_bytes", 8 * 1024**2)) <= 16 * 1024**2:
        raise ValueError("GPU latch budget exceeds 16 MiB per process")
    return value


def _save_json(path, value):
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    pending.replace(path)


def _control_server(server, root):
    """Explicit controller operations; no business request or state polling."""
    import torch

    while True:
        connection, _ = server.accept()
        with connection:
            connection.settimeout(10)
            try:
                payload = b""
                while not payload.endswith(b"\n") and len(payload) <= 8192:
                    chunk = connection.recv(8192)
                    if not chunk:
                        break
                    payload += chunk
                if len(payload) > 8192 or not payload.endswith(b"\n"):
                    raise ValueError("capture request must be one bounded JSON line")
                request = json.loads(payload)
                if request.get("request_drained") is not True:
                    raise ValueError("controller must attest request EOF and no next request")
                operation = request["operation"]
                if operation not in ("reset", "snapshot"):
                    raise ValueError("unsupported capture operation")
                label = request.get("label", "")
                if operation == "snapshot" and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", label):
                    raise ValueError("invalid phase label")
                with _lock:
                    devices = {v[0].device for v in _buffers.values()}
                    for device in devices:
                        torch.cuda.synchronize(device)
                    if operation == "reset":
                        for data, hit in _buffers.values():
                            data.zero_()
                            hit.zero_()
                        for device in devices:
                            torch.cuda.synchronize(device)
                        result = {"state": "CAPTURE_LATCHES_RESET", "PID": os.getpid()}
                    else:
                        destination = root / (str(os.getpid()) + "-" + label + ".pt")
                        if destination.exists():
                            raise ValueError("phase already exported; choose a new batch label")
                        tensors, hits = {}, {}
                        for key, (data, hit) in _buffers.items():
                            observed = hit.cpu().tolist()
                            if observed[0] == 0:
                                continue
                            if observed[0] != 1:
                                raise RuntimeError("target observed multiple times; phase not isolated: " + key)
                            tensors[key] = data.cpu()
                            hits[key] = observed
                        # Empty DP/CP ranks are explicit evidence, not a query hit.
                        # The enclosing controller must join all four ranks.
                        pass
                        pending = destination.with_suffix(".partial")
                        torch.save(tensors, pending)
                        size = pending.stat().st_size
                        if size > 20 * 1024**2:
                            raise RuntimeError("export exceeds the 20 MiB serialized budget")
                        digest = hashlib.sha256(pending.read_bytes()).hexdigest()
                        pending.replace(destination)
                        result = {"state": "CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE",
                                  "PID": os.getpid(), "coverage_empty": not bool(tensors), "file": str(destination), "bytes": size,
                                  "SHA256": digest, "identity": _config,
                                  "flashmla_loaded": _backend_identity, "hits": hits,
                                  "python_loaded": _source_identity,
                                  "metadata": {k: _metadata[k] for k in tensors}}
                        _save_json(destination.with_suffix(".json"), result)
                connection.sendall((json.dumps(result) + "\n").encode())
            except BaseException as error:
                connection.sendall((json.dumps({"state": "CAPTURE_CONTROL_FAILED",
                                                "error": repr(error)}) + "\n").encode())


def _ensure_server():
    global _server_pid
    if _server_pid == os.getpid():
        return
    root = Path(_config["output"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise RuntimeError("capture output directory must be private (mode 0700)")
    path = root / (str(os.getpid()) + ".sock")
    if len(str(path).encode()) >= 104 or path.exists():
        raise RuntimeError("private capture socket path occupied or too long")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o600)
    server.listen(1)
    _save_json(root / (str(os.getpid()) + "-control.json"), {
        "state": "CAPTURE_CONTROL_READY", "PID": os.getpid(),
        "UID": os.getuid(), "socket": str(path), "identity": _config,
        "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
    })
    _server_pid = os.getpid()
    threading.Thread(target=_control_server, args=(server, root), daemon=True,
                     name="bounded-attention-capture").start()


def _capture(name, value, positions, **metadata):
    import torch

    current = getattr(_context, "value", None)
    if current is None or not isinstance(value, torch.Tensor) or not value.is_cuda:
        return
    if positions.device != value.device or not positions.is_cuda:
        raise RuntimeError("capture query and value devices differ")
    rows = positions.numel()
    if not rows or value.numel() % rows:
        raise RuntimeError("capture row mapping is unsupported: " + name)
    original_dtype = str(value.dtype)
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        value = value.view(torch.uint8)
    width = value.numel() // rows
    if width * value.element_size() > 256 * 1024:
        raise RuntimeError("one latch exceeds 256 KiB: " + name)
    for target in _config["positions"]:
        key = "%s-L%d-T%d-%s" % (current["role"], current["layer"], target, name)
        with _lock:
            if key not in _buffers:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("capture geometry was not eagerly warmed: " + key)
                required = width * value.element_size() + 8
                used = sum(data.numel() * data.element_size() + 8 for data, _ in _buffers.values())
                if len(_buffers) >= 192 or used + required > _config.get("gpu_budget_bytes", 8 * 1024**2):
                    raise RuntimeError("bounded capture GPU/object budget exceeded")
                _buffers[key] = (torch.zeros((1, width), dtype=value.dtype, device=value.device),
                                 torch.zeros(2, dtype=torch.int32, device=value.device))
                attn = current.get("attn")
                fields = ("tp_size", "tp_rank", "n_heads", "kv_source_layer_id", "index_source_layer_id", "compress_ratio")
                topology = {field: getattr(attn, field) for field in fields if type(getattr(attn, field, None)) is int}
                _metadata[key] = {"original_dtype": original_dtype, "shape_per_query": [width],
                                  "cuda_device_index": value.device.index, "attention": topology, **metadata}
                _ensure_server()
            buffer, hit = _buffers[key]
            if buffer.shape != (1, width) or buffer.dtype != value.dtype:
                raise RuntimeError("capture query geometry changed after eager warmup: " + key)
        flat = value.reshape(rows, width)
        mask = positions.reshape(-1) == target
        selected = mask.to(torch.int64).argmax().reshape(1)
        found = mask.any()
        buffer.copy_(torch.where(found, flat.index_select(0, selected), buffer))
        hit[0].add_(mask.sum().to(torch.int32))
        hit[1].copy_(torch.where(found, torch.full_like(hit[1], target), hit[1]))


def _selected_rows(value, positions):
    """Capture selected KV rows per target, without flattening a padded pool."""
    width = value.shape[-1]
    for start in range(0, value.shape[1], 128):
        chunk = value[:, start:start + 128]
        _capture("KV_" + str(start), chunk, positions, record_bytes=width)


def _one_query(positions, target):
    """Select on GPU; -1 prevents a missing target from matching another latch."""
    import torch

    mask = positions.reshape(-1) == target
    row = mask.to(torch.int64).argmax().reshape(1)
    selected = positions.reshape(-1).index_select(0, row)
    return row, torch.where(mask.any(), selected, torch.full_like(selected, -1))


def bind_attention(module):
    _verify_source(module, "attention_v41")
    cls = module.AttentionV41FP8
    for method, role in (("_prefill_query", "P"), ("_forward_decode_body", "D")):
        original = getattr(cls, method)
        def wrap(original, role):
            @functools.wraps(original)
            def call(self, *args, **kwargs):
                if self.layer_id not in _config["layers"]:
                    return original(self, *args, **kwargs)
                previous = getattr(_context, "value", None)
                _context.value = {"attn": self, "role": role, "layer": self.layer_id}
                try:
                    return original(self, *args, **kwargs)
                finally:
                    _context.value = previous
            return call
        setattr(cls, method, wrap(original, role))
    for method in ("_select_indices", "_select_indices_decode"):
        original = getattr(cls, method)
        def wrap_selection(original):
            @functools.wraps(original)
            def call(self, *args, **kwargs):
                output = original(self, *args, **kwargs)
                if getattr(_context, "value", None) is not None:
                    positions = args[2] if len(args) > 2 else kwargs["positions"]
                    _capture("GLOBAL_selected", output, positions)
                return output
            return call
        setattr(cls, method, wrap_selection(original))


def bind_mega(module):
    _verify_source(module, "_v41_mega_attention")
    cls = module.MegaAttention
    for method in ("prefill", "decode"):
        original = getattr(cls, method)
        def wrap(original, method):
            @functools.wraps(original)
            def call(self, *args, **kwargs):
                global _backend_identity
                context = getattr(_context, "value", None)
                if context is None:
                    return original(self, *args, **kwargs)
                import torch
                if _backend_identity is None:
                    path = Path(self.backend.__file__).resolve()
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest != _config["flashmla_sha256"]:
                        raise RuntimeError("actual FlashMLA DSO differs from capture contract")
                    _backend_identity = {"path": str(path), "SHA256": digest}
                q, pool, indices, positions = args[:4]
                rows = positions.numel()
                index = indices.reshape(rows, -1)
                if not rows or index.shape[1] > 640:
                    raise RuntimeError("selected query width exceeds the capture contract")
                if q.shape[-2] == self.padded_heads:
                    q_stride = q.reshape(rows, self.padded_heads, 512).stride()
                else:
                    # Current pad_q pads each 16-channel chunk, producing a
                    # contiguous [M,padded_heads,512] allocation inside original.
                    q_stride = (self.padded_heads * 512, 512, 1)
                geometry = [rows, self.padded_heads, *q_stride, pool.shape[1], self.padded_groups, *pool.stride()]
                meta = torch.stack([torch.full_like(positions.reshape(-1), int(v)) for v in geometry], -1)
                _capture("call_geometry", meta, positions)
                _capture("indices", index, positions)
                _capture("sink", self.sink.reshape(1, -1).expand(rows, -1), positions)
                _capture("cos_sin", self.cos_sin.index_select(0, positions.long()), positions)
                _capture("softmax_scale", torch.full_like(positions.reshape(-1, 1), self.softmax_scale, dtype=torch.float64), positions)
                if method == "prefill":
                    lengths = args[4] if len(args) > 4 else kwargs.get("topk_length")
                    attn = context["attn"]
                    if attn.compress_ratio:
                        offsets, count, _ = attn._shared_attention["prefill_chunk_meta"]
                        base = offsets + count
                        if isinstance(base, torch.Tensor) and base.numel() not in (1, rows):
                            raise RuntimeError("ragged chunk metadata needs an explicit row map; unsupported capture")
                else:
                    lengths = kwargs.get("swa_lengths")
                    extra = kwargs.get("extra_pool")
                    slots = kwargs.get("extra_indices")
                    if extra is not None:
                        slots = slots.reshape(rows, -1)
                        if slots.shape[1] > 512 or extra.shape[-1] != 288:
                            raise RuntimeError("unsupported GLOBAL capture geometry")
                        _capture("GLOBAL_indices", slots, positions)
                        _capture("GLOBAL_entries", torch.full_like(positions.reshape(-1, 1), extra.shape[1]), positions)
                        strides = torch.stack([torch.full_like(positions.reshape(-1), int(v)) for v in extra.stride()], -1)
                        _capture("GLOBAL_stride", strides, positions)
                        extra_lengths = kwargs.get("extra_lengths")
                        _capture("GLOBAL_length", torch.full_like(positions.reshape(-1, 1), -1) if extra_lengths is None else extra_lengths.reshape(rows, -1), positions)
                # Gather only selected target queries, never M * 640 KV rows.
                # These device nodes also exist when eager warmup misses target.
                for target in _config["positions"]:
                    row, one_position = _one_query(positions, target)
                    one_index = index.index_select(0, row)
                    one_q = q.reshape(rows, -1, 512).index_select(0, row)
                    _capture("Q", self.pad_q(one_q), one_position, layout="native_mega_chunk_interleaved")
                    physical = one_index.clamp_min(0).long()
                    selected = pool[physical // pool.shape[1], physical % pool.shape[1]]
                    if method == "prefill":
                        if attn.compress_ratio:
                            one_base = base
                            if isinstance(base, torch.Tensor) and base.numel() > 1:
                                one_base = base.reshape(rows, -1).index_select(0, row)
                            groups = torch.where(one_index < 0, -1, (one_index >= one_base).to(torch.int32))
                        else:
                            groups = torch.where(one_index < 0, -1, torch.ones_like(one_index))
                        _capture("groups", groups, one_position, meaning="0=GLOBAL,1=SWA,-1=invalid")
                        _selected_rows(selected, one_position)
                    else:
                        if pool.shape[-1] != 528 or index.shape[1] > 128:
                            raise RuntimeError("unsupported SWA capture geometry")
                        _capture("SWA_raw", selected, one_position, record_bytes=528)
                        if extra is not None:
                            physical = slots.index_select(0, row).clamp_min(0).long()
                            records = extra[physical // extra.shape[1], physical % extra.shape[1]]
                            _capture("GLOBAL_raw", records, one_position, record_bytes=288)
                _capture("length", torch.full_like(positions.reshape(-1, 1), -1) if lengths is None else lengths.reshape(rows, -1), positions)
                output = original(self, *args, **kwargs)
                _capture("pair_data", output[0], positions)
                _capture("pair_scales", output[1], positions)
                return output
            return call
        setattr(cls, method, wrap(original, method))


def _verify_source(module, name):
    path = Path(module.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _config["python_sha256"][name]:
        raise RuntimeError("loaded Python differs from capture candidate: " + name)
    _source_identity[name] = {"path": str(path), "SHA256": digest}


def install():
    global _config, _installed
    if os.environ.get("PUBPREC_CAPTURE", "0") != "1":
        return False
    if _installed:
        return True
    _config = validate_config(json.loads(Path(os.environ["PUBPREC_CAPTURE_CONFIG"]).read_text()))
    targets = {
        "rtp_llm.models_py.modules.dsv41.fp8.attention_v41": bind_attention,
        "rtp_llm.models_py.modules.dsv41.fp8._v41_mega_attention": bind_mega,
    }
    if any(name in sys.modules for name in targets):
        raise RuntimeError("capture must be installed before model imports; no hot patching")
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            original = loader.exec_module
            def execute(module):
                original(module)
                targets[fullname](module)
            loader.exec_module = execute
            return spec
    sys.meta_path.insert(0, Finder())
    _installed = True
    return True
