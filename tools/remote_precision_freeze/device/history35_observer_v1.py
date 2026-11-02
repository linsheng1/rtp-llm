"""Finite historical7042/7043 adapter for the existing graph-safe latches.

Only snapshots natural tensors. Keeps the original decode CUDA Graph setting.
Reset/export and request isolation belong to the original device controller;
the GPU positions and input-token samples are joined with the sole native RID.
An empty DP/CP rank is recorded explicitly, never counted as a sampled query.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

BLOCK = 'rtp_llm.models_py.modules.dsv41.block'
ATTN = 'rtp_llm.models_py.modules.dsv41.fp8.attention_v41'
FRONT = 'rtp_llm.models_py.modules.dsv41.fp8.attention'
MOE = 'rtp_llm.models_py.modules.dsv41.moe.moe_layer'
COMP = 'rtp_llm.models_py.modules.dsv41.fp8.compressor'
GLOBAL = 'rtp_llm.models_py.modules.dsv41.fp8._v41_decode_global'
TARGETS = (BLOCK, ATTN, FRONT, MOE, COMP, GLOBAL)
_capture = None
_params = {}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), SHA256=hashlib.sha256(raw).hexdigest())


def rows(value, positions):
    need(value.numel() % positions.numel() == 0, 'actual tensor/position geometry differs')
    return value.reshape(positions.numel(), -1)


def record(label, value, positions, **metadata):
    if hasattr(value, 'device'):
        _capture._capture(label, rows(value, positions), positions, **metadata)


def parameter(name, value):
    if not hasattr(value, 'device'):
        return
    identity = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))
    old = _params.setdefault(name, (value, identity))
    need(old[0] is value and old[1] == identity, 'actual parameter reference changed: ' + name)


def bind_parameters(attn):
    layer = attn.layer_id
    for name in ['global_wkv', 'global_wgate', 'global_norm', 'kv_norm', 'q_norm']:
        parameter('L%d.%s' % (layer, name), getattr(attn, name, None))
    for name in ['wkv', 'wq_a_wkv']:
        linear = getattr(attn, name, None)
        for field in ['weight', 'weight_scales']:
            parameter('L%d.%s.%s' % (layer, name, field), getattr(linear, field, None))


def domain(block, args, kwargs, mode):
    import torch
    hidden = args[0] if args else kwargs['x']
    if mode == 'D':
        meta = args[1] if len(args) > 1 else kwargs['attn_metadata']
        count = hidden.shape[0] * hidden.shape[1]
        positions = meta.position_ids[:count].reshape(-1)
        valid = getattr(meta, 'valid_requests', None)
        if valid is not None:
            mask = valid[:hidden.shape[0]].reshape(-1, 1).expand(-1, hidden.shape[1]).reshape(-1)
            positions = torch.where(mask, positions, torch.full_like(positions, -1))
        ids = args[2] if len(args) > 2 else kwargs.get('input_ids')
    else:
        cp = getattr(block.attn, '_cp_ctx', None)
        need(cp is not None and cp.cp_size == 4 and cp.kv_cache_sharded,
             'history P requires actual CP4 sharded domain')
        positions = torch.where(cp.local_is_real, cp.global_positions,
                                torch.full_like(cp.global_positions, -1))
        ids = args[1] if len(args) > 1 else kwargs.get('input_ids')
    need(positions.is_cuda and hidden.numel() == positions.numel()*4*5120,
         'actual historical block rows/positions differ')
    return hidden, ids, positions


def block_wrapper(original, mode):
    @functools.wraps(original)
    def call(block, *args, **kwargs):
        if block.layer_id not in [0, 1, 2]:
            return original(block, *args, **kwargs)
        previous = getattr(_capture._context, 'value', None)
        if previous is not None and previous.get('block') is block:
            return original(block, *args, **kwargs)
        if mode == 'P':
            cp = getattr(block.attn, '_cp_ctx', None)
            if cp is None or cp.cp_size != 4 or not cp.kv_cache_sharded:
                # Unbound startup materialization has no CP row identity.
                # Actual request coverage is checked by the four-rank join.
                return original(block, *args, **kwargs)
        hidden, ids, positions = domain(block, args, kwargs, mode)
        _capture._context.value = dict(role=mode, layer=block.layer_id, attn=block.attn,
                                       block=block, positions=positions)
        try:
            bind_parameters(block.attn)
            record('block_entry4HC', hidden, positions)
            if ids is not None:
                need(ids.numel() == positions.numel(), 'actual input IDs/positions differ')
                record('actual_input_token', ids, positions)
            return original(block, *args, **kwargs)
        finally:
            _capture._context.value = previous
    return call


def attention_wrapper(original, method):
    @functools.wraps(original)
    def call(attn, *args, **kwargs):
        context = getattr(_capture._context, 'value', None)
        if context is None or attn is not context['attn']:
            return original(attn, *args, **kwargs)
        x = args[0] if args else kwargs['x']
        positions = context['positions']
        if method in ['_prefill_produce', '_forward_decode_body']:
            record('attn_input5120', x, positions)
            quantized = kwargs.get('shared_input_quant')
            if quantized is None and len(args) > 2:
                quantized = args[2]
            if quantized is not None:
                for i, value in enumerate(quantized):
                    record('actual_shared_input_quant_%d' % i, value, positions)
        result = original(attn, *args, **kwargs)
        if method in ['_prefill_query', '_forward_decode_body']:
            record('attn_output5120', result, positions)
        if attn.layer_id == 2 and method == '_prefill_produce':
            common, qkv = result[:2]
            cp = common.cp_ctx
            if (common.cp_on and tuple(cp.prefix_lengths_host) == (0,) and
                    len(cp.input_lengths_global_host) == 1 and
                    qkv.kv_full.shape[0] == cp.input_lengths_global_host[0]):
                import torch
                full_positions = torch.arange(qkv.kv_full.shape[0], device=x.device, dtype=positions.dtype)
                record('fresh_SWA_decoded512', qkv.kv_full, full_positions,
                       domain='actual single prefix0 CP gather, all ranks replica')
                # A complete compressed row has one actual CP owner. Access
                # only the two selected slot IDs, never copy a full cache pool.
                selected = torch.tensor([7042, 7043], device=x.device, dtype=positions.dtype)
                slots = attn._slots(attn._global_region(), selected, torch.zeros_like(selected))
                pool = attn._source_pool(attn._global_region())
                need(pool.dtype == torch.uint8 and pool.ndim == 3 and pool.shape[-1] == 288,
                     'actual GLOBAL288 pool required')
                valid = ((slots >= 0) & (selected < qkv.kv_full.shape[0]) &
                         ((selected+1) % attn.compress_ratio == 0))
                safe = slots.clamp_min(0)
                raw = pool[safe // pool.shape[1], safe % pool.shape[1]]
                actual = torch.where(valid, selected, torch.full_like(selected, -1))
                record('GLOBAL_raw288', raw, actual, domain='actual CP-owned completed row')
        if attn.layer_id == 2 and method == '_forward_decode_body':
            import torch
            from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
            meta = args[1] if len(args) > 1 else kwargs['metadata']
            count = positions.numel()
            slots = meta.pool_write_slot_mappings.get(SWA_KV)
            need(slots is not None, 'actual decode SWA write slots required')
            pool = attn._pool_view_3d_fp8(SWA_KV)
            need(pool.dtype == torch.uint8 and pool.ndim == 3 and pool.shape[-1] == 528,
                 'actual decode SWA528 pool required')
            slots = slots[:count].reshape(-1)
            safe = slots.clamp_min(0)
            raw = pool[safe // pool.shape[1], safe % pool.shape[1]]
            actual = torch.where(slots >= 0, positions, torch.full_like(positions, -1))
            record('SWA_raw528', raw, actual, domain='actual decode write slots after attention')
            req = meta.req_id_per_token[:count].long()
            slots = attn._slots(attn._global_region(), positions, req)
            pool = attn._source_pool(attn._global_region())
            safe = slots.clamp_min(0)
            raw = pool[safe // pool.shape[1], safe % pool.shape[1]]
            valid = (slots >= 0) & ((positions+1) % attn.compress_ratio == 0)
            actual = torch.where(valid, positions, torch.full_like(positions, -1))
            record('GLOBAL_raw288', raw, actual, domain='actual decode completed row after producer join')
        return result
    return call


def patch(name, module):
    expected = _capture._config['hook_source_sha256'][name]
    need(pin(module.__file__)['SHA256'] == expected, 'actual historical hook source drift: ' + name)
    _capture._source_identity[name] = pin(module.__file__)
    if name == BLOCK:
        for method, mode in [('forward', 'P'), ('_forward_prefill_fast_fp8', 'P'), ('forward_decode', 'D')]:
            original = getattr(module.Block, method)
            setattr(module.Block, method, block_wrapper(original, mode))
    elif name == ATTN:
        for method in ['_prefill_produce', '_prefill_query', '_forward_decode_body']:
            original = getattr(module.AttentionV41FP8, method)
            setattr(module.AttentionV41FP8, method, attention_wrapper(original, method))
        original = module.AttentionV41FP8._write_states
        signature = inspect.signature(original)
        @functools.wraps(original)
        def states(attn, *args, **kwargs):
            context = getattr(_capture._context, 'value', None)
            if context is not None and attn.layer_id == 2:
                bound = signature.bind(attn, *args, **kwargs).arguments
                for key in ['values', 'scores']:
                    record('compressor_'+key+'512', bound[key], bound['positions'])
            return original(attn, *args, **kwargs)
        module.AttentionV41FP8._write_states = states
    elif name == MOE:
        original = module.MoE.forward
        @functools.wraps(original)
        def moe(moe, *args, **kwargs):
            context = getattr(_capture._context, 'value', None)
            active = context is not None and context['layer'] in [0,1] and moe is context['block'].ffn
            if active:
                record('FFN_input5120', args[0] if args else kwargs['x'], context['positions'])
            result = original(moe, *args, **kwargs)
            if active:
                record('FFN_output5120', result, context['positions'])
            return result
        module.MoE.forward = moe
    elif name == COMP:
        original = module._linear_bf16_bf16_fp32
        @functools.wraps(original)
        def linear(x, weight, *args, **kwargs):
            result = original(x, weight, *args, **kwargs)
            context = getattr(_capture._context, 'value', None)
            if context is not None and context['role']=='D' and context['layer']==2:
                attn = context['attn']
                label = 'values' if weight is attn.global_wkv else 'scores' if weight is attn.global_wgate else None
                if label is not None:
                    record('compressor_'+label+'512', result, context['positions'])
            return result
        module._linear_bf16_bf16_fp32 = linear
    elif name == GLOBAL:
        original = module.try_produce_global
        @functools.wraps(original)
        def global_producer(attn, x, positions, req_ids, starts, **kwargs):
            projected = kwargs.get('projected')
            context = getattr(_capture._context, 'value', None)
            if context is not None and context['role']=='D' and attn.layer_id == 2 and projected is not None:
                for label, value in zip(['values','scores'], projected):
                    record('compressor_'+label+'512', value, context['positions'])
            return original(attn, x, positions, req_ids, starts, **kwargs)
        module.try_produce_global = global_producer


def install(config):
    global _capture
    need(config['schema']=='history35-inner-graph-latches-v1' and config['layers']==[0,1,2]
         and config['positions']==[7042,7043] and config['artifact_kind']=='inner_python_overlay'
         and config['wheel_sha256'] is None and config['gpu_budget_bytes']==4*1024**2,
         'explicit minimal historical inner profile required')
    need(set(config['hook_source_sha256']) == set(TARGETS), 'all historical source fingerprints required')
    path = Path(__file__).with_name('history35_latches_v1.py')
    need(pin(path)['SHA256']==config['latches_SHA256'], 'existing adapted latch source drift')
    spec = importlib.util.spec_from_file_location('_history35_existing_graph_latches', path)
    _capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_capture)
    _capture._config = config
    save = _capture._save_json
    def save_with_parameters(path, value):
        if value.get('state')=='CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE':
            import torch
            parameters = {}
            total = 0
            for name, (tensor, before) in _params.items():
                after = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
                need(before==after, 'parameter source identity changed at export')
                raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                total += len(raw)
                need(total <= 128*1024**2, 'bounded actual parameter CPU hashes')
                parameters[name] = dict(SHA256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                        shape=list(tensor.shape), dtype=str(tensor.dtype))
            value['actual_parameters_after_EOF'] = parameters
            value['all_registered_hit_counters'] = {key:hit.cpu().tolist()
                                                    for key,(_,hit) in _capture._buffers.items()}
            value['input_RID_binding_scope'] = 'controller-exclusive sole RPC + actual GPU positions/token samples; no GPU RID latch'
        save(path, value)
    _capture._save_json = save_with_parameters
    need(not any(name in sys.modules for name in TARGETS), 'no hot installation')
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in TARGETS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            need(spec is not None and spec.loader is not None, 'natural module loader required')
            original = spec.loader.exec_module
            def execute(module):
                original(module)
                patch(fullname, module)
            spec.loader.exec_module = execute
            return spec
    sys.meta_path.insert(0, Finder())
    return _capture
