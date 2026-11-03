"""Actual wrapper/record/latch/socket/export chain; synthetic operands only.

CPU mode virtualizes only CUDA presence and synchronization. GPU mode uses real
CUDA tensors/graphs. Neither mode is evidence about model numerical outputs.
"""
import argparse, hashlib, importlib.util, json, os, socket, sys
from pathlib import Path
from types import SimpleNamespace as NS


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_exact(actual, expected):
    """Typed byte comparison shared with the actual sampling comparator."""
    import torch
    need(actual.dtype == expected.dtype and actual.shape == expected.shape, 'typed geometry drift')
    a, b = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    equal = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return dict(byte_equal=equal, elements=a.numel(), unequal_elements=int((a != b).sum()))


def run(mode, config, observer, output_root):
    import torch
    need(mode in ['cpu', 'gpu'] and not output_root.exists(), 'fresh contract output')
    output_root.mkdir(mode=0o700)
    old_capture, old_sync = torch.cuda.is_current_stream_capturing, torch.cuda.synchronize
    old_full = torch.full
    if mode == 'cpu':
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU isolation')
        class CPUCudaBoundaryTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True
        def tensor(value):
            return value.as_subclass(CPUCudaBoundaryTensor)
        torch.cuda.is_current_stream_capturing = lambda: False
        torch.cuda.synchronize = lambda device=None: need(device is None or device.type == 'cpu', 'only CPU boundary shim')
        torch.full = lambda *args, **kw: tensor(old_full(*args, **kw))
        device = 'cpu'
    else:
        tensor = lambda value: value
        device = 'cuda'
    try:
        results = []
        # These production metadata helpers have no model/device initialization.
        site = Path(config['site'])
        for name in ['rtp_llm.models_py.modules.dsv41.attn_type',
                     'rtp_llm.models_py.modules.dsv41.kv_cache_utils',
                     'rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils']:
            path = site / (name.replace('.', '/') + '.py')
            sys.modules[name] = load(path, name)
        for role in ['P', 'D']:
            m = load(observer, 'h44_contract_' + role)
            cfg = dict(config['capture'], role=role, output=str(output_root / role))
            cap = m.install(cfg)
            # Test the actual wrapper functions directly, without importing any
            # production math module or substituting record/latch/export.
            sys.meta_path.pop(0)
            def control(operation, label=''):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(20)
                    client.connect(str(Path(cfg['output']) / (str(os.getpid()) + '.sock')))
                    client.sendall((json.dumps(dict(operation=operation, label=label, request_drained=True)) + '\n').encode())
                    raw = b''
                    while not raw.endswith(b'\n'):
                        part = client.recv(65536)
                        need(bool(part), 'complete control reply')
                        raw += part
                return json.loads(raw)
            def make(shape, dtype, fill=0):
                value = torch.full(shape, fill, dtype=torch.float32 if dtype == torch.float8_e4m3fn else dtype, device=device)
                return tensor(value.to(dtype))
            pos = make((1,), torch.int32)
            cp = NS(cp_size=4, kv_cache_sharded=True, input_lengths_global_host=(1,),
                    global_positions=pos, local_is_real=make((1,), torch.bool, True))
            table = make((1, 20), torch.int32)
            table.copy_(tensor(torch.arange(1, 21, device=device, dtype=torch.int32).reshape(1, 20)))
            meta = NS(position_ids=pos, valid_requests=make((1,), torch.bool, True),
                      req_id_per_token=make((1,), torch.int32), pool_block_tables={'swa_kv': table})
            cache = NS(get_seq_size_per_block=lambda tag: 512, get_kernel_seq_size_per_block=lambda tag: 512)
            attn = NS(_cp_ctx=cp, tp_size=1, n_heads=64, compress_ratio=1,
                      kv_source_layer_id=20, index_source_layer_id=24,
                      is_kv_source=False, is_index_source=True, _kv_cache=cache,
                      _global_region=lambda: 'csa_kv', _block_tables_by_type={'csa_kv': table})
            tiny = make((2,), torch.float32, 1)
            mega = NS(query_weight=tiny, query_scale=tiny, output_weight=tiny,
                      output_scale=tiny, sink=make((64,), torch.float32, 1),
                      cos_sin=make((16384, 64), torch.float32, 1), softmax_scale=0.125)
            attn._mega_attention = mega
            block = NS(layer_id=24, attn=attn)
            x = make((1, 4, 5120) if role == 'P' else (1, 1, 4, 5120), torch.bfloat16, 3)
            ids = make((1,) if role == 'P' else (1, 1), torch.int32, 123)
            qr = make((1, 64), torch.bfloat16, 2)
            quant = (make((1, 64), torch.float8_e4m3fn, 1), make((1, 2), torch.uint8, 127))
            q = make((1, 64, 512), torch.bfloat16, 7)
            project = m.mega_wrapper(lambda self, x, qr, qr_quantized=None: q, 'project_q')
            output = m.mega_wrapper(lambda self, out: make((1, 5120), torch.bfloat16, 9), 'project_output')
            if role == 'P':
                kv = make((640, 1, 512), torch.bfloat16, 5)
                indices = tensor(torch.arange(640, device=device, dtype=torch.int32).reshape(1, 640))
                core = m.mega_wrapper(lambda self, *a, **kw: (q + 2, make((1, 16), torch.float32, 1)), 'prefill')
            else:
                kv = make((32, 128, 528), torch.uint8, 17)
                global_pool = make((32, 512, 288), torch.uint8, 19)
                indices = make((1, 128), torch.int32, -1)
                extra = make((1, 512), torch.int32, -1)
                core = m.mega_wrapper(lambda self, *a, **kw: (q + 2, make((1, 16), torch.float32, 1)), 'decode')
            def body(block, *args):
                m.record('actual_attention_input5120', x.reshape(1, 4, 5120)[:, 0])
                projected = project(mega, qr, qr, qr_quantized=quant)
                if role == 'P':
                    m.context()['prefill_chunk_meta'] = (0, 512, 6918)
                    result = core(mega, projected, kv, indices, pos)
                else:
                    # Query-dependent values use graph operations, just as the
                    # actual block/attention wrappers see on replay.
                    absolute = pos.reshape(1, 1) - 127 + torch.arange(128, device=device).reshape(1, 128)
                    page = table.gather(1, (absolute // 512).clamp(0, 19).long())
                    indices.copy_((page * 128 + absolute % 128).int())
                    extra.fill_(-1)
                    result = core(mega, projected, kv, indices, pos, extra_pool=global_pool, extra_indices=extra)
                answer = output(mega, result[0])
                m.record('attention_output5120', answer)
                return answer
            wrapped = m.block_wrapper(body, role)
            def call():
                return wrapped(block, x, ids) if role == 'P' else wrapped(block, x, meta, ids)
            with torch.inference_mode():
                answer = call()
                need(m.context() is None, 'wrapper context restored')
                if mode == 'gpu':
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        answer = call()
                    execute = graph.replay
                else:
                    execute = call
                need(control('reset')['state'] == 'CAPTURE_LATCHES_RESET', 'real inference reset')
                execute()
                empty = control('snapshot', 'nohit')
                need(empty['state'] == 'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE' and empty['coverage_empty'], 'no target is explicit empty')
                need(control('reset')['state'] == 'CAPTURE_LATCHES_RESET', 'reset before selected query')
                pos.fill_(7045)
                execute()
                exported = control('snapshot', 'selected')
                need(exported['state'] == 'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE', str(exported))
                path = Path(exported['file'])
                need(hashlib.sha256(path.read_bytes()).hexdigest() == exported['SHA256'], 'actual serialized SHA')
                data = torch.load(path, map_location='cpu', weights_only=True)
                prefix = role + '-L24-T7045-'
                required = {'block_entry4HC', 'actual_input_token', 'actual_topology', 'actual_attention_input5120',
                            'Qr_before_Q_projection', 'Qr_actual_quant_0', 'Qr_actual_quant_1', 'Q_after_projection',
                            'Q_before_core', 'core_positions', 'sink', 'cos_sin_position_row', 'softmax_scale',
                            'core_branch', 'core_output', 'core_scales', 'output_projection5120', 'attention_output5120'}
                need({prefix + name for name in required} <= set(data), 'complete wrapper/record/export contract')
                need(set(data) == set(exported['hits']) and all(v == [1, 7045] for v in exported['hits'].values()), 'one hit per actual tensor')
                need(exported['actual_parameters_after_EOF'] and len(exported['all_registered_hit_counters']) <= 192, 'real parameter/counter export')
                checks = [compare_exact(data[prefix + 'block_entry4HC'], torch.full((1, 20480), 3, dtype=torch.bfloat16)),
                          compare_exact(data[prefix + 'core_output'], torch.full((1, 32768), 9, dtype=torch.bfloat16)),
                          compare_exact(data[prefix + 'actual_topology'], torch.tensor([[1, 64, 1, 20, 24, 0, 1]], dtype=torch.int32)),
                          compare_exact(data[prefix + 'core_positions'], torch.tensor([[7045]], dtype=torch.int32))]
                need(all(v['byte_equal'] for v in checks), 'actual export comparator')
                if role == 'P':
                    actual = torch.cat([data[prefix + 'prefill_selected_KV_chunk' + str(i)].reshape(128, 512) for i in range(5)])
                    need(compare_exact(actual, torch.full((640, 512), 5, dtype=torch.bfloat16))['byte_equal'], 'all 5 actual chunks reconstructed')
                else:
                    need(data[prefix + 'SWA_mapping_valid'].all() and data[prefix + 'SWA_window_mapping_valid'].all(), 'actual decode mapping')
                budget = sum(v.numel() * v.element_size() + 8 for v, _ in cap._buffers.values())
                need(budget <= 4194304 and len(cap._buffers) <= 192, 'unchanged latch budgets')
                duplicate = control('snapshot', 'selected')
                need(duplicate['state'] == 'CAPTURE_CONTROL_FAILED', 'immutable export cannot overwrite')
                results.append(dict(role=role, points=len(data), latch_bytes=budget, actual_export_SHA256=exported['SHA256'], typed_comparisons=checks))
        need(mode != 'cpu' or not torch.cuda.is_initialized(), 'CPU contract never initialized CUDA')
        return dict(state='ACTUAL_RECORD_EXPORT_CHAIN_PASS', mode=mode, roles=results, synthetic_only=True,
                    GPU_graph_tested=mode == 'gpu', model_requests=0, whole_model_fixed=False,
                    boundary_shims=['Tensor.is_cuda', 'cuda.is_current_stream_capturing', 'cuda.synchronize', 'torch.full CPU device marker'] if mode == 'cpu' else [])
    finally:
        torch.cuda.is_current_stream_capturing, torch.cuda.synchronize = old_capture, old_sync
        torch.full = old_full


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode', choices=['cpu', 'gpu'], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--observer', type=Path, required=True)
    p.add_argument('--observer-sha256', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    need(hashlib.sha256(a.config.read_bytes()).hexdigest() == a.config_sha256, 'config identity')
    need(hashlib.sha256(a.observer.read_bytes()).hexdigest() == a.observer_sha256, 'observer identity')
    result = run(a.mode, json.loads(a.config.read_text()), a.observer, a.output_root)
    pending = a.output_root / 'result-safe.partial'
    pending.write_text(json.dumps(result, indent=2))
    pending.replace(a.output_root / 'result-safe.json')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
