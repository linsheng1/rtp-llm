"""Three bounded core calls over existing capture34, without model requests.

Calibrate both original query outputs first. Then change only the 17 unequal
selected KV rows in the short workspace. Other queries/unused rows are synthetic;
this tests sufficiency at the consumer, not producer or transport provenance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

C = Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(pin, limit=8 * 1024**2):
    path = Path(pin['path'])
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
         'bounded immutable file required: ' + str(path))
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns) and sha(raw) == pin['SHA256'],
         'immutable input drift: ' + str(path))
    return raw


def module(pin, name):
    read(pin)
    spec = importlib.util.spec_from_file_location(name, pin['path'])
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inspect(plan):
    analysis = json.loads(read(plan['analysis']))
    previous_plan = json.loads(read(plan['analysis_plan']))
    request = json.loads(read(dict(path=str(Path(plan['analysis']['path']).parent /
                                    'actual-request-report.json'),
                                   SHA256=previous_plan['report']['SHA256'])))
    need(analysis['capture_plan_SHA256'] == plan['capture_plan_SHA256'] and
         analysis['candidate_commit'] == plan['candidate_commit'] and
         analysis['input_SHA256'] == plan['input_SHA256'] and
         request['model_requests_completed'] == 2, 'accepted capture identity differs')
    att = analysis['attention']
    need(att['canonical_logical_sequence_equal'] and att['common_logical_rows'] == 640
         and att['owner_head_parameters_equal'] and att['selected_KV_unequal_rows'] == 17,
         'actual selected-KV evidence differs')
    observer = module(previous_plan['helpers']['observer'], 'existing_capture34_observer')
    rows = {}
    originals = {}
    for case in ['short', 'full']:
        pin = request['barriers'][case]['proof']
        # The seed proof was already copied directly to this P device by analysis.
        local_pin = dict(path=str(Path(plan['analysis']['path']).parent / (case + '-barrier.json')),
                         SHA256=pin['SHA256'])
        barrier = json.loads(read(local_pin))
        entries = [dict(path=r['manifest']['path'], sha256=r['manifest']['SHA256'])
                   for r in barrier['ranks']]
        verified = observer.verify_transition_case(entries)
        need(verified == json.loads(read(barrier['transition_capture']['report'])) and
             verified['coverage_complete'] and
             verified['request_id'] == analysis['actual_request_ids'][case],
             'four-rank coverage/RID differs')
        row = next(r for r in verified['attention']['ranks'] if r['actual_query_owner'])
        originals[case] = row
        def member(label, query=True):
            raw = read(row['members'][label]['raw'])
            return raw[len(raw)//2:] if query else raw
        index_bytes = member('L2.attn.sparse_indices')
        indices = list(struct.unpack('<' + 'i' * (len(index_bytes)//4), index_bytes))
        valid = member('L2.attn.selected_valid', False)
        values = member('L2.attn.selected_KV', False)
        need(len(indices) == len(valid) == 640 and all(valid) and
             len(set(indices)) == 640 and len(values) == 640*1024,
             'actual640 valid unique original slots required')
        rows[case] = dict(indices=indices, kv=values, q=member('L2.attn.Q_before_core'),
                          output=member('L2.attn.core_output'), scales=member('L2.attn.core_scales'),
                          shape=row['input_metadata']['KV_shape'])
    need(rows['short']['q'] == rows['full']['q'], 'actual Q differs')
    bad = [i for i in range(640) if rows['short']['kv'][i*1024:(i+1)*1024] !=
           rows['full']['kv'][i*1024:(i+1)*1024]]
    need(len(bad) == 17, 'actual unequal KV count differs')
    env = json.loads(read(plan['environment']))
    manifest = json.loads(read(plan['manifest']))
    need(env['source_commit'] == manifest['source_commit'] == plan['candidate_commit'],
         'candidate/runtime drift')
    for key in ['PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH', 'CUDA_HOME']:
        need(os.environ.get(key) == env['environment'][key], 'private runtime env differs: ' + key)
    import torch
    need(not torch.cuda.is_initialized() and torch.__version__.startswith('2.11') and
         Path(torch.__file__).resolve().is_relative_to(Path(env['Torch_partial_site']).resolve()),
         'CPU Torch origin/no CUDA init')
    audit = module(plan['nvml_helper'], 'existing_bounded_checkpoint_audit')
    params, checkpoint = audit.checkpoint_tensors(['layers.2.attn.attn_sink'])
    sink = params['layers.2.attn.attn_sink']
    expected = originals['short']['parameters']
    need(sink['dtype'] == 'F32' and sha(sink['raw']) == expected['mega.sink']['SHA256'],
         'checkpoint sink differs from actual core parameter')
    config_raw = Path('/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash/config.json').read_bytes()
    config = json.loads(config_raw)
    config = {**config, **config.get('text_config', {})}
    scaling = config.get('rope_scaling') or {}
    from rtp_llm.models_py.modules.dsv41 import rope
    need(sha(Path(rope.__file__).read_bytes()) == plan['rope_source']['SHA256'],
         'actual imported RoPE source differs')
    table = rope.precompute_freqs_cis(64, expected['mega.cos_sin']['shape'][0],
                 scaling.get('original_max_position_embeddings', 0),
                 config.get('compress_rope_theta', 160000), scaling.get('factor', 1.0),
                 scaling.get('beta_fast', 32), scaling.get('beta_slow', 1))
    cos_sin = torch.cat((table.real, table.imag), -1).contiguous()
    need(sha(cos_sin.view(torch.uint8).numpy().tobytes()) == expected['mega.cos_sin']['SHA256'],
         'reconstructed complete RoPE table differs from actual parameter')
    mega_pin = originals['short']['sources']['prefill']
    need(mega_pin == originals['full']['sources']['prefill'], 'paired core Python differs')
    mega = module(mega_pin, 'actual_capture34_mega')
    backend = mega._backend()
    need(backend is not None and not torch.cuda.is_initialized(), 'actual CPU backend import/init')
    backend_path = Path(backend.__file__).resolve()
    backend_sha = hashlib.sha256(backend_path.read_bytes()).hexdigest()
    cpu = dict(state='CAPTURE34_KV_REPLAY_CPU_PASS', candidate_commit=plan['candidate_commit'],
               capture_plan_SHA256=plan['capture_plan_SHA256'], analysis=plan['analysis'],
               input_SHA256=plan['input_SHA256'], model_requests=0, primitive_calls=0,
               unequal_KV_rows=len(bad), exact_Q_equal=True,
               sink_SHA256=sha(sink['raw']), RoPE_SHA256=expected['mega.cos_sin']['SHA256'],
               checkpoint=checkpoint, model_config_SHA256=sha(config_raw), core_source=mega_pin,
               backend=dict(path=str(backend_path), SHA256=backend_sha),
               GPU_budget_bytes=1024**3, max_core_calls=3,
               synthetic_other_queries_and_unused_KV=True,
               source_remote_destination_byte_equality_proven=False)
    return cpu, rows, bad, sink, cos_sin, mega, backend, audit


def run(plan, items, proof):
    cpu, rows, bad, sink, cos_sin, mega, backend, audit = items
    need(cpu == json.loads(read(plan['CPU_result'])), 'CPU prepared inputs/parameters drift')
    import torch
    initial = audit.nvml()
    need(set(initial['compute_apps']) <= {os.getpid()} and initial['used_v2'] < 1024**3,
         'unchanged exclusive1GiB initialization guard')
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.synchronize()
    initial = audit.nvml()
    remaining = 1024**3 - initial['used_v2']
    need(remaining > 32*1024**2, 'original1GiB memory reserve')
    torch.cuda.set_per_process_memory_fraction(remaining/initial['total_v2'], 0)
    torch.cuda.reset_peak_memory_stats()
    def tensor(raw, dtype, shape):
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).cuda()
    context = SimpleNamespace(backend=backend, pad_q=lambda q: q,
               softmax_scale=0.04419417382415922,
               sink=tensor(sink['raw'], torch.float32, (64,)), cos_sin=cos_sin.cuda(),
               padded_groups=8)
    q = tensor(rows['short']['q'], torch.bfloat16, (1,64,512)).expand(2,-1,-1).contiguous()
    positions = torch.full((2,), 7168, dtype=torch.int32, device='cuda')
    lengths = torch.full((2,), 640, dtype=torch.int32, device='cuda')
    results = {}
    gpu = []
    short_kv = short_indices = None
    with torch.inference_mode():
        for case in ['short', 'full', 'short_KV17_replaced']:
            source = rows['full' if case == 'full' else 'short']
            if case != 'short_KV17_replaced':
                kv = torch.zeros(source['shape'], dtype=torch.bfloat16, device='cuda')
                slots = torch.tensor(source['indices'], dtype=torch.long, device='cuda')
                kv.index_copy_(0, slots, tensor(source['kv'], torch.bfloat16, (640,1,512)))
                indices = slots.to(torch.int32).view(1,640).expand(2,-1).contiguous()
                if case == 'short':
                    short_kv, short_indices = kv, indices
            else:
                kv, indices = short_kv, short_indices
                slots = torch.tensor([rows['short']['indices'][i] for i in bad],
                                     dtype=torch.long, device='cuda')
                values = b''.join(rows['full']['kv'][i*1024:(i+1)*1024] for i in bad)
                kv.index_copy_(0, slots, tensor(values, torch.bfloat16, (17,1,512)))
            output, scales = mega.MegaAttention.prefill(context, q, kv, indices, positions, lengths)
            torch.cuda.synchronize()
            original_bits = [value[1].contiguous().cpu().view(torch.uint8).numpy().tobytes()
                             for value in [output, scales]]
            results[case] = dict(output_SHA256=sha(original_bits[0]), scales_SHA256=sha(original_bits[1]),
                  equals_short=(original_bits == [rows['short']['output'], rows['short']['scales']]),
                  equals_full=(original_bits == [rows['full']['output'], rows['full']['scales']]))
            p = Path(plan['output_root']) / 'core-calls.partial'
            p.write_text(json.dumps(dict(primitive_calls=len(results), results=results), indent=2))
            p.replace(p.with_name('core-calls.json'))
            v = audit.nvml()
            need(set(v['compute_apps']) <= {os.getpid()} and v['used_v2'] <= 1024**3 and
                 torch.cuda.max_memory_allocated() <= remaining, 'original exclusive1GiB guard')
            gpu.append(dict(stage=case, used_v2=v['used_v2'], torch_peak=torch.cuda.max_memory_allocated()))
            del output, scales, slots
            if case == 'full':
                del kv, indices
            if case == 'full':
                need(results['short']['equals_short'] and results['full']['equals_full'],
                     'original output/scales not exactly reproduced; block causal attribution')
    return dict(cpu, state='CAPTURE34_KV_REPLAY_COMPLETE', primitive_calls=3,
                results=results, original_both_core_outputs_exact_reproduced=True,
                replacing_only17_KV_rows_explains_full_core_delta=results['short_KV17_replaced']['equals_full'],
                GPU_samples=gpu, admission=proof, operator_root_cause_proven=False,
                producer_inputs_equal_claimed=False, whole_model_fixed=False, formal_acceptance=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    plan = json.loads(read(dict(path=str(args.plan), SHA256=args.sha256)))
    need(os.getuid() == 500 and sha(Path(__file__).read_bytes()) == plan['script']['SHA256'], 'source/UID drift')
    if args.check:
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation')
    root = Path(plan['output_root'])
    need(root.parent == C and not root.exists(), 'new finite result root required')
    root.mkdir(mode=0o700)
    def save(name, value):
        p = root / (name+'.partial')
        p.write_text(json.dumps(value, sort_keys=True, indent=2))
        p.replace(root/name)
    try:
        proof = None
        if not args.check:
            proof = json.loads(read(dict(path=plan['admission_proof'], SHA256=os.environ['STATE_ADMISSION_SHA256'])))
            fields = Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
            need(proof['PID'] == os.getpid() and proof['UID'] == 500 and proof['start_ticks'] == fields[19]
                 and proof['plan_SHA256'] == args.sha256 and -2 <= time.time()-proof['at'] <= 45,
                 'same-PID admission proof differs')
        items = inspect(plan)
        result = items[0] if args.check else run(plan, items, proof)
        save('result-safe.json', result)
        save('status.json', dict(state='SUCCEEDED', model_requests=0,
             result=dict(path=str(root/'result-safe.json'), SHA256=sha((root/'result-safe.json').read_bytes()))))
        print(json.dumps(dict(state=result['state'], model_requests=0, primitive_calls=result['primitive_calls'])))
    except BaseException as error:
        save('status.json', dict(state='FAILED', error_kind=type(error).__name__, error=str(error), model_requests=0))
        raise


if __name__ == '__main__':
    main()
