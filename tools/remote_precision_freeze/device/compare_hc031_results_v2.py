"""CPU-only adapter to the existing complete-ID diagnostic comparator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--sha256', required=True)
a = p.parse_args()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert os.getuid() == 500 and sha(a.plan) == a.sha256
v = json.loads(a.plan.read_text())
assert sha(Path(__file__)) == v['script']['SHA256']
c = Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
root = Path(v['root'])
assert root.parent == c and not root.exists()
root.mkdir(mode=0o700)
def save(name, value):
    q = root/(name+'.partial')
    q.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    q.replace(root/name)
try:
    ep = c/'airline4-cpu-driver-v1/environment.json'
    assert sha(ep) == 'd5075b5c7ef68c5d1c1a4960f970c677a64d64227360c30505883abe2a39a2cc'
    envdoc = json.loads(ep.read_text())
    env = dict(envdoc['environment'])
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    env['PYTHONPATH'] = str(c/'inner-replay-tools-v2')+':'+str(c)+':'+env['PYTHONPATH']
    diag = Path(v['diagnostic']['path'])
    assert sha(diag) == v['diagnostic']['SHA256']
    argv = [envdoc['interpreter'], '-B', '-S', str(diag), '--action', 'compare',
            '--source', v['source']['path'], '--source-sha256', v['source']['SHA256'],
            '--output-dir', str(root/'comparison')]
    identities = {}
    actual_results = {}
    for name, ref in v['results'].items():
        path = Path(ref['path'])
        assert sha(path) == ref['SHA256']
        result = json.loads(path.read_text())
        assert result['response_eof_complete'] is True
        actual_results[name] = result
        identity = result['identity']
        assert identity['source_commit'] == v['candidate_commit']
        identities[name] = identity
        assert result['input_ids_sha256'] == v['input_SHA256']
        argv += ['--'+name, str(path), '--'+name+'-sha256', ref['SHA256']]
    first = next(iter(identities.values()))
    for identity in identities.values():
        for key in ['source_commit', 'native_sha256', 'SDK_packaged_DSO_SHA256',
                    'precision_config_sha256']:
            assert identity[key] == first[key], 'candidate/profile drift: '+key
    with (root/'compare.stdout').open('xb') as out, (root/'compare.stderr').open('xb') as err:
        q = subprocess.run(argv, env=env, stdout=out, stderr=err, timeout=90)
    assert q.returncode == 0, 'existing complete-ID comparator failed'
    summary = json.loads((root/'comparison/summary.json').read_text())
    # Compare existing native source continuation, without another RPC or a KV
    # boundary inference. R must be an exact prefix of the actual G+Y sequence.
    sp = Path(v['source']['path'])
    assert sha(sp) == v['source']['SHA256']
    source = json.loads(sp.read_text())
    assert source['response_eof_complete'] is True
    g, y = source['input_ids'], source['output_ids']
    offset = len(actual_results['baseline']['input_ids'])-len(g)
    assert 0 <= offset < len(y)
    expected_input = g+y[:offset]
    for result in actual_results.values():
        assert result['input_ids'] == expected_input
    expected = y[offset:]
    def difference(left, right):
        for i, (x, y) in enumerate(zip(left, right)):
            if x != y:return dict(index_zero_based=i,token_one_based=i+1)
        return None if len(left)==len(right) else dict(index_zero_based=min(len(left),len(right)),token_one_based=min(len(left),len(right))+1,common_prefix_only=True)
    summary['actual_source_suffix'] = dict(source_output_tokens=len(y),consumed_source_output_tokens=offset,
        remaining_actual_source_tokens=len(expected),
        comparisons={name:dict(complete_IDs_equal=result['output_ids']==expected,
            first_difference=difference(result['output_ids'],expected)) for name,result in actual_results.items()},
        scope='actual source token sequence only; no inferred computed/published KV boundary')
    reference = v['normal_reference']
    rp = Path(reference['path'])
    assert sha(rp) == reference['SHA256']
    previous = json.loads(rp.read_text())
    assert previous['response_eof_complete'] is True and previous['input_ids'] == expected_input
    assert previous['identity']['source_commit'] == v['candidate_commit']
    summary['cross_epoch_normal_complete_IDs_equal'] = previous['output_ids'] == actual_results['baseline']['output_ids']
    summary['cross_epoch_normal_first_difference'] = difference(previous['output_ids'],actual_results['baseline']['output_ids'])
    summary['role_epochs'] = {name: {role: dict(model_root=row['model_root'],
        GPU_ranks=[{k:r[k] for k in ['PID','UID','start_ticks']} for r in row['GPU_ranks']],
        namespace=row['namespace']) for role,row in identity['roles'].items()}
        for name,identity in identities.items()}
    # Restrict exported model-root identity to process identifiers.
    for roles in summary['role_epochs'].values():
        for role in roles.values():
            role['model_root'] = {k:role['model_root'][k] for k in ['PID','UID','start_ticks']}
    summary.update(candidate_commit=v['candidate_commit'], model_requests=0,
                   plan_SHA256=a.sha256, formal_acceptance=False)
    save('result-safe.json', summary)
    save('status.json', dict(state='SUCCEEDED', model_requests=0,
                            result_SHA256=sha(root/'result-safe.json')))
except BaseException as error:
    save('status.json', dict(state='BLOCKED', model_requests=0,
                            error_kind=type(error).__name__, error=str(error)[:1000]))
    raise
