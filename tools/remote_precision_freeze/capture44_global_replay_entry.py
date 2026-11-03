"""Existing finite operator adapter: recipe -> freeze -> check/render -> submit.

Uses the existing controller, private runtime and same-PID empty-GPU bootstrap.
No model launch, request, cache mutation, new scheduler or large-byte transfer.
"""
import argparse
import fcntl
import json
import sys
from pathlib import Path
import freeze_stage as f


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('command', choices=['freeze', 'check', 'render', 'submit'])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        recipe, rp = f.doc(args.recipe)
        need = f.need
        need(recipe['flow'] == 'capture44-global-replay' and recipe['mode'] in ['check', 'run'], 'finite replay recipe')
        actual, ap = f.doc(recipe['analysis_receipt'])
        result = actual['result']
        need(actual['controller']['state'] == actual['status']['state'] == 'SUCCEEDED' and
             result['state'] == 'COMPARED_ACTUAL_HISTORICAL7045_L24_ATTENTION_CHAIN', 'completed actual comparison required')
        files = {key:f.read(path)[1] for key,path in [
            ('entry', Path(__file__)), ('script', f.BASE/'capture44_global_replay.py'),
            ('bootstrap', f.BASE/'gpu_before_runtime_exec.py'),
            ('nvml_helper', Path(recipe['bench'])/'c3_FFN_MegaMHC_counterfactual_v1.py')]}
        audit,kp=f.doc(recipe['global_audit_receipt']);need(audit['controller']['state']==audit['status']['state']=='SUCCEEDED' and audit['result']['rows']['common']==511 and audit['result']['common_rows']['unequal_elements']==0,'actual GLOBAL row audit required')
        files['global_audit_receipt']=kp
        mode = recipe['mode']; tag = 'capture44-global-replay-'+mode+'-v1'
        cases = {}
        for case in ['source', 'normal']:
            role='D' if case=='source' else 'P'
            rows = [x for x in result['coverage'][case][role] if not x['empty']]
            need(len(rows) == 1, 'actual rank coverage')
            row = rows[0]
            cases[case] = dict(path=f.CONTROL+'/capture44-historical7045-comparison-v2/'+case+'-'+role+'-'+str(row['PID'])+'.pt', SHA256=row['SHA256'])
        plan = dict(schema='existing-capture44-global-replay-v1', mode=mode,
                    candidate_commit=result['candidate_commit'], comparison=actual['status']['result'],
                    global_audit=audit['status']['result'],cases=cases, backend=recipe['backend'], environment=recipe['environment'],
                    manifest=recipe['manifest'], output_root=f.CONTROL+'/'+tag,
                    local_pins=dict(files, recipe=rp, analysis_receipt=ap),
                    bench=str(Path(recipe['bench']).resolve()), model_requests=0, primitive_calls_max=3)
        for key in ['script', 'bootstrap', 'nvml_helper']:
            plan[key] = dict(path=f.CONTROL+'/'+Path(files[key]['path']).name, SHA256=files[key]['SHA256'])
        if mode == 'run':
            cpu, pin = f.doc(recipe['CPU_receipt'])
            need(cpu['controller']['state'] == cpu['status']['state'] == 'SUCCEEDED' and
                 cpu['result']['state'] == 'CPU_REPLAY_PINS_AND_ACTUAL_INPUTS_PASS', 'actual device CPU preflight required')
            plan['CPU_result'] = cpu['status']['result']
            plan['local_pins']['CPU_receipt'] = pin
        plan['admission_proof'] = f.CONTROL+'/'+tag+'-admission.json'
        program = plan['bootstrap'] if mode == 'run' else plan['script']
        argv = ['/opt/conda310/bin/python3.10', '-B', '-S', program['path'], '--plan',
                f.CONTROL+'/'+args.plan.name, '--sha256', '${PLAN_SHA256}']
        if mode == 'check':
            argv.append('--check')
        plan['job'] = dict(host=f.HOSTS['D'], name=tag, argv=argv, timeout=240,
                          verify=[dict(path=tag+'/status.json', json_fields=dict(state='SUCCEEDED', model_requests=0))])
        print(json.dumps(dict(state='FROZEN', SHA256=f.save_new(args.plan, plan))))
        return
    plan, pin = f.doc(args.plan)
    f.need(pin['SHA256'] == args.sha256 and plan['schema'] == 'existing-capture44-global-replay-v1', 'plan drift')
    for local in plan['local_pins'].values():
        f.need(f.read(local['path'])[1] == local, 'local frozen input drift')
    job = dict(plan['job'])
    job['argv'] = [args.sha256 if x == '${PLAN_SHA256}' else x for x in job['argv']]
    if args.command == 'check':
        print(json.dumps(dict(state='OFFLINE_PINS_PASS', model_requests=0)))
        return
    if args.command == 'render':
        print(json.dumps(job))
        return
    f.need(args.receipt is not None and not args.receipt.exists(), 'new submit receipt required')
    sys.path.insert(0, plan['bench'])
    from remote_control import remote
    from cluster23_dispatch import submit
    assets = {plan[k]['path']:Path(plan['local_pins'][k]['path']).read_text()
              for k in ['script', 'bootstrap', 'nvml_helper']}
    assets[f.CONTROL+'/'+args.plan.name] = args.plan.read_text()
    code = '''import os,json,hashlib,subprocess
from pathlib import Path
assert os.getuid()==500
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source/plan drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
p=Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
original=json.loads(p.read_text())['environment'];env={k:original[k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']};env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run(%r,env=dict(os.environ,**env),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_CLI='PASS',environment=env)))
''' % (assets, plan['environment']['path'], plan['environment']['SHA256'],
       ['/opt/conda310/bin/python3.10', '-B', '-S', plan['script']['path'], '--help'])
    with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepared = remote('cluster3', job['host'], code, 60)
        launch = submit('cluster3', job['host'], job['name'], job['argv'], job['timeout'], job['verify'],
                        env=prepared['environment'] if plan['mode']=='check' else {'CUDA_VISIBLE_DEVICES':''})
    f.save_new(args.receipt, dict(plan_SHA256=args.sha256, actual_CPU_CLI=prepared['actual_CPU_CLI'], launch=launch))
    print(json.dumps(dict(state='SUBMITTED_NOT_VALIDATED', model_requests=0)))


if __name__ == '__main__':
    main()
