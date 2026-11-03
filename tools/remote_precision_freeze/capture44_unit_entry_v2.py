"""Freeze the new graph-hook tests using the existing candidate-unit controller."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command', choices=['freeze','check','render','submit'])
    p.add_argument('--reference-plan', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--sha256')
    p.add_argument('--receipt', type=Path)
    a = p.parse_args()
    if a.command == 'freeze':
        previous, pin = f.doc(a.reference_plan)
        f.need(previous['schema']=='existing-candidate-module-unit-plan-v1' and
               previous['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c', 'actual candidate unit reference required')
        files = {key:f.read(path)[1] for key,path in [
            ('entry', Path(__file__)), ('script', f.BASE/'capture44_unit.py'),
            ('test', f.BASE/'test_capture44_graph_v2.py'),
            ('tracer', f.BASE/'device/history44_observer_v1.py'),
            ('latches', f.BASE/'device/history38_latches_v1.py'),
            ('chain', f.BASE/'capture44_chain_contract_v2.py'),
            ('config', Path(a.config)),
            ('bootstrap', f.BASE/'gpu_before_runtime_exec.py'),
            ('nvml_helper', Path(previous['local_pins']['nvml_helper']['path']))]}
        plan = {k:previous[k] for k in ['candidate_commit','environment','manifest','delayed_SHA256','bench']}
        plan['manifest']=dict(path=f.CONTROL+'/hc-tf32-031-inner-v1/manifest.json',SHA256='179428dbcd30ca64ce914ad0dad5f1b6d2bf5aa43682ffca8c5d87680f7c0115')
        tag = 'capture44-full-chain-gpu-v2'
        plan.update(schema='existing-history35-unit-v1', local_pins=dict(files, reference=pin),
                    output_root=f.CONTROL+'/'+tag, admission_proof=f.CONTROL+'/'+tag+'-admission.json', model_requests=0)
        for key in ['script','test','tracer','latches','bootstrap','nvml_helper','chain','config']:
            plan[key] = dict(path=f.CONTROL+'/'+Path(files[key]['path']).name, SHA256=files[key]['SHA256'])
        plan['job'] = dict(host=f.HOSTS['D'], name=tag, timeout=240,
             argv=['/opt/conda310/bin/python3.10','-B','-S',plan['bootstrap']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],
             verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan))))
        return
    plan, pin = f.doc(a.plan)
    f.need(pin['SHA256']==a.sha256 and plan['schema']=='existing-history35-unit-v1','frozen unit plan drift')
    for local in plan['local_pins'].values():
        f.need(f.read(local['path'])[1]==local,'immutable unit input drift')
    job = dict(plan['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in plan['job']['argv']])
    if a.command == 'check':
        print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)))
        return
    if a.command == 'render':
        print(json.dumps(job))
        return
    f.need(a.receipt is not None and not a.receipt.exists(),'new receipt required')
    sys.path.insert(0,plan['bench'])
    from remote_control import remote
    from cluster23_dispatch import submit
    assets={plan[k]['path']:Path(plan['local_pins'][k]['path']).read_text()
            for k in ['script','test','tracer','latches','bootstrap','nvml_helper','chain','config']}
    assets[f.CONTROL+'/'+a.plan.name] = a.plan.read_text()
    code='''import json,os,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable unit source/plan drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
r=subprocess.run(%r,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_prepare=json.loads(r.stdout))))
'''%(assets,['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256',a.sha256,'--check'])
    with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=remote('cluster3',job['host'],code,150)
        launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''})
    f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch))
    print(json.dumps(dict(state='SUBMITTED_NOT_VALIDATED',model_requests=0)))


if __name__ == '__main__':
    main()
