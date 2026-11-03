"""Frozen read-only CPU postprocess through the original device controller."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command',choices=['freeze','check','render','submit'])
    p.add_argument('--source',type=Path)
    p.add_argument('--normal',type=Path)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256')
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        recipe,rp=f.doc(a.source)
        actual,ap=f.doc(recipe['replay_receipt'])
        f.need(actual['controller']['state']==actual['status']['state']=='SUCCEEDED' and actual['result']['both_actual_outputs_and_scales_reproduced'],'actual calibrated SWA replay required')
        base,bp=f.doc(recipe['replay_plan'])
        files=dict(entry=f.read(__file__)[1],script=f.read(f.BASE/'capture40_swa_position.py')[1],codec=f.read(recipe['codec'])[1])
        tag='capture40-swa-position-v1'
        plan=dict(schema='capture40-SWA-position-CPU-v1',candidate_commit=base['candidate_commit'],
                  replay_plan=dict(path=f.CONTROL+'/'+Path(recipe['replay_plan']).name,SHA256=bp['SHA256']),
                  replay_result=actual['status']['result'],environment=base['environment'],
                  codec_SHA256=files['codec']['SHA256'],
                  script=dict(path=f.CONTROL+'/capture40_swa_position.py',SHA256=files['script']['SHA256']),
                  output_root=f.CONTROL+'/'+tag,local_pins=dict(files,recipe=rp,replay_plan=bp,replay_receipt=ap),model_requests=0)
        plan['job']=dict(host=f.HOSTS['D'],name=tag,timeout=180,
             argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],
             verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)))
        return
    plan,pin=f.doc(a.plan)
    f.need(pin['SHA256']==a.sha256 and plan['schema']=='capture40-SWA-position-CPU-v1','immutable analysis plan drift')
    for local in plan['local_pins'].values():
        f.need(f.read(local['path'])[1]==local,'local analysis source/evidence drift')
    job=dict(plan['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in plan['job']['argv']])
    if a.command=='check':
        print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)))
        return
    if a.command=='render':
        print(json.dumps(job))
        return
    f.need(a.receipt is not None and not a.receipt.exists(),'new submit receipt required')
    bench=f.BASE.parents[3]/'benchmarks/publication_precision_20261101'
    sys.path.insert(0,str(bench))
    from remote_control import remote
    from cluster23_dispatch import submit
    assets={plan['script']['path']:Path(plan['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
    code='''import json,os,hashlib,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable CPU analysis source/plan differs'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;doc=json.loads(raw)
env={k:doc['environment'][k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']};env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run(%r,env=dict(os.environ,**env),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_CLI='PASS',environment=env)))
'''%(assets,plan['environment']['path'],plan['environment']['SHA256'],job['argv'][:4]+['--help'])
    with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=remote('cluster3',job['host'],code,60)
        launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env=cpu['environment'])
    f.save_new(a.receipt,dict(plan_SHA256=a.sha256,actual_CPU_CLI=cpu['actual_CPU_CLI'],launch=launch))
    print(json.dumps(dict(state='SUBMITTED_NOT_COMPARED',model_requests=0)))


if __name__=='__main__':
    main()
