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
    p.add_argument('--manifest-receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        actual,ap=f.doc(a.source)
        f.need(actual['controller']['state']==actual['status']['state']=='SUCCEEDED','completed actual comparison')
        comparison,cp=f.doc(a.normal)
        f.need(comparison['schema']=='capture44-existing-historical-CPU-v1','actual comparison plan')
        result=actual['status']['result']
        files=dict(entry=f.read(__file__)[1],script=f.read(f.BASE/'capture44_global_audit.py')[1],actual=ap,comparison=cp)
        source=f.literals(f.BASE/'device/c3_hc031_capture44_core_v1.py')
        # Candidate codec/mapping fingerprints are pinned below from the
        # immutable runtime manifest, before any CPU tensor analysis.
        metadata,mp=f.doc(a.manifest_receipt)
        files['manifest_receipt']=mp
        plan=dict(schema='capture44-existing-historical-CPU-v1',candidate_commit=comparison['candidate_commit'],
                  comparison_plan=dict(path=f.CONTROL+'/'+a.normal.name,SHA256=cp['SHA256']),comparison_result=result,
                  environment=comparison['environment'],codecs=metadata['codecs'],
                  script=dict(path=f.CONTROL+'/capture44_global_audit.py',SHA256=files['script']['SHA256']),
                  output_root=f.CONTROL+'/capture44-global-readset-audit-v1',local_pins=files,model_requests=0)
        plan['job']=dict(host=f.HOSTS['D'],name='capture44-global-readset-audit-v1',timeout=180,
             argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],
             verify=[dict(path='capture44-global-readset-audit-v1/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)))
        return
    plan,pin=f.doc(a.plan)
    f.need(pin['SHA256']==a.sha256 and plan['schema']=='capture44-existing-historical-CPU-v1','immutable analysis plan drift')
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
