"""Thin HC adapter: recipe -> freeze/check/render -> original submit/controller.

The GPU job uses the unchanged reference-drain wrapper and pre-runtime bootstrap.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command',choices=['freeze','check','render','submit'])
    p.add_argument('--recipe',type=Path)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256')
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        recipe,rp=f.doc(a.recipe)
        actual,ap=f.doc(recipe['comparison_actual'])
        report=actual['history36-seam-comparison-v1/comparison-safe.json']
        f.need(report['value']['state']=='HISTORY36_EXISTING_PRODUCER_CAPTURE_COMPARED' and
               report['value']['candidate_commit']=='91b4635faf8c65cc67e66ab52a46f5c932cd08ea', 'actual accepted capture required')
        files={k:f.read(path)[1] for k,path in [('entry',Path(__file__)),('script',f.BASE/'history36_hc_replay.py'),
             ('bootstrap',f.BASE/'gpu_before_runtime_exec.py'),('nvml_helper',Path(recipe['bench'])/'c3_FFN_MegaMHC_counterfactual_v1.py'),
             ('drain_wrapper',Path(recipe['bench'])/'c3_after_reference_drain_v1.py')]}
        flags,fp=f.doc(recipe['flags_actual'])
        f.need(all(v['matches_reviewed_local'] for v in flags['sources'].values()),'actual production source comparison required')
        mode=recipe['mode'];f.need(mode in ['check','run'],'finite CPU/GPU mode required')
        tag='history36-hc-'+mode+'-v1'
        plan=dict(schema='existing-history36-HC-replay-v1',mode=mode,candidate_commit=report['value']['candidate_commit'],
             output_root=f.CONTROL+'/'+tag,analysis=dict(path=f.CONTROL+'/history36-seam-comparison-v1/comparison-safe.json',SHA256=report['SHA256']),
             analysis_plan=recipe['analysis_plan'],environment=recipe['environment'],manifest=recipe['manifest'],
             flags_receipt=dict(path=f.CONTROL+'/'+Path(fp['path']).name,SHA256=fp['SHA256']),
             production_sources={k:{x:v[x] for x in ['path','SHA256']} for k,v in flags['sources'].items()},
             local_pins=dict(files,recipe=rp,comparison=ap,flags=fp),model_requests=0,primitive_calls_max=7,
             bench=recipe['bench'],reference_model_root=f.CONTROL+'/airline4-P-history36')
        for k in ['script','bootstrap','nvml_helper','drain_wrapper']:
            plan[k]=dict(path=f.CONTROL+'/'+Path(files[k]['path']).name,SHA256=files[k]['SHA256'])
        if mode=='run':
            cpu,cp=f.doc(recipe['CPU_receipt'])
            f.need(cpu['controller']['state']==cpu['status']['state']=='SUCCEEDED' and cpu['result']['state']=='HISTORY36_HC_ACTUAL_INPUT_CPU_PASS','actual CPU input receipt required')
            plan['CPU_result']=cpu['status']['result'];plan['local_pins']['CPU_receipt']=cp
        plan['admission_proof']=f.CONTROL+'/'+tag+'-admission.json'
        argv=['/opt/conda310/bin/python3.10','-B','-S',plan['bootstrap' if mode=='run' else 'script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}']
        if mode=='run':
            argv=['/opt/conda310/bin/python3.10','-B','-S',plan['drain_wrapper']['path'],'--previous-root',plan['reference_model_root'],
                  '--state-root',f.CONTROL+'/'+tag+'-after-drain','--']+argv
        plan['job']=dict(host=f.HOSTS['P'],name=tag,argv=argv,timeout=540 if mode=='run' else 180,
             verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)))
        return
    plan,pin=f.doc(a.plan)
    f.need(pin['SHA256']==a.sha256 and plan['schema']=='existing-history36-HC-replay-v1','immutable replay plan differs')
    for local in plan['local_pins'].values():f.need(f.read(local['path'])[1]==local,'frozen source/evidence drift')
    job=dict(plan['job'],argv=[a.sha256 if v=='${PLAN_SHA256}' else v for v in plan['job']['argv']])
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    f.need(a.receipt is not None and not a.receipt.exists(),'new receipt required')
    sys.path.insert(0,plan['bench'])
    from remote_control import remote
    from cluster23_dispatch import submit
    assets={plan[k]['path']:Path(plan['local_pins'][k]['path']).read_text() for k in ['script','bootstrap','nvml_helper','drain_wrapper']}
    assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text()
    assets[plan['flags_receipt']['path']]=Path(plan['local_pins']['flags']['path']).read_text()
    code='''import os,json,hashlib,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable replay source/plan drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
p=Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
original=json.loads(p.read_text())['environment'];env={k:original[k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']};env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run(%r,env=dict(os.environ,**env),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_CLI='PASS',environment=env)))
'''%(assets,plan['environment']['path'],plan['environment']['SHA256'],['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--help'])
    with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=remote('cluster3',job['host'],code,60)
        launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env=cpu['environment'] if plan['mode']=='check' else {'CUDA_VISIBLE_DEVICES':''})
    f.save_new(a.receipt,dict(plan_SHA256=a.sha256,actual_CPU_CLI=cpu['actual_CPU_CLI'],launch=launch))
    print(json.dumps(dict(state='SUBMITTED_NOT_VALIDATED',model_requests=0)))


if __name__=='__main__':main()
