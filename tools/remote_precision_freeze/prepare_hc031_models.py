"""031 candidate, original model CLI/controller; real CPU prepare before submit.

Retains lease, namespace, UID, source/SDK, GPU-occupancy and original drain gates.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 p.add_argument('command',choices=['prepare','check','render','submit'])
 p.add_argument('--recipe',type=Path);p.add_argument('--directory',type=Path)
 p.add_argument('--plan',type=Path);p.add_argument('--sha256');p.add_argument('--receipt',type=Path)
 a=p.parse_args()
 if a.command!='prepare':
  plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256 and plan['schema']=='existing-hc031-model-epoch-v1','immutable candidate model plan')
  for v in plan['local_pins'].values():f.need(f.read(v['path'])[1]==v,'frozen source/evidence drift')
  if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
  if a.command=='render':print(json.dumps(plan['job']));return
  f.need(a.receipt is not None and not a.receipt.exists(),'new submit receipt required')
  sys.path.insert(0,plan['bench']);from cluster23_dispatch import submit
  job=plan['job']
  with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'])
  f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=launch));print(json.dumps(dict(state='SUBMITTED_NOT_READY',model_requests=0)));return
 recipe,rp=f.doc(a.recipe);candidate='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c'
 f.need(recipe['candidate_commit']==candidate and recipe['seconds']==1800 and recipe['namespace']=='pubprec_hc031_Dorigin_v1','exact source pilot protocol')
 unit,up=f.doc(recipe['unit_actual']);f.need(unit['controller']['state']==unit['status']['state']=='SUCCEEDED' and unit['result']['candidate_commit']==candidate and unit['result']['passed'] and unit['result']['tests_run']==3 and unit['result']['skipped']==0,'actual committed3test regression gate')
 files={k:f.read(path)[1] for k,path in [('entry',Path(__file__)),('helper',f.BASE/'device/c3_hc031_model_service_v1.py'),('keeper',f.BASE/'revalidate_hc031_static_weights.py'),('wrapper',Path(recipe['bench'])/'c3_after_reference_drain_v1.py')]}
 sys.path.insert(0,recipe['bench']);from remote_control import remote
 a.directory.mkdir(parents=True,exist_ok=True)
 for role in ['P','D']:
  ref,refp=f.doc(recipe['roles'][role]['reference_dispatch']);overlay,op=f.doc(recipe['roles'][role]['overlay_actual']);prepared=overlay['hc-tf32-031-inner-v1/prepare-result.json']['value']
  f.need(prepared['candidate_commit']==candidate and prepared['host']==f.HOSTS[role],'actual per-role candidate overlay')
  argv=list(ref['argv']);f.need(argv[3]==f.CONTROL+'/c3_airline4_model_service_v9.py','original source CLI profile')
  if '--mode' in argv:i=argv.index('--mode');del argv[i:i+2]
  argv[3]=f.CONTROL+'/c3_hc031_model_service_v1.py';root=f.CONTROL+'/airline4-'+role+'-hcfix-source1'
  changes={'--root':root,'--seconds':'1800','--namespace':recipe['namespace'],'--env-json':prepared['environment']['path'],'--env-sha256':prepared['environment']['SHA256'],
   '--manifest':prepared['manifest']['path'],'--manifest-sha256':prepared['manifest']['SHA256'],'--consumer-root':f.CONTROL+'/consumer91b-'+role+'-v3',
   '--keeper-root':f.CONTROL+'/hc031-static-'+role+'-v1','--keeper-helper':f.CONTROL+'/revalidate_hc031_static_weights.py','--keeper-helper-sha256':files['keeper']['SHA256']}
  for k,v in changes.items():f.option(argv,k);argv[argv.index(k)+1]=v
  f.need(f.option(argv,'--instance-group')=='c3_on_B_20261030' and f.option(argv,'--publication')=='1','real group/publication preserved')
  host=f.HOSTS[role]
  assets={f.CONTROL+'/'+Path(v['path']).name:Path(v['path']).read_text() for k,v in files.items() if k in ['helper','keeper','wrapper']}
  for path,raw in assets.items():
   code="import os,json\nfrom pathlib import Path\np=Path(%r);raw=%r\nif p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source drift'\nelse:\n with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())\nprint(json.dumps(dict(source_verified=True)))"%(path,raw)
   remote('cluster3',host,code,60)
  previous=f.CONTROL+'/airline4-'+role+'-history36'
  code='''import json,os,subprocess
from pathlib import Path
old=Path(%r);f=json.loads((old/'model-service-final.json').read_text());s=json.loads((old/'model-status.json').read_text())
assert f['state']=='MODEL_SERVICE_EXITED' and f['actual_model_root_returncode']==0 and s['cleanup_state']=='FINISHED' and not s['remaining_owned_processes'],'previous model not completely drained'
for r in [f['model_root']]+f['GPU_ranks']:
 p=Path('/proc')/str(r['PID'])
 if p.exists():
  v=p.joinpath('stat').read_text().rsplit(')',1)[1].split();assert v[19]!=str(r['start_ticks']) or v[0] in ['Z','X','x'],'old model epoch remains'
checks=[]
for mode in ['compose','check']:
 r=subprocess.run(%r+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180)
 checks.append(dict(mode=mode,returncode=r.returncode,stdout=r.stdout[-800:],stderr=r.stderr[-2000:]))
 assert r.returncode==0,checks
p=Path(%r)/'plan.json';print(json.dumps(dict(state='ACTUAL_HC031_CPU_PREPARE_PASS',checks=checks,model_requests=0,plan_SHA256=__import__('hashlib').sha256(p.read_bytes()).hexdigest())))
'''%(previous,argv,root)
  cpu=remote('cluster3',host,code,420);f.save_new(a.directory/(role+'-CPU.json'),cpu)
  job=dict(host=host,name='hc031-'+role+'-source1-model',timeout=2340,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/c3_after_reference_drain_v1.py','--previous-root',previous,'--state-root',f.CONTROL+'/hc031-'+role+'-source1-after-drain','--']+argv+['--mode','run'],verify=[dict(path=Path(root).name+'/model-service-final.json',json_fields=dict(state='MODEL_SERVICE_EXITED'))])
  plan=dict(schema='existing-hc031-model-epoch-v1',candidate_commit=candidate,role=role,model_root=root,model_plan_SHA256=cpu['plan_SHA256'],CPU=cpu,job=job,bench=recipe['bench'],model_requests=0,
    lifetime_accounting=dict(original_reference_wait=240,service_seconds=1800,original_drain_reserve=420,admission_cleanup_reserve=300,outer_seconds=2340),local_pins=dict(files,recipe=rp,unit=up,reference=refp,overlay=op))
  sha=f.save_new(a.directory/(role+'-model-plan.json'),plan)
  f.save_new(a.directory/(role+'-dispatch.json'),dict(role=role,host=host,root=Path(root).name,plan_SHA256=cpu['plan_SHA256'],argv=argv+['--mode','run'],parent_controller_status=f.CONTROL+'/'+job['name']+'-status.json',epoch_plan_SHA256=sha))
  print(json.dumps(dict(role=role,CPU='PASS',plan_SHA256=sha,model_requests=0)),flush=True)


if __name__=='__main__':main()
