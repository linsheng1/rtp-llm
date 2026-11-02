"""Frozen source-batch three argv -> original submit/controller; no new scheduler."""
import argparse,json,subprocess,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--submit',action='store_true');a=p.parse_args();plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256 and plan.get('flow')=='state-source','frozen source plan required');f.validate(plan)
 def command(*args):
  r=subprocess.run([sys.executable,'-B',str(f.BASE/'freeze_stage.py'),*args],capture_output=True,text=True,timeout=40);f.need(r.returncode==0,r.stdout+r.stderr);return json.loads(r.stdout)
 command('check','--plan',str(a.plan),'--sha256',a.sha256,'--cli-help');jobs={role:command('render','--plan',str(a.plan),'--sha256',a.sha256,'--role',role) for role in ['receiver','P','D']};a.evidence_dir.mkdir(parents=True,exist_ok=True);f.save_new(a.evidence_dir/'rendered.json',dict(plan_SHA256=a.sha256,jobs=jobs,model_requests=0));recipe,_=f.doc(plan['local_pins']['recipe']['path']);sys.path.insert(0,recipe['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={f.CONTROL+'/'+Path(plan['local_pins'][k]['path']).name:Path(plan['local_pins'][k]['path']).read_text() for k in recipe['helpers']};preflight={}
 for role,job in jobs.items():
  for name,raw in assets.items():
   code="""import json,os
from pathlib import Path
p=Path(%r);raw=%r
if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source drift'
else:
 with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
print(json.dumps(dict(source_verified=True)))"""%(name,raw)
   remote('cluster3',job['host'],code,60)
  code="""import json,os,subprocess
r=subprocess.run(%r+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CLI='PASS',model_requests=0)))"""%job['argv'][:4]
  preflight[role]=remote('cluster3',job['host'],code,60)
 # Real CPU driver/dependency closure and exact private input BEFORE any HTTP/RPC.
 code='''import json,os,subprocess,hashlib\nfrom pathlib import Path\nc=Path(%r);ep=c/'airline4-cpu-driver-v1/environment.json';assert hashlib.sha256(ep.read_bytes()).hexdigest()=='d5075b5c7ef68c5d1c1a4960f970c677a64d64227360c30505883abe2a39a2cc';edoc=json.loads(ep.read_text());env=dict(edoc['environment']);env['PYTHONPATH']=str(c/'inner-replay-tools-v2')+':'+str(c)+':'+env['PYTHONPATH'];assert env['CUDA_VISIBLE_DEVICES']==''\ncase=c/'airline4-native-case-v1/case-private.json';assert hashlib.sha256(case.read_bytes()).hexdigest()=='0e2f29a837287d6201c27e005c4f2f70ce40d41885235b3ce8075738a6c13753'\nr=subprocess.run([edoc['interpreter'],'-B','-S',str(c/'c3_history35_baseline_client_v2.py'),'--help'],env=env,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr\nprint(json.dumps(dict(actual_client_dependency_CLI='PASS',exact_private_input='PASS',model_requests=0)))'''%f.CONTROL
 preflight['client']=remote('cluster3',f.HOSTS['receiver'],code,60);f.save_new(a.evidence_dir/'device-CPU-prepare.json',dict(plan_SHA256=a.sha256,preflight=preflight,actual_RPC=0,complete_RPC_admission='executed by receiver after actual paired native probes'))
 if not a.submit:return
 for role in ['receiver','P','D']:
  job=jobs[role];root=Path(f.option(job['argv'],'--root' if role=='receiver' else '--output-root')).name;target='NORMAL_COMPLETE' if role=='receiver' else 'HISTORY35_BASELINE_CAPTURE_EXPORTED_AFTER_EOF';r=submit('cluster3',job['host'],'history35-'+plan['tag']+'-'+role,job['argv'],960 if role=='receiver' else 660,[dict(path=root+'/status.json',json_fields=dict(state=target))]);f.save_new(a.evidence_dir/(role+'-submit.json'),dict(plan_SHA256=a.sha256,launch=r))
 print(json.dumps(dict(state='THREE_FROZEN_JOBS_SUBMITTED_NOT_VALIDATED',model_requests='UNVERIFIED')))
if __name__=='__main__':main()
