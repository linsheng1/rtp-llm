"""Bind completed request proofs to the existing owned-P drain helper."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 p.add_argument('command',choices=['freeze','check','render','submit'])
 for flag in ['plan','model-plan','completed-proofs','dispatch-output','receipt']:p.add_argument('--'+flag,type=Path)
 p.add_argument('--sha256');p.add_argument('--name');a=p.parse_args()
 if a.command=='freeze':
  model,mp=f.doc(a.model_plan);proof,pp=f.doc(a.completed_proofs)
  f.need(model['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and model['role']=='P','only owned HC031 P transition')
  for value in model['local_pins'].values():f.need(f.read(value['path'])[1]==value,'target model source drift')
  reference,rp=f.doc(model['local_pins']['reference']['path']);dispatch,dp=f.doc(a.model_plan.parent/'P-dispatch.json')
  files=dict(entry=f.read(__file__)[1],handoff=f.read(f.BASE/'handoff_state_epoch.py')[1],wrapper=model['local_pins']['wrapper'],model=mp,proofs=pp,reference=rp,target_dispatch=dp)
  name=a.name;f.need(__import__('re').fullmatch('[A-Za-z0-9_-]+',name),'owned transition name required')
  v=dict(candidate_commit=model['candidate_commit'],bench=model['bench'],previous_root=f.option(reference['argv'],'--root'),previous_plan_SHA256=reference['plan_SHA256'],completed_stages=proof['completed_stages'],model_root=model['model_root'],model_plan_SHA256=model['model_plan_SHA256'],model_argv=dispatch['argv'],handoff_claim=f.CONTROL+'/'+name+'-claim.json',drain_root=f.CONTROL+'/'+name+'-drain',local_pins=files)
  for key in ['handoff','wrapper']:v[key]=dict(path=f.CONTROL+'/'+Path(files[key]['path']).name,SHA256=files[key]['SHA256'])
  v['job']=dict(host=f.HOSTS['P'],name=name,timeout=2340,argv=['/opt/conda310/bin/python3.10','-B','-S',v['handoff']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=model['job']['verify'])
  sha=f.save_new(a.plan,v);dispatch['parent_controller_status']=f.CONTROL+'/'+name+'-status.json';dispatch['actual_handoff_plan_SHA256']=sha;f.save_new(a.dispatch_output,dispatch);print(json.dumps(dict(SHA256=sha,model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'immutable handoff plan')
 for x in v['local_pins'].values():f.need(f.read(x['path'])[1]==x,'handoff source drift')
 job=dict(v['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in v['job']['argv']])
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(job));return
 sys.path.insert(0,v['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={v['handoff']['path']:Path(v['local_pins']['handoff']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
 code='''import os,json
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable handoff source'
 else:
  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
print(json.dumps(dict(source_verified=True)))
'''%assets
 remote('cluster3',job['host'],code,60)
 launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=launch));print(json.dumps(dict(state='HANDOFF_SUBMITTED_NOT_READY',model_requests=0)))

if __name__=='__main__':main()
