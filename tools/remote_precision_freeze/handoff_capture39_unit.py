"""Freeze/check/render -> original submit for a necessary owned diagnostic boundary."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--unit-plan',type=Path);p.add_argument('--old-model-plan',type=Path);p.add_argument('--source-proof',type=Path);p.add_argument('--normal-proof',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  unit,up=f.doc(a.unit_plan);old,op=f.doc(a.old_model_plan);f.need(unit['candidate_commit']==old['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and old['role']=='P' and unit['model_requests']==0,'current owned P/unit only')
  files=dict(entry=f.read(__file__)[1],handoff=f.read(f.BASE/'device/handoff_capture39_unit_v1.py')[1],unit=up,old=op,wrapper=f.read(Path(old['bench'])/'c3_after_reference_drain_v1.py')[1]);proofs=[]
  sys.path.insert(0,old['bench']);from remote_control import remote
  for key,path,tag,count in [('source',a.source_proof,'D1',167),('normal',a.normal_proof,'N1',176)]:
   d,files[key]=f.doc(path);s=d['status.json'];f.need(d['controller']['state']=='SUCCEEDED' and s['state']=='NORMAL_COMPLETE' and d['native_result']['EOF'] and d['native_result']['output_tokens']==count and s['source_calibration']['full_output_IDs_equal'] and len(s['capture_barriers']['EXPORT'])==2,'actual completed4rank capture required')
   remote_path=f.CONTROL+'/capture38-'+tag+'/status.json'
   code='import json,hashlib\nfrom pathlib import Path\np=Path(%r);raw=p.read_bytes();d=json.loads(raw);assert d==%r;print(json.dumps(dict(SHA256=hashlib.sha256(raw).hexdigest())))'%(remote_path,s)
   pin=remote('cluster3',f.HOSTS['receiver'],code,60)
   proofs.append(dict(URL='http://10.44.61.73:19091/publication-precision-20261101/control/capture38-'+tag+'/status.json',SHA256=pin['SHA256'],fields=dict(state='NORMAL_COMPLETE',model_requests=1,capture_phase='DONE',source_calibration=s['source_calibration'])))
  name='capture39-unit-after-capture38';v=dict(scope='owned P capture38 EOF to capture39 diagnostic-unit boundary',candidate_commit=unit['candidate_commit'],bench=old['bench'],previous_root=old['model_root'],previous_plan_SHA256=old['model_plan_SHA256'],completed_stages=proofs,unit_plan=dict(path=f.CONTROL+'/'+a.unit_plan.name,SHA256=up['SHA256']),unit_argv=[up['SHA256'] if x=='${PLAN_SHA256}' else x for x in unit['job']['argv']],handoff_claim=f.CONTROL+'/'+name+'-claim.json',drain_root=f.CONTROL+'/'+name+'-drain',local_pins=files)
  for key in ['handoff','wrapper']:v[key]=dict(path=f.CONTROL+'/'+Path(files[key]['path']).name,SHA256=files[key]['SHA256'])
  v['job']=dict(host=f.HOSTS['P'],name=name,timeout=600,argv=['/opt/conda310/bin/python3.10','-B','-S',v['handoff']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=unit['job']['verify'])
  print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'handoff plan drift')
 for x in v['local_pins'].values():f.need(f.read(x['path'])[1]==x,'handoff source drift')
 unit,_=f.doc(v['local_pins']['unit']['path'])
 for x in unit['local_pins'].values():f.need(f.read(x['path'])[1]==x,'actual unit input drift')
 j=dict(v['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in v['job']['argv']])
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(j));return
 sys.path.insert(0,v['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={unit[k]['path']:Path(unit['local_pins'][k]['path']).read_text() for k in ['script','test','tracer','latches','bootstrap','nvml_helper']};assets[v['unit_plan']['path']]=Path(v['local_pins']['unit']['path']).read_text();assets[v['handoff']['path']]=Path(v['local_pins']['handoff']['path']).read_text();assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text()
 code='''import os,json,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable CPU unit input drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
r=subprocess.run(%r,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_unit_prepare=json.loads(r.stdout),model_requests=0)))
'''%(assets,['/opt/conda310/bin/python3.10','-B','-S',unit['script']['path'],'--plan',v['unit_plan']['path'],'--sha256',v['unit_plan']['SHA256'],'--check'])
 cpu=remote('cluster3',j['host'],code,150);launch=submit('cluster3',j['host'],j['name'],j['argv'],j['timeout'],j['verify'],env={'CUDA_VISIBLE_DEVICES':''});f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch));print(json.dumps(dict(state='OWN_SAFE_BOUNDARY_AND_UNIT_SUBMITTED_NOT_COMPLETE',model_requests=0)))
if __name__=='__main__':main()
