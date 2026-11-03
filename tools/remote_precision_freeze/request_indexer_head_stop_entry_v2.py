"""Freeze/check/render -> original controller; completed own model boundary only."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--recipe',type=Path);p.add_argument('--sha256');p.add_argument('--model',type=Path);p.add_argument('--source',type=Path);p.add_argument('--normal',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  recipe,recipe_pin=f.doc(a.recipe);model,mp=f.doc(Path(recipe['model_plan']));bench=model['bench'];role=model['role'];f.need(role=='P','only own fresh-P boundary');sys.path.insert(0,bench);from remote_control import remote
  files=dict(recipe=recipe_pin,entry=f.read(__file__)[1],script=f.read(f.BASE/'request_completed_head_stop_v2.py')[1],wrapper=f.read(Path(bench)/'c3_after_reference_drain_v1.py')[1],model=mp);proofs=[]
  for case,path in recipe['completed_receipts'].items():
   doc,files[case]=f.doc(path);native=doc['native_safe'];f.need(doc['controller']['state']=='SUCCEEDED' and native['EOF'],'completed actual source/normal EOF required');url='http://10.44.61.73:19091/publication-precision-20261101/control/'+str(Path(native['path']).relative_to(f.CONTROL));proofs.append(dict(URL=url,SHA256=native['SHA256'],output_tokens=native['output_tokens']))
  tag=recipe['tag'];f.need(__import__('re').fullmatch('[A-Za-z0-9_-]+',tag),'new own boundary tag');v=dict(schema='completed-head-candidate-own-drain-v1',role=role,bench=bench,model_root=model['model_root'],model_plan_SHA256=model['model_plan_SHA256'],completed_stages=proofs,parent_controller_status=f.CONTROL+'/'+model['job']['name']+'-status.json',output_root=f.CONTROL+'/'+tag,local_pins=files)
  for k in ['script','wrapper']:v[k]=dict(path=f.CONTROL+'/'+Path(files[k]['path']).name,SHA256=files[k]['SHA256'])
  v['job']=dict(host=f.HOSTS[role],name=tag,timeout=90,argv=['/opt/conda310/bin/python3.10','-B','-S',v['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0,stage_kind='OWNED_STOP_REQUEST_ONLY'))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'own drain plan drift')
 for pin in v['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'frozen own drain input drift')
 job=dict(v['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in v['job']['argv']])
 if a.command=='render':print(json.dumps(job));return
 sys.path.insert(0,v['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={v['script']['path']:Path(v['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
 code='''import json,os,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable own drain source/plan drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
r=subprocess.run(%r+['--check'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_boundary_preflight=json.loads(r.stdout),model_requests=0)))'''%(assets,job['argv']);cpu=remote('cluster3',job['host'],code,90)
 if a.command=='check':print(json.dumps(cpu));return
 f.need(a.receipt is not None and not a.receipt.exists(),'new own drain receipt');launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''});f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch));print(json.dumps(dict(state='OWN_DRAIN_SUBMITTED_NOT_COMPLETE',model_requests=0)))
if __name__=='__main__':main()
