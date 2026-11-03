"""Freeze/check/render -> original controller; completed own model boundary only."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--model',type=Path);p.add_argument('--source',type=Path);p.add_argument('--normal',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  model,mp=f.doc(a.model);bench=model['bench'];sys.path.insert(0,bench);from remote_control import remote
  role=model['role'];files=dict(entry=f.read(__file__)[1],script=f.read(f.BASE/'drain_completed_capture41.py')[1],wrapper=f.read(Path(bench)/'c3_after_reference_drain_v1.py')[1],model=mp);proofs=[]
  for case,path,root,state,count in [('source',a.source,'capture41-source-source1','D_SOURCE_COMPLETE',295),('normal',a.normal,'capture41-N1','NORMAL_COMPLETE',176)]:
   doc,files[case]=f.doc(path);s=doc['status.json'];f.need(doc['controller']['state']=='SUCCEEDED' and s['state']==state and doc['native_result']['EOF'] and doc['native_result']['output_tokens']==count and s['source_calibration']['full_output_IDs_equal'],'actual completed two-request boundary')
   code='import json,hashlib\nfrom pathlib import Path\np=Path(%r);raw=p.read_bytes();assert json.loads(raw)==%r;print(json.dumps(dict(SHA256=hashlib.sha256(raw).hexdigest())))'%(f.CONTROL+'/'+root+'/status.json',s);pin=remote('cluster3',f.HOSTS['receiver'],code,60);ack=s['capture_barriers']['EXPORT'][role];url=lambda path:'http://10.44.61.73:19091/publication-precision-20261101/control/'+str(Path(path).relative_to(f.CONTROL));proofs.append(dict(URL=url(f.CONTROL+'/'+root+'/status.json'),SHA256=pin['SHA256'],state=state,output_tokens=count,role_ack=dict(URL=url(ack['path']),SHA256=ack['SHA256'])))
  tag='capture41-completed-'+role+'-drain-v1';v=dict(schema='completed-capture41-own-drain-v1',role=role,bench=bench,model_root=model['model_root'],model_plan_SHA256=model['model_plan_SHA256'],completed_stages=proofs,output_root=f.CONTROL+'/'+tag,local_pins=files)
  for k in ['script','wrapper']:v[k]=dict(path=f.CONTROL+'/'+Path(files[k]['path']).name,SHA256=files[k]['SHA256'])
  v['job']=dict(host=f.HOSTS[role],name=tag,timeout=300,argv=['/opt/conda310/bin/python3.10','-B','-S',v['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0,original_owned_model_drain_complete=True))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
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
