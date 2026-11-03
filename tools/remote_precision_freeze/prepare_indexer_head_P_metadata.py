"""Freeze actual P seed and cursor, then original metadata collector/controller."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--seed-receipt',type=Path);p.add_argument('--bench',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  s,sp=f.doc(a.seed_receipt);seed=s['native_safe'];f.need(seed['EOF'] and seed['output_tokens']==1 and seed['input_ids_SHA256']=='10932132ad309cbfd30ae5600e64d7f7bb2dd8d7d2ddb6af4c6349ce2381ef71','actual one-token frozen P seed EOF')
  sys.path.insert(0,str(a.bench));from remote_control import remote
  cursor=f.CONTROL+'/dgdet-stage-p-seed-headfbS1/manager-cursor.json'
  proof=remote('cluster3',f.HOSTS['receiver'],"import json,hashlib\nfrom pathlib import Path\np=Path(%r);assert p.is_file() and not p.is_symlink();print(json.dumps(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())))"%cursor,60)
  pins={key:f.read(path)[1] for key,path in [('entry',Path(__file__)),('script',f.BASE/'device/collect_indexer_head_P_manager.py'),('collector',a.bench/'collect_pilot_publication.py'),('P_collector',a.bench/'collect_c3_P_origin_manager_v2.py')]};pins['seed_receipt']=sp
  v=dict(candidate_commit='fb09988a5c552f4f5ff716472d8b50647af22d2c',seed={k:seed[k] for k in ['path','SHA256']},cursor=proof,root=f.CONTROL+'/headfb09988-P-manager-v1',bench=str(a.bench),local_pins=pins,model_requests=0)
  for k in ['script','collector','P_collector']:v[k]=dict(path=f.CONTROL+'/'+Path(pins[k]['path']).name,SHA256=pins[k]['SHA256'])
  v['job']=dict(host=f.HOSTS['receiver'],name='headfb09988-P-manager-v1',timeout=150,argv=['/opt/conda310/bin/python3.10','-B','-S',v['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path='headfb09988-P-manager-v1/status.json',json_fields=dict(state='P_MANAGER_METADATA_READY',model_requests=0))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'metadata plan drift')
 for value in v['local_pins'].values():f.need(f.read(value['path'])[1]==value,'metadata source drift')
 job=dict(v['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in v['job']['argv']])
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(job));return
 sys.path.insert(0,v['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={v[k]['path']:Path(v['local_pins'][k]['path']).read_text() for k in ['script','P_collector']};assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text()
 code='''import json,os
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable P metadata source'
 else:
  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
print(json.dumps(dict(source_verified=True)))
'''%assets
 remote('cluster3',job['host'],code,60);launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=launch));print(json.dumps(dict(state='SUBMITTED_NOT_VERIFIED',model_requests=0)))

if __name__=='__main__':main()
