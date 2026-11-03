"""One frozen read-only metadata batch, using original submit twice."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--bench',type=Path,required=True);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  files={k:f.read(path)[1] for k,path in [('entry',Path(__file__)),('D_script',f.BASE/'device/collect_indexer_head_fullwheel_D_metadata_v1.py'),('seed_script',f.BASE/'device/collect_hc031_manager_v1.py'),('collector',a.bench/'collect_pilot_publication.py')]};plan=dict(schema='existing-read-only-native-Manager-batch-v1',root=f.CONTROL+'/headwheel-D-metadata-v1',source=dict(path=f.CONTROL+'/headwheel-source1/request/native/result.json',SHA256='359bb0cecb5df549cfbbe0158a8a7203eb0848c9c60b87c0beab149123193a59'),native_URL='http://10.44.61.71:19111/359bb0cecb5df549cfbbe0158a8a7203eb0848c9c60b87c0',model_requests=0,local_pins=files)
  for k in ['D_script','seed_script','collector']:plan[k]=dict(path=f.CONTROL+'/'+Path(files[k]['path']).name,SHA256=files[k]['SHA256'])
  plan['jobs']={'D':dict(host=f.HOSTS['D'],name='headwheel-native-metadata-v1',argv=['/opt/conda310/bin/python3.10','-B','-S',plan['D_script']['path'],'--root',f.CONTROL+'/headwheel-native-D-v1'],timeout=150,verify=[dict(path='headwheel-native-D-v1/status.json',json_fields=dict(state='FETCHED_ONCE',returncode=0))]),'receiver':dict(host=f.HOSTS['receiver'],name='headwheel-Manager-metadata-v1',argv=['/opt/conda310/bin/python3.10','-B','-S',plan['seed_script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],timeout=150,verify=[dict(path='headwheel-D-metadata-v1/status.json',json_fields=dict(state='NATIVE_AND_MANAGER_METADATA_READY',model_requests=0))])};print(json.dumps(dict(SHA256=f.save_new(a.plan,plan))));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'metadata plan drift')
 for value in plan['local_pins'].values():f.need(f.read(value['path'])[1]==value,'source drift')
 jobs={role:dict(job,argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in job['argv']]) for role,job in plan['jobs'].items()}
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_METADATA_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(jobs));return
 sys.path.insert(0,str(a.bench));from remote_control import remote
 from cluster23_dispatch import submit
 job=jobs['receiver'];assets={plan['seed_script']['path']:Path(plan['local_pins']['seed_script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()};code='''import json,os\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw\n else:\n  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nprint(json.dumps(dict(published=True)))'''%assets;remote('cluster3',job['host'],code,60);launches={}
 for role in ['D','receiver']:
  job=jobs[role];launches[role]=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],assets={Path(plan['D_script']['path']).name:Path(plan['local_pins']['D_script']['path']).read_text()} if role=='D' else None)
 f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launches=launches));print(json.dumps(dict(state='METADATA_SUBMITTED_NOT_VERIFIED',model_requests=0)))
if __name__=='__main__':main()
