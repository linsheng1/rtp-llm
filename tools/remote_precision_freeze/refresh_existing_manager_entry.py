import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--bench',type=Path,required=True);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  script=f.read(f.BASE/'refresh_existing_manager.py')[1];collector=f.read(a.bench/'collect_pilot_publication.py')[1];root=f.CONTROL+'/capture34-source-locations-after-Consumer2';plan=dict(schema='existing-source-read-only-Manager-refresh-v1',root=root,original=dict(path=f.CONTROL+'/state91b-D-metadata-v1/manager-report.json',SHA256='a61387fce2f902f6cc34edaefd8bfcd2dc55085a447c03a3cc8a56349ac3659c'),collector=dict(path=f.CONTROL+'/collect_pilot_publication.py',SHA256=collector['SHA256']),script=dict(path=f.CONTROL+'/refresh_existing_manager.py',SHA256=script['SHA256']),local_pins=dict(script=script,collector=collector,entry=f.read(__file__)[1]),job=dict(host=f.HOSTS['receiver'],name='capture34-source-locations-readonly',timeout=90,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/refresh_existing_manager.py','--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=Path(root).name+'/result.json',json_fields=dict(state='EXISTING_SOURCE_MANAGER_LOCATIONS_READABLE',model_requests=0))]));print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'read-only Manager plan drift')
 for x in plan['local_pins'].values():f.need(f.read(x['path'])[1]==x,'source pin drift')
 sys.path.insert(0,str(a.bench));from remote_control import remote
 from cluster23_dispatch import submit
 j=plan['job'];argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in j['argv']];assets={plan['script']['path']:Path(plan['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()};code='''import json,os,subprocess\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw\n else:\n  with p.open('x') as f:f.write(raw);f.flush();os.fsync(f.fileno())\nr=subprocess.run(%r+['--check'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr\nprint(json.dumps(dict(actual_CPU_CLI='PASS',model_requests=0)))'''%(assets,argv);cpu=remote('cluster3',j['host'],code,60);out=submit('cluster3',j['host'],j['name'],argv,j['timeout'],j['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=out));print(json.dumps(dict(state='READONLY_MANAGER_QUERY_SUBMITTED',model_requests=0)))
if __name__=='__main__':main()
