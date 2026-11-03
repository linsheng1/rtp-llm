"""Frozen installation recipe -> original C3 device controller; zero model RPC."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--recipe',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  r,rp=f.doc(a.recipe);pins=dict(recipe=rp,entry=f.read(__file__)[1],script=f.read(f.BASE/'device/install_indexer_head_wheel_v2.py')[1]);f.need(r['candidate_commit']=='fb09988a5c552f4f5ff716472d8b50647af22d2c' and r['wheel']['SHA256']=='6a77459e273388b3e6143b9b2c45d16e083cc6c4d658d382438e5ed5abfb4645','actual frozen complete wheel required');f.need(r['role'] in ['P','D'],'C3 compute role');r.update(local_pins=pins,host=f.HOSTS[r['role']],script=dict(path=f.CONTROL+'/install_indexer_head_wheel_v2.py',SHA256=pins['script']['SHA256']))
  r['job']=dict(host=r['host'],name=r['tag'],timeout=900,argv=['/opt/conda310/bin/python3.10','-B','-S',r['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=Path(r['root']).name+'/status-v2.json',json_fields=dict(state='SUCCEEDED',model_requests=0))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,r),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'plan drift')
 for pin in v['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'source drift')
 job=dict(v['job'],argv=[a.sha256 if s=='${PLAN_SHA256}' else s for s in v['job']['argv']])
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(job));return
 f.need(a.receipt is not None and not a.receipt.exists(),'unique receipt');sys.path.insert(0,v['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={v['script']['path']:Path(v['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
 code="import json,os\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable drift'\n else:p.open('x').write(raw)\nprint(json.dumps(dict(published=True)))"%assets
 remote('cluster3',job['host'],code,60)
 cpu=remote('cluster3',job['host'],"import json,subprocess,os\nr=subprocess.run(%r+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr;print(json.dumps(dict(actual_CLI='PASS',model_requests=0)))"%job['argv'][:4],60)
 launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''});f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch));print(json.dumps(dict(state='SUBMITTED_NOT_COMPLETE',model_requests=0)))
if __name__=='__main__':main()
