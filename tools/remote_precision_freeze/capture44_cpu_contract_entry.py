"""Freeze actual new CPU gather contracts, then use the existing controller."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  files=dict(entry=f.read(__file__)[1],script=f.read(f.BASE/'capture44_cpu_contract.py')[1],observer=f.read(f.BASE/'device/history44_observer_v1.py')[1]);tag='capture44-gather-cpu-contract-v1';v=dict(schema='capture44-CPU-contract-v1',local_pins=files,environment=dict(path=f.CONTROL+'/hc-tf32-031-inner-v1/environment.json',SHA256='1bb3d2ba9082c9ce17c2990ff9a8448855d68c88d793d89b8d25538d8355f6b2'),result_path=f.CONTROL+'/'+tag+'-result-safe.json',model_requests=0)
  for k in ['script','observer']:v[k]=dict(path=f.CONTROL+'/'+Path(files[k]['path']).name,SHA256=files[k]['SHA256'])
  v['job']=dict(host=f.HOSTS['D'],name=tag,timeout=120,argv=['/opt/conda310/bin/python3.10','-B','-S',v['script']['path'],'--observer',v['observer']['path'],'--observer-sha256',v['observer']['SHA256'],'--self-sha256',v['script']['SHA256'],'--output',v['result_path']],verify=[dict(path=tag+'-result-safe.json',json_fields=dict(state='CPU_GATHER_CANONICAL_BOUNDS_AND_NO_HIT_PASS',contracts=6,model_requests=0,CUDA_initialized=False,synthetic_only=True))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'immutable CPU contract plan')
 for x in v['local_pins'].values():f.need(f.read(x['path'])[1]==x,'CPU contract source drift')
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(v['job']));return
 bench=f.BASE.parents[3]/'benchmarks/publication_precision_20261101';sys.path.insert(0,str(bench));from remote_control import remote
 from cluster23_dispatch import submit
 assets={v[k]['path']:Path(v['local_pins'][k]['path']).read_text() for k in ['script','observer']};assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text();job=v['job'];code='''import json,hashlib,os,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable CPU source/plan drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;e=json.loads(raw)['environment'];env={k:e[k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']};env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run(%r+['--help'],env=dict(os.environ,**env),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_CLI='PASS',env=env)))'''%(assets,v['environment']['path'],v['environment']['SHA256'],job['argv'][:4]);cpu=remote('cluster3',job['host'],code,60);f.need(a.receipt is not None and not a.receipt.exists(),'new CPU submit receipt');launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env=cpu['env']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch));print(json.dumps(dict(state='CPU_CONTRACT_SUBMITTED_NOT_PASSED',model_requests=0)))
if __name__=='__main__':main()
