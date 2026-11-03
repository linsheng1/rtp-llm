"""Freeze/read-only-check/render -> original controller, zero model RPC recovery."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--failed-capture',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 bench=f.BASE.parents[3]/'benchmarks/publication_precision_20261101';sys.path.insert(0,str(bench));from remote_control import remote
 from cluster23_dispatch import submit
 if a.command=='freeze':
  failed,fp=f.doc(a.failed_capture);f.need(failed['controller']['state']=='FAILED' and failed['native_result']['EOF'] and failed['native_result']['output_tokens']==167,'actual EOF/export failure required')
  c=f.CONTROL;meta=remote('cluster3',f.HOSTS['receiver'],"import json,hashlib\nfrom pathlib import Path\nc=Path(%r);p=c/'capture40-D1/D.json';cal=json.loads((c/'capture40-D1/normal-calibration-safe.json').read_text());assert cal['full_output_IDs_equal'] and cal['actual_profile_equal'] and cal['complete_EOF'];print(json.dumps(dict(probe=dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()),calibration=cal)))"%c,60)
  script=f.read(f.BASE/'device/recover_capture40_data_v1.py')[1];tag='capture40-D1-recovery-v1';v=dict(schema='capture40-typed-recovery-only-v1',candidate_commit='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c',native={k:failed['native_result'][k] for k in ['path','SHA256']},probe=meta['probe'],calibration=meta['calibration'],script=dict(path=c+'/recover_capture40_data_v1.py',SHA256=script['SHA256']),environment=dict(path=c+'/hc-tf32-031-inner-v1/environment.json',SHA256='1bb3d2ba9082c9ce17c2990ff9a8448855d68c88d793d89b8d25538d8355f6b2'),output_root=c+'/'+tag,local_pins=dict(entry=f.read(__file__)[1],script=script,failed_capture=fp),model_requests=0,parameter_hashes_verified=False,original_failed_stage_not_overridden=True)
  v['job']=dict(host=f.HOSTS['D'],name=tag,timeout=300,argv=['/opt/conda310/bin/python3.10','-B','-S',v['script']['path'],'--plan',c+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED_TYPED_RECOVERY_ONLY',model_requests=0))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'recovery plan drift')
 for x in v['local_pins'].values():f.need(f.read(x['path'])[1]==x,'recovery input drift')
 j=dict(v['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in v['job']['argv']])
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_RECOVERY_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(j));return
 assets={v['script']['path']:Path(v['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
 code='''import json,os,hashlib,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable recovery control drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;e=json.loads(raw)['environment'];env={k:e[k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']};env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run(%r,env=dict(os.environ,**env),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_CLI='PASS',environment=env,model_requests=0)))
'''%(assets,v['environment']['path'],v['environment']['SHA256'],j['argv'][:4]+['--help'])
 cpu=remote('cluster3',j['host'],code,60);launch=submit('cluster3',j['host'],j['name'],j['argv'],j['timeout'],j['verify'],env=cpu['environment']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu['actual_CPU_CLI'],launch=launch));print(json.dumps(dict(state='TYPED_RECOVERY_SUBMITTED_NOT_COMPLETE',model_requests=0)))
if __name__=='__main__':main()
