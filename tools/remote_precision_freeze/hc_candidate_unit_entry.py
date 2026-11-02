"""Persistent freeze/check/render -> original submit for candidate module tests."""
import argparse,json,subprocess,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--overlay',type=Path);p.add_argument('--worktree',type=Path);p.add_argument('--bench',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  overlay,pin=f.doc(a.overlay);prepared=overlay['hc-tf32-031-inner-v1/prepare-result.json']['value'];f.need(prepared['state']=='IMMUTABLE_PYTHON_CANDIDATE_OVERLAY_CREATED_NOT_IMPORTED','overlay not prepared');rel='rtp_llm/models_py/modules/dsv41/hc/delayed.py';raw,comp=f.read(a.worktree/rel);candidate=prepared['candidate_commit'];committed=subprocess.check_output(['git','-C',str(a.worktree),'show',candidate+':'+rel]);f.need(raw==committed,'candidate worktree differs from commit')
  files={key:f.read(path)[1] for key,path in [('entry',Path(__file__)),('script',f.BASE/'hc_candidate_unit.py'),('bootstrap',f.BASE/'gpu_before_runtime_exec.py'),('test',a.worktree/'rtp_llm/models_py/modules/dsv41/test/test_v41_mhc_prenorm.py'),('nvml_helper',a.bench/'c3_FFN_MegaMHC_counterfactual_v1.py')]};test=files['test'];f.need(Path(test['path']).read_bytes()==subprocess.check_output(['git','-C',str(a.worktree),'show',candidate+':rtp_llm/models_py/modules/dsv41/test/'+Path(test['path']).name]),'test is not committed candidate')
  tag='hc031-module-unit-v1';plan=dict(schema='existing-candidate-module-unit-plan-v1',candidate_commit=candidate,environment=prepared['environment'],manifest=prepared['manifest'],delayed_SHA256=comp['SHA256'],local_pins=dict(files,overlay=pin),output_root=f.CONTROL+'/'+tag,admission_proof=f.CONTROL+'/'+tag+'-admission.json',bench=str(a.bench.resolve()),model_requests=0)
  for k in ['script','test','bootstrap','nvml_helper']:plan[k]=dict(path=f.CONTROL+'/'+(('hc031_'+Path(files[k]['path']).name) if k=='test' else Path(files[k]['path']).name),SHA256=files[k]['SHA256'])
  plan['job']=dict(host=f.HOSTS['P'],name=tag,timeout=240,argv=['/opt/conda310/bin/python3.10','-B','-S',plan['bootstrap']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))]);print(json.dumps(dict(SHA256=f.save_new(a.plan,plan))));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'plan drift')
 for value in plan['local_pins'].values():f.need(f.read(value['path'])[1]==value,'source/receipt drift')
 job=dict(plan['job']);job['argv']=[a.sha256 if x=='${PLAN_SHA256}' else x for x in job['argv']]
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(job));return
 sys.path.insert(0,plan['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 assets={plan[k]['path']:Path(plan['local_pins'][k]['path']).read_text() for k in ['script','test','bootstrap']};assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text()
 code='''import os,json,subprocess\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable control differs'\n else:\n  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nr=subprocess.run(%r,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr\nprint(json.dumps(dict(CPU_prepare=json.loads(r.stdout))))'''%(assets,['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256',a.sha256,'--check'])
 cpu=remote('cluster3',job['host'],code,150);result=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''});f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=result));print(json.dumps(dict(state='SUBMITTED_NOT_VALIDATED',model_requests=0)))
if __name__=='__main__':main()
