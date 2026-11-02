"""Prepare only a new P observation epoch using unchanged observer10 sampling.

The real HC031 env/manifest and live Consumer/keeper are checked on P. D is
prepared separately with the original HC031 epoch entry. No environment build.
"""
import argparse,json,secrets,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['prepare','check','render','submit']);p.add_argument('--recipe',type=Path);p.add_argument('--directory',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command!='prepare':
  v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'immutable capture37 epoch')
  for x in v['local_pins'].values():f.need(f.read(x['path'])[1]==x,'capture37 epoch source drift')
  if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
  if a.command=='render':print(json.dumps(v['job']));return
  sys.path.insert(0,v['bench']);from cluster23_dispatch import submit
  j=v['job'];launch=submit('cluster3',j['host'],j['name'],j['argv'],j['timeout'],j['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=launch));print(json.dumps(dict(state='SUBMITTED_NOT_READY',model_requests=0)));return
 r,rp=f.doc(a.recipe);f.need(r['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and r['seconds']==1800 and r['namespace']=='pubprec_hc031_Dorigin_v1','current same-candidate two-case protocol only')
 b=Path(r['bench']);sys.path.insert(0,str(b));from remote_control import remote
 ref,refp=f.doc(r['P_reference_dispatch']);template,tp=f.doc(r['observer_template_reference']);binding,bp=f.doc(r['control_binding']);f.need(binding['candidate_commit']==r['candidate_commit'],'actual bound candidate controls')
 files=dict(entry=f.read(__file__)[1],helper=f.read(b/'c3_P_hc031_capture_model_v1.py')[1],wrapper=f.read(b/'c3_after_reference_drain_v1.py')[1])
 for n in ['p_only_state_observer_v10.py','p_only_state_observer_bootstrap_v10.py','p_only_state_observer_v8.py','p_only_state_observer_v2.py']:files[n]=f.read(b/n)[1]
 helper=f.literals(files['helper']['path']);hooks=helper['HOOK_MODULES'];root=f.CONTROL+'/airline4-P-hc031-capture37';argv=list(ref['argv']);oldroot=f.option(argv,'--root')
 if '--mode' in argv:i=argv.index('--mode');del argv[i:i+2]
 argv[3]=f.CONTROL+'/c3_P_hc031_capture_model_v1.py'
 changes={'--root':root,'--seconds':'1800','--namespace':r['namespace'],'--env-json':f.CONTROL+'/hc-tf32-031-inner-v1/environment.json','--env-sha256':'1bb3d2ba9082c9ce17c2990ff9a8448855d68c88d793d89b8d25538d8355f6b2','--manifest':f.CONTROL+'/hc-tf32-031-inner-v1/manifest.json','--manifest-sha256':'8614842e06e19bb5be272326ef990ff42263a837745fdbb12a22489e8f1e6ba5','--consumer-root':f.CONTROL+'/consumer91b-P-v3','--keeper-root':f.CONTROL+'/hc031-static-P-v1','--keeper-helper':f.CONTROL+'/revalidate_hc031_static_weights.py','--keeper-helper-sha256':'090521ba16350783e0a7b8d3b553f8f225e2131f2d7c551f68ae057fb460e868'}
 for key,value in changes.items():argv[argv.index(key)+1]=value
 code='''import json,hashlib
from pathlib import Path
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;m=json.loads(raw);assert m['source_commit']==%r
out=dict(hooks={n:m['production_Python'][n.replace('.','/')+'.py']['SHA256'] for n in %r},attention={n:m['production_Python'][n]['SHA256'] for n in %r},native={Path(n).name:v['SHA256'] for n,v in m['native'].items()})
print(json.dumps(out))
'''%(changes['--manifest'],changes['--manifest-sha256'],r['candidate_commit'],hooks,['rtp_llm/models_py/modules/dsv41/fp8/attention.py','rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py','rtp_llm/models_py/modules/dsv41/fp8/_v41_mega_attention.py'])
 actual=remote('cluster3',f.HOSTS['P'],code,60);template=json.loads(json.dumps(template).replace(oldroot,root));nonces={}
 for i,case in enumerate(template['cases']):
  nonce=secrets.token_hex(24);old=case['nonce'];case=json.loads(json.dumps(case).replace(old,nonce));nonces['short_nonce' if i==0 else 'full_nonce']=nonce
  case.update(source_commit=r['candidate_commit'],namespace=r['namespace'],input_ids_sha256='3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c',runtime_manifest_sha256=changes['--manifest-sha256'],native_sha256=actual['native'],hook_source_sha256=actual['hooks'],l2_attention_production_sha256=actual['attention']);template['cases'][i]=case
 a.directory.mkdir(parents=True,exist_ok=True);local=a.directory/'observer-template.json';tsha=f.save_new(local,template);f.save_new(a.directory/'nonces.json',nonces)
 path=f.CONTROL+'/observer-template-hc031-capture37.json';argv[argv.index('--observer-template')+1]=path;argv[argv.index('--observer-template-sha256')+1]=tsha
 assets={f.CONTROL+'/c3_P_hc031_capture_model_v1.py':Path(files['helper']['path']).read_text(),path:local.read_text()}
 code='''import json,os,hashlib,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable capture37 source'
 else:
  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
for name,expected in %r.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,'unchanged observer source drift'
checks=[];args=%r
for mode in ['compose','check']:
 q=subprocess.run(args+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180);checks.append(dict(mode=mode,returncode=q.returncode,stdout=q.stdout[-1200:],stderr=q.stderr[-2200:]));assert q.returncode==0,checks
c=Path(%r);manifest=json.loads((c/'observer-config-manifest.json').read_text());print(json.dumps(dict(state='ACTUAL_CAPTURE37_CPU_PREPARE_PASS',checks=checks,model_requests=0,plan_SHA256=hashlib.sha256((c/'plan.json').read_bytes()).hexdigest(),**{'observer-config-manifest.json':manifest})))
'''%(assets,{f.CONTROL+'/'+n:v['SHA256'] for n,v in files.items() if n.endswith('.py')},argv,root)
 cpu=remote('cluster3',f.HOSTS['P'],code,420);f.save_new(a.directory/'P-CPU.json',cpu)
 name='hc031-capture37-P-model';guard=f.CONTROL+'/hc031-capture37-P-after-drain';job=dict(host=f.HOSTS['P'],name=name,timeout=2340,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/c3_after_reference_drain_v1.py','--previous-root',r['previous_P_root'],'--state-root',guard,'--']+argv+['--mode','run'],verify=[dict(path=Path(root).name+'/model-service-final.json',json_fields=dict(state='MODEL_SERVICE_EXITED'))])
 v=dict(candidate_commit=r['candidate_commit'],role='P',bench=str(b),model_root=root,model_plan_SHA256=cpu['plan_SHA256'],job=job,model_requests=0,local_pins=dict(files,recipe=rp,reference=refp,template_reference=tp,binding=bp,template=f.read(local)[1]))
 sha=f.save_new(a.directory/'P-model-plan.json',v);f.save_new(a.directory/'P-dispatch.json',dict(role='P',host=f.HOSTS['P'],root=Path(root).name,plan_SHA256=cpu['plan_SHA256'],argv=argv+['--mode','run'],parent_controller_status=f.CONTROL+'/'+name+'-status.json',epoch_plan_SHA256=sha,launch_guard_root=guard));print(json.dumps(dict(CPU='PASS',plan_SHA256=sha,model_requests=0)))

if __name__=='__main__':main()
