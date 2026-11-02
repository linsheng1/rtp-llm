"""Completed capture30 -> bounded observer9 CPU prepare -> original controller."""
import argparse,hashlib,json,secrets,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--reference',type=Path);p.add_argument('--completed-capture',type=Path);p.add_argument('--models-dir',type=Path,required=True);p.add_argument('--bench',type=Path,required=True);p.add_argument('--receipt',type=Path);a=p.parse_args();sys.path.insert(0,str(a.bench));from remote_control import remote
 from cluster23_dispatch import submit
 if a.command=='freeze':
  ref,rpin=f.doc(a.reference);done,dpin=f.doc(a.completed_capture);value=done['value'];f.need(value['model_requests_completed']==2 and value['full_ids']['new_normal_equals_reference'] and value['full_ids']['new_Dremote_equals_reference'],'completed calibrated pair required');argv=ref['argv'][:];argv=argv[:argv.index('--mode')];old=f.option(argv,'--root');root=f.CONTROL+'/airline4-P-state91b-capture34';tp=f.CONTROL+'/observer-template-state91b-capture34.json'
  keys=['rtp_llm/models_py/modules/dsv41/fp8/attention.py','rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py','rtp_llm/models_py/modules/dsv41/fp8/_v41_mega_attention.py']
  code='''import json,hashlib,os,time
from pathlib import Path
old=Path(%r);consumer=json.loads((old.parent/'consumer91b-P-v2/status.json').read_text());assert consumer['state']=='READY' and consumer['REQUEST_BLOCKED'] is False and time.time()-consumer['heartbeat_at']<=45 and consumer['deadline_epoch']-time.time()>1875,'whole new Consumer lease required before model compose';s=json.loads((old/'model-service-final.json').read_text());c=json.loads((old/'model-status.json').read_text());assert s['state']=='MODEL_SERVICE_EXITED' and s['actual_model_root_returncode']==0 and c['cleanup_state']=='FINISHED' and not c['remaining_owned_processes'],'old original drain incomplete'
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;template=json.loads(raw);p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;m=json.loads(raw)
print(json.dumps(dict(template=template,source_pins={k:m['production_Python'][k]['SHA256'] for k in %r},old_root_rc=0)))'''%(old,f.option(argv,'--observer-template'),f.option(argv,'--observer-template-sha256'),f.option(argv,'--manifest'),f.option(argv,'--manifest-sha256'),keys)
  observed=remote('cluster3',f.HOSTS['P'],code,60);template=json.loads(json.dumps(observed['template']).replace(old,root));nonces={}
  diag=dict(schema='c3-l2-attention-readonly-v1',layer=2,query_position=7168,max_selected_rows=640,max_heads=64,original_extra_limit=4*1024**2,natural_one_chunk_only=True)
  for i,case in enumerate(template['cases']):
   nonce=secrets.token_hex(24);prior=case['nonce'];case=json.loads(json.dumps(case).replace(prior,nonce));case.update(l2_attention_diagnostic=diag,l2_attention_production_sha256=observed['source_pins']);template['cases'][i]=case;nonces['short_nonce' if i==0 else 'full_nonce']=nonce
  a.models_dir.mkdir(parents=True,exist_ok=True);f.save_new(a.models_dir/'nonces.json',nonces);local=a.models_dir/'observer-template.json';tsha=f.save_new(local,template)
  pins={name:f.read(a.bench/name)[1] for name in ['c3_P_state_model_service_v13.py','p_only_state_observer_v10.py','p_only_state_observer_bootstrap_v10.py','p_only_state_observer_v8.py','p_only_state_observer_v2.py']}
  argv[3]=f.CONTROL+'/c3_P_state_model_service_v13.py'
  for key,val in {'--root':root,'--seconds':'1800','--consumer-root':f.CONTROL+'/consumer91b-P-v2','--observer-template':tp,'--observer-template-sha256':tsha,'--observer-code':f.CONTROL+'/p_only_state_observer_v10.py','--observer-code-sha256':pins['p_only_state_observer_v10.py']['SHA256'],'--observer-bootstrap':f.CONTROL+'/p_only_state_observer_bootstrap_v10.py','--observer-bootstrap-sha256':pins['p_only_state_observer_bootstrap_v10.py']['SHA256']}.items():argv[argv.index(key)+1]=val
  # Source only. Preserve already-published parent/base bytes, send new files
  # in bounded chunks; never overwrite a previous version or runtime plan.
  assets={f.CONTROL+'/'+k:Path(v['path']).read_text() for k,v in pins.items()};assets[tp]=local.read_text()
  for name,raw in assets.items():
   code='''import json,os\nfrom pathlib import Path\np=Path(%r);raw=%r\nif p.exists():assert not p.is_symlink() and p.stat().st_uid==500 and p.read_text()==raw,'immutable source differs'\nelse:\n with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nprint(json.dumps(dict(source_verified=True)))'''%(name,raw);remote('cluster3',f.HOSTS['P'],code,60)
  code='''import json,subprocess,os,hashlib
from pathlib import Path
checks=[]
for mode in ['compose','check']:
 r=subprocess.run(%r+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180);checks.append(dict(mode=mode,rc=r.returncode,stdout=r.stdout[-1000:],stderr=r.stderr[-2400:]));assert r.returncode==0,checks
p=Path(%r);doc=json.loads((p/'plan.json').read_text());env=doc['environment'];raw=env.get('DSV4_CHUNK_TOKENS',env.get('DSV4_FLASH_MLA_SPARSE_Q_CHUNK','16384'));chunk=int(raw) if raw.isdecimal() else 16384;assert chunk>=1794,'actual streaming chunk requires a different sampling adapter'
print(json.dumps(dict(checks=checks,chunk_preflight=chunk,actual_RPC=0,plan_SHA256=hashlib.sha256((p/'plan.json').read_bytes()).hexdigest(),**{'observer-config-manifest.json':json.loads((p/'observer-config-manifest.json').read_text())})))'''%(argv,root)
  cpu=remote('cluster3',f.HOSTS['P'],code,420);cpu['argv']=argv;f.save_new(a.models_dir/'P-CPU.json',cpu);wrapper=f.read(a.bench/'c3_after_reference_drain_v1.py')[1];tag='state91b-capture34-P-model';guard=f.CONTROL+'/state91b-capture34-P-after-drain'
  plan=dict(schema='existing-safe-freshP-epoch-v1',candidate_commit=value['candidate_commit'],previous_root=old,previous_plan_SHA256=ref['plan_SHA256'],model_root=root,model_plan_SHA256=cpu['plan_SHA256'],CPU=cpu,local_pins=dict(reference=rpin,completed_capture=dpin,entry=f.read(__file__)[1],wrapper=wrapper,template=f.read(local)[1],**pins),job=dict(host=f.HOSTS['P'],name=tag,timeout=1500,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/c3_after_reference_drain_v1.py','--previous-root',old,'--state-root',guard,'--']+argv+['--mode','run'],verify=[dict(path=Path(root).name+'/model-service-final.json',json_fields=dict(state='MODEL_SERVICE_EXITED'))]));sha=f.save_new(a.plan,plan);f.save_new(a.models_dir/'P-dispatch.json',dict(role='P',host=f.HOSTS['P'],root=Path(root).name,plan_SHA256=cpu['plan_SHA256'],argv=argv+['--mode','run'],parent_controller_status=f.CONTROL+'/'+tag+'-status.json',p_launch_guard_root=guard,epoch_plan_SHA256=sha));print(json.dumps(dict(SHA256=sha,actual_RPC=0)));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'immutable epoch plan differs')
 for pin in plan['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'local source drift')
 j=plan['job'];result=submit('cluster3',j['host'],j['name'],j['argv'],j['timeout'],j['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=result));print(json.dumps(dict(state='AFTER_VERIFIED_OLD_DRAIN_SUBMITTED_NOT_READY',actual_RPC=0)))
if __name__=='__main__':main()
