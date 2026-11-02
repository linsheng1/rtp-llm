"""New-candidate capture model at a safe completed-pair boundary; D retained.
Reuse capture23 helper/protocol, original controller and original drain wrapper.
"""
import argparse,hashlib,json,secrets,sys
from pathlib import Path
import freeze_stage as f

def save_same(path,value):
 if path.exists():f.need(f.doc(path)[0]==value,'existing immutable CPU preparation differs')
 else:f.save_new(path,value)

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--bench',type=Path,required=True);p.add_argument('--models-dir',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--capture-reference',type=Path);p.add_argument('--overlay',type=Path);p.add_argument('--completed-capture',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args();sys.path.insert(0,str(a.bench));from remote_control import remote
 from cluster23_dispatch import submit
 if a.command=='freeze':
  ref,rpin=f.doc(a.reference);old,cpin=f.doc(a.capture_reference);over,opin=f.doc(a.overlay);prep=over['state-projection-91b-inner-v1/prepare-result.json']['value'];argv=old['argv'][:];argv=argv[:argv.index('--mode')];template_path=f.option(argv,'--observer-template');template_sha=f.option(argv,'--observer-template-sha256');previous=f.option(ref['argv'],'--root');root=f.CONTROL+'/airline4-P-state91b-capture30';tag='state91b-capture30'
  code='''import json,hashlib\nfrom pathlib import Path\np=Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r;proofs=[];c=Path(%r)\nfor action,tag in [('d-restore','fix91bD1'),('baseline','fix91bN1')]:\n q=c/('dgdet-stage-'+action+'-'+tag)/'status.json';v=json.loads(q.read_text());assert v['state']=='STAGE_COMPLETE' and v['actual_request_returncode']==0;proofs.append(dict(URL='http://10.44.61.73:19091/publication-precision-20261101/control/'+str(q.relative_to(c)),SHA256=hashlib.sha256(q.read_bytes()).hexdigest(),fields=dict(state='STAGE_COMPLETE',action=action)))\nprint(json.dumps(dict(template=json.loads(p.read_text()),completed=proofs)))'''%(template_path,template_sha,f.CONTROL)
  template=remote('cluster3',f.HOSTS['P'],'import json,hashlib\nfrom pathlib import Path\np=Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r;print(json.dumps(json.loads(p.read_text())))'%(template_path,template_sha),60)
  actual_pin=f.read(a.reference)[1]
  code="""import json,hashlib,os
from pathlib import Path
old=Path(%r);s=json.loads((old/'model-service-status.json').read_text());cfg=json.loads((old/'observer-runtime-config.json').read_text());c=old.parent
receiver=old/'arm-receiver';rv=json.loads((receiver/'status.json').read_text());assert rv['state']=='FAILED_OR_EXPIRED_NO_RETRY' and not rv['accepted'] and rv['failure']['reason']=='ARM exceeds actual model active lease' and not list((receiver/'ledger').iterdir())
for batch in ['capture27','capture28']:
 peer=json.loads((c/('airline4-P-state-pilot-'+batch+'-peer-P')/'status.json').read_text());assert peer['state']=='BLOCKED' and not peer['remaining_owned']
for case in cfg['cases']:assert not Path(case['arm_path']).exists() and not list(Path(case['evidence_dir']).glob('*-live-rid-bound.json'))
v=dict(state='OWN_CAPTURE_ZERO_RPC_NO_ACCEPTED_ARM',model_requests_completed=0,actual_RPC=0,own_arm_markers_absent=True,epochs=dict(P={k:s['model_root'][k] for k in ['PID','UID','start_ticks']}),scope='own capture27 rejected before native call, capture28 blocked before dispatch; external ingress not attested');p=c/'capture27-zero-rpc-handoff-proof.json'
if p.exists():assert not p.is_symlink() and json.loads(p.read_text())==v,'prior zeroRPC proof differs'
else:
 with p.open('x') as o:json.dump(v,o,indent=2,sort_keys=True);o.flush();os.fsync(o.fileno())
print(json.dumps(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())))"""%previous
  unused=remote('cluster3',f.HOSTS['P'],code,60);completed=[]
  oldroot=f.option(old['argv'],'--root');raw=json.dumps(template).replace(oldroot,root);template=json.loads(raw);nonces={}
  for i,case in enumerate(template['cases']):
   nonce=json.loads((a.models_dir/'nonces.json').read_text())['short_nonce' if i==0 else 'full_nonce'] if (a.models_dir/'nonces.json').exists() else secrets.token_hex(24);oldnonce=case['nonce'];case=json.loads(json.dumps(case).replace(oldnonce,nonce));case['runtime_manifest_sha256']=prep['manifest']['SHA256'];case['stage_query_layers']=[2,3,4,5,6,7];template['cases'][i]=case;nonces['short_nonce' if i==0 else 'full_nonce']=nonce
  a.models_dir.mkdir(parents=True,exist_ok=True);save_same(a.models_dir/'nonces.json',nonces);local_template=a.models_dir/'observer-template.json';save_same(local_template,template);tsha=f.read(local_template)[1]['SHA256'];tp=f.CONTROL+'/observer-template-state91b-capture30.json';helper=f.read(f.BASE/'device/c3_P_state_model_service_v11.py')[1]
  replace={'--root':root,'--namespace':'pubprec_state91b_Dorigin_v1','--seconds':'1800','--env-json':prep['environment']['path'],'--env-sha256':prep['environment']['SHA256'],'--manifest':prep['manifest']['path'],'--manifest-sha256':prep['manifest']['SHA256'],'--consumer-root':f.option(ref['argv'],'--consumer-root'),'--keeper-root':f.CONTROL+'/state-candidate91b-static-P-v2','--keeper-helper':f.option(ref['argv'],'--keeper-helper'),'--keeper-helper-sha256':f.option(ref['argv'],'--keeper-helper-sha256'),'--observer-template':tp,'--observer-template-sha256':tsha,'--observer-code':f.CONTROL+'/p_only_state_observer_v8.py','--observer-code-sha256':f.read(f.BASE/'device/p_only_state_observer_v8.py')[1]['SHA256'],'--observer-bootstrap':f.CONTROL+'/p_only_state_observer_bootstrap_v8.py','--observer-bootstrap-sha256':f.read(f.BASE/'device/p_only_state_observer_bootstrap_v8.py')[1]['SHA256']}
  for k,v in replace.items():argv[argv.index(k)+1]=v
  for source_name in ['p_only_state_observer_v8.py','p_only_state_observer_bootstrap_v8.py']:
   raw=Path(f.BASE/'device'/source_name).read_text();code="""import json,os
from pathlib import Path
p=Path(%r);raw=%r
if p.exists():assert not p.is_symlink() and p.read_text()==raw
else:
 with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
print(json.dumps(dict(source_verified=True)))"""%(f.CONTROL+'/'+source_name,raw)
   remote('cluster3',f.HOSTS['P'],code,60)
  argv[3]=f.CONTROL+'/c3_P_state_model_service_v11.py';assets={argv[3]:Path(helper['path']).read_text(),tp:local_template.read_text()};code='''import json,os,subprocess,hashlib\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw\n else:\n  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nchecks=[]\nfor mode in ['compose','check']:\n r=subprocess.run(%r+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180);checks.append(dict(mode=mode,rc=r.returncode,stdout=r.stdout[-500:],stderr=r.stderr[-2000:]));assert r.returncode==0,checks\nr=Path(%r);print(json.dumps(dict(checks=checks,actual_RPC=0,plan_SHA256=hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),**{'observer-config-manifest.json':json.loads((r/'observer-config-manifest.json').read_text())})))'''%(assets,argv,root);cpu=remote('cluster3',f.HOSTS['P'],code,420);cpu['argv']=argv;f.save_new(a.models_dir/'P-CPU.json',cpu);h=f.read(f.BASE/'handoff_zero_rpc_capture.py')[1];w=f.read(a.bench/'c3_after_reference_drain_v1.py')[1];guard=f.CONTROL+'/state91b-capture30-P-after-drain';plan=dict(schema='existing-safe-freshP-epoch-v1',candidate_commit='91b4635faf8c65cc67e66ab52a46f5c932cd08ea',previous_root=previous,previous_plan_SHA256=ref['plan_SHA256'],model_root=root,model_plan_SHA256=cpu['plan_SHA256'],model_argv=argv+['--mode','run'],unused_local_proof=unused,completed_stages=completed,CPU=cpu,handoff=dict(path=f.CONTROL+'/'+Path(h['path']).name,SHA256=h['SHA256']),wrapper=dict(path=f.CONTROL+'/'+Path(w['path']).name,SHA256=w['SHA256']),drain_root=guard,handoff_claim=f.CONTROL+'/state91b-capture30-handoff-claim.json',local_pins=dict(entry=f.read(__file__)[1],handoff=h,wrapper=w,reference=rpin,capture_reference=cpin,overlay=opin,template=f.read(local_template)[1],model_helper=helper,completed_capture=actual_pin,observer=f.read(f.BASE/'device/p_only_state_observer_v8.py')[1],bootstrap=f.read(f.BASE/'device/p_only_state_observer_bootstrap_v8.py')[1]),job=dict(host=f.HOSTS['P'],name='state91b-capture30-P-model',timeout=4500,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/'+Path(h['path']).name,'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=Path(root).name+'/model-service-final.json',json_fields=dict(state='MODEL_SERVICE_EXITED'))]));sha=f.save_new(a.plan,plan);f.save_new(a.models_dir/'P-dispatch.json',dict(role='P',host=f.HOSTS['P'],root=Path(root).name,plan_SHA256=cpu['plan_SHA256'],argv=plan['model_argv'],parent_controller_status=f.CONTROL+'/'+plan['job']['name']+'-status.json',p_launch_guard_root=guard,epoch_plan_SHA256=sha));print(json.dumps(dict(SHA256=sha,model_requests=0)));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'capture model plan drift')
 for value in plan['local_pins'].values():f.need(f.read(value['path'])[1]==value,'pinned source drift')
 job=dict(plan['job']);job['argv']=[a.sha256 if x=='${PLAN_SHA256}' else x for x in job['argv']]
 if a.command=='check':print(json.dumps(dict(state='CAPTURE_MODEL_OFFLINE_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(job));return
 raw=Path(plan['local_pins']['handoff']['path']).read_text();remote('cluster3',job['host'],"import json,os\nfrom pathlib import Path\np=Path(%r);raw=%r\nif p.exists():assert not p.is_symlink() and p.read_text()==raw\nelse:\n with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nprint(json.dumps(dict(handoff_source=True)))"%(plan['handoff']['path'],raw),60)
 code='''import json,os\nfrom pathlib import Path\np=Path(%r);raw=%r\nif p.exists():assert not p.is_symlink() and p.read_text()==raw\nelse:\n with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nprint(json.dumps(dict(plan_published=True)))'''%(f.CONTROL+'/'+a.plan.name,a.plan.read_text());remote('cluster3',job['host'],code,60);launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=launch));print(json.dumps(dict(state='CAPTURE_MODEL_HANDOFF_SUBMITTED_NOT_READY',model_requests=0)))
if __name__=='__main__':main()
