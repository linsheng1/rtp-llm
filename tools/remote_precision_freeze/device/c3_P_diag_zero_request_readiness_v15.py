"""Finite20s READY wait + readonly actual probe + canonical refs + empty observer. No RPC/ARM/HTTP."""
import hashlib,json,os,subprocess,time,argparse
from c3_P_state_pilot_peer_v16 import load_batch,OwnedRunner
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--batch-json',type=Path,required=True);p.add_argument('--batch-sha256',required=True);a=p.parse_args();batch=load_batch(a.batch_json,a.batch_sha256);N=Path(batch['p_model_root']);R=C/(batch['job_prefix']+'-zero-request-readiness');assert os.getuid()==500 and not R.exists();R.mkdir(mode=0o700)
PLAN=batch['p_plan_sha256'];HELPER=batch['p_helper_sha256'];PY='/opt/conda310/bin/python3.10';probe=C/'c3_P_state_runtime_probe_v11.py';empty=C/'c3_P_state_empty_probe_v11.py';proof=C/(batch['job_prefix']+'-startup-empty')
v=dict(state='WAITING_ACTUAL_DIAGNOSTIC_READY',PID=os.getpid(),started_at=time.time(),interval_seconds=20,model_requests=0,ARM_publications=0,HTTP_bridge_starts=0)
def save(**kwargs):
 v.update(kwargs,heartbeat_at=time.time());t=R/'status.json.partial';t.write_text(json.dumps(v,indent=2));t.replace(R/'status.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
runner=OwnedRunner(R,save)
def run(argv,label,timeout):
 runner.run(argv,label,dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout)
 return R/(label+'.stdout')

save()
try:
 assert sha(probe)=='2567c04d3ba8b636dcf3c20471ed1ea4e41478c6028682b85c4f457dc9742960' and sha(empty)=='307bc5cb801291ac4664900f5a346a103fa7022d534c6b8a348c88328fec34ad'
 end=time.monotonic()+300
 while True:
  path=N/'model-service-status.json'
  s=json.loads(path.read_text()) if path.exists() else dict(state='WAITING_FIRST_MODEL_STATUS');v['model_state']=s['state']
  parent=Path(batch['p_launch_guard_root'])/'status.json'
  if parent.exists():assert json.loads(parent.read_text())['state'] not in ['FAILED','BLOCKED','DIAGNOSTIC_SERVICE_EXITED'],'upstream diagnostic launch terminal'
  if s.get('model_READY') is True and s['state']=='MODEL_READY_KV_EMPTY_PENDING':break
  assert s['state'] not in ['FAILED','EXITED','TIMED_OUT','CANCELLED','DRAINING_OWN_MODEL'],'upstream service terminal/not admissible';assert time.monotonic()<end,'bounded current model READY wait failed';save();time.sleep(20)
 args=['--model-root',str(N),'--plan-sha256',PLAN,'--helper-sha256',HELPER]
 source=run(['sudo','-n','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+str(C)+':'+str(C/'inner-replay-tools-v2'),PY,'-B','-S',str(probe),'probe']+args,'actual-P-source',125)
 doc=json.loads(source.read_text());assert doc['schema']=='c3-P-state-model-source-v3' and doc['state']=='CURRENT_DIAGNOSTIC_P_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED';source_sha=sha(source);assert doc['diagnostic_runtime']['config']['SHA256']==batch['observer_config_sha256'],'actual startup config differs from immutable batch'
 refs=run([PY,'-B','-S',str(probe),'publish-refs']+args+['--source',str(source),'--source-sha256',source_sha],'canonical-refs',20)
 refdoc=json.loads(refs.read_text());assert refdoc['state']=='TWO_CANONICAL_READY_REFS_PUBLISHED'
 save(state='ACTUAL_SOURCE_AND_CANONICAL_REFS_COMPLETE',P_source=dict(path=str(source),SHA256=source_sha),refs=refdoc)
 run([PY,'-B','-S',str(empty)]+args+['--namespace','pubprec_state91b_Dorigin_v1','--output-dir',str(proof),'--seconds','120'],'zero-request-empty',125)
 e=json.loads((proof/'empty-proof.json').read_text());assert e['state']=='FRESH_P_ALLOCATOR_OWNER_CACHE_TREE_OBSERVED_EMPTY' and e['exit_code']==0 and e['observer_requests_issued']==0
 save(state='ACTUAL_DIAGNOSTIC_SOURCE_REFS_AND_EMPTY_COMPLETE',empty_proof=dict(path=str(proof/'empty-proof.json'),SHA256=sha(proof/'empty-proof.json')),proof_sample_at=e['observation']['latest_metric']['at'],fresh_proof_max_age_seconds_for_future_request=75,live_RID_capture='PENDING_ACTUAL_ARMED_FORWARD',finished_at=time.time())
except BaseException as e:save(state='BLOCKED',error_kind=type(e).__name__,error=str(e),finished_at=time.time());raise
