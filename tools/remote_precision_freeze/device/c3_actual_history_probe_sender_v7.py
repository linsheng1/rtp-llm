"""Reuse frozen CURRENT_V7 native-artifact probe; send metadata directly to C3 seed.

v3: parameterized model root/plan (dgdet relaunch), core4 SHA, explicit terminal
detection before any source request. No business input; probe body stays device-side.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
sys.path.insert(0,str(C/'inner-replay-tools-v2'));sys.path.insert(1,str(C))
TOOL=C/'c3_inner_precision_replay_v6.py'
PROOF_SHA='53a8003926b1df1077dfbfdcb129336b0af599a045317c0f297a0e3abe12dbfa'
STATE='CURRENT_V7_MODEL_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED'
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--role',choices=['P','D'],required=True)
p.add_argument('--model-root',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--parent-controller-status',type=Path,required=True)
p.add_argument('--fresh-proof',action='store_true');a=p.parse_args()
assert os.getuid()==500 and a.output_root.parent==C and not a.output_root.exists();a.output_root.mkdir(mode=0o700)
state={'state':'SOURCE_PROBE_RUNNING_NO_RPC','role':a.role,'at':time.time(),'model_requests':0}
def save():
 q=a.output_root/'status.json.partial';q.write_text(json.dumps(state,indent=2));q.replace(a.output_root/'status.json')
save();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('finite600s existing model readiness/probe barrier')));signal.alarm(600)
try:
 assert hashlib.sha256(TOOL.read_bytes()).hexdigest()=='37e17a974492dcd98dadf0ac3c487765b8ae24e622594753d42573b2ed18b5e1'
 deadline=time.monotonic()+540
 while True:
  if a.parent_controller_status.exists():
   parent=json.loads(a.parent_controller_status.read_bytes());assert parent['state'] not in ['FAILED','CANCELLED','TIMED_OUT','BLOCKED'],'parent model controller terminal before source request'
  if not (a.model_root/'model-service-status.json').exists() or not (a.model_root/'model-status.json').exists():
   assert time.monotonic()<deadline,'original readiness deadline before first model status';state.update(state='WAITING_FIRST_MODEL_STATUS',heartbeat_at=time.time());save();time.sleep(20);continue
  status=json.loads((a.model_root/'model-service-status.json').read_bytes());control=json.loads((a.model_root/'model-status.json').read_bytes())
  assert status['state'] not in ['FAILED','BLOCKED','MODEL_SERVICE_EXITED'] and control['state'] not in ['FAILED','CANCELLED','TIMED_OUT','SUCCEEDED'],'actual model dependency terminal before source request'
  assert time.time()-status.get('heartbeat_at',(a.model_root/'model-service-status.json').stat().st_mtime)<=45 and time.monotonic()<deadline,'original model heartbeat/readiness deadline'
  if status.get('model_READY') is True and status['state']=='MODEL_READY_KV_EMPTY_PENDING':break
  state.update(state='WAITING_ACTUAL_LOCAL_MODEL_READY',model_state=status['state'],heartbeat_at=time.time());save();time.sleep(20)
 with urlopen(Request('http://10.44.61.73:19107/ready/'+a.role,data=json.dumps({'role':a.role}).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=10) as response:assert response.status==200
 while True:
  try:
   with urlopen('http://10.44.61.73:19107/',timeout=10) as response:
    if response.status==200:break
  except __import__('urllib.error',fromlist=['HTTPError']).HTTPError as error:assert error.code==409
  assert time.monotonic()<deadline,'existing paired readiness barrier not complete';state.update(state='WAITING_PAIRED_ACTUAL_MODEL_READY_BARRIER',heartbeat_at=time.time());save();time.sleep(20)
 args=['sudo','-n','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+str(C/'inner-replay-tools-v2')+':'+str(C),'/opt/conda310/bin/python3.10','-B','-S',str(TOOL),'probe','--model-root',str(a.model_root),'--plan-sha256',a.plan_sha256]
 probe=subprocess.run(args,capture_output=True,text=True,timeout=125);assert probe.returncode==0,'existing readonly UID0 probe failed: '+probe.stderr[-1000:];value=json.loads(probe.stdout);assert value['observer_UID']==0;raw=json.dumps(value,sort_keys=True).encode();(a.output_root/'source.json').write_bytes(raw)
 assert value['state']==STATE and value['role']==a.role,'CURRENT_V7 probe state differs'
 with urlopen(Request('http://10.44.61.73:19107/'+a.role,data=raw,headers={'Content-Type':'application/json'},method='POST'),timeout=15) as response:
  assert response.status==200;received=json.loads(response.read(4096));assert received['SHA256']==hashlib.sha256(raw).hexdigest()
 if a.fresh_proof:
  while True:
   try:
    with urlopen('http://10.44.61.73:19107/all-probes',timeout=10) as response:
     assert response.status==200;break
   except __import__('urllib.error',fromlist=['HTTPError']).HTTPError as error:assert error.code==409
   assert time.monotonic()<deadline,'paired native-probe completion barrier expired';time.sleep(2)
 if a.fresh_proof:
  assert a.role=='P';helper=C/'prove_c3_fresh_p_empty_v4.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()==PROOF_SHA
  plan=json.loads((a.model_root/'plan.json').read_text());proof=C/(a.output_root.name+'-fresh-proof')
  argv=['/opt/conda310/bin/python3.10','-B','-S',str(helper),'--model-root',str(a.model_root),'--plan-sha256',a.plan_sha256,'--namespace',plan['namespace'],'--output-dir',str(proof),'--seconds','120']
  with (a.output_root/'proof.stdout').open('xb') as out,(a.output_root/'proof.stderr').open('xb') as err:q=subprocess.run(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=out,stderr=err,timeout=125)
  assert q.returncode==0,'actual fresh P native-log proof failed'
  raw=(proof/'empty-proof.json').read_bytes()
  with urlopen(Request('http://10.44.61.73:19107/fresh-proof',data=raw,headers={'Content-Type':'application/json'},method='POST'),timeout=10) as response:assert response.status==200
 state.update(state='ACTUAL_NATIVE_SOURCE_PROBE_UPLOADED_DEVICE_DIRECT_NO_RPC',received=received,finished_at=time.time());save();print(json.dumps(state))
except BaseException as e:
 state.update(state='BLOCKED',error_kind=type(e).__name__,error=str(e)[:1500],finished_at=time.time());save()
 try:
  with urlopen(Request('http://10.44.61.73:19107/failed/'+a.role,data=json.dumps(state).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=10) as response:assert response.status==200
 except Exception:pass
 raise