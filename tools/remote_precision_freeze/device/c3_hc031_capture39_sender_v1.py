"""Reuse frozen CURRENT_V7 native-artifact probe; send metadata directly to C3 seed.

v3: parameterized model root/plan (dgdet relaunch), core4 SHA, explicit terminal
detection before any source request. No business input; probe body stays device-side.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
sys.path.insert(0,str(C/'inner-replay-tools-v2'));sys.path.insert(1,str(C))
TOOL=C/'c3_hc031_capture39_core_v1.py'
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
 assert hashlib.sha256(TOOL.read_bytes()).hexdigest()=='03523d4ffa963bea7d5c865947155fe8f1a0318aa3e9d51852cc731ba40ecb50'
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

 import socket,stat
 controls=value['history_capture_controls'];completed=set()
 def socket_call(record,operation,label=''):
  proc=Path('/proc')/str(record['PID']);fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
  assert proc.stat().st_uid==record['UID']==500 and fields[19]==str(record['start_ticks']),'actual capture PID/UID/start drift'
  p=Path(record['socket']);assert p.is_socket() and p.stat().st_uid==500 and not p.stat().st_mode&0o077
  with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
   client.settimeout(60);client.connect(str(p));client.sendall((json.dumps(dict(operation=operation,label=label,request_drained=True))+'\n').encode());data=b''
   while not data.endswith(b'\n'):
    chunk=client.recv(65536);assert chunk and len(data)+len(chunk)<=2*1024**2;data+=chunk
  result=json.loads(data);assert result['state']==('CAPTURE_LATCHES_RESET' if operation=='reset' else 'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE'),result
  return result
 while True:
  assert time.monotonic()<deadline,'original finite source sender deadline'
  with urlopen('http://10.44.61.73:19107/capture/'+a.role,timeout=10) as response:command=json.loads(response.read(65536))
  action=command['action'];assert action!='BLOCKED',command
  if action in ['RESET','EXPORT'] and action not in completed:
   current=json.loads((a.model_root/'model-service-status.json').read_text());assert current['state']=='MODEL_READY_KV_EMPTY_PENDING' and current['model_READY'] and time.time()-current['heartbeat_at']<=45 and current['deadline_epoch']-420>time.time()+60,'actual model/lease unavailable during finite source phase'
   exports=[]
   for item in controls:
    record=item['record']
    if record is None:
     assert a.role=='P' and item['initial_latches_absent'] is True
     rank=item['rank'];directory=Path(item['expected_output'])
     if action=='RESET':
      assert not (directory/(str(rank['PID'])+'-control.json')).exists() and not (directory/(str(rank['PID'])+'.sock')).exists(),'current-epoch eager P must have no prior latch allocation/data'
      proc=Path('/proc')/str(rank['PID']);parts=proc.joinpath('stat').read_text().rsplit(')',1)[1].split();assert proc.stat().st_uid==500 and parts[19]==str(rank['start_ticks']) and parts[0] not in ['Z','X','x']
      exports.append(dict(state='LAZY_EAGER_P_LATCHES_ABSENT_BEFORE_FIRST_CP_FORWARD',**rank));continue
     path=directory/(str(rank['PID'])+'-control.json');record=json.loads(path.read_text());cfg_path=Path(item['config']['path']);raw_cfg=cfg_path.read_bytes();assert hashlib.sha256(raw_cfg).hexdigest()==item['config']['SHA256'];cfg=json.loads(raw_cfg)
     assert record['state']=='CAPTURE_CONTROL_READY' and record['identity']==cfg and all(str(record[k])==str(rank[k]) for k in ['PID','UID','start_ticks']),'post-EOF lazy P control identity differs'
    result=socket_call(record,'reset' if action=='RESET' else 'snapshot','capture-'+str(command.get('request_id',0)))
    if action=='EXPORT':
     path=Path(result['file']);assert path.parent==Path(record['identity']['output']) and not path.is_symlink() and path.stat().st_uid==500 and path.stat().st_size==result['bytes']<=8*1024**2
     raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==result['SHA256']
     req=Request('http://10.44.61.73:19107/snapshot-bytes/'+a.role+'/'+str(record['PID']),data=raw,headers={'Content-Type':'application/octet-stream','X-Content-SHA256':result['SHA256']},method='POST')
     with urlopen(req,timeout=30) as response:assert response.status==200
    exports.append(result)
   ack=dict(action=action,role=a.role,model_root=value['model_root'],rank_controls=exports,at=time.time(),model_requests=0)
   if action=='RESET' and a.fresh_proof:
    helper=C/'prove_c3_fresh_p_empty_v4.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()==PROOF_SHA
    plan=json.loads((a.model_root/'plan.json').read_text());root=C/(a.output_root.name+'-source-empty-proof')
    cmd=['/opt/conda310/bin/python3.10','-B','-S',str(helper),'--model-root',str(a.model_root),'--plan-sha256',a.plan_sha256,'--namespace',plan['namespace'],'--output-dir',str(root),'--seconds','120']
    with (a.output_root/'empty.stdout').open('xb') as out,(a.output_root/'empty.stderr').open('xb') as err:q=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=out,stderr=err,timeout=125)
    assert q.returncode==0,'post-probe current fresh P proof failed'
    proof=(root/'empty-proof.json').read_bytes();ack['fresh_empty_proof']=json.loads(proof);ack['fresh_empty_proof_source']=dict(path=str(root/'empty-proof.json'),SHA256=hashlib.sha256(proof).hexdigest())
   data=json.dumps(ack,sort_keys=True).encode();assert len(data)<=2*1024**2
   with urlopen(Request('http://10.44.61.73:19107/capture-ack/'+a.role,data=data,headers={'Content-Type':'application/json'},method='POST'),timeout=20) as response:assert response.status==200
   completed.add(action);state.update(state='CAPTURE39_'+action+'_ACKNOWLEDGED',heartbeat_at=time.time());save()
  if completed=={'RESET','EXPORT'}:break
  if action=='DONE':break
  time.sleep(1)
 assert completed=={'RESET','EXPORT'},'both finite sampling barriers required'
 state.update(state='CAPTURE39_BASELINE_CAPTURE_EXPORTED_AFTER_EOF',received=received,finished_at=time.time(),model_requests=0);save();print(json.dumps(state))
except BaseException as e:
 state.update(state='BLOCKED',error_kind=type(e).__name__,error=str(e)[:1500],finished_at=time.time());save()
 try:
  with urlopen(Request('http://10.44.61.73:19107/failed/'+a.role,data=json.dumps(state).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=10) as response:assert response.status==200
 except Exception:pass
 raise