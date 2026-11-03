"""Finite C3 device-to-device stage receiver v4 (dgdet 61887b8).

Generalizes c3_D_source_receiver_v2/v3: waits for the paired per-role CURRENT_V7
probes (P=10.44.12.175, D=10.44.61.71), builds the action identity with core4,
then sends exactly ONE real native request for the requested action.

Business IDs/tensors/protobufs never leave the device; only counts/SHAs here.
"""
import os,json,time,hashlib,argparse
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True);p.add_argument('--action',required=True,choices=['d-source'])
p.add_argument('--tag',required=True);p.add_argument('--port',type=int,default=19107);p.add_argument('--seconds',type=int,default=1200)
p.add_argument('--request-timeout-seconds',type=float,default=180)
p.add_argument('--case-private',type=Path,default=None);p.add_argument('--case-sha256',default=None)
p.add_argument('--fixture',type=Path);p.add_argument('--fixture-sha256')
p.add_argument('--fresh-proof',type=Path);p.add_argument('--fresh-proof-sha256')
p.add_argument('--p-publication',type=Path);p.add_argument('--p-publication-sha256')
p.add_argument('--p-seed-result',type=Path);p.add_argument('--p-seed-result-sha256');p.add_argument('--p-manager-report',type=Path);p.add_argument('--p-manager-report-sha256')
a=p.parse_args()
assert os.getuid()==500 and 0<a.seconds<=1800 and a.port==19107 and __import__('re').fullmatch('[A-Za-z0-9_-]{1,40}',a.tag)
a.root.mkdir(mode=0o700,exist_ok=False);deadline=time.time()+a.seconds;got={};ready_roles=set();failures={}
STATE='CURRENT_V7_MODEL_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED'
def save(n,v):
 t=a.root/(n+'.partial');t.write_text(json.dumps(v,indent=2));t.replace(a.root/n)
class H(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  admitted=(len(got)==2) if self.path=='/all-probes' else (ready_roles=={'P','D'});self.send_response(200 if admitted else 409);self.end_headers();self.wfile.write(b'{}')
 def do_POST(self):
  try:
   if self.path=='/fresh-proof':
    assert self.client_address[0]=='10.44.12.175';length=int(self.headers['Content-Length']);assert 0<length<200000;raw=self.rfile.read(length);v=json.loads(raw);assert v['state']=='FRESH_P_ALLOCATOR_OWNER_CACHE_TREE_OBSERVED_EMPTY' and v['observer_requests_issued']==0;q=a.root/'fresh-proof.json';assert not q.exists();q.write_bytes(raw);self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'SHA256':hashlib.sha256(raw).hexdigest()}).encode());return
   if self.path.startswith('/failed/'):
    role=self.path.rsplit('/',1)[1];assert role in ['P','D'] and self.client_address[0]=={'P':'10.44.12.175','D':'10.44.61.71'}[role]
    length=int(self.headers['Content-Length']);assert 0<length<4000;failures[role]=json.loads(self.rfile.read(length));self.send_response(200);self.end_headers();self.wfile.write(b'{}');return
   if self.path.startswith('/ready/'):
    role=self.path.rsplit('/',1)[1];assert role in ['P','D'] and self.client_address[0]=={'P':'10.44.12.175','D':'10.44.61.71'}[role]
    length=int(self.headers['Content-Length']);assert 0<length<1000;v=json.loads(self.rfile.read(length));assert v['role']==role
    ready_roles.add(role);self.send_response(200);self.end_headers();self.wfile.write(b'{}');return
   role={'/P':'P','/D':'D'}[self.path];host={'P':'10.44.12.175','D':'10.44.61.71'}[role]
   assert self.client_address[0]==host and role not in got
   length=int(self.headers['Content-Length']);assert 0<length<200000
   data=self.rfile.read(length);v=json.loads(data)
   assert v['role']==role and v['host']==host and v['state']==STATE
   assert -2<=time.time()-v['observed_at']<=45
   q=a.root/(role+'.json');q.write_bytes(data);got[role]={'path':str(q),'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest()}
   self.send_response(200);self.end_headers();self.wfile.write(json.dumps(got[role]).encode())
  except Exception as e:self.send_response(400);self.end_headers();self.wfile.write(str(e).encode())
server=HTTPServer(('0.0.0.0',a.port),H);server.timeout=2
save('status.json',{'state':'RECEIVING_CONTROL_METADATA','action':a.action,'tag':a.tag,'PID':os.getpid(),'started_at':time.time(),'deadline':deadline})
try:
 while (len(got)<2 or (a.action.endswith('restore') and not (a.root/'fresh-proof.json').exists())) and not failures and time.time()<deadline:server.handle_request()
finally:server.server_close()
state='ACTUAL_TWO_MODEL_PROBES_RECEIVED' if len(got)==2 else 'FAILED_PROBES_INCOMPLETE'
save('status.json',{'state':state,'action':a.action,'probes':got,'upstream_failures':failures,'model_requests':0,'finished_at':time.time()})
if len(got)!=2 or failures:raise SystemExit(2)
import subprocess
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');H=C/'inner-replay-tools-v2';tool=C/'c3_indexer_head_core_v1.py'
assert hashlib.sha256(tool.read_bytes()).hexdigest()=='d129cb35457ccb7617e1c8a1a1648c786858637c6488d38d252e54e352cb95ac'
# Established private source contains original G7041; never copied to control host.
case=C/'airline4-native-case-v1/case-private.json';case_sha='0e2f29a837287d6201c27e005c4f2f70ce40d41885235b3ce8075738a6c13753'
assert hashlib.sha256(case.read_bytes()).hexdigest()==case_sha
edoc=json.loads((C/'airline4-cpu-driver-v1/environment.json').read_text());assert hashlib.sha256((C/'airline4-cpu-driver-v1/environment.json').read_bytes()).hexdigest()=='d5075b5c7ef68c5d1c1a4960f970c677a64d64227360c30505883abe2a39a2cc'
env=dict(edoc['environment']);assert env['CUDA_VISIBLE_DEVICES']=='';env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(H)+':'+str(C)+':'+env['PYTHONPATH'];prefix=[edoc['interpreter'],'-B','-S',str(tool)];ident=a.root/'identity';result=a.root/'request';prepare=a.root/'prepare'
def call(name,argv,timeout):
 with (a.root/(name+'.stdout')).open('xb') as out,(a.root/(name+'.stderr')).open('xb') as err:r=subprocess.run(argv,env=env,stdout=out,stderr=err,timeout=timeout)
 save(name+'-exit.json',dict(actual_child_returncode=r.returncode));assert r.returncode==0,name+' actual CLI failed; request count must be inspected before retry'
try:
 save('status.json',dict(state='IDENTITY_BUILDING_ZERO_RPC',model_requests=0,probes=got))
 call('identity',prefix+['identity','--action','d-source','--p-source',got['P']['path'],'--p-source-sha256',got['P']['SHA256'],'--d-source',got['D']['path'],'--d-source-sha256',got['D']['SHA256'],'--request-timeout-seconds',str(a.request_timeout_seconds),'--output-dir',str(ident)],60)
 ip=ident/'identity.json';sha=hashlib.sha256(ip.read_bytes()).hexdigest();common=['run','--action','d-source','--case-private',str(case),'--case-sha256',case_sha,'--identity-json',str(ip),'--identity-sha256',sha,'--timeout-seconds',str(a.request_timeout_seconds)]
 call('prepare',prefix+common+['--output-dir',str(prepare)],30)
 prep=json.loads((prepare/'summary.json').read_text());assert prep['state']=='INNER_REPLAY_NATIVE_PROTOBUF_CHECK_NO_REQUEST' and prep['model_requests']==0 and prep['input_tokens']==7041 and prep['input_ids_sha256']=='22510597b00881511fec56f072a5ef0ed3593b8a32b069e4189f0d75f258e49c'
 with (a.root/'request-once.json').open('x') as marker:json.dump(dict(identity_SHA256=sha,case_SHA256=case_sha),marker);marker.flush();os.fsync(marker.fileno())
 save('status.json',dict(state='ACTUAL_D_SOURCE_REQUEST_EXECUTING',model_requests='SEE_DEVICE_NATIVE_STATUS',CPU_prepare_returncode=0,probes=got))
 call('request',prefix+common+['--execute','--output-dir',str(result)],210)
 summary=json.loads((result/'summary.json').read_text());assert summary['state']=='COMPLETE_INNER_NATIVE_REQUEST_EVIDENCE_PENDING' and summary['model_requests']==1
 save('status.json',dict(state='D_SOURCE_COMPLETE',model_requests=1,summary=summary,finished_at=time.time(),probes=got,CPU_prepare_returncode=0))
 print(json.dumps(dict(state='D_SOURCE_COMPLETE',model_requests=1,input_tokens=summary['input_tokens'],output_tokens=summary['output_tokens'])))
except BaseException as e:
 save('status.json',dict(state='FAILED_DEPENDENTS_BLOCKED',error_kind=type(e).__name__,error=str(e),model_requests='SEE_DEVICE_NATIVE_STATUS',finished_at=time.time()));raise
