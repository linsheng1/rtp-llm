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
p.add_argument('--root',type=Path,required=True);p.add_argument('--action',required=True,choices=['baseline'])
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
if len(got)==2 and (not a.action.endswith('restore') or (a.root/'fresh-proof.json').exists()):
 import subprocess
 C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');H2=C/'inner-replay-tools-v2';tool=C/'c3_hc031_inner_replay_v1.py'
 assert hashlib.sha256(tool.read_bytes()).hexdigest()=='f51de136bbd84385704813903f3708eaf5cf10f7719787d2e4b4bd3f71a13204'
 ep=C/'airline4-cpu-driver-v1/environment.json';assert hashlib.sha256(ep.read_bytes()).hexdigest()=='d5075b5c7ef68c5d1c1a4960f970c677a64d64227360c30505883abe2a39a2cc'
 edoc=json.loads(ep.read_text());env=edoc['environment'];assert env['CUDA_VISIBLE_DEVICES']=='' and edoc['interpreter']=='/opt/conda310/bin/python3.10'
 env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(H2)+':'+str(C)+':'+env['PYTHONPATH'];prefix=[edoc['interpreter'],'-B','-S',str(tool)]
 ident=C/('hc031-'+a.action+'-identity-'+a.tag);result=C/('hc031-'+a.action+'-native-'+a.tag)
 identargs=prefix+['identity','--action',(('d-restore' if json.loads(Path(got['P']['path']).read_text())['namespace']==json.loads(Path(got['D']['path']).read_text())['namespace'] else 'p-restore') if a.action=='baseline' else a.action),'--p-source',got['P']['path'],'--p-source-sha256',got['P']['SHA256'],'--d-source',got['D']['path'],'--d-source-sha256',got['D']['SHA256'],'--request-timeout-seconds',str(a.request_timeout_seconds),'--output-dir',str(ident)]
 save('status.json',{'state':'IDENTITY_BUILDING','action':a.action,'probes':got,'started_at':time.time()})
 with (a.root/'identity.stdout').open('wb') as out,(a.root/'identity.stderr').open('wb') as err:q=subprocess.run(identargs,env=env,stdout=out,stderr=err,timeout=60)
 save('identity-exit.json',{'actual_child_returncode':q.returncode})
 if q.returncode:
  save('status.json',dict(state='BLOCKED_IDENTITY_ZERO_RPC',actual_child_returncode=q.returncode,model_requests=0));raise SystemExit(q.returncode)
 ip=ident/'identity.json';isha=hashlib.sha256(ip.read_bytes()).hexdigest()
 diag=C/'c3_hc031_history_diagnostic_v1.py'
 assert hashlib.sha256(diag.read_bytes()).hexdigest()=='ceb7cde8ad7cde2e711b95b117dd116e721a428490931876260dccc61e006c89'
 source=C/'hc031-source-source1/request/native/result.json';source_sha='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
 argv=[edoc['interpreter'],'-B','-S',str(diag),'--action',a.action,'--source',str(source),'--source-sha256',source_sha,'--identity-json',str(ip),'--identity-sha256',isha,'--timeout-seconds',str(a.request_timeout_seconds),'--execute']
 if a.action=='d-restore':
  argv+=['--native-evidence',str(C/'state91b-D-metadata-v1/native-evidence.json'),'--native-evidence-sha256','4a9f2952d45ab5d894c7af2bc870d411bf4b6b4530c3bcd8f2d560858ef61350','--manager-report',str(C/'state91b-D-metadata-v1/manager-report.json'),'--manager-report-sha256','a61387fce2f902f6cc34edaefd8bfcd2dc55085a447c03a3cc8a56349ac3659c']
 if a.action.endswith('restore'):
  proof=a.root/'fresh-proof.json';assert proof.is_file(),'actual fresh P proof absent'
  argv+=['--fresh-proof',str(proof),'--fresh-proof-sha256',hashlib.sha256(proof.read_bytes()).hexdigest()]
 if a.action=='p-restore':
  assert a.p_seed_result and a.p_seed_result_sha256 and a.p_manager_report and a.p_manager_report_sha256
  argv+=['--p-seed-result',str(a.p_seed_result),'--p-seed-result-sha256',a.p_seed_result_sha256,'--p-manager-report',str(a.p_manager_report),'--p-manager-report-sha256',a.p_manager_report_sha256]
 argv+=['--output-dir',str(result)]
 # Same actual CLI and complete prepare path, without --execute, before RPC.
 pre=[v for v in argv if v!='--execute'];pre[pre.index('--output-dir')+1]=str(result)+'-preflight'
 with (a.root/'prepare.stdout').open('xb') as out,(a.root/'prepare.stderr').open('xb') as err:check=subprocess.run(pre,env=env,stdout=out,stderr=err,timeout=60)
 save('prepare-exit.json',dict(actual_child_returncode=check.returncode,model_requests=0))
 if check.returncode:
  save('status.json',dict(state='BLOCKED_ACTUAL_CLI_PREPARE_ZERO_RPC',actual_child_returncode=check.returncode,model_requests=0));raise SystemExit(check.returncode)
 with (a.root/'request-once.claim').open('x') as claim:json.dump(dict(action=a.action,at=time.time(),identity_SHA256=isha),claim);claim.flush();os.fsync(claim.fileno())
 save('status.json',{'state':'NATIVE_EXECUTING','action':a.action,'probes':got,'identity_SHA256':isha,'started_at':time.time()})
 with (a.root/'request.stdout').open('wb') as out,(a.root/'request.stderr').open('wb') as err:q=subprocess.run(argv,env=env,stdout=out,stderr=err,timeout=240)
 save('request-exit.json',{'actual_child_returncode':q.returncode})
 summary=json.loads((result/'summary.json').read_text()) if (result/'summary.json').exists() else {}
 safe={k:summary.get(k) for k in ['state','action','input_tokens','output_tokens','input_ids_sha256','output_ids_sha256','max_new_tokens','model_requests','generation_profile_id','actual_local_reuse_len','actual_remote_reuse_len','request_eof_complete','response_eof_complete','stop_reason','finish_reason'] if k in summary}
 state='STAGE_COMPLETE' if q.returncode==0 and summary.get('state')=='COMPLETE_ACTUAL_D_HISTORY_DIAGNOSTIC' and json.loads((result/'native/result.json').read_text()).get('response_eof_complete') is True else 'STAGE_FAILED'
 save('status.json',{'state':state,'action':a.action,'probes':got,'actual_request_returncode':q.returncode,'summary':safe,'finished_at':time.time()})
 print(json.dumps({'state':state,'action':a.action,'actual_request_returncode':q.returncode,'summary':safe}));raise SystemExit(q.returncode)
print(json.dumps({'state':state,'action':a.action,'probes':got}));raise SystemExit(2)