"""Prepared finite seed coordinator, max2 released native client phases.
Default does not bind/listen/send. Tokens/tensors remain device-local.
"""
import argparse,hashlib,json,os,sys,time,threading,socket
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');H=C/'inner-replay-tools-v2';sys.path[:0]=[str(C),str(H)]
N=None;PPLAN=None;DPLAN=None;HELPER='717bcfa095ddeabc01d9c0cac9ff73b8f663a79eecd90b873bcc1306faac99f5';PY='/opt/conda310/bin/python3.10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def publish(path,raw):
 path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(path),SHA256=sha(path))
def publish_identity_replica(path,raw):
 if path.exists():
  assert path.is_file() and not path.is_symlink() and path.stat().st_uid==os.getuid() and path.read_bytes()==raw,'immutable identity replica drift'
  return dict(path=str(path),SHA256=sha(path))
 return publish(path,raw)
from c3_P_state_pilot_peer_v22 import OwnedRunner,FailureLatch,load_batch

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--client',type=Path,required=True);p.add_argument('--client-sha256',required=True);p.add_argument('--prepare-argv-json',type=Path,required=True);p.add_argument('--prepare-argv-sha256',required=True);p.add_argument('--batch-json',type=Path,required=True);p.add_argument('--batch-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps(dict(state='PREPARED_NO_LISTENER_NO_PHASE_NO_REQUEST',max_request_CLI=2,max_native_RPC=2)));return 0
 global N,PPLAN,DPLAN
 batch=load_batch(a.batch_json,a.batch_sha256);N=Path(batch['p_model_root']);PPLAN=batch['p_plan_sha256'];DPLAN=batch['d_plan_sha256']
 assert os.getuid()==500 and sha(a.client)==a.client_sha256 and sha(a.prepare_argv_json)==a.prepare_argv_sha256
 import c3_P_origin_seed_pipeline_v4 as base
 assert sha(Path(base.__file__))=='4bbe6d02df164b4479fd854b42eea31d1e11940f46612044acbe02b100f1160c'
 base.dependencies();envdoc=json.loads(base.CPU_ENV.read_text());env=envdoc['environment'];env['PYTHONPATH']=str(C)+':'+str(H)+':'+env['PYTHONPATH'];assert env['CUDA_VISIBLE_DEVICES']=='' and envdoc['interpreter']==PY
 R=C/(batch['job_prefix']+'-coordinator');R.mkdir(mode=0o700,exist_ok=False);runtime=base.Runtime(R);runtime.deadline=time.time()+900;runtime.state['deadline_epoch']=runtime.deadline;runtime.save(state='STARTING_METADATA_PHASE_SERVER',max_request_CLI=2,max_native_RPC=2,request_dispatches=0)
 threading.Thread(target=runtime.heartbeat,daemon=True).start();condition=threading.Condition();commands={r:dict(action='WAIT',token='initial') for r in ['P','D']};answers={};latch=FailureLatch();latest={}
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def response(self,status,v):
   data=canonical(v);self.send_response(status);self.end_headers();self.wfile.write(data)
  def do_GET(self):
   try:
    role=self.path.rsplit('/',1)[1];assert self.path=='/peer/'+role and self.client_address[0]=={'P':'10.44.12.175','D':'10.44.61.71'}[role]
    with condition:
     old=latest.get(role)
     if commands[role]['token']==old or commands[role]['action']=='WAIT':condition.wait(20)
     value=dict(commands[role]);latest[role]=value['token']
    self.response(200,value)
   except BaseException as e:self.response(400,dict(error_kind=type(e).__name__))
  def do_POST(self):
   try:
    role=self.path.rsplit('/',1)[1];assert self.client_address[0]=={'P':'10.44.12.175','D':'10.44.61.71'}[role];size=int(self.headers['Content-Length']);assert 0<size<=2*1024**2;v=json.loads(self.rfile.read(size))
    with condition:
     if self.path=='/blocked/'+role:latch.fail(v)
     else:
      assert self.path=='/metadata/'+role and v['role']==role and v['token']==commands[role]['token'] and (v['token'],role) not in answers
      answers[(v['token'],role)]=v;commands[role]=dict(action='WAIT',token=v['token']+'-ack')
     condition.notify_all()
    self.response(200,dict(received=True))
   except BaseException as e:self.response(400,dict(error_kind=type(e).__name__))
 server=None
 runner=OwnedRunner(R,runtime.save,latch.gate)
 def checked_child(argv,label,env,timeout):return runner.run(argv,label,env,timeout,request=label in ['short-request','full-request'])
 def phase(action,roles,token,limit=150,**params):
  with condition:
   for role in roles:commands[role]=dict(action=action,token=token,**params)
   condition.notify_all()
  runtime.save(state='WAIT_'+token.upper());end=time.monotonic()+limit
  with condition:
   while not all((token,r) in answers for r in roles):
    latch.gate()
    assert time.monotonic()<end and time.time()<runtime.deadline,'finite phase exceeded'
    condition.wait(min(20,max(.1,end-time.monotonic())))
   latch.gate();return {r:answers[(token,r)] for r in roles}
 def identity(token,include_startup=False,sources=None):
  if sources is None:sources=phase('PROBE',['P','D'],token,include_startup=include_startup)
  files={}
  for role,report in sources.items():
   raw=report['source_raw'].encode();assert hashlib.sha256(raw).hexdigest()==report['source_SHA256'];files[role]=publish(R/(token+'-'+role+'.json'),raw)
  ip=R/(token+'-identity.json');argv=[PY,'-B','-S',str(C/'c3_P_state_runtime_probe_v13.py'),'identity','--model-root',str(N),'--plan-sha256',PPLAN,'--helper-sha256',HELPER,'--source',files['P']['path'],'--source-sha256',files['P']['SHA256'],'--d-source',files['D']['path'],'--d-source-sha256',files['D']['SHA256'],'--d-plan-sha256',DPLAN,'--d-model-root',batch['d_model_root']]
  assert checked_child(argv,token+'-identity',env,20)==0,'fresh actual pair identity failed';raw=(R/(token+'-identity.stdout')).read_bytes();publish(ip,raw)
  return ip,sources
 try:
  with socket.socket() as preflight:preflight.bind(('10.44.61.73',19108))
  server=ThreadingHTTPServer(('10.44.61.73',19108),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
  ip,sources=identity('prepare-probes',True);pr=sources['P'];configraw=pr['observer_config_raw'].encode();assert hashlib.sha256(configraw).hexdigest()==pr['observer_config_SHA256']==batch['observer_config_sha256'];cfg=publish(R/'observer-config.json',configraw)
  for ref in pr['refs']:
   raw=ref['raw'].encode();assert hashlib.sha256(raw).hexdigest()==ref['SHA256'] and canonical(json.loads(raw))==raw;path=Path(ref['path']);assert N in path.parents;publish_identity_replica(path,raw) # verify and reuse immutable same-model reference
  cli=json.loads(a.prepare_argv_json.read_text())['argv'];replacements={'IDENTITY':str(ip),'IDENTITY_SHA':sha(ip),'OBSERVER_CFG':cfg['path'],'OBSERVER_CFG_SHA':cfg['SHA256'],'OUTROOT':str(R/'client'),'ARM_DEADLINE':str(min(json.loads(ip.read_text())['roles'][r]['deadline_epoch'] for r in ['P','D'])-421),'P_ROOT':batch['p_model_root'],'D_ROOT':batch['d_model_root'],'P_PLAN':batch['p_plan_sha256'],'D_PLAN':batch['d_plan_sha256'],'SHORT_NONCE':batch['short_nonce'],'FULL_NONCE':batch['full_nonce']}
  cli=[replacements.get(v[2:-1],v) if v.startswith('${') and v.endswith('}') else v for v in cli]
  assert cli[:5]==[PY,'-B','-S',str(a.client),'prepare'] and not any('${' in x for x in cli),'exact frozen client prepare argv required'
  assert checked_child(cli,'client-prepare',env,30)==0,'native CPU prepare failed'
  prepared=R/'client/prepared-config.json';receiver=R/'client/receiver-config.json';assert prepared.is_file() and receiver.is_file()
  phase('BRIDGE_CONFIG',['P'],'start-bridge',30,config_raw=receiver.read_text(),config_SHA256=sha(receiver))
  _,shortsources=identity('short-probes') # both artifact probes finish before fresh empty
  proof=phase('EMPTY_SHORT',['P'],'short-empty',145)['P'];raw=proof['fresh_proof_raw'].encode();assert hashlib.sha256(raw).hexdigest()==proof['fresh_proof_SHA256'];fp=R/'short-empty-proof.json';publish(fp,raw)
  ip,_=identity('short-identity',sources=shortsources);e=json.loads(fp.read_text());assert e['state']=='FRESH_P_ALLOCATOR_OWNER_CACHE_TREE_OBSERVED_EMPTY' and time.time()-e['observation']['latest_metric']['at']<=75
  basecli=[PY,'-B','-S',str(a.client)];short=basecli+['short','--config-json',str(prepared),'--config-sha256',sha(prepared),'--identity-json',str(ip),'--identity-sha256',sha(ip),'--fresh-proof',str(fp),'--fresh-proof-sha256',sha(fp)]
  latch.gate();publish(R/'short-request-once.claim',canonical(dict(at=time.time(),identity_SHA256=sha(ip))));runtime.save(request_dispatches=1,model_requests='UNKNOWN_AFTER_SHORT_DISPATCH')
  assert checked_child(short,'short-request',env,195)==0,'short actual request failed; no retry'
  summary=json.loads((R/'client/short/summary.json').read_text());rid=summary['actualRID'] if 'actualRID' in summary else summary['request_id']
  b=phase('SHORT_BARRIER',['P'],'short-drain-barrier',145,request_id=rid)['P']['barrier'];bp=R/'short-barrier.json';publish(bp,canonical(b))
  ip,_=identity('full-probes');assert time.time()-b['observed_at']<=45,'short barrier stale before full'
  full=basecli+['full','--config-json',str(prepared),'--config-sha256',sha(prepared),'--identity-json',str(ip),'--identity-sha256',sha(ip),'--short-barrier',str(bp),'--short-barrier-sha256',sha(bp),'--short-result-sha256',sha(R/'client/short/summary.json')]
  latch.gate();publish(R/'full-request-once.claim',canonical(dict(at=time.time(),identity_SHA256=sha(ip))));runtime.save(request_dispatches=2,model_requests='UNKNOWN_AFTER_FULL_DISPATCH')
  assert checked_child(full,'full-request',env,195)==0,'full actual request failed; no retry'
  fullsummary=json.loads((R/'client/full/summary.json').read_text());fullrid=fullsummary['request_id']
  finalbarrier=phase('FULL_BARRIER',['P'],'full-drain-barrier',145,request_id=fullrid)['P']['barrier'];publish(R/'full-barrier.json',canonical(finalbarrier))
  runtime.save(state='BOTH_ACTUAL_CLIENT_PHASES_COMPLETE_CAPTURE_REVIEW_PENDING',model_requests=2,finished_at=time.time())
 except BaseException as e:runtime.save(state='BLOCKED' if not runtime.state.get('request_dispatches') else 'FAILED',error_kind=type(e).__name__,error=str(e),finished_at=time.time());raise
 finally:
  with condition:
   for role in ['P','D']:commands[role]=dict(action='STOP' if runtime.state['state']=='BOTH_ACTUAL_CLIENT_PHASES_COMPLETE_CAPTURE_REVIEW_PENDING' else 'BLOCKED',token='terminal',reason=runtime.state.get('error'))
   condition.notify_all()
  # Leave the terminal metadata endpoint available for one final bounded peer read.
  time.sleep(3)
  if server is not None:server.shutdown();server.server_close()
  runtime.done.set();runtime.save()
 return 0
if __name__=='__main__':raise SystemExit(main())
