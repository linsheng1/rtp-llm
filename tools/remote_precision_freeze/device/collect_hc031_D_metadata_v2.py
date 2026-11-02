"""Posthoc native metadata only; one seed-only HTTP fetch, no model request."""
import argparse,hashlib,json,os,re,socket,time
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
from urllib.request import urlopen
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
SHA='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa'
NONCE='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',type=Path,required=True);a=p.parse_args();assert os.getuid()==500 and socket.gethostbyname(socket.gethostname())=='10.44.61.71' and a.root.parent==C and not a.root.exists();a.root.mkdir(mode=0o700)
state={'state':'COLLECTING_POSTHOC_NATIVE_METADATA','PID':os.getpid(),'UID':500,'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'at':time.time(),'model_requests':0}
def save():
 state['heartbeat_at']=time.time();q=a.root/'status.json.partial';q.write_text(json.dumps(state,indent=2));q.replace(a.root/'status.json')
save()
try:
 with urlopen('http://10.44.61.73:19091/publication-precision-20261101/control/hc031-source-source1/request/native/result.json',timeout=15) as response:raw=response.read(1024**2+1)
 assert len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==SHA
 source=json.loads(raw);assert source['state']=='COMPLETE_NATIVE_RPC' and source['response_eof_complete'] is True
 root=C/'airline4-D-hcfix-source1';s=json.loads((root/'model-service-status.json').read_bytes());assert s['model_READY'] and s['state']=='MODEL_READY_KV_EMPTY_PENDING'
 expected=source['identity']['roles']['D'];epochs=lambda rows:{(v['PID'],v['UID'],str(v['start_ticks'])) for v in rows}
 assert epochs([s['model_root']]+s['GPU_ranks'])==epochs([expected['model_root']]+expected['GPU_ranks'])
 for pid,uid,start in epochs([s['model_root']]+s['GPU_ranks']):
  q=Path('/proc')/str(pid);f=(q/'stat').read_text().rsplit(')',1)[1].split();assert q.stat().st_uid==uid==500 and f[19]==start and f[0] not in ['Z','X']
 lines=[];logs=[]
 for path in [root/'model-stage.log',root/'runtime/logs/engine.log']:
  if not path.is_file():continue
  with path.open('rb') as f:
   st=os.fstat(f.fileno());assert st.st_size<=64*1024**2;data=f.read(st.st_size);assert len(data)==st.st_size
  logs.append({'path':str(path),'device':st.st_dev,'inode':st.st_ino,'captured_prefix_bytes':len(data),'captured_prefix_SHA256':hashlib.sha256(data).hexdigest()})
  for line in data.decode(errors='replace').splitlines():
   if re.search(r'DTOP_GPU (retained_plan computed=|committed computed_end=)',line):lines.append({'path':str(path),'line':line})
 assert lines,'no actual retained plan/commit lines'
 proof={'namespace':source['namespace'],'source_request_id':source['request_id'],'source_input_ids_sha256':digest(source['input_ids']),'source_output_ids_sha256':digest(source['output_ids']),'lines':lines,'log_prefixes':logs,'provenance_verified':False,'source_process_provenance_verified':False,'scope':'POSTHOC_NATIVE_PID_AND_MANAGER_METADATA_ONLY_NO_PRE_REQUEST_CURSOR','exact_request_provenance':'PENDING','source':{'path':'/home/admin/linshuo.zjl/publication-precision-20261101/control/hc031-source-source1/request/native/result.json','SHA256':SHA},'at':time.time()}
 raw=(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode();assert len(raw)<=65536;(a.root/'native-metadata.json').write_bytes(raw)
 deadline=time.time()+110;state.update(state='READY_ONE_SEED_METADATA_FETCH',proof_SHA256=hashlib.sha256(raw).hexdigest(),URL='http://10.44.61.71:19111/'+NONCE,deadline_epoch=deadline,successful_fetches=0);save()
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   if self.client_address[0]!='10.44.61.73' or self.path!='/'+NONCE or state['successful_fetches']!=0:self.send_error(403);return
   self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);self.wfile.flush();state.update(state='FETCHED_ONCE',successful_fetches=1,returncode=0)
 with HTTPServer(('10.44.61.71',19111),Handler) as server:
  while state['successful_fetches']==0:
   assert time.time()<deadline,'finite110s metadata fetch expired';server.timeout=min(20,deadline-time.time());server.handle_request();save()
 state.update(finished_at=time.time());save()
except BaseException as e:state.update(state='BLOCKED',error_kind=type(e).__name__,error=str(e)[:1500],finished_at=time.time());save();raise
print(json.dumps(state))
