"""Recover already-published typed latches after ONLY the parameter-hash limit.

Original 0/1-counter loop precedes tensor-file publication and parameter hashing.
No hash budget, GPU tensor, model hook, old plan or native request is modified.
Global GPU parameter fingerprints remain BLOCKED; only core-input data are usable.
"""
import argparse,hashlib,json,os,re,socket,time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
LIMIT_ERROR={'state':'CAPTURE_CONTROL_FAILED','error':"RuntimeError('original parameter hash budget')"}
def need(ok,s):
 if not ok:raise ValueError(s)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 need(not p.exists(),'immutable recovery output');t=p.with_suffix('.partial');t.write_text(json.dumps(v,sort_keys=True,indent=2));os.link(t,p);t.unlink()
def recover(record,label):
 import torch
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only recovery')
 need(re.fullmatch('[A-Za-z0-9_-]{1,64}',label),'bounded original label')
 pid=record['PID'];proc=Path('/proc')/str(pid);s=proc.joinpath('stat').read_text().rsplit(')',1)[1].split();need(proc.stat().st_uid==record['UID']==500 and s[19]==str(record['start_ticks']) and s[0] not in ['Z','X','x'],'actual rank epoch not live')
 cfg=record['identity'];need(cfg['schema']=='history40-inner-graph-latches-v1' and cfg['role']=='D' and cfg['layers']==[25] and cfg['positions']==[7169] and cfg['gpu_budget_bytes']==4194304,'actual profile required')
 latch=C/'history38_latches_v1.py';observer=C/'history40_observer_v1.py';need(digest(latch)==cfg['latches_SHA256'] and digest(observer)==cfg['control_source_SHA256'][observer.name],'original counter-before-publication source drift')
 path=Path(cfg['output'])/(str(pid)+'-'+label+'.pt');need(path.is_file() and not path.is_symlink() and path.stat().st_uid==500 and path.stat().st_size<=8*1024**2,'original published bounded typed file required')
 need(not path.with_suffix('.partial').exists(),'incomplete original export')
 data=torch.load(path,map_location='cpu',weights_only=True);need(isinstance(data,dict) and len(data)<=192,'original object budget');size=0;hits={};geometry={}
 for key,t in data.items():
  need(re.fullmatch('D-L25-T7169-[A-Za-z0-9_]+',key) and isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.ndim==2 and t.shape[0]==1,'original typed one-row selector');size+=t.numel()*t.element_size()+8;hits[key]=[1,7169];geometry[key]=dict(shape=list(t.shape),dtype=str(t.dtype))
 need(size<=4194304 and not torch.cuda.is_initialized(),'original buffer budget and CPU-only result')
 return dict(state='RECOVERED_TYPED_LATCHES_PARAMETER_HASH_BLOCKED',PID=pid,UID=500,start_ticks=record['start_ticks'],identity=cfg,file=str(path),SHA256=digest(path),bytes=path.stat().st_size,coverage_empty=not bool(data),hits=hits,geometry=geometry,counter_invariant_proof=dict(source_SHA256=digest(latch),every_registered_counter_checked_before_tensor_publication=True,nonzero_counter_required_exactly_one=True,all_zero_counters_omitted_from_file=True),original_export_error=LIMIT_ERROR,actual_parameters_after_EOF=None,parameter_hash_status='BLOCKED_ORIGINAL128MiB_LIMIT_UNCHANGED',model_requests=0)
def snapshot(record,label):
 p=Path(record['socket']);need(p.is_socket() and p.stat().st_uid==500 and not p.stat().st_mode&0o077,'owned private socket')
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
  s.settimeout(90);s.connect(str(p));s.sendall((json.dumps(dict(operation='snapshot',label=label,request_drained=True))+'\n').encode());data=b''
  while not data.endswith(b'\n'):
   part=s.recv(65536);need(part and len(data)+len(part)<=2*1024**2,'bounded complete reply');data+=part
 result=json.loads(data);need(result==LIMIT_ERROR,'only known post-publication parameter-hash failure may be recovered: '+str(result));return recover(record,label)
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();need(os.getuid()==500 and digest(a.plan)==a.sha256 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','owned CPU recovery plan');plan=json.loads(a.plan.read_text());need(digest(__file__)==plan['script']['SHA256'],'recovery source drift');root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
 try:
  opener=build_opener(ProxyHandler({}))
  def fetch(pin):
   with opener.open('http://10.44.61.73:19091/publication-precision-20261101/control/'+str(Path(pin['path']).relative_to(C)),timeout=15) as r:raw=r.read(8*1024**2+1)
   need(len(raw)<=8*1024**2 and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'immutable source evidence drift');return json.loads(raw)
  n=fetch(plan['native']);need(n['response_eof_complete'] and len(n['output_ids'])==167,'actual completed D request required');probe=fetch(plan['probe']);role=n['identity']['roles']['D'];need(probe['plan_SHA256']==role['plan_SHA256'] and probe['model_root']['PID']==role['model_root']['PID'],'native/probe epoch binding')
  rows=[]
  for item in probe['history_capture_controls']:
   rec=item['record'];need(rec is not None,'allD graph controls required');label='capture-'+str(n['request_id']);path=Path(rec['identity']['output'])/(str(rec['PID'])+'-'+label+'.pt')
   row=recover(rec,label) if path.exists() else snapshot(rec,label)
   if not row['coverage_empty']:
    import torch
    data=torch.load(row['file'],map_location='cpu',weights_only=True);need(data['D-L25-T7169-actual_input_token'].item()==n['output_ids'][0] and data['D-L25-T7169-core_positions'].item()==7169,'actual GPU token/position/RID join')
   rows.append(row);save(root/(str(rec['PID'])+'-recovered.json'),row)
  need(len(rows)==4 and sum(not r['coverage_empty'] for r in rows)==1,'four rank/one actual DP owner required');out=dict(state='RECOVERED_TYPED_CORE_SAMPLES_PARAMETER_HASH_BLOCKED',native=plan['native'],rank_controls=rows,model_requests=0,parameter_hashes_verified=False,original_stage_still_FAILED=True);save(root/'result-safe.json',out);save(root/'status.json',dict(state='SUCCEEDED_TYPED_RECOVERY_ONLY',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=digest(root/'result-safe.json'))));print(json.dumps(dict(state=out['state'],ranks=4,model_requests=0)))
 except BaseException as e:save(root/'status.json',dict(state='BLOCKED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
