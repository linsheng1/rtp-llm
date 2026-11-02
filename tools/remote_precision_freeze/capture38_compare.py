"""CPU-only comparison of two already completed same-D-epoch trajectory captures.

All IDs/tensors fetched device-direct; only equality/count/coordinate metadata exported.
No RPC, no CUDA, no whole-history-KV equality inference.
"""
import argparse,hashlib,json,os,re,signal,time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,why):
 if not ok:raise ValueError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();raw=a.plan.read_bytes();need(sha(raw)==a.sha256 and os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only actual plan required');plan=json.loads(raw);need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'comparator drift')
 root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
 def save(n,v):
  p=root/(n+'.partial');p.write_text(json.dumps(v,sort_keys=True,indent=2));p.replace(root/n)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite CPU trajectory join')));signal.alarm(120)
 save('status.json',dict(state='VERIFYING_COMPLETED_CAPTURES',model_requests=0,at=time.time()))
 try:
  opener=build_opener(ProxyHandler({}))
  def fetch(pin,name,maximum=8*1024**2):
   path=Path(pin['path']);need(C in path.parents,'owned source only');url='http://10.44.61.73:19091/publication-precision-20261101/control/'+str(path.relative_to(C))
   with opener.open(url,timeout=15) as response:data=response.read(maximum+1)
   need(len(data)<=maximum and sha(data)==pin['SHA256'],'actual immutable data hash differs')
   dest=root/name
   with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   return dest
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU-only check')
  native={};points={};coverage={}
  for case,count in [('source',167),('normal',176)]:
   n=json.loads(fetch(plan[case+'_native'],case+'-native-private.json').read_text());native[case]=n
   need(n['response_eof_complete'] and len(n['input_ids'])==7169 and len(n['output_ids'])==count and n['input_ids_sha256']==plan['R_SHA256'],'actual EOF/geometry differs')
   ackpin=plan[case+'_ack'];ack=json.loads(fetch(ackpin,case+'-ack.json').read_text());epoch=n['identity']['roles']['D'];ranks={r['PID']:r for r in epoch['GPU_ranks']}
   need(len(ranks)==4 and {r['world_rank'] for r in ranks.values()}==set(range(4)) and len(ack['rank_controls'])==4 and {r['PID'] for r in ack['rank_controls']}==set(ranks),'all4rank drain proofs required')
   need(all(str(ack['model_root'][k])==str(epoch['model_root'][k]) for k in ['PID','UID','start_ticks']),'EOF/export root epoch differs')
   points[case]={};coverage[case]=[]
   for record in ack['rank_controls']:
    cfg=record['identity'];pid=record['PID'];need(cfg['schema']=='history38-inner-graph-latches-v1' and cfg['source_commit']==plan['candidate_commit'] and cfg['role']=='D' and cfg['layers']==list(range(40)) and cfg['positions']==[7169,7200] and cfg['gpu_budget_bytes']==4194304,'actual profile drift')
    path=Path(ackpin['path']).parent/'capture-bytes/D'/(str(pid)+'.pt');local=fetch(dict(path=str(path),SHA256=record['SHA256']),case+'-'+str(pid)+'.pt');need(local.stat().st_size==record['bytes'],'capture size differs')
    data=torch.load(local,map_location='cpu',weights_only=True);need(bool(data)==(not record['coverage_empty']) and set(data)==set(record['hits']),'actual hit coverage differs');need(len(record['all_registered_hit_counters'])<=192 and all(v[0] in [0,1] for v in record['all_registered_hit_counters'].values()),'duplicate execution')
    coverage[case].append(dict(PID=pid,world_rank=ranks[pid]['world_rank'],empty=record['coverage_empty'],points=len(data),SHA256=record['SHA256']))
    for key,t in data.items():
     m=re.fullmatch(r'D-L(\d+)-T(7169|7200)-(block_entry4HC|actual_input_token)',key);need(m is not None,'unexpected sample key');layer,pos,label=int(m[1]),int(m[2]),m[3];need(record['hits'][key]==[1,pos] and t.device.type=='cpu' and t.shape[0]==1,'actual GPU position/row shape differs');coord=(layer,pos,label);need(coord not in points[case],'multiple DP owners for one row');points[case][coord]=t
   expected={(l,pos,'block_entry4HC') for l in range(40) for pos in [7169,7200]}|{(0,pos,'actual_input_token') for pos in [7169,7200]};need(set(points[case])==expected,'all40 blocks/two positions missing')
   for pos,index in [(7169,0),(7200,31)]:need(points[case][0,pos,'actual_input_token'].item()==n['output_ids'][index],'expected position formula not corroborated by actual GPU token')
  x,y=native['source'],native['normal'];need(x['input_ids']==y['input_ids'] and x['actual_native_request_profile']==y['actual_native_request_profile'],'same real input/profile required');dx,dy=x['identity']['roles']['D'],y['identity']['roles']['D']
  def epoch(row):return tuple(str(row[k]) for k in ['PID','UID','start_ticks'])
  need(epoch(dx['model_root'])==epoch(dy['model_root']) and sorted(epoch(r) for r in dx['GPU_ranks'])==sorted(epoch(r) for r in dy['GPU_ranks']) and all(dx[k]==dy[k] for k in ['source_commit','env_SHA256','manifest_SHA256','plan_SHA256','namespace','model_service_helper_SHA256']),'same retained D epoch/artifacts required')
  difference=next((i for i,(a,b) in enumerate(zip(x['output_ids'],y['output_ids'])) if a!=b),None);need(difference==32 and x['output_ids'][:32]==y['output_ids'][:32],'original divergence must be reproduced')
  checks=[]
  for pos in [7169,7200]:
   need(torch.equal(points['source'][0,pos,'actual_input_token'],points['normal'][0,pos,'actual_input_token']),'sampled decode inputs differ')
   for layer in range(40):
    a,b=points['source'][layer,pos,'block_entry4HC'],points['normal'][layer,pos,'block_entry4HC'];need(a.shape==b.shape==(1,20480) and a.dtype==b.dtype==torch.bfloat16,'real4HC shape/dtype required')
    byte_equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8));mask=a!=b;indices=mask.flatten().nonzero().flatten();checks.append(dict(position=pos,layer=layer,byte_equal=byte_equal,elements=a.numel(),numeric_unequal_elements=int(mask.sum()),first_difference_column=int(indices[0]) if indices.numel() else None,max_abs_difference=float((a.float()-b.float()).abs().max())))
  earliest={str(pos):next((v for v in checks if v['position']==pos and not v['byte_equal']),None) for pos in [7169,7200]}
  result=dict(state='COMPARED_COMPLETED_D_TRAJECTORIES',candidate_commit=plan['candidate_commit'],model_requests=0,same_D_epoch=True,actual_GPU_input_tokens_joined=True,output_token_first_difference_one_based=33,checks=checks,earliest_sampled_difference=earliest,coverage=coverage,whole_history_KV_equal_proven=False,formal_acceptance=False,raw_values_exported=False)
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes())),model_requests=0));print(json.dumps(dict(state='SUCCEEDED',earliest=earliest,model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
