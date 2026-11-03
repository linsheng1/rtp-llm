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
   if path.exists():
    need(path.is_file() and not path.is_symlink() and path.stat().st_uid==500,'owned local capture');data=path.read_bytes()
   else:
    with opener.open(url,timeout=15) as response:data=response.read(maximum+1)
   need(len(data)<=maximum and sha(data)==pin['SHA256'],'actual immutable data hash differs')
   dest=root/name
   with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   return dest
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU-only check')
  native={};points={};coverage={};parameters={}
  for case,count in [('source',167),('normal',176)]:
   n=json.loads(fetch(plan[case+'_native'],case+'-native-private.json').read_text());native[case]=n
   need(n['response_eof_complete'] and len(n['input_ids'])==7169 and len(n['output_ids'])==count and n['input_ids_sha256']==plan['R_SHA256'],'actual EOF/geometry differs')
   ackpin=plan[case+'_ack'];ack=json.loads(fetch(ackpin,case+'-ack.json').read_text());epoch=n['identity']['roles']['D'];ranks={r['PID']:r for r in epoch['GPU_ranks']}
   need(len(ranks)==4 and {r['world_rank'] for r in ranks.values()}==set(range(4)) and len(ack['rank_controls'])==4 and {r['PID'] for r in ack['rank_controls']}==set(ranks),'all4rank drain proofs required')
   if case=='source':
    need(ack['state']=='RECOVERED_TYPED_CORE_SAMPLES_PARAMETER_HASH_BLOCKED' and ack['native']==plan['source_native'] and ack['original_stage_still_FAILED'] is True,'completed D typed recovery source required')
   else:need(all(str(ack['model_root'][k])==str(epoch['model_root'][k]) for k in ['PID','UID','start_ticks']),'EOF/export root epoch differs')
   points[case]={};coverage[case]=[];parameters[case]={}
   for record in ack['rank_controls']:
    cfg=record['identity'];pid=record['PID'];need(cfg['schema']=='history40-inner-graph-latches-v1' and cfg['source_commit']==plan['candidate_commit'] and cfg['role']=='D' and cfg['layers']==[25] and cfg['positions']==[7169] and cfg['gpu_budget_bytes']==4194304,'actual profile drift')
    need(str(record.get('start_ticks',ranks[pid]['start_ticks']))==str(ranks[pid]['start_ticks']) and record.get('UID',500)==ranks[pid]['UID']==500,'actual typed rank epoch differs')
    proof=record.get('counter_invariant_proof');need(proof is not None and proof['every_registered_counter_checked_before_tensor_publication'] and proof['nonzero_counter_required_exactly_one'] and proof['all_zero_counters_omitted_from_file'] and proof['source_SHA256']==cfg['latches_SHA256'],'original unchanged counter-loop publication proof required')
    need(record['parameter_hash_status']=='BLOCKED_ORIGINAL128MiB_LIMIT_UNCHANGED' and record['actual_parameters_after_EOF'] is None,'GPU parameter fingerprints remain explicitly blocked')
    path=Path(record['file']);local=fetch(dict(path=str(path),SHA256=record['SHA256']),case+'-'+str(pid)+'.pt');need(local.stat().st_size==record['bytes'],'capture size differs')
    data=torch.load(local,map_location='cpu',weights_only=True);need(bool(data)==(not record['coverage_empty']) and set(data)==set(record['hits']),'actual hit coverage differs');need(len(data)<=192 and all(v==[1,7169] for v in record['hits'].values()) and sum(t.numel()*t.element_size()+8 for t in data.values())<=4194304,'original object/0-1/buffer bounds')
    parameters[case][pid]=record['actual_parameters_after_EOF'];coverage[case].append(dict(PID=pid,world_rank=ranks[pid]['world_rank'],empty=record['coverage_empty'],points=len(data),SHA256=record['SHA256']))
    for key,t in data.items():
     m=re.fullmatch(r'D-L(25)-T(7169)-(.+)',key);need(m is not None,'unexpected sample key');layer,pos,label=int(m[1]),int(m[2]),m[3];need(record['hits'][key]==[1,pos] and t.device.type=='cpu' and t.shape[0]==1,'actual GPU position/row shape differs');coord=(layer,pos,label);need(coord not in points[case],'multiple DP owners for one row');points[case][coord]=t
   required={'actual_input_token','actual_attention_input5120','Q_before_core','core_output','core_scales','SWA_raw','SWA_logical_indices','SWA_valid','SWA_mapping_valid','GLOBAL_raw','GLOBAL_logical_indices','GLOBAL_valid','GLOBAL_mapping_valid','core_positions','core_branch','sink','cos_sin_position_row','pre_AR5120','post_AR5120'}
   need(required<={key[2] for key in points[case]},'required real mega Q/core/readset samples missing; branch not assumed')
   need(points[case][25,7169,'actual_input_token'].item()==n['output_ids'][0],'actual GPU token not joined')
   need(points[case][25,7169,'core_positions'].item()==7169 and points[case][25,7169,'core_branch'].item()==1,'actual mega/position not corroborated')
   for name in ['SWA','GLOBAL']:need(points[case][25,7169,name+'_mapping_valid'].all().item(),'positive physical slot lacks unique logical mapping')
  x,y=native['source'],native['normal'];need(x['input_ids']==y['input_ids'] and x['actual_native_request_profile']==y['actual_native_request_profile'],'same real input/profile required');dx,dy=x['identity']['roles']['D'],y['identity']['roles']['D']
  def epoch(row):return tuple(str(row[k]) for k in ['PID','UID','start_ticks'])
  same_epoch=epoch(dx['model_root'])==epoch(dy['model_root']) and sorted(epoch(r) for r in dx['GPU_ranks'])==sorted(epoch(r) for r in dy['GPU_ranks'])
  need(all(dx[k]==dy[k] for k in ['source_commit','env_SHA256','manifest_SHA256','namespace','model_service_helper_SHA256']),'same actual runtime/artifacts required across explicitly recorded epochs')
  difference=next((i for i,(a,b) in enumerate(zip(x['output_ids'],y['output_ids'])) if a!=b),None);need(difference==32 and x['output_ids'][:32]==y['output_ids'][:32],'original divergence must be reproduced')
  need(all(v is None for case in parameters.values() for v in case.values()),'projection fingerprint limitation must be explicit; not accepted as equal hashes')
  checks=[]
  need(set(points['source'])==set(points['normal']),'natural branch sample presence differs')
  for key in sorted(points['source']):
   a,b=points['source'][key],points['normal'][key];need(a.shape==b.shape and a.dtype==b.dtype,'same natural geometry required');mask=a!=b;indices=mask.flatten().nonzero().flatten();checks.append(dict(layer=key[0],position=key[1],point=key[2],byte_equal=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),elements=a.numel(),dtype=str(a.dtype),numeric_unequal_elements=int(mask.sum()),first_difference_column=int(indices[0]) if indices.numel() else None,max_abs_difference=None if a.dtype in [torch.uint8,torch.bool] else float((a.float()-b.float()).abs().max())))
  readsets={}
  for name,width in [('SWA',528),('GLOBAL',288)]:
   key=lambda suffix:(25,7169,name+'_'+suffix)
   sa,sb=points['source'][key('logical_indices')].flatten(),points['normal'][key('logical_indices')].flatten();va,vb=points['source'][key('valid')].flatten(),points['normal'][key('valid')].flatten();ra,rb=points['source'][key('raw')].reshape(-1,width),points['normal'][key('raw')].reshape(-1,width)
   same_order=torch.equal(sa,sb) and torch.equal(va,vb);need(ra.shape[0]==sa.numel() and rb.shape[0]==sb.numel(),'raw readset geometry');bagA=sorted(sa[va].tolist());bagB=sorted(sb[vb].tolist());shared=set(bagA)&set(bagB);changed=[]
   for address in sorted(shared):
    ai=(va&(sa==address)).nonzero().flatten();bi=(vb&(sb==address)).nonzero().flatten();need(ai.numel()==bi.numel()==1,'selected logical row duplicated; comparison blocked');u,v=ra[ai[0]],rb[bi[0]]
    if not torch.equal(u,v):changed.append(dict(logical_row=int(address),unequal_bytes=int((u!=v).sum())))
   readsets[name]=dict(selected_width=sa.numel(),valid_rows_Drestore=int(va.sum()),valid_rows_normal=int(vb.sum()),same_order_and_mask=same_order,same_logical_bag=bagA==bagB,shared_logical_rows=len(shared),shared_rows_with_different_bytes=len(changed),first_changed_rows=changed[:8],raw_row_format_bytes=width)
  result=dict(state='COMPARED_COMPLETED_L25_Q_CORE_READSETS',candidate_commit=plan['candidate_commit'],model_requests=0,same_D_epoch=same_epoch,actual_GPU_input_token_position_joined=True,output_first_difference_one_based=33,checks=checks,readsets=readsets,coverage=coverage,GPU_projection_parameter_hashes_verified=False,cross_epoch_causal_gate_required=not same_epoch,core_counterfactual_reproduction_required=True,source_remote_destination_byte_equality_proven=False,whole_model_fixed=False,formal_acceptance=False,raw_values_exported=False,interpretation='typed core data only; original parameter-hash failure remainsBLOCKED; do not claim operator root before reproducing both core outputs on one GPU')
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes())),model_requests=0));print(json.dumps(dict(state='SUCCEEDED',readsets=readsets,model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
