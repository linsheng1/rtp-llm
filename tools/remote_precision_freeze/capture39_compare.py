"""Join already completed P entries and D prepared inputs; no RPC/GPU replay."""
import argparse,hashlib,json,os,re,signal,time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,why):
 if not ok:raise ValueError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();raw=a.plan.read_bytes();need(sha(raw)==a.sha256 and os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','actual CPU plan required');plan=json.loads(raw);need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'analysis source drift');root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new analysis root');root.mkdir(mode=0o700)
 def save(n,v):
  p=root/(n+'.partial');p.write_text(json.dumps(v,sort_keys=True,indent=2));p.replace(root/n)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite CPU input join')));signal.alarm(120);save('status.json',dict(state='JOINING_COMPLETED_CAPTURES',model_requests=0))
 try:
  opener=build_opener(ProxyHandler({}))
  def fetch(pin,name,maximum=8*1024**2):
   p=Path(pin['path']);need(C in p.parents,'owned capture only')
   with opener.open('http://10.44.61.73:19091/publication-precision-20261101/control/'+str(p.relative_to(C)),timeout=15) as r:b=r.read(maximum+1)
   need(len(b)<=maximum and sha(b)==pin['SHA256'],'actual capture hash differs');out=root/name
   with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
   return out
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU-only analysis')
  native={};points={};coverage={};params={};metadata={}
  for case,count in [('source',167),('normal',176)]:
   n=json.loads(fetch(plan[case+'_native'],case+'-native-private.json').read_text());native[case]=n;need(n['response_eof_complete'] and len(n['input_ids'])==7169 and len(n['output_ids'])==count and n['input_ids_sha256']==plan['R_SHA256'],'actual native calibration')
   points[case]={};coverage[case]={};params[case]={};metadata[case]={}
   for role in ['P','D']:
    ackpin=plan[case+'_ack_'+role];ack=json.loads(fetch(ackpin,case+'-'+role+'-ack.json').read_text());epoch=n['identity']['roles'][role];ranks={r['PID']:r for r in epoch['GPU_ranks']};need(len(ranks)==len(ack['rank_controls'])==4 and {r['PID'] for r in ack['rank_controls']}==set(ranks),'complete4rank role export required')
    need(all(str(ack['model_root'][k])==str(epoch['model_root'][k]) for k in ['PID','UID','start_ticks']),'actual EOF/role root differs');coverage[case][role]=[];points[case][role]={};params[case][role]={};metadata[case][role]={}
    for record in ack['rank_controls']:
     cfg=record['identity'];pos=7168 if role=='P' else 7169;layers=list(range(40 if role=='P' else 27));need(cfg['schema']=='history39-inner-graph-latches-v1' and cfg['source_commit']==plan['candidate_commit'] and cfg['role']==role and cfg['layers']==layers and cfg['positions']==[pos] and cfg['gpu_budget_bytes']==4194304,'actual selected role profile differs')
     pid=record['PID'];pt=Path(ackpin['path']).parent/'capture-bytes'/role/(str(pid)+'.pt');tensors=torch.load(fetch(dict(path=str(pt),SHA256=record['SHA256']),case+'-'+role+'-'+str(pid)+'.pt'),map_location='cpu',weights_only=True);need(bool(tensors)==(not record['coverage_empty']) and set(tensors)==set(record['hits']),'typed hit coverage differs');need(len(record['all_registered_hit_counters'])<=192 and all(v[0] in [0,1] for v in record['all_registered_hit_counters'].values()),'duplicate capture')
     params[case][role][pid]=record['actual_parameters_after_EOF'];coverage[case][role].append(dict(PID=pid,empty=record['coverage_empty'],points=len(tensors),SHA256=record['SHA256']))
     for key,t in tensors.items():
      m=re.fullmatch(role+r'-L(\d+)-T'+str(pos)+r'-(.+)',key);need(m is not None and int(m[1]) in layers and record['hits'][key]==[1,pos] and t.device.type=='cpu' and t.shape[0]==1,'actual GPU role/position differs');coordinate=(int(m[1]),m[2]);need(coordinate not in points[case][role],'duplicate CP/DP row owner');points[case][role][coordinate]=t;metadata[case][role][coordinate]=record['metadata'][key]
    for layer in layers:
     need((layer,'block_entry4HC') in points[case][role],'layer entry missing')
     if role=='D':
      need((layer,'actual_attention_input5120') in points[case][role] and (layer,'prepared_attn_present') in points[case][role],'actual D attention/prepared input missing')
      prepared=points[case][role][layer,'prepared_attn_present'].item();need(prepared in [0,1],'actual route flag')
      for key in ['prepared_normalized_input5120','prepared_HC_coeff24']:need(((layer,key) in points[case][role])==bool(prepared),'prepared tuple presence differs')
    token=points[case][role].get((0,'actual_input_token'));need(token is not None and token.numel()==1 and token.item()==(n['input_ids'][-1] if role=='P' else n['output_ids'][0]),'actual GPU input token not joined with native')
  x,y=native['source'],native['normal'];need(x['input_ids']==y['input_ids'] and x['output_ids'][:32]==y['output_ids'][:32] and x['output_ids'][32]!=y['output_ids'][32] and x['actual_native_request_profile']==y['actual_native_request_profile'],'original native input/profile/divergence must recur')
  def epoch(row):return tuple(str(row[k]) for k in ['PID','UID','start_ticks'])
  for role in ['P','D']:
   rx,ry=x['identity']['roles'][role],y['identity']['roles'][role];need(epoch(rx['model_root'])==epoch(ry['model_root']) and sorted(epoch(r) for r in rx['GPU_ranks'])==sorted(epoch(r) for r in ry['GPU_ranks']) and all(rx[k]==ry[k] for k in ['plan_SHA256','env_SHA256','manifest_SHA256','model_service_helper_SHA256']),'role epoch/artifact changed')
   need(params['source'][role]==params['normal'][role],'sampled actual parameter bytes changed')
  checks=[]
  for role in ['P','D']:
   keys=set(points['source'][role])|set(points['normal'][role])
   for layer,label in sorted(keys):
    t,u=points['source'][role].get((layer,label)),points['normal'][role].get((layer,label));need(t is not None and u is not None and t.shape==u.shape and t.dtype==u.dtype,'scene sample geometry or route differs')
    neq=t!=u;indices=neq.flatten().nonzero().flatten();checks.append(dict(role=role,position=7168 if role=='P' else 7169,layer=layer,point=label,elements=t.numel(),dtype=str(t.dtype),original_dtype=metadata['source'][role][layer,label]['original_dtype'],quant_buffer_may_include_padding='shared_input_quant' in label,byte_equal=torch.equal(t.contiguous().view(torch.uint8),u.contiguous().view(torch.uint8)),numeric_unequal_elements=int(neq.sum()),first_difference_column=int(indices[0]) if indices.numel() else None,max_abs_difference=None if t.dtype==torch.uint8 else float((t.float()-u.float()).abs().max())))
  Pfirst=next((v for v in checks if v['role']=='P' and v['point']=='block_entry4HC' and not v['byte_equal']),None)
  Dinputs=[v for v in checks if v['role']=='D' and v['point'] in ['block_entry4HC','prepared_HC_coeff24','prepared_normalized_input5120','actual_attention_input5120'] and not v['byte_equal']]
  result=dict(state='COMPARED_ACTUAL_PREPARED_INPUTS',candidate_commit=plan['candidate_commit'],model_requests=0,actual_GPU_tokens_joined=True,same_P_D_epochs=True,P_first_sampled_entry_difference=Pfirst,D_unequal_inputs=Dinputs,L25_unequal_points=[v for v in checks if v['role']=='D' and v['layer']==25 and not v['byte_equal']],checks=checks,coverage=coverage,sampled_parameter_bytes_unchanged=True,all_expert_rows_equal_proven=False,whole_history_KV_equal_proven=False,formal_acceptance=False,raw_values_exported=False)
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes())),model_requests=0));print(json.dumps(dict(state='SUCCEEDED',P_first=Pfirst,D_first=Dinputs[:1],model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
