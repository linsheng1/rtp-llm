"""Join historical7045 source D and normal P samples; no RPC/GPU/model change."""
import argparse,hashlib,json,os,re,signal
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();raw=a.plan.read_bytes();need(sha(raw)==a.sha256 and os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','actual CPU plan');plan=json.loads(raw);need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'actual comparator');root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
 def save(n,x):
  p=root/(n+'.partial');p.write_text(json.dumps(x,indent=2,sort_keys=True));p.replace(root/n)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite historical CPU join')));signal.alarm(120)
 try:
  opener=build_opener(ProxyHandler({}))
  def fetch(pin,name):
   p=Path(pin['path']);need(C in p.parents,'owned data path')
   if p.exists():need(not p.is_symlink() and p.stat().st_uid==500,'owned regular data');raw=p.read_bytes()
   else:
    with opener.open('http://10.44.61.73:19091/publication-precision-20261101/control/'+str(p.relative_to(C)),timeout=20) as r:raw=r.read(8*1024**2+1)
   need(len(raw)<=8*1024**2 and sha(raw)==pin['SHA256'],'exact bounded input bytes');out=root/name
   with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
   return out
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU only');native={};samples={};coverage={};params={}
  for case,nout in [('source',295),('normal',176)]:
   n=json.loads(fetch(plan[case+'_native'],case+'-native-private.json').read_text());native[case]=n;need(n['response_eof_complete'] and len(n['output_ids'])==nout,'natural calibrated native EOF');samples[case]={};coverage[case]={};params[case]={}
   for role in ['P','D']:
    ap=plan[case+'_ack_'+role];ack=json.loads(fetch(ap,case+'-'+role+'-ack.json').read_text());epoch=n['identity']['roles'][role];ranks={r['PID']:r for r in epoch['GPU_ranks']};need(len(ranks)==4 and {r['PID'] for r in ack['rank_controls']}==set(ranks),'all4rank export');need(all(str(ack['model_root'][k])==str(epoch['model_root'][k]) for k in ['UID','PID','start_ticks']),'same native/export model epoch');points={};coverage[case][role]=[];params[case][role]={}
    for rec in ack['rank_controls']:
     cfg=rec['identity'];pid=rec['PID'];need(cfg['schema']=='history44-inner-graph-latches-v1' and cfg['source_commit']==plan['candidate_commit'] and cfg['role']==role and cfg['positions']==[7045] and cfg['layers']==[24] and cfg['gpu_budget_bytes']==4194304 and cfg['indexer_extension']['revision']==45 and cfg['indexer_extension']['causal_prefix']==7046,'actual profile');need(all(v[0] in [0,1] for v in rec['all_registered_hit_counters'].values()) and len(rec['all_registered_hit_counters'])<=192,'actual0/1 counters')
     f=fetch(dict(path=str(Path(ap['path']).parent/'capture-bytes'/role/(str(pid)+'.pt')),SHA256=rec['SHA256']),case+'-'+role+'-'+str(pid)+'.pt');t=torch.load(f,map_location='cpu',weights_only=True);need(set(t)==set(rec['hits']) and bool(t)==(not rec['coverage_empty']),'actual typed hit coverage');need(sum(v.numel()*v.element_size()+8 for v in t.values())<=4194304,'original buffer bound');coverage[case][role].append(dict(PID=pid,world_rank=ranks[pid]['world_rank'],empty=not bool(t),points=len(t),SHA256=rec['SHA256']));params[case][role][pid]=rec['actual_parameters_after_EOF']
     for k,v in t.items():
      m=re.fullmatch(role+r'-L(\d+)-T7045-(.+)',k);need(m is not None and rec['hits'][k]==[1,7045] and v.device.type=='cpu' and v.shape[0]==1,'GPU position and unique sample');key=(int(m[1]),m[2]);need(key not in points,'one rank owner');points[key]=v
    samples[case][role]=points
  s,n=native['source'],native['normal'];sequence=s['input_ids']+s['output_ids'][:-1];need(len(s['input_ids'])==7041 and len(n['input_ids'])==7169 and n['input_ids']==sequence[:7169],'actual G+Y forcedR join');need(s['actual_native_request_profile']==n['actual_native_request_profile'],'actual generation profile')
  for role in ['P','D']:
   x,y=s['identity']['roles'][role],n['identity']['roles'][role];epoch=lambda r:tuple(str(r[k]) for k in ['UID','PID','start_ticks']);need(epoch(x['model_root'])==epoch(y['model_root']) and sorted(epoch(r) for r in x['GPU_ranks'])==sorted(epoch(r) for r in y['GPU_ranks']) and all(x[k]==y[k] for k in ['source_commit','env_SHA256','manifest_SHA256','plan_SHA256','namespace','model_service_helper_SHA256']),'same retained role epochs/runtime')
  need(not samples['source']['P'] and not samples['normal']['D'],'sourceG excludes7045 and normalD startsafter7168; otherroles must be empty');left,right=samples['source']['D'],samples['normal']['P'];need(left[24,'actual_input_token'].item()==right[24,'actual_input_token'].item()==n['input_ids'][7045]==s['output_ids'][4],'actual GPU token7045 joined to nativeIDs')
  # Actual indexer inputs/params precede the selector. No assumptions about
  # unrecorded rows, query batches, expert routes or hidden-state histories.
  pa={};pb={}
  for group,out in [(params['source']['D'],pa),(params['normal']['P'],pb)]:
   for perrank in group.values():
    for k,v in perrank.items():
     if k in out:need(out[k]==v,'same-role parameter replicas differ')
     out[k]=v
  shared=set(pa)&set(pb);need(shared,'actual parameter fingerprints present')
  need(sha(Path(plan['comparator']['path']).read_bytes())==plan['comparator']['SHA256'],'typed comparator identity')
  from capture44_chain_contract import compare_exact
  get=lambda side,k:side[24,k]
  prior_checks=[]
  for case,role,points in [('source','D',left),('normal','P',right)]:
   old=torch.load(fetch(plan['prior_samples'][case],case+'-prior-capture44.pt'),map_location='cpu',weights_only=True)
   for label in ['block_entry4HC','actual_attention_input5120','Q_before_core','core_output','core_scales','attention_output5120']:
    report=compare_exact(points[24,label],old[role+'-L24-T7045-'+label]);report.update(case=case,point=label);prior_checks.append(report)
  required=['indexer_q_payload','indexer_q_packed_sf','indexer_scaled_head_weights','indexer_visible','indexer_position_in_actual_block','indexer_key_valid','indexer_logits','indexer_selected_logical']+['indexer_key_raw_chunk'+str(i) for i in range(4)]
  need({(24,k) for k in required}<=set(left) and {(24,k) for k in required}<=set(right),'complete actual dense/paged indexer evidence required')
  checks=[]
  for label in ['actual_attention_input5120','indexer_raw_query_projection','indexer_q_payload','indexer_q_packed_sf','indexer_scaled_head_weights','indexer_visible','indexer_key_valid','indexer_logits']:
   if (24,label) not in left or (24,label) not in right:
    checks.append(dict(point=label,comparable=False,source_present=(24,label) in left,normal_present=(24,label) in right));continue
   x,y=get(left,label),get(right,label)
   if x.dtype!=y.dtype or x.shape!=y.shape:
    checks.append(dict(point=label,comparable=False,source_dtype=str(x.dtype),normal_dtype=str(y.dtype),source_shape=list(x.shape),normal_shape=list(y.shape)));continue
   report=compare_exact(x,y);report.update(point=label,comparable=True);checks.append(report)
  for points in [left,right]:need(get(points,'indexer_visible').item()==7046 and get(points,'indexer_position_in_actual_block').all(),'actual target7045 causal extent')
  raw={};decoded={};valid={};lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6])
  for case,side in [('source',left),('normal',right)]:
   raw[case]=torch.cat([get(side,'indexer_key_raw_chunk'+str(i)).reshape(-1,68) for i in range(4)]);need(raw[case].shape==(7046,68),'all causal key rows')
   valid[case]=get(side,'indexer_key_valid').flatten();payload=raw[case][:,:64].long();code=torch.stack([payload&15,payload>>4],-1).reshape(7046,128);sc=raw[case][:,64:].float();values=(lut[code].reshape(7046,4,32)*torch.exp2(sc-127)[:,:,None]).reshape(7046,128);decoded[case]=torch.where(valid[case][:,None],values,torch.zeros_like(values))
  vx,vy=valid['source'],valid['normal'];common=vx&vy;x,y=decoded['source'][common],decoded['normal'][common];key_report=dict(raw_byte_equal=torch.equal(raw['source'],raw['normal']),unequal_raw_bytes=int((raw['source']!=raw['normal']).sum()),source_valid=int(vx.sum()),normal_valid=int(vy.sum()),valid_mask_equal=torch.equal(vx,vy),common_decoded_unequal_elements=int((x!=y).sum()),common_decoded_unequal_rows=int((x!=y).any(-1).sum()),common_decoded_max_abs_difference=float((x-y).abs().max()) if x.numel() else 0.,decoded_finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
  query_values={}
  for case,side in [('source',left),('normal',right)]:
   payload=get(side,'indexer_q_payload').contiguous().view(torch.uint8).reshape(32,64).long();code=torch.stack([payload&15,payload>>4],-1).reshape(32,128);sc=get(side,'indexer_q_packed_sf').contiguous().view(torch.uint8).reshape(32,4).float();query_values[case]=(lut[code].reshape(32,4,32)*torch.exp2(sc-127)[:,:,None]).reshape(32,128)
  qx,qy=query_values['source'],query_values['normal'];query_report=dict(decoded_unequal_elements=int((qx!=qy).sum()),decoded_max_abs_difference=float((qx-qy).abs().max()),decoded_finite=bool(torch.isfinite(qx).all() and torch.isfinite(qy).all()))
  selected={};logits={};selection={}
  for case,side in [('source',left),('normal',right)]:
   selected[case]=get(side,'indexer_selected_logical').flatten().long();ids=selected[case];need(ids.numel()==512 and ids.unique().numel()==512 and ((ids>=0)&(ids<7046)).all(),'actual logical512unique members');logits[case]=get(side,'indexer_logits').flatten().float();need(logits[case].numel()==7046,'actual score prefix');scores=logits[case];finite=torch.isfinite(scores);need(finite.sum()>=512,'at least512finite scores');cutoff=scores.sort(descending=True).values[511];actual_min=scores[ids].min();stable=torch.argsort(scores,descending=True,stable=True)[:512];selection[case]=dict(logits_dtype=str(get(side,'indexer_logits').dtype),finite=int(finite.sum()),cutoff=float(cutoff),actual_selected_min=float(actual_min),all_selected_at_or_above_cutoff=bool((scores[ids]>=cutoff).all()),strictly_above=int((scores>cutoff).sum()),ties_at_cutoff=int((scores==cutoff).sum()),selected_ties_at_cutoff=int((scores[ids]==cutoff).sum()),stable_index_ascending_tie_policy_member_set_equal=set(stable.tolist())==set(ids.tolist()),GLOBAL_core_member_set_equal=set(ids.tolist())==set(get(side,'GLOBAL_canonical').flatten().tolist()) if case=='source' else set(ids.tolist())==set(get(side,'prefill_canonical').flatten()[get(side,'prefill_valid').flatten()&(get(side,'prefill_kind').flatten()==0)].tolist()))
  sx,nx=set(selected['source'].tolist()),set(selected['normal'].tolist());source_only=sorted(sx-nx);normal_only=sorted(nx-sx);exclusive=sorted(set(source_only+normal_only));rows=[dict(logical_key=k,source_score=float(logits['source'][k]),normal_score=float(logits['normal'][k]),source_selected=k in sx,normal_selected=k in nx) for k in exclusive]
  score_delta=logits['source']-logits['normal'];same_scores=torch.equal(get(left,'indexer_logits').contiguous().view(torch.uint8),get(right,'indexer_logits').contiguous().view(torch.uint8));result=dict(state='COMPARED_ACTUAL_L24_INDEXER_QUERY_KEYS_SCORES_SELECTION',candidate_commit=plan['candidate_commit'],model_requests=0,primitive_calls=0,same_P_D_epochs=True,GPU_tokens_joined=True,prior_capture44_checks=prior_checks,all_prior_capture44_common_points_byte_equal=all(r['byte_equal'] for r in prior_checks),actual_common_parameter_names=sorted(shared),actual_common_parameters_equal=all(pa[k]==pb[k] for k in shared),parameter_differences=[k for k in sorted(shared) if pa[k]!=pb[k]],checks=checks,keys=key_report,query=query_report,score_vectors_byte_equal=same_scores,score_unequal_elements=int((logits['source']!=logits['normal']).sum()),score_max_abs_difference=float(score_delta.abs().max()),selection=selection,selected_common=len(sx&nx),selected_source_only=len(source_only),selected_normal_only=len(normal_only),exclusive_member_scores=rows,coverage=coverage,whole_model_fixed=False,formal_acceptance=False,root_cause_proven=False)
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes()))));print(json.dumps(dict(state='SUCCEEDED',score_vectors_equal=same_scores,selected_source_only=len(source_only),selected_normal_only=len(normal_only),model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),model_requests=0));raise
if __name__=='__main__':main()
