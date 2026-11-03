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
     cfg=rec['identity'];pid=rec['PID'];need(cfg['schema']=='history44-inner-graph-latches-v1' and cfg['source_commit']==plan['candidate_commit'] and cfg['role']==role and cfg['positions']==[7045] and cfg['layers']==[24] and cfg['gpu_budget_bytes']==4194304,'actual profile');need(all(v[0] in [0,1] for v in rec['all_registered_hit_counters'].values()) and len(rec['all_registered_hit_counters'])<=192,'actual0/1 counters')
     f=fetch(dict(path=str(Path(ap['path']).parent/'capture-bytes'/role/(str(pid)+'.pt')),SHA256=rec['SHA256']),case+'-'+role+'-'+str(pid)+'.pt');t=torch.load(f,map_location='cpu',weights_only=True);need(set(t)==set(rec['hits']) and bool(t)==(not rec['coverage_empty']),'actual typed hit coverage');need(sum(v.numel()*v.element_size()+8 for v in t.values())<=4194304,'original buffer bound');coverage[case][role].append(dict(PID=pid,world_rank=ranks[pid]['world_rank'],empty=not bool(t),points=len(t),SHA256=rec['SHA256']));params[case][role][pid]=rec['actual_parameters_after_EOF']
     for k,v in t.items():
      m=re.fullmatch(role+r'-L(\d+)-T7045-(.+)',k);need(m is not None and rec['hits'][k]==[1,7045] and v.device.type=='cpu' and v.shape[0]==1,'GPU position and unique sample');key=(int(m[1]),m[2]);need(key not in points,'one rank owner');points[key]=v
    samples[case][role]=points
  s,n=native['source'],native['normal'];sequence=s['input_ids']+s['output_ids'][:-1];need(len(s['input_ids'])==7041 and len(n['input_ids'])==7169 and n['input_ids']==sequence[:7169],'actual G+Y forcedR join');need(s['actual_native_request_profile']==n['actual_native_request_profile'],'actual generation profile')
  for role in ['P','D']:
   x,y=s['identity']['roles'][role],n['identity']['roles'][role];epoch=lambda r:tuple(str(r[k]) for k in ['UID','PID','start_ticks']);need(epoch(x['model_root'])==epoch(y['model_root']) and sorted(epoch(r) for r in x['GPU_ranks'])==sorted(epoch(r) for r in y['GPU_ranks']) and all(x[k]==y[k] for k in ['source_commit','env_SHA256','manifest_SHA256','plan_SHA256','namespace','model_service_helper_SHA256']),'same retained role epochs/runtime')
  need(not samples['source']['P'] and not samples['normal']['D'],'sourceG excludes7045 and normalD startsafter7168; otherroles must be empty');left,right=samples['source']['D'],samples['normal']['P'];need(left[24,'actual_input_token'].item()==right[24,'actual_input_token'].item()==n['input_ids'][7045]==s['output_ids'][4],'actual GPU token7045 joined to nativeIDs')
  # Actual same-layer small parameter fingerprints, independently materialized on the two role devices.
  pa={};pb={}
  for group,out in [(params['source']['D'],pa),(params['normal']['P'],pb)]:
   for perrank in group.values():
    for k,v in perrank.items():
     if k in out:need(out[k]==v,'same-role parameter replicas differ')
     out[k]=v
  shared=set(pa)&set(pb);need(shared,'actual common producer parameters absent');checks=[]
  need(sha(Path(plan['comparator']['path']).read_bytes())==plan['comparator']['SHA256'],'typed comparator fingerprint')
  from capture44_chain_contract import compare_exact
  ordered=['block_entry4HC','actual_input_token','actual_attention_input5120','Qr_before_Q_projection','Qr_actual_quant_0','Qr_actual_quant_1','Q_after_projection','Q_before_core','core_positions','sink','cos_sin_position_row','softmax_scale','core_output','core_scales','output_projection5120','pre_AR5120','post_AR5120','attention_output5120']
  required={'block_entry4HC','actual_input_token','actual_attention_input5120','Qr_before_Q_projection','Q_after_projection','Q_before_core','core_positions','sink','cos_sin_position_row','softmax_scale','core_output','core_scales','output_projection5120','attention_output5120'}
  need({(24,k) for k in required}<=set(left) and {(24,k) for k in required}<=set(right),'required actual attention chain missing')
  for name in ordered:
   key=(24,name)
   if key not in left or key not in right:
    checks.append(dict(layer=24,point=name,comparable=False,source_present=key in left,normal_present=key in right));continue
   x,y=left[key],right[key]
   if x.dtype!=y.dtype or x.shape!=y.shape:
    checks.append(dict(layer=24,point=name,comparable=False,source_dtype=str(x.dtype),normal_dtype=str(y.dtype),source_shape=list(x.shape),normal_shape=list(y.shape)));continue
   report=compare_exact(x,y);report.update(layer=24,point=name,comparable=True,max_abs_difference=float((x.float()-y.float()).abs().max()));checks.append(report)
  need(compare_exact(left[24,'core_positions'],right[24,'core_positions'])['byte_equal'],'same actual position')
  get=lambda side,k:side[24,k]
  pvalid=get(right,'prefill_valid').flatten();pkind=get(right,'prefill_kind').flatten();pcanon=get(right,'prefill_canonical').flatten();pvalues=torch.cat([get(right,'prefill_selected_KV_chunk'+str(i)).reshape(-1,512) for i in range(5)])
  need(len(pvalues)==len(pcanon) and get(right,'prefill_mapping_valid').all(),'P actual BF16 canonical rows')
  pool_reports=[]
  for label,kind,width in [('GLOBAL',0,288),('SWA',1,528)]:
   valid=get(left,label+'_valid').flatten();canon=get(left,label+'_canonical').flatten();raw=get(left,label+'_raw').reshape(-1,width);need(get(left,label+'_mapping_valid').all(),'D actual mapping')
   if label=='SWA':need(get(left,'SWA_window_mapping_valid').all(),'SWA actual absolute token mapping')
   raw=raw[valid];dc=canon[valid];pc=pcanon[pvalid&(pkind==kind)];pv=pvalues[pvalid&(pkind==kind)]
   need(dc.unique().numel()==dc.numel() and pc.unique().numel()==pc.numel(),'unique actual selected canonical rows')
   if label=='GLOBAL':
    payload=raw[:,:256].long();codes=torch.stack([payload&15,payload>>4],dim=-1).reshape(-1,512);lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]);sc=raw[:,256:].contiguous().view(torch.float8_e4m3fn).float();decoded=(lut[codes].reshape(-1,32,16)*sc[:,:,None]).reshape(-1,512).bfloat16()
   else:
    scale_bytes=raw[:,512:];need(not (scale_bytes==255).any(),'finite UE8M0 scale');sc=torch.exp2(scale_bytes.float()-127);decoded=(raw[:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(-1,16,32)*sc[:,:,None]).reshape(-1,512).bfloat16()
   need(torch.isfinite(decoded).all(),'finite actual decoded KV')
   same_bag=sorted(dc.tolist())==sorted(pc.tolist());same_order=torch.equal(dc,pc);report=dict(pool=label,source_rows=dc.numel(),normal_rows=pc.numel(),canonical_bag_equal=same_bag,canonical_order_equal=same_order,decode_format='FP4 E2M1 adjacent-nibble group16 E4M3' if kind==0 else 'FP8 E4M3 group32 UE8M0')
   if same_bag:
    order=torch.tensor([dc.tolist().index(v) for v in pc.tolist()],dtype=torch.int64);x,y=decoded.index_select(0,order),pv;report.update(compare_exact(x,y));report['max_abs_difference']=float((x.float()-y.float()).abs().max()) if x.numel() else 0.;report['unequal_rows']=int((x!=y).any(-1).sum())
   pool_reports.append(report)
  prior_checks=[]
  for case,role,points in [('source','D',left),('normal','P',right)]:
   old=torch.load(fetch(plan['prior_samples'][case],case+'-prior-capture42.pt'),map_location='cpu',weights_only=True)
   for name,priorname in [('block_entry4HC','block_entry4HC'),('actual_attention_input5120','actual_attention_input5120'),('attention_output5120','seam_attn_out5120')]:
    report=compare_exact(points[24,name],old[role+'-L24-T7045-'+priorname]);report.update(case=case,point=name);prior_checks.append(report)
  first=next((r for r in checks if r.get('comparable') and not r['byte_equal']),None);result=dict(state='COMPARED_ACTUAL_HISTORICAL7045_L24_ATTENTION_CHAIN',model_requests=0,primitive_calls=0,candidate_commit=plan['candidate_commit'],GPU_tokens_joined=True,same_P_D_epochs=True,prior_capture42_checks=prior_checks,selected_KV=pool_reports,all_prior_capture42_common_points_byte_equal=all(r['byte_equal'] for r in prior_checks),first_canonical_tensor_difference=first,checks=checks,coverage=coverage,common_actual_parameter_names=sorted(shared),common_actual_parameters_byte_equal=all(pa[k]==pb[k] for k in shared),parameter_differences=[k for k in sorted(shared) if pa[k]!=pb[k]],whole_model_fixed=False,formal_acceptance=False,source_remote_destination_byte_equality_proven=False,expert_routes_inputs_quantization_scales_and_full_batches_equal_proven=False,coefficient_FFN_causality_proven=False,order_scope='actual L24 attention normalized input, Qr/projection/Q, selected canonical KV and per-pool order, core, projection and reduction; cross-pool kernel traversal not inferred')
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes()))));print(json.dumps(dict(state='SUCCEEDED',first_canonical_tensor_difference=first,model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),model_requests=0));raise
if __name__=='__main__':main()
