"""Read-only CPU join of the already completed GLOBAL sparse readsets."""
import argparse,collections,hashlib,json,os,signal
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(pin):
 p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and sha(p)==pin['SHA256'],'input fingerprint');return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();plan=read(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sha(__file__)==plan['script']['SHA256'],'CPU actual script');base=read(plan['comparison_plan']);prior=read(plan['comparison_result']);need(prior['GPU_tokens_joined'] and prior['same_P_D_epochs'] and prior['all_prior_capture42_common_points_byte_equal'],'same actual native history proof');env=read(base['environment'])
 for rel,digest in plan['codecs'].items():need(sha(Path(env['site'])/rel)==digest,'actual codec/mapping source')
 root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new CPU output');root.mkdir(mode=0o700)
 def save(name,v):
  p=root/(name+'.partial');p.write_text(json.dumps(v,indent=2,sort_keys=True));p.replace(root/name)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded GLOBAL CPU audit')));signal.alarm(120)
 try:
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU only');data={}
  for case,role in [('source','D'),('normal','P')]:
   rows=[r for r in prior['coverage'][case][role] if not r['empty']];need(len(rows)==1,'one actual target rank');r=rows[0];p=Path(base['output_root'])/(case+'-'+role+'-'+str(r['PID'])+'.pt');need(sha(p)==r['SHA256'],'typed actual snapshot');d=torch.load(p,map_location='cpu',weights_only=True);data[case]={k.split('-T7045-',1)[1]:v for k,v in d.items()}
  d,n=data['source'],data['normal'];get=lambda x,k:x[k].flatten();dv=get(d,'GLOBAL_valid');dc=get(d,'GLOBAL_canonical')[dv];raw=d['GLOBAL_raw'].reshape(-1,288)[dv];pv=get(n,'prefill_valid')&(get(n,'prefill_kind')==0);pc=get(n,'prefill_canonical')[pv];nv=torch.cat([n['prefill_selected_KV_chunk'+str(i)].reshape(-1,512) for i in range(5)])[pv]
  payload=raw[:,:256].long();code=torch.stack([payload&15,payload>>4],-1).reshape(-1,512);lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]);sc=raw[:,256:].contiguous().view(torch.float8_e4m3fn).float();decoded=(lut[code].reshape(-1,32,16)*sc[:,:,None]).reshape(-1,512).bfloat16();need(torch.isfinite(decoded).all() and torch.isfinite(nv).all(),'finite selected values');need(dc.unique().numel()==dc.numel() and pc.unique().numel()==pc.numel(),'unique actual canonical row ids')
  di={v:i for i,v in enumerate(dc.tolist())};ni={v:i for i,v in enumerate(pc.tolist())};common=sorted(set(di)&set(ni));donly=sorted(set(di)-set(ni));nonly=sorted(set(ni)-set(di));x=decoded[[di[k] for k in common]];y=nv[[ni[k] for k in common]]
  rowsha=lambda t:[hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for v in t]
  dh,nh=rowsha(decoded),rowsha(nv);dhonly=[dh[di[k]] for k in donly];nhonly=[nh[ni[k]] for k in nonly]
  def delta(x,y):
   need(x.shape==y.shape,'same row geometry');return dict(elements=x.numel(),unequal_elements=int((x!=y).sum()),unequal_rows=int((x!=y).any(-1).sum()),max_abs_difference=float((x.float()-y.float()).abs().max()) if x.numel() else 0.)
  actual_parameters={}
  for case,role in [('source','D'),('normal','P')]:
   ack=json.loads((Path(base['output_root'])/(case+'-'+role+'-ack.json')).read_text());params=[rec['actual_parameters_after_EOF'] for rec in ack['rank_controls'] if rec['hits']][0];actual_parameters[case]=dict(total_bytes=sum(v['bytes'] for v in params.values()),names=sorted(params))
  result=dict(state='AUDITED_ACTUAL_GLOBAL_SELECTED_ROWS',model_requests=0,primitive_calls=0,comparison_plan=plan['comparison_plan'],comparison_result=plan['comparison_result'],actual_parameters=actual_parameters,mapping_checks=dict(source_GLOBAL=bool(d['GLOBAL_mapping_valid'].all()),normal_P=bool(n['prefill_mapping_valid'].all())),topology=dict(source=d['actual_topology'].tolist(),normal=n['actual_topology'].tolist()),normal_chunk_meta=n['prefill_chunk_meta'].tolist(),rows=dict(source=len(di),normal=len(ni),common=len(common),source_only=len(donly),normal_only=len(nonly)),common_rows=delta(x,y),same_decoded_row_order=dh==nh,same_decoded_row_multiset=collections.Counter(dh)==collections.Counter(nh),exclusive_decoded_row_multisets_equal=collections.Counter(dhonly)==collections.Counter(nhonly),source_only_zero_rows=sum(not bool(decoded[di[k]].any()) for k in donly),normal_only_zero_rows=sum(not bool(nv[ni[k]].any()) for k in nonly),same_slot_order_values=delta(decoded,nv) if decoded.shape==nv.shape else None,canonical_coordinate_shift_histogram={str(int(k)):int(((dc-pc)==k).sum()) for k in torch.unique(dc-pc)} if dc.shape==pc.shape else None,source_canonical_range=[int(dc.min()),int(dc.max())],normal_canonical_range=[int(pc.min()),int(pc.max())],selected_raw_and_decoded_digest=dict(source_raw=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),source_decoded=hashlib.sha256(decoded.view(torch.uint8).numpy().tobytes()).hexdigest(),normal_BF16=hashlib.sha256(nv.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()),whole_model_fixed=False,root_cause_proven=False,cross_pool_kernel_traversal_order_proven=False)
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(result))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),model_requests=0));raise
if __name__=='__main__':main()
