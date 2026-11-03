"""CPU-only MXFP8 field comparison of already completed SWA samples."""
import argparse,hashlib,json,os
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(pin):
 p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and sha(p)==pin['SHA256'],'immutable input drift');return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();plan=load(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sha(__file__)==plan['script']['SHA256'],'CPU actual source');base=load(plan['replay_plan']);proof=load(plan['replay_result']);need(proof['both_actual_outputs_and_scales_reproduced'] and proof['primitive_calls']==2,'actual causal calibration required');env=load(base['environment']);need(sha(Path(env['site'])/'rtp_llm/models_py/modules/dsv41/fp8/_v41_swa_triton.py')==plan['codec_SHA256'],'actual codec source drift');root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
 def save(n,v):
  p=root/(n+'.partial');p.write_text(json.dumps(v,indent=2,sort_keys=True));p.replace(root/n)
 try:
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'no CUDA');data={}
  for case,pin in base['cases'].items():
   need(sha(pin['path'])==pin['SHA256'],'actual tensor drift');data[case]=torch.load(pin['path'],map_location='cpu',weights_only=True)
  get=lambda case,n:data[case]['D-L25-T7169-'+n]
  raw={c:get(c,'SWA_raw').reshape(128,528) for c in data};decoded={c:t[:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(128,16,32)*torch.exp2(t[:,512:].float()-127).unsqueeze(-1) for c,t in raw.items()};x,y=raw['source'],raw['normal'];a,b=decoded['source'],decoded['normal'];need(torch.isfinite(a).all() and torch.isfinite(b).all(),'finite decoded rows')
  group_bytes=(x[:,:512]!=y[:,:512]).reshape(128,16,32).sum(-1);scale_diff=x[:,512:]!=y[:,512:];value_diff=(a!=b).sum(-1);group_diff=(group_bytes>0)|scale_diff;delta=x[:,512:].int()-y[:,512:].int();hist=lambda t:{str(int(k)):int((t==k).sum()) for k in torch.unique(t)}
  report=dict(state='COMPARED_COMPLETED_SWA_MXFP8_FIELDS',model_requests=0,primitive_calls=0,raw_unequal_bytes=int((x!=y).sum()),payload_unequal_bytes=int((x[:,:512]!=y[:,:512]).sum()),scale_unequal_bytes=int(scale_diff.sum()),raw_changed_rows=int((x!=y).any(-1).sum()),raw_changed_groups=int(group_diff.sum()),changed_group_payload_byte_count_histogram=hist(group_bytes[group_diff]),source_minus_normal_scale_exponent_delta_histogram=hist(delta[group_diff]),decoded_unequal_elements=int((a!=b).sum()),decoded_changed_rows=int((a!=b).any(-1).any(-1).sum()),decoded_max_abs_difference=float((a-b).abs().max()),raw_changed_groups_with_exact_decoded_values=int((group_diff&(value_diff==0)).sum()),scale_equal_but_payload_different_groups=int(((~scale_diff)&(group_bytes>0)).sum()),decoded_difference_groups_with_equal_scales=int(((~scale_diff)&(value_diff>0)).sum()),selected_indices=dict(inverse_table_coordinates_min=int(get('source','SWA_logical_indices').min()),inverse_table_coordinates_max=int(get('source','SWA_logical_indices').max()),absolute_position_mapping_proven=False),causal_core_replay=plan['replay_result'],source_remote_destination_byte_equality_proven=False,whole_model_fixed=False)
  save('result-safe.json',report);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(report))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),model_requests=0));raise
if __name__=='__main__':main()
