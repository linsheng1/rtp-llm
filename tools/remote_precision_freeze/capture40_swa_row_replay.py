"""Two calibrated L25 core counterfactuals isolating one numerically unequal SWA row.

No model RPC, native source/kernel modification or projection-weight assumption.
Both actual FP8 outputs/scales must reproduce exactly before any causal claim.
"""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,s):
 if not ok:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(pin):
 p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and sha(p)==pin['SHA256'],'immutable input drift '+str(p));return json.loads(p.read_text())
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();plan=read(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and sha(__file__)==plan['script']['SHA256'],'owned exact replay')
 env=read(plan['environment']);manifest=read(plan['manifest']);need(env['source_commit']==manifest['source_commit']==plan['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c','actual candidate runtime')
 for key in ['bootstrap','nvml_helper']:need(sha(plan[key]['path'])==plan[key]['SHA256'],'admission source drift')
 need(sha(plan['backend']['path'])==plan['backend']['SHA256']=='fb8c9ab9439f12b920630906a67b62940b212dfb1e9e7b9015934ed49b0dd26c','actual FlashMLA native drift')
 compare=read(plan['comparison']);need(compare['state']=='COMPARED_COMPLETED_L25_Q_CORE_READSETS' and compare['readsets']['GLOBAL']['shared_rows_with_different_bytes']==0 and compare['readsets']['GLOBAL']['same_order_and_mask'] and compare['readsets']['SWA']['same_order_and_mask'] and compare['readsets']['SWA']['shared_rows_with_different_bytes']==90,'actual restricted single variable evidence')
 for pin in plan['cases'].values():need(sha(pin['path'])==pin['SHA256'],'actual source tensor bundle drift')
 need(sha(Path(env['site'])/'rtp_llm/models_py/modules/dsv41/fp8/_v41_mega_attention.py')=='200fd0220d13849887c5bb158679e0f64ab0e1fd5c7e9513db5adc79209d9e00','actual production mega adapter drift')
 root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new replay output');root.mkdir(mode=0o700)
 def save(n,v):
  p=root/(n+'.partial');p.write_text(json.dumps(v,sort_keys=True,indent=2));p.replace(root/n)
 try:
  if a.check:
   need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU check hidden')
   result=dict(state='CPU_REPLAY_PINS_AND_ACTUAL_INPUTS_PASS',model_requests=0)
   save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(result));return
  proof=read(dict(path=plan['admission_proof'],SHA256=os.environ['STATE_ADMISSION_SHA256']));f=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();need(proof['PID']==os.getpid() and proof['UID']==500 and proof['start_ticks']==f[19] and proof['plan_SHA256']==a.sha256 and -2<=time.time()-proof['at']<=45 and proof['before']['used_v2']==0 and not proof['before']['compute_apps'],'samePID strict empty admission')
  import torch
  import flash_mla.cuda as backend
  from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention as production
  need(sha(backend.__file__)==plan['backend']['SHA256'] and production._backend() is backend,'actual production backend identity')
  audit=module(plan['nvml_helper']['path'],'original_replay_audit');torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();before=audit.nvml();limit=1024**3;remaining=limit-before['used_v2'];need(remaining>128*1024**2 and set(before['compute_apps'])<={os.getpid()},'original exclusive1GiB remaining budget');torch.cuda.set_per_process_memory_fraction(remaining/before['total_v2'],0);torch.cuda.reset_peak_memory_stats()
  data={case:torch.load(pin['path'],map_location='cpu',weights_only=True) for case,pin in plan['cases'].items()};prefix='D-L25-T7169-';get=lambda case,name:data[case][prefix+name]
  common=['Q_before_core','GLOBAL_raw','GLOBAL_valid','SWA_valid','SWA_logical_indices','GLOBAL_logical_indices','core_positions','sink','cos_sin_position_row']
  for name in common:need(torch.equal(get('source',name).contiguous().view(torch.uint8),get('normal',name).contiguous().view(torch.uint8)),'uncontrolled core input '+name)
  q=get('source','Q_before_core').cuda().reshape(1,64,512).contiguous();pos=get('source','core_positions').cuda().to(torch.int32).reshape(1);sink=get('source','sink').cuda().reshape(64).contiguous();cosrow=get('source','cos_sin_position_row');need(cosrow.shape==(1,64),'actual cosine row shape');cos=torch.zeros((16384,64),device='cuda',dtype=cosrow.dtype);cos.index_copy_(0,pos.long(),cosrow.cuda())
  glob=get('source','GLOBAL_raw').cuda().reshape(4,128,288).contiguous();swavalid=get('source','SWA_valid').cuda().reshape(1,128);gvalid=get('source','GLOBAL_valid').cuda().reshape(1,512);si=torch.arange(128,device='cuda',dtype=torch.int32).reshape(1,128).masked_fill(~swavalid,-1);gi=torch.arange(512,device='cuda',dtype=torch.int32).reshape(1,512).masked_fill(~gvalid,-1)
  raw_swa={c:get(c,'SWA_raw').reshape(128,528) for c in ['source','normal']}
  def decode(raw):return raw[:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(128,16,32)*torch.exp2(raw[:,512:].float()-127).unsqueeze(-1)
  da,db=decode(raw_swa['source']),decode(raw_swa['normal']);changed=(da!=db).any(-1).any(-1).nonzero().flatten();need(changed.numel()==1,'one actual numeric row required');row=int(changed[0]);coordinate=int(get('source','SWA_logical_indices').flatten()[row]);swa_cases={}
  for case,other in [('source','normal'),('normal','source')]:
   hybrid=raw_swa[other].clone();hybrid[row].copy_(raw_swa[case][row]);swa_cases[case]=hybrid
  outputs=[];actual={};calls=0
  for case in ['source','normal']:
   swa=swa_cases[case].cuda().reshape(1,128,528).contiguous()
   o,scales,_=backend.fused_norm_rope_attn_rope_cast_decode(q,swa.unsqueeze(2),si,512**(-0.5),512,sink,None,glob.unsqueeze(2),gi,None,False,0.0,pos,False,64,cos,8,32,True,True,True);calls+=1;torch.cuda.synchronize()
   raw=o.detach().cpu().contiguous().view(torch.uint8).reshape(1,-1);sf=scales.detach().cpu().contiguous().view(torch.uint8).reshape(1,-1);expected=get(case,'core_output').contiguous().view(torch.uint8).reshape(1,-1);es=get(case,'core_scales').contiguous().view(torch.uint8).reshape(1,-1)
   actual[case]=raw;outputs.append(dict(case=case,actual_core_output_exact=torch.equal(raw,expected),actual_core_scales_exact=torch.equal(sf,es),output_SHA256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),captured_SHA256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),output_unequal_bytes=int((raw!=expected).sum()) if raw.shape==expected.shape else None,scales_shape=list(scales.shape),output_shape=list(o.shape)))
   current=audit.nvml();need(set(current['compute_apps'])<={os.getpid()} and current['used_v2']<=limit and torch.cuda.max_memory_allocated()<=remaining,'unchanged1GiB guard');del swa,o,scales
  reproduced=all(x['actual_core_output_exact'] and x['actual_core_scales_exact'] for x in outputs);different=not torch.equal(actual['source'],actual['normal']);result=dict(state='ACTUAL_L25_ONE_NUMERIC_ROW_CORE_REPLAY',model_requests=0,primitive_calls=calls,candidate_commit=plan['candidate_commit'],backend=plan['backend'],compact_selected_pools=True,cosine_only_actual_query_row_populated=True,other_inputs_byte_equal=True,only_variable='one numeric SWA row versus89 decoded-equivalent encodings',changed_SWA_rows=90,numerically_unequal_SWA_row_ordinal=row,inverse_table_coordinate=coordinate,absolute_position_mapping_proven=False,decoded_unequal_elements=int((da[row]!=db[row]).sum()),numeric_row_sufficient_with_other_encodings_changed=reproduced and different,both_actual_outputs_and_scales_reproduced=reproduced,SWA_difference_explains_complete_observed_core_difference=reproduced and different,results=outputs,torch_peak=torch.cuda.max_memory_allocated(),GPU_after=audit.nvml(),GPU_projection_parameter_hashes_verified=False,whole_model_fixed=False,formal_acceptance=False)
  save('result-safe.json',result);need(reproduced and different,'exact real-output calibration failed; no causal claim');save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(dict(state='SUCCEEDED',primitive_calls=calls,real_outputs_exact=True,model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
