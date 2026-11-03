"""Calibrate actual L24 P/D core outputs, then change only one GLOBAL member.

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
 compare=read(plan['comparison']);audit_result=read(plan['global_audit']);need(compare['state']=='COMPARED_ACTUAL_HISTORICAL7045_L24_ATTENTION_CHAIN' and audit_result['rows']==dict(common=511,normal=512,normal_only=1,source=512,source_only=1) and audit_result['common_rows']['unequal_elements']==0,'one actual GLOBAL membership difference');need(all(x['byte_equal'] for x in compare['checks'] if x.get('comparable') and x['point'] in ['Q_before_core','core_positions','sink','cos_sin_position_row','softmax_scale']),'same actual core common inputs')
 for pin in plan['cases'].values():need(sha(pin['path'])==pin['SHA256'],'actual source tensor bundle drift')
 need(sha(Path(env['site'])/'rtp_llm/models_py/modules/dsv41/fp8/_v41_mega_attention.py')=='200fd0220d13849887c5bb158679e0f64ab0e1fd5c7e9513db5adc79209d9e00','actual production mega adapter drift')
 root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new replay output');root.mkdir(mode=0o700)
 def save(n,v):
  p=root/(n+'.partial');p.write_text(json.dumps(v,sort_keys=True,indent=2));p.replace(root/n)
 try:
  import torch
  torch.set_num_threads(1)
  data={case:torch.load(pin['path'],map_location='cpu',weights_only=True) for case,pin in plan['cases'].items()}
  get=lambda case,name:data[case][('D' if case=='source' else 'P')+'-L24-T7045-'+name]
  for name in ['Q_before_core','core_positions','sink','cos_sin_position_row','softmax_scale']:need(torch.equal(get('source',name).contiguous().view(torch.uint8),get('normal',name).contiguous().view(torch.uint8)),'uncontrolled core input '+name)
  pvalid=get('normal','prefill_valid').flatten();kind=get('normal','prefill_kind').flatten();canon=get('normal','prefill_canonical').flatten();need(pvalid.all() and kind.numel()==640 and (kind[:512]==0).all() and (kind[512:]==1).all(),'actual P selected order and all640valid')
  normal=torch.cat([get('normal','prefill_selected_KV_chunk'+str(i)).reshape(128,512) for i in range(5)])
  need(get('source','GLOBAL_valid').all() and get('source','SWA_valid').all(),'actual D all512+128valid')
  raw=get('source','GLOBAL_raw').reshape(512,288);payload=raw[:,:256].long();codes=torch.stack([payload&15,payload>>4],-1).reshape(512,512);lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]);sc=raw[:,256:].contiguous().view(torch.float8_e4m3fn).float();global_values=(lut[codes].reshape(512,32,16)*sc[:,:,None]).reshape(512,512).bfloat16()
  swa=get('source','SWA_raw').reshape(128,528);sv=(swa[:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(128,16,32)*torch.exp2(swa[:,512:].float()-127)[:,:,None]).reshape(128,512).bfloat16();need(torch.equal(sv,normal[512:]),'SWA is controlled exact')
  dc=get('source','GLOBAL_canonical').flatten().tolist();pc=canon[:512].tolist();donly=set(dc)-set(pc);nonly=set(pc)-set(dc);need(len(donly)==len(nonly)==1,'one actual membership row');replacement=normal.clone();ordinal=pc.index(next(iter(nonly)));replacement[ordinal].copy_(global_values[dc.index(next(iter(donly)))]);source=torch.cat([global_values,sv]);need(int((replacement!=normal).any(-1).sum())==1,'exact one-row intervention')
  if a.check:
   need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU check hidden')
   result=dict(state='CPU_REPLAY_PINS_AND_ACTUAL_INPUTS_PASS',model_requests=0,actual_selected_rows=640,one_row_intervention=True,normal_order_preserved=True,CUDA_initialized=torch.cuda.is_initialized());need(not result['CUDA_initialized'],'CPU preparation initialized CUDA')
   save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(result));return
  proof=read(dict(path=plan['admission_proof'],SHA256=os.environ['STATE_ADMISSION_SHA256']));f=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();need(proof['PID']==os.getpid() and proof['UID']==500 and proof['start_ticks']==f[19] and proof['plan_SHA256']==a.sha256 and -2<=time.time()-proof['at']<=45 and proof['before']['used_v2']==0 and not proof['before']['compute_apps'],'samePID strict empty admission')
  import torch
  import flash_mla.cuda as backend
  from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention as production
  need(sha(backend.__file__)==plan['backend']['SHA256'] and production._backend() is backend,'actual production backend identity')
  audit=module(plan['nvml_helper']['path'],'original_replay_audit');torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();before=audit.nvml();limit=1024**3;remaining=limit-before['used_v2'];need(remaining>128*1024**2 and set(before['compute_apps'])<={os.getpid()},'original exclusive1GiB remaining budget');torch.cuda.set_per_process_memory_fraction(remaining/before['total_v2'],0);torch.cuda.reset_peak_memory_stats()
  q=get('normal','Q_before_core').cuda().reshape(1,64,512).contiguous();pos=get('normal','core_positions').cuda().to(torch.int32).reshape(1);sink=get('normal','sink').cuda().reshape(64).contiguous();cosrow=get('normal','cos_sin_position_row');cos=torch.zeros((16384,64),device='cuda',dtype=cosrow.dtype);cos.index_copy_(0,pos.long(),cosrow.cuda());indices=torch.arange(640,device='cuda',dtype=torch.int32).reshape(1,1,640);length=get('normal','prefill_lengths').cuda().to(torch.int32).reshape(1);need(int(length.cpu()[0])==640,'actual P length640');scale=float(get('normal','softmax_scale').item())
  calls=0;outputs=[];captured={case:(get(case,'core_output').contiguous().view(torch.uint8).reshape(1,-1),get(case,'core_scales').contiguous().view(torch.uint8).reshape(1,-1)) for case in ['source','normal']}
  def call(values):
   nonlocal calls
   kv=values.cuda().reshape(640,1,512).contiguous();o,scales,_,_=backend.fused_norm_rope_attn_rope_cast_fwd(q,kv,indices,scale,512,sink,length,False,0.0,pos,False,64,cos,8,32,True,True,True);calls+=1;torch.cuda.synchronize();out=o.detach().cpu().contiguous().view(torch.uint8).reshape(1,-1);out_scale=scales.detach().cpu().contiguous().view(torch.uint8).reshape(1,-1);current=audit.nvml();need(set(current['compute_apps'])<={os.getpid()} and current['used_v2']<=limit and torch.cuda.max_memory_allocated()<=remaining,'unchanged1GiB guard');return out,out_scale
  def matches(pair,expected):return all(torch.equal(x,y) for x,y in zip(pair,expected))
  for case,values in [('normal',normal),('source',source)]:
   pair=call(values);outputs.append(dict(case=case,actual_core_output_exact=torch.equal(pair[0],captured[case][0]),actual_core_scales_exact=torch.equal(pair[1],captured[case][1]),unequal_bytes=int((pair[0]!=captured[case][0]).sum()) if pair[0].shape==captured[case][0].shape else None))
  calibrated=all(r['actual_core_output_exact'] and r['actual_core_scales_exact'] for r in outputs);intervention=None
  if calibrated:
   pair=call(replacement);intervention=dict(changed_GLOBAL_rows=1,normal_order_preserved=True,source_actual_output_exact=matches(pair,captured['source']),normal_actual_output_exact=matches(pair,captured['normal']),GLOBAL_normal_row_ordinal=ordinal)
  result=dict(state='ACTUAL_L24_GLOBAL_MEMBER_CALIBRATION',candidate_commit=plan['candidate_commit'],backend=plan['backend'],model_requests=0,primitive_calls=calls,primitive_calls_max=3,both_actual_outputs_and_scales_reproduced=calibrated,calibration=outputs,intervention=intervention,one_GLOBAL_member_sufficient=bool(calibrated and intervention['source_actual_output_exact']),only_intervention='replace the one normal-only GLOBAL decoded row with actual source-only row at the same ordinal; keep normal order/Q/SWA/positions/params',actual_D_branch_reproduced_via_P_primitive=calibrated,torch_peak=torch.cuda.max_memory_allocated(),GPU_after=audit.nvml(),whole_model_fixed=False,formal_acceptance=False)
  save('result-safe.json',result);need(calibrated,'actual two-branch output calibration failed; intervention blocked');save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(dict(state='SUCCEEDED',primitive_calls=calls,one_GLOBAL_member_sufficient=result['one_GLOBAL_member_sufficient'],model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
