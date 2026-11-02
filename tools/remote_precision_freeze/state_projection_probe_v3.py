"""Capture23 STATE projection test: same observed BF16 row and real weights.
Only M, then N change. Other matrix rows are synthetic zeros, not original batch.
CPU check precedes five original native GEMM calls. No model request or KV writes.
"""
import argparse,hashlib,importlib.util,json,math,os,struct,time
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(pin,limit=8*1024**2):
    p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'bounded nonlink input required');before=p.stat();raw=p.read_bytes();after=p.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and sha(raw)==pin['SHA256'],'immutable input SHA differs: '+str(p));return raw
def load(pin,name):
    read(pin);s=importlib.util.spec_from_file_location(name,pin['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def inspect(plan):
    report=json.loads(read(plan['comparison']));need(report['state']=='COMPARED_EXISTING_FROZEN_CAPTURE' and report['capture_plan_SHA256']==plan['capture_plan_SHA256'] and all(v['byte_equal'] for v in report['transition_checks']),'current verified L0/L1 equality required')
    aroot=Path(plan['comparison']['path']).parent;phases={k:json.loads((aroot/(k+'-barrier.json')).read_text()) for k in ['short','full']}
    def row(phase,label,pos,width,dtype):
        result=[];pins=[]
        for rank in phases[phase]['ranks']:
            m=json.loads(read(rank['manifest']));need(m['request_id']==report['actual_request_ids'][phase] and m['input_ids_sha256']==plan['input_SHA256'],'actual manifest RID/input differs')
            samples=[v for v in m['samples'] if v['label']==label];need(len(samples)==1,'exact original sample required')
            v=samples[0];need(v['dtype']==dtype and v['sample_shape']==[2,width],'typed original sample differs')
            if v['valid'][pos-7167]:
                p=dict(path=str(Path(rank['manifest']['path']).parent/v['file']),SHA256=v['sha256']);raw=read(p);size=2 if dtype=='torch.bfloat16' else 4
                need(len(raw)==2*width*size,'raw original byte size differs');data=raw[(pos-7167)*width*size:(pos-7166)*width*size];result.append(data);pins.append(dict(**p,row_position=pos,row_SHA256=sha(data),original_shape=v.get('original_shape'),PID=m['PID']))
        need(result and all(v==result[0] for v in result),'original owner/replica rows disagree');return result[0],pins
    x,xp=row('full','2._prefill_produce.full_hidden5120',7167,5120,'torch.bfloat16')
    normal,np=row('full','2._write_states.raw_after',7167,1024,'torch.float32')
    restore,rp=row('short','2._read_state.original_result1024_post_gather',7167,1024,'torch.float32')
    values,vp=row('full','2._write_states.original_values512',7167,512,'torch.float32');scores,sp=row('full','2._write_states.original_scores512',7167,512,'torch.float32')
    before,bp=row('short','2._read_state.raw_before',7167,1024,'torch.float32')
    need(normal==values+scores and restore==before and normal!=restore,'STATE store/read boundary relation differs')
    base=load(plan['nvml_helper'],'original_NVML_and_checkpoint_audit')
    index=json.loads(read(dict(path='/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash/model.safetensors.index.json',SHA256=base.INDEX_SHA)))
    keys=['layers.2.attn.compressor.wkv.weight','layers.2.attn.compressor.wgate.weight','layers.2.attn.indexer.weights_proj.weight']
    params={};metadata=[]
    for filename in sorted({index['weight_map'][k] for k in keys}):
        subset=[k for k in keys if index['weight_map'][k]==filename];data,meta=base.checkpoint_tensors(subset);params.update(data);metadata.append(meta)
    for key in keys:need(params[key]['dtype']=='BF16' and params[key]['shape']==([32,5120] if 'weights_proj' in key else [512,5120]),'original compressor weight dtype/shape differs')
    env=json.loads(read(plan['environment']));manifest=json.loads(read(plan['manifest']))
    need(manifest['source_commit']==plan['candidate_commit'],'current native candidate manifest differs')
    for key in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']:need(os.environ.get(key)==env['environment'][key],'exact original runtime env differs: '+key)
    pplan=json.loads(read(plan['model_plan']));e=pplan['environment'];padded=4*1794;need(padded<=min(int(e.get('DSV41_SMALL_CP_X_GATHER_MAX_ROWS','131072')),65536) and padded*(5120*2-1024*4)<int(e.get('DSV41_LOCAL_PROJECT_MIN_SAVED_MB','160'))*1024**2,'normal original raw-gather projection geometry changed')
    sources={key:dict(path=member['path'],SHA256=member['SHA256']) for key,member in manifest['native'].items() if Path(key).name in ['libth_transformer.so','librtp_compute_ops.so']}
    for pin in sources.values():
        p=Path(pin['path']);need(p.is_file() and not p.is_symlink(),'native file identity');before=p.stat();h=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
        after=p.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and h.hexdigest()==pin['SHA256'],'native artifact drift')
    import torch
    need(not torch.cuda.is_initialized() and torch.__version__.startswith('2.11') and Path(torch.__file__).resolve().is_relative_to(Path(env['Torch_partial_site']).resolve()),'actual CPU Torch identity/no CUDA init')
    from rtp_llm.ops.compute_ops import rtp_llm_ops
    need(callable(rtp_llm_ops.cublas_gemm_bf16_bf16_fp32) and not torch.cuda.is_initialized(),'actual native operator CPU availability/no CUDA init')
    comp=load(plan['comparator'],'existing_typed_numeric_arithmetic')
    cpu=dict(state='STATE_PROJECTION_ACTUAL_INPUT_CPU_PASS',model_requests=0,primitive_calls=0,candidate_commit=plan['candidate_commit'],capture_plan_SHA256=plan['capture_plan_SHA256'],input_SHA256=plan['input_SHA256'],actual_request_ids=report['actual_request_ids'],actual_input_row=xp,normal_reference=np,restored_reference=rp,normal_store_equals_original_values_and_scores=True,restored_pool_owner_equals_all_four_gathered_reads=True,parameter_sources=metadata,native_sources=sources,normal_GEMM=dict(M=7169,N=512,K=5120,separate_values_and_scores=True),decode_GEMM_source_prediction=dict(M=1,N=1056,K=5120,actual_D_invocation_shape_observed=False),synthetic_other_rows=True,wholebatch_equal_claimed=False,transport_source_bytes_equal_proven=False,GPU_budget_bytes=1024**3)
    return cpu,base,comp,params,keys,x,normal,restore,env

def run(plan,items,before):
    cpu,base,comp,params,keys,xraw,normal,restore,env=items
    expected=json.loads(read(plan['CPU_receipt']));need(expected==cpu,'actual CPU preflight inputs/parameters changed')
    after_inspect=base.nvml();need(after_inspect['UUID']==before['UUID'] and set(after_inspect['compute_apps'])<={os.getpid()} and after_inspect['used_v2']-before['used_v2']<1024**3,'CPU/native-import initialization exceeded unchanged exclusive1GiB budget: '+json.dumps(after_inspect))
    import torch
    need(not torch.cuda.is_initialized() and torch.__version__.startswith('2.11') and Path(torch.__file__).resolve().is_relative_to(Path(env['Torch_partial_site']).resolve()),'original Torch identity/no prior CUDA context')
    from rtp_llm.ops.compute_ops import rtp_llm_ops
    op=rtp_llm_ops.cublas_gemm_bf16_bf16_fp32
    torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();initial=base.nvml();limit=1024**3;remaining=limit-(initial['used_v2']-before['used_v2']);need(remaining>128*1024**2,'original1GiB reserve');torch.cuda.set_per_process_memory_fraction(remaining/initial['total_v2'],0);torch.cuda.reset_peak_memory_stats()
    weights=[torch.frombuffer(bytearray(params[k]['raw']),dtype=torch.bfloat16).reshape(params[k]['shape']).cuda() for k in keys]
    query=torch.frombuffer(bytearray(xraw),dtype=torch.bfloat16).reshape(1,5120).cuda();outputs={};mem=[];calls=0
    with torch.inference_mode():
        for mode,n in [('prefill',7169),('single',1)]:
            x=torch.zeros((n,5120),device='cuda',dtype=torch.bfloat16);target=7167 if n>1 else 0;x[target].copy_(query[0]);parts=[]
            for w in weights[:2]:
                out=op(x,w);calls+=1;parts.append(out[target].contiguous().cpu().view(torch.uint8).numpy().tobytes());del out
            outputs[mode]=b''.join(parts);del x
            torch.cuda.synchronize();v=base.nvml();need(set(v['compute_apps'])<={os.getpid()} and v['used_v2']-before['used_v2']<=limit and torch.cuda.max_memory_allocated()<=remaining,'unchanged exclusive1GiB guard');mem.append(dict(stage=mode,used_v2=v['used_v2'],torch_peak=torch.cuda.max_memory_allocated()))
        fused=op(query,torch.cat(weights));calls+=1;outputs['single_fused']=fused[0,:1024].contiguous().cpu().view(torch.uint8).numpy().tobytes();torch.cuda.synchronize()
        v=base.nvml();need(set(v['compute_apps'])<={os.getpid()} and v['used_v2']-before['used_v2']<=limit and torch.cuda.max_memory_allocated()<=remaining,'unchanged exclusive1GiB final-call guard');mem.append(dict(stage='single_fused',used_v2=v['used_v2'],torch_peak=torch.cuda.max_memory_allocated()))
    checks={mode:dict(vs_normal=comp.numeric_report(raw,normal,'torch.float32',mode),vs_restored=comp.numeric_report(raw,restore,'torch.float32',mode),SHA256=sha(raw)) for mode,raw in outputs.items()}
    return dict(cpu,state='STATE_PROJECTION_SHAPE_COUNTERFACTUAL_COMPLETE',primitive_calls=calls,checks=checks,changing_only_M_changes_row=outputs['prefill']!=outputs['single'],changing_only_N_fusion_changes_row=outputs['single']!=outputs['single_fused'],normal_original_STATE_exact_reproduced=outputs['prefill']==normal,restored_original_STATE_exact_predicted=outputs['single_fused']==restore,GPU_samples=mem,GPU_before_any_Torch_or_native_import=before,GPU_after_inspection_imports=after_inspect,operator_phenomenon_proven=outputs['prefill']==normal and outputs['single_fused']==restore and normal!=restore,whole_model_fixed=False,formal_acceptance=False)

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();plan=json.loads(read(dict(path=str(a.plan),SHA256=a.sha256)));need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'operator script identity differs');root=Path(plan['output_root']);need(os.getuid()==500 and root.parent==C and not root.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==('0' if plan['mode']=='run' else ''),'new owned scoped operator batch required');root.mkdir(mode=0o700);start=time.time()
    def save(name,value):
        q=root/(name+'.partial');q.write_text(json.dumps(value,sort_keys=True,indent=2));q.replace(root/name)
    try:
        before=None
        if plan['mode']=='run':
            audit=load(plan['nvml_helper'],'pre_import_NVML_audit');before=audit.nvml();need(before['capability']==[10,3] and before['used_v2']==0 and not before['compute_apps'],'unchanged strict physical GPU0 empty admission before CUDA-capable imports: '+json.dumps(before))
        items=inspect(plan);result=run(plan,items,before) if plan['mode']=='run' else items[0];save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',finished_at=time.time(),started_at=start,result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes())),model_requests=0));print(json.dumps({k:result[k] for k in ['state','primitive_calls','normal_original_STATE_exact_reproduced','restored_original_STATE_exact_predicted'] if k in result}))
    except BaseException as e:save('status.json',dict(state='FAILED',error_kind=type(e).__name__,error=str(e),finished_at=time.time(),model_requests=0));raise
if __name__=='__main__':main()
