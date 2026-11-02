"""Same capture23 row/weights through deterministic DeepGEMM BF16->FP32.
No original cuBLAS experiment is repeated. Five new backend calls; no model RPC.
"""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
BASE_SHA='84918425f3e147bb8f556dc8819a447be8b676ac40043b9e1cdfc3e03af53a62'
BASE=Path(__file__).with_name('state_projection_probe_v3.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
sp=importlib.util.spec_from_file_location('frozen_STATE_input_and_NVML',BASE);base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
need,read,sha=base.need,base.read,base.sha
C=base.C

def run(plan,items,before):
    cpu,audit,comp,params,keys,xraw,normal,restore,env=items
    need(json.loads(read(plan['CPU_receipt']))==cpu,'actual CPU preflight input/parameter drift')
    import torch,deep_gemm
    dg=Path(env['DG_root'])/'deep_gemm';need(Path(deep_gemm.__file__).resolve().parent==dg.resolve(),'original DG import path')
    native=dg/'_C.cpython-310-aarch64-linux-gnu.so';need(sha(native.read_bytes())=='b0233518517afc95e17a998ae1c8bae7158fab60fba24aa7eeb5cd77d571c99e','same DG native extension')
    need(callable(deep_gemm.bf16_gemm_nt) and not torch.cuda.is_initialized(),'FP32-capable native BF16 API/no prior Torch context')
    after=audit.nvml();need(after['UUID']==before['UUID'] and set(after['compute_apps'])<={os.getpid()},'exclusive physical device changed during prepare')
    torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();initial=audit.nvml();limit=1024**3;remaining=limit-(initial['used_v2']-before['used_v2']);need(remaining>128*1024**2,'unchanged1GiB reserve');torch.cuda.set_per_process_memory_fraction(remaining/initial['total_v2'],0);torch.cuda.reset_peak_memory_stats()
    host=[torch.frombuffer(bytearray(params[k]['raw']),dtype=torch.bfloat16).reshape(params[k]['shape']) for k in keys]
    ref=torch.matmul(torch.frombuffer(bytearray(xraw),dtype=torch.bfloat16).reshape(1,5120).double(),torch.cat(host[:2]).double().T).float().view(torch.uint8).numpy().tobytes()
    weights=[w.cuda() for w in host];query=torch.frombuffer(bytearray(xraw),dtype=torch.bfloat16).reshape(1,5120).cuda();outputs={};samples=[];calls=0
    deep_gemm.use_deterministic_algorithms(True)
    with torch.inference_mode():
        for mode,n in [('prefill',7169),('single',1)]:
            x=torch.zeros((n,5120),device='cuda',dtype=torch.bfloat16);target=7167 if n>1 else 0;x[target].copy_(query[0]);parts=[]
            for w in weights[:2]:
                out=torch.empty((n,w.shape[0]),device='cuda',dtype=torch.float32);deep_gemm.bf16_gemm_nt(x,w,out);calls+=1;torch.cuda.synchronize();parts.append(out[target].contiguous().cpu().view(torch.uint8).numpy().tobytes());del out
            outputs[mode]=b''.join(parts);del x
            v=audit.nvml();need(set(v['compute_apps'])<={os.getpid()} and v['used_v2']-before['used_v2']<=limit and torch.cuda.max_memory_allocated()<=remaining,'unchanged exclusive1GiB guard');samples.append(dict(stage=mode,used_v2=v['used_v2'],torch_peak=torch.cuda.max_memory_allocated()))
        w=torch.cat(weights);out=torch.empty((1,1056),device='cuda',dtype=torch.float32);deep_gemm.bf16_gemm_nt(query,w,out);calls+=1;torch.cuda.synchronize();outputs['single_fused']=out[0,:1024].contiguous().cpu().view(torch.uint8).numpy().tobytes()
        v=audit.nvml();need(set(v['compute_apps'])<={os.getpid()} and v['used_v2']-before['used_v2']<=limit and torch.cuda.max_memory_allocated()<=remaining,'unchanged exclusive1GiB final-call guard');samples.append(dict(stage='single_fused',used_v2=v['used_v2'],torch_peak=torch.cuda.max_memory_allocated()))
    checks={mode:dict(vs_FP64_reference=comp.numeric_report(raw,ref,'torch.float32',mode),vs_old_normal=comp.numeric_report(raw,normal,'torch.float32',mode),SHA256=sha(raw)) for mode,raw in outputs.items()}
    target=torch.frombuffer(bytearray(ref),dtype=torch.float32)
    accurate=all(torch.allclose(torch.frombuffer(bytearray(raw),dtype=torch.float32),target,rtol=1e-5,atol=1e-5) for raw in outputs.values())
    equal=outputs['prefill']==outputs['single']==outputs['single_fused']
    return dict(cpu,state='DETERMINISTIC_STATE_PROJECTION_COMPONENT_COMPLETE',primitive_calls=calls,backend='same ad62 DeepGEMM bf16_gemm_nt FP32 output, deterministic=True',checks=checks,all_M_and_N_variants_byte_equal=equal,all_variants_close_to_FP64_reference=accurate,component_fix_pass=equal and accurate,GPU_samples=samples,GPU_before_import=before,model_requests=0,whole_model_fixed=False,formal_acceptance=False)

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();plan=json.loads(read(dict(path=str(a.plan),SHA256=a.sha256)));need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'new backend probe source differs');root=Path(plan['output_root']);need(os.getuid()==500 and root.parent==C and not root.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and plan['mode']=='run','new owned frozen GPU0-only root');root.mkdir(mode=0o700);start=time.time()
    def save(name,value):
        t=root/(name+'.partial');t.write_text(json.dumps(value,sort_keys=True,indent=2));t.replace(root/name)
    try:
        audit=base.load(plan['nvml_helper'],'strict_pre_import_NVML');before=audit.nvml();need(before['capability']==[10,3] and before['used_v2']==0 and not before['compute_apps'],'unchanged strict empty GPU0 gate: '+json.dumps(before))
        result=run(plan,base.inspect(plan),before);save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',started_at=start,finished_at=time.time(),result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes())),model_requests=0));print(json.dumps({k:result[k] for k in ['state','primitive_calls','component_fix_pass','all_M_and_N_variants_byte_equal','all_variants_close_to_FP64_reference']}))
    except BaseException as e:save('status.json',dict(state='FAILED',error_kind=type(e).__name__,error=str(e),finished_at=time.time(),model_requests=0));raise
if __name__=='__main__':main()
