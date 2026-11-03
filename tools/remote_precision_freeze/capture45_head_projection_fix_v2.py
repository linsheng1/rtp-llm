"""Actual captured row/weight replay. Surrounding rows are synthetic, explicitly.

Finite CPU preparation and at most two BF16 GEMMs; no model request or fix.
"""
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path
C = Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok, message):
    if not ok: raise ValueError(message)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(pin):
    need(sha(pin['path']) == pin['SHA256'], 'immutable input drift: '+pin['path'])
    return json.loads(Path(pin['path']).read_text())
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    v=read(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and sha(__file__)==v['script']['SHA256'],'owned exact source')
    env=read(v['environment']);manifest=read(v['manifest']);need(env['source_commit']==manifest['source_commit']==v['candidate_commit'],'candidate identity')
    comparison=read(v['comparison']);need(comparison['state']=='COMPARED_ACTUAL_L24_INDEXER_QUERY_KEYS_SCORES_SELECTION' and comparison['actual_common_parameters_equal'] and comparison['keys']['raw_byte_equal'],'actual capture prerequisite')
    root=Path(v['output_root']);need(root.parent==C and not root.exists(),'new result required');root.mkdir(mode=0o700)
    def save(n,x):
        tmp=root/(n+'.partial');tmp.write_text(json.dumps(x,indent=2,sort_keys=True));tmp.replace(root/n)
    try:
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open
        torch.set_num_threads(1)
        data={}
        for case,pin in v['cases'].items():
            need(sha(pin['path'])==pin['SHA256'],'captured tensor drift');data[case]=torch.load(pin['path'],map_location='cpu',weights_only=True)
        def get(case,label):return data[case][('D' if case=='source' else 'P')+'-L24-T7045-'+label].reshape(-1)
        x=get('normal','actual_attention_input5120');need(torch.equal(x.view(torch.uint8),get('source','actual_attention_input5120').view(torch.uint8)),'actual input differs')
        with safe_open(v['weight']['path'],framework='pt',device='cpu') as f:w=f.get_tensor(v['weight']['key'])
        need(w.shape==(32,5120) and w.dtype==x.dtype==torch.bfloat16 and hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==v['weight']['SHA256'],'actual projection weight differs')
        exact=F.linear(x.double().reshape(1,-1),w.double()).flatten();canonical=(exact*(1/64)).bfloat16()
        actual={case:get(case,'indexer_scaled_head_weights') for case in ['source','normal']}
        different=(actual['source']!=actual['normal']).nonzero().flatten().tolist();need(different==[11],'one actual differing head11 required')
        cpu=dict(state='CPU_ACTUAL_HEAD_PROJECTION_INPUTS_PASS',model_requests=0,CUDA_initialized=torch.cuda.is_initialized(),actual_input_byte_equal=True,weight=v['weight'],actual_different_heads=different,head11=dict(exact_FP64_raw=float(exact[11]),source_scaled=float(actual['source'][11]),normal_scaled=float(actual['normal'][11]),canonical_scaled=float(canonical[11])),canonical_matches={case:bool(torch.equal(canonical,actual[case])) for case in actual},surrounding_rows='synthetic replicas of the captured row; not a claim of full actual batch identity')
        if a.check:
            need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU must hide CUDA');result=cpu
        else:
            read(v['CPU_result']);proof=read(dict(path=v['admission_proof'],SHA256=os.environ['STATE_ADMISSION_SHA256']));fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
            need(proof['PID']==os.getpid() and proof['UID']==500 and proof['start_ticks']==fields[19] and proof['plan_SHA256']==a.sha256 and -2<=time.time()-proof['at']<=45 and proof['before']['used_v2']==0 and not proof['before']['compute_apps'],'samePID strict admission')
            pin=v['nvml_helper'];need(sha(pin['path'])==pin['SHA256'],'original NVML source');spec=importlib.util.spec_from_file_location('original_head_replay_audit',pin['path']);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
            torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();before=audit.nvml();limit=1024**3;remaining=limit-before['used_v2'];need(remaining>128*1024**2 and set(before['compute_apps'])<={os.getpid()},'original1GiB exclusive budget');torch.cuda.set_per_process_memory_fraction(remaining/before['total_v2'],0);torch.cuda.reset_peak_memory_stats()
            pin=v['kernel'];need(sha(pin['path'])==pin['SHA256'],'new kernel source pin');spec=importlib.util.spec_from_file_location('actual_fixed_head_projection',pin['path']);kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
            wg=w.cuda();rows=[]
            for m in [4,1762]:
                batch=x.reshape(1,-1).repeat(m,1).cuda();y=kernel.try_project_indexer_head_weights(batch,wg);need(y is not None,'actual kernel path required');scaled=(y[0].float()*(1/64)).bfloat16().cpu();torch.cuda.synchronize()
                rows.append(dict(M=m,head11_scaled=float(scaled[11]),matches_actual={case:bool(torch.equal(scaled,actual[case])) for case in actual},unequal_heads_to_actual={case:int((scaled!=actual[case]).sum()) for case in actual},canonical_unequal_heads=int((scaled!=canonical).sum())))
                del batch,y
                current=audit.nvml();need(current['used_v2']<=limit and set(current['compute_apps'])<={os.getpid()} and torch.cuda.max_memory_allocated()<=remaining,'original1GiB guard')
            need(all(r['canonical_unequal_heads']==0 for r in rows),'fixed outputs must match actual-input FP64 reference at both shapes')
            graph_input=x.reshape(1,-1).repeat(4,1).cuda();kernel.try_project_indexer_head_weights(graph_input,wg);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):graph_output=kernel.try_project_indexer_head_weights(graph_input,wg)
            graph.replay();torch.cuda.synchronize();need(torch.equal(graph_output[0].cpu(),exact.bfloat16()),'captured raw output reference')
            changed=x.clone();changed[17]+=0.125;graph_input[0].copy_(changed);graph.replay();torch.cuda.synchronize();reference=F.linear(changed.double().reshape(1,-1),w.double()).bfloat16().flatten();need(torch.equal(graph_output[0].cpu(),reference) and torch.equal(graph_output[1].cpu(),exact.bfloat16()),'graph replay must consume changed input and preserve other rows')
            current=audit.nvml();need(current['used_v2']<=limit and set(current['compute_apps'])<={os.getpid()} and torch.cuda.max_memory_allocated()<=remaining,'original1GiB graph guard')
            result=dict(cpu,state='ACTUAL_FIXED_HEAD_PROJECTION_TEST_PASS',kernel=v['kernel'],fixed_both_shapes_match_FP64=True,graph_changed_input_reference_pass=True,kernel_enqueues=6,graph_capture_enqueues=1,graph_replays=2,CPU_result=v['CPU_result'],primitive_calls=6,cases=rows,both_actual_vectors_reproduced=False,classification='proposed kernel unit only; actual model candidate not yet deployed',P_actual_original_M_verified=False,torch_peak=torch.cuda.max_memory_allocated(),GPU_after=audit.nvml(),whole_model_fixed=False,formal_acceptance=False)
        save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(dict(state='SUCCEEDED',model_requests=0,result_state=result['state'])))
    except BaseException as e:save('status.json',dict(state='FAILED',error_kind=type(e).__name__,error=str(e),model_requests=0));raise
if __name__=='__main__':main()
