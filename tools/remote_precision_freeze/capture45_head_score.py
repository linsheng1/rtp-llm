"""Two native scorer calls differ only in the captured headweights.

All query/key bytes are actual capture45 bytes. Three extra rows replicate
the captured query for native alignment; no full-batch identity is claimed.
"""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,s):
    if not ok:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(pin):
    need(sha(pin['path'])==pin['SHA256'],'immutable input drift '+pin['path']);return json.loads(Path(pin['path']).read_text())
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    v=read(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and sha(__file__)==v['script']['SHA256'],'owned exact source');env=read(v['environment']);manifest=read(v['manifest']);need(env['source_commit']==manifest['source_commit']==v['candidate_commit'],'actual candidate');comparison=read(v['comparison']);need(comparison['actual_common_parameters_equal'] and comparison['keys']['raw_byte_equal'] and comparison['query']['decoded_unequal_elements']==0,'actual equal input chain')
    for relative,pin in manifest['production_Python'].items():
        if relative.endswith(('/_indexer_score.py','/_v41_prefill_indexer.py','/_v41_mqa_layout.py')):
            need(sha(Path(env['site'])/relative)==pin['SHA256'],'actual scorer Python drift')
    root=Path(v['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
    def save(n,x):
        tmp=root/(n+'.partial');tmp.write_text(json.dumps(x,indent=2,sort_keys=True));tmp.replace(root/n)
    try:
        import torch
        torch.set_num_threads(1);data={}
        for case,pin in v['cases'].items():
            need(sha(pin['path'])==pin['SHA256'],'captured tensors drift');data[case]=torch.load(pin['path'],map_location='cpu',weights_only=True)
        def get(case,label):return data[case][('D' if case=='source' else 'P')+'-L24-T7045-'+label]
        for label in ['indexer_q_payload','indexer_q_packed_sf']+['indexer_key_raw_chunk'+str(i) for i in range(4)]:
            need(torch.equal(get('normal',label).contiguous().view(torch.uint8),get('source',label).contiguous().view(torch.uint8)),'uncontrolled scorer input '+label)
        raw=torch.cat([get('normal','indexer_key_raw_chunk'+str(i)).reshape(-1,68) for i in range(4)]);need(raw.shape==(7046,68) and get('normal','indexer_key_valid').all() and get('source','indexer_key_valid').all(),'all7046 causal keys valid')
        weights={case:get(case,'indexer_scaled_head_weights').reshape(1,32) for case in data};need(int((weights['normal']!=weights['source']).sum())==1,'one-head intervention')
        if a.check:
            need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU CUDA hidden');result=dict(state='CPU_HEADWEIGHT_SCORE_INPUTS_PASS',model_requests=0,keys=7046,actual_one_head_intervention=True,CUDA_initialized=False)
        else:
            read(v['CPU_result']);projection=read(v['projection_result']);need(projection['both_actual_vectors_reproduced'],'projection calibration must precede score intervention')
            proof=read(dict(path=v['admission_proof'],SHA256=os.environ['STATE_ADMISSION_SHA256']));fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();need(proof['PID']==os.getpid() and proof['UID']==500 and proof['start_ticks']==fields[19] and proof['plan_SHA256']==a.sha256 and -2<=time.time()-proof['at']<=45 and proof['before']['used_v2']==0 and not proof['before']['compute_apps'],'samePID strict admission')
            pin=v['nvml_helper'];need(sha(pin['path'])==pin['SHA256'],'original NVML source');spec=importlib.util.spec_from_file_location('original_head_score_audit',pin['path']);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
            torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();before=audit.nvml();limit=1024**3;remaining=limit-before['used_v2'];need(remaining>128*1024**2 and set(before['compute_apps'])<={os.getpid()},'original1GiB guard');torch.cuda.set_per_process_memory_fraction(remaining/before['total_v2'],0);torch.cuda.reset_peak_memory_stats()
            from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import score_indexer_chunk
            from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score
            backend=_indexer_score._deep_gemm;native=Path(backend.__file__).parent/'_C.cpython-310-aarch64-linux-gnu.so';need(sha(native)==v['native_SHA256'],'actual DeepGEMM native fingerprint')
            q=get('normal','indexer_q_payload').contiguous().view(torch.int8).reshape(1,32,64).repeat(4,1,1).cuda();sf=get('normal','indexer_q_packed_sf').reshape(1,32).repeat(4,1).cuda()
            # Match actual normal request key width; future keys are causally excluded.
            k=torch.zeros((7169,64),dtype=torch.int8);k[:7046]=raw[:,:64].contiguous().view(torch.int8);ks=torch.zeros(7169,dtype=torch.int32);ks[:7046]=raw[:,64:].contiguous().view(torch.int32).flatten();k=k.cuda();ks=ks.cuda();visible=torch.full((4,),7046,dtype=torch.int32,device='cuda');rows=[]
            for case in ['normal','source']:
                y=score_indexer_chunk(q,sf,k,ks,weights[case].repeat(4,1).cuda(),visible,return_bf16=True);torch.cuda.synchronize();actual=get(case,'indexer_logits').reshape(-1);got=y[0,:7046].cpu();rows.append(dict(case=case,actual_scores_byte_equal=torch.equal(got.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)),unequal_scores=int((got!=actual).sum()),max_abs_difference=float((got.float()-actual.float()).abs().max())))
                current=audit.nvml();need(current['used_v2']<=limit and set(current['compute_apps'])<={os.getpid()} and torch.cuda.max_memory_allocated()<=remaining,'original1GiB live guard')
            result=dict(state='ACTUAL_HEADWEIGHT_ONLY_SCORE_INTERVENTION_COMPLETE',model_requests=0,primitive_calls=2,calibration=rows,both_actual_score_vectors_reproduced=all(r['actual_scores_byte_equal'] for r in rows),only_changed_input='one captured BF16 headweight; Q/SF/all causal keys/causal bounds/native primitive/order unchanged',synthetic_surrounding_query_rows=3,torch_peak=torch.cuda.max_memory_allocated(),GPU_after=audit.nvml(),whole_model_fixed=False,formal_acceptance=False)
        save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(dict(state='SUCCEEDED',model_requests=0,result_state=result['state'])))
    except BaseException as e:save('status.json',dict(state='FAILED',error_kind=type(e).__name__,error=str(e),model_requests=0));raise
if __name__=='__main__':main()
