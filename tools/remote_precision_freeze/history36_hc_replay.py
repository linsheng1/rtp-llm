"""Existing capture36 only: calibrate HC inputs, vary one weight representation.

Then calibrate both MegaMHC rows and replace only the captured delayed pre-mix.
Other rows repeat the actual query. Zero model RPC, unchanged1GiB admission.
No L0 split16/1 experiment, expert/routing equivalence or whole-model claim.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
LIMIT=1024**3
RESERVE=192*1024**2


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(pin,limit=8*1024**2):
    p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'bounded original input '+str(p))
    a=p.stat();raw=p.read_bytes();b=p.stat()
    need((a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns) and sha(raw)==pin['SHA256'],'immutable input drift '+str(p))
    return raw


def module(pin,name):
    read(pin);s=importlib.util.spec_from_file_location(name,pin['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def bits(t):return t.contiguous().view(__import__('torch').uint8).numpy().tobytes()


def inspect(plan):
    import torch
    need(not torch.cuda.is_initialized(),'CPU inspection must not initialize CUDA')
    report=json.loads(read(plan['analysis']));ap=json.loads(read(plan['analysis_plan']))
    need(report['state']=='HISTORY36_EXISTING_PRODUCER_CAPTURE_COMPARED' and
         report['source_full_IDs_calibrated_241'] and report['normal_full_IDs_calibrated_143'] and
         all(p['all_actual_sample_owners_equal'] for p in report['actual_L1_HC_parameter_checks']), 'accepted actual4rank comparison required')
    need(report['epochs']['source']==report['epochs']['normal'],'same actual model pair required')
    labels=['after_Engram4HC','natural_attnHC_residual4HC','natural_attnHC_pre4','natural_attnHC_post4','natural_attnHC_comb16',
            'seam_actual_rows','seam_residual4HC','seam_attn_out5120','attnHC_pre4','attnHC_post4','attnHC_comb16','seam_fused_route',
            'seam_post4HC','FFNHC_pre4','FFNHC_post4','FFNHC_comb16','seam_FFN_norm5120','FFN_input5120']
    rows={};owners={};raw_pins={};actual_params={}
    parent=Path(plan['analysis']['path']).parent
    for case,role in [('source','D'),('normal','P')]:
        ack=json.loads(read(dict(path=str(parent/(case+'-ack.json')),SHA256=ap[case+'_ack']['SHA256'])))
        selected=[r for r in ack['rank_controls'] if role+'-L1-T7043-seam_residual4HC' in r['hits']]
        need(selected,'actual target owner missing')
        samples=[]
        for r in selected:
            pin=dict(path=str(parent/(case+'-'+str(r['PID'])+'.pt')),SHA256=r['SHA256'])
            read(pin);data=torch.load(pin['path'],map_location='cpu',weights_only=True)
            sample={k:data[role+'-L1-T7043-'+k] for k in labels}
            need(all(r['hits'][role+'-L1-T7043-'+k]==[1,7043] for k in labels),'sole target-hit proof differs')
            need(all(t.device.type=='cpu' and torch.isfinite(t).all() for t in sample.values()),'finite original CPU tensors required')
            samples.append(sample);raw_pins.setdefault(case,[]).append(pin)
        need(all(all(bits(s[k])==bits(samples[0][k]) for k in labels) for s in samples),'actual replicas disagree')
        rows[case]=samples[0];owners[case]=[r['PID'] for r in selected];actual_params[case]=selected[0]['actual_parameters_after_EOF']
        need(int(rows[case]['seam_actual_rows'].item())==({'source':1,'normal':1794}[case]) and
             int(rows[case]['seam_fused_route'].item())==1,'actual M/fused seam differs')
        need(bits(rows[case]['after_Engram4HC'])==bits(rows[case]['natural_attnHC_residual4HC'])==bits(rows[case]['seam_residual4HC']) and
             bits(rows[case]['natural_attnHC_pre4'])==bits(rows[case]['attnHC_pre4']) and
             bits(rows[case]['natural_attnHC_post4'])==bits(rows[case]['attnHC_post4']) and
             bits(rows[case]['natural_attnHC_comb16'])==bits(rows[case]['attnHC_comb16']) and
             bits(rows[case]['seam_FFN_norm5120'])==bits(rows[case]['FFN_input5120']), 'actual seam continuity differs')
    for label in ['after_Engram4HC','seam_attn_out5120','attnHC_post4','seam_post4HC','FFNHC_pre4','FFNHC_post4','FFNHC_comb16']:
        need(bits(rows['source'][label])==bits(rows['normal'][label]),'required equal row differs '+label)
    need(bits(rows['source']['attnHC_pre4'])!=bits(rows['normal']['attnHC_pre4']) and
         bits(rows['source']['seam_FFN_norm5120'])!=bits(rows['normal']['seam_FFN_norm5120']),'original HC/norm deltas required')
    env=json.loads(read(plan['environment']));manifest=json.loads(read(plan['manifest']))
    need(env['source_commit']==manifest['source_commit']==plan['candidate_commit'],'actual runtime candidate differs')
    need(torch.__version__.startswith('2.11') and Path(torch.__file__).resolve().is_relative_to(Path(env['Torch_partial_site']).resolve()),'actual private Torch origin')
    dg=Path(env['DG_root'])/'deep_gemm/_C.cpython-310-aarch64-linux-gnu.so'
    need(sha(dg.read_bytes())=='b0233518517afc95e17a998ae1c8bae7158fab60fba24aa7eeb5cd77d571c99e','actual native DeepGEMM identity')
    for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']:
        need(os.environ.get(k)==env['environment'][k],'private CPU/runtime env differs '+k)
    for pin in plan['production_sources'].values():read(pin,256*1024)
    flags=json.loads(read(plan['flags_receipt']))['flags']
    need(flags==dict(DSV4_MHC_SPLIT_MODE='fixed',DSV4_MHC_NUM_SPLITS=None,DSV4_MHC_PRE_GEMM_BACKEND=None),'actual source numerical policy differs')
    audit=module(plan['nvml_helper'],'existing_HC_checkpoint_and_NVML')
    mapping={'layers.1.hc_attn_fn':'L1.attnHC.fn','layers.1.hc_attn_base':'L1.attnHC.base','layers.1.hc_attn_scale':'L1.attnHC.scale',
             'layers.1.hc_ffn_fn':'L1.FFNHC.fn','layers.1.hc_ffn_base':'L1.FFNHC.base','layers.1.hc_ffn_scale':'L1.FFNHC.scale','layers.1.ffn_norm.weight':'L1.FFN_norm.weight'}
    params={};checkpoint=[]
    for key,label in mapping.items():
        value,meta=audit.checkpoint_tensors([key]);v=value[key]
        need(all(sha(v['raw'])==actual_params[case][label]['SHA256'] for case in rows),'checkpoint differs from actual owner '+key)
        params[key]=v;checkpoint.append(meta)
    need(sum(len(v['raw']) for v in params.values())<=16*1024**2,'original bounded checkpoint read reserve')
    cp=Path('/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash/config.json');config_raw=cp.read_bytes();config=json.loads(config_raw);cfg={**config,**config.get('text_config',{})}
    need(cfg['rms_norm_eps']==1e-20 and cfg['hc_eps']==1e-6 and cfg['hc_sinkhorn_iters']==20,'actual distinct HC/norm epsilon contract')
    cpu=dict(state='HISTORY36_HC_ACTUAL_INPUT_CPU_PASS',candidate_commit=plan['candidate_commit'],analysis=plan['analysis'],
             actual_request_ids=report['actual_request_ids'],epochs=report['epochs'],owners=owners,raw_exports=raw_pins,
             actual_M=dict(source=1,normal=1794),actual_flags=flags,checkpoint=checkpoint,config_SHA256=sha(config_raw),
             actual_input_and_output_SHA256={case:{k:sha(bits(v)) for k,v in values.items()} for case,values in rows.items()},
             GPU_budget_bytes=LIMIT,planned_tensor_workspace_reserve_bytes=RESERVE,synthetic_other_rows=True,
             model_requests=0,primitive_calls=0,fullbatch_or_expert_scale_equality_claimed=False,whole_model_fixed=False,formal_acceptance=False)
    need(not torch.cuda.is_initialized(),'CPU inspection initialized CUDA')
    return cpu,rows,params,cfg,audit,env


def run(plan,items):
    import torch
    cpu,rows,params,cfg,audit,env=items
    need(json.loads(read(plan['CPU_result']))==cpu,'actual CPU preflight input drift')
    proof=json.loads(read(dict(path=plan['admission_proof'],SHA256=os.environ['STATE_ADMISSION_SHA256'])))
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    need(proof['PID']==os.getpid() and proof['UID']==500 and proof['start_ticks']==fields[19] and
         proof['plan_SHA256']==plan['_SHA256'] and -2<=time.time()-proof['at']<=45 and
         proof['before']['used_v2']==0 and not proof['before']['compute_apps'],'original same-PID pre-runtime empty admission differs')
    for k,v in cpu['actual_flags'].items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v
    import deep_gemm
    from rtp_llm.models_py.modules.dsv41.hc import v41_prenorm as pr
    from rtp_llm.models_py.modules.dsv41.hc.delayed import _tile_ops
    from rtp_llm.models_py.kernels.cuda.mhc_split import mhc_prenorm_num_splits
    need(Path(deep_gemm.__file__).resolve().parent==(Path(env['DG_root'])/'deep_gemm').resolve(),'imported DeepGEMM origin')
    ops=_tile_ops();deep_gemm.use_deterministic_algorithms(True)
    torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();initial=audit.nvml();remaining=LIMIT-initial['used_v2']
    need(remaining>=RESERVE and set(initial['compute_apps'])<={os.getpid()},'unchanged1GiB context/tensor reserve')
    torch.cuda.set_per_process_memory_fraction(remaining/initial['total_v2'],0);torch.cuda.reset_peak_memory_stats()
    def tensor(v,shape):return v.reshape(shape).cuda()
    def param(k):
        v=params['layers.1.'+k];return torch.frombuffer(bytearray(v['raw']),dtype=torch.float32 if v['dtype']=='F32' else torch.bfloat16).reshape(v['shape']).cuda()
    def raw(t):return bits(t.detach().contiguous().cpu())
    memory=[];calls=[];values={}
    def finish(label,data,originals=None):
        torch.cuda.synchronize();v=audit.nvml();need(set(v['compute_apps'])<={os.getpid()} and v['used_v2']<=LIMIT and torch.cuda.max_memory_allocated()<=remaining,'unchanged exclusive1GiB budget')
        memory.append(dict(stage=label,used_v2=v['used_v2'],torch_peak=torch.cuda.max_memory_allocated()))
        checks={k:dict(SHA256=sha(b),byte_equal_original=(b==bits(originals[k])) if originals else None) for k,b in data.items()}
        calls.append(dict(label=label,outputs=checks));values[label]=data
        if originals:need(all(c['byte_equal_original'] for c in checks.values()),'original output calibration failed '+label)
    with torch.inference_mode():
        fn=param('hc_attn_fn').reshape(24,20480);prepared=pr.prepare_tf32_weight(fn);scale=param('hc_attn_scale').reshape(3);base=param('hc_attn_base').reshape(24)
        selected_splits={str(n):mhc_prenorm_num_splits(20480,tokens=n,device=torch.device('cuda:0')) for n in [1,1794]}
        for label,n,weight,case in [('prepared_M1',1,prepared,'source'),('raw_M1794',1794,fn,'normal'),('prepared_M1794',1794,prepared,None)]:
            x=tensor(rows['source']['after_Engram4HC'],(1,1,4,5120)).repeat(1,n,1,1);previous=torch.ones((1,n,4),device='cuda',dtype=torch.float32)
            out=ops.mhc_pre_big_fuse(x,weight,scale,base,rms_eps=cfg['rms_norm_eps'],mhc_pre_eps=cfg['hc_eps'],mhc_sinkhorn_eps=cfg['hc_eps'],
                mhc_post_mult_value=2.0,sinkhorn_repeat=cfg['hc_sinkhorn_iters'],previous_pre_mix=previous,gemm_backend='deepgemm')
            data={key:raw(t.reshape(n,-1)[0]) for key,t in zip(['pre','post','comb'],out[:3])}
            refs={key:rows[case]['natural_attnHC_'+suffix] for key,suffix in [('pre','pre4'),('post','post4'),('comb','comb16')]} if case else None
            finish(label,data,refs);del x,previous,out;gc.collect();torch.cuda.empty_cache()
        weight_boundary_proven=values['prepared_M1794']==values['prepared_M1'] and values['raw_M1794']!=values['prepared_M1794']
        need(weight_boundary_proven,'weight representation alone did not reproduce original coefficients')
        del fn,prepared,scale,base;gc.collect();torch.cuda.empty_cache()
        fn=pr.prepare_tf32_weight(param('hc_ffn_fn').reshape(24,20480));scale=param('hc_ffn_scale').reshape(3);base=param('hc_ffn_base').reshape(24);norm=param('ffn_norm.weight').reshape(5120)
        def mega(label,case,n,mix_case=None,comb_case=None):
            src=rows[case];previous=rows[mix_case or case];combs=rows[comb_case or case]
            x=tensor(src['seam_attn_out5120'],(1,5120)).repeat(n,1);res=tensor(src['seam_residual4HC'],(1,4,5120)).repeat(n,1,1)
            pm=tensor(previous['attnHC_pre4'],(1,4,1)).repeat(n,1,1);post=tensor(src['attnHC_post4'],(1,4,1)).repeat(n,1,1);comb=tensor(combs['attnHC_comb16'],(1,4,4)).repeat(n,1,1)
            nr=torch.empty_like(res);np=torch.empty_like(pm);npost=torch.empty_like(post);nc=torch.empty_like(comb);y=torch.empty_like(x)
            deep_gemm.mega_mhc(x=x,residual=res,shifted_prev_mix=pm,post_mix=post,comb_res_mix=comb,fn=fn,mix_scales=scale,mix_bases=base,
                hc_mult=4,hc_norm_eps=cfg['rms_norm_eps'],hc_pre_eps=cfg['hc_eps'],hc_post_scale=2.0,sinkhorn_eps=cfg['hc_eps'],num_sinkhorn_iters=cfg['hc_sinkhorn_iters'],
                rmsnorm_weight=norm,rmsnorm_eps=cfg['rms_norm_eps'],rmsnorm_scale=1.0,new_residual=nr,new_prev_mix=np,new_post_mix=npost,new_comb_res_mix=nc,y_bf16=y)
            data={key:raw(t[0]) for key,t in [('residual',nr),('pre',np),('post',npost),('comb',nc),('norm',y)]}
            refs={key:rows[case][point] for key,point in [('residual','seam_post4HC'),('pre','FFNHC_pre4'),('post','FFNHC_post4'),('comb','FFNHC_comb16'),('norm','seam_FFN_norm5120')]} if not mix_case and not comb_case else None
            finish(label,data,refs)
        mega('Mega_source', 'source',1);gc.collect();torch.cuda.empty_cache()
        mega('Mega_normal','normal',1794);gc.collect();torch.cuda.empty_cache()
        mega('Mega_only_previous_mix_from_normal','source',1,mix_case='normal');gc.collect();torch.cuda.empty_cache()
        pre_only_reproduced=values['Mega_only_previous_mix_from_normal']['norm']==values['Mega_normal']['norm']
        if not pre_only_reproduced:
            mega('Mega_then_only_comb_from_normal','source',1,mix_case='normal',comb_case='normal');gc.collect();torch.cuda.empty_cache()
        final='Mega_only_previous_mix_from_normal' if pre_only_reproduced else 'Mega_then_only_comb_from_normal'
        need(values[final]==values['Mega_normal'],'captured coefficient replacement did not reproduce complete normal seam outputs')
    return dict(cpu,state='HISTORY36_HC_WEIGHT_BOUNDARY_AND_SEAM_PROVEN',model_requests=0,primitive_calls=len(calls),calls=calls,
                actual_fixed_split_selection=selected_splits,only_weight_representation_changed_at_M1794_proves_HC_delta=weight_boundary_proven,
                only_previous_mix_reproduces_normal_norm=pre_only_reproduced,coefficient_replacement_reproduces_all_normal_seam_outputs=True,
                GPU_memory=memory,operator_root_cause_proven=True,whole_model_fixed=False,formal_acceptance=False,
                limit='Actual target row repeated for other execution rows; no fullbatch/expert/quantized-side-output equivalence. Calibration of all sampled visible outputs precedes attribution. Whole3route model acceptance remains required.')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    plan=json.loads(read(dict(path=str(a.plan),SHA256=a.sha256)));need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'probe source drift')
    need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')==('' if plan['mode']=='check' else '0'),'owned CPU or physicalGPU0 scope')
    root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new immutable result root required');root.mkdir(mode=0o700)
    def save(name,d):
        p=root/(name+'.partial');p.write_text(json.dumps(d,sort_keys=True,indent=2));p.replace(root/name)
    save('status.json',dict(state='VERIFYING_ACTUAL_INPUTS',model_requests=0,at=time.time()))
    try:
        items=inspect(plan);plan['_SHA256']=a.sha256;result=items[0] if plan['mode']=='check' else run(plan,items)
        save('result-safe.json',result);pin=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes()))
        save('status.json',dict(state='SUCCEEDED',model_requests=0,result=pin));print(json.dumps(dict(state=result['state'],model_requests=0,primitive_calls=result['primitive_calls'])))
    except BaseException as error:
        save('status.json',dict(state='FAILED',error_type=type(error).__name__,error=str(error),model_requests=0));raise


if __name__=='__main__':main()
