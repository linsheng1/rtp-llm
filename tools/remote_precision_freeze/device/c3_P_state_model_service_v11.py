"""C3 P-only state diagnostic service, derived from frozen model5 9382bdf7.

New explicit bootstrap/observer profile; no formal acceptance or RPC creation.
Original numeric argv, native artifacts, warmup and finite drain remain bound.

Explicit per-node env/manifest SHA, namespace, publication and group required.
Uses pinned v4 pure configuration and existing finite controller for orderly
root/process_manager drain. READY is actual4GPU ranks+logs+health; KVempty and
whole341/wheel/model correctness stay pending. No requests/Consumer start.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler,build_opener

OWNER='publication-precision-20261101';B1='91b4635faf8c65cc67e66ab52a46f5c932cd08ea'
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
P_HOSTS=['10.44.38.185','10.44.12.175'];D_HOST='10.44.61.71'
V4_SHA='f8eb6abd291edbb3c14b5d1fbd96713e2889b9dea03455bea10473c1a45242dd'
CONTROLLER_SHA='3ae7ea3166a5e88f75ccdc208bd849d37bb2abcfb2a7863358615ff541f3afdc'
NVML_SHA='51c18c45235bce6d2fda8b9b453f799bdc8a970c12fe5a96794990646c2c7760'
NUMERIC_DEFAULTS={
 'DSV41_FUSED_QUERY_QUANT':('1','rtp_llm/models_py/modules/dsv41/_v41_query_quant.py'),
 'DSV41_FUSED_QKV_ROPE_CACHE':('1','rtp_llm/models_py/modules/dsv41/fp8/_v41_qkv_rope_cache_triton.py'),
 'DSV41_FUSED_QB_KV':('0','rtp_llm/models_py/modules/dsv41/fp8/decode/compute_qkv.py'),
 'DSV41_FUSED_READY_QKV':('0','rtp_llm/models_py/modules/dsv41/fp8/decode/ready_qkv.py'),
 'DSV41_CUSTOM_QB':('0','rtp_llm/models_py/modules/dsv41/fp8/_v41_qb_kv.py'),
 'DSV4_FP8_QUANT_KERNEL':('auto','rtp_llm/models_py/kernels/cuda/fp8_kernel/fp8_kernel.py'),
}
NUMERIC=list(NUMERIC_DEFAULTS)
HOOK_MODULES=('rtp_llm.models_py.modules.dsv41.block','rtp_llm.models_py.modules.dsv41.hc.delayed','rtp_llm.models_py.modules.dsv41.moe.moe_layer','rtp_llm.models_py.modules.dsv41.fp8.attention_v41','rtp_llm.models_py.modules.dsv41.hc.v41_prenorm','rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional')
CLIENT_ENV=['TAIR_MEMPOOL_ENABLE_TENT','MC_TENT_CONF','TAIR_MEMPOOL_SOCK','TAIR_MEMPOOL_DATA_SOCK']
STOP=threading.Event();FINAL={'FAILED','TIMED_OUT','CANCELLED','SUCCEEDED','COMMAND_SUCCEEDED'}
DRAIN_RESERVE=420


class ObservationPending(Exception):pass


def require(ok,text):
    if not ok:raise ValueError(text)


def pin(path):
    s=path.stat();digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):digest.update(block)
    t=path.stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns),'file changed while hashing: '+str(path))
    return dict(bytes=t.st_size,SHA256=digest.hexdigest())


def save(path,value):
    p=path.with_name(path.name+'.partial-'+str(os.getpid()))
    with p.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(str(p),str(path))


def read(path,sha=None):
    observed=path.lstat()  # Missing atomic status is an observation race, not corruption.
    require(stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode) and observed.st_size<=4*1024**2,
            'bounded regular JSON required: '+str(path))
    if sha:require(pin(path)['SHA256']==sha,'JSON trusted SHA differs: '+str(path))
    return json.loads(path.read_text())


def load(path,sha,name):
    require(pin(path)['SHA256']==sha,'reviewed helper SHA differs: '+str(path))
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def exact_pin(path,value):require(pin(path)=={k:value[k] for k in ['bytes','SHA256']},'frozen member differs: '+str(path))


def target(a):
    base=load(C/'c3_airline4_model_service_v9.py','03d77ab5fa9e03e1f7af47ff0b049a25eb5b7df12fd6b8c8f41819d4370b7d2d','hcfix_original_model6')
    return base.target(a)



def numeric_profile(environment):
    require(all(k not in environment or isinstance(environment[k],str) for k in NUMERIC),'present numerical flags must be strings')
    return {key:dict(value=environment.get(key,default),source='explicit_target_env' if key in environment else 'verified_source_default',
                     default=default,source_path=relative,source_marker='os.environ.get("'+key+'", "'+default+'")')
            for key,(default,relative) in NUMERIC_DEFAULTS.items()}


def configure(a,e,profile):
    require(a.role=='P' and a.host==a.prefill_host=='10.44.12.175','P12-only diagnostic role required')
    require(a.namespace=='pubprec_state91b_Dorigin_v1' and 600<=a.seconds<=1800,'frozen D-origin diagnostic and original finite<=1800s required')
    base=load(C/'c3_airline4_model_service_v9.py','03d77ab5fa9e03e1f7af47ff0b049a25eb5b7df12fd6b8c8f41819d4370b7d2d','dgdet_reviewed_model8')
    plan=base.configure(a,e,profile)
    plan['schema']='c3-P-state-observer-model-plan-v3'
    plan['service_helper']=dict(path=str(Path(__file__).resolve()),**pin(Path(__file__)))
    args=list(plan['argv'])
    plan['original_model_argv']=list(args)
    observer=observer_inputs(a,plan)
    marker=args.index('rtp_llm.start_server')
    require(args[:marker+1]==[e['interpreter'],'-S','-m','rtp_llm.start_server'],'original -S -m argv differs')
    plan['argv']=[e['interpreter'],'-S',str(a.observer_bootstrap)]+args[marker+1:]
    plan['environment']['P_ONLY_OBSERVER_CONFIG']=str(a.root/'observer-runtime-config.json')
    plan['observer']=observer
    plan['base_service_helper']=dict(path=str(C/'c3_airline4_model_service_v9.py'),SHA256='03d77ab5fa9e03e1f7af47ff0b049a25eb5b7df12fd6b8c8f41819d4370b7d2d')
    plan['classification']='P HC seam transition-v4 observer diagnostic; b1/C1-native/ad62 inner; NOT original uninstrumented model5 or formal acceptance'
    return plan


def observer_inputs(a,plan):
    for path,sha in [(a.observer_template,a.observer_template_sha256),(a.observer_code,a.observer_code_sha256),(a.observer_bootstrap,a.observer_bootstrap_sha256)]:
        require(path is not None and path.is_absolute() and path.parent==C and not path.is_symlink() and re.fullmatch('[0-9a-f]{64}',sha or '') is not None,'explicit CONTROL-local observer input SHA required')
        require(pin(path)['SHA256']==sha,'observer input source differs')
    require(a.observer_code.name=='p_only_state_observer_v8.py' and a.observer_bootstrap.name=='p_only_state_observer_bootstrap_v8.py','reviewed observer sibling names required')
    template=read(a.observer_template,a.observer_template_sha256)
    require(template.get('schema')=='c3-p-state-observer-batch-v4' and len(template.get('cases',[]))==2,'two immutable diagnostic cases required')
    base=C/'p_only_state_observer_v2.py';require(not base.is_symlink() and pin(base)['SHA256']=='d1f4bfb16039ffcc43905fe3ae8bfc0b10dbacebcd895f994200a07845c7fe70','actual immutable observer2 base required')
    cases=template['cases'];require([c.get('expected_prefix_tokens') for c in cases]==[7168,0],'short then full order required')
    require(len({c['nonce'] for c in cases})==2 and len({c['arm_path'] for c in cases})==2 and len({c['identity_ref'] for c in cases})==2,'distinct predeclared case identities required')
    for case in cases:
        require('plan_sha256' not in case,'template cannot contain circular final plan hash')
        require(case.get('role')=='P' and case.get('host')==a.host and case.get('p_cuda_graph') is False and case.get('moedbg')==0,'fixed P eager MOEDBG0 required')
        require(case.get('layers')==[2,8] and case.get('positions')==[7167,7168] and case.get('hidden_width')==5120 and case.get('state_width')==1024,'bounded exact model observation geometry required')
        require(case.get('stage_query_layers')==[2,3,4,5,6,7],'bounded L2-to-L8 stage gap required')
        require(case.get('additional_layers')==[0,1] and case.get('additional_query_positions')==[7168] and case.get('additional_budget_bytes')==4*1024**2,'exact additional L0/1 HC/FFN seam observation within original16MiB required')
        require(case.get('input_ids_sha256')=='07cf16434cfb7964bb1090f9c4c4d4b30823be0ef3bb3ace8c58fd4de913522c' and case.get('runtime_manifest_sha256')==a.manifest_sha256,'exact N7169/model manifest binding required')
        require(re.fullmatch('[0-9a-f]{32,64}',case.get('nonce','')) is not None and 0<case.get('worker_timeout_seconds',120)<=120,'finite one-use nonce profile required')
        for key in ['arm_path','identity_ref','output_dir','batch_budget_dir','evidence_dir']:
            path=Path(case[key]);require(path.is_absolute() and '..' not in path.parts and a.root in path.parents,'all observer outputs must be under this unique model root')
        require(case.get('getter_path')==str(C/'rid-sidecar-P12-stage-v1/c3_cache_store_rid_getter_candidate_v1.so') and case.get('getter_sha256')=='50bc0cf24999fd4010c09a03f506c6fab8b6edad8c7f4a8a6e830bb5e9cd2659','frozen staged typed getter required')
        require(pin(Path(case['getter_path']))['SHA256']==case['getter_sha256'],'actual typed getter differs')
        manifest=read(a.manifest,a.manifest_sha256)
        expected={Path(name).name:row['SHA256'] for name,row in manifest['native'].items()}
        require(case.get('native_sha256')==expected,'actual manifest native fingerprints required')
        hook_expected={name:manifest['production_Python'][name.replace('.','/')+'.py']['SHA256'] for name in HOOK_MODULES}
        require(case.get('hook_source_sha256')==hook_expected,'six actual manifest hook source SHA pins required')
    require(cases[0]['batch_budget_dir']==cases[1]['batch_budget_dir'],'single batch disk budget required')
    return dict(hook_source_sha256=hook_expected,template=dict(path=str(a.observer_template),**pin(a.observer_template)),code=dict(path=str(a.observer_code),**pin(a.observer_code)),bootstrap=dict(path=str(a.observer_bootstrap),**pin(a.observer_bootstrap)),base=dict(path=str(base),**pin(base)),additional_layers=[0,1],additional_query_positions=[7168],additional_budget_bytes=4*1024**2,additional_budget_is_subset_of_total=True,derived_config_path=str(a.root/'observer-runtime-config.json'),derivation='exact template plus cases[*].plan_sha256=SHA256(plan.json); final-config SHA recorded separately to avoid self-reference',model_requests_max=2,layers=[2,8],positions=[7167,7168],gpu_snapshot_bytes_per_rank=16*1024**2,disk_bytes_per_rank=64*1024**2,batch_disk_bytes=512*1024**2)


def prepare_observer(a,plan):
    require(read(a.root/'plan.json')==plan,'frozen diagnostic plan differs')
    template=read(Path(plan['observer']['template']['path']),plan['observer']['template']['SHA256'])
    derived=json.loads(json.dumps(template));plan_sha=pin(a.root/'plan.json')['SHA256']
    for case in derived['cases']:case['plan_sha256']=plan_sha
    path=Path(plan['observer']['derived_config_path'])
    if path.exists():require(read(path)==derived and not path.is_symlink(),'immutable derived observer config differs')
    else:save(path,derived);path.chmod(0o400)
    expected=dict(schema='c3-P-state-observer-config-manifest-v3',plan_SHA256=plan_sha,template=plan['observer']['template'],code=plan['observer']['code'],bootstrap=plan['observer']['bootstrap'],base=plan['observer']['base'],hook_source_sha256=plan['observer']['hook_source_sha256'],derived_config=dict(path=str(path),**pin(path)),classification='diagnostic only; no runtime capture evidence')
    report=a.root/'observer-config-manifest.json'
    if report.exists():require(read(report)==expected,'derived observer manifest differs')
    else:save(report,expected);report.chmod(0o400)



def cache_identity_inputs(a,args,env):
    def arg(key):
        require(args.count(key)==1,'identity input CLI ambiguous: '+key)
        return args[args.index(key)+1]
    enabled=a.publication==1
    cp=int(arg('--prefill_cp_size'));shards=cp if arg('--prefill_cp_kv_cache_sharded')=='1' else 1
    extra=arg('--kvcm_model_extra_info')
    return dict(scope='CONFIGURATION_INPUTS_ONLY_NOT_RUNTIME_CANONICAL_HASH',
        profile=a.cache_profile,publication=enabled,instance_group=arg('--kvcm_instance_group'),
        instance_id_salt=arg('--kvcm_instance_id_salt'),configured_model_type=arg('--model_type'),
        act_type=arg('--act_type'),fp8_kv_cache=arg('--fp8_kv_cache'),
        BIZ_NAME=env['BIZ_NAME'],CHECKPOINT_PATH=env['CHECKPOINT_PATH'],
        model_extra_info=extra,publication_extra_suffix=('/decode-publish-v2/cp='+str(cp)+'/shards='+str(shards)+'/base='+arg('--seq_size_per_block')) if enabled else '',
        wire_tp=cp if enabled else int(arg('--tp_size')),wire_dp=1 if enabled else int(arg('--dp_size')),
        base_tokens=int(arg('--seq_size_per_block')),target_CP=cp,target_shards=shards,
        GDR=env['RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY'],
        actual_canonical_identity='PENDING_MODEL_LOG_AND_MANAGER_QUERY',
        actual_location_specs='PENDING_MODEL_LOG_AND_MANAGER_QUERY',
        causal_sources=['rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopCompletionBridge.inc:12',
                        'rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DtopCompletionBridge.inc:34',
                        'rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.cc:1399'])


def model_args(plan):
    original=plan['original_model_argv'];return original[original.index('rtp_llm.start_server')+1:]


def config_child(a):
    plan=read(a.root/'plan.json');require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU check CUDA must be empty')
    require(sys.version_info[:2]==(3,10) and sys.flags.no_site==1 and os.getuid()==500,'actual UID500/Python3.10-S required')
    runtime=read(Path(plan['runtime_env']['path']),plan['runtime_env']['SHA256'])
    import torch
    def forbid(*args,**kwargs):raise RuntimeError('CPU_CONFIG_CUDA_OR_TCPSTORE_FORBIDDEN')
    torch.cuda._lazy_init=forbid
    import torch.distributed
    torch.distributed.TCPStore=forbid
    require(torch.__version__.startswith('2.11') and Path(torch.__file__).resolve().is_relative_to(Path(runtime['Torch_partial_site']).resolve()),'actual target Torch2.11 source differs')
    import importlib
    auxiliaries={}
    for name in ['torchgen','functorch']:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        require(any(path.is_relative_to(Path(runtime[key]).resolve()) for key in ['Torch_partial_site','Torch_aux_site']),'actual Torch auxiliary source differs: '+name)
        auxiliaries[name]=str(path)
    from rtp_llm.server.server_args.server_args import setup_args
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.distribute.distributed_server import get_master
    cfg=setup_args(model_args(plan));pc=cfg.parallelism_config
    fifo=cfg.runtime_config.fifo_scheduler_config
    require(cfg.model_args.max_seq_len==fifo.max_batch_tokens_size==fifo.max_batch_tokens_without_cache==16384,'official parsed context/batch budget differs')
    require(cfg.runtime_config.warm_up and cfg.runtime_config.model_warm_up,'official global/model kernel warmup must both be1')
    from rtp_llm.utils.warmup import model_warm_up_enabled
    require(model_warm_up_enabled() and os.environ.get('WARM_UP')=='1' and os.environ.get('MODEL_WARM_UP')=='1',
            'official parser must publish effective model/kernel warmup enabled')
    require(bool(cfg.py_hw_kernel_config.enable_cuda_graph)==(a.role=='D'),'official role CUDAgraph differs')
    require(cfg.prefill_cp_config.prefill_cp_size==4 and cfg.prefill_cp_config.kv_cache_sharded,'official CP4/sharded input config differs')
    require((pc.world_size,pc.world_rank,pc.local_world_size)==(4,0,4),'official4/0/4 differs')
    require((pc.tp_size,pc.dp_size,pc.ep_size)==((4,1,4) if a.role=='P' else (1,4,4)),'official TP/DP/EP differs')
    ranks=[]
    for rank in range(4):
        set_parallelism_config(pc,rank,cfg.ffn_disaggregate_config,cfg.prefill_cp_config)
        ranks.append(dict(world_rank=pc.world_rank,local_rank=pc.local_rank,tp_rank=pc.tp_rank,dp_rank=pc.dp_rank))
    set_parallelism_config(pc,0,cfg.ffn_disaggregate_config,cfg.prefill_cp_config);ip,port=get_master(cfg.distribute_config,pc)
    require(ip==a.host and port=='' and not torch.cuda.is_initialized(),'official master/CPU config differs')
    save(a.root/'config-probe.json',dict(state='CPU_MODEL_CONFIG_PASS_NO_MODEL',exit_code=0,CUDA_initialized=False,TCPStore_constructed=False,
         ranks=ranks,master_ip=ip,plan_SHA256=pin(a.root/'plan.json')['SHA256'],parser=setup_args.__code__.co_filename,
         Torch=dict(path=torch.__file__,version=torch.__version__,auxiliaries=auxiliaries),
         parsed_profile=dict(max_seq_len=cfg.model_args.max_seq_len,max_batch_tokens_size=fifo.max_batch_tokens_size,
              max_batch_tokens_without_cache=fifo.max_batch_tokens_without_cache,warm_up=bool(cfg.runtime_config.warm_up),
              model_warm_up=bool(cfg.runtime_config.model_warm_up),effective_model_warm_up=model_warm_up_enabled(),
              extra_startup_grpc_probes_enabled=os.environ.get('DSV4_STARTUP_REAL_WARMUP')=='1',CUDAgraph=bool(cfg.py_hw_kernel_config.enable_cuda_graph),
              prefill_CP_size=cfg.prefill_cp_config.prefill_cp_size,prefill_sharded=bool(cfg.prefill_cp_config.kv_cache_sharded))))
    return 0


def consumer(a,controller,service,required_seconds=None):
    ready=read(a.consumer_root/'ready.json');status=read(a.consumer_root/'status.json');expected=ready['child'];actual=controller.identity(expected['PID'])
    require(ready['state']=='READY' and status['state']=='READY' and controller.same(expected,status['child']) and controller.same(expected,actual)
            and actual['UID']==500 and actual['state'] not in ('Z','X','x') and time.time()-status['heartbeat_at']<=45,'Consumer current READY/UID/start/heartbeat differs')
    require(status['deadline_epoch']>time.time()+(a.seconds+15 if required_seconds is None else required_seconds),'Consumer lease insufficient for this bounded model/drain')
    env=ready['consumer_environment'];ipc=Path(env['TAIR_MEMPOOL_SOCK']).parent
    require(ipc==a.consumer_root/'ipc' and env['TAIR_MEMPOOL_DATA_SOCK']==str(ipc/'d') and Path(env['MC_TENT_CONF'])==a.consumer_root/'tent.json','Consumer new IPC/TENT binding differs')
    require(service.sockets_owned(actual,ipc) is not None and service.ready_log(a.host,ready['ready_log_excerpt']),'actual Consumer owned sockets/RDMA evidence absent')
    return env,dict(ready=dict(path=str(a.consumer_root/'ready.json'),**pin(a.consumer_root/'ready.json')),child=actual,deadline_epoch=status['deadline_epoch'])


def keeper(a,controller,required_seconds=None):
    base=load(C/'c3_airline4_model_service_v9.py','03d77ab5fa9e03e1f7af47ff0b049a25eb5b7df12fd6b8c8f41819d4370b7d2d','hcfix_original_model6')
    return base.keeper(a,controller,required_seconds)



def descendant(current,root,controller):
    chain=[];seen=set()
    for _ in range(64):
        require(current and current['PID'] not in seen,'ancestry missing/cycle');seen.add(current['PID']);chain.append(current)
        if current['PID']==root['PID']:require(controller.same(root,current),'model root PID reused');return chain
        require(current['PPID']>1,'GPU PID is not model descendant: '+json.dumps(dict(sampled_PID=chain[0]['PID'],model_root_PID=root['PID'],chain=[{k:v.get(k) for k in ['PID','UID','PPID','start_ticks','state']} for v in chain])));current=controller.identity(current['PPID'])
    raise ValueError('ancestry budget exceeded')


def nvml(a,env,timeout=5):
    p=subprocess.run([env['interpreter'],'-S',str(a.nvml_helper),'--inventory-child'],env=dict(env['environment'],CUDA_VISIBLE_DEVICES=''),
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if p.returncode!=0:raise ObservationPending('NVML sample unavailable')
    value=json.loads(p.stdout)
    require(len(value['GPUs'])==4 and all(v['capability']==[10,3] for v in value['GPUs']),'actual four SM103 required');return value


def gpu_admission(value,consumer_pid):
    require(len(value['GPUs'])==4 and all(v['capability']==[10,3] and set(v['compute_apps'])<={consumer_pid} and v['used_bytes']<=4*1024**3 for v in value['GPUs']),
            'unknown GPU app, wrongSM or>4GiB/card before model')


def ranks(value,root,consumer_pid,controller):
    rows={};seen=set()
    for gpu in value['GPUs']:
        for pid in gpu['compute_apps']:
            if pid==consumer_pid:continue
            if pid in seen:continue
            seen.add(pid);current=controller.identity(pid);require(current and current['UID']==500,'unknown GPU PID/UID')
            chain=descendant(current,root,controller)
            comm=(Path('/proc')/str(pid)/'comm').read_text().strip();match=re.fullmatch('rtp_llm_rank-([0-3])',comm)
            if match:rows[pid]=dict(current,world_rank=int(match.group(1)),ancestry=chain)
    if len(rows)!=4 or {v['world_rank'] for v in rows.values()}!={0,1,2,3}:return None
    return list(rows.values())


def check(a,e,plan):
    work=a.root/'config-runtime';(work/'logs').mkdir(parents=True)
    env=dict(plan['environment'],CUDA_VISIBLE_DEVICES='',LOG_PATH=str(work/'logs'),NCCL_DEBUG_FILE=str(work/'logs/nccl-%h-%p.log'));argv=[e['interpreter'],'-S',str(Path(__file__).resolve()),'--config-child',
        '--root',str(a.root),'--role',a.role,'--host',a.host]
    timed_out=False
    with (a.root/'config-child.log').open('xb') as log:
        p=subprocess.Popen(argv,env=env,cwd=str(work),stdout=log,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out=True;p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            rc=p.returncode
    save(a.root/'config-exit.json',dict(actual_child_returncode=rc,timed_out=timed_out,plan_SHA256=pin(a.root/'plan.json')['SHA256']))
    require(not timed_out and rc==0,'CPU config child timeout/abnormal/nonzero; see config-child.log');probe=read(a.root/'config-probe.json')
    require(probe['exit_code']==0 and probe['plan_SHA256']==pin(a.root/'plan.json')['SHA256'],'CPU config probe binding differs')


def serve(a,e,plan,controller,service,n):
    out=dict(state='BLOCKED',role=a.role,host=a.host,namespace=a.namespace,publication=a.publication,classification=plan['classification'],
             model_READY=False,KV_empty_proof='PENDING_SEPARATE_OBSERVER',rank_exit_codes='UNKNOWN_NOT_DIRECT_CHILDREN',actual_controller_returncode=None,
             deadline_epoch=time.time()+a.seconds,drain_reserve_seconds=DRAIN_RESERVE)
    out['observer_config_manifest']=dict(path=str(a.root/'observer-config-manifest.json'),**pin(a.root/'observer-config-manifest.json'))
    p=None;code=2;failure=None;ready=False;seen_ranks=[];opener=build_opener(ProxyHandler({}));absolute=time.monotonic()+a.seconds
    try:
        require(not (a.root/'model-status.json').exists() and not (a.root/'model-spec.json').exists(),'new model batch/root required; no restart or hot change')
        probe=read(a.root/'config-probe.json');exit=read(a.root/'config-exit.json')
        require(exit['actual_child_returncode']==0 and not exit.get('timed_out') and exit['plan_SHA256']==probe['plan_SHA256']==pin(a.root/'plan.json')['SHA256'] and probe['state']=='CPU_MODEL_CONFIG_PASS_NO_MODEL','natural CPU config PASS required')
        consumer_env,consumer_proof=consumer(a,controller,service);shm=keeper(a,controller);n.compiler_guard()
        before=nvml(a,e)
        gpu_admission(before,consumer_proof['child']['PID'])
        for port in [18079,18080,18081,18089,18098,18107]:
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        uds=Path(plan['environment']['RTP_LLM_CPU_TP_BROADCASTER_DIR']);require(not uds.exists() and not uds.is_symlink(),'new short UDS directory required');uds.mkdir(mode=0o700)
        env=dict(plan['environment'])
        for key in CLIENT_ENV:
            require(env[key]==consumer_env[key],'frozen model/actual Consumer client binding differs: '+key)
        spec=dict(argv=plan['argv'],env=env,cwd=str(a.root/'runtime'),timeout_seconds=max(1,a.seconds-DRAIN_RESERVE),parent_first=True,
                  terminate_grace_seconds=300,descendant_terminate_grace_seconds=30,kill_grace_seconds=5,success_requires='exit_zero',
                  failure_patterns=['completion_stopped=1','KVCM quarantined','retaining pending session'],dependents=['requests'])
        save(a.root/'model-spec.json',spec);out.update(Consumer=consumer_proof,SHM=shm,NVML_before=before,plan_SHA256=pin(a.root/'plan.json')['SHA256'])
        argv=[e['interpreter'],'-S',str(a.controller),'--root',str(a.root),'--name','model','--interval','20','run','--spec',str(a.root/'model-spec.json')]
        with (a.root/'controller.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=str(a.root),stdout=log,stderr=subprocess.STDOUT)
        out.update(state='WAITING_REAL_MODEL_ROOT',controller_PID=p.pid);save(a.root/'model-service-status.json',out)
        first_log=None;next_sample=0
        while p.poll() is None:
            if STOP.is_set() or (a.root/'stop').exists() or time.monotonic()>=absolute-DRAIN_RESERVE:(a.root/'model-stop').touch(exist_ok=True)
            if time.monotonic()>=absolute:raise RuntimeError('total model-service deadline reached during drain')
            if time.monotonic()<next_sample:STOP.wait(min(1,max(0,next_sample-time.monotonic())));continue
            next_sample=time.monotonic()+20
            try:
                status=read(a.root/'model-status.json')
                out['deadline_warning']=bool(status.get('deadline_warning'))
                if status['state']=='FAILED' and status.get('cleanup_state')!='FINISHED':
                    out.update(state='FAILED_DRAINING_OWN_MODEL',controller_failure_reason=status.get('reason'),heartbeat_at=time.time())
                    save(a.root/'requests-blocked.json',dict(state='BLOCKED',cause=status.get('reason'),at=time.time()));save(a.root/'model-service-status.json',out);continue
                if status['state'] in FINAL:break
                if status['state'] in ('STOPPING','FAILED'):out['state']='DRAINING_OWN_MODEL';save(a.root/'model-service-status.json',out);continue
                remaining_dependency=max(0,absolute-time.monotonic())+15
                consumer(a,controller,service,remaining_dependency)
                # D keeper protects constructor reuse; current FD lease remains
                # required during this bounded initial inner service.
                if a.weights_mode=='retained':keeper(a,controller,remaining_dependency)
                root=status.get('child')
                if root:
                    actual=controller.identity(root['PID']);require(controller.same(root,actual) and actual['UID']==500,'actual model root identity differs')
                    out.update(state='MODEL_PROCESS_EXISTS_NOT_READY',model_root=actual,model_READY=False)
                    sample=nvml(a,e);save(a.root/'startup-nvml-observation.json',dict(observed_at=time.time(),NVML=sample,model_root=actual,consumer_child=consumer_proof['child']));owned=ranks(sample,actual,consumer_proof['child']['PID'],controller)
                    for path in [a.root/'model-stage.log',a.root/'runtime/logs/engine.log']:
                        if path.is_file() and path.stat().st_size:
                            with path.open('rb') as f:
                                head=f.read(128*1024);f.seek(max(0,path.stat().st_size-256*1024));tail=f.read()
                            text=(head+b'\n'+tail).decode(errors='replace')
                            match=re.search(r'.{0,120}(?:load(?:ed|ing)?\s+weights?|weights?.{0,30}load|loadWeight|load.*model).{0,160}',text,re.I)
                            if match and first_log is None:first_log=dict(path=str(path),excerpt=match.group(0),observed_at=time.time())
                    health=[]
                    for port in ([18080] if a.role=='P' else [18080,18089,18098,18107]):
                        try:
                            with opener.open('http://127.0.0.1:'+str(port)+'/health',timeout=2) as r:health.append(dict(port=port,status=r.status))
                        except OSError:health.append(dict(port=port,status='PENDING'))
                    out.update(heartbeat_at=time.time(),GPU_ranks=owned,health=health,first_model_weight_log=first_log)
                    if owned and first_log and all(v['status']==200 for v in health):
                        require(controller.same(actual,controller.identity(actual['PID'])) and all(controller.same(row,controller.identity(row['PID'])) for row in owned),'root/rank identity changed before READY publication')
                        seen_ranks=owned;out.update(state='MODEL_READY_KV_EMPTY_PENDING',model_READY=True)
                        if not ready:save(a.root/'model-ready.json',out);ready=True;out['model_was_READY']=True
                    save(a.root/'model-service-status.json',out)
            except FileNotFoundError:pass
            except (ObservationPending,subprocess.TimeoutExpired) as exc:
                out.update(state='MODEL_OBSERVATION_PENDING',model_READY=False,heartbeat_at=time.time(),observation_pending=str(exc)[:1000]);save(a.root/'model-service-status.json',out)
        p.wait(timeout=5);out['actual_controller_returncode']=p.returncode;final=read(a.root/'model-status.json');out['controller_final']=final
        if final['state'] in ('CANCELLED','TIMED_OUT'):require(final.get('remaining_owned_processes')==[],'owned model descendants not explicitly drained')
        require(not final.get('remaining_owned_processes'),'owned model descendants remain after cleanup')
        require(final['state'] in ('CANCELLED','TIMED_OUT','SUCCEEDED') and final['command_returncode'] in (0,-signal.SIGTERM),'model root/controller failed; retain actual rc')
        leftovers=[]
        for row in seen_ranks:
            actual=controller.identity(row['PID'])
            if controller.same(row,actual) and actual.get('state') not in ('Z','X','x'):leftovers.append(actual)
        require(not leftovers,'a recorded GPU rank remains live after controller termination')
        require(ready,'bounded startup ended without actual MODEL_READY; no retry or timeout relaxation')
        out.update(state='MODEL_SERVICE_EXITED',actual_model_root_returncode=final['command_returncode'],rank_exit_codes={str(v['PID']):'UNKNOWN_NOT_DIRECT_CHILD' for v in seen_ranks},
                   complete_native_exit_acceptance=False,termination_observation='owned tree gone; root rc observed; per-rank rc unknown');code=0
    except BaseException as exc:
        failure=str(exc);out.update(state='FAILED' if p else 'BLOCKED',error_type=type(exc).__name__,error=failure[:2048]);code=1 if p else 2
        save(a.root/'requests-blocked.json',dict(state='BLOCKED',cause=out['error'],at=time.time()))
        if p and p.poll() is None:
            (a.root/'model-stop').touch(exist_ok=True)
            try:p.wait(timeout=max(1,min(400,absolute-time.monotonic())))
            except subprocess.TimeoutExpired:out['controller_drain_unfinished']=True
        if p:out['actual_controller_returncode']=p.returncode
    out.update(exit_code=code,finished_at=time.time());save(a.root/'model-service-final.json',out);save(a.root/'model-service-status.json',out)
    print(json.dumps({k:out.get(k) for k in ['state','model_READY','actual_controller_returncode','error']}),flush=True);return code


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ['observer-template','observer-code','observer-bootstrap']:
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-sha256')
    p.add_argument('--mode',choices=['compose','check','run'],default='compose');p.add_argument('--config-child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['P','D'],required=True);p.add_argument('--host',required=True)
    p.add_argument('--prefill-host',choices=P_HOSTS);p.add_argument('--namespace');p.add_argument('--publication',type=int,choices=[0,1]);p.add_argument('--instance-group');p.add_argument('--cache-profile',choices=['reference','publication-candidate'])
    for name in ['env-json','manifest']:p.add_argument('--'+name,type=Path)
    for name in ['env-sha256','manifest-sha256']:p.add_argument('--'+name)
    p.add_argument('--seconds',type=int,default=7200);p.add_argument('--profile',type=Path,default=C/'airline4_v4_pure_profile_v1.py')
    p.add_argument('--controller',type=Path,default=C/'device_controller_cluster23_v1.py');p.add_argument('--nvml-helper',type=Path,default=C/'portable_h127_attention_inner.py')
    p.add_argument('--consumer-helper',type=Path,default=C/'c3_consumer_inner_service.py');p.add_argument('--consumer-helper-sha256')
    p.add_argument('--consumer-root',type=Path,default=C/'consumer-business-v2');p.add_argument('--weights-mode',choices=['cold','retained']);p.add_argument('--keeper-root',type=Path);p.add_argument('--keeper-helper',type=Path);p.add_argument('--keeper-helper-sha256')
    a=p.parse_args();os.umask(0o077)
    if a.config_child:return config_child(a)
    require(os.getuid()==500 and a.root.parent==C,'actual UID500/new direct CONTROL child required')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:s.bind((a.host,0))
    require(all(getattr(a,k) is not None for k in ['prefill_host','namespace','publication','cache_profile','weights_mode','instance_group','env_json','env_sha256','manifest','manifest_sha256']),'all explicit profile and per-node input pins required')
    e,m=target(a);profile=load(a.profile,V4_SHA,'airline4_v4_pure');plan=configure(a,e,profile)
    if a.root.exists():
        require(a.root.is_dir() and not a.root.is_symlink() and a.root.stat().st_uid==500 and (a.root/'.owner').read_text().strip()==OWNER,'existing plan root owner differs')
        require(read(a.root/'plan.json')==plan,'new batch/root required for any configuration change')
    else:
        a.root.mkdir(mode=0o700);(a.root/'.owner').write_text(OWNER+'\n');(a.root/'runtime/logs').mkdir(parents=True)
        (a.root/'runtime/site').symlink_to(Path(e['site']),target_is_directory=True)
        (a.root/'runtime/cuda-wrap').symlink_to(Path(e['environment']['CUDA_HOME']),target_is_directory=True);save(a.root/'plan.json',plan)
    prepare_observer(a,plan)
    if a.mode=='compose':print(json.dumps(dict(state='MODEL_PLAN_FROZEN_NOT_STARTED',plan_SHA256=pin(a.root/'plan.json')['SHA256'])));return 0
    if a.mode=='check':check(a,e,plan);return 0
    require(a.consumer_helper_sha256 is not None,'explicit currently reviewed Consumer helper SHA required for run')
    controller=load(a.controller,CONTROLLER_SHA,'airline4_finite_controller');n=load(a.nvml_helper,NVML_SHA,'airline4_nvml')
    service=load(a.consumer_helper,a.consumer_helper_sha256,'airline4_consumer_evidence')
    for number in [signal.SIGTERM,signal.SIGINT]:signal.signal(number,lambda *args:STOP.set())
    return serve(a,e,plan,controller,service,n)


if __name__=='__main__':raise SystemExit(main())
