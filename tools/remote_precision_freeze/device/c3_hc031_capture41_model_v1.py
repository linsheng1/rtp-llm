"""C3 STATE91b/ad62 model inner: original role/runtime guards, full-content weight proof.

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

OWNER='publication-precision-20261101';B1='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c'
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
    e=read(a.env_json,a.env_sha256);m=read(a.manifest,a.manifest_sha256)
    require(e['schema']=='c3-dg-batch-invariant-ad62-target-inner-env-v1' and m['state']=='C3_DG_BATCH_INVARIANT_AD62_INNER_PREPARED_NOT_IMPORTED'
            and m['source_commit']==B1 and m['inner_only'] is True and m['immutable_wheel'] is False and m['full_341_candidate_accepted'] is False
            and m['Cpp_diff_empty'] is True and m['base_native_source']=='c1b4b19d82f56507f89be86dddf90497501d32a6'
            and m['host']==a.host,'per-node target b1/ad62 inner contract differs')
    require(m['environment']['path']==str(a.env_json) and m['environment']['SHA256']==a.env_sha256,'manifest/env binding differs')
    require(m['DeepGEMM_batch_invariant'] is True and (Path(e['site'])/'rtp_llm/models_py/model_desc/deepseek_v41_base_model.py').read_text().count('deep_gemm.use_deterministic_algorithms(True)')==1,'frozen batch-invariant initialization required')
    site=Path(e['site']);require(site==a.manifest.parent/'site' and e['environment']['PYTHONPATH'].split(':')[0]==str(site),'new target site must be first')
    require(all(k not in e['environment'] or isinstance(e['environment'][k],str) for k in NUMERIC),'present DSV41 target flags must be strings')
    require('c3-runtime-ff13-remote/site' not in e['environment']['PYTHONPATH'],'old ff13 site cannot reenter PYTHONPATH')
    for name,value in m['production_Python'].items():exact_pin(site/name,value)
    # b467 target has none of these six flags. Preserve the actual normal
    # os.environ.get defaults, after checking them in the fully pinned source.
    for key,(default,relative) in NUMERIC_DEFAULTS.items():
        marker='os.environ.get("'+key+'", "'+default+'")'
        require(relative in m['production_Python'] and marker in (site/relative).read_text(),
                'pinned normal source default differs: '+key)
    for name,value in m['support'].items():exact_pin(site/name,value)
    for name,value in m['native'].items():exact_pin(site/name,value)
    dg=Path(e['DG_root'])/'deep_gemm'
    for name,value in m['DG_code_members'].items():exact_pin(dg/name,value)
    for key in ['nvcc_wrapper','patch_kernel']:exact_pin(Path(m[key]['path']),m[key])
    require(e['environment']['CUDA_HOME']==str(Path(m['nvcc_wrapper']['path']).parent.parent)
            and e['environment']['PATH'].split(':')[0]==str(Path(m['nvcc_wrapper']['path']).parent),'matching target CUDA wrapper contract differs')
    return e,m


def numeric_profile(environment):
    require(all(k not in environment or isinstance(environment[k],str) for k in NUMERIC),'present numerical flags must be strings')
    return {key:dict(value=environment.get(key,default),source='explicit_target_env' if key in environment else 'verified_source_default',
                     default=default,source_path=relative,source_marker='os.environ.get("'+key+'", "'+default+'")')
            for key,(default,relative) in NUMERIC_DEFAULTS.items()}


def configure(a,e,profile):
    require(a.host in (P_HOSTS if a.role=='P' else [D_HOST]) and a.prefill_host in P_HOSTS,'role/host mapping differs')
    require(a.role!='P' or a.host==a.prefill_host,'P must route as its actual host')
    require(a.cache_profile in ('reference','publication-candidate'),'explicit reference/publication-candidate cache profile required')
    require(a.publication==int(a.cache_profile=='publication-candidate'),'reference requires publication0 on every role; candidate requires publication1 on every role')
    require(isinstance(a.namespace,str) and re.fullmatch('[A-Za-z0-9_-]{8,100}',a.namespace) and a.namespace not in ('pubprec_20261101_pilot_dorigin_v1','pubprec_20261101_candidate_dorigin_v1'),'explicit new namespace required')
    require(a.instance_group=='c3_on_B_20261030','only existing explicit C3 instance group supported')
    require(600<=a.seconds<=7200,'finite600..7200s including drain budget required')
    require(a.weights_mode in ('cold','retained'),'explicit cold/retained static weights mode required')
    require(a.weights_mode!='cold' or a.role=='P','D requires retained static weights')
    if a.weights_mode=='retained':
        require(a.keeper_root is not None and a.keeper_helper is not None and re.fullmatch('[0-9a-f]{64}',a.keeper_helper_sha256 or '') is not None,'retained weights require explicit keeper root/helper/SHA')
        require(a.keeper_root.parent==C and a.keeper_helper.parent==C,'retained keeper evidence/helper must be direct CONTROL children')
    else:require(a.keeper_root is None and a.keeper_helper is None and a.keeper_helper_sha256 is None,'cold must not inherit a keeper binding')
    profile.HOSTS={'P':a.prefill_host,'D':D_HOST};profile.NAMESPACE=a.namespace;profile.R=Path(e['site']).parent
    args=profile.build_argv(a.role)
    replacements={'--max_seq_len':'16384','--max_batch_tokens_size':'16384','--max_batch_tokens_without_cache':'16384',
                  '--warm_up':'1','--model_warm_up':'1','--kvcm_instance_group':a.instance_group,
                  '--kvcm_instance_id_salt':a.namespace,'--kvcm_model_extra_info':'publication_precision_20261101_state91b_ad62'}
    for key,value in replacements.items():require(args.count(key)==1,'profile CLI flag ambiguous: '+key);args[args.index(key)+1]=value
    args=[e['interpreter'],'-S']+args[1:]
    log=a.root/'runtime/logs';inherited=dict(e['environment'])
    # HTTP_TIMEOUT is carried unchanged if present; no timeout relaxation here.
    if 'HTTP_TIMEOUT' not in inherited and 'HTTP_TIMEOUT' in os.environ:inherited['HTTP_TIMEOUT']=os.environ['HTTP_TIMEOUT']
    env,removed=profile.build_env(a.role,log,inherited=inherited,pythonpath=e['environment']['PYTHONPATH'],ldpath=e['environment']['LD_LIBRARY_PATH'])
    for key in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME','TMPDIR']:env[key]=e['environment'][key]
    numeric=numeric_profile(e['environment']);env.update({key:row['value'] for key,row in numeric.items()})
    env.pop('DSV4_MHC_NUM_SPLITS',None)
    env.update(DSV4_MHC_SPLIT_MODE='fixed',DSV4_STARTUP_REAL_WARMUP='0',
               RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH=str(a.publication),RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY='1',
               RTP_LLM_CPU_TP_BROADCASTER_DIR='/tmp/c3a4-'+hashlib.sha256(str(a.root).encode()).hexdigest()[:12],
               CUDA_VISIBLE_DEVICES='0,1,2,3')
    env.update(TAIR_MEMPOOL_ENABLE_TENT='1',MC_TENT_CONF=str(a.consumer_root/'tent.json'),
               TAIR_MEMPOOL_SOCK=str(a.consumer_root/'ipc/c'),TAIR_MEMPOOL_DATA_SOCK=str(a.consumer_root/'ipc/d'))
    routing=json.loads(env['MODEL_SERVICE_CONFIG']);routing['service_id']='c3_airline4_state91b_inner_ad62';env['MODEL_SERVICE_CONFIG']=json.dumps(routing)
    plan=dict(schema='c3-airline4-inner-model-plan-v7',source_commit=B1,DeepGEMM_batch_invariant=True,role=a.role,host=a.host,prefill_host=a.prefill_host,
                namespace=a.namespace,publication=a.publication,instance_group=a.instance_group,argv=args,environment=env,
                env_SHA256=a.env_sha256,manifest_SHA256=a.manifest_sha256,numeric_flags={k:env[k] for k in NUMERIC},numeric_flag_provenance=numeric,
                runtime_env=dict(path=str(a.env_json),SHA256=a.env_sha256),runtime_manifest=dict(path=str(a.manifest),SHA256=a.manifest_sha256),
                service_helper=dict(path=str(Path(__file__).resolve()),**pin(Path(__file__))),
                lifetime_seconds=a.seconds,drain_reserve_seconds=DRAIN_RESERVE,model_active_seconds=a.seconds-DRAIN_RESERVE,
                fresh_local_KV_empty_proof='PENDING_SEPARATE_OBSERVER',profile=a.cache_profile,cache_identity_contract=cache_identity_inputs(a,args,env),
                weights_mode=a.weights_mode,keeper_binding=None if a.weights_mode=='cold' else dict(root=str(a.keeper_root),helper=str(a.keeper_helper),helper_SHA256=a.keeper_helper_sha256),
                startup_profile=dict(label='normal-kernel-warmup inner; extraRPCwarmupdisabled',warm_up=1,model_warm_up=1,DSV4_STARTUP_REAL_WARMUP=0,
                    causal_sources=['rtp_llm/utils/warmup.py:model_warm_up_enabled','rtp_llm/models_py/modules/dsv41/fp8/_v41_attention_jit_warmup.py:1531','rtp_llm/models_py/modules/dsv41/fp8/_v41_joint_pool.py:500'],
                    kernel_and_pool_preparation_enabled=True,extra_startup_grpc_probes_enabled=False),
                classification='b1/C1-native/ad62 inner; NOT full341/wheel/model-correctness acceptance',cleared_discovery=removed)

    config=history_contract(a.history_config,a.history_config_sha256,a.role,a.host,a.env_sha256,a.manifest_sha256)
    original=list(plan['argv']);marker=original.index('rtp_llm.start_server')
    require(original[:marker+1]==[e['interpreter'],'-S','-m','rtp_llm.start_server'],'original startup argv differs')
    plan['original_model_argv']=original
    plan['argv']=[e['interpreter'],'-S',str(C/'history41_bootstrap_v1.py')]+original[marker+1:]
    plan['environment'].update(HISTORY_CAPTURE_CONFIG=str(a.history_config),HISTORY_CAPTURE_CONFIG_SHA256=a.history_config_sha256)
    plan['history_capture']=dict(config=dict(path=str(a.history_config),**pin(a.history_config)),profile=config,
        observer=dict(path=str(C/'history41_observer_v1.py'),**pin(C/'history41_observer_v1.py')),
        bootstrap=dict(path=str(C/'history41_bootstrap_v1.py'),**pin(C/'history41_bootstrap_v1.py')),
        latches=dict(path=str(C/'history38_latches_v1.py'),**pin(C/'history38_latches_v1.py')),
        GPU_RID_latch=False,controller_exclusive_RPC_required=True)
    plan['classification']='history41 selected historical positions; graph settings unchanged; inner diagnostic only'
    return plan


def history_contract(path,sha,role,host,env_sha,manifest_sha):
    require(path is not None and path.parent==C and sha is not None,'explicit owned history config required')
    cfg=read(path,sha)
    require(cfg['schema']=='history41-inner-graph-latches-v1' and cfg['source_commit']==B1
        and cfg['role']==role and cfg['host']==host and cfg['environment_SHA256']==env_sha
        and cfg['runtime_manifest_SHA256']==manifest_sha and cfg['wheel_sha256'] is None
        and cfg['artifact_kind']=='inner_python_overlay','actual history candidate/role/runtime differs')
    require(cfg['positions']==[7045] and cfg['layers']==list(range(26)) and cfg['gpu_budget_bytes']==4194304,
        'minimal historical capture profile only')
    require(Path(cfg['output']).parent==C and Path(cfg['output']).name==('h41-'+role),'private short history output required')
    for name in ['history41_observer_v1.py','history41_bootstrap_v1.py','history38_latches_v1.py']:
        require(pin(C/name)['SHA256']==cfg['control_source_SHA256'][name],'actual history helper drift')
    require(cfg['latches_SHA256']==cfg['control_source_SHA256']['history38_latches_v1.py'],'latches source pin differs')
    runtime=read(Path(cfg['manifest_path']),manifest_sha)
    for name,sha in cfg['hook_source_sha256'].items():
        rel=name.replace('.','/')+'.py'
        require(runtime['production_Python'][rel]['SHA256']==sha,'actual historical production fingerprint differs:'+rel)
    return cfg


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
    argv=plan['original_model_argv']
    return argv[argv.index('rtp_llm.start_server')+1:]


def keeper(a,controller,required_seconds=None):
    # Old READY proof expired. Reuse requires NEW full checkpoint-vs-table hash,
    # plus unchanged live FD/SH-flock/UID/start/heartbeat and entire-lease gates.
    require(a.weights_mode=='retained','this candidate uses full-content revalidated static weights')
    path=C/'revalidate_hc031_static_weights.py'
    require(a.keeper_helper==path and a.keeper_helper_sha256=='090521ba16350783e0a7b8d3b553f8f225e2131f2d7c551f68ae057fb460e868','exact full-content keeper helper required')
    helper=load(path,a.keeper_helper_sha256,'full_content_static_weight_admission')
    return helper.audit(a,controller,required_seconds)


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


def _original_static_keeper(a,controller,required_seconds=None):
    if a.weights_mode=='cold':
        require(a.role=='P','cold static weights allowed only for P')
        minimum=200*1024**3;proof=dict(mode='cold_static_allocation_no_keeper')
    else:
        require(a.weights_mode=='retained' and a.keeper_root is not None and a.keeper_helper is not None,'explicit retained keeper required')
        require(pin(a.keeper_helper)['SHA256']==a.keeper_helper_sha256,'retained keeper reviewed helper SHA differs')
        r=read(a.keeper_root/'ready.json');s=read(a.keeper_root/'status.json');expected=r['process'];actual=controller.identity(expected['PID'])
        for value in [r,s]:
            require(value['schema']=='c3-generic-static-engram-keeper-v3' and value['role']==a.role and value['host']==a.host
                    and value['weight_bytes_read']==0 and value['full_weight_content_hashed'] is False
                    and value['CUDA_initialized'] is False and value['KV_cache_held'] is False
                    and value['model_started'] is False and value['files_unlinked'] is False,'retained keeper role/host/static-only evidence differs')
        require(s['state']=='RUNNING' and r['state']=='STATIC_ENGRAM_MODEL_WEIGHT_LEASES_HELD' and controller.same(expected,s['process'])
                and controller.same(expected,actual) and actual['UID']==500 and actual['state'] not in ('Z','X','x')
                and 0<=time.time()-s['heartbeat_at']<=45,'retained keeper process/epoch/heartbeat differs')
        require(str(a.keeper_helper) in actual['argv'],'actual keeper process is not the reviewed helper path')
        require(r['deadline_epoch']==s['deadline_epoch'] and r['deadline_epoch']>time.time()+(a.seconds+15 if required_seconds is None else required_seconds),
                'retained keeper lease insufficient for full model/drain')
        require(r['source_model_may_exit_after_keeper_READY'] is True and r['source_manifest']==s['source_manifest']
                and r['provenance']==s['provenance'],'retained keeper source/provenance binding differs')
        source=read(Path(r['source_manifest']['path']),r['source_manifest']['SHA256'])
        require(source['schema']=='c3-model-engram-source-v3' and source['role']==a.role and source['host']==a.host
                and source['weight_bytes_read']==0 and source['classification']=='current-ready-loader-and-SH-lease-proof; no full weight hash',
                'retained weights require the actual ready-model adoption evidence; no implicit orphan adoption')
        provenance=r['provenance'];manifest=read(a.manifest,a.manifest_sha256);environment=read(a.env_json,a.env_sha256)
        require(provenance['runtime_manifest']['SHA256']==source['runtime_manifest']['SHA256']==a.manifest_sha256
                and provenance['runtime_manifest']['path']==source['runtime_manifest']['path']==str(a.manifest)
                and source['runtime_environment']['SHA256']==a.env_sha256,'retained source/new target runtime binding differs')
        loader=Path(environment['site'])/'rtp_llm/models_py/modules/dsv41/engram.py';expected_loader=manifest['production_Python']['rtp_llm/models_py/modules/dsv41/engram.py']
        require(provenance['loader_source']==source['loader_source'] and source['loader_source']['path']==str(loader)
                and source['loader_source']['logical_path']==str(loader) and source['loader_source']['resolved_path']==str(loader.resolve(strict=True)),
                'retained loader logical/resolved path/provenance differs')
        exact_pin(loader,expected_loader);require(pin(loader)=={k:source['loader_source'][k] for k in ['bytes','SHA256']},'retained loader full SHA differs')
        rows=r['files'];current=s['files'];sizes=[101377628352,101380404048]
        require(len(rows)==len(current)==len(source['tables'])==2 and sorted(v['bytes'] for v in rows)==sizes
                and len({v['path'] for v in rows})==2,'retained keeper exact two full static tables required')
        locks=Path('/proc/locks').read_text().splitlines();audited=[]
        for row in rows:
            path=Path(row['path']);require(path.parent==Path('/dev/shm') and re.fullmatch('rtp_llm_engram_500_[0-9a-f]{24}',path.name) and not path.is_symlink(),'retained table path differs')
            peer=next((v for v in current if v['path']==row['path']),None);origin=next((v for v in source['tables'] if v['path']==row['path']),None)
            require(peer is not None and origin is not None and all(row[k]==peer[k]==origin[k] for k in ['device','inode','UID','bytes']),
                    'retained ready/status/source table identity differs')
            require(type(row['FD']) is int and row['FD']==peer['FD'] and row['SH_data_lease_held'] is peer['SH_data_lease_held'] is True,'retained data FD/SH lease differs')
            opened=Path('/proc')/str(expected['PID'])/'fd'/str(row['FD']);p=path.stat();f=opened.stat()
            require(stat.S_ISREG(p.st_mode) and stat.S_ISREG(f.st_mode) and p.st_uid==f.st_uid==row['UID']==500
                    and p.st_ino==f.st_ino==row['inode'] and p.st_size==f.st_size==row['bytes'] and p.st_dev==f.st_dev==row['device'],
                    'retained actual table/FD device/inode/size/UID differs')
            fdinfo=(Path('/proc')/str(expected['PID'])/'fdinfo'/str(row['FD'])).read_text()
            flags=re.search(r'^flags:\s*([0-7]+)$',fdinfo,re.M);require(flags and int(flags[1],8)&os.O_ACCMODE==os.O_RDONLY,'retained table FD must be read-only')
            held=False
            for text in locks:
                fields=text.split()
                if len(fields)>=6 and fields[1:4]==['FLOCK','ADVISORY','READ'] and fields[4]==str(expected['PID']):
                    device_inode=fields[5].split(':')
                    if len(device_inode)==3 and (int(device_inode[0],16),int(device_inode[1],16),int(device_inode[2]))==(os.major(p.st_dev),os.minor(p.st_dev),p.st_ino):held=True
            require(held,'retained actual keeper PID READ FLOCK absent');audited.append(dict(path=str(path),device=p.st_dev,inode=p.st_ino,bytes=p.st_size,UID=500,FD=row['FD'],read_only_FD=True,READ_FLOCK_observed=True))
        minimum=4*1024**3;proof=dict(mode='retained_two_verified_static_weight_FDs',process=actual,ready=dict(path=str(a.keeper_root/'ready.json'),**pin(a.keeper_root/'ready.json')),
            helper=dict(path=str(a.keeper_helper),**pin(a.keeper_helper)),tables=audited,source_manifest=r['source_manifest'],full_weight_hash_accepted=False,KV_cache_reused=False)
    fs=os.statvfs('/dev/shm');free=fs.f_bavail*fs.f_frsize;require(free>=minimum,'actual freeSHM below explicit weights-mode admission')
    return dict(proof,free_bytes=free,minimum_bytes=minimum)


def descendant(current,root,controller):
    chain=[];seen=set()
    for _ in range(64):
        require(current and current['PID'] not in seen,'ancestry missing/cycle');seen.add(current['PID']);chain.append(current)
        if current['PID']==root['PID']:require(controller.same(root,current),'model root PID reused');return chain
        require(current['PPID']>1,'GPU PID is not model descendant: '+json.dumps(dict(sampled_PID=chain[0]['PID'],model_root_PID=root['PID'],chain=[{k:v.get(k) for k in ['PID','UID','PPID','start_ticks','state']} for v in chain]),sort_keys=True));current=controller.identity(current['PPID'])
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
    p.add_argument('--mode',choices=['compose','check','run'],default='compose');p.add_argument('--config-child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['P','D'],required=True);p.add_argument('--host',required=True)
    p.add_argument('--prefill-host',choices=P_HOSTS);p.add_argument('--namespace');p.add_argument('--publication',type=int,choices=[0,1]);p.add_argument('--instance-group');p.add_argument('--cache-profile',choices=['reference','publication-candidate'])
    for name in ['env-json','manifest']:p.add_argument('--'+name,type=Path)
    for name in ['env-sha256','manifest-sha256']:p.add_argument('--'+name)
    p.add_argument('--seconds',type=int,default=7200);p.add_argument('--profile',type=Path,default=C/'airline4_v4_pure_profile_v1.py')
    p.add_argument('--controller',type=Path,default=C/'device_controller_cluster23_v1.py');p.add_argument('--nvml-helper',type=Path,default=C/'portable_h127_attention_inner.py')
    p.add_argument('--consumer-helper',type=Path,default=C/'c3_consumer_inner_service.py');p.add_argument('--consumer-helper-sha256')
    p.add_argument('--consumer-root',type=Path,default=C/'consumer-business-v2');p.add_argument('--weights-mode',choices=['cold','retained']);p.add_argument('--keeper-root',type=Path);p.add_argument('--keeper-helper',type=Path);p.add_argument('--keeper-helper-sha256')
    p.add_argument('--history-config',type=Path);p.add_argument('--history-config-sha256')
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
    if a.mode=='compose':print(json.dumps(dict(state='MODEL_PLAN_FROZEN_NOT_STARTED',plan_SHA256=pin(a.root/'plan.json')['SHA256'])));return 0
    if a.mode=='check':check(a,e,plan);return 0
    require(a.consumer_helper_sha256 is not None,'explicit currently reviewed Consumer helper SHA required for run')
    controller=load(a.controller,CONTROLLER_SHA,'airline4_finite_controller');n=load(a.nvml_helper,NVML_SHA,'airline4_nvml')
    service=load(a.consumer_helper,a.consumer_helper_sha256,'airline4_consumer_evidence')
    for number in [signal.SIGTERM,signal.SIGINT]:signal.signal(number,lambda *args:STOP.set())
    return serve(a,e,plan,controller,service,n)


if __name__=='__main__':raise SystemExit(main())
