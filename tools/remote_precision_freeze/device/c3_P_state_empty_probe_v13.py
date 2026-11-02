"""Independent zero-request diagnostic-P empty proof; frozen empty v2 unchanged."""
from prove_c3_fresh_p_empty_v3 import *
from c3_P_state_runtime_probe_v13 import diagnostic_config
def current_model(a):
    plan,plan_pin=read(a.model_root/'plan.json',a.plan_sha256)
    require(plan['schema']=='c3-P-state-observer-model-plan-v3' and plan['role']=='P' and plan['host']==HOST and plan['prefill_host']==HOST
            and plan['namespace']==a.namespace and plan['publication']==1 and plan['profile']=='publication-candidate','current freshP publication-candidate plan required')
    diagnostic_config(a,plan)
    argv=plan['original_model_argv'];env=plan['environment']
    for key,value in [('--tp_size','4'),('--dp_size','1'),('--prefill_cp_size','4'),('--prefill_cp_kv_cache_sharded','1'),
                      ('--warm_up','1'),('--model_warm_up','1'),('--reuse_cache','1'),('--enable_device_cache','1'),('--enable_remote_cache','1')]:
        require(argv.count(key)==1 and argv[argv.index(key)+1]==value,'freshP parser input differs: '+key)
    require(env['DSV4_STARTUP_REAL_WARMUP']=='0' and env['RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH']=='1','extra-RPC-warmup-off/publication profile differs')
    require(plan['startup_profile']['kernel_and_pool_preparation_enabled'] is True and plan['startup_profile']['extra_startup_grpc_probes_enabled'] is False,
            'necessary kernel/pool warmup must be enabled; only extra startup RPC probes are disabled')
    ready,ready_pin=read(a.model_root/'model-ready.json');status,status_pin=read(a.model_root/'model-service-status.json')
    require(ready['state']=='MODEL_READY_KV_EMPTY_PENDING' and ready['model_READY'] is True
            and status['state']=='MODEL_READY_KV_EMPTY_PENDING' and status['model_READY'] is True
            and ready['plan_SHA256']==status['plan_SHA256']==a.plan_sha256
            and ready['namespace']==status['namespace']==a.namespace
            and time.time()-status['heartbeat_at']<=45,'current actual model READY/plan/heartbeat required')
    root=live(ready['model_root']);require(same(root,status['model_root']),'current/ready model-root identity differs')
    ranks=ready['GPU_ranks'];require(len(ranks)==4 and {r['world_rank'] for r in ranks}=={0,1,2,3},'actual four owned ready ranks required')
    require({r['PID'] for r in ranks}=={r['PID'] for r in status['GPU_ranks']},'current/ready rank identities differ')
    observed=[]
    for row in ranks:
        actual=live(row);ancestry(actual,root);require((Path('/proc')/str(actual['PID'])/'comm').read_text().strip()=='rtp_llm_rank-'+str(row['world_rank']),'actual native rank comm differs')
        current=next(r for r in status['GPU_ranks'] if r['PID']==row['PID']);require(same(row,current),'current/ready rank epoch differs');observed.append(dict(actual,world_rank=row['world_rank']))
    controller,_=read(a.model_root/'model-status.json');require(controller['state']=='RUNNING' and same(root,controller['child']) and time.time()-controller['heartbeat_at']<=45,'actual finite model controller must still run this root');live(controller['controller'])
    owner=next(row for row in observed if row['world_rank']==0)
    return dict(plan=plan_pin,ready=ready_pin,status=status_pin,model_root=root,all4_rank_identities=observed,
                allocator_owner=owner,ready_after_epoch=max(ready_pin['mtime_epoch'],float(ready['heartbeat_at'])),
                model_deadline_epoch=status['deadline_epoch'],
                native_logs=[a.model_root/'runtime/logs/engine.log',a.model_root/'model-stage.log'])

def observe(a):
    require(a.output_dir.parent==C and not a.output_dir.exists() and not a.output_dir.is_symlink(),'new direct CONTROL output directory required')
    a.output_dir.mkdir(mode=0o700);(a.output_dir/'.owner').write_text(OWNER+'\n')
    start=time.time();deadline=time.monotonic()+a.seconds
    result=dict(schema='c3-P-state-empty-proof-v3',model_classification='explicit diagnostic P observer model; not frozen model5',state='OBSERVING_NATIVE_POST_READY_ALLOCATOR_TREE',started_at=start,deadline_epoch=start+a.seconds,
                host=HOST,namespace=a.namespace,model_root_path=str(a.model_root),expected_plan_SHA256=a.plan_sha256,
                logger_timezone_observer_local_offset_seconds=int(datetime.datetime.now().astimezone().utcoffset().total_seconds()),
                output=str(a.output_dir),observer=identity(os.getpid()),observer_requests_issued=0,
                request_admission_contract='root must keep business request admission closed until this proof succeeds; external senders are not audited here',
                actual_restore_local_hit='NOT_OBSERVED',actual_restore_remote_interval='NOT_OBSERVED',
                D_computed_published_read_interval_join='PENDING_SEPARATE_EVIDENCE',business_acceptance=False,
                allocator_owner_contract='KVCacheManager.h isAllocatorOwner: tp_rank==0; P TP4/DP1 maps worldrank0 to TP0',
                native_metric_contract='KVCacheManager.cc collectGlobalCacheMetrics uses active coordinator blockTreeCache.getStats().tree_node_count; current owner backend identity also required')
    try:
        while True:
            require(not STOP and not (a.output_dir/'stop').exists(),'observer cancelled; no model process action')
            model=current_model(a);observation=scan(model,a.namespace)
            # Revalidate PID epochs and current READY after scanning, before proof publication.
            again=current_model(a);require(same(model['model_root'],again['model_root']) and same(model['allocator_owner'],again['allocator_owner']),'model changed during observation')
            result.update(heartbeat_at=time.time(),model={k:v for k,v in model.items() if k!='native_logs'},observation=observation,state=observation['state'])
            save(a.output_dir/'status.json',result)
            if observation['state']=='OBSERVED_NONEMPTY_LOCAL_TREE':result.update(exit_code=2);break
            if observation['state']=='FRESH_P_ALLOCATOR_OWNER_CACHE_TREE_OBSERVED_EMPTY':
                result.update(exit_code=0);save(a.output_dir/'empty-proof.json',result);break
            if time.monotonic()>=deadline:result.update(state='FAILED_NO_POST_READY_EMPTY_CACHE_PROOF',exit_code=2);break
            time.sleep(min(20,max(0,deadline-time.monotonic())))
    except BaseException as error:result.update(state='BLOCKED_NO_EMPTY_CACHE_PROOF',exit_code=1,error=type(error).__name__+': '+str(error)[:512])
    result['finished_at']=time.time();save(a.output_dir/'status.json',result);save(a.output_dir/'result.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--model-root',type=Path,default=C/'airline4-fresh-p-candidate-v1');p.add_argument('--plan-sha256',required=True)
    p.add_argument('--helper-sha256',required=True);p.add_argument('--namespace',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seconds',type=int,default=120)
    a=p.parse_args();os.umask(0o077)
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as binding:binding.bind((HOST,0))
    require(os.getuid()==500 and a.model_root.parent==C and not a.model_root.is_symlink() and (a.model_root/'.owner').read_text().strip()==OWNER,'actual UID500 C3 model-root owner required')
    require(1<=a.seconds<=180 and re.fullmatch('[0-9a-f]{64}',a.plan_sha256) and re.fullmatch('[A-Za-z0-9_-]{8,100}',a.namespace),'finite/pinned explicit observation inputs required')
    def stop(*unused):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    result=observe(a);print(json.dumps(summary(result),sort_keys=True));return result['exit_code']

if __name__=='__main__':raise SystemExit(main())
