"""Independent diagnostic-P readonly runtime/DSO probe and immutable READY refs.
Original v2 probe/identity and formal comparator are unchanged. No model/RPC start.
"""
import argparse,hashlib,signal,sys
from pathlib import Path
H=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control/inner-replay-tools-v2')
if H.is_dir():sys.path[:0]=[str(H.parent),str(H)]
from c3_inner_precision_replay_v6 import *
DIAG_SCHEMA='c3-P-state-observer-model-plan-v3'
OBSERVER_SHA='9d986d662754a43681f2a7e27ce35fe9b043d8afecdfa7bc93daade266202341'
BASE_OBSERVER_SHA='d1f4bfb16039ffcc43905fe3ae8bfc0b10dbacebcd895f994200a07845c7fe70'
BOOTSTRAP_SHA='4a39281b58a161f09fb31d5beb112abb38e7a615ebae52816e39e509b08ebddc'
HOOK_MODULES=('rtp_llm.models_py.modules.dsv41.block','rtp_llm.models_py.modules.dsv41.hc.delayed','rtp_llm.models_py.modules.dsv41.moe.moe_layer','rtp_llm.models_py.modules.dsv41.fp8.attention_v41','rtp_llm.models_py.modules.dsv41.hc.v41_prenorm','rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional')
EXPECTED_POLICY=dict(DeepGEMM_batch_invariant=True,numeric_flags=NUMERIC,DSV4_MHC_SPLIT_MODE='fixed',warm_up=1,model_warm_up=1,DSV4_STARTUP_REAL_WARMUP=0,kernel_flags=KERNEL)

def diagnostic_config(a,plan):
    need(plan['schema']==DIAG_SCHEMA and plan['role']=='P' and plan['host']=='10.44.12.175','explicit diagnostic P plan required')
    need(sha(a.helper_sha256) and plan['service_helper']['path']==str(C/'c3_P_state_model_service_v11.py') and plan['service_helper']['SHA256']==a.helper_sha256 and pin(Path(plan['service_helper']['path']))['SHA256']==a.helper_sha256,'explicit actual diagnostic helper pin differs')
    obs=plan['observer']
    for key,expected in [('code',OBSERVER_SHA),('bootstrap',BOOTSTRAP_SHA),('base',BASE_OBSERVER_SHA)]:need(pin(Path(obs[key]['path']))['SHA256']==obs[key]['SHA256']==expected,'actual observer source differs')
    template=read(Path(obs['template']['path']),obs['template']['SHA256']);derived=json.loads(json.dumps(template))
    for case in derived['cases']:case['plan_sha256']=a.plan_sha256
    current,cp=snapshot(Path(obs['derived_config_path']));need(current==derived and current['schema']=='c3-p-state-observer-batch-v4' and len(current['cases'])==2,'exact template+plan derivation differs')
    cm,mp=snapshot(a.model_root/'observer-config-manifest.json')
    need(cm['schema']=='c3-P-state-observer-config-manifest-v3' and cm['plan_SHA256']==a.plan_sha256 and cm['derived_config']==cp and all(cm[k]==obs[k] for k in ['template','code','bootstrap','base']),'actual independent observer config manifest differs')
    need(plan['original_model_argv'][:4]==[plan['argv'][0],'-S','-m','rtp_llm.start_server'] and plan['argv']==[plan['argv'][0],'-S',obs['bootstrap']['path']]+plan['original_model_argv'][4:],'bootstrap must preserve all original server flags exactly')
    need(plan['environment']['P_ONLY_OBSERVER_CONFIG']==obs['derived_config_path'],'actual observer config environment differs')
    need(all(case['additional_layers']==[0,1] and case['additional_query_positions']==[7168] and case['additional_budget_bytes']==4194304 for case in current['cases']) and obs['gpu_snapshot_bytes_per_rank']==16777216 and obs['additional_budget_is_subset_of_total'] is True,'additional L0/1 bound must fit original16MiB')
    manifest=read(Path(plan['runtime_manifest']['path']),plan['manifest_SHA256']);env=read(Path(plan['runtime_env']['path']),plan['env_SHA256'])
    hook_expected={name:manifest['production_Python'][name.replace('.','/')+'.py']['SHA256'] for name in HOOK_MODULES}
    need(obs['hook_source_sha256']==cm['hook_source_sha256']==hook_expected and all(case['hook_source_sha256']==hook_expected for case in current['cases']),'hook source config/manifest derivation differs')
    hook_pins={name:artifact_pin(Path(env['site'])/(name.replace('.','/')+'.py')) for name in HOOK_MODULES}
    need(all(hook_pins[name]['SHA256']==hook_expected[name] for name in HOOK_MODULES),'actual six hook source SHA differs')
    return current,dict(config=cp,manifest=mp,code=obs['code'],bootstrap=obs['bootstrap'],base=obs['base'],hook_sources=hook_pins)

def diagnostic_contract(a,plan,root,ranks):
    config,pins=diagnostic_config(a,plan)
    actual=(Path('/proc')/str(root['PID'])/'cmdline').read_bytes().split(b'\0');actual=[v.decode() for v in actual if v]
    need(actual==plan['argv'],'actual model root is not the diagnostic bootstrap argv')
    actual_env=dict(v.split('=',1) for v in (Path('/proc')/str(root['PID'])/'environ').read_bytes().decode().split('\0') if '=' in v)
    need(all(actual_env.get(k)==plan['environment'][k] for k in ['P_ONLY_OBSERVER_CONFIG','PYTHONPATH','LD_LIBRARY_PATH']+list(NUMERIC)+list(KERNEL)),'actual diagnostic model environment differs')
    markers=[];natural_hooks=[]
    for case in config['cases']:
        getter=artifact_pin(Path(case['getter_path']));need(getter['SHA256']==case['getter_sha256']=='50bc0cf24999fd4010c09a03f506c6fab8b6edad8c7f4a8a6e830bb5e9cd2659','actual RID getter source differs')
        for row in [root]+ranks:
            path=Path(case['evidence_dir'])/('%s-%s-%s-lazy-finder-installed.json'%(case['nonce'],row['PID'],row['start_ticks']))
            marker,mp=snapshot(path);need(marker['schema']=='c3-p-state-observer-evidence-v2' and marker['phase']=='lazy-finder-installed' and epoch(marker)==epoch(row) and marker['observer_sha256']==BASE_OBSERVER_SHA and marker['nonce']==case['nonce'],'actual parent/rank lazy install marker differs');markers.append(mp)
            extpath=Path(case['evidence_dir'])/('%s-%s-%s-extension-lazy-finder-installed.json'%(case['nonce'],row['PID'],row['start_ticks']))
            ext,xp=snapshot(extpath);need(epoch(ext)==epoch(row) and ext['nonce']==case['nonce'] and ext['phase']=='extension-lazy-finder-installed' and ext['extension_schema']=='c3-p-hc-seam-extension-v4' and ext['extension_source']==pins['code'] and ext['base_source']==pins['base'] and ext['bootstrap_source']==pins['bootstrap'],'actual extension+base+bootstrap startup identity differs');markers.append(xp)
            for module in ['rtp_llm.models_py.model_desc.deepseek_v41_base_model','rtp_llm.models_py.modules.dsv41.fp8.attention_v41','rtp_llm.models_py.modules.dsv41.fp8._v41_cp_async','rtp_llm.models_py.modules.dsv41.fp8._v41_producer_metadata','rtp_llm.models_py.modules.dsv41.block','rtp_llm.models_py.modules.dsv41.hc.delayed','rtp_llm.models_py.modules.dsv41.moe.moe_layer','rtp_llm.models_py.modules.dsv41.hc.v41_prenorm','rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional']:
                prefix='%s-%s-%s-'%(case['nonce'],row['PID'],row['start_ticks']);bp=Path(case['evidence_dir'])/(prefix+'hooked_'+module.replace('.','_')+'.json');ep=Path(case['evidence_dir'])/(prefix+'extension-hooked_'+module.replace('.','_')+'.json')
                entry=dict(PID=row['PID'],nonce=case['nonce'],module=module,state='PENDING_LAZY_IMPORT',base_hook=None,extension_hook=None)
                if bp.exists():
                    b,bpin=snapshot(bp);need(epoch(b)==epoch(row) and b['observer_sha256']==BASE_OBSERVER_SHA and b['nonce']==case['nonce'],'actual base natural hook differs');entry['base_hook']=bpin
                if ep.exists():
                    e,epin=snapshot(ep);need(epoch(e)==epoch(row) and e['nonce']==case['nonce'] and e['extension_source']==pins['code'] and e['base_source']==pins['base'] and e['bootstrap_source']==pins['bootstrap'],'actual extension natural hook source differs');entry['extension_hook']=epin
                    if module in HOOK_MODULES:need(e['module']==module and e['module_sha256']==case['hook_source_sha256'][module]==pins['hook_sources'][module]['SHA256'],'actual natural hook module SHA differs')
                if entry['base_hook'] is not None:need(entry['extension_hook'] is not None,'base natural module hook lacks extension chain');entry['state']='BASE_AND_EXTENSION_NATURAL_HOOKS_OBSERVED'
                elif entry['extension_hook'] is not None:entry['state']='EXTENSION_NEW_TARGET_NATURAL_HOOK_OBSERVED'
                natural_hooks.append(entry)
    return dict(**pins,lazy_install_markers=markers,natural_hook_observation=natural_hooks,diagnostic_probe=pin(Path(__file__).resolve()),getter_mapped_live_evidence='PENDING_FIRST_ACTUAL_ARMED_FORWARD',capture_or_live_RID_verified=False,classification='explicit observer diagnostic; not frozen model5/formal acceptance')

def probe_model(a):
    plan=read(a.model_root/'plan.json',a.plan_sha256);host=plan['host'];role=plan['role']
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as binding:binding.bind((host,0))
    need(role in ['P','D'] and host in MANIFESTS and plan['schema']==DIAG_SCHEMA and role=='P' and host=='10.44.12.175'
         and plan['source_commit']==B1 and plan['service_helper']['SHA256']==a.helper_sha256,'actual reviewed v5 role/model plan required')
    need(pin(Path(plan['service_helper']['path']))['SHA256']==a.helper_sha256,'actual model service helper bytes differ')
    environment=read(Path(plan['runtime_env']['path']),plan['env_SHA256']);manifest=read(Path(plan['runtime_manifest']['path']),plan['manifest_SHA256'])
    need(plan['env_SHA256']==ENV and plan['manifest_SHA256']==MANIFESTS[host] and manifest['source_commit']==B1 and manifest['host']==host,'actual target env/manifest differs')
    observed_native={}
    for name,expected in NATIVE.items():
        member=manifest['native']['rtp_llm/libs/'+name];current=artifact_pin(Path(member['path']))
        need(current['bytes']==member['bytes'] and current['SHA256']==member['SHA256']==expected,'actual native artifact differs: '+name);observed_native[name]=current
    dg_path=Path(environment['DG_root'])/'deep_gemm/_C.cpython-310-aarch64-linux-gnu.so';dg=artifact_pin(dg_path)
    need(dg['SHA256']==DG and manifest['DG_code_members'][dg_path.name]['SHA256']==DG,'actual old ad62 backend differs')
    ready_raw=(a.model_root/'model-ready.json');ready,ready_pin=snapshot(ready_raw)
    status_raw=a.model_root/'model-service-status.json';status,status_pin=snapshot(status_raw)
    need(ready['state']==status['state']=='MODEL_READY_KV_EMPTY_PENDING' and ready['model_READY'] is status['model_READY'] is True
         and ready['plan_SHA256']==status['plan_SHA256']==a.plan_sha256 and -2<=time.time()-status['heartbeat_at']<=45,
         'actual current READY/heartbeat differs')
    root=actual_live(ready['model_root']);need(epoch(root)==epoch(status['model_root']),'actual ready/current root epoch differs')
    mount_text=Path('/proc/self/mountinfo').read_text();legacy=legacy_backing(manifest)
    control,control_pin=snapshot(a.model_root/'model-status.json')
    need(control['state']=='RUNNING' and epoch(control['child'])==epoch(root)
         and -2<=time.time()-control['heartbeat_at']<=45,'actual finite model controller is not currently running this root')
    actual_live(control['controller'])
    ranks=[];mappings={name:[] for name in list(NATIVE)+['DG_native']}
    artifacts=dict(observed_native,DG_native=dg)
    bridges={key:mapped_identity_for(Path(value['path']),value,mount_text,manifest,legacy) for key,value in artifacts.items()}
    need(len(status['GPU_ranks'])==4 and {r['world_rank'] for r in status['GPU_ranks']}=={0,1,2,3}
         and {epoch(r) for r in status['GPU_ranks']}=={epoch(r) for r in ready['GPU_ranks']},'actual ready/current four rank epochs differ')
    for row in status['GPU_ranks']:
        current=actual_live(row);actual_ancestry(current,root)
        need((Path('/proc')/str(current['PID'])/'comm').read_text().strip()=='rtp_llm_rank-'+str(row['world_rank']),'actual model rank comm differs')
        ranks.append(dict(current,world_rank=row['world_rank']))
        maps_path=Path('/proc')/str(current['PID'])/'maps';need(maps_path.stat().st_size<=8*1024**2,'rank maps inventory too large')
        text=maps_path.read_text();need(len(text)<=8*1024**2,'rank maps read budget exceeded')
        for line in text.splitlines():
            key=match_native_map(line,artifacts,bridges)
            if key is not None:mappings[key].append(current['PID'])
    need(all(v for v in mappings.values()),'required native4/SDK/DG is not mapped by the actual model ranks')
    for row in [root]+ranks:actual_live(row)
    need(Path('/proc/self/mountinfo').read_text()==mount_text,'actual mount topology changed during source probe')
    for key,value in artifacts.items():
        current=stat_metadata(Path(value['path']).stat());need(all(current[k]==value[v] for k,v in [('dev','device'),('inode','inode'),('bytes','bytes'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns')]),'runtime artifact changed during maps sample')
    env=plan['environment'];argv=plan['argv']
    def argument(name):need(argv.count(name)==1,'ambiguous actual parsed profile');return argv[argv.index(name)+1]
    policy=dict(DeepGEMM_batch_invariant=plan['DeepGEMM_batch_invariant'],numeric_flags=plan['numeric_flags'],DSV4_MHC_SPLIT_MODE=env['DSV4_MHC_SPLIT_MODE'],
        warm_up=int(argument('--warm_up')),model_warm_up=int(argument('--model_warm_up')),
        DSV4_STARTUP_REAL_WARMUP=int(env['DSV4_STARTUP_REAL_WARMUP']),
        kernel_flags={key:env[key] for key in ['DSV4_USE_TK_MHC','DSV4_MHC_HEAD_FUSED','DSV41_FUSED_CROSS_LAYER_MHC','DSV41_ENGRAM_RETAINED_FUSION','RTP_LLM_STREAM_ASYNC']})
    result=dict(schema='c3-P-state-model-source-v3',state='CURRENT_DIAGNOSTIC_P_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED',observed_at=time.time(),
        observer_UID=os.getuid(),observer=process_identity(os.getpid()),probe_scope='read-only native hashes/mountinfo/maps and model metadata; no GPU/weights/business bytes',
        role=role,host=host,model_plan_schema=plan['schema'],model_service_helper_SHA256=a.helper_sha256,source_commit=B1,
        env_SHA256=ENV,manifest_SHA256=MANIFESTS[host],plan_SHA256=a.plan_sha256,ready_SHA256=ready_pin['SHA256'],
        namespace=plan['namespace'],publication=plan['publication'],cache_profile=plan['profile'],numeric_flags=plan['numeric_flags'],
        precision_config=policy,precision_config_sha256=native.digest(policy),model_root=root,GPU_ranks=ranks,
        heartbeat_at=min(status['heartbeat_at'],control['heartbeat_at']),deadline_epoch=min(status['deadline_epoch'],control['deadline_epoch']),model_READY=True,
        ready_snapshot_state=status['state'],native_artifacts=observed_native,DG_artifact=dg,
        actual_mapped_GPU_rank_PIDs={key:sorted(set(v)) for key,v in mappings.items()},
        native_mapping_bridges=bridges,actual_mountinfo_SHA256=__import__('hashlib').sha256(mount_text.encode()).hexdigest(),
        protocol_code=protocol_code(),
        source_documents=dict(plan=pin(a.model_root/'plan.json'),ready=ready_pin,status=status_pin,controller=control_pin,
                              environment=pin(Path(plan['runtime_env']['path'])),manifest=pin(Path(plan['runtime_manifest']['path']))))
    result['diagnostic_runtime']=diagnostic_contract(a,plan,root,ranks)
    need(policy==EXPECTED_POLICY,'diagnostic numerical policy differs')
    return result

def publish_refs(a):
    need(os.getuid()==500,'identity publication requires actual UID500')
    row=read(a.source,a.source_sha256);need(row['schema']=='c3-P-state-model-source-v3' and row['state']=='CURRENT_DIAGNOSTIC_P_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED' and row['plan_SHA256']==a.plan_sha256 and row['model_service_helper_SHA256']==a.helper_sha256 and -2<=time.time()-row['observed_at']<=45,'fresh actual diagnostic source required')
    plan=read(a.model_root/'plan.json',a.plan_sha256);config,pins=diagnostic_config(a,plan)
    root=actual_live(row['model_root']);ranks=[actual_live(v) for v in row['GPU_ranks']]
    for r in ranks:actual_ancestry(r,root)
    need(len(ranks)==4 and row['deadline_epoch']>time.time()+195,'four current ranks/full request lease required')
    model=dict(epoch=dict(host=row['host'],role='P',**{k:root[k] for k in ['PID','UID','start_ticks']}),plan_sha256=a.plan_sha256,runtime_manifest_sha256=row['manifest_SHA256'],native_sha256=NATIVE)
    reports=[]
    for case in config['cases']:
        path=Path(case['identity_ref']);need(a.model_root in path.parents and not path.exists() and not path.is_symlink(),'new task-private identity ref required')
        value=dict(schema='c3-p-state-model-identity-v2',nonce=case['nonce'],model=model,model_identity_sha256=native.digest(model),rank_epochs=[{k:r[k] for k in ['PID','UID','start_ticks']} for r in ranks],independent_source=dict(path=str(a.source),SHA256=a.source_sha256),model_deadline_epoch=row['deadline_epoch'])
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);raw=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();need(len(raw)<=65536,'identity ref too large')
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        reports.append(pin(path))
    return dict(schema='c3-P-state-ready-ref-publication-v1',state='TWO_CANONICAL_READY_REFS_PUBLISHED',refs=reports,model=model,model_requests=0,capture_verified=False)


def pair_identity(a):
    need(os.getuid()==500,'client identity requires actual UID500')
    p=read(a.source,a.source_sha256);d=read(a.d_source,a.d_source_sha256);now=time.time()
    need(p['schema']=='c3-P-state-model-source-v3' and p['state']=='CURRENT_DIAGNOSTIC_P_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED' and p['model_plan_schema']==DIAG_SCHEMA and p['model_service_helper_SHA256']==a.helper_sha256 and p['plan_SHA256']==a.plan_sha256,'explicit diagnostic P source differs')
    need(d['schema']=='c3-inner-replay-model-source-v1' and d['state']=='CURRENT_V7_MODEL_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED' and d['model_plan_schema']=='c3-airline4-inner-model-plan-v7' and d['model_service_helper_SHA256']==MODEL_HELPER and d['plan_SHA256']==a.d_plan_sha256,'actual original model5 D source differs')
    need(a.d_model_root is not None and a.d_model_root.parent==C and d['source_documents']['plan']['path']==str(a.d_model_root/'plan.json'),'explicit D batch root differs from actual source')
    for row,role,host in [(p,'P','10.44.12.175'),(d,'D','10.44.61.71')]:
        need(row['role']==role and row['host']==host and row['source_commit']==B1 and row['env_SHA256']==ENV and row['manifest_SHA256']==MANIFESTS[host] and row['publication']==1 and row['cache_profile']=='publication-candidate' and row['numeric_flags']==NUMERIC and row['precision_config']==EXPECTED_POLICY,'actual paired source/numeric identity differs')
        need(row['model_READY'] is True and -2<=now-row['observed_at']<=45 and -2<=now-row['heartbeat_at']<=45 and row['deadline_epoch']>now+195,'actual paired source READY/lease is stale or insufficient')
        need(len(row['GPU_ranks'])==4 and {r['world_rank'] for r in row['GPU_ranks']}=={0,1,2,3} and len({epoch(r) for r in [row['model_root']]+row['GPU_ranks']})==5,'actual paired rank epoch identities incomplete')
        need(all(row['native_artifacts'][n]['SHA256']==h for n,h in NATIVE.items()) and row['DG_artifact']['SHA256']==DG and row['protocol_code']==protocol_code(),'actual paired native/SDK/DG/protocol differs')
    return dict(schema='c3-p-state-diagnostic-replay-identity-v3',classification='explicit diagnostic P + original model5 D; not old replay identity or formal acceptance',source_commit=B1,base_native_source=C1,artifact_kind='native_incremental',native_sha256=NATIVE,DG_native_SHA256=DG,precision_config=EXPECTED_POLICY,precision_config_sha256=native.digest(EXPECTED_POLICY),namespace=p['namespace'],D_publication_namespace=d['namespace'],roles={'P':p,'D':d},generation_profile_id=business.PROFILE_ID,formal_acceptance=False,observed_at=now,model_requests=0)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['probe','publish-refs','identity']);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--helper-sha256',required=True);p.add_argument('--source',type=Path);p.add_argument('--source-sha256');p.add_argument('--d-source',type=Path);p.add_argument('--d-source-sha256');p.add_argument('--d-plan-sha256');p.add_argument('--d-model-root',type=Path);a=p.parse_args()
    expected_core={'driver':'37e17a974492dcd98dadf0ac3c487765b8ae24e622594753d42573b2ed18b5e1','token_fixture':'a1ed5084218ca8053788cc036d387a04b4cf723bfa60863c91bde206b784eb15','business_profile':'41a752d933e99413bd0195da9e668d649c9b4a4e0462069f8cc85270c0c3b427','comparison':'a258eb6864eb3df49920050d45c266902a9dec11111fb0433bf8af59c86e2885'}
    need({k:v['SHA256'] for k,v in protocol_code().items()}==expected_core,'frozen native replay dependencies differ')
    need(a.model_root.parent==C and sha(a.plan_sha256) and sha(a.helper_sha256),'explicit direct CONTROL diagnostic model/helper pins required')
    need(os.getuid() in ([0,500] if a.command=='probe' else [500]),'probe readonly observer UID0/500; refs UID500 only')
    signal.alarm(120)
    result=probe_model(a) if a.command=='probe' else publish_refs(a) if a.command=='publish-refs' else pair_identity(a);print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
