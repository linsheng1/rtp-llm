"""C3 native-inner replay/compare v2 with measured overlay backing identities; formal wheel gates unchanged.

run actions: normal(R,bypass,PD), d-source(G,natural max,PD), p-seed(real
R[:c+1],P-only,max1; not business generation), d-restore/p-restore(R,PD).
Default run is protobuf-only check; --execute sends ONE native request.
Input/output IDs, request protobufs, differences and tool data stay on device.
Identity is a pinned supplied current v5 model/manifest snapshot, structurally
checked here; this CPU client does not independently inspect remote /proc.
A completed request is not publication/read/byte/business acceptance.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import signal
import json
import os
from pathlib import Path
import re
import socket
import time
from types import SimpleNamespace

import token_fixture as native
import business_token_fixture as business
import precision_compare as comparison

C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
TASK=C.parent
B1='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c'
C1='c1b4b19d82f56507f89be86dddf90497501d32a6'
ENV='1bb3d2ba9082c9ce17c2990ff9a8448855d68c88d793d89b8d25538d8355f6b2'
DG='b0233518517afc95e17a998ae1c8bae7158fab60fba24aa7eeb5cd77d571c99e'
MODEL_HELPER='e8497c1715312b970fb2d8971c7e28ac8f324cd84b3749c1b6c25c2ec267ec7f'
NATIVE={
 'libth_transformer.so':'ac19b57615a45ffbc176287faed8152362372810f5e8c6cdb0dd69fdae57dd96',
 'libth_transformer_config.so':'29814f36bdd9aaa22645bc91fb8be416ebb1093caf6c61501314fe44460c2a4c',
 'librtp_compute_ops.so':'280a9064c45f8074864200816c79cd370869ba82a72da90cbf3d385ef84a6aab',
 'kv_cache_manager_client.so':'eccbafbcdc651c65df5c029f8f3326e954f1dbe601ed651b27c7fc69562d25ed'}
MANIFESTS={'10.44.12.175': '8614842e06e19bb5be272326ef990ff42263a837745fdbb12a22489e8f1e6ba5', '10.44.61.71': '179428dbcd30ca64ce914ad0dad5f1b6d2bf5aa43682ffca8c5d87680f7c0115'}
NUMERIC=dict(DSV41_FUSED_QUERY_QUANT='1',DSV41_FUSED_QKV_ROPE_CACHE='1',DSV41_FUSED_QB_KV='0',
             DSV41_FUSED_READY_QKV='0',DSV41_CUSTOM_QB='0',DSV4_FP8_QUANT_KERNEL='auto')
KERNEL=dict(DSV4_USE_TK_MHC='1',DSV4_MHC_HEAD_FUSED='1',DSV41_FUSED_CROSS_LAYER_MHC='1',
            DSV41_ENGRAM_RETAINED_FUSION='0',RTP_LLM_STREAM_ASYNC='0')
ACTIONS=['normal','d-source','p-seed','d-restore','p-restore']


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def pin(path,limit=32*1024**2):
    import hashlib
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'bounded regular evidence required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'evidence changed during read')
    return dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def snapshot(path):
    import hashlib
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024**2,'bounded regular JSON evidence required')
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read(32*1024**2+1);after=os.fstat(stream.fileno())
    need(len(raw)<=32*1024**2 and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),
         'JSON evidence changed while reading')
    return json.loads(raw),dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def read(path,expected):
    value,actual=snapshot(path);need(sha(expected) and actual['SHA256']==expected,'evidence whole SHA differs');return value


def epoch(row):return tuple(str(row[k]) for k in ['PID','UID','start_ticks'])


def protocol_code():
    return dict(driver=pin(Path(__file__).resolve()),token_fixture=pin(Path(native.__file__).resolve()),
                business_profile=pin(Path(business.__file__).resolve()),comparison=pin(Path(comparison.__file__).resolve()))


def common_identity(value):
    need(value.get('schema')=='c3-native-inner-replay-identity-v1' and value.get('artifact_kind')=='native_incremental'
         and value.get('source_commit')==B1 and value.get('base_native_source')==C1 and value.get('Cpp_diff_empty') is True
         and value.get('wheel_sha256') in (None,''),'explicit B1/C1 native-inner identity required; never fabricate wheel identity')
    need(value.get('native_sha256')==NATIVE and value.get('SDK_packaged_DSO_SHA256')==NATIVE['kv_cache_manager_client.so']
         and value.get('DG_native_SHA256')==DG,'actual four-native/SDK/ad62 identity differs')
    policy=value['precision_config'];need(value.get('precision_config_sha256')==native.digest(policy),'joint numerical policy SHA differs')
    need(policy['DeepGEMM_batch_invariant'] is True and policy['numeric_flags']==NUMERIC and policy['kernel_flags']==KERNEL and policy['DSV4_MHC_SPLIT_MODE']=='fixed'
         and policy['warm_up']==1 and policy['model_warm_up']==1 and policy['DSV4_STARTUP_REAL_WARMUP']==0,
         'frozen v5 normal-kernel-warmup inner policy differs')
    return dict(source_commit=B1,base_native_source=C1,artifact_kind='native_incremental',native_sha256=NATIVE,
                DG_native_SHA256=DG,runtime_environment_SHA256=ENV,precision_config_sha256=native.digest(policy),
                generation_profile_id=business.PROFILE_ID,formal_wheel_accepted=False)


def identity_check(value,action,at,required_seconds=0):
    common=common_identity(value);publication=action!='normal'
    need({k:v['SHA256'] for k,v in value['protocol_code'].items()}=={k:v['SHA256'] for k,v in protocol_code().items()},
         'actual probe/client protocol code fingerprints differ')
    need(value.get('namespace') and value.get('D_publication_enabled') is publication,'action/deployment publication identity differs')
    roles=value.get('roles',{});need(set(roles)=={'P','D'},'actual P/D runtime snapshots required')
    for role,row in roles.items():
        host=row['host'];need(host in (['10.44.38.185','10.44.12.175'] if role=='P' else ['10.44.61.71']),'C3 role/host differs')
        need(row['model_plan_schema']=='c3-airline4-inner-model-plan-v7' and row['source_commit']==B1
             and row['env_SHA256']==ENV and row['manifest_SHA256']==MANIFESTS[host]
             and row['model_service_helper_SHA256']==MODEL_HELPER
             and sha(row['plan_SHA256']) and sha(row['ready_SHA256']),'actual v5 runtime plan/env/manifest binding differs')
        need(bool(row['namespace']) and (row['namespace']==value['namespace'] or (role=='D' and action in ['p-seed','p-restore']))
             and row['publication']==int(publication)
             and row['cache_profile']==('publication-candidate' if publication else 'reference')
             and row['numeric_flags']==NUMERIC and row['precision_config_sha256']==common['precision_config_sha256'],
             'P/D namespace/publication/joint policy differs')
        need(row['state']=='MODEL_READY_KV_EMPTY_PENDING' and row['model_READY'] is True
             and -2<=at-row['heartbeat_at']<=45 and row['deadline_epoch']>at+required_seconds+15,
             'supplied current READY/lease snapshot does not admit request')
        processes=[row['model_root']]+row['GPU_ranks']
        need(len(row['GPU_ranks'])==4 and {r['world_rank'] for r in row['GPU_ranks']}=={0,1,2,3}
             and len({r['PID'] for r in processes})==5 and all(type(r['PID']) is int and r['PID']>0 and r['UID']==500
             and str(r['start_ticks']).isdigit() and int(r['start_ticks'])>0 for r in processes),'actual model/rank epoch metadata incomplete')
    expected_host='10.44.12.175'
    need(roles['P']['host']==expected_host,'this batch uses sequential source/fresh P12 cache instances')
    return common


def process_identity(pid):
    root=Path('/proc')/str(pid)
    try:
        fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        return dict(PID=pid,UID=root.stat().st_uid,start_ticks=fields[19],PPID=int(fields[1]),state=fields[0])
    except (OSError,IndexError,ValueError):return None


def actual_live(expected):
    current=process_identity(expected['PID']);need(current is not None and epoch(current)==epoch(expected)
        and current['UID']==500 and current['state'] not in ['Z','X','x'],'actual local model epoch is not live');return current


def actual_ancestry(child,root):
    current=child;seen=set()
    for _ in range(64):
        need(current is not None and current['PID'] not in seen,'missing/cyclic actual model ancestry');seen.add(current['PID'])
        if current['PID']==root['PID']:need(epoch(current)==epoch(root),'model-root epoch reused');return
        need(current['PPID']>1,'actual rank is not a model-root descendant');current=process_identity(current['PPID'])
    raise ValueError('actual process ancestry budget exceeded')


def artifact_pin(path):
    import hashlib
    before=path.stat();need(path.is_file() and 0<before.st_size<=2*1024**3,'bounded explicit runtime artifact required')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    after=path.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'runtime artifact changed while hashing')
    return dict(path=str(path),device=after.st_dev,inode=after.st_ino,bytes=after.st_size,SHA256=h.hexdigest(),
                UID=after.st_uid,mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns)


class MappingProofRequired(ValueError):
    def __init__(self,metadata):
        super().__init__('actual overlay/backing mapping identity proof is missing')
        self.metadata=metadata


def stat_metadata(st):
    return dict(inode=st.st_ino,dev=st.st_dev,major=os.major(st.st_dev),minor=os.minor(st.st_dev),
                bytes=st.st_size,UID=st.st_uid,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def covering_mount(path,text):
    def unescape(value):return re.sub(r'\\([0-7]{3})',lambda match:chr(int(match[1],8)),value)
    candidates=[]
    for line in text.splitlines():
        left,right=line.split(' - ',1);fields=left.split();more=right.split();mountpoint=Path(unescape(fields[4]))
        if path==mountpoint or mountpoint in path.parents:candidates.append((len(mountpoint.parts),fields,more,line))
    need(bool(candidates),'no actual mount covers runtime artifact')
    _,fields,more,line=max(candidates,key=lambda row:row[0]);opts=dict(v.split('=',1) for v in more[2].split(',') if '=' in v)
    result=dict(mountpoint=unescape(fields[4]),root=unescape(fields[3]),filesystem=more[0],raw_line=line)
    if more[0]=='overlay':
        need(result['root']=='/' and 'upperdir' in opts,'unsupported actual overlay geometry')
        result['upperdir']=unescape(opts['upperdir'])
    return result


def validate_backing_member(member,path,current,expected,mount):
    need(mount['filesystem']=='overlay' and mount['root']=='/','explicit actual overlay upper bridge required')
    backing=Path(mount['upperdir'])/path.relative_to(Path(mount['mountpoint']))
    need(member['candidate_path']==str(path) and member['backing_path']==str(backing),'frozen backing/logical path differs')
    overlay,under=member['overlay_stat'],member['backing_stat']
    for row in [overlay,under]:need(os.major(row['dev'])==row['major'] and os.minor(row['dev'])==row['minor'],'backing device encoding differs')
    need(overlay==current,'current artifact metadata differs from frozen overlay identity')
    need(all(overlay[key]==under[key] for key in ['inode','bytes','UID','mtime_ns','ctime_ns']),'actual overlay/upper metadata bridge differs')
    need(member['SHA256']==expected['SHA256'] and overlay['bytes']==expected['bytes'],'frozen backing SHA/size differs from actual pinned artifact')
    return dict(verification='frozen_overlay_upper_backing_proof',logical_path=str(path),backing_path=str(backing),
                overlay_stat=overlay,mapped_identity=under,SHA256=member['SHA256'],mount=mount)


def legacy_backing(manifest):
    # Reuse the already verified C1 four-file proof. No privilege request or
    # new read of an inaccessible backing tree is required for this path.
    path=C/'candidate-inner-v1/candidate-backing-proof-v2.json'
    if not path.exists():return None
    st=path.lstat();need(not path.is_symlink() and st.st_uid==0 and st.st_mode&0o022==0,'existing root-frozen backing proof ownership/mode differs')
    proof,reference=snapshot(path)
    need(proof['state']=='CANDIDATE_OVERLAY_BACKING_PROOF_PASS' and proof['schema']==1 and proof['UID']==0 and proof['owner_UID']==500
         and proof['candidate_root']==str(C/'candidate-inner-v1') and proof['source_commit']==C1
         and proof['manifest_SHA256']==manifest['base_manifest_SHA256'] and set(proof['members'])==set(NATIVE),
         'existing C1 backing proof source/manifest/member identity differs')
    return proof,reference


def mapped_identity_for(path,artifact,mount_text,manifest,legacy=None):
    current=stat_metadata(path.stat());need((current['dev'],current['inode'],current['bytes'],current['mtime_ns'],current['ctime_ns'])==
        (artifact['device'],artifact['inode'],artifact['bytes'],artifact['mtime_ns'],artifact['ctime_ns']),'runtime artifact changed after complete hash')
    mount=covering_mount(path,mount_text)
    if mount['filesystem']!='overlay':return None
    expected=dict(bytes=artifact['bytes'],SHA256=artifact['SHA256'])
    name=path.name
    if legacy is not None and name in legacy[0]['members']:
        proof,reference=legacy
        need(all(proof['mount'][key]==mount[key] for key in ['mountpoint','root','filesystem','upperdir']),'actual longest covering mount differs from frozen backing proof')
        bridge=validate_backing_member(proof['members'][name],path,current,expected,mount);bridge['proof']=reference;return bridge
    backing=Path(mount['upperdir'])/path.relative_to(Path(mount['mountpoint']))
    try:
        # Optional UID500 read-only path, used only when it is already readable.
        # No sudo, namespace change, new privilege or inode-only substitution.
        need(not backing.is_symlink() and backing.resolve(strict=True)==backing,'actual upper backing must be a canonical regular file')
        under_pin=artifact_pin(backing);under=stat_metadata(backing.stat())
    except (PermissionError,FileNotFoundError):return None
    need(under_pin['SHA256']==expected['SHA256'] and under_pin['bytes']==expected['bytes'],'actual readable upper backing bytes differ from frozen artifact')
    member=dict(candidate_path=str(path),backing_path=str(backing),overlay_stat=current,backing_stat=under,SHA256=under_pin['SHA256'])
    bridge=validate_backing_member(member,path,current,expected,mount);bridge['verification']='actual_observer_UID_readonly_upper_backing_bridge';bridge['observer_UID']=os.getuid();return bridge


def match_native_map(line,artifacts,bridges):
    fields=line.split(None,5)
    if len(fields)!=6 or not fields[5].startswith('/'):return None
    raw_path=fields[5];name=Path(raw_path.replace(' (deleted)','')).name
    if name==Path(artifacts['DG_native']['path']).name and 'deep_gemm' not in Path(raw_path.replace(' (deleted)','')).parts:
        return None  # Torch and fast_safetensors legitimately use the same _C basename.
    key=next((key for key,value in artifacts.items() if Path(value['path']).name==name),None)
    if key is None:return None
    need(not raw_path.endswith(' (deleted)'),'a required native artifact is mapped deleted')
    artifact=artifacts[key];logical=Path(artifact['path']);bridge=bridges.get(key)
    allowed={str(logical)}
    if bridge:allowed.add(bridge['backing_path'])
    need(raw_path in allowed,'required native basename loaded from another path')
    major,minor=(int(v,16) for v in fields[3].split(':'));actual=(major,minor,int(fields[4]))
    direct=(os.major(artifact['device']),os.minor(artifact['device']),artifact['inode'])
    if raw_path==str(logical) and actual==direct:return key
    if bridge:
        target=bridge['mapped_identity']
        need(actual==(target['major'],target['minor'],target['inode']),'actual VMA device/inode differs from validated upper backing')
        return key
    raise MappingProofRequired(dict(artifact=key,logical_path=str(logical),actual_VMA_device=[major,minor],actual_VMA_inode=int(fields[4]),
        logical_device=[direct[0],direct[1]],logical_inode=direct[2],
        required='actual readable upper backing metadata+SHA or an existing manifest-bound frozen backing proof; no new privilege is presumed'))


def environment_interpreter(plan):
    return read(Path(plan['runtime_env']['path']),plan['runtime_env']['SHA256'])['interpreter']


def probe_model(a):
    plan=read(a.model_root/'plan.json',a.plan_sha256);host=plan['host'];role=plan['role']
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as binding:binding.bind((host,0))
    need(role in ['P','D'] and host in MANIFESTS and plan['schema']=='c3-airline4-inner-model-plan-v7'
         and plan['source_commit']==B1 and plan['service_helper']['SHA256']==MODEL_HELPER,'actual reviewed v5 role/model plan required')
    need(pin(Path(plan['service_helper']['path']))['SHA256']==MODEL_HELPER,'actual model service helper bytes differ')
    history=plan['history_capture'];cfg=read(Path(history['config']['path']),history['config']['SHA256'])
    need(cfg['schema']=='history38-inner-graph-latches-v1' and cfg['source_commit']==B1
        and cfg['role']==role and cfg['host']==host and cfg['positions']==[7169,7200]
        and cfg['layers']==list(range(40)) and cfg['gpu_budget_bytes']==4194304,'actual history profile differs')
    for key in ['observer','bootstrap','latches']:
        need(pin(Path(history[key]['path']))['SHA256']==history[key]['SHA256'],'actual history helper bytes differ')
    original=plan['original_model_argv'];need(original[:4]==[environment_interpreter(plan),'-S','-m','rtp_llm.start_server']
        and plan['argv']==[original[0],'-S',history['bootstrap']['path']]+original[4:],
        'only explicit historical startup adapter may differ from original argv')

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
    controls=[]
    for rank in ranks:
        path=Path(cfg['output'])/(str(rank['PID'])+'-control.json')
        if not path.exists():
            args=plan['argv'];directory=Path(cfg['output'])
            need(role=='P' and args[args.index('--enable_cuda_graph')+1]=='0'
                and not path.exists() and not path.with_name(str(rank['PID'])+'.sock').exists(),
                'only absent current-epoch eager-P latches may be lazy; D graph controls mandatory')
            if directory.exists():
                need(directory.is_dir() and not directory.is_symlink() and directory.stat().st_uid==500
                    and directory.stat().st_mode&0o077==0,'owned private historical latch directory required')
                for old_path in directory.glob('*-control.json'):
                    old,_=snapshot(old_path);proc=Path('/proc')/str(old['PID'])
                    if proc.exists():
                        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                        need(fields[19]!=str(old['start_ticks']) or fields[0] in ['Z','X','x'],
                             'previous latch writer still live; do not share epochs')
            controls.append(dict(record=None,rank={k:rank[k] for k in ['PID','UID','start_ticks']},
                initial_latches_absent=True,expected_output=cfg['output'],config=history['config']))
            continue
        record,source=snapshot(path)
        need(record['state']=='CAPTURE_CONTROL_READY' and record['PID']==rank['PID'] and record['UID']==500
            and str(record['start_ticks'])==str(rank['start_ticks']) and record['identity']==cfg,
            'actual four-rank historical socket identity differs')
        socket_path=Path(record['socket']);need(socket_path.is_socket() and socket_path.stat().st_uid==500
            and socket_path.stat().st_mode&0o077==0,'actual owned private history socket missing')
        controls.append(dict(source=source,record=record))

    for row in [root]+ranks:actual_live(row)
    need(Path('/proc/self/mountinfo').read_text()==mount_text,'actual mount topology changed during source probe')
    for key,value in artifacts.items():
        current=stat_metadata(Path(value['path']).stat());need(all(current[k]==value[v] for k,v in [('dev','device'),('inode','inode'),('bytes','bytes'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns')]),'runtime artifact changed during maps sample')
    env=plan['environment'];argv=plan['argv']
    def argument(name):need(argv.count(name)==1,'ambiguous actual parsed profile');return argv[argv.index(name)+1]
    need(plan['DeepGEMM_batch_invariant'] is True,'actual batch-invariant plan required')
    policy=dict(DeepGEMM_batch_invariant=True,numeric_flags=plan['numeric_flags'],DSV4_MHC_SPLIT_MODE=env['DSV4_MHC_SPLIT_MODE'],
        warm_up=int(argument('--warm_up')),model_warm_up=int(argument('--model_warm_up')),
        DSV4_STARTUP_REAL_WARMUP=int(env['DSV4_STARTUP_REAL_WARMUP']),
        kernel_flags={key:env[key] for key in ['DSV4_USE_TK_MHC','DSV4_MHC_HEAD_FUSED','DSV41_FUSED_CROSS_LAYER_MHC','DSV41_ENGRAM_RETAINED_FUSION','RTP_LLM_STREAM_ASYNC']})
    result=dict(schema='c3-inner-replay-model-source-v1',state='CURRENT_V7_MODEL_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED',observed_at=time.time(),
        observer_UID=os.getuid(),observer=process_identity(os.getpid()),probe_scope='read-only native hashes/mountinfo/maps and model metadata; no GPU/weights/business bytes',
        role=role,host=host,model_plan_schema=plan['schema'],model_service_helper_SHA256=MODEL_HELPER,source_commit=B1,
        env_SHA256=ENV,manifest_SHA256=MANIFESTS[host],plan_SHA256=a.plan_sha256,ready_SHA256=ready_pin['SHA256'],
        namespace=plan['namespace'],publication=plan['publication'],cache_profile=plan['profile'],numeric_flags=plan['numeric_flags'],
        precision_config=policy,precision_config_sha256=native.digest(policy),model_root=root,GPU_ranks=ranks,
        heartbeat_at=min(status['heartbeat_at'],control['heartbeat_at']),deadline_epoch=min(status['deadline_epoch'],control['deadline_epoch']),model_READY=True,
        ready_snapshot_state=status['state'],native_artifacts=observed_native,DG_artifact=dg,
        actual_mapped_GPU_rank_PIDs={key:sorted(set(v)) for key,v in mappings.items()},
        native_mapping_bridges=bridges,actual_mountinfo_SHA256=__import__('hashlib').sha256(mount_text.encode()).hexdigest(),
        protocol_code=protocol_code(),history_capture_controls=controls,
        source_documents=dict(plan=pin(a.model_root/'plan.json'),ready=ready_pin,status=status_pin,controller=control_pin,
                              environment=pin(Path(plan['runtime_env']['path'])),manifest=pin(Path(plan['runtime_manifest']['path']))))
    return result


def build_identity(a):
    p=read(a.p_source,a.p_source_sha256);d=read(a.d_source,a.d_source_sha256)
    for row,role in [(p,'P'),(d,'D')]:
        need(row['schema']=='c3-inner-replay-model-source-v1' and row['state']=='CURRENT_V7_MODEL_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED'
             and row['role']==role and -2<=time.time()-row['observed_at']<=45,'fresh actual per-role local/proc/artifact probe required')
        need(all(row['native_artifacts'][name]['SHA256']==expected for name,expected in NATIVE.items()) and row['DG_artifact']['SHA256']==DG
             and all(row['actual_mapped_GPU_rank_PIDs'].get(key) for key in list(NATIVE)+['DG_native']),'actual observed native4/SDK/DG incomplete')
        need({k:v['SHA256'] for k,v in row['protocol_code'].items()}=={k:v['SHA256'] for k,v in protocol_code().items()},'per-role probe/client protocol source differs')
    need(p['precision_config']==d['precision_config'],'actual P/D joint numeric/kernel policy differs')
    roles={}
    for row in [p,d]:
        projected={k:row[k] for k in ['host','model_plan_schema','model_service_helper_SHA256','source_commit','env_SHA256','manifest_SHA256','plan_SHA256','ready_SHA256',
            'namespace','publication','cache_profile','numeric_flags','precision_config_sha256','model_root','GPU_ranks','heartbeat_at','deadline_epoch','model_READY']}
        projected['state']=row['ready_snapshot_state'];roles[row['role']]=projected
    value=dict(schema='c3-native-inner-replay-identity-v1',artifact_kind='native_incremental',source_commit=B1,base_native_source=C1,Cpp_diff_empty=True,
        wheel_sha256=None,native_sha256=NATIVE,SDK_packaged_DSO_SHA256=NATIVE['kv_cache_manager_client.so'],DG_native_SHA256=DG,
        namespace=p['namespace'],D_publication_enabled=bool(d['publication']),precision_config=p['precision_config'],precision_config_sha256=p['precision_config_sha256'],roles=roles,
        per_role_source_probes=dict(P=pin(a.p_source),D=pin(a.d_source)),identity_built_at=time.time(),
        protocol_code=protocol_code(),
        identity_scope='actual device-local native artifact hashes and live /proc probes; input/output IDs absent')
    identity_check(value,a.action,time.time(),a.request_timeout_seconds)
    need(not a.output_dir.exists(),'new identity output directory required');a.output_dir.mkdir(mode=0o700,parents=False)
    native.atomic(a.output_dir/'identity.json',value)
    return dict(state='INNER_REPLAY_IDENTITY_BUILT_FROM_ACTUAL_DEVICE_PROBES',action=a.action,identity=pin(a.output_dir/'identity.json'),
                namespace=value['namespace'],common_identity=common_identity(value),model_requests=0),0


def captured_ids(row):
    need(row.get('state')=='COMPLETE' and row.get('capture_complete') is True and row.get('natural_generation') is True
         and row.get('c3_record_profile')=='airline4_native_agent1024_user2048_seed42_v1'
         and row.get('output_ids_scope')=='complete_verified_stream' and row.get('role')=='tau2-agent','actual NEW complete native-airline agent capture required')
    response=row['response'];debug=response.get('debug_info') or {};extra=response.get('extra_outputs') or {}
    if isinstance(debug,str):debug=json.loads(debug)
    if isinstance(extra,str):extra=json.loads(extra)
    result=[]
    for kind in ['input','output']:
        raw=debug.get(kind+'_ids') or extra.get(kind+'_ids');ids=native.token_ids(raw)
        need(len(ids)==row[kind+'_ids_count'] and native.digest(ids)==row[kind+'_ids_flat_SHA256']
             and native.digest(raw)==row[kind+'_ids_SHA256'],'actual capture raw/flat ID count/SHA differs');result.append(ids)
    body=row['request'];need(body['max_tokens']==1024 and body['enable_thinking'] is False and body['temperature']==0
         and body['top_k']==1 and body['top_p']==1 and body['extra_configs']['random_seed']==42,'original actual agent sampling profile differs')
    need(not any(k in body or k in body.get('extra_configs',{}) for k in ['min_tokens','min_new_tokens','ignore_eos','ignoreEOS']),
         'forced length/EOS actual source cannot seed this normal profile')
    return result,body['max_tokens']


def case_inputs(case):
    need(case['schema']=='c3-airline4-selected-private-case-v1','actual selected new native case required')
    records={}
    for kind in ['G','R']:
        expected=case[kind+'_record'];path=Path(expected['path']);row=read(path,expected['SHA256'])
        need(pin(path)['bytes']==expected['bytes'],'actual source record size differs');records[kind]=captured_ids(row)
    g,y=records['G'][0];r=records['R'][0][0];c=case['checkpoint']
    need(g==case['G_input_ids'] and r==case['R_input_ids'] and y==case['reference_G_output_ids'],'case differs from actual captured G/Y/R')
    need(type(c) is int and len(g)<c<len(r) and c%512==0 and (g+y[:-1])[:c]==r[:c],
         'actual captured pair does not contain a complete D-new canonical prefix')
    return g,r,c,records['G'][1],records['R'][1]


def fresh_check(proof,value,at):
    p=value['roles']['P'];model=proof.get('model',{});metric=proof.get('observation',{}).get('latest_metric') or {}
    need(proof.get('state')=='FRESH_P_ALLOCATOR_OWNER_CACHE_TREE_OBSERVED_EMPTY' and proof.get('observer_requests_issued')==0
         and proof.get('namespace')==value['namespace'] and proof.get('host')==p['host']
         and proof.get('expected_plan_SHA256')==p['plan_SHA256'] and model.get('plan',{}).get('SHA256')==p['plan_SHA256']
         and model.get('ready',{}).get('SHA256')==p['ready_SHA256'],'actual freshP empty proof/namespace/plan binding differs')
    need(epoch(model['model_root'])==epoch(p['model_root']) and {epoch(r) for r in model['all4_rank_identities']}=={epoch(r) for r in p['GPU_ranks']},
         'empty proof refers to another cache process instance')
    owner=next(r for r in p['GPU_ranks'] if r['world_rank']==0)
    need(metric.get('PID')==owner['PID'] and metric.get('tree_node_count')==0 and -2<=at-metric.get('at',0)<=75,
         'missing recent actual allocator-owner empty metric')
    need(any(row.get('instance_id','').startswith(value['namespace']+'_') and (row.get('wire_tp'),row.get('wire_dp'),row.get('base_tokens'))==(4,1,128)
             for row in proof['observation'].get('instances',[])),'empty cache backend instance not bound')


def p_publication_check(marker,seed,fixture,namespace):
    r=fixture['request_ids'];c=fixture['checkpoint'];inputs,outputs=native.source_ids(seed)
    need(seed['state']=='COMPLETE_NATIVE_RPC' and seed.get('response_eof_complete') is True
         and seed.get('final_aux',{}).get('pd_sep',False) is False and inputs==r[:c+1] and len(outputs)==1,
         'actual P-only max1 seed of real R[:c+1] required')
    need(marker.get('source_kind')=='P' and marker.get('committed') is True and marker.get('namespace')==seed['namespace']==namespace
         and marker.get('source_request_id')==seed['request_id'] and marker.get('source_input_ids_sha256')==native.digest(inputs)
         and marker.get('source_output_ids_sha256')==native.digest(outputs)
         and marker.get('source_identity_sha256')==native.digest(seed['identity']),'actual P commit/source binding differs')
    need(type(marker.get('computed_kv_end')) is int and type(marker.get('published_kv_end')) is int
         and c<=marker['published_kv_end']<=marker['computed_kv_end']<=len(inputs)
         and marker.get('native_boundary_verified') is True and marker.get('manager_commit_and_lookup_verified') is True,
         'actual P native/Manager commit must cover checkpoint; input length is not runtime evidence')
    for key in ['native_evidence','manager_evidence']:
        evidence=marker.get(key,{})
        need(isinstance(evidence.get('path'),str) and sha(evidence.get('SHA256')) and pin(Path(evidence['path']))['SHA256']==evidence['SHA256'],
             'actual P commit evidence file missing/different: '+key)


def run(a):
    case=read(a.case_private,a.case_sha256);g,r,c,gmax,rmax=case_inputs(case)
    value=read(a.identity_json,a.identity_sha256);common=identity_check(value,a.action,time.time(),a.timeout_seconds)
    fixture=None
    if a.action in ['p-seed','d-restore','p-restore']:
        need(a.fixture is not None and a.fixture_sha256 is not None,'qualified actual-D fixture required')
        fixture=native.validate_fixture(read(a.fixture,a.fixture_sha256))
        need(fixture['request_ids']==r and fixture['checkpoint']==c and fixture['source_input_ids']==g,'qualified fixture differs from selected actual G/R')
        d_namespace=fixture['publication_evidence']['namespace']
        need((value['namespace']==d_namespace) if a.action=='d-restore' else (value['namespace']!=d_namespace),
             'D/P-origin runs require separate source namespaces; cannot reuse D objects as P-origin')
        if a.action in ['p-seed','p-restore']:
            need(value['roles']['D']['namespace']==d_namespace,'P-origin keeps the same D source deployment/numerical identity')
    if a.action=='p-restore':
        need(all(getattr(a,k) is not None for k in ['p_publication','p_publication_sha256','p_seed_result','p_seed_result_sha256']),'P restore requires actual seed and committed P producer evidence')
        p_publication_check(read(a.p_publication,a.p_publication_sha256),read(a.p_seed_result,a.p_seed_result_sha256),fixture,value['namespace'])
    proof=None
    if a.action.endswith('restore'):
        need(a.fresh_proof is not None and a.fresh_proof_sha256 is not None,'actual fresh cache proof required before restore')
        proof=read(a.fresh_proof,a.fresh_proof_sha256);fresh_check(proof,value,time.time())
    ids=g if a.action=='d-source' else r;maximum=gmax if a.action=='d-source' else rmax
    mode='source' if a.action=='d-source' else 'bypass' if a.action=='normal' else 'prepare' if a.action=='p-seed' else 'restore'
    if fixture is None:fixture=dict(request_ids=ids,checkpoint=0,fixture_sha256=native.digest(case))
    options=SimpleNamespace(mode=mode,max_new_tokens=maximum,pd=a.action!='p-seed',return_logits=a.return_logits,
            namespace=value['namespace'],endpoint=value['roles']['P']['host']+':18081',output=a.output_dir/'native',
            timeout_seconds=a.timeout_seconds,frame_budget_bytes=64*1024**2)
    a.output_dir.mkdir(mode=0o700,parents=False,exist_ok=False)
    route=business.decode_route('10.44.61.71',a.decode_grpc_port,a.decode_http_port)
    used_ids=fixture['p_prepare_ids'] if mode=='prepare' else fixture['request_ids'];used_max=1 if mode=='prepare' else maximum
    with business.requests(route,gate_identity=value):
        gate=native.protobuf_gate(used_ids,used_max,a.action!='p-seed',a.action!='normal',a.return_logits)
        native.atomic(a.output_dir/'protobuf-gate.json',gate)
        if not a.execute:
            result=dict(state='INNER_REPLAY_NATIVE_PROTOBUF_CHECK_NO_REQUEST',action=a.action,input_tokens=len(used_ids),input_ids_sha256=native.digest(used_ids),
                        max_new_tokens=used_max,model_requests=0,generation_profile_id=business.PROFILE_ID,common_identity=common,business_acceptance=False)
            native.atomic(a.output_dir/'summary.json',result);return result,0
        if proof is not None:
            claims=C/'c3-inner-replay-proof-claims-v1';claims.mkdir(mode=0o700,exist_ok=True)
            need(not claims.is_symlink() and claims.stat().st_uid==500,'owned actual proof-claim directory required')
            with (claims/(a.fresh_proof_sha256+'.json')).open('x') as claim:
                json.dump(dict(action=a.action,output=str(a.output_dir),fixture_sha256=fixture['fixture_sha256']),claim);claim.flush();os.fsync(claim.fileno())
        result=native.run_rpc(options,fixture,value)
    result['inner_replay']=dict(action=a.action,case_private=pin(a.case_private),identity_file=pin(a.identity_json),
        protocol_code=protocol_code(),
        common_identity=common,max_new_tokens=used_max,original_R_max_new_tokens=rmax,
        generation_profile_id=business.PROFILE_ID,fresh_proof=pin(a.fresh_proof) if proof is not None else None,
        P_seed_result=pin(a.p_seed_result) if a.action=='p-restore' else None,
        P_publication=pin(a.p_publication) if a.action=='p-restore' else None,
        P_seed_classification='real R[:checkpoint+1] prefill producer max1; NOT business generation' if mode=='prepare' else None,
        runtime_identity_scope='supplied pinned current v5 ready/plan/manifest metadata checked; remote /proc not independently read by this CPU driver',
        computed_published_read_chain='PENDING_SEPARATE_NATIVE_MANAGER_AND_BYTE_EVIDENCE',formal_wheel_accepted=False)
    native.atomic(a.output_dir/'native/result.json',result)
    aux=result['final_aux'];errors=[]
    if a.action=='normal' and (int(aux.get('local_reuse_len',-1)),int(aux.get('remote_reuse_len',-1)))!=(0,0):errors.append('normal bypass has actual cache hits')
    if a.action.endswith('restore') and (int(aux.get('local_reuse_len',-1)),int(aux.get('remote_reuse_len',-1)))!=(0,c):errors.append('fresh restore must actually report local0/exact remote checkpoint')
    report=dict(state='COMPLETE_INNER_NATIVE_REQUEST_EVIDENCE_PENDING' if not errors else 'INNER_REQUEST_CACHE_GATE_FAILED',action=a.action,errors=errors,
        model_requests=1,result=pin(a.output_dir/'native/result.json'),input_tokens=len(result['input_ids']),output_tokens=len(result['output_ids']),
        input_ids_sha256=result['input_ids_sha256'],output_ids_sha256=result['output_ids_sha256'],max_new_tokens=used_max,
        actual_local_reuse_len=aux.get('local_reuse_len'),actual_remote_reuse_len=aux.get('remote_reuse_len'),actual_PD=aux.get('pd_sep'),
        publication_committed='NOT_ESTABLISHED_BY_RPC_COMPLETION',business_acceptance=False,formal_wheel_accepted=False)
    native.atomic(a.output_dir/'summary.json',report);return report,0 if not errors else 2


def position(diff):return None if diff is None else {k:v for k,v in diff.items() if k not in ['left_id','right_id']}


def source_qualification(a,fixture):
    if a.qualification is None:return dict(passed=False,errors=['actual D native/Manager qualification report absent'])
    q=read(a.qualification,a.qualification_sha256);errors=[]
    if q.get('state')!='ACTUAL_D_PREFIX_COMMIT_QUALIFIED_READ_PENDING' or q.get('errors')!=[]:errors.append('actual source qualification did not pass')
    expected=dict(checkpoint=fixture['checkpoint'],actual_D_input_ids_sha256=fixture['source_input_ids_sha256'],
        actual_D_output_ids_sha256=fixture['source_output_ids_sha256'],R_input_ids_sha256=fixture['request_ids_sha256'],
        actual_computed_kv_end=fixture['publication_evidence']['computed_kv_end'],actual_published_kv_end=fixture['publication_evidence']['published_kv_end'])
    for key,value in expected.items():
        if q.get(key)!=value:errors.append('source qualification/fixture differs: '+key)
    if q.get('fixture',{}).get('SHA256')!=a.fixture_sha256:errors.append('source qualification emitted a different fixture file')
    for name in ['d_source','publication','native_evidence','manager_report']:
        entry=q.get('inputs',{}).get(name) or {}
        if not isinstance(entry.get('path'),str) or not sha(entry.get('SHA256')) or pin(Path(entry['path']))['SHA256']!=entry['SHA256']:
            errors.append('actual source proof input missing/different: '+name)
    if not errors:
        try:
            entry=q['inputs']['d_source'];source=read(Path(entry['path']),entry['SHA256'])
            identity_check(source['identity'],'d-source',source['started_at'],source['finished_at']-source['started_at'])
            need(source['inner_replay']['action']=='d-source' and source['namespace']==fixture['publication_evidence']['namespace']
                 and native.digest(source['identity'])==fixture['publication_evidence']['source_identity_sha256'],
                 'actual D source deployment/action identity differs from fixture publication')
        except (ValueError,KeyError,TypeError) as error:errors.append(str(error))
    return dict(passed=not errors,errors=errors,report=pin(a.qualification),
                scope='actual D computed/published/native/Manager qualification linked to immutable fixture; actual read checked separately')


def compare(a):
    fixture=native.validate_fixture(read(a.fixture,a.fixture_sha256));records={};values={};common=[]
    for name,action in [('normal','normal'),('d_result','d-restore'),('p_result','p-restore')]:
        value=read(getattr(a,name),getattr(a,name+'_sha256'));record=comparison.normalize_result(value)
        need(record['input_ids']==fixture['request_ids'],'all three results require exactly the same actual R IDs')
        replay=value.get('inner_replay',{});need(replay.get('action')==action and replay.get('generation_profile_id')==business.PROFILE_ID,'wrong replay action/profile')
        need({k:v['SHA256'] for k,v in replay['protocol_code'].items()}=={k:v['SHA256'] for k,v in protocol_code().items()},'request/compare protocol code differs')
        common.append(identity_check(value['identity'],action,value['started_at'],value['finished_at']-value['started_at']))
        need(replay['common_identity']==common[-1],'result common identity binding differs')
        need(replay['max_new_tokens']==replay['original_R_max_new_tokens'],'business continuation cap changed')
        if action.endswith('restore'):
            pp=replay.get('fresh_proof') or {};proof=read(Path(pp['path']),pp['SHA256']);fresh_check(proof,value['identity'],value['started_at'])
        records[name]=record;values[name]=value
    case_pin=values['normal']['inner_replay']['case_private']
    need(all(v['inner_replay']['case_private']['SHA256']==case_pin['SHA256'] for v in values.values()),'routes use different captured business cases')
    captured_case=read(Path(case_pin['path']),case_pin['SHA256']);case_inputs(captured_case)
    r_pin=captured_case['R_record'];reference_ids=captured_ids(read(Path(r_pin['path']),r_pin['SHA256']))[0][1]
    need(common[0]==common[1]==common[2],'B1/native4/SDK/DG/runtime env/joint numerical identity differs between routes')
    need(values['d_result']['namespace']==fixture['publication_evidence']['namespace']
         and values['p_result']['identity']['roles']['D']['namespace']==fixture['publication_evidence']['namespace'],
         'actual D producer/read namespace or retained D deployment differs')
    need(values['d_result']['namespace']!=values['p_result']['namespace'],'P-origin must have its own actual source namespace')
    d_process=values['d_result']['identity']['roles']['P']['model_root'];p_process=values['p_result']['identity']['roles']['P']['model_root']
    need(epoch(d_process)!=epoch(p_process),'different origin namespaces cannot be hot-switched on one cache process')
    baseline=records['normal'];need((int(baseline['aux'].get('local_reuse_len',-1)),int(baseline['aux'].get('remote_reuse_len',-1)))==(0,0),'normal baseline has actual cache hits')
    d_coverage=read(a.d_coverage,a.d_coverage_sha256) if a.d_coverage else None
    p_coverage=read(a.p_coverage,a.p_coverage_sha256) if a.p_coverage else None
    dc=comparison.coverage_check(records['d_result'],d_coverage,fixture,'D')
    pc=comparison.coverage_check(records['p_result'],p_coverage,fixture,'P')
    qualification=source_qualification(a,fixture)
    if d_coverage is not None:
        for key,expected in [('source_request_id',fixture['publication_evidence']['source_request_id']),
                             ('source_input_ids_sha256',fixture['source_input_ids_sha256']),('source_output_ids_sha256',fixture['source_output_ids_sha256']),
                             ('actual_computed_kv_end',fixture['publication_evidence']['computed_kv_end']),
                             ('actual_published_kv_end',fixture['publication_evidence']['published_kv_end'])]:
            if d_coverage.get(key)!=expected:dc['errors'].append('D source/read three-way binding differs: '+key)
        dc['passed']=not dc['errors']
    p_source=dict(passed=False,errors=[])
    try:
        replay=values['p_result']['inner_replay'];seed_pin=replay['P_seed_result'];marker_pin=replay['P_publication']
        seed=read(Path(seed_pin['path']),seed_pin['SHA256']);marker=read(Path(marker_pin['path']),marker_pin['SHA256'])
        p_publication_check(marker,seed,fixture,values['p_result']['namespace'])
        need(common_identity(seed['identity'])==common[0],'actual P-seed common numerical/artifact identity differs')
        need(seed['inner_replay']['action']=='p-seed','P source is not the explicit max1 prefill producer')
        if p_coverage is not None:
            for key,expected in [('source_request_id',seed['request_id']),('source_input_ids_sha256',seed['input_ids_sha256']),
                                 ('source_output_ids_sha256',seed['output_ids_sha256']),('actual_published_kv_end',marker['published_kv_end'])]:
                need(p_coverage.get(key)==expected,'P source/read binding differs: '+key)
        p_source['passed']=True
    except (ValueError,KeyError,TypeError,FileNotFoundError) as error:p_source['errors']=[str(error)]
    # Source runtime/Manager qualification is preserved in the actual fixture;
    # coverage_check adds actual fresh read interval and per-key three-end hashes.
    diffs=dict(D_vs_P=comparison.first_difference(records['d_result']['output_ids'],records['p_result']['output_ids']),
               D_vs_normal=comparison.first_difference(records['d_result']['output_ids'],baseline['output_ids']),
               P_vs_normal=comparison.first_difference(records['p_result']['output_ids'],baseline['output_ids']),
               normal_vs_captured_R=comparison.first_difference(baseline['output_ids'],reference_ids))
    complete=dc['passed'] and pc['passed'] and qualification['passed'] and p_source['passed'] and all(v is None for v in diffs.values())
    result=dict(state='INNER_THREE_ROUTE_EVIDENCE_AUDITED' if complete else 'INNER_THREE_ROUTE_NOT_VERIFIED',
        same_R_ids=True,checkpoint=fixture['checkpoint'],common_identity=common[0],D_remote_coverage=dc,P_remote_coverage=pc,
        D_source_qualification=qualification,P_source_commit=p_source,
        namespace_roles=dict(D_origin=dict(P=values['d_result']['identity']['roles']['P']['namespace'],D=values['d_result']['identity']['roles']['D']['namespace']),
            P_origin=dict(P_writer_reader=values['p_result']['identity']['roles']['P']['namespace'],D_publication=values['p_result']['identity']['roles']['D']['namespace'])),
        complete_output_ids_equal=all(v is None for v in diffs.values()),first_output_differences=diffs,
        verified_inner_chain_and_tokens_equal=complete,business_acceptance=False,formal_wheel_accepted=False,performance_accepted=False,
        official_task_and_tool_behavior='SEPARATE_EVIDENCE_REQUIRED',
        protocol_code=protocol_code(),
        output_metadata={name:dict(tokens=len(rec['output_ids']),SHA256=rec['output_ids_sha256']) for name,rec in records.items()})
    result['captured_R_output_metadata']=dict(tokens=len(reference_ids),SHA256=native.digest(reference_ids),record=r_pin)
    need(not a.output_dir.exists(),'new comparison output required');a.output_dir.mkdir(mode=0o700,parents=False)
    native.atomic(a.output_dir/'comparison-private.json',result)
    summary={k:v for k,v in result.items() if k not in ['first_output_differences','D_remote_coverage','P_remote_coverage']}
    summary['first_output_difference_positions']={k:position(v) for k,v in diffs.items()};summary['coverage_errors']={'D':dc['errors'],'P':pc['errors']}
    summary['private_report']=pin(a.output_dir/'comparison-private.json');native.atomic(a.output_dir/'summary.json',summary)
    return summary,0 if complete else 2


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--action',choices=ACTIONS,required=True);r.add_argument('--execute',action='store_true');r.add_argument('--return-logits',action='store_true')
    for name in ['case-private','identity-json']:
        r.add_argument('--'+name,type=Path,required=True);r.add_argument('--'+('case' if name=='case-private' else 'identity')+'-sha256',required=True)
    for name in ['fixture','fresh-proof','p-seed-result','p-publication']:
        r.add_argument('--'+name,type=Path);r.add_argument('--'+name+'-sha256')
    r.add_argument('--timeout-seconds',type=float,default=660);r.add_argument('--decode-grpc-port',type=int,default=18081);r.add_argument('--decode-http-port',type=int,default=18080)
    probe=sub.add_parser('probe');probe.add_argument('--model-root',type=Path,required=True);probe.add_argument('--plan-sha256',required=True)
    build=sub.add_parser('identity');build.add_argument('--action',choices=ACTIONS,required=True)
    for name in ['p-source','d-source']:
        build.add_argument('--'+name,type=Path,required=True);build.add_argument('--'+name+'-sha256',required=True)
    build.add_argument('--request-timeout-seconds',type=float,default=660)
    c=sub.add_parser('compare')
    for name in ['fixture','normal','d-result','p-result']:
        c.add_argument('--'+name,type=Path,required=True);c.add_argument('--'+name+'-sha256',required=True)
    for name in ['d-coverage','p-coverage','qualification']:
        c.add_argument('--'+name,type=Path);c.add_argument('--'+name+'-sha256')
    for q in [r,c,build]:q.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    need(os.getuid() in ({0,500} if a.command=='probe' else {500}) and socket.gethostbyname(socket.gethostname()) in MANIFESTS.keys()|{'10.44.61.73'},
         'C3 UID500 required; established controlled read-only probe alone may use UID0')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU driver must have empty CUDA_VISIBLE_DEVICES')
    for value in vars(a).values():
        if isinstance(value,Path):need(value.is_absolute() and '..' not in value.parts and TASK in value.parents,'device-local C3 task paths only')
    if a.command=='run':need(0<a.timeout_seconds<=1200 and a.decode_grpc_port==a.decode_http_port+1
        and a.decode_http_port in [18080,18089,18098,18107],'finite timeout/actual D DP port pair required')
    if a.command=='identity':need(0<a.request_timeout_seconds<=1200,'finite identity request budget required')
    if a.command=='probe':
        def stop(*unused):raise TimeoutError('finite120s read-only source probe deadline')
        signal.signal(signal.SIGALRM,stop);signal.alarm(120)
        try:
            try:result=probe_model(a);rc=0
            except MappingProofRequired as missing:result=dict(state='PENDING_ACTUAL_VMA_BACKING_PROOF',metadata=missing.metadata,observer_UID=os.getuid(),model_requests=0);rc=2
            print(json.dumps(result,sort_keys=True));return rc
        finally:signal.alarm(0)
    try:
        result,rc=(run(a) if a.command=='run' else compare(a) if a.command=='compare' else build_identity(a))
    except BaseException as error:
        # RPC/server exceptions can contain a business payload. Keep detail on
        # the device and emit only a bounded metadata failure to the controller.
        import traceback
        if not a.output_dir.exists():a.output_dir.mkdir(mode=0o700,parents=False)
        private=a.output_dir/('failure-private-'+str(time.time_ns())+'.json')
        native.atomic(private,dict(error_kind=type(error).__name__,traceback=traceback.format_exc()))
        result=dict(state='INNER_REPLAY_OR_COMPARISON_FAILED',error_kind=type(error).__name__,private_error=pin(private),
                    model_request_progress='SEE_DEVICE_NATIVE_STATUS_IF_PRESENT',business_acceptance=False,formal_wheel_accepted=False);rc=1
    print(json.dumps(result,sort_keys=True));return rc


if __name__=='__main__':raise SystemExit(main())
