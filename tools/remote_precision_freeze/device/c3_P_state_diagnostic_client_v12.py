"""Two explicit diagnostic cases; each CLI consumes its own fresh identity.
No old baseline identity gate is weakened. Transport is metadata HTTP only;
actual inference reuses frozen native.run_rpc/business.requests unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
H=C/'inner-replay-tools-v2'
if C.is_dir() and H.is_dir():sys.path[:0]=[str(C),str(H)]
import c3_inner_precision_replay_v6 as core
import c3_P_state_runtime_probe_v10 as probe
import c3_actual_d_history_diagnostic_v6 as diag
import c3_diagnostic_send_arm_adapter_v1 as arm
import c3_arm_http_transport_v1 as http
native=core.native;business=core.business;need=core.need
# P and D batch plans are explicit prepare inputs; neither old epoch is baked in.
PHELPER='8cd521110b99c47669efb1301e0212b0baef79c6066af50051ef7ed4a2fe51eb'
TRANSITION_QUERY_POINTS={'L0.attnHC.prenorm_raw_mixes24': (24, 'torch.float32'), 'L0.attnHC.residual4HC': (20480, 'torch.bfloat16'), 'L0.attnHC.pre_mix4': (4, 'torch.float32'), 'L0.attnHC.post4': (4, 'torch.float32'), 'L0.attnHC.comb16': (16, 'torch.float32'), 'L0.projectedAR5120': (5120, 'torch.bfloat16'), 'L0.FFN.input_norm5120': (5120, 'torch.bfloat16'), 'L0.FFN.result5120': (5120, 'torch.bfloat16'), 'L0.ffnHC.pre_mix4': (4, 'torch.float32'), 'L0.ffnHC.post4': (4, 'torch.float32'), 'L0.ffnHC.comb16': (16, 'torch.float32'), 'L0.block.result4HC': (20480, 'torch.bfloat16'), 'L1.beforeEngram4HC': (20480, 'torch.bfloat16'), 'L1.afterEngram4HC': (20480, 'torch.bfloat16'), 'L1.HC.residual4HC': (20480, 'torch.bfloat16'), 'L1.HC.readout5120': (5120, 'torch.bfloat16'), 'L1.attn_input_norm5120': (5120, 'torch.bfloat16')}
TRANSITION_QUERY_POINTS.update({'L3.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L4.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L5.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L6.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L7.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L2.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L3.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L4.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L5.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L6.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L7.stage.projectedAR5120': (5120, 'torch.bfloat16')})
TRANSITION_PARAMETERS={'L0.attnHC.fn_FP32': ([1, 491520], [24, 20480]), 'L0.attnHC.scale_FP32': ([1, 3], [3]), 'L0.attnHC.base_FP32': ([1, 24], [24])}
PROBE_SHA='b0c3b8d8239534aaa49ef869e7a111d59866b4e7684e5cbc5cf8868322bfee67'


def config_model(identity):
    p=identity['roles']['P'];r=p['model_root']
    return dict(epoch=dict(host=p['host'],role='P',**{k:r[k] for k in ['PID','UID','start_ticks']}),plan_sha256=p['plan_SHA256'],runtime_manifest_sha256=p['manifest_SHA256'],native_sha256=core.NATIVE)

def identity_gate(value,at,seconds=180,expected_epochs=None,*,expected_p_plan_sha256,expected_d_plan_sha256,expected_p_root,expected_d_root):
    need(core.sha(expected_p_plan_sha256) and core.sha(expected_d_plan_sha256) and Path(expected_p_root).parent==C and Path(expected_d_root).parent==C,'explicit expected paired batch root/plan pins required')
    need(value['schema']=='c3-p-state-diagnostic-replay-identity-v3' and value['source_commit']==core.B1 and value['base_native_source']==core.C1 and value['artifact_kind']=='native_incremental' and value['native_sha256']==core.NATIVE and value['DG_native_SHA256']==core.DG,'explicit new diagnostic identity/native gate')
    need(value['precision_config']==probe.EXPECTED_POLICY and value['precision_config_sha256']==native.digest(probe.EXPECTED_POLICY) and value['generation_profile_id']==business.PROFILE_ID,'original numerical/profile identity differs')
    need(set(value['roles'])=={'P','D'},'paired roles required')
    for role,host,plan,helper,schema in [('P','10.44.12.175',expected_p_plan_sha256,PHELPER,probe.DIAG_SCHEMA),('D','10.44.61.71',expected_d_plan_sha256,core.MODEL_HELPER,'c3-airline4-inner-model-plan-v7')]:
        row=value['roles'][role]
        need(row['source_documents']['plan']['path']==str(Path(expected_p_root if role=='P' else expected_d_root)/'plan.json'),'actual per-role model root differs from batch')
        need(row['role']==role and row['host']==host and row['plan_SHA256']==plan and row['model_service_helper_SHA256']==helper and row['model_plan_schema']==schema,'actual paired source/epoch schema differs')
        need(row['source_commit']==core.B1 and row['env_SHA256']==core.ENV and row['manifest_SHA256']==core.MANIFESTS[host] and row['publication']==1 and row['cache_profile']=='publication-candidate' and row['numeric_flags']==core.NUMERIC and row['precision_config']==probe.EXPECTED_POLICY,'actual paired runtime/numerical pins differ')
        need(row['model_READY'] is True and -2<=at-row['observed_at']<=45 and -2<=at-row['heartbeat_at']<=45 and row['deadline_epoch']>at+seconds+15,'actual READY45/active lease insufficient')
        ranks=row['GPU_ranks'];need(len(ranks)==4 and {r['world_rank'] for r in ranks}=={0,1,2,3} and len({core.epoch(r) for r in [row['model_root']]+ranks})==5,'five actual process epochs required')
        need(all(r['UID']==500 and type(r['PID']) is int and r['PID']>0 and str(r['start_ticks']).isdigit() for r in [row['model_root']]+ranks),'actual process identities differ')
        need(all(row['native_artifacts'][n]['SHA256']==h for n,h in core.NATIVE.items()) and row['DG_artifact']['SHA256']==core.DG and row['protocol_code']==core.protocol_code(),'source native/SDK/DG/protocol differs')
        if expected_epochs is not None:need(role_epochs(row)==expected_epochs[role],'case changed paired model epoch')
    observed=value['roles']['P']['diagnostic_runtime'];need(observed['code']['SHA256']==probe.OBSERVER_SHA and observed['base']['SHA256']==probe.BASE_OBSERVER_SHA and observed['bootstrap']['SHA256']==probe.BOOTSTRAP_SHA and core.sha(observed['config']['SHA256']),'actual observer/base/bootstrap/config source chain required')
    need(set(observed['hook_sources'])==set(probe.HOOK_MODULES) and all(core.sha(v['SHA256']) for v in observed['hook_sources'].values()),'six actual hook source pins required')
    need(value['namespace']==value['roles']['P']['namespace'] and value['D_publication_namespace']==value['roles']['D']['namespace'],'publication namespace pair differs')
    return config_model(value)

def role_epochs(row):return sorted([list(core.epoch(r)) for r in [row['model_root']]+row['GPU_ranks']])

def baseline_ids(row,ids):
    need(row['state']=='COMPLETE_NATIVE_RPC' and row.get('response_eof_complete') is True and row['input_ids']==ids and row['input_ids_sha256']==native.digest(ids),'actual complete same-wire baseline required')
    core.common_identity(row['identity'])
    need(row['final_aux'].get('local_reuse_len')==0 and row['final_aux'].get('remote_reuse_len')==0 and row['final_aux'].get('pd_sep') is True,'original normal baseline actual bypass gate')
    business.validate_profile(row['actual_native_request_profile']);outputs=native.token_ids(row['output_ids'])
    need(len(outputs)==143 and native.digest(outputs)==row['output_ids_sha256'],'same-candidate normal143 complete IDs required')
    return outputs

def barrier_gate(barrier,short,identity,at,expected_short_nonce):
    need(barrier['schema']=='c3-p-state-four-rank-short-drain-barrier-v2' and barrier['state']=='FOUR_ACTUAL_RANK_SHORT_CAPTURES_DRAINED' and barrier['nonce']==expected_short_nonce and barrier['request_id']==short['request_id'],'explicit short RID/nonce drain barrier differs')
    model=config_model(identity);need(barrier['model_identity_sha256']==native.digest(model)==short['model_identity_sha256'] and -2<=at-barrier['observed_at']<=45,'short barrier model/freshness differs')
    rows=barrier['ranks'];expected=identity['roles']['P']['GPU_ranks']
    need(len(rows)==4 and {r['world_rank'] for r in rows}=={0,1,2,3} and {core.epoch(r) for r in rows}=={core.epoch(r) for r in expected},'four actual same-epoch rank barriers required')
    total=0;coverage={}
    for r in rows:
        need(r['snapshot_tensor_references_released'] is True and r['terminal_event_query_completed'] is True and r['nonfinite_valid_count']==0,'short drain/refs/finite gate failed')
        for key in ('live_rid_marker','drain_marker','manifest'):
            p=r[key];need(Path(p['path']).is_absolute() and type(p['bytes']) is int and p['bytes']>0 and core.sha(p['SHA256']),'actual Ppeer evidence pin required')
        summaries=r['sample_summary'];need(r['sample_labels']==sorted({s['label'] for s in summaries}),'sample label summary differs')
        need(r['valid_count']==sum(s['valid_count'] for s in summaries) and r['nonfinite_valid_count']==sum(s['nonfinite_valid_count'] for s in summaries),'rank summary counts differ')
        for s in summaries:
            counts=s['per_position_valid_counts'];need(set(counts)=={'7167','7168'} and all(type(v) is int and v>=0 for v in counts.values()) and s['valid_count']==sum(counts.values()) and s['nonfinite_valid_count']==0,'bounded per-position valid/finite counts required')
            dst=coverage.setdefault(s['label'],{'7167':0,'7168':0})
            for p,v in counts.items():dst[p]+=v
        total+=r['valid_count']
    need(total>0 and barrier['coverage']==coverage,'actual four-rank aggregate coverage differs')
    for layer in (2,8):
        for label,pos in [(str(layer)+'._prefill_produce.full_hidden5120','7168'),(str(layer)+'._read_state.original_result1024_post_gather','7167')]:need(coverage.get(label,{}).get(pos,0)>=1,'short target position/label not actually covered')
    transition=barrier['transition_capture'];need(transition['schema']=='c3-p-hc-seam-four-rank-proof-v4' and transition['state']=='ACTUAL_HC_SEAM_QUERY_AND_PARAMETERS_CAPTURED' and transition['verifier_source']['SHA256']==probe.OBSERVER_SHA and core.sha(transition['report']['SHA256']),'actual four-rank HC seam raw capture proof required')
    need(transition['nonce']==barrier['nonce'] and transition['request_id']==short['request_id'] and transition['model_identity_sha256']==native.digest(model) and transition['query_position']==7168 and transition['coverage_complete'] is True,'transition actual case/query binding differs')
    points=transition['points'];need(len(points)==len(TRANSITION_QUERY_POINTS) and {r['point'] for r in points}==set(TRANSITION_QUERY_POINTS),'all actual query seams required')
    pids={r['PID'] for r in expected}
    for row in points:
        width,dtype=TRANSITION_QUERY_POINTS[row['point']];member=row['raw_member'];need(row['owner_PID'] in pids and row['query_position']==7168 and row['width']==width and row['original_dtype']==dtype and type(row['owner_cp_rank']) is int and 0<=row['owner_cp_rank']<4 and type(row['source_local_row']) is int and row['source_local_row']>=0,'typed actual transition query owner/width differs')
        need(Path(member['path']).is_absolute() and core.sha(member['SHA256']) and member['bytes']==2*width*(4 if dtype=='torch.float32' else 2) and row['row_offset_bytes']==member['bytes']//2,'actual transition query whole raw pin differs')
    parameters=transition['parameters'];need(len(parameters)==3 and {r['point'] for r in parameters}==set(TRANSITION_PARAMETERS),'three whole FP32 static parameter proofs required')
    for row in parameters:
        shape,original=TRANSITION_PARAMETERS[row['point']];need(row['shape']==shape and row['original_parameter_shape']==original and row['original_dtype']=='torch.float32' and row['byte_equal_across_ranks'] is True and len(row['raw_members'])==4 and {r['PID'] for r in row['raw_members']}==pids,'whole static parameter geometry/four ranks differs')
        need(all(Path(r['path']).is_absolute() and core.sha(r['SHA256']) and r['bytes']==shape[1]*4 for r in row['raw_members']),'actual static parameter whole raw pins differ')
    return dict(state='FOUR_RANK_SHORT_BARRIER_ACCEPTED',request_id=short['request_id'],model_identity_sha256=native.digest(model),raw_tensors_transferred=False)

def load_inputs(config):
    docs={k:core.read(Path(v['path']),v['SHA256']) for k,v in config['inputs'].items()}
    need(config['inputs']['source']['SHA256']==diag.SOURCE_SHA,'frozen actual D source differs')
    fixture=diag.derive(docs['source']);ids=fixture['request_ids']
    need(fixture==docs['fixture'] and len(ids)==7169 and fixture['checkpoint']==7168,'actual derived complete7169/checkpoint7168 required')
    need(native.digest(ids)==config['input_ids_sha256'],'actual case digest differs')
    normal=baseline_ids(docs['normal_result'],ids)
    need(config['namespace']==docs['source']['namespace'],'this capture uses the actual D-origin namespace')
    coverage=diag.metadata_coverage(docs['source'],docs['d_native_evidence'],docs['d_manager_report'])
    need(coverage['scope']=='NATIVE_AND_MANAGER_METADATA_ONLY' and coverage['formal_acceptance'] is False,'actual D metadata coverage semantics differ')
    return ids,fixture,normal

def prepare(a):
    out=a.outroot;need(out.is_absolute() and not out.exists(),'new client output root required');out.mkdir(mode=0o700)
    obs=core.read(a.observer_config_json,a.observer_config_sha256);value=core.read(a.identity_json,a.identity_sha256);model=identity_gate(value,time.time(),0,expected_p_plan_sha256=a.expected_p_plan_sha256,expected_d_plan_sha256=a.expected_d_plan_sha256,expected_p_root=a.expected_p_root,expected_d_root=a.expected_d_root)
    nonces=[a.short_nonce,a.full_nonce];need(len(set(nonces))==2 and all(isinstance(n,str) and arm.re.fullmatch('[0-9a-f]{32,64}',n) for n in nonces),'two explicit unique batch nonces required')
    need(obs['schema']=='c3-p-state-observer-batch-v4' and len(obs['cases'])==2 and [c['nonce'] for c in obs['cases']]==nonces and [c['expected_prefix_tokens'] for c in obs['cases']]==[7168,0],'exact v2 short/full config required')
    need(all(c['plan_sha256']==a.expected_p_plan_sha256 and c['runtime_manifest_sha256']==core.MANIFESTS['10.44.12.175'] and c['native_sha256']==core.NATIVE for c in obs['cases']),'actual v2 observer pins differ')
    need(core.pin(a.observer_config_json)['SHA256']==value['roles']['P']['diagnostic_runtime']['config']['SHA256'] and all(c['additional_layers']==[0,1] and c['additional_query_positions']==[7168] and c['additional_budget_bytes']==4194304 for c in obs['cases']),'prepared observer configuration must equal actual admitted runtime')
    hook_expected={n:p['SHA256'] for n,p in value['roles']['P']['diagnostic_runtime']['hook_sources'].items()};need(all(c['hook_source_sha256']==hook_expected for c in obs['cases']),'prepared hook sources differ from actual source observation')
    need(a.deadline_epoch>time.time() and a.deadline_epoch<=value['roles']['P']['deadline_epoch'],'ARM deadline exceeds actual P active lease')
    paths={k:getattr(a,k) for k in ('source','d_native_evidence','d_manager_report','normal_result')}
    pins={k:core.pin(p) for k,p in paths.items()}
    need(all(pins[k]['SHA256']==getattr(a,k+'_sha256') for k in pins),'explicit input file SHA differs')
    need(a.source_sha256==diag.SOURCE_SHA,'actual ce68 source required')
    fixture=diag.derive(core.read(a.source,a.source_sha256));native.atomic(out/'derived-fixture-private.json',fixture);pins['fixture']=core.pin(out/'derived-fixture-private.json')
    config=dict(schema='c3-P-state-two-phase-client-v4',expected_d_plan_sha256=a.expected_d_plan_sha256,expected_p_root=str(a.expected_p_root),expected_d_root=str(a.expected_d_root),expected_p_plan_sha256=a.expected_p_plan_sha256,nonces=nonces,outroot=str(out),inputs=pins,namespace=value['namespace'],input_ids_sha256=obs['cases'][0]['input_ids_sha256'],observer_config=core.pin(a.observer_config_json),cases=obs['cases'],expected_epochs={r:role_epochs(v) for r,v in value['roles'].items()},deadline_epoch=a.deadline_epoch,timeout_seconds=180,max_new_tokens=1024,model_requests_max=2,HTTP_endpoint='http://10.44.12.175:19109/arm')
    ids,fixture,_=load_inputs(config);route=business.decode_route('10.44.61.71',18081,18080);receiver=[]
    for i,c in enumerate(obs['cases']):
        with business.requests(route):req=native.make_request(ids,1024,True,i==0,False);gate=native.protobuf_gate(ids,1024,True,i==0,False)
        need(gate['input_tokens']==7169,'protobuf wire7169 omission')
        cfg=req.generate_config;summary={k:getattr(cfg,k) for k in arm.PROFILE_FIELDS if hasattr(cfg,k)};summary['random_seed']=cfg.random_seed.value
        profile=dict(id=business.PROFILE_ID,summary=summary,config_SHA256=hashlib.sha256(cfg.SerializeToString(deterministic=True)).hexdigest());decode=[{k:getattr(r,k) for k in ('role','role_str','ip','grpc_port','http_port')} for r in cfg.role_addrs]
        p=value['roles']['P'];modelroot=str(Path(p['source_documents']['plan']['path']).parent);ledger=Path(c['arm_path']).parent
        receiver.append(dict(label='short' if i==0 else 'full',nonce=c['nonce'],identity_ref=c['identity_ref'],model_root=modelroot,runtime_manifest=p['source_documents']['manifest']['path'],ledger_root=str(ledger),input_ids_sha256=config['input_ids_sha256'],input_tokens=7169,namespace=value['namespace'],endpoint='10.44.12.175:18081',profile=profile,decode_routes=decode,deadline_epoch=a.deadline_epoch,plan_sha256=a.expected_p_plan_sha256,runtime_manifest_sha256=model['runtime_manifest_sha256'],native_sha256=core.NATIVE))
    rroot=str(Path(receiver[0]['ledger_root']).parent);need(receiver[0]['ledger_root']==receiver[1]['ledger_root'],'shared receiver ledger root differs')
    rconfig=dict(mode='production',root=rroot,bind_host='10.44.12.175',port=19109,allowed_source_ip='10.44.61.73',seconds=1200,cases=receiver)
    native.atomic(out/'prepared-config.json',config);native.atomic(out/'receiver-config.json',rconfig)
    result=dict(state='TWO_PHASE_DIAGNOSTIC_CLIENT_PREPARED_NO_REQUEST',prepared_config=core.pin(out/'prepared-config.json'),receiver_config=core.pin(out/'receiver-config.json'),model_requests=0,wire_input_tokens=7169,max_new_tokens=1024,nonces=nonces)
    native.atomic(out/'prepare-summary.json',result);return result

def phase(a,*,transport=None):
    config=core.read(a.config_json,a.config_sha256);need(config['schema']=='c3-P-state-two-phase-client-v4','new diagnostic client config required')
    root=Path(config['outroot'])/a.command;need(not root.exists(),'single case attempt preserved; no retry');root.mkdir(mode=0o700)
    state=dict(state='ADMITTING',phase=a.command,started_at=time.time(),model_requests=0);wrappers=[]
    try:
        value=core.read(a.identity_json,a.identity_sha256);model=identity_gate(value,time.time(),180,config['expected_epochs'],expected_p_plan_sha256=config['expected_p_plan_sha256'],expected_d_plan_sha256=config['expected_d_plan_sha256'],expected_p_root=config['expected_p_root'],expected_d_root=config['expected_d_root']);ids,fixture,normal=load_inputs(config)
        i=0 if a.command=='short' else 1;c=config['cases'][i]
        need([row['nonce'] for row in config['cases']]==config['nonces'] and all(row['plan_sha256']==config['expected_p_plan_sha256'] for row in config['cases']),'prepared case batch pins differ')
        if i==0:
            core.fresh_check(core.read(a.fresh_proof,a.fresh_proof_sha256),value,time.time())
            claims=C/'c3-inner-replay-proof-claims-v1';claims.mkdir(mode=0o700,exist_ok=True)
            need(not claims.is_symlink() and claims.stat().st_uid==500,'owned fresh proof claim directory required')
            fd=os.open(claims/(a.fresh_proof_sha256+'.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as f:f.write(arm.canonical(dict(output=str(root),diagnostic=True)));f.flush();os.fsync(f.fileno())
        else:
            short=core.read(Path(config['outroot'])/'short/summary.json',a.short_result_sha256)
            need(short['state']=='SHORT_NATIVE_EOF_CACHE_GATE_PASS' and short['actual_local_reuse_len']==0 and short['actual_remote_reuse_len']==7168 and short['response_eof_complete'] is True,'prior actual short completion required')
            state['short_barrier']=barrier_gate(core.read(a.short_barrier,a.short_barrier_sha256),short,value,time.time(),config['nonces'][0])
        def current_model():return identity_gate(value,time.time(),180,config['expected_epochs'],expected_p_plan_sha256=config['expected_p_plan_sha256'],expected_d_plan_sha256=config['expected_d_plan_sha256'],expected_p_root=config['expected_p_root'],expected_d_root=config['expected_d_root'])
        output=root/'native';binding=arm.ArmBinding(output=output,namespace=config['namespace'],endpoint='10.44.12.175:18081',input_ids_sha256=config['input_ids_sha256'],fixture_sha256=fixture['fixture_sha256'],rpc_identity_sha256=native.digest(value),profile_id=business.PROFILE_ID,model=model,nonce=c['nonce'],deadline_epoch=config['deadline_epoch'])
        options=SimpleNamespace(mode='restore' if i==0 else 'bypass',max_new_tokens=1024,pd=True,return_logits=False,namespace=config['namespace'],endpoint=binding.endpoint,output=output,timeout_seconds=180,frame_budget_bytes=64*1024**2)
        chosen=transport if transport is not None else http.HTTPArmTransport()
        with business.requests(business.decode_route('10.44.61.71',18081,18080),gate_identity=value):
            with arm.instrument_generated_stub(binding,chosen,current_model) as wrappers:
                result=native.run_rpc(options,fixture,value)
        aux=result['final_aux'];need(result['state']=='COMPLETE_NATIVE_RPC' and result['response_eof_complete'] is True and result['input_ids']==ids and aux.get('pd_sep') is True,'actual same-wire complete PD request required')
        need((int(aux.get('local_reuse_len',-1)),int(aux.get('remote_reuse_len',-1)))==((0,7168) if i==0 else (0,0)),'actual cache gate failed')
        state.update(state='SHORT_NATIVE_EOF_CACHE_GATE_PASS' if i==0 else 'FULL_NATIVE_EOF_CACHE_GATE_AND_IDS_COMPARED',model_requests=1,request_id=result['request_id'],nonce=c['nonce'],model_identity_sha256=native.digest(model),identity=core.pin(a.identity_json),input_ids_sha256=result['input_ids_sha256'],output_ids_sha256=result['output_ids_sha256'],output_tokens=len(result['output_ids']),wire_input_tokens=len(ids),max_new_tokens=1024,response_eof_complete=True,actual_local_reuse_len=aux['local_reuse_len'],actual_remote_reuse_len=aux['remote_reuse_len'],native_result=core.pin(output/'result.json'),ARM=wrappers[0].committed_arm,expected_epochs=config['expected_epochs'],finished_at=time.time(),formal_acceptance=False)
        if i:
            state['full_output_ids_equal_normal143']=result['output_ids']==normal;state['normal_output_tokens']=143
            need(state['full_output_ids_equal_normal143'],'full complete IDs differ from same-candidate normal143; retain original result')
        native.atomic(root/'summary.json',state);return state
    except BaseException as error:
        committed=[w.committed_arm for w in wrappers if w.committed_arm is not None]
        if committed and state['model_requests']!=1:state.update(model_requests='UNKNOWN_AFTER_ARM_AND_SEND_ENTRY',ARM=committed[-1])
        state.update(state='FAILED_OR_BLOCKED_RETAINED_NO_RETRY',error_kind=type(error).__name__,error=str(error)[:1000],finished_at=time.time());native.atomic(root/'summary.json',state);raise

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['prepare','short','full']);p.add_argument('--identity-json',type=Path,required=True);p.add_argument('--identity-sha256',required=True)
    for n in ('d-native-evidence','d-manager-report','source','observer-config-json','case-private','fixture','p-seed-result','p-publication','normal-result','outroot','config-json','fresh-proof','short-barrier'):p.add_argument('--'+n,type=Path)
    for n in ('d-native-evidence-sha256','d-manager-report-sha256','source-sha256','observer-config-sha256','case-sha256','fixture-sha256','p-seed-result-sha256','p-publication-sha256','normal-result-sha256','config-sha256','fresh-proof-sha256','short-barrier-sha256','short-result-sha256'):p.add_argument('--'+n)
    p.add_argument('--expected-p-plan-sha256');p.add_argument('--short-nonce');p.add_argument('--full-nonce');p.add_argument('--expected-d-plan-sha256');p.add_argument('--expected-p-root',type=Path);p.add_argument('--expected-d-root',type=Path);p.add_argument('--deadline-epoch',type=float);a=p.parse_args()
    need(os.getuid()==500 and socket.gethostbyname(socket.gethostname())=='10.44.61.73' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','private CPU client only seed61.73 UID500/CVD empty')
    for module,h in [(core,'37e17a974492dcd98dadf0ac3c487765b8ae24e622594753d42573b2ed18b5e1'),(diag,'98237ef2bd4c08ad6a96f1ebb80f9075891547ab0789cd799341b2e189d68e4e'),(probe,PROBE_SHA),(arm,http.ADAPTER_SHA),(http,'05ab9d4f61aaf6636e1b418e20eff2b40ca4b4f8b316f56555203d5aaea33914')]:need(core.pin(Path(module.__file__))['SHA256']==h,'frozen diagnostic dependency differs')
    result=prepare(a) if a.command=='prepare' else phase(a);print(json.dumps({k:v for k,v in result.items() if k in ['state','model_requests','prepared_config','receiver_config','request_id','nonce','output_tokens','wire_input_tokens','max_new_tokens','full_output_ids_equal_normal143']}));return 0
if __name__=='__main__':raise SystemExit(main())
