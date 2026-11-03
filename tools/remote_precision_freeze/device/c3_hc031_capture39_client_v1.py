"""Single-request actual-D-history continuation diagnostic, never original R.

Every input token ID comes from ACTUAL G+Y[:-1][:7169] in one frozen D source.
No synthetic suffix, forced minimum/EOS, formal fixture or provenance assertion.
Native+Manager evidence admits only a metadata-scoped read diagnostic; exact
request provenance and all source/remote/destination bytes stay PENDING.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import socket
import time
from types import SimpleNamespace
from urllib.parse import urlparse,parse_qs

import c3_hc031_capture39_core_v1 as m
import c3_hc031_inner_replay_v1 as source_core

SOURCE_SHA='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa'
PINS={'c3_hc031_capture39_core_v1.py':'03523d4ffa963bea7d5c865947155fe8f1a0318aa3e9d51852cc731ba40ecb50',
      'token_fixture.py':'a1ed5084218ca8053788cc036d387a04b4cf723bfa60863c91bde206b784eb15',
      'business_token_fixture.py':'41a752d933e99413bd0195da9e668d649c9b4a4e0462069f8cc85270c0c3b427',
      'precision_compare.py':'a258eb6864eb3df49920050d45c266902a9dec11111fb0433bf8af59c86e2885'}
SCOPE='actual-D-history continuation diagnostic; NOT original business R; NOT formal acceptance'
CHECKPOINT=7168


def need(ok,message):
    if not ok:raise ValueError(message)


def dependencies():
    need(m.pin(Path(source_core.__file__))['SHA256']=='f51de136bbd84385704813903f3708eaf5cf10f7719787d2e4b4bd3f71a13204','historical source identity core differs')
    here=Path(__file__).resolve().parent
    for name,sha in PINS.items():need(m.pin(({'c3_hc031_capture39_core_v1.py':Path(m.__file__), 'token_fixture.py':Path(m.native.__file__), 'business_token_fixture.py':Path(m.business.__file__), 'precision_compare.py':Path(m.comparison.__file__)}[name].resolve()))['SHA256']==sha,'frozen diagnostic dependency differs: '+name)


def derive(source):
    g,y=m.native.source_ids(source)
    need(source.get('response_eof_complete') is True and source.get('final_aux',{}).get('pd_sep') is True,'actual complete D PD source required')
    need(len(g)==7041 and len(y)==295 and source.get('inner_replay',{}).get('action')=='d-source','frozen actual G7041/Y295 source required')
    source_core.identity_check(source['identity'],'d-source',source['started_at'],source['finished_at']-source['started_at'])
    sequence=g+y[:-1]
    need(len(sequence)>=CHECKPOINT+1,'actual computed-ID availability bound does not contain diagnostic continuation')
    ids=sequence[:CHECKPOINT+1]
    need(m.native.digest(ids)=='3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c','frozen actual new R SHA differs')
    return dict(request_ids=ids,p_prepare_ids=ids,checkpoint=CHECKPOINT,
        fixture_sha256=m.native.digest(dict(source_input_ids_sha256=m.native.digest(g),source_output_ids_sha256=m.native.digest(y),request_ids_sha256=m.native.digest(ids),scope=SCOPE)),
        scope=SCOPE,formula_is_runtime_evidence=False)


def metadata_coverage(source,native,manager):
    g,y=m.native.source_ids(source);namespace=source['namespace'];rid=source['request_id'];expected=dict(namespace=namespace,
        source_request_id=rid,source_input_ids_sha256=m.native.digest(g),source_output_ids_sha256=m.native.digest(y))
    for key,value in expected.items():need(native.get(key)==value,'actual native metadata/source binding differs: '+key)
    plans=[];commits=[]
    for item in native.get('lines',[]):
        text=item.get('line','') if isinstance(item,dict) else item
        pid=re.search(r'\[(\d+):\d+\]',text)
        if not pid:continue
        plan=re.search(r'retained_plan computed=(\d+) published=(\d+) tail=(\d+)',text)
        commit=re.search(r'committed computed_end=(\d+) canonical_keys=(\d+) new_locations=(\d+)',text)
        if plan:plans.append(dict(PID=int(pid[1]),computed=int(plan[1]),published=int(plan[2]),tail=int(plan[3])))
        if commit:commits.append(dict(PID=int(pid[1]),published=int(commit[1]),canonical_keys=int(commit[2]),new_locations=int(commit[3])))
    pairs={(p['PID'],p['computed'],p['published'],p['tail'],c['canonical_keys'],c['new_locations']) for p in plans for c in commits
           if p['PID']==c['PID'] and p['published']==c['published'] and CHECKPOINT<=p['published']<=p['computed']<=len(g)+len(y)-1
           and p['computed']-p['published']==p['tail'] and c['new_locations']>0}
    need(len(pairs)==1,'one actual same-PID retained-plan/commit pair must cover checkpoint')
    pid,computed,published,tail,key_count,new_count=next(iter(pairs))
    need(pid in {row['PID'] for row in source['identity']['roles']['D']['GPU_ranks']},'observed native commit PID not a source D GPU rank')
    expected_source=dict(namespace=namespace,request_id=rid,input_ids_sha256=m.native.digest(g),output_ids_sha256=m.native.digest(y),input_tokens=len(g),output_tokens=len(y))
    need(manager.get('state')=='PUBLICATION_OBSERVED_METADATA_READABLE' and manager.get('pending')==[],'actual Manager transaction/lookup metadata incomplete')
    for key,value in expected_source.items():need(manager.get('source',{}).get(key)==value,'actual Manager/source binding differs: '+key)
    selected=manager['transaction_evidence']['selected'];queries=manager['manager_read_evidence']['key_queries']
    need(selected.get('server_transaction_closed_successfully') is True and not selected.get('errors')
         and selected['instance_id'].startswith(namespace+'_') and len(selected['all_requested_keys'])==key_count
         and len(selected['newly_allocated_keys'])==new_count,'actual Manager commit/native count or namespace differs')
    start=selected['start'];finishes=selected['finish_rows']
    need(start['response']['header']['status']['code']=='OK' and len(finishes)==1 and finishes[0]['response']['header']['status']['code']=='OK'
         and finishes[0]['request']['write_session_id']==start['response']['write_session_id']
         and finishes[0]['request']['success_blocks']['offset']==new_count,'actual Start/Finish ACK session differs')
    need(queries and {str(q['key']) for q in queries}=={str(k) for k in selected['newly_allocated_keys']}
         and all(q['returned_locations_match_observed_commit'] is True and q['request']['instance_id']==selected['instance_id'] for q in queries),
         'actual Manager lookup does not return this observed committed D metadata')
    return dict(scope='NATIVE_AND_MANAGER_METADATA_ONLY',actual_computed_kv_end=computed,actual_published_kv_end=published,
        observed_D_PID=pid,canonical_keys=key_count,new_locations=new_count,D_new_interval=[7041,CHECKPOINT],D_new_tokens=127,
        supplied_native_provenance_verified=native.get('provenance_verified',False),exact_request_provenance='PENDING',
        source_remote_destination_byte_equality='PENDING',formal_acceptance=False)


def p_metadata_coverage(seed,fixture,manager,namespace):
    """Observed P Manager metadata only; no invented computed-KV boundary."""
    inputs,outputs=m.native.source_ids(seed)
    need(seed.get('response_eof_complete') is True and seed.get('final_aux',{}).get('pd_sep',False) is False
         and inputs==fixture['request_ids'][:CHECKPOINT+1] and len(outputs)==1,'actual complete P-only one-token seed of exact derived input required')
    meta=seed.get('actual_D_history_diagnostic',{})
    need(meta.get('action')=='p-seed' and meta.get('source',{}).get('SHA256')==SOURCE_SHA and meta.get('scope')==SCOPE
         and seed['namespace']==namespace,'P seed result/source/namespace binding differs')
    m.identity_check(seed['identity'],'p-seed',seed['started_at'],seed['finished_at']-seed['started_at'])
    need(seed['identity']['roles']['P']['host']=='10.44.12.175','actual P38 producer identity required')
    profile=seed['actual_native_request_profile'];m.business.validate_profile(profile)
    need(seed.get('generation_profile_id')==m.business.PROFILE_ID and profile.get('max_new_tokens')==1 and profile.get('role_addrs')==[],
         'actual P seed protobuf must bind normal sampling/max1/no D route')
    expected=dict(namespace=namespace,request_id=seed['request_id'],started_at=seed['started_at'],finished_at=seed['finished_at'],
        input_ids_sha256=m.native.digest(inputs),output_ids_sha256=m.native.digest(outputs),input_tokens=len(inputs),output_tokens=1)
    need(manager.get('schema')=='c3-P-origin-manager-metadata-v1' and manager.get('state')=='PUBLICATION_OBSERVED_METADATA_READABLE'
         and manager.get('pending')==[],'actual P Manager metadata report incomplete')
    for key,value in expected.items():need(manager.get('source',{}).get(key)==value,'actual P Manager/source binding differs: '+key)
    selected=manager['transaction_evidence']['selected'];reads=manager['manager_read_evidence'];queries=reads['key_queries']
    need(not manager['transaction_evidence'].get('pending',[]) and not reads.get('pending',[]) and not selected.get('errors')
         and selected.get('server_transaction_closed_successfully') is True,'P transaction/lookup pending or failed')
    instance=selected['instance_id'];keys=[str(k) for k in selected['all_requested_keys']];new=[str(k) for k in selected['newly_allocated_keys']]
    need(instance.startswith(namespace+'_') and len(keys)==len(set(keys))==14 and new and len(new)==len(set(new))
         and set(new).issubset(keys),'P Manager must cover all14 actual canonical keys')
    start=selected['start'];finishes=selected['finish_rows'];offset=manager['logger_UTC_offset_seconds']
    need(type(offset) is int and -86400<offset<86400 and len(finishes)==1,'actual P Manager timestamp offset and one Finish required')
    finish=finishes[0]
    def stamp(row):
        return datetime.datetime.strptime(row['request_begin_time'],'%Y.%m.%d,%H:%M:%S.%f').replace(tzinfo=datetime.timezone(datetime.timedelta(seconds=offset))).timestamp()
    start_at=stamp(start);finish_at=stamp(finish)
    need(seed['started_at']<=start_at<=finish_at<=seed['finished_at']+15,'P Manager transaction outside actual Pseed time window')
    for row,api in [(start,'StartWriteCache'),(finish,'FinishWriteCache')]:
        need(row.get('api_name')==api and row.get('client_ip')=='10.44.12.175' and row['request']['instance_id']==instance
             and row['response']['header']['status']['code']=='OK','actual P Manager client/API/instance/ACK differs')
    request_keys=[str(k) for k in start['request']['block_keys']]
    need(request_keys==keys,'P requested canonical keys differ from raw Start')
    mask=start['response']['block_mask']
    if set(mask)=={'offset'}:
        need(type(mask['offset']) is int and 0<=mask['offset']<=14,'actual P Start offset invalid');indices=list(range(mask['offset'],14))
    else:
        bits=mask.get('bool_masks',{}).get('values',[])
        need(set(mask)=={'bool_masks'} and len(bits)==14 and all(type(x) is bool for x in bits),'actual P Start bool mask invalid')
        indices=[i for i,present in enumerate(bits) if not present]
    need([keys[i] for i in indices]==new and len(start['response']['locations'])==len(new),'P Start mask/new keys/locations disagree')
    session=start['response']['write_session_id'];success=finish['request']['success_blocks']
    need(session and finish['request']['write_session_id']==session and set(success)=={'offset'} and success['offset']==len(new),
         'P Finish session or complete new-key ACK differs')
    info=reads['instance_response'];registered=info.get('instance_info',{})
    need(info['header']['status']['code']=='OK' and registered.get('instance_id')==instance,'P Manager registered instance differs')
    sizes={x['name']:int(x['size']) for x in registered['location_spec_infos']};groups={x['name']:x['spec_names'] for x in registered['location_spec_groups']}
    group_names=start['request'].get('location_spec_group_names',start['request'].get('location_spec_group_name'))
    need(isinstance(group_names,list) and len(group_names)==14 and len(queries)==14 and {str(q['key']) for q in queries}==set(keys),
         'all14 P canonical keys need actual same-instance queries')
    for q in queries:
        index=keys.index(str(q['key']));names=groups.get(group_names[index],[]);required={name:sizes[name] for name in names}
        need(required and q['required_spec_sizes']==required and q['request']['instance_id']==instance
             and [str(k) for k in q['request']['block_keys']]==[str(q['key'])] and q['response']['header']['status']['code']=='OK'
             and q.get('returned_locations_match_observed_commit') is True,'P key lookup/spec/commit binding differs')
        valid=q.get('valid_locations',[])
        def full(loc):
            specs=loc.get('location_specs',[]);names=[s['name'] for s in specs]
            if loc not in q['response'].get('locations',[]) or loc.get('type')!='ST_TAIRMEMPOOL' or int(loc.get('spec_size',-1))!=len(specs) or len(names)!=len(set(names)) or not set(required).issubset(names):return False
            for spec in specs:
                if spec['name'] not in required:continue
                uri=urlparse(spec['uri']);query=parse_qs(uri.query)
                if uri.scheme!='pace' or not uri.netloc or int(uri.path.strip('/'))<=0 or int(query['size'][0])!=required[spec['name']] or int(query['node_id'][0])<=0:return False
            return True
        need(valid and any(full(loc) for loc in valid),'P key lookup lacks full actual spec location metadata')
    return dict(scope='P_MANAGER_METADATA_ONLY',canonical_keys=14,new_locations=len(new),lookup_keys=14,
        P_native_computed_boundary='NOT_INDEPENDENTLY_OBSERVED',exact_request_provenance='PENDING',
        source_remote_destination_byte_equality='PENDING',formal_acceptance=False)


def run(a):
    source=m.read(a.source,a.source_sha256);fixture=derive(source);value=m.read(a.identity_json,a.identity_sha256)
    host=value['roles']['P']['host'];phase=('d-restore' if value['namespace']==source['namespace'] else 'p-restore') if a.action=='baseline' else a.action
    m.identity_check(value,phase,time.time(),a.timeout_seconds)
    need(value['D_publication_enabled'] is True,'diagnostic baseline/restore requires actual publication1 identity')
    if a.action=='d-restore':need(value['namespace']==source['namespace'],'D-history baseline/read must use actual common publication namespace')
    elif a.action!='baseline':need(value['namespace']!=source['namespace'] and value['roles']['D']['namespace']==source['namespace'],'P origin requires a separate P write/read namespace with retained D deployment')
    proof=None;coverage=None
    if a.action=='d-restore':
        need(all(getattr(a,k) is not None for k in ['native_evidence','native_evidence_sha256','manager_report','manager_report_sha256']),'actual native/Manager metadata evidence required')
        coverage=metadata_coverage(source,m.read(a.native_evidence,a.native_evidence_sha256),m.read(a.manager_report,a.manager_report_sha256))
    if a.action=='p-restore':
        need(all(getattr(a,k) is not None for k in ['p_seed_result','p_seed_result_sha256','p_manager_report','p_manager_report_sha256']),'actual P seed/Manager metadata evidence required')
        coverage=p_metadata_coverage(m.read(a.p_seed_result,a.p_seed_result_sha256),fixture,m.read(a.p_manager_report,a.p_manager_report_sha256),value['namespace'])
    if a.action.endswith('restore'):
        need(a.fresh_proof is not None and a.fresh_proof_sha256 is not None,'actual current fresh empty proof required')
        proof=m.read(a.fresh_proof,a.fresh_proof_sha256);m.fresh_check(proof,value,time.time())
    mode='bypass' if a.action=='baseline' else 'prepare' if a.action=='p-seed' else 'restore'
    options=SimpleNamespace(mode=mode,max_new_tokens=1024,pd=mode!='prepare',return_logits=False,namespace=value['namespace'],endpoint=host+':18081',
        output=a.output_dir/'native',timeout_seconds=a.timeout_seconds,frame_budget_bytes=64*1024**2)
    a.output_dir.mkdir(mode=0o700,parents=False,exist_ok=False);m.native.atomic(a.output_dir/'diagnostic-inputs-private.json',fixture)
    with m.business.requests(m.business.decode_route('10.44.61.71',a.decode_grpc_port,a.decode_http_port),gate_identity=value):
        gate=m.native.protobuf_gate(fixture['request_ids'],1 if mode=='prepare' else 1024,mode!='prepare',mode!='bypass',False)
        m.native.atomic(a.output_dir/'protobuf-gate.json',gate)
        if not a.execute:return dict(state='DIAGNOSTIC_PROTOBUF_CHECK_NO_REQUEST',scope=SCOPE,input_tokens=7169,input_ids_sha256=m.native.digest(fixture['request_ids']),model_requests=0,formal_acceptance=False),0
        if proof is not None:
            claims=m.C/'c3-inner-replay-proof-claims-v1';claims.mkdir(mode=0o700,exist_ok=True)
            need(not claims.is_symlink() and claims.stat().st_uid==500,'actual owned proof claim directory required')
            with (claims/(a.fresh_proof_sha256+'.json')).open('x') as f:json.dump(dict(output=str(a.output_dir),diagnostic=True),f);f.flush();os.fsync(f.fileno())
        result=m.native.run_rpc(options,fixture,value)
    result['actual_D_history_diagnostic']=dict(action=a.action,scope=SCOPE,source=m.pin(a.source),input_ids_sha256=m.native.digest(fixture['request_ids']),checkpoint=7168,
        derivation='actual source G + actual source Y excluding last sampled token, then first7169 actual IDs',metadata_coverage=coverage,
        native_evidence=m.pin(a.native_evidence) if a.native_evidence else None,manager_report=m.pin(a.manager_report) if a.manager_report else None,
        fresh_proof=m.pin(a.fresh_proof) if proof is not None else None,p_seed_result=m.pin(a.p_seed_result) if a.p_seed_result else None,
        p_manager_report=m.pin(a.p_manager_report) if a.p_manager_report else None,helper=m.pin(Path(__file__).resolve()),
        exact_request_provenance='PENDING',source_remote_destination_byte_equality='PENDING',formal_acceptance=False)
    m.native.atomic(a.output_dir/'native/result.json',result)
    aux=result['final_aux'];actual=(int(aux.get('local_reuse_len',-1)),int(aux.get('remote_reuse_len',-1)))
    correct=(actual==(0,0) if a.action=='baseline' else actual==(0,7168) if a.action.endswith('restore') else True)
    summary=dict(state='COMPLETE_ACTUAL_D_HISTORY_DIAGNOSTIC' if correct else 'DIAGNOSTIC_CACHE_GATE_FAILED',scope=SCOPE,action=a.action,
        result=m.pin(a.output_dir/'native/result.json'),input_tokens=len(result['input_ids']),input_ids_sha256=result['input_ids_sha256'],
        output_tokens=len(result['output_ids']),output_ids_sha256=result['output_ids_sha256'],actual_local_reuse_len=actual[0],actual_remote_reuse_len=actual[1],
        exact_request_provenance='PENDING',source_remote_destination_byte_equality='PENDING',metadata_coverage=coverage,formal_acceptance=False)
    m.native.atomic(a.output_dir/'summary.json',summary);return summary,0 if correct else 2


def compare(a):
    source=m.read(a.source,a.source_sha256);fixture=derive(source);records={};outputs={}
    for name in ['baseline','d_result','p_result']:
        path=getattr(a,name)
        if path is None:continue
        value=m.read(path,getattr(a,name+'_sha256'));record=m.comparison.normalize_result(value);meta=value['actual_D_history_diagnostic']
        need(record['input_ids']==fixture['request_ids'] and meta['source']['SHA256']==SOURCE_SHA and meta['scope']==SCOPE,'comparison must use this exact actual-D-history input')
        records[name]=record;outputs[name]=dict(tokens=len(record['output_ids']),SHA256=record['output_ids_sha256'])
    need('baseline' in records and len(records)>1,'baseline and at least one actual restore required')
    differences={name:m.position(m.comparison.first_difference(records[name]['output_ids'],records['baseline']['output_ids'])) for name in records if name!='baseline'}
    result=dict(state='TOKEN_ONLY_ACTUAL_D_HISTORY_DIAGNOSTIC_COMPARISON',scope=SCOPE,input_tokens=7169,input_ids_sha256=m.native.digest(fixture['request_ids']),
        output_metadata=outputs,first_difference_positions=differences,outputs_equal={name:diff is None for name,diff in differences.items()},
        exact_request_provenance='PENDING',source_remote_destination_byte_equality='PENDING',formal_acceptance=False,business_acceptance=False)
    a.output_dir.mkdir(mode=0o700,parents=False,exist_ok=False);m.native.atomic(a.output_dir/'summary.json',result);return result,0


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--action',choices=['baseline','d-restore','p-seed','p-restore','compare'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--timeout-seconds',type=float,default=660)
    p.add_argument('--decode-grpc-port',type=int,default=18081);p.add_argument('--decode-http-port',type=int,default=18080)
    for name in ['identity-json','fresh-proof','native-evidence','manager-report','p-seed-result','p-manager-report','baseline','d-result','p-result']:
        p.add_argument('--'+name,type=Path);p.add_argument('--'+('identity' if name=='identity-json' else name)+'-sha256')
    a=p.parse_args();os.umask(0o077);dependencies()
    need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and socket.gethostbyname(socket.gethostname()) in m.MANIFESTS.keys()|{'10.44.61.73'},'CPU-only actual C3 UID500 required')
    need(a.source_sha256==SOURCE_SHA and 0<a.timeout_seconds<=1200,'frozen actual source SHA and finite request budget required')
    for value in vars(a).values():
        if isinstance(value,Path):need(value.is_absolute() and '..' not in value.parts and m.TASK in value.parents,'C3 task-local paths only')
    if a.action!='compare':need(a.identity_json is not None and a.identity_sha256 is not None,'current actual runtime identity required')
    need(a.decode_grpc_port==a.decode_http_port+1 and a.decode_http_port in [18080,18089,18098,18107],'actual D DP port pair required')
    try:result,rc=(compare if a.action=='compare' else run)(a)
    except BaseException as error:
        import traceback
        if not a.output_dir.exists():a.output_dir.mkdir(mode=0o700,parents=False)
        private=a.output_dir/('failure-private-'+str(time.time_ns())+'.json');m.native.atomic(private,dict(error_kind=type(error).__name__,traceback=traceback.format_exc()))
        result=dict(state='ACTUAL_D_HISTORY_DIAGNOSTIC_FAILED',error_kind=type(error).__name__,private_error=m.pin(private),scope=SCOPE,formal_acceptance=False);rc=1
    print(json.dumps(result,sort_keys=True));return rc


if __name__=='__main__':raise SystemExit(main())
