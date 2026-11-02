"""One bounded read of existing terminal evidence; no model actions or polling."""
import argparse,json,sys,hashlib,time
from pathlib import Path
import freeze_stage as m

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--three-route-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan,pin=m.doc(a.plan);m.need(pin['SHA256']==a.sha256,'plan SHA differs');m.validate(plan)
    reference,refpin=m.doc(a.three_route_reference)
    m.need(reference['candidate_commit']==plan['candidate_commit'],'reference candidate differs')
    recipe,_=m.doc(plan['local_pins']['recipe']['path']);sys.path.insert(0,recipe['bench'])
    from remote_control import remote
    helper=m.read(__file__)[1];prefix='airline4-P-state-pilot-'+plan['tag']
    code='''import json,hashlib,time,os
from pathlib import Path
c=Path(%r);root=c/(%r+'-coordinator');ref=%r;report_path=c/%r
assert os.getuid()==500 and not report_path.exists()
def doc(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();raw=p.read_bytes();return json.loads(raw),dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest())
status,sp=doc(root/'status.json');assert status['state']=='BOTH_ACTUAL_CLIENT_PHASES_COMPLETE_CAPTURE_REVIEW_PENDING' and status['model_requests']==status['request_dispatches']==2
prepare,pp=doc(root/'client-prepare-exit.json');assert prepare['actual_child_returncode']==0 and not prepare['remaining_owned'] and not prepare['timed_out']
new={};results={};barriers={}
for phase in ['short','full']:
 v,rp=doc(root/'client'/phase/'native/result.json');summary,sump=doc(root/'client'/phase/'summary.json');exit,ep=doc(root/(phase+'-request-exit.json'))
 assert v['state']=='COMPLETE_NATIVE_RPC' and v['response_eof_complete'] and summary['model_requests']==1 and exit['actual_child_returncode']==0 and not exit['remaining_owned']
 assert v['termination_observation']['reached_output_cap'] is False
 b,bp=doc(root/(phase+'-barrier.json'));assert len(b['ranks'])==4 and {r['world_rank'] for r in b['ranks']}=={0,1,2,3} and b['request_id']==v['request_id'] and all(r['snapshot_tensor_references_released'] and r['terminal_event_query_completed'] for r in b['ranks'])
 assert b['transition_capture']['coverage_complete'] is True
 new[phase]=v;results[phase]=dict(result=rp,summary=sump,request_id=v['request_id'],input_tokens=len(v['input_ids']),input_SHA256=v['input_ids_sha256'],output_tokens=len(v['output_ids']),output_SHA256=v['output_ids_sha256'],EOF=True,reached_cap=False,runtime_EOS_verified=v['termination_observation']['runtime_model_eos_verified'],local_reuse=v['final_aux']['local_reuse_len'],remote_reuse=v['final_aux']['remote_reuse_len'],started_at=v['started_at'],finished_at=v['finished_at'])
 barriers[phase]=dict(proof=bp,ranks=4,state=b['state'],transition_state=b['transition_capture']['state'])
old={}
for key,row in ref['rows'].items():
 v,rp=doc(row['result']['path']);assert rp['SHA256']==row['result']['SHA256'];old[key]=v
normal=old['normal_current'];d=new['short'];full=new['full']
assert all(v['input_ids']==normal['input_ids'] for v in [d,full,old['Premote'],old['Dremote']])
profiles=[v['actual_native_request_profile'] for v in [d,full,normal,old['Premote'],old['Dremote']]];assert all(v==profiles[0] for v in profiles)
left,right=d['output_ids'],full['output_ids'];diff=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),min(len(left),len(right)) if len(left)!=len(right) else None)
proof,proofpin=doc(root/'short-empty-proof.json');age=d['started_at']-proof['observation']['latest_metric']['at'];assert -2<=age<=75
identity,_=doc(root/'short-identity-identity.json');epoch={r:{k:identity['roles'][r]['model_root'][k] for k in ['PID','UID','start_ticks']} for r in ['P','D']}
value=dict(state='FROZEN_ENTRY_TWO_REQUESTS_AND_FOUR_RANK_PASS_PRECISION_STILL_DIFFERENT',at=time.time(),plan_SHA256=%r,collector_SHA256=%r,candidate_commit=%r,model_requests_completed=2,total_candidate_requests_completed=ref['actual_completed_model_requests']+2,complete_device_prepare=dict(exit=pp,rc=0),coordinator_status=sp,results=results,barriers=barriers,epochs=epoch,fresh_empty_proof=proofpin,fresh_empty_age_at_native_request_seconds=age,all_input_ids_identical=True,all_generation_profiles_identical=True,full_ids=dict(new_normal_equals_reference=full['output_ids']==normal['output_ids'],new_Dremote_equals_reference=d['output_ids']==old['Dremote']['output_ids'],Premote_equals_new_normal=old['Premote']['output_ids']==full['output_ids'],Dremote_equals_new_normal=left==right,first_difference_zero_based=diff,first_difference_one_based=diff+1 if diff is not None else None),whole_precision_fixed=False,artifact_kind='native_incremental_inner_not_wheel_acceptance',wheel_SHA256=None,raw_IDs_or_tensors_exported=False)
with report_path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
raw=report_path.read_bytes();print(json.dumps(dict(report=dict(path=str(report_path),SHA256=hashlib.sha256(raw).hexdigest()),value=value)))
'''%(m.CONTROL,prefix,reference,'runtime-freeze-'+plan['tag']+'-actual.json',a.sha256,helper['SHA256'],plan['candidate_commit'])
    result=remote('cluster3',m.HOSTS['receiver'],code,60)
    terminal={}
    for role in ['receiver','P','D']:
        job=plan['jobs'][role]
        code='''import json\nfrom pathlib import Path\nc=Path(%r);v=json.loads((c/%r).read_text());assert v['state']=='SUCCEEDED' and v['command_returncode']==0\nprint(json.dumps({k:v.get(k) for k in ['state','command_returncode','finished_at','reason','controller']}))'''%(m.CONTROL,job['name']+'-status.json')
        terminal[role]=remote('cluster3',job['host'],code,60)
    result.update(controllers=terminal,reference_receipt=refpin,collector=helper)
    m.save_new(a.output,result)
    print(json.dumps(dict(state=result['value']['state'],model_requests=2,output=str(a.output))))
if __name__=='__main__':main()
