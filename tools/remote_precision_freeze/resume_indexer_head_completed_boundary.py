"""Recover only the witnessed stop-marker collision, using the original model/controller.

No old completion marker is repaired or fabricated. The cancelled outer tree
must be terminal and every recorded epoch gone before the already-prepared
fresh-P model can be submitted directly to the original controller.
"""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--recipe',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  recipe,rp=f.doc(a.recipe);model,mp=f.doc(recipe['prepared_model_plan']);q,qp=f.doc(recipe['quiescence']);old=q['headwheel-P-source1-status.json']
  f.need(model['role']=='P' and model['CPU']['state']=='ACTUAL_CPU_CONFIG_PASS_ZERO_RPC' and not q['remaining_same_epoch'] and q['gpu_query_rc']==0 and not q['gpu_apps'].strip(),'completed exact fresh-P preparation and quiescence required')
  f.need(old['state']=='CANCELLED' and old['reason']=='task stop requested' and old['command_returncode']==-9 and old['cleanup_state']=='FINISHED' and old['remaining_owned_processes']==[],'exact outer stop collision signature')
  f.need(recipe['previous_root']==f.CONTROL+'/airline4-P-headwheel-source1' and recipe['old_controller_status']==f.CONTROL+'/headwheel-P-source1-status.json','only this witnessed boundary')
  pins=dict(model['local_pins'],recovery_entry=f.read(__file__)[1],recovery_recipe=rp,prepared_model=mp,quiescence=qp)
  for k,path in recipe['completed_receipts'].items():
   r,pin=f.doc(path);f.need(r['controller']['state']=='SUCCEEDED' and r['native_safe']['EOF'],'completed source/normal requests required');pins['completed:'+k]=pin
  job=dict(model['job']);i=job['argv'].index('--');job['argv']=job['argv'][i+1:];f.need(job['argv'][3]==f.CONTROL+'/c3_indexer_head_fullwheel_model_v1.py','same prepared original full-wheel service')
  v=dict(model,job=job,local_pins=pins,recovery=dict(previous_root=recipe['previous_root'],old_controller_status=recipe['old_controller_status'],expected_epochs=q['expected'],old_model_root_returncode='UNKNOWN_SERVICE_WRAPPER_CANCELLED',old_final_marker_written=False))
  print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
 v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'plan drift')
 for p in v['local_pins'].values():f.need(f.read(p['path'])[1]==p,'frozen input drift')
 if a.command=='check':print(json.dumps(dict(state='OFFLINE_RECOVERY_PINS_PASS',model_requests=0)));return
 if a.command=='render':print(json.dumps(v['job']));return
 f.need(a.receipt is not None and not a.receipt.exists(),'unique submission receipt');sys.path.insert(0,v['bench']);from remote_control import remote
 from cluster23_dispatch import submit
 code='''import json,os,time
from pathlib import Path
r=%r;modelroot=Path(%r);assert os.getuid()==500 and not (modelroot/'model-service-status.json').exists(),'fresh model must not already be started'
old=json.loads(Path(r['old_controller_status']).read_text());assert old['state']=='CANCELLED' and old['cleanup_state']=='FINISHED' and not old['remaining_owned_processes']
for e in r['expected_epochs']:
 p=Path('/proc')/str(e['PID'])
 if p.exists():
  f=(p/'stat').read_text().rsplit(')',1)[1].split();assert not (p.stat().st_uid==e['UID'] and f[19]==str(e['start_ticks']) and f[0] not in ['Z','X','x']),'old owned epoch remains'
for name,state in [('consumer91b-P-v7','READY'),('hc031-static-P-v4','RUNNING')]:
 s=json.loads((Path(%r)/name/'status.json').read_text());assert s['state']==state and time.time()-s['heartbeat_at']<=45 and s['deadline_epoch']>time.time()+1815,'unchanged live dependency admission'
print(json.dumps(dict(state='ACTUAL_OLD_OWNED_TREE_GONE_PREPARED_FRESH_P_ADMITTED',model_requests=0,old_root_exit_code='UNKNOWN_NOT_RELABELLED')))
'''%(v['recovery'],v['model_root'],f.CONTROL)
 actual=remote('cluster3',v['job']['host'],code,60);j=v['job'];launch=submit('cluster3',j['host'],j['name'],j['argv'],j['timeout'],j['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,actual_boundary=actual,launch=launch));print(json.dumps(dict(state='ORIGINAL_MODEL_CONTROLLER_SUBMITTED_NOT_READY',model_requests=0)))
if __name__=='__main__':main()
