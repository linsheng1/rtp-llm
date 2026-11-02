"""Completed four-rank phase -> join existing own drain or request one -> old wrapper."""
import argparse,hashlib,json,os
from pathlib import Path
from urllib.request import urlopen
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert digest(a.plan)==a.sha256;v=json.loads(a.plan.read_text());assert os.getuid()==500 and digest(Path(__file__))==v['handoff']['SHA256'];old=Path(v['previous_root']);assert digest(old/'plan.json')==v['previous_plan_SHA256'];status=json.loads((old/'model-service-status.json').read_text());assert status['state'] in ['MODEL_READY_KV_EMPTY_PENDING','DRAINING_OWN_MODEL','MODEL_SERVICE_EXITED'],'old service not healthy/completing under original protocol'
for proof in v['completed_stages']:
 with urlopen(proof['URL'],timeout=10) as r:raw=r.read(1024**2+1)
 assert len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==proof['SHA256'];state=json.loads(raw);assert all(state[k]==x for k,x in proof['fields'].items()),'actual old batch incomplete'
 if proof.get('epoch_role'):
  assert all(state['epochs'][proof['epoch_role']][k]==status['model_root'][k] for k in ['PID','UID','start_ticks']),'completed batch belongs to different old epoch'
peer=json.loads((Path(v['failed_capture_peer_root'])/'status.json').read_text());assert peer['state']=='BLOCKED' and peer['remaining_owned']==[] and not peer.get('own_receiver_remaining_PID'),'old failed capture still has owned helper'
control=json.loads(Path(v['failed_capture_peer_controller']).read_text());assert control['cleanup_state']=='FINISHED' and not control['remaining_owned_processes'],'old peer finite controller cleanup incomplete'
wrapper=Path(v['wrapper']['path']);assert digest(wrapper)==v['wrapper']['SHA256'];assert digest(Path(v['model_root'])/'plan.json')==v['model_plan_SHA256'];claim=Path(v['handoff_claim']);assert not claim.exists();request_stop=status['state']=='MODEL_READY_KV_EMPTY_PENDING' and not (old/'stop').exists()
if request_stop:
 for row in [status['model_root']]+status['GPU_ranks']:
  proc=Path('/proc')/str(row['PID']);assert proc.stat().st_uid==row['UID']==500 and proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]==str(row['start_ticks']),'old owned epoch changed'
with claim.open('x') as out:json.dump(dict(plan_SHA256=a.sha256,old_process=status['model_root'],reason='failed sampling terminal; actual single RPC EOF; peer cleanup verified; corrected exact completion-set gate; D retained',action='request_own_drain' if request_stop else 'join_existing_original_drain'),out);out.flush();os.fsync(out.fileno())
if request_stop:(old/'stop').touch(exist_ok=False)
argv=['/opt/conda310/bin/python3.10','-B','-S',str(wrapper),'--previous-root',str(old),'--state-root',v['drain_root'],'--']+v['model_argv'];os.execv(argv[0],argv)
