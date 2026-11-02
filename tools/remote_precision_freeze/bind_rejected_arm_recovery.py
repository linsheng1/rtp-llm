"""Preserve a terminal empty ARM receiver; reuse only cases proven never armed.
No model restart, no lease extension and no bypass of the original admission.
"""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def bind(bench):
 old=bench/'c3_P_state_pilot_peer_v16.py';text=old.read_text()
 fn='''
def preserve_rejected_receiver(batch):
 recovery=batch['rejected_arm_recovery'];root=Path(batch['p_model_root']);receiver=root/'arm-receiver';archive=root/'arm-receiver-rejected-capture27'
 assert recovery['archive']==str(archive) and not archive.exists() and receiver.is_dir() and not receiver.is_symlink()
 p=receiver/'status.json';assert sha(p)==recovery['status_SHA256'];v=json.loads(p.read_text())
 assert v['state']=='FAILED_OR_EXPIRED_NO_RETRY' and v['failure']['reason']=='ARM exceeds actual model active lease' and not v['accepted'] and v['actual_exitcode']==1
 prior=process(v['process']['PID']);assert prior is None or str(prior['start_ticks'])!=str(v['process']['start_ticks']) or prior['state'] in ['Z','X','x'],'old metadata receiver still alive'
 assert not list((receiver/'ledger').iterdir()),'committed ledger cannot be reused'
 config=json.loads((root/'observer-runtime-config.json').read_text());assert sha(root/'observer-runtime-config.json')==batch['observer_config_sha256']
 for case in config['cases']:
  assert not Path(case['arm_path']).exists() and not list(Path(case['evidence_dir']).glob(case['nonce']+'-*live-rid-bound.json')),'case already armed or consumed'
 peer=C/'airline4-P-state-pilot-capture27-peer-P/status.json';terminal=json.loads(peer.read_text());assert terminal['state']=='BLOCKED' and not terminal['remaining_owned']
 # Rename preserves every old rejection byte; the new receiver creates its own
 # empty root and ledger at the unchanged observer path, with unused nonces.
 os.rename(receiver,archive)
 receipt=dict(state='PRESERVED_REJECTED_EMPTY_RECEIVER_ZERO_RPC',prior_status=dict(path=str(archive/'status.json'),SHA256=recovery['status_SHA256']),model_restarted=False,lease_extended=False,case_nonces_unchanged=True)
 with (root/'capture28-receiver-recovery.json').open('xb') as f:f.write(canonical(receipt));f.flush();os.fsync(f.fileno())

'''
 text=text.replace('\ndef main():',fn+'\ndef main():')
 needle="    path=R/'receiver-config.json';path.write_text(raw);tool=C/'c3_arm_http_transport_v1.py'"
 assert needle in text;text=text.replace(needle,"    preserve_rejected_receiver(batch)\n"+needle)
 new=bench/'c3_P_state_pilot_peer_v17.py';ast.parse(text)
 for p in [new,BASE/'device'/new.name]:
  if p.exists():assert p.read_text()==text
  else:p.write_text(text)
 old=bench/'c3_P_state_pilot_coordinator_v15.py';text=old.read_text().replace('c3_P_state_pilot_peer_v16','c3_P_state_pilot_peer_v17')
 before="min(json.loads(ip.read_text())['roles'][r]['deadline_epoch'] for r in ['P','D'])-1"
 assert before in text;text=text.replace(before,"min(json.loads(ip.read_text())['roles'][r]['deadline_epoch'] for r in ['P','D'])-421")
 new=bench/'c3_P_state_pilot_coordinator_v16.py';ast.parse(text)
 for p in [new,BASE/'device'/new.name]:
  if p.exists():assert p.read_text()==text
  else:p.write_text(text)
 # Readiness is unused here, but freeze checks the complete helper DAG.
 old=bench/'c3_P_diag_zero_request_readiness_v15.py';text=old.read_text().replace('c3_P_state_pilot_peer_v16','c3_P_state_pilot_peer_v17');new=bench/'c3_P_diag_zero_request_readiness_v16.py'
 for p in [new,BASE/'device'/new.name]:
  if p.exists():assert p.read_text()==text
  else:p.write_text(text)
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [bench/'c3_P_state_pilot_peer_v17.py',bench/'c3_P_state_pilot_coordinator_v16.py',new]}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--bench',type=Path,required=True);a=p.parse_args();print(json.dumps(bind(a.bench)))
