"""Bind two current-source finite D-trajectory captures to proven request controllers."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
B=HERE.parents[3]/'benchmarks/publication_precision_20261101'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def put(name,s):
 for p in [HERE/'device'/name,B/name]:
  if p.exists():assert p.read_text()==s,'immutable adapter drift '+str(p)
  else:p.write_text(s)
def get(name):return (HERE/'device'/name).read_text()
def main():
 core=digest(get('c3_hc031_capture38_core_v2.py'));diag=digest(get('c3_hc031_capture38_client_v2.py'))
 oldcore='bb2b5057a25f47946fce2af99e7221ca51eb503f8f63c9a4d8b5a7e9d8da4e3b'
 sender=get('c3_history36_baseline_sender_v1.py').replace(oldcore,core).replace('c3_inner_precision_replay_v10','c3_hc031_capture38_core_v2').replace('HISTORY36','CAPTURE38')
 # --fresh-proof is now an action flag: proof is taken after both probes at RESET.
 start=sender.index(' if a.fresh_proof:\n');end=sender.index('\n import socket,stat',start)
 sender=sender[:start]+sender[end:]
 start=sender.index("   if a.role=='D':");end=sender.index('   current=json.loads',start)
 sender=sender[:start]+sender[end:]
 marker='   exports=[]\n'
 fresh=get('c3_history36_source_sender_v1.py');fresh=fresh[fresh.index("   if action=='RESET' and a.role=='P':"):fresh.index('   data=json.dumps(ack',fresh.index("   if action=='RESET' and a.role=='P':"))]
 fresh=fresh.replace("if action=='RESET' and a.role=='P':","if action=='RESET' and a.fresh_proof:")
 skip="""   if a.role=='P':
    assert all(item['record'] is None and item['initial_latches_absent'] for item in controls),'P profile must not allocate hooks'
    ack=dict(action=action,role='P',rank_controls=[],model_root=value['model_root'],scope='EXPLICIT_D_ONLY_PROFILE_P_HAS_NO_HOOKS',at=time.time(),model_requests=0)
"""+fresh+"""    with urlopen(Request('http://10.44.61.73:19107/capture-ack/P',data=json.dumps(ack).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=20) as response:assert response.status==200
    completed.add(action)
    if action=='EXPORT':break
    time.sleep(1);continue
"""
 # Reindent reused fresh block beneath the explicit P-only branch.
 skip=skip.replace(fresh,''.join(' '+line+'\n' for line in fresh.splitlines()))
 sender=sender.replace(marker,skip+marker)
 start=sender.index("    result=socket_call(record,'snapshot'");end=sender.index("    if action=='EXPORT':",start)
 sender=sender[:start]+"""    result=socket_call(record,'reset' if action=='RESET' else 'snapshot','capture-'+str(command.get('request_id',0)))
"""+sender[end:]
 put('c3_hc031_capture38_sender_v2.py',sender)
 receiver=get('c3_history36_baseline_receiver_v1.py').replace(oldcore,core).replace('c3_inner_precision_replay_v10','c3_hc031_capture38_core_v2').replace('c3_history36_baseline_client_v1','c3_hc031_capture38_client_v2').replace('36b5add8859c0926f8082cded3bfa4400b9a2f50d7a96c47cfeb0ca245775c05',diag).replace('HISTORY36','CAPTURE38').replace("choices=['baseline']","choices=['baseline','d-restore']")
 receiver=receiver.replace('history36-source-source1/request/native/result.json','hc031-source-source1/request/native/result.json').replace('041ab911d8061900d937877742fee0d5577cdb60878cd21c951e25b164cc33d9','5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa').replace('07cf16434cfb7964bb1090f9c4c4d4b30823be0ef3bb3ace8c58fd4de913522c','3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c')
 receiver=receiver.replace("common=['--action','baseline'","common=['--action',a.action")
 start=receiver.index("    prepare=a.root/'prepare'");end=receiver.index("    phase='RESET'",start)
 prepare=receiver[start:end];receiver=receiver[:start]+receiver[end:]
 marker='    # Normal bypass needs zero-hit sampling latches; its actual cache0/0 gate is below.\n'
 proof="""    if a.action=='d-restore':
        proof=json.loads(Path(acks['RESET']['P']['path']).read_text())['fresh_empty_proof']
        import sys
        sys.path[:0]=[str(C),str(C/'inner-replay-tools-v2')]
        spec=importlib.util.spec_from_file_location('capture38_actual_core',tool)
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        core.fresh_check(proof,core.read(ip,identity_sha),time.time())
        save('fresh-empty-proof.json',proof)
        common+=['--fresh-proof',str(a.root/'fresh-empty-proof.json'),'--fresh-proof-sha256',digest((a.root/'fresh-empty-proof.json').read_bytes()),
                 '--native-evidence',str(C/'hc031-D-metadata-v2/native-evidence.json'),'--native-evidence-sha256','516f5a649f15ed56923b00e492273a5d2e84b123949f01f03e7fc79f27c5c918',
                 '--manager-report',str(C/'hc031-D-metadata-v2/manager-report.json'),'--manager-report-sha256','41b7f64ea44969402f61cb3f0865e39917341376a644571a8c6eb2cc882f1eaf']
"""
 receiver=receiver.replace(marker,proof+prepare)
 receiver=receiver.replace("==(0,0)","==((0,0) if a.action=='baseline' else (0,7168))")
 receiver=receiver.replace("reference=C/'state91b-baseline-native-fix91bN1/native/result.json'","reference=C/('hc031-baseline-native-hc031N3/native/result.json' if a.action=='baseline' else 'hc031-d-restore-native-hc031D2/native/result.json')")
 receiver=receiver.replace("=='7347e87250dd2ab0e93c1bcc3955d190ff11eb333ce39565b4870505915e60a3'","==('0b64ca486bb498216dfada47a974d653f562b160f8f7897fbf9a3fce3c7b0e36' if a.action=='baseline' else '506592ada04045010d963c0dda622a8da13266dc65ae245703c8175a8a30fc17')")
 receiver=receiver.replace("input_equal=native['input_ids']==old['input_ids'],","input_equal=native['input_ids']==old['input_ids'],actual_profile_equal=native['actual_native_request_profile']==old['actual_native_request_profile'],")
 receiver=receiver.replace("calibration['output_tokens']==143","calibration['actual_profile_equal'] and calibration['output_tokens']==(176 if a.action=='baseline' else 167)").replace('output_tokens=143',"output_tokens=calibration['output_tokens']")
 put('c3_hc031_capture38_receiver_v2.py',receiver)
if __name__=='__main__':main()
