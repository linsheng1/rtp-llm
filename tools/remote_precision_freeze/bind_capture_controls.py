"""One-time minimal versioned adapter for the current capture protocol.
Never overwrites old published helpers. Does not touch runtime or send requests.
"""
import argparse,hashlib,ast
from pathlib import Path

def bind(bench):
    def sha(name):return hashlib.sha256((bench/name).read_bytes()).hexdigest()
    def put(name,text):
        ast.parse(text);p=bench/name
        if p.exists():assert p.read_text()==text,'immutable helper differs: '+name
        else:p.write_text(text)
        return sha(name)
    s=(bench/'c3_P_state_runtime_probe_v7.py').read_text().replace('08b3feedea1d66143f2c876a415b0b99f5cd598f45310a644660f4e91b775e94',sha('c3_inner_precision_replay_v5.py'))
    probe=put('c3_P_state_runtime_probe_v8.py',s)
    s=(bench/'c3_P_state_empty_probe_v7.py').read_text().replace('c3_P_state_runtime_probe_v7','c3_P_state_runtime_probe_v8')
    empty=put('c3_P_state_empty_probe_v8.py',s)
    s=(bench/'c3_P_state_diagnostic_client_v9.py').read_text().replace('c3_P_state_runtime_probe_v7','c3_P_state_runtime_probe_v8').replace(sha('c3_P_state_runtime_probe_v7.py'),probe)
    s=s.replace('08b3feedea1d66143f2c876a415b0b99f5cd598f45310a644660f4e91b775e94',sha('c3_inner_precision_replay_v5.py')).replace('6875abc09d4bedf22985d07ce4b763bfb71af23c5cc6e8ced6a57c4c2637ac24',sha('c3_actual_d_history_diagnostic_v5.py'))
    s=s.replace('full_output_ids_equal_normal145','full_output_ids_equal_normal175').replace("state['normal_output_tokens']=145","state['normal_output_tokens']=175")
    client=put('c3_P_state_diagnostic_client_v10.py',s)
    def update(s):
        return s.replace('c3_P_state_runtime_probe_v7','c3_P_state_runtime_probe_v8').replace('c3_P_state_empty_probe_v7','c3_P_state_empty_probe_v8').replace(sha('c3_P_state_runtime_probe_v7.py'),probe).replace(sha('c3_P_state_empty_probe_v7.py'),empty)
    s=update((bench/'c3_P_state_pilot_peer_v11.py').read_text())
    peer=put('c3_P_state_pilot_peer_v12.py',s)
    s=update((bench/'c3_P_state_pilot_coordinator_v10.py').read_text()).replace('c3_P_state_pilot_peer_v11','c3_P_state_pilot_peer_v12')
    s=s.replace("def identity(token,include_startup=False):\n  sources=phase('PROBE',['P','D'],token,include_startup=include_startup);files={}","def identity(token,include_startup=False,sources=None):\n  if sources is None:sources=phase('PROBE',['P','D'],token,include_startup=include_startup)\n  files={}")
    old="  proof=phase('EMPTY_SHORT',['P'],'short-empty',145)['P'];"
    assert old in s;s=s.replace(old,"  _,shortsources=identity('short-probes') # both artifact probes finish before fresh empty\n"+old)
    s=s.replace("  ip,_=identity('short-probes');e=json.loads(fp.read_text());", "  ip,_=identity('short-identity',sources=shortsources);e=json.loads(fp.read_text());")
    put('c3_P_state_pilot_coordinator_v11.py',s)
    put('c3_P_diag_zero_request_readiness_v11.py',update((bench/'c3_P_diag_zero_request_readiness_v10.py').read_text()).replace('c3_P_state_pilot_peer_v11','c3_P_state_pilot_peer_v12'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bench',type=Path,required=True);bind(p.parse_args().bench)
