"""Rebind unchanged bounded stage capture after the exact completion-set fix."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def bind(bench):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mapping={'p_only_state_observer_v7.py':'p_only_state_observer_v8.py','p_only_state_observer_bootstrap_v7.py':'p_only_state_observer_bootstrap_v8.py',sha(bench/'p_only_state_observer_v7.py'):sha(bench/'p_only_state_observer_v8.py'),sha(bench/'p_only_state_observer_bootstrap_v7.py'):sha(bench/'p_only_state_observer_bootstrap_v8.py')};out={}
 for old,new in [('c3_P_state_model_service_v10.py','c3_P_state_model_service_v11.py'),('c3_P_state_runtime_probe_v10.py','c3_P_state_runtime_probe_v11.py'),('c3_P_state_empty_probe_v10.py','c3_P_state_empty_probe_v11.py'),('c3_P_state_diagnostic_client_v12.py','c3_P_state_diagnostic_client_v13.py'),('c3_P_state_pilot_peer_v14.py','c3_P_state_pilot_peer_v15.py'),('c3_P_state_pilot_coordinator_v13.py','c3_P_state_pilot_coordinator_v14.py'),('c3_P_diag_zero_request_readiness_v13.py','c3_P_diag_zero_request_readiness_v14.py')]:
  s=(bench/old).read_text()
  for before,after in mapping.items():s=s.replace(before,after)
  ast.parse(s)
  for p in [bench/new,BASE/'device'/new]:
   if p.exists():assert p.read_text()==s,'immutable completion adapter changed'
   else:p.write_text(s)
  mapping[old[:-3]]=new[:-3];mapping[sha(bench/old)]=sha(bench/new);out[new]=sha(bench/new)
 return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--bench',type=Path,required=True);a=p.parse_args();print(json.dumps(bind(a.bench),sort_keys=True))
