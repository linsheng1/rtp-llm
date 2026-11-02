"""Only extend the measured L2->L8 gap with11 query rows, same2-phase protocol."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
POINTS={**{f'L{i}.stage.attn_input_norm5120':(5120,'torch.bfloat16') for i in range(3,8)},**{f'L{i}.stage.projectedAR5120':(5120,'torch.bfloat16') for i in range(2,8)}}
def bind(bench):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();versions={};mapping={
 'p_only_state_observer_v6.py':'p_only_state_observer_v7.py','p_only_state_observer_bootstrap_v6.py':'p_only_state_observer_bootstrap_v7.py',
 sha(bench/'p_only_state_observer_v6.py'):sha(bench/'p_only_state_observer_v7.py'),sha(bench/'p_only_state_observer_bootstrap_v6.py'):sha(bench/'p_only_state_observer_bootstrap_v7.py')}
 def put(old,new,extra=None):
  raw=(bench/old).read_text()
  for before,after in mapping.items():raw=raw.replace(before,after)
  if extra:raw=extra(raw)
  ast.parse(raw)
  for p in [bench/new,BASE/'device'/new]:
   if p.exists():assert p.read_text()==raw,'immutable stage adapter changed'
   else:p.write_text(raw)
  mapping[old[:-3]]=new[:-3];mapping[sha(bench/old)]=sha(bench/new);versions[new]=sha(bench/new)
 def model(s):
  needle="        require(case.get('additional_layers')==[0,1]";assert needle in s;s=s.replace(needle,"        require(case.get('stage_query_layers')==[2,3,4,5,6,7],'bounded L2-to-L8 stage gap required')\n"+needle);return s
 put('c3_P_state_model_service_v9.py','c3_P_state_model_service_v10.py',model)
 put('c3_P_state_runtime_probe_v9.py','c3_P_state_runtime_probe_v10.py')
 put('c3_P_state_empty_probe_v9.py','c3_P_state_empty_probe_v10.py')
 def addpoints(s):
  needle='TRANSITION_PARAMETERS=';assert needle in s;s=s.replace(needle,'TRANSITION_QUERY_POINTS.update('+repr(POINTS)+')\n'+needle);return s
 put('c3_P_state_diagnostic_client_v11.py','c3_P_state_diagnostic_client_v12.py',addpoints)
 put('c3_P_state_pilot_peer_v13.py','c3_P_state_pilot_peer_v14.py',addpoints)
 put('c3_P_state_pilot_coordinator_v12.py','c3_P_state_pilot_coordinator_v13.py')
 put('c3_P_diag_zero_request_readiness_v12.py','c3_P_diag_zero_request_readiness_v13.py')
 return versions
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--bench',type=Path,required=True);a=p.parse_args();print(json.dumps(bind(a.bench),sort_keys=True))
