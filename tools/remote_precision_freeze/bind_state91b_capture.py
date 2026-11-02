"""Version only the existing capture23 bindings for real91b + observed normal143.
No GPU/SSH, new sampling points, observer changes or protocol threshold changes.
"""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def bind(bench):
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 versions={};mapping={
  '61887b83eb180619032a46f0f2dc0eb86248ee09':'91b4635faf8c65cc67e66ab52a46f5c932cd08ea',
  'pubprec_dgdet_61887b8_Dorigin_v2':'pubprec_state91b_Dorigin_v1',
  'c3_airline4_model_service_v8':'c3_airline4_model_service_v9',
  'f87b44fa6a275c4a98bd884f669a20721ad6a62df8f8900694f920cb14276f30':sha(bench/'c3_airline4_model_service_v9.py'),
  'c3_inner_precision_replay_v5':'c3_inner_precision_replay_v6',
  'c3426ffcfa38804821d7b28f8eeab095e5fd041840b6adf54807d33f78900260':sha(bench/'c3_inner_precision_replay_v6.py'),
  'c3_actual_d_history_diagnostic_v5':'c3_actual_d_history_diagnostic_v6',
  '23430903591eec6aebc7efc8b011dcf89c34e96939708c8c8dc15dfdc19bb4e8':sha(bench/'c3_actual_d_history_diagnostic_v6.py')}
 def put(old,new,changes=None):
  raw=(bench/old).read_text()
  for before,after in dict(mapping,**(changes or {})).items():raw=raw.replace(before,after)
  ast.parse(raw)
  for path in [bench/new,BASE/'device'/new]:
   if path.exists():assert path.read_text()==raw,'immutable capture adapter drift '+new
   else:path.write_text(raw)
  mapping[old[:-3]]=new[:-3];mapping[sha(bench/old)]=sha(bench/new);versions[new]=sha(bench/new)
 put('c3_P_state_model_service_v8.py','c3_P_state_model_service_v9.py')
 put('c3_P_state_runtime_probe_v8.py','c3_P_state_runtime_probe_v9.py')
 put('c3_P_state_empty_probe_v8.py','c3_P_state_empty_probe_v9.py')
 put('c3_P_state_diagnostic_client_v10.py','c3_P_state_diagnostic_client_v11.py',{'len(outputs)==175':'len(outputs)==143','normal175':'normal143',"state['normal_output_tokens']=175":"state['normal_output_tokens']=143"})
 put('c3_P_state_pilot_peer_v12.py','c3_P_state_pilot_peer_v13.py',{"Path(v['d_model_root']).name.startswith('airline4-pub-D-')":"Path(v['d_model_root']).name=='model91b-D-source-v1'"})
 put('c3_P_state_pilot_coordinator_v11.py','c3_P_state_pilot_coordinator_v12.py')
 put('c3_P_diag_zero_request_readiness_v11.py','c3_P_diag_zero_request_readiness_v12.py')
 return versions
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--bench',type=Path,required=True);a=p.parse_args();print(json.dumps(bind(a.bench),sort_keys=True))
