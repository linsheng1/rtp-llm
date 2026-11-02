"""Version the existing capture helper DAG for the bounded L2 add-on."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def bind(b):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();maps={};out={}
 for old,new in [('p_only_state_observer_v8.py','p_only_state_observer_v9.py'),('p_only_state_observer_bootstrap_v8.py','p_only_state_observer_bootstrap_v9.py')]:maps[old]=new;maps[sha(b/old)]=sha(b/new)
 for old,new in [('c3_P_state_model_service_v11.py','c3_P_state_model_service_v12.py'),('c3_P_state_runtime_probe_v11.py','c3_P_state_runtime_probe_v12.py'),('c3_P_state_empty_probe_v11.py','c3_P_state_empty_probe_v12.py'),('c3_P_state_diagnostic_client_v13.py','c3_P_state_diagnostic_client_v14.py'),('c3_P_state_pilot_peer_v18.py','c3_P_state_pilot_peer_v19.py'),('c3_P_state_pilot_coordinator_v18.py','c3_P_state_pilot_coordinator_v19.py'),('c3_P_diag_zero_request_readiness_v17.py','c3_P_diag_zero_request_readiness_v18.py')]:
  s=(b/old).read_text()
  for a,z in maps.items():s=s.replace(a,z)
  if old=='c3_P_state_model_service_v11.py':
   marker="        require(case.get('hook_source_sha256')==hook_expected,'six actual manifest hook source SHA pins required')"
   assert marker in s
   s=s.replace(marker,marker+"\n        require(case.get('l2_attention_diagnostic')==dict(schema='c3-l2-attention-readonly-v1',layer=2,query_position=7168,max_selected_rows=640,max_heads=64,original_extra_limit=4*1024**2,natural_one_chunk_only=True),'explicit bounded attention profile required')\n        keys=['rtp_llm/models_py/modules/dsv41/fp8/attention.py','rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py','rtp_llm/models_py/modules/dsv41/fp8/_v41_mega_attention.py'];require(case.get('l2_attention_production_sha256')=={k:manifest['production_Python'][k]['SHA256'] for k in keys},'actual attention method source map')\n        parent=C/'p_only_state_observer_v8.py';require(pin(parent)['SHA256']=='9d986d662754a43681f2a7e27ce35fe9b043d8afecdfa7bc93daade266202341','exact immutable parent8')")
  if old=='c3_P_state_pilot_peer_v18.py':
   before="MoE_supplemental=supplement,raw_tensors_exported=False";assert before in s;s=s.replace(before,"MoE_supplemental=supplement,attention=proof['attention'],raw_tensors_exported=False")
  if old=='c3_P_state_diagnostic_client_v13.py':
   before="    points=transition['points'];";assert before in s
   s=s.replace(before,"    consumer=transition['attention'];need(consumer['schema']=='c3-l2-attention-readonly-v1' and consumer['coverage_complete'] is True and consumer['request_id']==short['request_id'] and consumer['nonce']==barrier['nonce'] and consumer['model_identity_sha256']==barrier['model_identity_sha256'] and len(consumer['ranks'])==4,'actual L2 attention input/core/AR proof required before full request')\n"+before)
  ast.parse(s)
  for p in [b/new,BASE/'device'/new]:
   if p.exists():assert p.read_text()==s,'immutable helper differs'
   else:p.write_text(s)
  maps[old[:-3]]=new[:-3];maps[old]=new;maps[sha(b/old)]=sha(b/new);out[new]=sha(b/new)
 return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--bench',type=Path,required=True);a=p.parse_args();print(json.dumps(bind(a.bench),sort_keys=True))
