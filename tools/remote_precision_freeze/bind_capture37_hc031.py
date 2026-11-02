"""Version only the proven capture34 controls for HC031 and actual current R.

Observer10/parent8/base2 and their sampling points/budgets are unchanged.
No deployment, request, scheduler or runtime mutation occurs here.
"""
import argparse,ast,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--bench',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.bench;dest=Path(__file__).parent/'device'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 pairs=[('c3_P_state_model_service_v13.py','c3_P_hc031_capture_model_v1.py'),('c3_P_state_runtime_probe_v13.py','c3_P_hc031_capture_probe_v1.py'),('c3_P_state_empty_probe_v13.py','c3_P_hc031_capture_empty_v1.py'),('c3_P_state_diagnostic_client_v15.py','c3_P_hc031_capture_client_v1.py'),('c3_P_state_pilot_peer_v24.py','c3_P_hc031_capture_peer_v1.py'),('c3_P_state_pilot_coordinator_v24.py','c3_P_hc031_capture_coordinator_v1.py'),('c3_P_diag_zero_request_readiness_v23.py','c3_P_hc031_capture_readiness_v1.py')]
 replacements={
 '91b4635faf8c65cc67e66ab52a46f5c932cd08ea':'031fc2af33111e7bbe13bbca6e72ebd6fc094d7c',
 '07cf16434cfb7964bb1090f9c4c4d4b30823be0ef3bb3ace8c58fd4de913522c':'3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c',
 '57769b6b19d18558fbc3c472838b418be032642adf40a679fe61aa4b0fc643fd':'5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa',
 '4a9f2952d45ab5d894c7af2bc870d411bf4b6b4530c3bcd8f2d560858ef61350':'516f5a649f15ed56923b00e492273a5d2e84b123949f01f03e7fc79f27c5c918',
 'a61387fce2f902f6cc34edaefd8bfcd2dc55085a447c03a3cc8a56349ac3659c':'41b7f64ea44969402f61cb3f0865e39917341376a644571a8c6eb2cc882f1eaf',
 '7347e87250dd2ab0e93c1bcc3955d190ff11eb333ce39565b4870505915e60a3':'0b64ca486bb498216dfada47a974d653f562b160f8f7897fbf9a3fce3c7b0e36',
 'pubprec_state91b_Dorigin_v1':'pubprec_hc031_Dorigin_v1',
 'state91b-source-source1':'hc031-source-source1','state91b-D-metadata-v1':'hc031-D-metadata-v2',
 'state91b-baseline-native-fix91bN1':'hc031-baseline-native-hc031N3',
 'model91b-D-reader-v6':'airline4-D-hcfix-verify4',
 'normal143':'normal176',"normal_output_tokens']=143":"normal_output_tokens']=176",'len(outputs)==143':'len(outputs)==176',
 'short113':'short167','full143':'full176'}
 for old,new in pairs:replacements[Path(old).stem]=Path(new).stem
 for old,new in [('c3_airline4_model_service_v9.py','c3_hc031_model_service_v1.py'),('c3_inner_precision_replay_v6.py','c3_hc031_inner_replay_v1.py'),('c3_actual_d_history_diagnostic_v6.py','c3_hc031_history_diagnostic_v1.py')]:
  replacements[sha(b/old)]=sha(b/new);replacements[Path(old).stem]=Path(new).stem
 outputs={}
 for old,new in pairs:
  text=(b/old).read_text()
  # Full SHA replacements are applied before module names, never nonce prefixes.
  for key,value in sorted(replacements.items(),key=lambda kv:(len(kv[0])!=64,-len(kv[0]))):text=text.replace(key,value)
  if 'coordinator' in new:
   needle='base.dependencies();envdoc=json.loads(base.CPU_ENV.read_text())'
   assert needle in text
   text=text.replace(needle,"assert sha(base.CPU_ENV)=='d5075b5c7ef68c5d1c1a4960f970c677a64d64227360c30505883abe2a39a2cc';envdoc=json.loads(base.CPU_ENV.read_text())")
  ast.parse(text)
  for path in [b/new,dest/new]:
   if path.exists():assert not path.is_symlink() and path.read_text()==text,'immutable capture37 source drift'
   else:
    with path.open('x') as f:f.write(text)
  replacements[sha(b/old)]=sha(b/new);outputs[old]=dict(path=str(dest/new),SHA256=sha(b/new))
 for name in ['p_only_state_observer_v10.py','p_only_state_observer_bootstrap_v10.py','p_only_state_observer_v8.py','p_only_state_observer_v2.py']:
  assert (b/name).is_file();outputs[name]=dict(path=str(b/name),SHA256=sha(b/name),unchanged=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(dict(state='CONTROL_SOURCE_BOUND_NOT_DEPLOYED',candidate_commit='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c',model_requests=0,outputs=outputs),f,indent=2)
 print(json.dumps(dict(state='CONTROL_SOURCE_BOUND_NOT_DEPLOYED',model_requests=0)))

if __name__=='__main__':main()
