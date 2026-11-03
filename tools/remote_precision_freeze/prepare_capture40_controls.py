"""Bind the evidence-only L25 Q/core/KV sampler to the current HC031 runtime."""
from pathlib import Path
import hashlib,re
HERE=Path(__file__).resolve().parent;B=HERE.parents[3]/'benchmarks/publication_precision_20261101'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def read(n):return (HERE/'device'/n).read_text()
def put(n,s):
 for p in [HERE/'device'/n,B/n]:
  if p.exists():assert p.read_text()==s,'immutable source '+str(p)
  else:p.write_text(s)
 return sha(s)
def main():
 observer=read('history40_observer_v1.py');put('history40_observer_v1.py',observer)
 boot=read('history39_bootstrap_v1.py').replace('history39_bootstrap_v1','history40_bootstrap_v1').replace('history39_observer_v1','history40_observer_v1');put('history40_bootstrap_v1.py',boot)
 helper=read('c3_hc031_capture39_model_v1.py').replace('history39','history40').replace("cfg['positions']==([7168] if role=='P' else [7169]) and cfg['layers']==list(range(40 if role=='P' else 27))","cfg['positions']==[7169] and cfg['layers']==[25]").replace("('h39-'+role)","('h40-'+role)")
 mh=put('c3_hc031_capture40_model_v1.py',helper)
 core=read('c3_hc031_capture39_core_v1.py');oldmh=re.search(r"^MODEL_HELPER='([^']+)'",core,re.M)[1];core=core.replace(oldmh,mh).replace('history39-inner-graph-latches-v1','history40-inner-graph-latches-v1').replace("cfg['positions']==([7168] if role=='P' else [7169])","cfg['positions']==[7169]").replace("cfg['layers']==list(range(40 if role=='P' else 27))","cfg['layers']==[25]")
 ch=put('c3_hc031_capture40_core_v1.py',core)
 diag=read('c3_hc031_capture39_client_v1.py').replace('c3_hc031_capture39_core_v1','c3_hc031_capture40_core_v1').replace(sha(read('c3_hc031_capture39_core_v1.py')),ch);dh=put('c3_hc031_capture40_client_v1.py',diag)
 for kind in ['receiver','sender']:
  s=read('c3_hc031_capture38_'+kind+'_v2.py').replace(sha(read('c3_hc031_capture38_core_v2.py')),ch).replace('c3_hc031_capture38_core_v2','c3_hc031_capture40_core_v1').replace('c3_hc031_capture38_client_v2','c3_hc031_capture40_client_v1').replace(sha(read('c3_hc031_capture38_client_v2.py')),dh).replace('CAPTURE38','CAPTURE40');put('c3_hc031_capture40_'+kind+'_v1.py',s)
 print('current40 controls bound')
if __name__=='__main__':main()
