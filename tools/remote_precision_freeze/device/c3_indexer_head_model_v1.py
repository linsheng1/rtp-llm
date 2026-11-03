"""Candidate-only adapter; original service/controller/leases remain unchanged.

The unchanged checkpoint loader permits reuse of live fully verified031 static
FDs. Proof remains bound to031 and is explicitly linked, never relabeled.
"""
import hashlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
B1='fb09988a5c552f4f5ff716472d8b50647af22d2c'
HERE=Path(__file__).resolve();BASE=HERE.with_name('c3_hc031_model_service_v1.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='b2c79ba825b69e5f3b4137e87d973211503dfb4d0ee50f38b580f4f8a0c6c7b0'
spec=importlib.util.spec_from_file_location('_original031_for_head_candidate',BASE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.__file__=str(HERE);m.B1='fb09988a5c552f4f5ff716472d8b50647af22d2c'
original_keeper=m.keeper

def keeper(a,controller,required_seconds=None):
 current=m.read(a.manifest,a.manifest_sha256);parent_pin=current['parent_indexer_candidate'];parent=m.read(Path(parent_pin['path']),parent_pin['SHA256']);oldenv_pin=parent['environment'];oldenv=m.read(Path(oldenv_pin['path']),oldenv_pin['SHA256']);target=m.read(a.env_json,a.env_sha256)
 m.require(parent['source_commit']==oldenv['source_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and current['source_commit']==target['source_commit']==m.B1,'explicit original/new candidate chain')
 allowed={'rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py','rtp_llm/models_py/modules/dsv41/fp8/_v41_indexer_head_projection.py'}
 m.require(set(current['production_Python'])-set(parent['production_Python'])=={next(x for x in allowed if x.endswith('_v41_indexer_head_projection.py'))} and set(parent['production_Python'])<=set(current['production_Python']),'exact new production member')
 for name,pin in parent['production_Python'].items():
  if name not in allowed:m.require(current['production_Python'][name]==pin,'non-head production member changed')
 for key in ['native','DG_code_members','support','nvcc_wrapper','patch_kernel']:m.require(current[key]==parent[key],'checkpoint lease cannot bridge changed native/support runtime')
 old=SimpleNamespace(**dict(vars(a),env_json=Path(oldenv_pin['path']),env_sha256=oldenv_pin['SHA256'],manifest=Path(parent_pin['path']),manifest_sha256=parent_pin['SHA256']))
 proof=original_keeper(old,controller,required_seconds)
 loader='rtp_llm/models_py/modules/dsv41/engram.py';m.require((Path(target['site'])/loader).resolve()==(Path(oldenv['site'])/loader).resolve() and m.pin(Path(target['site'])/loader)==current['production_Python'][loader]==parent['production_Python'][loader],'checkpoint loader actual identity changed')
 return dict(proof,target_candidate_commit=m.B1,target_manifest_SHA256=a.manifest_sha256,original_full_content_lease_candidate=parent['source_commit'],original_full_content_lease_manifest=parent_pin,static_only_no_KV_reuse=True)
m.keeper=keeper
globals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})
if __name__=='__main__':raise SystemExit(main())
