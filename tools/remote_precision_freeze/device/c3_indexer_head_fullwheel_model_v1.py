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
WHEEL='6a77459e273388b3e6143b9b2c45d16e083cc6c4d658d382438e5ed5abfb4645'
original_keeper=m.keeper

def keeper(a,controller,required_seconds=None):
 current=m.read(a.manifest,a.manifest_sha256);innerpin=current['parent_inner_manifest'];inner=m.read(Path(innerpin['path']),innerpin['SHA256']);parentpin=inner['parent_indexer_candidate'];parent=m.read(Path(parentpin['path']),parentpin['SHA256']);ep=parent['environment'];oldenv=m.read(Path(ep['path']),ep['SHA256']);env=m.read(a.env_json,a.env_sha256)
 m.require(current['wheel']['SHA256']==WHEEL and current['source_commit']==inner['source_commit']==m.B1 and parent['source_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c','full wheel and original static proof chain')
 for n,pin in current['native'].items():m.require({k:pin[k] for k in ['bytes','SHA256']}=={k:parent['native'][n][k] for k in ['bytes','SHA256']},'checkpoint lease native bytes changed')
 for key in ['DG_code_members','nvcc_wrapper','patch_kernel']:m.require(current[key]==parent[key],'checkpoint dependency changed')
 loader='rtp_llm/models_py/modules/dsv41/engram.py';m.require(m.pin(Path(env['site'])/loader)==m.pin(Path(oldenv['site'])/loader)=={k:parent['production_Python'][loader][k] for k in ['bytes','SHA256']},'checkpoint loader bytes changed')
 old=SimpleNamespace(**dict(vars(a),env_json=Path(ep['path']),env_sha256=ep['SHA256'],manifest=Path(parentpin['path']),manifest_sha256=parentpin['SHA256']))
 proof=original_keeper(old,controller,required_seconds)
 return dict(proof,target_candidate_commit=m.B1,target_manifest_SHA256=a.manifest_sha256,immutable_wheel_SHA256=WHEEL,original_full_content_lease_manifest=parentpin,static_only_no_KV_reuse=True)
m.__dict__['WHEEL']=WHEEL
exec('def target(a):\n    e=read(a.env_json,a.env_sha256);m=read(a.manifest,a.manifest_sha256)\n    require(e[\'schema\']==\'c3-dg-batch-invariant-ad62-target-inner-env-v1\' and m[\'state\']==\'FULL_WHEEL_INSTALLED_CPU_PENDING\'\n            and m[\'source_commit\']==B1 and m[\'inner_only\'] is False and m[\'immutable_wheel\'] is True and m[\'wheel\'][\'SHA256\']==WHEEL and m[\'full_341_candidate_accepted\'] is False\n            and m[\'Cpp_diff_empty\'] is True and m[\'base_native_source\']==\'c1b4b19d82f56507f89be86dddf90497501d32a6\'\n            and m[\'host\']==a.host,\'per-node target b1/ad62 inner contract differs\')\n    require(m[\'environment\'][\'path\']==str(a.env_json) and m[\'environment\'][\'SHA256\']==a.env_sha256,\'manifest/env binding differs\')\n    require(m[\'DeepGEMM_batch_invariant\'] is True and (Path(e[\'site\'])/\'rtp_llm/models_py/model_desc/deepseek_v41_base_model.py\').read_text().count(\'deep_gemm.use_deterministic_algorithms(True)\')==1,\'frozen batch-invariant initialization required\')\n    site=Path(e[\'site\']);require(site==a.manifest.parent/\'site\' and e[\'environment\'][\'PYTHONPATH\'].split(\':\')[0]==str(site),\'new target site must be first\')\n    require(all(k not in e[\'environment\'] or isinstance(e[\'environment\'][k],str) for k in NUMERIC),\'present DSV41 target flags must be strings\')\n    require(\'c3-runtime-ff13-remote/site\' not in e[\'environment\'][\'PYTHONPATH\'],\'old ff13 site cannot reenter PYTHONPATH\')\n    for name,value in m[\'production_Python\'].items():exact_pin(site/name,value)\n    # b467 target has none of these six flags. Preserve the actual normal\n    # os.environ.get defaults, after checking them in the fully pinned source.\n    for key,(default,relative) in NUMERIC_DEFAULTS.items():\n        marker=\'os.environ.get("\'+key+\'", "\'+default+\'")\'\n        require(relative in m[\'production_Python\'] and marker in (site/relative).read_text(),\n                \'pinned normal source default differs: \'+key)\n    for name,value in m[\'support\'].items():exact_pin(site/name,value)\n    for name,value in m[\'native\'].items():exact_pin(site/name,value)\n    dg=Path(e[\'DG_root\'])/\'deep_gemm\'\n    for name,value in m[\'DG_code_members\'].items():exact_pin(dg/name,value)\n    for key in [\'nvcc_wrapper\',\'patch_kernel\']:exact_pin(Path(m[key][\'path\']),m[key])\n    require(e[\'environment\'][\'CUDA_HOME\']==str(Path(m[\'nvcc_wrapper\'][\'path\']).parent.parent)\n            and e[\'environment\'][\'PATH\'].split(\':\')[0]==str(Path(m[\'nvcc_wrapper\'][\'path\']).parent),\'matching target CUDA wrapper contract differs\')\n    return e,m',m.__dict__)
m.keeper=keeper
original_configure=m.configure
def configure(a,e,profile):
 plan=original_configure(a,e,profile);plan.update(immutable_wheel_SHA256=WHEEL,classification='complete immutable C3 wheel; diagnostic model gate pending');return plan
m.configure=configure
globals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})
if __name__=='__main__':raise SystemExit(main())
