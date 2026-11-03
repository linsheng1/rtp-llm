"""Thin profile adapter over the unchanged, calibrated capture44 service."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve();BASE=HERE.with_name('c3_hc031_capture44_model_v1.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='5c6243e15b64fac9d1af5214e8fa660291463924bed92773609ad5e1e3af7225'
spec=importlib.util.spec_from_file_location('_calibrated_model44_for45',BASE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.__file__=str(HERE)
def contract(path,sha,role,host,env_sha,manifest_sha):
 m.require(path is not None and path.parent==m.C and sha is not None,'explicit owned history45 config')
 cfg=m.read(path,sha)
 m.require(cfg['schema']=='history44-inner-graph-latches-v1' and cfg['source_commit']==m.B1 and cfg['role']==role and cfg['host']==host and cfg['environment_SHA256']==env_sha and cfg['runtime_manifest_SHA256']==manifest_sha and cfg['wheel_sha256'] is None and cfg['artifact_kind']=='inner_python_overlay','actual base44 candidate/runtime identity')
 m.require(cfg['positions']==[7045] and cfg['layers']==[24] and cfg['gpu_budget_bytes']==4194304 and Path(cfg['output']).parent==m.C and Path(cfg['output']).name=='h45-'+role,'same bounded7045/L24 extension')
 for name in ['history45_observer_v1.py','history45_bootstrap_v1.py','history44_observer_v1.py','history38_latches_v1.py']:
  m.require(m.pin(m.C/name)['SHA256']==cfg['control_source_SHA256'][name],'actual45 control source drift')
 m.require(cfg['indexer_extension']['revision']==45 and cfg['indexer_extension']['causal_prefix']==7046 and cfg['indexer_extension']['base44_SHA256']==cfg['control_source_SHA256']['history44_observer_v1.py'],'bounded base44 extension binding')
 runtime=m.read(Path(cfg['manifest_path']),manifest_sha)
 for name,sha in dict(cfg['hook_source_sha256'],**cfg['indexer_extension']['source_sha256']).items():m.require(runtime['production_Python'][name.replace('.','/')+'.py']['SHA256']==sha,'actual production source drift '+name)
 return cfg
m.history_contract=contract;original=m.configure
def configure(*a,**kw):
 v=original(*a,**kw);m.require(v['argv'][2]==str(m.C/'history44_bootstrap_v1.py'),'calibrated startup shape');v['argv'][2]=str(m.C/'history45_bootstrap_v1.py')
 for key,name in [('observer','history45_observer_v1.py'),('bootstrap','history45_bootstrap_v1.py')]:v['history_capture'][key]=dict(path=str(m.C/name),**m.pin(m.C/name))
 v['history_capture']['base44_service']=dict(path=str(BASE),**m.pin(BASE));v['classification']='history45 L24 indexer input/score/selection extension; calibrated base44 hooks unchanged; inner only';return v
m.configure=configure
globals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})
if __name__=='__main__':raise SystemExit(main())
