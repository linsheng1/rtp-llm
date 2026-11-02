"""Run the committed compressor module regressions, under original admission guards."""
import argparse,hashlib,importlib.util,json,os,time,unittest
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def read(pin):
 p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and digest(p)==pin['SHA256'],'pin drift: '+str(p));return json.loads(p.read_text())
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();plan=read(dict(path=str(a.plan),SHA256=a.sha256));env=read(plan['environment']);manifest=read(plan['manifest']);need(manifest['source_commit']==env['source_commit']==plan['candidate_commit'],'candidate drift')
 for key in ['script','test','bootstrap','nvml_helper']:need(digest(Path(plan[key]['path']))==plan[key]['SHA256'],'source drift '+key)
 site=Path(env['site']);rel='rtp_llm/models_py/modules/dsv41/fp8/compressor.py';need(digest(site/rel)==manifest['production_Python'][rel]['SHA256']==plan['compressor_SHA256'],'committed module drift')
 if a.check:
  need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU gate requires CUDA hidden');print(json.dumps(dict(state='CANDIDATE_UNIT_CPU_PINS_PASS',model_requests=0)));return
 root=Path(plan['output_root']);need(os.getuid()==500 and not root.exists(),'new owned result required');root.mkdir(mode=0o700)
 def save(name,v):
  t=root/(name+'.partial');t.write_text(json.dumps(v,sort_keys=True,indent=2));t.replace(root/name)
 try:
  proof=read(dict(path=plan['admission_proof'],SHA256=os.environ['STATE_ADMISSION_SHA256']));fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();need(proof['PID']==os.getpid() and proof['UID']==500 and proof['start_ticks']==fields[19] and proof['plan_SHA256']==a.sha256 and -2<=time.time()-proof['at']<=45,'same-PID admission differs')
  audit=module(plan['nvml_helper']['path'],'original_NVML_guard');before=proof['before'];need(before['used_v2']==0 and not before['compute_apps'],'original empty GPU guard')
  import torch,deep_gemm
  from rtp_llm.models_py.modules.dsv41.fp8 import compressor
  need(Path(compressor.__file__).resolve()==(site/rel).resolve(),'actual candidate import differs');dg=Path(deep_gemm.__file__).resolve().parent;need(dg==(Path(env['DG_root'])/'deep_gemm').resolve(),'DG import differs')
  need(digest(dg/'_C.cpython-310-aarch64-linux-gnu.so')=='b0233518517afc95e17a998ae1c8bae7158fab60fba24aa7eeb5cd77d571c99e','DG native drift')
  torch.cuda.set_device(0);torch.cuda.init();torch.cuda.synchronize();initial=audit.nvml();limit=1024**3;remaining=limit-initial['used_v2'];need(remaining>128*1024**2 and set(initial['compute_apps'])<={os.getpid()},'original exclusive1GiB guard');torch.cuda.set_per_process_memory_fraction(remaining/initial['total_v2'],0);torch.cuda.reset_peak_memory_stats()
  tests=module(plan['test']['path'],'committed_compressor_regression');suite=unittest.defaultTestLoader.loadTestsFromModule(tests);need(suite.countTestCases()==2,'exact2 committed GPU regressions required');r=unittest.TextTestRunner(verbosity=2).run(suite);torch.cuda.synchronize();torch.cuda.empty_cache();after=audit.nvml();result=dict(state='CANDIDATE_MODULE_REGRESSION_COMPLETE',candidate_commit=plan['candidate_commit'],module_path=compressor.__file__,module_SHA256=digest(site/rel),tests_run=r.testsRun,skipped=len(r.skipped),failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful() and r.testsRun==2 and not r.skipped,GPU_before=before,GPU_after=after,torch_peak=torch.cuda.max_memory_allocated(),model_requests=0,whole_model_fixed=False,formal_acceptance=False)
  save('result-safe.json',result);need(set(after['compute_apps'])<={os.getpid()} and after['used_v2']<=limit and result['torch_peak']<=remaining,'unchanged1GiB guard');need(result['passed'],'actual module regression failed');save('status.json',dict(state='SUCCEEDED',result=dict(path=str(root/'result-safe.json'),SHA256=digest(root/'result-safe.json')),model_requests=0));print(json.dumps(dict(passed=True,tests_run=2,skipped=0,model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
if __name__=='__main__':main()
