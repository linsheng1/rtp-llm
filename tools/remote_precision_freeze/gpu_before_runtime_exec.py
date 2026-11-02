"""Same-PID empty-device admission before private runtime initialization.
Original controller unchanged; all subsequent initialization is inside its budget.
"""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();need(digest(a.plan)==a.sha256,'plan drift');plan=json.loads(a.plan.read_text());need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','bootstrap must start outside CUDA-visible private runtime')
    need(digest(Path(__file__))==plan['bootstrap']['SHA256'] and digest(Path(plan['script']['path']))==plan['script']['SHA256'],'bootstrap/program source drift')
    pin=plan['nvml_helper'];need(digest(Path(pin['path']))==pin['SHA256'],'original NVML audit source drift');spec=importlib.util.spec_from_file_location('original_empty_device_audit',pin['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    before=m.nvml();need(before['capability']==[10,3] and before['used_v2']==0 and not before['compute_apps'],'unchanged strict empty physicalGPU0 admission before private runtime: '+json.dumps(before))
    proc=Path('/proc/self');fields=(proc/'stat').read_text().rsplit(') ',1)[1].split();proof=dict(plan_SHA256=a.sha256,bootstrap_SHA256=plan['bootstrap']['SHA256'],PID=os.getpid(),UID=proc.stat().st_uid,start_ticks=fields[19],at=time.time(),before=before,model_requests=0)
    path=Path(plan['admission_proof']);need(path.parent==a.plan.parent and not path.exists() and not path.is_symlink(),'new bounded admission proof required')
    with path.open('x') as out:json.dump(proof,out,sort_keys=True);out.flush();os.fsync(out.fileno())
    ep=Path(plan['environment']['path']);need(digest(ep)==plan['environment']['SHA256'],'environment drift');original=json.loads(ep.read_text())['environment'];env=dict(os.environ);env.update({k:original[k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']});env.update(CUDA_VISIBLE_DEVICES='0',STATE_ADMISSION_SHA256=digest(path))
    argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',str(a.plan),'--sha256',a.sha256]
    os.execve(argv[0],argv,env)
if __name__=='__main__':main()
