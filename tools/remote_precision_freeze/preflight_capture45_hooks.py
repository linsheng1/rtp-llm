"""Actual CUDA-hidden startup-hook import against the frozen D epoch/config.
No model construction, RPC, dynamic artifact replacement, or new scheduler.
"""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--model-plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m,mp=f.doc(a.model_plan);f.need(mp['SHA256']==a.sha256 and m['role']=='D','actual D epoch required')
 for pin in m['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'epoch source drift')
 cfg,cp=f.doc(m['local_pins']['config']['path']);sys.path.insert(0,m['bench']);from remote_control import remote
 code='''import json,os,hashlib,subprocess
from pathlib import Path
c=Path(%r);ep=c/'hc-tf32-031-inner-v1/environment.json';assert hashlib.sha256(ep.read_bytes()).hexdigest()=='1bb3d2ba9082c9ce17c2990ff9a8448855d68c88d793d89b8d25538d8355f6b2';e=json.loads(ep.read_text());env=dict(e['environment']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');env['PYTHONPATH']=str(c)+':'+env['PYTHONPATH']
p=c/'airline4-D-capture45/plan.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
program=%r
q=subprocess.run([e['interpreter'],'-B','-S','-c',program],env=dict(os.environ,**env),capture_output=True,text=True,timeout=90)
out=dict(actual_child_returncode=q.returncode,model_requests=0,stderr_tail=q.stderr[-2500:])
if q.returncode==0:out['actual']=json.loads(q.stdout.strip().splitlines()[-1])
else:out['stdout_tail']=q.stdout[-1500:]
print(json.dumps(out))
'''%(f.CONTROL,m['model_plan_SHA256'],"import json,hashlib,importlib;from pathlib import Path;c=Path("+repr(f.CONTROL)+");p=c/'capture45-D-config.json';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=="+repr(cp['SHA256'])+";cfg=json.loads(raw);import history45_observer_v1 as o;o.install(cfg);[importlib.import_module(n) for n in o._base.TARGETS];import torch;assert not torch.cuda.is_initialized();assert set(cfg['indexer_extension']['source_sha256'])<=set(o._capture._source_identity);print(json.dumps(dict(state='ACTUAL_STARTUP_HOOK_IMPORT_PASS_NO_CUDA_NO_MODEL',hooks=len(o._base.TARGETS),model_requests=0)))")
 out=remote('cluster3',f.HOSTS['D'],code,120);f.save_new(a.output,dict(model_epoch_plan=mp,entry=f.read(__file__)[1],result=out));f.need(out['actual_child_returncode']==0,json.dumps(out));print(json.dumps(out))
if __name__=='__main__':main()
