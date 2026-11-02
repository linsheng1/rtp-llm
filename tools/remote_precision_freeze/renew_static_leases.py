"""Preserve original live static FD leases via existing finite helper/controller.
CPU/control preparation only; zero weight bytes, GPU, KV or model requests.
"""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','submit']);p.add_argument('--bench',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
    sys.path.insert(0,str(a.bench));from remote_control import remote
    from cluster23_dispatch import submit
    name='renew_c3_static_engram_leases_v9.py';helper=f.read(a.bench/name)[1]
    if a.command=='freeze':
        jobs={};sources={}
        for role in ['P','D']:
            host=f.HOSTS[role]
            code='''import json,hashlib,time\nfrom pathlib import Path\nr=Path(%r)/'static-engram-next-v14';v=json.loads((r/'status.json').read_text());assert v['state']=='RUNNING' and time.time()-v['heartbeat_at']<45 and v['deadline_epoch']>time.time()+15\nprint(json.dumps(dict(path=str(r/'ready.json'),SHA256=hashlib.sha256((r/'ready.json').read_bytes()).hexdigest(),process=v['process'],deadline_epoch=v['deadline_epoch'])))'''%f.CONTROL
            source=remote('cluster3',host,code,60);sources[role]=source
            jobs[role]=dict(host=host,name='static-engram-next-v15-controller',timeout=3780,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/'+name,'--role',role,'--host',host,'--root',f.CONTROL+'/static-engram-next-v15','--from-root',f.CONTROL+'/static-engram-next-v14','--from-ready-sha256',source['SHA256'],'--seconds','3600'],verify=[dict(path='static-engram-next-v15/ready.json',json_fields=dict(state='STATIC_ENGRAM_MODEL_WEIGHT_LEASES_HELD',weight_bytes_read=0,CUDA_initialized=False,KV_cache_held=False))])
        plan=dict(schema='existing-static-FD-renewal-plan-v1',model_requests=0,weight_bytes=0,jobs=jobs,sources=sources,helper=helper,entry=f.read(__file__)[1]);print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256 and helper==plan['helper'] and f.read(__file__)[1]==plan['entry'],'renewal plan/code differs');results={}
    for role,job in plan['jobs'].items():results[role]=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],assets={name:Path(helper['path']).read_text()})
    f.need(a.receipt is not None,'receipt required');f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launches=results,model_requests=0));print(json.dumps(dict(state='SUBMITTED_NOT_READY',model_requests=0)))
if __name__=='__main__':main()
