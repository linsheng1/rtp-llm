"""Existing CPU compose/check -> original after-drain wrapper/controller.
Only for a new instance at an already finished or natural lease boundary.
Not an experiment driver; the subsequent requests require a new frozen plan.
"""
import argparse,hashlib,json,secrets,sys,time
from pathlib import Path
from freeze_stage import need,save_new,read,CONTROL,HOSTS,option

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ['p-reference','d-reference','observer-template','bench','output-dir','tag']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--seconds',type=int,default=1200)
    a=p.parse_args();need(600<=a.seconds<=1800,'original diagnostic lifetime bound');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    need(__import__('re').fullmatch('[A-Za-z0-9_-]{1,40}',a.tag),'invalid tag')
    bench=Path(a.bench);sys.path.insert(0,str(bench))
    from remote_control import remote
    from cluster23_dispatch import submit
    refs={r:json.loads(Path(getattr(a,r.lower()+'_reference')).read_text()) for r in ['P','D']}
    template=json.loads(Path(a.observer_template).read_text())
    oldroot=option(refs['P']['argv'],'--root');newp=CONTROL+'/airline4-P-dgdet-'+a.tag
    # This is a fresh observer plan; no old nonce/identity may be silently reused.
    raw=json.dumps(template).replace(oldroot,newp)
    template=json.loads(raw)
    nonce_file=out/'nonces.json'
    nonces=json.loads(nonce_file.read_text()) if nonce_file.exists() else {}
    for index,case in enumerate(template['cases']):
        key='short_nonce' if index==0 else 'full_nonce';nonce=nonces.get(key) or secrets.token_hex(24)
        old=case['nonce'];case=json.loads(json.dumps(case).replace(old,nonce));template['cases'][index]=case;nonces[key]=nonce
    template_path=CONTROL+'/observer-template-'+a.tag+'.json'
    template_raw=json.dumps(template,sort_keys=True,indent=2)+'\n';template_sha=hashlib.sha256(template_raw.encode()).hexdigest()
    if not nonce_file.exists():save_new(nonce_file,nonces)
    code="""import os,json,hashlib
from pathlib import Path
p=Path(%r);data=%r;assert os.getuid()==500
if p.exists():assert not p.is_symlink() and p.read_text()==data,'existing template identity differs'
else:
 with p.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest())))"""%(template_path,template_raw)
    need(remote('cluster3',HOSTS['P'],code)['SHA256']==template_sha,'template publish differs')
    for role in ['D','P']:
        old=refs[role];argv=old['argv'][:]
        if '--mode' in argv:argv=argv[:argv.index('--mode')]
        root=newp if role=='P' else CONTROL+'/airline4-pub-D-'+a.tag
        previous=option(argv,'--root')
        argv[argv.index('--root')+1]=root;argv[argv.index('--seconds')+1]=str(a.seconds)
        if role=='P':
            argv[argv.index('--observer-template')+1]=template_path
            argv[argv.index('--observer-template-sha256')+1]=template_sha
        code="""import os,json,subprocess,hashlib,time
from pathlib import Path
c=Path(%r);a=%r;checks=[]
previous=Path(%r);f=json.loads((previous/'model-service-final.json').read_text());s=json.loads((previous/'model-status.json').read_text());assert f['state']=='MODEL_SERVICE_EXITED' and f['actual_model_root_returncode']==0 and s['cleanup_state']=='FINISHED' and s['remaining_owned_processes']==[],'natural phase boundary not completed'
for flag,state in [('--consumer-root','READY'),('--keeper-root','RUNNING')]:
 v=json.loads((Path(a[a.index(flag)+1])/'status.json').read_text());assert v['state']==state and time.time()-v['heartbeat_at']<=45 and v['deadline_epoch']>time.time()+int(a[a.index('--seconds')+1])+15,'parent original lease insufficient'
for mode in ['compose','check']:
 q=subprocess.run(a+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180);checks.append(dict(mode=mode,rc=q.returncode,stdout=q.stdout[-1000:],stderr=q.stderr[-2500:]));assert q.returncode==0,checks
r=Path(a[a.index('--root')+1]);out=dict(checks=checks,actual_RPC=0,plan_SHA256=hashlib.sha256((r/'plan.json').read_bytes()).hexdigest())
f=r/'observer-config-manifest.json'
if f.exists():out['observer-config-manifest.json']=json.loads(f.read_text())
print(json.dumps(out))"""%(CONTROL,argv,previous)
        result=remote('cluster3',HOSTS[role],code,240);result['argv']=argv
        save_new(out/(role+'-CPU.json'),result)
        dispatch=dict(role=role,host=HOSTS[role],root=Path(root).name,plan_SHA256=result['plan_SHA256'],argv=argv+['--mode','run'])
        if a.execute:
            wrapper=['/opt/conda310/bin/python3.10','-B','-S',CONTROL+'/c3_after_reference_drain_v1.py',
                     '--previous-root',previous,'--state-root',CONTROL+'/root-'+a.tag+'-'+role+'-after-drain','--']+dispatch['argv']
            # Only waits for natural successful original lease drain. No stop/restart signal.
            dispatch['launch']=submit('cluster3',HOSTS[role],'root-'+a.tag+'-'+role+'-model',wrapper,2700,
                   [dict(path=Path(root).name+'/model-ready.json',json_fields=dict(state='MODEL_READY_KV_EMPTY_PENDING',model_READY=True))])
        save_new(out/(role+'-dispatch.json'),dispatch)
        print(json.dumps(dict(role=role,CPU='PASS',plan_SHA256=result['plan_SHA256'],submitted=a.execute)),flush=True)
if __name__=='__main__':main()
