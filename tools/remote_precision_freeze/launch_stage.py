"""Frozen argv -> existing submit/controller. No polling or model lifecycle changes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import freeze_stage as freeze


def command(*args):
    p = subprocess.run([sys.executable,'-B',str(freeze.BASE/'freeze_stage.py'),*args],
                       stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=40)
    freeze.need(p.returncode == 0, p.stdout + p.stderr)
    return json.loads(p.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--device-prepare',action='store_true')
    p.add_argument('--readiness',action='store_true')
    p.add_argument('--submit',action='store_true')
    a=p.parse_args()
    freeze.need(not (a.readiness and a.submit), 'readiness and request submission are separate finite phases')
    command('check','--plan',str(a.plan),'--sha256',a.sha256,'--cli-help')
    plan,_=freeze.doc(a.plan)
    freeze.need(plan.get('flow') == 'capture','this adapter executes only the existing capture protocol')
    # Render is the only source of downstream argv. No dispatch reads here.
    jobs={r:command('render','--plan',str(a.plan),'--sha256',a.sha256,'--role',r) for r in ['receiver','P','D']}
    a.evidence_dir.mkdir(parents=True,exist_ok=True)
    def save(name,value):
        target=a.evidence_dir/name
        freeze.save_new(target,dict(plan_SHA256=a.sha256,at=time.time(),**value))
    save('rendered-jobs.json',dict(jobs=jobs,model_requests=0))
    if not (a.device_prepare or a.readiness or a.submit):return
    recipe,_=freeze.doc(plan['local_pins']['recipe']['path'])
    bench=Path(recipe['bench']);sys.path.insert(0,str(bench))
    import remote_control,cluster23_dispatch
    # Pin transport and existing controller implementation too.
    for name in ['remote_control','cluster23_dispatch']:
        freeze.need(freeze.read(bench/(name+'.py'))[1]==plan['local_pins'][name], 'transport drift: '+name)
    lock=Path('/tmp/decode-remote-ssh-501/dsh-task-remote.lock');lock.parent.mkdir(mode=0o700,exist_ok=True)
    def remote(host,code,timeout=90):
        with lock.open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            return remote_control.remote('cluster3',host,code,timeout)
    # Only control source and safe metadata are published. Existing identities must match.
    publish={freeze.CONTROL+'/'+recipe['helpers'][k]:plan['local_pins'][k]
             for k in recipe['helpers']}
    publish.update(plan['device_files'])
    for role,host in [('receiver',freeze.HOSTS['receiver']),('P',freeze.HOSTS['P']),('D',freeze.HOSTS['D'])]:
        expected={target:pin['SHA256'] for target,pin in publish.items()}
        assets={freeze.CONTROL+'/'+recipe['helpers'][k]:Path(plan['local_pins'][k]['path']).read_text()
                for k in recipe['publish_helpers']}
        assets.update({target:Path(pin['path']).read_text() for target,pin in plan['device_files'].items()})
        # Verify existing bytes first; send only missing small SOURCE/control files.
        probe='''import os,json,hashlib
from pathlib import Path
missing=[]
for name,wanted in %r.items():
 p=Path(name)
 if p.exists():
  assert p.is_file() and not p.is_symlink() and p.stat().st_uid==500,'nonowned control file: '+name
  assert hashlib.sha256(p.read_bytes()).hexdigest()==wanted,'immutable device control drift: '+name
 else:missing.append(name)
print(json.dumps(missing))'''%expected
        missing=remote(host,probe)
        freeze.need(all(name in assets for name in missing),'missing frozen dependency has no source asset')
        assets={name:assets[name] for name in missing}
        # Bounded SOURCE-only chunks, never model/business data. No overwrite on drift.
        chunks=[{}]
        for target,raw in assets.items():
            if len(repr(chunks[-1]))+len(repr(raw))+len(target)>70000:chunks.append({})
            chunks[-1][target]=raw
        for chunk in chunks:
            if not chunk:continue
            code='''import os,json,hashlib
from pathlib import Path
expected=%r;assets=%r;assert os.getuid()==500
for name,raw in assets.items():
 p=Path(name);assert hashlib.sha256(raw.encode()).hexdigest()==expected[name]
 if p.exists():assert not p.is_symlink() and p.stat().st_uid==500 and p.read_text()==raw,'immutable device control differs: '+str(p)
 else:
  with p.open('x') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(control_files=len(assets))))'''%(expected,chunk)
            remote(host,code)
        code='''import json,hashlib
from pathlib import Path
pins=[]
for name,wanted in %r.items():
 p=Path(name);assert p.is_file() and not p.is_symlink() and p.stat().st_uid==500,'missing/nonowned control file: '+name
 actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==wanted,'device file SHA differs: '+name
 pins.append(dict(path=name,SHA256=actual))
print(json.dumps(pins))'''%expected
        pins=remote(host,code)
        # CPU uses the actual Python/import environment, complete load_batch, and all three
        # actual CLI preparations. No flags omitted; only --execute is removed.
        argv=jobs[role]['argv'][:-1]
        code='''import json,subprocess,os\nfrom pathlib import Path\nc=Path(%r);env=json.loads((c/'airline4-cpu-driver-v1/environment.json').read_text())['environment'] if %r=='receiver' else dict(os.environ,CUDA_VISIBLE_DEVICES='');env['PYTHONPATH']=str(c)+':'+str(c/'inner-replay-tools-v2')+':'+env.get('PYTHONPATH','');assert env['CUDA_VISIBLE_DEVICES']==''\ncommands=[%r,[%r,'-B','-S','-c',%r]];out=[]\nfor argv in commands:\n s=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=40);out.append(dict(rc=s.returncode,stdout=s.stdout,stderr=s.stderr[-2000:]));assert s.returncode==0,out\nprint(json.dumps(dict(state='ACTUAL_CLI_AND_BATCH_CPU_PASS',checks=out,model_requests=0)))'''%(freeze.CONTROL,role,argv,'/opt/conda310/bin/python3.10',
            "from pathlib import Path;from "+Path(recipe['helpers']['sender']).stem+" import load_batch;load_batch(Path("+repr(freeze.CONTROL+'/'+Path(recipe['batch']).name)+"),"+repr(plan['local_pins']['batch']['SHA256'])+");print('ACTUAL_BATCH_LOADED')")
        result=remote(host,code)
        save(role+'-device-CPU.json',dict(control_pins=pins,result=result))
        if role in ['P','D'] and not a.readiness:
            row=plan['roles'][role]
            code='''import json,time,os\nfrom pathlib import Path\nc=Path(%r);row=%r;r=Path(row['root']);v=json.loads((r/'model-service-status.json').read_text());assert v['state']=='MODEL_READY_KV_EMPTY_PENDING' and v['model_READY'] and time.time()-v['heartbeat_at']<=45\nassert v['plan_SHA256']==row['plan_SHA256'] and v['namespace']==row['namespace']\nassert v['deadline_epoch']-420-time.time()>195,'insufficient original active lease for phase, no model action'\nidentities=[v['model_root']]+v['GPU_ranks']\nfor who in identities:\n p=Path('/proc')/str(who['PID']);fields=(p/'stat').read_text().rsplit(') ',1)[1].split();assert p.stat().st_uid==int(who['UID']) and fields[19]==str(who['start_ticks']) and fields[0] not in ['Z','X','x']\nparent=json.loads(Path(row['parent_controller_status']).read_text());assert parent['state']=='RUNNING' and time.time()-parent['heartbeat_at']<=45\nprint(json.dumps(dict(state='LIVE_IDENTITY_READY_LEASE_PASS',model_requests=0,model_root=v['model_root'],ranks=v['GPU_ranks'],deadline_epoch=v['deadline_epoch'],active_remaining_seconds=v['deadline_epoch']-420-time.time(),parent_controller=parent['controller'])))'''%(freeze.CONTROL,row)
            save(role+'-live.json',dict(result=remote(host,code)))
    save('device-preflight-complete.json',dict(state='CPU_PASS_READY_WAIT_CONTROLLER_NEXT' if a.readiness else 'CPU_AND_LIVE_GATES_PASS_ACTUAL_PREPARE_STILL_REQUIRED',model_requests=0))
    if a.readiness:
        argv=['/opt/conda310/bin/python3.10','-B','-S',freeze.CONTROL+'/'+recipe['helpers']['readiness'],
              '--batch-json',freeze.CONTROL+'/'+Path(recipe['batch']).name,'--batch-sha256',plan['local_pins']['batch']['SHA256']]
        with lock.open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            result=cluster23_dispatch.submit('cluster3',freeze.HOSTS['P'],'root-'+plan['tag']+'-zero-ready',argv,700,
                  [dict(path='airline4-P-state-pilot-'+plan['tag']+'-zero-request-readiness/status.json',
                        json_fields=dict(state='ACTUAL_DIAGNOSTIC_SOURCE_REFS_AND_EMPTY_COMPLETE'))])
        save('readiness-submit.json',dict(result=result,model_requests=0))
    if a.submit:
        for role in ['receiver','P','D']:
            job=jobs[role]
            with lock.open('a') as guard:
                fcntl.flock(guard,fcntl.LOCK_EX)
                result=cluster23_dispatch.submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'])
            save(role+'-submit.json',dict(result=result,model_requests='PENDING_DEVICE_EVIDENCE'))
            freeze.need(result.get('actual_bootstrap_returncode')==0 or result.get('state')=='EXISTING_ATTEMPT_NOT_RESTARTED',
                        'controller submit failed; inspect device before any continuation')

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(state='BLOCKED_NO_BLIND_RETRY',reason=str(error))))
        raise SystemExit(1)
