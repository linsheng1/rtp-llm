"""Recipe -> real CPU compose/check -> immutable outer plans -> old submit.

Only C3's existing P12/D71 environment is used. Old model epochs must be gone;
Consumers and full-content static-weight leases remain live and are re-audited.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command',choices=['prepare','check','render','submit'])
    p.add_argument('--recipe',type=Path)
    p.add_argument('--directory',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--sha256')
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command!='prepare':
        plan,pin=f.doc(a.plan)
        f.need(pin['SHA256']==a.sha256 and plan['schema']=='capture41-existing-model-epoch-v1','immutable model plan differs')
        for local in plan['local_pins'].values():
            f.need(f.read(local['path'])[1]==local,'history model source/input drift')
        job=dict(plan['job'])
        if a.command=='check':
            print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)))
            return
        if a.command=='render':
            print(json.dumps(job))
            return
        f.need(a.receipt is not None and not a.receipt.exists(),'new submit receipt required')
        sys.path.insert(0,plan['bench'])
        from cluster23_dispatch import submit
        with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'])
        f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=launch))
        print(json.dumps(dict(state='SUBMITTED_NOT_READY',model_requests=0)))
        return
    recipe,rp=f.doc(a.recipe)
    f.need(recipe['schema']=='capture41-existing-model-recipe-v1' and recipe['seconds']==1800
           and recipe['namespace']=='pubprec_hc031_hist7045_v1','finite scoped history recipe required')
    unit,up=f.doc(recipe['unit_receipt'])
    f.need(unit['controller']['state']==unit['status']['state']=='SUCCEEDED' and unit['result']['passed']
           and unit['result']['tests_run']==3 and unit['result']['skipped']==0,'existing unchanged graph-latch contract evidence required')
    files={key:f.read(f.BASE/'device'/name)[1] for key,name in [
        ('helper','c3_hc031_capture41_model_v1.py'),('observer','history41_observer_v1.py'),
        ('bootstrap','history41_bootstrap_v1.py'),('latches','history38_latches_v1.py')]}
    files['drain_wrapper']=f.read(Path(recipe['bench'])/'c3_after_reference_drain_v1.py')[1]
    sys.path.insert(0,recipe['bench'])
    from remote_control import remote
    a.directory.mkdir(parents=True,exist_ok=True)
    for role in ['P','D']:
        ref,refpin=f.doc(recipe['roles'][role]['reference_dispatch'])
        argv=list(ref['argv'])
        f.need(argv[3]==f.CONTROL+'/c3_hc031_model_service_v1.py','original actual source profile required')
        root=f.CONTROL+'/airline4-'+role+'-capture41v2'
        argv[3]=f.CONTROL+'/c3_hc031_capture41_model_v1.py'
        for key,value in {'--root':root,'--seconds':'1800','--namespace':recipe['namespace'],
                          '--consumer-root':f.CONTROL+'/consumer91b-'+role+'-v4',
                          '--keeper-root':f.CONTROL+'/hc031-static-'+role+'-v2'}.items():
            argv[argv.index(key)+1]=value
        for flag in ['--mode']:
            if flag in argv:
                i=argv.index(flag)
                del argv[i:i+2]
        remote_config=f.CONTROL+'/capture41v2-'+role+'-config.json'
        manifest=dict(path=f.option(argv,'--manifest'),SHA256=f.option(argv,'--manifest-sha256'))
        host=f.HOSTS[role]
        code='''import json,os,hashlib
from pathlib import Path
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;m=json.loads(raw)
keys=%r
print(json.dumps(dict(hooks={k:m['production_Python'][k.replace('.','/')+'.py']['SHA256'] for k in keys})))
'''%(manifest['path'],manifest['SHA256'],[
            'rtp_llm.models_py.modules.dsv41.block','rtp_llm.models_py.modules.dsv41.fp8.attention_v41',
            'rtp_llm.models_py.modules.dsv41.moe.moe_layer','rtp_llm.models_py.modules.dsv41.hc.delayed',
            'rtp_llm.models_py.modules.base.cuda.norm'])
        with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            observed=remote('cluster3',host,code,60)
        config=dict(schema='history41-inner-graph-latches-v1',source_commit=recipe['candidate_commit'],
                    role=role,host=host,artifact_kind='inner_python_overlay',wheel_sha256=None,
                    input_ids_sha256=recipe['R_SHA256'],
                    layers=list(range(26)),positions=[7045],gpu_budget_bytes=4194304,
                    output=f.CONTROL+'/h41-'+role,hook_source_sha256=observed['hooks'],
                    environment_SHA256=f.option(argv,'--env-sha256'),runtime_manifest_SHA256=manifest['SHA256'],
                    manifest_path=manifest['path'],control_source_SHA256={Path(v['path']).name:v['SHA256']
                        for k,v in files.items() if k in ['observer','bootstrap','latches']},latches_SHA256=files['latches']['SHA256'])
        local_config=a.directory/('capture41v2-'+role+'-config.json')
        config_sha=f.save_new(local_config,config)
        argv+=['--history-config',remote_config,'--history-config-sha256',config_sha]
        assets={f.CONTROL+'/'+Path(v['path']).name:Path(v['path']).read_text() for v in files.values()}
        assets[remote_config]=local_config.read_text()
        for path,source in assets.items():
            code='''import json,os
from pathlib import Path
p=Path(%r);raw=%r;assert os.getuid()==500
if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable history source/config drift'
else:
 with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
print(json.dumps(dict(source_verified=True)))
'''%(path,source)
            with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                remote('cluster3',host,code,60)
        code='''import json,os,subprocess,importlib.util,time
from pathlib import Path
from types import SimpleNamespace
c=Path(%r);old=c/%r
state=json.loads((old/'model-status.json').read_text());live=[]
for row in state.get('owned_processes',[]):
 p=Path('/proc')/str(row['PID'])
 if p.exists():
  v=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
  if v[19]==str(row['start_ticks']) and v[0] not in ['Z','X','x']:live.append(row['PID'])
# CPU compose/check is CUDA-hidden; original drain wrapper gates GPU start.
argv=%r
def option(key):return argv[argv.index(key)+1]
spec=importlib.util.spec_from_file_location('same_history_model_admission',argv[3]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
args=SimpleNamespace(role=option('--role'),host=option('--host'),seconds=int(option('--seconds')),consumer_root=Path(option('--consumer-root')),
 keeper_root=Path(option('--keeper-root')),keeper_helper=Path(option('--keeper-helper')),keeper_helper_sha256=option('--keeper-helper-sha256'),
 weights_mode='retained',env_json=Path(option('--env-json')),env_sha256=option('--env-sha256'),manifest=Path(option('--manifest')),manifest_sha256=option('--manifest-sha256'))
controller=m.load(c/'device_controller_cluster23_v1.py',m.CONTROLLER_SHA,'same_original_controller_CPU')
service=m.load(Path(option('--consumer-helper')),option('--consumer-helper-sha256'),'same_actual_Consumer4_CPU')
consumer=m.consumer(args,controller,service);keeper=m.keeper(args,controller)
checks=[]
for mode in ['compose','check']:
 q=subprocess.run(argv+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180)
 checks.append(dict(mode=mode,returncode=q.returncode,stdout=q.stdout[-800:],stderr=q.stderr[-2000:]))
 assert q.returncode==0,checks
p=Path(option('--root'))/'plan.json';raw=p.read_bytes()
print(json.dumps(dict(state='HISTORY36_ACTUAL_CPU_PREPARE_PASS',checks=checks,plan_SHA256=__import__('hashlib').sha256(raw).hexdigest(),
 old_same_epoch_owned_live=live,old_terminal_preserved=True,Consumer=consumer[1],keeper=keeper,model_requests=0)))
'''%(f.CONTROL,recipe['roles'][role]['previous_root'],argv)
        with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            cpu=remote('cluster3',host,code,420)
        f.save_new(a.directory/(role+'-CPU.json'),cpu)
        old_root=f.CONTROL+'/'+recipe['roles'][role]['previous_root']
        wrapped=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/c3_after_reference_drain_v1.py','--previous-root',old_root,
                 '--state-root',f.CONTROL+'/capture41v2-normal2-'+role+'-after-drain','--']+argv+['--mode','run']
        job=dict(host=host,name='capture41v2-'+role+'-model',argv=wrapped,timeout=2340,
                 verify=[dict(path=Path(root).name+'/model-service-final.json',json_fields=dict(state='MODEL_SERVICE_EXITED'))])
        plan=dict(schema='capture41-existing-model-epoch-v1',candidate_commit=recipe['candidate_commit'],role=role,
                  model_root=root,model_plan_SHA256=cpu['plan_SHA256'],CPU=cpu,job=job,bench=recipe['bench'],model_requests=0,
                  lifetime_accounting=dict(service_seconds=1800,original_drain_reserve=420,outer_seconds=2340,
                     outer_additional_budget='original240s after-reference drain + original1800s service +300s admission/config/release; submit only when old full deadline is within240s'),
                  local_pins=dict(files,recipe=rp,unit_receipt=up,reference_dispatch=refpin,
                     config=f.read(local_config)[1],entry=f.read(__file__)[1]))
        plan_path=a.directory/(role+'-model-plan.json')
        sha=f.save_new(plan_path,plan)
        f.save_new(a.directory/(role+'-dispatch.json'),dict(role=role,host=host,root=Path(root).name,
            plan_SHA256=cpu['plan_SHA256'],argv=argv+['--mode','run'],parent_controller_status=f.CONTROL+'/'+job['name']+'-status.json',epoch_plan_SHA256=sha))
        print(json.dumps(dict(role=role,CPU='PASS',plan_SHA256=sha,model_requests=0)),flush=True)


if __name__=='__main__':
    main()
