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
INDEXER_MODULES=('rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer','rtp_llm.models_py.modules.dsv41.fp8._v41_decode_indexer','rtp_llm.models_py.modules.dsv41.fp8._indexer_score')


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
        f.need(pin['SHA256']==a.sha256 and plan['schema']=='capture45-existing-model-epoch-v1','immutable model plan differs')
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
    f.need(recipe['schema']=='capture45-existing-model-recipe-v1' and recipe['seconds']==1800
           and recipe['namespace']=='pubprec_hc031_hist7045_indexer45_v1','finite scoped history recipe required')
    unit,up=f.doc(recipe['unit_receipt'])
    f.need(unit['controller']['state']==unit['status']['state']=='SUCCEEDED' and unit['result']['passed']
           and unit['result']['tests_run']==3 and unit['result']['skipped']==0,'existing unchanged graph-latch contract evidence required')
    files={key:f.read(f.BASE/'device'/name)[1] for key,name in [
        ('helper','c3_hc031_capture45_model_v1.py'),('observer','history45_observer_v1.py'),
        ('bootstrap','history45_bootstrap_v1.py'),('latches','history38_latches_v1.py')]}
    files['base_bootstrap44']=f.read(f.BASE/'device/history44_bootstrap_v1.py')[1]
    files['base_service44']=f.read(f.BASE/'device/c3_hc031_capture44_model_v1.py')[1]
    files['base_observer44']=f.read(f.BASE/'device/history44_observer_v1.py')[1]
    files['base_core44']=f.read(f.BASE/'device/c3_hc031_capture44_core_v1.py')[1]
    files['base_client44']=f.read(f.BASE/'device/c3_hc031_capture44_client_v1.py')[1]
    files['failed_startup_wrapper']=f.read(f.BASE/'device/c3_after_failed_startup_cleanup_v1.py')[1]
    files['drain_wrapper']=f.read(Path(recipe['bench'])/'c3_after_reference_drain_v1.py')[1]
    new_contract,contract_pin=f.doc(recipe['readset_contract_receipt'])
    f.need(new_contract['controller']['state']=='SUCCEEDED' and new_contract['result']['state']=='CPU_GATHER_CANONICAL_BOUNDS_AND_NO_HIT_PASS' and new_contract['result']['contracts']==6 and new_contract['result']['CUDA_initialized'] is False,'actual new CPU readset contracts required')
    files['readset_contract_receipt']=contract_pin
    for kind in ['CPU', 'GPU']:
        actual, proof_pin=f.doc(recipe['chain_'+kind+'_receipt'])
        f.need(actual['controller']['state']=='SUCCEEDED','full-chain '+kind+' controller failed')
        if kind=='CPU':
            f.need(actual['result']['state']=='ACTUAL_INDEXER_CONTROL_CHAIN_PASS' and actual['result']['mode']=='cpu' and {x['role'] for x in actual['result']['roles']}=={'P','D'},'real CPU record/export contract')
        else:
            f.need(actual['status']['state']=='SUCCEEDED' and actual['result']['passed'] and actual['result']['tests_run']==1 and actual['result']['skipped']==0,'P eager / D graph control contract')
        files['chain_'+kind+'_receipt']=proof_pin
    budget=recipe['whole_chain_budget_seconds']
    f.need(budget==dict(handoff=240,load_READY=240,prepare=120,paired_probes=120,native_requests=360,exports=300,drain=420),'explicit unchanged bounded two-request chain')
    f.need(sum(v for k,v in budget.items() if k not in ['handoff','drain'])<=1800-420,'whole chain exceeds active lease')
    sys.path.insert(0,recipe['bench'])
    from remote_control import remote
    a.directory.mkdir(parents=True,exist_ok=True)
    for role in ['P','D']:
        ref,refpin=f.doc(recipe['roles'][role]['reference_dispatch'])
        argv=list(ref['argv'])
        f.need(argv[3]==f.CONTROL+'/c3_hc031_model_service_v1.py','original actual source profile required')
        root=f.CONTROL+'/airline4-'+role+'-capture45'
        argv[3]=f.CONTROL+'/c3_hc031_capture45_model_v1.py'
        for key,value in {'--root':root,'--seconds':'1800','--namespace':recipe['namespace'],
                          '--consumer-root':f.CONTROL+'/consumer91b-'+role+'-v5',
                          '--keeper-root':f.CONTROL+'/hc031-static-'+role+'-v3'}.items():
            argv[argv.index(key)+1]=value
        for flag in ['--mode']:
            if flag in argv:
                i=argv.index(flag)
                del argv[i:i+2]
        remote_config=f.CONTROL+'/capture45-'+role+'-config.json'
        manifest=dict(path=f.option(argv,'--manifest'),SHA256=f.option(argv,'--manifest-sha256'))
        host=f.HOSTS[role]
        code='''import json,os,hashlib
from pathlib import Path
p=Path(%r);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==%r;m=json.loads(raw)
keys=%r
print(json.dumps(dict(hooks={k:m['production_Python'][k.replace('.','/')+'.py']['SHA256'] for k in keys})))
'''%(manifest['path'],manifest['SHA256'],[
            'rtp_llm.models_py.modules.dsv41.block','rtp_llm.models_py.modules.dsv41.fp8.attention_v41',
            'rtp_llm.models_py.modules.dsv41.fp8.attention','rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv',
            'rtp_llm.models_py.modules.dsv41.fp8._v41_mega_attention','rtp_llm.models_py.modules.dsv41.utils',
            'rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer','rtp_llm.models_py.modules.dsv41.fp8._v41_decode_indexer','rtp_llm.models_py.modules.dsv41.fp8._indexer_score'])
        with open('/tmp/decode-remote-ssh-501/dsh-task-remote.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            observed=remote('cluster3',host,code,60)
        config=dict(schema='history44-inner-graph-latches-v1',source_commit=recipe['candidate_commit'],
                    role=role,host=host,artifact_kind='inner_python_overlay',wheel_sha256=None,
                    input_ids_sha256=recipe['R_SHA256'],
                    layers=[24],positions=[7045],gpu_budget_bytes=4194304,
                    output=f.CONTROL+'/h45-'+role,hook_source_sha256={k:v for k,v in observed['hooks'].items() if k not in INDEXER_MODULES},
                    indexer_extension=dict(revision=45,causal_prefix=7046,base44_SHA256=files['base_observer44']['SHA256'],source_sha256={k:observed['hooks'][k] for k in INDEXER_MODULES},checkpoint_config_SHA256='8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'),
                    environment_SHA256=f.option(argv,'--env-sha256'),runtime_manifest_SHA256=manifest['SHA256'],
                    manifest_path=manifest['path'],control_source_SHA256={Path(v['path']).name:v['SHA256']
                        for k,v in files.items() if k in ['observer','bootstrap','latches','base_observer44']},latches_SHA256=files['latches']['SHA256'])
        local_config=a.directory/('capture45-'+role+'-config.json')
        config_sha=f.save_new(local_config,config)
        argv+=['--history-config',remote_config,'--history-config-sha256',config_sha]
        assets={f.CONTROL+'/'+Path(v['path']).name:Path(v['path']).read_text() for v in files.values()}
        assets[remote_config]=local_config.read_text()
        groups=[];group={}
        for path,source in assets.items():
            if len(repr(dict(group,**{path:source})).encode())>80000:
                f.need(group,'single control source too large');groups.append(group);group={}
            group[path]=source
        if group:groups.append(group)
        for group in groups:
            code="""import json,os
from pathlib import Path
assert os.getuid()==500
for path,raw in %r.items():
 p=Path(path)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable history source/config drift'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
print(json.dumps(dict(source_verified=True)))
"""%group
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
consumer=m.consumer(args,controller,service,required_seconds=2055);keeper=m.keeper(args,controller,required_seconds=2055)
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
                 '--state-root',f.CONTROL+'/capture45-normal2-'+role+'-after-drain','--']+argv+['--mode','run']
        if recipe['roles'][role].get('failed_previous_startup'):
            failed,failed_pin=f.doc(recipe['roles'][role]['failed_model_plan'])
            f.need(failed['role']==role and failed['model_root']==old_root,'exact failed latest startup')
            files['failed_previous_plan']=failed_pin
            wrapped=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/c3_after_failed_startup_cleanup_v1.py','--previous-root',old_root,'--previous-plan-sha256',failed['model_plan_SHA256'],'--state-root',f.CONTROL+'/capture45-'+role+'-failed-startup-boundary','--']+argv+['--mode','run']
        job=dict(host=host,name='capture45-'+role+'-model',argv=wrapped,timeout=2340,
                 verify=[dict(path=Path(root).name+'/model-service-final.json',json_fields=dict(state='MODEL_SERVICE_EXITED'))])
        plan=dict(schema='capture45-existing-model-epoch-v1',candidate_commit=recipe['candidate_commit'],role=role,
                  model_root=root,model_plan_SHA256=cpu['plan_SHA256'],CPU=cpu,job=job,bench=recipe['bench'],model_requests=0,
                  whole_chain_budget_seconds=budget,
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
