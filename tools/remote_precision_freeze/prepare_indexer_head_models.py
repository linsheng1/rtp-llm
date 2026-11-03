"""New bounded HC031 epoch after the original successful model drain.

Reuses the exact model helper, compose/check and submit/controller. A recipe
may select one role so a healthy retained peer is never restarted with it.
"""
import argparse
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command', choices=['prepare', 'check', 'render', 'submit'])
    p.add_argument('--recipe', type=Path)
    p.add_argument('--directory', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--sha256')
    p.add_argument('--receipt', type=Path)
    a = p.parse_args()
    candidate = 'fb09988a5c552f4f5ff716472d8b50647af22d2c'
    if a.command != 'prepare':
        plan, pin = f.doc(a.plan)
        f.need(pin['SHA256'] == a.sha256 and plan['candidate_commit'] == candidate,
               'immutable actual HC031 epoch required')
        for value in plan['local_pins'].values():
            f.need(f.read(value['path'])[1] == value, 'epoch source drift')
        if a.command == 'check':
            print(json.dumps(dict(state='OFFLINE_EPOCH_PINS_PASS', model_requests=0)))
            return
        if a.command == 'render':
            print(json.dumps(plan['job']))
            return
        f.need(a.receipt is not None and not a.receipt.exists(), 'new receipt required')
        sys.path.insert(0, plan['bench'])
        from cluster23_dispatch import submit
        job = plan['job']
        result = submit('cluster3', job['host'], job['name'], job['argv'],
                        job['timeout'], job['verify'])
        f.save_new(a.receipt, dict(plan_SHA256=a.sha256, launch=result))
        print(json.dumps(dict(state='SUBMITTED_NOT_READY', model_requests=0)))
        return
    recipe, rp = f.doc(a.recipe)
    f.need(recipe['candidate_commit'] == candidate and recipe['seconds'] == 1800,
           'same candidate and original bounded model lifetime')
    f.need(set(recipe['roles']) <= {'P', 'D'} and recipe['roles'], 'explicit roles required')
    sys.path.insert(0, recipe['bench'])
    from remote_control import remote
    files = {name: f.read(path)[1] for name, path in [
        ('entry', Path(__file__)),
        ('helper', f.BASE/'device/c3_indexer_head_model_v1.py'),
        ('wrapper', Path(recipe['bench'])/'c3_after_reference_drain_v1.py')]}
    a.directory.mkdir(parents=True, exist_ok=True)
    for role, row in recipe['roles'].items():
        ref, refpin = f.doc(row['reference_dispatch'])
        argv = list(ref['argv'])
        f.need(argv[3] == f.CONTROL+'/c3_hc031_model_service_v1.py' and
               f.option(argv, '--manifest-sha256') in [
                   '8614842e06e19bb5be272326ef990ff42263a837745fdbb12a22489e8f1e6ba5',
                   '179428dbcd30ca64ce914ad0dad5f1b6d2bf5aa43682ffca8c5d87680f7c0115'],
               'actual candidate model reference required')
        if '--mode' in argv:
            i = argv.index('--mode')
            del argv[i:i+2]
        previous = row['previous_root']
        prepared,overlay_pin=f.doc(row['overlay_actual']);f.need(prepared['controller']['state']=='SUCCEEDED' and prepared['result']['candidate_commit']==candidate,'completed actual per-role overlay required');new=prepared['result']
        argv[3]=f.CONTROL+'/c3_indexer_head_model_v1.py'
        for flag,value in {'--env-json':new['environment']['path'],'--env-sha256':new['environment']['SHA256'],'--manifest':new['manifest']['path'],'--manifest-sha256':new['manifest']['SHA256'],'--consumer-root':f.CONTROL+'/consumer91b-'+role+'-v7','--keeper-root':f.CONTROL+'/hc031-static-'+role+'-v4'}.items():argv[argv.index(flag)+1]=value
        assets={f.CONTROL+'/'+Path(files['helper']['path']).name:Path(files['helper']['path']).read_text()}
        remote('cluster3',f.HOSTS[role],"import os,json\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable helper drift'\n else:\n  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())\nprint(json.dumps(dict(source_verified=True)))"%assets,60)
        root = f.remote_path(row['root'])
        f.need(Path(root).parent == Path(f.CONTROL) and root != previous,
               'new direct owned model root required')
        for key, value in [('--root', root), ('--seconds', '1800'),
                           ('--namespace', row['namespace'])]:
            argv[argv.index(key)+1] = value
        code = '''import json,os,subprocess,time,hashlib
from pathlib import Path
a=%r;c=Path(%r);checks=[]
for flag,state in [('--consumer-root','READY'),('--keeper-root','RUNNING')]:
 s=json.loads((Path(a[a.index(flag)+1])/'status.json').read_text())
 assert s['state']==state and time.time()-s['heartbeat_at']<=45 and s['deadline_epoch']>time.time()+1815,'existing dependency lease insufficient'
assert hashlib.sha256(Path(a[3]).read_bytes()).hexdigest()==%r,'actual model helper drift'
for mode in ['compose','check']:
 if mode=='check' and (c/'config-exit.json').exists():
  ex=json.loads((c/'config-exit.json').read_text());pr=json.loads((c/'config-probe.json').read_text());sha=hashlib.sha256((c/'plan.json').read_bytes()).hexdigest()
  assert ex['actual_child_returncode']==0 and not ex['timed_out'] and ex['plan_SHA256']==pr['plan_SHA256']==sha and pr['state']=='CPU_MODEL_CONFIG_PASS_NO_MODEL' and pr['exit_code']==0,'existing complete CPU identity differs'
  checks.append(dict(mode=mode,returncode=0,reused_completed_exact_plan=True));continue
 q=subprocess.run(a+['--mode',mode],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180)
 checks.append(dict(mode=mode,returncode=q.returncode,stdout=q.stdout[-800:],stderr=q.stderr[-2000:]))
 assert q.returncode==0,checks
print(json.dumps(dict(state='ACTUAL_CPU_CONFIG_PASS_ZERO_RPC',checks=checks,model_requests=0,plan_SHA256=hashlib.sha256((c/'plan.json').read_bytes()).hexdigest())))
''' % (argv, root, files['helper']['SHA256'])
        cpu = remote('cluster3', f.HOSTS[role], code, 420)
        f.save_new(a.directory/(role+'-CPU.json'), cpu)
        name = row['job_name']
        f.need(__import__('re').fullmatch('[A-Za-z0-9_-]+', name), 'owned job name required')
        job = dict(host=f.HOSTS[role], name=name, timeout=2340,
                   argv=['/opt/conda310/bin/python3.10', '-B', '-S',
                         f.CONTROL+'/c3_after_reference_drain_v1.py',
                         '--previous-root', previous, '--state-root',
                         f.CONTROL+'/'+name+'-after-drain', '--']+argv+['--mode', 'run'],
                   verify=[dict(path=Path(root).name+'/model-service-final.json',
                                json_fields=dict(state='MODEL_SERVICE_EXITED'))])
        plan = dict(schema='existing-hc031-model-epoch-v1', candidate_commit=candidate,
                    role=role, model_root=root, model_plan_SHA256=cpu['plan_SHA256'],
                    CPU=cpu, job=job, bench=recipe['bench'], model_requests=0,
                    local_pins=dict(files, recipe=rp, reference=refpin,overlay=overlay_pin))
        sha = f.save_new(a.directory/(role+'-model-plan.json'), plan)
        f.save_new(a.directory/(role+'-dispatch.json'), dict(
            role=role, host=f.HOSTS[role], root=Path(root).name,
            plan_SHA256=cpu['plan_SHA256'], argv=argv+['--mode', 'run'],
            parent_controller_status=f.CONTROL+'/'+name+'-status.json', epoch_plan_SHA256=sha))
        print(json.dumps(dict(role=role, CPU='PASS', plan_SHA256=sha, model_requests=0)), flush=True)


if __name__ == '__main__':
    main()
