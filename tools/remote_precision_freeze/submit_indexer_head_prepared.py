"""Reuse completed CPU prepare after verifying immutable device references.

This is only the existing frozen argv -> original submit path. It starts no
models, scheduler, polling loop, retry, or modified device admission.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--prepared-directory',type=Path,required=True)
    p.add_argument('--evidence-directory',type=Path,required=True)
    a=p.parse_args()
    plan,pin=f.doc(a.plan)
    f.need(pin['SHA256']==a.sha256 and plan['candidate_commit']=='fb09988a5c552f4f5ff716472d8b50647af22d2c','actual frozen candidate')
    f.validate(plan)
    rendered,rpin=f.doc(a.prepared_directory/'rendered.json')
    proof,ppin=f.doc(a.prepared_directory/'device-CPU-prepare.json')
    f.need(rendered['plan_SHA256']==proof['plan_SHA256']==a.sha256 and proof['actual_RPC']==0,'same complete CPU prepare')
    f.need(all(v.get('actual_CLI')=='PASS' for k,v in proof['preflight'].items() if k in ['receiver','P','D']),'actual three-role CLI prepare')
    recipe,_=f.doc(plan['local_pins']['recipe']['path'])
    def cli(*args):
        r=subprocess.run([sys.executable,'-B',str(f.BASE/'freeze_stage.py'),*args],capture_output=True,text=True,timeout=40)
        f.need(r.returncode==0,r.stdout+r.stderr)
        return json.loads(r.stdout)
    cli('check','--plan',str(a.plan),'--sha256',a.sha256,'--cli-help')
    jobs={role:cli('render','--plan',str(a.plan),'--sha256',a.sha256,'--role',role) for role in ['receiver','P','D']}
    f.need(jobs==rendered['jobs'],'frozen rendered argv reuse drift')
    a.evidence_directory.mkdir(parents=True,exist_ok=False)
    f.save_new(a.evidence_directory/'submission-plan.json',dict(stage_plan=pin,prepared_render=rpin,prepared_CPU=ppin,entry=f.read(__file__)[1],jobs=jobs,actual_RPC=0))
    sys.path.insert(0,recipe['bench'])
    from remote_control import remote
    from cluster23_dispatch import submit
    pins={f.CONTROL+'/'+Path(plan['local_pins'][k]['path']).name:plan['local_pins'][k]['SHA256'] for k in recipe['helpers']}
    for role,job in jobs.items():
        source="""import hashlib,json,os,subprocess
from pathlib import Path
assert os.getuid()==500
for name,expected in %r.items():
 p=Path(name);assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,'existing immutable control reference drift'
r=subprocess.run(%r+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps({'state':'ACTUAL_IMMUTABLE_PREPARED_REFERENCES_PASS','actual_RPC':0}))
"""%(pins,job['argv'][:4])
        proof=remote('cluster3',job['host'],source,60)
        f.save_new(a.evidence_directory/(role+'-reference-check.json'),proof)
    for role in ['receiver','P','D']:
        job=jobs[role]
        root=Path(f.option(job['argv'],'--root' if role=='receiver' else '--output-root')).name
        source=plan['action']=='d-source'
        target=('D_SOURCE_COMPLETE' if source else 'NORMAL_COMPLETE') if role=='receiver' else 'ACTUAL_NATIVE_SOURCE_PROBE_UPLOADED_DEVICE_DIRECT_NO_RPC'
        launch=submit('cluster3',job['host'],'headfix-'+plan['tag']+'-'+role,job['argv'],960 if role=='receiver' else 660,[dict(path=root+'/status.json',json_fields=dict(state=target))])
        f.save_new(a.evidence_directory/(role+'-submit.json'),dict(stage_plan_SHA256=a.sha256,launch=launch))
    print(json.dumps(dict(state='THREE_ORIGINAL_JOBS_SUBMITTED_NOT_VALIDATED',model_requests='UNVERIFIED')))


if __name__=='__main__':
    main()
