"""Freeze/check/render a single read-only postprocessing job; original submit only."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit'])
    p.add_argument('--capture-plan',type=Path);p.add_argument('--capture-sha256');p.add_argument('--capture-receipt',type=Path)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        c,cp=f.doc(a.capture_plan);f.need(cp['SHA256']==a.capture_sha256,'capture plan differs');f.validate(c)
        report,rp=f.doc(a.capture_receipt);f.need(report['value']['plan_SHA256']==a.capture_sha256,'capture result differs')
        batch=c['device_files'][f.CONTROL+'/'+Path(json.loads(Path(c['local_pins']['recipe']['path']).read_text())['batch']).name]
        files={key:f.read(f.BASE/name)[1] for key,name in [('entry','analyze_capture_entry_v2.py'),('script','analyze_capture.py')]}
        bench=Path(c['local_pins']['core']['path']).parent
        helpers={key:f.read(bench/name)[1] for key,name in [('comparator','compare_c3_P_hc_transition_v3.py'),('observer','p_only_state_observer_v6.py'),('primary','compare_c3_P_state_boundary_v1.py')]}
        for pin in helpers.values():pin['local_path']=pin['path'];pin['path']=f.CONTROL+'/'+Path(pin['path']).name
        name=c['tag']+'-existing-analysis-v1';remote_plan=f.CONTROL+'/'+a.plan.name
        plan=dict(schema='existing-c3-capture-analysis-v1',classification='READ_ONLY_EXISTING_CAPTURE_NO_REQUEST',candidate_commit=c['candidate_commit'],capture_plan_SHA256=a.capture_sha256,
             input_SHA256=c['input']['SHA256'],p_plan_SHA256=c['roles']['P']['plan_SHA256'],report=report['report'],
             batch=dict(path=f.CONTROL+'/'+Path(batch['path']).name,SHA256=batch['SHA256']),
             script=dict(path=f.CONTROL+'/analyze_capture.py',SHA256=files['script']['SHA256']),helpers=helpers,
             output_root=f.CONTROL+'/'+name,local_pins=dict(capture_plan=cp,capture_receipt=rp,**files),
             job=dict(host=c['roles']['P']['host'],name=name,timeout=150,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/analyze_capture.py','--plan',remote_plan,'--sha256','${PLAN_SHA256}'],
                      verify=[dict(path=name+'/status.json',json_fields=dict(state='COMPARED_EXISTING_FROZEN_CAPTURE',model_requests=0))]),model_requests=0)
        print(json.dumps(dict(state='FROZEN_READ_ONLY_POSTPROCESS',SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256 and plan['schema']=='existing-c3-capture-analysis-v1','analysis plan identity differs')
    for local in plan['local_pins'].values():f.need(f.read(local['path'])[1]==local,'analysis local pin drift')
    for local in plan['helpers'].values():f.need(f.read(local['local_path'])[1]['SHA256']==local['SHA256'],'comparison helper drift')
    job=dict(plan['job']);job['argv']=[a.sha256 if v=='${PLAN_SHA256}' else v for v in job['argv']]
    if a.command=='render':print(json.dumps(job));return
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_POSTPROCESS_PINS_PASS',model_requests=0)));return
    # Small source/metadata only. No business/tensor bytes are read on the Mac.
    capture=json.loads(Path(plan['local_pins']['capture_plan']['path']).read_text());bench=Path(capture['local_pins']['core']['path']).parent;sys.path.insert(0,str(bench))
    from remote_control import remote
    from cluster23_dispatch import submit
    assets={'analyze_capture.py':Path(plan['local_pins']['script']['path']).read_text()}
    for helper in plan['helpers'].values():
        data=Path(helper['local_path']).read_text();assets[Path(helper['path']).name]=data
    for name,data in assets.items():
        code='''import json,os\nfrom pathlib import Path\np=Path(%r);raw=%r;assert os.getuid()==500\nif p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable analysis source differs'\nelse:\n with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())\nprint(json.dumps(dict(source_verified=True)))'''%(f.CONTROL+'/'+name,data)
        remote('cluster3',job['host'],code,60)
    code='''import json,os,subprocess\nfrom pathlib import Path\np=Path(%r);raw=%r;assert os.getuid()==500\nif p.exists():assert not p.is_symlink() and p.read_text()==raw\nelse:\n with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())\ns=subprocess.run(%r,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=10);assert s.returncode==0,s.stderr\nprint(json.dumps(dict(actual_CPU_CLI='PASS',model_requests=0)))'''%(f.CONTROL+'/'+a.plan.name,a.plan.read_text(),job['argv'][:4]+['--help'])
    cpu=remote('cluster3',job['host'],code,60)
    result=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'])
    f.need(a.receipt is not None,'receipt path required');f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=result))
    print(json.dumps(dict(state='SUBMITTED_NOT_COMPARED',model_requests=0)))
if __name__=='__main__':main()
