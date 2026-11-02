"""Freeze -> check -> render -> original submit for one bounded STATE test."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--bench',type=Path);p.add_argument('--cpu-receipt',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.command=='freeze':
        bench=a.bench;files={key:f.read(path)[1] for key,path in [('entry',Path(__file__)),('script',f.BASE/'state_projection_fix_probe_v2.py'),('nvml_helper',bench/'c3_FFN_MegaMHC_counterfactual_v1.py'),('comparator',bench/'compare_c3_P_hc_transition_v3.py')]}
        files['bootstrap']=f.read(f.BASE/'gpu_before_runtime_exec.py')[1]
        mode='run' if a.cpu_receipt else 'check';tag='state-projection-fixed-capture23-'+mode+'-v3';plan=dict(schema='existing-state-projection-plan-v1',mode=mode,candidate_commit='61887b83eb180619032a46f0f2dc0eb86248ee09',capture_plan_SHA256='e9d7fc2c30a7c91080ae95ea00d52deed8bddf8141b85cb36c541fb1c98a6833',input_SHA256='07cf16434cfb7964bb1090f9c4c4d4b30823be0ef3bb3ace8c58fd4de913522c',model_requests=0,primitive_calls_max=5,output_root=f.CONTROL+'/'+tag,comparison=dict(path=f.CONTROL+'/capture23-existing-analysis-v1/comparison-safe.json',SHA256='b6b23ac04b96f6370b4bd2a998cba3a6a22de5cb7fd65d657e9a635ecfe4d14b'),environment=dict(path=f.CONTROL+'/dg-batch-invariant-model-inner-v1/environment.json',SHA256='145151075dd1f66e50fa2a5e69859afb0ec7b139efdb9b1c7d51a42586218343'),manifest=dict(path=f.CONTROL+'/dg-batch-invariant-model-inner-v1/manifest.json',SHA256='3d4a271e6ae3875e47e6597f964f810897e9734a93e660aabd62a88a4d022629'),model_plan=dict(path=f.CONTROL+'/airline4-P-dgdet-capture21/plan.json',SHA256='d3c26c53bcd284a41452da744481d367574e1d8e4977eed9bc7c74099d3d9551'),local_pins=files,bench=str(bench.resolve()))
        for key in ['script','nvml_helper','comparator','bootstrap']:plan[key]=dict(path=f.CONTROL+'/'+Path(files[key]['path']).name,SHA256=files[key]['SHA256'])
        if a.cpu_receipt:
            cpu,cp=f.doc(a.cpu_receipt);f.need(cpu['controller']['state']=='SUCCEEDED' and cpu['status']['state']=='SUCCEEDED' and cpu['result']['state']=='STATE_PROJECTION_ACTUAL_INPUT_CPU_PASS' and cpu['result']['primitive_calls']==0,'actual CPU preflight incomplete');plan['CPU_receipt']=cpu['status']['result'];plan['local_pins']['CPU_receipt']=cp
        plan['admission_proof']=f.CONTROL+'/'+tag+'-admission.json'
        plan['job']=dict(host='10.44.12.175',name=tag,timeout=240,argv=['/opt/conda310/bin/python3.10','-B','-S',plan['bootstrap']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path=tag+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(state='FROZEN_OPERATOR_'+mode.upper(),SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256 and plan['schema']=='existing-state-projection-plan-v1','plan identity differs')
    for v in plan['local_pins'].values():f.need(f.read(v['path'])[1]==v,'local pin drift')
    job=plan['job'];job=dict(job,argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in job['argv']])
    if a.command=='check':print(json.dumps(dict(state='FROZEN_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    sys.path.insert(0,plan['bench']);from remote_control import remote
    from cluster23_dispatch import submit
    source=Path(plan['local_pins']['script']['path']).read_text();bootstrap=Path(plan['local_pins']['bootstrap']['path']).read_text()
    code='''import os,json,subprocess,hashlib\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source/plan differs'\n else:\n  with p.open('x') as f:f.write(raw);f.flush();os.fsync(f.fileno())\ns=subprocess.run(%r,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert s.returncode==0,s.stderr\ne=Path(%r);assert hashlib.sha256(e.read_bytes()).hexdigest()==%r;original=json.loads(e.read_text())['environment'];env={k:original[k] for k in ['PYTHONPATH','LD_LIBRARY_PATH','PATH','CUDA_HOME']};env['CUDA_VISIBLE_DEVICES']=%r\nprint(json.dumps(dict(CPU_CLI='PASS',environment=env)))'''%({plan['script']['path']:source,plan['bootstrap']['path']:bootstrap,f.CONTROL+'/'+a.plan.name:a.plan.read_text()},job['argv'][:4]+['--help'],plan['environment']['path'],plan['environment']['SHA256'],'0' if plan['mode']=='run' else '')
    env=remote('cluster3',job['host'],code,60)
    launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''})
    f.need(a.receipt is not None,'receipt required');f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU_CLI=env['CPU_CLI'],launch=launch));print(json.dumps(dict(state='SUBMITTED_NOT_VALIDATED',model_requests=0)))
if __name__=='__main__':main()
