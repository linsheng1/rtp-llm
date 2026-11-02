"""Freeze actual EOF receipts, then existing CPU comparator/controller; zero RPC."""
import argparse
import json
from pathlib import Path
import sys
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command',choices=['freeze','check','render','submit'])
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256')
    p.add_argument('--normal',type=Path)
    p.add_argument('--d-result',type=Path)
    p.add_argument('--p-result',type=Path)
    p.add_argument('--root')
    p.add_argument('--bench',type=Path)
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        pins={name:f.read(path)[1] for name,path in [
            ('entry',Path(__file__)),('script',f.BASE/'device/compare_hc031_results_v2.py'),
            ('diagnostic',f.BASE/'device/c3_hc031_history_diagnostic_v1.py')]}
        results={}
        for key,path in [('baseline',a.normal),('d-result',a.d_result),('p-result',a.p_result)]:
            if path is None:continue
            value,pin=f.doc(path);native=value['native_result']
            f.need(native['EOF'] is True and native['input_SHA256']=='3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c','actual EOF on frozen HC031 input required')
            results[key]={k:native[k] for k in ['path','SHA256']};pins['receipt:'+key]=pin
        f.need('baseline' in results and len(results)>1,'normal and actual restore required')
        root=f.remote_path(a.root);f.need(Path(root).parent==Path(f.CONTROL),'owned output root')
        plan=dict(candidate_commit='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c',root=root,
            input_SHA256='3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c',
            source=dict(path=f.CONTROL+'/hc031-source-source1/request/native/result.json',SHA256='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa'),
            normal_reference=dict(path=f.CONTROL+'/hc031-baseline-native-hc031N1/native/result.json',SHA256='6c70f86ba92b085933ef371e17a9a380e622c90667e6c4eccae4283e88db9333'),results=results,local_pins=pins,bench=str(a.bench),model_requests=0)
        for key in ['script','diagnostic']:plan[key]=dict(path=f.CONTROL+'/'+Path(pins[key]['path']).name,SHA256=pins[key]['SHA256'])
        plan['job']=dict(host=f.HOSTS['receiver'],name=Path(root).name,timeout=150,
            argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],
            verify=[dict(path=Path(root).name+'/status.json',json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'immutable comparison plan drift')
    for value in plan['local_pins'].values():f.need(f.read(value['path'])[1]==value,'comparison source drift')
    job=dict(plan['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in plan['job']['argv']])
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    sys.path.insert(0,plan['bench']);from remote_control import remote
    from cluster23_dispatch import submit
    assets={plan['script']['path']:Path(plan['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
    code='''import json,os,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable comparison source drift'
 else:
  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
r=subprocess.run(%r+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CPU_CLI='PASS',model_requests=0)))
'''%(assets,job['argv'][:4])
    cpu=remote('cluster3',job['host'],code,60)
    result=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'])
    f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=result))
    print(json.dumps(dict(state='SUBMITTED_NOT_COMPARED',model_requests=0)))

if __name__=='__main__':main()
