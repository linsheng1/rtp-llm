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
    p.add_argument('--normal-reference',type=Path)
    p.add_argument('--root')
    p.add_argument('--bench',type=Path)
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        pins={name:f.read(path)[1] for name,path in [
            ('entry',Path(__file__)),('script',f.BASE/'device/compare_indexer_head_results_v1.py'),
            ('diagnostic',f.BASE/'device/c3_indexer_head_history_diagnostic_v1.py')]}
        results={}
        for key,path in [('baseline',a.normal),('d-result',a.d_result),('p-result',a.p_result)]:
            if path is None:continue
            value,pin=f.doc(path);native=value['native_safe'];f.need(value['controller']['state']=='SUCCEEDED','actual completed controller required')
            f.need(native['EOF'] is True and native['input_ids_SHA256']=='10932132ad309cbfd30ae5600e64d7f7bb2dd8d7d2ddb6af4c6349ce2381ef71','actual EOF on frozen HC031 input required')
            results[key]={k:native[k] for k in ['path','SHA256']};pins['receipt:'+key]=pin
        f.need(set(results)=={'baseline','d-result','p-result'},'all three actual completed routes required')
        reference,reference_pin=f.doc(a.normal_reference);ref=reference['native_safe'];f.need(reference['controller']['state']=='SUCCEEDED' and ref['EOF'] and ref['input_ids_SHA256']=='10932132ad309cbfd30ae5600e64d7f7bb2dd8d7d2ddb6af4c6349ce2381ef71','actual independent cross-epoch normal EOF required');pins['normal_reference_receipt']=reference_pin
        root=f.remote_path(a.root);f.need(Path(root).parent==Path(f.CONTROL),'owned output root')
        plan=dict(candidate_commit='fb09988a5c552f4f5ff716472d8b50647af22d2c',root=root,
            input_SHA256='10932132ad309cbfd30ae5600e64d7f7bb2dd8d7d2ddb6af4c6349ce2381ef71',
            source=dict(path=f.CONTROL+'/headfb09988-source1/request/native/result.json',SHA256='f0baffe5eda7c9adeea906630f59b8673869017342e584a2d4e503ef783672c6'),
            normal_reference={k:ref[k] for k in ['path','SHA256']},results=results,local_pins=pins,bench=str(a.bench),model_requests=0)
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
