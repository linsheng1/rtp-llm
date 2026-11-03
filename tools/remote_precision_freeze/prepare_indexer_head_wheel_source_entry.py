"""Freeze the passing candidate, reuse original source-only prep and controller."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command',choices=['freeze','check','render','submit'])
    p.add_argument('--recipe',type=Path);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256');p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command=='freeze':
        r,rp=f.doc(a.recipe);gate,gp=f.doc(r['comparison_receipt']);v=gate['result']
        f.need(gate['controller']['state']==gate['status']['state']=='SUCCEEDED' and
               v['candidate_commit']==r['candidate_commit']=='fb09988a5c552f4f5ff716472d8b50647af22d2c' and
               all(v['outputs_equal'].values()) and v['cross_epoch_normal_complete_IDs_equal'] and
               v['actual_generation_profiles_equal'],'actual three routes and regression required')
        files=dict(entry=f.read(__file__)[1],script=f.read(f.BASE/'device/prepare_indexer_head_wheel_source.py')[1],recipe=rp,comparison=gp)
        root=f.CONTROL+'/headfb09988-wheel-source-v1'
        plan=dict(candidate_commit=r['candidate_commit'],bench=r['bench'],local_pins=files,
                  script=dict(path=f.CONTROL+'/prepare_indexer_head_wheel_source.py',SHA256=files['script']['SHA256']),
                  gate_result_SHA256=gate['status']['result_SHA256'],combination=r['combination'],model_requests=0)
        plan['job']=dict(host=f.HOSTS['receiver'],name='headfb09988-wheel-source-v1',timeout=330,
            argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--root',root,
                  '--old-source',r['old_source'],'--new-source',r['new_source'],'--expected-old-head',r['old_head']],
            verify=[dict(path='headfb09988-wheel-source-v1/source-result.json',json_fields=dict(
                state='SOURCE_PREPARED_NOT_BUILT',actual_returncode=0,source_commit=r['candidate_commit']))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'source plan drift')
    for pin in plan['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'frozen source/gate drift')
    job=plan['job']
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    sys.path.insert(0,plan['bench']);from remote_control import remote
    from cluster23_dispatch import submit
    assets={plan['script']['path']:Path(plan['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()}
    code='''import os,json,subprocess
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source differs'
 else:
  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())
r=subprocess.run(%r+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(actual_CLI='PASS',model_requests=0)))'''%(assets,job['argv'][:4])
    cpu=remote('cluster3',job['host'],code,60)
    launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''})
    f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch));print(json.dumps(dict(state='SOURCE_ONLY_SUBMITTED_NOT_BUILT',model_requests=0)))
if __name__=='__main__':main()
