"""Actual C3 source/configuration receipts -> original finite CPU controller."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--recipe',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.command=='freeze':
        r,rp=f.doc(a.recipe);gate,gp=f.doc(r['comparison_receipt']);source,sp=f.doc(r['source_receipt']);g=gate['result']
        f.need(gate['controller']['state']==gate['status']['state']=='SUCCEEDED' and all(g['outputs_equal'].values()) and g['cross_epoch_normal_complete_IDs_equal'] and g['actual_generation_profiles_equal'],'completed model/normal-regression gate')
        f.need(source['controller']['state']=='SUCCEEDED' and source['result']['state']=='SOURCE_PREPARED_NOT_BUILT' and source['result']['source_commit']==g['candidate_commit']==r['candidate_commit']=='fb09988a5c552f4f5ff716472d8b50647af22d2c' and source['result']['new_source_clean'] and source['result']['old_HEAD_and_status_bytes_unchanged'],'actual source-only preparation required')
        bench=Path(r['bench']);files=dict(entry=f.read(__file__)[1],script=f.read(f.BASE/'device/build_indexer_head_wheel.py')[1],recipe=rp,model_gate=gp,source_gate=sp)
        for key,name in [('cache_helper','check_c3_native_cache_b1_v2.py'),('resource_helper','build_c3_native_b1.py'),('wheel_helper','cluster1_rtp_wheel_controller.py')]:files[key]=f.read(bench/name)[1]
        v=dict(candidate_commit=r['candidate_commit'],mode=r['mode'],bench=str(bench),source=source['result']['new_source'],root=f.CONTROL+'/headfb09988-full-wheel-v1',original_inputs=r['original_inputs'],cache_inputs=r['cache_inputs'],local_pins=files,model_requests=0,build_budget=dict(invocations=1,seconds=7200,CPUs=8,RAM_MiB=49152,max_wheel_bytes=2147483648))
        for key in ['script','cache_helper','resource_helper','wheel_helper']:
            name=Path(files[key]['path']).name if key=='script' else 'head-wheel-'+key+'-v1.py'
            v[key]=dict(path=f.CONTROL+'/'+name,SHA256=files[key]['SHA256'])
        if r['mode']=='build':
            actual,ap=f.doc(r['prepare_receipt']);f.need(actual['controller']['state']=='SUCCEEDED' and actual['status']['state']=='SUCCEEDED' and actual['status']['actual_build_invocations']==0,'actual full prepare completed without build')
            v['prepared']=actual['status']['prepared'];v['local_pins']['prepare_receipt']=ap
        else:f.need(r['mode']=='prepare','only prepare/build')
        tag='headfb09988-wheel-'+r['mode']+'-v1';status='prepare-status.json' if r['mode']=='prepare' else 'status.json'
        v['job']=dict(host=f.HOSTS['receiver'],name=tag,timeout=1800 if r['mode']=='prepare' else 7560,
            argv=['/opt/conda310/bin/python3.10','-B','-S',v['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}','--mode',r['mode']],
            verify=[dict(path='headfb09988-full-wheel-v1/'+status,json_fields=dict(state='SUCCEEDED',model_requests=0))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,v),model_requests=0)));return
    v,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'wheel plan drift')
    for pin in v['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'frozen wheel source/evidence drift')
    job=dict(v['job'],argv=[a.sha256 if x=='${PLAN_SHA256}' else x for x in v['job']['argv']])
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    f.need(a.receipt is not None and not a.receipt.exists(),'new receipt required');sys.path.insert(0,v['bench']);from remote_control import remote
    from cluster23_dispatch import submit
    assets={v[k]['path']:Path(v['local_pins'][k]['path']).read_text() for k in ['script','cache_helper','resource_helper','wheel_helper']};assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text();groups=[];group={}
    for name,raw in assets.items():
        if len(repr(dict(group,**{name:raw})).encode())>60000:groups.append(group);group={}
        group[name]=raw
    if group:groups.append(group)
    for group in groups:
        code='''import os,json
from pathlib import Path
for name,raw in %r.items():
 p=Path(name)
 if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable control source/plan differs'
 else:
  with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())
print(json.dumps(dict(published=True)))'''%group
        remote('cluster3',job['host'],code,60)
    code="import os,json,subprocess\nr=subprocess.run(%r+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr;print(json.dumps(dict(actual_CLI='PASS',model_requests=0)))"%job['argv'][:4]
    cpu=remote('cluster3',job['host'],code,60);launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify'],env={'CUDA_VISIBLE_DEVICES':''});f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=launch));print(json.dumps(dict(state='SUBMITTED_NOT_COMPLETE',mode=v['mode'],model_requests=0)))
if __name__=='__main__':main()
