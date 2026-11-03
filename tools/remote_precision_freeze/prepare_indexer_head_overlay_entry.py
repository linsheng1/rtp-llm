"""Persistent source-control-only recipe/plan preparation for the new candidate."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--recipe',type=Path);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--bench',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--role',choices=['P','D']);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(a.worktree),*args],text=True).strip()
    if a.command=='freeze':
        recipe,rp=f.doc(a.recipe);candidate=recipe['candidate_commit'];base='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c';f.need(git('rev-parse',candidate)==candidate and recipe['flow']=='indexer-head-source-overlay','actual committed candidate recipe required')
        rel='rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py';kernel='rtp_llm/models_py/modules/dsv41/fp8/_v41_indexer_head_projection.py';changed=git('diff','--name-only',base,candidate).splitlines();prod=[x for x in changed if x.startswith('rtp_llm/') and x.endswith('.py') and not {'test','tests','testdata'}&set(Path(x).parts)];f.need(set(prod)=={rel,kernel} and not any(x.startswith('rtp_llm/cpp/') or Path(x).suffix in ['.cc','.cu','.h','.hpp'] for x in changed),'only proven head projection production delta allowed')
        patch=a.plan.parent/'indexer-head-attention-source.patch'
        raw=subprocess.check_output(['git','-C',str(a.worktree),'diff',base,candidate,'--',rel]);f.need(len(raw)<16384,'bounded source patch only')
        if patch.exists():f.need(patch.read_bytes()==raw,'immutable patch differs')
        else:patch.open('xb').write(raw)
        unit,up=f.doc(recipe['unit_receipt']);f.need(unit['controller']['state']==unit['status']['state']=='SUCCEEDED' and unit['result']['fixed_both_shapes_match_FP64'] and unit['result']['graph_changed_input_reference_pass'],'actual GPU fix proof required')
        files={key:f.read(path)[1] for key,path in [('patch',patch),('kernel',a.worktree/kernel),('script',f.BASE/'prepare_indexer_head_overlay.py'),('entry',Path(__file__)),('overlay_helper',a.bench/'stage_c3_b1_engine_inner_v1.py')]};files.update(recipe=rp,unit_receipt=up)
        members={name:dict(fingerprint={k:f.read(a.worktree/name)[1][k] for k in ['bytes','SHA256']},Git_blob=git('rev-parse',candidate+':'+name)) for name in [rel,kernel]}
        f.need(members[kernel]['fingerprint']['SHA256']==unit['result']['kernel']['SHA256'],'tested committed kernel identity')
        host=f.HOSTS[a.role];plan=dict(schema='existing-immutable-source-overlay-plan-v1',host=host,role=a.role,base_commit=base,candidate_commit=candidate,attention_relative=rel,kernel_relative=kernel,members=members,base_manifest=recipe['roles'][a.role]['base_manifest'],base_environment=recipe['base_environment'],root=f.CONTROL+'/indexer-head-fb09988-inner-v1',unit_result=unit['status']['result'],local_pins=files)
        for key in ['script','overlay_helper','patch','kernel']:plan[key]=dict(path=f.CONTROL+'/'+Path(files[key]['path']).name,SHA256=files[key]['SHA256'])
        plan['job']=dict(host=host,name='indexer-head-overlay-'+a.role+'-v1',timeout=240,argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path='indexer-head-fb09988-inner-v1/prepare-result.json',json_fields=dict(state='IMMUTABLE_PYTHON_CANDIDATE_OVERLAY_CREATED_NOT_IMPORTED',candidate_commit=candidate))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'plan SHA differs')
    for v in plan['local_pins'].values():f.need(f.read(v['path'])[1]==v,'pinned source drift')
    sys.path.insert(0,str(a.bench));from remote_control import remote
    from cluster23_dispatch import submit
    job=dict(plan['job']);job['argv']=[a.sha256 if v=='${PLAN_SHA256}' else v for v in job['argv']]
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    assets={plan[k]['path']:Path(plan['local_pins'][k]['path']).read_text() for k in ['script','overlay_helper','patch','kernel']};assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text()
    # Bounded small SOURCE/control chunks only. Never forward artifacts or payloads.
    for path,raw in assets.items():
        code='''import json,os\nfrom pathlib import Path\np=Path(%r);raw=%r;assert os.getuid()==500\nif p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source differs'\nelse:\n with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())\nprint(json.dumps(dict(verified=True)))'''%(path,raw)
        remote('cluster3',job['host'],code,60)
    result=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.need(a.receipt is not None,'receipt required');f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=result));print(json.dumps(dict(state='SUBMITTED_NOT_PREPARED',model_requests=0)))
if __name__=='__main__':main()
