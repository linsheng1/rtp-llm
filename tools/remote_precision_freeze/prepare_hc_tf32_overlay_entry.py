"""Persistent source-control-only recipe/plan preparation for the new candidate."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import freeze_stage as f

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--bench',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--role',choices=['P','D']);p.add_argument('--sha256');p.add_argument('--receipt',type=Path);a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(a.worktree),*args],text=True).strip()
    if a.command=='freeze':
        candidate='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c';f.need(git('rev-parse',candidate)==candidate,'frozen HC candidate commit missing');rel='rtp_llm/models_py/modules/dsv41/hc/delayed.py';changed=git('diff','--name-only','91b4635faf8c65cc67e66ab52a46f5c932cd08ea',candidate).splitlines();prod=[x for x in changed if x.startswith('rtp_llm/') and x.endswith('.py') and not {'test','tests','testdata'}&set(Path(x).parts)];f.need(prod==[rel] and not any(x.startswith('rtp_llm/cpp/') or Path(x).suffix in ['.cc','.cu','.h','.hpp'] for x in changed),'exact production-only delta differs')
        files={key:f.read(path)[1] for key,path in [('patch',a.worktree/rel),('script',f.BASE/'prepare_hc_tf32_overlay.py'),('entry',Path(__file__)),('overlay_helper',a.bench/'stage_c3_b1_engine_inner_v1.py')]}
        host=f.HOSTS[a.role];plan=dict(schema='existing-immutable-source-overlay-plan-v1',host=host,role=a.role,base_commit='91b4635faf8c65cc67e66ab52a46f5c932cd08ea',candidate_commit=candidate,relative_path=rel,Git_blob=git('rev-parse',candidate+':'+rel),base_manifest=dict(path=f.CONTROL+'/state-projection-91b-inner-v1/manifest.json',SHA256='1762a637a8f68d2170fe520baed8a55f2e80c7eade3fe5182666708b6d86e0d9' if a.role=='P' else '1a239cd474d8b4eba4ae776ba3b4c96ab8b7e7babf6b20ba078aaa7bc4ef3304'),base_environment=dict(path=f.CONTROL+'/state-projection-91b-inner-v1/environment.json',SHA256='c4b90d002d2325d11cf329be89a25b5d8f01e00673a7a77078e4ca3c278a6fd8'),root=f.CONTROL+'/hc-tf32-031-inner-v1',source_receipt=dict(candidate_commit=candidate,base_commit='91b4635faf8c65cc67e66ab52a46f5c932cd08ea',production_changed_paths=prod,native_changed_paths=[],Git_blob=git('rev-parse',candidate+':'+rel)),local_pins=files)
        for key in ['script','overlay_helper','patch']:plan[key]=dict(path=f.CONTROL+'/'+('hc_tf32_candidate_031.py' if key=='patch' else Path(files[key]['path']).name),SHA256=files[key]['SHA256'])
        plan['job']=dict(host=host,name='hc-tf32-overlay-'+a.role+'-v1',timeout=240,argv=['/opt/conda310/bin/python3.10','-B','-S',plan['script']['path'],'--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],verify=[dict(path='hc-tf32-031-inner-v1/prepare-result.json',json_fields=dict(state='IMMUTABLE_PYTHON_CANDIDATE_OVERLAY_CREATED_NOT_IMPORTED',candidate_commit=candidate))])
        print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0)));return
    plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'plan SHA differs')
    for v in plan['local_pins'].values():f.need(f.read(v['path'])[1]==v,'pinned source drift')
    sys.path.insert(0,str(a.bench));from remote_control import remote
    from cluster23_dispatch import submit
    job=dict(plan['job']);job['argv']=[a.sha256 if v=='${PLAN_SHA256}' else v for v in job['argv']]
    if a.command=='check':print(json.dumps(dict(state='OFFLINE_PINS_PASS',model_requests=0)));return
    if a.command=='render':print(json.dumps(job));return
    assets={plan[k]['path']:Path(plan['local_pins'][k]['path']).read_text() for k in ['script','overlay_helper','patch']};assets[f.CONTROL+'/'+a.plan.name]=a.plan.read_text()
    # Bounded small SOURCE/control chunks only. Never forward artifacts or payloads.
    for path,raw in assets.items():
        code='''import json,os\nfrom pathlib import Path\np=Path(%r);raw=%r;assert os.getuid()==500\nif p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source differs'\nelse:\n with p.open('x') as out:out.write(raw);out.flush();os.fsync(out.fileno())\nprint(json.dumps(dict(verified=True)))'''%(path,raw)
        remote('cluster3',job['host'],code,60)
    result=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.need(a.receipt is not None,'receipt required');f.save_new(a.receipt,dict(plan_SHA256=a.sha256,launch=result));print(json.dumps(dict(state='SUBMITTED_NOT_PREPARED',model_requests=0)))
if __name__=='__main__':main()
