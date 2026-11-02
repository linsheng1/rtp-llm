"""Reuse the existing source-overlay routine and accepted dependencies.
Create only a new immutable Python candidate view; never build/install a wheel.
"""
import argparse,hashlib,importlib.util,json,os,socket
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
def read(pin):
    p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and sha(p)==pin['SHA256'],'input pin differs');return json.loads(p.read_text())
def save(path,v):
    with path.open('x') as out:json.dump(v,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();need(sha(a.plan)==a.sha256,'overlay plan SHA differs');plan=json.loads(a.plan.read_text());need(os.getuid()==500 and socket.gethostbyname(socket.gethostname())==plan['host'],'actual host/UID differs')
    base=read(plan['base_manifest']);env=read(plan['base_environment']);need(base['source_commit']==plan['base_commit']=='61887b83eb180619032a46f0f2dc0eb86248ee09' and base['Cpp_diff_empty'] and base['inner_only'] and not base['immutable_wheel'],'accepted base identity differs')
    hp=Path(plan['overlay_helper']['path']);need(sha(hp)==plan['overlay_helper']['SHA256'],'existing overlay routine differs');s=importlib.util.spec_from_file_location('existing_immutable_source_overlay',hp);helper=importlib.util.module_from_spec(s);s.loader.exec_module(helper)
    patch=Path(plan['patch']['path']);need(sha(patch)==plan['patch']['SHA256'],'candidate source differs');rel=plan['relative_path'];oldsite=Path(env['site']);need(helper.pin(oldsite/rel)==base['production_Python'][rel],'old production source drift')
    root=Path(plan['root']);need(root.parent==C and not root.exists(),'new candidate root required');root.mkdir(mode=0o700);site=root/'site';helper.overlay(oldsite,site,{rel:patch.read_bytes()})
    for name,wanted in base['production_Python'].items():
        if name!=rel:need(helper.pin(site/name)==wanted,'inherited source drift: '+name)
    e=json.loads(json.dumps(env));e.update(site=str(site),source_commit=plan['candidate_commit']);e['environment']['PYTHONPATH']=str(site)+':'+env['environment']['PYTHONPATH'].split(':',1)[1];e['environment']['LD_LIBRARY_PATH']=env['environment']['LD_LIBRARY_PATH'].replace(str(oldsite/'rtp_llm/libs'),str(site/'rtp_llm/libs'))
    ep=root/'environment.json';save(ep,e)
    m=json.loads(json.dumps(base));m.update(source_commit=plan['candidate_commit'],environment=dict(path=str(ep),**helper.pin(ep)),state='C3_DG_BATCH_INVARIANT_AD62_INNER_PREPARED_NOT_IMPORTED',engine_import='NOT_RUN_FOR_NEW_CANDIDATE',models_started=False,model_requests=0,build_invoked=False,STATE_projection_backend='SM100 DeepGEMM deterministic BF16-to-FP32',parent_STATE_candidate=dict(path=plan['base_manifest']['path'],SHA256=plan['base_manifest']['SHA256']),candidate_source_receipt=plan['source_receipt'])
    m['production_Python'][rel]=helper.pin(site/rel);m['production_changed_paths']=sorted(set(m.get('production_changed_paths',[]))|{rel});m['changed_members'][rel]=dict(path=str(site/rel),**helper.pin(site/rel),Git_blob=plan['Git_blob']);m['STATE_projection_delta']={rel:helper.pin(site/rel)}
    mp=root/'manifest.json';save(mp,m);save(root/'prepare-result.json',dict(state='IMMUTABLE_PYTHON_CANDIDATE_OVERLAY_CREATED_NOT_IMPORTED',host=plan['host'],candidate_commit=plan['candidate_commit'],environment=dict(path=str(ep),SHA256=sha(ep)),manifest=dict(path=str(mp),SHA256=sha(mp)),model_requests=0,wheel_built=False,native_rebuilt=False))
    print((root/'prepare-result.json').read_text())
if __name__=='__main__':main()
