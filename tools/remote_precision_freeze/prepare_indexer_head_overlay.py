"""Apply a bounded source diff on-device, then reuse the immutable overlay routine."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,s):
    if not ok:raise ValueError(s)
def read(pin):
    need(sha(pin['path'])==pin['SHA256'],'input pin drift');return json.loads(Path(pin['path']).read_text())
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();v=read(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and sha(__file__)==v['script']['SHA256'] and socket.gethostbyname(socket.gethostname())==v['host'],'exact source/host/UID')
    base=read(v['base_manifest']);env=read(v['base_environment']);need(base['source_commit']==env['source_commit']==v['base_commit'] and base['Cpp_diff_empty'] and base['inner_only'] and not base['immutable_wheel'],'unchanged native base identity')
    hp=Path(v['overlay_helper']['path']);need(sha(hp)==v['overlay_helper']['SHA256'],'original overlay routine');spec=importlib.util.spec_from_file_location('original_head_overlay',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    root=Path(v['root']);need(root.parent==C and not root.exists(),'new candidate root');root.mkdir(mode=0o700);oldsite=Path(env['site']);relative=v['attention_relative'];need(h.pin(oldsite/relative)==base['production_Python'][relative],'attention preimage drift')
    work=root/'source-diff';target=work/relative;target.parent.mkdir(parents=True);target.write_bytes((oldsite/relative).read_bytes());need(sha(v['patch']['path'])==v['patch']['SHA256'],'bounded source patch drift')
    for args in [['--check'],[]]:
        r=subprocess.run(['git','apply']+args+[v['patch']['path']],cwd=work,capture_output=True,text=True,timeout=15);need(r.returncode==0,'source patch failed: '+r.stderr)
    need(h.pin(target)==v['members'][relative]['fingerprint'],'patched source differs from committed candidate');kernel_relative=v['kernel_relative'];kernel=Path(v['kernel']['path']);need(h.pin(kernel)==v['members'][kernel_relative]['fingerprint'],'committed kernel fingerprint');compile(target.read_text(),relative,'exec');compile(kernel.read_text(),kernel_relative,'exec')
    site=root/'site';h.overlay(oldsite,site,{relative:target.read_bytes()});new=site/kernel_relative;need(not new.exists() and new.parent.is_dir() and not new.parent.is_symlink(),'new kernel real package parent')
    with new.open('xb') as f:f.write(kernel.read_bytes());f.flush();os.fsync(f.fileno())
    for name,pin in base['production_Python'].items():
        if name!=relative:need(h.pin(site/name)==pin,'inherited production source drift '+name)
    e=json.loads(json.dumps(env));e.update(site=str(site),source_commit=v['candidate_commit']);e['environment']['PYTHONPATH']=str(site)+':'+env['environment']['PYTHONPATH'].split(':',1)[1];e['environment']['LD_LIBRARY_PATH']=env['environment']['LD_LIBRARY_PATH'].replace(str(oldsite/'rtp_llm/libs'),str(site/'rtp_llm/libs'));ep=root/'environment.json';save(ep,e)
    m=json.loads(json.dumps(base));m.update(source_commit=v['candidate_commit'],environment=dict(path=str(ep),**h.pin(ep)),engine_import='NOT_RUN_FOR_NEW_CANDIDATE',models_started=False,model_requests=0,build_invoked=False,parent_indexer_candidate=v['base_manifest'],indexer_head_source_receipt=v['members'],indexer_head_fix_unit=v['unit_result'])
    for name in [relative,kernel_relative]:
        m['production_Python'][name]=h.pin(site/name);m['changed_members'][name]=dict(path=str(site/name),**h.pin(site/name),Git_blob=v['members'][name]['Git_blob'])
    m['production_changed_paths']=sorted(set(m['production_changed_paths'])|{relative,kernel_relative});mp=root/'manifest.json';save(mp,m)
    result=dict(state='IMMUTABLE_PYTHON_CANDIDATE_OVERLAY_CREATED_NOT_IMPORTED',host=v['host'],candidate_commit=v['candidate_commit'],environment=dict(path=str(ep),SHA256=sha(ep)),manifest=dict(path=str(mp),SHA256=sha(mp)),model_requests=0,wheel_built=False,native_rebuilt=False);save(root/'prepare-result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
