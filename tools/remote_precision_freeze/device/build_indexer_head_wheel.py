"""C3's existing Bazel configuration -> one full wheel, original controller.

Reuses existing resource/process, SDK, RECORD/ELF/source and BEP verifiers.
No runtime mutation, model launch, dependency install or cross-cluster bytes.
"""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,platform,re,shutil,signal,subprocess,time
from pathlib import Path
from urllib.parse import unquote,urlparse
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
TARGET='//rtp_llm:rtp_llm_cp310_aarch64_whl'
CANDIDATE='fb09988a5c552f4f5ff716472d8b50647af22d2c'
def need(ok,s):
    if not ok:raise ValueError(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(pin):
    need(sha(pin['path'])==pin['SHA256'],'input pin drift '+pin['path']);return json.loads(Path(pin['path']).read_text())
def module(pin,name):
    need(sha(pin['path'])==pin['SHA256'],'helper drift '+name);s=importlib.util.spec_from_file_location(name,pin['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def save(p,d):
    tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(d,indent=2,sort_keys=True));tmp.replace(p)
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--mode',choices=['prepare','build'],required=True);a=p.parse_args();os.umask(0o077);need((C/'.owner').read_text().strip()=='publication-precision-20261101','task owner');v=read(dict(path=str(a.plan),SHA256=a.sha256))
    need(os.getuid()==500 and platform.machine()=='aarch64' and platform.libc_ver()[1]=='2.32' and v['mode']==a.mode and sha(__file__)==v['script']['SHA256'],'actual source/ABI/UID/mode')
    h=module(v['cache_helper'],'unchanged_C3_cache_checks');b=module(v['resource_helper'],'unchanged_C3_resource_checks');w=module(v['wheel_helper'],'unchanged_wheel_RECORD_ELF_verifier');w.UID=500
    w.RUNTIME=['rtp_llm/models_py/modules/dsv41/fp8/attention_v41.py','rtp_llm/models_py/modules/dsv41/fp8/_v41_indexer_head_projection.py','rtp_llm/models_py/modules/dsv41/hc/delayed.py']
    root=Path(v['root']);need(root.parent==C,'owned direct control root');source=Path(v['source']);old=read(v['original_inputs']);cache=read(v['cache_inputs']);need(old['commit']=='c1b4b19d82f56507f89be86dddf90497501d32a6' and cache['source_commit']=='b1d8f4e984ddfc7df83a796f44752d50923fdb50' and cache['original_inputs']['SHA256']==v['original_inputs']['SHA256'],'original C3 native configuration chain')
    env=dict(cache['environment']);need(env.get('HOME')==os.environ.get('HOME') and env['CUDA_VISIBLE_DEVICES']=='','preserve actual HOME and CPU-only environment');w.env=lambda:env
    need(h.command(['git','-C',str(source),'rev-parse','HEAD']).decode().strip()==CANDIDATE and not h.command(['git','-C',str(source),'status','--porcelain','--untracked-files=all']).strip(),'actual clean candidate checkout')
    for name,tool in cache['tools'].items():need(h.pin(Path(tool['path']))=={k:tool[k] for k in ['bytes','SHA256']},'actual compiler identity changed '+name)
    for name,path in cache['overrides'].items():need(Path(path).is_dir(),'existing override missing '+name)
    need(sha(old['SDK_RPM']['path'])=='5b382202aa716a850984ce2acfb03981b05875b5b014cc8e9bc215dee22ddae4' and sha(Path(cache['SDK']['override_path'])/'file/sdk1025.rpm')==old['SDK_RPM']['SHA256'],'same original SDK RPM required')
    output=Path(cache['output_base']);need(output.is_dir() and not output.is_symlink(),'same existing real Bazel output_base');resources=b.resource_admission(h);inventory=h.process_inventory(output)
    need(not any(inventory[k] for k in ['live_compilers','mapped_mutable_outputs','live_model_launches','unknown']),'existing build output is occupied or unauditable')
    need(shutil.disk_usage(C).free>=16*1024**3,'original private build16GiB reserve')
    lock=(C/'headfb09988-wheel-cache.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.mode=='prepare':need(not root.exists(),'new wheel preparation root');root.mkdir(mode=0o700)
    else:need(root.is_dir() and root.stat().st_uid==500 and not root.is_symlink(),'existing owned preparation root')
    state=dict(state='PREPARING',candidate_commit=CANDIDATE,plan_SHA256=a.sha256,model_requests=0,heartbeat_at=time.time(),resources=resources);process=None
    def progress(**kw):state.update(kw,heartbeat_at=time.time());save(root/('prepare-status.json' if a.mode=='prepare' else 'status.json'),state)
    def child(argv,label,timeout):
        nonlocal process
        log=root/(label+'.log');bep=root/'wheel-build-events.jsonl';start=time.monotonic()
        with log.open('xb') as f:
            process=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            identity=h.process_identity(Path('/proc')/str(process.pid));save(root/(label+'-child.json'),dict(identity,argv=argv));progress(state='RUNNING_'+label.upper(),child=identity)
            semantic={}
            while process.poll() is None:
                left=timeout-(time.monotonic()-start)
                if left<=0:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    save(root/(label+'-exit.json'),dict(actual_returncode=process.returncode,timed_out=True));raise TimeoutError(label+' original finite deadline')
                try:process.wait(timeout=min(20,left))
                except subprocess.TimeoutExpired:pass
                if a.mode=='build':w.semantic_progress(root,source,semantic,process.pid)
                progress(state='RUNNING_'+label.upper() if process.returncode is None else 'VERIFYING_'+label.upper(),log_bytes=log.stat().st_size,actual_returncode=process.returncode,remaining_seconds=max(0,left),progress_correctness_not_inferred=True)
        save(root/(label+'-exit.json'),dict(actual_returncode=process.returncode,timed_out=False));need(process.returncode==0,label+' failed; inspect '+str(log));return log
    try:
        progress()
        if a.mode=='prepare':
            sdk=w.sdk_proof(Path(old['SDK_RPM']['path']),root)
            tracked=h.command(['git','-C',str(source),'ls-files','-z']).decode().split('\0');pointers=[]
            for name in tracked:
                if not name or not name.startswith(('rtp_llm/','3rdparty/','bazel/')):continue
                path=source/name
                if path.is_file():
                    with path.open('rb') as f:head=f.read(128)
                    if head.startswith(b'version https://git-lfs.github.com/spec/v1'):pointers.append(name)
            need(not pointers,'unresolved production LFS source: '+repr(pointers[:8]))
            argv=[x for x in cache['argv'][:-4] if x!='--check_up_to_date' and not x.startswith('--build_event_json_file=')]
            need(argv.count('build')==1 and '--batch' in argv and '--config=cuda13_arm' in argv and '--config=remote_kv_cache' in argv,'same actual C3 native flags')
            need(h.one_flag(argv,'--jobs')==h.one_flag(argv,'--local_cpu_resources')=='8' and h.one_flag(argv,'--local_ram_resources')=='49152','same8CPU48GiB policy')
            build=argv+['--build_event_publish_all_actions','--build_event_json_file='+str(root/'wheel-build-events.jsonl'),TARGET]
            query=[('cquery' if x=='build' else x) for x in argv]+['--output=files',TARGET]
            log=child(query,'analysis',900);need(log.stat().st_size<=32*1024**2,'bounded actual analysis log');outputs=[line.strip() for line in log.read_text().splitlines() if re.fullmatch(r'bazel-out/[^ ]+/rtp_llm/rtp_llm-[^ ]+\.whl',line.strip())]
            need(len(set(outputs))==1 and next(iter(set(outputs))).endswith('-cp310-cp310-linux_aarch64.whl'),'actual unique CP310 ARM wheel output required')
            version=next(ast.literal_eval(n.value) for n in ast.parse((source/'rtp_llm/release_version.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RELEASE_VERSION' for t in n.targets))
            prepared=dict(state='ACTUAL_C3_FULL_WHEEL_PREPARED_NOT_BUILT',source=str(source),candidate_commit=CANDIDATE,build_argv=build,environment=env,output_base=str(output),wheel_relative=outputs[0],SDK=sdk,version=version,resources=resources,inputs=dict(original=v['original_inputs'],cache=v['cache_inputs']),script=v['script'])
            save(root/'prepared.json',prepared);progress(state='SUCCEEDED',prepared=dict(path=str(root/'prepared.json'),SHA256=sha(root/'prepared.json')),actual_build_invocations=0);return
        prepared=read(v['prepared']);need(prepared['script']==v['script'] and prepared['candidate_commit']==CANDIDATE and prepared['source']==str(source),'exact complete CPU preparation required')
        for pin in prepared['SDK']['headers'].values():need(sha(pin['path'])==pin['SHA256'],'SDK header drift')
        need(sha(prepared['SDK']['post_DSO']['path'])=='eccbafbcdc651c65df5c029f8f3326e954f1dbe601ed651b27c7fc69562d25ed','SDK runtime proof drift')
        with (root/'build-once.json').open('x') as f:json.dump(dict(plan_SHA256=a.sha256,started_at=time.time(),actual_build_invocations=1),f);f.flush();os.fsync(f.fileno())
        child(prepared['build_argv'],'build',7200);events=h.parse_bep(root/'wheel-build-events.jsonl');completion=b.completed(h,0,events);need(any(e.get('id',{}).get('targetCompleted',{}).get('label')==TARGET and e.get('completed',{}).get('success') for e in events),'actual requested wheel target did not complete')
        path=source/prepared['wheel_relative'];need(path.is_file() and path.resolve().is_relative_to(output),'actual output must remain under pinned existing Bazel output_base')
        artifact=w.validate_wheel(path,source,prepared['version']);need(artifact['wheel']['bytes']<=2*1024**3,'original2GiB wheel publication budget')
        publish=root/'published';publish.mkdir(mode=0o700);dest=publish/path.name;tmp=dest.with_name(dest.name+'.partial')
        with path.open('rb') as src,tmp.open('xb') as out:shutil.copyfileobj(src,out,8*1024**2);out.flush();os.fsync(out.fileno())
        need(h.pin(tmp)==artifact['wheel'],'complete immutable copy hash');os.link(tmp,dest);tmp.unlink()
        result=dict(state='FULL_WHEEL_RECORD_SOURCE_SDK_VERIFIED_NOT_INSTALLED',candidate_commit=CANDIDATE,wheel=dict(path=str(dest),**artifact['wheel']),artifact=artifact,SDK=prepared['SDK'],BEP=completion,model_requests=0,actual_build_invocations=1,formal_model_acceptance=False,finished_at=time.time());save(publish/'manifest.json',result);progress(state='SUCCEEDED',result=dict(path=str(publish/'manifest.json'),SHA256=sha(publish/'manifest.json')),actual_build_invocations=1)
    except BaseException as e:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        progress(state='FAILED',error_kind=type(e).__name__,error=str(e),actual_returncode=None if process is None else process.returncode);save(root/'dependent-blocked.json',dict(state='BLOCKED',cause=str(e),model_requests=0));raise
    finally:lock.close()
if __name__=='__main__':main()
