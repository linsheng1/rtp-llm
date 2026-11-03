"""Install the frozen C3 wheel into a new private view, without changing services."""
import argparse,hashlib,json,os,shutil,socket,subprocess,time,zipfile
from pathlib import Path,PurePosixPath
from urllib.request import build_opener,ProxyHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,s):
 if not ok:raise ValueError(s)
def pin(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(bytes=Path(p).stat().st_size,SHA256=h.hexdigest())
def save(p,v):
 q=p.with_name(p.name+'.partial');q.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');q.replace(p)
def read(p):
 need(pin(p['path'])['SHA256']==p['SHA256'],'input drift '+p['path']);return json.loads(Path(p['path']).read_text())
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();v=read(dict(path=str(a.plan),SHA256=a.sha256))
 need(os.getuid()==500 and socket.gethostbyname(socket.gethostname())==v['host'] and pin(__file__)['SHA256']==v['script']['SHA256'],'host/UID/script')
 root=Path(v['root']);need(root.parent==C and not root.exists(),'new private install only');root.mkdir(mode=0o700);state=dict(state='RUNNING',phase='device_direct_download',model_requests=0,plan_SHA256=a.sha256,heartbeat_at=time.time())
 def progress(**kw):state.update(kw,heartbeat_at=time.time());save(root/'status.json',state)
 try:
  progress();need(shutil.disk_usage(root).free>8*1024**3,'8GiB installation reserve')
  opener=build_opener(ProxyHandler({}))
  for name,expected in [('manifest.json',v['publication']),('candidate.whl',v['wheel'])]:
   dest=root/name;received=0;last=time.monotonic()
   with opener.open(expected['URL'],timeout=60) as src,dest.open('xb') as out:
    while True:
     b=src.read(8*1024**2)
     if not b:break
     out.write(b);received+=len(b);need(received<=expected['bytes'],'download exceeds frozen size')
     if time.monotonic()-last>20:progress(received_bytes=received);last=time.monotonic()
    out.flush();os.fsync(out.fileno())
   need(pin(dest)=={k:expected[k] for k in ['bytes','SHA256']},'published complete identity '+name)
  pub=json.loads((root/'manifest.json').read_text());need(pub['wheel']['SHA256']==v['wheel']['SHA256'] and pub['candidate_commit']==v['candidate_commit'],'publication candidate')
  base=read(v['parent_manifest']);env=read(base['environment']);site=root/'site';site.mkdir();members=pub['artifact']['members'];progress(phase='verify_extract')
  with zipfile.ZipFile(root/'candidate.whl') as z:
   names=[i.filename for i in z.infolist() if not i.is_dir()];need(len(names)==len(set(names)) and set(names)==set(members),'exact complete member inventory')
   for n in names:
    rel=PurePosixPath(n);need(not rel.is_absolute() and '..' not in rel.parts,'safe wheel member');info=z.getinfo(n);need((info.external_attr>>16)&0o170000!=0o120000,'no wheel symlinks');dest=site/n;dest.parent.mkdir(parents=True,exist_ok=True)
    with z.open(n) as src,dest.open('xb') as out:shutil.copyfileobj(src,out,8*1024**2)
    need(pin(dest)==members[n],'extracted member differs '+n)
  oldprod=base['production_Python'];newprod=pub['artifact']['production_Python'];newprod={n:members[n] for n in newprod}
  delta=dict(only_parent=sorted(set(oldprod)-set(newprod)),only_wheel=sorted(set(newprod)-set(oldprod)),changed=[n for n in sorted(set(oldprod)&set(newprod)) if {k:oldprod[n][k] for k in ['bytes','SHA256']}!=newprod[n]])
  e=json.loads(json.dumps(env));oldsite=e['site'];e['site']=str(site);e['environment']['PYTHONPATH']=str(site)+':'+e['environment']['PYTHONPATH'].split(':',1)[1];e['environment']['LD_LIBRARY_PATH']=e['environment']['LD_LIBRARY_PATH'].replace(oldsite+'/rtp_llm/libs',str(site/'rtp_llm/libs'));e['immutable_wheel_SHA256']=v['wheel']['SHA256'];ep=root/'environment.json';save(ep,e)
  m=json.loads(json.dumps(base));m.update(state='FULL_WHEEL_INSTALLED_CPU_PENDING',inner_only=False,immutable_wheel=True,wheel=v['wheel'],parent_inner_manifest=v['parent_manifest'],production_Python=newprod,environment=dict(path=str(ep),**pin(ep)),package_delta=delta)
  for key in ['native','support']:
   for n,old in base[key].items():
    need(n in members and members[n]=={k:old[k] for k in ['bytes','SHA256']},'native/generated support differs '+n);m[key][n]=dict(path=str(site/n),**members[n])
  mp=root/'runtime-manifest.json';save(mp,m);progress(phase='CPU_natural_import')
  code="import json,torch,rtp_llm;from rtp_llm.models_py.modules.dsv41.fp8 import _v41_indexer_head_projection as h;from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as p;print(json.dumps(dict(rtp=rtp_llm.__file__,head=h.__file__,torch=torch.__version__,protobuf=p.__file__)))"
  childenv=dict(os.environ);childenv.update(e['environment']);childenv['CUDA_VISIBLE_DEVICES']=''
  q=subprocess.run([e['interpreter'],'-B','-S','-c',code],env=childenv,capture_output=True,text=True,timeout=180);(root/'CPU-import.log').write_text(q.stdout+'\n'+q.stderr);need(q.returncode==0,'CPU natural import failed; inspect CPU-import.log');imports=json.loads(q.stdout.strip().splitlines()[-1]);need(all(str(site) in imports[k] for k in ['rtp','head','protobuf']),'natural imports leave installed wheel')
  result=dict(state='FULL_WHEEL_INSTALLED_CPU_PASS',candidate_commit=v['candidate_commit'],wheel=v['wheel'],publication=v['publication'],environment=dict(path=str(ep),**pin(ep)),manifest=dict(path=str(mp),**pin(mp)),package_delta=delta,CPU=dict(actual_returncode=q.returncode,imports=imports),member_count=len(members),model_requests=0,formal_model_acceptance=False)
  save(root/'result.json',result);progress(state='SUCCEEDED',phase='terminal',result=dict(path=str(root/'result.json'),**pin(root/'result.json')))
 except BaseException as ex:progress(state='FAILED',phase='terminal',error=type(ex).__name__+': '+str(ex));raise
if __name__=='__main__':main()
