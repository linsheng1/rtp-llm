"""Re-establish static weight identity after a lease gap, by FULL content comparison.
No CUDA, model, KV, writes to weights, unlink or implicit adoption of old READY.
The original device controller owns lifetime/failure supervision.
"""
import argparse,fcntl,hashlib,json,os,re,signal,socket,stat,struct,threading,time
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');STOP=threading.Event();CHUNK=8*1024**2

def need(ok,why):
 if not ok:raise ValueError(why)
def pin(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
 return dict(bytes=path.stat().st_size,SHA256=h.hexdigest())
def read(p):
 path=Path(p['path']);need(path.is_file() and not path.is_symlink() and path.stat().st_size<=4*1024**2 and pin(path)['SHA256']==p['SHA256'],'metadata identity drift: '+str(path));return json.loads(path.read_text())
def save(path,v):
 t=path.with_name(path.name+'.partial');t.write_text(json.dumps(v,indent=2,sort_keys=True));os.replace(t,path)
def inputs(environment,manifest):
 e=read(environment);m=read(manifest);need(e['source_commit']==m['source_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c','candidate identity')
 loader=Path(e['site'])/'rtp_llm/models_py/modules/dsv41/engram.py';need(pin(loader)==m['production_Python']['rtp_llm/models_py/modules/dsv41/engram.py'],'unchanged checkpoint loader')
 root=Path('/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash');index=root/'model.safetensors.index.json';need(pin(index)['SHA256']=='74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8','original checkpoint index');wm=json.loads(index.read_text())['weight_map'];keys=sorted(k[:-len('embed.weight')] for k in wm if k.endswith('.engram.embed.weight'));need(keys==['layers.1.engram.','layers.14.engram.'],'exact two source tables')
 tables=[]
 for prefix in keys:
  identity=[str(root.resolve()),prefix];parts=[]
  for suffix,dtype in [('embed.weight','F8_E4M3'),('embed.scale','F8_E8M0')]:
   key=prefix+suffix;p=root/wm[key];st=p.stat();identity.extend([str(p.resolve()),str(st.st_size),str(st.st_mtime_ns)])
   with p.open('rb') as f:
    length=struct.unpack('<Q',f.read(8))[0];need(length<16*1024**2,'bounded safetensors header');header=f.read(length);entry=json.loads(header)[key]
   start,end=entry['data_offsets'];elements=1
   for n in entry['shape']:elements*=n
   need(entry['dtype']==dtype and elements==end-start and 8+length+end<=st.st_size,'one-byte raw source tensor identity/length')
   parts.append(dict(key=key,path=str(p),header_SHA256=hashlib.sha256(header).hexdigest(),header_bytes=length+8,start=start,end=end,shape=entry['shape'],dtype=dtype,source_stat=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,device=st.st_dev,inode=st.st_ino)))
  source_identity='|'.join(identity);path=Path('/dev/shm')/('rtp_llm_engram_500_'+hashlib.sha256(source_identity.encode()).hexdigest()[:24]);st=path.lstat();size=sum(p['end']-p['start'] for p in parts);need(stat.S_ISREG(st.st_mode) and not path.is_symlink() and st.st_uid==500 and st.st_size==size,'existing owned full-size table')
  tables.append(dict(path=str(path),source_identity=source_identity,device=st.st_dev,inode=st.st_ino,UID=st.st_uid,bytes=st.st_size,parts=parts))
 need(sorted(t['bytes'] for t in tables)==[101377628352,101380404048],'original table sizes')
 return dict(environment=environment,manifest=manifest,loader=dict(path=str(loader),resolved_path=str(loader.resolve()),**pin(loader)),index=dict(path=str(index),**pin(index)),tables=tables)

def audit(a,controller,required_seconds=None):
 """Model admission: fresh live keeper + original strict FD/flock/epoch checks."""
 r=read(dict(path=str(a.keeper_root/'ready.json'),SHA256=pin(a.keeper_root/'ready.json')['SHA256']));s=json.loads((a.keeper_root/'status.json').read_text());need(r['schema']==s['schema']=='full-content-revalidated-static-weights-v1' and r['state']=='FULL_CONTENT_VERIFIED_STATIC_WEIGHT_LEASES_HELD' and s['state']=='RUNNING','full-content keeper not current READY');need(r['role']==s['role']==a.role and r['host']==s['host']==a.host,'keeper role/host')
 need(pin(a.keeper_helper)['SHA256']==a.keeper_helper_sha256==r['helper_SHA256']==s['helper_SHA256'],'keeper source SHA');actual=controller.identity(r['process']['PID']);need(controller.same(r['process'],actual) and controller.same(r['process'],s['process']) and actual['UID']==500 and actual['state'] not in ('Z','X','x') and str(a.keeper_helper) in actual['argv'],'keeper actual PID/UID/start/source')
 need(0<=time.time()-s['heartbeat_at']<=45 and r['deadline_epoch']==s['deadline_epoch'] and r['deadline_epoch']>time.time()+(a.seconds+15 if required_seconds is None else required_seconds),'keeper heartbeat/whole-model-and-drain lease')
 need(r['inputs']==s['inputs'] and r['files']==s['files'] and r['full_weight_content_hashed'] is s['full_weight_content_hashed'] is True and r['CUDA_initialized'] is s['CUDA_initialized'] is False and r['KV_cache_held'] is s['KV_cache_held'] is False and r['files_unlinked'] is s['files_unlinked'] is False,'full-content static-only proof')
 need(r['inputs']['environment']==dict(path=str(a.env_json),SHA256=a.env_sha256) and r['inputs']['manifest']==dict(path=str(a.manifest),SHA256=a.manifest_sha256),'current candidate binding')
 need(inputs(r['inputs']['environment'],r['inputs']['manifest'])==r['inputs'],'checkpoint/loader/table identity changed after full-content verification');locks=Path('/proc/locks').read_text().splitlines();audited=[]
 for row in r['files']:
  need(row['actual_SHA256']==row['checkpoint_SHA256'] and row['verified_bytes']==row['bytes'] and row['SH_data_lease_held'] is True,'complete matching byte hash required');path=Path(row['path']);p=path.lstat();fd=Path('/proc')/str(actual['PID'])/'fd'/str(row['FD']);f=fd.stat();need(stat.S_ISREG(p.st_mode) and stat.S_ISREG(f.st_mode) and all(row[k]==v for k,v in [('device',p.st_dev),('inode',p.st_ino),('bytes',p.st_size),('UID',p.st_uid)]) and (p.st_dev,p.st_ino,p.st_size,p.st_uid)==(f.st_dev,f.st_ino,f.st_size,f.st_uid) and p.st_uid==500,'live read-only table FD identity');flags=re.search(r'^flags:\s*([0-7]+)$',fd.with_name(str(row['FD'])).parent.parent.joinpath('fdinfo',str(row['FD'])).read_text(),re.M);need(flags and int(flags[1],8)&os.O_ACCMODE==os.O_RDONLY,'read-only FD required')
  held=False
  for line in locks:
   fields=line.split()
   if len(fields)>=6 and fields[1:4]==['FLOCK','ADVISORY','READ'] and fields[4]==str(actual['PID']):
    x=fields[5].split(':')
    if len(x)==3 and (int(x[0],16),int(x[1],16),int(x[2]))==(os.major(p.st_dev),os.minor(p.st_dev),p.st_ino):held=True
  need(held,'actual keeper PID READ FLOCK absent');audited.append(row)
 fs=os.statvfs('/dev/shm');free=fs.f_bavail*fs.f_frsize;need(free>=4*1024**3,'original retained freeSHM4GiB guard');return dict(mode='retained_two_FULL_CONTENT_revalidated_static_weight_FDs',process=actual,ready=dict(path=str(a.keeper_root/'ready.json'),**pin(a.keeper_root/'ready.json')),tables=audited,full_weight_hash_accepted=True,KV_cache_reused=False,free_bytes=free,minimum_bytes=4*1024**3)

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();plan=read(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only owned process');need(pin(Path(__file__))['SHA256']==plan['script']['SHA256'],'script pin');observed=inputs(plan['environment'],plan['manifest']);need(observed==plan['inputs'] and socket.gethostbyname(socket.gethostname())==plan['host'],'actual prepare inputs/host')
 if a.check:print(json.dumps(dict(state='STATIC_WEIGHT_FULL_CONTENT_PREPARE_PASS',model_requests=0)));return
 root=Path(plan['output_root']);need(root.parent==C and not root.exists() and 7200<=plan['seconds']<=14400,'new finite keeper');root.mkdir(mode=0o700);fds=[];deadline=time.time()+plan['seconds'];fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();process=dict(PID=os.getpid(),UID=500,start_ticks=fields[19]);state=dict(schema='full-content-revalidated-static-weights-v1',state='VERIFYING',role=plan['role'],host=plan['host'],process=process,helper_SHA256=plan['script']['SHA256'],plan_SHA256=a.sha256,deadline_epoch=deadline,heartbeat_at=time.time(),inputs=observed,full_weight_content_hashed=False,CUDA_initialized=False,KV_cache_held=False,model_requests=0,model_started=False,files_unlinked=False,files=[],total_verified_bytes=0)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:STOP.set())
 try:
  # Hold BOTH data SH leases before reading any content, without touching old proofs.
  for table in observed['tables']:
   fd=os.open(table['path'],os.O_RDONLY|os.O_NOFOLLOW);fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB);fds.append(fd);st=os.fstat(fd);need((st.st_dev,st.st_ino,st.st_size,st.st_uid)==(table['device'],table['inode'],table['bytes'],500),'opened table identity changed')
  save(root/'status.json',state);last=time.monotonic()
  for table,fd in zip(observed['tables'],fds):
   expected=hashlib.sha256();actual=hashlib.sha256();offset=0
   for part in table['parts']:
    with Path(part['path']).open('rb') as src:
     src.seek(part['header_bytes']+part['start']);remaining=part['end']-part['start']
     while remaining:
      need(not STOP.is_set() and time.time()<deadline,'cancelled or finite verification deadline');n=min(CHUNK,remaining);x=src.read(n);y=os.pread(fd,n,offset);need(len(x)==len(y)==n,'short full-content source/table read');expected.update(x);actual.update(y);remaining-=n;offset+=n;state['total_verified_bytes']+=n
      if time.monotonic()-last>=20:state.update(heartbeat_at=time.time(),progress=dict(table=table['path'],verified_bytes=offset,source_plus_table_bytes_read=state['total_verified_bytes']*2));save(root/'status.json',state);last=time.monotonic()
   need(expected.hexdigest()==actual.hexdigest(),'FULL CONTENT MISMATCH; dependent model BLOCKED');state['files'].append({k:table[k] for k in ['path','device','inode','UID','bytes']}|dict(FD=fd,SH_data_lease_held=True,verified_bytes=offset,checkpoint_SHA256=expected.hexdigest(),actual_SHA256=actual.hexdigest()))
  need(inputs(plan['environment'],plan['manifest'])==observed,'source changed during full verification');state.update(state='RUNNING',full_weight_content_hashed=True,heartbeat_at=time.time(),verified_at=time.time());ready=dict(state,state='FULL_CONTENT_VERIFIED_STATIC_WEIGHT_LEASES_HELD');save(root/'ready.json',ready);save(root/'status.json',state)
  while not STOP.wait(20):
   if time.time()>=deadline:break
   state.update(heartbeat_at=time.time(),remaining_seconds=deadline-time.time(),deadline_warning=deadline-time.time()<900);save(root/'status.json',state)
  state.update(state='EXITED',returncode=0,reason='bounded lease ended' if time.time()>=deadline else 'own controller requested stop',heartbeat_at=time.time());save(root/'status.json',state)
 except BaseException as e:state.update(state='FAILED',returncode=1,error_kind=type(e).__name__,error=str(e),heartbeat_at=time.time());save(root/'status.json',state);raise
 finally:
  for fd in fds:os.close(fd)
if __name__=='__main__':main()
