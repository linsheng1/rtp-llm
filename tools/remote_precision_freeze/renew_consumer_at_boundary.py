"""Plan-bound successor of the same Consumer at a verified idle model boundary.
Reuses Consumer4 and original controller. No installs, model signals or port changes.
"""
import argparse,hashlib,importlib.util,json,os,socket,subprocess,time
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,why):
 if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def same(who):
 p=Path('/proc')/str(who['PID'])
 if not p.exists():return False
 f=(p/'stat').read_text().rsplit(')',1)[1].split();return p.stat().st_uid==who['UID']==500 and f[19]==str(who['start_ticks']) and f[0] not in ('Z','X','x')
def model_gone(root,expected_sha):
 root=Path(root);need(sha(root/'plan.json')==expected_sha,'previous model plan drift');f=read(root/'model-service-final.json');s=read(root/'model-status.json');need(f['state']=='MODEL_SERVICE_EXITED' and f['actual_model_root_returncode']==0 and s['cleanup_state']=='FINISHED' and not s['remaining_owned_processes'],'previous model not completely drained')
 for row in [f['model_root']]+f.get('GPU_ranks',[]):need(not same(row),'old model epoch remains live')
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();need(os.getuid()==500 and sha(a.plan)==a.sha256,'owned immutable handoff plan');v=read(a.plan);need(sha(Path(__file__))==v['script']['SHA256'],'handoff source drift');need(socket.gethostbyname(socket.gethostname())==v['host'],'exact role host');helper=Path(v['helper']['path']);need(sha(helper)==v['helper']['SHA256'],'Consumer4 source drift');old=Path(v['old_root']);new=Path(v['new_root']);need(old.parent==new.parent==C and not new.exists(),'new Consumer root required');model_gone(v['previous_model_root'],v['previous_model_plan_SHA256']);need(sha(old/'ready.json')==v['old_ready_SHA256'],'old immutable READY drift');ready=read(old/'ready.json');status=read(old/'status.json');need(ready['child']==status['child']==v['old_child'],'old Consumer epoch drift')
 for root in v['must_have_no_model']:
  q=Path(root)/'model-service-status.json'
  if q.exists():
   s=read(q);need(s.get('model_root') is None and s['state'] in ('BLOCKED','FAILED'),'unexpected successor model already active')
 r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=8);need(r.returncode==0 and not r.stdout.strip(),'GPU occupancy is not empty at Consumer handoff')
 need(status['state']=='READY' and status['REQUEST_BLOCKED'] is False and 0<=time.time()-status['heartbeat_at']<=45 and same(v['old_child']),'old Consumer not current healthy owned epoch')
 controller=read(Path(v['old_controller']));need(controller['state']=='RUNNING' and same(controller['child']) and 0<=time.time()-controller['heartbeat_at']<=45,'old Consumer helper/controller not live')
 argv=[x.decode() for x in (Path('/proc')/str(controller['child']['PID'])/'cmdline').read_bytes().split(b'\0') if x];need(argv==v['old_argv'],'old helper argv differs')
 spec=importlib.util.spec_from_file_location('exact_existing_Consumer4',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 def option(flag):return v['new_argv'][v['new_argv'].index(flag)+1]
 manifest=h.read_json(Path(option('--manifest')),option('--manifest-sha256'));members=h.manifest_contract(manifest);closure=Path(option('--closure'))
 for name,pin in members.items():need(h.fingerprint(closure/name)=={k:pin[k] for k in ['bytes','SHA256']},'same-node Consumer closure drift: '+name)
 h.spec_contract(h.read_json(Path(option('--old-spec')),option('--old-spec-sha256')));need(sha(C/'consumer_peer_observation_v2.py')==h.OBSERVER_SHA,'peer observer source drift')
 result=subprocess.run(v['new_argv']+['--help'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=15);need(result.returncode==0,'actual Consumer CLI failed')
 if a.check:print(json.dumps(dict(state='IDLE_BOUNDARY_AND_ACTUAL_CLI_PASS',actual_RPC=0)));return
 root=Path(v['output_root']);need(root.parent==C and not root.exists(),'new handoff evidence root');root.mkdir(mode=0o700)
 def save(**kw):
  s=dict(schema='existing-consumer-idle-handoff-v1',heartbeat_at=time.time(),model_requests=0,model_operations=0,**kw);tmp=root/'status.partial';tmp.write_text(json.dumps(s,indent=2));tmp.replace(root/'status.json')
 try:
  need(not (old/'stop').exists(),'old stop already claimed');fd=os.open(old/'stop',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd);until=time.monotonic()+120
  while True:
   status=read(old/'status.json');outer=read(Path(v['old_controller']));save(state='DRAINING_ONLY_OWN_IDLE_CONSUMER',old_state=status['state'],old_child=v['old_child'],outer_state=outer['state'])
   if status['state']=='EXITED' and status['exit_code']==0 and status['actual_child_returncode'] in (0,-15) and not same(v['old_child']) and outer['state']=='SUCCEEDED' and outer['command_returncode']==0 and not same(controller['child']):break
   need(status['state'] not in ('FAILED','BLOCKED') and outer['state'] not in ('FAILED','BLOCKED'),'old Consumer orderly drain failed');need(time.monotonic()<until,'original120s own Consumer drain exceeded');time.sleep(20)
  model_gone(v['previous_model_root'],v['previous_model_plan_SHA256'])
  for port in (16864,16865,16866,16867):
   with socket.socket() as s:s.bind((v['host'],port))
  save(state='OLD_CONSUMER_DRAINED_STARTING_SAME_BINARY_SUCCESSOR',old_actual_child_returncode=status['actual_child_returncode'],new_root=str(new),new_consumer_launches=1,old_files_preserved=True)
  os.execve(v['new_argv'][0],v['new_argv'],dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3'))
 except BaseException as e:save(state='BLOCKED',error_kind=type(e).__name__,error=str(e),finished_at=time.time());raise
if __name__=='__main__':main()
