"""Verify completed own batch, request original stop; never wait via the240s helper.

Only the original model controller owns draining,420s reserve and exit status.
This finite command acknowledges a request, not model cleanup or acceptance.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
from urllib.request import urlopen
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fetch(pin):
 with urlopen(pin['URL'],timeout=15) as f:raw=f.read(1024**2+1)
 need(len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'completed stage/ack drift');return json.loads(raw)
def epoch(r):return tuple(str(r[k]) for k in ['UID','PID','start_ticks'])
def live(r):
 p=Path('/proc')/str(r['PID'])
 if not p.exists():return False
 v=(p/'stat').read_text().rsplit(')',1)[1].split();return p.stat().st_uid==r['UID']==500 and v[19]==str(r['start_ticks']) and v[0] not in ['Z','X','x']
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();need(os.getuid()==500 and sha(a.plan)==a.sha256,'exact owned plan');v=json.loads(a.plan.read_text());need(sha(__file__)==v['script']['SHA256'],'stop source drift');old=Path(v['model_root']);need(old.parent==C and sha(old/'plan.json')==v['model_plan_SHA256'],'exact owned model plan');model=json.loads((old/'plan.json').read_text());need(model['drain_reserve_seconds']==420,'unchanged original drain reserve');s=json.loads((old/'model-service-status.json').read_text());need(-2<=time.time()-s.get('heartbeat_at',s.get('finished_at',0))<=45 or s['state'] in ['MODEL_SERVICE_EXITED','FAILED'],'fresh active model status or explicit final');parent=json.loads(Path(v['parent_controller_status']).read_text())
 for proof in v['completed_stages']:
  d=fetch(proof);need(d['state']=='COMPLETE_NATIVE_RPC' and d['response_eof_complete'] and len(d['output_ids'])==proof['output_tokens'],'completed actual native EOF required');role=d['identity']['roles'][v['role']];need(epoch(role['model_root'])==epoch(s['model_root']) and {epoch(r) for r in role['GPU_ranks']}=={epoch(r) for r in s['GPU_ranks']} and len(role['GPU_ranks'])==4,'native results must bind same owned4rank epoch')
 root=Path(v['output_root']);need(root.parent==C and not root.exists(),'new own stop claim only');final=old/'model-service-final.json'
 if not final.exists():need(live(parent['controller']),'original model supervisor must own active drain')
 if s['state']=='MODEL_READY_KV_EMPTY_PENDING':need(s['model_READY'] and all(live(r) for r in [s['model_root']]+s['GPU_ranks']),'actual owned model PID/start')
 else:need(s['state'] in ['MODEL_SERVICE_EXITED','MODEL_DRAINING','MODEL_STOPPING','FAILED'] or (old/'stop').exists() or (old/'model-stop').exists(),'unrecognized model transition')
 if a.check:print(json.dumps(dict(state='TWO_NATIVE_EOF_ALL4RANK_OWNED_BOUNDARY_PASS',model_requests=0)));return
 root.mkdir(mode=0o700);now=time.time();requested=False
 if not final.exists() and not (old/'stop').exists() and s['state']=='MODEL_READY_KV_EMPTY_PENDING':
  fd=os.open(old/'stop',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd);requested=True
 deadline=s['deadline_epoch']
 if (old/'stop').exists():deadline=min(deadline,(old/'stop').stat().st_mtime+420)
 out=dict(state='SUCCEEDED',stage_kind='OWNED_STOP_REQUEST_ONLY',model_requests=0,owned_stop_requested=requested,model_root=s['model_root'],original_outer_controller=parent['controller'],parent_controller_status=v['parent_controller_status'],authoritative_original_drain_deadline=deadline,original_model_deadline=s['deadline_epoch'],original_drain_reserve_seconds=420,actual_drain_complete=False,at=now)
 q=root/'status.partial';q.write_text(json.dumps(out,indent=2));q.replace(root/'status.json');print(json.dumps(out))
if __name__=='__main__':main()
