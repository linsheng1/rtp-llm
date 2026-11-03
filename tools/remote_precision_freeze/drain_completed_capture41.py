"""Verified two-request/four-rank boundary -> original owned model drain."""
import argparse,hashlib,json,os,time
from pathlib import Path
from urllib.request import urlopen
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def fetch(pin):
 with urlopen(pin['URL'],timeout=15) as f:raw=f.read(1024**2+1)
 need(len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'completed stage/ack drift');return json.loads(raw)
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();need(sha(a.plan)==a.sha256 and os.getuid()==500,'actual owned plan');v=json.loads(a.plan.read_text());need(sha(__file__)==v['script']['SHA256'],'exact drain control');old=Path(v['model_root']);need(old.parent==C and sha(old/'plan.json')==v['model_plan_SHA256'],'exact owned model');s=json.loads((old/'model-service-status.json').read_text());need(s['state']=='MODEL_READY_KV_EMPTY_PENDING' and s['model_READY'] and -2<=time.time()-s['heartbeat_at']<=45,'current idle model health');current=[s['model_root']]+s['GPU_ranks']
 for r in current:
  q=Path('/proc')/str(r['PID']);f=q.joinpath('stat').read_text().rsplit(')',1)[1].split();need(q.stat().st_uid==r['UID']==500 and f[19]==str(r['start_ticks']) and f[0] not in ['Z','X','x'],'owned PID/start identity')
 for proof in v['completed_stages']:
  d=fetch(proof);need(d['state']==proof['state'] and d['model_requests']==1 and d['capture_phase']=='DONE' and d['source_calibration']['complete_EOF'] and d['source_calibration']['full_output_IDs_equal'] and d['source_calibration']['output_tokens']==proof['output_tokens'],'actual requests not complete');ack=fetch(proof['role_ack']);need(all(str(ack['model_root'][k])==str(s['model_root'][k]) for k in ['UID','PID','start_ticks']) and {r['PID'] for r in ack['rank_controls']}=={r['PID'] for r in s['GPU_ranks']} and len(ack['rank_controls'])==4,'completed stage belongs to another model epoch')
 need(sha(v['wrapper']['path'])==v['wrapper']['SHA256'],'original drain wrapper');root=Path(v['output_root']);need(root.parent==C and not root.exists() and not (old/'stop').exists(),'one new own drain only')
 if a.check:print(json.dumps(dict(state='TWO_EOF_ALL4RANK_OWNED_BOUNDARY_PASS',model_requests=0)));return
 root.mkdir(mode=0o700)
 with (root/'claim.json').open('x') as f:json.dump(dict(plan_SHA256=a.sha256,old_model_root=s['model_root'],reason='completed capture41; required next startup sampling only; retain runtime storage and keepers'),f);f.flush();os.fsync(f.fileno())
 fd=os.open(old/'stop',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
 # Existing finite240s wrapper owns waiting and verifies successful final root/process cleanup.
 program="import json;from pathlib import Path;r=Path("+repr(str(root))+");p=r/'status.partial';p.write_text(json.dumps(dict(state='SUCCEEDED',model_requests=0,original_owned_model_drain_complete=True)));p.replace(r/'status.json')"
 argv=['/opt/conda310/bin/python3.10','-B','-S',v['wrapper']['path'],'--previous-root',str(old),'--state-root',str(root/'original-wrapper'),'--','/opt/conda310/bin/python3.10','-B','-S','-c',program];os.execv(argv[0],argv)
if __name__=='__main__':main()
