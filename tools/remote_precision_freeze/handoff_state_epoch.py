"""Safe completed-request boundary -> original own drain wrapper -> frozen fresh P."""
import argparse,hashlib,json,os
from pathlib import Path
from urllib.request import urlopen
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert digest(a.plan)==a.sha256;v=json.loads(a.plan.read_text());assert os.getuid()==500 and digest(Path(__file__))==v['handoff']['SHA256'];old=Path(v['previous_root']);assert digest(old/'plan.json')==v['previous_plan_SHA256'];status=json.loads((old/'model-service-status.json').read_text());assert status['state']=='MODEL_READY_KV_EMPTY_PENDING' and status['model_READY'] is True
for row in [status['model_root']]+status['GPU_ranks']:
 proc=Path('/proc')/str(row['PID']);assert proc.stat().st_uid==row['UID']==500 and proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]==str(row['start_ticks']),'old owned epoch changed'
for proof in v['completed_stages']:
 with urlopen(proof['URL'],timeout=10) as r:raw=r.read(1024**2+1)
 assert len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==proof['SHA256'];state=json.loads(raw);assert all(state[k]==x for k,x in proof['fields'].items()),'prior actual request not complete'
wrapper=Path(v['wrapper']['path']);assert digest(wrapper)==v['wrapper']['SHA256'];assert digest(Path(v['model_root'])/'plan.json')==v['model_plan_SHA256'];claim=Path(v['handoff_claim']);assert not claim.exists()
with claim.open('x') as out:json.dump(dict(plan_SHA256=a.sha256,old_process=status['model_root'],reason='fresh allocator instance required AFTER completed source request; healthy D retained'),out);out.flush();os.fsync(out.fileno())
(old/'stop').touch(exist_ok=False)
argv=['/opt/conda310/bin/python3.10','-B','-S',str(wrapper),'--previous-root',str(old),'--state-root',v['drain_root'],'--']+v['model_argv'];os.execv(argv[0],argv)
