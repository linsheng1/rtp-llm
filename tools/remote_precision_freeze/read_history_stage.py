"""One bounded read of original controller evidence, never a sampling loop/RPC.

Only counts, hashes, process identities and existing control status leave devices.
Raw native token IDs and captured tensors remain on the cluster.
"""
import argparse
import json
from pathlib import Path
import sys
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',action='store_true')
    a=p.parse_args()
    plan,pin=f.doc(a.plan)
    f.need(pin['SHA256']==a.sha256 and plan['candidate_commit']=='91b4635faf8c65cc67e66ab52a46f5c932cd08ea','actual frozen candidate plan required')
    f.validate(plan)
    recipe,_=f.doc(plan['local_pins']['recipe']['path'])
    sys.path.insert(0,recipe['bench'])
    from remote_control import remote
    root=f.option(plan['jobs']['receiver']['argv'],'--root')
    f.need(Path(root).parent==Path(f.CONTROL),'owned direct task root required')
    code='''import json,time,hashlib
from pathlib import Path
c=Path(%r);root=Path(%r);o=dict(at=time.time())
for n in ['status.json','source-calibration-safe.json']:
 p=root/n
 if p.exists():o[n]=json.loads(p.read_text())
p=root/'request-once.json';o['request_once_exists']=p.exists()
p=root/'request/native/result.json';o['native_result_exists']=p.exists()
if p.exists():
 raw=p.read_bytes();d=json.loads(raw);o['native_result']=dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),state=d.get('state'),input_tokens=len(d['input_ids']),output_tokens=len(d['output_ids']),EOF=d.get('response_eof_complete'),request_id=d.get('request_id'),namespace=d.get('namespace'),input_SHA256=d.get('input_ids_sha256'))
p=root/'request/native/status.json'
if p.exists():
 d=json.loads(p.read_text());o['request/native/status.json']={k:v for k,v in d.items() if k in ['state','request_id','started_at','finished_at','received_output_tokens','response_eof_complete','mode','input_tokens']}
print(json.dumps(o))
'''%(f.CONTROL,root)
    out=remote('cluster3',plan['jobs']['receiver']['host'],code,60)
    if a.models:
        out['models']={}
        for role in ['P','D']:
            row=plan['roles'][role]
            code='''import json,os
from pathlib import Path
r=Path(%r);p=r/'model-service-status.json';d=json.loads(p.read_text());out={k:d.get(k) for k in ['state','model_READY','deadline_epoch','heartbeat_at','actual_model_root_returncode','error']}
for field in ['model_root','GPU_ranks']:
 values=d.get(field,[]) if field=='GPU_ranks' else [d[field]] if d.get(field) else [];out[field]=[]
 for v in values:
  p=Path('/proc')/str(v['PID']);live=False
  if p.exists():
   s=p.joinpath('stat').read_text().rsplit(')',1)[1].split();live=p.stat().st_uid==v['UID'] and s[19]==str(v['start_ticks']) and s[0] not in ['Z','X','x']
  out[field].append(dict(PID=v['PID'],UID=v['UID'],start_ticks=v['start_ticks'],actual_same_epoch_live=live))
print(json.dumps(out))
'''%row['root']
            out['models'][role]=remote('cluster3',row['host'],code,60)
    f.save_new(a.output,out)
    print(json.dumps(dict(state=out.get('status.json',{}).get('state'),native_result=out.get('native_result'),models=out.get('models'))))


if __name__=='__main__':main()
