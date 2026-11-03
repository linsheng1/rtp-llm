"""One bounded controller/EOF read; only counts, hashes and identity metadata leave C3."""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'plan drift');f.validate(plan)
 sys.path.insert(0,str(Path(plan['local_pins']['core']['path']).parent));from remote_control import remote
 root=f.option(plan['jobs']['receiver']['argv'],'--root');tag=plan['tag']
 code='''import json,time,hashlib
from pathlib import Path
c=Path(%r);r=Path(%r);out=dict(at=time.time(),plan_SHA256=%r)
for key,p in [('controller',c/%r),('status.json',r/'status.json')]:
 if p.exists():out[key]=json.loads(p.read_text())
out['request_once_exists']=(r/'request-once.json').exists()
p=r/'request/native/result.json'
if p.exists():
 raw=p.read_bytes();d=json.loads(raw);out['native_result']=dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),EOF=d['response_eof_complete'],input_tokens=len(d['input_ids']),output_tokens=len(d['output_ids']),request_id=d['request_id'],input_SHA256=d['input_ids_sha256'])
print(json.dumps(out))
'''%(f.CONTROL,root,a.sha256,'capture41-'+tag+'-receiver-status.json')
 out=remote('cluster3',f.HOSTS['receiver'],code,60);f.save_new(a.output,out);print(json.dumps(dict(controller=out.get('controller',{}).get('state'),state=out.get('status.json',{}).get('state'),native=out.get('native_result'),error=out.get('status.json',{}).get('error'))))
if __name__=='__main__':main()
