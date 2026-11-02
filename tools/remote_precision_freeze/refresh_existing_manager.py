"""Frozen read-only relookup of the original source publication after Consumer handoff."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert os.getuid()==500 and sha(a.plan)==a.sha256;v=json.loads(a.plan.read_text());assert sha(Path(__file__))==v['script']['SHA256'];report=Path(v['original']['path']);assert sha(report)==v['original']['SHA256'];old=json.loads(report.read_text());helper=Path(v['collector']['path']);assert sha(helper)==v['collector']['SHA256'];proc=Path('/proc')/'1078884';assert proc.stat().st_uid==500 and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=='18146271';assert old['pending']==[] and old['transaction_evidence']['selected']
 if a.check:print(json.dumps(dict(state='EXISTING_MANAGER_READ_PREPARE_PASS',model_requests=0)));return
 from urllib.request import Request,build_opener,ProxyHandler
 spec=importlib.util.spec_from_file_location('same_original_manager_collector',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);op=build_opener(ProxyHandler({}));count=0
 def post(endpoint,body):
  nonlocal count
  assert endpoint in ['getInstanceInfo','getCacheLocation'];count+=1;req=Request('http://127.0.0.1:29384/api/'+endpoint,data=m.canonical(body),headers={'Content-Type':'application/json'})
  with op.open(req,timeout=15) as response:raw=response.read(4*1024**2+1)
  assert len(raw)<=4*1024**2;return json.loads(raw)
 result=m.manager_reads(post,old['transaction_evidence']['selected']);out=dict(state='EXISTING_SOURCE_MANAGER_LOCATIONS_READABLE' if not result['pending'] else 'BLOCKED',at=time.time(),model_requests=0,Manager_read_calls=count,original_report=v['original'],result=result,source_remote_destination_byte_equality_proven=False,scope='current Manager metadata for original selected source publication; not all cached bytes/origins')
 root=Path(v['root']);assert not root.exists();root.mkdir(mode=0o700)
 with (root/'result.json').open('x') as f:json.dump(out,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 assert not result['pending'],result['pending'];print(json.dumps(dict(state=out['state'],model_requests=0,Manager_read_calls=count)))
if __name__=='__main__':main()
