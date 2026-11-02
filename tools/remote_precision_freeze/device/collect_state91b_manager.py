"""Read-only new candidate D metadata, reusing original collector arithmetic."""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
from urllib.request import urlopen,Request,build_opener,ProxyHandler
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(a.plan)==a.sha256;v=json.loads(a.plan.read_text());assert os.getuid()==500 and sha(Path(__file__))==v['seed_script']['SHA256'];root=Path(v['root']);root.mkdir(mode=0o700,exist_ok=False)
def save(name,x):
 t=root/(name+'.partial');t.write_text(json.dumps(x,sort_keys=True,indent=2));t.replace(root/name)
try:
 source_path=Path(v['source']['path']);assert sha(source_path)==v['source']['SHA256'];source=json.loads(source_path.read_text());assert source['response_eof_complete'] is True
 with urlopen(v['native_URL'],timeout=15) as response:raw=response.read(65537)
 assert len(raw)<=65536;proof=json.loads(raw);assert proof['source']==v['source'] and proof['source_request_id']==source['request_id'] and proof['namespace']==source['namespace'];(root/'native-evidence.json').write_bytes(raw)
 collector=Path(v['collector']['path']);assert sha(collector)==v['collector']['SHA256'];sp=importlib.util.spec_from_file_location('original_Manager_collector',collector);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);meta=m.source_metadata(source,source['namespace']);proc=Path('/proc')/'1078884';fields=(proc/'stat').read_text().rsplit(') ',1)[1].split();assert proc.stat().st_uid==500 and fields[19]=='18146271';access=proc.joinpath('cwd').resolve()/'logs/access.log';st=access.stat();start=max(0,st.st_size-16*1024**2);rows,scan=m.read_access(access,start,16*1024**2);joined=m.transactions(rows,meta,'10.44.61.71',source['namespace'],28800,15);pending=list(scan['errors'])+list(joined['pending']);assert not scan['truncated_start'] and not scan['incomplete_final_record'];opener=build_opener(ProxyHandler({}))
 def post(endpoint,body):
  assert endpoint in ['getInstanceInfo','getCacheLocation'];req=Request('http://127.0.0.1:29384/api/'+endpoint,data=m.canonical(body),headers={'Content-Type':'application/json'})
  with opener.open(req,timeout=15) as response:raw=response.read(4*1024**2+1)
  assert len(raw)<=4*1024**2;return json.loads(raw)
 reads=m.manager_reads(post,joined['selected']) if joined['selected'] else None
 if reads:pending+=reads['pending']
 report=dict(state='PUBLICATION_OBSERVED_METADATA_READABLE' if not pending else 'PENDING_PUBLICATION_EVIDENCE',source=meta,log_scan=scan,transaction_evidence=joined,manager_read_evidence=reads,pending=pending,scope='POSTHOC_NATIVE_PID_AND_MANAGER_METADATA_ONLY_NO_PRE_REQUEST_CURSOR',source_remote_destination_byte_equality='PENDING',exact_request_provenance='PENDING',formal_acceptance=False);save('manager-report.json',report);assert not pending,'metadata not complete; no model replay'
 save('status.json',dict(state='NATIVE_AND_MANAGER_METADATA_READY',model_requests=0,files={name:dict(path=str(root/name),SHA256=sha(root/name)) for name in ['native-evidence.json','manager-report.json']}));print(json.dumps(dict(state='NATIVE_AND_MANAGER_METADATA_READY',model_requests=0)))
except BaseException as e:save('status.json',dict(state='BLOCKED',error=str(e),error_kind=type(e).__name__,model_requests=0));raise
