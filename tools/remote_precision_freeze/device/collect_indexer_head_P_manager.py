"""Existing P Manager collector, bound to actual pre-seed cursor and HC031 EOF."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert os.getuid()==500 and sha(a.plan)==a.sha256;v=json.loads(a.plan.read_text());assert sha(Path(__file__))==v['script']['SHA256'];root=Path(v['root']);root.mkdir(mode=0o700,exist_ok=False)
def save(name,value):
 t=root/(name+'.partial');t.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');t.replace(root/name)
try:
 source=Path(v['seed']['path']);assert sha(source)==v['seed']['SHA256'];seed=json.loads(source.read_text())
 assert seed['response_eof_complete'] is True and seed['identity']['source_commit']==v['candidate_commit'] and seed['namespace']=='pubprec_headfb09988_Porigin_v1' and seed['input_ids_sha256']=='10932132ad309cbfd30ae5600e64d7f7bb2dd8d7d2ddb6af4c6349ce2381ef71'
 cp=Path(v['cursor']['path']);assert sha(cp)==v['cursor']['SHA256'];cursor=json.loads(cp.read_text());proc=Path('/proc')/str(cursor['manager']['PID']);fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split();assert proc.stat().st_uid==cursor['manager']['UID']==500 and fields[19]==cursor['manager']['start_ticks']=='18146271'
 log=proc.joinpath('cwd').resolve()/'logs/access.log';st=log.lstat();assert str(log)==cursor['path'] and (st.st_dev,st.st_ino)==(cursor['device'],cursor['inode']) and st.st_size>=cursor['offset'] and cursor['observed_at']<=seed['started_at']
 for key in ['collector','P_collector']:assert sha(Path(v[key]['path']))==v[key]['SHA256']
 sp=importlib.util.spec_from_file_location('original_P_manager_collector',v['P_collector']['path']);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 result=m.collect(source,v['seed']['SHA256'],seed['namespace'],Path(v['collector']['path']),log,cursor['offset']);save('manager-report.json',result)
 assert result['state']=='PUBLICATION_OBSERVED_METADATA_READABLE' and not result['pending'],'actual P publication not complete; never replay seed'
 selected=result['transaction_evidence']['selected'];assert len(selected['all_requested_keys'])==len(selected['newly_allocated_keys'])==14
 save('status.json',dict(state='P_MANAGER_METADATA_READY',model_requests=0,canonical_keys=14,new_locations=14,report=dict(path=str(root/'manager-report.json'),SHA256=sha(root/'manager-report.json'))))
except BaseException as error:save('status.json',dict(state='BLOCKED',model_requests=0,error_kind=type(error).__name__,error=str(error)[:1000]));raise
