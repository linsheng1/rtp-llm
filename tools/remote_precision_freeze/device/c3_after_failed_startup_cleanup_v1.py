"""A terminal failed startup is preserved; verify its owned tree is gone, then exec.
No stop, retry, install, model RPC, monitor or modified success criterion.
"""
import argparse,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--previous-root',type=Path,required=True);p.add_argument('--previous-plan-sha256',required=True);p.add_argument('--state-root',type=Path,required=True);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args();assert os.getuid()==500 and a.argv[0]=='--';old=a.previous_root;assert hashlib.sha256((old/'plan.json').read_bytes()).hexdigest()==a.previous_plan_sha256;s=json.loads((old/'model-status.json').read_text());f=json.loads((old/'model-service-final.json').read_text());assert s['state']=='FAILED' and s['command_returncode']==1 and s['cleanup_state']=='FINISHED' and not s['remaining_owned_processes'] and f['state']=='FAILED' and f['exit_code']==1,'original failure/cleanup must be retained'
for who in s.get('owned_processes',[]):
 q=Path('/proc')/str(who['PID'])
 if q.exists():
  fields=(q/'stat').read_text().rsplit(')',1)[1].split();assert q.stat().st_uid!=who['UID'] or fields[19]!=str(who['start_ticks']) or fields[0] in ['Z','X','x'],'old failed owned process remains running'
assert not a.state_root.exists();a.state_root.mkdir(mode=0o700);v=dict(state='FAILED_STARTUP_CLEANUP_VERIFIED_NEXT_ATTEMPT',old_state='FAILED',old_returncode=1,old_failure_preserved=True,no_old_running_owned_process=True,model_requests=0);(a.state_root/'status.json').write_text(json.dumps(v));argv=a.argv[1:];assert argv[0]=='/opt/conda310/bin/python3.10';os.execv(argv[0],argv)
