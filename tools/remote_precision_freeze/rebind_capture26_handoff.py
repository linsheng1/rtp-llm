"""Keep the CPU-prepared model plan; bind the actual failed-controller schema.
Only the pre-model handoff helper changes. No model/config/root rebuild.
"""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--parent',type=Path);p.add_argument('--bench',type=Path,required=True);p.add_argument('--dispatch',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  old,parent=f.doc(a.parent);plan=json.loads(json.dumps(old));helper=f.read(f.BASE/'handoff_state_epoch_v4.py')[1];plan['handoff']=dict(path=f.CONTROL+'/handoff_state_epoch_v4.py',SHA256=helper['SHA256']);plan['local_pins'].update(handoff=helper,parent_epoch_plan=parent,rebind_entry=f.read(__file__)[1]);job=plan['job'];job['name']='state91b-capture26-P-model-guardv2';job['argv'][3]=plan['handoff']['path'];job['argv'][job['argv'].index('--plan')+1]=f.CONTROL+'/'+a.plan.name;sha=f.save_new(a.plan,plan);f.save_new(a.dispatch,dict(role='P',host=job['host'],root=Path(plan['model_root']).name,plan_SHA256=plan['model_plan_SHA256'],argv=plan['model_argv'],parent_controller_status=f.CONTROL+'/'+job['name']+'-status.json',p_launch_guard_root=plan['drain_root'],epoch_plan_SHA256=sha));print(json.dumps(dict(SHA256=sha,model_plan_reused=True,model_requests=0)));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'new handoff plan drift')
 for value in plan['local_pins'].values():f.need(f.read(value['path'])[1]==value,'source drift')
 sys.path.insert(0,str(a.bench));from remote_control import remote
 from cluster23_dispatch import submit
 job=dict(plan['job']);job['argv']=[a.sha256 if x=='${PLAN_SHA256}' else x for x in job['argv']];assets={plan['handoff']['path']:Path(plan['local_pins']['handoff']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()};code='''import json,hashlib,os\nfrom pathlib import Path\nr=Path(%r);assert hashlib.sha256((r/'plan.json').read_bytes()).hexdigest()==%r;exit=json.loads((r/'config-exit.json').read_text());assert exit['actual_child_returncode']==0 and not exit['timed_out'] and exit['plan_SHA256']==%r\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw\n else:\n  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nprint(json.dumps(dict(original_CPU_model_plan_reused=True,actual_RPC=0)))'''%(plan['model_root'],plan['model_plan_SHA256'],plan['model_plan_SHA256'],assets);check=remote('cluster3',job['host'],code,60);launch=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,check=check,launch=launch));print(json.dumps(dict(state='SAFE_HANDOFF_SUBMITTED_NOT_MODEL_READY',model_requests=0)))
if __name__=='__main__':main()
