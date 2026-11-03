"""Original orderly idle Consumer handoff, bound to actual v2 and drained capture36.
No binary/SDK/port changes and no model signals. CPU preflight precedes own stop.
"""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 p.add_argument('command',choices=['freeze','check','render','submit'])
 p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--parent-plan',type=Path)
 p.add_argument('--model',type=Path);p.add_argument('--role',choices=['P','D']);p.add_argument('--bench',type=Path,required=True);p.add_argument('--receipt',type=Path)
 a=p.parse_args();sys.path.insert(0,str(a.bench));from remote_control import remote
 from cluster23_dispatch import submit
 if a.command=='freeze':
  parent,pp=f.doc(a.parent_plan);model,mp=f.doc(a.model)
  f.need(parent['role']==a.role and model['role']==a.role and parent['host']==model['host']==f.HOSTS[a.role],'actual role/cluster differs')
  old=parent['new_root'];new=f.CONTROL+'/consumer91b-'+a.role+'-v5';host=parent['host'];oldargv=parent['new_argv'];argv=oldargv[:];argv[argv.index('--root')+1]=new
  code="import json,hashlib\nfrom pathlib import Path\np=Path(%r)/'ready.json';raw=p.read_bytes();d=json.loads(raw);print(json.dumps(dict(ready_SHA256=hashlib.sha256(raw).hexdigest(),child=d['child'])))"%old
  current=remote('cluster3',host,code,60);script=f.read(f.BASE/'renew_consumer_at_boundary.py')[1];helper=f.read(a.bench/'c3_consumer_inner_service_v4.py')[1];tag='state031-'+a.role+'-consumer-v5'
  plan=dict(schema='existing-consumer-boundary-renewal-v1',role=a.role,host=host,old_root=old,new_root=new,old_ready_SHA256=current['ready_SHA256'],old_child=current['child'],
    old_controller=f.CONTROL+'/'+parent['job']['name']+'-status.json',old_argv=oldargv,new_argv=argv,previous_model_root=f.option(model['argv'],'--root'),previous_model_plan_SHA256=model['plan_SHA256'],
    must_have_no_model=[f.CONTROL+'/airline4-'+a.role+'-capture42'],output_root=f.CONTROL+'/'+tag+'-handoff',script=dict(path=f.CONTROL+'/renew_consumer_at_boundary.py',SHA256=script['SHA256']),
    helper=dict(path=f.CONTROL+'/c3_consumer_inner_service_v4.py',SHA256=helper['SHA256']),local_pins=dict(parent=pp,model=mp,script=script,helper=helper,entry=f.read(__file__)[1]),actual_RPC=0,new_binary_or_install=False,
    job=dict(host=host,name=tag,timeout=14640,argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/renew_consumer_at_boundary.py','--plan',f.CONTROL+'/'+a.plan.name,'--sha256','${PLAN_SHA256}'],
       verify=[dict(path=Path(new).name+'/status.json',json_fields=dict(state='EXITED',exit_code=0))]))
  print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),actual_RPC=0)));return
 plan,pin=f.doc(a.plan);f.need(pin['SHA256']==a.sha256,'Consumer handoff plan drift')
 for pin in plan['local_pins'].values():f.need(f.read(pin['path'])[1]==pin,'local pin drift')
 job=dict(plan['job']);job['argv']=[a.sha256 if x=='${PLAN_SHA256}' else x for x in job['argv']];
 if a.command=='render':print(json.dumps(job));return
 assets={plan['script']['path']:Path(plan['local_pins']['script']['path']).read_text(),f.CONTROL+'/'+a.plan.name:a.plan.read_text()};code='''import json,os,subprocess\nfrom pathlib import Path\nfor name,raw in %r.items():\n p=Path(name)\n if p.exists():assert not p.is_symlink() and p.read_text()==raw,'immutable source/control differs'\n else:\n  with p.open('x') as o:o.write(raw);o.flush();os.fsync(o.fileno())\nr=subprocess.run(%r+['--check'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr\nprint(json.dumps(dict(actual_CLI_and_idle_preflight=json.loads(r.stdout),actual_RPC=0)))'''%(assets,job['argv']);cpu=remote('cluster3',job['host'],code,150)
 if a.command=='check':print(json.dumps(cpu));return
 result=submit('cluster3',job['host'],job['name'],job['argv'],job['timeout'],job['verify']);f.save_new(a.receipt,dict(plan_SHA256=a.sha256,CPU=cpu,launch=result));print(json.dumps(dict(state='IDLE_CONSUMER_HANDOFF_SUBMITTED_NOT_READY',actual_RPC=0)))

if __name__=='__main__':main()
