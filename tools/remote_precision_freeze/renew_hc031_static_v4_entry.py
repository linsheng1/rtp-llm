"""New finite keeper plan, unchanged full-content verifier and original controller.
Old keeper/plan remains untouched; both can hold read-only shared leases.
"""
import argparse,json,sys
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('command',choices=['freeze','check','render','submit']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--parent-plan',type=Path);p.add_argument('--bench',type=Path,required=True);p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.command=='freeze':
  old,pin=f.doc(a.parent_plan)
  for local in old['local_pins'].values():f.need(f.read(local['path'])[1]==local,'parent source pin drift')
  plan=json.loads(json.dumps(old));role=plan['role'];tag='hc031-static-'+role+'-v4';plan.update(seconds=14400,output_root=f.CONTROL+'/'+tag)
  plan['local_pins'].update(parent_plan=pin,renew_entry=f.read(__file__)[1]);job=plan['job'];job.update(name=tag+'-controller',timeout=14460);job['argv'][job['argv'].index('--plan')+1]=f.CONTROL+'/'+a.plan.name;job['verify'][0]['path']=tag+'/status.json'
  print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),model_requests=0,full_content_check_required=True)));return
 import revalidate_static_weights_entry as original
 sys.argv=[sys.argv[0],a.command,'--plan',str(a.plan),'--sha256',a.sha256,'--bench',str(a.bench)]
 if a.receipt:sys.argv+=['--receipt',str(a.receipt)]
 original.main()
if __name__=='__main__':main()
