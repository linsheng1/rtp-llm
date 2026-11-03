"""Current HC031 source + actual model epochs -> original frozen three-end jobs."""
import argparse,json
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 r,rp=f.doc(a.recipe)
 f.need(r['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and r['action'] in ['baseline','d-restore'] and r['namespace']=='pubprec_hc031_hist7045_v1','current two-case diagnostic only')
 files=dict(recipe=rp,freeze_stage=f.read(f.BASE/'freeze_stage.py')[1],adapter=f.read(__file__)[1],launcher=f.read(f.BASE/'launch_capture41.py')[1])
 for k,n in r['helpers'].items():files[k]=f.read(Path(r['bench'])/n)[1]
 for k in ['remote_control','cluster23_dispatch']:files[k]=f.read(Path(r['bench'])/(k+'.py'))[1]
 core=f.literals(files['core']['path']);diag=f.literals(files['diagnostic']['path'])
 f.need(core['B1']==r['candidate_commit'] and core['MODEL_HELPER']==files['model_helper']['SHA256'],'model helper/candidate differs')
 f.need(diag['SOURCE_SHA']=='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa','historical source must remain unchanged')
 for k in ['sender','receiver']:f.need(files['core']['SHA256'] in Path(files[k]['path']).read_text(),'bound core drift')
 f.need(files['diagnostic']['SHA256'] in Path(files['receiver']['path']).read_text(),'bound diagnostic drift')
 source,sp=f.doc(r['source_input_receipt']);files['source_input_receipt']=sp
 f.need(source['source_SHA256']==diag['SOURCE_SHA'] and source['R_input_ids_sha256']=='3982930c351652d4c939acb4e883924ee85bc381454b3eeff16622e8f5edf63c','actual derived input receipt differs')
 roles,jobs={},{}
 for role in ['P','D']:
  dispatch,files['dispatch:'+role]=f.doc(r['roles'][role]['dispatch']);argv=dispatch['argv']
  f.need(dispatch['host']==f.HOSTS[role] and f.option(argv,'--namespace')==r['namespace'] and f.option(argv,'--env-sha256')==core['ENV'] and f.option(argv,'--manifest-sha256')==core['MANIFESTS'][f.HOSTS[role]],'actual model epoch/runtime differs')
  roles[role]=dict(host=f.HOSTS[role],root=f.remote_path(f.option(argv,'--root')),plan_SHA256=dispatch['plan_SHA256'],namespace=r['namespace'],parent_controller_status=dispatch['parent_controller_status'])
  jobs[role]=dict(host=f.HOSTS[role],argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/'+r['helpers']['sender'],'--role',role,'--model-root',roles[role]['root'],'--plan-sha256',dispatch['plan_SHA256'],'--output-root',f.CONTROL+'/capture41-'+r['tag']+'-probe-'+role,'--parent-controller-status',dispatch['parent_controller_status']]+(['--fresh-proof'] if role=='P' and r['action']=='d-restore' else []))
 jobs['receiver']=dict(host=f.HOSTS['receiver'],argv=['/opt/conda310/bin/python3.10','-B','-S',f.CONTROL+'/'+r['helpers']['receiver'],'--root',f.CONTROL+'/capture41-'+r['tag'],'--action',r['action'],'--tag',r['tag'],'--seconds','900','--request-timeout-seconds','180'])
 plan=dict(schema='existing-c3-stage-freeze-v1',candidate_commit=r['candidate_commit'],cluster='cluster3',action=r['action'],tag=r['tag'],roles=roles,jobs=jobs,local_pins=files,input=dict(tokens=7169,SHA256=source['R_input_ids_sha256'],source_path=source['source_path'],source_SHA256=source['source_SHA256']),fresh_proof_max_age_seconds=75,scope='historical7045 original source versus normal prefill; entries0..25 and actual attentioninputs; source and normal referenceIDs exact; no whole-model acceptance')
 f.validate(plan);print(json.dumps(dict(SHA256=f.save_new(a.plan,plan),state='FROZEN',model_requests=0)))
if __name__=='__main__':main()
