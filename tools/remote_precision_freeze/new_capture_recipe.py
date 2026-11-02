"""Create a new immutable capture recipe from real model CPU/dispatch receipts."""
import argparse,json
from pathlib import Path
from freeze_stage import save_new,read,CONTROL,need

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ['reference-recipe','models-dir','output-dir','tag']:p.add_argument('--'+key,required=True)
    a=p.parse_args();models=Path(a.models_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    r=json.loads(Path(a.reference_recipe).read_text());r['tag']=a.tag
    dispatch={role:json.loads((models/(role+'-dispatch.json')).read_text()) for role in ['P','D']}
    pcpu=json.loads((models/'P-CPU.json').read_text());batch=json.loads(Path(r['batch']).read_text())
    batch['job_prefix']='airline4-P-state-pilot-'+a.tag
    batch.update(json.loads((models/'nonces.json').read_text()))
    batch['observer_config_sha256']=pcpu['observer-config-manifest.json']['derived_config']['SHA256']
    parent_name=dispatch['P']['launch']['receipt']['name']
    need(parent_name.startswith('root-') and parent_name.endswith('-P-model'), 'actual P parent identity differs')
    batch['p_launch_guard_root']=CONTROL+'/'+parent_name[:-len('-model')]+'-after-drain'
    for role in ['P','D']:
        d=dispatch[role];batch[role.lower()+'_model_root']=CONTROL+'/'+d['root'];batch[role.lower()+'_plan_sha256']=d['plan_SHA256']
        r['roles'][role]=dict(dispatch=str((models/(role+'-dispatch.json')).resolve()),launch_receipt=str((models/(role+'-dispatch.json')).resolve()),parent_controller_status=CONTROL+'/'+d['launch']['receipt']['name']+'-status.json')
    r['helpers'].update(receiver='c3_P_state_pilot_coordinator_v11.py',sender='c3_P_state_pilot_peer_v12.py',
          readiness='c3_P_diag_zero_request_readiness_v11.py',probe_P='c3_P_state_runtime_probe_v8.py',
          fresh_proof='c3_P_state_empty_probe_v8.py',client='c3_P_state_diagnostic_client_v10.py')
    bench=Path(r['bench'])
    for key,field in [('probe_P','p_probe_sha256'),('fresh_proof','p_empty_sha256')]:batch[field]=read(bench/r['helpers'][key])[1]['SHA256']
    r['publish_helpers']=list(r['helpers']) # source only; existing device files must match
    argv=json.loads(Path(r['prepare']).read_text());argv['argv'][3]=CONTROL+'/'+r['helpers']['client']
    b=out/(a.tag+'-batch.json');prepare=out/(a.tag+'-prepare-argv.json')
    save_new(b,batch);save_new(prepare,argv)
    r['batch']=str(b.resolve());r['prepare']=str(prepare.resolve());r['observer_manifest']=str((models/'P-CPU.json').resolve())
    save_new(out/'recipe.json',r)
if __name__=='__main__':main()
