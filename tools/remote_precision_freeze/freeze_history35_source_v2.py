"""Narrow source adapter for the existing freeze/check/render three-end flow."""
import argparse
import json
from pathlib import Path
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args()
    recipe,rpin=f.doc(a.recipe)
    f.need(recipe['flow']=='history35-source' and recipe['namespace']=='pubprec_state91b_Dorigin_history35_v1',
           'explicit new history namespace required')
    bench=Path(recipe['bench'])
    files=dict(recipe=rpin,freeze_stage=f.read(f.BASE/'freeze_stage.py')[1],
               adapter=f.read(__file__)[1],launcher=f.read(f.BASE/'launch_history35_source_v2.py')[1],
               remote_control=f.read(bench/'remote_control.py')[1],
               cluster23_dispatch=f.read(bench/'cluster23_dispatch.py')[1])
    for name,basename in recipe['helpers'].items():
        files[name]=f.read(bench/basename)[1]
    core=f.literals(files['core']['path'])
    f.need(core['B1']==recipe['candidate_commit'] and core['MODEL_HELPER']==files['model_helper']['SHA256'],
           'actual source candidate/model helper differs')
    f.need(all(files['core']['SHA256'] in Path(files[name]['path']).read_text() for name in ['receiver','sender']),
           'actual receiver/sender core pin differs')
    roles,jobs={},{}
    for role in ['P','D']:
        dispatch,files['dispatch:'+role]=f.doc(recipe['roles'][role]['dispatch'])
        argv=dispatch['argv']
        f.need(dispatch['host']==f.HOSTS[role] and f.option(argv,'--role')==role and
               f.option(argv,'--namespace')==recipe['namespace'] and
               f.option(argv,'--env-sha256')==core['ENV'] and
               f.option(argv,'--manifest-sha256')==core['MANIFESTS'][f.HOSTS[role]],'actual frozen role/runtime differs')
        roles[role]=dict(host=f.HOSTS[role],root=f.remote_path(f.option(argv,'--root')),
                         plan_SHA256=dispatch['plan_SHA256'],namespace=recipe['namespace'],
                         parent_controller_status=dispatch['parent_controller_status'])
        jobs[role]=dict(host=f.HOSTS[role],argv=['/opt/conda310/bin/python3.10','-B','-S',
             f.CONTROL+'/'+recipe['helpers']['sender'],'--role',role,'--model-root',roles[role]['root'],
             '--plan-sha256',dispatch['plan_SHA256'],'--output-root',f.CONTROL+'/history35-source2-probe-'+role,
             '--parent-controller-status',dispatch['parent_controller_status']])
    source,files['source_input_receipt']=f.doc(recipe['source_input_receipt'])
    f.need(source['input_tokens']==7041 and recipe['G_input_ids_SHA256']=='22510597b00881511fec56f072a5ef0ed3593b8a32b069e4189f0d75f258e49c',
           'original real source input receipt required')
    jobs['receiver']=dict(host=f.HOSTS['receiver'],argv=['/opt/conda310/bin/python3.10','-B','-S',
        f.CONTROL+'/'+recipe['helpers']['receiver'],'--root',f.CONTROL+'/history35-source-source2',
        '--action','d-source','--tag',recipe['tag'],'--seconds','900','--request-timeout-seconds','180'])
    plan=dict(schema='existing-c3-stage-freeze-v1',flow='state-source',cluster='cluster3',
              scope='one real source request, current original controller, graph settings retained, finite four-rank sampling barriers',
              candidate_commit=recipe['candidate_commit'],action='d-source',tag=recipe['tag'],
              roles=roles,jobs=jobs,local_pins=files,input=dict(tokens=7041,SHA256=recipe['G_input_ids_SHA256'],
              source_input_receipt=files['source_input_receipt']),fresh_proof_max_age_seconds=75,
              runtime_gates='original native identity/protobuf prepare/once/EOF plus existing graph latches and full241 IDs calibration')
    f.validate(plan)
    print(json.dumps(dict(state='FROZEN',SHA256=f.save_new(a.plan,plan),model_requests=0)))


if __name__=='__main__':
    main()
