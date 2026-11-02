"""Bind the next normal batch to the actual completed source; no device writes.

This reuses the existing calibrated native baseline path. New source files and
recipe are created exclusively; old code, plans, fingerprints and results stay
immutable. freeze/check/render/submit remains a separate required step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import freeze_stage as f


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source-actual',type=Path,required=True)
    p.add_argument('--reference-recipe',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--model-directory',type=Path,required=True)
    a=p.parse_args()
    actual,actual_pin=f.doc(a.source_actual)
    f.need(actual['status.json']['state']=='D_SOURCE_COMPLETE' and
           actual['status.json']['source_calibration']['full_output_IDs_equal'] and
           actual['native_result']['EOF'] and actual['native_result']['output_tokens']==241,
           'actual completed calibrated source/export receipt required')
    src=actual['native_result']
    f.need(src['path']==f.CONTROL+'/history36-source-source1/request/native/result.json',
           'this adapter only binds its actual new source root')
    recipe,rp=f.doc(a.reference_recipe)
    e=f.BASE
    core=f.read(e/'device/c3_inner_precision_replay_v10.py')[1]['SHA256']
    original='2141067423f6fa268786f7585717f0f5b4338c5a8206b7735d1d424368cc9a92'
    def adapt(s):
        return s.replace('history35','history36').replace('HISTORY35','HISTORY36').replace(
          'c3_inner_precision_replay_v9','c3_inner_precision_replay_v10').replace(
          'c3_inner_precision_replay_v8','c3_inner_precision_replay_v10').replace(
          '4eef2d5151049411ff7a71fb02164e7e5f64d9b5da49b3005748de1b80d36c67',core).replace(
          'c43698f733eddc9db9f194bf84b00bf00e98ec96a49db1ea005cb8674b843195',core).replace(
          original,src['SHA256']).replace('history36-source-source3/request/native/result.json',
          'history36-source-source1/request/native/result.json')
    bench=Path(recipe['bench'])
    client=adapt((e/'device/c3_history35_baseline_client_v2.py').read_text())
    new_client='c3_history36_baseline_client_v1.py'
    client_sha=hashlib.sha256(client.encode()).hexdigest()
    receiver=adapt((e/'device/c3_history35_baseline_receiver_v2.py').read_text()).replace(
          'c3_history36_baseline_client_v2.py',new_client).replace(
          'c5a6b4a914c3c64b7aa3938260c94dfa24adb04af35a49c9acb547056b5cbe59',client_sha)
    sender=adapt((e/'device/c3_history35_baseline_sender_v2.py').read_text())
    assets={new_client:client,'c3_history36_baseline_receiver_v1.py':receiver,
            'c3_history36_baseline_sender_v1.py':sender}
    for name,raw in assets.items():
        for dest in [e/'device'/name,bench/name]:
            f.need(not dest.exists(),'new actual-source-bound code only: '+str(dest))
            with dest.open('x') as out:out.write(raw)
    recipe.update(flow='history36-baseline',namespace='pubprec_state91b_Dorigin_history36_v1',
                  tag='h36normal1',source_input_receipt=str(a.source_actual),
                  observed_source_result={k:src[k] for k in ['path','SHA256']},
                  observed_source_receipt=actual_pin,adapter=f.read(__file__)[1])
    recipe['helpers'].update(core='c3_inner_precision_replay_v10.py',
        model_helper='c3_history36_model_service_v1.py',receiver='c3_history36_baseline_receiver_v1.py',
        sender='c3_history36_baseline_sender_v1.py',diagnostic=new_client)
    for role in ['P','D']:
        recipe['roles'][role]['dispatch']=str(a.model_directory/(role+'-dispatch.json'))
    recipe.pop('previous_actual_zero_RPC_receipt',None)
    recipe['control_fix']='exact current completed source core10/native receipt, current runtime core10; existing baseline stdout CPU prepare and original guards retained'
    print(json.dumps(dict(state='ACTUAL_SOURCE_BOUND_RECIPE_NOT_SUBMITTED',SHA256=f.save_new(a.recipe,recipe))))


if __name__=='__main__':main()
