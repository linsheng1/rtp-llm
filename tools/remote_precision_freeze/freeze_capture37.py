"""Current real HC031 receipts -> unchanged capture/check/render protocol."""
import argparse,json
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r,rp=f.doc(a.recipe)
 f.need(r['flow']=='capture' and r['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and r['tag']=='capture37','exact next two-request batch')
 v=f.build(r);v['local_pins'].update(recipe=rp,freeze_stage=f.read(f.BASE/'freeze_stage.py')[1],capture_flow=f.read(f.BASE/'capture_flow.py')[1],launcher=f.read(f.BASE/'launch_stage.py')[1],adapter=f.read(__file__)[1],binding_entry=f.read(f.BASE/'bind_capture37_hc031.py')[1],model_prepare_entry=f.read(f.BASE/'prepare_capture37_models.py')[1])
 for role in ['P','D']:
  v['local_pins']['epoch_plan:'+role]=f.read(r['epoch_plans'][role])[1]
 f.validate(v);print(json.dumps(dict(state='FROZEN_OFFLINE_ONLY',SHA256=f.save_new(a.output,v),model_requests=0)))

if __name__=='__main__':main()
