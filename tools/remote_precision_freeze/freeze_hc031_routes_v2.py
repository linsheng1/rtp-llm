"""Bind the existing route builder to its candidate031 launcher; no RPC/SSH."""
import argparse,json
from pathlib import Path
import freeze_stage as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 recipe,rp=f.doc(a.recipe)
 f.need(recipe['candidate_commit']=='031fc2af33111e7bbe13bbca6e72ebd6fc094d7c' and recipe['launcher']=='launch_hc031_routes_v2.py','only existing candidate031 route adapter')
 plan=f.build(recipe);plan['local_pins'].update(recipe=rp,freeze_stage=f.read(f.BASE/'freeze_stage.py')[1],route_launcher=f.read(f.BASE/'launch_hc031_routes_v2.py')[1],adapter=f.read(__file__)[1]);f.validate(plan)
 print(json.dumps(dict(state='FROZEN_OFFLINE_ONLY',SHA256=f.save_new(a.output,plan),model_requests=0)))
if __name__=='__main__':main()
