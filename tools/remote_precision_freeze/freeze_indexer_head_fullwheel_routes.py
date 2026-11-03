"""Bind the existing route builder to its candidate031 launcher; no RPC/SSH."""
import argparse,json
from pathlib import Path
import freeze_stage_fullwheel as f

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 recipe,rp=f.doc(a.recipe)
 f.need(recipe['candidate_commit']=='fb09988a5c552f4f5ff716472d8b50647af22d2c' and recipe['launcher']=='launch_indexer_head_fullwheel_routes.py','only existing candidate031 route adapter')
 plan=f.build(recipe);plan['local_pins'].update(recipe=rp,freeze_stage=f.read(f.BASE/'freeze_stage_fullwheel.py')[1],route_launcher=f.read(f.BASE/'launch_indexer_head_fullwheel_routes.py')[1],adapter=f.read(__file__)[1]);f.validate(plan)
 print(json.dumps(dict(state='FROZEN_OFFLINE_ONLY',SHA256=f.save_new(a.output,plan),model_requests=0)))
if __name__=='__main__':main()
