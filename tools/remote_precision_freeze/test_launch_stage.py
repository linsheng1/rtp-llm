"""Exercise the persistent CLI through render and original submit, with no devices."""
import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import freeze_stage as freeze
import launch_stage

class LaunchTests(unittest.TestCase):
    def test_actual_entry_renders_three_jobs_and_submits_unchanged(self):
        recipe=Path(os.environ['PRECISION_CAPTURE_RECIPE'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);planpath=root/'plan.json'
            frozen=launch_stage.command('freeze','--recipe',str(recipe),'--output',str(planpath))
            plan=json.loads(planpath.read_text());calls=[];scripts=[]
            def remote(cluster,host,source,timeout=60):
                self.assertEqual(cluster,'cluster3');compile(source,'<actual-device-control>','exec')
                scripts.append((host,source));return [] if 'print(json.dumps(missing))' in source else {'state':'MOCK_NO_DEVICE_NO_RPC'}
            def submit(cluster,host,name,argv,timeout,verify):
                calls.append((host,name,argv,timeout,verify));return {'actual_bootstrap_returncode':0}
            modules={'remote_control':types.SimpleNamespace(remote=remote),
                     'cluster23_dispatch':types.SimpleNamespace(submit=submit)}
            args=['launch_stage.py','--plan',str(planpath),'--sha256',frozen['SHA256'],
                  '--evidence-dir',str(root/'evidence'),'--device-prepare','--submit']
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',args):launch_stage.main()
            self.assertEqual(len(calls),3)
            for actual,role in zip(calls,['receiver','P','D']):
                job=plan['jobs'][role]
                self.assertEqual(actual,(job['host'],job['name'],job['argv'],job['timeout'],job['verify']))
            self.assertTrue(scripts)
            self.assertTrue((root/'evidence/device-preflight-complete.json').is_file())
if __name__=='__main__':unittest.main()
