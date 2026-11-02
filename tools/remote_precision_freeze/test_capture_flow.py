"""Read-only regression against real safe receipts; paths supplied explicitly."""
import copy,json,os,tempfile,unittest
from pathlib import Path
import freeze_stage as m

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.recipe=json.loads(Path(os.environ['PRECISION_CAPTURE_RECIPE']).read_text())
    def test_real_capture_argv(self):
        p=m.build(self.recipe);m.validate(p)
        self.assertEqual(p['max_model_requests'],2)
        self.assertEqual(p['fresh_proof_max_age_seconds'],75)
        self.assertTrue(all(j['argv'][-1]=='--execute' for j in p['jobs'].values()))
    def test_unknown_peer_arg_rejected(self):
        p=m.build(self.recipe);p['jobs']['P']['argv']+=['--ignore-proof']
        with self.assertRaisesRegex(ValueError,'actual CLI argv rejected'):m.validate(p)
    def test_wrong_candidate_rejected(self):
        self.recipe['candidate_commit']='0'*40
        with self.assertRaisesRegex(ValueError,'candidate differs'):m.build(self.recipe)
    def test_wrong_normal_reference_rejected(self):
        self.recipe['normal_reference_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'normal reference'):m.build(self.recipe)
    def test_new_tag_cannot_relabel_old_batch(self):
        self.recipe['tag']='renamed'
        with self.assertRaisesRegex(ValueError,'batch identity'):m.build(self.recipe)
    def test_current_capture_model_not_ordinary_model(self):
        self.recipe['helpers']['model_P']=self.recipe['helpers']['model_D']
        with self.assertRaisesRegex(ValueError,'batch script differs'):m.build(self.recipe)
    def test_changed_observer_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            value=json.loads(Path(self.recipe['batch']).read_text());value['observer_config_sha256']='0'*64
            p=Path(d)/'batch.json';p.write_text(json.dumps(value));self.recipe['batch']=str(p)
            with self.assertRaisesRegex(ValueError,'observer config differs'):m.build(self.recipe)

if __name__=='__main__':unittest.main()
