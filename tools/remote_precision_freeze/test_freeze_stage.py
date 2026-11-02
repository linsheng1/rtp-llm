"""Offline negative cases for the stage binding, using safe existing receipts."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import freeze_stage as m


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads(Path(os.environ['PRECISION_STAGE_RECIPE']).read_text())

    def test_current_receipts_generate_all_three_argv(self):
        plan = m.build(self.recipe)
        m.validate(plan)
        self.assertEqual(set(plan['jobs']), {'receiver', 'P', 'D'})
        self.assertIn('--fresh-proof', plan['jobs']['P']['argv'])
        self.assertNotIn('--fresh-proof', plan['jobs']['D']['argv'])
        self.assertIn('--p-seed-result', plan['jobs']['receiver']['argv'])

    def test_candidate_drift_blocks(self):
        self.recipe['candidate_commit'] = '0' * 40
        with self.assertRaisesRegex(ValueError, 'candidate differs'):
            m.build(self.recipe)

    def test_namespace_action_mismatch_blocks(self):
        self.recipe['action'] = 'd-restore'
        with self.assertRaisesRegex(ValueError, 'namespaces'):
            m.build(self.recipe)

    def test_duplicate_option_blocks(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            m.option(['--role', 'P', '--role', 'D'], '--role')

    def test_code_change_after_freeze_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.py'
            path.write_text('original')
            pin = m.read(path)[1]
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'drift'):
                m.validate(dict(schema='existing-c3-stage-freeze-v1', local_pins={'control': pin}))

    def test_existing_plan_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            m.save_new(path, {'first': True})
            with self.assertRaises(FileExistsError):
                m.save_new(path, {'second': True})
            self.assertEqual(json.loads(path.read_text()), {'first': True})

    def test_wrong_core_binding_blocks_before_cli(self):
        self.recipe['helpers']['core'] = 'c3_inner_precision_replay_v4.py'
        with self.assertRaisesRegex(ValueError, 'receiver has different frozen dependency'):
            m.build(self.recipe)

    def test_invalid_tag_blocks(self):
        self.recipe['tag'] = '../existing-stage'
        with self.assertRaisesRegex(ValueError, 'tag'):
            m.build(self.recipe)

    def test_paths_cannot_escape_device_control_root(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            m.remote_path(m.CONTROL + '/../private')

    def test_actual_receiver_cli_rejects_unknown_downstream_flag(self):
        plan = m.build(self.recipe)
        plan['jobs']['receiver']['argv'] += ['--wrong-downstream-option', 'value']
        with self.assertRaisesRegex(ValueError, 'actual CLI argv rejected'):
            m.validate(plan)


if __name__ == '__main__':
    unittest.main()
