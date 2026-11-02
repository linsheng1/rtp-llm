"""CPU regression for the actual stage observer's terminal admission check."""
import ast,types,unittest
from pathlib import Path

class StageCompletionTests(unittest.TestCase):
 def setUp(self):
  source=Path(__file__).with_name('device')/'p_only_state_observer_v8.py';tree=ast.parse(source.read_text());names={'REQUIRED','STAGE_POINTS','STAGE_QUERY_LAYERS','SUPPLEMENTAL_REQUIRED'};nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)];nodes.append(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Record'));base=type('AlreadyValidatedOriginalTerminal',(),{'completion_ready_to_record':lambda self:True});ns=dict(base=types.SimpleNamespace(Record=base));exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-stage-observer-completion','exec'),ns);self.required=set(ns['REQUIRED'])|ns['SUPPLEMENTAL_REQUIRED'];self.record=ns['Record'].__new__(ns['Record']);self.record.domains_validated=set(range(8));self.record.additional_points=set(self.required);self.record.events=[dict(point='L0.MoE.actual_launch')];self.record.pending_hc=None;self.record.problem=None
 def test_complete_original_and_all_six_new_domains_pass(self):self.assertTrue(self.record.completion_ready_to_record())
 def test_only_old_domains_cannot_publish_completion(self):
  self.record.domains_validated={0,1};self.assertFalse(self.record.completion_ready_to_record())
 def test_missing_actual_stage_point_cannot_publish_completion(self):
  self.record.additional_points.remove('L3.stage.attn_input_norm5120');self.assertFalse(self.record.completion_ready_to_record())
 def test_unplanned_domain_is_rejected(self):
  self.record.domains_validated.add(8);self.assertFalse(self.record.completion_ready_to_record())

if __name__=='__main__':unittest.main()
