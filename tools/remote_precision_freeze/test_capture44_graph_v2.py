"""Only new wrapper-to-export CUDA Graph control contract; no model request."""
import importlib.util,json
from pathlib import Path
import unittest
class FullChain(unittest.TestCase):
 def test_actual_wrapper_context_latches_graph_export_and_comparator(self):
  here=Path(__file__).parent;s=importlib.util.spec_from_file_location('exact_capture44_chain',here/'capture44_chain_contract_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  result=m.run('gpu',json.loads((here/'chain-config-v1.json').read_text()),here/'history44_observer_v1.py',here/'h44chain-gpu-v2')
  self.assertTrue(result['GPU_graph_tested']);self.assertEqual(result['state'],'ACTUAL_RECORD_EXPORT_CHAIN_PASS');self.assertEqual([r['role'] for r in result['roles']],['P','D'])
