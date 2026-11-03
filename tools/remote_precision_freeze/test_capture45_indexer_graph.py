"""New indexer-control contracts only; calibrated attention math is not rerun."""
import importlib.util,json
from pathlib import Path
import unittest
class IndexerChain(unittest.TestCase):
 def test_actual_dense_and_paged_query_key_score_capture_and_graph(self):
  here=Path(__file__).parent;s=importlib.util.spec_from_file_location('actual_indexer45_chain',here/'capture45_indexer_contract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);r=m.run('gpu',json.loads((here/'capture45-indexer-contract-config-v1.json').read_text()),here/'history45_observer_v1.py',here/'h45chain-gpu-v1');self.assertEqual(r['state'],'ACTUAL_INDEXER_CONTROL_CHAIN_PASS');self.assertEqual([x['keys'] for x in r['roles']],[7046,7046]);self.assertTrue(r['roles'][1]['GPU_graph'])
