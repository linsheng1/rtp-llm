"""Critical no-stop-before-model-drain gates, using the actual guard function."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parent/'renew_consumer_at_boundary.py';s=importlib.util.spec_from_file_location('actual_consumer_boundary',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Admission(unittest.TestCase):
 def case(self,t):
  r=Path(t);(r/'plan.json').write_bytes(b'{}');f=dict(state='MODEL_SERVICE_EXITED',actual_model_root_returncode=0,model_root=dict(PID=999999,UID=500,start_ticks='1'),GPU_ranks=[]);c=dict(cleanup_state='FINISHED',remaining_owned_processes=[])
  (r/'model-service-final.json').write_text(json.dumps(f));(r/'model-status.json').write_text(json.dumps(c));return r,f,c
 def test_complete_idle_boundary(self):
  with tempfile.TemporaryDirectory() as t:
   r,_,_=self.case(t)
   with patch.object(m,'same',lambda _:False):m.model_gone(r,hashlib.sha256(b'{}').hexdigest())
 def test_live_model_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   r,_,_=self.case(t)
   with patch.object(m,'same',lambda _:True),self.assertRaises(ValueError):m.model_gone(r,hashlib.sha256(b'{}').hexdigest())
 def test_remaining_owned_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   r,f,c=self.case(t);c['remaining_owned_processes']=[123];(r/'model-status.json').write_text(json.dumps(c))
   with self.assertRaises(ValueError):m.model_gone(r,hashlib.sha256(b'{}').hexdigest())
 def test_failed_root_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   r,f,c=self.case(t);f['actual_model_root_returncode']=1;(r/'model-service-final.json').write_text(json.dumps(f))
   with self.assertRaises(ValueError):m.model_gone(r,hashlib.sha256(b'{}').hexdigest())
 def test_source_drift_blocks(self):
  with tempfile.TemporaryDirectory() as t:
   r,_,_=self.case(t)
   with self.assertRaises(ValueError):m.model_gone(r,'0'*64)
if __name__=='__main__':unittest.main()
