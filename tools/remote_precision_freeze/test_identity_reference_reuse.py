"""Exercise the actual same-instance immutable identity publication helper."""
import ast,hashlib,json,os,tempfile,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent
class IdentityReuse(unittest.TestCase):
 def helper(self):
  tree=ast.parse((BASE/'device/c3_P_state_pilot_coordinator_v17.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['sha','publish','publish_identity_replica']];ns=dict(Path=Path,hashlib=hashlib,os=os);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-identity-helper>','exec'),ns);return ns['publish_identity_replica']
 def test_new_reference(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'identity.json';v=self.helper()(p,b'{}');self.assertEqual(p.read_bytes(),b'{}');self.assertEqual(v['SHA256'],hashlib.sha256(b'{}').hexdigest())
 def test_identical_reference_is_reused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'identity.json';p.write_bytes(b'{}');before=p.stat();self.helper()(p,b'{}');after=p.stat();self.assertEqual((before.st_ino,before.st_mtime_ns),(after.st_ino,after.st_mtime_ns))
 def test_drift_is_rejected_without_overwrite(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'identity.json';p.write_bytes(b'old')
   with self.assertRaises(AssertionError):self.helper()(p,b'new')
   self.assertEqual(p.read_bytes(),b'old')
if __name__=='__main__':unittest.main()
