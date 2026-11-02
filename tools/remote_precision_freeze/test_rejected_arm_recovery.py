"""Exercise actual recovery guards and actual coordinator deadline expression."""
import ast,hashlib,json,os,tempfile,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent
class Recovery(unittest.TestCase):
 def setup_case(self,tmp,committed=False,consumed=False):
  c=Path(tmp);n=c/'airline4-P-test';r=n/'arm-receiver';(r/'ledger').mkdir(parents=True);e=n/'evidence';e.mkdir();a=r/'ledger/case.arm.json'
  cfg={'cases':[dict(nonce='abc',arm_path=str(a),evidence_dir=str(e))]};(n/'observer-runtime-config.json').write_text(json.dumps(cfg))
  if committed:a.write_text('{}')
  if consumed:(e/'abc-1-live-rid-bound.json').write_text('{}')
  status={'state':'FAILED_OR_EXPIRED_NO_RETRY','failure':{'reason':'ARM exceeds actual model active lease'},'accepted':[],'actual_exitcode':1,'process':{'PID':123,'start_ticks':'100'}};(r/'status.json').write_text(json.dumps(status))
  peer=c/'airline4-P-state-pilot-capture27-peer-P';peer.mkdir();(peer/'status.json').write_text(json.dumps({'state':'BLOCKED','remaining_owned':[]}))
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  b={'p_model_root':str(n),'observer_config_sha256':sha(n/'observer-runtime-config.json'),'rejected_arm_recovery':{'archive':str(n/'arm-receiver-rejected-capture27'),'status_SHA256':sha(r/'status.json')}}
  tree=ast.parse((BASE/'device/c3_P_state_pilot_peer_v17.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='preserve_rejected_receiver');ns=dict(Path=Path,os=os,json=json,sha=sha,C=c,process=lambda _:None,canonical=lambda v:json.dumps(v).encode());exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-recovery>','exec'),ns);return ns['preserve_rejected_receiver'],b,r
 def test_preserves_rejection_and_unused_case(self):
  with tempfile.TemporaryDirectory() as t:
   fn,b,r=self.setup_case(t);before=(r/'status.json').read_bytes();fn(b);self.assertFalse(r.exists());self.assertEqual((Path(b['rejected_arm_recovery']['archive'])/'status.json').read_bytes(),before)
 def test_committed_ledger_blocks_reuse(self):
  with tempfile.TemporaryDirectory() as t:
   fn,b,r=self.setup_case(t,committed=True)
   with self.assertRaises(AssertionError):fn(b)
   self.assertTrue(r.exists())
 def test_live_rid_blocks_reuse(self):
  with tempfile.TemporaryDirectory() as t:
   fn,b,r=self.setup_case(t,consumed=True)
   with self.assertRaises(AssertionError):fn(b)
   self.assertTrue(r.exists())
 def test_actual_deadline_expression_respects_P_reserve(self):
  tree=ast.parse((BASE/'device/c3_P_state_pilot_coordinator_v16.py').read_text());d=next(x for x in ast.walk(tree) if isinstance(x,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='ARM_DEADLINE' for k in x.keys));expr=next(v for k,v in zip(d.keys,d.values) if isinstance(k,ast.Constant) and k.value=='ARM_DEADLINE')
  with tempfile.TemporaryDirectory() as t:
   ip=Path(t)/'identity.json';ip.write_text(json.dumps({'roles':{'P':{'deadline_epoch':2000},'D':{'deadline_epoch':2500}}}));deadline=float(eval(compile(ast.Expression(expr),'<actual-deadline>','eval'),dict(json=json,ip=ip)));self.assertLess(deadline,2000-420);self.assertEqual(deadline,1579)
if __name__=='__main__':unittest.main()
