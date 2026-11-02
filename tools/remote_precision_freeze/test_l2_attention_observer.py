"""Actual add-on wrapper order/cleanup and CPU logical-index/budget gates."""
import hashlib,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent/'device/p_only_state_observer_v10.py';spec=importlib.util.spec_from_file_location('actual_l2_addon_test',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tensor:
 def __init__(self,shape,dtype='torch.bfloat16'):self.shape=shape;self.dtype=dtype
 def element_size(self):return 1 if self.dtype=='torch.uint8' else 2
 def is_floating_point(self):return self.dtype!='torch.uint8'
class Vector:
 def numel(self):return 1
class Original:
 def __init__(self,log):self.log=log
 def project_q(self,linear,qr):self.log.append('q');return Tensor((1,64,512))
 def prefill(self,q,kv,indices,pos,lengths):self.log.append('core');return Tensor((1,8,4096),'torch.uint8'),Tensor((1,8,128),'torch.uint8')
class Attention:
 layer_id=2;softmax_scale=0.044;n_heads=64;head_dim=512;compress_ratio=2;tp_size=1
 def __init__(self,log):self.log=log;self._mega_attention=Original(log);self._shared_attention={'prefill_chunk_meta':(0,3584,7041)}
 def _prefill_output_all_reduce(self,out):self.log.append('AR')
 def _project_prefill_q(self,*a):raise AssertionError('unused split path')
 def _prefill_output_proj_into(self,*a):raise AssertionError('unused split path')
class Gate(unittest.TestCase):
 def test_install_preserves_actual_probe_markers(self):
  calls=[];fake=types.SimpleNamespace(observers=[types.SimpleNamespace(config={})])
  with patch.object(m,'Dispatcher',lambda _:fake),patch.object(m.parent,'Finder',lambda _:object()),patch.object(m.sys,'meta_path',[]),patch.object(m.base,'evidence',lambda c,p,v:calls.append(p)),patch.object(m.parent,'extension_evidence',lambda c,p,v:calls.append(p)):
   m.install({})
  self.assertEqual(calls,['lazy-finder-installed','extension-lazy-finder-installed','attention-lazy-finder-installed'])
 def test_budget_stays_inside_original_limits(self):self.assertLessEqual(m.attention_budget()['peak_upper_bound'],4*1024**2)
 def test_actual_logical_SWA_shift(self):
  left=m.canonical_kv_indices([4,3584,-1],2,0,3584,7041);right=m.canonical_kv_indices([4,10625,-1],2,0,3584,0);self.assertEqual(left,right)
 def test_length_and_invalid_tail(self):
  self.assertEqual(m.canonical_kv_indices([7,-1,9],2,0,3584,0),[('global',7),None,None])
  with self.assertRaises(RuntimeError):m.canonical_kv_indices([1],2,0,3584,0)
 def run_original(self,fail=False):
  log=[];attn=Attention(log);cp=types.SimpleNamespace(global_positions=Vector());observer=object.__new__(m.Observer);observer.cp=lambda *_:cp;observer.config={'l2_attention_production_sha256':{Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
  r=types.SimpleNamespace(problem=None,attn_seen=set(),attn_branch=None,attn_sources={},attn_parameters={},attn_sparse_count=0,events=[])
  r.guard=lambda f:f() if r.problem is None else None
  r.backend=types.SimpleNamespace(point=lambda *a:None,selected_kv=lambda *a:r.attn_seen.update(['L2.attn.selected_KV','L2.attn.selected_valid']))
  qkv=types.SimpleNamespace(qr=Tensor((1,1280)));common=types.SimpleNamespace(cp_on=True,cp_ctx=cp);saved=sys.modules.get(m.parent.ATTENTION);sys.modules[m.parent.ATTENTION]=types.SimpleNamespace(_FLASH_MLA_SPARSE_Q_CHUNK=16384)
  def original(a,*args,**kw):
   q=a._mega_attention.project_q(None,qkv.qr)
   if fail:raise ValueError('original failure')
   a._mega_attention.prefill(q,None,None,None,None);a._prefill_output_all_reduce(Tensor((1,5120)));return Tensor((1,5120))
  token=m._scope.set(r)
  try:
   if fail:
    with self.assertRaises(ValueError):observer.sparse_wrapper(original)(attn,qkv,common,kv=Tensor((4000,1,512)),indices=Tensor((1,640),'torch.int32'),topk_length=None)
   else:observer.sparse_wrapper(original)(attn,qkv,common,kv=Tensor((4000,1,512)),indices=Tensor((1,640),'torch.int32'),topk_length=None)
  finally:
   m._scope.reset(token)
   if saved is None:del sys.modules[m.parent.ATTENTION]
   else:sys.modules[m.parent.ATTENTION]=saved
  self.assertFalse(any(x in attn.__dict__ for x in ['_prefill_output_all_reduce','_project_prefill_q','_prefill_output_proj_into']));self.assertFalse(any(x in attn._mega_attention.__dict__ for x in ['project_q','prefill']));return log,r
 def test_original_order_and_single_calls(self):
  log,r=self.run_original();self.assertEqual(log,['q','core','AR']);self.assertEqual(r.attn_branch,'mega');self.assertTrue(set(m.CORE_POINTS)<=r.attn_seen)
 def test_original_failure_cleans_temporary_shadows(self):self.assertEqual(self.run_original(True)[0],['q'])
if __name__=='__main__':unittest.main()
