"""New prepared-tuple/P-domain contracts; no production math experiment."""
import importlib.util,json,os,socket
from pathlib import Path
from types import SimpleNamespace
import unittest

def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class PreparedTrajectory(unittest.TestCase):
 def setup_case(self,suffix,role='D'):
  import torch
  here=Path(__file__).parent;latch=load(here/'history38_latches_v1.py','h39_latch_'+suffix);trace=load(here/'history39_observer_v1.py','h39_trace_'+suffix)
  output=here/('h39t-'+suffix);self.assertFalse(output.exists());latch._config=dict(output=str(output),role=role,layers=list(range(40 if role=='P' else 27)),positions=[7168] if role=='P' else [7169],gpu_budget_bytes=4194304,synthetic=True);trace._capture=latch
  def control(operation,label=''):
   with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
    c.settimeout(20);c.connect(str(output/(str(os.getpid())+'.sock')));c.sendall((json.dumps(dict(operation=operation,label=label,request_drained=True))+'\n').encode());data=b''
    while not data.endswith(b'\n'):
     part=c.recv(65536);self.assertTrue(part);data+=part
   return json.loads(data)
  return torch,latch,trace,control
 def test_actual_prepared_inputs_are_independent_of_equal_residual(self):
  torch,latch,trace,control=self.setup_case('prepared-v1')
  with torch.inference_mode():
   pos=torch.zeros(1,device='cuda',dtype=torch.int64);ids=torch.zeros((1,1),device='cuda',dtype=torch.int64);meta=SimpleNamespace(position_ids=pos,valid_requests=torch.ones(1,device='cuda',dtype=torch.bool));hidden=torch.zeros((1,1,4,5120),device='cuda',dtype=torch.bfloat16);norm=torch.zeros((1,1,5120),device='cuda',dtype=torch.bfloat16);pre=torch.ones((1,4),device='cuda');post=torch.full((1,4,1),2.,device='cuda');comb=torch.full((1,4,4),3.,device='cuda')
   blocks=[SimpleNamespace(layer_id=i,attn=SimpleNamespace(layer_id=i),attn_hc=SimpleNamespace(pre_mix_out=pre)) for i in range(27)]
   attention=trace.attention_wrapper(lambda a,x,m:x+1,'_forward_decode_body')
   def original(b,x,m,i,**kw):attention(b.attn,kw['_prepared_attn'][1],m);return x+1
   wrapped=trace.block_wrapper(original,'D')
   def run():
    x=hidden
    for b in blocks:x=wrapped(b,x,meta,ids,_prepared_attn=(x,norm,post,comb))
    return x
   for _ in range(3):run()
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=run()
   self.assertLessEqual(len(latch._buffers),192);self.assertLess(sum(d.numel()*d.element_size()+8 for d,h in latch._buffers.values()),4194304)
   cases=[]
   for case in range(2):
    self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET');pos.fill_(7169);ids.fill_(123);norm.fill_(7+case);pre.fill_(1+case);graph.replay();torch.cuda.synchronize();result=control('snapshot','case'+str(case));self.assertEqual(result['state'],'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE');cases.append(torch.load(result['file'],map_location='cpu',weights_only=True))
   a,b=cases
   for layer in range(27):
    prefix=f'D-L{layer}-T7169-';self.assertTrue(torch.equal(a[prefix+'block_entry4HC'],b[prefix+'block_entry4HC']));self.assertFalse(torch.equal(a[prefix+'prepared_normalized_input5120'],b[prefix+'prepared_normalized_input5120']));self.assertFalse(torch.equal(a[prefix+'prepared_HC_coeff24'],b[prefix+'prepared_HC_coeff24']));self.assertEqual(b[prefix+'actual_attention_input5120'][0,0].item(),8)
   self.assertTrue(torch.equal(out,torch.full_like(out,27)))
 def test_prefill40_layers_only_actual_CP_owner_and_nested_single_hit(self):
  torch,latch,trace,control=self.setup_case('prefill-v1','P')
  with torch.inference_mode():
   pos=torch.tensor([7168,7169],device='cuda',dtype=torch.int64);cp=SimpleNamespace(cp_size=4,kv_cache_sharded=True,global_positions=pos,local_is_real=torch.tensor([True,False],device='cuda'));h=torch.ones((2,4,5120),device='cuda',dtype=torch.bfloat16);ids=torch.tensor([123,456],device='cuda',dtype=torch.int64)
   for layer in range(40):
    block=SimpleNamespace(layer_id=layer,attn=SimpleNamespace(layer_id=layer,_cp_ctx=cp));inner=trace.block_wrapper(lambda b,x,i,p:x,'P');outer=trace.block_wrapper(lambda b,x,i,p:inner(b,x,i,p),'P');outer(block,h,ids,pos)
   torch.cuda.synchronize();result=control('snapshot','prefill');self.assertEqual(result['state'],'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE');self.assertEqual(len(result['hits']),41);self.assertTrue(all(h==[1,7168] for h in result['hits'].values()))
   self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET');self.assertTrue(control('snapshot','empty')['coverage_empty'])
 def test_fused_L25_seam_preserves_natural_values(self):
  torch,latch,trace,control=self.setup_case('seam-v1')
  with torch.inference_mode():
   pos=torch.zeros(1,device='cuda',dtype=torch.int64)
   def unit():return SimpleNamespace(fn=torch.zeros(1,device='cuda'),base=torch.zeros(1,device='cuda'),scale=torch.zeros(1,device='cuda'),pre_mix_out=torch.ones((1,4),device='cuda'))
   class Block:
    def _inject_engram(self,x):return x+1
    def _try_mega_mhc(self,x,r,p,c,**kw):self.ffn_hc.pre_mix_out=self.attn_hc.pre_mix_out+2;return r+2,x+3,p+4,c+5
   trace.patch_seam_block(SimpleNamespace(Block=Block));block=Block();block.layer_id=25;block.attn_hc=unit();block.ffn_hc=unit();block.ffn_norm=SimpleNamespace(weight=torch.ones(5120,device='cuda',dtype=torch.bfloat16));latch._context.value=dict(role='D',layer=25,block=block,positions=pos,attn=SimpleNamespace())
   r=torch.zeros((1,4,5120),device='cuda',dtype=torch.bfloat16);x=torch.ones((1,5120),device='cuda',dtype=torch.bfloat16);p=torch.ones((1,4,1),device='cuda');c=torch.ones((1,4,4),device='cuda')
   def run():return block._try_mega_mhc(x,block._inject_engram(r),p,c)
   run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=run()
   self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET');pos.fill_(7169);graph.replay();torch.cuda.synchronize();result=control('snapshot','seam');self.assertEqual(result['state'],'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE');t=torch.load(result['file'],map_location='cpu',weights_only=True);prefix='D-L25-T7169-';self.assertEqual(t[prefix+'seam_residual4HC'][0,0].item(),1);self.assertEqual(t[prefix+'seam_FFN_norm5120'][0,0].item(),4);self.assertEqual(t[prefix+'FFNHC_pre4'][0,0].item(),3);self.assertTrue(torch.equal(out[0],torch.full_like(r,3)));self.assertTrue(torch.equal(r,torch.zeros_like(r)));latch._context.value=None
if __name__=='__main__':unittest.main()
