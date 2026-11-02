"""Contracts for the new D-only profile, especially inference-mode reset/reuse."""
import importlib.util
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class ContinuationGraph(unittest.TestCase):
 def setup_case(self,suffix):
  import torch
  here=Path(__file__).parent
  latch=load(here/'history38_latches_v1.py','h38_latch_'+suffix)
  trace=load(here/'history38_observer_v1.py','h38_trace_'+suffix)
  output=here/('h38t-'+suffix);self.assertFalse(output.exists())
  latch._config=dict(output=str(output),layers=list(range(40)),positions=[7169,7200],gpu_budget_bytes=4194304,synthetic=True)
  trace._capture=latch
  def control(op,label=''):
   with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
    client.settimeout(20);client.connect(str(output/(str(os.getpid())+'.sock')))
    client.sendall((json.dumps(dict(operation=op,label=label,request_drained=True))+'\n').encode());data=b''
    while not data.endswith(b'\n'):
     part=client.recv(65536);self.assertTrue(part);data+=part
   return json.loads(data)
  return torch,latch,trace,control

 def test_all40_layers_graph_late_positions_and_inference_reset_between_requests(self):
  torch,latch,trace,control=self.setup_case('all40-v1')
  with torch.inference_mode():
   pos=torch.zeros(1,device='cuda',dtype=torch.int64);ids=torch.zeros((1,1),device='cuda',dtype=torch.int64)
   meta=SimpleNamespace(position_ids=pos,valid_requests=torch.ones(1,device='cuda',dtype=torch.bool))
   hidden=torch.zeros((1,1,4,5120),device='cuda',dtype=torch.bfloat16)
   blocks=[SimpleNamespace(layer_id=i,attn=SimpleNamespace(layer_id=i)) for i in range(40)]
   wrapped=trace.block_wrapper(lambda b,x,m,t:x+1,'D')
   def run():
    x=hidden
    for b in blocks:x=wrapped(b,x,meta,ids)
    return x
   for _ in range(3):run()
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=run()
   self.assertTrue(all(torch.is_inference(v[0]) for v in latch._buffers.values()))
   self.assertEqual(len(latch._buffers),82)
   budget=sum(d.numel()*d.element_size()+h.numel()*h.element_size() for d,h in latch._buffers.values())
   self.assertLess(budget,4194304)
   for case in range(2):
    self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET')
    for position in [7169,7200]:
     pos.fill_(position);ids.fill_(position+case);hidden.fill_(case+1);graph.replay()
    torch.cuda.synchronize();result=control('snapshot','case'+str(case));self.assertEqual(result['state'],'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE')
    values=torch.load(result['file'],map_location='cpu',weights_only=True)
    self.assertEqual(len(values),82)
    for position in [7169,7200]:
     self.assertEqual(values[f'D-L0-T{position}-actual_input_token'].item(),position+case)
     for layer in range(40):
      key=f'D-L{layer}-T{position}-block_entry4HC'
      self.assertEqual(result['hits'][key],[1,position]);self.assertTrue(torch.equal(values[key],torch.full_like(values[key],case+1+layer)))
    self.assertTrue(torch.equal(out,torch.full_like(out,case+41)))

 def test_invalid_DP_row_has_no_hits_and_duplicate_execution_fails(self):
  torch,latch,trace,control=self.setup_case('empty-v1')
  with torch.inference_mode():
   p=torch.tensor([7169],device='cuda',dtype=torch.int64);ids=torch.ones((1,1),device='cuda',dtype=torch.int64)
   meta=SimpleNamespace(position_ids=p,valid_requests=torch.zeros(1,device='cuda',dtype=torch.bool))
   h=torch.zeros((1,1,4,5120),device='cuda',dtype=torch.bfloat16);b=SimpleNamespace(layer_id=0,attn=SimpleNamespace(layer_id=0))
   wrapped=trace.block_wrapper(lambda b,x,m,i:x,'D');wrapped(b,h,meta,ids);torch.cuda.synchronize()
   result=control('snapshot','invalid');self.assertTrue(result['coverage_empty'])
   self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET')
   meta.valid_requests.fill_(True);wrapped(b,h,meta,ids);wrapped(b,h,meta,ids);torch.cuda.synchronize()
   bad=control('snapshot','duplicate');self.assertEqual(bad['state'],'CAPTURE_CONTROL_FAILED');self.assertIn('multiple times',bad['error'])

if __name__=='__main__':unittest.main()
