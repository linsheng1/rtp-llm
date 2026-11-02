"""New history hooks: late graph positions, nested P calls and empty ranks.

Small synthetic contract tests only. No model weights, native requests, or
claims about actual historical numerical values.
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HistoryGraphContract(unittest.TestCase):
    def setup_case(self, suffix):
        import torch
        here = Path(__file__).parent
        latch = load(here/'history35_latches_v1.py', 'synthetic_history_latch_'+suffix)
        trace = load(here/'history36_observer_v2.py', 'synthetic_history_trace_'+suffix)
        output = here/('h36t-'+hashlib.sha256(Path(__file__).name.encode()).hexdigest()[:6]+'-'+suffix)
        self.assertFalse(output.exists())
        latch._config = dict(output=str(output), layers=[0,1,2], positions=[7042,7043],
                             gpu_budget_bytes=4*1024**2, synthetic=True)
        trace._capture = latch
        def control(operation, label=''):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(20)
                client.connect(str(output/(str(os.getpid())+'.sock')))
                client.sendall((json.dumps(dict(operation=operation, label=label, request_drained=True))+'\n').encode())
                data = b''
                while not data.endswith(b'\n'):
                    part = client.recv(65536)
                    self.assertTrue(part)
                    data += part
            return json.loads(data)
        return torch, latch, trace, control

    def test_late_decode_graph_keeps_actual_positions_and_ids(self):
        torch, latch, trace, control = self.setup_case('d')
        positions = torch.zeros(1, device='cuda', dtype=torch.int64)
        meta = SimpleNamespace(position_ids=positions, valid_requests=torch.ones(1,device='cuda',dtype=torch.bool))
        hidden = torch.zeros((1,1,4,5120),device='cuda',dtype=torch.bfloat16)
        ids = torch.zeros((1,1),device='cuda',dtype=torch.int64)
        attn = SimpleNamespace(layer_id=0)
        block = SimpleNamespace(layer_id=0, attn=attn)
        body = trace.attention_wrapper(lambda attn,x,metadata:x+1, '_forward_decode_body')
        def original(block, x, metadata, input_ids):
            return body(block.attn, x[:,:,0,:], metadata)
        wrapped = trace.block_wrapper(original, 'D')
        for _ in range(3):
            wrapped(block,hidden,meta,ids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapped(block,hidden,meta,ids)
        self.assertEqual(control('reset')['state'], 'CAPTURE_LATCHES_RESET')
        positions.fill_(7043)
        ids.fill_(12345)
        hidden.fill_(7)
        graph.replay()
        torch.cuda.synchronize()
        result = control('snapshot','late')
        self.assertEqual(result['state'],'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE')
        tensors = torch.load(result['file'],weights_only=True,map_location='cpu')
        self.assertEqual(tensors['D-L0-T7043-actual_input_token'].tolist(),[[12345]])
        self.assertEqual(tensors['D-L0-T7043-attn_input5120'][0,0].item(),7)
        self.assertEqual(tensors['D-L0-T7043-attn_output5120'][0,0].item(),8)
        self.assertTrue(all(h==[1,7043] for h in result['hits'].values()))
        graph.replay()
        torch.cuda.synchronize()
        duplicate = control('snapshot','duplicate')
        self.assertEqual(duplicate['state'],'CAPTURE_CONTROL_FAILED')
        self.assertIn('multiple times',duplicate['error'])

    def test_nested_prefill_is_single_hit_and_empty_rank_is_explicit(self):
        torch, latch, trace, control = self.setup_case('p')
        pos = torch.tensor([7042,7043],device='cuda',dtype=torch.int64)
        cp = SimpleNamespace(cp_size=4,kv_cache_sharded=True,global_positions=pos,
                              local_is_real=torch.ones(2,device='cuda',dtype=torch.bool))
        block = SimpleNamespace(layer_id=0,attn=SimpleNamespace(layer_id=0,_cp_ctx=cp))
        hidden = torch.zeros((2,4,5120),device='cuda',dtype=torch.bfloat16)
        ids = torch.tensor([10,11],device='cuda',dtype=torch.int64)
        inner = trace.block_wrapper(lambda b,x,token,positions:x, 'P')
        outer = trace.block_wrapper(lambda b,x,token,positions:inner(b,x,token,positions), 'P')
        outer(block,hidden,ids,pos)
        torch.cuda.synchronize()
        result = control('snapshot','nested')
        self.assertFalse(result['coverage_empty'])
        self.assertEqual(sorted(result['hits'].values()),[[1,7042],[1,7042],[1,7043],[1,7043]])
        self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET')
        empty = control('snapshot','empty')
        self.assertEqual(empty['state'],'CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE')
        self.assertTrue(empty['coverage_empty'])
        self.assertEqual(torch.load(empty['file'],weights_only=True,map_location='cpu'),{})


    def test_natural_fused_seam_preserves_values_and_late_graph_position(self):
        torch,latch,trace,control=self.setup_case('seam')
        pos=torch.zeros(1,device='cuda',dtype=torch.int64)
        def unit():
            return SimpleNamespace(fn=torch.zeros(1,device='cuda'),base=torch.zeros(1,device='cuda'),scale=torch.zeros(1,device='cuda'),pre_mix_out=torch.ones((1,4),device='cuda'))
        class Block:
            def _inject_engram(self,x):return x+1
            def _try_mega_mhc(self,x,r,p,c,**kwargs):
                self.ffn_hc.pre_mix_out=self.attn_hc.pre_mix_out+2
                return r+2,x+3,p+4,c+5
        trace.patch_seam_block(SimpleNamespace(Block=Block))
        block=Block();block.layer_id=1;block.attn_hc=unit();block.ffn_hc=unit();block.ffn_norm=SimpleNamespace(weight=torch.ones(5120,device='cuda',dtype=torch.bfloat16))
        latch._context.value=dict(role='D',layer=1,block=block,positions=pos,attn=SimpleNamespace())
        r=torch.zeros((1,4,5120),device='cuda',dtype=torch.bfloat16);x=torch.ones((1,5120),device='cuda',dtype=torch.bfloat16);p=torch.ones((1,4,1),device='cuda');c=torch.ones((1,4,4),device='cuda')
        def run():
            injected=block._inject_engram(r)
            return block._try_mega_mhc(x,injected,p,c)
        run();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=run()
        self.assertEqual(control('reset')['state'],'CAPTURE_LATCHES_RESET')
        pos.fill_(7043);graph.replay();torch.cuda.synchronize();data=control('snapshot','fused');t=torch.load(data['file'],weights_only=True,map_location='cpu')
        prefix='D-L1-T7043-'
        self.assertTrue(all(v==[1,7043] for v in data['hits'].values()))
        self.assertEqual(t[prefix+'after_Engram4HC'][0,0].item(),1)
        self.assertEqual(t[prefix+'seam_residual4HC'][0,0].item(),1)
        self.assertEqual(t[prefix+'seam_fused_route'].item(),1)
        self.assertEqual(t[prefix+'seam_post4HC'][0,0].item(),3)
        self.assertEqual(t[prefix+'seam_FFN_norm5120'][0,0].item(),4)
        self.assertTrue(torch.equal(result[0],torch.full_like(r,3)))
        self.assertTrue(torch.equal(r,torch.zeros_like(r)))
        self.assertEqual(t[prefix+'seam_actual_rows'].item(),1)
        control('reset')
        latch._context.value['positions']=torch.tensor([7042,7043],device='cuda',dtype=torch.int64)
        block.attn_hc.pre_mix_out=torch.ones((2,4),device='cuda')
        block._try_mega_mhc(x.repeat(2,1),r.repeat(2,1,1),p.repeat(2,1,1),c.repeat(2,1,1))
        torch.cuda.synchronize();data=control('snapshot','different-M');t=torch.load(data['file'],weights_only=True,map_location='cpu')
        self.assertEqual(t[prefix+'seam_actual_rows'].item(),2)
        latch._context.value=None

    def test_natural_unfused_hc_and_norm_capture_only_l1(self):
        torch,latch,trace,control=self.setup_case('fallback')
        pos=torch.tensor([7043],device='cuda',dtype=torch.int64)
        class HC:
            def _pre_impl(self,x):
                self.pre_mix_out=torch.ones((1,4),device=x.device)
                return x[:,0,:],torch.ones((1,4,1),device=x.device),torch.ones((1,4,4),device=x.device)
        class Norm:
            def forward(self,x):return x+2
        trace.patch_seam_hc(SimpleNamespace(DelayedHCUnit=HC));trace.patch_seam_norm(SimpleNamespace(RMSNorm=Norm))
        a,f=HC(),HC()
        for u in [a,f]:u.fn=u.base=u.scale=torch.zeros(1,device='cuda')
        n=Norm();n.weight=torch.ones(5120,device='cuda',dtype=torch.bfloat16)
        block=SimpleNamespace(attn_hc=a,ffn_hc=f,ffn_norm=n)
        latch._context.value=dict(role='D',layer=1,block=block,positions=pos,attn=SimpleNamespace())
        r=torch.ones((1,4,5120),device='cuda',dtype=torch.bfloat16)
        y=f._pre_impl(r)[0];out=n.forward(y);torch.cuda.synchronize()
        data=control('snapshot','fallback');t=torch.load(data['file'],weights_only=True,map_location='cpu');prefix='D-L1-T7043-'
        self.assertEqual(t[prefix+'natural_FFNHC_residual4HC'][0,0].item(),1)
        self.assertEqual(t[prefix+'natural_FFNHC_pre4'][0,0].item(),1)
        self.assertEqual(t[prefix+'natural_FFN_norm_input5120'][0,0].item(),1)
        self.assertEqual(t[prefix+'natural_FFN_norm_output5120'][0,0].item(),3)
        self.assertTrue(torch.equal(out,torch.full_like(y,3)))
        latch._context.value=None


if __name__ == '__main__':
    unittest.main()
