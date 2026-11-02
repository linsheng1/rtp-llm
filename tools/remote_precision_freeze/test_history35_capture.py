"""New history hooks: late graph positions, nested P calls and empty ranks.

Small synthetic contract tests only. No model weights, native requests, or
claims about actual historical numerical values.
"""
import importlib.util
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
        trace = load(here/'history35_observer_v1.py', 'synthetic_history_trace_'+suffix)
        output = here/('h35unit-'+suffix)
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


if __name__ == '__main__':
    unittest.main()
