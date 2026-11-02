"""Compressor STATE must keep the same FP32 row across prefill/decode shapes."""
import unittest
import torch
from rtp_llm.models_py.modules.dsv41.fp8.compressor import _linear_bf16_bf16_fp32

class CompressorProjectionInvarianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest('SM100 DeepGEMM regression requires CUDA SM100')
        import deep_gemm
        deep_gemm.use_deterministic_algorithms(True)

    def test_same_row_preserves_FP32_across_M_and_projection_fusion(self):
        generator = torch.Generator(device='cpu').manual_seed(7167)
        row = (torch.randn((1, 5120), generator=generator) * .1).bfloat16()
        weight = (torch.randn((1056, 5120), generator=generator) * .1).bfloat16()
        x, w = row.cuda(), weight.cuda()
        with torch.inference_mode():
            one = _linear_bf16_bf16_fp32(x, w[:1024])
            batch = x.expand(65, -1).contiguous()
            many = _linear_bf16_bf16_fp32(batch, w[:1024])
            fused = _linear_bf16_bf16_fp32(x, w)
            self.assertEqual(one.dtype, torch.float32)
            self.assertTrue(torch.equal(one.view(torch.uint8), many[32:33].view(torch.uint8)))
            self.assertTrue(torch.equal(one.view(torch.uint8), fused[:, :1024].contiguous().view(torch.uint8)))
            ref = row.double().matmul(weight[:1024].double().T).float()
            self.assertTrue(torch.allclose(one.cpu(), ref, rtol=1e-5, atol=1e-5))

    def test_preserves_leading_dimensions(self):
        x = torch.zeros((2, 3, 5120), device='cuda', dtype=torch.bfloat16)
        w = torch.ones((512, 5120), device='cuda', dtype=torch.bfloat16)
        out = _linear_bf16_bf16_fp32(x, w)
        self.assertEqual(tuple(out.shape), (2, 3, 512))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(out).item(), 0)

if __name__ == '__main__':
    unittest.main()
