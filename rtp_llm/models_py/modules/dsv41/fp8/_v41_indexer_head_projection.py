"""Batch-invariant BF16 indexer head projection for V4.1-Flash.

The small head projection controls a discrete TopK. A BF16 GEMM may change
its reduction with M and round a near-midpoint dot product differently in
prefill and decode. Reduce each row/head with one fixed FP64 tree, then
round the raw projection once to BF16, before the existing head scaling.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _head_projection_kernel(
    x, weight, output,
    X_STRIDE: tl.constexpr, W_STRIDE: tl.constexpr,
    K: tl.constexpr, HEADS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.arange(0, BLOCK)
    values = tl.load(x + row * X_STRIDE + columns, columns < K, other=0).to(tl.float64)
    weights = tl.load(weight + head * W_STRIDE + columns, columns < K, other=0).to(tl.float64)
    projected = tl.sum(values * weights, axis=0)
    tl.store(output + row * HEADS + head, projected)


def try_project_indexer_head_weights(
    x: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor | None:
    """Return raw BF16 weights; preserve unsupported/training implementations."""
    if not (
        x.ndim == weight.ndim == 2
        and x.is_cuda and weight.is_cuda and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[1] == weight.shape[1] == 5120 and weight.shape[0] == 32
        and x.stride(1) == weight.stride(1) == 1
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        and torch.cuda.get_device_capability(x.device)[0] == 10
    ):
        return None
    output = torch.empty((x.shape[0], 32), dtype=torch.bfloat16, device=x.device)
    if x.shape[0]:
        _head_projection_kernel[(x.shape[0], 32)](
            x, weight, output, x.stride(0), weight.stride(0),
            5120, 32, 8192, num_warps=8,
        )
    return output
