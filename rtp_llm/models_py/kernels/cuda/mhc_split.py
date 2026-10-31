"""Token-independent split-K selection for mHC prenorm DeepGEMM calls.

Changing the split count changes FP32 accumulation order for the same row.
Use one count per device/K geometry in serving; retain the historical policy
as an explicit diagnostic option. CUDA imports are lazy so warmup planning
with an explicit SM count can run without initializing a device.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import torch

_BLOCK_M = 64
_BLOCK_K = 64
_REFERENCE_TOKENS = 192
_TOKEN_MODES = frozenset(("token", "legacy", "dynamic"))


@lru_cache(maxsize=None)
def _device_num_sms(index: int) -> int:
    import torch

    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def _resolve_num_sms(device, num_sms: Optional[int]) -> int:
    if num_sms is not None:
        return max(int(num_sms), 1)
    if isinstance(device, int):
        index = device
    else:
        import torch

        index = None if device is None else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    return max(_device_num_sms(index), 1)


def mhc_prenorm_num_splits(
    k: int,
    *,
    block_k: int = _BLOCK_K,
    tokens: Optional[int] = None,
    grid_size: Optional[int] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    num_sms: Optional[int] = None,
) -> int:
    """Select split-K; default fixed50 for 152 SMs and K=20480.

    ``DSV4_MHC_SPLIT_MODE=token`` opts into the old T-dependent policy.
    ``DSV4_MHC_NUM_SPLITS`` pins an explicit positive count in either mode.
    Fixed selection uses the legacy three-M-tile branch (T=129..192), avoiding
    the extra CTA wave of fixed76 while remaining independent of actual T.
    Device SM properties alone are cached; environment-sensitive results are
    not cached, so warmup and serving always evaluate the same policy.
    """
    k, block_k = int(k), int(block_k)
    if k <= 0 or block_k <= 0:
        raise ValueError(f"mHC K and block_k must be positive, got {k}, {block_k}")
    k_blocks = (k + block_k - 1) // block_k
    override = os.environ.get("DSV4_MHC_NUM_SPLITS", "").strip()
    if override:
        try:
            splits = int(override)
        except ValueError as error:
            raise ValueError(
                "DSV4_MHC_NUM_SPLITS must be a positive integer"
            ) from error
        if splits <= 0:
            raise ValueError("DSV4_MHC_NUM_SPLITS must be a positive integer")
        return min(splits, k_blocks)

    mode = os.environ.get("DSV4_MHC_SPLIT_MODE", "").strip().lower()
    if mode in _TOKEN_MODES:
        if grid_size is None:
            if tokens is None:
                raise ValueError("token split policy needs tokens or grid_size")
            grid_size = (max(int(tokens), 1) + _BLOCK_M - 1) // _BLOCK_M
        m_tiles = max(int(grid_size), 1)
    else:
        m_tiles = (_REFERENCE_TOKENS + _BLOCK_M - 1) // _BLOCK_M
    return max(min(_resolve_num_sms(device, num_sms) // m_tiles, k_blocks // 4), 1)
