"""FlashMLA index plans with fixed SWA slots before canonical global KV.

The caller still owns per-forward plan caching and CED/layout invalidation.
Unsupported metadata returns None before allocation/launch; execution errors
propagate.
"""

from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in ("triton", "triton.language"):
        raise
    triton = None


_MAX_WIDTH = 2048
_INT32_MAX = 2**31 - 1


@lru_cache(maxsize=16)
def _device_supported(device):
    return torch.cuda.get_device_capability(device)[0] == 10


if triton is not None:

    @triton.jit(do_not_specialize=["OFFSET", "GLOBAL_COUNT", "SWA_START"])
    def _prefill_index_plan_kernel(
        selected,
        positions,
        indices,
        lengths,
        OFFSET,
        GLOBAL_COUNT,
        SWA_START,
        SELECTED_STRIDE: tl.constexpr,
        POSITION_STRIDE: tl.constexpr,
        K: tl.constexpr,
        WINDOW: tl.constexpr,
        PADDED: tl.constexpr,
        BLOCK: tl.constexpr,
        SELECTED_BLOCK: tl.constexpr,
        TENSOR_META: tl.constexpr = False,
        OFFSET_STRIDE: tl.constexpr = 0,
        COUNT_STRIDE: tl.constexpr = 0,
        START_STRIDE: tl.constexpr = 0,
    ):
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, BLOCK)
        selected_column = tl.arange(0, SELECTED_BLOCK)
        if selected.dtype.element_ty == tl.int32:
            sentinel: tl.constexpr = 0x7FFFFFFF
        else:
            sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF
        picked = tl.load(
            selected + row * SELECTED_STRIDE + selected_column,
            selected_column < K,
            other=sentinel,
        )
        # Atomic top-k emission order must not change FlashMLA's reduction order.
        picked = tl.sort(picked, descending=False)
        picked = picked.to(tl.int64)
        position = tl.load(positions + row * POSITION_STRIDE)
        if TENSOR_META:
            offset = tl.load(OFFSET + row * OFFSET_STRIDE).to(tl.int64)
            size = tl.load(GLOBAL_COUNT + row * COUNT_STRIDE).to(tl.int64)
            start = tl.load(SWA_START + row * START_STRIDE).to(tl.int64)
        else:
            offset = OFFSET.to(tl.int64)
            size = GLOBAL_COUNT.to(tl.int64)
            start = SWA_START.to(tl.int64)
        # Match decode's fixed logical slot j even when the workspace omits
        # earlier history. Invalid SWA slots must not shift later columns.
        swpos = tl.maximum(position.to(tl.int64) - WINDOW + 1, 0) + column.to(tl.int64)
        swa_index = tl.where(
            (swpos >= start) & (swpos <= position), offset + size + swpos - start, -1
        ).to(tl.int32)
        tl.store(indices + row * PADDED + column, swa_index, column < WINDOW)
        # Preserve the existing global sort, duplicate and int32-cast semantics,
        # but partition invalids only inside its own fixed K-column region.
        value = tl.where(picked >= 0, offset + picked, -1).to(tl.int32)
        active = selected_column < K
        valid = active & (value >= 0)
        prefix = tl.cumsum(valid.to(tl.int32), axis=0)
        count = tl.sum(valid.to(tl.int32), axis=0)
        destination = tl.where(valid, prefix - 1, count + selected_column - prefix)
        tl.store(indices + row * PADDED + WINDOW + destination, value, active)
        # FlashMLA scans an extent, including holes, rather than a valid count.
        tl.store(
            indices + row * PADDED + column,
            -1,
            (column >= K + WINDOW) & (column < PADDED),
        )
        tl.store(lengths + row, WINDOW + K)

else:
    _prefill_index_plan_kernel = None


def build_index_plan_eager(
    selected, positions, offset, global_count, swa_start, window_size=128
):
    """Build the same fixed-slot layout for unsupported fused metadata/devices."""
    position = positions[:, None].long()
    swpos = (position - window_size + 1).clamp_min(0) + torch.arange(
        window_size, device=selected.device
    )[None]
    swidx = torch.where(
        (swpos >= swa_start) & (swpos <= position),
        offset + global_count + swpos - swa_start,
        -1,
    ).int()
    ordered = selected.sort(dim=-1).values
    global_idx = torch.where(ordered >= 0, offset + ordered, -1).int()
    global_idx = global_idx.gather(
        1, torch.argsort(global_idx < 0, dim=-1, stable=True)
    )
    indices = torch.cat((swidx, global_idx), -1)
    extent = window_size + selected.shape[1]
    lengths = torch.full(
        (selected.shape[0],), extent, dtype=torch.int32, device=selected.device
    )
    if extent % 64:
        indices = torch.nn.functional.pad(indices, (0, 64 - extent % 64), value=-1)
    return indices, lengths


def is_supported(selected, positions, offset, global_count, swa_start, window_size):
    """Accept scalar metadata or the caller's per-row [M,1] integer views."""
    if not (
        _prefill_index_plan_kernel is not None
        and torch.version.hip is None
        and type(window_size) is int
        and 1 <= window_size <= _MAX_WIDTH
        and selected.is_cuda
        and selected.dtype in (torch.int32, torch.int64)
        and selected.ndim == 2
        and 1 <= selected.shape[1] <= _MAX_WIDTH - window_size
        and selected.stride(1) == 1
        and selected.stride(0) >= selected.shape[1]
        and positions.device == selected.device
        and positions.ndim == 1
        and positions.shape[0] == selected.shape[0]
        and positions.dtype in (torch.int32, torch.int64)
        and positions.stride(0) > 0
    ):
        return False
    metadata = (offset, global_count, swa_start)
    if all(type(value) is int for value in metadata):
        if not (
            0 <= offset <= _INT32_MAX
            and 0 <= global_count <= _INT32_MAX - offset
            and -(2**31) <= swa_start <= _INT32_MAX
        ):
            return False
    elif not all(
        isinstance(value, torch.Tensor)
        and value.device == selected.device
        and value.dtype in (torch.int32, torch.int64)
        and value.shape == (selected.shape[0], 1)
        and value.stride(0) >= 0
        for value in metadata
    ):
        return False
    return _device_supported(selected.device)


def try_build_index_plan(
    selected,
    positions,
    offset,
    global_count,
    swa_start,
    window_size=128,
    *,
    out=None,
    lengths_out=None,
):
    """Return contiguous int32 (indices[M,pad64(K+window)], lengths[M]).

    Metadata follows _prefill_chunk_meta's scalar or per-row [M,1] contract.
    Global selected IDs are sorted; duplicates are retained. Negative selected
    entries are normalized to -1 before offsetting. SWA occupies [0,window)
    with decode's max(position-window+1,0)+j slots, masking missing/future rows
    in place. Global entries occupy the following K columns; lengths is the
    scan extent window+K, excluding pad64 alignment, even when slots are holes.
    Optional outputs avoid allocation during graph capture/replay. No tensor
    is cached, synchronized, or retained by the module.
    """
    if not is_supported(
        selected, positions, offset, global_count, swa_start, window_size
    ):
        return None
    rows, k = selected.shape
    padded = ((k + window_size + 63) // 64) * 64
    for output, shape in ((out, (rows, padded)), (lengths_out, (rows,))):
        if output is not None and not (
            output.device == selected.device
            and output.dtype == torch.int32
            and output.shape == shape
            and output.is_contiguous()
        ):
            return None
    # Reject any output sharing storage with an input or the other output.
    # Storage metadata is host-visible and never requires a CUDA synchronize.
    provided = [x for x in (out, lengths_out) if x is not None and x.numel()]
    sources = [x for x in (selected, positions) if x.numel()]
    sources.extend(
        x
        for x in (offset, global_count, swa_start)
        if isinstance(x, torch.Tensor) and x.numel()
    )
    storage = {x.untyped_storage().data_ptr() for x in sources}
    for output in provided:
        pointer = output.untyped_storage().data_ptr()
        if pointer in storage:
            return None
        storage.add(pointer)
    if out is None:
        out = torch.empty((rows, padded), dtype=torch.int32, device=selected.device)
    if lengths_out is None:
        lengths_out = torch.empty((rows,), dtype=torch.int32, device=selected.device)
    if rows:
        _prefill_index_plan_kernel[(rows,)](
            selected,
            positions,
            out,
            lengths_out,
            offset,
            global_count,
            swa_start,
            selected.stride(0),
            positions.stride(0),
            k,
            window_size,
            padded,
            triton.next_power_of_2(padded),
            triton.next_power_of_2(k),
            **(
                {
                    "TENSOR_META": True,
                    "OFFSET_STRIDE": offset.stride(0),
                    "COUNT_STRIDE": global_count.stride(0),
                    "START_STRIDE": swa_start.stride(0),
                }
                if isinstance(offset, torch.Tensor)
                else {}
            ),
            num_warps=4,
        )
    return out, lengths_out
