"""Exact greedy selection from TP-local base logits and rounded Markov bias."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sharded_greedy_partial_kernel(
    B,
    X,
    P,
    S,
    BS: tl.constexpr,
    XS: tl.constexpr,
    WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    WITH_SUMEXP: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    bias = tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    base = tl.load(X + row * XS + i, valid, 0).to(tl.float32)
    value = base + bias
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (value == maximum))
    idx = tl.min(tl.where(wins, i + OFFSET, 2147483647), 0)
    if WITH_SUMEXP:
        # sum(exp(value - block max)); 0 for a block with no finite entry.
        live = valid & ~nan & (value > -float("inf"))
        shifted = tl.where(live, value - maximum, -float("inf"))
        tl.store(S + row * PARTS + part, tl.sum(tl.exp(shifted), 0))
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(P + (row * PARTS + part) * 2, maximum)
    tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


@triton.jit
def _sharded_greedy_finish_kernel(
    P,
    OUT,
    BS: tl.constexpr,
    PARTS: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < PARTS * WORLD
    rank, part = i // PARTS, i % PARTS
    offset = ((rank * BS + row) * PARTS + part) * 2
    value = tl.load(P + offset, valid, -float("inf"))
    idx = tl.load(P + offset + 1, valid, 0).to(tl.int32, bitcast=True)
    # The NVLink push transport changes +0 to -0 as its Lamport sentinel.
    # Indices are nonnegative int32 bits, so strip that sign bit to recover ID 0.
    idx &= 2147483647
    valid &= idx != 2147483647
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (value == maximum))
    result = tl.min(tl.where(wins, idx, 2147483647), 0)
    tl.store(OUT + row, result.to(tl.int64))


@triton.jit
def _top_prob_kernel(
    P,
    S,
    TOP,
    TS,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Softmax top-1 probability exp(max - logsumexp) = 1 / sum_b(s_b * exp(m_b - max)), from one rank's blocks.
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < PARTS
    block_max = tl.load(P + (row * PARTS + i) * 2, valid, -float("inf"))
    block_sum = tl.load(S + row * PARTS + i, valid, 0.0)
    nan = valid & (block_max != block_max)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    live = valid & ~nan & (block_max > -float("inf"))
    maximum = tl.max(tl.where(live, block_max, -float("inf")), 0)
    total = tl.sum(tl.where(live, block_sum * tl.exp(block_max - maximum), 0.0), 0)
    top = tl.where(has_nan | (total <= 0.0), float("nan"), 1.0 / total)
    tl.store(TOP + row * TS, top)


def sharded_greedy_step(
    bias, base_local, *, group, vocab_start, gather=None, top_prob_out=None
):
    """Fused BuildStepLocal + vocab gather + argmax, without materializing logits.

    Equivalent to argmax of rank-ordered ``build_step_local``/all_gather over the
    sharded vocab, excluding padding. The transport carries one (value,
    global-index-bits) pair per 4096-wide block of the shard per row; indices
    move as bits and are never converted numerically to float.

    ``top_prob_out`` ([rows] float32, any stride), when given, receives each row's softmax
    top-1 probability from the same pass (NaN for a row with a NaN logit). The
    per-block sums are not transported, so it needs a single-rank group.
    """
    assert bias.ndim == base_local.ndim == 2
    assert bias.shape[0] == base_local.shape[0]
    assert bias.shape[1] <= base_local.shape[1]
    assert bias.stride(1) == base_local.stride(1) == 1
    with_sumexp = top_prob_out is not None
    assert not with_sumexp or group.world_size == 1
    rows, width = bias.shape
    block = 4096
    parts = triton.cdiv(base_local.shape[1], block)
    assert parts > 0
    partial = torch.empty((rows, parts, 2), device=bias.device, dtype=torch.float32)
    block_sums = (
        torch.empty((rows, parts), device=bias.device, dtype=torch.float32)
        if with_sumexp
        else partial
    )
    _sharded_greedy_partial_kernel[(rows, parts)](
        bias,
        base_local,
        partial,
        block_sums,
        bias.stride(0),
        base_local.stride(0),
        width,
        vocab_start,
        parts,
        block,
        with_sumexp,
        num_warps=4,
    )
    if with_sumexp:
        _top_prob_kernel[(rows,)](
            partial,
            block_sums,
            top_prob_out,
            top_prob_out.stride(0),
            parts,
            triton.next_power_of_2(parts),
            num_warps=1,
        )
    # The padded partition width fixes the transport shape on all ranks;
    # WIDTH masks real entries, including a completely empty final shard.
    if gather is not None:
        gathered = gather(partial.view(rows, parts * 2))
    else:
        gathered = group.all_gather(partial, dim=0) if group.world_size > 1 else partial
    result = torch.empty(rows, device=bias.device, dtype=torch.int64)
    _sharded_greedy_finish_kernel[(rows,)](
        gathered,
        result,
        rows,
        parts,
        group.world_size,
        triton.next_power_of_2(parts * group.world_size),
        num_warps=4,
    )
    return result
