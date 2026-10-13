"""Python wrapper of the fused learned block head (csrc/moe/expert_stream/spec_head.cuh).

Offline use plus the opt-in serving path behind ``SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD``
(``run_spec_head_serving``). It is its own JIT module (``expert_stream_spec_head``), so the
production expert-stream device module's build key is untouched.

``run_spec_head`` launches K1-K4 (PCA + fc_in, gated middle block, fc_out, fused tail) on the
current stream of ``x``'s device; ``run_spec_head_tail`` launches only the tail from logits.
The result is one candidate record (spec_candidates.h convention): count 0 or 1, the expert
and its block score; ``read_record`` decodes it on the host.
"""

from __future__ import annotations

import struct

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe.expert_stream_transport import CAND_STRIDE

HEAD_MAX_TOKENS = 8  # spec_head_tail.cuh kHeadMaxTokens
HEAD_EXPERTS = 384
DEFAULT_PCA_RANK = 256  # b1_head.HeadConfig defaults
DEFAULT_HIDDEN = 512

# spec_candidates.h: u32 seq, u16 count, u16 flags, then entry 0 = u16 expert, u8 rank, pad, f32 margin.
_RECORD_FMT = "<xxxxHxxHBxf"


class SpecHeadWeights(msgspec.Struct, frozen=True):
    """Device weights in the argument order of ``SpecHeadKernel::run``.

    bf16 weights and fp32 biases; ``pos_bias`` and the ``agg_*`` tensors are empty ([0]) for
    heads without them (T0 has no aggregator).
    """

    w_pca: torch.Tensor
    w_in_pca: torch.Tensor
    b_in: torch.Tensor
    w_in_sparse: torch.Tensor
    row_bias: torch.Tensor
    pos_bias: torch.Tensor
    w_r1: torch.Tensor
    b_r1: torch.Tensor
    w_r2: torch.Tensor
    b_r2: torch.Tensor
    w_gate: torch.Tensor
    b_gate: torch.Tensor
    w_out: torch.Tensor
    b_out: torch.Tensor
    agg_w1: torch.Tensor
    agg_row_bias: torch.Tensor
    agg_w2: torch.Tensor
    agg_b2: torch.Tensor
    rows: int


class SpecHeadWorkspace(msgspec.Struct, frozen=True):
    """Scratch and output buffers a launch writes; fp32 between kernels."""

    h: torch.Tensor
    pre: torch.Tensor
    hid: torch.Tensor
    logits: torch.Tensor
    block_scores: torch.Tensor
    features: torch.Tensor
    record: torch.Tensor


@cache_once
def _module():
    return load_jit(
        "expert_stream_spec_head",
        cuda_files=["moe/expert_stream/spec_head.cuh"],
        cuda_wrappers=[
            ("spec_head_run", "SpecHeadKernel::run"),
            ("spec_head_tail", "SpecHeadKernel::tail"),
            ("spec_head_serve", "SpecHeadKernel::serve"),
            ("spec_head_serve_empty", "SpecHeadKernel::serve_empty"),
        ],
    )


def new_workspace(
    *,
    device: torch.device | str,
    features: bool = False,
    pca_rank: int = DEFAULT_PCA_RANK,
    hidden: int = DEFAULT_HIDDEN,
    experts: int = HEAD_EXPERTS,
) -> SpecHeadWorkspace:
    """Allocate the buffers for one head; ``features`` also keeps the tail's [experts, 10] block features."""
    f32 = dict(dtype=torch.float32, device=device)
    return SpecHeadWorkspace(
        h=torch.zeros(HEAD_MAX_TOKENS, pca_rank, **f32),
        pre=torch.zeros(HEAD_MAX_TOKENS, hidden, **f32),
        hid=torch.zeros(HEAD_MAX_TOKENS, hidden, **f32),
        logits=torch.zeros(HEAD_MAX_TOKENS, experts, **f32),
        block_scores=torch.zeros(experts, **f32),
        features=torch.zeros(experts, 10, **f32) if features else torch.empty(0, **f32),
        # torch's allocator aligns well past 128 bytes.
        record=torch.zeros(CAND_STRIDE, dtype=torch.uint8, device=device),
    )


def run_spec_head(
    *,
    weights: SpecHeadWeights,
    workspace: SpecHeadWorkspace,
    x: torch.Tensor,
    scores: torch.Tensor,
    src_ids: torch.Tensor,
    src_w: torch.Tensor,
    positions: torch.Tensor,
    row: int,
    hot_slots: torch.Tensor,
    hot_capacity: int,
    ram_slot: torch.Tensor,
    target: int,
    threshold: torch.Tensor,
) -> None:
    """K1-K4 on the current stream: x bf16 [tokens, H], scores fp32 [tokens_max, E], routes int64/fp32 [tokens, K].

    ``row`` indexes the per-(source, target) bias tables, ``target`` is the ``ram_slot`` row of the
    target layer, ``threshold`` is fp32 [1] on the device.
    """
    w, ws = weights, workspace
    _module().spec_head_run(
        x, scores, src_ids, src_w, positions,
        w.w_pca, w.w_in_pca, w.b_in, w.w_in_sparse, w.row_bias, w.pos_bias,
        w.w_r1, w.b_r1, w.w_r2, w.b_r2, w.w_gate, w.b_gate, w.w_out, w.b_out,
        w.agg_w1, w.agg_row_bias, w.agg_w2, w.agg_b2,
        int(row), hot_slots, int(hot_capacity), ram_slot, int(target), threshold,
        ws.h, ws.pre, ws.hid, ws.logits, ws.block_scores, ws.features, ws.record,
    )  # fmt: skip


def run_spec_head_tail(
    *,
    weights: SpecHeadWeights,
    workspace: SpecHeadWorkspace,
    logits: torch.Tensor,
    positions: torch.Tensor,
    row: int,
    hot_slots: torch.Tensor,
    hot_capacity: int,
    ram_slot: torch.Tensor,
    target: int,
    threshold: torch.Tensor,
) -> None:
    """K4 alone: fp32 per-token ``logits`` [tokens, E] to the block score, skips, threshold and cap-1 pick."""
    w, ws = weights, workspace
    _module().spec_head_tail(
        logits, positions, w.agg_w1, w.agg_row_bias, w.agg_w2, w.agg_b2,
        int(row), hot_slots, int(hot_capacity), ram_slot, int(target), threshold,
        ws.block_scores, ws.features, ws.record,
    )  # fmt: skip


def run_spec_head_serving(
    *,
    weights: SpecHeadWeights,
    workspace: SpecHeadWorkspace,
    x: torch.Tensor | None,
    tokens: int,
    scores: torch.Tensor,
    src_ids: torch.Tensor,
    src_w: torch.Tensor,
    positions: torch.Tensor,
    row: int,
    hot_slots: torch.Tensor,
    hot_capacity: int,
    ram_slot: torch.Tensor,
    target: int,
    threshold: torch.Tensor,
    state: torch.Tensor,
    candidates: torch.Tensor,
) -> None:
    """K1-K4 publishing into the post's candidate-page slot, on the current stream; graph-capturable.

    The slot is ``(state[kPosted] - 1) % kCandRecords`` of the pinned uint8 ``candidates`` page, read
    on the device, with the seq stored last. ``x`` None or ``tokens`` outside 1..HEAD_MAX_TOKENS
    publishes count 0 with the oversize flag. ``src_ids``/``src_w``/``positions`` lead with ``tokens`` rows.
    """
    module = _module()
    address = int(candidates.data_ptr())
    if x is None or not 1 <= tokens <= HEAD_MAX_TOKENS:
        module.spec_head_serve_empty(state, address)
        return
    w, ws = weights, workspace
    module.spec_head_serve(
        x[:tokens], scores, src_ids[:tokens], src_w[:tokens], positions[:tokens],
        w.w_pca, w.w_in_pca, w.b_in, w.w_in_sparse, w.row_bias, w.pos_bias,
        w.w_r1, w.b_r1, w.w_r2, w.b_r2, w.w_gate, w.b_gate, w.w_out, w.b_out,
        w.agg_w1, w.agg_row_bias, w.agg_w2, w.agg_b2,
        int(row), hot_slots, int(hot_capacity), ram_slot, int(target), threshold,
        ws.h, ws.pre, ws.hid, ws.logits, ws.block_scores, ws.features,
        state, address,
    )  # fmt: skip


def read_record(workspace: SpecHeadWorkspace) -> tuple[int, int, float]:
    """(count, expert or -1, block score) of the last launch; synchronizes the record's device."""
    raw = bytes(workspace.record.cpu().tolist())
    count, expert, _rank, score = struct.unpack_from(_RECORD_FMT, raw)
    if count == 0:
        return 0, -1, 0.0
    return int(count), int(expert), float(score)
