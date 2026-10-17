"""DSpark's sampled (temperature > 0) chain accept for the cases the classic kernel (reject_sampling.py) does not
cover: a one-hot draft distribution, which greedy drafting of a sampled request (SGLANG_DSPARK_SAMPLED_GREEDY_DRAFT)
produces.

The draft proposed X_i = argmax, so q_i is a point mass at X_i: X_i is accepted with probability p_i(X_i), and a
rejection samples the residual normalize(max(0, p_i - q_i)), which is p_i with X_i removed. Neither needs q in memory.

Rows past ``verify_lens - 1`` drafts were never verified (a confidence cut's dead rows): their target logits, draft
probabilities and candidates are not read, and the chain ends there, the bonus coming from p at the last live row.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs

_BLOCK_V = 4096


def sampled_verify_target_probs(
    *,
    target_logits: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    verify_num_draft_tokens: int,
    bs: int,
) -> torch.Tensor:
    """[bs, verify_num_draft_tokens, vocab] target p: temperature, then top-k/top-p when the batch asks for them (the
    same construction as the classic accept's)."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp

    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        return SoftmaxTemp.execute(
            logits=target_logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=verify_num_draft_tokens,
        ).view(bs, verify_num_draft_tokens, -1)
    return build_dflash_verify_target_probs(
        next_token_logits=target_logits,
        sampling_info=sampling_info,
        draft_token_num=verify_num_draft_tokens,
        bs=bs,
        max_top_k=draft_input.max_top_k,
        uniform_top_k_value=draft_input.uniform_top_k_value,
    )


class AcceptSampledChain:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: Optional[torch.Tensor],
        sampling_info,
        draft_input: DFlashDraftInputV2,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _accept_sampled_chain(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            chain_accept=sampled_chain_accept,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: Optional[torch.Tensor],
        sampling_info,
        draft_input: DFlashDraftInputV2,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _accept_sampled_chain(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            chain_accept=sampled_chain_accept_triton,
        )


def _accept_sampled_chain(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    sampling_info,
    draft_input: DFlashDraftInputV2,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
    chain_accept,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(correct_len int32, bonus int64, cap_trim_lens int32). A cut ends the chain itself, so nothing is trimmed after
    the fact and cap_trim_lens is zero."""
    bs = candidates.shape[0]
    target_probs = sampled_verify_target_probs(
        target_logits=target_logits,
        sampling_info=sampling_info,
        draft_input=draft_input,
        verify_num_draft_tokens=verify_num_draft_tokens,
        bs=bs,
    )
    # Columns 0..S-2 are the per-draft coins, column S-1 the bonus sample's.
    uniforms = torch.rand(
        (bs, verify_num_draft_tokens), dtype=torch.float32, device=candidates.device
    )
    correct_len, bonus = chain_accept(
        candidates=candidates,
        target_probs=target_probs,
        draft_probs=draft_probs,
        verify_lens=cutoff_verify_lens,
        uniforms=uniforms,
    )
    return correct_len, bonus, torch.zeros_like(correct_len)


def _live_drafts(verify_lens: Optional[torch.Tensor], bs: int, num_slots: int, device):
    if verify_lens is None:
        return torch.full((bs,), num_slots - 1, dtype=torch.int64, device=device)
    return (verify_lens.to(device=device, dtype=torch.int64) - 1).clamp(0, num_slots - 1)


def _inverse_cdf(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per row, the first index whose running sum exceeds u * total; the last positive index if rounding leaves none."""
    cums = weights.double().cumsum(-1)
    target = u.double() * cums[:, -1]
    hit = cums > target[:, None]
    vocab = weights.shape[-1]
    idx = torch.arange(vocab, device=weights.device)
    first_hit = torch.where(hit, idx, vocab).min(-1).values
    last_pos = torch.where(weights > 0, idx, -1).max(-1).values
    return torch.where(first_hit < vocab, first_hit, last_pos.clamp_min(0))


def sampled_chain_accept(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    verify_lens: Optional[torch.Tensor],
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference. ``candidates`` [bs, S] (slot 0 the anchor), ``target_probs`` [bs, S, V], ``draft_probs``
    [bs, S - 1, V] or None for a one-hot draft at each candidate, ``uniforms`` [bs, S]."""
    correct_len, weights = sampled_chain_bonus_weights(
        candidates=candidates,
        target_probs=target_probs,
        draft_probs=draft_probs,
        verify_lens=verify_lens,
        uniforms=uniforms,
    )
    bonus = _inverse_cdf(weights, uniforms[:, -1])
    return correct_len.to(torch.int32), bonus.to(torch.int64)


def sampled_chain_bonus_weights(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    verify_lens: Optional[torch.Tensor],
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's correct_len [bs] and the unnormalized [bs, V] distribution its bonus is drawn from."""
    bs, num_slots, vocab = target_probs.shape
    device = target_probs.device
    target_probs = target_probs.float()
    live_drafts = _live_drafts(verify_lens, bs, num_slots, device)
    drafts = candidates[:, 1:].to(torch.int64)
    p_x = target_probs[:, :-1].gather(-1, drafts[..., None]).squeeze(-1)
    if draft_probs is None:
        q_x = torch.ones_like(p_x)
        q_rows = torch.zeros_like(target_probs[:, :-1]).scatter_(-1, drafts[..., None], 1.0)
    else:
        draft_probs = draft_probs.float()
        q_x = draft_probs.gather(-1, drafts[..., None]).squeeze(-1)
        q_rows = torch.where(
            (draft_probs >= 0) & (draft_probs <= 1), draft_probs, torch.zeros_like(draft_probs)
        )
    q_ok = (q_x > 0) & (q_x <= 1)
    steps = torch.arange(1, num_slots, device=device)
    live = steps[None, :] <= live_drafts[:, None]
    coins = uniforms[:, : num_slots - 1]

    accept = live & q_ok & (coins * q_x < p_x)
    correct_len = accept.to(torch.int64).cumprod(-1).sum(-1)

    rows = torch.arange(bs, device=device)
    p_row = target_probs[rows, correct_len]
    stop_row = correct_len.clamp_max(num_slots - 2)
    residual = (p_row - q_rows[rows, stop_row]).clamp_min(0)
    weights = torch.where((correct_len < live_drafts)[:, None], residual, p_row)
    # A residual with no mass (p == q on the rejected row) falls back to p.
    weights = torch.where((weights.sum(-1) > 0)[:, None], weights, p_row)
    return correct_len, weights


@triton.jit
def _chain_residual_partials_kernel(
    cand_ptr,
    tp_ptr,
    dp_ptr,
    live_drafts_ptr,
    residual_partials_ptr,
    p_partials_ptr,
    stride_cand_b,
    stride_tp_b,
    stride_tp_s,
    stride_dp_b,
    stride_dp_s,
    vocab,
    num_blocks,
    NUM_SLOTS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ONE_HOT_Q: tl.constexpr,
):
    """Per (request, live row, vocab block): the block's mass of the row's residual max(0, p - q) (the last live row,
    which has no draft after it, uses p), and of p itself for a residual with no mass."""
    pid_row = tl.program_id(0)
    blk = tl.program_id(1)
    b = pid_row // NUM_SLOTS
    row = pid_row % NUM_SLOTS
    live_drafts = tl.load(live_drafts_ptr + b).to(tl.int32)
    row_live = row <= live_drafts
    is_draft_row = row < live_drafts

    offs = blk * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = (offs < vocab) & row_live
    p = tl.load(
        tp_ptr + b.to(tl.int64) * stride_tp_b + row * stride_tp_s + offs,
        mask=vmask,
        other=0.0,
    ).to(tl.float32)
    if ONE_HOT_Q:
        nxt = tl.load(
            cand_ptr + b.to(tl.int64) * stride_cand_b + row + 1,
            mask=is_draft_row,
            other=-1,
        )
        q = tl.where(offs == nxt, 1.0, 0.0)
    else:
        q = tl.load(
            dp_ptr + b.to(tl.int64) * stride_dp_b + row * stride_dp_s + offs,
            mask=vmask & is_draft_row,
            other=0.0,
        ).to(tl.float32)
        q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
    val = tl.where(is_draft_row, tl.maximum(p - q, 0.0), p)
    out = pid_row * num_blocks + blk
    tl.store(residual_partials_ptr + out, tl.sum(val, axis=0), mask=row_live)
    tl.store(p_partials_ptr + out, tl.sum(p, axis=0), mask=row_live)


@triton.jit
def _chain_accept_kernel(
    cand_ptr,
    tp_ptr,
    dp_ptr,
    live_drafts_ptr,
    residual_partials_ptr,
    p_partials_ptr,
    uniform_ptr,
    correct_len_ptr,
    bonus_ptr,
    stride_cand_b,
    stride_tp_b,
    stride_tp_s,
    stride_dp_b,
    stride_dp_s,
    stride_uni_b,
    vocab,
    num_blocks,
    NUM_SLOTS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NB_PAD: tl.constexpr,
    ONE_HOT_Q: tl.constexpr,
):
    b = tl.program_id(0)
    b64 = b.to(tl.int64)
    cand_base = cand_ptr + b64 * stride_cand_b
    tp_base = tp_ptr + b64 * stride_tp_b
    dp_base = dp_ptr + b64 * stride_dp_b
    uni_base = uniform_ptr + b64 * stride_uni_b
    live_drafts = tl.load(live_drafts_ptr + b).to(tl.int32)

    correct_len = live_drafts * 0
    going = live_drafts >= 0
    for t in tl.static_range(1, NUM_SLOTS):
        live = t <= live_drafts
        x = tl.load(cand_base + t, mask=live, other=0)
        p = tl.load(tp_base + (t - 1) * stride_tp_s + x, mask=live, other=0.0).to(
            tl.float32
        )
        if ONE_HOT_Q:
            q = 1.0
        else:
            q = tl.load(dp_base + (t - 1) * stride_dp_s + x, mask=live, other=0.0).to(
                tl.float32
            )
        coin = tl.load(uni_base + (t - 1))
        # As in the classic kernel: X came from q, so a q(X) that is not a positive probability rejects.
        accepted = going & live & (q > 0.0) & (q <= 1.0) & (coin * q < p)
        correct_len = tl.where(accepted, t, correct_len)
        going = accepted

    # The bonus row: the residual after a rejection, p after the last live draft; p again if the residual is empty.
    row = correct_len
    blocks = tl.arange(0, NB_PAD)
    bmask = blocks < num_blocks
    part_base = (b * NUM_SLOTS + row) * num_blocks
    residual_parts = tl.load(residual_partials_ptr + part_base + blocks, mask=bmask, other=0.0)
    has_residual = tl.sum(residual_parts, axis=0) > 0.0
    p_parts = tl.load(p_partials_ptr + part_base + blocks, mask=bmask, other=0.0)
    parts = tl.where(has_residual, residual_parts, p_parts)
    total = tl.sum(parts, axis=0)
    target_u = tl.load(uni_base + (NUM_SLOTS - 1)) * total
    cums = tl.cumsum(parts, axis=0)
    hit = (cums > target_u) & bmask
    last_pos_blk = tl.max(tl.where(bmask & (parts > 0.0), blocks, 0), axis=0)
    blk = tl.min(tl.where(hit, blocks, NB_PAD), axis=0)
    blk = tl.where(blk < NB_PAD, blk, last_pos_blk)
    before = tl.sum(tl.where(blocks < blk, parts, 0.0), axis=0)

    is_draft_row = (row < live_drafts) & has_residual
    offs = blk * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = offs < vocab
    p_vals = tl.load(tp_base + row * stride_tp_s + offs, mask=vmask, other=0.0).to(
        tl.float32
    )
    if ONE_HOT_Q:
        nxt = tl.load(cand_base + row + 1, mask=is_draft_row, other=-1)
        q_vals = tl.where(offs == nxt, 1.0, 0.0)
    else:
        q_vals = tl.load(
            dp_base + row * stride_dp_s + offs, mask=vmask & is_draft_row, other=0.0
        ).to(tl.float32)
        q_vals = tl.where((q_vals >= 0.0) & (q_vals <= 1.0), q_vals, 0.0)
    vals = tl.where(is_draft_row, tl.maximum(p_vals - q_vals, 0.0), p_vals)
    local = before + tl.cumsum(vals, axis=0)
    in_blk = tl.arange(0, BLOCK_V)
    local_hit = (local > target_u) & vmask
    pick = tl.min(tl.where(local_hit, in_blk, BLOCK_V), axis=0)
    last_pos = tl.max(tl.where(vmask & (vals > 0.0), in_blk, 0), axis=0)
    pick = tl.where(pick < BLOCK_V, pick, last_pos)

    tl.store(correct_len_ptr + b, correct_len)
    tl.store(bonus_ptr + b, (blk * BLOCK_V + pick).to(tl.int64))


def sampled_chain_accept_triton(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    verify_lens: Optional[torch.Tensor],
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, num_slots, vocab = target_probs.shape
    device = target_probs.device
    one_hot_q = draft_probs is None
    assert target_probs.stride(-1) == 1
    if one_hot_q:
        # Never read under ONE_HOT_Q; any pointer will do.
        draft_probs = target_probs
    else:
        assert draft_probs.shape == (bs, num_slots - 1, vocab)
        assert draft_probs.stride(-1) == 1
    assert candidates.stride(-1) == 1 and uniforms.is_contiguous()
    live_drafts = _live_drafts(verify_lens, bs, num_slots, device)
    num_blocks = triton.cdiv(vocab, _BLOCK_V)
    residual_partials = torch.empty(
        (bs * num_slots, num_blocks), dtype=torch.float32, device=device
    )
    p_partials = torch.empty_like(residual_partials)
    _chain_residual_partials_kernel[(bs * num_slots, num_blocks)](
        candidates,
        target_probs,
        draft_probs,
        live_drafts,
        residual_partials,
        p_partials,
        candidates.stride(0),
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs.stride(0),
        draft_probs.stride(1),
        vocab,
        num_blocks,
        NUM_SLOTS=num_slots,
        BLOCK_V=_BLOCK_V,
        ONE_HOT_Q=one_hot_q,
    )
    correct_len = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    _chain_accept_kernel[(bs,)](
        candidates,
        target_probs,
        draft_probs,
        live_drafts,
        residual_partials,
        p_partials,
        uniforms,
        correct_len,
        bonus,
        candidates.stride(0),
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs.stride(0),
        draft_probs.stride(1),
        uniforms.stride(0),
        vocab,
        num_blocks,
        NUM_SLOTS=num_slots,
        BLOCK_V=_BLOCK_V,
        NB_PAD=triton.next_power_of_2(num_blocks),
        ONE_HOT_Q=one_hot_q,
    )
    return correct_len, bonus


def preload_sampled_accept_kernels(
    *, device, verify_num_draft_tokens: int, vocab: int
) -> None:
    """Load the one-hot chain accept's kernels at worker init. The decode warm-up's sampled requests reach them too,
    but a launch whose warm-up is skipped would otherwise device-load them on its first sampled verify."""
    candidates = torch.zeros((1, verify_num_draft_tokens), dtype=torch.int64, device=device)
    target_probs = torch.full(
        (1, verify_num_draft_tokens, vocab), 1.0 / vocab, dtype=torch.float32, device=device
    )
    uniforms = torch.full((1, verify_num_draft_tokens), 0.5, device=device)
    sampled_chain_accept_triton(
        candidates=candidates,
        target_probs=target_probs,
        draft_probs=None,
        verify_lens=torch.ones((1,), dtype=torch.int64, device=device),
        uniforms=uniforms,
    )
