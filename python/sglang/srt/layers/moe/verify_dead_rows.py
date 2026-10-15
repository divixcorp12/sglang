"""Dead rows of a speculative verify: rows the verify computes but routes no experts for.

DSpark's top-probability gate (speculative/dspark_components/dspark_top_prob_gate.py) cuts a verify at a draft
position, but the verify still runs every row of its one captured graph. The rows past the cut are dead. The EXL3
expert-stream MoE gives a dead row the first row's experts at weight 0, so the route planner, which dedupes an expert's
routes onto its first, opens no lane, miss read or CPU job for it. It also writes -1 for a dead row into the RAM-miss
backend's routes, which a CPU expert job's token mask, the protect set and the prefetch scorer read. The scorer's input
rows of a dead row are the first row's.

The routing kernels index expert tables by id without a bounds check, so a dead route must stay a valid expert id;
-1 only goes into ``routes``, whose readers skip it. Row 0 (the anchor) is never dead.

A dead route still counts toward the first row's experts in the planner's route counts (the residency scores) and the
graph counters, and dead rows still run attention, the shared experts and Engram lookups.

Outside a gated verify every row is live, so the masking is an identity wherever it is captured or run.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


class VerifyDeadRows:
    """``dead`` ([max_rows, 1] bool) is read by every EXL3 MoE layer that ``attach_verify_dead_rows`` reached; a
    captured graph reads it at its fixed address, so it is only ever written in place."""

    def __init__(self, *, max_rows: int, device) -> None:
        self.dead = torch.zeros((max_rows, 1), dtype=torch.bool, device=device)
        self._row_ids = torch.arange(max_rows, device=device).view(-1, 1)

    def set_live_rows(self, live_rows: torch.Tensor) -> None:
        """Rows at or past ``live_rows`` (a one-element device tensor, at least 1) are dead until ``clear``."""
        torch.ge(self._row_ids, live_rows.reshape(1, 1), out=self.dead)

    def clear(self) -> None:
        self.dead.zero_()


def mask_dead_routes(
    *, topk_ids: torch.Tensor, topk_weights: torch.Tensor, dead: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``topk_ids`` with each dead row's replaced by row 0's, and ``topk_weights`` with each dead row's zeroed."""
    row_dead = dead[: topk_ids.shape[0]]
    return (
        torch.where(row_dead, topk_ids[:1], topk_ids),
        torch.where(row_dead, 0.0, topk_weights),
    )


def dead_routes_cleared(
    *, flat_ids: torch.Tensor, top_k: int, dead: torch.Tensor
) -> torch.Tensor:
    """Flat [rows * top_k] route ids with every dead row's set to -1."""
    rows = flat_ids.numel() // top_k
    return torch.where(dead[:rows], -1, flat_ids.reshape(rows, top_k)).reshape(-1)


def dead_rows_as_row0(*, x: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
    """[rows, hidden] ``x`` with each dead row replaced by row 0."""
    return torch.where(dead[: x.shape[0]], x[:1], x)


def mask_dead_rows_for_graph(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    dead: torch.Tensor,
    routes: Optional[torch.Tensor],
    x: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The in-graph MoE's masking in one kernel per layer: ``mask_dead_routes``'s ids and weights, the cleared routes
    written into ``routes[:rows * top_k]`` when given, and ``dead_rows_as_row0`` of ``x`` when given (else None)."""
    rows, top_k = topk_ids.shape
    if not topk_ids.is_cuda:
        ids, weights = mask_dead_routes(topk_ids=topk_ids, topk_weights=topk_weights, dead=dead)
        if routes is not None:
            cleared = dead_routes_cleared(flat_ids=topk_ids.reshape(-1), top_k=top_k, dead=dead)
            routes[: rows * top_k].copy_(cleared)
        return ids, weights, None if x is None else dead_rows_as_row0(x=x, dead=dead)
    topk_ids, topk_weights = topk_ids.contiguous(), topk_weights.contiguous()
    ids = torch.empty_like(topk_ids)
    weights = torch.empty_like(topk_weights)
    x_out = None
    if x is not None:
        x = x.contiguous()
        x_out = torch.empty_like(x)
    hidden = 0 if x is None else x.shape[1]
    _mask_dead_rows_kernel[(rows,)](
        topk_ids,
        topk_weights,
        dead,
        ids,
        weights,
        topk_ids if routes is None else routes,
        topk_ids if x is None else x,
        topk_ids if x_out is None else x_out,
        top_k,
        hidden,
        triton.next_power_of_2(top_k),
        1024,
        routes is not None,
        x is not None,
        num_warps=4,
    )
    return ids, weights, x_out


@triton.jit
def _mask_dead_rows_kernel(
    IDS,
    W,
    DEAD,
    IDS_OUT,
    W_OUT,
    ROUTES,
    X,
    X_OUT,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    K_BLOCK: tl.constexpr,
    H_BLOCK: tl.constexpr,
    HAS_ROUTES: tl.constexpr,
    HAS_X: tl.constexpr,
):
    row = tl.program_id(0)
    dead = tl.load(DEAD + row) != 0
    k = tl.arange(0, K_BLOCK)
    in_k = k < TOP_K
    ids = tl.load(IDS + row * TOP_K + k, in_k, 0)
    ids_row0 = tl.load(IDS + k, in_k, 0)
    tl.store(IDS_OUT + row * TOP_K + k, tl.where(dead, ids_row0, ids), in_k)
    weights = tl.load(W + row * TOP_K + k, in_k, 0.0)
    tl.store(W_OUT + row * TOP_K + k, tl.where(dead, 0.0, weights), in_k)
    if HAS_ROUTES:
        tl.store(ROUTES + row * TOP_K + k, tl.where(dead, -1, ids.to(tl.int64)), in_k)
    if HAS_X:
        source = tl.where(dead, 0, row)
        for start in range(0, HIDDEN, H_BLOCK):
            h = start + tl.arange(0, H_BLOCK)
            in_h = h < HIDDEN
            tl.store(X_OUT + row * HIDDEN + h, tl.load(X + source * HIDDEN + h, in_h), in_h)


def attach_verify_dead_rows(model: torch.nn.Module, dead_rows: VerifyDeadRows) -> int:
    """Point every expert-streamed EXL3 MoE layer of ``model`` at ``dead_rows``; returns how many it reached."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.exl3.exl3 import Exl3MoEMethod

    attached = 0
    for module in model.modules():
        if (
            isinstance(module, FusedMoE)
            and isinstance(module.quant_method, Exl3MoEMethod)
            and not module.quant_method.draft
        ):
            module.quant_method.verify_dead_rows = dead_rows.dead
            attached += 1
    return attached
