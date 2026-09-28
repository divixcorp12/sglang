"""Offline value of expert prefetch: budget recall of cache misses and an in-graph side-stream timing model."""

from __future__ import annotations

import math

import torch

# E28 fit of the in-graph miss copy kernel (11.5 GiB/s): 0.2239 ms/row + 0.006 ms per launch.
IN_GRAPH_ROW_MS = 0.2239
IN_GRAPH_FIXED_MS = 0.006


def offered_ids(
    scores: torch.Tensor, resident: torch.Tensor, budget: int
) -> torch.Tensor:
    """Top-``budget`` non-resident-scored expert ids per row, best first (priority/delivery order)."""
    return scores.masked_fill(resident, float("-inf")).topk(budget, dim=1).indices


def budget_hits(
    scores: torch.Tensor, topk_ids: torch.Tensor, resident: torch.Tensor, budget: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: native experts missing from the hot cache, and those the top-``budget`` non-resident scores cover."""
    offered = offered_ids(scores, resident, budget)
    native = topk_ids.long()
    missed = ~resident.gather(1, native)
    covered = (native.unsqueeze(2) == offered.unsqueeze(1)).any(dim=2)
    return missed.sum(dim=1), (missed & covered).sum(dim=1)


def oracle_hits(missed: torch.Tensor, budget: int) -> torch.Tensor:
    """Perfect-recall, perfect-precision arm: posts ``min(misses, budget)`` rows, all of them hits."""
    return missed.clamp(max=budget)


def side_stream_ready_rows(*, budget: int, window_ms: float) -> int:
    """Rows a one-launch-per-row side-stream copy lands within ``window_ms``; later rows are not delivered."""
    return max(
        0, min(budget, math.floor(window_ms / (IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS)))
    )
