"""Offline value of expert prefetch: budget recall of cache misses and in-graph/doorbell timing models."""

from __future__ import annotations

import math

import torch

# E28 fit of the in-graph miss copy kernel (11.5 GiB/s): 0.2239 ms/row + 0.006 ms per launch.
IN_GRAPH_ROW_MS = 0.2239
IN_GRAPH_FIXED_MS = 0.006
# E32: doorbell thread on a torch-created stream, 3 rows in 0.628 ms (12.4 GiB/s).
DOORBELL_ROW_MS = 0.209
DOORBELL_FIXED_MS = 0.007
# E29: a wait on an already-landed request costs 15-22 us.
DOORBELL_COMPLETED_WAIT_MS = 0.02


def offered_ids(scores: torch.Tensor, resident: torch.Tensor, budget: int) -> torch.Tensor:
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


def _wait_past_window(*, posted, window_ms: float, reaction_ms: float) -> torch.Tensor:
    if torch.is_tensor(posted):
        landing_ms = reaction_ms + posted.to(torch.float64) * DOORBELL_ROW_MS + DOORBELL_FIXED_MS
        return (landing_ms - window_ms).clamp(min=0.0)
    landing_ms = reaction_ms + posted * DOORBELL_ROW_MS + DOORBELL_FIXED_MS
    return torch.as_tensor(max(0.0, landing_ms - window_ms), dtype=torch.float64)


def doorbell_saving_ms(
    *, hits: torch.Tensor, budget: "int | torch.Tensor", window_ms: float, reaction_ms: float
) -> torch.Tensor:
    """In-graph copy time the hits avoid, minus the wait for a ``budget``-row request (``all_or_nothing``
    delivery: resolve waits for every posted row). ``budget`` is the posted count, a scalar for a fixed-size
    request or a per-row tensor (e.g. the oracle's ``min(misses, budget)``)."""
    wait = _wait_past_window(posted=budget, window_ms=window_ms, reaction_ms=reaction_ms)
    return hits.to(torch.float64) * IN_GRAPH_ROW_MS - wait - DOORBELL_COMPLETED_WAIT_MS


def prefix_wait_ms(
    *, offered: torch.Tensor, topk_ids: torch.Tensor, resident: torch.Tensor, window_ms: float, reaction_ms: float
) -> torch.Tensor:
    """``prefix`` delivery: rows land in priority order: the wait only runs through the deepest offered
    rank that resolves an actual miss (a row with no hits waits nothing past ``DOORBELL_COMPLETED_WAIT_MS``,
    charged uniformly by the caller alongside ``doorbell_saving_ms``)."""
    native = topk_ids.long()
    missed = ~resident.gather(1, native)
    budget = offered.shape[1]
    hit_at_rank = torch.stack(
        [((native == offered[:, k : k + 1]) & missed).any(dim=1) for k in range(budget)], dim=1
    )
    any_hit = hit_at_rank.any(dim=1)
    ranks = torch.arange(1, budget + 1, device=offered.device)
    deepest_rank = (hit_at_rank.long() * ranks).amax(dim=1)
    wait = _wait_past_window(posted=deepest_rank, window_ms=window_ms, reaction_ms=reaction_ms)
    return torch.where(any_hit, wait, torch.zeros_like(wait))


def side_stream_ready_rows(*, budget: int, window_ms: float) -> int:
    """Rows a one-launch-per-row side-stream copy lands within ``window_ms``; later rows are not delivered."""
    return max(0, min(budget, math.floor(window_ms / (IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS))))
