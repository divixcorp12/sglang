"""Device buffers the prefetch planner reads, and in-graph counters of prefetch value."""

from __future__ import annotations

from typing import Sequence

import torch


class PrefetchCandidateBank:
    """Per target MoE layer, the ``width`` experts with the most expected routes this forward.

    Rows are rewritten in place so addresses survive CUDA graph capture; ids are
    initialised to distinct experts so an unwritten row is still a valid index.
    """

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        width: int,
        device: torch.device,
    ) -> None:
        if width < 1:
            raise ValueError("prefetch candidate width must be positive")
        self.width = width
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.ids = torch.arange(width, dtype=torch.int64, device=device).repeat(len(self._rows), 1)
        self.scores = torch.zeros((len(self._rows), width), dtype=torch.float32, device=device)
        # IDs deliberately retain in-range fallbacks so pre-existing readers never
        # index a sentinel. This tensor is the separate, graph-stable contract for
        # whether each fallback is actually an offer.
        self.valid = torch.zeros((len(self._rows), width), dtype=torch.bool, device=device)

    def write(
        self,
        target_layer: int,
        expert_scores: torch.Tensor,
        *,
        expert_to_slot: torch.Tensor | None = None,
    ) -> None:
        """Rewrite one target's rows with its ``width`` best candidates.

        With ``expert_to_slot`` omitted, ranks every expert by summed score, matching
        the pre-residency-filtering behaviour existing callers still rely on. With it
        given, residency and non-finite scores are excluded *before* truncation to
        ``width``, so a resident expert can never crowd out a nonresident one; ties and
        excluded scores break by ascending expert id, via a stable descending sort over
        distinct expert ids. When fewer than ``width`` candidates survive exclusion, the
        remaining rows fall back to excluded experts (still valid, distinct ids) rather
        than a sentinel. ``valid_for`` distinguishes those non-offers from the selected
        eligible prefix, so a fallback can never become a garbage pull index.
        """
        summed = expert_scores.sum(dim=0)
        excluded = ~torch.isfinite(summed)
        if expert_to_slot is not None:
            excluded = excluded | (expert_to_slot >= 0)
        ranked = torch.where(excluded, summed.new_full((), float("-inf")), summed)
        # The candidate bank is deliberately the reference top-W path. Serving
        # top-1 avoids it altogether through the JIT selector owned by
        # PrefetchPuller; recall, calibration, and analysis keep this stable
        # sort to preserve their complete candidate-bank contract.
        order = torch.argsort(ranked, descending=True, stable=True)
        top = order[: self.width]
        row = self._rows[target_layer]
        self.ids[row].copy_(top)
        self.scores[row].copy_(summed.index_select(0, top))
        self.valid[row].copy_((~excluded).index_select(0, top))

    def ids_for(self, target_layer: int) -> torch.Tensor:
        return self.ids[self._rows[target_layer]]

    def scores_for(self, target_layer: int) -> torch.Tensor:
        return self.scores[self._rows[target_layer]]

    def valid_for(self, target_layer: int) -> torch.Tensor:
        """Persistent offer validity for ``ids_for(target_layer)``'s fallback-safe IDs."""
        return self.valid[self._rows[target_layer]]


class DedicatedPrefetchSlot:
    """The one speculative row appended after ``[0, capacity + demand_rows)`` per layer.

    Reserved inside the existing per-layer contiguous cache allocation (plan section
    7.1, "Scratch layout and budget"): its index is ``capacity + demand_rows``, so it
    adds no new resident-capacity slot and no new demand-scratch row, only the one
    trailing row appended to the allocation. It must never be published into
    ``expert_to_slot`` and never become a residency eviction or promotion destination.
    """

    def __init__(self, *, capacity: int, demand_rows: int) -> None:
        if capacity < 0 or demand_rows < 0:
            raise ValueError("dedicated prefetch slot capacity and demand rows must be nonnegative")
        self.capacity = capacity
        self.demand_rows = demand_rows
        self.index = capacity + demand_rows

    def assert_within_allocation(self, allocation_rows: int) -> None:
        """Raise unless the reserved row is exactly the trailing row of ``allocation_rows``."""
        if allocation_rows != self.index + 1:
            raise ValueError("dedicated prefetch slot is not the trailing row of the cache allocation")

    def assert_excluded_from_mapping(self, expert_to_slot: torch.Tensor) -> None:
        """Raise if any expert is mapped to the reserved row through the permanent mapping."""
        if bool((expert_to_slot == self.index).any()):
            raise RuntimeError("dedicated prefetch slot is reachable through expert_to_slot")


class BudgetRecall:
    """Non-resident native routes, and those the first ``budget`` non-resident candidates cover.

    Discontinuity at c48b3c69e5 (2026-09-16): before it, ``budget_recall`` counted
    non-resident coverage within the top-W bank; after, within the non-resident-only
    bank. Figures recorded before that commit (llapor 0.383/0.389, apex 0.424/0.420)
    are not comparable to figures after it. The rise affects both predictors by an
    amount that grows with the write-to-observe distance, so next-layer and same-layer
    arms are not comparable to each other across the seam either, not only to their own
    pre-change values.
    """

    def __init__(self, *, layer_ids: Sequence[int], budget: int, device: torch.device) -> None:
        if budget < 1:
            raise ValueError("prefetch budget must be positive")
        self.budget = budget
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.counts = torch.zeros((len(self._rows), 2), dtype=torch.int64, device=device)

    def observe(
        self,
        *,
        target_layer: int,
        candidate_ids: torch.Tensor,
        candidate_valid: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_to_slot: torch.Tensor,
    ) -> None:
        resident = expert_to_slot >= 0
        offered_mask = candidate_valid & ~resident.index_select(0, candidate_ids)
        offered_mask &= torch.cumsum(offered_mask.to(torch.int64), dim=0) <= self.budget
        offered = torch.where(offered_mask, candidate_ids, torch.full_like(candidate_ids, -1))
        native = topk_ids.reshape(-1).long()
        missed = ~resident.index_select(0, native)
        covered = (native.unsqueeze(1) == offered.unsqueeze(0)).any(dim=1)
        row = self.counts[self._rows[target_layer]]
        row[0].add_(missed.sum())
        row[1].add_((missed & covered).sum())

    def snapshot(self) -> dict[int, tuple[int, int]]:
        values = self.counts.cpu().tolist()
        return {layer_id: tuple(values[row]) for layer_id, row in self._rows.items()}
