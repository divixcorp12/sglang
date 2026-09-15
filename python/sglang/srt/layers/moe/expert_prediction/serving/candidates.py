"""Device buffers the prefetch planner reads, and in-graph counters of prefetch value."""

from __future__ import annotations

from typing import Sequence

import torch


class PrefetchCandidateBank:
    """Per target MoE layer, the ``width`` experts with the most expected routes this forward.

    Rows are rewritten in place so addresses survive CUDA graph capture; ids are
    initialised to distinct experts so an unwritten row is still a valid index.
    """

    def __init__(self, *, layer_ids: Sequence[int], width: int, device: torch.device) -> None:
        if width < 1:
            raise ValueError("prefetch candidate width must be positive")
        self.width = width
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.ids = torch.arange(width, dtype=torch.int64, device=device).repeat(len(self._rows), 1)
        self.scores = torch.zeros((len(self._rows), width), dtype=torch.float32, device=device)

    def write(self, target_layer: int, expert_scores: torch.Tensor) -> None:
        top = torch.topk(expert_scores.sum(dim=0), self.width)
        row = self._rows[target_layer]
        self.ids[row].copy_(top.indices)
        self.scores[row].copy_(top.values)

    def ids_for(self, target_layer: int) -> torch.Tensor:
        return self.ids[self._rows[target_layer]]

    def scores_for(self, target_layer: int) -> torch.Tensor:
        return self.scores[self._rows[target_layer]]


class BudgetRecall:
    """Non-resident native routes, and those the first ``budget`` non-resident candidates cover."""

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
        topk_ids: torch.Tensor,
        expert_to_slot: torch.Tensor,
    ) -> None:
        resident = expert_to_slot >= 0
        offered_mask = ~resident.index_select(0, candidate_ids)
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
