"""Activation-aware, boundary-materialized policy for MoE expert residency."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from operator import index
from typing import Iterable, Sequence

import torch


class TransferPriority(IntEnum):
    """Order exact correctness work ahead of optional background work."""

    EXACT_DEMAND = 0
    BACKGROUND_PROMOTION = 1


@dataclass(frozen=True)
class ResidencyTransfer:
    """One transfer candidate with an explicit scheduler priority."""

    expert_id: int
    priority: TransferPriority


@dataclass(frozen=True)
class ResidencyDecision:
    """Boundary-time desired set and the mutations required to reach it."""

    desired_experts: tuple[int, ...]
    promotions: tuple[int, ...]
    evictions: tuple[int, ...]
    ranked_experts: tuple[int, ...]


@dataclass
class ResidencyPolicyMetrics:
    """Cumulative, low-volume policy counters."""

    recorded_count_batches: int = 0
    recorded_routes: int = 0
    boundary_updates: int = 0
    promotions: int = 0
    evictions: int = 0
    exact_demand_experts: int = 0
    background_promotion_experts: int = 0


class ExpertResidencyPolicy:
    """Select a bounded hot-expert set from exponentially decayed route counts.

    ``record_counts`` is safe on the route hot path: it only performs an
    in-place device operation and never reads a CUDA value in Python. Calling
    ``materialize_boundary`` intentionally transfers the fixed-size scores to
    the CPU, so callers must use it only at a request or prefill boundary.
    """

    def __init__(
        self,
        num_experts: int,
        capacity: int,
        *,
        decay: float = 0.95,
        promotion_margin: float = 0.0,
        device: torch.device | str | None = None,
        initial_scores: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        self.num_experts = index(num_experts)
        self.capacity = index(capacity)
        self.decay = float(decay)
        self.promotion_margin = float(promotion_margin)
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 0 <= self.capacity <= self.num_experts:
            raise ValueError("capacity must be between zero and num_experts")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be between zero and one")
        if self.promotion_margin < 0.0:
            raise ValueError("promotion_margin must be nonnegative")
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._pending_counts = torch.zeros(
            self.num_experts, dtype=torch.float32, device=self.device
        )
        self._scores = torch.zeros_like(self._pending_counts)
        if initial_scores is not None:
            scores = torch.as_tensor(
                initial_scores, dtype=self._scores.dtype, device=self.device
            ).reshape(-1)
            if scores.numel() != self.num_experts:
                raise ValueError("initial_scores must have one entry per expert")
            if not torch.isfinite(scores).all() or (scores < 0).any():
                raise ValueError("initial_scores must be finite and nonnegative")
            self._scores.copy_(scores)
        self._metrics = ResidencyPolicyMetrics()

    @property
    def pending_counts(self) -> torch.Tensor:
        """Return the fixed device counter buffer without materializing it."""
        return self._pending_counts

    def record_counts(self, counts: torch.Tensor) -> None:
        """Accumulate a fixed per-expert count buffer without a host readback."""
        if counts.numel() != self.num_experts:
            raise ValueError("counts must have one entry per expert")
        if counts.device != self.device:
            raise ValueError("counts must be on the policy device")
        self._pending_counts.add_(counts.reshape(-1).to(dtype=self._pending_counts.dtype))
        self._metrics.recorded_count_batches += 1

    def record_routes(self, expert_ids: torch.Tensor) -> None:
        """Accumulate routed IDs on-device; callers may supply precomputed counts instead."""
        if expert_ids.device != self.device:
            raise ValueError("expert_ids must be on the policy device")
        if expert_ids.numel() == 0:
            return
        route_counts = torch.bincount(
            expert_ids.reshape(-1).to(dtype=torch.long), minlength=self.num_experts
        )
        if route_counts.numel() != self.num_experts:
            raise ValueError("expert_ids must be in the policy expert range")
        self.record_counts(route_counts)
        self._metrics.recorded_routes += expert_ids.numel()

    def materialize_boundary(
        self, resident_experts: Iterable[int]
    ) -> ResidencyDecision:
        """Apply decay and return a deterministic desired set at an explicit boundary."""
        self._scores.mul_(self.decay).add_(self._pending_counts)
        self._pending_counts.zero_()
        score_values = self._scores.detach().cpu().tolist()
        observed_resident = self._deduplicate_and_validate(resident_experts)
        resident = self._limited_resident(observed_resident, score_values)
        ranked = tuple(
            expert_id
            for expert_id, _ in sorted(
                enumerate(score_values), key=lambda item: (-item[1], item[0])
            )
        )
        desired = self._select_desired(ranked, score_values, resident)
        observed_resident_set = set(observed_resident)
        promotions = tuple(
            expert_id for expert_id in desired if expert_id not in observed_resident_set
        )
        evictions = tuple(
            expert_id for expert_id in observed_resident if expert_id not in desired
        )
        self._metrics.boundary_updates += 1
        self._metrics.promotions += len(promotions)
        self._metrics.evictions += len(evictions)
        return ResidencyDecision(
            desired_experts=desired,
            promotions=promotions,
            evictions=evictions,
            ranked_experts=ranked,
        )

    def schedule_transfers(
        self, *, exact_demand: Sequence[int], decision: ResidencyDecision
    ) -> tuple[ResidencyTransfer, ...]:
        """Return exact demand before optional background promotions.

        Exact transfers retain first-seen route order. Background promotions are
        omitted when exact demand already covers the same expert.
        """
        exact = self._deduplicate_and_validate(exact_demand)
        exact_set = set(exact)
        background = tuple(
            expert_id for expert_id in decision.promotions if expert_id not in exact_set
        )
        self._metrics.exact_demand_experts += len(exact)
        self._metrics.background_promotion_experts += len(background)
        return tuple(
            ResidencyTransfer(expert_id, TransferPriority.EXACT_DEMAND)
            for expert_id in exact
        ) + tuple(
            ResidencyTransfer(expert_id, TransferPriority.BACKGROUND_PROMOTION)
            for expert_id in background
        )

    def snapshot_metrics(self) -> dict[str, int]:
        """Materialize only host-maintained cumulative counters."""
        return asdict(self._metrics)

    def _select_desired(
        self,
        ranked: tuple[int, ...],
        score_values: list[float],
        resident: tuple[int, ...],
    ) -> tuple[int, ...]:
        if self.capacity == 0:
            return ()
        desired = list(resident[: self.capacity])
        resident_set = set(desired)
        for expert_id in ranked:
            if len(desired) >= self.capacity:
                break
            if score_values[expert_id] <= 0.0:
                break
            if expert_id not in resident_set:
                desired.append(expert_id)
                resident_set.add(expert_id)
        for candidate in ranked:
            if candidate in resident_set:
                continue
            if score_values[candidate] <= 0.0 or not desired:
                break
            victim = min(
                desired, key=lambda expert_id: (score_values[expert_id], -expert_id)
            )
            if score_values[candidate] <= score_values[victim] + self.promotion_margin:
                break
            desired.remove(victim)
            resident_set.remove(victim)
            desired.append(candidate)
            resident_set.add(candidate)
        return tuple(
            sorted(desired, key=lambda expert_id: (-score_values[expert_id], expert_id))
        )

    def _limited_resident(
        self, resident: tuple[int, ...], score_values: list[float]
    ) -> tuple[int, ...]:
        if len(resident) <= self.capacity:
            return resident
        return tuple(
            sorted(resident, key=lambda expert_id: (-score_values[expert_id], expert_id))[
                : self.capacity
            ]
        )

    def _deduplicate_and_validate(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        result = []
        seen = set()
        for value in expert_ids:
            expert_id = index(value)
            if not 0 <= expert_id < self.num_experts:
                raise ValueError("expert ID is outside the policy range")
            if expert_id not in seen:
                result.append(expert_id)
                seen.add(expert_id)
        return tuple(result)
