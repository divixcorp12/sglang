"""Activation-aware, boundary-materialized policy for MoE expert residency."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from operator import index
from typing import Iterable, NamedTuple, Sequence

import numpy as np
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
    in-place device operation and never reads a CUDA value in Python.
    ``advance`` folds pending counts into the scores on the device; ``decide``
    and ``materialize_boundary`` intentionally transfer the fixed-size scores
    to the CPU, so callers must use them only at a residency boundary.

    Scores decay by ``decay`` once per boundary, or once per ``decay_tokens``
    routed tokens when both it and the boundary's token count are given. A
    candidate replaces its victim only when it leads by ``promotion_margin``
    plus ``promotion_sigmas`` standard deviations of count noise.
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
        decay_tokens: int | None = None,
        promotion_sigmas: float = 0.0,
    ) -> None:
        self.num_experts = index(num_experts)
        self.capacity = index(capacity)
        self.decay = float(decay)
        self.promotion_margin = float(promotion_margin)
        self.decay_tokens = None if decay_tokens is None else index(decay_tokens)
        self.promotion_sigmas = float(promotion_sigmas)
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 0 <= self.capacity <= self.num_experts:
            raise ValueError("capacity must be between zero and num_experts")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be between zero and one")
        if self.promotion_margin < 0.0:
            raise ValueError("promotion_margin must be nonnegative")
        if self.decay_tokens is not None and self.decay_tokens < 1:
            raise ValueError("decay_tokens must be positive")
        if not math.isfinite(self.promotion_sigmas) or self.promotion_sigmas < 0.0:
            raise ValueError("promotion_sigmas must be finite and nonnegative")
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

    def boundary_decay(self, tokens: int | None = None) -> float:
        """Multiplier ``advance`` applies to the scores at a boundary of ``tokens``."""
        if self.decay_tokens is not None and tokens is not None:
            return self.decay ** (index(tokens) / self.decay_tokens)
        return self.decay

    def advance(self, tokens: int | None = None) -> None:
        """Decay the scores and fold in pending counts without a host readback."""
        self._scores.mul_(self.boundary_decay(tokens)).add_(self._pending_counts)
        self._pending_counts.zero_()

    def materialize_boundary(
        self, resident_experts: Iterable[int], tokens: int | None = None
    ) -> ResidencyDecision:
        """Apply decay and return a deterministic desired set at an explicit boundary."""
        self.advance(tokens)
        return self.decide(resident_experts)

    def decide(self, resident_experts: Iterable[int]) -> ResidencyDecision:
        """Return the deterministic desired set for the current scores."""
        return decide_residency_policies([self], [resident_experts])[0]

    def _decide_from_host(
        self,
        score_values: list[float],
        ranked: list[int],
        rank: list[int],
        resident_experts: Iterable[int],
    ) -> ResidencyDecision:
        """Decide from host scores, their ``(-score, expert)`` order and its inverse.

        ``rank[expert]`` is the expert's position in ``ranked``, so ordering by
        rank is ordering by ``(-score, expert)`` and the smallest ``(score,
        -expert)`` victim is the member with the largest rank.
        """
        observed_resident = self._deduplicate_and_validate(resident_experts)
        resident = self._limited_resident(observed_resident, rank)
        desired = self._select_desired(ranked, score_values, rank, resident)
        observed_resident_set = set(observed_resident)
        desired_set = set(desired)
        promotions = tuple(
            expert_id for expert_id in desired if expert_id not in observed_resident_set
        )
        evictions = tuple(
            expert_id for expert_id in observed_resident if expert_id not in desired_set
        )
        self._metrics.boundary_updates += 1
        self._metrics.promotions += len(promotions)
        self._metrics.evictions += len(evictions)
        return ResidencyDecision(
            desired_experts=desired,
            promotions=promotions,
            evictions=evictions,
            ranked_experts=tuple(ranked),
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
        ranked: list[int],
        score_values: list[float],
        rank: list[int],
        resident: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Fill free capacity in rank order, then swap in leaders over the weakest members.

        Members admitted in rank order only get weaker, so the weakest member is
        either the weakest remaining resident or the last admitted expert: a
        queue of residents and a stack of admissions replace a linear victim
        search per candidate.
        """
        if self.capacity == 0:
            return ()
        residents = list(resident[: self.capacity])
        members = set(residents)
        admitted: list[int] = []
        position = 0
        expert_count = len(ranked)
        while position < expert_count and len(members) < self.capacity:
            expert_id = ranked[position]
            if score_values[expert_id] <= 0.0:
                break
            if expert_id not in members:
                admitted.append(expert_id)
                members.add(expert_id)
            position += 1
        victims: list[int] | None = None
        victim_position = 0
        while position < expert_count:
            candidate = ranked[position]
            position += 1
            if candidate in members:
                continue
            if score_values[candidate] <= 0.0 or not members:
                break
            if victims is None:
                victims = sorted(residents, key=rank.__getitem__, reverse=True)
            from_residents = victim_position < len(victims) and (
                not admitted or rank[victims[victim_position]] > rank[admitted[-1]]
            )
            victim = victims[victim_position] if from_residents else admitted[-1]
            if score_values[candidate] <= score_values[victim] + self._promotion_threshold(
                score_values[candidate], score_values[victim]
            ):
                break
            if from_residents:
                victim_position += 1
            else:
                admitted.pop()
            members.remove(victim)
            members.add(candidate)
            admitted.append(candidate)
        return tuple(sorted(members, key=rank.__getitem__))

    def _promotion_threshold(self, candidate: float, victim: float) -> float:
        """Score lead a candidate needs over its victim before it is promoted.

        For independent routes a score's variance is at most its mean: raw
        counts from a seed or a long prefill are Poisson-like, and decayed sums
        settle near half their mean. Two scores therefore differ by noise of at
        most about ``sqrt(candidate + victim)``, and requiring
        ``promotion_sigmas`` of it stops swaps between experts with
        indistinguishable rates, each of which costs a row copy and saves none.
        """
        return self.promotion_margin + self.promotion_sigmas * math.sqrt(
            max(candidate + victim, 0.0)
        )

    def _limited_resident(
        self, resident: tuple[int, ...], rank: list[int]
    ) -> tuple[int, ...]:
        if len(resident) <= self.capacity:
            return resident
        return tuple(sorted(resident, key=rank.__getitem__)[: self.capacity])

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


class DeviceResidencyDecision(NamedTuple):
    """Fixed-shape per-layer residency changes; entries past each count are unused.

    ``promotions`` lists promoted experts in rank order, ``evictions`` the
    evicted residents worst first, both ``[layers, max_promotions]``.
    ``needed_promotions`` equals the uncapped promotion count up to
    ``max_promotions + 1``, so it exceeds ``max_promotions`` exactly when the
    cap truncated the decision.
    """

    promotions: torch.Tensor
    promotion_counts: torch.Tensor
    evictions: torch.Tensor
    eviction_counts: torch.Tensor
    needed_promotions: torch.Tensor


def residency_rank_keys(scores: torch.Tensor) -> torch.Tensor:
    """Unique int64 keys ordering experts by ``(-score, expert)`` when sorted descending.

    Nonnegative float32 scores order like their IEEE bit patterns; adding
    ``0.0`` folds ``-0.0`` into ``0.0`` so the bits compare as the values do.
    The low 16 bits break ties toward the lower expert ID.
    """
    num_experts = scores.shape[-1]
    if num_experts > 1 << 16:
        raise ValueError("residency rank keys support at most 65536 experts")
    bits = (scores.to(torch.float32) + 0.0).view(torch.int32).to(torch.int64)
    ids = torch.arange(num_experts, dtype=torch.int64, device=scores.device)
    return bits * (1 << 16) + (num_experts - 1 - ids)


def decide_residency_on_device(
    scores: torch.Tensor,
    resident: torch.Tensor,
    capacity: torch.Tensor,
    *,
    promotion_margin: float,
    promotion_sigmas: float,
    max_promotions: int,
    active: torch.Tensor | None = None,
) -> DeviceResidencyDecision:
    """Decide every layer's promotions and evictions with fixed-shape device ops.

    Equivalent to :meth:`ExpertResidencyPolicy.decide` on each row when the
    residents fit the capacity, truncated to the first ``max_promotions``
    promotions in rank order. Candidates are nonresidents with a positive
    score in ``(-score, expert)`` order. Free capacity admits the leading
    candidates; candidate ``i`` after them then replaces the ``i``-th worst
    member, ``(score, -expert)`` ascending, for as long as its lead clears
    ``promotion_margin + promotion_sigmas * sqrt(candidate + victim)``. The
    swaps form a prefix: a member admitted by a swap that becomes the weakest
    can only be challenged by a later, weaker candidate, which fails the same
    test the next worst original member would. Nothing is read on the host,
    so a CUDA graph can capture the decision.
    """
    num_layers, num_experts = scores.shape
    width = min(index(max_promotions), num_experts)
    if width < 1:
        raise ValueError("max_promotions must be positive")
    device = scores.device
    positions = torch.arange(num_experts, dtype=torch.int64, device=device)
    window = positions[: min(width + 1, num_experts)]
    keys = residency_rank_keys(scores)
    resident = resident.to(torch.bool)
    capacity = capacity.to(torch.int64)
    candidates = ~resident & (scores > 0)
    resident_count = resident.sum(dim=1)
    candidate_count = candidates.sum(dim=1)
    candidate_order = torch.sort(
        torch.where(candidates, keys, torch.full_like(keys, -1)), dim=1, descending=True
    ).indices
    fill = torch.minimum((capacity - resident_count).clamp(min=0), candidate_count)
    admitted = torch.zeros_like(resident).scatter_(
        1, candidate_order, positions.unsqueeze(0) < fill.unsqueeze(1)
    )
    members = resident | admitted
    member_count = resident_count + fill
    victims = torch.sort(
        torch.where(members, keys, torch.full_like(keys, torch.iinfo(torch.int64).max)),
        dim=1,
    ).indices[:, : window.numel()]
    swap_positions = fill.unsqueeze(1) + window.unsqueeze(0)
    swap_candidates = candidate_order.gather(1, swap_positions.clamp(max=num_experts - 1))
    exact = scores.to(torch.float64)
    candidate_scores = exact.gather(1, swap_candidates)
    victim_scores = exact.gather(1, victims)
    lead = torch.sqrt(torch.clamp(candidate_scores + victim_scores, min=0.0))
    threshold = lead * float(promotion_sigmas) + float(promotion_margin)
    swaps = (
        (swap_positions < candidate_count.unsqueeze(1))
        & (window.unsqueeze(0) < member_count.unsqueeze(1))
        & (capacity > 0).unsqueeze(1)
        & (candidate_scores > victim_scores + threshold)
    )
    swap_count = torch.cumprod(swaps.to(torch.int64), dim=1).sum(dim=1)
    needed = torch.where(capacity > 0, fill + swap_count, torch.zeros_like(fill))
    if active is not None:
        needed = torch.where(active.to(torch.bool), needed, torch.zeros_like(needed))
    promotion_counts = torch.clamp(needed, max=width)
    eviction_counts = torch.clamp(promotion_counts - fill, min=0)
    return DeviceResidencyDecision(
        promotions=candidate_order[:, :width],
        promotion_counts=promotion_counts,
        evictions=victims[:, :width],
        eviction_counts=eviction_counts,
        needed_promotions=needed,
    )


def advance_residency_policies(
    policies: Sequence[ExpertResidencyPolicy], tokens: int | None = None
) -> None:
    """Advance every policy at one boundary with three fused launches per group.

    Policies sharing a device and boundary decay are decayed, folded and
    cleared by one ``torch._foreach_*`` call each, which is bit-identical to
    :meth:`ExpertResidencyPolicy.advance` on every policy.
    """
    groups: dict[tuple[torch.device, float], list[ExpertResidencyPolicy]] = {}
    for policy in policies:
        groups.setdefault((policy.device, policy.boundary_decay(tokens)), []).append(
            policy
        )
    for (_, decay), members in groups.items():
        scores = [policy._scores for policy in members]
        pending = [policy._pending_counts for policy in members]
        torch._foreach_mul_(scores, decay)
        torch._foreach_add_(scores, pending)
        torch._foreach_zero_(pending)


def decide_residency_policies(
    policies: Sequence[ExpertResidencyPolicy],
    resident_experts: Sequence[Iterable[int]],
) -> list[ResidencyDecision]:
    """Decide every policy from one score readback per device and expert count.

    The stacked scores are ranked by one stable host argsort of the negated
    scores, which orders ties by expert ID exactly as the per-layer
    ``(-score, expert)`` sort did; decisions equal per-policy :meth:`decide`.
    """
    policies = list(policies)
    residents = list(resident_experts)
    if len(policies) != len(residents):
        raise ValueError("each residency policy needs one resident expert set")
    groups: dict[tuple[torch.device, int], list[int]] = {}
    for position, policy in enumerate(policies):
        groups.setdefault((policy.device, policy.num_experts), []).append(position)
    decisions: list[ResidencyDecision | None] = [None] * len(policies)
    for positions in groups.values():
        scores = (
            torch.stack([policies[position]._scores.detach() for position in positions])
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        group_decisions = decide_residency_policies_from_host_scores(
            [policies[position] for position in positions],
            [residents[position] for position in positions],
            [scores[row] for row in range(len(positions))],
        )
        for position, decision in zip(positions, group_decisions):
            decisions[position] = decision
    return decisions


def decide_residency_policies_from_host_scores(
    policies: Sequence[ExpertResidencyPolicy],
    resident_experts: Sequence[Iterable[int]],
    host_scores: Sequence[torch.Tensor | np.ndarray],
) -> list[ResidencyDecision]:
    """Decide from an already-completed CPU snapshot, without reading the device.

    The resident sets are sampled when this function runs. A score copy may
    finish after another slot update; using the current residents prevents a
    stale snapshot from evicting or preserving a slot by an outdated mapping.
    """
    if not len(policies) == len(resident_experts) == len(host_scores):
        raise ValueError("each policy needs one resident set and host score row")
    decisions = []
    for policy, residents, row in zip(policies, resident_experts, host_scores):
        if isinstance(row, torch.Tensor):
            if row.device.type != "cpu":
                raise ValueError("score snapshots must already be on the CPU")
            values = row.numpy().astype(np.float64, copy=False)
        else:
            values = np.asarray(row, dtype=np.float64)
        if values.shape != (policy.num_experts,):
            raise ValueError("score snapshot shape does not match policy")
        ranked = np.argsort(-values, kind="stable")
        inverse = np.empty_like(ranked)
        inverse[ranked] = np.arange(policy.num_experts)
        decisions.append(
            policy._decide_from_host(
                values.tolist(), ranked.tolist(), inverse.tolist(), residents
            )
        )
    return decisions
