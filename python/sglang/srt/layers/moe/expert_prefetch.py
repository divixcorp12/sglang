"""Bounded next-layer MoE expert prefetch policy and CUDA coordination."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from operator import index
from typing import Callable, Mapping, Sequence

import torch


class SparseNextLayerPolicy:
    """Rank a bounded next-layer candidate set from sparse route history."""

    def __init__(self, max_candidates: int):
        self.max_candidates = index(max_candidates)
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")

    def predict(
        self,
        routed_experts: Sequence[int],
        popularity: Mapping[int, float],
        affinity: Mapping[tuple[int, int], float],
        resident_experts: frozenset[int] | set[int],
    ) -> tuple[int, ...]:
        """Return nonresident candidates ranked by adjacent-layer evidence."""
        routed = {index(expert_id) for expert_id in routed_experts}
        resident = {index(expert_id) for expert_id in resident_experts}
        scores: defaultdict[int, float] = defaultdict(float)
        for (source, target), value in affinity.items():
            if source in routed and target not in resident and value > 0:
                scores[index(target)] += float(value)
        selected = [
            expert_id
            for expert_id, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[: self.max_candidates]
        ]
        for expert_id, value in popularity.items():
            if len(selected) >= self.max_candidates:
                break
            expert_id = index(expert_id)
            if expert_id not in resident and expert_id not in scores and value > 0:
                selected.append(expert_id)
        return tuple(selected)


@dataclass
class ExpertPrefetchStats:
    predicted_experts: int = 0
    useful_experts: int = 0
    wasted_experts: int = 0
    actual_experts: int = 0
    submitted_bytes: int = 0
    cache_pollution_bytes: int = 0
    evictions: int = 0
    overlap_achieved: int = 0
    event_wait_enqueues: int = 0
    synchronous_corrections: int = 0


class ExpertPrefetchCoordinator:
    """Own one prefetch stream while callers retain placement and correctness.

    The caller supplies a transfer operation that respects ``protected_slots`` and
    must invoke ``synchronous_correction`` after the real next-layer route is
    known. This module never changes ModelOpt tensor layout or hot-cache slots.
    """

    def __init__(
        self,
        enabled: bool = False,
        *,
        device: torch.device | None = None,
        copy_stream: torch.cuda.Stream | None = None,
        ready_event: torch.cuda.Event | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self._copy_stream = copy_stream
        self._ready_event = ready_event
        if self.enabled and self._copy_stream is None:
            if not torch.cuda.is_available():
                raise RuntimeError("expert prefetch requires CUDA")
            self.device = self.device or torch.device(
                "cuda", torch.cuda.current_device()
            )
            self._copy_stream = torch.cuda.Stream(device=self.device)
        if self.enabled and self._ready_event is None:
            self._ready_event = torch.cuda.Event()
        self._protected_slots: set[int] = set()
        self._inflight: tuple[int, ...] = ()
        self._stats = ExpertPrefetchStats()

    @property
    def protected_slots(self) -> frozenset[int]:
        return frozenset(self._protected_slots)

    def protect_slots(self, slots: Sequence[int]) -> None:
        self._protected_slots.update(index(slot) for slot in slots)

    def release_slots(self, slots: Sequence[int]) -> None:
        self._protected_slots.difference_update(index(slot) for slot in slots)

    def launch(
        self,
        candidates: Sequence[int],
        *,
        protected_slots: Sequence[int],
        submit: Callable[[tuple[int, ...]], int | None],
    ) -> bool:
        """Submit candidates on the dedicated stream, or no-op when disabled."""
        if not self.enabled:
            return False
        predicted = tuple(dict.fromkeys(index(expert_id) for expert_id in candidates))
        if not predicted:
            return False
        if self._inflight:
            raise RuntimeError("cannot replace an unfinished expert prefetch")
        self.protect_slots(protected_slots)
        try:
            if self._copy_stream is None:
                submitted_bytes = submit(predicted)
            else:
                with torch.cuda.stream(self._copy_stream):
                    submitted_bytes = submit(predicted)
                    assert self._ready_event is not None
                    self._ready_event.record(self._copy_stream)
        except Exception:
            self._protected_slots.clear()
            raise
        self._inflight = predicted
        self._stats.predicted_experts += len(predicted)
        self._stats.submitted_bytes += int(submitted_bytes or 0)
        return True

    def prepare_for_lookup(self) -> None:
        """Order lookup after every inflight mutation, then unlock its slots."""
        if not self._inflight:
            return
        try:
            if self._ready_event is not None and self.device is not None:
                torch.cuda.current_stream(self.device).wait_event(self._ready_event)
                self._stats.event_wait_enqueues += 1
        finally:
            self._inflight = ()
            self._protected_slots.clear()

    def synchronous_correction(
        self,
        actual_experts: Sequence[int],
        correct: Callable[[tuple[int, ...]], None],
    ) -> None:
        """Order correction after speculative writes and retain authoritative gather."""
        actual = tuple(dict.fromkeys(index(expert_id) for expert_id in actual_experts))
        inflight = self._inflight
        overlap = set(actual).intersection(inflight)
        self.prepare_for_lookup()
        correct(actual)
        self._stats.actual_experts += len(actual)
        self._stats.useful_experts += len(overlap)
        self._stats.wasted_experts += len(set(inflight) - set(actual))
        if overlap:
            self._stats.overlap_achieved += 1
        self._stats.synchronous_corrections += 1

    def record_placement(self, *, cache_pollution_bytes: int, evictions: int) -> None:
        self._stats.cache_pollution_bytes += index(cache_pollution_bytes)
        self._stats.evictions += index(evictions)

    def snapshot_stats(self) -> dict[str, int | float]:
        result = asdict(self._stats)
        predicted = self._stats.predicted_experts
        actual = self._stats.actual_experts
        result["precision"] = (
            self._stats.useful_experts / predicted if predicted else 0.0
        )
        result["recall"] = self._stats.useful_experts / actual if actual else 0.0
        return result
