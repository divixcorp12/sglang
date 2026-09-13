"""Bounded next-layer MoE expert prefetch policy and CUDA coordination."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from operator import index
from typing import Callable, Mapping, Sequence

import torch

from sglang.srt.layers.moe.expert_transfer import (
    AsyncExpertTransferExecutor,
    ExpertTransferTicket,
    FixedRowTransferPlan,
)


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
        for expert_id, value in sorted(
            popularity.items(), key=lambda item: (-item[1], item[0])
        ):
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
    """Coordinate prefetch tickets while callers retain placement and correctness.

    The caller supplies a transfer operation that respects ``protected_slots`` and
    must invoke ``synchronous_correction`` after the real next-layer route is
    known. The callback executes on the device-shared expert transfer stream;
    this module never changes ModelOpt tensor layout or hot-cache slots.
    """

    def __init__(
        self,
        enabled: bool = False,
        *,
        device: torch.device | None = None,
        copy_stream: torch.cuda.Stream | None = None,
        ready_event: torch.cuda.Event | None = None,
        executor: AsyncExpertTransferExecutor | None = None,
        transfer_plan: FixedRowTransferPlan | None = None,
        max_transfer_rows: int = 256,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self._executor = executor
        self._transfer_plan = transfer_plan
        if self.enabled and self.device is None and executor is not None:
            self.device = executor.device
        if self.enabled and self.device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("expert prefetch requires CUDA")
            self.device = self.device or torch.device(
                "cuda", torch.cuda.current_device()
            )
        if self.enabled and self._executor is None:
            if copy_stream is None and ready_event is None:
                assert self.device is not None
                self._executor = AsyncExpertTransferExecutor.for_device(self.device)
            else:
                assert self.device is not None
                self._executor = AsyncExpertTransferExecutor(
                    self.device,
                    max_inflight=1,
                    stream=copy_stream,
                    event_factory=(lambda: ready_event)
                    if ready_event is not None
                    else None,
                    stream_context=torch.cuda.stream,
                )
        if self.enabled and self._transfer_plan is None:
            assert self.device is not None
            self._transfer_plan = FixedRowTransferPlan(
                max_rows=index(max_transfer_rows), device=self.device
            )
        self._protected_slots: set[int] = set()
        self._inflight_ticket: ExpertTransferTicket | None = None
        self._active_prediction: tuple[int, ...] = ()
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
        if self._inflight_ticket is not None:
            raise RuntimeError("cannot replace an unfinished expert prefetch")
        self.protect_slots(protected_slots)
        try:
            assert self._executor is not None
            assert self._transfer_plan is not None
            self._transfer_plan.set_rows(
                predicted,
                tuple(index(slot) for slot in protected_slots),
                (0,) * len(predicted),
            )
            submitted_bytes = 0

            def callback() -> None:
                nonlocal submitted_bytes
                submitted_bytes = int(submit(predicted) or 0)

            producer_stream = (
                torch.cuda.current_stream(self.device)
                if self.device is not None and self.device.type == "cuda"
                else None
            )
            self._inflight_ticket = self._executor.submit_callback(
                self._transfer_plan, callback, producer_stream=producer_stream
            )
        except Exception:
            self._protected_slots.clear()
            raise
        self._active_prediction = predicted
        self._stats.predicted_experts += len(predicted)
        self._stats.submitted_bytes += submitted_bytes
        return True

    def prepare_for_lookup(self) -> None:
        """Order lookup after every inflight mutation, then unlock its slots."""
        if self._inflight_ticket is None:
            return
        try:
            assert self._executor is not None
            consumer_stream = (
                torch.cuda.current_stream(self.device)
                if self.device is not None and self.device.type == "cuda"
                else None
            )
            self._executor.wait(self._inflight_ticket, consumer_stream)
            self._stats.event_wait_enqueues += 1
        finally:
            self._inflight_ticket = None
            self._active_prediction = ()
            self._protected_slots.clear()

    def synchronous_correction(
        self,
        actual_experts: Sequence[int],
        correct: Callable[[tuple[int, ...]], None],
    ) -> None:
        """Order correction after speculative writes and retain authoritative gather."""
        actual = tuple(dict.fromkeys(index(expert_id) for expert_id in actual_experts))
        inflight = self._active_prediction
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
