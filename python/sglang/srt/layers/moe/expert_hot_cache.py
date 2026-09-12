"""Fixed CUDA slots for frequently selected host-resident expert rows."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from sglang.srt.layers.moe.expert_prefetch import (
    ExpertPrefetchCoordinator,
    SparseNextLayerPolicy,
)
from sglang.srt.layers.moe.expert_stream import ExpertStreamer, _tensor_data

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HotCacheUpdateStats:
    promoted_experts: int
    evicted_experts: int
    migration_bytes: int


class ExpertHotCache:
    """Own stable slot tensors; reassign only between serialized gather calls.

    Slots and lookups live on the streamer's CUDA source device, or the current
    CUDA device when all backing sources are on the host. Like ExpertStreamer,
    updates and gathers must run on the same CUDA stream. Sources must remain
    unchanged while their rows are resident in the cache.
    """

    def __init__(self, streamer: ExpertStreamer, capacity: int):
        capacity = index(capacity)
        if not 0 <= capacity <= streamer.num_experts:
            raise ValueError("hot cache capacity must be within the expert count")
        self.streamer = streamer
        self.capacity = capacity
        self.bytes_per_expert = streamer.bytes_per_expert
        self.capacity_bytes = capacity * self.bytes_per_expert
        devices = {
            _tensor_data(getattr(streamer.layer, name)).device
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cuda"
        }
        if len(devices) > 1:
            raise ValueError("hot cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (capacity,) + tuple(source.shape[1:]),
                dtype=source.dtype,
                device=self.device,
            )
            for name in streamer.tensor_names
            for source in [_tensor_data(getattr(streamer.layer, name))]
        }
        self.expert_to_slot = torch.full(
            (streamer.num_experts,), -1, dtype=torch.long, device=self.device
        )
        self.slot_to_expert = [-1] * capacity
        streamer.hot_cache = self

    @staticmethod
    def capacity_for_budget(streamer: ExpertStreamer, budget_bytes: int) -> int:
        """Round a runtime tensor byte budget down to complete expert slots."""
        budget_bytes = index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError("hot cache byte budget cannot be negative")
        return min(streamer.num_experts, budget_bytes // streamer.bytes_per_expert)

    def reassign(self, expert_ids: Sequence[int]) -> HotCacheUpdateStats:
        """Retain matching slots and copy only newly admitted experts."""
        desired = [index(expert_id) for expert_id in expert_ids]
        if len(desired) > self.capacity:
            raise ValueError("expert selection exceeds hot cache capacity")
        if len(set(desired)) != len(desired):
            raise ValueError("hot cache expert IDs must be unique")
        if any(
            expert_id < 0 or expert_id >= self.streamer.num_experts
            for expert_id in desired
        ):
            raise ValueError("hot cache expert ID is outside the expert range")
        wanted = set(desired)
        existing = set(self.slot_to_expert) - {-1}
        promoted = [expert_id for expert_id in desired if expert_id not in existing]
        evicted = existing - wanted
        if not promoted and not evicted:
            return HotCacheUpdateStats(0, 0, 0)
        next_slots = [
            expert_id if expert_id in wanted else -1
            for expert_id in self.slot_to_expert
        ]
        free_slots = [
            slot for slot, expert_id in enumerate(next_slots) if expert_id == -1
        ]
        for slot, expert_id in zip(free_slots, promoted):
            source_ids = torch.tensor([expert_id], dtype=torch.long, device=self.device)
            outputs = {
                name: tensor[slot : slot + 1] for name, tensor in self.tensors.items()
            }
            self.streamer._copy_source_rows(source_ids, outputs)
            next_slots[slot] = expert_id
        mapping = [-1] * self.streamer.num_experts
        for slot, expert_id in enumerate(next_slots):
            if expert_id != -1:
                mapping[expert_id] = slot
        self.expert_to_slot.copy_(
            torch.tensor(mapping, dtype=torch.long, device=self.device)
        )
        self.slot_to_expert[:] = next_slots
        return HotCacheUpdateStats(
            len(promoted), len(evicted), len(promoted) * self.bytes_per_expert
        )

    def prefetch_destinations(
        self, candidates: Sequence[int], protected_slots: Sequence[int]
    ) -> tuple[tuple[int, int], ...]:
        """Select stable unprotected victim slots for speculative admissions."""
        protected = {index(slot) for slot in protected_slots}
        if any(slot < 0 or slot >= self.capacity for slot in protected):
            raise ValueError("protected hot cache slot is outside capacity")
        existing = set(self.slot_to_expert) - {-1}
        pending = []
        for expert_id in candidates:
            expert_id = index(expert_id)
            if expert_id in existing or expert_id in pending:
                continue
            if expert_id < 0 or expert_id >= self.streamer.num_experts:
                raise ValueError("hot cache expert ID is outside the expert range")
            pending.append(expert_id)
        writable = [slot for slot in range(self.capacity) if slot not in protected]
        writable.sort(key=lambda slot: (self.slot_to_expert[slot] >= 0, slot))
        return tuple(zip(pending, writable))

    def assign_prefetch(
        self, placements: Sequence[tuple[int, int]]
    ) -> HotCacheUpdateStats:
        """Copy speculative rows into destinations selected before stream launch."""
        assignments = tuple((index(expert), index(slot)) for expert, slot in placements)
        if len({expert for expert, _ in assignments}) != len(assignments):
            raise ValueError("hot cache expert IDs must be unique")
        if len({slot for _, slot in assignments}) != len(assignments):
            raise ValueError("hot cache destination slots must be unique")
        if any(
            expert < 0
            or expert >= self.streamer.num_experts
            or slot < 0
            or slot >= self.capacity
            for expert, slot in assignments
        ):
            raise ValueError("hot cache prefetch placement is outside capacity")
        current = list(self.slot_to_expert)
        existing = set(current) - {-1}
        if any(expert in existing for expert, _ in assignments):
            raise ValueError("hot cache prefetch must not replace a resident expert")
        evictions = sum(current[slot] >= 0 for _, slot in assignments)
        for expert, slot in assignments:
            source_ids = torch.tensor([expert], dtype=torch.long, device=self.device)
            outputs = {
                name: tensor[slot : slot + 1] for name, tensor in self.tensors.items()
            }
            self.streamer._copy_source_rows(source_ids, outputs)
            current[slot] = expert
        mapping = [-1] * self.streamer.num_experts
        for slot, expert in enumerate(current):
            if expert >= 0:
                mapping[expert] = slot
        self.expert_to_slot.copy_(
            torch.tensor(mapping, dtype=torch.long, device=self.device)
        )
        self.slot_to_expert[:] = current
        return HotCacheUpdateStats(
            len(assignments), evictions, len(assignments) * self.bytes_per_expert
        )

    def lookup(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slot IDs (-1 for misses) and a hit mask for valid expert IDs."""
        if source_ids.device != self.device:
            raise ValueError("selected expert IDs must use the hot cache CUDA device")
        slots = self.expert_to_slot[source_ids.long()]
        return slots, slots >= 0

    def data_ptrs(self) -> tuple[int, ...]:
        """Expose slot and mapping pointers for stable-allocation verification."""
        return tuple(tensor.data_ptr() for tensor in self.tensors.values()) + (
            self.expert_to_slot.data_ptr(),
        )


def normalize_expert_frequency_seed(data: Mapping[str, Any]) -> torch.Tensor:
    """Normalize recorder steps or routing artifacts to CPU [layer, expert] counts."""
    if not isinstance(data, Mapping):
        raise ValueError("expert frequency seed must be a mapping")
    key = next(
        (name for name in ("logical_count", "count", "mass") if name in data), None
    )
    if key is None:
        raise ValueError("expert frequency seed requires logical_count, count, or mass")
    counts = torch.as_tensor(data[key], dtype=torch.float64, device="cpu")
    expected_ndim = 3 if key == "logical_count" else 2
    if counts.ndim != expected_ndim or counts.numel() == 0:
        raise ValueError("expert frequency seed has invalid dimensions")
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("expert frequency seed must contain finite nonnegative counts")
    return counts.sum(dim=0) if key == "logical_count" else counts


@dataclass
class _OperationalCounters:
    requested_rows: int = 0
    miss_rows: int = 0
    hot_hits: int = 0
    file_misses: int | None = None
    d2d_bytes: int = 0
    h2d_bytes: int = 0
    backing_source_bytes: int = 0
    file_source_bytes: int | None = None
    requested_unique_experts: int = 0
    promotions: int = 0
    evictions: int = 0
    migration_bytes: int = 0
    residency_bytes: int = 0
    pinned_hits: int = 0
    pinned_misses: int = 0
    pinned_admissions: int = 0
    pinned_evictions: int = 0
    file_fallbacks: int | None = None
    transfer_wait_ns: int = 0
    gather_fallbacks: int = 0


class ExpertHotCacheManager:
    """Allocate a global slot budget and observe the recorder after each forward.

    Routing counts are borrowed synchronously, never retained or counted again.
    Placement and gathers must share the streamer's serialized CUDA stream.
    Startup may set each layer's `_nvfp4_file_source_bytes_per_expert` to enable
    exact file attribution; absent metadata is reported as unknown, not guessed.
    """

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        budget_bytes: int,
        seed_path: str | None,
        dynamic: bool,
        update_prefill_tokens: int,
        min_residence_forwards: int,
        benefit_ratio: float,
        log_interval: int = 100,
        metrics_path: str | os.PathLike[str] | None = None,
        route_history_limit: int = 32,
    ) -> ExpertHotCacheManager | None:
        budget_bytes = index(budget_bytes)
        if budget_bytes == 0:
            return None
        update_prefill_tokens = index(update_prefill_tokens)
        min_residence_forwards = index(min_residence_forwards)
        log_interval = index(log_interval)
        route_history_limit = index(route_history_limit)
        if (
            budget_bytes < 0
            or update_prefill_tokens < 1
            or min_residence_forwards < 0
            or log_interval < 1
            or route_history_limit < 1
        ):
            raise ValueError("invalid expert hot cache budget or update interval")
        if not math.isfinite(benefit_ratio) or benefit_ratio < 0:
            raise ValueError(
                "expert hot cache benefit ratio must be finite and nonnegative"
            )
        streamers = {}
        for module in model.modules():
            streamer = getattr(module, "_nvfp4_expert_streamer", None)
            if streamer is None:
                continue
            layer_id = index(streamer.layer_id)
            if layer_id < 0 or layer_id in streamers:
                raise ValueError(
                    "expert hot cache requires unique nonnegative layer IDs"
                )
            streamers[layer_id] = streamer
        if not streamers:
            return None
        seed = None
        if seed_path is not None:
            path = Path(seed_path)
            if path.suffix == ".json":
                with path.open() as source:
                    payload = json.load(source)
            else:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            seed = normalize_expert_frequency_seed(payload)
            if any(
                layer_id >= seed.shape[0] or seed.shape[1] != streamer.num_experts
                for layer_id, streamer in streamers.items()
            ):
                raise ValueError(
                    "expert frequency seed does not match model layers and experts"
                )
        candidates = sorted(
            (
                -(float(seed[layer_id, expert_id]) if seed is not None else 1.0)
                * streamer.bytes_per_expert,
                expert_id,
                layer_id,
            )
            for layer_id, streamer in streamers.items()
            for expert_id in range(streamer.num_experts)
        )
        selected = {layer_id: [] for layer_id in streamers}
        remaining = budget_bytes
        for _, expert_id, layer_id in candidates:
            slot_bytes = streamers[layer_id].bytes_per_expert
            if slot_bytes <= remaining:
                selected[layer_id].append(expert_id)
                remaining -= slot_bytes
        if not any(selected.values()):
            return None
        manager = cls()
        manager.streamers = streamers
        manager.caches = {}
        manager.dynamic = dynamic
        manager.update_prefill_tokens = update_prefill_tokens
        manager.min_residence_forwards = min_residence_forwards
        manager.benefit_ratio = benefit_ratio
        manager.log_interval = log_interval
        manager.metrics_path = Path(metrics_path) if metrics_path else None
        manager.route_history_limit = route_history_limit
        manager._forward_count = 0
        manager._last_update = {layer_id: 0 for layer_id in streamers}
        manager._last_gather = {
            layer_id: streamer.last_gather_stats
            for layer_id, streamer in streamers.items()
        }
        manager._last_pinned_cache_stats = {
            layer_id: manager._pinned_cache_stats(streamer)
            for layer_id, streamer in streamers.items()
        }
        manager._counters = {
            mode: {layer_id: _OperationalCounters() for layer_id in streamers}
            for mode in ("prefill", "decode")
        }
        manager._counters["speculative"] = {
            layer_id: _OperationalCounters() for layer_id in streamers
        }
        manager._route_popularity = {
            mode: {layer_id: defaultdict(float) for layer_id in streamers}
            for mode in manager._counters
        }
        manager._route_affinity = {mode: {} for mode in manager._counters}
        for layer_id, expert_ids in selected.items():
            if expert_ids:
                cache = ExpertHotCache(streamers[layer_id], len(expert_ids))
                manager.caches[layer_id] = cache
                manager._record_update(layer_id, cache.reassign(expert_ids))
        devices = {cache.device for cache in manager.caches.values()}
        logger.info(
            "Expert hot cache startup %s",
            json.dumps(
                {
                    "requested_bytes": budget_bytes,
                    "residency_bytes": manager.residency_bytes,
                    "slots": sum(cache.capacity for cache in manager.caches.values()),
                    "layers": len(manager.caches),
                    "cuda_allocated_bytes": sum(
                        torch.cuda.memory_allocated(device) for device in devices
                    ),
                    "cuda_reserved_bytes": sum(
                        torch.cuda.memory_reserved(device) for device in devices
                    ),
                },
                sort_keys=True,
            ),
        )
        return manager

    def enable_next_layer_prefetch(self, max_candidates: int) -> None:
        """Attach default-off sparse prefetch callbacks between adjacent layers."""
        max_candidates = index(max_candidates)
        if max_candidates <= 0:
            return
        policy = SparseNextLayerPolicy(max_candidates)
        self.prefetch_coordinators = {}
        layer_ids = sorted(self.streamers)
        for current_layer, next_layer in zip(layer_ids, layer_ids[1:]):
            cache = self.caches.get(next_layer)
            if cache is None:
                continue
            next_streamer = self.streamers[next_layer]
            if next_streamer.pinned_host_cache is None:
                continue
            coordinator = ExpertPrefetchCoordinator(enabled=True, device=cache.device)
            next_streamer.prefetch_coordinator = coordinator
            self.prefetch_coordinators[next_layer] = coordinator

            def schedule(
                source_ids,
                current_layer=current_layer,
                next_layer=next_layer,
                cache=cache,
                coordinator=coordinator,
            ):
                popularity = self._route_popularity["decode"][next_layer]
                affinity = self._route_affinity["decode"].get(
                    (current_layer, next_layer), {}
                )
                resident = set(cache.slot_to_expert) - {-1}
                candidates = policy.predict(
                    source_ids.tolist(), popularity, affinity, resident
                )
                placements = cache.prefetch_destinations(
                    candidates, coordinator.protected_slots
                )
                if not placements:
                    return
                predicted = tuple(expert for expert, _ in placements)
                destination_slots = tuple(slot for _, slot in placements)

                def submit(submitted):
                    if submitted != predicted:
                        raise ValueError("prefetch candidates changed after placement")
                    update = cache.assign_prefetch(placements)
                    coordinator.record_placement(
                        cache_pollution_bytes=update.migration_bytes,
                        evictions=update.evicted_experts,
                    )
                    return update.migration_bytes

                coordinator.launch(
                    predicted, protected_slots=destination_slots, submit=submit
                )

            self.streamers[current_layer].next_layer_prefetch = schedule

    @property
    def residency_bytes(self) -> int:
        return sum(cache.capacity_bytes for cache in self.caches.values())

    def _record_update(self, layer_id: int, update: HotCacheUpdateStats) -> None:
        counters = self._counters["prefill"][layer_id]
        counters.promotions += update.promoted_experts
        counters.evictions += update.evicted_experts
        counters.migration_bytes += update.migration_bytes

    @staticmethod
    def _pinned_cache_stats(streamer: ExpertStreamer) -> tuple[int, int]:
        cache = streamer.pinned_host_cache
        if cache is None:
            return (0, 0)
        return (cache.stats.populated_rows, cache.stats.evictions)

    @staticmethod
    def _phase(forward_batch: ForwardBatch) -> str:
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return "speculative"
        if mode.is_extend_without_speculative():
            return "prefill"
        return "decode"

    @staticmethod
    def _prune_counts(counts: defaultdict, limit: int) -> None:
        if len(counts) <= limit:
            return
        retained = sorted(counts, key=lambda key: (-counts[key], key))[:limit]
        for key in tuple(counts):
            if key not in retained:
                del counts[key]

    def _record_route_statistics(self, phase: str, counts: torch.Tensor) -> None:
        active = {}
        for layer_id in self.streamers:
            row = counts[layer_id].detach().to(device="cpu", dtype=torch.float64)
            nonzero = torch.nonzero(row, as_tuple=False).flatten().tolist()
            popularity = self._route_popularity[phase][layer_id]
            for expert_id in nonzero:
                popularity[int(expert_id)] += float(row[expert_id])
            self._prune_counts(popularity, self.route_history_limit)
            active[layer_id] = sorted(
                nonzero, key=lambda expert_id: (-float(row[expert_id]), expert_id)
            )[: self.route_history_limit]
        layer_ids = sorted(active)
        for previous, following in zip(layer_ids, layer_ids[1:]):
            left = counts[previous].detach().to(device="cpu", dtype=torch.float64)
            right = counts[following].detach().to(device="cpu", dtype=torch.float64)
            affinity = self._route_affinity[phase].setdefault(
                (previous, following), defaultdict(float)
            )
            for source_expert in active[previous]:
                for target_expert in active[following]:
                    affinity[(source_expert, target_expert)] += float(
                        left[source_expert] * right[target_expert]
                    )
            self._prune_counts(affinity, self.route_history_limit)

    def snapshot_route_statistics(self) -> dict[str, dict[str, dict[str, list]]]:
        result = {}
        for phase in self._counters:
            popularity = {
                str(layer_id): [
                    [int(expert_id), float(value)]
                    for expert_id, value in sorted(
                        values.items(), key=lambda item: (-item[1], item[0])
                    )
                ]
                for layer_id, values in self._route_popularity[phase].items()
            }
            affinity = {
                f"{source}->{target}": [
                    [int(left), int(right), float(value)]
                    for (left, right), value in sorted(
                        values.items(), key=lambda item: (-item[1], item[0])
                    )
                ]
                for (source, target), values in self._route_affinity[phase].items()
            }
            result[phase] = {"popularity": popularity, "affinity": affinity}
        return result

    def _write_trace(self, phase: str) -> None:
        if self.metrics_path is None:
            return
        trace = {
            "timestamp_ns": time.time_ns(),
            "phase": phase,
            "counters": self.snapshot_counters(),
            "route_statistics": self.snapshot_route_statistics(),
        }
        with self.metrics_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(trace, sort_keys=True) + "\n")

    def snapshot_counters(self) -> dict[str, dict[str, dict[str, int | None]]]:
        """Return JSON-compatible cumulative totals and current allocation gauges."""
        result = {}
        for mode, layers in self._counters.items():
            result[mode] = {}
            for layer_id, counters in layers.items():
                row = asdict(counters)
                cache = self.caches.get(layer_id)
                row["residency_bytes"] = cache.capacity_bytes if cache else 0
                result[mode][str(layer_id)] = row
        coordinators = getattr(self, "prefetch_coordinators", {})
        if coordinators:
            result["prefetch"] = {
                str(layer_id): coordinator.snapshot_stats()
                for layer_id, coordinator in coordinators.items()
            }
        return result

    def on_expert_distribution(
        self, forward_batch: ForwardBatch, single_pass_data: Mapping[str, Any]
    ) -> None:
        """Account for fresh gathers and change slots only on profitable prefills."""
        counts = single_pass_data.get("global_physical_count")
        if counts is None:
            return
        if counts.ndim != 2 or any(
            layer_id >= counts.shape[0] or counts.shape[1] != streamer.num_experts
            for layer_id, streamer in self.streamers.items()
        ):
            raise ValueError(
                "recorder counts do not match expert cache layers and experts"
            )
        self._forward_count += 1
        mode = self._phase(forward_batch)
        prefill = mode == "prefill"
        self._record_route_statistics(mode, counts)
        qualifying = (
            self.dynamic
            and prefill
            and (forward_batch.extend_num_tokens or 0) >= self.update_prefill_tokens
        )
        for layer_id, streamer in self.streamers.items():
            row = counts[layer_id]
            stats = streamer.last_gather_stats
            if stats is not self._last_gather[layer_id]:
                self._last_gather[layer_id] = stats
                counters = self._counters[mode][layer_id]
                counters.requested_rows += stats.requested_rows
                counters.miss_rows += stats.miss_rows
                counters.hot_hits += stats.hot_hit_rows
                counters.d2d_bytes += stats.d2d_bytes
                counters.h2d_bytes += stats.h2d_bytes
                counters.backing_source_bytes += stats.source_bytes
                counters.pinned_hits += stats.pinned_host_hit_rows
                counters.pinned_misses += stats.pinned_host_miss_rows
                counters.transfer_wait_ns += getattr(stats, "transfer_wait_ns", 0)
                counters.gather_fallbacks += int(
                    getattr(stats, "gather_fallback_used", False)
                )
                counters.requested_unique_experts += int(
                    torch.count_nonzero(row).item()
                )
                file_bytes = getattr(
                    streamer.layer, "_nvfp4_file_source_bytes_per_expert", None
                )
                if file_bytes is not None:
                    counters.file_source_bytes = (
                        counters.file_source_bytes or 0
                    ) + stats.miss_rows * file_bytes
                    counters.file_misses = (counters.file_misses or 0) + (
                        stats.miss_rows if file_bytes else 0
                    )
                    counters.file_fallbacks = (counters.file_fallbacks or 0) + (
                        stats.pinned_host_miss_rows
                        if streamer.pinned_host_cache is not None
                        else stats.miss_rows
                    )
                previous_admissions, previous_evictions = self._last_pinned_cache_stats[
                    layer_id
                ]
                admissions, evictions = self._pinned_cache_stats(streamer)
                counters.pinned_admissions += admissions - previous_admissions
                counters.pinned_evictions += evictions - previous_evictions
                self._last_pinned_cache_stats[layer_id] = (admissions, evictions)
            cache = self.caches.get(layer_id)
            if (
                not qualifying
                or cache is None
                or self._forward_count - self._last_update[layer_id]
                < self.min_residence_forwards
            ):
                continue
            desired = torch.argsort(row, descending=True, stable=True)[
                : cache.capacity
            ].tolist()
            existing = set(cache.slot_to_expert) - {-1}
            promoted = set(desired) - existing
            if not promoted:
                continue
            evicted = existing - set(desired)
            saved = (
                sum(float(row[expert]) for expert in promoted)
                - sum(float(row[expert]) for expert in evicted)
            ) * streamer.bytes_per_expert
            migration = len(promoted) * streamer.bytes_per_expert
            if saved > migration * self.benefit_ratio:
                self._record_update(layer_id, cache.reassign(desired))
                self._last_update[layer_id] = self._forward_count
        if self._forward_count % self.log_interval == 0:
            self._write_trace(mode)
