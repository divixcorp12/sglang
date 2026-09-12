"""Fixed CUDA slots for frequently selected host-resident expert rows."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_stream import ExpertStreamer, _tensor_data


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
