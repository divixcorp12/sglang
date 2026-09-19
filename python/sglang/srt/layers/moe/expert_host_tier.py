"""CPU-side pieces of the pinned host expert tier: the slot LRU and exact slabs.

They are kept apart from the streamer, and import the CUDA host-registration
helpers only when asked to register, so the slot policy and the slab layout
are testable on a CPU-only host.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Collection,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
)

import torch

PAGE_BYTES = 4096


class PinnedGatherResult(NamedTuple):
    """Row counters of one pinned-tier gather."""

    hit_rows: int
    miss_rows: int
    populated_bytes: int
    fallback_used: bool


class PinnedSlotTable(Protocol):
    """The slot bookkeeping ``ExpertPinnedHostCache`` delegates to.

    ``PinnedSlotLRU`` is the default. A format whose slots are also managed by
    someone else (a native reader thread) supplies its own through
    ``pinned_tier_options(layer)["slot_table"]``; ``before_host_use(cache)`` runs
    before every host-side use of the tier, so that owner can pause and the cache
    can refresh its device slot map. Host uses nest (``ExpertPinnedHostCache.host_use``
    calls both hooks at every level), so a table that pauses an owner counts depth
    and pauses on the outermost ``before`` and resumes on the outermost ``after``.
    """

    capacity: int

    @property
    def slot_to_expert(self) -> Sequence[int]: ...

    @property
    def expert_to_slot(self) -> Mapping[int, int]: ...

    def __contains__(self, expert_id: int) -> bool: ...

    def touch(self, expert_id: int) -> None: ...

    def assign(
        self, expert_id: int, protected: Collection[int] = frozenset()
    ) -> tuple[int, Optional[int]]: ...

    def release(self, slot: int) -> None: ...

    def mapping(self, num_experts: int) -> list[int]: ...

    def before_host_use(self, cache: Any) -> None: ...

    def after_host_use(self, cache: Any) -> None: ...


class PinnedSlotLRU:
    """Slot bookkeeping of a bounded host row cache.

    Free slots are handed out lowest index first. A full cache evicts the
    least recently used expert that neither ``is_pinned`` nor the caller's
    ``protected`` set covers. When only ``protected`` experts can go, the
    oldest of them does: that is an over-capacity call reassigning its own
    slots. ``touch`` and ``assign`` are O(1); only covered experts are
    skipped when choosing a victim.
    """

    def __init__(
        self, capacity: int, is_pinned: Optional[Callable[[int], bool]] = None
    ):
        self.capacity = int(capacity)
        self.is_pinned = is_pinned
        self.slot_to_expert = [-1] * self.capacity
        # Oldest first: iteration order is the eviction order.
        self.expert_to_slot: "OrderedDict[int, int]" = OrderedDict()
        self._free = list(range(self.capacity))
        heapq.heapify(self._free)

    def __contains__(self, expert_id: int) -> bool:
        return expert_id in self.expert_to_slot

    def touch(self, expert_id: int) -> None:
        self.expert_to_slot.move_to_end(expert_id)

    def assign(
        self, expert_id: int, protected: Collection[int] = frozenset()
    ) -> tuple[int, Optional[int]]:
        """Give ``expert_id`` a slot; returns the slot and the evicted expert or None.

        ``protected`` holds the experts of the caller's current request, which
        are evicted only when nothing else can be.
        """
        if expert_id in self.expert_to_slot:
            raise ValueError(f"expert {expert_id} already holds a pinned slot")
        evicted = None
        if self._free:
            slot = heapq.heappop(self._free)
        else:
            evicted = self._victim(protected)
            slot = self.expert_to_slot.pop(evicted)
        self.slot_to_expert[slot] = expert_id
        self.expert_to_slot[expert_id] = slot
        return slot, evicted

    def _victim(self, protected: Collection[int]) -> int:
        fallback = None
        for expert_id in self.expert_to_slot:
            if self.is_pinned is not None and self.is_pinned(expert_id):
                continue
            if expert_id not in protected:
                return expert_id
            if fallback is None:
                fallback = expert_id
        if fallback is not None:
            return fallback
        raise RuntimeError("every pinned host slot holds a protected expert")

    def release(self, slot: int) -> None:
        """Free ``slot``, forgetting its expert (used to roll back a failed read)."""
        expert_id = self.slot_to_expert[slot]
        if expert_id < 0:
            return
        self.expert_to_slot.pop(expert_id, None)
        self.slot_to_expert[slot] = -1
        heapq.heappush(self._free, slot)

    def mapping(self, num_experts: int) -> list[int]:
        """Each expert's slot, or -1."""
        mapping = [-1] * num_experts
        for slot, expert_id in enumerate(self.slot_to_expert):
            if expert_id >= 0:
                mapping[expert_id] = slot
        return mapping

    def before_host_use(self, cache) -> None:
        """Nothing else owns these slots."""
        return None

    def after_host_use(self, cache) -> None:
        return None


def allocate_host_slab(
    rows: int, row_shape: tuple[int, ...], dtype: torch.dtype, *, register: bool
) -> torch.Tensor:
    """A page-aligned ``[rows, *row_shape]`` host tensor of exactly its size.

    PyTorch's pinned allocator rounds every allocation up to a power of two,
    so a slab just past 1 GiB would pin 2 GiB. This one takes plain host
    memory plus one page of alignment slack, and ``register`` pins it with
    ``cudaHostRegister`` in row-aligned chunks, as the host arena does.
    Page alignment also lets io_uring fill it with ``O_DIRECT``.
    """
    shape = (int(rows),) + tuple(int(dimension) for dimension in row_shape)
    nbytes = math.prod(shape) * dtype.itemsize
    storage = torch.empty(nbytes + PAGE_BYTES, dtype=torch.uint8, device="cpu")
    start = (-storage.data_ptr()) % PAGE_BYTES
    slab = storage[start : start + nbytes].view(dtype).view(shape)
    if register and nbytes:
        # Imported here: expert_stream imports this module while the model
        # loader package is still importing, and mem_cache.pool_host's package
        # import is heavy.
        from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

        _cuda_host_register(slab, registration_granularity_bytes=nbytes // shape[0])
    return slab


def release_host_slabs(slabs: Sequence[torch.Tensor]) -> None:
    """Unregister slabs that ``allocate_host_slab(..., register=True)`` registered."""
    if not slabs:
        return
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

    for slab in slabs:
        _cuda_host_unregister(slab)
