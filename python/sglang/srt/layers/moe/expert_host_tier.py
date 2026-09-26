"""CPU-side pieces of the pinned host expert tier: the slot LRU and exact slabs.

They are kept apart from the streamer, and import the CUDA host-registration
helpers only when asked to register, so the slot policy and the slab layout
are testable on a CPU-only host.
"""

from __future__ import annotations

import ctypes
import heapq
import itertools
import math
import weakref
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
)

import torch

from sglang.srt.layers.engram_row_cache import cache_stats_sink

if TYPE_CHECKING:
    from sglang.srt.layers.moe.host_numa import Placement

PAGE_BYTES = 4096
_LRU_INDEX = itertools.count()
_LIVE_LRUS: "weakref.WeakSet[PinnedSlotLRU]" = weakref.WeakSet()


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


class PinnedRowFills(Protocol):
    """Asynchronous reads of a layer's missing rows into its pinned slots (SGLANG_DSV41_ENABLE_PREFILL_FILLS).

    Driven by ``ExpertPinnedHostCache`` inside one host use. ``fill_begin`` claims slots for ``experts`` in order,
    until one has no victim (a ``protected`` row goes only with ``fallback``), maps them at once and starts reading;
    it returns the claimed prefix's slots and the evictions. A claimed slot is never a victim until ``fill_end``.
    ``fill_wait(rows)`` returns once the first ``rows`` claimed rows are in their slabs, and raises if the fill failed
    first. ``fill_landed`` returns how many claimed rows have landed so far, a prefix of the claim order, without
    blocking. ``fill_end`` joins the reads; False means the fill failed and released its rows that did not land.
    """

    def fill_begin(
        self, experts: Sequence[int], protected: Iterable[int], fallback: bool
    ) -> tuple[list[int], int]: ...

    def fill_wait(self, rows: int) -> None: ...

    def fill_landed(self) -> int: ...

    def fill_end(self) -> bool: ...


class PinnedSlotLRU:
    """Slot bookkeeping of a bounded host row cache.

    Free slots are handed out lowest index first. A full cache evicts the
    least recently used expert that neither ``is_pinned`` nor the caller's
    ``protected`` set covers. When only ``protected`` experts can go, the
    oldest of them does: that is an over-capacity call reassigning its own
    slots. ``touch`` and ``assign`` are O(1); only covered experts are
    skipped when choosing a victim.

    The counters (``stats()``) only count: none is read by a decision. ``hits``
    counts ``touch`` calls, which the tier makes once per resident route (a
    repeated expert counts each time); ``admissions`` counts ``assign`` calls,
    each a distinct missing expert. The two are therefore not one unit, so a
    miss rate needs the tier's route-level ``lookup_misses`` (``tier_snapshot``).
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
        self.hits = 0
        self.admissions = 0
        self.evictions = 0
        # Evictions that took an expert of the caller's own request.
        self.protected_evictions = 0
        self.releases = 0
        # The tier's route-level counters, bound when it first uses this table.
        self._tier_stats = None
        self._index = next(_LRU_INDEX)
        _LIVE_LRUS.add(self)
        self._sink = cache_stats_sink()

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "occupancy": len(self.expert_to_slot),
            "hits": self.hits,
            "admissions": self.admissions,
            "evictions": self.evictions,
            "protected_evictions": self.protected_evictions,
            "releases": self.releases,
        }

    def __contains__(self, expert_id: int) -> bool:
        return expert_id in self.expert_to_slot

    def touch(self, expert_id: int) -> None:
        self.expert_to_slot.move_to_end(expert_id)
        self.hits += 1
        if self._sink is not None:
            self._sink.maybe_write("pinned_tier", tier_snapshot)

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
            self.evictions += 1
            self.protected_evictions += evicted in protected
        self.slot_to_expert[slot] = expert_id
        self.expert_to_slot[expert_id] = slot
        self.admissions += 1
        if self._sink is not None:
            self._sink.maybe_write("pinned_tier", tier_snapshot)
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
        self.releases += 1

    def mapping(self, num_experts: int) -> list[int]:
        """Each expert's slot, or -1."""
        mapping = [-1] * num_experts
        for slot, expert_id in enumerate(self.slot_to_expert):
            if expert_id >= 0:
                mapping[expert_id] = slot
        return mapping

    def before_host_use(self, cache) -> None:
        """Nothing else owns these slots; remember the tier's counters."""
        if self._tier_stats is None:
            self._tier_stats = getattr(cache, "stats", None)

    def after_host_use(self, cache) -> None:
        return None


def tier_snapshot() -> dict:
    """Cumulative counters of every live ``PinnedSlotLRU``, in construction order.

    The scalars sum over the layers; ``layers`` holds each table's own counters.
    ``lookup_*`` and ``populated_bytes`` are the tier's route-level counters
    (``PinnedHostCacheStats``), present once a tier has used its table.
    """
    tables = sorted(_LIVE_LRUS, key=lambda table: table._index)
    layers = [table.stats() for table in tables]
    snapshot = {
        key: sum(layer[key] for layer in layers)
        for key in ("capacity", "occupancy", "hits", "admissions", "evictions", "protected_evictions", "releases")
    }
    tier = [table._tier_stats for table in tables if table._tier_stats is not None]
    for key in ("lookup_hits", "lookup_misses", "populated_rows", "populated_bytes"):
        snapshot[key] = sum(getattr(stats, key) for stats in tier)
    snapshot["layers"] = {
        key: [layer[key] for layer in layers] for key in ("occupancy", "hits", "admissions", "evictions")
    }
    return snapshot


def allocate_host_slab(
    rows: int,
    row_shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    register: bool,
    placement: "Placement" = (),
) -> torch.Tensor:
    """A page-aligned ``[rows, *row_shape]`` host tensor of exactly its size.

    PyTorch's pinned allocator rounds every allocation up to a power of two,
    so a slab just past 1 GiB would pin 2 GiB. This one takes plain host
    memory plus one page of alignment slack, and ``register`` pins it with
    ``cudaHostRegister`` in row-aligned chunks, as the host arena does.
    Page alignment also lets io_uring fill it with ``O_DIRECT``.

    A non-empty ``placement`` (host_numa) maps the slab on its own and binds
    its rows to NUMA nodes in proportion before any page is touched.
    """
    shape = (int(rows),) + tuple(int(dimension) for dimension in row_shape)
    nbytes = math.prod(shape) * dtype.itemsize
    if placement:
        from sglang.srt.layers.moe.host_numa import allocate_bound, split_rows

        row_bytes = nbytes // shape[0] if shape[0] else 0
        storage = allocate_bound(nbytes, split_rows(shape[0], placement), row_bytes)
        start = 0
    else:
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


# The list only makes the quarantine countable. The protection is the extra reference taken below: interpreter
# finalization clears module globals, which would free the slabs of a list that were the only owner.
_QUARANTINED: list[torch.Tensor] = []


def quarantine_host_slabs(slabs: Iterable[torch.Tensor]) -> None:
    """Keep slabs alive, and registered, until the process ends.

    For a tier whose GPU readers cannot be shown to have finished (a CUDA error, or a synchronize that did not
    return): recycling that storage could feed a reader another expert's bytes. Each slab gets a reference that
    is never released.
    """
    for slab in slabs:
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(slab))
        _QUARANTINED.append(slab)


def quarantined_slab_count() -> int:
    return len(_QUARANTINED)
