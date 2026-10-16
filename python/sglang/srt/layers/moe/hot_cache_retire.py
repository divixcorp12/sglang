"""Retire the trailing slots of each layer's expert hot cache before CUDA graph capture
(SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP, step 2).

``retire_hot_slots`` is the one place a hot cache's capacity is lowered. Per layer it evicts whatever the retired
slots hold (their experts fall back to the RAM tier or NVMe, as any miss does), lowers every capacity holder in
``CAPACITY_HOLDERS``, pushes the new hot sets to the RAM-miss service, re-ranks the GPU residency updater's victims, and
releases the retired rows' VMM granules. Every base address that startup recorded (the RAM-miss copy table, graph-gather
segment tables) stays valid for the kept rows; a stray access to a retired row faults.

It refuses unless the hot cache is on VMM memory, the insert-on-miss stage is DIRECT (no scratch rows, no side-pull
row), no CUDA graph has been captured (the GPU residency updater is still disabled; capture enables it), and the RAM-miss
copy engine is not armed. After retiring it walks the holders and refuses if any capacity-named attribute it does not
know about appears, so a new holder cannot be missed silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch

from sglang.srt.layers.moe.hot_cache_vmm import HotCacheMapOutOfMemory

logger = logging.getLogger(__name__)

# Every capacity value per layer, as (owner, attribute). retire_hot_slots lowers each; _check_capacity_holders refuses
# an unlisted capacity-named attribute on these owners. The exempt ones are not a hot slot count.
CAPACITY_HOLDERS = (
    ("cache", "capacity"),
    ("cache", "capacity_bytes"),
    ("cache", "allocation_bytes"),
    ("policy", "capacity"),
    ("updater", "capacity[row]"),
    ("updater", "slot_valid[row]"),
    ("updater", "scratch_columns[row]"),
    ("row_planner", "scratch_base"),
    ("row_backend", "hot_capacity"),
    ("service", "_spec_hot[row]"),
    ("row_backend", "spec_score.hot_capacity"),
)
# Capacity-named attributes that are not the layer's slot count: the updater's bank width (an upper bound, kept), the
# pinned RAM tier's rows, and byte totals of other regions.
EXEMPT_CAPACITY_ATTRS = frozenset(
    {
        "max_capacity",
        "upper_capacity",  # the reservation's bound, fixed at construction
        "scratch_bytes",
        "prefetch_pull_bytes",
        "pinned_host_cache",
        "reserves_prefetch_pull_row",
    }
)


@dataclass(frozen=True)
class RetireResult:
    kept: Dict[int, int]
    retired: Dict[int, Tuple[int, int]]  # layer -> [first, end) of the retired slots
    evicted: Dict[int, List[int]] = field(default_factory=dict)
    freed_bytes: int = 0
    # A grow that stopped early: layer -> slots asked for but not added, and why it stopped.
    short: Dict[int, int] = field(default_factory=dict)
    short_reason: str = ""


class RetireRefused(RuntimeError):
    pass


def _service():
    from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService

    return Exl3RamMissService._instance


def census_error(manager, *, service: Any = "auto") -> Optional[str]:
    """Why the hot cache is not in a resizable, consistent state (the residency updater enabled, an armed copy
    engine, a capacity holder disagreeing with its layer's capacity), or None when it is."""
    service = _service() if service == "auto" else service
    try:
        check_retire_allowed(manager, service)
        _check_capacity_holders(
            manager, manager.gpu_residency, service, {layer: c.capacity for layer, c in manager.caches.items()}
        )
    except RetireRefused as error:
        return str(error)
    return None


def check_retire_allowed(manager, service) -> None:
    """Refuse once anything froze the slot count: a captured graph or an armed copy engine."""
    updater = getattr(manager, "gpu_residency", None)
    if updater is None:
        raise RetireRefused("hot slot retire needs the GPU residency updater")
    if bool(updater.enabled.item()):
        raise RetireRefused("hot slot retire after CUDA graph capture (the residency updater is enabled)")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RetireRefused("hot slot retire during CUDA graph capture")
    if service is not None and getattr(service, "_copy_armed", False):
        raise RetireRefused("hot slot retire after the RAM-miss copy engine armed")
    for layer_id, cache in manager.caches.items():
        if cache.vmm_backing is None:
            raise RetireRefused(f"layer {layer_id}'s hot cache is not on VMM memory")
        if cache.scratch_rows or cache.reserves_prefetch_pull_row:
            raise RetireRefused(f"layer {layer_id}'s hot cache has scratch or side-pull rows past its slots")


def retire_hot_slots(manager, keep: Mapping[int, int], *, service: Any = "auto") -> RetireResult:
    """Lower each listed layer's hot capacity to ``keep[layer]`` slots (only lower), then release the memory."""
    for layer_id, k in keep.items():
        if k > manager.caches[layer_id].capacity:
            raise RetireRefused(f"layer {layer_id}: keep {k} outside [0, {manager.caches[layer_id].capacity}]")
    return set_hot_capacity(manager, keep, service=service)


def set_hot_capacity(
    manager,
    keep: Mapping[int, int],
    *,
    service: Any = "auto",
    free_memory: Optional[Callable[[], int]] = None,
    keep_free: int = 0,
    strict: bool = True,
) -> RetireResult:
    """Set each listed layer's hot capacity to ``keep[layer]`` slots, lower or higher. Lowering evicts the retired
    slots and then releases their memory; raising maps the memory for the new slots first and then adds them FREE
    (promotions fill them). A layer grows only up to its cache's ``upper_capacity``. ``RetireResult.freed_bytes`` is
    net: negative when the call mapped more than it released.

    Growing never fails the call. Layer by layer, with ``free_memory`` given, a layer grows only while
    ``free_memory()`` less its mapping cost stays at or above ``keep_free``; a device out-of-memory while mapping
    rolls that layer's mapping back. Either way growing stops there: the layers already grown keep their slots, the
    rest keep their capacity, every holder agrees, and ``RetireResult.short`` records what was not added.

    After the resize every holder is checked against the kept capacities. ``strict`` (the default) raises on a
    disagreement; otherwise (the startup calibration) every listed layer is set back to its capacity before the
    call, the error is logged, and the result reports the whole change as short, so the launch goes on at the
    configured size."""
    service = _service() if service == "auto" else service
    check_retire_allowed(manager, service)
    updater = manager.gpu_residency
    _check_unlisted_holders(manager, updater, service, keep)  # before anything changes
    floor = 2 * getattr(updater, "victim_lanes", 0)
    for layer_id, k in keep.items():
        cache = manager.caches[layer_id]
        upper = getattr(cache, "upper_capacity", cache.capacity)
        if not 0 <= k <= upper:
            raise RetireRefused(f"layer {layer_id}: keep {k} outside [0, {upper}]")
        if k < floor:
            raise RetireRefused(f"layer {layer_id}: keep {k} below twice the graph-gather victim lanes ({floor})")
        if k > updater.max_capacity:
            raise RetireRefused(f"layer {layer_id}: keep {k} beyond the residency bank's {updater.max_capacity} slots")
    before = {layer_id: manager.caches[layer_id].capacity for layer_id in keep}
    kept, retired, evicted, freed = {}, {}, {}, 0
    short, short_reason = {}, ""
    for layer_id, k in keep.items():
        cache = manager.caches[layer_id]
        c = cache.capacity
        urow = updater.layer_ids.index(layer_id)
        if k > c and not short_reason and free_memory is not None:
            need, free = cache.vmm_backing.bytes_to_back(k), free_memory()
            if free - need < keep_free:
                short_reason = (
                    f"layer {layer_id} needs {need / 2**20:.1f} MiB with {free / 2**20:.1f} MiB free, "
                    f"{keep_free / 2**20:.1f} MiB must stay free"
                )
        if k > c and not short_reason:
            try:
                freed -= cache.vmm_backing.set_backed_rows(k)  # memory first, then the slots that use it
            except HotCacheMapOutOfMemory as error:
                short_reason = f"layer {layer_id}: {error}"
        if k > c and short_reason:
            short[layer_id] = k - c
            kept[layer_id], retired[layer_id] = c, (c, c)
            continue
        if k > c:
            _set_capacity(cache, manager.residency_policies.get(layer_id), updater, urow, k)
        elif k < c:
            evicted[layer_id] = _evict_tail(cache, updater, urow, k, c)
            _set_capacity(cache, manager.residency_policies.get(layer_id), updater, urow, k)
        kept[layer_id], retired[layer_id] = k, (k, c)
    _refresh_service(manager, updater, service, kept)
    if hasattr(updater, "_rank_victims"):
        updater._rank_victims(updater.route_counts > 0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for layer_id, (k, c) in retired.items():
        if k < c:
            freed += manager.caches[layer_id].vmm_backing.release_tail(k)
    try:
        _check_capacity_holders(manager, updater, service, kept)
    except RetireRefused as error:
        if strict:
            raise
        logger.warning(
            "hot cache capacity: the holders disagree after the resize (%s); setting every layer back to its "
            "capacity before it and starting at the configured size",
            error,
        )
        back = set_hot_capacity(manager, before, service=service)
        return RetireResult(
            back.kept,
            {layer_id: (c, c) for layer_id, c in back.kept.items()},
            evicted,
            freed + back.freed_bytes,
            {layer_id: keep[layer_id] - c for layer_id, c in before.items() if keep[layer_id] != c},
            f"rolled back: {error}",
        )
    logger.info(
        "hot cache capacity: %d -> %d slots over %d layers (%d retired, %d added), %d experts evicted, %+.1f MiB",
        sum(c for _, c in retired.values()),
        sum(kept.values()),
        len(kept),
        sum(max(0, c - k) for k, c in retired.values()),
        sum(max(0, k - c) for k, c in retired.values()),
        sum(len(v) for v in evicted.values()),
        -freed / 2**20,
    )
    if short:
        logger.warning(
            "hot cache capacity: grow stopped short by %d slots over %d layers (%s)",
            sum(short.values()),
            len(short),
            short_reason,
        )
    return RetireResult(kept, retired, evicted, freed, short, short_reason)


def _evict_tail(cache, updater, urow: int, k: int, c: int) -> List[int]:
    """Clear slots [k, c) on the device bank (authoritative under GPU residency) and the host lists."""
    tail = updater.slot_to_expert[urow, k:c]
    experts = [int(e) for e in tail.tolist() if e >= 0]
    if experts:
        updater.mapping[urow, torch.tensor(experts, device=updater.mapping.device)] = -1
    updater.slot_to_expert[urow, k:c] = -1
    updater.slot_state[urow, k:c] = 0  # FREE, and invalid once capacity drops
    for s in range(k, c):
        cache.slot_to_expert[s] = -1
    return experts


def _set_capacity(cache, policy, updater, urow: int, k: int) -> None:
    """Every capacity holder of one layer to ``k`` slots. Below the old count the tail was evicted first; above it
    the new slots start FREE and empty (the device bank already holds them so: -1 and FREE past the capacity)."""
    bpe = cache.bytes_per_expert
    c = cache.capacity
    cache.capacity = k
    cache.capacity_bytes = k * bpe
    cache.allocation_bytes = k * bpe
    if k < c:
        del cache.slot_to_expert[k:]
        del cache.slot_states[k:]
        del cache._slot_generations[k:]
    else:
        free = type(cache.slot_states[0])(0) if cache.slot_states else 0
        cache.slot_to_expert.extend([-1] * (k - c))
        cache.slot_states.extend([free] * (k - c))
        cache._slot_generations.extend([0] * (k - c))
        updater.slot_to_expert[urow, c:k] = -1
        updater.slot_state[urow, c:k] = 0  # FREE
    cache.slot_state = updater.slot_state[urow, :k]
    cache.slot_generations = updater.slot_generations[urow, :k]
    if policy is not None:
        policy.capacity = k
    updater.capacity[urow] = k
    updater.slot_valid[urow].copy_(updater.slot_ids < k)
    if hasattr(updater, "scratch_columns"):
        updater.scratch_columns[urow].copy_(k + updater.miss_columns)
    planner = getattr(cache.streamer, "row_planner", None)
    if planner is not None:
        planner.scratch_base = k


def _refresh_service(manager, updater, service, kept) -> None:
    """The RAM-miss service's per-row hot capacity, its spec-score rows, and the hot sets it reads."""
    if service is None:
        return
    for layer_id, k in kept.items():
        row = service.row_of(layer_id)
        backend = service._row_backends.get(row)
        if backend is not None:
            backend.hot_capacity = k
        if row in getattr(service, "_spec_hot", {}):
            service._spec_hot[row] = (service._spec_hot[row][0], k)
    if getattr(service, "_spec_gpu", None) is not None:
        service._bind_spec_scores()
    if getattr(service, "gpu_hot_enabled", False):
        service._hot_snapshot.copy_(updater.slot_to_expert)
        service._refresh_hot_lists()
        service.before_host_use()
        try:
            for layer_id, experts in service._hot_lists.items():
                service.host.set_hot(service.row_of(layer_id), experts)
        finally:
            service.after_host_use()


def _capacity_attrs(obj) -> Dict[str, Any]:
    names = vars(obj) if hasattr(obj, "__dict__") else {}
    return {n: v for n, v in names.items() if "capacity" in n.lower() and n not in EXEMPT_CAPACITY_ATTRS}


def _owners(manager, service, layer_id) -> Dict[str, Any]:
    cache = manager.caches[layer_id]
    owners = {
        "cache": cache,
        "policy": manager.residency_policies.get(layer_id),
        "row_planner": getattr(cache.streamer, "row_planner", None),
    }
    if service is not None:
        owners["row_backend"] = service._row_backends.get(service.row_of(layer_id))
    return {name: obj for name, obj in owners.items() if obj is not None}


def _check_unlisted_holders(manager, updater, service, layers) -> None:
    """Refuse a capacity-named attribute retire does not know how to lower."""
    listed = {attr for _, attr in CAPACITY_HOLDERS}
    for layer_id in layers:
        for owner_name, obj in _owners(manager, service, layer_id).items():
            for attr in _capacity_attrs(obj):
                if attr not in listed:
                    raise RetireRefused(f"unlisted capacity holder {owner_name}.{attr}: add it to CAPACITY_HOLDERS")
    for attr in _capacity_attrs(updater):
        if attr != "capacity":
            raise RetireRefused(f"unlisted capacity holder updater.{attr}: add it to CAPACITY_HOLDERS")


def _check_capacity_holders(manager, updater, service, kept) -> None:
    """Refuse if a kept layer's capacity holder disagrees with what it kept."""
    for layer_id, k in kept.items():
        owners = _owners(manager, service, layer_id)
        for owner_name, obj in owners.items():
            for attr, value in _capacity_attrs(obj).items():
                if attr in ("capacity", "hot_capacity") and int(value) != k:
                    raise RetireRefused(f"layer {layer_id}: {owner_name}.{attr} is {value}, not {k}")
        urow = updater.layer_ids.index(layer_id)
        if int(updater.capacity[urow]) != k or int(updater.slot_valid[urow].sum()) != k:
            raise RetireRefused(f"layer {layer_id}: updater capacity/slot_valid disagree with {k}")
        if owners.get("row_planner") is not None and owners["row_planner"].scratch_base != k:
            raise RetireRefused(f"layer {layer_id}: row_planner.scratch_base is not {k}")
    # A row's SpecScoreRow scores the prefetch for its *target* row (the L+2 scorer), so it carries that row's hot
    # capacity, not its own layer's; and a resized target changes the score of a row that was not resized itself.
    for layer_id in manager.caches if service is not None else ():
        backend = service._row_backends.get(service.row_of(layer_id))
        spec = getattr(backend, "spec_score", None)
        if spec is not None:
            target = _layer_of_row(manager, service, spec.target)
            want = manager.caches[target].capacity if target is not None else None
            if want is not None and int(spec.hot_capacity) != int(want):
                raise RetireRefused(
                    f"layer {layer_id}: spec_score.hot_capacity is {spec.hot_capacity}, not its target layer "
                    f"{target}'s {want}"
                )


def _layer_of_row(manager, service, row: int) -> Optional[int]:
    """The hot-cached layer whose RAM-miss service row is ``row``."""
    for layer_id in manager.caches:
        if service.row_of(layer_id) == row:
            return layer_id
    return None
