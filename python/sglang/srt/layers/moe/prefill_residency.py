"""Layer-major prefill expert residency (SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS).

A layer-major pass runs every chunk of the uncached suffix through layer L before L+1, so the experts one chunk of L
reads are the ones the next chunk of L routes to again. Without residency each chunk reads them again: the eager gather
stages a layer's misses in a staging buffer every group overwrites, and the prefill share keeps only ~64 of them in the
pinned tier. ``PrefillResidency`` keeps them on the GPU for the rest of the layer's chunks instead.

One buffer of ``rows`` expert rows serves the active layer only. ``ExpertStreamer._gather_cached`` asks it for every
hot-cache miss: a resident expert is scattered from it into staging like a hot hit, a real miss is read as before and
then adopted into the buffer's next free row. ``activate`` (one per layer, from ``Exl3LayerMajorResidency``) forgets the
previous layer's rows. The bytes are copies of the same rows, so the gather's output is the same.

Stage 0: the buffer is a plain allocation, so a launch must leave its memory free (a smaller hot cache).

Stage 1 (SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND): ``LentPrefillResidency`` takes its rows from the late layers' hot-cache
slots instead. The pass loop never runs those layers (their MoE runs in ``finish_pass``), so for one forward their slots
hold residency rows; ``restore`` reloads each written slot's original expert from the pinned RAM tier before the tail.
Slot maps, generations, scores and lease tables are not touched, except that a slot whose reload failed, or whose map
changed under the lend, is evicted (mapping -1, slot FREE).
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SLOT_FREE = 0
_SLOT_READY = 3


def _specs_of(streamer) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {name: (streamer.spec(name).row_shape, streamer.spec(name).dtype) for name in streamer.tensor_names}


def _streamer_fits(streamer, num_experts: int, specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]]) -> bool:
    if streamer.num_experts != num_experts or set(streamer.tensor_names) != set(specs):
        return False
    return all(
        (tuple(streamer.spec(name).row_shape), streamer.spec(name).dtype) == (tuple(shape), dtype)
        for name, (shape, dtype) in specs.items()
    )


@triton.jit
def _lent_rows_copy_kernel(
    base_ptrs,
    slots,
    entries,
    flat_ptr,
    flat_rows,
    row_bytes,
    TO_LENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program i copies between flat row ``flat_rows[i]`` and lent entry ``entries[i]``'s slot row.

    ``base_ptrs[e]`` is the address of entry e's donor-layer tensor and ``slots[e]`` its slot, so one launch serves
    rows of many layers."""
    entry = tl.load(entries + tl.program_id(0)).to(tl.int64)
    flat_row = tl.load(flat_rows + tl.program_id(0)).to(tl.int64)
    lent_ptr = tl.load(base_ptrs + entry).to(tl.pointer_type(tl.uint8))
    slot = tl.load(slots + entry).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < row_bytes
    if TO_LENT:
        values = tl.load(flat_ptr + flat_row * row_bytes + offsets, mask=mask, other=0)
        tl.store(lent_ptr + slot * row_bytes + offsets, values, mask=mask)
    else:
        values = tl.load(lent_ptr + slot * row_bytes + offsets, mask=mask, other=0)
        tl.store(flat_ptr + flat_row * row_bytes + offsets, values, mask=mask)


class PrefillResidency:
    """``rows`` expert rows per tensor name, and which expert of the active layer each holds."""

    def __init__(self, *, rows: int, num_experts: int, specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
                 device: torch.device):
        if rows <= 0:
            raise ValueError("prefill residency needs at least one row")
        self.rows = rows
        self.num_experts = num_experts
        self.specs = dict(specs)
        self.tensors = {
            name: torch.empty((rows,) + tuple(row_shape), dtype=dtype, device=device)
            for name, (row_shape, dtype) in self.specs.items()
        }
        self.row_of_expert = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        self.used = 0
        self.streamer: Any = None
        # Per-pass counters, for the pass's log line.
        self.adopted_rows = 0
        self.hit_rows = 0
        logger.info(
            "layer-major expert residency: %d rows, %.1f MiB",
            rows,
            sum(t.numel() * t.element_size() for t in self.tensors.values()) / 2**20,
        )

    @classmethod
    def for_streamer(cls, streamer, rows: int) -> "PrefillResidency":
        return cls(rows=rows, num_experts=streamer.num_experts, specs=_specs_of(streamer),
                   device=streamer.hot_cache.device)

    def fits(self, streamer) -> bool:
        """Whether ``streamer``'s rows have this buffer's names, shapes and dtypes."""
        return _streamer_fits(streamer, self.num_experts, self.specs)

    def activate(self, streamer) -> None:
        """Serve ``streamer``'s layer from now on, with no rows resident."""
        self.deactivate()
        if not self.fits(streamer):
            raise ValueError(f"layer {streamer.layer_id}'s expert rows do not fit the residency buffer")
        self.row_of_expert.fill_(-1)
        self.used = 0
        self.streamer = streamer
        streamer.prefill_residency = self

    def deactivate(self) -> None:
        if self.streamer is not None:
            self.streamer.prefill_residency = None
            self.streamer = None

    def lookup(self, source_ids: torch.Tensor) -> torch.Tensor:
        """Each source's residency row, -1 when it is not resident."""
        return self.row_of_expert[source_ids.long()]

    def adopt(self, experts: torch.Tensor, staged: Mapping[str, torch.Tensor], count: int) -> int:
        """Copy the first ``count`` staged rows (``experts``' rows, in order) into free rows; returns how many fit.

        Queued on the current stream after the gather that filled ``staged``, so it reads the gathered bytes; the
        buffer is never reused within a layer, so nothing can overwrite a row the gather still reads."""
        take = min(count, self.rows - self.used)
        if take <= 0:
            return 0
        for name, buffer in self.tensors.items():
            # The staged misses are the staging buffer's first rows, and the free rows are contiguous.
            buffer[self.used : self.used + take].copy_(staged[name][:take])
        rows = torch.arange(self.used, self.used + take, dtype=torch.long, device=self.row_of_expert.device)
        self.row_of_expert[experts[:take].long()] = rows
        self.used += take
        self.adopted_rows += take
        return take

    def scatter(self, rows: torch.Tensor, positions: torch.Tensor, count: int,
                outputs: Mapping[str, torch.Tensor]) -> None:
        """Copy residency ``rows`` to staging ``positions`` (``count`` of each) in every output."""
        if count <= 0:
            return
        from sglang.srt.layers.moe.expert_stream import _scatter_hot_rows_kernel

        for name, output in outputs.items():
            buffer = self.tensors[name]
            if not output.is_cuda:
                output[positions] = buffer[rows]
                continue
            # Row by row, as the hot hits are: an index_select would hold up to a whole group's rows twice.
            row_bytes = buffer[0].numel() * buffer.element_size()
            _scatter_hot_rows_kernel[(count, triton.cdiv(row_bytes, 1024))](
                buffer.view(torch.uint8), rows, positions, output.view(torch.uint8), row_bytes, BLOCK=1024
            )
        self.hit_rows += count

    def take_pass_counts(self) -> tuple[int, int]:
        counts = (self.adopted_rows, self.hit_rows)
        self.adopted_rows = self.hit_rows = 0
        return counts


class LentRow(NamedTuple):
    layer_id: int
    cache: Any
    slot: int


def _slot_tables(manager, layer_id: int, cache):
    """A layer's (slot_to_expert, generations, expert_to_slot, slot_state) on the device, authoritative ones.

    Under the GPU residency update the device bank owns slot_to_expert and the host list is stale."""
    updater = getattr(manager, "gpu_residency", None)
    if updater is not None and getattr(cache, "device_residency", None) is updater:
        slot_to_expert = updater.slot_to_expert[updater.layer_ids.index(layer_id), : cache.capacity]
    else:
        slot_to_expert = torch.tensor(cache.slot_to_expert, dtype=torch.long, device=cache.expert_to_slot.device)
    return slot_to_expert, cache.slot_generations, cache.expert_to_slot, cache.slot_state


class LentPrefillResidency(PrefillResidency):
    """``PrefillResidency`` whose rows are hot-cache slots of layers the pass does not run, lent for one forward.

    ``plan`` lists the lendable slots in the order they are taken. Row r is ``plan[r]``'s slot in every tensor of
    its layer's cache. ``adopt`` overwrites the next free rows; ``restore`` puts every written slot's original expert
    back from the pinned RAM tier. The slot map is never written: what a slot maps to at ``begin`` is what it maps to
    at ``restore``, or the slot is evicted.
    """

    def __init__(self, *, manager, plan: Sequence[LentRow], num_experts: int,
                 specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]], device: torch.device):
        if not plan:
            raise ValueError("lending needs at least one hot-cache slot")
        self.manager = manager
        self.plan = list(plan)
        self.num_experts = num_experts
        self.specs = dict(specs)
        self.device = device
        self.tensors = {}
        self.row_of_expert = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        self.used = 0
        # Every layer of the pass restarts at row 0, so the rows to reload are those below the most any layer used.
        self.written = 0
        self.streamer: Any = None
        self.adopted_rows = 0
        self.hit_rows = 0
        self.restored = False
        self.reloaded_rows = 0
        self.pinned_hit_rows = 0
        self.pinned_miss_rows = 0
        self.fallback_gathers = 0
        self.evicted_rows = 0
        self.reload_ms = 0.0
        self._tables: Optional[tuple[torch.Tensor, dict[str, torch.Tensor]]] = None
        self._take_snapshot()
        self.rows = len(self.plan)
        logger.info("layer-major expert residency: %d hot-cache slots lent over %d layers", self.rows,
                    len({row.layer_id for row in self.plan}))

    def _read(self) -> list[list[int]]:
        """Each plan entry's [expert, generation, mapped slot, state] now, in one host sync."""
        parts = []
        for layer_id, group in itertools.groupby(self.plan, key=lambda row: row.layer_id):
            group = list(group)
            cache = group[0].cache
            slots = torch.tensor([row.slot for row in group], dtype=torch.long, device=self.device)
            slot_to_expert, generations, expert_to_slot, state = _slot_tables(self.manager, layer_id, cache)
            experts = slot_to_expert[slots]
            mapped = torch.where(experts >= 0, expert_to_slot[experts.clamp(min=0)], torch.full_like(experts, -1))
            parts.append(torch.stack((experts, generations[slots], mapped, state[slots].long())))
        return torch.cat(parts, dim=1).t().tolist()

    def _take_snapshot(self) -> None:
        """Record each slot's expert, generation and mapping; a slot that is neither READY nor FREE is not lent."""
        snapshot = self._read()
        keep = [i for i, row in enumerate(snapshot) if row[3] in (_SLOT_FREE, _SLOT_READY)]
        if len(keep) != len(snapshot):
            logger.warning("layer-major expert residency: %d slots are mid-promotion and not lent",
                           len(snapshot) - len(keep))
        self.plan = [self.plan[i] for i in keep]
        self.snapshot = [snapshot[i] for i in keep]
        # A FREE slot holds no expert worth restoring, whatever stale id the bank keeps for it.
        self.original = [row[0] if row[3] == _SLOT_READY else -1 for row in self.snapshot]

    def verify(self) -> list[int]:
        """Plan entries whose expert, generation, mapping or state differs from the snapshot."""
        now = self._read()
        return [i for i, (a, b) in enumerate(zip(now, self.snapshot)) if a != b]

    def _device_tables(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._tables is None:
            slots = torch.tensor([row.slot for row in self.plan], dtype=torch.long, device=self.device)
            bases = {
                name: torch.tensor([row.cache.tensors[name].data_ptr() for row in self.plan], dtype=torch.long,
                                   device=self.device)
                for name in self.specs
            }
            self._tables = (slots, bases)
        return self._tables

    def _copy(self, entries: torch.Tensor, flat: Mapping[str, torch.Tensor], flat_rows: torch.Tensor, count: int,
              *, to_lent: bool) -> None:
        """Copy ``count`` rows between lent ``entries`` and rows ``flat_rows`` of ``flat``, in every tensor."""
        for name, output in flat.items():
            if not output.is_cuda:
                for entry, flat_row in zip(entries.tolist(), flat_rows.tolist()):
                    lent = self.plan[entry].cache.tensors[name][self.plan[entry].slot]
                    if to_lent:
                        lent.copy_(output[flat_row])
                    else:
                        output[flat_row].copy_(lent)
                continue
            slots, bases = self._device_tables()
            row_bytes = output[0].numel() * output.element_size()
            _lent_rows_copy_kernel[(count, triton.cdiv(row_bytes, 1024))](
                bases[name], slots, entries, output.view(torch.uint8), flat_rows, row_bytes,
                TO_LENT=to_lent, BLOCK=1024,
            )

    def adopt(self, experts: torch.Tensor, staged: Mapping[str, torch.Tensor], count: int) -> int:
        take = min(count, self.rows - self.used)
        if take <= 0:
            return 0
        entries = torch.arange(self.used, self.used + take, dtype=torch.long, device=self.device)
        self._copy(entries, staged, torch.arange(take, dtype=torch.long, device=self.device), take, to_lent=True)
        self.row_of_expert[experts[:take].long()] = entries
        self.used += take
        self.written = max(self.written, self.used)
        self.adopted_rows += take
        return take

    def scatter(self, rows: torch.Tensor, positions: torch.Tensor, count: int,
                outputs: Mapping[str, torch.Tensor]) -> None:
        if count <= 0:
            return
        self._copy(rows, outputs, positions, count, to_lent=False)
        self.hit_rows += count

    def restore(self) -> None:
        """Put every written slot's original expert back and evict what cannot be, then sync the device.

        A slot whose map changed since ``begin`` is not reloaded: the bytes belong to whatever the map names now."""
        self.deactivate()
        started = time.perf_counter()
        changed = set(self.verify())
        failed: set[int] = set()
        by_layer: dict[int, list[int]] = {}
        for entry in range(self.written):
            if entry not in changed and self.original[entry] >= 0:
                by_layer.setdefault(self.plan[entry].layer_id, []).append(entry)
        for entries in by_layer.values():
            failed.update(self._reload_layer(entries))
        if changed or failed:
            logger.error("layer-major expert residency: evicting %d slots whose map changed and %d that did not reload",
                         len(changed), len(failed))
            self._evict(sorted(changed | failed))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reload_ms = (time.perf_counter() - started) * 1000
        self.restored = True

    def _reload_layer(self, entries: list[int]) -> list[int]:
        """Reload one layer's written entries; returns the entries that were not restored."""
        from sglang.srt.layers.moe.expert_stream import _staging_buffer

        cache = self.plan[entries[0]].cache
        streamer = cache.streamer
        tier = streamer.pinned_host_cache
        if tier is None or not tier.capacity:
            logger.error("layer-major expert residency: layer %d has no pinned tier to reload from",
                         self.plan[entries[0]].layer_id)
            return list(entries)
        floor = streamer._staging_floor_rows()
        for start in range(0, len(entries), floor):
            part = entries[start : start + floor]
            n = len(part)
            try:
                ids = torch.tensor([self.original[i] for i in part], dtype=torch.long,
                                   device=tier.expert_to_slot.device)
                staging = {
                    name: _staging_buffer(name, n, floor, streamer.spec(name).row_shape,
                                          streamer.spec(name).dtype, self.device)
                    for name in streamer.tensor_names
                }
                result = tier.gather_rows(ids, {k: v for k, v in staging.items() if k in tier.cached_names})
                uncached = {k: v for k, v in staging.items() if k not in tier.cached_names}
                if uncached:
                    streamer._copy_source_rows(ids, uncached)
                self._copy(torch.tensor(part, dtype=torch.long, device=self.device), staging,
                           torch.arange(n, dtype=torch.long, device=self.device), n, to_lent=True)
            except Exception:
                logger.exception("layer-major expert residency: reload of layer %d failed",
                                 self.plan[part[0]].layer_id)
                return list(entries[start:])
            self.reloaded_rows += n
            self.pinned_hit_rows += result.hit_rows
            self.pinned_miss_rows += result.miss_rows
            self.fallback_gathers += int(result.fallback_used)
        return []

    def _evict(self, indices: list[int]) -> None:
        """Logically evict plan entries' slots: mapping -1, slot FREE, as ``hot_cache_retire._evict_tail``."""
        from sglang.srt.layers.moe import hot_cache_retire

        manager = self.manager
        updater = getattr(manager, "gpu_residency", None)
        by_layer: dict[int, list[int]] = {}
        for i in indices:
            by_layer.setdefault(self.plan[i].layer_id, []).append(self.plan[i].slot)
        for layer_id, slots in by_layer.items():
            cache = manager.caches[layer_id]
            slot_ids = torch.tensor(slots, dtype=torch.long, device=self.device)
            if updater is not None and getattr(cache, "device_residency", None) is updater:
                row = updater.layer_ids.index(layer_id)
                experts = updater.slot_to_expert[row, slot_ids]
                safe = experts.clamp(min=0)
                owned = (experts >= 0) & (updater.mapping[row, safe] == slot_ids)
                updater.mapping[row, safe[owned]] = -1
                updater.slot_to_expert[row, slot_ids] = -1
                updater.slot_state[row, slot_ids] = _SLOT_FREE
                for slot in slots:
                    cache.slot_to_expert[slot] = -1
            else:
                from sglang.srt.layers.moe.expert_hot_cache import HotCacheSlotState

                with cache._slot_batch():
                    for slot in slots:
                        cache.slot_to_expert[slot] = -1
                        cache._set_slot_state(slot, HotCacheSlotState.FREE)
            self.evicted_rows += len(slots)
        if updater is not None:
            hot_cache_retire._refresh_service(
                manager, updater, hot_cache_retire._service(), {layer: manager.caches[layer].capacity
                                                                 for layer in by_layer}
            )
            if hasattr(updater, "_rank_victims"):
                updater._rank_victims(updater.route_counts > 0)

    def summary(self) -> str:
        return (
            f"{self.rows} slots lent, {self.written} written, {self.reloaded_rows} reloaded in {self.reload_ms:.0f} ms "
            f"(pinned tier {self.pinned_hit_rows} hits, {self.pinned_miss_rows} misses, "
            f"{self.fallback_gathers} fallback gathers), {self.evicted_rows} evicted"
        )


class Exl3LayerMajorResidency:
    """``layer_major.protocols.ExpertResidency`` over the hot-cached layers' streamers: one ``PrefillResidency``
    serves each layer of the pass in turn. A layer without a hot cache (its gather never asks) streams as before."""

    def __init__(self, model_runner, rows: int):
        self.model_runner = model_runner
        self.rows = rows
        self.buffer: Optional[PrefillResidency] = None

    def _streamer(self, layer_id: int):
        manager = getattr(self.model_runner, "expert_hot_cache_manager", None)
        cache = None if manager is None else manager.caches.get(layer_id)
        if cache is None or not cache.capacity:
            return None
        return cache.streamer

    def begin(self, layer_ids: range) -> None:
        if self.buffer is None:
            first = next((s for s in map(self._streamer, layer_ids) if s is not None), None)
            if first is None:
                return
            # Allocated at the first pass and kept: warm-up prompts never reach the layer-major threshold.
            self.buffer = PrefillResidency.for_streamer(first, min(self.rows, first.num_experts))
        self.buffer.take_pass_counts()

    def make_resident(self, layer_id: int) -> None:
        if self.buffer is None:
            return
        streamer = self._streamer(layer_id)
        # A mixed-K checkpoint's layers differ in row shape; one whose rows are not the buffer's streams without it.
        if streamer is None or not self.buffer.fits(streamer):
            self.buffer.deactivate()
        else:
            self.buffer.activate(streamer)

    def restore(self) -> None:
        if self.buffer is not None:
            self.buffer.deactivate()

    def end(self) -> None:
        if self.buffer is None:
            return
        self.buffer.deactivate()
        adopted, hits = self.buffer.take_pass_counts()
        logger.info("layer-major expert residency: %d rows adopted, %d resident hits", adopted, hits)


class Exl3LendingLayerMajorResidency(Exl3LayerMajorResidency):
    """``Exl3LayerMajorResidency`` over lent late-layer hot-cache slots (``LentPrefillResidency``), one per pass.

    The donors are the hot-cached layers after the pass's last layer, taken from the last layer down and, within a
    layer, in slot order."""

    def _plan(self, manager, layer_ids: range, first, rows: int) -> list[LentRow]:
        specs = _specs_of(first)
        plan: list[LentRow] = []
        last = layer_ids[-1]
        for layer_id in sorted((layer for layer in manager.caches if layer > last), reverse=True):
            cache = manager.caches[layer_id]
            if not cache.capacity or not _streamer_fits(cache.streamer, first.num_experts, specs):
                continue
            plan.extend(LentRow(layer_id, cache, slot) for slot in range(cache.capacity))
            if len(plan) >= rows:
                break
        return plan[:rows]

    def begin(self, layer_ids: range) -> None:
        self.buffer = None
        manager = getattr(self.model_runner, "expert_hot_cache_manager", None)
        if manager is None or not len(layer_ids):
            return
        first = next((s for s in map(self._streamer, layer_ids) if s is not None), None)
        if first is None:
            return
        manager.quiesce_for_prefill_lend()
        plan = self._plan(manager, layer_ids, first, min(self.rows, first.num_experts))
        if not plan:
            logger.warning("layer-major expert residency: no hot-cache slot to lend; the pass streams without it")
            return
        buffer = LentPrefillResidency(
            manager=manager, plan=plan, num_experts=first.num_experts, specs=_specs_of(first),
            device=first.hot_cache.device,
        )
        manager.set_prefill_lend(True)
        self.buffer = buffer

    def restore(self) -> None:
        buffer = self.buffer
        if buffer is None or buffer.restored:
            return
        buffer.restore()
        # Left active when restore raised: some slot may map an expert it no longer holds, and the fail-stop check
        # then stops the process at the end of the forward.
        buffer.manager.set_prefill_lend(False)

    def end(self) -> None:
        buffer = self.buffer
        if buffer is None:
            return
        buffer.deactivate()
        adopted, hits = buffer.take_pass_counts()
        logger.info("layer-major expert residency (lent): %d rows adopted, %d resident hits; %s", adopted, hits,
                    buffer.summary())
        self.buffer = None
