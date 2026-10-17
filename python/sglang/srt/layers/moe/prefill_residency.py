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
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import torch
import triton

logger = logging.getLogger(__name__)


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
        specs = {name: (streamer.spec(name).row_shape, streamer.spec(name).dtype) for name in streamer.tensor_names}
        return cls(rows=rows, num_experts=streamer.num_experts, specs=specs, device=streamer.hot_cache.device)

    def fits(self, streamer) -> bool:
        """Whether ``streamer``'s rows have this buffer's names, shapes and dtypes."""
        if streamer.num_experts != self.num_experts or set(streamer.tensor_names) != set(self.specs):
            return False
        return all(
            (tuple(streamer.spec(name).row_shape), streamer.spec(name).dtype) == (tuple(shape), dtype)
            for name, (shape, dtype) in self.specs.items()
        )

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
        if streamer is None:
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
