"""Doorbell expert-row copies: GPU-posted plans copied by a CPU spin thread.

Inside a CUDA graph, ``ExpertDoorbellCopier.post`` launches a kernel that
writes a copy plan into a pinned request page and bumps its sequence last. A
C++ thread spinning on that sequence copies the planned pinned rows to their
device slots with one batched copy on its own CUDA stream, then queues a
four-byte publish of the sequence into a device completion word.
``ExpertDoorbellCopier.wait`` launches a chain of short kernels that block
until that word reaches the request's sequence, or after ``timeout_polls``
polls in total fall back to the in-graph copy of the same plan. Neither call
synchronizes the host or breaks the graph.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.expert_cache_transfer import (
    ExpertRowSegments,
    _validate_plan,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_PAGE_HEADER_BYTES = 16
_RECORD_HEADER_BYTES = 16
_TAG_BASE = 8
_PUBLISH_RING = 1024
_TRACE_ROWS = 4096
_POLL_MODES = {"acquire": 0, "volatile": 1, "noncoherent": 2}
_STATE_WORDS = {
    "posted": 0,
    "fallback_count": 1,
    "degraded": 2,
    "timeouts": 3,
    "waits": 4,
    "last_polls": 5,
    "record_mismatches": 6,
    "resolved": 7,
}
_COUNTERS = (
    "serviced",
    "skipped_abandoned",
    "skipped_overrun",
    "invalid_records",
    "copy_errors",
    "rows_copied",
    "bytes_copied",
    "trace_count",
    "running",
    "spin_cpu",
    "last_seen",
)
_STATUSES = (
    "pending",
    "serviced",
    "skipped_abandoned",
    "skipped_overrun",
    "invalid_record",
    "copy_failed",
)
_TRACE_FIELDS = (
    "seq",
    "count",
    "seen_ns",
    "enqueued_ns",
    "complete_ns",
    "bytes",
    "status",
    "publish_slot",
)


@functools.cache
def _jit_expert_doorbell_module() -> Module:
    names = (
        "expert_doorbell_post",
        "expert_doorbell_wait",
        "expert_doorbell_start",
        "expert_doorbell_stop",
        "expert_doorbell_pause",
        "expert_doorbell_counters",
        "expert_doorbell_trace",
    )
    return load_jit(
        "expert_doorbell",
        cuda_files=["moe/expert_doorbell.cuh"],
        cuda_wrappers=[(name, name) for name in names],
    )


def _record_bytes(capacity: int) -> int:
    return (_RECORD_HEADER_BYTES + 12 * capacity + 7) // 8 * 8


class ExpertDoorbellCopier:
    """Copies planned expert rows through a CPU thread without a host sync.

    ``post`` and ``wait`` only launch kernels, so both can be captured in a
    CUDA graph and replayed with plans and counts that change between replays.
    Each ``tag`` names one outstanding request, e.g. the MoE layer the rows are
    for; ``wait(tag)`` waits for the latest ``post`` with that tag.

    Measured limit (RTX 5090, driver 610.57, torch 2.13): copies the thread
    queues while a CUDA graph launch runs do not execute until that launch's
    GPU work ends, so inside a graph the thread cannot overlap the graph's
    compute and a wait for a request posted in a graph falls back.

    Invariants the caller must uphold:

    * A destination slot named by a posted request must not be read by any
      kernel launched before that request's ``wait`` returns, and must not be
      written by anyone else until then. The thread writes slots on its own
      stream at any time between ``post`` and completion.
    * After a timeout the thread may still be finishing copies it already
      queued for that request. They write the same source rows into the same
      slots the fallback wrote, so a slot of a timed-out request must not be
      reassigned to a different row until the thread has caught up
      (``stats()["last_seen"]`` at or past that request and nothing pending).
    * Source rows must stay allocated, registered and unchanged while any
      request naming them is outstanding.
    * At most ``ring - 1`` requests may be posted between a ``post`` and its
      ``wait``; older records are overwritten and a timed-out wait on an
      overwritten record copies nothing (counted in ``record_mismatches``).
    """

    def __init__(
        self,
        segments: ExpertRowSegments,
        capacity: int,
        *,
        ring: int = 64,
        max_tags: int = 64,
        cpu_core: int = 71,
        timeout_polls: int = 2_000_000,
        degraded_polls: int = 4_096,
        prefer_overlap: bool = True,
        poll_mode: str = "acquire",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if ring < 2:
            raise ValueError("ring must hold at least two requests.")
        if max_tags <= 0:
            raise ValueError("max_tags must be positive.")
        if timeout_polls < 0 or degraded_polls < 0:
            raise ValueError("poll limits must not be negative.")
        if poll_mode not in _POLL_MODES:
            raise ValueError(f"poll_mode must be one of {sorted(_POLL_MODES)}.")
        device = segments.table.device
        self.segments = segments
        self.capacity = capacity
        self.ring = ring
        self.max_tags = max_tags
        self.timeout_polls = timeout_polls
        self.degraded_polls = degraded_polls
        self.poll_mode = _POLL_MODES[poll_mode]
        self.device = device
        self.page = torch.zeros(
            _PAGE_HEADER_BYTES + ring * _record_bytes(capacity),
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.state = torch.zeros(_TAG_BASE + max_tags, dtype=torch.int32, device=device)
        self.done = torch.zeros(1, dtype=torch.int32, device=device)
        self.fallback_rows = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.fallback_slots = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.publish_words = torch.zeros(
            _PUBLISH_RING, dtype=torch.int32, pin_memory=True
        )
        self.segment_table = torch.tensor(
            [
                [
                    source.data_ptr(),
                    destination.data_ptr(),
                    source.numel() // source.shape[0] * source.element_size(),
                    source.shape[0],
                    destination.shape[0],
                ]
                for source, destination in segments.pairs
            ],
            dtype=torch.int64,
        )
        self._module = _jit_expert_doorbell_module()
        self._handle = self._module.expert_doorbell_start(
            self.page,
            self.segment_table,
            self.done,
            self.publish_words,
            capacity,
            ring,
            device.index if device.index is not None else torch.cuda.current_device(),
            cpu_core,
            int(prefer_overlap),
        )
        if self._handle < 0:
            raise RuntimeError("expert doorbell thread failed to start.")
        self._stopped = False

    def _check_tag(self, tag: int) -> None:
        if not 0 <= tag < self.max_tags:
            raise ValueError(f"tag must be in [0, {self.max_tags}).")

    def post(
        self,
        source_rows: torch.Tensor,
        destination_slots: torch.Tensor,
        count: torch.Tensor,
        tag: int = 0,
    ) -> None:
        """Launch the kernel that posts ``count`` planned rows for ``tag``."""
        if self._stopped:
            raise RuntimeError("expert doorbell thread is stopped.")
        self._check_tag(tag)
        _validate_plan(self.device, source_rows, destination_slots, count)
        if source_rows.numel() != self.capacity:
            raise ValueError("plan tensors must match the copier capacity.")
        self._module.expert_doorbell_post(
            self.page,
            self.state,
            source_rows,
            destination_slots,
            count,
            tag,
            self.capacity,
            self.ring,
        )

    def wait(self, tag: int = 0) -> None:
        """Launch the kernels that wait for ``tag``'s request or copy it in-graph."""
        self._check_tag(tag)
        self._module.expert_doorbell_wait(
            self.page,
            self.state,
            self.done,
            self.fallback_rows,
            self.fallback_slots,
            self.segments.table,
            tag,
            self.capacity,
            self.ring,
            self.timeout_polls,
            self.degraded_polls,
            self.poll_mode,
        )

    def pause(self) -> None:
        """Keep the thread spinning but stop it servicing requests."""
        self._module.expert_doorbell_pause(self._handle, 1)

    def resume(self) -> None:
        self._module.expert_doorbell_pause(self._handle, 0)

    def stop(self) -> None:
        """Drain every request posted so far, then join the thread."""
        if self._stopped:
            return
        torch.cuda.synchronize(self.device)
        self._module.expert_doorbell_stop(self._handle)
        self._stopped = True

    def stats(self) -> dict[str, int]:
        """Thread counters plus the device state words (synchronizes the device)."""
        counters = torch.zeros(len(_COUNTERS), dtype=torch.int64)
        found = self._module.expert_doorbell_counters(self._handle, counters)
        stats = dict(zip(_COUNTERS, counters.tolist())) if found else {"running": 0}
        state = self.state.cpu().tolist()
        stats.update({name: state[index] for name, index in _STATE_WORDS.items()})
        stats["done"] = int(self.done.item())
        return stats

    def trace(self) -> list[dict[str, int | str]]:
        """The thread's most recent requests, oldest first."""
        rows = torch.zeros((_TRACE_ROWS, len(_TRACE_FIELDS)), dtype=torch.int64)
        written = self._module.expert_doorbell_trace(self._handle, rows)
        entries = []
        for values in rows[:written].tolist():
            entry: dict[str, int | str] = dict(zip(_TRACE_FIELDS, values))
            entry["status"] = _STATUSES[values[_TRACE_FIELDS.index("status")]]
            entries.append(entry)
        return entries

    def __enter__(self) -> ExpertDoorbellCopier:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def __del__(self) -> None:
        if getattr(self, "_stopped", True) is False:
            self._module.expert_doorbell_stop(self._handle)
            self._stopped = True
