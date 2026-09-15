"""Doorbell expert-row copies: GPU-posted plans copied by a CPU spin thread.

Inside a CUDA graph, ``ExpertDoorbellCopier.post`` launches a kernel that
writes a copy plan into a pinned request page and bumps its sequence last. A
C++ thread spinning on that sequence copies the planned pinned rows to their
device slots with one batched copy on a CUDA stream, then queues an eight-byte
publish of ``{sequence, sequence if copied else 0}`` into device completion
words.
``ExpertDoorbellCopier.resolve`` launches a chain of short kernels that block
until those words reach the tag's request and set the tag's device delivered
flag, or give up after ``timeout_polls`` and report nothing delivered; when
the thread had already committed to the request they drain until its copies
land. ``wait`` is ``resolve`` followed by the in-graph copy of whatever was
not delivered. None of these calls synchronizes the host or breaks the graph.
"""

from __future__ import annotations

import atexit
import functools
import time
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.expert_cache_transfer import (
    ExpertRowSegments,
    _validate_plan,
    copy_expert_row_segments_gpu,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ABANDONED_BASE = 16
_RECORD_HEADER_BYTES = 16
_TAG_BASE = 13
_POLL_SECONDS = 250e-9
_FATAL_BACKSTOP = 4
_PUBLISH_RING = 1024
_TRACE_ROWS = 4096
_POLL_MODES = {"acquire": 0, "volatile": 1, "noncoherent": 2}
_HEAD_STORES = {"release": 0, "volatile": 1}
_COPY_APIS = {"batch": 0, "per_segment": 1}
_SRC_ACCESS_ORDERS = {"stream": 1, "during_call": 2, "any": 3}
_STATE_WORDS = {
    "posted": 0,
    "disabled": 1,
    "degraded": 2,
    "timeouts": 3,
    "waits": 4,
    "last_polls": 5,
    "record_mismatches": 6,
    "resolved": 7,
    "drain_timeouts": 8,
    "drains": 9,
    "drain_pending": 10,
    "disabled_posts": 11,
    "fatal_timeouts": 12,
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
    "last_copy_error",
    "copy_api",
    "src_access_order",
    "external_stream",
    "late_completions",
    "discarded_disabled",
)
_RESET_AFTER_CAPTURE = (
    "degraded",
    "timeouts",
    "waits",
    "last_polls",
    "record_mismatches",
    "drain_timeouts",
    "drains",
    "drain_pending",
)
_STATUSES = (
    "pending",
    "serviced",
    "skipped_abandoned",
    "skipped_overrun",
    "invalid_record",
    "copy_failed",
    "discarded_disabled",
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
        "expert_doorbell_resolve",
        "expert_doorbell_start",
        "expert_doorbell_stop",
        "expert_doorbell_pause",
        "expert_doorbell_inject",
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


def _header_bytes(max_tags: int) -> int:
    return (_ABANDONED_BASE + 4 * max_tags + 7) // 8 * 8


def _capturing() -> bool:
    return torch.cuda.graphs.is_current_stream_capturing()


_LIVE_COPIERS: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_live_copiers() -> None:
    """Stop every copier still running at interpreter exit, while CUDA is still up.

    Otherwise the process-wide thread registry is destroyed during static
    teardown with its threads still running.
    """
    for copier in list(_LIVE_COPIERS):
        copier.stop()


class ExpertDoorbellCopier:
    """Copies planned expert rows through a CPU thread without a host sync.

    ``post``, ``resolve`` and ``wait`` only launch kernels, so they can be
    captured in a CUDA graph and replayed with plans and counts that change
    between replays. Each ``tag`` names one outstanding request, e.g. the
    target MoE layer of its rows; ``resolve(tag)`` resolves the latest ``post``
    with that tag, independently of requests outstanding for other tags.

    ``head_store`` selects how the poster publishes the sequence to the host:
    ``"release"`` (a system-scope release store) or ``"volatile"`` (a volatile
    global store).

    ``copy_api`` selects how the thread issues row copies: ``"batch"`` (one
    ``cudaMemcpyBatchAsync`` per request, every copy with ``src_access_order``
    ``"stream"``, ``"during_call"`` or ``"any"``) or ``"per_segment"`` (one
    ``cudaMemcpyAsync`` per segment row, stream order only). ``stream`` is the
    ``torch.cuda.Stream`` the thread copies on; the copier keeps a reference to
    it. Without one the thread creates its own stream, whose copies CUDA-graph
    replays hold back (E32), so ``post`` and ``resolve`` refuse to be captured
    in a graph then; that mode exists for probes. ``stats()`` reports the
    configured values and ``last_copy_error``, the CUDA status of the latest
    failed copy call.

    ``segments`` is one ``ExpertRowSegments`` copied by every request, or a
    sequence of them, one per tag: a request copies its tag's set and ``wait``
    copies its residual through that set, so one thread serves layers whose
    rows live in different tensors.

    Invariants the caller must uphold:

    * A destination slot named by a posted request must not be read by any
      kernel launched before that request's ``resolve`` returns, and must not
      be written by anyone else until then. The thread writes slots on its
      own stream at any time between ``post`` and completion.
    * The plan tensors passed to ``post`` must not change until that tag's
      ``wait`` launched, since ``wait`` copies the undelivered plan from them.
    * A resolve that times out on a request the thread has not committed to
      reports it undelivered at once: the thread commits (writes its claim)
      before it checks whether the request was abandoned, and copies only
      after that check. A committed request is drained until its copies land
      (``drains``). A drain that runs out of ``drain_polls`` disables the
      copier for good (``disabled``, ``drain_timeouts``) and keeps waiting
      for that committed request's copies, so no copy ever lands after its
      resolve returned; once disabled every later ``post`` posts nothing
      (``disabled_posts``) and the thread discards what it had not committed
      to (``discarded_disabled``). That wait is fail-stop: if the copies have
      not landed ``fatal_wait_s`` later, a watchdog thread reports an ERROR on
      stderr and aborts the process. A kernel-side bound of four times that
      wait backstops a watchdog that could not act; reaching it is counted
      (``fatal_timeouts``) and resolves undelivered.
    * Source rows must stay allocated, registered and unchanged while any
      request naming them is outstanding.
    * At most ``ring - 1`` requests may be posted between a ``post`` and its
      ``resolve``; an older record is overwritten and its request reports
      undelivered (counted in ``record_mismatches``).
    """

    def __init__(
        self,
        segments: ExpertRowSegments | Sequence[ExpertRowSegments],
        capacity: int,
        *,
        ring: int = 64,
        max_tags: int = 64,
        cpu_core: int = 71,
        timeout_polls: int = 2_000_000,
        degraded_polls: int = 4_096,
        drain_polls: int = 8_000_000,
        fatal_wait_s: float = 30.0,
        prefer_overlap: bool = True,
        poll_mode: str = "acquire",
        head_store: str = "release",
        copy_api: str = "batch",
        src_access_order: str = "stream",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if ring < 2:
            raise ValueError("ring must hold at least two requests.")
        if max_tags <= 0:
            raise ValueError("max_tags must be positive.")
        if timeout_polls < 0 or degraded_polls < 0 or drain_polls < 0:
            raise ValueError("poll limits must not be negative.")
        if not fatal_wait_s > 0:
            raise ValueError("fatal_wait_s must be positive.")
        if poll_mode not in _POLL_MODES:
            raise ValueError(f"poll_mode must be one of {sorted(_POLL_MODES)}.")
        if head_store not in _HEAD_STORES:
            raise ValueError(f"head_store must be one of {sorted(_HEAD_STORES)}.")
        if copy_api not in _COPY_APIS:
            raise ValueError(f"copy_api must be one of {sorted(_COPY_APIS)}.")
        if src_access_order not in _SRC_ACCESS_ORDERS:
            raise ValueError(
                f"src_access_order must be one of {sorted(_SRC_ACCESS_ORDERS)}."
            )
        if copy_api == "per_segment" and src_access_order != "stream":
            raise ValueError("src_access_order applies only to copy_api='batch'.")
        segment_sets = (
            (segments,) if isinstance(segments, ExpertRowSegments) else tuple(segments)
        )
        if not segment_sets:
            raise ValueError("segments must hold at least one segment set.")
        if len(segment_sets) > 1 and max_tags < len(segment_sets):
            raise ValueError("max_tags must cover every segment set.")
        device = segment_sets[0].table.device
        if any(segment_set.table.device != device for segment_set in segment_sets):
            raise ValueError("every segment set must share one CUDA device.")
        if stream is not None and stream.device != device:
            raise ValueError("stream must be on the copier's device.")
        self.stream = stream
        self.segment_sets = segment_sets
        self.segments = segment_sets[0]
        self.capacity = capacity
        self.ring = ring
        self.max_tags = max_tags
        self.timeout_polls = timeout_polls
        self.degraded_polls = degraded_polls
        self.drain_polls = drain_polls
        self.fatal_wait_s = fatal_wait_s
        self.fatal_polls = int(_FATAL_BACKSTOP * fatal_wait_s / _POLL_SECONDS) + 1
        self.poll_mode = _POLL_MODES[poll_mode]
        self.head_store = _HEAD_STORES[head_store]
        self.device = device
        self.header_bytes = _header_bytes(max_tags)
        self.page = torch.zeros(
            self.header_bytes + ring * _record_bytes(capacity),
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.state = torch.zeros(_TAG_BASE + max_tags, dtype=torch.int32, device=device)
        self.delivered = torch.zeros(max_tags, dtype=torch.int32, device=device)
        self.done = torch.zeros(2, dtype=torch.int32, device=device)
        self._undelivered = torch.zeros((max_tags, 1), dtype=torch.int32, device=device)
        self._one = torch.ones(1, dtype=torch.int32, device=device)
        self._posted_plans: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.publish_words = torch.zeros(
            2 * _PUBLISH_RING, dtype=torch.int32, pin_memory=True
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
                for segment_set in segment_sets
                for source, destination in segment_set.pairs
            ],
            dtype=torch.int64,
        )
        set_sizes = [len(segment_set.pairs) for segment_set in segment_sets]
        self.set_offsets = torch.tensor(
            [sum(set_sizes[:index]) for index in range(len(set_sizes) + 1)],
            dtype=torch.int64,
        )
        self._module = _jit_expert_doorbell_module()
        self._handle = self._module.expert_doorbell_start(
            self.page,
            self.segment_table,
            self.set_offsets,
            self.done,
            self.publish_words,
            capacity,
            ring,
            self.header_bytes,
            max_tags,
            device.index if device.index is not None else torch.cuda.current_device(),
            cpu_core,
            int(prefer_overlap),
            _COPY_APIS[copy_api],
            _SRC_ACCESS_ORDERS[src_access_order],
            stream.cuda_stream if stream is not None else 0,
            int(fatal_wait_s * 1e9),
        )
        if self._handle < 0:
            raise RuntimeError("expert doorbell thread failed to start.")
        self._stopped = False
        _LIVE_COPIERS.add(self)

    def _check_tag(self, tag: int) -> None:
        if not 0 <= tag < self.max_tags:
            raise ValueError(f"tag must be in [0, {self.max_tags}).")

    def _check_capture(self) -> None:
        if self.stream is None and _capturing():
            raise RuntimeError(
                "expert doorbell copies on a thread-created stream are held behind "
                "CUDA-graph replays; pass stream=torch.cuda.Stream() to capture them."
            )

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
        self._check_capture()
        _validate_plan(self.device, source_rows, destination_slots, count)
        if source_rows.numel() != self.capacity:
            raise ValueError("plan tensors must match the copier capacity.")
        self._posted_plans[tag] = (source_rows, destination_slots, count)
        self._module.expert_doorbell_post(
            self.page,
            self.state,
            self.delivered,
            source_rows,
            destination_slots,
            count,
            tag,
            self.capacity,
            self.ring,
            self.header_bytes,
            self.head_store,
        )

    def resolve(self, tag: int = 0) -> torch.Tensor:
        """Launch the kernels that resolve ``tag``'s request.

        Returns the tag's int32 delivered flag, a one-element device view that
        holds 1 once the resolve launched if the thread copied the whole plan,
        and 0 if it copied none of it.
        """
        self._check_tag(tag)
        self._check_capture()
        self._module.expert_doorbell_resolve(
            self.page,
            self.state,
            self.delivered,
            self.done,
            tag,
            self.capacity,
            self.ring,
            self.header_bytes,
            self.timeout_polls,
            self.degraded_polls,
            self.drain_polls,
            self.fatal_polls,
            self.poll_mode,
        )
        return self.delivered[tag : tag + 1]

    def undelivered_count(self, tag: int, count: torch.Tensor) -> torch.Tensor:
        """The planned row count still to copy after ``resolve(tag)``: ``count`` or 0."""
        buffer = self._undelivered[tag]
        torch.sub(self._one, self.delivered[tag : tag + 1], out=buffer)
        torch.mul(buffer, count, out=buffer)
        return buffer

    def wait(self, tag: int = 0) -> None:
        """Resolve ``tag``'s request, then copy in-graph whatever was not delivered."""
        self._check_tag(tag)
        if tag not in self._posted_plans:
            raise RuntimeError(f"no request was posted for tag {tag}.")
        source_rows, destination_slots, count = self._posted_plans[tag]
        self.resolve(tag)
        copy_expert_row_segments_gpu(
            self.segment_sets[tag if len(self.segment_sets) > 1 else 0],
            source_rows,
            destination_slots,
            self.undelivered_count(tag, count),
        )

    def pause(self) -> None:
        """Keep the thread spinning but stop it servicing requests."""
        self._module.expert_doorbell_pause(self._handle, 1)

    def resume(self) -> None:
        self._module.expert_doorbell_pause(self._handle, 0)

    def inject_fault(self, service_delay_s: float = 0.0, fail_copies: bool = False) -> None:
        """Make the thread sleep ``service_delay_s`` after committing to each request and
        before queuing its copies, or report every copy as failed without issuing it; the
        defaults clear both."""
        self._module.expert_doorbell_inject(
            self._handle, int(service_delay_s * 1e9), int(fail_copies)
        )

    def reset_wait_state(self) -> None:
        """Zero the resolve counters and the degraded flag, keeping sequences and ``disabled``.

        Resolves captured while the thread is quiesced time out and leave the
        copier degraded; call this before resuming so serving starts with full
        budgets and counters that only count serving resolves.
        """
        torch.cuda.synchronize(self.device)
        for name in _RESET_AFTER_CAPTURE:
            self.state[_STATE_WORDS[name]] = 0

    def quiesce(self, timeout_s: float = 10.0) -> None:
        """Pause the thread once every request posted so far is published.

        Synchronizes the device. Use it before capturing a CUDA graph so the
        thread issues no copies while the capture runs, and ``resume`` after.
        """
        torch.cuda.synchronize(self.device)
        posted = int(self.state[_STATE_WORDS["posted"]].item()) & 0xFFFFFFFF
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            trace = self.trace()
            last = trace[-1] if trace else None
            caught_up = posted == 0 or (
                last is not None and last["seq"] == posted and last["complete_ns"] != 0
            )
            if caught_up:
                self.pause()
                return
        raise RuntimeError("expert doorbell thread did not catch up before the quiesce timeout.")

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
        stats["done"] = int(self.done[0].item())
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
