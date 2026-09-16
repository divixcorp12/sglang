"""Bounded, nonblocking hand-off of device telemetry to CPU writers.

The serving thread is allowed to enqueue a stream-ordered copy and query an
event.  It never waits for that copy or serializes a metric record.  A slot
keeps its pinned CPU buffers exclusively from copy submission through the
background writer, so a later GPU copy cannot mutate data being formatted.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


class AsyncTelemetryBackend(Protocol):
    """Device-specific ownership and event operations used by ``AsyncTelemetry``.

    ``enqueue`` must copy every source into its matching owned buffer and then
    record ``event`` after the copies on the source stream.  It must return
    without waiting.  Test fakes use this same small protocol without CUDA.
    """

    def allocate(self, sources: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def event(self) -> Any: ...

    def enqueue(
        self, buffers: Mapping[str, Any], sources: Mapping[str, Any], event: Any
    ) -> None: ...

    def ready(self, event: Any) -> bool: ...

    def synchronize(self, event: Any) -> None: ...


class _CpuEvent:
    """CPU snapshots are immediately complete but still use the same lifecycle."""


class TorchTelemetryBackend:
    """Pinned-buffer CUDA backend, with a CPU implementation for unit-only users."""

    def __init__(self, sources: Mapping[str, torch.Tensor]) -> None:
        devices = {source.device for source in sources.values()}
        if len(devices) != 1:
            raise ValueError("a telemetry snapshot must use one source device")
        self._device = next(iter(devices))

    def allocate(self, sources: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return {
            name: torch.empty_like(source, device="cpu", pin_memory=source.is_cuda)
            for name, source in sources.items()
        }

    def event(self) -> torch.cuda.Event | _CpuEvent:
        if self._device.type != "cuda":
            return _CpuEvent()
        return torch.cuda.Event(enable_timing=False)

    def enqueue(
        self,
        buffers: Mapping[str, torch.Tensor],
        sources: Mapping[str, torch.Tensor],
        event: torch.cuda.Event | _CpuEvent,
    ) -> None:
        for name, source in sources.items():
            buffers[name].copy_(source, non_blocking=source.is_cuda)
        if isinstance(event, _CpuEvent):
            return
        event.record(torch.cuda.current_stream(self._device))

    def ready(self, event: torch.cuda.Event | _CpuEvent) -> bool:
        return isinstance(event, _CpuEvent) or event.query()

    def synchronize(self, event: torch.cuda.Event | _CpuEvent) -> None:
        if not isinstance(event, _CpuEvent):
            event.synchronize()


@dataclass
class _Slot:
    buffers: Mapping[str, Any]
    event: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AsyncTelemetry:
    """A fixed-size asynchronous snapshot pool plus one bounded CPU writer.

    ``schedule`` and ``poll`` are intended for the inference thread.  They use
    only nonblocking backend operations and ``put_nowait``.  A full copy pool
    or writer queue drops the optional sample instead of delaying inference.
    ``close`` is deliberately the sole operation allowed to wait: callers use
    it during non-serving teardown to flush accepted samples.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, Any],
        backend: AsyncTelemetryBackend,
        writer: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
        slots: int = 2,
        writer_jobs: int = 2,
        thread_name: str = "moe-telemetry-writer",
    ) -> None:
        if slots < 1:
            raise ValueError("telemetry needs at least one snapshot slot")
        if writer_jobs < 1:
            raise ValueError("telemetry writer queue must have positive capacity")
        self._backend = backend
        self._writer = writer
        self._lock = threading.Lock()
        self._free = [_Slot(backend.allocate(sources), backend.event()) for _ in range(slots)]
        self._pending: list[_Slot] = []
        self._jobs: queue.Queue[_Slot | None] = queue.Queue(maxsize=writer_jobs)
        self._closed = False
        self._close_prepared = False
        self._worker_stopped = False
        self._shutdown_lock = threading.Lock()
        self._stats = {
            "accepted": 0,
            "completed": 0,
            "dropped_pending": 0,
            "dropped_writer": 0,
            "writer_failures": 0,
        }
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self._thread.start()

    def schedule(self, sources: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
        """Copy one exact stream-ordered sample, or drop it without waiting."""
        self.poll()
        with self._lock:
            if self._closed or not self._free:
                self._stats["dropped_pending"] += 1
                return False
            slot = self._free.pop()
            # A shallow, read-only metadata copy prevents a caller from changing
            # forward/phase attribution after this accepted snapshot.
            slot.metadata = MappingProxyType(dict(metadata))
            self._pending.append(slot)
            self._stats["accepted"] += 1
            try:
                # Keep the slot pending until the nonblocking submission has
                # returned.  Teardown may synchronize and hand pending slots
                # to the writer immediately after this lock is released, so it
                # must never observe a slot whose copy has not been submitted.
                self._backend.enqueue(slot.buffers, sources, slot.event)
            except Exception:
                self._pending.remove(slot)
                self._free.append(slot)
                self._stats["accepted"] -= 1
                self._stats["dropped_pending"] += 1
                logger.exception("Could not queue optional telemetry snapshot")
                return False
        return True

    def poll(self) -> None:
        """Move completed copies to the writer without waiting for either side."""
        with self._lock:
            if self._closed:
                return
            for slot in tuple(self._pending):
                if not self._backend.ready(slot.event):
                    continue
                self._pending.remove(slot)
                try:
                    self._jobs.put_nowait(slot)
                except queue.Full:
                    self._release(slot)
                    self._stats["dropped_writer"] += 1

    def stats(self) -> dict[str, int]:
        """Return host-maintained observability counters without touching a device."""
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        """Flush accepted samples; call only after serving has stopped."""
        with self._shutdown_lock:
            self._prepare_close()
            self._finish_close()

    def prepare_close(self) -> None:
        """Synchronize and enqueue pending work, but keep the writer alive.

        Owners with several telemetry pools call this on all pools before
        ``finish_close``.  That avoids a writer for a later ordered record
        waiting on an event whose pool has not yet been flushed.
        """
        with self._shutdown_lock:
            self._prepare_close()

    def finish_close(self) -> None:
        """Wait for a prepared writer at teardown and terminate its thread."""
        with self._shutdown_lock:
            self._prepare_close()
            self._finish_close()

    def _prepare_close(self) -> None:
        with self._lock:
            if self._close_prepared:
                return
            self._closed = True
            pending = tuple(self._pending)
            self._pending.clear()
        # This is the explicitly non-serving teardown path.  Every queued copy
        # is made visible before handing the slot to the worker.
        for slot in pending:
            self._backend.synchronize(slot.event)
            self._jobs.put(slot)

        with self._lock:
            self._close_prepared = True

    def _finish_close(self) -> None:
        if self._worker_stopped:
            return
        self._jobs.join()
        self._jobs.put(None)
        self._thread.join()
        self._worker_stopped = True

    def _run(self) -> None:
        while True:
            slot = self._jobs.get()
            if slot is None:
                self._jobs.task_done()
                return
            try:
                self._writer(slot.buffers, slot.metadata)
            except Exception:
                with self._lock:
                    self._stats["writer_failures"] += 1
                logger.exception("Optional telemetry writer failed")
            finally:
                with self._lock:
                    self._stats["completed"] += 1
                    self._release(slot)
                self._jobs.task_done()

    def _release(self, slot: _Slot) -> None:
        slot.metadata = MappingProxyType({})
        self._free.append(slot)
