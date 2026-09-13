"""Fixed-plan coordination for asynchronous NVFP4 expert transfers.

The executor is deliberately a control-plane primitive: callers reserve cache
slots, fill a ``FixedRowTransferPlan``, and supply the six existing row-copy
operations.  It serializes those operations on one device stream and provides
one completion event for their atomic publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from threading import Lock
from typing import Callable, Sequence

import torch
from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu
from sglang.srt.layers.moe.expert_dma import ExpertDMABackend


NVFP4_TRANSFER_TENSOR_COUNT = 6
_COPY_BACKENDS = frozenset(("gpu", "dma"))


class FixedRowTransferPlan:
    """Stable device buffers describing one bounded expert-row transfer.

    The six tensors are allocated once and retain their addresses across
    requests.  A plan cannot be rewritten until its submitted transfer has
    completed, preventing a later request from changing rows observed by an
    already enqueued copy operation.
    """

    def __init__(self, max_rows: int, device: torch.device | str):
        self.max_rows = index(max_rows)
        if self.max_rows < 1:
            raise ValueError("transfer plan max_rows must be positive")
        self.device = _canonical_device(device)
        self.source_rows = torch.zeros(
            self.max_rows, dtype=torch.int64, device=self.device
        )
        self.secondary_source_rows = torch.zeros(
            self.max_rows, dtype=torch.int64, device=self.device
        )
        self.destination_slots = torch.zeros(
            self.max_rows, dtype=torch.int32, device=self.device
        )
        self.destination_slots_long = torch.zeros(
            self.max_rows, dtype=torch.int64, device=self.device
        )
        self.generations = torch.zeros(
            self.max_rows, dtype=torch.int64, device=self.device
        )
        self.count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._row_count = 0
        self._active_sequence: int | None = None

    @property
    def row_count(self) -> int:
        """Return the CPU-known active prefix length without reading CUDA state."""
        return self._row_count

    @property
    def src(self) -> torch.Tensor:
        """Compatibility alias for source row IDs."""
        return self.source_rows

    @property
    def dst(self) -> torch.Tensor:
        """Compatibility alias for destination slot IDs."""
        return self.destination_slots

    def data_ptrs(self) -> tuple[int, ...]:
        """Expose allocations for graph-safety checks without reallocating."""
        return (
            self.source_rows.data_ptr(),
            self.secondary_source_rows.data_ptr(),
            self.destination_slots.data_ptr(),
            self.generations.data_ptr(),
            self.destination_slots_long.data_ptr(),
            self.count.data_ptr(),
        )

    def set_rows(
        self,
        source_rows: torch.Tensor | Sequence[int],
        destination_slots: torch.Tensor | Sequence[int],
        generations: torch.Tensor | Sequence[int],
    ) -> None:
        """Replace the active prefix while preserving all plan allocations."""
        if self._active_sequence is not None:
            raise RuntimeError("cannot rewrite a transfer plan while it is in flight")
        source = self._as_vector(source_rows, "source_rows")
        destination = self._as_vector(destination_slots, "destination_slots")
        generation = self._as_vector(generations, "generations")
        row_count = source.numel()
        if destination.numel() != row_count or generation.numel() != row_count:
            raise ValueError("transfer plan rows must have the same number of entries")
        if row_count > self.max_rows:
            raise ValueError("transfer plan row count exceeds fixed capacity")
        self.source_rows.zero_()
        self.destination_slots.zero_()
        self.generations.zero_()
        self.destination_slots_long.zero_()
        if row_count:
            self.source_rows[:row_count].copy_(
                source.to(device=self.device, dtype=torch.int64)
            )
            self.destination_slots[:row_count].copy_(
                destination.to(device=self.device, dtype=torch.int32)
            )
            self.destination_slots_long[:row_count].copy_(
                destination.to(device=self.device, dtype=torch.int64)
            )
            self.generations[:row_count].copy_(
                generation.to(device=self.device, dtype=torch.int64)
            )
        self.count.fill_(row_count)
        self._row_count = row_count

    @staticmethod
    def _as_vector(values: torch.Tensor | Sequence[int], name: str) -> torch.Tensor:
        tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if tensor.ndim != 1:
            raise ValueError(f"transfer plan {name} must be one-dimensional")
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"transfer plan {name} must contain integer values")
        return tensor

    def _claim(self, sequence: int) -> None:
        if self._active_sequence is not None:
            raise RuntimeError("transfer plan is already submitted")
        self._active_sequence = sequence

    def _release(self, sequence: int) -> None:
        if self._active_sequence == sequence:
            self._active_sequence = None


@dataclass(frozen=True)
class ExpertTransferTicket:
    """Identity of one atomic six-tensor transfer in an executor ring."""

    device: torch.device
    slot: int
    sequence: int
    plan: FixedRowTransferPlan


@dataclass(frozen=True)
class ExpertCopySubmission:
    """One atomic six-tensor submission and its requested/actual path."""

    ticket: ExpertTransferTicket
    requested_backend: str
    actual_backend: str
    rows: int
    bytes: int
    submissions: int
    fallbacks: int


class AsyncExpertTransferExecutor:
    """One ordered expert-transfer stream and bounded completion-event ring."""

    _by_device: dict[tuple[str, int | None], "AsyncExpertTransferExecutor"] = {}
    _registry_lock = Lock()

    def __init__(
        self,
        device: torch.device | str,
        *,
        max_inflight: int = 8,
        stream=None,
        event_factory: Callable[[], object] | None = None,
        stream_context: Callable[[object], object] | None = None,
    ) -> None:
        self.device = _canonical_device(device)
        self.max_inflight = index(max_inflight)
        if self.max_inflight < 1:
            raise ValueError("expert transfer ring size must be positive")

        device_module = None
        if stream is None or event_factory is None or stream_context is None:
            if self.device.type != "cuda":
                raise ValueError("expert transfer executor requires a CUDA device")
            device_module = torch.get_device_module(self.device)
        self.stream = stream or device_module.Stream(device=self.device)
        self._event_factory = event_factory or device_module.Event
        self._stream_context = stream_context or device_module.stream
        self._events = [self._event_factory() for _ in range(self.max_inflight)]
        self._tickets: list[ExpertTransferTicket | None] = [
            None for _ in range(self.max_inflight)
        ]
        self._next_slot = 0
        self._next_sequence = 1

    @classmethod
    def for_device(
        cls, device: torch.device | str | None = None, *, max_inflight: int = 8
    ) -> "AsyncExpertTransferExecutor":
        """Return the process-wide executor dedicated to one CUDA device."""
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("expert transfer requires CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
        resolved = _canonical_device(device)
        if resolved.type != "cuda":
            raise ValueError("expert transfer executor requires a CUDA device")
        key = (resolved.type, resolved.index)
        with cls._registry_lock:
            executor = cls._by_device.get(key)
            if executor is None:
                executor = cls(resolved, max_inflight=max_inflight)
                cls._by_device[key] = executor
            elif executor.max_inflight != index(max_inflight):
                raise ValueError("shared expert transfer executor ring size differs")
            return executor

    def submit(
        self,
        plan: FixedRowTransferPlan,
        copy_operations: Sequence[Callable[[], None]],
        *,
        producer_stream=None,
    ) -> ExpertTransferTicket:
        """Enqueue exactly six NVFP4 copies and record one completion event."""
        operations = tuple(copy_operations)
        if len(operations) != NVFP4_TRANSFER_TENSOR_COUNT:
            raise ValueError("an expert transfer requires all six NVFP4 tensor copies")
        if not all(callable(operation) for operation in operations):
            raise TypeError("expert transfer copy operations must be callable")
        return self._submit_operations(
            plan, operations, producer_stream=producer_stream
        )

    def submit_callback(
        self,
        plan: FixedRowTransferPlan,
        callback: Callable[[], None],
        *,
        producer_stream=None,
    ) -> ExpertTransferTicket:
        """Run one legacy copy callback on the shared executor stream.

        This compatibility path keeps the existing prefetch callback contract
        intact until callers expose the six ModelOpt copies individually.
        New call sites should use :meth:`submit` so the six-copy invariant is
        explicit in their interface.
        """
        if not callable(callback):
            raise TypeError("expert transfer callback must be callable")
        return self._submit_operations(
            plan, (callback,), producer_stream=producer_stream
        )

    def is_complete(self, ticket: ExpertTransferTicket) -> bool:
        """Return whether a current ticket's six copies have all completed."""
        event = self._event_for(ticket)
        complete = bool(event.query())
        if complete:
            ticket.plan._release(ticket.sequence)
        return complete

    def wait(self, ticket: ExpertTransferTicket, consumer_stream=None) -> None:
        """Order a consumer stream and permit plan reuse on that stream."""
        event = self._event_for(ticket)
        if consumer_stream is None:
            consumer_stream = torch.get_device_module(self.device).current_stream(
                self.device
            )
        event.wait(consumer_stream)
        ticket.plan._release(ticket.sequence)

    def _submit_operations(
        self,
        plan: FixedRowTransferPlan,
        operations: Sequence[Callable[[], None]],
        *,
        producer_stream,
    ) -> ExpertTransferTicket:
        if plan.row_count < 1:
            raise ValueError("cannot submit an empty expert transfer plan")
        if (
            plan.device != self.device
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            raise ValueError("transfer plan and executor must use the same CUDA device")
        slot = self._acquire_slot()
        sequence = self._next_sequence
        self._next_sequence += 1
        plan._claim(sequence)
        ticket = ExpertTransferTicket(self.device, slot, sequence, plan)
        try:
            with self._stream_context(self.stream):
                if producer_stream is not None:
                    self.stream.wait_stream(producer_stream)
                for operation in operations:
                    operation()
                self._events[slot].record(self.stream)
        except Exception:
            plan._release(sequence)
            raise
        self._tickets[slot] = ticket
        self._next_slot = (slot + 1) % self.max_inflight
        return ticket

    def _acquire_slot(self) -> int:
        for offset in range(self.max_inflight):
            slot = (self._next_slot + offset) % self.max_inflight
            previous = self._tickets[slot]
            if previous is None:
                return slot
            if self._events[slot].query():
                previous.plan._release(previous.sequence)
                return slot
        raise RuntimeError("expert transfer ticket ring is full")

    def _event_for(self, ticket: ExpertTransferTicket):
        if ticket.device != self.device:
            raise ValueError("expert transfer ticket belongs to a different device")
        current = self._tickets[ticket.slot]
        if current is None or current.sequence != ticket.sequence:
            raise RuntimeError("expert transfer ticket is stale")
        return self._events[ticket.slot]


def submit_expert_row_copies(
    executor: AsyncExpertTransferExecutor,
    plan: FixedRowTransferPlan,
    tensor_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    backend: str,
    source_rows_cpu: Sequence[int],
    destination_slots_cpu: Sequence[int],
    secondary_source_rows_cpu: Sequence[int] | None = None,
    use_secondary_source_rows: Sequence[bool] | None = None,
    producer_stream=None,
) -> ExpertCopySubmission:
    """Submit six row copies, retaining a device-plan GPU path and CPU DMA path."""
    requested_backend = _normalize_copy_backend(backend)
    pairs = tuple(tensor_pairs)
    if len(pairs) != NVFP4_TRANSFER_TENSOR_COUNT:
        raise ValueError("an expert transfer requires six tensor pairs")
    if (
        len(source_rows_cpu) != plan.row_count
        or len(destination_slots_cpu) != plan.row_count
    ):
        raise ValueError("CPU transfer rows must match the fixed plan prefix")
    row_bytes = sum(source[0].numel() * source.element_size() for source, _ in pairs)
    submitted_bytes = plan.row_count * row_bytes
    secondary_rows_cpu = (
        source_rows_cpu
        if secondary_source_rows_cpu is None
        else secondary_source_rows_cpu
    )
    use_secondary_rows = (
        tuple(False for _ in pairs)
        if use_secondary_source_rows is None
        else tuple(use_secondary_source_rows)
    )
    if len(use_secondary_rows) != len(pairs):
        raise ValueError("secondary source row selectors must match tensor pairs")
    if any(use_secondary_rows) and secondary_source_rows_cpu is None:
        raise ValueError("secondary source rows are required by a selected tensor")
    if len(secondary_rows_cpu) != plan.row_count:
        raise ValueError("secondary source rows must match the fixed plan prefix")
    if any(use_secondary_rows) or any(
        source.device.type == "cuda" for source, _ in pairs
    ):
        return _submit_mixed_expert_row_copies(
            executor,
            plan,
            pairs,
            requested_backend,
            source_rows_cpu,
            secondary_rows_cpu,
            use_secondary_rows,
            destination_slots_cpu,
            producer_stream,
            submitted_bytes,
        )

    if requested_backend == "gpu" and all(
        _can_copy_with_gpu(source, destination) for source, destination in pairs
    ):
        ticket = executor.submit(
            plan,
            [
                lambda source=source, destination=destination: copy_expert_rows_gpu(
                    source,
                    destination,
                    plan.source_rows,
                    plan.destination_slots,
                    plan.count,
                )
                for source, destination in pairs
            ],
            producer_stream=producer_stream,
        )
        return ExpertCopySubmission(
            ticket, requested_backend, "gpu", plan.row_count, submitted_bytes, 1, 0
        )

    if requested_backend == "dma" and all(
        _can_copy_with_dma(source, destination) for source, destination in pairs
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("expert DMA transfers are not CUDA-graph-capturable")
        paths: list[str] = []
        dma = ExpertDMABackend()

        def copy_dma(source: torch.Tensor, destination: torch.Tensor) -> None:
            paths.append(
                dma.copy_rows(
                    source, destination, source_rows_cpu, destination_slots_cpu
                )
            )

        ticket = executor.submit(
            plan,
            [
                lambda source=source, destination=destination: copy_dma(
                    source, destination
                )
                for source, destination in pairs
            ],
            producer_stream=producer_stream,
        )
        actual_backend = "dma" if all(path == "dma" for path in paths) else "fallback"
        return ExpertCopySubmission(
            ticket,
            requested_backend,
            actual_backend,
            plan.row_count,
            submitted_bytes,
            1,
            int(actual_backend == "fallback"),
        )

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("expert copy fallback is not CUDA-graph-capturable")
    ticket = executor.submit(
        plan,
        [
            lambda source=source, destination=destination: _copy_rows_fallback(
                source, destination, source_rows_cpu, destination_slots_cpu
            )
            for source, destination in pairs
        ],
        producer_stream=producer_stream,
    )
    return ExpertCopySubmission(
        ticket, requested_backend, "fallback", plan.row_count, submitted_bytes, 1, 1
    )


def _submit_mixed_expert_row_copies(
    executor: AsyncExpertTransferExecutor,
    plan: FixedRowTransferPlan,
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    requested_backend: str,
    source_rows_cpu: Sequence[int],
    secondary_rows_cpu: Sequence[int],
    use_secondary_rows: Sequence[bool],
    destination_slots_cpu: Sequence[int],
    producer_stream,
    submitted_bytes: int,
) -> ExpertCopySubmission:
    paths: list[str] = []
    dma = ExpertDMABackend() if requested_backend == "dma" else None
    copy_operations: list[Callable[[], None]] = []
    for (source, destination), use_secondary in zip(pairs, use_secondary_rows):
        source_rows = plan.secondary_source_rows if use_secondary else plan.source_rows
        source_rows_cpu_for_tensor = (
            secondary_rows_cpu if use_secondary else source_rows_cpu
        )
        if requested_backend == "gpu" and _can_copy_with_gpu(source, destination):
            copy_operations.append(
                lambda source=source, destination=destination, source_rows=source_rows: (
                    _copy_gpu_rows(source, destination, source_rows, plan, paths)
                )
            )
        elif requested_backend == "dma" and _can_copy_with_dma(source, destination):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("expert DMA transfers are not CUDA-graph-capturable")
            assert dma is not None
            copy_operations.append(
                lambda source=source, destination=destination, rows=source_rows_cpu_for_tensor: (
                    _copy_dma_rows(
                        dma, source, destination, rows, destination_slots_cpu, paths
                    )
                )
            )
        elif source.device.type == "cuda" and destination.device.type == "cuda":
            copy_operations.append(
                lambda source=source, destination=destination, source_rows=source_rows: (
                    _copy_d2d_rows(source, destination, source_rows, plan, paths)
                )
            )
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("expert copy fallback is not CUDA-graph-capturable")
            copy_operations.append(
                lambda source=source, destination=destination, rows=source_rows_cpu_for_tensor: (
                    _copy_fallback_rows(
                        source, destination, rows, destination_slots_cpu, paths
                    )
                )
            )
    ticket = executor.submit(plan, copy_operations, producer_stream=producer_stream)
    actual_backend = _actual_backend(requested_backend, paths)
    return ExpertCopySubmission(
        ticket,
        requested_backend,
        actual_backend,
        plan.row_count,
        submitted_bytes,
        1,
        int(actual_backend == "fallback"),
    )


def _copy_gpu_rows(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows: torch.Tensor,
    plan: FixedRowTransferPlan,
    paths: list[str],
) -> None:
    copy_expert_rows_gpu(
        source, destination, source_rows, plan.destination_slots, plan.count
    )
    paths.append("gpu")


def _copy_dma_rows(
    dma: ExpertDMABackend,
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows_cpu: Sequence[int],
    destination_slots_cpu: Sequence[int],
    paths: list[str],
) -> None:
    paths.append(
        dma.copy_rows(source, destination, source_rows_cpu, destination_slots_cpu)
    )


def _copy_d2d_rows(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows: torch.Tensor,
    plan: FixedRowTransferPlan,
    paths: list[str],
) -> None:
    copy_expert_rows_gpu(
        source, destination, source_rows, plan.destination_slots, plan.count
    )
    paths.append("d2d")


def _copy_fallback_rows(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows_cpu: Sequence[int],
    destination_slots_cpu: Sequence[int],
    paths: list[str],
) -> None:
    _copy_rows_fallback(source, destination, source_rows_cpu, destination_slots_cpu)
    paths.append("fallback")


def _actual_backend(requested_backend: str, paths: Sequence[str]) -> str:
    if "fallback" in paths:
        return "fallback"
    return (
        requested_backend
        if all(path == requested_backend for path in paths)
        else "mixed"
    )


def _normalize_copy_backend(backend: str) -> str:
    if backend not in _COPY_BACKENDS:
        raise ValueError("SGLANG_MOE_EXPERT_COPY_BACKEND must be gpu or dma")
    return backend


def _can_copy_with_gpu(source: torch.Tensor, destination: torch.Tensor) -> bool:
    return (
        source.device.type == "cpu"
        and source.is_pinned()
        and destination.device.type == "cuda"
    )


def _can_copy_with_dma(source: torch.Tensor, destination: torch.Tensor) -> bool:
    return _can_copy_with_gpu(source, destination)


def _copy_rows_fallback(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows_cpu: Sequence[int],
    destination_slots_cpu: Sequence[int],
) -> None:
    source_rows = torch.as_tensor(
        source_rows_cpu, dtype=torch.long, device=source.device
    )
    destination_slots = torch.as_tensor(
        destination_slots_cpu, dtype=torch.long, device=destination.device
    )
    if (
        source.device.type == "cpu"
        and not source.is_pinned()
        and destination.device.type == "cuda"
    ):
        source_bytes = source.reshape(source.shape[0], -1).view(torch.uint8)
        staging = torch.empty(
            (len(source_rows_cpu),) + tuple(source_bytes.shape[1:]),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        torch.index_select(source_bytes, 0, source_rows, out=staging)
        destination_bytes = destination.reshape(destination.shape[0], -1).view(
            torch.uint8
        )
        destination_bytes.index_copy_(
            0, destination_slots, staging.to(destination.device, non_blocking=True)
        )
        return
    _copy_rows_by_bytes(
        source, destination, source_rows, destination_slots, len(source_rows_cpu)
    )


def _copy_rows_by_bytes(
    source: torch.Tensor,
    destination: torch.Tensor,
    source_rows: torch.Tensor,
    destination_slots: torch.Tensor,
    row_count: int,
) -> None:
    rows = (
        source.reshape(source.shape[0], -1)
        .view(torch.uint8)
        .index_select(0, source_rows[:row_count])
    )
    if rows.device != destination.device:
        rows = rows.to(destination.device, non_blocking=source.is_pinned())
    destination.reshape(destination.shape[0], -1).view(torch.uint8).index_copy_(
        0, destination_slots[:row_count], rows
    )


def _canonical_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved
