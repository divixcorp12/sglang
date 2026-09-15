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

import numpy as np
import torch
from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu
from sglang.srt.layers.moe.expert_dma import ExpertDMARowRoute
from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor


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
        self._rows64 = torch.zeros(
            (4, self.max_rows), dtype=torch.int64, device=self.device
        )
        self._rows32 = torch.zeros(
            self.max_rows + 1, dtype=torch.int32, device=self.device
        )
        self.source_rows = self._rows64[0]
        self.secondary_source_rows = self._rows64[1]
        self.destination_slots_long = self._rows64[2]
        self.generations = self._rows64[3]
        self.destination_slots = self._rows32[: self.max_rows]
        self.count = self._rows32[self.max_rows :]
        if self.device.type == "cuda":
            self._host_rows64 = torch.zeros_like(self._rows64, device="cpu").pin_memory()
            self._host_rows32 = torch.zeros_like(self._rows32, device="cpu").pin_memory()
            self._upload_event = torch.cuda.Event()
        else:
            self._host_rows64 = self._rows64
            self._host_rows32 = self._rows32
            self._upload_event = None
        self._host_rows64_view = self._host_rows64.numpy()
        self._host_rows32_view = self._host_rows32.numpy()
        self._uploaded = False
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
        secondary_source_rows: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Replace the active prefix while preserving all plan allocations.

        Rows are staged in persistent pinned host buffers and reach the device
        through two non-blocking copies of the packed int64 and int32 plans.
        Entries past the prefix are zero. ``secondary_source_rows``, when
        given, replaces the secondary prefix; otherwise it is left unchanged.
        """
        if self._active_sequence is not None:
            raise RuntimeError("cannot rewrite a transfer plan while it is in flight")
        source = self._as_vector(source_rows, "source_rows")
        destination = self._as_vector(destination_slots, "destination_slots")
        generation = self._as_vector(generations, "generations")
        secondary = (
            None
            if secondary_source_rows is None
            else self._as_vector(secondary_source_rows, "secondary_source_rows")
        )
        row_count = source.shape[0]
        if destination.shape[0] != row_count or generation.shape[0] != row_count:
            raise ValueError("transfer plan rows must have the same number of entries")
        if secondary is not None and secondary.shape[0] != row_count:
            raise ValueError("transfer plan rows must have the same number of entries")
        if row_count > self.max_rows:
            raise ValueError("transfer plan row count exceeds fixed capacity")
        if self._upload_event is not None and self._uploaded:
            self._upload_event.synchronize()
        rows64 = self._host_rows64_view
        rows32 = self._host_rows32_view
        rows64[0, :row_count] = source
        rows64[2, :row_count] = destination
        rows64[3, :row_count] = generation
        rows64[(0, 2, 3), row_count:] = 0
        if secondary is not None:
            rows64[1, :row_count] = secondary
            rows64[1, row_count:] = 0
        rows32[:row_count] = destination
        rows32[row_count : self.max_rows] = 0
        rows32[self.max_rows] = row_count
        if self._upload_event is not None:
            self._rows64.copy_(self._host_rows64, non_blocking=True)
            self._rows32.copy_(self._host_rows32, non_blocking=True)
            self._upload_event.record()
            self._uploaded = True
        self._row_count = row_count

    @staticmethod
    def _as_vector(values: torch.Tensor | Sequence[int], name: str) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            if values.is_floating_point() or values.is_complex():
                raise ValueError(f"transfer plan {name} must contain integer values")
            values = values.detach().cpu().numpy()
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"transfer plan {name} must be one-dimensional")
        if array.dtype.kind not in "iub":
            raise ValueError(f"transfer plan {name} must contain integer values")
        return array

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
    extra_plans: tuple[FixedRowTransferPlan, ...] = ()

    def release_plans(self) -> None:
        """Permit every plan this ticket claimed to be rewritten."""
        self.plan._release(self.sequence)
        for plan in self.extra_plans:
            plan._release(self.sequence)


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
            (plan,), operations, producer_stream=producer_stream
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
            (plan,), (callback,), producer_stream=producer_stream
        )

    def submit_batch(
        self,
        plans: Sequence[FixedRowTransferPlan],
        callback: Callable[[], None],
        *,
        producer_stream=None,
    ) -> ExpertTransferTicket:
        """Run one callback issuing several plans' copies behind one completion event.

        Every plan is claimed until the shared ticket completes, so a whole
        residency update occupies one ring slot however many layers it changes.
        """
        if not plans:
            raise ValueError("an expert transfer batch requires at least one plan")
        if not callable(callback):
            raise TypeError("expert transfer callback must be callable")
        return self._submit_operations(
            tuple(plans), (callback,), producer_stream=producer_stream
        )

    def is_complete(self, ticket: ExpertTransferTicket) -> bool:
        """Return whether a current ticket's six copies have all completed."""
        event = self._event_for(ticket)
        complete = bool(event.query())
        if complete:
            ticket.release_plans()
        return complete

    def has_completed(self, ticket: ExpertTransferTicket) -> bool:
        """Like :meth:`is_complete`, but a ticket whose ring slot was reused is complete.

        A ring slot is reused only after its event completed, so a caller that
        polls late still learns its copies landed instead of seeing a stale ticket.
        """
        if ticket.device != self.device:
            raise ValueError("expert transfer ticket belongs to a different device")
        current = self._tickets[ticket.slot]
        if current is None or current.sequence != ticket.sequence:
            return True
        return self.is_complete(ticket)

    def wait(self, ticket: ExpertTransferTicket, consumer_stream=None) -> None:
        """Order a consumer stream and permit plan reuse on that stream."""
        event = self._event_for(ticket)
        if consumer_stream is None:
            consumer_stream = torch.get_device_module(self.device).current_stream(
                self.device
            )
        event.wait(consumer_stream)
        ticket.release_plans()

    def _submit_operations(
        self,
        plans: Sequence[FixedRowTransferPlan],
        operations: Sequence[Callable[[], None]],
        *,
        producer_stream,
    ) -> ExpertTransferTicket:
        for plan in plans:
            if plan.row_count < 1:
                raise ValueError("cannot submit an empty expert transfer plan")
            if (
                plan.device != self.device
                and self.device.type == "cuda"
                and torch.cuda.is_available()
            ):
                raise ValueError(
                    "transfer plan and executor must use the same CUDA device"
                )
        if len({id(plan) for plan in plans}) != len(plans):
            raise ValueError("an expert transfer batch cannot repeat a plan")
        slot = self._acquire_slot()
        sequence = self._next_sequence
        self._next_sequence += 1
        ticket = ExpertTransferTicket(
            self.device, slot, sequence, plans[0], tuple(plans[1:])
        )
        claimed = []
        try:
            for plan in plans:
                plan._claim(sequence)
                claimed.append(plan)
            with self._stream_context(self.stream):
                if producer_stream is not None:
                    self.stream.wait_stream(producer_stream)
                for operation in operations:
                    operation()
                self._events[slot].record(self.stream)
        except Exception:
            for plan in claimed:
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
                previous.release_plans()
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
    routes = ExpertRowCopyRoutes(
        tensor_pairs,
        backend=backend,
        use_secondary_source_rows=use_secondary_source_rows,
    )
    (submission,) = submit_expert_row_copy_batch(
        executor,
        [
            ExpertRowCopyRequest(
                routes,
                plan,
                source_rows_cpu,
                destination_slots_cpu,
                secondary_source_rows_cpu,
            )
        ],
        producer_stream=producer_stream,
    )
    return submission


_ROUTE_GPU = "gpu"
_ROUTE_DMA = "dma"
_ROUTE_D2D = "d2d"
_ROUTE_FALLBACK = "fallback"


class ExpertRowCopyRoutes:
    """The copy path of each of one layer's six source and destination pairs.

    Paths follow the selection rules of :func:`submit_expert_row_copies`: a
    pair set with CUDA sources or secondary rows picks a path per pair, and
    otherwise all six use the requested backend when every source is GPU
    readable, or the fallback. Readability, DMA tensor validation and
    registration bounds are resolved once here, so a caller holding routes for
    stable tensors pays none of them per submission.
    """

    def __init__(
        self,
        tensor_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        backend: str,
        use_secondary_source_rows: Sequence[bool] | None = None,
    ) -> None:
        self.requested_backend = _normalize_copy_backend(backend)
        self.pairs = tuple(tensor_pairs)
        if len(self.pairs) != NVFP4_TRANSFER_TENSOR_COUNT:
            raise ValueError("an expert transfer requires six tensor pairs")
        self.use_secondary_rows = (
            tuple(False for _ in self.pairs)
            if use_secondary_source_rows is None
            else tuple(bool(flag) for flag in use_secondary_source_rows)
        )
        if len(self.use_secondary_rows) != len(self.pairs):
            raise ValueError("secondary source row selectors must match tensor pairs")
        self.row_bytes = sum(
            source[0].numel() * source.element_size() for source, _ in self.pairs
        )
        mixed = any(self.use_secondary_rows) or any(
            source.device.type == "cuda" for source, _ in self.pairs
        )
        if mixed:
            kinds = tuple(
                self._mixed_kind(source, destination) for source, destination in self.pairs
            )
        elif self.requested_backend == _ROUTE_GPU and all(
            _can_copy_with_gpu(source, destination) for source, destination in self.pairs
        ):
            kinds = (_ROUTE_GPU,) * len(self.pairs)
        elif self.requested_backend == _ROUTE_DMA and all(
            _can_copy_with_dma(source, destination) for source, destination in self.pairs
        ):
            kinds = (_ROUTE_DMA,) * len(self.pairs)
        else:
            kinds = (_ROUTE_FALLBACK,) * len(self.pairs)
        self.kinds = kinds
        self._dma_routes = tuple(
            ExpertDMARowRoute(source, destination) if kind == _ROUTE_DMA else None
            for (source, destination), kind in zip(self.pairs, kinds)
        )
        paths = [
            ("dma" if dma_route.aot_available else "fallback")
            if dma_route is not None
            else kind
            for kind, dma_route in zip(kinds, self._dma_routes)
        ]
        self.actual_backend = (
            _actual_backend(self.requested_backend, paths)
            if mixed
            else ("fallback" if "fallback" in paths else kinds[0])
        )
        self.fallbacks = int(self.actual_backend == "fallback")
        self.host_issued = any(kind in (_ROUTE_DMA, _ROUTE_FALLBACK) for kind in kinds)

    def _mixed_kind(self, source: torch.Tensor, destination: torch.Tensor) -> str:
        if self.requested_backend == _ROUTE_GPU and _can_copy_with_gpu(source, destination):
            return _ROUTE_GPU
        if self.requested_backend == _ROUTE_DMA and _can_copy_with_dma(source, destination):
            return _ROUTE_DMA
        if source.device.type == "cuda" and destination.device.type == "cuda":
            return _ROUTE_D2D
        return _ROUTE_FALLBACK

    def copy_rows(
        self,
        plan: FixedRowTransferPlan,
        source_rows_cpu: Sequence[int],
        secondary_rows_cpu: Sequence[int],
        destination_slots_cpu: Sequence[int],
        coalesced: dict,
    ) -> None:
        """Issue the six row copies of one plan on the current stream."""
        for (source, destination), kind, use_secondary, dma_route in zip(
            self.pairs, self.kinds, self.use_secondary_rows, self._dma_routes
        ):
            rows_cpu = secondary_rows_cpu if use_secondary else source_rows_cpu
            if dma_route is not None:
                dma_route.copy_rows(rows_cpu, destination_slots_cpu, coalesced)
            elif kind == _ROUTE_FALLBACK:
                _copy_rows_fallback(source, destination, rows_cpu, destination_slots_cpu)
            else:
                copy_expert_rows_gpu(
                    source,
                    destination,
                    plan.secondary_source_rows if use_secondary else plan.source_rows,
                    plan.destination_slots,
                    plan.count,
                )


@dataclass(frozen=True)
class ExpertRowCopyRequest:
    """One plan's rows and the routes that copy them."""

    routes: ExpertRowCopyRoutes
    plan: FixedRowTransferPlan
    source_rows_cpu: Sequence[int]
    destination_slots_cpu: Sequence[int]
    secondary_source_rows_cpu: Sequence[int] | None = None


def submit_expert_row_copy_batch(
    executor: AsyncExpertTransferExecutor,
    requests: Sequence[ExpertRowCopyRequest],
    *,
    producer_stream=None,
) -> tuple[ExpertCopySubmission, ...]:
    """Submit several plans' six-tensor copies behind one executor ticket.

    Each request is validated as :func:`submit_expert_row_copies` validates
    its arguments and reports its own :class:`ExpertCopySubmission`; all of
    them share the returned ticket.
    """
    requests = tuple(requests)
    if not requests:
        raise ValueError("an expert transfer batch requires at least one request")
    for request in requests:
        rows = request.plan.row_count
        if (
            len(request.source_rows_cpu) != rows
            or len(request.destination_slots_cpu) != rows
        ):
            raise ValueError("CPU transfer rows must match the fixed plan prefix")
        secondary = request.secondary_source_rows_cpu
        if any(request.routes.use_secondary_rows) and secondary is None:
            raise ValueError("secondary source rows are required by a selected tensor")
        if secondary is not None and len(secondary) != rows:
            raise ValueError("secondary source rows must match the fixed plan prefix")
    if any(request.routes.host_issued for request in requests):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("expert host-issued copies are not CUDA-graph-capturable")

    def copy_all() -> None:
        coalesced: dict = {}
        for request in requests:
            request.routes.copy_rows(
                request.plan,
                request.source_rows_cpu,
                request.source_rows_cpu
                if request.secondary_source_rows_cpu is None
                else request.secondary_source_rows_cpu,
                request.destination_slots_cpu,
                coalesced,
            )

    ticket = executor.submit_batch(
        [request.plan for request in requests], copy_all, producer_stream=producer_stream
    )
    return tuple(
        ExpertCopySubmission(
            ticket,
            request.routes.requested_backend,
            request.routes.actual_backend,
            request.plan.row_count,
            request.plan.row_count * request.routes.row_bytes,
            1,
            request.routes.fallbacks,
        )
        for request in requests
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
    return is_gpu_readable_host_tensor(source) and destination.device.type == "cuda"


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
        and not is_gpu_readable_host_tensor(source)
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
        rows = rows.to(
            destination.device, non_blocking=is_gpu_readable_host_tensor(source)
        )
    destination.reshape(destination.shape[0], -1).view(torch.uint8).index_copy_(
        0, destination_slots[:row_count], rows
    )


def _canonical_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved
