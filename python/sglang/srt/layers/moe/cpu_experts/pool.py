"""CPU compute of a streamed layer's RAM-tier experts, in place over the pinned slabs.

``CpuExpertQuantTrait`` is the interface one expert format's CPU kernel implements;
everything format-specific lives behind it. ``CpuExpertPool`` registers every streamed
layer's pinned slabs with a trait once and runs the kernel over them. Each streamed
layer owns one ``ExpertPinnedHostCache``, so a layer's host slot ids index its own slabs
and the kernel's "expert id" is that slot. Decode only (batch 1).

The Python half of the RAM-miss integration is ``cpu_experts/service.py``; the native
CPU expert thread is ``expert_stream/host/cpu_experts.h``.
"""

import ctypes
import os
import threading
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch


CPU_EXPERTS_FORWARD_ABI_VERSION = 1


class CpuExpertsForwardCall(ctypes.Structure):
    """``SglangCpuExpertsForward`` (``expert_stream/host/cpu_experts_abi.h``); the field order is the C struct's."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("rows", ctypes.c_int32),
        ("layer", ctypes.c_int64),
        ("x", ctypes.c_void_p),
        ("slots", ctypes.POINTER(ctypes.c_int32)),
        ("weights", ctypes.POINTER(ctypes.c_float)),
        ("out", ctypes.POINTER(ctypes.c_float)),
        ("k", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("accumulate", ctypes.c_int32),
    ]


CPU_EXPERTS_LAYER_ABI_VERSION = 1
CPU_EXPERTS_MAX_SLABS = 8


class CpuExpertsLayer(ctypes.Structure):
    """``SglangCpuExpertsLayer`` (``expert_stream/host/cpu_experts_abi.h``); the field order is the C struct's."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("capacity", ctypes.c_int32),
        ("hidden", ctypes.c_int32),
        ("intermediate", ctypes.c_int32),
        ("activation", ctypes.c_int32),
        ("act_limit", ctypes.c_float),
        ("slab_count", ctypes.c_int32),
        ("slabs", ctypes.c_void_p * CPU_EXPERTS_MAX_SLABS),
        ("slot_bytes", ctypes.c_uint64 * CPU_EXPERTS_MAX_SLABS),
        ("params", ctypes.c_void_p),
    ]


# A CpuExpertForward as ctypes: a native forward's address cast to it, or a Python fake the engine can call.
CpuExpertForward = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(CpuExpertsForwardCall))


class CpuExpertQuantTrait(Protocol):
    """One expert format's CPU kernel.

    ``slab_names`` are the pinned-tier tensors it reads. ``native_forward`` and
    ``native_set_cores`` are the kernel's native half, which the RAM-miss service's CPU
    expert thread calls without Python (``expert_stream/host/cpu_experts.h``). That
    forward reads ``x`` as fp16 ``[hidden]`` (the post kernel stages it so) and the
    weights as fp32, and takes ``register_layer``'s handle as an int64 layer argument.
    """

    name: str
    slab_names: tuple[str, ...]
    # The SwiGLU clamp the layers run with; the service sets it on first registration.
    act_limit: Optional[float]
    # Dtypes of the hidden state and routing weights in, and of the output.
    x_dtype: torch.dtype
    weights_dtype: torch.dtype
    out_dtype: torch.dtype

    def check_environment(self) -> None:
        """Raise if the process is set up in a way that breaks the kernel's threads."""
        ...

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The hidden size (length of x and out), as the layer's slabs encode it."""
        ...

    def register_layer(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> Any:
        """Register one layer's slab rows once, as views.

        The handle's expert ids are the host slots.
        """
        ...

    def forward(
        self,
        handle: Any,
        x: torch.Tensor,
        slots: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
        threads: int,
    ) -> None:
        """Overwrite ``out`` with the routed sum over ``slots`` (int64 ``[1, k]``).

        A slot of -1 is skipped.
        """
        ...

    def free_layer(self, handle: Any) -> None:
        """Release the kernel's state for ``handle``."""
        ...

    def native_forward(self) -> int:
        """The address of the kernel's ``CpuExpertForward`` C function."""
        ...

    def native_keep_warm(self) -> int:
        """The address of the kernel's ``CpuExpertKeepWarm`` C function."""
        ...

    def native_set_cores(self, cores: Sequence[int]) -> None:
        """Place the kernel's workers on ``cores`` before its first forward.

        The calling thread is worker 0.
        """
        ...


class CpuExpertPool:
    """A trait's kernel layers, one per streamed layer, registered once over the slabs.

    ``slabs_by_layer[layer]`` is that layer's ``ExpertPinnedHostCache.tensors``. Row
    contents change as the tier evicts and fills, but the registered views stay valid,
    so nothing re-registers.

    One thread calls ``compute``, and it must call ``bind_current_thread`` first: the
    kernel's workers inherit the affinity of the thread that first runs a job, which is
    also its worker 0. ``close`` frees the kernel's layers.
    """

    def __init__(
        self,
        trait: CpuExpertQuantTrait,
        slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        cores: Sequence[int],
        threads: int,
    ):
        cores = sorted(set(cores))
        if len(cores) < 2:
            # Spinning workers sharing one core livelock
            # (DSV41_REFERENCE.md section 28.2).
            raise ValueError(f"the CPU expert pool needs at least 2 cores, got {cores}")
        if not 1 <= threads <= len(cores):
            raise ValueError(f"{threads} pool threads on {len(cores)} cores")
        trait.check_environment()
        self.trait = trait
        self.cores = tuple(cores)
        self.threads = threads
        self.capacity: dict[int, int] = {}
        self._handles: dict[int, Any] = {}
        self._bound_threads: set[int] = set()
        try:
            for layer, slabs in slabs_by_layer.items():
                self._register(layer, slabs)
        except BaseException:
            self.close()
            raise

    def _register(self, layer: int, slabs: Mapping[str, torch.Tensor]) -> None:
        """Validate ``layer``'s slabs (equal rows, contiguous CPU) and register them."""
        views = {name: slabs[name] for name in self.trait.slab_names}
        rows = {int(t.shape[0]) for t in views.values()}
        if len(rows) != 1:
            raise ValueError(
                f"layer {layer}: pinned slabs disagree on their row count {rows}"
            )
        capacity = rows.pop()
        for name, slab in views.items():
            if slab.device.type != "cpu" or not slab.is_contiguous():
                raise ValueError(
                    f"layer {layer} {name}: slab must be a contiguous CPU tensor"
                )
        self.capacity[layer] = capacity
        if capacity:
            self._handles[layer] = self.trait.register_layer(views, capacity)

    def bind_current_thread(self) -> None:
        """Pin the calling thread to the pool's cores; the compute thread, once."""
        os.sched_setaffinity(0, self.cores)
        self._bound_threads.add(threading.get_native_id())

    def compute(
        self,
        layer: int,
        host_slots: torch.Tensor,
        weights: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Overwrite ``out`` ``[1, H]`` with this layer's CPU share of the routed sum.

        ``host_slots`` is int64 ``[1, k]`` (-1 skips) and ``weights`` is ``[1, k]``,
        with the dtypes the trait names. Must run on a thread that called
        ``bind_current_thread``.
        """
        if threading.get_native_id() not in self._bound_threads:
            raise RuntimeError(
                "compute() from a thread that has not called bind_current_thread()"
            )
        if x.shape[0] != 1:
            raise ValueError(
                f"the CPU expert pool is batch-1 decode only, got {x.shape[0]} rows"
            )
        trait = self.trait
        if (
            host_slots.dtype != torch.int64
            or host_slots.shape[0] != 1
            or weights.shape != host_slots.shape
        ):
            raise ValueError(
                "host_slots must be int64 [1, k] with weights of the same shape"
            )
        if (
            x.dtype != trait.x_dtype
            or weights.dtype != trait.weights_dtype
            or out.dtype != trait.out_dtype
        ):
            raise ValueError(
                f"{trait.name} takes x {trait.x_dtype}, weights {trait.weights_dtype}, out {trait.out_dtype}"
            )
        if layer not in self.capacity:
            raise ValueError(f"layer {layer} is not in the CPU expert pool")
        capacity = self.capacity[layer]
        top = int(host_slots.max())
        if top >= capacity:
            # The kernel would skip an out-of-range slot silently.
            raise ValueError(
                f"layer {layer}: host slot {top} is outside the tier's {capacity} rows"
            )
        if capacity == 0:
            out.zero_()
            return
        trait.forward(self._handles[layer], x, host_slots, weights, out, self.threads)

    def close(self) -> None:
        """Free every registered layer; safe to call twice."""
        handles, self._handles = self._handles, {}
        for handle in handles.values():
            self.trait.free_layer(handle)
