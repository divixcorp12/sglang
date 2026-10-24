"""What an expert format's CPU kernel gives the RAM-miss service: each layer's slabs and parameters.

The native half is ``expert_stream/host/cpu_experts/kernel.hpp``, compiled into the host module: the host calls the
kernel's ``make_layer`` with a
``CpuExpertLayerSpec`` (``ExpertStreamHost.set_cpu_layer``) and runs its forwards on the CPU expert threads.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Protocol

import torch


@dataclasses.dataclass(frozen=True)
class CpuExpertLayerSpec:
    """One layer's pinned host tier as the kernel's ``make_layer`` reads it (``ExpertLayer``), and the quant's parameters.

    ``params`` is the format's parameter list as the host takes it (int64 values; the kernel's C++ builds and validates
    its parameter struct from them). ``slabs`` holds one ``(address, slot bytes)`` pair per slab in the quant's order, ``(0, 0)`` for an absent optional
    slab; slot ``s`` of slab ``i`` starts at ``address + s * slot_bytes``. ``keep`` holds the tensors those addresses
    point into: whoever registers the spec keeps them alive for as long as the host may read them. ``plan`` describes
    the kernel specialization the params chose, for the registration log ("" when the quant has one).
    """

    capacity: int
    hidden: int
    intermediate: int
    act_limit: float
    slabs: tuple[tuple[int, int], ...]
    params: tuple[int, ...]
    keep: tuple = ()
    activation: int = 0
    plan: str = ""


class CpuExpertQuantTrait(Protocol):
    """One expert format's CPU kernel.

    ``slab_names`` are the pinned-tier tensors it reads; ``layer_spec`` is what the RAM-miss service gives the host
    (``expert_stream/host/cpu_experts/kernel.hpp``); its CPU expert threads run the kernel without Python.
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

    def attach_host(self, host) -> None:
        """Take the host module whose compiled-in kernel the layers run on: what ``layer_spec`` reads of the kernel
        (EXL3's plan list, ``host.cpu_kernel_plans()``) comes from it."""
        ...

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """One layer's first ``capacity`` slab rows, checked, as the kernel's ``make_layer`` takes them."""
        ...
