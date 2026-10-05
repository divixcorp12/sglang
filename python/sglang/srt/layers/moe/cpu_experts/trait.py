"""What an expert format's CPU kernel gives the RAM-miss service: its kernel's address and each layer's slabs.

The native half is ``expert_stream/host/cpu_experts/kernel.hpp``: the host calls the kernel's ``make_layer`` with a
``CpuExpertLayerSpec`` (``ExpertStreamHost.set_cpu_layer``) and runs its forwards on the CPU expert threads.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Protocol

import torch


@dataclasses.dataclass(frozen=True)
class CpuExpertLayerSpec:
    """One layer's pinned host tier as the kernel's ``make_layer`` reads it (``LayerSlabs``), and the quant's parameter
    bytes.

    ``slabs`` holds one ``(address, slot bytes)`` pair per slab in the quant's order, ``(0, 0)`` for an absent optional
    slab; slot ``s`` of slab ``i`` starts at ``address + s * slot_bytes``. ``keep`` holds the tensors those addresses
    point into: whoever registers the spec keeps them alive for as long as the host may read them.
    """

    capacity: int
    hidden: int
    intermediate: int
    act_limit: float
    slabs: tuple[tuple[int, int], ...]
    params: bytes
    keep: tuple = ()
    activation: int = 0


class CpuExpertQuantTrait(Protocol):
    """One expert format's CPU kernel.

    ``slab_names`` are the pinned-tier tensors it reads; ``kernel_address`` and ``layer_spec`` are what the RAM-miss
    service gives the host (``expert_stream/host/cpu_experts/kernel.hpp``); its CPU expert threads run the kernel
    without Python.
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

    def kernel_address(self) -> int:
        """The address of the format's ``CpuExpertKernel`` (``expert_stream/host/cpu_experts/kernel.hpp``)."""
        ...

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """One layer's first ``capacity`` slab rows, checked, as the kernel's ``make_layer`` takes them."""
        ...
