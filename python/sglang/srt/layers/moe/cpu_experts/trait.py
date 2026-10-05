"""What an expert format's CPU kernel gives the RAM-miss service: its kernel's address and each layer's slabs.

The native half is ``expert_stream/host/cpu_experts/kernel.hpp``: the host calls the kernel's ``make_layer`` with a
``CpuExpertLayerSpec`` (``ExpertStreamHost.set_cpu_layer``) and runs its forwards on the CPU expert threads.
"""

from __future__ import annotations

import dataclasses


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
