"""CPU experts inside the EXL3 RAM-miss service (plan 2026-09-29-dsv41-cpu-experts, "Step B").

The service's own threads run the kernel: the grant tags ``split[n]`` of a copy-engine request's n resident lanes
CPU, the copy thread hands them to the CPU expert thread (expert_stream/host/cpu_experts.h), and the copy wait
releases the decode stream once both the copies and the CPU are done. This module owns the Python half: the quant
trait, the pinned rows the post kernel and the CPU exchange, the lazy per-layer registration, and the split table.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.policy import parse_core_list, split_table

logger = logging.getLogger(__name__)

LANES = 8  # expert_lease_block.LANES: the split table covers n = 0..LANES resident lanes


def cpu_trait_for(format_key: str, ext=None):
    """The quant trait of a streamed expert format; only EXL3 has a CPU kernel today."""
    if format_key == "exl3":
        from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
        from sglang.srt.layers.quantization.exl3_ext import exl3_ext

        return Exl3CpuQuantTrait(ext if ext is not None else exl3_ext(), act_limit=None)
    raise ValueError(f"CPU experts have no kernel for expert format {format_key!r}")


def configured_split() -> list[int]:
    """SGLANG_DSV41_CPU_EXPERTS_SPLIT when set, else k*(n) from the configured per-expert costs."""
    spec = envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.get()
    if spec:
        split = [int(v) for v in spec.split(",")]
        if len(split) != LANES + 1 or any(not 0 <= k <= n for n, k in enumerate(split)):
            raise ValueError(
                f"SGLANG_DSV41_CPU_EXPERTS_SPLIT must list {LANES + 1} counts with 0 <= split[n] <= n, got {spec!r}"
            )
        return split
    return split_table(
        LANES,
        envs.SGLANG_DSV41_CPU_EXPERTS_CPU_MS.get(),
        envs.SGLANG_DSV41_CPU_EXPERTS_LINK_MS.get(),
        envs.SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS.get(),
    ).tolist()


def core_node(core: int, root: str = "/sys/devices/system/cpu") -> int:
    for name in os.listdir(os.path.join(root, f"cpu{core}")):
        if name.startswith("node") and name[4:].isdigit():
            return int(name[4:])
    raise RuntimeError(f"cannot find the NUMA node of CPU {core}")


def slot_nodes(capacity: int, placement) -> list[int]:
    """Each host slot's NUMA node as the pinned tier bound it (host_numa.split_rows per named slab); [] unplaced."""
    if not placement:
        return []
    from sglang.srt.layers.moe.host_numa import split_rows

    nodes = [-1] * capacity
    for node, first, count in split_rows(capacity, placement):
        nodes[first : first + count] = [node] * count
    return nodes


class CpuExpertService:
    """The Python half of CPU experts for one ``ExpertStreamHost`` (the RAM-miss service's)."""

    def __init__(
        self,
        host,
        trait,
        slabs_by_row: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        hidden: int,
        cores: Sequence[int],
        threads: int,
        split: Sequence[int],
        placement=(),
        pin: bool = True,
    ):
        cores = sorted(set(cores))
        if len(cores) < 2:
            # Many spinning workers on one core livelocked the box (DSV41_REFERENCE section 28).
            raise ValueError(f"CPU experts need at least 2 cores, got {cores}")
        if not 1 <= threads <= len(cores):
            raise ValueError(f"{threads} CPU expert threads on {len(cores)} cores")
        trait.check_environment()
        self.host, self.trait = host, trait
        self.slabs_by_row = dict(slabs_by_row)
        self.hidden, self.cores, self.threads = int(hidden), tuple(cores), int(threads)
        self.split = list(split)
        rows = len(self.slabs_by_row)
        if sorted(self.slabs_by_row) != list(range(rows)):
            raise ValueError("CPU experts need the slabs of every service row 0..rows-1")
        # fp16 input rows padded to 16 bytes (the post's vector stores), fp32 output rows. Pinned: the device reads and
        # writes them through UVA.
        x_bytes = -(-2 * self.hidden // 16) * 16
        self.x_rows = torch.zeros((rows, x_bytes), dtype=torch.uint8)
        self.out_rows = torch.zeros((rows, self.hidden), dtype=torch.float32)
        if pin:
            self.x_rows, self.out_rows = self.x_rows.pin_memory(), self.out_rows.pin_memory()
        self.handles: dict[int, object] = {}
        self._cores_set = False
        host.enable_cpu_experts(
            trait.native_forward(), self.split, self.cores, self.x_rows, self.out_rows, threads=self.threads
        )
        preferred = core_node(self.cores[0]) if placement else -1
        for row, slabs in self.slabs_by_row.items():
            nodes = slot_nodes(self._capacity(slabs), placement)
            if nodes:
                host.set_cpu_slot_nodes(row, nodes, preferred)
        self._last_stats = host.cpu_stats()
        logger.info(
            "CPU experts on: %s trait, cores %s, %d threads, split %s, NUMA preference node %s",
            trait.name, list(self.cores), self.threads, self.split, preferred if placement else "off",
        )

    def _capacity(self, slabs: Mapping[str, torch.Tensor]) -> int:
        rows = {int(slabs[name].shape[0]) for name in self.trait.slab_names}
        if len(rows) != 1:
            raise ValueError(f"the pinned slabs disagree on their row count {rows}")
        return rows.pop()

    def registered(self, row: int) -> bool:
        return row in self.handles

    def register(self, row: int, act_limit: Optional[float]) -> None:
        """Register ``row``'s pinned slabs with the trait and let the grant send its lanes to the CPU.

        On a layer's first graph-path forward, a warmup before capture: the activation limit is the layer's
        MoE runner config, which the service does not see at startup. Every layer must use the same limit.
        """
        if row in self.handles:
            return
        if self.trait.act_limit is None:
            self.trait.act_limit = act_limit
        elif self.trait.act_limit != act_limit:
            raise ValueError(f"CPU experts: layer rows use activation limits {self.trait.act_limit} and {act_limit}")
        if not self._cores_set:
            # Before the kernel's first forward, which the CPU expert thread runs once a row is registered.
            self.trait.native_set_cores(self.cores)
            self._cores_set = True
        slabs = self.slabs_by_row[row]
        capacity = self._capacity(slabs)
        if capacity == 0:
            return
        handle = self.trait.register_layer({name: slabs[name] for name in self.trait.slab_names}, capacity)
        self.handles[row] = handle
        self.host.set_cpu_layer(row, int(handle))

    def retune(self) -> Optional[list[int]]:
        """P3: recompute the split from the CPU's measured per-expert cost since the last call, when it has run.

        The link and handoff costs stay as configured. Returns the new table when it changed.
        """
        stats = self.host.cpu_stats()
        lanes = stats["lanes"] - self._last_stats["lanes"]
        ns = stats["forward_ns"] - self._last_stats["forward_ns"]
        if lanes < 64:
            return None
        self._last_stats = stats
        c_cpu = ns / lanes / 1e6
        split = split_table(
            LANES, c_cpu, envs.SGLANG_DSV41_CPU_EXPERTS_LINK_MS.get(), envs.SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS.get()
        ).tolist()
        if split == self.split:
            return None
        self.split = split
        self.host.set_cpu_split(split)
        logger.info("CPU experts: measured %.3f ms per expert, split now %s", c_cpu, split)
        return split


def cpu_expert_cores() -> tuple[list[int], int]:
    """SGLANG_DSV41_CPU_EXPERTS_CORES and _THREADS, checked."""
    spec = envs.SGLANG_DSV41_CPU_EXPERTS_CORES.get()
    if not spec:
        raise ValueError("SGLANG_DSV41_CPU_EXPERTS needs SGLANG_DSV41_CPU_EXPERTS_CORES (a taskset list, e.g. 18-29)")
    cores = parse_core_list(spec)
    threads = envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.get() or len(cores)
    return cores, threads
