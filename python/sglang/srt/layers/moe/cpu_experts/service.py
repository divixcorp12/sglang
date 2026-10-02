"""CPU experts inside the EXL3 RAM-miss service (plan 2026-09-29-dsv41-cpu-experts, "Step B").

The service's own threads run the kernel.
The post types the last ``split[n]`` of a captured post's n eligible lanes CPU (RAM hits, and NVMe misses with
SGLANG_DSV41_CPU_EXPERTS_MISSES); the device plan sorts them, so those are the lowest-scored.
The service thread hands them to the CPU expert thread (expert_stream/host/cpu_experts.h): the hits at once into part 0
of the row's output, each miss into part 1 as soon as its read landed. The copy wait releases the decode stream once
the copies and the CPU are done.

This module owns the Python half: the quant trait, the pinned rows the post kernel and the CPU exchange,
the lazy per-layer registration, and the split table.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.policy import (
    format_calibration,
    parse_core_list,
    split_from_grid,
    split_table,
)

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
        self.calibrated = False
        self._calibration_stats = {"jobs": 0, "lanes": 0, "forward_ns": 0}
        self._calibration_scratch: Optional[torch.Tensor] = None
        rows = len(self.slabs_by_row)
        if sorted(self.slabs_by_row) != list(range(rows)):
            raise ValueError("CPU experts need the slabs of every service row 0..rows-1")
        # fp16 input rows padded to 16 bytes (the post's vector stores), fp32 output rows. Pinned: the device reads and
        # writes them through UVA.
        x_bytes = -(-2 * self.hidden // 16) * 16
        self.x_rows = torch.zeros((rows, x_bytes), dtype=torch.uint8)
        # Two parts per row: the CPU hits' partial sum (part 0) and the CPU misses' (part 1), summed by the route tables.
        self.out_rows = torch.zeros((rows, 2, self.hidden), dtype=torch.float32)
        if pin:
            self.x_rows, self.out_rows = self.x_rows.pin_memory(), self.out_rows.pin_memory()
        self.handles: dict[int, object] = {}
        self._cores_set = False
        host.enable_cpu_experts(
            trait.native_forward(), self.split, self.cores, self.x_rows, self.out_rows, threads=self.threads
        )
        self._last_stats = host.cpu_stats()
        logger.info(
            "CPU experts on: %s trait, cores %s, %d threads, split %s",
            trait.name, list(self.cores), self.threads, self.split,
        )

    def _capacity(self, slabs: Mapping[str, torch.Tensor]) -> int:
        rows = {int(slabs[name].shape[0]) for name in self.trait.slab_names}
        if len(rows) != 1:
            raise ValueError(f"the pinned slabs disagree on their row count {rows}")
        return rows.pop()

    def registered(self, row: int) -> bool:
        return row in self.handles

    def attach_device(self, device_side) -> None:
        """The chain's device side, which types CPU lanes only for registered rows: told of every row, now and later."""
        self.device_side = device_side
        for row in self.handles:
            device_side.set_row_cpu(row)

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
        if getattr(self, "device_side", None) is not None:
            self.device_side.set_row_cpu(row)

    def log_stats(self) -> dict[str, int]:
        """Log the CPU expert thread's cumulative counters: an A/B arm reads its per-expert cost under load here."""
        stats = {key: value - self._calibration_stats[key] for key, value in self.host.cpu_stats().items()}
        lanes = stats["lanes"]
        logger.info(
            "CPU experts stats: %d jobs, %d lanes, %.3f ms per lane, split %s",
            stats["jobs"], lanes, stats["forward_ns"] / lanes / 1e6 if lanes else 0.0, self.split,
        )
        return stats

    def retune(self) -> Optional[list[int]]:
        """P3: recompute the split from the CPU's measured per-expert cost since the last call, when it has run.

        The link and handoff costs stay as configured. Returns the new table when it changed.
        """
        if self.calibrated:
            return None
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

    def calibrate(self, device: int) -> Optional[list[int]]:
        """Measure the split once on the loaded model and keep it (spec 2026-10-01-cpu-split-calibration).

        The caller owns the tier (the RAM thread paused) and has not armed the copy engine. Returns the new split, or
        None when calibration is off, the split is fixed by SGLANG_DSV41_CPU_EXPERTS_SPLIT, or it could not run (a
        warning says why, and the split stays as it was).
        """
        if envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.get() or not envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.get():
            return None
        row = next((r for r in sorted(self.handles) if self._capacity(self.slabs_by_row[r]) >= LANES), None)
        if row is None:
            logger.warning(
                "CPU experts calibration skipped: no registered row has %d RAM slots; keeping split %s",
                LANES,
                self.split,
            )
            return None
        reps = envs.SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS.get()
        before = self.host.cpu_stats()
        scratch = None
        try:
            expert_bytes = self.host.copy_expert_bytes(row)
            scratch = torch.empty(
                LANES * expert_bytes, dtype=torch.uint8, device="cpu" if device < 0 else torch.device("cuda", device)
            )
            grid = self.host.calibrate_cpu_split(row, device=device, reps=reps, scratch=scratch).tolist()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            # A timed-out DMA may still write into the scratch: never hand its block back to the allocator.
            self._calibration_scratch = scratch
            self._exclude_calibration_stats(before)
            logger.warning("CPU experts calibration failed (%s); keeping split %s", error, self.split)
            return None
        self._exclude_calibration_stats(before)
        split = split_from_grid(grid)
        self.split, self.calibrated = split, True
        self.host.set_cpu_split(split)
        report = format_calibration(grid, split, row=row, expert_bytes=expert_bytes, reps=reps)
        print(report, flush=True)
        logger.info("%s", report)
        logger.debug("CPU experts calibration grid, ms (row 1 + n is both[n][k]): %s", grid)
        return split

    def _exclude_calibration_stats(self, before: dict[str, int]) -> None:
        """Calibration's own CPU jobs are not decode's: log_stats subtracts them and retune starts after them."""
        after = self.host.cpu_stats()
        self._calibration_stats = {key: after[key] - before[key] for key in after}
        self._last_stats = after


def cpu_expert_cores() -> tuple[list[int], int]:
    """SGLANG_DSV41_CPU_EXPERTS_CORES and _THREADS, checked."""
    spec = envs.SGLANG_DSV41_CPU_EXPERTS_CORES.get()
    if not spec:
        raise ValueError("SGLANG_DSV41_CPU_EXPERTS needs SGLANG_DSV41_CPU_EXPERTS_CORES (a taskset list, e.g. 18-29)")
    cores = parse_core_list(spec)
    threads = envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.get() or len(cores)
    return cores, threads
