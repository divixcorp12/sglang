"""CPU experts inside the EXL3 RAM-miss service: the Python half.

The service's own threads run the kernel. For each captured post, the post kernel types
the last ``split[n]`` of its ``n`` eligible lanes as CPU lanes: RAM hits, and NVMe
misses when ``SGLANG_DSV41_CPU_EXPERTS_MISSES`` is set. The device plan sorts the
lanes, so those are the lowest-scored. The service thread hands them to the CPU expert
thread (``expert_stream/host/cpu_experts.h``): hits at once into part 0 of the row's
output, each miss into part 1 as soon as its read lands. The copy wait releases the
decode stream once the copies and the CPU are both done.

This module owns the quant trait lookup, the pinned rows the post kernel and the CPU
exchange, the lazy per-layer registration, the split table (configured, re-derived from
measured CPU cost, or calibrated at startup), and the CPU stats logging. The split
arithmetic is in ``cpu_experts/policy.py``.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.policy import (
    format_calibration,
    split_from_grid,
    split_table,
)

logger = logging.getLogger(__name__)


def cpu_trait_for(format_key: str, ext=None):
    """The quant trait of a streamed expert format; only EXL3 has a CPU kernel today."""
    if format_key == "exl3":
        from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
        from sglang.srt.layers.quantization.exl3.ext import exl3_ext

        return Exl3CpuQuantTrait(ext if ext is not None else exl3_ext(), act_limit=None)
    raise ValueError(f"CPU experts have no kernel for expert format {format_key!r}")


def configured_split(lanes: int) -> list[int]:
    """The split table for a ``lanes``-lane build: ``SGLANG_DSV41_CPU_EXPERTS_SPLIT`` when set, else ``k_star(n)``.

    The fallback uses the configured per-expert CPU, link and handoff costs. Raises
    ``ValueError`` for a malformed explicit split.
    """
    spec = envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.get()
    if spec:
        split = [int(v) for v in spec.split(",")]
        if len(split) != lanes + 1 or any(not 0 <= k <= n for n, k in enumerate(split)):
            raise ValueError(
                f"SGLANG_DSV41_CPU_EXPERTS_SPLIT must list {lanes + 1} counts with 0 <= split[n] <= n, got {spec!r}"
            )
        return split
    return split_table(
        lanes,
        envs.SGLANG_DSV41_CPU_EXPERTS_CPU_MS.get(),
        envs.SGLANG_DSV41_CPU_EXPERTS_LINK_MS.get(),
        envs.SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS.get(),
    ).tolist()


class CpuExpertService:
    """The Python half of CPU experts for one ``ExpertStreamHost``.

    Created by the RAM-miss service at start-up, after the copy engine is enabled and
    before the service thread starts. It owns the pinned input and output rows that the
    post kernel and the native CPU expert thread exchange: fp16 inputs padded to 16
    bytes (the post's vector stores) and fp32 outputs, both read and written by the
    device through UVA. Each output row has two parts, the CPU hits' partial sum
    (part 0) and the CPU misses' (part 1), which the route tables sum.

    Layers register lazily with ``register``, on a layer's first graph-path forward (a
    warm-up before capture); the device side learns of each registered row through
    ``attach_device``. ``split[n]`` is the number of CPU lanes for ``n`` eligible lanes.
    It starts as configured and can be replaced by ``retune`` (measured CPU cost) or
    ``calibrate`` (startup measurement); a calibrated split is final, so ``retune`` then
    does nothing. Calibration's own CPU jobs are subtracted from ``log_stats`` and from
    ``retune``'s baseline.

    At several NUMA nodes there is one service per group (``CpuExpertGroups``): each runs
    its own engine on its node's cores and writes its own two output parts of a row. A
    service made with ``shared`` uses that service's pinned rows and layer specs.
    """

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
        group: int = 0,
        shared: Optional["CpuExpertService"] = None,
    ):
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_engine_cores

        cores = sorted(set(cores))
        check_engine_cores(cores, threads)
        trait.check_environment()
        self.host, self.trait = host, trait
        self.group = group
        # The split table covers n = 0..lanes resident lanes, as the host's wire does.
        self.lanes = host.wire.lanes
        self.slabs_by_row = dict(slabs_by_row)
        self.hidden, self.cores, self.threads = int(hidden), tuple(cores), int(threads)
        self.split = list(split)
        self.calibrated = False
        self._calibration_stats = {"jobs": 0, "lanes": 0, "forward_ns": 0}
        self._calibration_scratch: Optional[torch.Tensor] = None
        rows = len(self.slabs_by_row)
        if sorted(self.slabs_by_row) != list(range(rows)):
            raise ValueError(
                "CPU experts need the slabs of every service row 0..rows-1"
            )
        if shared is not None:
            self.x_rows, self.out_rows, self.layers = shared.x_rows, shared.out_rows, shared.layers
        else:
            # Row layout is in the class doc; pinned so the device reaches them by UVA.
            x_bytes = -(-2 * self.hidden // 16) * 16
            self.x_rows = torch.zeros((rows, x_bytes), dtype=torch.uint8)
            self.out_rows = torch.zeros((rows, 2 * host.nodes, self.hidden), dtype=torch.float32)
            if pin:
                self.x_rows, self.out_rows = (
                    self.x_rows.pin_memory(),
                    self.out_rows.pin_memory(),
                )
            self.layers: dict[int, object] = {}
        keep_warm_us = envs.SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US.get()
        host.enable_cpu_experts(
            trait.kernel_address(),
            self.split,
            self.cores,
            self.x_rows,
            self.out_rows,
            threads=self.threads,
            group=self.group,
            keep_warm_us=max(keep_warm_us, 0),
        )
        self._last_stats = host.cpu_stats(self.group)
        logger.info(
            "CPU experts group %d on: %s trait, cores %s, %d threads, split %s",
            self.group,
            trait.name,
            list(self.cores),
            self.threads,
            self.split,
        )

    def _capacity(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The slab row count of one layer, which every slab must agree on."""
        rows = {int(slabs[name].shape[0]) for name in self.trait.slab_names}
        if len(rows) != 1:
            raise ValueError(f"the pinned slabs disagree on their row count {rows}")
        return rows.pop()

    def _group_slots(self, row: int) -> int:
        """The slots of ``row`` this service's NUMA group holds: the row's slab rows at one group."""
        if self.host.nodes == 1:
            return self._capacity(self.slabs_by_row[row])
        lo, hi = self.host.node_ranges[self.group][row]
        return hi - lo

    def registered(self, row: int) -> bool:
        """Whether ``row`` has been registered with the kernel."""
        return row in self.layers

    def attach_device(self, device_side) -> None:
        """Attach the device side, which types CPU lanes for registered rows only.

        It is told of every row registered now and of each registered later.
        """
        self.device_side = device_side
        for row in self.layers:
            device_side.set_row_cpu(row)

    def register(self, row: int, act_limit: Optional[float]) -> None:
        """Register ``row``'s pinned slabs with the trait and enable its CPU lanes.

        Called on a layer's first graph-path forward, a warm-up before capture: the
        activation limit is the layer's MoE runner config, which the service does not
        see at start-up. Every layer must use the same limit. A no-op once registered.
        """
        if row in self.layers:
            return
        if self.trait.act_limit is None:
            self.trait.act_limit = act_limit
        elif self.trait.act_limit != act_limit:
            raise ValueError(
                f"CPU experts: layer rows use activation limits {self.trait.act_limit} and {act_limit}"
            )
        slabs = self.slabs_by_row[row]
        capacity = self._capacity(slabs)
        if capacity == 0:
            return
        spec = self.trait.layer_spec({name: slabs[name] for name in self.trait.slab_names}, capacity)
        self.layers[row] = spec
        self.host.set_cpu_layer(row, spec)
        if getattr(self, "device_side", None) is not None:
            self.device_side.set_row_cpu(row)

    def log_stats(self) -> dict[str, int]:
        """Log the CPU expert thread's cumulative counters, minus calibration's jobs.

        This is where per-expert CPU cost under load is read. Returns the counters.
        """
        stats = {
            key: value - self._calibration_stats[key]
            for key, value in self.host.cpu_stats(self.group).items()
        }
        lanes = stats["lanes"]
        logger.info(
            "CPU experts group %d stats: %d jobs, %d lanes, %.3f ms per lane, split %s",
            self.group,
            stats["jobs"],
            lanes,
            stats["forward_ns"] / lanes / 1e6 if lanes else 0.0,
            self.split,
        )
        return stats

    def retune(self) -> Optional[list[int]]:
        """Recompute the split from the CPU's measured per-expert cost since last call.

        Needs at least 64 CPU lanes since the last call, and does nothing after a
        calibration. The link and handoff costs stay as configured. Returns the new
        table when it changed, else ``None``.
        """
        if self.calibrated:
            return None
        stats = self.host.cpu_stats(self.group)
        lanes = stats["lanes"] - self._last_stats["lanes"]
        ns = stats["forward_ns"] - self._last_stats["forward_ns"]
        if lanes < 64:
            return None
        self._last_stats = stats
        c_cpu = ns / lanes / 1e6
        split = split_table(
            self.lanes,
            c_cpu,
            envs.SGLANG_DSV41_CPU_EXPERTS_LINK_MS.get(),
            envs.SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS.get(),
        ).tolist()
        if split == self.split:
            return None
        self.split = split
        self.host.set_cpu_split(split, group=self.group)
        logger.info(
            "CPU experts group %d: measured %.3f ms per expert, split now %s",
            self.group,
            c_cpu,
            split,
        )
        return split

    def calibrate(self, device: int) -> Optional[list[int]]:
        """Measure the split once on the loaded model and keep it.

        The caller owns the tier (the RAM thread paused) and has not armed the copy
        engine. Returns the new split, or ``None`` when calibration is off, the split
        is fixed by ``SGLANG_DSV41_CPU_EXPERTS_SPLIT``, or it could not run (a warning
        says why and the split stays as it was). The native measurement is
        ``expert_stream/host/split_calibration.h``.
        """
        if (
            envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.get()
            or not envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.get()
        ):
            return None
        row = calibration_row({r: self._group_slots(r) for r in self.layers}, self.lanes)
        if row is None:
            logger.warning(
                "CPU experts group %d calibration skipped: no registered row has %d RAM slots; keeping split %s",
                self.group,
                self.lanes,
                self.split,
            )
            return None
        reps = envs.SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS.get()
        before = self.host.cpu_stats(self.group)
        scratch = None
        try:
            expert_bytes = self.host.copy_expert_bytes(row)
            scratch = torch.empty(
                self.lanes * expert_bytes,
                dtype=torch.uint8,
                device="cpu" if device < 0 else torch.device("cuda", device),
            )
            grid = self.host.calibrate_cpu_split(
                row, device=device, reps=reps, scratch=scratch, group=self.group
            ).tolist()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            # A timed-out DMA may still write into the scratch: keep its block out of
            # the allocator.
            self._calibration_scratch = scratch
            self._exclude_calibration_stats(before)
            logger.warning(
                "CPU experts group %d calibration failed (%s); keeping split %s",
                self.group,
                error,
                self.split,
            )
            return None
        self._exclude_calibration_stats(before)
        split = split_from_grid(grid)
        self.split, self.calibrated = split, True
        self.host.set_cpu_split(split, group=self.group)
        report = format_calibration(
            grid, split, row=row, expert_bytes=expert_bytes, reps=reps
        )
        print(report, flush=True)
        logger.info("%s", report)
        logger.debug(
            "CPU experts calibration grid, ms (row 1 + n is both[n][k]): %s", grid
        )
        return split

    def _exclude_calibration_stats(self, before: dict[str, int]) -> None:
        """Account calibration's own CPU jobs apart from decode's.

        ``log_stats`` subtracts them and ``retune`` starts counting after them.
        """
        after = self.host.cpu_stats(self.group)
        self._calibration_stats = {key: after[key] - before[key] for key in after}
        self._last_stats = after


def calibration_row(capacities: Mapping[int, int], lanes: int) -> Optional[int]:
    """The first row calibration can run on, or ``None`` to keep the configured split.

    Calibration needs ``lanes`` RAM slots in one row and one expert of scratch per
    lane, and its grid grows with ``lanes``; a tier whose rows all hold fewer slots
    skips it rather than failing the launch.
    """
    return next((r for r in sorted(capacities) if capacities[r] >= lanes), None)


class CpuExpertGroups:
    """One CpuExpertService per NUMA group of the host, each on its node's cores (spec 2026-10-03, Part 3).

    The services share the pinned rows (each group writes its own two output parts of a row) and the kernel's layer
    registrations: one layer addresses a whole slab, so the host shares each row's layer among the groups' engines. The
    split is configured, re-tuned and calibrated per group.
    """

    def __init__(self, host, trait, slabs_by_row, *, hidden: int, plans, split: Sequence[int], pin: bool = True):
        self.services: list[CpuExpertService] = []
        for plan in plans:
            self.services.append(
                CpuExpertService(
                    host,
                    trait,
                    slabs_by_row,
                    hidden=hidden,
                    cores=plan.cpu,
                    threads=plan.threads,
                    split=split,
                    pin=pin,
                    group=plan.group,
                    shared=self.services[0] if self.services else None,
                )
            )
        self.x_rows, self.out_rows = self.services[0].x_rows, self.services[0].out_rows

    def registered(self, row: int) -> bool:
        return self.services[0].registered(row)

    def register(self, row: int, act_limit: Optional[float]) -> None:
        """Register ``row`` once; the host shares its layer among every group's engine."""
        self.services[0].register(row, act_limit)

    def attach_device(self, device_side) -> None:
        self.services[0].attach_device(device_side)

    def retune(self) -> list[Optional[list[int]]]:
        return [service.retune() for service in self.services]

    def log_stats(self) -> list[dict[str, int]]:
        return [service.log_stats() for service in self.services]

    def calibrate(self, device: int) -> list[Optional[list[int]]]:
        return [service.calibrate(device) for service in self.services]
