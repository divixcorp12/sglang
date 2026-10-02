"""The DSpark draft's non-resident routed experts on the CPU (eager).

With ``SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`` each draft stage's ``FusedMoE`` loads its experts into host RAM and
keeps only its resident set (``draft_resident.py``) and any fused shared expert on the GPU. This module computes the
rest with the CPU expert kernel. The stages register here as they finish loading; the pool is built on first use.

One worker thread runs the kernel, bound to the pool's cores, so the caller (the scheduler) never is. The kernel's
op releases the GIL, so the caller runs the resident experts on the GPU meanwhile.
"""

import atexit
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.policy import parse_core_list
from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertPool
from sglang.srt.layers.moe.cpu_experts.service import cpu_trait_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftLayer:
    """One draft stage: its expert tensors in the pinned tier's slab layout, which experts the CPU computes, and
    the SwiGLU clamp."""

    slabs: Mapping[str, torch.Tensor]
    on_cpu: torch.Tensor
    act_limit: Optional[float]


class DraftCpuStats:
    """Cost and routing of the draft's CPU share, logged every ``log_every`` stage calls.

    ``passes`` counts weight reads: the kernel reads an expert once per two rows routed to it. ``skips`` counts
    stage calls whose routes were all resident.
    """

    def __init__(self, log_every: int):
        self.log_every = log_every
        self.calls = 0
        self.skips = 0
        self.seconds: list[float] = []
        self.unions: list[int] = []
        self.passes: list[int] = []

    def record(self, slots: torch.Tensor, seconds: float) -> None:
        used = torch.bincount(slots[slots >= 0])
        used = used[used > 0]
        self.unions.append(int(used.numel()))
        self.passes.append(int(((used + 1) // 2).sum()))
        self.seconds.append(seconds)
        self._count()

    def record_skip(self) -> None:
        self.skips += 1
        self._count()

    def _count(self) -> None:
        self.calls += 1
        if self.calls % self.log_every == 0:
            logger.info(self.summary())
            self.seconds, self.unions, self.passes = [], [], []

    def summary(self) -> str:
        head = f"DSpark CPU experts: {self.calls} stage calls, {self.skips} without CPU work"
        if not self.seconds:
            return head
        ms = sorted(s * 1e3 for s in self.seconds)
        window = len(ms)
        return (
            f"{head}; last {window} CPU calls: {sum(ms) / window:.2f} ms mean, "
            f"{ms[int(0.9 * (window - 1))]:.2f} ms p90, union {sum(self.unions) / window:.1f}, "
            f"weight passes {sum(self.passes) / window:.1f}"
        )


class DraftCpuExperts:
    """The draft stages' CPU expert pool and the worker thread that runs it."""

    def __init__(
        self,
        trait,
        layers: Mapping[int, DraftLayer],
        *,
        cores: Sequence[int],
        threads: int,
        log_every: int = 300,
    ):
        self.pool = CpuExpertPool(
            trait, {key: layer.slabs for key, layer in layers.items()}, cores=cores, threads=threads
        )
        self.on_cpu = {key: layer.on_cpu for key, layer in layers.items()}
        self.stats = DraftCpuStats(log_every)
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dspark-cpu-experts",
            initializer=self.pool.bind_current_thread,
        )

    def cpu_slots(self, key: int, ids: torch.Tensor) -> torch.Tensor:
        on_cpu = self.on_cpu[key]
        valid = (ids >= 0) & (ids < len(on_cpu))
        keep = valid & on_cpu[ids.clamp(0, len(on_cpu) - 1)]
        return torch.where(keep, ids, torch.full_like(ids, -1))

    def submit(
        self, key: int, ids: torch.Tensor, x: torch.Tensor, weights: torch.Tensor
    ) -> Optional[Future]:
        """Start stage ``key``'s CPU routes over ``x`` ``[m, H]``; None when every route is resident."""
        slots = self.cpu_slots(key, ids)
        if not bool((slots >= 0).any()):
            self.stats.record_skip()
            return None
        x16 = x.to(torch.float16).cpu()
        w16 = weights.to(torch.float16).cpu()
        return self._worker.submit(self._run, key, x16, slots, w16)

    def _run(self, key: int, x16: torch.Tensor, slots: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x16.shape, dtype=torch.float32)
        start = time.perf_counter()
        self.pool.compute_rows(key, slots, w16, x16, out)
        self.stats.record(slots, time.perf_counter() - start)
        return out

    def close(self) -> None:
        self._worker.shutdown(wait=True)
        self.pool.close()


class DraftCpuExpertsRegistry:
    """Draft stages registered at load, and the runtime built from them on first use."""

    def __init__(self):
        self._layers: dict[int, DraftLayer] = {}
        self._runtime: Optional[DraftCpuExperts] = None
        self._lock = threading.Lock()

    def register(
        self, slabs: Mapping[str, torch.Tensor], on_cpu: torch.Tensor, act_limit: Optional[float]
    ) -> int:
        """Add one stage; returns its key. Every stage must share one activation limit (one kernel trait)."""
        with self._lock:
            if self._runtime is not None:
                raise RuntimeError("a DSpark draft stage registered after the CPU experts started")
            limits = {layer.act_limit for layer in self._layers.values()}
            if limits and act_limit not in limits:
                raise ValueError(
                    f"DSpark draft stages disagree on the activation limit: {sorted(limits)} and {act_limit}"
                )
            key = len(self._layers)
            self._layers[key] = DraftLayer(slabs, on_cpu, act_limit)
            return key

    def runtime(self) -> DraftCpuExperts:
        with self._lock:
            if self._runtime is None:
                if not self._layers:
                    raise RuntimeError("no DSpark draft stage registered for CPU experts")
                cores = parse_core_list(envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get())
                threads = envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.get() or len(cores)
                trait = cpu_trait_for("exl3")
                trait.act_limit = next(iter(self._layers.values())).act_limit
                self._runtime = DraftCpuExperts(trait, self._layers, cores=cores, threads=threads)
                atexit.register(self.close)
                logger.info(
                    "DSpark CPU experts: %d draft stages on cores %s, %d threads; %s experts on the CPU",
                    len(self._layers),
                    cores,
                    threads,
                    [int(layer.on_cpu.sum()) for layer in self._layers.values()],
                )
            return self._runtime

    def close(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            logger.info(runtime.stats.summary())
            runtime.close()


DRAFT_CPU_EXPERTS = DraftCpuExpertsRegistry()
