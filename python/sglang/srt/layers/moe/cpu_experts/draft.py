"""The DSpark draft's non-resident routed experts on the CPU (eager).

With ``SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`` each draft stage's ``FusedMoE`` loads its experts into host RAM and
keeps only its resident set (``draft_resident.py``) and any fused shared expert on the GPU. This module computes the
rest with the format's CPU expert kernel, through the host module's ``kernel_layer`` / ``kernel_forward``
(``expert_stream_transport.py``), whose forward takes every row of a stage call at once. The stages register here as
they finish loading; the runtime is built on first use.

One worker thread runs the kernel; ``kernel_forward`` pins it and its workers to the draft's cores, so the caller (the
scheduler) never is. The forward releases the GIL, so the caller runs the resident experts on the GPU meanwhile.
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
from sglang.srt.layers.moe.cpu_experts.draft_resident import load_resident_set
from sglang.srt.layers.moe.cpu_experts.service import cpu_trait_for
from sglang.srt.layers.moe.cpu_experts.threading_config import (
    ThreadingConfig,
    check_engine_cores,
    check_not_reserved,
    parse_cpu_list,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftLayer:
    """One draft stage: its expert tensors in the pinned tier's slab layout, which experts the CPU computes, and
    the SwiGLU clamp."""

    slabs: Mapping[str, torch.Tensor]
    on_cpu: torch.Tensor
    act_limit: Optional[float]


class DraftKernel:
    """A format's CPU expert kernel as the draft calls it: ``make`` a layer over a stage's slabs, ``forward`` m rows,
    ``drop`` the layer. Production goes through the host module's ``kernel_*`` exports."""

    def __init__(self, trait):
        self.trait = trait

    def make(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> int:
        from sglang.kernels.ops.moe import expert_stream_transport as es

        return es.kernel_layer(self.trait.kernel_address(), self.trait.layer_spec(slabs, capacity))

    def forward(self, layer, x16, slots, weights, out, threads: int, cores: Sequence[int]) -> None:
        from sglang.kernels.ops.moe import expert_stream_transport as es

        status, why = es.kernel_forward(layer, x16, slots, weights, out, threads=threads, cores=cores)
        if status:
            raise RuntimeError(f"DSpark draft CPU experts: {self.trait.name} kernel refused a forward: {why}")

    def drop(self, layer) -> None:
        from sglang.kernels.ops.moe import expert_stream_transport as es

        es.kernel_drop(layer)


def draft_kernel_for(format_key: str, act_limit: Optional[float]) -> DraftKernel:
    """The draft's kernel for ``format_key``, with the stages' SwiGLU clamp."""
    trait = cpu_trait_for(format_key)
    trait.check_environment()
    trait.act_limit = act_limit
    return DraftKernel(trait)


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
    """The draft stages' kernel layers and the worker thread that runs their forwards."""

    def __init__(
        self,
        kernel,
        layers: Mapping[int, DraftLayer],
        *,
        cores: Sequence[int],
        threads: int,
        log_every: int = 300,
    ):
        self.kernel = kernel
        self.cores = list(cores)
        self.threads = threads
        # Every slab holds one row per expert of the stage, so the first slab's rows are the layer's capacity.
        self.capacity = {key: int(next(iter(layer.slabs.values())).shape[0]) for key, layer in layers.items()}
        self._layers = {key: kernel.make(layer.slabs, self.capacity[key]) for key, layer in layers.items()}
        self.on_cpu = {key: layer.on_cpu for key, layer in layers.items()}
        self.stats = DraftCpuStats(log_every)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dspark-cpu-experts")

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
        # kernel_forward's dtypes: x fp16, slots int32, weights fp32.
        x16 = x.to(torch.float16).cpu().contiguous()
        w32 = weights.to(torch.float32).cpu().contiguous()
        return self._worker.submit(self._run, key, x16, slots, w32)

    def _run(self, key: int, x16: torch.Tensor, slots: torch.Tensor, w32: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x16.shape, dtype=torch.float32)
        start = time.perf_counter()
        self.kernel.forward(
            self._layers[key], x16, slots.to(torch.int32).contiguous(), w32, out, self.threads, self.cores
        )
        self.stats.record(slots, time.perf_counter() - start)
        return out

    def close(self) -> None:
        self._worker.shutdown(wait=True)
        layers, self._layers = self._layers, {}
        for layer in layers.values():
            self.kernel.drop(layer)


class DraftCpuExpertsRegistry:
    """Draft stages registered at load, and the runtime built from them on first use."""

    def __init__(self):
        self._layers: dict[int, DraftLayer] = {}
        self._stages: set[int] = set()
        self._runtime: Optional[DraftCpuExperts] = None
        self._lock = threading.Lock()

    def register(
        self,
        slabs: Mapping[str, torch.Tensor],
        on_cpu: torch.Tensor,
        act_limit: Optional[float],
        *,
        layer_id: int,
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
            self._stages.add(layer_id)
            self._layers[key] = DraftLayer(slabs, on_cpu, act_limit)
            return key

    def runtime(self) -> DraftCpuExperts:
        with self._lock:
            if self._runtime is None:
                if not self._layers:
                    raise RuntimeError("no DSpark draft stage registered for CPU experts")
                path = envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get()
                extra = sorted(set(load_resident_set(path)) - self._stages) if path else []
                if extra:
                    raise ValueError(
                        f"DSpark draft resident set {path} lists stages {extra} that this draft does not have "
                        f"(it has {sorted(self._stages)}); recalibrate it from this draft's routes"
                    )
                named = envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get()
                if named:
                    cores = parse_cpu_list(named)
                    for core in cores:
                        check_not_reserved(core)
                    threads = envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.get() or len(cores)
                else:
                    # The plan the RAM-miss service resolves too, so the draft and its threads never share a core.
                    cores = list(
                        ThreadingConfig.from_env(
                            cpu_experts=envs.SGLANG_DSV41_CPU_EXPERTS.get(),
                            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
                        ).draft_cpus
                    )
                    threads = len(cores)
                check_engine_cores(cores, threads)
                kernel = draft_kernel_for("exl3", next(iter(self._layers.values())).act_limit)
                self._runtime = DraftCpuExperts(kernel, self._layers, cores=cores, threads=threads)
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
