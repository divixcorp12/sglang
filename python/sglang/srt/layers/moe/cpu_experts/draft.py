"""The DSpark draft's non-resident routed experts on the CPU, in the decode graph.

With ``SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`` each draft stage's ``FusedMoE`` loads its experts into host RAM and
keeps only its resident set (``draft_resident.py``) and any fused shared expert on the GPU (``DraftResidentMoe``). The
rest run on node 0's CPU expert engine (host/cpu_experts.h, one team per node), or, with the target's CPU experts off,
on a draft-only engine, over the draft channel (``dspark_draft_cpu.py``): the
stage's call posts its CPU share (device-only), runs its GPU share, then its finish waits on the channel's gate and
adds the CPU rows, all on the stream, so a CUDA graph captures the whole call. The stages register here as they finish
loading; ``prepare()`` builds the runtime before capture.
"""

import atexit
import logging
import threading
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
    """A format's CPU expert kernel as the draft thread takes it: the kernel's address and a stage's layer spec."""

    def __init__(self, trait):
        self.trait = trait

    def address(self) -> int:
        return self.trait.kernel_address()

    def spec(self, slabs: Mapping[str, torch.Tensor], capacity: int):
        return self.trait.layer_spec(slabs, capacity)


def draft_kernel_for(format_key: str, act_limit: Optional[float]) -> DraftKernel:
    """The draft's kernel for ``format_key``, with the stages' SwiGLU clamp."""
    trait = cpu_trait_for(format_key)
    trait.check_environment()
    trait.act_limit = act_limit
    return DraftKernel(trait)


def cpu_slots(on_cpu: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``ids`` with every route the CPU does not compute (resident, fused shared, -1, out of range) set to -1: the
    mask draft_post_kernel applies on the device, as a host reference."""
    valid = (ids >= 0) & (ids < len(on_cpu))
    keep = valid & on_cpu[ids.clamp(0, len(on_cpu) - 1)]
    return torch.where(keep, ids, torch.full_like(ids, -1))


def _ram_miss_service():
    from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService

    return Exl3RamMissService.get()


def _new_host(areas, **kw):
    from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuHost

    return DraftCpuHost(areas, **kw)


def _new_device(areas, on_cpu: torch.Tensor, device):
    from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuDevice

    return DraftCpuDevice(areas, on_cpu, device)


class DraftCpuExperts:
    """The draft stages' CPU share: the draft channel's pinned areas, its device half and the draft CPU thread."""

    def __init__(
        self,
        kernel,
        layers: Mapping[int, DraftLayer],
        *,
        cores: Sequence[int],
        threads: int,
        device=None,
    ):
        from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas
        from sglang.srt.layers.moe.exl3_ram_miss import watchdog_wait_s

        self.kernel = kernel
        self.cores = list(cores)
        self.threads = threads
        keys = sorted(layers)
        if keys != list(range(len(keys))):
            raise ValueError(f"DSpark draft stage keys must be 0..n-1, not {keys}")
        # Every slab holds one row per expert of the stage, so the first slab's rows are the layer's capacity.
        self.capacity = {key: int(next(iter(layer.slabs.values())).shape[0]) for key, layer in layers.items()}
        self.on_cpu = {key: layer.on_cpu for key, layer in layers.items()}
        hidden = {int(layer.slabs["w13_suh"].shape[-1]) for layer in layers.values()}
        if len(hidden) != 1:
            raise ValueError(f"DSpark draft stages disagree on the hidden size: {sorted(hidden)}")
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        self.areas = DraftCpuAreas(len(keys), hidden.pop(), pin=torch.cuda.is_available())
        fatal_wait_s = watchdog_wait_s(envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get())
        if envs.SGLANG_DSV41_CPU_EXPERTS.get():
            # One team per node: node 0's CPU expert engine serves the draft channel too.
            self.host = _ram_miss_service().draft_host(self.areas, fatal_wait_s=fatal_wait_s)
        else:
            self.host = _new_host(
                self.areas,
                cores=self.cores,
                threads=threads,
                spin_us=envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US.get(),
                keep_warm_us=envs.SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US.get(),
                fatal_wait_s=fatal_wait_s,
            )
        try:
            for key in keys:
                self.host.set_layer(key, kernel.address(), kernel.spec(layers[key].slabs, self.capacity[key]))
            self.host.start()
        except BaseException:
            self.host.stop()
            raise
        on_cpu = torch.stack([layers[key].on_cpu.to(torch.uint8) for key in keys])
        self.device_half = _new_device(self.areas, on_cpu, device) if device is not None else None

    def cpu_slots(self, key: int, ids: torch.Tensor) -> torch.Tensor:
        return cpu_slots(self.on_cpu[key], ids)

    def post(self, key: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Stage ``key``'s CPU share of one call (x [M, H], M <= 16); device-only."""
        self.device_half.post(key, x, topk_ids, topk_weights)

    def finish(self, key: int, out: torch.Tensor) -> None:
        """Wait (on the stream) for the posted share and add it into ``out`` [M, H] fp32; device-only."""
        self.device_half.finish(key, out)

    def stats(self) -> dict:
        return self.host.stats()

    def close(self) -> None:
        self.host.stop()


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

    def prepare(self) -> None:
        """Build and start the runtime (host work: the pinned areas, the thread, the device module); idempotent. Call
        before capturing the draft: runtime() refuses to build inside a capture."""
        self.runtime()

    def runtime(self) -> DraftCpuExperts:
        with self._lock:
            if self._runtime is None:
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("prepare the DSpark draft before capture (DRAFT_CPU_EXPERTS.prepare())")
                if not self._layers:
                    raise RuntimeError("no DSpark draft stage registered for CPU experts")
                path = envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get()
                extra = sorted(set(load_resident_set(path)) - self._stages) if path else []
                if extra:
                    raise ValueError(
                        f"DSpark draft resident set {path} lists stages {extra} that this draft does not have "
                        f"(it has {sorted(self._stages)}); recalibrate it from this draft's routes"
                    )
                if envs.SGLANG_DSV41_CPU_EXPERTS.get():
                    # Node 0's CPU expert team runs the draft: no cores of its own (the gate refuses named ones).
                    cores, threads = [], 0
                else:
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
                    "DSpark CPU experts: %d draft stages on %s; %s experts on the CPU",
                    len(self._layers),
                    f"cores {cores}, {threads} threads" if cores else "node 0's CPU expert team",
                    [int(layer.on_cpu.sum()) for layer in self._layers.values()],
                )
            return self._runtime

    def close(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            logger.info("DSpark CPU experts: %s", runtime.stats())
            runtime.close()


DRAFT_CPU_EXPERTS = DraftCpuExpertsRegistry()
