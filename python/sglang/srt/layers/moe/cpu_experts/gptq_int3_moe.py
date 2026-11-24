"""Experimental: the MiMo hybrid GPTQ INT3 checkpoint's routed experts on the CPU, without the expert engine.

Each MoE layer copies its routes' inputs to the host, runs them through the typed GPTQ INT3 kernel (one synchronous job
on a shared Team) and returns the routed sum to the device. Experts are read from the sealed export on demand through
``ExportedExperts`` (trusted manifest, per-payload SHA, owner and path checks), adapted losslessly into the kernel's
planes and kept in a bounded per-layer LRU of slots. Nothing is requantized; activations stay floating point.

This path is opt-in (``SGLANG_GPTQ_INT3_MANIFEST_SHA256``), serves tensor parallel size 1 only, and is not engine
serving: no GPU experts, graphs, prefetch or speculative draft.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.gptq_int3 import (
    SLAB_NAMES,
    ExportedExperts,
    GptqInt3CpuTrait,
)
from sglang.srt.layers.moe.cpu_experts.threading_config import parse_cpu_list

logger = logging.getLogger(__name__)

HYBRID_FORMAT = "mimo-hybrid-gptq-v1"
# The export's routed-expert block (config.json "experimental_quantization") in the reader's POLICY terms.
_DECLARED_POLICY = {"quant_method": "gptq", "bits": 3, "group_size": 128, "sym": True, "desc_act": False}
# A load summary is logged after this many expert reads; arbitrary.
_LOG_EVERY_LOADS = 2000
# A native-call summary is logged after this many calls: 60 decode steps of MiMo V2.6 Pro's 69 MoE layers; arbitrary.
_LOG_EVERY_CALLS = 4140
# The native job refuses a call whose float scratch exceeds 512 MiB (kernel.cpp, GptqInt3CpuKernel::prepare).
_SCRATCH_FLOATS = 512 * 1024 * 1024 // 4


def declares_gptq_int3_experts(config: Any) -> bool:
    """Whether a model config declares the hybrid GPTQ INT3 export (routed experts INT3, the rest unchanged)."""
    declared = getattr(config, "experimental_quantization", None)
    return isinstance(declared, dict) and declared.get("format") == HYBRID_FORMAT


def _slab_shapes(hidden: int, intermediate: int) -> dict[str, tuple[tuple[int, int], torch.dtype]]:
    return dict(
        zip(
            SLAB_NAMES,
            (
                ((2 * intermediate, hidden // 4), torch.uint8),
                ((2 * intermediate, hidden // 8), torch.uint8),
                ((2 * intermediate, hidden // 128), torch.float16),
                ((hidden, intermediate // 4), torch.uint8),
                ((hidden, intermediate // 8), torch.uint8),
                ((hidden, intermediate // 128), torch.float16),
            ),
        )
    )


class GptqInt3Runtime:
    """Process-wide state: the verified export reader, the native Team and the loader pool."""

    def __init__(
        self,
        *,
        root: str,
        manifest_sha256: str,
        threads: int,
        cores: list[int],
        load_threads: int,
        verify_threads: int,
        experts_per_layer: int,
    ):
        from sglang.kernels.ops.moe.gptq_int3_moe import gptq_int3_moe_type

        GptqInt3CpuTrait().check_environment()
        if experts_per_layer < 8:
            raise ValueError("SGLANG_GPTQ_INT3_EXPERTS_PER_LAYER must hold at least one token's 8 routes")
        if cores and len(cores) < threads:
            raise ValueError(f"SGLANG_GPTQ_INT3_CORES names {len(cores)} cores for {threads} threads")
        self.native = gptq_int3_moe_type()(threads, torch.tensor(cores[:threads], dtype=torch.int32, device="cpu"))
        self.reader = ExportedExperts(
            root, expected_manifest_sha256=manifest_sha256, repack=self.native.adapt
        )
        self.max_rows = int(self.native.max_rows())
        self.max_routes = int(self.native.max_routes())
        self.pool = ThreadPoolExecutor(max_workers=load_threads, thread_name_prefix="int3-load")
        self.experts_per_layer = experts_per_layer
        self.loads = 0
        self.load_seconds = 0.0
        self.forward_seconds = 0.0
        self.forward_calls = self.forward_rows = 0
        self._window_calls, self._window_seconds = 0, 0.0
        logger.info(
            "GPTQ INT3 CPU experts: export %s (manifest %s), %d threads, %d loaders, %d experts per layer",
            root, manifest_sha256, threads, load_threads, experts_per_layer,
        )
        if verify_threads > 0:
            threading.Thread(
                target=self._verify_all, args=(verify_threads,), name="int3-verify", daemon=True
            ).start()

    def _verify_all(self, threads: int) -> None:
        """Every expert file through ``load`` once, so later misses of each take ``load_into``'s fast path."""
        owners = [
            (layer, expert)
            for layer in range(1, 70)
            for expert in range(288)
            if f"experts/layer-{layer:02d}/expert-{expert:03d}/weights.safetensors" in self.reader.files
        ]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="int3-verify") as pool:
            # _verify_one returns nothing: a future that held load()'s planes would keep every expert (~300 GB).
            failures = sum(future.exception() is not None for future in [
                pool.submit(self._verify_one, layer, expert) for layer, expert in owners
            ])
        # A failure is not fatal here: that file's own miss goes through load() and raises there.
        logger.info(
            "GPTQ INT3 CPU experts: verified %d expert files in %.0f s (%d failed)",
            len(owners) - failures, time.perf_counter() - start, failures,
        )


    def _verify_one(self, layer: int, expert: int) -> None:
        self.reader.load(layer, expert)


    def record_loads(self, count: int, seconds: float) -> None:
        before = self.loads // _LOG_EVERY_LOADS
        self.loads += count
        self.load_seconds += seconds
        if self.loads // _LOG_EVERY_LOADS != before:
            logger.info(
                "GPTQ INT3 CPU experts: %d expert loads in %.1f s; %.1f s in native forwards",
                self.loads, self.load_seconds, self.forward_seconds,
            )


    def record_forward(self, rows: int, seconds: float) -> None:
        self.forward_seconds += seconds
        self.forward_calls += 1
        self.forward_rows += rows
        self._window_calls += 1
        self._window_seconds += seconds
        if self._window_calls == _LOG_EVERY_CALLS:
            logger.info(
                "GPTQ INT3 CPU experts: %d native calls, %d rows, %.1f s; last %d calls %.2f ms each",
                self.forward_calls, self.forward_rows, self.forward_seconds,
                self._window_calls, 1000 * self._window_seconds / self._window_calls,
            )
            self._window_calls, self._window_seconds = 0, 0.0



_RUNTIME: Optional[GptqInt3Runtime] = None


def gptq_int3_runtime(config: Any) -> GptqInt3Runtime:
    """The process's runtime for ``config``'s export, created on first use. Refuses a launch that has not opted in."""
    global _RUNTIME
    declared = config.experimental_quantization
    if {k: declared.get("routed_experts", {}).get(k) for k in _DECLARED_POLICY} != _DECLARED_POLICY:
        raise ValueError(f"{HYBRID_FORMAT}: routed experts must be {_DECLARED_POLICY}")
    sha = envs.SGLANG_GPTQ_INT3_MANIFEST_SHA256.get()
    if not sha:
        raise ValueError(
            f"the checkpoint declares {HYBRID_FORMAT} routed experts; set SGLANG_GPTQ_INT3_MANIFEST_SHA256 to the "
            "export's trusted manifest SHA256 to serve them on the CPU (experimental)"
        )
    if _RUNTIME is None:
        _RUNTIME = GptqInt3Runtime(
            root=config.name_or_path,
            manifest_sha256=sha,
            threads=envs.SGLANG_GPTQ_INT3_THREADS.get(),
            cores=parse_cpu_list(envs.SGLANG_GPTQ_INT3_CORES.get()),
            load_threads=envs.SGLANG_GPTQ_INT3_LOAD_THREADS.get(),
            verify_threads=envs.SGLANG_GPTQ_INT3_VERIFY_THREADS.get(),
            experts_per_layer=envs.SGLANG_GPTQ_INT3_EXPERTS_PER_LAYER.get(),
        )
    return _RUNTIME


class GptqInt3LayerCache:
    """One MoE layer's resident experts: ``capacity`` slots of the kernel's six slabs, evicted least recently used."""

    def __init__(
        self, *, runtime: GptqInt3Runtime, layer_id: int, num_experts: int, hidden: int, intermediate: int
    ):
        self.runtime, self.layer_id = runtime, layer_id
        self.capacity = min(runtime.experts_per_layer, num_experts)
        # torch.empty commits no pages: a slot costs memory once an expert is copied into it.
        self.slabs = {
            name: torch.empty((self.capacity, *shape), dtype=dtype, device="cpu")
            for name, (shape, dtype) in _slab_shapes(hidden, intermediate).items()
        }
        spec = GptqInt3CpuTrait(act_limit=0.0).layer_spec(self.slabs, self.capacity)
        self.hidden, self.intermediate = spec.hidden, spec.intermediate
        # The model is built under a CUDA default device; every host tensor names its device.
        self.slab_ptrs = torch.tensor([p for p, _ in spec.slabs], dtype=torch.int64, device="cpu")
        self.slab_bytes = torch.tensor([b for _, b in spec.slabs], dtype=torch.int64, device="cpu")
        self.lookup = torch.full((num_experts,), -1, dtype=torch.int32, device="cpu")
        self.slot_of: OrderedDict[int, int] = OrderedDict()
        self.free: list[int] = list(range(self.capacity))

    def ensure(self, experts: list[int]) -> None:
        """Make every expert in ``experts`` resident; ``lookup`` then maps each to its slot."""
        if len(experts) > self.capacity:
            raise ValueError(f"layer {self.layer_id}: {len(experts)} experts in one call exceed {self.capacity} slots")
        needed = set(experts)
        missing = [e for e in experts if e not in self.slot_of]
        for e in experts:
            if e in self.slot_of:
                self.slot_of.move_to_end(e)
        for e in missing:
            self._assign(e, needed)
        if not missing:
            return
        start = time.perf_counter()
        # list() waits for every load and re-raises the first failure.
        list(self.runtime.pool.map(self._load, missing))
        self.runtime.record_loads(len(missing), time.perf_counter() - start)

    def _load(self, expert: int) -> None:
        slot = self.slot_of[expert]
        self.runtime.reader.load_into(self.layer_id, expert, {name: self.slabs[name][slot] for name in SLAB_NAMES})

    def _assign(self, expert: int, needed: set[int]) -> None:
        if self.free:
            slot = self.free.pop()
        else:
            victim = next(e for e in self.slot_of if e not in needed)
            slot = self.slot_of.pop(victim)
            self.lookup[victim] = -1
        self.slot_of[expert] = slot
        self.lookup[expert] = slot


class GptqInt3CpuExperts(nn.Module):
    """Stands in for a MiMo MoE layer's ``FusedMoE``: ``forward(hidden_states, topk_output)`` returns the routed sum."""

    quant_method = None
    should_fuse_routed_scaling_factor_in_topk = False

    def __init__(
        self, *, config: Any, layer_id: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int
    ):
        super().__init__()
        self.runtime = gptq_int3_runtime(config)
        if top_k > self.runtime.max_routes:
            raise ValueError(f"GPTQ INT3 CPU experts take at most {self.runtime.max_routes} routes, not {top_k}")
        self.layer_id, self.top_k, self.hidden_size = layer_id, top_k, hidden_size
        self.cache = GptqInt3LayerCache(
            runtime=self.runtime,
            layer_id=layer_id,
            num_experts=num_experts,
            hidden=hidden_size,
            intermediate=intermediate_size,
        )
        # Rows per native call: every route of a call must be resident at once.
        self.rows_per_call = max(1, min(self.runtime.max_rows, self.cache.capacity // top_k))
        # Rows per call of a grouped batch, within the kernel's float scratch (x, result, and per route middle, down).
        per_row = 2 * hidden_size + top_k * (hidden_size + intermediate_size)
        self.rows_per_group_call = max(1, min(self.runtime.max_rows, _SCRATCH_FLOATS // per_row))

    def forward(self, hidden_states: torch.Tensor, topk_output: Any) -> torch.Tensor:
        rows = hidden_states.shape[0]
        if rows == 0:
            return torch.zeros_like(hidden_states)
        x = hidden_states.to(device="cpu", dtype=torch.float16).contiguous()
        ids = topk_output.topk_ids.to(device="cpu", dtype=torch.int64)
        weights = topk_output.topk_weights.to(device="cpu", dtype=torch.float32).contiguous()
        out = torch.empty((rows, self.hidden_size), dtype=torch.float32, device="cpu")
        if rows <= self.rows_per_call:
            self._forward_chunk(x, ids, weights, out)
        else:
            self._forward_grouped(x, ids, weights, out)
        return out.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _forward_chunk(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, out: torch.Tensor
    ) -> None:
        valid = ids >= 0
        self.cache.ensure(ids[valid].unique().tolist())
        slots = torch.where(valid, self.cache.lookup[ids.clamp(min=0)], -1).to(torch.int32).contiguous()
        start = time.perf_counter()
        self.runtime.native.forward(
            self.cache.hidden,
            self.cache.intermediate,
            self.cache.capacity,
            0.0,
            self.cache.slab_ptrs,
            self.cache.slab_bytes,
            x,
            slots,
            weights,
            out,
        )
        self.runtime.record_forward(int(x.shape[0]), time.perf_counter() - start)

    def _forward_grouped(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Run a batch expert group by expert group, so a layer loads each expert at most once per batch.

        A prefill routes to most experts; row chunks of ``rows_per_call`` would evict and reload them every chunk.
        Each group fits the slots; its calls mask the other routes out (slot -1) and the groups' partial sums add up.
        """
        experts = ids[ids >= 0].unique().tolist()
        capacity = self.cache.capacity
        groups = [experts[start : start + capacity] for start in range(0, len(experts), capacity)]
        partials: list[Optional[torch.Tensor]] = [None] * len(groups)
        # The most-resident group runs first, so the later ones evict only experts this batch is done with.
        for index in sorted(range(len(groups)), key=lambda g: -sum(e in self.cache.slot_of for e in groups[g])):
            partials[index] = self._group_partial(x, ids, weights, groups[index])
        # Summing in group order, not run order, keeps the output independent of what an earlier batch left resident.
        out.zero_()
        for partial in partials:
            out.add_(partial)

    def _group_partial(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, group: list[int]
    ) -> torch.Tensor:
        member = torch.isin(ids, torch.tensor(group, dtype=torch.int64, device="cpu"))
        rows = member.any(dim=1).nonzero().squeeze(1)
        masked = torch.where(member, ids, -1)
        self.cache.ensure(group)
        partial = torch.zeros((ids.shape[0], self.hidden_size), dtype=torch.float32, device="cpu")
        for begin in range(0, rows.numel(), self.rows_per_group_call):
            chunk = rows[begin : begin + self.rows_per_group_call]
            part = torch.empty((chunk.numel(), self.hidden_size), dtype=torch.float32, device="cpu")
            self._forward_chunk(x[chunk], masked[chunk], weights[chunk], part)
            partial[chunk] = part
        return partial



