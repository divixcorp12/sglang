"""CPU compute of a streamed layer's RAM-tier experts, in place over the pinned tier's slabs.

Plan: docs/superpowers/plans/2026-09-29-dsv41-cpu-experts.md. Each streamed layer owns one
ExpertPinnedHostCache, so a layer's host slot ids index its own slabs, and the CPU kernel's
"expert id" is that slot. Decode only (batch 1).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES


def parse_core_list(spec: str) -> list[int]:
    """Cores from a taskset-style list such as "36-47,50"."""
    cores: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        cores.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return sorted(set(cores))


def k_star(n: int, c_cpu_ms: float, c_link_ms: float, handoff_ms: float) -> int:
    """How many of a layer's ``n`` RAM-tier experts go to the CPU.

    Minimizes max(handoff + k * c_cpu, (n - k) * c_link) over k = 0..n. A tie goes to the larger k:
    equal layer time, and the link's bandwidth stays free for the GPU's own copies.
    """
    best_k, best = 0, max(handoff_ms, n * c_link_ms)
    for k in range(1, n + 1):
        cost = max(handoff_ms + k * c_cpu_ms, (n - k) * c_link_ms)
        if cost <= best:
            best_k, best = k, cost
    return best_k


def split_table(
    max_n: int, c_cpu_ms: float, c_link_ms: float, handoff_ms: float
) -> torch.Tensor:
    """``k_star(n)`` for n = 0..max_n, int32 ``[max_n + 1]``, for a device-side lookup."""
    return torch.tensor(
        [k_star(n, c_cpu_ms, c_link_ms, handoff_ms) for n in range(max_n + 1)],
        dtype=torch.int32,
    )


class Exl3CpuExpertPool:
    """The CPU kernel's layers, one per streamed layer, registered once over the pinned slabs.

    Row contents change as the tier evicts and fills, but the registered views stay valid, so
    nothing re-registers. ``slabs_by_layer[layer]`` is that layer's ``ExpertPinnedHostCache.tensors``.
    The kernel's workers inherit the affinity of the thread that first runs a job, and that thread
    is worker 0, so ``compute`` binds its calling thread to ``cores`` on first use.
    """

    def __init__(
        self,
        ext: Any,
        slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        cores: Sequence[int],
        threads: int,
        act_limit: float,
    ):
        if len(set(cores)) < 2:
            # Many spinning workers on one core livelocked the box (DSV41_REFERENCE section 28).
            raise ValueError(
                f"the CPU expert pool needs at least 2 cores, got {sorted(set(cores))}"
            )
        if not 1 <= threads <= len(set(cores)):
            raise ValueError(f"{threads} pool threads on {len(set(cores))} cores")
        if os.environ.get("EXL3_MOE_CPU_PIN") != "0":
            raise ValueError(
                "EXL3_MOE_CPU_PIN=0 must be set: the kernel would otherwise pin its workers to the first cores"
            )
        self.ext = ext
        self.cores = tuple(sorted(set(cores)))
        self.threads = threads
        self.capacity: dict[int, int] = {}
        self._handles: dict[int, int] = {}
        self._keepalive = slabs_by_layer
        self._bound_threads: set[int] = set()
        try:
            for layer, slabs in slabs_by_layer.items():
                self._register(layer, slabs, act_limit)
        except BaseException:
            self.close()
            raise

    def _register(
        self, layer: int, slabs: Mapping[str, torch.Tensor], act_limit: float
    ) -> None:
        rows = {int(slabs[name].shape[0]) for name in EXL3_STREAMED_NAMES}
        if len(rows) != 1:
            raise ValueError(
                f"layer {layer}: pinned slabs disagree on their row count {rows}"
            )
        capacity = rows.pop()
        for name in EXL3_STREAMED_NAMES:
            if slabs[name].device.type != "cpu" or not slabs[name].is_contiguous():
                raise ValueError(
                    f"layer {layer} {name}: slab must be a contiguous CPU tensor"
                )
        if capacity == 0:
            return
        w13_t, w13_u, w13_v = slabs["w13_trellis"], slabs["w13_suh"], slabs["w13_svh"]
        w2_t, w2_u, w2_v = slabs["w2_trellis"], slabs["w2_suh"], slabs["w2_svh"]
        rows_of = range(capacity)
        # Gate is w13 part 0 and up is part 1; each [slot, part] view is contiguous.
        handle = self.ext.exl3_moe_cpu_make_layer(
            [w13_t[s, 0] for s in rows_of],
            [w13_u[s, 0] for s in rows_of],
            [w13_v[s, 0] for s in rows_of],
            [w13_t[s, 1] for s in rows_of],
            [w13_u[s, 1] for s in rows_of],
            [w13_v[s, 1] for s in rows_of],
            [w2_t[s] for s in rows_of],
            [w2_u[s] for s in rows_of],
            [w2_v[s] for s in rows_of],
            [],
            [],
            [],
            0,
            act_limit,
            0,
        )
        self._handles[layer] = handle
        self.capacity[layer] = capacity

    def compute(
        self,
        layer: int,
        host_slots: torch.Tensor,
        weights: torch.Tensor,
        x16: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Write this layer's CPU share of the routed sum to ``out`` (fp32 ``[1, H]``, overwritten).

        ``host_slots`` int64 ``[1, k]`` (-1 skips), ``weights`` fp16 ``[1, k]``, ``x16`` fp16 ``[1, H]``.
        """
        if x16.shape[0] != 1:
            raise ValueError(
                f"the CPU expert pool is batch-1 decode only, got {x16.shape[0]} rows"
            )
        if (
            host_slots.dtype != torch.int64
            or host_slots.shape[0] != 1
            or weights.shape != host_slots.shape
        ):
            raise ValueError(
                "host_slots must be int64 [1, k] with weights of the same shape"
            )
        capacity = self.capacity[layer]
        if int(host_slots.max()) >= capacity:
            # The kernel would skip it silently.
            raise ValueError(
                f"layer {layer}: host slot {int(host_slots.max())} is outside the tier's {capacity} rows"
            )
        tid = threading.get_native_id()
        if tid not in self._bound_threads:
            os.sched_setaffinity(0, self.cores)
            self._bound_threads.add(tid)
        self.ext.exl3_moe_cpu_forward(
            self._handles[layer], x16, host_slots, weights, out, self.threads
        )

    def close(self) -> None:
        handles, self._handles = self._handles, {}
        for handle in handles.values():
            self.ext.exl3_moe_cpu_free_layer(handle)
