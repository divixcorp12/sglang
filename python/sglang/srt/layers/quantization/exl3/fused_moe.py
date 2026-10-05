"""exllamav3's fused exl3_moe over hot-cache slots, for in-graph decode and verify, 1-16 tokens.

The graph gather returns ``remap``: one hot-cache slot per route (hits in place,
misses in scratch rows; tokens may share a slot, a token's own routes are distinct). The fused kernel treats slots as its
"experts": nine pointer tables hold every slot's row addresses (fixed for the
life of the hot cache), ``expert_count`` marks the routed slots, and the
deterministic path (output scratch + exl3_moe_gather, the FUSED_DET mode) makes
replays bitwise reproducible. Every op here is capture-safe: no host reads, no
allocation that depends on data. This is the call sequence the P2 probe
(test/manual/dsv41/test_exl3_moe_probe_gpu.py) gated.
"""

from __future__ import annotations

from typing import Mapping

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.exl3.ext import exl3_ext

ACT_SILU = 0
ROW_TILE = 16  # fused-kernel rows per slot tile: a slot holds one row per token
# The P2 probe's "num_active" (Task 1 Step 5, $ANA/probe-exl3-moe.json): 6 when the
# static six-expert launch passed parity and bitwise replay, else -1 (all-fused).
NUM_ACTIVE = 6

_SHARED_TEMPS: dict = {}


def shared_temps(device, hidden: int, inter: int):
    """The fused kernel's four temp buffers, one set per device and shape for every layer.

    Layers run one after another on one stream, so they never use the set at once;
    per-layer sets would cost ~10 MB x 40 layers.
    """
    key = (str(device), hidden, inter)
    temps = _SHARED_TEMPS.get(key)
    if temps is None:
        concurrency = exl3_ext().exl3_moe_max_concurrency(torch.device(device).index)
        half = dict(dtype=torch.float16, device=device)
        temps = (
            torch.empty((concurrency, ROW_TILE, hidden), **half),
            torch.empty((concurrency, ROW_TILE, hidden), **half),
            torch.empty((concurrency, ROW_TILE, inter), **half),
            torch.empty((concurrency, ROW_TILE, inter), **half),
        )
        _SHARED_TEMPS[key] = temps
    return temps


_PROJECTIONS = (("gate", "w13", 0), ("up", "w13", 1), ("down", "w2", 0))
_KINDS = ("trellis", "suh", "svh")


def slot_pointer_tables(
    tensors: Mapping[str, torch.Tensor], slots: int
) -> dict[str, torch.Tensor]:
    """Nine int64 [slots] tables of row addresses: gate = w13 part 0, up = w13 part 1, down = w2."""
    device = tensors["w13_trellis"].device
    return {
        f"{proj}_{kind}": torch.tensor(
            [
                tensors[f"{prefix}_{kind}"][slot, part].data_ptr()
                for slot in range(slots)
            ],
            dtype=torch.int64,
            device=device,
        )
        for proj, prefix, part in _PROJECTIONS
        for kind in _KINDS
    }


def route_tables(remap, expert_count, ones, weights, keep, token_sorted=None, top_k=1):
    """Fill ``expert_count`` from ``remap``; return (inv_order, weight_sorted fp16, det tables).

    Routes are ranked by slot, ties by route index (a stable sort: several tokens may route one slot); with
    ``token_sorted`` given, each rank's token, ``route // top_k``, is written into it.
    ``keep`` (fp32 [1]) scales every route weight and, when 0, empties ``expert_count``,
    so a dropped layer runs no expert.
    ``det`` is exllamav3's device-built deterministic table stack
    ``[expert_start, expert_start, count > 0]``.
    """
    expert_count.zero_().index_add_(0, remap, ones)
    order = torch.argsort(remap, stable=True)
    inv_order = torch.empty_like(order).scatter_(
        0, order, torch.arange(order.numel(), device=order.device)
    )
    if token_sorted is not None:
        torch.floor_divide(order, top_k, out=token_sorted)
    weight_sorted = (weights[order].float() * keep).to(torch.float16)
    # A dropped layer runs no expert: nothing reads rows that may be half written.
    expert_count.mul_((keep > 0).to(torch.int64))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    det = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
    return inv_order, weight_sorted, det


class Exl3FusedMoE:
    """Static buffers and pointer tables of one streamed layer's in-graph fused MoE."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        slots: int,
        hidden: int,
        inter: int,
        top_k: int,
        device,
        tokens: int = 1,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Exl3FusedMoE must be built before CUDA-graph capture (in a warmup)"
            )
        ext = exl3_ext()
        self.ext = ext
        self.slots = slots
        self.top_k = top_k
        self.tokens = tokens
        routes = tokens * top_k
        self.tables = slot_pointer_tables(tensors, slots)
        self.bits = {
            "gate": tensors["w13_trellis"].shape[-1] // 16,
            "up": tensors["w13_trellis"].shape[-1] // 16,
            "down": tensors["w2_trellis"].shape[-1] // 16,
        }
        half = dict(dtype=torch.float16, device=device)
        self.expert_count = torch.zeros(slots + 1, dtype=torch.int64, device=device)
        self.ones = torch.ones(routes, dtype=torch.int64, device=device)
        self.token_sorted = torch.zeros(routes, dtype=torch.int64, device=device)
        self.scratch = torch.empty((routes, hidden), dtype=torch.float32, device=device)
        self.out = torch.empty((tokens, hidden), dtype=torch.float32, device=device)
        self.x16 = torch.empty((tokens, hidden), **half)
        # The temps stay ROW_TILE rows a slot: a slot holds at most one row per token.
        (
            self.temp_state_g,
            self.temp_state_u,
            self.temp_intermediate_g,
            self.temp_intermediate_u,
        ) = shared_temps(device, hidden, inter)
        # SGLANG_DSV41_ENABLE_LAYER_FUSION: route_tables and the copies around it as one kernel, into these buffers.
        self.layer_fusion = envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.get()
        if self.layer_fusion:
            self.remap64 = torch.zeros(routes, dtype=torch.int64, device=device)
            self.inv_order = torch.zeros(routes, dtype=torch.int64, device=device)
            self.weight_sorted = torch.zeros(routes, **half)
            self.det = torch.zeros((3, slots + 1), dtype=torch.int64, device=device)

    def _fused_route_tables(self, x, topk_weights, remap, keep, m, cpu=None):
        from sglang.kernels.ops.moe.exl3_route_tables import exl3_moe_route_tables

        cpu_lanes, dst_slots, cpu_out, cpu_part_stride = cpu if cpu is not None else (None, None, 0, 0)
        routes = m * self.top_k
        exl3_moe_route_tables(
            remap.contiguous(),
            topk_weights.contiguous(),
            keep,
            x.contiguous(),
            self.remap64[:routes],
            self.x16[:m],
            self.out[:m],
            self.expert_count,
            self.inv_order[:routes],
            self.weight_sorted[:routes],
            self.det,
            cpu_lanes=cpu_lanes,
            dst_slots=dst_slots,
            cpu_out=cpu_out,
            cpu_part_stride=cpu_part_stride,
            token_sorted_out=self.token_sorted[:routes],
        )
        return self.remap64[:routes], self.inv_order[:routes], self.weight_sorted[:routes], self.det

    def run(self, x, topk_weights, remap, keep, act_limit: float, cpu=None) -> torch.Tensor:
        """x [M, H], 1 <= M <= tokens, any float dtype; topk_weights [M * top_k]; remap [M * top_k] slots, int64
        (int32 too with layer fusion); keep fp32 [1]. Returns the fp32 [M, H] output, a view of this object's buffer.

        ``cpu`` = (cpu_lanes, dst_slots, cpu_out address, part stride), CPU experts only, one token: the routes the
        CPU computed are left out and the partial sums CC flagged seed the output (exl3_route_tables.cuh). Layer
        fusion only."""
        m = x.shape[0]  # a host-side shape read: capture-safe
        if not 1 <= m <= self.tokens:
            raise ValueError(f"exl3 in-graph MoE runs 1-{self.tokens} tokens, not {m}")
        if cpu is not None and m != 1:
            raise RuntimeError(f"CPU experts run one token, not {m}")
        if cpu is not None and not self.layer_fusion:
            raise RuntimeError("CPU experts need SGLANG_DSV41_ENABLE_LAYER_FUSION: only its route tables leave CPU routes out")
        routes = m * self.top_k
        x16, out = self.x16[:m], self.out[:m]
        token_sorted, scratch = self.token_sorted[:routes], self.scratch[:routes]
        if self.layer_fusion:
            remap, inv_order, weight_sorted, det = self._fused_route_tables(
                x, topk_weights, remap, keep, m, cpu
            )
        else:
            x16.copy_(x)
            inv_order, weight_sorted, det = route_tables(
                remap,
                self.expert_count,
                self.ones[:routes],
                topk_weights,
                keep,
                token_sorted=token_sorted,
                top_k=self.top_k,
            )
            out.zero_()
        t = self.tables
        self.ext.exl3_moe(
            x16,
            out,
            self.expert_count,
            token_sorted,
            weight_sorted,
            self.temp_state_g,
            self.temp_state_u,
            self.temp_intermediate_g,
            self.temp_intermediate_u,
            ACT_SILU,
            self.bits["gate"],
            self.bits["up"],
            self.bits["down"],
            t["gate_trellis"],
            t["gate_suh"],
            t["gate_svh"],
            t["up_trellis"],
            t["up_suh"],
            t["up_svh"],
            t["down_trellis"],
            t["down_suh"],
            t["down_svh"],
            False,
            True,
            False,
            True,
            False,
            True,
            float(act_limit),
            # One token routes NUM_ACTIVE distinct slots; several share slots unpredictably, and exllamav3 takes
            # any count by ticket, so -1 sizes the launch for the most it can hold.
            NUM_ACTIVE if m == 1 else -1,
            scratch,
            det[0],
            1,
            ROW_TILE,
            16,
        )
        self.ext.exl3_moe_gather(
            out,
            scratch,
            remap,
            inv_order,
            det[1, : self.slots],
            det[0, : self.slots],
            det[2, : self.slots],
            weight_sorted,
        )
        return out


def exl3_fused_moe_for(layer, streamer) -> Exl3FusedMoE:
    """The layer's fused MoE, built on first use (a warmup forward, before capture)."""
    fused = getattr(layer, "_exl3_fused_moe", None)
    if fused is None:
        cache = streamer.hot_cache
        rows = streamer.graph_gather_rows
        top_k = layer.top_k
        # The route buffers hold up to `tokens` tokens' top_k routes. A slot holds at most one route per token, and
        # the fused kernel skips a slot with more rows than its ROW_TILE temp tile. DIRECT resolves misses into
        # resident hot slots before this kernel; other modes need scratch rows to hold every route that is not
        # resident.
        if rows % top_k:
            raise ValueError(
                f"exl3 in-graph MoE needs graph_gather_rows ({rows}) to be a multiple of top_k ({top_k})"
            )
        tokens = rows // top_k
        if tokens > ROW_TILE:
            raise ValueError(f"exl3 in-graph MoE: {tokens} tokens exceed the fused kernel's {ROW_TILE}-row tile")
        updater = getattr(cache, "device_residency", None)
        direct = (
            getattr(updater, "insert_direct", False)
            and getattr(streamer.row_backend, "name", None) == "exl3_ram_miss"
        )
        if direct and cache.capacity < rows:
            raise ValueError(
                f"exl3 DIRECT needs at least top_k resident slots per token, one per route ({cache.capacity} < {rows})"
            )
        if not direct and cache.scratch_rows < rows:
            raise ValueError(
                f"exl3 in-graph MoE needs a scratch row per route ({cache.scratch_rows} < {rows})"
            )
        if cache.reserves_prefetch_pull_row:
            raise ValueError(
                "exl3 in-graph MoE does not cover a prefetch-pull row (expert prefetch is off for EXL3)"
            )
        slots = cache.capacity + cache.scratch_rows
        fused = Exl3FusedMoE(
            cache.tensors,
            slots,
            hidden=cache.tensors["w13_suh"].shape[-1],
            inter=cache.tensors["w2_suh"].shape[-1],
            top_k=top_k,
            device=cache.device,
            tokens=tokens,
        )
        layer._exl3_fused_moe = fused
    return fused
