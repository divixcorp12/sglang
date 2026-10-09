"""The NVMe-to-RAM prefetch's Python side (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
Phase 1): the host's bounds, the model's router gates registered at load, and the per-row target table the host's
speculative threads score with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

# The host's bounds: host/gate_scorer.h GateScorer::kDepth and kMaxPerLayer, host/ram_prefetch.h SpecPool::kMaxShare.
MAX_PER_TOKEN = 12
MAX_PER_LAYER = 8
MAX_SPEC_SHARE = 4

# SGLANG_DSV41_RAM_PREFETCH_SCORER's values.
SCORERS = ("cpu", "gpu")


@dataclass(frozen=True)
class RouterGate:
    weight: torch.Tensor  # [experts, hidden], the router's own parameter (any device)
    bias: Optional[torch.Tensor]  # [experts] e_score_correction_bias; None on a hash-routed layer
    top_k: int


_GATES: dict[int, RouterGate] = {}


def register_router_gate(layer_id: int, weight: torch.Tensor, bias: Optional[torch.Tensor], top_k: int) -> None:
    """Registers ``layer_id``'s router; the same gate again is a no-op, another one raises (keyed by layer id only,
    so a second model's gate would otherwise replace the first's)."""
    gate = RouterGate(weight, bias, int(top_k))
    old = _GATES.get(int(layer_id))
    if old is not None and (old.weight is not weight or old.bias is not bias or old.top_k != gate.top_k):
        raise ValueError(f"RAM prefetch: layer {int(layer_id)} already has a router gate from another model or load")
    _GATES[int(layer_id)] = gate


def register_moe_gates(layers: Mapping[int, object], top_k: int) -> int:
    """Registers every MoE layer's router: ``layers`` maps a layer id to its decoder layer, whose ``mlp`` holds
    ``gate`` (``weight``, ``e_score_correction_bias``) and ``is_hash``. A dense layer has no gate. Returns how many."""
    registered = 0
    for layer_id, layer in layers.items():
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        if gate is None or not hasattr(gate, "weight"):
            continue
        bias = None if getattr(mlp, "is_hash", False) else getattr(gate, "e_score_correction_bias", None)
        register_router_gate(layer_id, gate.weight, bias, top_k)
        registered += 1
    return registered


def registered_gates() -> dict[int, RouterGate]:
    return dict(_GATES)


def clear_router_gates() -> None:
    _GATES.clear()


@dataclass(frozen=True)
class PrefetchTables:
    targets: torch.Tensor  # int64 [rows, 2]: per source row its target row and gate index, or (-1, -1)
    gates: torch.Tensor  # bf16 [n, experts, hidden], host memory
    bias: torch.Tensor  # fp32 [n, experts]
    top_k: int


@dataclass(frozen=True)
class PrefetchTargets:
    targets: torch.Tensor  # int64 [rows, 2]: per source row its target row and gate index, or (-1, -1)
    gates: tuple  # RouterGate per gate index: the target layer's registered gate, uncopied
    top_k: int


def prefetch_targets(layer_ids: Sequence[int], gates: Mapping[int, RouterGate], *, hidden: int) -> PrefetchTargets:
    """Row r targets row r + 1 when that row is the next layer (layer_ids[r] + 1) and its gate has a bias; the last row,
    a non-consecutive layer and a hash-routed one target nothing. Every target's gate has ``hidden`` columns, one
    expert count and one top_k."""
    picks = []
    for row in range(len(layer_ids) - 1):
        nxt = layer_ids[row + 1]
        gate = gates.get(nxt)
        if nxt == layer_ids[row] + 1 and gate is not None and gate.bias is not None:
            picks.append((row, gate, nxt))
    if not picks:
        raise ValueError("RAM prefetch: no streamed row has a next layer with a registered, biased router gate")
    experts = int(picks[0][1].weight.shape[0])
    top_k = picks[0][1].top_k
    for _, gate, layer_id in picks:
        if gate.weight.dim() != 2 or int(gate.weight.shape[1]) != hidden:
            raise ValueError(
                f"RAM prefetch: layer {layer_id}'s gate has hidden size {int(gate.weight.shape[-1])}, "
                f"the CPU rows {hidden}"
            )
        if int(gate.weight.shape[0]) != experts:
            raise ValueError(
                f"RAM prefetch: layer {layer_id}'s gate has {int(gate.weight.shape[0])} experts, not {experts}"
            )
        if gate.top_k != top_k:
            raise ValueError(f"RAM prefetch: layer {layer_id}'s top_k {gate.top_k} is not {top_k}")
    targets = torch.full((len(layer_ids), 2), -1, dtype=torch.int64)
    for index, (row, _, _) in enumerate(picks):
        targets[row, 0], targets[row, 1] = row + 1, index
    return PrefetchTargets(targets, tuple(gate for _, gate, _ in picks), top_k)


def prefetch_tables(
    layer_ids: Sequence[int], gates: Mapping[int, RouterGate], *, hidden: int, node: Optional[int] = None
) -> PrefetchTables:
    """prefetch_targets with each target's gate copied to host memory once, bound to NUMA ``node`` when given, for the
    CPU scorer. The host scorer reads bf16, so an fp32 router (``router_fp32``) is rounded to bf16 here: its ranking can
    differ slightly from the model's, which costs prefetch hit rate and never correctness."""
    picked = prefetch_targets(layer_ids, gates, hidden=hidden)
    n = len(picked.gates)
    experts = int(picked.gates[0].weight.shape[0])
    row_bytes = experts * hidden * 2
    if node is None:
        flat = torch.empty((n, row_bytes), dtype=torch.uint8)
    else:
        from sglang.srt.layers.moe.host_numa import allocate_bound

        flat = allocate_bound(n * row_bytes, [(node, 0, n)], row_bytes).view(n, row_bytes)
    weights = flat.view(torch.bfloat16).view(n, experts, hidden)
    bias = torch.empty((n, experts), dtype=torch.float32)
    for index, gate in enumerate(picked.gates):
        weights[index].copy_(gate.weight.detach().to(device="cpu", dtype=torch.bfloat16))
        bias[index].copy_(gate.bias.detach().to(device="cpu", dtype=torch.float32))
    return PrefetchTables(picked.targets, weights, bias, picked.top_k)


@dataclass(frozen=True)
class SpecScoreRow:
    """One source row's GPU scoring (SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu): its target row's live router gate and VRAM
    hot slots, which the scoring kernels' captured parameters point at."""

    weight: torch.Tensor  # [experts, hidden], bf16 or fp32 (router_fp32): the router's own parameter
    bias: torch.Tensor  # [experts] fp32
    target: int
    hot_slots: torch.Tensor  # int64: the target layer's VRAM slots' experts, -1 empty
    hot_capacity: int


def spec_score_rows(picked: PrefetchTargets, hot: Mapping[int, tuple]) -> dict:
    """The GPU scorer's per-row table: each source row whose target row's hot slots are known (``hot``: row -> (hot
    slots, capacity), filled as rows attach). Refuses a gate the kernels cannot read."""
    rows = {}
    for row, (target, index) in enumerate(picked.targets.tolist()):
        if target < 0 or target not in hot:
            continue
        gate = picked.gates[index]
        if gate.weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"RAM prefetch: the GPU scorer reads a bf16 or fp32 gate; row {target}'s is {gate.weight.dtype}"
            )
        if gate.bias.dtype != torch.float32:
            raise ValueError(f"RAM prefetch: the GPU scorer reads an fp32 bias; row {target}'s is {gate.bias.dtype}")
        hot_slots, capacity = hot[target]
        rows[row] = SpecScoreRow(gate.weight.detach(), gate.bias.detach(), int(target), hot_slots, int(capacity))
    return rows


@dataclass(frozen=True)
class GpuScorer:
    """The GPU scorer as the service started it: the targets and their live gates, and the candidate page the select
    kernel writes and the host reads."""

    picked: PrefetchTargets
    candidates: torch.Tensor
    per_token: int
    top_k_only: bool
