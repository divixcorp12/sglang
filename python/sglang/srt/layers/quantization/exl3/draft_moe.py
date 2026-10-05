"""The DSpark draft's GPU experts in the decode graph: one slot slab and Exl3FusedMoE (D2-1's in-graph MoE).

A stage's resident experts (all of them without CPU experts) are copied into a [slots, part, ...] slab per EXL3 name,
in sorted id order. expert_to_slot sends each resident id to its slot and every other id (CPU-owned, -1, out of range)
to the sink slot `slots`: exllamav3's exl3_moe ignores expert_count's last column, so sink routes add nothing. Nothing
here reads the device: remap is a gather on a device table.
"""

from typing import Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.quantization.exl3.fused_moe import ROW_TILE, Exl3FusedMoE

# The fused route tables take at most this many routes a call (exl3_route_tables.cuh, kMaxRoutes).
FUSED_MAX_ROUTES = 64


def draft_slot_map(resident_ids: Sequence[int], *, n_experts: int, device) -> torch.Tensor:
    """int64 [n_experts + 1]: the slot of each resident id, and `len(resident_ids)` (the sink) for every other id and at
    index n_experts, where remap sends ids outside [0, n_experts)."""
    ids = sorted(set(int(e) for e in resident_ids))
    if any(e < 0 or e >= n_experts for e in ids):
        raise ValueError(f"DSpark draft resident ids {ids} outside [0, {n_experts})")
    table = torch.full((n_experts + 1,), len(ids), dtype=torch.int64)
    table[ids] = torch.arange(len(ids), dtype=torch.int64)
    return table.to(device)


class DraftResidentMoe:
    """A draft stage's GPU experts as one slot slab, for Exl3FusedMoE. Route r goes to slot expert_to_slot[id]; every
    id that is not resident (CPU-owned, -1, out of range) goes to the sink slot `slots`, which the fused kernel
    ignores."""

    TOKENS = ROW_TILE  # the most tokens any stage's call holds (the fused kernel's row tile); `tokens` is this stage's

    def __init__(self, layer, resident_ids: Sequence[int], n_experts: int, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # The fused MoE sizes its temps by the device's index (exl3_moe_max_concurrency).
            device = torch.device("cuda", torch.cuda.current_device())
        self.ids = sorted(set(int(e) for e in resident_ids))
        if not self.ids:
            raise ValueError("a DSpark draft stage with no GPU expert has no GPU share; CPU-only stages are not supported")
        self.n_experts = n_experts
        self.slots = len(self.ids)
        self.expert_to_slot = draft_slot_map(self.ids, n_experts=n_experts, device=device)
        index = torch.tensor(self.ids, dtype=torch.long)

        def slab(name):
            data = getattr(layer, name).data
            if self.ids == list(range(data.shape[0])) and data.device == device and data.is_contiguous():
                return data  # every expert resident where it already lives: the parameters are the slab, no copy
            return data.index_select(0, index.to(data.device)).to(device).contiguous()

        self.tensors = {name: slab(name) for name in EXL3_STREAMED_NAMES}
        self.hidden = self.tensors["w13_suh"].shape[-1]
        self.inter = self.tensors["w2_suh"].shape[-1]
        self.top_k = layer.top_k
        # One call's tokens: the row tile, and at most FUSED_MAX_ROUTES routes. A larger M runs in chunks of `tokens`.
        self.tokens = min(self.TOKENS, FUSED_MAX_ROUTES // self.top_k)
        if self.tokens < 1:
            raise ValueError(f"a DSpark draft stage's top_k ({self.top_k}) exceeds the fused MoE's {FUSED_MAX_ROUTES} routes")
        self.device = device
        self.keep = torch.ones(1, dtype=torch.float32, device=device)
        self.fused = None

    def prepare(self) -> None:
        """Build the fused MoE's static buffers and pointer tables (host work): before capture."""
        if self.fused is None:
            self.fused = Exl3FusedMoE(
                self.tensors, self.slots, self.hidden, self.inter, self.top_k, self.device, tokens=self.tokens
            )

    @staticmethod
    def remap_with(table: torch.Tensor, topk_ids: torch.Tensor, *, n_experts: int) -> torch.Tensor:
        """[M, k] ids -> [M * k] int64 slots through `table`; elementwise, so capture-safe."""
        ids = topk_ids.reshape(-1).to(torch.int64)
        bad = (ids < 0) | (ids >= n_experts)
        return table[torch.where(bad, torch.full_like(ids, n_experts), ids)]

    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor:
        return self.remap_with(self.expert_to_slot, topk_ids, n_experts=self.n_experts)

    def run(self, x, topk_ids, topk_weights, act_limit) -> torch.Tensor:
        """x [M, H], M <= tokens; returns the fp32 [M, H] output of the resident routes, a view of the fused MoE's
        buffer (valid until its next run)."""
        if self.fused is None:
            raise RuntimeError("DraftResidentMoe.run before prepare()")
        weights = topk_weights.reshape(-1).to(torch.float32)
        return self.fused.run(x, weights, self.remap(topk_ids), self.keep, act_limit)


def prepare_dspark_draft_graph(model) -> int:
    """Prepare every DSpark draft stage's MoE for capture (host work: the fused MoE's buffers and tables, and the CPU
    runtime when any stage has CPU experts). Returns how many draft MoE layers it prepared."""
    count, cpu = 0, False
    for module in model.modules():
        moe = getattr(module, "exl3_draft_moe", None)
        if moe is None:
            continue
        moe.prepare()
        count += 1
        cpu = cpu or hasattr(module, "exl3_cpu_draft_key")
    if cpu:
        from sglang.srt.layers.moe.cpu_experts import draft

        draft.DRAFT_CPU_EXPERTS.prepare()
    return count
