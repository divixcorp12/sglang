"""L+2 fast path, fused-kernel form: a folded T0 / T3 head (l2_fast_torch.FoldedHead) on the spec_head JIT kernels.

to_kernel_weights() stores the folded matrices bf16 and the biases fp32 (D3). KernelHead owns the device weights, a
workspace and the skip / threshold buffers, and launches the fused path on the current stream: score() launches and reads
the record, launch() only launches (no host read, graph-capturable).

Inputs: a FoldedHead (l2_fast_torch.fold) and serving's inputs: biased fp32 gate scores, x, source routes and weights.
Output: the cap-1 read as (count, expert, value) plus the fp32 [E] block scores; count 0 reads nothing.
Usage: head = l2_fast_kernels.KernelHead(folded, device="cuda")
       count, expert, value, block = head.score(scores=s, x=x, src_ids=ids, src_w=w, positions=pos, row=0,
                                                skip=skip, threshold=thr)
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_fast_torch  # noqa: E402

from sglang.kernels.ops.moe import spec_head  # noqa: E402

# Weight matrices the kernels read as bf16; every other field is an fp32 bias or table.
MATRIX_FIELDS = frozenset(
    {"w_pca", "w_in_pca", "w_in_sparse", "w_r1", "w_r2", "w_gate", "w_out", "agg_w1", "agg_w2"}
)


def to_kernel_weights(folded: l2_fast_torch.FoldedHead, *, device: torch.device | str) -> spec_head.SpecHeadWeights:
    """Device copy of ``folded`` in the kernels' storage precision: bf16 matrices, fp32 biases (D3)."""
    parts = {}
    for name in folded.__struct_fields__:
        if name == "rows":
            continue
        dtype = torch.bfloat16 if name in MATRIX_FIELDS else torch.float32
        parts[name] = getattr(folded, name).detach().to(device=device, dtype=dtype).contiguous()
    return spec_head.SpecHeadWeights(rows=len(folded.rows), **parts)


class KernelHead:
    """One folded head's device weights and buffers; ``skip`` is expressed as row 0 of a one-row ``ram_slot`` map."""

    def __init__(self, folded: l2_fast_torch.FoldedHead, *, device: torch.device | str, features: bool = False):
        experts = folded.w_out.shape[0]
        self.weights = to_kernel_weights(folded, device=device)
        self.workspace = spec_head.new_workspace(
            device=device, features=features, pca_rank=folded.w_pca.shape[0], hidden=folded.b_in.shape[0],
            experts=experts)
        self.hot_slots = torch.full((1,), -1, dtype=torch.int64, device=device)  # hot_capacity 0: none hot
        self.ram_slot = torch.full((1, experts), -1, dtype=torch.int32, device=device)  # 0 mapped (skipped), -1 not
        self.threshold = torch.zeros(1, dtype=torch.float32, device=device)

    def launch(self, *, scores: torch.Tensor, x: torch.Tensor, src_ids: torch.Tensor, src_w: torch.Tensor,
               positions: torch.Tensor, row: int, skip: torch.Tensor, threshold: float | torch.Tensor) -> None:
        """K1-K4 on the current stream; no host read, so it can be captured in a CUDA graph.

        Inside a graph pass ``threshold`` as a device tensor and keep every input at its captured address.
        """
        self.ram_slot[0].copy_(skip.to(torch.int32) - 1)
        if isinstance(threshold, torch.Tensor):
            self.threshold.copy_(threshold.reshape(1))
        else:
            self.threshold.fill_(float(threshold))
        spec_head.run_spec_head(
            weights=self.weights, workspace=self.workspace, x=x.to(torch.bfloat16), scores=scores, src_ids=src_ids,
            src_w=src_w, positions=positions, row=row, hot_slots=self.hot_slots, hot_capacity=0,
            ram_slot=self.ram_slot, target=0, threshold=self.threshold)

    def score(self, *, scores: torch.Tensor, x: torch.Tensor, src_ids: torch.Tensor, src_w: torch.Tensor,
              positions: torch.Tensor, row: int, skip: torch.Tensor, threshold: float | torch.Tensor
              ) -> tuple[int, int, float, torch.Tensor]:
        """(count, expert or -1, value, fp32 [E] block scores) of one group; synchronizes to read the record."""
        self.launch(scores=scores, x=x, src_ids=src_ids, src_w=src_w, positions=positions, row=row, skip=skip,
                    threshold=threshold)
        count, expert, value = spec_head.read_record(self.workspace)
        return count, expert, value, self.workspace.block_scores.clone()
