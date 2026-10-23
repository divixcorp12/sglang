"""trace.residency_capture's per-row hook on the RAM-miss row post.

Right after a row's post kernel typed its lanes, ``ResidencyHook.launch`` runs the snapshot and label kernels into
the hook's staging buffers and copies them into the route log's residency rings. Record layout:
``residency_capture_ref``'s module docstring.
"""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.kernels.ops.moe.expert_stream_transport import (
    run_residency_snapshot,
    run_route_labels,
)
from sglang.srt.layers.moe.residency_capture_ref import (
    HORIZONS,
    MASK_WORDS,
    horizon_targets,
)


class ResidencyHook:
    """One map row's residency capture: its precomputed horizon targets/valid and the kernels' staging buffers.

    Every tensor is allocated here, at attach, before any decode graph is captured; ``launch`` allocates nothing.
    """

    def __init__(
        self,
        row: int,
        *,
        targets: Sequence[int],
        valid: Sequence[bool],
        width: int,
        topk: int,
        device,
    ) -> None:
        if len(targets) != HORIZONS or len(valid) != HORIZONS:
            raise ValueError(f"row {row} needs {HORIZONS} horizon targets and flags, got {len(targets)}/{len(valid)}")
        if width < 1 or topk < 1 or width % topk:
            raise ValueError(f"row {row}: {width} routes are not whole tokens of top-{topk}")
        self.row = row
        self.tokens, self.topk = width // topk, topk
        self.targets = torch.tensor(list(targets), dtype=torch.int32, device=device)
        self.valid = torch.tensor([int(bool(v)) for v in valid], dtype=torch.uint8, device=device)
        self.masks = torch.zeros((HORIZONS, MASK_WORDS), dtype=torch.int64, device=device)
        self.out_valid = torch.zeros(HORIZONS, dtype=torch.uint8, device=device)
        self.snap_stamp = torch.zeros(1, dtype=torch.int64, device=device)
        self.kinds = torch.zeros(width, dtype=torch.uint8, device=device)
        self.label_stamp = torch.zeros(1, dtype=torch.int64, device=device)

    def launch(self, *, side, routes: torch.Tensor, planned: torch.Tensor, count: torch.Tensor, route_log) -> None:
        """Snapshot the map bank's next four rows, label this post's routes, and log both (captured in the graph).

        Must run after ``side.post`` on the same stream: the labels read the ``lane_kind`` it wrote.
        """
        # The device side's own build, so the capture adds no JIT module of its own.
        module = side._kernels()
        run_residency_snapshot(
            ram_slot=side.map_bank["ram_slot"],
            targets=self.targets,
            valid=self.valid,
            out_masks=self.masks,
            out_valid=self.out_valid,
            out_stamp=self.snap_stamp,
            module=module,
        )
        # ``planned`` is padded past the wire's lanes; the post typed only the first ``lanes`` of them.
        lanes = side.lane_kind.numel()
        run_route_labels(
            routes=routes,
            planned=planned[:lanes],
            count=count,
            lane_kind=side.lane_kind,
            out_kinds=self.kinds,
            out_stamp=self.label_stamp,
            module=module,
        )
        route_log.record_residency(
            self.row,
            masks=self.masks,
            valid=self.out_valid,
            snap_stamp=self.snap_stamp,
            kinds=self.kinds.view(self.tokens, self.topk),
            label_stamp=self.label_stamp,
        )


def residency_hook_for(row: int, *, layer_ids: Sequence[int], width: int, topk: int, device) -> ResidencyHook:
    """The hook of map row ``row``, with its targets/valid from ``horizon_targets`` over every row's ``layer_ids``."""
    targets, valid = horizon_targets(layer_ids)
    return ResidencyHook(row, targets=targets[row], valid=valid[row], width=width, topk=topk, device=device)
