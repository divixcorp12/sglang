"""A persistent, CUDA-graph-capturable one-row side-stream expert pull.

``ExpertGpuPullPipeline`` owns one long-lived side CUDA stream per device and,
per target, the persistent state a captured fork/join needs: an
``ExpertRowSegments`` table, a capacity-1 ``ExpertRowPlan``, and timing-disabled
ready/done events. ``post_target`` forks a one-row pull for a target onto the
side stream; ``join_target`` folds its completion back onto the caller's
current stream. Both run during eager execution and during graph capture;
replay executes only the captured dependency edges, never Python.

This module carries no predictor, no scoring, and no serving wiring: callers
supply the plan contents (which expert row to pull, and whether to pull at
all via ``count``) and are responsible for joining every posted target before
a captured region ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import (
    ExpertRowSegments,
    copy_expert_row_segments_gpu,
)
from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan


@dataclass
class ExpertGpuPullTarget:
    """Persistent per-target state for one captured side-stream pull.

    ``segments`` and ``plan`` hold the only tensors this target addresses, so
    the target keeps them alive for its own lifetime. ``slot`` is the
    dedicated destination row this target's pull always writes; it is never
    published into a resident expert-to-slot map by this module. ``ready``
    and ``done`` are timing-disabled events private to this target: they must
    not be shared with any other target, including one built for the same
    tag on a later graph state.
    """

    tag: str
    device: torch.device
    segments: ExpertRowSegments
    plan: ExpertRowPlan
    slot: int
    ready: torch.cuda.Event = field(init=False)
    done: torch.cuda.Event = field(init=False)

    def __post_init__(self) -> None:
        if self.plan.capacity != 1:
            raise ValueError("an ExpertGpuPullTarget plan must have capacity 1.")
        if self.segments.table.device != self.device:
            raise ValueError("segments must be allocated on the target's device.")
        self.ready = torch.cuda.Event(enable_timing=False)
        self.done = torch.cuda.Event(enable_timing=False)


class ExpertGpuPullPipeline:
    """Owns one device's long-lived side stream and its registered targets.

    Create the pipeline, then every target it will pull for, before warmup
    and graph capture. Two independently executable graph states must use
    two separate ``ExpertGpuPullPipeline`` instances: nothing here is safe to
    share across overlapping replays.
    """

    def __init__(self, device: torch.device | str) -> None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        self.side_stream = torch.cuda.Stream(device=self.device)
        self._targets: dict[str, ExpertGpuPullTarget] = {}

    def create_target(
        self,
        tag: str,
        segments: ExpertRowSegments,
        plan: ExpertRowPlan,
        slot: int,
    ) -> ExpertGpuPullTarget:
        """Register and return persistent state for ``tag``.

        ``tag`` must be unique within this pipeline. ``plan`` must already be
        a capacity-1 ``ExpertRowPlan`` allocated on this pipeline's device;
        callers fill ``plan.expert_ids``, ``plan.slots`` and ``plan.count``
        before each ``post_target`` in eager mode, or before each replay
        under capture.
        """
        if tag in self._targets:
            raise ValueError(f"target tag {tag!r} is already registered.")
        target = ExpertGpuPullTarget(
            tag=tag, device=self.device, segments=segments, plan=plan, slot=slot
        )
        self._targets[tag] = target
        return target

    def target(self, tag: str) -> ExpertGpuPullTarget:
        return self._targets[tag]

    def post_target(self, target: ExpertGpuPullTarget) -> None:
        """Fork ``target``'s one-row pull from the current stream onto the side stream.

        The pull kernel loads ``target.plan.count`` itself; a count of zero
        launches without payload traffic but still executes and still
        records ``done``, so a captured count-zero pull remains part of any
        replay's dependency graph and overhead measurement.
        """
        origin = torch.cuda.current_stream(target.device)
        target.ready.record(origin)
        with torch.cuda.stream(self.side_stream):
            self.side_stream.wait_event(target.ready)
            copy_expert_row_segments_gpu(
                target.segments,
                target.plan.expert_ids,
                target.plan.slots,
                target.plan.count,
            )
            target.done.record(self.side_stream)

    def join_target(self, target: ExpertGpuPullTarget) -> None:
        """Join ``target``'s pull onto the current stream.

        Every posted target must be joined before its captured region ends,
        including a target-disabled or model-tail replay: an unjoined fork
        leaves the side stream free to overwrite ``target``'s destination
        before the next consumer reads it.
        """
        origin = torch.cuda.current_stream(target.device)
        origin.wait_event(target.done)

    def join_all(self) -> None:
        """Join every target this pipeline has posted for, in registration order."""
        for target in self._targets.values():
            self.join_target(target)
