"""The DSpark draft's CPU experts in the decode graph: the draft lease channel (draft_channel.h) from Python.

* ``DraftWire`` mirrors draft_channel.h (test_dspark_draft_channel_layout checks the two agree);
* ``DraftCpuAreas`` owns the channel buffer and the per-stage pinned areas the host thread reads and writes;
* ``DraftCpuDevice`` queues the device half: ``post`` (stage the CPU share, publish a record) and ``finish`` (close the
  gate, the stream's wait, add the host's rows). Both are device-only, so a graph captures them.

The protocol is the lease channel's (LEASE_PROTOCOL.md, "The lease channel"); only the record and the areas are the
draft's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@dataclass(frozen=True)
class DraftWire:
    """draft_channel.h's layout: the page (head, a ring of `records` records), the completion block (done words, the
    gate) in one buffer of `channel_bytes`, and the record's fields."""

    head: int = 0
    ring: int = 128
    records: int = 4
    record_bytes: int = 128
    done: int = 640
    gate: int = 768
    channel_bytes: int = 4096
    max_rows: int = 16
    max_k: int = 8
    rec_stage: int = 4
    rec_rows: int = 6
    rec_k: int = 7
    rec_epoch: int = 8


_GATE_OPEN_0 = 1  # gate_word(0, open): an untouched 0 would block the first wait (0 is below kGateOpen)
_STATE_WORDS = 6


def _pinned_zeros(shape, dtype, pin: bool) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, pin_memory=pin)


class DraftCpuAreas:
    """The draft channel's pinned buffers: `channel` uint8 [4096] (page + completion), and per stage `x` fp16
    [stages, 16, H], `slots` int32 [stages, 16, 8], `weights` fp32 [stages, 16, 8], `out` fp32 [stages, 16, H]."""

    def __init__(self, stages: int, hidden: int, *, pin: bool = True):
        w = DraftWire()
        if stages < 1 or hidden < 1 or hidden % 8:
            raise ValueError(f"draft CPU areas need stages >= 1 and a hidden size divisible by 8, not {stages}, {hidden}")
        self.wire, self.stages, self.hidden = w, stages, hidden
        # 128-byte alignment for the channel: the record slots and the gate each own whole lines.
        raw = _pinned_zeros(w.channel_bytes + 128, torch.uint8, pin)
        start = (-raw.data_ptr()) % 128
        self.channel = raw[start : start + w.channel_bytes]
        self.channel[w.gate : w.gate + 4].view(torch.int32)[0] = _GATE_OPEN_0
        self.x = _pinned_zeros((stages, w.max_rows, hidden), torch.float16, pin)
        self.slots = _pinned_zeros((stages, w.max_rows, w.max_k), torch.int32, pin)
        self.weights = _pinned_zeros((stages, w.max_rows, w.max_k), torch.float32, pin)
        self.out = _pinned_zeros((stages, w.max_rows, hidden), torch.float32, pin)


@cache_once
def device_module() -> Module:
    """The draft channel's kernels (draft_kernels.cuh)."""
    return load_jit(
        "dspark_draft_channel",
        cuda_files=["moe/expert_stream/draft_kernels.cuh"],
        cuda_wrappers=[
            ("post", "expert_stream::draft::DraftChannelKernels::post"),
            ("finish", "expert_stream::draft::DraftChannelKernels::finish"),
        ],
    )


class DraftCpuDevice:
    """The device side: `state` int32 [6] on the GPU (the channel's words), `on_cpu` uint8 [stages, E] (which experts
    of each stage the host computes)."""

    def __init__(self, areas: DraftCpuAreas, on_cpu: torch.Tensor, device):
        if on_cpu.dim() != 2 or on_cpu.shape[0] != areas.stages:
            raise ValueError(f"on_cpu must be [stages={areas.stages}, experts], not {tuple(on_cpu.shape)}")
        self.areas, self.wire = areas, areas.wire
        self.state = torch.zeros(_STATE_WORDS, dtype=torch.int32, device=device)
        self.on_cpu = on_cpu.to(device=device, dtype=torch.uint8).contiguous()
        self.module = device_module()

    def post(self, stage: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Stage `x` [M, H] and the CPU-owned routes of `topk_ids`/`topk_weights` [M, k]; publish a record if any."""
        rows, k = topk_ids.shape
        if rows > self.wire.max_rows or k > self.wire.max_k:
            raise ValueError(f"a draft CPU call holds {self.wire.max_rows} rows of {self.wire.max_k} routes, not {rows}x{k}")
        a = self.areas
        self.module.post(
            x.contiguous(),
            topk_ids.to(torch.int64).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            self.on_cpu[stage],
            self.state,
            a.channel.data_ptr(),
            a.x[stage].data_ptr(),
            a.slots[stage].data_ptr(),
            a.weights[stage].data_ptr(),
            stage,
        )

    def finish(self, stage: int, out: torch.Tensor) -> None:
        """Wait (on the stream) for the posted record's done word and add the host's rows into `out` [M, H] fp32."""
        self.module.finish(self.state, self.areas.channel.data_ptr(), self.areas.out[stage].data_ptr(), out)
