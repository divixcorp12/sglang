"""The DSpark draft's CPU experts in the decode graph: the draft lease channel (draft_channel.h) from Python.

* ``DraftWire`` mirrors draft_channel.h (test_dspark_draft_channel_layout checks the two agree);
* ``DraftCpuAreas`` owns the channel buffer and the per-stage pinned areas the host thread reads and writes;
* ``DraftCpuDevice`` queues the device half: ``post`` (stage the CPU share, publish a record) and ``finish`` (close the
  gate, the stream's wait, add the host's rows). Both are device-only, so a graph captures them;
* ``DraftCpuHost`` runs the host half, the draft-only CPU expert engine (host/cpu_experts.h), in the expert-stream host module;
* ``SharedDraftHost`` serves the channel from an ``ExpertStreamHost`` group's CPU expert engine instead (one team per node).

The protocol is the lease channel's (LEASE_PROTOCOL.md, "The lease channel"); only the record and the areas are the
draft's.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
import sys
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

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
        extra_cuda_cflags=["-DSGLANG_DRAFT_REQUEST_TRACE=1"] if os.environ.get("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX") else [],
        cuda_wrappers=[
            ("post", "expert_stream::draft::DraftChannelKernels::post"),
            ("finish", "expert_stream::draft::DraftChannelKernels::finish"),
            ("clock", "expert_stream::draft::DraftChannelKernels::clock"),
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
        prefix = os.environ.get("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX")
        if prefix:
            # Calibration IO is initialization only, never post()/finish() or graph replay.
            self.calibrate_clock(str(prefix) + f".{os.getpid()}.draft-clock.json")

    def calibrate_clock(self, path: str) -> dict:
        samples = []
        out = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        with torch.cuda.device(self.state.device):
            self.module.clock(self.state, out.data_ptr())
            torch.cuda.synchronize()
            for _ in range(64):
                before = time.monotonic_ns()
                self.module.clock(self.state, out.data_ptr())
                torch.cuda.synchronize()
                after = time.monotonic_ns()
                samples.append(dict(before=before, gpu=int(out[0]), after=after))
        data = dict(clock="GPU globaltimer ns vs CLOCK_MONOTONIC", samples=samples,
                    offset_low=max(s["before"] - s["gpu"] for s in samples),
                    offset_high=min(s["after"] - s["gpu"] for s in samples))
        if data["offset_low"] > data["offset_high"]:
            raise RuntimeError("GPU/CPU clock calibration intervals do not overlap")
        Path(path).write_text(json.dumps(data, indent=2) + "\n")
        return data

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


class DraftCpuHost:
    """The host half: one draft CPU thread over `areas`, on `cores` with `threads` workers (the first core its own).

    Idle as the target's CPU expert engine: the team is held `keep_warm_us` in register work after each job, then
    `spin_us` in PAUSE (-1: until the next post), then the thread polls the head with 50 us sleeps. A record that stays
    incomplete for `fatal_wait_s` ends the process. Set every stage's layer, then `start`; `stop` is idempotent.
    """

    def __init__(
        self,
        areas: DraftCpuAreas,
        *,
        cores: Sequence[int],
        threads: int,
        spin_us: int,
        keep_warm_us: int,
        fatal_wait_s: float,
        variant: Optional[str] = None,
        layout: str = "exl3",
    ):
        from sglang.kernels.ops.moe import expert_stream_transport as ops

        self.areas = areas
        self._ops = ops
        self._module = ops._host_module(layout, variant)
        self._keep: dict[int, tuple] = {}  # each stage's spec.keep: the slabs its layer reads
        self.handle = int(
            self._module.expert_stream_draft_cpu_open(
                areas.channel,
                areas.x,
                areas.slots,
                areas.weights,
                areas.out,
                int(areas.hidden),
                int(areas.stages),
                int(threads),
                torch.tensor(list(cores), dtype=torch.int64),
                ops._spin_ns(int(spin_us)),
                int(keep_warm_us * 1000),
                int(fatal_wait_s * 1e9),
            )
        )
        _LIVE.add(self)

    def set_layer(self, stage: int, kernel: int, spec) -> None:
        """Stage `stage`'s layer from `kernel`'s make_layer over `spec` (a ``CpuExpertLayerSpec``); before start."""
        slabs, params = self._ops._layer_tensors(spec)
        self._module.expert_stream_draft_cpu_set_layer(
            self.handle,
            int(stage),
            int(kernel),
            slabs,
            int(spec.capacity),
            int(spec.hidden),
            int(spec.intermediate),
            int(spec.activation),
            float(spec.act_limit),
            params,
        )
        self._keep[int(stage)] = spec.keep

    def start(self) -> None:
        self._module.expert_stream_draft_cpu_start(self.handle)

    def stop(self) -> None:
        self._module.expert_stream_draft_cpu_stop(self.handle)

    def stats(self) -> dict:
        out = torch.zeros(7, dtype=torch.int64)
        self._module.expert_stream_draft_cpu_stats(self.handle, out)
        jobs, rows, forward_ns, holds, collided_jobs, shared_routes, collided_forward_ns = (int(v) for v in out.tolist())
        return {
            "jobs": jobs,
            "rows": rows,
            "forward_ns": forward_ns,
            "keep_warm_calls": holds,
            "collided_jobs": collided_jobs,
            "shared_routes": shared_routes,
            "collided_forward_ns": collided_forward_ns,
        }



class SharedDraftHost:
    """The draft channel served by an ExpertStreamHost group's CPU expert engine (one team per node, plan 2026-10-06
    Task 11), with DraftCpuHost's interface so the registry and the tests drive either. The engine's team, idle and
    watchdog are the host's; ``stop`` detaches the draft source (the engine stops with the host)."""

    def __init__(self, host, areas: DraftCpuAreas, *, group: int, fatal_wait_s: float):
        from sglang.kernels.ops.moe import expert_stream_transport as ops

        self.host, self.areas, self.group = host, areas, int(group)
        self._ops = ops
        self._module = host._module
        self._keep: dict[int, tuple] = {}
        self._last_stats: dict = {}
        self._module.expert_stream_draft_open(
            host.handle, self.group, areas.channel, areas.x, areas.slots, areas.weights, areas.out,
            int(areas.hidden), int(areas.stages), int(fatal_wait_s * 1e9),
        )
        host._draft_keep = (areas, self)  # the engine reads the areas until the host stops

    def set_layer(self, stage: int, kernel: int, spec) -> None:
        slabs, params = self._ops._layer_tensors(spec)
        self._module.expert_stream_draft_set_layer(
            self.host.handle, self.group, int(stage), int(kernel), slabs, int(spec.capacity), int(spec.hidden),
            int(spec.intermediate), int(spec.activation), float(spec.act_limit), params,
        )
        self._keep[int(stage)] = spec.keep

    def start(self) -> None:
        self._module.expert_stream_draft_start(self.host.handle, self.group)

    def _host_open(self) -> bool:
        """False once the service closed the host's native handle (its shutdown runs before the registry's atexit
        close); the engine and this source went with it, and every call on the handle raises "unknown handle"."""
        close = getattr(self.host, "_close", None)
        return close is None or close.alive

    def stop(self) -> None:
        if self._host_open():
            self._module.expert_stream_draft_stop(self.host.handle, self.group)

    def stats(self) -> dict:
        """The counters; once the host is closed, the last ones read here (``{}`` if none was)."""
        if not self._host_open():
            return self._last_stats
        out = torch.zeros(7, dtype=torch.int64)
        self._module.expert_stream_draft_stats(self.host.handle, self.group, out)
        jobs, rows, forward_ns, holds, collided_jobs, shared_routes, collided_forward_ns = (int(v) for v in out.tolist())
        self._last_stats = {"jobs": jobs, "rows": rows, "forward_ns": forward_ns, "keep_warm_calls": holds,
                            "collided_jobs": collided_jobs, "shared_routes": shared_routes,
                            "collided_forward_ns": collided_forward_ns}
        return self._last_stats

_LIVE: "weakref.WeakSet[DraftCpuHost]" = weakref.WeakSet()


@atexit.register
def _stop_live() -> None:
    """Stop every live draft CPU thread at exit, before the areas it reads are freed."""
    for host in list(_LIVE):
        try:
            host.stop()
        except Exception as error:  # noqa: BLE001 - keep stopping the others
            sys.stderr.write(f"DSpark draft CPU experts: stopping a thread failed: {error!r}\n")
