"""Reference model of the device slot map (analysis/dsv41-drive/LEASE_PROTOCOL.md): lane typing in the post and
delta application. The CUDA post kernel and map_bulk_apply transcribe it; the parity tests compare the two."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Optional, Sequence

from sglang.kernels.ops.moe.expert_lease_block import wire_layout


class LaneKind(IntEnum):
    """lease_layout.h kKind*: what moves a lane's bytes and what the device waits on."""

    HIT_COPY = 1
    HIT_SM = 2
    HIT_CPU = 3
    MISS_GPU = 4
    MISS_CPU = 5


class LaneOverflow(ValueError):
    """Forced lanes (DIRECT found them no VRAM victim, the post's spill) cannot be CPU lanes: no host lanes (the copy
    engine is not armed) or no CPU layer. The post then serves the unforced prefix and flags the forward
    (exl3_ram_miss_post_kernel)."""


def type_lanes(
    experts: Sequence[int],
    ram_slot: Sequence[int],
    staging: Sequence[int],
    split: Sequence[int],
    *,
    lanes: int,
    captured: bool,
    copy_armed: bool,
    hit_copy: str,
    cpu_on: bool,
    cpu_misses: bool,
    ce_ok: bool = True,
    cpu_ok: bool = True,
    dst_ok: Optional[Sequence[bool]] = None,
    nodes: int = 1,
    forced_from: Optional[int] = None,
) -> tuple[list[LaneKind], list[int]]:
    """Each lane's kind and source slot: its RAM slot for a hit, the next staging slot of its home node for a miss.

    ``staging`` is node-major, ``nodes * lanes`` slots, and ``split`` node-major, ``nodes * (lanes + 1)`` counts. A
    miss takes the next slot of its home node's list; node n's CPU takes the last ``split[n][k]`` of its k eligible
    lanes in plan order (miss_keys descending, so the lowest-scored): RAM hits, plus NVMe misses with
    ``cpu_misses``.

    Lanes from ``forced_from`` on (spill, SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES) found no VRAM victim: each is a
    CPU lane whatever the split. A forced hit runs from its RAM slot; a forced miss gets slot -1 and no staging slot,
    since the host reads it into a RAM victim (RamTier::reserve_victims_locked). The split counts only the lanes before
    ``forced_from``. Raises LaneOverflow when forced lanes cannot be CPU lanes (no host lanes, no CPU layer)."""
    if len(experts) > lanes:
        raise ValueError(f"a request has at most {lanes} lanes, got {len(experts)}")
    if len(set(experts)) != len(experts):
        raise ValueError(f"a request names an expert twice: {list(experts)}")
    forced_from = len(experts) if forced_from is None else forced_from
    home = wire_layout(lanes, nodes).home
    slots, hit, taken = [], [], [0] * nodes
    for j, e in enumerate(experts):
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
            continue
        if j >= forced_from:
            slots.append(-1)  # host-placed: read into a RAM victim, never a staging slot
            hit.append(False)
            continue
        node = home(e)
        m = taken[node]
        if m >= lanes or staging[node * lanes + m] < 0:
            raise ValueError(f"a miss lane has no staging slot on node {node}")
        slots.append(staging[node * lanes + m])
        hit.append(False)
        taken[node] += 1
    host_lanes = captured and copy_armed
    can_cpu = host_lanes and cpu_on and cpu_ok
    if forced_from < len(experts) and not can_cpu:
        raise LaneOverflow(f"lanes {forced_from}.. found no victim and cannot be CPU lanes")
    eligible = [j < forced_from and can_cpu and (h or cpu_misses) for j, h in enumerate(hit)]
    take = [0] * nodes
    for node in range(nodes):
        n = sum(1 for e, ok in zip(experts, eligible) if ok and home(e) == node)
        take[node] = split[node * (lanes + 1) + n] if n else 0
    cpu = [j >= forced_from for j in range(len(experts))]
    for j in reversed(range(len(experts))):
        node = home(experts[j])
        if take[node] > 0 and eligible[j]:
            cpu[j] = True
            take[node] -= 1
    copy_ok = host_lanes and hit_copy == "ce" and ce_ok
    kinds = []
    for j, (h, c) in enumerate(zip(hit, cpu)):
        if c:
            kinds.append(LaneKind.HIT_CPU if h else LaneKind.MISS_CPU)
        elif h:
            dst = True if dst_ok is None else dst_ok[j]
            kinds.append(LaneKind.HIT_COPY if copy_ok and dst else LaneKind.HIT_SM)
        else:
            kinds.append(LaneKind.MISS_GPU)
    return kinds, slots


class MapReplica:
    """The device's map bank: ram_slot [rows][experts], staging [rows][nodes * lanes], map_chain and map_applied [rows].

    map_chain starts at 1 and the attach delta has tag 1, so a zero-filled delta record never matches."""

    def __init__(self, rows: int, experts: int, lanes: int, nodes: int = 1):
        self.ram_slot = [[-1] * experts for _ in range(rows)]
        self.staging = [[-1] * (nodes * lanes) for _ in range(rows)]
        self.map_chain = [1] * rows
        self.map_applied = [0] * rows

    def apply_delta(self, row: int, tag: int, staging: Sequence[int], entries: Iterable[tuple[int, int]]) -> None:
        if self.map_applied[row] == tag:
            return
        for e, s in entries:
            self.ram_slot[row][e] = s
        self.staging[row] = list(staging)
        self.map_applied[row] = tag

    def apply_bulk(self, entries: Iterable[tuple[int, int, int]]) -> None:
        for row, e, s in entries:
            self.ram_slot[row][e] = s
