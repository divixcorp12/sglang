"""Reference model of the device slot map (analysis/dsv41-drive/LEASE_PROTOCOL.md): lane typing in the post and
delta application. The CUDA post kernel and map_bulk_apply transcribe it; the parity tests compare the two."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Optional, Sequence

LANES = 8


class LaneKind(IntEnum):
    """lease_layout.h kKind*: what moves a lane's bytes and what the device waits on."""

    HIT_COPY = 1
    HIT_SM = 2
    HIT_CPU = 3
    MISS_GPU = 4
    MISS_CPU = 5


def type_lanes(
    experts: Sequence[int],
    ram_slot: Sequence[int],
    staging: Sequence[int],
    split: Sequence[int],
    *,
    captured: bool,
    copy_armed: bool,
    hit_copy: str,
    cpu_on: bool,
    cpu_misses: bool,
    ce_ok: bool = True,
    cpu_ok: bool = True,
    dst_ok: Optional[Sequence[bool]] = None,
) -> tuple[list[LaneKind], list[int]]:
    """Each lane's kind and source slot: its RAM slot for a hit, the m-th staging slot for the m-th miss.

    The CPU takes the last split[n] of the n eligible lanes in plan order (miss_keys descending, so the
    lowest-scored): RAM hits, plus NVMe misses with ``cpu_misses``."""
    if len(experts) > LANES:
        raise ValueError(f"a request has at most {LANES} lanes, got {len(experts)}")
    if len(set(experts)) != len(experts):
        raise ValueError(f"a request names an expert twice: {list(experts)}")
    slots, hit, m = [], [], 0
    for e in experts:
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
            continue
        if m >= LANES or staging[m] < 0:
            raise ValueError("a miss lane has no staging slot")
        slots.append(staging[m])
        hit.append(False)
        m += 1
    host_lanes = captured and copy_armed
    eligible = [host_lanes and cpu_on and cpu_ok and (h or cpu_misses) for h in hit]
    n = sum(eligible)
    take = split[n] if n else 0
    cpu = [False] * len(experts)
    for j in reversed(range(len(experts))):
        if take == 0:
            break
        if eligible[j]:
            cpu[j] = True
            take -= 1
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
    """The device's map bank: ram_slot [rows][experts], staging [rows][LANES], map_chain and map_applied [rows].

    map_chain starts at 1 and the attach delta has tag 1, so a zero-filled delta record never matches."""

    def __init__(self, rows: int, experts: int):
        self.ram_slot = [[-1] * experts for _ in range(rows)]
        self.staging = [[-1] * LANES for _ in range(rows)]
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
