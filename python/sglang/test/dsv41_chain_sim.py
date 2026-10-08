"""The device's side of the slot-map protocol, in Python, for the CPU tests (analysis/dsv41-drive/LEASE_PROTOCOL.md).

This drives the REAL C++ service through the request page, the completion block and the map delta block, as the post
kernel, S and CW do. It is a stand-in for the CUDA kernels and is not evidence about them: it types lanes with
``ram_slot_map.type_lanes``, the reference the kernels are tested against on the GPU.
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import (
    RECORD_FLAG_CAPTURED,
    WORDS,
    hot_record_bytes,
    page_word,
    piece_word,
)
from sglang.srt.layers.moe.ram_slot_map import LaneKind, MapReplica, type_lanes

ALL_PIECES = 0xFF  # a PieceMask word's bits once every piece of the row is published


def _i32(tensor: torch.Tensor, offset: int, count: int = 1) -> torch.Tensor:
    return tensor[offset : offset + 4 * count].view(torch.int32)


def _i16(tensor: torch.Tensor, offset: int, count: int = 1) -> torch.Tensor:
    return tensor[offset : offset + 2 * count].view(torch.int16)


def _u16(tensor: torch.Tensor, offset: int) -> torch.Tensor:
    return tensor[offset : offset + 2].view(torch.int16)


def _signed32(value: int) -> int:
    return value - (1 << 32) if value >= (1 << 31) else value


@dataclass
class SimRequest:
    seq: int
    gen: int
    idx: int
    row: int
    experts: tuple[int, ...]
    kinds: list[LaneKind]
    slots: list[int]
    dst: list[int]
    chain: int


class ChainSim:
    def __init__(self, host, page, slabs, *, epoch: int = 0):
        # A weak reference: a test may collect the host and keep posting into its page, as a device would.
        self._host = weakref.ref(host)
        self.page, self.slabs, self.epoch = page, slabs, epoch
        self.block, self.hot_page = host.lease_block, host.hot_page
        self.layers, self.experts = host.layers, host.experts
        self.wire = host.wire
        self.replica = MapReplica(host.layers, host.experts, self.wire.lanes, self.wire.nodes)

    @property
    def host(self):
        host = self._host()
        if host is None:
            raise RuntimeError("the host was collected")
        return host

    # ---- words of the blocks ----

    def _u64(self, offset: int) -> torch.Tensor:
        return self.block[offset : offset + 8].view(torch.int64)

    def read_u64(self, offset: int) -> int:
        return int(self._u64(offset)[0]) & 0xFFFFFFFFFFFFFFFF

    def delta(self, row: int) -> tuple[int, list[int], list[tuple[int, int]]]:
        """Row ``row``'s map delta record: (tag, staging, entries), as the post reads it; staging is every node's list,
        node-major."""
        w = self.wire
        base = w.lease_block_bytes + row * w.delta_stride
        f = w.delta_fields
        tag = self.read_u64(base + f["tag"])
        count = int(_i32(self.block, base + f["count"])[0])
        staging = (
            self.block[base + f["staging"] : base + f["staging"] + 2 * w.nodes * w.lanes].view(torch.int16).tolist()
        )
        entries = base + f["entries"]
        flat = self.block[entries : entries + 4 * w.delta_max_entries].view(torch.int16).tolist()
        return tag, staging, [(flat[2 * i], flat[2 * i + 1]) for i in range(count)]

    def staging(self, row: int) -> list[int]:
        return self.delta(row)[1]

    def last_chain(self, row: int) -> int:
        return self.replica.map_chain[row]

    def copy_armed(self) -> bool:
        return int(_i32(self.block, self.wire.copy_armed)[0]) != 0

    def split(self) -> list[int]:
        """Every node's split table, node-major."""
        w = self.wire
        return sum((_i32(self.block, w.split + n * w.split_stride, w.lanes + 1).tolist() for n in range(w.nodes)), [])

    def piece_word(self, req: SimRequest, lane: int) -> int:
        w = self.wire
        return self.read_u64(w.piece_mask + (req.idx * w.lanes + lane) * w.piece_mask_line_bytes)

    def copy_done(self, req: SimRequest) -> int:
        return self.read_u64(self.wire.copy_done + req.idx * self.wire.copy_done_bytes)

    def copy_gate(self) -> int:
        return int(_i32(self.block, self.wire.copy_gate)[0]) & 0xFFFFFFFF

    def _set_gate(self, word: int) -> None:
        _i32(self.block, self.wire.copy_gate)[0] = _signed32(word)

    def read_slot(self, row: int, slot: int) -> dict[str, torch.Tensor]:
        return {name: tensor[slot].clone() for name, tensor in self.slabs[row].items()}

    # ---- the device's steps ----

    def apply_pending(self, row: int) -> bool:
        """The post's delta apply: False when the host has not yet published the row's delta (the post would wait)."""
        tag, staging, entries = self.delta(row)
        if tag != self.replica.map_chain[row]:
            return False
        self.replica.apply_delta(row, tag, staging, entries)
        return True

    def apply_bulk_like_device(self, bulk) -> None:
        """map_bulk_apply: every row's pending decode delta first, then the bulk entries."""
        for row in range(self.layers):
            self.apply_pending(row)
        self.replica.apply_bulk([tuple(int(v) for v in entry) for entry in bulk])

    def sync_bulk(self) -> None:
        """Exl3RamMissService.after_host_use: take the host's bulk delta (the caller owns the tier: no thread, or
        paused) and apply it as map_bulk_apply does."""
        self.apply_bulk_like_device(self.host.take_bulk_delta().tolist())

    def post(
        self,
        row: int,
        experts: Sequence[int],
        *,
        protect: Optional[Sequence[int]] = None,
        dst: Optional[Sequence[int]] = None,
        captured: bool = False,
        hit_copy: str = "ce",
        cpu_on: bool = False,
        cpu_misses: bool = False,
        cpu_ok: bool = True,
        ce_ok: bool = True,
        weights: Optional[Sequence[float]] = None,
        hot: Sequence[int] = (),
        hot_seq: Optional[int] = None,
        kinds: Optional[Sequence[int]] = None,
        slots: Optional[Sequence[int]] = None,
        chain: Optional[int] = None,
        forced_from: Optional[int] = None,
    ) -> SimRequest:
        """The post kernel: apply the row's pending delta, type the lanes from the replica, then the hot record, the
        record and demand_head. ``kinds``/``slots`` override the typing and ``chain`` the map-chain number (a malformed post); ``hot`` is the VRAM hot
        set the hot record carries, ``hot_seq`` the seq it names (a stale record).
        ``forced_from`` types the lanes from it on as spill does (type_lanes)."""
        experts = tuple(int(e) for e in experts)
        if experts and not self.apply_pending(row):
            raise AssertionError(f"row {row}: the host has not published delta {self.replica.map_chain[row]}")
        if kinds is None:
            typed, slot_list = type_lanes(
                experts, self.replica.ram_slot[row], self.replica.staging[row], self.split(),
                lanes=self.wire.lanes, nodes=self.wire.nodes, captured=captured, copy_armed=self.copy_armed(), hit_copy=hit_copy, cpu_on=cpu_on,
                cpu_misses=cpu_misses, cpu_ok=cpu_ok, ce_ok=ce_ok,
                forced_from=forced_from,
            )
        else:
            # Raw ints, not LaneKind: a malformed post may name a kind the device never writes.
            typed, slot_list = [int(k) for k in kinds], list(slots)
        forged, chain = chain, 0
        if any(k in (LaneKind.MISS_GPU, LaneKind.MISS_CPU) for k in typed):
            self.replica.map_chain[row] += 1
            chain = self.replica.map_chain[row]
        if forged is not None:
            chain = forged
        head = page_word(self.page, "demand_head")
        seq = (head + 1) & 0xFFFFFFFF
        if seq == 0:
            seq = 1
            self.epoch += 1
        gen = (self.epoch << 32) | seq
        w = self.wire
        idx = (seq - 1) % w.demand_records
        dst = list(range(len(experts))) if dst is None else list(dst)
        weights = [1.0] * len(experts) if weights is None else list(weights)
        protect = list(dict.fromkeys(experts)) if protect is None else list(protect)
        self._write_hot(seq, hot, hot_seq)
        record = w.demand_ring + idx * w.record_bytes
        f, n = w.record_fields, w.lanes
        count = len(experts)
        ids = list(dict.fromkeys(int(e) for e in protect))[:n]
        assert count <= n, "a record carries at most the build's lanes"

        def padded(values, fill, dtype):
            return torch.tensor(list(values)[:count] + [fill] * (n - count), dtype=dtype)

        _i32(self.page, record + f["seq"])[0] = 0
        _u16(self.page, record + f["row"])[0] = row
        if w.packed_counts:
            self.page[record + f["counts"]] = count | len(ids) << 4
        else:
            self.page[record + f["counts"]] = count
            self.page[record + w.protect_count] = len(ids)
        self.page[record + f["flags"]] = RECORD_FLAG_CAPTURED if captured else 0
        self.page[record + f["chain"] : record + f["chain"] + 8].view(torch.int64)[0] = chain
        _i32(self.page, record + f["epoch"])[0] = _signed32(self.epoch & 0xFFFFFFFF)
        for word in range(w.kind_words):
            lanes = typed[8 * word : 8 * word + 8]
            _i32(self.page, record + f["kinds"] + 4 * word)[0] = _signed32(
                sum((int(kind) & 0xF) << 4 * j for j, kind in enumerate(lanes))
            )
        _i16(self.page, record + f["protect"], n)[:] = torch.tensor(ids + [-1] * (n - len(ids)), dtype=torch.int16)
        _i16(self.page, record + f["lane_expert"], n)[:] = padded(experts, -1, torch.int16)
        _i16(self.page, record + f["lane_slot"], n)[:] = padded(slot_list, -1, torch.int16)
        _i16(self.page, record + f["lane_dst"], n)[:] = padded(dst, -1, torch.int16)
        lane_weight = record + f["lane_weight"]
        self.page[lane_weight : lane_weight + 4 * n].view(torch.float32)[:] = padded(weights, 0.0, torch.float32)
        _i32(self.page, record + f["seq"])[0] = _signed32(seq)
        _i32(self.page, WORDS["demand_head"])[0] = _signed32(seq)
        return SimRequest(seq, gen, idx, row, experts, list(typed), slot_list, dst, chain)

    def _write_hot(self, seq: int, hot: Sequence[int], hot_seq: Optional[int]) -> None:
        hot_page = self.hot_page
        if hot_page is None:
            return
        experts = self.experts
        stride = hot_record_bytes(experts)
        record = hot_page[(seq - 1) % self.wire.demand_records * stride :][:stride]
        _i32(record, 0)[0] = 0
        bitmap = record[8 : 8 + (experts + 7) // 8]
        bitmap.zero_()
        for expert in hot:
            bitmap[expert // 8] = int(bitmap[expert // 8]) | (1 << (expert % 8))
        _i32(record, 0)[0] = _signed32(seq if hot_seq is None else hot_seq)

    def served(self, req: SimRequest) -> bool:
        """S's condition: every MISS_GPU lane's PieceMask word carries G with every piece."""
        return all(
            self.piece_word(req, j) == piece_word(req.gen, ALL_PIECES)
            for j, kind in enumerate(req.kinds)
            if kind == LaneKind.MISS_GPU
        )

    def wait_served(self, req: SimRequest, timeout_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self.served(req):
            if time.monotonic() > deadline:
                return False
            time.sleep(20e-6)
        return True

    def wait_handled(self, req: SimRequest, timeout_s: float = 1.0) -> bool:
        """The service has finished the record (handled_through reached its seq): for records nothing on the device
        waits for, such as SM-only hits."""
        deadline = time.monotonic() + timeout_s
        while ((self.host.handled_through() - req.seq) & 0xFFFFFFFF) >= (1 << 31):
            if time.monotonic() > deadline:
                return False
            time.sleep(20e-6)
        return True

    def close_copy_gate(self, req: SimRequest) -> None:
        """CW's close of the gate for a request with host lanes, without waiting."""
        self._set_gate(lease.gate_word(req.seq, "closed"))

    def needs_copy_wait(self, req: SimRequest) -> bool:
        return any(k in (LaneKind.HIT_COPY, LaneKind.HIT_CPU, LaneKind.MISS_CPU) for k in req.kinds)

    def copy_wait(self, req: SimRequest, timeout_s: float = 2.0) -> bool:
        """CW, the stream wait and CC: close the gate for G when a lane is the host's, then wait for CopyDone == G."""
        if not self.needs_copy_wait(req):
            return True
        self._set_gate(lease.gate_word(req.seq, "closed"))
        deadline = time.monotonic() + timeout_s
        while self.copy_done(req) != req.gen:
            if time.monotonic() > deadline:
                return False
            time.sleep(20e-6)
        return True
