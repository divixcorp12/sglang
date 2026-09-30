"""The device's side of the lease protocol, in Python, for the CPU tests (analysis/dsv41-drive/LEASE_PROTOCOL.md).

This drives the REAL C++ service through the request page and the lease block. It is a stand-in for the CUDA
kernels and is not evidence about them: it is written from the same specification, so a property of the device holds
here by construction and only a GPU test can show it of the kernels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import (
    DEMAND_RECORDS,
    DEMAND_RING,
    MAX_IDS,
    RECORD_BYTES,
    RECORD_FIELDS,
    WORDS,
    page_word,
    piece_word,
)

ALL_PIECES = 0xFF  # a PieceMask word's bits once every piece of the row is published


def _i32(tensor: torch.Tensor, offset: int, count: int = 1) -> torch.Tensor:
    return tensor[offset : offset + 4 * count].view(torch.int32)


def _u16(tensor: torch.Tensor, offset: int) -> torch.Tensor:
    return tensor[offset : offset + 2].view(torch.int16)


def post_record(page: torch.Tensor, row: int, protect: Sequence[int], *, armed: bool) -> int:
    """Post a demand record as the post kernel does (seq 0, payload, seq, then demand_head); returns its seq.

    Plain tensor stores in program order: x86 keeps them in order, which is the seqlock order the service relies on."""
    seq = (page_word(page, "demand_head") + 1) & 0xFFFFFFFF
    if seq == 0:
        seq = 1
    record = DEMAND_RING + (seq - 1) % DEMAND_RECORDS * RECORD_BYTES
    ids = list(dict.fromkeys(int(e) for e in protect))[:MAX_IDS]
    _i32(page, record + RECORD_FIELDS["seq"])[0] = 0
    _u16(page, record + RECORD_FIELDS["row"])[0] = row
    _u16(page, record + RECORD_FIELDS["protect_count"])[0] = len(ids)
    _u16(page, record + RECORD_FIELDS["armed"])[0] = int(armed)
    _i32(page, record + RECORD_FIELDS["protect"], MAX_IDS)[:] = torch.tensor(ids + [-1] * (MAX_IDS - len(ids)))
    _i32(page, record + RECORD_FIELDS["seq"])[0] = seq - (1 << 32) if seq >= (1 << 31) else seq
    _i32(page, WORDS["demand_head"])[0] = seq - (1 << 32) if seq >= (1 << 31) else seq
    return seq


def served(page: torch.Tensor, seq: int) -> bool:
    """demand_done reached ``seq`` (cyclically): the service served it. There is no failed state."""
    return ((page_word(page, "demand_done") - seq) & 0xFFFFFFFF) < (1 << 31)


def wait_served(page: torch.Tensor, seq: int, timeout_s: float) -> bool:
    """Poll demand_done as S does; False at the deadline (S would trap there)."""
    deadline = time.monotonic() + timeout_s
    while not served(page, seq):
        if time.monotonic() > deadline:
            return False
        time.sleep(20e-6)
    return True


@dataclass
class SimRequest:
    seq: int
    gen: int
    idx: int
    row: int
    lanes: tuple[int, ...]


@dataclass
class SimWait:
    served: bool
    go: int  # lanes the chain copies: every lane once served, else 0
    ctx: list = field(default_factory=list)  # per lane its (tag, host_slot) when served


class LeaseSim:
    def __init__(self, host, page, slabs, *, epoch: int = 0):
        self.host, self.page, self.slabs, self.epoch = host, page, slabs, epoch
        self.block = host.lease_block

    # ---- words of the block ----

    def _u64(self, offset: int) -> torch.Tensor:
        return self.block[offset : offset + 8].view(torch.int64)

    def read_u64(self, offset: int) -> int:
        return int(self._u64(offset)[0]) & 0xFFFFFFFFFFFFFFFF

    def write_u64(self, offset: int, value: int) -> None:
        self._u64(offset)[0] = value - (1 << 64) if value >= (1 << 63) else value

    def row_result(self, req: SimRequest, lane: int) -> dict:
        """The service's RowResult for a lane, as W1 or S read it after acquiring ``ready``."""
        base = lease.ROW_RESULT + (req.idx * lease.LANES + lane) * lease.ROW_RESULT_BYTES
        tag, gen = lease.untag(self.read_u64(base))
        return {"tag": tag, "gen": gen, "host_slot": int(_i32(self.block, base + lease.ROW_RESULT_FIELDS["host_slot"])[0])}

    def piece_word(self, req: SimRequest, lane: int) -> int:
        """The lane's PieceMask word (area P): ``generation << 8 | bits``, written only by the service."""
        return self.read_u64(lease.PIECE_MASK + (req.idx * lease.LANES + lane) * lease.PIECE_MASK_LINE_BYTES)

    def copy_done(self, req: SimRequest) -> int:
        """The generation in the request's CopyDone word (area C), as CC reads it."""
        return self.read_u64(lease.COPY_DONE + req.idx * lease.COPY_DONE_BYTES)

    def copy_gate(self) -> int:
        """Area C's copy-wait gate (lease.gate_word), the word the decode stream's cuStreamWaitValue32 waits on."""
        return int(_i32(self.block, lease.COPY_GATE)[0]) & 0xFFFFFFFF

    def close_copy_gate(self, req: SimRequest) -> None:
        """CW's close of the gate for a request with COPYING or CPU lanes."""
        self._set_gate(lease.gate_word(req.seq, "closed"))

    def open_copy_gate(self, req: SimRequest) -> None:
        """CW's own open of G's gate, when CopyDone was already there after its close."""
        self._set_gate(lease.gate_word(req.seq, "open"))

    def _set_gate(self, word: int) -> None:
        _i32(self.block, lease.COPY_GATE)[0] = word - (1 << 32) if word >= (1 << 31) else word

    def dst_slot(self, req: SimRequest, lane: int) -> int:
        """The lane's destination slot as the post kernel wrote it into the LaneRequest."""
        base = lease.LANE_REQUEST + req.idx * lease.LANE_REQUEST_BYTES + lease.LANE_REQUEST_FIELDS["dst_slot"]
        return int(_i32(self.block, base + 4 * lane)[0])

    def done_word(self, req: SimRequest) -> int:
        return self.read_u64(lease.DONE + req.idx * lease.DONE_BYTES)

    def done(self, req: SimRequest, generation: Optional[int] = None) -> None:
        """CW's Done: no kernel of the request reads one of its leased slots any more."""
        self.write_u64(lease.DONE + req.idx * lease.DONE_BYTES, req.gen if generation is None else generation)

    def sm_fetch(self, req: SimRequest, dst: dict, names: Sequence[str]) -> None:
        """CW's SM half (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES): copy ``names`` of every COPYING lane from its
        leased slot into its destination row. The caller publishes Done after it."""
        for lane in range(len(req.lanes)):
            result = self.row_result(req, lane)
            if result["tag"] != lease.COPYING or result["gen"] != req.gen:
                continue
            for name in names:
                dst[name][self.dst_slot(req, lane)].copy_(self.slabs[req.row][name][result["host_slot"]])

    # ---- the device's steps ----

    def post(
        self,
        row: int,
        lanes: Sequence[int],
        *,
        write_lane_request: bool = True,
        protect: Optional[Sequence[int]] = None,
        dst: Optional[Sequence[int]] = None,
        captured: bool = False,
        cpu_weights: Optional[Sequence[float]] = None,
    ) -> SimRequest:
        """The post kernel: the LaneRequest, then the demand record (armed iff there are lanes) and demand_head.
        ``protect`` overrides the record's protect ids (the post writes the routes, not the lanes); ``dst`` are the
        lanes' destination slots, ``captured`` the LaneRequest flag, ``cpu_weights`` the lanes' routing weights."""
        head = page_word(self.page, "demand_head")
        seq = (head + 1) & 0xFFFFFFFF
        if seq == 0:
            seq = 1
            self.epoch += 1
        gen = (self.epoch << 32) | seq
        idx = (seq - 1) % DEMAND_RECORDS
        lanes = tuple(int(e) for e in lanes)
        if write_lane_request:
            base = lease.LANE_REQUEST + idx * lease.LANE_REQUEST_BYTES
            fields = lease.LANE_REQUEST_FIELDS
            self.write_u64(base + fields["gen"], 0)  # invalidate first, as the seqlock writer does
            _i32(self.block, base + fields["count"])[0] = len(lanes)
            _i32(self.block, base + fields["flags"])[0] = lease.LANE_REQUEST_FLAG_CAPTURED if captured else 0
            padded = list(lanes) + [-1] * (lease.LANES - len(lanes))
            _i32(self.block, base + fields["expert"], lease.LANES)[:] = torch.tensor(padded, dtype=torch.int32)
            slots = list(dst or []) + [-1] * (lease.LANES - len(dst or []))
            _i32(self.block, base + fields["dst_slot"], lease.LANES)[:] = torch.tensor(slots, dtype=torch.int32)
            weights = list(cpu_weights or []) + [0.0] * (lease.LANES - len(cpu_weights or []))
            _i32(self.block, base + fields["weight"], lease.LANES)[:] = torch.tensor(weights, dtype=torch.float32).view(
                torch.int32
            )
            self.write_u64(base + fields["gen"], gen)
        protect = list(dict.fromkeys(lanes)) if protect is None else list(protect)
        assert post_record(self.page, row, protect, armed=bool(lanes)) == seq
        return SimRequest(seq, gen, idx, row, lanes)

    def wait(self, req: SimRequest, timeout_s: float = 1.0) -> SimWait:
        """S's judgement once demand_done reached the request: every lane is published for G, a LOADING lane with
        every piece. Raises on a published request that breaks that (S would trap); a timeout is ``served=False``."""
        if not wait_served(self.page, req.seq, timeout_s):
            return SimWait(False, 0, [])
        ctx = []
        for lane in range(len(req.lanes)):
            result = self.row_result(req, lane)
            if result["gen"] != req.gen or result["tag"] == 0:
                raise AssertionError(f"lane {lane} of a served request is not published: {result}")
            if result["tag"] == lease.LOADING and self.piece_word(req, lane) != piece_word(req.gen, ALL_PIECES):
                raise AssertionError(f"lane {lane} of a served request is missing pieces")
            ctx.append((result["tag"], result["host_slot"]))
        return SimWait(True, len(req.lanes), ctx)

    def copy(self, req: SimRequest, waited: SimWait) -> list[dict]:
        """What C1 and S would deliver: the bytes of each lane's leased slot, read now."""
        return [
            {name: tensor[slot].clone() for name, tensor in self.slabs[req.row].items()} for _, slot in waited.ctx[: waited.go]
        ]
