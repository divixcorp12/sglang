"""The device's side of the lease protocol, in Python, for the CPU tests (LEASE_PROTOCOL.md sections 7.3 and 7.4).

This drives the REAL C++ service through the request page and the lease block. It is a stand-in for the CUDA
kernels and is not evidence about them: it is written from the same specification, so a property of the device
(for example "a skipped copy emits no acknowledgement") holds here by construction and only a GPU test can show
it of the kernels. Device stores to the acknowledgement and terminal words go through an outbox, so a test can
deliver them in any order across words, as the model does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import DEMAND_RECORDS, page_word, piece_word, sim_post, sim_wait

DEMAND_TAG = 1  # the tag of a LaneRequest generation word
ALL_PIECES = 0xFF  # a PieceMask word's bits once every piece of the row is published


@dataclass
class SimRequest:
    seq: int
    gen: int
    idx: int
    row: int
    lanes: tuple[int, ...]


@dataclass
class SimWait:
    status: int  # sim_wait: 1 served, 2 failed, 0 timed out, 3 fatal already raised
    go: int  # the copy count the wait kernel would commit; 0 fails closed
    ctx: list = field(default_factory=list)  # per lane (slot, slot_generation) when go > 0
    reason: str = ""


class LeaseSim:
    def __init__(self, host, page, slabs, *, epoch: int = 0):
        self.host, self.page, self.slabs, self.epoch = host, page, slabs, epoch
        self.block = host.lease_block
        self.layout = host.lease_layout
        self.outbox: list[tuple[str, int, tuple]] = []

    # ---- words of the block ----

    def _u64(self, offset: int) -> torch.Tensor:
        return self.block[offset : offset + 8].view(torch.int64)

    def _i32(self, offset: int, count: int = 1) -> torch.Tensor:
        return self.block[offset : offset + 4 * count].view(torch.int32)

    def _d(self, offset: int) -> int:
        return self.layout.d_offset + offset

    def read_u64(self, offset: int) -> int:
        return int(self._u64(offset)[0]) & 0xFFFFFFFFFFFFFFFF

    def write_u64(self, offset: int, value: int) -> None:
        self._u64(offset)[0] = value - (1 << 64) if value >= (1 << 63) else value

    def row_result(self, req: SimRequest, lane: int) -> dict:
        """The service's RowResult for a lane, as the wait kernel would read it after acquiring ``ready``."""
        base = lease.ROW_RESULT + (req.idx * lease.LANES + lane) * lease.ROW_RESULT_BYTES
        tag, gen = lease.untag(self.read_u64(base))
        fields = lease.ROW_RESULT_FIELDS
        return {
            "tag": tag,
            "gen": gen,
            "slot_generation": int(self._i32(base + fields["slot_generation"])[0]) & 0xFFFFFFFF,
            "host_slot": int(self._i32(base + fields["host_slot"])[0]),
            "expert": int(self._i32(base + fields["expert"])[0]),
        }

    def piece_word(self, req: SimRequest, lane: int) -> int:
        """The lane's PieceMask word (area P): ``generation << 8 | bits``, written only by the service."""
        return self.read_u64(self.layout.piece_offset + (req.idx * lease.LANES + lane) * lease.PIECE_MASK_LINE_BYTES)

    def copy_done(self, req: SimRequest) -> tuple[int, int, int]:
        """(tag, generation, lane mask) of the request's CopyDone word (area C), as the copy wait would read it."""
        base = self.layout.copy_offset + req.idx * lease.COPY_DONE_BYTES
        tag, gen = lease.untag(self.read_u64(base + lease.COPY_DONE_FIELDS["gen"]))
        return tag, gen, int(self._i32(base + lease.COPY_DONE_FIELDS["mask"])[0]) & 0xFFFFFFFF

    def ack_offset(self, req: SimRequest, lane: int) -> int:
        return self._d(lease.LANE_ACK + (req.idx * lease.LANES + lane) * lease.LANE_ACK_BYTES)

    def terminal_offset(self, req: SimRequest) -> int:
        return self._d(lease.TERMINAL + req.idx * lease.TERMINAL_BYTES)

    # ---- the device's steps ----

    def post(
        self,
        row: int,
        lanes: Sequence[int],
        *,
        armed: bool = True,
        write_lane_request: bool = True,
        need: Optional[Sequence[int]] = None,
        protect: Optional[Sequence[int]] = None,
        dst: Optional[Sequence[int]] = None,
        copy_engine: bool = False,
    ) -> SimRequest:
        """``need`` and ``protect`` override what the post kernel would put in the record (its routes, not its lanes);
        ``dst`` are the lanes' destination slots and ``copy_engine`` the LaneRequest flag (LEASE_PROTOCOL.md 7.6)."""
        head = page_word(self.page, "demand_head")
        seq = (head + 1) & 0xFFFFFFFF
        if seq == 0:
            seq = 1
            self.epoch += 1
        gen = (self.epoch << 32) | seq
        idx = (seq - 1) % DEMAND_RECORDS
        lanes = tuple(int(e) for e in lanes)
        if write_lane_request:
            base = self._d(lease.LANE_REQUEST + idx * lease.LANE_REQUEST_BYTES)
            fields = lease.LANE_REQUEST_FIELDS
            self.write_u64(base + fields["gen"], 0)  # invalidate first, as the seqlock writer does
            self._i32(base + fields["count"], 2)[:] = torch.tensor([len(lanes), row], dtype=torch.int32)
            padded = list(lanes) + [-1] * (lease.LANES - len(lanes))
            self._i32(base + fields["expert"], lease.LANES)[:] = torch.tensor(padded, dtype=torch.int32)
            slots = list(dst or []) + [-1] * (lease.LANES - len(dst or []))
            self._i32(base + fields["dst_slot"], lease.LANES)[:] = torch.tensor(slots, dtype=torch.int32)
            flags = lease.LANE_REQUEST_FLAG_COPY_ENGINE if copy_engine else 0
            self._i32(base + fields["flags"])[:] = torch.tensor([flags], dtype=torch.int32)
            self.write_u64(base + fields["gen"], lease.tagged(DEMAND_TAG, gen))
        mapping = self.host.mapping(row)
        distinct = list(dict.fromkeys(lanes))
        need = [e for e in distinct if mapping[e] < 0] if need is None else list(need)
        protect = distinct if protect is None else list(protect)
        assert sim_post(self.page, row, need=need, protect=protect, armed=armed, lanes=len(lanes)) == seq
        return SimRequest(seq, gen, idx, row, lanes)

    def wait(self, req: SimRequest, timeout_s: float = 1.0, *, publish_terminal: bool = True) -> SimWait:
        """The wait kernel's decisions: validate every lane's row result; commit or fail closed.

        A lane granted under tag LOADING (piece streaming) is accepted as the stream kernel judges it once it has
        acquired a served ``demand_done``: its PieceMask word, re-read now, must carry every piece under the
        request's generation (piece-streaming plan section 5); anything less fails closed, as an identity failure."""
        status = sim_wait(self.page, req.seq, timeout_s)
        if status != 1:
            return self._fail(req, status, f"status {status}", publish_terminal)
        ctx = []
        for lane, expert in enumerate(req.lanes):
            result = self.row_result(req, lane)
            loaded = result["tag"] == lease.LOADING and self.piece_word(req, lane) == piece_word(req.gen, ALL_PIECES)
            if result["gen"] != req.gen or not (result["tag"] == lease.READY or loaded) or result["expert"] != expert or result["host_slot"] < 0:
                return self._fail(req, status, f"lane {lane}: {result}", publish_terminal)
            ctx.append((result["host_slot"], result["slot_generation"]))
        return SimWait(status, len(req.lanes), ctx)

    def _fail(self, req: SimRequest, status: int, reason: str, publish_terminal: bool) -> SimWait:
        if publish_terminal and req.lanes:
            self.terminal(req, (1 << len(req.lanes)) - 1)
        return SimWait(status, 0, [], reason)

    def copy(self, req: SimRequest, waited: SimWait) -> list[dict]:
        """What the copy kernel would deliver: the bytes of each committed lane's leased slot, read now."""
        return [
            {name: tensor[slot].clone() for name, tensor in self.slabs[req.row].items()} for slot, _ in waited.ctx[: waited.go]
        ]

    def ack(self, req: SimRequest, waited: SimWait, *, lanes: Optional[Sequence[int]] = None) -> None:
        """The acknowledgement kernel: for each committed lane re-read SlotGen and publish CONSUMED or VIOLATED."""
        generations = self.host.mapped_slot_generations(req.row)
        for lane in range(waited.go) if lanes is None else lanes:
            slot, granted = waited.ctx[lane]
            outcome = lease.CONSUMED if generations[slot] == granted else lease.VIOLATED
            self.outbox.append(("ack", lane, (self.ack_offset(req, lane), lease.tagged(outcome, req.gen))))

    def terminal(self, req: SimRequest, mask: int, reason: int = 1) -> None:
        self.outbox.append(("term", 0, (self.terminal_offset(req), mask, reason, lease.tagged(1, req.gen))))

    def deliver(self, order: Optional[Sequence[int]] = None) -> None:
        """Land pending device stores on the block; ``order`` picks which, in what order (default: all, FIFO)."""
        picked = list(range(len(self.outbox))) if order is None else list(order)
        for i in picked:
            kind, _, payload = self.outbox[i]
            if kind == "ack":
                self.write_u64(payload[0], payload[1])
            else:
                offset, mask, reason, word = payload
                self._i32(offset, 2)[:] = torch.tensor([mask, reason], dtype=torch.int32)
                self.write_u64(offset + lease.TERMINAL_FIELDS["gen"], word)
        remaining = [item for j, item in enumerate(self.outbox) if j not in set(picked)]
        self.outbox = remaining
