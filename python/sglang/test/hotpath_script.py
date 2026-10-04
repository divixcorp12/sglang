"""A scripted RAM-miss scenario in the production configuration (row images, the copy engine on the CPU backend, GPU
hot), driven in pump mode, snapshotting everything the device and the
eager callers can observe after each step. Plan 2026-09-29-hotpath-zero-overhead Tasks 2 and 4."""

from __future__ import annotations

import hashlib

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import (
    HOT_RECORDS,
    ExpertStreamHost,
    hot_record_bytes,
    new_page,
    page_word,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

# Seven slots per row, three of them staging: four mappable rows, so the script evicts from its fourth post on.
CAPACITY = 7
STAGING = 3
LAYERS = 2
EXPERTS = 8
DST_ROWS = 6
FUNCTIONAL = (
    "served", "touch_only", "rows_read", "read_errors", "overruns",
    "evictions", "ram_insert_skipped", "no_victim", "version",
)


def build_host(tmp_path, *, variant=None, threaded=False, copy_spin_us=200):
    s = ram_miss_setup(tmp_path, capacity=CAPACITY, layers=LAYERS, experts=EXPERTS, row_images=True,
                       mirror_weights=(1.0, 1.0))
    page = new_page(pin=False, wire=wire_layout(8))
    hot = torch.zeros(HOT_RECORDS * hot_record_bytes(EXPERTS), dtype=torch.uint8)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32),
                            hot_page=hot, variant=variant)
    host.enable_copy_engine(-1, spin_us=copy_spin_us)
    dst = {}
    for row in range(LAYERS):
        dst[row] = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[row].items()}
        table = torch.tensor(
            [[t.data_ptr(), dst[row][n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[row].items()],
            dtype=torch.int64)
        host.set_copy_table(row, table, DST_ROWS)
    host.reserve_staging(STAGING)
    host.arm_copy_engine()
    return s, page, host, ChainSim(host, page, s.slabs), dst


# (kind, row, lanes, extra): "post" posts and serves a request whose lanes are the experts listed; "copy" releases every
# held copy mark; "hot" sets the row's VRAM-hot experts, which every later post of that row writes into its sidecar
# record. Every step ends with pump() until idle.
SCRIPT = [
    ("post", 0, [0, 1], {}),             # two misses: read into staging slots, the delta maps them
    ("post", 0, [0, 2], {"captured": True, "dst": [0, 1]}),  # hit 0 through the copy engine, miss 2
    ("copy", 0, None, {}),
    ("post", 0, [3, 4], {}),             # fills the four mappable rows and evicts one
    ("post", 0, [5, 6], {}),             # evicts two LRU rows
    ("post", 1, [7], {"armed": False}),  # a touch-only record on row 1 (7 is its protect id, not a lane)
    ("hot", 0, [5], {}),                 # from here expert 5 of row 0 is VRAM-hot: never a victim
    ("post", 0, [1, 2, 3], {}),
    ("post", 1, [0, 1, 2], {}),
]


def _drain(host):
    for _ in range(64):
        if host.pump() == 0:
            return
    raise AssertionError("pump never went idle")


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _records(page, host) -> list[dict]:
    """Every posted demand record still in the ring (seq 1..demand_head, the last ``demand_records`` of them): its sequence
    word as the ring holds it, and whether the service has handled it."""
    head = page_word(page, "demand_head")
    handled = host.handled_through()
    w = host.wire
    out = []
    for seq in range(max(1, head - w.demand_records + 1), head + 1):
        base = w.demand_ring + (seq - 1) % w.demand_records * w.record_bytes
        out.append({
            "seq": seq,
            "ring_seq": int(page[base : base + 4].view(torch.int32)[0]) & 0xFFFFFFFF,
            "served": seq <= handled,
        })
    return out


def completed_copies(s, sim, req, dst) -> list[dict]:
    """The copy-engine lanes of ``req`` whose CopyDone the service published: each lane's destination row, and whether
    its bytes (every streamed name) are the checkpoint's bytes of its expert."""
    if sim.copy_done(req) != req.gen:
        return []
    out = []
    for lane, expert in enumerate(req.experts):
        if req.kinds[lane] != LaneKind.HIT_COPY:
            continue
        slot = req.dst[lane]
        oracle = s.reference(s.tables.layer_ids[req.row], [expert])
        out.append({
            "seq": req.seq, "lane": lane, "row": req.row, "dst_slot": slot, "expert": expert,
            "digest": "".join(_digest(dst[req.row][n][slot]) for n in EXL3_STREAMED_NAMES),
            "exact": all(same_bytes(dst[req.row][n][slot], oracle[n][0]) for n in EXL3_STREAMED_NAMES),
        })
    return out


def snapshot(s, page, host, sim, reqs, dst, copies):
    snap = {"rows": {}, "deltas": [list(sim.delta(row)) for row in range(LAYERS)], "page": {
        "handled": host.handled_through(), "records": _records(page, host)}}
    for row in range(LAYERS):
        info = host.slot_info(row)
        ready = {}
        for slot, (state, expert, _stamp) in enumerate(info):
            if state == 2 and expert >= 0:
                oracle = s.reference(s.tables.layer_ids[row], [expert])
                ready[str(slot)] = {
                    "expert": expert,
                    "digest": "".join(_digest(s.slabs[row][n][slot]) for n in EXL3_STREAMED_NAMES),
                    "exact": all(same_bytes(s.slabs[row][n][slot], oracle[n][0]) for n in EXL3_STREAMED_NAMES),
                }
        snap["rows"][str(row)] = {
            "slot_info": [list(i[:2]) for i in info], "mapping": host.mapping(row), "ready": ready,
            "device_map": list(sim.replica.ram_slot[row]),
        }
    snap["results"] = [
        {"seq": r.seq, "kinds": [int(k) for k in r.kinds], "slots": list(r.slots),
         "pieces": [sim.piece_word(r, lane) for lane in range(len(r.experts))], "copy_done": sim.copy_done(r)}
        for r in reqs[-2:]
    ]
    snap["copies"] = list(copies)
    snap["dst"] = {str(row): "".join(_digest(dst[row][n]) for n in EXL3_STREAMED_NAMES) for row in sorted(dst)}
    counters = host.counters()
    snap["counters"] = {k: counters[k] for k in FUNCTIONAL}
    return snap


def run_script(s, page, host, sim, dst):
    reqs, snaps, hot, copies = [], [], {0: [], 1: []}, []
    for kind, row, lanes, extra in SCRIPT:
        if kind == "post":
            if extra.get("armed", True):
                reqs.append(sim.post(row, lanes, dst=extra.get("dst"), captured=extra.get("captured", False), hot=hot[row]))
            else:
                sim.post(row, [], protect=lanes, hot=hot[row])  # no lanes: a touch record
            _drain(host)
        elif kind == "copy":
            host.copy_engine_release(-1)
            assert host.copy_engine_idle(5.0)
            _drain(host)
            copies += completed_copies(s, sim, reqs[-1], dst)
        elif kind == "hot":
            hot[row] = list(lanes)
        _drain(host)
        snaps.append(snapshot(s, page, host, sim, reqs, dst, copies))
    return snaps
