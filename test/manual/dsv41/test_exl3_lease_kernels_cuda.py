"""The slot-map chain's kernels on a real GPU against the real C++ service (LEASE_PROTOCOL.md).

Every gather is the production ``Exl3RamMissRowBackend.post`` (lease_chain_rig.Chain): post -> C1 -> S -> CW -> stream
wait -> CC. Every byte check reads a snapshot taken on the gather's stream before anything synchronizes, so a chain
that ended before its copies landed would show as stale bytes.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import CAPACITY, EXPERTS, LAYERS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe import expert_stream_transport as ops  # noqa: E402
from sglang.kernels.ops.moe.expert_stream_transport import (  # noqa: E402
    RECORD_FLAG_CAPTURED,
    RECORD_ID_MAX,
    ExpertStreamDevice,
    new_page,
)
from sglang.srt.layers.moe.ram_slot_map import LaneKind, type_lanes  # noqa: E402

W = lease.wire_layout(8)

CONFIGS = {
    "plain": {},
    "pdl": {"lease_pdl": True},
    "three_mirrors": {"mirror_weights": (1.0, 1.0, 1.0)},
}


@pytest.fixture(params=list(CONFIGS), ids=list(CONFIGS))
def chain(request, tmp_path):
    c = Chain(tmp_path, **CONFIGS[request.param])
    try:
        yield c
    finally:
        c.close()


def kinds_words(kinds, lanes):
    """The record's kinds words: lane j's kind in word j // 8, bits 4 * (j % 8)."""
    words = [0] * lease.wire_layout(lanes).kind_words
    for j, kind in enumerate(kinds):
        words[j // 8] |= int(kind) << (4 * (j % 8))
    return words


def _step(c, experts, row=0):
    c.plan(experts, row)
    c.gather(row)
    snapshot = c.snapshot(row)
    torch.cuda.synchronize()
    c.check(experts, snapshot, row)


def test_misses_hits_and_evictions_are_delivered_byte_exact(chain):
    """All misses, then all hits, then a mix that must evict: every lane's destination row holds its expert's
    checkpoint bytes, and the device's map stays the host's."""
    c = chain
    _step(c, [0, 1, 2, 3, 4, 5])
    assert c.handled()
    assert c.host.counters()["rows_read"] == 6
    _step(c, [3, 0, 5, 1])
    assert c.kinds(4) == [LaneKind.HIT_SM] * 4
    assert c.handled() and c.host.counters()["rows_read"] == 6, "a resident expert was read again"
    _step(c, [2, 9, 10, 1, 11])
    assert c.handled()
    _step(c, [12])  # applies the previous chain's delta
    assert c.handled()
    counters = c.host.counters()
    assert counters["rows_read"] == 10 and counters["evictions"] >= 1
    assert {2, 9, 10, 1, 11, 12} <= c.resident()
    _step(c, [2])
    assert c.device_map(0) == c.host.mapping(0)


def test_a_request_without_lanes_passes_through(chain):
    c = chain
    c.plan([])
    c.gather()
    torch.cuda.synchronize()
    _step(c, [7])
    assert c.handled()


def test_the_captured_chain_replays_byte_exact_through_ring_reuse_and_eviction(tmp_path):
    """Both rows in one graph, replayed with new plans under capacity pressure, three times round the 16-deep ring."""
    c = Chain(tmp_path)
    try:
        for row in range(LAYERS):
            _step(c, [0, 1], row)  # load the kernels before capture
        assert c.handled()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            for row in range(LAYERS):
                c.gather(row)
        rng = random.Random(7)
        steps = 3 * W.demand_records // LAYERS
        # "served" counts the requests that read a row, "touch_only" the all-hit ones.
        before = c.host.counters()
        served = before["served"] + before["touch_only"]
        for _ in range(steps):
            plans = {row: rng.sample(range(EXPERTS), rng.randint(1, TOP_K)) for row in range(LAYERS)}
            with torch.cuda.stream(stream):
                for row in range(LAYERS):
                    c.plan(plans[row], row)
                graph.replay()
                snapshots = {row: c.snapshot(row) for row in range(LAYERS)}
            stream.synchronize()
            for row in range(LAYERS):
                c.check(plans[row], snapshots[row], row)
        assert c.handled()
        counters = c.host.counters()
        assert counters["served"] + counters["touch_only"] - served == steps * LAYERS
        assert counters["overruns"] == 0 and counters["evictions"] > 0
    finally:
        c.close()


_TRAP_SCRIPT = """
import sys, time
import torch
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
c = Chain(sys.argv[1], timeout_ms=300, start=False)
c.plan([4])
start = time.perf_counter()
c.gather()
try:
    torch.cuda.synchronize()
    print("reached", flush=True)
except RuntimeError as error:
    print(f"trapped {time.perf_counter() - start:.3f} {error}", flush=True)
import os
os._exit(0)
"""


def test_s_traps_at_its_deadline_when_nothing_serves(tmp_path):
    """No service thread: the miss's pieces are never published, and S traps once its deadline passes -- not before,
    and not never. The chain has no failure word to read; the trap is the failure."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_TRAP_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=120,
    )
    assert "reached" not in result.stdout, result.stdout
    line = next((line for line in result.stdout.splitlines() if line.startswith("trapped")), None)
    assert line is not None, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    elapsed = float(line.split()[1])
    assert 0.3 <= elapsed < 10.0, line


def test_the_device_refuses_what_its_kernels_cannot_read(tmp_path):
    c = Chain(tmp_path)
    try:
        kwargs = dict(device="cuda", layers=LAYERS, experts=EXPERTS, timeout_ms=100, piece_runs=c.host.piece_runs(),
                      row_capacities=[CAPACITY] * LAYERS)
        with pytest.raises(ValueError, match="pinned"):
            ExpertStreamDevice(new_page(pin=False, wire=wire_layout(8)), c.host.lease_block, **kwargs)
        with pytest.raises(ValueError, match="row capacities"):
            ExpertStreamDevice(c.page, c.host.lease_block, **{**kwargs, "row_capacities": [CAPACITY]})
        with pytest.raises(ValueError, match="piece_runs"):
            ExpertStreamDevice(c.page, c.host.lease_block, **{**kwargs, "piece_runs": c.host.piece_runs()[:1]})
        plan, backend = c.plans[0], c.backends[0]
        with pytest.raises(ValueError, match="segment_map"):
            c.dev.stream(0, backend.planned, plan.count, plan.slots, c.segments[0], backend.stream_maps[0][:-1])
        with pytest.raises(ValueError, match="count"):
            c.dev.post(0, backend.planned, plan.count.long(), backend.routes, plan.slots)
    finally:
        c.close()


@pytest.mark.parametrize("case", ["experts", "row_capacity", "page"])
def test_the_post_launch_refuses_what_a_narrow_record_cannot_carry(tmp_path, case):
    """The post writes i16 ids with 16-byte stores: its launcher refuses more experts or slots than an i16 carries,
    and a page off 128-byte alignment (each record one prefetch pair), before anything is launched."""
    c = Chain(tmp_path, start=False)
    try:
        c.plan([1], 0)
        backend, plan = c.backends[0], c.plans[0]
        backend._stage_planned(plan)
        if case == "experts":
            c.dev.experts = RECORD_ID_MAX + 1
        elif case == "row_capacity":
            c.dev._row_capacities = (RECORD_ID_MAX + 1,) * len(c.dev._row_capacities)
        else:
            c.dev.page = torch.zeros(W.page_bytes + 128, dtype=torch.uint8).pin_memory()[64 : 64 + W.page_bytes]
        with pytest.raises(RuntimeError, match="128-byte" if case == "page" else "32767"):
            c.dev.post(0, backend.planned, plan.count, backend.routes, plan.slots)
    finally:
        c.close()


@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_post_stages_the_cpu_input_and_each_lanes_routing_weight(tmp_path, x_dtype):
    """A captured post whose lanes the device types CPU stages x as fp16 in the row's host row, and the record carries
    each lane's weight -- the sum over the routes naming the lane's expert -- and its destination slot; bytes past the
    row stay untouched. Nothing serves: only the post runs."""
    hidden = 64
    c = Chain(tmp_path, start=False, copy_engine=True)
    try:
        x_rows = torch.full((LAYERS, 2 * hidden + 16), 0xAB, dtype=torch.uint8).pin_memory()
        c.dev.cpu_x_rows = x_rows
        row = 1
        c.dev.set_row_cpu(row)
        block = c.host.lease_block
        block[W.copy_armed : W.copy_armed + 4].view(torch.int32)[0] = 1
        block[W.split : W.split + 4 * (W.lanes + 1)].view(torch.int32)[:] = torch.arange(W.lanes + 1)
        c.dev.map_bulk_apply(torch.tensor([[row, 9, 0], [row, 5, 1]], dtype=torch.int32))  # both lanes RAM hits
        c.plan([9, 5], row)
        backend, plan = c.backends[row], c.plans[row]
        backend.routes[:4] = torch.tensor([5, 9, 7, 5])
        backend.planned[:TOP_K].copy_(plan.expert_ids)
        weights = torch.tensor([[0.5, 0.25, 0.125, 0.0625]], device="cuda")
        x = (torch.randn(1, hidden, device="cuda") * 30).to(x_dtype)
        plan.slots[:2] = torch.tensor([5, 3], dtype=torch.int32)
        c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=(x, weights))
        torch.cuda.synchronize()
        assert c.kinds(2) == [LaneKind.HIT_CPU, LaneKind.HIT_CPU]
        seq = int(c.dev.stats()["posted"]) & 0xFFFFFFFF
        record = W.demand_ring + (seq - 1) % W.demand_records * W.record_bytes
        page = c.page
        assert int(page[record + W.record_fields["flags"]]) == RECORD_FLAG_CAPTURED

        def field(name, dtype, n):
            at = record + W.record_fields[name]
            return page[at : at + n * dtype.itemsize].view(dtype).tolist()

        assert int(page[record + W.record_fields["counts"]]) & 0xF == 2
        assert field("kinds", torch.int32, 1) == kinds_words([LaneKind.HIT_CPU, LaneKind.HIT_CPU], W.lanes)
        assert field("lane_weight", torch.float32, W.lanes) == [0.25, 0.5 + 0.0625] + [0.0] * (W.lanes - 2)
        assert field("lane_dst", torch.int16, 2) == [5, 3]
        assert field("lane_expert", torch.int16, W.lanes) == [9, 5] + [-1] * (W.lanes - 2)
        staged = x_rows[row, : 2 * hidden].view(torch.float16)
        assert torch.equal(staged.view(torch.int16), x.cpu().half().reshape(-1).view(torch.int16))
        assert (x_rows[row, 2 * hidden :] == 0xAB).all() and (x_rows[1 - row] == 0xAB).all()
        with pytest.raises(ValueError, match="captured"):
            c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, cpu_input=(x, weights))
        with pytest.raises(ValueError, match="does not fit"):
            c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True,
                       cpu_input=(torch.zeros(1, 4 * hidden, device="cuda"), weights))
    finally:
        # The posted request is never served: nothing waits on it, and the host stops with it unserved.
        c.close()


@pytest.mark.parametrize("lanes, count", [(8, 8), (16, 16), (16, 9), (24, 24), (24, 17), (32, 32), (32, 17)])
def test_the_post_record_round_trips_at_every_lane_width(lanes, count):
    """The post kernel built for ``lanes`` writes a record the service's reader decodes lane by lane: even lanes are
    RAM hits at their map slot, odd lanes misses at the delta's staging slots, and the count is exactly ``count``."""
    w = lease.wire_layout(lanes)
    experts, row_capacity = 2 * lanes, 64

    page = ops.new_page(pin=True, wire=w)
    block = lease.new_lease_block(1, pin=True, wire=w)
    delta = w.lease_block_bytes
    block[delta : delta + 8].view(torch.int64)[0] = 1  # the attach delta's tag: map_chain starts at 1
    staging_ids = torch.arange(40, 40 + w.lanes, dtype=torch.int16)
    block[delta + w.delta_fields["staging"] :][: 2 * w.lanes].view(torch.int16)[:] = staging_ids
    ram_slot = torch.full((1, experts), -1, dtype=torch.int32, device="cuda")
    for j in range(0, count, 2):
        ram_slot[0, j] = j // 2
    planned = torch.arange(count, dtype=torch.int64, device="cuda")
    dst = torch.arange(10, 10 + count, dtype=torch.int32, device="cuda")
    state = torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32, device="cuda")
    out = {n: torch.zeros(w.lanes, dtype=torch.int32, device="cuda") for n in ("kind", "slot", "dst_1")}
    host_rows_1 = torch.zeros(w.lanes, dtype=torch.int64, device="cuda")
    zeros_u8 = torch.zeros(1, dtype=torch.uint8, device="cuda")
    no_i64 = torch.empty(0, dtype=torch.int64, device="cuda")
    no_i32 = torch.empty(0, dtype=torch.int32, device="cuda")
    ops._device_module("exl3", lanes).expert_stream_post(
        page, state, planned, torch.tensor([count], dtype=torch.int32, device="cuda"), planned.clone(), 0, experts,
        int(block.data_ptr()), 5_000_000_000, 0, 0, no_i64, 0, dst, 0, ram_slot,
        torch.full((1, w.lanes), -1, dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int64, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"),
        zeros_u8, zeros_u8.clone(), torch.zeros(1, dtype=torch.int32, device="cuda"), row_capacity, 0, 0, 0, 0, 3,
        out["kind"], out["slot"], torch.zeros(w.lanes, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"), host_rows_1, out["dst_1"], no_i32, 0, no_i32, 1, 16, 0,
        torch.zeros(1, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"),
        0, torch.zeros(ops.DRIP_TABLE_SLOTS, dtype=torch.int64, device="cuda"),
        torch.zeros(w.lanes, dtype=torch.int64, device="cuda"), torch.zeros(w.lanes, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.full((1, w.nodes * (w.lanes + 1)), -1, dtype=torch.int32, device="cuda"), no_i64, 0,
    )
    torch.cuda.synchronize()
    record = page[w.demand_ring : w.demand_ring + w.record_bytes].clone()
    got = ops.read_record_fields(record, 1, variant="instr", lanes=lanes)
    kinds = [LaneKind.HIT_SM if j % 2 == 0 else LaneKind.MISS_GPU for j in range(count)]
    assert got["status"] == "ok"
    assert got["lanes"] == [
        {"expert": j, "slot": j // 2 if j % 2 == 0 else 40 + j // 2, "dst": 10 + j, "weight": 0.0, "kind": int(kinds[j])}
        for j in range(count)
    ]
    assert got["protect"] == list(range(count))
    words = record[w.record_fields["kinds"] :][: 4 * w.kind_words].view(torch.int32).tolist()
    assert [x & 0xFFFFFFFF for x in words] == kinds_words(kinds, lanes)


@pytest.mark.parametrize("lanes, nodes", [(8, 1), (8, 2), (16, 2)])
def test_each_device_build_is_compiled_for_its_node_count(lanes, nodes):
    assert int(ops._device_module("exl3", lanes, nodes).expert_stream_wire_nodes()) == nodes




def _raw_post(lanes, nodes, planned, ram, staging, split=None, captured=False, cpu_on=False, scratch_words=None,
              scratch_out=None, row_split=None, copy_ce=False, dst=None, spill=False, routes=None, miss_cut=0,
              split_model=None):
    """One post of the node-aware device build over one row: ``ram`` its slot map, ``staging`` node-major, ``split``
    node-major (written to the block and armed when given). Returns the lanes' kinds, slots and nodes.
    ``row_split``: the row's own node-major table (SGLANG_DSV41_CPU_EXPERTS_SPLIT_LAYERS), else -1 (the block's).
    ``scratch_words``: the GPU drip's scratch table (u64 per slot), written to the block and read by the post as that
    many slots; ``scratch_out`` (a list) then receives C1s's compacted (rows, dst slots, count).
    ``copy_ce``: the copy engine may take hits (hit_copy ce, the row's copy table set, w.lanes destination rows);
    ``dst`` the lanes' destination slots (default 0..count-1), with ``spill`` making -1 a forced lane; ``routes`` the
    record's routes (default the planned experts once each); ``miss_cut`` the split miss cut (bound 3);
    ``split_model`` the split model's cost words, whose counters are then appended to the result."""
    w = lease.wire_layout(lanes, nodes)
    count, experts = len(planned), len(ram)
    page = ops.new_page(pin=True, wire=w)
    block = lease.new_lease_block(1, pin=True, wire=w)
    delta = w.lease_block_bytes
    block[delta : delta + 8].view(torch.int64)[0] = 1
    block[delta + w.delta_fields["staging"] :][: 2 * nodes * w.lanes].view(torch.int16)[:] = torch.tensor(
        staging, dtype=torch.int16
    )
    scratch_slots = 0 if scratch_words is None else len(scratch_words)
    scratch_table = torch.zeros(ops.DRIP_TABLE_SLOTS, dtype=torch.int64, device="cuda")
    if scratch_slots:
        block[w.copy_armed : w.copy_armed + 4].view(torch.int32)[0] = 1
        scratch_table[:scratch_slots] = torch.tensor(
            [x - (1 << 64) if x >= 1 << 63 else x for x in scratch_words], dtype=torch.int64
        )
    if split is not None:
        block[w.copy_armed : w.copy_armed + 4].view(torch.int32)[0] = 1
        for node in range(nodes):
            at = w.split + node * w.split_stride
            block[at : at + 4 * (w.lanes + 1)].view(torch.int32)[:] = torch.tensor(
                split[node * (w.lanes + 1) : (node + 1) * (w.lanes + 1)], dtype=torch.int32
            )
    cuda = dict(device="cuda")
    planned_t = torch.tensor(planned, dtype=torch.int64, **cuda)
    out = {n: torch.zeros(w.lanes, dtype=torch.int32, **cuda) for n in ("kind", "slot", "node", "dst_1")}
    zeros_u8 = torch.zeros(1, dtype=torch.uint8, **cuda)
    no_i64, no_i32 = torch.empty(0, dtype=torch.int64, **cuda), torch.empty(0, dtype=torch.int32, **cuda)
    if cpu_on:  # a CPU lane needs a staged input, else the post traps
        cpu_buf = torch.zeros(64, dtype=torch.float32).pin_memory()
        cpu_args = (torch.zeros(1, 8, dtype=torch.float32, **cuda), int(cpu_buf.data_ptr()),
                    torch.ones(count, dtype=torch.float32, **cuda))
    else:
        cpu_buf, cpu_args = None, (no_i32, 0, no_i32)
    scratch_rows = torch.full((w.lanes,), -7, dtype=torch.int64, **cuda)
    scratch_dst = torch.full((w.lanes,), -7, dtype=torch.int32, **cuda)
    scratch_go = torch.full((1,), -7, dtype=torch.int32, **cuda)
    model = no_i64
    if split_model is not None:
        model = torch.tensor(list(split_model) + [0] * (1 + 2 * nodes), dtype=torch.int64, **cuda)
    count_t = torch.tensor([count], dtype=torch.int32, **cuda)
    ops._device_module("exl3", lanes, nodes).expert_stream_post(
        page, torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32, **cuda), planned_t, count_t,
        planned_t.clone() if routes is None else torch.tensor(routes, dtype=torch.int64, **cuda), 0, experts,
        int(block.data_ptr()), 5_000_000_000, 0, 0, no_i64, 0,
        torch.tensor(list(range(count)) if dst is None else dst, dtype=torch.int32, **cuda), int(captured),
        torch.tensor([ram], dtype=torch.int32, **cuda), torch.full((1, nodes * w.lanes), -1, dtype=torch.int32, **cuda),
        torch.ones(1, dtype=torch.int64, **cuda), torch.zeros(1, dtype=torch.int64, **cuda),
        torch.ones(1, dtype=torch.uint8, **cuda) if copy_ce else zeros_u8, torch.ones(1, dtype=torch.uint8, **cuda),
        torch.full((1,), w.lanes, dtype=torch.int32, **cuda), 64,
        int(copy_ce), int(cpu_on), 0, miss_cut, 3, out["kind"], out["slot"], out["node"],
        torch.zeros(1, dtype=torch.int32, **cuda),
        torch.zeros(w.lanes, dtype=torch.int64, **cuda), out["dst_1"], *cpu_args, 1, 32, int(spill),
        torch.zeros(1, dtype=torch.int32, **cuda), torch.zeros(1, dtype=torch.int64, **cuda),
        scratch_slots, scratch_table, scratch_rows, scratch_dst, scratch_go,
        torch.tensor([row_split if row_split is not None else [-1] * (nodes * (w.lanes + 1))], dtype=torch.int32,
                     **cuda),
        model, 0,
    )
    torch.cuda.synchronize()
    del cpu_buf
    if scratch_out is not None:
        go = int(scratch_go[0])
        scratch_out.append((scratch_rows[:go].tolist(), scratch_dst[:go].tolist(), go))
    count = int(count_t[0])
    result = [out[n][:count].tolist() for n in ("kind", "slot", "node")]
    if split_model is not None:
        result.append(model[len(split_model) :].tolist())
    return result


@pytest.mark.parametrize("lanes", [8, 16, 32, 40])
@pytest.mark.parametrize("nodes", [1, 2])
def test_each_miss_takes_the_next_staging_slot_of_its_home_node(lanes, nodes):
    """The device's typing equals ram_slot_map.type_lanes's on random plans of hits and misses on every node."""
    w = lease.wire_layout(lanes, nodes)
    rng = random.Random(lanes * 10 + nodes)
    experts = 4 * lanes
    for _ in range(20):
        count = rng.randint(1, lanes)
        planned = rng.sample(range(experts), count)
        ram = [rng.randrange(64) if rng.random() < 0.4 else -1 for _ in range(experts)]
        staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
        kinds, slots, homes = _raw_post(lanes, nodes, planned, ram, staging)
        want_kinds, want_slots = type_lanes(
            planned, ram, staging, [0] * (nodes * (w.lanes + 1)), lanes=w.lanes, captured=False, copy_armed=False,
            hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=nodes,
        )
        assert (kinds, slots) == ([int(k) for k in want_kinds], want_slots)
        assert homes == [w.home(e) for e in planned]


def test_the_split_is_per_node_and_a_zero_split_keeps_a_node_off_the_cpu():
    lanes, nodes = 8, 2
    w = lease.wire_layout(lanes, nodes)
    split = [0] * (nodes * (w.lanes + 1))
    split[3] = 2  # node 0: 2 of 3 eligible lanes; node 1's table is all zero
    planned, ram = [0, 1, 2, 3, 4, 5], list(range(16))
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    kinds, _, _ = _raw_post(lanes, nodes, planned, ram, staging, split=split, captured=True, cpu_on=True)
    want, _ = type_lanes(planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True, hit_copy="sm",
                         cpu_on=True, cpu_misses=False, nodes=nodes)
    assert kinds == [int(k) for k in want] == [int(LaneKind.HIT_SM), int(LaneKind.HIT_SM), int(LaneKind.HIT_CPU),
                                               int(LaneKind.HIT_SM), int(LaneKind.HIT_CPU), int(LaneKind.HIT_SM)]


def test_a_row_split_replaces_the_lease_split_for_its_row_only():
    """SGLANG_DSV41_CPU_EXPERTS_SPLIT_LAYERS: a row with its own table types its lanes by it on every node, whatever
    the lease's tables say (what calibration and the retune write); -1 at its first word keeps the lease's. Mutations:
    the post ignoring row_split, or reading it for every row -- red on one of the two posts."""
    lanes, nodes = 8, 2
    w = lease.wire_layout(lanes, nodes)
    lease_split = [n for _ in range(nodes) for n in range(w.lanes + 1)]  # every eligible lane the CPU's
    own = [max(n - 2, 0) for _ in range(nodes) for n in range(w.lanes + 1)]  # two lanes a node to the link
    planned, ram = [0, 1, 2, 3, 4, 5], list(range(16))
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    for row_split, split in ((own, own), (None, lease_split)):
        kinds, _, _ = _raw_post(lanes, nodes, planned, ram, staging, split=lease_split, captured=True, cpu_on=True,
                                row_split=row_split)
        want, _ = type_lanes(planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True,
                             hit_copy="sm", cpu_on=True, cpu_misses=False, nodes=nodes)
        assert kinds == [int(k) for k in want]
    assert kinds.count(int(LaneKind.HIT_CPU)) == 6


def _split_model_case(rng, lanes, nodes, *, copy_ce=True, count=None, hit_rate=0.6, forced=None, coarse=True):
    """A random record for the split model: plan, slot map, destinations (some out of range), routes naming each
    planned expert 1-4 times plus unplanned ones, a spill point, and cost words on a coarse grid so ties happen."""
    w = lease.wire_layout(lanes, nodes)
    experts = 4 * w.lanes
    count = rng.randint(1, w.lanes) if count is None else count
    planned = rng.sample(range(experts), count)
    ram = [rng.randrange(64) if rng.random() < hit_rate else -1 for _ in range(experts)]
    forced_from = rng.randint(0, count) if forced is None else forced
    dst = [(w.lanes if rng.random() < 0.15 else j) if j < forced_from else -1 for j in range(count)]
    routes = [e for e in planned for _ in range(rng.randint(1, 4))] + rng.sample(range(experts), 3)
    rng.shuffle(routes)
    step = 100 if coarse else 1
    words = []
    for _ in range(nodes):
        words += [rng.choice([-1, 0, 2]) * step, rng.randint(0, 5) * step, rng.randint(0, 5) * step,
                  rng.randint(0, 4) * step, rng.randint(0, 3) * step]
    words += [rng.randint(1, 15) * step, rng.randint(0, 10) * step, rng.randint(0, 20) * step,
              rng.choice([0, rng.randint(1, 15) * step]), rng.randint(0, 5) * step]  # stream_lane (0: off), _base
    return w, planned, ram, forced_from, dst, routes, words


def _split_model_want(w, planned, ram, staging, split, forced_from, dst, routes, words, copy_ce):
    stats = [0] * (1 + 2 * w.nodes)
    kinds, slots = type_lanes(
        planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True, hit_copy="ce" if copy_ce else "sm",
        cpu_on=True, cpu_misses=False, ce_ok=copy_ce, dst_ok=[0 <= d < w.lanes for d in dst], nodes=w.nodes,
        forced_from=forced_from, miss_cut=1, miss_cut_max=3, split_model=words,
        tokens=[routes.count(e) for e in planned], model_stats=stats)
    return [int(k) for k in kinds], slots, stats


@pytest.mark.parametrize("lanes, nodes", [(8, 1), (8, 2), (16, 2), (40, 2)])
def test_the_split_model_types_lanes_like_the_reference(lanes, nodes):
    """Experiment (SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL): 300 random records -- hits, NVMe misses, forced lanes from a
    spill point, destinations past the row's, routes naming experts 1-4 times, cost words on a grid that makes ties --
    type kind for kind and slot for slot as ram_slot_map.type_lanes(split_model=...), counters included; with the
    copy engine off the table and the miss cut type them. Mutations: ties broken last-lane first, a forced miss's
    landing counted from 0, the copy FIFO per node, a non-copyable hit offered to the copy engine -- red."""
    rng = random.Random(1000 * lanes + nodes)
    w = lease.wire_layout(lanes, nodes)
    split = [max(0, (2 * n) // 3) for _ in range(nodes) for n in range(w.lanes + 1)]
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    moved = 0
    for i in range(300):
        copy_ce = i % 10 != 0
        _, planned, ram, forced_from, dst, routes, words = _split_model_case(rng, lanes, nodes)
        kinds, slots, _, stats = _raw_post(
            lanes, nodes, planned, ram, staging, split=split, captured=True, cpu_on=True, copy_ce=copy_ce, dst=dst,
            spill=True, routes=routes, miss_cut=1, split_model=words)
        want = _split_model_want(w, planned, ram, staging, split, forced_from, dst, routes, words, copy_ce)
        assert (kinds, slots, stats) == want, (planned, ram, forced_from, dst, routes, words, copy_ce)
        moved += sum(stats[1:])
    assert moved > 0


# (count, hit rate, forced_from as a fraction of count or None, copy engine): no eligible hit, all forced, copy off,
# every lane hits, one lane.
SPLIT_MODEL_EDGES = [(6, 0.0, None, True), (6, 1.0, 0, True), (6, 1.0, None, False), (8, 1.0, 1.0, True),
                     (1, 1.0, 1.0, True)]


@pytest.mark.parametrize("count, hit_rate, forced, copy_ce", SPLIT_MODEL_EDGES)
def test_the_split_model_edges_type_lanes_like_the_reference(count, hit_rate, forced, copy_ce):
    """The split model at its edges, 2 nodes: no eligible hit (counters stay 0), every lane forced (all CPU), the copy
    engine off (the table and cut type the lanes, counters stay 0), every lane an unforced hit, one lane."""
    lanes, nodes = 8, 2
    w = lease.wire_layout(lanes, nodes)
    rng = random.Random(count * 7 + int(10 * hit_rate))
    split = [max(0, (2 * n) // 3) for _ in range(nodes) for n in range(w.lanes + 1)]
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    for _ in range(20):
        _, planned, ram, forced_from, dst, routes, words = _split_model_case(
            rng, lanes, nodes, count=count, hit_rate=hit_rate,
            forced=None if forced is None else int(forced * count))
        kinds, slots, _, stats = _raw_post(
            lanes, nodes, planned, ram, staging, split=split, captured=True, cpu_on=True, copy_ce=copy_ce, dst=dst,
            spill=True, routes=routes, miss_cut=1, split_model=words)
        want = _split_model_want(w, planned, ram, staging, split, forced_from, dst, routes, words, copy_ce)
        assert (kinds, slots, stats) == want
        if hit_rate == 0.0 or not copy_ce:
            assert stats == [0] * (1 + 2 * nodes)
        if forced_from == 0:
            assert all(k in (int(LaneKind.HIT_CPU), int(LaneKind.MISS_CPU)) for k in kinds)


def test_the_split_model_off_types_lanes_as_the_table():
    """An empty split-model tensor is the table and the miss cut, bit for bit, on the same random records."""
    lanes, nodes = 16, 2
    w = lease.wire_layout(lanes, nodes)
    rng = random.Random(5)
    split = [max(0, (2 * n) // 3) for _ in range(nodes) for n in range(w.lanes + 1)]
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    for _ in range(100):
        _, planned, ram, forced_from, dst, routes, _ = _split_model_case(rng, lanes, nodes)
        kinds, slots, _ = _raw_post(lanes, nodes, planned, ram, staging, split=split, captured=True, cpu_on=True,
                                    copy_ce=True, dst=dst, spill=True, routes=routes, miss_cut=1)
        want, want_slots = type_lanes(
            planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True, hit_copy="ce", cpu_on=True,
            cpu_misses=False, dst_ok=[0 <= d < w.lanes for d in dst], nodes=nodes, forced_from=forced_from,
            miss_cut=1, miss_cut_max=3)
        assert (kinds, slots) == ([int(k) for k in want], want_slots)


def test_split_model_off_words_type_lanes_as_the_table():
    """Cost words whose copy_lane is SPLIT_MODEL_OFF (the service's words until its calibration publishes, and after
    one fails) type as the table and the miss cut, bit for bit, and count nothing. Mutation: the post ignoring the
    copy_lane sign -- red."""
    from sglang.srt.layers.moe.ram_slot_map import split_model_off_words

    lanes, nodes = 16, 2
    w = lease.wire_layout(lanes, nodes)
    rng = random.Random(6)
    split = [max(0, (2 * n) // 3) for _ in range(nodes) for n in range(w.lanes + 1)]
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    for _ in range(100):
        _, planned, ram, forced_from, dst, routes, _ = _split_model_case(rng, lanes, nodes)
        kinds, slots, _, stats = _raw_post(lanes, nodes, planned, ram, staging, split=split, captured=True,
                                           cpu_on=True, copy_ce=True, dst=dst, spill=True, routes=routes, miss_cut=1,
                                           split_model=split_model_off_words(nodes))
        want, want_slots = type_lanes(
            planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True, hit_copy="ce", cpu_on=True,
            cpu_misses=False, dst_ok=[0 <= d < w.lanes for d in dst], nodes=nodes, forced_from=forced_from,
            miss_cut=1, miss_cut_max=3)
        assert (kinds, slots) == ([int(k) for k in want], want_slots) and stats == [0] * (1 + 2 * nodes)


def test_the_calibration_pull_copies_pinned_host_ranges():
    """The split model calibration's stand-in for S's pulls copies each {source, destination, bytes} range of pinned
    host memory into device memory, aligned and not, with S's grid."""
    host = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8).pin_memory()
    dst = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    ranges = [(0, 0, 4096), (8192, 8192, 300_000), (1, 400_001, 777)]  # (source offset, destination offset, bytes)
    table = torch.tensor([[host.data_ptr() + s, dst.data_ptr() + d, n] for s, d, n in ranges], dtype=torch.int64,
                         device="cuda")
    ops._device_module("exl3", 8, 1).expert_stream_calibration_pull(table)
    torch.cuda.synchronize()
    for s, d, n in ranges:
        assert torch.equal(dst[d : d + n].cpu(), host[s : s + n]), (s, d, n)
    assert not dst[4096:8192].any()


def _scratch_word(tag, row, expert):
    """A scratch-table word as host/gpu_drip.h publishes it."""
    return tag << 32 | row << 16 | expert


@pytest.mark.parametrize("lanes, nodes", [(8, 1), (40, 2)])
def test_a_landed_drip_is_typed_hit_scratch_and_leaves_the_split(lanes, nodes):
    """The GPU drip: the CUDA post's typing equals type_lanes(scratch=...) on random plans whose experts are partly
    published in the scratch table, with every eligible lane the CPU's (split[n] = n), so a scratched RAM hit both
    leaves its node's n and is HIT_SCRATCH; C1s's compaction lists exactly those lanes. A word of another row, or one
    with tag 0, names nothing. Mutations: match_scratch ignoring the row or the tag, or type_lanes leaving a scratched
    lane eligible -- red on the kinds."""
    w = lease.wire_layout(lanes, nodes)
    rng = random.Random(lanes * 10 + nodes)
    experts = 4 * lanes
    split = [n for _ in range(nodes) for n in range(w.lanes + 1)]
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    for _ in range(20):
        count = rng.randint(1, lanes)
        planned = rng.sample(range(experts), count)
        ram = [rng.randrange(64) if rng.random() < 0.6 else -1 for _ in range(experts)]
        chosen = rng.sample(planned, min(count, rng.randint(0, 6)))
        decoys = [e for e in planned if e not in chosen][:2]
        words = [_scratch_word(rng.randrange(1, 1 << 32), 0, e) for e in chosen]
        words += [_scratch_word(5, 1, e) for e in decoys[:1]] + [_scratch_word(0, 0, e) for e in decoys[1:]]
        rng.shuffle(words)
        scratch = [next((s for s, x in enumerate(words) if x >> 32 and x & 0xFFFFFFFF == e), -1) for e in planned]
        compacted = []
        kinds, slots, _ = _raw_post(lanes, nodes, planned, ram, staging, split=split, captured=True, cpu_on=True,
                                    scratch_words=words or [0], scratch_out=compacted)
        want_kinds, want_slots = type_lanes(
            planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True, hit_copy="sm", cpu_on=True,
            cpu_misses=False, nodes=nodes, scratch=scratch,
        )
        assert (kinds, slots) == ([int(k) for k in want_kinds], want_slots)
        scratched = [j for j, k in enumerate(want_kinds) if k == LaneKind.HIT_SCRATCH]
        assert len(scratched) == len(chosen)
        assert compacted == [([scratch[j] for j in scratched], scratched, len(scratched))]


_NODE_TRAP_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from test_exl3_lease_kernels_cuda import _raw_post
staging = [100 + k for k in range(8)] + [200, 201] + [-1] * 6
print("slots", _raw_post(8, 2, [1, 3, 0, 2, 4, 6, 8], [-1] * 16, staging)[1], flush=True)
try:
    _raw_post(8, 2, [1, 3, 5], [-1] * 16, staging)
    print("reached", flush=True)
except Exception as error:
    print("trapped", type(error).__name__, flush=True)
"""


def test_a_node_out_of_staging_traps_while_the_other_node_has_slots():
    """Review Focus 1: node 1 has 2 staging slots and node 0 eight. Seven misses, two on node 1, take node 1's two
    and node 0's first five; three misses on node 1 trap, although node 0 has slots to spare."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_NODE_TRAP_SCRIPT), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=120,
    )
    assert "slots [200, 201, 100, 101, 102, 103, 104]" in result.stdout, (result.stdout, result.stderr[-2000:])
    assert "reached" not in result.stdout and "trapped" in result.stdout, (result.stdout, result.stderr[-2000:])


@pytest.mark.parametrize(
    "kinds, nodes_of, parts",
    [
        ([LaneKind.HIT_CPU, LaneKind.HIT_CPU], [0, 1], 0b0101),
        ([LaneKind.HIT_CPU, LaneKind.MISS_CPU], [0, 1], 0b1001),
        ([LaneKind.MISS_CPU], [1], 0b1000),
        ([LaneKind.HIT_CPU, LaneKind.HIT_CPU, LaneKind.MISS_CPU, LaneKind.MISS_CPU], [0, 1, 1, 0], 0b1111),
    ],
    ids=["hits_on_both", "hit_n0_miss_n1", "miss_n1", "all_four"],
)
def test_the_copy_wait_flags_the_cpu_part_of_each_lanes_node(kinds, nodes_of, parts):
    """Two nodes: a lane's CPU part is 2 * node + (miss). CW collects the bits into ce_mask[2] and CC passes every one
    on in cpu_lanes[1]. Mutation: CC masking the parts with 0x3 drops node 1's."""
    lanes, nodes = 8, 2
    w = lease.wire_layout(lanes, nodes)
    block = lease.new_lease_block(1, pin=True, wire=w)
    block[w.copy_done : w.copy_done + 8].view(torch.int64)[0] = 1  # request 1's CopyDone: CW opens the gate itself
    cuda = dict(device="cuda")
    count = len(kinds)
    lane_kind = torch.zeros(w.lanes, dtype=torch.int32, **cuda)
    lane_kind[:count] = torch.tensor([int(k) for k in kinds], dtype=torch.int32)
    lane_node = torch.zeros(w.lanes, dtype=torch.int32, **cuda)
    lane_node[:count] = torch.tensor(nodes_of, dtype=torch.int32)
    state = torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32, **cuda)
    state[ops.STATE_WORDS["pending"]] = 1
    ce_mask = torch.full((3,), -1, dtype=torch.int32, **cuda)
    cpu_lanes = torch.full((2,), -1, dtype=torch.int32, **cuda)
    ops._device_module("exl3", lanes, nodes).expert_stream_lease_copy_wait(
        state, torch.tensor([count], dtype=torch.int32, **cuda), int(block.data_ptr()), lane_kind,
        torch.arange(w.lanes, dtype=torch.int32, **cuda), lane_node, torch.arange(w.lanes, dtype=torch.int32, **cuda),
        0, 0, ce_mask, cpu_lanes, 0,
    )
    torch.cuda.synchronize()
    assert ce_mask[2].item() == parts
    assert cpu_lanes.tolist() == [(1 << count) - 1, parts]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
