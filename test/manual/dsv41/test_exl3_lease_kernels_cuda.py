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
        zeros_u8, zeros_u8.clone(), torch.zeros(1, dtype=torch.int32, device="cuda"), row_capacity, 0, 0, 0,
        out["kind"], out["slot"], torch.zeros(w.lanes, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"), host_rows_1, out["dst_1"], no_i32, 0, no_i32, 0,
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




def _raw_post(lanes, nodes, planned, ram, staging, split=None, captured=False, cpu_on=False):
    """One post of the node-aware device build over one row: ``ram`` its slot map, ``staging`` node-major, ``split``
    node-major (written to the block and armed when given). Returns the lanes' kinds, slots and nodes."""
    w = lease.wire_layout(lanes, nodes)
    count, experts = len(planned), len(ram)
    page = ops.new_page(pin=True, wire=w)
    block = lease.new_lease_block(1, pin=True, wire=w)
    delta = w.lease_block_bytes
    block[delta : delta + 8].view(torch.int64)[0] = 1
    block[delta + w.delta_fields["staging"] :][: 2 * nodes * w.lanes].view(torch.int16)[:] = torch.tensor(
        staging, dtype=torch.int16
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
    ops._device_module("exl3", lanes, nodes).expert_stream_post(
        page, torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32, **cuda), planned_t,
        torch.tensor([count], dtype=torch.int32, **cuda), planned_t.clone(), 0, experts, int(block.data_ptr()),
        5_000_000_000, 0, 0, no_i64, 0, torch.arange(count, dtype=torch.int32, **cuda), int(captured),
        torch.tensor([ram], dtype=torch.int32, **cuda), torch.full((1, nodes * w.lanes), -1, dtype=torch.int32, **cuda),
        torch.ones(1, dtype=torch.int64, **cuda), torch.zeros(1, dtype=torch.int64, **cuda),
        zeros_u8, torch.ones(1, dtype=torch.uint8, **cuda), torch.full((1,), w.lanes, dtype=torch.int32, **cuda), 64,
        0, int(cpu_on), 0, out["kind"], out["slot"], out["node"], torch.zeros(1, dtype=torch.int32, **cuda),
        torch.zeros(w.lanes, dtype=torch.int64, **cuda), out["dst_1"], *cpu_args, 0,
    )
    torch.cuda.synchronize()
    del cpu_buf
    return [out[n][:count].tolist() for n in ("kind", "slot", "node")]


@pytest.mark.parametrize("lanes", [8, 16, 32])
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
