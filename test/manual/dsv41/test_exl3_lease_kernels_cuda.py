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

from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe.expert_stream_transport import (  # noqa: E402
    DEMAND_RECORDS,
    DEMAND_RING,
    LANE_BYTES,
    LANE_FIELDS,
    PAGE_BYTES,
    RECORD_BYTES,
    RECORD_FIELDS,
    RECORD_FLAG_CAPTURED,
    RECORD_ID_MAX,
    ExpertStreamDevice,
    new_page,
)
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

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
        steps = 3 * DEMAND_RECORDS // LAYERS
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
            ExpertStreamDevice(new_page(pin=False), c.host.lease_block, **kwargs)
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
    and a page off 16-byte alignment, before anything is launched."""
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
            c.dev.page = torch.zeros(PAGE_BYTES + 16, dtype=torch.uint8).pin_memory()[1 : 1 + PAGE_BYTES]
        with pytest.raises(RuntimeError, match="16-byte" if case == "page" else "32767"):
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
        block[lease.COPY_ARMED : lease.COPY_ARMED + 4].view(torch.int32)[0] = 1
        block[lease.SPLIT : lease.SPLIT + 4 * (lease.LANES + 1)].view(torch.int32)[:] = torch.arange(lease.LANES + 1)
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
        record = DEMAND_RING + (seq - 1) % DEMAND_RECORDS * RECORD_BYTES
        page = c.page
        assert int(page[record + RECORD_FIELDS["flags"] : record + RECORD_FIELDS["flags"] + 4].view(torch.int32)[0]) == RECORD_FLAG_CAPTURED

        def lane_field(lane, name, dtype):
            at = record + RECORD_FIELDS["lanes"] + lane * LANE_BYTES + LANE_FIELDS[name]
            return page[at : at + 4].view(dtype)[0].item()

        assert [lane_field(j, "weight", torch.float32) for j in range(2)] == [0.25, 0.5 + 0.0625]
        assert [lane_field(j, "dst", torch.int32) for j in range(2)] == [5, 3]
        assert [lane_field(j, "expert", torch.int32) for j in range(lease.LANES)] == [9, 5] + [-1] * (lease.LANES - 2)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
