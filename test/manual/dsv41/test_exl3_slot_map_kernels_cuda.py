"""The slot-map post on a real GPU (LEASE_PROTOCOL.md): delta application, lane typing against the Python reference,
the bounded wait for a delta the host never published, and a miss streamed from its staging slot.

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

from lease_chain_rig import EXPERTS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind, LaneOverflow, type_lanes  # noqa: E402

W = lease.wire_layout(8)
SPLIT = [0, 1, 1, 2, 3, 3, 4, 5, 5]
ALL_CPU = list(range(W.lanes + 1))  # split[n] = n: every eligible unforced lane is the CPU's


def _i32(t, offset, count=1):
    return t[offset : offset + 4 * count].view(torch.int32)


def _write_delta(c, row, tag, staging, entries=(), w=W):
    """The host's delta publication, done by the test: payload, then the tag (x86 keeps tensor stores in order)."""
    base = w.lease_block_bytes + row * w.delta_stride
    f = w.delta_fields
    block = c.host.lease_block
    _i32(block, base + f["count"])[0] = len(entries)
    block[base + f["staging"] : base + f["staging"] + 2 * w.lanes].view(torch.int16)[:] = torch.tensor(
        list(staging) + [-1] * (w.lanes - len(staging)), dtype=torch.int16)
    for i, (expert, slot) in enumerate(entries):
        block[base + f["entries"] + 4 * i : base + f["entries"] + 4 * i + 4].view(torch.int16)[:] = torch.tensor(
            [expert, slot], dtype=torch.int16)
    block[base + f["tag"] : base + f["tag"] + 8].view(torch.int64)[0] = tag


def _set_host_words(c, *, armed, split=SPLIT, w=W):
    block = c.host.lease_block
    _i32(block, w.copy_armed)[0] = int(armed)
    _i32(block, w.split, w.lanes + 1)[:] = torch.tensor(split, dtype=torch.int32)


def _post(c, experts, row=0, *, captured=False, cpu=False):
    c.plan(experts, row)
    backend, plan = c.backends[row], c.plans[row]
    backend._stage_planned(plan)
    cpu_input = None
    if cpu:
        cpu_input = (torch.zeros(1, 64, device="cuda"), torch.ones(TOP_K, device="cuda"))
    c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=captured, cpu_input=cpu_input)
    torch.cuda.synchronize()
    return c.kinds(len(experts)), c.dev.lane_slot[: len(experts)].tolist()


def _chain_of_last_record(c):
    seq = int(c.dev.stats()["posted"]) & 0xFFFFFFFF
    record = W.demand_ring + (seq - 1) % W.demand_records * W.record_bytes
    page = c.page
    at = record + W.record_fields["chain"]
    return int(page[at : at + 8].view(torch.int64)[0])


@pytest.fixture
def idle(tmp_path):
    """No service thread: the test plays the host's words itself."""
    c = Chain(tmp_path, start=False, copy_engine=True)
    try:
        yield c
    finally:
        c.close()


@pytest.mark.parametrize("cpu_misses", [False, True])
@pytest.mark.parametrize("hit_copy", ["ce", "sm"])
def test_post_types_lanes_like_the_reference(tmp_path, hit_copy, cpu_misses):
    """200 random maps and plans: the CUDA typing is ram_slot_map.type_lanes, kind for kind and slot for slot."""
    c = Chain(tmp_path, start=False, copy_engine=True, hit_copy=hit_copy, cpu_misses=cpu_misses)
    try:
        hidden = 64
        c.dev.cpu_x_rows = torch.zeros((2, 2 * hidden), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=True)
        rng = random.Random(7)
        staging = rng.sample(range(8, 14), 6)
        _write_delta(c, 0, 1, staging)  # the attach delta's contents, before the device reads it
        for _ in range(200):
            ram = [-1] * EXPERTS
            for e, slot in zip(rng.sample(range(EXPERTS), rng.randint(0, W.lanes)), rng.sample(range(W.lanes), W.lanes)):
                ram[e] = slot  # distinct RAM slots, none of them staging, as the host keeps them
            c.dev.map_bulk_apply(torch.tensor([[0, e, s] for e, s in enumerate(ram)], dtype=torch.int32))
            experts = rng.sample(range(EXPERTS), rng.randint(1, TOP_K))
            got = _post(c, experts, captured=True, cpu=True)
            want = type_lanes(experts, ram, staging, SPLIT, captured=True, copy_armed=True, hit_copy=hit_copy,
                              cpu_on=True, cpu_misses=cpu_misses, lanes=W.lanes)
            assert (got[0], got[1]) == ([int(k) for k in want[0]], want[1]), (experts, ram, staging)
            chain = _chain_of_last_record(c)
            if chain:
                _write_delta(c, 0, chain, staging)  # the host's empty delta for this chain: the map is the bulk's
    finally:
        c.close()


def test_post_applies_the_pending_delta_once(idle):
    c = idle
    kinds, slots = _post(c, [5])  # the tag-1 attach delta is applied first: a miss into its first staging slot
    assert kinds == [LaneKind.MISS_GPU] and slots == [c.device_staging()[0]]
    assert _chain_of_last_record(c) == 2
    _write_delta(c, 0, 2, [3, 4, 5, 6, 7, 8], [(5, 2)])
    kinds, slots = _post(c, [5])
    assert kinds == [LaneKind.HIT_SM] and slots == [2]
    c.dev.map_bulk_apply(torch.tensor([[0, 5, 6]], dtype=torch.int32))
    torch.cuda.synchronize()
    assert _post(c, [5])[1] == [6], "tag 2 was applied again over the bulk entry"


def test_post_applies_a_full_delta(idle):
    """Every one of DELTA_MAX_ENTRIES entries lands, and the staging slots with them: the delta's loads cover the
    whole record, not just its first words."""
    c = idle
    _post(c, [5])  # applies the attach delta; the miss makes map chain 2
    entries = [(e, e % W.lanes) for e in range(W.delta_max_entries)]
    _write_delta(c, 0, 2, [8, 9, 10, 11, 12, 13], entries)
    kinds, slots = _post(c, [5])
    assert kinds == [LaneKind.HIT_SM] and slots == [5]
    assert c.device_map(0) == [e % W.lanes for e in range(EXPERTS)]
    assert c.device_staging(0)[:6] == [8, 9, 10, 11, 12, 13]


_TRAP_SCRIPT = """
import sys, time
import torch
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
c = Chain(sys.argv[1], timeout_ms=300, start=False)
c.plan([5])
c.backends[0]._stage_planned(c.plans[0])
b, p = c.backends[0], c.plans[0]
c.dev.post(0, b.planned, p.count, b.routes, p.slots)  # a miss: map chain 2, which the host never publishes
torch.cuda.synchronize()
c.plan([6])
b._stage_planned(p)
start = time.perf_counter()
c.dev.post(0, b.planned, p.count, b.routes, p.slots)
try:
    torch.cuda.synchronize()
    print("reached", flush=True)
except RuntimeError as error:
    print(f"trapped {time.perf_counter() - start:.3f} {error}", flush=True)
import os
os._exit(0)
"""


def test_post_waits_for_delta_then_traps_at_deadline(tmp_path):
    """Review Focus 1: the second post finds tag 1 against map chain 2. It waits for its deadline, then traps; it never
    types lanes from the stale map."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_TRAP_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=120,
    )
    assert "reached" not in result.stdout, result.stdout
    line = next((line for line in result.stdout.splitlines() if line.startswith("trapped")), None)
    assert line is not None, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    assert 0.3 <= float(line.split()[1]) < 10.0, line


_REPEATED_EXPERT_SCRIPT = """
import sys
import torch
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
c = Chain(sys.argv[1], start=False)
c.plan([5, 6])
b, p = c.backends[0], c.plans[0]
b._stage_planned(p)
b.planned[1] = 5  # a repeated expert: the reference raises, so the post must trap
c.dev.post(0, b.planned, p.count, b.routes, p.slots)
try:
    torch.cuda.synchronize()
    print("reached", flush=True)
except RuntimeError as error:
    print(f"trapped {error}", flush=True)
import os
os._exit(0)
"""


def test_post_traps_on_a_repeated_expert(tmp_path):
    """Review Focus 5: once type_lanes loads every lane before deciding, a plan the reference rejects still traps
    instead of being typed."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_REPEATED_EXPERT_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=120,
    )
    assert "reached" not in result.stdout, result.stdout
    assert any(line.startswith("trapped") for line in result.stdout.splitlines()), (
        result.returncode, result.stdout[-2000:], result.stderr[-2000:])


def test_miss_streams_from_the_staging_slot_byte_exact(tmp_path):
    c = Chain(tmp_path)
    try:
        c.plan([9])
        c.gather()
        snapshot = c.snapshot()
        torch.cuda.synchronize()
        c.check([9], snapshot)
        assert c.handled()
        staging_slot = c.dev.lane_slot[0].item()
        assert c.host.mapping(0)[9] == staging_slot
        c.plan([9])  # the next chain applies the delta: a hit at that slot
        c.gather()
        torch.cuda.synchronize()
        assert c.kinds(1) == [LaneKind.HIT_SM] and c.dev.lane_slot[0].item() == staging_slot
        assert c.device_map(0) == c.host.mapping(0)
    finally:
        c.close()


def _post_spill(c, experts, forced_from, flag, overflows, row=0):
    """A captured post whose lanes from forced_from on have no VRAM victim (dst -1), as DIRECT's spill leaves them."""
    c.plan(experts, row)
    backend, plan = c.backends[row], c.plans[row]
    # The live lanes keep their destination rows; plan.slots carries the previous call's -1 otherwise.
    plan.slots[: len(experts)] = torch.arange(len(experts), dtype=torch.int32, device=plan.slots.device)
    plan.slots[forced_from : len(experts)] = -1
    backend._stage_planned(plan)
    cpu_input = (torch.zeros(1, 64, device="cuda"), torch.ones(TOP_K, device="cuda"))
    c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=cpu_input,
               spill=(flag, overflows))
    torch.cuda.synchronize()
    count = int(plan.count[0])
    return count, c.kinds(count), c.dev.lane_slot[:count].tolist()


@pytest.mark.parametrize("armed", [True, False], ids=["armed", "unarmed"])
def test_post_spills_forced_lanes_like_the_reference(tmp_path, armed):
    """200 random maps, plans and spill points. Armed, every forced lane becomes a CPU lane whatever the split (a
    forced miss with slot -1, never a staging slot) and the count stands. Unarmed (Review Focus 1, the one overflow
    left), any forced lane overflows: the post serves the live prefix, writes its count, sets the flag and counts the
    overflow once. Mutations: type a forced lane by the split; give a forced miss a staging slot -- red."""
    c = Chain(tmp_path, start=False, copy_engine=True, hit_copy="ce", cpu_misses=False)
    try:
        hidden = 64
        c.dev.cpu_x_rows = torch.zeros((2, 2 * hidden), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=armed)
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        rng = random.Random(11)
        staging = rng.sample(range(8, 14), 6)
        _write_delta(c, 0, 1, staging)
        spilled = overflowed = 0
        for _ in range(200):
            ram = [-1] * EXPERTS
            for e, slot in zip(rng.sample(range(EXPERTS), rng.randint(0, W.lanes)), rng.sample(range(W.lanes), W.lanes)):
                ram[e] = slot
            c.dev.map_bulk_apply(torch.tensor([[0, e, s] for e, s in enumerate(ram)], dtype=torch.int32))
            experts = rng.sample(range(EXPERTS), rng.randint(1, TOP_K))
            forced_from = rng.randint(0, len(experts))
            flag.zero_()
            before = int(overflows.item())
            got = _post_spill(c, experts, forced_from, flag, overflows)
            ref = dict(captured=True, copy_armed=armed, hit_copy="ce", cpu_on=True, cpu_misses=False, lanes=W.lanes)
            try:
                kinds, slots = type_lanes(experts, ram, staging, SPLIT, forced_from=forced_from, **ref)
                want = (len(experts), [int(k) for k in kinds], slots, 0)
            except LaneOverflow:
                kinds, slots = type_lanes(experts[:forced_from], ram, staging, SPLIT, **ref)
                want = (forced_from, [int(k) for k in kinds], slots, 1)
            assert (*got, int(flag.item())) == want, (experts, ram, forced_from)
            assert int(overflows.item()) - before == want[3]
            spilled += want[3] == 0 and forced_from < len(experts)
            overflowed += want[3]
            chain = _chain_of_last_record(c)
            if chain:
                _write_delta(c, 0, chain, staging)
        assert spilled > 0 if armed else overflowed > 0
    finally:
        c.close()


def test_a_40_lane_post_makes_36_forced_misses_on_one_node_cpu_lanes(tmp_path):
    """Review Focus 2 at the record's full width: 36 distinct NVMe misses on one node of a 40-lane wire, 8 with VRAM
    victims. The 8 live misses take the node's 8 staging slots; the 28 forced ones are CPU misses with slot -1 (the
    host places them, Task 9). Count 36 stands, no flag, no trap: however many misses one node has, nothing overflows.
    Mutation: let a forced miss draw from staging -- the 9th traps."""
    w = lease.wire_layout(40)
    c = Chain(tmp_path, start=False, copy_engine=True, lanes=40, top_k=36, dst_rows=36, experts=48, capacity=24)
    try:
        c.dev.cpu_x_rows = torch.zeros((2, 128), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=True, split=[0] * (w.lanes + 1), w=w)
        staging = list(range(10, 18))
        _write_delta(c, 0, 1, staging, w=w)
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        experts = list(range(36))
        c.plan(experts)
        backend, plan = c.backends[0], c.plans[0]
        plan.slots[:8] = torch.arange(8, dtype=torch.int32, device=plan.slots.device)
        plan.slots[8:36] = -1
        backend._stage_planned(plan)
        cpu_input = (torch.zeros(1, 64, device="cuda"), torch.ones(36, device="cuda"))
        c.dev.post(0, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=cpu_input,
                   spill=(flag, overflows))
        torch.cuda.synchronize()
        assert (int(plan.count[0]), int(flag.item()), int(overflows.item())) == (36, 0, 0)
        assert c.kinds(36) == [int(LaneKind.MISS_GPU)] * 8 + [int(LaneKind.MISS_CPU)] * 28
        assert c.dev.lane_slot[:36].tolist() == staging + [-1] * 28
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
