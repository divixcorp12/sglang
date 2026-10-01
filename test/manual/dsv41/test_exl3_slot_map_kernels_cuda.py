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
from sglang.kernels.ops.moe.expert_stream_transport import DEMAND_RECORDS, DEMAND_RING, RECORD_BYTES, RECORD_FIELDS  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind, type_lanes  # noqa: E402

SPLIT = [0, 1, 1, 2, 3, 3, 4, 5, 5]


def _i32(t, offset, count=1):
    return t[offset : offset + 4 * count].view(torch.int32)


def _write_delta(c, row, tag, staging, entries=()):
    """The host's delta publication, done by the test: payload, then the tag (x86 keeps tensor stores in order)."""
    base = lease.DELTA_BASE + row * lease.DELTA_STRIDE
    f = lease.DELTA_FIELDS
    block = c.host.lease_block
    _i32(block, base + f["count"])[0] = len(entries)
    _i32(block, base + f["staging"], lease.LANES)[:] = torch.tensor(list(staging) + [-1] * (lease.LANES - len(staging)))
    for i, (expert, slot) in enumerate(entries):
        _i32(block, base + f["entries"] + 8 * i, 2)[:] = torch.tensor([expert, slot])
    block[base + f["tag"] : base + f["tag"] + 8].view(torch.int64)[0] = tag


def _set_host_words(c, *, armed, split=SPLIT):
    block = c.host.lease_block
    _i32(block, lease.COPY_ARMED)[0] = int(armed)
    _i32(block, lease.SPLIT, lease.LANES + 1)[:] = torch.tensor(split, dtype=torch.int32)


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
    record = DEMAND_RING + (seq - 1) % DEMAND_RECORDS * RECORD_BYTES
    page = c.page
    lo = int(_i32(page, record + RECORD_FIELDS["chain"])[0]) & 0xFFFFFFFF
    hi = int(_i32(page, record + RECORD_FIELDS["chain_hi"])[0]) & 0xFFFFFFFF
    return hi << 32 | lo


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
            for e, slot in zip(rng.sample(range(EXPERTS), rng.randint(0, 8)), rng.sample(range(8), 8)):
                ram[e] = slot  # distinct RAM slots, none of them staging, as the host keeps them
            c.dev.map_bulk_apply(torch.tensor([[0, e, s] for e, s in enumerate(ram)], dtype=torch.int32))
            experts = rng.sample(range(EXPERTS), rng.randint(1, TOP_K))
            got = _post(c, experts, captured=True, cpu=True)
            want = type_lanes(experts, ram, staging, SPLIT, captured=True, copy_armed=True, hit_copy=hit_copy,
                              cpu_on=True, cpu_misses=cpu_misses)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
