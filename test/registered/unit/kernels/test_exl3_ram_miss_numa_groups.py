"""The RAM tier with two NUMA groups (spec 2026-10-03-numa-node-distributor-design, Part 3 and Testing 3): each
group serves only its home lanes (expert % 2), stages and evicts only in its own slots, and the combiner merges
the groups' map deltas into one per record. ChainSim plays the device with the node-aware reference typing."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, paused, ram_miss_setup

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROW = 1
EXPERTS = 8
HALVES = [[(0, 4)] * 2, [(4, 8)] * 2]  # [group][row] -> (lo, hi): node 0 slots 0-3, node 1 slots 4-7


def _host(tmp_path, *, ranges=HALVES, capacity=8, staging=2):
    s = ram_miss_setup(tmp_path, capacity=capacity, experts=EXPERTS)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    host = ExpertStreamHost(
        s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), variant="instr",
        node_ranges=ranges,
    )
    host.reserve_staging(staging)
    return s, page, host, ChainSim(host, page, s.slabs)


def _serve(sim, host, experts):
    req = sim.post(ROW, experts)
    assert host.pump() == 1 and sim.wait_served(req)
    return req


def _lists(staging):
    return staging[:8], staging[8:]


def test_reserve_staging_takes_each_groups_lowest_slots_and_publishes_every_nodes_list(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        tag, staging, entries = sim.delta(ROW)
        assert (tag, entries) == (1, [])
        assert _lists(staging) == ([0, 1] + [-1] * 6, [4, 5] + [-1] * 6)
    finally:
        host.stop()


def test_each_group_serves_only_its_home_lanes(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        _serve(sim, host, [2, 1])
        mapping = host.mapping(ROW)
        assert 0 <= mapping[2] < 4 and 4 <= mapping[1] < 8
        assert host.group_counters(0)["rows_read"] == 1 and host.group_counters(1)["rows_read"] == 1
    finally:
        host.stop()


def test_misses_homed_on_one_node_publish_one_delta_with_the_other_nodes_list_unchanged(tmp_path):
    """Review Focus 1, host side: only group 1 reports, and the delta it writes carries node 0's list as reserved."""
    s, page, host, sim = _host(tmp_path)
    try:
        req = _serve(sim, host, [1, 3])
        tag, staging, entries = sim.delta(ROW)
        assert tag == req.chain == 2
        assert _lists(staging) == ([0, 1] + [-1] * 6, [6, 7] + [-1] * 6)
        assert entries == [(1, 4), (3, 5)]
    finally:
        host.stop()


def test_both_nodes_missing_in_one_record_yield_exactly_one_merged_delta(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        req = _serve(sim, host, [2, 1])
        tag, staging, entries = sim.delta(ROW)
        assert tag == req.chain == 2, "one delta per record, not one per group"
        assert _lists(staging) == ([2, 1] + [-1] * 6, [6, 5] + [-1] * 6)
        assert entries == [(2, 0), (1, 4)], "node 0's entries, then node 1's"
        _serve(sim, host, [2, 1])  # both hits now: the next post applied the merged delta
        assert host.mapping(ROW)[2] == 0 and host.mapping(ROW)[1] == 4
    finally:
        host.stop()


def test_victims_come_only_from_the_groups_range(tmp_path):
    s, page, host, sim = _host(tmp_path, staging=1)
    try:
        for expert in (0, 2, 4, 1, 3, 5):  # fills both ranges: three mapped rows and one staging slot each
            _serve(sim, host, [expert])
        _serve(sim, host, [7])
        mapping = host.mapping(ROW)
        assert mapping[1] == -1, "node 1's LRU row is the victim"
        assert all(0 <= mapping[e] < 4 for e in (0, 2, 4)), "node 0 lost nothing"
        assert 4 <= mapping[7] < 8
        _serve(sim, host, [6])
        assert host.mapping(ROW)[0] == -1 and 0 <= host.mapping(ROW)[6] < 4
    finally:
        host.stop()


def test_no_slot_outside_a_groups_range_is_ever_staged_or_taken(tmp_path):
    """Review Focus 5: slot 4 straddles the seam and belongs to no group. Forty random records of hits and misses on
    both nodes never stage it, map it or evict into it."""
    import random

    rng = random.Random(4)
    s, page, host, sim = _host(tmp_path, ranges=[[(0, 4)] * 2, [(5, 9)] * 2], capacity=9)
    try:
        for _ in range(40):
            experts = rng.sample(range(EXPERTS), rng.randint(1, 2))
            _serve(sim, host, experts)
            assert 4 not in sim.staging(ROW)
            state, expert, _ = host.slot_info(ROW)[4]
            assert (state, expert) == (0, -1)
        assert 4 not in host.mapping(ROW)
    finally:
        host.stop()


def test_a_lane_less_record_stamps_each_experts_slot_with_its_own_groups_tick(tmp_path):
    """Review S5: both groups see the record's protect ids but each stamps only its home experts, with its own clock
    (a foreign stamp would carry the other group's tick and race its owner)."""
    s, page, host, sim = _host(tmp_path, staging=1)
    try:
        for expert in (2, 0, 4, 1):  # node 0's clock reaches 3, node 1's 1
            _serve(sim, host, [expert])
        stamp_before = {e: host.slot_info(ROW)[host.mapping(ROW)[e]][2] for e in (2, 1)}
        req = sim.post(ROW, [], protect=[2, 1])
        assert host.pump() == 1 and sim.wait_handled(req)
        info = host.slot_info(ROW)
        assert info[host.mapping(ROW)[2]][2] > stamp_before[2] and info[host.mapping(ROW)[2]][2] >= 4
        assert info[host.mapping(ROW)[1]][2] == 2, "node 1's own clock, not node 0's"
    finally:
        host.stop()


def test_every_group_parks_for_a_pause_and_serves_after_it(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        host.start_thread(fatal_wait_s=60.0)
        req = sim.post(ROW, [2, 1])
        assert sim.wait_served(req, timeout_s=5.0)
        with paused(host):
            assert [state for state, _, _ in host.slot_info(ROW)].count(2) == 2
        req = sim.post(ROW, [4, 3])
        assert sim.wait_served(req, timeout_s=5.0)
        assert sim.wait_handled(req)
    finally:
        host.stop()


def test_a_group_with_no_slots_in_a_row_opens_and_its_staging_is_refused(tmp_path):
    """Node 1's range of row 0 is empty (all of the row's slots are node 0's). The tier opens, as it did before the
    groups, and reserve_staging refuses the row with its message. Mutation: the empty range is refused at open
    ("outside the row"), from the wrong call. A capacity-0 row itself never reaches the tier: the slab table refuses
    it first (_slab_table), at db8b55e497 too."""
    s = ram_miss_setup(tmp_path, capacity=2, experts=EXPERTS)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    host = ExpertStreamHost(
        s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), variant="instr",
        node_ranges=[[(0, 2)] * 2, [(2, 2)] * 2],
    )
    try:
        with pytest.raises(RuntimeError, match="row 0 has too few slots to stage in group 1"):
            host.reserve_staging(1)
    finally:
        host.stop()


def test_ranges_that_overlap_or_leave_the_row_are_refused(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=8, experts=EXPERTS)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    for ranges, match in [([[(0, 5)] * 2, [(4, 8)] * 2], "overlap"), ([[(0, 4)] * 2, [(4, 9)] * 2], "outside")]:
        with pytest.raises(ValueError, match=match):
            ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32),
                             variant="instr", node_ranges=ranges)


def test_the_native_start_thread_refuses_a_reserved_core_of_any_group(tmp_path):
    """The raw FFI, which ExpertStreamHost.start_thread's own check never reaches. Mutation: the native refusal is
    gone, so a thread pins itself to a core that takes NVMe completion interrupts."""
    s, page, host, sim = _host(tmp_path)
    try:
        for cores in ([64, 0], [0, 71]):
            with pytest.raises(RuntimeError, match=r"cores 64-71 are reserved \(NVMe completion interrupts"):
                host._module.expert_stream_start_thread(
                    host.handle, torch.tensor(cores, dtype=torch.int64), int(1e9), 5_000_000, 0
                )
            assert not host.threaded
    finally:
        host.stop()


def test_the_native_start_thread_warns_when_an_inherited_affinity_covers_the_reserved_cores(tmp_path):
    """Mutation: the warning is gone. The subprocess widens its own affinity to core 64 and starts the threads with
    -1 (inherit); the C++ fprintf goes to the real stderr, so it is read from the child's."""
    if (os.cpu_count() or 0) < 72:
        pytest.skip("needs cores 64-71")
    body = "import os; os.sched_setaffinity(0, {0, 64})\nhost.start_thread(fatal_wait_s=60.0)\nhost.stop()\nprint('reached')\n"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT) + body, str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert "reached" in result.stdout, result.stderr
    assert "run under taskset -c 0-63" in result.stderr


_SCRIPT = """
import pathlib, sys, torch
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
s = ram_miss_setup(pathlib.Path(sys.argv[1]), capacity=8, experts=8)
page = new_page(pin=False, wire=wire_layout(8, 2))
host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 8), -1, dtype=torch.int32), variant="instr",
                        node_ranges=[[(0, 4)] * 2, [(4, 8)] * 2])
host.reserve_staging(2)
sim = ChainSim(host, page, s.slabs)
"""


def test_a_miss_on_another_nodes_staging_slot_fail_stops(tmp_path):
    """A device that put node 1's miss in node 0's staging slot 0 would have it read into node 0's memory: group 1
    checks its own list and fail-stops. A row-wide list would accept slot 0 (mutant: red)."""
    body = "sim.post(1, [1], kinds=[LaneKind.MISS_GPU], slots=[0])\nhost.pump()\nprint('reached')\n"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT) + body, str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert_aborted(result, "is not a staging slot")


_LAGGING = textwrap.dedent(_SCRIPT) + """
for i in range(18):  # node 0's misses only, served by group 0 alone: group 1 falls past the ring behind
    req = sim.post(1, [(0, 2, 4, 6)[i % 4]])
    assert host.pump_group(0) == 1 and sim.wait_served(req)
req = sim.post(1, [1])  # group 1's first miss, chain 20 against the row's chain it last saw
for _ in range(40):
    host.pump()
    if sim.wait_served(req):
        break
assert sim.wait_served(req)
print('reached')
"""


def test_a_group_the_device_lapped_serves_its_next_miss(tmp_path):
    """A record with no lane of a group's node is never waited on for that group, so it can fall a ring behind
    (delta 10); its next own miss must still pass the chain check, against the row's published chain."""
    result = subprocess.run(
        [sys.executable, "-c", _LAGGING, str(tmp_path)], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0 and "reached" in result.stdout, result.stderr[-2000:]


_STALE_HOT = (
    textwrap.dedent(_SCRIPT).replace("variant=\"instr\",", "variant=\"instr\", hot_page=new_hot_page(8, pin=False),")
    .replace("import ExpertStreamHost, new_page", "import ExpertStreamHost, new_hot_page, new_page")
    + """
sim.post(1, [0], hot_seq=7)  # node 0's miss with no hot record: group 0 would fail-stop, group 1 has no work in it
assert host.pump_group(1) == 1
print(host.group_counters(1)['overruns'])
print('reached')
"""
)


def test_a_group_with_no_host_lane_skips_a_record_whose_hot_set_is_gone(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _STALE_HOT, str(tmp_path)], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0 and result.stdout.split() == ["1", "reached"], result.stderr[-2000:]
