"""The real service on the slot-map protocol, driven by a Python stand-in for the device (CPU): staging slots, victims
chosen at record time, map deltas, and the host's checks of what the device typed (LEASE_PROTOCOL.md).

The stand-in (sglang/test/dsv41_chain_sim.py) types lanes with the Python reference; each test names the mutation of
the service it must fail under.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup, run_host_script, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

EXPERTS = 16
CAPACITY = 6
K = 2


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def world(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=CAPACITY, layers=1, experts=EXPERTS)
    page = new_page(pin=False)
    host = ExpertStreamHost(
        s.tables, page=page, slot_map=torch.full((1, EXPERTS), -1, dtype=torch.int32), variant="instr"
    )
    host.attach_row(0, K)
    yield s, host, ChainSim(host, page, s.slabs)
    host.stop()


def _serve(host, sim, experts, **kw):
    req = sim.post(0, experts, **kw)
    assert host.pump() == 1
    assert sim.wait_served(req)
    return req


def test_attach_reserves_k_staging_slots_and_publishes_tag_one(world):
    """Mutation: attach maps the staging slots, reserves the wrong count, or publishes no tag-1 delta."""
    _, host, sim = world
    tag, staging, entries = sim.delta(0)
    assert tag == 1 and entries == []
    assert sorted(s for s in staging if s >= 0) == [0, 1] and staging[K:] == [-1] * (8 - K)
    assert host.mapping(0) == [-1] * EXPERTS


def test_attach_row_is_once_per_row(world):
    _, host, _ = world
    with pytest.raises(RuntimeError, match="command failed"):
        host.attach_row(0, K)


def test_miss_reads_into_staging_and_the_delta_maps_it(world):
    """Mutation: the read lands elsewhere than the named staging slot, or the delta omits the insert."""
    s, host, sim = world
    staging = sim.staging(0)
    req = _serve(host, sim, [3])
    assert req.kinds == [LaneKind.MISS_GPU] and req.slots == [staging[0]] and req.chain == 2
    tag, after, entries = sim.delta(0)
    assert tag == 2 and entries == [(3, staging[0])]
    assert staging[0] not in after and len([x for x in after if x >= 0]) == K
    landed = sim.read_slot(0, staging[0])
    reference = s.reference(0, [3])
    assert all(same_bytes(landed[n], reference[n][0]) for n in EXL3_STREAMED_NAMES)
    assert host.mapping(0)[3] == staging[0]  # the host mirror, once the bytes landed


def test_hit_after_insert_uses_the_ram_slot(world):
    """Mutation: the host refuses a hit the delta mapped, or the next chain does not see the delta."""
    _, host, sim = world
    first = _serve(host, sim, [3])
    req = sim.post(0, [3])
    assert host.pump() == 1
    assert req.kinds == [LaneKind.HIT_SM] and req.slots == first.slots and req.chain == 0


def test_victim_is_never_routed_hot_or_staging(world):
    """Mutation: take_victim_locked ignores the hot set, the request's routes, or the staging state."""
    _, host, sim = world
    for e in range(4):  # capacity 6, K 2: the four mappable slots now hold 0..3, 0 the oldest
        _serve(host, sim, [e])
    host.set_hot(0, [0])  # the oldest is VRAM-hot and 1 is routed: the victim must be 2
    _serve(host, sim, [9], protect=[9, 1])
    _, staging, entries = sim.delta(0)
    assert [e for e, slot in entries if slot == -1] == [2]
    assert host.mapping(0)[2] == -1 and host.mapping(0)[9] >= 0


def test_no_victim_skips_ram_insert(world):
    """Mutation: a miss with no evictable slot aborts, maps over a hot expert, or changes the staging list."""
    _, host, sim = world
    for e in range(4):
        _serve(host, sim, [e])
    host.set_hot(0, [0, 1, 2, 3])
    before = sim.staging(0)
    req = _serve(host, sim, [7])
    tag, staging, entries = sim.delta(0)
    assert tag == req.chain and entries == [] and staging == before
    assert host.counters()["ram_insert_skipped"] == 1
    assert host.mapping(0)[7] == -1


def test_two_misses_take_both_staging_slots_and_two_victims(world):
    _, host, sim = world
    for e in range(4):
        _serve(host, sim, [e])
    staging = sim.staging(0)
    req = _serve(host, sim, [10, 11])
    assert req.slots == staging[:2]
    _, after, entries = sim.delta(0)
    assert sorted(e for e, slot in entries if slot == -1) == [0, 1]  # the two oldest
    assert {(10, staging[0]), (11, staging[1])} <= set(entries)
    assert not set(staging[:2]) & set(after)


def test_unarmed_record_only_stamps(world):
    """A record with no lanes refreshes recency and changes no map."""
    _, host, sim = world
    for e in range(4):
        _serve(host, sim, [e])
    sim.post(0, [], protect=[0])  # 0 is now the most recent: the next victim is 1
    assert host.pump() == 1
    _serve(host, sim, [12])
    assert [e for e, slot in sim.delta(0)[2] if slot == -1] == [1]


def test_delta_published_before_the_read(world):
    """Mutation: the delta is published after the read, so a slow read would leave the device's next post waiting."""
    _, host, sim = world
    host.start_thread()
    host.inject(0.3)  # every read sleeps 300 ms first
    req = sim.post(0, [3])
    deadline = time.monotonic() + 0.25
    while sim.delta(0)[0] != req.chain and time.monotonic() < deadline:
        time.sleep(1e-3)
    assert sim.delta(0)[0] == req.chain and not sim.served(req)
    assert sim.wait_served(req, timeout_s=5.0)


def _script(tmp_path, body):
    return run_host_script(tmp_path, body, capacity=4)


def test_a_hit_the_tier_does_not_hold_aborts(tmp_path):
    """Mutation: the host serves a hit lane whose slot does not hold its expert (a stale device map)."""
    out = _script(tmp_path, """
        req = sim.post(0, [2], kinds=[2], slots=[3])
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "the tier does not")


def test_a_miss_outside_the_staging_list_aborts(tmp_path):
    out = _script(tmp_path, """
        req = sim.post(0, [2], kinds=[4], slots=[3])
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "not a staging slot")


def test_a_map_chain_out_of_order_aborts(tmp_path):
    """Mutation: the host accepts a record whose chain skips one, so its deltas and the device's map diverge."""
    out = _script(tmp_path, """
        sim.replica.map_chain[0] += 1
        sim.replica.map_applied[0] = sim.replica.map_chain[0]
        req = sim.post(0, [2])
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "map chain")


def test_failed_read_aborts_with_fatal(tmp_path):
    out = _script(tmp_path, """
        host.inject(0, True, 0)
        req = sim.post(0, [2])
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "a test fault failed the read")
