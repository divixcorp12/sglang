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
    host.reserve_staging(K)
    yield s, host, ChainSim(host, page, s.slabs)
    host.stop()


def _serve(host, sim, experts, **kw):
    req = sim.post(0, experts, **kw)
    assert host.pump() == 1
    assert sim.wait_served(req)
    return req


def test_reserve_staging_takes_free_slots_and_publishes_tag_one_before_any_post(world):
    """Mutation: the reservation maps the staging slots, reserves the wrong count, evicts, or publishes no tag-1 delta."""
    _, host, sim = world
    tag, staging, entries = sim.delta(0)
    assert tag == 1 and entries == []
    assert sorted(s for s in staging if s >= 0) == [0, 1] and staging[K:] == [-1] * (8 - K)
    assert host.mapping(0) == [-1] * EXPERTS
    assert host.counters()["evictions"] == 0
    assert host.take_bulk_delta().numel() == 0  # nothing was evicted or mapped: no bulk delta
    assert [state for state, _, _ in host.slot_info(0)] == [3] * K + [0] * (CAPACITY - K)


def test_a_full_fill_after_the_reservation_never_takes_a_staging_slot(world):
    """Mutation: take_slot_locked treats a kStaging slot as free or as a victim."""
    _, host, sim = world
    staging = [s for s in sim.staging(0) if s >= 0]
    taken = set()
    for expert in range(EXPERTS):  # far more than the CAPACITY - K mappable slots: admissions evict
        slot, _ = host.assign(0, expert)
        assert slot not in staging
        taken.add(slot)
    assert taken == set(range(CAPACITY)) - set(staging)
    assert [state for state, _, _ in host.slot_info(0)][:K] == [3] * K
    assert sim.delta(0)[:2] == (1, sim.staging(0))  # a fill publishes no delta of its own


@pytest.mark.parametrize("capacity, want, staged", [(6, 8, 5), (6, 2, 2), (2, 8, 1)])
def test_reserve_staging_clamps_to_the_rows_capacity_minus_one(tmp_path, capacity, want, staged):
    s = ram_miss_setup(tmp_path, capacity=capacity, layers=1, experts=EXPERTS)
    host = ExpertStreamHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full((1, EXPERTS), -1, dtype=torch.int32), variant="instr"
    )
    try:
        host.reserve_staging(want)
        assert sum(state == 3 for state, _, _ in host.slot_info(0)) == staged
    finally:
        host.stop()


def test_reserve_staging_refuses_a_row_with_one_slot(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=1, layers=1, experts=EXPERTS)
    host = ExpertStreamHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full((1, EXPERTS), -1, dtype=torch.int32), variant="instr"
    )
    try:
        with pytest.raises(RuntimeError, match="row 0 has too few slots to stage"):
            host.reserve_staging(K)
    finally:
        host.stop()


def test_reserve_staging_is_once_and_before_any_slot_is_filled(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=CAPACITY, layers=1, experts=EXPERTS)
    host = ExpertStreamHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full((1, EXPERTS), -1, dtype=torch.int32), variant="instr"
    )
    try:
        host.assign(0, 3)
        with pytest.raises(RuntimeError, match="before any slot is filled"):
            host.reserve_staging(K)
        fresh = ExpertStreamHost(
            s.tables, page=new_page(pin=False), slot_map=torch.full((1, EXPERTS), -1, dtype=torch.int32),
            variant="instr",
        )
        try:
            fresh.reserve_staging(K)
            with pytest.raises(RuntimeError, match="once"):
                fresh.reserve_staging(K)
        finally:
            fresh.stop()
    finally:
        host.stop()


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


def test_admission_never_takes_staging(world):
    """Mutation: take_admit_slot_locked or assign may take a kStaging slot (then a later miss reads over a mapped row)."""
    _, host, sim = world
    staging = {s for s in sim.staging(0) if s >= 0}
    for e in range(10):  # eager admissions past the tier's size
        host.assign(0, e)
    assert not staging & {s for s in host.mapping(0) if s >= 0}
    assert sum(state == 3 for state, _, _ in host.slot_info(0)) == K


def test_bulk_after_pending_decode_delta(world):
    """The device applies the row's pending decode delta before the bulk entries, and then holds the host's map.
    Mutation: the bulk is taken before a decode delta's mirror updates (then the replica and the host disagree)."""
    _, host, sim = world
    _serve(host, sim, [3])  # decode delta tag 2 is published, not yet applied by the device
    host.assign(0, 5)
    bulk = host.take_bulk_delta().tolist()
    assert [0, 5, host.mapping(0)[5]] in bulk
    sim.apply_bulk_like_device(bulk)
    assert sim.replica.ram_slot[0] == host.mapping(0)
    assert host.take_bulk_delta().numel() == 0, "a bulk delta is taken once"


def test_a_failed_fills_unmaps_are_in_the_bulk_delta(world):
    """R1-5: take_bulk_delta joins a running fill first, so a failed fill's unmaps of its unlanded rows reach the
    device. Mutation: the bulk is taken before the fill's epilogue."""
    s, host, sim = world
    path = s.tables.paths[int(s.tables.extents[0, 6, 0, 0])]
    with open(path, "r+b") as f:
        f.truncate(int(s.tables.extents[0, 6, 0, 1]) + 100)
    slots, _ = host.fill_begin(0, [6])
    bulk = host.take_bulk_delta().tolist()
    assert [0, 6, slots[0]] in bulk and [0, 6, -1] in bulk  # mapped by the claim, unmapped by the failed epilogue
    sim.apply_bulk_like_device(bulk)
    assert sim.replica.ram_slot[0][6] == -1 == host.mapping(0)[6]


def test_a_pause_first_serves_every_record_posted_before_it(world):
    """An all-HIT_SM chain never waits on the host, so its record may still be unread when an eager path pauses the
    service. Parked unread, it would be checked after the eager path moved its expert, and the host would fail-stop on
    a correct device. The service drains the ring before it parks. Mutation: park without draining."""
    _, host, sim = world
    _serve(host, sim, [3])
    host.start_thread()
    for _ in range(40):
        time.sleep(0.05)  # past the spin budget: the service polls from its idle sleep, so the pause can land first
        req = sim.post(0, [3])
        assert req.kinds == [LaneKind.HIT_SM]
        host.pause(5.0)
        try:
            assert host.handled_through() == req.seq, "a record posted before the pause was left unread"
        finally:
            host.resume()


def test_closing_admission_stops_new_service(world):
    _, host, sim = world
    host.close_admission()
    sim.post(0, [3])
    assert host.pump() == 0


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
        req = sim.post(0, [2], chain=3)
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "map chain")


def test_a_whole_record_with_a_kind_the_device_never_writes_aborts(tmp_path):
    """A record that passes its seqlock but names kind 0 is not torn: it is malformed, and the host fail-stops instead
    of skipping it as an overrun (the device would wait on it until S's deadline). Mutation: malformed is an overrun."""
    out = _script(tmp_path, """
        sim.post(0, [2], kinds=[0], slots=[0])
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "malformed record")


def test_failed_read_aborts_with_fatal(tmp_path):
    out = _script(tmp_path, """
        host.inject(0, True, 0)
        req = sim.post(0, [2])
        host.pump()
        print("reached")
    """)
    assert_aborted(out, "a test fault failed the read")
