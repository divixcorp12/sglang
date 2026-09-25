"""Native prefetch, the service's half (CPU; plan 2026-09-25-dsv41-native-prefetch): leases, skips, priority, judging.

The device posts one request on the prefetch page (row, expert, destination slot, then a tagged generation). The
service leases the row's pinned slot and hands a copy job to the copy thread, or skips it without reading anything.
The CPU copy backend lands a job's bytes only when the test releases its mark, so each test can hold a copy in flight
and look at what the service published meanwhile. The device is LeaseSim plus direct page writes.
"""

import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import (
    PREFETCH_FIELDS,
    PREFETCH_SKIP_REASONS,
    PREFETCH_TAG_COPIED,
    PREFETCH_TAG_REQUEST,
    PREFETCH_TAG_SKIPPED,
    Exl3RamMissHost,
    new_page,
    new_prefetch_page,
    page_word,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROW = 1
DST_ROWS = 6
MASK56 = (1 << 56) - 1


class Prefetcher:
    """The plan and commit kernels' page traffic, from Python."""

    def __init__(self, page):
        self.page, self.gen = page, 0

    def _i32(self, name):
        offset = PREFETCH_FIELDS[name]
        return self.page[offset : offset + 4].view(torch.int32)

    def _u64(self, name):
        offset = PREFETCH_FIELDS[name]
        return self.page[offset : offset + 8].view(torch.int64)

    def post(self, row, expert, dst):
        self.gen += 1
        self._i32("req_row")[0] = row
        self._i32("req_expert")[0] = expert
        self._i32("req_dst")[0] = dst
        self._u64("req_gen")[0] = (PREFETCH_TAG_REQUEST << 56) | self.gen
        return self.gen

    def done(self):
        """(tag, generation, reason) of the done line."""
        word = int(self._u64("done_gen")[0]) & ((1 << 64) - 1)
        return word >> 56, word & MASK56, int(self._i32("done_reason")[0])


def _setup(tmp_path, *, copy_engine=True, arm=True):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False)
    host = Exl3RamMissHost(
        s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False, pack_workers=2
    )
    host.enable_lease_mode()
    host.enable_two_phase()
    host.enable_piece_stream()
    dst = None
    if copy_engine:
        host.enable_copy_engine(-1, spin_us=200)
        dst = {n: torch.zeros((DST_ROWS,) + tuple(slab.shape[1:]), dtype=slab.dtype) for n, slab in s.slabs[ROW].items()}
        table = torch.tensor(
            [[slab.data_ptr(), dst[n].data_ptr(), slab[0].numel() * slab.element_size()] for n, slab in s.slabs[ROW].items()],
            dtype=torch.int64,
        )
        host.set_copy_table(ROW, table, DST_ROWS)
        if arm:
            host.arm_copy_engine()
    pf_page = new_prefetch_page(pin=False)
    if copy_engine:
        host.enable_native_prefetch(pf_page)
    return s, page, host, LeaseSim(host, page, s.slabs), Prefetcher(pf_page), dst


def _load(sim, host, experts):
    """Make ``experts`` resident in row ROW through a plain request, and retire its leases."""
    req = sim.post(ROW, experts)
    assert host.pump() == 1
    ctx = []
    for lane in range(len(req.lanes)):
        result = sim.row_result(req, lane)
        ctx.append((result["host_slot"], result["slot_generation"]))
    sim.ack(req, type("W", (), {"go": len(ctx), "ctx": ctx})())
    sim.deliver()
    host.pump()


def _slot_of(host, expert):
    return host.mapping(ROW)[expert]


def _rows_equal(dst, slabs, dst_row, host_slot):
    return all(torch.equal(dst[n][dst_row].view(torch.uint8), slabs[n][host_slot].view(torch.uint8)) for n in dst)


def test_native_prefetch_needs_the_copy_engine(tmp_path):
    s, page, host, sim, pf, _ = _setup(tmp_path, copy_engine=False)
    try:
        with pytest.raises(RuntimeError, match="copy engine"):
            host.enable_native_prefetch(new_prefetch_page(pin=False))
    finally:
        host.stop()


def test_a_ready_row_is_leased_copied_and_published_only_after_the_observed_completion(tmp_path):
    """Mutants: publish COPIED at the grant, or release the lease at the grant -- red on the checks made while the
    mark is held."""
    s, page, host, sim, pf, dst = _setup(tmp_path)
    try:
        _load(sim, host, [3])
        slot = _slot_of(host, 3)
        gen = pf.post(ROW, 3, 4)
        assert host.pump() == 3
        assert host.prefetch_lease() == (True, ROW, slot)
        assert host.slot_info(ROW)[slot][2] == 1, "the prefetch did not lease its pinned slot"
        time.sleep(0.05)
        assert pf.done()[1] != gen, "the done word was published before the copy completed"
        assert not any(dst[n][4].view(torch.uint8).any() for n in dst)
        host.copy_engine_release(1)
        assert host.copy_engine_idle(5.0)
        assert pf.done() == (PREFETCH_TAG_COPIED, gen, 0)
        assert _rows_equal(dst, s.slabs[ROW], 4, slot)
        assert host.prefetch_lease()[0] is False and host.slot_info(ROW)[slot][2] == 0
        c = host.counters()
        assert (c["prefetch_requests"], c["prefetch_issued"], c["prefetch_copied"]) == (1, 1, 1), c
        assert host.pump() == 0, "a served request was served again"
    finally:
        host.stop()


@pytest.mark.parametrize("case", ["not_ready", "unarmed", "invalid_dst", "invalid_expert"])
def test_a_request_the_service_cannot_copy_is_skipped_without_a_lease_or_a_read(tmp_path, case):
    s, page, host, sim, pf, dst = _setup(tmp_path, arm=case != "unarmed")
    try:
        _load(sim, host, [3])
        rows_read = host.counters()["rows_read"]
        expert = {"not_ready": 5, "invalid_expert": 99}.get(case, 3)
        gen = pf.post(ROW, expert, DST_ROWS if case == "invalid_dst" else 2)
        assert host.pump() == 3
        reason = {"not_ready": "not_ready", "unarmed": "unarmed"}.get(case, "invalid")
        assert pf.done() == (PREFETCH_TAG_SKIPPED, gen, PREFETCH_SKIP_REASONS[reason])
        c = host.counters()
        assert c["prefetch_issued"] == 0 and host.copy_engine_marked() == 0 and host.prefetch_lease()[0] is False
        assert c["rows_read"] == rows_read, "a prefetch read NVMe"
        assert c[f"prefetch_skipped_{reason}"] == 1
        assert _slot_of(host, 5) < 0
    finally:
        host.stop()


def test_demand_is_served_before_a_prefetch_posted_at_the_same_time(tmp_path):
    s, page, host, sim, pf, dst = _setup(tmp_path)
    try:
        _load(sim, host, [2, 3])
        gen = pf.post(ROW, 3, 4)
        req = sim.post(ROW, [2])
        assert host.pump() == 1 and page_word(page, "demand_done") == req.seq
        assert pf.done()[1] != gen
        assert host.pump() == 3
    finally:
        host.stop()


def test_a_prefetch_job_waits_on_the_copy_thread_while_a_demand_job_is_in_flight(tmp_path):
    """Demand before prefetch on the link: the copy thread issues no prefetch copy while a demand job it issued has
    not completed, and issues it at once afterwards. Mutant: issue prefetch jobs in arrival order -- red on the mark
    count."""
    s, page, host, sim, pf, dst = _setup(tmp_path)
    try:
        _load(sim, host, [2, 3])
        demand = sim.post(ROW, [2], dst=[0], copy_engine=True)
        assert host.pump() == 1
        deadline = time.time() + 5
        while host.copy_engine_marked() < 1 and time.time() < deadline:
            time.sleep(0.002)
        assert host.copy_engine_marked() == 1  # the demand job is on the stream, its mark held
        gen = pf.post(ROW, 3, 4)
        assert host.pump() == 3
        time.sleep(0.05)
        assert host.copy_engine_marked() == 1, "a prefetch was issued while a demand job was in flight"
        assert host.counters()["prefetch_held"] >= 1
        host.copy_engine_release(1)  # the demand job completes
        deadline = time.time() + 5
        while host.copy_engine_marked() < 2 and time.time() < deadline:
            time.sleep(0.002)
        assert host.copy_engine_marked() == 2, "the held prefetch was never issued"
        host.copy_engine_release(1)
        assert host.copy_engine_idle(5.0)
        assert sim.copy_done(demand)[1] == demand.gen and pf.done()[:2] == (PREFETCH_TAG_COPIED, gen)
    finally:
        host.stop()


def test_a_demand_whose_only_victim_is_under_a_prefetch_copy_defers_until_it_completes(tmp_path):
    """The prefetch lease is an E1 lease like any other: the pinned slot a prefetch reads is never a victim."""
    s, page, host, sim, pf, dst = _setup(tmp_path)
    try:
        _load(sim, host, [0, 1, 2, 3])
        slot = _slot_of(host, 3)
        expected = {n: s.slabs[ROW][n][slot].clone() for n in dst}
        pf.post(ROW, 3, 4)
        assert host.pump() == 3
        demand = sim.post(ROW, [4], protect=[0, 1, 2, 4])
        assert host.pump() == 0, "a leased slot was evicted under a prefetch copy"
        assert host.counters()["deferred"] >= 1
        host.copy_engine_release(1)
        assert host.copy_engine_idle(5.0)
        assert all(torch.equal(dst[n][4].view(torch.uint8), expected[n].view(torch.uint8)) for n in dst)
        assert host.pump() == 1 and page_word(page, "demand_done") == demand.seq
        assert _slot_of(host, 4) == slot
    finally:
        host.stop()


@pytest.mark.parametrize("routed", [True, False])
def test_the_target_layers_next_request_judges_a_copied_row_used_or_wasted(tmp_path, routed):
    s, page, host, sim, pf, dst = _setup(tmp_path)
    try:
        _load(sim, host, [2, 3])
        pf.post(ROW, 3, 4)
        assert host.pump() == 3
        host.copy_engine_release(-1)
        assert host.copy_engine_idle(5.0)
        # The target layer's own request: a touch-only record (no lanes) carrying its routes as protect.
        sim.post(ROW, [], armed=False, protect=[3, 2] if routed else [2])
        assert host.pump() == 1
        c = host.counters()
        assert (c["prefetch_used"], c["prefetch_wasted"]) == ((1, 0) if routed else (0, 1)), c
        sim.post(ROW, [], armed=False, protect=[3])
        assert host.pump() == 1
        c2 = host.counters()
        assert c2["prefetch_used"] + c2["prefetch_wasted"] == 1, "one copied row was judged twice"
    finally:
        host.stop()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
