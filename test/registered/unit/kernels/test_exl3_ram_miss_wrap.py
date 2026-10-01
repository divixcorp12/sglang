"""The service walks the 32-bit request sequence through its wrap like the device does (CPU).

The device skips sequence 0 on wrap (post kernel and ChainSim.post: `if (seq == 0) seq = 1`), so the service must
too. pump_demand advances with `skip_zero(next + 1u)`; without the skip, after 0xFFFFFFFF it would spend one
iteration on a phantom seq 0: the handled seq steps back to 0 and the ring slot `(0 - 1) % records` is read as if it
held a request. The invariant is that the record seq the service handles never takes the value 0. A lapped ring can
resume at 0 as well.

That phantom looks different on the two page states, so both are covered:
  used page   the slot holds an older sequence, the seqlock read fails, one overrun is counted and the handled seq
              is 0.
  fresh page  the slot holds zeros, the read SUCCEEDS as an empty touch request, no overrun is counted, and the
              phantom is served.
A test that only looked for the overrun counter would pass on a fresh page.
"""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import (
    DEMAND_RECORDS,
    DEMAND_RING,
    RECORD_BYTES,
    WORDS,
    ExpertStreamHost,
    new_page,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

STALE_SEQ = 0xFFFFFFF0  # what the slot before the wrap holds in a run that has been going a while


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _signed(value):
    return value - 2**32 if value >= 2**31 else value


def _set_word(page, offset, value):
    page[offset : offset + 4].view(torch.int32)[0] = _signed(value)


def _word(page, offset):
    return int(page[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF


def _host(tmp_path, seed, used):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    _set_word(page, WORDS["demand_head"], seed)
    if used:
        # The slot a phantom seq 0 would read: (0 - 1) % records, as the service computes it.
        _set_word(page, DEMAND_RING + (DEMAND_RECORDS - 1) * RECORD_BYTES, STALE_SEQ)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), variant="instr")
    host.reserve_staging(1)
    return page, host, ChainSim(host, page, s.slabs)


def _drive(tmp_path, *, seed, used):
    """Seed demand_head at `seed`, post four demands through the wrap, pump exactly four times.

    Returns the posted sequences, the handled seq after every pump, the result of one more pump and the counters."""
    page, host, sim = _host(tmp_path, seed, used)
    try:
        # Each a miss: the device posts the next one only after this chain was served, so post and pump alternate.
        posted, done = [], []
        for expert in (0, 1, 2, 3):
            posted.append(sim.post(1, [expert]).seq)
            assert host.pump() == 1
            done.append(host.handled_through())
        idle = host.pump()
        return posted, done, idle, host.counters()
    finally:
        host.stop()


@pytest.mark.parametrize("used", [True, False], ids=["used_page", "fresh_page"])
def test_the_service_skips_sequence_zero_at_the_wrap(tmp_path, used):
    posted, done, idle, counters = _drive(tmp_path, seed=0xFFFFFFFD, used=used)
    assert posted == [0xFFFFFFFE, 0xFFFFFFFF, 1, 2]  # the device side skips 0
    # One pump per posted record, each leaving the handled seq at that record's sequence: no phantom iteration, so it
    # never steps back to 0 and nothing is left over for a fifth pump.
    assert done == posted
    assert idle == 0
    assert counters["served"] == 4
    assert counters["touch_only"] == 0  # a fresh page reads the phantom as an empty touch record
    assert counters["overruns"] == 0  # a used page fails the phantom's seqlock read


def test_the_control_a_service_opened_at_the_wrap_already_skips_zero(tmp_path):
    """open() derives its first sequence from demand_head with the skip, so a page whose head is already 0xFFFFFFFF
    is served correctly. It shows a failure above is the per-request advance, not the setup.

    If the test above fails, read this one first: when this control fails too, the setup is wrong (seeding, slot
    arithmetic, pump return codes), not the service."""
    posted, done, idle, counters = _drive(tmp_path, seed=0xFFFFFFFF, used=True)
    assert posted == [1, 2, 3, 4]
    assert done == posted and idle == 0
    assert counters["overruns"] == 0


def test_a_lap_that_would_resume_at_sequence_zero_skips_it(tmp_path):
    """Posting a full ring, starting two before the wrap, leaves head - next == records, and a lap resumes at
    head - (records - 2), which is 0 here. The service must resume at 1 instead: the handled seq never takes the value
    0 and the records that survived the lap are all served. Only records nothing waits for lap (a post with host work
    waits for its chain), so the ring is filled with touch records."""
    page, host, sim = _host(tmp_path, 0xFFFFFFFD, used=False)
    try:
        posted = [sim.post(1, [], protect=[i % 6]).seq for i in range(DEMAND_RECORDS)]
        assert posted[:3] == [0xFFFFFFFE, 0xFFFFFFFF, 1] and posted[-1] == DEMAND_RECORDS - 2
        done = []
        for _ in range(DEMAND_RECORDS + 2):
            if host.pump() == 0:
                break
            done.append(host.handled_through())
        assert 0 not in done, done
        assert done[-1] == posted[-1]
        assert done == sorted(done)  # the survivors, in order, none twice
        assert host.counters()["overruns"] > 0  # the lap happened
    finally:
        host.stop()


@pytest.mark.parametrize("used", [True, False], ids=["used_page", "fresh_page"])
def test_the_service_thread_serves_a_real_waiter_for_every_sequence_through_the_wrap(tmp_path, used):
    """The pump-driven cases above are deterministic and see the phantom's signature; this one runs the real service
    thread with a waiter on each sequence, as S waits, so it also shows that nothing wedges: every request is served
    and its pieces are published for its generation."""
    page, host, sim = _host(tmp_path, 0xFFFFFFFD, used)
    host.start_thread(fatal_wait_s=5.0)
    try:
        requests = []
        for expert in (0, 1, 2, 3):
            req = sim.post(1, [expert])
            assert sim.wait_served(req, timeout_s=10)
            requests.append(req.seq)
        assert requests == [0xFFFFFFFE, 0xFFFFFFFF, 1, 2]
        # Every served sequence was posted, so nothing was spent on a phantom one.
        counters = host.counters()
        assert counters["served"] == 4 and counters["touch_only"] == 0 and counters["overruns"] == 0
    finally:
        host.stop()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
