"""Leases across the 32-bit sequence wrap (CPU); LEASE_PROTOCOL.md 11 and items 7(b) and 7(d) of section 18.2.

The generation the service keys a lease by is the device's own 56-bit word (epoch << 32 | seq), echoed from the lane
request, so an acknowledgement after the wrap must still retire the lease. Written from the requirement text before
the service code; each test names the mutation it must fail under.
"""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import DEMAND_RECORDS, WORDS, Exl3RamMissHost, new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

BELOW_WRAP = 6


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(90, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def world(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=3)
    page = new_page(pin=False)
    for name in ("demand_head", "demand_done"):
        page[WORDS[name] : WORDS[name] + 4].view(torch.int32)[0] = -BELOW_WRAP  # 2**32 - 6
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_lease_mode()
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _leases(host, row):
    return [info[2] for info in host.slot_info(row)]


def test_leases_retire_across_the_wrap_because_the_service_echoes_the_devices_generation(world):
    """Mutation: the service keys leases by the 32-bit sequence alone, or by an epoch it counts itself: an
    acknowledgement carrying the device's epoch after the wrap then retires nothing and the lease is stuck."""
    s, page, host, sim = world
    epochs = []
    for i in range(3 * BELOW_WRAP):
        req = sim.post(i % 2, [i % 6])
        assert host.pump() == 1
        waited = sim.wait(req)
        assert waited.status == 1 and waited.go == 1, f"request {i}"
        epochs.append(req.gen >> 32)
        sim.ack(req, waited)
        sim.deliver()
        host.pump()
    assert epochs[0] == 0 and epochs[-1] == 1, "the run did cross the wrap"
    assert _leases(host, 0) == [0, 0, 0] and _leases(host, 1) == [0, 0, 0]
    assert host.counters()["leases_acked"] == 3 * BELOW_WRAP


def test_an_armed_request_after_a_lap_that_crosses_the_wrap_is_still_served_and_leased(world):
    """Mutation: the lap resume loses the request, or the service's own epoch falls behind the device's when it skips
    across the wrap (the model's counterexample). Unarmed records pile up past the ring while the service lags."""
    s, page, host, sim = world
    for _ in range(DEMAND_RECORDS + BELOW_WRAP):
        sim.post(0, [], armed=False)  # nothing waits on these: the device does not stop, the service lags
    armed = sim.post(0, [3])
    assert armed.gen >> 32 == 1, "the armed request is past the wrap"
    while host.pump():
        pass
    assert host.counters()["overruns"] > 0, "precondition: the service lapped"
    waited = sim.wait(armed)
    assert waited.status == 1 and waited.go == 1 and _leases(host, 0).count(1) == 1
    sim.ack(armed, waited)
    sim.deliver()
    host.pump()
    assert _leases(host, 0) == [0, 0, 0]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
