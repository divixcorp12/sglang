"""The lane count each request carries, so lanes per layer can be read from a trace (CPU)."""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    host.enable_trace()
    yield host, LeaseSim(host, page, s.slabs)
    host.stop()


def _serve(host, sim, row, lanes):
    req = sim.post(row, lanes)
    assert host.pump() == 1
    assert sim.wait(req, timeout_s=1.0).served
    sim.done(req)


def test_a_request_carries_its_lane_count_not_the_number_of_rows_it_read(tier):
    host, sim = tier
    _serve(host, sim, 1, [2] * 5)  # five lanes, all naming one missing row
    (record,) = host.drain_trace()
    assert record["lanes"] == 5 and record["rows_asked"] == 1


def test_an_all_ram_hit_request_still_says_how_many_lanes_it_had(tier):
    """The case lanes exist for: a layer whose lanes were all RAM hits reads nothing, so rows_asked says 0 and only
    `lanes` says the layer had work at all."""
    host, sim = tier
    _serve(host, sim, 1, [2])  # make expert 2 resident
    _serve(host, sim, 1, [2] * 4)
    _, hit_only = host.drain_trace()
    assert hit_only["status"] == "no_read" and hit_only["rows_asked"] == 0 and hit_only["lanes"] == 4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
