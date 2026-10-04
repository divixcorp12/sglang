"""The lane count each request carries, so lanes per layer can be read from a trace (CPU)."""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, stage_fields, stage_trace_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False, wire=wire_layout(8))
    host = attached_host(s, page, k=2)
    host.enable_trace()
    yield host, ChainSim(host, page, s.slabs)
    host.stop()


def _serve(host, sim, row, lanes):
    req = sim.post(row, lanes)
    assert host.pump() == 1
    assert sim.wait_served(req, timeout_s=1.0)


def test_a_request_carries_its_lane_count_not_the_number_of_rows_it_read(tier):
    host, sim = tier
    _serve(host, sim, 1, [2])
    _serve(host, sim, 1, [3])
    host.drain_trace()
    _serve(host, sim, 1, [2, 3, 4])  # three lanes, two of them RAM hits: one row read
    (record,) = host.drain_trace()
    assert record["lanes"] == 3 and record["rows_asked"] == 1


def test_an_all_ram_hit_request_still_says_how_many_lanes_it_had(tier):
    """The case lanes exist for: a layer whose lanes were all RAM hits reads nothing, so rows_asked says 0 and only
    `lanes` says the layer had work at all."""
    host, sim = tier
    _serve(host, sim, 1, [2])  # make experts 2 and 3 resident
    _serve(host, sim, 1, [3])
    _serve(host, sim, 1, [2, 3])
    _, _, hit_only = host.drain_trace()
    assert hit_only["status"] == "no_read" and hit_only["rows_asked"] == 0 and hit_only["lanes"] == 2


def test_a_16_lane_trace_decodes_every_row_past_the_8_lane_limit(tmp_path):
    """The host's stage record has ``stage_trace_rows(16)`` = 32 per-row slots and its words follow the lane count: a
    decoder fixed at 8 lanes would read the 16-lane words as 8-lane ones (wrong field offsets, and only 8 of the 14 rows
    a request asked for)."""
    s = ram_miss_setup(tmp_path, capacity=32, experts=24)
    wire = lease.wire_layout(16)
    page = new_page(pin=False, wire=wire)
    host = attached_host(s, page, k=16, lanes=16)
    host.enable_trace()
    sim = ChainSim(host, page, s.slabs)
    try:
        req = sim.post(1, list(range(14)))
        assert host.pump() == 1 and sim.wait_served(req, timeout_s=5.0)
        (record,) = host.drain_trace()
    finally:
        host.stop()
    assert len(stage_fields(16)) > len(stage_fields(8)) and stage_trace_rows(16) == 32
    assert record["lanes"] == 14 and record["rows_asked"] == 14 and record["rows_untraced"] == 0
    assert [r["row"] for r in record["row_pack"]] == list(range(14))
    assert record["status"] == "served" and record["done"] > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
