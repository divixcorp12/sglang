"""The planned lane count each request carries, so lanes per layer can be read from a trace (CPU)."""

import faulthandler
import re
from pathlib import Path

import pytest
import torch

import sglang.kernels.ops.moe.exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

CUH = Path(ops.__file__).resolve().parents[2] / "jit" / "csrc" / "moe" / "exl3_ram_miss.cuh"


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.enable_trace()
    yield s, page, host
    host.stop()


def _serve(page, host, row, need, protect, **post):
    seq = sim_post(page, row, need=need, protect=protect, **post)
    assert host.pump() == 1
    assert sim_wait(page, seq, timeout_s=1.0) == 1
    return seq


def test_a_request_carries_the_planned_lane_count_not_the_number_of_rows_it_read(tier):
    s, page, host = tier
    _serve(page, host, 1, need=[2], protect=[2], lanes=5)  # five planned lanes, one of them a miss
    (record,) = host.drain_trace()
    assert record["lanes"] == 5 and record["rows_asked"] == 1


def test_an_all_ram_hit_request_still_says_how_many_lanes_it_planned(tier):
    """The case lanes exist for: a layer whose lanes were all RAM hits reads nothing, so rows_asked says 0
    and only `lanes` says the layer had work at all."""
    s, page, host = tier
    _serve(page, host, 1, need=[2], protect=[2])  # make expert 2 resident
    _serve(page, host, 1, need=[], protect=[2], lanes=4)
    _, hit_only = host.drain_trace()
    assert hit_only["status"] == "no_read" and hit_only["rows_asked"] == 0 and hit_only["lanes"] == 4


def test_lanes_are_not_clamped_to_the_ids_a_record_can_carry(tier):
    s, page, host = tier
    _serve(page, host, 1, need=[0, 1, 2], protect=[0, 1, 2], lanes=10)  # a record holds at most 8 ids
    (record,) = host.drain_trace()
    assert record["lanes"] == 10 and record["rows_asked"] == 3


def test_a_touch_only_record_carries_lanes(tier):
    s, page, host = tier
    _serve(page, host, 1, need=[2], protect=[2])
    seq = sim_post(page, 1, need=[], protect=[2], armed=False, lanes=3)
    assert host.pump() == 1 and seq
    _, touch = host.drain_trace()
    assert touch["kind"] == "touch" and touch["lanes"] == 3


def test_an_advisory_carries_the_rows_it_asks_for(tier):
    s, page, host = tier
    sim_post(page, 1, need=[4, 5], protect=[4, 5], advisory=True, lanes=2)
    assert host.pump() == 2
    (record,) = host.drain_trace()
    assert record["kind"] == "advisory" and record["lanes"] == 2


def test_without_an_explicit_count_the_simulated_device_posts_one_lane_per_need_id(tier):
    s, page, host = tier
    _serve(page, host, 1, need=[2, 5], protect=[2, 5])
    (record,) = host.drain_trace()
    assert record["lanes"] == 2


def test_the_device_writes_the_lane_count_into_the_record_it_posts():
    """The post kernel cannot run here; check its source stores the count the host reads at kRecLanes."""
    source = CUH.read_text()
    assert "words[kRecLanes / 4] = lanes;" in source
    assert re.search(r"lanes = static_cast<uint32_t>\(max\(count\[0\], 0\)\)", source)
    assert source.count("write_record(") == 3  # the definition and the demand and advisory posts
    assert "seq, 1u, ahead_count);" in source and "armed ? 1u : 0u, lanes);" in source
