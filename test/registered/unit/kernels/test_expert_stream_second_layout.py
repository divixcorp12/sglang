"""A second row layout builds from a trait and two bindings files: the test-only TwoNameLayout (expert_stream_two_name/)
gets its own host module from HostExports and, with CUDA, its own device module."""

import dataclasses
from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, TransportBuild, new_page, sim_post, sim_wait
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

TWO_NAME = Path(__file__).resolve().parent / "expert_stream_two_name"
TWO = TransportBuild(
    host_source=str(TWO_NAME / "two_name_host.cpp"),
    device_source=str(TWO_NAME / "two_name_device.cuh"),
    device_layout="sglang::expert_stream::testing::TwoNameLayout",
)


@pytest.fixture
def two(monkeypatch):
    monkeypatch.setitem(transport.LAYOUTS, "two", TWO)


def test_the_second_layout_has_its_own_host_module(two):
    """Red when HostExports ignores its Layout (say, a body naming Exl3RowLayout again): the two-name module would
    report EXL3's six names, or share EXL3's module."""
    assert transport.host_layout("two") == (("a", "b"), 0b10)
    assert transport.host_layout("exl3") == (tuple(EXL3_STREAMED_NAMES), transport.host_layout("exl3")[1])
    assert transport._host_module("two") is not transport._host_module("exl3")


def test_a_two_name_host_serves_a_demand_through_its_own_module(two, tmp_path, monkeypatch):
    """The EXL3 fixture narrowed to its first two names is a two-name table: the two-name host reads a row into
    both names' slabs. Red when ExpertStreamHost drops its layout, e.g. enable_trace calling a bare _stage_words(),
    which loads the EXL3 module."""
    s = ram_miss_setup(tmp_path)
    t = s.tables
    keep = t.segments[:, 0] < 2
    tables = dataclasses.replace(
        t, slabs=t.slabs[:, :2].contiguous(), row_bytes=t.row_bytes[:2].contiguous(), segments=t.segments[keep].contiguous()
    )
    loaded = []
    real = transport._host_module
    monkeypatch.setattr(transport, "_host_module", lambda layout="exl3": loaded.append(layout) or real(layout))
    page = new_page(pin=False)
    slot_map = torch.full(tuple(t.starts.shape), -1, dtype=torch.int32)
    host = ExpertStreamHost(tables, page=page, slot_map=slot_map, direct=False, layout="two")
    try:
        assert (host.layout_names, host.small_mask) == (("a", "b"), 0b10)
        host.enable_trace()
        seq = sim_post(page, 1, need=[2], protect=[2], layout="two")
        assert host.pump() == 1
        assert sim_wait(page, seq, timeout_s=1.0, layout="two") == 1
        slot = int(slot_map[1, 2])
        reference = s.reference(1, [2])
        for name in EXL3_STREAMED_NAMES[:2]:
            assert same_bytes(s.slabs[1][name][slot], reference[name][0]), name
    finally:
        host.stop()
    assert loaded and set(loaded) == {"two"}, loaded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the device module needs CUDA")
def test_the_second_layout_builds_its_device_module(two):
    """Red when the row-copy kernels name EXL3 or need more of a layout than its trait."""
    assert transport._device_module("two") is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
