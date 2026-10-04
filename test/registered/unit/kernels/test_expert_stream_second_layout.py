"""A second row layout builds from a trait and two bindings files: the test-only TwoNameLayout (expert_stream_two_name/)
gets its own host module from HostExports and, with CUDA, its own device module."""

import dataclasses
from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, TransportBuild, new_page
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

TWO_NAME = Path(__file__).resolve().parent / "expert_stream_two_name"
# The instrumented build only (the tests' default variant, sglang.test.expert_stream_variant): the fixture's
# enable_trace needs it, and the layout is what this file tests, not the build.
TWO = TransportBuild(
    host_sources={"instr": str(TWO_NAME / "two_name_host.cpp")},
    device_source=str(TWO_NAME / "two_name_device.cuh"),
    device_layout="sglang::expert_stream::testing::TwoNameLayout",
)


@pytest.fixture
def two(monkeypatch):
    monkeypatch.setitem(transport.LAYOUTS, "two", TWO)


def test_the_second_layout_has_its_own_host_module(two):
    """Red when HostExports ignores its Layout (say, a body naming Exl3RowLayout again): the two-name module would
    report EXL3's six names. Red too when a cached loader is keyed by call form again: _host_module() and
    _host_module("exl3") would be two loads of one module."""
    assert transport.host_layout("two") == (("a", "b"), 0b10)
    assert transport.host_layout("exl3") == (tuple(EXL3_STREAMED_NAMES), 0b110110)
    # Read from each module itself, not through host_layout's cache: two keys naming one .so would agree here.
    two_module, exl3_module = transport._host_module("two"), transport._host_module("exl3")
    assert str(two_module.expert_stream_layout_names()) == "a\nb"
    assert str(exl3_module.expert_stream_layout_names()) == "\n".join(EXL3_STREAMED_NAMES)
    # One module per layout whatever the call form (cache_once keys f(), f(x) and f(layout=x) apart).
    assert transport._host_module() is transport._host_module("exl3") is transport._host_module(layout="exl3")


def test_a_two_name_host_serves_a_demand_through_its_own_module(two, tmp_path, monkeypatch):
    """The EXL3 fixture narrowed to its first two names is a two-name table: the two-name host reads a row into
    both names' slabs. Red when ExpertStreamHost drops its layout, e.g. enable_trace calling a bare _stage_words(),
    which loads the EXL3 module."""
    s = ram_miss_setup(tmp_path)
    t = s.tables
    keep = t.segments[:, 0] < 2
    # A row image is read whole, and the reader refuses a read of bytes no segment names: the one part (one root) is
    # cut to the kept names' bytes, [0, the end of the last kept segment), which the image lays out first.
    kept = t.segments[keep]
    assert t.parts == 1 and int(kept[:, 2].min()) == 0
    extents = t.extents.clone()
    extents[..., 2] = int((kept[:, 2] + kept[:, 3]).max())
    tables = dataclasses.replace(
        t, slabs=t.slabs[:, :2].contiguous(), row_bytes=t.row_bytes[:2].contiguous(),
        segments=kept.contiguous(), extents=extents,
    )
    loaded = []
    real = transport._host_module
    monkeypatch.setattr(
        transport, "_host_module",
        lambda layout="exl3", variant=None, lanes=8: loaded.append(layout) or real(layout, variant, lanes),
    )
    page = new_page(pin=False)
    slot_map = torch.full(tuple(t.starts.shape), -1, dtype=torch.int32)
    host = ExpertStreamHost(tables, page=page, slot_map=slot_map, layout="two")
    try:
        assert (host.layout_names, host.small_mask) == (("a", "b"), 0b10)
        host.reserve_staging(1)
        host.enable_trace()
        sim = ChainSim(host, page, None)
        req = sim.post(1, [2])
        assert host.pump() == 1
        assert sim.wait_served(req, timeout_s=1.0)
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
