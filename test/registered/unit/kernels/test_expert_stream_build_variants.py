"""The host transport ships two builds (plan 2026-09-29-hotpath-zero-overhead D2/D3): production, with no metrics,
trace or fault state on the request path, and instrumented. A service loads the instrumented one exactly when this
process writes a stream trace or injects a RAM-miss fault."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.setattr(ops, "_DEFAULT_VARIANT", None)


def test_the_production_build_is_the_default(no_override):
    with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(""), envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override(""):
        assert ops.host_variant() == "prod"


def test_a_stream_trace_selects_the_instrumented_build(no_override, tmp_path):
    with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(str(tmp_path / "t")):
        assert ops.host_variant() == "instr"


def test_a_ram_miss_fault_selects_the_instrumented_build(no_override):
    with envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override("5:1"):
        assert ops.host_variant() == "instr"


@pytest.mark.parametrize("variant", ops.VARIANTS)
def test_each_module_names_its_build_and_a_host_keeps_the_one_it_loaded(variant, tmp_path):
    assert str(ops._host_module("exl3", variant).expert_stream_build_name()) == variant
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant=variant)
    try:
        assert host.variant == variant
    finally:
        host.stop()


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="variant"):
        ops._host_module("exl3", "fast")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
