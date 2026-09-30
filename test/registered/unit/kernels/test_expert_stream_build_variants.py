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


def test_a_production_host_reports_only_the_core_counters(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="prod")
    try:
        assert tuple(host.counters()) == ops.CORE_COUNTERS
    finally:
        host.stop()


def test_the_instrumented_host_still_reports_every_counter(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="instr")
    try:
        assert tuple(host.counters()) == tuple(ops.COUNTERS) and set(ops.CORE_COUNTERS) < set(ops.COUNTERS)
    finally:
        host.stop()


@pytest.mark.parametrize("variant", ["instr", "instr_tsan"])
def test_every_instrumented_variant_reports_every_counter(variant):
    # The filter is "not prod", not "== instr": the TSan build is instrumented too (final review, Minor 2). A stub
    # module stands in for the TSan .so, which needs the compiler's TSan runtime preloaded.
    class Module:
        @staticmethod
        def expert_stream_counters(handle, out):
            out.copy_(torch.arange(out.numel(), dtype=torch.int64))

    host = ExpertStreamHost.__new__(ExpertStreamHost)
    host.variant, host.handle, host._module = variant, 0, Module()
    assert host.counters() == {name: i for i, name in enumerate(ops.COUNTERS)}
    host.variant = "prod"
    assert tuple(host.counters()) == ops.CORE_COUNTERS


def test_the_python_core_counters_are_the_hosts():
    mask = int(ops._host_module("exl3", "prod").expert_stream_core_counter_mask())
    assert {name for i, name in enumerate(ops.COUNTERS) if mask >> i & 1} == set(ops.CORE_COUNTERS)



# Plan Task 10 (Review Focus 5): a test-only or trace call on the production module raises, naming the instrumented
# build; the build was chosen once, at ExpertStreamHost construction.
TEST_ONLY_CALLS = {
    "inject": lambda h: h.inject(fail_reads=True),
    "inject_fault": lambda h: h.inject_fault(part=0, part_error=5),
    "inject_lease": lambda h: h.inject_lease(0, 0, 1),
    "trace": lambda h: h.enable_trace(16),
}


@pytest.mark.parametrize("name", sorted(TEST_ONLY_CALLS))
def test_test_only_calls_refuse_on_prod(name, tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="prod")
    try:
        with pytest.raises(RuntimeError, match="instrumented host build"):
            TEST_ONLY_CALLS[name](host)
    finally:
        host.stop()


def test_a_faulted_read_refuses_on_prod(tmp_path):
    s = ram_miss_setup(tmp_path)
    with pytest.raises(RuntimeError, match="instrumented host build"):
        ops.read_rows_with_fault(s.tables, 0, [1], [0], [], [], variant="prod", part=0, part_error=5)


# The Python wrappers refuse before building a tensor; these call the production module's C++ exports directly, so
# the C++ refusal is pinned too (a wrapper that forgot to refuse would otherwise reach a silent C++ fallback).
RAW_EXPORTS = {
    "inject": lambda m, h: m.expert_stream_inject(h, 0, 1, 0),
    "inject_fault": lambda m, h: m.expert_stream_inject_fault(h, ops._fault_tensor(part=0, part_error=5)),
    "inject_lease": lambda m, h: m.expert_stream_inject_lease(h, 0, 0, 1),
    "copy_engine_fail": lambda m, h: m.expert_stream_copy_engine_fail(h, 1, 0),
    "copy_engine_ballast": lambda m, h: m.expert_stream_copy_engine_ballast(h, 0, 0, 0),
    "trace_clock_reads": lambda m, h: m.expert_stream_trace_clock_reads(),
}


def test_every_test_only_export_is_listed_and_exported_by_both_builds():
    assert sorted(RAW_EXPORTS) == sorted(set(ops.TEST_ONLY_EXPORTS) - {"read_rows_faulted", "read_rows_sqes"})
    for variant in ops.VARIANTS:
        module = ops._host_module("exl3", variant)
        for name in ops.TEST_ONLY_EXPORTS:
            assert hasattr(module, f"expert_stream_{name}"), (variant, name)


@pytest.mark.parametrize("name", sorted(RAW_EXPORTS))
def test_the_prod_module_refuses_each_test_only_export_itself(name, tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="prod")
    try:
        with pytest.raises(RuntimeError, match=f"{name} is test-only: it exists in the instrumented host build"):
            RAW_EXPORTS[name](host._module, host.handle)
    finally:
        host.stop()


@pytest.mark.parametrize("helper", ("read_rows_with_fault", "read_rows_sqes"))
def test_the_module_level_test_only_helpers_refuse_on_prod(helper, tmp_path):
    s = ram_miss_setup(tmp_path)
    calls = {
        "read_rows_with_fault": lambda: ops.read_rows_with_fault(s.tables, 0, [1], [0], [], [], variant="prod"),
        "read_rows_sqes": lambda: ops.read_rows_sqes(s.tables, 0, [1], [0], variant="prod"),
    }
    export = {"read_rows_with_fault": "read_rows_faulted"}.get(helper, helper)  # the name the error carries
    with pytest.raises(RuntimeError, match=f"{export} is test-only"):
        calls[helper]()


def test_a_traced_read_on_prod_reads_exact_bytes_with_an_empty_record_and_refuses_a_fault(tmp_path):
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
    from sglang.test.dsv41_ram_miss_fixtures import same_bytes

    s = ram_miss_setup(tmp_path)
    result, record = ops.read_rows_traced(s.tables, 0, [1, 2], [0, 1], variant="prod", piece_stream=True)
    assert result == 1 and record["ok"] == 1
    assert record["rows_asked"] == 0 and record["bytes"] == 0, "production has no trace: the stages stay 0"
    oracle = s.reference(s.tables.layer_ids[0], [1, 2])
    assert all(same_bytes(s.slabs[0][n][slot], oracle[n][i]) for n in EXL3_STREAMED_NAMES for i, slot in enumerate([0, 1]))
    with pytest.raises(RuntimeError, match="a fault on read_rows_traced is test-only"):
        ops.read_rows_traced(s.tables, 0, [1], [0], variant="prod", part=0, part_error=5)
    _, instr = ops.read_rows_traced(s.tables, 0, [1, 2], [0, 1], variant="instr")
    assert instr["rows_asked"] == 2 and instr["bytes"] > 0, "the instrumented build still fills the record"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
