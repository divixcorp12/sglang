"""The production host module carries no trace or fault machinery (plan 2026-09-29-hotpath-zero-overhead Task 10):
its symbol table has none of their types, and the instrumented module's does (so the check can fail).

The optimizer inlines most of the machinery into its callers, so in the instrumented module only some of the names
survive as symbols of their own (FaultyReader and traced_clock_reads, with the compilers used so far). The check that
it can fail is therefore anchored on what is always out of line there: every RowReader/RamTier method it emits is
mangled with its FaultyReader reader and its InstrBuild policy. `InstrBuild` in the production module would mean
some instrumented instantiation leaked into it."""

import shutil
import subprocess

import pytest

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

FORBIDDEN = ("StageRing", "ReadFault", "FaultyReader", "fault_from", "traced_clock_reads", "SqeRecord",
             "apply_pending_fault", "TraceState", "InstrBuild", "FaultState", "TierFaults", "set_fault")
ALWAYS_IN_INSTR = ("FaultyReader", "InstrBuild")


def _loaded_path(module_name: str) -> str:
    # load_jit names the file sgl_kernel_jit_<module name>.so.
    for line in open("/proc/self/maps"):
        path = line.split()[-1]
        if path.endswith((f"/{module_name}.so", f"/sgl_kernel_jit_{module_name}.so")):
            return path
    raise AssertionError(f"{module_name}.so is not mapped")


LANES = 8  # the default build; load_jit names the module _l<lanes>


def _symbols(variant: str) -> str:
    ops._host_module("exl3", variant, LANES).expert_stream_build_name()
    path = _loaded_path(f"expert_stream_host_exl3_{variant}_l{LANES}")
    return subprocess.run(["nm", "-C", path], capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(shutil.which("nm") is None, reason="binutils nm is not installed")
def test_prod_has_no_trace_or_fault_symbols_and_instr_has_them():
    prod, instr = _symbols("prod"), _symbols("instr")
    assert "RamTier" in prod, "the prod module's symbol table is stripped: this check cannot see anything"
    assert [name for name in FORBIDDEN if name in prod] == []
    assert [name for name in ALWAYS_IN_INSTR if name not in instr] == [], "the instrumented build lost its machinery"
    assert "ProdBuild" in prod and "ProdBuild" not in instr


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
