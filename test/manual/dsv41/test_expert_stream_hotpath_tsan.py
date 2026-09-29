"""ThreadSanitizer over the concurrent parties of the single-owner tier (plan 2026-09-29-hotpath-zero-overhead
Task 16). Manual: needs a compiler with a TSan runtime and a few minutes. Each child preloads that runtime, so the host
module's accesses are checked; Python and torch are not instrumented and are suppressed (tsan.supp).

Two children, each its own interpreter:

- the stress child runs ``run_stress`` (Task 4) with its fill phase on, so the device, the copy thread, unpaused Python
  calls, the pauser, and a prefill fill thread whose epilogue the owner runs (Task 15) all meet under TSan;
- the fills child runs the prefill-fill and ownership suites (preflight F16: the stress alone never started a fill),
  with every instrumented host they build loaded from the TSan module.

The compiler is ``$CXX`` (default ``c++``), the one ``load_jit`` builds with. GCC names its runtime through
``-print-file-name=libtsan.so``, which may be a linker script (``INPUT ( /usr/lib64/libtsan.so.2.0.0 )``) naming a
runtime package that is not installed; Clang's is ``libclang_rt.tsan.so`` in its ``-print-runtime-dir``."""

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SUPP = Path(__file__).with_name("tsan.supp")
REPO = Path(__file__).resolve().parents[3]
TSAN_OPTIONS = f"halt_on_error=1 report_signal_unsafe=0 second_deadlock_stack=1 suppressions={SUPP}"

# The runtime is already mapped once the interpreter starts; dropping LD_PRELOAD keeps it out of the compiler that
# load_jit runs for the TSan module's first build (cc1plus under a preloaded libtsan segfaults).
STRESS_CHILD = textwrap.dedent("""
    import os
    os.environ.pop("LD_PRELOAD", None)
    import sys
    from pathlib import Path
    from sglang.kernels.ops.moe import expert_stream_transport as ops
    ops._ALLOW_TSAN = True
    sys.path.insert(0, "test/registered/unit/kernels")
    from test_expert_stream_hotpath_stress import run_stress
    report = run_stress(Path(sys.argv[1]), variant="instr_tsan", seconds=20.0, seed=7, fills=True)
    assert report["stats"]["errors"] == [] and report["fatal"] == 0, report["stats"]
    assert report["stats"]["fills"] > 0 and report["stats"]["pauses"] > 0, report["stats"]
    print("TSAN-STRESS-OK", report["stats"]["armed"], "fills", report["stats"]["fills"])
""")

# The registered conftests pick the instrumented build ("instr"); here every instrumented host is the TSan module.
FILLS_CHILD = textwrap.dedent("""
    import os
    os.environ.pop("LD_PRELOAD", None)
    import sys
    import pytest
    from sglang.kernels.ops.moe import expert_stream_transport as ops
    ops._ALLOW_TSAN = True
    _load = ops._host_module_cached
    ops._host_module_cached = lambda layout, variant: (
        ops._host_module_tsan(layout) if variant == "instr" else _load(layout, variant))
    code = pytest.main(["-q", "-p", "no:randomly", "-p", "no:cacheprovider", "-x", *sys.argv[1:]])
    print("TSAN-FILLS-EXIT", int(code))
    sys.exit(int(code))
""")

# Timing tests, not ordering ones: under TSan's 5-15x slowdown their wall-clock bounds do not hold.
FILLS_TARGETS = [
    "test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py",
    "test/registered/unit/kernels/test_expert_stream_ownership.py",
    "--deselect",
    "test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py::test_fill_wait_returns_as_a_prefix_lands",
    "--deselect",
    "test/registered/unit/kernels/test_expert_stream_ownership.py::test_a_set_hot_burst_past_the_ring_is_applied_in_order",
]


def tsan_runtime() -> tuple[str | None, str]:
    """(the TSan runtime to preload, or None; why not)."""
    cxx = os.environ.get("CXX", "c++")
    if "clang" in os.path.basename(cxx):
        out = subprocess.run([cxx, "-print-runtime-dir"], capture_output=True, text=True).stdout.strip()
        path = os.path.join(out, "libclang_rt.tsan.so") if out else ""
    else:
        path = subprocess.run([cxx, "-print-file-name=libtsan.so"], capture_output=True, text=True).stdout.strip()
        if path and os.path.isfile(path) and os.path.getsize(path) < 4096:
            text = Path(path).read_bytes()
            if not text.startswith(b"\x7fELF"):  # a linker script names the real runtime
                match = re.search(r"INPUT\s*\(\s*(\S+)", text.decode(errors="replace"))
                path = match.group(1) if match else ""
    if not path or not os.path.exists(path):
        return None, f"{cxx}: no TSan runtime ({path or 'none reported'} does not exist)"
    return path, ""


def _run(child: str, args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    runtime, why = tsan_runtime()
    if runtime is None:
        pytest.skip(why)
    options = TSAN_OPTIONS
    if "clang_rt" in runtime and shutil.which("llvm-symbolizer") is None:
        # Without llvm-symbolizer Clang's runtime falls back to addr2line, whose reply parser trips a CHECK that then
        # deadlocks re-symbolizing its own failure (divix01, Clang 21): report module+offset frames instead, and
        # symbolize them offline (addr2line -f -C -e <module> <offset>).
        options += " symbolize=0"
    env = dict(os.environ, LD_PRELOAD=runtime, TSAN_OPTIONS=options, OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", child, *args], env=env, cwd=REPO, capture_output=True, text=True,
                          timeout=1800)
    (tmp_path / "child.stdout").write_text(proc.stdout)
    (tmp_path / "child.stderr").write_text(proc.stderr)
    return proc


def _no_race(proc):
    assert "WARNING: ThreadSanitizer" not in proc.stderr, proc.stderr[-12000:]


def test_the_single_owner_tier_is_race_free_under_tsan(tmp_path):
    (tmp_path / "s").mkdir()
    proc = _run(STRESS_CHILD, [str(tmp_path / "s")], tmp_path)
    _no_race(proc)
    assert proc.returncode == 0 and "TSAN-STRESS-OK" in proc.stdout, (proc.stdout[-4000:], proc.stderr[-8000:])


def test_prefill_fills_and_ownership_are_race_free_under_tsan(tmp_path):
    proc = _run(FILLS_CHILD, FILLS_TARGETS, tmp_path)
    _no_race(proc)
    assert proc.returncode == 0 and "TSAN-FILLS-EXIT 0" in proc.stdout, (proc.stdout[-6000:], proc.stderr[-8000:])
