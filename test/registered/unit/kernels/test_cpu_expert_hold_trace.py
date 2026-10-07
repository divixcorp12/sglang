"""Keep-warm handoff attribution, bounded records, and production compile-out (Linux CPU)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux/OpenMP diagnostics")


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    work = tmp_path_factory.mktemp("hold-trace")
    root = Path(__file__).resolve().parents[4]
    source = work / "probe.cpp"
    source.write_text(r'''
#include "moe/expert_stream/host/cpu_experts/keep_warm.hpp"
#include <thread>
using namespace sglang::cpu_experts;
int main() {
  uint32_t a=0, b=0;
  for (int i=0; i<2; ++i) {
    std::thread post([&] { usleep(20000); __atomic_add_fetch(i ? &b : &a, 1u, __ATOMIC_RELEASE); });
    auto start=hold_trace::now();
    if (i) keep_warm_either<Isa::Scalar>(Isa::Scalar, {}, 4, &a, 1, &b, 0, start, start+1000000000);
    else keep_warm<Isa::Scalar>(Isa::Scalar, {}, 4, &a, 0, start, start+1000000000);
    post.join();
  }
}
''')
    production = work / "production.cpp"
    production.write_text(r'''
#include "moe/expert_stream/host/cpu_experts/hold_trace.hpp"
using namespace sglang::cpu_experts::hold_trace;
static_assert(!kOn);
int main() { uint32_t word=0; Capture<false> c(4,&word,0,nullptr,0,0,0);
 c.enter(0); c.ready(0); c.exit(0); c.finish(); }
''')
    result = []
    for name, src, flags in [("probe", source, ["-DSGLANG_CPU_EXPERT_HOLD_TRACE=1", "-fopenmp"]),
                             ("production", production, [])]:
        binary = work / name
        subprocess.run(["g++", "-std=c++20", "-O2", "-pthread", *flags,
                        "-I" + str(root / "python/sglang/kernels/jit/csrc"),
                        str(src), "-o", str(binary)], check=True, capture_output=True)
        result.append(binary)
    return result


def run(binaries, tmp_path, capacity="65536"):
    subprocess.run([str(binaries[0])], check=True,
                   env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                        "SGLANG_CPU_EXPERT_HOLD_TRACE_PREFIX": str(tmp_path / "hold"),
                        "SGLANG_CPU_EXPERT_HOLD_TRACE_CAPACITY": capacity})
    return [json.loads(line) for line in next(tmp_path.glob("hold.*.jsonl")).read_text().splitlines()]


def test_both_word_paths_record_workers_and_join(binaries, tmp_path):
    records = run(binaries, tmp_path)
    assert records[-1]["holds_seen"] == 2 and records[-1]["dropped_holds"] == 0
    assert len(records[1:-1]) == 8
    for hold in (1, 2):
        group = [r for r in records[1:-1] if r["hold"] == hold]
        assert {r["worker"] for r in group} == set(range(4))
        assert len({r["tid"] for r in group}) == 4
        assert all(r["begin"] <= r["enter"] <= r["ready"] <= r["exit"] <= r["end"] for r in group)
        assert all(r["observed_a"] == 1 for r in group)
        assert all(r["either"] == (hold == 2) for r in group)
        assert all(r["observed_b"] == (1 if hold == 2 else 0) for r in group)


def test_capacity_drops_whole_holds(binaries, tmp_path):
    records = run(binaries, tmp_path, capacity="6")
    assert len(records[1:-1]) == 4
    assert records[-1]["holds_seen"] == 2 and records[-1]["dropped_holds"] == 1


def test_disabled_capture_has_no_trace_symbols_or_files(binaries, tmp_path):
    subprocess.run([str(binaries[1])], check=True,
                   env={**os.environ, "SGLANG_CPU_EXPERT_HOLD_TRACE_PREFIX": str(tmp_path / "forbidden")})
    assert not list(tmp_path.iterdir())
    symbols = subprocess.check_output(["nm", "-u", str(binaries[1])], text=True)
    assert not any(s in symbols for s in ("clock_gettime", "fopen", "getenv", "syscall"))
