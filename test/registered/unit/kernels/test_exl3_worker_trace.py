"""Worker attribution, complete-forward overflow, and compile-out checks (Linux CPU only)."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux/OpenMP diagnostics")


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    root = Path(__file__).resolve().parents[4]
    work = tmp_path_factory.mktemp("worker-trace")
    source = work / "probe.cpp"
    source.write_text(r'''
#include "worker_trace.hpp"
#include "moe/expert_stream/host/cpu_experts/team.hpp"
#include <omp.h>
using namespace sglang::exl3_cpu::worker_trace;
int main() {
  for (int f=0; f<2; ++f) {
    Capture<ON> capture(4, 6, 3);
    capture.team_start();
    sglang::cpu_experts::run_team(4, [&](int w, int) {
      for (int p : {0,1,2,3,5}) {
        capture.begin(w,p);
        if (p==0 && w==3) usleep(20000);
        if (p==1 && w==2) {
          auto start=clock_ns(CLOCK_THREAD_CPUTIME_ID);
          while (clock_ns(CLOCK_THREAD_CPUTIME_ID)-start<3000000) asm volatile("" ::: "memory");
        }
        capture.add_work(w,p,w+1);
        capture.work_end(w,p);
        #pragma omp barrier
        capture.end(w,p);
      }
      capture.worker_end(w);
    }, capture);
    capture.finish();
  }
}
''')
    production = work / "production.cpp"
    production.write_text(r'''
#include "worker_trace.hpp"
#include "moe/expert_stream/host/cpu_experts/team.hpp"
using namespace sglang::exl3_cpu::worker_trace;
static_assert(std::is_empty_v<Capture<false>>);
int main() { Capture<false> c(4,6,3); c.team_start(); c.worker_start(0);
 c.begin(0,0); c.add_work(0,0,3); c.work_end(0,0); c.end(0,0); c.worker_end(0); c.finish();
 sglang::cpu_experts::run_team(1, [](int, int) {}); }
''')
    outputs = []
    for name, src, flags in [("probe", source, ["-DON=true"]), ("production", production, [])]:
        binary = work / name
        subprocess.run(["g++", "-std=c++20", "-O2", "-fopenmp", *flags, "-I",
                        str(root / "python/sglang/kernels/jit/csrc/exl3/optimized"),
                        "-I", str(root / "python/sglang/kernels/jit/csrc"),
                        str(src), "-o", str(binary)], check=True, capture_output=True)
        outputs.append(binary)
    return outputs


def run(probes, tmp_path, **options):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
           "SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX": str(tmp_path / "phases"),
           "SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US": "0", **options}
    subprocess.run([str(probes[0])], check=True, env=env)
    return [json.loads(line) for line in next(tmp_path.glob("phases.*.jsonl")).open()]


def test_workers_distinguish_sleep_from_compute(probes, tmp_path):
    records = run(probes, tmp_path)
    assert len(records[1:-1]) == 40 and records[-1]["dropped_forwards"] == 0
    assert records[-1]["forwards_retained"] == 2
    first = [r for r in records[1:-1] if r["forward"] == 1]
    sleeper = next(r for r in first if r["worker"] == 3 and r["phase"] == 0)
    assert sleeper["work_end"] - sleeper["begin"] >= 19_000_000
    assert sleeper["cpu_work_end"] - sleeper["cpu_begin"] < 10_000_000
    assert sleeper["nvcsw"] >= 1
    other = next(r for r in first if r["worker"] == 0 and r["phase"] == 0)
    assert other["end"] - other["work_end"] >= 10_000_000
    busy = next(r for r in first if r["worker"] == 2 and r["phase"] == 1)
    assert busy["cpu_work_end"] - busy["cpu_begin"] >= 3_000_000
    assert {r["phase"] for r in first} == {0, 1, 2, 3, 5}
    assert len({r["tid"] for r in first}) == 4
    assert all(r["units"] == r["worker"] + 1 for r in first)
    assert all(r["begin"] <= r["work_end"] <= r["end"] for r in first)
    assert all(r["team_begin"] <= r["enter"] <= r["ready"] <= r["begin"] for r in first)


def test_capacity_admits_only_complete_forwards(probes, tmp_path):
    records = run(probes, tmp_path, SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY="25")
    assert len(records[1:-1]) == 20
    assert records[-1]["forwards_retained"] == records[-1]["dropped_forwards"] == 1


def test_duration_filter_retains_no_fast_forwards(probes, tmp_path):
    records = run(probes, tmp_path, SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US="1000000000")
    assert len(records) == 2 and records[-1]["forwards_seen"] == 2


def test_production_has_no_timing_resource_or_file_symbols(probes, tmp_path):
    subprocess.run([str(probes[1])], check=True, cwd=tmp_path,
                   env={**os.environ, "SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX": str(tmp_path / "forbidden")})
    assert not list(tmp_path.iterdir())
    symbols = subprocess.check_output(["nm", "-u", str(probes[1])], text=True)
    assert not any(s in symbols for s in ("clock_gettime", "getrusage", "fopen", "getenv", "syscall"))


def test_build_flag_requires_job_attribution(monkeypatch):
    from sglang.srt.layers.quantization.exl3.ext import worker_trace_defines
    monkeypatch.setenv("SGLANG_DSV41_CPU_EXPERTS", "1")
    monkeypatch.delenv("SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX", raising=False)
    assert worker_trace_defines() == []
    monkeypatch.setenv("SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX", "/diagnostic/phases")
    monkeypatch.delenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", raising=False)
    with pytest.raises(ValueError, match="attribution"):
        worker_trace_defines()
    monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", "/diagnostic/events")
    assert worker_trace_defines() == ["-DEXL3_MOE_CPU_WORKER_TRACE=1"]
