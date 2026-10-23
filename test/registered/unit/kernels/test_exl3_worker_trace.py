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
#include <vector>
using namespace sglang::exl3_cpu::worker_trace;
// The kernel reads no environment: install the trace options as the library's configure op does, from argv: prefix,
// capacity, min_us, scratch prefix and gate path (the trace group's cpu_worker members and gate_path).
int main(int argc, char** argv) {
  if (argc != 6) return 4;
  set_options(Options{argv[1], atoll(argv[2]), atoll(argv[3]), argv[4], argv[5]});
  std::vector<float> scratch;
  for (int f=0; f<2; ++f) {
    Capture<ON> capture(4, 6, 3);
    capture.grow(scratch, 1048576, "probe");
    if (scratch.size()!=1048576 || scratch.back()!=0) return 3;
    capture.team_start();
    sglang::cpu_experts::Team team({}, 4, nullptr, 0);
    team.run([&](int w, int) {
      capture.worker_start(w);
      for (int p : {0,1,2,3,5}) {
        capture.begin(w,p);
        if (p==0 && w==3) usleep(20000);
        if (p==1 && w==2) {
          auto start=clock_ns(CLOCK_THREAD_CPUTIME_ID);
          while (clock_ns(CLOCK_THREAD_CPUTIME_ID)-start<3000000) asm volatile("" ::: "memory");
        }
        capture.add_work(w,p,w+1);
        capture.work_end(w,p);
        team.barrier();
        capture.end(w,p);
      }
      capture.worker_end(w);
    });
    capture.finish();
  }
}
''')
    production = work / "production.cpp"
    production.write_text(r'''
#include "worker_trace.hpp"
#include "moe/expert_stream/host/cpu_experts/team.hpp"
using namespace sglang::exl3_cpu::worker_trace;
#include <vector>
static_assert(std::is_empty_v<Capture<false>>);
int main() { Capture<false> c(4,6,3); std::vector<int> v; c.grow(v,8,"production");
 if(v.size()!=8 || v.back()!=0) return 3; c.team_start(); c.worker_start(0);
 c.begin(0,0); c.add_work(0,0,3); c.work_end(0,0); c.end(0,0); c.worker_end(0); c.finish();
 sglang::cpu_experts::Team t({}, 1, nullptr, 0); t.run([](int, int) {}); }
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


def run(probes, tmp_path, gate_path=None, **worker):
    """``worker`` overrides keys of the trace group's ``cpu_worker`` object; ``gate_path`` is its top-level key. The
    probe installs them through the kernel's options (set_options), the way the library's configure op does."""
    member = {"prefix": str(tmp_path / "phases"), "capacity": 65536, "min_us": 0, "scratch_prefix": "", **worker}
    env = {k: v for k, v in os.environ.items() if k != "SGLANG_EXPERT_STREAM_TRACE"}
    env.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE")
    args = [member["prefix"], str(member["capacity"]), str(member["min_us"]), member["scratch_prefix"], gate_path or ""]
    subprocess.run([str(probes[0]), *args], check=True, env=env)
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
    records = run(probes, tmp_path, capacity=25)
    assert len(records[1:-1]) == 20
    assert records[-1]["forwards_retained"] == records[-1]["dropped_forwards"] == 1


def test_duration_filter_retains_no_fast_forwards(probes, tmp_path):
    records = run(probes, tmp_path, min_us=1000000000)
    assert len(records) == 2 and records[-1]["forwards_seen"] == 2


def test_production_has_no_timing_resource_or_file_symbols(probes, tmp_path):
    subprocess.run([str(probes[1])], check=True, cwd=tmp_path,
                   env={**os.environ, "SGLANG_EXPERT_STREAM_TRACE": json.dumps({"cpu_worker": {"prefix": str(tmp_path / "forbidden")}})})
    assert not list(tmp_path.iterdir())
    symbols = subprocess.check_output(["nm", "-u", str(probes[1])], text=True)
    assert not any(s in symbols for s in ("clock_gettime", "getrusage", "fopen", "getenv", "syscall"))


def test_build_flag_requires_job_attribution():
    from sglang.kernels.ops.moe.cpu_kernel_build import worker_trace_defines
    from sglang.srt.layers.moe.expert_engine.config import (
        CpuExpertsConfig,
        CpuWorkerTraceConfig,
        ExpertStreamConfig,
        JobTraceConfig,
        TraceConfig,
    )

    def config(*, worker="", job=""):
        return ExpertStreamConfig(
            cpu_experts=CpuExpertsConfig(enabled=True),
            trace=TraceConfig(cpu_worker=CpuWorkerTraceConfig(prefix=worker), job=JobTraceConfig(prefix=job)),
        )

    assert worker_trace_defines(config()) == []
    with pytest.raises(ValueError, match="attribution"):
        worker_trace_defines(config(worker="/diagnostic/phases"))
    assert worker_trace_defines(config(worker="/diagnostic/phases", job="/diagnostic/events")) == [
        "-DEXL3_MOE_CPU_WORKER_TRACE=1"
    ]


def test_scratch_growth_is_attributed_before_team_entry(probes, tmp_path):
    run(probes, tmp_path, scratch_prefix=str(tmp_path / "scratch"))
    data = [json.loads(line) for line in next(tmp_path.glob("scratch.*.jsonl")).open()]
    summaries = [r for r in data if r.get("kind") == "scratch"]
    growth = [r for r in data if r.get("kind") == "growth"]
    assert [r["growths"] for r in summaries] == [1, 0]
    assert len(growth) == 1 and growth[0]["forward"] == 1
    assert growth[0]["old_size"] == growth[0]["old_capacity"] == 0
    assert growth[0]["size"] == 1048576 and growth[0]["moved"]
    assert summaries[0]["initialized_bytes"] == 4 * 1048576
    assert summaries[0]["minflt"] >= growth[0]["minflt"] > 0
    assert all(r["begin"] <= r["end"] and r["cpu_begin"] <= r["cpu_end"] for r in summaries + growth)
    assert summaries[0]["begin"] <= growth[0]["begin"] <= growth[0]["end"] <= summaries[0]["end"]
    assert data[-1]["dropped_growths"] == 0


def test_closed_gate_does_not_consume_worker_capacity(probes, tmp_path):
    gate = tmp_path / "gate"
    gate.write_bytes(bytes(4))
    data = run(probes, tmp_path, gate_path=str(gate), capacity=1)
    assert len(data) == 2 and data[-1]["dropped_forwards"] == 0
    assert data[-1]["forwards_seen"] == 2
