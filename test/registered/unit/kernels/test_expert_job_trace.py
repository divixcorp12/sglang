"""Bounded concurrent tracing and the production compile-out contract (CPU only)."""

import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="module")
def trace_probe(tmp_path_factory):
    root = Path(__file__).resolve().parents[4]
    work = tmp_path_factory.mktemp("job-trace-build")
    source = work / "probe.cpp"
    source.write_text(r'''
#include "job_trace.h"
#include <thread>
using namespace sglang::expert_stream;
int main(int argc, char**) {
  { JobTrace<false> prod("prod"); prod.emit("forbidden", 0, 0, 0, 0);
    prod.resources("forbidden", "forbidden", 0, 0, 0); }
  JobTrace<true> trace("probe");
  if (argc > 1) {
    trace.resources("cpu_faults_start", "cpu_switches_start", 0, 0, 1);
    volatile char* memory = new char[4 << 20];
    for (int i=0; i<(4 << 20); i+=4096) memory[i] = 1;
    usleep(2000);
    delete[] memory;
    trace.resources("cpu_faults_end", "cpu_switches_end", 0, 0, 1);
    return 0;
  }
  std::thread a([&] { for (int i=0; i<70000; ++i) trace.emit("a", i, 1, i, 0); });
  std::thread b([&] { for (int i=0; i<70000; ++i) trace.emit("b", i, 1, i, 1); });
  a.join(); b.join();
}
''')
    binary = work / "probe"
    subprocess.run(
        ["g++", "-std=c++20", "-O2", "-pthread", "-I",
         str(root / "python/sglang/kernels/jit/csrc/moe/expert_stream/host"),
         str(source), "-o", str(binary)], check=True, capture_output=True,
    )
    return binary


def test_bounded_trace_preserves_distinct_concurrent_events(trace_probe, tmp_path):
    prefix = tmp_path / "events"
    subprocess.run([str(trace_probe)], check=True,
                   env={**os.environ, "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": str(prefix)})
    files = list(tmp_path.glob("events.*.jsonl"))
    assert len(files) == 1 and ".probe." in files[0].name
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert records[0]["clock"] == "CLOCK_MONOTONIC"
    events = records[1:-1]
    assert len(events) == 131072 and records[-1] == {"dropped": 8928}
    assert len({(e["group"], e["seq"]) for e in events}) == len(events)
    assert all(e["ns"] >= records[0]["monotonic_ns"] for e in events)


def test_disabled_trace_writes_nothing(trace_probe, tmp_path):
    env = dict(os.environ)
    env.pop("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", None)
    subprocess.run([str(trace_probe)], check=True, env=env, cwd=tmp_path)
    assert not list(tmp_path.iterdir())


def test_thread_resource_snapshots_observe_faults_and_sleep(trace_probe, tmp_path):
    subprocess.run([str(trace_probe), "resources"], check=True, env={**os.environ,
                   "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": str(tmp_path / "events"),
                   "SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE": "1"})
    records = [json.loads(line) for line in next(tmp_path.glob("events.*.jsonl")).read_text().splitlines()]
    events = {r["event"]: r for r in records[1:-1]}
    assert len(events) == 4 and records[-1]["dropped"] == 0
    assert events["cpu_faults_end"]["a"] > events["cpu_faults_start"]["a"]
    assert events["cpu_faults_end"]["b"] >= events["cpu_faults_start"]["b"]
    assert events["cpu_faults_end"]["c"] == events["cpu_faults_start"]["c"] > 0
    assert events["cpu_switches_end"]["a"] > events["cpu_switches_start"]["a"]
    assert events["cpu_switches_end"]["c"] >= events["cpu_switches_start"]["c"]


def test_classifier_respects_dma_completion_interval(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[4] / "benchmarks/dsv41_baseline/expert_job_trace.py"
    spec = importlib.util.spec_from_file_location("expert_job_trace", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header = {"schema": 1, "clock": "CLOCK_MONOTONIC", "monotonic_ns": 1, "epoch_ns": 1}
    def event(kind, ns, gen=1, seq=1, group=0, a=0, b=0, c=0):
        return dict(event=kind, ns=ns, row=0, gen=gen, seq=seq, group=group, a=a, b=b, c=c)
    def write(name, events):
        (tmp_path / name).write_text("\n".join(json.dumps(e) for e in [header, *events, {"dropped": 0}]))
    for end, expected in [(200000, "cpu_last_confirmed"), (50000, "dma_last_confirmed"), (100000, "ambiguous")]:
        write("events.1.exl3-cpu-exp0.jsonl", [event("cpu_submit", 1000), event("cpu_start", 2000),
              event("cpu_shape", 3000, a=2, b=1, c=1), event("cpu_end", end)])
        write("events.1.copy.jsonl", [event("copy_submit", 1000, c=1),
              event("copy_issue", 2000, a=1024, b=1), event("copy_dma_observed", 120000, a=80000),
              event("group_done", 210000), event("gate_open", 220000)])
        result, _ = module.summarize(str(tmp_path / "events.*.jsonl"))
        assert result["rows"][0]["last"] == expected
        assert result["layer_rows"][0]["last"] == expected
        assert result["rows"][0]["token_expert_routes"] == 1
    write("events.1.copy.jsonl", [event("copy_submit", 1000, c=1), event("copy_issue", 2000, a=1024, b=1),
          event("group_done", 210000), event("gate_open", 220000)])
    result, _ = module.summarize(str(tmp_path / "events.*.jsonl"))
    assert result["rows"][0]["last"] == "ambiguous"
    assert result["rows"][0]["dma_bound_source"] == "retirement_fallback"
    write("events.1.copy.jsonl", [event("copy_submit", 1000)])
    with pytest.raises(ValueError, match="incomplete copy"):
        module.summarize(str(tmp_path / "events.*.jsonl"))
