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
int main() {
  { JobTrace<false> prod("prod"); prod.emit("forbidden", 0, 0, 0, 0); }
  JobTrace<true> trace("probe");
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
