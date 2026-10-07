"""Request timing and the optimized noinline trigger; native CPU probe, no model."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def arrival_probe(tmp_path_factory):
    work = tmp_path_factory.mktemp("draft-arrival")
    source = work / "probe.cpp"
    source.write_text(r'''
#include "job_trace.h"
#include <thread>
using namespace sglang::expert_stream;
int64_t ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000000000LL+t.tv_nsec; }
int main() {
#ifdef PROD
  JobTrace<false> trace("prod"); trace.emit("forbidden",0,0,0,0);
#else
  JobTrace<true> trace("arrival");
  // Ready handshake also makes this probe usable for a real magic-trace attach.
  puts("ready"); fflush(stdout); getchar();
  for (uint32_t seq=1; seq<=2; ++seq) {
    trace.draft_observe(&trace, seq);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    trace.draft_observe(&trace, seq); // must preserve the first observation
    trace.draft_selected(ns(), 2, 7, seq, 123456);
    trace.draft_prepared(ns(), 2, 7, seq);
    trace.emit("draft_start", 2, 7, seq, -1);
    trace.draft_finished(6000000, 2, 7, seq);
  }
#endif
}
''')
    binaries = []
    for mode in ("instr", "prod"):
        binary = work / mode
        subprocess.run(["g++", "-std=c++20", "-O3", "-pthread", "-rdynamic",
                        *( ["-DPROD"] if mode == "prod" else []), "-I",
                        str(ROOT / "python/sglang/kernels/jit/csrc/moe/expert_stream/host"),
                        str(source), "-o", str(binary)], check=True, capture_output=True)
        binaries.append(binary)
    return binaries


@pytest.mark.parametrize("pending,forward,reason", [(1000,0,0), (0,1000,1), (10000,10000,None)])
def test_first_observation_and_threshold(arrival_probe, tmp_path, pending, forward, reason):
    subprocess.run([str(arrival_probe[0])], input="x", text=True, check=True, env={**os.environ,
        "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": str(tmp_path / "events"),
        "SGLANG_DRAFT_PENDING_TRIGGER_US": str(pending), "SGLANG_DRAFT_FORWARD_TRIGGER_US": str(forward)})
    records = [json.loads(l) for l in next(tmp_path.glob("events.*")).read_text().splitlines()]
    events = records[1:-1]
    assert records[-1] == {"dropped": 0}
    for seq in (1,2):
        e = {r["event"]: r for r in events if r["seq"] == seq}
        assert e["draft_selected"]["ns"] - e["draft_observed"]["ns"] >= 2_000_000
        assert e["draft_observed"]["a"] == 123456 and e["draft_observed"]["b"] > 0
        assert all(r["gen"] == 7 and r["row"] == 2 for r in e.values())
        assert e["draft_selected"]["ns"] <= e["draft_record_ready"]["ns"] <= e["draft_payload_ready"]["ns"]
    triggers = [r for r in events if r["event"] == "draft_trigger"]
    assert len(triggers) == (reason is not None)
    if triggers:
        assert triggers[0]["a"] == reason and triggers[0]["b"] >= triggers[0]["c"]


def test_trigger_survives_optimized_build_and_compiles_out_of_prod(arrival_probe):
    for binary, expected in zip(arrival_probe, (True,False)):
        symbols = subprocess.check_output(["nm", "--defined-only", str(binary)], text=True)
        assert ("sglang_draft_delay_trigger" in symbols) == expected
    assembly = subprocess.check_output(["objdump", "-d", str(arrival_probe[0])], text=True)
    assert "call" in assembly and "<sglang_draft_delay_trigger>" in assembly
