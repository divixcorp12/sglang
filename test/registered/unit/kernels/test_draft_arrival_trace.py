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
int run() {
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
    trace.draft_arrival(2, 7, seq, 123456, ns() - 123456 - 10000000);
    trace.draft_prepared(ns(), 2, 7, seq);
    trace.emit("draft_start", 2, 7, seq, -1);
    trace.draft_finished(6000000, 2, 7, seq);
  }
#endif
  return 0;
}
int main() {
  if (getenv("WORKER_PROBE")) { std::thread worker(run); worker.join(); return 0; }
  return run();
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


@pytest.mark.parametrize("pending,forward,reason", [(1000,0,0), (0,1000,1), (0,0,2), (10000,10000,None)])
def test_first_observation_and_threshold(arrival_probe, tmp_path, pending, forward, reason):
    subprocess.run([str(arrival_probe[0])], input="x", text=True, check=True, env={**os.environ,
        "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": str(tmp_path / "events"),
        "SGLANG_DRAFT_PENDING_TRIGGER_US": str(pending), "SGLANG_DRAFT_FORWARD_TRIGGER_US": str(forward),
        "SGLANG_DRAFT_ARRIVAL_TRIGGER_US": "1000" if reason == 2 else "0"})
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


@pytest.mark.parametrize("worker", [False, True])
def test_magic_trace_trigger_and_decode(arrival_probe, tmp_path, worker):
    import select
    import time
    tool = os.environ.get("SGLANG_TEST_MAGIC_TRACE")
    if not tool:
        pytest.skip("set SGLANG_TEST_MAGIC_TRACE for a live Intel PT smoke capture")
    probe = subprocess.Popen([str(arrival_probe[0])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, env={**os.environ, **({"WORKER_PROBE":"1"} if worker else {}),
                       "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": str(tmp_path/"events"),
                       "SGLANG_DRAFT_PENDING_TRIGGER_US":"1000"})
    trace = None
    try:
        assert probe.stdout.readline().strip() == "ready"
        tids = [int(p.name) for p in Path(f"/proc/{probe.pid}/task").iterdir() if int(p.name)!=probe.pid]
        tid = tids[0] if worker else probe.pid
        trace = subprocess.Popen([tool,"attach","-pid",str(tid),"-trigger","sglang_draft_delay_trigger",
            "-snapshot-size",os.environ.get("SGLANG_TEST_MAGIC_SNAPSHOT_SIZE","256K"),"-working-directory",str(tmp_path/"magic-work"),
            "-output",str(tmp_path/"trigger.fxt.gz")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = ""
        deadline = time.monotonic()+20
        while "[ Attached." not in log:
            assert trace.poll() is None, log
            assert time.monotonic()<deadline, log
            if select.select([trace.stdout],[],[],.1)[0]:
                log += trace.stdout.readline()
        probe.communicate("x",timeout=10)
        tail,_=trace.communicate(timeout=60)
        log+=tail
        (tmp_path/"magic.log").write_text(log)
        assert trace.returncode == 0, log
        assert "Snapshot taken" in log, log
        assert (tmp_path/"trigger.fxt.gz").stat().st_size > 100
    finally:
        for process in (probe,trace):
            if process is not None and process.poll() is None:
                process.terminate(); process.wait(timeout=10)


def test_perf_compat_preserves_data_and_only_normalizes_trace_start():
    import importlib.util
    spec = importlib.util.spec_from_file_location("compat", ROOT/"benchmarks/dsv41_baseline/magic_trace_perf_compat.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    line=" 807316/807316  737088.440126849:          1 branches:u:   tr strt jmp       0 [unknown] ([unknown]) =>     7f37efe45b61 read+0x11 (/usr/lib64/libc.so.6)\n"
    result=module.normalize(line)
    assert result == line.replace("tr strt jmp", "tr strt    ")
    assert len(result)==len(line)
    assert module.normalize(line.replace("tr strt jmp", "return"))==line.replace("tr strt jmp", "return")
    assert module.normalize("symbol containing tr strt jmp\n")=="symbol containing tr strt jmp\n"


def test_magic_selects_engine_before_inherited_openmp_worker_names():
    import importlib.util
    spec=importlib.util.spec_from_file_location("capture",ROOT/"benchmarks/dsv41_baseline/run_omp_serving_capture.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    threads=[dict(tid=tid,comm="exl3-cpu-exp0") for tid in (861080,861071,861072)]
    rows=[dict(pid=858012,expert_threads=threads)]*4
    assert module.select_engine_leader(rows)==(858012,861071)
    with pytest.raises(RuntimeError,match="one owned"):
        module.select_engine_leader([*rows,dict(pid=123,expert_threads=threads)])


def test_arrival_analysis_joins_epochs_and_bounds_clock_uncertainty(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location("arrival",ROOT/"benchmarks/dsv41_baseline/analyze_draft_arrival.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    events=[]
    names=["draft_observed","draft_selected","draft_record_ready","draft_payload_ready","draft_start","draft_end"]
    for epoch in (1,2):
        for i,name in enumerate(names):
            events.append(dict(event=name,ns=110000+i*1000,row=0,gen=epoch,seq=1,group=-1,a=100000,b=99,c=0))
    path=tmp_path/"events.42.engine.jsonl"
    def write(footer):
        path.write_text("\n".join(json.dumps(e) for e in [*reversed(events),*footer]))
    write([dict(dropped=0)])
    (tmp_path/"events.42.draft-clock.json").write_text(json.dumps(dict(offset_low=1000,offset_high=3000)))
    (tmp_path/"events.42.draft-clock-end.json").write_text(json.dumps(dict(offset_low=2000,offset_high=2500)))
    result=module.summarize(tmp_path)
    assert len(result["requests"])==2 and {r["epoch"] for r in result["requests"]}=={1,2}
    assert all(r["publication_to_observe_us"]==[7.5,8.0] for r in result["requests"])
    assert result["stats"]["select_us"]["max"]==1.0
    write([dict(dropped=1)])
    with pytest.raises(ValueError,match="overflow"):
        module.summarize(tmp_path)
    write([])
    with pytest.raises(ValueError,match="footer"):
        module.summarize(tmp_path)
    write([dict(dropped=0)])
    (tmp_path/"events.42.draft-clock-end.json").write_text(json.dumps(dict(offset_low=4000,offset_high=4500)))
    with pytest.raises(ValueError,match="clock anchor intervals disagree"):
        module.summarize(tmp_path)
