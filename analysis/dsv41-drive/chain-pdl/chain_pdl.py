#!/usr/bin/env python3
"""The real lease chain, captured in a CUDA graph, timed with and without PDL (plan Task 8; test hook builds).

Scenarios (StreamService, test/manual/dsv41/test_exl3_piece_stream_cuda.py; TOP_K lanes of one layer):
  all_hit        every lane a RAM hit: W1 claims them all, S has nothing to wait for; the chain's own cost
  mixed          half the lanes RAM misses each replay (the host reads them): the chain behind a real read
  all_hit_ce     all_hit with the copy engine and the copy wait (CW) in the chain
  all_hit_ce_sm  all_hit_ce with SM small copies (CW reads the small tensors), as the production recipe runs
Modes: off (production module), pdl (EXL3_RAM_MISS_TEST_PDL), pdl_early (+ EXL3_RAM_MISS_TEST_PDL_EARLY).

The chain is captured stage by stage (post, W1, C1, A1, S, A2, [CW], F, total), not through StreamService.step(),
which ends in torch.cuda.synchronize() and so cannot be captured. Every module variant is built and loaded before
any service starts, and the run needs CUDA_MODULE_LOADING=EAGER: no kernel may load for the first time after the
copy engine arms (LEASE_PROTOCOL.md 7.6). Every 20th replay's destination bytes are checked against the source rows.
Per scenario and mode it also records the captured graph's edges (did PDL survive capture as programmatic edges?).
With --stamp it runs the EXL3_RAM_MISS_TEST_PDL_STAMP builds instead and records the stamp ring's edge gaps and
prologues; those timings carry the stamps' own cost and are not used for the saving.

    CUDA_MODULE_LOADING=EAGER PYTHONPATH=<pdl-probe worktree>/python python chain_pdl.py \\
        --repo <pdl-probe worktree> --tmp <dir> --out chain.jsonl [--stamp]
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chain_report  # noqa: E402

PDL, EARLY, STAMP = "EXL3_RAM_MISS_TEST_PDL", "EXL3_RAM_MISS_TEST_PDL_EARLY", "EXL3_RAM_MISS_TEST_PDL_STAMP"
MODES = {"off": [], "pdl": [PDL], "pdl_early": [PDL, EARLY]}
SCENARIOS = {"all_hit": {}, "mixed": {}, "all_hit_ce": {"copy_engine": True},
             "all_hit_ce_sm": {"copy_engine": True, "sm_small": True}}
STAMP_WORDS = 16384 * 4 + 1  # lease_device.cuh kPdlStampSlots * 4 + the count


def build(defines: list[str], stamp: bool):
    """The device module for `defines` (+ the stamp ring and its readback when `stamp`); None = production."""
    from sglang.kernels.jit.utils.compile.loader import load_jit
    from sglang.kernels.ops.moe import expert_stream_transport as ops

    if not defines and not stamp:
        return None
    flags = defines + ([STAMP] if stamp else [])
    wrappers = ops._device_wrappers("exl3") + (
        [("expert_stream_pdl_stamps", "LeaseProtocolKernel::pdl_stamps")] if stamp else [])
    return load_jit("expert_stream_exl3", "chainpdl", "-".join(flags), cuda_files=[ops.LAYOUTS["exl3"].device_source],
                    cuda_wrappers=wrappers, extra_cuda_cflags=[f"-D{f}" for f in flags])


def chain(s) -> None:
    """One layer's chain in production order, capturable (StreamService.step minus its synchronize)."""
    s.post()
    s.hit_wait()
    s.copy1()
    s.ack1()
    s.stream()
    s.ack2()
    if s.copy_engine:
        s.copy_wait()
    s.finalize()
    s.total()


def retired(s) -> bool:
    c = s.counters()
    return c["leases_granted"] == c["leases_acked"] + c["leases_voided"] + c["leases_copied"]


def graph_edges(edges_mod, graph) -> list[list]:
    buf = torch.zeros(1 << 20, dtype=torch.uint8)
    n = edges_mod.chain_graph_edges(int(graph.raw_cuda_graph()), buf)
    rows = bytes(buf[:n].tolist()).decode().splitlines()
    return [[f, t, int(ty), int(port)] for f, t, ty, port in (r.split("\t") for r in rows)]


def run(repo: Path, tmp: Path, scenario: str, mode: str, module, edges_mod, replays: int, stamp: bool) -> list[dict]:
    sys.path.insert(0, str(repo / "test/manual/dsv41"))
    from test_exl3_piece_stream_cuda import TOP_K, StreamService

    from sglang.kernels.ops.moe import expert_stream_transport as ops

    original = ops._device_module
    if module is not None:
        ops._device_module = lambda layout="exl3": module  # before the service: its warm-up request uses it too
    work = tmp / f"{scenario}-{mode}{'-stamp' if stamp else ''}"
    work.mkdir(parents=True, exist_ok=True)  # write_fake_exl3 writes into it but does not create it
    try:
        s = StreamService(work, **SCENARIOS[scenario])
    finally:
        ops._device_module = original
    records = []
    try:
        base = list(range(TOP_K))
        s.plan(base)
        chain(s)  # loads the rows: every later all-hit request hits RAM
        torch.cuda.synchronize()
        assert s.until(lambda: retired(s)), s.counters()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            chain(s)
        edges = graph_edges(edges_mod, graph)
        records.append({"kind": "graph", "scenario": scenario, "mode": mode, "stamp": stamp, "edges": edges,
                        "programmatic": sum(1 for e in edges if e[2] == 1)})
        torch.cuda.synchronize()
        assert s.until(lambda: retired(s)), s.counters()
        ring = torch.zeros(STAMP_WORDS, dtype=torch.int64)
        times, hits = [], []
        for r in range(replays + 10):
            if scenario == "mixed":
                # keep half of the previous plan (RAM hits) and bring in half new experts (RAM misses)
                keep = base[TOP_K // 2:]
                fresh = [(base[-1] + 1 + k) % 16 for k in range(TOP_K)]
                base = keep + [e for e in fresh if e not in keep][: TOP_K - len(keep)]
            s.plan(base)
            hits.append(sum(s.host.contains(s.row, e) for e in base))
            if stamp and r == 10:
                module.expert_stream_pdl_stamps(ring)  # drop the warm-up replays' stamps
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graph.replay()
            e1.record()
            torch.cuda.synchronize()
            assert s.until(lambda: retired(s)), s.counters()
            if r % 20 == 0:
                want = s.expected(base)
                for lane, expert in enumerate(base):
                    for n in s.names:
                        got = s.dest[n][lane].cpu().view(torch.uint8)
                        assert torch.equal(got, want[expert][n].view(torch.uint8)), (scenario, mode, lane, n)
            if r >= 10:
                times.append(e0.elapsed_time(e1) * 1e3)
        rec = {"kind": "chain", "scenario": scenario, "mode": mode, "stamp": stamp,
               "replay_us_p50": round(statistics.median(times), 2), "replay_us_p10": round(sorted(times)[len(times) // 10], 2),
               "hits_mean": round(statistics.mean(hits), 2), "replays": replays}
        if stamp:
            module.expert_stream_pdl_stamps(ring)
            count = int(ring[-1])
            if count > 16384:
                raise SystemExit(f"stamp ring overflowed: {count} stamps")
            words = ring[:-1].view(16384, 4)[:count].tolist()
            reps = chain_report.replays(words)
            records.append({"kind": "stamps", "scenario": scenario, "mode": mode, "stamps": count,
                            "replays_seen": len(reps), "edges": chain_report.edge_summary(reps),
                            "prologue": chain_report.prologue(reps)})
        else:
            records.append(rec)
        return records
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--stamp", action="store_true")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    if os.environ.get("CUDA_MODULE_LOADING") != "EAGER":
        raise SystemExit("run with CUDA_MODULE_LOADING=EAGER (the copy engine refuses lazy module loading)")
    from sglang.kernels.jit.utils.compile.loader import load_jit

    edges_mod = load_jit("chain_pdl_graph", cuda_files=[str(HERE / "chain_graph.cuh")],
                         cuda_wrappers=[("chain_graph_edges", "chain_graph_edges")])
    modules = {mode: build(defines, a.stamp) for mode, defines in MODES.items()}  # all built before any service
    tmp = Path(a.tmp or tempfile.mkdtemp(prefix="chain-pdl-"))
    tmp.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as out:
        for scenario in a.scenarios.split(","):
            for mode in MODES:
                for rec in run(repo, tmp, scenario, mode, modules[mode], edges_mod, a.replays, a.stamp):
                    line = json.dumps(rec)
                    print(line if rec["kind"] != "graph" else json.dumps({**rec, "edges": len(rec["edges"])}),
                          flush=True)
                    out.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
