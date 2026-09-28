# Expert-stream transfer measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "are our syncs and blocks granular enough?" answerable with numbers on each host (divix01 at PCIe
Gen3 now, a Gen5 host later). The plan measures three things, each on its own: the copy mechanisms against the link
(A), what PDL would save on the lease chain (B), and where each piece's time goes inside the chain (C).

**Architecture:** Part A is analysis-only. It extends the existing C1 copy bench into a portable sweep that reports
GB/s against bytes in flight for every candidate way of moving pinned host bytes, plus the link ceiling, read
latency and flag round trip it measured on that host. Part B first bounds PDL's saving on a skeleton chain that has
the production launch shapes (analysis only). If that bound is large enough, it measures the real chain through a
test-only compile hook. Part C adds a diagnostic device timeline to the lease chain: a fixed device ring stamped
under `-DEXPERT_STREAM_TIMELINE`, drained at replay boundaries, calibrated to the host clock, and joined with the
host's stage trace. The ring is off by default and the default build compiles to identical SASS.

**Tech Stack:** CUDA 13.4 on divix01 (`sm_120f`, RTX 5090, PCIe Gen3 x16), PTX inline asm (`ld.global.cv`,
`cp.async`, `cp.async.bulk`, `mbarrier`, `griddepcontrol`), CUDA runtime (`cudaMemcpyBatchAsync`), tvm-ffi JIT
(`load_jit`), PyTorch CUDA graphs and events, pytest.

**Spec:** No spec file. The spec is the team-lead request of 2026-09-27 (the user's question, quoted in the Goal).
These documents travel with it, and an executor reads each one when its task cites it:
`docs/superpowers/plans/2026-09-25-dsv41-copy-compute-overlap.md:44-60` (Gen3 C1 sweep),
`analysis/dsv41-drive/NC_VISIBILITY.md:205-225` (visibility, `.nc` staleness, bandwidth cell),
`analysis/dsv41-drive/task6-microbench/results.md:130-131` (711 ns serial acquire),
`analysis/dsv41-drive/MOE_SERVICE_TRACE_PLAN.md` section 2 (device ring design),
`docs/superpowers/plans/2026-09-27-expert-stream-native-sync.md` (named helpers, SASS gate),
`analysis/dsv41-drive/LEASE_PROTOCOL.md` (E1 amendment: piece visibility), `CLAUDE.md` (nsys and L2 rules),
`.claude/rules/divix01-run-protocol.md`, `.claude/skills/env-var-conventions/SKILL.md`,
`.claude/skills/add-jit-kernel/SKILL.md` (PDL rules).

## Decision table

The plan ends in decisions. After Tasks 6, 8 and 13, fill in the last column in each part's `results.md`. Decide
**per host**: a "no" on Gen3 says nothing about Gen5. Every row's evidence comes from one host's runs.

| Follow-up | Justified when (all conditions, on the same host) | Numbers from | Expected on divix01 (Gen3) |
|---|---|---|---|
| Deeper unroll in S (`stream_copy_slice` U=1 -> 4) | `sm_cv16` G8 U4 >= 1.05 x `s_pattern` (G8 U1) GB/s, **and** C's copy share (`copy_ns / s_wall_ns`) >= 0.20 | A Task 3, C Task 13 | No: every SM cell measured 12.1-12.4 GB/s |
| cp.async ring or TMA in S | best `ldgsts` or `tma` cell with <= 32 KiB per block in flight >= 1.10 x best `sm_cv*` cell, its fresh check passed, **and** C's copy share >= 0.20 | A Task 4, C Task 13 | No, unless a bulk path beats the 12.3 GB/s SM ceiling (the copy engine gets 13.5) |
| Wider copy-wait SM reads (more than 1 block) | `cw_pattern` (G1 U4, 16 KiB in flight) < 0.90 x measured ceiling, **and** C's `cw_wait_ns` > 0 on >= 10% of copy-engine requests | A Task 3, C Task 13 | No: 16 KiB exceeds Gen3's bandwidth-delay product (~11.2 KiB) |
| PDL on the chain | Task 7 saving >= 2 us per layer (80 us per 40-layer step) to run Task 8; then Task 8 all-hit saving >= 1% of untraced ms/token (0.67 ms/step at 66.8) to put a ship decision to the user | B Tasks 7-8 | Unknown |
| `cudaMemcpyBatchAsync` in the copy thread | `ce_batch_small` >= 1.5 x `ce_each_small` GB/s, or host ns per call <= 0.7 x, at 4 lanes, **and** C: CW spun (`cw_done.bits == 1`) on >= 10% of copy-engine requests | A Task 5, C Task 13 | Plausible: small copy-engine copies run at 1.5-5 GB/s |
| Fewer, larger pieces (8 -> 4) | C: (`pass_fixed_ns` + median `publish_to_seen_ns`) >= 0.25 x `per_piece_ns` | C Task 13 | No: about 6% by the Gen3 estimate |
| More, smaller pieces (8 -> 16; wire change: mask width) | C: `tail_ns` (S exit - last publish) >= 2 x `per_piece_ns` on >= 50% of requests, **and** the fixed-cost ratio above < 0.10 | C Task 13 | Unknown |
| None of the above | every row fails | | |

Why these thresholds. On Gen5 (~50 GB/s practical, 711 ns read latency) the bandwidth-delay product is about
36 KiB. S keeps about 32 KiB in flight (8 x 256 threads x one 16 B load), and the copy wait keeps 16 KiB
(1 x 256 x 4 x 16 B), which caps it near 22 GB/s. Fixed per-piece costs are the 8.6 us flag visibility plus S's
per-pass overhead (mask poll, `kDemandDone` read, barriers, `__nanosleep(256)`). A piece's copy time falls from
~138 us per 1.66 MB at Gen3 to ~33 us at Gen5, so those costs grow from ~6% of it to 25%+. Part A measures the
latency and ceiling on each host. Part C measures the fixed cost and the per-piece copy time.

## Global Constraints

- Nothing here changes default behaviour. Every production-code edit sits behind `#ifdef EXPERT_STREAM_TIMELINE`,
  `#ifdef EXL3_RAM_MISS_TEST_PDL` / `EXL3_RAM_MISS_TEST_PDL_EARLY`, a host `if (c.trace)`, or
  `SGLANG_DSV41_DEBUG_DEVICE_TIMELINE`, which defaults to `False`.
- The default build of the expert_stream device module must produce the same SASS, instruction for instruction:
  `sass_gate.py diff <base> <after> --exact` must pass (native-sync plan, Task 0).
- Part A (Tasks 1-6) and Task 7 add files under `analysis/dsv41-drive/` only. No production file changes.
- Part B measures PDL. It never turns PDL on in a production launch.
- **Gate:** Task 8 and Part C (Tasks 9-13) start only after `expert-stream-native-sync` has merged into
  `origin/master`. Those tasks edit the same three headers. Check with
  `git fetch origin && git merge-base --is-ancestor origin/expert-stream-native-sync origin/master && echo MERGED`.
  If it does not print `MERGED`, stop and report. Do not branch from an unmerged tip without the user's say.
- Visibility contract for any read of host bytes that may still be written (tag LOADING): use `ld.global.cv`, or
  order the read after an acquire. Never use `.nc`; the only `.nc` read is the fresh check's negative control, which
  the sweep never includes. A `cp.async.bulk` (async-proxy) read needs `fence.proxy.async.global` after the
  generic-proxy acquire. Weak or plain loads, such as `sgl_kernel` `AlignedVector::load`, are not acceptable for
  S-style reads.
- Working sets for device-to-device work exceed the 96 MiB L2. A GB/s figure above the host's theoretical PCIe
  payload ceiling (15.75 GB/s for Gen3 x16) is a defect, not a result.
- divix01 protocol (`.claude/rules/divix01-run-protocol.md`):
  - Code reaches divix01 by `git push`, then `git fetch` into a private worktree.
  - Set `PYTHONPATH=$PWD/python` and check that `sglang.__file__` lies in that worktree.
  - Run CPU jobs under `taskset -c 0-63` and GPU jobs under
    `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`.
  - Read suite status from `${PIPESTATUS[0]}`, and record the command next to every count you quote.
- Env vars follow `.claude/skills/env-var-conventions/SKILL.md`: an `EnvBool` in `python/sglang/srt/environ.py`,
  read with `.get()`, and a `DEBUG_` verb for diagnostic-only knobs.
- No nsys trace is required. If one is used, take ms/token or step-tail idle only from `--cuda-graph-trace=graph`,
  never from node mode.
- A run with the timeline on is not a throughput run: its ms/token includes the drain copy and the calibration.

## Review Focus

1. **A copy variant that reads stale host bytes.** This happens with a weak or `.nc` load, or with a bulk copy
   missing the proxy fence. It passes every static-byte correctness check. Expected: each swept method passes the
   fresh check, and the `.nc` negative control fails it; if the control passes, the check is blind and the sweep
   refuses to report. Pinned in Task 3 (`sm_cv16`, `sm_cv32`, `nc_control`) and Task 4 (`ldgsts`, `tma`).
2. **The timeline module loaded, or one of its kernels launched for the first time, after the copy engine arms.**
   Under the copy engine, a first kernel launch while CW spins can stall the copy thread until fail-stop
   (LEASE_PROTOCOL.md 7.6). Expected: `ExpertStreamDevice.__init__` binds and calibrates before any capture or
   arming. Pinned by Task 12's `test_timeline_is_bound_and_calibrated_at_construction`.
3. **The ring is overwritten between drains, or a drain overlaps a replay.** Either way, partial data gets used
   silently. Expected: `lost` and `torn` are counted, and the report refuses a window with `lost > 0` unless
   `--allow-loss`. Pinned by Task 9's decoder tests, Task 11's `test_overwrite_is_detected`, and Task 13's
   `test_report_refuses_lost_records`.
4. **A mismatched join.** Cases: an advisory or touch record, two lanes naming one expert, a READY lane (a hit W1
   missed), or a device request with no host record. Expected: each is excluded or mapped explicitly, and each is
   counted in coverage; none is dropped silently. Pinned by Task 13's join tests.
5. **A GB/s cell above the link ceiling.** Causes include an L2-resident destination or host-side API time counted
   as transfer time. Expected: `mech_report.py` lists it and exits 1. Pinned by Task 1's
   `test_above_ceiling_cells_are_reported` and by rotating destination slots in Task 3.

---

## Part A: portable copy-mechanism sweep (starts now; `analysis/` only)

Branch `expert-stream-transfer-measurement` from `origin/master`. Create it with superpowers:using-git-worktrees,
then `git push -u origin expert-stream-transfer-measurement`. The divix01 worktree for every Part A and Task 7 run:

```bash
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-xfer origin/expert-stream-transfer-measurement \
  && git -C /data/models/slang/nvfp4-work/wt-xfer log -1 --oneline'
```

After each later push, refresh it:
`ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement'`.

Expected outcome on Gen3, stated up front so the run can confirm or refute it:
- Every SM method (`sm_cv16`, `sm_cv32`, `ldgsts`, `tma`) sits at 12.1-12.4 GB/s once it has about 11 KiB in flight.
- The copy engine gets 13.5-13.8 GB/s.
- The knees sit near the measured bandwidth-delay product: latency 711 ns x ceiling ~12.3 GB/s ≈ 8.7 KiB, against
  15.75 x 711 ≈ 11.2 KiB at the theoretical ceiling.
- A bulk path above 12.5 GB/s would be the first SM path to close the gap to the copy engine. Record it as that, not
  as noise.

### Task 1: The sweep's report module (pure, CPU-tested)

**Files:**
- Create: `analysis/dsv41-drive/copy-mechanism/mech_report.py`
- Test: `analysis/dsv41-drive/copy-mechanism/mech_report_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `link_gbs(gen: int, width: int) -> float`; `bdp_bytes(latency_ns: float, gbs: float) -> int`;
  `knee(points: list[tuple[int, float]], fraction: float = 0.95) -> int | None`;
  `above_ceiling(cells: list[dict], ceiling: float) -> list[dict]`; `summarize(records: list[dict]) -> dict`.
  The record shapes, written by Task 3's driver:
  - meta: `{"meta": True, "host": str, "pcie_gen": int, "pcie_width": int, ...}`
  - cell: `{"kind": "cell", "method": str, "name": str | None, "in_flight": int, "gbs": float, ...}`
  - latency: `{"kind": "latency", "serial_acquire_ns": float, "device_acquire_ns": float}`
  - pingpong: `{"kind": "pingpong", "rtt_ns_p50": int, "rtt_ns_min": int, "rounds": int}`
  - fresh: `{"kind": "fresh", "method": str, "fresh": bool}`

- [ ] **Step 1: Write the failing tests**

```python
"""mech_report: link ceilings, knees, bandwidth-delay products and the refusals (CPU, no torch)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mech_report as report  # noqa: E402


def _meta(gen=3, width=16):
    return {"meta": True, "host": "h", "pcie_gen": gen, "pcie_width": width}


def _cell(method, in_flight, gbs, name=None):
    return {"kind": "cell", "method": method, "in_flight": in_flight, "gbs": gbs, "name": name}


def test_gen3_x16_is_15_75_and_gen5_x16_is_63():
    assert report.link_gbs(3, 16) == pytest.approx(15.754, abs=0.01)
    assert report.link_gbs(5, 16) == pytest.approx(63.015, abs=0.01)
    with pytest.raises(ValueError):
        report.link_gbs(2, 16)


def test_bdp_is_latency_times_rate():
    assert report.bdp_bytes(711, 50.0) == 35550  # the Gen5 figure the plan argues from
    assert report.bdp_bytes(711, 15.75) == 11198


def test_knee_is_the_fewest_bytes_within_five_percent_of_the_best():
    points = [(4096, 5.0), (8192, 11.8), (16384, 12.2), (32768, 12.3)]
    assert report.knee(points) == 8192  # 11.8 >= 0.95 * 12.3 = 11.685
    assert report.knee([]) is None


def test_above_ceiling_cells_are_reported():
    cells = [_cell("sm_cv16", 32768, 12.3), _cell("sm_cv16", 65536, 16.1)]
    assert report.above_ceiling(cells, 15.75) == [cells[1]]


def test_summarize_names_methods_knees_and_the_named_cells():
    records = [
        _meta(),
        _cell("sm_cv16", 8192, 11.8), _cell("sm_cv16", 32768, 12.3, name="s_pattern"),
        _cell("ce_each", 0, 13.6),
        {"kind": "latency", "serial_acquire_ns": 711.0, "device_acquire_ns": 117.0},
        {"kind": "pingpong", "rtt_ns_p50": 4000, "rtt_ns_min": 3500, "rounds": 64},
        {"kind": "fresh", "method": "sm_cv16", "fresh": True},
        {"kind": "fresh", "method": "nc_control", "fresh": False},
    ]
    s = report.summarize(records)
    assert s["theoretical_gbs"] == pytest.approx(15.75, abs=0.01)
    assert s["measured_ceiling_gbs"] == 13.6
    assert s["methods"]["sm_cv16"]["knee_bytes"] == 8192
    assert s["methods"]["ce_each"]["knee_bytes"] is None  # a copy-engine call has no in-flight knob
    assert s["named"] == {"s_pattern": 12.3}
    assert s["bdp_bytes"] == report.bdp_bytes(711.0, 13.6)
    assert s["unsafe"] == [] and s["control_blind"] is False


def test_a_method_that_failed_its_fresh_check_is_unsafe_and_a_passing_control_is_blind():
    records = [
        _meta(), _cell("tma", 65536, 12.0),
        {"kind": "fresh", "method": "tma", "fresh": False},
        {"kind": "fresh", "method": "nc_control", "fresh": True},
    ]
    s = report.summarize(records)
    assert s["unsafe"] == ["tma"]
    assert s["control_blind"] is True


def test_summarize_needs_exactly_one_meta():
    with pytest.raises(ValueError):
        report.summarize([_cell("sm_cv16", 1, 1.0)])
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `python3 -m pytest analysis/dsv41-drive/copy-mechanism/mech_report_test.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'mech_report'`.

- [ ] **Step 3: Write `mech_report.py`**

```python
#!/usr/bin/env python3
"""Summaries of a copy-mechanism sweep (mech_bench.py JSONL): link ceilings, knees, bandwidth-delay product.

Pure Python (no torch, no CUDA) so it runs anywhere; tested by mech_report_test.py.

    python3 mech_report.py <results.jsonl>   # markdown summary; exit 1 on an above-ceiling cell, an unsafe method
                                             # or a blind fresh check
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

# Payload GB/s per PCIe lane after 128b/130b line coding, before TLP/DLLP overhead: Gen3 x16 = 15.75.
LANE_GBS = {3: 8.0 * 128 / 130 / 8, 4: 16.0 * 128 / 130 / 8, 5: 32.0 * 128 / 130 / 8}
KNEE_FRACTION = 0.95
CONTROL = "nc_control"  # the fresh check's negative control: it must come out stale


def link_gbs(gen: int, width: int) -> float:
    if gen not in LANE_GBS:
        raise ValueError(f"PCIe gen {gen} has no ceiling here; add it to LANE_GBS")
    return LANE_GBS[gen] * width


def bdp_bytes(latency_ns: float, gbs: float) -> int:
    """Bytes that must be in flight to keep a `gbs` link busy at `latency_ns` per read (ns x GB/s = bytes)."""
    return round(latency_ns * gbs)


def knee(points: list[tuple[int, float]], fraction: float = KNEE_FRACTION) -> int | None:
    """The fewest bytes in flight whose GB/s reaches `fraction` of the best point's."""
    if not points:
        return None
    best = max(gbs for _, gbs in points)
    return min(in_flight for in_flight, gbs in points if gbs >= fraction * best)


def above_ceiling(cells: list[dict], ceiling: float) -> list[dict]:
    """Cells faster than the link allows: L2-resident or mistimed, never a result."""
    return [cell for cell in cells if cell["gbs"] > ceiling]


def summarize(records: list[dict]) -> dict:
    metas = [r for r in records if r.get("meta")]
    if len(metas) != 1:
        raise ValueError(f"expected exactly one meta record, found {len(metas)}")
    meta = metas[0]
    cells = [r for r in records if r.get("kind") == "cell"]
    ceiling = link_gbs(meta["pcie_gen"], meta["pcie_width"])
    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for cell in cells:
        by_method[cell["method"]].append((cell["in_flight"], cell["gbs"]))
    measured = max((gbs for points in by_method.values() for _, gbs in points), default=0.0)
    latency = next((r["serial_acquire_ns"] for r in records if r.get("kind") == "latency"), None)
    rtt = next((r["rtt_ns_p50"] for r in records if r.get("kind") == "pingpong"), None)
    fresh = {r["method"]: r["fresh"] for r in records if r.get("kind") == "fresh"}
    return {
        "host": meta["host"],
        "theoretical_gbs": round(ceiling, 2),
        "measured_ceiling_gbs": measured,
        "above_ceiling": above_ceiling(cells, ceiling),
        "serial_acquire_ns": latency,
        "flag_rtt_ns": rtt,
        "bdp_bytes": bdp_bytes(latency, measured) if latency else None,
        "unsafe": sorted(m for m, ok in fresh.items() if m != CONTROL and not ok),
        "control_blind": bool(fresh.get(CONTROL, False)),
        "methods": {
            method: {
                "best_gbs": max(gbs for _, gbs in points),
                "knee_bytes": None if method.startswith("ce") else knee(points),
                "share_of_measured": round(max(gbs for _, gbs in points) / measured, 3) if measured else None,
            }
            for method, points in sorted(by_method.items())
        },
        "named": {r["name"]: r["gbs"] for r in cells if r.get("name")},
    }


def main() -> int:
    records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    s = summarize(records)
    print(f"# {s['host']}: theoretical {s['theoretical_gbs']} GB/s, measured ceiling {s['measured_ceiling_gbs']} GB/s")
    print(f"serial acquire {s['serial_acquire_ns']} ns, flag RTT p50 {s['flag_rtt_ns']} ns, BDP {s['bdp_bytes']} B\n")
    print("| method | best GB/s | knee (bytes in flight) | share of measured |\n|---|---:|---:|---:|")
    for method, m in s["methods"].items():
        print(f"| {method} | {m['best_gbs']} | {m['knee_bytes']} | {m['share_of_measured']} |")
    print(f"\nnamed cells: {s['named']}")
    bad = False
    if s["above_ceiling"]:
        print(f"ABOVE CEILING: {s['above_ceiling']}")
        bad = True
    if s["unsafe"]:
        print(f"UNSAFE (failed the fresh check): {s['unsafe']}")
        bad = True
    if s["control_blind"]:
        print("FRESH CHECK BLIND: the .nc control read fresh bytes, so no fresh verdict above means anything")
        bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python3 -m pytest analysis/dsv41-drive/copy-mechanism/mech_report_test.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add analysis/dsv41-drive/copy-mechanism/mech_report.py analysis/dsv41-drive/copy-mechanism/mech_report_test.py
git commit -m "analysis(copy-mechanism): report module for the portable copy sweep (ceilings, knees, refusals)"
```

### Task 2: Capability probe (256-bit loads, LDGSTS, TMA from host memory, batch memcpy)

**Files:**
- Create: `analysis/dsv41-drive/copy-mechanism/probe.cuh`
- Create: `analysis/dsv41-drive/copy-mechanism/probe.py`
- Create (divix01 run output, committed): `analysis/dsv41-drive/copy-mechanism/gen3/probe.json`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `probe.json`, shaped
  `{"host": str, "gpu": str, "cuda": str, "probes": {name: {"build": bool, "ok": bool, "load_opcodes": [str], "error": str | None}}}`
  with `name` in `v8_weak, v8_cv, ldgsts, bulk, batch`. Tasks 3-5 read `probes[name]["ok"]` to decide which methods
  to build and sweep. Nothing is assumed: a method whose probe is not `ok` is not built.

- [ ] **Step 1: Write `probe.cuh`**

```cpp
// Capability probes for the copy-mechanism sweep. probe.py builds each probe into a module of its own (one -DPROBE_*
// per build), so an instruction the toolchain rejects fails that probe's build only, and runs each in a process of
// its own, so a fault in one (an illegal address from a bulk read of host memory, say) cannot poison the others.
// Every probe reads a pinned host buffer and writes device memory; probe.py compares the bytes.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

__device__ __forceinline__ uint32_t probe_smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

#if defined(PROBE_V8_WEAK) || defined(PROBE_V8_CV)
// One 32-byte load per thread, stored back as two 16-byte stores so only the load's width is under test.
__global__ void probe_kernel(const uint8_t* src, uint8_t* dst) {
  const uint8_t* s = src + 32 * threadIdx.x;
  uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
#ifdef PROBE_V8_CV
  asm volatile("ld.global.cv.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3), "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
               : "l"(s)
               : "memory");
#else
  asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3), "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
               : "l"(s)
               : "memory");
#endif
  uint8_t* d = dst + 32 * threadIdx.x;
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d + 16), "r"(v4), "r"(v5), "r"(v6), "r"(v7) : "memory");
}
constexpr int kProbeThreads = 256;  // 8 KiB
#endif

#ifdef PROBE_LDGSTS
__global__ void probe_kernel(const uint8_t* src, uint8_t* dst) {
  __shared__ alignas(16) uint8_t buf[256 * 16];
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(probe_smem(buf + 16 * threadIdx.x)),
               "l"(src + 16 * threadIdx.x)
               : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  reinterpret_cast<uint4*>(dst)[threadIdx.x] = reinterpret_cast<const uint4*>(buf)[threadIdx.x];
}
constexpr int kProbeThreads = 256;  // 4 KiB
#endif

#ifdef PROBE_BULK
__global__ void probe_kernel(const uint8_t* src, uint8_t* dst) {
  constexpr uint32_t kBytes = 4096;
  __shared__ alignas(128) uint8_t buf[kBytes];
  __shared__ alignas(8) uint64_t bar;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(probe_smem(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // The contract's shape: a generic-proxy view of host bytes handed to the async proxy.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(probe_smem(&bar)), "r"(kBytes)
                 : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     probe_smem(buf)),
                 "l"(src), "r"(kBytes), "r"(probe_smem(&bar))
                 : "memory");
  }
  asm volatile(
      "{\n .reg .pred done;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0;\n @!done bra WAIT;\n}\n" ::"r"(
          probe_smem(&bar))
      : "memory");
  reinterpret_cast<uint4*>(dst)[threadIdx.x] = reinterpret_cast<const uint4*>(buf)[threadIdx.x];
}
constexpr int kProbeThreads = 256;  // 4 KiB
#endif

#ifdef PROBE_BATCH
// Host API only: four 1 KiB host-to-device copies in one cudaMemcpyBatchAsync (CUDA 13.4 signature, no failIdx).
void probe_run(tvm::ffi::TensorView src, tvm::ffi::TensorView dst) {
  void* dsts[4];
  const void* srcs[4];
  size_t sizes[4];
  for (int i = 0; i < 4; ++i) {
    dsts[i] = static_cast<uint8_t*>(dst.data_ptr()) + 1024 * i;
    srcs[i] = static_cast<const uint8_t*>(src.data_ptr()) + 1024 * i;
    sizes[i] = 1024;
  }
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attr_index = 0;
  const auto stream = host::LaunchKernel::resolve_device(dst.device());
  CHECK_CUDA(cudaMemcpyBatchAsync(dsts, srcs, sizes, 4, &attr, &attr_index, 1, stream)) << "cudaMemcpyBatchAsync";
}
#else
void probe_run(tvm::ffi::TensorView src, tvm::ffi::TensorView dst) {
  const auto stream = host::LaunchKernel::resolve_device(dst.device());
  host::LaunchKernel(1, kProbeThreads, stream)(
      probe_kernel, static_cast<const uint8_t*>(src.data_ptr()), static_cast<uint8_t*>(dst.data_ptr()));
}
#endif

}  // namespace sglang
```

- [ ] **Step 2: Write `probe.py`**

```python
#!/usr/bin/env python3
"""Which copy mechanisms exist on this host: builds and runs each probe of probe.cuh in a process of its own.

    PYTHONPATH=<repo>/python python probe.py --repo <repo> --out probe.json

A probe is `ok` when it built, ran, and copied the right bytes. `load_opcodes` are the SASS load opcodes of its
module, so a 32-byte PTX load that ptxas split into two 16-byte ones shows as such.
"""
import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PROBES = {"v8_weak": "PROBE_V8_WEAK", "v8_cv": "PROBE_V8_CV", "ldgsts": "PROBE_LDGSTS", "bulk": "PROBE_BULK",
          "batch": "PROBE_BATCH"}
BYTES = {"v8_weak": 8192, "v8_cv": 8192, "ldgsts": 4096, "bulk": 4096, "batch": 4096}
LOADS = ("LDG", "LDGSTS", "UBLKCP", "UTMA")


def child(repo: pathlib.Path, name: str) -> dict:
    import torch

    import sglang

    if not str(pathlib.Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    from sglang.kernels.jit.utils.compile.loader import load_jit

    try:
        mod = load_jit(
            "copy_mech_probe", name,
            cuda_files=[str(HERE / "probe.cuh")],
            cuda_wrappers=[("probe_run", "probe_run")],
            extra_cuda_cflags=[f"-D{PROBES[name]}"],
        )
    except Exception as exc:  # a rejected instruction is a result, not a crash
        return {"build": False, "ok": False, "load_opcodes": [], "error": str(exc)[-600:]}
    n = BYTES[name]
    src = (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8).pin_memory()
    dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
    mod.probe_run(src, dst)
    torch.cuda.synchronize()
    return {"build": True, "ok": bool(torch.equal(dst.cpu(), src)), "load_opcodes": opcodes(), "error": None}


def opcodes() -> list[str]:
    cuobjdump = shutil.which("cuobjdump") or os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin/cuobjdump")
    cache = os.environ["SGLANG_JIT_CACHE_DIR"]
    found = set()
    for so in pathlib.Path(cache).rglob("*.so"):
        sass = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True).stdout
        for line in sass.splitlines():
            parts = line.split("*/")
            if len(parts) < 2:
                continue
            op = parts[1].strip().lstrip("@!P0123456789 ").split(" ")[0]
            if op.startswith(LOADS):
                found.add(op)
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out")
    ap.add_argument("--child")
    a = ap.parse_args()
    repo = pathlib.Path(a.repo).resolve()
    if a.child:
        print(json.dumps(child(repo, a.child)), flush=True)
        return 0
    import torch

    results = {}
    for name in PROBES:
        cache = tempfile.mkdtemp(prefix=f"probe-{name}-", dir=HERE)
        env = dict(os.environ, SGLANG_JIT_CACHE_DIR=cache)
        run = subprocess.run([sys.executable, __file__, "--repo", str(repo), "--child", name], env=env,
                             capture_output=True, text=True, timeout=900)
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        if run.returncode == 0 and lines:
            results[name] = json.loads(lines[-1])
        else:
            results[name] = {"build": None, "ok": False, "load_opcodes": [],
                             "error": f"exit {run.returncode}: {run.stderr[-600:]}"}
        shutil.rmtree(cache, ignore_errors=True)
        print(name, results[name], flush=True)
    out = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
           "probes": results}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Commit, push, and run the probe on divix01**

```bash
git add analysis/dsv41-drive/copy-mechanism/probe.cuh analysis/dsv41-drive/copy-mechanism/probe.py
git commit -m "analysis(copy-mechanism): capability probe for 256-bit loads, LDGSTS, TMA from host memory, batch memcpy"
git push origin expert-stream-transfer-measurement
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && mkdir -p analysis/dsv41-drive/copy-mechanism/gen3 \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/copy-mechanism/probe.py --repo $PWD \
     --out analysis/dsv41-drive/copy-mechanism/gen3/probe.json; echo EXIT=$?'
```

Expected: `EXIT=0` and five lines, one per probe. Whether each probe is `ok` is the finding, so no probe's outcome
is expected in advance. `batch` should be `ok` on CUDA 13.4 (the header declares it). If `bulk` faults, the parent
records it with `ok: false` and carries on. That is the "TMA from host memory does not work" answer.

- [ ] **Step 4: Copy `probe.json` back, commit it**

```bash
scp divix01:/data/models/slang/nvfp4-work/wt-xfer/analysis/dsv41-drive/copy-mechanism/gen3/probe.json \
  analysis/dsv41-drive/copy-mechanism/gen3/probe.json
git add analysis/dsv41-drive/copy-mechanism/gen3/probe.json
git commit -m "analysis(copy-mechanism): divix01 Gen3 capability probe results"
```

(This `scp` copies a small result file back to the laptop. The run protocol forbids copying a working tree to
divix01; copying results back is not that.)

### Task 3: SM load sweep, read latency, flag round trip, fresh check

**Files:**
- Create: `analysis/dsv41-drive/copy-mechanism/mech_bench.cuh`
- Create: `analysis/dsv41-drive/copy-mechanism/mech_bench.py`

**Interfaces:**
- Consumes: `gen3/probe.json` shape (Task 2); `mech_report.summarize` (Task 1).
- Produces (FFI, all in namespace `sglang`):
  - `mech_copy(jobs: int64[n,3] cuda, kind: int, grid: int, a: int, b: int) -> None`, where kind 0 = `.cv`
    (a = unroll U in {1,2,4,8,16}, b = width 16|32) and kind 1 = `.nc` control (a = 1, b = 16);
  - `mech_fresh(jobs, words: int32[2] pinned, src: uint8 pinned, pattern: uint8 cpu, kind, grid, a, b) -> None`;
  - `mech_latency(host_word: int32[1] pinned, dev_word: int32[1] cuda, out: int64[3] cuda, n: int) -> None`;
  - `mech_pingpong(words: int32[2] pinned, out: int64[rounds] cuda, rounds: int) -> None`.
- Produces (Python): the `Cell` namedtuple `(method, grid, a, b, in_flight, name, rows, which, run)` and the
  generator registry `CELL_GENERATORS: list[Callable[[mod, probe], Iterable[Cell]]]` plus `FRESH_CHECKS:
  list[tuple[str, int, int, int, int, str]]` of `(method, kind, grid, a, b, probe_or_empty)`. Tasks 4 and 5 append
  to both lists. `KIND_*` constants.

- [ ] **Step 1: Write `mech_bench.cuh` (SM loads, latency, ping-pong, fresh check)**

```cpp
// Copy-mechanism sweep kernels. A job is {source, destination, bytes}, every field a multiple of 16 (the six EXL3
// segment row sizes are multiples of 512). Every host read obeys the lease visibility contract (LEASE_PROTOCOL.md
// E1 amendment): ld.global.cv, or cp.async.cg issued after the acquire, or a bulk read after
// fence.proxy.async.global. The one exception is kind 1 (.nc), the fresh check's negative control, never swept.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace sglang {
namespace mech {

constexpr int kThreads = 256;
constexpr int kKindCv = 0;
constexpr int kKindNc = 1;
constexpr int kKindLdgsts = 2;
constexpr int kKindTma = 3;

struct Job {
  int64_t src, dst, bytes;
};

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The fresh check's midpoint. Every block counts itself in words[1], then its thread 0 polls words[0] with an
// acquire until the host has rewritten the source and released the word. The block barrier after the poll orders
// the other threads' second-pass reads after it.
__device__ void fresh_barrier(uint32_t* words) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd_system(words + 1, 1u);
    while (ld_acquire(words) == 0)
      __nanosleep(1000);
  }
  __syncthreads();
}

template <int W>
struct Vec {
  uint32_t v[W / 4];
};

template <int W, bool kNc>
__device__ __forceinline__ Vec<W> load(const uint8_t* p) {
  Vec<W> r;
  if constexpr (W == 16) {
    if constexpr (kNc) {
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    } else {
      asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    }
  } else {
#ifdef MECH_V8
    asm volatile("ld.global.cv.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]),
                   "=r"(r.v[7])
                 : "l"(p)
                 : "memory");
#endif
  }
  return r;
}

template <int W>
__device__ __forceinline__ void store(uint8_t* p, const Vec<W>& r) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(r.v[0]), "r"(r.v[1]), "r"(r.v[2]), "r"(r.v[3])
               : "memory");
  if constexpr (W == 32) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p + 16), "r"(r.v[4]), "r"(r.v[5]), "r"(r.v[6]),
                 "r"(r.v[7])
                 : "memory");
  }
}

// S's access pattern generalised: chunks of kThreads * U units dealt round-robin over the grid, U loads issued
// before any store. U = 1, W = 16, grid 8 is exactly stream_copy_slice (row_copy_kernels.cuh).
template <int U, int W, bool kNc>
__global__ __launch_bounds__(kThreads, 1) void sm_kernel(const Job* jobs, int64_t njobs, uint32_t* fresh) {
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) fresh_barrier(fresh);
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t units = jobs[j].bytes / W;
      for (int64_t chunk = blockIdx.x; chunk * kThreads * U < units; chunk += gridDim.x) {
        Vec<W> v[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
          if (u < units) v[k] = load<W, kNc>(src + W * u);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
          if (u < units) store<W>(dst + W * u, v[k]);
        }
      }
    }
  }
}

// One thread: n serial acquires of a pinned host word, then of a device word (the 711 ns / 117 ns figures).
__global__ void latency_kernel(const uint32_t* host_word, const uint32_t* dev_word, int64_t n, int64_t* out) {
  uint32_t sink = 0;
  const uint64_t t0 = now_ns();
  for (int64_t i = 0; i < n; ++i)
    sink += ld_acquire(host_word);
  const uint64_t t1 = now_ns();
  for (int64_t i = 0; i < n; ++i)
    sink += ld_acquire(dev_word);
  const uint64_t t2 = now_ns();
  out[0] = static_cast<int64_t>(t1 - t0);
  out[1] = static_cast<int64_t>(t2 - t1);
  out[2] = sink;
}

// The device half of a flag round trip: release ping, acquire-poll pong, per round. RTT/2 bounds one-way visibility.
__global__ void pingpong_kernel(uint32_t* ping, const uint32_t* pong, int64_t rounds, int64_t* out) {
  for (int64_t r = 1; r <= rounds; ++r) {
    const uint64_t t0 = now_ns();
    st_release(ping, static_cast<uint32_t>(r));
    while (ld_acquire(pong) != static_cast<uint32_t>(r)) {
    }
    out[r - 1] = static_cast<int64_t>(now_ns() - t0);
  }
}

inline void cpu_relax() {
#if defined(__x86_64__)
  _mm_pause();
#endif
}

// Every copy kernel of this file behind one switch; Tasks 4 (kinds 2, 3) extend it.
inline void launch(int64_t kind, const Job* jobs, int64_t njobs, int64_t grid, int64_t a, int64_t b, uint32_t* fresh,
                   cudaStream_t stream) {
  const int g = static_cast<int>(grid);
  if (kind == kKindNc) {
    host::RuntimeCheck(a == 1 && b == 16, "the .nc control is U=1, W=16 only");
    host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 16, true>, jobs, njobs, fresh);
    return;
  }
  host::RuntimeCheck(kind == kKindCv, "kind: 0 (.cv) or 1 (.nc control); 2 and 3 arrive in Task 4");
  if (b == 16) {
    switch (a) {
      case 1: host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 16, false>, jobs, njobs, fresh); return;
      case 2: host::LaunchKernel(g, kThreads, stream)(sm_kernel<2, 16, false>, jobs, njobs, fresh); return;
      case 4: host::LaunchKernel(g, kThreads, stream)(sm_kernel<4, 16, false>, jobs, njobs, fresh); return;
      case 8: host::LaunchKernel(g, kThreads, stream)(sm_kernel<8, 16, false>, jobs, njobs, fresh); return;
      case 16: host::LaunchKernel(g, kThreads, stream)(sm_kernel<16, 16, false>, jobs, njobs, fresh); return;
    }
  }
#ifdef MECH_V8
  if (b == 32) {
    switch (a) {
      case 1: host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 32, false>, jobs, njobs, fresh); return;
      case 2: host::LaunchKernel(g, kThreads, stream)(sm_kernel<2, 32, false>, jobs, njobs, fresh); return;
      case 4: host::LaunchKernel(g, kThreads, stream)(sm_kernel<4, 32, false>, jobs, njobs, fresh); return;
      case 8: host::LaunchKernel(g, kThreads, stream)(sm_kernel<8, 32, false>, jobs, njobs, fresh); return;
    }
  }
#endif
  host::RuntimeCheck(false, "no .cv kernel for this (unroll, width); width 32 needs the MECH_V8 build");
}

}  // namespace mech

void mech_copy(tvm::ffi::TensorView jobs, int64_t kind, int64_t grid, int64_t a, int64_t b) {
  const auto stream = host::LaunchKernel::resolve_device(jobs.device());
  mech::launch(kind, static_cast<const mech::Job*>(jobs.data_ptr()), jobs.size(0), grid, a, b, nullptr, stream);
}

// Launches `kind` with the fresh midpoint, waits for every block to reach it, rewrites the pinned source with
// `pattern`, releases the flag, and waits for the kernel. The caller checks that the destination holds `pattern`.
void mech_fresh(tvm::ffi::TensorView jobs, tvm::ffi::TensorView words, tvm::ffi::TensorView src,
                tvm::ffi::TensorView pattern, int64_t kind, int64_t grid, int64_t a, int64_t b) {
  auto* w = static_cast<uint32_t*>(words.data_ptr());
  __atomic_store_n(w, 0u, __ATOMIC_RELEASE);
  __atomic_store_n(w + 1, 0u, __ATOMIC_RELEASE);
  const auto stream = host::LaunchKernel::resolve_device(jobs.device());
  mech::launch(kind, static_cast<const mech::Job*>(jobs.data_ptr()), jobs.size(0), grid, a, b, w, stream);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (__atomic_load_n(w + 1, __ATOMIC_ACQUIRE) != static_cast<uint32_t>(grid)) {
    mech::cpu_relax();
    host::RuntimeCheck(std::chrono::steady_clock::now() < deadline, "fresh check: blocks never reached the midpoint");
  }
  std::memcpy(src.data_ptr(), pattern.data_ptr(), static_cast<size_t>(src.size(0)));
  __atomic_store_n(w, 1u, __ATOMIC_RELEASE);
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "fresh check";
}

void mech_latency(tvm::ffi::TensorView host_word, tvm::ffi::TensorView dev_word, tvm::ffi::TensorView out, int64_t n) {
  const auto stream = host::LaunchKernel::resolve_device(out.device());
  host::LaunchKernel(1, 1, stream)(mech::latency_kernel, static_cast<const uint32_t*>(host_word.data_ptr()),
                                   static_cast<const uint32_t*>(dev_word.data_ptr()), n,
                                   static_cast<int64_t*>(out.data_ptr()));
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "latency";
}

void mech_pingpong(tvm::ffi::TensorView words, tvm::ffi::TensorView out, int64_t rounds) {
  auto* w = static_cast<uint32_t*>(words.data_ptr());
  __atomic_store_n(w, 0u, __ATOMIC_RELEASE);
  __atomic_store_n(w + 1, 0u, __ATOMIC_RELEASE);
  const auto stream = host::LaunchKernel::resolve_device(out.device());
  host::LaunchKernel(1, 1, stream)(mech::pingpong_kernel, w, w + 1, rounds, static_cast<int64_t*>(out.data_ptr()));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (uint32_t r = 1; r <= static_cast<uint32_t>(rounds); ++r) {
    while (__atomic_load_n(w, __ATOMIC_ACQUIRE) != r) {
      mech::cpu_relax();
      host::RuntimeCheck(std::chrono::steady_clock::now() < deadline, "ping-pong: the kernel stopped answering");
    }
    __atomic_store_n(w + 1, r, __ATOMIC_RELEASE);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "ping-pong";
}

}  // namespace sglang
```

- [ ] **Step 2: Write `mech_bench.py`**

```python
#!/usr/bin/env python3
"""GB/s against bytes in flight for every way to move pinned host bytes into VRAM, on this host's link.

One command per host. The meta record carries the host's PCIe generation and width, so runs on different links
compare directly:

    PYTHONPATH=<repo>/python python mech_bench.py --repo <repo> --probe probe.json --out <host>.jsonl
    python3 mech_report.py <host>.jsonl

Methods (probe.json decides which exist here):
  sm_cv16 / sm_cv32   ld.global.cv, 16- or 32-byte accesses, U in flight per thread, G blocks of 256
                      (s_pattern = S: G8 U1 W16; cw_pattern = the copy wait: G1 U4 W16)
  ldgsts, tma         Task 4;  ce_each, ce_batch and the *_small workloads: Task 5
Every method first passes a byte check; the fresh check (kind "fresh" records) runs with the .nc control.
"""
import argparse
import collections
import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SEGMENTS = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)
ALL_SEGMENTS = tuple(range(len(SEGMENTS)))
ROWS = 160  # 2.1 GB of pinned source rows: no launch reuses a row within 40 launches
SLOTS = 16  # 213 MB of destination slots, rotated: past the 96 MiB L2
N_ROWS = 4
FRESH_BYTES = 64 << 10  # small enough that one block's share fits its L1, so a stale .nc read shows
KIND_CV, KIND_NC, KIND_LDGSTS, KIND_TMA = 0, 1, 2, 3
WRAPPERS = ["mech_copy", "mech_fresh", "mech_latency", "mech_pingpong"]
BUILD_FLAGS = []  # (probe name, define) pairs; Task 4 and 5 append theirs
Cell = collections.namedtuple("Cell", "method grid a b in_flight name rows which run")


def probe_ok(probe: dict, name: str) -> bool:
    return bool(probe["probes"].get(name, {}).get("ok"))


def load(repo: Path, probe: dict):
    from sglang.kernels.jit.utils.compile.loader import load_jit

    flags = ["-DMECH_V8"] if probe_ok(probe, "v8_cv") else []
    flags += [f"-D{define}" for name, define in BUILD_FLAGS if probe_ok(probe, name)]
    variant = "-".join(sorted(f[2:] for f in flags)) or "base"
    return load_jit("copy_mech_bench", variant, cuda_files=[str(HERE / "mech_bench.cuh")],
                    cuda_wrappers=[(n, n) for n in WRAPPERS], extra_cuda_cflags=flags)


class Rows:
    """Pinned source rows (the six real EXL3 segments) and rotated VRAM destination slots."""

    def __init__(self, node: int):
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

        self.src = [allocate_host_slab(ROWS, (b,), torch.uint8, register=True,
                                       placement=((node, ROWS * b),) if node >= 0 else ()) for b in SEGMENTS]
        for s in self.src:
            for r in range(ROWS):
                s[r, :512].fill_(r % 251)
        self.dst = [torch.zeros((SLOTS, b), dtype=torch.uint8, device="cuda") for b in SEGMENTS]
        self.row = 0
        self.slot = 0

    def jobs(self, n: int, which=ALL_SEGMENTS):
        pairs = []
        for _ in range(n):
            pairs.append((self.row, self.slot))
            self.row, self.slot = (self.row + 1) % ROWS, (self.slot + 1) % SLOTS
        table = [[self.src[k][row].data_ptr(), self.dst[k][slot].data_ptr(), SEGMENTS[k]]
                 for row, slot in pairs for k in which]
        return pairs, torch.tensor(table, dtype=torch.int64)

    def check(self, pairs, which) -> bool:
        return all(torch.equal(self.dst[k][slot].cpu(), self.src[k][row]) for row, slot in pairs for k in which)


def sm_cells(mod, probe):
    widths = [16] + ([32] if probe_ok(probe, "v8_cv") else [])
    for w in widths:
        for grid in (1, 2, 4, 8, 16, 32):
            for unroll in (1, 2, 4, 8, 16):
                if w == 32 and unroll == 16:
                    continue
                name = {(16, 8, 1): "s_pattern", (16, 1, 4): "cw_pattern"}.get((w, grid, unroll))
                yield Cell(f"sm_cv{w}", grid, unroll, w, grid * 256 * unroll * w, name, N_ROWS, ALL_SEGMENTS,
                           lambda cpu, dev, g=grid, u=unroll, w=w: mod.mech_copy(dev, KIND_CV, g, u, w))


CELL_GENERATORS = [sm_cells]
# (method, kind, grid, a, b, probe that must be ok, or "")
FRESH_CHECKS = [("sm_cv16", KIND_CV, 8, 1, 16, ""), ("sm_cv32", KIND_CV, 8, 1, 32, "v8_cv"),
                ("nc_control", KIND_NC, 8, 1, 16, "")]


def measure(cell: Cell, rows: Rows, reps: int) -> dict:
    times, host_ns = [], []
    for i in range(reps + 3):
        pairs, table = rows.jobs(cell.rows, cell.which)
        dev = table.to("cuda")
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        h = cell.run(table, dev)
        e1.record()
        torch.cuda.synchronize()
        if i == 0 and not rows.check(pairs, cell.which):
            raise SystemExit(f"{cell.method} g{cell.grid} a{cell.a} b{cell.b} copied the wrong bytes")
        if i >= 3:
            times.append(e0.elapsed_time(e1))
            host_ns.append(h)
    ms = statistics.median(times)
    nbytes = cell.rows * sum(SEGMENTS[k] for k in cell.which)
    return {"kind": "cell", "method": cell.method, "name": cell.name, "grid": cell.grid, "a": cell.a, "b": cell.b,
            "in_flight": cell.in_flight, "rows": cell.rows, "bytes": nbytes, "ms_p50": round(ms, 4),
            "gbs": round(nbytes / (ms * 1e-3) / 1e9, 3),
            "host_ns_p50": statistics.median(host_ns) if host_ns[0] is not None else None}


def fresh(mod, kind, grid, a, b) -> bool:
    src = torch.full((FRESH_BYTES,), 0x11, dtype=torch.uint8).pin_memory()
    pattern = torch.full((FRESH_BYTES,), 0x22, dtype=torch.uint8)
    dst = torch.zeros(FRESH_BYTES, dtype=torch.uint8, device="cuda")
    words = torch.zeros(2, dtype=torch.int32).pin_memory()
    jobs = torch.tensor([[src.data_ptr(), dst.data_ptr(), FRESH_BYTES]], dtype=torch.int64, device="cuda")
    mod.mech_fresh(jobs, words, src, pattern, kind, grid, a, b)
    torch.cuda.synchronize()
    return bool((dst.cpu() == 0x22).all())


def pcie(index: int) -> tuple[int, int, int, int]:
    q = "pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-i", str(index)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return tuple(int(v) for v in out.split(", "))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", type=int, default=0, help="NUMA node of the source rows; -1 for no binding")
    ap.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi index of the GPU under test")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    probe = json.loads(Path(a.probe).read_text())
    mod = load(repo, probe)
    rows = Rows(a.node)
    gen, width, gen_now, width_now = pcie(a.gpu_index)
    out = open(a.out, "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")

    emit({"meta": True, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "pcie_gen": gen,
          "pcie_width": width, "pcie_gen_at_start": gen_now, "pcie_width_at_start": width_now,
          "cuda": torch.version.cuda, "node": a.node, "sglang": sglang.__file__,
          "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "probe": {k: v["ok"] for k, v in probe["probes"].items()}})
    host_word = torch.zeros(1, dtype=torch.int32).pin_memory()
    dev_word = torch.zeros(1, dtype=torch.int32, device="cuda")
    lat = torch.zeros(3, dtype=torch.int64, device="cuda")
    n = 1000
    samples = []
    for _ in range(5):
        mod.mech_latency(host_word, dev_word, lat, n)
        samples.append(lat.cpu().tolist())
    emit({"kind": "latency", "serial_acquire_ns": statistics.median(s[0] for s in samples) / n,
          "device_acquire_ns": statistics.median(s[1] for s in samples) / n})
    words = torch.zeros(2, dtype=torch.int32).pin_memory()
    rtt = torch.zeros(256, dtype=torch.int64, device="cuda")
    mod.mech_pingpong(words, rtt, 256)
    r = rtt.cpu().tolist()[16:]  # the first rounds include the kernel's start
    emit({"kind": "pingpong", "rtt_ns_p50": int(statistics.median(r)), "rtt_ns_min": int(min(r)), "rounds": len(r)})
    for method, kind, grid, ua, ub, need in FRESH_CHECKS:
        if need and not probe_ok(probe, need):
            continue
        emit({"kind": "fresh", "method": method, "fresh": fresh(mod, kind, grid, ua, ub)})
    for generator in CELL_GENERATORS:
        for cell in generator(mod, probe):
            emit(measure(cell, rows, a.reps))
    out.close()
    sys.path.insert(0, str(HERE))
    import mech_report

    records = [json.loads(line) for line in open(a.out) if line.strip()]
    # One file per run: a second meta in the file (an appended rerun) is refused by summarize.
    s = mech_report.summarize(records)
    print(json.dumps({k: s[k] for k in ("measured_ceiling_gbs", "theoretical_gbs", "bdp_bytes", "unsafe",
                                        "control_blind")}))
    return 1 if s["above_ceiling"] or s["unsafe"] or s["control_blind"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Commit, push, run a short smoke on divix01**

```bash
git add analysis/dsv41-drive/copy-mechanism/mech_bench.cuh analysis/dsv41-drive/copy-mechanism/mech_bench.py
git commit -m "analysis(copy-mechanism): SM .cv load sweep, read latency, flag round trip and fresh check"
git push origin expert-stream-transfer-measurement
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && rm -f /data/models/slang/nvfp4-work/xfer-smoke.jsonl \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/copy-mechanism/mech_bench.py --repo $PWD \
     --probe analysis/dsv41-drive/copy-mechanism/gen3/probe.json --out /data/models/slang/nvfp4-work/xfer-smoke.jsonl --reps 5 \
     2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=0`. The fresh records read `sm_cv16 fresh: true` and `nc_control fresh: false`.
`s_pattern` is 12.0-12.4 GB/s. `serial_acquire_ns` is within 600-850 (task6-microbench recorded 711). No cell is
above 15.75.

- If `nc_control` reads `true`, the check is blind. Stop and report. Do not tune `FRESH_BYTES` until it comes out
  stale: that would be fitting the check to the result.
- If the run exits 1 on an above-ceiling cell, the destination or the timing is wrong. Stop and report the cell.

### Task 4: LDGSTS producer-warp ring and TMA bulk copies

**Files:**
- Modify: `analysis/dsv41-drive/copy-mechanism/mech_bench.cuh` (add both kernels; extend `mech::launch`)
- Modify: `analysis/dsv41-drive/copy-mechanism/mech_bench.py` (register cells and fresh checks)

**Interfaces:**
- Consumes: `mech::launch`, `Job`, `fresh_barrier`, `smem`, `kThreads`, `kKindLdgsts`, `kKindTma` (Task 3);
  `CELL_GENERATORS`, `FRESH_CHECKS`, `BUILD_FLAGS`, `Cell`, `probe_ok` (Task 3); probes `ldgsts`, `bulk` (Task 2).
- Produces: `mech_copy` kinds 2 (a = stages in {2,4,8}, b = 0) and 3 (a = stages in {2,4,8}, b = chunk bytes in
  {4096, 16384}). Methods `ldgsts` and `tma` are in the JSONL.

- [ ] **Step 1: Add the two kernels to `mech_bench.cuh`, inside `namespace mech` just before `cpu_relax`**

```cpp
#if defined(MECH_LDGSTS) || defined(MECH_BULK)
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred done;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n @!done bra WAIT;\n}\n" ::"r"(
          smem(bar)),
      "r"(parity)
      : "memory");
}
#endif

#ifdef MECH_LDGSTS
constexpr int kLdgstsPer = 8;                       // 16-byte units per producer lane per stage
constexpr int kLdgstsUnits = 32 * kLdgstsPer;       // units per stage
constexpr int kLdgstsStage = 16 * kLdgstsUnits;     // 4 KiB
constexpr int kConsumers = kThreads - 32;

// Warp 0 produces: each lane issues its cp.async.cg reads of a stage (L2 only, never L1, so no stale L1 line), then
// arrives on full[slot] when they land (cp.async.mbarrier.arrive.noinc). Warps 1-7 consume: wait full, store the
// stage to the destination, arrive on empty[slot]. In flight per block: at most STAGES * 4 KiB.
template <int STAGES>
__global__ __launch_bounds__(kThreads, 1) void ldgsts_kernel(const Job* jobs, int64_t njobs, uint32_t* fresh) {
  __shared__ alignas(128) uint8_t ring[STAGES][kLdgstsStage];
  __shared__ alignas(8) uint64_t full[STAGES];
  __shared__ alignas(8) uint64_t empty[STAGES];
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 32);
      mbar_init(&empty[s], kConsumers);
    }
  }
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  int64_t local = 0;  // stages this block has used, across passes and jobs: ring position and parity
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) fresh_barrier(fresh);
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t units = jobs[j].bytes / 16;
      const int64_t stages = (units + kLdgstsUnits - 1) / kLdgstsUnits;
      for (int64_t g = blockIdx.x; g < stages; g += gridDim.x, ++local) {
        const int slot = static_cast<int>(local % STAGES);
        const uint32_t phase = static_cast<uint32_t>((local / STAGES) & 1);
        const int64_t base = g * kLdgstsUnits;
        if (warp == 0) {
          if (local >= STAGES) mbar_wait(&empty[slot], phase ^ 1u);
          for (int k = 0; k < kLdgstsPer; ++k) {
            const int64_t u = base + k * 32 + lane;
            if (u < units) {
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem(ring[slot] + 16 * (k * 32 + lane))),
                           "l"(src + 16 * u)
                           : "memory");
            }
          }
          asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem(&full[slot])) : "memory");
        } else {
          mbar_wait(&full[slot], phase);
          for (int i = threadIdx.x - 32; i < kLdgstsUnits; i += kConsumers) {
            const int64_t u = base + i;
            if (u < units) {
              const uint4 v = *reinterpret_cast<const uint4*>(ring[slot] + 16 * i);
              asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(dst + 16 * u), "r"(v.x), "r"(v.y),
                           "r"(v.z), "r"(v.w)
                           : "memory");
            }
          }
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem(&empty[slot])) : "memory");
        }
      }
    }
  }
}
#endif

#ifdef MECH_BULK
// One thread per block. For each of its chunks: bulk-read host -> shared (async proxy, completion on full[slot]),
// then bulk-write shared -> VRAM. The producer runs STAGES - 1 chunks ahead of the consumer; before reusing a slot it
// waits (wait_group.read 0) for every bulk write to have read its shared source. In flight per block: <= STAGES *
// chunk. The fence.proxy.async.global after the fresh acquire is the contract for async-proxy reads of host bytes.
template <int STAGES>
__global__ __launch_bounds__(1, 1) void tma_kernel(const Job* jobs, int64_t njobs, int64_t chunk, uint32_t* fresh) {
  extern __shared__ __align__(128) uint8_t tma_ring[];
  __shared__ alignas(8) uint64_t full[STAGES];
  for (int s = 0; s < STAGES; ++s)
    mbar_init(&full[s], 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  int64_t local = 0;
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) {
      fresh_barrier(fresh);
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t bytes = jobs[j].bytes;
      const int64_t chunks = (bytes + chunk - 1) / chunk;
      const int64_t mine = chunks > blockIdx.x ? (chunks - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
      for (int64_t k = 0; k < mine + STAGES - 1; ++k) {
        if (k < mine) {
          const int64_t pos = local + k;
          const int slot = static_cast<int>(pos % STAGES);
          const int64_t off = (blockIdx.x + k * gridDim.x) * chunk;
          const uint32_t n = static_cast<uint32_t>(min(chunk, bytes - off));
          if (pos >= STAGES) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem(&full[slot])), "r"(n)
                       : "memory");
          asm volatile(
              "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                  smem(tma_ring + slot * chunk)),
              "l"(src + off), "r"(n), "r"(smem(&full[slot]))
              : "memory");
        }
        const int64_t c = k - (STAGES - 1);
        if (c >= 0) {
          const int64_t pos = local + c;
          const int slot = static_cast<int>(pos % STAGES);
          const int64_t off = (blockIdx.x + c * gridDim.x) * chunk;
          const uint32_t n = static_cast<uint32_t>(min(chunk, bytes - off));
          mbar_wait(&full[slot], static_cast<uint32_t>((pos / STAGES) & 1));
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst + off),
                       "r"(smem(tma_ring + slot * chunk)), "r"(n)
                       : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
      local += mine;
    }
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
#endif
```

- [ ] **Step 2: Extend `mech::launch` in `mech_bench.cuh`**

Replace the line
`host::RuntimeCheck(kind == kKindCv, "kind: 0 (.cv) or 1 (.nc control); 2 and 3 arrive in Task 4");` with:

```cpp
  if (kind == kKindLdgsts) {
#ifdef MECH_LDGSTS
    switch (a) {
      case 2: host::LaunchKernel(g, kThreads, stream)(ldgsts_kernel<2>, jobs, njobs, fresh); return;
      case 4: host::LaunchKernel(g, kThreads, stream)(ldgsts_kernel<4>, jobs, njobs, fresh); return;
      case 8: host::LaunchKernel(g, kThreads, stream)(ldgsts_kernel<8>, jobs, njobs, fresh); return;
    }
#endif
    host::RuntimeCheck(false, "ldgsts: stages 2, 4 or 8, in the MECH_LDGSTS build");
  }
  if (kind == kKindTma) {
#ifdef MECH_BULK
    host::RuntimeCheck(b % 16 == 0 && a * b <= 64 * 1024, "tma: chunk a multiple of 16, stages * chunk <= 64 KiB");
    const size_t shared = static_cast<size_t>(a * b);
    switch (a) {
      case 2:
        CHECK_CUDA(cudaFuncSetAttribute(tma_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        host::LaunchKernel(g, 1, stream, shared)(tma_kernel<2>, jobs, njobs, b, fresh);
        return;
      case 4:
        CHECK_CUDA(cudaFuncSetAttribute(tma_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        host::LaunchKernel(g, 1, stream, shared)(tma_kernel<4>, jobs, njobs, b, fresh);
        return;
      case 8:
        CHECK_CUDA(cudaFuncSetAttribute(tma_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        host::LaunchKernel(g, 1, stream, shared)(tma_kernel<8>, jobs, njobs, b, fresh);
        return;
    }
#endif
    host::RuntimeCheck(false, "tma: stages 2, 4 or 8, in the MECH_BULK build");
  }
  host::RuntimeCheck(kind == kKindCv, "kind: 0 (.cv), 1 (.nc control), 2 (ldgsts) or 3 (tma)");
```

- [ ] **Step 3: Register the cells and fresh checks in `mech_bench.py`**

After `CELL_GENERATORS = [sm_cells]` and the `FRESH_CHECKS` list, add:

```python
BUILD_FLAGS += [("ldgsts", "MECH_LDGSTS"), ("bulk", "MECH_BULK")]


def ldgsts_cells(mod, probe):
    if not probe_ok(probe, "ldgsts"):
        return
    for grid in (1, 2, 4, 8, 16):
        for stages in (2, 4, 8):
            yield Cell("ldgsts", grid, stages, 0, grid * stages * 4096, None, N_ROWS, ALL_SEGMENTS,
                       lambda cpu, dev, g=grid, s=stages: mod.mech_copy(dev, KIND_LDGSTS, g, s, 0))


def tma_cells(mod, probe):
    if not probe_ok(probe, "bulk"):
        return
    for grid in (1, 2, 4, 8, 16):
        for chunk, stages in ((4096, 2), (4096, 4), (4096, 8), (16384, 2), (16384, 4)):
            yield Cell("tma", grid, stages, chunk, grid * stages * chunk, None, N_ROWS, ALL_SEGMENTS,
                       lambda cpu, dev, g=grid, s=stages, c=chunk: mod.mech_copy(dev, KIND_TMA, g, s, c))


CELL_GENERATORS += [ldgsts_cells, tma_cells]
FRESH_CHECKS += [("ldgsts", KIND_LDGSTS, 8, 4, 0, "ldgsts"), ("tma", KIND_TMA, 8, 4, 4096, "bulk")]
```

`FRESH_BYTES` is 64 KiB with 8 blocks. The TMA chunking there (8 KiB per block, 4 KiB chunks) exercises the
two-chunk pipeline.

- [ ] **Step 4: Commit, push, smoke on divix01**

Run the Task 3 Step 3 commands with this commit message:
`analysis(copy-mechanism): cp.async producer-warp ring and TMA bulk copies, with fresh checks`.
Expected: `EXIT=0`, `ldgsts` and `tma` fresh `true` (when their probes are `ok`), `nc_control` `false`. Every
`ldgsts`/`tma` cell at or below the link ceiling.

If the TMA kernel fails to launch because it needs more dynamic shared memory than the device allows, the
`RuntimeCheck` above names it. Record the error in the JSONL by hand. Do not shrink the sweep silently.

### Task 5: Copy engine per segment against `cudaMemcpyBatchAsync`, and the small-segment workload

**Files:**
- Modify: `analysis/dsv41-drive/copy-mechanism/mech_bench.cuh` (add `mech_ce_each`, `mech_ce_batch`)
- Modify: `analysis/dsv41-drive/copy-mechanism/mech_bench.py` (register CE and small cells)

**Interfaces:**
- Consumes: `Job`, `Cell`, `CELL_GENERATORS`, `WRAPPERS`, `probe_ok`, `SEGMENTS` (Task 3); probe `batch` (Task 2).
- Produces: `mech_ce_each(jobs_cpu: int64[n,3], dev: any cuda tensor) -> int` (host ns spent in the API calls) and
  `mech_ce_batch(jobs_cpu, dev) -> int`. Methods `ce_each`, `ce_batch`, `ce_each_small`, `ce_batch_small`,
  `sm_small`. Each cell record carries `host_ns_p50`.

- [ ] **Step 1: Add the host functions to `mech_bench.cuh`, before the closing `}  // namespace sglang`**

```cpp
// Copy-engine baselines, timed on the host as well: the API cost per call is what the service's copy thread pays.
int64_t mech_ce_each(tvm::ffi::TensorView jobs, tvm::ffi::TensorView dev) {
  const auto stream = host::LaunchKernel::resolve_device(dev.device());
  const auto* j = static_cast<const mech::Job*>(jobs.data_ptr());
  const int64_t n = jobs.size(0);
  const auto t0 = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < n; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(reinterpret_cast<void*>(j[i].dst), reinterpret_cast<const void*>(j[i].src),
                               static_cast<size_t>(j[i].bytes), cudaMemcpyHostToDevice, stream))
        << "cudaMemcpyAsync";
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

#ifdef MECH_BATCH
int64_t mech_ce_batch(tvm::ffi::TensorView jobs, tvm::ffi::TensorView dev) {
  const auto stream = host::LaunchKernel::resolve_device(dev.device());
  const auto* j = static_cast<const mech::Job*>(jobs.data_ptr());
  const int64_t n = jobs.size(0);
  std::vector<void*> dsts(n);
  std::vector<const void*> srcs(n);
  std::vector<size_t> sizes(n);
  for (int64_t i = 0; i < n; ++i) {
    dsts[i] = reinterpret_cast<void*>(j[i].dst);
    srcs[i] = reinterpret_cast<const void*>(j[i].src);
    sizes[i] = static_cast<size_t>(j[i].bytes);
  }
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attr_index = 0;
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_CUDA(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), static_cast<size_t>(n), &attr, &attr_index, 1,
                                  stream))
      << "cudaMemcpyBatchAsync";
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
#endif
```

Add `#include <vector>` to the includes. The timer covers only the batch call, so building the vectors stays out
of the host number, as the per-segment path has no such work.

- [ ] **Step 2: Register the cells in `mech_bench.py`**

```python
WRAPPERS += ["mech_ce_each"]
BUILD_FLAGS += [("batch", "MECH_BATCH")]
SMALL = (1, 2, 4, 5)  # indices of the four small segments (20 KiB, 9 KiB, 4.5 KiB, 10 KiB)


def ce_cells(mod, probe):
    batch = probe_ok(probe, "batch")
    for n in (1, 2, 4):
        yield Cell("ce_each", 0, n, 0, 0, None, n, ALL_SEGMENTS, lambda cpu, dev: mod.mech_ce_each(cpu, dev))
        if batch:
            yield Cell("ce_batch", 0, n, 0, 0, None, n, ALL_SEGMENTS, lambda cpu, dev: mod.mech_ce_batch(cpu, dev))
    for lanes in (1, 4, 8):
        yield Cell("ce_each_small", 0, lanes, 0, 0, None, lanes, SMALL, lambda cpu, dev: mod.mech_ce_each(cpu, dev))
        if batch:
            yield Cell("ce_batch_small", 0, lanes, 0, 0, None, lanes, SMALL,
                       lambda cpu, dev: mod.mech_ce_batch(cpu, dev))
        yield Cell("sm_small", 8, 1, 16, 8 * 256 * 16, None, lanes, SMALL,
                   lambda cpu, dev: mod.mech_copy(dev, KIND_CV, 8, 1, 16))


CELL_GENERATORS += [ce_cells]
```

`mech_ce_batch` exists only in the `MECH_BATCH` build, so register its wrapper conditionally inside `load()`.
Replace `cuda_wrappers=[(n, n) for n in WRAPPERS]` with:

```python
                    cuda_wrappers=[(n, n) for n in WRAPPERS + (["mech_ce_batch"] if probe_ok(probe, "batch") else [])],
```

`measure()` copies the job table to the device before the events, and a CE cell ignores that copy. The host
number comes back from `run`.

- [ ] **Step 3: Commit, push, smoke on divix01**

Run the Task 3 Step 3 commands with this commit message:
`analysis(copy-mechanism): copy engine per segment against cudaMemcpyBatchAsync, and the small-segment workload`.
Expected: `EXIT=0`. `ce_each` at `n=4` is 13.4-13.8 GB/s (the 2026-09-25 bench recorded 13.61). The `*_small`
cells are well below the link: this is the 1.5-5 GB/s small-copy regime `NVME_PINNED_PREFETCH_HANDOFF.md:55-58`
recorded.

### Task 6: The Gen3 run, results, and the Gen5 one-liner

**Files:**
- Create: `analysis/dsv41-drive/copy-mechanism/README.md`
- Create: `analysis/dsv41-drive/copy-mechanism/results.md`
- Create (run output): `analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl`

**Interfaces:**
- Consumes: everything above.
- Produces: the Part A rows of the Decision table, filled in for divix01.

- [ ] **Step 1: The full run on divix01**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && rm -f analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/copy-mechanism/mech_bench.py --repo $PWD \
     --probe analysis/dsv41-drive/copy-mechanism/gen3/probe.json --out analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
     2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}" \
  && python3 analysis/dsv41-drive/copy-mechanism/mech_report.py analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl; echo REPORT_EXIT=$?'
scp divix01:/data/models/slang/nvfp4-work/wt-xfer/analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
  analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl
```

Expected: `EXIT=0` and `REPORT_EXIT=0`. If either is 1, the report names the cause (above ceiling, unsafe, blind).
Record it in `results.md` and stop before drawing conclusions.

- [ ] **Step 2: Write `README.md`**

````markdown
# Copy-mechanism sweep

GB/s against bytes in flight for every way the expert stream could move pinned host bytes into VRAM, plus the
link's ceiling, a serial PCIe read's latency and a host-flag round trip, on whatever host it runs on. Plan:
`docs/superpowers/plans/2026-09-27-expert-stream-transfer-measurement.md`, Part A.

## Run on a new host (the Gen5 one-liner)

```bash
# In a checkout of the branch, with a CUDA toolkit the sglang JIT resolves and one GPU visible:
export PYTHONPATH=$PWD/python D=analysis/dsv41-drive/copy-mechanism H=$(hostname)
python $D/probe.py --repo $PWD --out $D/$H-probe.json \
  && python $D/mech_bench.py --repo $PWD --probe $D/$H-probe.json --out $D/$H.jsonl --node 0 \
  && python3 $D/mech_report.py $D/$H.jsonl
```

Use `--node -1` on a host where NUMA binding is unwanted. On divix01, prefix the Python commands with
`flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63` (run protocol).

## Reading it

- `measured_ceiling_gbs` is the best cell of any method. `theoretical_gbs` is the PCIe payload ceiling.
- `bdp_bytes` = `serial_acquire_ns` x measured ceiling: the bytes in flight a copy needs to fill the link.
- A method's `knee_bytes` is the fewest bytes in flight within 5% of its best.
- `s_pattern` (grid 8, 1 load per thread, 16 B) is S. `cw_pattern` (grid 1, 4 loads, 16 B) is the copy wait's
  SM reads.
- `fresh` records: every swept method must read bytes the host rewrote mid-kernel, and the `.nc` control must not.
- The exit status is 1 on an above-ceiling cell, an unsafe method, or a blind fresh check.
````

- [ ] **Step 3: Write `results.md` from the run**

It has four sections: the `mech_report.py` output (pasted verbatim), the commands (the Step 1 block), the probe
table from `gen3/probe.json`, and the Part A rows of the Decision table. Fill those rows as
`<condition>: <measured numbers> -> yes/no`. Example of the required form:
`Deeper unroll: sm_cv16 G8 U4 = 12.31 vs s_pattern 12.27 (1.003x < 1.05) -> no (on Gen3)`.
State whether the Gen3 expectation at the top of Part A held. Where it did not, say which cell broke it.

- [ ] **Step 4: Commit**

```bash
git add analysis/dsv41-drive/copy-mechanism/README.md analysis/dsv41-drive/copy-mechanism/results.md \
  analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl
git commit -m "analysis(copy-mechanism): divix01 Gen3 sweep results and the per-host one-liner"
git push origin expert-stream-transfer-measurement
```

---

## Part B: PDL on the lease chain, measured (not shipped)

**What the early trigger may and may not do (verified 2026-09-27).** The CUDA Programming Guide (13.4, "Programmatic
Dependent Launch") says `cudaGridDependencySynchronize()` (`griddepcontrol.wait`) "will block until all primary
kernels the secondary kernel is dependent on have completed and flushed results to global memory". The documented
note on `PDLWaitPrimary` in `sgl_kernel/utils.cuh` says the same. So an early `launch_dependents` only lets the
dependent be *scheduled* early. Every write the primary makes, before or after the trigger, is visible once the
dependent's wait returns. The inline comment inside `PDLTriggerSecondary` ("only covers writes issued BEFORE
launch_dependents") contradicts that guarantee; do not rely on it. `pdl_early` is therefore a legitimate ship
candidate, not just a measurement. The one hazard, in both modes, is a read of upstream data placed before the wait.

### Task 7: Skeleton chain, PDL upper bound (analysis only, starts now)

**Files:**
- Create: `analysis/dsv41-drive/chain-pdl/skeleton.cuh`
- Create: `analysis/dsv41-drive/chain-pdl/skeleton.py`
- Create: `analysis/dsv41-drive/chain-pdl/README.md`
- Create: `analysis/dsv41-drive/chain-pdl/results.md`

**Interfaces:**
- Consumes: `sgl_kernel/utils.cuh` `device::PDLWaitPrimary<bool>`, `device::PDLTriggerSecondary<bool>`,
  `host::LaunchKernel::enable_pdl`.
- Produces: `skel_stage(words: int64[k] cuda, index: int, grid: int, block: int, work_ns: int, mode: int)`, where
  mode 0 means no PDL, 1 means PDL with the implicit trigger at exit, and 2 means PDL with the trigger right after
  the wait. The JSONL records are `{"kind": "skeleton", "mode", "work_ns", "layers", "replay_us_p50",
  "per_layer_us_p50"}`. Task 8 reads mode 1's and mode 2's per-layer saving against mode 0.

- [ ] **Step 1: Write `skeleton.cuh`**

```cpp
// The lease chain's launch shapes with no protocol: every stage waits for its predecessor under PDL, reads the
// predecessor's word and writes its own, so each edge is a real data dependency, and spins `work_ns` standing in for
// the stage's body. mode 0: no PDL; 1: PDL, trigger implied at exit; 2: PDL, trigger right after the wait (valid per
// the PTX ISA: griddepcontrol.wait waits for the primary grid to COMPLETE and its memory to be visible).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

template <int kMode>
__global__ void skel_stage_kernel(int64_t* words, int64_t index, int64_t work_ns) {
  device::PDLWaitPrimary<kMode != 0>();
  device::PDLTriggerSecondary<kMode == 2>();
  if (threadIdx.x != 0) return;
  uint64_t t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (uint64_t t = t0; static_cast<int64_t>(t - t0) < work_ns;)
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  if (blockIdx.x != 0) return;
  // Stage 0 counts replays; every later stage copies its predecessor, so after R replays every word reads R.
  words[index] = index == 0 ? words[0] + 1 : words[index - 1];
}

void skel_stage(tvm::ffi::TensorView words, int64_t index, int64_t grid, int64_t block, int64_t work_ns, int64_t mode) {
  const auto stream = host::LaunchKernel::resolve_device(words.device());
  auto* w = static_cast<int64_t*>(words.data_ptr());
  const dim3 g(static_cast<unsigned>(grid)), b(static_cast<unsigned>(block));
  switch (mode) {
    case 0: host::LaunchKernel(g, b, stream)(skel_stage_kernel<0>, w, index, work_ns); return;
    case 1: host::LaunchKernel(g, b, stream).enable_pdl(true)(skel_stage_kernel<1>, w, index, work_ns); return;
    case 2: host::LaunchKernel(g, b, stream).enable_pdl(true)(skel_stage_kernel<2>, w, index, work_ns); return;
  }
  host::RuntimeCheck(false, "mode: 0, 1 or 2");
}

}  // namespace sglang
```

- [ ] **Step 2: Write `skeleton.py`**

```python
#!/usr/bin/env python3
"""Upper bound on what PDL saves per layer on the lease chain: a skeleton with production launch shapes.

The chain post -> W1 -> C1 -> A1 -> S -> A2 -> CW -> F (exl3_ram_miss.py post(); shapes from the launchers:
1x32, 1x32, 8x256, 1x8, 8x256, 1x8, 1x256, 1x32) is captured 40 times in one CUDA graph, each layer separated by
a non-PDL 'moe' stage (the real graph has attention and MoE between chains). Time per replay with CUDA events.

    PYTHONPATH=<repo>/python python skeleton.py --repo <repo> --out skeleton.jsonl
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAIN = (("post", 1, 32), ("w1", 1, 32), ("c1", 8, 256), ("a1", 1, 8), ("s", 8, 256), ("a2", 1, 8), ("cw", 1, 256),
         ("f", 1, 32))
MOE = ("moe", 1, 256)
LAYERS = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", type=int, default=200)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    from sglang.kernels.jit.utils.compile.loader import load_jit

    mod = load_jit("chain_pdl_skeleton", cuda_files=[str(HERE / "skeleton.cuh")],
                   cuda_wrappers=[("skel_stage", "skel_stage")])
    stages = []
    for _ in range(LAYERS):
        stages += [(name, g, b, True) for name, g, b in CHAIN] + [(*MOE, False)]
    out = open(a.out, "a")
    for work_ns in (0, 2000):
        for mode in (0, 1, 2):
            words = torch.zeros(len(stages), dtype=torch.int64, device="cuda")
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for i, (_, g, b, pdl) in enumerate(stages):  # warm: loads the module's kernels before capture
                    mod.skel_stage(words, i, g, b, work_ns, mode if pdl else 0)
            torch.cuda.synchronize()
            words.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for i, (_, g, b, pdl) in enumerate(stages):
                    mod.skel_stage(words, i, g, b, work_ns, mode if pdl else 0)
            torch.cuda.synchronize()
            words.zero_()
            times = []
            for r in range(a.replays + 20):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graph.replay()
                e1.record()
                torch.cuda.synchronize()
                if r >= 20:
                    times.append(e0.elapsed_time(e1) * 1e3)
            got = set(words.cpu().tolist())
            if got != {a.replays + 20}:
                raise SystemExit(f"mode {mode}: stage words {sorted(got)[:5]}, not all {a.replays + 20}: an edge lost "
                                 "its ordering")
            us = statistics.median(times)
            rec = {"kind": "skeleton", "mode": mode, "work_ns": work_ns, "layers": LAYERS,
                   "replay_us_p50": round(us, 2), "per_layer_us_p50": round(us / LAYERS, 3)}
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Commit, push, run on divix01**

```bash
git add analysis/dsv41-drive/chain-pdl/skeleton.cuh analysis/dsv41-drive/chain-pdl/skeleton.py
git commit -m "analysis(chain-pdl): skeleton lease chain with production launch shapes, PDL off / implicit / early trigger"
git push origin expert-stream-transfer-measurement
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && rm -f analysis/dsv41-drive/chain-pdl/skeleton.jsonl \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/chain-pdl/skeleton.py --repo $PWD \
     --out analysis/dsv41-drive/chain-pdl/skeleton.jsonl 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"'
scp divix01:/data/models/slang/nvfp4-work/wt-xfer/analysis/dsv41-drive/chain-pdl/skeleton.jsonl analysis/dsv41-drive/chain-pdl/
```

Expected: `EXIT=0` and six records. The "every word reads R" check passes in all modes; if it fails in mode 1 or 2,
PDL broke an edge, which is itself a finding to report. Mode 1 and 2 per-layer times are at or below mode 0's.

- [ ] **Step 4: Write `README.md` and `results.md`, commit**

`README.md` covers:
- What the skeleton models.
- What it leaves out: real work per stage, the host round trip, and PDL on C1 (`expert_cache_transfer.cuh`, which
  Task 8 does not hook).
- The run command from Step 3.

`results.md` covers:
- The six records.
- The saving per layer and per 40-layer step for modes 1 and 2 at both `work_ns`.
- The Task 8 gate verdict: saving >= 2 us/layer in mode 1 or mode 2 at `work_ns` 2000 -> run Task 8, else stop
  Part B here and mark the PDL row of the Decision table "no (bound below threshold)".

```bash
git add analysis/dsv41-drive/chain-pdl/README.md analysis/dsv41-drive/chain-pdl/results.md analysis/dsv41-drive/chain-pdl/skeleton.jsonl
git commit -m "analysis(chain-pdl): Gen3 skeleton results and the Task 8 gate"
git push origin expert-stream-transfer-measurement
```

### Task 8: The real chain under a PDL test hook (gated)

**Gate:** start only if (a) the Global Constraints gate prints `MERGED`, and (b) Task 7's `results.md` says
"run Task 8". Otherwise mark this task skipped with the reason.

Branch `expert-stream-pdl-probe` from `origin/master` (after the merge). It is not merged unless the user decides
to ship PDL. Its results go back onto `expert-stream-transfer-measurement` in `chain-pdl/results.md`.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (the `kTestPdl`/`kTestPdlEarly` constants)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh`: entry of
  `exl3_ram_miss_post_kernel`, `exl3_ram_miss_lease_stream_hit_wait_kernel`, `exl3_ram_miss_lease_stage_ack_kernel`,
  `exl3_ram_miss_lease_finalize_kernel`, and their four launchers' `LaunchKernel(...)` calls
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh`: entry of
  `exl3_ram_miss_lease_stream_kernel`, `exl3_ram_miss_lease_copy_wait_kernel`, and their launchers
- Create: `analysis/dsv41-drive/chain-pdl/chain_pdl.py`

**Interfaces:**
- Consumes: `device_module_with_hooks(defines)` (`expert_stream_transport.py`); `StreamService`, `TOP_K`
  (`test/manual/dsv41/test_exl3_piece_stream_cuda.py`); `sass_gate.py` (native-sync plan, Task 0, on divix01).
- Produces: records `{"kind": "chain", "scenario": "all_hit"|"mixed"|"all_hit_ce", "mode": "off"|"pdl"|"pdl_early",
  "replay_us_p50", "hits_mean", "replays"}`.

- [ ] **Step 1: Add the hook constants to `lease_device.cuh`**, after `constexpr int kStateWords = 20;`:

```cpp
// Test builds only (device_module_with_hooks): launch the chain's kernels with PDL and wait for the predecessor as
// each kernel's first statement, executed by every thread (add-jit-kernel skill: every thread that reads producer
// data executes the wait). _EARLY also triggers the dependent right after the wait. Production defines neither, so
// both calls compile to nothing and the launchers pass enable_pdl(false), which adds no launch attribute.
#ifdef EXL3_RAM_MISS_TEST_PDL
constexpr bool kTestPdl = true;
#else
constexpr bool kTestPdl = false;
#endif
#ifdef EXL3_RAM_MISS_TEST_PDL_EARLY
constexpr bool kTestPdlEarly = true;
#else
constexpr bool kTestPdlEarly = false;
#endif
```

- [ ] **Step 2: Add the entry wait to the six kernels, and `enable_pdl` to their launchers**

The first statement of each of the six kernels' bodies (before the `__grid_constant__` parameter unpacking is
fine, since those are register copies) becomes:

```cpp
  device::PDLWaitPrimary<device::expert_stream::kTestPdl>();
  device::PDLTriggerSecondary<device::expert_stream::kTestPdlEarly>();
```

Each launcher's `LaunchKernel(...)` call gains `.enable_pdl(device::expert_stream::kTestPdl)` before its call
operator. For example, the finalize launcher becomes
`LaunchKernel(1, device::expert_stream::kBlock, stream).enable_pdl(device::expert_stream::kTestPdl)(exl3_ram_miss_lease_finalize_kernel, params);`.
C1 (`copy_expert_row_segments_gpu`, a different module) and the `torch.add` are not hooked. Their edges stay
ordinary, and A1, launched with the attribute after C1, still waits for C1 to complete.

- [ ] **Step 3: Prove the default build unchanged, and the hook builds compile**

```bash
git push -u origin expert-stream-pdl-probe
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-pdl origin/expert-stream-pdl-probe \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-pdl-base origin/master \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-pdl-base pdl-base.sass \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-pdl pdl-after.sass \
  && python3 sass_gate.py diff pdl-base.sass pdl-after.sass --exact; echo EXIT=$?'
```

Expected: `GATE PASS`, `EXIT=0`. If `sass_gate.py` is gone from `scratch-atomic-probe`, recreate it from the
native-sync plan's Task 0 Step 3 (the file is given in full there).

- [ ] **Step 4: Write `chain_pdl.py` on the transfer-measurement branch**

```python
#!/usr/bin/env python3
"""The real lease chain, captured in a CUDA graph, timed with and without PDL (test hook builds).

Scenarios (StreamService, test/manual/dsv41/test_exl3_piece_stream_cuda.py; TOP_K lanes of one layer):
  all_hit     every lane a RAM hit: W1 claims them all, S has nothing to wait for; the chain's own cost
  mixed       half the lanes RAM misses each replay (the host reads them): the chain behind a real read
  all_hit_ce  all_hit with the copy engine and the copy wait (CW) in the chain
Modes: off (production module), pdl (EXL3_RAM_MISS_TEST_PDL), pdl_early (+ EXL3_RAM_MISS_TEST_PDL_EARLY).
Every 20th replay's destination bytes are checked against the source rows.

    PYTHONPATH=<pdl-probe worktree>/python python chain_pdl.py --repo <pdl-probe worktree> --tmp <dir> --out chain.jsonl
"""
import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch

MODES = {"off": None, "pdl": ["EXL3_RAM_MISS_TEST_PDL"],
         "pdl_early": ["EXL3_RAM_MISS_TEST_PDL", "EXL3_RAM_MISS_TEST_PDL_EARLY"]}


def retired(s) -> bool:
    c = s.counters()
    return c["leases_granted"] == c["leases_acked"] + c["leases_voided"] + c["leases_copied"]


def run(repo: Path, tmp: Path, scenario: str, mode: str, replays: int) -> dict:
    sys.path.insert(0, str(repo / "test/manual/dsv41"))
    from test_exl3_piece_stream_cuda import TOP_K, StreamService

    from sglang.kernels.ops.moe.expert_stream_transport import device_module_with_hooks

    s = StreamService(tmp / f"{scenario}-{mode}", copy_engine=scenario == "all_hit_ce")
    try:
        if MODES[mode] is not None:
            s.dev._module = device_module_with_hooks(MODES[mode])
        base = list(range(TOP_K))
        s.plan(base)
        s.step()  # loads the rows: every later all-hit request hits RAM
        torch.cuda.synchronize()
        assert s.until(lambda: retired(s)), s.counters()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            s.step()
            s.total()
        torch.cuda.synchronize()
        assert s.until(lambda: retired(s)), s.counters()
        times, hits = [], []
        for r in range(replays + 10):
            if scenario == "mixed":
                # keep half of the previous plan (RAM hits) and bring in half new experts (RAM misses)
                keep = base[TOP_K // 2:]
                fresh = [(base[-1] + 1 + k) % 16 for k in range(TOP_K - len(keep))]
                base = keep + [e for e in fresh if e not in keep][: TOP_K - len(keep)]
            s.plan(base)
            hits.append(sum(s.host.contains(s.row, e) for e in base))
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
        return {"kind": "chain", "scenario": scenario, "mode": mode, "replay_us_p50": round(statistics.median(times), 2),
                "hits_mean": round(statistics.mean(hits), 2), "replays": replays}
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", type=int, default=200)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    tmp = Path(a.tmp or tempfile.mkdtemp(prefix="chain-pdl-"))
    tmp.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as out:
        for scenario in ("all_hit", "mixed", "all_hit_ce"):
            for mode in MODES:
                rec = run(repo, tmp, scenario, mode, a.replays)
                print(json.dumps(rec), flush=True)
                out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

The script runs against the `wt-pdl` worktree, whose `test/manual/dsv41` holds `StreamService`. The copy engine
needs `CUDA_MODULE_LOADING=EAGER` (`check_copy_engine_module_loading`), so the run sets it.

- [ ] **Step 5: Run on divix01**

```bash
git add analysis/dsv41-drive/chain-pdl/chain_pdl.py
git commit -m "analysis(chain-pdl): real lease chain in a graph, timed with the PDL test hooks"
git push origin expert-stream-transfer-measurement
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && cd /data/models/slang/nvfp4-work/wt-pdl && mkdir -p /data/models/slang/nvfp4-work/scratch-chain-pdl \
  && PYTHONPATH=$PWD/python CUDA_MODULE_LOADING=EAGER OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python /data/models/slang/nvfp4-work/wt-xfer/analysis/dsv41-drive/chain-pdl/chain_pdl.py \
     --repo $PWD --tmp /data/models/slang/nvfp4-work/scratch-chain-pdl \
     --out /data/models/slang/nvfp4-work/wt-xfer/analysis/dsv41-drive/chain-pdl/chain.jsonl 2>&1 | tail -10; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=0`, nine records, and no byte-check assertion. `all_hit` `hits_mean` equals `TOP_K` (6). `mixed`
has `hits_mean` near 3.

- [ ] **Step 6: Results, and hand the decision to the user**

Copy `chain.jsonl` back (`scp`) and add a "Real chain" section to `chain-pdl/results.md`. It contains:
- the nine records;
- per scenario, the saving of `pdl` and `pdl_early` against `off`, in us per layer and x 40 per step;
- the Decision table's PDL row: `all_hit` saving x 40 against 0.67 ms per step.

Do not open a PR that turns PDL on. The results section ends with the one-line question for the user: ship PDL on
the chain, yes or no.

```bash
git add analysis/dsv41-drive/chain-pdl/chain.jsonl analysis/dsv41-drive/chain-pdl/results.md
git commit -m "analysis(chain-pdl): real-chain PDL results for the user's ship decision"
git push origin expert-stream-transfer-measurement
```

---

## Part C: device stage timeline for the lease chain (diagnostic, off by default)

**Gate:** the Global Constraints merge check prints `MERGED`. Then branch `expert-stream-device-timeline` from
`origin/master`, push it, and create divix01 worktrees: `wt-timeline` at the branch and `wt-timeline-base` at the
branch point. Use the same commands as Task 8 Step 3, with those names.

Design (MOE_SERVICE_TRACE_PLAN.md section 2, carried over to the lease chain):
- The ring is a fixed, preallocated device buffer. Records carry a ticket and the request generation.
- Nothing inside the captured graph allocates, synchronizes, or copies.
- The ring is drained by a copy that is stream-ordered after the replay, so no kernel writes while it is read.
- Overwrite is detected (`lost`), never silently absorbed.
- `%globaltimer` is calibrated to host `CLOCK_MONOTONIC` by an in-process round trip. The calibration error
  (half the round trip) is reported, and raw GPU and CPU clocks are never subtracted.
- Every report states its join coverage.

Which S block stamps what:
- **first-seen** comes from block 0's lane threads. Every block polls the masks on its own; block 0's view is the
  one the stamps sample.
- **copied** comes from every block's leader after the pass barrier, with the block id in the record. S has no
  inter-block barrier, so a piece is copied only when all 8 blocks have copied their slices. The join takes the
  maximum over blocks.

This costs 8 stamps per piece-pass instead of 1. It is the only way to see a piece's true completion without
adding a barrier.

Clock resolution: `%globaltimer`'s update interval is not documented for `sm_120`. Some GPUs advance it in 1 us
steps unless a profiler is attached. Task 13's report prints the smallest positive gap between consecutive stamps
of one block (`gpu_tick_ns`). A per-pass or visibility number below about 5 x that tick is unresolved, and
`results.md` must say so rather than quote it.

### Task 9: Timeline record format and decoder (Python, CPU-tested)

**Files:**
- Create: `python/sglang/kernels/ops/moe/expert_stream_timeline.py`
- Test: `test/registered/unit/layers/moe/test_expert_stream_timeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EVENTS: dict[int, str]`, codes 1-13: `post_before, post_publish, w1_enter, w1_exit, s_enter, s_admit, s_seen,
    s_copied, s_exit, s_commit, cw_enter, cw_done, finalize`.
  - `RECORD_WORDS = 4`, `DEFAULT_CAPACITY = 1 << 15`, `ROW_NONE = 0xFFFF`.
  - `pack(ticket, generation, code, block, lane, bits, row, extra, gpu_ns) -> list[int]`: the int64 words.
  - `window(head: int, prev_head: int, capacity: int) -> tuple[int, int]`, returning `(first ticket, lost)`.
  - `decode_window(rows: list[list[int]], first: int, head: int, prev_head: int) -> TimelineDrain`.
  - `@dataclass TimelineEvent(ticket, generation, event: str, block, lane, bits, row, extra, gpu_ns)` with property
    `seq`.
  - `@dataclass TimelineDrain(events: list[TimelineEvent], head: int, lost: int, torn: int)`.
  - `calibration(gpu_ns: list[int], host_ns: list[list[int]]) -> dict`, with keys
    `offset_ns, error_ns, rounds, host_ns`. It is used as `host = gpu - offset_ns`.

- [ ] **Step 1: Write the failing tests**

```python
"""The device timeline's ring format, overwrite accounting and clock calibration (CPU)."""

import pytest

from sglang.kernels.ops.moe import expert_stream_timeline as tl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

S_SEEN = next(code for code, name in tl.EVENTS.items() if name == "s_seen")


def _ring(capacity, tickets):
    rows = [[0, 0, 0, 0] for _ in range(capacity)]
    for t in tickets:
        rows[t % capacity] = tl.pack(t, (3 << 32) | 77, S_SEEN, 2, 5, 0b1010, 11, 42, 1_000 + t)
    return rows


def _rows(ring, first, head):
    return [ring[t % len(ring)] for t in range(first, head)]


def test_a_record_round_trips_every_field():
    ring = _ring(8, range(1))
    first, lost = tl.window(1, 0, 8)
    drain = tl.decode_window(_rows(ring, first, 1), first, 1, 0)
    (e,) = drain.events
    assert (e.ticket, e.event, e.block, e.lane, e.bits, e.row, e.extra, e.gpu_ns) == (0, "s_seen", 2, 5, 0b1010, 11, 42, 1000)
    assert e.generation == (3 << 32) | 77 and e.seq == 77
    assert (drain.head, drain.lost, drain.torn) == (1, 0, 0)


def test_words_above_2_63_survive_the_int64_round_trip():
    words = tl.pack(5, (1 << 63) | 9, S_SEEN, 0, 0, 0, 0, 0xFFFF, (1 << 63) + 7)
    assert all(-(1 << 63) <= w < (1 << 63) for w in words)  # storable in an int64 tensor
    (e,) = tl.decode_window([words], 5, 6, 5).events
    assert e.generation == (1 << 63) | 9 and e.gpu_ns == (1 << 63) + 7 and e.extra == 0xFFFF


def test_a_ring_lapped_between_drains_counts_the_lost_tickets():
    ring = _ring(4, range(6))  # tickets 0 and 1 overwritten by 4 and 5
    first, lost = tl.window(6, 0, 4)
    assert (first, lost) == (2, 2)
    drain = tl.decode_window(_rows(ring, first, 6), first, 6, 0)
    assert [e.ticket for e in drain.events] == [2, 3, 4, 5] and drain.lost == 2


def test_a_record_with_the_wrong_ticket_is_torn_not_used():
    ring = _ring(4, range(3))
    ring[1][0] = 0  # never finished
    drain = tl.decode_window(_rows(ring, 0, 3), 0, 3, 0)
    assert [e.ticket for e in drain.events] == [0, 2] and drain.torn == 1


def test_an_unknown_event_code_is_torn():
    ring = _ring(4, range(1))
    ring[0][2] = (ring[0][2] & ~0xFF) | 0xEE
    assert tl.decode_window(_rows(ring, 0, 1), 0, 1, 0).torn == 1


def test_a_head_that_moved_backwards_is_refused():
    with pytest.raises(ValueError, match="backwards"):
        tl.window(3, 5, 8)


def test_calibration_takes_the_tightest_round_and_reports_half_its_width():
    cal = tl.calibration([1_000_500, 2_000_100], [[500, 1500], [100, 300]])
    assert cal["error_ns"] == 100  # round 1: width 200
    assert cal["host_ns"] == 200 and cal["offset_ns"] == 2_000_100 - 200 and cal["rounds"] == 2
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py -q`
Expected: `ModuleNotFoundError: ... expert_stream_timeline`.

- [ ] **Step 3: Write `expert_stream_timeline.py` (the pure part)**

```python
"""Device stage timeline for the expert-stream lease chain (diagnostic: SGLANG_DSV41_DEBUG_DEVICE_TIMELINE).

The chain's kernels, built with -DEXPERT_STREAM_TIMELINE, stamp events into a fixed device ring
(lease_device.cuh, timeline_stamp). A record is four uint64 words, stored in an int64 tensor:

  w0  ticket + 1 (0: never written); the record sits at ring[ticket % capacity]
  w1  the request's generation, (epoch << 32) | seq; 0 for none
  w2  event | block << 8 | lane << 16 | bits << 24 | row << 32 | extra << 48  (row 0xFFFF: the kernel has no row)
  w3  %globaltimer ns, read before the ticket is taken

The head word counts every ticket issued. A drain copies head and ring stream-ordered after a replay, so no record
is being written while it is read; tickets older than head - capacity were overwritten and are counted `lost`, and
a record whose w0 is not its ticket + 1 is counted `torn` and never used.
"""
from __future__ import annotations

from dataclasses import dataclass

EVENTS = {
    1: "post_before", 2: "post_publish", 3: "w1_enter", 4: "w1_exit", 5: "s_enter", 6: "s_admit", 7: "s_seen",
    8: "s_copied", 9: "s_exit", 10: "s_commit", 11: "cw_enter", 12: "cw_done", 13: "finalize",
}
RECORD_WORDS = 4
DEFAULT_CAPACITY = 1 << 15  # 1 MiB; a 40-layer decode step stamps ~9k records at 3.16 lanes per layer
ROW_NONE = 0xFFFF
_U64 = (1 << 64) - 1


def _signed(word: int) -> int:
    word &= _U64
    return word - (1 << 64) if word >= 1 << 63 else word


def pack(ticket, generation, code, block, lane, bits, row, extra, gpu_ns) -> list[int]:
    w2 = (code & 0xFF) | (block & 0xFF) << 8 | (lane & 0xFF) << 16 | (bits & 0xFF) << 24 | (row & 0xFFFF) << 32 \
        | (extra & 0xFFFF) << 48
    return [_signed(ticket + 1), _signed(generation), _signed(w2), _signed(gpu_ns)]


@dataclass(frozen=True)
class TimelineEvent:
    ticket: int
    generation: int
    event: str
    block: int
    lane: int
    bits: int
    row: int
    extra: int
    gpu_ns: int

    @property
    def seq(self) -> int:
        return self.generation & 0xFFFFFFFF


@dataclass(frozen=True)
class TimelineDrain:
    events: list[TimelineEvent]
    head: int
    lost: int
    torn: int


def window(head: int, prev_head: int, capacity: int) -> tuple[int, int]:
    """The first ticket still in the ring since the last drain, and how many were overwritten before it."""
    if head < prev_head:
        raise ValueError(f"the head moved backwards ({prev_head} -> {head}): the ring was reset or rebound")
    first = max(prev_head, head - capacity)
    return first, first - prev_head


def decode_window(rows: list[list[int]], first: int, head: int, prev_head: int) -> TimelineDrain:
    """`rows[i]` is the ring row of ticket `first + i`, for every ticket in [first, head)."""
    events, torn = [], 0
    for i, row in enumerate(rows):
        ticket = first + i
        w0, w1, w2, w3 = (word & _U64 for word in row)
        code = w2 & 0xFF
        if w0 != ticket + 1 or code not in EVENTS:
            torn += 1
            continue
        events.append(TimelineEvent(ticket, w1, EVENTS[code], (w2 >> 8) & 0xFF, (w2 >> 16) & 0xFF,
                                    (w2 >> 24) & 0xFF, (w2 >> 32) & 0xFFFF, (w2 >> 48) & 0xFFFF, w3))
    return TimelineDrain(events, head, first - prev_head, torn)


def calibration(gpu_ns: list[int], host_ns: list[list[int]]) -> dict:
    """GPU-to-host clock offset from a round trip: in round r the host stamps t0, releases the flag, the GPU sees it
    at g (its clock), and the host stamps t1 once it sees the GPU's answer, so g happened within [t0, t1] on the host
    clock. The tightest round wins; the error is half its width. host = gpu - offset_ns."""
    best = min(range(len(gpu_ns)), key=lambda r: host_ns[r][1] - host_ns[r][0])
    t0, t1 = host_ns[best]
    mid = (t0 + t1) // 2
    return {"offset_ns": gpu_ns[best] - mid, "error_ns": (t1 - t0 + 1) // 2, "rounds": len(gpu_ns), "host_ns": mid}
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"`
Expected: `7 passed`, `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/ops/moe/expert_stream_timeline.py test/registered/unit/layers/moe/test_expert_stream_timeline.py
git commit -m "feat(expert-stream): device timeline record format, overwrite-aware decoder and clock calibration (diagnostic)"
```

### Task 10: Host trace schema 8: per-piece publish clock and per-row expert

The host's per-piece `piece_publish` field is a **sequence number**, not a time (`reader_base.h:200`,
`row_reader.h:1341`). Of the per-piece fields, only `piece_cqe` is a clock, and it records vetting, not
publishing. The host record also does not say which expert a row ordinal is, while the device knows lanes and
experts, not ordinals. Without these two fields the join could only bound publish-to-seen from below. This task
adds both, under the existing `if (c.trace)` guards, so a trace-off run is unchanged.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_base.h:196-265` (doc block; two new
  `StageRecord` arrays after `piece_publish_refused`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h:663` (row expert),
  `:1320-1345` (publish clock)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`STAGE_FIELDS`, `stage_records`)
- Modify: `python/sglang/srt/layers/moe/exl3_stream_trace.py` (`RAM_MISS_TRACE_SCHEMA = 8`, docstrings)
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_trace_export.py` (`LAYOUT_PIN`)
- Test: `test/registered/unit/layers/moe/test_expert_stream_timeline.py` (decode of the new fields)

**Interfaces:**
- Consumes: nothing from Task 9.
- Produces: each `stage_records(...)[i]["pieces"][k]` gains `"publish_ns": list[int]` (length `STAGE_PIECES`,
  host `CLOCK_MONOTONIC`, 0 = never) and `"expert": int`. The JSONL `ram_miss_request` line (schema 8) carries
  them in `pieces[k]`.

- [ ] **Step 1: Write the failing decode test** (append to `test_expert_stream_timeline.py`)

```python
def test_schema_8_stage_records_carry_publish_clocks_and_row_experts():
    import torch

    from sglang.kernels.ops.moe import expert_stream_transport as ops

    words = torch.zeros((1, len(ops.STAGE_FIELDS)), dtype=torch.int64)
    field = {name: i for i, name in enumerate(ops.STAGE_FIELDS)}
    words[0, field["kind"]] = 0  # demand
    words[0, field["status"]] = 1  # served
    words[0, field["rows_asked"]] = 1
    words[0, field["piece_stream"]] = 1
    words[0, field["piece_publish_ns_0_3"]] = 123_456
    words[0, field["row_expert_0"]] = 7
    (record,) = ops.stage_records(words)
    assert record["pieces"][0]["publish_ns"][3] == 123_456
    assert record["pieces"][0]["expert"] == 7
    assert not any(name.startswith(("piece_publish_ns_", "row_expert_")) for name in record)
```

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py -q -k schema_8`
Expected: FAIL with `KeyError: 'piece_publish_ns_0_3'`.

- [ ] **Step 2: Add the C++ fields and stamps**

In `reader_base.h`, add to the doc block after the `piece_publish_refused` line:

```cpp
// Device timeline join (schema 8). Stamped only when tracing, like every field here:
//   piece_publish_ns[k][j]  the clock just before piece j of row k was published (before its first readiness-word
//                           store), so a device first-seen minus it bounds that piece's visibility from above;
//   row_expert[k]           row k's expert id: the device knows lanes and experts, not the read's row ordinals.
```

and to `StageRecord`, after `int64_t piece_publish_refused = 0;`:

```cpp
  int64_t piece_publish_ns[kTraceRows][kPieces] = {};
  int64_t row_expert[kTraceRows] = {};
```

In `row_reader.h:663`, replace
`if (c.trace && ordinal < static_cast<size_t>(kTraceRows)) c.trace->row_admit[ordinal] = admitted;` with:

```cpp
      if (c.trace && ordinal < static_cast<size_t>(kTraceRows)) {
        c.trace->row_admit[ordinal] = admitted;
        c.trace->row_expert[ordinal] = (*c.experts)[ordinal];
      }
```

In the publish function (`row_reader.h`, the block ending
`c.trace->piece_publish[r.ordinal][j] = seq;`), add as the first statement after
`const bool twice = ...;` and the `last_publish_delay_ns` block, before `if (c.publish != nullptr ...)`:

```cpp
    const int64_t publish_clock = c.trace ? now_ns() : 0;  // schema 8; no clock read when not tracing
```

and replace `if (r.ordinal < static_cast<size_t>(kTraceRows)) c.trace->piece_publish[r.ordinal][j] = seq;` with:

```cpp
      if (r.ordinal < static_cast<size_t>(kTraceRows)) {
        c.trace->piece_publish[r.ordinal][j] = seq;
        c.trace->piece_publish_ns[r.ordinal][j] = publish_clock;
      }
```

- [ ] **Step 3: Extend `STAGE_FIELDS` and `stage_records`**

At the end of `STAGE_FIELDS` (after `"pieces_published", "pieces_out_of_order", "piece_publish_refused",`):

```python
    *(f"piece_publish_ns_{k}_{j}" for k in range(STAGE_TRACE_ROWS) for j in range(STAGE_PIECES)),
    *(f"row_expert_{k}" for k in range(STAGE_TRACE_ROWS)),
```

In `stage_records`, add to each `pieces` entry dict:

```python
                "publish_ns": [record[f"piece_publish_ns_{k}_{j}"] for j in range(STAGE_PIECES)],
                "expert": record[f"row_expert_{k}"],
```

and, next to the loop that deletes the `piece_publish_{k}_{j}` keys, delete the new ones for every `k`:

```python
        for k in range(STAGE_TRACE_ROWS):
            del record[f"row_expert_{k}"]
            for j in range(STAGE_PIECES):
                del record[f"piece_publish_ns_{k}_{j}"]
```

Extend the docstring's schema paragraph: "Schema 8 adds `pieces[].publish_ns`, the clock just before each piece's
publish, and `pieces[].expert`, the row's expert id."

- [ ] **Step 4: Bump the schema and the layout pin**

In `exl3_stream_trace.py`, set `RAM_MISS_TRACE_SCHEMA = 8`. Extend the comment above it and the
`record_ram_miss_requests` docstring: "8: adds pieces[].publish_ns and pieces[].expert (device timeline join); no
earlier field changed meaning."

Compute the new digest and paste the pair into `LAYOUT_PIN` in `test_exl3_ram_miss_trace_export.py`:

```bash
PYTHONPATH=$PWD/python python -c "import hashlib, sglang.kernels.ops.moe.expert_stream_transport as o; print(hashlib.sha256('\n'.join(o.STAGE_FIELDS).encode()).hexdigest()[:16])"
```

The pin becomes `LAYOUT_PIN = (8, "<printed digest>")`.

- [ ] **Step 5: Run the CPU tests (they build the host module, which checks the C++ word count)**

```bash
PYTHONPATH=$PWD/python taskset -c 0-63 python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py \
  test/registered/unit/kernels/test_exl3_ram_miss_trace_export.py test/registered/unit/layers/moe/test_exl3_stream_trace.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: all pass, `EXIT=0`. A `C++ StageRecord has N words, STAGE_FIELDS M` error means Step 2 and Step 3 do not
match.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_base.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py python/sglang/srt/layers/moe/exl3_stream_trace.py \
  test/registered/unit/kernels/test_exl3_ram_miss_trace_export.py test/registered/unit/layers/moe/test_expert_stream_timeline.py
git commit -m "feat(expert-stream): stage trace schema 8, per-piece publish clock and per-row expert (trace-only)"
```

### Task 11: Device stamps, the timeline module build, bind and calibrate; SASS and GPU proof

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (event constants; ring, helper,
  bind, calibrate under `#ifdef EXPERT_STREAM_TIMELINE`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (post, stream hit wait, finalize)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh` (S, CW)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_timeline.py` (`timeline_module`, `DeviceTimeline`)
- Test: `test/registered/unit/layers/moe/test_expert_stream_timeline.py` (source rules)
- Test: `test/manual/dsv41/test_expert_stream_timeline_cuda.py` (GPU)

**Interfaces:**
- Consumes: `EVENTS`, `window`, `decode_window`, `calibration`, `DEFAULT_CAPACITY`, `RECORD_WORDS` (Task 9);
  `global_ns`, `ld_acquire_sys`, `st_release_sys`, `kPending`, `kPendingEpoch` (lease_device.cuh, native-sync);
  `_device_wrappers`, `LAYOUTS` (expert_stream_transport.py).
- Produces:
  - C++ (timeline build only): `expert_stream_timeline_bind(head: int64[1] cuda, ring: int64[N,4] cuda)` and
    `expert_stream_timeline_calibrate(words: int32[2] pinned, gpu_ns: int64[R] cuda, host_ns: int64[R,2] cpu,
    rounds: int)`.
  - Python: `timeline_module(layout="exl3") -> Module`, cached, built with `-DEXPERT_STREAM_TIMELINE`.
  - Python: `class DeviceTimeline(device, capacity=DEFAULT_CAPACITY)` with `bind(module)`,
    `calibrate(module) -> dict` (rounds fixed at construction, default 32), `drain_async() -> None`, and
    `poll(final=False) -> Optional[TimelineDrain]`.

- [ ] **Step 1: Write the failing source tests** (append to `test_expert_stream_timeline.py`)

```python
import re

from sglang.test.expert_stream_sources import MOE


def _header(name):
    return (MOE / "expert_stream" / name).read_text()


def test_the_header_event_codes_are_the_decoders():
    codes = dict(re.findall(r"constexpr uint32_t kTimeline(\w+) = (\d+);", _header("lease_device.cuh")))
    snake = {re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower(): int(code) for name, code in codes.items()}
    assert snake == {name: code for code, name in tl.EVENTS.items()}


def test_every_stamp_is_compiled_only_in_the_timeline_build():
    # A stamp outside `#ifdef EXPERT_STREAM_TIMELINE` would change the production kernels.
    for name in ("lease_device.cuh", "lease_kernels.cuh", "row_copy_kernels.cuh"):
        depth, inside = 0, []
        for number, line in enumerate(_header(name).splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#if"):
                depth += 1
                inside.append(stripped == "#ifdef EXPERT_STREAM_TIMELINE")
            elif stripped.startswith("#endif"):
                depth -= 1
                inside.pop()
            elif "timeline_stamp(" in stripped or "timeline_generation(" in stripped or "g_timeline" in stripped:
                assert any(inside), f"{name}:{number} stamps outside #ifdef EXPERT_STREAM_TIMELINE"
```

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py -q -k "event_codes or stamp"`
Expected: `test_the_header_event_codes_are_the_decoders` fails (no constants yet). The stamp test passes
vacuously; it becomes the guard once the stamps exist.

- [ ] **Step 2: Add the event constants and the timeline core to `lease_device.cuh`**, after `global_ns()`:

```cpp
// Device stage timeline (plan 2026-09-27-expert-stream-transfer-measurement, Part C). Diagnostic only: everything
// below and every stamp site is compiled only with -DEXPERT_STREAM_TIMELINE (expert_stream_timeline.timeline_module),
// so the default build's SASS is unchanged. Record format: expert_stream_timeline.py, whose test reads these codes.
constexpr uint32_t kTimelinePostBefore = 1;
constexpr uint32_t kTimelinePostPublish = 2;
constexpr uint32_t kTimelineW1Enter = 3;
constexpr uint32_t kTimelineW1Exit = 4;
constexpr uint32_t kTimelineSEnter = 5;
constexpr uint32_t kTimelineSAdmit = 6;
constexpr uint32_t kTimelineSSeen = 7;
constexpr uint32_t kTimelineSCopied = 8;
constexpr uint32_t kTimelineSExit = 9;
constexpr uint32_t kTimelineSCommit = 10;
constexpr uint32_t kTimelineCwEnter = 11;
constexpr uint32_t kTimelineCwDone = 12;
constexpr uint32_t kTimelineFinalize = 13;
constexpr int64_t kTimelineRowNone = 0xFFFF;  // CW and F take no row; the join keys them by generation

#ifdef EXPERT_STREAM_TIMELINE
struct TimelineRing {
  unsigned long long* head;
  uint64_t* ring;
  uint64_t capacity;
};
__device__ TimelineRing g_timeline;  // bound once before capture (expert_stream_timeline_bind); null ring: no stamps

SGL_DEVICE uint64_t timeline_generation(const int32_t* state) {
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  return seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
}

// One record. The clock is read first, so a stamp is the event's time and not the atomic's. Plain stores, not
// st_relaxed_sys: nothing reads the ring while a kernel runs, since the drain is a copy stream-ordered after the replay.
SGL_DEVICE void timeline_stamp(uint32_t event, uint64_t generation, int64_t row, int lane, uint32_t bits, uint32_t extra) {
  const uint64_t now = global_ns();
  const TimelineRing t = g_timeline;
  if (t.ring == nullptr) return;
  const uint64_t ticket = atomicAdd(t.head, 1ull);
  uint64_t* r = t.ring + 4 * (ticket % t.capacity);
  r[1] = generation;
  r[2] = static_cast<uint64_t>(event & 0xFFu) | (static_cast<uint64_t>(blockIdx.x & 0xFFu) << 8) |
         (static_cast<uint64_t>(static_cast<uint32_t>(lane) & 0xFFu) << 16) |
         (static_cast<uint64_t>(bits & 0xFFu) << 24) | (static_cast<uint64_t>(row & 0xFFFF) << 32) |
         (static_cast<uint64_t>(extra & 0xFFFFu) << 48);
  r[3] = now;
  r[0] = ticket + 1;
}
#endif
```

Then, **after** the closing `}  // namespace device::expert_stream`, inside `namespace sglang`, add the host side:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
// Round-trip calibration: per round the host stamps t0, releases words[1] = r; this kernel sees it, stamps the GPU
// clock and releases words[0] = r; the host stamps t1 on seeing it. expert_stream_timeline.calibration reduces it.
__global__ void expert_stream_timeline_calibrate_kernel(uint8_t* words, uint64_t* gpu_ns, int64_t rounds) {
  using namespace device::expert_stream;
  for (int64_t r = 1; r <= rounds; ++r) {
    while (ld_acquire_sys(words + 4) != static_cast<uint32_t>(r)) {
    }
    gpu_ns[r - 1] = global_ns();
    st_release_sys(words, static_cast<uint32_t>(r));
  }
}

inline void expert_stream_timeline_bind(tvm::ffi::TensorView head, tvm::ffi::TensorView ring) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto capacity = SymbolicSize{"capacity"};
  expert_stream::verify_named("head", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), head);
  expert_stream::verify_named(
      "ring", TensorMatcher({capacity, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), ring);
  const device::expert_stream::TimelineRing t{
      static_cast<unsigned long long*>(head.data_ptr()), static_cast<uint64_t*>(ring.data_ptr()),
      static_cast<uint64_t>(ring.size(0))};
  CHECK_CUDA(cudaMemcpyToSymbol(device::expert_stream::g_timeline, &t, sizeof(t))) << "timeline bind";
}

inline void expert_stream_timeline_calibrate(
    tvm::ffi::TensorView words, tvm::ffi::TensorView gpu_ns, tvm::ffi::TensorView host_ns, int64_t rounds) {
  using namespace host;
  RuntimeCheck(rounds > 0 && gpu_ns.size(0) >= rounds && host_ns.size(0) >= rounds, "calibrate: too few slots");
  auto* w = static_cast<uint32_t*>(words.data_ptr());
  __atomic_store_n(w, 0u, __ATOMIC_RELEASE);
  __atomic_store_n(w + 1, 0u, __ATOMIC_RELEASE);
  const auto stream = LaunchKernel::resolve_device(gpu_ns.device());
  LaunchKernel(1, 1, stream)(
      expert_stream_timeline_calibrate_kernel, static_cast<uint8_t*>(words.data_ptr()),
      static_cast<uint64_t*>(gpu_ns.data_ptr()), rounds);
  auto* out = static_cast<int64_t*>(host_ns.data_ptr());
  for (int64_t r = 1; r <= rounds; ++r) {
    timespec t0{}, t1{}, start{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    __atomic_store_n(w + 1, static_cast<uint32_t>(r), __ATOMIC_RELEASE);
    while (__atomic_load_n(w, __ATOMIC_ACQUIRE) != static_cast<uint32_t>(r)) {
      clock_gettime(CLOCK_MONOTONIC, &t1);
      RuntimeCheck(t1.tv_sec - start.tv_sec < 2, "calibrate: the kernel stopped answering");
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    out[2 * (r - 1)] = static_cast<int64_t>(t0.tv_sec) * 1000000000LL + t0.tv_nsec;
    out[2 * (r - 1) + 1] = static_cast<int64_t>(t1.tv_sec) * 1000000000LL + t1.tv_nsec;
  }
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "calibrate";
}
#endif
```

Add `#include <ctime>` to `lease_device.cuh`'s includes, and `#include "tensor_checks.h"` if `lease_device.cuh`
does not already see `expert_stream::verify_named`: `lease_kernels.cuh` includes it after `lease_device.cuh`, so
include it explicitly.

- [ ] **Step 3: Add the stamp sites.** Each is its own `#ifdef EXPERT_STREAM_TIMELINE` block, with no change to
  the surrounding code.

`lease_kernels.cuh`, `exl3_ram_miss_post_kernel`, immediately before
`st_release_sys(page + kDemandHead, seq);`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  const uint64_t tl_generation = (static_cast<uint64_t>(static_cast<uint32_t>(state[kEpoch])) << 32) | seq;
  timeline_stamp(kTimelinePostBefore, tl_generation, row, 0, armed ? 1u : 0u, lanes);
#endif
```

and immediately after it:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  timeline_stamp(kTimelinePostPublish, tl_generation, row, 0, armed ? 1u : 0u, lanes);
#endif
```

`exl3_ram_miss_lease_stream_hit_wait_kernel`, right after `if (threadIdx.x != 0) return;`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  const uint64_t tl_generation = device::expert_stream::timeline_generation(state);
  device::expert_stream::timeline_stamp(device::expert_stream::kTimelineW1Enter, tl_generation, row, 0, 0, 0);
#endif
```

and after the `lease_hit_wait_body(...)` call:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  device::expert_stream::timeline_stamp(
      device::expert_stream::kTimelineW1Exit, tl_generation, row, 0, 0, static_cast<uint32_t>(go_1[0]));
#endif
```

`exl3_ram_miss_lease_finalize_kernel`, right after `const bool served = ...;`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  timeline_stamp(kTimelineFinalize, generation, kTimelineRowNone, 0, served ? 1u : 0u, static_cast<uint32_t>(copied));
#endif
```

`row_copy_kernels.cuh`, `exl3_ram_miss_lease_stream_kernel`. Make these five insertions:

1. After the `__syncthreads();` that follows the `tid < kLeaseLanes` shared-state initialisation:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  if (tid == 0) timeline_stamp(kTimelineSEnter, generation, row, 0, sh.aborting, static_cast<uint32_t>(unclaimed));
#endif
```

2. In the leader-lane block, replace nothing. Around the admission statement
`if (sh.admitted[tid] == 0 && !stream_admit(...)) { sh.identity = 1; }` put, immediately before it:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
      const int32_t tl_was_admitted = sh.admitted[tid];
#endif
```

and immediately after it:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
      if (blockIdx.x == 0 && tl_was_admitted == 0 && sh.admitted[tid] != 0) {
        timeline_stamp(kTimelineSAdmit, generation, row, tid, sh.loading[tid], static_cast<uint32_t>(planned[tid]));
      }
#endif
```

and after `sh.todo[tid] = bits & ~sh.done[tid];` (inside the `if (sh.admitted[tid] != 0)` block):

```cpp
#ifdef EXPERT_STREAM_TIMELINE
        if (blockIdx.x == 0 && sh.todo[tid] != 0) {
          timeline_stamp(kTimelineSSeen, generation, row, tid, sh.todo[tid], sh.loading[tid] == 0 ? 1u : 0u);
        }
#endif
```

(`extra` 1 marks a READY lane: every piece was already there, so it has no publish-to-seen latency.)

3. In the `if (tid == 0)` block after the copy barrier, before its `for (int lane ...)` loop that folds `todo` into
`done`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
      for (int lane = 0; lane < kLeaseLanes; ++lane) {
        if (sh.todo[lane] != 0) timeline_stamp(kTimelineSCopied, generation, row, lane, sh.todo[lane], 0);
      }
#endif
```

4. After `if (tid != 0) return;` that follows the main loop:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  timeline_stamp(kTimelineSExit, generation, row, 0, sh.aborting, static_cast<uint32_t>(min(polls, int64_t{65535})));
#endif
```

5. After `const bool commit = ...;`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  timeline_stamp(kTimelineSCommit, generation, row, 0, commit ? 1u : 0u, 0);
#endif
```

`exl3_ram_miss_lease_copy_wait_kernel`, after `state[kCopyWaits] += 1;`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  timeline_stamp(kTimelineCwEnter, generation, kTimelineRowNone, 0, 0, mask);
#endif
```

After `if (word != expected) state[kCopySpun] += 1;`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  const uint32_t tl_spun = word != expected ? 1u : 0u;
#endif
```

and immediately before `go_ce[0] = __popc(mask);`:

```cpp
#ifdef EXPERT_STREAM_TIMELINE
  timeline_stamp(kTimelineCwDone, generation, kTimelineRowNone, 0, tl_spun, 0);
#endif
```

(A CW that failed reaches no `cw_done` stamp. The join reports it as `cw_enter` without `cw_done`.)

- [ ] **Step 4: Add `timeline_module` and `DeviceTimeline` to `expert_stream_timeline.py`**

```python
import torch

from sglang.kernels.jit.utils import cache_once

TIMELINE_WRAPPERS = [("expert_stream_timeline_bind", "expert_stream_timeline_bind"),
                     ("expert_stream_timeline_calibrate", "expert_stream_timeline_calibrate")]


def timeline_module(layout: str = "exl3"):
    return _timeline_module_cached(layout)


@cache_once
def _timeline_module_cached(layout: str):
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.ops.moe.expert_stream_transport import LAYOUTS, _device_wrappers

    return load_jit(f"expert_stream_{layout}", "timeline", cuda_files=[LAYOUTS[layout].device_source],
                    cuda_wrappers=_device_wrappers(layout) + TIMELINE_WRAPPERS,
                    extra_cuda_cflags=["-DEXPERT_STREAM_TIMELINE"])


class DeviceTimeline:
    """The ring, its pinned drain buffers and the calibration buffers, all allocated here, before any capture."""

    def __init__(self, device, capacity: int = DEFAULT_CAPACITY, rounds: int = 32) -> None:
        self.capacity = capacity
        self.head = torch.zeros(1, dtype=torch.int64, device=device)
        self.ring = torch.zeros((capacity, RECORD_WORDS), dtype=torch.int64, device=device)
        self._head_host = torch.zeros(1, dtype=torch.int64).pin_memory()
        self._ring_host = torch.zeros((capacity, RECORD_WORDS), dtype=torch.int64).pin_memory()
        self._words = torch.zeros(2, dtype=torch.int32).pin_memory()
        self._gpu_ns = torch.zeros(rounds, dtype=torch.int64, device=device)
        self._host_ns = torch.zeros((rounds, 2), dtype=torch.int64)
        self._event = None
        self._prev_head = 0

    def bind(self, module) -> None:
        module.expert_stream_timeline_bind(self.head, self.ring)

    def calibrate(self, module) -> dict:
        """Synchronizes the current stream. Never call it while capturing."""
        rounds = int(self._gpu_ns.numel())
        module.expert_stream_timeline_calibrate(self._words, self._gpu_ns, self._host_ns, rounds)
        return calibration(self._gpu_ns.cpu().tolist(), self._host_ns.tolist())

    def drain_async(self) -> None:
        """Queue a copy of head and ring on the current stream, after everything already on it (a replay boundary).
        At most one drain is in flight; call poll() to collect it."""
        if self._event is not None:
            return
        self._head_host.copy_(self.head, non_blocking=True)
        self._ring_host.copy_(self.ring, non_blocking=True)
        self._event = torch.cuda.Event()
        self._event.record()

    def poll(self, *, final: bool = False):
        if self._event is None:
            return None
        if final:
            self._event.synchronize()
        elif not self._event.query():
            return None
        head = int(self._head_host[0])
        first, _ = window(head, self._prev_head, self.capacity)
        index = torch.arange(first, head, dtype=torch.int64) % self.capacity
        drain = decode_window(self._ring_host[index].tolist(), first, head, self._prev_head)
        self._prev_head, self._event = head, None
        return drain
```

- [ ] **Step 5: Write the GPU test** `test/manual/dsv41/test_expert_stream_timeline_cuda.py`

```python
"""The device timeline against the real chain and service (GPU). Run on divix01 holding cc-gpu.lock, with PYTHONPATH
pointing at the tree under test.

- every stage stamps, in order, for a request with RAM misses; every S block stamps its copies
- a ring too small for the traffic reports the overwrite and keeps only the newest tickets
- a timeline build whose ring was never bound stamps nothing and still delivers the right bytes
- the calibration's error is under a millisecond and two calibrations agree within their errors
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_piece_stream_cuda import TOP_K, StreamService  # noqa: E402

from sglang.kernels.ops.moe import expert_stream_timeline as tl  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _retired(s):
    c = s.counters()
    return c["leases_granted"] == c["leases_acked"] + c["leases_voided"] + c["leases_copied"]


def _service(tmp_path, capacity=tl.DEFAULT_CAPACITY, bind=True):
    s = StreamService(tmp_path)
    s.dev._module = tl.timeline_module()
    timeline = tl.DeviceTimeline("cuda", capacity=capacity)
    if bind:
        timeline.bind(s.dev._module)
    s.plan([])
    s.step()  # compiles and loads the timeline build before anything is measured
    torch.cuda.synchronize()
    timeline.drain_async()
    timeline.poll(final=True)
    return s, timeline


def test_every_stage_stamps_in_order_and_every_s_block_stamps_its_copies(tmp_path):
    s, timeline = _service(tmp_path)
    try:
        experts = list(range(TOP_K))  # a fresh service: every lane a RAM miss
        s.plan(experts)
        s.step()
        torch.cuda.synchronize()
        assert s.until(lambda: _retired(s))
        timeline.drain_async()
        drain = timeline.poll(final=True)
        assert (drain.lost, drain.torn) == (0, 0)
        seqs = {e.seq for e in drain.events if e.event == "post_publish"}
        (seq,) = seqs
        mine = [e for e in drain.events if e.seq == seq]
        names = [e.event for e in mine]
        for name in ("post_before", "post_publish", "w1_enter", "w1_exit", "s_enter", "s_admit", "s_seen",
                     "s_copied", "s_exit", "s_commit", "finalize"):
            assert name in names, name
        first = {name: min(e.gpu_ns for e in mine if e.event == name) for name in set(names)}
        assert first["post_before"] <= first["post_publish"] <= first["w1_enter"] <= first["w1_exit"] <= first["s_enter"]
        assert first["s_enter"] <= first["s_seen"] <= first["s_copied"] <= first["s_commit"] <= first["finalize"]
        assert {e.block for e in mine if e.event == "s_copied"} == set(range(8))
        for lane in {e.lane for e in mine if e.event == "s_admit"}:
            seen = 0
            for e in mine:
                if e.event == "s_seen" and e.lane == lane:
                    seen |= e.bits
            assert seen == 0xFF, (lane, bin(seen))
        assert {e.extra for e in mine if e.event == "s_admit"} <= set(experts)
    finally:
        s.close()


def test_overwrite_is_detected(tmp_path):
    s, timeline = _service(tmp_path, capacity=16)
    try:
        for start in (0, 6):
            s.plan([(start + k) % 16 for k in range(TOP_K)])
            s.step()
            torch.cuda.synchronize()
            assert s.until(lambda: _retired(s))
        timeline.drain_async()
        drain = timeline.poll(final=True)
        assert drain.lost > 0
        tickets = [e.ticket for e in drain.events]
        assert tickets == list(range(drain.head - 16, drain.head)) and drain.torn == 0
    finally:
        s.close()


def test_an_unbound_timeline_build_stamps_nothing_and_copies_right(tmp_path):
    s, timeline = _service(tmp_path, bind=False)
    try:
        experts = list(range(TOP_K))
        s.plan(experts)
        s.step()
        torch.cuda.synchronize()
        assert s.until(lambda: _retired(s))
        assert int(timeline.head.item()) == 0
        want = s.expected(experts)
        for lane, expert in enumerate(experts):
            for n in s.names:
                assert torch.equal(s.dest[n][lane].cpu().view(torch.uint8), want[expert][n].view(torch.uint8))
    finally:
        s.close()


def test_calibration_is_tight_and_repeatable(tmp_path):
    timeline = tl.DeviceTimeline("cuda")
    module = tl.timeline_module()
    one, two = timeline.calibrate(module), timeline.calibrate(module)
    assert 0 < one["error_ns"] < 1_000_000 and 0 < two["error_ns"] < 1_000_000
    assert abs(one["offset_ns"] - two["offset_ns"]) <= one["error_ns"] + two["error_ns"] + 50_000
```

- [ ] **Step 6: Run the CPU tests locally, then the SASS gate and the GPU tests on divix01**

```bash
PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py -q -p no:randomly; echo "EXIT=$?"
git add -A python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh python/sglang/kernels/ops/moe/expert_stream_timeline.py \
  test/registered/unit/layers/moe/test_expert_stream_timeline.py test/manual/dsv41/test_expert_stream_timeline_cuda.py
git commit -m "feat(expert-stream): device stage timeline stamps under EXPERT_STREAM_TIMELINE, bind and clock calibration"
git push origin expert-stream-device-timeline
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-timeline && git fetch origin && git checkout --detach origin/expert-stream-device-timeline \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-timeline-base tl-base.sass \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-timeline tl-after.sass \
  && python3 sass_gate.py diff tl-base.sass tl-after.sass --exact; echo GATE_EXIT=$?'
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-timeline && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/manual/dsv41/test_expert_stream_timeline_cuda.py test/manual/dsv41/test_exl3_piece_stream_cuda.py \
     test/manual/dsv41/test_exl3_copy_engine_cuda.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected:
- The local run shows every test passing, including both source tests.
- `GATE PASS` and `GATE_EXIT=0`: the default build is unchanged, which is the "costs nothing when disabled" proof.
  The graph is unchanged too, since the default module and its launch shapes are the same objects.
- The printed path lies under `wt-timeline/python/`.
- `EXIT=0`, with the two existing files' counts equal to their counts at `wt-timeline-base`. Run the same command
  there to get that baseline, and record both commands and counts.

If `sass_gate.py build` builds the module under a different name than the timeline build uses, the gate still
compares only the default module, which is intended.

### Task 12: Wiring: env var, the device side, the service drain, the trace lines

**Files:**
- Modify: `python/sglang/srt/environ.py` (next to `SGLANG_DSV41_EXPERT_TRACE_PATH`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, `_kernels`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`check_device_timeline`, `ensure_started`, `attach`,
  `_trace_step`)
- Modify: `python/sglang/srt/layers/moe/exl3_stream_trace.py` (`DEVICE_TIMELINE_SCHEMA`, `record_device_timeline`,
  `record_clock_calibration`)
- Test: `test/registered/unit/layers/moe/test_expert_stream_timeline.py`

**Interfaces:**
- Consumes: `DeviceTimeline`, `timeline_module`, `TimelineDrain` (Tasks 9, 11).
- Produces:
  - `envs.SGLANG_DSV41_DEBUG_DEVICE_TIMELINE` (`EnvBool(False)`).
  - `check_device_timeline(enabled: bool, trace_enabled: bool) -> None`, which raises `RuntimeError`.
  - `ExpertStreamDevice(..., timeline: Optional[DeviceTimeline] = None)` with attributes `.timeline` and
    `.timeline_calibration: Optional[dict]`.
  - `Exl3StreamTrace.record_device_timeline(drain: TimelineDrain) -> None`, which writes
    `{"kind": "device_timeline", "schema": 1, "forward", "head", "lost", "torn", "events": [[ticket, generation,
    event, block, lane, bits, row, extra, gpu_ns], ...], "t"}`.
  - `Exl3StreamTrace.record_clock_calibration(cal: dict) -> None`, which writes
    `{"kind": "clock_calibration", "schema": 1, "offset_ns", "error_ns", "rounds", "host_ns", "t"}`.

- [ ] **Step 1: Write the failing tests** (append to `test_expert_stream_timeline.py`)

```python
import json

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamDevice, new_page
from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_ram_miss import check_device_timeline
from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace


def test_the_timeline_is_off_by_default():
    assert envs.SGLANG_DSV41_DEBUG_DEVICE_TIMELINE.get() is False


def test_the_timeline_without_the_stage_trace_is_refused():
    with pytest.raises(RuntimeError, match="SGLANG_DSV41_EXPERT_TRACE_PATH"):
        check_device_timeline(True, False)
    check_device_timeline(True, True)
    check_device_timeline(False, False)


class _FakeTimeline:
    def __init__(self):
        self.calls = []

    def bind(self, module):
        self.calls.append(("bind", module))

    def calibrate(self, module):
        self.calls.append(("calibrate", module))
        return {"offset_ns": 1, "error_ns": 2, "rounds": 3, "host_ns": 4}


def test_timeline_is_bound_and_calibrated_at_construction(monkeypatch):
    # Before any capture and before the copy engine arms: a first kernel launch after arming can stall the copy
    # thread (LEASE_PROTOCOL.md 7.6), so the timeline module's kernels must all have run here.
    import torch

    import sglang.kernels.ops.moe.expert_stream_timeline as timeline_ops

    sentinel = object()
    monkeypatch.setattr(timeline_ops, "timeline_module", lambda layout="exl3": sentinel)
    fake = _FakeTimeline()
    dev = ExpertStreamDevice(new_page(pin=False), torch.zeros((1, 4), dtype=torch.int32), device="cpu", layers=1,
                             timeout_ms=1, advise=False, timeline=fake)
    assert fake.calls == [("bind", sentinel), ("calibrate", sentinel)]
    assert dev._kernels() is sentinel and dev.timeline_calibration["error_ns"] == 2


def test_the_trace_writes_timeline_and_calibration_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = Exl3StreamTrace(str(path))
    event = tl.TimelineEvent(4, (1 << 32) | 9, "s_seen", 0, 2, 0x0F, 3, 0, 123)
    trace.record_device_timeline(tl.TimelineDrain([event], 5, 0, 0))
    trace.record_clock_calibration({"offset_ns": 7, "error_ns": 8, "rounds": 32, "host_ns": 9})
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    timeline = next(line for line in lines if line["kind"] == "device_timeline")
    assert timeline["schema"] == 1 and timeline["events"] == [[4, (1 << 32) | 9, "s_seen", 0, 2, 0x0F, 3, 0, 123]]
    assert (timeline["head"], timeline["lost"], timeline["torn"]) == (5, 0, 0)
    cal = next(line for line in lines if line["kind"] == "clock_calibration")
    assert (cal["offset_ns"], cal["error_ns"]) == (7, 8)
```

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py -q`
Expected: the four new tests fail: the env var, `check_device_timeline`, the `timeline` argument and the trace
methods do not exist yet.

- [ ] **Step 2: The env var** (in `environ.py`, directly after `SGLANG_DSV41_ROUTER_CAPTURE_PATH`)

```python
    # Diagnostic: build the RAM-miss chain's kernels with the device stage timeline (-DEXPERT_STREAM_TIMELINE), stamp
    # every stage and piece into a device ring, drain it at each trace step and calibrate the GPU clock to the host's
    # (expert_stream_timeline.py). Needs SGLANG_DSV41_EXPERT_TRACE_PATH, which the lines are written to. Adds a drain
    # copy per step: a run with it on is not a throughput run.
    SGLANG_DSV41_DEBUG_DEVICE_TIMELINE = EnvBool(False)
```

- [ ] **Step 3: `ExpertStreamDevice`**

Add `timeline=None` as the last keyword parameter of `__init__`. At the end of `__init__`, add:

```python
        # Device stage timeline (diagnostic): bound and calibrated here, before any capture and before the copy engine
        # can arm, so every kernel of the timeline build has been loaded and launched once (LEASE_PROTOCOL.md 7.6).
        self.timeline = timeline
        self.timeline_calibration = None
        if timeline is not None:
            from sglang.kernels.ops.moe import expert_stream_timeline

            self._module = expert_stream_timeline.timeline_module(self._layout)
            timeline.bind(self._module)
            self.timeline_calibration = timeline.calibrate(self._module)
```

`_kernels()` needs no change: it returns `self._module` once set.

- [ ] **Step 4: The service**

In `exl3_ram_miss.py`, add at module level:

```python
def check_device_timeline(enabled: bool, trace_enabled: bool) -> None:
    """SGLANG_DSV41_DEBUG_DEVICE_TIMELINE writes into the stage trace; refuse it without one."""
    if enabled and not trace_enabled:
        raise RuntimeError(
            "exl3 RAM miss: SGLANG_DSV41_DEBUG_DEVICE_TIMELINE needs SGLANG_DSV41_EXPERT_TRACE_PATH: the device "
            "timeline is written into the stage trace"
        )
```

In `ensure_started`, directly after the router-capture refusal:

```python
        check_device_timeline(envs.SGLANG_DSV41_DEBUG_DEVICE_TIMELINE.get(), get_exl3_stream_trace().enabled)
```

In `attach`, where `ExpertStreamDevice(` is constructed, add `timeline=self._new_timeline(cache.device),` to its
keyword arguments. After the construction, add:

```python
            if self.device_side.timeline_calibration is not None:
                from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

                get_exl3_stream_trace().record_clock_calibration(self.device_side.timeline_calibration)
```

Add the methods:

```python
    TIMELINE_RECALIBRATE_DRAINS = 256  # the clocks drift; ~0.5 ms of round trips every 256 steps

    def _new_timeline(self, device):
        if not envs.SGLANG_DSV41_DEBUG_DEVICE_TIMELINE.get():
            return None
        from sglang.kernels.ops.moe.expert_stream_timeline import DeviceTimeline

        self._timeline_drains = 0
        return DeviceTimeline(device)

    def _drain_timeline(self, trace, *, final: bool) -> None:
        timeline = getattr(self.device_side, "timeline", None) if self.device_side is not None else None
        if timeline is None:
            return
        drain = timeline.poll(final=final)
        if drain is not None:
            trace.record_device_timeline(drain)
            self._timeline_drains += 1
        if final or torch.cuda.is_current_stream_capturing():
            return
        if self._timeline_drains and self._timeline_drains % self.TIMELINE_RECALIBRATE_DRAINS == 0:
            trace.record_clock_calibration(timeline.calibrate(self.device_side._kernels()))
        timeline.drain_async()
```

In `_trace_step`, after the `if not trace.enabled: return` guard, add `self._drain_timeline(trace, final=final)`.
`Exl3RamMissService.__init__` already sets `self.device_side = None`, so `_drain_timeline` is safe before `attach`.

- [ ] **Step 5: The trace methods** (in `exl3_stream_trace.py`, next to `record_ram_miss_requests`)

```python
DEVICE_TIMELINE_SCHEMA = 1  # expert_stream_timeline.py's record format; bump with it


    def record_device_timeline(self, drain) -> None:
        """One drained window of the device stage timeline (SGLANG_DSV41_DEBUG_DEVICE_TIMELINE). Every gpu_ns is the
        GPU's %globaltimer: map it with a clock_calibration line (host = gpu - offset_ns), never subtract it from a
        host stamp raw. `lost` tickets were overwritten before this drain, and `torn` records were refused."""
        if self._file is None:
            return
        line = {"kind": "device_timeline", "schema": DEVICE_TIMELINE_SCHEMA, "forward": self.forwards,
                "head": drain.head, "lost": drain.lost, "torn": drain.torn,
                "events": [[e.ticket, e.generation, e.event, e.block, e.lane, e.bits, e.row, e.extra, e.gpu_ns]
                           for e in drain.events],
                "t": round(time.monotonic(), 6)}
        self._file.write(json.dumps(line) + "\n")

    def record_clock_calibration(self, cal: dict) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps({"kind": "clock_calibration", "schema": DEVICE_TIMELINE_SCHEMA, **cal,
                                     "t": round(time.monotonic(), 6)}) + "\n")
```

The first line is module level, next to `RAM_MISS_TRACE_SCHEMA`. The two methods go inside the trace class.

- [ ] **Step 6: Run the CPU tests**

```bash
PYTHONPATH=$PWD/python taskset -c 0-63 python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline.py \
  test/registered/unit/layers/moe/test_exl3_stream_trace.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: all pass, `EXIT=0`. Record the command and counts.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/kernels/ops/moe/expert_stream_transport.py \
  python/sglang/srt/layers/moe/exl3_ram_miss.py python/sglang/srt/layers/moe/exl3_stream_trace.py \
  test/registered/unit/layers/moe/test_expert_stream_timeline.py
git commit -m "feat(expert-stream): SGLANG_DSV41_DEBUG_DEVICE_TIMELINE wiring: bind and calibrate at construction, drain per trace step"
```

### Task 13: The join, the per-request and per-step report, and the divix01 run

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_stream_timeline_join.py`
- Create: `analysis/dsv41-drive/stream-timeline/timeline_report.py`
- Create: `analysis/dsv41-drive/stream-timeline/README.md`
- Create: `analysis/dsv41-drive/stream-timeline/results.md`
- Test: `test/registered/unit/layers/moe/test_expert_stream_timeline_join.py`

**Interfaces:**
- Consumes: trace lines `ram_miss_request` (schema 8: `request.seq`, `request.type`, `layer`, `stages_ns.observed`,
  `pieces[k].publish_ns`, `pieces[k].expert`); `device_timeline` and `clock_calibration` (schema 1).
- Produces:
  - `load(lines: list[dict]) -> Joined`.
  - `request_report(req: JoinedRequest) -> dict`, with keys `seq, row, post_to_observed_ns, w1_ns, s_wall_ns,
    copy_ns, latency_exposed_ns, host_wait_ns, tail_ns, cw_wait_ns, publish_to_seen_ns: list[int],
    pass_samples: list[tuple[int, int]], error_ns`.
  - `summarize(joined: Joined) -> dict`, with keys `coverage, calibration_error_ns, gpu_tick_ns, lost, torn,
    requests, per_step, fit {pass_fixed_ns, per_piece_ns, n}, medians`.
  - `class LostRecords(Exception)`.

Definitions the report states, all in one request's block-0 S window `W = [s_enter, s_exit]`, mapped to the host
clock:
- `publish_to_seen_ns`, per piece of a LOADING lane: `host(s_seen) - publish_ns`. It includes the flag visibility
  and S's poll cadence, and carries ± `error_ns`.
- `copy_ns = |∪ [seen_j, copied_j]|`, where `copied_j` is the maximum over blocks. It is GPU-only, with no clock
  error.
- `latency_exposed_ns = |∪ [publish_j, seen_j] \ copy intervals|`: time a piece was published and S was neither
  seeing it nor copying anything.
- `host_wait_ns = |W \ ∪ [publish_j, copied_j]|`: nothing published and uncopied, so S waits on the host.
- `tail_ns = s_exit - max_j publish_j`.
- `pass_samples`: per (lane, pass), `(pieces in that pass, copied_max - seen)`. A least-squares fit over all
  samples gives `pass_fixed_ns` (the intercept: per-pass fixed cost) and `per_piece_ns` (the slope). These are the
  granularity numbers in the Decision table.
- `cw_wait_ns = cw_done - cw_enter` when both exist.
- `post_to_observed_ns = observed - host(post_publish)`.

Join rules:
- Device requests are keyed by `(seq, row)`, with `row` taken from `post_publish`.
- Host records match only when `type == "demand"`; advisory and touch records are excluded.
- A lane's host row ordinal is the piece entry whose `expert` equals the lane's `s_admit.extra`. Two lanes with
  one expert share it.
- READY lanes (`s_seen.extra == 1`) have no publish-to-seen and are counted apart.
- A device request with no host record is counted unmatched. Coverage = matched / device requests with an armed
  `post_publish` (`bits == 1`).

- [ ] **Step 1: Write the failing tests**

```python
"""Joining the device stage timeline with the host stage trace, and the partition of S's time (CPU)."""

import pytest

from sglang.srt.layers.moe import expert_stream_timeline_join as join
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

GEN = (0 << 32) | 5  # seq 5
OFFSET = 1_000_000  # gpu = host + OFFSET


def _ev(ticket, event, gpu_host_ns, lane=0, bits=0, extra=0, block=0, row=3):
    return [ticket, GEN, event, block, lane, bits, row, extra, gpu_host_ns + OFFSET]


def _lines(extra_events=(), host_type="demand", lost=0):
    events = [
        _ev(0, "post_before", 100, bits=1, extra=1), _ev(1, "post_publish", 110, bits=1, extra=1),
        _ev(2, "w1_enter", 200), _ev(3, "w1_exit", 300),
        _ev(4, "s_enter", 1_000),
        _ev(5, "s_admit", 1_010, lane=0, bits=1, extra=7),
        # piece 0 published at 1_500, seen at 2_000, copied by blocks 0 and 1 at 2_400 and 2_600
        _ev(6, "s_seen", 2_000, lane=0, bits=0b01),
        _ev(7, "s_copied", 2_400, lane=0, bits=0b01, block=0), _ev(8, "s_copied", 2_600, lane=0, bits=0b01, block=1),
        # piece 1 published at 4_000, seen at 4_100, copied at 4_900
        _ev(9, "s_seen", 4_100, lane=0, bits=0b10),
        _ev(10, "s_copied", 4_900, lane=0, bits=0b10, block=0),
        _ev(11, "s_exit", 5_000), _ev(12, "s_commit", 5_010, bits=1),
        _ev(13, "finalize", 5_100, bits=1, row=0xFFFF),
        *extra_events,
    ]
    host = {"kind": "ram_miss_request", "schema": 8, "layer": 10, "request": {"seq": 5, "type": host_type},
            "row": 3, "stages_ns": {"observed": 150}, "status": "served",
            "pieces": [{"row": 0, "expert": 7, "publish_ns": [1_500, 4_000] + [0] * 6}]}
    return [
        {"kind": "clock_calibration", "schema": 1, "offset_ns": OFFSET, "error_ns": 20, "rounds": 32, "host_ns": 0},
        {"kind": "device_timeline", "schema": 1, "forward": 1, "head": len(events), "lost": lost, "torn": 0,
         "events": events},
        host,
    ]


def test_the_partition_of_s_time_adds_up_and_the_numbers_are_the_definitions():
    joined = join.load(_lines())
    (req,) = joined.requests
    r = join.request_report(req)
    assert r["s_wall_ns"] == 4_000
    assert r["copy_ns"] == (2_600 - 2_000) + (4_900 - 4_100)
    assert r["latency_exposed_ns"] == (2_000 - 1_500) + (4_100 - 4_000)
    assert r["host_wait_ns"] == (1_500 - 1_000) + (4_000 - 2_600) + (5_000 - 4_900)
    assert r["copy_ns"] + r["latency_exposed_ns"] + r["host_wait_ns"] == r["s_wall_ns"]
    assert r["publish_to_seen_ns"] == [500, 100]
    assert r["tail_ns"] == 5_000 - 4_000
    assert r["post_to_observed_ns"] == 150 - 110
    assert r["w1_ns"] == 100 and r["error_ns"] == 20
    assert sorted(r["pass_samples"]) == [(1, 600), (1, 800)]


def test_the_fit_separates_fixed_per_pass_cost_from_per_piece_time():
    fit = join.fit_passes([(1, 150), (2, 250), (4, 450)])
    assert fit["pass_fixed_ns"] == pytest.approx(50) and fit["per_piece_ns"] == pytest.approx(100)


def test_advisory_records_do_not_join_and_coverage_says_so():
    s = join.summarize(join.load(_lines(host_type="advisory")))
    assert s["coverage"]["matched"] == 0 and s["coverage"]["device_requests"] == 1


def test_a_ready_lane_has_no_publish_to_seen():
    ready = [_ev(20, "s_admit", 1_020, lane=1, bits=0, extra=9), _ev(21, "s_seen", 1_030, lane=1, bits=0xFF, extra=1),
             _ev(22, "s_copied", 1_900, lane=1, bits=0xFF, block=0)]
    (req,) = join.load(_lines(ready)).requests
    r = join.request_report(req)
    assert r["publish_to_seen_ns"] == [500, 100] and r["ready_lanes"] == 1


def test_report_refuses_lost_records():
    with pytest.raises(join.LostRecords):
        join.summarize(join.load(_lines(lost=3)))
    assert join.summarize(join.load(_lines(lost=3)), allow_loss=True)["lost"] == 3
```

`record_ram_miss_requests` writes `layer` (a layer id) but not the streamed row index, which is what the device
stamps carry. Step 2 adds `"row": record["row"]` to the line. It is additive and falls under Task 10's schema-8
bump; Step 2 also amends that bump's comment.

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline_join.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 2: Write `expert_stream_timeline_join.py`**

```python
"""Join the device stage timeline (device_timeline lines) with the host stage trace (ram_miss_request, schema >= 8)
on one clock, and partition each request's S window. Definitions: plan 2026-09-27-expert-stream-transfer-measurement,
Task 13. Every cross-clock number carries the calibration error of the calibration nearest to it."""
from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from dataclasses import dataclass, field


class LostRecords(Exception):
    pass


@dataclass
class JoinedRequest:
    seq: int
    row: int
    forward: int
    events: list[list]  # device events of this generation: [ticket, generation, event, block, lane, bits, row, extra, gpu_ns]
    host: dict | None
    offset_ns: int
    error_ns: int


@dataclass
class Joined:
    requests: list[JoinedRequest] = field(default_factory=list)
    unmatched: int = 0
    lost: int = 0
    torn: int = 0
    calibrations: list[dict] = field(default_factory=list)


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(i for i in intervals if i[1] > i[0]):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _length(intervals) -> int:
    return sum(hi - lo for lo, hi in _union(list(intervals)))


def _minus(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for lo, hi in _union(a):
        cur = lo
        for blo, bhi in _union(b):
            if bhi <= cur or blo >= hi:
                continue
            if blo > cur:
                out.append((cur, blo))
            cur = max(cur, bhi)
        if cur < hi:
            out.append((cur, hi))
    return out


def _clip(intervals, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in intervals if min(b, hi) > max(a, lo)]


def load(lines: list[dict]) -> Joined:
    joined = Joined()
    cals = sorted((l for l in lines if l.get("kind") == "clock_calibration"), key=lambda c: c["host_ns"])
    if not cals:
        raise ValueError("no clock_calibration line: device and host clocks cannot be joined")
    joined.calibrations = cals
    host = {}
    for l in lines:
        if l.get("kind") == "ram_miss_request" and l.get("schema", 0) >= 8 and l["request"]["type"] == "demand":
            host[(l["request"]["seq"], l["row"])] = l
    for l in lines:
        if l.get("kind") != "device_timeline":
            continue
        joined.lost += l["lost"]
        joined.torn += l["torn"]
        by_gen = defaultdict(list)
        for e in l["events"]:
            by_gen[e[1]].append(e)
        for generation, events in by_gen.items():
            post = [e for e in events if e[2] == "post_publish"]
            if not post or post[0][5] != 1:
                continue  # unarmed (touch-only) or its post fell outside this drain
            seq, row = generation & 0xFFFFFFFF, post[0][6]
            gpu = post[0][8]
            keys = [c["host_ns"] + c["offset_ns"] for c in cals]  # each calibration's instant on the GPU clock
            k = min(max(bisect.bisect_left(keys, gpu), 0), len(cals) - 1)
            if k > 0 and abs(keys[k - 1] - gpu) < abs(keys[k] - gpu):
                k -= 1
            rec = host.get((seq, row))
            if rec is None:
                joined.unmatched += 1
            joined.requests.append(JoinedRequest(seq, row, l["forward"], sorted(events, key=lambda e: e[0]), rec,
                                                 cals[k]["offset_ns"], cals[k]["error_ns"]))
    return joined


def request_report(req: JoinedRequest) -> dict:
    h = lambda gpu: gpu - req.offset_ns  # noqa: E731  -- host clock of a GPU stamp
    ev = req.events
    first = lambda name, block=0: next((e for e in ev if e[2] == name and e[3] == block), None)  # noqa: E731
    out = {"seq": req.seq, "row": req.row, "forward": req.forward, "error_ns": req.error_ns, "ready_lanes": 0}
    post, w1_in, w1_out = first("post_publish"), first("w1_enter"), first("w1_exit")
    out["w1_ns"] = w1_out[8] - w1_in[8] if w1_in and w1_out else None
    out["post_to_observed_ns"] = (req.host["stages_ns"]["observed"] - h(post[8])
                                  if req.host and post and req.host["stages_ns"].get("observed") else None)
    cw_in, cw_done = first("cw_enter"), first("cw_done")
    out["cw_wait_ns"] = cw_done[8] - cw_in[8] if cw_in and cw_done else None
    out["cw_spun"] = bool(cw_done[5]) if cw_done else None
    s_in, s_out = first("s_enter"), first("s_exit")
    if not (s_in and s_out):
        out.update(s_wall_ns=None, copy_ns=None, latency_exposed_ns=None, host_wait_ns=None, tail_ns=None,
                   publish_to_seen_ns=[], pass_samples=[])
        return out
    lo, hi = h(s_in[8]), h(s_out[8])
    expert_of = {e[4]: e[7] for e in ev if e[2] == "s_admit" and e[3] == 0}
    publish = {}
    if req.host:
        for piece in req.host["pieces"]:
            for lane, expert in expert_of.items():
                if expert == piece["expert"]:
                    for j, t in enumerate(piece["publish_ns"]):
                        if t:
                            publish[(lane, j)] = t
    copied = defaultdict(int)  # (lane, piece) -> latest block's copied stamp
    for e in ev:
        if e[2] == "s_copied":
            for j in range(8):
                if e[5] >> j & 1:
                    copied[(e[4], j)] = max(copied[(e[4], j)], h(e[8]))
    seen, p2s, samples, ready = {}, [], [], set()
    for e in ev:
        if e[2] != "s_seen" or e[3] != 0:
            continue
        pieces = [j for j in range(8) if e[5] >> j & 1]
        if e[7] == 1:
            ready.add(e[4])
        for j in pieces:
            seen[(e[4], j)] = h(e[8])
            if e[7] != 1 and (e[4], j) in publish:
                p2s.append(seen[(e[4], j)] - publish[(e[4], j)])
        done = max((copied.get((e[4], j), 0) for j in pieces), default=0)
        if done:
            samples.append((len(pieces), done - h(e[8])))
    copy_iv = _clip([(seen[k], copied[k]) for k in seen if k in copied], lo, hi)
    latency_iv = _minus(_clip([(publish[k], seen[k]) for k in seen if k in publish], lo, hi), copy_iv)
    busy = _clip([(publish.get(k, seen[k]), copied[k]) for k in seen if k in copied], lo, hi)
    out.update(
        s_wall_ns=hi - lo,
        copy_ns=_length(copy_iv),
        latency_exposed_ns=_length(latency_iv),
        host_wait_ns=_length(_minus([(lo, hi)], busy)),
        tail_ns=hi - max(publish.values()) if publish else None,
        publish_to_seen_ns=p2s,
        pass_samples=samples,
        ready_lanes=len(ready),
    )
    return out


def fit_passes(samples: list[tuple[int, int]]) -> dict:
    """Least squares of (pieces in a pass) -> (pass time): intercept = fixed per-pass cost, slope = per-piece time."""
    n = len(samples)
    if n < 2 or len({x for x, _ in samples}) < 2:
        return {"pass_fixed_ns": None, "per_piece_ns": None, "n": n}
    mx = sum(x for x, _ in samples) / n
    my = sum(y for _, y in samples) / n
    slope = sum((x - mx) * (y - my) for x, y in samples) / sum((x - mx) ** 2 for x, _ in samples)
    return {"pass_fixed_ns": my - slope * mx, "per_piece_ns": slope, "n": n}


def summarize(joined: Joined, *, allow_loss: bool = False) -> dict:
    if joined.lost and not allow_loss:
        raise LostRecords(f"{joined.lost} device records were overwritten before a drain; rerun with a larger ring "
                          "or pass allow_loss to accept a partial window")
    reports = [request_report(r) for r in joined.requests if r.host is not None]
    per_step = defaultdict(lambda: defaultdict(int))
    for r in reports:
        for key in ("s_wall_ns", "copy_ns", "latency_exposed_ns", "host_wait_ns", "w1_ns", "cw_wait_ns"):
            if r.get(key) is not None:
                per_step[r["forward"]][key] += r[key]
    p2s = [x for r in reports for x in r["publish_to_seen_ns"]]
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    return {
        "coverage": {"device_requests": len(joined.requests), "matched": len(joined.requests) - joined.unmatched,
                     "pieces_with_publish": len(p2s)},
        "calibration_error_ns": max((c["error_ns"] for c in joined.calibrations), default=None),
        "gpu_tick_ns": min((b[8] - a[8] for r in joined.requests for a, b in zip(r.events, r.events[1:])
                            if a[3] == b[3] and b[8] > a[8]), default=None),
        "lost": joined.lost, "torn": joined.torn, "requests": reports,
        "per_step": {step: dict(v) for step, v in sorted(per_step.items())},
        "fit": fit_passes([s for r in reports for s in r["pass_samples"]]),
        "medians": {"publish_to_seen_ns": med(p2s),
                    "tail_ns": med([r["tail_ns"] for r in reports if r["tail_ns"] is not None]),
                    "copy_share": med([r["copy_ns"] / r["s_wall_ns"] for r in reports if r["s_wall_ns"]]),
                    "cw_spun_share": (sum(bool(r["cw_spun"]) for r in reports if r["cw_spun"] is not None)
                                      / max(1, sum(r["cw_spun"] is not None for r in reports)))},
    }
```

In `exl3_stream_trace.py`'s `record_ram_miss_requests`, add `"row": record["row"],` to `line` next to `"layer"`.
Also add "and the line's `row`, the streamed row index" to the schema-8 comment.

- [ ] **Step 3: Run the tests and see them pass**

```bash
PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/moe/test_expert_stream_timeline_join.py \
  test/registered/unit/kernels/test_exl3_ram_miss_trace_export.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: all pass. If the export test pins a line's exact key set, add `"row"` to its expectation in this commit.

- [ ] **Step 4: Write the CLI `analysis/dsv41-drive/stream-timeline/timeline_report.py`**

```python
#!/usr/bin/env python3
"""Per-request and per-step stage timeline of the lease chain from one trace run with
SGLANG_DSV41_DEBUG_DEVICE_TIMELINE=1 and SGLANG_DSV41_EXPERT_TRACE_PATH set.

    PYTHONPATH=<repo>/python python3 timeline_report.py <trace.jsonl> [--from-forward N] [--allow-loss] [--json out.json]

Prints coverage, calibration error, loss, the pass fit (fixed per-pass cost, per-piece copy time), medians, and the
Decision-table ratios of plan 2026-09-27-expert-stream-transfer-measurement.
"""
import argparse
import json
import sys

from sglang.srt.layers.moe.expert_stream_timeline_join import LostRecords, load, summarize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--from-forward", type=int, default=0, help="skip warm-up forwards below this one")
    ap.add_argument("--allow-loss", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    lines = [json.loads(line) for line in open(a.trace) if line.strip()]
    lines = [l for l in lines if l.get("kind") == "clock_calibration" or l.get("forward", 0) >= a.from_forward]
    try:
        s = summarize(load(lines), allow_loss=a.allow_loss)
    except LostRecords as exc:
        print(f"REFUSED: {exc}")
        return 2
    fit, med = s["fit"], s["medians"]
    print(f"coverage {s['coverage']}  calibration error <= {s['calibration_error_ns']} ns  lost {s['lost']} torn {s['torn']}")
    print(f"GPU clock tick (smallest positive gap between one block's stamps): {s['gpu_tick_ns']} ns")
    print(f"pass fit: fixed {fit['pass_fixed_ns']} ns, per piece {fit['per_piece_ns']} ns (n={fit['n']})")
    print(f"medians: {med}")
    if fit["per_piece_ns"] and med["publish_to_seen_ns"] is not None:
        ratio = (fit["pass_fixed_ns"] + med["publish_to_seen_ns"]) / fit["per_piece_ns"]
        print(f"fixed-cost ratio (fewer pieces if >= 0.25, more allowed if < 0.10): {ratio:.3f}")
        tails = [r["tail_ns"] for r in s["requests"] if r["tail_ns"] is not None]
        long_tail = sum(t >= 2 * fit["per_piece_ns"] for t in tails) / max(1, len(tails))
        print(f"share of requests with tail >= 2 x per-piece: {long_tail:.3f}")
    print(f"copy share (unroll/TMA rows need >= 0.20): {med['copy_share']}")
    print(f"CW spun share (batch memcpy row needs >= 0.10): {med['cw_spun_share']}")
    steps = list(s["per_step"].values())
    if steps:
        keys = sorted({k for v in steps for k in v})
        print("per-step means (ns): " + ", ".join(f"{k} {sum(v.get(k, 0) for v in steps) / len(steps):.0f}" for k in keys))
    if a.json:
        json.dump(s, open(a.json, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Commit, push, run a short arm on divix01**

```bash
git add python/sglang/srt/layers/moe/expert_stream_timeline_join.py python/sglang/srt/layers/moe/exl3_stream_trace.py \
  analysis/dsv41-drive/stream-timeline/timeline_report.py test/registered/unit/layers/moe/test_expert_stream_timeline_join.py \
  test/registered/unit/kernels/test_exl3_ram_miss_trace_export.py
git commit -m "feat(expert-stream): join the device timeline with the stage trace; per-request partition and pass fit"
git push origin expert-stream-device-timeline
```

Run the production arm driver that the most recent piece-streaming measurement used, in `wt-timeline`. Before
launching, read `analysis/dsv41-drive/copy-overlap/README.md` and the latest `run_arm.sh` invocation recorded in
`docs/superpowers/plans/2026-09-26-dsv41-compute-transfer-gap-experiments-handoff.md`. Launch it with every env var
of that invocation, plus `SGLANG_DSV41_EXPERT_TRACE_PATH=<scratch>/timeline-trace.jsonl` and
`SGLANG_DSV41_DEBUG_DEVICE_TIMELINE=1`.

Keep the run short: two sessions. Keep `NSYS_GPU_METRICS=0`; no nsys is needed. Follow the lock order: the disk
lock, then `cc-gpu.lock`. Then:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-timeline && PYTHONPATH=$PWD/python taskset -c 0-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/stream-timeline/timeline_report.py <scratch>/timeline-trace.jsonl \
  --from-forward 32 --json <scratch>/timeline-report.json; echo EXIT=$?'
```

Expected: `EXIT=0`, `lost 0`, `torn 0`, coverage matched above 95% of device requests, and calibration error below
50 us. If loss is reported, raise `DEFAULT_CAPACITY` for this run only through a local constant change in the
worktree. That is a throwaway edit, reverted after; record it. Do not commit it.

- [ ] **Step 6: Write `README.md` and `results.md`, fill in the Decision table, commit**

`README.md` covers:
- How to enable the timeline (the two env vars).
- That a timeline run is not a throughput run.
- The definitions block of this task.
- The report command.
- The join rules and their exclusions.

`results.md` covers:
- The report output, pasted verbatim.
- The commands.
- The run's commit.
- The Decision table rows that need C's numbers (unroll, TMA/cp.async, wider CW, batch memcpy, fewer pieces, more
  pieces). Each row gets its numbers in the Task 6 form: `condition: numbers -> yes/no (on Gen3)`. Combine with
  Part A's numbers from `analysis/dsv41-drive/copy-mechanism/results.md` and Part B's from
  `analysis/dsv41-drive/chain-pdl/results.md`.

```bash
git add analysis/dsv41-drive/stream-timeline/README.md analysis/dsv41-drive/stream-timeline/results.md
git commit -m "analysis(stream-timeline): divix01 Gen3 device timeline results and the Decision table"
git push origin expert-stream-device-timeline
```

To rerun Parts A and C on the Gen5 host:
- Part A: run the Task 6 README one-liner.
- Part C: run Step 5 with the same env vars.
- Fill the Decision table's last column again, for that host.
