# DSpark Graphed Verify: Union Curve and Go/No-Go Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure DSV4.1's per-layer expert union for a multi-token DSpark verify and project graphed-verify tok/s
against plain decode with CPU experts. This decides whether the graphed-verify build (DSV41_REFERENCE.md §33.3 items
1-8) goes ahead, and at what gather width W.

**Architecture:** One new offline script, `scripts/dsv41/verify_union.py`. It turns the DSV4.1 one-token router
capture into verify-shaped forwards (windows of `width` tokens, advancing `stride` = accept length) and replays them
through the existing residency simulator (`cpu_expert_sim.replay_nm`) with CPU experts off, as §33.3 v1 requires. It
costs each verify with the simulator's link/NVMe model and compares against the same simulator's plain-decode arm with
CPU hits (§31.2's 75.94 ms/token). `replay_nm` gains one pass-through parameter, the gather shortlist width
(`miss_rows`), so the replay inserts up to W misses per layer instead of 6.

**Tech Stack:** Python 3, numpy, pytest; runs on divix01 CPU only.

**Spec:** `DSV41_REFERENCE.md` §33.3, the closing paragraph "Next step if resumed" (measure the union offline before any
D2 code; build items 1-5 only if the projection beats the baseline), and §10 "Measurement gate" item 1. §33.3's
file:line audit is stale; the 2026-10-05 re-audit at `1636e58758` is summarized under Context and replaces it.

## Context: the re-audit at `1636e58758` (what changed since §33.3)

- **Item 2 (record width) is mostly done.** `LeaseLayout<NumLanes,NumNodes>` is built for 1..32 lanes
  (`lease_layout.h:17-18,33-39,117`). Lane masks are u32 and static-asserted ≤32 (`row_copy_kernels.cuh:263-264`).
  The attach check compares against the build's lanes (`exl3_ram_miss.py:1196-1202`). **New hard cap: W ≤ 32**, below
  the 36 distinct experts a 6-token top-6 union can reach. The fused route plan's `MAX_ROUTES=32` caps the same way.
- **Items 1, 3, 4, 6a, 6b, 7, 8 are unchanged**, with sites moved:
  - DIRECT needs `capacity ≥ 2 × width`: `expert_residency_gpu.py:377-383`.
  - The overflow re-verify is still missing: `model_runner.py:759-764`.
  - The fused plan needs `shape[0]==1`: `expert_route_plan.py:151`.
  - The fused MoE raises on `x.shape[0]!=1`: `quantization/exl3/fused_moe.py:171-174`, and needs
    `graph_gather_rows == top_k` (`:247-250`).
  - The CE barrier tests `is_decode()`: `exl3_ram_miss.py:1428-1432`.
  - The gates: `expert_stream_requirements_exl3.py:134-156`.
- **Item 5 (CPU experts):** the kernel takes `rows`, but the production path is still one token everywhere:
  - `call.rows = 1` (`cpu_experts.h:255-264`);
  - one summed weight per lane on the wire (`lease_kernels.cuh:207-213`, `lease_layout.h:70`);
  - one staged input row (`lease_kernels.cuh:428`).
  - CPU experts also require the fused plan (`exl3_ram_miss.py:1101-1106`).
  - So v1 keeps CPU experts **off** for verify. That is the arm this plan projects.
- **New width-bound pieces:**
  - per-node staging of `kLanes` slots with `home = expert % nodes` (`lease_device.cuh:429`);
  - split calibration at `kLanes`;
  - cpu-insert miss bitmasks in u32 (`ram_tier.h:1815`).

## Global Constraints

- Code is written on the laptop, committed, pushed (`git push origin dsv41-dspark-graph`), and pulled on divix01 into
  a private worktree `wt-dsv41-dspark-graph`. No rsync/scp of trees. Never push master. No amend, rebase or stash.
- On divix01:
  - run with `PYTHONPATH=$PWD/python`, and print `sglang.__file__` before trusting a result;
  - put every CPU job under `taskset -c 0-63` with `OMP_NUM_THREADS` capped; cores 64-71 stay free;
  - read `PIPESTATUS[0]` for any piped pytest;
  - do no bulk reads of `/mnt/nvme1` or `/mnt/nvme2`. The trace lives on `/data`.
- Trace: `/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl`.
  - DSV4.1, 40 layers, 384 routed experts, top-6.
  - 6,153 decode `graph_routes` lines, 26 requests, `dropped` 0.
- Commit trailers:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq`.
- Existing `replay_nm` callers must be unaffected: `miss_rows` defaults to 6, today's value.

## Review Focus

1. A window must never join tokens of two requests, or a decode run across an eager prefill. Expected: each
   window's tokens are consecutive decode forwards of one rid; an eager forward stays in place.
2. Overlapping windows (`stride < width`) re-count tokens on purpose, because a verify re-routes the tokens it
   rejected. Expected: the union statistics and the verify count reflect overlapping windows, not a deduplicated
   token stream.
3. Unions wider than the shortlist. Expected: `replay_nm` still counts every miss in `n`/`m`. Only insertion is capped
   at `miss_rows`, which `graph_forward` already does by zipping against the shortlist.
4. Projection rows with W too large for the layer's VRAM capacity (`capacity < 2W`, item 1). Expected: flagged
   `capacity_ok: False` and excluded from the gate. They are not silently counted as a win.
5. The baseline must be the simulator's own plain-decode arm on the same trace. A hardcoded 13.5 tok/s would compare
   two different cost models. Expected: the baseline lands within ±3% of §31.2's 75.94 ms/token, or the run stops.

---

### Task 1: `replay_nm` takes the gather shortlist width

**Files:**
- Modify: `scripts/dsv41/cpu_expert_sim.py` (`replay_nm`, signature at line 241 and `DirectInsertReplay(...)` at line ~286)
- Test: `test/registered/unit/kernels/test_cpu_expert_sim.py`

**Interfaces:**
- Produces: `replay_nm(..., miss_rows: int = 6) -> dict`, passed to `tier_sim.DirectInsertReplay(initial, capacity,
  num_experts, miss_rows=miss_rows)`.

- [ ] **Step 1: Write the failing test** (append before `if __name__` or at the end of the file if it has none)

```python
def test_miss_rows_widens_the_insert_shortlist():
    # One layer, 16 hot slots holding experts 0-15. Both forwards route 20-27, 8 VRAM misses. With the default
    # shortlist of 6, two of them stay uninserted and miss again (RAM hits, n = 2); with 8, all 8 land.
    loaded = {
        "layer_ids": [0],
        "hot_capacity": {0: 16},
        "forwards": [_graph(1, list(range(20, 28)), list(range(16))), _graph(2, list(range(20, 28)), list(range(16)))],
    }
    narrow = replay_nm(loaded, ram_rows=64, num_experts=64)
    wide = replay_nm(loaded, ram_rows=64, num_experts=64, miss_rows=8)
    assert (narrow["n"] + narrow["m"]).tolist() == [[8], [2]]
    assert (wide["n"] + wide["m"]).tolist() == [[8], [0]]
```

- [ ] **Step 2: Run it and watch it fail**

Run (divix01, in `wt-dsv41-dspark-graph`):
`PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_cpu_expert_sim.py -k miss_rows; echo EXIT=${PIPESTATUS[0]}`
Expected: FAIL with `TypeError: replay_nm() got an unexpected keyword argument 'miss_rows'`.

- [ ] **Step 3: Implement.** Add `miss_rows: int = 6,` after `protect_reads: bool = True,` in `replay_nm`'s
  signature. Add to the docstring: "``miss_rows`` is the DIRECT victim shortlist, the most misses a layer inserts per
  forward (6 at batch size 1; a verify's gather width W)." Change the constructor call to:

```python
    sim = tier_sim.DirectInsertReplay(initial, capacity, num_experts, miss_rows=miss_rows)
```

- [ ] **Step 4: Run the whole file**

Run: same command without `-k`.
Expected: all pass, including the new test (the existing count + 1).

- [ ] **Step 5: Commit**

```bash
git add scripts/dsv41/cpu_expert_sim.py test/registered/unit/kernels/test_cpu_expert_sim.py
git commit -m "feat(cpu-expert-sim): replay_nm takes the DIRECT shortlist width (miss_rows), for verify-width gathers"
```

### Task 2: verify windows and the union curve

**Files:**
- Create: `scripts/dsv41/verify_union.py`
- Test: `test/registered/unit/kernels/test_verify_union.py`

**Interfaces:**
- Consumes: the `tier_sim.load_forwards` dict (`layer_ids`, `hot_capacity`, `forwards`). Each graph forward is
  `{"kind": "graph", "phase", "tokens", "rids", "seq", "forward_pass_id", "routes": {layer: [ids]}, "misses": {layer:
  n}, "hot"}`.
- Produces:
  - `window_forwards(loaded: dict, width: int, stride: int) -> dict`;
  - `union_stats(loaded: dict) -> dict` with keys `verifies, mean, p50, p95, p99, max, per_layer_mean`;
  - `shrink_hot(loaded: dict, slots: int) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""DSpark verify windows over a one-token decode trace, and the union curve (CPU only)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

from verify_union import shrink_hot, union_stats, window_forwards  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _decode(seq, routes, rid="a"):
    return {
        "kind": "graph", "seq": seq, "phase": "decode", "tokens": 1, "rids": [rid], "forward_pass_id": seq,
        "routes": {0: routes}, "misses": {0: len(routes)}, "hot": {0: [0]},
    }


def _eager(forward):
    return {"kind": "eager", "forward": forward, "phase": "extend", "tokens": 256, "rids": ["a"],
            "forward_pass_id": None, "counts": {0: ([1], [1])}, "misses": {0: 1}}


def _loaded(forwards):
    return {"layer_ids": [0], "hot_capacity": {0: 8}, "hot_layer_ids": [0], "forwards": forwards}


def test_windows_union_routes_in_first_appearance_order_and_advance_by_stride():
    loaded = _loaded([_decode(i, r) for i, r in enumerate([[1, 2], [2, 3], [4], [5, 1], [6]])])
    out = window_forwards(loaded, width=3, stride=2)["forwards"]
    assert [f["routes"][0] for f in out] == [[1, 2, 3, 4], [4, 5, 1, 6], [6]]
    assert [f["tokens"] for f in out] == [3, 3, 1]
    assert [f["seq"] for f in out] == [0, 2, 4]
    assert [f["misses"][0] for f in out] == [5, 4, 1]


def test_a_window_never_spans_a_request_or_an_eager_forward():
    forwards = [_decode(0, [1]), _decode(1, [2]), _eager(7), _decode(2, [3]), _decode(3, [4], rid="b")]
    out = window_forwards(_loaded(forwards), width=4, stride=4)["forwards"]
    assert [(f["kind"], f.get("routes", {}).get(0)) for f in out] == [
        ("graph", [1, 2]), ("eager", None), ("graph", [3]), ("graph", [4]),
    ]


def test_width_one_is_the_trace_itself():
    loaded = _loaded([_decode(i, [i, i + 1]) for i in range(4)])
    assert window_forwards(loaded, width=1, stride=1)["forwards"] == loaded["forwards"]


@pytest.mark.parametrize("width,stride", [(0, 1), (3, 0), (3, 4)])
def test_a_stride_outside_one_to_width_is_refused(width, stride):
    with pytest.raises(ValueError, match="stride"):
        window_forwards(_loaded([_decode(0, [1])]), width=width, stride=stride)


def test_union_stats_count_distinct_experts_per_verify_and_layer():
    loaded = _loaded([_decode(i, r) for i, r in enumerate([[1, 2, 2], [3], [4, 5, 6, 7]])])
    stats = union_stats(loaded)
    assert stats["verifies"] == 3 and stats["max"] == 4
    assert stats["mean"] == pytest.approx(7 / 3) and stats["per_layer_mean"] == [pytest.approx(7 / 3)]


def test_shrink_hot_takes_slots_off_every_layer_and_refuses_an_empty_layer():
    loaded = _loaded([_decode(0, [1])])
    assert shrink_hot(loaded, 3)["hot_capacity"] == {0: 5}
    assert shrink_hot(loaded, 0) is loaded
    with pytest.raises(ValueError, match="slots"):
        shrink_hot(loaded, 8)
```

- [ ] **Step 2: Run and watch them fail**

Run: `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_verify_union.py; echo EXIT=${PIPESTATUS[0]}`
Expected: collection error `ModuleNotFoundError: No module named 'verify_union'` (EXIT=2).

- [ ] **Step 3: Implement** `scripts/dsv41/verify_union.py` (Task 3 adds the projection and CLI to this file)

```python
#!/usr/bin/env python3
"""DSpark verify in the decode graph, offline: the per-layer expert union of a multi-token verify, and its tok/s.

A verify of ``width`` tokens routes every token through each layer, and the layer's gather moves the union of their
experts. ``window_forwards`` turns a one-token decode trace (``tier_sim.load_forwards``) into verify forwards: each
window is ``width`` consecutive decode tokens of one request, and the next window starts ``stride`` tokens later
(the accept length). The true next tokens' routes stand in for the draft tokens' (teacher forcing). A window never
spans a request or an eager forward. Residency decays per forward, so a verify decays the insert scores as one token
(the simulator's ``graph_forward`` counts one); that is a small bias toward stickier residency.

``project`` replays the windows through ``cpu_expert_sim.replay_nm`` with CPU experts off (v1, DSV41_REFERENCE.md
§33.3), and ``baseline`` is plain decode with CPU hits on the same trace and cost model (§31.2).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _rid(forward: dict) -> tuple:
    return tuple(forward.get("rids") or ())


def _merge(window: list[dict]) -> dict:
    first = window[0]
    routes = {layer: list(dict.fromkeys(e for f in window for e in f["routes"][layer])) for layer in first["routes"]}
    misses = {layer: sum(f["misses"].get(layer, 0) for f in window) for layer in first["misses"]}
    return {**first, "tokens": len(window), "routes": routes, "misses": misses}


def window_forwards(loaded: dict, width: int, stride: int) -> dict:
    """``loaded`` with every run of one request's decode forwards replaced by its verify windows."""
    if width < 1 or not 1 <= stride <= width:
        raise ValueError(f"want width >= 1 and stride in 1..width, got width {width}, stride {stride}")
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        out.extend(_merge(run[start : start + width]) for start in range(0, len(run), stride))
        run.clear()

    for forward in loaded["forwards"]:
        decode = forward["kind"] == "graph" and forward["phase"] == "decode"
        if decode and run and _rid(forward) == _rid(run[0]):
            run.append(forward)
            continue
        flush()
        if decode:
            run.append(forward)
        else:
            out.append(forward)
    flush()
    return {**loaded, "forwards": out}


def union_stats(loaded: dict) -> dict:
    """Distinct experts per decode forward (a verify, once windowed) and layer."""
    sizes = np.array(
        [[len(set(r)) for r in f["routes"].values()] for f in loaded["forwards"]
         if f["kind"] == "graph" and f["phase"] == "decode"],
        dtype=np.int64,
    )
    flat = sizes.reshape(-1)
    return {
        "verifies": int(sizes.shape[0]),
        "mean": float(flat.mean()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": int(flat.max()),
        "per_layer_mean": [float(x) for x in sizes.mean(axis=0)],
    }


def shrink_hot(loaded: dict, slots: int) -> dict:
    """``loaded`` with ``slots`` fewer hot slots per layer: the VRAM a resident draft takes from the target."""
    if slots == 0:
        return loaded
    capacity = {layer: c - slots for layer, c in loaded["hot_capacity"].items()}
    if min(capacity.values()) < 1:
        raise ValueError(f"taking {slots} slots leaves a layer with {min(capacity.values())}")
    return {**loaded, "hot_capacity": capacity}
```

- [ ] **Step 4: Run them**

Run: the Step 2 command.
Expected: 8 passed (EXIT=0).

- [ ] **Step 5: Commit**

```bash
git add scripts/dsv41/verify_union.py test/registered/unit/kernels/test_verify_union.py
git commit -m "feat(dspark-verify): verify windows over the one-token router trace, and the per-layer union curve"
```

### Task 3: projection, baseline, gate and CLI

**Files:**
- Modify: `scripts/dsv41/verify_union.py`
- Test: `test/registered/unit/kernels/test_verify_union.py`

**Interfaces:**
- Consumes: Task 1's `replay_nm(..., miss_rows=)` and Task 2's `window_forwards`, `union_stats` and `shrink_hot`.
  Also `cpu_expert_sim.CostModel`, `slot_map_costs`, `CALIBRATED_KS`, `MAX_ROUTES`, `STAGING_MERGED`, and
  `split_table` (re-exported by `cpu_expert_sim`).
- Produces:
  - `verify_ms(n, m, *, c_link, nvme_ms, gpu_ms) -> np.ndarray` (per verify);
  - `overflow_rate(n, m, lanes) -> float`;
  - `project(loaded, *, width, stride, lanes, draft_slots, ram_rows, num_experts, c_link, nvme_ms, gpu_ms) -> dict`;
  - `baseline(loaded, *, ram_rows, num_experts, c_link, nvme_ms, gpu_ms, handoff, c_cpu, split_c_cpu) -> dict`;
  - `gate(rows, base, *, width, stride, draft_slots, gain, draft_ms_floor, max_overflow) -> dict`;
  - `main()`.

- [ ] **Step 1: Write the failing tests** (append to `test_verify_union.py`; add `import numpy as np` after
  `import sys`, and extend the import line to
  `from verify_union import baseline, gate, overflow_rate, project, shrink_hot, union_stats, verify_ms, window_forwards`)

```python
def test_verify_ms_adds_link_rows_nvme_waits_and_gpu_once_per_verify():
    n, m = np.array([[1, 2], [0, 0]]), np.array([[1, 0], [0, 1]])
    out = verify_ms(n, m, c_link=1.0, nvme_ms=1.5, gpu_ms=14.0)
    assert out.tolist() == [14.0 + 4 * 1.0 + 1 * 1.5, 14.0 + 1 * 1.0 + 1 * 1.5]


def test_overflow_rate_is_the_share_of_verify_layers_needing_more_lanes_than_the_record_has():
    n, m = np.array([[8, 3], [2, 2]]), np.array([[1, 0], [0, 0]])
    assert overflow_rate(n, m, lanes=8) == pytest.approx(1 / 4)


def test_project_costs_windows_and_flags_a_width_the_vram_cannot_hold():
    loaded = _loaded([_decode(i, [i % 5, 10 + i % 7]) for i in range(12)])
    row = project(loaded, width=4, stride=2, lanes=4, draft_slots=1, ram_rows=64, num_experts=64,
                  c_link=1.0, nvme_ms=1.5, gpu_ms=14.0)
    assert row["verifies"] == 6 and row["capacity_ok"] is False  # 8 - 1 = 7 slots < 2 * 4
    assert row["tok_s_no_draft"] == pytest.approx(2 * 1000.0 / row["verify_ms"])
    assert row["union"]["max"] <= 8


def test_baseline_is_the_slot_map_hits_arm():
    loaded = _loaded([_decode(i, [i % 5, 10 + i % 7]) for i in range(12)])
    base = baseline(loaded, ram_rows=64, num_experts=64, c_link=1.0, nvme_ms=1.5, gpu_ms=14.0, handoff=0.02,
                    c_cpu=0.63, split_c_cpu=0.52)
    assert base["tok_s"] == pytest.approx(1000.0 / base["ms_per_token"]) and base["ms_per_token"] >= 14.0


def _row(lanes, verify, overflow, ok=True, width=6, stride=3, draft_slots=4):
    return {"width": width, "stride": stride, "draft_slots": draft_slots, "lanes": lanes, "verify_ms": verify,
            "overflow": overflow, "capacity_ok": ok}


def test_gate_takes_the_cheapest_admissible_lane_count_and_needs_draft_room():
    base = {"ms_per_token": 76.0}
    rows = [_row(8, 150.0, 0.30), _row(16, 160.0, 0.01), _row(24, 158.0, 0.0, ok=False), _row(32, 170.0, 0.0)]
    out = gate(rows, base, width=6, stride=3, draft_slots=4, gain=1.10, draft_ms_floor=5.0, max_overflow=0.02)
    assert out["lanes"] == 16
    assert out["draft_budget_ms"] == pytest.approx(3 * 76.0 / 1.10 - 160.0)
    assert out["go"] is (out["draft_budget_ms"] >= 5.0)


def test_gate_says_no_when_no_lane_count_is_admissible():
    out = gate([_row(8, 100.0, 0.5)], {"ms_per_token": 76.0}, width=6, stride=3, draft_slots=4, gain=1.10,
               draft_ms_floor=5.0, max_overflow=0.02)
    assert out["go"] is False and "overflow" in out["why"]
```

- [ ] **Step 2: Run and watch them fail**

Run: the Task 2 Step 2 command.
Expected: collection error `ImportError: cannot import name 'baseline' from 'verify_union'` (EXIT=2).

- [ ] **Step 3: Implement.** Add these imports after `sys.path.insert(0, HERE)`:

```python
from cpu_expert_sim import (  # noqa: E402
    CALIBRATED_KS,
    MAX_ROUTES,
    STAGING_MERGED,
    CostModel,
    replay_nm,
    slot_map_costs,
    split_table,
)
```

and append:

```python
def verify_ms(n: np.ndarray, m: np.ndarray, *, c_link: float, nvme_ms: float, gpu_ms: float) -> np.ndarray:
    """ms per verify with CPU experts off: every miss crosses the link, an NVMe miss also waits for its read."""
    return ((n + m) * c_link + m * nvme_ms).sum(axis=1) + gpu_ms


def overflow_rate(n: np.ndarray, m: np.ndarray, lanes: int) -> float:
    """Share of (verify, layer) pairs whose misses exceed the record's lanes: the overflow re-verify's rate."""
    return float(((n + m) > lanes).mean())


def project(
    loaded: dict, *, width: int, stride: int, lanes: int, draft_slots: int, ram_rows: int, num_experts: int,
    c_link: float, nvme_ms: float, gpu_ms: float,
) -> dict:
    """One verify arm: windows of ``width`` advancing ``stride``, a ``lanes``-wide DIRECT shortlist, ``draft_slots``
    hot slots per layer given to the draft. CPU experts off, deferred RAM inserts (the slot-map recipe)."""
    windows = window_forwards(shrink_hot(loaded, draft_slots), width, stride)
    nm = replay_nm(windows, ram_rows, num_experts, True, "insert_all", None, ram_insert="deferred", miss_rows=lanes)
    n, m = nm["n"], nm["m"]
    per = verify_ms(n, m, c_link=c_link, nvme_ms=nvme_ms, gpu_ms=gpu_ms)
    ms = float(per.mean())
    union = union_stats(windows)
    return {
        "width": width,
        "stride": stride,
        "lanes": lanes,
        "draft_slots": draft_slots,
        "verifies": int(len(per)),
        "union": {k: union[k] for k in ("mean", "p95", "p99", "max")},
        "lanes_p99": float(np.percentile((n + m).reshape(-1), 99)),
        "overflow": overflow_rate(n, m, lanes),
        "capacity_ok": min(windows["hot_capacity"].values()) >= 2 * lanes,
        "nvme_reads_per_verify": float(m.sum() / max(len(per), 1)),
        "hot_hit_rate": nm["residency"]["hot_hit_rate"],
        "verify_ms": ms,
        "tok_s_no_draft": stride * 1000.0 / ms,
    }


def baseline(
    loaded: dict, *, ram_rows: int, num_experts: int, c_link: float, nvme_ms: float, gpu_ms: float, handoff: float,
    c_cpu: float, split_c_cpu: float,
) -> dict:
    """Plain decode with CPU hits on the same trace: slot_map_results' K = 0, protect-reads, ``hits`` arm."""
    split = split_table(MAX_ROUTES, split_c_cpu, c_link, handoff).tolist()
    nm = replay_nm(loaded, ram_rows, num_experts, True, STAGING_MERGED, split, ram_insert="deferred")
    model = CostModel([c_cpu] * len(CALIBRATED_KS), c_link=c_link, handoff=handoff, nvme_ms=nvme_ms, gpu_ms=gpu_ms)
    ms = slot_map_costs(nm["n"], nm["m"], nm["kh"], nm["km"], model)["pessimistic"]
    return {"ms_per_token": ms, "tok_s": 1000.0 / ms, "split": split}


def gate(
    rows: list[dict], base: dict, *, width: int, stride: int, draft_slots: int, gain: float, draft_ms_floor: float,
    max_overflow: float,
) -> dict:
    """GO when, at the cheapest lane count within the overflow and VRAM limits, a verify leaves at least
    ``draft_ms_floor`` ms for the draft while still beating the baseline by ``gain``."""
    picked = [r for r in rows if (r["width"], r["stride"], r["draft_slots"]) == (width, stride, draft_slots)]
    ok = [r for r in picked if r["overflow"] <= max_overflow and r["capacity_ok"]]
    if not ok:
        return {"go": False, "why": f"no lane count keeps overflow <= {max_overflow} within VRAM capacity"}
    best = min(ok, key=lambda r: r["verify_ms"])
    budget = stride * base["ms_per_token"] / gain - best["verify_ms"]
    return {
        "go": budget >= draft_ms_floor,
        "lanes": best["lanes"],
        "draft_budget_ms": budget,
        "why": f"W={best['lanes']}: {budget:.1f} ms per verify left for the draft at {gain:.2f}x the baseline "
               f"(floor {draft_ms_floor} ms)",
    }


def _pairs(text: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in p.split(":")) for p in text.split(",")]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


_LOADED: dict = {}


def _work(job: tuple) -> dict:
    (width, stride), lanes, slots, kw = job
    return project(_LOADED["trace"], width=width, stride=stride, lanes=lanes, draft_slots=slots, **kw)


def _init(path: str) -> None:
    import tier_sim

    _LOADED["trace"] = tier_sim.load_forwards(path)


def main() -> None:
    import argparse
    import json
    from concurrent.futures import ProcessPoolExecutor

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--out", required=True, help="JSON results path; a markdown table is printed")
    p.add_argument("--pairs", type=_pairs, default=_pairs("4:2,4:3,6:2,6:3,6:4"), help="width:stride list")
    p.add_argument("--lanes", type=_ints, default=_ints("8,16,24,32"), help="record lanes W (wire cap 32)")
    p.add_argument("--draft-slots", type=_ints, default=_ints("0,4,13"),
                   help="hot slots per layer the draft takes (hybrid 2.33 GB ~ 4, resident 6.82 GB ~ 13)")
    p.add_argument("--ram-rows", type=int, default=8063)
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--c-link", type=float, default=1.0)
    p.add_argument("--nvme-ms", type=float, default=1.5)
    p.add_argument("--gpu-ms", type=float, default=14.0, help="GPU compute per forward (verify and plain decode)")
    p.add_argument("--handoff", type=float, default=0.02)
    p.add_argument("--measured-c-cpu", type=float, default=0.63)
    p.add_argument("--split-c-cpu", type=float, default=0.52)
    p.add_argument("--gain", type=float, default=1.10)
    p.add_argument("--draft-ms-floor", type=float, default=5.0)
    p.add_argument("--max-overflow", type=float, default=0.02)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()

    _init(args.trace)
    loaded = _LOADED["trace"]
    kw = dict(ram_rows=args.ram_rows, num_experts=args.num_experts, c_link=args.c_link, nvme_ms=args.nvme_ms,
              gpu_ms=args.gpu_ms)
    base = baseline(loaded, handoff=args.handoff, c_cpu=args.measured_c_cpu, split_c_cpu=args.split_c_cpu, **kw)
    curve = {f"{w}:{s}": union_stats(window_forwards(loaded, w, s)) for w, s in args.pairs}
    jobs = [(pair, lanes, slots, kw) for pair in args.pairs for lanes in args.lanes for slots in args.draft_slots]
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.trace,)) as pool:
        rows = list(pool.map(_work, jobs))
    for row in rows:
        row["draft_budget_ms_parity"] = row["stride"] * base["ms_per_token"] - row["verify_ms"]
    verdict = gate(rows, base, width=6, stride=3, draft_slots=4, gain=args.gain, draft_ms_floor=args.draft_ms_floor,
                   max_overflow=args.max_overflow)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "baseline": base, "union_curve": curve, "rows": rows, "gate": verdict}, f,
                  indent=1, default=str)
    print(f"baseline: {base['ms_per_token']:.2f} ms/token = {base['tok_s']:.2f} tok/s, split {base['split']}")
    print("| width:stride | union mean | p95 | p99 | max |\n|---|---|---|---|---|")
    for key, u in curve.items():
        print(f"| {key} | {u['mean']:.2f} | {u['p95']:.0f} | {u['p99']:.0f} | {u['max']} |")
    print("\n| w:s | W | draft slots | overflow | cap ok | NVMe/verify | verify ms | tok/s (no draft) | draft ms at parity |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['width']}:{r['stride']} | {r['lanes']} | {r['draft_slots']} | {r['overflow']:.3f} | "
              f"{r['capacity_ok']} | {r['nvme_reads_per_verify']:.2f} | {r['verify_ms']:.1f} | "
              f"{r['tok_s_no_draft']:.2f} | {r['draft_budget_ms_parity']:.1f} |")
    print(f"\ngate: {'GO' if verdict['go'] else 'NO-GO'} -- {verdict['why']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run them**

Run: the Task 2 Step 2 command, then `test_cpu_expert_sim.py` as in Task 1 Step 4.
Expected: `test_verify_union.py` 14 passed; `test_cpu_expert_sim.py` all passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/dsv41/verify_union.py test/registered/unit/kernels/test_verify_union.py
git commit -m "feat(dspark-verify): project graphed-verify tok/s against plain decode with CPU hits, and the go/no-go gate"
```

### Task 4: run the projection on the DSV4.1 trace

**Files:** none in the repo. Output goes to
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/graph-verify/`.

- [ ] **Step 1: Push and pull**

```bash
git push origin dsv41-dspark-graph
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph origin/dsv41-dspark-graph \
  && git -C /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph log -1 --oneline'
```

Expected: the log line shows Task 3's commit.

- [ ] **Step 2: Run the two test files there** (the Task 3 Step 4 commands, in `wt-dsv41-dspark-graph`).
  Expected: all pass, EXIT=0 for both.

- [ ] **Step 3: Run the projection**

```bash
cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph
G=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/graph-verify; mkdir -p $G
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 taskset -c 0-63 /data/models/slang/.venv/bin/python \
  scripts/dsv41/verify_union.py \
  /data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl \
  --out $G/projection.json --jobs 8 > $G/projection.md 2> $G/projection.err; echo EXIT=$?
```

Expected: EXIT=0. The first line of `projection.md` gives a baseline within ±3% of §31.2's 75.94 ms/token
(73.7-78.2). **If not, stop.** The cost model or the recipe arguments differ from §31.2's: find which before reading
any verify row. The table holds 5 pairs × 4 W × 3 draft-slot rows (60), and a gate line.

- [ ] **Step 4: Sanity-check the rows before reading them**
  - The `4:2`/`6:3` union means sit between 6 (all tokens share their experts) and 6 × width.
  - `overflow` falls as W grows.
  - `capacity_ok` is False wherever `hot_capacity - draft_slots < 2W`; `router.json`'s `hot_capacity` gives the
    per-layer capacity.

  A violation is a bug in Task 2 or 3. Fix it with a failing test first, before any doc text.

### Task 5: record the result and the decision

**Files:**
- Modify: `DSV41_REFERENCE.md`. Add `### 33.5 The verify union curve and the graphed-verify projection (2026-10-05)`
  after §33.4, or after §33.3 if §33.4 has not reached this branch.

- [ ] **Step 1: Write §33.5.** It covers:
  - the command and output path;
  - the baseline line;
  - the union-curve table (width:stride, mean, p95, p99, max);
  - the projection rows for draft slots 4, all W, pairs `6:3` and `6:4` (the measured accept lengths 2.88 and 3.4,
    §33.2);
  - the gate line and the draft budget at parity;
  - the modelling limits:
    - teacher-forced routes for draft tokens;
    - per-verify (not per-token) residency decay;
    - `--gpu-ms` 14 for a 6-token verify, which is optimistic (it adds the attention break and Engram breaks of §33.3
      item 6);
    - CPU experts off.
  - **Verdict.**
    - GO: name W, and the build order from the Context section (overflow re-verify and W cap first, then the
      multi-row EXL3 MoE, cross-token dedup in the fused plan, the wire, staging and split at W, the DSpark side, the
      small items, and the gates last).
    - NO-GO: name the cheapest lever that would change it, from the draft-budget column.

- [ ] **Step 2: Commit and push**

```bash
git add DSV41_REFERENCE.md
git commit -m "docs(dsv41): DSpark verify union curve and graphed-verify projection (section 33.5)"
git push origin dsv41-dspark-graph
```

- [ ] **Step 3: Stop for the decision.** The build plans for §33.3 items 1-8 are written only after the human partner
  reads §33.5's verdict. Present GO/NO-GO with the gate line and the draft budget.
