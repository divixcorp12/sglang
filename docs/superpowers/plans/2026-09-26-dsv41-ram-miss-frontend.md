# DSV4.1 decode: size, then trim, the RAM-miss frontend (W1/C1/A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out from existing traces how much decode time W1 → C1 → A1 can cost now that the copy engine carries
the RAM hits. Then, only if the bound earns it, try `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US=0` and a reset-only frontend.

**Architecture:** Three gated stages.
1. Task 1 is an offline analysis of the node-mode trace we already have. It gives an upper bound on the saving, in
   ms/step, for each option.
2. Task 2 is an env-var-only A/B. It runs only if Task 1's bound for it clears Gate A.
3. Tasks 3 and 4 build and measure a reset-only chain behind a new flag, `post -> R -> S -> A2 -> CW -> F`, where R is
   W1's per-request reset without the polling and without C1/A1. They run only if Task 1 clears Gate B.

Task 5 runs whatever the gates say. It measures resident-first's missed-only compute tail right after real copies,
with copy-engine-only overlap and one final gather, to decide whether resident-first stays shelved. It is a
microbenchmark and builds nothing.

Task 6 writes up whatever happened, including a negative result.

Coverage of the critique's five experiments:

| Critique experiment | Task |
|---|---|
| 1. The existing frontend at 100 vs 0 | Task 2 |
| 2. Reset-only chain keeping `S -> A2 -> CW -> F` | Tasks 3 and 4 |
| 3. All-resident, all-CE, all-NVMe and mixed routes; warmup, eager fallback, failures, repeated replay | Task 3, Step 2 |
| 4. Layer latency, transfer completion, actual PCIe traffic | Task 1's copy metrics on every trace; PCIe RX on the traced arms of Tasks 2 and 4 |
| 5. Resident-first only after measuring the post-transfer missed-only tail; CE-only overlap, one gather | Task 5 |

**Tech Stack:** Python 3 + sqlite3 (trace analysis), CUDA C++ JIT kernels (`exl3_ram_miss.cuh`), the C++ RAM-miss
service, pytest, nsys node-mode traces, and `benchmarks/dsv41_baseline/run_arm.sh`.

**Spec:** None as a file. The spec is the critique pasted in the 2026-09-26 session, plus the response to it. Its
binding points are restated here:
- W1 does not start the copy-engine DMA; the host service does, after Post. W1 can therefore cost time only by
  delaying S (the NVMe pieces) or F.
- Setting `HIT_WAIT_US=0` still leaves one inspection pass, C1 and A1. It removes only the extra polling.
- A reset-only frontend must reset on every replay, including the failure and empty-request paths: `go_1`, `go_2`,
  `claimed[]`, `violated`, `stream_count` and `stream_abort`. It must keep S (S checks that the demand read
  succeeded), CW with its SM phase and SmAck (`LEASE_PROTOCOL.md` §7.6), and F.
- It must work unarmed too: eager forwards, and the first `COPY_ENGINE_ARM_DECODES` (16) decodes after capture. There,
  every RAM hit is READY and goes through S and A2 instead of C1 and A1.
- Measure `post -> F` per layer and the full step, not W1's duration alone.
- Resident-first is not built here. `analysis/dsv41-drive/resident-first/bench-run2.txt` shows the missed-only
  launch at 100.9 us/layer against 101.1 for the single launch, but that bench never ran after a real copy. Task 5
  measures that case, with copy-engine-only overlap and one final gather in the original routing order. Resident-first
  is revisited only if Task 5 clears Gate C.

## Global Constraints

- **Git:**
  - Work on `origin/master` directly (detached HEAD). Fetch and check fast-forward before every push. Never
    force-push.
  - Stage files by name; never commit `.omc/`.
  - Commit trailer:
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF`.
- **Running code:**
  - Every run happens on divix01, in a private worktree at the pushed commit, with `PYTHONPATH=$WT/python` (print
    `sglang.__file__`).
  - Never test in `dsv41-direct-prod` or `dsv41-direct-live`.
  - Read `PIPESTATUS` for any piped pytest.
- **divix01 cores and locks:**
  - Scratch goes on `/mnt/nvme1/frontend/`, never `/` or `/tmp`.
  - CPU jobs run under `taskset -c 18-35,54-63`.
  - GPU work runs under `cc-gpu.lock` on cores 32-63. Take `rowimg-disk.lock` first, then `cc-gpu.lock`.
  - Bound trace-analysis memory: `ulimit -v 16000000`.
- **Arms:**
  - Production is STOPPED. Arms run A then B, once each (no ABBA), on port 30021.
  - Traced arms are node mode only (never graph-mode nsys with the copy engine on). They use
    `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp` and `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:45056`.
  - They run with `NSYS_GPU_METRICS=1` for PCIe RX, under the root-capture guards below. If the root session fails
    to start or stop, rerun that arm with `NSYS_GPU_METRICS=0` and ledger a Ruling: the kernel and copy tables still
    answer everything except PCIe RX.
  - Read kernel durations from node-mode traces, never ms/token.
- **Mutants:** only in a private worktree, reverted with `git checkout --`, then the suite rerun green.
- **Code style:**
  - `msgspec.Struct`, not `@dataclass`. No defensive `getattr`.
  - Env vars go through `EnvBool` in `environ.py` plus a `Dsv41Config` field.
  - Comments follow `.claude/rules/comment-style.md`.
- **Worktree guard:** no compound git or heredoc commands over ssh. Write scripts to the scratchpad and scp them to
  `/mnt/nvme1/frontend/`.
- **Asking first:** ask the user before changing `benchmarks/dsv41_baseline/arm_env.py` `base_env()` or touching the
  production checkout.
- **Gates** (all on Task 1's node-mode numbers, per steady decode step):
  - Gate A runs Task 2 if `hit_wait0_bound_ms_per_step >= 1.0`.
  - Gate B runs Tasks 3 and 4 if `frontend_bound_ms_per_step >= 1.0`.
  - Below both, skip to Task 5 and record a no-go.
  - Gate C, on Task 5's 4+2 split: resident-first is worth a design pass only if the serial path (copy, then the
    six-expert launch) minus the overlapped path (copy concurrent with the resident launch, then the missed launch
    and one gather) is at least 25 us/layer. That is 1 ms/token over 40 layers, the per-expert-compute plan's own bar.
- **PCIe root captures** (`NSYS_GPU_METRICS=1`):
  - The root session's scratch lands in `/tmp/nsys-root` on the root volume. Before each traced arm, check that
    `df --output=avail /` shows at least 2 GB.
  - After each traced arm, list the root sessions (`sudo -n /usr/local/sbin/nsys-profile sessions list`) and shut
    down any `dsv41-pcie-*` session left over. On 2026-09-25 an orphan held `cc-gpu.lock`.
  - Why 1.0 ms: untraced A/Bs move ~0.5 ms/token between arms (§27.14's SM small copies: 111.5 → 111.0 was within
    noise). A smaller real effect would not show.

## Review Focus

1. **The first replay after a failed request.**
   - Risk: a stale `stream_abort`, `go_2`, `violated` or `claimed[]` left by the previous replay fails the next,
     healthy request, or worse, commits it.
   - Expected: the next step returns `keep == 1` with correct bytes.
   - Pinned by `test_a_reset_chain_replay_after_a_stream_abort_is_clean` (Task 3).
2. **Unarmed replays and eager forwards under the reset chain.**
   - With no W1 claim, every READY hit is S's.
   - Expected: identical bytes, and every lease retires through A2.
   - Pinned by parametrizing `test_the_captured_chain_waits_in_cw_and_replays_right_armed_and_unarmed` (Task 3).
3. **Fallback RAM hits held until A2.**
   - Their leases now live through S. Under a small pinned tier with eviction pressure, no lease may be lost or doubly
     signalled.
   - Pinned by parametrizing `test_every_lane_holds_its_row_under_host_slot_victim_reuse`, which asserts
     `lease_double_signal == 0` and every lease retired (Task 3).
4. **A request failed before CW.**
   - It must still publish SmAck and release its COPYING leases with no W1 in the chain.
   - Pinned by parametrizing `test_a_request_failed_before_cw_still_acknowledges_and_releases_its_copying_leases`
     (Task 3).
5. **The flag without its prerequisites.**
   - `SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND=1` without the copy engine or piece streaming must refuse at
     startup, not silently run the W1 chain.
   - Pinned by `test_reset_frontend_needs_the_copy_engine_and_piece_streaming` (Task 3).

---

### Task 1: Size the frontend from the existing node-mode trace (step 0)

**Files:**
- Create: `analysis/dsv41-drive/frontend/frontend_bound.py`
- Test: `analysis/dsv41-drive/frontend/test_frontend_bound.py`
- Input (divix01, read-only): `/mnt/nvme1/dsv41-nsys/sm-small-B-node-20260926-034122.sqlite` (the recipe with SM small
  copies, node mode)

**Interfaces:**
- Produces:
  - `split_layers(kernels: list[tuple[int, int, str]]) -> list[Layer]`
  - `saving_bound_ns(cut_ns: int, cw_ns: int, cw_floor_ns: int) -> int`
  - `summarize(steps: list[list[Layer]], budget_ns: int) -> dict`
  - `copy_metrics(steps: list[list[Layer]], copies: list[tuple[int, int, int]]) -> dict`, where `copies` are the
    copy-engine H2D copies `(start, end, bytes)`, sorted by start.
  - A CLI, `frontend_bound.py <trace.sqlite> [--skip 20] [--budget-us 100] [--json out.json]`, printing the JSON keys
    `steps`, `layers_per_step`, `frontend_ms_per_step`, `frontend_bound_ms_per_step`, `w1_ms_per_step`, `w1_p50_us`,
    `w1_p90_us`, `w1_budget_hit_frac`, `hit_wait0_bound_ms_per_step`, `cw_floor_us`, `cw_spin_ms_per_step` and
    `post_to_f_p50_us`.
  - It also prints the transfer-completion keys `copy_bytes_per_step`, `copy_busy_ms_per_step`,
    `copy_done_after_post_p50_us` (when a layer's last copy lands, measured from post) and
    `cw_end_after_copy_p50_us` (how long the chain runs after its last copy lands).
  - Tasks 2 and 4 run this CLI on their traces.

**The bound, and why it is a bound.**
- The DMA is started by the host after Post, so CW's end on a layer is at least CopyDone.
- If CW spun for `spin = cw_ns - cw_floor_ns` waiting for CopyDone, then shortening the chain before CW by `cut` only
  adds spin until `cut > spin`.
- So the saving on a layer is at most `max(0, cut - spin)`.
- For the reset chain, `cut` = the frontend span (`S.start - post.end`) minus nothing; R's own cost is ignored, which
  makes this an upper bound. For `HIT_WAIT_US=0`, `cut = W1_ns - W1_floor`, with `W1_floor` = p10 of W1 durations
  (one pass).
- S's own wait on NVMe pieces can hide part of `cut` as well. The trace cannot separate that out, so the bound
  overstates the saving, never understates it.
- `cw_floor_ns` is the p5 of CW durations over the steady steps.

- [ ] **Step 1: Write the failing tests**

```python
"""frontend_bound.py: per-layer chain grouping and the saving bound. CPU only."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from frontend_bound import copy_metrics, saving_bound_ns, split_layers  # noqa: E402

US = 1000


def _chain(t0, w1_us, s_us, cw_us):
    """One layer at t0 (ns): post 2 us, W1, C1 1 us, A 1 us, S, A 1 us, CW, F 1 us, back to back."""
    out, t = [], t0
    for label, dur in (("post", 2), ("W1", w1_us), ("C1", 1), ("A", 1), ("S", s_us), ("A", 1), ("CW", cw_us), ("F", 1)):
        out.append((t, t + dur * US, label))
        t += dur * US
    return out


def test_split_layers_starts_a_layer_at_each_post_and_takes_the_first_a_as_a1():
    kernels = _chain(0, 50, 30, 20) + _chain(1_000_000, 5, 10, 400)
    layers = split_layers(kernels)
    assert len(layers) == 2
    assert layers[0].s_start - layers[0].post_end == (50 + 1 + 1) * US  # W1 + C1 + A1
    assert layers[1].w1_ns == 5 * US and layers[1].cw_ns == 400 * US


def test_a_cut_hidden_behind_cw_spin_saves_nothing_and_the_excess_is_the_bound():
    # CW spun 70 us past its floor: removing 50 us before it only lengthens the spin.
    assert saving_bound_ns(cut_ns=50 * US, cw_ns=80 * US, cw_floor_ns=10 * US) == 0
    # CW spun 10 us: 40 of the 50 us come off the layer.
    assert saving_bound_ns(cut_ns=50 * US, cw_ns=20 * US, cw_floor_ns=10 * US) == 40 * US
    # CW under its floor is not negative spin.
    assert saving_bound_ns(cut_ns=50 * US, cw_ns=5 * US, cw_floor_ns=10 * US) == 50 * US


def test_copy_metrics_gives_each_layer_the_copies_that_start_inside_its_chain():
    # Layer 0: post ends at 2 us, CW ends at 105 us. Layer 1: post ends at 1002 us, CW ends at 1060 us.
    layers = split_layers(_chain(0, 50, 30, 20) + _chain(1_000_000, 5, 30, 20))
    copies = [
        (2 * US, 101 * US, 1000),                # layer 0: lands 99 us after post, 4 us before CW ends
        (1_002 * US, 1_012 * US, 500),           # layer 1, first copy
        (1_022 * US, 1_050 * US, 500),           # layer 1, last: lands 48 us after post, 10 us before CW ends
        (1_070 * US, 1_080 * US, 999),           # after layer 1's CW: belongs to no layer
    ]
    r = copy_metrics([layers], copies)
    assert r["copy_bytes_per_step"] == 2000
    assert abs(r["copy_busy_ms_per_step"] - 0.137) < 1e-9
    assert r["copy_done_after_post_p50_us"] == 73.5
    assert r["cw_end_after_copy_p50_us"] == 7.0


def test_the_cli_reads_a_node_mode_export_and_reports_per_step_bounds(tmp_path):
    names = {
        1: "exl3_ram_miss_post_kernel",
        2: "exl3_ram_miss_lease_stream_hit_wait_kernel",
        3: "copy_expert_row_segments_gpu_kernel",
        4: "exl3_ram_miss_lease_stage_ack_kernel",
        5: "exl3_ram_miss_lease_stream_kernel",
        6: "exl3_ram_miss_lease_copy_wait_kernel",
        7: "exl3_ram_miss_lease_finalize_kernel",
    }
    ids = {"post": 1, "W1": 2, "C1": 3, "A": 4, "S": 5, "CW": 6, "F": 7}
    db = tmp_path / "t.sqlite"
    c = sqlite3.connect(db)
    c.execute("create table StringIds (id integer, value text)")
    c.executemany("insert into StringIds values (?, ?)", names.items())
    c.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, correlationId int, shortName int, "
              "streamId int, graphId int)")
    c.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, bytes int, streamId int, copyKind int, "
              "graphNodeId int)")
    rows, copies = [], []
    for step in range(2):
        base = step * 10_000_000
        # Layer 1: W1 polls 100 us (budget), CW at its floor. Layer 2: W1 one pass, CW spins 300 us.
        for chain in (_chain(base, 100, 30, 10), _chain(base + 1_000_000, 5, 30, 310)):
            for s, e, label in chain:
                rows.append((s, e, step + 1, ids[label], 7, 1))
            post_end = chain[0][1]
            cw_end = next(e for _, e, label in chain if label == "CW")
            copies.append((post_end, cw_end - 4 * US, 1000, 141, 1, None))  # H2D, outside the graph
    copies.append((0, 1 * US, 64, 141, 2, None))  # a D2H readback: not a copy-engine H2D
    c.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?)", rows)
    c.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?, ?)", copies)
    c.commit()
    c.close()
    out = subprocess.run([sys.executable, str(HERE / "frontend_bound.py"), str(db), "--skip", "0"],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["steps"] == 2 and r["layers_per_step"] == 2
    assert r["w1_budget_hit_frac"] == 0.5
    # cw_floor = p5 of CW = 10 us. Layer 1: cut = 102 us, no spin. Layer 2: cut 7 us, spin 300 us -> 0.
    assert abs(r["frontend_bound_ms_per_step"] - 0.102) < 1e-9
    # W1 floor = p10 of W1 = 5 us. Layer 1: 95 us, no spin. Layer 2: 0.
    assert abs(r["hit_wait0_bound_ms_per_step"] - 0.095) < 1e-9
    assert r["copy_bytes_per_step"] == 2000 and r["cw_end_after_copy_p50_us"] == 4.0
```

- [ ] **Step 2: Run the tests to see them fail**

On divix01, in a private worktree `wt-frontend` at the pushed commit:

```bash
cd /data/models/slang/nvfp4-work/wt-frontend
taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly analysis/dsv41-drive/frontend/test_frontend_bound.py; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: collection error `ModuleNotFoundError: No module named 'frontend_bound'` (4 tests), EXIT=2. Push the test first as a
red commit: `test(dsv41): frontend bound analysis (red)`.

- [ ] **Step 3: Write `frontend_bound.py`**

```python
#!/usr/bin/env python3
"""Upper bounds on what trimming the RAM-miss frontend can save, from a NODE-mode decode trace (sqlite export).

Per layer (a post kernel starts one): the frontend span post.end -> S.start, W1's duration and CW's. The DMA is
started by the host after post, so a layer's saving from a cut before CW is at most cut - CW's spin past its floor.
S's own NVMe wait can hide more, so these are upper bounds. Node mode inflates small kernels: never read ms/token.

    python3 frontend_bound.py <trace.sqlite> [--skip 20] [--budget-us 100] [--json out.json]
"""

import argparse
import bisect
import collections
import json
import sqlite3
import statistics

import msgspec

CHAIN = {
    "exl3_ram_miss_post_kernel": "post",
    "exl3_ram_miss_lease_stream_hit_wait_kernel": "W1",
    "exl3_ram_miss_lease_stream_reset_kernel": "R",
    "copy_expert_row_segments_gpu_kernel": "C1",
    "exl3_ram_miss_lease_stage_ack_kernel": "A",
    "exl3_ram_miss_lease_stream_kernel": "S",
    "exl3_ram_miss_lease_copy_wait_kernel": "CW",
    "exl3_ram_miss_lease_finalize_kernel": "F",
}


class Layer(msgspec.Struct, frozen=True):
    post_start: int
    post_end: int
    s_start: int
    w1_ns: int  # 0 when the chain has no W1 (the reset chain)
    cw_ns: int
    cw_end: int
    f_end: int


def split_layers(kernels: list[tuple[int, int, str]]) -> list[Layer]:
    layers, cur = [], None
    for start, end, label in kernels:
        if label == "post":
            if cur is not None and {"S", "F"} <= cur.keys():
                layers.append(_layer(cur))
            cur = {"post": (start, end)}
        elif cur is not None and label not in cur:
            cur[label] = (start, end)  # first occurrence: the first A is A1
    if cur is not None and {"S", "F"} <= cur.keys():
        layers.append(_layer(cur))
    return layers


def _layer(k: dict) -> Layer:
    w1 = k.get("W1", (0, 0))
    cw = k.get("CW", (0, 0))
    return Layer(post_start=k["post"][0], post_end=k["post"][1], s_start=k["S"][0], w1_ns=w1[1] - w1[0],
                 cw_ns=cw[1] - cw[0], cw_end=cw[1], f_end=k["F"][1])


def copy_metrics(steps: list[list[Layer]], copies: list[tuple[int, int, int]]) -> dict:
    """A layer's copies are the H2D copies that start between its post and its CW's end: CW waits for all of them."""
    starts = [c[0] for c in copies]
    done_after_post, after_copy = [], []
    total_bytes = busy = 0
    for layer in (l for step in steps for l in step):
        mine = copies[bisect.bisect_left(starts, layer.post_start) : bisect.bisect_right(starts, layer.cw_end)]
        if not mine:
            continue
        last = max(e for _, e, _ in mine)
        total_bytes += sum(b for _, _, b in mine)
        busy += sum(e - s for s, e, _ in mine)
        done_after_post.append(last - layer.post_end)
        after_copy.append(layer.cw_end - last)
    n = len(steps)
    return {
        "copy_bytes_per_step": total_bytes / n,
        "copy_busy_ms_per_step": busy / n / 1e6,
        "copy_done_after_post_p50_us": statistics.median(done_after_post) / 1e3 if done_after_post else 0.0,
        "cw_end_after_copy_p50_us": statistics.median(after_copy) / 1e3 if after_copy else 0.0,
    }


def saving_bound_ns(cut_ns: int, cw_ns: int, cw_floor_ns: int) -> int:
    return max(0, cut_ns - max(0, cw_ns - cw_floor_ns))


def _pct(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(steps: list[list[Layer]], budget_ns: int) -> dict:
    layers = [layer for step in steps for layer in step]
    cw_floor = _pct([l.cw_ns for l in layers], 0.05)
    w1s = [l.w1_ns for l in layers if l.w1_ns > 0]
    w1_floor = _pct(w1s, 0.10) if w1s else 0
    n = len(steps)

    def per_step_ms(f) -> float:
        return sum(f(l) for l in layers) / n / 1e6

    return {
        "steps": n,
        "layers_per_step": len(layers) // n,
        "frontend_ms_per_step": per_step_ms(lambda l: l.s_start - l.post_end),
        "frontend_bound_ms_per_step": per_step_ms(
            lambda l: saving_bound_ns(l.s_start - l.post_end, l.cw_ns, cw_floor)),
        "w1_ms_per_step": per_step_ms(lambda l: l.w1_ns),
        "w1_p50_us": statistics.median(w1s) / 1e3 if w1s else 0.0,
        "w1_p90_us": _pct(w1s, 0.90) / 1e3 if w1s else 0.0,
        "w1_budget_hit_frac": sum(w >= 0.9 * budget_ns for w in w1s) / len(w1s) if w1s else 0.0,
        "hit_wait0_bound_ms_per_step": per_step_ms(
            lambda l: saving_bound_ns(max(0, l.w1_ns - w1_floor), l.cw_ns, cw_floor) if l.w1_ns else 0),
        "cw_floor_us": cw_floor / 1e3,
        "cw_spin_ms_per_step": per_step_ms(lambda l: max(0, l.cw_ns - cw_floor)),
        "post_to_f_p50_us": statistics.median(l.f_end - l.post_end for l in layers) / 1e3,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--skip", type=int, default=20, help="leading graph steps to drop (capture warm-up, unarmed)")
    ap.add_argument("--budget-us", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    by_step = collections.defaultdict(list)
    for start, end, corr, name in c.execute(
        "select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where graphId != 0 order by start"
    ):
        label = CHAIN.get(names[name])
        if label is not None:
            by_step[corr].append((start, end, label))
    ordered = sorted(by_step.values(), key=lambda k: k[0][0])[a.skip :]
    steps = [split_layers(k) for k in ordered]
    result = summarize(steps, budget_ns=a.budget_us * 1000)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        # The copy engine's copies: host to device (copyKind 1) and outside the graph. nsys 2026.3's export has no
        # graphId on memcpy rows, only graphNodeId (NULL outside a graph); see ce_trace.py.
        cols = {r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        outside = "graphId = 0" if "graphId" in cols else "coalesce(graphNodeId, 0) = 0"
        copies = c.execute(
            f"select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 1 and {outside} order by start"
        ).fetchall()
        result.update(copy_metrics(steps, copies))
    print(json.dumps(result, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to see them pass**

Same command as Step 2. Expected: `4 passed`, EXIT=0.

- [ ] **Step 5: Commit and push**

```bash
git add analysis/dsv41-drive/frontend/frontend_bound.py
git commit -m "feat(dsv41): frontend bound analysis for node-mode decode traces"
```

- [ ] **Step 6: Run it on the existing trace and check it against `ce_trace.py`**

```bash
cd /data/models/slang/nvfp4-work/wt-frontend
mkdir -p /mnt/nvme1/frontend
ulimit -v 16000000; taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/frontend/frontend_bound.py /mnt/nvme1/dsv41-nsys/sm-small-B-node-20260926-034122.sqlite --json /mnt/nvme1/frontend/step0-sm-small.json
ulimit -v 16000000; taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/copy-engine/ce_trace.py /mnt/nvme1/dsv41-nsys/sm-small-B-node-20260926-034122.sqlite --json /mnt/nvme1/frontend/step0-ce-trace.json
```

Expected:
- `layers_per_step` is the MoE layer count.
- `w1_ms_per_step` matches `ce_trace.py`'s per-step W1 sum within 2%.
- `cw_spin_ms_per_step` is close to §27.14's CW 54.7 ms/step minus the floor.

If `w1_ms_per_step` disagrees, the grouping is wrong: stop and fix `split_layers`.

- [ ] **Step 7: Apply the gates**

Ledger one line: `Task 1: frontend_bound_ms_per_step=<x>, hit_wait0_bound_ms_per_step=<y>, w1_budget_hit_frac=<z> ->
Gate A <pass|fail>, Gate B <pass|fail>`. If both fail, go to Task 5.

---

### Task 2: `HIT_WAIT_US` 100 vs 0 (only if Gate A passes)

**Files:**
- Create: `analysis/dsv41-drive/frontend/drive_hit_wait.sh`

- Modify: `analysis/dsv41-drive/pcie-trace/pcie_decode.py` (take its two reports and the clock offset as arguments)

**Interfaces:**
- Consumes: Task 1's CLI.
- Produces:
  - arm directories `frontend-hw-A` and `frontend-hw-B`;
  - traces `frontend-hw-A-node-*` and `frontend-hw-B-node-*`, each with a `-pcie` report;
  - `/mnt/nvme1/frontend/hw-{A,B}.json` and `/mnt/nvme1/frontend/hw-{A,B}-pcie.txt`;
  - `pcie_decode.py <main.sqlite> <pcie.sqlite>`, which Task 4 reuses.

No production code changes, so there is no red test. The driver is exercised by the arms themselves. If Gate A fails
but Gate B passes, do Step 0 at the start of Task 4 instead.

- [ ] **Step 0: Make `pcie_decode.py` take its inputs**

Today it hardcodes `D`, both report names and `OFF = 530_792_635` (the difference between the two sessions' start
times).
- Replace them with `argparse` positionals `main_db` and `pcie_db`, and an optional `--offset-ns`.
- When `--offset-ns` is not given, compute it as the pcie report's session start minus the main report's, in
  nanoseconds, from each export's `TARGET_INFO_SESSION_START_TIME` table (`utcEpochNs`). If that table or column is
  absent, list the tables with `.tables`, find the session start column, and ledger a Ruling.
- Also replace the hardcoded `streamId=141` and the 40-layer step split. Take the copy-engine stream as the stream
  holding the most H2D bytes outside the graph, and the layer count from Task 1's `layers_per_step`
  (`--layers`, default 40).

Check: run it on the original pair with the old constant,
`pcie_decode.py /mnt/nvme1/.../pcie-node-20260925-170510.sqlite /mnt/nvme1/.../pcie-node-20260925-170510-pcie.sqlite`.
Locate the pair with `ls /mnt/nvme1/dsv41-nsys/pcie-node-20260925-170510*`; if it is not on divix01, use
`/home/dimitri/data/divix/nsys-reports/` on the laptop under `systemd-run --user --scope -p MemoryMax=8G`. The
unmodified script reads the laptop path; to diff on divix01, copy the old version into the scratch directory and point
its `D` at the divix01 path.
- The computed offset must equal 530,792,635 ns within 1 us.
- The output must match the unmodified script's output at the parent commit byte for byte. Run both on the same pair
  and `diff` them.

Commit: `chore(dsv41): pcie_decode.py takes its reports and offset as arguments`.

- [ ] **Step 1: Write the driver**

```bash
#!/usr/bin/env bash
# HIT_WAIT_US A/B (plan 2026-09-26-dsv41-ram-miss-frontend, Task 2): A = production recipe (100 us), B = 0; once each,
# then node-mode traced A and B for the per-layer frontend bound. Production must be stopped.
# Usage: drive_hit_wait.sh SHA [ab|traced|all]
set -u
WT=/data/models/slang/nvfp4-work/wt-frontend-arms
SHA=$1
MODE=${2:-all}
FLAG=SGLANG_DSV41_RAM_MISS_HIT_WAIT_US=0
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
say() { echo "$(date +%T) $*"; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do say "cc-gpu.lock held; waiting"; sleep 60; done; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
if [ "$MODE" = ab ] || [ "$MODE" = all ]; then
    wait_gpu; say "arm A"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh frontend-hw-A 30021; say "A rc=$?"
    wait_gpu; say "arm B"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh frontend-hw-B 30021 $FLAG; say "B rc=$?"
fi
if [ "$MODE" = traced ] || [ "$MODE" = all ]; then
    # nsys adds ~17 GB of anon memory on node 0 (DSV41_REFERENCE.md section 27.12). NSYS_GPU_METRICS=1 adds the root
    # PCIe session, whose scratch is /tmp/nsys-root on the root volume; an orphan held cc-gpu.lock on 2026-09-25.
    for arm in A B; do
        extra=""; [ $arm = B ] && extra=$FLAG
        avail=$(df --output=avail -B1M / | tail -1)
        [ "$avail" -ge 2048 ] || { say "root volume has ${avail} MiB free; need 2048 for the PCIe session"; exit 1; }
        wait_gpu; say "arm $arm traced"
        NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node NSYS_GPU_METRICS=1 EXPECT_SHA=$SHA \
            bash benchmarks/dsv41_baseline/run_arm.sh frontend-hw-$arm-node 30021 $extra \
            SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:45056; say "$arm traced rc=$?"
        for s in $(sudo -n /usr/local/sbin/nsys-profile sessions list 2>/dev/null | grep -o 'dsv41-pcie-[A-Za-z0-9_.-]*'); do
            say "orphan root session $s: shutting it down"
            sudo -n /usr/local/sbin/nsys-profile shutdown --session="$s"
        done
    done
fi
say "DRIVER DONE"
```

- [ ] **Step 2: Commit, push, and make the arms worktree**

```bash
git add analysis/dsv41-drive/frontend/drive_hit_wait.sh
git commit -m "chore(dsv41): HIT_WAIT_US A/B driver"
```

On divix01: `git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-frontend-arms origin/master`.

- [ ] **Step 3: Run it**

```bash
cd /data/models/slang/nvfp4-work/wt-frontend-arms
nohup bash analysis/dsv41-drive/frontend/drive_hit_wait.sh $(git rev-parse HEAD) all > /mnt/nvme1/frontend/hw-drive.log 2>&1 &
```

Expected: `A rc=0`, `B rc=0`, `A traced rc=0`, `B traced rc=0`, then `DRIVER DONE`.

- [ ] **Step 4: Read the results**

- `python benchmarks/dsv41_baseline/paired.py <A dir> <B dir>` gives pooled client ms/token, TTFT for sessions 0 and
  1, and whether the output is identical. It must be identical: the wait budget only moves lanes between stages.
- Run `frontend_bound.py` on both traced sqlite exports, writing `hw-A.json` and `hw-B.json`.
- Export each `-pcie.nsys-rep` to sqlite (`nsys export --type sqlite`, under `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp`). Run
  `pcie_decode.py` on each pair, writing `hw-{A,B}-pcie.txt`: RX GB/s per link state and per step.

Expected for B:
- `w1_p90_us` well under 100 and `w1_budget_hit_frac ≈ 0`.
- `post_to_f_p50_us` lower than A's by roughly the W1 drop, or unchanged if CW spin absorbed it.
- `copy_bytes_per_step` equal to A's within 1%: the route and hit set are the same.
- `copy_done_after_post_p50_us` is unchanged, since the host drives the DMA; if it moved, say why.
- PCIe RX per step equal within noise, and "no copy, GPU in S" time lower if S started earlier.

- [ ] **Step 5: Decide**

- If B is faster by at least 1.0 ms/token untraced AND `post_to_f_p50_us` fell, ASK the user whether to put
  `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US=0` in `base_env()`. Do not edit it before they answer.
- Otherwise, ledger the no-go.

---

### Task 3: Reset-only frontend behind `SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND` (only if Gate B passes)

**Files:**
- Modify: `python/sglang/srt/environ.py` (next to `SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES`, ~:1912)
- Modify: `python/sglang/srt/dsv41_config.py` (field and loader, next to `enable_ram_miss_sm_small_copies`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py`:
  - `check_reset_frontend` next to `check_sm_small_copies` (~:294);
  - the backend constructor (~:482-514);
  - the two-phase branch of `post` (~:560-614);
  - the config wiring next to `hit_wait_ns` (~:862, ~:983).
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`:
  - split the opening of `lease_hit_wait_body` (:720-762) into `lease_stage1_open`;
  - a new `exl3_ram_miss_lease_stream_reset_kernel` after `exl3_ram_miss_lease_stream_hit_wait_kernel` (~:877);
  - its launcher next to `exl3_ram_miss_lease_stream_hit_wait` (~:2055).
- Modify: `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (the kernel name list ~:1131, and a `stream_reset` method
  next to `hit_wait` ~:1406)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` §7.6 (the reset chain, and fallback hits acknowledged at A2)
- Test: `test/manual/dsv41/test_exl3_piece_stream_cuda.py` (`StreamService`: a `reset_frontend` parameter and a
  `reset()` method)
- Test: `test/manual/dsv41/test_exl3_copy_engine_cuda.py` (parametrize, plus one new test)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py` (the CPU refusal test)

**Interfaces:**
- Produces:
  - `envs.SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND` (EnvBool, default False)
  - `Dsv41Config.enable_ram_miss_reset_frontend: bool`
  - `check_reset_frontend(cfg: Dsv41Config) -> None`, raising `RuntimeError`
  - `DeviceSide.stream_reset(count) -> None`
  - the backend constructor keyword `reset_frontend: bool = False`, raising `ValueError` unless `copy_engine` and
    `piece_stream` are both on
  - kernel `exl3_ram_miss_lease_stream_reset_kernel`, which `frontend_bound.py`'s CHAIN already names `R`

**The design:**
- Chain: `post -> R -> S -> A2 -> CW -> F`, then `go_total = go_1 + go_2 + go_ce` as today (`go_1` is 0).
- R writes everything W1 writes except `host_rows_1` entries beyond the reset, `dst_slots_1`, `origin_1`,
  `lane_ctx_1`, and `kW1Passes`. Nothing reads those when `go_1 == 0`: `stage_ack` guards on `entry < go_1`, and C1
  copies `go_1` rows.
- R keeps W1's request checks (`kSticky`, fatal, shutdown, `planned_count` bound → `kReqFailed` and `kFailReason`).
  It claims nothing, so `claimed[]` is all 0.
- S already covers this case:
  - it admits READY and LOADING lanes and copies them (then A2 acknowledges them);
  - it hands COPYING lanes to CW (`stream_admit`, `sh.mine[lane] = 0`);
  - CW reads COPYING lanes from the row results, not from `claimed` (:1602-1605).
  - Existing test `test_s_hands_a_copying_lane_w1_did_not_claim_to_the_copy_wait` is exactly this state with W1
    present.
- Known accounting change, to write down in LEASE_PROTOCOL: on S's failure paths, `unclaimed` (:1276-1278) now counts
  COPYING lanes too, so `ram_miss` and `kUnservedMisses` on a failed request include them.

- [ ] **Step 1: Write the failing CPU test** (in `test_exl3_ram_miss_copy_engine.py`)

```python
def test_reset_frontend_needs_the_copy_engine_and_piece_streaming():
    """R claims nothing, so the RAM hits must reach CW (copy engine) or S (piece streaming); without both the chain
    would drop them."""
    from sglang.srt.layers.moe.exl3_ram_miss import check_reset_frontend

    base = _config(enable_ram_miss_copy_engine=True, enable_ram_miss_piece_stream=True,
                   enable_ram_miss_reset_frontend=True)
    check_reset_frontend(base)
    for missing in ("enable_ram_miss_copy_engine", "enable_ram_miss_piece_stream"):
        with pytest.raises(RuntimeError, match="SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND"):
            check_reset_frontend(msgspec.structs.replace(base, **{missing: False}))
```

`_config` is the file's existing `Dsv41Config` builder. If the file has none, add one that builds
`Dsv41Config.from_env()` under `monkeypatch`, in that file's existing style. If `Dsv41Config` is not a
`msgspec.Struct`, use its own copy idiom and ledger a Ruling.

- [ ] **Step 2: Write the failing GPU tests** (in `test_exl3_copy_engine_cuda.py`)

Make the frontend a parameter of the chain helpers, not a new copy of every test:

```python
FRONTENDS = ("w1", "reset")


def _frontend(s):
    if s.reset_frontend:
        s.reset()
    else:
        s.hit_wait()
        s.copy1()
        s.ack1()


def _snapshot_step(s):
    """The chain in production order, then a copy of every destination on the decode stream, taken before any sync."""
    s.post()
    _frontend(s)
    s.stream()
    s.ack2()
    s.copy_wait()
    s.finalize()
    snapshot = {n: s.dest[n].clone() for n in s.names}
    s.total()
    torch.cuda.synchronize()
    return snapshot


@pytest.fixture(params=FRONTENDS)
def ce(tmp_path, request):
    s = StreamService(tmp_path, copy_engine=True, reset_frontend=request.param == "reset")
    try:
        yield s
    finally:
        s.close()
```

- In `StreamService` (`test_exl3_piece_stream_cuda.py`), add `reset_frontend: bool = False` to the constructor and
  store it. Add:

```python
    def reset(self):
        self.dev.stream_reset(self.count)
```

  Also make its own `step()` call `reset()` in place of `hit_wait(); copy1(); ack1()` when `reset_frontend` is set.
  Also make `_chain` (the captured-graph builder the capture test uses) do the same.
- Parametrize over `FRONTENDS`, building the service with `reset_frontend=`, the tests that build their own service:
  - `test_the_captured_chain_waits_in_cw_and_replays_right_armed_and_unarmed` (Review Focus 2);
  - `test_a_request_failed_before_cw_still_acknowledges_and_releases_its_copying_leases` (Review Focus 4).
- `test_every_lane_holds_its_row_under_host_slot_victim_reuse` covers Review Focus 3 through the parametrized `ce`
  fixture.
- In `test_hits_go_to_the_copy_engine_while_s_streams_the_misses`, the assertion `go_1 == 0` must hold for both.
- Add the Review Focus 1 test:

```python
@pytest.mark.parametrize("frontend", FRONTENDS)
def test_a_reset_chain_replay_after_a_stream_abort_is_clean(tmp_path, frontend):
    """A step whose S aborts leaves stream_abort, go_2 and the counter set; the frontend of the next step must clear
    them, or a healthy request after a failed one fails (or commits another replay's count)."""
    s = StreamService(tmp_path, copy_engine=True, reset_frontend=frontend == "reset")
    try:
        s.plan([0, 1, 2])
        s.step()
        assert s.until(lambda: _all_retired(s))
        experts = [0, 9, 1, 10, 2, 11]  # hits for the copy engine, misses for S
        s.plan(experts)
        s.set_stream_fault(abort_block=1)
        _snapshot_step(s)
        assert s.keep.item() == 0.0 and int(s.dev.stream_abort.item()) != 0
        s.set_stream_fault()  # all zero: production
        assert s.until(lambda: _all_retired(s)), s.counters()
        for _ in range(3):
            s.plan(experts)
            snapshot = _snapshot_step(s)
            assert s.keep.item() == 1.0, (s.counters(), s.stats())
            _check(s, experts, snapshot)
            assert s.until(lambda: _all_retired(s)), s.counters()
    finally:
        s.close()
```

`set_stream_fault` stands for the file's existing way of setting `STREAM_FAULT_WORDS`:
`s.dev.stream_fault[STREAM_FAULT_WORDS["abort_block"]] = 1` to set it, `s.dev.stream_fault.zero_()` to clear it
(`test_exl3_piece_stream_cuda.py` ~:825-841, ~:1128). Write those lines directly.

- Add the three routes and the eager path that nothing above covers. Together with the tests above, this covers
  experiment 3:

| Case | Test |
|---|---|
| mixed | `test_hits_go_to_the_copy_engine_while_s_streams_the_misses` |
| all-CE | `test_a_delayed_completion_is_waited_for_and_the_lease_holds_until_it` |
| all-resident | new, below |
| all-NVMe | new, below |
| warmup / unarmed replay | the capture test |
| eager fallback | new, below |
| failures | failed-before-CW, S abort then replay |
| repeated replay | 40-step victim reuse, capture test |

```python
@pytest.mark.parametrize("frontend", FRONTENDS)
def test_an_all_resident_step_between_copy_engine_steps_moves_nothing_and_breaks_nothing(tmp_path, frontend):
    """Every expert already in VRAM plans zero RAM-miss lanes: the frontend must still reset (the critique's empty
    path), F must keep the layer, and the copy-engine steps on either side must be unaffected."""
    s = StreamService(tmp_path, copy_engine=True, reset_frontend=frontend == "reset")
    try:
        s.plan([0, 1, 2])
        s.step()
        assert s.until(lambda: _all_retired(s))
        for experts in ([0, 1, 2], [], [0, 1, 2], []):
            s.plan(experts)
            snapshot = _snapshot_step(s)
            assert s.keep.item() == 1.0, (experts, s.counters(), s.stats())
            assert int(s.dev.go_total.item()) == len(experts)
            if experts:
                _check(s, experts, snapshot)
            assert s.until(lambda: _all_retired(s)), s.counters()
    finally:
        s.close()


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_an_all_nvme_step_is_streamed_whole_by_s(tmp_path, frontend):
    """A fresh service holds nothing in RAM: every lane is read from NVMe, S streams all six, CW has nothing to wait
    for, and the bytes are right."""
    s = StreamService(tmp_path, copy_engine=True, reset_frontend=frontend == "reset")
    try:
        experts = list(range(TOP_K))
        s.plan(experts)
        snapshot = _snapshot_step(s)
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        _check(s, experts, snapshot)
        assert int(s.dev.go_2.item()) == TOP_K and int(s.dev.go_ce.item()) == 0 and int(s.dev.go_1.item()) == 0
        assert s.until(lambda: _all_retired(s)), s.counters()
    finally:
        s.close()


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_an_eager_forward_copies_every_hit_through_the_sm_path(tmp_path, frontend):
    """An eager forward posts without the copy engine (the backend passes copy_engine=capturing): every hit is READY,
    not COPYING. With the reset frontend S copies them all and A2 releases their leases; nothing is left for CW."""
    s = StreamService(tmp_path, copy_engine=True, reset_frontend=frontend == "reset")
    try:
        s.plan([0, 1, 2])
        s.step()
        assert s.until(lambda: _all_retired(s))
        waits = s.stats()["copy_waits"]
        experts = [2, 0, 1]
        s.plan(experts)
        s.post(copy_engine=False)
        _frontend(s)
        s.stream()
        s.ack2()
        s.copy_wait()
        s.finalize()
        snapshot = {n: s.dest[n].clone() for n in s.names}
        s.total()
        torch.cuda.synchronize()
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        _check(s, experts, snapshot)
        assert int(s.dev.go_ce.item()) == 0 and s.stats()["copy_waits"] == waits
        assert int(s.dev.go_1.item()) + int(s.dev.go_2.item()) == len(experts)
        if frontend == "reset":
            assert int(s.dev.go_1.item()) == 0
        assert s.until(lambda: _all_retired(s)), s.counters()
    finally:
        s.close()
```

Test harness changes these need, in `StreamService`:
- `post(self, copy_engine=None)`: `None` means `self.copy_engine`; `False` posts without `dst_slots`/`copy_engine`,
  as the non-copy-engine branch does.
- `plan([])` must give `count == 0`. If it cannot, give `plan` an explicit empty-plan path and ledger a Ruling.

If F's `keep` for an empty request is not 1.0 under the `w1` parametrization, the `w1` value is the spec: assert that
value for both, and ledger a Ruling.

- [ ] **Step 3: Run the tests to see them fail**

```bash
# CPU, on divix01
taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py; echo "EXIT=${PIPESTATUS[0]}"
# GPU
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_exl3_copy_engine_cuda.py; echo "EXIT=${PIPESTATUS[0]}"
```

Expected:
- CPU: an ImportError for `check_reset_frontend`.
- GPU: every `reset` parametrization errors, because `stream_reset` is missing. Write the `StreamService` keyword and
  `post(copy_engine=)` in the red commit, since they are test harness.
- Every `w1` parametrization passes, including the new abort-replay, all-resident, all-NVMe and eager tests. That
  proves the W1 chain already meets them. A `w1` failure is a finding about the current code: stop and debug it
  before going on.

Push this as the red commit: `test(dsv41): reset-only RAM-miss frontend (red)`.

- [ ] **Step 4: Split W1's opening and add R** (`exl3_ram_miss.cuh`)

Move lines 738-762 of `lease_hit_wait_body` (the stores through the `!ok` return) into:

```cpp
// Stage 1's opening, shared by W1 and R: clears every per-request word a later kernel reads and marks a request that
// cannot be served failed. Returns false when there is nothing to claim.
__device__ __forceinline__ bool lease_stage1_open(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ count,
    int64_t lanes,
    uint8_t* __restrict__ lease,
    int64_t* __restrict__ host_rows_1,
    int32_t* __restrict__ go_1,
    int32_t* __restrict__ claimed,
    int32_t* __restrict__ violated,
    int64_t* planned_count_out,
    uint32_t* seq_out,
    uint64_t* generation_out) {
  go_1[0] = 0;      // fail closed: the single commit point is the last store of W1
  violated[0] = 0;  // stage 1 opens the chain, so it is where the shared violation flag is cleared
  for (int64_t i = 0; i < lanes; ++i) {
    claimed[i] = 0;
    host_rows_1[i] = 0;
  }
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
  bool ok = state[kSticky] == 0 && ld_acquire_sys(page + kFatal) == 0 &&
            ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
  if (ok && (planned_count > kLeaseLanes || planned_count > lanes)) {
    ok = false;
    state[kFailReason] = static_cast<int32_t>(kLeaseReasonCount);
  }
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  *planned_count_out = planned_count;
  *seq_out = seq;
  *generation_out =
      seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
  if (!ok) {
    state[kReqFailed] = 1;
    return false;
  }
  return seq != 0 && planned_count != 0;
}
```

`lease_hit_wait_body` then starts:

```cpp
  const uint64_t start = global_ns();
  int64_t planned_count;
  uint32_t seq;
  uint64_t generation;
  if (!lease_stage1_open(page, state, count, lanes, lease, host_rows_1, go_1, claimed, violated, &planned_count, &seq,
                         &generation)) {
    return;
  }
```

The poll loop and commit below are unchanged. Then add, after `exl3_ram_miss_lease_stream_hit_wait_kernel`:

```cpp
// Piece streaming's frontend without stage 1 (SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND): W1's resets and request
// checks, claiming nothing, so S copies every READY lane and hands every COPYING lane to CW. C1 and A1 are not launched.
__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_lease_stream_reset_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ count,
    int64_t lanes,
    int64_t* __restrict__ host_rows_1,
    uint8_t* __restrict__ lease,
    int32_t* __restrict__ go_1,
    int32_t* __restrict__ claimed,
    int32_t* __restrict__ violated,
    int32_t* __restrict__ go_2,
    uint32_t* __restrict__ stream_count,
    int32_t* __restrict__ stream_abort) {
  if (threadIdx.x != 0) return;
  go_2[0] = 0;
  stream_count[0] = 0u;
  stream_abort[0] = 0;
  int64_t planned_count;
  uint32_t seq;
  uint64_t generation;
  exl3_ram_miss_device::lease_stage1_open(
      page, state, count, lanes, lease, host_rows_1, go_1, claimed, violated, &planned_count, &seq, &generation);
}
```

- Add the host launcher `exl3_ram_miss_lease_stream_reset`, mirroring `exl3_ram_miss_lease_stream_hit_wait` (~:2055):
  the same `lanes` derivation, the same one-block `kBlock` launch on the current stream, and only these arguments.
- Register the name in the ops module's kernel name list (~:1131).

- [ ] **Step 5: The ops method and the chain**

In `python/sglang/kernels/ops/moe/exl3_ram_miss.py`, next to `hit_wait`:

```python
    def stream_reset(self, count) -> None:
        """Piece streaming's frontend without stage 1: W1's resets and request checks, claiming no lane."""
        self._check_buffers(count=(count, torch.int32))
        if self.lease_block is None or not self.piece_stream:
            raise RuntimeError("the reset frontend needs a lease block and piece streaming")
        self._kernels().exl3_ram_miss_lease_stream_reset(
            self.page, self.state, count, self.host_rows_1, self._lease_address, self.go_1, self.claimed,
            self.violated, self.go_2, self.stream_count, self.stream_abort,
        )
```

In `python/sglang/srt/layers/moe/exl3_ram_miss.py`:
- Constructor: add `reset_frontend: bool = False`. After the copy-engine check:

```python
        if reset_frontend and not (copy_engine and self.piece_stream):
            raise ValueError("the reset frontend runs in the copy-engine piece-streaming chain only")
        self.reset_frontend = reset_frontend
```

- `post`: replace the three frontend calls (`hit_wait`, the C1 `copy_expert_row_segments_gpu`, `stage_ack(1)`) with:

```python
        if self.reset_frontend:
            # post -> R -> S -> A2 -> CW -> F: S copies every READY lane and CW awaits every COPYING one.
            self.device_side.stream_reset(plan.count)
        else:
            self.device_side.hit_wait(self.row, self.planned, plan.count, plan.slots, self.hit_wait_ns)
            copy_expert_row_segments_gpu(
                self.segments[tag], self.device_side.host_rows_1, self.device_side.dst_slots_1, self.device_side.go_1
            )
            self.device_side.stage_ack(1)
```

  Everything after it is unchanged. The `reset_frontend` constructor check guarantees the `piece_stream` branch
  follows.
- Config: add `SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND = EnvBool(False)` in `environ.py`, with a two-line comment:
  "Piece streaming's frontend is W1's reset only: no stage-1 poll, no C1/A1. Needs the copy engine and piece
  streaming." Add `enable_ram_miss_reset_frontend: bool` to `Dsv41Config` and its loader.
- Add `check_reset_frontend(cfg)` next to `check_sm_small_copies`, with the message
  `"exl3 RAM miss: SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND needs SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE and SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM"`.
  Call it everywhere `check_sm_small_copies` is called (use `find_referencing_symbols`).
- Pass `reset_frontend=cfg.enable_ram_miss_reset_frontend` wherever `hit_wait_ns=` is passed (~:983).

- [ ] **Step 6: Run the tests to see them pass**

Run the Step 3 commands again. Expected: all pass (CPU EXIT=0; GPU EXIT=0, both parametrizations). The known
`other_streams[kernel]` flake is not in this file.

- [ ] **Step 7: The mutants**

Use a private worktree `wt-frontend-mut`. Apply each mutant, run the GPU file, revert, and ledger the result.

| Mutant | Must be caught by |
|---|---|
| R omits `stream_abort[0] = 0` | `test_a_reset_chain_replay_after_a_stream_abort_is_clean[reset]` |
| R omits `go_2[0] = 0` | the same, or the victim-reuse test |
| `lease_stage1_open` omits the `claimed[i] = 0` loop, and a W1 step precedes R | a new `w1 -> reset` step in the abort-replay test if nothing catches it; ledger a Ruling if one is added |
| `post` skips `stage_ack(2)` under `reset_frontend` | victim reuse (`_all_retired` times out) |
| `lease_stage1_open` returns before the count bound check | the existing count-overflow test if one exists; else ledger as uncaught and add one |
| R returns before its resets when `count[0] == 0` (the empty path) | `test_an_all_resident_step_between_copy_engine_steps_moves_nothing_and_breaks_nothing[reset]`, through a stale `go_2` in `go_total` |

After reverting, rerun the file: it must be green.

- [ ] **Step 8: Full suites, docs, commit**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly > /mnt/nvme1/frontend/suite.log 2>&1; echo "EXIT=$?"; tail -3 /mnt/nvme1/frontend/suite.log
```

Expected: 1733 + 1 passed (the new CPU test), 1 skipped. Compare against the same command at the parent commit if the
count differs.

Update `LEASE_PROTOCOL.md` §7.6 with the reset chain, fallback hits acknowledged at A2, and the `unclaimed` accounting
change.

```bash
git add python/sglang/srt/environ.py python/sglang/srt/dsv41_config.py python/sglang/srt/layers/moe/exl3_ram_miss.py \
  python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh python/sglang/kernels/ops/moe/exl3_ram_miss.py \
  analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "feat(dsv41): reset-only RAM-miss frontend behind SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND"
```

---

### Task 4: Reset-frontend arms (only if Task 3 is done)

**Files:**
- Create: `analysis/dsv41-drive/frontend/drive_reset.sh`. It is `drive_hit_wait.sh` with:
  - `FLAG=SGLANG_DSV41_ENABLE_RAM_MISS_RESET_FRONTEND=1`;
  - arm names `frontend-rs-A`, `frontend-rs-B`, `frontend-rs-A-node` and `frontend-rs-B-node`;
  - the header line naming Task 4.

  Write the full file; do not source the other one.

**Interfaces:**
- Consumes: Task 1's CLI, Task 3's flag.
- Produces: `/mnt/nvme1/frontend/rs-{A,B}.json` and the arm directories.

- [ ] **Step 1: Write, commit and push the driver**

```bash
git add analysis/dsv41-drive/frontend/drive_reset.sh
git commit -m "chore(dsv41): reset-frontend A/B driver"
```

- [ ] **Step 2: Run it**

Move `wt-frontend-arms` to the new commit (`git -C ... checkout --detach origin/master`), then run
`nohup bash analysis/dsv41-drive/frontend/drive_reset.sh <sha> all > /mnt/nvme1/frontend/rs-drive.log 2>&1 &`.

Expected: all four `rc=0`, then `DRIVER DONE`.

- [ ] **Step 3: Read the results**

- `paired.py` A vs B: the output must be identical, and gives ms/token and TTFT.
- `pcie_decode.py` on each traced pair (Task 2 Step 0; do that step now if Task 2 was skipped), writing
  `rs-{A,B}-pcie.txt`. B's PCIe RX per step must equal A's within noise: the chain moves the same bytes.
- `frontend_bound.py` on both traces. Also compare `copy_bytes_per_step`, `copy_done_after_post_p50_us` and
  `cw_end_after_copy_p50_us`. Bytes should be equal, since fallback hits now go through S only when unarmed. The time
  from the last copy landing to CW's end should not rise. Expected for B:
  - `frontend_ms_per_step` near the R + gap cost only;
  - `post_to_f_p50_us` lower than A's by up to Task 1's `frontend_bound_ms_per_step` / layers.
  - If `cw_spin_ms_per_step` rose by about what the frontend lost, the saving was absorbed; say so.

- [ ] **Step 4: Decide**

If B is faster by at least 1.0 ms/token untraced, with identical output, ASK the user whether to add the flag to
`base_env()`. Otherwise, ledger the no-go; the flag stays default off.

---

### Task 5: Resident-first's post-transfer tail, with copy-engine-only overlap (always runs; measurement only)

**Why:** `bench-run2.txt` timed the missed-only launch on weights that no copy had just written. The critique names
two gaps: overlap with real transfers, and the cache state of freshly copied rows. The two missed rows are ~26.6 MB,
well under the 96 MiB L2, so an H2D write may leave them in L2.

This task adds, to the existing microbenchmark, arms with real H2D copies of each layer's missed rows from pinned
memory, captured in the same graph. It also adds the one arrangement the critique says to try first: copy on a side
stream concurrent with the resident launch, then the missed launch, then one gather in the original routing order
(`two()`'s single `exl3_moe_gather`). No production code changes.

**Files:**
- Modify: `analysis/dsv41-drive/resident-first/split_launch_bench.py`
- Modify: `analysis/dsv41-drive/resident-first/run.sh` (the output directory only if needed; it already runs the
  parity test and the bench)
- Test: `test/manual/dsv41/test_exl3_moe_split_parity_cuda.py` (one new test)

**Interfaces:**
- Produces, in `split_launch_bench.py`:
  - `add_pinned_sources(layers: list[Layer], split: tuple[int, int]) -> None`, which sets
    `Layer.pinned: dict[str, torch.Tensor]` (the missed rows' bytes, pinned) and
    `Layer.missed_phys: list[int]`;
  - `copy_missed(L: Layer, stream: torch.cuda.Stream | None = None) -> None`, which enqueues the H2D copies of
    `L.missed_phys` rows, one `copy_(non_blocking=True)` per tensor per row;
  - the new arms `copy_only/<tag>`, `copy_then_one/<tag>`, `copy_then_miss/<tag>` and `overlap/<tag>`.
- `Layer` gains `pinned: dict = {}` and `missed_phys: list = []` (msgspec mutable defaults are safe).

**The arms**, per layer, inside one 40-layer graph as the existing arms are:

| Arm | What it measures |
|---|---|
| `copy_only/<tag>` | the copies alone |
| `copy_then_one/<tag>` | today's order: copies, then the six-expert launch (`one`) |
| `copy_then_miss/<tag>` | copies, then `two(..., launches=("miss",))`: the missed-only tail on freshly written rows |
| `overlap/<tag>` | side stream: `copy_missed`; main: route tables and the resident launch; the main stream waits for the side; then the missed launch and the one gather |

`overlap` is `two()` with an event wait between its launches. Write it as `two(..., between=...)`, where `two` gains
an optional `between: Callable[[], None] | None = None` called after the resident launch. Fork the side stream before
the route-tables kernel: `side.wait_stream(main)`, then `with torch.cuda.stream(side): copy_missed(L)`, then
`main.wait_stream(side)` in `between`. Stream forks and joins are capturable.

The derived numbers:
- **Serial cost:** `copy_then_one`.
- **Overlapped cost:** `overlap`.
- **Gate C value:** `(copy_then_one - overlap) / LAYERS` us/layer.
- **Cache effect:** `(copy_then_miss - copy_only) - miss_only`, per layer. A negative value means freshly copied rows
  make the tail faster.

- [ ] **Step 1: Write the failing test** (in `test_exl3_moe_split_parity_cuda.py`)

```python
def test_the_bench_copy_lands_in_the_rows_the_missed_launch_reads(slot_rows):
    """The resident-first bench's copy arms are only meaningful if copy_missed writes the rows the missed launch
    reads: poisoned sources must change the layer output, and the true bytes must restore it bitwise."""
    bench = _bench_module()
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator().manual_seed(7)
    layers = bench.build_layers(bench._parity_module(), slot_rows, device, gen)[:1]
    L = layers[0]
    par = bench._parity_module()
    bench.one(par, L)
    want = L.fused.out.clone()
    bench.add_pinned_sources(layers, (4, 2))
    true_bytes = {n: t.clone() for n, t in L.pinned.items()}
    for t in L.pinned.values():
        t.zero_()
    bench.copy_missed(L)
    bench.one(par, L)
    torch.cuda.synchronize()
    assert not torch.equal(_bits(L.fused.out), _bits(want)), "poisoned rows did not reach the launch"
    for n, t in L.pinned.items():
        t.copy_(true_bytes[n])
    bench.copy_missed(L)
    bench.one(par, L)
    torch.cuda.synchronize()
    assert torch.equal(_bits(L.fused.out), _bits(want))
```

`_bench_module()` loads `analysis/dsv41-drive/resident-first/split_launch_bench.py` by path with
`importlib.util.spec_from_file_location`, the way that bench loads this test module (`_parity_module`, bench :47).
Add it next to `_bits`.

Run:

```bash
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_exl3_moe_split_parity_cuda.py -k bench_copy; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: FAIL with `AttributeError: module ... has no attribute 'add_pinned_sources'`. Commit it red.

- [ ] **Step 2: Implement `add_pinned_sources`, `copy_missed`, `between`, and the four arms**

```python
def add_pinned_sources(layers: list[Layer], split) -> None:
    """Pinned copies of each layer's missed rows: the copy arms move these bytes H2D as the RAM tier would."""
    for L in layers:
        missed = (~L.hits[split]).nonzero().flatten()
        L.missed_phys = sorted({int(p) for p in (L.remap[missed].long() % PHYS).tolist()})
        L.pinned = {
            name: t[L.missed_phys].to("cpu").pin_memory() for name, t in L.rows.items()
        }


def copy_missed(L: Layer, stream=None) -> None:
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        for name, t in L.rows.items():
            for j, p in enumerate(L.missed_phys):
                t[p].copy_(L.pinned[name][j], non_blocking=True)
```

- If Step 1 shows that `L.rows` is not what the pointer tables reference (the poisoned run still matches), the
  launch reads `Exl3FusedMoE`'s own copy. Find it with `find_symbol Exl3FusedMoE/__init__ include_body=true`, copy
  into those tensors instead, and ledger a Ruling.
- In `main()`, for each split in `SPLITS[1:]`, call `add_pinned_sources(layers, split)` next to `prepare_static`
  (both before capture and before timing, as the existing comment requires), then capture the four arms.
- Their names contain `/<tag>`, so `split_of` picks them up.
- Add `import contextlib`.
- Update the module docstring's arm list with the four new arms, one line each.

- [ ] **Step 3: Run the test to see it pass**

Same command. Expected: `1 passed`, EXIT=0. Then run the whole parity file: all pass.

- [ ] **Step 4: Commit, push, and run the bench**

```bash
git add analysis/dsv41-drive/resident-first/split_launch_bench.py test/manual/dsv41/test_exl3_moe_split_parity_cuda.py
git commit -m "bench(dsv41): resident-first tail after real copies, with copy-engine-only overlap"
```

On divix01, in `wt-frontend` at the pushed commit:

```bash
/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh analysis/dsv41-drive/resident-first/run.sh /mnt/nvme1/frontend/resident-first; echo "EXIT=$?"
```

Expected:
- EXIT=0; not 75 (lock timeout), else rerun.
- `one` within 2% of bench-run2's 101.13 us/layer.
- `copy_only/4+2` near 2 x 13.3 MB / 13.6 GB/s ≈ 1.96 ms per layer, if the copies are on the copy engine. That is
  ~78 ms/token over 40 layers, which dominates every copy arm. The derived numbers above subtract it.
- Implied bandwidth, one, still ~790 GB/s.

- [ ] **Step 5: Apply Gate C**

Ledger: `Task 5: serial <x> us/layer, overlapped <y>, saving <x-y>, cache effect <z> -> Gate C <pass|fail>`.
- A pass means only that a resident-first design pass is justified. Write that up as a proposal in Task 6; do not
  build it here.
- A fail keeps resident-first shelved, now with the real-transfer evidence.

---

### Task 6: Write-up

**Files:**
- Modify: `DSV41_REFERENCE.md`: a new §27.15 after §27.14, and a line in §27.4 Next steps.

- [ ] **Step 1: Write §27.15**, "Decode RAM-miss frontend: sized, <result>". Cover:
  - Task 1's table for the existing trace: frontend, W1 p50/p90, budget-hit fraction, CW floor and spin, both bounds.
    Include the bound's derivation in two lines, and the fact that it overstates.
  - Which gates passed.
  - For each task that ran: commits, tests (with commands and counts), mutants, the arms table (A, B, traced A and B;
    ms/token, TTFT, output identical), and trace numbers.
  - Trace numbers include post -> F, copy bytes, copy busy, copy done after post, CW end after copy, and PCIe RX per
    link state.
  - The route and lifecycle coverage table from Task 3.
  - Task 5's arms table, the serial vs overlapped saving, the cache effect, and Gate C's verdict on resident-first.
  - A no-go, if that is the result, with the numbers that made it one.
  - Evidence paths on divix01 (`/mnt/nvme1/frontend/`).
- [ ] **Step 2: Commit and push**

```bash
git add DSV41_REFERENCE.md
git commit -m "docs(dsv41): RAM-miss frontend sizing and result (27.15)"
```

- [ ] **Step 3: Final review.** Whole-branch review on the most capable model, then the fix pass per the executing
  skill. Afterwards, remove the `wt-frontend`, `wt-frontend-mut` and `wt-frontend-arms` worktrees.
