# Expert-stream hot path with zero overhead: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RAM-miss request path, from the device's demand post to `kDemandDone`, meet four rules in the
production build:

- zero CPU copies of expert bytes;
- no metrics, clock or fault work;
- no heap allocation;
- no mutex, condition variable or other kernel sleep, except the io_uring completion wait, which the user chose to keep
  (D5).

**Architecture:**

1. **Delete the packed path.** Row images become the only reader, so `RowReader` is the tier's `Source`.
2. **Add a compile-time `Build` policy** (`ProdBuild`/`InstrBuild`) that removes stats, trace and fault state from the
   production instantiation. Production keeps a line-private, single-writer `CoreStats` block for the shutdown and
   fail-stop logs.
3. **Replace every per-request container** with fixed-capacity storage bounded by the wire format.
4. **Make the service thread the single owner of the tier state.**
   - Pausing hands ownership to the eager caller.
   - Copy completions reach the service through an SPSC ring.
   - Unpaused Python calls go through a command ring.
   - The copy engine's queue becomes an SPSC ring, with a futex wake only when its thread sleeps.

**Tech stack:**

- C++20, the header-only JIT host module (tvm-ffi), liburing 2.12, Linux futex;
- pytest, and an LD_PRELOAD counting shim written in C;
- `nm`/`objdump` (binutils);
- the dsv41 decode-arm harness (`benchmarks/dsv41_baseline/run_arm.sh`).

**Spec:** `docs/superpowers/specs/2026-09-29-hotpath-audit.md`, the audit. Its inventory rows (C1-C4, M1-M14, A1-A12,
L1-L15) are cited by ID below. The user's decisions on the spec's open questions:

| Decision | Choice |
|---|---|
| D1 | Keep a minimal `CoreStats` |
| D2 | Two builds |
| D3 | InstrBuild is picked implicitly when the trace or a fault env var is set |
| D4 | **Delete** the packed path |
| D5 | Keep `WAIT_MODE=block` |
| D6 | The watchdog's clock moves to the watchdog thread (20 ms) |
| D7 | Accept that COPYING leases are released one poll later |
| D8 | Keep the protocol-serialized ring |
| Scope | Phases 1, 2 and 3 now. Phase 0 (the WAIT_MODE A/B) is dropped |

Path prefixes used throughout:

- `H/` = `python/sglang/kernels/jit/csrc/moe/expert_stream/host/`
- `OPS` = `python/sglang/kernels/ops/moe/expert_stream_transport.py`
- `SVC` = `python/sglang/srt/layers/moe/exl3_ram_miss.py`
- `T/` = `test/registered/unit/kernels/`

## Global Constraints

### Branch and code movement

- Create the branch `cc/hotpath-zero-overhead` from `origin/master` = `ba01695c35`. Never push to or merge into
  `master`, never push to the retired `shared` remote, and never amend, rebase or force-push.
- Every commit message ends with these two trailers:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R
  ```
- Code reaches divix01 only by commit, then `git push origin cc/hotpath-zero-overhead`, then fetch into a private
  worktree: `/data/models/slang/nvfp4-work/wt-hotpath` (the branch) and `/data/models/slang/nvfp4-work/wt-hotpath-base`
  (`ba01695c35`, for the A arms and the baselines).
- No rsync, scp or `git archive`. Never run anything in `cc-expert-prediction/dsv41-direct-prod`.

### Running on divix01

- Every run uses `PYTHONPATH=$PWD/python` and first prints `sglang.__file__`, which must be under that worktree's
  `python/`.
- Any piped pytest reads `${PIPESTATUS[0]}`. A result that came through a pipe without it is unverified.
- The registered-suite target is `test/registered/unit/kernels` with `-p no:randomly`. Compare its counts to the same
  command at `ba01695c35` (Task 1), and record the exact command next to every count quoted. The layers suite
  `test/registered/unit/layers/moe` is run and compared the same way.
- CPU work runs under `taskset -c 0-63` with `OMP_NUM_THREADS=8`. GPU work, and any suite that touches CUDA, runs under
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. Cores 64-71 stay free.
- **Lock order:** `rowimg-disk.lock` first, then `cc-gpu.lock`. `run_arm.sh` takes `cc-gpu.lock` itself (non-blocking),
  so an arm driver holds `rowimg-disk.lock` and polls for the GPU. It never holds the GPU lock while waiting for the
  disk, and it is never started under an outer `flock` on the disk lock.
- **Foreign-process gates** match executable names (`pgrep -x`), never argv substrings. For pytest, match a python exe
  whose argv contains exactly the tokens `-m` `pytest`.
- **Mutants** go only in a private worktree (`git worktree add --detach /data/models/slang/nvfp4-work/wt-hotpath-mut
  <commit>`). Revert with `git checkout --`, re-run green, and record both results. Never commit a mutant.
- **generations.json:** the arm-only registration commit is reverted after the arms with
  `git revert --no-commit <sha>`, then a commit with the message
  `Revert the arm-only generations.json registration (<sha>)` and both trailers.

### Environment variables

- Read `.claude/skills/env-var-conventions/SKILL.md` before touching `python/sglang/srt/environ.py`.
- This plan removes exactly two knobs, `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` and
  `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES`, through the `_DEPRECATED_ENVS` registry: a set value warns, following
  the 2026-09-27 doorbell precedent.
- It adds no new environment variable. The build is picked from `SGLANG_DSV41_EXPERT_TRACE_PATH` and
  `SGLANG_TEST_DSV41_RAM_MISS_FAULT`, which already exist.

### Production behavior

- The production recipe's decode output must be byte-identical to master's (the Task 18 arms).
- Every row a service publishes must be byte-identical to `RamMissSetup.reference` (the Task 2 golden).
- The wire formats do not change: page, lease block, hot sidecar, prefetch page, `StageRecord`, fault tensor
  (`kFaultWords` stays 32; words 19 and 20 become reserved).
- `WAIT_MODE=block` stays the default. `RamThread` keeps its 50 µs idle sleep after `spin_us` of idle, because that is
  the idle path (spec L12).

### Hardware

- divix01 runs kernel `6.12.0-211.60.1.el10_2.x86_64` with liburing 2.12 and `ulimit -l` unlimited; the laptop runs
  7.0. Both kernels support O_DIRECT on tmpfs (6.6+), and the test fixture requires it.

### Run templates

Every task uses these, defined once here. `WT=/data/models/slang/nvfp4-work/wt-hotpath`.

```bash
# SYNC: laptop -> divix01 private worktree
git push origin cc/hotpath-zero-overhead
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-hotpath && git fetch origin \
  && git checkout --detach origin/cc/hotpath-zero-overhead && git log -1 --oneline && git status --short'

# CPU-TEST <files...>
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python \
  OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <files...> -q -rs -p no:randomly 2>&1 | tail -25; \
  echo "EXIT=${PIPESTATUS[0]}"'

# SUITE: kernels + layers/moe (touches CUDA in one tier test, so it takes the GPU lock)
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python \
  OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels test/registered/unit/layers/moe -q -p no:randomly 2>&1 | tail -5; \
  echo "EXIT=${PIPESTATUS[0]}"'
```

**LOCAL-TEST** is the same pytest command run on the laptop:

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python -m pytest <files...> -q -p no:randomly
```

A task step names a template and its files. A step marked "(laptop)" may use LOCAL-TEST. Every commit that a later
step runs on divix01 is followed by SYNC.

## Review Focus

These inputs are implied by the spec but not exercised by any task's main-line tests. Each line names its test and the
task that owns it.

1. **An eager Python call while the service thread runs unpaused.** `assign`, `touch` and `fill_begin` must refuse with
   a clear error; `mapping`, `slot_info`, `lease_entry` and `lru_order` must return a consistent snapshot. None may
   race the service. Test: `test_unpaused_eager_calls_refuse_or_snapshot` (Task 13).
2. **A launch that still sets the removed knobs, or a checkpoint without row images.** `ROW_IMAGES=0` or
   `PACK_WORKERS=8` warns and is ignored. Missing images refuse at startup with a message naming
   `scripts/dsv41/build_row_images.py`. Tests: `test_removed_ram_miss_knobs_warn` and
   `test_the_service_refuses_a_checkpoint_without_row_images` (Task 7).
3. **A copy completion that lands while the service is paused.** `pause()` must retire the COPYING lease itself, and
   not refuse a pause that has nothing outstanding. Test: `test_a_pause_retires_a_copy_that_completed_while_parked`
   (Task 14).
4. **A burst of `set_hot` larger than the command ring during a long read.** No command may be dropped, and they are
   applied in order before the next demand. Test: `test_a_set_hot_burst_past_the_ring_is_applied_in_order` (Task 13).
5. **A test-only or trace call on the production module.** It raises a `RuntimeError` naming the instrumented build;
   the build is chosen once, at `ExpertStreamHost` construction. Test: `test_test_only_calls_refuse_on_prod`
   (Task 10).

---

## Phase A: characterization (Tasks 1-4)

These tests land first and must pass at `ba01695c35`'s code. Later phases keep them green unchanged. Only the Task 3
assertions that are explicitly written to flip in Phases D and E are allowed to change.

### Task 1: Branch, worktrees and baselines

**Files:**
- Create: `analysis/dsv41-drive/hotpath/baseline.md`

**Interfaces:**
- Produces: the baseline counts (`kernels`, `layers/moe` and the GPU list below) that every later suite count is
  compared against.

- [ ] **Step 1: Create the branch.** On the laptop:

  ```bash
  cd /home/dimitri/data/divix/sglang-nvfp4 && git fetch origin
  git worktree add -b cc/hotpath-zero-overhead ../wt-hotpath-zero-overhead ba01695c35
  cd ../wt-hotpath-zero-overhead && git log -1 --oneline   # expect ba01695c35
  ```

- [ ] **Step 2: Create the divix01 worktrees.** On divix01:

  ```bash
  ssh divix01 'git -C /data/models/slang/sglang fetch origin \
    && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-hotpath-base ba01695c35 \
    && git -C /data/models/slang/nvfp4-work/wt-hotpath-base log -1 --oneline'
  ```

  `wt-hotpath` is created in Step 5, after the first push.

- [ ] **Step 3: Record the suite baselines at `ba01695c35`.** Run SUITE with `wt-hotpath-base` in place of
  `wt-hotpath`, and record the passed/skipped/failed counts and the command. Then run the GPU list under the GPU lock
  from `wt-hotpath-base`:

  ```bash
  ssh divix01 'cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
    CUDA_HOME=/usr/local/cuda-13.4 && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
    /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
      test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
      test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py \
      test/manual/dsv41/test_exl3_two_phase_parity_cuda.py test/manual/dsv41/test_exl3_two_phase_failure_cuda.py \
      test/manual/dsv41/test_exl3_two_phase_timing_cuda.py test/manual/dsv41/test_exl3_native_prefetch_cuda.py \
      test/manual/dsv41/test_exl3_task5_item4_gpu.py test/manual/dsv41/test_exl3_task5_item6_shutdown_gpu.py \
      test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"'
  ```

  `test_exl3_piece_stream_cuda.py` is not in the list: it exercises the packed path, and Task 6 deletes it.

- [ ] **Step 4: Write `baseline.md`** with the three counts, the commands, the date and the divix01 kernel
  (`uname -r`).

- [ ] **Step 5: Commit, push, create `wt-hotpath`, and SYNC.**

  ```bash
  git add analysis/dsv41-drive/hotpath/baseline.md
  git commit -m "analysis(hotpath): suite and GPU baselines at ba01695c35

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
  git push -u origin cc/hotpath-zero-overhead
  ssh divix01 'git -C /data/models/slang/sglang fetch origin && git -C /data/models/slang/sglang worktree add --detach \
    /data/models/slang/nvfp4-work/wt-hotpath origin/cc/hotpath-zero-overhead'
  ```

### Task 2: A golden for the tier, the leases, the SQEs and the bytes

**Files:**
- Create: `T/test_expert_stream_hotpath_golden.py`
- Create: `T/golden/hotpath_golden.json` (generated at master code, then committed)
- Create: `python/sglang/test/hotpath_script.py` (the scripted scenario, shared with Task 4)

**Interfaces:**
- Consumes: `ram_miss_setup(tmp_path, row_images=True, ...)`, `ExpertStreamHost`, `LeaseSim`, `new_page`,
  `page_word`, `read_rows_sqes`.
- Produces:
  - `hotpath_script.build_host(tmp_path, *, variant=None) -> (s, page, host, sim, dst)`, a pump-mode host in the
    production configuration: leases, two-phase, piece streaming, the copy engine on the CPU backend, and GPU hot;
  - `hotpath_script.SCRIPT`, a list of steps;
  - `hotpath_script.run_script(s, page, host, sim, dst) -> list[dict]`, one snapshot per step.

  The test compares the snapshots to the JSON.

**The scenario and its snapshots.** Each step drives the pump-mode host, then calls `host.pump()` until it returns 0.
That drain step is what keeps D7 (a copy completion is applied at the next pump) invisible to the golden. Each step
then snapshots the following, with times stripped:

- per row: `slot_info` and `mapping`;
- per ring index: `lease_entry(idx)`;
- page words: `demand_done`, `fatal`, and every posted record's status;
- per lane: row results `(tag, gen, host_slot, expert)`, the piece words, and `copy_done`;
- the functional counters, which Task 9 keeps in `CoreStats`: `served_requests`, `touch_only`, `rows_read`,
  `read_errors`, `overruns`, `late_after_fatal`, `evictions`, `deferred`, `deferred_reuse`, `no_victim`, `version`;
- a SHA-256 of every slab row whose state is READY, and the byte identity of each against `s.reference`.

- [ ] **Step 1: Write `python/sglang/test/hotpath_script.py`.**

```python
"""A scripted RAM-miss scenario in the production configuration (row images, leases, two-phase, piece streaming,
the copy engine on the CPU backend, GPU hot), driven in pump mode, snapshotting everything the device and the
eager callers can observe after each step. Plan 2026-09-29-hotpath-zero-overhead Tasks 2 and 4."""

from __future__ import annotations

import hashlib

import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import (
    HOT_RECORDS,
    ExpertStreamHost,
    hot_record_bytes,
    new_page,
    page_word,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

CAPACITY = 4
LAYERS = 2
EXPERTS = 8
DST_ROWS = 6
FUNCTIONAL = (
    "served_requests", "touch_only", "rows_read", "read_errors", "overruns", "late_after_fatal",
    "evictions", "deferred", "deferred_reuse", "no_victim", "version",
)


def build_host(tmp_path, *, variant=None, threaded=False):
    s = ram_miss_setup(tmp_path, capacity=CAPACITY, layers=LAYERS, experts=EXPERTS, row_images=True,
                       mirror_weights=(1.0, 1.0))
    page = new_page(pin=False)
    hot = torch.zeros(HOT_RECORDS * hot_record_bytes(EXPERTS), dtype=torch.uint8)
    kwargs = {} if variant is None else {"variant": variant}
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32),
                            direct=True, hot_page=hot, **kwargs)
    host.enable_lease_mode()
    host.enable_two_phase()
    host.enable_piece_stream()
    host.enable_copy_engine(-1, spin_us=200)
    dst = {}
    for row in range(LAYERS):
        dst[row] = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[row].items()}
        table = torch.tensor(
            [[t.data_ptr(), dst[row][n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[row].items()],
            dtype=torch.int64)
        host.set_copy_table(row, table, DST_ROWS)
    host.arm_copy_engine()
    host.enable_gpu_hot()
    return s, page, host, LeaseSim(host, page, s.slabs), dst


# (kind, row, lanes, extra): "post" posts and serves a request whose lanes are the experts listed; "ack" acknowledges
# the last served request's committed lanes; "term" voids its lanes with a terminal; "copy" releases every held copy
# mark; "hot" sets the row's VRAM-hot experts, which every later post of that row writes into its sidecar record.
# Every step ends with pump() until idle.
SCRIPT = [
    ("post", 0, [0, 1], {}),             # two misses: read, LOADING grant, pieces
    ("ack", 0, None, {}),
    ("post", 0, [0, 2], {"copy_engine": True, "dst": [0, 1]}),  # hit 0 through the copy engine, miss 2
    ("copy", 0, None, {}),
    ("ack", 0, None, {}),
    ("post", 0, [3, 4], {}),             # fills capacity 4: evictions from here on
    ("ack", 0, None, {}),
    ("post", 0, [5, 6], {}),             # evicts two LRU rows
    ("term", 0, None, {}),               # the device gives up: leases voided
    ("post", 1, [7], {"armed": False}),  # an unarmed touch-only record on row 1
    ("hot", 0, [5], {}),                 # from here expert 5 of row 0 is VRAM-hot: never a victim
    ("post", 0, [1, 2, 3], {}),
    ("ack", 0, None, {}),
    ("post", 1, [0, 1, 2, 3], {}),
    ("ack", 1, None, {}),
]


def _drain(host):
    for _ in range(64):
        if host.pump() == 0:
            return
    raise AssertionError("pump never went idle")


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def snapshot(s, page, host, sim, reqs):
    snap = {"rows": {}, "entries": [host.lease_entry(i) for i in range(16)], "page": {
        "demand_done": page_word(page, "demand_done"), "fatal": page_word(page, "fatal")}}
    for row in range(LAYERS):
        info = host.slot_info(row)
        ready = {}
        for slot, (state, expert, _leases, _gen) in enumerate(info):
            if state == 2 and expert >= 0:
                oracle = s.reference(s.tables.layer_ids[row], [expert])
                ready[str(slot)] = {
                    "expert": expert,
                    "digest": "".join(_digest(s.slabs[row][n][slot]) for n in EXL3_STREAMED_NAMES),
                    "exact": all(same_bytes(s.slabs[row][n][slot], oracle[n][0]) for n in EXL3_STREAMED_NAMES),
                }
        snap["rows"][str(row)] = {"slot_info": [list(i) for i in info], "mapping": host.mapping(row), "ready": ready}
    snap["results"] = [
        {"seq": r.seq, "lanes": [dict(sim.row_result(r, lane)) for lane in range(len(r.lanes))],
         "pieces": [sim.piece_word(r, lane) for lane in range(len(r.lanes))], "copy_done": list(sim.copy_done(r))}
        for r in reqs[-2:]
    ]
    counters = host.counters()
    snap["counters"] = {k: counters[k] for k in FUNCTIONAL}
    return snap


def write_hot_record(page, host, seq, hot):
    """The post kernel's GPU-hot sidecar record for demand ``seq`` (test_exl3_ram_miss_tier._post_gpu_hot's layout):
    written before the record is posted, as the device orders it. Every armed post needs one in GPU-hot mode, or the
    service fails the request as a lapped sidecar."""
    stride = hot_record_bytes(host.experts)
    start = (seq - 1) % HOT_RECORDS * stride
    record = host.hot_page[start : start + stride]
    record[:4].view(torch.int32)[0] = 0
    record[4:8].view(torch.int32)[0] = host.experts
    record[8 : 8 + (host.experts + 7) // 8].zero_()
    for expert in hot:
        record[8 + expert // 8] = int(record[8 + expert // 8]) | (1 << (expert % 8))
    record[:4].view(torch.int32)[0] = seq


def accept(sim, req):
    """The wait kernel's view of a served request, as test_exl3_ram_miss_copy_engine._accept_and_ack builds it: every
    lane's (host_slot, slot_generation), and the lanes the device will acknowledge (READY or LOADING; a COPYING lane
    is released by its copy's completion, never by an ack)."""
    ctx, ack_lanes = [], []
    for lane in range(len(req.lanes)):
        result = sim.row_result(req, lane)
        ctx.append((result["host_slot"], result["slot_generation"]))
        if result["gen"] == req.gen and result["tag"] in (lease.READY, lease.LOADING):
            ack_lanes.append(lane)
    return type("Waited", (), {"go": len(ctx), "ctx": ctx})(), ack_lanes


def next_seq(page) -> int:
    seq = (page_word(page, "demand_head") + 1) & 0xFFFFFFFF
    return seq or 1


def run_script(s, page, host, sim, dst):
    reqs, waits, snaps, hot = [], [], [], {0: [], 1: []}
    for kind, row, lanes, extra in SCRIPT:
        if kind == "post":
            write_hot_record(page, host, next_seq(page), hot[row])
            req = sim.post(row, lanes, armed=extra.get("armed", True), dst=extra.get("dst"),
                           copy_engine=extra.get("copy_engine", False))
            _drain(host)
            reqs.append(req)
            waits.append(accept(sim, req) if extra.get("armed", True) else None)
        elif kind == "ack":
            waited, ack_lanes = waits[-1]
            sim.ack(reqs[-1], waited, lanes=ack_lanes)
            sim.deliver()
        elif kind == "term":
            sim.terminal(reqs[-1], (1 << len(reqs[-1].lanes)) - 1)
            sim.deliver()
        elif kind == "copy":
            host.copy_engine_release(-1)
            assert host.copy_engine_idle(5.0)
        elif kind == "hot":
            hot[row] = list(lanes)
        _drain(host)
        snaps.append(snapshot(s, page, host, sim, reqs))
    return snaps
```

- [ ] **Step 2: Write `T/test_expert_stream_hotpath_golden.py`.**

```python
"""The hot path's observable behavior, pinned at ba01695c35 (plan 2026-09-29-hotpath-zero-overhead Task 2): the tier's
slots and map, the lease entries, the row results, piece words and CopyDone the device reads, the functional
counters, the bytes of every READY row (and their identity with the checkpoint), and the SQEs a row-image read
prepares. Every later phase of the plan (packed-path deletion, the Build policy, allocation and lock removal) must
leave this file green without editing the golden. Regenerate only at master's code:
``python test/registered/unit/kernels/test_expert_stream_hotpath_golden.py --regen``."""

import json
import sys
from pathlib import Path

import torch

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test import hotpath_script as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

GOLDEN = Path(__file__).parent / "golden" / "hotpath_golden.json"
SQE_REQUESTS = [(0, [0], [0]), (0, [1, 2, 3], [1, 2, 3]), (1, [7, 0, 5, 2], [0, 1, 2, 3])]


def scenario(tmp_path):
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        return hp.run_script(s, page, host, sim, dst)
    finally:
        host.stop()


def sqe_golden(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, layers=2, experts=8, row_images=True, mirror_weights=(1.0, 1.0))
    out = []
    for row, experts, slots in SQE_REQUESTS:
        result, sqes, info, _record = read_rows_sqes(s.tables, row, experts, slots, direct=True)
        oracle = s.reference(s.tables.layer_ids[row], experts)
        exact = all(same_bytes(s.slabs[row][n][slot], oracle[n][i])
                    for n in EXL3_STREAMED_NAMES for i, slot in enumerate(slots))
        out.append({"result": result, "sqes": [list(q) for q in sqes], "exact": exact,
                    "info": {k: info[k] for k in ("sqes", "descriptors", "credit")}})
    return out


def test_the_scripted_scenario_matches_the_golden(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    snaps = scenario(tmp_path)
    assert len(snaps) == len(golden["scenario"])
    for step, (got, want) in enumerate(zip(snaps, golden["scenario"])):
        assert json.loads(json.dumps(got)) == want, f"step {step} ({hp.SCRIPT[step][0]}) diverged"
    for snap in snaps:
        for row in snap["rows"].values():
            assert all(entry["exact"] for entry in row["ready"].values()), "a READY row differs from the checkpoint"


def test_row_image_reads_prepare_the_golden_sqes_and_land_exact_bytes(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    got = sqe_golden(tmp_path)
    assert json.loads(json.dumps(got)) == golden["sqes"]
    assert all(entry["exact"] and entry["result"] == 1 for entry in got)


if __name__ == "__main__" and "--regen" in sys.argv:
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path.home()) as a, tempfile.TemporaryDirectory(dir=Path.home()) as b:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps({"scenario": scenario(Path(a) / "s"), "sqes": sqe_golden(Path(b) / "q")},
                                     indent=1, sort_keys=True))
    print("wrote", GOLDEN)
```

`tempfile` under `$HOME` keeps the images on a disk filesystem. `ram_miss_setup` writes the fake checkpoint into the
directory it is given, so `Path(a) / "s"` must not exist yet; the fixture's `write_fake_exl3` creates it.

- [ ] **Step 3: Generate the golden at master code (laptop).**
  `python test/registered/unit/kernels/test_expert_stream_hotpath_golden.py --regen`.
  Check the JSON by eye:
  - every `exact` is `true`;
  - `fatal` is 0;
  - row 0's final `slot_info` shows two evictions (`evictions` ≥ 2);
  - step 2's lane 0 row result has tag COPYING (`lease.COPYING`);
  - step 9 (the terminal) leaves `lease_entry` inactive for its index.

  If an expectation fails, the script, not the code, is wrong: fix `SCRIPT` and regenerate.

- [ ] **Step 4: Run it (laptop), then on divix01.** Use LOCAL-TEST, then CPU-TEST, on
  `test/registered/unit/kernels/test_expert_stream_hotpath_golden.py`. Expected: 2 passed.

- [ ] **Step 5: Commit.**

  ```bash
  git add python/sglang/test/hotpath_script.py test/registered/unit/kernels/test_expert_stream_hotpath_golden.py \
          test/registered/unit/kernels/golden/hotpath_golden.json
  git commit -m "test(expert-stream): hot-path golden -- tier, leases, SQEs and bytes pinned at ba01695c35

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
  ```

### Task 3: An LD_PRELOAD counting shim for the service and copy threads

**Why a shim.** It measures the **production** build as shipped, with no hook compiled into it. It interposes `malloc`
and its family, `pthread_mutex_lock`/`trylock`, `pthread_cond_*`, `clock_gettime` and `nanosleep`/`clock_nanosleep`,
and counts calls made by the two threads that name themselves `*-ram-miss` and `*-copy-eng` (`pthread_setname_np`,
which the shim also interposes).

Calls from the host module go through its PLT, so the preload sees them, and `std::mutex::lock` inlines to
`pthread_mutex_lock`. vDSO calls made inside libc are not seen, and none are expected.

**Files:**
- Create: `python/sglang/test/hotpath_shim.c`
- Create: `python/sglang/test/hotpath_shim.py`
- Create: `T/test_expert_stream_hotpath_shim.py`
- Modify: `analysis/dsv41-drive/hotpath/baseline.md` (master's per-request counts)

**Interfaces:**
- Produces:
  - `hotpath_shim.build(tmp_dir) -> Path`, the compiled `.so`;
  - `hotpath_shim.run_child(shim, *, variant, requests, warmup, tmp) -> dict`, which runs the threaded production
    scenario in a subprocess with `LD_PRELOAD` and returns
    `{"service": {kind: count}, "copy": {kind: count}, "requests": n}`;
  - the kinds: `malloc`, `free`, `mutex`, `cond`, `clock`, `sleep`.

- [ ] **Step 1: Write `python/sglang/test/hotpath_shim.c`.**

```c
// LD_PRELOAD counting shim for plan 2026-09-29-hotpath-zero-overhead: counts allocator, mutex, condvar, clock and
// sleep calls made by the RAM-miss service thread (a name ending "-ram-miss") and the copy-engine thread ("-copy-eng")
// while armed. Test only; never loaded in production.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { kMalloc, kFree, kMutex, kCond, kClock, kSleep, kKinds };
static _Atomic long counts[2][kKinds];
static _Atomic int armed;
static __thread int who = -1;  // 0 service, 1 copy engine, -1 untracked

#define COUNT(kind) do { if (who >= 0 && atomic_load_explicit(&armed, memory_order_relaxed)) \
    atomic_fetch_add_explicit(&counts[who][kind], 1, memory_order_relaxed); } while (0)

extern void* __libc_malloc(size_t);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
extern void* __libc_memalign(size_t, size_t);
extern void __libc_free(void*);

void* malloc(size_t n) { COUNT(kMalloc); return __libc_malloc(n); }
void* calloc(size_t a, size_t b) { COUNT(kMalloc); return __libc_calloc(a, b); }
void* realloc(void* p, size_t n) { COUNT(kMalloc); return __libc_realloc(p, n); }
void* memalign(size_t a, size_t n) { COUNT(kMalloc); return __libc_memalign(a, n); }
void* aligned_alloc(size_t a, size_t n) { COUNT(kMalloc); return __libc_memalign(a, n); }
int posix_memalign(void** out, size_t a, size_t n) {
  COUNT(kMalloc);
  void* p = __libc_memalign(a, n);
  if (p == NULL) return 12;  // ENOMEM
  *out = p;
  return 0;
}
void free(void* p) { if (p) COUNT(kFree); __libc_free(p); }

// Resolved once, in a constructor, before any tracked thread exists: a lazy dlsym on a tracked thread could allocate
// (dlerror's buffer) and be counted as the hot path's.
static int (*real_mutex_lock)(pthread_mutex_t*);
static int (*real_mutex_trylock)(pthread_mutex_t*);
static int (*real_cond_wait)(pthread_cond_t*, pthread_mutex_t*);
static int (*real_cond_timedwait)(pthread_cond_t*, pthread_mutex_t*, const struct timespec*);
static int (*real_cond_clockwait)(pthread_cond_t*, pthread_mutex_t*, clockid_t, const struct timespec*);
static int (*real_cond_signal)(pthread_cond_t*);
static int (*real_cond_broadcast)(pthread_cond_t*);
static int (*real_clock_gettime)(clockid_t, struct timespec*);
static int (*real_nanosleep)(const struct timespec*, struct timespec*);
static int (*real_clock_nanosleep)(clockid_t, int, const struct timespec*, struct timespec*);
static int (*real_setname)(pthread_t, const char*);

__attribute__((constructor)) static void resolve(void) {
  real_mutex_lock = dlsym(RTLD_NEXT, "pthread_mutex_lock");
  real_mutex_trylock = dlsym(RTLD_NEXT, "pthread_mutex_trylock");
  real_cond_wait = dlsym(RTLD_NEXT, "pthread_cond_wait");
  real_cond_timedwait = dlsym(RTLD_NEXT, "pthread_cond_timedwait");
  real_cond_clockwait = dlsym(RTLD_NEXT, "pthread_cond_clockwait");
  real_cond_signal = dlsym(RTLD_NEXT, "pthread_cond_signal");
  real_cond_broadcast = dlsym(RTLD_NEXT, "pthread_cond_broadcast");
  real_clock_gettime = dlsym(RTLD_NEXT, "clock_gettime");
  real_nanosleep = dlsym(RTLD_NEXT, "nanosleep");
  real_clock_nanosleep = dlsym(RTLD_NEXT, "clock_nanosleep");
  real_setname = dlsym(RTLD_NEXT, "pthread_setname_np");
}

int pthread_mutex_lock(pthread_mutex_t* m) { COUNT(kMutex); return real_mutex_lock(m); }
int pthread_mutex_trylock(pthread_mutex_t* m) { COUNT(kMutex); return real_mutex_trylock(m); }
int pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m) { COUNT(kCond); return real_cond_wait(c, m); }
int pthread_cond_timedwait(pthread_cond_t* c, pthread_mutex_t* m, const struct timespec* t) { COUNT(kCond); return real_cond_timedwait(c, m, t); }
int pthread_cond_clockwait(pthread_cond_t* c, pthread_mutex_t* m, clockid_t k, const struct timespec* t) { COUNT(kCond); return real_cond_clockwait(c, m, k, t); }
int pthread_cond_signal(pthread_cond_t* c) { COUNT(kCond); return real_cond_signal(c); }
int pthread_cond_broadcast(pthread_cond_t* c) { COUNT(kCond); return real_cond_broadcast(c); }
int clock_gettime(clockid_t k, struct timespec* t) { COUNT(kClock); return real_clock_gettime(k, t); }
int nanosleep(const struct timespec* a, struct timespec* b) { COUNT(kSleep); return real_nanosleep(a, b); }
int clock_nanosleep(clockid_t k, int f, const struct timespec* a, struct timespec* b) { COUNT(kSleep); return real_clock_nanosleep(k, f, a, b); }

int pthread_setname_np(pthread_t thread, const char* name) {
  if (pthread_equal(thread, pthread_self())) {
    size_t n = strlen(name);
    if (n >= 9 && strcmp(name + n - 9, "-ram-miss") == 0) who = 0;
    else if (n >= 9 && strcmp(name + n - 9, "-copy-eng") == 0) who = 1;
  }
  return real_setname(thread, name);
}

void hotpath_shim_arm(int on) { atomic_store(&armed, on); }
void hotpath_shim_reset(void) { for (int t = 0; t < 2; ++t) for (int k = 0; k < kKinds; ++k) atomic_store(&counts[t][k], 0); }
long hotpath_shim_count(int thread, int kind) { return atomic_load(&counts[thread][kind]); }
```

- [ ] **Step 2: Write `python/sglang/test/hotpath_shim.py`.** It is the builder, the child program and the parent-side
  runner. The child drives the threaded production scenario:
  - a device loop posts, waits, accepts and acks;
  - a releaser thread frees copy marks;
  - every armed post writes its hot record.

```python
"""Builds the LD_PRELOAD counting shim and runs the threaded production-configuration scenario under it in a child
process (plan 2026-09-29-hotpath-zero-overhead Task 3). ``run_child`` returns per-thread counts over ``requests``
requests posted after ``warmup`` requests (the window is armed only around the measured requests)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

KINDS = ("malloc", "free", "mutex", "cond", "clock", "sleep")
SOURCE = Path(__file__).with_name("hotpath_shim.c")


def build(tmp_dir) -> Path:
    out = Path(tmp_dir) / "hotpath_shim.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o", str(out), str(SOURCE), "-ldl", "-lpthread"], check=True)
    return out


CHILD = textwrap.dedent(
    """
    import ctypes, json, sys, threading, time
    from pathlib import Path
    from sglang.kernels.ops.moe.expert_stream_transport import sim_wait
    from sglang.test import hotpath_script as hp

    shim_path, variant, requests, warmup, tmp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    shim = ctypes.CDLL(shim_path)
    shim.hotpath_shim_count.restype = ctypes.c_long
    s, page, host, sim, dst = hp.build_host(Path(tmp) / "s", variant=None if variant == "default" else variant)
    host.start_thread(fatal_wait_s=60.0, spin_us=50_000)  # 50 ms of spin: the measured window never idles into sleep
    stop = threading.Event()

    def releaser():
        while not stop.is_set():
            host.copy_engine_release(-1)
            time.sleep(0.0005)

    t = threading.Thread(target=releaser, daemon=True)
    t.start()

    def one(i):
        lanes = [i % hp.EXPERTS, (i + 3) % hp.EXPERTS]
        seq = hp.next_seq(page)
        hp.write_hot_record(page, host, seq, [])
        req = sim.post(0, lanes, dst=[0, 1], copy_engine=True)
        assert sim_wait(page, req.seq, 10.0) == 1, "request not served"
        waited, ack_lanes = hp.accept(sim, req)
        sim.ack(req, waited, lanes=ack_lanes)
        sim.deliver()

    for i in range(warmup):
        one(i)
    shim.hotpath_shim_reset()
    shim.hotpath_shim_arm(1)
    for i in range(warmup, warmup + requests):
        one(i)
    shim.hotpath_shim_arm(0)
    stop.set()
    t.join()
    counts = {name: {kind: shim.hotpath_shim_count(th, k) for k, kind in enumerate(%r)}
              for th, name in enumerate(("service", "copy"))}
    host.stop()
    print("HOTPATH-COUNTS " + json.dumps({**counts, "requests": requests}))
    """
    % (KINDS,)
)


def run_child(shim: Path, *, variant: str = "default", requests: int = 200, warmup: int = 50, tmp) -> dict:
    env = dict(os.environ, LD_PRELOAD=str(shim))
    proc = subprocess.run([sys.executable, "-c", CHILD, str(shim), variant, str(requests), str(warmup), str(tmp)],
                          env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("HOTPATH-COUNTS ")), None)
    assert proc.returncode == 0 and line, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(line.split(" ", 1)[1])
```

  The `variant` keyword is accepted as `"default"` until Task 8 adds `ExpertStreamHost(variant=...)`. From Task 8 on,
  the tests pass `"prod"`.

- [ ] **Step 3: Write `T/test_expert_stream_hotpath_shim.py`.** For now it holds only the shim's self-test and a
  characterization test that prints master's numbers. The zero-count assertions are added by Tasks 9, 11, 12 and 15.

```python
"""The counting shim (plan 2026-09-29-hotpath-zero-overhead Task 3), and the hot path's per-request allocator, mutex,
condvar, clock and sleep counts on the service and copy threads, measured on the build production loads."""

import ctypes
import subprocess
import sys
import textwrap

import pytest

from sglang.test import hotpath_shim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def shim(tmp_path_factory):
    return hotpath_shim.build(tmp_path_factory.mktemp("shim"))


def test_the_shim_counts_a_named_threads_calls_and_nothing_else(shim, tmp_path):
    """Self-test: a thread named x-ram-miss makes exactly 3 mallocs, 3 frees, 2 mutex locks and 1 clock read, and an
    unnamed thread's identical calls are not counted."""
    src = tmp_path / "probe.c"
    src.write_text(textwrap.dedent(r'''
        #define _GNU_SOURCE
        #include <pthread.h>
        #include <stdlib.h>
        #include <stdio.h>
        #include <time.h>
        #include <dlfcn.h>
        static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        static void work(void) { struct timespec t; for (int i = 0; i < 3; ++i) free(malloc(64));
          pthread_mutex_lock(&m); pthread_mutex_unlock(&m); pthread_mutex_lock(&m); pthread_mutex_unlock(&m);
          clock_gettime(CLOCK_MONOTONIC, &t); }
        static void* named(void* a) { pthread_setname_np(pthread_self(), "x-ram-miss"); work(); return 0; }
        static void* plain(void* a) { work(); return 0; }
        int main(void) { void (*arm)(int) = dlsym(RTLD_DEFAULT, "hotpath_shim_arm");
          long (*count)(int, int) = dlsym(RTLD_DEFAULT, "hotpath_shim_count"); arm(1);
          pthread_t a, b; pthread_create(&a, 0, named, 0); pthread_join(a, 0);
          pthread_create(&b, 0, plain, 0); pthread_join(b, 0);
          printf("%ld %ld %ld %ld\n", count(0, 0), count(0, 1), count(0, 2), count(0, 4)); return 0; }
    '''))
    exe = tmp_path / "probe"
    subprocess.run(["cc", "-O0", "-o", str(exe), str(src), "-ldl", "-lpthread"], check=True)
    out = subprocess.run([str(exe)], env={"LD_PRELOAD": str(shim)}, capture_output=True, text=True, check=True)
    malloc, free, mutex, clock = map(int, out.stdout.split())
    assert (malloc, free, mutex, clock) == (3, 3, 2, 1)


def test_characterize_the_hot_path(shim, tmp_path):
    """Not an assertion about zero: prints the per-request counts of the build the service loads, so the baseline
    (master) and every later phase can be compared. The zero-count tests below replace nothing here."""
    counts = hotpath_shim.run_child(shim, tmp=tmp_path)
    per = {th: {k: round(v / counts["requests"], 2) for k, v in counts[th].items()} for th in ("service", "copy")}
    print("HOTPATH per request", per)
    assert counts["service"]["malloc"] >= 0
```

- [ ] **Step 4: Run it (laptop, then CPU-TEST) with `-s`.** Record the printed per-request counts in `baseline.md`
  under "master, per request". Expectation from the spec, stated as a check on the shim rather than an assertion:
  - service `malloc` in the tens per request;
  - service `mutex` of about 5-8 per request, plus polls;
  - service `clock` ≥ 2 per request;
  - copy `malloc` > 0 (the per-pass deque).

  If the service's malloc count is 0 on master, the shim is not intercepting the host module. Stop and fix it
  (check `LD_PRELOAD` reached the child with `cat /proc/<pid>/maps`) before going on.

- [ ] **Step 5: Commit.**

  ```bash
  git add python/sglang/test/hotpath_shim.c python/sglang/test/hotpath_shim.py \
          test/registered/unit/kernels/test_expert_stream_hotpath_shim.py analysis/dsv41-drive/hotpath/baseline.md
  git commit -m "test(expert-stream): LD_PRELOAD counting shim for the service and copy threads; master's per-request counts

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
  ```

### Task 4: A threaded stress test of every concurrent party against the service thread

**Files:**
- Create: `T/test_expert_stream_hotpath_stress.py`

**Interfaces:**
- Consumes: `hotpath_script.build_host`, `accept`, `next_seq` and `write_hot_record` (Task 2).
- Produces: `run_stress(tmp_path, *, variant=None, seconds, seed) -> dict`, its invariant report, reused by the TSan run
  in Task 16.

**The parties:**
- **Device:** posts a seeded mix of requests on both rows, with misses, hits and copy-engine lanes; waits; acks, or
  voids with a terminal (10%).
- **Copy releaser:** releases copy marks every 0.5 ms.
- **Noise:** `set_hot`, `counters()`, `mapping`, `mapped_slot_generations`.
- **Pauser:** every 200 ms it quiesces the device, calls `pause` (a refusal is legal), makes eager reads and one
  `assign` of an expert that is not resident, then calls `resume`.

**Invariants at the end** (after a final pause, once everything is acked):
- `fatal == 0` and `read_errors == 0`;
- every lease is 0 and every `lease_entry` is inactive;
- `mapping(row)[e] == slot` exactly when `slot_info[slot]` is `(READY, e)`;
- every READY slot's bytes equal the checkpoint's;
- `served_requests + touch_only` equals the number of armed posts;
- no request timed out.

- [ ] **Step 1: Write the test.**

```python
"""Every party that touches the RAM-miss service concurrently -- the device, the copy thread's completions, unpaused
Python calls, and an eager caller's pause/resume -- against the running service thread, with the invariants that the
tier mutex protects today checked at the end (plan 2026-09-29-hotpath-zero-overhead Task 4). Green at ba01695c35;
the lock-free single-owner tier (Tasks 13-15) must keep it green."""

import random
import threading
import time

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import page_word, sim_wait
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test import hotpath_script as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import same_bytes

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


def run_stress(tmp_path, *, variant=None, seconds=8.0, seed=1):
    s, page, host, sim, dst = hp.build_host(tmp_path, variant=variant)
    host.start_thread(fatal_wait_s=60.0, spin_us=2000)
    rng = random.Random(seed)
    stop, quiet, idle = threading.Event(), threading.Event(), threading.Event()
    stats = {"armed": 0, "timeouts": 0, "pauses": 0, "refused": 0, "errors": []}

    def guard(fn):
        def run():
            try:
                fn()
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                stats["errors"].append(repr(error))
                stop.set()
        return threading.Thread(target=run, daemon=True)

    def device():
        while not stop.is_set():
            if quiet.is_set():
                idle.set()
                time.sleep(0.001)
                continue
            idle.clear()
            row = rng.randrange(hp.LAYERS)
            lanes = rng.sample(range(hp.EXPERTS), rng.randint(1, 3))
            hp.write_hot_record(page, host, hp.next_seq(page), [])
            use_copy = rng.random() < 0.5
            req = sim.post(row, lanes, dst=list(range(len(lanes))) if use_copy else None, copy_engine=use_copy)
            stats["armed"] += 1
            if sim_wait(page, req.seq, 10.0) != 1:
                stats["timeouts"] += 1
                stop.set()
                return
            waited, ack_lanes = hp.accept(sim, req)
            if rng.random() < 0.1:
                sim.terminal(req, (1 << len(req.lanes)) - 1)
            else:
                sim.ack(req, waited, lanes=ack_lanes)
            sim.deliver()

    def releaser():
        while not stop.is_set():
            host.copy_engine_release(-1)
            time.sleep(0.0005)

    def noise():
        while not stop.is_set():
            host.set_hot(rng.randrange(hp.LAYERS), rng.sample(range(hp.EXPERTS), 2))
            host.counters()
            host.mapping(rng.randrange(hp.LAYERS))
            host.mapped_slot_generations(rng.randrange(hp.LAYERS))
            time.sleep(0.0002)

    def pauser():
        while not stop.is_set():
            time.sleep(0.2)
            quiet.set()
            idle.wait(5.0)
            try:
                host.pause(5.0)
            except RuntimeError:
                stats["refused"] += 1
                quiet.clear()
                continue
            try:
                stats["pauses"] += 1
                host.slot_info(0)
                resident = set(e for e in host.mapping(0) if e >= 0)
                missing = [e for e in range(hp.EXPERTS) if host.mapping(0)[e] < 0]
                if missing:
                    host.assign(0, missing[0], protected=list(resident)[:1])
            finally:
                host.resume()
                quiet.clear()

    threads = [guard(f) for f in (device, releaser, noise, pauser)]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(20.0)
    host.copy_engine_release(-1)
    assert host.copy_engine_idle(5.0)
    host.pause(5.0)  # final quiesce: every ack was delivered, the copy thread is idle
    report = {"stats": stats, "fatal": page_word(page, "fatal"), "counters": host.counters(), "rows": {}}
    for row in range(hp.LAYERS):
        info, mapping = host.slot_info(row), host.mapping(row)
        report["rows"][row] = {"info": info, "mapping": mapping}
        for slot, (state, expert, leases, _gen) in enumerate(info):
            assert leases == 0, f"row {row} slot {slot} still leased"
            if state == 2:
                assert mapping[expert] == slot
                oracle = s.reference(s.tables.layer_ids[row], [expert])
                assert all(same_bytes(s.slabs[row][n][slot], oracle[n][0]) for n in EXL3_STREAMED_NAMES)
        for expert, slot in enumerate(mapping):
            assert slot < 0 or (info[slot][0] == 2 and info[slot][1] == expert)
    assert all(not host.lease_entry(i)["active"] for i in range(16))
    host.resume()
    host.stop()
    return report


def test_every_party_against_the_service_keeps_the_tier_invariants(tmp_path):
    report = run_stress(tmp_path, seconds=8.0, seed=1)
    stats, counters = report["stats"], report["counters"]
    assert stats["errors"] == [] and stats["timeouts"] == 0
    assert report["fatal"] == 0 and counters["read_errors"] == 0
    assert stats["armed"] > 200 and stats["pauses"] > 0
    assert counters["served_requests"] + counters["touch_only"] == stats["armed"]
```

- [ ] **Step 2: Run it** with LOCAL-TEST, then with CPU-TEST three times in a row
  (`--count` is not installed, so loop in bash). Expected: 3 × 1 passed at master code.

  If the pauser's `assign` of a non-resident expert fails because every slot is leased, that is legal. Wrap that one
  call with `pytest.raises` only if it is observed, and record why.

- [ ] **Step 3: Commit** (`test(expert-stream): threaded stress of device, copy completions, unpaused calls and pause
  against the service`, with the trailers).

---
## Phase B: delete the packed path (Tasks 5-7)

### Task 5: Move every suite onto row-image tables (the packed code still exists)

This task changes tests only, and they must stay green against the unchanged C++. It separates "the suites now test
the only reader" from "the other reader is gone" (Task 6), so a red in Task 6 can only mean a deletion broke
something.

**Files:**
- Modify: `python/sglang/test/dsv41_ram_miss_fixtures.py` (`ram_miss_setup`)
- Modify: every `T/test_exl3_ram_miss_*.py`, `T/test_expert_stream_*.py` and
  `test/registered/unit/layers/moe/test_exl3_*.py` that builds a host or reader with `direct=False`
- Test: the same files

**Interfaces:**
- Produces:
  - `ram_miss_setup(tmp_path, ..., row_images: bool = True)`;
  - `require_o_direct(path: str) -> None`, which raises `RuntimeError("... needs a filesystem that takes O_DIRECT
    ...")`.

- [ ] **Step 1: Write the failing fixture test** in `T/test_exl3_ram_miss_row_images.py`, next to
  `test_the_fixture_puts_a_reused_test_on_row_images_read_with_o_direct`:

```python
def test_the_fixture_builds_row_image_tables_by_default_and_requires_o_direct(tmp_path, monkeypatch):
    s = ram_miss_setup(tmp_path / "a")
    assert s.tables.row_images and all(path.endswith(".rows") for path in s.tables.paths)
    from sglang.test import dsv41_ram_miss_fixtures as fx

    monkeypatch.setattr(fx, "_takes_o_direct", lambda path: False)
    with pytest.raises(RuntimeError, match="O_DIRECT"):
        fx.ram_miss_setup(tmp_path / "b")
```

- [ ] **Step 2: Run it.** Use LOCAL-TEST on that test. Expected: FAIL. The default is still shard tables, and
  `_takes_o_direct` does not exist yet.

- [ ] **Step 3: Change the fixture.** In `dsv41_ram_miss_fixtures.py`:
  - add the helper below;
  - change the default to `row_images: bool = True`;
  - at the end of the `if row_images:` branch, call `require_o_direct(images_root)` on the first image file.

```python
def _takes_o_direct(path: str) -> bool:
    try:
        os.close(os.open(path, os.O_RDONLY | os.O_DIRECT))
        return True
    except OSError:
        return False


def require_o_direct(path: str) -> None:
    """The reader reads row images with O_DIRECT only (plan 2026-09-29-hotpath-zero-overhead): a test filesystem that
    refuses it (tmpfs before Linux 6.6, some overlays) must fail loudly here, not run a buffered read that production
    can no longer take."""
    if not _takes_o_direct(path):
        raise RuntimeError(f"{path}: the RAM-miss tests need a filesystem that takes O_DIRECT (tmpfs needs Linux 6.6+)")
```

  Where the branch builds `tables`:

```python
    tables = exl3_ram_miss_tables(layout, fmt.segment_map(), slabs, **mirrors)
    if row_images:
        require_o_direct(tables.paths[0])
```

- [ ] **Step 4: Pass `direct=True` everywhere.** Replace every `direct=False` in the tests that build an
  `ExpertStreamHost` or call `read_rows_*`/`read_rows_once` with `direct=True`:

  ```bash
  grep -rln "direct=False" test/registered/unit/kernels test/registered/unit/layers/moe \
    | xargs sed -i 's/direct=False/direct=True/g'
  git diff --stat
  ```

  Then inspect every changed line whose `direct` is not an argument of `ExpertStreamHost`, `read_rows_*`,
  `read_rows_once` or `Exl3ShardRowSource.for_layer`. Revert any other hit by hand: `Exl3ExpertFormat(...,
  direct=False)` and `Exl3ShardRowSource` are the eager Python reader, which keeps its buffered default.

- [ ] **Step 5: Pin the bounce-only tests to shard tables.** These are the tests `NOT_REUSED` and `ONE_READ_PER_PART`
  name in `T/test_exl3_ram_miss_row_images.py`, plus every test in `T/test_exl3_ram_miss_pack_workers.py`. In each,
  change its `ram_miss_setup(...)` call to pass `row_images=False` and its `direct=True` back to `direct=False`. The
  point is that they keep testing what they tested until Task 6 deletes them.

  List the names:

  ```bash
  python - <<'PY'
  import ast, pathlib
  src = pathlib.Path("test/registered/unit/kernels/test_exl3_ram_miss_row_images.py").read_text()
  tree = ast.parse(src)
  for node in tree.body:
      if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("NOT_REUSED", "ONE_READ_PER_PART"):
          print(sorted(ast.literal_eval(node.value)))
  PY
  ```

  For each name, `grep -rn "def <name>" test/registered/unit/kernels` gives its file.

- [ ] **Step 6: Run the suites.** Use LOCAL-TEST on `test/registered/unit/kernels test/registered/unit/layers/moe`.

  Every failure is one of three things:
  1. **A bounce-only assertion missed by Step 5.** Pin it the same way, and add its name to a
     `BOUNCE_ONLY_PINNED` list in the row-images test module, with a one-line reason.
  2. **Fake-expert dimensions that are not image-aligned** (a test's explicit `hidden`/`inter`). Change them to
     `ROW_IMAGE_DIM` multiples.
  3. **A genuine difference between the readers.** Stop and report it: the deletion would lose coverage.

  Expected: the pass count equals the Task 1 baseline, except for the new fixture test (+1).

- [ ] **Step 7: SUITE on divix01** (commit and SYNC first). Expected: the baseline counts + 1.

- [ ] **Step 8: Commit.**

  ```bash
  git commit -am "test(expert-stream): every RAM-miss suite reads row images with O_DIRECT; bounce-only tests pinned

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
  ```

### Task 6: Delete PackReader, PackPool, AnyReader and the bounce-only tests; refuse shard tables and buffered reads

**CRTP decision: keep `ReaderCore<Derived, ...>` for now.** With one derived reader, CRTP no longer earns polymorphism.
But the base/derived split still separates the io_uring pipeline (credit, descriptors, legs, generations, reap) from
the direct-mode destination and publishing hooks (`destination`, `advance`, `collect`, `publish_landed`). The pipeline
tests (`test_expert_stream_reader_split.py`, the SQE golden) are written against that boundary.

Folding `RowReader` into `ReaderCore` is a mechanical merge with no hot-path effect: after inlining, a CRTP call is a
direct call. It is recorded as a follow-up, not done here, because it would put a large relocation diff under the
lock-free changes this plan's review has to read.

`Build` (Task 8) is threaded through `ReaderCore` as an explicit parameter.

**Files:**
- Delete: `H/pack_reader.h`, `H/pack_pool.h`, `H/any_reader.h`, `T/test_exl3_ram_miss_pack_workers.py`,
  `test/manual/dsv41/test_exl3_piece_stream_cuda.py`
- Modify: `H/ram_tier.h`, `H/ram_thread.h`, `H/ffi_exports.h`, `H/row_reader.h`, `H/reader_core.h` (comments only),
  `H/read_fault.h`, `H/reader_base.h` (comments only), `H/row_tables.h`
- Modify: `OPS`, `SVC`, `python/sglang/test/dsv41_ram_miss_fixtures.py`, `T/test_exl3_ram_miss_row_images.py`, and the
  test modules that hold the pinned bounce-only tests (Task 5 Step 5)

**Interfaces:**
- Produces:
  - `using Source = RowReader<Layout, Reader>` in `HostExports` (Task 8 adds `Build`);
  - `RowReader(Tables tables, bool direct)`, which throws `std::invalid_argument` unless `tables.images && direct`;
  - `ExpertStreamHost(tables, *, page, slot_map, lease_block=None, hot_page=None, layout="exl3")`, with no `direct`
    and no `pack_workers`;
  - `read_rows_once(tables, row, experts, slots, *, step=..., layout=...)`, with no `direct`. The same holds for
    `read_rows_traced`, `read_rows_with_fault`, `read_rows_sqes` and `read_rows_pieces`;
  - `_table_args(tables, direct=True)`, which keeps its internal `direct` so a test can hand C++ `direct=0`.

- [ ] **Step 1: Write the failing refusal tests** in `T/test_exl3_ram_miss_row_images.py`:

```python
def test_the_reader_refuses_shard_tables(tmp_path):
    """The packed path is gone (plan 2026-09-29-hotpath-zero-overhead D4): a table that is not a row-image table is
    refused at open, naming the converter, never read through a bounce buffer."""
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables

    s = ram_miss_setup(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="row image"):
        exl3_ram_miss_tables(s.layout, s.fmt.segment_map(), s.slabs)  # no row_images: shard tables


def test_the_reader_refuses_buffered_reads_of_row_images(tmp_path):
    """Row images are read with O_DIRECT only: a buffered read would copy through the page cache (spec section 3)."""
    s = ram_miss_setup(tmp_path)
    fault = ops._fault_tensor()
    record = torch.zeros(ops._stage_words("exl3"), dtype=torch.int64)
    with pytest.raises(Exception, match="O_DIRECT"):
        ops._host_module("exl3").expert_stream_read_rows_traced(
            *ops._table_args(s.tables, False), 0, torch.tensor([1]), torch.tensor([0]), 8, fault, record)
```

  Read the exact positional signature of `expert_stream_read_rows_traced` from `read_rows_traced` in `OPS` before
  writing the call. The shape above follows `read_rows_sqes`'s; adjust the trailing arguments to match.

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: both FAIL. Shard tables are still accepted, and a buffered read
  succeeds.

- [ ] **Step 3: Delete the C++ packed path.**
  - `git rm` the three headers.
  - In `H/ffi_exports.h`, include `row_reader.h` instead of `any_reader.h` and set
    `using Source = RowReader<Layout, Reader>;`.
  - Remove the `pack_workers`/`pack_split` parameters from `open`, `read_rows`, `read_rows_traced`,
    `read_rows_faulted`, `read_rows_sqes` and `read_rows_pieces`.
  - Delete the `pack_pool_affinity` and `pack_worker_cpus` functions and their `TVM_FFI_DLL_EXPORT_TYPED_FUNC` lines.
  - Wherever the fault tensor's `pack_workers`/`pack_split` words were read, the reader no longer has `set_pack`: stop
    reading those words.

  In `H/row_reader.h`, replace the constructor and delete `set_pack`, `pack_workers`, `pack_split`, `pack_split_`,
  `packing_cpus` and `unfinished_jobs`:

```cpp
  // Row images read with O_DIRECT are the only reader (plan 2026-09-29-hotpath-zero-overhead D4): shard tables would
  // need the bounce-and-pack copy, and a buffered read copies through the page cache.
  RowReader(Tables tables, bool direct) : Base(std::move(tables), direct) {
    if (!t_.images) {
      throw std::invalid_argument(
          error_prefix<Layout>() + "the reader reads row image tables only (build them with "
          "scripts/dsv41/build_row_images.py); shard tables were refused");
    }
    if (!direct) throw std::invalid_argument(error_prefix<Layout>() + "row images are read with O_DIRECT only");
  }

  // Kept for the stage record's schema-5 fields: a row-image read never packs.
  unsigned pack_workers() const {
    return 0;
  }
  unsigned pack_split() const {
    return 0;
  }
```

  In `H/ram_tier.h`:
  - drop the constructor's `pack_workers` parameter and construct `reader_(std::move(tables), direct)`;
  - delete `packing_cpus()`.

  In `H/ram_thread.h`, delete the `else` block that narrows the unpinned service thread off the packing CPUs
  (`:339-352` at `ba01695c35`), so an unpinned thread keeps its inherited affinity.

  In `H/read_fault.h`, mark words 19 and 20 as reserved in the `fault_from` comment: "words 19-20 (formerly
  pack_workers, pack_split) are reserved and ignored".

  In `H/reader_core.h` and `H/reader_base.h`, update the comments that name `PackReader`, the bounce or the packing
  pool as a second reader: state the one reader, and keep "bank" and "slot" as pipeline-state names. Change no code
  line there.

- [ ] **Step 4: Update the Python side.**
  - In `OPS`, remove `pack_workers` and `pack_split` from `ExpertStreamHost.__init__`, `read_rows_*` and
    `_fault_tensor`. Word 19 and word 20 stay zero, and passing `pack_workers=`/`pack_split=` now raises `TypeError`.
  - Remove `direct` from the public helpers and from `ExpertStreamHost` (always `1` to C++).
  - Delete the `pack_pool_affinity` and `pack_worker_cpus` wrappers.
  - In `SVC`, stop passing `pack_workers=`, and have `exl3_ram_miss_tables` raise
    `ValueError("the RAM-miss reader reads row images only: ... scripts/dsv41/build_row_images.py")` when
    `row_images is None`. Delete its shard-table body (the code after the `if row_images is not None:` return, at
    `SVC:103-...`) and every helper only it used. Confirm each helper with `grep -n` before deleting it.
  - In the fixture, delete the `row_images=False` path and the parameter itself, and delete `require_o_direct`'s
    `_takes_o_direct` monkeypatch seam only if nothing uses it.

  Find the other callers of the removed arguments:

  ```bash
  grep -rn "pack_workers\|pack_split\|direct=\|row_images=False\|exl3_ram_miss_tables(" \
    python/sglang test/registered test/manual scripts benchmarks --include=*.py | grep -v "/analysis/"
  ```

  Every hit is either updated here or is `Exl3ShardRowSource`/`Exl3ExpertFormat`'s own `direct` (the eager Python
  reader), which stays.

  The characterization helpers are included. `python/sglang/test/hotpath_script.py` (`ExpertStreamHost(...,
  direct=True, ...)`) and `T/test_expert_stream_hotpath_golden.py` (`read_rows_sqes(..., direct=True)`) drop the
  keyword. That is the only edit these two files get in this task; the golden JSON is not touched.

- [ ] **Step 5: Delete the bounce-only tests** that Task 5 Step 5 pinned, and the `BOUNCE_ONLY_PINNED` list.
  - Delete `T/test_exl3_ram_miss_pack_workers.py` and `test/manual/dsv41/test_exl3_piece_stream_cuda.py`.
  - In `T/test_exl3_ram_miss_row_images.py`, delete the reuse machinery, since every suite now runs on images
    directly: `SETUP_USERS`, the three fixtures, `_row_images`, `_clone`, `NOT_REUSED`, `ONE_READ_PER_PART`, `_reuse`
    and its three calls, and `test_the_fixture_puts_a_reused_test_on_row_images_read_with_o_direct`.
  - Keep every test below the `# ---- The direct mode lands the same bytes as the bounce path ----` banner whose body
    does not compare against a shard read. A test that reads the same request through shard tables to compare bytes
    is rewritten to compare against `s.reference(...)`, the checkpoint oracle, which is what it guarded.

- [ ] **Step 6: Run everything.** Use LOCAL-TEST on the two refusal tests: they now PASS. Then run LOCAL-TEST on
  `test/registered/unit/kernels test/registered/unit/layers/moe` and on
  `test/registered/unit/kernels/test_expert_stream_hotpath_golden.py`. The golden must pass unedited.

  Record the delta against Task 5's count as `-(deleted) +2`. List every deleted test by name in the commit message
  body, in these groups:
  - the pack-workers file;
  - the bounce-only tests;
  - the reuse-machinery clones, which are no longer separate test IDs because the suites run on images themselves.

- [ ] **Step 7: SUITE and the golden on divix01** (commit and SYNC first). Same delta as Step 6.

- [ ] **Step 8: Commit** (`refactor(expert-stream)!: delete the packed path -- RowReader is the only reader; refuse
  shard tables and buffered reads`, with the deleted-test list and the trailers).

### Task 7: Environment, config and service: row images are mandatory

**Knob decision (env-var-conventions Rule 5).** Both knobs are **removed** and registered in `_DEPRECATED_ENVS` with a
note, so a set value warns and is otherwise ignored.

A knob whose only legal value is the default (`EnvBool(True)` that must never be set to False) states nothing a reader
can act on, and Rule 5's registry is the documented path for a knob that goes away. Warning rather than refusing
follows the doorbell precedent, and matters here: the production recipe and every archived arm script still set
`SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1` and `SGLANG_DSV41_RAM_MISS_PACK_WORKERS=8`.

The **refusal** a user needs, when a checkpoint has no images, stays where it is: `open_row_images` fails at startup and
names the converter.

**Files:**
- Modify: `python/sglang/srt/environ.py`, `python/sglang/srt/dsv41_config.py`, `SVC`,
  `benchmarks/dsv41_baseline/arm_env.py`, `benchmarks/dsv41_baseline/test_dsv41_baseline.py` (if it pins these keys),
  `test/registered/unit/test_dsv41_config.py`, `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`
- Test: `test/registered/unit/test_dsv41_config.py`, `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`

**Interfaces:**
- Consumes: `open_row_images`, whose error must name `scripts/dsv41/build_row_images.py`. Check it; if it does not,
  add the name to its `RuntimeError`.
- Produces: `Dsv41Config`, without `ram_miss_pack_workers` or `enable_ram_miss_row_images`.

- [ ] **Step 1: Write the failing tests.**

  In `test/registered/unit/test_dsv41_config.py`:

```python
def test_removed_ram_miss_knobs_warn(monkeypatch):
    import warnings
    from sglang.srt import environ

    monkeypatch.setenv("SGLANG_DSV41_RAM_MISS_PACK_WORKERS", "8")
    monkeypatch.setenv("SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        environ._handle_deprecated_envs()
    text = " ".join(str(w.message) for w in caught)
    assert "SGLANG_DSV41_RAM_MISS_PACK_WORKERS" in text and "SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES" in text
    assert not hasattr(environ.envs, "SGLANG_DSV41_RAM_MISS_PACK_WORKERS")
    assert not hasattr(environ.envs, "SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES")
```

  Check how `_DeprecatedEnv.apply` reports, `warnings.warn` or `logger.warning`, by reading `environ.py:2280-2295`. If
  it logs, use `caplog` instead of `catch_warnings`.

  In `test_exl3_ram_miss_service.py`, use the file's existing service-construction helper:

```python
def test_the_service_refuses_a_checkpoint_without_row_images(tmp_path):
    """Row images are mandatory (D4): mirror roots without exl3_row_images refuse at startup, naming the converter."""
    with pytest.raises(RuntimeError, match="build_row_images"):
        _start_service_on_mirrors_without_images(tmp_path)
```

  Write `_start_service_on_mirrors_without_images` from the module's existing service builder: mirror dirs holding
  shard copies and no `exl3_row_images`. Read the module's helpers first and reuse their arguments.

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL (the knobs still exist, and the service still has the
  flag-off path).

- [ ] **Step 3: Implement the knob removal.**
  - In `environ.py`, delete the two descriptors and their comments (`:1831-1835`, `:1865-1870` at `ba01695c35`). Edit
    the `SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM` comment so it no longer names pack workers ("Needs
    SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE and SGLANG_DSV41_ENABLE_RAM_MISS_LEASES. Off by default.").
  - Add the two entries to `_DEPRECATED_ENVS`, below the doorbell block:

```python
_PACKED_PATH_REMOVED_NOTE = (
    "The RAM-miss packed (bounce and pack) path was removed on 2026-09-29; row images read with O_DIRECT are the only "
    "reader. Build them with scripts/dsv41/build_row_images.py. Unset this env."
)
```

```python
    # The RAM-miss packed path and its knobs (removed 2026-09-29). The production recipe's archived arms still set
    # both, so a set value warns rather than refuses; a checkpoint without row images refuses at startup instead.
    **{
        name: _DeprecatedEnv(note=_PACKED_PATH_REMOVED_NOTE)
        for name in ("SGLANG_DSV41_RAM_MISS_PACK_WORKERS", "SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES")
    },
```

  Keep `_PACKED_PATH_REMOVED_NOTE` beside `_DOORBELL_REMOVED_NOTE`.

- [ ] **Step 4: Implement config and service.**
  - In `dsv41_config.py`, delete the two fields and their `envs` reads.
  - In `SVC`:
    - `open_service_row_images` loses its flag check, keeps its lease, mirror and `uring_direct` refusals with their
      messages rephrased from "SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES needs X" to "the RAM-miss service reads row
      images and needs X", and returns a `RowImageSet` (never `None`).
    - `check_piece_stream` loses its `row_images` argument and its publisher check.
    - Delete the `enable_prefill_fills and not tables.row_images` refusal (`SVC:807-809`).
    - The startup log line prints `row images on` unconditionally. Task 8 adds `build <variant>`.
  - In `arm_env.base_env()`, delete the two lines and the row-images comment block above
    `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES`.

  Also update the docstring of `benchmarks/dsv41_baseline/arm_env.py`, if it lists them, and every test that
  constructed `Dsv41Config(...)` with the two fields:

  ```bash
  grep -rn "ram_miss_pack_workers\|enable_ram_miss_row_images" python test benchmarks
  ```

  **LEASES becomes required.** The service now always reads row images, and images require leases
  (`SVC:331-333`), so a launch with `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` unset now refuses at startup with that
  message. This is intended (open decision 1). Add a test pinning the message:

```python
def test_the_service_refuses_to_start_without_leases(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="LEASES"):
        _start_service_with(tmp_path, leases=False)
```

  (`_start_service_with` is the module's existing builder with its lease flag exposed. Add the keyword if it has
  none.)

- [ ] **Step 5: Run the tests.** Use LOCAL-TEST on `test/registered/unit/test_dsv41_config.py`,
  `test/registered/unit/layers/moe` and `benchmarks/dsv41_baseline/test_dsv41_baseline.py`. Expected: PASS. Then run
  SUITE on divix01.

- [ ] **Step 6: Commit** (`refactor(dsv41)!: row images are mandatory -- remove SGLANG_DSV41_RAM_MISS_PACK_WORKERS and
  SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES (deprecated, warn)`, with the trailers).

---
## Phase C: the Build policy (Tasks 8-10)

### Task 8: `ProdBuild`/`InstrBuild` scaffolding, two host modules, and variant selection

This task is behavior-neutral: both builds compile the same code paths. Tasks 9 and 10 make them differ.

**Files:**
- Create: `H/build_policy.h`
- Create: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host_instr.cpp`
- Create: `test/registered/unit/kernels/conftest.py`, `test/registered/unit/layers/moe/conftest.py`
- Create: `T/test_expert_stream_build_variants.py`
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`, `H/reader_core.h`, `H/row_reader.h`,
  `H/ram_tier.h`, `H/ram_thread.h`, `H/copy_engine.h`, `H/ffi_exports.h`, `OPS`, `SVC` (the log line),
  `python/sglang/test/expert_stream_sources.py` (`host_sources()` gains the instr TU)

**Interfaces:**
- Produces, in C++:
  - `sglang::expert_stream::ProdBuild` and `InstrBuild`, each with `static constexpr bool kMetrics`,
    `static constexpr bool kFaults` and `static constexpr std::string_view kName` (`"prod"`/`"instr"`);
  - `template <class Derived, ExpertRowLayout Layout, AsyncFileReader Reader, class Build> class ReaderCore`, which
    exposes `using BuildType = Build;`;
  - `template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build> class RowReader`;
  - `RamTier<Source>`, which reads `using Build = typename Source::BuildType;`;
  - `template <class Build> class CopyEngine`;
  - `template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build> struct HostExports`;
  - the new export `expert_stream_build_name() -> std::string`.
- Produces, in Python:
  - `OPS.VARIANTS = ("prod", "instr")`;
  - `OPS.host_variant() -> str`;
  - `OPS._host_module(layout="exl3", variant=None)`;
  - `ExpertStreamHost(..., variant: Optional[str] = None)`, with the attribute `host.variant`;
  - `OPS._DEFAULT_VARIANT: Optional[str]`, the tests' override;
  - every module-level helper (`read_rows_*`, `sim_post`, `sim_wait`, `seqlock_stress`, `_stage_words`) takes
    `variant: Optional[str] = None`.

- [ ] **Step 1: Write the failing tests** in `T/test_expert_stream_build_variants.py`:

```python
"""The host transport ships two builds (plan 2026-09-29-hotpath-zero-overhead D2/D3): production, with no metrics,
trace or fault state on the request path, and instrumented. A service loads the instrumented one exactly when this
process writes a stream trace or injects a RAM-miss fault."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.setattr(ops, "_DEFAULT_VARIANT", None)


def test_the_production_build_is_the_default(no_override):
    with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(""), envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override(""):
        assert ops.host_variant() == "prod"


def test_a_stream_trace_selects_the_instrumented_build(no_override, tmp_path):
    with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(str(tmp_path / "t")):
        assert ops.host_variant() == "instr"


def test_a_ram_miss_fault_selects_the_instrumented_build(no_override):
    with envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override("5:1"):
        assert ops.host_variant() == "instr"


@pytest.mark.parametrize("variant", ops.VARIANTS)
def test_each_module_names_its_build_and_a_host_keeps_the_one_it_loaded(variant, tmp_path):
    assert str(ops._host_module("exl3", variant).expert_stream_build_name()) == variant
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant=variant)
    try:
        assert host.variant == variant
    finally:
        host.stop()


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="variant"):
        ops._host_module("exl3", "fast")
```

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL (`VARIANTS` and `host_variant` do not exist).

- [ ] **Step 3: Write `H/build_policy.h`.**

```cpp
// The compile-time build policy of the expert-stream host transport (plan 2026-09-29-hotpath-zero-overhead, spec
// section 4.1). ProdBuild carries no metrics, trace or fault state on the request path; InstrBuild carries all of it.
// Each is instantiated in its own module: exl3_ram_miss_host.cpp (prod), exl3_ram_miss_host_instr.cpp (instr).
#pragma once

#include <string_view>
#include <type_traits>

namespace sglang::expert_stream {

struct ProdBuild {
  static constexpr bool kMetrics = false;
  static constexpr bool kFaults = false;
  static constexpr std::string_view kName = "prod";
};

struct InstrBuild {
  static constexpr bool kMetrics = true;
  static constexpr bool kFaults = true;
  static constexpr std::string_view kName = "instr";
};

template <class B>
concept BuildPolicy = std::is_same_v<B, ProdBuild> || std::is_same_v<B, InstrBuild>;

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: Thread `Build` through the templates.**
  - `ReaderCore`: add `class Build` as the last template parameter, with `static_assert(BuildPolicy<Build>);` and
    `using BuildType = Build;`. `Build` is explicit and never derived from `Derived`, because `Derived` is incomplete
    while the base is instantiated.
  - `RowReader<Layout, Reader, Build>` derives from `ReaderCore<RowReader<Layout, Reader, Build>, Layout, Reader,
    Build>`.
  - `RamTier<Source>`: `using Build = typename Source::BuildType;`.
  - `RamThread<Tier>`: `using Build = typename Tier::Build;`.
  - `CopyEngine` becomes `template <class Build> class CopyEngine`, constructed as `CopyEngine<Build>` in
    `RamTier::enable_copy_engine`.
  - `HostExports<Layout, Reader, Build>`, with `using Source = RowReader<Layout, Reader, Build>;` and the export below.
    Add its line to `EXPERT_STREAM_HOST_EXPORTS`.

```cpp
  static std::string build_name() {
    return std::string(Build::kName);
  }
```

```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_build_name, Exports::build_name);                         \
```

  The instantiation files. In Task 8 both builds keep `FaultyReader` (Task 10 changes production's):

```cpp
// exl3_ram_miss_host.cpp -- the production build.
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/faulty_reader.h"
#include "expert_stream/host/ffi_exports.h"
#include "expert_stream/host/uring_reader.h"

namespace sglang {
using Exl3Reader = expert_stream::FaultyReader<expert_stream::UringReader>;
static_assert(expert_stream::AsyncFileReader<Exl3Reader>);
using Exl3HostExports = expert_stream::HostExports<exl3::Exl3RowLayout, Exl3Reader, expert_stream::ProdBuild>;
EXPERT_STREAM_HOST_EXPORTS(Exl3HostExports)
}  // namespace sglang
```

```cpp
// exl3_ram_miss_host_instr.cpp -- the instrumented build: stage trace, full counters and test-only faults.
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/faulty_reader.h"
#include "expert_stream/host/ffi_exports.h"
#include "expert_stream/host/uring_reader.h"

namespace sglang {
using Exl3Reader = expert_stream::FaultyReader<expert_stream::UringReader>;
static_assert(expert_stream::AsyncFileReader<Exl3Reader>);
using Exl3HostExports = expert_stream::HostExports<exl3::Exl3RowLayout, Exl3Reader, expert_stream::InstrBuild>;
EXPERT_STREAM_HOST_EXPORTS(Exl3HostExports)
}  // namespace sglang
```

- [ ] **Step 5: Python variant selection.** In `OPS`, replace `TransportBuild.host_source` with
  `host_sources: dict[str, str]`, and the loader with the following:

```python
LAYOUTS = {
    "exl3": TransportBuild(
        host_sources={"prod": "moe/exl3_ram_miss_host.cpp", "instr": "moe/exl3_ram_miss_host_instr.cpp"},
        device_source="moe/exl3_ram_miss.cuh",
        device_layout="sglang::exl3::Exl3RowLayout",
    )
}

VARIANTS = ("prod", "instr")
# Tests set this (test/registered/unit/{kernels,layers/moe}/conftest.py) to load the instrumented build, whose
# test-only entry points (faults, the stage trace) most of them use. None: host_variant() decides.
_DEFAULT_VARIANT: Optional[str] = None


def host_variant() -> str:
    """The host build a new service loads (spec D3): the instrumented one when this process writes a stream trace
    (SGLANG_DSV41_EXPERT_TRACE_PATH) or injects a RAM-miss fault (SGLANG_TEST_DSV41_RAM_MISS_FAULT), else production."""
    if _DEFAULT_VARIANT is not None:
        return _DEFAULT_VARIANT
    if envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get() or envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get():
        return "instr"
    return "prod"


def _host_module(layout: str = "exl3", variant: Optional[str] = None) -> Module:
    variant = host_variant() if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError(f"unknown host build variant {variant!r}; expected one of {VARIANTS}")
    return _host_module_cached(layout, variant)


@cache_once
def _host_module_cached(layout: str, variant: str) -> Module:
    return load_jit(
        f"expert_stream_host_{layout}_{variant}",
        cpp_files=[LAYOUTS[layout].host_sources[variant]],
        extra_cflags=["-fvisibility=hidden", "-fvisibility-inlines-hidden"],
        extra_ldflags=["-luring", "-lpthread", "-ldl"],
        header_only=False,
    )
```

  Import `envs` from `sglang.srt.environ` at the top of `OPS`, if it is not imported already.

  Then:
  - Give every module-level helper that calls `_host_module(layout)` a `variant: Optional[str] = None` keyword, and
    pass it through (`_host_module(layout, variant)`). The call sites are at `OPS:139,248,323,359,380,419,438,505,
    723,731,737` at `ba01695c35`.
  - `_host_layout_cached` keeps loading the default variant: the layout is the same in both.
  - In `ExpertStreamHost.__init__`, add `variant: Optional[str] = None`, then:

```python
        self.variant = host_variant() if variant is None else variant
        self._module = _host_module(self._layout, self.variant)
```

  The two conftests have the same body:

```python
"""Expert-stream tests load the instrumented host build unless a test names a variant: most use its test-only entry
points (faults, the stage trace). Plan 2026-09-29-hotpath-zero-overhead Task 8."""

import pytest


@pytest.fixture(autouse=True)
def _instrumented_expert_stream_host(monkeypatch):
    from sglang.kernels.ops.moe import expert_stream_transport as ops

    monkeypatch.setattr(ops, "_DEFAULT_VARIANT", "instr")
```

  In `SVC`, append `build %s` with `self.host.variant` to the "exl3 RAM miss thread started" log line. The Task 18
  driver checks for it.

  In `python/sglang/test/expert_stream_sources.py`, make `host_sources()` return both instantiation files first:

```python
def host_sources() -> tuple[Path, ...]:
    """The EXL3 host instantiations (production, instrumented) first, then every transport host header."""
    return (MOE / "exl3_ram_miss_host.cpp", MOE / "exl3_ram_miss_host_instr.cpp",
            *sorted((MOE / "expert_stream" / "host").glob("*.h")))
```

- [ ] **Step 6: Run the tests.** Use LOCAL-TEST on `T/test_expert_stream_build_variants.py`, then on the two full
  unit suites and the golden. Expected: all PASS.

  The characterization shim test's child passes `variant="default"`. Change `hotpath_shim.run_child`'s child program
  to pass `variant=None if variant == "default" else variant` (it already does) and have its test call
  `run_child(shim, variant="prod", tmp=...)`. Record the prod numbers, which still equal master's.

- [ ] **Step 7: SUITE on divix01**, commit and SYNC. Commit message: `feat(expert-stream): ProdBuild/InstrBuild policy
  and two host modules; the service picks instr for a trace or a fault`, with the trailers.

### Task 9: Gate metrics and trace; `CoreStats`; move the watchdog's clock; pace without clocks

**Files:**
- Modify: `H/build_policy.h` (counters), `H/tier_protocol.h`, `H/ram_tier.h`, `H/ram_thread.h`, `H/copy_engine.h`,
  `H/reader_core.h`, `H/row_reader.h`, `H/reader_base.h`, `H/ffi_exports.h`, `OPS`
- Modify: `T/test_exl3_ram_miss_stage_trace_causal.py` (`NON_TRACE_CLOCK_READS`), and the five `busy_since_ns()`
  callers (`T/test_exl3_ram_miss_lease_defer.py:79`, `T/test_exl3_ram_miss_lease_thread.py:199`,
  `test/manual/dsv41/test_exl3_two_phase_timing_cuda.py:333,363`, `test/manual/dsv41/test_exl3_task5_item4_gpu.py:152`)
- Test: `T/test_expert_stream_build_variants.py`, `T/test_expert_stream_hotpath_shim.py`

**Interfaces:**
- Consumes: `Build::kMetrics`.
- Produces:
  - `constexpr bool is_core_counter(int k)`;
  - `template <int N> struct LineCounters`, the single-writer relaxed counters;
  - `template <bool On> struct Stats`;
  - `RamTier::count<Counter K>(int64_t n = 1)` (service or owner) and `RamTier::copy_count<Counter K>(int64_t n = 1)`
    (copy thread);
  - `RamTier::busy_episode() -> uint64_t`, with `expert_stream_busy_episode` replacing `expert_stream_busy_since`;
  - `ExpertStreamHost.busy_episode() -> int`, replacing `busy_since_ns()`;
  - `OPS.CORE_COUNTERS: tuple[str, ...]`. A production host's `counters()` returns exactly those keys.

**Which counters are core.** These stay in production:

| Counter(s) | Why |
|---|---|
| `served_requests`, `touch_only`, `rows_read`, `read_errors` | the shutdown line |
| `overruns`, `late_after_fatal`, `no_victim`, `piece_stream_refused`, `slots_quarantined` | failure evidence |
| `evictions`, `deferred`, `deferred_reuse` | policy |
| `version` | functional: Python's LRU view |
| `running`, `spin_cpu` | set once |
| `copy_errors`, `copy_generation_mismatches` | the copy thread's failure evidence |

Everything else is a metric, present only in InstrBuild.

- [ ] **Step 1: Write the failing tests.** Add to `T/test_expert_stream_build_variants.py`:

```python
def test_a_production_host_reports_only_the_core_counters(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="prod")
    try:
        assert tuple(host.counters()) == ops.CORE_COUNTERS
    finally:
        host.stop()


def test_the_instrumented_host_still_reports_every_counter(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="instr")
    try:
        assert tuple(host.counters()) == tuple(ops.COUNTERS) and set(ops.CORE_COUNTERS) < set(ops.COUNTERS)
    finally:
        host.stop()
```

  Add to `T/test_expert_stream_hotpath_shim.py`:

```python
def test_the_prod_service_thread_reads_no_clock_per_request(shim, tmp_path):
    """Spec M3/M4/M8 and D6: the watchdog's episode word, turn-counted progress and iteration-budget pacing leave
    the service thread no clock read while it serves (the watchdog thread reads the clock instead)."""
    counts = hotpath_shim.run_child(shim, variant="prod", tmp=tmp_path)
    assert counts["service"]["clock"] == 0, counts
```

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL (`CORE_COUNTERS` does not exist, and the clock count is
  greater than 0).

- [ ] **Step 3: Add the counter types to `H/build_policy.h`.** `Counter` and `kCounterCount` stay in
  `tier_protocol.h`. `build_policy.h` holds the generic storage, and `is_core_counter` is defined in
  `tier_protocol.h` next to the enum.

```cpp
#include <atomic>
#include <cstdint>

namespace sglang::expert_stream {

// Counters with one writer each (the thread that owns the instance): add() is a relaxed load and a relaxed store of
// the sum, which on x86 is a plain add with no lock prefix and no fence. Readers on other threads load relaxed; a
// count may lag its writer, never tear. Own cache line(s), so a writer never shares a line with another thread.
template <int N>
struct alignas(64) LineCounters {
  int64_t v[N] = {};
  void add(int k, int64_t n = 1) {
    std::atomic_ref<int64_t> word(v[k]);
    word.store(word.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }
  void set(int k, int64_t value) {
    std::atomic_ref<int64_t>(v[k]).store(value, std::memory_order_relaxed);
  }
  int64_t get(int k) const {
    return std::atomic_ref<int64_t>(const_cast<int64_t&>(v[k])).load(std::memory_order_relaxed);
  }
};

// Metrics: InstrBuild keeps today's shared atomics (several writers per slot); ProdBuild has none.
template <bool On, int N>
struct Stats;

template <int N>
struct Stats<false, N> {
  void add(int, int64_t = 1) {}
  void store(int, int64_t) {}
  int64_t get(int) const {
    return 0;
  }
};

template <int N>
struct Stats<true, N> {
  std::atomic<int64_t> v[N]{};
  void add(int k, int64_t n = 1) {
    v[k].fetch_add(n, std::memory_order_relaxed);
  }
  void store(int k, int64_t value) {
    v[k].store(value, std::memory_order_relaxed);
  }
  int64_t get(int k) const {
    return v[k].load(std::memory_order_relaxed);
  }
};

static_assert(std::is_empty_v<Stats<false, 1>>, "the production build carries no metric storage");

}  // namespace sglang::expert_stream
```

  In `tier_protocol.h`, after the enum:

```cpp
// Counters the production build keeps (plan 2026-09-29-hotpath-zero-overhead D1): the shutdown line's served, rows and
// errors, the failure evidence a fail-stop message prints, the admission policy's outcomes, and the functional version.
constexpr bool is_core_counter(int k) {
  switch (k) {
    case kServedRequests: case kTouchOnly: case kRowsRead: case kReadErrors: case kEvictions: case kOverruns:
    case kLateAfterFatal: case kNoVictim: case kVersion: case kRunning: case kSpinCpu: case kDeferred:
    case kDeferredReuse: case kPieceStreamRefused: case kSlotsQuarantined: case kCopyErrors:
    case kCopyGenerationMismatches:
      return true;
    default:
      return false;
  }
}
```

- [ ] **Step 4: Rewrite every counter site in `RamTier`.** Replace the member
  `std::atomic<int64_t> counters_[kCounterCount];` with:

```cpp
  LineCounters<kCounterCount> core_;       // service thread (or the caller owning the tier while it is paused)
  LineCounters<kCounterCount> copy_core_;  // copy thread only
  Stats<Build::kMetrics, kCounterCount> stats_;  // InstrBuild: any thread, relaxed RMW

 public:
  template <Counter K>
  void count(int64_t n = 1) {
    if constexpr (is_core_counter(K)) {
      core_.add(K, n);
    } else {
      stats_.add(K, n);
    }
  }
  template <Counter K>
  void copy_count(int64_t n = 1) {
    if constexpr (is_core_counter(K)) {
      copy_core_.add(K, n);
    } else {
      stats_.add(K, n);
    }
  }
```

  Then make the mechanical edits:
  - `counters_[kX].fetch_add(n)` becomes `count<kX>(n)` on the service thread (and in any `RamTier` method a paused
    caller runs), or `copy_count<kX>(n)` in `copy_completed`, `copy_acked`, `prefetch_completed`, `copy_failed`,
    `release_copied` and every `CopyEngine` method. Task 14 moves the release onto the service, and its counters
    become `count<>` then.
  - The ternary at `ram_tier.h:598-602` becomes three explicit branches.
  - `counters_[kPiecePublishRefused].store(x)` becomes `stats_.store(kPiecePublishRefused, x)`.
  - `set_counter(i, v)` becomes `core_.set(i, v)`. Its callers pass only `kRunning`/`kSpinCpu`; add
    `static_assert`-able overloads if you prefer.
  - The `kCopyLatencyMaxNs` CAS loop moves into an `if constexpr (Build::kMetrics)` block, operating on
    `stats_.v[kCopyLatencyMaxNs]`.

  `CopyEngine` receives `RamTier*` (Task 12 templates it on the owner); until then give it `LineCounters<>* core` and
  `Stats<>* stats` pointers in place of `std::atomic<int64_t>* counters`.

  `counters(int64_t* out)` becomes:

```cpp
  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = core_.get(i) + copy_core_.get(i) + stats_.get(i);
  }
```

  After this step, `grep -n "counters_\[" H/*.h` must print nothing.

- [ ] **Step 5: Gate the stage trace** (`M9`). In `RamTier`, wrap the trace state in a conditional member:

```cpp
  struct TraceState {
    std::atomic<bool> on{false};
    std::mutex mutex;  // guards ring against a drain racing enable_trace (Python only)
    std::unique_ptr<StageRing> ring;
    StageRecord stage{};
    StageRecord* cur = nullptr;
    int64_t last_done = 0;
  };
  struct NoTraceState {};
  [[no_unique_address]] std::conditional_t<Build::kMetrics, TraceState, NoTraceState> trace_;
```

  Then:
  - Every use of the old `trace_on_`, `trace_mutex_`, `ring_`, `stage_`, `cur_` and `last_done_` becomes a `trace_.X`
    use inside `if constexpr (Build::kMetrics) { ... }`.
  - `begin_stage`, `end_stage` and `apply_pending_fault` get `if constexpr` bodies: empty in production.
  - A local `StageRecord* cur = nullptr; if constexpr (Build::kMetrics) cur = trace_.cur;` is what `serve` passes to
    `reader_.read`.
  - `enable_trace`, `drain_trace` and `trace_dropped` throw in production:

```cpp
  void enable_trace(size_t capacity) {
    if constexpr (!Build::kMetrics) {
      throw std::runtime_error(error_prefix<Layout>() + "the stage trace is in the instrumented host build only "
                               "(set SGLANG_DSV41_EXPERT_TRACE_PATH so the service loads it)");
    } else {
      // unchanged body, on trace_.*
    }
  }
```

  In `ReaderCore`:

```cpp
  struct NoTrace {};
  using TracePtr = std::conditional_t<Build::kMetrics, StageRecord*, NoTrace>;
```

  `Call::trace` becomes `[[no_unique_address]] TracePtr trace{};`, and `read(..., StageRecord* trace, ...)` stores it
  only `if constexpr (Build::kMetrics)`. Add these helpers:

```cpp
  template <class F>
  void on_trace(F&& f) {
    if constexpr (Build::kMetrics) {
      if (c_.trace != nullptr) f(*c_.trace);
    }
  }
  int64_t trace_stamp() {
    if constexpr (Build::kMetrics) {
      return stamp(c_.trace);
    } else {
      return 0;
    }
  }
```

  Rewrite every `if (c.trace) X;` / `if (trace) X;` in `reader_core.h` and `row_reader.h` as
  `on_trace([&](StageRecord& t) { X; });`, with `c.trace->` spelled `t.`. Every `stamp(c.trace)` becomes
  `trace_stamp()`. The sites are the M9 list in the spec. After the edit, this must print nothing:

  ```bash
  grep -n "c\.trace\|c_\.trace\|stamp(c" H/reader_core.h H/row_reader.h | grep -v "on_trace\|trace_stamp\|TracePtr\|if constexpr"
  ```

  `worker_stamp` in `read_fault.h` was the packing pool's; delete it.

- [ ] **Step 6: Move the watchdog's clock** (M3, D6). In `RamTier`, replace `std::atomic<int64_t> busy_since_{0};`:

```cpp
  // The watchdog's hung-request marker (D6): nonzero while a demand, an advisory or a fill is in service, a new value
  // per episode. The service stores it with no clock read; the watchdog thread times how long one value persists.
  std::atomic<uint64_t> busy_{0};
  uint64_t episodes_ = 0;  // the service thread's, or a fill's (they never run at once: a fill needs the pause)

  void begin_busy() {
    busy_.store(++episodes_, std::memory_order_release);
  }
  void end_busy() {
    busy_.store(0, std::memory_order_release);
  }

 public:
  uint64_t busy_episode() const {
    return busy_.load(std::memory_order_acquire);
  }
```

  Replace `busy_since_.store(now_ns())` (`ram_tier.h:224,1369,1827`) with `begin_busy()`, and `busy_since_.store(0)`
  (`:227,1415,1844`) with `end_busy()`. `RamThread::watch()` becomes:

```cpp
  void watch() {
    int64_t fatal_since = 0;
    bool reported = false;
    uint64_t episode = 0;      // the busy episode last seen, 0: idle
    int64_t episode_since = 0;  // when the watchdog first saw it
    while (!watch_stop_.load()) {
      const uint32_t fatal = load_acquire(page_ + kFatal);
      const int64_t now = now_ns();
      if (fatal != 0) {
        if (!reported) {
          reported = true;
          std::fprintf(stderr, "ERROR %srequest %u timed out or failed; the process must stop\n",
                       error_prefix<typename Tier::Layout>().c_str(), fatal);
          std::fflush(stderr);
        }
        if (fatal_since == 0) fatal_since = now;
      }
      // D6: the clock is read here, every 20 ms, never by the service. One episode held past fatal_wait is a hung
      // request; detection is at most 20 ms late against a 30 s deadline.
      const uint64_t busy = tier_->busy_episode();
      if (busy != episode) {
        episode = busy;
        episode_since = now;
      }
      const bool fatal_held = !stop_.load() && fatal_since != 0 && now - fatal_since > fatal_wait_ns_;
      const bool stuck = episode != 0 && now - episode_since > fatal_wait_ns_;
      if (fatal_held || stuck) {
        std::fprintf(stderr, "ERROR %s%s for %.1f s (fatal %u, busy %u); aborting instead of hanging decode\n",
                     error_prefix<typename Tier::Layout>().c_str(),
                     stuck ? "a request stayed in service" : "the fatal word stayed raised without the process stopping",
                     static_cast<double>(fatal_wait_ns_) / 1e9, fatal, load_acquire(page_ + kBusySeq));
        std::fflush(stderr);
        prctl(PR_SET_DUMPABLE, 0);
        std::abort();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
```

  The FFI `busy_since` becomes `busy_episode` (export `expert_stream_busy_episode`). In Python, `busy_since_ns()`
  becomes `busy_episode()`; update the five callers (the Files list). Their assertions are `== 0` or `max(...) > 0`,
  which hold unchanged for an episode.

- [ ] **Step 7: Turn-counted progress** (M4). In `ReaderCore::read`, replace the clock gate:

```cpp
    // Progress runs every turn: the caller's hook (retire_leases) returns at once when no lane is outstanding, and
    // gating it on a clock put a clock read on every turn of the hot path (spec M4).
    while (true) {
      if (progress) progress();
```

  Delete `kProgressIntervalNs` and `next_progress_ns`. Task 11 turns `progress` into a template parameter; until then
  it is still a `std::function` and `if (progress)` tests it for emptiness.

  If a test pins the 200 µs interval, update it to pin "called at least once per turn while a lane is outstanding",
  and name it in the commit body. Find them with `grep -rn "kProgressIntervalNs\|200 us\|progress" T/*.py`.

- [ ] **Step 8: Pace without clocks** (M8). In `RamThread`, compute the idle budget once in `start()`. Its clock reads
  run on the calling thread, at setup:

```cpp
  // How many idle polls approximate spin_ns (setup only: two clock reads on the caller's thread). An idle poll is at
  // least one _mm_pause plus three empty pumps, so the real spin is at least spin_ns; the budget is a floor.
  static uint64_t idle_budget(int64_t spin_ns) {
    constexpr int kProbe = 4096;
    const int64_t t0 = now_ns();
    for (int i = 0; i < kProbe; ++i)
      _mm_pause();
    const int64_t per_pause = std::max<int64_t>(1, (now_ns() - t0) / kProbe);
    return static_cast<uint64_t>(std::max<int64_t>(1, spin_ns / per_pause));
  }
```

  Set `spin_iters_ = idle_budget(spin_ns_);` before `thread_ = std::thread(...)`. In `run()`, delete `last_active`,
  and change the loop tail to:

```cpp
      if (tier_->pump_demand() || tier_->pump_prefetch() || tier_->pump_advice()) {
        idle = 0;
        iterations = 0;  // one heartbeat per request served
        continue;
      }
      if (++idle < spin_iters_) {
        _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));  // the idle path (spec L12): kept
      }
```

  In `CopyEngine::run`, apply the same change:
  - an `idle_iters`/`spin_iters_` budget from the same helper (move `idle_budget` to `reader_base.h` so both
    include it);
  - `issue()`'s `start`/`kCopyIssueNs`/`kCopyBytes` and `record_latency` go inside `if constexpr (Build::kMetrics)`;
  - `job.submit_ns = now_ns()` (`ram_tier.h:867`) and `read_ns` (`:568`) are set only `if constexpr (Build::kMetrics)`;
  - the stop deadline is read only once `stop_` is set.

- [ ] **Step 9: Replace `NON_TRACE_CLOCK_READS`.** In `T/test_exl3_ram_miss_stage_trace_causal.py`, the dict becomes
  the exact set of remaining non-trace clock lines:
  - the clock itself, and `stamp`;
  - `pause()`'s and `fill_wait`'s deadlines;
  - `CopyEngine::wait_idle`, `stop`'s deadline and the FFI `copy_engine_idle`;
  - the two `idle_budget` reads;
  - the watchdog's `const int64_t now = now_ns();`;
  - the `sim_wait`/`seqlock_stress` test exports;
  - the metrics-gated copy timings (inside `if constexpr`).

  Build it by running the test once and copying the actual `Counter`. Then check every line against this list, and
  add a comment on each group, as the file does today. Delete any key that no longer occurs; the dict must not list a
  line the sources do not have. The test compares exact counts, so a new clock read added later still fails it.

  Add a module constant and a test pinning that no clock line reads the clock on the service's request path:

```python
def test_no_ungated_clock_read_remains_on_the_request_path():
    text = joined_text(host_sources())
    assert "busy_since_" not in text and "kProgressIntervalNs" not in text
```

  (The shim test in Step 1 is the runtime proof; this is the source-level one.)

- [ ] **Step 10: Python.** Add to `OPS`:

```python
CORE_COUNTERS = (
    "served_requests", "touch_only", "rows_read", "read_errors", "evictions", "overruns", "late_after_fatal",
    "no_victim", "version", "running", "spin_cpu", "deferred", "deferred_reuse", "piece_stream_refused",
    "slots_quarantined", "copy_errors", "copy_generation_mismatches",
)
```

  Keep them in `COUNTERS` order (`sorted(CORE_COUNTERS, key=COUNTERS.index)`), since the prod test compares tuples.
  `counters()` becomes:

```python
    def counters(self) -> dict[str, int]:
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.expert_stream_counters(self.handle, out)
        values = dict(zip(COUNTERS, out.tolist()))
        return values if self.variant == "instr" else {k: values[k] for k in CORE_COUNTERS}
```

  Add a test asserting that `is_core_counter` and `CORE_COUNTERS` agree. It needs a new export,
  `expert_stream_core_counter_mask() -> int64`, a bitmask over `Counter` built from `is_core_counter`:

```python
def test_the_python_core_counters_are_the_hosts():
    mask = int(ops._host_module("exl3", "prod").expert_stream_core_counter_mask())
    assert {name for i, name in enumerate(ops.COUNTERS) if mask >> i & 1} == set(ops.CORE_COUNTERS)
```

  Search Python and the tests for readers of a non-core counter on a host that may be prod
  (`grep -rn 'counters()\["' python test`). The service's shutdown line and fail-stop message print whatever
  `counters()` returns, so they need no change. A test that reads a metric counter runs on the instr conftest
  default, so it needs no change either.

- [ ] **Step 11: Run the tests.** Use LOCAL-TEST on the build-variants test, the shim test's clock assertion, the
  causal trace test, the golden, and both unit suites. Expected: PASS; on the shim, the prod service's clock count
  is 0. Then run SUITE on divix01 and CPU-TEST on the shim test with `-s`, and record the per-request table in
  `baseline.md` under "after Task 9".

- [ ] **Step 12: Commit** (`perf(expert-stream): compile metrics and the stage trace out of ProdBuild; CoreStats; the
  watchdog times episodes; no clock on the service path`, with the trailers).

### Task 10: Gate the faults and the test-only exports; prove production carries none

**Files:**
- Modify: `H/reader_core.h`, `H/row_reader.h`, `H/ram_tier.h`, `H/read_fault.h`, `H/ffi_exports.h`,
  `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`
- Create: `T/test_expert_stream_prod_build_symbols.py`, `test/manual/dsv41/conftest.py` (the Task 8 conftest body: the
  GPU tests use faults and traces, so they load the instrumented build)
- Test: `T/test_expert_stream_build_variants.py`

**Interfaces:**
- Produces:
  - the production module is `HostExports<Exl3RowLayout, UringReader, ProdBuild>`, with no `FaultyReader`;
  - every test-only export raises `RuntimeError("<name> is test-only: it exists in the instrumented host build")` on
    production;
  - `OPS.TEST_ONLY_EXPORTS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests.**

  In `T/test_expert_stream_build_variants.py`:

```python
TEST_ONLY_CALLS = {
    "inject": lambda h: h.inject(fail_reads=True),
    "inject_fault": lambda h: h.inject_fault(ops._fault_tensor(part=0, part_error=5)),
    "inject_done_stall": lambda h: h.inject_done_stall(0.001),
    "inject_lease": lambda h: h.inject_lease(0, 0, 1),
    "trace": lambda h: h.enable_trace(16),
}


@pytest.mark.parametrize("name", sorted(TEST_ONLY_CALLS))
def test_test_only_calls_refuse_on_prod(name, tmp_path):
    s = ram_miss_setup(tmp_path, capacity=2)
    host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
                            variant="prod")
    try:
        with pytest.raises(RuntimeError, match="instrumented host build"):
            TEST_ONLY_CALLS[name](host)
    finally:
        host.stop()


def test_a_faulted_read_refuses_on_prod(tmp_path):
    s = ram_miss_setup(tmp_path)
    with pytest.raises(RuntimeError, match="instrumented host build"):
        ops.read_rows_with_fault(s.tables, 0, [1], [0], variant="prod", part=0, part_error=5)
```

  Read each method's exact name and signature in `OPS` (`inject`, `inject_fault`, `inject_done_stall`,
  `inject_lease`, and the trace enabler, which may be named `enable_stage_trace`) before fixing the lambdas.

  `T/test_expert_stream_prod_build_symbols.py`:

```python
"""The production host module carries no trace or fault machinery (plan 2026-09-29-hotpath-zero-overhead Task 10):
its symbol table has none of their types, and the instrumented module's does (so the check can fail)."""

import re
import shutil
import subprocess

import pytest

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

FORBIDDEN = ("StageRing", "ReadFault", "FaultyReader", "fault_from", "traced_clock_reads", "SqeRecord",
             "apply_pending_fault", "TraceState")


def _loaded_path(module_name: str) -> str:
    for line in open("/proc/self/maps"):
        path = line.split()[-1]
        if path.endswith(f"/{module_name}.so"):
            return path
    raise AssertionError(f"{module_name}.so is not mapped")


def _symbols(variant: str) -> str:
    ops._host_module("exl3", variant).expert_stream_build_name()
    path = _loaded_path(f"expert_stream_host_exl3_{variant}")
    return subprocess.run(["nm", "-C", path], capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(shutil.which("nm") is None, reason="binutils nm is not installed")
def test_prod_has_no_trace_or_fault_symbols_and_instr_has_them():
    prod, instr = _symbols("prod"), _symbols("instr")
    assert "RamTier" in prod, "the prod module's symbol table is stripped: this check cannot see anything"
    assert [name for name in FORBIDDEN if name in prod] == []
    assert [name for name in FORBIDDEN if name not in instr] == [], "the instrumented build lost its machinery"
```

  If the check finds `RamTier` absent because everything is inlined, keep the sanity anchor on a symbol that must
  exist. `HostExports<...ProdBuild>::open` is an exported entry's target and stays out-of-line; change the sanity
  string to `"HostExports"`. Record the choice in the commit message.

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL. Production still wraps `FaultyReader` and runs the fault
  code.

- [ ] **Step 3: Gate the faults.**

  In `ReaderCore`:

```cpp
  struct FaultState {
    ReadFault fault{};
    bool part_fired = false;
    int64_t retired = 0;
    Completion stale{0, 0};
    uint32_t stale_index = 0;
    bool stale_waiting = false;
    bool stale_armed = false;
    int64_t publishes = 0;
    std::vector<SqeRecord>* sqe_log = nullptr;
  };
  struct NoFaultState {};
  [[no_unique_address]] std::conditional_t<Build::kFaults, FaultState, NoFaultState> faults_;
```

  Then:
  - Move `fault_`, `part_fired_`, `retired_`, `stale_`, `stale_index_`, `stale_waiting_`, `stale_armed_`,
    `publishes_` and `sqe_log_` into it.
  - Every use becomes `faults_.X` inside `if constexpr (Build::kFaults)`: `read()`'s `max_outstanding` and
    `hold_until`, `admit_batch`'s poison and stale-arming, all of `reap()`'s fault branches, `process()`'s fault block
    and its `part` division, `retire()`'s stale and poison, `publish_collected`'s `twice` and `last_publish_delay`,
    `holding_for_probe`, `refill()`'s `sqe_log_`, and `RowReader::publish_landed`'s `delay` lambda.
  - `set_fault`, `set_sqe_log`, `fault_matches_sub` and `poison_slot` exist only with `requires Build::kFaults`.
  - `process()` keeps `++cqes_` only under `kMetrics` (it is a diagnostic). `stale_cqes_`, `generation_wraps_`,
    `cut_reads_`, `gap_cuts_`, `fixed_cuts_`, `fanout_sqes_` and `publish_refused_` move under `kMetrics` in the same
    way, via `Stats`-style empty members or `if constexpr`, keeping their getters, which return 0 in production.
  - `reap()` in production becomes:

```cpp
  void reap(bool ready) {
    Call& c = c_;
    if constexpr (Build::kFaults) {
      if (!ready && c.pending == 0 && !faults_.held.empty()) { /* the hold_ordinal release, unchanged */ return; }
    }
    if constexpr (Build::kMetrics) {
      if (c.trace != nullptr && c.submitted == 0) c.submitted = stamp(c.trace);
    }
    unsigned wait_nr = ready ? 0u : 1u;
    if constexpr (Build::kFaults) {
      if (faults_.fault.reverse_cqes && c.pending > 0) wait_nr = c.pending;
    }
    const int rc = submit(wait_nr);
    if (rc < 0) {
      const bool soft = rc == -EINTR || rc == -EAGAIN || rc == -EBUSY;
      if (!soft || ++c.soft_errors > kMaxSoftErrors) {
        c.failed = true;
        return;
      }
    } else {
      c.soft_errors = 0;
    }
    const int64_t returned = trace_stamp();
    completions_.clear();
    again_.clear();
    c.pending -= io_.reap(completions_);
    if constexpr (Build::kFaults) apply_reap_faults();  // reverse, hold_ordinal, stale: the old bodies, moved
    for (size_t k = 0; k < completions_.size(); ++k)
      process(completions_[k], returned);
    on_trace([&](StageRecord& t) {
      if (!completions_.empty()) {
        if (c.first_seen == 0) c.first_seen = returned;
        c.last_seen = returned;
      }
    });
    if (c.failed) return;
    for (uint32_t index : again_)
      queue_push(index);
  }
```

  Move `held_` into `FaultState` as `std::vector<Completion> held`. `read()`'s `held_.clear()` and the loop's
  `held_.empty()` tests become `if constexpr` / `held_empty()` returning `true` in production. `c.first_seen`,
  `c.last_seen` and `c.submitted` are used only for the trace, so move them inside `TracePtr`'s metrics-only struct
  or leave them as plain fields: they cost a store, not a branch. Leave them, and say so in a comment.

  In `RamTier`, `delay_ns_`, `fail_reads_`, `delay_after_`, `abandon_after_`, `done_stall_ns_`, `fault_mutex_`,
  `pending_fault_` and `fault_pending_` move into a `TierFaults` struct behind
  `[[no_unique_address]] std::conditional_t<Build::kFaults, TierFaults, NoTierFaults> faults_;`. `serve()`'s fault
  lines (`:1691-1701`) and `pump_demand`'s stall (`:189-191`) become `if constexpr (Build::kFaults)`.

  In production, `serve()`'s read call passes an abandon predicate that is the advisory rule alone:

```cpp
      const auto abandon = [&](size_t admitted) {
        bool stop = advisory && (demand_pending() || pause_requested_.load() || stop_requested_.load());
        if constexpr (Build::kFaults) {
          const int64_t after = faults_.abandon_after.load();
          stop = stop || (advisory && after > 0 && admitted >= static_cast<size_t>(after));
        }
        return stop;
      };
```

  `inject`, `inject_fault` and `inject_done_stall` throw in production, with the message from the Interfaces block.
  Write that message once as a helper:

```cpp
  [[noreturn]] static void test_only(const char* name) {
    throw std::runtime_error(std::string(name) + " is test-only: it exists in the instrumented host build");
  }
```

  In `H/ffi_exports.h`, these exports call `test_only("<name>")` under `if constexpr (!Build::kFaults)`:
  `read_rows_faulted`, `read_rows_sqes`, `inject`, `inject_fault`, `inject_done_stall`, `inject_lease`,
  `copy_engine_fail`, `copy_engine_ballast`, `seqlock_stress` and `trace_clock_reads`. `trace_enable`/`trace_drain`
  refuse through `RamTier` (Task 9). `read_rows_traced` and `read_rows_pieces` stay available in both builds (they
  are reads with an optional record): in production, a non-null record simply stays zero. Document that in their
  Python docstrings.

  `sim_post`, `sim_wait`, `publish_piece`, `piece_geometry` and `piece_runs` stay in both builds: they are
  device-protocol simulators and pure geometry, with no fault state. The shim's production child uses `sim_wait`.

  In `exl3_ram_miss_host.cpp`, use `using Exl3Reader = expert_stream::UringReader;` and drop the `faulty_reader.h`
  include.

  Add `OPS.TEST_ONLY_EXPORTS` listing those names, for documentation. Make the Python wrappers for them raise the same
  error when `variant == "prod"`, before any tensor is built, so the error names the build even before C++ is
  reached.

- [ ] **Step 4: Run the tests.** Use LOCAL-TEST on the two test files, the golden, the shim tests and both unit
  suites. Expected: PASS. The golden and the stress test run on instr (the conftest default); the shim's clock test
  runs on prod.

  Add a production run of the golden: parametrize `test_the_scripted_scenario_matches_the_golden` over
  `variant in ("prod", "instr")`, passing it to `hp.build_host(tmp_path, variant=variant)`. The JSON is unchanged,
  because the functional counters it records are core counters present in both builds.

- [ ] **Step 5: SUITE on divix01**, commit and SYNC. Commit message: `perf(expert-stream): faults and test-only
  exports are InstrBuild only; ProdBuild reads through UringReader; nm proof`, with the trailers.

---
## Phase D: allocation removal (Tasks 11-12)

### Task 11: No heap allocation in the tier or the reader

**Files:**
- Create: `H/fixed_vec.h`
- Modify: `H/tier_protocol.h` (`Request`, `read_record`, `listed`), `H/ram_tier.h`, `H/reader_core.h`,
  `H/row_reader.h`, `H/ffi_exports.h` (the read entry points' callbacks), `H/read_fault.h` (`abandon_after`)
- Test: `T/test_expert_stream_hotpath_shim.py`, `T/test_expert_stream_fixed_vec.py` (new)

**Interfaces:**
- Produces:
  - `template <class T, size_t N> class FixedVec`, with the members `push_back`, `assign`, `clear`, `size`, `empty`,
    `begin`, `end`, `operator[]`, `back`, `span()` and an implicit `operator std::span<const T>()`;
  - `constexpr size_t kWanted = 2 * kMaxIds + kLeaseLanes;` (24);
  - `ReaderCore::read(int64_t layer, std::span<const int32_t> experts, std::span<const int64_t> slots, size_t step,
    Abandon&& abandon, StageRecord* trace, std::vector<uint8_t>* packed, size_t max_reading_rows, Progress&& progress,
    const PiecePublish* publish)`, templated on `Abandon` and `Progress`;
  - `struct NoProgress { void operator()() const {} };`.

- [ ] **Step 1: Write the failing tests.**

  `T/test_expert_stream_fixed_vec.py` compiles a tiny C++ program against the header, so a regression in the
  container is caught without the module:

```python
"""FixedVec, the service's per-request container (plan 2026-09-29-hotpath-zero-overhead Task 11): bounded by the wire
format, never allocating, loud on overflow."""

import subprocess
import textwrap

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_fixed_vec_pushes_assigns_clears_and_throws_on_overflow(tmp_path):
    src = tmp_path / "t.cpp"
    src.write_text(textwrap.dedent(f'''
        #include "{MOE}/expert_stream/host/fixed_vec.h"
        #include <cassert>
        #include <stdexcept>
        using sglang::expert_stream::FixedVec;
        int main() {{
          FixedVec<int, 3> v; v.push_back(1); v.push_back(2);
          assert(v.size() == 2 && v[1] == 2 && v.back() == 2 && !v.empty());
          int src[] = {{7, 8, 9}}; v.assign(src, src + 3); assert(v.size() == 3 && v[2] == 9);
          std::span<const int> s = v; assert(s.size() == 3 && s[0] == 7);
          bool threw = false; try {{ v.push_back(4); }} catch (const std::logic_error&) {{ threw = true; }}
          assert(threw && v.size() == 3);
          v.clear(); assert(v.empty());
          static_assert(sizeof(FixedVec<int, 3>) <= 3 * sizeof(int) + sizeof(size_t));
          return 0;
        }}
    '''))
    exe = tmp_path / "t"
    subprocess.run(["c++", "-std=c++20", "-O1", "-o", str(exe), str(src)], check=True)
    subprocess.run([str(exe)], check=True)
```

  Add to `T/test_expert_stream_hotpath_shim.py`:

```python
@pytest.mark.parametrize("variant", ["prod", "instr"])
def test_the_service_thread_allocates_nothing_per_request(shim, tmp_path, variant):
    """Spec A1-A6, A9: after warm-up the service thread makes no allocator call while it serves (the instrumented
    build too, with its trace off: its metrics are fixed-size)."""
    counts = hotpath_shim.run_child(shim, variant=variant, tmp=tmp_path)
    assert counts["service"]["malloc"] == 0 and counts["service"]["free"] == 0, counts
```

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: the `FixedVec` test fails to compile (no header), and the
  allocation test fails with the service's malloc > 0.

- [ ] **Step 3: Write `H/fixed_vec.h`.**

```cpp
// A fixed-capacity vector for the service's per-request state (plan 2026-09-29-hotpath-zero-overhead Task 11): the
// wire format bounds every per-request list (kMaxIds need and protect ids, kLeaseLanes lanes), so nothing on the
// request path needs the heap. Overflow throws: it means a caller broke that bound, never a valid request.
#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace sglang::expert_stream {

template <class T, size_t N>
class FixedVec {
 public:
  void push_back(const T& value) {
    if (n_ == N) overflow();
    data_[n_++] = value;
  }
  template <class It>
  void assign(It first, It last) {
    clear();
    for (; first != last; ++first)
      push_back(*first);
  }
  void clear() {
    n_ = 0;
  }
  size_t size() const {
    return n_;
  }
  bool empty() const {
    return n_ == 0;
  }
  T* begin() {
    return data_;
  }
  T* end() {
    return data_ + n_;
  }
  const T* begin() const {
    return data_;
  }
  const T* end() const {
    return data_ + n_;
  }
  T& operator[](size_t i) {
    return data_[i];
  }
  const T& operator[](size_t i) const {
    return data_[i];
  }
  T& back() {
    return data_[n_ - 1];
  }
  std::span<const T> span() const {
    return {data_, n_};
  }
  operator std::span<const T>() const {
    return span();
  }

 private:
  [[noreturn]] static void overflow() {
    throw std::logic_error("expert stream: a per-request list exceeded its wire-format bound");
  }
  T data_[N]{};
  size_t n_ = 0;
};

// Membership in any contiguous id list (FixedVec, std::vector, std::span).
template <class Ids, class Id>
bool listed(const Ids& ids, Id id) {
  return std::find(std::begin(ids), std::end(ids), id) != std::end(ids);
}

}  // namespace sglang::expert_stream
```

  Delete `tier_protocol.h`'s vector-only `listed`.

- [ ] **Step 4: Make `Request` fixed-size** (A1-A3, A11). In `tier_protocol.h`:

```cpp
constexpr size_t kWanted = 2 * kMaxIds + kLeaseLanes;  // a request's distinct experts: need, protect, lanes

struct Request {
  uint32_t seq = 0;
  int64_t row = 0;
  uint32_t after = 0;
  bool armed = true;
  uint32_t lanes = 0;
  FixedVec<int32_t, kMaxIds> need;
  FixedVec<int32_t, kMaxIds> protect;
  const uint8_t* hot_bitmap = nullptr;  // GPU hot mode: RamTier::hot_scratch_, valid until the next record read
  // Lease mode: the device's lane list and 56-bit request generation, from the lane request (not the record).
  uint64_t gen = 0;
  FixedVec<int32_t, kLeaseLanes> lane_experts;
  FixedVec<int32_t, kLeaseLanes> lane_dst;  // the plan's destination slot per lane, -1 unknown
  uint32_t lane_flags = 0;                  // kLeaseLrFlag*
};
```

  `read_record`'s two `assign` lines compile unchanged. In `RamTier`, add these members and initialize them in the
  constructor body:

```cpp
  std::vector<uint8_t> hot_scratch_;  // (experts_+7)/8 bytes, sized at construction: the hot bitmap of the record read
  std::vector<uint8_t> fill_packed_;  // run_fill's per-row packed flags, sized to the largest row capacity
```

```cpp
    hot_scratch_.assign(static_cast<size_t>((experts_ + 7) / 8), 0);
    packed_.reserve(kWanted);
    piece_targets_.reserve(kWanted);
    int64_t widest = 0;
    for (int64_t c : capacity)
      widest = std::max(widest, c);
    fill_packed_.reserve(static_cast<size_t>(widest));
```

  `read_gpu_hot` becomes:

```cpp
  bool read_gpu_hot(uint32_t expected, Request* request) {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    uint32_t count = 0;
    std::memcpy(&count, record + 4, 4);
    if (count != experts_) return false;
    const size_t bytes = hot_scratch_.size();
    std::memcpy(hot_scratch_.data(), record + kHotHeaderBytes, bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire(record) != expected) return false;
    if (experts_ % 8 != 0 && (hot_scratch_[bytes - 1] & static_cast<uint8_t>(~((1u << (experts_ % 8)) - 1u))) != 0)
      return false;
    request->hot_bitmap = hot_scratch_.data();
    return true;
  }
```

  It loses `const`, because it writes the service-owned scratch. `apply_gpu_hot` reads `request.hot_bitmap[expert / 8]`
  unchanged. `read_lane_request`'s two `assign(ptr, ptr + count)` lines compile unchanged with `FixedVec`.

- [ ] **Step 5: Make the `serve`/`defers` locals fixed-size** (A4, A5), and use spans.
  - In `defers()`: `FixedVec<int32_t, kWanted> wanted;`.
  - In `serve()`: `FixedVec<int32_t, kWanted> wanted, missing; FixedVec<int64_t, kWanted> slots;`. Everything else is
    unchanged: the loops use `push_back`, `listed`, `size()` and `operator[]`.
  - Change the parameter types:
    - `census_locked(int64_t row, std::span<const int32_t> wanted)`;
    - `take_slot_locked(int64_t row, std::span<const int32_t> protect, bool fallback, int64_t* evicted)`;
    - `take_admit_slot_locked(..., std::span<const int32_t> protect, ...)`;
    - `init_piece_words_locked(const Request&, std::span<const int32_t> missing)`.
  - `grant_lane_group_locked`'s `const std::vector<int64_t>* loading` becomes `const std::span<const int64_t>* loading`
    (callers pass `&slots_span` where `auto slots_span = slots.span();`), and its `std::find` over it is unchanged.
  - `victim_census`'s FFI caller builds a `std::vector` from the tensor (a Python path) and passes it: the implicit
    conversion to span works.

- [ ] **Step 6: Make the reader's callbacks templates** (A6).

  In `ReaderCore`:
  - `Call::experts`/`Call::slots` become `std::span<const int32_t>`/`std::span<const int64_t>`;
  - `(*c.experts)[i]` becomes `c.experts[i]`, and `(*c.slots)[i]` becomes `c.slots[i]`, including in
    `RowReader::image_iovecs`/`poison_slot`;
  - `c.total = experts.size();` is unchanged.

  The new signature:

```cpp
  template <class Abandon, class Progress = NoProgress>
  int read(
      int64_t layer,
      std::span<const int32_t> experts,
      std::span<const int64_t> slots,
      size_t step,
      Abandon&& abandon,
      StageRecord* trace = nullptr,
      std::vector<uint8_t>* packed = nullptr,
      size_t max_reading_rows = SIZE_MAX,
      Progress&& progress = Progress{},
      const PiecePublish* publish = nullptr) {
```

  In the loop, `if (progress) progress();` becomes the call below. `admit(const std::function<...>&)` becomes
  `template <class Abandon> void admit(Abandon& abandon)`.

```cpp
      if constexpr (!std::is_same_v<std::decay_t<Progress>, NoProgress>) progress();
```

  In `RamTier::serve`, pass the lambdas directly (the abandon lambda from Task 10 Step 3), and
  `[this] { retire_leases(); }` for progress. `run_fill` passes `[](size_t) { return false; }`, `NoProgress{}` and
  `&fill_packed_`.

  `read_fault.h`'s `abandon_after` returns the lambda type:

```cpp
inline auto abandon_after(int64_t after) {
  return [after](size_t admitted) { return after > 0 && admitted >= static_cast<size_t>(after); };
}
```

  The FFI read entry points (`read_rows*`) build `std::vector` ids from tensors, which is a Python path, and pass them
  (implicit span).

- [ ] **Step 7: Reserve for the widest reap** (A9). In `ReaderCore::open`, after `io_.init(queue_depth())`, add
  `completions_.reserve(queue_depth() + 1);`. Keep the existing `reserve(extents + 1)` in `size_extents` (a smaller
  bound); this one supersedes it.

- [ ] **Step 8: Run the tests.** Use LOCAL-TEST on the `FixedVec` test, the shim tests, the golden, the stress test
  and both unit suites. Expected: PASS, with the service's `malloc` and `free` at 0 in both variants. If a count
  remains, rerun the child with `ltrace -f -e malloc` scoped to the service thread's tid (`ltrace -p <tid>`), or add a
  `backtrace()` in the shim behind a `HOTPATH_SHIM_BACKTRACE=1` check read once in its constructor. The shim is test
  code, so a raw `getenv` is fine there. Find the site and fix it the same way.

- [ ] **Step 9: SUITE on divix01; CPU-TEST on the shim with `-s`.** Record the table in `baseline.md` under
  "after Task 11". Commit and SYNC. Commit message: `perf(expert-stream): no allocation on the service thread --
  fixed-size requests, spans and template callbacks`, with the trailers.

### Task 12: The copy engine: SPSC job ring, fixed queues, no condvar, wake only a sleeping thread

**Files:**
- Create: `H/spsc_ring.h`
- Modify: `H/copy_engine.h`, `H/ram_tier.h` (construction, `submit` call)
- Test: `T/test_expert_stream_hotpath_shim.py`, `T/test_expert_stream_spsc_ring.py` (new),
  `T/test_exl3_ram_miss_copy_engine.py` (unchanged; must stay green)

**Interfaces:**
- Produces:
  - `template <class T, size_t N> class SpscRing` with `bool push(const T&)`, `bool pop(T*)` and `bool empty() const`,
    where `N` is a power of two;
  - `template <class T, size_t N> class FixedDeque` with `push_back`, `front`, `pop_front`, `empty`, `size`,
    `operator[]` and `erase_if(pred)`;
  - `futex_wait(std::atomic<uint32_t>*, uint32_t expected, int64_t timeout_ns)` and
    `futex_wake(std::atomic<uint32_t>*)`;
  - `CopyEngine<Build, Owner>`, whose owner hooks are `bool copy_completed(const CopyJob&)`,
    `bool copy_acked(const CopyJob&)` and `void copy_failed(const CopyJob&, int)`, plus the counters through
    `owner->template copy_count<K>(n)`;
  - `constexpr size_t kCopyRing = 32;`, which exceeds the kDemandRecords + 1 = 17 jobs that can be outstanding at once.

- [ ] **Step 1: Write the failing tests.**

  `T/test_expert_stream_spsc_ring.py` compiles a two-thread program: a producer pushes 10M sequenced items, the
  consumer checks order and completeness. Compile it with `-fsanitize=thread` when the compiler supports it, else
  `-O2`.

```python
"""SpscRing and FixedDeque (plan 2026-09-29-hotpath-zero-overhead Task 12): order, completeness, full and empty, under
a real producer thread (and ThreadSanitizer where the compiler has it)."""

import subprocess
import textwrap

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PROGRAM = '''
#include "{moe}/expert_stream/host/spsc_ring.h"
#include <cassert>
#include <cstdint>
#include <thread>
using namespace sglang::expert_stream;
int main() {{
  SpscRing<uint64_t, 32> ring;
  uint64_t v = 0;
  assert(ring.empty() && !ring.pop(&v));
  for (uint64_t i = 0; i < 32; ++i) assert(ring.push(i));
  assert(!ring.push(99));  // full at exactly N
  for (uint64_t i = 0; i < 32; ++i) {{ assert(ring.pop(&v) && v == i); }}
  constexpr uint64_t kItems = {items};
  std::thread producer([&] {{ for (uint64_t i = 1; i <= kItems; ++i) while (!ring.push(i)) {{}} }});
  uint64_t next = 1;
  while (next <= kItems) if (ring.pop(&v)) {{ assert(v == next); ++next; }}
  producer.join();
  FixedDeque<int, 8> d; d.push_back(1); d.push_back(2); d.push_back(3);
  d.erase_if([](int x) {{ return x == 2; }});
  assert(d.size() == 2 && d.front() == 1 && d[1] == 3); d.pop_front(); assert(d.front() == 3);
  return 0;
}}
'''


@pytest.mark.parametrize("sanitize", [False, True], ids=["plain", "tsan"])
def test_the_ring_delivers_every_item_in_order_across_threads(tmp_path, sanitize):
    src = tmp_path / "ring.cpp"
    src.write_text(PROGRAM.format(moe=MOE, items=200_000 if sanitize else 10_000_000))
    exe = tmp_path / "ring"
    flags = ["-fsanitize=thread", "-O1", "-g"] if sanitize else ["-O2"]
    built = subprocess.run(["c++", "-std=c++20", *flags, "-o", str(exe), str(src), "-lpthread"],
                           capture_output=True, text=True)
    if sanitize and built.returncode != 0:
        pytest.skip(f"no ThreadSanitizer here: {built.stderr[-300:]}")
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, env={"TSAN_OPTIONS": "halt_on_error=1"})
    assert run.returncode == 0, run.stderr[-3000:]
```

  Add to the shim test:

```python
@pytest.mark.parametrize("variant", ["prod", "instr"])
def test_the_copy_thread_allocates_nothing_and_waits_on_no_condvar(shim, tmp_path, variant):
    """Spec A7, A8, L8, L9: no per-pass deque, no queue node, no condition variable. The copy thread's one remaining
    lock is the tier mutex taken to release a COPYING lease (Task 14 removes it): at most one per job."""
    counts = hotpath_shim.run_child(shim, variant=variant, tmp=tmp_path)
    copy = counts["copy"]
    assert copy["malloc"] == 0 and copy["free"] == 0 and copy["cond"] == 0, counts
    assert copy["mutex"] <= counts["requests"], counts
```

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL (no header; the copy thread allocates and waits on a
  condvar).

- [ ] **Step 3: Write `H/spsc_ring.h`.**

```cpp
// Lock-free single-producer single-consumer ring, a fixed-capacity deque for one thread, and futex wait/wake (plan
// 2026-09-29-hotpath-zero-overhead Tasks 12-14). Nothing here allocates after construction or takes a lock.
#pragma once

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sglang::expert_stream {

// One producer thread, one consumer thread. Each index lives on its owner's cache line beside that owner's cached
// copy of the other index, so a push or pop that finds room touches no line the other side writes.
template <class T, size_t N>
class SpscRing {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "SpscRing capacity is a power of two");

 public:
  bool push(const T& value) {  // producer
    const uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_seen_ == N) {
      tail_seen_ = tail_.load(std::memory_order_acquire);
      if (head - tail_seen_ == N) return false;
    }
    slots_[head & (N - 1)] = value;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }
  bool pop(T* out) {  // consumer
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_seen_) {
      head_seen_ = head_.load(std::memory_order_acquire);
      if (tail == head_seen_) return false;
    }
    *out = slots_[tail & (N - 1)];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }
  bool empty() const {  // either side; exact only on the consumer
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

 private:
  alignas(64) std::atomic<uint64_t> head_{0};
  uint64_t tail_seen_ = 0;  // producer's copy of tail_
  alignas(64) std::atomic<uint64_t> tail_{0};
  uint64_t head_seen_ = 0;  // consumer's copy of head_
  alignas(64) std::array<T, N> slots_{};
};

// A circular FIFO for one thread (the copy thread's in-flight, held and acking jobs).
template <class T, size_t N>
class FixedDeque {
 public:
  bool push_back(const T& value) {
    if (n_ == N) return false;
    data_[(head_ + n_++) % N] = value;
    return true;
  }
  T& front() {
    return data_[head_];
  }
  void pop_front() {
    head_ = (head_ + 1) % N;
    --n_;
  }
  bool empty() const {
    return n_ == 0;
  }
  size_t size() const {
    return n_;
  }
  T& operator[](size_t i) {
    return data_[(head_ + i) % N];
  }
  // Keeps order; removes every element `pred` accepts.
  template <class Pred>
  void erase_if(Pred pred) {
    size_t kept = 0;
    for (size_t i = 0; i < n_; ++i) {
      T& value = data_[(head_ + i) % N];
      if (!pred(value)) data_[(head_ + kept++) % N] = value;
    }
    n_ = kept;
  }

 private:
  std::array<T, N> data_{};
  size_t head_ = 0;
  size_t n_ = 0;
};

inline void futex_wait(std::atomic<uint32_t>* word, uint32_t expected, int64_t timeout_ns) {
  timespec timeout{static_cast<time_t>(timeout_ns / 1000000000), static_cast<long>(timeout_ns % 1000000000)};
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0);
}

inline void futex_wake(std::atomic<uint32_t>* word) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

}  // namespace sglang::expert_stream
```

  The `FixedDeque` test in Step 1 calls `erase_if` with a predicate over `int`, and `push_back` returns bool. Callers
  in the copy engine check it and fail stop (`raise_fatal` through the owner) on false. False is impossible under the
  17-job bound, and failing stop is the E5-safe answer.

- [ ] **Step 4: Rewrite `CopyEngine`.** Template it on `Owner` (the `RamTier`), and delete `Handler`, `Failure` and
  the three `std::function` members. `RamTier` befriends `CopyEngine<Build, RamTier>`, whose calls to
  `copy_completed`, `copy_acked` and `copy_failed` stay private to it.

  The members that change:

```cpp
template <class Build, class Owner>
class CopyEngine {
 public:
  CopyEngine(std::unique_ptr<CopyBackend> backend, int64_t rows, int64_t spin_ns, Owner* owner, std::string prefix,
             std::string thread_name);

  // Service thread only. Takes no lock and never blocks: at most kDemandRecords + 1 jobs are outstanding, and the ring
  // holds kCopyRing. The wake is a syscall only when the copy thread has gone to sleep (idle past spin_ns).
  void submit(const CopyJob& job) {
    if (!jobs_.push(job)) {
      owner_->copy_failed(job, kRingOverflow);
      return;
    }
    submitted_.store(submitted_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with run()'s sleeping_ store and ring re-check
    if (sleeping_.load(std::memory_order_relaxed)) {
      wake_.fetch_add(1, std::memory_order_relaxed);
      futex_wake(&wake_);
    }
  }

  // Every submitted job has completed (or failed) and been handed back: submitted_ is the service's, finished_ the
  // copy thread's, each written by one thread.
  bool idle() const {
    return finished_.load(std::memory_order_acquire) == submitted_.load(std::memory_order_acquire);
  }

  bool wait_idle(int64_t deadline_ns) {  // a paused caller or a test: not the hot path
    while (!idle()) {
      if (now_ns() > deadline_ns) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
  }

  void stop(int64_t drain_ns) {
    if (!thread_.joinable()) return;
    stop_deadline_.store(now_ns() + drain_ns, std::memory_order_relaxed);
    stop_.store(true, std::memory_order_release);
    wake_.fetch_add(1, std::memory_order_relaxed);
    futex_wake(&wake_);
    thread_.join();
  }

 private:
  static constexpr int kRingOverflow = -1000;

  void finish() {  // copy thread
    finished_.store(finished_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  }

  void run() {
    pthread_setname_np(pthread_self(), thread_name_.c_str());
    const std::string error = backend_->init();
    {
      std::lock_guard<std::mutex> guard(start_mutex_);  // the start handshake: setup, not the hot path
      init_error_ = error;
      started_ = true;
    }
    ready_cv_.notify_all();
    if (!error.empty()) return;
    FixedDeque<CopyJob, kCopyRing> in_flight, held, acking;
    uint64_t idle_iters = 0;
    while (true) {
      bool demand_fresh = false;
      bool progressed = false;
      CopyJob job;
      while (jobs_.pop(&job)) {
        progressed = true;
        if (job.prefetch) {
          push_or_fail(held, job);
          continue;
        }
        demand_fresh = true;
        issue_or_fail(job, in_flight);
      }
      bool demand_in_flight = false;
      for (size_t i = 0; i < in_flight.size(); ++i)
        demand_in_flight = demand_in_flight || !in_flight[i].prefetch;
      if (!held.empty() && (demand_fresh || demand_in_flight)) {
        if (!held_counted_) owner_->template copy_count<kPrefetchHeld>();
        held_counted_ = true;
      }
      const bool stopping = stop_.load(std::memory_order_acquire);
      while (!held.empty() && ((!demand_fresh && !demand_in_flight) || stopping)) {
        held_counted_ = false;
        CopyJob next = held.front();
        held.pop_front();
        issue_or_fail(next, in_flight);
      }
      while (!in_flight.empty() && broken_ == 0) {
        const int state = backend_->query(in_flight.front().token);
        if (state == CopyBackend::kPending) break;
        if (state != CopyBackend::kDone) {
          broken_ = state;
          break;
        }
        const CopyJob done = in_flight.front();
        in_flight.pop_front();
        record_latency(done);
        if (owner_->copy_completed(done)) {
          finish();
        } else {
          push_or_fail(acking, done);
        }
        progressed = true;
      }
      const size_t before = acking.size();
      acking.erase_if([&](CopyJob& waiting) {
        if (!owner_->copy_acked(waiting)) return false;
        finish();
        return true;
      });
      progressed = progressed || acking.size() != before;
      if (broken_ != 0) {
        while (!in_flight.empty()) {
          finish_failed(in_flight.front(), broken_);
          in_flight.pop_front();
        }
      }
      if (stopping && held.empty() &&
          ((in_flight.empty() && acking.empty()) || now_ns() > stop_deadline_.load(std::memory_order_relaxed)))
        break;
      if (progressed) {
        idle_iters = 0;
      } else if (!in_flight.empty() || !held.empty() || !acking.empty() || ++idle_iters < spin_iters_) {
        _mm_pause();
      } else {
        const uint32_t seen = wake_.load(std::memory_order_acquire);
        sleeping_.store(true, std::memory_order_seq_cst);
        if (jobs_.empty() && !stop_.load(std::memory_order_acquire)) futex_wait(&wake_, seen, 1000000);  // 1 ms cap
        sleeping_.store(false, std::memory_order_relaxed);
        idle_iters = 0;
      }
    }
    backend_->shutdown(in_flight.empty() && broken_ == 0);
  }

  void push_or_fail(FixedDeque<CopyJob, kCopyRing>& queue, const CopyJob& job) {
    if (!queue.push_back(job)) finish_failed(job, kRingOverflow);
  }

  void issue_or_fail(CopyJob& job, FixedDeque<CopyJob, kCopyRing>& in_flight) {
    if (broken_ != 0) {
      finish_failed(job, broken_);
      return;
    }
    if (const int error_code = issue(job)) {
      broken_ = error_code;
      finish_failed(job, error_code);
      return;
    }
    push_or_fail(in_flight, job);
  }

  void finish_failed(const CopyJob& job, int error_code) {
    owner_->template copy_count<kCopyErrors>();
    owner_->copy_failed(job, error_code);
    finish();
  }

  SpscRing<CopyJob, kCopyRing> jobs_;
  std::atomic<uint64_t> submitted_{0};  // service thread
  std::atomic<uint64_t> finished_{0};   // copy thread
  std::atomic<bool> sleeping_{false};
  std::atomic<uint32_t> wake_{0};
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> stop_deadline_{0};
  uint64_t spin_iters_ = 1;  // idle_budget(spin_ns), at construction
  std::mutex start_mutex_;   // start() handshake only
  std::condition_variable ready_cv_;
  bool started_ = false;
  std::string init_error_;
  Owner* owner_;
  // backend_, tables_, prefix_, thread_name_, thread_, broken_, held_counted_ and the ballast atomics are unchanged.
};
```

  `issue()` and `record_latency()` keep their bodies, with the timing and `kCopyIssueNs`/`kCopyBytes`/
  `kCopyLatency*` inside `if constexpr (Build::kMetrics)` (Task 9). `start()` keeps its handshake on `start_mutex_`
  and `ready_cv_`. `set_table`, `eligible` and `host_backend` are unchanged.

  In `RamTier`:
  - `enable_copy_engine` constructs `std::make_unique<CopyEngine<Build, RamTier>>(std::move(backend), layers_, spin_ns,
    this, prefix, name)`;
  - `copy_engine_`'s type follows;
  - `copy_failed` must tolerate the overflow code: its body prints the error and calls `raise_fatal`, which is already
    the right response.

- [ ] **Step 5: Rewrite `HostCopyBackend` without a lock or an allocation.** It is the CPU backend the tests and the
  shim use. `issue`, `mark` and `query` run on the copy thread; `release` and `fail` run on a test thread.

```cpp
class HostCopyBackend : public CopyBackend {
 public:
  static constexpr int kMarks = 64;      // > kCopyRing: a mark's slot is reused only after it was released
  static constexpr int kEntries = 256;   // per mark: kLeaseLanes lanes x the row layout's names, and a ballast copy

  std::string init() override {
    return "";
  }
  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {  // copy thread
    if (fail_issue_.load(std::memory_order_acquire)) return 999;
    Mark& mark = marks_[open_ % kMarks];
    if (mark.count == kEntries) return 998;
    mark.entries[mark.count++] = CopyEntry{src, dst, bytes, false};
    return 0;
  }
  int mark(int64_t* token) override {  // copy thread: closes the open mark and publishes it
    *token = open_;
    ++open_;
    marks_[open_ % kMarks].count = 0;
    marked_.store(open_, std::memory_order_release);
    return 0;
  }
  int query(int64_t token) override {  // copy thread
    if (fail_query_.load(std::memory_order_acquire)) return 997;
    return token < completed_.load(std::memory_order_acquire) ? kDone : kPending;
  }
  void shutdown(bool) override {}

  // Test thread: land the bytes of the next `marks` closed marks (-1: all), in order, then publish them complete.
  void release(int64_t marks) {
    const int64_t closed = marked_.load(std::memory_order_acquire);
    const int64_t from = completed_.load(std::memory_order_relaxed);
    const int64_t upto = marks < 0 ? closed : std::min(closed, from + marks);
    for (int64_t t = from; t < upto; ++t) {
      const Mark& mark = marks_[t % kMarks];
      for (int i = 0; i < mark.count; ++i)
        std::memcpy(reinterpret_cast<void*>(mark.entries[i].dst), reinterpret_cast<const void*>(mark.entries[i].src),
                    static_cast<size_t>(mark.entries[i].bytes));
    }
    completed_.store(upto, std::memory_order_release);
  }
  void fail(bool issue, bool query) {
    fail_issue_.store(issue, std::memory_order_release);
    fail_query_.store(query, std::memory_order_release);
  }
  int64_t marked() const {
    return marked_.load(std::memory_order_acquire);
  }

 private:
  struct Mark {
    CopyEntry entries[kEntries];
    int count = 0;
  };
  std::array<Mark, kMarks> marks_{};
  int64_t open_ = 0;                  // copy thread
  std::atomic<int64_t> marked_{0};    // copy thread writes
  std::atomic<int64_t> completed_{0}; // test thread writes
  std::atomic<bool> fail_issue_{false};
  std::atomic<bool> fail_query_{false};
};
```

  Before replacing the old `HostCopyBackend`, read it (`copy_engine.h:169-225`) for its exact semantics:
  - what `release(marks)` meant for a partial count;
  - whether `marked()` counted closed marks;
  - how `fail(issue, query)` behaved.

  Keep those semantics; the copy-engine tests pin them. If `CopyEntry` has no fourth field, drop the `false`.

- [ ] **Step 6: Run the tests.** Use LOCAL-TEST on the ring test, the shim tests,
  `T/test_exl3_ram_miss_copy_engine.py`, `T/test_exl3_native_prefetch_service.py`, the golden, the stress test and
  both unit suites. Expected: PASS.

- [ ] **Step 7: SUITE on divix01; CPU-TEST on the shim with `-s`.** Record the table under "after Task 12". Commit
  and SYNC. Commit message: `perf(expert-stream): copy engine on an SPSC job ring -- no deque, no condvar, a futex
  wake only for a sleeping thread`, with the trailers.

---
## Phase E: the lock-free single-owner tier (Tasks 13-16)

**The ownership rule, stated once and used by every task below:**

1. **Who owns the tier.** Everything in `RamTier` but its atomics is owned by exactly one thread at a time:
   - the service thread while it runs;
   - the Python caller that paused it, from the moment the service parks until `resume()`;
   - the caller of `pump()` when there is no thread.
2. **The copy thread owns nothing of the tier.** It reads `slot_gen_` and the lease block, writes CopyDone, and hands
   finished jobs back through an SPSC ring (Task 14).
3. **The fill thread owns nothing of the tier.** It drives the reader while the service is parked, and publishes only
   `fill_landed_`/`fill_state_`. Its epilogue runs at `fill_join()` on the owner (Task 15).
4. **Unpaused Python calls** either are lock-free reads of published words (`mapping`, `counters`, `busy_episode`,
   `graph_leases_outstanding`), or go through a command ring the service drains between requests (`set_hot`,
   `inject_lease`, the snapshots), or refuse (`assign`, `touch`, `has`, `release`, `fill_begin`).
5. **`caller_mutex_`** serializes Python-side callers against each other only. The service, copy and fill threads
   never take it: the Task 15 shim test proves it.

### Task 13: Ownership handoff, the command ring, and snapshots

**Files:**
- Modify: `H/ram_tier.h`, `H/ram_thread.h`, `H/ffi_exports.h`, `OPS` (only docstrings: behavior is unchanged for
  callers)
- Create: `T/test_expert_stream_ownership.py`

**Interfaces:**
- Produces, on `RamTier`:
  - `bool caller_owns() const`;
  - `void set_parked(bool)`;
  - `std::mutex& caller_mutex()`;
  - `void drain_commands()`, for the owner;
  - `void run_as_owner(Command)`;
  - `struct Command`, with the kinds `kSetHot`, `kSnapshot` and `kInjectLease`;
  - `static constexpr int kMaxHotWords = 16`, so `set_hot` refuses layouts of more than 1024 experts.
- Produces, on `RamThread`: `pause` and `resume` take `caller_mutex()`; the loop drains commands before every
  iteration and before parking.

**What each Python-facing method becomes:**

| Method | Unpaused, thread running | Owner (paused or pump mode) |
|---|---|---|
| `mapping(row)` | lock-free: acquire loads of the published slot map (`map_`) | the same |
| `counters`, `busy_episode`, `layer_rows` | lock-free relaxed reads | the same |
| `slot_info`, `slot_to_expert`, `lease_entry`, `lru_order`, `victim_census`, `prefetch_lease` | snapshot command, answered between requests | direct |
| `set_hot` | command (bitmap payload) | direct |
| `inject_lease` (instr) | command | direct |
| `set_prefill_share` | `std::atomic<int64_t>`, relaxed; read only by owner-run admissions | the same |
| `has`, `touch`, `assign`, `release`, `fill_begin` | **refuse**: "needs the service thread paused" | direct |

- [ ] **Step 1: Write the failing tests** in `T/test_expert_stream_ownership.py`:

```python
"""The single-owner tier (plan 2026-09-29-hotpath-zero-overhead Task 13): what an unpaused Python call does while the
service thread runs, and that queued commands are applied in order before the next request."""

import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture
def running(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, layers=2, experts=8)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 8), -1, dtype=torch.int32))
    for expert in range(4):
        host.assign(0, expert)  # pump mode: the caller owns the tier
    host.start_thread(fatal_wait_s=60.0, spin_us=2000)
    yield s, page, host
    host.stop()


def test_unpaused_eager_calls_refuse_or_snapshot(running):
    s, page, host = running
    for call in (lambda: host.assign(0, 5), lambda: host.touch(0, 1), lambda: host.contains(0, 1)):
        with pytest.raises(RuntimeError, match="paused"):
            call()
    info = host.slot_info(0)
    mapping = host.mapping(0)
    assert sorted(e for _, e, _, _ in info if e >= 0) == [0, 1, 2, 3]
    assert all(mapping[e] >= 0 and info[mapping[e]][1] == e for e in range(4))
    assert host.lease_entry(0)["active"] in (0, False)
    host.pause(5.0)
    try:
        slot, _evicted = host.assign(0, 5)  # the owner may assign (it evicts an LRU row: capacity is full)
        assert slot >= 0 and host.mapping(0)[5] == slot
    finally:
        host.resume()


def test_a_set_hot_burst_past_the_ring_is_applied_in_order(running):
    s, page, host = running
    host.inject(delay_s=0.3)  # instr: every advisory and demand read sleeps 300 ms first
    seq = sim_post(page, 1, need=[6], protect=[6])  # a long read on row 1 keeps the service busy
    time.sleep(0.05)
    start = time.perf_counter()
    for i in range(200):  # the command ring holds 64: the producer must wait, never drop
        host.set_hot(0, [i % 4])
    host.set_hot(0, [0, 1, 2])
    assert sim_wait(page, seq, 5.0) == 1
    census = host.victim_census(0, [])
    assert census["evictable"] == 1, census  # only expert 3 is neither hot nor wanted: the LAST set_hot won
    assert time.perf_counter() - start > 0.1, "the burst was applied while the read ran: nothing was queued"
```

  Check the exact names of the tier's Python methods (`contains` or `has`, `touch`, `victim_census`'s return shape,
  `inject`'s keyword for the delay) in `OPS` before running. Adapt the calls to them, and keep the assertions.

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL. `assign` succeeds unpaused today (it takes the mutex), and
  `set_hot` applies at once, so the timing assertion fails.

- [ ] **Step 3: Implement.** In `RamTier`:

```cpp
 public:
  static constexpr int kMaxHotWords = 16;  // set_hot's bitmap: 1024 experts

  struct Command {
    enum Kind : uint8_t { kSetHot, kSnapshot, kInjectLease };
    Kind kind = kSetHot;
    int64_t row = 0;
    int64_t arg = 0;    // kSnapshot: the snapshot's own argument; kInjectLease: the slot
    int64_t delta = 0;  // kInjectLease
    uint64_t hot[kMaxHotWords] = {};
    int64_t (*snapshot)(RamTier*, const Command&) = nullptr;
    int64_t* out = nullptr;
    const void* input = nullptr;  // kSnapshot: an argument the caller keeps alive (victim_census's wanted list)
    int64_t* result = nullptr;    // kSnapshot: the thunk's return value
    std::atomic<uint32_t>* done = nullptr;
  };

  // The tier's owner is the service thread while it runs, else the caller (paused service, or pump mode).
  bool caller_owns() const {
    return !threaded_.load(std::memory_order_acquire) || parked_.load(std::memory_order_acquire);
  }
  void set_parked(bool parked) {
    parked_.store(parked, std::memory_order_release);
  }
  std::mutex& caller_mutex() {
    return caller_mutex_;
  }

  // The owner, between requests: apply every queued Python command, in order.
  void drain_commands() {
    Command command;
    while (commands_.pop(&command))
      apply_command(command);
  }

  // A Python-side call that needs the tier. Runs here when this caller owns the tier; otherwise it is queued for the
  // service (drain_commands), and a snapshot waits for its answer. If the service parks or stops before it drains
  // the queue, the caller then owns the tier and drains it itself: a snapshot is never left unanswered.
  void run_as_owner(Command command) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (caller_owns()) {
      drain_commands();  // anything queued before the handoff comes first
      apply_command(command);
      return;
    }
    std::atomic<uint32_t> done{0};
    if (command.kind == Command::kSnapshot) command.done = &done;
    while (!commands_.push(command)) {
      if (caller_owns()) {
        drain_commands();
        continue;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (command.done == nullptr) return;
    while (done.load(std::memory_order_acquire) == 0) {
      if (caller_owns()) drain_commands();
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }

 private:
  void apply_command(const Command& c) {
    switch (c.kind) {
      case Command::kSetHot:
        set_hot_owned(c.row, c.hot);
        break;
      case Command::kInjectLease:
        inject_lease_owned(c.row, c.arg, c.delta);
        break;
      case Command::kSnapshot: {
        const int64_t value = c.snapshot(this, c);
        if (c.result != nullptr) *c.result = value;
        if (c.done != nullptr) c.done->store(1, std::memory_order_release);
        break;
      }
    }
  }

  void require_owner(const char* what) const {
    if (!caller_owns()) throw std::runtime_error(error_prefix<Layout>() + what + " needs the service thread paused");
  }

  std::atomic<bool> parked_{false};
  std::mutex caller_mutex_;  // Python-side callers only; the service, copy and fill threads never take it
  SpscRing<Command, 64> commands_;
```

  The waiting caller cannot hang on a dead service. If the service parks or exits, `caller_owns()` becomes true and
  the caller drains the queue itself. A service that is alive but hung inside a read is aborted by the watchdog (D6).

  **The public methods.** Each public method keeps its name and signature, and its body becomes one of the three
  forms below. `mapping` becomes lock-free:

```cpp
  // Any thread: the published slot map, which holds a slot for an expert exactly while that slot is READY
  // (publish_map runs only on READY, and -1 on every unmap).
  void mapping(int64_t row, int64_t* out) const {
    for (int64_t expert = 0; expert < experts_; ++expert)
      out[expert] = __atomic_load_n(map_ + row * experts_ + expert, __ATOMIC_ACQUIRE);
  }
```

  A snapshot, taking `slot_info` as the example:

```cpp
  void slot_info(int64_t row, int64_t* out) {
    Command c;
    c.kind = Command::kSnapshot;
    c.row = row;
    c.out = out;
    c.snapshot = [](RamTier* self, const Command& cmd) -> int64_t {
      self->slot_info_owned(cmd.row, cmd.out);
      return 0;
    };
    run_as_owner(c);
  }
```

  `lease_entry`, `slot_to_expert`, `prefetch_lease` and `lru_order` follow the same form; `lru_order` returns its count
  through `c.result`. `victim_census(row, wanted)` passes `&wanted` in `c.input` and writes the census to three
  `int64_t` in a local array through `c.out`. Each `*_owned` function is today's body minus its `lock_guard`, which
  Task 15 deletes everywhere.

  A mutator:

```cpp
  int64_t assign(int64_t row, int64_t expert, std::span<const int32_t> protect, bool fallback, int64_t* evicted) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("assign");
    drain_commands();
    return assign_owned(row, expert, protect, fallback, evicted);
  }
```

  `has`, `touch`, `release` and `fill_begin` take the same form. `set_hot` builds a `kSetHot` command from the expert
  list (a bitmap into `hot`), after refusing `experts_ > kMaxHotWords * 64`. `inject_lease` builds a `kInjectLease`
  command, and still throws `test_only` in production (Task 10). `set_prefill_share` stores into
  `std::atomic<int64_t> prefill_share_`, which the owner's `take_admit_slot_locked` reads relaxed.

  In `RamThread`:

```cpp
  int pause(int64_t timeout_ns) {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    tier_->request_pause(true);
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(true);
    const int64_t deadline = now_ns() + timeout_ns;
    while (!paused_.load()) {
      if (now_ns() > deadline) {
        resume_locked();
        return 0;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    // The service parked (it drained the command ring first): this caller owns the tier until resume.
    tier_->wait_copy_idle_owned(now_ns() + timeout_ns);
    tier_->retire_leases(true);
    if (tier_->graph_leases_outstanding() > 0) {
      resume_locked();
      return 2;
    }
    return 1;
  }

  void resume() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    resume_locked();
  }

 private:
  // The owner hands the tier back: its writes happen-before the service's next request through the release of
  // pause_requested_ (the loop acquires it). parked_ is cleared first, so a caller that then takes caller_mutex()
  // sees the service as the owner and queues instead of touching the tier.
  void resume_locked() {
    tier_->fill_join();
    tier_->skip_advice_posted_so_far();
    tier_->set_parked(false);
    pause_requested_.store(false, std::memory_order_release);
    tier_->request_pause(false);
  }
```

  The loop:

```cpp
    while (!stop_.load(std::memory_order_relaxed)) {
      if ((++iterations & 1023u) == 1u) store_release(page_ + kHeartbeat, ++heartbeat);
      tier_->drain_commands();
      if (pause_requested_.load(std::memory_order_acquire)) {
        tier_->drain_commands();  // nothing queued before the pause is left behind
        tier_->set_parked(true);
        paused_.store(true);
        while (pause_requested_.load(std::memory_order_acquire) && !stop_.load())
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        paused_.store(false);
        continue;
      }
```

  `stop()`:
  - joins the service thread, then clears `threaded_`, so a waiting caller owns and drains the queue;
  - takes `caller_mutex()` around its `set_threaded(false)` and `request_stop` lines, never around the join. A caller
    waiting for a snapshot holds that mutex, and that caller then drains the queue itself because `caller_owns()`
    turns true.

  `wait_copy_idle_owned` is `copy_engine_ == nullptr || copy_engine_->wait_idle(deadline)` for now. Task 14 adds its
  drain. The FFI's `copy_engine_idle` calls a public `wait_copy_idle` that does the same.

  `drain_commands()` costs the service one relaxed load and one compare per loop iteration when the ring is empty: the
  consumer's cached head equals its tail, so the producer's line is re-read only when that cache is stale.

- [ ] **Step 4: Run the tests.** Use LOCAL-TEST on the ownership tests, the stress test (three runs), the golden and
  both unit suites.

  An existing test that calls a now-refusing method on a running, unpaused service was racing the device and is
  wrong: fix it to pause first, and name it in the commit body. Find candidates with
  `grep -rn "start_thread" T test/registered/unit/layers/moe`, and check each file's calls after it.

- [ ] **Step 5: SUITE on divix01**, commit and SYNC. Commit message: `feat(expert-stream): the service owns the tier
  -- pause hands it over, unpaused calls queue or snapshot, eager mutators refuse unpaused`, with the trailers.

### Task 14: Copy completions come back to the owner through an SPSC ring

**Files:**
- Modify: `H/ram_tier.h` (`copy_completed`, `copy_acked`, `release_copied`, `prefetch_completed`, `pump_demand`,
  `serve`'s progress hook, `wait_copy_idle*`), `H/ffi_exports.h` (`copy_engine_idle`)
- Test: `T/test_expert_stream_ownership.py`, `T/test_exl3_ram_miss_copy_engine.py`,
  `T/test_exl3_native_prefetch_service.py`

**Interfaces:**
- Produces:
  - `RamTier::drain_copy_completions()`, for the owner;
  - `RamTier::wait_copy_idle_owned(int64_t deadline_ns) -> bool`, which waits for the engine to be idle and then
    drains;
  - the public `wait_copy_idle(deadline)`, which drains when the caller owns the tier.
- A job is handed back only after its CopyDone (or, for a prefetch, its E6 check) was done on the copy thread.
  PrefetchDone is published by the owner.

- [ ] **Step 1: Write the failing tests** in `T/test_expert_stream_ownership.py`. Reuse `hotpath_script.build_host`
  (pump mode) for the first, and a threaded variant for the second:

```python
from sglang.test import hotpath_script as hp


def _copy_request(s, page, host, sim):
    """A resident expert 0 of row 0, then a request whose lane 0 is COPYING; returns (req, slot)."""
    hp.write_hot_record(page, host, hp.next_seq(page), [])
    first = sim.post(0, [0])
    while host.pump():
        pass
    waited, lanes = hp.accept(sim, first)
    sim.ack(first, waited, lanes=lanes)
    sim.deliver()
    while host.pump():
        pass
    hp.write_hot_record(page, host, hp.next_seq(page), [])
    req = sim.post(0, [0], dst=[0], copy_engine=True)
    while host.pump():
        pass
    return req, host.mapping(0)[0]


def test_a_copying_lease_is_released_at_the_owners_next_poll_not_by_the_copy_thread(tmp_path):
    """D7: the copy thread publishes CopyDone and hands the job back; the owner releases the lease when it next runs."""
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        req, slot = _copy_request(s, page, host, sim)
        host.copy_engine_release(-1)
        deadline = time.time() + 5
        while sim.copy_done(req)[:2] != (hp.lease.COPIED, req.gen):
            assert time.time() < deadline
            time.sleep(0.001)
        assert host.slot_info(0)[slot][2] == 1, "the copy thread released the lease itself"
        host.pump()
        assert host.slot_info(0)[slot][2] == 0, "the owner's poll did not release it"
    finally:
        host.stop()


def test_a_pause_retires_a_copy_that_completed_while_parked(tmp_path):
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        req, slot = _copy_request(s, page, host, sim)
        host.start_thread(fatal_wait_s=60.0, spin_us=2000)
        threading.Timer(0.2, lambda: host.copy_engine_release(-1)).start()
        host.pause(5.0)  # waits for the copy engine to go idle, then the pausing caller drains and retires
        try:
            assert host.slot_info(0)[slot][2] == 0
        finally:
            host.resume()
    finally:
        host.stop()
```

  `hp.lease` is `hotpath_script`'s import of `expert_lease_block`. Import it explicitly in the test if you prefer.

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: the first FAILS, because the copy thread releases the lease
  before any pump. The second passes today (the mutex path); it is here so that Task 14 keeps it green.

- [ ] **Step 3: Implement.** In `RamTier`:

```cpp
  SpscRing<CopyJob, kCopyRing> copy_done_;  // copy thread -> owner: jobs whose copies completed (and were acked)

  // Copy thread. Completion was observed. E6 here, on the words this thread may read; CopyDone published here, so the
  // device's wait ends as soon as the DMA did; the lease itself is released by the owner (drain_copy_completions).
  bool copy_completed(const CopyJob& job) {
    const uint32_t* generations = slot_gen_ + slot_gen_base_[job.row];
    for (int i = 0; i < job.count; ++i) {
      const CopyLane& lane = job.lanes[i];
      if (load_acquire(reinterpret_cast<const uint8_t*>(generations + lane.host_slot)) != lane.slot_generation) {
        copy_count<kCopyGenerationMismatches>();
        raise_fatal(static_cast<uint32_t>(job.gen));
        return true;  // E5: the lease stays held
      }
    }
    if (!job.prefetch) {
      uint8_t* done = lease_ + lease_c_ + job.idx * kLeaseCopyDoneBytes;
      std::memcpy(done + kLeaseCdMask, &job.mask, 4);
      store_release64(done + kLeaseCdGen, tagged_word(kLeaseTagCopied, job.gen));
      if (job.sm) return false;  // released once the copy wait acknowledged its SM reads (copy_acked)
    }
    hand_back(job);
    return true;
  }

  bool copy_acked(const CopyJob& job) {  // copy thread
    const uint64_t word = load_acquire64(lease_ + lease_d_ + kLeaseSmAck + job.idx * kLeaseSmAckBytes);
    if (tag_of(word) != kLeaseTagSmAck || generation_of(word) < job.gen) return false;
    hand_back(job);
    return true;
  }

  void hand_back(const CopyJob& job) {  // copy thread
    // At most kDemandRecords + 1 jobs are outstanding and the ring holds kCopyRing: a full ring is an internal error,
    // and the answer that keeps every lease held (E5) is to fail stop.
    if (!copy_done_.push(job)) {
      copy_count<kCopyErrors>();
      raise_fatal(static_cast<uint32_t>(job.gen));
    }
  }

 public:
  // The owner: release every lease the copy thread handed back, in completion order.
  void drain_copy_completions() {
    CopyJob job;
    while (copy_done_.pop(&job)) {
      if (job.prefetch) {
        prefetch_completed_owned(job);
      } else {
        release_copied_owned(job);
      }
    }
  }

  bool wait_copy_idle_owned(int64_t deadline_ns) {
    const bool idle = copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
    drain_copy_completions();
    return idle;
  }

  bool wait_copy_idle(int64_t deadline_ns) {  // FFI copy_engine_idle
    const bool idle = copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (caller_owns()) drain_copy_completions();
    return idle;
  }
```

  `release_copied_owned` is today's `release_copied` body without its `lock_guard`, with `count<>` for its counters.
  `prefetch_completed_owned` is today's `prefetch_completed` without the E6 check (it now runs on the copy thread)
  and without its `lock_guard`. Its latency metric line goes inside `if constexpr (Build::kMetrics)`; its body
  publishes `PrefetchDone` COPIED, as today.

  Drain at three points:
  - the top of `pump_demand`, before `retire_leases`;
  - `serve`'s progress hook, which becomes `[this] { drain_copy_completions(); retire_leases(); }`;
  - `wait_copy_idle_owned` (`pause()`).

  CopyDone's semantics for the device are unchanged. What moves (D7) is the release of the host-side lease, and
  through it the moment the slot can again be a victim.

- [ ] **Step 4: Run the tests.** Use LOCAL-TEST on the ownership tests, the copy-engine suite, the native-prefetch
  suite, the golden (both variants), the stress test (three runs) and both unit suites.

  The copy-engine tests that assert a released lease right after `copy_engine_idle(...)` pass unchanged, because the
  FFI drains when the caller owns the tier (pump mode). A threaded test that asserts it without pausing now needs an
  `_until`. Adjust it with an `_until(lambda: ...)` wait and a comment naming D7, and list it in the commit body.

- [ ] **Step 5: SUITE on divix01**, commit and SYNC. Commit message: `feat(expert-stream): copy completions return to
  the owner through an SPSC ring (D7); the copy thread never touches the tier`, with the trailers.

### Task 15: Delete the tier mutex; the fill's epilogue moves to the owner; prove no lock

**Files:**
- Modify: `H/ram_tier.h`
- Test: `T/test_expert_stream_hotpath_shim.py`, `T/test_expert_stream_ownership.py`

**Interfaces:**
- `RamTier` has no `mutex_`.
- `lease_changes_` is a plain `uint64_t` owned by the owner.
- `lanes_outstanding_` stays `std::atomic<int64_t>`, written only by the owner (relaxed load plus store), and read
  lock-free by `graph_leases_outstanding()`.
- `rows_demand`/`rows_advisory` become `LineCounters`-style relaxed words.
- `fill_join()` runs `finish_fill_owned()` once, after the join.

- [ ] **Step 1: Write the failing tests.** Add to `T/test_expert_stream_hotpath_shim.py`:

```python
@pytest.mark.parametrize("variant", ["prod", "instr"])
def test_the_service_and_copy_threads_take_no_lock_and_never_wait_on_a_condvar(shim, tmp_path, variant):
    """Spec L1-L10: over the measured requests neither thread takes a mutex, waits on a condition variable or sleeps.
    The request path's only kernel wait is the io_uring completion (D5), which is not a lock."""
    counts = hotpath_shim.run_child(shim, variant=variant, tmp=tmp_path)
    for thread in ("service", "copy"):
        assert counts[thread]["mutex"] == 0 and counts[thread]["cond"] == 0, counts
    assert counts["service"]["sleep"] == 0, counts
```

  Add to `T/test_expert_stream_ownership.py` (a source-level backstop to the runtime proof):

```python
from sglang.test.expert_stream_sources import MOE


def test_the_tier_declares_only_the_callers_mutex():
    text = (MOE / "expert_stream" / "host" / "ram_tier.h").read_text()
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    assert code.count("std::mutex ") == 2, "caller_mutex_ and the InstrBuild trace ring's guard, nothing else"
    assert "std::mutex mutex_" not in code and "lock_guard<std::mutex> guard(mutex_)" not in code
```

- [ ] **Step 2: Run them.** Use LOCAL-TEST. Expected: FAIL (the service takes the tier mutex).

- [ ] **Step 3: Implement.**
  - Delete `std::mutex mutex_;` and every `std::lock_guard<std::mutex> guard(mutex_);` in `ram_tier.h`. At
    `ba01695c35` they are at `:263,268,280,299,342,420,447,577,613,691,900,989,1002,1017,1027,1032,1041,1048,1061,1074,
    1149,1179,1210,1398,1539,1589,1734,1775`; the Task 13/14 `*_owned` functions already dropped theirs.
  - Keep the `_locked` suffixes: they now mean "the owner's". Add one line above the first to say so.
  - `prefill_share_` is the atomic from Task 13. `tick_`, `tiers_`, `outstanding_`, `judge_`, `prefetch_lease_`,
    `deferred_*` and `settled_seq_` are owner-only, and need no change.
  - `lease_changes_` becomes `uint64_t lease_changes_ = 0;`, with `++lease_changes_` in `release_lease_locked` and
    `prefetch_completed_owned`, and a plain read in `deferral_may_retry`.
  - In `lanes_outstanding_`, `fetch_add` and `fetch_sub` become relaxed load-plus-store pairs:

```cpp
    lanes_outstanding_.store(lanes_outstanding_.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
```

  - The per-row `rows_demand` and `rows_advisory` become `std::atomic_ref<int64_t>` relaxed stores of the sum
    (`layer_rows` reads them relaxed from any thread).
  - Move the fill epilogue to the owner. `run_fill` keeps its read and its `fill_landed_`/`fill_state_` stores, records
    `fill_result_ = result`, and no longer touches `tiers_`:

```cpp
  void fill_join() {  // the owner (the caller of fill_end/resume, or the destructor)
    if (fill_thread_.joinable()) fill_thread_.join();
    if (fill_unfinished_) finish_fill_owned();
  }

  // The fill's epilogue, on the owner after the join: a fill thread never writes the tier (ownership rule 3).
  void finish_fill_owned() {
    fill_unfinished_ = false;
    Tier& tier = tiers_[fill_row_];
    for (size_t i = 0; i < fill_slots_.size(); ++i) {
      const int64_t slot = fill_slots_[i];
      tier.filling[slot] = 0;
      if (fill_result_ != 1 && !(i < fill_packed_.size() && fill_packed_[i] != 0)) release_locked(fill_row_, slot);
    }
    if (fill_result_ != 1) {
      count<kReadErrors>();
      count<kVersion>();
    }
  }
```

  `fill_begin` sets `fill_unfinished_ = true` when it starts a thread. The busy episode and `fill_state_` are set by the
  fill thread as today (`begin_busy()`/`end_busy()`, Task 9). A slot stays `filling` until `fill_end()` or `resume()`
  joins, which is conservative: it cannot be a victim or be released in that window.

- [ ] **Step 4: Run the tests.** Use LOCAL-TEST on the shim tests (all four zero assertions), the ownership tests, the
  stress test (five runs), the golden (both variants), the prefill-fills suites
  (`T/test_exl3_ram_miss_prefill_fills.py`, `test/registered/unit/layers/moe/test_exl3_prefill_fills_service.py`) and
  both unit suites. Expected: all PASS.

- [ ] **Step 5: SUITE on divix01; CPU-TEST on the shim with `-s`** (record the final table in `baseline.md`). Then
  CPU-TEST the stress test ten times in a loop, and record 10/10. Commit and SYNC. Commit message:
  `perf(expert-stream)!: delete the tier mutex -- the service thread is the single owner; no lock on the hot path`,
  with the trailers.

### Task 16: ThreadSanitizer and lease-invariant mutants

**Files:**
- Create: `test/manual/dsv41/test_expert_stream_hotpath_tsan.py`, `test/manual/dsv41/tsan.supp`
- Modify: `OPS` (the private TSan loader)
- Create: `analysis/dsv41-drive/hotpath/mutants.md`

**Interfaces:**
- Produces: `OPS._host_module_tsan(layout="exl3") -> Module`, the instrumented TU built with
  `-fsanitize=thread -O1 -g`, for this manual test only. `ExpertStreamHost(..., variant="instr_tsan")` is accepted
  only when `OPS._ALLOW_TSAN` is `True`; the manual test sets it.

- [ ] **Step 1: The TSan loader.** In `OPS`:

```python
_ALLOW_TSAN = False  # the manual TSan test only (test/manual/dsv41/test_expert_stream_hotpath_tsan.py)


@cache_once
def _host_module_tsan(layout: str = "exl3") -> Module:
    return load_jit(
        f"expert_stream_host_{layout}_instr_tsan",
        cpp_files=[LAYOUTS[layout].host_sources["instr"]],
        extra_cflags=["-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fsanitize=thread", "-O1", "-g"],
        extra_ldflags=["-luring", "-lpthread", "-ldl", "-fsanitize=thread"],
        header_only=False,
    )
```

  `_host_module(layout, variant)` returns `_host_module_tsan(layout)` for `variant == "instr_tsan"` when
  `_ALLOW_TSAN` is set, and raises the unknown-variant `ValueError` otherwise.

- [ ] **Step 2: The manual test.** It runs `run_stress` (Task 4) in a child process on the TSan module, with
  `LD_PRELOAD` of the compiler's `libtsan.so`:

```python
"""ThreadSanitizer over the concurrent parties of the single-owner tier (plan 2026-09-29-hotpath-zero-overhead
Task 16). Manual: needs a compiler with libtsan and ~2 minutes. The child preloads libtsan, so the host module's
accesses are checked; Python and torch are not instrumented and are suppressed (tsan.supp)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SUPP = Path(__file__).with_name("tsan.supp")

CHILD = textwrap.dedent("""
    import sys
    from pathlib import Path
    from sglang.kernels.ops.moe import expert_stream_transport as ops
    ops._ALLOW_TSAN = True
    sys.path.insert(0, "test/registered/unit/kernels")
    from test_expert_stream_hotpath_stress import run_stress
    report = run_stress(Path(sys.argv[1]), variant="instr_tsan", seconds=20.0, seed=7)
    assert report["stats"]["errors"] == [] and report["fatal"] == 0, report["stats"]
    print("TSAN-STRESS-OK", report["stats"]["armed"])
""")


def test_the_single_owner_tier_is_race_free_under_tsan(tmp_path):
    libtsan = subprocess.run(["c++", "-print-file-name=libtsan.so"], capture_output=True, text=True).stdout.strip()
    if not libtsan or not os.path.exists(libtsan):
        pytest.skip("no libtsan.so for this compiler")
    env = dict(os.environ, LD_PRELOAD=libtsan,
               TSAN_OPTIONS=f"halt_on_error=1 report_signal_unsafe=0 second_deadlock_stack=1 suppressions={SUPP}")
    proc = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path / "s")], env=env, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0 and "TSAN-STRESS-OK" in proc.stdout, proc.stderr[-8000:]
```

  `tsan.supp`:

```
# The interpreter and torch are not instrumented; races TSan attributes to them are false positives here.
called_from_lib:libpython3
called_from_lib:libtorch_cpu.so
called_from_lib:libc10.so
called_from_lib:libgomp.so
```

- [ ] **Step 3: Run it on divix01.** Use CPU-TEST with `taskset -c 0-63` and no GPU lock. Record the result in
  `mutants.md`. If libtsan cannot be preloaded into the interpreter ("cannot allocate memory in static TLS block", or
  an interpreter crash before the child starts), record the error verbatim and mark TSan as **not buildable here**.
  The lead's instruction was "if it can be built". The mutants below still run.

  A race TSan reports inside the host module is a bug: fix it in the owning task's code, in a new commit, and re-run.

- [ ] **Step 4: Mutants.** Run in `/data/models/slang/nvfp4-work/wt-hotpath-mut`, created at the branch HEAD. Apply
  each mutant, run its target, then `git checkout --`, re-run green, and record both results in `mutants.md` in a
  table (mutant, file:line, target, red?, restored green?).

| # | Mutation | Target that must go red |
|---|---|---|
| M1 | `wait_copy_idle_owned` skips `drain_copy_completions()` | `test_a_pause_retires_a_copy_that_completed_while_parked` |
| M2 | `apply_command` ignores `kSetHot` | `test_a_set_hot_burst_past_the_ring_is_applied_in_order` |
| M3 | `run_as_owner` always applies directly (no `caller_owns()` check) | the TSan test, and `test_unpaused_eager_calls_refuse_or_snapshot` (the refusals) |
| M4 | `release_copied_owned` releases every held lane, not only `copy_engine` ones | `T/test_exl3_ram_miss_copy_engine.py` |
| M5 | `resume_locked` clears `pause_requested_` before `set_parked(false)` | the TSan test (the stress pauser's post-resume `set_hot` races) |
| M6 | `drain_copy_completions` is dropped from `pump_demand`'s top | `test_a_copying_lease_is_released_at_the_owners_next_poll_not_by_the_copy_thread`, the golden |
| M7 | `run_fill` keeps writing `tier.filling[slot] = 0` itself (the old epilogue) | the TSan test, via `T/test_exl3_ram_miss_prefill_fills.py` run under the TSan child |

  A mutant that survives is recorded with its reason. M5 and M7 can survive when TSan is not buildable, because x86
  ordering hides them in the stress test. When TSan is unavailable, that is the stated limit of this proof.

- [ ] **Step 5: Commit** (`test(expert-stream): TSan over the single-owner tier (manual) and lease-invariant mutants`,
  with the trailers). The mutants themselves are never committed.

---
## Phase F: verification on divix01 (Tasks 17-18)

### Task 17: The suites and the GPU tests against the baselines

**Files:**
- Create: `analysis/dsv41-drive/hotpath/results.md`

- [ ] **Step 1: Run SUITE at the branch HEAD** (after SYNC). Record the counts and the command. Explain the delta
  against Task 1's baseline exactly, as a list:
  - the deleted tests (Task 6's commit body);
  - the added tests, by file: golden 2 (4 with Task 10's parametrization), shim, stress 1, build variants, the symbols
    test, `FixedVec` 1, the ring 2, ownership, and the Task 5-7 refusals.

  An unexplained count is a failure. Find it before going on.

- [ ] **Step 2: Run the GPU list** from Task 1 Step 3, minus `test_exl3_piece_stream_cuda.py`, under the GPU lock from
  `wt-hotpath`. The `test/manual/dsv41/conftest.py` from Task 10 makes these tests load the instrumented build. Expected:
  - Task 1's GPU pass count, minus the deleted file's;
  - no new failure;
  - an explained skip count.

- [ ] **Step 3: Run the NVMe and fixed-buffer manual tests** that use the reader, under `rowimg-disk.lock` then the
  GPU lock, in that order:
  - `test/manual/dsv41/test_expert_stream_read_cuts_nvme.py`;
  - `test/manual/dsv41/test_expert_stream_fixed_buffers_big.py`.

  Compare them with the same command at `wt-hotpath-base`.

- [ ] **Step 4: Run the shim and the stress test on divix01** under `taskset -c 0-63`: CPU-TEST on
  `T/test_expert_stream_hotpath_shim.py -s` and on `T/test_expert_stream_hotpath_stress.py` (ten runs). Record the final
  per-request table, which must be zero for the service thread's malloc, free, mutex, cond, clock and sleep, and for
  the copy thread's malloc, free, mutex and cond. Put master's and the branch's tables side by side in `results.md`.

- [ ] **Step 5: Commit** (`analysis(hotpath): suites, GPU tests and hot-path counts at the branch head`, with the
  trailers).

### Task 18: Decode arms A (master) / B (branch) / A2 (master), with perf stat and a shim-counted server

**Files:**
- Create: `analysis/dsv41-drive/hotpath/drive_hotpath_arms.sh` (from `analysis/dsv41-drive/reader-crtp/
  drive_reader_crtp_pair.sh`)
- Create: `analysis/dsv41-drive/hotpath/hotpath_report.py`
- Modify then revert: `benchmarks/dsv41_baseline/generations.json`
- Modify: `analysis/dsv41-drive/hotpath/results.md`

**Interfaces:** Consumes these:
- `run_arm.sh` (with `DSV41_WORKTREE` and `EXPECT_SHA`);
- `generations.register`;
- `analysis/dsv41-drive/mirror3/mirror3_report.py` (`decode`, `identity`, `ram_miss`, `timed_window_utc`,
  `session_clocks`);
- `analysis/dsv41-drive/iopoll-cuts/thread_sampler.py` (`report`);
- the shim (`python/sglang/test/hotpath_shim.c`).

**What this measures, stated up front.**
- **Byte identity.** B's output must equal A's, byte for byte (the production recipe).
- **ms/token.** The service thread is not on the GPU's critical path except through a miss's latency. The expected
  change is within noise to slightly better; the spec estimated a few µs per request.
- **Verdict.** There is no promotion rule, because this is a refactor. The pass condition is identity, `read_errors
  == 0`, and B no slower than `mean(A, A2) + max(1.5, |A2 − A|)` ms/token.
- **The CPU and counter numbers** are reported for the record: the service thread's CPU seconds, perf's cycles,
  instructions and context switches per served request, and the RAM-miss counters from each server's shutdown line.

**Arms, in this order, all on one pinned tier:**

| Arm | Worktree, SHA | Extra `KEY=VAL` overrides |
|---|---|---|
| `A` | `wt-hotpath-base`, `ba01695c35` | `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1 SGLANG_DSV41_RAM_MISS_PACK_WORKERS=8` (master's recipe: the branch's `arm_env` no longer sets them) |
| `B` | `wt-hotpath`, the branch HEAD | none |
| `A2` | as A | as A |
| `C` (not timed) | as B | `LD_PRELOAD=<shim .so>` and `HOTPATH_SHIM_OUT=$OUT/C-shim.json`: a whole-run count of the production server's service thread |

**The tier.** Take the full tier `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:61440,1:40960`, `SGLANG_MOE_PINNED_HOST_MB=102400`
only if both nodes clear their gates:
- node 0: `MemFree + Active(file) + Inactive(file)` ≥ 61440 + 4096 + 15360;
- node 1: ≥ 40960 + 4096 + 10000.

Otherwise use `0:51200,1:40960`/`92160`. The tier is decided once, before A, and passed to every arm as identical
overrides. Before each later arm, the gate is re-checked at the same values, and the pass stops if they are short
rather than run mismatched arms.

- [ ] **Step 1: Arm C's hook in the shim.** Add an exit dump to `hotpath_shim.c`: a constructor reads
  `HOTPATH_SHIM_OUT` (test code, raw `getenv`) and, when it is set, arms at load. A destructor writes
  `{"service": {...}, "copy": {...}}` to that path. Add a self-test in `T/test_expert_stream_hotpath_shim.py`: run the
  probe program with `HOTPATH_SHIM_OUT` set, and read the JSON. Commit it on the branch, with the trailers.

- [ ] **Step 2: Register both trees.** In the branch's `benchmarks/dsv41_baseline/generations.json`, register
  `generations.register('<ba01695c35:python tree>', 'hotpath-base')` and
  `generations.register('<HEAD:python tree>', 'hotpath-zero-overhead')`, using `git rev-parse <sha>:python`. Commit
  (`bench(dsv41): register the hotpath A/B trees for the arms`, with the trailers), push and SYNC. Then check both
  worktrees on divix01 with `check_worktree`: at their commits, clean, with `sglang.__file__` under their own
  `python/`.

- [ ] **Step 3: Write the driver.** Copy `drive_reader_crtp_pair.sh`, and keep every guard unchanged:
  - the NVIDIA driver version check and the port checks;
  - `check_worktree` with the generations gate;
  - the EXL3 launch-gate assertion before every arm, which resolves the requirements with that arm's own `python/`
    tree and stops unless they are EXL3;
  - the exe-name foreign-process gate and `wait_for_quiet_box` (`pgrep -x`, and `-m pytest` as exact tokens);
  - the disk lock taken inside the driver (`exec 8>`, `flock 8`), never under an outer flock; the GPU lock is polled,
    and taken by `run_arm.sh`;
  - PID tracking and cleanup, the memory samples, the SM-clock sampler;
  - stop on `read_errors > 0`.

  Change only the following:
  - **The arm table above.** Both A arms run `EXPECT_SHA=ba01695c35 DSV41_WORKTREE=wt-hotpath-base`, and B and C run
    the branch HEAD. Every arm uses the branch's `run_arm.sh`, `arm_env` and `generations.json`, as the pair driver
    does.
  - **The tier gates** described above.
  - **A build check after "fired up":** B's and C's `server.log` must contain `exl3 RAM miss thread started` with
    `build prod`. A's must not contain `build` at all, since master has no such field. Arms that fail it stop the pass.
  - **The thread sampler**, started once `run_arm.sh` logs `server pid=N` (as in `drive_iopoll_cuts_arms.sh`), writing
    `$OUT/<arm>-threads.jsonl`.
  - **perf stat on the service thread**, started once the server logs "fired up":

```bash
perf_service() {  # <arm> <server pid>: perf stat of the service thread until the arm ends (no root)
    local arm=$1 spid=$2 tid paranoid
    paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
    tid=$(for t in /proc/$spid/task/*; do [ "$(cat $t/comm 2>/dev/null)" = exl3-ram-miss ] && basename $t; done | head -1)
    [ -n "$tid" ] || { say "arm $arm: no exl3-ram-miss thread"; return 1; }
    if [ "$paranoid" -gt 2 ]; then say "arm $arm: perf_event_paranoid=$paranoid, perf stat skipped"; return 0; fi
    taskset -c 20-23 perf stat -x, -e cycles:u,instructions:u,context-switches,cpu-migrations \
        -t "$tid" -o "$OUT/$arm-perf.csv" < /dev/null 8>&- &
    ppid_perf=$!; track "$ppid_perf" "$arm perf stat"
}
```

    Call `perf_service` in `run_one`'s watch loop once "fired up" appears. Stop it with `kill -INT` after
    `wait "$ARM_PID"`; perf writes its CSV on SIGINT.

    With `perf_event_paranoid ≤ 2`, `:u` events on one's own process need no root, and `context-switches` is a
    software event. Record the paranoid value in `results.md`.

  - **The arm loop:** `A B A2 C`. C runs only if A, B and A2 passed. C is excluded from the timing statistics.

- [ ] **Step 4: Write `hotpath_report.py`.** For each arm it reports:
  - `mirror3_report.decode(run)` (session and pooled ms/token), TTFT, and `identity(A_run, X_run)`;
  - `ram_miss(run)`: `rows_read`, `served_requests` and `read_errors` from the shutdown line;
  - `thread_sampler.report(path, *timed_window_utc(run))` for the `exl3-ram-miss` thread's CPU seconds;
  - the perf CSV, divided by `served_requests` (cycles, instructions and context switches per served request);
  - for C, the shim JSON: the service thread's whole-run counts, which must be 0 for malloc, free, mutex and cond. The
    copy thread's counts are reported but not asserted, because libcuda's `cuMemcpyAsync`/`cuEventQuery` may allocate
    inside the driver.

  It also computes `baseline = mean(A, A2)`, B's Δ, the drift `A2 − A`, and the pass condition from "What this
  measures". It writes `$OUT/arms-report.json`.

- [ ] **Step 5: Launch.** Launch on divix01 with `nohup` and no outer flock: the driver takes the disk lock, and
  `run_arm.sh` takes the GPU lock.

  ```bash
  ssh divix01 'cd /data/models/slang/nvfp4-work/wt-hotpath && nohup bash analysis/dsv41-drive/hotpath/drive_hotpath_arms.sh \
    /data/models/slang/nvfp4-work/wt-hotpath-base ba01695c35 /data/models/slang/nvfp4-work/wt-hotpath $(git rev-parse HEAD) \
    /mnt/nvme1/dsv41-hotpath/$(date +%Y%m%d-%H%M%S) 30031 > /mnt/nvme1/dsv41-hotpath/driver.log 2>&1 &'
  ```

  Monitor `driver.log` until it finishes. Expect about 5 minutes per timed arm plus C.

- [ ] **Step 6: Record the results.** Write `results.md`'s arms section:
  - the tier used and why;
  - per arm: ms/token, TTFT, Δ, identity, the RAM-miss counters, the service CPU and the perf per-request numbers;
  - C's shim counts;
  - the drift, the verdict, and the limits of one pass.

  Commit (`analysis(hotpath): decode arms A/B/A2 and the shim-counted production server`, with the trailers).

- [ ] **Step 7: Revert the registration.** Run `git revert --no-commit <Step 2 sha>`, then commit with the message
  `Revert the arm-only generations.json registration (<sha>)` and both trailers. Push, SYNC, and confirm
  `git -C wt-hotpath status --short` is clean.

---

## Self-review against the spec

| Spec item | Where the plan implements it |
|---|---|
| Hot-path boundary (spec section 1) | Tasks 2-4 characterize it. The ownership rule (Phase E preamble) restates it for threads. |
| C1: zero-copy confirmed | Task 2's byte identity; Task 6 refuses buffered reads (`direct=false`) in C++. |
| C2 / D4: delete the packed path | Tasks 5-7. |
| M1: counters | Task 9 (`CoreStats` and `Stats`). |
| M2 | Task 9 (`stats_.store`). |
| M3: watchdog clock | Task 9 Step 6 (episode word). |
| M4 | Task 9 Step 7. |
| M5-M7 | Task 9 Step 8. |
| M8 | Task 9 Step 8. |
| M9: trace | Task 9 Step 5. |
| M10 | Task 10 Step 3 (diagnostics under `kMetrics`). |
| M11-M13 | Task 10 Step 3; production uses `UringReader`. |
| M14 | Task 10 (`sqe_log` under faults). |
| A1-A6, A9 | Task 11. |
| A7, A8 | Task 12. |
| A10 | Tasks 11 and 15 (`fill_packed_`, the epilogue). |
| A11 | Task 11. |
| A12 | Unchanged (error paths). |
| L1-L7 | Tasks 13-15 (owner-only, mutex deleted). |
| L8, L9 | Task 12. |
| L10 | Task 14. |
| L11 | D5: kept. |
| L12 | Kept, as the idle path. |
| L13 | Task 9/10 (instr only). |
| L14 | Deleted with the packed path. |
| L15 | Python-only, unchanged. |
| D1 | Task 9. |
| D2 | Task 8. |
| D3 | Task 8 (`host_variant`). |
| D6 | Task 9 Step 6. |
| D7 | Task 14. |
| D8 | Unchanged: the ring stays protocol-serialized. The fill thread still issues on it only while the service is parked, and ownership rule 3 keeps it off the tier. |

**Proof requirements from the lead:**

| Requirement | Where |
|---|---|
| Concurrent-poster stress | Task 4, run through Tasks 13-16 |
| No mutex on the hot path | the Task 15 shim test (runtime, on production) and the source backstop |
| TSan | Task 16, best effort as instructed |
| Lease mutants | Task 16 M1-M7 |
| No allocation | the Task 11/12 shim tests, on production and instrumented |
| No metrics in ProdBuild | the Task 10 `nm` test, `static_assert(std::is_empty_v<Stats<false, 1>>)`, and the Task 9 shim clock test |
| divix01 | Tasks 17-18 |

**Placeholder scan:** no step says "TBD" or "similar to". Where a step asks the implementer to read an existing
signature first, it is because this plan was written against `ba01695c35` and that signature is outside the plan's
code. The step names the file and line, and says what to keep.

**Type consistency:**
- `kCopyRing` (32) sizes the job ring, the completion ring and the copy thread's `FixedDeque`s. `kWanted` (24) sizes
  every per-request list.
- `Command` is used identically in Tasks 13 and 16.
- `busy_episode`, `CORE_COUNTERS`, `VARIANTS`, `host_variant`, `hp.build_host`/`accept`/`next_seq`/`write_hot_record`,
  and `hotpath_shim.run_child(shim, *, variant, requests, warmup, tmp)` are spelled the same in every task that uses
  them.

## Open decisions for the user

1. **`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` becomes required.** Row images require leases (`SVC:331-333`), and they are
   now the only reader, so a launch without leases refuses at startup (Task 7). Should a follow-up remove the knob, or
   flip its default to on? This plan does neither.
2. **`HostCopyBackend` stays in the production build.** It is selected at run time by `device < 0`, lock-free and
   allocation-free after Task 12. That lets the CPU shim test prove the copy thread in production. Making it
   instrumented-only would remove that proof.
3. **`set_hot` in the non-DIRECT hot mode is now applied before the next request**, not immediately. Production uses
   DIRECT with the GPU hot sidecar, so it never calls `set_hot` after startup. The same holds for a native-prefetch
   `PrefetchDone`, which is published at the owner's next poll: native prefetch is off in the recipe.
4. **CRTP stays for now** (Task 6's rationale). Folding `RowReader` into `ReaderCore` is a follow-up with no hot-path
   effect.
5. **The analysis scripts that drive the packed path** (`analysis/dsv41-drive/bench_pack_workers.py`,
   `pack-pool-bench/`, `pack-workers-odirect*/`) are historical records tied to their commits, and are left untouched.
   They no longer run at the branch head.
