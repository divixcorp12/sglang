# IOPOLL Wait Fix and Device-Sized Read Cuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SGLANG_EXPERT_STREAM_URING_MODE=iopoll` stop costing 10 ms/token by fixing two defects in the expert-stream reader:
- **The wait trap.** An IOPOLL ring is never block-waited through `io_uring_submit_and_wait` / `io_uring_wait_cqe`.
- **Punted reads.** Every read is cut into legs the block device takes whole, so polled reads are no longer punted to io-wq.

It also changes the SQPOLL idle default to 10 s and measures the result in decode arms.

**Architecture:**
- **Wait trap (`UringReader`).** The wait trap is a `UringReader`-only change. On an IOPOLL ring without SQPOLL, waiting always runs the existing reap loop (`io_uring_get_events` with `min_complete=0`, then peek) instead of `io_uring_enter(min_complete>0)`, which polls inside the kernel while holding `uring_lock`.
- **Read cuts (`read_cuts.h`).** A new pure header reads each mirror file's queue limits from sysfs and plans the cuts. It cuts at `min(max_sectors_kb, (max_segments − 1) pages)` and at every iovec join that is not on the device's `virt_boundary`.
- **Legs (`ReaderCore`).** `ReaderCore::plan_legs` composes those cuts with Task 7's registered-buffer fan-out into one leg list. The existing leg machinery then does the rest unchanged: per-leg state, all-or-nothing SQE reservation, leg-only resubmit, fail-once-after-drain, and piece publish on the last leg.
- **Sizing.** Leg storage and the default queue depth are sized at `open()` from the longest read and the smallest cut.

**Tech Stack:** C++20 header-only JIT host module (tvm-ffi), liburing 2.12, Linux io_uring (`IORING_SETUP_IOPOLL`, `READV`, `READV_FIXED`), sysfs block queue attributes, pytest, the dsv41 decode-arm harness (`benchmarks/dsv41_baseline/run_arm.sh`).

**Spec:** `analysis/dsv41-drive/iopoll/diagnosis.md` on `cc/iopoll-diag` (the root cause, the microbenchmark and its numbers), plus the team-lead brief of 2026-09-28 quoted under **Requirements**. Read the diagnosis first: every number this plan predicts comes from it.

## Requirements

The team-lead brief (2026-09-28), which records the user's choice of "the full fix":
1. **Wait trap.** Never block-wait via `io_uring_submit_and_wait` with `min_complete>0` on an IOPOLL ring without SQPOLL. This applies in `UringReader::submit` (`uring_reader.h:~203`) and in `wait_one` (`:~457`). Use the reaping loop instead. Decide between an internal reap and a `from_env` refusal, and justify the choice.
2. **Read cuts.**
   - Cut each READV / READV_FIXED into legs at the device's `max_sectors_kb` and at page-gap (NVMe `virt_boundary`) joins. The limit is read from sysfs per mirror file's block device at open, with a safe default and a clear log line when it cannot be read.
   - Reuse the Task 7 fan-out leg machinery.
   - Decide whether this is default-on or opt-in. Default-on needs the decode pair to support it.
   - Say how the queue depth is resized, and keep the widest-read admission check.
3. **Idle default.** Change the `SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS` default from 1000 to 10000, with a test.
4. **Tests.**
   - Native leg-cutting tests at a lowered test cap, and gap-cut detection.
   - A punt check (0 io-wq workers under IOPOLL, on divix01).
   - Golden and byte identity.
   - Mutants: skipping the gap cut, cutting past the device limit, and others.
5. **Decode check.**
   - Arms at one commit, on the full tier when both nodes have room, else `0:51200,1:40960`: A = master default, B = cuts on in default mode, C = cuts on + `MODE=iopoll` with the fixed wait, and an optional D = `sqpoll_iopoll`.
   - Report ms/token, byte identity, io-wq worker counts and CPU.
   - Promote only on ≥ 1.5 ms/token with identical output.
   - Model the driver on `drive_uring_reg_arms.sh`, including the EXL3 gate and the `generations.json` revert.
6. **SPCC drive.** Record its slow episodes as an open item; they are not in scope.

## Global Constraints

- **Branch and worktrees.**
  - Branch `cc/iopoll-read-cuts` is cut from `origin/cc/iopoll-diag`, which is master `e1559b948b` plus analysis only (its `python/` tree equals master's).
  - Laptop worktree: `/tmp/claude-1000/-home-dimitri-data-divix-sglang-nvfp4/<session>/scratchpad/wt-read-cuts`.
  - divix01 private worktree: `/data/models/slang/nvfp4-work/wt-iopoll-cuts`.
  - Never push to or merge into `master`. Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- **Code reaches divix01 only by commit → `git push origin cc/iopoll-read-cuts` → fetch** into the private worktree. No rsync, scp or `git archive`.
- **Interpreter check.** Every divix01 run uses `PYTHONPATH=$PWD/python` and first prints `sglang.__file__`, which must lie under `wt-iopoll-cuts/python`.
- **Pipes.** A piped pytest reports `${PIPESTATUS[0]}`. A result that came through a pipe without it is unverified.
- **Suite target.** The registered-suite target is `test/registered/unit/kernels` with `-p no:randomly`. Compare its counts to the same command at the base (Task 0), and record the exact command next to every count quoted.
- **CPU work** runs under `taskset -c 0-63` with `OMP_NUM_THREADS=8`.
- **GPU work** (and any suite that touches CUDA) runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`.
- **Disk work** (the manual NVMe test and the arms) takes `rowimg-disk.lock` first.
- **Lock order:** `rowimg-disk.lock`, then `cc-gpu.lock`. `run_arm.sh` takes `cc-gpu.lock` itself (non-blocking), so the arm driver holds the disk lock and polls for the GPU. Cores 64-71 stay free.
- **Process gates.** A foreign-process gate matches executable names (`pgrep -x`), never argv substrings. For pytest, match a python exe whose argv holds exactly the tokens `-m` `pytest`.
- **Mutants** go only in a private worktree (`git worktree add --detach /data/models/slang/nvfp4-work/wt-<name> <commit>`). Revert with `git checkout --`, re-run green, and record both results. Never commit a mutant.
- **Default behavior must not change.** With every `SGLANG_EXPERT_STREAM_URING_*` unset (MODE=default, READ_CUTS=auto → off), the reader prepares the same opcodes and the same SQE set, and writes the same bytes as at `e1559b948b`. `test_expert_stream_reader_golden.py` passes unedited in every task.
- **Commit trailers.** Every commit message ends with these two lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R
  ```
  No amend, no rebase, no force push.
- **No root.** No sudo, no module reload, no sysfs writes. System changes are recommendations for the user only.
- **The new knob** `SGLANG_EXPERT_STREAM_URING_READ_CUTS` (`auto|0|1`, default `auto`) is read in C++ with `getenv`, like the other nine `SGLANG_EXPERT_STREAM_URING_*` knobs. `.claude/skills/env-var-conventions/SKILL.md` Rule 1 (define in `Envs`) cannot apply to a value read only by C++; the Codex knobs follow the same precedent. Migrating all ten is open decision 3.
- **Host facts** (2026-09-28): kernel `6.12.0-211.60.1.el10_2`, liburing 2.12, `nvme.poll_queues=1`, `io_poll=1`, THP `always`. The mirror roots are:

  | root | device | fs | `max_sectors_kb` | `max_segments` | `virt_boundary_mask` |
  |---|---|---|---|---|---|
  | `/mnt/nvme0` | nvme0n1 | xfs | 512 | 128 | 4095 |
  | `/mnt/nvme4` | nvme2n1 | ext4 | 256 | 65 | 4095 |
  | `/mnt/nvme2` | nvme3n1 | xfs | 512 | 128 | 4095 |

  Probe these at runtime; never hard-code them.

Run template, used by every task:

```bash
# SYNC: laptop -> divix01 private worktree
git push origin cc/iopoll-read-cuts
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-iopoll-cuts && git fetch origin \
  && git checkout --detach origin/cc/iopoll-read-cuts && git log -1 --oneline && git status --short'

# CPU-TEST <files...>
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-iopoll-cuts && export PYTHONPATH=$PWD/python \
  OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <files...> -q -rs -p no:randomly 2>&1 | tail -25; \
  echo "EXIT=${PIPESTATUS[0]}"'

# SUITE (one tier test touches CUDA, so it takes the GPU lock)
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-iopoll-cuts && export PYTHONPATH=$PWD/python \
  OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -5; \
  echo "EXIT=${PIPESTATUS[0]}"'
```

## Review Focus

1. **A leg that starts mid-page.** `max_sectors_kb` alone is not enough: a 512 KiB leg starting 512 B into a page spans 129 pages, one more than a Samsung's `max_segments` of 128, and would still be split, so still punted. The cut is `min(max_sectors_kb · 1024, (max_segments − 1) · 4096)` rounded down to whole pages: 520192 B on the Samsungs, 262144 B on the SPCC. Pinned in Task 3 Step 1 (`cut_bytes_for` cases) and in Task 6's punt check.
2. **A drive whose limits cannot be read.** tmpfs, a missing attribute or an unresolvable `/sys/dev/block` link must fall back to 128 KiB on a 4 KiB boundary. It must print exactly one `read cuts:` line naming the reason, and never crash or silently turn cuts off. Pinned in Task 3 Step 1 (the `/dev/shm` fallback and the missing-attribute case).
3. **An explicit `QUEUE_DEPTH` below the widest read's leg bound.** It must refuse `open()` with a message naming the knob. It must never hang on an SQ too small for an all-or-nothing reservation. The default depth must grow with the leg bound. Pinned in Task 4 Step 1 (`test_credit_scales_with_the_leg_bound`, `test_explicit_depth_below_the_leg_bound_is_refused`).
4. **A short or failing cut leg.** A short leg must resubmit only itself, from its own `done`. A failing leg must fail the call once, after every leg is reaped. A held leg must keep its sub-read unretired and its pieces unpublished. Cuts multiply legs per read by about 5-10×, so every Task 7 fan-out guarantee is now on the hot path of a default-mode read. Pinned in Task 5 Step 1 (four leg tests at `leg_cut_cap=8192`).
5. **Different limits per drive in one read set.** A row's parts on the SPCC must be cut at 262144 while the Samsung parts are cut at 520192, so the limit is looked up per file (`limits_[read.file]`), never as one global minimum. The global minimum sizes only storage and credit. Pinned in Task 6 Step 2 (per-root `device_limits`) and in Task 7's `check_modes`, which requires one `read cuts:` line per drive with those values.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/read_cuts.h` | Create (Task 3) | `DeviceLimits`, `cut_bytes_for`, `limits_from_queue_dir`, `queue_dir_for`, `device_limits(fd)`, `CutLeg`, `cut_legs`, `leg_bound` |
| `.../host/uring_options.h` | Modify (Tasks 1, 2) | `READ_CUTS` knob (`UringReadCuts`), `read_cuts_on()`, `SQ_THREAD_IDLE_MS` default 10000, `polls_in_wait()`, `blocking_wait()`, `effective_wait_name()` |
| `.../host/uring_reader.h` | Modify (Task 2) | The reap loop for IOPOLL without SQPOLL in `submit` and `wait_one`; diagnostics gain `read_cuts=` and `effective_wait=` |
| `.../host/reader_core.h` | Modify (Tasks 4, 5) | Per-file `limits_`, runtime `leg_stride_`/`iov_stride_`, `kMaxLegs = 255`, depth scaling and admission refusal (Task 4); cut legs in `plan_legs`, counters, `set_leg_cut_cap`, one `read cuts:` line per drive (Task 5) |
| `.../host/any_reader.h` | Modify (Task 5) | Forward `set_leg_cut_cap`, `cut_reads`, `gap_cuts`, `min_cut_bytes`, `leg_stride` |
| `.../host/read_fault.h` | Modify (Task 5) | `kFaultWords` 31 → 32; word 31 = `leg_cut_cap` |
| `.../host/ffi_exports.h` | Modify (Task 5) | Apply word 31 before `open()` at the four entry points; `read_rows_sqes` info 7 → 11 words; `read_rows_faulted` results 10 → 12 |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | Modify (Task 5) | `_fault_tensor(leg_cut_cap=0)`; info keys `cut_reads`, `gap_cuts`, `min_cut_bytes`, `leg_stride`; stats keys `cut_reads`, `gap_cuts` |
| `test/registered/unit/kernels/test_expert_stream_uring_options.py` | Modify (Tasks 1, 2) | Idle default, `READ_CUTS` parsing, the wait trap in the fake-liburing contract |
| `test/registered/unit/kernels/test_expert_stream_read_cuts.py` | Create (Tasks 3, 4, 5) | Native unit tests of `read_cuts.h`; FFI tests of cut legs, depth, identity |
| `test/manual/dsv41/test_expert_stream_read_cuts_nvme.py` | Create (Task 6) | divix01 only: per-root sysfs limits, the io-wq punt check under IOPOLL, the too-small-cut refusal |
| `analysis/dsv41-drive/uring-config/HANDOFF.md` | Modify (Task 1) | The `SQ_THREAD_IDLE_MS` default row, 1000 → 10000; add a `READ_CUTS` row |
| `analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh` | Create (Task 7) | Decode arms A, B, C, (D), A′ |
| `analysis/dsv41-drive/iopoll-cuts/thread_sampler.py` | Create (Task 7) | Samples the server process tree's threads (comm, CPU) every 2 s; reports io-wq and service-thread CPU over the timed window |
| `analysis/dsv41-drive/iopoll-cuts/results.md` | Create (Task 7) | The arms' results and verdict |

---

### Task 0: Workspace and baselines

**Files:** none changed.

- [ ] **Step 1: Create the branch and worktrees**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4 && git fetch origin
WT=/tmp/claude-1000/-home-dimitri-data-divix-sglang-nvfp4/<session>/scratchpad/wt-read-cuts
git worktree add -b cc/iopoll-read-cuts $WT origin/cc/iopoll-diag
git -C $WT push -u origin cc/iopoll-read-cuts
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-iopoll-cuts origin/cc/iopoll-read-cuts \
  && git -C /data/models/slang/nvfp4-work/wt-iopoll-cuts log -1 --oneline'
```

Confirm that `git rev-parse origin/cc/iopoll-read-cuts:python` equals `git rev-parse e1559b948b:python`.

- [ ] **Step 2: Baseline the suites.** Run SUITE and record `N passed / M skipped / K failed` with the command. Then run CPU-TEST on:
  - `test_expert_stream_reader_golden.py`
  - `test_expert_stream_fixed_buffers.py`
  - `test_expert_stream_uring_options.py`
  - `test_expert_stream_uring_native.py`
  - `test_expert_stream_uring_integration.py`

  Record each count. Every later task compares against these numbers.

---

### Task 1: The `READ_CUTS` knob and the 10 s SQPOLL idle default

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_options.h`
- Modify: `test/registered/unit/kernels/test_expert_stream_uring_options.py`
- Modify: `analysis/dsv41-drive/uring-config/HANDOFF.md` (the options table)

**Interfaces:**
- Produces:
  - `enum class UringReadCuts { Auto, Off, On };`
  - `UringOptions::read_cuts` (default `Auto`);
  - `bool UringOptions::read_cuts_on() const`, which is `On`, or `Auto && iopoll()`;
  - `const char* UringOptions::read_cuts_name() const`, which returns `"auto"`, `"off"` or `"on"`;
  - `UringOptions::sq_thread_idle_ms` default `10000`.
  - Env: `SGLANG_EXPERT_STREAM_URING_READ_CUTS` ∈ {`auto`, `0`, `1`}; anything else is `invalid_argument` naming the key.

- [ ] **Step 1: Write the failing tests.** In `test_uring_environment_validation`, change the probe program so it prints the defaults it checks, then add the new cases:

```python
    source.write_text(r"""
#include "moe/expert_stream/host/uring_options.h"
#include <iostream>
int main() {
 try {
   auto o = sglang::expert_stream::UringOptions::from_env();
   std::cout << o.queue_depth << " " << o.sq_thread_idle_ms << " " << o.read_cuts_name() << " "
             << o.read_cuts_on();
 }
 catch (const std::exception& e) { std::cerr << e.what(); return 2; }
}
""")
    ...
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    # Defaults: depth 0 (16 * parts), SQPOLL idle 10 s (user, 2026-09-28), cuts auto and off outside IOPOLL.
    assert subprocess.check_output([binary], env=env, text=True) == "0 10000 auto 0"

    def probe(**values):
        return subprocess.check_output(
            [binary], env=env | {PREFIX + k: v for k, v in values.items()}, text=True
        ).split()

    assert probe(MODE="iopoll")[2:] == ["auto", "1"]           # auto follows IOPOLL
    assert probe(MODE="sqpoll_iopoll")[2:] == ["auto", "1"]
    assert probe(MODE="sqpoll")[2:] == ["auto", "0"]
    assert probe(READ_CUTS="1")[2:] == ["on", "1"]
    assert probe(MODE="iopoll", READ_CUTS="0")[2:] == ["off", "0"]
    assert probe(SQ_THREAD_IDLE_MS="1000")[1] == "1000"          # an explicit value still wins
```

Then add `("READ_CUTS", "auto")` to the accepted list and `("READ_CUTS", "on")`, `("READ_CUTS", "2")` and `("READ_CUTS", "")` to the rejected list.

- [ ] **Step 2: Run it and confirm it fails**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_uring_options.py::test_uring_environment_validation`
Expected: FAIL. The default output is `0`, not `0 10000 auto 0`, and `read_cuts_name` is not a member.

- [ ] **Step 3: Implement.** In `uring_options.h`:

```cpp
enum class UringReadCuts { Auto, Off, On };
```

In `struct UringOptions`, change `unsigned sq_thread_idle_ms = 1000;` to `unsigned sq_thread_idle_ms = 10000;`. After `bool diagnostics = false;` add:

```cpp
  // Cut every read into legs the block device takes whole (read_cuts.h; plan 2026-09-28-iopoll-read-cuts). auto: on
  // exactly when IOPOLL is, where an uncut read is punted to io-wq (analysis/dsv41-drive/iopoll/diagnosis.md).
  UringReadCuts read_cuts = UringReadCuts::Auto;
  bool read_cuts_on() const {
    return read_cuts == UringReadCuts::On || (read_cuts == UringReadCuts::Auto && iopoll());
  }
  const char* read_cuts_name() const {
    return read_cuts == UringReadCuts::Auto ? "auto" : read_cuts == UringReadCuts::On ? "on" : "off";
  }
```

In `from_env()`, change the idle line and add the knob:

```cpp
    o.sq_thread_idle_ms = number<unsigned>("SQ_THREAD_IDLE_MS", 10000, 0, std::numeric_limits<unsigned>::max());
    ...
    const auto cuts = value("READ_CUTS", "auto");
    if (cuts == "auto")
      o.read_cuts = UringReadCuts::Auto;
    else if (cuts == "0")
      o.read_cuts = UringReadCuts::Off;
    else if (cuts == "1")
      o.read_cuts = UringReadCuts::On;
    else
      invalid("READ_CUTS", "expected auto, 0, or 1");
```

In `HANDOFF.md`, change the `SQ_THREAD_IDLE_MS` row's default to `10000` and add a sentence on the change: "(1000 before 2026-09-28; user decision)". After `DIAGNOSTICS`, add the row `| READ_CUTS | auto | auto (on with IOPOLL), 0, 1: cut reads into device-sized legs (plan 2026-09-28-iopoll-read-cuts) |`.

- [ ] **Step 4: Run it and confirm it passes**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_uring_options.py`
Expected: every test passes. Nothing reads `read_cuts` yet.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_options.h \
  test/registered/unit/kernels/test_expert_stream_uring_options.py analysis/dsv41-drive/uring-config/HANDOFF.md
git commit -m "feat(expert-stream): READ_CUTS knob (auto|0|1) and a 10 s SQPOLL idle default

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

---

### Task 2: The wait trap. An IOPOLL ring without SQPOLL always waits by reaping

**Recorded decision 1: an internal reap, not a refusal.** On an IOPOLL ring without SQPOLL there is no blocking wait to preserve. `io_uring_enter(GETEVENTS, min_complete=n)` spins in the kernel's poll loop, so the waiting thread is 100% busy either way: 1.99 CPU-s per 2 s in both `iopoll+block` and `iopoll+spin` (diagnosis table 3). The reap loop costs the same CPU and releases `uring_lock` between passes, so a read punted to io-wq is no longer serialized behind completions (row p50 3233 → 1609 µs).

A `from_env` refusal would break every existing `MODE=iopoll` config, including the Codex S4 recipe, for no gain. It would also leave the trap reachable through any future default change. So `WAIT_MODE=block` on such a ring becomes the reap loop, and the diagnostics line says so (`effective_wait=reap`). `sqpoll_iopoll` is unaffected: its waiter sleeps in the CQ wait while the SQ thread polls.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_options.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_reader.h:201-220` (`submit`), `:453-460` (`wait_one`), `:462-490` (`diagnostics`)
- Modify: `test/registered/unit/kernels/test_expert_stream_uring_options.py` (`_FAKE_LIBURING`, `_FAKE_CASES`)

**Interfaces:**
- Consumes: `UringOptions::read_cuts_on()`, `read_cuts_name()` (Task 1).
- Produces:
  - `bool UringOptions::polls_in_wait() const`, which is `iopoll() && !sqpoll()`;
  - `bool UringOptions::blocking_wait() const`, which is `wait_mode == Block && !polls_in_wait()`;
  - `const char* UringOptions::effective_wait_name() const`, which returns `"block"`, `"reap"` or `"spin"`;
  - `bool UringReader::read_cuts() const`, which returns `options_.read_cuts_on()`;
  - the diagnostics line gains ` read_cuts=<auto|on|off>:<0|1> effective_wait=<block|reap|spin>` at its end.

- [ ] **Step 1: Write the failing tests.** In `_FAKE_LIBURING`, count `io_uring_wait_cqe` calls and record the SQPOLL idle value passed at setup:

```cpp
inline int wait_cqe_calls=0;
inline unsigned last_sq_thread_idle=0;
inline int io_uring_queue_init_params(unsigned n, io_uring* r, io_uring_params* p) {
 ++setups; if (setup_error) return setup_error; r->flags=p->flags; r->sq.clear(); r->cq.clear();
 last_sq_thread_idle=p->sq_thread_idle;
 p->sq_entries=n; p->cq_entries=2*n; p->features=17; last_ring=r; return 0;
}
inline int io_uring_wait_cqe(io_uring* r,io_uring_cqe** c) { ++wait_cqe_calls; return io_uring_peek_cqe(r,c); }
```

Note: `io_uring_peek_cqe` must be declared before `io_uring_wait_cqe`. Move the `wait_cqe` definition below `peek_cqe`, as it already is.

In `_FAKE_CASES`, just before `env("MODE","iopoll");` / `{ UringReader r;assert(r.init(8));rejected(...` (the O_DIRECT refusal), add:

```cpp
 // The wait trap (plan 2026-09-28-iopoll-read-cuts Task 2): IOPOLL without SQPOLL never waits in
 // io_uring_submit_and_wait / io_uring_wait_cqe, whatever WAIT_MODE says -- it reaps with GETEVENTS min_complete=0.
 env("WAIT_MODE","block");
 for(const char* mode : {"iopoll", "sqpoll_iopoll", "default"}) {
 env("MODE",mode);UringReader r;assert(r.init(8));r.configure_resources(files,buffers,true);
 const bool trap=std::string(mode)=="iopoll";
 delay_completions=trap;cq_observations=0;
 const int waits=wait_calls, events=get_events_calls, cqe_waits=wait_cqe_calls;
 assert(r.prep_read(99,arena,16,0,40));assert(r.submit(1)>=0);
 std::vector<ReadCompletion> out;assert(r.reap(out)==1 && out[0].data==40);
 if(trap)assert(wait_calls==waits && get_events_calls>events);
 else assert(wait_calls==waits+1);
 // drain -> wait_one: a read in flight with no completion yet.
 delay_completions=trap;const int events2=get_events_calls;
 assert(r.prep_read(99,arena,16,0,41));assert(r.submit(0)>=0);r.drain(1);
 if(trap)assert(wait_cqe_calls==cqe_waits && get_events_calls>events2);
 else if(std::string(mode)=="default")assert(wait_cqe_calls==cqe_waits+1);  // SQPOLL drains by peeking instead
 assert(delayed_cq.empty());delay_completions=false;
 }
 env("MODE","sqpoll");env("WAIT_MODE","spin");
 { UringReader r;assert(r.init(8));assert(last_sq_thread_idle==10000); }   // the 10 s default reaches the ring
```

Two things to check before running:
- The `sqpoll_iopoll` case: the fake publishes completions immediately (`delay_completions` false), so `submit_and_wait` and `wait_cqe` see them. This is the path whose waiter is allowed to block.
- The `default` case: `wait_calls` rises by one because `submit(1)` calls `io_uring_submit_and_wait`.

- [ ] **Step 2: Run it and confirm it fails**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_uring_options.py::test_uring_reader_driver_contract`
Expected: FAIL (an `assert` abort in the reader binary). The `iopoll` case calls `io_uring_submit_and_wait` (`wait_calls==waits` fails).

- [ ] **Step 3: Implement.** In `uring_options.h`, inside `struct UringOptions`:

```cpp
  // IOPOLL without SQPOLL: io_uring_enter(GETEVENTS, min_complete>0) polls the device inside the kernel holding the
  // ring's uring_lock, and a read punted to io-wq cannot queue itself on the poll list until the waiter lets go: the
  // punted reads then issue one after another behind completions (+1.5 ms per row; diagnosis.md table 3). Such a
  // ring therefore always waits with min_complete=0 passes.
  bool polls_in_wait() const {
    return iopoll() && !sqpoll();
  }
  bool blocking_wait() const {
    return wait_mode == UringWaitMode::Block && !polls_in_wait();
  }
  const char* effective_wait_name() const {
    return blocking_wait() ? "block" : polls_in_wait() ? "reap" : "spin";
  }
```

In `uring_reader.h`, `submit`: replace

```cpp
    if (options_.wait_mode == UringWaitMode::Block)
      return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_);
```

with

```cpp
    if (options_.blocking_wait())
      return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_);
```

The loop below is unchanged. It already calls `io_uring_get_events` when `iopoll() && !sqpoll()`.

Replace `wait_one` with:

```cpp
  void wait_one() {
    io_uring_cqe* cqe = nullptr;
    if (options_.polls_in_wait()) {
      // Never io_uring_wait_cqe here: see UringOptions::polls_in_wait.
      for (;;) {
        const int rc = io_uring_peek_cqe(&ring_, &cqe);
        if (rc == 0) break;
        if (rc != -EAGAIN && rc != -EINTR) std::terminate();
        const int got = io_uring_get_events(&ring_);
        if (got < 0 && !soft_error(got)) std::terminate();
        spin_hint();
      }
      retire(cqe);
      return;
    }
    int rc;
    do {
      rc = io_uring_wait_cqe(&ring_, &cqe);
    } while (soft_error(rc));
    if (rc < 0) std::terminate();
    retire(cqe);
  }
```

Add a public accessor after `fixed_reads()`:

```cpp
  // Cut reads into device-sized legs (READ_CUTS; ReaderCore plans them): the resolved option, for diagnostics.
  bool read_cuts() const {
    return options_.read_cuts_on();
  }
```

In `diagnostics()`, append ` read_cuts=%s:%d effective_wait=%s` to the format string after `register_ms=%.1f`, and add the matching arguments `options_.read_cuts_name(), options_.read_cuts_on() ? 1 : 0, options_.effective_wait_name()`.

- [ ] **Step 4: Run it and confirm it passes**

Run: CPU-TEST `test_expert_stream_uring_options.py test_expert_stream_uring_native.py test_expert_stream_uring_integration.py`
Expected: every test passes. The native matrix's `iopoll`/`block` cases now go through the reap loop and must still read correct bytes. Where the laptop kernel lacks IOPOLL support, those cases skip as they did in Task 0.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_options.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_reader.h \
  test/registered/unit/kernels/test_expert_stream_uring_options.py
git commit -m "fix(expert-stream): an IOPOLL ring without SQPOLL waits by reaping, never in submit_and_wait

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

---

### Task 3: `read_cuts.h`, the pure cut planner and the sysfs limits

**Recorded decision 2: the cut rule.**
- **Size.** A leg is at most `cut_bytes = min(max_sectors_kb · 1024, (max_segments − 1) · 4096)`, rounded down to whole pages and at least one page. The `max_segments − 1` term covers a leg that starts mid-page (Review Focus 1).
- **Gaps.** A new leg starts at every join where the earlier iovec does not end, or the next does not start, on `virt_boundary_mask + 1`. `virt_boundary_mask = 0` means no gap rule. A missing attribute counts as 4095, the conservative choice.
- **Fallback.** If a file's queue cannot be found, cut at 128 KiB on a 4 KiB boundary, and say so once.
- **Alignment.** Cuts are whole pages from a leg's start, and gap cuts fall at segment joins, which are 512 B-aligned (`kImageAlign`). So every leg's file offset and length stays a multiple of 512, and each leg is a legal O_DIRECT read.

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/read_cuts.h`
- Create: `test/registered/unit/kernels/test_expert_stream_read_cuts.py` (the native unit part)

**Interfaces:**
- Produces (namespace `sglang::expert_stream`):
  - `constexpr int64_t kCutPage = 4096; constexpr int64_t kFallbackCutBytes = 131072; constexpr uint64_t kFallbackVirtMask = 4095;`
  - `struct DeviceLimits { int64_t cut_bytes; uint64_t virt_mask; std::string source; };`
  - `int64_t cut_bytes_for(int64_t max_sectors_kb, int64_t max_segments);`
  - `bool limits_from_queue_dir(const std::string& dir, DeviceLimits* out, std::string* why);`
  - `std::string queue_dir_for(dev_t dev, std::string* why);`
  - `DeviceLimits device_limits(int fd);` (fallback with `source = "fallback: <why>"`)
  - `struct CutLeg { unsigned first; unsigned count; int64_t bytes; bool gap; };`
  - `unsigned cut_legs(const iovec* in, unsigned count, const DeviceLimits& lim, iovec* out, unsigned max_out, CutLeg* legs, unsigned max_legs);`, which returns `max_legs + 1` on overflow
  - `size_t leg_bound(int64_t longest, size_t iovecs, int64_t cut_bytes);`, which is `longest / cut_bytes + 2 * iovecs`. That counts the size cuts, one extra leg per gap-separated run, and one per registered-buffer change at a join.

- [ ] **Step 1: Write the failing test** `test/registered/unit/kernels/test_expert_stream_read_cuts.py`:

```python
"""Device-sized read cuts in the expert-stream reader (plan 2026-09-28-iopoll-read-cuts). The native part compiles
read_cuts.h alone; the FFI part (Tasks 4-5) reads through the reader with a test-only cut cap (fault word
leg_cut_cap). Why the cuts exist: analysis/dsv41-drive/iopoll/diagnosis.md."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
INCLUDE = ROOT / "python/sglang/kernels/jit/csrc"

_UNIT = r"""
#include "moe/expert_stream/host/read_cuts.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
using namespace sglang::expert_stream;

static DeviceLimits lim(int64_t cut, uint64_t mask) { return DeviceLimits{cut, mask, "test"}; }

struct Plan { std::vector<CutLeg> legs; std::vector<iovec> iov; unsigned n; };
static Plan plan(std::vector<iovec> in, const DeviceLimits& l, unsigned max_legs = 64) {
  Plan p; p.legs.resize(max_legs); p.iov.resize(in.size() + max_legs);
  p.n = cut_legs(in.data(), in.size(), l, p.iov.data(), p.iov.size(), p.legs.data(), max_legs);
  return p;
}
// The legs tile the input: same bytes, same addresses, in order, every leg within the cut.
static void tiles(const std::vector<iovec>& in, const Plan& p, int64_t cut) {
  size_t i = 0, off = 0, k = 0;
  for (unsigned g = 0; g < p.n; ++g) {
    assert(p.legs[g].first == k && p.legs[g].bytes > 0 && p.legs[g].bytes <= cut);
    int64_t sum = 0;
    for (unsigned j = 0; j < p.legs[g].count; ++j, ++k) {
      assert(p.iov[k].iov_base == static_cast<uint8_t*>(in[i].iov_base) + off);
      sum += p.iov[k].iov_len; off += p.iov[k].iov_len;
      if (off == in[i].iov_len) { ++i; off = 0; }
    }
    assert(sum == p.legs[g].bytes);
  }
  assert(i == in.size());
}

int main(int argc, char** argv) {
  // cut_bytes_for: max_sectors_kb, and max_segments less one page (a leg that starts mid-page spans one more).
  assert(cut_bytes_for(512, 128) == 520192);   // Samsung 990 EVO Plus on divix01
  assert(cut_bytes_for(256, 65) == 262144);    // SPCC on divix01
  assert(cut_bytes_for(1024, 33) == 131072);
  assert(cut_bytes_for(1, 128) == 4096);       // never below one page
  assert(cut_bytes_for(10, 0) == 8192);        // max_segments unknown: max_sectors_kb only, whole pages

  uint8_t* a = static_cast<uint8_t*>(std::aligned_alloc(4096, 1 << 20));
  uint8_t* b = static_cast<uint8_t*>(std::aligned_alloc(4096, 1 << 20));
  const int64_t C = 8192;
  { // size only: 3C + 512 in one page-aligned iovec -> C, C, C, 512
    std::vector<iovec> in{{a, 3 * C + 512}};
    Plan p = plan(in, lim(C, 4095)); assert(p.n == 4); tiles(in, p, C);
    for (unsigned g = 0; g < p.n; ++g) assert(!p.legs[g].gap);
  }
  { // gap: the first iovec ends off the page (9216 B) -> a new leg at the join even under a huge cut
    std::vector<iovec> in{{a, 9216}, {b, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 2 && !p.legs[0].gap && p.legs[1].gap); tiles(in, p, 1 << 30);
  }
  { // aligned join: one leg
    std::vector<iovec> in{{a, 8192}, {b, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 1 && p.legs[0].count == 2);
  }
  { // the next iovec starts off the page -> gap
    std::vector<iovec> in{{a, 8192}, {b + 512, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 2 && p.legs[1].gap);
  }
  { // no virt boundary (mask 0): no gap rule
    std::vector<iovec> in{{a, 9216}, {b + 512, 4096}};
    Plan p = plan(in, lim(1 << 30, 0)); assert(p.n == 1);
  }
  { // gap and size together: 9216 -> C, 1024 | gap | 3C -> C, C, C
    std::vector<iovec> in{{a, 9216}, {b, 3 * C}};
    Plan p = plan(in, lim(C, 4095)); assert(p.n == 5); tiles(in, p, C);
    assert(!p.legs[0].gap && !p.legs[1].gap && p.legs[2].gap && !p.legs[3].gap && !p.legs[4].gap);
  }
  { // overflow: more legs than the caller sized
    std::vector<iovec> in{{a, 10 * C}};
    assert(plan(in, lim(C, 4095), 4).n == 5);
  }
  { // leg_bound holds over random shapes (512-aligned, the reader's alignment)
    uint64_t s = 88172645463325252ull;
    auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int t = 0; t < 2000; ++t) {
      const unsigned n = 1 + next() % 6; std::vector<iovec> in; int64_t total = 0;
      uint8_t* at = a;
      for (unsigned i = 0; i < n; ++i) {
        const size_t len = 512 * (1 + next() % 200);
        const size_t skip = 512 * (next() % 9);
        at += skip; if (at + len > a + (1 << 20)) break;
        in.push_back({at, len}); total += len; at += len;
      }
      if (in.empty()) continue;
      const int64_t cut = 4096 * (1 + next() % 16);
      Plan p = plan(in, lim(cut, 4095), 255);
      assert(p.n <= leg_bound(total, in.size(), cut)); tiles(in, p, cut);
    }
  }
  { // a queue directory: values read, source named; a missing attribute is a clear refusal
    const std::string dir = argv[1];
    std::ofstream(dir + "/max_sectors_kb") << "256\n";
    std::ofstream(dir + "/max_segments") << "65\n";
    std::ofstream(dir + "/virt_boundary_mask") << "4095\n";
    DeviceLimits d; std::string why;
    assert(limits_from_queue_dir(dir, &d, &why) && d.cut_bytes == 262144 && d.virt_mask == 4095 && d.source == dir);
    std::remove((dir + "/virt_boundary_mask").c_str());
    assert(limits_from_queue_dir(dir, &d, &why) && d.virt_mask == 4095);   // absent: conservative 4095
    std::remove((dir + "/max_segments").c_str());
    assert(!limits_from_queue_dir(dir, &d, &why) && why.find("max_segments") != std::string::npos);
  }
  { // a file on tmpfs has no block queue: the fallback, with its reason
    const int fd = open(argv[2], O_RDONLY);
    assert(fd >= 0);
    const DeviceLimits d = device_limits(fd);
    close(fd);
    assert(d.cut_bytes == kFallbackCutBytes && d.virt_mask == kFallbackVirtMask);
    assert(d.source.rfind("fallback: ", 0) == 0);
    std::cout << "fallback source: " << d.source << "\n";
  }
  std::cout << "PASS read_cuts\n";
}
"""


def test_read_cuts_planner_and_limits(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    if not Path("/dev/shm").is_dir():
        pytest.skip("no /dev/shm (tmpfs) for the fallback case")
    source = tmp_path / "cuts.cpp"
    source.write_text(_UNIT)
    binary = tmp_path / "cuts"
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Wextra", "-I", str(INCLUDE), str(source), "-o", str(binary)],
        check=True,
    )
    queue = tmp_path / "queue"
    queue.mkdir()
    shm = Path("/dev/shm") / f"read-cuts-{os.getpid()}"
    shm.write_bytes(b"\0" * 4096)
    try:
        done = subprocess.run([binary, str(queue), str(shm)], text=True, capture_output=True, timeout=30)
    finally:
        shm.unlink()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS read_cuts" in done.stdout
```

- [ ] **Step 2: Run it and confirm it fails**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_read_cuts.py::test_read_cuts_planner_and_limits`
Expected: FAIL at compile time. `moe/expert_stream/host/read_cuts.h` does not exist.

- [ ] **Step 3: Implement** `python/sglang/kernels/jit/csrc/moe/expert_stream/host/read_cuts.h`:

```cpp
// Cutting a read into legs its block device takes whole (plan 2026-09-28-iopoll-read-cuts). Under IORING_SETUP_IOPOLL
// io_uring issues a read non-blocking and the polled bio carries REQ_NOWAIT; the block layer refuses to split such a
// bio (-EAGAIN) and io_uring punts the read to an io-wq worker. A read needs a split when it is larger than the
// queue's max_sectors_kb / max_segments, or when two of its iovecs meet off the NVMe PRP boundary (virt_boundary_mask).
// Cutting at both keeps every leg one request (analysis/dsv41-drive/iopoll/diagnosis.md).
#pragma once

#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>

namespace sglang::expert_stream {

constexpr int64_t kCutPage = 4096;
constexpr int64_t kFallbackCutBytes = 128 * 1024;
constexpr uint64_t kFallbackVirtMask = 4095;

struct DeviceLimits {
  int64_t cut_bytes = kFallbackCutBytes;   // the largest leg: whole pages
  uint64_t virt_mask = kFallbackVirtMask;  // a join off (mask + 1) starts a new leg; 0: no such rule
  std::string source;                      // the queue directory, "fallback: <why>", or a test's label
};

// The largest leg the queue takes as one request: max_sectors_kb, and max_segments pages less one (a leg that starts
// mid-page spans one page more than its length), rounded down to whole pages, never below one page. max_segments 0
// (unknown) leaves max_sectors_kb alone.
inline int64_t cut_bytes_for(int64_t max_sectors_kb, int64_t max_segments) {
  int64_t bytes = max_sectors_kb * 1024;
  if (max_segments > 1) bytes = std::min(bytes, (max_segments - 1) * kCutPage);
  return std::max(kCutPage, bytes / kCutPage * kCutPage);
}

inline bool read_sysfs_number(const std::string& path, int64_t* out) {
  std::ifstream f(path);
  long long v = -1;
  if (!(f >> v) || v < 0) return false;
  *out = static_cast<int64_t>(v);
  return true;
}

// Limits from a block queue directory (.../queue). virt_boundary_mask absent (older kernels) counts as 4095.
inline bool limits_from_queue_dir(const std::string& dir, DeviceLimits* out, std::string* why) {
  int64_t kb = 0, segments = 0, mask = 0;
  if (!read_sysfs_number(dir + "/max_sectors_kb", &kb) || kb == 0) {
    *why = dir + "/max_sectors_kb unreadable";
    return false;
  }
  if (!read_sysfs_number(dir + "/max_segments", &segments) || segments == 0) {
    *why = dir + "/max_segments unreadable";
    return false;
  }
  if (!read_sysfs_number(dir + "/virt_boundary_mask", &mask)) mask = static_cast<int64_t>(kFallbackVirtMask);
  *out = DeviceLimits{cut_bytes_for(kb, segments), static_cast<uint64_t>(mask), dir};
  return true;
}

// The queue directory of the block device holding a file (st_dev): /sys/dev/block/M:m resolved, and for a partition
// its disk's. Empty with the reason when there is none (tmpfs and other major-0 filesystems, a missing link).
inline std::string queue_dir_for(dev_t dev, std::string* why) {
  if (major(dev) == 0) {
    *why = "not on a block device (st_dev major 0)";
    return {};
  }
  const std::string link = "/sys/dev/block/" + std::to_string(major(dev)) + ":" + std::to_string(minor(dev));
  char real[PATH_MAX];
  if (realpath(link.c_str(), real) == nullptr) {
    *why = link + " does not resolve";
    return {};
  }
  std::string dir(real);
  struct stat st;
  if (stat((dir + "/partition").c_str(), &st) == 0) dir = dir.substr(0, dir.rfind('/'));
  if (stat((dir + "/queue").c_str(), &st) != 0) {
    *why = dir + "/queue absent";
    return {};
  }
  return dir + "/queue";
}

// The file's device limits, or the fallback (128 KiB, 4 KiB boundary) with its reason in `source`.
inline DeviceLimits device_limits(int fd) {
  std::string why = "fstat failed";
  struct stat st;
  if (fstat(fd, &st) == 0) {
    const std::string dir = queue_dir_for(st.st_dev, &why);
    DeviceLimits found;
    if (!dir.empty() && limits_from_queue_dir(dir, &found, &why)) return found;
  }
  DeviceLimits fallback;
  fallback.source = "fallback: " + why;
  return fallback;
}

// One leg over `out`: iovecs [first, first + count), `bytes` long; `gap` when a boundary gap (not the size) opened it.
struct CutLeg {
  unsigned first = 0;
  unsigned count = 0;
  int64_t bytes = 0;
  bool gap = false;
};

// Cut a read's iovecs into legs: a new leg at every join off the virt boundary, and wherever a leg reaches
// lim.cut_bytes (splitting that iovec in two). `out` receives the legs' iovecs in order (at most count + legs - 1),
// `legs` the legs over them. Returns the leg count, or max_legs + 1 when `legs` or `out` is too small.
inline unsigned cut_legs(
    const iovec* in, unsigned count, const DeviceLimits& lim, iovec* out, unsigned max_out, CutLeg* legs,
    unsigned max_legs) {
  const auto on_boundary = [&](uintptr_t at) { return lim.virt_mask == 0 || (at & lim.virt_mask) == 0; };
  unsigned n = 0, k = 0;
  for (unsigned i = 0; i < count; ++i) {
    auto* at = static_cast<uint8_t*>(in[i].iov_base);
    size_t left = in[i].iov_len;
    const bool gap = i > 0 && !(on_boundary(reinterpret_cast<uintptr_t>(in[i - 1].iov_base) + in[i - 1].iov_len) &&
                                on_boundary(reinterpret_cast<uintptr_t>(at)));
    bool open_leg = n == 0 || gap;
    bool first_piece = true;
    while (left > 0) {
      if (!open_leg && legs[n - 1].bytes >= lim.cut_bytes) open_leg = true;
      if (open_leg) {
        if (n == max_legs) return max_legs + 1;
        legs[n++] = CutLeg{k, 0, 0, first_piece && gap};
        open_leg = false;
      }
      if (k == max_out) return max_legs + 1;
      CutLeg& leg = legs[n - 1];
      const size_t take = std::min<size_t>(left, static_cast<size_t>(lim.cut_bytes - leg.bytes));
      out[k++] = iovec{at, take};
      ++leg.count;
      leg.bytes += static_cast<int64_t>(take);
      at += take;
      left -= take;
      first_piece = false;
    }
  }
  return n;
}

// An upper bound on the legs of a read of `longest` bytes over `iovecs` iovecs cut at `cut_bytes`: the size cuts, one
// more per gap-separated run, and one more per registered-buffer change at a join (fixed read modes).
inline size_t leg_bound(int64_t longest, size_t iovecs, int64_t cut_bytes) {
  return static_cast<size_t>(longest / cut_bytes) + 2 * iovecs;
}

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: Run it and confirm it passes**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_read_cuts.py::test_read_cuts_planner_and_limits`
Expected: PASS, printing a `fallback source: fallback: not on a block device (st_dev major 0)` line.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/read_cuts.h \
  test/registered/unit/kernels/test_expert_stream_read_cuts.py
git commit -m "feat(expert-stream): read_cuts.h, device-sized leg planner and sysfs queue limits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

---

### Task 4: Leg storage sized at open, and the queue depth that follows it

This task changes storage only. With cuts off, every read keeps exactly the legs it has today (one, or the fixed fan-out), and the golden SQE set is unchanged.

**Recorded decision 3: depth resizing.** Credit counts SQEs (Task 7's invariant). Cuts multiply SQEs per read by the leg count, so a fixed default credit of `16 · parts` would cut the rows in flight by about 10×.
- **Default.** The default credit becomes `min(32768, 16 · parts · leg_stride_)` when cuts are on, and stays `16 · parts` when they are off.
- **Explicit.** An explicit `QUEUE_DEPTH` is kept, but `open()` refuses it when it is below `leg_stride_`, whenever cuts or fixed reads are on. This generalizes the existing fixed-mode check, which today compares against `max_iovecs()`.
- **Why a refusal, not a wait.** `leg_stride_` is the leg bound of the longest read. An all-or-nothing reservation of more SQEs than the SQ holds would never fit, even with `pending == 0`, and would hang the reader.
- **Ring size.** The ring is sized to the credit (`io_.init(queue_depth())`). For production (3 roots, longest part ~4.4 MB, smallest cut 262144, 6 iovecs) that is `16 · 3 · (16 + 12) = 1344` SQEs, which the kernel rounds to 2048 SQ / 4096 CQ entries.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_core.h`
- Modify: `test/registered/unit/kernels/test_expert_stream_read_cuts.py`

**Interfaces:**
- Consumes: `read_cuts.h` (Task 3), `UringOptions::read_cuts_on()` (Task 1).
- Produces:
  - `ReaderCore` members: `std::vector<DeviceLimits> limits_` (one per file, in `t_.paths` order), `bool cuts_`, `unsigned leg_stride_`, `size_t iov_stride_`, `int64_t leg_cut_cap_` (0: device limits).
  - `static constexpr unsigned kMaxLegs = 255` (the 8-bit leg field of the tag).
  - Accessors `int64_t leg_stride() const`, `int64_t min_cut_bytes() const` (0 when cuts are off).
  - A private `int64_t longest_read() const`, the longest `t_.extents` length.

- [ ] **Step 1: Write the failing tests** (appended to `test_expert_stream_read_cuts.py`; the FFI plumbing they need arrives in Task 5, so they are written now and fail at the plumbing too):

```python
import errno

import torch

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes, read_rows_with_fault
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

PREFIX = "SGLANG_EXPERT_STREAM_URING_"
EXPERTS = [10, 3, 7, 0, 11, 5, 1, 8, 2, 9, 4]
SLOTS = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
CUT = 8192


@pytest.fixture
def uring_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(PREFIX):
            monkeypatch.delenv(key)

    def set_(**values):
        for key, value in values.items():
            monkeypatch.setenv(PREFIX + key, str(value))

    return set_


def _setup(tmp_path, images, weights=None):
    root = tmp_path / "ckpt"
    root.mkdir()
    dims = {} if images else dict(hidden=256, inter=512)
    return ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)


def _snapshot(slabs):
    if isinstance(slabs, dict):
        return {k: _snapshot(v) for k, v in slabs.items()}
    return slabs.clone()


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return same_bytes(a, b)


def _read(s, **faults):
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, direct=False, max_sqes=65536, **faults)
    return result, log, info, record, _snapshot(s.slabs)


def _pieces(images, on):
    return ({"piece_stream": True} | ({} if images else {"pack_workers": 2})) if on else {}


def test_cuts_off_keeps_todays_credit_and_legs(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    result, _, info, _, _ = _read(s)
    assert result == 1 and info["cut_reads"] == 0 and info["min_cut_bytes"] == 0
    assert info["credit"] == 16 * 3


def test_credit_scales_with_the_leg_bound(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    result, _, info, _, _ = _read(s, leg_cut_cap=CUT)
    assert result == 1 and info["leg_stride"] > 1 and info["min_cut_bytes"] == CUT
    assert info["credit"] == min(32768, 16 * 3 * info["leg_stride"])


def test_explicit_depth_below_the_leg_bound_is_refused(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    uring_env(QUEUE_DEPTH=2)
    with pytest.raises(RuntimeError, match="SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=2"):
        _read(s, leg_cut_cap=CUT)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_read_cuts.py -k "credit or depth"`
Expected: FAIL, because `_fault_tensor` has no `leg_cut_cap` and `info` has no `cut_reads`. This stays red until Task 5 Step 3. Commit Task 4 only after Step 4 shows the golden and Task 7 suites unchanged.

- [ ] **Step 3: Implement in `reader_core.h`.** Add `#include "read_cuts.h"` after `#include "piece_geometry.h"`.

Replace `static constexpr unsigned kMaxLegs = 16;` and its comment with:

```cpp
  // Legs per read are sized at open() (leg_stride_): the fixed fan-out (a leg per registered buffer) and, with read
  // cuts, device-sized legs (read_cuts.h). kMaxLegs is the tag's 8-bit leg field.
  static constexpr unsigned kMaxLegs = 255;
```

Add public accessors next to `fanout_sqes()`:

```cpp
  // Legs a read may have (storage stride) and the smallest cut in force (0: cuts off).
  int64_t leg_stride() const {
    return leg_stride_;
  }
  int64_t min_cut_bytes() const {
    if (!cuts_) return 0;
    int64_t least = INT64_MAX;
    for (const auto& l : limits_)
      least = std::min(least, l.cut_bytes);
    return least;
  }
```

In the constructor, after `configured_queue_depth_ = ...`:

```cpp
    cuts_requested_ = UringOptions::from_env().read_cuts_on();
```

In `open()`, replace the block from `if (!derived().open_memory()) return false;` through the end of the `if (fixed_reads()) { ... }` refusal block with:

```cpp
    if (!derived().open_memory()) return false;
    size_legs();
    if (!size_extents()) return false;
    if (!io_.init(queue_depth())) return false;
    if ((fixed_reads() || cuts_) && queue_depth() < leg_stride_) {
      throw std::runtime_error(
          error_prefix<Layout>() + "reads fanned out or cut into legs need a queue depth of at least " +
          std::to_string(leg_stride_) + " (the widest read's legs); SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=" +
          std::to_string(queue_depth()));
    }
```

Add `size_legs()` and `longest_read()` in the private section, next to `size_extents()`:

```cpp
  // Per-file limits (read cuts) and the leg/iovec strides every descriptor's storage uses. With cuts off a read has
  // one leg, or (fixed modes) one per registered buffer its iovecs meet: at most max_iovecs(), today's bound.
  void size_legs() {
    cuts_ = cuts_requested_ || leg_cut_cap_ > 0;
    limits_.clear();
    for (size_t f = 0; f < fds_.size(); ++f) {
      if (leg_cut_cap_ > 0) {
        limits_.push_back(DeviceLimits{std::max(kCutPage, leg_cut_cap_ / kCutPage * kCutPage), kFallbackVirtMask,
                                       "test cap"});
      } else {
        limits_.push_back(device_limits(fds_[f]));
      }
    }
    const size_t iovecs = std::max<size_t>(1, derived().max_iovecs());
    size_t want = iovecs;
    if (cuts_) want = leg_bound(longest_read(), iovecs, min_cut_bytes());
    if (want > kMaxLegs) {
      throw std::runtime_error(
          error_prefix<Layout>() + "reads cut at " + std::to_string(min_cut_bytes()) + " B need up to " +
          std::to_string(want) + " legs, more than " + std::to_string(kMaxLegs) + "; the cut is too small for reads of " +
          std::to_string(longest_read()) + " B");
    }
    leg_stride_ = static_cast<unsigned>(want);
    iov_stride_ = iovecs + (cuts_ ? leg_stride_ : 0);
  }

  int64_t longest_read() const {
    int64_t longest = 0;
    for (const auto& e : t_.extents)
      longest = std::max(longest, e.length);
    return longest;
  }
```

Change `queue_depth()` to:

```cpp
  unsigned queue_depth() const {
    if (configured_queue_depth_ != 0) return configured_queue_depth_;
    const size_t per_read = cuts_ ? leg_stride_ : 1;
    return static_cast<unsigned>(
        std::min<size_t>(32768, static_cast<size_t>(kQueueDepth) * static_cast<size_t>(t_.parts) * per_read));
  }
```

In `size_extents()`, replace `legs_.assign(extents * kMaxLegs, Leg{});` with `legs_.assign(extents * leg_stride_, Leg{});` and `iovecs_.assign(extents * derived().max_iovecs(), iovec{});` with `iovecs_.assign(extents * iov_stride_, iovec{});`. Also add the scratch buffers:

```cpp
    iov_scratch_.assign(iov_stride_, iovec{});
    cut_scratch_.assign(leg_stride_, CutLeg{});
    fixed_scratch_.assign(iov_stride_, FixedLeg{});
```

Then replace every remaining `* kMaxLegs` index with `* leg_stride_`: `plan_legs`, `refill`, `process` (the stale check and `Leg& g`) and `retire`. Replace every `* derived().max_iovecs()` iovec index with `* iov_stride_`: `plan_legs` and `refill`.

In `plan_legs`, replace `FixedLeg fixed[kMaxLegs];` with `FixedLeg* fixed = fixed_scratch_.data();`. Task 5 rewrites `plan_legs`; this step only swaps the storage.

In `refill`, generalize the SQ-room check from fixed reads to any read with more than one leg:

```cpp
      if constexpr (requires(const Reader& reader) { reader.sq_space(); }) {
        if (n > 1 && io_.sq_space() < n) break;
      }
```

Add the members after `unsigned configured_queue_depth_ = 0;`:

```cpp
  bool cuts_requested_ = false;       // READ_CUTS resolved (UringOptions::read_cuts_on)
  bool cuts_ = false;                 // in force: requested, or a test cap
  int64_t leg_cut_cap_ = 0;           // test only (fault word leg_cut_cap): every file cut at this, 0: device limits
  std::vector<DeviceLimits> limits_;  // per file, t_.paths order
  unsigned leg_stride_ = 1;
  size_t iov_stride_ = 1;
  std::vector<iovec> iov_scratch_;
  std::vector<CutLeg> cut_scratch_;
  std::vector<FixedLeg> fixed_scratch_;
```

`FixedLeg` (`{unsigned first, count; int buffer; size_t bytes;}`) is declared in `file_reader.h`, which `reader_core.h` already includes. Add `#include <cstdio>` for the Task 5 `fprintf` if it is not already present.

- [ ] **Step 4: Run the unchanged suites**

Run: CPU-TEST `test_expert_stream_reader_golden.py test_expert_stream_fixed_buffers.py test_expert_stream_uring_integration.py`
Expected: identical counts to Task 0. The golden pins that the SQE set with cuts off is unchanged.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_core.h \
  test/registered/unit/kernels/test_expert_stream_read_cuts.py
git commit -m "refactor(expert-stream): size leg storage and default credit at open from the read-cut leg bound

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

---

### Task 5: Cut legs in `plan_legs`, with counters, the test cap and the FFI plumbing

**How a read becomes legs (and why nothing downstream changes).** `plan_legs` builds the destination iovecs as today. With cuts on, it rewrites them through `cut_legs`, using the limits of that read's file, into runs, each within the device limits. In a fixed read mode each run is further cut by `fixed_legs` at registered-buffer changes. The result is the same `Leg{start, bytes, expected, done, first_iov, iov_count, buffer, state}` list Task 7 introduced, so these all apply unchanged:
- `refill`'s all-or-nothing reservation;
- `process`'s per-leg short and retry handling (`advance_leg` trims that leg's own iovec slice);
- the fail-once-after-drain path;
- `retire` only when every leg is Done;
- piece publish at retire.

`d.legs` stays a `uint8_t`, because `leg_stride_ ≤ kMaxLegs = 255`.

**Counters** (first plans only, like `fixed_cuts`): `cut_reads`, the reads planned as more than one cut run, and `gap_cuts`, the runs opened by a boundary gap. Each drive gets one stderr line when cuts are on, printed at open.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_core.h` (`plan_legs`, `open`, counters, `set_leg_cut_cap`)
- Modify: `.../host/any_reader.h`, `.../host/read_fault.h`, `.../host/ffi_exports.h`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py`
- Modify: `test/registered/unit/kernels/test_expert_stream_read_cuts.py`

**Interfaces:**
- Consumes: `cut_legs`, `CutLeg`, `DeviceLimits` (Task 3); `limits_`, `leg_stride_`, `iov_stride_`, the scratch buffers (Task 4).
- Produces:
  - `ReaderCore`: `void set_leg_cut_cap(int64_t cap)` (before `open()`), `int64_t cut_reads() const`, `int64_t gap_cuts() const`. `AnyReader` forwards these plus `leg_stride` and `min_cut_bytes`.
  - Fault word 31 `leg_cut_cap`; `kFaultWords = 32`. Python `_fault_tensor(..., leg_cut_cap: int = 0)` is appended last.
  - `read_rows_sqes` info grows to 11 int64: `[result, sqes, descriptors, credit, cqes, fixed_cuts, fanout_sqes, cut_reads, gap_cuts, min_cut_bytes, leg_stride]`. The Python dict gains the four new keys.
  - `read_rows_faulted` results grow to 12, with `[10] cut_reads, [11] gap_cuts` after the first read. `stats` gains both.
  - The stderr line, one per drive, at open when cuts are on: `expert stream io_uring: read cuts: drive=<i> dev=<M>:<m> cut_bytes=<n> virt_mask=<n> source=<queue dir | fallback: why | test cap>`.

- [ ] **Step 1: Write the failing tests** (append):

```python
@pytest.mark.parametrize("images", [False, True], ids=["bounce", "images"])
@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_cut_reads_are_byte_identical_and_within_the_cut(tmp_path, uring_env, images, pieces):
    s = _setup(tmp_path, images, (1.0, 1.0, 1.0))
    base_result, base_log, _, base_rec, base_bytes = _read(s, **_pieces(images, pieces))
    result, log, info, rec, cut_bytes = _read(s, leg_cut_cap=CUT, **_pieces(images, pieces))
    assert base_result == result == 1 and _equal(cut_bytes, base_bytes)
    assert info["cut_reads"] > 0 and all(0 < length <= CUT for _, _, length, _ in log)
    assert all(offset % 512 == 0 and length % 512 == 0 for _, offset, length, _ in log)
    assert rec["retried_bytes"] == 0 and rec["submitted_bytes"] == base_rec["submitted_bytes"]

    # The cut SQEs cover exactly the uncut ones' file ranges.
    def ranges(entries):
        spans = sorted((f, o, o + n) for f, o, n, _ in entries)
        merged = []
        for f, a, b in spans:
            if merged and merged[-1][0] == f and merged[-1][2] == a:
                merged[-1] = (f, merged[-1][1], b)
            else:
                merged.append((f, a, b))
        return merged

    assert ranges(log) == ranges(base_log)


def test_cuts_are_off_by_default_outside_iopoll(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    _, base_log, _, _, _ = _read(s)
    uring_env(MODE="default", READ_CUTS="auto")
    result, log, info, _, _ = _read(s)
    assert result == 1 and info["cut_reads"] == info["gap_cuts"] == 0 and sorted(log) == sorted(base_log)


def test_read_cuts_on_uses_the_files_device_limits(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    _, _, _, _, base_bytes = _read(s)
    uring_env(READ_CUTS=1)
    result, log, info, _, cut_bytes = _read(s)
    assert result == 1 and _equal(cut_bytes, base_bytes)
    assert info["min_cut_bytes"] >= 4096 and all(length <= info["min_cut_bytes"] for _, _, length, _ in log)


def test_gap_cuts_at_slab_rows_off_the_page(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    off_page = [name for layer in s.slabs.values() for name, t in layer.items()
                if (t.numel() * t.element_size() // t.shape[0]) % 4096]
    assert off_page, "the fixture has no slab row off the page; gap cuts are untested"
    _, _, _, _, base_bytes = _read(s)
    result, _, info, _, cut_bytes = _read(s, leg_cut_cap=1 << 30)   # a huge cut: only gaps cut
    assert result == 1 and info["gap_cuts"] > 0 and _equal(cut_bytes, base_bytes)


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_cut_legs_completing_out_of_order(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    result, _, info, _, cut_bytes = _read(s, leg_cut_cap=CUT, reverse_cqes=True, **_pieces(True, pieces))
    assert result == 1 and info["cut_reads"] > 0 and _equal(cut_bytes, base_bytes)


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_a_held_cut_leg_keeps_its_read_unretired_and_its_pieces_unpublished(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    result, _, _, rec, cut_bytes = _read(s, leg_cut_cap=CUT, hold_ordinal=0, leg=1, **_pieces(True, pieces))
    assert result == 1 and _equal(cut_bytes, base_bytes)
    if pieces:
        assert rec["pieces_published"] == len(EXPERTS) * 8 and rec["piece_publish_refused"] == 0


def test_one_short_cut_leg_resubmits_only_that_leg(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    _, clean_log, _, _, base_bytes = _read(s, leg_cut_cap=CUT)
    result, log, _, rec, cut_bytes = _read(s, leg_cut_cap=CUT, part=0, part_short=512, leg=1)
    assert result == 1 and _equal(cut_bytes, base_bytes)
    assert len(log) == len(clean_log) + 1
    extra = sorted(set(log) - set(clean_log))
    assert len(extra) == 1 and rec["retried_bytes"] == extra[0][2]


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_one_failing_cut_leg_fails_the_read_once_after_every_leg_is_reaped(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    stats, cqes = {}, []
    first, then = read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=False, part=0, part_error=errno.EIO,
        ordinal=0, leg=1, leg_cut_cap=CUT, stats=stats, cqes=cqes, **_pieces(True, pieces))
    assert (first, then) == (0, 1)
    assert stats["unfinished_jobs"] == 0 and stats["cut_reads"] > 0


@pytest.mark.parametrize("submit_first", [False, True], ids=["unconsumed", "in_flight"])
def test_ring_reset_with_cut_legs(tmp_path, uring_env, submit_first):
    s = _setup(tmp_path, True)
    first, then = read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=False,
        submit_error=errno.EIO, submit_call=1, submit_first=submit_first, leg_cut_cap=CUT)
    assert (first, then) == (0, 1)


@pytest.mark.parametrize("read_mode", ["fixed", "readv_fixed"])
def test_cuts_compose_with_the_fixed_fan_out(tmp_path, uring_env, read_mode):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s)
    uring_env(READ_MODE=read_mode)
    try:
        result, log, info, _, cut_bytes = _read(s, leg_cut_cap=CUT, fixed_chunk_cap=64 * 1024)
    except RuntimeError as e:
        if "unsupported by the running kernel" in str(e) or "requires liburing 2.10" in str(e):
            pytest.skip(str(e))
        raise
    assert result == 1 and _equal(cut_bytes, base_bytes)
    assert info["cut_reads"] > 0 and info["fixed_cuts"] > 0 and all(length <= CUT for _, _, length, _ in log)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: CPU-TEST `test/registered/unit/kernels/test_expert_stream_read_cuts.py`
Expected: every FFI test FAILs (`_fault_tensor() got an unexpected keyword argument 'leg_cut_cap'`). The Task 3 native test still passes.

- [ ] **Step 3: Implement.**

`read_fault.h`: add `word 31 (leg_cut_cap, not a fault) cuts every read at that many bytes before the reader opens (0: READ_CUTS and the device limits)` to the layout comment, and change `constexpr int64_t kFaultWords = 31;` to `32`.

`ffi_exports.h`: at each of the four sites (lines ~176, 233, 299 and 399), add a line after `reader.set_fixed_chunk_cap(f[28]);`:

```cpp
    reader.set_leg_cut_cap(f[31]);
```

In `read_rows_sqes`, update the doc comment to "`info` 11 int64: ... fixed_cuts, fanout_sqes, cut_reads, gap_cuts, min_cut_bytes, leg_stride". Require `info` of size 11 wherever it is verified (search `TensorMatcher({7})`), and after `out[6] = reader.fanout_sqes();` add:

```cpp
    out[7] = reader.cut_reads();
    out[8] = reader.gap_cuts();
    out[9] = reader.min_cut_bytes();
    out[10] = reader.leg_stride();
```

In `read_rows_faulted`, update the comment to `results[0..11]` "..., and its fixed_cuts, fanout_sqes, cut_reads and gap_cuts after the first read". Require size 12 where `results` is verified, and after `out[9] = reader.fanout_sqes();` add:

```cpp
    out[10] = reader.cut_reads();
    out[11] = reader.gap_cuts();
```

`any_reader.h`, after `fanout_sqes()`:

```cpp
  void set_leg_cut_cap(int64_t cap) {
    visit([&](auto& r) { r.set_leg_cut_cap(cap); });
  }
  int64_t cut_reads() const {
    return visit([](auto& r) { return r.cut_reads(); });
  }
  int64_t gap_cuts() const {
    return visit([](auto& r) { return r.gap_cuts(); });
  }
  int64_t min_cut_bytes() const {
    return visit([](auto& r) { return r.min_cut_bytes(); });
  }
  int64_t leg_stride() const {
    return visit([](auto& r) { return r.leg_stride(); });
  }
```

`reader_core.h`, the public members:

```cpp
  // Test only (fault word leg_cut_cap): cut every read at `cap` bytes (whole pages) on a 4 KiB boundary, whatever
  // READ_CUTS says, before open(). 0 restores READ_CUTS and the device limits.
  void set_leg_cut_cap(int64_t cap) {
    leg_cut_cap_ = std::max<int64_t>(0, cap);
  }
  // Reads planned as more than one cut run, and the runs a boundary gap opened (first plans).
  int64_t cut_reads() const {
    return cut_reads_;
  }
  int64_t gap_cuts() const {
    return gap_cuts_;
  }
```

with members `int64_t cut_reads_ = 0, gap_cuts_ = 0;`.

In `open()`, right after `size_legs();`:

```cpp
    if (cuts_) {
      std::vector<bool> said(devs_.size(), false);
      for (size_t f = 0; f < fds_.size(); ++f) {
        const size_t drive = file_drive_[f];
        if (drive >= said.size() || said[drive]) continue;
        said[drive] = true;
        const auto dev = static_cast<dev_t>(devs_[drive]);
        std::fprintf(
            stderr,
            "expert stream io_uring: read cuts: drive=%zu dev=%u:%u cut_bytes=%lld virt_mask=%llu source=%s\n",
            drive, major(dev), minor(dev), static_cast<long long>(limits_[f].cut_bytes),
            static_cast<unsigned long long>(limits_[f].virt_mask), limits_[f].source.c_str());
      }
    }
```

Replace `plan_legs` entirely:

```cpp
  // A descriptor's legs, on its first preparation: its iovecs from `done` 0 (destination); with read cuts, rewritten
  // into runs within its file's device limits (cut_legs); in a fixed read mode each run cut again at registered-
  // buffer changes (fixed_legs). Without either it is one leg, today's read. Each leg's `start` is the prefix sum of
  // the earlier legs' bytes; a leg wholly past the read's end-of-file expectation is Done at once.
  void plan_legs(uint32_t index) {
    ExtentDesc& d = descs_[index];
    iovec* iov = &iovecs_[static_cast<size_t>(index) * iov_stride_];
    unsigned count = derived().destination(d, iov);
    Leg* legs = &legs_[static_cast<size_t>(index) * leg_stride_];
    CutLeg* runs = cut_scratch_.data();
    unsigned run_count = 1;
    runs[0] = CutLeg{0, count, d.read->length, false};
    if (cuts_) {
      run_count = cut_legs(iov, count, limits_[d.read->file], iov_scratch_.data(), static_cast<unsigned>(iov_stride_),
                           runs, leg_stride_);
      if (run_count > leg_stride_)
        throw std::logic_error(error_prefix<Layout>() + "a read needs more legs than open() sized");
      count = runs[run_count - 1].first + runs[run_count - 1].count;
      std::copy(iov_scratch_.begin(), iov_scratch_.begin() + count, iov);
      if (run_count > 1) ++cut_reads_;
      for (unsigned r = 0; r < run_count; ++r)
        gap_cuts_ += runs[r].gap ? 1 : 0;
    }
    unsigned n = 0;
    int64_t start = 0;
    for (unsigned r = 0; r < run_count; ++r) {
      FixedLeg* parts = fixed_scratch_.data();
      unsigned k = 1;
      parts[0] = FixedLeg{0, runs[r].count, -1, static_cast<size_t>(runs[r].bytes)};
      if constexpr (requires(const Reader& reader, const iovec* v, unsigned c, FixedLeg* out) {
                      reader.fixed_legs(v, c, out);
                    }) {
        if (fixed_reads()) k = io_.fixed_legs(iov + runs[r].first, runs[r].count, parts);
      }
      for (unsigned l = 0; l < k; ++l) {
        if (n == leg_stride_) throw std::logic_error(error_prefix<Layout>() + "a read needs more legs than open() sized");
        const int64_t bytes = static_cast<int64_t>(parts[l].bytes);
        legs[n++] = Leg{start, bytes, std::clamp<int64_t>(d.expected - start, 0, bytes), 0,
                        runs[r].first + parts[l].first, parts[l].count, parts[l].buffer, LegState::Idle};
        start += bytes;
      }
    }
    if (start != d.read->length) throw std::logic_error(error_prefix<Layout>() + "a read's legs do not cover the read");
    if constexpr (requires(Reader& reader) { reader.note_fanout(1u); }) {
      if (fixed_reads()) io_.note_fanout(n);
    }
    if (fixed_reads() && n > 1) {
      ++fixed_cuts_;
      fanout_sqes_ += n;
    }
    for (unsigned l = 0; l < n; ++l)
      if (legs[l].expected == 0) legs[l].state = LegState::Done;
    d.legs = static_cast<uint8_t>(n);
  }
```

`FixedLeg` (`file_reader.h:54`) is `{unsigned first, count; int buffer; size_t bytes;}`, the order the aggregate above uses.

`expert_stream_transport.py`:
- `_fault_tensor`: add the keyword `leg_cut_cap: int = 0,` after `ring_reset_fail`, and append `leg_cut_cap,` as the last list element.
- `read_rows_sqes`: `info = torch.zeros(11, dtype=torch.int64)`. Unpack with `result, count, descriptors, credit, cqes, fixed_cuts, fanout_sqes, cut_reads, gap_cuts, min_cut_bytes, leg_stride = info.tolist()`, and add `cut_reads=cut_reads, gap_cuts=gap_cuts, min_cut_bytes=min_cut_bytes, leg_stride=leg_stride` to the dict. Extend the docstring.
- `read_rows_with_fault`: `results = torch.zeros(12, ...)`, and add `cut_reads=int(results[10]), gap_cuts=int(results[11])` to `stats.update`. Extend the docstring: "`leg_cut_cap` (bytes, 0: READ_CUTS and the device limits) cuts every read into legs of at most that many bytes on a 4 KiB boundary, whatever READ_CUTS says; `leg` then narrows the faults to a cut leg as well".

- [ ] **Step 4: Run it and confirm it passes**

Run: CPU-TEST on:
- `test/registered/unit/kernels/test_expert_stream_read_cuts.py`
- `test_expert_stream_reader_golden.py`
- `test_expert_stream_fixed_buffers.py`
- `test_expert_stream_uring_options.py`
- `test_expert_stream_uring_integration.py`

Expected: the new file passes entirely, and the others keep their Task 0 counts. Then run SUITE and compare with Task 0: the delta should be exactly the new test file's count.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/{reader_core.h,any_reader.h,read_fault.h,ffi_exports.h} \
  python/sglang/kernels/ops/moe/expert_stream_transport.py test/registered/unit/kernels/test_expert_stream_read_cuts.py
git commit -m "feat(expert-stream): cut every read into device-sized legs (READ_CUTS), reusing the fan-out legs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

---

### Task 6: divix01 real-kernel verification, the punt check and mutants

**Files:**
- Create: `test/manual/dsv41/test_expert_stream_read_cuts_nvme.py`

**Interfaces:** Consumes everything above through the FFI and `read_cuts.h`.

- [ ] **Step 1: Write the manual test.** It runs on divix01 only, under the disk lock. `IOPOLL_CUTS_DIR` names a directory on an NVMe mirror root, which the test clears; for example `/mnt/nvme4/nvfp4-work/iopoll-cuts-tests`, since the SPCC's 256 KiB limit is the strictest.

```python
"""divix01 only (plan 2026-09-28-iopoll-read-cuts Task 6): the mirror roots' sysfs limits, and the io-wq punt check
through the real reader under IORING_SETUP_IOPOLL. Run under rowimg-disk.lock with IOPOLL_CUTS_DIR on an NVMe root.
Each phase runs in a fresh interpreter: io-wq workers belong to the process and linger after their last request, so
a shared process would carry one phase's workers into the next."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOTS = {"/mnt/nvme0": 520192, "/mnt/nvme2": 520192, "/mnt/nvme4": 262144}
LAYER = "dsv41_flash/exl3_row_images/layer-000.rows"
INCLUDE = Path(__file__).resolve().parents[3] / "python/sglang/kernels/jit/csrc"
pytestmark = pytest.mark.skipif(not all(Path(r, LAYER).exists() for r in ROOTS), reason="divix01 mirror roots only")

_LIMITS = r"""
#include "moe/expert_stream/host/read_cuts.h"
#include <fcntl.h>
#include <cstdio>
#include <unistd.h>
int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    int fd = open(argv[i], O_RDONLY | O_DIRECT);
    auto d = sglang::expert_stream::device_limits(fd);
    std::printf("%s %lld %llu %s\n", argv[i], (long long)d.cut_bytes, (unsigned long long)d.virt_mask, d.source.c_str());
    close(fd);
  }
}
"""


def test_sysfs_limits_of_the_mirror_roots(tmp_path):
    (tmp_path / "limits.cpp").write_text(_LIMITS)
    subprocess.run(["c++", "-std=c++20", "-I", str(INCLUDE), str(tmp_path / "limits.cpp"), "-o",
                    str(tmp_path / "limits")], check=True)
    out = subprocess.check_output([tmp_path / "limits", *[str(Path(r, LAYER)) for r in ROOTS]], text=True)
    for line, (root, cut) in zip(out.splitlines(), ROOTS.items()):
        path, got, mask, source = line.split(" ", 3)
        assert int(got) == cut and int(mask) == 4095 and source.endswith("/queue"), line


# One phase: build the fixture, read it N times through the reader with O_DIRECT under MODE=iopoll, and report the
# io-wq workers seen in /proc/self/task, the SQE lengths, and whether the slabs match the reference.
_PHASE = r"""
import json, os, sys, threading, time
from pathlib import Path
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes
root = Path(sys.argv[1]); cap = int(sys.argv[2])
s = ram_miss_setup(root, capacity=8, experts=8, layers=2, row_images=True, hidden=2048, inter=1536)
seen, stop = set(), threading.Event()
def sample():
    while not stop.is_set():
        for t in os.listdir("/proc/self/task"):
            try:
                if open(f"/proc/self/task/{t}/comm").read().startswith("iou-wrk"): seen.add(t)
            except OSError: pass
        time.sleep(0.002)
th = threading.Thread(target=sample); th.start()
lengths, ok, info = [], True, {}
try:
    for _ in range(20):
        result, log, info, _ = read_rows_sqes(s.tables, 1, list(range(8)), list(range(8)), direct=True,
                                              max_sqes=65536, leg_cut_cap=cap)
        ok = ok and result == 1
        lengths += [n for _, _, n, _ in log]
finally:
    stop.set(); th.join()
print(json.dumps({"ok": ok, "workers": len(seen), "max_len": max(lengths), "cut_reads": info.get("cut_reads", 0)}))
"""


def _phase(tmp, env_extra, cap=0):
    work = Path(os.environ["IOPOLL_CUTS_DIR"]) / tmp
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_EXPERT_STREAM_URING_")}
    env |= {"SGLANG_EXPERT_STREAM_URING_MODE": "iopoll", **env_extra}
    done = subprocess.run([sys.executable, "-c", _PHASE, str(work), str(cap)], env=env, text=True,
                          capture_output=True, timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.skipif("IOPOLL_CUTS_DIR" not in os.environ, reason="set IOPOLL_CUTS_DIR to a dir on an NVMe root")
def test_iopoll_punts_uncut_reads_and_never_cut_ones():
    cut = _phase("cut", {"SGLANG_EXPERT_STREAM_URING_READ_CUTS": "1"})
    uncut = _phase("uncut", {"SGLANG_EXPERT_STREAM_URING_READ_CUTS": "0"})
    # The control must punt, or the check proves nothing: its reads must exceed the device's limit.
    assert uncut["ok"] and uncut["max_len"] > 520192, f"raise hidden/inter: reads of {uncut['max_len']} B"
    assert uncut["workers"] > 0, uncut
    assert cut["ok"] and cut["cut_reads"] > 0 and cut["workers"] == 0, cut


@pytest.mark.skipif("IOPOLL_CUTS_DIR" not in os.environ, reason="set IOPOLL_CUTS_DIR to a dir on an NVMe root")
def test_a_cut_too_small_for_the_reads_is_refused():
    work = Path(os.environ["IOPOLL_CUTS_DIR"]) / "refuse"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_EXPERT_STREAM_URING_")}
    done = subprocess.run([sys.executable, "-c", _PHASE, str(work), "4096"], env=env, text=True,
                          capture_output=True, timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    assert done.returncode != 0 and "legs, more than 255" in done.stderr, done.stderr[-2000:]
```

The refusal test needs the longest read to exceed `(255 − 2·6) · 4096 ≈ 995 KB`, which the control phase's `max_len > 520192` does not guarantee. If the refusal test fails because the read succeeded, raise `hidden`/`inter` in `_PHASE` until `uncut["max_len"]` exceeds 1 MiB, and record the dims used.

- [ ] **Step 2: Run the verification on divix01** (SYNC first):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-iopoll-cuts && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  IOPOLL_CUTS_DIR=/mnt/nvme4/nvfp4-work/iopoll-cuts-tests CUDA_HOME=/usr/local/cuda-13.4 \
  && mkdir -p $IOPOLL_CUTS_DIR \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/manual/dsv41/test_expert_stream_read_cuts_nvme.py \
     test/registered/unit/kernels/test_expert_stream_uring_native.py \
     test/registered/unit/kernels/test_expert_stream_uring_integration.py \
     test/registered/unit/kernels/test_expert_stream_read_cuts.py \
     -q -rs -p no:randomly --basetemp=/mnt/nvme2/nvfp4-work/iopoll-cuts-native 2>&1 | tail -15; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: EXIT=0, with every manual test passing and none skipped. The punt check shows `workers > 0` uncut and `workers == 0` cut. Also run SUITE and record its counts against Task 0.

- [ ] **Step 3: Mutants** in a private worktree at this commit:

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cuts-mutant origin/cc/iopoll-read-cuts'
```

For each mutant: apply it, run its detecting tests (the same commands as Step 2, but in `wt-cuts-mutant`), record the result, `git checkout -- <file>`, and re-run green.

| # | Mutant (file: change) | Must fail |
|---|---|---|
| M1 | `read_cuts.h` `cut_legs`: `const bool gap = false;` (skip the gap cut) | `test_read_cuts_planner_and_limits`, `test_gap_cuts_at_slab_rows_off_the_page` |
| M2 | `read_cuts.h` `cut_legs`: room `lim.cut_bytes + 512 - leg.bytes` (cut one block past the limit) | `test_read_cuts_planner_and_limits`, `test_cut_reads_are_byte_identical_and_within_the_cut`, the punt check (`workers > 0` with cuts on) |
| M3 | `uring_options.h`: `blocking_wait()` returns `wait_mode == UringWaitMode::Block` (the trap restored) | `test_uring_reader_driver_contract` |
| M4 | `reader_core.h` `queue_depth()`: `per_read = 1` (depth not scaled) | `test_credit_scales_with_the_leg_bound` |
| M5 | `reader_core.h` `size_legs()`: `want = iovecs` whatever `cuts_` (storage not sized) | the `logic_error` "more legs than open() sized" in `test_cut_reads_are_byte_identical_and_within_the_cut` |
| M6 | `read_cuts.h` `device_limits`: return `DeviceLimits{INT64_MAX / 2, 0, "fallback"}` on failure (cuts silently off) | `test_read_cuts_planner_and_limits` |
| M7 | `reader_core.h` `process`: retire as soon as `d.legs_inflight == 0` without checking every leg Done | `test_one_short_cut_leg_resubmits_only_that_leg`, `test_a_held_cut_leg_keeps_its_read_unretired_and_its_pieces_unpublished` |

Remove the worktree afterwards: `git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-cuts-mutant`.

- [ ] **Step 4: Record and commit.** Record Step 2's counts and the mutant table with its results (killed / survived, plus the restored-green count) in the commit message body. If any mutant survives, stop and add the missing test before going on.

```bash
git add test/manual/dsv41/test_expert_stream_read_cuts_nvme.py
git commit -m "test(expert-stream): divix01 read-cut limits and the IOPOLL io-wq punt check

<Step 2 counts and command; the M1-M7 table with results>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

---

### Task 7: Decode arms on divix01 (one commit)

**Files:**
- Create: `analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh`, `analysis/dsv41-drive/iopoll-cuts/thread_sampler.py`, `analysis/dsv41-drive/iopoll-cuts/results.md`
- Modify then revert: `benchmarks/dsv41_baseline/generations.json` (arm-only registration)

**Interfaces:** Consumes `drive_uring_reg_arms.sh` (master) as the template, plus `run_arm.sh`, `generations.register`, `mirror3_report.py` (`decode`, `identity`, `ram_miss`, `timed_window_utc`, `clock_summary`, `session_clocks`, `node0_gate`).

**What this measures, stated up front.**
- **The regression.** The +10.2 ms/token IOPOLL regression should vanish in C. Prediction: C within ±1 ms/token of A (diagnosis: row p50 1609 µs with IOPOLL + reap vs 1687 µs default).
- **Cuts in default mode (B).** B's microbenchmark gain is −40 µs per row. Its decode effect is bounded above by ~1.8 ms/token and is probably smaller. The expected verdict is "no promotion".
- **SQPOLL (D).** D costs a full SQ core (~98% system time), per Codex S5–S7.

**Arms** (all ten `SGLANG_EXPERT_STREAM_URING_*` set explicitly, as `run_arm.sh` `KEY=VAL` overrides, in this order):

| Arm | Overrides |
|---|---|
| `A` | `MODE=default READ_CUTS=0 WAIT_MODE=block QUEUE_DEPTH=0 FIXED_FILES=0 READ_MODE=normal SQ_THREAD_IDLE_MS=10000 SQ_THREAD_CPU=-1 SLAB_ARENA=0 DIAGNOSTICS=1` |
| `B` | as A, `READ_CUTS=1` |
| `C` | as B, `MODE=iopoll` (WAIT_MODE=block, which Task 2 turns into the reap loop) |
| `D` (optional; open decision 1) | as B, `MODE=sqpoll_iopoll SQ_THREAD_CPU=20` |
| `A2` | as A (drift control: the pass is bracketed by two baselines) |

A runs this commit with cuts off in default mode. Tasks 1-5's golden and suites show that this prepares master's SQE set and writes master's bytes, so A stands for "master default" without a second worktree. That matches the uring-reg precedent of one commit per pass.

**Budget and repeats (recorded decision 4).**
- **One pass**, A, B, C, (D), A2: about 5 min per arm, so ~25 min. Both limits of one pass are stated in `results.md`: A2 bounds drift, but one pair per arm cannot separate a 1 ms effect from noise.
- **A confirmation** of 3 alternating pairs of that arm vs A (~30 min more) runs **only if** an arm's Δ vs mean(A, A2) is ≤ −1.0 ms/token. That is the only case where one pass cannot decide the 1.5 ms bar.
- **No confirmation** when every Δ is > −1.0 (the expected case) or when A and A2 differ by more than 1.0. In the second case the pass is invalid; re-run it once.

**Promotion rule.** An arm is promoted only if Δ ≤ −1.5 ms/token vs mean(A, A2), with identical output to A, and (if run) the confirmation's mean Δ ≤ −1.5. Promotion means:
- **B wins:** a follow-up flips `READ_CUTS` to default `1` (one line, plus its options-test assertion). It is recorded as a user decision, not done here.
- **C wins:** a recipe change to `MODE=iopoll`, which is the user's call.
- **Otherwise:** `READ_CUTS` stays `auto`, so it stays on with IOPOLL, where it is required, and the recipe keeps `MODE=default`.

- [ ] **Step 1: Register the tree for the arms.** Run `generations.register('$(git rev-parse HEAD:python)', 'iopoll-read-cuts')`, then commit (`bench(dsv41): register the iopoll-read-cuts tree for the A/B/C arms`, with the trailers). Push and SYNC. On divix01, confirm that HEAD is the pushed commit and the worktree is clean.

- [ ] **Step 2: Write `thread_sampler.py`**:

```python
"""Per-thread CPU of a server process tree, every `period` s, as JSON lines (plan 2026-09-28-iopoll-read-cuts Task 7).
Usage: thread_sampler.py <root pid> <out.jsonl> [period]. `report(samples, start, end)` sums, over [start, end]:
io-wq workers (comm iou-wrk*: distinct tids, max live, CPU s), the SQ thread (iou-sqp*), and the RAM-miss service
thread (comm exl3-ram-miss)."""

import datetime
import json
import os
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")


def tree(root):
    children = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(pid))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += children.get(p, [])
    return out


def sample(root):
    rows = []
    for pid in tree(root):
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
                f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            rows.append({"pid": pid, "tid": int(tid), "comm": comm, "state": f[0],
                         "cpu_s": (int(f[11]) + int(f[12])) / TICK})
    return rows


def main(root, out, period=2.0):
    with open(out, "a") as fh:
        while os.path.exists(f"/proc/{root}"):
            fh.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "tasks": sample(root)}) + "\n")
            fh.flush()
            time.sleep(period)


def report(path, start, end):
    """start/end: aware datetimes (mirror3_report.timed_window_utc)."""
    rows = [json.loads(line) for line in open(path)]
    inside = [r for r in rows if start <= datetime.datetime.fromisoformat(r["utc"]) <= end]
    if len(inside) < 2:
        return {"samples": len(inside)}

    def cpu(pred):
        first = {t["tid"]: t["cpu_s"] for t in inside[0]["tasks"] if pred(t["comm"])}
        total = 0.0
        for t in inside[-1]["tasks"]:
            if pred(t["comm"]):
                total += t["cpu_s"] - first.get(t["tid"], 0.0)  # a thread born inside counts from 0
        return round(total, 2)

    wrk = lambda c: c.startswith("iou-wrk")
    tids = {t["tid"] for r in inside for t in r["tasks"] if wrk(t["comm"])}
    return {
        "samples": len(inside),
        "window_s": (datetime.datetime.fromisoformat(inside[-1]["utc"])
                     - datetime.datetime.fromisoformat(inside[0]["utc"])).total_seconds(),
        "iowq_distinct": len(tids),
        "iowq_max_live": max(sum(1 for t in r["tasks"] if wrk(t["comm"])) for r in inside),
        "iowq_cpu_s": cpu(wrk),
        "sq_thread_cpu_s": cpu(lambda c: c.startswith("iou-sqp")),
        "ram_miss_cpu_s": cpu(lambda c: c == "exl3-ram-miss"),
    }


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
```

- [ ] **Step 3: Write the driver.** Copy `analysis/dsv41-drive/uring-reg/drive_uring_reg_arms.sh` to `analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh`. Keep every guard unchanged:
- the NVIDIA driver version and the port checks;
- `check_worktree` with the `generations` gate;
- `exl3_gate` before every arm;
- the exe-name foreign-process gate and `wait_for_quiet_box`;
- the disk lock taken inside the driver (`exec 8>`/`flock 8`), never under an outer `flock`;
- the tier gates: full `0:61440,1:40960` if node 0 ≥ share+4096+15360 and node 1 ≥ share+4096+10000, else `0:51200,1:40960`. The tier is settled before the first arm and applied to every arm;
- the PID tracking and cleanup, the memory samples and the startup sampler;
- `read_errors == 0`.

Change only the following:

```bash
URING_A=(
    SGLANG_EXPERT_STREAM_URING_MODE=default SGLANG_EXPERT_STREAM_URING_READ_CUTS=0
    SGLANG_EXPERT_STREAM_URING_WAIT_MODE=block SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=0
    SGLANG_EXPERT_STREAM_URING_FIXED_FILES=0 SGLANG_EXPERT_STREAM_URING_READ_MODE=normal
    SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS=10000 SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU=-1
    SGLANG_EXPERT_STREAM_URING_SLAB_ARENA=0 SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS=1
)
with_() { local out=() kv k; for kv in "${URING_A[@]}"; do k=${kv%%=*}; for o in "$@"; do [ "${o%%=*}" = "$k" ] && kv=$o; done; out+=("$kv"); done; echo "${out[@]}"; }
URING_B=($(with_ SGLANG_EXPERT_STREAM_URING_READ_CUTS=1))
URING_C=($(with_ SGLANG_EXPERT_STREAM_URING_READ_CUTS=1 SGLANG_EXPERT_STREAM_URING_MODE=iopoll))
URING_D=($(with_ SGLANG_EXPERT_STREAM_URING_READ_CUTS=1 SGLANG_EXPERT_STREAM_URING_MODE=sqpoll_iopoll \
                 SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU=20))
ARMS=(A B C ${RUN_D:+D} A2)   # RUN_D=1 adds D (open decision 1)
FIRST_ARM=A
```

`check_modes` checks the `expert stream io_uring: mode=` line and the per-drive `read cuts:` lines:

```bash
check_modes() {  # <arm> <run_dir>
    local arm=$1 run=$2 line cuts
    line=$(grep -m1 'expert stream io_uring: mode=' "$run/server.log")
    [ -n "$line" ] || { say "arm $arm: no io_uring mode line"; return 1; }
    cuts=$(grep -c 'expert stream io_uring: read cuts: drive=' "$run/server.log")
    case $arm in
        A|A2) [[ $line == *"mode=default "* && $line == *"read_cuts=off:0"* && $line == *"effective_wait=block"* ]] && [ "$cuts" = 0 ] ;;
        B) [[ $line == *"mode=default "* && $line == *"read_cuts=on:1"* && $line == *"effective_wait=block"* ]] && [ "$cuts" = 3 ] ;;
        C) [[ $line == *"mode=iopoll "* && $line == *"read_cuts=on:1"* && $line == *"effective_wait=reap"* ]] && [ "$cuts" = 3 ] ;;
        D) [[ $line == *"mode=sqpoll_iopoll "* && $line == *"read_cuts=on:1"* && $line == *"sq_thread_idle_ms=10000"* ]] && [ "$cuts" = 3 ] ;;
    esac || { say "arm $arm: io_uring lines do not show the arm's settings: $line (read cuts lines: $cuts)"; return 1; }
    # Review Focus 5: each drive is cut at its own limit (SPCC 262144, Samsungs 520192).
    if [ "$cuts" = 3 ]; then
        grep 'read cuts: drive=' "$run/server.log" | grep -q 'cut_bytes=262144' \
          && [ "$(grep 'read cuts: drive=' "$run/server.log" | grep -c 'cut_bytes=520192')" = 2 ] \
          || { say "arm $arm: per-drive cut_bytes are not 262144 + 2 x 520192"; return 1; }
    fi
}
```

`uring_lines` keeps every `expert stream io_uring:` line, including `read cuts:`, in `$OUT/<arm>-uring.txt`. It drops the `fixed_reads line` check (no fixed modes here).

The thread sampler starts once `run_arm.sh` logs `server pid=N`, and runs until the arm ends. Place it in `run_one`'s watch loop next to the memory sample:

```bash
        if [ -z "${tpid:-}" ]; then
            spid=$(sed -n 's/^server pid=\([0-9]*\).*/\1/p' "$OUT/$arm-run_arm.log" 2>/dev/null | head -1)
            if [ -n "$spid" ]; then
                taskset -c 20-23 $PY "$WT/analysis/dsv41-drive/iopoll-cuts/thread_sampler.py" "$spid" "$OUT/$arm-threads.jsonl" 2 \
                    < /dev/null 8>&- &
                tpid=$!; track "$tpid" "$arm thread sampler"
            fi
        fi
```

Reset `tpid=` at the start of `run_one`, and call `stop_pid "$tpid"` after `wait "$ARM_PID"`.

The arm loop replaces the R0/R3 pair:

```bash
for arm in "${ARMS[@]}"; do
    case $arm in A|A2) set -- "${URING_A[@]}" ;; B) set -- "${URING_B[@]}" ;; C) set -- "${URING_C[@]}" ;; D) set -- "${URING_D[@]}" ;; esac
    run_one "$arm" "$@" || { say "arm $arm failed; stopping the pass"; rc=1; break; }
done
```

The report computes the following with `mirror3_report` plus `thread_sampler.report(path, *m.timed_window_utc(run))`, and writes `$OUT/arms-report.json`:
- per arm: `decode`, TTFT, `ram_miss`, the clock summaries and the thread report;
- `identity(A_run, X_run)` for every X;
- `baseline = mean(A, A2)` pooled ms/token, and each arm's Δ against it;
- `drift = A2 − A`.

It exits non-zero if any output differs from A's.

Commit and push:

```bash
git add analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh analysis/dsv41-drive/iopoll-cuts/thread_sampler.py
git commit -m "bench(dsv41): iopoll-read-cuts decode arms A/B/C/(D)/A2 with a per-thread io-wq sampler

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
```

Then SYNC.

- [ ] **Step 4: Launch.** Use no outer flock: the driver takes the disk lock itself, and `run_arm.sh` takes the GPU lock.

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-iopoll-cuts && mkdir -p /mnt/nvme1/iopoll-cuts \
  && OMP_NUM_THREADS=8 setsid nohup taskset -c 0-63 bash analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh \
     $PWD $(git rev-parse HEAD) /mnt/nvme1/iopoll-cuts 30031 > /mnt/nvme1/iopoll-cuts/driver.log 2>&1 < /dev/null &'
```

Poll `driver.log` no more often than every 10 minutes. Start no other disk or GPU work until `DRIVER DONE`. If the confirmation rule fires, run it as a second pass with `ARMS=(A X A X A X)` (edit the array on the laptop, commit, push, SYNC; never on divix01) into `/mnt/nvme1/iopoll-cuts-confirm`.

- [ ] **Step 5: Write `results.md`**, in the shape of `analysis/dsv41-drive/uring-reg/results.md`:
- the verdict first;
- commit, tree and generation label; host facts; tier and node gates;
- per arm: session and pooled ms/token, TTFT, Δ vs mean(A, A2), byte identity vs A, `rows_read/served/read_errors`, startup to "fired up";
- per arm, the io_uring and read-cuts lines, and the thread report (io-wq distinct/max live/CPU s, SQ thread CPU, RAM-miss thread CPU), set against the Codex S4 (12 workers, 4.3 s) and S7 (6 workers) figures;
- drift (A2 − A), the promotion rule and its outcome, and the limits of one pass;
- the open item: the SPCC slow episodes (diagnosis.md, "System"), not in scope.

- [ ] **Step 6: Revert the arm-only registration** and commit the results:

```bash
git revert --no-edit <the Step 1 commit>   # a new commit; never amend or rebase
git add analysis/dsv41-drive/iopoll-cuts/results.md
git commit -m "analysis(iopoll-cuts): decode arms A/B/C/(D)/A2 results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R"
git push origin cc/iopoll-read-cuts
```

The revert's message must also end with both trailers: use `git revert --no-commit`, then `git commit` with the message `Revert the arm-only generations.json registration (<sha>)` and the trailers.

---

## Self-review notes (for the executor)

- **Spec coverage:**
  - Wait trap in `submit` and `wait_one`, with the internal-reap decision justified: Task 2.
  - Read cuts at the sysfs `max_sectors_kb` (with `max_segments`) and at virt-boundary gaps, with the fallback and its log line: Tasks 3 and 5.
  - Reuse of the fan-out leg machinery: Task 5's `plan_legs`, plus the four leg tests.
  - Knob default-on or opt-in: recorded decision 5 and Task 7's promotion rule.
  - Depth resizing and the admission check: Task 4, recorded decision 3.
  - Idle default 10000: Task 1 (option test) and Task 2 (the fake ring receives 10000).
  - Native cut tests at a lowered test cap, gap detection, golden and byte identity: Tasks 3-5.
  - Punt check: Task 6.
  - Mutants (gap skip, cut past the limit, and five more): Task 6.
  - Decode arms with the EXL3 gate, the tier gates and the `generations.json` revert: Task 7.
  - SPCC as an open item: Task 7 `results.md` and the open decisions below.
- **Names used across tasks:**
  - Options: `UringReadCuts`, `UringOptions::{read_cuts, read_cuts_on, read_cuts_name, polls_in_wait, blocking_wait, effective_wait_name}`, and `UringReader::read_cuts`.
  - `read_cuts.h`: `DeviceLimits`, `cut_bytes_for`, `limits_from_queue_dir`, `queue_dir_for`, `device_limits`, `CutLeg`, `cut_legs`, `leg_bound`, `kCutPage`, `kFallbackCutBytes`, `kFallbackVirtMask`.
  - `ReaderCore`: `size_legs`, `longest_read`, `limits_`, `cuts_requested_`, `cuts_`, `leg_cut_cap_`, `leg_stride_`, `iov_stride_`, `iov_scratch_`, `cut_scratch_`, `fixed_scratch_`, `kMaxLegs = 255`, `set_leg_cut_cap`, `cut_reads`, `gap_cuts`, `min_cut_bytes`, `leg_stride`.
  - Fault word 31 `leg_cut_cap` (`kFaultWords = 32`).
  - Info keys `cut_reads`, `gap_cuts`, `min_cut_bytes`, `leg_stride`; stats keys `cut_reads`, `gap_cuts`.
- **Placeholders:** the `<session>` path segment and `<the Step 1 commit>` are values known only at execution time. Nothing else is deferred.

## Decisions recorded (2026-09-28)

1. **Wait trap:** an internal reap, not a `from_env` refusal (Task 2 gives the reasoning). `WAIT_MODE=block` on `MODE=iopoll` reports `effective_wait=reap`.
2. **Cut rule:**
   - Size: `min(max_sectors_kb · 1024, (max_segments − 1) · 4096)`, rounded down to whole pages.
   - Gaps: a new leg at every join off `virt_boundary_mask + 1`.
   - Fallback: 128 KiB on a 4 KiB boundary, with one log line per drive.
   - Limits apply per file.
3. **Depth:** the default credit is `16 · parts · leg_stride` with cuts on (at most 32768). An explicit `QUEUE_DEPTH` below `leg_stride` is refused at `open()`. `kMaxLegs` rises from 16 to 255 (the tag's leg field), and a read needing more is refused at `open()`.
4. **Arms:** one pass A, B, C, (D), A2. A confirmation runs only when an arm lands at Δ ≤ −1.0 ms/token.
5. **Knob default:** `READ_CUTS=auto`, which is on exactly under IOPOLL, where an uncut read is always punted. Default-on in every mode waits for Task 7's B arm to clear the 1.5 ms/token bar, per the user's wish to measure first.

## Open decisions for the user

1. **Arm D (`sqpoll_iopoll`):** run it or not? It measures the 10 s idle default and cuts under SQPOLL, but it costs a full SQ core in any deployment. Recommendation: skip it unless you would consider SQPOLL in the recipe.
2. **If B clears the bar:** pre-approve flipping `READ_CUTS` to default `1` in a follow-up, or decide after reading `results.md`?
3. **Env-var registry:** the ten `SGLANG_EXPERT_STREAM_URING_*` knobs, now including `READ_CUTS`, are read in C++ and are absent from `environ.py`. Should a follow-up register them in `Envs` as documentation (the Python side would only pass them through)?
4. **SPCC drive (nvme2n1, `/mnt/nvme4`):** its slow episodes (per-read p50 10–35 ms in 7 of 24 multi-row runs) gate row latency. Not in scope. Options: `nvme smart-log /dev/nvme2` (root), or moving that mirror to a Samsung-class drive.
5. **Execution method:** subagent-driven or native.
