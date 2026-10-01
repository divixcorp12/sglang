# Expert-Stream Record Read Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RAM-miss service's read of a demand record (`pump_demand` → `read_record` → `read_gpu_hot`) cost one
overlapped batch of L3 round trips instead of three or four serial ones. On a dedicated physical core, the service
thread busy-polls the request page with no PAUSE.

**Architecture:** The GPU's posted writes land in the socket-0 L3 through DDIO, so every line the service reads per
request is a fresh L3 miss.
- Findings 1-2 start all of those misses at once:
  - a software prefetch of the record and hot-record lines as soon as the head shows the post;
  - one constant-size 128-byte copy of the record that depends on nothing it contains.
- Finding 3 places the record's two lines in one 128-byte-aligned block, so the L2's adjacent-line prefetcher fetches
  the pair together.
- Finding 4 (option B) removes the ~45 ns PAUSE quantum from detection. The service thread spins without PAUSE or
  sleep on a physical core that no other thread may use.

**Tech Stack:** C++20 host module (`-O3`; g++ 14.3 on divix01 defaults to `-march=x86-64-v3`), CUDA device headers,
Python/pytest, Linux `perf` (user-mode core events; uncore events need root).

**Spec:** None separate. The "Decisions" section below is the spec, taken from the cache-line analysis of 2026-10-01
(divix01: 2 × Xeon Gold 6154 Skylake-SP, GPU `37:00.0` on NUMA node 0, production server cores `0-7,16-17,36-53`).
Rulings made during execution are provisional against it.

**Base:** The head of `expert-stream-record-narrow` once its plan
(`2026-10-01-expert-stream-record-narrow.md`) has finished, including its final review. This plan needs that branch's
narrow record layout (i16 ids, `kRecLaneWeight = 96`).

## Decisions

- **D1 (finding 1): prefetch every line of the request at once.**
  - `pump_demand` issues `_mm_prefetch(..., _MM_HINT_T0)` for the record's two lines and for every line of the
    request's hot record.
  - It does so right after it computes the record's address, so all three or four L3 misses overlap.
  - Today the hot-record line is read only after `read_record` returns, which makes it serial.
- **D2 (finding 2): `read_record` copies the whole record with one constant-size `memcpy`.**
  - The copy (`kRecordBytes` = 128 bytes) sits between the two seq loads, then the reader decodes from the local
    copy.
  - The variable-length copies of af5b31c5d2 are removed. They made every line-1 load depend on the counts byte
    through a ladder of about 15 branches.
  - Decoding writes all 8 protect ids and all 8 lanes unconditionally, then sets each list's size once
    (`FixedVec::resize`).
  - Only the first `count` lane kinds are validated.
  - A protect count above `kMaxIds` becomes malformed, like a lane count above `kLeaseLanes`. Today it is silently
    read as 8 ids, and the device never writes one.
- **D3 (finding 3): one record is one 128-byte prefetch pair.**
  - `kDemandRing` 64 → 128 and `kRecordBytes` 256 → 128, so `kPageBytes` 4160 → 2176. The ring then fits one 4 KiB
    page.
  - The post launcher's page-alignment refusal tightens from 16 to 128 bytes. This is a performance requirement:
    it only matters for the pairing.
  - Unpinned CPU pages in CPU-only tests are 64-byte aligned (the torch CPU allocator). Nothing on the host side
    refuses them, because the pairing does not change correctness.
- **D4 (finding 4, option B): `start_thread(cpu_core=N, busy_poll=True)`.**
  - The service loop never PAUSEs and never sleeps, except while parked by `pause()`.
  - The C++ `start_thread` refuses `busy_poll` when:
    - `cpu_core < 0`;
    - any SMT sibling of `cpu_core` (the core included, from sysfs `thread_siblings_list`) is in the caller's affinity
      mask, which the process's other threads inherit;
    - any such sibling is among the CPU experts' cores.
  - The refusals live in C++, as the owner asked for record-narrow's refusals.
  - Production setting: `SGLANG_DSV41_RAM_MISS_SPIN_CORE` (`EnvInt(None)`). Unset keeps today's behaviour; set, the
    service starts with `cpu_core=<value>, busy_poll=True`.
  - The divix01 arm recipe:
    - `SERVER_CORES` becomes `"0-7,16,36-52"`;
    - `SPIN_CORE = 17` (its sibling is 53);
    - the env carries `SGLANG_DSV41_RAM_MISS_SPIN_CORE=17`.
  - This makes arms run after the change not comparable to earlier cells: the server loses physical core 17. Record
    that next to any arm number.

## Global Constraints

- divix01 run protocol (`.claude/rules/divix01-run-protocol.md`):
  - Commit, `git push origin expert-stream-read-record`, and run in a private worktree under
    `/data/models/slang/nvfp4-work/wt-read-record-*`.
  - Set `PYTHONPATH=$PWD/python` (that tree's `python/`).
  - CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS=8`.
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock`.
  - Read `${PIPESTATUS[0]}` for any piped pytest.
- Registered suite command (record its counts with it):
  `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly`
- GPU tests command, from `test/manual/dsv41`:
  `PYTHONPATH=$W/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest test_exl3_lease_kernels_cuda.py test_exl3_slot_map_kernels_cuda.py -q -p no:randomly`
- **Benchmark command** (`$W` = the worktree), from `$W/test/manual/dsv41`:
  `PYTHONPATH=$W/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 36-52 /data/models/slang/.venv/bin/python bench_exl3_service_read.py --cpu-core 17 --perf`
  - Add `--busy-poll` from Task 6 on.
  - `36-52` keeps the Python process on node 0 and off core 17's sibling 53. It lies inside the GPU convention's
    `32-63`.
  - Run it only when no registered suite runs on the box (`pgrep -f "pytest test/registered"` prints nothing). The
    seqlock and timing numbers are load-sensitive.
- Every constant in `lease_layout.h` is mirrored in `python/sglang/kernels/ops/moe/expert_stream_transport.py`.
  `test_exl3_ram_miss_device_args.py`'s parity test fails on any mismatch.
- Every new test-only C++ export must be added to:
  - `TEST_ONLY_EXPORTS` (`expert_stream_transport.py:73`);
  - `RAW_EXPORTS` and the module-level helper list in `test/registered/unit/kernels/test_expert_stream_build_variants.py`.
- Edit C++ and Python sources directly, not through scripts that rewrite files (owner's instruction).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never amend, rebase or force-push.
- The owner runs commands that need root (Task 7). Hand them over verbatim and wait for their output.

## Review Focus

1. **The CPU-experts check has no test.** A busy-poll core shared with CPU expert cores
   (`SGLANG_DSV41_CPU_EXPERTS_CORES`) must be refused (D4), but Task 6's tests do not enable CPU experts. The
   reviewer checks that `check_dedicated_core` is passed `RamTier::cpu_cores()`, and that `_start_cpu_experts` runs
   before `start_thread` in `exl3_ram_miss.py`.
2. **The production wiring has no unit test.** `SGLANG_DSV41_RAM_MISS_SPIN_CORE` unset must start the thread exactly
   as before (no `cpu_core`, no `busy_poll`); set, it must pass both. The reviewer reads `exl3_ram_miss.py`'s
   `start_thread` call against this.
3. **No field may be read from the shared record after its copy.** After D2, nothing past the copy may read the
   shared `record` except the second seq load. A field read from the page after that check could see the next
   writer's bytes.
4. **Threads that pin themselves after start are not checked.** The dedicated-core check runs once, at
   `start_thread`. A thread that later pins itself onto 17 or 53 (the copy thread, a CPU expert helper, NCCL, or
   OpenMP with `OMP_PROC_BIND`) is not caught. The reviewer lists every `sched_setaffinity` / `pthread_setaffinity_np`
   in the server's code paths and confirms none can choose 17 or 53 under the arm recipe.
5. **An overrun after a lap must still be detected with 128-byte records.** The service resumes at `head - 14` after
   a lap. With 128-byte records the lapped slot's prefetch may fetch a line the device is rewriting; the read's seq
   check must still report it torn, never accept it.

---

### Task 0: Branch and worktree

**Files:** this plan (moved onto the new branch).

- [ ] **Step 1: Create the branch and worktree from the finished record-narrow head**

```bash
cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-wt-record-narrow
git fetch origin
git worktree add -b expert-stream-read-record ../sglang-nvfp4-wt-read-record origin/expert-stream-record-narrow
cp docs/superpowers/plans/2026-10-01-expert-stream-read-record.md ../sglang-nvfp4-wt-read-record/docs/superpowers/plans/
rm docs/superpowers/plans/2026-10-01-expert-stream-read-record.md   # it was never committed on record-narrow
cd ../sglang-nvfp4-wt-read-record
git add docs/superpowers/plans/2026-10-01-expert-stream-read-record.md
git commit -m "$(cat <<'EOF'
plan(expert-stream-read-record): the service reads a demand record in one batch of L3 round trips

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push -u origin expert-stream-read-record
```

Expected: `git log --oneline -2` shows the plan commit on top of record-narrow's last commit.

---

### Task 1: Benchmark of the service's record read, and the baseline

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (a `pause_ns` export)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`pause_ns` wrapper, `TEST_ONLY_EXPORTS`)
- Modify: `test/registered/unit/kernels/test_expert_stream_build_variants.py` (`RAW_EXPORTS`, helper list)
- Modify: `test/manual/dsv41/lease_chain_rig.py` (`Chain(..., gpu_hot=False)`)
- Create: `test/manual/dsv41/bench_exl3_service_read.py`

**Interfaces:**
- Produces:
  - `expert_stream_transport.pause_ns(*, layout="exl3", variant=None) -> float`: ns per `_mm_pause`, instr only.
  - `Chain(..., gpu_hot=True)`: a pinned hot page shared by the host and the device.
  - The bench CLI: `--posts N --warmup N --gap-us F --cpu-core N [--busy-poll] [--perf]`.

- [ ] **Step 1: The `pause_ns` export**

In `ffi_test_exports.h`, after `trace_clock_reads`:

```cpp
  // Test only: the measured cost of one _mm_pause in ns, the service's idle-poll quantum (RamThread::run).
  static double pause_ns() {
    if constexpr (!Build::kFaults) {
      test_only("pause_ns");
    } else {
      constexpr int kProbe = 1 << 16;
      const int64_t start = now_ns();
      for (int i = 0; i < kProbe; ++i)
        _mm_pause();
      return static_cast<double>(now_ns() - start) / kProbe;
    }
  }
```

and add to `EXPERT_STREAM_HOST_TEST_EXPORTS_OF`, after the `trace_clock_reads` line (remove that line's `;`, so the
new line ends the macro):

```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_clock_reads, Exports::trace_clock_reads);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pause_ns, Exports::pause_ns);
```

In `expert_stream_transport.py`, add `"pause_ns",` to `TEST_ONLY_EXPORTS`, and after `seqlock_stress`:

```python
def pause_ns(*, layout: str = "exl3", variant: Optional[str] = None) -> float:
    """Test only: ns per _mm_pause on this core, the RAM-miss service's idle-poll quantum. Instrumented build only."""
    _refuse_test_only("pause_ns", variant)
    return float(_host_module(layout, variant).expert_stream_pause_ns())
```

In `test_expert_stream_build_variants.py`, add to `RAW_EXPORTS`:

```python
    "pause_ns": lambda m, h: m.expert_stream_pause_ns(),
```

Then add `"pause_ns"` to the `helper` parametrize tuple of `test_the_module_level_test_only_helpers_refuse_on_prod`,
with `"pause_ns": lambda: ops.pause_ns(variant="prod"),` in its `calls` dict.

- [ ] **Step 2: `Chain(gpu_hot=True)`**

In `lease_chain_rig.py`:
- Add `new_hot_page` to the `expert_stream_transport` import.
- Add the keyword `gpu_hot=False` to `Chain.__init__`.
- Before `self.host = ExpertStreamHost(`, add:

```python
            self.hot_page = new_hot_page(EXPERTS, pin=True) if gpu_hot else None
```

Then pass `hot_page=self.hot_page` to both `ExpertStreamHost(...)` and `ExpertStreamDevice(...)`.

- [ ] **Step 3: The benchmark**

Create `test/manual/dsv41/bench_exl3_service_read.py`:

```python
"""The RAM-miss service's cost to serve one demand record, as its stage trace times it: from the moment the service
sees the post (observed) to the moment it has finished the record (done). Records carry SM-hit lanes only, with a GPU
hot record, so the service reads the record and its hot record and touches the slots: the read path, nothing else.

Each post is served before the next, and the service idles --gap-us between them, so every record is found from an
idle poll: production's decode steady state. --perf counts the service thread's user-mode events with perf stat -t.

Run on divix01 under cc-gpu.lock from test/manual/dsv41, PYTHONPATH at the tree under test (the plan's Global
Constraints have the command).
"""

import argparse
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from lease_chain_rig import TOP_K, Chain  # noqa: E402

import sglang  # noqa: E402
from sglang.kernels.ops.moe import expert_stream_transport  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

PERF_EVENTS = (
    "instructions:u",
    "br_misp_retired.all_branches:u",
    "machine_clears.memory_ordering:u",
    "mem_load_retired.l3_hit:u",
    "mem_load_l3_hit_retired.xsnp_none:u",
    "mem_load_l3_miss_retired.local_dram:u",
    "mem_load_l3_miss_retired.remote_dram:u",
)


def service_tid() -> int:
    for task in Path("/proc/self/task").iterdir():
        if (task / "comm").read_text().strip().endswith("ram-miss"):
            return int(task.name)
    raise RuntimeError("no RAM-miss service thread in this process")


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--gap-us", type=float, default=100.0)
    parser.add_argument("--cpu-core", type=int, default=-1)
    parser.add_argument("--busy-poll", action="store_true")
    parser.add_argument("--perf", action="store_true")
    args = parser.parse_args()
    print("sglang:", sglang.__file__)
    print(f"pause: {expert_stream_transport.pause_ns(variant='instr'):.1f} ns")
    with tempfile.TemporaryDirectory() as tmp:
        c = Chain(Path(tmp), start=False, gpu_hot=True)
        try:
            row = 0
            experts = list(range(TOP_K))
            c.dev.map_bulk_apply(torch.tensor([[row, e, e] for e in experts], dtype=torch.int32))
            c.host.enable_trace(capacity=args.posts + args.warmup + 64)
            busy = {"busy_poll": True} if args.busy_poll else {}
            c.host.start_thread(cpu_core=args.cpu_core, fatal_wait_s=60.0, **busy)
            print("service core:", c.host.counters()["spin_cpu"], "busy_poll:", args.busy_poll)
            c.plan(experts, row)
            backend, plan = c.backends[row], c.plans[row]
            backend._stage_planned(plan)
            hot_slots = torch.arange(TOP_K, dtype=torch.int64, device="cuda")

            def post_and_wait() -> None:
                c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, hot_slots=hot_slots,
                           hot_capacity=TOP_K)
                torch.cuda.synchronize()
                assert c.handled(timeout_s=5.0), "the service did not finish the record"
                time.sleep(args.gap_us * 1e-6)

            for _ in range(args.warmup):
                post_and_wait()
            assert c.kinds(TOP_K) == [LaneKind.HIT_SM] * TOP_K, c.kinds(TOP_K)
            c.host.drain_trace()
            overruns = c.host.counters()["overruns"]
            perf = perf_out = None
            if args.perf:
                perf_out = Path(tmp) / "perf.csv"
                perf = subprocess.Popen(["perf", "stat", "-x,", "-o", str(perf_out), "-t", str(service_tid()),
                                         "-e", ",".join(PERF_EVENTS)])
                time.sleep(0.5)
            for _ in range(args.posts):
                post_and_wait()
            if perf is not None:
                perf.send_signal(signal.SIGINT)
                perf.wait()
            spans = [r["done"] - r["observed"] for r in c.host.drain_trace()]
            assert len(spans) == args.posts, (len(spans), args.posts)
            print(f"span ns over {len(spans)}: p10 {percentile(spans, 0.1)} median {int(statistics.median(spans))} "
                  f"p90 {percentile(spans, 0.9)} p99 {percentile(spans, 0.99)}")
            print("overruns in the window:", c.host.counters()["overruns"] - overruns)
            if perf_out is not None:
                for line in perf_out.read_text().splitlines():
                    fields = line.split(",")
                    if line.startswith("#") or len(fields) < 3 or not fields[0].replace(".", "").isdigit():
                        continue
                    print(f"perf {fields[2]}: {float(fields[0]) / args.posts:.2f} per post")
        finally:
            c.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit, push, and run the build-variant tests**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py \
  test/registered/unit/kernels/test_expert_stream_build_variants.py test/manual/dsv41/lease_chain_rig.py \
  test/manual/dsv41/bench_exl3_service_read.py
git commit -m "$(cat <<'EOF'
bench(expert-stream-read-record): the service's span to serve one SM-hit record with a GPU hot record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

On divix01, create `wt-read-record-base` at this commit and run
`... -m pytest test/registered/unit/kernels/test_expert_stream_build_variants.py -q -p no:randomly`.
Expected: PASS, including the new `pause_ns` cases.

- [ ] **Step 5: The baseline**

In `wt-read-record-base`:
- Run the full registered suite. Record its counts as `base`.
- Run the GPU tests. Expected: EXIT=0.
- Run the benchmark command three times.

Expected:
- `pause:` about 40-50 ns: Skylake-SP's PAUSE is about 140 cycles.
- `service core: 17 busy_poll: False`.
- `overruns in the window: 0`.
- A span distribution.
- Per-post perf counts, with `mem_load_l3_miss_retired.local_dram` near 0 per post if DDIO lands the record lines
  in L3.

Write all three runs into "Results" under `base`. Keep `wt-read-record-base` for Task 7's root runs.

---

### Task 2: A test-only view of `read_record`, and characterization tests

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (`read_record_fields`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (wrapper, `TEST_ONLY_EXPORTS`)
- Modify: `test/registered/unit/kernels/test_expert_stream_build_variants.py`
- Create: `test/registered/unit/kernels/test_exl3_ram_miss_read_record.py`

**Interfaces:**
- Produces:
  - `expert_stream_transport.read_record_fields(record: torch.Tensor, expected: int, *, layout="exl3", variant=None) -> dict`
    with keys `status` (`"ok"|"torn"|"malformed"`), `row`, `captured`, `chain`, `gen`, `protect` (list), and
    `lanes` (list of dicts `expert, slot, dst, kind, weight`).
  - `READ_RECORD_WORDS = 55`.
  - The test file's `write_record(...)` helper, used by Task 4.

- [ ] **Step 1: The export**

In `ffi_test_exports.h`, after `seqlock_stress`:

```cpp
  // Test only: read_record over one record (record: CPU uint8 [kRecordBytes]) as the service reads seq `expected`.
  // out int64 [6 + kMaxIds + 1 + 5 * kMaxIds] = {status (RecordRead: 0 ok, 1 torn, 2 malformed), row, captured,
  // chain, gen, protect count, protect ids, lane count, then per lane: expert, slot, dst, kind, the weight's bits}.
  static void read_record_fields(TensorView record, int64_t expected, TensorView out) {
    if constexpr (!Build::kFaults) {
      test_only("read_record_fields");
    } else {
      {
        using namespace host;
        auto cpu = SymbolicDevice{};
        expert_stream::verify_named(
            "record", TensorMatcher({kRecordBytes}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), record);
        expert_stream::verify_named(
            "out", TensorMatcher({6 + kMaxIds + 1 + 5 * kMaxIds}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
      }
      Request request;
      const RecordRead read =
          read_record(static_cast<const uint8_t*>(record.data_ptr()), static_cast<uint32_t>(expected), &request);
      auto* w = static_cast<int64_t*>(out.data_ptr());
      w[0] = static_cast<int64_t>(read);
      w[1] = request.row;
      w[2] = request.captured ? 1 : 0;
      w[3] = static_cast<int64_t>(request.chain);
      w[4] = static_cast<int64_t>(request.gen);
      w[5] = static_cast<int64_t>(request.protect.size());
      for (size_t i = 0; i < request.protect.size(); ++i)
        w[6 + i] = request.protect[i];
      w[6 + kMaxIds] = static_cast<int64_t>(request.lanes.size());
      for (size_t j = 0; j < request.lanes.size(); ++j) {
        const Lane& lane = request.lanes[j];
        int32_t bits;
        std::memcpy(&bits, &lane.weight, 4);
        int64_t* l = w + 7 + kMaxIds + 5 * j;
        l[0] = lane.expert;
        l[1] = lane.slot;
        l[2] = lane.dst;
        l[3] = lane.kind;
        l[4] = bits;
      }
    }
  }
```

Add to the macro, after `expert_stream_seqlock_stress`:

```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_record_fields, Exports::read_record_fields);         \
```

In `expert_stream_transport.py`, add `"read_record_fields",` to `TEST_ONLY_EXPORTS`, and after `seqlock_stress`:

```python
_RECORD_LANES = 8  # kMaxIds == kLeaseLanes
READ_RECORD_WORDS = 6 + _RECORD_LANES + 1 + 5 * _RECORD_LANES


def read_record_fields(record: torch.Tensor, expected: int, *, layout: str = "exl3",
                       variant: Optional[str] = None) -> dict:
    """Test only: the service's read_record over one RECORD_BYTES record, expecting seq ``expected``. Instrumented
    build only."""
    _refuse_test_only("read_record_fields", variant)
    out = torch.zeros(READ_RECORD_WORDS, dtype=torch.int64)
    _host_module(layout, variant).expert_stream_read_record_fields(record, int(expected), out)
    w = out.tolist()
    protect, lanes = w[5], w[6 + _RECORD_LANES]
    base = 7 + _RECORD_LANES
    return {
        "status": ("ok", "torn", "malformed")[w[0]],
        "row": w[1],
        "captured": bool(w[2]),
        "chain": w[3] & 0xFFFFFFFFFFFFFFFF,
        "gen": w[4] & 0xFFFFFFFFFFFFFFFF,
        "protect": w[6 : 6 + protect],
        "lanes": [
            {
                "expert": w[base + 5 * j],
                "slot": w[base + 5 * j + 1],
                "dst": w[base + 5 * j + 2],
                "kind": w[base + 5 * j + 3],
                "weight": struct.unpack("<f", struct.pack("<i", w[base + 5 * j + 4]))[0],
            }
            for j in range(lanes)
        ],
    }
```

Add `import struct` to the module's imports if it is not there. In `test_expert_stream_build_variants.py`:
- Add to `RAW_EXPORTS`:
  `"read_record_fields": lambda m, h: m.expert_stream_read_record_fields(torch.zeros(ops.RECORD_BYTES, dtype=torch.uint8), 1, torch.zeros(ops.READ_RECORD_WORDS, dtype=torch.int64)),`
- Add `"read_record_fields"` to the module-level helper tuple, with
  `"read_record_fields": lambda: ops.read_record_fields(torch.zeros(ops.RECORD_BYTES, dtype=torch.uint8), 1, variant="prod"),`.

- [ ] **Step 2: The characterization tests**

Create `test/registered/unit/kernels/test_exl3_ram_miss_read_record.py`:

```python
"""The service's record reader against records written byte by byte (CPU): every field round-trips, the kinds of
unused lanes are never judged, and a torn or malformed record is reported as such. read_record_fields is the
instrumented build's view of read_record (host/tier_protocol.h)."""

import random
import struct

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import (
    RECORD_BYTES,
    RECORD_FIELDS,
    RECORD_FLAG_CAPTURED,
    RECORD_ID_MAX,
)
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LANES = 8
KINDS = [int(k) for k in (LaneKind.HIT_COPY, LaneKind.HIT_SM, LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.MISS_CPU)]


def _put(record: torch.Tensor, field: str, fmt: str, *values) -> None:
    data = struct.pack("<" + fmt, *values)
    at = RECORD_FIELDS[field]
    record[at : at + len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8)


def write_record(*, seq, row=0, captured=False, chain=0, epoch=0, protect=(), lanes=(), kinds=None, counts=None):
    """One record as the post kernel's write_record lays it out. lanes: (expert, slot, dst, weight, kind) tuples;
    ``kinds`` and ``counts`` override the packed words (a record the device never writes)."""
    record = torch.zeros(RECORD_BYTES, dtype=torch.uint8)
    if kinds is None:
        kinds = sum(lane[4] << (4 * j) for j, lane in enumerate(lanes))
    if counts is None:
        counts = len(lanes) | len(protect) << 4

    def ids(values):
        return list(values) + [-1] * (LANES - len(values))

    _put(record, "seq", "I", seq)
    _put(record, "row", "H", row)
    _put(record, "counts", "B", counts)
    _put(record, "flags", "B", RECORD_FLAG_CAPTURED if captured else 0)
    _put(record, "chain", "Q", chain)
    _put(record, "epoch", "I", epoch)
    _put(record, "kinds", "I", kinds)
    _put(record, "protect", "8h", *ids(protect))
    _put(record, "lane_expert", "8h", *ids([lane[0] for lane in lanes]))
    _put(record, "lane_slot", "8h", *ids([lane[1] for lane in lanes]))
    _put(record, "lane_dst", "8h", *ids([lane[2] for lane in lanes]))
    _put(record, "lane_weight", "8f", *([lane[3] for lane in lanes] + [0.0] * (LANES - len(lanes))))
    return record


def read(record, expected):
    return ops.read_record_fields(record, expected, variant="instr")


@pytest.mark.parametrize("lane_count, protect_count", [(0, 0), (1, 8), (6, 6), (8, 0), (8, 8)])
def test_every_field_of_a_whole_record_round_trips(lane_count, protect_count):
    rng = random.Random(9 * lane_count + protect_count)

    def ids(n):
        return rng.sample([0, 1, RECORD_ID_MAX] + list(range(2, 64)), n)

    lanes = [
        (e, s, d, rng.choice([0.25, -1.5, 0.5]), rng.choice(KINDS))
        for e, s, d in zip(ids(lane_count), ids(lane_count), ids(lane_count))
    ]
    protect = ids(protect_count)
    record = write_record(seq=37, row=RECORD_ID_MAX, captured=True, chain=(1 << 40) + 5, epoch=9, protect=protect,
                          lanes=lanes)
    got = read(record, 37)
    assert got["status"] == "ok"
    assert (got["row"], got["captured"], got["chain"], got["gen"]) == (RECORD_ID_MAX, True, (1 << 40) + 5, 9 << 32 | 37)
    assert got["protect"] == protect
    assert [(l["expert"], l["slot"], l["dst"], l["weight"], l["kind"]) for l in got["lanes"]] == lanes


def test_the_kinds_of_unused_lanes_are_never_judged():
    lanes = [(1, 2, 3, 0.5, int(LaneKind.HIT_SM)), (4, 5, 6, 0.25, int(LaneKind.MISS_GPU))]
    kinds = int(LaneKind.HIT_SM) | int(LaneKind.MISS_GPU) << 4 | 0xFFFFFF00
    got = read(write_record(seq=5, lanes=lanes, kinds=kinds), 5)
    assert got["status"] == "ok" and [l["kind"] for l in got["lanes"]] == [LaneKind.HIT_SM, LaneKind.MISS_GPU]


@pytest.mark.parametrize("case", ["lane_count", "kind_zero", "kind_six"])
def test_a_record_the_device_never_writes_is_malformed(case):
    override = {"lane_count": {"counts": 9}, "kind_zero": {"kinds": 0}, "kind_six": {"kinds": 6}}[case]
    record = write_record(seq=5, lanes=[(1, 2, 3, 0.5, int(LaneKind.HIT_SM))], **override)
    assert read(record, 5)["status"] == "malformed"


def test_a_record_whose_seq_is_not_the_expected_one_is_torn():
    record = write_record(seq=5 + 16, lanes=[(1, 2, 3, 0.5, int(LaneKind.HIT_SM))])
    assert read(record, 5)["status"] == "torn"
```

These pin today's behaviour, which Task 4's rewrite must keep, so they pass on the current code.

- [ ] **Step 3: Commit, push, run**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py \
  test/registered/unit/kernels/test_expert_stream_build_variants.py \
  test/registered/unit/kernels/test_exl3_ram_miss_read_record.py
git commit -m "$(cat <<'EOF'
test(expert-stream-read-record): read_record's fields round-trip; unused lanes are not judged

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

Run (divix01, fresh worktree `wt-read-record-dev` at this commit; update it with `git checkout --detach origin/...`
in later tasks):
`... -m pytest test/registered/unit/kernels/test_exl3_ram_miss_read_record.py test/registered/unit/kernels/test_expert_stream_build_variants.py -q -p no:randomly`
Expected: PASS.

---

### Task 3: One record is one 128-byte prefetch pair (D3)

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` (`kDemandRing`, `kRecordBytes`, the
  static asserts)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (the post launcher's page-alignment
  `RuntimeCheck`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (`seqlock_stress`'s buffer
  alignment)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`PAGE_BYTES`, `RECORD_BYTES`, `DEMAND_RING`)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (the request page table)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_read_record.py`,
  `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`

**Interfaces:**
- Produces: `kDemandRing = 128`, `kRecordBytes = 128`, `kPageBytes = 2176`; Python `DEMAND_RING = 128`,
  `RECORD_BYTES = 128`, `PAGE_BYTES = 2176`. Task 4 relies on `kRecLaneWeight + 4 * kMaxIds == kRecordBytes`.

- [ ] **Step 1: The failing tests**

Append to `test_exl3_ram_miss_read_record.py`, and add `DEMAND_RECORDS, DEMAND_RING, PAGE_BYTES` to its import:

```python
def test_a_record_is_one_128_byte_prefetch_pair():
    """The record's two cache lines are one 128-byte-aligned block, so an L2 miss on the first makes the adjacent-line
    prefetcher fetch the second; the ring starts on such a block."""
    assert RECORD_BYTES == 128 and DEMAND_RING % 128 == 0
    assert RECORD_FIELDS["lane_weight"] + 4 * LANES == RECORD_BYTES
    assert PAGE_BYTES == DEMAND_RING + DEMAND_RECORDS * RECORD_BYTES
```

In `test_exl3_lease_kernels_cuda.py`, `test_the_post_launch_refuses_what_a_narrow_record_cannot_carry`, change the
`page` case to a 64-byte-aligned view and its match:

```python
        else:
            c.dev.page = torch.zeros(PAGE_BYTES + 128, dtype=torch.uint8).pin_memory()[64 : 64 + PAGE_BYTES]
        with pytest.raises(RuntimeError, match="128-byte" if case == "page" else "32767"):
```

- [ ] **Step 2: Watch them fail**

Commit only the two tests (`test(expert-stream-read-record): a record is one 128-byte prefetch pair`), push, update
`wt-read-record-dev`, and run:
- `... -m pytest test/registered/unit/kernels/test_exl3_ram_miss_read_record.py -q -p no:randomly -k prefetch_pair`
  Expected: FAIL, `assert 256 == 128`.
- The GPU command with `-k "narrow_record and page"` on `test_exl3_lease_kernels_cuda.py`.
  Expected: FAIL, `DID NOT RAISE`: a 64-byte-aligned page passes today's 16-byte check.

- [ ] **Step 3: The layout**

In `lease_layout.h`:

```cpp
constexpr int64_t kDemandRing = 128;  // a 128-byte block of its own: each record below is one prefetch pair
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kRecordBytes = 128;  // two cache lines, one 128-byte-aligned block (the L2's adjacent-line pair)
```

Replace `static_assert(kRecLaneWeight + 4 * kMaxIds <= kRecordBytes, "record");` with:

```cpp
static_assert(kRecLaneWeight + 4 * kMaxIds == kRecordBytes, "the payload is the whole record: read_record copies it");
static_assert(kDemandRing % 128 == 0 && kRecordBytes == 128, "a record's two lines are one 128-byte prefetch pair");
```

In `expert_stream_transport.py`: `PAGE_BYTES = 2176`, `RECORD_BYTES = 128`, `DEMAND_RING = 128`.

In `lease_kernels.cuh`, the post launcher's page check becomes:

```cpp
    RuntimeCheck(reinterpret_cast<uintptr_t>(page.data_ptr()) % 128 == 0,
                 "page: must be 128-byte aligned, so each record's two cache lines are one prefetch pair (128-byte block)");
```

In `ffi_test_exports.h` `seqlock_stress`: `alignas(64) uint8_t record[kRecordBytes]` → `alignas(128) uint8_t record[kRecordBytes]`.

In `LEASE_PROTOCOL.md`, update the request-page table:
- the ring starts at byte 128;
- records are 128 bytes;
- the page is 2176 bytes.

Add one sentence: "A record is one 128-byte-aligned block, so the host's L2 fetches its second line with its first."

- [ ] **Step 4: Run, then commit and push**

```bash
grep -rn "4160\|RECORD_BYTES = 256\|kRecordBytes = 256" python/sglang/kernels python/sglang/test test/manual/dsv41 \
  test/registered/unit/kernels analysis/dsv41-drive/LEASE_PROTOCOL.md
```

Expected: no output. Unrelated `4160`s in `test/registered/unit/spec` and `test/registered/kernels/ops/attention`
are outside the search on purpose.

Run in `wt-read-record-dev` after pushing:
- `test_exl3_ram_miss_read_record.py` and `test_exl3_ram_miss_device_args.py`: PASS (the parity test sees the new
  constants);
- the GPU tests: EXIT=0;
- the full registered suite: EXIT=0, `base` counts + Task 2's and this task's tests.

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "$(cat <<'EOF'
expert-stream(read-record): a demand record is one 128-byte prefetch pair; the ring starts on a 128-byte block

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

- [ ] **Step 5: Benchmark** in a fresh worktree `wt-read-record-pair` at this commit: the benchmark command, three
  runs. Record them under `pair`.

---

### Task 4: `read_record` copies the record once and decodes from the copy (D2)

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h` (`read_record`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/fixed_vec.h` (`resize`)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_read_record.py`

**Interfaces:**
- Consumes: `kRecLaneWeight + 4 * kMaxIds == kRecordBytes` (Task 3); `read_record_fields` (Task 2).
- Produces: `FixedVec<T, N>::resize(size_t n)`, which throws past N and keeps the first n entries the caller wrote.

- [ ] **Step 1: The failing test**

In `test_a_record_the_device_never_writes_is_malformed`, add the case `"protect_count"` with the override
`{"counts": 1 | 9 << 4}`. The parametrize becomes `["lane_count", "protect_count", "kind_zero", "kind_six"]`.

Commit (`test(expert-stream-read-record): a protect count above eight is malformed`), push, run
`-k "malformed and protect_count"`.
Expected: FAIL. `status` is `"ok"`: today's reader keeps the first 8 ids.

- [ ] **Step 2: `FixedVec::resize`**

In `fixed_vec.h`, after `clear()`:

```cpp
  // Sets the size to n; the caller has already written the first n entries through operator[]. Past N throws.
  void resize(size_t n) {
    if (n > N) overflow();
    n_ = n;
  }
```

- [ ] **Step 3: The reader**

Replace `read_record` in `tier_protocol.h`, keeping the comment block above it and adding its last two sentences:

```cpp
// Seqlock read: the writer stores the payload, fences, then the seq word last, so a record whose seq reads `expected`
// both before and after the payload is whole. Records nothing waits for lap the ring unread, so a torn one must be
// detectable. A whole record whose counts or kinds are out of range is malformed: the device never writes one.
// The payload is copied once, at a constant size, between the two seq loads: both cache lines' loads issue together,
// before anything depends on the counts, and nothing after the second seq load reads the shared record.
enum class RecordRead { kOk, kTorn, kMalformed };

inline RecordRead read_record(const uint8_t* record, uint32_t expected, Request* request) {
  if (load_acquire(record + kRecSeq) != expected) return RecordRead::kTorn;
  alignas(64) uint8_t raw[kRecordBytes];
  std::memcpy(raw, record, kRecordBytes);
  std::atomic_thread_fence(std::memory_order_acquire);
  if (load_acquire(record + kRecSeq) != expected) return RecordRead::kTorn;
  uint16_t row;
  uint8_t counts, flags;
  uint64_t chain;
  uint32_t epoch, kinds;
  int16_t protect_ids[kMaxIds], expert[kMaxIds], slot[kMaxIds], dst[kMaxIds];
  float weight[kMaxIds];
  std::memcpy(&row, raw + kRecRow, 2);
  std::memcpy(&counts, raw + kRecCounts, 1);
  std::memcpy(&flags, raw + kRecFlags, 1);
  std::memcpy(&chain, raw + kRecChain, 8);
  std::memcpy(&epoch, raw + kRecEpoch, 4);
  std::memcpy(&kinds, raw + kRecKinds, 4);
  std::memcpy(protect_ids, raw + kRecProtect, sizeof(protect_ids));
  std::memcpy(expert, raw + kRecLaneExpert, sizeof(expert));
  std::memcpy(slot, raw + kRecLaneSlot, sizeof(slot));
  std::memcpy(dst, raw + kRecLaneDst, sizeof(dst));
  std::memcpy(weight, raw + kRecLaneWeight, sizeof(weight));
  const int count = counts & 0xF;
  const int protect = counts >> 4;
  if (count > kLeaseLanes || protect > kMaxIds) return RecordRead::kMalformed;
  bool bad_kind = false;
  for (int j = 0; j < kLeaseLanes; ++j) {
    const uint32_t kind = (kinds >> (4 * j)) & 0xFu;
    bad_kind |= j < count && (kind < kKindHitCopy || kind > kKindMissCpu);
  }
  if (bad_kind) return RecordRead::kMalformed;
  request->seq = expected;
  request->gen = static_cast<uint64_t>(epoch) << 32 | expected;
  request->row = row;
  request->captured = (flags & kRecFlagCaptured) != 0;
  request->chain = chain;
  for (int i = 0; i < kMaxIds; ++i)
    request->protect[i] = protect_ids[i];
  request->protect.resize(protect);
  for (int j = 0; j < kLeaseLanes; ++j) {
    request->lanes[j] = Lane{expert[j], slot[j], dst[j], weight[j], static_cast<uint8_t>((kinds >> (4 * j)) & 0xFu)};
  }
  request->lanes.resize(count);
  return RecordRead::kOk;
}
```

Add `static_assert(kRecLaneWeight + sizeof(float) * kMaxIds == kRecordBytes, "read_record copies the whole record");`
right above `read_record`.

- [ ] **Step 4: Run, commit, push**

Run `test_exl3_ram_miss_read_record.py` and `test_exl3_ram_miss_tier.py` (the seqlock stress lives there) in
`wt-read-record-dev` at the pushed commit. Expected: PASS, including `protect_count` and
`test_the_seqlock_reader_never_accepts_a_torn_record` (`torn == 0`).

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/fixed_vec.h
git commit -m "$(cat <<'EOF'
expert-stream(read-record): read_record copies the whole record at a constant size and decodes from the copy

Both cache lines' loads issue together, before anything depends on the counts; a protect count above eight is
malformed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

- [ ] **Step 5: The machine code**

On divix01, compile a probe of `read_record` at this commit. `$W` is `wt-read-record-dev`; `$TI` is
`/data/models/slang/.venv/lib/python3.13/site-packages/tvm_ffi/include`; `$DL` is
`/data/models/slang/.venv/lib/python3.13/site-packages/tvm_ffi/3rdparty/dlpack/include`.

```bash
mkdir -p /data/models/slang/nvfp4-work/read-record/probe && cd /data/models/slang/nvfp4-work/read-record/probe
cat > rr_probe.cpp <<"EOF"
#include "tier_protocol.h"
using namespace sglang::expert_stream;
__attribute__((noinline)) RecordRead probe_read_record(const uint8_t* record, uint32_t expected, Request* request) {
  return read_record(record, expected, request);
}
EOF
H=$W/python/sglang/kernels/jit/csrc/moe/expert_stream/host
g++ -std=c++20 -O3 -fPIC -I$H -I$H/.. -I$TI -I$DL -I$W/python/sglang/kernels/jit/include -c rr_probe.cpp -o rr_probe.o
objdump -d --no-show-raw-insn -C rr_probe.o | awk '/<probe_read_record/{f=1} f' | awk 'NR>1 && /^$/{exit} {print}' > rr.s
grep -c "" rr.s; grep -cE "^\s+[0-9a-f]+:\s+j[a-z]+ " rr.s; grep -E "vmovdq|movdq" rr.s | head
```

Expected:
- The copy appears as 16- or 32-byte vector loads (`vmovdqu`/`vmovdqa`) of the record, with no branch between
  the two seq loads.
- No byte-ladder loops.
- Fewer instructions and fewer conditional jumps than the af5b31c5d2 probe. That probe's disassembly was 452 lines,
  with a branch ladder per variable-length copy; its source is
  `divix01:/data/models/slang/nvfp4-work/record-narrow/rr/rr.s` if it still exists.

Record the line and jump counts in "Results".

- [ ] **Step 6: Benchmark** in a fresh worktree `wt-read-record-copy`: three runs, recorded under `copy`.

---

### Task 5: Prefetch every line of the request when its post is seen (D1)

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`pump_demand`; a new
  `prefetch_request`)

**Interfaces:**
- Consumes: `kRecordBytes == 128` (Task 3); `hot_page_`, `hot_stride_` (a multiple of `kHotAlignment` = 64).
- Produces: `void RamTier::prefetch_request(const uint8_t* record, uint32_t seq) const`.

This task changes no behaviour, so no unit test can see it. Its tests are the whole suite (unchanged) and the
benchmark.

- [ ] **Step 1: The prefetch**

In `ram_tier.h`, after `read_gpu_hot`:

```cpp
  // Starts the loads of every line the request's read will touch: the record's two lines and its hot record's. The
  // device has just written each, so each is an L3 miss; issued together they overlap instead of queueing behind
  // read_record's branches. Called once the head shows the post: a line prefetched earlier would be refetched.
  void prefetch_request(const uint8_t* record, uint32_t seq) const {
    static_assert(kRecordBytes == 128, "a record is two lines");
    _mm_prefetch(reinterpret_cast<const char*>(record), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(record + 64), _MM_HINT_T0);
    if (hot_page_ == nullptr) return;
    const uint8_t* hot = hot_page_ + static_cast<int64_t>((seq - 1u) % kHotRecords) * hot_stride_;
    for (int64_t line = 0; line < hot_stride_; line += 64)
      _mm_prefetch(reinterpret_cast<const char*>(hot + line), _MM_HINT_T0);
  }
```

In `pump_demand`, right after `uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);`:

```cpp
    prefetch_request(record, next_demand_);
```

`<immintrin.h>` is already included for `_mm_pause` and `_mm_sfence`. If the build says `_mm_prefetch` is
undeclared, add `#include <immintrin.h>` to `ram_tier.h`'s includes.

- [ ] **Step 2: Commit, push, suite, GPU tests, benchmark**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h
git commit -m "$(cat <<'EOF'
expert-stream(read-record): the service prefetches the record's and the hot record's lines when it sees the post

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

In a fresh worktree `wt-read-record-prefetch`:
- The full registered suite: EXIT=0, the Task 3 counts + 1 (Task 4's new case).
- The GPU tests: EXIT=0.
- The benchmark, three runs, recorded under `prefetch`. Expected: the span median at or below `copy`.
  `mem_load_retired.l3_hit` per post drops: a demand load that finds its line already prefetched is not counted as
  an L3 hit.

---

### Task 6: The service busy-polls a dedicated physical core (D4, option B)

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/core_topology.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h` (constructor, `run`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`start_thread`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`cpu_cores()`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h` (`CpuExpertEngine::cores()`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`start_thread(busy_poll=...)`)
- Modify: `python/sglang/srt/environ.py` (`SGLANG_DSV41_RAM_MISS_SPIN_CORE`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (the `start_thread` call)
- Modify: `benchmarks/dsv41_baseline/arm_env.py` (`SERVER_CORES`, `SPIN_CORE`, `base_env`)
- Modify: `benchmarks/dsv41_baseline/test_dsv41_baseline.py`
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md`
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_thread.py`

**Interfaces:**
- Produces:
  - `ExpertStreamHost.start_thread(*, cpu_core=-1, fatal_wait_s=30.0, spin_us=5000, busy_poll=False)`.
  - FFI `expert_stream_start_thread(handle, cpu_core, fatal_wait_ns, spin_ns, busy_poll)`.
  - `check_dedicated_core(int core, const std::vector<int>& cpu_expert_cores, const std::string& prefix)`.
  - `arm_env.SPIN_CORE = 17`.

Read `.claude/skills/env-var-conventions/SKILL.md` before Step 6.

- [ ] **Step 1: The failing tests**

Append to `test_exl3_ram_miss_thread.py`, and add `contextlib` and `from pathlib import Path` to its imports:

```python
def _plain_host(tmp_path):
    s = ram_miss_setup(tmp_path)
    return ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32))


def _cpu_list(text: str) -> set[int]:
    cores = set()
    for item in text.strip().split(","):
        first, _, last = item.partition("-")
        cores.update(range(int(first), int(last or first) + 1))
    return cores


def _physical_core_pair() -> tuple[int, int]:
    """A core of this process's affinity with exactly one SMT sibling, both in the affinity and below 64; or skip."""
    mask = os.sched_getaffinity(0)
    for core in sorted(mask):
        path = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list")
        if not path.exists():
            continue
        siblings = _cpu_list(path.read_text())
        if len(siblings) == 2 and all(c < 64 and c in mask for c in siblings):
            return core, next(c for c in siblings if c != core)
    pytest.skip("no SMT core pair in this process's affinity")


@contextlib.contextmanager
def _affinity(cores):
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cores)
    try:
        yield
    finally:
        os.sched_setaffinity(0, before)


def _service_cpu_s() -> float:
    for task in Path("/proc/self/task").iterdir():
        if (task / "comm").read_text().strip().endswith("ram-miss"):
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")  # utime + stime
    raise AssertionError("no RAM-miss service thread")


def test_a_busy_polling_service_needs_a_core(tmp_path):
    host = _plain_host(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="busy_poll needs cpu_core"):
            host.start_thread(busy_poll=True)
        assert not host.threaded
    finally:
        host.stop()


@pytest.mark.parametrize("shared", ["core", "sibling"])
def test_a_busy_polling_service_refuses_a_physical_core_it_would_share(tmp_path, shared):
    """The caller's affinity (which the process's other threads inherit) holds the core itself, or only its sibling."""
    core, sibling = _physical_core_pair()
    kept = core if shared == "core" else sibling
    host = _plain_host(tmp_path)
    try:
        with _affinity(os.sched_getaffinity(0) - ({core, sibling} - {kept})):
            with pytest.raises(RuntimeError, match=f"core {kept} shares"):
                host.start_thread(cpu_core=core, busy_poll=True)
        assert not host.threaded
    finally:
        host.stop()


@pytest.mark.parametrize("busy", [False, True])
def test_a_busy_polling_service_spins_where_a_default_one_sleeps(tmp_path, busy):
    """Idle for half a second: a busy-polling service uses its core the whole time, a default one (1 ms of spin) sleeps.
    Both still park for a pause and stop."""
    core, sibling = _physical_core_pair()
    host = _plain_host(tmp_path)
    try:
        with _affinity(os.sched_getaffinity(0) - {core, sibling}):
            host.start_thread(cpu_core=core, busy_poll=busy, spin_us=1000)
        assert host.counters()["spin_cpu"] == core
        before = _service_cpu_s()
        time.sleep(0.5)
        used = _service_cpu_s() - before
        assert (used > 0.3) if busy else (used < 0.1), (busy, used)
        host.pause(5.0)
        host.resume()
    finally:
        host.stop()
```

Commit (`test(expert-stream-read-record): a busy-polling service needs a physical core of its own, and never sleeps`),
push, run `-k busy_polling` on `test_exl3_ram_miss_thread.py`.
Expected: FAIL with `TypeError: ... unexpected keyword argument 'busy_poll'`. The default-mode case
(`busy=False`) fails the same way, since it also passes `busy_poll`.

- [ ] **Step 2: The topology check**

Create `host/core_topology.h`:

```cpp
// The physical-core check a busy-polling service thread needs (RamThread, busy_poll).
#pragma once

#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixed_vec.h"

namespace sglang::expert_stream {

// The SMT siblings of `core`, the core included, from sysfs ("17,53" or "16-17").
inline std::vector<int> core_siblings(int core) {
  std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(core) + "/topology/thread_siblings_list");
  std::string list;
  if (!std::getline(in, list)) throw std::runtime_error("cannot read the SMT siblings of core " + std::to_string(core));
  std::vector<int> cores;
  std::stringstream items(list);
  for (std::string item; std::getline(items, item, ',');) {
    const size_t dash = item.find('-');
    const int first = std::stoi(item.substr(0, dash));
    const int last = dash == std::string::npos ? first : std::stoi(item.substr(dash + 1));
    for (int c = first; c <= last; ++c)
      cores.push_back(c);
  }
  return cores;
}

// A busy-polling service never yields its core, so it needs the physical core to itself: no SMT sibling of `core`
// (the core included) may be in the caller's affinity, which the process's other threads inherit, or among the CPU
// experts' cores, which their threads pin themselves to.
inline void check_dedicated_core(int core, const std::vector<int>& cpu_expert_cores, const std::string& prefix) {
  if (core < 0) {
    throw std::runtime_error(prefix + "busy_poll needs cpu_core: a busy-polling service is pinned to a core of its own");
  }
  cpu_set_t caller;
  CPU_ZERO(&caller);
  if (pthread_getaffinity_np(pthread_self(), sizeof(caller), &caller) != 0)
    throw std::runtime_error(prefix + "busy_poll: cannot read the caller's affinity");
  for (const int sibling : core_siblings(core)) {
    if (CPU_ISSET(sibling, &caller)) {
      throw std::runtime_error(
          prefix + "busy_poll: core " + std::to_string(sibling) + " shares the physical core of cpu_core " +
          std::to_string(core) + " and is in the caller's affinity, which the process's other threads inherit");
    }
    if (listed(cpu_expert_cores, sibling)) {
      throw std::runtime_error(
          prefix + "busy_poll: core " + std::to_string(sibling) + " shares the physical core of cpu_core " +
          std::to_string(core) + " and is a CPU expert core (SGLANG_DSV41_CPU_EXPERTS_CORES)");
    }
  }
}

}  // namespace sglang::expert_stream
```

In `cpu_experts.h`, in `CpuExpertEngine`'s public section:

```cpp
  const std::vector<int>& cores() const {
    return config_.cores;
  }
```

In `ram_tier.h`, next to the other CPU-expert accessors (by `cpu_`):

```cpp
  // The CPU experts' cores, empty without CPU experts. The caller's, before the service thread starts.
  std::vector<int> cpu_cores() const {
    return cpu_ != nullptr ? cpu_->cores() : std::vector<int>{};
  }
```

- [ ] **Step 3: The thread**

In `ram_thread.h`:
- The constructor gains `bool busy_poll` after `spin_ns`, stored in `busy_poll_(busy_poll)`.
- Add the member `bool busy_poll_;` after `int64_t spin_ns_;`.
- Extend the class comment's first sentence with: "or, with busy_poll, on a core of its own (start_thread checked),
  spins with no PAUSE and never sleeps".

In `run()`:

```cpp
      if (tier_->pump_demand()) {
        idle = 0;
        continue;
      }
      if (busy_poll_) continue;  // a physical core of its own: no PAUSE quantum on detection, no sleep
      if (++idle < spin_iters_) {
```

- [ ] **Step 4: The FFI entry point**

In `ffi_exports.h`:
- Add `#include "core_topology.h"`.
- `start_thread` takes `int64_t busy_poll` after `spin_ns`.
- After `std::shared_ptr<RamTier<Source>> tier = find(handle);` and before the registry lock, add:

```cpp
    if (busy_poll != 0) check_dedicated_core(static_cast<int>(cpu_core), tier->cpu_cores(), error_prefix<Layout>());
```

The `make_shared<Thread>(...)` call passes `busy_poll != 0` after `spin_ns`.

In `expert_stream_transport.py`, `start_thread`:

```python
    def start_thread(self, *, cpu_core: int = -1, fatal_wait_s: float = 30.0, spin_us: int = 5000,
                     busy_poll: bool = False) -> None:
        """Serve requests on a C++ thread (no more ``pump()``), with the fail-stop watchdog.

        ``cpu_core`` -1 inherits the caller's affinity; cores 64-71 are reserved (D19). ``busy_poll`` spins on
        ``cpu_core`` with no PAUSE and no sleep; the C++ side refuses it unless that physical core is the service's alone.
        """
        if 64 <= cpu_core <= 71:
            raise ValueError(
                f"cpu_core {cpu_core}: cores 64-71 are reserved (NVMe completion interrupts are pinned there)"
            )
        self._module.expert_stream_start_thread(
            self.handle, cpu_core, int(fatal_wait_s * 1e9), int(spin_us * 1e3), int(busy_poll)
        )
        self.threaded = True
```

- [ ] **Step 5: Watch the tests pass**

Commit, push, update `wt-read-record-dev`, and run `test_exl3_ram_miss_thread.py`.
Expected: PASS. All of the file's earlier tests still pass: the default path is unchanged.

- [ ] **Step 6: Production wiring**

In `environ.py`, directly after `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS = EnvInt(2000)`:

```python
    # The core the RAM-miss service thread busy-polls the request page on, with no PAUSE and no sleep; unset, it
    # inherits the server's affinity and spins with PAUSE, then sleeps. Set, the service refuses to start unless the
    # core's whole physical core is its own: no SMT sibling in the server's affinity or SGLANG_DSV41_CPU_EXPERTS_CORES.
    SGLANG_DSV41_RAM_MISS_SPIN_CORE = EnvInt(None)
```

In `exl3_ram_miss.py`, replace `host.start_thread(fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms))` with:

```python
            spin_core = envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.get()
            if spin_core is None:
                host.start_thread(fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms))
            else:
                host.start_thread(
                    cpu_core=spin_core, busy_poll=True, fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms)
                )
```

In `arm_env.py`:
- Change `SERVER_CORES` to `"0-7,16,36-52"`.
- Add `SPIN_CORE = 17` under it, with the comment: "the RAM-miss service busy-polls cpu 17; its SMT sibling 53 and
  17 itself are left out of SERVER_CORES so the physical core is the service's alone (plan
  2026-10-01-expert-stream-read-record D4). Arms from here on have one physical core fewer: not comparable to
  earlier cells."
- Add `"SGLANG_DSV41_RAM_MISS_SPIN_CORE": str(SPIN_CORE),` to `base_env()`, after
  `"SGLANG_DSV41_RAM_MISS_TIMEOUT_MS"`.

In `test_dsv41_baseline.py`, after `test_server_cores_touch_no_node_1_core`:

```python
def test_the_ram_miss_spin_core_has_its_physical_core_to_itself():
    # divix01: cpu n and n + 36 are one physical core (thread_siblings_list).
    siblings = {arm_env.SPIN_CORE, arm_env.SPIN_CORE + 36}
    taken = _cores(arm_env.SERVER_CORES) | _cores(arm_env.DRIVER_CORES) | _cores(arm_env.FREE_CORES)
    assert arm_env.SPIN_CORE in NODE0_CPUS and not (siblings & taken), sorted(siblings & taken)
    assert arm_env.base_env()["SGLANG_DSV41_RAM_MISS_SPIN_CORE"] == str(arm_env.SPIN_CORE)
```

Run it, as `benchmarks/dsv41_baseline` runs its own tests:
`cd benchmarks/dsv41_baseline && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test_dsv41_baseline.py -q -p no:randomly -k "cores or spin_core"`.
Expected: PASS.

In `LEASE_PROTOCOL.md`, in the section that describes the service thread's idle spin (`grep -n "spin" analysis/dsv41-drive/LEASE_PROTOCOL.md`),
add: "With `SGLANG_DSV41_RAM_MISS_SPIN_CORE`, the service busy-polls that core with no PAUSE and never sleeps;
`start_thread` refuses unless no SMT sibling of the core is in the server's affinity or among the CPU experts' cores."

- [ ] **Step 7: Commit, push, suite, GPU tests, benchmark**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/core_topology.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/exl3_ram_miss.py benchmarks/dsv41_baseline/arm_env.py \
  benchmarks/dsv41_baseline/test_dsv41_baseline.py analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "$(cat <<'EOF'
expert-stream(read-record): the RAM-miss service busy-polls a physical core of its own

start_thread(busy_poll=True) spins with no PAUSE and never sleeps, and refuses a core whose SMT sibling is in the
caller's affinity or among the CPU experts' cores. Production: SGLANG_DSV41_RAM_MISS_SPIN_CORE; the arm recipe gives
it cpu 17 and leaves 17 and 53 out of SERVER_CORES.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

In a fresh worktree `wt-read-record-final`:
- The full registered suite: EXIT=0, the Task 5 counts + 5 (this task's tests).
- The GPU tests: EXIT=0.
- The benchmark with `--busy-poll`, three runs, recorded under `busy`. Expected:
  - `service core: 17 busy_poll: True`;
  - the span median at or below `prefetch`. The span starts at detection, so it does not show the PAUSE quantum
    removed: `pause:` is that saving's upper bound, about half of it on average.
- One more run without `--busy-poll` in the same worktree, recorded as `final-nobusy`. It separates Tasks 3-5 from
  Task 6 at the final code.

---

### Task 7: Root measurements, results and cleanup

**Files:**
- Modify: this plan's "Results" section.

- [ ] **Step 1: Hand the owner the root commands**

Root is needed for uncore counters and the DDIO MSR. Ask the owner to run these on divix01 with the GPU lock held and
no suite running, once in `wt-read-record-base` and once in `wt-read-record-final`. In the final worktree, add
`--busy-poll` to the benchmark.

```bash
W=/data/models/slang/nvfp4-work/wt-read-record-base   # then wt-read-record-final
cd $W/test/manual/dsv41
sudo rdmsr -p 0 0xc8b   # IIO_LLC_WAYS: the L3 ways DDIO may allocate (Skylake-SP default 0x600, two ways)
EV=unc_cha_tor_inserts.io_hit,unc_cha_tor_inserts.io_miss,unc_cha_tor_inserts.io_miss_itom,unc_cha_tor_inserts.io_miss_rfo,unc_iio_txn_req_of_cpu.mem_write.part0,unc_iio_txn_req_of_cpu.mem_write.part1,unc_iio_txn_req_of_cpu.mem_write.part2,unc_iio_txn_req_of_cpu.mem_write.part3,unc_iio_data_req_of_cpu.mem_write.part0,unc_iio_data_req_of_cpu.mem_write.part1,unc_iio_data_req_of_cpu.mem_write.part2,unc_iio_data_req_of_cpu.mem_write.part3
flock /data/models/slang/nvfp4-work/cc-gpu.lock sudo perf stat -a -x, -e $EV -o /tmp/read-record-uncore.csv -- \
  env PYTHONPATH=$W/python OMP_NUM_THREADS=8 taskset -c 36-52 /data/models/slang/.venv/bin/python \
  bench_exl3_service_read.py --cpu-core 17 --posts 20000 --gap-us 100
sudo perf stat -a -x, -e $EV -o /tmp/read-record-idle.csv -- sleep 10   # the same counters with no posts
cat /tmp/read-record-uncore.csv /tmp/read-record-idle.csv
```

If `rdmsr` is missing, `sudo dnf install msr-tools && sudo modprobe msr` provides it.

How to read them:
- **DDIO allocation:** `io_hit / (io_hit + io_miss)` for the run minus idle is the share of inbound writes that hit
  the L3.
- **Write sizes:** `txn_req_of_cpu.mem_write` per post (run minus idle, scaled to the run's length, divided by
  20000) is the PCIe write transactions per record.
- **Bytes per transaction:** `data_req_of_cpu.mem_write` counts 4-byte units, so 4 × data / txn is the bytes per
  transaction.

These numbers decide whether a later plan should write the record with full-line stores (finding 6, outside this
plan).

- [ ] **Step 2: Results**

Under "Results", add one table row per label (`base`, `pair`, `copy`, `prefetch`, `busy`, `final-nobusy`) with:
- the commit;
- the median of three runs' span p10, median, p90 and p99;
- the per-post `br_misp_retired`, `machine_clears.memory_ordering`, `mem_load_retired.l3_hit` and
  `mem_load_l3_miss_retired.local_dram`;
- `pause` ns;
- the suite counts.

Below the table, add:
- the probe's instruction and jump counts (Task 4 Step 5);
- the owner's uncore numbers (base and final);
- one sentence per finding saying whether its expected effect showed.

- [ ] **Step 3: Remove the divix01 worktrees and commit**

```bash
ssh divix01 'for w in base dev pair copy prefetch final; do git -C /data/models/slang/sglang worktree remove --force /data/models/slang/nvfp4-work/wt-read-record-$w; done; rm -rf /data/models/slang/nvfp4-work/read-record/probe'
git add docs/superpowers/plans/2026-10-01-expert-stream-read-record.md
git commit -m "$(cat <<'EOF'
docs(expert-stream-read-record): the service's read span per change, and the L3 and PCIe counts behind it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-read-record
```

---

## Results

(Filled in by Tasks 1, 3, 4, 5, 6 and 7: command, then number.)

Bench: `bench_exl3_service_read.py --cpu-core 17 --perf` from `test/manual/dsv41`, under `cc-gpu.lock` and
`taskset -c 36-52`, nothing else on the box. Span = the stage trace's `done - observed`. Perf counts are the service
thread's user-mode events per post, and include its idle polling between posts (100 us gap).

**base** (e732163d16; service path as 6166309231)
- PAUSE: 38.6 ns.
- Span (ns), three runs: median 419 / 416 / 414; p10 393 / 389 / 380; p90 467 / 466 / 462; p99 600 / 673 / 598.
- Overruns in the window: 0, 0, 0.
- Per post: instructions 200726 / 197660 / 197897 (mostly the idle poll); branch misses 16.4 / 15.2 / 16.3;
  memory-ordering clears 0.04 / 0.07 / 0.05; L3 hits 1.97 / 2.00 / 2.07 (all xsnp_none); local DRAM 0.03; remote
  DRAM 0.00. The record's lines arrive in the L3 (DDIO), not DRAM.
- Suite at 6166309231: 1218 passed, 2 failed, 22 skipped. The census (fixed in e732163d16) and the seqlock stress's
  acceptance floor (87 < 100, torn 0; 3/3 pass alone). GPU tests: 24 passed.
