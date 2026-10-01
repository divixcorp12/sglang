# Narrow Demand Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the post kernel's host-memory traffic. The demand record is written in 9 relaxed system-scope stores
instead of 56 (Tasks 1-4). The map delta is read with 16-byte loads issued together instead of up to 41 serial loads
(Tasks 5 and 7). Lane typing reads no host memory and issues its GPU loads in batches (Task 6). After the tag
acquire, the post waits about one more PCIe round trip instead of one per delta entry.

**Architecture:** One wire-format change to the demand record (`lease_layout.h`), applied at once to its four
writers/readers: the device writer `write_record`, the host reader `read_record`, the C++ seqlock stress writer, and
the Python mirror plus the Python simulator's writer. Ids (expert, slot, dst, protect) go from i32 to i16; counts,
flags and kinds are packed into bytes and nibbles; the lanes go from an array of 16-byte structs to a struct of
arrays, so each field is one or two `st.relaxed.sys.global.v4.b32` stores. The routing weight stays f32 (see
"Decisions"). The record stays 256 B, so `PAGE_BYTES`, the ring and every other page offset are unchanged.

**Tech Stack:** CUDA C++ (sglang JIT kernels, inline PTX), C++ host module, Python/torch, pytest.

**Spec:** this conversation's design, restated in "Decisions" below; no separate spec file.

## Decisions

- **Weight stays f32.** The CPU-experts quality gate is already borderline (E31 fails on one flip, KL 0.0018 mean,
  undecided: `2026-09-29-dsv41-cpu-experts.md`, "Quality gate"). The weight is summed from `topk_weights`
  (`exl3.py:592`), which is fp32. With the struct-of-arrays lanes, fp16 weights would save one 16-byte store out of
  nine, which is not worth a new accuracy question.
- **i16 for ids, with -1 for none.** Bounds: DSV4.1 has 384 experts; a RAM slot holds one 13.3 MB expert row
  (`analysis/dsv41-drive/early-prefetch/window_replay.py:57`), so 32767 slots would be 435 GB for one layer; dst is a
  VRAM slot, fewer still. Python refuses experts or row capacities above 32767 at construction, and the device traps
  any id outside [-1, 32767] while packing.
- **Nibbles for counts and kinds** (count and protect count are at most 8; kinds are 1..5). They save no stores on
  their own, but they make the header fit the 16-byte stores.
- **The record stays 256 B.** Shrinking it to 128 would change `PAGE_BYTES` and the ring arithmetic for no saved
  store.

### The new record (byte offsets)

| Offset | Field | Type | Store |
|---|---|---|---|
| 0 | `seq` | u32 | seqlock word, unchanged: 0, release fence, payload, release |
| 4 | `row` | u16 | one u32 store with the next two |
| 6 | `counts` | u8 | lane count in bits 0-3, protect count in bits 4-7 |
| 7 | `flags` | u8 | `kRecFlagCaptured` = 1 |
| 8 | `chain` | u64 | one v2 store |
| 16 | `epoch` | u32 | one v4 store with `kinds` and 8 zero bytes |
| 20 | `kinds` | u32 | lane j's kind in bits 4j..4j+3 |
| 24 | (zero) | 8 B | |
| 32 | `protect` | i16[8] | one v4 store |
| 48 | `lane_expert` | i16[8] | one v4 store |
| 64 | `lane_slot` | i16[8] | one v4 store |
| 80 | `lane_dst` | i16[8] | one v4 store |
| 96 | `lane_weight` | f32[8] | two v4 stores |
| 128-255 | unused | | |

Payload: 1 u32 + 1 v2 + 7 v4 = 9 stores (today: 7 header + 48 lane-loop stores). Plus the two seq stores either way.

## Global Constraints

- Every committed change is pushed and run on divix01 in a private worktree, never the production checkout
  (`.claude/rules/divix01-run-protocol.md`): `git push origin <branch>`, then
  `git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-<name> origin/<branch>`.
- Every run uses `PYTHONPATH=$PWD/python` and prints `sglang.__file__` before its result is trusted.
- CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS=8`; GPU jobs under
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 <cmd>`.
- Any piped pytest run reads `${PIPESTATUS[0]}`, never the pipeline's status.
- The registered suite target is `test/registered/unit/kernels`, compared against the merge-base run with the same
  command.
- Inline PTX stays in `lease_device.cuh` (`test_expert_stream_sync_primitives.py`
  `test_inline_ptx_lives_only_in_the_lease_device_helpers`); `asm volatile(` must stay on one line so the volatile rule
  in that file still passes.
- `lease_layout.h` is the only C++ home of a wire constant; every one of its constants has an equal entry in
  `PYTHON_WIRE` (`test_exl3_ram_miss_device_args.py`).
- Comments follow `.claude/rules/comment-style.md`: one or two lines, ASCII, the fact the reader cannot see.
- Work on a new branch `expert-stream-record-narrow` cut from `dsv41-cpu-dispatch` HEAD (`9e6d79e138`), in its own
  laptop worktree.
- **The GPU check** (run where a task says so; `<wt>` is the divix01 worktree for the pushed commit, `<label>` a
  short name for the results). Create the worktree first with the fetch + `worktree add --detach` command above.

  ```bash
  W=/data/models/slang/nvfp4-work/<wt>
  cd /data/models/slang/nvfp4-work/scratch-atomic-probe
  PYTHONPATH=$W/python python3 sass_gate.py build $W <label>.sass
  awk '/exl3_ram_miss_post_kernel/,/EXIT/' <label>.sass | grep -oE "(LDG|STG)\.E(\.64|\.128)?\.STRONG\.SYS" | sort | uniq -c
  cd $W/test/manual/dsv41
  PYTHONPATH=$W/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
    /data/models/slang/.venv/bin/python -m pytest test_exl3_lease_kernels_cuda.py test_exl3_slot_map_kernels_cuda.py -q -p no:randomly; echo "EXIT=$?"
  for mode in hits delta; do
    PYTHONPATH=$W/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
      /data/models/slang/.venv/bin/python bench_exl3_post_record.py --mode $mode
    NSYS_TMPDIR=/mnt/nvme1/nsys-tmp PYTHONPATH=$W/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
      nsys profile --trace=cuda -o /data/models/slang/nvfp4-work/record-narrow/<label>-$mode \
      /data/models/slang/.venv/bin/python bench_exl3_post_record.py --mode $mode
    nsys stats --report cuda_gpu_kern_sum /data/models/slang/nvfp4-work/record-narrow/<label>-$mode.nsys-rep | grep exl3_ram_miss_post
  done
  ```

  `sass_gate.py` is the native-sync plan's gate (`2026-09-27-expert-stream-native-sync.md`, Task 0); if it is gone,
  rebuild it from that plan's listing. Expected: the GPU tests report EXIT=0, and the `sglang:` line names
  `$W/python`. Record the store/load counts, us/post and kernel median for both modes in "Results", under `<label>`.

## Review Focus

1. **A page whose base is not 16-byte aligned.** A misaligned v4 store faults inside a captured graph. Expected: the
   constructor refuses the page with a clear error. Test added in Task 2.
2. **An id that does not fit in i16** (a config with more than 32767 experts or slots, or a bad dst from the plan).
   Expected: refused at construction (experts, capacities); a device trap for anything that gets past it, not a
   silently wrapped id that serves the wrong expert. Test for the refusal in Task 2; the trap is reviewed in Task 3.
3. **A torn record under the new layout.** The v4 stores are not single-copy atomic as a whole. Expected: the
   seqlock still rejects every torn read. The stress test is updated in Task 3, and it must still report 0 torn.
4. **A record with fewer than 8 lanes.** Expected: unused lanes read -1 ids, weight 0.0, kind 0, and the host reads
   exactly `count` lanes. Pinned by the GPU test in Task 3 (2 lanes of 8).
5. **A plan the reference rejects** (a repeated expert, a miss with no staging slot) once `type_lanes` batches its
   loads (Task 6). Expected: the post still traps rather than typing lanes from it. The repeated-expert trap test is
   added in Task 6; `test_post_types_lanes_like_the_reference` covers every accepted plan.

---

### Task 1: Baseline the post kernel's cost

**Files:**
- Create: `test/manual/dsv41/bench_exl3_post_record.py`

**Interfaces:**
- Consumes: `lease_chain_rig.Chain`, `TOP_K` (`test/manual/dsv41/lease_chain_rig.py:32-43`).
- Produces: `bench_exl3_post_record.py` printing `post: <us> us/post`; baseline numbers recorded in this plan's
  "Results" section.

- [ ] **Step 1: Write the benchmark**

```python
"""Eager post-kernel cost with SM-hit lanes only: records of SM hits lap the ring, so nothing has to serve them.

--mode hits: the row's delta is already applied, so the post reads only the delta's tag.
--mode delta: before each post the row's applied word is reset, so every post re-applies the same full delta
(DELTA_MAX_ENTRIES entries that leave the map as it is). This is the path a post takes after any miss.

Run on divix01 under cc-gpu.lock from test/manual/dsv41, with PYTHONPATH pointing at the tree under test. For the
kernel's own duration, run it under nsys and read exl3_ram_miss_post_kernel's row in cuda_gpu_kern_sum.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from lease_chain_rig import CAPACITY, EXPERTS, TOP_K, Chain  # noqa: E402

import sglang  # noqa: E402
from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402


def write_delta(block: torch.Tensor, row: int, tag: int, staging, entries) -> None:
    """The host's delta publication, done here: payload, then the tag (x86 keeps tensor stores in order)."""
    base = lease.DELTA_BASE + row * lease.DELTA_STRIDE
    f = lease.DELTA_FIELDS
    flat = [v for pair in entries for v in pair]
    block[base + f["count"] : base + f["count"] + 4].view(torch.int32)[0] = len(entries)
    block[base + f["staging"] : base + f["staging"] + 4 * lease.LANES].view(torch.int32)[:] = torch.tensor(
        staging, dtype=torch.int32)
    block[base + f["entries"] : base + f["entries"] + 4 * len(flat)].view(torch.int32)[:] = torch.tensor(
        flat, dtype=torch.int32)
    block[base + f["tag"] : base + f["tag"] + 8].view(torch.int64)[0] = tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hits", "delta"], default="hits")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=2000)
    args = parser.parse_args()
    print("sglang:", sglang.__file__)
    with tempfile.TemporaryDirectory() as tmp:
        c = Chain(Path(tmp), start=False)
        try:
            row = 0
            experts = list(range(TOP_K))
            c.dev.map_bulk_apply(torch.tensor([[row, e, e] for e in experts], dtype=torch.int32))
            c.plan(experts, row)
            backend, plan = c.backends[row], c.plans[row]
            backend._stage_planned(plan)
            applied = c.dev.map_bank["map_applied"]
            if args.mode == "delta":
                torch.cuda.synchronize()
                tag = int(c.dev.map_bank["map_chain"][row])
                staging = c.dev.map_bank["staging"][row].tolist()
                entries = [(e, e) for e in range(CAPACITY)] + [(e, -1) for e in range(CAPACITY, EXPERTS)]
                assert len(entries) == lease.DELTA_MAX_ENTRIES
                write_delta(c.host.lease_block, row, tag, staging, entries)

            def post() -> None:
                if args.mode == "delta":
                    applied[row] = 0
                c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots)

            for _ in range(args.warmup):
                post()
            torch.cuda.synchronize()
            assert c.kinds(TOP_K) == [LaneKind.HIT_SM] * TOP_K, c.kinds(TOP_K)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                post()
            end.record()
            end.synchronize()
            print(f"post ({args.mode}): {start.elapsed_time(end) * 1000 / args.iters:.2f} us/post over {args.iters} "
                  f"({TOP_K} SM-hit lanes)")
        finally:
            c.close()


if __name__ == "__main__":
    main()
```

The delta mode's event loop also times the `applied[row] = 0` fill kernel, so read the nsys kernel median for the
post itself. Task 7 changes `write_delta` along with the delta layout.

- [ ] **Step 2: Commit and push**

```bash
git add test/manual/dsv41/bench_exl3_post_record.py
git commit -m "$(cat <<'EOF'
bench(expert-stream-record-narrow): the post kernel's cost with SM-hit lanes only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-record-narrow
```

- [ ] **Step 3: Run the baseline on divix01**

Run the GPU check (Global Constraints) with `<wt>` = `wt-record-narrow-base` at this commit and `<label>` = `base`.
Expected: both modes print a us/post figure and an nsys kernel median. The delta mode's median should sit well above
the hits mode's, since that gap is the serial delta reads this plan removes. Record everything in "Results".

- [ ] **Step 4: Record the registered-suite baseline**

```bash
cd /data/models/slang/nvfp4-work/wt-record-narrow-base
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Record the passed/skipped/failed counts and EXIT in "Results".

---

### Task 2: Refuse what the narrow record cannot carry

> **Revised during execution (owner's call):** the refusals live in C++, where the ids are narrowed, not in the
> Python constructors below. `expert_stream_open` (`host/ffi_exports.h`) refuses `starts.size(1)` or any `capacity`
> above `kRecIdMax`, before its table checks. The post launcher (`LeaseProtocolKernel::post`) refuses `experts` or
> `row_capacity` above `kRecIdMax`, and a page off 16-byte alignment, before any matcher. `kRecIdMax` lands in
> `lease_layout.h` in this task, mapped in `PYTHON_WIRE`, so Task 3 does not add it again. Tests:
> `test_the_host_module_refuses_tables_its_records_cannot_carry` (registered, CPU) calls `expert_stream_open`
> directly, and `test_the_post_launch_refuses_what_a_narrow_record_cannot_carry` (manual GPU) covers the device
> launcher. Commits `7f51f0060e` (RED) and `67567732ba` (GREEN). The steps below are the original Python version,
> superseded.

**Files:**
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, ~line 1327;
  `ExpertStreamHost.__init__`, ~line 803; the `RECORD_FIELDS` block ~line 684)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`

**Interfaces:**
- Produces: `expert_stream_transport.RECORD_ID_MAX = 32767` (Task 3's `PYTHON_WIRE` maps `kRecIdMax` to it).

- [ ] **Step 1: Write the failing tests** (append after `test_the_timeout_must_be_positive`)

```python
def test_more_experts_than_a_record_id_carries_are_refused():
    with pytest.raises(ValueError, match="32767"):
        _device(experts=ram_miss.RECORD_ID_MAX + 1)


def test_a_row_capacity_a_record_slot_cannot_carry_is_refused():
    with pytest.raises(ValueError, match="32767"):
        _device(row_capacities=[5, ram_miss.RECORD_ID_MAX + 1])


def test_a_page_off_16_byte_alignment_is_refused():
    # The post writes the record with 16-byte stores; a misaligned one faults inside the captured graph.
    with pytest.raises(ValueError, match="16-byte"):
        _device(page=torch.zeros(PAGE_BYTES + 1, dtype=torch.uint8)[1:])


@pytest.mark.parametrize("experts, capacity", [(ram_miss.RECORD_ID_MAX + 1, 5), (4, ram_miss.RECORD_ID_MAX + 1)])
def test_the_host_refuses_tables_its_deltas_cannot_carry(experts, capacity):
    # The host writes expert ids and slots into the i16 map delta (Task 7); checked before anything else is read.
    tables = types.SimpleNamespace(
        starts=torch.zeros((1, experts), dtype=torch.int64), capacity=torch.tensor([capacity], dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="32767"):
        ram_miss.ExpertStreamHost(tables, page=torch.zeros(PAGE_BYTES, dtype=torch.uint8),
                                  slot_map=torch.full((1, experts), -1, dtype=torch.int32))
```

Add `import types` to the file's imports.

- [ ] **Step 2: Run them to see them fail** (divix01 worktree, after push; or laptop if the module imports)

Run: `PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly -k "record_id or record_slot or alignment or deltas_cannot"`
Expected: FAIL; `RECORD_ID_MAX` does not exist.

- [ ] **Step 3: Implement**

Next to `RECORD_FLAG_CAPTURED = 1` in `expert_stream_transport.py`:

```python
RECORD_ID_MAX = 32767  # experts, slots and destinations are i16 in the record
```

In `ExpertStreamDevice.__init__`, right after the `timeout_ms <= 0` check:

```python
        if page.data_ptr() % 16:
            raise ValueError("page must be 16-byte aligned: the post writes the record with 16-byte stores")
        if experts > RECORD_ID_MAX:
            raise ValueError(f"{experts} experts: a demand record carries expert ids up to {RECORD_ID_MAX}")
        if any(int(c) > RECORD_ID_MAX for c in row_capacities):
            raise ValueError(f"row capacities {list(row_capacities)}: a demand record carries slots up to {RECORD_ID_MAX}")
```

At the very top of `ExpertStreamHost.__init__` (before the page check, so a bad table fails before anything else is
read):

```python
        experts, capacity = int(tables.starts.shape[1]), int(tables.capacity.max())
        if experts > RECORD_ID_MAX or capacity > RECORD_ID_MAX:
            raise ValueError(
                f"{experts} experts, row capacity {capacity}: records and map deltas carry ids up to {RECORD_ID_MAX}"
            )
```

- [ ] **Step 4: Run them to see them pass, plus the whole file**

Run: `... -m pytest test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly; echo "EXIT=$?"`
Expected: PASS, EXIT=0.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/ops/moe/expert_stream_transport.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py
git commit -m "$(cat <<'EOF'
expert-stream(record-narrow): refuse ids a narrow record cannot carry and a page off 16-byte alignment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: The narrow record, on every writer and reader

One commit: the parity test requires the C++ and Python layouts to agree, and the host and device must agree with
each other.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h:14-33,77-78`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (new store helpers after
  `st_release_sys64`, `:53-55`; `RecordFields` and `write_record`, `:136-164`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh:187-190` (the `write_record` call)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h` (`read_record`, `:129-162`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (`seqlock_stress`, `:567-605`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`RECORD_FIELDS`, `LANE_BYTES`, `LANE_FIELDS`, ~`:684-691`)
- Modify: `python/sglang/test/dsv41_chain_sim.py` (imports `:18-32`, helpers `:37-46`, the record write `:194-216`)
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py` (`PYTHON_WIRE`)
- Modify: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (imports `:24-34`; the record checks `:206-216`)
- Modify: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py:63-66`
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md:34-47` (the record table)

**Interfaces:**
- Consumes: `RECORD_ID_MAX` (Task 2).
- Produces: `RECORD_FIELDS` keys `seq, row, counts, flags, chain, epoch, kinds, protect, lane_expert, lane_slot,
  lane_dst, lane_weight`; `LANE_BYTES` and `LANE_FIELDS` are removed. C++: `kRecCounts`, `kRecLaneExpert`,
  `kRecLaneSlot`, `kRecLaneDst`, `kRecLaneWeight`, `kRecIdMax`; `kRecCount`, `kRecChainHi`, `kRecProtectCount`,
  `kRecLanes`, `kLaneBytes`, `kLaneExpert`, `kLaneSlot`, `kLaneDst`, `kLaneWeight` are removed.
  `struct RecordFields` (in `lease_device.cuh`) holds the record's payload as the post knows it, and the writer
  becomes `write_record(uint8_t* record, uint32_t seq, const RecordFields& fields)`. Its only caller is the post
  kernel (`lease_kernels.cuh:187-190`).

- [ ] **Step 1: Write the failing parity expectation**

In `PYTHON_WIRE` (`test_exl3_ram_miss_device_args.py`), replace every `kRec*`/`kLane*` entry with:

```python
    "kRecSeq": ram_miss.RECORD_FIELDS["seq"],
    "kRecRow": ram_miss.RECORD_FIELDS["row"],
    "kRecCounts": ram_miss.RECORD_FIELDS["counts"],
    "kRecFlags": ram_miss.RECORD_FIELDS["flags"],
    "kRecFlagCaptured": ram_miss.RECORD_FLAG_CAPTURED,
    "kRecChain": ram_miss.RECORD_FIELDS["chain"],
    "kRecEpoch": ram_miss.RECORD_FIELDS["epoch"],
    "kRecKinds": ram_miss.RECORD_FIELDS["kinds"],
    "kRecProtect": ram_miss.RECORD_FIELDS["protect"],
    "kRecLaneExpert": ram_miss.RECORD_FIELDS["lane_expert"],
    "kRecLaneSlot": ram_miss.RECORD_FIELDS["lane_slot"],
    "kRecLaneDst": ram_miss.RECORD_FIELDS["lane_dst"],
    "kRecLaneWeight": ram_miss.RECORD_FIELDS["lane_weight"],
    "kRecIdMax": ram_miss.RECORD_ID_MAX,
```

and in `expert_stream_transport.py` replace `RECORD_FIELDS`, `LANE_BYTES` and `LANE_FIELDS` with:

```python
RECORD_FIELDS = {
    "seq": 0, "row": 4, "counts": 6, "flags": 7, "chain": 8, "epoch": 16, "kinds": 20,
    "protect": 32, "lane_expert": 48, "lane_slot": 64, "lane_dst": 80, "lane_weight": 96,
}
```

Update the comment above it: each record carries `MAX_IDS` lanes as one i16 array per id field and an f32 weight
array; the counts byte holds the lane count (bits 0-3) and the protect count (bits 4-7), and the kinds word holds a
nibble per lane.

- [ ] **Step 2: Run the parity test to see it fail**

Run: `... -m pytest test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly -k wire_header`
Expected: FAIL; the C++ header still has the old names.

- [ ] **Step 3: The layout header**

In `lease_layout.h`, replace `kRecSeq` through `kRecKinds` (lines 16-32; keep `kRecordBytes`, `kMaxIds` and
`kRecFlagCaptured`) with:

```cpp
constexpr int64_t kRecSeq = 0;            // u32 seqlock word: 0 while the payload is rewritten, the seq stored last
constexpr int64_t kRecRow = 4;            // u16
constexpr int64_t kRecCounts = 6;         // u8: lanes in bits 0-3, protect ids in bits 4-7
constexpr int64_t kRecFlags = 7;          // u8
constexpr uint32_t kRecFlagCaptured = 1;  // posted from a captured graph
constexpr int64_t kRecChain = 8;          // u64: the row's map-chain number, 0 when no lane misses
constexpr int64_t kRecEpoch = 16;         // u32: the device's epoch, so G = epoch << 32 | seq
constexpr int64_t kRecKinds = 20;         // u32: lane j's kKind in bits 4j..4j+3
constexpr int64_t kRecProtect = 32;       // i16[kMaxIds]: every routed expert of the request, -1 past the count
constexpr int64_t kRecLaneExpert = 48;    // i16[kMaxIds], -1 past the count
constexpr int64_t kRecLaneSlot = 64;      // i16[kMaxIds]: the RAM slot of a hit, the staging slot of a miss
constexpr int64_t kRecLaneDst = 80;       // i16[kMaxIds]: the VRAM destination slot
constexpr int64_t kRecLaneWeight = 96;    // f32[kMaxIds]: the lane expert's routing weight
constexpr int64_t kRecIdMax = 32767;      // the largest expert, slot or destination an i16 field carries
```

Replace the two record static_asserts (`:77-78`) with:

```cpp
static_assert(kMaxIds == 8, "the record's 16-byte stores and its kinds word hold 8 lanes");
static_assert(kRecCounts == kRecRow + 2 && kRecFlags == kRecRow + 3, "row, counts and flags are one u32 store");
static_assert(kRecChain % 8 == 0 && kRecKinds == kRecEpoch + 4, "chain is one v2 store; epoch and kinds one v4");
static_assert(kDemandRing % 16 == 0 && kRecordBytes % 16 == 0 && kRecEpoch % 16 == 0 && kRecProtect % 16 == 0 &&
                  kRecLaneExpert % 16 == 0 && kRecLaneSlot % 16 == 0 && kRecLaneDst % 16 == 0 &&
                  kRecLaneWeight % 16 == 0,
              "the record's v4 stores are 16-byte aligned");
static_assert(kRecLaneWeight + 4 * kMaxIds <= kRecordBytes, "record");
```

- [ ] **Step 4: The device writer**

In `lease_device.cuh`, after `st_release_sys64`:

```cpp
SGL_DEVICE void st_relaxed_sys_v2(uint8_t* address, uint32_t x, uint32_t y) {
  asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(address), "r"(x), "r"(y) : "memory");
}

SGL_DEVICE void st_relaxed_sys_v4(uint8_t* address, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
```

Before `write_record`:

```cpp
// Two record ids as one i16 pair, -1 for none; traps on an id the record cannot carry rather than wrap it.
SGL_DEVICE uint32_t pack_ids(int64_t lo, int64_t hi) {
  if (lo < -1 || lo > kRecIdMax || hi < -1 || hi > kRecIdMax) __trap();
  return (static_cast<uint32_t>(lo) & 0xFFFFu) | (static_cast<uint32_t>(hi) << 16);
}
```

Before `write_record`, the struct it takes (it follows `TypedLanes`, which it points to):

```cpp
// One request's record as the post knows it; write_record narrows and packs it into the wire layout.
struct RecordFields {
  int64_t row;
  uint32_t flags;           // kRecFlag*
  uint64_t chain;           // the row's map-chain number, 0 when no lane misses
  uint32_t epoch;           // so G = epoch << 32 | seq
  const int32_t* protect;   // protect_count routed experts
  int protect_count;
  int64_t count;            // lanes
  const int64_t* planned;   // count lane experts
  const int32_t* dst;       // count VRAM destination slots
  const float* weight;      // count routing weights
  const TypedLanes* lanes;  // count kinds and source slots
};
```

Replace `write_record` with the version below. `seq` stays a separate argument: it is the seqlock word, not part of
the payload. Keep the existing seqlock comment above it.

```cpp
SGL_DEVICE void write_record(uint8_t* record, uint32_t seq, const RecordFields& f) {
  if (f.count > kMaxIds || f.protect_count > kMaxIds) __trap();
  uint32_t protect_w[kMaxIds / 2], expert_w[kMaxIds / 2], slot_w[kMaxIds / 2], dst_w[kMaxIds / 2];
  uint32_t weight_w[kMaxIds];
  uint32_t kinds = 0;
  for (int i = 0; i < kMaxIds; i += 2) {
    const bool a = i < f.count, b = i + 1 < f.count;
    protect_w[i / 2] =
        pack_ids(i < f.protect_count ? f.protect[i] : -1, i + 1 < f.protect_count ? f.protect[i + 1] : -1);
    expert_w[i / 2] = pack_ids(a ? f.planned[i] : -1, b ? f.planned[i + 1] : -1);
    slot_w[i / 2] = pack_ids(a ? f.lanes->slot[i] : -1, b ? f.lanes->slot[i + 1] : -1);
    dst_w[i / 2] = pack_ids(a ? f.dst[i] : -1, b ? f.dst[i + 1] : -1);
  }
  for (int i = 0; i < kMaxIds; ++i) {
    const bool used = i < f.count;
    weight_w[i] = __float_as_uint(used ? f.weight[i] : 0.0f);
    kinds |= (used ? static_cast<uint32_t>(f.lanes->kind[i]) : 0u) << (4 * i);
  }
  const uint32_t counts = static_cast<uint32_t>(f.count) | static_cast<uint32_t>(f.protect_count) << 4;
  const uint32_t head = (static_cast<uint32_t>(f.row) & 0xFFFFu) | counts << 16 | (f.flags & 0xFFu) << 24;
  st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(record + kRecRow, head);
  st_relaxed_sys_v2(
      record + kRecChain, static_cast<uint32_t>(f.chain & 0xFFFFFFFFull), static_cast<uint32_t>(f.chain >> 32));
  st_relaxed_sys_v4(record + kRecEpoch, f.epoch, kinds, 0u, 0u);
  st_relaxed_sys_v4(record + kRecProtect, protect_w[0], protect_w[1], protect_w[2], protect_w[3]);
  st_relaxed_sys_v4(record + kRecLaneExpert, expert_w[0], expert_w[1], expert_w[2], expert_w[3]);
  st_relaxed_sys_v4(record + kRecLaneSlot, slot_w[0], slot_w[1], slot_w[2], slot_w[3]);
  st_relaxed_sys_v4(record + kRecLaneDst, dst_w[0], dst_w[1], dst_w[2], dst_w[3]);
  st_relaxed_sys_v4(record + kRecLaneWeight, weight_w[0], weight_w[1], weight_w[2], weight_w[3]);
  st_relaxed_sys_v4(record + kRecLaneWeight + 16, weight_w[4], weight_w[5], weight_w[6], weight_w[7]);
  st_release_sys(record + kRecSeq, seq);
}
```

The `// Deduced, not st_relaxed_sys<uint8_t>` comment and its `uint8_t` store go away along with the loop.

In `lease_kernels.cuh`, replace the call (`:188-190`) with:

```cpp
  write_record(
      record, seq,
      RecordFields{
          .row = p.row,
          .flags = p.captured != 0 ? kRecFlagCaptured : 0u,
          .chain = chain,
          .epoch = epoch,
          .protect = protect,
          .protect_count = protect_count,
          .count = count,
          .planned = p.planned,
          .dst = p.dst_slots,
          .weight = weight,
          .lanes = &typed,
      });
```

- [ ] **Step 5: The host reader**

In `tier_protocol.h`, replace `read_record`'s body between the first seq check and
`std::atomic_thread_fence(std::memory_order_acquire);`:

```cpp
  uint16_t row;
  uint8_t counts, flags;
  uint64_t chain;
  uint32_t epoch, kinds;
  int16_t protect_ids[kMaxIds], expert[kMaxIds], slot[kMaxIds], dst[kMaxIds];
  float weight[kMaxIds];
  std::memcpy(&row, record + kRecRow, 2);
  std::memcpy(&counts, record + kRecCounts, 1);
  std::memcpy(&flags, record + kRecFlags, 1);
  std::memcpy(&chain, record + kRecChain, 8);
  std::memcpy(&epoch, record + kRecEpoch, 4);
  std::memcpy(&kinds, record + kRecKinds, 4);
  std::memcpy(protect_ids, record + kRecProtect, sizeof(protect_ids));
  std::memcpy(expert, record + kRecLaneExpert, sizeof(expert));
  std::memcpy(slot, record + kRecLaneSlot, sizeof(slot));
  std::memcpy(dst, record + kRecLaneDst, sizeof(dst));
  std::memcpy(weight, record + kRecLaneWeight, sizeof(weight));
  int count = counts & 0xF;
  const int protect = counts >> 4;
  if (count > kLeaseLanes) count = kLeaseLanes + 1;  // judged once the seq re-check says the record is whole
  request->seq = expected;
  request->gen = static_cast<uint64_t>(epoch) << 32 | expected;
  request->row = row;
  request->captured = (flags & kRecFlagCaptured) != 0;
  request->chain = chain;
  request->protect.clear();
  for (int i = 0; i < std::min<int>(protect, kMaxIds); ++i)
    request->protect.push_back(protect_ids[i]);
  request->lanes.clear();
  for (int j = 0; j < std::min<int>(count, kLeaseLanes); ++j) {
    Lane l;
    l.expert = expert[j];
    l.slot = slot[j];
    l.dst = dst[j];
    l.weight = weight[j];
    l.kind = static_cast<uint8_t>((kinds >> (4 * j)) & 0xFu);
    request->lanes.push_back(l);
  }
```

If `FixedVec` has no `clear()` for `protect`, use the same reset `lanes` uses (it calls `clear()` at `:147`).

- [ ] **Step 6: The C++ seqlock stress writer**

In `ffi_test_exports.h` `seqlock_stress`, replace the writer's per-round fields and stores:

```cpp
          const uint16_t row = static_cast<uint16_t>(round), count = count_of(round);
          const uint8_t counts = static_cast<uint8_t>(count << 4);  // protect ids only, no lanes
          const uint8_t flags = static_cast<uint8_t>(round & 1u);
          const int16_t id = static_cast<int16_t>(round & 0x7FFFu);
          store_release(record + kRecSeq, 0u);
          std::atomic_thread_fence(std::memory_order_seq_cst);
          std::memset(record + 4, 0, kRecordBytes - 4);
          std::memcpy(record + kRecRow, &row, 2);
          std::memcpy(record + kRecCounts, &counts, 1);
          std::memcpy(record + kRecFlags, &flags, 1);
          for (int i = 0; i < count; ++i)
            std::memcpy(record + kRecProtect + 2 * i, &id, 2);
```

and in the reader's check, compare `id == static_cast<int16_t>(round & 0x7FFFu)` instead of
`id == static_cast<int32_t>(round)`.

- [ ] **Step 7: The Python simulator's writer**

In `dsv41_chain_sim.py`: drop `LANE_BYTES` and `LANE_FIELDS` from the import; add next to `_i32`:

```python
def _i16(tensor: torch.Tensor, offset: int, count: int = 1) -> torch.Tensor:
    return tensor[offset : offset + 2 * count].view(torch.int16)
```

and replace the record write (from `_i32(self.page, record + f["seq"])[0] = 0` through the end of the lane loop):

```python
        count = len(experts)
        ids = list(dict.fromkeys(int(e) for e in protect))[:MAX_IDS]
        assert count <= 15, "the counts byte holds 4 bits of lane count"

        def padded(values, fill, dtype):
            return torch.tensor(list(values)[:count] + [fill] * (LANES - count), dtype=dtype)

        _i32(self.page, record + f["seq"])[0] = 0
        _u16(self.page, record + f["row"])[0] = row
        self.page[record + f["counts"]] = count | len(ids) << 4
        self.page[record + f["flags"]] = RECORD_FLAG_CAPTURED if captured else 0
        self.page[record + f["chain"] : record + f["chain"] + 8].view(torch.int64)[0] = chain
        _i32(self.page, record + f["epoch"])[0] = _signed32(self.epoch & 0xFFFFFFFF)
        _i32(self.page, record + f["kinds"])[0] = _signed32(sum((int(typed[j]) & 0xF) << 4 * j for j in range(count)))
        _i16(self.page, record + f["protect"], MAX_IDS)[:] = torch.tensor(ids + [-1] * (MAX_IDS - len(ids)), dtype=torch.int16)
        _i16(self.page, record + f["lane_expert"], LANES)[:] = padded(experts, -1, torch.int16)
        _i16(self.page, record + f["lane_slot"], LANES)[:] = padded(slot_list, -1, torch.int16)
        _i16(self.page, record + f["lane_dst"], LANES)[:] = padded(dst, -1, torch.int16)
        lane_weight = record + f["lane_weight"]
        self.page[lane_weight : lane_weight + 4 * LANES].view(torch.float32)[:] = padded(weights, 0.0, torch.float32)
```

The two `_i32(...)` lines after it (the seq and `demand_head`) stay as they are.

- [ ] **Step 8: The manual GPU tests**

`test_exl3_lease_kernels_cuda.py`: drop `LANE_BYTES` and `LANE_FIELDS` from the import, and replace the flags assert,
`lane_field` and the three lane asserts (`:208-216`) with:

```python
        assert int(page[record + RECORD_FIELDS["flags"]]) == RECORD_FLAG_CAPTURED

        def field(name, dtype, n):
            at = record + RECORD_FIELDS[name]
            return page[at : at + n * dtype.itemsize].view(dtype).tolist()

        assert int(page[record + RECORD_FIELDS["counts"]]) & 0xF == 2
        assert field("kinds", torch.int32, 1)[0] == LaneKind.HIT_CPU | LaneKind.HIT_CPU << 4
        assert field("lane_weight", torch.float32, lease.LANES) == [0.25, 0.5 + 0.0625] + [0.0] * (lease.LANES - 2)
        assert field("lane_dst", torch.int16, 2) == [5, 3]
        assert field("lane_expert", torch.int16, lease.LANES) == [9, 5] + [-1] * (lease.LANES - 2)
```

`test_exl3_slot_map_kernels_cuda.py:63-66`: in `_chain_of_last_record`, replace the `lo`/`hi` reads and the
`return hi << 32 | lo` with one u64 read:

```python
    at = record + RECORD_FIELDS["chain"]
    return int(page[at : at + 8].view(torch.int64)[0])
```

- [ ] **Step 9: The protocol doc**

Replace the record table in `analysis/dsv41-drive/LEASE_PROTOCOL.md:34-47` with "The new record" table from this
plan's "Decisions" (offset, field, type), and note that the payload is written as one u32, one v2 and seven v4
relaxed stores between the two seq stores.

- [ ] **Step 10: Laptop check, then commit and push**

```bash
grep -rnE "kRecCount\b|kRecChainHi|kRecProtectCount|kRecLanes|kLaneBytes|kLane(Expert|Slot|Dst|Weight)|LANE_BYTES|LANE_FIELDS|\"chain_hi\"|\"protect_count\"" python test analysis/dsv41-drive/LEASE_PROTOCOL.md
```

Expected: no output. Then:

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py python/sglang/test/dsv41_chain_sim.py \
  test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "$(cat <<'EOF'
expert-stream(record-narrow): i16 ids, packed counts and kinds, struct-of-arrays lanes in 16-byte stores

The post writes its demand record in 9 relaxed stores instead of 56. Weights stay f32.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-record-narrow
```

- [ ] **Step 11: Registered suite on divix01** (worktree `wt-record-narrow` at the pushed commit)

```bash
cd /data/models/slang/nvfp4-work/wt-record-narrow
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: EXIT=0 and the Task 1 baseline counts plus the 5 tests from Task 2. In particular
`test_exl3_ram_miss_tier.py` (the seqlock stress reports 0 torn), `test_exl3_ram_miss_cpu_experts.py` (weights and
slots reach the CPU job), `test_exl3_ram_miss_slot_map.py` (a kind-0 record is malformed) and
`test_expert_stream_sync_primitives.py` (PTX and volatile conventions) must pass.

If anything fails, fix it in a new commit; do not amend.

---


### Task 4: GPU check of the narrow record

**Files:**
- Modify: this plan's "Results" section.

- [ ] **Step 1: Run the GPU check** (Global Constraints) with `<wt>` = `wt-record-narrow` at the Task 3 commit and
  `<label>` = `record`.

Expected:
- In the post kernel, at least 7 `STG.E.128.STRONG.SYS` and at least 1 `STG.E.64.STRONG.SYS`, with no `ATOM` or `CAS`
  loop near them. A CAS loop is the native-sync plan's failure mode for narrow relaxed stores.
- The GPU tests report EXIT=0.
- The hits-mode kernel median is at or below `base`.

Record the numbers in "Results" whatever they are; don't tune here.

- [ ] **Step 2: Commit the results**

```bash
git add docs/superpowers/plans/2026-10-01-expert-stream-record-narrow.md
git commit -m "$(cat <<'EOF'
docs(expert-stream-record-narrow): the narrow record's SASS counts and post-kernel cost

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The map delta, loaded 16 bytes at a time

No wire change. `apply_map_delta` currently reads the delta one word at a time, and each entry's store depends on
its load, so a full delta costs about one PCIe round trip per entry. This task splits it into three steps:
- `await_map_delta`: the tag spin.
- `load_map_delta`: issues every payload load before using any of them.
- `apply_map_delta`: validates and applies from registers.

The row's device-side words go into a `RowMap` struct.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (a load helper after
  `st_relaxed_sys_v4`; `apply_map_delta`, `:166-189`, is replaced)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (the post's delta call, `:110-116`; the
  bulk apply loop, `:216-223`)
- Test: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`

**Interfaces:**
- Consumes: `st_relaxed_sys_v4` (Task 3) sits next to the new load helper.
- Produces, in `lease_device.cuh`:
  - `SGL_DEVICE uint4 ld_relaxed_sys_v4(const uint8_t* address)`
  - `struct RowMap { int32_t* ram_slot; int32_t* staging; int64_t* map_chain; int64_t* map_applied; int64_t experts;
    uint32_t row_capacity; }`, where `map_chain` and `map_applied` point at the row's own word
  - `struct MapDelta { uint32_t count; int32_t staging[kLeaseLanes]; int32_t expert[kDeltaMaxEntries];
    int32_t slot[kDeltaMaxEntries]; }`
  - `SGL_DEVICE bool await_map_delta(const uint8_t* delta, const RowMap& map, uint64_t deadline)`: true when the
    published delta is not applied yet
  - `SGL_DEVICE MapDelta load_map_delta(const uint8_t* delta)`
  - `SGL_DEVICE void apply_map_delta(const MapDelta& d, const RowMap& map)`

  Task 6 uses all of them; Task 7 changes only `load_map_delta`'s body.

- [ ] **Step 1: Write the test** (append to `test_exl3_slot_map_kernels_cuda.py`, after
  `test_post_applies_the_pending_delta_once`)

```python
def test_post_applies_a_full_delta(idle):
    """Every one of DELTA_MAX_ENTRIES entries lands, and the staging slots with them: the delta's loads cover the
    whole record, not just its first words."""
    c = idle
    _post(c, [5])  # applies the attach delta; the miss makes map chain 2
    entries = [(e, e % 8) for e in range(lease.DELTA_MAX_ENTRIES)]
    _write_delta(c, 0, 2, [8, 9, 10, 11, 12, 13], entries)
    kinds, slots = _post(c, [5])
    assert kinds == [LaneKind.HIT_SM] and slots == [5]
    assert c.device_map(0) == [e % 8 for e in range(EXPERTS)]
    assert c.device_staging(0)[:6] == [8, 9, 10, 11, 12, 13]
```

`EXPERTS` (16) equals `DELTA_MAX_ENTRIES`, so every expert gets an entry; slots 0-7 stay below `CAPACITY` (14) and
away from the staging slots 8-13. Add `EXPERTS` to the `lease_chain_rig` import if it is not there.

This test passes on the current code too. It pins the behavior the wide loads must keep, so run it before changing
anything:

Run (divix01, GPU lock, at the Task 4 commit plus this test): `... -m pytest test_exl3_slot_map_kernels_cuda.py -q -p no:randomly -k full_delta`
Expected: PASS.

- [ ] **Step 2: The load helper and the structs**

In `lease_device.cuh`, after `st_relaxed_sys_v4`:

```cpp
SGL_DEVICE uint4 ld_relaxed_sys_v4(const uint8_t* address) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(address) : "memory");
  return v;
}
```

Replace `apply_map_delta` (`:166-189`, with its comment) with:

```cpp
// One row of the device's map bank (ExpertStreamDevice.map_bank), device memory.
struct RowMap {
  int32_t* ram_slot;     // [experts]: the expert's RAM slot, -1 when not resident
  int32_t* staging;      // [kLeaseLanes]: the row's staging slots
  int64_t* map_chain;    // the row's chain word
  int64_t* map_applied;  // the chain number of the row's last applied delta
  int64_t experts;
  uint32_t row_capacity;
};

// A row's map delta in registers: load_map_delta issues every load before it uses any, so their round trips overlap.
struct MapDelta {
  uint32_t count;
  int32_t staging[kLeaseLanes];
  int32_t expert[kDeltaMaxEntries];
  int32_t slot[kDeltaMaxEntries];
};

// Waits, bounded by `deadline`, until the host has published the delta that follows the row's last map chain; true
// when it is not applied yet. The host publishes a chain's delta before it reads that chain's misses, so the wait is
// taken only when the host fell a whole token behind.
SGL_DEVICE bool await_map_delta(const uint8_t* delta, const RowMap& map, uint64_t deadline) {
  const uint64_t want = static_cast<uint64_t>(*map.map_chain);
  while (ld_acquire_sys64(delta + kDeltaTag) != want) {
    if (static_cast<int64_t>(global_ns() - deadline) >= 0) __trap();  // the host never published it
    __nanosleep(256);
  }
  return static_cast<uint64_t>(*map.map_applied) != want;
}

// The delta's payload; only after await_map_delta's acquire of its tag.
SGL_DEVICE MapDelta load_map_delta(const uint8_t* delta) {
  static_assert(kDeltaStaging % 16 == 0 && kDeltaEntries % 16 == 0, "16-byte delta loads");
  static_assert(kLeaseLanes % 4 == 0 && kDeltaMaxEntries % 2 == 0, "whole 16-byte loads");
  constexpr int kStagingLoads = kLeaseLanes / 4;
  constexpr int kEntryLoads = kDeltaMaxEntries / 2;  // {expert, slot} pairs, two a load
  uint4 v[kStagingLoads + kEntryLoads];
  MapDelta d;
  d.count = ld_relaxed_sys<uint32_t>(delta + kDeltaCount);
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i)
    v[i] = ld_relaxed_sys_v4(delta + kDeltaStaging + 16 * i);
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i)
    v[kStagingLoads + i] = ld_relaxed_sys_v4(delta + kDeltaEntries + 16 * i);
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i) {
    d.staging[4 * i] = static_cast<int32_t>(v[i].x);
    d.staging[4 * i + 1] = static_cast<int32_t>(v[i].y);
    d.staging[4 * i + 2] = static_cast<int32_t>(v[i].z);
    d.staging[4 * i + 3] = static_cast<int32_t>(v[i].w);
  }
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i) {
    const uint4 e = v[kStagingLoads + i];
    d.expert[2 * i] = static_cast<int32_t>(e.x);
    d.slot[2 * i] = static_cast<int32_t>(e.y);
    d.expert[2 * i + 1] = static_cast<int32_t>(e.z);
    d.slot[2 * i + 1] = static_cast<int32_t>(e.w);
  }
  return d;
}

// Validates a loaded delta and applies it to the row once: map_applied takes the row's chain number.
SGL_DEVICE void apply_map_delta(const MapDelta& d, const RowMap& map) {
  if (d.count > static_cast<uint32_t>(kDeltaMaxEntries)) __trap();
#pragma unroll
  for (int i = 0; i < kDeltaMaxEntries; ++i) {
    if (static_cast<uint32_t>(i) < d.count) {
      const int32_t expert = d.expert[i];
      const int32_t slot = d.slot[i];
      if (expert < 0 || expert >= map.experts || slot < -1 || slot >= static_cast<int32_t>(map.row_capacity)) __trap();
      map.ram_slot[expert] = slot;
    }
  }
#pragma unroll
  for (int k = 0; k < kLeaseLanes; ++k)
    map.staging[k] = d.staging[k];
  *map.map_applied = *map.map_chain;
}
```

The fixed-count unrolled loops with an `i < d.count` guard keep `MapDelta` in registers. A loop bounded by
`d.count` would index the arrays dynamically and push them to local memory.

- [ ] **Step 3: The two callers**

In the post kernel (`lease_kernels.cuh`), replace the `ram_slot_row`/`staging_row` locals and the `apply_map_delta`
call with:

```cpp
      const RowMap map{
          .ram_slot = p.ram_slot + p.row * p.experts,
          .staging = p.staging + p.row * kLeaseLanes,
          .map_chain = p.map_chain + p.row,
          .map_applied = p.map_applied + p.row,
          .experts = p.experts,
          .row_capacity = p.row_capacity,
      };
      const uint8_t* delta = p.lease + kDeltaBase + p.row * kDeltaStride;
      if (await_map_delta(delta, map, deadline)) apply_map_delta(load_map_delta(delta), map);
```

and pass `map.ram_slot` and `map.staging` to `type_lanes` where it took `ram_slot_row` and `staging_row`.

In `exl3_ram_miss_map_bulk_apply_kernel`, replace the `apply_map_delta(...)` call with:

```cpp
    const RowMap map{
        .ram_slot = p.ram_slot + row * p.experts,
        .staging = p.staging + row * kLeaseLanes,
        .map_chain = p.map_chain + row,
        .map_applied = p.map_applied + row,
        .experts = p.experts,
        .row_capacity = static_cast<uint32_t>(p.row_capacity[row]),
    };
    if (await_map_delta(delta, map, global_ns())) apply_map_delta(load_map_delta(delta), map);
```

The tag check just above it stays, so `await_map_delta` returns at once there, as `apply_map_delta` did before.

- [ ] **Step 4: Commit, push, and run the registered suite**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py
git commit -m "$(cat <<'EOF'
expert-stream(record-narrow): the post loads a map delta with 16-byte loads issued together

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-record-narrow
```

Run the registered suite (Task 3 Step 11's command) in a fresh divix01 worktree `wt-record-narrow-delta` at this
commit. Expected: EXIT=0 and the same counts as Task 3. `test_expert_stream_sync_primitives.py` must pass: the new
`asm volatile(` is on one line in `lease_device.cuh`.

- [ ] **Step 5: Run the GPU check** with `<wt>` = `wt-record-narrow-delta` and `<label>` = `delta-wide`.

Expected:
- The post kernel has at least 10 `LDG.E.128.STRONG.SYS` (2 staging + 8 entries).
- The GPU tests report EXIT=0, including `test_post_applies_a_full_delta`, `test_post_applies_the_pending_delta_once`,
  `test_post_types_lanes_like_the_reference` and `test_post_waits_for_delta_then_traps_at_deadline`.
- The delta-mode kernel median drops toward the hits-mode median.

Record the numbers in "Results".

---

### Task 6: Lane typing reads no host memory and batches its loads

`type_lanes` takes 16 positional arguments, six of them bools that compile fine in any order. It reads one host
word, `split[n]`, after the copy-armed acquire. It also interleaves dependent GPU loads with its trap checks, so each
lane's map load waits for its plan load.

After this task:
- The post loads the split table before the tag spin. The host stores it relaxed, at any time, ordered by nothing:
  `ram_tier.h` `set_cpu_split`, "Any time; the device reads each entry once per post".
- The post issues the copy-armed acquire after the delta's loads, so those loads are already in flight.
- `type_lanes` takes three structs and loads plan, map and staging before deciding anything.

The semantics stay `ram_slot_map.type_lanes`, and the reference test guards that.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (`type_lanes`, `:191-236`, is replaced)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (the post kernel's thread-0 block,
  `:106-121`)
- Test: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`

**Interfaces:**
- Consumes: `RowMap`, `MapDelta`, `await_map_delta`, `load_map_delta`, `apply_map_delta`, `ld_relaxed_sys_v4`
  (Task 5).
- Produces:
  - `struct LanePlan { const int64_t* planned; const int32_t* dst; int64_t count; }`
  - `struct LanePolicy { bool host_lanes, hit_copy_ce, cpu_on, cpu_misses, ce_ok, cpu_ok; int32_t dst_rows;
    int32_t split[kLeaseLanes + 1]; }`
  - `SGL_DEVICE void load_split(const uint8_t* split, int32_t (&out)[kLeaseLanes + 1])`
  - `SGL_DEVICE void type_lanes(const LanePlan&, const RowMap&, const LanePolicy&, TypedLanes& out)`

- [ ] **Step 1: Write the trap test** (append to `test_exl3_slot_map_kernels_cuda.py`)

```python
_REPEATED_EXPERT_SCRIPT = """
import sys
import torch
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
c = Chain(sys.argv[1], start=False)
c.plan([5, 6])
b, p = c.backends[0], c.plans[0]
b._stage_planned(p)
b.planned[1] = 5  # a repeated expert: the reference raises, so the post must trap
c.dev.post(0, b.planned, p.count, b.routes, p.slots)
try:
    torch.cuda.synchronize()
    print("reached", flush=True)
except RuntimeError as error:
    print(f"trapped {error}", flush=True)
import os
os._exit(0)
"""


def test_post_traps_on_a_repeated_expert(tmp_path):
    """Review Focus 5: once type_lanes loads every lane before deciding, a plan the reference rejects still traps
    instead of being typed."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_REPEATED_EXPERT_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=120,
    )
    assert "reached" not in result.stdout, result.stdout
    assert any(line.startswith("trapped") for line in result.stdout.splitlines()), (
        result.returncode, result.stdout[-2000:], result.stderr[-2000:])
```

Run it on the Task 5 commit plus this test (divix01, GPU lock): `... -m pytest test_exl3_slot_map_kernels_cuda.py -q -p no:randomly -k repeated_expert`
Expected: PASS. It pins existing behavior, which Steps 2-3 must keep.

- [ ] **Step 2: The structs, the split load, and `type_lanes`**

In `lease_device.cuh`, replace `type_lanes` (keep its leading comment, adding "It reads no host memory: the caller
loads the split table into the policy.") with:

```cpp
// The plan as the post reads it: count planned experts and their VRAM destination slots, device memory.
struct LanePlan {
  const int64_t* planned;
  const int32_t* dst;
  int64_t count;
};

// Everything besides the map that decides a lane's kind, loaded before type_lanes runs.
struct LanePolicy {
  bool host_lanes;   // a captured post while the copy engine is armed (kCopyArmed)
  bool hit_copy_ce;  // SGLANG_DSV41_RAM_HIT_COPY=ce
  bool cpu_on;
  bool cpu_misses;
  bool ce_ok;   // the row's copy table is set
  bool cpu_ok;  // the row's CPU layer is registered
  int32_t dst_rows;
  int32_t split[kLeaseLanes + 1];  // kSplit: CPU lanes per n eligible lanes
};

// kSplit's table in three 16-byte loads; the last three words loaded lie past the table and are dropped.
SGL_DEVICE void load_split(const uint8_t* split, int32_t (&out)[kLeaseLanes + 1]) {
  static_assert(kSplit % 16 == 0 && kSplit + 48 <= kLeaseBlockBytes && kLeaseLanes + 1 <= 12, "three loads");
  uint4 v[3];
#pragma unroll
  for (int i = 0; i < 3; ++i)
    v[i] = ld_relaxed_sys_v4(split + 16 * i);
  const uint32_t words[12] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w,
                              v[2].x, v[2].y, v[2].z, v[2].w};
#pragma unroll
  for (int n = 0; n <= kLeaseLanes; ++n)
    out[n] = static_cast<int32_t>(words[n]);
}

SGL_DEVICE void type_lanes(const LanePlan& plan, const RowMap& map, const LanePolicy& policy, TypedLanes& out) {
  if (plan.count > kMaxIds) __trap();
  int64_t expert[kMaxIds];
  int32_t dst[kMaxIds];
  int32_t ram[kMaxIds];
  int32_t staging[kLeaseLanes];
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j) {
    expert[j] = j < plan.count ? plan.planned[j] : 0;
    dst[j] = j < plan.count ? plan.dst[j] : -1;
  }
#pragma unroll
  for (int k = 0; k < kLeaseLanes; ++k)
    staging[k] = map.staging[k];
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j)
    if (j < plan.count && (expert[j] < 0 || expert[j] >= map.experts)) __trap();
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j)
    ram[j] = j < plan.count ? map.ram_slot[expert[j]] : -1;
  bool hit[kMaxIds];
  bool eligible[kMaxIds];
  int m = 0;
  int n = 0;
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j) {
    if (j >= plan.count) break;
    for (int i = 0; i < j; ++i)
      if (expert[i] == expert[j]) __trap();
    hit[j] = ram[j] >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(ram[j]) >= map.row_capacity) __trap();
      out.slot[j] = ram[j];
    } else {
      if (m >= kLeaseLanes || staging[m] < 0) __trap();
      out.slot[j] = staging[m++];
    }
    eligible[j] = policy.host_lanes && policy.cpu_on && policy.cpu_ok && (hit[j] || policy.cpu_misses);
    n += eligible[j] ? 1 : 0;
  }
  int take = n > 0 ? policy.split[n] : 0;
  if (take < 0 || take > n) __trap();
  const bool copy_ok = policy.host_lanes && policy.hit_copy_ce && policy.ce_ok;
  for (int64_t j = plan.count - 1; j >= 0; --j) {
    const bool cpu = take > 0 && eligible[j];
    if (cpu) --take;
    uint8_t kind;
    if (cpu) {
      kind = hit[j] ? kKindHitCpu : kKindMissCpu;
    } else if (hit[j]) {
      kind = copy_ok && dst[j] >= 0 && dst[j] < policy.dst_rows ? kKindHitCopy : kKindHitSm;
    } else {
      kind = kKindMissGpu;
    }
    out.kind[j] = kind;
  }
}
```

Every bounds check that guards an address (`expert[j]` before `map.ram_slot[expert[j]]`) still runs before that
load. Only the order among independent loads changes, and the set of conditions that trap is the reference's, as
before.

- [ ] **Step 3: The post kernel's thread-0 block**

Replace the `if (count > 0) { ... }` block from Task 5 with:

```cpp
    if (count > 0) {
      const RowMap map{
          .ram_slot = p.ram_slot + p.row * p.experts,
          .staging = p.staging + p.row * kLeaseLanes,
          .map_chain = p.map_chain + p.row,
          .map_applied = p.map_applied + p.row,
          .experts = p.experts,
          .row_capacity = p.row_capacity,
      };
      LanePolicy policy{
          .host_lanes = false,
          .hit_copy_ce = p.hit_copy_ce != 0,
          .cpu_on = p.cpu_on != 0,
          .cpu_misses = p.cpu_misses != 0,
          .ce_ok = p.ce_ok[p.row] != 0,
          .cpu_ok = p.cpu_ok[p.row] != 0,
          .dst_rows = p.dst_rows[p.row],
      };
      // Before the tag spin: the host stores split relaxed at any time, ordered by nothing (ram_tier.h set_cpu_split).
      load_split(p.lease + kSplit, policy.split);
      const uint8_t* delta = p.lease + kDeltaBase + p.row * kDeltaStride;
      const bool pending = await_map_delta(delta, map, deadline);
      MapDelta d;
      if (pending) d = load_map_delta(delta);
      // After the tag's acquire, as before, and issued while the delta's loads are in flight.
      policy.host_lanes = p.captured != 0 && ld_acquire_sys(p.lease + kCopyArmed) == 1u;
      if (pending) apply_map_delta(d, map);
      type_lanes(LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = count}, map, policy, typed);
      for (int64_t j = 0; j < count; ++j)
        any_cpu |= is_cpu_kind(typed.kind[j]) ? 1 : 0;
    }
```

`typed` is `__shared__`; `type_lanes` writes it through the reference, as before.

- [ ] **Step 4: Commit, push, and run the registered suite**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py
git commit -m "$(cat <<'EOF'
expert-stream(record-narrow): lane typing takes a plan, a row map and a policy, and reads no host memory

The split table loads before the delta's tag spin; the copy-armed acquire is issued behind the delta's loads.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-record-narrow
```

Run the registered suite in a fresh divix01 worktree `wt-record-narrow-typing` at this commit. Expected: EXIT=0 and
the Task 5 counts.

- [ ] **Step 5: Run the GPU check** with `<wt>` = `wt-record-narrow-typing` and `<label>` = `typing`.

Expected:
- The post kernel has at least 13 `LDG.E.128.STRONG.SYS` (3 split + 10 delta).
- The GPU tests report EXIT=0. `test_post_types_lanes_like_the_reference` (200 random maps and plans, 4 parameter
  sets) and `test_post_traps_on_a_repeated_expert` are the ones this task can break.
- Both modes' kernel medians are at or below `delta-wide`.

Record the numbers in "Results".

---

### Task 7: The map delta in i16

The wire change for the delta. Expert, slot and staging go to i16, so the payload is one u32 (count) plus five
16-byte loads instead of ten. Task 2's refusals (experts and capacities ≤ 32767, on both the host and the device)
cover every value the host writes.

New delta layout (offsets within a row's 256-byte delta record):

| Offset | Field | Type |
|---|---|---|
| 0 | `tag` | u64, stored last with a release |
| 8 | `count` | u32 |
| 16 | `staging` | i16[8] |
| 32 | `entries` | {i16 expert, i16 slot}[16] |

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` (`kDeltaStaging`, `kDeltaEntries` and
  their comments, `:70-73`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (`load_map_delta`'s body)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`publish_delta_locked`, `:1036-1050`)
- Modify: `python/sglang/kernels/ops/moe/expert_lease_block.py` (`DELTA_FIELDS` and its comment, `:33-37`)
- Modify: `python/sglang/test/dsv41_chain_sim.py` (`delta`, `:86-93`)
- Modify: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py` (`_write_delta`)
- Modify: `test/manual/dsv41/bench_exl3_post_record.py` (`write_delta`)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md:63` (the delta row)
- Test: `test/registered/unit/kernels/test_exl3_lease_block.py`

**Interfaces:**
- Consumes: `load_map_delta` and `MapDelta` (Task 5); `MapDelta` keeps its int32 fields, so nothing past the load
  changes.
- Produces: `DELTA_FIELDS = {"tag": 0, "count": 8, "staging": 16, "entries": 32}`, with staging and entries i16.

- [ ] **Step 1: Write the failing layout test**

In `test_a_delta_record_fits_its_stride_and_each_starts_a_new_line`, replace the second and third asserts with:

```python
    assert f["staging"] % 16 == 0 and f["entries"] % 16 == 0, "the post reads both with 16-byte loads"
    assert f["entries"] == f["staging"] + 2 * lease.LANES, "i16 staging slots"
    assert f["entries"] + 4 * lease.DELTA_MAX_ENTRIES <= lease.DELTA_STRIDE and lease.DELTA_STRIDE % 128 == 0
```

Run: `... -m pytest test/registered/unit/kernels/test_exl3_lease_block.py -q -p no:randomly`
Expected: FAIL; `entries` is 48, not 32.

- [ ] **Step 2: Python layout and its comment**

```python
# The map delta block, at DELTA_BASE of the same allocation: one DELTA_STRIDE record per row, {u64 tag; u32 count;
# i16 staging[LANES] @16; {i16 expert, i16 slot}[DELTA_MAX_ENTRIES] @32}. The tag is stored last with a release; a
# zero tag is never a written delta.
DELTA_FIELDS = {"tag": 0, "count": 8, "staging": 16, "entries": 32}
```

- [ ] **Step 3: C++ layout**

```cpp
constexpr int64_t kDeltaStaging = 16;  // i16[kLeaseLanes]: the row's staging slots after this delta, -1 past K
constexpr int64_t kDeltaEntries = 32;  // {i16 expert, i16 slot}[kDeltaMaxEntries]: ram_slot[expert] = slot, -1 unmaps
```

Add, beside the other static_asserts:

```cpp
static_assert(kDeltaEntries + 4 * kDeltaMaxEntries <= kDeltaStride, "delta record");
```

- [ ] **Step 4: The host writer**

In `publish_delta_locked`, replace the two loops with:

```cpp
    for (int k = 0; k < kLeaseLanes; ++k) {
      const int16_t slot = static_cast<int16_t>(k < static_cast<int>(staging.size()) ? staging[k] : -1);
      std::memcpy(d + kDeltaStaging + 2 * k, &slot, 2);
    }
    for (int i = 0; i < count; ++i) {
      const int16_t entry[2] = {static_cast<int16_t>(entries[i][0]), static_cast<int16_t>(entries[i][1])};
      std::memcpy(d + kDeltaEntries + 4 * i, entry, 4);
    }
```

- [ ] **Step 5: The device load**

Replace `load_map_delta`'s body after `d.count = ...`:

```cpp
  static_assert(kDeltaStaging % 16 == 0 && kDeltaEntries % 16 == 0, "16-byte delta loads");
  static_assert(kLeaseLanes == 8 && kDeltaMaxEntries % 4 == 0, "staging is one load; entries four to a load");
  constexpr int kEntryLoads = kDeltaMaxEntries / 4;
  uint4 v[1 + kEntryLoads];
  v[0] = ld_relaxed_sys_v4(delta + kDeltaStaging);
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i)
    v[1 + i] = ld_relaxed_sys_v4(delta + kDeltaEntries + 16 * i);
  const auto lo = [](uint32_t w) { return static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu)); };
  const auto hi = [](uint32_t w) { return static_cast<int32_t>(static_cast<int16_t>(w >> 16)); };
  const uint32_t staging_words[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    d.staging[2 * k] = lo(staging_words[k]);
    d.staging[2 * k + 1] = hi(staging_words[k]);
  }
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i) {
    const uint32_t words[4] = {v[1 + i].x, v[1 + i].y, v[1 + i].z, v[1 + i].w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      d.expert[4 * i + w] = lo(words[w]);
      d.slot[4 * i + w] = hi(words[w]);
    }
  }
  return d;
```

Delete the Task 5 static_asserts and `kStagingLoads` that this body replaces.

- [ ] **Step 6: The Python reader and the two test writers**

`dsv41_chain_sim.py` `delta`:

```python
        base = lease.DELTA_BASE + row * lease.DELTA_STRIDE
        f = lease.DELTA_FIELDS
        tag = self.read_u64(base + f["tag"])
        count = int(_i32(self.block, base + f["count"])[0])
        staging = self.block[base + f["staging"] : base + f["staging"] + 2 * LANES].view(torch.int16).tolist()
        flat = self.block[base + f["entries"] : base + f["entries"] + 4 * lease.DELTA_MAX_ENTRIES].view(torch.int16).tolist()
        return tag, staging, [(flat[2 * i], flat[2 * i + 1]) for i in range(count)]
```

`test_exl3_slot_map_kernels_cuda.py` `_write_delta`, the staging and entry stores:

```python
    block[base + f["staging"] : base + f["staging"] + 2 * lease.LANES].view(torch.int16)[:] = torch.tensor(
        list(staging) + [-1] * (lease.LANES - len(staging)), dtype=torch.int16)
    for i, (expert, slot) in enumerate(entries):
        block[base + f["entries"] + 4 * i : base + f["entries"] + 4 * i + 4].view(torch.int16)[:] = torch.tensor(
            [expert, slot], dtype=torch.int16)
```

`bench_exl3_post_record.py` `write_delta`, the staging and entries stores:

```python
    block[base + f["staging"] : base + f["staging"] + 2 * lease.LANES].view(torch.int16)[:] = torch.tensor(
        staging, dtype=torch.int16)
    block[base + f["entries"] : base + f["entries"] + 2 * len(flat)].view(torch.int16)[:] = torch.tensor(
        flat, dtype=torch.int16)
```

- [ ] **Step 7: The protocol doc**

In `LEASE_PROTOCOL.md:63`, change the delta row to: `tag` u64 @0, `count` u32 @8, `staging` i16[8] @16, 16
`{i16 expert, i16 slot}` entries @32. Add one sentence that the post reads the payload with one u32 load and five
16-byte loads, issued together after the tag's acquire.

- [ ] **Step 8: Run the layout tests, then commit and push**

Run: `... -m pytest test/registered/unit/kernels/test_exl3_lease_block.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly; echo "EXIT=$?"`
Expected: PASS, EXIT=0. The parity test picks up the new `kDeltaStaging`/`kDeltaEntries` through `DELTA_FIELDS`.

```bash
grep -rn "kDeltaEntries + 8\|kDeltaStaging + 4\|DELTA_FIELDS\[\"entries\"\] + 8" python test
```

Expected: no output.

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/ops/moe/expert_lease_block.py python/sglang/test/dsv41_chain_sim.py \
  test/registered/unit/kernels/test_exl3_lease_block.py test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py \
  test/manual/dsv41/bench_exl3_post_record.py analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "$(cat <<'EOF'
expert-stream(record-narrow): the map delta carries i16 slots and experts; the post reads it in five 16-byte loads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push origin expert-stream-record-narrow
```

- [ ] **Step 9: Registered suite and GPU check**

Run the registered suite in a fresh divix01 worktree `wt-record-narrow-final` at this commit (EXIT=0, Task 6 counts).
Then run the GPU check with `<wt>` = `wt-record-narrow-final` and `<label>` = `final`.

Expected:
- The post kernel has at least 8 `LDG.E.128.STRONG.SYS` (3 split + 5 delta).
- The GPU tests report EXIT=0. `test_post_applies_a_full_delta` is the one that catches a wrong i16 unpack.
- The delta-mode median is at or below `typing`.

---

### Task 8: Results and cleanup

**Files:**
- Modify: this plan's "Results" section.

- [ ] **Step 1: Summarize**

Under "Results", add a table with one row per label (`base`, `record`, `delta-wide`, `typing`, `final`). Columns:
- hits-mode kernel median
- delta-mode kernel median
- hits-mode us/post
- `STG.E.128`/`LDG.E.128` counts in the post kernel

Below it, write one line per task on what its change was worth.

- [ ] **Step 2: Remove the divix01 worktrees**

```bash
for wt in base "" delta typing final; do
  git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-record-narrow${wt:+-$wt}
done
```

- [ ] **Step 3: Commit the results**

```bash
git add docs/superpowers/plans/2026-10-01-expert-stream-record-narrow.md
git commit -m "$(cat <<'EOF'
docs(expert-stream-record-narrow): post-kernel cost per step, base to final

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

## Results

(Filled in by Tasks 1, 4, 5, 6, 7 and 8: command, then number.)

Command: `divix01:/data/models/slang/nvfp4-work/record-narrow/gpu_check.sh <wt> <label>`. SASS counts are static, over
the post kernel's non-PDL function section. Kernel medians come from nsys `cuda_gpu_kern_sum` with `--iters 5000`.
us/post is eager wall time, so it includes launch overhead.

Registered suite: `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 python -m pytest
test/registered/unit/kernels -q -p no:randomly`.

**base** (7e9d1ccbfa)
- Hits: median 7776 ns, 15.25 us/post.
- Delta: median 36416 ns, 57.64 us/post.
- SASS: 41 LDG.E.STRONG.SYS, 2 LDG.E.64, 1 STG.E.128, 4 STG.E.STRONG.SYS, 16 STG.E.U8, 5 MEMBAR.ALL.SYS,
  1 MEMBAR.SC.SYS, 6 CCTL.IVALL.
- GPU tests: 19 passed.
- Suite: 1216 passed, 22 skipped, EXIT=0.

**record** (af5b31c5d2)
- Hits: median 7264 ns, 13.57 us/post.
- Delta: median 34240 ns, 54.45 us/post.
- SASS: 41 LDG.E.STRONG.SYS, 2 LDG.E.64, 8 STG.E.128, 1 STG.E.64, 4 STG.E.STRONG.SYS, 16 STG.E.U8,
  5 MEMBAR.ALL.SYS, 1 MEMBAR.SC.SYS, 6 CCTL.IVALL. No ATOM or CAS.
- GPU tests: 22 passed (19 + the three Task 2 refusals).
- Suite: 1218 passed, 22 skipped, EXIT=0.

**delta-wide** (15dc001d0c, Task 5)
- Hits: median 7264 ns, 14.16 us/post. Delta: median 11424 ns, 32.25 us/post.
- SASS: 10 LDG.E.128, 3 LDG.E.STRONG.SYS (was 41), 8 STG.E.128. GPU tests 23 passed; suite 1218 passed, 22 skipped.

**typing** (92d82a481d, Task 6)
- Hits: median 7648 ns, 14.77 us/post. Delta: median 11488 ns, 31.90 us/post.
- SASS: 12 LDG.E.128 (the split table's three loads, the third narrowed to 32 bits), 3 LDG.E.STRONG.SYS, 8 STG.E.128.
- GPU tests 24 passed; suite 1218 passed, 22 skipped. The hits regression is the split table loaded on every post.

**final** (94203ea5be, Task 7)
- Hits: median 7776 ns, 14.39 us/post. Delta: median 10240 ns, 30.15 us/post.
- SASS: 7 LDG.E.128 (5 delta, 2 split), 3 LDG.E.STRONG.SYS, 8 STG.E.128. GPU tests 24 passed; suite 1218 passed, 22 skipped.

**split** (f9a6b0ace4, Task 6 ruling: the split table is read only when a lane can be CPU-eligible)
- Hits: median 6464 ns, 13.74 us/post. Delta: median 8960 ns, 30.60 us/post.
- SASS: 7 LDG.E.128 (static count; the split loads are now predicated), 3 LDG.E.STRONG.SYS, 8 STG.E.128.
- GPU tests 24 passed; suite 1218 passed, 22 skipped, EXIT=0.

| label | hits median | delta median | hits us/post | STG.E.128 / LDG.E.128 |
|---|---|---|---|---|
| base | 7776 ns | 36416 ns | 15.25 | 1 / 0 |
| record | 7264 ns | 34240 ns | 13.57 | 8 / 0 |
| delta-wide | 7264 ns | 11424 ns | 14.16 | 8 / 10 |
| typing | 7648 ns | 11488 ns | 14.77 | 8 / 12 |
| final | 7776 ns | 10240 ns | 14.39 | 8 / 7 |
| split | 6464 ns | 8960 ns | 13.74 | 8 / 7 |

- Task 4 (record in v4 stores): hits -512 ns, delta -2176 ns.
- Task 5 (wide delta loads): delta -22816 ns; the 41 scalar host loads become 10 wide ones.
- Task 6 (typing restructure): neutral on delta; it cost hits +384 ns until the split gate, which took hits to 6464 ns
  (-800 ns against delta-wide). That gain is measured, not attributed: no profile separates it.
- Task 7 (i16 delta): delta -1248 ns, five 16-byte loads instead of ten.
- Base to split: hits -1312 ns (-17%), delta -27456 ns (-75%).

