# N-Lane Lease Layout (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A demand record carries any lane count 1 <= N <= 32, with every wire offset derived from one trait, `LeaseLayout<NumLanes, NumNodes>`, and the runtime choosing N from the model's planned gather width.

**Architecture:** `lease_layout.h` becomes a class template whose `static constexpr` members are every offset and size; each JIT build (device and host) is compiled with `-DSGLANG_EXPERT_STREAM_LANES=N` and uses the alias `Wire = LeaseLayout<N, 1>`. Python mirrors the same formulas in one function, `wire_layout(lanes, nodes)`, checked against a compiled probe of the trait. At N = 8 the wire is byte-identical to v2, so every existing test is the regression gate. Phase 2 (NUMA groups) is a separate plan written after this one lands.

**Tech Stack:** C++20 / CUDA (nvcc with `--expt-relaxed-constexpr`), TVM-FFI JIT modules built by `sglang.kernels.jit.utils.load_jit`, Python 3.13, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-numa-node-distributor-design.md` (Part 1 and Phase 1).

## Global Constraints

- `1 <= NumLanes <= 32`; `kLanes = round_up(NumLanes, 8)`; `NumNodes >= 1` (Phase 1 builds only use 1).
- At `LeaseLayout<8, 1>` every offset equals v2: `kRecordBytes == 128`, `kPageBytes == 2176`, `kRecLaneWeight == 96`, `kLeaseCopyDone == 0x4000`, `kSplit == 16768`, `kLeaseBlockBytes == 20480`, `kDeltaEntries == 32`, `kDeltaMaxEntries == 16`, `kDeltaStride == 256`. Pinned by `static_assert`s in `lease_layout.h` and by the Python test.
- The demand ring stays 16 records; PieceMask piece bits stay 8 (pieces, not lanes).
- N is chosen from the planned gather width (`plan_gather_width`), rounded up to 8; a width above 32 is refused at service start. No env var.
- Code reaches divix01 only by commit and push of branch `numa-node-distributor` and a private worktree (`.claude/rules/divix01-run-protocol.md`). Pushing the branch is externally visible: get the user's OK once before the first push. Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- Comments follow `.claude/rules/comment-style.md`; commits end with the two trailer lines `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft`. Stage files by name.
- Read pytest's status from `PIPESTATUS[0]`, never from a pipeline.

### Run templates (used by every task)

`SYNC` (laptop, after committing):

```bash
git push origin numa-node-distributor
ssh divix01 'set -e; R=/data/models/slang/sglang; W=/data/models/slang/nvfp4-work/wt-nlane;
  git -C $R fetch origin;
  if [ -d $W ]; then git -C $W checkout --detach origin/numa-node-distributor;
  else git -C $R worktree add --detach $W origin/numa-node-distributor; fi;
  git -C $W log -1 --oneline'
```

`RUN_CPU <files...>` (divix01):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest <files...> -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

`RUN_GPU <files...>` (divix01, under the GPU lock; exit 75 means the lock timed out, not a failure):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh \
  /data/models/slang/.venv/bin/python -m pytest <files...> -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

Before trusting any run, check `sglang.__file__` once per worktree:
`ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang;print(sglang.__file__)"'`
Expected: a path under `wt-nlane/python/`.

### Baseline (before Task 1)

- [ ] **Step B1:** `SYNC`, then record the baseline counts of the suites this plan touches:

```
RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/registered/unit/kernels/test_exl3_lease_block.py test/registered/unit/kernels/test_exl3_ram_miss_read_record.py test/registered/unit/kernels/test_ram_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_expert_stream_build_variants.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_tier.py test/registered/unit/kernels/test_exl3_ram_miss_wrap.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py
RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_cpu_split_calibration_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py
```

Write both pass/skip/fail counts into the ledger as `Baseline:`. Every later "existing suites" step compares against these counts. Call this file list `SUITE_CPU` / `SUITE_GPU`.

## Review Focus

1. **Calibration at large N** (`split_calibration.h`, `service.py` calibrate): the grid grows from 45 cells to about 590 at N=32 and needs N RAM slots and N experts of scratch. A tier with fewer than N slots in a row must skip calibration and use the configured split, not fail the launch. Pinned in Task 5.
2. **Seqlock over a multi-line record**: at N=32 a record is 512 bytes (8 lines), so a torn read is likelier. `read_record` must still reject every torn copy. Pinned in Task 3 (`seqlock_stress` at N=32).
3. **CPU lanes above bit 7**: the old `ce_mask`/`cpu_lanes` words packed 8 lanes. A CPU lane at index >= 8 must reach the route tables and the direct gather. Pinned in Task 4.
4. **The eager path's `planned` padding** (`exl3_ram_miss.py:555`): it pads to `max(capacity, MAX_IDS)`. At N=16, an eager post with 9-16 lanes must not trap. Pinned in Task 6.
5. **A width planned between 9 and 32 on a small tier**: `reserve_staging(min(N, capacity - 1))` must still leave at least one fill slot, and a wider post traps on "no staging slot" exactly as at N=8. Pinned in Task 6.

---

### Task 1: `LeaseLayout` trait, Python `wire_layout`, probe test

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` (whole file)
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout_probe.cpp`
- Modify: `python/sglang/kernels/ops/moe/expert_lease_block.py` (add `WireLayout`, `wire_layout`, `wire_probe`)
- Create: `test/registered/unit/kernels/test_expert_stream_lease_layout.py`
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py` (drop the constexpr parser checks of the wire header)

**Interfaces:**
- Produces (C++): `template <int NumLanes, int NumNodes = 1> struct sglang::expert_stream::wire::LeaseLayout` with the members listed in Step 3; `using Wire = LeaseLayout<SGLANG_EXPERT_STREAM_LANES, 1>;`; `wire_round_up(int64_t, int64_t)`.
- Produces (C++): during Tasks 1-2 only, the old free names stay as aliases (`inline constexpr auto kRecSeq = Wire::kRecSeq;` ...), so every other file still compiles. Task 2 deletes them.
- Produces (Python): `expert_lease_block.wire_layout(lanes: int, nodes: int = 1) -> WireLayout` (cached, frozen dataclass); `WireLayout.cpp_constants() -> dict[str, int]`; `expert_lease_block.wire_probe(lanes: int, nodes: int) -> dict[str, int]`, which builds and calls the probe.

- [ ] **Step 1: Write the failing test** `test/registered/unit/kernels/test_expert_stream_lease_layout.py`:

```python
"""LeaseLayout<NumLanes, NumNodes> (lease_layout.h) and its Python mirror wire_layout agree, and (8, 1) is wire v2."""

import pytest

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

GRID = [(lanes, nodes) for lanes in (1, 6, 8, 13, 32) for nodes in (1, 2)]


@pytest.mark.parametrize("lanes, nodes", GRID)
def test_the_python_layout_is_the_cpp_trait(lanes, nodes):
    assert lease.wire_layout(lanes, nodes).cpp_constants() == lease.wire_probe(lanes, nodes)


def test_eight_lanes_on_one_node_is_wire_v2():
    w = lease.wire_layout(8)
    assert (w.record_bytes, w.page_bytes, w.lease_block_bytes) == (128, 2176, 20480)
    assert w.record_fields == {
        "seq": 0, "row": 4, "counts": 6, "flags": 7, "chain": 8, "epoch": 16, "kinds": 20,
        "protect": 32, "lane_expert": 48, "lane_slot": 64, "lane_dst": 80, "lane_weight": 96,
    }
    assert (w.copy_done, w.copy_gate, w.copy_armed, w.split) == (0x4000, 0x4080, 0x4100, 16768)
    assert (w.delta_fields, w.delta_max_entries, w.delta_stride) == (
        {"tag": 0, "count": 8, "staging": 16, "entries": 32}, 16, 256,
    )
    assert w.packed_counts


@pytest.mark.parametrize("lanes, rounded", [(1, 8), (8, 8), (9, 16), (13, 16), (17, 24), (32, 32)])
def test_lanes_round_up_to_eight(lanes, rounded):
    assert lease.wire_layout(lanes).lanes == rounded


@pytest.mark.parametrize("lanes", [0, 33])
def test_a_lane_count_outside_1_to_32_is_refused(lanes):
    with pytest.raises(ValueError, match="1..32"):
        lease.wire_layout(lanes)


def test_wider_records_round_to_whole_line_pairs():
    assert [lease.wire_layout(n).record_bytes for n in (16, 24, 32)] == [256, 384, 512]
    assert not lease.wire_layout(16).packed_counts
```

- [ ] **Step 2: Run it to verify it fails.** Commit the test alone (`git add test/registered/unit/kernels/test_expert_stream_lease_layout.py; git commit -m "test(expert-stream): LeaseLayout trait and its Python mirror (failing)"` with the trailers), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_expert_stream_lease_layout.py`.
  Expected: FAIL, `AttributeError: module ... has no attribute 'wire_layout'`.

- [ ] **Step 3: Replace `lease_layout.h`** with the trait. Keep the file's opening comment block, minus the sentence about the parser ("This file holds only `constexpr <type> kName = <integer expression>;` lines ..."), which no longer holds:

```cpp
#pragma once

#include <cstdint>

// The lane count of this build: every JIT build of the device kernels and the host passes -DSGLANG_EXPERT_STREAM_LANES.
#ifndef SGLANG_EXPERT_STREAM_LANES
#define SGLANG_EXPERT_STREAM_LANES 8
#endif

namespace sglang::expert_stream::wire {

constexpr int64_t wire_round_up(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

template <int NumLanes, int NumNodes = 1>
struct LeaseLayout {
  static_assert(1 <= NumLanes && NumLanes <= 32, "a record carries 1..32 lanes");
  static_assert(NumNodes >= 1, "at least one NUMA node");
  // Lane arrays are i16, so a multiple of 8 lanes is whole 16-byte vector loads and stores.
  static constexpr int kLanes = static_cast<int>(wire_round_up(NumLanes, 8));
  static constexpr int kNodes = NumNodes;

  // ---- Request page: device-written, host-read ----
  static constexpr int64_t kDemandHead = 0;    // u32: the last posted seq, stored with a release
  static constexpr int64_t kDemandRing = 128;  // a 128-byte block of its own
  static constexpr uint32_t kDemandRecords = 16;
  static constexpr int64_t kRecSeq = 0;    // u32 seqlock word: 0 while the payload is rewritten, the seq stored last
  static constexpr int64_t kRecRow = 4;    // u16
  static constexpr int64_t kRecCounts = 6; // u8: see kPackedCounts
  static constexpr int64_t kRecFlags = 7;  // u8
  static constexpr uint32_t kRecFlagCaptured = 1;
  static constexpr int64_t kRecChain = 8;  // u64: the row's map-chain number, 0 when no lane misses
  static constexpr int64_t kRecEpoch = 16; // u32: the device's epoch, so G = epoch << 32 | seq
  static constexpr int64_t kRecKinds = 20; // u32[kKindWords]: lane j's kind in word j / 8, bits 4(j % 8)..+3
  static constexpr int kKindWords = kLanes / 8;
  // v2 packs lanes (bits 0-3) and protect ids (bits 4-7) into kRecCounts; wider records hold the lane count there and
  // the protect count in its own byte.
  static constexpr bool kPackedCounts = kLanes == 8;
  static constexpr int64_t kRecProtectCount = kRecKinds + 4 * kKindWords;  // u8, unpacked counts only
  static constexpr int64_t kRecHeaderBytes = wire_round_up(kRecProtectCount + (kPackedCounts ? 0 : 1), 16);
  static constexpr int64_t kRecProtect = kRecHeaderBytes;         // i16[kLanes]
  static constexpr int64_t kRecLaneExpert = kRecProtect + 2 * kLanes;   // i16[kLanes]
  static constexpr int64_t kRecLaneSlot = kRecLaneExpert + 2 * kLanes;  // i16[kLanes]: a hit's RAM slot, a miss's staging slot
  static constexpr int64_t kRecLaneDst = kRecLaneSlot + 2 * kLanes;     // i16[kLanes]: the VRAM destination slot
  static constexpr int64_t kRecLaneWeight = kRecLaneDst + 2 * kLanes;   // f32[kLanes]
  static constexpr int64_t kRecPayloadEnd = kRecLaneWeight + 4 * kLanes;
  static constexpr int64_t kRecordBytes = wire_round_up(kRecPayloadEnd, 128);  // whole L2 adjacent-line pairs
  static constexpr int64_t kRecIdMax = 32767;
  static constexpr int64_t kPageBytes = kDemandRing + kDemandRecords * kRecordBytes;
  static constexpr uint32_t kKindHitCopy = 1;
  static constexpr uint32_t kKindHitSm = 2;
  static constexpr uint32_t kKindHitCpu = 3;
  static constexpr uint32_t kKindMissGpu = 4;
  static constexpr uint32_t kKindMissCpu = 5;
  static constexpr int64_t kHotHeaderBytes = 8;
  static constexpr int64_t kHotAlignment = 64;
  static constexpr uint32_t kHotRecords = kDemandRecords;

  // ---- Completion block: host-written, device-read, 4096-byte aligned ----
  static constexpr int64_t kLeaseBlockAlign = 4096;
  static constexpr int64_t kLeasePieceMask = 0;  // u64[kDemandRecords][kLanes], one 128-byte line each
  static constexpr int64_t kLeasePieceMaskLineBytes = 128;
  static constexpr int64_t kLeaseCopyDone = kLeasePieceMask + kDemandRecords * kLanes * kLeasePieceMaskLineBytes;
  static constexpr int64_t kLeaseCopyDoneBytes = 8;
  static constexpr int64_t kLeaseCopyGate = kLeaseCopyDone + 128;
  static constexpr uint32_t kLeaseGateClosed = 0x80000001u;
  static constexpr uint32_t kLeaseGateOpen = 1;
  static constexpr uint32_t kLeaseGateSeqShift = 2;
  static constexpr uint32_t kLeaseGateSeqMask = 0x1FFFFFFF;
  static constexpr int64_t kCopyArmed = kLeaseCopyGate + 128;  // u32
  static constexpr int64_t kSplit = kCopyArmed + 128;  // i32[kNodes][kSplitStride / 4]: CPU lanes per n eligible lanes
  static constexpr int64_t kSplitStride = wire_round_up(4 * (kLanes + 1), 16);  // a node's table, in 16-byte loads
  static constexpr int64_t kLeaseBlockBytes = wire_round_up(kSplit + kNodes * kSplitStride, kLeaseBlockAlign);

  // ---- Map delta block: one record per row after the completion block ----
  static constexpr int64_t kDeltaBase = kLeaseBlockBytes;
  static constexpr int64_t kDeltaTag = 0;      // u64, stored last with a release
  static constexpr int64_t kDeltaCount = 8;    // u32
  static constexpr int64_t kDeltaStaging = 16; // i16[kNodes][kLanes], -1 past the list
  static constexpr int64_t kDeltaEntries = wire_round_up(kDeltaStaging + 2 * kNodes * kLanes, 16);
  static constexpr int64_t kDeltaMaxEntries = 2 * kLanes;  // an insert and an eviction per miss
  static constexpr int64_t kDeltaStride = wire_round_up(kDeltaEntries + 4 * kDeltaMaxEntries, 256);
};

using Wire = LeaseLayout<SGLANG_EXPERT_STREAM_LANES, 1>;

using V2 = LeaseLayout<8, 1>;
static_assert(V2::kRecordBytes == 128 && V2::kPageBytes == 2176 && V2::kRecLaneWeight == 96, "v2 request page");
static_assert(V2::kLeaseCopyDone == 0x4000 && V2::kSplit == 16768 && V2::kLeaseBlockBytes == 20480, "v2 block");
static_assert(V2::kDeltaEntries == 32 && V2::kDeltaMaxEntries == 16 && V2::kDeltaStride == 256, "v2 delta");
static_assert(Wire::kSplit % 16 == 0 && Wire::kRecProtect % 16 == 0 && Wire::kRecordBytes % 128 == 0, "alignment");

// Transitional aliases so every user still compiles; Task 2 rewrites the users to Wire:: and deletes this block.
inline constexpr auto kDemandHead = Wire::kDemandHead;
inline constexpr auto kDemandRing = Wire::kDemandRing;
inline constexpr auto kDemandRecords = Wire::kDemandRecords;
inline constexpr auto kRecordBytes = Wire::kRecordBytes;
inline constexpr int kMaxIds = Wire::kLanes;
inline constexpr auto kRecSeq = Wire::kRecSeq;
inline constexpr auto kRecRow = Wire::kRecRow;
inline constexpr auto kRecCounts = Wire::kRecCounts;
inline constexpr auto kRecFlags = Wire::kRecFlags;
inline constexpr auto kRecFlagCaptured = Wire::kRecFlagCaptured;
inline constexpr auto kRecChain = Wire::kRecChain;
inline constexpr auto kRecEpoch = Wire::kRecEpoch;
inline constexpr auto kRecKinds = Wire::kRecKinds;
inline constexpr auto kRecProtect = Wire::kRecProtect;
inline constexpr auto kRecLaneExpert = Wire::kRecLaneExpert;
inline constexpr auto kRecLaneSlot = Wire::kRecLaneSlot;
inline constexpr auto kRecLaneDst = Wire::kRecLaneDst;
inline constexpr auto kRecLaneWeight = Wire::kRecLaneWeight;
inline constexpr auto kRecIdMax = Wire::kRecIdMax;
inline constexpr auto kPageBytes = Wire::kPageBytes;
inline constexpr auto kKindHitCopy = Wire::kKindHitCopy;
inline constexpr auto kKindHitSm = Wire::kKindHitSm;
inline constexpr auto kKindHitCpu = Wire::kKindHitCpu;
inline constexpr auto kKindMissGpu = Wire::kKindMissGpu;
inline constexpr auto kKindMissCpu = Wire::kKindMissCpu;
inline constexpr auto kHotHeaderBytes = Wire::kHotHeaderBytes;
inline constexpr auto kHotAlignment = Wire::kHotAlignment;
inline constexpr auto kHotRecords = Wire::kHotRecords;
inline constexpr int64_t kLeaseRing = Wire::kDemandRecords;
inline constexpr int64_t kLeaseLanes = Wire::kLanes;
inline constexpr auto kLeaseBlockAlign = Wire::kLeaseBlockAlign;
inline constexpr auto kLeasePieceMask = Wire::kLeasePieceMask;
inline constexpr auto kLeasePieceMaskLineBytes = Wire::kLeasePieceMaskLineBytes;
inline constexpr auto kLeaseCopyDone = Wire::kLeaseCopyDone;
inline constexpr auto kLeaseCopyDoneBytes = Wire::kLeaseCopyDoneBytes;
inline constexpr auto kLeaseCopyGate = Wire::kLeaseCopyGate;
inline constexpr auto kLeaseGateClosed = Wire::kLeaseGateClosed;
inline constexpr auto kLeaseGateOpen = Wire::kLeaseGateOpen;
inline constexpr auto kLeaseGateSeqShift = Wire::kLeaseGateSeqShift;
inline constexpr auto kLeaseGateSeqMask = Wire::kLeaseGateSeqMask;
inline constexpr auto kCopyArmed = Wire::kCopyArmed;
inline constexpr auto kSplit = Wire::kSplit;
inline constexpr auto kLeaseBlockBytes = Wire::kLeaseBlockBytes;
inline constexpr auto kDeltaBase = Wire::kDeltaBase;
inline constexpr auto kDeltaStride = Wire::kDeltaStride;
inline constexpr auto kDeltaTag = Wire::kDeltaTag;
inline constexpr auto kDeltaCount = Wire::kDeltaCount;
inline constexpr auto kDeltaStaging = Wire::kDeltaStaging;
inline constexpr auto kDeltaEntries = Wire::kDeltaEntries;
inline constexpr auto kDeltaMaxEntries = Wire::kDeltaMaxEntries;

}  // namespace sglang::expert_stream::wire
```

The old file's `static_assert`s that are true for every N (record vector stores 16-byte aligned, `kSplit` inside the block, `kLeaseBlockBytes % kLeaseBlockAlign == 0`, `kDeltaEntries + 4 * kDeltaMaxEntries <= kDeltaStride`) move inside the struct body; the two that pin `kMaxIds == 8` and `kRecLaneWeight + 4 * kMaxIds == kRecordBytes` are deleted.

- [ ] **Step 4: Create the probe** `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout_probe.cpp`:

```cpp
// Test-only: prints LeaseLayout's members for the lane/node pairs test_expert_stream_lease_layout checks.
#include <sgl_kernel/tensor.h>

#include <string>

#include "lease_layout.h"

namespace {

using namespace ::sglang::expert_stream::wire;

template <class L>
std::string members() {
  std::string out;
  auto put = [&](const char* name, int64_t value) { out += std::string(name) + "=" + std::to_string(value) + "\n"; };
#define P(name) put(#name, static_cast<int64_t>(L::name))
  P(kLanes); P(kNodes); P(kDemandHead); P(kDemandRing); P(kDemandRecords); P(kRecSeq); P(kRecRow); P(kRecCounts);
  P(kRecFlags); P(kRecFlagCaptured); P(kRecChain); P(kRecEpoch); P(kRecKinds); P(kKindWords); P(kPackedCounts);
  P(kRecProtectCount); P(kRecHeaderBytes); P(kRecProtect); P(kRecLaneExpert); P(kRecLaneSlot); P(kRecLaneDst);
  P(kRecLaneWeight); P(kRecPayloadEnd); P(kRecordBytes); P(kRecIdMax); P(kPageBytes); P(kKindHitCopy); P(kKindHitSm);
  P(kKindHitCpu); P(kKindMissGpu); P(kKindMissCpu); P(kHotHeaderBytes); P(kHotAlignment); P(kHotRecords);
  P(kLeaseBlockAlign); P(kLeasePieceMask); P(kLeasePieceMaskLineBytes); P(kLeaseCopyDone); P(kLeaseCopyDoneBytes);
  P(kLeaseCopyGate); P(kLeaseGateClosed); P(kLeaseGateOpen); P(kLeaseGateSeqShift); P(kLeaseGateSeqMask);
  P(kCopyArmed); P(kSplit); P(kSplitStride); P(kLeaseBlockBytes); P(kDeltaBase); P(kDeltaTag); P(kDeltaCount);
  P(kDeltaStaging); P(kDeltaEntries); P(kDeltaMaxEntries); P(kDeltaStride);
#undef P
  out.pop_back();
  return out;
}

template <int N>
std::string for_nodes(int64_t nodes) {
  return nodes == 1 ? members<LeaseLayout<N, 1>>() : members<LeaseLayout<N, 2>>();
}

std::string probe(int64_t lanes, int64_t nodes) {
  if (nodes != 1 && nodes != 2) return "";
  switch (lanes) {
    case 1: return for_nodes<1>(nodes);
    case 6: return for_nodes<6>(nodes);
    case 8: return for_nodes<8>(nodes);
    case 13: return for_nodes<13>(nodes);
    case 32: return for_nodes<32>(nodes);
    default: return "";
  }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lease_layout_probe, probe);
```

- [ ] **Step 5: Add `WireLayout` to `expert_lease_block.py`** (after the imports, before `RING`; the module-level constants below it stay until Task 6, now computed from `wire_layout(8)`):

```python
import functools
from dataclasses import dataclass


def _round_up(value: int, align: int) -> int:
    return -(-value // align) * align


@dataclass(frozen=True)
class WireLayout:
    """lease_layout.h's LeaseLayout<lanes, nodes>: the request page, completion block and map delta offsets."""

    lanes: int
    nodes: int

    # Request page.
    demand_ring = 128
    demand_records = 16
    record_id_max = 32767

    @property
    def kind_words(self) -> int:
        return self.lanes // 8

    @property
    def packed_counts(self) -> bool:
        return self.lanes == 8

    @property
    def protect_count(self) -> int:
        return 20 + 4 * self.kind_words

    @property
    def header_bytes(self) -> int:
        return _round_up(self.protect_count + (0 if self.packed_counts else 1), 16)

    @property
    def record_fields(self) -> dict[str, int]:
        h, n = self.header_bytes, self.lanes
        return {
            "seq": 0, "row": 4, "counts": 6, "flags": 7, "chain": 8, "epoch": 16, "kinds": 20,
            "protect": h, "lane_expert": h + 2 * n, "lane_slot": h + 4 * n, "lane_dst": h + 6 * n,
            "lane_weight": h + 8 * n,
        }

    @property
    def record_bytes(self) -> int:
        return _round_up(self.record_fields["lane_weight"] + 4 * self.lanes, 128)

    @property
    def page_bytes(self) -> int:
        return self.demand_ring + self.demand_records * self.record_bytes

    # Completion block.
    block_align = 4096
    piece_mask = 0
    piece_mask_line_bytes = 128
    copy_done_bytes = 8

    @property
    def copy_done(self) -> int:
        return self.piece_mask + self.demand_records * self.lanes * self.piece_mask_line_bytes

    @property
    def copy_gate(self) -> int:
        return self.copy_done + 128

    @property
    def copy_armed(self) -> int:
        return self.copy_gate + 128

    @property
    def split(self) -> int:
        return self.copy_armed + 128

    @property
    def split_stride(self) -> int:
        return _round_up(4 * (self.lanes + 1), 16)

    @property
    def lease_block_bytes(self) -> int:
        return _round_up(self.split + self.nodes * self.split_stride, self.block_align)

    # Map delta.
    @property
    def delta_fields(self) -> dict[str, int]:
        return {"tag": 0, "count": 8, "staging": 16, "entries": _round_up(16 + 2 * self.nodes * self.lanes, 16)}

    @property
    def delta_max_entries(self) -> int:
        return 2 * self.lanes

    @property
    def delta_stride(self) -> int:
        return _round_up(self.delta_fields["entries"] + 4 * self.delta_max_entries, 256)

    def cpp_constants(self) -> dict[str, int]:
        """The trait's members by their C++ names, as lease_layout_probe prints them."""
        f, d = self.record_fields, self.delta_fields
        return {
            "kLanes": self.lanes, "kNodes": self.nodes, "kDemandHead": 0, "kDemandRing": self.demand_ring,
            "kDemandRecords": self.demand_records, "kRecSeq": f["seq"], "kRecRow": f["row"],
            "kRecCounts": f["counts"], "kRecFlags": f["flags"], "kRecFlagCaptured": 1, "kRecChain": f["chain"],
            "kRecEpoch": f["epoch"], "kRecKinds": f["kinds"], "kKindWords": self.kind_words,
            "kPackedCounts": int(self.packed_counts), "kRecProtectCount": self.protect_count,
            "kRecHeaderBytes": self.header_bytes, "kRecProtect": f["protect"], "kRecLaneExpert": f["lane_expert"],
            "kRecLaneSlot": f["lane_slot"], "kRecLaneDst": f["lane_dst"], "kRecLaneWeight": f["lane_weight"],
            "kRecPayloadEnd": f["lane_weight"] + 4 * self.lanes, "kRecordBytes": self.record_bytes,
            "kRecIdMax": self.record_id_max, "kPageBytes": self.page_bytes, "kKindHitCopy": 1, "kKindHitSm": 2,
            "kKindHitCpu": 3, "kKindMissGpu": 4, "kKindMissCpu": 5, "kHotHeaderBytes": 8, "kHotAlignment": 64,
            "kHotRecords": self.demand_records, "kLeaseBlockAlign": self.block_align,
            "kLeasePieceMask": self.piece_mask, "kLeasePieceMaskLineBytes": self.piece_mask_line_bytes,
            "kLeaseCopyDone": self.copy_done, "kLeaseCopyDoneBytes": self.copy_done_bytes,
            "kLeaseCopyGate": self.copy_gate, "kLeaseGateClosed": GATE["closed"], "kLeaseGateOpen": GATE["open"],
            "kLeaseGateSeqShift": GATE_SEQ_SHIFT, "kLeaseGateSeqMask": GATE_SEQ_MASK, "kCopyArmed": self.copy_armed,
            "kSplit": self.split, "kSplitStride": self.split_stride, "kLeaseBlockBytes": self.lease_block_bytes,
            "kDeltaBase": self.lease_block_bytes, "kDeltaTag": d["tag"], "kDeltaCount": d["count"],
            "kDeltaStaging": d["staging"], "kDeltaEntries": d["entries"],
            "kDeltaMaxEntries": self.delta_max_entries, "kDeltaStride": self.delta_stride,
        }


MAX_LANES = 32


@functools.cache
def wire_layout(lanes: int, nodes: int = 1) -> WireLayout:
    """Return the wire layout for ``lanes`` (rounded up to 8) on ``nodes`` NUMA nodes."""
    if not 1 <= lanes <= MAX_LANES:
        raise ValueError(f"a demand record carries 1..{MAX_LANES} lanes, not {lanes}")
    if nodes < 1:
        raise ValueError(f"the wire needs at least one node, not {nodes}")
    return WireLayout(_round_up(lanes, 8), nodes)


def wire_probe(lanes: int, nodes: int) -> dict[str, int]:
    """Test only: LeaseLayout<lanes, nodes>'s members as the C++ compiler computes them."""
    from sglang.kernels.jit.utils import load_jit

    module = load_jit(
        "expert_stream_lease_layout_probe",
        cpp_files=["moe/expert_stream/lease_layout_probe.cpp"],
        header_only=False,
    )
    text = str(module.expert_stream_lease_layout_probe(lanes, nodes))
    if not text:
        raise ValueError(f"the probe has no LeaseLayout<{lanes}, {nodes}> instantiation")
    return {name: int(value) for name, value in (line.split("=") for line in text.split("\n"))}
```

`GATE`, `GATE_SEQ_SHIFT` and `GATE_SEQ_MASK` are defined later in the module; `cpp_constants` runs only after import, so the forward reference is fine. Then make the existing module constants derive from it (same values as today):

```python
_V2 = wire_layout(8)
RING = _V2.demand_records
LANES = _V2.lanes
BLOCK_ALIGN = _V2.block_align
BLOCK_BYTES = _V2.lease_block_bytes
PIECE_MASK = _V2.piece_mask
PIECE_MASK_LINE_BYTES = _V2.piece_mask_line_bytes
COPY_DONE = _V2.copy_done
COPY_DONE_BYTES = _V2.copy_done_bytes
COPY_GATE = _V2.copy_gate
COPY_ARMED = _V2.copy_armed
SPLIT = _V2.split
DELTA_BASE = _V2.lease_block_bytes
DELTA_STRIDE = _V2.delta_stride
DELTA_FIELDS = _V2.delta_fields
DELTA_MAX_ENTRIES = _V2.delta_max_entries
```

Keep `GATE`, `GATE_SEQ_SHIFT`, `GATE_SEQ_MASK` as literals (lane-independent).

- [ ] **Step 6: Drop the parser checks** from `test_exl3_ram_miss_device_args.py`: delete `_wire`, `PYTHON_WIRE`, `test_the_wire_header_is_the_python_layout`, and the `wire_header` import. Change `test_no_other_source_defines_a_wire_constant` to take its names from the probe:

```python
def test_no_other_source_defines_a_wire_constant():
    """A layout constant re-added beside its user compiles and then drifts; this names the file that re-added it."""
    wire = set(lease.wire_probe(8, 1)) - {"kLanes", "kNodes"}
    for path in (*host_sources(), *device_sources()):
        clash = wire & set(_NAME.findall(path.read_text()))
        assert not clash, f"{path.name} redefines wire constants {sorted(clash)}: define them only in lease_layout.h"
```

Change `test_the_device_state_words_are_the_python_state_words` to seed the parse from the probe instead of the header, `_constants(*device_sources(), known={**lease.wire_probe(8, 1), "kMaxIds": 8, "kLeaseLanes": 8, "kLeaseRing": 16})` (the three old names are still used unprefixed until Task 2), and in `_constants` skip any expression containing `::` (the device sources reference `Wire::` members after Task 2):

```python
    for name, expression in re.findall(pattern, joined_text(paths), re.MULTILINE):
        if "::" in expression:
            continue
```

In `test_hot_sidecar_layout_and_384_expert_size_match_the_native_abi`, replace `wire = _wire()` and its three `wire[...]` comparisons with `wire = lease.wire_probe(8, 1)` and the same three comparisons.

- [ ] **Step 7: Run the tests.** Commit (`git add` the five files; message `feat(expert-stream): LeaseLayout<NumLanes, NumNodes> trait and its Python mirror`), `SYNC`, then:
  `RUN_CPU test/registered/unit/kernels/test_expert_stream_lease_layout.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/registered/unit/kernels/test_exl3_lease_block.py`
  Expected: all pass (the probe compiles once, about 30 s).

- [ ] **Step 8: Existing suites unchanged.** `RUN_CPU SUITE_CPU`. Expected: the Baseline counts, plus this task's new tests.

---

### Task 2: Rewrite every user to `Wire::`, delete the aliases, build per lane count

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` (delete the alias block)
- Modify (mechanical): `expert_stream/lease_device.cuh`, `lease_kernels.cuh`, `row_copy_kernels.cuh`, `host/ram_tier.h`, `host/tier_protocol.h`, `host/copy_engine.h`, `host/cpu_experts.h`, `host/ffi_exports.h`, `host/ffi_test_exports.h`, `host/split_calibration.h`, `host/reader_base.h`, `host/piece_geometry.h`, `bench/src/device_sim.h`, `bench/src/device_sim.cpp`, `bench/src/stack.h`, `bench/src/full_stack.cpp`, `bench/src/self_test.cpp`, `moe/exl3_ram_miss.cuh`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`_host_module*`, `_device_module*`, `device_module_with_hooks` take `lanes`)
- The rename script in Step 3 runs once from the scratchpad and is not committed.

**Interfaces:**
- Consumes: `Wire` and the trait members from Task 1.
- Produces: no free wire constants remain. `kMaxIds`, `kLeaseLanes` are spelled `Wire::kLanes`; `kLeaseRing` is `Wire::kDemandRecords`.
- Produces (Python): `_host_module(layout="exl3", variant=None, lanes=8)`, `_device_module(layout="exl3", lanes=8)`, `device_module_with_hooks(defines, layout="exl3", lanes=8)`. Module names gain `_l{lanes}`; each build passes `-DSGLANG_EXPERT_STREAM_LANES={lanes}` (`extra_cflags` for host, `extra_cuda_cflags` for device).

- [ ] **Step 1: Write the failing test** (append to `test/registered/unit/kernels/test_expert_stream_lease_layout.py`):

```python
from sglang.test.expert_stream_sources import device_sources, host_sources, joined_text, wire_header
import re

FREE_NAME = re.compile(r"(?<![\w:])k(MaxIds|LeaseLanes|LeaseRing|RecordBytes|PageBytes|RecLane\w+|Delta\w+|Split)\b")


def test_no_source_uses_a_free_wire_name():
    """Every wire offset is spelled Wire::k..., so a build's lane count reaches every use."""
    for path in (*host_sources(), *device_sources()):
        text = re.sub(r"//[^\n]*", "", path.read_text())
        assert not FREE_NAME.search(text), f"{path.name} still uses a free wire name"
    assert "inline constexpr auto" not in wire_header().read_text()
```

(Move the two new imports to the top of the file.)

- [ ] **Step 2: Run it to verify it fails.** Commit the test, `SYNC`, `RUN_CPU test/registered/unit/kernels/test_expert_stream_lease_layout.py::test_no_source_uses_a_free_wire_name`. Expected: FAIL naming `copy_engine.h` (or another host file).

- [ ] **Step 3: Rewrite the users.** Run from the repo root, from the scratchpad (not committed):

```python
import pathlib, re

ROOT = pathlib.Path("python/sglang/kernels/jit/csrc/moe")
FILES = [
    "exl3_ram_miss.cuh", "expert_stream/lease_device.cuh", "expert_stream/lease_kernels.cuh",
    "expert_stream/row_copy_kernels.cuh", *[f"expert_stream/host/{n}" for n in (
        "ram_tier.h", "tier_protocol.h", "copy_engine.h", "cpu_experts.h", "ffi_exports.h", "ffi_test_exports.h",
        "split_calibration.h", "reader_base.h", "piece_geometry.h")],
    *[f"expert_stream/bench/src/{n}" for n in (
        "device_sim.h", "device_sim.cpp", "stack.h", "full_stack.cpp", "self_test.cpp")],
]
NAMES = """kDemandHead kDemandRing kDemandRecords kRecordBytes kMaxIds kRecSeq kRecRow kRecCounts kRecFlags
kRecFlagCaptured kRecChain kRecEpoch kRecKinds kRecProtect kRecLaneExpert kRecLaneSlot kRecLaneDst kRecLaneWeight
kRecIdMax kPageBytes kKindHitCopy kKindHitSm kKindHitCpu kKindMissGpu kKindMissCpu kHotHeaderBytes kHotAlignment
kHotRecords kLeaseRing kLeaseLanes kLeaseBlockAlign kLeasePieceMask kLeasePieceMaskLineBytes kLeaseCopyDone
kLeaseCopyDoneBytes kLeaseCopyGate kLeaseGateClosed kLeaseGateOpen kLeaseGateSeqShift kLeaseGateSeqMask kCopyArmed
kSplit kLeaseBlockBytes kDeltaBase kDeltaStride kDeltaTag kDeltaCount kDeltaStaging kDeltaEntries
kDeltaMaxEntries""".split()
RENAME = {"kMaxIds": "kLanes", "kLeaseLanes": "kLanes", "kLeaseRing": "kDemandRecords"}
WIRE = "::sglang::expert_stream::wire::Wire::"
pattern = re.compile(r"(?<![\w:])((?:\w+::)*)(" + "|".join(sorted(NAMES, key=len, reverse=True)) + r")\b")


def sub(m):
    prefix, name = m.group(1), RENAME.get(m.group(2), m.group(2))
    return (WIRE if prefix else "Wire::") + name


for f in FILES:
    p = ROOT / f
    p.write_text(pattern.sub(sub, p.read_text()))
```

Then by hand:
- `bench/src/device_sim.h:23`: replace `constexpr int kLanes = 8;  // wire::kLeaseLanes` with `constexpr int kLanes = ::sglang::expert_stream::wire::Wire::kLanes;` and delete the now-redundant `static_assert` at `device_sim.cpp:13`.
- Any file that uses bare `Wire::` without `using namespace ::sglang::expert_stream::wire;` (the compiler says `'Wire' has not been declared`): add `using ::sglang::expert_stream::wire::Wire;` inside that file's namespace.
- Delete the alias block at the end of `lease_layout.h`.
- Comments that mention `kMaxIds`/`kLeaseLanes` by name: the script already rewrote them to `Wire::kLanes`; read each diff hunk and fix any sentence that no longer reads.

- [ ] **Step 4: Pass the lane count to every build** in `expert_stream_transport.py`:

```python
def _host_module(layout: str = "exl3", variant: Optional[str] = None, lanes: int = 8) -> Module:
    """Return the cached host module for ``layout``, ``variant`` (default build) and ``lanes``."""
    variant = host_variant() if variant is None else variant
    lanes = expert_lease_block.wire_layout(lanes).lanes
    if variant == "instr_tsan" and _ALLOW_TSAN:
        return _host_module_tsan(layout, lanes)
    if variant not in VARIANTS:
        raise ValueError(f"unknown host build variant {variant!r}; expected one of {VARIANTS}")
    if variant not in LAYOUTS[layout].host_sources:
        raise ValueError(f"layout {layout!r} has no {variant!r} host build variant")
    return _host_module_cached(layout, variant, lanes)


@cache_once
def _host_module_cached(layout: str, variant: str, lanes: int) -> Module:
    # Hidden visibility keeps HostExports' registries and members private to each
    # module's .so; only the TVM_FFI_DLL_EXPORT entry points are exported.
    return load_jit(
        f"expert_stream_host_{layout}_{variant}_l{lanes}",
        cpp_files=[LAYOUTS[layout].host_sources[variant]],
        extra_cflags=["-fvisibility=hidden", "-fvisibility-inlines-hidden", f"-DSGLANG_EXPERT_STREAM_LANES={lanes}"],
        extra_ldflags=["-luring", "-lpthread", "-ldl"],
        header_only=False,
    )
```

Apply the same to `_host_module_tsan(layout, lanes)` (name `..._instr_tsan_l{lanes}`, the define appended to its `extra_cflags`), `_device_module(layout, lanes=8)` / `_device_module_cached(layout, lanes)` (name `expert_stream_{layout}_l{lanes}`, `extra_cuda_cflags=[f"-DSGLANG_EXPERT_STREAM_LANES={lanes}"]`), and `device_module_with_hooks(defines, layout="exl3", lanes=8)` (the define appended after the hooks). Every existing positional call keeps working because `lanes` defaults to 8; `cache_once` keys stay positional (`_host_module_cached(layout, variant, lanes)`).

- [ ] **Step 5: Run the tests.** Commit (stage each modified file by name; message `refactor(expert-stream): every wire offset through Wire::, one build per lane count`), `SYNC`, then
  `RUN_CPU test/registered/unit/kernels/test_expert_stream_lease_layout.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`. Expected: PASS.

- [ ] **Step 6: Existing suites unchanged at N = 8.** `RUN_CPU SUITE_CPU` and `RUN_GPU SUITE_GPU`. Expected: the Baseline counts.

- [ ] **Step 7: The bench still builds and self-tests.** On divix01 (the bench is outside the JIT; build only, no timing):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && taskset -c 0-63 cmake -S python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B /data/models/slang/nvfp4-work/nlane-bench -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ -DEXL3_TORCH_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/torch -DEXL3_CXX11_ABI=1 -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src >/dev/null && taskset -c 0-63 cmake --build /data/models/slang/nvfp4-work/nlane-bench -j 16 2>&1 | tail -3; echo EXIT=${PIPESTATUS[0]}'
```

  Expected: `EXIT=0`. Then run the bench's self-test target (`ls /data/models/slang/nvfp4-work/nlane-bench` to find the self-test binary built from `self_test.cpp`; run it under `taskset -c 0-63`). Expected: exit 0.

---

### Task 3: Generic record writer, reader, delta and split loads

**Files:**
- Modify: `expert_stream/lease_device.cuh` (`TypedLanes`, `write_record`, `RowMap`, `MapDelta`, `load_map_delta`, `apply_map_delta`, `LanePolicy`, `load_split`, `type_lanes`)
- Modify: `expert_stream/host/tier_protocol.h` (`read_record`; `kWanted`)
- Modify: `expert_stream/host/ffi_test_exports.h:704-790` (`seqlock_stress` writer, `read_record_fields` output)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`read_record_fields`, `seqlock_stress` take `lanes`; `READ_RECORD_WORDS` becomes a function)
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_read_record.py`
- Modify: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (parametrize over lanes)

**Interfaces:**
- Consumes: `Wire` members (`kLanes`, `kKindWords`, `kPackedCounts`, `kRecProtectCount`, `kRecHeaderBytes`, `kSplitStride`).
- Produces (Python): `read_record_words(lanes: int) -> int` (= `6 + L + 1 + 5 * L` with `L = wire_layout(lanes).lanes`); `read_record_fields(record, expected, *, layout="exl3", variant=None, lanes=8)`; `seqlock_stress(seconds, *, layout="exl3", variant=None, lanes=8)`; `encode_record(lanes, *, seq, row, chain, epoch, flags, protect, lanes_)`, a pure-Python record writer for tests (Step 1).

- [ ] **Step 1: Write the failing tests.** In `test_exl3_ram_miss_read_record.py`, replace the 8-lane encoder and its literals with a lane-generic one and parametrize the round trip:

```python
import struct

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as ram_miss


def encode(lanes, *, seq, row=3, chain=0, epoch=0, flags=0, protect=(), kinds=(), experts=(), slots=(), dsts=(),
           weights=()):
    """A record as the post writes it, for LeaseLayout<lanes>."""
    w = lease.wire_layout(lanes)
    f, n = w.record_fields, w.lanes
    raw = bytearray(w.record_bytes)
    count = len(kinds)
    struct.pack_into("<IH", raw, f["seq"], seq, row)
    if w.packed_counts:
        raw[f["counts"]] = count | len(protect) << 4
    else:
        raw[f["counts"]] = count
        raw[w.protect_count] = len(protect)
    raw[f["flags"]] = flags
    struct.pack_into("<QI", raw, f["chain"], chain, epoch)
    words = [0] * w.kind_words
    for j, kind in enumerate(kinds):
        words[j // 8] |= kind << (4 * (j % 8))
    struct.pack_into(f"<{w.kind_words}I", raw, f["kinds"], *words)
    pad = lambda xs: list(xs) + [-1] * (n - len(xs))
    struct.pack_into(f"<{n}h", raw, f["protect"], *pad(protect))
    struct.pack_into(f"<{n}h", raw, f["lane_expert"], *pad(experts))
    struct.pack_into(f"<{n}h", raw, f["lane_slot"], *pad(slots))
    struct.pack_into(f"<{n}h", raw, f["lane_dst"], *pad(dsts))
    struct.pack_into(f"<{n}f", raw, f["lane_weight"], *(list(weights) + [0.0] * (n - len(weights))))
    return torch.tensor(list(raw), dtype=torch.uint8)


@pytest.mark.parametrize("lanes, count, protect", [(8, 8, 0), (8, 1, 8), (16, 16, 16), (16, 9, 3), (32, 32, 32), (32, 17, 0)])
def test_read_record_round_trips_every_lane(lanes, count, protect):
    kinds = [1 + j % 5 for j in range(count)]
    rec = encode(lanes, seq=7, protect=range(protect), kinds=kinds, experts=range(100, 100 + count),
                 slots=range(count), dsts=range(50, 50 + count), weights=[0.5 + j for j in range(count)])
    out = ram_miss.read_record_fields(rec, 7, variant="instr", lanes=lanes)
    assert out["lanes"] == [
        {"expert": 100 + j, "slot": j, "dst": 50 + j, "weight": 0.5 + j, "kind": kinds[j]} for j in range(count)
    ]
    assert out["protect"] == list(range(protect))


@pytest.mark.parametrize("lanes", [16, 32])
def test_a_count_past_the_lane_width_is_malformed(lanes):
    rec = encode(lanes, seq=7, kinds=[1])
    rec[lease.wire_layout(lanes).record_fields["counts"]] = lanes + 1
    assert ram_miss.read_record_fields(rec, 7, variant="instr", lanes=lanes)["status"] == "malformed"


@pytest.mark.parametrize("lanes", [8, 32])
def test_no_torn_record_is_accepted(lanes):
    accepted, torn = ram_miss.seqlock_stress(2.0, variant="instr", lanes=lanes)
    assert accepted > 0 and torn == 0
```

Keep the file's other tests; rewrite any that build records with the old 8-lane `struct` formats (`"8h"`, `"8f"`, `| len(protect) << 4`, `RECORD_BYTES == 128`) to call `encode(8, ...)`. Read the existing file first and keep each test's assertion; only the record construction changes. If `read_record_fields` currently returns a different key set than `status`/`lanes`/`protect`, use its existing keys in the asserts above.

In `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`, add a lanes parameter to the post round-trip test: build the device and host for `lanes in (8, 16, 32)` (`ExpertStreamDevice(..., lanes=lanes)` arrives in Task 6, so in this task call `_device_module("exl3", lanes)` directly as that file's helpers already do for the default build), post `lanes` planned experts, and read the record back with `read_record_fields(..., lanes=lanes)`. The kinds-word literal at `:193,214,227,236-239` (`HIT_CPU | HIT_CPU << 4`) becomes a helper `kinds_words(kinds, lanes)` that returns the `kind_words` u32 list.

- [ ] **Step 2: Run to verify they fail.** Commit the tests, `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_read_record.py`. Expected: FAIL with `TypeError: read_record_fields() got an unexpected keyword argument 'lanes'`.

- [ ] **Step 3: Generic `read_record`** in `tier_protocol.h` (replaces the parse after the second seq check; the copy-and-recheck part is unchanged except `raw[Wire::kRecordBytes]`):

```cpp
  uint16_t row;
  uint8_t counts, flags;
  uint64_t chain;
  uint32_t epoch, kinds[Wire::kKindWords];
  int16_t protect_ids[Wire::kLanes], expert[Wire::kLanes], slot[Wire::kLanes], dst[Wire::kLanes];
  float weight[Wire::kLanes];
  std::memcpy(&row, raw + Wire::kRecRow, 2);
  std::memcpy(&counts, raw + Wire::kRecCounts, 1);
  std::memcpy(&flags, raw + Wire::kRecFlags, 1);
  std::memcpy(&chain, raw + Wire::kRecChain, 8);
  std::memcpy(&epoch, raw + Wire::kRecEpoch, 4);
  std::memcpy(kinds, raw + Wire::kRecKinds, sizeof(kinds));
  std::memcpy(protect_ids, raw + Wire::kRecProtect, sizeof(protect_ids));
  std::memcpy(expert, raw + Wire::kRecLaneExpert, sizeof(expert));
  std::memcpy(slot, raw + Wire::kRecLaneSlot, sizeof(slot));
  std::memcpy(dst, raw + Wire::kRecLaneDst, sizeof(dst));
  std::memcpy(weight, raw + Wire::kRecLaneWeight, sizeof(weight));
  const int count = Wire::kPackedCounts ? (counts & 0xF) : counts;
  const int protect = Wire::kPackedCounts ? (counts >> 4) : raw[Wire::kRecProtectCount];
  if (count > Wire::kLanes || protect > Wire::kLanes) return RecordRead::kMalformed;
  const auto kind_of = [&](int j) { return (kinds[j / 8] >> (4 * (j % 8))) & 0xFu; };
  bool bad_kind = false;
  for (int j = 0; j < count; ++j)
    bad_kind |= kind_of(j) < Wire::kKindHitCopy || kind_of(j) > Wire::kKindMissCpu;
  if (bad_kind) return RecordRead::kMalformed;
  request->seq = expected;
  request->gen = static_cast<uint64_t>(epoch) << 32 | expected;
  request->row = row;
  request->captured = (flags & Wire::kRecFlagCaptured) != 0;
  request->chain = chain;
  for (int i = 0; i < Wire::kLanes; ++i)
    request->protect[i] = protect_ids[i];
  request->protect.resize(protect);
  for (int j = 0; j < Wire::kLanes; ++j)
    request->lanes[j] = Lane{expert[j], slot[j], dst[j], weight[j], static_cast<uint8_t>(kind_of(j))};
  request->lanes.resize(count);
  return RecordRead::kOk;
```

Delete the `static_assert(Wire::kRecLaneWeight + sizeof(float) * Wire::kLanes == Wire::kRecordBytes, ...)` above it (the record now rounds up) and replace it with `static_assert(Wire::kRecPayloadEnd <= Wire::kRecordBytes, "read_record copies the whole payload");`.

Also in `ram_tier.h:494-498` (`prefetch_request`), replace the two-line prefetch and its `kRecordBytes == 128` assert with:

```cpp
    for (int64_t line = 0; line < Wire::kRecordBytes; line += 64)
      _mm_prefetch(reinterpret_cast<const char*>(record + line), _MM_HINT_T0);
```

and update its comment's "the record's two lines" to "the record's lines".

- [ ] **Step 4: Generic `write_record`** in `lease_device.cuh`:

```cpp
SGL_DEVICE void write_record(uint8_t* record, uint32_t seq, const RecordFields& f) {
  constexpr int L = Wire::kLanes;
  constexpr int kHeaderWords = static_cast<int>((Wire::kRecHeaderBytes - Wire::kRecEpoch) / 4);
  static_assert(kHeaderWords % 4 == 0 && L % 8 == 0, "the header and lane arrays are whole 16-byte stores");
  if (f.count > L || f.protect_count > L) __trap();
  uint32_t protect_w[L / 2], expert_w[L / 2], slot_w[L / 2], dst_w[L / 2], weight_w[L];
  uint32_t header_w[kHeaderWords] = {};
#pragma unroll
  for (int i = 0; i < L; i += 2) {
    const bool a = i < f.count, b = i + 1 < f.count;
    protect_w[i / 2] =
        pack_ids(i < f.protect_count ? f.protect[i] : -1, i + 1 < f.protect_count ? f.protect[i + 1] : -1);
    expert_w[i / 2] = pack_ids(a ? f.planned[i] : -1, b ? f.planned[i + 1] : -1);
    slot_w[i / 2] = pack_ids(a ? f.lanes->slot[i] : -1, b ? f.lanes->slot[i + 1] : -1);
    dst_w[i / 2] = pack_ids(a ? f.dst[i] : -1, b ? f.dst[i + 1] : -1);
  }
  header_w[0] = f.epoch;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    const bool used = i < f.count;
    weight_w[i] = __float_as_uint(used ? f.weight[i] : 0.0f);
    header_w[1 + i / 8] |= (used ? static_cast<uint32_t>(f.lanes->kind[i]) : 0u) << (4 * (i % 8));
  }
  uint32_t counts = static_cast<uint32_t>(f.count);
  if constexpr (Wire::kPackedCounts) {
    counts |= static_cast<uint32_t>(f.protect_count) << 4;
  } else {
    constexpr int b = static_cast<int>(Wire::kRecProtectCount - Wire::kRecEpoch);
    header_w[b / 4] |= static_cast<uint32_t>(f.protect_count) << (8 * (b % 4));
  }
  const uint32_t head = (static_cast<uint32_t>(f.row) & 0xFFFFu) | counts << 16 | (f.flags & 0xFFu) << 24;
  st_relaxed_sys<uint32_t>(record + Wire::kRecSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(record + Wire::kRecRow, head);
  st_relaxed_sys_v2(
      record + Wire::kRecChain, static_cast<uint32_t>(f.chain & 0xFFFFFFFFull), static_cast<uint32_t>(f.chain >> 32));
#pragma unroll
  for (int w = 0; w < kHeaderWords; w += 4)
    st_relaxed_sys_v4(record + Wire::kRecEpoch + 4 * w, header_w[w], header_w[w + 1], header_w[w + 2], header_w[w + 3]);
#pragma unroll
  for (int w = 0; w < L / 2; w += 4) {
    st_relaxed_sys_v4(record + Wire::kRecProtect + 4 * w, protect_w[w], protect_w[w + 1], protect_w[w + 2], protect_w[w + 3]);
    st_relaxed_sys_v4(record + Wire::kRecLaneExpert + 4 * w, expert_w[w], expert_w[w + 1], expert_w[w + 2], expert_w[w + 3]);
    st_relaxed_sys_v4(record + Wire::kRecLaneSlot + 4 * w, slot_w[w], slot_w[w + 1], slot_w[w + 2], slot_w[w + 3]);
    st_relaxed_sys_v4(record + Wire::kRecLaneDst + 4 * w, dst_w[w], dst_w[w + 1], dst_w[w + 2], dst_w[w + 3]);
  }
#pragma unroll
  for (int w = 0; w < L; w += 4)
    st_relaxed_sys_v4(record + Wire::kRecLaneWeight + 4 * w, weight_w[w], weight_w[w + 1], weight_w[w + 2], weight_w[w + 3]);
  st_release_sys(record + Wire::kRecSeq, seq);
}
```

At L = 8 this issues the same stores as v2 in the same order except that the four i16 arrays interleave per 16-byte block instead of array by array; all are relaxed stores before the release, so the reader sees no difference. Update the function's comment: "Traps when the lane or protect count exceeds Wire::kLanes."

- [ ] **Step 5: Generic delta and split loads** in `lease_device.cuh`:

```cpp
SGL_DEVICE MapDelta load_map_delta(const uint8_t* delta) {
  constexpr int kStagingLoads = Wire::kLanes / 8;  // node 0's list; Phase 2 reads every node's
  constexpr int kEntryLoads = static_cast<int>(Wire::kDeltaMaxEntries / 4);
  static_assert(Wire::kDeltaStaging % 16 == 0 && Wire::kDeltaEntries % 16 == 0, "16-byte delta loads");
  MapDelta d;
  d.count = ld_relaxed_sys<uint32_t>(delta + Wire::kDeltaCount);
  uint4 v[kStagingLoads + kEntryLoads];
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i)
    v[i] = ld_relaxed_sys_v4(delta + Wire::kDeltaStaging + 16 * i);
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i)
    v[kStagingLoads + i] = ld_relaxed_sys_v4(delta + Wire::kDeltaEntries + 16 * i);
  const auto lo = [](uint32_t w) { return static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu)); };
  const auto hi = [](uint32_t w) { return static_cast<int32_t>(static_cast<int16_t>(w >> 16)); };
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i) {
    const uint32_t words[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      d.staging[8 * i + 2 * k] = lo(words[k]);
      d.staging[8 * i + 2 * k + 1] = hi(words[k]);
    }
  }
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i) {
    const uint4 e = v[kStagingLoads + i];
    const uint32_t words[4] = {e.x, e.y, e.z, e.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      d.expert[4 * i + w] = lo(words[w]);
      d.slot[4 * i + w] = hi(words[w]);
    }
  }
  return d;
}

SGL_DEVICE void load_split(const uint8_t* split, int32_t (&out)[Wire::kLanes + 1]) {
  constexpr int kLoads = static_cast<int>(Wire::kSplitStride / 16);
  static_assert(Wire::kSplit % 16 == 0 && Wire::kSplit + Wire::kSplitStride <= Wire::kLeaseBlockBytes, "split loads");
  uint4 v[kLoads];
#pragma unroll
  for (int i = 0; i < kLoads; ++i)
    v[i] = ld_relaxed_sys_v4(split + 16 * i);
#pragma unroll
  for (int n = 0; n <= Wire::kLanes; ++n) {
    const uint4 q = v[n / 4];
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    out[n] = static_cast<int32_t>(words[n % 4]);
  }
}
```

In `TypedLanes`, `RowMap` comments, `MapDelta`, `LanePolicy`, `apply_map_delta` and `type_lanes`, the Task 2 rename already made every bound `Wire::kLanes`; delete `load_map_delta`'s old `static_assert(kLeaseLanes == 8 ...)` and `load_split`'s "three loads" assert (both replaced above), and update `load_split`'s comment to "Loads kSplit's table in whole 16-byte loads; words past the table are dropped."

- [ ] **Step 6: Lane-generic test exports and Python wrappers.**
  - `ffi_test_exports.h` seqlock writer (`:708-719`): build the counts the same way as `write_record` (packed nibble when `Wire::kPackedCounts`, else the lane count byte and `record[Wire::kRecProtectCount] = count`).
  - `read_record_fields` export (`:752-790`): the out tensor is `int64 [6 + Wire::kLanes + 1 + 5 * Wire::kLanes]` (already spelled through `Wire::kLanes` after Task 2; verify).
  - Python: replace `_RECORD_LANES` and the constant `READ_RECORD_WORDS` with

    ```python
    def read_record_words(lanes: int = 8) -> int:
        """Words of the C++ read_record result: 6 scalars, the protect ids, the lane count and 5 words per lane."""
        n = expert_lease_block.wire_layout(lanes).lanes
        return 6 + n + 1 + 5 * n
    ```

    and give `read_record_fields` and `seqlock_stress` a `lanes: int = 8` keyword that selects `_host_module(layout, variant, lanes)` and sizes the output with `read_record_words(lanes)`; inside `read_record_fields`, replace `w[6 + _RECORD_LANES]` with `w[6 + n]` and `base = 7 + _RECORD_LANES` with `base = 7 + n`, where `n = expert_lease_block.wire_layout(lanes).lanes`. Update `test_expert_stream_build_variants.py:133-134,169-170` to call `read_record_words()`.

- [ ] **Step 7: Run.** Commit, `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_read_record.py test/registered/unit/kernels/test_expert_stream_build_variants.py`. Expected: PASS, including N=16 and N=32. Then `RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py`. Expected: PASS at 8, 16, 32.

- [ ] **Step 8: Existing suites unchanged.** `RUN_CPU SUITE_CPU` and `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus new tests.

---

### Task 4: CPU lane masks past bit 7 (`ce_mask`, `cpu_lanes`)

**Files:**
- Modify: `expert_stream/row_copy_kernels.cuh:249-262` (params doc and the three shift constants), `:325-360` (CW store, CC), `:495-508` (matchers)
- Modify: `exl3/exl3_route_tables.cuh:72-75`, `expert_residency/direct_gather.cuh:116`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py:2230-2232`, `exl3_route_tables.py:58` (doc), `expert_residency_direct_gather.py:115` (doc)
- Test: `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`

**Interfaces:**
- Produces: `ce_mask` is int32 `[3]` = {copy-engine and CPU lanes, CPU lanes, CPU output parts}; `cpu_lanes` is int32 `[2]` = {CPU lanes, parts (bit 0: part 0, bit 1: part 1)}, or empty when CPU experts are off. Every lane mask is a u32 bit per lane, so `static_assert(Wire::kLanes <= 32)`.

- [ ] **Step 1: Write the failing test** in `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`: add a case at `lanes=16` whose split sends the last 4 of 12 eligible lanes to the CPU (lanes 8-11), runs the chain's CW and CC, and asserts `device_side.cpu_lanes.tolist() == [0xF00, parts]` with `parts` the part bits the existing 8-lane case asserts, and that the route tables rank lanes 8-11 past every column. Base it on the file's existing 8-lane CPU test: copy that test, change the device to `lanes=16`, the plan to 12 experts, and the split to `[0] * 12 + [4] + [0] * 4`.

- [ ] **Step 2: Run to verify it fails.** Commit the test, `SYNC`, `RUN_GPU test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`. Expected: FAIL (`cpu_lanes` has one element, or lanes 8-11 are not marked).

- [ ] **Step 3: Widen the words.** In `row_copy_kernels.cuh`, delete `kCeMaskCpuShift`, `kCeMaskPartShift`, `kCpuLanesPartShift` and their comment, add `static_assert(Wire::kLanes <= 32, "a lane mask is one u32");`, and:

```cpp
  // CW, at the end (replaces the single-word store):
  if (threadIdx.x != 0) return;
  p.ce_mask[0] = p.ce_mask[1] = p.ce_mask[2] = 0;
  if (planned_count == 0) return;
  const uint32_t mask = copying | cpu;
  if (mask == 0) return;
  uint8_t* const gate = lease + Wire::kLeaseCopyGate;
  st_relaxed_sys<uint32_t>(gate, copy_gate_word(seq, Wire::kLeaseGateClosed));
  p.ce_mask[0] = static_cast<int32_t>(mask);
  p.ce_mask[1] = static_cast<int32_t>(cpu);
  p.ce_mask[2] = static_cast<int32_t>(cpu_parts);
```

```cpp
  // CC:
  if (threadIdx.x != 0) return;
  const uint32_t armed = static_cast<uint32_t>(p.ce_mask[0]);
  if (p.cpu_lanes != nullptr) p.cpu_lanes[0] = p.cpu_lanes[1] = 0;
  if (armed == 0) return;
  // ... unchanged CopyDone check ...
  if (p.cpu_lanes != nullptr) {
    p.cpu_lanes[0] = p.ce_mask[1];
    p.cpu_lanes[1] = p.ce_mask[2] & 0x3;
  }
```

Update the `CopyWaitParams::ce_mask` and `CopyCommitParams::cpu_lanes` comments to the new layouts. Matchers: `ce_mask` `TensorMatcher({3})`; `cpu_lanes` stays `TensorMatcher({-1})` with `RuntimeCheck(cpu_lanes.size(0) == 0 || cpu_lanes.size(0) == 2, "cpu_lanes: two words, or empty when CPU experts are off")`.

`exl3_route_tables.cuh:72-75`:

```cpp
  const uint32_t cpu = cpu_lanes != nullptr ? static_cast<uint32_t>(cpu_lanes[0]) : 0u;
  const uint32_t parts = cpu_lanes != nullptr ? static_cast<uint32_t>(cpu_lanes[1]) : 0u;
  const bool part0 = (parts & 1u) != 0;
  const bool part1 = (parts >> 1 & 1u) != 0;
```

`direct_gather.cuh:116`: `const uint32_t cpu = cpu_lanes != nullptr ? static_cast<uint32_t>(cpu_lanes[0]) : 0u;` and its comment becomes "Word 0 of CC's pair: the CPU lanes."

Python `expert_stream_transport.py`: `self.ce_mask = torch.zeros(3, ...)`, `self.cpu_lanes = torch.zeros(2, ...)`. Search for every other allocation or reader: `git grep -n "ce_mask\|cpu_lanes" -- python test` and give each one the two- or three-word shape; update the docstrings in `exl3_route_tables.py:58` and `expert_residency_direct_gather.py:115` ("int32 `[2]`: CPU lanes, then the part bits").

- [ ] **Step 4: Run.** Commit, `SYNC`, `RUN_GPU test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py`. Expected: PASS.

- [ ] **Step 5: Existing suites unchanged.** `RUN_CPU SUITE_CPU` and `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus new tests.

---

### Task 5: Host capacities that scale with N

**Files:**
- Modify: `expert_stream/host/cpu_experts.h:111-114` (`kRing`)
- Modify: `expert_stream/host/piece_geometry.h:137-142` (`kPieceTargets`)
- Modify: `expert_stream/host/reader_base.h:136-146` (`kTraceRows`, `kTraceExtents`)
- Modify: `expert_stream/host/ffi_exports.h:447` (comment), `ffi_test_exports.h:406-408` (message)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py:768-772` (`STAGE_TRACE_ROWS`), `:1702-1707` (calibration shape)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py:271-288` (skip calibration on a small tier)
- Test: `test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`, `test/registered/unit/kernels/test_exl3_ram_miss_split.py:547`, `test/registered/unit/kernels/test_cpu_expert_pool.py`, `test/registered/unit/kernels/test_cpu_expert_keep_warm.py`

**Interfaces:**
- Produces (C++): `CpuExpertEngine::kRing = std::bit_ceil(Wire::kDemandRecords * (Wire::kLanes + 1))` (256 at N=8, as today); `kPieceTargets = Wire::kLanes`; `kTraceRows = 2 * Wire::kLanes`, `kTraceExtents = 2 * kTraceRows`.
- Produces (Python): `stage_trace_rows(lanes: int = 8) -> int` (= `2 * wire_layout(lanes).lanes`); `calibration_shape(lanes: int = 8) -> tuple[int, int]` (= `(L + 2, L + 1)`).

- [ ] **Step 1: Write the failing tests.**
  - `test_exl3_cpu_split_calibration.py`: replace `LANES = 8`, `[0] * 9` and `(LANES + 2, LANES + 1)` with values from `lease.wire_layout(lanes)` and parametrize the shape test over `lanes in (8, 16)`, calling the calibration export through `_host_module("exl3", None, lanes)`; assert `out.shape == ram_miss.calibration_shape(lanes)`.
  - `test_exl3_ram_miss_split.py:547`: replace the literal row count with `ram_miss.stage_trace_rows()`.
  - `test_cpu_expert_pool.py` and `test_cpu_expert_keep_warm.py`: replace each `[0] * 9` with `[0] * (lease.LANES + 1)` and `list(range(9))` with `list(range(lease.LANES + 1))`; leave the report strings `"k=1..8"` as they are (they describe the 8-lane default build).
  - New, in `test_exl3_cpu_split_calibration.py`, Review Focus 1:

    ```python
    def test_a_tier_smaller_than_the_lane_width_skips_calibration(monkeypatch):
        """Calibration needs one RAM slot and one expert of scratch per lane; a smaller tier keeps the configured split."""
        from sglang.srt.layers.moe.cpu_experts import service

        calls = []
        monkeypatch.setattr(service, "_calibrate_native", lambda *a, **k: calls.append(a))
        split = service.calibrated_or_configured(capacity=12, lanes=16, configured=[0] * 17)
        assert split == [0] * 17 and calls == []
    ```

    If `service.py`'s calibration path is not split into a native call and a decision function, Step 3 creates exactly these two names: `_calibrate_native(...)` wraps the existing host call, and `calibrated_or_configured(*, capacity, lanes, configured)` returns `configured` when `capacity < lanes` and otherwise calls `_calibrate_native`.

- [ ] **Step 2: Run to verify they fail.** Commit the tests, `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_cpu_split_calibration.py test/registered/unit/kernels/test_exl3_ram_miss_split.py`. Expected: FAIL (`calibration_shape` / `stage_trace_rows` / `calibrated_or_configured` missing).

- [ ] **Step 3: Implement.**
  - `cpu_experts.h`: `#include <bit>`; `static constexpr size_t kRing = std::bit_ceil(static_cast<size_t>(Wire::kDemandRecords) * (Wire::kLanes + 1));` keep the `static_assert` as is.
  - `piece_geometry.h`: `constexpr int kPieceTargets = ::sglang::expert_stream::wire::Wire::kLanes;  // one readiness word per lane`.
  - `reader_base.h`: `constexpr int kTraceRows = 2 * ::sglang::expert_stream::wire::Wire::kLanes;` and `constexpr int kTraceExtents = 2 * kTraceRows;`; rewrite the comment's "2 * kMaxIds = 16 distinct experts (need and protect ids, 8 each)" to "2 * Wire::kLanes distinct experts (need and protect ids)".
  - `ffi_exports.h:447`: comment `out float64 [kCalibRows, kCalibCols] ms`. `ffi_test_exports.h:406-408`: message `"masks must be [rows, 1.." + std::to_string(kPieceTargets) + "] readiness words"`.
  - Python:

    ```python
    def stage_trace_rows(lanes: int = 8) -> int:
        """kTraceRows: the need and protect ids a request can read, two per lane."""
        return 2 * expert_lease_block.wire_layout(lanes).lanes


    def calibration_shape(lanes: int = 8) -> tuple[int, int]:
        """The calibration grid (kCalibRows, kCalibCols)."""
        n = expert_lease_block.wire_layout(lanes).lanes
        return (n + 2, n + 1)
    ```

    Set `STAGE_TRACE_ROWS = stage_trace_rows()` (kept for the 8-lane callers until Task 6), and in the calibration wrapper replace `torch.zeros((10, 9), ...)` with `torch.zeros(calibration_shape(lanes), ...)`, giving the wrapper a `lanes: int = 8` keyword that selects `_host_module(layout, variant, lanes)`.
  - `service.py`: introduce `_calibrate_native` and `calibrated_or_configured` as described in Step 1, and route the existing `capacity >= LANES` branch through `calibrated_or_configured`.

- [ ] **Step 4: Run.** Commit, `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_cpu_split_calibration.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py`. Expected: PASS.

- [ ] **Step 5: Existing suites unchanged.** `RUN_CPU SUITE_CPU`, `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus new tests.

---

### Task 6: The runtime picks N; Python threads one `WireLayout` everywhere

**Files:**
- Modify: `python/sglang/kernels/ops/moe/expert_lease_block.py` (delete the `_V2` module constants; `lease_block_bytes`, `new_lease_block`, `check_lease_block` take `wire`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (delete `PAGE_BYTES`, `RECORD_BYTES`, `RECORD_FIELDS`, `MAX_IDS`, `DEMAND_*`, `STAGE_TRACE_ROWS`; `new_page(*, pin, wire)`; `ExpertStreamHost(..., lanes)`, `ExpertStreamDevice(..., lanes)` own `self.wire`)
- Modify: `python/sglang/srt/layers/moe/ram_slot_map.py` (`LANES` constant removed; functions take `lanes`)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py` (`LANES` from the host's wire)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`plan_gather_width`, `ensure_started`, `:555`, `:1134`, `:1273`)
- Modify: `python/sglang/test/dsv41_chain_sim.py` (`:21-30, 94-220`)
- Test: `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`, `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`, `test/registered/unit/kernels/test_ram_slot_map.py`, `test/registered/unit/kernels/test_exl3_lease_block.py`, and every test that imports a deleted constant (`git grep -ln "PAGE_BYTES\|RECORD_BYTES\|RECORD_FIELDS\|MAX_IDS\|STAGE_TRACE_ROWS\|lease\.\(LANES\|RING\|SPLIT\|DELTA_\|COPY_\|BLOCK_BYTES\)" -- test python/sglang/test`)

**Interfaces:**
- Consumes: `wire_layout`, `_host_module(..., lanes)`, `_device_module(..., lanes)`, `read_record_words`, `stage_trace_rows`, `calibration_shape`.
- Produces: `Exl3RamMissService.lanes: int` (set at start, `wire_layout(max(planned widths)).lanes`, 8 when nothing was planned); `ExpertStreamHost.wire`, `ExpertStreamDevice.wire` (`WireLayout`); `ram_slot_map.type_lanes(..., lanes)`; `expert_lease_block.lease_block_bytes(rows, wire)`, `new_lease_block(rows, *, pin, wire)`, `check_lease_block(block, rows, *, need_pinned, wire)`; `expert_stream_transport.new_page(*, pin, wire)`.

- [ ] **Step 1: Write the failing tests** in `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py` (use that file's existing service fixture; the names below are the ones it uses at `:198-206` and `:1014-1023`, adapt to the fixture if it differs):

```python
@pytest.mark.parametrize("width, lanes", [(1, 8), (6, 8), (8, 8), (9, 16), (12, 16), (32, 32)])
def test_the_service_builds_for_the_widest_planned_gather(service, width, lanes):
    service.plan_gather_width(width)
    assert service.resolved_lanes() == lanes


def test_a_gather_wider_than_32_is_refused(service):
    with pytest.raises(ValueError, match="1..32"):
        service.plan_gather_width(33)


def test_the_eager_plan_pads_to_the_lane_width(service):
    """Review Focus 4: an eager post with up to N lanes fits the planned tensor."""
    service.plan_gather_width(12)
    assert service.planned_padding(capacity=5) == 16


def test_staging_keeps_a_fill_slot_on_a_small_tier(service):
    """Review Focus 5: N staging slots never take a row's last slot."""
    service.plan_gather_width(16)
    assert service.staging_for(capacity=10) == 9
```

`resolved_lanes()`, `planned_padding(capacity)` and `staging_for(capacity)` are small pure methods Step 3 adds so the decisions are testable without a GPU; `ensure_started` calls them.

`test_exl3_ram_miss_attach_lanes.py:72,80`: the refusal case becomes `[33]`, and `[1, 6, 8, 12]` are accepted. `test_ram_slot_map.py:59-61`: `match="at most 8"` becomes a call with `lanes=8` (unchanged message) plus a new case `lanes=16` accepting 16 experts and refusing 17. `test_exl3_lease_block.py`: its v2 literals stay, computed from `lease.wire_layout(8)`; add `lease_block_bytes(1, lease.wire_layout(32)) == lease.wire_layout(32).lease_block_bytes + 4096`.

- [ ] **Step 2: Run to verify they fail.** Commit the tests, `SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_ram_slot_map.py`. Expected: FAIL (`resolved_lanes` missing; 12-wide plan refused).

- [ ] **Step 3: Implement.**
  - `exl3_ram_miss.py`:

    ```python
    def plan_gather_width(self, rows: int) -> None:
        """Plan a layer whose graph gather misses ``rows`` ids; the widest layer sets the build's lane count.

        Only valid before the service starts.
        """
        if self.host is not None:
            raise RuntimeError(
                "exl3 RAM miss: the graph gather width was planned after the service started"
            )
        planned = max(1, int(rows))
        wire_layout(planned)  # refuses a width outside 1..32
        self._gather_planned = planned if self._gather_planned is None else max(self._gather_planned, planned)

    def resolved_lanes(self) -> int:
        """The lane count the service builds for: the widest planned gather, rounded up to 8."""
        return wire_layout(self._gather_planned or 1).lanes

    def planned_padding(self, capacity: int) -> int:
        return max(capacity, self.resolved_lanes())

    def staging_for(self, capacity: int) -> int:
        return min(self._gather_planned or self.resolved_lanes(), capacity - 1)
    ```

    In `ensure_started`, set `self.lanes = self.resolved_lanes()` and `self.wire = wire_layout(self.lanes)` first; pass `lanes=self.lanes` to `ExpertStreamHost(...)` and later to `ExpertStreamDevice(...)`; build the page with `new_page(pin=pin, wire=self.wire)`; call `host.reserve_staging(...)` with the same value the code computes today (the planned width, capped by the host's existing `min(k, capacity - 1)` rule). Delete `self.staging_slots` (its role is `_gather_planned`) after checking its other readers with `git grep -n staging_slots`. At `:555` pad to `max(capacity, self.lanes)`; at `:1134` compare against `self.lanes`; at `:1273` pass `self.lanes` to `GraphRouteLog`.
  - `expert_stream_transport.py`: `ExpertStreamHost.__init__` and `ExpertStreamDevice.__init__` take `lanes: int = 8`, store `self.wire = expert_lease_block.wire_layout(lanes)`, load `_host_module(layout, variant, self.wire.lanes)` / `_device_module(layout, self.wire.lanes)`, and size every lane tensor (`staging (layers, lanes)`, `lane_kind`, `lane_slot`, `host_rows_1`, `dst_slots_1`, `split`) from `self.wire.lanes`, the page check from `self.wire.page_bytes`, the lease block from `new_lease_block(rows, pin=pin, wire=self.wire)`. `reserve_staging(self, k: int | None = None)` defaults to `self.wire.lanes`. `set_cpu_split` checks `len(split) == self.wire.lanes + 1`. The calibration wrapper passes `self.wire.lanes`. Delete the module constants listed under Files and fix each import the deletion breaks (the grep in Files lists them).
  - `expert_lease_block.py`: delete the `_V2` block; `lease_block_bytes(rows, wire)` returns `wire.lease_block_bytes + _round_up(rows * wire.delta_stride, wire.block_align)`; `new_lease_block` / `check_lease_block` take `wire` and use `wire.block_align`.
  - `ram_slot_map.py`: delete `LANES = 8`; each function that bounded by `LANES` takes `lanes: int` and refuses `len(experts) > lanes` with the existing message, `at most {lanes}`.
  - `cpu_experts/service.py`: delete `LANES = 8`; read `host.wire.lanes` where it used `LANES`.
  - `dsv41_chain_sim.py`: take a `wire` (default `wire_layout(8)`) and write records with the same counts/kinds encoding as `encode` in Task 3's test (packed nibble at 8, separate protect byte otherwise; `kind_words` u32s).

- [ ] **Step 4: Run.** Commit, `SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_ram_slot_map.py test/registered/unit/kernels/test_exl3_lease_block.py`. Expected: PASS.

- [ ] **Step 5: Existing suites unchanged.** `RUN_CPU SUITE_CPU` and `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus new tests. Also `RUN_GPU test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py` with its `range(8)` / `e % 8` literals rewritten to `wire.lanes`.

---

### Task 7: Bench and device simulator at any N

**Files:**
- Modify: `expert_stream/bench/src/device_sim.cpp:174-189` (record encoding), `bench/src/self_test.cpp:150-224`
- Modify: `expert_stream/bench/CMakeLists.txt` (an `EXPERT_STREAM_LANES` cache variable passed as `-DSGLANG_EXPERT_STREAM_LANES`)

**Interfaces:**
- Consumes: `Wire` members.
- Produces: `cmake -DEXPERT_STREAM_LANES=16` builds the bench and self-test for 16 lanes (default 8).

- [ ] **Step 1: Write the failing test.** In `self_test.cpp`, replace every literal tied to 8 lanes (`std::array<int16_t, 8>`, `std::array<int32_t, 9>`, `memcpy(..., 16)`, `memcmp(..., 16|32)`, `kRecCounts == (2 | 2 << 4)`, `kRecKinds == (3u | 4u << 4)`) with `Wire::kLanes`-based sizes and add a check that decodes the simulator's record through `read_record` for a full `Wire::kLanes`-lane request (every lane a hit) and compares every lane. Configure a second build at 16 lanes:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && taskset -c 0-63 cmake -S python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B /data/models/slang/nvfp4-work/nlane-bench16 -DEXPERT_STREAM_LANES=16 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ -DEXL3_TORCH_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/torch -DEXL3_CXX11_ABI=1 -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src >/dev/null && taskset -c 0-63 cmake --build /data/models/slang/nvfp4-work/nlane-bench16 -j 16 2>&1 | tail -3; echo EXIT=${PIPESTATUS[0]}'
```

- [ ] **Step 2: Run to verify it fails.** Commit the self-test change, `SYNC`, run the command above, then the 16-lane self-test binary. Expected: the build ignores `EXPERT_STREAM_LANES` (still 8 lanes), so the full-width check decodes only 8 lanes, or the self-test fails on the nibble counts.

- [ ] **Step 3: Implement.**
  - `CMakeLists.txt`: `set(EXPERT_STREAM_LANES 8 CACHE STRING "lanes per demand record (1..32)")` and `add_compile_definitions(SGLANG_EXPERT_STREAM_LANES=${EXPERT_STREAM_LANES})`.
  - `device_sim.cpp:174-189`: encode counts and kinds as `write_record` does (packed nibble when `Wire::kPackedCounts`, else the lane-count byte and `rec[Wire::kRecProtectCount]`; kinds into `Wire::kKindWords` u32 words at `Wire::kRecKinds`).

- [ ] **Step 4: Run.** Commit, `SYNC`, build both (`nlane-bench` at 8 from Task 2 Step 7, `nlane-bench16` above) and run both self-tests. Expected: exit 0 for both.

---

### Task 8: End-to-end on divix01 at N = 8

**Files:** none changed; results recorded in the ledger.

- [ ] **Step 1: Full suites against the baseline.** `RUN_CPU SUITE_CPU`, `RUN_GPU SUITE_GPU`, and `RUN_CPU test/registered/unit/kernels` (the registered kernel suite, per the run protocol). Compare counts with the Baseline and with the same target at the merge-base (`git merge-base master numa-node-distributor`, in a second private worktree). Expected: only the tests this plan added differ.

- [ ] **Step 2: The wire is byte-identical at N=8.** Check that the production device and host modules still build as `expert_stream_exl3_l8` / `expert_stream_host_exl3_prod_l8`, and that `test_eight_lanes_on_one_node_is_wire_v2` passed in Step 1.

- [ ] **Step 3: No ms/token regression.** Run the bench-service full-stack job (command file per `exl3bench.service`; restore `service-command.txt` afterwards and verify with `cmp`) with the worktree's `run_full_stack.sh`, 1, 3 and 5 experts, against the same job at the merge-base. Expected: p50 within the run-to-run noise of the merge-base (record both medians and the spread).

- [ ] **Step 4: Clean up.** `ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-nlane'` and remove `nlane-bench`, `nlane-bench16` (ask the user before deleting the build directories).
