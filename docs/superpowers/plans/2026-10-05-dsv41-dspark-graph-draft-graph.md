# DSpark draft in the decode graph: one lease channel protocol, two clients

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The DSpark draft runs in the breakable decode graph on DSV4.1 EXL3, CPU experts included, with no host
sync and no Python on its MoE path. The device↔host handoff it uses is the target's lease protocol, extracted into
one templated channel that both clients instantiate.

**Architecture:**
- **One protocol, two instantiations.** The target's request page, seqlocked record ring, generation `G`, CopyDone,
  gate encoding, Dekker close/open, `cuStreamWaitValue32_v2` wait node and the commit's trap move into a template,
  `LeaseChannel<Spec>`:
  - `lease_channel_layout.h`: the spec and the gate encoding;
  - `lease_channel.cuh`: the device half;
  - `host/lease_channel.h`: the host half.

  The target is ported onto it first, byte-identical (Task 1). The draft is the second instantiation, `DraftSpec`, with
  its own page, record type and thread (Tasks 3-4). Payload areas (the x rows, the CPU output) and threads stay
  separate per client. Only the protocol is shared.
- **The draft's GPU share** runs the existing graph-safe `Exl3FusedMoE` (D2-1) over a slot slab of the resident draft
  experts. A device map sends every other route to a sink column. This replaces `exl3_moe_loop` and
  `exl3_moe_accumulate`, which read the host (Task 2).
- **The draft's CPU share** posts through the draft channel:
  - The post kernel stages `x [M, H]` (fp16) and the per-token routes `[M, k]` into pinned areas, writes a record, and
    publishes the head.
  - The draft CPU thread, the OpenMP master on the draft cores, holds its team in keep-warm watching the head word,
    which the GPU's store moves, so no syscall is needed. It runs the M-row forward, then completes through the
    channel.
  - The finish kernel closes the gate, the stream waits on it, and the commit kernel checks `done == G` and adds the
    CPU output into the fused MoE's output.
  - The GPU share runs between post and finish, so the two overlap.
- **The worker** captures the EXL3 draft's decode graphs (lifting D2-3's `draft_runs_exl3` gate). It builds the draft's
  static buffers and starts its CPU thread before capture.

**Tech Stack:** C++20 (host module, JIT), CUDA (device module, JIT), Python/torch, pytest, CUDA graphs (breakable
backend). divix01 for every run.

**Spec:**
- `DSV41_REFERENCE.md` §33.3 item 7: "Graphing the draft needs a capturable multi-token EXL3 MoE for its 128 resident
  experts. None exists".
- §33.8: "What D2-3 does not do: decode graphs for an EXL3 draft".
- `docs/superpowers/plans/2026-09-19-dsv41-dspark.md`, Phase D2 item 2: "Draft under a graph".
- `analysis/dsv41-drive/LEASE_PROTOCOL.md`: the protocol Task 1 extracts.
- The owner's decisions in the 2026-10-05 session:
  - full draft support in graphs, CPU experts included;
  - **one protocol**: a single templated protocol with two versions, not a second wire. Separate read areas and
    threads are fine;
  - the target is ported onto the template first.

There is no separate design doc. The design and its findings are recorded here.

## Findings from the code read for this plan

- **The draft already sits on the breakable machinery.**
  - The draft worker's `ModelRunner` captures through `DecodeCudaGraphRunner` (`decode_cuda_graph_runner.py:216`).
  - The backend comes from the shared `cfg.decode.backend` (`runner_backend/utils.py:54`), so an EXL3 launch gives it
    `BreakableCudaGraphBackend`.
  - It captures `ForwardMode.TARGET_VERIFY` at width `decode_num_tokens_per_req` (`spec_info.py:299-300`) and bs 1.
  - Only two things keep it eager:
    - `capture_decode_cuda_graph = ... and not draft_runs_exl3(self.draft_model)` (`dspark_worker_v2.py:537`);
    - `Exl3MoEMethod.apply`'s draft branches, which call `assert_not_capturing` (`exl3.py:533`, `:615`).
  - The draft model (`models/deepseek_v4_dspark.py`) has no other host read on its forward path, and no Engram.
- **`Exl3FusedMoE` (`exl3/fused_moe.py:100`) runs any fixed slot set for 1-16 tokens in a graph.**
  - It needs a `{w13,w2}_{trellis,suh,svh}` slab `[slots, part, ...]`, a device `remap` of route → slot, and
    `keep = ones(1)`.
  - It is not tied to a streamer: `test_exl3_fused_moe_multitoken_gpu.py` captures it over a bare 16-slot dict.
  - exllamav3's `exl3_moe` treats `expert_count`'s last column as ignored (`num_experts = expert_count.size(0) - 1`).
    So routes remapped to slot `slots` (the sink) drop out. Task 2 pins this at the pinned exllamav3 commit
    (`ext.py`: `02aef45`), with layer fusion both on and off.
- **The target's CPU lanes cannot carry the draft as they are.** Each of these is reason enough:
  - Records are one token: `lease_kernels.cuh:428`, `CpuJob` `rows = 1`, and the summed per-lane weight.
  - A `HIT_CPU` lane must name a tier slot (`ram_tier.h:1680`).
  - Completion runs through the copy thread.
  - The protocol assumes one poster per page.

  The protocol underneath is generic. What is specific is the record and who completes it. That is the line the
  template draws.
- **`CpuExpertKernel::forward` already takes M rows** (`ForwardCall.rows`, `kernel.hpp:58-68`; EXL3 `kMaxRows`
  65536). `kernel_forward` (`ffi_test_exports.h:859`) is the eager draft's call today. The draft thread calls
  `forward` directly. `CpuExpertEngine` (one-token `CpuJob`) is not changed.

## Global Constraints

- **BS1 and eager are unchanged.**
  - Without `--speculative-algorithm DSPARK` no new code runs.
  - The target's page, completion block and delta block are **byte-identical** after Task 1:
    - `LeaseLayout`'s `static_assert`s (`lease_layout.h:114-120`) stay as they are;
    - `test_expert_stream_lease_layout` stays green.
- **CPU experts stay off for a target verify.** `SGLANG_DSV41_CPU_EXPERTS=1` with speculation stays refused
  (§33.3 item 5). The draft's CPU experts (`SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`) are what this plan graphs.
- **No host sync on the draft MoE path.** Every per-call quantity is computed on the device, which means:
  - no `.cpu()`, `.item()`, `.tolist()` or `bool(tensor)`;
  - no Python executor;
  - no `torch.where` with a data-dependent shape.

  Host reads at build time (`prepare`) are fine.
- **Draft M is at most 16 per fused call** (`ROW_TILE`, `fused_moe.py`). A larger M (the draft's prompt extend)
  runs in chunks of 16 through the same path, eagerly.
- **Do not edit `model_runner.py`** (frozen). The worker hook lives in `dspark_worker_v2.py`.
- **Speculative identifiers follow `.claude/skills/speculative-naming/SKILL.md`.** Counters end in `_ct`.
- **Env vars follow `.claude/skills/env-var-conventions/SKILL.md`.** There are two new ones, both in `environ.py`
  beside the other `SGLANG_DSV41_DSPARK_*` entries and read through `envs.X.get()`:
  - `SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US = EnvInt(100_000)`: the draft CPU thread's hold after the warm
    window. -1 never releases.
  - `SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH = EnvBool(False)`: keeps an EXL3 draft eager. This is the A/B switch.
- **No `fable` model** for any subagent or reviewer (owner's standing rule). Use `opus` for the most capable tier.
- **divix01 protocol** (`.claude/rules/divix01-run-protocol.md`):
  - Push the branch, then `git fetch` and `git checkout --detach origin/dsv41-dspark-graph` in
    `/data/models/slang/nvfp4-work/wt-dsv41-dspark-graph`.
  - Run with `PYTHONPATH=$PWD/python` and print `sglang.__file__` first.
  - Read `${PIPESTATUS[0]}`.
  - CPU jobs run under `taskset -c 0-63`; GPU work under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c
    32-63`.
  - A job that also reads checkpoint rows takes `rowimg-disk.lock` first.
  - EXL3 GPU tests need `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`.
  - A C++ change cold-rebuilds every JIT module (50-100 s each). Warm them in the parent (`spawn_child`) before any
    child test.
- **Commits:** no amend, rebase, stash or force-push. Stage files by name. Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq
  ```

The laptop has no torch or CUDA, so every test runs on divix01. `<FILES>` and `<K>` vary:

```bash
# CPU
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/dg.log 2>&1; echo EXIT=${PIPESTATUS[0]}; grep -E "^FAILED|^ERROR" /tmp/dg.log | head; tail -3 /tmp/dg.log'

# GPU
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/dg.log 2>&1; echo EXIT=${PIPESTATUS[0]}; grep -E "^FAILED|^ERROR" /tmp/dg.log | head; tail -3 /tmp/dg.log'
```

`<K>` = `not nothing_matches_this` selects every test in `<FILES>`.

**The target protocol suite**, run after every task that touches the channel (Tasks 1, 3, 4):
- CPU: `test/registered/unit/kernels/test_*exl3*.py test/registered/unit/kernels/test_*expert*.py` with `-n 8`.
- GPU: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_lease_ordering_cuda.py
  test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py
  test/manual/dsv41/test_exl3_moe_split_parity_cuda.py test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py`

## Review Focus

1. **The target's protocol after the port.** Expected: no observable change, meaning:
   - the same bytes on the page and in the completion block;
   - the same fences, in the same order, around the same stores;
   - the same traps.

   Task 1 pins the layout with `static_assert`s and a probe test. It pins behavior with the target protocol suite,
   including the byte-exact captured replay and the armed/unarmed CW tests. It runs two mutants against the template
   (drop the Dekker fence; drop the commit's trap), and each must turn a target test red.
2. **A torn or lapped draft record.**
   - The draft posts at most one record before its finish waits, so a lap is impossible. The draft thread treats a torn
     read or a lap as a protocol failure and fail-stops with the seq.
   - A silent skip would hang the stream on its gate.
   - Task 4 pins the fail-stop with a test export that writes a torn record.
3. **The draft CPU thread dies or hangs mid-forward.**
   - Expected: the process aborts within the watchdog bound (`watchdog_wait_s(SGLANG_DSV41_RAM_MISS_TIMEOUT_MS)`),
     with a line naming the draft and the seq. It must not wait forever on the gate.
   - A refused forward fail-stops at once.
   - Task 4 pins both (the fake kernel's `hold` and `fail`).
4. **Routes that are not on the CPU and not resident** (an out-of-range id, `-1`, a fused shared expert id ≥
   `n_routed`). Expected:
   - `-1` and out-of-range ids reach neither share;
   - a fused shared id is resident by construction (`_attach_cpu_draft`);
   - a route that is neither resident nor on the CPU is impossible by construction, and the build refuses one.

   Task 2 pins the sink for `-1` and out of range. Task 5 pins the build's refusal.
5. **Teardown with a closed draft gate** (stop while a forward is pending, e.g. at server shutdown mid-step).
   Expected: `stop()` opens a closed gate through the channel's `open_closed_gate`, as the target does. The device
   commit then traps rather than reading partial sums, and the process is ending anyway. Task 4 pins that `stop()`
   returns and the gate reads open.

---

### Task 1: Extract the lease channel template; the target instantiates it, byte-identical

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_channel_layout.h`
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_channel.cuh`
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/lease_channel.h`
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/stream_wait.h` (moved from `row_copy_kernels.cuh:368-380`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` (`TargetChannel` alias and asserts)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (`copy_gate_word`,
  `pending_generation`, `ring_index`, `write_record`'s seqlock begin/end, through the template)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (the post's seq/epoch advance and head
  publish, `:170-176`, `:240`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh` (CW's close/Dekker `:316-332`, CC's
  check `:346-355`, the wait node `:543-552`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h` (`read_record`'s seqlock copy,
  `record_offset`, `reached`, `skip_zero`, `load_acquire`/`store_release`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`copy_completed` `:1167-1176`,
  `gate_word` `:612`, `cas_gate` `:1188`, `open_closed_gate` `:627`)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (a new opening section, "The lease channel")
- Test: `test/registered/unit/kernels/test_lease_channel_layout.py` (new)

**Interfaces:**
- Produces (C++, namespace `sglang::expert_stream::channel`, used by Tasks 3-4):
  - `template <int64_t Head, int64_t Ring, uint32_t Records, int64_t RecordBytes, int64_t Done, int64_t Gate> struct
    ChannelSpec`, with `kHead, kRing, kRecords, kRecordBytes, kDone, kDoneBytes = 8, kGate`.
  - `SGL_HD constexpr uint32_t gate_word(uint32_t seq, uint32_t low)`, and the constants `kGateClosed = 0x80000001u`,
    `kGateOpen = 1`, `kGateSeqShift = 2`, `kGateSeqMask = 0x1FFFFFFF`.
  - Device (`lease_channel.cuh`, namespace `device::expert_stream::channel`):
    - `uint32_t advance(int32_t* state)`;
    - `uint64_t generation(uint32_t seq, uint32_t epoch)`;
    - `template <class S> uint8_t* record_at(uint8_t* page, uint32_t seq)`;
    - `template <class S> void begin_record(uint8_t* record)`;
    - `template <class S> void end_record(uint8_t* record, uint32_t seq)`;
    - `template <class S> void publish_head(uint8_t* page, uint32_t seq)`;
    - `template <class S> void close_gate(uint8_t* lease, uint32_t seq, uint64_t gen)`: the close plus the Dekker check;
    - `template <class S> void commit_or_trap(const uint8_t* lease, uint32_t seq, uint64_t gen)`.
  - Host (`host/lease_channel.h`):
    - `template <class S> const uint8_t* record_at(const uint8_t* page, uint32_t seq)`;
    - `template <class S> bool read_seqlocked(const uint8_t* record, uint32_t expected, uint8_t (&raw)[S::kRecordBytes])`;
    - `template <class S> void complete(uint8_t* lease, uint32_t seq, uint64_t gen)`;
    - `template <class S> uint32_t gate(const uint8_t* lease)`;
    - `template <class S> void open_closed_gate(uint8_t* lease)`.
  - `stream_wait.h`: `expert_stream::stream_wait_value32()` (moved unchanged), plus a new
    `void enqueue_gate_wait(cudaStream_t stream, uint64_t gate_address)`. It throws on a failed `CUresult`.
  - `using TargetChannel = ChannelSpec<Wire::kDemandHead, Wire::kDemandRing, Wire::kDemandRecords,
    Wire::kRecordBytes, Wire::kLeaseCopyDone, Wire::kLeaseCopyGate>;` in `lease_layout.h`.

- [ ] **Step 1: Write the failing layout test.** It reads the channel's offsets through the existing layout probe
  (`expert_lease_block.wire_probe`, built from `lease_layout_probe.cpp`). It asserts the target channel's six numbers
  equal the wire's own:

```python
"""The target's lease channel (lease_channel_layout.h) names exactly the wire's own offsets: Task 1's port is
byte-identical."""

import pytest

from sglang.kernels.ops.moe import expert_lease_block
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.mark.parametrize("lanes,nodes", [(8, 1), (8, 2), (16, 1), (32, 2)])
def test_the_target_channel_is_the_wire(lanes, nodes):
    probe = expert_lease_block.wire_probe(lanes, nodes)
    wire = expert_lease_block.wire_layout(lanes, nodes)
    assert probe["chan_head"] == wire.demand_head == 0
    assert probe["chan_ring"] == wire.demand_ring
    assert probe["chan_records"] == wire.demand_records
    assert probe["chan_record_bytes"] == wire.record_bytes
    assert probe["chan_done"] == wire.lease_copy_done
    assert probe["chan_gate"] == wire.lease_copy_gate


def test_the_gate_encoding_is_shared():
    probe = expert_lease_block.wire_probe(8, 1)
    for seq in (1, 2, 0x1FFFFFFF, 0x20000001):
        assert probe[f"gate_open_{seq}"] == expert_lease_block.gate_word(seq, "open")
        assert probe[f"gate_closed_{seq}"] == expert_lease_block.gate_word(seq, "closed")
```

  Before writing it, read `lease_layout_probe.cpp` and `expert_lease_block.wire_probe` (`expert_lease_block.py:168`)
  for the probe's key/value output format. Name the new keys in that format. If `WireLayout` names the fields
  differently (`demand_ring` etc.), use its names; the assertions stay as written.

- [ ] **Step 2: Run it. Expected: FAIL** with `KeyError: 'chan_head'` (the probe has no channel keys).
  Run CPU: `<FILES>` = `test/registered/unit/kernels/test_lease_channel_layout.py`.

- [ ] **Step 3: Write `lease_channel_layout.h`.**

```cpp
// The lease channel: the device-to-host request handoff every expert-stream client shares (LEASE_PROTOCOL.md, "The
// lease channel"). A client is a ChannelSpec: where its head word, record ring, done words and gate sit. The record
// payload, what the host does with it, and the data areas are the client's own.
//
//   page   device-written, host-read: head (u32, the last posted seq, release-stored), then a ring of `Records`
//          records, each a seqlock (seq word first: 0 while rewritten, the seq stored last with a release)
//   lease  host-written, device-read: done[Records] (u64 G = epoch << 32 | seq, release-stored), and the gate (u32)
//
// The gate is the stream's wake-up: closed(G) while a wait holds the stream, open(G) once done[G] holds. The device
// closes it and re-checks done after a system fence; the host stores done, fences, and opens it if it reads closed(G).
// One side always sees the other's store (a Dekker pair), and both open with the same word.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define SGL_HD __host__ __device__
#else
#define SGL_HD
#endif

namespace sglang::expert_stream::channel {

constexpr uint32_t kGateClosed = 0x80000001u;  // bit 31 set: a cyclic GEQ wait against kGateOpen blocks
constexpr uint32_t kGateOpen = 1;
constexpr uint32_t kGateSeqShift = 2;
constexpr uint32_t kGateSeqMask = 0x1FFFFFFF;

// The gate word for record `seq`: its sequence number in the high bits, the closed/open state in the low bits. An
// open word is in [1, 2^31), so cuStreamWaitValue32's cyclic GEQ against kGateOpen passes it and blocks a closed one.
SGL_HD constexpr uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & kGateSeqMask) << kGateSeqShift) | low;
}

template <int64_t Head, int64_t Ring, uint32_t Records, int64_t RecordBytes, int64_t Done, int64_t Gate>
struct ChannelSpec {
  static constexpr int64_t kHead = Head;
  static constexpr int64_t kRing = Ring;
  static constexpr uint32_t kRecords = Records;
  static constexpr int64_t kRecordBytes = RecordBytes;
  static constexpr int64_t kDone = Done;
  static constexpr int64_t kDoneBytes = 8;
  static constexpr int64_t kGate = Gate;
  static_assert(Head % 4 == 0 && Ring % 128 == 0 && RecordBytes % 128 == 0, "records are whole 128-byte line pairs");
  static_assert(Records >= 2, "a ring of at least two records");
  static_assert(Done % 8 == 0 && Gate % 128 == 0, "done words are u64; the gate has a line of its own");
  static_assert(Gate >= Done + Records * kDoneBytes || Gate + 4 <= Done, "the gate is not a done word");
};

}  // namespace sglang::expert_stream::channel
```

- [ ] **Step 4: Write `lease_channel.cuh`.** Each function body is the target's current code, moved: the source line
  is named in each comment. `st_release_sys`, `ld_acquire_sys64`, `st_relaxed_sys`, `st_release_sys64` stay in
  `lease_device.cuh` (`:49-63`). This header includes `lease_device.cuh`'s primitive block, so move those four
  primitives into it first and have `lease_device.cuh` include it.

```cpp
// The device half of the lease channel (lease_channel_layout.h). Bodies moved from lease_kernels.cuh (advance,
// publish_head), lease_device.cuh (the record seqlock) and row_copy_kernels.cuh (close_gate, commit_or_trap).
#pragma once

#include <cuda/atomic>
#include "lease_channel_layout.h"
#include "lease_primitives.cuh"  // st_relaxed_sys, st_release_sys, ld_acquire_sys64, st_release_sys64 (moved)

namespace device::expert_stream::channel {

using namespace ::sglang::expert_stream::channel;

// The device's own state words (never on the wire): the target's kPosted/kPending/kEpoch/kPendingEpoch indices.
constexpr int kPosted = 0, kPending = 1, kEpoch = 2, kPendingEpoch = 3;

// The next seq, never 0; a wrap bumps the epoch, so G = epoch << 32 | seq never repeats. (lease_kernels.cuh:170-176)
SGL_DEVICE uint32_t advance(int32_t* state) {
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) {
    seq = 1;
    state[kEpoch] += 1;
  }
  state[kPosted] = static_cast<int32_t>(seq);
  return seq;
}

SGL_DEVICE uint64_t generation(uint32_t seq, uint32_t epoch) {
  return (static_cast<uint64_t>(epoch) << 32) | seq;
}

template <class S>
SGL_DEVICE int64_t ring_index(uint32_t seq) {
  return static_cast<int64_t>((seq - 1u) % S::kRecords);
}

template <class S>
SGL_DEVICE uint8_t* record_at(uint8_t* page, uint32_t seq) {
  return page + S::kRing + ring_index<S>(seq) * S::kRecordBytes;
}

// The seqlock's open: seq = 0, then a release fence, so no payload store passes it. (lease_device.cuh write_record)
template <class S>
SGL_DEVICE void begin_record(uint8_t* record) {
  st_relaxed_sys<uint32_t>(record, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
}

// The seqlock's close: the seq, stored last with a release. (lease_device.cuh:256)
template <class S>
SGL_DEVICE void end_record(uint8_t* record, uint32_t seq) {
  st_release_sys(record, seq);
}

// A release orders every earlier store of this thread: the record (and any client area) comes first.
// (lease_kernels.cuh:240)
template <class S>
SGL_DEVICE void publish_head(uint8_t* page, uint32_t seq) {
  st_release_sys(page + S::kHead, seq);
}

// Closes the gate for G, then re-checks done[G] after a system fence and opens the gate itself if the host already
// completed: the device half of the Dekker pair with host complete(). (row_copy_kernels.cuh:316-332)
template <class S>
SGL_DEVICE void close_gate(uint8_t* lease, uint32_t seq, uint64_t gen) {
  uint8_t* const gate = lease + S::kGate;
  st_relaxed_sys<uint32_t>(gate, gate_word(seq, kGateClosed));
  __threadfence_system();
  if (ld_acquire_sys64(lease + S::kDone + ring_index<S>(seq) * S::kDoneBytes) == gen)
    st_release_sys(gate, gate_word(seq, kGateOpen));
}

// After the stream's wait: the gate is only the wake-up, done[G] is what commits. Its acquire orders the host's
// results before every later read of them. Only a teardown opens a gate without done[G]. (row_copy_kernels.cuh:353)
template <class S>
SGL_DEVICE void commit_or_trap(const uint8_t* lease, uint32_t seq, uint64_t gen) {
  if (ld_acquire_sys64(lease + S::kDone + ring_index<S>(seq) * S::kDoneBytes) != gen) __trap();
}

}  // namespace device::expert_stream::channel
```

  `SGL_DEVICE` is the macro `lease_device.cuh` already uses. Include the header that defines it. `lease_primitives.cuh`
  holds exactly the four functions at `lease_device.cuh:49-66` plus `st_relaxed_sys`/`st_relaxed_sys_v2`, moved
  verbatim; `lease_device.cuh` then includes it.

- [ ] **Step 5: Write `host/lease_channel.h`.**

```cpp
// The host half of the lease channel (lease_channel_layout.h). Bodies moved from tier_protocol.h (read_record's
// seqlock copy) and ram_tier.h (copy_completed, cas_gate, open_closed_gate).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include "../lease_channel_layout.h"

namespace sglang::expert_stream::channel {

inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}
inline void store_release64(uint8_t* address, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(address), value, __ATOMIC_RELEASE);
}

template <class S>
const uint8_t* record_at(const uint8_t* page, uint32_t seq) {
  return page + S::kRing + static_cast<int64_t>((seq - 1u) % S::kRecords) * S::kRecordBytes;
}

// Copies the record into `raw` if its seq word reads `expected` both before and after the copy (tier_protocol.h
// read_record): false when the writer rewrote it meanwhile. The copy is one constant-size memcpy between the two seq
// loads, and nothing after the second load reads the shared record.
template <class S>
bool read_seqlocked(const uint8_t* record, uint32_t expected, uint8_t (&raw)[S::kRecordBytes]) {
  if (load_acquire(record) != expected) return false;
  std::memcpy(raw, record, S::kRecordBytes);
  std::atomic_thread_fence(std::memory_order_acquire);
  asm volatile("" ::: "memory");
  return load_acquire(record) == expected;
}

template <class S>
uint32_t gate(const uint8_t* lease) {
  return load_acquire(lease + S::kGate);
}

// A locked cmpxchg on the host line is atomic against the device's posted stores to it. (ram_tier.h cas_gate)
template <class S>
void cas_gate(uint8_t* lease, uint32_t expected, uint32_t desired) {
  __atomic_compare_exchange_n(
      reinterpret_cast<uint32_t*>(lease + S::kGate), &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
}

// Publishes done[G], then opens the gate if the device closed it for G: the host half of the Dekker pair with the
// device close_gate. A stale open for G meets closed(G + k) and changes nothing. (ram_tier.h copy_completed)
template <class S>
void complete(uint8_t* lease, uint32_t seq, uint64_t gen) {
  store_release64(lease + S::kDone + static_cast<int64_t>((seq - 1u) % S::kRecords) * S::kDoneBytes, gen);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const uint32_t closed = gate_word(seq, kGateClosed);
  if (gate<S>(lease) == closed) cas_gate<S>(lease, closed, gate_word(seq, kGateOpen));
}

// Teardown: with no completer left, a closed wait would hold its stream forever. The device commit then traps unless
// done[G] is there. (ram_tier.h open_closed_gate)
template <class S>
void open_closed_gate(uint8_t* lease) {
  const uint32_t g = gate<S>(lease);
  if ((g & 0x80000000u) != 0) cas_gate<S>(lease, g, g & ~0x80000000u);
}

}  // namespace sglang::expert_stream::channel
```

- [ ] **Step 6: Write `stream_wait.h`.** Move `stream_wait_value32`, `StreamWaitValue32` and `kStreamWaitValueGeq`
  out of `row_copy_kernels.cuh:368-380` verbatim, and add:

```cpp
// Queues the stream's wait for the gate at `gate_address` to read open (lease_channel_layout.h); captured as a
// memory-op node when the stream is capturing. Throws if the driver lacks cuStreamWaitValue32_v2 or the call fails.
inline void enqueue_gate_wait(cudaStream_t stream, uint64_t gate_address) {
  const auto fn = stream_wait_value32();
  if (fn == nullptr)
    throw std::runtime_error("the gate wait needs cuStreamWaitValue32_v2, which libcuda.so.1 does not provide");
  const int r = fn(static_cast<void*>(stream), gate_address, ::sglang::expert_stream::channel::kGateOpen,
                   kStreamWaitValueGeq);
  if (r != 0) throw std::runtime_error("cuStreamWaitValue32_v2 on the gate failed: CUresult " + std::to_string(r));
}
```

  `row_copy_kernels.cuh` includes it. `lease_copy_wait` (`:543-552`) replaces its inline call and `RuntimeCheck` with
  `enqueue_gate_wait(stream, lease_address + Wire::kLeaseCopyGate)`. Keep the earlier
  `RuntimeCheck(stream_wait_value32() != nullptr, ...)`, so the refusal still comes before CW is launched.

- [ ] **Step 7: Port the target.** Mechanical. Each call site keeps its comment, and the body becomes the template
  call:
  - `lease_layout.h`: `#include "lease_channel_layout.h"`. Define `kLeaseGateClosed/kLeaseGateOpen/kLeaseGateSeqShift/
    kLeaseGateSeqMask` as `channel::kGate*`. Add the `TargetChannel` alias after `Wire`, with
    `static_assert(TargetChannel::kDone == Wire::kLeaseCopyDone && TargetChannel::kGate == Wire::kLeaseCopyGate)`.
  - `lease_device.cuh`:
    - `copy_gate_word` returns `channel::gate_word`;
    - `ring_index` returns `channel::ring_index<TargetChannel>`;
    - `pending_generation` returns `channel::generation(seq, epoch)`;
    - in `write_record`, the `seq = 0` store and fence become `channel::begin_record<TargetChannel>(record)`, and the
      final store at `:256` becomes `channel::end_record<TargetChannel>(record, seq)`.
  - `lease_kernels.cuh`:
    - `:170-176` becomes `const uint32_t seq = channel::advance(state);` followed by
      `const uint32_t epoch = static_cast<uint32_t>(state[kEpoch]);`;
    - `:240` becomes `channel::publish_head<TargetChannel>(p.page, seq);`.
  - `row_copy_kernels.cuh`:
    - CW's tail (`st_relaxed_sys` close through the self-open) becomes
      `channel::close_gate<TargetChannel>(lease, seq, generation);`;
    - CC's trap line becomes `channel::commit_or_trap<TargetChannel>(p.lease, seq, generation);`.
  - `tier_protocol.h`:
    - `read_record`'s first seq check, memcpy, fence and second check become
      `if (!channel::read_seqlocked<TargetChannel>(record, expected, raw)) return RecordRead::kTorn;`;
    - `load_acquire` delegates to `channel::load_acquire`.
  - `ram_tier.h`:
    - `copy_completed` becomes `channel::complete<TargetChannel>(lease_, seq, job.gen);`;
    - `gate_word` returns `channel::gate_word`;
    - `cas_gate` delegates to `channel::cas_gate<TargetChannel>(lease_, ...)`;
    - `open_closed_gate` delegates to `channel::open_closed_gate<TargetChannel>(lease_)`.
  - `lease_layout_probe.cpp`: emit `chan_head`, `chan_ring`, `chan_records`, `chan_record_bytes`, `chan_done`,
    `chan_gate` from `TargetChannel`, and `gate_open_<seq>` and `gate_closed_<seq>` for the four seqs of Step 1, in
    the probe's existing output format.

  Before editing, re-read each named range: line numbers move. No other target code changes.

- [ ] **Step 8: Run the layout test. Expected: PASS** (4 + 1 cases).

- [ ] **Step 9: Run the target protocol suite** (CPU with `-n 8`, then GPU, both listed in Global Constraints).
  Expected: the same pass/skip counts as at this task's BASE. Run the same selection at BASE first and record both
  counts side by side.

- [ ] **Step 10: Mutants**, in a private worktree on divix01 (`.claude/rules/divix01-run-protocol.md`, "Mutants").
  Each must turn the named test red, and the restored tree must be green again.
  - In `lease_channel.cuh` `close_gate`, delete `__threadfence_system();`. Expected red:
    `test_exl3_lease_ordering_cuda.py::test_the_captured_chain_waits_in_cw_and_replays_right_armed_and_unarmed`. If it
    stays green (a fence race is probabilistic), record that, and run the next mutant as the deterministic one.
  - In `commit_or_trap`, replace `__trap();` with `return;`. Expected red:
    `test_exl3_lease_kernels_cuda.py` (a gate opened without CopyDone must trap). Name the exact test that turns red.
  - In `host/lease_channel.h` `complete`, swap the done store and the gate CAS. Expected red:
    `test_exl3_ram_miss_cpu_experts.py` or `test_exl3_ram_miss_copy_engine.py` (a CopyDone-before-gate assertion).
    Name the test.

  Record each result in the ledger.

- [ ] **Step 11: `LEASE_PROTOCOL.md`.** Add an opening section, "The lease channel", that states:
  - the page/lease split;
  - the seqlock;
  - `G`;
  - done[];
  - the gate encoding;
  - the Dekker pair (device `close_gate` vs host `complete`);
  - the wait node;
  - the commit trap.

  Name the target as its first client (`TargetChannel`). The existing "Wire (v2)" and "Copy engine" sections link to
  it instead of restating the gate. Change no other wording.

- [ ] **Step 12: Commit** in two parts:
  - `test(lease-channel): the target's channel names the wire's own offsets and gate encoding (failing)`, with the test;
  - `refactor(lease-channel): one templated lease channel; the target is its first client, byte-identical`, with the
    rest.

---

### Task 2: The draft's GPU share on a slot slab, graph-safe

**Files:**
- Create: `python/sglang/srt/layers/quantization/exl3/draft_moe.py`
- Test: `test/registered/unit/layers/quantization/test_exl3_draft_moe.py` (CPU, the remap)
- Test: `test/manual/dsv41/test_exl3_draft_moe_gpu.py` (GPU, parity, sink, capture)

**Interfaces:**
- Consumes: `Exl3FusedMoE(tensors, slots, hidden, inter, top_k, device, tokens)` and `.run(x, topk_weights, remap,
  keep, act_limit)` (`fused_moe.py:103`, `:179`); `EXL3_STREAMED_NAMES` (`exl3_expert_format.py:43`).
- Produces:

```python
class DraftResidentMoe:
    """A draft stage's GPU experts as one slot slab, for Exl3FusedMoE. Route r goes to slot expert_to_slot[id];
    every id that is not resident (CPU-owned, -1, out of range) goes to the sink slot `slots`, which the fused
    kernel ignores."""
    TOKENS = 16  # ROW_TILE: one fused call's tokens; a larger M runs in chunks of 16
    def __init__(self, layer, resident_ids: Sequence[int], n_experts: int, device): ...
    slots: int
    expert_to_slot: torch.Tensor  # int64 [n_experts + 1] on device; index n_experts is the sink entry for bad ids
    def prepare(self) -> None: ...  # builds the Exl3FusedMoE (host work); call before capture
    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor: ...  # [M, k] ids -> [M * k] int64 slots, device only
    def run(self, x, topk_ids, topk_weights, act_limit) -> torch.Tensor: ...  # fp32 [M, H], M <= TOKENS
```

- [ ] **Step 1: Write the CPU remap test.**

```python
"""DraftResidentMoe's route remap: resident ids to their slot, everything else to the sink (CPU)."""

import torch

from sglang.srt.layers.quantization.exl3.draft_moe import DraftResidentMoe, draft_slot_map
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_resident_ids_map_to_their_slot_and_the_rest_to_the_sink():
    table = draft_slot_map([3, 0, 7], n_experts=8, device="cpu")
    ids = torch.tensor([[0, 3, 5], [7, -1, 8], [1, 0, 2]])
    remap = DraftResidentMoe.remap_with(table, ids, n_experts=8)
    sink = 3
    assert remap.tolist() == [0, 1, sink, 2, sink, sink, sink, 0, sink]
    assert remap.dtype == torch.int64


def test_slots_follow_the_sorted_resident_ids():
    table = draft_slot_map([5, 2], n_experts=6, device="cpu")
    assert table[:6].tolist() == [2, 2, 0, 2, 2, 1]
    assert table[6].item() == 2  # the bad-id entry is the sink too
```

- [ ] **Step 2: Run it. Expected: FAIL** with `ModuleNotFoundError: ... draft_moe`.

- [ ] **Step 3: Write `draft_moe.py`.**

```python
"""The DSpark draft's GPU experts in the decode graph: one slot slab and Exl3FusedMoE (D2-1's in-graph MoE).

A stage's resident experts (all of them without CPU experts) are copied into a [slots, part, ...] slab per EXL3 name,
in sorted id order. expert_to_slot sends each resident id to its slot and every other id (CPU-owned, -1, out of range)
to the sink slot `slots`: exllamav3's exl3_moe ignores expert_count's last column, so sink routes add nothing. Nothing
here reads the device: remap is a gather on a device table.
"""

from typing import Sequence

import torch

from sglang.srt.layers.quantization.exl3.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.quantization.exl3.fused_moe import ROW_TILE, Exl3FusedMoE


def draft_slot_map(resident_ids: Sequence[int], *, n_experts: int, device) -> torch.Tensor:
    """int64 [n_experts + 1]: slot of each resident id, `len(resident_ids)` (the sink) for the rest and at index
    n_experts (where remap sends ids outside [0, n_experts))."""
    ids = sorted(set(int(e) for e in resident_ids))
    if any(e < 0 or e >= n_experts for e in ids):
        raise ValueError(f"DSpark draft resident ids {ids} outside [0, {n_experts})")
    table = torch.full((n_experts + 1,), len(ids), dtype=torch.int64)
    table[ids] = torch.arange(len(ids), dtype=torch.int64)
    return table.to(device)


class DraftResidentMoe:
    TOKENS = ROW_TILE

    def __init__(self, layer, resident_ids: Sequence[int], n_experts: int, device):
        self.ids = sorted(set(int(e) for e in resident_ids))
        if not self.ids:
            raise ValueError("a DSpark draft stage with no GPU expert has no GPU share; CPU-only stages are not supported")
        self.n_experts = n_experts
        self.slots = len(self.ids)
        self.expert_to_slot = draft_slot_map(self.ids, n_experts=n_experts, device=device)
        index = torch.tensor(self.ids, dtype=torch.long)
        self.tensors = {n: getattr(layer, n).data.index_select(0, index).to(device).contiguous() for n in EXL3_STREAMED_NAMES}
        self.hidden = layer.hidden_size
        self.inter = layer.intermediate_size_per_partition
        self.top_k = layer.top_k
        self.device = device
        self.keep = torch.ones(1, dtype=torch.float32, device=device)
        self.fused = None

    def prepare(self) -> None:
        if self.fused is None:
            self.fused = Exl3FusedMoE(
                self.tensors, self.slots, self.hidden, self.inter, self.top_k, self.device, tokens=self.TOKENS
            )

    @staticmethod
    def remap_with(table: torch.Tensor, topk_ids: torch.Tensor, *, n_experts: int) -> torch.Tensor:
        ids = topk_ids.reshape(-1).to(torch.int64)
        bad = (ids < 0) | (ids >= n_experts)
        return table[torch.where(bad, torch.full_like(ids, n_experts), ids)]

    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor:
        return self.remap_with(self.expert_to_slot, topk_ids, n_experts=self.n_experts)

    def run(self, x, topk_ids, topk_weights, act_limit) -> torch.Tensor:
        if self.fused is None:
            raise RuntimeError("DraftResidentMoe.run before prepare()")
        return self.fused.run(x, topk_weights.reshape(-1), self.remap(topk_ids), self.keep, act_limit)
```

  Before writing it, check the attribute names on the layer: `hidden_size`, `intermediate_size_per_partition` and
  `top_k` as `FusedMoE` defines them, and how `exl3_fused_moe_for` (`fused_moe.py:265`) reads hidden and inter. Use
  the same expressions it uses. `torch.where` here has a static shape (elementwise), so it is capture-safe.

- [ ] **Step 4: Run the CPU test. Expected: PASS** (2/2).

- [ ] **Step 5: Write the GPU test.** Template: `test_exl3_fused_moe_multitoken_gpu.py` and
  `test_exl3_moe_probe_gpu.py`, for loading real EXL3 slots and building the dense reference. Cases:
  1. **Parity, eager.** For M ∈ {1, 2, 5, 6, 16}, with all 8 test experts resident, `run` matches
     `exl3_moe_loop(x, w, ids, ...)` within the tolerance `test_exl3_fused_moe_multitoken_gpu.py` uses.
  2. **The sink drops routes.** Resident = {0, 2, 5}; ids include 1 and 7 (not resident), -1 and 9 (out of range).
     `run` equals `exl3_moe_accumulate(..., experts=[0, 2, 5])` over the same ids. Run it with
     `SGLANG_DSV41_ENABLE_LAYER_FUSION` on and off (`envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override`).
  3. **Capture.** `prepare()`, one eager warmup, then capture `run` at M = 5 under `torch.cuda.graph`. Replay with
     new ids and weights copied into the static inputs. Replay equals eager, byte-exact.
  4. **No host read.** Under `torch.cuda.set_sync_debug_mode("error")`, an eager `run` after `prepare()` raises nothing.

- [ ] **Step 6: Run the GPU test. Expected:** case 2 decides the sink. If it fails, the pinned exllamav3 does not
  ignore the last column. Then rule in the ledger, and change `remap` to send non-resident routes to slot 0 with
  their weight zeroed: `topk_weights * (~sink).float()`. That is still static. Re-run until all four cases pass.

- [ ] **Step 7: Commit:** `test(dspark): the draft's GPU share on a slot slab (failing)`, then
  `feat(dspark): DraftResidentMoe, the draft's resident experts through Exl3FusedMoE with a sink slot`.

---

### Task 3: The draft channel: its spec, its page, and the post/finish kernels

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/draft_channel.h` (the spec and record, shared by host and
  device)
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/draft_kernels.cuh` (post, finish, commit)
- Create: `python/sglang/kernels/ops/moe/dspark_draft_cpu.py` (the layout mirror, the device ops, later the host)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout_probe.cpp` (draft keys)
- Test: `test/registered/unit/kernels/test_dspark_draft_channel_layout.py` (CPU)
- Test: `test/manual/dsv41/test_dspark_draft_channel_cuda.py` (GPU, with a Python host stand-in)

**Interfaces:**
- Consumes: Task 1's `ChannelSpec`, the device `channel::*` functions and `enqueue_gate_wait`.
- Produces:

```cpp
namespace sglang::expert_stream::draft {
constexpr int kMaxRows = 16;    // DraftResidentMoe.TOKENS: one call's tokens
constexpr int kMaxK = 8;        // routes per token
// Record (64 B payload in a 128-B slot): seq u32 @0, stage u16 @4, rows u8 @6, k u8 @7, epoch u32 @8.
constexpr int64_t kRecSeq = 0, kRecStage = 4, kRecRows = 6, kRecK = 7, kRecEpoch = 8;
using DraftChannel = channel::ChannelSpec</*Head*/ 0, /*Ring*/ 128, /*Records*/ 4, /*RecordBytes*/ 128,
                                          /*Done*/ 640, /*Gate*/ 768>;
constexpr int64_t kChannelBytes = 4096;  // page and completion block in one pinned buffer
}
```

- Python (`dspark_draft_cpu.py`):

```python
@dataclass(frozen=True)
class DraftWire:  # mirrors draft_channel.h; test_dspark_draft_channel_layout checks the two agree
    head: int = 0; ring: int = 128; records: int = 4; record_bytes: int = 128; done: int = 640; gate: int = 768
    channel_bytes: int = 4096; max_rows: int = 16; max_k: int = 8

class DraftCpuAreas:
    """The draft channel's pinned buffers: `channel` uint8 [4096] (page + completion), and per stage `x` fp16
    [stages, 16, H], `slots` int32 [stages, 16, 8], `weights` fp32 [stages, 16, 8], `out` fp32 [stages, 16, H]."""
    def __init__(self, stages: int, hidden: int, *, pin: bool = True): ...

class DraftCpuDevice:
    """The device side: `state` int32 [6] on the GPU (channel words), `on_cpu` uint8 [stages, E]."""
    def __init__(self, areas: DraftCpuAreas, on_cpu: torch.Tensor, device): ...
    def post(self, stage: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None: ...
    def finish(self, stage: int, out: torch.Tensor) -> None: ...  # out fp32 [M, H] on the GPU, added into
```

- [ ] **Step 1: Write the CPU layout test.** It uses the probe, as in Task 1:

```python
"""The draft channel's offsets and limits: Python's DraftWire mirrors draft_channel.h."""

from sglang.kernels.ops.moe import expert_lease_block
from sglang.kernels.ops.moe.dspark_draft_cpu import DraftWire
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def test_the_python_mirror_is_the_header():
    probe = expert_lease_block.wire_probe(8, 1)
    w = DraftWire()
    assert (probe["draft_head"], probe["draft_ring"], probe["draft_records"], probe["draft_record_bytes"]) == (
        w.head, w.ring, w.records, w.record_bytes)
    assert (probe["draft_done"], probe["draft_gate"], probe["draft_channel_bytes"]) == (w.done, w.gate, w.channel_bytes)
    assert (probe["draft_max_rows"], probe["draft_max_k"]) == (w.max_rows, w.max_k)
```

- [ ] **Step 2: Run it. Expected: FAIL** with an `ImportError` (`dspark_draft_cpu`).

- [ ] **Step 3: Write `draft_channel.h`** with the namespace block above, plus `static_assert(DraftChannel::kGate +
  4 <= kChannelBytes)`. Add the probe keys to `lease_layout_probe.cpp`. Write `DraftWire` and `DraftCpuAreas`
  (`torch.zeros(..., pin_memory=pin)`) in `dspark_draft_cpu.py`.

- [ ] **Step 4: Run the layout test. Expected: PASS.**

- [ ] **Step 5: Write the GPU kernel test.** A Python thread stands in for the host half. It polls
  `areas.channel`'s head through a numpy view, checks the record's seq, writes `out[stage, :rows] = rows-dependent
  values`, and then mimics `complete`. It stores done (u64) and sets the gate to `open` if it reads `closed`. The real
  host half is Task 4's. This test pins the device half. Cases:
  1. **Post.** `post(stage=1, x[5, H] bf16, ids[5, 3], w[5, 3])` with `on_cpu[1]` true for {2, 4}. After
     `synchronize`:
     - `areas.x[1, :5]` equals `x.half()`;
     - `areas.slots[1, :5]` holds the id where `on_cpu`, else -1;
     - `areas.weights[1, :5]` holds the weight where the slot ≥ 0, else 0;
     - the record reads seq 1, stage 1, rows 5, k 3;
     - the head reads 1.
  2. **No CPU route, no post.** `on_cpu` all false: the head stays where it was, and `finish` leaves `out` unchanged
     with no wait (the stand-in thread never answers, and the call returns).
  3. **Finish waits and adds.** The stand-in answers after 50 ms. `finish(1, out)` returns at once (async).
     `synchronize` takes ≥ 50 ms, and `out` gains the stand-in's rows exactly.
  4. **Captured.** Capture `post` + a dummy GPU op + `finish` at M = 5 in `torch.cuda.graph`. Replay 3 times with the
     stand-in answering. Each replay advances seq by 1, the gate is open after each, and `out` is right each time.
  5. **Commit traps without done.** In a `spawn_child` subprocess (core dumps off), the stand-in opens the gate
     without storing done. The child dies with a CUDA error (illegal instruction / unspecified launch failure).

- [ ] **Step 6: Run it. Expected: FAIL** with `AttributeError` (no `DraftCpuDevice`).

- [ ] **Step 7: Write `draft_kernels.cuh`** and its loader. The kernels use only the Task 1 channel functions for the
  protocol:

```cpp
// The draft channel's device side (draft_channel.h, LEASE_PROTOCOL.md "The lease channel", second client).
//   post    stages x [M, H] (fp16) and the CPU-owned routes [M, k] of stage `stage`, writes the record and publishes the
//           head; posts nothing when no route is on the CPU (state[kPending] = 0)
//   finish  closes the gate for the pending G (and opens it itself if done[G] already holds), then the stream waits
//   commit  checks done[G] (traps without it) and adds the stage's CPU output into `out`
#pragma once
#include "lease_channel.cuh"
#include "draft_channel.h"
#include "stream_wait.h"

namespace device::expert_stream::draft {
using namespace ::sglang::expert_stream::draft;
namespace ch = ::device::expert_stream::channel;

struct PostParams {
  const void* x; int32_t x_dtype;          // 0 fp16, 1 bf16, 2 fp32
  const int64_t* ids; const float* weights; // [M, k] each, contiguous
  const uint8_t* on_cpu;                   // [E] of this stage
  int32_t rows, k, experts, stage, hidden;
  int32_t* state;                          // device, kStateWords
  uint8_t* channel;                        // pinned (UVA)
  __half* x_dst; int32_t* slots_dst; float* weights_dst;  // pinned, this stage's areas
};

__global__ void draft_post_kernel(const __grid_constant__ PostParams p) {
  __shared__ int any;
  if (threadIdx.x == 0) any = 0;
  __syncthreads();
  for (int r = threadIdx.x; r < p.rows * p.k; r += blockDim.x) {
    const int t = r / p.k, i = r % p.k;
    const int64_t id = p.ids[r];
    const bool cpu = id >= 0 && id < p.experts && p.on_cpu[id];
    __stwt(p.slots_dst + t * kMaxK + i, cpu ? static_cast<int32_t>(id) : -1);
    __stwt(p.weights_dst + t * kMaxK + i, cpu ? p.weights[r] : 0.0f);
    if (cpu) any = 1;  // benign race: every writer stores 1
  }
  __syncthreads();
  if (!any) {
    if (threadIdx.x == 0) p.state[ch::kPending] = 0;
    return;
  }
  for (int64_t e = threadIdx.x; e < static_cast<int64_t>(p.rows) * p.hidden; e += blockDim.x) {
    float v = p.x_dtype == 0 ? __half2float(static_cast<const __half*>(p.x)[e])
            : p.x_dtype == 1 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.x)[e])
                             : static_cast<const float*>(p.x)[e];
    __stwt(p.x_dst + e, __float2half(v));
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) return;
  const uint32_t seq = ch::advance(p.state);
  const uint32_t epoch = static_cast<uint32_t>(p.state[ch::kEpoch]);
  uint8_t* rec = ch::record_at<DraftChannel>(p.channel, seq);
  ch::begin_record<DraftChannel>(rec);
  st_relaxed_sys<uint32_t>(rec + kRecStage,
      (static_cast<uint32_t>(p.stage) & 0xFFFFu) | (static_cast<uint32_t>(p.rows) << 16) | (static_cast<uint32_t>(p.k) << 24));
  st_relaxed_sys<uint32_t>(rec + kRecEpoch, epoch);
  ch::end_record<DraftChannel>(rec, seq);
  ch::publish_head<DraftChannel>(p.channel, seq);
  p.state[ch::kPending] = static_cast<int32_t>(seq);
  p.state[ch::kPendingEpoch] = static_cast<int32_t>(epoch);
}

struct FinishParams { const int32_t* state; uint8_t* channel; };

__global__ void draft_finish_kernel(const __grid_constant__ FinishParams p) {
  if (threadIdx.x != 0) return;
  const uint32_t seq = static_cast<uint32_t>(p.state[ch::kPending]);
  if (seq == 0) return;  // nothing posted: the gate stays open from the last completion
  const uint32_t epoch = static_cast<uint32_t>(p.state[ch::kPendingEpoch]);
  ch::close_gate<DraftChannel>(p.channel, seq, ch::generation(seq, epoch));
}

struct CommitParams {
  const int32_t* state; const uint8_t* channel;
  const float* cpu_out;  // pinned, this stage's [kMaxRows, hidden]
  float* out;            // device [rows, hidden]
  int32_t rows, hidden;
};

__global__ void draft_commit_kernel(const __grid_constant__ CommitParams p) {
  const uint32_t seq = static_cast<uint32_t>(p.state[ch::kPending]);
  if (seq == 0) return;
  // Every block checks with its own acquire before reading the host's rows: one block's acquire orders only its own
  // later reads.
  if (threadIdx.x == 0)
    ch::commit_or_trap<DraftChannel>(p.channel, seq,
                                     ch::generation(seq, static_cast<uint32_t>(p.state[ch::kPendingEpoch])));
  __syncthreads();
  const int64_t n = static_cast<int64_t>(p.rows) * p.hidden;
  for (int64_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x)
    p.out[e] += __ldcv(p.cpu_out + e);
}

}  // namespace device::expert_stream::draft
```

  A wait with nothing posted still passes, because the gate holds the last `open(G)`, and an untouched gate at
  start is 0. 0 is below `kGateOpen`, so the cyclic GEQ blocks. **`DraftCpuAreas` must therefore initialise the gate
  word to `gate_word(0, kGateOpen)` = 1.**

  The launch wrapper class `DraftChannelKernels` has static `post(...)` and `finish(...)`, with `TensorView` checks in
  the style of `lease_copy_wait`. `finish` launches `draft_finish_kernel`<<<1, 32>>>, then
  `enqueue_gate_wait(stream, channel + DraftChannel::kGate)`, then `draft_commit_kernel`<<<min(ceil(rows*hidden/256),
  64), 256>>>. All three are on the current stream, so a capture records the wait node between them.

  The Python loader in `dspark_draft_cpu.py` is
  `load_jit("dspark_draft_channel", cuda_files=["moe/expert_stream/draft_kernels.cu"], cuda_wrappers=[...])`,
  following `_device_module_cached` (`expert_stream_transport.py:2221`). Create `draft_kernels.cu` as the one-line
  translation unit including the header, if `load_jit` needs a `.cu`. Check what `cuda_files` accepts:
  `exl3_ram_miss.cuh` is passed directly, so a `.cuh` works too.

  `DraftCpuDevice.post` passes `x_dtype` from `x.dtype`, `ids` as int64 contiguous, `weights` as fp32 contiguous, and
  the stage's area pointers. It raises if `rows > 16` or `k > 8`; those are host-side shape checks, so they are
  capture-safe.

- [ ] **Step 8: Run the GPU kernel test. Expected: PASS** (5/5).

- [ ] **Step 9: Run the target protocol suite.** Expected: unchanged counts. Task 3 only adds probe keys and new
  files, but the probe file is shared.

- [ ] **Step 10: Commit:** `test(dspark): the draft channel's layout and device half (failing)`, then
  `feat(dspark): the draft channel, the lease channel's second client: post, finish and commit kernels`.

---

### Task 4: The draft CPU thread: the host half of the draft channel

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (exports `draft_cpu_open`,
  `draft_cpu_set_layer`, `draft_cpu_start`, `draft_cpu_stop`, `draft_cpu_stats`; macro list `:785`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (the fake kernel takes M rows;
  `draft_test_post`, `draft_test_tear`)
- Modify: `python/sglang/kernels/ops/moe/dspark_draft_cpu.py` (`DraftCpuHost`)
- Modify: `python/sglang/srt/environ.py` (`SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US`)
- Test: `test/registered/unit/kernels/test_dspark_draft_cpu_thread.py` (CPU, instr build, fake kernel)

**Interfaces:**
- Consumes: Task 1's host `channel::*` functions; Task 3's `DraftChannel`, record offsets and `DraftCpuAreas`;
  `CpuExpertKernel::forward/keep_warm(…, release_ns)`; `idle_budget`; `fail_stop`; `now_ns`; `Base::layer_shape` and
  `Base::params_bytes` (`ffi_exports.h:385`).
- Produces:

```python
class DraftCpuHost:
    def __init__(self, areas: DraftCpuAreas, *, cores: Sequence[int], threads: int, spin_us: int, keep_warm_us: int,
                 fatal_wait_s: float, variant: Optional[str] = None): ...
    def set_layer(self, stage: int, kernel: int, spec) -> None: ...  # spec: CpuExpertLayerSpec; before start()
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def stats(self) -> dict: ...  # {"jobs", "rows", "forward_ns", "keep_warm_calls"}
```

```cpp
// host/draft_cpu_thread.h
class DraftCpuThread {
 public:
  struct Config { uint8_t* channel; const uint8_t* x; const int32_t* slots; const float* weights; float* out;
                  int64_t hidden; int stages; int threads; std::vector<int> cores;
                  int64_t spin_ns, keep_warm_ns, fatal_wait_ns; };
  explicit DraftCpuThread(Config);
  void set_layer(int stage, cpu_experts::ExpertLayer layer);
  void start(); void stop();
  int64_t jobs() const; int64_t rows() const; int64_t forward_ns() const;
};
```

- [ ] **Step 1: Write the CPU test.** Use the instr build, and drive the device side through the test export
  `draft_test_post(channel, stage, rows, k, seq, epoch)`. It writes exactly what `draft_post_kernel`'s thread 0 writes,
  in the same order, with host atomics. A test helper `_finish(areas, timeout)` mimics `draft_finish_kernel` plus the
  wait and commit:
  - it sets the gate to closed(seq) and reads done after a fence (opening the gate itself if done is already there);
  - it then spins until the gate reads open;
  - it then asserts done equals G.

  Cases:
  1. **One stage, M rows.** Write x, slots and weights for 5 rows (`slots[t] = [t, -1, 2]`), post, and finish:
     - `out[stage, t, j]` equals the fake's multi-row formula, `j + Σ_i w·(s+1)` over row t's valid slots;
     - `test_kernel_calls` shows one call per row, with core = `cores[0]`;
     - `stats()["jobs"] == 1` and `stats()["rows"] == 5`.
  2. **Three stages in turn**, as a draft step does. Each finishes with its own stage's layer: the fake records the
     layer's hidden size, and each stage's layer gets a different hidden size.
  3. **The GPU's head store ends the hold.**
     - With `keep_warm_us = 500_000` and `spin_us = -1`, after one job `test_keep_warm_calls() >= 1`.
     - A post then finishes in < 5 ms (no sleep to wake from), and `test_keep_warm_calls()` grows by exactly 1 after
       it.
     - A mutant that hands keep_warm a word the device never writes must turn this red.
  4. **Idle sleep.** With `spin_us = 20_000` and `keep_warm_us = 0`, 0.3 s after a job the draft thread's CPU time
     (from `/proc/self/task/*/comm` == `dspark-cpu`, the `_engine_cpu_s` pattern of `test_cpu_expert_keep_warm.py`)
     grows by < 0.05 s. The next post still finishes, within 50 µs plus the forward, so assert < 20 ms.
  5. **A refused forward fail-stops.** In a `spawn_child` child, the fake has `fail=1`. The child exits by abort, and
     its stderr names `DSpark draft CPU experts` and the seq.
  6. **A torn record fail-stops.** `draft_test_tear` publishes a head whose record seq word is 0. The child aborts,
     and stderr says `torn`.
  7. **The watchdog.** The fake is held (`test_kernel_hold(core)`) and `fatal_wait_s = 0.5`. The child aborts within 2
     s, and stderr names the draft and the seq.
  8. **Stop with a closed gate.** Post with the fake held, close the gate (step 1 of `_finish` only), then release the
     hold and call `stop()` at once. `stop()` returns within 5 s, and the gate reads open.

  Every child-process case goes through `spawn_child` (`dsv41_ram_miss_fixtures.py`) with the instr variant warmed in
  the parent.

- [ ] **Step 2: Run it. Expected: FAIL** with `AttributeError: ... DraftCpuHost`.

- [ ] **Step 3: Make the fake kernel take M rows** (`ffi_test_exports.h:690-722`). Loop `t` over `c.rows` (`max(1,
  c.rows)`): row t's sum reads `c.slots[t*c.k + i]` and `c.weights[t*c.k + i]`, and writes `c.out[t*hidden + j]`.
  Push one `Call` per row. Rows = 1 is exactly today's behavior, so every existing fake-kernel test is unaffected.
  Also record the layer's hidden size in `Call` (an extra trailing field). `test_kernel_calls` widens to read it, and
  the existing keys are unchanged.

- [ ] **Step 4: Write `draft_cpu_thread.h`.** The loop, with each protocol step a Task 1 call:

```cpp
// The DSpark draft's CPU experts, the host half of the draft channel (draft_channel.h): one thread, the OpenMP master
// of the draft's team on the draft cores (ThreadingConfig.draft_cpus). It reads each posted record, runs the stage's
// M-row forward over the staged x and routes into the stage's out rows, and completes the record through the lease
// channel (done, then the Dekker open of the gate).
//
// Idle: for keep_warm_ns after a job the team runs register work, then PAUSE, inside the kernel's keep_warm, which
// watches the channel's head word: the GPU's release store of the next head ends the hold with no syscall. spin_ns
// after the warm window the hold releases the team and the thread polls the head with 50 us sleeps (the GPU cannot
// ring a futex). A negative spin_ns holds the team until the next post, however long.
//
// Failure: a refused forward, a torn record or a lapped ring (the device posts once and waits) fail-stops; a watchdog
// thread fail-stops when a posted record stays incomplete for fatal_wait_ns.
class DraftCpuThread {
 public:
  // ... Config as in Interfaces ...
  void run() {
    pin();  // pthread_setname_np "dspark-cpu", affinity cores[0]
    constexpr int64_t kNever = INT64_MAX;
    uint32_t next = 1;
    int64_t warm_until = 0;
    int64_t release_at = config_.spin_ns < 0 ? kNever : now_ns() + config_.spin_ns;
    const uint64_t spin_iters = idle_budget(config_.spin_ns);
    uint64_t idle = 0;
    while (!stop_.load(std::memory_order_acquire)) {
      const uint32_t head = channel::load_acquire(config_.channel + DraftChannel::kHead);
      if (head != 0 && reached(head, next)) {
        if (head != next) fail_stop("DSpark draft CPU experts: record " + std::to_string(next) + " lapped (head " +
                                    std::to_string(head) + "); the device posts one record per wait");
        warm_until = serve(next) + config_.keep_warm_ns;
        release_at = config_.spin_ns < 0 ? kNever : warm_until + config_.spin_ns;
        next = skip_zero(next + 1u);
        idle = 0;
      } else if (release_at != 0) {
        hold(head, warm_until, release_at);
        if (channel::load_acquire(config_.channel + DraftChannel::kHead) == head) release_at = 0;
      } else if (++idle >= spin_iters) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      } else {
        _mm_pause();
      }
    }
  }

  // Reads record `seq`, runs its forward, completes it; returns the forward's end.
  int64_t serve(uint32_t seq) {
    alignas(64) uint8_t raw[DraftChannel::kRecordBytes];
    const uint8_t* rec = channel::record_at<DraftChannel>(config_.channel, seq);
    if (!channel::read_seqlocked<DraftChannel>(rec, seq, raw))
      fail_stop("DSpark draft CPU experts: record " + std::to_string(seq) + " torn");
    uint32_t word, epoch;
    std::memcpy(&word, raw + kRecStage, 4);
    std::memcpy(&epoch, raw + kRecEpoch, 4);
    const int stage = word & 0xFFFF, rows = (word >> 16) & 0xFF, k = word >> 24;
    if (stage >= config_.stages || rows < 1 || rows > kMaxRows || k < 1 || k > kMaxK || !layers_[stage].kernel)
      fail_stop("DSpark draft CPU experts: record " + std::to_string(seq) + " malformed (stage " +
                std::to_string(stage) + ", rows " + std::to_string(rows) + ", k " + std::to_string(k) + ")");
    // The slot and weight areas are kMaxK wide per token; the kernel reads [rows, k] contiguous, so compact them.
    int32_t slots[kMaxRows * kMaxK];
    float weights[kMaxRows * kMaxK];
    const int32_t* s = config_.slots + static_cast<int64_t>(stage) * kMaxRows * kMaxK;
    const float* w = config_.weights + static_cast<int64_t>(stage) * kMaxRows * kMaxK;
    for (int t = 0; t < rows; ++t)
      for (int i = 0; i < k; ++i) {
        slots[t * k + i] = s[t * kMaxK + i];
        weights[t * k + i] = w[t * kMaxK + i];
      }
    cpu_experts::ForwardCall call;
    call.rows = rows; call.k = k; call.threads = config_.threads; call.cores = config_.cores;
    call.x = config_.x + static_cast<int64_t>(stage) * kMaxRows * config_.hidden * 2;
    call.slots = slots; call.weights = weights;
    call.out = config_.out + static_cast<int64_t>(stage) * kMaxRows * config_.hidden;
    const int64_t start = now_ns();
    try {
      layers_[stage].kernel->forward(layers_[stage], call);
    } catch (const std::exception& e) {
      fail_stop("DSpark draft CPU experts: forward of record " + std::to_string(seq) + " (stage " +
                std::to_string(stage) + ") failed: " + e.what());
    }
    const int64_t end = now_ns();
    add(forward_ns_, end - start); add(jobs_, 1); add(rows_, rows);
    channel::complete<DraftChannel>(config_.channel, seq, static_cast<uint64_t>(epoch) << 32 | seq);
    completed_.store(seq, std::memory_order_release);
    return end;
  }
  // hold(): config.kernel->keep_warm(cores, threads, reinterpret_cast<const uint32_t*>(channel + kHead), head,
  //         warm_until, release_at), fail_stop on an exception, as CpuExpertEngine::hold.
  // watch(): every 20 ms, if head (acquire) != completed_ and the same head has stood for fatal_wait_ns: fail_stop
  //          naming "DSpark draft CPU experts", the seq and fatal_wait. Stops with the thread.
  // stop(): stop_ = true; join the thread and the watchdog; channel::open_closed_gate<DraftChannel>(config_.channel).
};
```

  - **Where `reached`/`skip_zero` come from:** they are `tier_protocol.h`'s (`:99-107`). Include it, or move both into
    `host/lease_channel.h` as part of this task. Moving them is better, since Task 1 made the channel their home: move
    them and make `tier_protocol.h` use them.
  - **The kernel:** all stages share one kernel (the registry enforces one trait), so `hold` uses `layers_[0].kernel`
    once any layer is set. `start()` refuses unless every stage has a layer.
  - **Clock reads:** the first-release `now_ns()` and the forward's two stamps are new.
    `test_exl3_ram_miss_stage_trace_causal.py` scans the expert-stream headers by exact line, so add each new line to
    `NON_TRACE_CLOCK_READS` with a comment naming the draft CPU thread. That test is in the Task 1 suite, so a missing
    entry shows up there.

- [ ] **Step 5: The exports.**
  - `draft_cpu_open(channel, x, slots, weights, out, hidden, stages, threads, cores, spin_ns, keep_warm_ns,
    fatal_wait_ns) -> handle`. It checks:
    - `channel` uint8 `[4096]`;
    - `x` fp16 `[stages, 16, hidden]` (by hand, as `kernel_forward` checks fp16);
    - `slots` int32 and `weights` fp32 `[stages, 16, 8]`;
    - `out` fp32 `[stages, 16, hidden]`, all CPU;
    - cores with `check_engine_cores`' C++ twin: the same rules `enable_cpu_experts` applies. Reuse its code path;
      don't copy it.
  - `draft_cpu_set_layer(handle, stage, kernel, slabs, capacity, hidden, intermediate, activation, act_limit,
    params)`: `make_layer(Base::layer_shape(...), Base::params_bytes(...))`.
  - `draft_cpu_start`, `draft_cpu_stop`, `draft_cpu_stats(handle, out int64 [4])`.
  - **Registry:** a static `std::map<int64_t, std::unique_ptr<DraftCpuThread>>` under a mutex, as `thread_registry`
    is.
  - **Test exports (instr):**
    - `draft_test_post(channel, stage, rows, k, seq, epoch)`;
    - `draft_test_tear(channel, seq)`, which stores the head with the record's seq word left 0;
    - `draft_test_finish_close(channel, seq, epoch) -> int`. It returns 1 if it opened the gate itself, so `_finish`
      uses the same close code as the device's Dekker half, on the host.

  Add the five production exports to `EXPERT_STREAM_HOST_EXPORTS` and the three test exports to the test macro.

- [ ] **Step 6: `DraftCpuHost`** in `dspark_draft_cpu.py`:
  - it wraps the exports through `_host_module(layout, variant)` (`expert_stream_transport.py`);
  - it converts µs to ns with the transport's `_spin_ns` for `spin_us`;
  - `set_layer` uses `_layer_tensors(spec)` and keeps `spec.keep` alive, as `kernel_layer` does (`:1195-1208`).

  Add `SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US = EnvInt(100_000)` to `environ.py` beside
  `SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES`, with a two-line comment in the style of `SGLANG_DSV41_CPU_EXPERTS_IDLE_SPIN_US`.

- [ ] **Step 7: Run the CPU test. Expected: PASS** (8/8). Then run the target protocol suite (CPU with `-n 8`, and
  the GPU list). Expected: unchanged counts. The fake kernel change and the export additions touch every host build.

- [ ] **Step 8: Mutant.** In `run()`, pass `hold` a pointer to a local `uint32_t` instead of the head word. Expected:
  case 3 turns red, because the post is no longer seen until `release_at`. Revert, then re-run green.

- [ ] **Step 9: Commit:** `test(dspark): the draft CPU thread serves the draft channel (failing)`, then
  `feat(dspark): the draft CPU thread, the draft channel's host half, on the lease channel's host functions`.

---

### Task 5: The draft MoE in the graph: `Exl3MoEMethod` on the GPU share and the draft channel

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3/exl3.py`:
  - `get_quant_method` `:109-120`: a `draft=` flag;
  - `_attach_cpu_draft` `:495-527`: build `DraftResidentMoe`, register `on_cpu`;
  - `_apply_cpu_draft` `:529`: post, fused, finish;
  - `apply` `:614`: the fully resident draft goes through `DraftResidentMoe`.
- Modify: `python/sglang/srt/layers/moe/cpu_experts/draft.py`:
  - the registry's `runtime()` builds `DraftCpuAreas`, `DraftCpuDevice` and `DraftCpuHost`;
  - `DraftCpuExperts`' executor path is removed;
  - new `prepare()`.
- Create: `prepare_dspark_draft_graph(model)` in `python/sglang/srt/layers/quantization/exl3/draft_moe.py`
- Modify: `test/registered/unit/layers/quantization/test_exl3_moe_method.py` (`test_apply_adds_the_gpu_and_cpu_shares`
  `:199` mocks the old calls; rewrite it against the new seams)
- Modify: `test/registered/unit/kernels/test_dspark_draft_cpu_experts.py` (the executor tests go; masking and registry
  tests stay)
- Modify: `test/manual/dsv41/test_dspark_hybrid_draft_gpu.py` (eager and captured parity)

**Interfaces:**
- Consumes: `DraftResidentMoe` (Task 2); `DraftCpuAreas`, `DraftCpuDevice` (Task 3); `DraftCpuHost` (Task 4).
- Produces:
  - `DRAFT_CPU_EXPERTS.prepare() -> None`: builds and starts the runtime (host work), idempotent.
  - `DraftCpuExperts.post(key, x, topk_ids, topk_weights)` and `.finish(key, out)`, device-only.
  - `prepare_dspark_draft_graph(model) -> int`: the number of draft MoE layers prepared.

- [ ] **Step 1: Rewrite the unit test seams.**
  - In `test_exl3_moe_method.py`, replace `test_apply_adds_the_gpu_and_cpu_shares` with
    `test_a_cpu_draft_call_posts_runs_the_gpu_share_then_finishes`. Monkeypatch the runtime with a recorder whose
    `post` and `finish` append `("post", key)` and `("finish", key)`. Give the layer a `DraftResidentMoe` stand-in
    whose `run` appends `"gpu"` and returns `torch.ones(M, H)`. Assert:
    - the order is `[("post", k), "gpu", ("finish", k)]`;
    - the output is `ones * scale`, in the dtype of x.
  - Add `test_a_large_m_runs_in_chunks_of_16`: M = 40 gives three post/gpu/finish triples, with rows 16, 16 and 8.
  - Add `test_a_route_neither_resident_nor_on_the_cpu_is_refused_at_attach`: an `on_cpu` mask with a hole outside
    the resident set raises `ValueError` naming the ids.
  - In `test_dspark_draft_cpu_experts.py`, delete the executor tests (`test_the_kernel_runs_on_the_worker...`,
    `test_a_call_with_only_gpu_routes_skips_the_cpu`, the dtype/close one). Keep the masking (`cpu_slots`) and registry
    tests. Add `test_prepare_builds_the_runtime_once`, with a fake host factory counting builds.

- [ ] **Step 2: Run them (CPU). Expected: FAIL** on the new tests (no `post`/`finish`, no `prepare`).

- [ ] **Step 3: Implement.**
  - **`get_quant_method`:** pass `draft=is_dspark_draft_expert_module(prefix)`. `Exl3MoEMethod.__init__` stores it.
  - **`_attach_cpu_draft`:**
    - build `layer.exl3_draft_moe = DraftResidentMoe(layer, sorted(gpu), n_routed + n_fused_shared, device)` instead
      of the per-expert `exl3_gpu_w13/w2` dicts;
    - check that `on_cpu | resident` covers `range(n_routed)`, and raise `ValueError` with the missing ids otherwise;
    - register as today.
  - **The fully resident draft** (no CPU experts): at `process_weights_after_loading`, when `self.draft`, build
    `layer.exl3_draft_moe = DraftResidentMoe(layer, range(E), E, device)` over the layer's own GPU params. The
    `index_select` copies them, so also drop the original params' storage, so VRAM is not doubled:
    `layer.w13_trellis.data = torch.empty(0, ...)` etc. after the copy. The `exl3_w13` list views become unused.
    Before dropping anything, grep for every reader of `exl3_w13`, `exl3_w2` and the `w13_*`/`w2_*` params of a draft
    layer (the loader's post-load, `record_draft_routes`, the probe tests). Drop only if the draft path is the sole
    reader after this task; otherwise keep the params, and record the VRAM cost as a ruling.
  - **`apply`:** for `self.draft`, route to `_apply_draft(layer, x, topk_weights, topk_ids, limit, scale)`:

```python
def _apply_draft(self, layer, x, topk_weights, topk_ids, swiglu_limit):
    """The draft stage's routed experts, graph-safe: the CPU share posted first, the GPU share while it runs, then
    the wait that adds it. 16 tokens per pass (DraftResidentMoe.TOKENS); a longer x (the prompt's extend) loops."""
    moe = layer.exl3_draft_moe
    cpu = draft.DRAFT_CPU_EXPERTS.runtime() if self.cpu_draft else None
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for lo in range(0, x.shape[0], moe.TOKENS):  # host shapes only
        hi = min(lo + moe.TOKENS, x.shape[0])
        xs, ids, ws = x[lo:hi], topk_ids[lo:hi], topk_weights[lo:hi]
        if cpu is not None:
            cpu.post(layer.exl3_cpu_draft_key, xs, ids, ws)
        part = moe.run(xs, ids, ws, swiglu_limit)
        if cpu is not None:
            cpu.finish(layer.exl3_cpu_draft_key, part)
        out[lo:hi] = part
    return out.to(x.dtype)
```

    The scaling and fused-shared handling stay as the current branches apply them. Read `apply` `:600-636` and keep
    the same `scale` placement. Drop `record_draft_routes` from the draft path: it reads the host. Keep it under
    `not torch.cuda.is_current_stream_capturing()` if a test depends on it; grep for its users first.
  - **`draft.py`:**
    - `runtime()` resolves cores as today, then builds `DraftCpuAreas(stages, hidden)`, `DraftCpuHost(...)` with
      `spin_us=SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US`, `keep_warm_us=SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US` and
      `fatal_wait_s=watchdog_wait_s(SGLANG_DSV41_RAM_MISS_TIMEOUT_MS)`, then `set_layer` per stage (`trait.layer_spec`
      over the stage's slabs, capacity = E), then `start()`, then `DraftCpuDevice(areas, on_cpu_stack, device)`.
    - `DraftCpuExperts` keeps `stats` (from `host.stats()`) and `close` (`host.stop()`).
    - It raises if `runtime()` is first called while capturing ("prepare the DSpark draft before capture").
  - **`prepare_dspark_draft_graph(model)`:** for each module with `exl3_draft_moe`, call `.prepare()`. If any layer is
    a CPU draft, call `DRAFT_CPU_EXPERTS.prepare()`. Return the count.

- [ ] **Step 4: Run the CPU tests. Expected: PASS.** Run the full `test_exl3_moe_method.py`,
  `test_dspark_draft_cpu_experts.py`, `test_dspark_draft_resident.py`, `test_exl3_fused_moe.py` and
  `test_exl3_draft_moe.py`.

- [ ] **Step 5: Extend the GPU test** `test_dspark_hybrid_draft_gpu.py`:
  - **`test_hybrid_draft_matches_the_gpu_loop`:** keep it, now through `_apply_draft`, at M ∈ {1, 5, 16, 40}.
  - **Add `test_the_captured_hybrid_draft_matches_eager`:** `prepare_dspark_draft_graph`, eager warmup, then capture
    one stage's `apply` at M = 5. Replay 5 times with new routes, against eager each time, byte-exact. Assert
    `stats()["jobs"]` grew by the number of replays whose routes had a CPU expert.
  - **Add `test_the_draft_path_reads_no_host`:** under `torch.cuda.set_sync_debug_mode("error")`, an eager
    `_apply_draft` at M = 5 after prepare raises nothing.

- [ ] **Step 6: Run the GPU test. Expected: PASS.**

- [ ] **Step 7: Commit:** `test(dspark): the draft MoE posts its CPU share, runs its GPU share, then finishes
  (failing)`, then `feat(dspark): the draft MoE in the graph: the GPU share and the draft channel, no host sync`.

---

### Task 6: The worker captures an EXL3 draft

**Files:**
- Modify: `python/sglang/srt/speculative/dspark_components/dspark_graphed_verify.py` (`draft_runs_exl3` stays;
  new `draft_graph_allowed(draft_model) -> bool`)
- Modify: `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` (`init_cuda_graphs` `:534-568`)
- Modify: `python/sglang/srt/environ.py` (`SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH`)
- Test: `test/registered/unit/speculative/test_dspark_graphed_verify.py`

**Interfaces:**
- Consumes: `prepare_dspark_draft_graph(model)` (Task 5).
- Produces: `draft_graph_allowed(draft_model) -> bool`, which is true unless the draft is EXL3 and
  `SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH` is set.

- [ ] **Step 1: Tests.**
  - `test_an_exl3_draft_captures_unless_disabled`: with a fake model whose `quant_config.get_name()` is `"exl3"`, the
    helper returns True, and False under `envs.SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH.override(True)`. A non-EXL3
    draft is always True.
  - `test_init_cuda_graphs_prepares_the_exl3_draft_before_capture`: with a stub `_draft_worker.init_cuda_graphs` that
    records the call order, and `prepare_dspark_draft_graph` monkeypatched to append `"prepare"`, the order is
    `["prepare", "capture"]` with `capture_decode_cuda_graph=True`. With the env set, there is no prepare and the call
    gets `capture_decode_cuda_graph=False`.

  Build the worker stub the way the existing tests in this file build theirs. Read them first.

- [ ] **Step 2: Run. Expected: FAIL** (`draft_graph_allowed` undefined).

- [ ] **Step 3: Implement.**

```python
def draft_graph_allowed(draft_model) -> bool:
    """An EXL3 draft captures its decode graphs (its MoE is graph-safe: draft_moe.py and the draft channel) unless
    SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH keeps it eager for an A/B."""
    return not (draft_runs_exl3(draft_model) and envs.SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH.get())
```

  In `init_cuda_graphs`:
  - `capture_decode_cuda_graph = self._decode_graph_allowed and draft_graph_allowed(self.draft_model)`;
  - when that is True and `draft_runs_exl3(self.draft_model)`, call `prepare_dspark_draft_graph(self.draft_model)`
    before `self._draft_worker.init_cuda_graphs(...)`;
  - log `"DSpark: EXL3 draft graphs on (%d draft MoE layers prepared)"`.

  Add the env var with a comment: "Keeps an EXL3 DSpark draft eager (no decode graph); the draft MoE then runs the same
  graph-safe path eagerly. For A/B only."

- [ ] **Step 4: Run. Expected: PASS.** Also run `test_expert_stream_requirements_exl3.py`. The gate is unchanged:
  DSpark with a decode graph already requires D2-3's graphed-verify configuration, and the draft shares that backend.

- [ ] **Step 5: Commit:** `test(dspark): an EXL3 draft is prepared and captured unless disabled (failing)`, then
  `feat(dspark): capture the EXL3 draft's decode graphs; SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH keeps it eager`.

---

### Task 7: End to end on divix01, and the record

**Files:**
- Modify: `analysis/dsv41-drive/dspark/graphed_verify.py` (two arms; read it first, since D2-3 built it)
- Modify: `DSV41_REFERENCE.md` (§33.9)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` ("The lease channel": the draft as the second client)

**Interfaces:**
- Consumes: everything above. The D2-3 driver's arm/summary structure (`arm_environment`, `summarize`).

- [ ] **Step 1: Add the arms.** Both run D2-3's `graphed` configuration (W = 8) with
  `SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1` and the hybrid draft's resident set:
  - `draft-eager`: with `SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH=1`;
  - `draft-graph`: without it.

  The summary per arm gives:
  - tok/s;
  - accept length;
  - verify re-run rate;
  - the draft CPU thread's jobs, rows and forward ms (from `DraftCpuStats`, logged at close);
  - `text_matches` against `draft-eager`, per session.

- [ ] **Step 2: Run** on divix01 with D2-3's run procedure:
  - `rowimg-disk.lock`, then `cc-gpu.lock`;
  - the recipe's `taskset -c 0-5,36-41`;
  - sessions 0-7, 128 tokens.

  Write the output under
  `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/graph-verify/draft-graph/`.
  Expected:
  - both arms run to the end;
  - `draft-graph`'s log shows the draft captured (`"EXL3 draft graphs on"`);
  - text matches `draft-eager` in every session, or a mismatch is a tie per §33.2's margin probe. Record which, with
    the margin.

- [ ] **Step 3: Record §33.9 in `DSV41_REFERENCE.md`**, in the §33.8 shape (what was built; the run with its
  numbers, files and caveats; what it decides; what it does not do). It must state:
  - the draft step's time with and without the graph;
  - whether DSpark's tok/s moved;
  - the draft CPU forward ms per stage;
  - that the target's CPU experts are still off in verify;
  - that W = 8 still re-runs every verify (§33.8). So graphed DSpark's end-to-end number is still bounded by the
    re-verify until the W sweep.

- [ ] **Step 4: `LEASE_PROTOCOL.md`.** Under "The lease channel", add the draft as the second client:
  - `DraftChannel`;
  - its record;
  - its one-record-per-wait rule (so no lap);
  - the draft CPU thread as completer;
  - the head word as the keep-warm's wake-up.

- [ ] **Step 5: Commit:** `docs(dsv41): section 33.9 -- the DSpark draft in the decode graph; the lease channel's two
  clients`.

---

## Self-review notes

- **Coverage:**
  - Phase D2 item 2 (draft under a graph): Tasks 2, 5 and 6.
  - §33.3 item 7 (a capturable multi-token EXL3 MoE for the draft's resident experts): Task 2.
  - The owner's one-protocol decision: Task 1, with Tasks 3-4 instantiating it.
  - CPU experts in the graphed draft: Tasks 3-5.
  - The measurement: Task 7.
- **Type consistency:**
  - `DraftResidentMoe.TOKENS` = 16 = `kMaxRows`.
  - `kMaxK` = 8 ≥ the draft's top-k (3).
  - `DraftCpuAreas` shapes match the `draft_cpu_open` checks.
  - `draft_test_post`'s record matches `draft_post_kernel`'s writes.
  - The gate starts at `gate_word(0, kGateOpen)` = 1 (Task 3, Step 7).
- **Not in this plan:** multi-token CPU experts for the *target* (D2-4); the W sweep; the epilogue under a narrowed
  gather. Neither client's payload is shared: separate areas and threads, by the owner's choice.
