# DSV4.1: DSpark with the target's and the draft's CPU experts at once — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve DSV4.1 EXL3 with DSpark speculative decoding while both the target's CPU experts
(`SGLANG_DSV41_CPU_EXPERTS=1`) and the draft's CPU experts (`SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1`) are on, and
make that the production recipe once a server A/B against today's production clears the owner's bar. In steady state
a verify is never re-run eagerly. The one remaining eager re-verify is the one the copy engine forces before it arms.

**Architecture:**
- **Every route has a lane.** A 6-token verify at top-6 routes at most 36 distinct experts per layer, and the record
  gets one lane per route: `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES` stays unset, so the miss width equals the
  routes. The wire is widened to 64 lanes and builds at 40 (36 rounded up to the wire's 8). Its lane masks become 64-bit
  only in builds above 32 lanes. The one-token build keeps its 32-bit types, and a SASS and `.text` digest test pins it
  byte for byte.
- **CPU lanes take M tokens.** The post stages all M token rows and writes a per-lane token table, holding each token's
  routing weight per lane. The CPU expert thread runs one M-row forward per job, and the route tables seed each token's
  output from its own partial.
- **Victims for the first V lanes only.** A new DIRECT knob, `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` (V), gives
  VRAM victims and staging slots only to the first V lanes in plan order. Every later lane is a **forced CPU lane**
  (spill).
- **A forced RAM hit runs from its RAM slot.**
- **A forced NVMe miss gets no staging slot (slot −1).** The host reads it straight into a RAM victim of the layer's
  own tier, computes it there, and keeps it cached.
- **Two invariants are asserted:** live misses ≤ V = staging per node, and a victim always exists. The second is a
  per-node tier-capacity check at start-up.
- **Overflow and the eager re-verify (GPU-only):** the post overflows only when a forced lane cannot be the CPU's,
  which means the copy engine is unarmed or the CPU layer is not yet registered. Both occur only before arming.
- **The draft keeps its own lease channel and thread** on dedicated node-0 cores (12-15). The target's node-0 team
  shrinks to 6-11; node 1 keeps 18-27.

**Tech Stack:** CUDA (JIT `lease_kernels.cuh`, `lease_device.cuh`, `row_copy_kernels.cuh`, `exl3_route_tables.cuh`,
`direct_gather.cuh`), C++20 host (`host/cpu_experts.h`, `host/ram_tier.h`, `host/copy_engine.h`,
`host/split_calibration.h`, `host/ffi_exports.h`), Python (sglang EXL3 expert stream, DIRECT residency, server-args
gate), pytest (registered CPU/GPU units, manual `test/manual/dsv41` CUDA suites), bash (`run_arm.sh`, `launch_prod.sh`),
`cuobjdump`/`objcopy` for the one-token build digest.

**Spec:** The owner's request (2026-10-06, relayed by team-lead): "we absolutely need to support both cpu_experts and
dspark cpu_experts at the same time". It came with two amendments the same day:
- (A) widen the record so a 6-token verify never exceeds it, with no second record per layer and the one-token path
  byte-identical;
- (B) a forced NVMe miss must always have somewhere to land, never falling back to re-verify.

The evidence this plan argues from is `DSV41_REFERENCE.md` §33.3–§33.9 on `origin/master`; §33.9 reaches this branch
in Task 1. Base: `origin/dsv41-cpu-plan-m2` (the multi-token DSV4.1 CPU plan, `CHUNK_M` token chunks, `MAX_M = 4`),
which the verify's M-row forwards use.

---

## Owner decisions

Each needs the owner's call. The plan proceeds on the recommendation and records the choice where it lands.

1. **When production flips to DSpark.** Evidence today: graphed DSpark ran 2.85 tok/s (§33.9) against production's
   ~13 tok/s plain decode with CPU experts (§30.1, §33.5). §33.5's v2 projection, multi-token CPU experts in the verify,
   ranges from +25% to −13% against 13.17 tok/s. This plan builds the mode and a one-line switch (`PROD_DSPARK` in
   `arm_env.py`).
   **Recommendation:** flip only if the A/B (Task 15) shows `dspark-both` at or below production's median ms/token
   and passes the text bar, or on the owner's explicit go regardless. Task 16 is gated on this.
2. **Cores for the two CPU-expert clients.** Node 0 has no free physical core under today's recipe: 0-5 server, 6-15
   target, 16 copy, 17 RAM. Options:
   - **(a) Recommended.** The draft takes named cores 12-15 and the target's node-0 team shrinks to 6-11, leaving
     node 1's 18-27 as is. No code change: `ThreadingConfig` already leaves named draft cores out of every derived
     role. No spin interaction: each team spins only on its own cores. The draft weights are first-touch on node 0,
     local to its cores.
   - **(b) Time-share 6-15.** Both teams' keep-warm and idle spin (100 ms PAUSE after 2 ms of register work) would
     fight for the same logical cores. That needs spin coordination code and `ThreadingConfig` changes that refuse
     today's rules.
   - **(c) The draft on node 1's 32-34.** Only 3 cores, and the draft weights would be remote (node 0).
3. **Lane count, victim lanes, staging.**
   - **Record:** one lane per route, which is `MISS_LANES` unset: a 36-lane gather on a **40-lane wire**.
     - 36 is the maximum distinct experts of 6 tokens at top-6 (§33.5: max 36, p99 32).
     - The wire rounds lanes up to 8 so its i16 lane arrays are whole 16-byte stores (`lease_layout.h:38-39`). A
       36-lane build *is* a 40-lane build.
     - 40-lane, 2-node wire, computed from the layout formulas: 640-byte record (128 at 8 lanes), 10,368-byte request
       page, 86,016-byte lease block (20,480), 512-byte delta stride (256). All of it is pinned host memory, about
       80 KiB in total.
     - The type limit moves to 64 lanes, one u64 mask. Block size 6 (7 tokens, 42 routes) would then need only a
       rebuild, not a redesign.
   - **Victims:** **V = 8**, D2-3's miss width: 8 VRAM victims per layer, so the DIRECT floor is 16 slots
     (≤ ~23.7 at the hot cache below).
   - **Staging: 8 slots per node, unchanged from the D2-3 arms.** Only live (victim) misses stage, and they number at
     most V. That is an invariant, asserted on the device.
   - **Forced NVMe misses are host-placed (recommended): they land in a RAM victim the host picks per record. Extra
     pinned RAM: 0 bytes.**
     - Each such miss evicts one RAM row on its node and is cached in its place, which is today's insert-on-miss
       policy.
     - The guarantee is a start-up check on every layer and node: `slots ≥ staging + 36 lanes + the layer's VRAM-hot
       capacity`. That is 8 + 36 + ~24 = 68 against ~80 slots per node per layer at `PINNED_HOST_NUMA_MB=0:40960,1:40960`.
       The derivation is in Task 9.
     - The rejected options, measured against the 80 GiB tier (13,315,584 B per expert row, 40 MoE layers, 2 nodes):

     | Option | Pinned RAM | Verdict |
     |---|---|---|
     | Staging sized for the worst case (all 36 lanes NVMe misses on one node: 36 slots per node per layer) | 2 × 36 × 40 × 13,315,584 B = **35.7 GiB**, vs 7.9 GiB at 8 | Unaffordable: 45% of each node's 40 GiB tier |
     | Per-node shared scratch (staging is per layer, but one record is in flight, so a scratch can be shared across layers) | 36 rows per node = 457 MiB per node, **914 MiB** in all | Affordable, but needs a reader destination override, a scratch CPU layer, and node-0 headroom the 2026-10-02 note says is thin |
     | **Host-placed into a RAM victim** | **0** | Recommended |

4. **Hot cache under DSpark.** **Recommendation:** `SGLANG_MOE_HOT_GPU_MB=12040`, the hybrid draft's value (§33.4,
   §33.5), which leaves 4040 MiB of production's 16080 for the draft's resident experts, dense weights and KV. Task 15
   gates it: the server's KV pool must be no smaller than the production arm's in the same A/B. It also bounds Task 9's
   start-up check, which uses the layer's VRAM capacity.
5. **The eager re-verify stays GPU-only, and in steady state it never runs.** The ways a verify could overflow:

   | Case | Cause | What happens |
   |---|---|---|
   | 1 | Copy engine not yet armed (the first `COPY_ENGINE_ARM_DECODES` verifies after capture), or the CPU layer not registered (warm-up only) | The only one left: overflows and re-runs eagerly |
   | 2 | A forced NVMe miss without a staging slot | Cannot happen: it is host-placed (Task 9) |
   | 3 | More distinct misses than lanes | Cannot happen: a lane per route (Tasks 2 and 8) |

   **Recommendation:** keep the re-run GPU-only. Case 1 is a start-up transient, and CPU experts are not armed then
   anyway. Task 15 reports the re-verify count, which must equal the verifies before arming.

## Global Constraints

- Branch `dsv41-dspark-both-cpu`, worktree `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-both-cpu`, cut from
  `origin/dsv41-cpu-plan-m2`. Push only this branch (`git push origin dsv41-dspark-both-cpu`).
- No push to `master`. No amend, rebase, stash or force. A fix-up is a new commit.
- Do not edit `python/sglang/srt/model_executor/model_runner.py`: it is frozen (`.claude/rules/modify-component-must-read.md`).
- Never use the `fable` model for any subagent.
- Stage files by name (`git add <path>...`). Never `git add -A` or `.`.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq
  ```
- Running code (`.claude/rules/divix01-run-protocol.md`):
  - Commit and push, then on divix01 run in a private worktree at the pushed commit:
    ```bash
    ssh divix01 'git -C /data/models/slang/sglang fetch origin && \
      (git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-both-cpu origin/dsv41-dspark-both-cpu \
       || git -C /data/models/slang/nvfp4-work/wt-both-cpu checkout --detach origin/dsv41-dspark-both-cpu) && \
      git -C /data/models/slang/nvfp4-work/wt-both-cpu log -1 --oneline'
    ```
  - Every command sets `PYTHONPATH=$PWD/python` in that worktree. Print `sglang.__file__` once per session and check
    it is under `wt-both-cpu`.
  - CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS=8`.
  - Read pytest's status as `; echo "EXIT=${PIPESTATUS[0]}"` whenever output is piped.
  - Record the exact command next to every result.
- GPU steps:
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock`, on cores 32-63 unless the step names cores.
  - A step that needs both locks takes `rowimg-disk.lock` first.
  - Production holds `cc-gpu.lock` for its whole lifetime. The executor waits on the lock and **never starts or stops
    production**; GPU windows are the owner's.
- Environment variables go through `sglang.srt.environ.envs` (`env-var-conventions` skill). The one new variable is
  `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES = EnvInt(0)`.
- Speculative names follow the `speculative-naming` skill (`accept_length`, `spec_verify_ct`, `verify`).
- Non-speculative decode stays byte-for-byte what it is:
  - Every new path is gated on `Wire::kLanes > 32`, on `cpu_tokens_max > 1`, on `spill`, or on
    `victim_lanes < miss_rows`. A BS1 launch (8-lane wire, one token, no victim knob) has none of them.
  - Task 2's digest test pins the BS1 code its mask change touches.
  - The BS1 suites named in each task stay green.
- The verify is `gamma + 1 = 6` tokens at `speculative_dspark_block_size=5` (`speculative_hook.py:680-704`,
  `dspark_config.py:131-132`), not 5. Tests and comments say 6.

## Review Focus

1. **A captured verify before the copy engine arms** (the first `COPY_ENGINE_ARM_DECODES` forwards after capture, and
   every eager graph-path verify). This is the one overflow left. No forced lane can be a CPU lane: `host_lanes` is
   false. Expected: the post serves the victims' prefix, writes the count, flags the forward, and the eager re-verify
   gives the right text. No trap. Owner: Task 4 (`armed=False`), Task 11 (unarmed replay).
2. **Every non-victim lane an NVMe miss, all on one node.** This cannot overflow, and it is asserted.
   - The device types each forced miss `MISS_CPU` with slot −1, never a staging slot. It traps if a *live* miss finds
     no staging (live ≤ V = staging per node).
   - The host reads each forced miss into a RAM victim of that node and fail-stops if none exists, which Task 10's
     start-up capacity check rules out.
   - Owner: Task 4 (36 forced misses on a 40-lane post), Task 9 (36 host-placed misses on one node), Task 11 (end to
     end).
3. **A verify with fewer tokens than the rows hold** (a 3-token post into 4-token rows, a short eager verify). Expected:
   the host reads the count from the table header. It neither assumes `tokens_max` nor reads stale rows. Owner: Task 5,
   Task 6 (3 tokens in 4-token rows).
4. **A token that routes none of a job's lanes, and stale output rows from an earlier record.** Expected:
   - that token's row gets slot −1 everywhere and its partial is exactly 0;
   - the engine zeroes a non-accumulating per-token job's rows itself instead of trusting the kernel to;
   - and the real kernel skips only slot −1, never a zero weight (`forward_plan.hpp:699-702`).
   Owner: Task 6 (out rows pre-filled with 7.0, tokens that skip lanes).
5. **36 distinct misses in one layer.** This cannot overflow, and it is asserted.
   - With a lane per route, the miss count never exceeds the lanes.
   - Under spill the gather never calls the clamp, which asserts it is not spilling.
   - The post still traps on `count > Wire::kLanes`. The updater refuses spill with a narrowed `MISS_LANES`.
   - Owner: Task 8 (`test_spill_never_clamps`, `test_spill_needs_a_lane_per_route`), Task 11 (a 36-distinct-expert
     verify, no overflow).

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` | modify | 1..64 lanes; `LaneMask`, `kWideLanes`, mask word counts |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lane_mask.cuh` | create | `lowest_lane`, `lane_count`, `load_lane_mask` over u32/u64 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh` | modify | CW/CC on `LaneMask`, wide `ce_mask`/`cpu_lanes` formats |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout_probe.cpp` | modify | the 40-lane probe case |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h` | modify | `CopyJob` masks on `LaneMask` |
| `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` | modify | wide destinations kernel; commit templated on mask and width |
| `python/sglang/srt/layers/moe/ram_slot_map.py` | modify | reference typing: forced lanes (hit: RAM slot; miss: slot −1), `LaneOverflow` |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` | modify | device `type_lanes`: forced lanes; false only when no CPU lane is possible |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` | modify | post: spill fallback, M-token staging, token table |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/cpu_token_table.h` | create | the token table's layout constants, shared by device and host |
| `python/sglang/kernels/ops/moe/expert_lease_block.py` | modify | `MAX_LANES = 64`, `cpu_row_bytes` |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | modify | device `post` (`spill`, M tokens), both `enable_cpu_experts`, wide mask tensors, calibration lanes |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h` | modify | `CpuJob.per_token/lanes`, `CpuExpertConfig.tokens`, `run_job` expansion |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` | modify | `LaneMask` masks; record jobs carry lanes; host-placed forced misses |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h` | modify | calibrate up to a runtime lane count |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` | modify | `enable_cpu_experts(tokens)`, `calibrate_cpu_split(lanes)` |
| `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh` | modify | wide CPU mask; lift the one-token CPU refusal |
| `python/sglang/srt/layers/quantization/exl3/fused_moe.py` | modify | lift `m != 1` with `cpu` |
| `python/sglang/srt/environ.py` | modify | `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` |
| `python/sglang/srt/layers/moe/expert_residency_gpu.py` | modify | victim cap, idle destinations, spill reads the flag, clamp assert, floor 2V |
| `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` | modify | idle destination arguments |
| `python/sglang/srt/layers/moe/expert_hot_cache.py` | modify | allocator floor 2V |
| `python/sglang/srt/layers/moe/exl3_expert_format.py` | modify | plan the staging width V |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | modify | staging width, CPU rows by tokens, attach, spill words, capacity check |
| `python/sglang/srt/layers/moe/cpu_experts/service.py` | modify | rows sized by tokens, calibration lanes |
| `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` | modify | the gate |
| `scripts/dsv41/bs1_build_digest.py` | create | SASS and `.text` digests of the one-token build |
| `benchmarks/dsv41_baseline/arm_env.py`, `launch_prod.sh` | modify | DSpark mode, `PROD_DSPARK` |
| `scripts/expert_prediction/benchmarks/run_capture_sessions.py` | modify | record `sglext.spec_tokens_details` |
| `scripts/dsv41/dspark_text_band.py` | create | the §33.2/§33.9 near-tie text bar |
| `analysis/dsv41-drive/dspark/both_cpu_ab.py` | create | the three-arm server A/B |
| `analysis/dsv41-drive/LEASE_PROTOCOL.md`, `DSV41_REFERENCE.md` | modify | protocol and results |

---

### Task 1: Bring master's DSpark draft graph into the branch

The base lacks `origin/master`'s last 24 commits: §33.9's draft graph, the draft watchdog fixes, and §33.9 itself. A
trial merge conflicts in one hunk only: both sides added fields to `DraftCpuThread`'s member block.

**Files:**
- Modify (merge): `python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`

**Interfaces:**
- Consumes: `origin/master` at or after `3c115b9fb5`.
- Produces: the branch with `DraftResidentMoe`, the draft graph, and both sides' `DraftCpuThread` counters.

- [ ] **Step 1: Merge**

```bash
cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-both-cpu
git fetch origin
git merge --no-ff --no-commit origin/master
git status --short | grep '^UU'
```
Expected: exactly `UU python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`.

- [ ] **Step 2: Resolve the one hunk as the union of both sides**

Replace the conflict block (from `<<<<<<< HEAD` to `>>>>>>> origin/master`) with:

```cpp
  std::atomic<bool> stop_{false}, watchdog_stop_{false};  // the watchdog stops after the run thread has joined
  std::atomic<uint32_t> completed_{0}, head_at_stop_{0};
  std::atomic<int64_t> jobs_{0}, rows_{0}, forward_ns_{0}, holds_{0}, collided_jobs_{0}, shared_routes_{0},
      collided_forward_ns_{0};
```

Then `grep -n '<<<<<<<\|>>>>>>>' python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`. Expected:
no output.

- [ ] **Step 3: Commit the merge and push**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h
git commit --no-edit \
  --trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
  --trailer "Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-dspark-both-cpu
```

- [ ] **Step 4: Run the tests the merge touched, on divix01**

```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
  test/registered/unit/kernels/test_dspark_draft_cpu_thread.py \
  test/registered/unit/kernels/test_expert_stream_build_variants.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. Record the pass counts with the command.

---

### Task 2: Widen the wire to 64 lanes, with the one-token build unchanged

**Why 40 lanes.** A 6-token verify at top-6 routes at most 36 distinct experts per layer. A record with a lane per route
therefore never overflows, and Task 8 asserts it. `LeaseLayout` rounds lanes up to 8, because its lane arrays are i16
and whole 16-byte stores (`lease_layout.h:38-39`), so 36 builds as 40. The type limit moves to 64, one u64 mask.

**What changes** (research inventory, 2026-10-06). Every offset in `LeaseLayout` is already derived from `kLanes`; the
only fixed values are the `<8,1>` pins at `lease_layout.h:128-131`. The 32-lane limits are:
- **Per-build code**, compiled with `-DSGLANG_EXPERT_STREAM_LANES`:
  - the `static_assert` at `lease_layout.h:36`;
  - CW and CC (`row_copy_kernels.cuh:264, 278-295, 313-314, 326-356, 487-492`), with `uint32_t` lane masks and the
    `ce_mask[3]`/`cpu_lanes[2]` words;
  - the host's `CopyJob::mask/cpu_mask` (`copy_engine.h:44,49`) and `CpuMissBatch::sent` (`ram_tier.h:1602`) with
    their shifts (`ram_tier.h:1672,1677,1717,1791,1796`).
- **Shared by every build:**
  - `direct_gather.cuh`: a 32-thread destinations kernel, the commit's 32-entry arrays, and a `uint32_t` read of
    `cpu_lanes[0]`;
  - `exl3_route_tables.cuh:75-83`: a `uint32_t` CPU mask.
- **Python:** `expert_lease_block.MAX_LANES = 32`.

**Unaffected:**
- PieceMask, the delta block and the protect list scale with `kLanes`.
- §33.3's "8-bit masks" are piece bits (`kRowPieces = 8`), not lane bits.
- The dedup route planner already takes 64 routes.

**How the one-token build stays byte-identical:**
- The mask type is `LaneMask = std::conditional_t<(kLanes > 32), uint64_t, uint32_t>`, and wide-only words sit under
  `if constexpr`.
- The shared kernels are templated on the mask, and their narrow instantiation is the old code.
- The destinations kernel keeps its 32-thread body untouched, and a new 64-thread kernel serves wider shortlists.
- This task's first step records the BS1 build's SASS per kernel and its host `.text`, before any source change. Its
  test then requires both to be unchanged.

**Files:**
- Create: `scripts/dsv41/bs1_build_digest.py`, `test/manual/dsv41/test_bs1_build_digest.py`,
  `test/manual/dsv41/golden/bs1_build_digest.json`, `python/sglang/kernels/jit/csrc/moe/expert_stream/lane_mask.cuh`,
  `test/manual/dsv41/test_direct_gather_wide_gpu.py`
- Modify:
  - `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h`, `lease_layout_probe.cpp`, `row_copy_kernels.cuh`
  - `python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h`, `host/ram_tier.h`
  - `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh`
  - `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh`
  - `python/sglang/kernels/ops/moe/expert_lease_block.py`, `python/sglang/kernels/ops/moe/expert_stream_transport.py`
- Test:
  - `test/registered/unit/kernels/test_expert_stream_lease_layout.py`
  - `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`
  - `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`
  - `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`
  - `test/manual/dsv41/test_exl3_moe_split_parity_cuda.py`

**Interfaces:**
- Produces:
  - `LeaseLayout::kWideLanes`, `::LaneMask`, `::kCeMaskWords` (3, or 5 wide), `::kCpuLaneWords` (2, or 3 wide);
  - Python `WireLayout.wide_lanes`, `.ce_mask_words`, `.cpu_lane_words`;
  - `MAX_LANES = 64`.
  - Wide word formats keep the narrow words in place and append the high halves:
    - `ce_mask = {copy|cpu lo, cpu lo, parts, copy|cpu hi, cpu hi}`;
    - `cpu_lanes = {cpu lo, parts, cpu hi}`.
  - `direct_gather_destinations` takes shortlists of 1-64 entries; `direct_commit_gather` takes 1-64 lanes and a 2- or
    3-word `cpu_lanes`.

- [ ] **Step 1: Record the one-token build's digest before any change**

Create `scripts/dsv41/bs1_build_digest.py`:
```python
"""The one-token (BS1) build's machine code, digested (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 2).

Widening the wire past 32 lanes edits source the BS1 build compiles too. Its instantiations must compile to the same
machine code, which is what keeps BS1 outputs and timing unchanged. This runs the BS1 suites into a fresh JIT cache (so
every module they load is built here), then records:
  - for every kernel in the BS1 device modules and the shared DIRECT and route-table modules: the sha256 of its SASS
    (cuobjdump -sass), instruction words and encodings, addresses stripped, keyed by its demangled name with the wide
    template arguments this plan adds normalised away;
  - for every 8-lane host module: the sha256 of its .text section.
Run on divix01 under cc-gpu.lock from the worktree root:
    python scripts/dsv41/bs1_build_digest.py --write OUT.json      (record)
    python scripts/dsv41/bs1_build_digest.py --compare GOLDEN.json (exit 1 on any difference; keys only in the new
                                                                    build, the wide kernels, are ignored)
    ... --compare GOLDEN.json --permanent   the kernels no later task of the plan changes: every device kernel but the
                                            post (Tasks 4-5 change it on purpose, inert at one token), no host .text
                                            (Tasks 6 and 9 change the host on purpose; test_expert_stream_hotpath_golden
                                            pins its behaviour)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUITES = [
    "test/manual/dsv41/test_exl3_lease_kernels_cuda.py",
    "test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py",
    "test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py",
    "test/manual/dsv41/test_exl3_moe_split_parity_cuda.py",
    "test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py",
]
DEVICE = re.compile(r"^(expert_stream_exl3_l8(_n2)?|expert_residency_direct_.*|exl3_moe_route_tables.*)$")
HOST = re.compile(r"^expert_stream_host_exl3_(prod|instr)_l8(_n2)?$")
# The wide template arguments this plan adds to shared kernels; their narrow instantiation is the old kernel.
RENAMES = [
    (re.compile(r"direct_commit_gather_kernel<unsigned int, 32>"), "direct_commit_gather_kernel"),
    (re.compile(r"(exl3_moe_route_tables_kernel<[^<>]*?), unsigned int>"), r"\1>"),
]


def _demangle(name: str) -> str:
    out = subprocess.run(["c++filt", name], capture_output=True, text=True, check=True).stdout.strip()
    for pattern, repl in RENAMES:
        out = pattern.sub(repl, out)
    return out


def _sass(so: str) -> dict[str, str]:
    text = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True, check=True).stdout
    digests, name, lines = {}, None, []

    def close():
        if name is not None:
            digests[_demangle(name)] = hashlib.sha256("\n".join(lines).encode()).hexdigest()

    for line in text.splitlines():
        match = re.match(r"\s*Function : (\S+)", line)
        if match:
            close()
            name, lines = match.group(1), []
        elif name is not None and "/*" in line:
            lines.append(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line).strip())  # drop the address column
    close()
    return digests


def _text(so: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".bin") as out:
        subprocess.run(["objcopy", "-O", "binary", "--only-section=.text", so, out.name], check=True)
        return hashlib.sha256(open(out.name, "rb").read()).hexdigest()


def collect() -> dict:
    cache = os.path.join(REPO, ".bs1-digest-cache")
    shutil.rmtree(cache, ignore_errors=True)
    env = os.environ | {"SGLANG_JIT_CACHE_DIR": cache, "PYTHONPATH": os.path.join(REPO, "python")}
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", *SUITES], env=env, cwd=REPO).returncode
    if rc != 0:
        raise SystemExit(f"the BS1 suites failed (exit {rc}); a digest of a red build proves nothing")
    result = {"nvcc": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.splitlines()[-1],
              "device": {}, "host": {}}
    import torch

    result["arch"] = "sm_%d%d" % torch.cuda.get_device_capability()
    for root, _, files in os.walk(cache):
        for f in files:
            if not f.endswith(".so"):
                continue
            module, path = f[:-3], os.path.join(root, f)
            if DEVICE.match(module):
                for kernel, digest in _sass(path).items():
                    result["device"][f"{module}::{kernel}"] = digest
            elif HOST.match(module):
                result["host"][module] = _text(path)
    shutil.rmtree(cache, ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write")
    group.add_argument("--compare")
    parser.add_argument("--permanent", action="store_true")
    args = parser.parse_args()
    now = collect()
    if args.write:
        with open(args.write, "w") as f:
            json.dump(now, f, indent=1, sort_keys=True)
        return
    with open(args.compare) as f:
        golden = json.load(f)
    if (golden["arch"], golden["nvcc"]) != (now["arch"], now["nvcc"]):
        print(f"toolchain differs: golden {golden['arch']} {golden['nvcc']}, now {now['arch']} {now['nvcc']}")
        raise SystemExit(2)
    kinds = ("device",) if args.permanent else ("device", "host")
    diffs = [f"{kind} {key}" for kind in kinds for key, digest in golden[kind].items()
             if now[kind].get(key) != digest and not (args.permanent and "exl3_ram_miss_post_kernel" in key)]
    print("\n".join(diffs) or "BS1 build unchanged")
    raise SystemExit(1 if diffs else 0)


if __name__ == "__main__":
    main()
```
Create `test/manual/dsv41/test_bs1_build_digest.py`:
```python
"""The one-token build's kernels compile to the machine code recorded before the wire was widened (plan
2026-10-06-dsv41-dspark-both-cpu-experts Task 2): the SASS of every BS1 kernel the widening touched (CW, CC, the DIRECT
destinations and commit kernels, the route tables) and of the unchanged ones (C1, S). Same machine code, same outputs
and timing. The post kernel and the host .text are compared in Task 2 only (bs1_build_digest.py --permanent): later
tasks change them on purpose, inert at one token. Skips on another GPU arch or nvcc than the golden's. Run on divix01
under cc-gpu.lock (it runs the BS1 suites)."""

import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "bs1_build_digest.json")


def test_the_one_token_build_is_unchanged():
    run = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "dsv41", "bs1_build_digest.py"), "--compare", GOLDEN, "--permanent"],
        capture_output=True, text=True, cwd=REPO,
    )
    if run.returncode == 2:
        pytest.skip(run.stdout.strip())
    assert run.returncode == 0, run.stdout + run.stderr
```
Run it once at the unmodified commit, which is the merge from Task 1:
```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu   # checked out at the Task 1 merge plus these two new files, no source change
mkdir -p test/manual/dsv41/golden
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python \
  /data/models/slang/.venv/bin/python scripts/dsv41/bs1_build_digest.py --write test/manual/dsv41/golden/bs1_build_digest.json
```
Expected: exit 0. The golden lists the kernels of `expert_stream_exl3_l8`, `expert_stream_exl3_l8_n2` and
`expert_residency_direct_*`, plus `exl3_moe_route_tables*`, and the `.text` digests of the 8-lane host modules. Copy
the golden back to the laptop worktree with `scp` (a generated test fixture, not a working tree). Commit the script,
the test and the golden before Step 3 touches any source:
```bash
git add scripts/dsv41/bs1_build_digest.py test/manual/dsv41/test_bs1_build_digest.py test/manual/dsv41/golden/bs1_build_digest.json
git commit -m "test(expert-stream): record the one-token build's SASS and host .text before the wire widens

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

- [ ] **Step 2: Write the failing wide-wire tests**

`test_expert_stream_lease_layout.py`:
- `GRID` becomes `[(lanes, nodes) for lanes in (1, 6, 8, 13, 24, 32, 40) for nodes in (1, 2)]`.
- `test_a_lane_count_outside_1_to_32_is_refused` becomes:
  ```python
  @pytest.mark.parametrize("lanes", [0, 65])
  def test_a_lane_count_outside_1_to_64_is_refused(lanes):
      with pytest.raises(ValueError, match="1..64"):
          lease.wire_layout(lanes)
  ```
- Add:
  ```python
  def test_a_verify_wire_is_40_lanes_with_wide_masks():
      """36 routes (6 tokens at top-6) round up to 40 lanes. Its record and blocks follow from the formulas; its lane
      masks are u64, so CW's and CC's words gain the high halves. Every build up to 32 keeps the narrow words."""
      w = lease.wire_layout(36, 2)
      assert w.lanes == 40 and w.wide_lanes and (w.ce_mask_words, w.cpu_lane_words) == (5, 3)
      assert (w.record_bytes, w.page_bytes, w.lease_block_bytes) == (640, 10368, 86016)
      assert (w.copy_done, w.copy_gate, w.copy_armed, w.split) == (81920, 82048, 82176, 82304)
      assert (w.delta_max_entries, w.delta_stride) == (80, 512)
      assert not w.packed_counts
      for lanes in (8, 16, 24, 32):
          narrow = lease.wire_layout(lanes, 2)
          assert not narrow.wide_lanes and (narrow.ce_mask_words, narrow.cpu_lane_words) == (3, 2)
  ```

`test_exl3_ram_miss_attach_lanes.py`:
- add `(36, 40), (40, 40)` to the `rows, lanes` parametrization of `test_a_gather_within_the_planned_lanes_attaches`;
- in `test_a_gather_wider_than_32_is_refused_when_planned`, plan `65` and match `"1..64"`, and rename it `..._wider_than_64_...`.

`test_exl3_lease_kernels_cuda.py`: `@pytest.mark.parametrize("lanes", [8, 16, 32, 40])` on
`test_each_miss_takes_the_next_staging_slot_of_its_home_node`.

`test_exl3_cpu_lane_order_cuda.py`: add the following test. `Chain` takes `staging=36` so the eager gather that loads
36 experts into RAM has a staging slot per miss.
```python
def test_a_40_lane_build_carries_36_cpu_lanes_through_cw_and_cc(tmp_path):
    """Lanes 32-35 live in the masks' high words: CW marks all 36 RAM-hit lanes CPU, CC publishes {low, parts, high},
    and the host's cpu_mask hands all 36 RAM slots to one CPU job. Mutation: keep CW's mask a u32 -- lanes 32-35 are
    lost and the copy wait never covers them."""
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

    row, lanes, experts = 0, 40, list(range(36))
    c = Chain(tmp_path, copy_engine=True, start=False, lanes=lanes, top_k=36, dst_rows=36, capacity=80, staging=36)
    try:
        x_rows = torch.zeros((LAYERS, 2 * HIDDEN), dtype=torch.uint8).pin_memory()
        out_rows = torch.zeros((LAYERS, 2, HIDDEN), dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(c.host.test_kernel_address(zero=True), list(range(lanes + 1)), cores, x_rows,
                                  out_rows, threads=2)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        backend, plan, dev = c.backends[row], c.plans[row], c.dev
        assert (dev.ce_mask.numel(), dev.cpu_lanes.numel()) == (5, 3)
        c.plan(experts, row)
        c.gather(row)
        torch.cuda.synchronize()
        assert c.handled() and set(experts) <= c.resident(row)
        with paused(c.host):
            ram_slot = {e: s for s, (state, e, _) in enumerate(c.host.slot_info(row)) if e >= 0}
        c.host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
        c.host.arm_copy_engine()
        backend._stage_planned(plan)
        x = torch.randn(1, HIDDEN, device="cuda").half()
        dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True,
                 cpu_input=(x, torch.ones(36, device="cuda")))
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
        assert _until(lambda: len(c.host.test_kernel_calls()) == 1)
        torch.cuda.synchronize()
        assert c.kinds(36) == [int(LaneKind.HIT_CPU)] * 36
        assert dev.cpu_lanes.tolist() == [-1, PART_HITS, 0xF]
        (call,) = c.host.test_kernel_calls()
        assert sorted(call["slots"]) == sorted(ram_slot[e] for e in experts)
    finally:
        c.close()
```

`test_exl3_moe_split_parity_cuda.py`, after `test_cpu_lanes_with_a_zero_partial_are_the_gpu_run_without_them`:
```python
def test_cpu_lanes_past_32_rank_their_routes_out(slot_rows):
    """A 40-lane build's cpu_lanes is {low, parts, high}: CPU lanes 32-35 (their dst_slots entries the routes' slots,
    lanes 0-31 naming no slot) leave the fused MoE exactly as low lanes do, bit for bit against the masked run.
    Mutation: read only the low word -- the routes stay in."""
    device = slot_rows["w13_trellis"].device
    fused = _fused(slot_rows, device, True)
    hidden = slot_rows["w13_suh"].shape[-1]
    zero = torch.zeros((1, hidden), dtype=torch.float32).pin_memory()
    keep = torch.ones(1, device=device)
    gen = torch.Generator().manual_seed(933)
    for trial in range(TRIALS * 4):
        x, weights, remap, _ = _inputs(gen, fused.slots, hidden, device, 0)
        high = int(torch.randint(1, 16, (1,), generator=gen))  # which of lanes 32-35 are the CPU's
        dst = torch.full((36,), -1, dtype=torch.int32, device=device)
        dst[32:36] = remap[:4].to(torch.int32)  # lane 32 + i names route i's slot
        lanes = torch.tensor([0, PART_HITS, high], dtype=torch.int32, device=device)
        on_cpu = torch.tensor([i < 4 and bool(high >> i & 1) for i in range(TOP_K)], device=device)
        want = fused.run(x, torch.where(on_cpu, torch.zeros_like(weights), weights), remap, keep, ACT_LIMIT).clone()
        got = fused.run(x, weights, remap, keep, ACT_LIMIT, cpu=(lanes, dst, zero.data_ptr(), 0)).clone()
        assert torch.equal(_bits(got), _bits(want)), f"high={high:#x} remap={remap.tolist()}"
```

Create `test/manual/dsv41/test_direct_gather_wide_gpu.py`:
```python
"""DIRECT's kernels at a verify's width (33-64 lanes): the 64-thread destinations kernel equals the torch chain
(GpuResidencyUpdater.gather_destinations) on random shortlists, and the commit leaves out CPU lanes named by a 3-word
cpu_lanes, high word included (GPU, plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 2)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
WIDTH, SLOTS, EXPERTS = 36, 64, 96


def _chain(route_slots, victims, valid, miss_count, remap, base):
    """The torch chain of GpuResidencyUpdater.gather_destinations, without spill."""
    hazard = (route_slots.unsqueeze(1) == victims.unsqueeze(0)).any(dim=0)
    order = torch.argsort((hazard | ~valid).to(torch.uint8), stable=True)
    usable, usable_valid = victims.index_select(0, order), (valid & ~hazard).index_select(0, order)
    live = (torch.arange(victims.numel(), device="cuda") < miss_count) & usable_valid
    destinations = torch.where(live, usable, torch.zeros_like(usable))
    rank = (remap - base).clamp(min=0, max=victims.numel() - 1)
    return destinations, live, torch.where(remap >= base, destinations.index_select(0, rank), remap)


def test_the_wide_destinations_kernel_is_the_torch_chain():
    from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_gather_destinations

    gen = torch.Generator().manual_seed(36)
    for trial in range(200):
        ids = torch.randint(0, EXPERTS, (WIDTH,), generator=gen).cuda()
        expert_to_slot = torch.where(torch.rand(EXPERTS, generator=gen) < 0.4,
                                     torch.randint(0, SLOTS, (EXPERTS,), generator=gen), torch.full((EXPERTS,), -1)).cuda()
        victims = torch.randperm(SLOTS, generator=gen)[:WIDTH].cuda()
        valid = (torch.rand(WIDTH, generator=gen) < 0.8).cuda()
        miss_count = torch.randint(0, WIDTH + 1, (1,), generator=gen, dtype=torch.int32).cuda()
        remap = torch.where(torch.rand(WIDTH, generator=gen) < 0.5, torch.randint(0, SLOTS, (WIDTH,), generator=gen),
                            SLOTS + torch.randint(0, WIDTH, (WIDTH,), generator=gen)).cuda()
        want = _chain(expert_to_slot[ids], victims, valid, miss_count, remap, SLOTS)
        slots_out = torch.zeros(WIDTH, dtype=torch.int32, device="cuda")
        dest_out = torch.zeros(WIDTH, dtype=torch.int64, device="cuda")
        live_out = torch.zeros(WIDTH, dtype=torch.bool, device="cuda")
        remap_out = torch.zeros(WIDTH, dtype=torch.int64, device="cuda")
        direct_gather_destinations(ids, expert_to_slot, victims, valid, miss_count, remap, SLOTS, slots_out, dest_out,
                                   live_out, remap_out)
        assert torch.equal(dest_out, want[0]) and torch.equal(live_out, want[1]) and torch.equal(remap_out, want[2]), trial
        assert torch.equal(slots_out, want[0].to(torch.int32)), trial


def test_the_commit_leaves_out_cpu_lanes_past_32():
    from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_commit_gather

    cuda = dict(device="cuda")
    destinations = torch.arange(WIDTH, dtype=torch.int64, **cuda)
    live = torch.ones(WIDTH, dtype=torch.bool, **cuda)
    new_experts = torch.arange(10, 10 + WIDTH, dtype=torch.int64, **cuda)
    mapping = torch.full((EXPERTS + 1,), -1, dtype=torch.int64, **cuda)
    slot_to_expert = torch.full((SLOTS + 1,), -1, dtype=torch.int64, **cuda)
    slot_state = torch.zeros(SLOTS + 1, dtype=torch.uint8, **cuda)
    generations = torch.zeros(SLOTS + 1, dtype=torch.int64, **cuda)
    counters = [torch.zeros(1, dtype=torch.int64, **cuda) for _ in range(3)]
    cpu_lanes = torch.tensor([1, 1, 0b1010], dtype=torch.int32, **cuda)  # lanes 0, 33 and 35 are the CPU's
    direct_commit_gather(
        destinations, live, new_experts, mapping, slot_to_expert, slot_state, generations, *counters,
        torch.tensor([WIDTH], dtype=torch.int32, **cuda), torch.ones(1, **cuda), torch.tensor([WIDTH], dtype=torch.int32, **cuda),
        ready=3, free_state=0, cpu_lanes=cpu_lanes,
    )
    cpu = {0, 33, 35}
    assert [int(mapping[10 + j]) for j in range(WIDTH)] == [-1 if j in cpu else j for j in range(WIDTH)]
    assert int(counters[0].item()) == WIDTH - 3 and int(counters[2].item()) == 0  # insertions; nothing truncated
```

- [ ] **Step 3: Run and see them fail**

On divix01:
- CPU: `test/registered/unit/kernels/test_expert_stream_lease_layout.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`.
  Expected: `ValueError: a demand record carries 1..32 lanes, not 36`.
- GPU lock: `test/manual/dsv41/test_direct_gather_wide_gpu.py` (expected `the shortlist must hold 1-32 entries`) and
  the new lane-order and split-parity tests.

- [ ] **Step 4: The layout** (`lease_layout.h`, `lease_layout_probe.cpp`, `expert_lease_block.py`)

In `lease_layout.h`, add `#include <type_traits>`. Change the assert to
`static_assert(1 <= NumLanes && NumLanes <= 64, "a record carries 1..64 lanes");`. After `kNodes`, add:
```cpp
  // A mask with a bit per lane: one u32 up to 32 lanes, so the narrow builds' code is unchanged, else one u64.
  static constexpr bool kWideLanes = kLanes > 32;
  using LaneMask = std::conditional_t<kWideLanes, uint64_t, uint32_t>;
  // CW's words for CC (ce_mask: copy | cpu, cpu, parts) and CC's for the fused MoE and the DIRECT commit (cpu_lanes:
  // cpu, parts); a wide build appends the masks' high halves, so the narrow words keep their places.
  static constexpr int kCeMaskWords = kWideLanes ? 5 : 3;
  static constexpr int kCpuLaneWords = kWideLanes ? 3 : 2;
```
In `lease_layout_probe.cpp`, add `case 40: return for_nodes<40>(nodes);` to `probe` and
`case 40: return channel_for_nodes<40>(nodes);` to `channel_probe`.

`expert_lease_block.py`: `MAX_LANES = 64`. Add to `WireLayout`:
```python
    @property
    def wide_lanes(self) -> bool:
        """Lane masks are u64 (LeaseLayout::kWideLanes): more than 32 lanes."""
        return self.lanes > 32

    @property
    def ce_mask_words(self) -> int:
        return 5 if self.wide_lanes else 3

    @property
    def cpu_lane_words(self) -> int:
        return 3 if self.wide_lanes else 2
```

- [ ] **Step 5: Mask helpers** — create `python/sglang/kernels/jit/csrc/moe/expert_stream/lane_mask.cuh`:
```cpp
// Lane masks of the RAM-miss wire (lease_layout.h LaneMask): one u32 up to 32 lanes, one u64 above. The u32 overloads
// are the intrinsics the narrow code always used, so a narrow build compiles to what it did.
#pragma once

#include <cstdint>

namespace sglang::expert_stream {

__device__ __forceinline__ int lowest_lane(uint32_t mask) {
  return __ffs(mask) - 1;
}
__device__ __forceinline__ int lowest_lane(uint64_t mask) {
  return __ffsll(static_cast<long long>(mask)) - 1;
}
__device__ __forceinline__ int lane_count(uint32_t mask) {
  return __popc(mask);
}
__device__ __forceinline__ int lane_count(uint64_t mask) {
  return __popcll(mask);
}

// A lane mask from its words: `lo`, and for a u64 mask `hi` as the high half.
template <typename MaskT>
__device__ __forceinline__ MaskT load_lane_mask(const int32_t* lo, const int32_t* hi) {
  if constexpr (sizeof(MaskT) == 8) {
    return static_cast<MaskT>(static_cast<uint32_t>(*lo)) | static_cast<MaskT>(static_cast<uint32_t>(*hi)) << 32;
  } else {
    return static_cast<uint32_t>(*lo);
  }
}

}  // namespace sglang::expert_stream
```

- [ ] **Step 6: CW and CC** (`row_copy_kernels.cuh`)

Add `#include "lane_mask.cuh"`. Replace the two `static_assert`s with
```cpp
static_assert(device::expert_stream::Wire::kLanes <= 64, "a lane mask is one u64");
static_assert(2 * device::expert_stream::Wire::kNodes <= 32, "a part mask is one u32");
```
In `CopyCommitParams`, replace the comment of `cpu_lanes` with "CPU experts: Wire::kCpuLaneWords words, {the lanes the
CPU computed (low half), the output parts holding their partial sums (bit 2g + 0: group g's CPU hits', bit 2g + 1: its
CPU misses'), the lanes' high half in a wide build}, else 0; null when off."

In CW, at the top: `using LaneMask = Wire::LaneMask;`. Make these replacements:
- `__shared__ uint32_t copying;` → `__shared__ LaneMask copying;`
- `__shared__ uint32_t cpu;` → `__shared__ LaneMask cpu;`
- `uint32_t c = 0, u = 0, parts = 0;` → `LaneMask c = 0, u = 0;` plus `uint32_t parts = 0;`
- `u |= 1u << lane;` → `u |= LaneMask{1} << lane;`
- `c |= 1u << lane;` → `c |= LaneMask{1} << lane;`
- the copy loop becomes
  `for (LaneMask lanes = copying; lanes != 0; lanes &= lanes - 1) { const int lane = expert_stream::lowest_lane(lanes); ...`
Replace the tail from `p.ce_mask[0] = p.ce_mask[1] = p.ce_mask[2] = 0;` through `p.ce_mask[2] = ...;` with:
```cpp
  p.ce_mask[0] = p.ce_mask[1] = p.ce_mask[2] = 0;
  if constexpr (Wire::kWideLanes) p.ce_mask[3] = p.ce_mask[4] = 0;
  if (planned_count == 0) return;
  const LaneMask mask = copying | cpu;
  if (mask == 0) return;
  // CC reads these after the stream's wait, stream-ordered; nothing on the host reads them.
  p.ce_mask[0] = static_cast<int32_t>(static_cast<uint32_t>(mask));
  p.ce_mask[1] = static_cast<int32_t>(static_cast<uint32_t>(cpu));
  p.ce_mask[2] = static_cast<int32_t>(cpu_parts);
  if constexpr (Wire::kWideLanes) {
    p.ce_mask[3] = static_cast<int32_t>(static_cast<uint32_t>(mask >> 32));
    p.ce_mask[4] = static_cast<int32_t>(static_cast<uint32_t>(cpu >> 32));
  }
```
In CC, replace from `const uint32_t armed = ...` to the end of the kernel:
```cpp
  using LaneMask = Wire::LaneMask;
  const LaneMask armed = expert_stream::load_lane_mask<LaneMask>(p.ce_mask, p.ce_mask + 3);
  if (p.cpu_lanes != nullptr) {
    p.cpu_lanes[0] = p.cpu_lanes[1] = 0;
    if constexpr (Wire::kWideLanes) p.cpu_lanes[2] = 0;
  }
  if (armed == 0) return;
  const uint32_t seq = static_cast<uint32_t>(p.state[kPending]);
  const uint64_t generation = pending_generation(p.state);
  // Only a teardown opens a gate without CopyDone: the service is gone, and the copies may not have landed.
  channel::commit_or_trap<TargetChannel>(p.lease, seq, generation);
  if (p.cpu_lanes != nullptr) {
    p.cpu_lanes[0] = p.ce_mask[1];
    p.cpu_lanes[1] = p.ce_mask[2];
    if constexpr (Wire::kWideLanes) p.cpu_lanes[2] = p.ce_mask[4];
  }
}
```
(In a narrow build `load_lane_mask<uint32_t>` reads `ce_mask[0]` only, as before.)
In the launcher, `TensorMatcher({3})` for `ce_mask` becomes `TensorMatcher({Wire::kCeMaskWords})`, and the cpu_lanes
check becomes:
```cpp
    RuntimeCheck(
        cpu_lanes.size(0) == 0 || cpu_lanes.size(0) == Wire::kCpuLaneWords,
        "cpu_lanes: Wire::kCpuLaneWords words, or empty when CPU experts are off");
```
`expert_stream_transport.py` (`ExpertStreamDevice.__init__`):
- `self.ce_mask = torch.zeros(3, ...)` → `torch.zeros(self.wire.ce_mask_words, ...)`;
- `self.cpu_lanes = torch.zeros(2, ...)` → `torch.zeros(self.wire.cpu_lane_words, ...)`.
Their comments gain "; a wide wire appends the masks' high halves".

- [ ] **Step 7: Host masks** (`copy_engine.h`, `ram_tier.h`)

`copy_engine.h` `CopyJob`: `uint32_t mask = 0;` → `Wire::LaneMask mask = 0;` and `uint32_t cpu_mask = 0;` →
`Wire::LaneMask cpu_mask = 0;`.
`ram_tier.h`:
- `uint32_t sent = 0;  // bit i: miss i went to the CPU` → `Wire::LaneMask sent = 0;  // bit i: miss i went to the CPU`;
- `job.cpu_mask |= 1u << j;` → `job.cpu_mask |= Wire::LaneMask{1} << j;`;
- `job.mask |= 1u << j;` → `job.mask |= Wire::LaneMask{1} << j;`;
- `__builtin_popcount(job.cpu_mask)` → `std::popcount(job.cpu_mask)`;
- `misses->sent |= 1u << i;` → `misses->sent |= Wire::LaneMask{1} << i;`.
The tests `(job.cpu_mask >> lane.lane & 1u)` and `(misses->sent >> i & 1u)` stay correct for either type. Then
`grep -n "1u << \(j\|i\|lane\)" python/sglang/kernels/jit/csrc/moe/expert_stream/host/*.h`. Expected: no lane shift left
on a `uint32_t` (node masks such as `1u << job.group` stay).

- [ ] **Step 8: The shared kernels**

`direct_gather.cuh`, after the existing destinations kernel, add the wide kernel:
```cpp
// The same for a shortlist of 33-64 entries (a verify with a lane per route): two warps, thread j owns entry j, and the
// usable entries' order crosses the warps through per-warp counts. Same outputs as the torch chain; the narrow kernel
// above is left as it was for the one-token build.
template <typename IdT, typename RemapInT, typename RemapOutT>
__global__ __launch_bounds__(2 * kDirectGatherWarp, 1) void direct_gather_destinations_wide_kernel(
    const IdT* __restrict__ topk_ids,
    int top_k,
    const int64_t* __restrict__ expert_to_slot,
    const int64_t* __restrict__ victims,
    const bool* __restrict__ victim_valid,
    int width,
    const int32_t* __restrict__ miss_count,
    const RemapInT* __restrict__ remap_in,
    int64_t scratch_base,
    int32_t* __restrict__ destination_slots_out,
    int64_t* __restrict__ destinations_out,
    bool* __restrict__ live_out,
    RemapOutT* __restrict__ remap_out) {
  constexpr int kThreads = 2 * kDirectGatherWarp;
  __shared__ int64_t usable[kThreads];
  __shared__ bool usable_valid[kThreads];
  __shared__ int64_t destinations[kThreads];
  __shared__ int warp_good[2], warp_bad[2];
  const int t = static_cast<int>(threadIdx.x);
  const int warp = t / kDirectGatherWarp;
  const unsigned lane = static_cast<unsigned>(t % kDirectGatherWarp);
  const bool entry = t < width;
  const int64_t victim = entry ? victims[t] : 0;
  const bool valid = entry && victim_valid[t];
  bool hazard = false;
  if (entry) {
    for (int i = 0; i < top_k; ++i) {
      hazard |= expert_to_slot[static_cast<int64_t>(topk_ids[i])] == victim;
    }
  }
  const bool good = valid && !hazard;
  const unsigned good_mask = __ballot_sync(0xffffffffu, entry && good);
  const unsigned bad_mask = __ballot_sync(0xffffffffu, entry && !good);
  if (lane == 0) {
    warp_good[warp] = __popc(good_mask);
    warp_bad[warp] = __popc(bad_mask);
  }
  __syncthreads();
  const unsigned earlier = (1u << lane) - 1u;
  const int goods = warp_good[0] + warp_good[1];
  const int good_before = (warp == 1 ? warp_good[0] : 0) + __popc(good_mask & earlier);
  const int bad_before = (warp == 1 ? warp_bad[0] : 0) + __popc(bad_mask & earlier);
  if (entry) {
    const int position = good ? good_before : goods + bad_before;
    usable[position] = victim;
    usable_valid[position] = good;
  }
  __syncthreads();
  if (entry) {
    const bool live = t < miss_count[0] && usable_valid[t];
    const int64_t destination = live ? usable[t] : 0;
    destinations[t] = destination;
    destinations_out[t] = destination;
    destination_slots_out[t] = static_cast<int32_t>(destination);
    live_out[t] = live;
  }
  __syncthreads();
  for (int i = t; i < top_k; i += kThreads) {
    const int64_t remap = static_cast<int64_t>(remap_in[i]);
    int64_t rank = remap - scratch_base;
    rank = rank < 0 ? 0 : (rank > width - 1 ? width - 1 : rank);
    remap_out[i] = static_cast<RemapOutT>(remap >= scratch_base ? destinations[rank] : remap);
  }
}
```
In `direct_gather_destinations_gpu`:
- the width check becomes `RuntimeCheck(0 < W_.unwrap() && W_.unwrap() <= 2 * kDirectGatherWarp, "the shortlist must hold 1-64 entries");`;
- the launch becomes:
  ```cpp
  const bool wide = W_.unwrap() > kDirectGatherWarp;
  host::LaunchKernel(1, wide ? 2 * kDirectGatherWarp : kDirectGatherWarp, stream)(
      wide ? direct_gather_destinations_wide_kernel<IdT, RemapInT, RemapOutT>
           : direct_gather_destinations_kernel<IdT, RemapInT, RemapOutT>,
      ...the same arguments...);
  ```
Commit kernel: make it `template <typename MaskT, int kMaxWidth> __global__ __launch_bounds__(1, 1) void direct_commit_gather_kernel(...)`. Inside:
- replace the first line's read with
  `const MaskT cpu = cpu_lanes != nullptr ? expert_stream::load_lane_mask<MaskT>(cpu_lanes, cpu_lanes + 2) : MaskT{0};`;
- the arrays `[kDirectGatherWarp]` become `[kMaxWidth]`;
- `(cpu >> j & 1u)` becomes `(cpu >> j & MaskT{1})`;
- `__popc(cpu)` becomes `expert_stream::lane_count(cpu)`.
Include `../expert_stream/lane_mask.cuh`. In `direct_commit_gather_gpu`, the width check becomes `<= 2 * kDirectGatherWarp`
("the commit must cover 1-64 lanes"), and `cpu_lanes` accepts `{2}` or `{3}`:
```cpp
    const int64_t words = cpu_lanes.value().size(0);
    RuntimeCheck(words == 2 || words == 3, "cpu_lanes: two words, or three for a wide wire");
    expert_stream::verify_named(
        "cpu_lanes", TensorMatcher({words}).with_dtype<int32_t>().with_device<kDLCUDA>(device), cpu_lanes.value());
```
Launch `direct_commit_gather_kernel<uint32_t, kDirectGatherWarp>` when the width is at most 32 and `cpu_lanes` is absent
or 2 words, else `direct_commit_gather_kernel<uint64_t, 2 * kDirectGatherWarp>`.

`exl3_route_tables.cuh`: give `exl3_moe_route_tables_kernel` a fourth template parameter `typename MaskT`. Replace
```cpp
  const uint32_t cpu = cpu_lanes != nullptr ? static_cast<uint32_t>(cpu_lanes[0]) : 0u;
```
with `const MaskT cpu = cpu_lanes != nullptr ? expert_stream::load_lane_mask<MaskT>(cpu_lanes, cpu_lanes + 2) : MaskT{0};`
(`parts` still reads `cpu_lanes[1]`). In `ranked`, the loop becomes
`for (MaskT lanes = cpu; lanes != 0; lanes &= lanes - 1) { const int lane = expert_stream::lowest_lane(lanes); ...`.
In the launcher:
- the cpu_lanes size check accepts 0, 2 or 3 words;
- `cpu_on` is `cpu_lanes.size(0) >= 2`;
- launch `<RemapT, WeightT, XT, uint64_t>` when the size is 3, else `<RemapT, WeightT, XT, uint32_t>`.
Include `../expert_stream/lane_mask.cuh`.

- [ ] **Step 9: Python caps and docs**

- `exl3_ram_miss.py`: `plan_gather_width`'s docstring "outside 1..32" becomes "outside 1..64".
- `expert_stream.py:1489`: "at most 32 routes of one token or 64 of several" is unchanged; it is the planner's limit,
  which already holds 36 routes.
- The gate's 1-32 stays until Task 12.

- [ ] **Step 10: Run and pass, then prove the one-token build unchanged**

On divix01:
- CPU: the two registered files of Step 2, plus `test/registered/unit/kernels/test_lease_channel_layout.py`,
  `test_exl3_lease_block.py` and `test_expert_stream_hotpath_golden.py` (host behaviour of the 8-lane build, pinned).
- GPU lock:
  - `test/manual/dsv41/test_direct_gather_wide_gpu.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_moe_split_parity_cuda.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`;
  - then `test/manual/dsv41/test_bs1_build_digest.py`;
  - and the full comparison, host `.text` and post kernel included:
    `python scripts/dsv41/bs1_build_digest.py --compare test/manual/dsv41/golden/bs1_build_digest.json`.

Expected: `EXIT=0` for all, and both digest runs print `BS1 build unchanged`. Record the full comparison's output in
the commit message: it is this task's evidence that the BS1 outputs and timing are unchanged.

If a host `.text` digest differs, diff `objdump -d --no-show-raw-insn` of the two `.so` files. Record whether every
difference is an address operand (a moved `.rodata` string) or a real instruction change. A real change in a narrow
instantiation is a defect in Steps 6-8, to be fixed, not waived. If a kernel's SASS differs, do the same with
`cuobjdump -sass -fun <name>`.

- [ ] **Step 11: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout_probe.cpp \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lane_mask.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh \
  python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh \
  python/sglang/kernels/ops/moe/expert_lease_block.py python/sglang/kernels/ops/moe/expert_stream_transport.py \
  python/sglang/srt/layers/moe/exl3_ram_miss.py \
  test/registered/unit/kernels/test_expert_stream_lease_layout.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py \
  test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py \
  test/manual/dsv41/test_exl3_moe_split_parity_cuda.py test/manual/dsv41/test_direct_gather_wide_gpu.py
git commit -m "feat(expert-stream): the wire carries up to 64 lanes (u64 masks above 32); the one-token build is unchanged

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 3: The reference types forced lanes

`ram_slot_map.type_lanes` is the host reference the device typing transcribes, and the CUDA parity suite compares
against it. It learns **forced lanes**: the lanes from `forced_from` on found no VRAM victim and must be CPU lanes.

**Files:**
- Modify: `python/sglang/srt/layers/moe/ram_slot_map.py` (`type_lanes`, new `LaneOverflow`)
- Test: `test/registered/unit/kernels/test_ram_slot_map.py`

**Interfaces:**
- Consumes: Task 2's `wire_layout` up to 64 lanes.
- Produces: `type_lanes(..., forced_from: Optional[int] = None)`, returning `(kinds, slots)` as before.
  - A forced hit is `HIT_CPU` at its RAM slot.
  - A forced miss is `MISS_CPU` with slot −1, and never takes a staging slot: the host places it (Task 9).
  - The split counts only lanes `< forced_from`.
  - It raises `LaneOverflow(ValueError)` only when no forced lane can be the CPU's: no host lanes, or no CPU layer.
  - An unforced miss with no staging slot still raises a plain `ValueError` (the device traps).

- [ ] **Step 1: Write the failing tests** (append to `test_ram_slot_map.py`; add `LaneOverflow` to the import)

```python
from sglang.srt.layers.moe.ram_slot_map import LaneKind, LaneOverflow, MapReplica, type_lanes


def test_forced_lanes_are_cpu_lanes_outside_the_split():
    """Lanes 0-1 found VRAM victims, lanes 2-3 did not (spill). The split sees only lane 0, the one eligible unforced
    lane (split[1] = 1), so it is the CPU's; lane 1, an unforced miss, stays on the GPU in staging slot 9. The forced
    hit is a CPU lane at its RAM slot; the forced miss is a CPU miss with slot -1, though staging slot 10 is free: the
    host reads it into a RAM victim (Task 9), so staging only ever holds live misses."""
    ram = [-1] * 16
    ram[1], ram[2] = 4, 5
    staging = [9, 10] + [-1] * 6
    kinds, slots = _type([1, 0, 2, 7], ram, staging, cpu_on=True, forced_from=2)
    assert kinds == [LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.HIT_CPU, LaneKind.MISS_CPU]
    assert slots == [4, 9, 5, -1]


def test_36_forced_misses_on_one_node_need_no_staging():
    """Review Focus 2 at the record's full width: 36 lanes on a 40-lane, 2-node wire, 8 of them with VRAM victims,
    every lane an NVMe miss homed on node 0 (even experts). The 8 live misses take node 0's 8 staging slots; the 28
    forced ones take none. No LaneOverflow, whatever one node holds."""
    lanes, nodes = 40, 2
    experts = [2 * e for e in range(36)]
    staging = list(range(100, 108)) + [-1] * (lanes - 8) + [-1] * lanes  # node 0's list, then node 1's
    kinds, slots = type_lanes(
        experts, [-1] * 80, staging, [0] * (nodes * (lanes + 1)), lanes=lanes, captured=True, copy_armed=True,
        hit_copy="ce", cpu_on=True, cpu_misses=False, nodes=nodes, forced_from=8,
    )
    assert kinds == [LaneKind.MISS_GPU] * 8 + [LaneKind.MISS_CPU] * 28
    assert slots == list(range(100, 108)) + [-1] * 28


def test_forced_from_the_count_forces_nothing():
    ram = [-1] * 16
    ram[1] = 4
    staging = [9, 10] + [-1] * 6
    assert _type([1, 0], ram, staging, cpu_on=True, forced_from=2) == _type([1, 0], ram, staging, cpu_on=True)


@pytest.mark.parametrize(
    "changes",
    [{"copy_armed": False}, {"captured": False}, {"cpu_on": False}, {"cpu_ok": False}],
    ids=["unarmed", "eager", "cpu-off", "no-cpu-layer"],
)
def test_a_forced_lane_that_cannot_be_the_cpus_overflows(changes):
    ram = [-1] * 16
    ram[1], ram[2] = 4, 5
    with pytest.raises(LaneOverflow):
        _type([1, 2], ram, forced_from=1, **{"cpu_on": True, **changes})


def test_an_unforced_miss_without_staging_still_raises_plainly():
    """A live miss always has a staging slot (live <= victim lanes = staging per node): its absence is a broken
    invariant, a plain ValueError (the device traps), never an overflow."""
    with pytest.raises(ValueError) as refused:
        _type([0, 7], [-1] * 16, NO_STAGING, cpu_on=True, forced_from=1)
    assert not isinstance(refused.value, LaneOverflow)
```

- [ ] **Step 2: Run them and see them fail**

On divix01 in the worktree (pushed first):
`PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q test/registered/unit/kernels/test_ram_slot_map.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"`.
Expected: an `ImportError` (`LaneOverflow`), `EXIT=2`.

- [ ] **Step 3: Implement**

In `ram_slot_map.py`, add above `type_lanes`:

```python
class LaneOverflow(ValueError):
    """Forced lanes (DIRECT found them no VRAM victim, the post's spill) cannot be CPU lanes: no host lanes (the copy
    engine is not armed) or no CPU layer. The post then serves the unforced prefix and flags the forward
    (exl3_ram_miss_post_kernel)."""
```

Add `forced_from: Optional[int] = None` as the last keyword of `type_lanes`. Append to its docstring:

```
    Lanes from ``forced_from`` on (spill, SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES) found no VRAM victim: each is a
    CPU lane whatever the split. A forced hit runs from its RAM slot; a forced miss gets slot -1 and no staging slot,
    since the host reads it into a RAM victim (RamTier::reserve_victims_locked). The split counts only the lanes before
    ``forced_from``. Raises LaneOverflow when forced lanes cannot be CPU lanes (no host lanes, no CPU layer).
```

Replace the body from `home = wire_layout(lanes, nodes).home` through the end of the `for j in reversed(...)` loop:

```python
    forced_from = len(experts) if forced_from is None else forced_from
    home = wire_layout(lanes, nodes).home
    slots, hit, taken = [], [], [0] * nodes
    for j, e in enumerate(experts):
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
            continue
        if j >= forced_from:
            slots.append(-1)  # host-placed: read into a RAM victim, never a staging slot
            hit.append(False)
            continue
        node = home(e)
        m = taken[node]
        if m >= lanes or staging[node * lanes + m] < 0:
            raise ValueError(f"a miss lane has no staging slot on node {node}")
        slots.append(staging[node * lanes + m])
        hit.append(False)
        taken[node] += 1
    host_lanes = captured and copy_armed
    can_cpu = host_lanes and cpu_on and cpu_ok
    if forced_from < len(experts) and not can_cpu:
        raise LaneOverflow(f"lanes {forced_from}.. found no victim and cannot be CPU lanes")
    eligible = [j < forced_from and can_cpu and (h or cpu_misses) for j, h in enumerate(hit)]
    take = [0] * nodes
    for node in range(nodes):
        n = sum(1 for e, ok in zip(experts, eligible) if ok and home(e) == node)
        take[node] = split[node * (lanes + 1) + n] if n else 0
    cpu = [j >= forced_from for j in range(len(experts))]
    for j in reversed(range(len(experts))):
        node = home(experts[j])
        if take[node] > 0 and eligible[j]:
            cpu[j] = True
            take[node] -= 1
```
Everything after (`copy_ok = ...`, the kinds loop) is unchanged.

- [ ] **Step 4: Run and pass**

Same command. Expected: `EXIT=0`, every earlier test in the file still passing.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/ram_slot_map.py test/registered/unit/kernels/test_ram_slot_map.py
git commit -m "feat(exl3-ram-miss): the lane-typing reference takes forced lanes (spill): CPU lanes, misses host-placed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 4: The post types forced lanes on the device and falls back to the victims' prefix

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (`LanePlan`, `type_lanes`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (`PostParams`, post kernel, launcher)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, `.post`)
- Modify: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (the two raw `expert_stream_post` calls)
- Test: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`

**Interfaces:**
- Consumes: Task 3's `type_lanes(..., forced_from=)` and `LaneOverflow`.
- Produces:
  - `ExpertStreamDevice.post(..., spill=None)`, where `spill = (overflow_flag int32[1], gather_overflow int64[1])`
    on the device;
  - the post's FFI tail `..., cpu_weights, spill, overflow_flag, gather_overflow, use_pdl`;
  - with `spill`, a lane whose `dst_slots` entry is −1 is forced. A forced miss is `MISS_CPU` with lane slot −1.
  - When forced lanes cannot be CPU lanes (unarmed, or no CPU layer: case 1, the only overflow), the post writes
    `count[0] = live`, sets `*overflow_flag = 1`, and adds 1 to `*gather_overflow`.
  - A live miss without a staging slot traps: live ≤ V = staging per node, so it is an assert.

- [ ] **Step 1: Write the failing tests** (append to `test_exl3_slot_map_kernels_cuda.py`)

```python
from sglang.srt.layers.moe.ram_slot_map import LaneOverflow  # noqa: E402

ALL_CPU = list(range(W.lanes + 1))  # split[n] = n: every eligible unforced lane is the CPU's


def _post_spill(c, experts, forced_from, flag, overflows, row=0):
    """A captured post whose lanes from forced_from on have no VRAM victim (dst -1), as DIRECT's spill leaves them."""
    c.plan(experts, row)
    backend, plan = c.backends[row], c.plans[row]
    plan.slots[forced_from : len(experts)] = -1
    backend._stage_planned(plan)
    cpu_input = (torch.zeros(1, 64, device="cuda"), torch.ones(TOP_K, device="cuda"))
    c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=cpu_input,
               spill=(flag, overflows))
    torch.cuda.synchronize()
    count = int(plan.count[0])
    return count, c.kinds(count), c.dev.lane_slot[:count].tolist()


@pytest.mark.parametrize("armed", [True, False], ids=["armed", "unarmed"])
def test_post_spills_forced_lanes_like_the_reference(tmp_path, armed):
    """200 random maps, plans and spill points. Armed, every forced lane becomes a CPU lane whatever the split (a
    forced miss with slot -1, never a staging slot) and the count stands. Unarmed (Review Focus 1, the one overflow
    left), any forced lane overflows: the post serves the live prefix, writes its count, sets the flag and counts the
    overflow once. Mutations: type a forced lane by the split; give a forced miss a staging slot -- red."""
    c = Chain(tmp_path, start=False, copy_engine=True, hit_copy="ce", cpu_misses=False)
    try:
        hidden = 64
        c.dev.cpu_x_rows = torch.zeros((2, 2 * hidden), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=armed)
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        rng = random.Random(11)
        staging = rng.sample(range(8, 14), 6)
        _write_delta(c, 0, 1, staging)
        spilled = overflowed = 0
        for _ in range(200):
            ram = [-1] * EXPERTS
            for e, slot in zip(rng.sample(range(EXPERTS), rng.randint(0, W.lanes)), rng.sample(range(W.lanes), W.lanes)):
                ram[e] = slot
            c.dev.map_bulk_apply(torch.tensor([[0, e, s] for e, s in enumerate(ram)], dtype=torch.int32))
            experts = rng.sample(range(EXPERTS), rng.randint(1, TOP_K))
            forced_from = rng.randint(0, len(experts))
            flag.zero_()
            before = int(overflows.item())
            got = _post_spill(c, experts, forced_from, flag, overflows)
            ref = dict(captured=True, copy_armed=armed, hit_copy="ce", cpu_on=True, cpu_misses=False, lanes=W.lanes)
            try:
                kinds, slots = type_lanes(experts, ram, staging, SPLIT, forced_from=forced_from, **ref)
                want = (len(experts), [int(k) for k in kinds], slots, 0)
            except LaneOverflow:
                kinds, slots = type_lanes(experts[:forced_from], ram, staging, SPLIT, **ref)
                want = (forced_from, [int(k) for k in kinds], slots, 1)
            assert (*got, int(flag.item())) == want, (experts, ram, forced_from)
            assert int(overflows.item()) - before == want[3]
            spilled += want[3] == 0 and forced_from < len(experts)
            overflowed += want[3]
            chain = _chain_of_last_record(c)
            if chain:
                _write_delta(c, 0, chain, staging)
        assert spilled > 0 if armed else overflowed > 0
    finally:
        c.close()


def test_a_40_lane_post_makes_36_forced_misses_on_one_node_cpu_lanes(tmp_path):
    """Review Focus 2 at the record's full width: 36 distinct NVMe misses on one node of a 40-lane wire, 8 with VRAM
    victims. The 8 live misses take the node's 8 staging slots; the 28 forced ones are CPU misses with slot -1 (the
    host places them, Task 9). Count 36 stands, no flag, no trap: however many misses one node has, nothing overflows.
    Mutation: let a forced miss draw from staging -- the 9th traps."""
    w = lease.wire_layout(40)
    c = Chain(tmp_path, start=False, copy_engine=True, lanes=40, top_k=36, dst_rows=36, experts=48, capacity=24)
    try:
        c.dev.cpu_x_rows = torch.zeros((2, 128), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=True, split=[0] * (w.lanes + 1), w=w)
        staging = list(range(10, 18))
        _write_delta(c, 0, 1, staging, w=w)
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        experts = list(range(36))
        c.plan(experts)
        backend, plan = c.backends[0], c.plans[0]
        plan.slots[:8] = torch.arange(8, dtype=torch.int32, device=plan.slots.device)
        plan.slots[8:36] = -1
        backend._stage_planned(plan)
        cpu_input = (torch.zeros(1, 64, device="cuda"), torch.ones(36, device="cuda"))
        c.dev.post(0, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=cpu_input,
                   spill=(flag, overflows))
        torch.cuda.synchronize()
        assert (int(plan.count[0]), int(flag.item()), int(overflows.item())) == (36, 0, 0)
        assert c.kinds(36) == [int(LaneKind.MISS_GPU)] * 8 + [int(LaneKind.MISS_CPU)] * 28
        assert c.dev.lane_slot[:36].tolist() == staging + [-1] * 28
    finally:
        c.close()
```
`_write_delta` and `_set_host_words` take the wire as a keyword: add `w=W` to both signatures and use `w` for every
`W.` inside them, so the existing 8-lane callers are unchanged.

- [ ] **Step 2: Run and see them fail**

On divix01, GPU lock, cores 32-63:
```bash
PYTHONPATH=$PWD/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py \
  -k "spill or forced" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `TypeError: post() got an unexpected keyword argument 'spill'`.

- [ ] **Step 3: Implement the device typing** (`lease_device.cuh`)

Replace `struct LanePlan`:

```cpp
// The plan as the post reads it, in device memory: `count` planned experts and their VRAM destination slots. Lanes
// from forced_from on found no VRAM victim (spill: DIRECT's narrowed verify with CPU experts) and must be CPU lanes;
// forced_from == count forces none.
struct LanePlan {
  const int64_t* planned;
  const int32_t* dst;
  int64_t count;
  int64_t forced_from;
};
```

Replace the comment above `type_lanes` and its signature, and change its first loop and its typing loop as below.
The `expert`/`dst`/`ram`/`staging` loads and the `take` computation stay as they are.

```cpp
// Types each lane of the plan: its kind and source slot. Transcribes ram_slot_map.type_lanes, the host reference.
//
// A hit takes its RAM slot, and an unforced miss the next slot of its home node's staging list. Node n's CPU takes the
// last split[n][k] of its k eligible unforced lanes in plan order. A forced lane (j >= plan.forced_from) is a CPU lane
// whatever the split: a hit at its RAM slot, a miss with slot -1 and no staging slot, which the host reads into a RAM
// victim (RamTier::reserve_victims_locked). Traps where the reference raises ValueError: a plan wider than
// Wire::kLanes, an expert out of range or repeated, a hit slot past the row's capacity, an unforced miss with no
// staging slot on its node (live misses <= victim lanes = staging per node: an assert), a split entry above its n.
// Returns false where the reference raises LaneOverflow: forced lanes with no host lanes or no CPU layer; `out` is then
// partial. Reads no host memory; the caller loads the split table into the policy.
SGL_DEVICE bool type_lanes(const LanePlan& plan, const RowMap& map, const LanePolicy& policy, TypedLanes& out) {
```

The first per-lane loop becomes:

```cpp
  bool hit[Wire::kLanes];
  bool eligible[Wire::kLanes];
  int m[Wire::kNodes] = {};
  int n[Wire::kNodes] = {};
  const bool can_cpu = policy.host_lanes && policy.cpu_on && policy.cpu_ok;
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j) {
    if (j >= plan.count) break;
    for (int i = 0; i < j; ++i)
      if (expert[i] == expert[j]) __trap();
    const int node = Wire::home(expert[j]);
    const bool forced = j >= plan.forced_from;
    out.node[j] = node;
    hit[j] = ram[j] >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(ram[j]) >= map.row_capacity) __trap();
      out.slot[j] = ram[j];
    } else if (forced) {
      out.slot[j] = -1;  // host-placed: the host reads it into a RAM victim of its node
    } else {
      if (m[node] >= Wire::kLanes || staging[node * Wire::kLanes + m[node]] < 0) __trap();
      out.slot[j] = staging[node * Wire::kLanes + m[node]++];
    }
    if (forced && !can_cpu) return false;
    eligible[j] = !forced && can_cpu && (hit[j] || policy.cpu_misses);
    n[node] += eligible[j] ? 1 : 0;
  }
```

The typing loop becomes:

```cpp
  for (int64_t j = plan.count - 1; j >= 0; --j) {
    const int node = out.node[j];
    const bool forced = j >= plan.forced_from;
    const bool cpu = forced || (take[node] > 0 && eligible[j]);
    if (cpu && !forced) --take[node];
    uint8_t kind;
    if (cpu) {
      kind = hit[j] ? Wire::kKindHitCpu : Wire::kKindMissCpu;
    } else if (hit[j]) {
      kind = copy_ok && dst[j] >= 0 && dst[j] < policy.dst_rows ? Wire::kKindHitCopy : Wire::kKindHitSm;
    } else {
      kind = Wire::kKindMissGpu;
    }
    out.kind[j] = kind;
  }
  return true;
}
```

- [ ] **Step 4: Implement the post** (`lease_kernels.cuh`)

In `PostParams`, change `const int32_t* count;` to `int32_t* count;  // the plan's miss count; a spill overflow lowers it`
and append after `cpu_weights_count`:

```cpp
  // Spill (GpuResidencyUpdater.victim_lanes below its miss lanes, CPU experts on): a lane whose dst_slots entry is -1
  // found no VRAM victim and must be a CPU lane. When forced lanes cannot be (copy engine unarmed, no CPU layer), the
  // post serves the live prefix, writes it to count and flags the forward in DIRECT's words: overflow_flag (int32,
  // sticky) and gather_overflow (this layer's int64 counter). Both unused when spill is 0.
  int64_t spill;
  int32_t* overflow_flag;
  int64_t* gather_overflow;
```

In `exl3_ram_miss_post_kernel`, change `const int64_t count = ...` to `int64_t count = ...` (thread 0 may lower it; the
other threads read only `any_cpu`). Replace

```cpp
      type_lanes(LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = count}, map, policy, typed);
```
with
```cpp
      // Spill: DIRECT gives the lanes past its victims destination -1 (GpuResidencyUpdater.gather_destinations), and
      // the live lanes are a prefix.
      int64_t live = count;
      if (p.spill != 0)
        for (int64_t j = 0; j < count; ++j)
          if (p.dst_slots[j] < 0) {
            live = j;
            break;
          }
      if (!type_lanes(
              LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = count, .forced_from = live},
              map,
              policy,
              typed)) {
        // Forced lanes cannot be the CPU's (the copy engine is not armed, or no CPU layer yet): serve the live prefix
        // and flag the forward, as clamp_gather_misses does without CPU experts. S, CW, CC and the DIRECT commit read
        // the count written here.
        count = live;
        p.count[0] = static_cast<int32_t>(live);
        *p.overflow_flag = 1;
        *p.gather_overflow += 1;
        type_lanes(
            LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = live, .forced_from = live}, map, policy, typed);
      }
```

In the launcher `post(...)`, add parameters after `tvm::ffi::TensorView cpu_weights,`:

```cpp
      int64_t spill,
      tvm::ffi::TensorView overflow_flag,
      tvm::ffi::TensorView gather_overflow,
```
Before `const auto stream = ...`, add:
```cpp
    if (spill != 0) {
      expert_stream::verify_named(
          "overflow_flag", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), overflow_flag);
      expert_stream::verify_named(
          "gather_overflow", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), gather_overflow);
    }
```
In the `PostParams{...}` initializer, change `.count = static_cast<const int32_t*>(count.data_ptr()),` to
`.count = static_cast<int32_t*>(count.data_ptr()),` and append:
```cpp
        .spill = spill,
        .overflow_flag = spill != 0 ? static_cast<int32_t*>(overflow_flag.data_ptr()) : nullptr,
        .gather_overflow = spill != 0 ? static_cast<int64_t*>(gather_overflow.data_ptr()) : nullptr,
```

- [ ] **Step 5: Python** (`expert_stream_transport.py`)

In `ExpertStreamDevice.__init__`, after `self._no_cpu = ...`:
```python
        # The spill words of a post without spill: never read (the post's spill is 0).
        self._no_spill = (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )
```
`post` gains a last keyword `spill=None`. Append to its docstring:
```
        ``spill`` is DIRECT's ``(overflow_flag int32 [1], gather_overflow int64 [1])`` for this row, given when lanes
        may have no VRAM victim (SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES): a lane whose ``dst_slots`` entry is -1
        becomes a CPU lane, or, when one cannot, the post serves the live prefix, lowers ``count`` and flags both words.
```
Before `bank = self.map_bank`:
```python
        spill_on, (overflow_flag, gather_overflow) = 0, self._no_spill
        if spill is not None:
            overflow_flag, gather_overflow = spill
            for name, t, dtype in (
                ("overflow_flag", overflow_flag, torch.int32),
                ("gather_overflow", gather_overflow, torch.int64),
            ):
                if t.dtype != dtype or t.numel() != 1 or t.device != self.state.device:
                    raise ValueError(f"{name} must be one {dtype} word on {self.state.device}")
            spill_on = 1
```
In the `expert_stream_post(...)` call, insert after `cpu_weights,`:
```python
            spill_on,
            overflow_flag,
            gather_overflow,
```

- [ ] **Step 6: Update the raw calls in `test_exl3_lease_kernels_cuda.py`**

Both direct `expert_stream_post(` calls end in the CPU triple, then `use_pdl`:
- In the first, replace the tail `no_i32, 0, no_i32, 0,` with
  `no_i32, 0, no_i32, 0, torch.zeros(1, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"), 0,`.
- In the second, replace `*cpu_args, 0,` with
  `*cpu_args, 0, torch.zeros(1, dtype=torch.int32, **cuda), torch.zeros(1, dtype=torch.int64, **cuda), 0,`.

Then confirm nothing else calls the module directly:
`grep -rn "expert_stream_post(" python test analysis scripts benchmarks | grep -v "\.cuh:\|\.h:"`. Expected: only
`expert_stream_transport.py` and the two lines above.

- [ ] **Step 7: Run and pass, plus the BS1 lease suites**

```bash
PYTHONPATH=$PWD/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
  test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_lease_ordering_cuda.py \
  2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. The first run rebuilds the JIT modules (50-100 s each, `divix01-run-protocol.md`); a
`TimeoutExpired` on a cold cache is the compiler, not a hang.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/ops/moe/expert_stream_transport.py \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py
git commit -m "feat(exl3-ram-miss): the post makes victimless lanes CPU lanes (spill), else serves the prefix and flags

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 5: The post stages every token and writes the token table

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/cpu_token_table.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh`
- Modify: `python/sglang/kernels/ops/moe/expert_lease_block.py` (`cpu_row_bytes`, constants)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, `.enable_cpu_experts`, `.post`, `.cpu_out_part_stride`)
- Modify: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (raw calls again)
- Test: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`

**Interfaces:**
- Consumes: Task 4's FFI tail.
- Produces:
  - FFI tail `..., cpu_weights, cpu_tokens_max, cpu_x_token_bytes, spill, overflow_flag, gather_overflow, use_pdl`.
  - `expert_lease_block.cpu_row_bytes(hidden, tokens, lanes) -> int`, `CPU_TOKEN_TABLE_HEADER = 16`,
    `CPU_TOKENS_MAX = 32`.
  - `ExpertStreamDevice.cpu_tokens_max`, `.cpu_x_token_bytes`. `enable_cpu_experts` takes `out_rows` of shape
    `[layers, 2 * nodes, tokens, hidden]` for a multi-token row.
  - Row layout: `tokens_max` inputs of `ceil16(2 * hidden)` bytes each. With `tokens_max > 1` the table follows them:
    - `+0`: u32 tokens;
    - `+16 + 4 j`: u32 lane j's token mask;
    - `+16 + 4 lanes + 4 (t lanes + j)`: fp32 bits of token t's weight for lane j.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_slot_map_kernels_cuda.py`)

```python
from sglang.kernels.ops.moe.expert_lease_block import CPU_TOKEN_TABLE_HEADER, cpu_row_bytes  # noqa: E402


def test_a_verify_post_stages_every_token_and_the_token_table(idle):
    """Three tokens in rows that hold four (Review Focus 3); three RAM-hit lanes, all the CPU's. Each token's input is
    staged at its own offset as fp16, the header holds 3, each lane's mask has a bit per token that routes its expert,
    and each such token's own weight sits in the table. The record's lane weight is still the sum (the one-token
    path's). Mutations: stage token 0 only; t = r / route_count instead of r / top_k; mask from the summed weight."""
    c = idle
    hidden, tokens_max, lanes = 64, 4, W.lanes
    xb = 2 * hidden
    x_rows = torch.zeros((2, cpu_row_bytes(hidden, tokens_max, lanes)), dtype=torch.uint8).pin_memory()
    out_rows = torch.zeros((2, 2, tokens_max, hidden), dtype=torch.float32).pin_memory()
    c.dev.enable_cpu_experts(x_rows, out_rows)
    assert (c.dev.cpu_tokens_max, c.dev.cpu_x_token_bytes) == (tokens_max, xb)
    c.dev.set_row_cpu(0)
    _set_host_words(c, armed=True, split=ALL_CPU)
    _write_delta(c, 0, 1, [9, 10, 11, 12, 13, 14])
    experts = [3, 5, 7]
    c.dev.map_bulk_apply(torch.tensor([[0, e, s] for s, e in enumerate(experts)], dtype=torch.int32))
    c.plan(experts)
    backend, plan = c.backends[0], c.plans[0]
    backend._stage_planned(plan)
    routes = [[3, 5, 8, 9, 10, 11], [7, 3, 12, 13, 14, 15], [5, 7, 8, 12, 9, 13]]
    weights = torch.tensor([[0.01 * (10 * t + i + 1) for i in range(TOP_K)] for t in range(3)], device="cuda")
    x = torch.randn(3, hidden, device="cuda")
    route_ids = torch.tensor(sum(routes, []), dtype=torch.int64, device="cuda")
    c.dev.post(0, backend.planned, plan.count, route_ids, plan.slots, captured=True, cpu_input=(x, weights))
    torch.cuda.synchronize()
    assert c.kinds(3) == [LaneKind.HIT_CPU] * 3
    row = x_rows[0]
    for t in range(3):
        assert torch.equal(row[t * xb : (t + 1) * xb].view(torch.float16), x[t].half().cpu()), t
    table = tokens_max * xb
    assert int(row[table : table + 4].view(torch.int32)[0]) == 3
    planned = backend.planned[:3].tolist()
    for j, expert in enumerate(planned):
        at = table + CPU_TOKEN_TABLE_HEADER + 4 * j
        mask = int(row[at : at + 4].view(torch.int32)[0])
        assert mask == sum(1 << t for t in range(3) if expert in routes[t]), (j, expert)
        for t in range(3):
            if expert in routes[t]:
                at = table + CPU_TOKEN_TABLE_HEADER + 4 * lanes + 4 * (t * lanes + j)
                assert float(row[at : at + 4].view(torch.float32)[0]) == pytest.approx(
                    float(weights[t, routes[t].index(expert)])
                ), (j, t)
```

- [ ] **Step 2: Run and see it fail**

Same command as Task 4 Step 2 with `-k "every_token"`. Expected: `ImportError: cannot import name 'CPU_TOKEN_TABLE_HEADER'`.

- [ ] **Step 3: Create `cpu_token_table.h`**

```cpp
// The token table of a CPU experts input row that holds several tokens (plan
// 2026-10-06-dsv41-dspark-both-cpu-experts, a DSpark verify).
//
// A row holds tokens_max staged inputs, x_token_bytes apart (fp16, each padded to 16 bytes). With tokens_max > 1 the
// table follows them; the post kernel writes it for every lane of the record:
//   +0                                       u32  the record's tokens, 1..tokens_max
//   +kHeaderBytes + 4 j                      u32  lane j's token mask: bit t when token t routes lane j's expert
//   +kHeaderBytes + 4 lanes + 4 (t lanes + j)  u32  the fp32 bits of token t's routing weight for lane j's expert
// The CPU expert thread reads it to run one forward of `tokens` rows, each with its own weights
// (CpuExpertEngine::run_job). A one-token row has no table: the record's lane weight serves. The Python mirror is
// expert_lease_block.cpu_row_bytes.
#pragma once

#include <cstdint>

namespace sglang::expert_stream {

struct CpuTokenTable {
  static constexpr int64_t kMaxTokens = 32;    // one u32 mask bit per token
  static constexpr int64_t kHeaderBytes = 16;  // the token count, padded
};

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: The post** (`lease_kernels.cuh`)

Add `#include "cpu_token_table.h"` after `#include "lease_device.cuh"`. In `PostParams`, after `cpu_weights_count`:
```cpp
  // A verify's tokens (cpu_x's rows; 1 for one token), the host rows' token capacity (1: one-token rows with no token
  // table) and the bytes between two tokens' staged inputs (cpu_token_table.h).
  int64_t cpu_tokens;
  int64_t cpu_tokens_max;
  int64_t cpu_x_token_bytes;
```
Replace `stage_cpu_input`:
```cpp
// Stages the layer's input rows as fp16 into the host row, token t at t * cpu_x_token_bytes, 8 elements per 16-byte
// store, using the whole block. The launcher checks hidden % 8 == 0 and the alignment. Each thread fences its own
// stores at system scope before the barrier, so thread 0's later release of the record and demand_head orders all.
SGL_DEVICE void stage_cpu_input(const PostParams& p) {
  const int64_t vectors = p.cpu_hidden / 8;
  for (int64_t v = threadIdx.x; v < p.cpu_tokens * vectors; v += blockDim.x) {
    const int64_t t = v / vectors, c = v % vectors;
    __align__(16) __half h[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      h[k] = __float2half_rn(cpu_input_value(p.cpu_x_src, p.cpu_x_dtype, t * p.cpu_hidden + 8 * c + k));
    __stwt(
        reinterpret_cast<uint4*>(p.cpu_x_dst + t * p.cpu_x_token_bytes + 16 * c),
        *reinterpret_cast<const uint4*>(h));
  }
  __threadfence_system();
  __syncthreads();
}
```
In the kernel, after `if (threadIdx.x != 0) return;`, add:
```cpp
  // A multi-token row's token table (cpu_token_table.h): written for every lane before the record's release.
  uint8_t* const table = p.cpu_x_dst != nullptr && p.cpu_tokens_max > 1
                             ? p.cpu_x_dst + p.cpu_tokens_max * p.cpu_x_token_bytes
                             : nullptr;
  const int64_t top_k = p.cpu_tokens > 0 ? p.cpu_weights_count / p.cpu_tokens : 0;
  if (table != nullptr) st_relaxed_sys<uint32_t>(table, static_cast<uint32_t>(p.cpu_tokens));
```
Replace the lane-weight block (`weight[j] = 0.0f; if (p.cpu_x_dst != nullptr) { for ... } }`) with:
```cpp
    // The lane expert's routing weight summed over the routes that name it (one route at batch size 1), and for a
    // multi-token row each token's own weight and the mask of tokens that route it. 0 when CPU experts are off.
    weight[j] = 0.0f;
    if (p.cpu_x_dst != nullptr) {
      uint32_t routed = 0;
      for (int64_t r = 0; r < p.route_count && r < p.cpu_weights_count; ++r) {
        if (p.routes[r] != p.planned[j]) continue;
        const float w = cpu_input_value(p.cpu_weights, p.cpu_weights_dtype, r);
        weight[j] += w;
        if (table != nullptr) {
          const int64_t t = r / top_k;
          routed |= 1u << t;
          st_relaxed_sys<uint32_t>(
              table + expert_stream::CpuTokenTable::kHeaderBytes + 4 * Wire::kLanes + 4 * (t * Wire::kLanes + j),
              __float_as_uint(w));
        }
      }
      if (table != nullptr) st_relaxed_sys<uint32_t>(table + expert_stream::CpuTokenTable::kHeaderBytes + 4 * j, routed);
    }
```
In the launcher, add parameters after `cpu_weights,` (before Task 4's `spill`): `int64_t cpu_tokens_max, int64_t cpu_x_token_bytes,`.
Replace the check `RuntimeCheck(cpu_x.dim() == 2 && cpu_x.size(0) == 1, ...)` with:
```cpp
      RuntimeCheck(
          cpu_tokens_max >= 1 && cpu_tokens_max <= expert_stream::CpuTokenTable::kMaxTokens,
          "CPU experts: the rows hold 1-32 tokens");
      RuntimeCheck(
          cpu_x.dim() == 2 && cpu_x.size(0) >= 1 && cpu_x.size(0) <= cpu_tokens_max,
          "CPU experts: cpu_x is [tokens, hidden] with at most the rows' tokens");
      RuntimeCheck(cpu_weights.numel() % cpu_x.size(0) == 0, "CPU experts: cpu_weights holds top_k weights a token");
```
After `RuntimeCheck(cpu_hidden > 0 && cpu_hidden % 8 == 0, ...)` add:
```cpp
      RuntimeCheck(
          cpu_x_token_bytes >= 2 * cpu_hidden && cpu_x_token_bytes % 16 == 0,
          "CPU experts: cpu_x_token_bytes holds one fp16 row, in 16-byte steps");
```
Append to `PostParams{...}`:
```cpp
        .cpu_tokens = cpu_input ? cpu_x.size(0) : 1,
        .cpu_tokens_max = cpu_tokens_max,
        .cpu_x_token_bytes = cpu_x_token_bytes,
```

- [ ] **Step 5: Python**

`expert_lease_block.py`, at module level after the existing constants:
```python
# The CPU experts' input rows (cpu_token_table.h): CpuTokenTable::kHeaderBytes and kMaxTokens.
CPU_TOKEN_TABLE_HEADER = 16
CPU_TOKENS_MAX = 32


def cpu_row_bytes(hidden: int, tokens: int, lanes: int) -> int:
    """Bytes of one CPU experts input row: ``tokens`` fp16 inputs, each padded to 16 bytes, then, for more than one
    token, the token table: a 16-byte header, a u32 mask per lane and an fp32 weight per token and lane."""
    x = -(-2 * hidden // 16) * 16
    return tokens * x + (CPU_TOKEN_TABLE_HEADER + 4 * lanes + 4 * tokens * lanes if tokens > 1 else 0)
```
`expert_stream_transport.py`:
- Import `CPU_TOKENS_MAX, cpu_row_bytes` from `expert_lease_block`.
- In `ExpertStreamDevice.__init__`, after `self.cpu_out_rows = None`:
  ```python
          # The rows' token capacity and the bytes between two tokens' staged inputs (enable_cpu_experts); 1 and 0
          # until then.
          self.cpu_tokens_max, self.cpu_x_token_bytes = 1, 0
  ```
- In `enable_cpu_experts`, replace the final `self.cpu_x_rows, self.cpu_out_rows = x_rows, out_rows` with:
  ```python
          tokens = int(out_rows.shape[2]) if out_rows.dim() == 4 else 1
          hidden = int(out_rows.shape[-1])
          if not 1 <= tokens <= CPU_TOKENS_MAX:
              raise ValueError(f"CPU expert rows hold 1-{CPU_TOKENS_MAX} tokens, not {tokens}")
          if x_rows.shape[1] < cpu_row_bytes(hidden, tokens, self.wire.lanes):
              raise ValueError(
                  f"x_rows holds {x_rows.shape[1]} bytes a row; {tokens} tokens of {hidden} need "
                  f"{cpu_row_bytes(hidden, tokens, self.wire.lanes)}"
              )
          self.cpu_x_rows, self.cpu_out_rows = x_rows, out_rows
          self.cpu_tokens_max, self.cpu_x_token_bytes = tokens, -(-2 * hidden // 16) * 16
  ```
  Append to its docstring: ``A multi-token row's ``out_rows`` is ``[layers, 2 * nodes, tokens, hidden]``; ``x_rows``
  then holds ``expert_lease_block.cpu_row_bytes`` a row (the token table follows the inputs).``
- `cpu_out_part_stride`: change `if self.cpu_out_rows.dim() == 3` to `if self.cpu_out_rows.dim() >= 3`. Its docstring
  gains: "(``tokens * hidden`` for a multi-token row)".
- In `post`, replace the CPU input block's shape handling:
  ```python
              cpu_x, cpu_weights = cpu_input
              tokens = cpu_x.shape[0] if cpu_x.dim() == 2 else 1
              if tokens > self.cpu_tokens_max:
                  raise ValueError(f"a {tokens}-token input does not fit rows of {self.cpu_tokens_max} tokens")
              if cpu_x.shape[-1] * 2 > self.cpu_x_rows.shape[1]:
                  raise ValueError(
                      f"a {cpu_x.shape[-1]}-wide input does not fit the {self.cpu_x_rows.shape[1]}-byte row"
                  )
              cpu_x, cpu_weights = cpu_x.reshape(tokens, -1), cpu_weights.reshape(-1)
              cpu_x_dst = int(self.cpu_x_rows[row].data_ptr())
  ```
  Before the module call, add
  `token_bytes = self.cpu_x_token_bytes or -(-2 * int(cpu_x.shape[-1]) // 16) * 16`. In the call, insert after
  `cpu_weights,` (before `spill_on,`): `self.cpu_tokens_max, token_bytes,`. The `cpu_x.reshape(1, -1)` this replaces
  was the latent overflow the research found (an `[M, H]` input became one `[1, M*H]` row).

- [ ] **Step 6: Raw calls again** (`test_exl3_lease_kernels_cuda.py`)

Insert `1, 16,` right after the CPU triple in both calls, before Task 4's three spill arguments. The first call's tail
becomes `no_i32, 0, no_i32, 1, 16, 0, torch.zeros(1, ...int32...), torch.zeros(1, ...int64...), 0,`. The second's
becomes `*cpu_args, 1, 32, 0, ...` (its CPU input is `[1, 8]` fp32, so 16 bytes of fp16 suffice; 32 is fine too).

- [ ] **Step 7: Run and pass** — same suites as Task 4 Step 7. Expected: `EXIT=0`.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/cpu_token_table.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/ops/moe/expert_lease_block.py python/sglang/kernels/ops/moe/expert_stream_transport.py \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py
git commit -m "feat(exl3-cpu-experts): a verify's post stages every token and each lane's per-token weights

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 6: The CPU expert thread runs one M-row forward per job

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`submit_host_lanes`, `submit_landed_cpu_misses`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`enable_cpu_experts`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamHost.enable_cpu_experts`)
- Test: `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`

**Interfaces:**
- Consumes: Task 5's row layout and `cpu_row_bytes`.
- Produces:
  - `CpuJob.per_token` (bool) and `CpuJob.lanes[kLanes]`;
  - `CpuExpertConfig.tokens`, `.x_token_bytes`;
  - FFI `expert_stream_enable_cpu_experts(handle, group, kernel, split, cores, x_rows, out_rows, hidden, parts, tokens, threads, spin_ns, keep_warm_ns)`;
  - `ExpertStreamHost.enable_cpu_experts` accepts 4-dim `out_rows`.
  - A per-token job on a multi-token row runs `rows = tokens` from the table. The engine zeroes its rows first when the
    job does not accumulate, then always accumulates.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_cpu_lane_order_cuda.py`)

```python
@pytest.mark.parametrize("tokens", [1, 3])
def test_a_verify_records_cpu_lanes_run_one_forward_of_its_tokens(tmp_path, tokens):
    """Three RAM hits, every lane the CPU's (split[3] = 3), in rows that hold 4 tokens. The record's CPU job runs one
    forward of `tokens` rows: row t names a lane's RAM slot only where token t routes its expert (-1 elsewhere, Review
    Focus 4) with t's own weight, and its output row is t's partial alone, over stale bytes (7.0). One token in 4-token
    rows reads the table too. Mutations: every row the record's summed weight; the slot kept where t does not route;
    no zeroing of a non-accumulating job's rows."""
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

    row, experts, tokens_max = 0, [3, 5, 7], 4
    lanes = lease.wire_layout(8).lanes
    c = Chain(tmp_path, copy_engine=True, start=False)
    try:
        x_rows = torch.zeros((LAYERS, lease.cpu_row_bytes(HIDDEN, tokens_max, lanes)), dtype=torch.uint8).pin_memory()
        out_rows = torch.full((LAYERS, 2, tokens_max, HIDDEN), 7.0, dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(c.host.test_kernel_address(), list(range(lanes + 1)), cores, x_rows, out_rows,
                                  threads=2)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        backend, plan, dev = c.backends[row], c.plans[row], c.dev
        c.plan(experts, row)
        c.gather(row)
        torch.cuda.synchronize()
        assert c.handled() and set(experts) <= c.resident(row)
        with paused(c.host):
            ram_slot = {e: s for s, (state, e, _) in enumerate(c.host.slot_info(row)) if e >= 0}
        c.host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
        c.host.arm_copy_engine()

        c.plan(experts, row)
        backend._stage_planned(plan)
        planned = backend.planned[:3].tolist()
        routes = [[3, 5, 8, 9, 10, 11], [7, 3, 12, 13, 14, 15], [5, 8, 12, 9, 13, 10]][:tokens]
        weights = torch.tensor([[0.01 * (10 * t + i + 1) for i in range(TOP_K)] for t in range(tokens)], device="cuda")
        route_ids = torch.tensor(sum(routes, []), dtype=torch.int64, device="cuda")
        x = torch.randn(tokens, HIDDEN, device="cuda").half()
        dev.post(row, backend.planned, plan.count, route_ids, plan.slots, captured=True, cpu_input=(x, weights))
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
        assert _until(lambda: len(c.host.test_kernel_calls()) == tokens)
        torch.cuda.synchronize()

        assert c.kinds(3) == [LaneKind.HIT_CPU] * 3
        assert dev.cpu_lanes.tolist() == [0b111, PART_HITS]
        for t, call in enumerate(c.host.test_kernel_calls()):
            want_slots = [ram_slot[e] if e in routes[t] else -1 for e in planned]
            want_weights = [float(weights[t, routes[t].index(e)]) if e in routes[t] else 0.0 for e in planned]
            assert call["slots"] == want_slots, (t, call)
            assert call["weights"] == pytest.approx(want_weights), (t, call)
            assert call["accumulate"], "a per-token job accumulates into the rows the engine zeroed"
            partial = sum(w * (s + 1) for s, w in zip(want_slots, want_weights))
            assert torch.allclose(out_rows[row, 0, t], torch.full((HIDDEN,), partial), atol=1e-4), t
    finally:
        c.close()
```
Token 2 (`[5, 8, 12, 9, 13, 10]`) routes expert 5 only: lanes 3 and 7 get −1 there.

- [ ] **Step 2: Run and see it fail**

`-k "one_forward_of_its_tokens"` under the GPU lock, as in Task 4. Expected: a `ValueError` from
`ExpertStreamHost.enable_cpu_experts` (`out_rows must be ... tensor`).

- [ ] **Step 3: `cpu_experts.h`**

Add `#include "../cpu_token_table.h"` and `#include <array>`. Replace `struct CpuJob`:

```cpp
/// One forward over up to Wire::kLanes lanes of one row.
///
/// A record produces at most one job for its CPU hits (part 0) and one per batch of CPU misses that landed together
/// (part 1). Every miss job after the first adds into part 1 in landing order, so that part's fp32 sum order varies
/// from run to run. A record's job is per_token: on a multi-token row (CpuExpertConfig::tokens > 1) the row's token
/// table gives each token its own slots and weights, `lanes` naming each job lane's column there.
struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;
  bool accumulate = false;  // add into the part rather than overwrite it
  bool per_token = false;   // a record's job: a multi-token row's table gives each token its weights
  uint32_t seq = 0;         // from claim()
  int32_t k = 0;
  int32_t slots[wire::Wire::kLanes] = {};
  float weights[wire::Wire::kLanes] = {};
  int32_t lanes[wire::Wire::kLanes] = {};  // each job lane's record lane
};
```
In `CpuExpertConfig`, after `hidden`:
```cpp
  int64_t tokens = 1;         // tokens a row holds; above 1 a token table follows the inputs (cpu_token_table.h)
  int64_t x_token_bytes = 0;  // bytes between two tokens' staged inputs
```
In `validate()`, after the hidden check:
```cpp
    if (c.tokens < 1 || c.tokens > CpuTokenTable::kMaxTokens || (c.tokens > 1 && c.x_token_bytes < 2 * c.hidden))
      throw std::runtime_error(prefix_ + "the CPU expert rows hold 1-32 tokens of the hidden size");
```
Replace `run_job`'s head, from `cpu_experts::ForwardCall call;` through `call.cores = config_.cores;`:
```cpp
    cpu_experts::ForwardCall call;
    call.rows = 1;
    call.k = job.k;
    call.threads = config_.threads;
    call.x = config_.x_base + job.row * config_.x_stride;
    call.slots = job.slots;
    call.weights = job.weights;
    call.out =
        reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride);
    call.accumulate = job.accumulate;
    call.cores = config_.cores;
    if (job.per_token && config_.tokens > 1) {
      call.rows = static_cast<int32_t>(expand_tokens(job, call.x));
      call.slots = token_slots_.data();
      call.weights = token_weights_.data();
      // A token that routes none of the lanes must read 0, whatever an earlier record left: zero the rows here rather
      // than trust the kernel's overwrite, then accumulate.
      if (!job.accumulate) std::memset(call.out, 0, static_cast<size_t>(call.rows) * config_.hidden * sizeof(float));
      call.accumulate = true;
    }
```
Add these private members after `run_job`:
```cpp
  /// A record's job on a multi-token row: each token's slots and weights from the row's token table
  /// (cpu_token_table.h), -1 where the token does not route the lane's expert (the kernel skips only slot -1, never a
  /// zero weight). Returns the record's tokens; fails stop on a count the row cannot hold.
  uint32_t expand_tokens(const CpuJob& job, const uint8_t* x) {
    constexpr int64_t kLanes = wire::Wire::kLanes;
    const uint8_t* table = x + config_.tokens * config_.x_token_bytes;
    const uint32_t tokens = load_u32(table);
    if (tokens < 1 || tokens > static_cast<uint32_t>(config_.tokens)) {
      fail_stop(prefix_ + "row " + std::to_string(job.row) + "'s token table holds " + std::to_string(tokens) +
                " tokens");
      return 1;
    }
    for (uint32_t t = 0; t < tokens; ++t)
      for (int32_t i = 0; i < job.k; ++i) {
        const int32_t lane = job.lanes[i];
        const bool routed = (load_u32(table + CpuTokenTable::kHeaderBytes + 4 * lane) >> t & 1u) != 0;
        const uint32_t bits = load_u32(table + CpuTokenTable::kHeaderBytes + 4 * kLanes + 4 * (t * kLanes + lane));
        token_slots_[t * job.k + i] = routed ? job.slots[i] : -1;
        token_weights_[t * job.k + i] = routed ? std::bit_cast<float>(bits) : 0.0f;
      }
    return tokens;
  }

  static uint32_t load_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
  }
```
And data members next to `compute_ns_`:
```cpp
  // A per-token job's expanded slots and weights, [tokens][k]; this thread only.
  std::array<int32_t, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> token_slots_{};
  std::array<float, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> token_weights_{};
```

- [ ] **Step 4: `ram_tier.h`**

In `submit_host_lanes`, after `cpu_job.seq = first;` add `cpu_job.per_token = true;`. Inside its lane loop, before
`++cpu_job.k;`, add `cpu_job.lanes[cpu_job.k] = lane.lane;`.
In `submit_landed_cpu_misses`, after `cpu_job.accumulate = ...;` add `cpu_job.per_token = true;`. Before
`++cpu_job.k;`, add `cpu_job.lanes[cpu_job.k] = static_cast<int32_t>(plan.miss_lane[i]);`.
`split_calibration.h`'s jobs stay `per_token = false`: they are single-token by construction.

- [ ] **Step 5: `ffi_exports.h` `enable_cpu_experts`**

Add `int64_t tokens,` after `int64_t parts,`. Replace the block from `if (out_rows.size(1) < ...` through
`config.out_part_stride = ...;` with:
```cpp
    if (tokens < 1 || tokens > expert_stream::CpuTokenTable::kMaxTokens)
      throw std::runtime_error(error_prefix<Layout>() + "CPU expert rows hold 1-32 tokens");
    if (tokens > 1 && parts != 2)
      throw std::runtime_error(error_prefix<Layout>() + "a multi-token CPU expert row has two parts per group");
    config.tokens = tokens;
    config.x_token_bytes = (2 * hidden + 15) / 16 * 16;
    const int64_t table = tokens > 1 ? expert_stream::CpuTokenTable::kHeaderBytes +
                                           4 * expert_stream::Wire::kLanes * (1 + tokens)
                                     : 0;
    if (x_rows.size(1) < tokens * config.x_token_bytes + table)
      throw std::runtime_error(error_prefix<Layout>() + "x_rows is narrower than its tokens and token table");
    if (out_rows.size(1) < expert_stream::Wire::kNodes * parts * tokens * hidden)
      throw std::runtime_error(error_prefix<Layout>() + "out_rows is narrower than every group's parts");
    config.out_base = static_cast<uint8_t*>(out_rows.data_ptr()) + group * parts * tokens * hidden * sizeof(float);
    config.out_stride = out_rows.size(1) * static_cast<int64_t>(sizeof(float));
    config.out_part_stride = parts == 2 ? tokens * hidden * static_cast<int64_t>(sizeof(float)) : 0;
```
Include `cpu_token_table.h` there if it is not reached through `cpu_experts.h`.

- [ ] **Step 6: `ExpertStreamHost.enable_cpu_experts`** (`expert_stream_transport.py`)

Replace the `out_rows` validation and the `parts`/`hidden` lines:
```python
        if (
            out_rows.dtype != torch.float32
            or out_rows.device.type != "cpu"
            or not out_rows.is_contiguous()
            or not (
                (out_rows.dim() == 2 and self.nodes == 1)
                or (out_rows.dim() in (3, 4) and out_rows.shape[1] == 2 * self.nodes)
            )
        ):
            raise ValueError(
                "out_rows must be a contiguous host float32 [rows, hidden] at one node, [rows, 2 * nodes, hidden], or "
                "[rows, 2 * nodes, tokens, hidden] for multi-token rows"
            )
        parts = 1 if out_rows.dim() == 2 else 2
        tokens = int(out_rows.shape[2]) if out_rows.dim() == 4 else 1
        hidden = int(out_rows.shape[-1])
```
Pass `int(tokens),` after `parts,` in the module call. Append to the docstring: ``With ``[rows, 2 * nodes, tokens,
hidden]`` rows a record's CPU job runs one forward of its tokens from the row's token table.``

- [ ] **Step 7: Run and pass**

GPU lock, cores 32-63:
`test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_cpu_split_calibration_cuda.py`.
CPU, cores 0-63:
`test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_service.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`.
Expected: `EXIT=0` for both. If a registered test calls `expert_stream_enable_cpu_experts` with the old arity, it
fails with a TypeError; add `1` for `tokens` after `parts` there, and list each such file in the commit.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py
git commit -m "feat(exl3-cpu-experts): a verify's CPU job runs one forward of its tokens from the row's token table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 7: The route tables seed every token from its own partial

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh`
- Modify: `python/sglang/srt/layers/quantization/exl3/fused_moe.py` (`Exl3FusedMoE.run`)
- Test: `test/manual/dsv41/test_exl3_moe_split_parity_cuda.py`

**Interfaces:**
- Consumes: Task 6's part layout. Part p's token t is at `cpu_out + p * part_stride + t * hidden`, with
  `part_stride = tokens * hidden`.
- Produces: `Exl3FusedMoE.run(x [M, H], ..., cpu=(cpu_lanes, dst_slots, cpu_out, part_stride))` for any
  `1 <= M <= tokens`.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_moe_split_parity_cuda.py`)

```python
VERIFY_TOKENS = 4


def _fused_tokens(slot_rows, device, tokens):
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.exl3.fused_moe import Exl3FusedMoE

    with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(True):
        return Exl3FusedMoE(
            slot_rows, slot_rows["w13_trellis"].shape[0], hidden=slot_rows["w13_suh"].shape[-1],
            inter=slot_rows["w2_suh"].shape[-1], top_k=TOP_K, device=device, tokens=tokens,
        )


def test_a_verify_cpu_partial_seeds_every_token(slot_rows):
    """M = 4 tokens of a DSpark verify. The CPU lanes are distinct slots of the routes' union (lane i's dst_slots
    entry is its slot), so every route on one leaves the fused MoE in every token, and token t's [hidden] partial seeds
    token t. With a zero partial the run is, bit for bit, the run with those routes' weights zeroed; with the GPU's own
    per-token partial of those routes it is the full run up to fp32 reassociation. Mutation: seed only i < hidden (the
    one-token kernel) -- tokens 1-3 lose their partial."""
    device = slot_rows["w13_trellis"].device
    fused = _fused_tokens(slot_rows, device, VERIFY_TOKENS)
    hidden = slot_rows["w13_suh"].shape[-1]
    partial = torch.zeros((VERIFY_TOKENS, hidden), dtype=torch.float32).pin_memory()
    zero = torch.zeros_like(partial).pin_memory()
    keep = torch.ones(1, device=device)
    gen = torch.Generator().manual_seed(932)
    for trial in range(TRIALS * 4):
        remap = torch.cat([torch.randperm(fused.slots, generator=gen)[:TOP_K] for _ in range(VERIFY_TOKENS)]).to(device)
        weights = torch.softmax(torch.randn(VERIFY_TOKENS, TOP_K, generator=gen), 1).reshape(-1).to(device)
        x = (torch.randn((VERIFY_TOKENS, hidden), generator=gen) * 0.5).to(device)
        union = sorted(set(remap.tolist()))
        picked = [s for s in union if int(torch.randint(0, 2, (1,), generator=gen))] or union[:1]
        lanes = torch.tensor(picked, dtype=torch.int32, device=device)
        on_cpu = torch.isin(remap, lanes.long())
        words = torch.tensor([(1 << len(picked)) - 1, PART_HITS], dtype=torch.int32, device=device)
        want = fused.run(x, weights, remap, keep, ACT_LIMIT).clone()
        without = fused.run(x, torch.where(on_cpu, torch.zeros_like(weights), weights), remap, keep, ACT_LIMIT).clone()
        partial.copy_(fused.run(x, torch.where(on_cpu, weights, torch.zeros_like(weights)), remap, keep, ACT_LIMIT))
        got_zero = fused.run(x, weights, remap, keep, ACT_LIMIT, cpu=(words, lanes, zero.data_ptr(), 0)).clone()
        assert torch.equal(_bits(got_zero), _bits(without)), f"picked={picked}"
        got = fused.run(x, weights, remap, keep, ACT_LIMIT, cpu=(words, lanes, partial.data_ptr(), 0)).clone()
        scale = float(want.abs().max())
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5 * scale), f"picked={picked}"
```

- [ ] **Step 2: Run and see it fail**

On divix01, GPU lock, `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`:
`-k a_verify_cpu_partial_seeds_every_token`. Expected: `RuntimeError: CPU experts run one token, not 4`.

- [ ] **Step 3: Implement**

`fused_moe.py` `run`: delete
```python
        if cpu is not None and m != 1:
            raise RuntimeError(f"CPU experts run one token, not {m}")
```
and replace the docstring's "CPU experts only, one token:" with "CPU experts only, any M (each token's partial at
``t * hidden`` within each part):".

`exl3_route_tables.cuh`:
- Delete `RuntimeCheck(!cpu_on || M_.unwrap() == 1, "CPU experts run one token (x has one row)");`.
- In the kernel's comment, replace "(plan 2026-09-29-dsv41-cpu-experts; cpu_lanes non-null; one token only)" with
  "(plan 2026-09-29-dsv41-cpu-experts; cpu_lanes non-null; any M tokens, plan 2026-10-06-dsv41-dspark-both-cpu-experts)".
- Replace "part p at cpu_out + p * part_stride (host memory, [hidden] fp32)" with "part p at cpu_out + p *
  part_stride (host memory, [tokens][hidden] fp32)".
- Replace the loop comment `// CPU experts run one token, so a seeded part is read only where i < hidden.` with
  `// Each token's partial is its own [hidden] row of each part, so element i of the output reads element i of each.`
  The loop itself already reads `cpu_out + part * part_stride + i` for `i < tokens * hidden`, which is exactly the
  `[tokens][hidden]` part layout.

- [ ] **Step 4: Run and pass**

The whole file `test/manual/dsv41/test_exl3_moe_split_parity_cuda.py` and
`test/manual/dsv41/test_exl3_fused_moe_multitoken_gpu.py`, GPU lock. Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh \
  python/sglang/srt/layers/quantization/exl3/fused_moe.py test/manual/dsv41/test_exl3_moe_split_parity_cuda.py
git commit -m "feat(exl3-cpu-experts): the fused MoE's route tables seed each verify token from its own CPU partial

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 8: DIRECT gives victims to the first V lanes of a lane-per-route record

Spill always runs with one lane per route: `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES` is unset, so the miss width is
the routes (36). The lane count therefore never falls below the distinct misses (Review Focus 5). The gather calls
`clamp_gather_misses` only when the miss width is below the routes (`expert_stream.py:1578-1580`), so under spill it
never runs, and the clamp asserts so. The overflow flag must still be read after every verify, because the post can set
it (case 1), so `narrow_gather` covers spill too.

**Files:**
- Modify: `python/sglang/srt/environ.py` (next to `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES`, line ~524)
- Modify: `python/sglang/srt/layers/moe/expert_residency_gpu.py` (`_init_insert_direct`, `_rank_victims`, `gather_destinations`, `fused_gather_destinations`, `clamp_gather_misses`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` (the wide destinations kernel and its launcher)
- Modify: `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` (`direct_gather_destinations`)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (allocator floors)
- Modify: `python/sglang/srt/layers/moe/exl3_expert_format.py` (`plan_graph_gather`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`plan_staging_width`, `staging_width`)
- Test: `test/registered/unit/layers/moe/test_expert_residency_gpu.py`, `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`, `test/manual/dsv41/test_direct_gather_wide_gpu.py`

**Interfaces:**
- Consumes: Task 2's wide destinations kernel.
- Produces:
  - `envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` (`EnvInt(0)`);
  - `GpuResidencyUpdater.victim_lanes: int`, equal to `miss_rows` when the env is unset;
  - spill (`victim_lanes < miss_rows`) requires `miss_rows` equal to every layer's routes. It sets `narrow_gather`.
  - Under spill a lane that is not live gets destination `max_capacity` and destination slot −1 (the post's spill
    marker); otherwise (0, 0) as before.
  - `direct_gather_destinations(..., idle_destination=0, idle_slot=0)`. Nonzero idle values always run the wide
    kernel, so the narrow BS1 kernel stays untouched.
  - `Exl3RamMissService.plan_staging_width(rows)`.

- [ ] **Step 1: Write the failing tests**

Append to `TestInsertOnMiss` in `test_expert_residency_gpu.py` (the class holding `_narrow`). Two tokens of top-2 route
4 ids a layer, so `_narrow(model, 0, fused)` builds a lane-per-route gather of width 4.

```python
    # ----- spill: victims for the first V lanes of a lane-per-route gather (SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES) -----

    def _spill(self, fused, lanes=0, victims=1):
        from sglang.srt.environ import envs

        with envs.SGLANG_DSV41_CPU_EXPERTS.override(True), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(
            victims
        ):
            return self._narrow(_model(), lanes, fused)

    def test_victim_lanes_need_cpu_experts_and_a_value_below_the_lanes(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(1), self.assertRaisesRegex(
            ValueError, "VICTIM_LANES=1 needs SGLANG_DSV41_CPU_EXPERTS=1"
        ):
            self._narrow(_model(), 0, False)
        with self.assertRaisesRegex(ValueError, "below the 4 miss lanes"):
            self._spill(False, victims=4)

    def test_spill_needs_a_lane_per_route(self):
        """Review Focus 5 by construction: a narrowed MISS_LANES could leave more misses than lanes, so spill refuses it."""
        with self.assertRaisesRegex(ValueError, "unset SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES"):
            self._spill(False, lanes=2)

    def test_victim_lanes_cap_the_shortlist_and_the_floor(self):
        """Only shortlist column 0 is ever a victim once a boundary ranks; the floor is twice the victim lanes, and the
        manager reads the overflow flag after every verify (the post can set it: case 1)."""
        for fused in (False, True):
            self.model = _model()
            manager = self._spill(fused)
            updater = manager.gpu_residency
            self.assertEqual((updater.miss_rows, updater.victim_lanes), (4, 1))
            self.assertTrue(updater.narrow_gather and manager.narrow_graph_gather)
            graph, static, outputs = self.capture(manager, tokens=2)
            for _ in range(2):
                self.replay_verify(manager, graph, static, outputs, [[[0, 1], [0, 1]]] * LAYERS, check_outputs=False)
            self.assertFalse(bool(updater.victim_valid[:, 1:].any()))
            self.assertTrue(bool(updater.victim_valid[:, 0].all()))

    def test_victimless_lanes_take_the_idle_destination_in_both_paths(self):
        """Two misses, one victim: lane 0 is live at the victim, lanes 1-3 are not and get (slot_dump, -1), the marker
        the post spills to the CPU; their routes remap past every slot column. The torch chain and the fused (wide)
        kernel agree."""
        from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_gather_destinations

        manager = self._spill(False)
        updater, streamer = manager.gpu_residency, manager.streamers[0]
        dump = updater.max_capacity
        base = streamer.row_planner.scratch_base
        updater.victims[0].copy_(torch.tensor([3, 5, 6, 7], device="cuda"))
        updater.victim_valid[0].copy_(torch.tensor([True, False, False, False], device="cuda"))
        streamer._graph_miss_count.fill_(2)
        remap = torch.tensor([base, base + 1, base + 1, base], dtype=torch.int64, device="cuda")
        got = updater.gather_destinations(0, remap, torch.full((4,), -1, dtype=torch.int64, device="cuda"), base)
        _, _, destinations, live = updater._pending_commit
        self.assertEqual(destinations.tolist(), [3, dump, dump, dump])
        self.assertEqual(live.tolist(), [True, False, False, False])
        self.assertEqual(streamer._graph_destination_slots[:4].tolist(), [3, -1, -1, -1])
        self.assertEqual(got.tolist(), [3, dump, dump, 3])
        ids = torch.tensor([7, 9, 9, 7], dtype=torch.int64, device="cuda")
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.int64, device="cuda")
        slots_out = torch.zeros(4, dtype=torch.int32, device="cuda")
        dest_out = torch.zeros(4, dtype=torch.int64, device="cuda")
        live_out = torch.zeros(4, dtype=torch.bool, device="cuda")
        remap_out = torch.zeros(4, dtype=torch.int64, device="cuda")
        direct_gather_destinations(
            ids, expert_to_slot, updater.victims[0], updater.victim_valid[0], streamer._graph_miss_count, remap, base,
            slots_out, dest_out, live_out, remap_out, idle_destination=dump, idle_slot=-1,
        )
        self.assertEqual(slots_out.tolist(), [3, -1, -1, -1])
        self.assertEqual(dest_out.tolist(), [3, dump, dump, dump])
        self.assertEqual(live_out.tolist(), [True, False, False, False])
        self.assertEqual(remap_out.tolist(), got.tolist())

    def test_spill_never_clamps(self):
        """Review Focus 5: a lane per route leaves no count past the lanes, so the gather never calls the clamp; four
        misses into one victim post with the count intact. Mutation: call the clamp under spill -- its assert fires."""
        from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

        for fused in (False, True):
            self.model = _model()
            manager = self._spill(fused)
            with unittest.mock.patch.object(GpuResidencyUpdater, "clamp_gather_misses", side_effect=AssertionError):
                graph, static, outputs = self.capture(manager, tokens=2)
                mapping = manager.caches[0].expert_to_slot.tolist()
                outsiders = [e for e, s in enumerate(mapping) if s < 0]
                hits = [
                    [[e for e, s in enumerate(manager.caches[layer].expert_to_slot.tolist()) if s >= 0][:TOP_K]] * 2
                    for layer in range(1, LAYERS)
                ]
                self.replay_verify(manager, graph, static, outputs, [[outsiders[0:2], outsiders[2:4]]] + hits,
                                   check_outputs=False)
            self.assertEqual(int(manager.streamers[0]._graph_miss_count.item()), 4, f"fused={fused}")
            self.assertEqual(int(manager.gpu_residency.overflow_flag.item()), 0, f"fused={fused}")
```
`EXPERTS` (12) is the module constant. The victims are slot ids under the 8-slot layers `_narrow` builds. The NVFP4
harness has no post, so the victimless routes' rows are not checked (`check_outputs=False`).

Append to `test/manual/dsv41/test_direct_gather_wide_gpu.py`:
```python
def test_idle_destinations_run_the_wide_kernel_at_any_width():
    """At a width the narrow kernel could take, nonzero idle values still give (idle, idle_slot) for every lane that
    is not live, so the narrow BS1 kernel never needs them."""
    from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_gather_destinations

    cuda = dict(device="cuda")
    ids = torch.tensor([1, 2, 3, 4], **cuda)
    expert_to_slot = torch.full((8,), -1, dtype=torch.int64, **cuda)
    out = [torch.zeros(4, dtype=t, **cuda) for t in (torch.int32, torch.int64, torch.bool, torch.int64)]
    direct_gather_destinations(
        ids, expert_to_slot, torch.tensor([3, 5, 6, 7], **cuda), torch.tensor([True, False, False, False], **cuda),
        torch.tensor([4], dtype=torch.int32, **cuda), torch.tensor([20, 21, 22, 23], **cuda), 20, *out,
        idle_destination=99, idle_slot=-1,
    )
    assert out[0].tolist() == [3, -1, -1, -1] and out[1].tolist() == [3, 99, 99, 99]
```

Append to `test_exl3_ram_miss_attach_lanes.py`:
```python
def test_victim_lanes_stage_their_width_not_the_lanes(tiers):
    """Spill: the post types up to 36 lanes, but only the V victim lanes stage (a forced miss never does), so a row
    reserves V staging slots (Exl3ExpertFormat.plan_graph_gather plans it)."""
    service, streamers = tiers
    with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(2):
        streamers[0].format.plan_graph_gather(streamers[0], 36)
    assert service.resolved_lanes() == 40
    assert service.staging_width() == 2
    assert service.staging_for(CAPACITY) == 2
```

- [ ] **Step 2: Run and see them fail**

On divix01:
- GPU lock: `test/registered/unit/layers/moe/test_expert_residency_gpu.py -k "victim or spill"` and
  `test/manual/dsv41/test_direct_gather_wide_gpu.py`;
- CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -k victim_lanes`.
Expected: `AttributeError: ... SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES`, and a `TypeError` on `idle_destination`.

- [ ] **Step 3: The env** (`environ.py`, right after `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES = EnvInt(0)`)

```python
    # Spill (SGLANG_DSV41_CPU_EXPERTS, a DSpark verify with a lane per route: MISS_LANES unset): only the first V of a
    # gather's miss lanes in plan order take a VRAM victim and a staging slot; the post makes every later lane a CPU
    # lane, a miss of which the host reads into a RAM victim. 0: every miss lane may take a victim. 1 <= V < the routes.
    SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES = EnvInt(0)
```

- [ ] **Step 4: The updater** (`expert_residency_gpu.py`)

In `_init_insert_direct`, right after `width = self.miss_rows`:
```python
        victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
        if victims and not envs.SGLANG_DSV41_CPU_EXPERTS.get():
            raise ValueError(
                f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} needs SGLANG_DSV41_CPU_EXPERTS=1: only the "
                "CPU serves the lanes that take no victim"
            )
        if victims and not 1 <= victims < width:
            raise ValueError(
                f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} must be below the {width} miss lanes"
            )
        if victims and any(s.graph_miss_width != s.graph_gather_rows for s in self.streamers):
            # A lane per route: a verify's distinct misses never outnumber its lanes, so nothing is clamped.
            raise ValueError(
                "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES gives every route a lane: unset "
                "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES"
            )
        # Lanes that may take a victim; the rest are the CPU's (the post's spill).
        self.victim_lanes = victims or width
```
Change the floor check to `if cache.capacity < 2 * self.victim_lanes:`. In its message, "twice its graph-gather miss
lanes; layer {layer_id} has {cache.capacity} slots for {width} lanes" becomes "twice its graph-gather victim lanes;
layer {layer_id} has {cache.capacity} slots for {self.victim_lanes} lanes". Change the `self.narrow_gather = any(...)`
line to:
```python
        # The overflow flag is read after every verify that can set it: a narrowed gather's clamp, or spill's post
        # (forced lanes before the copy engine arms).
        self.narrow_gather = self.victim_lanes < width or any(
            streamer.graph_miss_width < streamer.graph_gather_rows for streamer in self.streamers
        )
```
Append to the docstring: "With spill (``victim_lanes`` below ``miss_rows``, a lane per route) only the first
``victim_lanes`` shortlist columns are victims, so the floor is ``2 * victim_lanes``."

In `_rank_victims`, after `self.victim_valid.copy_(ranked.values[:, :width] < never)`:
```python
        if self.victim_lanes < width:
            # Spill: only the first victim_lanes columns are victims; the post sends the other lanes to the CPU.
            self.victim_valid[:, self.victim_lanes :] = False
```
Add a method after `_rank_victims`:
```python
    def _idle_destination(self) -> tuple[int, int]:
        """(destination, destination slot) of a lane that is not live. (0, 0) normally, a slot the clamp keeps any copy
        out of. With spill (slot_dump, -1): -1 tells the post the lane found no victim, and a route remapped to
        slot_dump is past every slot column, so the fused MoE skips it and the commit writes only the dump column."""
        return (self.max_capacity, -1) if self.victim_lanes < self.miss_rows else (0, 0)
```
In `gather_destinations`, replace
```python
        destinations = torch.where(live, usable, torch.zeros_like(usable))
        # The plan's slot buffer has a row per route; the lanes are the first miss_rows.
        streamer._graph_destination_slots[: self.miss_rows].copy_(destinations.to(torch.int32))
```
with
```python
        idle, idle_slot = self._idle_destination()
        destinations = torch.where(live, usable, torch.full_like(usable, idle))
        # The plan's slot buffer has a row per route; the lanes are the first miss_rows.
        streamer._graph_destination_slots[: self.miss_rows].copy_(
            torch.where(live, destinations, torch.full_like(destinations, idle_slot)).to(torch.int32)
        )
```
In `fused_gather_destinations`, compute `idle, idle_slot = self._idle_destination()` first and pass
`idle_destination=idle, idle_slot=idle_slot` to `direct_gather_destinations`.

At the top of `clamp_gather_misses`, add:
```python
        # Spill gives every route a lane (_init_insert_direct), so its gathers never call this.
        assert self.victim_lanes == self.miss_rows, "spill never clamps: its gather has a lane per route"
```
The body is otherwise unchanged.

- [ ] **Step 5: The kernel** (`direct_gather.cuh`, `expert_residency_direct_gather.py`)

The narrow (32-thread) destinations kernel is not touched. The wide kernel from Task 2 gains
`int64_t idle_destination, int32_t idle_slot` after `remap_out`, and its live block becomes:
```cpp
  if (entry) {
    const bool live = t < miss_count[0] && usable_valid[t];
    const int64_t destination = live ? usable[t] : idle_destination;
    destinations[t] = destination;
    destinations_out[t] = destination;
    destination_slots_out[t] = live ? static_cast<int32_t>(destination) : idle_slot;
    live_out[t] = live;
  }
```
Its comment gains: "A lane that is not live gets idle_destination and idle_slot (0 and 0, or slot_dump and -1 under
spill: GpuResidencyUpdater._idle_destination)." In `direct_gather_destinations_gpu`:
- add `int64_t idle_destination, int64_t idle_slot` after `remap_out`;
- set `const bool wide = W_.unwrap() > kDirectGatherWarp || idle_destination != 0 || idle_slot != 0;`;
- pass `idle_destination, static_cast<int32_t>(idle_slot)` to the wide kernel only.

Python: `direct_gather_destinations(..., remap_out, idle_destination: int = 0, idle_slot: int = 0)`. Pass
`int(idle_destination), int(idle_slot)` after `remap_out` in `.run(...)`.

- [ ] **Step 6: Allocator floor and staging width**

`expert_hot_cache.py`, before `floors = {`:
```python
        # Spill: only the victim lanes need VRAM slots (GpuResidencyUpdater._init_insert_direct's floor).
        victim_lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
```
Change the floor expression to
`2 * (min(miss_lanes[layer_id], victim_lanes) if victim_lanes else miss_lanes[layer_id]) if direct else 0`, and
"(twice its graph-gather miss lanes)" in the refusal text to "(twice its graph-gather victim lanes)".

`exl3_ram_miss.py`, in `Exl3RamMissService` after `plan_gather_width`:
```python
    def plan_staging_width(self, rows: int) -> None:
        """Plan the staging slots a row reserves under spill: the victim lanes, since only live misses stage (a forced
        miss is read into a RAM victim). Only valid before the service starts; the widest plan wins."""
        if self.host is not None:
            raise RuntimeError("exl3 RAM miss: the staging width was planned after the service started")
        planned = max(1, int(rows))
        self._staging_planned = planned if self._staging_planned is None else max(self._staging_planned, planned)
```
Initialize `self._staging_planned = None` in `__init__`, next to `self._gather_planned`. `staging_width` returns
`self._staging_planned or self._gather_planned or self.resolved_lanes()`, and its docstring gains "the planned victim
lanes under spill,".

`exl3_expert_format.py` `plan_graph_gather`, inside the `if isinstance(...)` branch after `plan_gather_width(rows)`:
```python
            victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
            if victims and victims < rows:
                Exl3RamMissService.get().plan_staging_width(victims)
```
Import `envs` there if the module does not already.

- [ ] **Step 7: Run and pass**

- GPU lock: the whole `test/registered/unit/layers/moe/test_expert_residency_gpu.py`, plus
  `test/manual/dsv41/test_direct_gather_wide_gpu.py`, `test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py`,
  `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py` and `test/manual/dsv41/test_bs1_build_digest.py`.
- CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`.
Expected: `EXIT=0`, and the digest test still `BS1 build unchanged`.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/expert_residency_gpu.py \
  python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh \
  python/sglang/kernels/ops/moe/expert_residency_direct_gather.py python/sglang/srt/layers/moe/expert_hot_cache.py \
  python/sglang/srt/layers/moe/exl3_expert_format.py python/sglang/srt/layers/moe/exl3_ram_miss.py \
  test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py \
  test/manual/dsv41/test_direct_gather_wide_gpu.py
git commit -m "feat(moe-residency): SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES gives victims to the first V of a lane-per-route record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 9: The host reads a forced CPU miss into a RAM victim

A forced NVMe miss arrives with lane slot −1. The host takes a RAM victim from the record's node range
(`take_victim_locked`) **before** the staging misses take theirs. It reads the row there, hands that slot to the CPU
(a part-1 job, as any CPU miss), and maps the expert in the tier and the delta, so it stays cached. No staging slot is
used, no pinned RAM is added, and nothing overflows.

**Why a victim always exists.** A node's range of a layer holds:
- its staging slots S (state `kStaging`, never victims);
- the record's routed experts that are resident (`wanted`, never victims);
- VRAM-hot experts (`tier.hot`, never victims);
- the rest, which are victims.

Let D ≤ 36 be the distinct routes and h the forced misses placed on the node. At most D − h routes are resident, and at
most H (the layer's VRAM capacity) are hot. The victims number at least `(hi − lo) − S − (D − h) − H`, which is ≥ h
whenever `hi − lo ≥ S + D + H`. Task 10 checks `hi − lo ≥ staging + lanes + VRAM capacity` for every layer and node at
start-up, and refuses the launch otherwise. Under the recipe that is 8 + 36 + ~24 = 68 against ~80 slots per node per
layer: 161-162 tier rows per layer from 80 GiB of 13,315,584-byte rows, split 1:1 by `split_rows`. Victims for forced
misses are taken first, so the bound covers them. Staging misses that find no victim keep today's behaviour: read,
not cached. A missing victim for a forced miss is therefore a broken invariant: `fail_record`.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`classify_lanes_locked`, `reserve_victims_locked`, `serve_record`)
- Test: `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`

**Interfaces:**
- Consumes: Task 4's forced `MISS_CPU` lanes with slot −1; Task 2's 40-lane wire.
- Produces: a record may carry `kKindMissCpu` lanes with slot −1. The host reads each into a RAM victim of its home
  node's range, computes it there, and maps it in the tier and the record's delta.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_cpu_lane_order_cuda.py`)

```python
def test_36_forced_cpu_misses_on_one_node_are_read_into_ram_victims(tmp_path):
    """Review Focus 2 at the record's full width: 36 cold experts on one node, every lane forced (no VRAM victim:
    MISS_CPU, slot -1). The host reads each into a RAM victim of the row's own tier, the CPU computes it there (part 1),
    and the tier maps it afterwards. No staging slot is touched, nothing overflows, the device's count stands.
    Mutations: read a forced miss into a staging slot (the 7th of the 6 collides); skip the insert (the experts are not
    resident afterwards)."""
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

    row, lanes, experts = 0, 40, list(range(36))
    c = Chain(tmp_path, copy_engine=True, start=False, lanes=lanes, top_k=36, dst_rows=36, capacity=48)
    try:
        x_rows = torch.zeros((LAYERS, 2 * HIDDEN), dtype=torch.uint8).pin_memory()
        out_rows = torch.zeros((LAYERS, 2, HIDDEN), dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(c.host.test_kernel_address(zero=True), [0] * (lanes + 1), cores, x_rows, out_rows,
                                  threads=2)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        c.host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
        c.host.arm_copy_engine()
        backend, plan, dev = c.backends[row], c.plans[row], c.dev
        with paused(c.host):
            staging_before = [s for s, (state, e, _) in enumerate(c.host.slot_info(row)) if state == STAGING_STATE]
        c.plan(experts, row)
        plan.slots[:36] = -1  # every lane forced
        backend._stage_planned(plan)
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        x = torch.randn(1, HIDDEN, device="cuda").half()
        dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True,
                 cpu_input=(x, torch.ones(36, device="cuda")), spill=(flag, overflows))
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
        torch.cuda.synchronize()
        assert c.handled()
        assert (int(plan.count[0]), int(flag.item())) == (36, 0)
        assert c.kinds(36) == [int(LaneKind.MISS_CPU)] * 36
        assert dev.cpu_lanes.tolist() == [-1, PART_MISSES, 0xF]
        assert set(experts) <= c.resident(row), "every forced miss is cached in the RAM victim it was read into"
        with paused(c.host):
            info = c.host.slot_info(row)
            slot_of = {e: s for s, (state, e, _) in enumerate(info) if e >= 0}
            staging_after = [s for s, (state, e, _) in enumerate(info) if state == STAGING_STATE]
        assert staging_after == staging_before, "no staging slot was used"
        computed = sorted(s for call in c.host.test_kernel_calls() for s in call["slots"])
        assert computed == sorted(slot_of[e] for e in experts)
    finally:
        c.close()
```
Define `STAGING_STATE = 3  # slot_info's state of a staging slot (tier_protocol.h: kFree 0, kReady 2, kStaging 3)`
next to `READY_STATE` in the test file. `READY_STATE, FREE_STATE = 3, 0` there are the residency updater's codes, not
the tier's. The `Chain` default reserves 6 staging slots and 48 slots, so `48 − 6 = 42 ≥ 36` victims exist with
nothing hot.

- [ ] **Step 2: Run and see it fail**

GPU lock: `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py -k forced_cpu_misses`. Expected: a fail-stop
`... (expert 0, slot -1) is not a staging slot` from `classify_lanes_locked`.

- [ ] **Step 3: Implement** (`ram_tier.h`)

In `classify_lanes_locked`, replace the head of the miss branch:
```cpp
      if (is_miss(lane.kind)) {
        if (!listed(own.staging, lane.slot) || tier.state[lane.slot] != kStaging) fail(" is not a staging slot");
        if (listed(plan->slots, static_cast<int64_t>(lane.slot))) fail(" shares its staging slot with another miss");
```
with
```cpp
      if (is_miss(lane.kind)) {
        if (lane.slot < 0) {
          // A forced CPU miss (spill): no staging slot; reserve_victims_locked reads it into a RAM victim.
          if (lane.kind != Wire::kKindMissCpu) fail(" is a GPU miss without a staging slot");
        } else {
          if (!listed(own.staging, lane.slot) || tier.state[lane.slot] != kStaging) fail(" is not a staging slot");
          if (listed(plan->slots, static_cast<int64_t>(lane.slot))) fail(" shares its staging slot with another miss");
        }
```
The rest of the branch is unchanged (the tier check, `late_cpu`, the pushes).

Change `reserve_victims_locked` to take `RecordPlan& plan` (not `const`), and put this loop at its start, before the
existing one over `plan.missing`:
```cpp
    // Forced CPU misses first (spill, slot -1): each is read straight into a victim, which it takes over, so it is
    // cached with no staging slot. The start-up capacity check (Exl3RamMissService.attach) leaves every node range a
    // victim for each, so none missing is a broken invariant, not a skipped insert.
    bool placed[Wire::kLanes] = {};
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      if (plan.slots[i] >= 0) continue;
      int32_t old = -1;
      const int64_t victim = take_victim_locked(group, request.row, plan.wanted, &old);
      if (victim < 0) fail_record(request, "a forced CPU miss found no RAM victim on its node");
      if (old >= 0) {
        part.entries[part.count][0] = old;
        part.entries[part.count][1] = -1;
        ++part.count;
      }
      part.entries[part.count][0] = plan.missing[i];
      part.entries[part.count][1] = static_cast<int32_t>(victim);
      ++part.count;
      tier.state[victim] = kStaging;  // being read; commit_inserted_locked makes it READY
      plan.slots[i] = victim;
      placed[i] = true;
      inserted[i] = true;
    }
```
Start the existing loop's body with `if (placed[i]) continue;`. `part` and `own` are declared above both loops, as
they are now. The delta holds at most two entries per miss, 72 at 36 misses, within `kDeltaMaxEntries = 80` at 40
lanes. In `serve_record`, `RecordPlan plan;` is already mutable, so the call is unchanged.
`take_victim_locked` excludes `kStaging`, non-READY, hot and wanted slots, so a slot just taken is never taken again in
the same record.

- [ ] **Step 4: Run and pass**

GPU lock: `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_bs1_build_digest.py`.
CPU: `test/registered/unit/kernels/test_expert_stream_hotpath_golden.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_tier.py`.

Expected: `EXIT=0`. The digest test compares device kernels only (`--permanent`), and those are untouched here. The
host's BS1 behaviour stays pinned by `test_expert_stream_hotpath_golden.py`, which must pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py
git commit -m "feat(exl3-ram-miss): a forced CPU miss is read into a RAM victim of its node and cached, never staged

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 10: The service sizes CPU rows for the verify, wires the spill words, guarantees room for forced misses, and caps calibration

**Files:**
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py` (`CpuExpertService.__init__`/`.calibrate`, `CpuExpertGroups.__init__`)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/policy.py` (`capped_split`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`ensure_started`, `_start_cpu_experts`, `attach`, `spill_room_shortfall`, `Exl3RamMissRowBackend.__init__`/`.post`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h`, `host/ram_tier.h` (`calibrate_cpu_split`), `host/ffi_exports.h`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamHost.calibrate_cpu_split`)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (CPU expert section)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`, `test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`

**Interfaces:**
- Consumes: Tasks 2-9.
- Produces:
  - `CpuExpertService(..., tokens: int = 1, calibration_lanes: Optional[int] = None)` and
    `CpuExpertGroups(..., tokens=1, calibration_lanes=None)`;
  - `Exl3RamMissRowBackend.spill`, `None` or `(overflow_flag, gather_overflow[row : row + 1])`;
  - `attach` accepts CPU experts at a verify gather. Under spill it refuses a layer whose node ranges cannot hold
    `staging + lanes + VRAM capacity` slots.
  - `exl3_ram_miss.spill_room_shortfall(ranges, staging, lanes, hot) -> list[tuple[int, int, int]]`;
  - `ExpertStreamHost.calibrate_cpu_split(..., lanes: Optional[int] = None)`;
  - `policy.capped_split(grid, width, configured) -> list[int]`.

The calibration cap: the split is indexed by the eligible unforced lanes, which under spill are the live lanes, at most
V. A 40-lane calibration would time 860 cells, and on the GPU it would need 40 experts of scratch (40 × 13,315,584 B ≈
508 MiB of VRAM at arming). Calibrating to V = 8 times the 60 cells of today's BS1 grid in a 106 MiB scratch, and keeps
the configured split above V. Those entries are never indexed.

- [ ] **Step 1: Write the failing tests**

In `test_exl3_ram_miss_attach_lanes.py`, delete `test_cpu_experts_refuse_a_miss_width_below_the_routes` and add:
```python
def _spill_updater():
    updater = DirectUpdaterStandIn(LAYERS, CAPACITY, EXPERTS)
    updater.miss_rows, updater.victim_lanes = 36, 8
    updater.overflow_flag = torch.zeros(1, dtype=torch.int32)
    updater.gather_overflow = torch.zeros(LAYERS, dtype=torch.int64)
    return updater


def _attach_verify(service, streamer, updater):
    streamer._graph_pinned_tier = True
    streamer.hot_cache = SimpleNamespace(device="cpu", capacity=CAPACITY)
    streamer.graph_gather_rows, streamer.graph_miss_lanes = 36, 0  # a lane per route
    streamer.residency_row = 1
    streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
    service.attach(SimpleNamespace(register_fail_stop_check=lambda check: None, gpu_residency=updater), streamer)


def test_cpu_experts_attach_a_verify_gather_and_wire_its_spill_words(tiers, monkeypatch):
    """Six tokens of top-6 with a lane per route (36 on a 40-lane wire), 8 of them victim lanes: the layer attaches with
    CPU experts on, and its backend posts with DIRECT's overflow flag and its own row of the overflow counter (a view:
    the post's increment is the updater's). The room check is the next test's."""
    service, streamers = tiers
    service.plan_gather_width(36)
    service.ensure_started()
    monkeypatch.setattr(module, "Exl3RamMissRowBackend", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(service, "_check_spill_room", lambda row, streamer, width: None)
    service.cpu_experts = SimpleNamespace(attach_device=lambda device_side: None)
    updater = _spill_updater()
    try:
        _attach_verify(service, streamers[0], updater)
    finally:
        service.cpu_experts = None
    flag, counter = streamers[0].row_backend.spill
    assert flag is updater.overflow_flag
    counter.add_(1)
    assert updater.gather_overflow.tolist() == [0, 1]


def test_spill_refuses_a_tier_without_a_victim_for_every_forced_miss(tiers, monkeypatch):
    """Review Focus 2 at start-up: a forced CPU miss is read into a RAM victim, so every node range of the layer must hold
    staging + 36 lanes + the VRAM-hot slots. These 3-slot tiers cannot: refused at attach, not fail-stopped mid-verify."""
    service, streamers = tiers
    service.plan_gather_width(36)
    service.ensure_started()
    monkeypatch.setattr(module, "Exl3RamMissRowBackend", lambda *args, **kwargs: SimpleNamespace())
    service.cpu_experts = SimpleNamespace(attach_device=lambda device_side: None)
    try:
        with pytest.raises(ValueError, match="reads every forced CPU miss into a RAM victim"):
            _attach_verify(service, streamers[0], _spill_updater())
    finally:
        service.cpu_experts = None


@pytest.mark.parametrize(
    "ranges, short",
    [
        ([(0, 80), (80, 161)], []),  # the recipe: 80 and 81 slots, 8 + 36 + 24 = 68 needed
        ([(0, 60), (60, 161)], [(0, 60, 68)]),
        ([(0, 161)], []),  # one node
    ],
)
def test_spill_room_is_staging_plus_lanes_plus_hot_per_node(ranges, short):
    assert module.spill_room_shortfall(ranges, staging=8, lanes=36, hot=24) == short


def test_a_captured_cpu_expert_gather_posts_its_spill_words(monkeypatch):
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    assert backend.spill is None
    backend.spill = (torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64))
    backend.post(0, _CapturedPlan())
    assert side.calls[0][1]["spill"] is backend.spill


def test_cpu_rows_hold_the_verify_tokens():
    """A 6-token verify's service rows: x rows of cpu_row_bytes(hidden, 6, lanes), out rows [rows, 2 * nodes, 6,
    hidden]; one token keeps today's shapes."""
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertService

    cores = sorted(os.sched_getaffinity(0))[:2]
    for tokens, out_shape in ((1, (2, 2, 64)), (6, (2, 2, 6, 64))):
        host = SimpleNamespace(
            wire=lease.wire_layout(40), nodes=1, enable_cpu_experts=lambda *a, **k: None, cpu_stats=lambda group: {},
        )
        trait = SimpleNamespace(check_environment=lambda: None, kernel_address=lambda: 1, name="t")
        service = CpuExpertService(
            host, trait, {0: {}, 1: {}}, hidden=64, cores=cores, threads=2, split=[0] * 41, pin=False, tokens=tokens,
        )
        assert tuple(service.out_rows.shape) == out_shape
        assert service.x_rows.shape[1] == lease.cpu_row_bytes(64, tokens, 40)
```
Add `import os` to the file's imports if missing.

In `test_exl3_cpu_split_calibration.py`, add:
```python
def test_a_capped_calibration_times_only_its_lanes(tmp_path):
    """Spill caps calibration at the victim lanes (the split is only indexed below them): a 16-lane host told 4 lanes
    times the 4-lane cells, leaves every other cell 0, and needs 4 experts of scratch."""
    _, host, _, _keep = _host(tmp_path, capacity=20, lanes=16)
    jobs_before = host.cpu_stats()["jobs"]
    scratch = torch.zeros(4 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch, lanes=4)
    assert tuple(grid.shape) == (18, 17)
    assert all(grid[0, k] > 0 for k in range(1, 5)) and not any(grid[0, 5:])
    for n in range(1, 17):
        row = grid[1 + n]
        assert all(row[k] > 0 for k in range(n + 1)) if n <= 4 else not any(row), n
    assert host.cpu_stats()["jobs"] - jobs_before == (4 + 10) * 2


def test_the_capped_split_keeps_the_configured_entries_above_the_cap():
    from sglang.srt.layers.moe.cpu_experts.policy import capped_split, split_from_grid

    grid = [[0.0] * 17 for _ in range(18)]
    for n in range(1, 5):
        for k in range(n + 1):
            grid[1 + n][k] = 10.0 - k  # more CPU lanes are faster: split[n] = n
    configured = list(range(100, 117))
    assert capped_split(grid, 4, configured) == [0, 1, 2, 3, 4] + configured[5:]
    assert capped_split(grid, 4, configured)[:5] == split_from_grid([r[:5] for r in grid[:6]])
```

- [ ] **Step 2: Run and see them fail**

CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -k "spill or verify_tokens or verify_gather"` and
`test/registered/unit/kernels/test_exl3_cpu_split_calibration.py -k capped`. Expected: failures on `spill` (no
attribute), `tokens` (unexpected keyword), `spill_room_shortfall`, `lanes` (unexpected keyword) and `capped_split`.

- [ ] **Step 3: Implement the service rows and the spill words**

`service.py`:
- `CpuExpertService.__init__` gains `tokens: int = 1, calibration_lanes: Optional[int] = None` after `shared`, and
  stores `self.calibration_lanes = min(int(calibration_lanes or self.lanes), self.lanes)`.
- Replace the row allocation in the `else:` branch:
  ```python
              # Row layout is in the class doc; pinned so the device reaches them by UVA. A verify's rows hold its tokens
              # and the token table (cpu_token_table.h).
              self.x_rows = torch.zeros((rows, cpu_row_bytes(self.hidden, tokens, self.lanes)), dtype=torch.uint8)
              shape = (2 * host.nodes, self.hidden) if tokens == 1 else (2 * host.nodes, tokens, self.hidden)
              self.out_rows = torch.zeros((rows, *shape), dtype=torch.float32)
  ```
  Import `cpu_row_bytes` from `sglang.kernels.ops.moe.expert_lease_block`.
- Class doc: after "Each output row has two parts, ..." add "A verify's rows (``tokens`` > 1) hold each token's input
  and output, and the post's token table after the inputs. Under spill, calibration stops at ``calibration_lanes``
  (the victim lanes), the most eligible lanes a split entry is ever read for."
- `CpuExpertGroups.__init__` gains `tokens: int = 1, calibration_lanes: Optional[int] = None` and passes both to each
  `CpuExpertService`.

`exl3_ram_miss.py`:
- `_start_cpu_experts`: before `return CpuExpertGroups(`:
  ```python
          # A DSpark verify gathers tokens x top_k routes a layer; the CPU rows hold that many tokens.
          tokens = max((s.graph_gather_rows // s.layer.top_k for s in streamers.values() if s.graph_gather_rows), default=1)
          # Spill: a split entry is only read for the live lanes, at most the victim lanes.
          victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get() or None
  ```
  Pass `tokens=tokens, calibration_lanes=victims`.
- `ensure_started`: after `node_ranges = group_ranges(...)` (or `None` at one node), store
  `self._node_ranges = node_ranges`. Initialize `self._node_ranges = None` in `__init__`.
- `attach`: delete the block
  ```python
        if self.cpu_experts is not None and width < streamer.graph_gather_rows:
            raise ValueError(
                f"exl3 RAM miss: CPU experts serve one token; ..."
            )
  ```
- The staging-warning line `want = max(1, streamer.graph_miss_width)` becomes:
  ```python
        victims = getattr(manager.gpu_residency, "victim_lanes", None)
        want = max(1, min(streamer.graph_miss_width, victims) if victims else streamer.graph_miss_width)
  ```
  Its comment gains "; under spill, the victim lanes (only live misses stage)".
- Right after `streamer.row_backend = Exl3RamMissRowBackend(...)`:
  ```python
        updater = manager.gpu_residency
        if self.cpu_experts is not None and getattr(updater, "victim_lanes", width) < width:
            # Spill: forced CPU misses land in RAM victims (RamTier::reserve_victims_locked), so the layer must have
            # room for them; the post flags this layer itself only before the copy engine arms.
            self._check_spill_room(row, streamer, width)
            residency_row = streamer.residency_row
            streamer.row_backend.spill = (
                updater.overflow_flag,
                updater.gather_overflow[residency_row : residency_row + 1],
            )
  ```
  `width` here is the `streamer.graph_miss_width` read earlier in `attach`, and `row` the service row computed there.
- Add the module-level function and the method:
  ```python
  def spill_room_shortfall(ranges, *, staging: int, lanes: int, hot: int) -> list[tuple[int, int, int]]:
      """The node ranges of a layer that cannot give every forced CPU miss a RAM victim, as (group, slots, needed).

      A forced miss is read into a victim of its node's range: neither a staging slot (`staging`), nor an expert the
      record routes (at most `lanes` less the forced misses themselves), nor a VRAM-hot expert (at most `hot`). So a
      range of at least staging + lanes + hot slots always has one (plan 2026-10-06, Task 9)."""
      need = staging + lanes + hot
      return [(g, hi - lo, need) for g, (lo, hi) in enumerate(ranges) if hi - lo < need]
  ```
  ```python
      def _check_spill_room(self, row: int, streamer, width: int) -> None:
          """Refuse a layer whose node ranges cannot place every forced CPU miss (spill_room_shortfall)."""
          capacity = int(self.host.tables.capacity[row])
          ranges = self._node_ranges[row] if self._node_ranges is not None else [(0, capacity)]
          short = spill_room_shortfall(
              ranges, staging=self.staging_for(capacity), lanes=width, hot=int(streamer.hot_cache.capacity)
          )
          if short:
              raise ValueError(
                  f"exl3 RAM miss: spill reads every forced CPU miss into a RAM victim; layer {streamer.layer_id}'s "
                  f"node ranges (group, slots, needed) {short} are too small: raise SGLANG_MOE_PINNED_HOST_NUMA_MB or "
                  "lower SGLANG_MOE_HOT_GPU_MB"
              )
  ```
  If `ensure_started` builds the per-row ranges under another name than `node_ranges`, bind that list. It is the one
  passed to `ExpertStreamHost(..., node_ranges=...)`.
- `Exl3RamMissRowBackend.__init__`: after `self.cpu_input = None` add
  `self.spill = None  # DIRECT's (overflow flag, this layer's counter) under spill; set by Exl3RamMissService.attach`.
- `Exl3RamMissRowBackend.post`: add `spill=self.spill,` to `side.post(...)`.

- [ ] **Step 4: Implement the calibration cap**

`split_calibration.h`:
- add `int lanes = kCalibLanes;  // the most lanes measured, 1..kCalibLanes (spill: the victim lanes)` to
  `CalibrationSetup`;
- in `calibrate_split`, the three loops' bound `kCalibLanes` becomes `s.lanes`. The grid keeps its
  `kCalibRows × kCalibCols` shape, with the unmeasured cells 0.

`ram_tier.h` `calibrate_cpu_split`:
- add `int64_t lanes,` after `reps`;
- after the `reps` check add
  `if (lanes < 1 || lanes > kCalibLanes) throw std::runtime_error(prefix + "lanes must be 1.." + std::to_string(kCalibLanes));`;
- the slot check and the scratch `need` use `lanes` in place of `kCalibLanes`;
- set `s.lanes = static_cast<int>(lanes);`.

`ffi_exports.h` `calibrate_cpu_split`: add `int64_t lanes,` after `int64_t reps,` and pass `lanes` after `reps`.
`ExpertStreamHost.calibrate_cpu_split`: add `lanes: Optional[int] = None`. Pass `int(lanes or self.wire.lanes)` after
`int(reps),`. Docstring: "``lanes`` caps the lanes measured (default the wire's); cells past it stay 0, and
``scratch`` needs that many experts."

`policy.py`:
```python
def capped_split(grid, width: int, configured) -> list[int]:
    """The split from a grid measured up to `width` lanes: split_from_grid's entries 0..width, then the configured
    entries above, which a split capped at the live lanes never reads."""
    return split_from_grid([row[: width + 1] for row in grid[: width + 2]]) + list(configured[width + 1 :])
```
`CpuExpertService.calibrate`:
- use `width = self.calibration_lanes` in place of `self.lanes` for `calibration_row(..., width)` and the scratch size
  (`width * expert_bytes`);
- pass `lanes=width` to `calibrate_cpu_split`;
- set `split = capped_split(grid, width, self.split)`. The `format_calibration` call takes the sliced grid
  `[row[: width + 1] for row in grid[: width + 2]]`.

- [ ] **Step 5: LEASE_PROTOCOL.md**

At the end of the CPU expert thread section, add:
```markdown
**A verify's CPU lanes (plan 2026-10-06-dsv41-dspark-both-cpu-experts).** A verify gathers with a lane per route (36
at 6 tokens of top-6, on a 40-lane wire whose lane masks are u64). A row's pinned input holds the verify's tokens
(`cpu_tokens_max`, 6 at `speculative_dspark_block_size=5`), then a token table the post writes for every lane: the token
count, a mask of the tokens that route the lane's expert, and each such token's weight (`cpu_token_table.h`). A record's
CPU job runs one forward of the record's tokens from it, and each token's partial is its own `[hidden]` row of the part,
which the route tables add to that token. The record's summed lane weight still serves one-token rows.

**Spill.** With `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` (V) below the lanes, DIRECT gives VRAM victims and
staging slots to the first V lanes only and marks the rest with destination slot -1. The post makes each marked lane a
CPU lane, outside the split: a RAM hit runs from its RAM slot, and an NVMe miss gets slot -1. The host reads each such
miss into a RAM victim of its node's range, taken before the staging misses take theirs. The CPU computes it there,
and the tier keeps it (the record's delta maps it). Every node range is checked at attach to hold staging + lanes +
VRAM-hot slots, so a victim always exists, and a live miss always has its staging slot (live <= V = staging per node).
Both are asserted (fail_record, __trap), never handled. The post overflows only when forced lanes cannot be CPU lanes
at all: before the copy engine arms (or before a row's CPU layer is registered). Then it serves the live prefix,
writes that count, and sets DIRECT's overflow flag and the layer's counter, and the DSpark worker re-runs the verify
eagerly. The draft (the second client) is untouched: separate channel, areas, thread and cores, on the same stream
strictly before the verify.
```

- [ ] **Step 6: Run and pass**

CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_service.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`.
GPU lock: `test/manual/dsv41/test_cpu_split_calibration_cuda.py`. Expected: `EXIT=0`. A registered test that calls
`expert_stream_calibrate_cpu_split` directly with the old arity gets `lanes` added after `reps`. List each such file in
the commit.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/service.py python/sglang/srt/layers/moe/cpu_experts/policy.py \
  python/sglang/srt/layers/moe/exl3_ram_miss.py \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py analysis/dsv41-drive/LEASE_PROTOCOL.md \
  test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py
git commit -m "feat(exl3-cpu-experts): verify-sized CPU rows, spill words, room for every forced miss, calibration capped at the victim lanes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 11: End to end on the GPU: a captured 6-token verify with CPU experts and spill

This task runs the D2-2 rig (`test_exl3_verify_miss_lanes_gpu.py`) with CPU experts on: the real EXL3 lease chain, the
real optimized CPU kernel, a captured verify, and a lane per route on a 40-lane wire. It is the end-to-end proof of
the owner's goal: in steady state no verify overflows. Two cases matter:
- 36 distinct experts (Review Focus 5);
- every non-victim lane an NVMe miss on one node (Review Focus 2).
Only the replay before the copy engine arms overflows (Review Focus 1).

**Files:**
- Create: `test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py`

**Interfaces:**
- Consumes:
  - everything above;
  - `Exl3MoEMethod._apply_graph`, `_apply_streamed`;
  - `ExpertHotCacheManager.from_model(..., graph_gather_miss_lanes=0)`;
  - `manager.take_verify_overflow()`, `manager.suspend_graph_gather()`;
  - `exl3_ram_miss.COPY_ENGINE_ARM_DECODES`.

- [ ] **Step 1: Write the test**

```python
"""A captured 6-token verify through the real EXL3 lease chain with CPU experts and spill (GPU, real CPU kernel).

Plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 11. A lane per route (36 on a 40-lane wire), two victim lanes.
  - Before the copy engine arms, forced lanes cannot be the CPU's: the replay overflows (Review Focus 1, the one
    overflow left) and the eager re-run is exact.
  - Armed, 36 distinct RAM-resident experts (Review Focus 5) are served with no overflow: two on the GPU, the rest on
    the CPU, every token within the CPU kernel's bar of the fp32 reference.
  - Armed, 36 distinct cold experts, all NVMe misses on the one node (Review Focus 2) are served with no overflow:
    the live misses stage, the forced ones are read into RAM victims and computed there, and stay cached.
Run on divix01 from the pushed worktree, holding rowimg-disk.lock then cc-gpu.lock, on the recipe's server cores (the
CPU expert team derives node 0's 6-15):
  CUDA_MODULE_LOADING=EAGER SGLANG_DSV41_CPU_EXPERTS=1 EXL3_MOE_CPU_PIN=0 SGLANG_EXL3_SRC=... SGLANG_EXL3_CPU_CXX=...
  taskset -c 0-5,36-41 python -m pytest -q test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_ram_miss_graph_gpu import HIDDEN, INTER, _reference, _rel, _source_rows  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC") and os.environ.get("SGLANG_DSV41_CPU_EXPERTS") == "1"),
    reason="needs a GPU, SGLANG_EXL3_SRC and SGLANG_DSV41_CPU_EXPERTS=1 (the optimized CPU build)",
)

ACT_LIMIT = 10.0
TOP_K, TOKENS, VICTIMS, EXPERTS = 6, 6, 2, 128
LANES = TOKENS * TOP_K  # 36: a lane per route
HOT = 8  # VRAM slots per layer; the DIRECT floor is 2 * VICTIMS
TIER = 64  # pinned rows: >= staging (2) + 36 lanes + HOT (8) = 46, Task 10's room check
CPU_BOUND = 2e-2  # the CPU kernel's own bar (test_exl3_moe_split_parity_cuda.py)


def _distinct(experts):
    """Six tokens of top-6 over 36 distinct experts: token t routes experts[6t .. 6t + 5]."""
    return [experts[TOP_K * t : TOP_K * (t + 1)] for t in range(TOKENS)]


def test_a_verify_spills_every_victimless_lane_to_the_cpu_and_never_overflows_once_armed(tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.moe.ram_slot_map import LaneKind
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import service_row_images

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path, num_experts=EXPERTS)
    source_cuda = {name: rows.cuda() for name, rows in source.items()}
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    service = service_module.Exl3RamMissService.get()
    try:
        with (
            service_row_images(tmp_path),
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"),
            envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(True),
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(True),
            envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.override(True),
            envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.override(False),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(VICTIMS),
        ):
            model, layer = torch.nn.Module(), torch.nn.Module()
            layer.layer_id, layer.top_k = 0, TOP_K
            fmt = Exl3ExpertFormat(layout, 0, source_root=str(tmp_path))
            fmt.max_gather_rows = 8
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, TIER, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=HOT * streamer.bytes_per_expert, seed_path=None, dynamic=True,
                update_prefill_tokens=16, min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=TOKENS, graph_gather_miss_lanes=0, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            updater = manager.gpu_residency
            assert streamer.graph_miss_width == LANES and (updater.miss_rows, updater.victim_lanes) == (LANES, VICTIMS)
            assert service.staging_width() == VICTIMS

            generator = torch.Generator(device="cpu").manual_seed(41)
            x = (torch.randn((TOKENS, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.softmax(torch.randn((TOKENS, TOP_K), generator=generator), 1).to("cuda")
            ids = torch.tensor([list(range(TOP_K))] * TOKENS, device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            manager.discard_graph_capture_routes()
            assert service.lanes == 40 and service.cpu_experts is not None

            def outsiders():
                mapping = updater.mapping[0, :EXPERTS].cpu().tolist()
                return [expert for expert, slot in enumerate(mapping) if slot < 0]

            def in_ram(experts):
                """Load experts into the pinned tier through the eager path (the service maps them on the device)."""
                with manager.suspend_graph_gather():
                    for start in range(0, len(experts), TOP_K):
                        chunk = experts[start : start + TOP_K]
                        Exl3MoEMethod._apply_streamed(
                            layer, streamer, x[:1], weights[:1], torch.tensor([chunk], device="cuda", dtype=torch.int32),
                            ACT_LIMIT,
                        )
                torch.cuda.synchronize()

            def replay(routes):
                updater.overflow_flag.zero_()
                ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                service.fail_stop_check()
                assert updater.insertion_truncated[0].item() == 0
                return int(updater.overflow_flag.item())

            def exact(result, routes):
                for t, route in enumerate(routes):
                    ref = _reference(x[t : t + 1], weights[t], torch.tensor(route), source_cuda)
                    assert _rel(result[t : t + 1], ref) <= CPU_BOUND, (t, route)

            def served_without_overflow(routes):
                inserted = updater.gather_insertions[0].item()
                assert replay(routes) == 0, "an armed verify overflowed"
                count = int(streamer._graph_miss_count.item())
                kinds = service.device_side.lane_kind[:count].tolist()
                cpu = sum(k in (int(LaneKind.HIT_CPU), int(LaneKind.MISS_CPU)) for k in kinds)
                live = updater.gather_insertions[0].item() - inserted
                assert count == LANES and live <= VICTIMS and cpu >= LANES - VICTIMS, (count, live, kinds)
                exact(out, routes)

            # Review Focus 1: before the copy engine arms, forced lanes cannot be the CPU's; the re-run is exact.
            warm = outsiders()[:LANES]
            in_ram(warm)
            routes = _distinct(warm)
            assert replay(routes) == 1
            assert manager.take_verify_overflow()
            with manager.suspend_graph_gather():
                eager = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids, ACT_LIMIT)
            torch.cuda.synchronize()
            exact(eager, routes)

            service._copy_decodes = service_module.COPY_ENGINE_ARM_DECODES
            service._arm_copy_engine()
            assert service._copy_armed
            before = updater.gather_overflow[0].item()

            # Review Focus 5: 36 distinct RAM-resident experts, a lane each.
            in_ram(warm)
            served_without_overflow(_distinct(warm))

            # Review Focus 2: 36 distinct cold experts, every one an NVMe miss on the one node. The live ones stage, the
            # forced ones are read into RAM victims, and all stay cached in the tier afterwards.
            cold = [e for e in outsiders() if e not in warm][:LANES]
            assert len(cold) == LANES
            served_without_overflow(_distinct(cold))
            resident = {e for e, s in enumerate(service.host.mapping(0).tolist()) if s >= 0}
            assert len(set(cold) - resident) <= VICTIMS, "the forced misses are cached in their RAM victims"
            assert updater.gather_overflow[0].item() == before, "no armed verify overflowed"
    finally:
        service.shutdown()
```
`service.host.mapping(row)` is the host mirror of a row's expert→RAM-slot map (`ExpertStreamHost.mapping`, used by
`lease_chain_rig.Chain`). Staging misses whose victim the tier skipped may be uncached, hence the `<= VICTIMS`.

- [ ] **Step 2: Run on divix01**

```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; flock 9
PYTHONPATH=$PWD/python CUDA_MODULE_LOADING=EAGER SGLANG_DSV41_CPU_EXPERTS=1 EXL3_MOE_CPU_PIN=0 \
  SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build \
  SGLANG_EXL3_CPU_CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
  CUDA_HOME=/usr/local/cuda-13.2 OMP_NUM_THREADS=8 taskset -c 0-5,36-41 \
  /data/models/slang/.venv/bin/python -m pytest -q -s test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py \
  2>&1 | tail -20; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. If it fails, debug with `superpowers:systematic-debugging` before changing the bar. A
`fail_stop` or `__trap` is a defect in Tasks 2-10. It is never a reason to drop an assertion.

- [ ] **Step 3: Re-run the D2-2 rig and the BS1 graph suites** (same locks, cores 32-63):
`test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_graph_apply_gpu.py test/manual/dsv41/test_bs1_build_digest.py`.
Expected: `EXIT=0`.

- [ ] **Step 4: Commit**

```bash
git add test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py
git commit -m "test(exl3-cpu-experts): a 36-lane verify spills to the CPU end to end and never overflows once armed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 12: The gate admits the target's CPU experts under a graphed DSpark verify

The refusal at `expert_stream_requirements_exl3.py:171-181` dates from `3937a8833a` (2026-09-30), before a verify
could run in the breakable graph. The narrowest change:
- CPU experts still require the breakable decode graph.
- Under speculation they ride `_check_graphed_verify`, which already admits only DSpark, static verify and DIRECT,
  with a remedy that does not point at eager decode.
- Its miss-lane rule follows Task 2's wider wire. `MISS_LANES` may be 1-64, or unset (a lane per route) when
  `VICTIM_LANES` is set. Spill always has a lane per route, so the gate refuses `VICTIM_LANES` with `MISS_LANES` set.
- A non-speculative launch takes exactly today's path.

**Files:**
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Produces: `_check_graphed_verify(cfg, remedy: str = _EAGER_VERIFY_REMEDY)`, `_CPU_EXPERTS_VERIFY_REMEDY`,
  `_check_victim_lanes()`.

- [ ] **Step 1: Update and add tests**

In the parametrize of `test_a_graphed_dspark_verify_needs_its_configuration`:
- the `MISS_LANES` 0 row now matches `"MISS_LANES=1-64"`;
- the `MISS_LANES` 33 row becomes 65, matching `"MISS_LANES=1-64"`;
- the last row is replaced with
  `({}, {**GRAPHED_VERIFY, "SGLANG_DSV41_CPU_EXPERTS": True}, "SGLANG_DSV41_CPU_EXPERTS needs"),`.
Then change the function's tail to:
```python
    if match != "SGLANG_DSV41_CPU_EXPERTS needs":
        assert "--cuda-graph-backend-decode disabled" in str(raised.value)
```
Replace `test_cpu_experts_need_the_breakable_decode_graph_without_speculation` with:
```python
@pytest.mark.parametrize(
    "changes, env",
    [
        ({}, _EAGER_DECODE),  # both phases disabled
        ({"cuda_graph_config": FULL_BS1}, {}),
        ({"speculative_algorithm": "DSPARK"}, _EAGER_DECODE),
    ],
    ids=["disabled", "full", "spec-disabled"],
)
def test_cpu_experts_need_the_breakable_decode_graph(model_dir, changes, env):
    """Refused before the speculative and backend rules; the refusal never suggests disabled decode graphs."""
    with pytest.raises(ValueError, match="pass --cuda-graph-backend-decode breakable") as refused:
        _gate(_launch(model_dir, **changes), **{**CPU_EXPERTS_ENV, **env})
    assert "disabled" not in str(refused.value)


DSPARK_BREAKABLE = dict(speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1)
# The recipe's DSpark mode: a lane per route (MISS_LANES unset), 8 victim lanes.
SPILL = {"SGLANG_MOE_EXPERT_GRAPH_GATHER": True, **DIRECT, "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": 8}


def test_cpu_experts_with_a_graphed_dspark_verify_pass(model_dir):
    _gate(_launch(model_dir, **DSPARK_BREAKABLE), **CPU_EXPERTS_ENV, **GRAPHED_VERIFY)
    _gate(_launch(model_dir, **DSPARK_BREAKABLE), **CPU_EXPERTS_ENV, **SPILL)


@pytest.mark.parametrize(
    "launch, env, match",
    [
        ({"speculative_algorithm": "EAGLE"}, GRAPHED_VERIFY, "graphs the verify of DSpark only"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 0}, "MISS_LANES=1-64"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_RAGGED_VERIFY_MODE": "compact"}, "SGLANG_RAGGED_VERIFY_MODE=static"),
    ],
)
def test_cpu_experts_under_speculation_need_the_graphed_dspark_verify(model_dir, launch, env, match):
    """The graphed verify's own rules, with a remedy for a CPU-experts launch: the verify must be graphed, so the
    eager-decode remedy is never offered."""
    with pytest.raises(ValueError, match=match) as refused:
        _gate(_launch(model_dir, **(DSPARK_BREAKABLE | launch)), **CPU_EXPERTS_ENV, **env)
    assert "--cuda-graph-backend-decode disabled" not in str(refused.value)
    assert "SGLANG_DSV41_CPU_EXPERTS" in str(refused.value)


@pytest.mark.parametrize(
    "env, match",
    [
        ({"SGLANG_DSV41_CPU_EXPERTS": False}, "needs SGLANG_DSV41_CPU_EXPERTS=1"),
        ({"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 36}, "unset SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES"),
    ],
)
def test_victim_lanes_need_cpu_experts_and_a_lane_per_route(model_dir, env, match):
    """Spill's victimless lanes are the CPU's, and its record has a lane per route, so a 6-token verify never
    outnumbers its lanes (Review Focus 5)."""
    with pytest.raises(ValueError, match=match):
        _gate(_launch(model_dir, **DSPARK_BREAKABLE), **{**CPU_EXPERTS_ENV, **SPILL, **env})
```

- [ ] **Step 2: Run and see them fail** — `test/registered/unit/test_expert_stream_requirements_exl3.py`, CPU.
Expected: failures in the new tests, still refused "without speculative decoding".

- [ ] **Step 3: Implement**

```python
_EAGER_VERIFY_REMEDY = "or pass --cuda-graph-backend-decode disabled to run the DSpark verify eagerly"
# SGLANG_DSV41_CPU_EXPERTS computes in the captured graph's copy wait, so its verify cannot go eager.
_CPU_EXPERTS_VERIFY_REMEDY = "SGLANG_DSV41_CPU_EXPERTS serves the DSpark verify only in the decode graph"


def _check_graphed_verify(cfg, remedy: str = _EAGER_VERIFY_REMEDY) -> None:
```
Inside, replace each `{_EAGER_VERIFY_REMEDY}` with `{remedy}`. Replace the lane rule
`and 1 <= lanes <= 32` with
```python
        and (1 <= lanes <= 64 or (lanes == 0 and envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get() > 0))
```
In its message, `f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES=1-32 (got {lanes}): a verify routes more experts than the
32 lanes; {remedy}"` becomes `f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES=1-64, or unset with
SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (got {lanes}): a lane per route needs spill's CPU lanes; {remedy}"`.
Then:
```python
def _check_victim_lanes() -> None:
    """SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (spill): the CPU takes the lanes past the victims, and every route has
    a lane, so a verify's distinct misses never outnumber its lanes."""
    victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
    if not victims:
        return
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        raise ValueError(f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} needs SGLANG_DSV41_CPU_EXPERTS=1")
    lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.get()
    if lanes:
        raise ValueError(
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES gives every route a lane: unset "
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES (got {lanes})"
        )
```
(`victims` below the routes is the updater's check, `GpuResidencyUpdater._init_insert_direct`: the gate does not know the
verify width.)
In `_check`, replace the CPU-experts refusal and the speculative block:
```python
    cpu_experts = envs.SGLANG_DSV41_CPU_EXPERTS.get()
    speculative = getattr(cfg, "speculative_algorithm", None) is not None
    if cpu_experts and graph.decode.backend != Backend.BREAKABLE:
        # Checked before the speculative and backend rules: each would point a CPU-experts launch at the other's
        # decode backend.
        raise ValueError(
            "SGLANG_DSV41_CPU_EXPERTS computes experts inside the captured decode graph's copy wait; "
            "pass --cuda-graph-backend-decode breakable"
        )
    if speculative and graph.decode.backend != Backend.DISABLED:
        _check_graphed_verify(cfg, _CPU_EXPERTS_VERIFY_REMEDY if cpu_experts else _EAGER_VERIFY_REMEDY)
    _check_victim_lanes()
```
Module docstring: replace the speculative bullet with:
```
* Speculative decoding as DSpark: its verify eager (decode graphs disabled), or in the breakable decode graph on DIRECT
  residency at 1-64 miss lanes with a static verify (§33.8). The target's CPU experts serve that graphed verify; with
  SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (and a lane per route: MISS_LANES unset) the lanes past the victims are
  theirs (spill).
```
In `_check_dspark_cpu_experts`, the comment "SGLANG_DSV41_CPU_EXPERTS, which also selects it, is refused under
speculation, so these two defines are the way in" becomes "SGLANG_DSV41_CPU_EXPERTS also selects it, but a draft-only
launch has it off, so these two defines are the way in. The recipe sets them in either case."

- [ ] **Step 4: Run and pass** — the whole file. Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py test/registered/unit/test_expert_stream_requirements_exl3.py
git commit -m "feat(exl3-gate): the target's CPU experts serve a graphed DSpark verify; VICTIM_LANES needs them

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 13: The recipe's DSpark mode and the production switch

Which of the DSpark arms' overrides are still required with a graphed verify, and why (research, 2026-10-06):

| Override in `ab_cpu_draft.COMMON` | Verdict for this mode | Evidence |
|---|---|---|
| `SGLANG_DSV41_CPU_EXPERTS=0` | **Dropped**: the point of this plan | Tasks 2-12 |
| `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0` | **Required** | `layer_major/gate.py:38-50` refuses any speculative algorithm |
| `SGLANG_DSV41_ENABLE_PREFILL_FILLS=0` | **Dropped** | refused only without graph gather (`exl3_expert_format.py:226-231`); `graphed_verify.py` already keeps 1 |
| `SGLANG_MOE_EXPERT_GRAPH_GATHER/GPU_RESIDENCY_UPDATE/INSERT_ON_MISS_STAGE/FUSED_PLAN=0` | **Dropped** | the graphed verify needs them on (`_check_graphed_verify`) |
| `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=0`, `..._ENGRAM_DEVICE_WAIT=0` | **Dropped**, with a fallback | A verify never takes the native captured lookup (`engram.py:124-139`: one token, decode mode); it runs the eager break, and the uring store serves N tokens (`engram_file_table.py:147-163`). Device wait engages only in a one-token decode capture, which a DSpark target never captures. No DSpark run has had them on, so Task 15's smoke checks them, with the fallback named there. |
| `SGLANG_SM120_FLASHMLA_BACKEND=triton` | **Kept** | No refusal: flashinfer handles up to 64 query rows (`flash_mla_sm120.py:212, 317-341`). But every DSpark text result (§33.2, §33.8, §33.9) is triton, and §33.2 names the sm120 attention at 6 rows as a drift suspect. Dropping it is its own A/B. |
| `SGLANG_EXL3_CPU_ACT_RESIDUAL=1`, `SGLANG_EXL3_CPU_ACT_BLOCK=128` | **Kept** | `_check_dspark_cpu_experts` requires them; with CPU experts on they equal the build's own values (`ext.py:76-89`) |

**Files:**
- Modify: `benchmarks/dsv41_baseline/arm_env.py`, `benchmarks/dsv41_baseline/launch_prod.sh`
- Create: `benchmarks/dsv41_baseline/test_dspark_recipe.py`
- Test: `test/registered/unit/layers/moe/test_threading_config.py`

**Interfaces:**
- Produces:
  - `arm_env.DSPARK_DRAFT`, `DSPARK_RESIDENT`, `DSPARK_DRAFT_CORES = "12-15"`, `DSPARK_ARGV`;
  - `dspark_env() -> dict[str, str]`, the overrides on top of `base_env()`;
  - `prod_env() -> dict[str, str]`;
  - `PROD_DSPARK = False`;
  - `ServerArgs.dspark: bool = False`.
  - `ServerArgs.prod()` sets `dspark=PROD_DSPARK`, and `launch_prod.sh` uses `prod_env()`.

- [ ] **Step 1: Write the failing tests**

`benchmarks/dsv41_baseline/test_dspark_recipe.py`:
```python
"""The DSpark mode of the recipe: both CPU-expert clients on, the graphed verify's configuration, and argv the server
parses as DSpark (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 13)."""

import argparse

import arm_env


def test_dspark_env_turns_both_cpu_expert_clients_on_with_spill():
    env = arm_env.arm_env(arm_env.dspark_env())
    assert env["SGLANG_DSV41_CPU_EXPERTS"] == "1"
    assert env["SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"] == "1"
    assert env["SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES"] == "12-15"
    assert "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES" not in env  # a lane per route: 36 on a 40-lane wire
    assert env["SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES"] == "8"
    assert env["SGLANG_RAGGED_VERIFY_MODE"] == "static"
    assert env["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"] == "0"
    assert env["SGLANG_MOE_HOT_GPU_MB"] == "12040"
    assert env["SGLANG_DSV41_ENABLE_PREFILL_FILLS"] == "1"  # the recipe's, kept


def test_prod_is_unchanged_until_the_switch():
    assert arm_env.PROD_DSPARK is False
    assert arm_env.prod_env() == arm_env.base_env()
    assert "--speculative-algorithm" not in arm_env.ServerArgs.prod().argv()


def test_the_dspark_argv_parses_as_dspark_at_block_size_5():
    from sglang.srt.server_args import ServerArgs

    argv = arm_env.ServerArgs(port=1, dspark=True).argv()
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv[3:])
    assert ns.speculative_algorithm == "DSPARK"
    assert ns.speculative_draft_model_path == arm_env.DSPARK_DRAFT
    assert int(ns.speculative_dspark_block_size) == 5
    assert ns.max_running_requests == 1
```
Append to `test_threading_config.py`:
```python
def test_the_dspark_recipe_layout_keeps_both_cpu_expert_clients_apart(divix01):
    """arm_env's DSpark mode: server 0-5,36-41, RAM 17 by name, target THREADS=10, the draft named on 12-15. The draft's
    cores leave every derived role, so node 0's target team is 6-11, node 1's 18-27, copy 16, node 1's RAM thread 35;
    no core or SMT sibling is in two roles. Mutation: leave named draft cores out of `taken` -- the target takes 6-15."""
    config = resolve(divix01, affinity=RECIPE_SERVER, cpu_experts=True, threads=10, spin_core=17, draft=True,
                     draft_cores="12-15")
    assert config.copy_cpus == (16,)
    assert [(p.ram, p.cpu) for p in config.plans] == [(17, tuple(range(6, 12))), (35, tuple(range(18, 28)))]
    assert config.draft_cpus == (12, 13, 14, 15)
    roles = [*config.copy_cpus, *config.draft_cpus] + [c for p in config.plans for c in (p.ram, *p.cpu)]
    physical = [c % 36 for c in roles]
    assert len(set(physical)) == len(physical)
```

- [ ] **Step 2: Run and see them fail**

On divix01:
- `cd benchmarks/dsv41_baseline && PYTHONPATH=$WT/python:. taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q test_dspark_recipe.py`;
- `test/registered/unit/layers/moe/test_threading_config.py -k dspark_recipe`.
Expected: `AttributeError: module 'arm_env' has no attribute 'dspark_env'`. The threading test may already pass:
`ThreadingConfig` needs no change for option (a). It then stands as the guard for that layout.

- [ ] **Step 3: Implement `arm_env.py`**

After `PROD_HOST`:
```python
# DSpark with both CPU-expert clients (plan 2026-10-06-dsv41-dspark-both-cpu-experts). Production serves it only once
# PROD_DSPARK is True (Owner decision 1, the A/B of that plan's Task 15).
PROD_DSPARK = False
DSPARK_DRAFT = f"{CC}/dsv41-dspark-draft"
# Each stage's top-32 draft experts stay on the GPU, the other 96 on the CPU (§33.4).
DSPARK_RESIDENT = f"{CC}/analysis/dsv41-dspark/cpu-draft-routes/resident-top32.json"
# The draft's CPU experts by name: node 0's 12-15, so the target's node-0 team derives 6-11 (Owner decision 2a).
DSPARK_DRAFT_CORES = "12-15"
# gamma = 5 draft tokens, a 6-token verify (speculative_hook.py).
DSPARK_ARGV = (
    "--speculative-algorithm", "DSPARK",
    "--speculative-draft-model-path", DSPARK_DRAFT,
    "--speculative-dspark-block-size", "5",
)


def dspark_env() -> dict[str, str]:
    """The DSpark mode's overrides on base_env: the graphed verify with the target's CPU experts and spill, and the
    draft's CPU experts. Every value's reason is in the plan's Task 13 table."""
    return {
        # The verify in the breakable decode graph with a lane per route (MISS_LANES unset: 36 lanes, a 40-lane wire),
        # the first 8 VRAM victims; the rest are CPU lanes, a miss among them read into a RAM victim (Owner decision 3).
        "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "8",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        # Layer-major prefill refuses speculative decoding (layer_major/gate.py).
        "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0",
        # The draft's CPU experts on their own cores, reading the optimized build (its gate names both defines).
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": "1",
        "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": DSPARK_DRAFT_CORES,
        "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": DSPARK_RESIDENT,
        "SGLANG_EXL3_CPU_ACT_RESIDUAL": "1",
        "SGLANG_EXL3_CPU_ACT_BLOCK": "128",
        # 4040 MiB of the hot cache go to the draft's resident experts, dense weights and KV (Owner decision 4).
        "SGLANG_MOE_HOT_GPU_MB": "12040",
        # Every DSpark text result so far ran triton attention (§33.2, §33.8, §33.9).
        "SGLANG_SM120_FLASHMLA_BACKEND": "triton",
    }


def prod_env() -> dict[str, str]:
    """Production's env: the base recipe, with the DSpark mode once PROD_DSPARK is set."""
    return arm_env(dspark_env()) if PROD_DSPARK else base_env()
```
`ServerArgs`: add the field `dspark: bool = False` with the comment "the DSpark mode's argv (DSPARK_ARGV); its env is
dspark_env()". `prod` returns `cls(port=PROD_PORT, host=PROD_HOST, dspark=PROD_DSPARK)`. At the end of `argv`, before
`return argv`:
```python
        if self.dspark:
            argv += list(DSPARK_ARGV)
```

- [ ] **Step 4: `launch_prod.sh`**

Replace both uses of `arm_env.base_env()`, the DRY_RUN printer and the exec'd launcher, with `arm_env.prod_env()`.
In the header comment, "the arm_env base recipe, unchanged" becomes "arm_env.prod_env(): the base recipe, plus the
DSpark mode when arm_env.PROD_DSPARK is set". The DRY_RUN branch also prints the line
`print("dspark:", arm_env.PROD_DSPARK)`.

- [ ] **Step 5: Run and pass**, plus `DRY_RUN=1 benchmarks/dsv41_baseline/launch_prod.sh` from the worktree on divix01.
It takes no lock and starts nothing. Expected: `dspark: False`, and the env and argv byte-identical to production's
current DRY_RUN output. Diff them and record it.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/dsv41_baseline/arm_env.py benchmarks/dsv41_baseline/launch_prod.sh \
  benchmarks/dsv41_baseline/test_dspark_recipe.py test/registered/unit/layers/moe/test_threading_config.py
git commit -m "feat(dsv41-baseline): the recipe's DSpark mode with both CPU-expert clients; production behind PROD_DSPARK (off)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 14: A/B tooling: accept length from the driver, the text band, the three-arm driver

**Files:**
- Modify: `scripts/expert_prediction/benchmarks/run_capture_sessions.py`
- Create: `scripts/dsv41/dspark_text_band.py`
- Create: `analysis/dsv41-drive/dspark/both_cpu_ab.py`
- Test: `test/registered/unit/scripts/test_dspark_text_band.py`, `test/registered/unit/scripts/test_run_capture_sessions_spec.py`

**Interfaces:**
- Produces:
  - `run_capture_sessions.chunk_spec_details(chunk) -> dict | None`. Each results record gains `spec_tokens_details`,
    `None` without speculation. The request sends `return_spec_tokens_details: true`.
  - `dspark_text_band.compare(base, other, band=1.4) -> dict`.
  - `both_cpu_ab.py OUT [ARM...]`.

- [ ] **Step 1: Write the failing tests**

`test/registered/unit/scripts/test_run_capture_sessions_spec.py`:
```python
"""The session driver records a speculative server's per-request details (accept length's inputs) (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _driver():
    path = os.path.join(ROOT, "scripts", "expert_prediction", "benchmarks", "run_capture_sessions.py")
    spec = importlib.util.spec_from_file_location("run_capture_sessions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sglext_chunk_carries_the_spec_details():
    driver = _driver()
    chunk = {"choices": [], "sglext": {"spec_tokens_details": {"spec_verify_ct": 40, "spec_accept_length": 3.2}}}
    assert driver.chunk_spec_details(chunk) == {"spec_verify_ct": 40, "spec_accept_length": 3.2}
    assert driver.chunk_spec_details({"choices": [{"delta": {}}]}) is None
    assert driver.chunk_spec_details({"choices": [], "sglext": {"cached_tokens_details": {}}}) is None
```
`test/registered/unit/scripts/test_dspark_text_band.py`:
```python
"""The DSpark text bar: a run's text may leave the base run's only at a near-tie of the base (§33.2, §33.9) (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _band():
    spec = importlib.util.spec_from_file_location("dspark_text_band", os.path.join(ROOT, "scripts", "dsv41", "dspark_text_band.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokens(*rows):
    return [{"token": chosen, "top": top} for chosen, top in rows]


def test_identical_text_passes_with_no_flips():
    band = _band()
    run = [{"session_id": "s", "tokens": _tokens(("a", [["a", -0.1], ["b", -2.0]]))}]
    assert band.compare(run, run)["pass"] and band.compare(run, run)["flips"] == []


def test_a_flip_within_the_band_passes_and_one_past_it_or_off_the_top_k_fails():
    band = _band()
    base = [{"session_id": "s", "tokens": _tokens(("a", [["a", -0.1], ["b", -1.2], ["c", -3.0]]))}]
    near = [{"session_id": "s", "tokens": _tokens(("b", [["b", -0.2], ["a", -0.3]]))}]
    far = [{"session_id": "s", "tokens": _tokens(("c", [["c", -0.2], ["a", -0.3]]))}]
    off = [{"session_id": "s", "tokens": _tokens(("z", [["z", -0.2], ["a", -0.3]]))}]
    assert band.compare(base, near)["pass"]  # 1.1 behind the base argmax
    assert not band.compare(base, far)["pass"]  # 2.9 behind
    assert not band.compare(base, off)["pass"]  # not in the base's top-k: unbounded
    assert band.compare(base, near)["flips"][0]["gap"] == 1.1


def test_different_prompts_are_refused():
    band = _band()
    try:
        band.compare([{"session_id": "a", "tokens": []}], [{"session_id": "b", "tokens": []}])
    except ValueError:
        return
    raise AssertionError("ran different prompts")
```

- [ ] **Step 2: Run and see them fail** — CPU. Expected: `AttributeError: ... chunk_spec_details` and `FileNotFoundError`.

- [ ] **Step 3: Implement**

`run_capture_sessions.py`:
- Add above `_stream_chat`:
  ```python
  def chunk_spec_details(chunk: dict):
      """A speculative server's per-request details (spec_verify_ct, spec_accept_length, ...) from its sglext chunk, or
      None (return_spec_tokens_details; a non-speculative server sends none)."""
      return (chunk.get("sglext") or {}).get("spec_tokens_details")
  ```
- In `_stream_chat`'s request body, add `"return_spec_tokens_details": True,`. Initialize `spec_details = None`. In
  the chunk loop, right after `chunk = json.loads(payload)`:
  ```python
              spec_details = chunk_spec_details(chunk) or spec_details
  ```
  Return `"spec_tokens_details": spec_details` in the result dict.
- In the success `record`, add `"spec_tokens_details": result["spec_tokens_details"],`.

`scripts/dsv41/dspark_text_band.py`:
```python
"""The DSpark text bar (DSV41_REFERENCE.md §33.2, §33.9): exact text is the wrong test for a speculative run, because
its verify forward settles near-ties differently from one-token decode. Of two logprob_probe.py outputs of the same
prompts (scripts/expert_prediction/prefetch/logprob_probe.py, --top-logprobs 5), the other run may leave the base run
only where its token is within `band` nats of the base's argmax at the first divergence. 1.4 is §33.2's
verify-vs-decode band; §33.9's largest draft-to-draft divergence was 1.125.

    python scripts/dsv41/dspark_text_band.py BASE.json OTHER.json [--band 1.4]
"""

import argparse
import json

BAND_NATS = 1.4


def compare(base, other, band: float = BAND_NATS) -> dict:
    if [a["session_id"] for a in base] != [b["session_id"] for b in other]:
        raise ValueError("the two probes ran different prompts")
    flips, compared = [], 0
    for a, b in zip(base, other):
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            compared += 1
            if x["token"] == y["token"]:
                continue
            top = dict(x["top"])
            gap = round(x["top"][0][1] - top[y["token"]], 6) if y["token"] in top else float("inf")
            flips.append({"session_id": a["session_id"], "index": index, "gap": gap, "within": gap <= band})
            break
    return {
        "prompts": len(base),
        "compared_tokens": compared,
        "flips": flips,
        "max_gap": max((f["gap"] for f in flips), default=0.0),
        "pass": all(f["within"] for f in flips),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--band", type=float, default=BAND_NATS)
    args = parser.parse_args()
    with open(args.base) as f:
        base = json.load(f)
    with open(args.other) as f:
        other = json.load(f)
    report = compare(base, other, args.band)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
```

`analysis/dsv41-drive/dspark/both_cpu_ab.py`:
```python
"""Server A/B (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 15): today's production recipe against DSpark with
both CPU-expert clients, and DSpark with the draft's only (the target's CPU experts off, §33.9's configuration in a
server) to isolate them.

Per arm: run_arm.sh's timed set over the 8 corpus sessions (ms/token, and accept length from the driver's
spec_tokens_details), then a short-lived server on the same env for logprob_probe.py (8 prompts, 128 tokens, top-5).
The text bar is dspark_text_band.py against prod. Run on divix01 from the pushed worktree holding rowimg-disk.lock
only: run_arm.sh takes cc-gpu.lock itself, and the probe phase takes it here (lock order: disk, then GPU).

    flock /data/models/slang/nvfp4-work/rowimg-disk.lock python analysis/dsv41-drive/dspark/both_cpu_ab.py OUT [ARM ...]
"""

import fcntl
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

PORT = 30017
PROBE_PORT = 30018
SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
DRAFT_ONLY = {
    "SGLANG_DSV41_CPU_EXPERTS": "0",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": "8",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "0",
}
# arm: (overrides on base_env, DSpark argv)
ARMS = {
    "prod": ({}, False),
    "dspark-draft-only": ({**arm_env.dspark_env(), **DRAFT_ONLY}, True),
    "dspark-both": (arm_env.dspark_env(), True),
}


def _overrides(arm: str, out: str) -> dict:
    overrides, _ = ARMS[arm]
    return {**overrides, "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.metrics.jsonl")}


def run_timed(arm: str, out: str) -> int:
    _, dspark = ARMS[arm]
    env = os.environ | {
        "DSV41_RUN_ROOT": out,
        "DSV41_SESSION_INDICES": ",".join(str(i) for i in range(8)),
        "DSV41_EXTRA_SERVER_ARGS": shlex.join(arm_env.DSPARK_ARGV) if dspark else "",
    }
    cmd = [os.path.join(REPO, "benchmarks", "dsv41_baseline", "run_arm.sh"), arm, str(PORT)]
    cmd += [f"{k}={v}" for k, v in _overrides(arm, out).items()]
    print("===", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, cwd=REPO).returncode


def _healthy(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def run_probe(arm: str, out: str) -> int:
    """The arm's server on PROBE_PORT under cc-gpu.lock, for logprob_probe.py only."""
    _, dspark = ARMS[arm]
    env = os.environ | arm_env.arm_env(_overrides(arm, out)) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PROBE_PORT, dspark=dspark).argv()
    with open(arm_env.GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(os.path.join(out, f"{arm}.probe-server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        try:
            if not _healthy(PROBE_PORT, time.monotonic() + arm_env.HEALTH_TIMEOUT_S):
                return 1
            probe = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")
            return subprocess.run(
                ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, probe, "--port", str(PROBE_PORT),
                 "--sessions", SESSIONS, "--prompts", "8", "--max-tokens", "128", "--top-logprobs", "5",
                 "--out", os.path.join(out, f"{arm}.probe.json")], cwd=REPO,
            ).returncode
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def _results(out: str, arm: str) -> list[dict]:
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    with open(os.path.join(out, "servers", arm, runs[-1], "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(out: str) -> dict:
    sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
    import dspark_text_band

    summary = {}
    for arm in ARMS:
        try:
            rows = _results(out, arm)
        except FileNotFoundError:
            continue
        ms = [1000.0 / r["decode_tokens_per_sec"] for r in rows if r.get("decode_tokens_per_sec")]
        spec = [r.get("spec_tokens_details") or {} for r in rows]
        verifies = sum(s.get("spec_verify_ct", 0) for s in spec)
        entry = {
            "sessions": len(rows),
            "ms_per_token_median": statistics.median(ms) if ms else None,
            "ms_per_token": ms,
            "accept_length": (sum(r["completion_tokens"] for r in rows) / verifies) if verifies else None,
        }
        metrics = os.path.join(out, f"{arm}.metrics.jsonl")
        if os.path.exists(metrics):
            with open(metrics) as f:
                last = json.loads([line for line in f if line.strip()][-1])
            graphed = last.get("counters", {}).get("graphed_verify")
            if graphed and graphed.get("graphed_verify_ct"):
                entry["reverify_rate"] = graphed["verify_overflow_ct"] / graphed["graphed_verify_ct"]
                entry["reverify_ct"] = graphed["verify_overflow_ct"]
        probes = {a: os.path.join(out, f"{a}.probe.json") for a in ("prod", arm)}
        if arm != "prod" and all(os.path.exists(p) for p in probes.values()):
            with open(probes["prod"]) as f:
                base = json.load(f)
            with open(probes[arm]) as f:
                other = json.load(f)
            entry["text_vs_prod"] = dspark_text_band.compare(base, other)
        summary[arm] = entry
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    out = os.path.abspath(sys.argv[1])
    arms = sys.argv[2:] or list(ARMS)
    os.makedirs(out, exist_ok=True)
    for arm in arms:
        for step in (run_timed, run_probe):
            rc = step(arm, out)
            print(f"{arm} {step.__name__}: rc={rc}", flush=True)
            if rc:
                sys.exit(rc)
    print(json.dumps(summarize(out), indent=2), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run and pass** — both new test files, CPU. Then
`/data/models/slang/.venv/bin/python -c "import ast,sys; ast.parse(open('analysis/dsv41-drive/dspark/both_cpu_ab.py').read())"`.
Expected: `EXIT=0`, no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/expert_prediction/benchmarks/run_capture_sessions.py scripts/dsv41/dspark_text_band.py \
  analysis/dsv41-drive/dspark/both_cpu_ab.py test/registered/unit/scripts/test_dspark_text_band.py \
  test/registered/unit/scripts/test_run_capture_sessions_spec.py
git commit -m "feat(dspark-ab): the driver records accept-length inputs; the near-tie text bar; the three-arm server A/B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 15: Smoke, then the server A/B on divix01, then §33.10

Needs a GPU window: production down, owned by the owner. The executor waits on the locks.

**Files:**
- Modify: `DSV41_REFERENCE.md` (new §33.10, after §33.9)

- [ ] **Step 1: Smoke `dspark-both` (2 sessions)**

```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu
G=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/both-cpu
mkdir -p $G/smoke
flock /data/models/slang/nvfp4-work/rowimg-disk.lock env DSV41_RUN_ROOT=$G/smoke \
  DSV41_EXTRA_SERVER_ARGS="$(/data/models/slang/.venv/bin/python -c 'import sys; sys.path.insert(0,"benchmarks/dsv41_baseline"); import arm_env, shlex; print(shlex.join(arm_env.DSPARK_ARGV))')" \
  benchmarks/dsv41_baseline/run_arm.sh dspark-both 30017 \
  $(/data/models/slang/.venv/bin/python -c 'import sys; sys.path.insert(0,"benchmarks/dsv41_baseline"); import arm_env; print(" ".join(f"{k}={v}" for k,v in arm_env.dspark_env().items()))')
```
Every one of these must hold in `$G/smoke/servers/dspark-both/run-*/server.log`. Record each line, or record its
absence as a failure:
1. `DSpark: EXL3 draft graphs on`.
2. `numa dspark draft: node0 cpus=12-15 (4)`, and CPU experts groups on cores `[6, ..., 11]` and `[18, ..., 27]`.
3. The decode graph captured as `TARGET_VERIFY`, and `exl3 RAM miss copy engine armed after`.
4. No `fail-stop`, `__trap`, `RemoteDisconnected`, `CUDA error`, or `out of memory`.
5. The KV pool line (`max_total_num_tokens`). Record it for Step 2's bar.
6. The wire and room: the service logs a 40-lane build, and no layer was refused by the spill room check (Task 10). The
   CPU calibration line (`CPU experts group N: ... split`, measured to 8 lanes, or the `calibration skipped/failed`
   warning), and how long startup took from `Load weight end` to `/health` 200.

Engram fallback: if the log shows an Engram failure (`engram` in a traceback), re-run with
`SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=0 SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT=0` appended. Then commit that pair
into `dspark_env()` with the traceback quoted in its comment, as a fix-up commit.

- [ ] **Step 2: The A/B**

```bash
flock /data/models/slang/nvfp4-work/rowimg-disk.lock /data/models/slang/.venv/bin/python \
  analysis/dsv41-drive/dspark/both_cpu_ab.py $G/ab 2>&1 | tee $G/ab/driver.log; echo "EXIT=${PIPESTATUS[0]}"
```
Bars, each recorded from `$G/ab/summary.json` and the logs:
1. All three arms finish with `rc=0`, with no fail-stop or trap.
2. `dspark-both` and `dspark-draft-only` both have `text_vs_prod.pass` true: every first divergence from production
   is within 1.4 nats of production's argmax (§33.2, §33.9). Report the flips and `max_gap`.
3. **No verify re-runs in steady state.** Two checks on `dspark-both`:
   - `reverify_ct` must not exceed the verifies the server ran before its copy engine armed. That count is the `N` of
     `exl3 RAM miss copy engine armed after N decode forwards` in the server log, plus one for the warm-up's stale
     flag.
   - `gather_overflow` must not grow after the armed line. Read the per-layer counters from the metrics file's first
     and last records after arming.
   For contrast, report `dspark-draft-only.reverify_rate`, which §33.8/§33.9 put at 1.00. Any armed overflow is a
   failed bar and a defect in Tasks 2-10, not noise.
4. `dspark-both`'s server log shows nonzero target CPU-expert jobs (the periodic `CPU experts group` stats).
5. `dspark-both`'s KV pool is at least the `prod` arm's.
6. Report for all three: `ms_per_token_median` with the per-session list, `accept_length`, and the reverify rate.
   This is the input to Owner decision 1.

A failed bar is reported as failed, with its numbers. Do not re-run an arm to get a different number without saying so.

- [ ] **Step 3: Write §33.10** in `DSV41_REFERENCE.md`, after §33.9's "What this plan does not do" list:
  - the commit and the plan;
  - what changed (Tasks 2-13, one line each);
  - Steps 1-2's commands;
  - a table of the three arms × (ms/token median, accept length, reverify rate, text pass and max gap, KV pool);
  - each bar's verdict;
  - the gaps: the draft CPU forward ms is still unmeasured (§33.9), and the CPU split is calibrated per one-token job;
  - the recommendation for Owner decision 1.

Use only measured numbers. Where a number is missing, say so.

- [ ] **Step 4: Commit and push**

```bash
git add DSV41_REFERENCE.md
git commit -m "docs(dsv41): section 33.10 -- DSpark with both CPU-expert clients, the server A/B against production

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-dspark-both-cpu
```

---

### Task 16: Flip production to DSpark (owner-gated)

Run only on the owner's go (Owner decision 1), given in their own words after reading §33.10.

**Files:**
- Modify: `benchmarks/dsv41_baseline/arm_env.py` (`PROD_DSPARK = True`), `benchmarks/dsv41_baseline/test_dspark_recipe.py`

- [ ] **Step 1: Change the test first**

Replace `test_prod_is_unchanged_until_the_switch` with:
```python
def test_prod_serves_the_dspark_mode():
    assert arm_env.PROD_DSPARK is True
    assert arm_env.prod_env() == arm_env.arm_env(arm_env.dspark_env())
    argv = arm_env.ServerArgs.prod().argv()
    assert argv[argv.index("--speculative-algorithm") + 1] == "DSPARK"
```
Run it and see it fail (`PROD_DSPARK is False`).

- [ ] **Step 2: Set `PROD_DSPARK = True`.** Add a comment above it naming §33.10 and the owner's go (date, words).
Run the test and see it pass. Then run `DRY_RUN=1 benchmarks/dsv41_baseline/launch_prod.sh` and check `dspark: True`,
the DSpark argv, and `dspark_env()`'s values in the printed env.

- [ ] **Step 3: Commit and push the branch.** Merging to `master` and restarting production are the owner's steps.
This plan does neither.

```bash
git add benchmarks/dsv41_baseline/arm_env.py benchmarks/dsv41_baseline/test_dspark_recipe.py
git commit -m "feat(dsv41-baseline): production serves DSpark with both CPU-expert clients (owner go, §33.10)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-dspark-both-cpu
```

---

## Self-review notes (writer's pass, 2026-10-06, after amendments A and B)

- **Spec coverage.**
  - Both clients at once: Tasks 2-13.
  - The verify, graphed: Tasks 2-11. The eager re-verify is unchanged and GPU-only, and in steady state it never runs
    (Owner decision 5). Task 15 bounds its count by the pre-arming verifies.
  - Amendment A, a record a 6-token verify cannot exceed: Task 2 (40-lane wire, u64 masks, shared kernels, BS1 digest)
    and Task 8 (spill requires a lane per route; the clamp asserts).
  - Amendment B, forced misses always land: Tasks 3-4 (no staging for forced misses), Task 9 (host-placed into RAM
    victims) and Task 10 (start-up room check). The costs are in Owner decision 3.
  - The one overflow left (case 1): Tasks 4 and 11.
  - Prefill: unchanged. The target's CPU experts type lanes only on a captured post (`lease_kernels.cuh:151`), prefill
    is eager, and the DSpark draft does not run at prefill (`dspark_worker_v2.py:653-745`).
  - Cores: Owner decision 2 and Task 13's layout test.
  - Lease and pinned-tier interplay: separate channels and areas (`LEASE_PROTOCOL.md:71`); strictly sequential on one
    stream; draft weights pageable, not tier rows (`exl3.py:431-441`). Forced misses take tier victims only in the
    target's own rows. Task 10 documents it.
  - Gate: Task 12. Recipe and launch: Task 13. Validation: Tasks 2-11, 14, 15.
- **Type consistency.**
  - The post's FFI tail order is `cpu_x, cpu_x_dst, cpu_weights, cpu_tokens_max, cpu_x_token_bytes, spill,
    overflow_flag, gather_overflow, use_pdl` in Tasks 4 and 5 and their raw-call edits.
  - The mask words are `ce_mask` {lo, cpu lo, parts, hi, cpu hi} and `cpu_lanes` {cpu lo, parts, cpu hi}, in Task 2's
    CW/CC, route tables, commit and transport.
  - `victim_lanes` is used in Tasks 8, 10 and 12.
  - `cpu_row_bytes(hidden, tokens, lanes)` is used in Tasks 5, 6 and 10.
  - `spill = (overflow_flag, gather_overflow[row:row+1])` is used in Tasks 4 and 10.
  - Lane slot −1 for a forced miss is used in Tasks 3, 4 and 9.
- **Known open item, not a placeholder.** The BS1 digest's host `.text` comparison is meaningful only within Task 2.
  Tasks 6 and 9 change the host on purpose, so the permanent test compares device kernels, and the host's BS1
  behaviour stays pinned by `test_expert_stream_hotpath_golden.py`.
