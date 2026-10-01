# DSV4.1 Device Slot Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lease protocol's host grant with a device-held copy of the pinned RAM tier's expert-to-slot map. The GPU types every missing expert itself as one of:
- a RAM hit, copied by the copy engine or by SMs;
- a RAM hit computed on the CPU;
- an NVMe miss streamed into VRAM;
- an NVMe miss computed on the CPU.

NVMe misses land in per-layer staging slots that swap into the RAM tier, and the host sends the map down only as deltas.

**Architecture:**
- **One post per layer.** It applies the layer's pending map delta, types the lanes from the map, and publishes a record. The record names, for every lane, its kind, source slot and VRAM destination.
- **The host is an executor and the policy owner.**
  - The copy thread issues DMA for copy-engine hits straight from the record.
  - The service thread reads NVMe misses into the staging slots the record names.
  - The CPU expert thread computes CPU lanes.
  - The service picks RAM victims and publishes the resulting map delta, which the device applies at the start of that layer's next miss chain.
- **Removed:** leases, RowResult, `demand_done`, `Done`, W1 and deferral.
- **Kept:** the copy wait's gate and `CopyDone`, which is how a captured graph waits on host work.

**Tech Stack:**
- CUDA JIT kernels (`python/sglang/kernels/jit/csrc/moe/expert_stream/`);
- C++ host service (`host/*.h`, tvm-ffi exports);
- Python (`exl3_ram_miss.py`, `expert_stream_transport.py`, `cpu_experts/`);
- pytest, CPU suites under `test/registered/unit/kernels`, GPU suites under `test/manual/dsv41`;
- the replay simulator `scripts/dsv41/cpu_expert_sim.py`.

**Spec:** this document's `## Design` section. It was agreed in conversation on 2026-09-30 and builds on `analysis/dsv41-drive/LEASE_PROTOCOL.md` (the protocol as it is at `59cfb07c99`) and on the staging replay on branch `dsv41-staging-sim` (`6a6bb18dbb`, `8b8f796fd0`; outputs `divix01:/mnt/nvme1/cpu-p1/staging-policies.{json,txt}`).

## Global Constraints

- **Branch.** Work on `dsv41-device-slot-map`, off `origin/master` `59cfb07c99`, in the laptop worktree `/home/dimitri/data/divix/wt-device-slot-map`. Push only that branch.
  - Never push master without the user's approval.
  - Never amend, rebase or force-push.
  - Stage files by name.
- **Commit trailer**, on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01SpADbKigFM3GYRraGyx46R
  ```
- **Code reaches divix01 only by commit → push → `git fetch` into a private worktree** (`/data/models/slang/nvfp4-work/wt-<name>`). No rsync, scp or git archive.
  - Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- **CPU suites.**
  - Run with `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest ... -q -p no:randomly`.
  - Print `sglang.__file__` once per worktree.
  - Read `${PIPESTATUS[0]}` whenever output is piped.
  - Record the command next to every count you quote.
- **GPU suites.**
  - Run with `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 ...` and `SGLANG_EXL3_SRC` set, as the existing manual suites do.
  - Server launches, smoke tests and A/B arms need the user's explicit go-ahead.
- **Cores.** Cores 64-71 stay free. Cores 18-29 belong to another lane, so CPU jobs use `0-17,30-63`.
- **Settings.** `EXL3_MOE_CPU_PIN=0`.
- **Environment variables.** Read `.claude/skills/env-var-conventions/SKILL.md` before touching `python/sglang/srt/environ.py`.
- **Comments.** Follow `.claude/rules/comment-style.md`: one or two lines, ASCII only, facts not visible from the code, no history.
- **Mutants.** Apply them only in a private divix01 worktree. Revert, re-run green, and record both results.
- **Don't touch:**
  - Codex worktrees, processes or `codex/*` branches;
  - root or sudo system changes;
  - `IORING_REGISTER_CLONE_BUFFERS`, which is never used.
- **No backwards-compatible shim for the old wire.** The old protocol is replaced in place, as the minimal-lease rewrite was. `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US`, which is gone, must warn and be ignored, like `_LEASE_CHAIN_MINIMAL_NOTE` in `environ.py`.
- **Review.** A separate reviewer approves each task. The implementer never self-approves.

## Review Focus

Each line below has a test in the task that owns the code.

1. **A late or missing delta.** The device reaches a layer's next miss chain before the host has published the previous chain's delta. Expected: the post waits, bounded by `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`, then traps. It must never type lanes from a stale map. Tests: Task 2 (`test_delta_published_before_service_returns`) and Task 5 (`test_post_waits_for_delta_then_traps_at_deadline`).
2. **No evictable RAM slot.** Every slot in a layer is VRAM-hot, routed, or staging. Expected: the miss is still served and computed, and the expert is simply not cached in RAM. There is no abort, the staging list is unchanged, and `ram_insert_skipped` counts it. Test: Task 2 (`test_no_victim_skips_ram_insert`).
3. **The ring laps under SM-only chains.** In SM hit mode with no misses, the device never waits on the host, so the 16-record ring can lap. Expected: the host loses stamps for the lapped records, the map is untouched, and nothing deadlocks. Test: Task 3 (`test_lapped_hit_only_records_do_not_stall`).
4. **A CPU miss whose read fails.** The NVMe read for a CPU-computed miss fails or faults. Expected: the process aborts with FATAL before `CopyDone`, and the CPU never runs on partial bytes. Test: Task 6 (`test_cpu_miss_read_failure_aborts_before_copy_done`).
5. **Eager host use between replays.** A prefill admission or eager assign happens while the service is paused. Expected:
   - it never takes a staging slot;
   - its changes reach the device as a bulk delta that is applied after any pending decode delta;
   - after mixed eager and decode traffic, the device map equals the host's `slot_map`.

   Tests: Task 4 (`test_bulk_after_pending_decode_delta`, `test_admission_never_takes_staging`) and Task 7 (`test_device_map_matches_host_after_mixed_replay`).

## Revision 1 (binding; overrides any task or Design text it contradicts)

This revision comes from the plan review (critic verdict REVISE) and from the user's decision of 2026-09-30 to use two CPU result rows. Each item names the task that owns it.

**R1-1. Two CPU result rows, for a future second CPU thread** (Tasks 6 and 5).
- **The rows.** `out_rows` becomes fp32 `[rows, 2, hidden]`:
  - part 0 holds the CPU-hit partial sum;
  - part 1 holds the CPU-miss partial sum.
- **The jobs.** Each part is written by one job, which overwrites it.
  - `CpuJob` gains `part` (0 or 1).
  - `CpuExpertConfig` gains `out_part_stride`.
  - The forward's C ABI is unchanged: the engine passes `out_base + row*out_stride + part*out_part_stride`.
- **When the jobs are submitted.**
  - The hit job (part 0) is submitted at record time, with every `kHitCpu` lane.
  - The miss job (part 1) is one job for every `kMissCpu` lane of the request. It is submitted when the last piece of the last `kMissCpu` lane has landed.
  - The two jobs share no state, so a second engine could run part 1. That engine is not built here.
- **The device side.**
  - CC writes a new device word, `cpu_parts` (int32 `[1]`): bit 0 if any `kHitCpu` lane, bit 1 if any `kMissCpu` lane. It writes it next to `cpu_lanes` and only after its `CopyDone` acquire.
  - `exl3_route_tables` takes `cpu_out` (row r, part 0), `cpu_part_stride` and `cpu_parts`, and seeds `out_zero[i] = (bit0 ? ldcv(part0[i]) : 0) + (bit1 ? ldcv(part1[i]) : 0)`. It never reads a part whose bit is clear.
  - `cpu_lanes` still masks every CPU lane, whether hit or miss.
- **Tests (Task 6):**
  - `test_hit_and_miss_cpu_parts_both_seed_the_output`: one request with one `kHitCpu` and one `kMissCpu` lane, against the eager reference;
  - `test_stale_part_is_never_read`: part 1 is pre-poisoned with NaN, and a request with hits only must give a finite, correct output.

**R1-2. A lapped record whose lanes are all `kHitSm` is skipped, not fatal** (Tasks 2 and 3).
- **The bug.** `pump_demand` fail-stops when a record with lanes fails its sidecar or seq recheck (`ram_tier.h:249-255`). Under this plan, SM-only chains never wait on the host, so they can lap.
- **The rule.** A record whose every lane is `kHitSm` counts a failed recheck as `lapped` and is skipped. Only a record with a miss lane or a host lane (`kHitCopy`, `kHitCpu`, `kMissCpu`) fail-stops; the device always waits on those, so they cannot lap.
- **Test.** `test_lapped_hit_only_records_do_not_stall` builds the GPU-hot page and runs with GPU-hot on.

**R1-3. The copy job covers `kMissCpu` too** (Tasks 3 and 6). Host step 3 builds the `CopyJob` when any lane is `kHitCopy`, `kHitCpu` or `kMissCpu`. A job whose only host lanes are `kMissCpu` has no DMA token and completes on its CPU job alone.

**R1-4. Per-row eligibility is checked on the device and enforced on the host** (Tasks 5 and 2).
- **The device bank.** `map_bank` gains:
  - `ce_ok` (uint8 `[rows]`), set when the row's copy table is registered with non-empty entries;
  - `dst_rows` (int32 `[rows]`);
  - `cpu_ok` (uint8 `[rows]`), set at the row's `set_cpu_layer`.
- **Typing.** `kHitCopy` requires `ce_ok[row] && dst < dst_rows[row]`. CPU kinds require `cpu_ok[row]`. Otherwise the lane falls back to `kHitSm` or `kMissGpu`.
- **The reference.** `type_lanes` gains `ce_ok: bool`, `cpu_ok: bool` and `dst_ok: list[bool]` keyword arguments with these semantics.
- **The host.** It fail-stops on a kind the row is not eligible for: "request G lane j: kind K on an ineligible row". It also counts `kCopyFallbacks` per kind.

**R1-5. Eager use: the bulk delta is taken after the fills settle** (Task 4). `expert_stream_take_bulk_delta` first runs `fill_join()` and `finish_fill_owned()`, the same calls `resume_locked` makes (`ram_thread.h:138-141`). This captures every `publish_map(-1)` a failed fill makes. Then `resume()` restarts the service.

**R1-6. The device map is never copied whole** (Task 5).
- `ram_slot` starts at -1 on the device. The host's bulk log records from tier construction.
- `Exl3RamMissService.attach` takes and applies the bulk delta after the map bank is allocated and every row's attach delta is published.
- The Task 5 Step 7 text "ram_slot from the host's slot_map (one copy, at attach only)" is void.

**R1-7. Tags start at 1** (Tasks 1, 2 and 5).
- The attach delta has `tag = 1`. `map_chain` starts at 1 and `map_applied` at 0, and the reference `MapReplica` matches.
- A zero-filled delta record has tag 0, and `map_chain` is never 0, so a delta that was never written cannot be applied.
- A map chain sets `map_chain[row] += 1`, so the first miss chain is 2.
- The Design table's "starts at -1" and the earlier "tag 0" text are void.

**R1-8. K reaches the host through a new export** (Task 2). `expert_stream_attach_row(handle, row, k)`, called from `Exl3RamMissService.attach` once per row with `k = graph_gather_rows`. It:
- takes K free slots;
- marks them STAGING;
- publishes the tag-1 delta.

Rows are built in the `RamTier` constructor (`ram_tier.h:68-79`), and there is no `attach_row` there today.

**R1-9. RAM reads only for miss lanes** (Tasks 0 and 2).
- **Today.** `serve` reads every expert in `protect ∪ lanes` that has no RAM slot (`ram_tier.h:1611-1640`), including VRAM-hot routed experts, and S waits for those reads.
- **The new design.** It reads only miss lanes.
- **Task 0** adds a replay arm, `protect_reads ∈ {on, off}`, to measure what this change costs. If `off` loses more than 0.5 ms/token, stop and report to the user before Task 2.

**R1-10. The host mirror is written after the bytes land** (Task 2).
- `publish_delta_locked` publishes the device delta before the reads, as the Design says.
- The host's `slot_map` entry for the inserted expert, made by `publish_map(e, s)`, is written when the lane's last piece lands. That keeps `mapping()`'s READY-only contract (`ram_tier.h:1041`).
- The unmap of the victim, `publish_map(old, -1)`, is written at once.

**R1-11. One home for the split table** (Task 6).
- The split lives only in the completion block's `split[9]`. `CpuExpertService` writes it through a new export, `expert_stream_set_split(handle, split)`, both at start and on retune.
- Delete `RamTier::cpu_split_`, and the split argument of `enable_cpu_experts`.

**R1-12. The post's restructuring, in order** (Task 5).
1. Compute the deadline (`globaltimer + timeout_ns`).
2. Thread 0 traps if `count > 8`, if `count > lanes`, or if two lanes name the same expert.
3. Thread 0 applies the delta, against that deadline.
4. Thread 0 types the lanes.
5. Thread 0 writes `any_cpu` to shared memory, followed by `__syncthreads()`.
6. If `any_cpu`, the whole block stages `x`, then `__threadfence_system()`.
7. Thread 0 writes the sidecar, the record and `demand_head`.

**R1-13. `RamTier::open` seeds `next_demand_` from `demand_head`** (Task 2). Today it reads `kDemandDone` (`ram_tier.h:103`).

**R1-14. Test files and importers** (Task 2).
- **The wrap file.** Merge `_lease_wrap.py`'s cases into the existing `test/registered/unit/kernels/test_exl3_ram_miss_wrap.py`. Do not create a second file of that name. The "`demand_done` never 0" case becomes "the record seq never 0".
- **Files to update.** Every one of these must build and pass at the end of Task 2, or the task that next owns it. Each goes in the commit that changes it, staged by name.
  - **Production and support code:**
    - `host/ffi_test_exports.h`, `host/ram_thread.h`, `host/ram_tier.h`, `host/tier_protocol.h`
    - `kernels/ops/moe/expert_stream_transport.py`
    - `python/sglang/test/dsv41_ram_miss_fixtures.py`, `python/sglang/test/hotpath_script.py`
    - `scripts/dsv41/exl3_stage_trace_overhead.py`
    - `test/registered/unit/kernels/golden/hotpath_golden.json`
  - **`test/registered/unit/kernels/` tests:**
    - `test_exl3_ram_miss_{copy_engine,cpu_experts,piece_stream,prefill_share,stage_trace,stage_trace_causal,stage_trace_lanes,thread,tier,trace_export,wrap}.py`
    - `test_expert_stream_{build_variants,hotpath_stress,ownership,second_layout}.py`
  - **Elsewhere:** `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`.
- **Regenerating the golden file.** Use the command `hotpath_script.py`'s docstring gives. Record why each counter changed.

**R1-15. Fixtures and helpers that a task creates rather than finds.** Any fixture or helper a test uses that the repo does not have, the owning task creates in the test file or in `dsv41_ram_miss_fixtures.py`. That covers:
- `_tiny_trace`, `run_arm` and `choose_lane_kinds`. These are wrappers over the sim's real entry point, `replay_nm` (`scripts/dsv41/cpu_expert_sim.py:233`).
- `chain_sim*` and every `ChainSim` method the tests call: `delta`, `staging`, `kinds`, `slots`, `set_gpu_hot`, `hold_service`/`release_service`, `block_copies`/`unblock_copies`, `counter`, `read_slot`, `image`, `dst_bytes`, `cpu_out`, `reference_forward`, `cpu_job_started_after_last_piece`, `last_chain` and `apply_bulk_like_device`.
- `rig` and `rig_subprocess`, rewritten from `lease_chain_rig.py`.
- `backend_mock`, `gate` and `graph_rig`. `graph_rig` extends `test_exl3_ram_miss_graph_gpu.py`'s existing setup.
- `assert_device_trapped`, factored out of `test_exl3_lease_kernels_cuda.py:130`'s inline check.

In Task 1, the Python mirror the plan calls `WIRE` is the existing `PYTHON_WIRE`/`WORDS`/`lease.*` structure in `expert_stream_transport.py`. Extend that, and do not invent a second one. `rig.set_map_random` sets the staging list through a test export that writes the delta record, and sets `ram_slot` through the bulk path.

**R1-16. Smaller corrections.**
- **PDL list.** `PDL_KERNELS` in `test_exl3_ram_miss_device_args.py` drops W1. `map_bulk_apply` is never PDL. (Task 5)
- **Bulk-apply wiring moves.** Task 4's Python wiring of `after_host_use` → `map_bulk_apply` moves to Task 5. Task 4 tests the host side through `ChainSim` only, so there is no Python fallback in production code.
- **Delta ordering test.** Task 7 adds a mutant, "store the delta tag before its entries", which `test_device_map_matches_host_after_mixed_replay` must catch. That replaces the coverage lost with `_lease_publication.py`.
- **Wire table wording.** In the wire table, PieceMask and `CopyDone` keep their contents but move offset, to 0 and 16384. "Unchanged" means the encoding.
- **Safety point 3, reworded.** A victim is safe because every post applies the row's pending delta before it types lanes. So no lane of a later chain can name the victim's old mapping, and no lane of the current chain is routed to it.
- **Duplicate experts.** `plan_unique_routes_kernel` emits unique experts. `type_lanes`, in both the reference and CUDA, refuses duplicates (`ValueError` or a trap).

**R1-17. CPU activations, the full path** (Tasks 5 and 6).
- **Up.** The post stages `x` as fp16 (2·H bytes) into `x_rows[row]` only when a lane is a CPU lane (R1-12).
- **CPU compute.**
  - The hit job (part 0) starts at record time, so it overlaps the NVMe reads.
  - The miss job (part 1) starts after the last CPU-miss piece lands.
- **Down.** The copy thread publishes `CopyDone = G` only after DMA, part 0 and part 1 are all done. CC's acquire orders both parts before the route tables read them over PCIe.
- **When the rows are rewritten.** `x_rows[row]` and `out_rows[row][*]` are rewritten only by the row's next CPU chain, which is posted only after this chain's CC.
- **Placement.** Both rows stay pinned where `CpuExpertService` allocates them today. Their per-layer cost is 2·H bytes up and at most 8·H bytes down. The first served CPU-miss arm reports it.

---

## Design

### Terms

- **Row.** One streamed MoE layer (`tables.layer_ids[r]`). Each row has its own RAM tier, the `Tier` in `ram_tier.h`. Its slot indices run over `[0, capacity[row])`, and a slot's address for name n is `slabs[row][n] + slot * row_bytes[n]` (`row_reader.h:370-381`, `copy_engine.h:654-656`).
- **Lane.** One VRAM-missing routed expert of a request, at plan position j, with `count <= 8` (`plan.expert_ids`, sorted by `miss_keys` when CPU experts are on, `expert_route_plan.cuh:69-77`). Lane j's VRAM destination is DIRECT's victim `plan.slots[j]` (`direct_gather.cuh:70-75`).
- **K, the staging slots per row.** `K = graph_gather_rows`, at most 8 (`exl3_ram_miss.py:804-809`). These K slots of the row's tier are never mapped. A miss is read into one of them.
- **Map chain.** A row's request that has at least one miss lane. Only map chains change the map. The device counts them per row in `map_chain[row]`, a u64 starting at 0.

### Device state (VRAM, one bank per process)

| Tensor | Shape, dtype | Written by | Meaning |
|---|---|---|---|
| `ram_slot` | `[rows, experts]` int32 | post (delta apply), bulk apply | The tier slot of each expert, or -1 when it is not in RAM. It is a copy of the host's `slot_map`. |
| `staging` | `[rows, 8]` int32 | post (delta apply), bulk apply | The row's K staging slots. Entries past K are -1. |
| `map_chain` | `[rows]` int64 | post | The row's last map-chain number c. |
| `map_applied` | `[rows]` int64 | post, bulk apply | The tag of the last decode delta applied, so a delta is never applied twice. It starts at -1, so the first post applies the tag-0 attach delta (the initial staging list). |
| `lane_kind` | `[8]` uint8 | post | Each lane's kind (below). |
| `lane_slot` | `[8]` int32 | post | Each lane's source slot: its RAM slot for a hit, its staging slot for a miss. |

These replace the per-chain `claimed[8]`. `go_1`, `host_rows_1` and `dst_slots_1` stay, and the post now fills them.

### Lane kinds

| Kind | Value | When | Moves the bytes | Completion the device waits on |
|---|---|---|---|---|
| `kHitCopy` | 1 | A RAM hit, captured, `copy_armed`, `SGLANG_DSV41_RAM_HIT_COPY=ce` | The copy thread, with `cuMemcpyAsync` | `CopyDone` and the gate |
| `kHitSm` | 2 | A RAM hit, otherwise: eager, unarmed, or `=sm` | C1, using SM loads | Stream order |
| `kHitCpu` | 3 | A RAM hit in the CPU tail | Nothing. The CPU thread computes it from the slot. | `CopyDone` and the gate |
| `kMissGpu` | 4 | An NVMe miss outside the CPU tail | The service reads it into a staging slot, then S copies each piece | PieceMask bits tagged with G |
| `kMissCpu` | 5 | An NVMe miss in the CPU tail (`SGLANG_DSV41_CPU_EXPERTS_MISSES=1`) | The service reads it into a staging slot. The CPU computes it once every piece has landed. | `CopyDone` and the gate |

**The CPU tail.** The eligible lanes are the hits, plus the misses when `CPU_EXPERTS_MISSES` is set. Eligibility also needs a captured post with CPU experts on and `copy_armed`. Let n be the number of eligible lanes. The last `split[n]` of them, in plan order, go to the CPU. Plan order is `miss_keys` descending, so these are the lowest-scored lanes. The split table is 9 int32 in pinned host memory, read by the post. The retune writes it.

### Wire, version 2 (`lease_layout.h`, mirrored in `expert_stream_transport.py` and `expert_lease_block.py`)

**Request page.** Device-written, host-read, 4,160 B. The ring index is `idx = (seq - 1) % 16`.

| Field | Offset | Contents |
|---|---|---|
| `demand_head` | 0 | u32, the last posted seq, stored with a release |
| ring | 64 | 16 records × 256 B |

A record:

| Field | Offset | Contents |
|---|---|---|
| `seq` | 0 | u32 seqlock |
| `row` | 4 | u16 |
| `count` | 6 | u16 lanes |
| `flags` | 8 | u32: CAPTURED = 1 |
| `chain` | 12 | u32: the low 32 bits of this record's map-chain number, or 0 when it has no miss lane |
| `chain_hi` | 16 | u32 |
| `protect_count` | 20 | u16 |
| `protect[8]` | 32 | i32: every routed expert of the request |
| `lane[8]` | 64 | 16 B each: `{expert i32, slot i32, dst i32, weight f32}` |
| `kind[8]` | 192 | u8 |

**Completion block.** Host-written, device-read, 20,480 B, 4096-aligned.

| Field | Offset | Contents |
|---|---|---|
| `PieceMask[16][8]` | 0 | One 128 B line each: `G << 8 \| piece bits`. Unchanged. |
| `CopyDone[16]` | 16384 | u64 G. Unchanged. |
| `COPY_GATE` | 16512 | u32 on its own line. The encoding is unchanged. |
| `copy_armed` | 16640 | u32: 1 once the service has armed the copy engine |
| `split[9]` | 16768 | i32 |

**Delta block.** Host-written, device-read, `rows × 256 B`, 4096-aligned. One record per row:

| Field | Offset | Contents |
|---|---|---|
| `tag` | 0 | u64: the map-chain number this delta follows. Stored last, with a release. |
| `count` | 8 | u32: entries used |
| `staging[8]` | 16 | i32: the row's staging list after this delta |
| `entry[16]` | 48 | `{expert i32, slot i32}`, meaning `ram_slot[row][expert] = slot`, where -1 unmaps |

16 entries cover the worst case: 8 inserts plus 8 evictions. At attach the host writes `tag = 0` with the initial staging list and no entries.

**Hot sidecar.** Unchanged (`kHotHeaderBytes`, `kHotAlignment`, `kHotRecords`).

**Removed from the wire:** RowResult, LaneRequest (folded into the record), `Done`, `demand_done`, and the `kLeaseTag*` constants.

### The chain (one stream, captured)

`post → C1 → S → CW → stream wait → CC`, then the fused MoE and DIRECT commit as today.

1. **post** runs as one block of 32 threads, with thread 0 doing the serial part.
   1. **Delta apply.** Wait, with a deadline, until `delta[row].tag == map_chain[row]`. Then, unless `map_applied[row] == tag`, apply the entries and the staging list and set `map_applied[row] = tag`.
   2. **Typing.** For each lane j: `s = ram_slot[row][expert[j]]`. A lane with `s >= 0` is a hit at slot s. Otherwise it is a miss, at `staging[row][m]` for the m-th miss, and a negative entry traps. Pick the CPU tail and the kinds as above. Write `lane_kind`, `lane_slot`, and the C1 compaction (`go_1`, `host_rows_1`, `dst_slots_1` for `kHitSm` lanes).
   3. **Map-chain count.** If any lane is a miss, `map_chain[row] += 1` and `record.chain = map_chain[row]`.
   4. **Stage CPU input.** The CPU input `x` is staged when any lane is a CPU lane. The whole block does this, as today.
   5. **Publish.** Write the hot sidecar, then the record behind its seqlock, then `demand_head` with a release. Set `state[kPending]` to the seq when the record has lanes, or 0, and set the S deadline as today.
2. **C1.** Unchanged: `copy_expert_row_segments_gpu(segments, host_rows_1, dst_slots_1, go_1)`.
3. **S** owns the `kMissGpu` lanes. For each piece bit tagged with G in the lane's PieceMask, it copies from `lane_slot` (a staging slot) to `dst`. It ends when every owned lane's mask is complete, and traps at its deadline. It no longer reads `demand_done` or RowResult.
4. **CW** builds the copy mask (`kHitCopy`) and the CPU mask (`kHitCpu | kMissCpu`) from `lane_kind`. When SM small copies are on, it copies the `sm_table` entries for `kHitCopy` lanes from `lane_slot`. If either mask is non-empty it closes the gate and runs the Dekker pair against `CopyDone`, unchanged. It writes no `Done`.
5. **Stream wait and CC** are unchanged. CC sets `cpu_lanes` to the CPU mask, which now includes `kMissCpu`.
6. **DIRECT commit** is unchanged. CPU lanes, hits and misses alike, stay unmapped in VRAM. A `kMissCpu` expert still enters the RAM tier through the staging swap.

### The host, per record (`RamTier::pump_demand` → `handle_record`)

1. Acquire `demand_head` and read the record behind its seqlock. A lapped record is skipped and counted in `lapped`, as today. A record with no lanes runs `touch_request`, as today.
2. Read the GPU-hot sidecar into `tier.hot`. Stamp the protect list and every hit lane's slot.
3. **Copy job.** If any lane is `kHitCopy` or `kHitCpu`, build a `CopyJob` from the record's lanes (`host_slot = lane.slot`, `dst_slot = lane.dst`, `weight`) and submit it at once. If any lane is `kMissCpu`, set `job.late_cpu = <count of kMissCpu lanes>`.
4. **Victims and the delta, before any read is issued.** For each miss lane (expert e, staging slot s):
   1. Choose a victim v with `take_victim_locked(row, wanted = protect ∪ lane experts)`. It returns a free slot if there is one. Otherwise it returns the READY slot with the lowest stamp that is not hot, not wanted and not staging.
   2. If a v exists:
      - set the tier slot s to `{e, READY}` and stamp it;
      - unmap v's old expert and mark v STAGING;
      - add the entries `(e, s)` and `(old(v), -1)` (or only `(e, s)` when v was free);
      - in the staging list, replace s with v.
   3. If no v exists: the expert is not cached and s stays staging. Count it in `ram_insert_skipped`.
   4. Publish the delta (entries, staging, count, then `tag = record.chain` with a release). Because this comes before the reads, a chain the device has seen served always has its delta published.
5. **Misses.** For each miss lane, initialise its PieceMask word for G, then issue its NVMe reads into `lane.slot` (the staging slot). Pieces publish as today (`publish_piece`).
6. **CPU misses.** When a `kMissCpu` lane's last piece lands, send the copy thread a `LateCpu{gen, slot, weight}` message. The copy thread submits it to the CPU engine as part of job G.
7. **Copy thread completion.** Job G completes when:
   - its DMA token has completed;
   - every CPU job it submitted is done (`cpu->done(seq)`);
   - `late_cpu` late jobs have arrived and completed.

   It then calls `copy_completed` (the `CopyDone` release, the fence, the gate CAS), unchanged.

**Removed from the host:**
- leases (`leases[]`, `Outstanding`, `open_lease_entry_locked`, `retire_leases`, `release_lease_locked`, `release_copied_owned`, `copy_acked`, `lease_changes_`);
- `defers` and both deferral kinds;
- `grant_lanes_locked` and `choose_cpu_lanes_locked` (the device does this now);
- the `demand_done` store, `init_piece_words_locked`'s RowResult half, and `graph_leases_outstanding`.

`pause` waits for the service to go idle and for the copy job ring to drain, instead of checking for outstanding leases.

### Eager host use (paused; `before_host_use` / `after_host_use`)

- **While paused,** eager `assign`, prefill fills and admissions run against the host tier as today, with one change: `take_admit_slot_locked` and `assign` exclude STAGING slots. Every change to `slot_map` is appended to a host-side bulk list `{row, expert, slot}`.
- **On `after_host_use`, before the service resumes,** `Exl3RamMissService` takes the bulk list (`host.take_bulk_delta()`) and runs `map_bulk_apply` on the current stream.
- **`map_bulk_apply`** first applies any pending decode delta per row, honouring `map_applied`. That delta is already published, because pause waits for the service to idle. It then writes the bulk entries.
- **Ordering.** The same stream orders this before the next forward's posts.

### Why this is safe without leases

1. **One chain in flight.** All rows' chains run in one stream, and a chain that waits on the host (S, or CW's gate) ends before the next post.
2. **A row's slots are read only by that row's chains** (per-row tiers). Its next chain runs after the current one has ended.
3. **What chain c's readers can touch.** Chain c reads only hit slots (mapped and routed) and its own staging slots. The host writes only staging slots, and only the ones the record names. Victims are unmapped by the delta, which the device applies in chain c+1's post. Before then they are not routed, so nobody reads them. They are rewritten only once a later post names them as staging.
4. **CPU and DMA readers finish before `CopyDone`,** which CC requires before the chain ends.

### Not built

- A GPU-chosen RAM victim.
- A global (cross-row) staging pool.
- Pull requests in the middle of a layer.
- A wire-compatible mode.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h` | Wire v2 constants | Rewrite |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | Python wire mirror, device-side buffers, launchers | Modify |
| `python/sglang/kernels/ops/moe/expert_lease_block.py` | Python mirror of the block offsets | Modify |
| `python/sglang/srt/layers/moe/ram_slot_map.py` | **New.** `LaneKind`, the `type_lanes` reference, and `MapReplica` (applying decode and bulk deltas). This is the Python model the CPU and GPU parity tests share. | Create |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` | Device helpers: `apply_map_delta`, `type_lanes`, record writes | Modify |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` | post v2, `map_bulk_apply`. W1 is deleted. | Modify |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh` | S on `kMissGpu`, CW masks from `lane_kind`, no `Done` | Modify |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h` | Record read v2, counters | Modify |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` | `handle_record`, `take_victim_locked`, staging state, delta publication, bulk list. Leases are deleted. | Modify |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h` | Jobs from records, `LateCpu`, completion. No lease release. | Modify |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h`, `ffi_test_exports.h` | `take_bulk_delta`, `delta_block`, staging introspection. Lease exports are deleted. | Modify |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | The backend's `post` (no W1), attach (map bank, initial deltas), `after_host_use` bulk apply | Modify |
| `python/sglang/srt/layers/moe/cpu_experts/service.py` | The split table becomes pinned and device-read | Modify |
| `python/sglang/srt/environ.py`, `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` | `SGLANG_DSV41_RAM_HIT_COPY`, `SGLANG_DSV41_CPU_EXPERTS_MISSES`; `HIT_WAIT_US` warns and is ignored | Modify |
| `python/sglang/test/dsv41_lease_sim.py` → `python/sglang/test/dsv41_chain_sim.py` | The CPU stand-in for the device, on wire v2 | Rename and rewrite |
| `scripts/dsv41/cpu_expert_sim.py`, `scripts/dsv41/tier_sim.py` | Staging capacity and CPU-miss policies | Modify (Task 0) |
| `analysis/dsv41-drive/LEASE_PROTOCOL.md` | The protocol as it now is | Rewrite (Task 8) |

---

### Task 0: Replay the two open policy questions

Do this before any protocol code. Its results set two defaults: K's capacity cost, and whether `CPU_EXPERTS_MISSES` defaults on.

**Files:**
- Modify: `scripts/dsv41/cpu_expert_sim.py`, `scripts/dsv41/tier_sim.py`
- Test: `test/registered/unit/kernels/test_cpu_expert_sim.py`

**Interfaces:**
- Consumes: the staging arms from `origin/dsv41-staging-sim` (`ram_insert` in `{"miss", "none", "deferred"}`).
- Produces:
  - the CLI flags `--staging-reserve K` and `--cpu-misses`;
  - JSON keys `staging_reserve` and `cpu_misses` per arm;
  - `divix01:/mnt/nvme1/cpu-p1/slot-map-policies.{json,txt}`.

- [ ] **Step 1: Bring the staging arms onto this branch**

```bash
cd /home/dimitri/data/divix/wt-device-slot-map
git fetch origin && git merge --no-ff origin/dsv41-staging-sim -m "merge(sim): staging-buffer RAM insert arms"
```

- [ ] **Step 2: Write the failing tests**

```python
def test_staging_reserve_shrinks_each_row_by_k(tmp_path):
    trace = _tiny_trace(layers=2, experts=16, tokens=40, seed=3)
    full = run_arm(trace, ram_rows=20, ram_insert="deferred", staging_reserve=0)
    reserved = run_arm(trace, ram_rows=20, ram_insert="deferred", staging_reserve=2)
    assert reserved.ram_capacity_per_row == [c - 2 for c in full.ram_capacity_per_row]
    assert reserved.ram_hit_lanes <= full.ram_hit_lanes


def test_cpu_misses_take_the_tail_of_all_residual_lanes():
    # Lanes sorted by key, highest first: hit, miss, hit, miss. Split for n=4 is 3.
    kinds = choose_lane_kinds(hit=[True, False, True, False], split=[0, 1, 1, 2, 3], cpu_misses=True)
    assert kinds == ["copy", "cpu", "cpu", "cpu"]
    kinds = choose_lane_kinds(hit=[True, False, True, False], split=[0, 1, 1, 2, 3], cpu_misses=False)
    assert kinds == ["copy", "nvme", "cpu", "nvme"]  # n = 2 hits, split[2] = 1


def test_cpu_miss_costs_nvme_then_cpu_and_inserts_ram_not_vram():
    trace = _tiny_trace(layers=1, experts=8, tokens=30, seed=5)
    arm = run_arm(trace, ram_rows=4, ram_insert="deferred", cpu_misses=True, split=[0, 1, 1, 2, 3, 3, 4, 5, 5])
    assert arm.vram_inserts_from_cpu_lanes == 0
    assert arm.ram_inserts_from_cpu_misses > 0
```

`_tiny_trace`, `run_arm` and `choose_lane_kinds` are the test-file helper and the sim's entry points. If `run_arm` does not exist under that name, wrap the sim's existing per-arm function, and keep the wrapper in the test file.

- [ ] **Step 3: Run them and confirm they fail**

Run on divix01, in a worktree at the pushed commit:
```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_cpu_expert_sim.py -q -p no:randomly -k "staging_reserve or cpu_miss"; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL, because neither the keyword nor the function exists.

- [ ] **Step 4: Implement**

- **`staging_reserve=K`:** subtract K from every row's RAM capacity before the replay.
- **`cpu_misses=True`:**
  - The eligible set is every residual lane in key order, and the last `split[n]` of them are CPU lanes.
  - A CPU miss costs `nvme_ms + c_cpu` on the CPU path. It is not inserted into VRAM, and it is inserted into RAM under the arm's `ram_insert` rule.
  - A CPU hit is costed as today.

- [ ] **Step 5: Run the tests and confirm they pass**

Use the same command as Step 3. Expected: PASS. Also run the whole file, and `test/manual/dsv41/test_tier_sim.py`, which must stay green.

- [ ] **Step 6: Run the replay**

```
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python scripts/dsv41/cpu_expert_sim.py \
  /data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl \
  --staging --staging-reserve 0,6,8 --cpu-misses both \
  --out /mnt/nvme1/cpu-p1/slot-map-policies.json > /mnt/nvme1/cpu-p1/slot-map-policies.txt
```

Report, for `ram_insert=deferred` × K ∈ {0, 6, 8} × CPU experts {off, on hits-only, on hits+misses}:
- hot hit rate;
- RAM hit rate, as a fraction of lanes;
- NVMe reads per token;
- ms/token.

**Decision rule.** `CPU_EXPERTS_MISSES` defaults on only if hits+misses beats hits-only by more than 0.5 ms/token. The sim's noise is 0, but served A/Bs move about 0.5 ms between identical arms (§27.15).

- [ ] **Step 7: Commit**

```bash
git add scripts/dsv41/cpu_expert_sim.py scripts/dsv41/tier_sim.py test/registered/unit/kernels/test_cpu_expert_sim.py
git commit -m "sim(dsv41): staging reserve per row and CPU-computed NVMe misses"
```

---

### Task 1: Wire v2 constants and their mirrors

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py`, `python/sglang/kernels/ops/moe/expert_lease_block.py`
- Create: `python/sglang/srt/layers/moe/ram_slot_map.py`
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`, `test/registered/unit/kernels/test_ram_slot_map.py` (new)

**Interfaces:**
- Produces:
  - the C++ constants listed below (namespace `sglang::expert_stream::wire`);
  - the Python mirror dict `WIRE` in `expert_stream_transport.py`, keyed by the same names;
  - `ram_slot_map.LaneKind` (IntEnum: `HIT_COPY=1, HIT_SM=2, HIT_CPU=3, MISS_GPU=4, MISS_CPU=5`);
  - `ram_slot_map.type_lanes(experts: list[int], ram_slot: list[int], staging: list[int], split: list[int], *, captured: bool, copy_armed: bool, hit_copy: str, cpu_on: bool, cpu_misses: bool) -> tuple[list[LaneKind], list[int]]`;
  - `ram_slot_map.MapReplica(rows: int, experts: int)` with `.apply_delta(row, tag, staging, entries)`, `.apply_bulk(entries)`, `.ram_slot`, `.staging`, `.map_chain`, `.map_applied`.

- [ ] **Step 1: Write the failing tests**

`test/registered/unit/kernels/test_ram_slot_map.py`:
```python
from sglang.srt.layers.moe.ram_slot_map import LaneKind, MapReplica, type_lanes

SPLIT = [0, 1, 1, 2, 3, 3, 4, 5, 5]


def test_hits_and_misses_take_map_and_staging_slots_in_order():
    kinds, slots = type_lanes(
        [5, 9, 7], ram_slot=[-1] * 5 + [11, -1, -1, -1, -1], staging=[40, 41, -1, -1, -1, -1, -1, -1],
        split=SPLIT, captured=True, copy_armed=True, hit_copy="ce", cpu_on=False, cpu_misses=False,
    )
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_GPU, LaneKind.MISS_GPU]
    assert slots == [11, 40, 41]


def test_cpu_tail_hits_only_and_with_misses():
    ram = [-1] * 10
    ram[1], ram[3] = 21, 23
    args = dict(ram_slot=ram, staging=[50, 51, 52, -1, -1, -1, -1, -1], split=SPLIT,
                captured=True, copy_armed=True, hit_copy="ce", cpu_on=True)
    kinds, _ = type_lanes([1, 2, 3, 4], cpu_misses=False, **args)  # hits are lanes 0 and 2, n=2, split 1
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_GPU, LaneKind.HIT_CPU, LaneKind.MISS_GPU]
    kinds, _ = type_lanes([1, 2, 3, 4], cpu_misses=True, **args)  # n=4, split[4] = 3: lanes 1, 2 and 3
    assert kinds == [LaneKind.HIT_COPY, LaneKind.MISS_CPU, LaneKind.HIT_CPU, LaneKind.MISS_CPU]


def test_eager_or_unarmed_posts_use_sm_and_no_cpu():
    ram = [7, -1]
    for captured, armed in ((False, True), (True, False)):
        kinds, _ = type_lanes([0, 1], ram_slot=ram, staging=[3, -1, -1, -1, -1, -1, -1, -1], split=SPLIT,
                              captured=captured, copy_armed=armed, hit_copy="ce", cpu_on=True, cpu_misses=True)
        assert kinds == [LaneKind.HIT_SM, LaneKind.MISS_GPU]


def test_sm_mode_hits_never_copy_engine():
    kinds, _ = type_lanes([0], ram_slot=[4], staging=[-1] * 8, split=SPLIT, captured=True, copy_armed=True,
                          hit_copy="sm", cpu_on=False, cpu_misses=False)
    assert kinds == [LaneKind.HIT_SM]


def test_replica_applies_a_decode_delta_once():
    m = MapReplica(rows=1, experts=4)
    m.apply_delta(0, tag=2, staging=[9, -1, -1, -1, -1, -1, -1, -1], entries=[(2, 5), (1, -1)])
    m.ram_slot[0][2] = 6  # a bulk entry applied later must survive a re-application attempt
    m.apply_delta(0, tag=2, staging=[9, -1, -1, -1, -1, -1, -1, -1], entries=[(2, 5), (1, -1)])
    assert m.ram_slot[0][2] == 6 and m.map_applied[0] == 2
```

In `test_exl3_ram_miss_device_args.py`, extend the existing mirror test, which parses `constexpr` lines from `lease_layout.h` and compares them with the Python mirror. It must now cover every name in Step 3's header, and must fail when any of them is missing from `WIRE`.

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_ram_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL. The import error comes from `ram_slot_map`, and the mirror test fails on the new names.

- [ ] **Step 3: Rewrite `lease_layout.h`**

Keep its header comment, which says the parser takes only `constexpr <type> kName = <integer expression>;` lines. Replace the body with:

```cpp
// Request page: device-written, host-read.
constexpr int64_t kDemandHead = 0;  // u32: the last posted seq
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kRecordBytes = 256;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;           // u32 seqlock word: 0 while the payload is rewritten, the seq stored last
constexpr int64_t kRecRow = 4;           // u16
constexpr int64_t kRecCount = 6;         // u16: lanes
constexpr int64_t kRecFlags = 8;         // u32
constexpr uint32_t kRecFlagCaptured = 1;
constexpr int64_t kRecChain = 12;        // u32 low half of the row's map-chain number, 0 when no lane misses
constexpr int64_t kRecChainHi = 16;      // u32 high half
constexpr int64_t kRecProtectCount = 20; // u16
constexpr int64_t kRecProtect = 32;      // i32[kMaxIds]: every routed expert of the request
constexpr int64_t kRecLanes = 64;        // kMaxIds lanes of kLaneBytes
constexpr int64_t kLaneBytes = 16;
constexpr int64_t kLaneExpert = 0;       // i32
constexpr int64_t kLaneSlot = 4;         // i32: the RAM slot of a hit, the staging slot of a miss
constexpr int64_t kLaneDst = 8;          // i32: the VRAM destination slot
constexpr int64_t kLaneWeight = 12;      // f32: the lane's routing weight
constexpr int64_t kRecKinds = 192;       // u8[kMaxIds]
constexpr int64_t kPageBytes = kDemandRing + kDemandRecords * kRecordBytes;

constexpr uint32_t kKindHitCopy = 1;
constexpr uint32_t kKindHitSm = 2;
constexpr uint32_t kKindHitCpu = 3;
constexpr uint32_t kKindMissGpu = 4;
constexpr uint32_t kKindMissCpu = 5;

constexpr int64_t kHotHeaderBytes = 8;
constexpr int64_t kHotAlignment = 64;
constexpr uint32_t kHotRecords = kDemandRecords;

// Completion block: host-written, device-read.
constexpr int64_t kLeaseRing = 16;   // == kDemandRecords
constexpr int64_t kLeaseLanes = 8;   // == kMaxIds
constexpr int64_t kLeaseBlockAlign = 4096;
constexpr int64_t kLeasePieceMask = 0;
constexpr int64_t kLeasePieceMaskLineBytes = 128;
constexpr int64_t kLeaseCopyDone = kLeasePieceMask + kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
constexpr int64_t kLeaseCopyDoneBytes = 8;
constexpr int64_t kLeaseCopyGate = kLeaseCopyDone + 128;
constexpr uint32_t kLeaseGateClosed = 0x80000001u;
constexpr uint32_t kLeaseGateOpen = 1;
constexpr uint32_t kLeaseGateSeqShift = 2;
constexpr uint32_t kLeaseGateSeqMask = 0x1FFFFFFF;
constexpr int64_t kCopyArmed = kLeaseCopyGate + 128;  // u32: 1 once the service armed its copy engine
constexpr int64_t kSplit = kCopyArmed + 128;          // i32[kLeaseLanes + 1]: CPU lanes per n eligible lanes
constexpr int64_t kLeaseBlockBytes = 20480;

// Delta block: host-written, device-read, one record per row.
constexpr int64_t kDeltaStride = 256;
constexpr int64_t kDeltaTag = 0;      // u64: the map-chain number this delta follows, stored last with a release
constexpr int64_t kDeltaCount = 8;    // u32
constexpr int64_t kDeltaStaging = 16; // i32[kLeaseLanes]
constexpr int64_t kDeltaEntries = 48; // {i32 expert, i32 slot}[kDeltaMaxEntries]; slot -1 unmaps
constexpr int64_t kDeltaMaxEntries = 16;

static_assert(kRecKinds + kMaxIds <= kRecordBytes, "record");
static_assert(kRecLanes + kMaxIds * kLaneBytes <= kRecKinds, "record lanes");
static_assert(kSplit + 4 * (kLeaseLanes + 1) <= kLeaseBlockBytes, "completion block");
static_assert(kLeaseBlockBytes % kLeaseBlockAlign == 0, "the block is whole pages");
static_assert(kDeltaEntries + 8 * kDeltaMaxEntries <= kDeltaStride, "delta record");
```

The block keeps the `kLease*` names so that `copy_engine.h`'s gate code and the PieceMask readers compile unchanged. Renaming them is not in scope.

- [ ] **Step 4: Update the Python mirrors**

Add or replace every name above in `WIRE` (`expert_stream_transport.py`) and in `expert_lease_block.py`. Delete `kLeaseRowResult*`, `kLeaseTag*`, `kLeaseLaneRequest*`, `kLeaseLr*`, `kLeaseDone*` and `kRecArmed`.

- [ ] **Step 5: Write `ram_slot_map.py`**

```python
"""Reference model of the device slot map: lane typing in the post and delta application (LEASE_PROTOCOL.md)."""
from enum import IntEnum
from typing import Iterable, Sequence

LANES = 8


class LaneKind(IntEnum):
    HIT_COPY = 1
    HIT_SM = 2
    HIT_CPU = 3
    MISS_GPU = 4
    MISS_CPU = 5


def type_lanes(experts, ram_slot, staging, split, *, captured, copy_armed, hit_copy, cpu_on, cpu_misses):
    if len(experts) > LANES:
        raise ValueError(f"a request has at most {LANES} lanes, got {len(experts)}")
    slots, hit, m = [], [], 0
    for e in experts:
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
        else:
            if m >= LANES or staging[m] < 0:
                raise ValueError("a miss lane has no staging slot")
            slots.append(staging[m])
            hit.append(False)
            m += 1
    host_lanes = captured and copy_armed
    eligible = [host_lanes and cpu_on and (h or cpu_misses) for h in hit]
    n = sum(eligible)
    take = split[n] if n else 0
    cpu = [False] * len(experts)
    for j in reversed(range(len(experts))):
        if take == 0:
            break
        if eligible[j]:
            cpu[j] = True
            take -= 1
    kinds = []
    for h, c in zip(hit, cpu):
        if c:
            kinds.append(LaneKind.HIT_CPU if h else LaneKind.MISS_CPU)
        elif h:
            kinds.append(LaneKind.HIT_COPY if host_lanes and hit_copy == "ce" else LaneKind.HIT_SM)
        else:
            kinds.append(LaneKind.MISS_GPU)
    return kinds, slots


class MapReplica:
    def __init__(self, rows: int, experts: int):
        self.ram_slot = [[-1] * experts for _ in range(rows)]
        self.staging = [[-1] * LANES for _ in range(rows)]
        self.map_chain = [1] * rows  # the attach delta has tag 1; a zero-filled delta record never matches
        self.map_applied = [0] * rows

    def apply_delta(self, row: int, tag: int, staging: Sequence[int], entries: Iterable[tuple[int, int]]):
        if self.map_applied[row] == tag:
            return
        for e, s in entries:
            self.ram_slot[row][e] = s
        self.staging[row] = list(staging)
        self.map_applied[row] = tag

    def apply_bulk(self, entries: Iterable[tuple[int, int, int]]):
        for row, e, s in entries:
            self.ram_slot[row][e] = s
```

Keep the reference identical to the CUDA, and test them against each other in Task 5. Both start `map_applied` at -1, so the first post applies the tag-0 attach delta.

- [ ] **Step 6: Run the tests and confirm they pass**

Use the same command as Step 2. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h python/sglang/kernels/ops/moe/expert_stream_transport.py python/sglang/kernels/ops/moe/expert_lease_block.py python/sglang/srt/layers/moe/ram_slot_map.py test/registered/unit/kernels/test_ram_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py
git commit -m "wire(dsv41-slot-map): v2 request record, completion block, per-row delta block; Python reference of lane typing and the map replica"
```

At this commit, code built on the v1 names fails to compile. That is expected, and the next tasks fix it. Do not quote suite counts at this commit, beyond the two files above.

---

### Task 2: The host tier — staging, victims at record time, deltas, no leases

**Files:**
- Modify: `host/ram_tier.h`, `host/tier_protocol.h`, `host/ffi_exports.h`, `host/ffi_test_exports.h`
- Rename and rewrite: `python/sglang/test/dsv41_lease_sim.py` → `python/sglang/test/dsv41_chain_sim.py`
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py` (new)
- Delete: `test_exl3_ram_miss_leases.py`, `_lease_defer.py`, `_lease_publication.py`, `_task5_item5_ack_independence.py`
- Rewrite: `test_exl3_ram_miss_lease_service.py` → `_record_service.py`; `_lease_thread.py` → `_record_thread.py`; `_lease_wrap.py` → `_wrap.py`. Keep each file's wrap, lap and fail-stop cases, adapted to v2 records.

**Interfaces:**
- Consumes: Task 1's constants; `MapReplica` and `type_lanes` (the sim uses them to play the device).
- Produces:
  - **C++:**
    - `RamTier::handle_record(int64_t row, const Record& r, uint64_t gen)`;
    - `int32_t take_victim_locked(int64_t row, const FixedVec<int32_t, 16>& wanted)`, returning -1 for none;
    - `void publish_delta_locked(int64_t row, uint64_t tag, const DeltaBuilder& d)`;
    - the slot state `kStaging`;
    - counters `kRamInsertSkipped` and `kLapped` (kept).
  - **FFI test exports:** `expert_stream_delta_record(handle, row) -> (tag, staging[8], entries[n,2])` and `expert_stream_staging(handle, row) -> int32[8]`.
  - **FFI production exports:** `expert_stream_delta_block(handle) -> int64` (its address) and `expert_stream_take_bulk_delta(handle) -> int32[n,3]`.
  - **Python:** `ChainSim(host, rows, experts, capacity, k)` with:
    - `.post(row, experts, protect, captured=True, armed=False, hit_copy="ce", cpu_on=False, cpu_misses=False, split=None) -> int` (the seq; it applies the pending delta and types the lanes through `ram_slot_map`);
    - `.wait_served(seq)` (it plays S: every `MISS_GPU` lane complete);
    - `.copy_wait(seq)` (it plays CW and CC);
    - `.replica` (a `MapReplica`).

- [ ] **Step 1: Write the failing tests**

`test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py`, using the existing `host` fixtures from `python/sglang/test/dsv41_ram_miss_fixtures.py` (row images, 1 row, capacity 6, 16 experts, K = 2). Read that file first for the fixture names, and reuse them.

```python
def test_attach_reserves_k_staging_slots_and_publishes_tag_zero(chain_sim):
    tag, staging, entries = chain_sim.delta(0)
    assert tag == 1 and entries == [] and sorted(s for s in staging if s >= 0) == [0, 1]


def test_miss_reads_into_staging_and_delta_maps_it_next_chain(chain_sim):
    seq = chain_sim.post(0, experts=[3], protect=[3])
    chain_sim.wait_served(seq)
    tag, staging, entries = chain_sim.delta(0)
    assert tag == 2
    assert (3, 0) in entries                 # expert 3 now lives in the first staging slot
    assert 0 not in staging and len([s for s in staging if s >= 0]) == 2
    assert chain_sim.read_slot(0, 0) == chain_sim.image(0, 3)  # the bytes landed


def test_delta_published_before_service_returns(chain_sim):
    seq = chain_sim.post(0, experts=[4, 5], protect=[4, 5])
    chain_sim.wait_served(seq)
    assert chain_sim.delta(0)[0] == chain_sim.last_chain(0)  # never behind once the chain is served


def test_hit_after_insert_uses_the_ram_slot(chain_sim):
    chain_sim.wait_served(chain_sim.post(0, experts=[3], protect=[3]))
    seq = chain_sim.post(0, experts=[3], protect=[3], captured=False)
    assert chain_sim.kinds(seq) == ["HIT_SM"] and chain_sim.slots(seq) == [0]


def test_victim_is_never_routed_hot_or_staging(chain_sim):
    for e in range(4):  # capacity 6, K 2: the four mappable slots now hold 0..3, 0 the oldest
        chain_sim.wait_served(chain_sim.post(0, experts=[e], protect=[e]))
    chain_sim.set_gpu_hot(0, [0])  # the oldest is VRAM-hot, 1 is routed: the victim must be 2
    seq = chain_sim.post(0, experts=[9], protect=[9, 1])
    chain_sim.wait_served(seq)
    _, staging, entries = chain_sim.delta(0)
    assert [e for e, s in entries if s == -1] == [2]


def test_no_victim_skips_ram_insert(chain_sim):
    # capacity 6, K 2: four mapped experts, all hot.
    for e in range(4):
        chain_sim.wait_served(chain_sim.post(0, experts=[e], protect=[e]))
    chain_sim.set_gpu_hot(0, [0, 1, 2, 3])
    before = chain_sim.delta(0)[1]
    seq = chain_sim.post(0, experts=[7], protect=[7])
    chain_sim.wait_served(seq)
    tag, staging, entries = chain_sim.delta(0)
    assert entries == [] and staging == before
    assert chain_sim.counter("ram_insert_skipped") == 1


def test_failed_read_aborts_with_fatal_and_no_delta(chain_sim_script):
    out = chain_sim_script("fault_read=1; seq = sim.post(0, experts=[3], protect=[3]); sim.wait_served(seq)")
    assert_aborted(out, "FATAL")
```

`chain_sim`, `chain_sim_script` and `assert_aborted` follow the existing fixture pattern: `run_host_script` and `assert_aborted` in `dsv41_ram_miss_fixtures.py`. The script fixture runs in a subprocess that must die of SIGABRT.

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL. `dsv41_chain_sim` does not exist and the build fails on v1 names.

- [ ] **Step 3: Implement the record read**

In `tier_protocol.h`:
- replace `read_record` and `read_lane_request` with one seqlocked `read_record_v2(page, idx, Record&)`;
- give `Record` the fields `seq, row, count, flags, chain, protect_count, protect[8], lanes[8]{expert, slot, dst, weight}, kind[8]`;
- validate kinds in `1..5` and `count <= 8`. Anything else is fail-stop: "request G: malformed record".

- [ ] **Step 4: Implement the tier state**

In `ram_tier.h`:
- **Delete:**
  - `leases`, `Outstanding`, `outstanding_`, `open_lease_entry_locked`, `retire_leases`, `release_lease_locked`;
  - `lease_changes_`, `defers`, `census_locked`, `grant_lanes_locked`, `choose_cpu_lanes_locked`;
  - the `demand_done` store in `pump_demand` (:288-289), and every `kDeferred*`/`kLease*` counter.
- **Add** `kStaging` to the slot states. `attach_row` takes K free slots, marks them STAGING, and publishes `tag 0`.
- **`take_victim_locked`** is `take_slot_locked` (:1538-1572) without the lease and `filling` arms. It never returns STAGING. It returns -1 instead of failing.
- **`handle_record`** follows the Design's host steps 2–6. It publishes the delta before it issues the first read.
- **`publish_delta_locked`:**
  - write `count`, `staging[8]` and the entries with plain stores;
  - `_mm_sfence()`;
  - `__atomic_store_n(tag, value, __ATOMIC_RELEASE)`.
  - Also update `slot_map` (the host mirror) with the existing `publish_map` for each entry.

- [ ] **Step 5: Rewrite the sim**

Rewrite `dsv41_lease_sim.py` as `dsv41_chain_sim.py`. It plays the device on wire v2:
- **post:**
  - apply `delta[row]` into `self.replica` when `tag == map_chain[row]`, or raise "delta behind";
  - `type_lanes`;
  - write the record and `demand_head`.
- **`wait_served`** polls PieceMask for `MISS_GPU` lanes.
- **`copy_wait`** acquires `CopyDone == G`.

Update every importer of the old module. Grep for `dsv41_lease_sim`.

- [ ] **Step 6: Run the tests and confirm they pass**

Use the same command as Step 2, plus the rewritten `_record_service`, `_record_thread` and `_wrap` files. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/test/dsv41_chain_sim.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_record_service.py test/registered/unit/kernels/test_exl3_ram_miss_record_thread.py test/registered/unit/kernels/test_exl3_ram_miss_wrap.py
git rm python/sglang/test/dsv41_lease_sim.py test/registered/unit/kernels/test_exl3_ram_miss_leases.py test/registered/unit/kernels/test_exl3_ram_miss_lease_defer.py test/registered/unit/kernels/test_exl3_ram_miss_lease_publication.py test/registered/unit/kernels/test_exl3_ram_miss_task5_item5_ack_independence.py test/registered/unit/kernels/test_exl3_ram_miss_lease_service.py test/registered/unit/kernels/test_exl3_ram_miss_lease_thread.py test/registered/unit/kernels/test_exl3_ram_miss_lease_wrap.py
git commit -m "host(dsv41-slot-map): records v2, per-row staging slots, victims at record time, map deltas; leases and deferral removed"
```

---

### Task 3: The copy thread on records, and pause without leases

**Files:**
- Modify: `host/copy_engine.h`, `host/ram_tier.h` (`copy_completed`, `drain_copy_completions`), `host/ram_thread.h` (`pause`)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py` (rewrite its lease assertions), `test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py` (add cases)

**Interfaces:**
- Consumes: `Record` and `handle_record` from Task 2.
- Produces:
  - `CopyJob` gains `late_cpu` (int) and loses its lease fields;
  - `CopyEngine::submit_late_cpu(uint64_t gen, int32_t slot, float weight)`;
  - `ram_thread pause()` returns 2 while a copy job is outstanding.

- [ ] **Step 1: Write the failing tests**

```python
def test_copy_lanes_issue_from_the_record_and_publish_copy_done(chain_sim_ce):
    chain_sim_ce.wait_served(chain_sim_ce.post(0, experts=[3], protect=[3]))  # 3 enters RAM
    seq = chain_sim_ce.post(0, experts=[3], protect=[3], captured=True, armed=True)
    assert chain_sim_ce.kinds(seq) == ["HIT_COPY"]
    chain_sim_ce.copy_wait(seq)
    assert chain_sim_ce.dst_bytes(seq, lane=0) == chain_sim_ce.image(0, 3)


def test_lapped_hit_only_records_do_not_stall(chain_sim_sm):
    chain_sim_sm.wait_served(chain_sim_sm.post(0, experts=[3], protect=[3]))
    chain_sim_sm.hold_service()
    for _ in range(40):  # past the 16-record ring with no host wait
        chain_sim_sm.post(0, experts=[3], protect=[3], captured=True, armed=True, hit_copy="sm")
    chain_sim_sm.release_service()
    seq = chain_sim_sm.post(0, experts=[8], protect=[8])
    chain_sim_sm.wait_served(seq)  # the next miss chain is served normally
    assert chain_sim_sm.counter("lapped") > 0


def test_pause_refuses_while_a_copy_job_is_outstanding(chain_sim_ce):
    chain_sim_ce.wait_served(chain_sim_ce.post(0, experts=[3], protect=[3]))
    chain_sim_ce.block_copies()
    chain_sim_ce.post(0, experts=[3], protect=[3], captured=True, armed=True)
    assert chain_sim_ce.host.pause_once() == 2
    chain_sim_ce.unblock_copies()
```

`chain_sim_ce` and `chain_sim_sm` are `chain_sim` with the copy engine on and armed, using the `faulty`/ballast backend that `test_exl3_ram_miss_copy_engine.py` already uses for host-only copies.

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL.

- [ ] **Step 3: Implement**

- **`copy_engine.h`:**
  - `CopyLane` comes from the record lanes;
  - `issue` (:621-668) is unchanged except for its source;
  - in `run` (:500-575), the completion also requires `job.late_arrived == job.late_cpu`, with each late job's `cpu->done(seq)`.
- **Delete** `copy_acked` and `release_copied_owned`. `drain_copy_completions` only recycles jobs.
- **`pause`** checks the service is idle and the copy ring is empty, in place of `graph_leases_outstanding`.

- [ ] **Step 4: Run the tests and confirm they pass**

Same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py
git commit -m "host(dsv41-slot-map): copy jobs straight from records, late CPU jobs, pause on idle; copy-side lease release removed"
```

---

### Task 4: Eager host use and the bulk delta

**Files:**
- Modify: `host/ram_tier.h` (`assign`, `take_admit_slot_locked`, `fill_begin`, the bulk list), `host/ffi_exports.h`
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`after_host_use`)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py`, `test_exl3_ram_miss_prefill_share.py`, `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`, `test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py`

**Interfaces:**
- Consumes: `kStaging`, `publish_map`, `MapReplica.apply_bulk`.
- Produces:
  - `expert_stream_take_bulk_delta(handle) -> int32 [n, 3]` (`row, expert, slot`), which clears the list;
  - `Exl3RamMissService._apply_bulk_delta()`, called from `after_host_use` when the pause depth reaches 0, before `host.resume()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_admission_never_takes_staging(chain_sim):
    staging = {s for s in chain_sim.staging(0) if s >= 0}
    chain_sim.host.pause_blocking()
    for e in range(10):
        chain_sim.host.assign(0, e, protect=[], fallback=False)  # eager admissions past the tier's size
    assert not staging & {s for s in chain_sim.host.mapping_row(0) if s >= 0}
    chain_sim.host.resume()


def test_bulk_after_pending_decode_delta(chain_sim):
    seq = chain_sim.post(0, experts=[3], protect=[3])
    chain_sim.wait_served(seq)  # decode delta tag 1 is published, not yet applied
    chain_sim.host.pause_blocking()
    chain_sim.host.assign(0, 5, protect=[], fallback=False)
    bulk = chain_sim.host.take_bulk_delta()
    chain_sim.host.resume()
    chain_sim.apply_bulk_like_device(bulk)  # pending decode delta first, then bulk
    assert chain_sim.replica.ram_slot[0] == chain_sim.host.mapping_row(0)
```

`apply_bulk_like_device` in `ChainSim` mirrors `map_bulk_apply`:
- for each row, `replica.apply_delta(row, *delta(row))` when `tag == map_chain[row]`;
- then `replica.apply_bulk(bulk)`.

- [ ] **Step 2: Run them and confirm they fail**

```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL.

- [ ] **Step 3: Implement**

- **`ram_tier.h`:**
  - `take_admit_slot_locked` (:1505-1535) and `assign` (:373-391) skip `kStaging`;
  - every `publish_map(row, expert, slot)` called outside `handle_record` appends `{row, expert, slot}` to `bulk_`, under the tier lock;
  - add the export `take_bulk_delta`.
- **`exl3_ram_miss.py`, `after_host_use`:**
  - at depth 0, call `bulk = self.host.take_bulk_delta()`;
  - if `bulk.numel()`, call `self.device_side.map_bulk_apply(bulk)` (Task 5's launcher) on the current stream;
  - then `self.host.resume()`.

  Until Task 5 lands, `device_side.map_bulk_apply` is the Python fallback `MapReplica.apply_bulk` against the CPU replica. Only CPU tests run before Task 5.

- [ ] **Step 4: Run the tests and confirm they pass**

Same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/srt/layers/moe/exl3_ram_miss.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py
git commit -m "host(dsv41-slot-map): eager admissions skip staging slots and queue a bulk map delta applied before resume"
```

---

### Task 5: Device — post v2 (delta apply and typing), `map_bulk_apply`, S and CW on lane kinds; W1 removed

**Files:**
- Modify: `lease_device.cuh`, `lease_kernels.cuh`, `row_copy_kernels.cuh`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (device bank, `post` args, `map_bulk_apply`; delete `hit_wait`), `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`Exl3RamMissRowBackend.post`, attach)
- Test (GPU, manual): `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py` (new), `test/manual/dsv41/lease_chain_rig.py` (rewrite), `test_exl3_lease_kernels_cuda.py` (rewrite), `test_exl3_lease_ordering_cuda.py` (keep the ordering cases that still apply: SM reads before the gate, captured node signature, first armed replay, module load during an armed wait; delete the lease-release cases)
- Test (CPU): `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py` (the chain shape is post → C1 → S → CW → wait → CC, with no W1)

**Interfaces:**
- Consumes: Task 1's wire and `type_lanes`; Task 2's delta block address (`expert_stream_delta_block`) and staging.
- Produces:
  - `ExpertStreamDevice.map_bank` with `ram_slot`, `staging`, `map_chain`, `map_applied`, `lane_kind`, `lane_slot`;
  - `ExpertStreamDevice.post(row, planned, count, routes, dst_slots, hot_slots, hot_capacity, *, captured, cpu_input, hit_copy, cpu_on, cpu_misses)`;
  - `ExpertStreamDevice.map_bulk_apply(bulk: torch.Tensor[int32, n, 3])`.

- [ ] **Step 1: Write the failing CPU shape test**

In `test_exl3_ram_miss_device_args.py`, replace the CW, stream-wait and CC shape test with:
```python
def test_chain_has_no_hit_wait_and_post_writes_c1_compaction(backend_mock):
    calls = backend_mock.record_post()
    assert [c.name for c in calls] == ["post", "c1", "stream", "copy_wait"]
    assert "hit_wait" not in dir(backend_mock.device_side)
```

`backend_mock` is the existing fixture in that file that records `device_side` calls. Extend it if needed.

- [ ] **Step 2: Write the failing GPU tests**

`test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`:
```python
@pytest.mark.parametrize("cpu_misses", [False, True])
@pytest.mark.parametrize("hit_copy", ["ce", "sm"])
def test_post_types_lanes_like_the_reference(rig, hit_copy, cpu_misses):
    rng = random.Random(7)
    for _ in range(200):
        experts = rng.sample(range(rig.experts), rng.randint(1, 6))
        rig.set_map_random(rng)  # random ram_slot row and staging list through the bulk path
        got = rig.post_and_read_kinds(experts, captured=True, armed=True, hit_copy=hit_copy,
                                      cpu_on=True, cpu_misses=cpu_misses)
        want = type_lanes(experts, rig.ram_slot_row(), rig.staging_row(), rig.split, captured=True,
                          copy_armed=True, hit_copy=hit_copy, cpu_on=True, cpu_misses=cpu_misses)
        assert got == want


def test_post_applies_the_pending_delta_once(rig):
    rig.publish_delta(row=0, tag=1, staging=[2, 3], entries=[])
    rig.post_and_read_kinds([5])                       # a miss: map chain 2
    rig.publish_delta(row=0, tag=2, staging=[3, 4], entries=[(5, 2)])
    kinds, slots = rig.post_and_read_kinds([5])
    assert kinds == [LaneKind.HIT_SM] and slots == [2]
    rig.bulk_apply([(0, 5, 6)])
    assert rig.post_and_read_kinds([5])[1] == [6]      # tag 2 is not re-applied over the bulk entry


def test_post_waits_for_delta_then_traps_at_deadline(rig_subprocess):
    out = rig_subprocess("rig.post_and_read_kinds([5]); rig.post_and_read_kinds([6], timeout_ms=200)")
    assert_device_trapped(out)  # the second post found tag 1 against map_chain 2


def test_miss_streams_from_the_staging_slot_byte_exact(rig):
    kinds, slots = rig.post_and_read_kinds([9])
    rig.serve()  # the real host reads expert 9 into slots[0]
    assert rig.vram_dst_bytes(lane=0) == rig.image(9)
```

`rig` is the rewritten `lease_chain_rig.py`: a real host, an `ExpertStreamDevice` and the production backend driving `backend.post`. Keep its constructor arguments, and add `set_map_random`, `post_and_read_kinds`, `publish_delta` (through a test export that writes the delta block directly), `bulk_apply`, `serve`, `vram_dst_bytes` and `image`. `assert_device_trapped` checks that the subprocess died with a CUDA "unspecified launch failure" / illegal instruction, as the existing S-deadline test does at `test_exl3_lease_kernels_cuda.py:130`. Reuse its helper.

- [ ] **Step 3: Run them and confirm they fail**

CPU:
```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```

GPU, on divix01:
```
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python SGLANG_EXL3_SRC=<as the existing manual suites> /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL for both.

- [ ] **Step 4: Implement the device helpers in `lease_device.cuh`**

```cuda
// The host publishes a map chain's delta before it serves the chain; this waits only when the host fell behind.
__device__ inline void apply_map_delta(const uint8_t* delta_row, int32_t* ram_slot_row, int32_t* staging_row,
                                       const int64_t* map_chain, int64_t* map_applied, int64_t row,
                                       int32_t experts, int32_t row_capacity, uint64_t deadline) {
  const uint64_t want = static_cast<uint64_t>(map_chain[row]);
  while (ld_acquire_sys64(delta_row + wire::kDeltaTag) != want) {
    if (globaltimer_ns() > deadline) __trap();
  }
  if (static_cast<uint64_t>(map_applied[row]) == want) return;
  const uint32_t n = ld_relaxed_sys<uint32_t>(delta_row + wire::kDeltaCount);
  if (n > wire::kDeltaMaxEntries) __trap();
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t e = ld_relaxed_sys<int32_t>(delta_row + wire::kDeltaEntries + 8 * i);
    const int32_t s = ld_relaxed_sys<int32_t>(delta_row + wire::kDeltaEntries + 8 * i + 4);
    if (e < 0 || e >= experts || s < -1 || s >= row_capacity) __trap();
    ram_slot_row[e] = s;
  }
  for (int k = 0; k < wire::kLeaseLanes; ++k)
    staging_row[k] = ld_relaxed_sys<int32_t>(delta_row + wire::kDeltaStaging + 4 * k);
  map_applied[row] = static_cast<int64_t>(want);
}
```

Use the helpers that already exist in `lease_device.cuh` (`ld_acquire_sys64`, the globaltimer read used for S's deadline). Use their exact names; if one differs, use the existing one.

`type_lanes` is the CUDA transcription of `ram_slot_map.type_lanes`, with the same loop order, the same reversed tail walk, and a trap instead of `ValueError`. It writes `lane_kind[j]`, `lane_slot[j]` and the C1 compaction.

- [ ] **Step 5: Implement the post**

In `lease_kernels.cuh`, `PostParams` gains:
- `delta`, `ram_slot`, `staging`, `map_chain`, `map_applied`, `lane_kind`, `lane_slot`;
- `split` (the pinned completion-block address), `copy_armed`;
- `hit_copy_ce`, `cpu_on`, `cpu_misses`, `row_capacity`, `experts`;
- `go_1`, `host_rows_1`, `dst_slots_1`.

Thread 0 runs, in order:
1. `apply_map_delta`, with the S deadline;
2. `type_lanes`;
3. the map-chain count;
4. the record write, as v2 fields behind the seqlock;
5. the `demand_head` release.

The CPU input staging is unchanged, except that it is skipped when no lane is a CPU lane. Delete `exl3_ram_miss_lease_hit_wait_kernel` (W1) and its launcher.

Add `map_bulk_apply_kernel`, one warp:
- for each row, the same `apply_map_delta`, with a zero-wait deadline: the tag must already equal `map_chain`, or nothing is pending;
- then entries `[n,3]`.

- [ ] **Step 6: Implement S and CW**

In `row_copy_kernels.cuh`:
- **S** owns lanes with `lane_kind == kKindMissGpu`, sourced from `lane_slot[j]`. It ends when every owned lane's PieceMask word is `G << 8 | full`. It traps at its deadline, or on a mask that names G with bits outside the lane's piece set.
- Delete S's `demand_done` and RowResult reads.
- **CW**'s masks come from `lane_kind`. Delete the `Done` store and its `__threadfence_system`, except the one before the gate close, which the Dekker pair needs.

- [ ] **Step 7: Implement the Python side**

- **`ExpertStreamDevice`:**
  - allocate `map_bank` at attach: `ram_slot` from the host's `slot_map` (one copy, at attach only), `staging` = -1, `map_chain` = 0, `map_applied` = -1, so the first post applies the tag-0 delta;
  - pass the new params;
  - delete `hit_wait`.
- **`Exl3RamMissRowBackend.post` (RM:443-476):** drop `side.hit_wait(...)`. The chain becomes `side.post(...)`, then `copy_expert_row_segments_gpu(...)`, then `side.stream(...)`, then `side.copy_wait(...)`.

- [ ] **Step 8: Run the tests and confirm they pass**

Use the Step 3 commands, plus the rewritten `test_exl3_lease_kernels_cuda.py` and `test_exl3_lease_ordering_cuda.py`. Expected: PASS. Record the counts and the commands.

- [ ] **Step 9: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh python/sglang/kernels/ops/moe/expert_stream_transport.py python/sglang/srt/layers/moe/exl3_ram_miss.py test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/lease_chain_rig.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_lease_ordering_cuda.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py
git commit -m "device(dsv41-slot-map): post applies the row's map delta and types lanes from the device map; S and CW on lane kinds; W1 removed"
```

---

### Task 6: CPU-computed NVMe misses, the switchable hit copy, env vars and gates

**Files:**
- Modify: `host/ram_tier.h` (the `LateCpu` send), `host/copy_engine.h`, `python/sglang/srt/layers/moe/cpu_experts/service.py` (the pinned split table, read by the device), `python/sglang/srt/environ.py`, `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py`, `test/registered/unit/test_expert_stream_requirements_exl3.py`, `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`

**Interfaces:**
- Produces:
  - `SGLANG_DSV41_RAM_HIT_COPY`: `"ce"` or `"sm"`, default `"ce"`;
  - `SGLANG_DSV41_CPU_EXPERTS_MISSES`: bool, default from Task 0's decision;
  - `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US`: warns and is ignored;
  - `CpuExpertService.split_pinned`, int32 `[9]` pinned, handed to `ExpertStreamDevice` at attach.

- [ ] **Step 1: Read `.claude/skills/env-var-conventions/SKILL.md`**

- [ ] **Step 2: Write the failing tests**

```python
def test_cpu_miss_runs_after_its_pieces_and_completes_copy_done(chain_sim_cpu):
    seq = chain_sim_cpu.post(0, experts=[3], protect=[3], captured=True, armed=True, cpu_on=True,
                             cpu_misses=True, split=[0, 1, 1, 2, 3, 3, 4, 5, 5])
    assert chain_sim_cpu.kinds(seq) == ["MISS_CPU"]
    chain_sim_cpu.copy_wait(seq)
    assert chain_sim_cpu.cpu_out(0) == chain_sim_cpu.reference_forward(0, slots=chain_sim_cpu.slots(seq))
    assert chain_sim_cpu.cpu_job_started_after_last_piece(seq)


def test_cpu_miss_read_failure_aborts_before_copy_done(chain_sim_script):
    out = chain_sim_script(
        "fault_read=1; seq = sim.post(0, experts=[3], protect=[3], captured=True, armed=True, cpu_on=True, "
        "cpu_misses=True); sim.copy_wait(seq)"
    )
    assert_aborted(out, "FATAL")
    assert "copy_done" not in out.stdout  # the sim prints copy_done only when it acquired G


def test_cpu_misses_needs_cpu_experts(gate):
    with pytest.raises(ValueError, match="SGLANG_DSV41_CPU_EXPERTS_MISSES"):
        gate(env={"SGLANG_DSV41_CPU_EXPERTS_MISSES": "1"})


def test_hit_wait_us_warns_and_is_ignored(caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_DSV41_RAM_MISS_HIT_WAIT_US", "50")
    import importlib, sglang.srt.environ as environ
    importlib.reload(environ)
    assert "HIT_WAIT_US" in caplog.text
```

`chain_sim_cpu` uses the stub CPU forward that `test_exl3_ram_miss_cpu_experts.py` already registers. Read that file and reuse its trait.

In the GPU file `test_exl3_cpu_lane_order_cuda.py`, add `test_cpu_tail_includes_misses_when_enabled`. It extends the existing test at :62: with the flag on, the lowest-key lane is a miss, it becomes `MISS_CPU`, it stays unmapped in VRAM, and it appears in the next delta as a RAM insert.

- [ ] **Step 3: Run them and confirm they fail**

```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/test_expert_stream_requirements_exl3.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL.

- [ ] **Step 4: Implement**

- **Host:**
  - when a `kMissCpu` lane's reads have all landed (`publish_landed` sees the lane's last piece), `handle_record` calls `copy_engine_->submit_late_cpu(gen, slot, weight)`;
  - the copy thread submits a one-lane `CpuJob` and counts it against `job.late_cpu`.
- **Python:**
  - `CpuExpertService` allocates `split_pinned` and writes the split there, both at start and on retune (one int32 store per entry);
  - the device reads it at every post.
- **Env and gate:**
  - add `SGLANG_DSV41_RAM_HIT_COPY` and `SGLANG_DSV41_CPU_EXPERTS_MISSES` per the skill;
  - `CPU_EXPERTS_MISSES` requires `SGLANG_DSV41_CPU_EXPERTS=1`, and the gate lists it with the other CPU-expert prerequisites;
  - `RAM_HIT_COPY=sm` with CPU experts on is allowed, because CPU completion still uses the copy thread and the gate;
  - `HIT_WAIT_US` joins the warn-and-ignore list.

- [ ] **Step 5: Run the tests and confirm they pass**

Use the same command, plus the GPU file under `cc-gpu.lock`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h python/sglang/srt/layers/moe/cpu_experts/service.py python/sglang/srt/environ.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/test_expert_stream_requirements_exl3.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py
git commit -m "cpu-experts(dsv41-slot-map): CPU-computed NVMe misses, a device-read split table, switchable hit copy; HIT_WAIT_US ignored"
```

---

### Task 7: End-to-end graph replays and the replica check

**Files:**
- Modify: `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`, `test/manual/dsv41/test_exl3_copy_engine_cuda.py`
- Test: as above

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("hit_copy", ["ce", "sm"])
def test_device_map_matches_host_after_mixed_replay(graph_rig, hit_copy):
    graph_rig.capture(hit_copy=hit_copy)
    rng = random.Random(11)
    for step in range(400):
        graph_rig.replay(routes=graph_rig.random_routes(rng))
        if step % 50 == 25:
            graph_rig.eager_prefill(rng)  # admissions while paused, applied as a bulk delta
    torch.cuda.synchronize()
    for row in range(graph_rig.rows):
        assert graph_rig.device_ram_slot(row) == graph_rig.host_mapping(row)
        assert set(graph_rig.device_staging(row)) == set(graph_rig.host_staging(row))


def test_replay_outputs_match_eager_reference(graph_rig):
    graph_rig.capture(hit_copy="ce")
    for routes in graph_rig.fixed_route_sequence(64):
        assert torch.equal(graph_rig.replay_output(routes), graph_rig.eager_reference(routes))
```

`graph_rig` is the existing fixture of `test_exl3_ram_miss_graph_gpu.py`, the captured MoE end to end. Keep its construction, and add `device_ram_slot`, `host_mapping`, `device_staging`, `host_staging` and `eager_prefill`.

- [ ] **Step 2: Run them and confirm they fail**

```
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python SGLANG_EXL3_SRC=<as the existing manual suites> /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_copy_engine_cuda.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: FAIL. The new helpers do not exist yet.

- [ ] **Step 3: Implement the rig helpers**

Implement the helpers, then fix whatever the end-to-end run turns up. Any fix goes in the task that owns the code, as a new commit.

- [ ] **Step 4: Run the full suites and confirm they pass**

- **CPU:** `test/registered/unit/kernels` plus `test/registered/unit/scripts/test_cpu_experts_quality.py`. Compare the counts with a run at the merge base `59cfb07c99`, and explain the delta as the files deleted and added.
- **GPU:** every `test/manual/dsv41/*_cuda.py` and `*_gpu.py` this plan touched, plus `test_expert_route_plan_fused.py` and `test_expert_residency_gpu.py`.

Record every command and count.

- [ ] **Step 5: Run the mutants**

Run them in a private divix01 worktree. After each one, revert it and re-run green.
- **The post ignores the delta tag** (applies without waiting). Expected caught by: `test_post_applies_the_pending_delta_once` and the replica test.
- **`map_applied` is not checked.** Expected caught by: the bulk-survives case.
- **The host victim may be a staging slot.** Expected caught by: `test_victim_is_never_routed_hot_or_staging` and the replica test.
- **`type_lanes` takes the head instead of the tail.** Expected caught by: the parity test and `test_cpu_lane_order`.
- **The copy completion ignores `late_cpu`.** Expected caught by: `test_cpu_miss_runs_after_its_pieces...`.

Record each mutant's failing tests.

- [ ] **Step 6: Commit**

```bash
git add test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_copy_engine_cuda.py
git commit -m "test(dsv41-slot-map): captured replays keep the device map equal to the host's, eager prefill included; outputs match eager"
```

---

### Task 8: Protocol document and the reference

**Files:**
- Rewrite: `analysis/dsv41-drive/LEASE_PROTOCOL.md`. Keep the file name; its title becomes "The slot-map protocol".
- Modify: `DSV41_REFERENCE.md`. Add a new §31 before `## Sources`.

- [ ] **Step 1: Rewrite the protocol document**

Rewrite `LEASE_PROTOCOL.md` from this plan's Design section, for the code as built. Cover:
- the parties and wire v2, with the offsets from `lease_layout.h`;
- the chain;
- the host per record;
- deltas and bulk deltas;
- why it is safe without leases;
- the copy engine and gate, unchanged except the masks;
- PDL, shutdown, fail-stop and tests.

Also fix the discrepancy the host map found: copy-thread and CPU-thread aborts can happen after a chain's pieces are served.

- [ ] **Step 2: Add §31 to the reference**

Cover:
- what changed;
- Task 0's replay table, with its decision;
- the suite counts with their commands;
- the mutants;
- what is not done: no served A/B, no quality gate, and an `SGLANG_DSV41_RAM_HIT_COPY=sm` arm that has never been measured.

- [ ] **Step 3: Have a separate fact-check agent check every figure**

The agent checks every figure against its source, as was done for §30. Apply its fixes.

- [ ] **Step 4: Commit**

```bash
git add analysis/dsv41-drive/LEASE_PROTOCOL.md DSV41_REFERENCE.md
git commit -m "docs(dsv41-slot-map): the slot-map protocol as built; DSV41_REFERENCE §31"
```

---

## After the plan (not tasks; each needs the user's go-ahead)

1. **Served smoke** of the production recipe, using `analysis/dsv41-drive/lease-minimal/smoke.sh`, adapted.
2. **Decode A/B** against master: `RAM_HIT_COPY=ce` and `=sm`, CPU experts on, and `CPU_EXPERTS_MISSES` on and off.
3. **Quality gate** (E31) for CPU-computed misses.
4. **Update the "Lease Chain Map" artifact** to the new protocol.
