# Chain PDL skeleton: divix01 results (RTX 5090, sm_120f, CUDA 13.4, PCIe Gen3 x16)

Run 2026-09-27 at commit 9e0a536251 in the private worktree `wt-xfer`. `sglang.__file__` was
`/data/models/slang/nvfp4-work/wt-xfer/python/sglang/__init__.py`. 200 timed replays after 20 warm-up replays, CUDA
events, median. Command: `README.md`, "Run". The script exited 0, and the every-word-reads-R check passed in all
three modes, so PDL broke no edge.

## Records (`skeleton.jsonl`)

| mode | work_ns | replay us p50 | per layer us p50 |
|---|---:|---:|---:|
| 0 (no PDL) | 0 | 242.35 | 6.059 |
| 1 (PDL, implicit trigger) | 0 | 211.42 | 5.286 |
| 2 (PDL, early trigger) | 0 | 185.81 | 4.645 |
| 0 (no PDL) | 2000 | 953.54 | 23.838 |
| 1 (PDL, implicit trigger) | 2000 | 922.66 | 23.066 |
| 2 (PDL, early trigger) | 2000 | 894.27 | 22.357 |

## Saving against mode 0

The per-step figure is for the 40-layer step. The share is of the untraced 66.8 ms/token decode step.

| mode | work_ns | per layer (us) | per step (us) | share of 66.8 ms |
|---|---:|---:|---:|---:|
| 1 (implicit trigger) | 0 | 0.773 | 30.9 | 0.046% |
| 2 (early trigger) | 0 | 1.413 | 56.5 | 0.085% |
| 1 (implicit trigger) | 2000 | 0.772 | 30.9 | 0.046% |
| 2 (early trigger) | 2000 | 1.482 | 59.3 | 0.089% |

Mode 1 and mode 2 are faster than mode 0 at both work levels, as expected. The saving does not depend on `work_ns`:
PDL overlaps launch latency, not the stage bodies. It comes to about 0.1 us per edge with the implicit trigger and
about 0.18 us with the early one, over the 8 PDL edges of a layer.

This is a **launch-latency-only saving**, not a strict upper bound. Every skeleton stage runs `griddepcontrol.wait`
as its first instruction, so there is no prologue for PDL to overlap. Real stages do prologue work before their
dependent read (parameter loads, address setup, CW's mbarrier init), and a production PDL placement could overlap
that. The margin to the gate is 0.52 us per layer, about 65 ns per edge over 8 edges. If the real prologues average
more than about 65 ns before the wait, the real saving could clear 2 us/layer. Against that, the real stages also
wait on host flags, which PDL cannot shorten. A skeleton mode with 200-500 ns of pre-wait work would measure that
sensitivity; it was not run.

## Task 8 gate

Rule: run Task 8 if the saving is >= 2 us per layer in mode 1 or mode 2 at `work_ns` 2000.

Measured: mode 1 saves 0.772 us per layer and mode 2 saves 1.482 us per layer. Both are below 2 us, so **do not run
Task 8**. Part B stops here on divix01.

Decision-table row "PDL on the chain": **no (bound below threshold)**, subject to the prologue caveat above. The best case, mode 2, saves 59 us per step,
0.09% of 66.8 ms/token, against the 1% (0.67 ms/step) needed to put a ship decision to the user. This verdict is for
Gen3 divix01. Launch latency is a property of the GPU and driver, not the link, so a Gen5 host with the same GPU is not
expected to differ much. That expectation is untested.

Task 8 is gated twice: by this bound, and by the expert-stream-native-sync merge. It was not started. (Superseded:
the pre-wait probe below flipped the gate, the merge landed, and Task 8 ran; see "Real chain".)

## Pre-wait work probe (2026-09-27, commit 3ee723bd8b)

The review's caveat was that the skeleton has no prologue for PDL to overlap. This probe gives every chain stage a
spin of `pre_ns` on all threads before `griddepcontrol.wait`, at `work_ns` 2000, and runs modes 0/1/2 at
`pre_ns` 0, 200 and 500 in one job. The `moe` stage has none. Every-word-reads-R check: passed in all nine records,
`SKEL_EXIT=0`.

```bash
skeleton.py --repo $PWD --out analysis/dsv41-drive/chain-pdl/skeleton-prewait.jsonl --work-ns 2000 --pre-ns 0,200,500
python3 skeleton_report.py skeleton-prewait.jsonl
```

| pre-wait ns | mode 0 us/layer | mode 1 us/layer | mode 2 us/layer | mode 1 saving | mode 2 saving | mode 2 per step | share of 66.8 ms | >= 2 us gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 24.168 | 23.387 | 22.461 | 0.782 | 1.708 | 68.3 us | 0.102% | no |
| 200 | 25.912 | 25.136 | 22.649 | 0.776 | **3.263** | 130.5 us | 0.195% | **yes** |
| 500 | 28.212 | 27.446 | 22.962 | 0.766 | **5.250** | 210.0 us | 0.314% | **yes** |

The pre-0 row repeats the first run within noise (mode 2 saving 1.71 vs 1.48 us/layer).

- **With the early trigger (mode 2), the prologue is almost entirely hidden.** Mode 2's per-layer time rises only
  0.19 us at 200 ns and 0.50 us at 500 ns, against 1.74 and 4.04 us for mode 0. So each edge hides about 200-500 ns
  of prologue behind its predecessor's body.
- **With the implicit trigger (mode 1), none of it is hidden.** The saving stays at 0.77-0.78 us. The dependent cannot
  launch until the primary exits, so its prologue still runs after the primary.

**Does the PDL row flip?** Yes, for the early trigger, once real stages do about 200 ns or more of work before their
dependent read: the skeleton's saving crosses the 2 us/layer gate (3.26 us at 200 ns, 5.25 us at 500 ns). The Task 8
gate therefore reads "run Task 8" for mode 2, provided the real chain's pre-wait work is at least about 200 ns per
stage. That is unmeasured: Task 8, or a trace of the real stages' prologues, would measure it.

Two things do not change:
- **Task 8 stays blocked** by the expert-stream-native-sync merge gate (NOT_MERGED at the time of writing).
- **Even the flipped figure is small.** It is 0.20-0.31% of the 66.8 ms step. This is **above** the 0.1% the request
  anticipated, but still below the 1% (0.67 ms/step) needed to put a ship decision to the user.

Decision-table row "PDL on the chain", updated: **gate met for the early trigger if real prologues are >= ~200 ns
(skeleton: 3.26-5.25 us/layer); gate not met for the implicit trigger (0.77 us/layer). Ship bar not met (<= 0.31% of
step). Task 8 remains gated on the merge.**

## Real chain (Task 8, 2026-09-27)

Hooks: commit ab620a244c on `expert-stream-pdl-probe`, cut from origin/master 7da74eb569 and never merged. Harness
and results: `expert-stream-transfer-measurement` after the merge commit 478f401989. The first timing run is at
7602f6fee6. The pinned rounds are at 5bbc05c4f8. Every run used private worktree `wt-pdl`, with
`sglang.__file__` = `/data/models/slang/nvfp4-work/wt-pdl/python/sglang/__init__.py`, under the exclusive
`cc-gpu.lock` on cores 32-63, with `CUDA_MODULE_LOADING=EAGER`.

Modes:
- **off**: the production module.
- **pdl**: `EXL3_RAM_MISS_TEST_PDL`. Each hooked kernel runs `griddepcontrol.wait` first and is launched with
  `enable_pdl`, so the trigger is implicit, at exit.
- **pdl_early**: `EXL3_RAM_MISS_TEST_PDL` + `EXL3_RAM_MISS_TEST_PDL_EARLY`. Each hooked kernel also triggers
  `launch_dependents` right after its wait (ruling A).

Hooked kernels: post, W1 (stream_hit_wait), A1/A2 (stage_ack), S (stream), CW (copy_wait) and F (finalize). Not
hooked: C1 (`copy_expert_row_segments_gpu`) and the closing `torch.add`.

Scenarios (`chain_pdl.py`): the chain for one layer, captured in a CUDA graph stage by stage and replayed 200 times
after 10 warm-up replays.

| scenario | what it is |
|---|---|
| `all_hit` | Every lane is a RAM hit. |
| `mixed` | Half the lanes are RAM misses the host reads on each replay. |
| `all_hit_ce` | The copy engine is on, and CW is in the chain. |
| `all_hit_ce_sm` | The copy engine plus SM small copies: the production recipe. |

One replay is one layer's chain, so (off − mode) is the saving per layer.

### SASS gate: the hooks are test-only

```bash
sass_gate.py build wt-pdl-base <base>; sass_gate.py build wt-pdl <branch>   # CUDA_VISIBLE_DEVICES="", default build
sass_gate.py diff <base> <branch> --exact
```

The diff of origin/master 7da74eb569 against ab620a244c, both default builds, gives **GATE PASS** over 11 functions.
In the hook builds, the counts of `ACQBULK` (griddepcontrol.wait) / `PREEXIT` (launch_dependents) are:

| build | ACQBULK / PREEXIT |
|---|---|
| default | 0 |
| PDL | 6 |
| PDL + EARLY | 12 |

The later harness commits (7602f6fee6, 1059d04c00, 5bbc05c4f8) touch only `chain_pdl.py` and `chain_report*.py`. The
hook branch is unchanged, so the gate stands.

### Correctness at every mode

The lease suites, `test_exl3_ram_miss_cuda.py` and `test_exl3_ram_miss_graph_gpu.py` were run in each of four modes:
- the merge-commit baseline 7da74eb569;
- off, pdl and pdl_early at ab620a244c.

The PDL defines were injected by `pdl_mode_plugin.py`. Settings for every run: `SGLANG_EXL3_SRC`, `TMPDIR`,
`SGLANG_JIT_CACHE_DIR` and `--basetemp` under /mnt/nvme1, and `flock -s cc-gpu.lock`. The row-image suite took
`rowimg-disk.lock` first.

| suite | base | off | pdl | pdl_early |
|---|---|---|---|---|
| main: `test_exl3_ram_miss_two_phase{,_victim}.py`, `_lease_service.py`, `_piece_stream.py` (registered); `test_exl3_two_phase_{failure,parity,timing}_cuda.py`, `test_exl3_piece_stream_cuda.py`, `test_exl3_ram_miss_cuda.py`, `test_exl3_ram_miss_graph_gpu.py` (manual) | 167 passed | 167 passed | 167 passed | 167 passed |
| `test_exl3_piece_stream_row_images_cuda.py` | 25 passed | 25 passed | 25 passed | 25 passed |

Every suite read `PIPESTATUS[0]` = 0, and the failure-ID lists are identical (all empty). The plugin's summary confirms
the hooked module was loaded: 65 loads in the main suites and 27 in the row-image suite.

The timing runs check correctness too:
- Every 20th replay's destination bytes are compared with the source rows. All passed.
- After every replay, `leases_granted == acked + voided + copied`.
- The service counters are identical across modes within each scenario (for example, mixed: served 211, rows_read
  636).

### PDL edges survive graph capture

How it was checked: `chain_graph.cuh` calls `cudaGraphGetEdges` (the edge-data form) on the captured `cudaGraph_t`
(`torch.cuda.CUDAGraph(keep_graph=True).raw_cuda_graph()`). It names each node with `cudaFuncGetName` and records the
edge type and from-port for every edge. Results, identical in every run and round:

| scenario | off | pdl / pdl_early |
|---|---|---|
| no CW | 0 of 7 programmatic | 5 of 7 |
| with CW | 0 of 9 programmatic | 6 of 9 |

- **Programmatic edges** (type 1, from-port 1 = `cudaGraphKernelNodePortProgrammatic`): post→W1, C1→A1, A1→S, S→A2,
  A2→F, and A2→CW, CW→F with CW.
- **Full edges:** W1→C1 and F→add, whose destinations have no PDL attribute. With CW there is also a third full edge,
  add→add.

The instantiated graph honours them. In the pdl_early stamp run, dependents enter before their primaries exit. For
example, in `all_hit` S→A2, A2's entry is 29 µs before S's exit, and its body starts 256 ns after it. That can only
happen over a programmatic edge.

### Real pre-wait work per stage

SASS (PDL build): the instructions before `ACQBULK` are post 2, W1 2, A 2, S 3, CW 2, F 2. All of them are
`LDC`/`S2R`/`LDCU.64` (constant bank and special registers), with no `LDG`/`STG`/`ATOM`. The code as written has
essentially **no prologue** for PDL to overlap: every stage's first global access is chain state it must read after
the wait.

Timing mode (`EXL3_RAM_MISS_TEST_PDL_STAMP`, `chain-stamps.jsonl`, 100 replays): thread 0 of every block stamps entry,
the moment its wait returns, and exit. The prologue is (waited − entry) for launches that entered after their
predecessor exited. p50 in every scenario and mode: post 32 ns, W1 16-32, A 32, CW 32, F 32, and S 96-160 (S also
initialises its per-block shared state). The stamps' own cost is included, so these figures are upper bounds.

p50 gap from the primary's exit to the dependent's body (ns), `all_hit`:

| edge | off | pdl | pdl_early |
|---|---:|---:|---:|
| post→W1 | 672 | 768 | 240 |
| A1→S | 768 | 640 | 256 |
| S→A2 | 640 | 512 | 256 |
| A2→F | 640 | 704 | 448 |

`all_hit_ce_sm`:

| edge | off | pdl | pdl_early |
|---|---:|---:|---:|
| post→W1 | 704 | 736 | 256 |
| A1→S | 736 | 640 | 384 |
| S→A2 | 640 | 512 | 256 |
| A2→CW | 672 | 512 | 320 |
| CW→F | 640 | 736 | 224 |

- **The early trigger removes 0.2-0.5 µs of launch latency per hooked edge,** about 1.5 µs over a layer's edges.
- **The implicit trigger removes about 0.1 µs,** and sometimes adds that much.
- **W1→A1 runs through C1,** which is unhooked: 298 µs in `all_hit`, which is C1's copy.

### Timing

**First run** (`chain.jsonl`, one record each, service thread unpinned), replay p50 in µs:

| scenario | off | pdl | pdl_early | hits |
|---|---:|---:|---:|---:|
| all_hit | 368.45 | 367.78 | 366.88 | 6 |
| mixed | 892.18 | 847.55 | 434.85 | 3 |
| all_hit_ce | 402.59 | 401.25 | 399.55 | 6 |
| all_hit_ce_sm | 354.21 | 354.02 | 353.78 | 6 |

**The mixed figures are confounded, not a PDL effect.** The service thread's work was identical in all three runs
(served 211, rows_read 636), but the thread was unpinned and ran on core 44 (off), 39 (pdl) and 54 (pdl_early).
`mixed` is bound by the host's reads, so it timed the core. The harness now pins the thread through the service's own
knob, `ExpertStreamHost.start_thread(cpu_core=--spin-cpu)`, and keeps its own main thread and the packing workers,
which inherit its affinity, off that core. Every record asserts `spin_cpu` equals the pinned core. It also runs modes
in rounds, rotating their order each round, and `chain_report.savings` takes the median over rounds.

**Pinned run** (`chain-rounds.jsonl`, core 40 on NUMA node 0, the GPU's node; 5 rounds; `spin_cpu` = 40 in all 60
records). Replay p50, µs, median [min-max] over rounds:

| scenario | off | pdl | pdl_early |
|---|---|---|---|
| all_hit | 374.10 [373.97-374.32] | 373.58 [373.50-373.86] | 373.04 [372.72-373.17] |
| mixed | 790.14 [741.65-802.62] | 802.94 [775.09-828.38] | 762.54 [749.22-784.94] |
| all_hit_ce | 407.07 [406.75-409.14] | 408.19 [405.46-409.07] | 406.67 [406.51-408.56] |
| all_hit_ce_sm | 360.88 [360.34-361.36] | 360.30 [360.08-360.70] | 361.84 [360.62-361.94] |

**Core check** (`chain-mixed-cpu54.jsonl`: mixed pinned to core 54 on node 1, 3 rounds): off 472.21 [451.81-499.92],
pdl 485.70 [445.62-657.09], pdl_early 481.49 [464.16-600.70]. The core alone moves `mixed` by about 300 µs, which
accounts for the first run's 2x. On neither core does the mode order the results consistently. A node-1 service thread
serving `mixed` 1.7x faster than a node-0 one is a separate finding, not investigated here.

Saving per layer = median(off) − median(mode), pinned run. The ship bar is 1% of the step: 668 µs at 66.8 ms/token
(16.7 µs/layer) or 1.11 ms at ~111 ms/token (27.8 µs/layer).

| scenario | mode | per layer (µs) | per 40-layer step (µs) | share of 66.8 ms | share of 111 ms | >= 2 µs/layer gate | >= 1% ship bar |
|---|---|---:|---:|---:|---:|---|---|
| all_hit | pdl | 0.52 | 20.8 | 0.031% | 0.019% | no | no |
| all_hit | pdl_early | 1.06 | 42.4 | 0.063% | 0.038% | no | no |
| mixed | pdl | -12.8 | — | — | — | noise (range ±30 µs) | no |
| mixed | pdl_early | 27.6 | — | — | — | noise (ranges overlap; core 54 gives -9.3) | no |
| all_hit_ce | pdl | -1.12 | -44.8 | -0.067% | -0.040% | no | no |
| all_hit_ce | pdl_early | 0.40 | 16.0 | 0.024% | 0.014% | no | no |
| all_hit_ce_sm | pdl | 0.58 | 23.2 | 0.035% | 0.021% | no | no |
| all_hit_ce_sm | pdl_early | -0.96 | -38.4 | -0.057% | -0.035% | no | no |

(`python3 chain_report.py chain-rounds.jsonl`.)

**Reading.**
- **`all_hit` is the only scenario where the saving is resolved:** the three modes' ranges do not overlap. The early
  trigger saves 1.06 µs/layer (1.57 unpinned) and the implicit trigger 0.52, which matches the skeleton (1.41-1.71 and
  0.77) and the stamped edge gaps.
- **In the copy-engine scenarios the saving is below the noise** (±1 µs), although the stamped edges shrink just as
  much. The critical path there is CW waiting for the copy thread's CopyDone. With pdl_early, F enters 146-257 µs before CW exits (CW->F
  entry after exit, stamp run): CW spends that long waiting. So removing launch latency ahead of CW does not shorten the layer.
- **`all_hit_ce_sm` is the production recipe, and there PDL saves nothing measurable.**
- **Nothing reaches the 2 µs/layer gate or the 1% ship bar.** The first run's `all_hit_ce` 3.04 µs did not repeat
  pinned (0.40).

### Safety of the early trigger (ruling A), per kernel

Applies to post, W1, A1, A2, S, CW and F.

1. **Nothing before the wait touches global memory.** The SASS above shows 2-3 `LDC`/`S2R`/`LDCU.64` before
   `ACQBULK` in every hooked kernel, with no `LDG`, `STG` or `ATOM`.
   - `TestPdlEntry` is the first statement of each kernel body, and the trigger comes after it.
   - Under `_STAMP` only, thread 0 writes its entry stamp to the test ring `g_pdl_stamp` before the wait. That build
     is used only for the stamp run and writes no chain data.
2. **Early residency cannot starve a primary.** A dependent launches only after every block of its primary has
   triggered or exited. Each primary triggers right after its own wait, so all of its blocks are already resident when
   the dependent can be scheduled.
   - Grids: S is 8 blocks × 256 threads (`__launch_bounds__(256, 1)`); post, W1, A, CW and F are 1 block each.
   - At most two chain kernels are resident at once, at most 9 blocks on an RTX 5090 with 170 SMs.
   - The dependent waits in `griddepcontrol.wait` (`ACQBULK`), holding one block's registers and shared memory, and
     takes no scheduling slot the primary needs.
   - Measured: the waiting dependent is resident for up to 309 µs (mixed S→A2), and the primary's timing is
     unchanged.
3. **The host protocol is unaffected.** The hooks change only when a kernel launches, not what it reads or writes after
   its wait. Every access to `state[]`, `count`, `go_*`, `lane_ctx`, `claimed`, the row results and the host-pinned
   request and ack words happens after the wait, in the same order as the production build. Evidence:
   - The service counters are identical across modes: served, rows_read, evictions, and granted = acked + voided +
     copied.
   - The byte checks and the full lease, two-phase, piece-stream and row-image suites are identical in every mode.
4. **Ordering is transitive through each primary's own wait.**
   - `griddepcontrol.wait` in kernel K returns only after K's primary has completed and its memory is visible.
   - That primary passed its own wait before any chain access, so it completed after its own primary completed, and so
     on up the chain.
   - Unhooked kernels (C1, `torch.add`) have no PDL attribute: they wait for their predecessor in the ordinary way, and
     they never trigger early. Their dependents (A1 after C1, the next layer's post after `torch.add`) are released
     only when they exit, and still execute the wait (C1→A1 is a programmatic edge with an implicit trigger at exit).
   - So when kernel K passes its wait, every earlier kernel on the stream has completed, exactly as without PDL.

### Pre-wait hoist table (analysis only; placement unchanged, no timing mode run)

Rule: movable means reads of data no in-flight predecessor writes, and no global writes. `state[]`, `count`, `go_*`,
`lane_ctx*`, `claimed` and every device buffer the chain writes are not movable.

In pdl_early, A1, S, A2, CW and F can all be resident at once, so anything any of them writes is off limits. That
includes `page+kFatal`, which A (`LK:787`) and F (`LK:864,866`) write.

Paths: `LK` = `lease_kernels.cuh`, `LD` = `lease_device.cuh`, `RC` = `row_copy_kernels.cuh`.

| kernel | movable items (source) | why safe | est. ns on the critical path | first dependent read after the wait |
|---|---|---|---:|---|
| post | exit test `LK:64`; `map_row` arithmetic `LK:70`; `(experts+7)/8` `LK:104` | parameters only | ~0: its predecessor `torch.add` is unhooked, so there is no window | `state[kSticky]` `LK:65` |
| W1 | register setup and the row-table address `LK:549`, `LD:382` | parameters only | ~8 | `count[0]` `LD:361` |
| W1 | **the row-table capacity read `LD:382`** (host-pinned) | written once at init (`host/ram_tier.h:1324-1325`, then `_mm_sfence`) and never again; address from parameters only; no order relative to a predecessor's output is needed | 750 (one host-pinned load, today serial after the two acquires at `LD:364`) | |
| A1 / A2 | `any_violated=0` + `__syncthreads` `LK:764-765` (shared); `entry` `LK:766`; LaneAck base `LK:779` | shared memory and parameters | A1 ~0 (after unhooked C1), A2 ~8 | `go_count[0]` `LK:767` |
| S | shared-state zeroing `RC:266-271,305-310` (not `sh.mine`, which reads `claimed`); `piece_stride` `RC:318` | shared memory and parameters | ~20 | `count[0]` `RC:263` |
| S | **the same capacity word, `RC:300`** | as for W1's read; the `seq != 0` guard becomes a select | 750 | |
| S | `fault[kStreamFaultAbortBlock]` `RC:290`; `segment_map`/`segments` `RC:96,101-102` | host-set or host-built, no chain writer, addresses from parameters | 0 (overlaps the capacity read; the tables are cached by the time the last piece is copied) | |
| CW | `lease_c`/`lease_d`/mask arithmetic `RC:530-531,550,595`; `sm_table` entries `RC:575-579` | parameters; host-built table with no chain writer | ~5 + 110 × `sm_count` (independent loads, one round once hoisted) | `state[kPending]` `RC:540` or `count[0]` `RC:590` |
| F | exit test `LK:830`; LaneAck base `LK:849` | parameters only | ~3 | `count[0]` `LK:831` |

Not movable:
- post: `count`, `planned`, `slot_map`, `routes`, `hot_slots` (treated conservatively) and `last_routes`.
- W1: `state[kPending*]` and the row-result polling (its address depends on `seq`). `start = global_ns()` stays where
  it is, because moving it would spend W1's budget in the wait.
- A: the row-table `base` `LK:774`, because `row` comes from `lane_ctx`. The SlotGen acquire `LK:775`.
- S: `state`, `claimed`, the row results and piece masks, and `piece_runs` (address from `planned`).
- CW: the row results, LaneRequest and CopyDone polls.
- F: the LaneAck words, which A1 and A2 write.

The early kFatal check can only make a kernel fail sooner, and it adds a host-pinned read (~750 ns) without removing
the post-wait checks (`LK:65`, `LD:364`, `RC:272-273`, `LK:837`), which must stay. Net benefit: 0 in every kernel.

**Sum for one layer:** W1 ~758 + S ~770 + A2 ~8 + F ~3 + post/A1 ~0 ≈ **1.54 µs/layer upper bound**. That is 61.6 µs
per step: 0.092% of 66.8 ms and 0.055% of 111 ms (with CW: +5 ns + 110 × `sm_count`).

It is an upper bound because it lands only when every lane is claimed without W1 reaching its budget, and the host has
served before S starts. That is `all_hit`. In the host-bound or copy-engine cases (the production recipe) it is about
40 ns.

- **Gate:** added to `all_hit`'s measured 1.06 µs, it reaches about 2.6 µs/layer, which crosses the 2 µs gate. Neither
  it nor the real `all_hit_ce_sm` figure approaches the 1% ship bar.
- **Proposed variant, not run:** 1.5 of those 1.54 µs are two reads of one static word, the row-table capacity. Pass
  it as a kernel parameter, since it is known at init. That removes both reads in **every** mode, PDL or not, and it
  needs no PDL. The same holds for A's row-table `base` (`LK:774`) if `row` were a parameter: +750 ns in A2, not
  counted above. Its worth would be at most about 1.5 µs/layer, 0.09% of the step, off the critical path in the
  production recipe.
- **Unresolved:** `sm_count` in production; whether any kernel on this stream writes `hot_slots`; and the C1 and
  `torch.add` launch sites, which were assumed to carry no trigger. The captured graph's full W1→C1 and F→add edges are
  consistent with that assumption.

### Decision-table row: PDL on the chain (Gen3 divix01, real chain)

**No, do not ship.** The largest saving resolved above noise is 1.06 µs/layer (`all_hit`, early trigger): 42 µs per
step, 0.063% of 66.8 ms and 0.038% of 111 ms. That is below the 2 µs/layer gate, and more than 15x below the 1% ship
bar. In the production recipe (`all_hit_ce_sm`) the saving is within ±1 µs of zero, because the layer waits on the
copy thread, not on launches. The skeleton's prologue caveat does not apply: the real stages do 16-160 ns before their
wait, not the 200-500 ns that flipped the skeleton's gate. The hoist analysis adds at most 1.5 µs/layer, and all of
that is available without PDL by passing one static word as a kernel parameter.

**Question for the user:** ship PDL on the lease chain, yes or no? (Recommendation: no.)

## Decision and productionization (2026-09-27)

The user decided to ship PDL on the lease chain, with the early trigger, behind `SGLANG_DSV41_ENABLE_LEASE_PDL` (off by
default, on in the recipe). It is on branch `expert-stream-lease-pdl`: tests 64204c3638, kernels 45949cf721, row
capacity as a kernel argument 1540bbc243 / 5d91f083d5, and origin/master 48467f2ade merged at a2f0ae97b0. The
safety argument is LEASE_PROTOCOL.md 7.7. The capacity timing did not resolve: the effect is below between-process
noise (up to 14 µs). The expected effect is ~0.75 µs per removed host load, at most ~1.5 µs/layer in `all_hit`. The
change rests on its SASS (one `LDG.E.STRONG.SYS` removed in each of the three kernels) and on the proof that the value
is constant.

### Copy-engine `[kernel]` stall control

A mutant suite (stage_ack trigger above its wait) failed once in
`test_exl3_copy_engine_cuda.py::test_work_queued_behind_the_graph_on_other_streams_does_not_hold_the_copy_back[kernel]`.
Replay 0 of the armed copy-engine graph, with a kernel queued behind it on another stream, took exactly the 2.000 s
RAM-miss deadline (`keep` still 1.0). DSV41_REFERENCE.md 27.14 records this variant as a known flake (2 of 8 at base).

To test whether PDL makes the flake worse, the control ran that test, all 3 params, 20 rounds, with six trees
round-robin under the exclusive `cc-gpu.lock`.
- In the PDL trees, the suite plugin forced `lease_pdl` on every CUDA lease device (3 per run).
- The implicit-trigger tree is the merged tip with every `PDLTriggerSecondary` call deleted and the waits kept: a scratch
  mutant, never committed.
- The mutant and implicit trees have 19 runs because round 1 was cut short to yield the GPU.

| tree | h2d | d2h | kernel | kernel stalls (round) |
|---|---:|---:|---:|---|
| base, origin/master 48467f2ade (no PDL code) | 0/20 | 0/20 | **4/20** | 4, 14, 16, 18 |
| a2f0ae97b0, flag off | 0/20 | 0/20 | **2/20** | 4, 8 |
| **a2f0ae97b0, flag on** | 0/20 | 0/20 | **3/20** | 1, 3, 11 |
| a2f0ae97b0 on, implicit trigger | 0/19 | 0/19 | **6/19** | 6, 7, 8, 10, 13, 14 |
| 45949cf721, flag on | 0/20 | 0/20 | **6/20** | 1, 4, 9, 10, 12, 14 |
| mutant, trigger above wait, flag on | 0/19 | 0/19 | **2/19** | 6, 13 |

Every failure is `[kernel]` replay 0 at 2.000 s. h2d and d2h never failed in 237 runs.

**Reading: this is a pre-existing copy-engine stall, not made worse by PDL.**
- The shipped configuration (flag on) stalls 3/20, against base 4/20.
- Pooled, the four PDL-on trees stall 17/78 (22%) and the two trees without PDL 6/40 (15%). A two-sided Fisher test
  gives p ≈ 0.5.
- The rates do not order by feature: flag off is below base, the implicit trigger is no better than the early one, and
  the mutant is among the lowest.
- So the mutant shows no scheduling hazard in this test. The source-order test (`test_exl3_ram_miss_device_args.py`)
  and the SASS gate are what catch a trigger moved above its wait.
