# NUMA Node Distributor (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One expert group per NUMA node: each group owns the slots of the pinned tier bound to its node, a RAM/NVMe thread and a CPU-expert engine on that node's cores, and serves only the experts homed there, so an expert's bytes and its compute never cross the socket link. One `ThreadingConfig` resolves every core.

**Architecture:** The wire already has the node axis (Phase 1: `LeaseLayout<NumLanes, NumNodes>`, node-major staging and split). Each JIT build now also gets `-DSGLANG_EXPERT_STREAM_NODES`. On the host, `RamTier` stays the one object behind one FFI handle and keeps everything a record has once (page, lease block, host map mirror, copy engine, the eager Python paths); the per-node serve state moves into `NumaGroup` (its slot range per row, staging, LRU clock, reader ring, cursor, counters, CPU engine), and `NumaNodeDistributor` holds the groups, the home rule `expert % nodes` and the combiner that merges the groups' map deltas. `RamThread` runs one service thread per group plus one watchdog. The copy engine takes one job part per group and stores CopyDone only once every part's DMA and its group's CPU jobs are done. The device picks a miss's staging slot from its home node's list and the CPU lanes per node. The CPU kernel takes an engine handle (its own core list) on every call. At one node every one of these is today's code path with one element, and the wire is byte-identical to v2.

**Tech Stack:** C++20 / CUDA (nvcc `--expt-relaxed-constexpr`), TVM-FFI JIT modules (`sglang.kernels.jit.utils.load_jit`), OpenMP (libgomp, GCC 15), Python 3.13, pytest, CMake for the bench.

**Spec:** `docs/superpowers/specs/2026-10-03-numa-node-distributor-design.md`: Part 2, Part 3, "Errors", "One node is today's runtime", Testing items 2-6 (Phase 2). Phase 1 (`docs/superpowers/plans/2026-10-03-n-lane-lease-layout.md`) is landed on this branch.

## Global Constraints

- Home rule: `home(layer, expert) = expert % num_nodes`, one function in C++ (`LeaseLayout::home`, host and device) and one in Python (`WireLayout.home`); callers never compute it themselves.
- The wire's node axis is the group index `0..nodes-1`, in `SGLANG_MOE_PINNED_HOST_NUMA_MB` order; the NUMA node id is `NodePlan.node`. Without that variable there is one node (the GPU's).
- `1 <= NumNodes`; a build has `Wire = LeaseLayout<SGLANG_EXPERT_STREAM_LANES, SGLANG_EXPERT_STREAM_NODES>`, both defines defaulting to 8 and 1.
- At `(8, 1)` the wire is v2: the `static_assert`s in `lease_layout.h` and `test_eight_lanes_on_one_node_is_wire_v2` stay as they are. One-node module names stay `expert_stream_{layout}_l{lanes}` and `expert_stream_host_{layout}_{variant}_l{lanes}`; a multi-node build appends `_n{nodes}`.
- Staging per node: `reserve_staging(k)` gives each group `min(k, range - 1)` slots, the lowest of its range, `1 <= k <= Wire::kLanes`.
- `ThreadingConfig` per node (spec Part 2): usable = the node's physical cores, minus the server's affinity, minus the reserved cores `64-71`, minus the SMT siblings of anything already chosen; RAM/NVMe core = the highest usable core; CPU engine = the remaining usable cores, ascending, capped at `SGLANG_DSV41_CPU_EXPERTS_THREADS` when set, `cpu[0]` the engine thread (worker 0); SQPOLL core = the next usable core when io_uring SQPOLL is on. Copy thread on the GPU's node.
- Override `SGLANG_EXPERT_NUMA_CORES`, e.g. `"1:ram=35,cpu=18-33"`, replaces the derived plan of each node it names, validated by the same rules.
- Refuse at start: a node with no usable core, any overlap with the server's affinity or the reserved set, an engine with fewer than 2 cores, a core outside its node, SMT siblings in one plan. Nothing degrades silently.
- Log one line per node at start, exactly `numa node1: ram=35 cpu=18-33 (16) sq=-`.
- Expected on divix01 (server `0-7,16,36-52`, CPU experts on, `SGLANG_DSV41_CPU_EXPERTS_THREADS=16`): node 0 `ram=17 cpu=8-15`, node 1 `ram=35 cpu=18-33`, copy thread on node 0.
- Combiner: the last group to report writes the row's one delta (every node's staging list, every reporting group's entries) and then its tag, entries before the tag with a release, as today. PieceMask needs no combiner.
- The single copy thread stores CopyDone for record G only after its DMA and every group's CPU `done(seq)` for G.
- Errors: any group's fail-stop stops the whole service; a stalled group never lets CopyDone be stored, and the copy-wait timeout's abort names the group whose part or `done(seq)` is missing.
- One node and N = 8 is today's runtime: every task keeps `SUITE_CPU` and `SUITE_GPU` green. A test changes only where an API it calls changes (listed in the task's Files); no assertion about one-node behaviour is weakened.
- `SGLANG_EXPERT_NUMA_CORES` follows `.claude/skills/env-var-conventions/SKILL.md`: an `EnvStr` on `Envs` in `python/sglang/srt/environ.py`, read only through `envs.SGLANG_EXPERT_NUMA_CORES.get()`, overridden in tests with `.override(...)`.
- Comments follow `.claude/rules/comment-style.md`. Edits to `python/sglang/srt/layers/moe/exl3_ram_miss.py` touch no frozen class; read `.claude/rules/modify-component-must-read.md` before an edit that does.
- Code reaches divix01 only by commit, `git push origin numa-node-distributor` and a pulled private worktree (`.claude/rules/divix01-run-protocol.md`). Pushing is externally visible: get the user's OK once before the first push. Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- **The checkout is shared with the user's concurrent work** (uncommitted edits, notably under `python/sglang/srt/layers/quantization/nvfp4_cpu/`). Stage only your own hunks: `git add <file>` for files only this plan touches, `git add -p <file>` where the user also has edits. Run `git diff --cached --stat` before every commit and check it lists exactly the task's files.
- Commits end with the two trailer lines `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft`. A failing-test commit and its fix are separate commits; never amend or rebase.
- Read pytest's status from `PIPESTATUS[0]`, never from a pipeline. Record the exact command next to every count you quote.
- GPU work runs only through `gpu-run.sh` (exit 75 = the lock timed out, not a failure); CPU work under `taskset -c 0-63`.

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

`RUN_GPU <files...>` (divix01, under the GPU lock):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh \
  /data/models/slang/.venv/bin/python -m pytest <files...> -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

`RUN_EXT <files...>` (divix01; the manual CPU-kernel tests need the EXL3 extension built with the optimized kernel, from a private build directory as `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` makes one, on 4 cores of node 1). Refuse to run it while `pgrep -f sglang.launch_server` finds a server:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && B=/data/models/slang/nvfp4-work/nlane-exl3-build;
  [ -d $B/resid_b128_cpu_v1 ] || { mkdir -p $B && cp -a ~/.cache/sglang/exl3_ext/resid_b128_cpu_v1 $B/; };
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 EXL3_MOE_CPU_PIN=0 SGLANG_DSV41_CPU_EXPERTS=1 \
  SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 SGLANG_EXL3_BUILD_DIR=$B \
  SGLANG_EXL3_CPU_CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ CUDA_HOME=/usr/local/cuda-13.4 taskset -c 18-21 \
  /data/models/slang/.venv/bin/python -m pytest <files...> -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

`CPU_CHECKS <out> [baseline_out]` (divix01, the bit-exact gate for any change to the optimized EXL3 kernel): `baseline` mode at the merge-base once (Step B1), `check` mode at the task's commit with `slabs`:

```bash
ssh divix01 'M=$(git -C /data/models/slang/sglang merge-base origin/master origin/numa-node-distributor);
  [ -d /data/models/slang/nvfp4-work/wt-nlane-base ] || git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-nlane-base $M;
  bash /data/models/slang/nvfp4-work/wt-nlane/test/manual/dsv41/run_exl3_cpu_forward_checks.sh baseline /data/models/slang/nvfp4-work/wt-nlane-base /data/models/slang/nvfp4-work/nlane-cpu-base 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'
ssh divix01 'bash /data/models/slang/nvfp4-work/wt-nlane/test/manual/dsv41/run_exl3_cpu_forward_checks.sh check /data/models/slang/nvfp4-work/wt-nlane /data/models/slang/nvfp4-work/nlane-cpu-<task> /data/models/slang/nvfp4-work/nlane-cpu-base slabs 2>&1 | tail -20; echo EXIT=${PIPESTATUS[0]}'
```

Expected of `check`: `ALL GREEN (check)`.

`BENCH <dir> [cmake -D flags]` (divix01; the C++ bench is outside the JIT): configure and build, then the full stack's self-test (synthetic rows, a fake forward, CPUs 0-3):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && B=/data/models/slang/nvfp4-work/<dir>; taskset -c 0-63 cmake -S python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $B <flags> -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ -DEXL3_TORCH_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/torch -DEXL3_CXX11_ABI=1 -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src >/dev/null && taskset -c 0-63 cmake --build $B -j 16 2>&1 | tail -3; echo BUILD=${PIPESTATUS[0]};
  mkdir -p $B/images && taskset -c 0-63 $B/exl3_full_stack_prod --self-test --image-dir=$B/images 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'
```

Expected: `BUILD=0` and `EXIT=0`.

Before trusting any run, check `sglang.__file__` once per worktree:
`ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang;print(sglang.__file__)"'`
Expected: a path under `wt-nlane/python/`.

### Baseline (before Task 1)

- [ ] **Step B1:** `SYNC`, then record the counts of the suites this plan touches:

```
RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_tier.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_expert_stream_ownership.py test/registered/unit/kernels/test_exl3_ram_miss_wrap.py test/registered/unit/kernels/test_exl3_ram_miss_read_record.py test/registered/unit/kernels/test_expert_stream_lease_layout.py test/registered/unit/kernels/test_exl3_lease_block.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/registered/unit/kernels/test_ram_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_expert_stream_build_variants.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_expert_stream_uring_options.py test/registered/unit/kernels/test_expert_stream_uring_integration.py test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_nvfp4_cpu_build.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py test/registered/unit/layers/moe/test_host_numa.py test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_host_slab_arena.py benchmarks/dsv41_baseline/test_dsv41_baseline.py
RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_cpu_split_calibration_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_dsv41_layer_fusion_gpu.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py
RUN_EXT test/manual/dsv41/test_cpu_expert_pool_exl3.py
```

Then run `CPU_CHECKS`'s `baseline` command (no server may be running). Write the three pass/skip/fail counts and the baseline's `ALL GREEN (baseline)` line into the ledger as `Baseline:`. Call the file lists `SUITE_CPU`, `SUITE_GPU` and `SUITE_EXT`; every "existing suites" step below runs all three and compares with these counts plus the task's new tests.

## Spec deltas

Each is a place where the code forces a decision the spec did not make. The decision is binding for every task.

1. **The node count reaches the builds as a define, not a template argument.** The host's `Layout` template parameter is the *row* layout (`ExpertRowLayout`, ram_tier.h:84), and every host and device file uses the global alias `Wire` (lease_layout.h:103). So `lease_layout.h` gets `SGLANG_EXPERT_STREAM_NODES` beside `SGLANG_EXPERT_STREAM_LANES`, `Wire = LeaseLayout<LANES, NODES>`, and every `load_jit` call passes `-DSGLANG_EXPERT_STREAM_NODES={nodes}`. Module names gain `_n{nodes}` only above one node, so every one-node module keeps its name and the Phase 1 build checks hold. Why: a template parameter would have to be threaded through `RamTier`, `RamThread`, `CopyEngine`, `CpuExpertEngine`, `HostExports` and every kernel, for no gain over the alias Phase 1 already uses for lanes.
2. **A NumaGroup is a split of `RamTier`, not N `RamTier`s.** `RamTier` keeps what a record has once: the page, the lease block, the host map mirror, the per-row slot tables (partitioned by slot range), the copy engine, the eager Python paths and the one FFI handle. `NumaGroup` (new `host/numa_group.h`) holds what a node's RAM thread owns: its slot range of every row, its staging slots, its copy of each row's map chain, its LRU clock, its reader (one io_uring ring, with its own SQPOLL core), its demand cursor, its scratch, its counters, its busy episode and its `CpuExpertEngine`. `NumaNodeDistributor` (new `host/numa_distributor.h`) owns the groups, the home rule and the combiner. `RamThread` runs one service thread per group and one watchdog for all of them, not the spec's watchdog per group: the copy-wait gate is one word, and several watchdogs would race to report one stall. Why the split: N `RamTier`s would each build a copy engine and a lease-block view, CopyDone and the gate must stay single, and every one of the ~45 FFI exports would need a forwarding layer that routes by expert; the split keeps the handle and the eager paths as they are. Two group threads touch one row's `Tier` concurrently only at disjoint indices: slot-indexed arrays at their own ranges, expert-indexed arrays at their own homes. The shared per-row words become per group (`staging`, `chain`, prefill `owned`) or atomic (`rows_demand`).
3. **Slot ranges come from the exact `mbind` bindings.** No slot-to-node API exists, and `plan_bindings` rounds every node change to 2 MiB, so `split_rows` is only approximately where the pages are. `allocate_bound` now records the bindings it applied (`tensor._numa_bindings`, byte ranges from the tensor's base), and one function, `host_numa.slot_nodes`, derives each slot's node from them across every named slab of a layer. A slot belongs to node n when n holds at least 99% of its bytes (arbitrary; a whole layer's sign-vector slabs fit one 2 MiB page and so sit on one node, about 0.1% of a slot's bytes). A slot below that, one straddling a seam, belongs to no group: it is dead padding, never staged, filled or mapped. Each node's slots must form one contiguous range per row, and a group needs at least 2 slots in every row, else start is refused. Why: the spec's success criterion (every CPU job reads a slot on its worker's node) cannot hold for a slot whose large tensor straddles the seam.
4. **The copy thread is pinned.** It is the only thread created on the caller's affinity today (copy_engine.h `start()`). `enable_copy_engine` takes the `ThreadingConfig.copy_cpus`: the server's affinity on the GPU's node, or, when the server has no core there, the node's CPUs minus the reserved set and every plan's cores. The watchdog keeps inheriting the server's affinity (it sleeps 20 ms per turn). The GPU's node is its PCI device's sysfs `numa_node`, from `torch.cuda.get_device_properties`.
5. **CopyDone waits on every group.** The copy engine's single `cpu_` pointer and its single-producer job ring become one engine pointer and one SPSC ring per group. A `CopyJob` becomes one group's part of a record: it carries `group` and `groups`, the bitmask of nodes that have a host lane (HIT_COPY, HIT_CPU or MISS_CPU) in the record, which every group computes from the whole record it read. The copy thread completes a part when its DMA and its group's `done(seq)` are done and stores CopyDone when the record's `groups` are all complete. While it cannot, it publishes a stall word (`group << 8 | reason`); the watchdog's copy-wait abort appends it above one group: `; group 1: its CPU job 17 is not done` or `; group 1: sent no part` (C++ knows group indices; `ThreadingConfig.log_lines` maps them to node ids at start).
6. **`SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU` is resolved in Python.** C++ reads it with `getenv` (uring_options.h:134). It is declared on `Envs` and read only by `ThreadingConfig`, together with the SQPOLL mode (`SGLANG_EXPERT_STREAM_URING_MODE`, also declared). Each group's SQPOLL core reaches C++ through `expert_stream_open`'s `sq_thread_cpus` and overrides `UringOptions::from_env`'s value in every ring a `RamTier` opens. The standalone reader exports (`read_rows*`) and the native reader harness keep `from_env` for their own tests.
7. **NVFP4 follows last, separately.** The NVFP4 kernel mirrors the ABI but also has a process-wide `forward_mutex` that returns 3 to a second concurrent forward, which would fail-stop a second engine. The user is editing `nvfp4_cpu/` now, so its ABI change is Task 11 alone, marked as touching files under concurrent edit. Until then NVFP4 compiles against the new ABI header unchanged: the engine field is appended at the struct's end (Delta 13).
8. **ChainSim and `ram_slot_map` read every node.** Both read node 0's staging list and split table today. The Python reference keeps flat lists in the wire's node-major order (staging `[nodes * lanes]`, split `[nodes * (lanes + 1)]`), so a one-node caller passes exactly what it passes today; `type_lanes` and `MapReplica` gain `nodes: int = 1`.
9. **CPU output parts per group.** Two groups' CPU hits in one record would both overwrite part 0 of the row's output. `out_rows` becomes `[rows, 2 * nodes, hidden]`; group g's hits go to part `2g`, its misses to part `2g + 1` (each group's engine gets `out_base` offset by `2g` parts). The copy wait's part mask (`cpu_lanes[1]`) has a bit per part, and the route tables add every set part in ascending order, so one node's sum is part 0 then part 1, as today, bit for bit. The copy wait learns each lane's group from a new post output, `lane_node`.
10. **Who reports to the combiner.** Only the groups that have a miss lane in record G report for G (every group computes that set from the record); the last of them writes the delta. A group without misses in G keeps its staging list unchanged, and its list is taken from the combiner's per-row copy, which it last wrote when it last reported. That copy is safe to read: the device posts G only after it applied the row's previous delta, and the last reporter checks the row's written chain with an acquire before it writes, so the earlier writer's stores are visible. Each group keeps its own copy of every row's chain and advances it on every record that has a miss anywhere. A group with no miss in a miss record is therefore never waited for, which keeps a lagging group from blocking a delta it has nothing to add to.
11. **Eager paths place an expert in its home range.** `take_slot_locked` and `take_admit_slot_locked` (assign, prefill fills) scan only the home group's range. Otherwise a group thread would later serve a hit whose slot another group owns, racing that group's stamps. Every group's ring registers the whole tier, as today's one ring does, so the prefill fill keeps reading through one reader (group 0's; every group is parked during a fill) with its claim-order progress unchanged, and its rows still land on their home node because the pages are bound. Cost: the tier is registered once per group (pin accounting and registration time double at two groups); Task 10 compares the arms' start times.
12. **When `ThreadingConfig` derives full plans.** With one node from no `SGLANG_MOE_PINNED_HOST_NUMA_MB`, CPU experts off and no override, the plan is today's: the RAM thread pinned to `SGLANG_DSV41_RAM_MISS_SPIN_CORE` (busy-poll) or inheriting, the copy thread inheriting. Any other launch derives every node's plan and validates it. Consequence: landing this turns two groups on for every launch whose `SGLANG_MOE_PINNED_HOST_NUMA_MB` names two nodes, which includes `arm_env.PINNED_HOST_NUMA_MB` (production). Task 10 measures that configuration before production pulls the branch.
13. **The CPU ABI's engine handle.** `SglangCpuExpertsForward` gains `int64_t engine` at its end (positional initializers in the NVFP4 sources and tests keep compiling) and `SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION` becomes 2. Engine 0 means no engine: workers unpinned, which is what a kernel did before `set_cores`. `CpuExpertKeepWarm` takes the engine first. EXL3's `set_cores` is deleted with its globals. Engines are immutable after `engine_create`.
14. **`OMP_THREAD_LIMIT`.** Two engines are two OpenMP teams in one process. The arms' scripts set `OMP_THREAD_LIMIT=16`, and whether libgomp counts that limit per team or over the process is not something this plan relies on: `ThreadingConfig` refuses an `OMP_THREAD_LIMIT` below the sum of every engine's workers, and Task 2 shows two full teams run at once at exactly that sum.
15. **The duplicated checks get one implementation each.** `threading_config.py` owns `RESERVED_CORES`, `check_not_reserved` and `check_engine_cores`. `ExpertStreamHost.start_thread` (the API that direct users and the existing reserved-core test call), the fallback `CpuExpertPool` and `CpuExpertService` call them; the C++ `64-71` refusal in `ffi_exports.h` goes; `core_topology.h`'s sibling check stays as the C++ assertion. `arm_env`'s by-hand sibling test is replaced by a `ThreadingConfig` resolution of its constants.
16. **`SGLANG_DSV41_CPU_EXPERTS_CORES` and `SGLANG_DSV41_RAM_MISS_SPIN_CORE` with nodes.** `CORES`, when set, is the `cpu=` override of the node that holds those cores; cores on two nodes, cores off the node set, or a node named both by `CORES` and by `SGLANG_EXPERT_NUMA_CORES` are refused. (Before Phase 2 a `CORES` list on node 1 with the tier on node 0 ran cross-socket; now it is refused.) `SPIN_CORE`, when set, is the `ram=` of its node and busy-polls. A derived RAM core also busy-polls: it is the highest usable core whose SMT siblings are all outside the server's affinity, which is exactly `core_topology.h`'s condition.
17. **The THREADS cap truncates the core list.** "Capped by `SGLANG_DSV41_CPU_EXPERTS_THREADS`" is read as: the engine's cores are the first `THREADS` usable cores, and it runs one worker per core. The spec's expected `cpu=18-33 (16)` for node 1 is what `THREADS=16` gives; with `THREADS` unset node 1 gets `18-34 (17)`.
18. **The stage trace records group 0's requests only** above one node. It is one `StageRecord` filled by one thread (InstrBuild diagnostics); a group other than 0 skips `begin_stage`.
19. **Where `ThreadingConfig` runs.** It sits beside `service.py` as the spec says, so importing it imports `sglang.srt.layers.moe`, whose `__init__` imports torch. Its logic (`ThreadingConfig.resolve`) takes the topology, affinity and settings as arguments and reads nothing, so its test runs on fake sysfs trees; the gate run is `RUN_CPU`, and a laptop run needs a venv with torch.
20. **Locality is checked through ranges.** Socket-link counters are unreadable, so Task 10 checks (a) the server's `/proc/<pid>/numa_maps`, where every `bind:N` mapping must hold pages on node N only (the same `move_pages`-level fact `page_nodes` samples in-process, but read from outside the server and over every page), (b) each group's RAM, SQPOLL and CPU worker threads' affinities, and (c) that the host fail-stops any lane whose slot is outside its home group's range, which makes (a) cover every CPU job and every staging slot. No per-job export is added.
21. **The device's staging bank is 2-D.** `map_bank["staging"]` becomes `[layers, nodes * lanes]` (node-major), so the launchers' tensor matchers stay 2-D and a row's lists are one contiguous span.
22. **The concurrent CPU-experts plan.** `docs/superpowers/plans/2026-10-03-cpu-experts-normalized-interface.md` (in progress in this checkout) moves both kernels' core handling into `cpu_experts_common/team.hpp` as one process-wide `Cores` with `set_cores`, and makes a concurrent EXL3 forward return 3. Both contradict Part 3's per-engine cores, which need two engines' forwards to run at once. Tasks 2 and 11 are written against the kernels as they are at this plan's base. If the normalized interface lands first, Tasks 2 and 11 become one task on `cpu_experts_common/`: `Cores` becomes the engine's core list behind an `engine_create`/`engine_free` pair in the common C ABI macro, `forward` and `keep_warm` take the engine, and status 3 is kept only for a `free_layer` racing a forward, never for two forwards. Their tests stay as written. The order is the user's to choose (see the reply).
23. **Two groups cannot be timed under the bench service.** `exl3bench.service` isolates node 1's CPUs only (18-33 and their siblings, `bench/service/README.txt`), and its unit is root-installed. A two-group full stack needs node-0 workers too, so Task 10 times one group and two groups outside the service, back to back with no server running, which is noisier than the service's partition. Isolating node-0 cores as well needs an administrator reinstall of the service; the reply asks the user.

## User decisions (2026-10-04)

1. **CPU ABI order:** the normalized CPU-experts interface (`cpu_experts_common/`, started in 1d7e41e463 and a0e863e5e0) lands first. Tasks 2 and 11 are then rewritten as one task on `cpu_experts_common`: `engine_create`/`engine_free` in the common C ABI macro, per-engine cores, and status 3 only for a `free_layer` racing a forward. Tasks 1 and 3-6 do not depend on it and may run before it. Task 7 needs the engine handle.
2. **Production rollout:** land as designed, with no opt-in flag. Production does not pull this branch until Task 10's ms/token arms pass.
3. **Core clash with the bench driver:** node 0's derived engine cores (8-15) overlap `arm_env.DRIVER_CORES`. Accept the contention. CPU-expert arm numbers include the driver's interference, and reports must say so.
4. **Two-group timing:** outside `exl3bench.service` (Task 10's default), with extra repetitions to absorb the noise.
5. **`SGLANG_DSV41_CPU_EXPERTS_THREADS`** truncates the engine's core list (Spec delta 17). This was the plan's default and the user did not override it.

## Review Focus

1. **All misses of a record homed on one node, with that node out of staging slots.** The device must trap on that node's list while the other node's list is untouched, and the host's merged delta from a single reporter must carry the silent node's staging list unchanged. Pinned in Task 4 (`test_a_node_out_of_staging_traps_while_the_other_node_has_slots`) and Task 6 (`test_misses_homed_on_one_node_publish_one_delta_with_the_other_nodes_list_unchanged`).
2. **One group stalled while the other completes.** CopyDone must never be stored, and the abort must name the stalled group, not the one that finished. Pinned in Task 7 (`test_a_group_whose_cpu_job_stalls_is_named_by_the_copy_wait_abort`, `test_a_group_that_never_sends_its_part_is_named_by_the_copy_wait_abort`).
3. **Two engines' OpenMP teams running at once.** Each team on its own cores, bit-exact against one engine, and no team short of workers at `OMP_THREAD_LIMIT` equal to the sum. Pinned in Task 2 (`test_two_engines_at_once_match_one_engine_bit_for_bit`, `test_each_engines_workers_run_on_its_own_cores`, `test_two_full_teams_run_at_once_at_a_thread_limit_of_their_sum`) and Task 1 (`OMP_THREAD_LIMIT` refusal).
4. **ThreadingConfig on a box whose server affinity eats a node's cores.** Start must be refused naming the node, not fall back to the other node's cores or to an unpinned thread. Pinned in Task 1 (`test_a_server_affinity_covering_a_node_is_refused`).
5. **A seam-straddling slot.** It must belong to no group, never be staged or filled, and every other slot must keep its node. Pinned in Task 5 (`test_a_slot_across_a_seam_belongs_to_no_group`) and Task 6 (`test_no_slot_outside_a_groups_range_is_ever_staged_or_taken`).

---

### Task 1: `ThreadingConfig`, the core env vars, one copy of each core check

**Files:**
- Create: `python/sglang/srt/layers/moe/cpu_experts/threading_config.py`
- Create: `test/registered/unit/layers/moe/test_threading_config.py`
- Modify: `python/sglang/srt/environ.py` (after `SGLANG_DSV41_RAM_MISS_SPIN_CORE`, ~:1848: the two URING fields; after `SGLANG_DSV41_CPU_EXPERTS_THREADS`, ~:1913: `SGLANG_EXPERT_NUMA_CORES`)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/pool.py:134-140` (`CpuExpertPool.__init__`'s two checks)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py:98-104` (`CpuExpertService.__init__`'s two checks)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py:1352-1356` (`ExpertStreamHost.start_thread`'s reserved-core check)
- Modify: `benchmarks/dsv41_baseline/test_dsv41_baseline.py:1129-1139` (`test_the_ram_miss_spin_core_has_its_physical_core_to_itself`)

**Interfaces:**
- Consumes: `host_numa.parse_placement` (host_numa.py:55); `envs` fields `SGLANG_MOE_PINNED_HOST_NUMA_MB`, `SGLANG_DSV41_CPU_EXPERTS_CORES`, `SGLANG_DSV41_CPU_EXPERTS_THREADS`, `SGLANG_DSV41_RAM_MISS_SPIN_CORE`.
- Produces (Python, `sglang.srt.layers.moe.cpu_experts.threading_config`):
  - `RESERVED_CORES: frozenset[int]` (64..71); `parse_cpu_list(spec: str) -> list[int]`; `parse_numa_cores(spec: str) -> dict[int, dict[str, list[int]]]`;
  - `check_not_reserved(core: int) -> None`; `check_engine_cores(cores: Sequence[int], threads: int) -> None`;
  - `pci_numa_node(bus_id: str, root: str = "/sys/bus/pci/devices") -> Optional[int]`; `gpu_numa_node(device: Optional[int]) -> Optional[int]`;
  - `@dataclass(frozen=True) Topology(node_cpus: Mapping[int, tuple[int, ...]], siblings: Mapping[int, frozenset[int]])` with `from_sysfs(root: str = "/sys/devices/system")`, `node_of(cpu) -> Optional[int]`, `physical(node) -> list[int]`;
  - `@dataclass(frozen=True) CoreSettings(cpu_experts: bool = False, cores: str = "", threads: int = 0, spin_core: Optional[int] = None, sqpoll: bool = False, sq_thread_cpu: Optional[int] = None, numa_cores: str = "", omp_thread_limit: Optional[int] = None)`;
  - `@dataclass(frozen=True) NodePlan(group: int, node: int, ram: Optional[int], cpu: tuple[int, ...], sq: Optional[int], busy_poll: bool)` with `threads -> int` (`len(cpu)`) and `log_line() -> str`;
  - `@dataclass(frozen=True) ThreadingConfig(plans: tuple[NodePlan, ...], copy_cpus: tuple[int, ...], gpu_node: int)` with `resolve(*, nodes: Sequence[int], gpu_node: Optional[int], affinity: Iterable[int], topology: Topology, settings: CoreSettings) -> ThreadingConfig` (static), `from_env(*, cpu_experts: bool, device: Optional[int]) -> ThreadingConfig` (class method), `nodes -> int`, `log_lines() -> list[str]`.
- Produces (env, `Envs`): `SGLANG_EXPERT_NUMA_CORES = EnvStr("")`, `SGLANG_EXPERT_STREAM_URING_MODE = EnvStr("default")`, `SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU = EnvInt(None)`.

- [ ] **Step 1: Write the failing test** `test/registered/unit/layers/moe/test_threading_config.py`:

```python
"""ThreadingConfig on fake sysfs trees: the design's derivation, today's one-node runtime, the override and every
refusal (spec 2026-10-03-numa-node-distributor-design, Part 2)."""

import pytest

from sglang.srt.layers.moe.cpu_experts import threading_config as tc
from sglang.srt.layers.moe.cpu_experts.threading_config import (
    CoreSettings,
    NodePlan,
    ThreadingConfig,
    Topology,
    check_engine_cores,
    check_not_reserved,
    parse_cpu_list,
    parse_numa_cores,
    pci_numa_node,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DIVIX01 = {0: "0-17,36-53", 1: "18-35,54-71"}
SERVER = frozenset(parse_cpu_list("0-7,16,36-52"))


def fake_sysfs(root, nodes, pairs):
    """A /sys/devices/system tree: each node's cpulist, with cpu c and c + pairs one physical core (c < pairs)."""
    for node, cpus in nodes.items():
        node_dir = root / "node" / f"node{node}"
        node_dir.mkdir(parents=True)
        (node_dir / "cpulist").write_text(cpus + "\n")
        for cpu in parse_cpu_list(cpus):
            topology = root / "cpu" / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            low = cpu % pairs
            (topology / "thread_siblings_list").write_text(f"{low},{low + pairs}\n")
    return Topology.from_sysfs(str(root))


@pytest.fixture
def divix01(tmp_path):
    return fake_sysfs(tmp_path, DIVIX01, 36)


def resolve(topology, *, nodes=(0, 1), affinity=SERVER, gpu_node=0, **settings):
    return ThreadingConfig.resolve(
        nodes=list(nodes), gpu_node=gpu_node, affinity=affinity, topology=topology, settings=CoreSettings(**settings)
    )


def test_divix01_derives_the_designs_plan(divix01):
    config = resolve(divix01, cpu_experts=True, threads=16)
    assert config.plans == (
        NodePlan(group=0, node=0, ram=17, cpu=tuple(range(8, 16)), sq=None, busy_poll=True),
        NodePlan(group=1, node=1, ram=35, cpu=tuple(range(18, 34)), sq=None, busy_poll=True),
    )
    assert config.copy_cpus == tuple(sorted(SERVER))
    assert config.log_lines() == [
        "numa node0: ram=17 cpu=8-15 (8) sq=-",
        "numa node1: ram=35 cpu=18-33 (16) sq=-",
        "numa copy thread: node0 cpus=0-7,16,36-52",
    ]


def test_without_a_threads_cap_the_engine_takes_every_remaining_core(divix01):
    config = resolve(divix01, cpu_experts=True)
    assert config.plans[1].cpu == tuple(range(18, 35)) and config.plans[1].threads == 17


def test_sqpoll_takes_the_next_usable_core_below_the_ram_core(divix01):
    config = resolve(divix01, cpu_experts=True, sqpoll=True)
    assert [(p.ram, p.sq, p.cpu[0], p.cpu[-1]) for p in config.plans] == [(17, 15, 8, 14), (35, 34, 18, 33)]


def test_two_nodes_without_cpu_experts_pin_only_the_ram_threads(divix01):
    config = resolve(divix01, spin_core=17)
    assert [(p.ram, p.cpu, p.busy_poll) for p in config.plans] == [(17, (), True), (35, (), True)]


def test_one_node_without_numa_or_cpu_experts_is_todays_runtime(divix01):
    everything = frozenset(range(72))
    config = resolve(divix01, nodes=(0,), affinity=everything)
    assert config.plans == (NodePlan(group=0, node=0, ram=None, cpu=(), sq=None, busy_poll=False),)
    assert config.copy_cpus == ()
    pinned = resolve(divix01, nodes=(0,), affinity=everything, spin_core=17, sq_thread_cpu=15, sqpoll=True)
    assert pinned.plans[0] == NodePlan(group=0, node=0, ram=17, cpu=(), sq=15, busy_poll=True)


def test_the_override_replaces_the_plans_of_the_nodes_it_names(divix01):
    config = resolve(divix01, cpu_experts=True, sqpoll=True, numa_cores="1:ram=34,cpu=18-25,27,sq=33")
    assert config.plans[1] == NodePlan(
        group=1, node=1, ram=34, cpu=(18, 19, 20, 21, 22, 23, 24, 25, 27), sq=33, busy_poll=True
    )
    assert (config.plans[0].ram, config.plans[0].sq) == (17, 15), "node 0 is still derived"


def test_cpu_experts_cores_is_the_cpu_override_of_its_node(divix01):
    config = resolve(divix01, cpu_experts=True, cores="20-23")
    assert config.plans[1].cpu == (20, 21, 22, 23) and config.plans[1].ram == 35


def test_a_server_affinity_covering_a_node_is_refused(divix01):
    """Review Focus 4: the server's affinity takes all of node 1. Start is refused naming the node; nothing falls back
    to node 0's cores or to an unpinned thread."""
    with pytest.raises(ValueError, match="node 1 has no usable core"):
        resolve(divix01, affinity=SERVER | frozenset(parse_cpu_list("18-35,54-71")), cpu_experts=True)


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"numa_cores": "1:ram=35,cpu=18-33", "affinity": SERVER | {20}}, "core 20 is in the server's affinity"),
        ({"numa_cores": "1:ram=64,cpu=18-33"}, "core 64 is reserved"),
        ({"numa_cores": "1:ram=35,cpu=18"}, "at least 2 cores"),
        ({"numa_cores": "1:ram=17,cpu=18-33"}, "core 17 is on node 0"),
        ({"numa_cores": "1:ram=35,cpu=18-33,54"}, "cores 18 and 54 share a physical core"),
        ({"numa_cores": "2:ram=35,cpu=18-33"}, "node 2 is not one of the tier's nodes"),
        ({"numa_cores": "1:ram=35,cpu=18-33", "cores": "20-23"}, "both"),
        ({"cores": "10-11,20-21"}, "two nodes"),
        ({"cores": "18-29", "nodes": (0,)}, "core 18 is on node 1"),
        ({"spin_core": 8}, "core 8 shares a physical core with the server's affinity"),
        ({"omp_thread_limit": 16}, "OMP_THREAD_LIMIT"),
    ],
)
def test_each_refusal(divix01, settings, match):
    settings = dict(settings)
    affinity = settings.pop("affinity", SERVER)
    nodes = settings.pop("nodes", (0, 1))
    with pytest.raises(ValueError, match=match):
        resolve(divix01, nodes=nodes, affinity=affinity, cpu_experts=True, **settings)


@pytest.mark.parametrize(
    "spec, match",
    [
        ("1:ram=35;1:ram=34", "twice"),
        ("x:ram=1", "node"),
        ("1:foo=3", "unknown key"),
        ("1:ram=34-35", "one core"),
        ("1:18-33", "key"),
    ],
)
def test_a_malformed_override_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_numa_cores(spec)


def test_the_override_parses_lists_inside_a_key():
    assert parse_numa_cores("1:ram=35,cpu=18-20,22,sq=34; 0:ram=17") == {
        1: {"ram": [35], "cpu": [18, 19, 20, 22], "sq": [34]},
        0: {"ram": [17]},
    }


def test_the_shared_core_checks():
    assert parse_cpu_list("36-38, 40,36") == [36, 37, 38, 40]
    with pytest.raises(ValueError, match="backwards"):
        parse_cpu_list("9-3")
    with pytest.raises(ValueError, match="64-71"):
        check_not_reserved(71)
    check_not_reserved(63)
    with pytest.raises(ValueError, match="at least 2 cores"):
        check_engine_cores([4, 4], 1)
    with pytest.raises(ValueError, match="3 CPU expert threads on 2 cores"):
        check_engine_cores([4, 5], 3)


def test_the_gpus_node_comes_from_its_pci_device(tmp_path):
    (tmp_path / "0000:41:00.0").mkdir()
    (tmp_path / "0000:41:00.0" / "numa_node").write_text("1\n")
    (tmp_path / "0000:01:00.0").mkdir()
    (tmp_path / "0000:01:00.0" / "numa_node").write_text("-1\n")
    assert pci_numa_node("0000:41:00.0", str(tmp_path)) == 1
    assert pci_numa_node("0000:01:00.0", str(tmp_path)) is None, "-1: the platform reports no node"


def test_from_env_reads_every_core_setting(monkeypatch):
    from sglang.srt.environ import envs

    seen = {}
    monkeypatch.setattr(ThreadingConfig, "resolve", staticmethod(lambda **kw: seen.update(kw) or "resolved"))
    monkeypatch.setattr(Topology, "from_sysfs", classmethod(lambda cls, root=tc.SYSFS: "topology"))
    monkeypatch.setattr(tc, "_affinity", lambda: frozenset({0, 1}))
    monkeypatch.setattr(tc, "gpu_numa_node", lambda device: None)
    monkeypatch.setenv("OMP_THREAD_LIMIT", "24")
    with envs.SGLANG_DSV41_CPU_EXPERTS_CORES.override("18-29"), \
            envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.override(4), \
            envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.override(17), \
            envs.SGLANG_EXPERT_STREAM_URING_MODE.override("sqpoll_iopoll"), \
            envs.SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU.override(15), \
            envs.SGLANG_EXPERT_NUMA_CORES.override("1:ram=35,cpu=18-33"), \
            envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("1:1024,0:1024"):
        assert ThreadingConfig.from_env(cpu_experts=True, device=None) == "resolved"
    assert seen == {
        "nodes": [1, 0],
        "gpu_node": None,
        "affinity": frozenset({0, 1}),
        "topology": "topology",
        "settings": CoreSettings(
            cpu_experts=True, cores="18-29", threads=4, spin_core=17, sqpoll=True, sq_thread_cpu=15,
            numa_cores="1:ram=35,cpu=18-33", omp_thread_limit=24,
        ),
    }
```

- [ ] **Step 2: Run it to verify it fails.** Commit the test alone:

```bash
git add test/registered/unit/layers/moe/test_threading_config.py
git diff --cached --stat
git commit -m "test(numa): ThreadingConfig on fake sysfs trees (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_threading_config.py`.
Expected: FAIL, `ModuleNotFoundError: No module named 'sglang.srt.layers.moe.cpu_experts.threading_config'`.

- [ ] **Step 3: Implement.**

`python/sglang/srt/environ.py`, after `SGLANG_DSV41_RAM_MISS_SPIN_CORE`:

```python
    # io_uring options the C++ reader also reads itself (host/uring_options.h); declared here because
    # ThreadingConfig resolves the SQPOLL thread's core from them. The service passes every ring its core explicitly.
    SGLANG_EXPERT_STREAM_URING_MODE = EnvStr("default")
    # The SQPOLL thread's core with one NUMA group, -1 or unset unpinned; refused above one group (use
    # SGLANG_EXPERT_NUMA_CORES's sq=).
    SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU = EnvInt(None)
```

and after `SGLANG_DSV41_CPU_EXPERTS_THREADS`:

```python
    # Per-node thread plans replacing the derived ones (cpu_experts/threading_config.py), e.g.
    # "1:ram=35,cpu=18-33,sq=34;0:ram=17": nodes separated by ";", keys ram, cpu, sq, each a taskset list. Validated
    # like a derived plan; a refused plan stops the start. Empty derives every node's plan.
    SGLANG_EXPERT_NUMA_CORES = EnvStr("")
```

`python/sglang/srt/layers/moe/cpu_experts/threading_config.py`:

```python
"""Where the expert stream's threads run: one plan per NUMA node of the pinned tier.

ThreadingConfig is the only reader of the core settings (SGLANG_DSV41_CPU_EXPERTS_CORES and _THREADS,
SGLANG_DSV41_RAM_MISS_SPIN_CORE, SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU, SGLANG_EXPERT_NUMA_CORES), of the process
affinity and of the reserved cores; C++ receives resolved core lists only. The rules are the design's
(docs/superpowers/specs/2026-10-03-numa-node-distributor-design.md, Part 2). ``resolve`` reads nothing, so it runs
on any topology; ``from_env`` gathers the machine's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence

# NVMe completion interrupts are pinned to these cores on divix01.
RESERVED_CORES = frozenset(range(64, 72))
SYSFS = "/sys/devices/system"
PCI_DEVICES = "/sys/bus/pci/devices"


def parse_cpu_list(spec: str) -> list[int]:
    """Cores from a sysfs or taskset list such as "0-17,36-53", sorted and unique."""
    cores: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        first, last = int(lo), int(hi) if sep else int(lo)
        if last < first:
            raise ValueError(f"core range {part!r} runs backwards")
        cores.update(range(first, last + 1))
    return sorted(cores)


def _format_cpus(cores: Sequence[int]) -> str:
    """The inverse of parse_cpu_list: "8-15", "0-7,16,36-52", or "-" for none."""
    runs: list[list[int]] = []
    for core in sorted(cores):
        if runs and core == runs[-1][1] + 1:
            runs[-1][1] = core
        else:
            runs.append([core, core])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs) or "-"


def parse_numa_cores(spec: str) -> dict[int, dict[str, list[int]]]:
    """SGLANG_EXPERT_NUMA_CORES as {node: {"ram" | "cpu" | "sq": cores}}; raises ValueError when malformed."""
    plans: dict[int, dict[str, list[int]]] = {}
    for entry in filter(None, (e.strip() for e in spec.split(";"))):
        node_text, sep, body = entry.partition(":")
        if not sep or not node_text.strip().isdigit():
            raise ValueError(f"SGLANG_EXPERT_NUMA_CORES entry {entry!r} does not start with a node, as in 1:ram=35")
        node = int(node_text)
        if node in plans:
            raise ValueError(f"SGLANG_EXPERT_NUMA_CORES names node {node} twice")
        items: dict[str, list[str]] = {}
        key = None
        for item in (i.strip() for i in body.split(",")):
            name, eq, value = item.partition("=")
            if eq:
                key = name.strip()
                if key not in ("ram", "cpu", "sq"):
                    raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: unknown key {key!r}; expected ram, cpu or sq")
                if key in items:
                    raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: node {node} sets {key} twice")
                items[key] = [value]
            elif key is None:
                raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: {item!r} on node {node} follows no key")
            else:
                items[key].append(item)
        plans[node] = {k: parse_cpu_list(",".join(v)) for k, v in items.items()}
        for k in ("ram", "sq"):
            if k in plans[node] and len(plans[node][k]) != 1:
                raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: node {node}'s {k} is one core")
    return plans


def check_not_reserved(core: int) -> None:
    """Raise ValueError for a core in RESERVED_CORES."""
    if core in RESERVED_CORES:
        raise ValueError(f"core {core} is reserved: cores 64-71 take NVMe completion interrupts")


def check_engine_cores(cores: Sequence[int], threads: int) -> None:
    """Raise ValueError unless a CPU expert engine can run ``threads`` workers on ``cores``."""
    distinct = sorted(set(cores))
    if len(distinct) < 2:
        # Spinning workers sharing one core livelock (DSV41_REFERENCE.md section 28.2).
        raise ValueError(f"CPU experts need at least 2 cores, got {distinct}")
    if not 1 <= threads <= len(distinct):
        raise ValueError(f"{threads} CPU expert threads on {len(distinct)} cores")


def pci_numa_node(bus_id: str, root: str = PCI_DEVICES) -> Optional[int]:
    """The NUMA node of PCI device ``bus_id`` ("0000:41:00.0"), None when the platform reports none (-1)."""
    with open(os.path.join(root, bus_id, "numa_node")) as f:
        node = int(f.read())
    return node if node >= 0 else None


def gpu_numa_node(device: Optional[int]) -> Optional[int]:
    """The NUMA node of CUDA device ``device``; None without a device or a reported node."""
    if device is None:
        return None
    import torch

    p = torch.cuda.get_device_properties(device)
    return pci_numa_node(f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0")


def _affinity() -> frozenset[int]:
    return frozenset(os.sched_getaffinity(0))


@dataclass(frozen=True)
class Topology:
    node_cpus: Mapping[int, tuple[int, ...]]  # node -> its CPUs, ascending
    siblings: Mapping[int, frozenset[int]]  # CPU -> its SMT siblings, itself included

    @classmethod
    def from_sysfs(cls, root: str = SYSFS) -> "Topology":
        node_dir = os.path.join(root, "node")
        node_cpus: dict[int, tuple[int, ...]] = {}
        for name in sorted(os.listdir(node_dir)):
            if name.startswith("node") and name[4:].isdigit():
                with open(os.path.join(node_dir, name, "cpulist")) as f:
                    node_cpus[int(name[4:])] = tuple(parse_cpu_list(f.read()))
        siblings = {}
        for cpus in node_cpus.values():
            for cpu in cpus:
                with open(os.path.join(root, "cpu", f"cpu{cpu}", "topology", "thread_siblings_list")) as f:
                    siblings[cpu] = frozenset(parse_cpu_list(f.read()))
        return cls(node_cpus, siblings)

    def node_of(self, cpu: int) -> Optional[int]:
        return next((node for node, cpus in self.node_cpus.items() if cpu in cpus), None)

    def physical(self, node: int) -> list[int]:
        """The node's physical cores, each named by its lowest-numbered SMT thread."""
        return [cpu for cpu in self.node_cpus[node] if min(self.siblings[cpu]) == cpu]


@dataclass(frozen=True)
class CoreSettings:
    cpu_experts: bool = False  # SGLANG_DSV41_CPU_EXPERTS
    cores: str = ""  # SGLANG_DSV41_CPU_EXPERTS_CORES
    threads: int = 0  # SGLANG_DSV41_CPU_EXPERTS_THREADS, 0: no cap
    spin_core: Optional[int] = None  # SGLANG_DSV41_RAM_MISS_SPIN_CORE
    sqpoll: bool = False  # SGLANG_EXPERT_STREAM_URING_MODE names a sqpoll mode
    sq_thread_cpu: Optional[int] = None  # SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU
    numa_cores: str = ""  # SGLANG_EXPERT_NUMA_CORES
    omp_thread_limit: Optional[int] = None  # OMP_THREAD_LIMIT


@dataclass(frozen=True)
class NodePlan:
    group: int  # the wire's node axis: home(expert) == group
    node: int  # the NUMA node id
    ram: Optional[int]  # the RAM/NVMe thread's core; None inherits the server's affinity
    cpu: tuple[int, ...]  # the CPU expert engine's cores, cpu[0] its thread (worker 0); () without CPU experts
    sq: Optional[int]  # the io_uring SQPOLL thread's core; None unpinned
    busy_poll: bool

    @property
    def threads(self) -> int:
        return len(self.cpu)

    def cores(self) -> list[int]:
        return [c for c in (self.ram, self.sq) if c is not None] + list(self.cpu)

    def log_line(self) -> str:
        ram = "-" if self.ram is None else str(self.ram)
        sq = "-" if self.sq is None else str(self.sq)
        return f"numa node{self.node}: ram={ram} cpu={_format_cpus(self.cpu)} ({self.threads}) sq={sq}"


@dataclass(frozen=True)
class ThreadingConfig:
    plans: tuple[NodePlan, ...]  # one per node of the tier, in placement order
    copy_cpus: tuple[int, ...]  # the copy thread's affinity; () inherits the server's
    gpu_node: int

    @property
    def nodes(self) -> int:
        return len(self.plans)

    def log_lines(self) -> list[str]:
        lines = [plan.log_line() for plan in self.plans]
        if self.copy_cpus:
            lines.append(f"numa copy thread: node{self.gpu_node} cpus={_format_cpus(self.copy_cpus)}")
        return lines

    @classmethod
    def from_env(cls, *, cpu_experts: bool, device: Optional[int]) -> "ThreadingConfig":
        """The machine's plan: its sysfs topology, this process's affinity and the core env vars."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.host_numa import parse_placement

        limit = os.environ.get("OMP_THREAD_LIMIT")  # OpenMP's own variable, outside Envs
        settings = CoreSettings(
            cpu_experts=cpu_experts,
            cores=envs.SGLANG_DSV41_CPU_EXPERTS_CORES.get(),
            threads=envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.get(),
            spin_core=envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.get(),
            sqpoll="sqpoll" in envs.SGLANG_EXPERT_STREAM_URING_MODE.get(),
            sq_thread_cpu=envs.SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU.get(),
            numa_cores=envs.SGLANG_EXPERT_NUMA_CORES.get(),
            omp_thread_limit=int(limit) if limit else None,
        )
        placement = parse_placement(envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.get())
        gpu_node = gpu_numa_node(device)
        return cls.resolve(
            nodes=[node for node, _ in placement] or [gpu_node if gpu_node is not None else 0],
            gpu_node=gpu_node,
            affinity=_affinity(),
            topology=Topology.from_sysfs(),
            settings=settings,
        )

    @staticmethod
    def resolve(
        *,
        nodes: Sequence[int],
        gpu_node: Optional[int],
        affinity: Iterable[int],
        topology: Topology,
        settings: CoreSettings,
    ) -> "ThreadingConfig":
        """Every node's plan by the design's rules, validated; raises ValueError naming the node and the rule."""
        affinity = frozenset(affinity)
        gpu = gpu_node if gpu_node is not None else nodes[0]
        overrides = parse_numa_cores(settings.numa_cores)
        for node in overrides:
            if node not in nodes:
                raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: node {node} is not one of the tier's nodes {list(nodes)}")
        if settings.cores:
            cores = parse_cpu_list(settings.cores)
            homes = {topology.node_of(c) for c in cores}
            if len(homes) != 1:
                raise ValueError(f"SGLANG_DSV41_CPU_EXPERTS_CORES {settings.cores!r} spans two nodes")
            home = homes.pop()
            if home not in nodes:
                raise ValueError(f"SGLANG_DSV41_CPU_EXPERTS_CORES: core {cores[0]} is on node {home}, off the tier")
            if "cpu" in overrides.get(home, {}):
                raise ValueError(
                    f"node {home}'s CPU cores are set both by SGLANG_DSV41_CPU_EXPERTS_CORES and SGLANG_EXPERT_NUMA_CORES"
                )
            overrides.setdefault(home, {})["cpu"] = cores
        if settings.spin_core is not None:
            check_not_reserved(settings.spin_core)
            home = topology.node_of(settings.spin_core)
            if home not in nodes:
                raise ValueError(f"SGLANG_DSV41_RAM_MISS_SPIN_CORE: core {settings.spin_core} is on node {home}")
            overrides.setdefault(home, {}).setdefault("ram", [settings.spin_core])
        if settings.sq_thread_cpu is not None and settings.sq_thread_cpu >= 0:
            if len(nodes) > 1:
                raise ValueError(
                    "SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU names one core for several NUMA groups; "
                    "set each group's sq= in SGLANG_EXPERT_NUMA_CORES"
                )
            overrides.setdefault(nodes[0], {}).setdefault("sq", [settings.sq_thread_cpu])
        derive = len(nodes) > 1 or settings.cpu_experts or bool(settings.numa_cores)
        if not derive:
            ram = settings.spin_core
            sq = overrides.get(nodes[0], {}).get("sq", [None])[0] if settings.sqpoll else None
            plan = NodePlan(group=0, node=nodes[0], ram=ram, cpu=(), sq=sq, busy_poll=ram is not None)
            return ThreadingConfig((plan,), (), gpu)
        plans = []
        taken: set[int] = set()
        for group, node in enumerate(nodes):
            plan = _derive(group, node, overrides.get(node, {}), topology, affinity, settings, taken)
            _check_plan(plan, topology, affinity, settings)
            taken.update(s for c in plan.cores() for s in topology.siblings[c])
            plans.append(plan)
        workers = sum(p.threads for p in plans)
        if settings.omp_thread_limit is not None and settings.omp_thread_limit < workers:
            raise ValueError(
                f"OMP_THREAD_LIMIT={settings.omp_thread_limit} is below the {workers} CPU expert workers of all nodes; "
                "two engines' teams run at once"
            )
        copy = sorted(c for c in topology.node_cpus[gpu] if c in affinity)
        if not copy:
            copy = sorted(c for c in topology.node_cpus[gpu] if c not in RESERVED_CORES and c not in taken)
        return ThreadingConfig(tuple(plans), tuple(copy), gpu)


def _derive(group, node, override, topology, affinity, settings, taken) -> NodePlan:
    """One node's plan: the override's keys as given, every other role from the node's usable cores."""
    usable = [
        c
        for c in reversed(topology.physical(node))
        if c not in affinity and c not in RESERVED_CORES and not (topology.siblings[c] & taken)
    ]
    named = {c for cores in override.values() for c in cores}
    free = [c for c in usable if c not in named]

    def dedicated(core: int) -> bool:
        return not (topology.siblings[core] & affinity)

    if "ram" in override:
        ram = override["ram"][0]
    else:
        ram = next((c for c in free if dedicated(c)), None)
        if ram is None:
            raise ValueError(
                f"node {node} has no usable core for its RAM thread"
                if not usable
                else f"node {node} has no usable core whose physical core is outside the server's affinity"
            )
        free.remove(ram)
    sq = None
    if settings.sqpoll:
        if "sq" in override:
            sq = override["sq"][0]
        elif free:
            sq = free.pop(0)
    cpu: tuple[int, ...] = ()
    if settings.cpu_experts:
        listed = override["cpu"] if "cpu" in override else sorted(free)
        cpu = tuple(listed[: settings.threads] if settings.threads else listed)
    if not usable and not override:
        raise ValueError(f"node {node} has no usable core")
    return NodePlan(group=group, node=node, ram=ram, cpu=cpu, sq=sq, busy_poll=True)


def _check_plan(plan: NodePlan, topology: Topology, affinity: frozenset[int], settings: CoreSettings) -> None:
    """The design's refusals for one plan, derived or given."""
    cores = plan.cores()
    for core in cores:
        if core in RESERVED_CORES:
            raise ValueError(f"node {plan.node}: core {core} is reserved (64-71 take NVMe completion interrupts)")
        home = topology.node_of(core)
        if home != plan.node:
            raise ValueError(f"node {plan.node}: core {core} is on node {home}")
        if core in affinity:
            raise ValueError(f"node {plan.node}: core {core} is in the server's affinity")
    for a, b in combinations(cores, 2):
        if b in topology.siblings[a]:
            raise ValueError(f"node {plan.node}: cores {min(a, b)} and {max(a, b)} share a physical core")
    if plan.ram is not None and topology.siblings[plan.ram] & affinity:
        raise ValueError(
            f"node {plan.node}: core {plan.ram} shares a physical core with the server's affinity, "
            "so its busy-polling RAM thread would not have the core to itself"
        )
    if settings.cpu_experts:
        check_engine_cores(plan.cpu, plan.threads)
```

The derivation, by hand: node 0's usable cores, highest first, are `17, 15, 14, ..., 8` (0-7 and 16 are the server's), so `ram=17` and the engine `8-15`; node 1's are `35..18`, so `ram=35`, `sq=34` with SQPOLL, and the engine the rest ascending. A server affinity covering `18-35` leaves node 1 nothing, and the RAM step raises `node 1 has no usable core for its RAM thread`.

`pool.py`, `CpuExpertPool.__init__`, replace the two `if` checks (`:134-140`, from `cores = sorted(set(cores))` through the `threads` check) with:

```python
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_engine_cores

        cores = sorted(set(cores))
        check_engine_cores(cores, threads)
```

`service.py`, `CpuExpertService.__init__`, the same replacement for `:98-104`.

`expert_stream_transport.py`, `ExpertStreamHost.start_thread`, replace the `if 64 <= cpu_core <= 71: raise ValueError(...)` block with:

```python
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_not_reserved

        check_not_reserved(cpu_core)
```

`benchmarks/dsv41_baseline/test_dsv41_baseline.py`, replace `test_the_ram_miss_spin_core_has_its_physical_core_to_itself` with:

```python
def test_the_ram_miss_spin_core_has_its_physical_core_to_itself(tmp_path):
    # ThreadingConfig is the one copy of the rule (busy-polling core: no SMT sibling in the server's affinity).
    tc = pytest.importorskip("sglang.srt.layers.moe.cpu_experts.threading_config")
    for node, cpus in {0: "0-17,36-53", 1: "18-35,54-71"}.items():
        node_dir = tmp_path / "node" / f"node{node}"
        node_dir.mkdir(parents=True)
        (node_dir / "cpulist").write_text(cpus)
        for cpu in _cores(cpus):
            topology = tmp_path / "cpu" / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "thread_siblings_list").write_text(f"{cpu % 36},{cpu % 36 + 36}")
    nodes = [int(entry.split(":")[0]) for entry in arm_env.PINNED_HOST_NUMA_MB.split(",")]
    config = tc.ThreadingConfig.resolve(
        nodes=nodes, gpu_node=0, affinity=_cores(arm_env.SERVER_CORES),
        topology=tc.Topology.from_sysfs(str(tmp_path)), settings=tc.CoreSettings(spin_core=arm_env.SPIN_CORE),
    )
    assert config.plans[0].ram == arm_env.SPIN_CORE and config.plans[0].busy_poll
    assert not (set(config.copy_cpus) & _cores(arm_env.DRIVER_CORES))
    assert arm_env.base_env()["SGLANG_DSV41_RAM_MISS_SPIN_CORE"] == str(arm_env.SPIN_CORE)
```

- [ ] **Step 4: Run.** Commit:

```bash
git add python/sglang/srt/layers/moe/cpu_experts/threading_config.py python/sglang/srt/layers/moe/cpu_experts/pool.py python/sglang/srt/layers/moe/cpu_experts/service.py benchmarks/dsv41_baseline/test_dsv41_baseline.py
git add -p python/sglang/srt/environ.py python/sglang/kernels/ops/moe/expert_stream_transport.py
git diff --cached --stat
git commit -m "feat(numa): ThreadingConfig resolves every expert-stream core; one copy of each core check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_threading_config.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py benchmarks/dsv41_baseline/test_dsv41_baseline.py`.
Expected: PASS (`test_a_reserved_or_unusable_core_is_refused` still raises `ValueError` matching `64-71`).

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU`. Expected: Baseline counts plus this task's tests. A laptop run of the new test is optional (`python -m pytest test/registered/unit/layers/moe/test_threading_config.py` in a venv with torch).

---

### Task 2: CPU kernel engine handle in the EXL3 C ABI

Written against the kernel at this plan's base; read Spec delta 22 before starting it.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_expert_forward_abi.h` (version 2, `engine` appended)
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/cpu_experts_cabi.h` (`set_cores` out; `engine_create`, `engine_free`; `keep_warm` takes the engine)
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp:2103-2135` (core globals), `:2167-2184` (`ForwardCtx`), `:2543-2612` (`forward_raw`), `:2615-2627` (`exl3_moe_cpu_forward_raw`), `:2673-2693` (`sglang_exl3_cpu_experts_forward`), `:2745-2790` (`keep_warm`, `set_cores`)
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/forward_plan.hpp:152-178` (`run_team`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h:49-87` (`CpuExpertKeepWarm`, `CpuExpertConfig`), `:236-237` and `:250-262` (`run`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h:352-399` (`enable_cpu_experts`), `ffi_test_exports.h:650-655` (`test_keep_warm`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamHost.enable_cpu_experts`, ~:1614)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/pool.py:18-118` (ABI mirror, Protocol), `exl3.py:141-172`, `service.py:128-141,186-189`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/cpu_forward.cpp:195-205,294-295`, `full_stack.cpp:134-145,285-297,505-509`, `stack.h:95-98,154-166`
- Modify (tests whose API changes): `test/registered/unit/kernels/test_cpu_expert_pool.py:330-425` (`FakeServiceTrait`, `FakeHost`), `test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py:41-110` (`FakeForward`, `_host`), `test/manual/dsv41/test_cpu_expert_pool_exl3.py:174-206` (keep-warm prototype)
- Create: `test/manual/dsv41/test_cpu_expert_engines_exl3.py`

**Interfaces:**
- Produces (C ABI, `cpu_expert_forward_abi.h`): `SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION 2u`; `SglangCpuExpertsForward` = v1's fields, then `int64_t engine` (0: no engine).
- Produces (EXL3 kernel): `int sglang_exl3_cpu_experts_engine_create(const int32_t* cores, int32_t n, int64_t* engine)` (0, 1, 2); `int sglang_exl3_cpu_experts_engine_free(int64_t engine)`; `int sglang_exl3_cpu_experts_keep_warm(int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns)`; `sglang_exl3_cpu_experts_forward` reads `call->engine` and returns 2 for an unknown engine or `threads` above its cores. `sglang_exl3_cpu_experts_set_cores` is deleted.
- Produces (host): `using CpuExpertKeepWarm = int (*)(int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns);` `CpuExpertConfig::engine` (`int64_t`, default 0). FFI `expert_stream_enable_cpu_experts(handle, forward, engine, split, cores, x_rows, out_rows, hidden, parts, threads, spin_ns, keep_warm, keep_warm_ns)`.
- Produces (Python): `ExpertStreamHost.enable_cpu_experts(forward, split, cores, x_rows, out_rows, *, threads, engine: int = 0, spin_us=50_000, keep_warm=0, keep_warm_us=0)`; `pool.CPU_EXPERTS_FORWARD_ABI_VERSION = 2`; `CpuExpertsForwardCall` gains `("engine", ctypes.c_int64)` last; `CpuExpertQuantTrait.native_create_engine(cores: Sequence[int]) -> int` and `native_free_engine(engine: int) -> None` replace `native_set_cores`; `CpuExpertService.engine: int`.

- [ ] **Step 1: Write the failing tests.**

`test/manual/dsv41/test_cpu_expert_engines_exl3.py`:

```python
"""Two CPU expert engines of the EXL3 kernel on disjoint cores (spec 2026-10-03-numa-node-distributor-design,
Testing 4). Each engine runs its own OpenMP team on its own cores; two running at once give, bit for bit, what one
gives alone. Needs the optimized ext build (RUN_EXT) and 4 cores in the affinity mask."""

import ctypes
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

sys.path.insert(0, os.path.dirname(__file__))
from test_cpu_expert_pool_exl3 import CAP, LIMIT, _random_slabs  # noqa: E402

HIDDEN, INTER = 5120, 2304  # DeepSeek V4.1's shape: the DSV4.1 plan on AVX-512BW
REPEATS = 40
KEEP_WARM = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64)


def _kernel():
    from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
    from sglang.srt.layers.quantization.exl3_ext import cpu_act_defines, exl3_ext, optimized_cpu

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the engine ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < 4:
        pytest.skip("needs 4 cores in the affinity mask")
    return Exl3CpuQuantTrait(exl3_ext(), act_limit=LIMIT), cores[:4]


def _forward(trait, layer, x, slots, weights, engine, threads):
    from sglang.srt.layers.moe.cpu_experts.pool import (
        CPU_EXPERTS_FORWARD_ABI_VERSION,
        CpuExpertForward,
        CpuExpertsForwardCall,
    )

    s = torch.tensor(slots, dtype=torch.int32)
    w = torch.tensor(weights, dtype=torch.float32)
    out = torch.full((HIDDEN,), float("nan"))
    call = CpuExpertsForwardCall(
        abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION, rows=1, layer=layer, x=x.data_ptr(),
        slots=ctypes.cast(s.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
        weights=ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        out=ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)), k=len(slots), threads=threads, accumulate=0,
        engine=engine,
    )
    return CpuExpertForward(trait.native_forward())(ctypes.byref(call)), out


def _inputs():
    g = torch.Generator().manual_seed(20261004)
    return [
        ((torch.randn(HIDDEN, generator=g) * scale).half(), [i % CAP, (i + 2) % CAP, (i + 5) % CAP], [0.5, 0.3, 0.2])
        for i, scale in enumerate([1.0, 4.0, 8.0, 0.5] * 3)
    ]


def test_two_engines_at_once_match_one_engine_bit_for_bit(monkeypatch):
    """Review Focus 3. Mutants: a process-wide core list (engine B's team pinned onto A's cores) -- red on the
    affinity test below; a shared static scratch -- red here."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    a, b = trait.native_create_engine(cores[:2]), trait.native_create_engine(cores[2:4])
    layer = trait.register_layer(_random_slabs(20261003, HIDDEN, INTER), CAP)
    inputs = _inputs()
    try:
        want = []
        for x, slots, weights in inputs:
            rc, out = _forward(trait, layer, x, slots, weights, a, 2)
            assert rc == 0 and torch.isfinite(out).all()
            want.append(out)
        bad = []

        def run(engine):
            for _ in range(REPEATS):
                for i, (x, slots, weights) in enumerate(inputs):
                    rc, out = _forward(trait, layer, x, slots, weights, engine, 2)
                    if rc != 0 or not torch.equal(out, want[i]):
                        bad.append((engine, i, rc))

        workers = [threading.Thread(target=run, args=(engine,)) for engine in (a, b)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert bad == []
    finally:
        trait.free_layer(layer)
        trait.native_free_engine(a)
        trait.native_free_engine(b)


def test_each_engines_workers_run_on_its_own_cores(monkeypatch):
    """Two keep-warms at once, one per engine: while they run, the process has a thread pinned to each of the four
    engine cores. Mutant: pin every worker from one core list -- red."""
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    a, b = trait.native_create_engine(cores[:2]), trait.native_create_engine(cores[2:4])
    keep_warm = KEEP_WARM(trait.native_keep_warm())
    word = torch.zeros(1, dtype=torch.int32)
    deadline = time.monotonic_ns() + 10_000_000_000
    rc = {}
    runners = [
        threading.Thread(target=lambda e=e: rc.setdefault(e, keep_warm(e, 2, word.data_ptr(), 0, deadline)))
        for e in (a, b)
    ]
    try:
        for runner in runners:
            runner.start()
        time.sleep(0.3)
        pinned = set()
        for tid in os.listdir("/proc/self/task"):
            mask = os.sched_getaffinity(int(tid))
            if len(mask) == 1:
                pinned |= mask
        word[0] = 1
        for runner in runners:
            runner.join(5)
        assert set(cores) <= pinned, (cores, sorted(pinned))
        assert rc == {a: 0, b: 0}
    finally:
        trait.native_free_engine(a)
        trait.native_free_engine(b)


def test_the_engine_abi_refuses_what_it_cannot_run(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    with pytest.raises(RuntimeError, match="refused engine cores"):
        trait.native_create_engine([cores[0], cores[0]])
    a = trait.native_create_engine(cores[:2])
    layer = trait.register_layer(_random_slabs(1, HIDDEN, INTER), CAP)
    x, slots, weights = _inputs()[0]
    keep_warm = KEEP_WARM(trait.native_keep_warm())
    word = torch.zeros(1, dtype=torch.int32)
    try:
        assert _forward(trait, layer, x, slots, weights, a, 3)[0] == 2, "more workers than the engine's cores"
        assert _forward(trait, layer, x, slots, weights, 999, 2)[0] == 2, "an engine never created"
        assert _forward(trait, layer, x, slots, weights, 0, 2)[0] == 0, "engine 0: unpinned workers"
        assert keep_warm(a, 3, word.data_ptr(), 0, time.monotonic_ns()) == 2
        trait.native_free_engine(a)
        assert _forward(trait, layer, x, slots, weights, a, 2)[0] == 2, "a freed engine"
    finally:
        trait.free_layer(layer)


_LIMIT_SCRIPT = """
import os, sys, threading
sys.path.insert(0, sys.argv[1])
import pytest
from test_cpu_expert_engines_exl3 import HIDDEN, INTER, _forward, _inputs, _kernel
from test_cpu_expert_pool_exl3 import CAP, _random_slabs
trait, cores = _kernel()
a, b = trait.native_create_engine(cores[:2]), trait.native_create_engine(cores[2:4])
layer = trait.register_layer(_random_slabs(5, HIDDEN, INTER), CAP)
rcs = []
def run(engine):
    for x, slots, weights in _inputs() * 10:
        rcs.append(_forward(trait, layer, x, slots, weights, engine, 2)[0])
workers = [threading.Thread(target=run, args=(e,)) for e in (a, b)]
[w.start() for w in workers]
[w.join() for w in workers]
print("rcs", sorted(set(rcs)))
"""


def test_two_full_teams_run_at_once_at_a_thread_limit_of_their_sum():
    """Review Focus 3: OMP_THREAD_LIMIT equal to the two engines' workers (2 + 2) leaves neither team short, which is
    the bound ThreadingConfig enforces. libgomp reads the limit at load, so this runs in a child."""
    env = dict(os.environ, OMP_THREAD_LIMIT="4", EXL3_MOE_CPU_PIN="0")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_LIMIT_SCRIPT), os.path.dirname(__file__)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "rcs [0]" in result.stdout, result.stdout[-2000:]
```

In `test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py`, `FakeForward._run` also records `self.engines.append(c.engine)` (initialize `self.engines = []` in `__init__`), `_host` takes `engine=0` and passes `engine=engine` to `host.enable_cpu_experts`, and add:

```python
def test_the_kernel_engine_reaches_every_cpu_forward(tmp_path):
    """The engine handle enable_cpu_experts takes is the one every forward of the CPU expert thread carries, so the
    kernel runs that engine's team on that engine's cores. Mutant: leave call.engine at 0 -- red."""
    forward = FakeForward()
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n1=1), forward=forward, engine=41)
    try:
        _load(sim, host, [2])
        host.set_cpu_layer(ROW, HANDLE)
        req = _post(sim, [2])
        assert req.kinds == [LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: len(forward.engines) == 1)
        assert sim.copy_wait(req)
        assert forward.engines == [41]
    finally:
        host.stop()
```

In `test/registered/unit/kernels/test_cpu_expert_pool.py`: `FakeServiceTrait.native_set_cores` becomes

```python
    def native_create_engine(self, cores):
        self.events.append(("engine", list(cores)))
        return 0xE1

    def native_free_engine(self, engine):
        self.events.append(("free", engine))
```

`FakeHost.enable_cpu_experts` gains `engine=0` and stores it as the last element of `self.enabled`; `test_service_registers_a_row_once_after_the_cores_and_the_activation_limit` asserts `host.enabled == (0xF00D, [0] * (lease.wire_layout(8).lanes + 1), [4, 5, 6], (2, 16), (2, 2, 8), 2, 0xE1)` and `trait.events == [("engine", [4, 5, 6]), ("register", 10.0), ("register", 10.0)]`.

In `test/manual/dsv41/test_cpu_expert_pool_exl3.py`'s keep-warm test, the prototype becomes `ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64)` and every call passes engine `0` first (`keep_warm(0, 2, word.data_ptr(), 0, far)` and so on); the expectations stay as they are.

- [ ] **Step 2: Run them to verify they fail.** Commit the tests alone:

```bash
git add test/manual/dsv41/test_cpu_expert_engines_exl3.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_pool.py test/manual/dsv41/test_cpu_expert_pool_exl3.py
git diff --cached --stat
git commit -m "test(cpu-experts): per-engine cores, two engines at once (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_pool.py`, `RUN_EXT test/manual/dsv41/test_cpu_expert_engines_exl3.py`.
Expected: FAIL: `TypeError: enable_cpu_experts() got an unexpected keyword argument 'engine'`; `AttributeError: 'Exl3CpuQuantTrait' object has no attribute 'native_create_engine'`.

- [ ] **Step 3: Implement.**

`cpu_expert_forward_abi.h`: `#define SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION 2u`; after `int32_t accumulate;` add

```c
    // The kernel's engine (its *_engine_create): its workers run on that engine's cores. 0: no engine, unpinned
    // workers. Last, so v1's positional initializers still compile.
    int64_t engine;
```

and extend the struct's comment by one sentence: "`engine` selects the kernel's core list; a kernel refuses (2) an engine it never created."

`cpu_experts_cabi.h`: delete `sglang_exl3_cpu_experts_set_cores` and its comment; change `keep_warm`'s declaration to take `int64_t engine` first; add

```c
// An engine: worker i of every forward and keep-warm that names it runs on cores[i], the calling thread as worker 0.
// Cores must be distinct and in [0, CPU_SETSIZE). Engines are immutable and independent, so two engines' forwards may
// run at once from two threads. Returns 0 and the handle (never 0), 1 on a kernel error, 2 on invalid arguments.
int sglang_exl3_cpu_experts_engine_create(const int32_t* cores, int32_t n, int64_t* engine) EXL3_CPU_NOEXCEPT;
// Frees `engine`; a forward already running keeps its cores. Returns 0, or 2 for an unknown engine.
int sglang_exl3_cpu_experts_engine_free(int64_t engine) EXL3_CPU_NOEXCEPT;
```

`moe_mul1.cpp:2103-2135`, replace `g_cores_mutex`, `g_configured_cores`, `g_compute_started`, `g_compute_cores`, `freeze_compute_cores` and `pin_compute_worker` with:

```cpp
// CPU expert engines (sglang_exl3_cpu_experts_engine_create): handle h is g_engines[h - 1], an immutable core list,
// null once freed. Handle 0 is no engine: its workers are not pinned.
std::mutex g_engines_mutex;
std::vector<std::shared_ptr<const std::vector<int>>> g_engines;

// The cores of `engine` (null for engine 0). *found is false for a handle never created, or freed. The shared_ptr
// keeps the list alive through a forward that races a free.
std::shared_ptr<const std::vector<int>> engine_cores(int64_t engine, bool* found)
{
    *found = true;
    if (engine == 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_engines_mutex);
    if (engine < 1 || engine > static_cast<int64_t>(g_engines.size()) || !g_engines[engine - 1]) {
        *found = false;
        return nullptr;
    }
    return g_engines[engine - 1];
}

// Inside a parallel region: pins OpenMP worker `worker` to cores[worker] (no cores: no-op), setting pin_error if it
// cannot. Each engine's master thread has its own libgomp team, so pinned_core is per worker of that team.
inline void pin_compute_worker(int worker, const std::vector<int>* cores, std::atomic<int>& pin_error)
{
    if (cores == nullptr || cores->empty()) return;
    const int core = (*cores)[worker];
    static thread_local int pinned_core = -1;
    if (pinned_core != core || sched_getcpu() != core) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
            pin_error.store(1, std::memory_order_relaxed);
        else pinned_core = core;
    }
}
```

`ForwardCtx` (`:2167`): add `const std::vector<int>* cores = nullptr;  // the engine's worker cores; null: unpinned`.

`forward_raw` (`:2543`): add a last parameter `const std::vector<int>* cores` and set `ctx.cores = cores;` after `ctx.m_total = m_total;`. `exl3_moe_cpu_forward_raw` (`:2615`) passes `nullptr` (the torch op path stays unpinned, as it was without `set_cores`).

`forward_plan.hpp` `run_team`, replace its first three lines (`freeze_compute_cores(); TORCH_CHECK(g_compute_cores.empty() || ...)`) with

```cpp
        TORCH_CHECK(ctx.cores == nullptr || size_t(count) <= ctx.cores->size(),
                    "CPU expert worker count exceeds the engine's cores");
```

and the call inside the region with `pin_compute_worker(worker, ctx.cores, pin_error);`.

`sglang_exl3_cpu_experts_forward` (`:2673`): after the argument checks,

```cpp
    bool found = false;
    const auto cores = engine_cores(c.engine, &found);
    if (!found || (cores && static_cast<size_t>(c.threads) > cores->size())) return 2;
```

and pass `cores.get()` as `forward_raw`'s last argument.

`sglang_exl3_cpu_experts_keep_warm` (`:2748`): signature `(int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns)`; replace `freeze_compute_cores(); if (!g_compute_cores.empty() && ...) return 2;` with the same three-line `engine_cores` lookup on `threads`, and the region's pin call with `pin_compute_worker(omp_get_thread_num(), cores.get(), pin_error);`.

Replace `sglang_exl3_cpu_experts_set_cores` (`:2775-2790`) with:

```cpp
extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_engine_create(
    const int32_t* cores, int32_t n, int64_t* engine) noexcept
{
    if (cores == nullptr || engine == nullptr || n < 1 || n > CPU_SETSIZE) return 2;
    for (int i = 0; i < n; ++i) {
        if (cores[i] < 0 || cores[i] >= CPU_SETSIZE) return 2;
        for (int j = 0; j < i; ++j) if (cores[i] == cores[j]) return 2;
    }
    try {
        auto list = std::make_shared<const std::vector<int>>(cores, cores + n);
        std::lock_guard<std::mutex> lock(g_engines_mutex);
        g_engines.push_back(std::move(list));
        *engine = static_cast<int64_t>(g_engines.size());
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_engine_free(int64_t engine) noexcept
{
    std::lock_guard<std::mutex> lock(g_engines_mutex);
    if (engine < 1 || engine > static_cast<int64_t>(g_engines.size()) || !g_engines[engine - 1]) return 2;
    g_engines[engine - 1].reset();
    return 0;
}
```

`cpu_experts.h`: `using CpuExpertKeepWarm = int (*)(int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns);`; in `CpuExpertConfig` after `forward`: `int64_t engine = 0;  // the kernel's engine (its core list), passed on every forward and keep-warm; 0: none`; in `run()`, `call.engine = config_.engine;` after `call.accumulate = ...`, and the keep-warm call becomes `config_.keep_warm(config_.engine, config_.threads, reinterpret_cast<const uint32_t*>(&kick_), kick, warm_until)`.

`ffi_exports.h` `enable_cpu_experts`: add `int64_t engine` after `int64_t forward`, set `config.engine = engine;`, and add "`engine` is the kernel's engine handle (0: none)." to its comment. `ffi_test_exports.h` `test_keep_warm` takes `(int64_t, int32_t, const uint32_t* word, uint32_t seen, int64_t deadline_ns)`.

`expert_stream_transport.py` `ExpertStreamHost.enable_cpu_experts`: add the keyword `engine: int = 0` (docstring: "``engine`` is the kernel's engine handle (``native_create_engine``), carried by every forward and keep-warm; 0 runs unpinned workers") and pass `int(engine)` after `int(forward)`.

`pool.py`: `CPU_EXPERTS_FORWARD_ABI_VERSION = 2`; append `("engine", ctypes.c_int64)` to `CpuExpertsForwardCall._fields_`; in the Protocol replace `native_set_cores` with

```python
    def native_create_engine(self, cores: Sequence[int]) -> int:
        """Create the kernel's engine on ``cores``: worker i of each call naming it runs on cores[i], the calling
        thread as worker 0. Returns its handle, never 0."""
        ...

    def native_free_engine(self, engine: int) -> None:
        """Free an engine ``native_create_engine`` returned."""
        ...
```

and update the class docstring's "``native_forward`` and ``native_set_cores``" to "``native_forward`` and ``native_create_engine``".

`exl3.py`: replace `native_set_cores` with

```python
    def native_create_engine(self, cores) -> int:
        """Create the kernel's engine on ``cores`` (the calling engine thread is worker 0)."""
        import ctypes

        fn = self._native("sglang_exl3_cpu_experts_engine_create")
        fn.argtypes, fn.restype = (
            [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)],
            ctypes.c_int,
        )
        array = (ctypes.c_int32 * len(cores))(*cores)
        engine = ctypes.c_int64(0)
        result = fn(array, len(cores), ctypes.byref(engine))
        if result != 0:
            raise RuntimeError(f"the EXL3 CPU kernel refused engine cores {list(cores)} ({result})")
        return engine.value

    def native_free_engine(self, engine) -> None:
        """Free an engine native_create_engine returned."""
        import ctypes

        fn = self._native("sglang_exl3_cpu_experts_engine_free")
        fn.argtypes, fn.restype = [ctypes.c_int64], ctypes.c_int
        if fn(engine) != 0:
            raise RuntimeError(f"the EXL3 CPU kernel has no engine {engine}")
```

`service.py`: in `__init__`, before `host.enable_cpu_experts(...)`, `self.engine = trait.native_create_engine(self.cores)`, and pass `engine=self.engine`; delete `self._cores_set` and, in `register`, the `if not self._cores_set:` block.

Bench: `cpu_forward.cpp` creates one engine in `main` (replacing the `set_cores` call at `:294`: `if (sglang_exl3_cpu_experts_engine_create(cores.data(), cores.size(), &g_engine)) throw ...;` with a file-scope `int64_t g_engine = 0;`) and sets `call.engine = g_engine;` at `:203`. `full_stack.cpp` does the same at `:507` (`int64_t engine`), stores it in `StackConfig::engine` (new member in `stack.h` after `keep_warm_ns`, copied to `cpu.engine` at `:164`), and sets `call.engine` in the bare forward at `:294`.

- [ ] **Step 4: Run.** Commit:

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_expert_forward_abi.h python/sglang/srt/layers/quantization/exl3_cpu/optimized/cpu_experts_cabi.h python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp python/sglang/srt/layers/quantization/exl3_cpu/optimized/forward_plan.hpp python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/srt/layers/moe/cpu_experts/pool.py python/sglang/srt/layers/moe/cpu_experts/exl3.py python/sglang/srt/layers/moe/cpu_experts/service.py python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/cpu_forward.cpp python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/full_stack.cpp python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h
git add -p python/sglang/kernels/ops/moe/expert_stream_transport.py
git diff --cached --stat
git commit -m "feat(cpu-experts): an engine handle carries each CPU expert engine's cores; set_cores removed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py`, `RUN_EXT test/manual/dsv41/test_cpu_expert_engines_exl3.py test/manual/dsv41/test_cpu_expert_pool_exl3.py`. Expected: PASS.

- [ ] **Step 5: Bit-exact gate and existing suites.** `CPU_CHECKS` `check` mode with `<task>` = `t2` (no server running). Expected: `ALL GREEN (check)`: every ISA tier's dumps equal the merge-base's, through both registrations, and the bare and full-stack benches' frozen outputs hold. Then `RUN_CPU SUITE_CPU`, `RUN_EXT SUITE_EXT`. Expected: Baseline counts plus this task's tests. NVFP4 is untouched: `test_nvfp4_cpu_experts.py` and `test_nvfp4_cpu_build.py` still pass with the v2 header (they ignore `engine`).

---

### Task 3: The node count through the builds, and the home rule

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h:16-19` (the define), `:27-33` (`home`), `:103` (`Wire`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout_probe.cpp:24` (print `home7`)
- Modify: `python/sglang/kernels/ops/moe/expert_lease_block.py:35-150` (`WireLayout.home`, `cpp_constants`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py:135-190` (`_host_module`, `_host_module_cached`, `_host_module_tsan`), `:2001-2058` (`_LEASE_METHODS`, `_device_module`, `_device_module_cached`, `device_module_with_hooks`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`wire_lanes`, `wire_nodes` and their two export lines)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh:295` (`LeaseProtocolKernel::wire_nodes`)
- Test: `test/registered/unit/kernels/test_expert_stream_lease_layout.py`, `test/registered/unit/kernels/test_expert_stream_build_variants.py`, `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`

**Interfaces:**
- Produces (C++): `SGLANG_EXPERT_STREAM_NODES` (default 1); `Wire = LeaseLayout<SGLANG_EXPERT_STREAM_LANES, SGLANG_EXPERT_STREAM_NODES>`; `static constexpr int LeaseLayout::home(int64_t expert)` = `expert % kNodes`.
- Produces (Python): `WireLayout.home(expert: int) -> int`; `_host_module(layout="exl3", variant=None, lanes=8, nodes=1)`, `_device_module(layout="exl3", lanes=8, nodes=1)`, `device_module_with_hooks(defines, layout="exl3", lanes=8, nodes=1)`; module names gain `_n{nodes}` only for `nodes > 1`; host export `expert_stream_wire_nodes() -> int`, `expert_stream_wire_lanes() -> int`; device export `expert_stream_wire_nodes() -> int`.

- [ ] **Step 1: Write the failing tests.**

In `test_expert_stream_lease_layout.py`:

```python
@pytest.mark.parametrize("nodes", [1, 2, 3])
def test_an_expert_is_homed_on_expert_mod_nodes(nodes):
    w = lease.wire_layout(8, nodes)
    assert [w.home(e) for e in range(7)] == [e % nodes for e in range(7)]
```

(`test_the_python_layout_is_the_cpp_trait` also covers `home7` once Step 3 adds it to both sides.)

In `test_expert_stream_build_variants.py`:

```python
@pytest.mark.parametrize("nodes", [1, 2])
def test_each_host_build_is_compiled_for_its_node_count(nodes):
    module = ops._host_module("exl3", "instr", 8, nodes)
    assert (int(module.expert_stream_wire_lanes()), int(module.expert_stream_wire_nodes())) == (8, nodes)


def test_one_node_and_two_nodes_are_separate_modules():
    assert ops._host_module("exl3", "instr", 8, 1) is ops._host_module("exl3", "instr", 8)
    assert ops._host_module("exl3", "instr", 8, 2) is not ops._host_module("exl3", "instr", 8, 1)
```

In `test_exl3_lease_kernels_cuda.py`:

```python
@pytest.mark.parametrize("lanes, nodes", [(8, 1), (8, 2), (16, 2)])
def test_each_device_build_is_compiled_for_its_node_count(lanes, nodes):
    assert int(ops._device_module("exl3", lanes, nodes).expert_stream_wire_nodes()) == nodes
```

- [ ] **Step 2: Run them to verify they fail.** Commit the tests alone (`git add` the three test files; `git diff --cached --stat`; message `test(expert-stream): builds per node count and the home rule (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/kernels/test_expert_stream_lease_layout.py test/registered/unit/kernels/test_expert_stream_build_variants.py`, `RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py -k node_count`.
Expected: FAIL: `AttributeError: 'WireLayout' object has no attribute 'home'`; `TypeError: _host_module() takes from 0 to 3 positional arguments but 4 were given`.

- [ ] **Step 3: Implement.**

`lease_layout.h`, after the lanes define:

```cpp
// The NUMA node count of this build: every JIT build of the device kernels and the host passes
// -DSGLANG_EXPERT_STREAM_NODES alongside the lanes.
#ifndef SGLANG_EXPERT_STREAM_NODES
#define SGLANG_EXPERT_STREAM_NODES 1
#endif
```

inside `LeaseLayout`, after `kNodes`:

```cpp
  // The node whose group serves `expert`: its staging list, its CPU split and its slots. The one home rule, so a
  // popularity table can replace it here without touching a caller.
  static constexpr int home(int64_t expert) {
    return static_cast<int>(expert % kNodes);
  }
```

`using Wire = LeaseLayout<SGLANG_EXPERT_STREAM_LANES, SGLANG_EXPERT_STREAM_NODES>;` and one more assertion beside the v2 ones: `static_assert(LeaseLayout<8, 2>::home(7) == 1 && V2::home(7) == 0, "home is expert % nodes");`.

`lease_layout_probe.cpp`: after the `P(...)` lines, `put("home7", L::home(7));`.

`expert_lease_block.py`, in `WireLayout`:

```python
    def home(self, expert: int) -> int:
        """The node whose group serves ``expert``: LeaseLayout::home, the one home rule."""
        return expert % self.nodes
```

and add `"home7": self.home(7)` to `cpp_constants`.

`expert_stream_transport.py`: `_host_module(layout="exl3", variant=None, lanes=8, nodes=1)` validates with `expert_lease_block.wire_layout(lanes, nodes)` and calls `_host_module_tsan(layout, lanes, nodes)` / `_host_module_cached(layout, variant, lanes, nodes)` positionally. Both cached loaders take `nodes` last, build the name with `_suffix(lanes, nodes)` and add the define:

```python
def _suffix(lanes: int, nodes: int) -> str:
    """A build's module-name suffix: one node keeps the Phase 1 names, so its modules and caches are unchanged."""
    return f"_l{lanes}" if nodes == 1 else f"_l{lanes}_n{nodes}"
```

`f"expert_stream_host_{layout}_{variant}{_suffix(lanes, nodes)}"`, `f"expert_stream_host_{layout}_instr_tsan{_suffix(lanes, nodes)}"`, and in every `extra_cflags` list `f"-DSGLANG_EXPERT_STREAM_NODES={nodes}"` after the lanes define. The device side the same: `_device_module(layout="exl3", lanes=8, nodes=1)` -> `_device_module_cached(layout, wire.lanes, wire.nodes)`, name `f"expert_stream_{layout}{_suffix(lanes, nodes)}"`, `extra_cuda_cflags` gains the nodes define, and `device_module_with_hooks(defines, layout="exl3", lanes=8, nodes=1)` likewise. `_LEASE_METHODS` gains `"expert_stream_wire_nodes": "wire_nodes"`.

`ffi_exports.h`, in `HostExports`:

```cpp
  // The wire this module was compiled for (-DSGLANG_EXPERT_STREAM_LANES / _NODES), for the Python side's checks.
  static int64_t wire_lanes() {
    return Wire::kLanes;
  }
  static int64_t wire_nodes() {
    return Wire::kNodes;
  }
```

with `TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_wire_lanes, Exports::wire_lanes);` and `..._wire_nodes` in `EXPERT_STREAM_HOST_EXPORTS`. `lease_kernels.cuh`, in `LeaseProtocolKernel`:

```cpp
  /// The node count this module was compiled for (-DSGLANG_EXPERT_STREAM_NODES).
  static int64_t wire_nodes() {
    return expert_stream::wire::Wire::kNodes;
  }
```

- [ ] **Step 4: Run.** Commit (`git add` the four C++/probe files and `expert_lease_block.py`; `git add -p python/sglang/kernels/ops/moe/expert_stream_transport.py`; `git diff --cached --stat`; message `feat(expert-stream): SGLANG_EXPERT_STREAM_NODES in every build; the home rule` with the trailers). `SYNC`, the Step 2 runs. Expected: PASS, including `test_the_python_layout_is_the_cpp_trait` for every `(lanes, nodes)` in its grid.

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU`, `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus this task's tests. Check one-node names are unchanged (`load_jit` caches under `$SGLANG_JIT_CACHE_DIR/<target>/<module_name>/`, default `~/.cache/sglang/jit`): `ssh divix01 'ls -d ~/.cache/sglang/jit/*/expert_stream_host_exl3_*_l8* ~/.cache/sglang/jit/*/expert_stream_exl3_l8*'`. Expected: the Phase 1 directory names, plus `_l8_n2` ones from this task's tests, and no `_n1` directory. (The build key inside changes: the define list grew.)

---

### Task 4: Device per-node staging, per-node split, per-group CPU parts

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh:160-163` (`TypedLanes`), `:248-262` (`RowMap`, `MapDelta`), `:278-332` (`load_map_delta`, `apply_map_delta`), `:342-367` (`LanePolicy`, `load_split`), `:369-431` (`type_lanes`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh:41-60` (`PostParams.staging` comment, `lane_node`), `:125`, `:143`, `:193-197`, `:272`, `:326-334` and `:371` (post launcher: `lane_node` after `lane_slot`, staging matcher), `:489` (bulk-apply matcher)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh:240-260` (`CopyWaitParams.lane_node`, `CopyCommitParams` comment), `:282-298` (part bits), and the `lease_copy_wait` launcher (`lane_node` after `lane_slot`)
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh:41-46,82-91` (sum every flagged part)
- Modify: `python/sglang/kernels/ops/moe/exl3_route_tables.py:58-62` (docstring: a bit per part)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__` ~:2141-2260: `nodes`, `self.lane_node`, the 2-D staging bank; `post` and `copy_wait` pass `self.lane_node`)
- Modify: `python/sglang/srt/layers/moe/ram_slot_map.py:20-100` (`type_lanes`, `MapReplica`)
- Modify (API change): `test/manual/dsv41/test_exl3_lease_kernels_cuda.py:285-292` (`test_the_post_record_round_trips_at_every_lane_width` passes the new `lane_node` output)
- Test: `test/registered/unit/kernels/test_ram_slot_map.py`, `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`, `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`

**Interfaces:**
- Consumes: `Wire::home`, `Wire::kNodes`, `Wire::kSplitStride`, `Wire::kDeltaStaging` (Task 3); `WireLayout.home`.
- Produces (device): `TypedLanes::node[Wire::kLanes]`; `RowMap::staging` and `MapDelta::staging` are `[Wire::kNodes * Wire::kLanes]` node-major; `LanePolicy::split[Wire::kNodes][Wire::kLanes + 1]`; `load_split(const uint8_t* split, int32_t (&out)[Wire::kNodes][Wire::kLanes + 1])`; `PostParams::lane_node`, `CopyWaitParams::lane_node` (`int32_t*` `[lanes]`); CPU output part of a lane = `2 * node + (kind == kKindMissCpu)`, flagged in `cpu_lanes[1]` bit by bit.
- Produces (Python): `ram_slot_map.type_lanes(experts, ram_slot, staging, split, *, lanes, captured, copy_armed, hit_copy, cpu_on, cpu_misses, ce_ok=True, cpu_ok=True, dst_ok=None, nodes=1)` with `staging` `[nodes * lanes]` and `split` `[nodes * (lanes + 1)]` node-major; `MapReplica(rows, experts, lanes, nodes=1)`; `ExpertStreamDevice(..., lanes=8, nodes=1)` with `self.lane_node` and `map_bank["staging"]` `[layers, nodes * lanes]`; the post FFI takes `lane_node` after `lane_slot`, the copy-wait FFI after `lane_slot`.

- [ ] **Step 1: Write the failing tests.**

`test/registered/unit/kernels/test_ram_slot_map.py`, add:

```python
from sglang.kernels.ops.moe.expert_lease_block import wire_layout


def _staging(nodes, lanes, counts):
    """Node-major lists: node n's k-th staging slot is 100 * (n + 1) + k, its first counts[n] of them valid."""
    return [100 * (n + 1) + k if k < counts[n] else -1 for n in range(nodes) for k in range(lanes)]


def test_each_miss_takes_the_next_staging_slot_of_its_home_node():
    ram = [-1] * 16
    ram[2], ram[5] = 7, 9
    kinds, slots = type_lanes(
        [1, 2, 3, 4, 5, 6], ram, _staging(2, 8, [8, 8]), [0] * 18, lanes=8, captured=False, copy_armed=False,
        hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=2,
    )
    # experts 1 and 3 are node 1's misses, 4 and 6 node 0's; 2 and 5 are hits
    assert slots == [200, 7, 201, 100, 9, 101]
    assert kinds == [LaneKind.MISS_GPU, LaneKind.HIT_SM, LaneKind.MISS_GPU, LaneKind.MISS_GPU, LaneKind.HIT_SM,
                     LaneKind.MISS_GPU]


def test_misses_homed_on_one_node_draw_only_on_that_nodes_list():
    """Review Focus 1, the reference: all misses on node 1 use node 1's list and leave node 0's untouched; one miss
    past node 1's list raises even though node 0 has slots."""
    staging = _staging(2, 8, [8, 2])
    _, slots = type_lanes([1, 3], [-1] * 16, staging, [0] * 18, lanes=8, captured=False, copy_armed=False,
                          hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=2)
    assert slots == [200, 201]
    with pytest.raises(ValueError, match="no staging slot on node 1"):
        type_lanes([1, 3, 5], [-1] * 16, staging, [0] * 18, lanes=8, captured=False, copy_armed=False,
                   hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=2)


def test_the_cpu_split_is_per_node_and_a_zero_split_leaves_a_node_on_the_gpu():
    lanes = 8
    ram = list(range(16))  # every expert a hit
    split = [0] * (2 * (lanes + 1))
    split[0 * (lanes + 1) + 3] = 2  # node 0: 2 of its 3 eligible lanes on the CPU
    kinds, _ = type_lanes(
        [0, 1, 2, 3, 4, 5], ram, _staging(2, lanes, [8, 8]), split, lanes=lanes, captured=True, copy_armed=True,
        hit_copy="sm", cpu_on=True, cpu_misses=False, nodes=2,
    )
    # node 0's lanes are 0, 2, 4: the last two go to the CPU; node 1's split is all zero
    assert kinds == [LaneKind.HIT_SM, LaneKind.HIT_SM, LaneKind.HIT_CPU, LaneKind.HIT_SM, LaneKind.HIT_CPU,
                     LaneKind.HIT_SM]


def test_one_node_is_the_flat_lists_of_today():
    replica = MapReplica(2, 16, 8)
    assert len(replica.staging[0]) == 8
    assert len(MapReplica(2, 16, 8, nodes=2).staging[0]) == 16
    assert wire_layout(8, 2).home(5) == 1
```

(`LaneKind`, `MapReplica`, `type_lanes` and `pytest` are already imported there; add what is missing.)

`test/manual/dsv41/test_exl3_lease_kernels_cuda.py`, add a raw-post helper next to `test_the_post_record_round_trips_at_every_lane_width` and three tests:

```python
from sglang.srt.layers.moe.ram_slot_map import type_lanes  # noqa: E402


def _raw_post(lanes, nodes, planned, ram, staging, split=None, captured=False, cpu_on=False):
    """One post of the node-aware device build over one row: ``ram`` its slot map, ``staging`` node-major, ``split``
    node-major (written to the block and armed when given). Returns the lanes' kinds, slots and nodes."""
    w = lease.wire_layout(lanes, nodes)
    count, experts = len(planned), len(ram)
    page = ops.new_page(pin=True, wire=w)
    block = lease.new_lease_block(1, pin=True, wire=w)
    delta = w.lease_block_bytes
    block[delta : delta + 8].view(torch.int64)[0] = 1
    block[delta + w.delta_fields["staging"] :][: 2 * nodes * w.lanes].view(torch.int16)[:] = torch.tensor(
        staging, dtype=torch.int16
    )
    if split is not None:
        block[w.copy_armed : w.copy_armed + 4].view(torch.int32)[0] = 1
        for node in range(nodes):
            at = w.split + node * w.split_stride
            block[at : at + 4 * (w.lanes + 1)].view(torch.int32)[:] = torch.tensor(
                split[node * (w.lanes + 1) : (node + 1) * (w.lanes + 1)], dtype=torch.int32
            )
    cuda = dict(device="cuda")
    planned_t = torch.tensor(planned, dtype=torch.int64, **cuda)
    out = {n: torch.zeros(w.lanes, dtype=torch.int32, **cuda) for n in ("kind", "slot", "node", "dst_1")}
    zeros_u8 = torch.zeros(1, dtype=torch.uint8, **cuda)
    no_i64, no_i32 = torch.empty(0, dtype=torch.int64, **cuda), torch.empty(0, dtype=torch.int32, **cuda)
    ops._device_module("exl3", lanes, nodes).expert_stream_post(
        page, torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32, **cuda), planned_t,
        torch.tensor([count], dtype=torch.int32, **cuda), planned_t.clone(), 0, experts, int(block.data_ptr()),
        5_000_000_000, 0, 0, no_i64, 0, torch.arange(count, dtype=torch.int32, **cuda), int(captured),
        torch.tensor([ram], dtype=torch.int32, **cuda), torch.full((1, nodes * w.lanes), -1, dtype=torch.int32, **cuda),
        torch.ones(1, dtype=torch.int64, **cuda), torch.zeros(1, dtype=torch.int64, **cuda),
        zeros_u8, torch.ones(1, dtype=torch.uint8, **cuda), torch.full((1,), w.lanes, dtype=torch.int32, **cuda), 64,
        0, int(cpu_on), 0, out["kind"], out["slot"], out["node"], torch.zeros(1, dtype=torch.int32, **cuda),
        torch.zeros(w.lanes, dtype=torch.int64, **cuda), out["dst_1"], no_i32, 0, no_i32, 0,
    )
    torch.cuda.synchronize()
    return [out[n][:count].tolist() for n in ("kind", "slot", "node")]


@pytest.mark.parametrize("lanes", [8, 16, 32])
@pytest.mark.parametrize("nodes", [1, 2])
def test_each_miss_takes_the_next_staging_slot_of_its_home_node(lanes, nodes):
    """The device's typing equals ram_slot_map.type_lanes's on random plans of hits and misses on every node."""
    w = lease.wire_layout(lanes, nodes)
    rng = random.Random(lanes * 10 + nodes)
    experts = 4 * lanes
    for _ in range(20):
        count = rng.randint(1, lanes)
        planned = rng.sample(range(experts), count)
        ram = [rng.randrange(64) if rng.random() < 0.4 else -1 for _ in range(experts)]
        staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
        kinds, slots, homes = _raw_post(lanes, nodes, planned, ram, staging)
        want_kinds, want_slots = type_lanes(
            planned, ram, staging, [0] * (nodes * (w.lanes + 1)), lanes=w.lanes, captured=False, copy_armed=False,
            hit_copy="ce", cpu_on=False, cpu_misses=False, nodes=nodes,
        )
        assert (kinds, slots) == ([int(k) for k in want_kinds], want_slots)
        assert homes == [w.home(e) for e in planned]


def test_the_split_is_per_node_and_a_zero_split_keeps_a_node_off_the_cpu():
    lanes, nodes = 8, 2
    w = lease.wire_layout(lanes, nodes)
    split = [0] * (nodes * (w.lanes + 1))
    split[3] = 2  # node 0: 2 of 3 eligible lanes; node 1's table is all zero
    planned, ram = [0, 1, 2, 3, 4, 5], list(range(16))
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    kinds, _, _ = _raw_post(lanes, nodes, planned, ram, staging, split=split, captured=True, cpu_on=True)
    want, _ = type_lanes(planned, ram, staging, split, lanes=w.lanes, captured=True, copy_armed=True, hit_copy="sm",
                         cpu_on=True, cpu_misses=False, nodes=nodes)
    assert kinds == [int(k) for k in want] == [int(LaneKind.HIT_SM), int(LaneKind.HIT_SM), int(LaneKind.HIT_CPU),
                                               int(LaneKind.HIT_SM), int(LaneKind.HIT_CPU), int(LaneKind.HIT_SM)]


_NODE_TRAP_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from test_exl3_lease_kernels_cuda import _raw_post
staging = [100 + k for k in range(8)] + [200, 201] + [-1] * 6
print("slots", _raw_post(8, 2, [1, 3, 0, 2, 4, 6, 8], [-1] * 16, staging)[1], flush=True)
try:
    _raw_post(8, 2, [1, 3, 5], [-1] * 16, staging)
    print("reached", flush=True)
except Exception as error:
    print("trapped", type(error).__name__, flush=True)
"""


def test_a_node_out_of_staging_traps_while_the_other_node_has_slots():
    """Review Focus 1: node 1 has 2 staging slots and node 0 eight. Seven misses, two on node 1, take node 1's two
    and node 0's first five; three misses on node 1 trap, although node 0 has slots to spare."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_NODE_TRAP_SCRIPT), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=120,
    )
    assert "slots [200, 201, 100, 101, 102, 103, 104]" in result.stdout, (result.stdout, result.stderr[-2000:])
    assert "reached" not in result.stdout and "trapped" in result.stdout, (result.stdout, result.stderr[-2000:])
```

In `test_the_post_record_round_trips_at_every_lane_width`, pass a `torch.zeros(w.lanes, dtype=torch.int32, device="cuda")` for `lane_node` right after `out["slot"]`.

`test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`, add:

```python
@pytest.mark.parametrize("parts", [0b01, 0b10, 0b11, 0b0101, 0b1111, 0b1010])
def test_route_tables_seed_every_flagged_cpu_part_in_part_order(parts):
    """Two groups' CPU partial sums are four parts: 2g the hits', 2g + 1 the misses'. The seed adds every flagged part,
    lowest first, so one node's (parts 0 and 1) is bit for bit today's."""
    from sglang.kernels.ops.moe.exl3_route_tables import exl3_moe_route_tables

    routes, slots, hidden = 4, 12, 256
    dev = "cuda"
    gen = torch.Generator().manual_seed(parts)
    host = (torch.randn(4, hidden, generator=gen) * 1e3).pin_memory()
    remap = torch.arange(routes, dtype=torch.int32, device=dev)
    weights = torch.ones(routes, device=dev)
    keep = torch.ones(1, device=dev)
    x = torch.randn(1, hidden, device=dev)
    out = torch.full((1, hidden), 5.0, device=dev)
    args = dict(
        remap64_out=torch.empty(routes, dtype=torch.int64, device=dev),
        x16_out=torch.empty(1, hidden, dtype=torch.float16, device=dev), out_zero=out,
        expert_count=torch.empty(slots + 1, dtype=torch.int64, device=dev),
        inv_order=torch.empty(routes, dtype=torch.int64, device=dev),
        weight_sorted=torch.empty(routes, dtype=torch.float16, device=dev),
        det=torch.empty(3, slots + 1, dtype=torch.int64, device=dev),
    )
    exl3_moe_route_tables(
        remap, weights, keep, x, **args, cpu_lanes=torch.tensor([0, parts], dtype=torch.int32, device=dev),
        dst_slots=torch.arange(routes, dtype=torch.int32, device=dev), cpu_out=host.data_ptr(), cpu_part_stride=hidden,
    )
    torch.cuda.synchronize()
    want = torch.zeros(hidden)
    for p in range(4):
        if parts >> p & 1:
            want = want + host[p]
    assert torch.equal(out.cpu()[0], want)
```

- [ ] **Step 2: Run them to verify they fail.** Commit the tests alone (`git add test/registered/unit/kernels/test_ram_slot_map.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`; `git diff --cached --stat`; message `test(expert-stream): per-node staging and split on the device and the reference (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/kernels/test_ram_slot_map.py`, `RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_dsv41_layer_fusion_gpu.py -k "node or part"`.
Expected: FAIL: `TypeError: type_lanes() got an unexpected keyword argument 'nodes'`; the post FFI refuses the extra `lane_node` argument; the route tables read only parts 0 and 1 (`0b0101` and `0b1111` mismatch).

- [ ] **Step 3: Implement.**

`ram_slot_map.py`, `type_lanes` gains `nodes: int = 1` (docstring: "``staging`` is node-major, ``nodes * lanes`` slots, and ``split`` node-major, ``nodes * (lanes + 1)`` counts. A miss takes the next slot of its home node's list; node n's CPU takes the last ``split[n][k]`` of its k eligible lanes in plan order."). Its body from the staging loop on becomes:

```python
    home = wire_layout(lanes, nodes).home
    slots, hit, taken = [], [], [0] * nodes
    for e in experts:
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
            continue
        node = home(e)
        m = taken[node]
        if m >= lanes or staging[node * lanes + m] < 0:
            raise ValueError(f"a miss lane has no staging slot on node {node}")
        slots.append(staging[node * lanes + m])
        hit.append(False)
        taken[node] += 1
    host_lanes = captured and copy_armed
    eligible = [host_lanes and cpu_on and cpu_ok and (h or cpu_misses) for h in hit]
    take = [0] * nodes
    for node in range(nodes):
        n = sum(1 for e, ok in zip(experts, eligible) if ok and home(e) == node)
        take[node] = split[node * (lanes + 1) + n] if n else 0
    cpu = [False] * len(experts)
    for j in reversed(range(len(experts))):
        node = home(experts[j])
        if take[node] > 0 and eligible[j]:
            cpu[j] = True
            take[node] -= 1
```

(the kind loop after it is unchanged; `wire_layout` is imported from `sglang.kernels.ops.moe.expert_lease_block`). `MapReplica.__init__(self, rows, experts, lanes, nodes=1)` sizes `self.staging` rows `[-1] * (nodes * lanes)`; its docstring says `staging [rows][nodes * lanes]`.

`lease_device.cuh`: `TypedLanes` gains `int32_t node[Wire::kLanes];  // the lane's home node (Wire::home of its expert)`. `RowMap::staging` comment becomes `// [Wire::kNodes * Wire::kLanes], node-major: the row's staging slots per node`; `MapDelta::staging[Wire::kNodes * Wire::kLanes]`. In `load_map_delta`, `constexpr int kStagingLoads = Wire::kNodes * Wire::kLanes / 8;` (drop the "node 0's list" comment). In `apply_map_delta`, the staging copy runs `k < Wire::kNodes * Wire::kLanes`. `LanePolicy::split` becomes `int32_t split[Wire::kNodes][Wire::kLanes + 1];  // Wire::kSplit: per node, CPU lanes per n eligible lanes`, and:

```cpp
// Loads every node's Wire::kSplit table in whole 16-byte loads; words past a table are dropped.
SGL_DEVICE void load_split(const uint8_t* split, int32_t (&out)[Wire::kNodes][Wire::kLanes + 1]) {
  constexpr int kLoads = static_cast<int>(Wire::kSplitStride / 16);
  static_assert(
      Wire::kSplit % 16 == 0 && Wire::kSplit + Wire::kNodes * Wire::kSplitStride <= Wire::kLeaseBlockBytes,
      "split loads");
#pragma unroll
  for (int node = 0; node < Wire::kNodes; ++node) {
    uint4 v[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i)
      v[i] = ld_relaxed_sys_v4(split + node * Wire::kSplitStride + 16 * i);
#pragma unroll
    for (int n = 0; n <= Wire::kLanes; ++n) {
      const uint4 q = v[n / 4];
      const uint32_t words[4] = {q.x, q.y, q.z, q.w};
      out[node][n] = static_cast<int32_t>(words[n % 4]);
    }
  }
}
```

`type_lanes` (`:375-431`) becomes (its doc comment: "A hit takes its RAM slot, and a miss the next slot of its home node's staging list. Node n's CPU takes the last split[n][k] of its k eligible lanes in plan order. Traps where the reference raises: ..., a miss with no staging slot on its node, a split entry above its n."):

```cpp
SGL_DEVICE void type_lanes(const LanePlan& plan, const RowMap& map, const LanePolicy& policy, TypedLanes& out) {
  if (plan.count > Wire::kLanes) __trap();
  int64_t expert[Wire::kLanes];
  int32_t dst[Wire::kLanes];
  int32_t ram[Wire::kLanes];
  int32_t staging[Wire::kNodes * Wire::kLanes];
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j) {
    expert[j] = j < plan.count ? plan.planned[j] : 0;
    dst[j] = j < plan.count ? plan.dst[j] : -1;
  }
#pragma unroll
  for (int k = 0; k < Wire::kNodes * Wire::kLanes; ++k)
    staging[k] = map.staging[k];
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j)
    if (j < plan.count && (expert[j] < 0 || expert[j] >= map.experts)) __trap();
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j)
    ram[j] = j < plan.count ? map.ram_slot[expert[j]] : -1;
  bool hit[Wire::kLanes];
  bool eligible[Wire::kLanes];
  int m[Wire::kNodes] = {};
  int n[Wire::kNodes] = {};
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j) {
    if (j >= plan.count) break;
    for (int i = 0; i < j; ++i)
      if (expert[i] == expert[j]) __trap();
    const int node = Wire::home(expert[j]);
    out.node[j] = node;
    hit[j] = ram[j] >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(ram[j]) >= map.row_capacity) __trap();
      out.slot[j] = ram[j];
    } else {
      if (m[node] >= Wire::kLanes || staging[node * Wire::kLanes + m[node]] < 0) __trap();
      out.slot[j] = staging[node * Wire::kLanes + m[node]++];
    }
    eligible[j] = policy.host_lanes && policy.cpu_on && policy.cpu_ok && (hit[j] || policy.cpu_misses);
    n[node] += eligible[j] ? 1 : 0;
  }
  int take[Wire::kNodes];
#pragma unroll
  for (int node = 0; node < Wire::kNodes; ++node) {
    take[node] = n[node] > 0 ? policy.split[node][n[node]] : 0;
    if (take[node] < 0 || take[node] > n[node]) __trap();
  }
  const bool copy_ok = policy.host_lanes && policy.hit_copy_ce && policy.ce_ok;
  for (int64_t j = plan.count - 1; j >= 0; --j) {
    const int node = out.node[j];
    const bool cpu = take[node] > 0 && eligible[j];
    if (cpu) --take[node];
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
}
```

`lease_kernels.cuh`: `PostParams` gains `int32_t* lane_node;` after `lane_slot` (comment block above it: "each lane's kind, source slot and home node"); the staging comment says `[rows, Wire::kNodes * Wire::kLanes]`; `:125` and `:272` index `p.staging + p.row * (Wire::kNodes * Wire::kLanes)` (and `row * ...`); the post writes `p.lane_node[j] = used ? typed.node[j] : 0;` beside `p.lane_slot[j]`; the launcher takes `tvm::ffi::TensorView lane_node` after `lane_slot`, verifies it like `lane_slot`, and both staging matchers (`:371`, `:489`) become `TensorMatcher({Rows_, Wire::kNodes * Wire::kLanes})`.

`row_copy_kernels.cuh`: `CopyWaitParams` gains `const int32_t* lane_node;  // each lane's home node: its CPU part pair` after `lane_slot`; `CopyCommitParams::cpu_lanes`' comment becomes "the output parts holding their partial sums, bit 2g + 0 group g's CPU hits', bit 2g + 1 its CPU misses'"; in the CW kernel's thread 0 loop:

```cpp
      const uint32_t pair = 2u * static_cast<uint32_t>(p.lane_node[lane]);
      if (kind == Wire::kKindHitCpu) parts |= 1u << pair;
      if (kind == Wire::kKindMissCpu) parts |= 2u << pair;
```

`static_assert(2 * device::expert_stream::Wire::kNodes <= 32, "a part mask is one u32");` next to the lanes assertion; the `lease_copy_wait` launcher takes `lane_node` after `lane_slot`.

`exl3_route_tables.cuh`: replace `part0`/`part1` and their trap with

```cpp
  // A one-part row holds part 0 only.
  if ((parts >> 1) != 0u && part_stride == 0) __trap();
```

and the seed with

```cpp
    float seed = 0.0f;
    if (kept)
      for (uint32_t bits = parts; bits != 0; bits &= bits - 1)  // lowest part first: one node adds 0 then 1, as before
        seed += __ldcv(cpu_out + static_cast<int64_t>(__ffs(bits) - 1) * part_stride + i);
```

and its comment (`:41-46`) to "part p at cpu_out + p * part_stride for every bit p of cpu_lanes[1]: bit 2g group g's CPU hits', bit 2g + 1 its CPU misses'". `exl3_route_tables.py`'s docstring the same.

`expert_stream_transport.py`, `ExpertStreamDevice.__init__` takes `nodes: int = 1`, sets `self.wire = expert_lease_block.wire_layout(lanes, nodes)`, allocates `map_bank["staging"]` as `(layers, self.wire.nodes * self.wire.lanes)` and `self.lane_node = torch.zeros(self.wire.lanes, dtype=torch.int32, device=device)` beside `lane_slot`, loads `_device_module(layout, self.wire.lanes, self.wire.nodes)`, and passes `self.lane_node` after `self.lane_slot` in `post` and in `copy_wait`.

- [ ] **Step 4: Run.** Commit (`git add` the six device/route-table/`ram_slot_map.py` files; `git add -p python/sglang/kernels/ops/moe/expert_stream_transport.py`; `git diff --cached --stat`; message `feat(expert-stream): a miss takes its home node's staging slot; the CPU split and CPU parts are per node` with the trailers). `SYNC`, the Step 2 runs. Expected: PASS.

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU`, `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus this task's tests; at one node the typing, the parts and the seed are today's, so `test_exl3_cpu_lane_order_cuda.py` and the layer-fusion parity tests pass unchanged.

---

### Task 5: Each group's slot range, from the bindings the tier applied

**Files:**
- Modify: `python/sglang/srt/layers/moe/host_numa.py:222-254` (`allocate_bound` records `_numa_bindings`); add `MIN_LOCAL`, `slot_nodes`, `group_ranges` after `allocate_bound`
- Modify: `python/sglang/srt/layers/moe/expert_host_tier.py:300-303` (`allocate_host_slab` copies `_numa_bindings` onto the slab view, beside `_numa_bound_bytes`)
- Test: `test/registered/unit/layers/moe/test_host_numa.py` (new class `TestSlotNodes`)

**Interfaces:**
- Consumes: `plan_bindings` (host_numa.py:156), `allocate_bound` (:222), the arena owner attribute `_expert_stream_slab_arena` (expert_host_tier.py).
- Produces: `tensor._numa_bindings: list[tuple[int, int, int]]`, (node, start, end) byte offsets from the tensor's own base, exactly the ranges `_mbind` was called with; `host_numa.MIN_LOCAL = 0.99`; `host_numa.slot_nodes(slabs: Mapping[str, torch.Tensor], capacity: int, min_local: float = MIN_LOCAL) -> list[Optional[int]]`; `host_numa.group_ranges(rows: Sequence[Sequence[Optional[int]]], nodes: Sequence[int]) -> list[list[tuple[int, int]]]` indexed `[group][row] -> (lo, hi)`.

- [ ] **Step 1: Write the failing test.** Append to `test/registered/unit/layers/moe/test_host_numa.py` (add `group_ranges`, `slot_nodes` to the `host_numa` import list):

```python
def _bound(rows, row_bytes, seam, nodes=(0, 1)):
    """A uint8 [rows, row_bytes] slab as allocate_bound leaves it, its bytes bound to nodes[0] below ``seam`` and
    nodes[1] from it; no real mbind."""
    with patch.object(host_numa, "_mbind"):
        owner = allocate_bound(rows * row_bytes, [(nodes[0], 0, 1)], rows * row_bytes)
    slab = owner.view(rows, row_bytes)  # a new tensor object: the attribute goes on it, as allocate_host_slab does
    slab._numa_bindings = [(nodes[0], 0, seam), (nodes[1], seam, -(-rows * row_bytes // HUGE_BYTES) * HUGE_BYTES)]
    return slab


class TestSlotNodes(unittest.TestCase):
    def test_allocate_bound_records_exactly_the_bindings_it_applied(self):
        rows, row_bytes = 10, 3 * MIB + 100
        calls = []
        with patch.object(host_numa, "_mbind", side_effect=lambda a, n, node: calls.append((a, n, node))):
            slab = allocate_bound(rows * row_bytes, [(0, 0, 6), (1, 6, 4)], row_bytes)
        base = slab.data_ptr()
        self.assertEqual(slab._numa_bindings, [(node, a - base, a - base + n) for a, n, node in calls])

    def test_slots_take_the_node_that_holds_their_bytes(self):
        slab = _bound(10, MIB, 4 * MIB)
        self.assertEqual(slot_nodes({"w": slab}, 10), [0] * 4 + [1] * 6)

    def test_a_slot_across_a_seam_belongs_to_no_group(self):
        """Review Focus 5: 3 MiB rows with the node change at 8 MiB, which rounding put inside row 2. Row 2 belongs to
        no node and so to no group; rows 0-1 are node 0's and rows 3-5 node 1's."""
        slab = _bound(6, 3 * MIB, 8 * MIB)
        owners = slot_nodes({"w": slab}, 6)
        self.assertEqual(owners, [0, 0, None, 1, 1, 1])
        self.assertEqual(group_ranges([owners], [0, 1]), [[(0, 2)], [(3, 6)]])

    def test_a_small_slab_wholly_on_one_node_does_not_move_a_slot(self):
        """A layer's sign-vector slabs fit one 2 MiB page, so all of them sit on one node: under 1% of a slot."""
        big = _bound(6, 3 * MIB, 9 * MIB)
        small = _bound(6, 4096, 2 * MIB, nodes=(0, 0))
        self.assertEqual(slot_nodes({"trellis": big, "suh": small}, 6), [0, 0, 0, 1, 1, 1])

    def test_an_arena_slabs_offset_into_its_owner_counts(self):
        with patch.object(host_numa, "_mbind"):
            arena = allocate_bound(12 * MIB, [(0, 0, 1)], 12 * MIB)
        arena._numa_bindings = [(0, 0, 6 * MIB), (1, 6 * MIB, 12 * MIB)]
        first, second = arena[: 6 * MIB].view(3, 2 * MIB), arena[6 * MIB :].view(3, 2 * MIB)
        for slab in (first, second):
            slab._expert_stream_slab_arena = arena
        self.assertEqual(slot_nodes({"a": first}, 3), [0, 0, 0])
        self.assertEqual(slot_nodes({"b": second}, 3), [1, 1, 1])
        self.assertEqual(slot_nodes({"a": first, "b": second}, 3), [None, None, None])

    def test_a_slab_without_a_placement_is_refused(self):
        with self.assertRaisesRegex(ValueError, "without a placement"):
            slot_nodes({"w": torch.zeros(4, 16, dtype=torch.uint8)}, 4)

    def test_group_ranges_refuse_a_split_or_starved_node(self):
        with self.assertRaisesRegex(ValueError, "not one contiguous range"):
            group_ranges([[0, 0, 1, 1, 0]], [0, 1])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            group_ranges([[0, 1, 1, 1]], [0, 1])
        with self.assertRaisesRegex(ValueError, "node 2"):
            group_ranges([[0, 0, 2, 2]], [0, 1])
        self.assertEqual(group_ranges([[0, 0, 1, 1], [0, 0, 0, 1, 1]], [0, 1]), [[(0, 2), (0, 3)], [(2, 4), (3, 5)]])
```

- [ ] **Step 2: Run it to verify it fails.** Commit the test alone (`git add test/registered/unit/layers/moe/test_host_numa.py`; `git diff --cached --stat`; message `test(numa): slot nodes and group ranges from the applied bindings (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_host_numa.py`.
Expected: FAIL, `ImportError: cannot import name 'group_ranges' from 'sglang.srt.layers.moe.host_numa'`.

- [ ] **Step 3: Implement.** In `allocate_bound`, collect `applied = []` and append `(node, lo, hi)` in the loop that calls `_mbind(base + lo, hi - lo, node)`; after it, `tensor._numa_bindings = applied`; the zero-byte early return sets `tensor._numa_bindings = []`; extend its docstring by one sentence: "The ranges themselves, as (node, start, end) byte offsets from the tensor's base, are its ``_numa_bindings``: ``slot_nodes`` reads them, so a group's slots are exactly the pages bound to its node." In `allocate_host_slab`, beside `slab._numa_bound_bytes = storage._numa_bound_bytes`, add `slab._numa_bindings = storage._numa_bindings`. Then, after `allocate_bound`:

```python
# A slot belongs to a node holding this share of its bytes. Arbitrary: a layer's sign-vector slabs fit one 2 MiB page
# and so sit on one node, about 0.1% of a DeepSeek V4.1 slot; a slot across a seam holds far less on either node.
MIN_LOCAL = 0.99


def slot_nodes(slabs: Mapping[str, torch.Tensor], capacity: int, min_local: float = MIN_LOCAL) -> list[Optional[int]]:
    """Each of a layer's ``capacity`` slots' NUMA node, from the bindings ``allocate_bound`` applied to its slabs.

    A slot's bytes are its row of every named slab (an arena slab at its offset into the arena). It belongs to the
    node holding at least ``min_local`` of them; None, belonging to no NUMA group, when no node does. Raises ValueError
    for a slab allocated without a placement.
    """
    per_slot = [Counter() for _ in range(capacity)]
    for name, slab in slabs.items():
        owner = getattr(slab, "_expert_stream_slab_arena", slab)
        bindings = getattr(owner, "_numa_bindings", None)
        if bindings is None:
            raise ValueError(f"slab {name} has no NUMA bindings: it was allocated without a placement")
        if capacity == 0 or slab.numel() == 0:
            continue
        row_bytes = slab.numel() * slab.element_size() // slab.shape[0]
        offset = slab.data_ptr() - owner.data_ptr()
        for slot in range(capacity):
            lo = offset + slot * row_bytes
            for node, start, end in bindings:
                overlap = min(lo + row_bytes, end) - max(lo, start)
                if overlap > 0:
                    per_slot[slot][node] += overlap
    owners: list[Optional[int]] = []
    for counts in per_slot:
        total = sum(counts.values())
        node, held = counts.most_common(1)[0] if counts else (None, 0)
        owners.append(node if total and held >= min_local * total else None)
    return owners


def group_ranges(rows: Sequence[Sequence[Optional[int]]], nodes: Sequence[int]) -> list[list[tuple[int, int]]]:
    """Per NUMA group (in ``nodes`` order), per row: the slot range [lo, hi) its node holds.

    ``rows[r]`` is ``slot_nodes``' answer for row r. Raises ValueError when a node's slots of a row are not one
    contiguous range, a slot names a node outside ``nodes``, or a group holds fewer than 2 slots of a row (it could
    not stage a miss and keep a hit at once).
    """
    ranges: list[list[tuple[int, int]]] = [[] for _ in nodes]
    for row, owners in enumerate(rows):
        stray = sorted({owner for owner in owners if owner is not None and owner not in nodes})
        if stray:
            raise ValueError(f"row {row}: slots on node {stray[0]}, which is not one of the tier's nodes {list(nodes)}")
        for group, node in enumerate(nodes):
            slots = [slot for slot, owner in enumerate(owners) if owner == node]
            if len(slots) < 2:
                raise ValueError(f"row {row}: node {node} holds {len(slots)} slots; a NUMA group needs at least 2")
            if slots[-1] + 1 - slots[0] != len(slots):
                raise ValueError(f"row {row}: node {node}'s slots are not one contiguous range")
            ranges[group].append((slots[0], slots[-1] + 1))
    return ranges
```

(`Mapping`, `Optional` join the `typing` import.)

- [ ] **Step 4: Run.** Commit (`git add python/sglang/srt/layers/moe/host_numa.py python/sglang/srt/layers/moe/expert_host_tier.py`; `git diff --cached --stat`; message `feat(numa): every slab records its bindings; slot_nodes and group_ranges derive the groups from them` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_host_numa.py test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_host_slab_arena.py`. Expected: PASS.

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU`. Expected: Baseline counts plus this task's tests.

---

### Task 6: NUMA groups in the RAM tier: per-node serve state, lane filter, staging, victims and the merged delta

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/numa_group.h`, `host/numa_distributor.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (whole record path; ctor `:90-133`; `open` `:145`; `pump_demand` `:204-249`; `handled_through` `:252`; the eager paths `:296-400`; `reserve_staging` `:758-786`; `counters` `:937`; `publish_delta_locked` `:1191-1210` deleted; `take_admit_slot_locked`/`take_slot_locked` `:1239-1308`; `stamp_routed_locked`, `touch_request`, `take_victim_locked` `:1318-1366`; `collect_wanted_locked` through `handle_record` `:1395-1701`; members `:1703-1789`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h` (one service thread per group, one watchdog)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_core.h:255-320` and `uring_reader.h:60-90,415-430` (`set_sq_thread_cpu`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`open` `:183-261`: `ranges`, `sq_thread_cpus`; `start_thread` `:569-604`: per-group cores, the `64-71` refusal and its warning removed; new `group_counters`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h:576-600` (`pump` pumps every group; `handled_through`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h:120-180` (the `RamTier` constructor's two new arguments, one group; the `RamThread` core list)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamHost.__init__` `:1218-1322`, `start_thread` `:1337`, new `group_counters`)
- Modify: `python/sglang/test/dsv41_chain_sim.py:60-110` (every node's staging list and split table)
- Test: create `test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py`

**Interfaces:**
- Consumes: `Wire::home`, `Wire::kNodes` (Task 3); `type_lanes(..., nodes=)`, `MapReplica(..., nodes=)` (Task 4).
- Produces (C++): `struct GroupRow { int64_t lo, hi; FixedVec<int32_t, Wire::kLanes> staging; uint64_t chain; int64_t owned; }`; `template <class Source> struct NumaGroup` (members below); `struct DeltaReport`; `template <class Source> class NumaNodeDistributor` with `home`, `size`, `group(g)`, `home_group(expert)`, `miss_nodes(request)`, `host_nodes(request)`, `seed(row, g, staging)`, `report(request, g, part, expected) -> bool`, `publish(lease, request, reporters) -> bool`, `publish_seed(lease, row)`; `RamTier(page, slot_map, lease, lease_bytes, tables, capacity, direct, hot_page, hot_bytes, std::vector<std::vector<std::pair<int64_t, int64_t>>> ranges, std::vector<int> sq_thread_cpus)`; `RamTier::pump_demand(int g)`, `pump_demand()` (every group, pump mode), `busy_episode(int g)`, `set_counter(int g, int index, int64_t value)`, `groups()`, `group_counters(int g, int64_t* out)`; `RamThread(std::shared_ptr<Tier>, std::vector<int> cpu_cores, int64_t fatal_wait_ns, int64_t spin_ns, bool busy_poll)`; `ReaderCore::set_sq_thread_cpu(int cpu)` (`kEnvSqThreadCpu = -2` keeps `UringOptions::from_env`'s); FFI `expert_stream_open(..., lease, hot_page, ranges int64 [nodes, rows, 2], sq_thread_cpus int64 [nodes])`, `expert_stream_start_thread(handle, cpu_cores int64 [nodes], fatal_wait_ns, spin_ns, busy_poll)`, `expert_stream_group_counters(handle, group, out int64 [kCounterCount])`.
- Produces (Python): `ExpertStreamHost(tables, *, page, slot_map, lease_block=None, hot_page=None, layout="exl3", variant=None, lanes=8, node_ranges: Optional[Sequence[Sequence[tuple[int, int]]]] = None, sq_thread_cpus: Optional[Sequence[int]] = None)` with `self.nodes`, `self.node_ranges`; `start_thread(*, cpu_core: int | Sequence[int] = -1, fatal_wait_s=30.0, spin_us=5000, busy_poll=False)`; `group_counters(group: int) -> dict[str, int]`. `ChainSim` reads `host.wire.nodes`.
- One-node identity: `ranges` defaults to `[[(0, capacity[row]) for row]]`, so group 0 owns every slot, its staging slots are `0..k-1`, `publish` writes exactly v2's delta, and the counters, thread name and messages are today's.

- [ ] **Step 1: Write the failing test** `test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py`:

```python
"""The RAM tier with two NUMA groups (spec 2026-10-03-numa-node-distributor-design, Part 3 and Testing 3): each
group serves only its home lanes (expert % 2), stages and evicts only in its own slots, and the combiner merges
the groups' map deltas into one per record. ChainSim plays the device with the node-aware reference typing."""

import subprocess
import sys
import textwrap

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, paused, ram_miss_setup

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROW = 1
EXPERTS = 8
HALVES = [[(0, 4)] * 2, [(4, 8)] * 2]  # [group][row] -> (lo, hi): node 0 slots 0-3, node 1 slots 4-7


def _host(tmp_path, *, ranges=HALVES, capacity=8, staging=2):
    s = ram_miss_setup(tmp_path, capacity=capacity, experts=EXPERTS)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    host = ExpertStreamHost(
        s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), variant="instr",
        node_ranges=ranges,
    )
    host.reserve_staging(staging)
    return s, page, host, ChainSim(host, page, s.slabs)


def _serve(sim, host, experts):
    req = sim.post(ROW, experts)
    assert host.pump() == 1 and sim.wait_served(req)
    return req


def _lists(staging):
    return staging[:8], staging[8:]


def test_reserve_staging_takes_each_groups_lowest_slots_and_publishes_every_nodes_list(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        tag, staging, entries = sim.delta(ROW)
        assert (tag, entries) == (1, [])
        assert _lists(staging) == ([0, 1] + [-1] * 6, [4, 5] + [-1] * 6)
    finally:
        host.stop()


def test_each_group_serves_only_its_home_lanes(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        _serve(sim, host, [2, 1])
        mapping = host.mapping(ROW)
        assert 0 <= mapping[2] < 4 and 4 <= mapping[1] < 8
        assert host.group_counters(0)["rows_read"] == 1 and host.group_counters(1)["rows_read"] == 1
    finally:
        host.stop()


def test_misses_homed_on_one_node_publish_one_delta_with_the_other_nodes_list_unchanged(tmp_path):
    """Review Focus 1, host side: only group 1 reports, and the delta it writes carries node 0's list as reserved."""
    s, page, host, sim = _host(tmp_path)
    try:
        req = _serve(sim, host, [1, 3])
        tag, staging, entries = sim.delta(ROW)
        assert tag == req.chain == 2
        assert _lists(staging) == ([0, 1] + [-1] * 6, [6, 7] + [-1] * 6)
        assert entries == [(1, 4), (3, 5)]
    finally:
        host.stop()


def test_both_nodes_missing_in_one_record_yield_exactly_one_merged_delta(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        req = _serve(sim, host, [2, 1])
        tag, staging, entries = sim.delta(ROW)
        assert tag == req.chain == 2, "one delta per record, not one per group"
        assert _lists(staging) == ([2, 1] + [-1] * 6, [6, 5] + [-1] * 6)
        assert entries == [(2, 0), (1, 4)], "node 0's entries, then node 1's"
        _serve(sim, host, [2, 1])  # both hits now: the next post applied the merged delta
        assert host.mapping(ROW)[2] == 0 and host.mapping(ROW)[1] == 4
    finally:
        host.stop()


def test_victims_come_only_from_the_groups_range(tmp_path):
    s, page, host, sim = _host(tmp_path, staging=1)
    try:
        for expert in (0, 2, 4, 1, 3, 5):  # fills both ranges: three mapped rows and one staging slot each
            _serve(sim, host, [expert])
        _serve(sim, host, [7])
        mapping = host.mapping(ROW)
        assert mapping[1] == -1, "node 1's LRU row is the victim"
        assert all(0 <= mapping[e] < 4 for e in (0, 2, 4)), "node 0 lost nothing"
        assert 4 <= mapping[7] < 8
        _serve(sim, host, [6])
        assert host.mapping(ROW)[0] == -1 and 0 <= host.mapping(ROW)[6] < 4
    finally:
        host.stop()


def test_no_slot_outside_a_groups_range_is_ever_staged_or_taken(tmp_path):
    """Review Focus 5: slot 4 straddles the seam and belongs to no group. Forty random records of hits and misses on
    both nodes never stage it, map it or evict into it."""
    import random

    rng = random.Random(4)
    s, page, host, sim = _host(tmp_path, ranges=[[(0, 4)] * 2, [(5, 9)] * 2], capacity=9)
    try:
        for _ in range(40):
            experts = rng.sample(range(EXPERTS), rng.randint(1, 2))
            _serve(sim, host, experts)
            assert 4 not in sim.staging(ROW)
            state, expert, _ = host.slot_info(ROW)[4]
            assert (state, expert) == (0, -1)
        assert 4 not in host.mapping(ROW)
    finally:
        host.stop()


def test_every_group_parks_for_a_pause_and_serves_after_it(tmp_path):
    s, page, host, sim = _host(tmp_path)
    try:
        host.start_thread(fatal_wait_s=60.0)
        req = sim.post(ROW, [2, 1])
        assert sim.wait_served(req, timeout_s=5.0)
        with paused(host):
            assert [state for state, _, _ in host.slot_info(ROW)].count(2) == 2
        req = sim.post(ROW, [4, 3])
        assert sim.wait_served(req, timeout_s=5.0)
        assert sim.wait_handled(req)
    finally:
        host.stop()


def test_ranges_that_overlap_or_leave_the_row_are_refused(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=8, experts=EXPERTS)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    for ranges, match in [([[(0, 5)] * 2, [(4, 8)] * 2], "overlap"), ([[(0, 4)] * 2, [(4, 9)] * 2], "outside")]:
        with pytest.raises(ValueError, match=match):
            ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32),
                             variant="instr", node_ranges=ranges)


_SCRIPT = """
import pathlib, sys, torch
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
s = ram_miss_setup(pathlib.Path(sys.argv[1]), capacity=8, experts=8)
page = new_page(pin=False, wire=wire_layout(8, 2))
host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 8), -1, dtype=torch.int32), variant="instr",
                        node_ranges=[[(0, 4)] * 2, [(4, 8)] * 2])
host.reserve_staging(2)
sim = ChainSim(host, page, s.slabs)
"""


def test_a_miss_on_another_nodes_staging_slot_fail_stops(tmp_path):
    """A device that put node 1's miss in node 0's staging slot 0 would have it read into node 0's memory: group 1
    checks its own list and fail-stops. A row-wide list would accept slot 0 (mutant: red)."""
    body = "sim.post(1, [1], kinds=[LaneKind.MISS_GPU], slots=[0])\nhost.pump()\nprint('reached')\n"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT) + body, str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert_aborted(result, "is not a staging slot")
```

- [ ] **Step 2: Run it to verify it fails.** Commit the test alone (`git add test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py`; `git diff --cached --stat`; message `test(expert-stream): two NUMA groups in the RAM tier (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py`.
Expected: FAIL, `TypeError: ExpertStreamHost.__init__() got an unexpected keyword argument 'node_ranges'`.

- [ ] **Step 3: Implement.**

**`host/numa_group.h`** (new):

```cpp
// One NUMA node's share of the RAM tier (spec 2026-10-03-numa-node-distributor-design, Part 3): its slots of every
// row, and the serve state its RAM thread owns. NumaNodeDistributor holds one per node of the build.
#pragma once

#include "copy_engine.h"

namespace sglang::expert_stream {

// A group's part of one row.
struct GroupRow {
  int64_t lo = 0;  // the group's slots of the row are [lo, hi)
  int64_t hi = 0;
  // Its staging slots, in the order the device gives them to this node's misses, and the row's map chain as this
  // group last served it (0: reserve_staging has not run, so the row serves no miss).
  FixedVec<int32_t, Wire::kLanes> staging;
  uint64_t chain = 0;
  int64_t owned = 0;  // prefill-owned slots of the range (Tier::prefill_owned)
};

// Everything one group's service thread owns. The tier's single-owner rule holds per group: its service thread while
// it runs, and the caller that paused every group (or called pump()) otherwise.
template <class Source>
struct NumaGroup {
  NumaGroup(int index, int sq_thread_cpu, Tables tables, bool direct, std::vector<GroupRow> rows)
      : index(index), sq_thread_cpu(sq_thread_cpu), reader(std::move(tables), direct), rows(std::move(rows)) {}

  bool owns(int64_t row, int64_t slot) const {
    return slot >= rows[row].lo && slot < rows[row].hi;
  }

  const int index;          // the wire's node axis: Wire::home(expert) == index for every expert it serves
  const int sq_thread_cpu;  // its ring's SQPOLL core; -1 unpinned, ReaderCore::kEnvSqThreadCpu the env's
  Source reader;            // its own io_uring ring
  std::vector<GroupRow> rows;
  std::unique_ptr<CpuExpertEngine> cpu;  // CPU experts on this node's cores, when enabled
  uint64_t tick = 0;                     // its LRU clock: stamps compare only within its own slots
  uint32_t next_demand = 1;
  std::atomic<uint32_t> handled{0};  // the last seq this group finished
  std::atomic<uint64_t> busy{0};     // RamTier::busy_episode of this group
  uint64_t episodes = 0;
  int64_t demands_read = 0;
  std::vector<uint8_t> packed;
  std::vector<PieceTarget> piece_targets;
  PiecePublish piece_publish;
  std::vector<uint8_t> hot_scratch;
  LineCounters<kCounterCount> core;  // RamTier::count's block for this group's thread
};

}  // namespace sglang::expert_stream
```

**`host/numa_distributor.h`** (new):

```cpp
// The NUMA groups of one RAM tier, the home rule, and the combiner that merges the groups' map deltas
// (spec 2026-10-03-numa-node-distributor-design, Part 3). At one node: one group, and every report completes.
#pragma once

#include "numa_group.h"

namespace sglang::expert_stream {

// One group's part of a record's map delta: its entries ({expert, slot}, slot -1 an eviction) and its staging list
// after its victims.
struct DeltaReport {
  int count = 0;
  int32_t entries[Wire::kDeltaMaxEntries][2];
  FixedVec<int32_t, Wire::kLanes> staging;
};

template <class Source>
class NumaNodeDistributor {
 public:
  using Group = NumaGroup<Source>;

  NumaNodeDistributor(std::vector<std::unique_ptr<Group>> groups, int64_t rows)
      : groups_(std::move(groups)), rows_(static_cast<size_t>(rows)) {}

  static int home(int64_t expert) {
    return Wire::home(expert);
  }
  int size() const {
    return static_cast<int>(groups_.size());
  }
  Group& group(int g) {
    return *groups_[g];
  }
  const Group& group(int g) const {
    return *groups_[g];
  }
  Group& home_group(int64_t expert) {
    return *groups_[home(expert)];
  }

  // The nodes with a miss lane in `request`: the groups that report its delta. Every group computes it from the
  // whole record it read.
  static uint32_t miss_nodes(const Request& request) {
    uint32_t nodes = 0;
    for (const Lane& lane : request.lanes)
      if (is_miss(lane.kind)) nodes |= 1u << home(lane.expert);
    return nodes;
  }

  // The nodes with a host lane (HIT_COPY, HIT_CPU, MISS_CPU): the groups that send the copy engine a part.
  static uint32_t host_nodes(const Request& request) {
    uint32_t nodes = 0;
    for (const Lane& lane : request.lanes)
      if (lane.kind == Wire::kKindHitCopy || lane.kind == Wire::kKindHitCpu || lane.kind == Wire::kKindMissCpu)
        nodes |= 1u << home(lane.expert);
    return nodes;
  }

  // reserve_staging's lists: group g's slots before any record. The owner, before any thread runs.
  void seed(int64_t row, int g, const FixedVec<int32_t, Wire::kLanes>& staging) {
    rows_[row].staging[g] = staging;
  }

  // Group g reports its part of `request`'s delta; `expected` is miss_nodes(request). Returns true for exactly one
  // reporter, the last, which then publishes. The state word is (chain << 32 | groups reported), so a report of the
  // row's next chain never counts this one's bits. Every write of the word is a read-modify-write, so the publisher's
  // acquire sees every earlier report of this row, this record's and the reports that left the other nodes' lists.
  bool report(const Request& request, int g, const DeltaReport& part, uint32_t expected) {
    RowCombine& row = rows_[request.row];
    row.parts[g] = part;
    row.staging[g] = part.staging;
    const uint64_t chain = (request.chain & 0xFFFFFFFFull) << 32;
    uint64_t seen = row.reported.load(std::memory_order_relaxed);
    uint64_t next;
    do {
      const uint64_t mask = (seen & ~0xFFFFFFFFull) == chain ? seen & 0xFFFFFFFFull : 0;
      next = chain | mask | (1ull << g);
    } while (!row.reported.compare_exchange_weak(seen, next, std::memory_order_acq_rel, std::memory_order_relaxed));
    return (next & 0xFFFFFFFFull) == expected;
  }

  // The last reporter writes the row's one delta: every node's staging list, the reporters' entries in node order,
  // then the tag with a release (entries before the tag, as the device reads them). False when the row's previous
  // delta is not chain - 1, which the device's post order rules out.
  bool publish(uint8_t* lease, const Request& request, uint32_t reporters) {
    RowCombine& row = rows_[request.row];
    if (row.written.load(std::memory_order_acquire) + 1 != request.chain) return false;
    int32_t entries[Wire::kDeltaMaxEntries][2];
    int count = 0;
    for (int g = 0; g < size(); ++g) {
      if ((reporters >> g & 1u) == 0) continue;
      for (int i = 0; i < row.parts[g].count; ++i) {
        entries[count][0] = row.parts[g].entries[i][0];
        entries[count][1] = row.parts[g].entries[i][1];
        ++count;
      }
    }
    write(lease, request.row, request.chain, entries, count);
    return true;
  }

  // reserve_staging's tag-1 delta: every node's seeded list, no entries.
  void publish_seed(uint8_t* lease, int64_t row) {
    write(lease, row, 1, nullptr, 0);
  }

 private:
  struct RowCombine {
    std::atomic<uint64_t> reported{0};
    std::atomic<uint64_t> written{0};  // the chain of the row's last delta
    std::array<FixedVec<int32_t, Wire::kLanes>, Wire::kNodes> staging;
    std::array<DeltaReport, Wire::kNodes> parts;
  };

  // The payload with plain stores, an sfence, then the tag with a release (the device acquires the tag before it
  // reads the rest). The device read the row's previous delta in its last post, which came before this record, so
  // nothing reads the record while it is rewritten.
  void write(uint8_t* lease, int64_t row_index, uint64_t tag, const int32_t (*entries)[2], int count) {
    RowCombine& row = rows_[row_index];
    uint8_t* d = lease + Wire::kDeltaBase + row_index * Wire::kDeltaStride;
    const uint32_t n = static_cast<uint32_t>(count);
    std::memcpy(d + Wire::kDeltaCount, &n, 4);
    for (int node = 0; node < Wire::kNodes; ++node) {
      const FixedVec<int32_t, Wire::kLanes>& list = row.staging[node];
      for (int k = 0; k < Wire::kLanes; ++k) {
        const int16_t slot = static_cast<int16_t>(k < static_cast<int>(list.size()) ? list[k] : -1);
        std::memcpy(d + Wire::kDeltaStaging + 2 * (node * Wire::kLanes + k), &slot, 2);
      }
    }
    for (int i = 0; i < count; ++i) {
      const int16_t entry[2] = {static_cast<int16_t>(entries[i][0]), static_cast<int16_t>(entries[i][1])};
      std::memcpy(d + Wire::kDeltaEntries + 4 * i, entry, 4);
    }
    _mm_sfence();
    store_release64(d + Wire::kDeltaTag, tag);
    row.written.store(tag, std::memory_order_release);
  }

  std::vector<std::unique_ptr<Group>> groups_;
  std::vector<RowCombine> rows_;
};

}  // namespace sglang::expert_stream
```

(`RowCombine` holds atomics, so `rows_` is sized once in the constructor and never grows.)

**`host/reader_core.h` / `uring_reader.h`:** `BasicUringReader` gains `int sq_override_ = kEnvSqThreadCpu;` with `static constexpr int kEnvSqThreadCpu = -2;` and

```cpp
  // Replaces SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU for this ring: the RAM tier's groups get their cores from
  // ThreadingConfig. -1 leaves the SQPOLL thread unpinned; kEnvSqThreadCpu keeps the env's value. Before init().
  void set_sq_thread_cpu(int cpu) {
    sq_override_ = cpu;
  }
```

and in `init`, right after `options_ = UringOptions::from_env();`:

```cpp
    if (sq_override_ != kEnvSqThreadCpu) {
      if (sq_override_ >= 0 && !options_.sqpoll())
        throw std::invalid_argument("SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU: requires a sqpoll mode");
      options_.sq_thread_cpu = sq_override_;
    }
```

`ReaderCore` forwards it: `void set_sq_thread_cpu(int cpu) { io_.set_sq_thread_cpu(cpu); }` (comment: "Before open(): the ring's SQPOLL core (BasicUringReader::set_sq_thread_cpu).").

**`host/ram_tier.h`:**

- Includes `numa_distributor.h`. `Tier` loses `staging`, `chain` and `owned` (moved to `GroupRow`); its comment says so; `rows_demand` is now written by any group with `std::atomic_ref<int64_t>(...).fetch_add(n, std::memory_order_relaxed)`.
- `using Group = NumaGroup<Source>;`. Members `reader_`, `cpu_`, `tick_`, `next_demand_`, `handled_`, `demands_read_`, `busy_`, `episodes_`, `packed_`, `hot_scratch_`, `piece_targets_`, `piece_publish_` and `core_` are deleted; `NumaNodeDistributor<Source> dist_;` replaces them, declared after `tiers_`.
- The class comment's "Threads" paragraph gains: "With several NUMA groups (numa_group.h) each group's service thread serves the lanes homed on its node, touching only its own slots of a row's slot-indexed arrays and its own experts of the expert-indexed ones; the single-owner rule holds per group."
- Constructor: two new last parameters `std::vector<std::vector<std::pair<int64_t, int64_t>>> ranges` (`[group][row]`) and `std::vector<int> sq_thread_cpus`; the initializer builds `dist_(make_groups(tables, capacity, direct, ranges, sq_thread_cpus), tables.layers)` before `tiers_`, from:

```cpp
  // One group per node of the build, over its slot range of every row. Throws on a range count other than
  // Wire::kNodes, a range outside its row, or two that overlap.
  static std::vector<std::unique_ptr<Group>> make_groups(
      const Tables& tables,
      const std::vector<int64_t>& capacity,
      bool direct,
      const std::vector<std::vector<std::pair<int64_t, int64_t>>>& ranges,
      const std::vector<int>& sq_thread_cpus) {
    if (static_cast<int>(ranges.size()) != Wire::kNodes || static_cast<int>(sq_thread_cpus.size()) != Wire::kNodes)
      throw std::runtime_error(error_prefix<Layout>() + "this build serves " + std::to_string(Wire::kNodes) + " NUMA groups");
    std::vector<std::unique_ptr<Group>> groups;
    for (int g = 0; g < Wire::kNodes; ++g) {
      if (static_cast<int64_t>(ranges[g].size()) != tables.layers)
        throw std::runtime_error(error_prefix<Layout>() + "a NUMA group needs a slot range per row");
      std::vector<GroupRow> rows(static_cast<size_t>(tables.layers));
      for (int64_t row = 0; row < tables.layers; ++row) {
        const auto [lo, hi] = ranges[g][row];
        if (lo < 0 || hi > capacity[row] || lo >= hi)
          throw std::runtime_error(error_prefix<Layout>() + "group " + std::to_string(g) + "'s slots of row " + std::to_string(row) + " are outside the row");
        for (int other = 0; other < g; ++other) {
          const auto [olo, ohi] = ranges[other][row];
          if (lo < ohi && olo < hi)
            throw std::runtime_error(error_prefix<Layout>() + "groups " + std::to_string(other) + " and " + std::to_string(g) + " overlap in row " + std::to_string(row));
        }
        rows[row].lo = lo;
        rows[row].hi = hi;
      }
      groups.push_back(std::make_unique<Group>(g, sq_thread_cpus[g], tables, direct, std::move(rows)));
    }
    return groups;
  }
```

  and the body sizes every group's `hot_scratch`, `packed` and `piece_targets` as it sized the tier's, and calls `group.reader.set_piece_stream(true)` for each.
- `open()`: for each group, `group.reader.set_sq_thread_cpu(group.sq_thread_cpu)`, `if (!group.reader.open()) return false;`, `group.next_demand = skip_zero(load_acquire(page_ + Wire::kDemandHead) + 1u);`.
- `busy_episode(int g)` returns `dist_.group(g).busy`; `set_counter(int g, int index, int64_t value)` writes `dist_.group(g).core`; `count<K>(Group& g, n)` counts into `g.core`, and the existing `count<K>(n)` (the eager paths, on the owner with every group parked) into group 0's. `int groups() const { return dist_.size(); }`. `counters(out)` sums every group's block, `copy_core_` and `stats_`, except `kSpinCpu`, which is group 0's; `group_counters(int g, int64_t* out)` copies group g's block.
- `handled_through()`: the least of the groups', across the wrap:

```cpp
  uint32_t handled_through() const {
    uint32_t least = dist_.group(0).handled.load(std::memory_order_acquire);
    for (int g = 1; g < dist_.size(); ++g) {
      const uint32_t h = dist_.group(g).handled.load(std::memory_order_acquire);
      if (!reached(h, least)) least = h;
    }
    return least;
  }
```

- `pump_demand(int g)` is today's `pump_demand()` with `next_demand_` -> `group.next_demand`, `handled_` -> `group.handled`, `count<K>(...)` -> `count<K>(group, ...)`, `read_gpu_hot(group, ...)`, `apply_gpu_hot(group, request)`, `handle_record(group, request)`, and `begin_stage`/`end_stage` only when `g == 0`. `pump_demand()` (pump mode) runs `pump_demand(g)` for every group and returns whether group 0 handled a record.
- `stage_record(Group& group)` returns null unless `group.index == 0` (one `StageRecord`, filled by one thread).
- `read_gpu_hot(Group&, ...)` uses `group.hot_scratch`; `apply_gpu_hot(Group& group, const Request& request)` writes only the group's experts: `for (int64_t expert = group.index; expert < experts_; expert += Wire::kNodes)`.
- `reserve_staging(k)`:

```cpp
  void reserve_staging(int64_t k) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("reserve_staging");
    if (k < 1 || k > Wire::kLanes)
      throw std::runtime_error(error_prefix<Layout>() + "a row has 1.." + std::to_string(Wire::kLanes) + " staging slots");
    for (int64_t row = 0; row < layers_; ++row) {
      const Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        const GroupRow& own = dist_.group(g).rows[row];
        if (own.chain != 0) throw std::runtime_error(error_prefix<Layout>() + "reserve_staging is once");
        if (own.hi - own.lo < 2)
          throw std::runtime_error(
              error_prefix<Layout>() + "row " + std::to_string(row) + " has too few slots to stage" +
              (dist_.size() > 1 ? " in group " + std::to_string(g) : std::string()));
      }
      for (int64_t slot = 0; slot < tier.capacity; ++slot)
        if (tier.state[slot] != kFree)
          throw std::runtime_error(error_prefix<Layout>() + "reserve_staging is before any slot is filled");
    }
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        GroupRow& own = dist_.group(g).rows[row];
        const int64_t want = std::min(k, own.hi - own.lo - 1);
        for (int64_t slot = own.lo; slot < own.lo + want; ++slot) {
          tier.state[slot] = kStaging;
          own.staging.push_back(static_cast<int32_t>(slot));
        }
        own.chain = 1;
        dist_.seed(row, g, own.staging);
      }
      dist_.publish_seed(lease_, row);
    }
  }
```

- `touch_request(Group&, ...)` and `stamp_routed_locked(Group& group, Tier& tier, int32_t expert)` stamp with `++group.tick`; `disown_locked(GroupRow& own, Tier& tier, int64_t slot)` decrements `own.owned`.
- `take_victim_locked(Group& group, int64_t row, std::span<const int32_t> wanted, int32_t* old)`: today's body with both loops over `slot` in `[own.lo, own.hi)` (`const GroupRow& own = group.rows[row];`) and `count<kEvictions>(group)`.
- Eager paths: `take_admit_slot_locked(int64_t row, int64_t expert, std::span<const int32_t> protect, bool fallback, int64_t* evicted, bool stop_at_share = false)` and `take_slot_locked(int64_t row, int64_t expert, ...)` scan `[own.lo, own.hi)` of `dist_.home_group(expert)`; the share test uses the sum of every group's `rows[row].owned`, and the share bookkeeping the home group's. `assign`, `fill_begin` (per expert) and `touch` pass the expert and stamp with the home group's tick. `release_locked`, `finish_fill_owned` and the prefill-share eviction end ownership through `disown_locked(owner_row(row, slot), tier, slot)`, where `GroupRow& owner_row(int64_t row, int64_t slot)` finds the group whose range holds the slot. `run_fill` reads through `dist_.group(0).reader` and holds group 0's busy episode (every group is parked during a fill, and every ring registers the whole tier).
- `collect_wanted_locked(Group& group, Tier& tier, const Request& request, RecordPlan* plan)`: unchanged except that a protect id is stamped only when `Wire::home(expert) == group.index`.
- `classify_lanes_locked(Group& group, Tier& tier, const Request& request, RecordPlan* plan)`: the lane loop starts with `if (Wire::home(lane.expert) != group.index) continue;`; the miss test reads `listed(own.staging, lane.slot)`; the hit test adds `!group.owns(request.row, lane.slot)` to its failing condition; the stamp is `++group.tick`; `cpu_` is `group.cpu`. The chain test at the end becomes:

```cpp
    GroupRow& own = group.rows[request.row];
    if (NumaNodeDistributor<Source>::miss_nodes(request) != 0) {
      // Every group follows the row's chain, its own misses or not, so each can check the next one.
      if (request.chain != own.chain + 1) {
        fail_record(
            request,
            "map chain " + std::to_string(request.chain) + ", the row expects " + std::to_string(own.chain + 1));
      }
      own.chain = request.chain;
    } else if (request.chain != 0) {
      fail_record(request, "map chain " + std::to_string(request.chain) + " on a record without a miss");
    }
```

- `reserve_victims_locked(Group& group, Tier& tier, const Request& request, const RecordPlan& plan, bool* inserted)`:

```cpp
  void reserve_victims_locked(Group& group, Tier& tier, const Request& request, const RecordPlan& plan, bool* inserted) {
    GroupRow& own = group.rows[request.row];
    DeltaReport part;
    part.staging = own.staging;
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      int32_t old = -1;
      const int64_t victim = take_victim_locked(group, request.row, plan.wanted, &old);
      if (victim < 0) {
        this->template count<kRamInsertSkipped>(group);
        continue;
      }
      if (old >= 0) {
        part.entries[part.count][0] = old;
        part.entries[part.count][1] = -1;
        ++part.count;
      }
      part.entries[part.count][0] = plan.missing[i];
      part.entries[part.count][1] = static_cast<int32_t>(plan.slots[i]);
      ++part.count;
      const int32_t slot = static_cast<int32_t>(plan.slots[i]);
      tier.state[victim] = kStaging;
      for (int32_t& s : part.staging)
        if (s == slot) s = static_cast<int32_t>(victim);
      inserted[i] = true;
    }
    own.staging = part.staging;
    const uint32_t reporters = NumaNodeDistributor<Source>::miss_nodes(request);
    if (dist_.report(request, group.index, part, reporters) && !dist_.publish(lease_, request, reporters))
      fail_record(request, "the row's previous map delta is not chain " + std::to_string(request.chain - 1));
  }
```

- `serve_record(Group& group, ...)`, `handle_record(Group& group, ...)`, `read_misses(Group& group, ...)`, `init_piece_words_locked(Group& group, ...)`, `submit_host_lanes(Group& group, ...)`, `submit_landed_cpu_misses(Group& group, ...)`, `commit_inserted_locked(Group& group, ...)`, `begin_busy(Group&)`, `end_busy(Group&)`, `apply_pending_fault(Group&)` take the group and use its members in place of the deleted tier ones; `serve_record` calls `reserve_victims_locked` when `reads`, exactly as today.
- CPU experts at this task stay on group 0: `enable_cpu_experts`, `set_cpu_layer`, `set_cpu_split`, `cpu_stats`, `cpu_cores`, `calibrate_cpu_split` use `dist_.group(0).cpu`, and `enable_copy_engine` and `enable_cpu_experts` throw `"the copy engine and CPU experts with several NUMA groups need the combiner's completion side"` when `Wire::kNodes > 1` (Task 7 lifts this). `~RamTier` stops every group's CPU engine after the copy engine.

**`host/ram_thread.h`:** `RamThread(std::shared_ptr<Tier> tier, std::vector<int> cpu_cores, int64_t fatal_wait_ns, int64_t spin_ns, bool busy_poll)`, one entry per group. `start()` makes `threads_[g] = std::thread([this, g] { run(g); })` for every group and waits on each `pin_error_[g]` (a `std::unique_ptr<std::atomic<int>[]>`); any error stops and joins all of them before it throws `could not pin the service thread of group G to core C`. `stop()` joins every service thread, then the watchdog. `pause()` waits until every `parked_epoch_[g]` (also an array) equals the epoch. `run(int g)` is today's `run()` with `tier_->pump_demand(g)`, `tier_->set_counter(g, ...)`, `cpu_cores_[g]`, and the thread name `Layout::kName + "-ram-miss"` plus the group digit when `Wire::kNodes > 1`. `watch()` keeps one `(episode, since)` pair per group (`tier_->busy_episode(g)`); a stuck group's line is `FATAL <prefix>group G: a request stayed in service for ...` above one group and today's text at one group. The gate rule is unchanged here (Task 7 adds the stall).

**`host/ffi_exports.h`:** `open` takes `TensorView ranges` (int64 `[Wire::kNodes, rows, 2]`, CPU) and `TensorView sq_thread_cpus` (int64 `[Wire::kNodes]`) after `hot_page`, verified with `TensorMatcher`s of those shapes, and passes them to `RamTier` as `std::vector<std::vector<std::pair<int64_t, int64_t>>>` and `std::vector<int>`. `start_thread(int64_t handle, TensorView cpu_cores, int64_t fatal_wait_ns, int64_t spin_ns, int64_t busy_poll)`: `cpu_cores` int64 `[Wire::kNodes]`; each core must be `< CPU_SETSIZE`; the `64-71` refusal and the inherited-affinity warning are deleted (`check_not_reserved` in Python owns that rule); with `busy_poll`, `check_dedicated_core(core, tier->cpu_cores(), prefix)` for every group's core. New export `group_counters(int64_t handle, int64_t group, TensorView out)` (`out` int64 `[kCounterCount]`) with its `EXPERT_STREAM_HOST_EXPORTS` line. `ffi_test_exports.h`'s `pump` calls `tier->pump_demand()` (every group).

**`bench/src/stack.h`:** the `RamTier` constructor gets `{{{0, config_.rows.capacity}, ...one per row}}` and `{-1}`; the `RamThread` gets `std::vector<int>{config_.service_cpu}`.

**`expert_stream_transport.py`, `ExpertStreamHost`:**

```python
        # NUMA groups: node_ranges[g][row] = (lo, hi), group g's slots of the row; one group owning every slot by
        # default. sq_thread_cpus: each group's SQPOLL core (-1 unpinned); None keeps the uring env's.
        capacity = [int(c) for c in tables.capacity.tolist()]
        if node_ranges is None:
            node_ranges = [[(0, c) for c in capacity]]
        self.nodes = len(node_ranges)
        self.node_ranges = [[(int(lo), int(hi)) for lo, hi in rows] for rows in node_ranges]
        for g, rows in enumerate(self.node_ranges):
            if len(rows) != len(capacity):
                raise ValueError(f"node_ranges[{g}] has {len(rows)} rows, the tables {len(capacity)}")
            for row, (lo, hi) in enumerate(rows):
                if not 0 <= lo < hi <= capacity[row]:
                    raise ValueError(f"group {g}'s slots [{lo}, {hi}) of row {row} are outside its {capacity[row]} slots")
                for other in range(g):
                    olo, ohi = self.node_ranges[other][row]
                    if lo < ohi and olo < hi:
                        raise ValueError(f"groups {other} and {g} overlap in row {row}")
        self.wire = expert_lease_block.wire_layout(lanes, self.nodes)
```

(the `self.wire = ...` line at the top of `__init__` moves here, before the page check), `_host_module(self._layout, self.variant, self.wire.lanes, self.nodes)`, and the `expert_stream_open` call gains, after the hot page:

```python
                torch.tensor(self.node_ranges, dtype=torch.int64).reshape(self.nodes, len(capacity), 2),
                torch.tensor(
                    [-2] * self.nodes if sq_thread_cpus is None else [int(c) for c in sq_thread_cpus],
                    dtype=torch.int64,
                ),
```

`start_thread`:

```python
    def start_thread(self, *, cpu_core=-1, fatal_wait_s: float = 30.0, spin_us: int = 5000, busy_poll: bool = False) -> None:
        """Serve requests on one C++ thread per NUMA group (no more ``pump()``), with the watchdog.

        ``cpu_core`` is one core per group (a sequence), or one int for a single group; -1 inherits the caller's
        affinity. Cores 64-71 are reserved for NVMe completion interrupts. ``busy_poll`` spins on each group's core
        with no PAUSE and no sleep; the C++ side refuses it unless each physical core is its service's alone.
        """
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_not_reserved

        cores = [int(cpu_core)] * self.nodes if isinstance(cpu_core, int) else [int(c) for c in cpu_core]
        if len(cores) != self.nodes or (self.nodes > 1 and isinstance(cpu_core, int) and cpu_core >= 0):
            raise ValueError(f"start_thread takes one core per NUMA group ({self.nodes}), got {cpu_core!r}")
        for core in cores:
            check_not_reserved(core)
        self._module.expert_stream_start_thread(
            self.handle, torch.tensor(cores, dtype=torch.int64), int(fatal_wait_s * 1e9), int(spin_us * 1e3),
            int(busy_poll),
        )
        self.threaded = True

    def group_counters(self, group: int) -> dict[str, int]:
        """NUMA group ``group``'s own counters (its service thread's), by the names ``counters`` uses."""
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.expert_stream_group_counters(self.handle, int(group), out)
        return dict(zip(COUNTERS, out.tolist()))
```

(`COUNTERS` is the module's name tuple that `counters()` already zips with.)

**`dsv41_chain_sim.py`:** `self.replica = MapReplica(host.layers, host.experts, self.wire.lanes, self.wire.nodes)`; `delta()` reads `2 * w.nodes * w.lanes` staging bytes; `split()` returns the node-major concatenation `sum((_i32(self.block, w.split + n * w.split_stride, w.lanes + 1).tolist() for n in range(w.nodes)), [])`; `post()` passes `nodes=self.wire.nodes` to `type_lanes`. Their docstrings say "every node's list, node-major".

- [ ] **Step 4: Run.** Commit:

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/numa_group.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/numa_distributor.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_core.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_reader.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h python/sglang/test/dsv41_chain_sim.py
git add -p python/sglang/kernels/ops/moe/expert_stream_transport.py
git diff --cached --stat
git commit -m "feat(expert-stream): NUMA groups serve their home lanes from their own slots; the combiner merges one delta per record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py`. Expected: PASS.

- [ ] **Step 5: Existing suites, and the one-node wire.** `RUN_CPU SUITE_CPU`, `RUN_GPU SUITE_GPU`. Expected: Baseline counts plus this task's tests; in particular `test_exl3_ram_miss_slot_map.py` (staging, deltas), `test_exl3_ram_miss_thread.py` (watchdog, pinning, refusals, with `test_a_reserved_or_unusable_core_is_refused` still raising `ValueError` "64-71" from `check_not_reserved`), `test_exl3_ram_miss_prefill_fills.py`, `test_exl3_ram_miss_prefill_share.py` and `test_exl3_ram_miss_tier.py` unchanged. Then `BENCH nlane-bench`: `BUILD=0`, `EXIT=0`.

---

### Task 7: The completion side: one copy part per group, CopyDone on every group, per-group CPU engines, the watchdog naming a group

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h:36-58` (`CopyJob`), `:415-750` (`CopyEngine`: constructor, `submit`, `idle`, `set_cpu`, `run`, `issue`, members)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`enable_copy_engine`, `enable_cpu_experts`, `set_cpu_layer`, `set_cpu_split`, `cpu_stats`, `cpu_cores`, `calibrate_cpu_split`, `store_split`, `submit_host_lanes`, new `copy_stall`; the Task 6 guard deleted; `InstrBuild` fault `group_stall_ns`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h:45-100` (`CalibrationSetup::first_slot`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h` (`watch()`'s copy-wait line)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h:352-460` (`enable_cpu_experts`, `set_cpu_split`, `cpu_stats`, `calibrate_cpu_split` take `group`), `ffi_test_exports.h` (new `inject_group_stall`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`enable_cpu_experts`, `set_cpu_split`, `cpu_stats`, `calibrate_cpu_split` gain `group: int = 0`; new `inject_group_stall`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h:154-166` (group 0 in `enable_cpu_experts`)
- Test: create `test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py`

**Interfaces:**
- Consumes: `NumaNodeDistributor::host_nodes`, `NumaGroup::cpu`, `GroupRow` (Task 6); `CpuExpertConfig::engine` (Task 2).
- Produces (C++): `CopyJob::group` (`int`) and `CopyJob::groups` (`uint32_t`, bit g: group g sends a part of this record); `CopyEngine(std::unique_ptr<CopyBackend>, int64_t rows, int groups, int64_t spin_ns, Owner*, std::string prefix, std::string thread_name)`; `CopyEngine::submit(int group, const CopyJob&)`; `CopyEngine::set_cpu(int group, CpuExpertEngine*)`; `CopyEngine::stall() -> uint64_t` (`seq << 32 | group << 8 | reason`, reason `kStallCpu = 1`, `kStallNoPart = 2`, 0 none); `RamTier::enable_cpu_experts(int group, CpuExpertConfig, std::vector<int64_t> split)`, `set_cpu_split(int group, const int64_t*, int64_t)`, `cpu_stats(int group, int64_t*)`, `calibrate_cpu_split(int group, int64_t row, ...)`, `copy_stall() -> std::string`; `CalibrationSetup::first_slot`; FFI `expert_stream_enable_cpu_experts(handle, group, forward, engine, split, cores, x_rows, out_rows, hidden, parts, threads, spin_ns, keep_warm, keep_warm_ns)` with `out_rows` `[rows, >= Wire::kNodes * parts * hidden]` and group g's parts at `2g, 2g + 1`; `expert_stream_set_cpu_split(handle, group, split)`, `expert_stream_cpu_stats(handle, group, out)`, `expert_stream_calibrate_cpu_split(handle, group, row, device, reps, scratch, scratch_bytes, timeout_ns, out)`; test export `expert_stream_inject_group_stall(handle, group, ns)`.
- Produces (Python): `ExpertStreamHost.enable_cpu_experts(forward, split, cores, x_rows, out_rows, *, threads, group=0, engine=0, spin_us=50_000, keep_warm=0, keep_warm_us=0)` with `out_rows` `[rows, 2 * nodes, hidden]` (or `[rows, hidden]` at one node); `set_cpu_split(split, group=0)`; `cpu_stats(group=0)`; `calibrate_cpu_split(row, *, device, reps, scratch, timeout_s=1.0, group=0)`; `inject_group_stall(group: int, seconds: float)` (instrumented only).

- [ ] **Step 1: Write the failing test** `test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py`:

```python
"""The completion side with two NUMA groups (spec 2026-10-03-numa-node-distributor-design, Part 3 "Combiner" and
"Errors"; Testing 3): each group's CPU lanes run on its own engine into its own output parts, the copy thread stores
CopyDone only when every group with a host lane is done, and a stalled group is named by the copy-wait abort.

The Python fake forwards run on the CPU expert threads and need the GIL, so every wait polls from Python."""

import ctypes
import os
import subprocess
import sys
import textwrap
import threading
import time

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertForward
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROW, ROWS, HIDDEN, HANDLE, DST_ROWS, EXPERTS = 1, 2, 8, 7, 6, 8
HALVES = [[(0, 10)] * 2, [(10, 20)] * 2]
ONE_OF_TWO = [0, 0, 1] + [0] * 6  # of n = 2 eligible lanes, 1 on the CPU
NONE = [0] * 9


class FakeForward:
    """out[j] = j + sum_i weights[i] * (slots[i] + 1), or that sum added when accumulating. Records each call's slots
    and engine; with a ``gate``, holds every call until it is set."""

    def __init__(self, gate=None):
        self.calls, self.gate = [], gate
        self.c = CpuExpertForward(self._run)

    def _run(self, call):
        c = call.contents
        if self.gate is not None:
            self.gate.wait()
        slots = [c.slots[i] for i in range(c.k)]
        total = sum(c.weights[i] * (slots[i] + 1) for i in range(c.k))
        for j in range(HIDDEN):
            c.out[j] = (c.out[j] if c.accumulate else j) + total
        self.calls.append((slots, c.engine))
        return 0

    @property
    def address(self) -> int:
        return ctypes.cast(self.c, ctypes.c_void_p).value


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _host(tmp_path, forwards, *, split, arm=True):
    s = ram_miss_setup(tmp_path, capacity=20, experts=EXPERTS, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False, wire=wire_layout(8, 2))
    host = ExpertStreamHost(
        s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), variant="instr",
        node_ranges=HALVES,
    )
    host.reserve_staging(2)
    host.enable_copy_engine(-1, spin_us=200)
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    if arm:
        host.arm_copy_engine()
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 4, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))
    for group, forward in enumerate(forwards):
        # "native": the instrumented build's fake forward, which a blocking call (calibration) cannot deadlock on
        address = host.test_forward_address(1000) if forward == "native" else forward.address
        host.enable_cpu_experts(
            address, split[group], cores[2 * group : 2 * group + 2], x_rows, out_rows, threads=2, spin_us=200,
            group=group, engine=group + 1,
        )
    host.set_cpu_layer(ROW, HANDLE)
    sim = ChainSim(host, page, s.slabs)
    req = sim.post(ROW, [0, 1, 2, 3])  # make all four resident: two misses per node
    assert host.pump() == 1 and sim.wait_served(req)
    return s, host, sim, out_rows


def _cpu_post(sim):
    return sim.post(ROW, [0, 1, 2, 3], captured=True, cpu_on=True, dst=[0, 1, 2, 3])


def test_each_groups_cpu_lanes_go_to_its_own_engine_and_part(tmp_path):
    a, b = FakeForward(), FakeForward()
    s, host, sim, out_rows = _host(tmp_path, [a, b], split=[ONE_OF_TWO, ONE_OF_TWO])
    try:
        req = _cpu_post(sim)
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.HIT_COPY, LaneKind.HIT_CPU, LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: len(a.calls) == 1 and len(b.calls) == 1)
        assert sim.copy_wait(req)
        slot2, slot3 = host.mapping(ROW)[2], host.mapping(ROW)[3]
        assert 0 <= slot2 < 10 <= slot3 < 20
        assert a.calls == [([slot2], 1)] and b.calls == [([slot3], 2)]
        assert out_rows[ROW, 0].tolist() == [j + slot2 + 1 for j in range(HIDDEN)], "group 0's hits: part 0"
        assert out_rows[ROW, 2].tolist() == [j + slot3 + 1 for j in range(HIDDEN)], "group 1's hits: part 2"
        assert not out_rows[ROW, 1].any() and not out_rows[ROW, 3].any()
        assert host.cpu_stats(group=0)["jobs"] == 1 and host.cpu_stats(group=1)["jobs"] == 1
    finally:
        host.stop()


def test_copydone_waits_for_every_groups_cpu_engine(tmp_path):
    gate = threading.Event()
    a, b = FakeForward(), FakeForward(gate)
    s, host, sim, out_rows = _host(tmp_path, [a, b], split=[ONE_OF_TWO, ONE_OF_TWO])
    try:
        req = _cpu_post(sim)
        assert host.pump() == 1
        assert _wait(lambda: len(a.calls) == 1)
        time.sleep(0.05)
        assert sim.copy_done(req) != req.gen, "CopyDone before group 1's CPU job finished"
        gate.set()
        assert sim.copy_wait(req)
    finally:
        gate.set()
        host.stop()


def test_a_zero_split_keeps_a_node_off_the_cpu_and_copydone_off_its_engine(tmp_path):
    gate = threading.Event()
    a, b = FakeForward(), FakeForward(gate)
    s, host, sim, out_rows = _host(tmp_path, [a, b], split=[ONE_OF_TWO, NONE])
    try:
        req = _cpu_post(sim)
        assert req.kinds == [LaneKind.HIT_COPY, LaneKind.HIT_COPY, LaneKind.HIT_CPU, LaneKind.HIT_COPY]
        assert host.pump() == 1
        assert sim.copy_wait(req), "group 1 has no CPU lane: CopyDone must not wait on its engine"
        assert b.calls == []
    finally:
        gate.set()
        host.stop()


def test_each_groups_split_is_its_own_node_table(tmp_path):
    s, host, sim, out_rows = _host(tmp_path, [FakeForward(), FakeForward()], split=[ONE_OF_TWO, NONE])
    try:
        host.set_cpu_split([0, 1] + [0] * 7, group=1)
        assert sim.split() == ONE_OF_TWO + [0, 1] + [0] * 7
    finally:
        host.stop()


def test_calibration_runs_on_its_groups_engine(tmp_path):
    s, host, sim, out_rows = _host(tmp_path, ["native", "native"], split=[NONE, NONE], arm=False)
    try:
        scratch = torch.empty(8 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
        grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch, group=1)
        assert grid.numel() > 0
        assert host.cpu_stats(group=1)["jobs"] > 0 and host.cpu_stats(group=0)["jobs"] == 0
    finally:
        host.stop()


_SCRIPT = """
import ctypes, os, pathlib, sys, time, torch
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertForward
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
ROW = 1
s = ram_miss_setup(pathlib.Path(sys.argv[1]), capacity=20, experts=8, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
page = new_page(pin=False, wire=wire_layout(8, 2))
host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 8), -1, dtype=torch.int32), variant="instr",
                        node_ranges=[[(0, 10)] * 2, [(10, 20)] * 2])
host.reserve_staging(2)
host.enable_copy_engine(-1, spin_us=200, wait_timeout_ms=200)
dst = {n: torch.zeros((6,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
host.set_copy_table(ROW, torch.tensor([[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()]
                                       for n, t in s.slabs[ROW].items()], dtype=torch.int64), 6)
host.arm_copy_engine()
x_rows, out_rows = torch.zeros((2, 16), dtype=torch.uint8), torch.zeros((2, 4, 8), dtype=torch.float32)
stuck = CpuExpertForward(lambda call: time.sleep(1000) or 0)
cores = sorted(os.sched_getaffinity(0))
split = [0, 0, 1] + [0] * 6
host.enable_cpu_experts(host.test_forward_address(0), split, cores[0:2], x_rows, out_rows, threads=2, spin_us=200,
                        group=0)
host.enable_cpu_experts(ctypes.cast(stuck, ctypes.c_void_p).value, split, cores[2:4], x_rows, out_rows, threads=2,
                        spin_us=200, group=1)
host.set_cpu_layer(ROW, 7)
sim = ChainSim(host, page, s.slabs)
req = sim.post(ROW, [0, 1, 2, 3])
assert host.pump() == 1 and sim.wait_served(req)
"""


def _run(tmp_path, body):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT) + textwrap.dedent(body), str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )


def test_a_group_whose_cpu_job_stalls_is_named_by_the_copy_wait_abort(tmp_path):
    """Review Focus 2: group 0's CPU lane completes, group 1's forward never returns. CopyDone is never stored, and
    the 200 ms copy-wait abort names group 1, not group 0."""
    result = _run(tmp_path, """
        req = sim.post(ROW, [0, 1, 2, 3], captured=True, cpu_on=True, dst=[0, 1, 2, 3])
        assert host.pump() == 1
        host.start_thread(fatal_wait_s=60.0)
        sim.close_copy_gate(req)
        time.sleep(3.0)
        print("reached")
    """)
    assert_aborted(result, "a copy wait held the decode stream")
    assert "group 1: its CPU job" in result.stderr, result.stderr[-2000:]


def test_a_group_that_never_sends_its_part_is_named_by_the_copy_wait_abort(tmp_path):
    """Review Focus 2: both nodes have a copy-engine lane; group 1's thread stalls before it reads the record, so its
    part never reaches the copy thread. Group 0's part alone must not store CopyDone."""
    result = _run(tmp_path, """
        host.start_thread(fatal_wait_s=60.0)
        host.inject_group_stall(1, 10.0)
        req = sim.post(ROW, [0, 1], captured=True, dst=[0, 1])
        sim.close_copy_gate(req)
        time.sleep(3.0)
        print("reached")
    """)
    assert_aborted(result, "a copy wait held the decode stream")
    assert "group 1: sent no part" in result.stderr, result.stderr[-2000:]
```

- [ ] **Step 2: Run it to verify it fails.** Commit the test alone (`git add test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py`; `git diff --cached --stat`; message `test(expert-stream): CopyDone over every NUMA group; a stalled group is named (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py`.
Expected: FAIL: `enable_copy_engine` raises "the copy engine and CPU experts with several NUMA groups need the combiner's completion side" (Task 6's guard).

- [ ] **Step 3: Implement.**

`copy_engine.h`, `CopyJob` gains, after `late_seq`:

```cpp
  int group = 0;        // the NUMA group whose lanes these are: its CPU engine's done() completes cpu_seq/late_seq
  uint32_t groups = 1;  // bit g: group g sends a part of this record; CopyDone waits for every one
```

and its comment's last sentence becomes: "A record's host lanes arrive as one job per NUMA group that has any (`groups`); CopyDone covers every host lane once every part is done."

`CopyEngine`: the constructor takes `int groups` after `rows` and makes `jobs_` one ring per group; `cpu_` becomes `std::array<std::atomic<CpuExpertEngine*>, Wire::kNodes> cpu_{}`; `submitted_` becomes `std::array<std::atomic<uint64_t>, Wire::kNodes> submitted_{}` (each written by its group's thread only). New members:

```cpp
  // The parts of each ring index's record seen so far: the copy thread's only.
  struct Assembly {
    uint64_t gen = 0;
    uint32_t got = 0;
  };
  std::array<Assembly, Wire::kDemandRecords> assembly_{};
  // Why CopyDone is not stored, for the watchdog: seq << 32 | group << 8 | reason; 0 when nothing waits. The copy
  // thread's, read relaxed by the watchdog.
  std::atomic<uint64_t> stall_{0};
  std::vector<std::unique_ptr<SpscRing<CopyJob, kCopyRing>>> jobs_;  // one per group: each group pushes its own
```

(`kStallCpu = 1`, `kStallNoPart = 2` as public `static constexpr uint32_t` members, which `RamTier::copy_stall` reads.) `submit(int group, const CopyJob& job)` pushes to `jobs_[group]`, bumps `submitted_[group]`, then the existing fence and wake. `idle()` compares `finished_` with the sum of `submitted_`. `set_cpu(int group, CpuExpertEngine* cpu)` stores `cpu_[group]`. `uint64_t stall() const { return stall_.load(std::memory_order_relaxed); }`. In `issue`, the "carries CPU lanes with no CPU expert engine" test reads `cpu_[job.group]`. In `run`, the pop loop drains every group's ring (`for (auto& ring : jobs_) while (ring->pop(&job)) {...}`), and the completion loop becomes:

```cpp
      while (!in_flight.empty() && broken_ == 0) {
        const CopyJob& head = in_flight.front();
        const int state = head.token == kNoToken ? CopyBackend::kDone : backend_->query(head.token);
        if (state == CopyBackend::kPending) break;
        if (state != CopyBackend::kDone) {
          broken_ = state;
          break;
        }
        // Set before the service threads started.
        const CpuExpertEngine* cpu = cpu_[head.group].load(std::memory_order_relaxed);
        const uint32_t pending = head.cpu_mask != 0 && !cpu->done(head.cpu_seq) ? head.cpu_seq
                                 : head.late_cpu > 0 && !cpu->done(head.late_seq) ? head.late_seq
                                                                                   : 0;
        if (pending != 0) {
          stall_.store(static_cast<uint64_t>(pending) << 32 | static_cast<uint64_t>(head.group) << 8 | kStallCpu,
                       std::memory_order_relaxed);
          break;
        }
        const CopyJob done = in_flight.front();
        in_flight.pop_front();
        record_latency(done);
        Assembly& record = assembly_[done.idx];
        if (record.gen != done.gen) record = Assembly{done.gen, 0};
        record.got |= 1u << done.group;
        if (record.got == done.groups) {
          owner_->copy_completed(done);
          stall_.store(0, std::memory_order_relaxed);
        } else {
          const int missing = __builtin_ctz(done.groups & ~record.got);
          stall_.store(static_cast<uint64_t>(missing) << 8 | kStallNoPart, std::memory_order_relaxed);
        }
        finish();
        progressed = true;
      }
```

(at one group every job has `groups == 1`, so each completes its record at once, exactly as before.)

`ram_tier.h`:

- Delete Task 6's guard in `enable_copy_engine` and `enable_cpu_experts`. `enable_copy_engine` passes `dist_.size()` to the engine.
- `enable_cpu_experts(int g, CpuExpertConfig config, std::vector<int64_t> split)`: today's body on `dist_.group(g).cpu`, `store_split(g, ...)`, the thread name `Layout::kName + "-cpu-exp"` plus the group digit when `Wire::kNodes > 1`, and `copy_engine_->set_cpu(g, engine.get())`; refuses `g` outside `0..groups()-1` and a second enable of the same group.
- `set_cpu_layer(row, handle)` sets the layer on every group's engine that exists (the kernel's layer handle addresses the whole slab, so one handle serves every group); throws when none exists.
- `set_cpu_split(int g, const int64_t* split, int64_t count)`, `cpu_stats(int g, int64_t* out)`; `cpu_cores()` returns every group's engine cores (for `check_dedicated_core`).
- `store_split(int g, ...)` writes `lease_ + Wire::kSplit + g * Wire::kSplitStride`.
- `calibrate_cpu_split(int g, int64_t row, ...)`: `s.cpu = dist_.group(g).cpu.get()`, `s.first_slot = dist_.group(g).rows[row].lo`, and the capacity check reads the group's range: `"it needs " + std::to_string(kCalibLanes) + " RAM slots of group " + std::to_string(g) + " in row ..."` when `hi - lo < kCalibLanes`.
- `submit_host_lanes(Group& group, ...)`: `job.group = group.index; job.groups = NumaNodeDistributor<Source>::host_nodes(request);` before `copy_engine_->submit(group.index, job);`, and `cpu_` is `group.cpu`.
- `copy_stall()`:

```cpp
  // The copy thread's reason for withholding CopyDone, for the watchdog's abort: "; group G: its CPU job S is not
  // done" or "; group G: sent no part". Empty at one group, whose abort line stays as it was, or when nothing waits.
  std::string copy_stall() const {
    if (Wire::kNodes == 1 || copy_engine_ == nullptr) return "";
    const uint64_t word = copy_engine_->stall();
    const std::string group = "; group " + std::to_string(word >> 8 & 0xFF);
    switch (word & 0xFF) {
      case Engine::kStallCpu:
        return group + ": its CPU job " + std::to_string(word >> 32) + " is not done";
      case Engine::kStallNoPart:
        return group + ": sent no part";
      default:
        return "";
    }
  }
```

- `TierFaults` gains `std::array<std::atomic<int64_t>, Wire::kNodes> group_stall_ns{};` and `void inject_group_stall(int g, int64_t ns) requires(Build::kFaults)`; `pump_demand(g)`, once it has seen a posted record and before it reads it: `if constexpr (Build::kFaults) { if (const int64_t ns = faults_.group_stall_ns[g].exchange(0)) fault_delay(ns); }`.

`split_calibration.h`: `CalibrationSetup` gains `int64_t first_slot = 0;  // the first of the kCalibLanes host slots measured (a NUMA group's lowest)`; `calibration_run` uses `job.slots[i] = static_cast<int32_t>(s.first_slot + i)` and `entry.src + static_cast<uint64_t>(s.first_slot + k + j) * bytes`; its comments say "slots first_slot.. first_slot + kCalibLanes - 1".

`ram_thread.h` `watch()`: the abort line becomes

```cpp
        const std::string why = stuck ? "" : tier_->copy_stall();
        std::fprintf(
            stderr,
            "FATAL %s%s for %.1f s%s; aborting instead of hanging decode\n",
            error_prefix<typename Tier::Layout>().c_str(),
            stuck ? "a request stayed in service" : "a copy wait held the decode stream",
            static_cast<double>(stuck ? fatal_wait_ns_ : tier_->copy_wait_timeout_ns()) / 1e9,
            why.c_str());
```

(the stuck branch keeps Task 6's group prefix.)

`ffi_exports.h`: `enable_cpu_experts(int64_t handle, int64_t group, int64_t forward, int64_t engine, TensorView split, ...)`; `out_rows` must be at least `Wire::kNodes * parts * hidden` wide (`"out_rows is narrower than every group's parts"`), and `config.out_base = static_cast<uint8_t*>(out_rows.data_ptr()) + group * parts * hidden * sizeof(float)`; `config.out_stride` stays the full row. `set_cpu_split(handle, group, split)`, `cpu_stats(handle, group, out)`, `calibrate_cpu_split(handle, group, row, ...)` pass the group through. `ffi_test_exports.h`: `inject_group_stall(int64_t handle, int64_t group, int64_t ns)` (test-only like `inject`) and its export line.

`expert_stream_transport.py`: `enable_cpu_experts` takes `group: int = 0`; its `out_rows` check accepts `[rows, hidden]` only at one node and otherwise `[rows, 2 * self.nodes, hidden]` (`"out_rows must be a contiguous host float32 [rows, hidden] at one node or [rows, 2 * nodes, hidden] tensor"`), computes `parts = 1 if out_rows.dim() == 2 else 2`, and passes `out_rows.view(out_rows.shape[0], -1)` and `group`. `set_cpu_split(split, group=0)`, `cpu_stats(group=0)` and `calibrate_cpu_split(..., group=0)` pass `group`. New:

```python
    def inject_group_stall(self, group: int, seconds: float) -> None:
        """Test only: NUMA group ``group``'s service thread sleeps ``seconds`` before it reads its next record."""
        _refuse_test_only("inject_group_stall", self.variant)
        self._module.expert_stream_inject_group_stall(self.handle, int(group), int(seconds * 1e9))
```

`bench/src/stack.h`: `tier_->enable_cpu_experts(0, std::move(cpu), ...)`.

- [ ] **Step 4: Run.** Commit:

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h
git add -p python/sglang/kernels/ops/moe/expert_stream_transport.py
git diff --cached --stat
git commit -m "feat(expert-stream): CopyDone waits on every NUMA group's part and CPU engine; the copy-wait abort names the group

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py`. Expected: PASS.

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU`, `RUN_GPU SUITE_GPU`, `BENCH nlane-bench`. Expected: Baseline counts plus Tasks 1-7's tests; `test_a_copy_wait_held_past_its_timeout_aborts_the_process` still matches its one-node line; `test_exl3_copy_engine_cuda.py` and `test_cpu_split_calibration_cuda.py` unchanged; `BUILD=0`, `EXIT=0`.

---

### Task 8: The NUMA groups in `Exl3RamMissService`: ThreadingConfig, slot ranges, a CPU service per group, the copy thread's cores

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py:34-55` (imports), `:865-1029` (`ensure_started`), `:1031-1086` (`_start_cpu_experts`), `:1171-1186` (`ExpertStreamDevice(...)` gets `nodes`)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py` (`CpuExpertService` gains `group` and `shared`; new `CpuExpertGroups`; `cpu_expert_cores` deleted), `policy.py:19-29` (`parse_core_list` deleted)
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py:227-230` (the `SGLANG_DSV41_CPU_EXPERTS_CORES` requirement deleted)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h` (`CopyEngine` takes `std::vector<int> cpus` and pins its thread in `run()`), `ram_tier.h` (`enable_copy_engine(..., std::vector<int> cpus)`), `ffi_exports.h:324` (`enable_copy_engine(handle, device, spin_ns, wait_timeout_ns, cpus)`), `bench/src/stack.h` (passes `{config_.copy_cpu}` when set, `{}` otherwise)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py:1570-1583` (`enable_copy_engine(..., cpus=())`)
- Modify (tests whose API changes): `test/registered/unit/kernels/test_cpu_expert_pool.py` (`FakeHost`: `nodes`, `node_ranges`, `group` keywords; `test_parse_core_list` moves to `test_threading_config.py` as `parse_cpu_list`), `test/registered/unit/test_expert_stream_requirements_exl3.py:344-363` (`SGLANG_DSV41_CPU_EXPERTS_CORES` leaves `CPU_EXPERTS_ENV`)
- Test: create `test/registered/unit/layers/moe/test_exl3_ram_miss_service_numa.py`; add to `test/registered/unit/kernels/test_cpu_expert_pool.py` and `test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py`

**Interfaces:**
- Consumes: `ThreadingConfig`, `NodePlan` (Task 1); `slot_nodes`, `group_ranges` (Task 5); `ExpertStreamHost(node_ranges=, sq_thread_cpus=)`, `start_thread(cpu_core=[...])`, `group_counters` (Task 6); `enable_cpu_experts(group=, engine=)`, `set_cpu_split(group=)`, `cpu_stats(group=)`, `calibrate_cpu_split(group=)` (Task 7); `ExpertStreamDevice(nodes=)` (Task 4).
- Produces: `Exl3RamMissService.numa: ThreadingConfig`; `CpuExpertService(host, trait, slabs_by_row, *, hidden, cores, threads, split, pin=True, group=0, shared: Optional[CpuExpertService] = None)` (with `shared`, it uses that service's `x_rows`, `out_rows` and `handles`); `CpuExpertGroups(host, trait, slabs_by_row, *, hidden, plans: Sequence[NodePlan], split, pin=True)` with `services`, `x_rows`, `out_rows`, `register`, `registered`, `attach_device`, `retune() -> list`, `log_stats() -> list`, `calibrate(device) -> list`; `ExpertStreamHost.enable_copy_engine(device, *, spin_us=5000, wait_timeout_ms=2000, cpus: Sequence[int] = ())`.

- [ ] **Step 1: Write the failing tests.**

`test/registered/unit/layers/moe/test_exl3_ram_miss_service_numa.py`:

```python
"""Exl3RamMissService with two NUMA groups (spec 2026-10-03-numa-node-distributor-design, Part 3): ThreadingConfig's
plans reach the service threads, the tier's slot ranges reach the host, and a demand or an eager admission lands
each expert in its home group's slots. The topology and the page bindings are stood in for (Tasks 1 and 5 test
them); the thread runs, no device."""

import contextlib
import faulthandler
import os

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan, ThreadingConfig
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_fake_exl3 import write_fake_exl3
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, paused, service_row_images

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 8


@pytest.fixture
def two_groups(tmp_path, monkeypatch):
    faulthandler.dump_traceback_later(120, exit=True)
    cores = sorted(os.sched_getaffinity(0))
    plans = (
        NodePlan(group=0, node=0, ram=cores[0], cpu=(), sq=None, busy_poll=False),
        NodePlan(group=1, node=1, ram=cores[1], cpu=(), sq=None, busy_poll=False),
    )
    monkeypatch.setattr(
        ThreadingConfig, "from_env", classmethod(lambda cls, **kw: ThreadingConfig(plans, (cores[2],), 0))
    )
    monkeypatch.setattr(module, "slot_nodes", lambda slabs, capacity: [0] * (capacity // 2) + [1] * (capacity // 2))
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS, hidden=ROW_IMAGE_DIM, inter=ROW_IMAGE_DIM)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    caches = {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(service_row_images(tmp_path))
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
            for layer_id in range(LAYERS):
                layer = torch.nn.Module()
                layer.layer_id = layer_id
                fmt = Exl3ExpertFormat(layout, layer_id, source_root=str(tmp_path))
                streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
                layer._nvfp4_expert_streamer = streamer
                caches[layer_id] = ExpertPinnedHostCache(
                    streamer, CAPACITY, device="cpu", **fmt.pinned_tier_options(layer)
                )
        service = module.Exl3RamMissService.get()
        service.plan_gather_width(1)
        yield service, caches, cores
        service.shutdown()
    module.Exl3RamMissService._instance = None
    faulthandler.cancel_dump_traceback_later()


def test_each_group_runs_on_its_plans_core_over_its_half_of_every_row(two_groups, capfd):
    service, caches, cores = two_groups
    caches[1].ensure_rows(torch.tensor([4, 3]))
    host = service.host
    assert host.nodes == 2 and service.numa.nodes == 2
    assert host.node_ranges == [[(0, 4)] * LAYERS, [(4, 8)] * LAYERS]
    assert host.group_counters(0)["spin_cpu"] == cores[0] and host.group_counters(1)["spin_cpu"] == cores[1]
    host.stop()
    assert "exl3 RAM miss group 1 counters" in capfd.readouterr().err


def test_an_eager_admission_lands_each_expert_in_its_home_groups_slots(two_groups):
    service, caches, cores = two_groups
    caches[1].ensure_rows(torch.tensor([4, 3]))
    row = service.row_of(1)
    with paused(service.host):
        mapping = service.host.mapping(row)
    assert 0 <= mapping[4] < 4, "expert 4 is node 0's"
    assert 4 <= mapping[3] < 8, "expert 3 is node 1's"


def test_a_demand_on_each_node_is_served_by_its_group(two_groups):
    service, caches, cores = two_groups
    caches[1].ensure_rows(torch.tensor([0]))  # starts the service
    sim = ChainSim(service.host, service.page, {})
    row = service.row_of(1)
    type(service.host).pause(service.host, 10.0)
    try:
        sim.sync_bulk()
    finally:
        type(service.host).resume(service.host)
    before = [service.host.group_counters(g)["rows_read"] for g in (0, 1)]
    req = sim.post(row, [1, 2])  # one miss per node: expert 2 is node 0's, expert 1 node 1's
    assert sim.wait_served(req, timeout_s=10.0) and sim.wait_handled(req, timeout_s=10.0)
    after = [service.host.group_counters(g)["rows_read"] for g in (0, 1)]
    assert [a - b for a, b in zip(after, before)] == [1, 1]
```

In `test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py`:

```python
def test_the_copy_thread_runs_on_the_cpus_it_is_given(tmp_path):
    """ThreadingConfig puts the copy thread on the GPU's node; enable_copy_engine's cpus are its affinity."""
    s, page, host, sim = _host(tmp_path)
    try:
        cpu = sorted(os.sched_getaffinity(0))[-1]
        host.enable_copy_engine(-1, cpus=[cpu])
        tids = [
            int(tid) for tid in os.listdir("/proc/self/task")
            if open(f"/proc/self/task/{tid}/comm").read().strip() == "exl3-copy-eng"
        ]
        assert len(tids) == 1 and os.sched_getaffinity(tids[0]) == {cpu}
    finally:
        host.stop()
```

(add `import os` there if missing.)

In `test/registered/unit/kernels/test_cpu_expert_pool.py`, delete `test_parse_core_list` and the `parse_core_list` import (Task 1's `test_the_shared_core_checks` asserts the same input on `parse_cpu_list`); `FakeHost.__init__(self, lanes=8, nodes=1)` sets `self.nodes = nodes`, `self.enables = []`; `enable_cpu_experts(..., *, threads, group=0, engine=0, keep_warm=0, keep_warm_us=0)` appends `(group, ...)` to `self.enables` and keeps `self.enabled` as the last tuple with `group` appended; `set_cpu_split(self, split, group=0)` appends `list(split)` to `self.splits` and `(group, list(split))` to `self.group_splits = []`; `cpu_stats(self, group=0)`; `calibrate_cpu_split(..., timeout_s=1.0, group=0)` records `group` last in each `self.calibrations` entry. `test_service_registers_a_row_once_...` asserts `host.enabled == (0xF00D, [0] * (lease.wire_layout(8).lanes + 1), [4, 5, 6], (2, 16), (2, 2, 8), 2, 0xE1, 0)`; the calibration tests' `host.calibrations` tuples gain a trailing `0`. Add:

```python
def test_cpu_expert_groups_run_one_engine_per_node_and_register_each_layer_once():
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertGroups
    from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan

    host, trait = FakeHost(nodes=2), FakeServiceTrait()
    plans = [
        NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True),
        NodePlan(group=1, node=1, ram=35, cpu=(18, 19, 20), sq=None, busy_poll=True),
    ]
    split = [0] * (host.wire.lanes + 1)
    groups = CpuExpertGroups(host, trait, {r: _fake_slabs() for r in range(2)}, hidden=8, plans=plans, split=split, pin=False)
    assert [(e[0], e[3], e[6]) for e in host.enables] == [(0, [8, 9], 2), (1, [18, 19, 20], 3)]
    assert tuple(groups.out_rows.shape) == (2, 4, 8), "two output parts per group"
    assert groups.services[1].out_rows is groups.services[0].out_rows
    groups.register(1, 10.0)
    groups.register(1, 10.0)
    assert [e for e in trait.events if e[0] == "register"] == [("register", 10.0)], "one kernel layer serves both groups"
    assert host.layers == {1: 100} and groups.registered(1)
```

(`e[3]` is the cores argument and `e[6]` threads in the `(group, forward, split, cores, x_shape, out_shape, threads, engine)` tuple `enable_cpu_experts` appends; write the append in that order.)

In `test/registered/unit/test_expert_stream_requirements_exl3.py`, delete `SGLANG_DSV41_CPU_EXPERTS_CORES="18-29",` from `CPU_EXPERTS_ENV` and the `"SGLANG_DSV41_CPU_EXPERTS_CORES": ""` entry of the `off` mapping.

- [ ] **Step 2: Run them to verify they fail.** Commit the tests alone (`git add` the five test files; `git diff --cached --stat`; message `test(numa): the service runs a NUMA group per node; the copy thread's cores (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/layers/moe/test_exl3_ram_miss_service_numa.py test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/test_expert_stream_requirements_exl3.py`.
Expected: FAIL: `AttributeError: <module 'sglang.srt.layers.moe.exl3_ram_miss'> has no attribute 'slot_nodes'`; `TypeError: enable_copy_engine() got an unexpected keyword argument 'cpus'`; `ImportError: cannot import name 'CpuExpertGroups'`; the launch gate still requires `SGLANG_DSV41_CPU_EXPERTS_CORES`.

- [ ] **Step 3: Implement.**

`copy_engine.h`: the constructor takes `std::vector<int> cpus` last (stored as `cpus_`); `run()`, before `backend_->init()`:

```cpp
    std::string error;
    if (!cpus_.empty()) {
      // ThreadingConfig.copy_cpus: the GPU's node. A thread created later inherits the enabling caller's affinity.
      cpu_set_t set;
      CPU_ZERO(&set);
      for (const int cpu : cpus_)
        CPU_SET(cpu, &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) error = "cannot pin the copy thread to its cores";
    }
    if (error.empty()) error = backend_->init();
```

(`init_error_ = error;` as before.) `RamTier::enable_copy_engine(int64_t device, int64_t spin_ns, int64_t wait_timeout_ns, std::vector<int> cpus)` passes them through. `ffi_exports.h` `enable_copy_engine(int64_t handle, int64_t device, int64_t spin_ns, int64_t wait_timeout_ns, TensorView cpus)` (int64 `[n]`, CPU; empty inherits). `stack.h`: `tier_->enable_copy_engine(-1, kCopySpinNs, config_.wait_timeout_ns, config_.copy_cpu >= 0 ? std::vector<int>{config_.copy_cpu} : std::vector<int>{})` (the `PinScope` around it stays: the watchdog still inherits it). `expert_stream_transport.py`:

```python
    def enable_copy_engine(
        self, device: int, *, spin_us: int = 5000, wait_timeout_ms: int = 2000, cpus: Sequence[int] = ()
    ) -> None:
        """...existing text... ``cpus`` is the copy thread's affinity (ThreadingConfig.copy_cpus: the GPU's node);
        empty inherits the caller's."""
        self._module.expert_stream_enable_copy_engine(
            self.handle, int(device), int(spin_us * 1e3), int(wait_timeout_ms * 1e6),
            torch.tensor([int(c) for c in cpus], dtype=torch.int64),
        )
```

`service.py`: delete `cpu_expert_cores` and the `parse_core_list` import (and `parse_core_list` from `policy.py`). `CpuExpertService.__init__` gains `group: int = 0, shared: Optional["CpuExpertService"] = None`; it sets `self.group = group`; with `shared` it takes `self.x_rows, self.out_rows, self.handles = shared.x_rows, shared.out_rows, shared.handles` instead of allocating, and without it allocates `self.out_rows` as `(rows, 2 * host.nodes, self.hidden)`; it passes `group=self.group` to `host.enable_cpu_experts`, `host.set_cpu_split`, `host.cpu_stats` and `host.calibrate_cpu_split`; `log_stats` and `retune` log "CPU experts group %d: ..." with `self.group` first; `calibrate`'s row choice uses each registered row's slots in this group:

```python
    def _group_slots(self, row: int) -> int:
        """The slots of ``row`` this service's NUMA group holds: the row's slab rows at one group."""
        if self.host.nodes == 1:
            return self._capacity(self.slabs_by_row[row])
        lo, hi = self.host.node_ranges[self.group][row]
        return hi - lo
```

(`calibration_row({r: self._group_slots(r) for r in self.handles}, self.lanes)`). Then, at the end of the module:

```python
class CpuExpertGroups:
    """One CpuExpertService per NUMA group of the host, each on its node's cores (spec 2026-10-03, Part 3).

    The services share the pinned rows (each group writes its own two output parts of a row) and the kernel's layer
    registrations: one layer handle addresses a whole slab, so every group's engine gets the same handle. The split is
    configured, re-tuned and calibrated per group.
    """

    def __init__(self, host, trait, slabs_by_row, *, hidden: int, plans, split: Sequence[int], pin: bool = True):
        self.services: list[CpuExpertService] = []
        for plan in plans:
            self.services.append(
                CpuExpertService(
                    host, trait, slabs_by_row, hidden=hidden, cores=plan.cpu, threads=plan.threads, split=split,
                    pin=pin, group=plan.group, shared=self.services[0] if self.services else None,
                )
            )
        self.x_rows, self.out_rows = self.services[0].x_rows, self.services[0].out_rows

    def registered(self, row: int) -> bool:
        return self.services[0].registered(row)

    def register(self, row: int, act_limit: Optional[float]) -> None:
        """Register ``row`` once; the host gives its handle to every group's engine."""
        self.services[0].register(row, act_limit)

    def attach_device(self, device_side) -> None:
        self.services[0].attach_device(device_side)

    def retune(self) -> list[Optional[list[int]]]:
        return [service.retune() for service in self.services]

    def log_stats(self) -> list[dict[str, int]]:
        return [service.log_stats() for service in self.services]

    def calibrate(self, device: int) -> list[Optional[list[int]]]:
        return [service.calibrate(device) for service in self.services]
```

`CpuExpertService.register`: the host's `set_cpu_layer` sets every group's engine (Task 7), so registering through one service is enough; no change.

`exl3_ram_miss.py`: import `from sglang.srt.layers.moe.cpu_experts.threading_config import ThreadingConfig` and `from sglang.srt.layers.moe.host_numa import group_ranges, slot_nodes` at the top. In `ensure_started`, after the `host_layout()` check:

```python
        numa = ThreadingConfig.from_env(
            cpu_experts=envs.SGLANG_DSV41_CPU_EXPERTS.get(),
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        )
        for line in numa.log_lines():
            logger.info("exl3 RAM miss %s", line)
        node_ranges = None
        if numa.nodes > 1:
            # Rows in layer order, as exl3_ram_miss_tables numbers them; a slot across a seam is in no range.
            node_ranges = group_ranges(
                [slot_nodes(s.pinned_host_cache.tensors, s.pinned_host_cache.capacity) for _, s in sorted(streamers.items())],
                [plan.node for plan in numa.plans],
            )
```

`self.wire = wire_layout(self.lanes, numa.nodes)`; `ExpertStreamHost(...)` gains `node_ranges=node_ranges, sq_thread_cpus=[-1 if p.sq is None else p.sq for p in numa.plans]`; `host.enable_copy_engine(..., cpus=numa.copy_cpus)`; `self._start_cpu_experts(cfg, host, fmt, streamers, pin, numa)`; the thread start replaces the `spin_core` branch:

```python
            cores = [-1 if plan.ram is None else plan.ram for plan in numa.plans]
            host.start_thread(
                cpu_core=cores if numa.nodes > 1 else cores[0],
                busy_poll=all(plan.busy_poll for plan in numa.plans),
                fatal_wait_s=watchdog_wait_s(cfg.ram_miss_timeout_ms),
            )
```

(one node without `SGLANG_DSV41_RAM_MISS_SPIN_CORE`: `-1`, no busy-poll, as before; with it: that core, busy-poll, as before); store `self.numa = numa` with the other fields. `_start_cpu_experts(cfg, host, fmt, streamers, pin, numa)` drops `cpu_expert_cores` and returns `CpuExpertGroups(host, trait, caches, hidden=hidden.pop(), plans=numa.plans, split=configured_split(host.wire.lanes), pin=pin)`. `ExpertStreamDevice(...)` gets `nodes=self.wire.nodes`. `self.cpu_experts.retune()`, `.log_stats()` and `.calibrate(...)` keep their call sites (the container has the same names).

`expert_stream_requirements_exl3.py`: delete the `("SGLANG_DSV41_CPU_EXPERTS_CORES (a taskset list)", ...)` entry; ThreadingConfig derives the cores and refuses at start.

`ExpertStreamHost.stop()`, after its counters line (whose shape arms grep, so it stays as it is), above one group:

```python
                if self.nodes > 1:
                    for group in range(self.nodes):
                        sys.stderr.write(
                            f"exl3 RAM miss group {group} counters " + json.dumps(self.group_counters(group)) + "\n"
                        )
```

and `test_each_group_runs_on_its_plans_core_over_its_half_of_every_row` ends with `service.host.stop()` and asserts, with pytest's `capfd`, that `exl3 RAM miss group 1 counters` is in the captured stderr (add `capfd` to its arguments).

- [ ] **Step 4: Run.** Commit:

```bash
git add python/sglang/srt/layers/moe/cpu_experts/service.py python/sglang/srt/layers/moe/cpu_experts/policy.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h
git add -p python/sglang/srt/layers/moe/exl3_ram_miss.py python/sglang/kernels/ops/moe/expert_stream_transport.py
git diff --cached --stat
git commit -m "feat(numa): Exl3RamMissService runs a NUMA group per node from ThreadingConfig; a CPU service per group; the copy thread on the GPU's node

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, the Step 2 runs. Expected: PASS.

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU test/registered/unit/test_expert_stream_requirements_exl3.py`, `RUN_GPU SUITE_GPU`, `RUN_EXT SUITE_EXT`, `BENCH nlane-bench`. Expected: Baseline counts plus Tasks 1-8's tests (`test_parse_core_list` moved, not lost). `test_exl3_ram_miss_service.py` runs the one-node path through `ThreadingConfig.from_env` (no `SGLANG_MOE_PINNED_HOST_NUMA_MB`, CPU experts off: today's plan) and is unchanged.

---

### Task 9: Two groups in the C++ full-stack bench and in the CPU-forward A/B

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/CMakeLists.txt:30-34` (`EXPERT_STREAM_NODES`)
- Modify: `bench/src/device_sim.h:84-100`, `device_sim.cpp:42-150` (every node's staging list and split; misses take their home node's list)
- Modify: `bench/src/placement.h:23-31`, `placement.cpp:60-140` (one `GroupPlacement` per group; `bind_pages`; per-group node rules; `expected_threads`)
- Modify: `bench/src/stack_fixture.h:16-40`, `stack_fixture.cpp:60-185` (`kGroupSlots`, `capacity()`, `slot_of`, per-group binding, `out_stride` of every group's parts)
- Modify: `bench/src/stack.h:90-180` (`StackConfig` per group; `Stack` builds the groups)
- Modify: `bench/src/full_stack.cpp:44-130` (options), `:180-330` (`Bench`: slots, `bare_call(row, k, group)`, `validate`), `:495-560` (`main`: one kernel engine per group)
- Modify: `bench/src/self_test.cpp` (its `Placement` literal becomes one group; a two-group case when `Wire::kNodes == 2`)
- Modify: `test/manual/dsv41/exl3_cpu_forward_ab.py` (`dump --registration engines`)
- Modify: `bench/README.txt` ("Full-stack bench": the two-group flags and validation)

**Interfaces:**
- Consumes: the engine ABI (Task 2); `RamTier`'s `ranges` and per-group `enable_cpu_experts` (Tasks 6-7); `RamThread`'s per-group cores (Task 6).
- Produces: `cmake -DEXPERT_STREAM_NODES=2` builds the bench for two groups (default 1); `struct GroupPlacement { int service; std::vector<int32_t> workers; int node; }`, `Placement::groups` (one per `Wire::kNodes`); flags `--service-cpu=17,35 --cpus=8-15/18-33 --worker-node=0,1` (one entry per group, `/` between groups' CPU lists); `StackFixture::slot_of(int64_t expert) -> int32_t`, `StackFixture::capacity()`; `exl3_cpu_forward_ab.py dump --registration engines --cores A-B` (two engines of `THREADS` workers each, on the first and second half of the cores, running every case at once).

- [ ] **Step 1: Write the failing tests.**

`self_test.cpp`: `FakeCall` gains `int64_t engine;` and `fake_forward` records `call->engine`; then, after `test_stack`, add and call from `run_self_test` after `test_stack(placement, image_dir)`:

```cpp
// Two NUMA groups (Wire::kNodes == 2): each group's CPU lanes reach the forward with its own engine and only its own
// slots, and each group's service thread runs on its own CPU.
void test_two_groups(const Placement& placement, const std::filesystem::path& dir) {
  if constexpr (w::Wire::kNodes != 2) {
    return;
  } else {
    constexpr int64_t kGroup = 7;  // per group: 3 staging slots, 4 mappable
    const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
    std::vector<std::array<AlignedBuffer, kNames>> slabs(kSelfRows);
    RowSet set;
    set.layout = layout;
    set.experts = kSelfExperts;
    set.capacity = 2 * kGroup;
    for (int64_t row = 0; row < kSelfRows; ++row) {
      std::array<uint8_t*, kNames> bases{};
      for (int n = 0; n < kNames; ++n) {
        slabs[row][n] = aligned_zeroed(2 * kGroup * 512);
        bases[n] = slabs[row][n].get();
      }
      set.slabs.push_back(bases);
      const auto path = dir / ("selftest2-layer-" + std::to_string(row) + ".rows");
      write_row_image(
          path,
          layout,
          kSelfExperts,
          [&](int64_t e, uint8_t* image) {
            for (int n = 0; n < kNames; ++n)
              std::memset(image + layout.name_offsets[n], pattern(row, e, n), 512);
          },
          "");
      set.paths.push_back(path.string());
    }
    AlignedBuffer x = aligned_zeroed(kSelfRows * 2 * kSelfHidden);
    AlignedBuffer out = aligned_zeroed(kSelfRows * 4 * kSelfHidden * 4);  // two parts per group
    StackConfig config;
    config.rows = set;
    config.staging = 3;
    config.forward = &fake_forward;
    config.x_base = x.get();
    config.x_stride = 2 * kSelfHidden;
    config.out_base = out.get();
    config.out_stride = 4 * kSelfHidden * 4;
    config.hidden = kSelfHidden;
    config.copy_cpu = placement.copy;
    config.ranges = {{0, kGroup}, {kGroup, 2 * kGroup}};
    for (int g = 0; g < 2; ++g) {
      StackConfig::Group group;
      group.service_cpu = placement.groups[g].service;
      group.cores.assign(placement.groups[g].workers.begin(), placement.groups[g].workers.end());
      group.engine = g + 1;
      group.split = {0, 1, 2, 3, 4, 5, 6, 7, 8};
      config.groups.push_back(group);
    }
    fake_calls.clear();
    PinScope writer(placement.writer);
    Stack<BenchBuild> stack(std::move(config));
    DeviceSim sim(stack.page(), stack.lease(), kSelfRows, kSelfExperts);
    const int32_t four[] = {0, 1, 2, 3};
    load_experts(sim, 0, four, 3, soon());
    stack.set_cpu_layer(0, kFakeHandle);
    sim.set_row_cpu(0);
    const float ones[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const SimRequest r = sim.post(0, four, ones, /*captured=*/true, soon());
    CHECK(sim.copy_wait(r, soon()));
    {
      std::lock_guard<std::mutex> guard(fake_mutex);
      CHECK(fake_calls.size() == 2);  // one CPU-hit job per group
      for (const FakeCall& call : fake_calls) {
        const int64_t g = call.engine - 1;
        CHECK(g == 0 || g == 1);
        for (int32_t slot : call.slots)
          CHECK(slot >= g * kGroup && slot < (g + 1) * kGroup);
      }
    }
    for (int g = 0; g < 2; ++g)
      CHECK(stack.group_counters(g)[es::kSpinCpu] == placement.groups[g].service);
  }
}
```

(`Stack::group_counters(int g)` returns `std::array<int64_t, es::kCounterCount>` from the tier's `group_counters`. `production_placement()` and `test_placement()` build the new `Placement` shape, one group.)

`exl3_cpu_forward_ab.py`: `dump` accepts `--registration engines` and `--cores` (a taskset list of `2 * THREADS` cores). The test is the comparison itself (Step 4): a dump through two engines at once must be bitwise the merge-base's `table` dump.

- [ ] **Step 2: Run them to verify they fail.** Commit (`git add` the two files; `git diff --cached --stat`; message `test(bench): two NUMA groups in the self-test; the CPU-forward A/B through two engines (failing)` with the trailers). `SYNC`, `BENCH nlane-bench2 -DEXPERT_STREAM_NODES=2`. Expected: `BUILD` fails (`StackConfig` has no `ranges`, `Placement` no `groups`); `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && ... python test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa bw --registration engines --cores 18-25 --out /tmp/x.pt'` exits 2 (`invalid choice: 'engines'`).

- [ ] **Step 3: Implement.**

`CMakeLists.txt`, after the lanes block:

```cmake
set(EXPERT_STREAM_NODES 1 CACHE STRING "NUMA groups of the stack (1..8)")
if(NOT EXPERT_STREAM_NODES MATCHES "^[0-9]+$" OR EXPERT_STREAM_NODES LESS 1 OR EXPERT_STREAM_NODES GREATER 8)
  message(FATAL_ERROR "EXPERT_STREAM_NODES must be 1..8")
endif()
add_compile_definitions(SGLANG_EXPERT_STREAM_NODES=${EXPERT_STREAM_NODES})
```

`device_sim`: `constexpr int kNodes = ::sglang::expert_stream::wire::Wire::kNodes;`; `staging_` becomes `std::vector<std::array<int32_t, kNodes * kLanes>>`, `staging(row)` returns that array; `apply_pending` reads `for (int k = 0; k < kNodes * kLanes; ++k)` at `kDeltaStaging + 2 * k`; and the typing in `post` becomes ram_slot_map.type_lanes's:

```cpp
  bool hit[kLanes] = {};
  int m[kNodes] = {};
  for (int j = 0; j < count; ++j) {
    r.experts[j] = experts[j];
    const int32_t slot = ram_slot(row, experts[j]);
    if (slot >= 0) {
      r.slots[j] = slot;
      hit[j] = true;
      continue;
    }
    const int node = w::Wire::home(experts[j]);
    if (m[node] >= kLanes || staging_[row][node * kLanes + m[node]] < 0)
      throw std::runtime_error("a miss lane has no staging slot on node " + std::to_string(node));
    r.slots[j] = staging_[row][node * kLanes + m[node]++];
  }
  const bool host_lanes = captured && load_acquire<uint32_t>(lease_ + w::Wire::kCopyArmed) != 0;
  bool eligible[kLanes] = {};
  int n[kNodes] = {};
  for (int j = 0; j < count; ++j) {
    eligible[j] = host_lanes && row_cpu_[row] != 0 && hit[j];
    n[w::Wire::home(experts[j])] += eligible[j] ? 1 : 0;
  }
  int take[kNodes] = {};
  for (int node = 0; node < kNodes; ++node) {
    const auto* split = reinterpret_cast<const int32_t*>(lease_ + w::Wire::kSplit + node * w::Wire::kSplitStride);
    take[node] = n[node] > 0 ? __atomic_load_n(split + n[node], __ATOMIC_RELAXED) : 0;
  }
  bool cpu[kLanes] = {};
  for (int j = count - 1; j >= 0; --j) {
    const int node = w::Wire::home(experts[j]);
    if (take[node] > 0 && eligible[j]) {
      cpu[j] = true;
      --take[node];
    }
  }
```

`placement.h`/`.cpp`: `Placement` becomes `{ int writer; int copy; int host_node; std::vector<GroupPlacement> groups; }` with `struct GroupPlacement { int service = -1; std::vector<int32_t> workers; int node = 1; };` (comment: "one per NUMA group of the build; at one group the workers sit on `node` and the service on `host_node`, the measured cross-socket layout; above one group a group's service and workers sit on its own node"). `validate_placement` runs today's role checks over every group's service and workers (no CPU in two roles; nothing on a service's sibling) and, with `check_nodes`, requires each worker on its group's `node`, each service on `host_node` at one group and on its group's `node` above one, and writer and copy on `host_node`. `expected_threads` lists every group's service, the copy CPU twice (copy thread and watchdog) and every group's workers. Add:

```cpp
// Binds the whole pages of [address, address + bytes) to NUMA node `node`, moving any already touched
// (mbind MPOL_BIND | MPOL_MF_MOVE). The pages that hold a range's ends are left where they are. Throws on failure.
void bind_pages(void* address, size_t bytes, int node);
```

implemented with `syscall(SYS_mbind, first, last - first, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_MOVE)` over `first = round_up(address, page)` and `last = round_down(address + bytes, page)`, with `unsigned long mask = 1ul << node;` and `MPOL_BIND = 2`, `MPOL_MF_MOVE = 1 << 1` as named constants.

`stack_fixture`: `static constexpr int64_t kGroupSlots = 8;  // per NUMA group: 5 experts + 3 staging slots` replaces `kCapacity`; `int64_t capacity() const { return kGroupSlots * Wire::kNodes; }`; the slot of an expert as the tier picks it:

```cpp
  // Where the tier maps `expert`: its home group's slots, after that group's kStaging staging slots took and returned
  // the first misses (the lowest free slot each time), so a group's j-th expert is at its j-th slot.
  static int32_t slot_of(int64_t expert) {
    return static_cast<int32_t>(::sglang::expert_stream::wire::Wire::home(expert) * kGroupSlots + expert / ::sglang::expert_stream::wire::Wire::kNodes);
  }
```

The constructor takes the group nodes (`std::vector<int> group_nodes`), allocates every slab with `capacity()` rows, and above one group calls `bind_pages(base + g * kGroupSlots * row_bytes[n], kGroupSlots * row_bytes[n], group_nodes[g])` for every slab and group before `write_row_image`. `out_stride` is `2 * Wire::kNodes * hidden * sizeof(float)`. `preload_slots` copies expert e into `slot_of(e)`; `register_layer` registers `capacity()` slots.

`stack.h`: `StackConfig` replaces `service_cpu`, `threads`, `cores`, `split` with

```cpp
  struct Group {
    int service_cpu = -1;
    std::vector<int> cores;  // worker 0 first: its CPU expert thread pins itself there
    int64_t engine = 0;      // the kernel's engine on those cores
    es::CpuExpertForward forward = nullptr;  // null: the stack's forward
    std::array<int64_t, es::Wire::kLanes + 1> split{};
  };
  std::vector<Group> groups;                          // one per Wire::kNodes
  std::vector<std::pair<int64_t, int64_t>> ranges;    // group g's slots of every row
```

and the `Stack` constructor builds the tier with `ranges` repeated for every row (`{{0, capacity}}` when `ranges` is empty at one group), calls `enable_cpu_experts(g, ...)` for every group with `cpu.out_base = config_.out_base`, the group's `cores`, `threads = cores.size()`, `engine` and split, checks `check_dedicated_core` for every group's service CPU, and starts `Thread(tier_, service_cpus, ...)`. `group_counters(g)` forwards to the tier.

`full_stack.cpp`: `Options` holds `std::optional<std::string> service_cpus, cpus, worker_nodes`; `resolve_placement` splits `--service-cpu` and `--worker-node` on `,` and `--cpus` on `/` into `Wire::kNodes` groups (throws "give one entry per NUMA group (N)" otherwise); the defaults at one group are today's (`17`, `18-33`, `1`); above one group `--self-test` defaults to writer 0, copy 1, services `2,3`, workers `4/5`, and a measured run is refused without explicit flags. `main` creates one engine per group with `sglang_exl3_cpu_experts_engine_create(group.workers...)`; `stack_config` fills `groups` (split `n -> n`, every eligible lane on the CPU, as today) and `ranges` (`{g * kGroupSlots, (g + 1) * kGroupSlots}`). In `Bench`, `enter_stack_phase` checks `sim_->ram_slot(row, e) == StackFixture::slot_of(e)`; `bare_call(int64_t row, int k, int group)` forwards the experts of `experts_[k]` homed on `group` (slots `slot_of(e)`, their weights) on that group's engine with its worker count into `out_row(row) + 2 * group * hidden`, called pinned to that group's worker 0; `validate(k, via_stack)` at one group is today's (bit-exact against `reference-e{k}.bin`); above one group the bare forwards come first, since a bare forward once the stack exists would share worker 0's core with a CPU expert thread (the class doc): `main`'s validate-only flow calls `record_bare(k)` for every k before any stack call, and BM_bare is not registered above one group.

```cpp
  // Above one group: every row's bare forward of each group's experts, on that group's engine and worker 0, kept for
  // validate_groups. Before the stack exists.
  void record_bare(int k) {
    const int64_t hidden = fixture_.hidden();
    std::vector<float>& bare = bare_[k];
    bare.assign(static_cast<size_t>(rows() * es::Wire::kNodes * hidden), 0.0f);
    for (int g = 0; g < es::Wire::kNodes; ++g) {
      if (!group_has_lanes(k, g)) continue;
      PinScope caller(placement_.groups[g].workers.front());
      for (int64_t row = 0; row < rows(); ++row)
        bare_into(row, k, g, bare.data() + (row * es::Wire::kNodes + g) * hidden);
    }
  }

  // Above one group: each group's part of the stack's output equals, bit for bit, its bare forward; the parts' sum
  // matches the single-engine reference to within fp32 reassociation.
  void validate_groups(int k) {
    const int64_t hidden = fixture_.hidden();
    const std::vector<float>& bare = bare_.at(k);
    std::vector<float> stacked(static_cast<size_t>(rows() * hidden), 0.0f);
    for (int64_t row = 0; row < rows(); ++row) {
      float* out = fixture_.out_row(row);
      std::fill(out, out + 2 * es::Wire::kNodes * hidden, std::numeric_limits<float>::quiet_NaN());
      stack_call(row, k);
      for (int g = 0; g < es::Wire::kNodes; ++g) {
        if (!group_has_lanes(k, g)) continue;
        const float* part = out + 2 * g * hidden;
        if (std::memcmp(part, bare.data() + (row * es::Wire::kNodes + g) * hidden, hidden * sizeof(float)) != 0)
          throw std::runtime_error(
              "row " + std::to_string(row) + ": group " + std::to_string(g) + "'s stack part differs from its bare forward");
        for (int64_t h = 0; h < hidden; ++h)
          stacked[row * hidden + h] += part[h];
      }
    }
    check_reference_close(options_.references / ("reference-e" + std::to_string(k) + ".bin"), stacked, 1e-5f);
  }
```

where `bare_into(row, k, group, float* out)` is `bare_call`'s forward over the experts of `experts_[k]` homed on `group` (slots `slot_of(e)`, their weights, the group's engine and worker count) into `out`; `group_has_lanes(k, g)` is whether any of `experts_[k]` is homed on `g`; `bare_` is a `std::map<int, std::vector<float>>` member; and `check_reference_close` (in `stack_fixture.cpp`, beside `check_reference`) reads the reference as `compare_reference` does and throws when any `|a - b| > tol * max(1, |b|)`.

`exl3_cpu_forward_ab.py`: `--registration` accepts `engines` and `dump` takes `--cores`; with `engines` it registers the slab ABI (`register_slabs`), creates two engines with `Exl3CpuQuantTrait(ext, act_limit=limit).native_create_engine(cores[:THREADS])` and `(cores[THREADS:2 * THREADS])`, runs the whole case list on both at once (one Python thread per engine, each case through the C ABI as `test_cpu_expert_engines_exl3._forward` does, `threads=THREADS`), exits 1 naming the case when the two engines' outputs differ, and saves engine A's outputs under the same case names. Keep one process per ISA tier, as `dump` does.

`bench/README.txt`, "Full-stack bench": a paragraph on `-DEXPERT_STREAM_NODES=2`, the per-group flags, the binding of each group's slots to its node, and `validate_groups`.

- [ ] **Step 4: Run.** Commit (`git add` every file above; `git diff --cached --stat`; message `feat(bench): the full stack and the CPU-forward A/B run two NUMA groups` with the trailers). `SYNC`, then:
  1. `BENCH nlane-bench` and `BENCH nlane-bench2 -DEXPERT_STREAM_NODES=2`. Expected: `BUILD=0`, `EXIT=0` for both (the second runs `test_two_groups`).
  2. The real two-group validation (CPU only, no server running, so the cores below are idle): `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=24 OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw taskset -c 8-35,52 /data/models/slang/nvfp4-work/nlane-bench2/exl3_full_stack_prod --validate-only --writer-cpu=16 --copy-cpu=52 --service-cpu=17,35 --cpus=8-15/18-33 --worker-node=0,1 --image-dir=/data/models/slang/nvfp4-work/nlane-bench2/images 2>&1 | tail -3; echo EXIT=${PIPESTATUS[0]}'`. Expected: `EXIT=0` and the verification line.
  3. The A/B through two engines, per ISA tier: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nlane && for isa in bw avx2 scalar; do PYTHONPATH=$PWD/python EXL3_MOE_CPU_PIN=0 SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/nlane-exl3-build taskset -c 18-25 /data/models/slang/.venv/bin/python test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa $isa --registration engines --cores 18-25 --out /data/models/slang/nvfp4-work/nlane-cpu-t9/ab-$isa-engines.pt && /data/models/slang/.venv/bin/python test/manual/dsv41/exl3_cpu_forward_ab.py compare /data/models/slang/nvfp4-work/nlane-cpu-base/ab-$isa-table.pt /data/models/slang/nvfp4-work/nlane-cpu-t9/ab-$isa-engines.pt | tail -1; done; echo EXIT=$?'`. Expected: three `N/N bit-exact` lines.

- [ ] **Step 5: Existing gates.** `CPU_CHECKS` `check` with `<task>` = `t9` (the one-group bench and every ISA dump, bit-exact against the merge-base). Expected: `ALL GREEN (check)`.

---

### Task 10: End to end on divix01: the GPU chain at two nodes, locality, ms/token

**Files:**
- Modify: `test/manual/dsv41/lease_chain_rig.py:40-110` (`Chain` gains `nodes` and `node_ranges`)
- Test: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`, `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`
- Results only (no code): the ledger entries of Steps 4-7.

**Interfaces:**
- Consumes: everything above.
- Produces: `Chain(tmp_path, *, ..., nodes=1, node_ranges=None)`.

- [ ] **Step 1: Write the failing GPU tests.** In `test_exl3_lease_kernels_cuda.py`:

```python
def test_a_two_node_chain_delivers_every_lane_byte_exact(tmp_path):
    """The real post, C1, S, CW and CC against two NUMA groups: misses on both nodes land in their own node's staging
    slots, every lane's bytes arrive exactly as with one group, and each expert ends up in its home group's slots."""
    capacity = 2 * CAPACITY
    ranges = [[(0, CAPACITY)] * LAYERS, [(CAPACITY, capacity)] * LAYERS]
    c = Chain(tmp_path, capacity=capacity, nodes=2, node_ranges=ranges)
    try:
        rng = random.Random(10)
        for _ in range(16):
            row = rng.randrange(LAYERS)
            _step(c, rng.sample(range(EXPERTS), TOP_K), row)
        for row in range(LAYERS):
            for expert, slot in enumerate(c.host.mapping(row)):
                if slot >= 0:
                    assert (slot < CAPACITY) == (expert % 2 == 0), (row, expert, slot)
    finally:
        c.close()
```

In `test_exl3_cpu_lane_order_cuda.py`:

```python
def test_two_nodes_type_their_cpu_lanes_by_their_own_split_and_part(tmp_path):
    """Two NUMA groups on the real chain: each node's CPU lanes are the tail of its own eligible lanes (its own split
    table), each group's engine computes only its own slots, and CC flags group g's CPU-hit part as bit 2g."""
    from lease_chain_rig import CAPACITY

    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu
    from sglang.srt.layers.moe.ram_slot_map import type_lanes

    row, experts = 0, [0, 1, 2, 3, 4, 5]
    w = lease.wire_layout(8, 2)
    split = [0] * 3 + [1] + [0] * (w.lanes - 3)  # 3 eligible lanes on a node: 1 on its CPU
    forwards = [_Forward(), _Forward()]
    ranges = [[(0, CAPACITY)] * LAYERS, [(CAPACITY, 2 * CAPACITY)] * LAYERS]
    c = Chain(tmp_path, capacity=2 * CAPACITY, nodes=2, node_ranges=ranges, copy_engine=True, start=False)
    try:
        x_rows = torch.zeros((LAYERS, 2 * HIDDEN), dtype=torch.uint8).pin_memory()
        out_rows = torch.zeros((LAYERS, 4, HIDDEN), dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:4]
        for g in range(2):
            c.host.enable_cpu_experts(
                forwards[g].address, split, cores[2 * g : 2 * g + 2], x_rows, out_rows, threads=2, group=g,
                engine=g + 1,
            )
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        c.plan(experts, row)
        c.gather(row)  # an uncaptured gather: all six become RAM-resident, no CPU lane
        torch.cuda.synchronize()
        assert c.handled() and set(experts) <= c.resident(row)
        c.host.set_cpu_layer(row, HANDLE)
        c.host.arm_copy_engine()
        backend, plan, dev = c.backends[row], c.plans[row], c.dev
        x = torch.randn(1, HIDDEN, device="cuda").half()
        weights = torch.ones(1, len(experts), device="cuda")
        backend._stage_planned(plan)
        dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=(x, weights))
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
        # The fake forwards take the GIL: let them run before anything blocks in the host.
        assert _until(lambda: len(forwards[0].calls) == 1 and len(forwards[1].calls) == 1)
        torch.cuda.synchronize()
        order = [int(e) for e in plan.expert_ids[: len(experts)].tolist()]
        want, _ = type_lanes(
            order, c.host.mapping(row), [-1] * (2 * w.lanes), split * 2, lanes=w.lanes, captured=True,
            copy_armed=True, hit_copy="ce", cpu_on=True, cpu_misses=False, nodes=2,
        )
        assert c.kinds(len(experts)) == want
        for g, forward in enumerate(forwards):
            ((layer, slots, _),) = forward.calls
            assert layer == HANDLE and all(g * CAPACITY <= s < (g + 1) * CAPACITY for s in slots)
        assert dev.cpu_lanes.tolist()[1] == 0b101, "group 0's hits in part 0, group 1's in part 2"
    finally:
        c.close()
```

- [ ] **Step 2: Run them to verify they fail.** Commit the tests alone (`git add` both; `git diff --cached --stat`; message `test(expert-stream): the GPU chain with two NUMA groups (failing)` with the trailers). `SYNC`, `RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py -k "two_node"`.
Expected: FAIL, `TypeError: Chain.__init__() got an unexpected keyword argument 'nodes'`.

- [ ] **Step 3: Implement.** `Chain.__init__` takes `nodes=1, node_ranges=None`; `wire = wire_layout(lanes, nodes)`; `ExpertStreamHost(..., lanes=lanes, node_ranges=node_ranges)`; `ExpertStreamDevice(..., lanes=lanes, nodes=nodes)`. Commit (`git add test/manual/dsv41/lease_chain_rig.py`; message `test(expert-stream): the lease chain rig runs NUMA groups` with the trailers), `SYNC`, the Step 2 run. Expected: PASS.

- [ ] **Step 4: Every suite against the baseline and the merge-base.** `RUN_CPU SUITE_CPU` plus every test file Tasks 1-9 created, `RUN_GPU SUITE_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`, `RUN_EXT SUITE_EXT test/manual/dsv41/test_cpu_expert_engines_exl3.py`, `CPU_CHECKS check` (`t10`), and the registered kernel suite at both commits:

```bash
ssh divix01 'for W in wt-nlane wt-nlane-base; do cd /data/models/slang/nvfp4-work/$W && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -2; echo "$W EXIT=${PIPESTATUS[0]}"; done'
```

Expected: only the tests this plan added (and the ones it moved or retired: `test_parse_core_list`) differ between the two counts; the Baseline's counts plus this plan's tests for the rest. Record each command beside its counts.

- [ ] **Step 5: The one-node wire is v2 and its modules keep their names.** In the Step 4 runs, `test_eight_lanes_on_one_node_is_wire_v2` and `test_each_host_build_is_compiled_for_its_node_count[1]` passed; `ls -d ~/.cache/sglang/jit/*/expert_stream_host_exl3_prod_l8 ~/.cache/sglang/jit/*/expert_stream_exl3_l8` both exist.

- [ ] **Step 6: Locality on a running server (needs the user's go-ahead: an arm takes the GPU, and production must be down for it).** From the branch's worktree, launch one arm with the production recipe (two NUMA nodes in `PINNED_HOST_NUMA_MB`) plus CPU experts, so every role exists: `SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_DSV41_CPU_EXPERTS_THREADS=16` and the CPU-experts prerequisites `CPU_EXPERTS_ENV` lists in `test/registered/unit/test_expert_stream_requirements_exl3.py`, through `benchmarks/dsv41_baseline/run_arm.sh numa-locality <port> KEY=VAL ...` under the run protocol's lock order (`rowimg-disk.lock`, then `run_arm.sh` takes `cc-gpu.lock`). While its timed set runs, with `PID` the server's pid:
  1. `grep "numa node" server.log`. Expected: exactly `numa node0: ram=17 cpu=8-15 (8) sq=-`, `numa node1: ram=35 cpu=18-33 (16) sq=-` and `numa copy thread: node0 cpus=0-7,16,36-52`.
  2. Thread placement: `for t in /proc/$PID/task/*; do printf "%s %s\n" "$(cat $t/comm)" "$(grep Cpus_allowed_list $t/status | cut -f2)"; done | sort | uniq -c | grep -E "exl3-(ram-miss|cpu-exp|copy-eng)"`. Expected: `exl3-ram-miss0` on 17, `exl3-ram-miss1` on 35, `exl3-copy-eng` on `0-7,16,36-52`, the `exl3-cpu-exp0` team on single CPUs within 8-15 and `exl3-cpu-exp1` within 18-33 (the OpenMP workers inherit the engine thread's name).
  3. Pages: `grep -E "bind:[01]" /proc/$PID/numa_maps`. Expected: every `bind:0` mapping's page counts only `N0=`, every `bind:1` only `N1=` (each group's slots are exactly a binding's rows, Task 5; a lane outside its group's slots fail-stops, Task 6; so every CPU job's slot and every staging slot is on its home node).
  4. Work on both nodes: after the arm's server stops, `grep "exl3 RAM miss group" server.log` shows both groups' counters (Task 8), each with nonzero `rows_read` and `served`.
  Record the four outputs in the ledger.

- [ ] **Step 7: ms/token does not regress (same go-ahead as Step 6).** Production recipe, CPU experts off (production's configuration), two arms per condition with cold servers, paired: A = the merge-base worktree (one group), B = the branch (two groups, because `PINNED_HOST_NUMA_MB` names two nodes). `run_arm.sh numa-a1 <port>` and `numa-a2` from `wt-nlane-base`, `numa-b1`, `numa-b2` from `wt-nlane` (each worktree registered in `generations.json` first, as `run_arm.sh` requires), then `python benchmarks/dsv41_baseline/paired.py <a1> <b1>` and `<a2> <b2>`. Expected: B's decode ms/token within A-vs-A's spread (`paired.py <a1> <a2>`), and B's server start (each arm's `run-manifest.json`) within a few seconds of A's, the cost of registering the tier once per group (Spec delta 11). Record all the outputs. A regression is a finding to report, not something to tune in this plan.

- [ ] **Step 8: The full-stack bench, two groups vs one and N = 8 vs the merge-base.** No server running. The bench service isolates only node 1's CPUs (`bench/service/README.txt`), so two groups cannot run inside it; run all three builds outside it, back to back, three rounds each, same CPUs where they overlap: the merge-base bench at one group (`--service-cpu=17 --cpus=18-33`), the branch at one group (the same), the branch at two groups (`--service-cpu=17,35 --cpus=8-15/18-33 --worker-node=0,1`, `OMP_THREAD_LIMIT=24`), each `exl3_full_stack_prod --benchmark_filter=BM_stack --benchmark_repetitions=3` under `taskset -c 8-35,52` with `run_full_stack.sh`'s OpenMP and ISA environment. Expected: the two one-group p50s within each other's spread (N = 8 is today); the two-group p50 recorded per expert count beside them.

- [ ] **Step 9: Clean up.** `ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-nlane-base'`; ask the user before deleting `nlane-bench2`, `nlane-exl3-build` and the `nlane-cpu-*` directories.

---

### Task 11: NVFP4 follows the engine ABI (touches files under concurrent edit)

**Schedule this task separately.** The user is editing `python/sglang/srt/layers/quantization/nvfp4_cpu/` (and planning `cpu_experts_common/`, Spec delta 22). Before starting: `git status --short python/sglang/srt/layers/quantization/` and `git log -3 --oneline -- python/sglang/srt/layers/quantization/nvfp4_cpu`; if either shows work in progress, stop and ask the user. Stage with `git add -p` only.

**Files:**
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/cpu_experts_cabi.h:33-42` (`set_cores` out; `engine_create`, `engine_free`)
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp:30-75` (the forward mutex and the core globals), `:160-231` (`free_layer`, `set_cores`, `forward`), `moe_mul1.h:142-146`, `forward_plan.hpp:144-178` (`run_team`)
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/README.md:40-47`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/bench/src/cpu_forward.cpp:371-381`, `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp:143-145`
- Modify (tests whose API changes): `test/registered/unit/kernels/test_nvfp4_cpu_experts.py` (the child's `set_cores` lines become `engine_create`), `test/registered/unit/kernels/test_nvfp4_cpu_build.py:26-29` (the exported symbols)

**Interfaces:**
- Consumes: `SglangCpuExpertsForward::engine`, ABI version 2 (Task 2).
- Produces: `int sglang_nvfp4_cpu_experts_engine_create(const int32_t* cores, int32_t n, int64_t* engine)` (0; 2 for invalid, duplicate or not-allowed cores, as `set_cores` checked them); `int sglang_nvfp4_cpu_experts_engine_free(int64_t engine)`; the forward reads `call->engine` (0: unpinned; unknown or freed: 2; `threads` above the engine's cores: 1, as the team-size check did); two forwards on any engines run at once; status 3 only for `free_layer` while a forward runs.

- [ ] **Step 1: Write the failing test.** In `test_nvfp4_cpu_experts.py`, the child takes a fifth argument, the number of concurrent engines (default 1). With cores given it creates `engines` engines over consecutive slices of the core list (`lib.sglang_nvfp4_cpu_experts_engine_create(slice, n, byref(handle))`, printing `engine <status>`), and with two engines runs each call list on both from two Python threads at once, printing `forward <engine> <threads> <status> <written|untouched>` and finally `same` when both engines' outputs are byte-equal. The existing expectations change only in the configuration lines (`cores 0` becomes `engine 0`; `test_cores_cannot_change_after_the_first_forward` becomes `test_a_freed_engine_is_refused`: after `engine_free`, a forward on it prints status 2). Add:

```python
def test_two_engines_forward_at_once_without_a_busy_status(library):
    """Part 3: a CPU expert engine per NUMA group, both forwarding at once. The process-wide forward mutex returned 3
    to the second; per-engine state lets both run and agree byte for byte."""
    lines = _run(library, "2,2,2", cores=_allowed(4), engines="2")
    assert lines[:2] == ["engine 0", "engine 0"]
    assert all(line.split()[3] == "0" for line in lines if line.startswith("forward"))
    assert lines[-1] == "same"
```

(`_run` passes `engines` as `sys.argv[5]`.) `test_nvfp4_cpu_build.py`'s symbol list replaces `sglang_nvfp4_cpu_experts_set_cores` with `sglang_nvfp4_cpu_experts_engine_create` and `sglang_nvfp4_cpu_experts_engine_free`.

- [ ] **Step 2: Run it to verify it fails.** Commit the tests alone (`git add -p` the two files; `git diff --cached --stat` lists only them; message `test(nvfp4-cpu): per-engine cores, two engines at once (failing)` with the trailers). `SYNC`, `RUN_CPU test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_nvfp4_cpu_build.py`.
Expected: FAIL, `AttributeError: .../libnvfp4.so: undefined symbol: sglang_nvfp4_cpu_experts_engine_create`.

- [ ] **Step 3: Implement.** In `moe_mul1.cpp`: replace `forward_mutex` with `std::shared_mutex layer_mutex;` (comment: "Forwards hold it shared, so any number run at once; free_layer takes it exclusively and returns 3 while a forward runs, rather than free a layer under it."); the forward takes `std::shared_lock<std::shared_mutex> lock(layer_mutex, std::try_to_lock)` and returns 3 only if `free_layer` holds it; `free_layer` takes `std::unique_lock(..., std::try_to_lock)`. Replace the core globals and `freeze_compute_cores`/`pin_compute_worker` with Task 2 Step 3's `g_engines_mutex`, `g_engines`, `engine_cores` and `pin_compute_worker(worker, cores, pin_error)` code blocks, copied verbatim; `engine_create` keeps `set_cores`' checks (each core distinct, in `[0, CPU_SETSIZE)` and in `sched_getaffinity(0)` of the caller, else 2). The forward looks up `call->engine` (unknown or freed: 2) and passes its cores through `ForwardCtx` (a `const std::vector<int>* cores` member) to `run_team`, which keeps its team-size check against `ctx.cores->size()` and its pin check. `cpu_experts_cabi.h` declares the two new functions with Task 2's comments and drops `set_cores`; `moe_mul1.h:142-146` drops the core globals' declarations; `README.md:40-47` shows `engine_create` and `call.engine`. The NVFP4 bench (`cpu_forward.cpp:381`) and `nvfp4_cpu_forward_ab.cpp:143` create one engine and set `call.engine`.

- [ ] **Step 4: Run.** Commit (`git add -p` every file above; `git diff --cached --stat` lists exactly them; message `feat(nvfp4-cpu): engine handles replace set_cores; forwards no longer exclude each other` with the trailers). `SYNC`, the Step 2 run, then `bash test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh` in the worktree as its header describes (its bit-exact gate against the merge-base). Expected: PASS, and the script's all-green line.

- [ ] **Step 5: Existing suites.** `RUN_CPU SUITE_CPU`. Expected: Baseline counts plus every task's tests.
