# NUMA node distributor and N-lane lease layout: design

Date: 2026-10-03. Status: approved in conversation, awaiting spec review.

## Goal

Two changes to the DSV4.1 expert-stream runtime (`python/sglang/kernels/jit/csrc/moe/expert_stream/`,
`python/sglang/srt/layers/moe/`):

1. **N lanes.** A demand record carries any lane count 1 <= N <= 32 instead of exactly 8. Every wire offset is
   derived from one trait, `LeaseLayout<NumLanes, NumNodes>`.
2. **One expert group per NUMA node.** A `NumaNodeDistributor` holds one `NumaGroup` per node. Each group owns the
   part of the pinned tier bound to its node, a RAM/NVMe thread, and a CPU-expert engine with its own OpenMP team,
   all on that node's cores. An expert is served (read from NVMe, held in RAM, computed on the CPU) only by its home
   node's group, so its bytes and its compute never cross the socket link. A `ThreadingConfig` class is the one
   place that reads core-related env vars and topology.

Success: on a two-node launch, every CPU-expert job reads a slot on the worker's own node and every NVMe miss lands
on its expert's home node; at N=8 and one node the wire is byte-identical to today and every existing test passes
unchanged; ms/token does not regress against the single-group runtime.

## Context: the runtime today

- **Pinned tier.** One slab (or arena) per streamed layer, allocated by `ExpertPinnedHostCacheManager`
  (`srt/layers/moe/expert_stream.py:815-880`). `SGLANG_MOE_PINNED_HOST_NUMA_MB="0:40960,1:40960"` binds contiguous
  row ranges to nodes with `mbind` (`host_numa.py:107-153`): a layer's low slots are on node 0, its high slots on
  node 1. Nothing uses this: `RamTier`'s victim choice is a node-blind LRU (`host/ram_tier.h:1346`).
- **One of everything.** `Exl3RamMissService` is a process singleton with one `RamTier`, one `RamThread`
  (`host/ram_thread.h`), one copy thread (`host/copy_engine.h`) and at most one `CpuExpertEngine`
  (`host/cpu_experts.h:109-311`). The NVMe io_uring reads run inline on the RAM thread (`ram_tier.h:1133, 1586`);
  only the optional SQPOLL kernel thread is separate.
- **CPU kernel cores are process-wide.** `sglang_exl3_cpu_experts_set_cores` fills `g_configured_cores`, frozen at
  the first forward (`exl3_cpu/optimized/moe_mul1.cpp:2103-2140`). Two engines cannot use different cores.
- **Wire (v2), `lease_layout.h`.** A 16-record request ring the device writes and the host reads under a per-record
  seqlock; a completion block (PieceMask per lane, CopyDone per record, the copy gate, the CPU split `kSplit`); one
  map delta per row (`tag`, entries, `staging` i16[8]). Only one record is in flight: the device posts G+1 after
  G's chain ended, which waits on CopyDone and PieceMask (`analysis/dsv41-drive/LEASE_PROTOCOL.md`).
- **The device picks a miss's staging slot**: the m-th miss takes the m-th entry of the row's staging list
  (`lease_device.cuh:386`), node-blind.
- **8 is hard-wired** in about 140 places across 18 C++/CUDA files (`kMaxIds`, `kLeaseLanes`), in the record format
  (32 header bytes + 12 per lane = 128 at 8; `counts` packs lanes and protect ids in 4 bits each; `kinds` packs
  4 bits x 8 lanes in one u32; device 16-byte vector loads of i16[8]), and in the Python mirrors
  (`kernels/ops/moe/expert_stream_transport.py` `MAX_IDS`, `_RECORD_LANES`; `expert_lease_block.py`;
  `srt/layers/moe/exl3_ram_miss.py`). `test_exl3_ram_miss_device_args` parses the `constexpr` lines of
  `lease_layout.h` to check those mirrors.
- **Core settings are spread out.** `SGLANG_DSV41_CPU_EXPERTS_CORES`/`_THREADS` (`cpu_experts/service.py:336-342`),
  `SGLANG_DSV41_RAM_MISS_SPIN_CORE` (`exl3_ram_miss.py:957`), `SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU`
  (`host/uring_options.h:137`); the >= 2 cores and threads <= cores checks are in both `pool.py` and `service.py`;
  the 64-71 reservation in three places; SMT-sibling exclusion in C++ (`core_topology.h`) and by hand in
  `benchmarks/dsv41_baseline/arm_env.py`.
- **divix01.** Node 0 = CPUs 0-17, 36-53; node 1 = 18-35, 54-71; the GPU is on node 0; the server runs on
  `0-7,16,36-52`; 64-71 take NVMe interrupts. One GPU, tp = 1. CPU experts are off in prod today.

## Design

### Part 1: `LeaseLayout<NumLanes, NumNodes>`

- `lease_layout.h` defines `template <int NumLanes, int NumNodes> struct LeaseLayout` whose `static constexpr`
  members are every offset and size the free constants hold today. `static_assert(1 <= NumLanes && NumLanes <= 32)`,
  `static_assert(NumNodes >= 1)`.
- **Lane width.** `kLanes = round_up(NumLanes, 8)`, so each i16 lane array is whole 16-byte vector loads. The
  device's per-array loads become `kLanes / 8` loads.
- **Record.** `kRecCounts` becomes two u8 fields (lanes, protect ids), each up to 32. `kRecKinds` becomes
  `kLanes / 8` u32 words, 4 bits per lane. Lane arrays (`protect`, `lane_expert`, `lane_slot`, `lane_dst` i16,
  `lane_weight` f32) are `kLanes` long. `kRecordBytes` rounds the record up to a multiple of 128 bytes (one L2
  adjacent-line pair). At `NumLanes = 8, NumNodes = 1` every offset equals today's, checked by `static_assert`s
  against the v2 numbers.
- **Completion block.** PieceMask `[kDemandRecords][kLanes]`, one 128-byte line each. `kSplit` becomes
  `i32[NumNodes][kLanes + 1]`: CPU lanes per n eligible lanes, per node. `kLeaseBlockBytes` is derived and rounded
  to 4096.
- **Delta.** `staging` becomes `i16[NumNodes][kLanes]` (node-major); `kDeltaMaxEntries = 2 * kLanes`; the stride is
  derived and rounded to 128.
- The ring stays 16 records (`kDemandRecords`); piece bits per lane stay 8 (pieces, not lanes).
- **Choosing N.** N is the service's planned gather width (`graph_gather_rows`, top-k at decode batch 1), rounded up
  to 8. The JIT builds the device kernels and the host tier for that N; a launch whose width exceeds 32 is refused
  at start. No env var.
- **Who takes the trait.** The device kernels (`lease_device.cuh`, `lease_kernels.cuh`, `row_copy_kernels.cuh`),
  the host tier (`ram_tier.h`, `tier_protocol.h`, `copy_engine.h`, `cpu_experts.h`, `fixed_vec.h` capacities,
  `split_calibration.h`, the FFI exports) and the bench (`bench/src/device_sim.*`, `stack.h`, `full_stack.cpp`)
  take it as a template argument, carried through the host's existing `Layout` template parameter where one exists.
- **Python.** `MAX_IDS`, `_RECORD_LANES` and the copied offsets become one function,
  `lease_layout(num_lanes, num_nodes)`, in `kernels/ops/moe/expert_stream_transport.py`, used by
  `expert_lease_block.py` and `exl3_ram_miss.py`. `test_exl3_ram_miss_device_args` stops parsing `constexpr` lines:
  it compiles a probe printing the trait's members for N in {1, 6, 8, 13, 32} and nodes in {1, 2} and compares them
  to the function.

### Part 2: `ThreadingConfig`

- A Python class beside `srt/layers/moe/cpu_experts/service.py`, built once by `ThreadingConfig.from_env()`. It is
  the only reader of `SGLANG_DSV41_CPU_EXPERTS_CORES`, `SGLANG_DSV41_CPU_EXPERTS_THREADS`,
  `SGLANG_DSV41_RAM_MISS_SPIN_CORE`, `SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU`, the process affinity and the
  reserved cores. C++ receives resolved core lists only. The duplicated checks in `pool.py`, `service.py`,
  `ffi_exports.h` and `arm_env.py` are removed; `core_topology.h`'s sibling check stays as a C++ assertion.
- The node set is the nodes named in `SGLANG_MOE_PINNED_HOST_NUMA_MB`; without it, one node (the GPU's).
- **Per node it derives a `NodePlan`** from `/sys/devices/system/node/node*/cpulist` and each CPU's
  `topology/thread_siblings_list`:
  1. Usable = the node's physical cores, minus the server's affinity, minus the reserved set (64-71), minus the SMT
     siblings of anything already chosen.
  2. RAM/NVMe core = the highest usable core.
  3. CPU engine: the engine thread is worker 0; the workers are the remaining usable cores, capped by
     `SGLANG_DSV41_CPU_EXPERTS_THREADS` when set.
  4. SQPOLL core (when io_uring SQPOLL is on) = the next usable core.
- The copy thread goes on the GPU's node (the GPU's sysfs `numa_node`).
- **Override.** `SGLANG_EXPERT_NUMA_CORES`, e.g. `"1:ram=35,cpu=18-33"`, replaces the derived plan of each node it
  names and is validated by the same rules. The new env var follows `.claude/skills/env-var-conventions`.
- **Validation.** Refuse to start on: a node with no usable core, any overlap with the server's affinity or the
  reserved set, an engine with fewer than 2 cores, a core outside its node, or SMT siblings in one plan.
- Log one line per node at start, e.g. `numa node1: ram=35 cpu=18-33 (16) sq=-`.
- Expected on divix01: node 0 `ram=17 cpu=8-15`, node 1 `ram=35 cpu=18-33`, copy thread on node 0.

### Part 3: `NumaNodeDistributor` and `NumaGroup`

- **Home rule.** `home(layer, expert) = expert % num_nodes`, one function in C++ (host and device) and Python, so a
  popularity table can replace it later without touching callers.
- **`NumaGroup`** (one per node), owning:
  - its slot range: the slots of each layer's slab already bound to its node by the existing `mbind` split (no new
    allocation); its own slot table and LRU over that range, so victims are chosen only there;
  - its staging slots, reserved inside its own range (`reserve_staging` per node, `min(kLanes, range - 1)` each);
  - a `RamThread` pinned to the node's RAM core, which also runs the io_uring reads (and its own watchdog);
  - a `CpuExpertEngine` whose thread is pinned to the node's engine core and whose OpenMP team runs on the node's
    worker cores.
- **`NumaNodeDistributor`** owns the groups, the home rule, and the combiner below. At one node it is a thin wrapper
  over one group.
- **Dispatch: no extra hop.** Each group's RAM thread polls the one request ring, reads record G under its seqlock,
  and takes only the lanes whose home is its node. Reading is safe: both readers are read-only, and G cannot be
  overwritten while either group still works on it, because the device posts G+1 only after G's chain ended and
  G's slot is reused only at G+16.
- **Combiner (completion side).** The host-written words that one group cannot own alone:
  - **Map delta.** Each group stages its row entries (inserts, evictions) and its node's staging list for record G
    with the distributor. The last group to report writes the row's one delta (entries, all nodes' staging lists)
    and its tag, entries before the tag with a release, as today.
  - **CopyDone and the gate.** The single copy thread stays on the GPU's node. Its completion for G waits on the DMA
    and on every group's CPU `done(seq)` for G before it stores CopyDone and opens the gate.
  - **PieceMask** needs nothing: each lane belongs to exactly one group.
- **Device changes** (small, in the lease kernels):
  - a miss takes the next staging slot of node `home(expert)` from that node's list;
  - the CPU lanes are chosen per node with `kSplit[node][n]`, so a node with fewer workers gets fewer CPU lanes.
    The split is configured and calibrated per group (`service.py` `configured_split`, `retune`, `calibrate`).
- **CPU kernel.** `g_compute_cores` becomes per-engine state: the C ABI gains an engine handle, created from a core
  list and passed to every forward and keep-warm call, replacing the process-wide `set_cores`, so each
  `CpuExpertEngine` runs its own team on its own cores. `cpu_experts_cabi.h` and its bench
  callers change with it; the NVFP4 kernel, which mirrors the same ABI, follows.

### Part 4: CPU experts for Qwen through the lease backend

Qwen3.8-Flash-Next (ModelOpt NVFP4 experts, top-k 10, 512 experts per layer, flashinfer_cutlass MoE) runs the
zero-host gather today: `InGraphRowBackend` copies every routed expert from pinned RAM with SM loads over UVA
(`expert_cache_transfer.cuh`), and every expert fits in RAM. The lease path is not a separate pipeline: it is another
row backend in the same `ExpertStreamer._gather_graph`, sharing the route plan (`expert_route_plan.cuh`), the direct
gather and commit (`expert_residency/direct_gather.cuh`), the residency updater, and the SM copy kernel itself (C1).
So the copy kernels are not merged. Qwen gets CPU experts by running the lease backend in a full-resident mode:

- **Backend choice.** With CPU experts off, Qwen keeps `InGraphRowBackend` (no host round trip). With them on, it
  uses the lease backend, whose lanes are then only `HIT_SM` (C1, the same `.nc` UVA kernel), `HIT_COPY` and
  `HIT_CPU`; no lane misses.
- **Lane width.** top-k 10 needs N = 16, delivered by Part 1 (including `cpu_lanes` and every reader of it).
- **NVFP4 row layout.** An `ExpertRowLayout` for Qwen's six streamed tensors (`w13_weight`, `w2_weight`,
  `w13_blockscale_swizzled`, `w2_blockscale_swizzled`, `g1_alphas`, `g2_alphas`), built as a second entry of
  `LAYOUTS` in `expert_stream_transport.py` (host and device), beside `"exl3"`.
- **Full-resident tier.** The RAM tier registered over the existing all-expert pinned buffers with every expert
  mapped at start, no NVMe reader, and no staging slots; a record that types a miss is a fail-stop.
- **NVFP4 CPU trait.** A `CpuExpertQuantTrait` for NVFP4 in `cpu_experts/` (`cpu_trait_for` knows only `"exl3"`
  today), loading the NVFP4 kernel (`nvfp4_cpu_ext.py`) through the shared C ABI; Qwen runs its `GenericShape` plan
  until a measured Qwen shape is added to `shapes.hpp`.
- **Output merge for flashinfer_cutlass.** The CPU's lanes are removed from the fused MoE's top-k (routed to the
  dump slot with zero weight) and `cpu_out` (both parts) is added to the MoE output in fp32 after it. This is the one
  format-specific piece; EXL3 keeps its route-tables seeding.

Out of scope for Part 4: NVMe-backed Qwen (a partial tier), and retiring `InGraphRowBackend`.

### Errors

- Any group's fail-stop stops the whole service, as one fail-stop does today.
- A stalled group never lets CopyDone be stored; the existing copy-wait timeout aborts, and its message names the
  group whose `done(seq)` was missing.
- `ThreadingConfig` and the N > 32 check refuse at start; nothing degrades silently.

### One node is today's runtime

With one node and N = 8 the wire is byte-identical to v2, the distributor wraps one group, and every existing test
must pass unchanged. This is the degenerate case of the design, not a flag.

## Phases

1. **N lanes, one node.** The trait, the templated device and host code, the Python `lease_layout` function and the
   probe test. Gate: byte-identical v2 offsets at N = 8 and the existing lease, RAM-miss and full-stack tests green.
2. **NUMA groups.** `ThreadingConfig`, per-engine CPU cores, per-node staging and split on the device, the groups,
   the dispatch filter and the combiner.
3. **Qwen CPU experts** (Part 4), after phase 1; independent of phase 2. Gate: Qwen with CPU experts matches the
   GPU-only output within the CPU kernel's tolerance, and Qwen with them off is unchanged (same backend, same
   ms/token).

## Testing

Each run names its files narrowly, per `.claude/rules/divix01-run-protocol.md`.

1. Layout probe vs `lease_layout()` for N in {1, 6, 8, 13, 32}, nodes in {1, 2}; v2 offsets at (8, 1).
2. Device lease kernels (`test_exl3_lease_kernels_cuda.py`) at N in {8, 16, 32}: full and partial records; per-node
   staging, including every miss homed on one node and a node out of staging slots (trap); per-node split.
3. Host tier: each group takes only its lanes; both nodes missing in one record yields exactly one merged delta;
   victims only from the group's range; CopyDone waits for every engine; a stalled group times out naming itself.
4. CPU kernel: two engines on disjoint cores running at once are bit-exact against one engine; the C ABI's
   per-engine core configuration.
5. `ThreadingConfig` on fake sysfs trees: the divix01 topology, one node, overlap with the server's cores, the
   override, each refusal. CPU-only, runs on the laptop.
6. divix01: the `exl3_cpu_forward_ab` and full-stack bench, two groups vs one and N = 8 vs today; every CPU job's
   slot and every miss's staging slot checked to be on its home node with `page_nodes`; ms/token under the bench
   service. Socket-link counters are not readable (`perf_event_paranoid=2`), so locality is checked by page
   placement and the worker's core, not by link traffic.

## Out of scope

The server's own core layout; more than one GPU or tp > 1; a popularity-based home table; wiring the NVFP4 kernel
into the runtime; N > 32.
