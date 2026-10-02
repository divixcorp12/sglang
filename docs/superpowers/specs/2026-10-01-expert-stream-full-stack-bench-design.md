# Expert-stream full-stack CPU-expert benchmark: design

Date: 2026-10-01. Branch: `expert-stream-cpu-bench`.

## Purpose

Measure what the expert-stream service stack costs on top of the raw EXL3 CPU kernel. A C++ writer plays the GPU:
it posts CPU-expert requests into the lease lanes, the real host stack serves them, and the result comes back through
the response lane. The headline number is `overhead = (post -> CopyDone) - (bare forward)`, per expert count, in one
process on production's core placement.

In scope, all real code, unmodified:

- `host/ram_tier.h` (`RamTier`), with its copy engine on the host backend (`enable_copy_engine(-1)`);
- `host/ram_thread.h` (`RamThread`), busy-polling on its own core;
- `host/cpu_experts.h` (`CpuExpertEngine`), via `RamTier::enable_cpu_experts`;
- `host/row_reader.h` + `UringReader` (io_uring, O_DIRECT), which load the experts into RAM slots;
- `srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`, the optimized kernel, through
  `sglang_exl3_cpu_experts_forward`.

Out of scope: CUDA, the GPU kernels, Python and tvm-ffi at run time. The writer stands in for the post kernel, S and
CW.

## Decisions (agreed 2026-10-01)

1. Purpose: stack overhead vs the raw kernel, not decode-step realism or a soak.
2. Lanes: phase 1 CPU hits (`kKindHitCpu`); phase 2 CPU misses (`kKindMissCpu`), later.
3. Approach A: a native C++ device stand-in over the real headers. Not the FFI exports, not Python `ChainSim`.
4. Two binaries: `ProdBuild` for the headline numbers, `InstrBuild` for the breakdown.
5. Style: Google Benchmark, like `bench/src/cpu_forward.cpp`: same fixture, references, options, counters.
6. It runs inside the `exl3bench.service` isolated partition, through `service-command.txt`.
7. The partition widens from `18-33,54-69` to `16-33,52-69`, so placement matches production exactly.

## Placement

| Role | CPU | NUMA node | Why |
|---|---|---|---|
| Writer (GPU stand-in) | 16 | 0 | node 0, next to the service, like the GPU's DDIO into node 0's L3 |
| `RamThread` service, busy-poll | 17 | 0 | production's service core (read-record plan, option B) |
| CPU expert worker 0 (engine thread) | 18 | 1 | production's caller core |
| CPU expert helpers | 19-33 | 1 | production's 16-worker team |

SMT siblings 52-69 are reserved and idle. The request page and lease block are first-touched by the writer (node
0); the slabs, x and output buffers by worker 0 (node 1). The runner refuses any other split, any CPU outside the
process's allowed set, and duplicate CPUs. All four are options (`--writer-cpu`, `--service-cpu`, `--cpus`), with
these defaults.

Fidelity limit, stated in the README: the writer's stores reach the service by cache coherence between two node-0
cores, not by PCIe/DDIO. The pickup component is a lower bound on what the GPU path sees.

## Files

All under `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/`.

| File | Responsibility |
|---|---|
| `src/stack_fixture.h/.cpp` | From `Fixture`: EXL3 slabs per layer, row-image files, `Tables`, x/output buffers |
| `src/device_sim.h/.cpp` | The device's side of the lease protocol (port of `python/sglang/test/dsv41_chain_sim.py`) |
| `src/stack.h` | Owns `RamTier<RowReader<Exl3RowLayout, UringReader, Build>>` and `RamThread`; setup/teardown order |
| `src/full_stack.cpp` | Google Benchmark main: options, validation, `BM_bare`, `BM_stack`, `--self-test` |
| `CMakeLists.txt` | Targets `exl3_full_stack_prod` and `exl3_full_stack_instr` |
| `run_full_stack.sh` | Process rounds and the results record, shaped like `run.sh` |
| `README.txt` | A "Full-stack bench" section |
| `service/exl3bench-isolation`, `service/exl3bench-run`, `service/install.sh`, `service/README.txt` | Partition `16-33,52-69` |

### `stack_fixture`

- Input: the existing `Fixture` (8 layers, 5 experts, H=5120, I=2304, 9 matrices per expert: gate, up and down
  trellis/suh/svh) and `--image-dir`.
- Slabs: per layer, the six `Exl3RowLayout::kNames` tensors with capacity C = 8 slots (5 experts + 3 staging),
  shaped as the pinned tier holds them (`w13_*` as `[slot, 2, ...]` gate/up, `w2_*` as `[slot, 1, ...]`). They are
  64-byte aligned host allocations, first-touched on node 1. The bench keeps the slab memory alive for the
  registered layers' lifetime.
- Row images: one file per layer, written at setup into `--image-dir` in `scripts/dsv41/build_row_images.py`'s
  layout: a row per expert at `e * row_stride`, name offsets inside the row, offsets and lengths multiples of 512.
  The directory must accept O_DIRECT (checked by opening a probe file with it), so `/data`, not tmpfs.
- `Tables`: built directly (no `tables_from`), as `exl3_ram_miss.py::_row_image_tables` builds them, then checked
  with `check_image_tables<Exl3RowLayout>`. `images = true`, `direct = true`.
- x: per row, the layer's FP16 fixture input, at `x_base + row * x_stride`. Output: per row, two FP32 parts of H
  (`out_part_stride = H * 4`): part 0 the CPU hits' sum, part 1 the CPU misses'. With `CopyDone`, this is the
  response lane.

### `device_sim`

A C++ port of `ChainSim`'s device role. It is a stand-in for the CUDA kernels, not evidence about them.

- `post(row, experts, weights, captured) -> gen`. The record goes at `page + kDemandRing + ((seq - 1) % 16) * 128`,
  with seq = head + 1 (0 skipped, epoch incremented) and `gen = epoch << 32 | seq`. Write order:
  1. `seq = 0`;
  2. the payload (row, counts, flags, chain, epoch, kinds, protect, lane expert/slot/dst/weight, `-1` padded);
  3. a release store of seq;
  4. a release store of `kDemandHead`.
- Lane typing follows `ram_slot_map.type_lanes`. A lane is a hit when the replica has `ram_slot[e] >= 0`; the m-th
  miss takes `staging[m]`. `host_lanes = captured && kCopyArmed`. Of the n eligible lanes, the last `split[n]`
  (read from `kSplit`) go to the CPU. The bench sets `split[n] = n`, so every eligible lane goes to the CPU.
  Uncaptured posts type misses `kKindMissGpu`; they are used to load experts.
- Slot-map replica: `map_chain[row]` starts at 1. A post with a miss lane increments it and writes it as `chain`.
  Before typing, the delta at `lease + kDeltaBase + row * kDeltaStride` is applied when its tag (an acquire load)
  equals `map_chain[row]`. Posting with a delta still owed waits for it.
- `copy_wait(gen, deadline)` stores the gate closed for the record's seq, then spins until `CopyDone[idx] == gen`.
  The host opens the gate (`copy_completed`'s CAS); the writer does not.
- `wait_pieces(gen, lane, deadline)` waits until `PieceMask[idx][lane]` reads `piece_word(gen, 0xFF)`. Setup loads
  use it.
- x is written before the post's payload, as `stage_cpu_input` does.

### `stack.h`

`template <class Build> class Stack`. `Reader` is `UringReader` for `ProdBuild` and `InstrUringReader` for
`InstrBuild`, matching `exl3_ram_miss_host.cpp` and `exl3_ram_miss_host_instr.cpp` minus `FaultyReader`.

Setup order (the CPU-experts test's, `test_exl3_ram_miss_cpu_experts.py::_host`):

1. Construct `RamTier(page, slot_map, lease, lease_bytes, tables, capacity, direct=true, hot_page=nullptr, 0)`,
   then `open()`.
2. `reserve_staging(3)`.
3. `enable_copy_engine(-1, spin_ns, wait_timeout_ns)`, then `arm_copy_engine(true)`.
4. `enable_cpu_experts(config, split)`, with forward = `sglang_exl3_cpu_experts_forward`, `threads = 16`,
   `cores = 18..33`, and `sglang_exl3_cpu_experts_set_cores` called first.
5. Construct `RamThread(tier, service_cpu, fatal_wait_ns, spin_ns, busy_poll=true)` and `start()`.
6. Load the experts: one uncaptured post per layer naming all 5 experts, waiting for every lane's pieces.
7. Per layer: register over per-slot views of the tier's slabs, with `exl3_moe_cpu_make_layer` and the same
   arguments `cpu_experts/exl3.py::register_layer` passes, then `set_cpu_layer(row, handle)`.

Buffers: page = `kPageBytes`, zeroed. Lease = 4096-aligned, zeroed, `kLeaseBlockBytes + roundup4096(rows * 256)`.
`slot_map` = int32 `[rows][experts]`, filled with -1.

Teardown: `open_closed_gate()`, then `RamThread::stop()`, then `final_settle()`, then free the layer handles.

### `full_stack.cpp`

Options and output mirror `cpu_forward.cpp`: `--fixture`, `--reference-dir`, `--warmup-forwards`, `--gap-us`,
`--validate-only`, Google Benchmark's own flags. New options: `--image-dir`, `--writer-cpu`, `--service-cpu`,
`--cpus`, `--wait-timeout-ms` (default 2000), `--self-test`.

The main thread pins itself to the writer CPU before setup, so the writer's clock and stores run there.

Benchmarks, each for k in {1, 3, 5} (experts 0..k-1 with the fixture's routing weights), rotating the 8 layers
between calls, outside the timed interval:

- `BM_bare/experts:k`: the C ABI forward on the same handles, slots and output memory, called from the writer
  thread. Times the call.
- `BM_stack/experts:k`: write x for the row, post a captured record of k `kKindHitCpu` lanes, close the gate, spin
  until `CopyDone == gen`. `t0` is before the x write, `t1` when CopyDone is seen.

Counters per benchmark: mean, `p50_us`, `p95_us`, `p99_us` (per-call samples, computed outside timing, as in
`cpu_forward.cpp`). `BM_stack` adds `overhead_p50_us`, its p50 minus the same-process `BM_bare` p50 for the same k.

The `InstrBuild` binary adds breakdown counters, per call, from `enable_trace` / `drain_trace` and `cpu_stats`:

- `pickup_us = observed - t0`;
- `service_us = done - observed`;
- `forward_us` = the change in `compute_ns` across the call (exact with one request in flight);
- `handoff_us = (t1 - done) - forward_us`: the CPU job queue, done-word and copy-thread CopyDone path.

They are reported as p50/p95.

Before timing: all 24 outputs (8 layers x k = 1/3/5) through the stack match `reference-e{k}.bin` bit-exactly
(part 0), and so do the bare forward's. After timing: all 8 outputs for the k just timed are checked again. Counters
must reconcile: CPU jobs = stack calls, no `kKindHitCopy`, `kKindHitSm` or miss lanes during timing.

### `run_full_stack.sh`

Usage: `run_full_stack.sh BUILD_DIR RESULTS_DIR [options...]`. It refuses an existing `RESULTS_DIR`. Like `run.sh`,
it writes `environment.txt` (date, uname, cgroup, allowed CPUs and mems, binary and fixture hashes, `ldd`) and
alternates `prod`/`instr` process order across `EXL3_BENCH_ROUNDS` rounds (default 8). It also writes each
process's exit status, and exits nonzero if any process failed. Under the service:

```
printf '%s\n' /bin/bash <checkout>/.../bench/run_full_stack.sh /data/models/exl3_exp/google_benchmark/full-stack-build \
  > /data/models/exl3_exp/google_benchmark/service-command.txt
```

When `EXL3BENCH_RESULTS` is set and `RESULTS_DIR` is omitted, it writes there.

## Partition change

- `exl3bench-isolation` and `exl3bench-run`: `target` / `cpus` = `16-33,52-69`.
- `install.sh` and `service/README.txt` follow.
- The existing `run.sh` bench is unchanged; it pins itself to 18-33.
- Installing needs one `sudo bash service/install.sh` by the user, with the service stopped.
- Effect while a bench runs: cores 16, 17, 52 and 53 leave the general pool. Today they carry `event_engine` and
  tokio threads, which the isolation moves elsewhere.

## Error handling

Every failure exits nonzero with a message; no timing from a failed run is reported.

- A `copy_wait` or `wait_pieces` past its deadline prints gen, row, lanes, kinds, and the tier's counters and
  `cpu_stats`.
- The stack's own fail-stops stay as they are: a failed forward, a torn record, the watchdog's hung request. They
  abort the process, and `run_full_stack.sh` records the status.
- Setup refuses, before any timing:
  - a NUMA split other than the placement table's;
  - a CPU outside the allowed set, or duplicate CPUs;
  - a non-O_DIRECT image directory;
  - a fixture size/hash or reference mismatch;
  - `check_image_tables` failing;
  - a worker affinity check failing (as `cpu_forward.cpp` checks it).

## Testing

- `--self-test`: about 1 s, no isolation, any CPUs (it pins nothing; run it under `taskset -c 0-15`). It drives the
  real stack on synthetic rows with a fake forward. Its fixed scenarios assert what `DeviceSim` produced, with
  expected values written into the test from `type_lanes` / `ChainSim`:
  - the record bytes (field offsets, `-1` padding, kinds nibbles, flags);
  - the seqlock order (seq stays 0 until the payload is in place);
  - delta application and the chain number after misses;
  - typing with `split[n] = n` and with `split = 0` (no CPU lanes, so no copy wait needed);
  - an epoch wrap at seq 0.
- `--validate-only`: full setup and the 24 bit-exact checks through the stack, with no timing.
- Mutants, applied on divix01 in a private worktree, recorded and reverted (run protocol):
  - a wrong slot in one lane must fail the bit-exact check;
  - dropping the release store of `kDemandHead` must hit the copy-wait deadline with its message.
- Builds run in the protocol's divix01 worktree. Only these binaries' modes run for this change; no registered
  pytest suite is affected, since no Python or shared header changes.

## Phase 2: CPU misses (later; designed here, planned separately)

- The same loop with misses. Before each call, the writer evicts the target expert by loading other experts
  (uncaptured posts). The call then posts MISS_CPU lanes, which the service reads into staging through io_uring,
  then computes into part 1.
- Correctness is a tolerance check against the reference: part 1 sums in landing order, so it is not bit-exact.
- Image placement: `--image-dir` on NVMe, with `rowimg-disk.lock` taken by `run_full_stack.sh`, and page cache
  dropped per round (or the image larger than RAM's headroom, recorded in `environment.txt`).
- It reports read + compute, and the `InstrBuild` trace's `submit`/`first_cqe`/`last_cqe`.

## Non-goals

- GPU-side timing, PCIe or DDIO effects.
- Batch > 1 or multi-record overlap: one request in flight, as at decode batch 1.
- Changing any production header to make it benchable. If a needed hook is missing, the plan stops and asks.
