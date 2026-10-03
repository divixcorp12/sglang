# Native NVFP4 CPU forward benchmark

Follows `kernels/jit/csrc/moe/expert_stream/bench/run.sh`: two separate
executables with fixed worker teams, alternating backend order each round,
Google Benchmark manual wall timing, fresh JSON/log files and environment
records. No Python, PyTorch, CUDA or OpenMP is required.

`nvfp4_cpu_baseline` uses the plugin's scalar dot-product path, compiled for
the compiler's default CPU target. `nvfp4_cpu_optimized` compiles the same
source with `-march=native`, selecting AVX2 on the x86 reference host. This
compares the plugin's own scalar/native paths, not NVFP4 against EXL3. Both
use the same persistent thread pool, weights, FP16 input, FP32 computation,
routing coefficients and `CpuExpertForward` ABI.

## Build

From the checkout root, on the Linux machine where timing will run:

```sh
src=python/sglang/srt/layers/quantization/nvfp4_cpu
build=/absolute/path/nvfp4-cpu-build
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DNVFP4_CPU_NATIVE=ON -DNVFP4_BUILD_BENCHMARK=ON
cmake --build "$build" -j4
ctest --test-dir "$build" --output-on-failure
```

This builds the shared plugin, a native correctness harness and both benchmark
executables in BUILD_DIR. Google Benchmark v1.9.4 uses the same pinned commit
as the EXL3 benchmark; it is fetched if not installed. For offline builds,
install its CMake package and set `NVFP4_FETCH_BENCHMARK=OFF`, or point
`FETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK` to a local checkout of that version.
The `bench` directory can also be configured directly to build only the two
benchmark executables.

## Run and worker sweep

Run inside an already isolated shell; the script changes no cgroups, services
or system NUMA policy. The default core list is 18-33, workers=16, NUMA node=1,
eight alternating rounds and 512 calls per benchmark. Reserve SMT siblings
too when measuring. Do not taskset the whole launch to one CPU: helpers need
all requested cores. Worker zero first-touches the fixture on its CPU. A
process-wide `numactl` policy, if desired, must be set by the caller.

```sh
NVFP4_BENCH_ROUNDS=8 NVFP4_BENCH_WORKERS=1,4,8,16 \
  bash "$src/bench/run.sh" "$build" /absolute/path/NEW_RESULTS_DIR \
    --cpus=18-33 --numa-node=1
```

Each worker count and backend gets its own process. Result names are
`workers-N-round-R-{baseline,optimized}.{json,log}`. The runner refuses an
existing results directory. It captures executable SHA256s, fixture SHA256
when supplied, `compile_commands.json`, dynamic dependencies, launch arguments,
CPU topology and allowed CPU/memory masks.

Defaults are **synthetic** packed weights and scales generated with an
explicit seed, H=5120, N=2304, five resident expert slots, one layer and expert
counts 1,3,5. There is no pretrained checkpoint or full-model quality claim.
The input source and dimensions are recorded in each JSON file.

Small smoke test (choose permitted cores on the specified NUMA node):

```sh
NVFP4_BENCH_ROUNDS=2 NVFP4_BENCH_WORKERS=1,3 \
  bash "$src/bench/run.sh" "$build" /absolute/path/NEW_SMOKE_DIR \
    --cpus=0-2 --numa-node=0 --hidden=80 --intermediate=80 \
    --warmup-forwards=2 --benchmark_min_time=8x
```

Pass `--experts=1,3,5,8 --capacity=8` to include eight routed experts,
`--layers=N` to cycle through multiple layer images, or `--gap-us=N` to sleep
before each timed forward. Gap sleep is excluded from timing; pool wakeup is
included. W13 layout is `--w13-layout=0` (gate/up), `1` (up/gate), or `2`
(alternating 64-row up/gate blocks; N must be divisible by 64).

## Timing and correctness

Before timing, each layer/count is checked against an independent FP64
decoded-weight reference. Scales are unswizzled by iterating the reshape/
permute axes; the oracle does not call the production kernel. It uses full
precision activations and ordinary SiLU, matching the CPU's W4A16 convention.
The tolerance is `abs_error <= 1e-4 + 1e-4*abs(reference)`; non-finite data and
outputs fail. This is not a GPU W4A4 reference.

Every measured workload repeats all layer comparisons before and after its
timing loop. Warmup defaults to 128 calls. Actual helper/caller thread affinity
is checked before and after the benchmark, not just configured optimistically.
Reference decoding, registration, warmup, allocation and validation are all
outside the measured interval. The measured call includes the plugin's worker
wakeup, both expert matvec stages, SiLU and routed FP32 reduction.

JSON counters include p50/p95/p99 latency in microseconds, workers, experts,
layers, packed weight/scale bytes per forward and `logical_weight_GBps`.
That rate is logical slab bytes divided by call time, **not measured DRAM
bandwidth**. Cached weights can make it exceed actual memory bandwidth.
The harness excludes the service job ring, pinned UVA handoff, PCIe/NVMe
transfers and the surrounding GPU MoE graph. A bare-forward result is not
end-to-end tokens/s.

## Frozen GPU-layout fixture

Create a reproducible synthetic fixture once, then give the same file to both
executables (existing files are refused):

```sh
"$build/nvfp4_cpu_optimized" --cpus=18-33 --numa-node=1 \
  --write-fixture=/absolute/path/NEW_FIXTURE.bin
bash "$src/bench/run.sh" "$build" /absolute/path/NEW_RESULTS_DIR \
  --fixture=/absolute/path/NEW_FIXTURE.bin
```

A native exporter can write real slab images using this format. All scalar
fields are little-endian; there are no struct padding bytes or pointers:

| Order | Type | Meaning |
| --- | --- | --- |
| 1 | 8 bytes | ASCII `NVF4B001` |
| 2 | six uint32 | H, N, capacity, layers, W13 layout, separate-up-alpha (0/1) |
| 3 | three float32 | pre-SiLU clamp limit, inverse input scale13, inverse input scale2 |
| 4 | repeated per layer | slab0..slab6, then FP16 input[H] |

Each slab stores all slots contiguously. Slab order: packed W13, packed W2,
128x4-swizzled W13 block scales, 128x4-swizzled W2 block scales, FP32 gate
alphas, FP32 down alphas, optional FP32 up alphas (zero bytes when disabled).
Per-slot byte sizes are respectively `N*H`, `H*N/2`,
`pad(2*N,128)*pad(H/16,4)`, `pad(H,128)*pad(N/16,4)`, 4, 4, and optional 4.
Both input-scale reciprocals and the clamp are currently shared across fixture
layers: export only layers with matching metadata, or benchmark separate files.
GEMM alphas must have their activation factors canceled using the header's
reciprocals; weight-only alphas use 1. Real GPU fixtures must use one of the
supported row-major/128x4 layouts; TRTLLM shuffles and Marlin are unsupported.

The reader validates dimensions, byte extents, finite scales/inputs, and
truncation/trailing bytes. Payloads above 2 GiB are refused. When loading a
fixture its metadata controls shape/layout; synthetic shape options are ignored.
Routing uses the first k host slots with equal FP32 weights `1/k`.

## Validation record

The setup is exercised with small synthetic cases, fixture write/read round
trips, multiple worker counts, reversed round order, layout variants and invalid
fixture/core checks. These are harness smoke checks, not performance results
from an isolated production-size run.
