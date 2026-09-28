# NUMA placement results on divix01

The GPU and its PCIe root port are on node 0 (CPUs 0-17, 36-53). All four NVMe drives (nvme0-nvme3, which hold every
mirror) are on node 1 (CPUs 18-35, 54-71).

## 1. RAM-miss service thread: node 0 vs node 1 (2026-09-27)

**Question.** Does pinning the RAM-miss service thread on node 1, the drives' node, instead of node 0, the GPU's
node, change decode ms/token? A harness measurement had suggested it might. The "mixed" host-read-bound scenario in
`../chain-pdl/results.md` (branch `expert-stream-transfer-measurement`) took ~790 us/layer with the thread on core 40
and ~475 us with it on core 54.

**Answer.** No. B - A = +0.3 ms/token on the median, inside each arm's own spread and under the 1 ms/token noise floor.

### Setup

- The setting is `SGLANG_DSV41_RAM_MISS_SERVICE_CPU` (this branch, 85f4ad6962). It defaults to -1, which leaves the
  thread unpinned. The thread's actual core is the `spin_cpu` counter (kSpinCpu), logged at shutdown.
- The harness is `benchmarks/dsv41_baseline/run_arm.sh`. It runs untraced, with a sequential driver and 2 timed
  sessions after its readiness warm-up. The code is this branch at 2b5a183e1a (generation `service-cpu-85f4ad6962`).
- **The recipe is the production recipe with an even NUMA split, `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:51200,1:51200`.
  This is NOT the production recipe.** Production's `0:61440,1:40960` did not fit node 0 while co-tenants held its
  page cache, and the user chose the even split for both arms. Compare these ms/token figures A against B only, never
  with recorded recipe numbers.
- Arm A pins the thread to core 53 (node 0; its hyperthread sibling is 17). Arm B pins it to core 54 (node 1; its
  sibling is 18). Both siblings were at least 98.5% idle before launch.
  - Core 40 was dropped because its sibling, core 4, is a would-be packing-worker CPU.
  - With row images on, production runs no packing pool at all: no `exl3-pack` threads exist.
- Every arm's gate check resolved EXL3 requirements (not the NVFP4 fallback). Every server logged
  `exl3 RAM miss thread started ... row images on, copy engine on`.

### Runs

| run | core | kSpinCpu | pooled ms/token | long session (85 tokens) ms/token | outputs vs A1 |
|---|---|---|---|---|---|
| A1 | 53 | 53 | 109.5 | 107.4 | - |
| B2 | 54 | 54 | 109.9 | 107.9 | byte-identical |
| A3 | 53 | 53 | 109.0 | 107.1 | byte-identical |
| B4 | 54 | 54 | 185.0 (excluded) | 184.4 | byte-identical |
| B5 | 54 | 54 | 108.7 | 106.6 | byte-identical |
| A6 | 53 | 53 | 108.4 | 106.3 | byte-identical |

- A /proc watcher confirmed each run's placement. The service thread's allowed set was `{53}` or `{54}`, and its
  last CPU matched.
- The short session (7 tokens) is 136.5-138.7 ms/token in every clean run.
- **Why B4 is excluded.** Another lane's CPU test suite ran during B4. That suite was the `dbl/head-kernels` pytest,
  with JIT `cc1plus` under `taskset 0-63` and its own `exl3-ram-miss` thread. B4's two sessions both ran ~185
  ms/token, against ~109 elsewhere. B5 and A6 replaced it, each started only with no pytest or cc1plus running on the
  box. The watcher saw no foreign thread after 22:49:23, and B5 started at 22:50:41. The run order was A B A B(x) B A.

### Verdict

| arm | runs | median ms/token | spread |
|---|---|---|---|
| A (core 53, node 0) | 109.5, 109.0, 108.4 | 109.0 | 1.1 |
| B (core 54, node 1) | 109.9, 108.7 | 109.3 | 1.2 |

B - A = +0.3 ms/token, which is noise. The harness's node effect does not appear in serving decode, at least at the
even split with row images on.

### Where the data is

The run directories are
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/svccpu-evensplit-notprod-*`.
Metrics were produced by `analysis/dsv41-drive/final-arms/arm_metrics.py` and saved as
`divix01:/mnt/nvme1/tmp-ests/service-cpu/arm_metrics.json`. The thread watcher's log is `threads-all.log` in the same
directory.

## 2. GPU read of node-0 vs node-1 pinned memory

See `node-read-probe.md`.
