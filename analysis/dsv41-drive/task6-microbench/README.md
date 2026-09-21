# Task 6 microbenchmarks: `T(n)` and `g`

Two GPU microbenchmarks that decide the per-row-versus-two-phase verdict of Task 6
(`docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`). Both quantities were assumed, not measured.
The plan says the margin is `k`-free (1.1-1.4% of step time against a 1.5% resolution, `G* = 1.114 ms`), so no lane
measurement can settle it; only these two can.

**Read this first: neither design is new.** Both already had pre-registrations in this directory's parent, and the brief this
work started from predates them. What this directory adds is the missing pieces:

| quantity | pre-registration | what already existed | what this directory adds |
|---|---|---|---|
| `T(n)`, `delta` | `../C_MEASUREMENT_PREREG.md` | a frozen harness `../c_measurement/c_harness.py` and analysis; run 1 on 2026-09-21 completed and is **INVALID** (its P-state gate demands P0, the card idles at P1; section 20 leaves that to the lead) | `gather_tn.py`: a gate-free driver that imports the frozen harness unmodified and reports `delta` in us and ms/step |
| `g` | `../G_MEASUREMENT_PREREG.md` | a frozen analysis `../g_measurement/g_analysis.py` (hash-registered) and no harness; the run was cancelled in favour of measuring `g` on real kernels inside arm A3 | `g_harness.py` + `g_kernels.cu`: the harness section 4 specifies, writing the JSONL `g_analysis.py` reads |

Nothing here edits the plan, `PER_ROW_TRANSFER.md`, the frozen files, or either pre-registration. Anything a run of these
scripts prints is **not a registered result** unless it went through the frozen analysis and passed its gates.

## 1. `T(n)`: `gather_tn.py`

**What is measured.** The GPU-timeline time of one launch of the production gather,
`copy_expert_row_segments_gpu` (`python/sglang/kernels/ops/moe/expert_cache_transfer.py`, grid 8 x 256), moving `n = 1..6`
rows of 13,315,584 B (the six real segments) from pinned host slabs into VRAM slots. Per-row transfer launches one row per
request, so the cost of a count-1 launch against the batched launches is what matters, not one bandwidth figure `c`
(`PER_ROW_TRANSFER.md` OPEN 1; `C_MEASUREMENT_PREREG.md` section 0).

**What it reports.**

* `T(n)` at p50, min and p90, with implied GB/s beside the link spec (15.75 GB/s Gen3 x16) and flagged IMPOSSIBLE above it.
* A line `T(n) = f + c_m * n` fitted on the *batched* counts `n = 2..6` only and extrapolated to `n = 1`. **`delta = T(1) -
  (f + c_m)`** is what a count-1 launch costs beyond what the batched launches predict. It is the headline because per-row's
  exposed copy after the last row arrives is `T(1)` while two-phase's is `T(m)`, and the plan books the difference as
  `(m-1) * c`; `delta` is the shortfall of that booking, per read request.
* The other reading of OPEN 1's wording, `T(1) - T(k)/k` for k = 2..6, beside it. It includes `f/k`, so it is larger
  whenever there is a fixed launch cost; both are printed so the reader can choose, and the choice is stated where used.
* Both carried to ms/step with **13.97 read requests per step**, against `G* = 1.114 ms` (`delta = 80 us` is 1.1 ms/step).
* Bootstrap 95% interval over launches, and the per-pass values, so run-to-run agreement is visible.
* Controls at every n: `repeat` (the same rows every launch, the L2 control: if it is not visibly faster than `cold`, either
  L2 does not hold host reads on this part or the harness cannot see it), `ce` (`cudaMemcpyAsync`, the copy-engine yardstick),
  and eager versus graph launch (production runs it as a graph node).

**Why it is trustworthy, and why it uses the frozen harness.** `c_harness.py`'s `RealDevice` already does what the brief's
traps demand: pinned slabs from the service's own allocator, a permutation ring so no row is re-read within 150 rows
(2.0 GB = 20.9 x the 96 MiB L2; the plan quotes ~128 MB, the measured L2 is 96 MiB), 64 destination slots, a 600 us spin
before each timed launch so host launch latency stays outside `T`, and a copy check that the intended rows moved. It is
imported, not copied, so it cannot drift; its sha256 is recorded in `meta.json`.

**What it does differently from `c_harness.py`, and what that costs.** No gates: clocks, P-state, link generation, other GPU
processes and load average are recorded and printed, never used to refuse. That is the point (it runs on today's card), and it
is also why the output is not a registered `c`. It measures graph launch at every n, not only n = 3. It does not run the
`hot` or `nvme` arms.

```text
CC=/data/models/slang/nvfp4-work/cc-expert-prediction
W=/data/models/slang/nvfp4-work/cc-microbench          # a detached worktree of dsv41-microbench, made through git
cd $W
export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
$CC/analysis/dsv41-phase3b/gpu-run.sh /data/models/slang/.venv/bin/python \
    analysis/dsv41-drive/task6-microbench/gather_tn.py run --repo $W --out <outdir> --nodes 0 --passes 5
python3 analysis/dsv41-drive/task6-microbench/gather_tn.py analyse <outdir>/results.jsonl.gz      # CPU, stdlib only
```

`--nodes 0,1` adds the second NUMA node (2.0 GB of pinned memory each; the GPU is on node 0, production's slab placement is
unknown, `TOPOLOGY.md`). `--check-only` does allocation, registration and the copy check and exits.

## 2. `g`: `g_harness.py`

**What is measured.** `g = d(T_replay)/dN`, the increase in the replay time of a captured graph per additional stage triple
in series, in microseconds (`G_MEASUREMENT_PREREG.md` 1.2). A triple is `W_s`, `C_s`, `A_s`: three dependent kernel nodes.

**Empty and active are measured separately, on purpose.** "An empty triple is three launches and no host read; an active
one pays poll round trips, an ack fence and the copy path." A single blended `g` is the wrong model.

| variant | `W_s` | `C_s` (production copy kernel) | `A_s` |
|---|---|---|---|
| `empty` (`g_e`) | empty-range exit, `go = 0`, no host word read | `count = 0` | acknowledges nothing |
| `active_p{1,4,6}` (`g_a(p)`) | `p` serial `ld.acquire.sys` loads of mapped host words (each a PCIe round trip), `go = 1`. **Registered `p = 4`** | `count = 1`, a 4 KiB row | `__threadfence_system()` + `st.release.sys` to a distinct mapped line |
| `control20` | `empty` plus a 20 us spin in `W_s`: the **positive control**, the slope must rise by 20 +- 1 us | as empty | as empty |
| `empty_base8k` | `empty` with 8,000 filler nodes underneath: the slope must hold at production graph size | | |

Sweep `N in {0,40,80,160,320,640}` (base8k: `{0,160,640}`), 20 event-timed batches of 50 replays per cell, cell order
randomised, **3 processes** (`--process 0/1/2`). Each run first proves the graphs do what they claim: an active chain of 3
triples has 9 nodes, leaves `go = 1` and writes 3 acks; an empty one has 9 nodes, leaves `go = 0` and writes no ack.
Node counts are read from the graph (`cudaGraphGetNodes`) and gated by the frozen analysis (`nodes == 3N`).

**How it becomes a number for Task 6.** The frozen `../g_measurement/g_analysis.py` prices the two kinds separately, with the
plan's exposure counts (`../g_measurement/g_exposure_counts.py`, measured lanes of `task1f`, per decode step):

```text
of 159.55 extra triples per step:   85.10 empty tail stages           always exposed   -> g_e
                                    48.93 extra active, no-read layers always exposed   -> g_a
                                     4.65 extra miss-row stages        always exposed   -> g_a
                                    20.86 hit-lane stages, read layers HIDE behind the read wait (class C)
G_X (all exposed) = (85.10 g_e + (48.93 + 20.86 + 4.65) g_a) / 1000     ms/step
G_H (C hidden)    = (85.10 g_e + (48.93         + 4.65) g_a) / 1000     ms/step
per-row's net at best order = 4.93 - G  (gross 4.93 ms is proportional to c, measured by section 1)
G* = 4.93 - 0.015 x 254.4 = 1.114 ms; uniform-g crossing 6.98 us (all exposed) / 8.03 us (C hidden)
```

```text
CC=/data/models/slang/nvfp4-work/cc-expert-prediction; W=/data/models/slang/nvfp4-work/cc-microbench; O=<outdir>
cd $W; export PYTHONPATH=$W/python CUDA_HOME=/usr/local/cuda-13.2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 analysis/dsv41-drive/task6-microbench/g_harness.py build            # CPU, no lock: compiles g_kernels.cu with nvcc
for p in 0 1 2; do      # one lock hold per process is fine; three processes so between-process variance is in the interval
  $CC/analysis/dsv41-phase3b/gpu-run.sh /data/models/slang/.venv/bin/python \
      analysis/dsv41-drive/task6-microbench/g_harness.py run --repo $W --out $O --process $p
done
python3 analysis/dsv41-drive/g_measurement/g_analysis.py $O/results.jsonl   # the frozen verdict; INVALID means quote nothing
python3 analysis/dsv41-drive/task6-microbench/g_harness.py summary $O/results.jsonl   # slopes and G even when INVALID: preliminary only
```

Two additions beyond the registered design, both reported separately and neither seen by the frozen analysis:

* **Poll-latency probe** (every process from `925354fff4` on, printed and stored in `meta_process*.json`): mean latency of serial `ld.acquire.sys`
  loads of the pinned request page versus a device word. It shows whether the stand-in `W_s` poll really leaves the GPU (a PCIe
  round trip) or is served from L2 (which would understate `g_a`).
* **`--ext`** (unregistered variants, written to `results_ext.jsonl`): empty and `active_p4` triples on top of 1000, 2000, 4000
  and 8000 filler nodes, N = 0, 160, 640. It answers whether the per-triple cost depends on the size of the graph it sits in (the
  plan's decode graph has ~8,000 nodes); the registered `empty_base8k` variant only covers the empty triple at 8,000.

`--keepalive` (off by default, recorded) sends tiny H2D copies during cells; use it only if the link idles to Gen1 during the
empty cells and the frozen link gate then refuses (see results.md for what was seen).

## 3. Layout of this directory

```text
README.md        this file
results.md       preliminary readings from 2026-09-21, their card conditions, and what is stale in the brief
gather_tn.py     T(n) and delta                 g_harness.py, g_kernels.cu     the stage-triple harness
data/tn_prelim/  raw T(n) samples (.gz), meta, and the analysis text
data/g_prelim/   raw g cells (.gz) for 3 processes (results and results_ext), meta per process
data/probe/      the poll-latency probe's own meta
```

## 4. The traps, and how each is handled

* **L2, not HBM.** `T(n)`: ring of 150 rows = 2.0 GB per node, minimum reuse distance recorded per cell, `repeat` control at
  every n, implied GB/s checked against 15.75 GB/s and against the copy engine. `g` moves no data (4 KiB or nothing), so L2
  does not enter it; that is deliberate. Note the CLAUDE.md says ~128 MB, the measured L2 of this part is 96 MiB.
* **Percentiles need an exclusively held card.** Every cell records min next to p50/p90 (`T(n)`) or all 20 batch times (`g`),
  the load average, and the other GPU processes seen by `nvidia-smi` at 250 ms. `nvidia-smi --query-compute-apps=pid,used_memory
  --format=csv` before and after. `gpu-run.sh` holds `cc-gpu.lock`; it does not stop a process that is already on the card, so
  say whether any was.
* **The interpreter trap.** `/data/models/slang/.venv/bin/python` imports sglang from an unrelated tree unless
  `PYTHONPATH=$PWD/python` names the tree under test. Both scripts refuse when `sglang.__file__` is not under `--repo`, and
  record `sglang_file` and the git HEAD.
* **The link idles at Gen1.** Both harnesses do untimed warm-up load and record the link generation per cell; the frozen
  `g_analysis.py` gates on it.
* **Never rank kernels or read step-tail idle from a graph-mode trace.** Nothing here uses Nsight; both quantities come from
  CUDA event timing.
* **CPU jobs on divix01** run under `taskset -c 0-63` with `OMP_NUM_THREADS=1`; `gpu-run.sh` pins to 32-63. Cores 64-71 stay
  free (71 is production's doorbell spin core).

## 5. What these do not settle

* **`g` with stand-ins.** `W_s` and `A_s` are written to the contract of `PER_ROW_TRANSFER.md` 5.5, not the production kernels of
  a per-stage protocol (none exists: that is Task 6's work). The stand-ins bound `g`. Ready-at-launch is the best case for polling:
  a wait that really waits adds a detection delay this run never sees (`G_MEASUREMENT_PREREG.md` L10).
* **Exposure (class C).** Whether the 20.86 hit-lane stages hide is a property of a real decode, not of a synthetic graph;
  `EXPOSURE-DEPENDENT` is a verdict of the frozen rule for exactly that reason.
* **Both results feed one inequality.** `g` and `c` both move the crossing, `c` more (10% of `c` is about 3 us of `g*`). Read
  both before saying anything about Task 6.
