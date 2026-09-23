# Engram served path: where the time goes — analysis handoff

Date: 2026-09-22. Branch `codex/nvfp4-expert-stream-main`, head `1d75e639b3`.
divix01 worktree `/data/models/slang/nvfp4-work/wt-p1bench`, left at `d337301dd1`
(pull it forward before new work).

**Read `DSV41_REFERENCE.md` sections 21 and 22 first.** This file is the operating
brief: what is settled, what the traces are, what to try next, and the traps that
have already cost GPU time. It does not repeat the evidence tables in those sections.

---

## 1. The one-paragraph state

The served engram path is **not** GPU-compute bound and **not** graph-structure bound.
Its two measured costs are a PCIe link that the host board caps at Gen3, and a large
number of tiny device-to-host readbacks that exist only to order host-side bookkeeping.
A third of the traced wall time is host Python that no capture has yet attributed,
because both traces ran without CPU sampling. Removing the last decode graph break
(layer 14, `d337301dd1`) worked exactly as designed and changed throughput by nothing
measurable — which is the strongest single piece of evidence for the paragraph above.

## 2. What is settled, and must not be re-derived

1. **The gather kernel is at line rate on a Gen3 link.** `_gather_host_rows_kernel`
   runs flat at 11.5–12.4 GB/s across a 1,700x range of transfer sizes, in two
   independent traces, carrying 187 GB and 736 GB respectively and accounting for
   **91% of all eager GPU kernel time** in both. `nvidia-smi` reports
   `gpumax=5, hostmax=3, current=3`: the RTX 5090 is Gen5-capable and the board is
   Gen3. This is the same wall already recorded in `MOE_EXPERT_TRANSFER.md`, "The link
   is at line rate, and the host caps it at Gen3", confirmed by the owner as hardware.
   **No batching, packing, coalescing or transfer-size change can win anything here.**
   A transfer-size sweep would only re-confirm the ceiling; do not run one.

2. **Graphing layer 14 removed a real break and bought nothing.** Graph launches per
   decode token went 2.000 → 1.000, exactly, measured against generated tokens.
   Token-weighted throughput went 2.777 → 2.724/2.720, a difference that does not
   resolve at n=1 against per-session spread of 2.157–2.891. Do not revert it — it is
   structurally correct — and do not look for the win by re-tuning the graph.

3. **Decode is host-blocked, not GPU-starved.** In the layer-1-only trace the CUDA
   thread spent 98% of decode wall inside two calls: 63% `cudaGraphLaunch` and 35% a
   2 KB device-to-host copy.

4. **The `io_uring` reader is the reader.** `SGLANG_MOE_EXPERT_FILE_READER` resolves
   to `uring_direct` in `server-env-actual.json`; the pre-`a50d7683cb` verdicts that
   said `mmap` were grading the harness process, not the server. Settled; see
   `DSV41_REFERENCE.md` section 20.

## 3. The open candidates, ranked

### A. The `.item()` sync — highest value, not yet attempted

`gather_rows` in `python/sglang/srt/layers/moe/expert_stream.py` syncs the stream on
the host once per admitted chunk, to order chunk N+1's admission against chunk N's
copy. Its own docstring states this. The cost, measured:

| trace | readbacks | avg size | host blocked |
|---|---:|---:|---:|
| 110 tokens | 20,017 | 31 B | 31.9 s |
| 485 tokens | 78,647 | 41 B | **64.1 s of 502 s wall** |

The GPU-side transfer is 0.7 µs. Everything else is the host waiting for the stream.
The ordering requirement is real, so this is not a delete — the question is whether a
CUDA event wait can enforce the same order without returning to the host. **Unevaluated.
Nobody has costed the replacement or checked whether the admission decision genuinely
needs the value on the host** (if the host branches on the hit count, an event does not
suffice and this is a redesign, not a fix). Establish that first.

### B. The unattributed 69 s of host Python

Half the layer-1 trace is the CUDA thread running host code with the GPU idle and OSRT
showing only 0.81 s of OS-level blocking. **Both traces ran `--sample=none`, so there
are no callstacks and this is genuinely unattributed.** Do not assume it is the engram
path; the eager expert gather is only 15.19 s of GPU time inside that window.

Next step is one traced arm with `NSYS_SAMPLE=cpu` (`run_arm.sh` already plumbs
`--sample`). This is cheap and is the single highest-information run available.

### C. Larger batch

`MOE_EXPERT_TRANSFER.md` already argues this is the remaining way at the idle link
window, because more tokens per forward amortize each miss row without needing
prediction. The served harness runs `--max-running-requests 1`. Untested here.

### D. Not worth doing

Prefetch, speculation and re-timing on the wire are closed on measurement in
`MOE_EXPERT_TRANSFER.md`, "The pattern": on a saturated link, re-timing or re-grouping
bytes does not help. The Gen3 finding above is why.

## 4. Artifacts

Traces on divix01, `/mnt/nvme1/dsv41-nsys/`. Analyse them **there**, under
`taskset -c 0-63`; a node-mode report once out-of-memoried the laptop.

| file | size | what |
|---|---:|---|
| `engram-on-trace-20260922-221602.nsys-rep` | 42 MB | layer 1 only, 110 tokens, 2 sessions |
| `engram-on-trace-20260922-221602.sqlite` | 172 MB | export of the above |
| `layer14-trace2-20260922-231638.nsys-rep` | 164 MB | layers 1+14, 485 tokens, 8 sessions |
| `layer14-trace2.sqlite` | 651 MB | export of the above |
| `layer14-trace-20260922-230112.nsys-rep` | 361 KB | **dead capture, delete it** (see §5) |

Both `.sqlite` files are derived and safe to delete; regenerate with
`nsys export --type sqlite -o <name>.sqlite <report>.nsys-rep` (~30 s).

Arm run directories, under
`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/`:

| arm | run | verdict |
|---|---|---|
| `engram-on-chunked2` | `run-20260922-210952` | failed, residency 5.53 GiB |
| `engram-on-trace` | `run-20260922-221546` | **void** — traced and capped to 2 sessions, gate refused it |
| `layer14-graph` | `run-20260922-224520` | failed, residency 3.03 GiB |
| `layer14-trace2` | `run-20260922-231623` | **passed** bar the acknowledged step-latency gap |

Only `layer14-trace2` passed its gate. Quote throughput from `engram-on-trace` for
nothing at all.

## 5. Traps that have already cost time

- **divix01's `/tmp` is too small for a full-length capture.** It is on the root xfs
  volume at ~88% full; nsys wants 200 MiB of scratch. The failure is not clean: nsys
  warns, then the server is killed mid-run, the driver reports `RemoteDisconnected`,
  and the report lands at ~361 KB. A 2-session capture fits and hides it. Always set
  `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp` for a traced arm.
- **A graph-mode report's kernel table omits the graph body.** No kernel row has
  `graphId > 0`; every kernel you can see is eager work. Never rank kernels from one.
- **`CUPTI_ACTIVITY_KIND_GRAPH_TRACE` in these reports describes the warm-up.** All its
  rows carry negative timestamps and correlation IDs below the captured window's
  minimum. Read naively it yields a confident per-execution average of something that
  is not in the trace, and sums to more than wall time.
- **`cudaMemcpyAsync` host time is not transfer time.** Join to the GPU-side record by
  `correlationId` and read `bytes` before calling anything bandwidth. In this workload
  the expensive ones move 31–41 bytes.
- **Do not infer decode steps from `alloc_decode_kernel`.** It fires twice per token in
  both builds. Count generated tokens from `results.jsonl` and divide.
- **`DSV41_MAX_SESSIONS` shortens the timed set and the result gate then refuses the
  arm** (`2 records (expected 8)`). Fine for a trace you only want per-unit costs from;
  fatal if you wanted a verdict.
- **The generation gate blocks an unregistered `python/` tree in preflight**, before any
  GPU is used. Register the new tree hash before launching, and write the key with a
  real JSON tool — a shell-quoting slip once baked literal quote characters into it.

## 6. Running an arm

```bash
# laptop
git push shared codex/nvfp4-expert-stream-main

# divix01
cd /data/models/slang/nvfp4-work/wt-p1bench
git fetch /data/models/slang/nvfp4-work/remotes/sglang-nvfp4.git codex/nvfp4-expert-stream-main
git checkout --detach <sha>
tree=$(git rev-parse HEAD:python)   # register this if new

cd benchmarks/dsv41_baseline
EXPECT_SHA=$(git rev-parse HEAD) NSYS_TRACE=1 NSYS_TMPDIR=/mnt/nvme1/nsys-tmp \
  ./run_arm.sh <arm-name> 7877 \
  SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1 SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=0
```

Drop `NSYS_TRACE`/`NSYS_TMPDIR` for an untraced arm. ~20 min untraced, ~25 traced.
`run_arm.sh` takes `cc-gpu.lock` itself. Never checkout the worktree while an arm runs.

Compare only against arms at the **same** `SGLANG_MOE_PINNED_HOST_MB`. The harness now
runs 51200; the 2.741/2.102 cells in `DSV41_REFERENCE.md` were measured at 71680 and are
not comparable. Every cell here is n=1 against per-session spread of roughly 2.1–2.9, so
**nothing under about 10% is resolvable without repeats** — budget three arms per cell.

## 7. Standing constraints

- Every CPU job runs under `taskset -c 0-63`, threads capped. Cores 64–71 stay free;
  core 71 is production's doorbell spin core.
- Code is written on the laptop, committed, pushed to `shared`, pulled on divix01.
  Never rsync/scp a tree there. Push to `shared` only, never `origin`.
- Mutants go in a private worktree and are never committed. `git checkout --` on a file
  with uncommitted work destroys it — that has happened in this repo.
- Print `sglang.__file__` before trusting any test result; the venv imports from
  `main-port-probe-7bc4eb` unless `PYTHONPATH` points at the tree under test.
- Read `${PIPESTATUS[0]}`, not the pipeline's status, for any suite piped to `tail`.
- Target `test/registered/unit/kernels`, not the whole tree, which fails collection on
  this box for pre-existing environmental reasons.

## 8. Open questions nobody has answered

- Why expert-shard page-cache residency grows across an arm at all, and why it varied
  5.53 → 3.03 → 0 GiB across three arms that read comparable volumes. It is the only
  check failing the gate.
- Whether the ~60 tokens/session corpus behind a 39–68 s TTFT explains the consistent
  0.69–0.71 ratio between served and Engine cells (`DSV41_REFERENCE.md` section 20).
- Whether the mirrored arm's 8.9% extra bytes is real or page-cache noise; one arm per
  cell, half an hour apart, no cache control.
