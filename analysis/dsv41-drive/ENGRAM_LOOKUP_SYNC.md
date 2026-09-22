# The engram file-table lookup is the served path's largest host stall (2026-09-22)

**Status: measured, not fixed.** No code change has been made for this. The finding is that
one `.cpu()` inside the breakable decode graph accounts for **76.6 s of host block in a
419 s window** on the HTTP serving path, and that this is the same time already visible as
"long `cudaGraphLaunch` calls" counted from the other side.

Read `## What is not established` before quoting any of this as a speed-up opportunity.

## The call site

`python/sglang/srt/layers/engram_file_table.py:105`

```python
def lookup(self, indices: torch.Tensor) -> torch.Tensor:
    assert_not_capturing("EngramFileTable.lookup")
    flat = indices.reshape(-1).cpu().numpy()      # <- this line
```

(`_engram_file_table_lookup` is `python/sglang/srt/layers/engram.py:73`.)

reached, every decode step, through the eager break of the breakable decode graph:

```
decode_cuda_graph_runner.execute
  breakable_cuda_graph_backend.replay -> breakable_cuda_graph.replay -> replay_fn   (:269)
    engram.py:73  _engram_file_table_lookup
      engram_file_table.py:105   indices.reshape(-1).cpu().numpy()
```

**This is by design and the design says so.** `_engram_file_table_lookup` carries
`@eager_on_graph(True, capture_stub=_engram_lookup_capture_stub)` and its docstring reads
"The file-table lookup reads its ids on the host, so under a breakable CUDA graph it runs
as an eager break". What was not known is what it costs. It is also why decode graphs on
this configuration are breakable with `segments=3, breaks=2` at all.

## Measured

Two independent instruments agree.

### Nsight, the leases-on arm (419.4 s window, 4 timed sessions)

Report `/mnt/nvme1/dsv41-nsys/phase1-leases-20260922-041258.nsys-rep`, exported alongside
it as `.sqlite`.

| device-to-host copies | calls | host block | device copy time | bytes |
|---|---:|---:|---:|---:|
| destination **pageable** | 3,815 | **76.6 s** | 1.73 ms | 1.65 MB |
| destination pinned | 37,736 | 0.39 s | 13.04 ms | 158 KB |

**7.5 % of the copies carry 99.5 % of the block**, and the separator is the destination
being pageable. The 37,736 pinned copies are innocent at 10 us each; they are cub's
internal result counts (`DeviceSelect::Flagged` / `DeviceReduce::Sum`, 36,785 NVTX ranges
each, which is what identifies them).

The expensive ones concentrate hard: **601 copies of 192 B cost 43.9 s (73 ms each)** and
**818 of 256 B cost 25.3 s (30.9 ms each)** -- 1,419 calls, 69.2 s. All on one thread, one
stream, spread evenly from 1.4 s to 420 s, so per-step rather than a startup artifact.

The mechanism is ordinary CUDA semantics: **a `cudaMemcpyAsync` whose destination is
pageable is synchronous.** The driver drains the stream before it can stage the copy, so
each of these tiny reads is a full device synchronization. `.cpu()` allocates a pageable
destination.

### The backtrace capture (2 sessions)

Report `/mnt/nvme1/dsv41-nsys/btrace-memcpy-20260922-144026.nsys-rep`, exported as
`/mnt/nvme1/dsv41-nsys/btrace.sqlite`. 1,028 blocking copies, 60.4 s, bursty (minimum gap
0 ms, mean 229 ms). Frame 22 of every captured stack is
`torch::autograd::THPVariable_cpu`, which is specifically `tensor.cpu()` -- not `.item()`
(`_local_scalar_dense`) and not `.tolist()`.

### py-spy on the scheduler process (90 s at 100 Hz, decode)

`cc-expert-prediction/analysis/engram-sync/pyspy-scheduler-20260922.raw`.

| share | leaf frame |
|---:|---|
| **53.86 %** | `lookup (engram_file_table.py:105)` |
| 22.95 % | `read (exl3_shard_row_source.py:198)` |
| 7.93 % | `_apply_streamed (quantization/exl3.py:497)` |
| 4.14 % | `read (uring_file_reader.py:140)` |
| 0.96 % | `decide_residency_policies (expert_residency.py:471)` |

The 53.86 % is the leaf, i.e. time inside line 105 itself, and its stack runs through
`decode_cuda_graph_runner.execute`, so these are decode samples rather than prefill.

## What this reframes

The "77 s on `cudaMemcpyAsync`" and the "278 `cudaGraphLaunch` calls over 100 ms carrying
82.8 s of the 84.25 s total" recorded earlier are **the same phenomenon counted from two
sides**, not two independent problems. Each decode step breaks its graph, drains the
pipeline to bring engram indices to the host, does a host-side numpy gather against the
file table, and pushes the rows back. With one CUDA thread the host cannot run ahead
while that happens.

## What this is NOT

Recorded because two plausible hypotheses were wrong, and acting on either would have
wasted the effort:

1. **Not `expert_distribution.py`.** `on_forward_pass_start` does
   `input_ids.cpu().tolist()` twice per pass, which is the same shape of mistake -- but
   that gatherer is selected by `per_token` and the arms run `per_pass`, which maps to a
   different class entirely.
2. **Not `expert_stream.py`'s hot-cache lookup.** Its `.tolist()` / `.item()` /
   `slots.cpu()` were the leading hypothesis for several steps. `slots.cpu()` at
   `expert_stream.py:374` is inside the `output.device.type == "cpu"` branch, which this
   configuration never takes.

The distinction matters for what a fix looks like. Those would have been gratuitous
bookkeeping, deletable. This one is load-bearing: `EngramFileTable` gathers rows from a
**file** through numpy, so the indices genuinely have to reach the host.

## What is not established

- **76.6 s of host block is not 76.6 s of recoverable wall time.** It is a serialization
  cost: the host cannot enqueue the next step while it waits. How much is recoverable
  depends on what the GPU could have been doing instead, which is not measured. Do not
  quote "18 % of wall time" as a projected speed-up.
- The nsys figures come from **n=1 arms**, one leases-on and one 2-session diagnostic.
- The 419 s trace is graph-mode, so its kernel table omits the graph body (`CLAUDE.md`).
  Nothing here is read from that table.
- The py-spy sample is a single 90 s window of one arm.
- Whether the engram working set would fit on device, or how often indices repeat across
  steps, is **not measured**. Both decide whether any of the options below are viable.

## Options, none of them tried

Roughly in increasing order of invasiveness. Each is a design change, not a bug fix, and
each should be measured against the recorded served-path baseline (`DSV41_REFERENCE.md`
section 20: 2.102 tok/s token-weighted, leases off, mirrors off).

1. **Pinned staging.** Copy indices into a reusable pinned buffer with
   `non_blocking=True` instead of `.cpu()`. Removes the implicit drain but still needs a
   sync before numpy can read the buffer, so this only helps if the sync can be deferred.
2. **Prefetch the indices one step ahead**, so the transfer overlaps the previous step's
   compute. Needs the indices to be known a step early, which they may not be.
3. **Keep hot engram rows resident on device** and fall back to the file table only on a
   miss. This is the same shape as the expert hot cache already in the tree, and would cut
   the break out of most steps rather than making it cheaper.

## Reproducing

GPU work goes through `run_arm.sh`, which takes `cc-gpu.lock`. Cores 64-71 stay free.

```bash
# the backtrace capture (needs bbd2418c36 or later for the two knobs)
cd /data/models/slang/nvfp4-work/wt-p1bench/benchmarks/dsv41_baseline
EXPECT_SHA=<head> NSYS_TRACE=1 DSV41_MAX_SESSIONS=2 NSYS_SAMPLE=process-tree \
  NSYS_LAUNCH_ARGS='--cudabacktrace=memory:10000000 --python-backtrace=cuda' \
  ./run_arm.sh btrace-memcpy 7877 SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=0
```

A shortened arm fails the result gate by design (`2 records (expected 8)`); the report is
written before that gate runs, so the abort is expected and harmless.

Traps, all of them hit on the way to this result:

- **`--cudabacktrace` and `--python-backtrace` are application scope**: they go on
  `nsys launch`, not `nsys start`, and are silently inert unless CPU sampling is on, which
  this harness disables by default. Hence `NSYS_SAMPLE`.
- **`--python-backtrace=cuda` produced no table anyway.** The native stack cannot
  substitute: CPython 3.11+ inlines Python-to-Python calls, so an arbitrarily deep Python
  chain collapses into two `_PyEval_EvalFrameDefault` frames. This is why py-spy was
  needed at all.
- **py-spy must target the `sglang::scheduler` process, not the launcher.** The launcher
  is an idle uvicorn; the decode loop is in the scheduler. A dump of the wrong process
  looks like a working profile of a process doing nothing.
- **py-spy `--nonblocking` failed** with "Failed to find python version from target
  process" against this Python 3.13; ordinary mode works.
- The 10 ms `--cudabacktrace` threshold is deliberate: the copies of interest are 31-73 ms
  and the innocent ones are 10 us, so the threshold separates them by a factor of 1,000
  and keeps both overhead and report size down.

## Artifacts

| what | where (divix01) |
|---|---|
| leases-on arm trace, 419 s | `/mnt/nvme1/dsv41-nsys/phase1-leases-20260922-041258.nsys-rep` (+ `.sqlite`) |
| backtrace capture, 2 sessions | `/mnt/nvme1/dsv41-nsys/btrace-memcpy-20260922-144026.nsys-rep` |
| its export | `/mnt/nvme1/dsv41-nsys/btrace.sqlite` |
| py-spy scheduler profile | `cc-expert-prediction/analysis/engram-sync/pyspy-scheduler-20260922.raw` |
| arm run dirs | `cc-expert-prediction/dsv41-baseline/servers/<arm>/run-<stamp>/` |

The two `.sqlite` exports are 455 MB and 651 MB. Analyse them on divix01 under
`taskset -c 0-63`; do not copy them to the laptop, and do not load either under `/tmp`
there (`CLAUDE.md`: a RAM-backed tmpfs, and a 250 MB report once out-of-memoried the
laptop).
