# Sleep-free lease wait: end-to-end A/B, 2026-09-28

**Verdict: no measurable effect, and none was possible under this recipe.** The base recipe runs piece
streaming with the copy engine, so every RAM-miss request takes post → hit_wait → stream → copy_wait → finalize.
None of the three rewritten waits run on that path: `exl3_ram_miss_wait_kernel`, `lease_wait` and
`lease_rest_wait`. The only change active in the production shape is the new CPU monitor thread. It costs about
2–3% of a core per `ExpertStreamDevice`, polling even while the server is idle. The outputs are byte-identical and
the latency delta is +0.2 ms/token, which is noise.

## Arms

| | A (baseline) | B (candidate) |
|---|---|---|
| Code commit | `b5d50d0734` (polling waits, `__nanosleep(256)`) | `565a76c9fc` (`codex/sleep-free-lease-wait` tip) |
| Arm branch / HEAD | `arm/sleepfree-base` / `ecb8123e85` | `arm/sleepfree-cand` / `32d9a21316` |
| `python/` tree (registered) | `c1fbc137a1f8` = `sleepfree-base-b5d50d0734` | `8244785c9bb5` = `sleepfree-cand-565a76c9fc` |
| divix01 worktree | `/data/models/slang/nvfp4-work/wt-sf-base` | `/data/models/slang/nvfp4-work/wt-sf-cand` |
| Run dir (`.../cc-expert-prediction/dsv41-baseline/servers/`) | `sleepfree-A-base/run-20260928-023125` | `sleepfree-B-cand/run-20260928-023555` |
| Overrides | none (base `arm_env.py`; identical at both commits) | none |
| `SGLANG_EXPERT_STREAM_URING_*` | unset | unset |
| KV tokens (`max_total_num_tokens`) | 77,824 | 69,632 |
| run_arm rc / verdict | 0 / valid except acknowledged step-latency gap | 0 / same |

Each arm branch carries one commit on its code commit, which registers the `python/` tree in `generations.json`.
The arms ran in the order A then B, one timed run each, with 2 timed sessions per arm (indices 0 and 1). Both used
port 30031.

## Results (`arm_metrics.py`, `paired.py`)

| Session | Tokens | A ms/token | B ms/token | Δ | A TTFT | B TTFT | SM clock at start A / B | Output |
|---|---|---|---|---|---|---|---|---|
| `cfq-train-Single_CDW/2015/page_35.pdf-2` | 7 | 127.5 | 127.6 | +0.1 | 8.60 s | 8.65 s | 2970 / 2970 MHz | byte-identical |
| `cfq-train-Single_ETR/2004/page_261.pdf-1` | 85 | 101.9 | 102.1 | +0.2 | 8.06 s | 8.10 s | 2970 / 2970 MHz | byte-identical |
| **Pooled** | 90 | **103.6** | **103.8** | **+0.2** | 8.33 s mean | 8.37 s mean | | |

- Both sessions start at the same SM clock (2970 MHz) in both arms, so both sessions are comparable. Session 0 has
  only 6 inter-token gaps, so session 1 (84 gaps) carries the pooled figure.
- `paired.py` A→B: B/A tok/s ratio 0.9986 in both sessions, median delta −0.012 tok/s, 0/2 B wins, sign-test
  p = 1.0. Two sessions cannot establish significance, but the delta is ~0.2%, far under the 1 ms/token noise floor.
- Client-side gaps: p50 102.7 vs 102.2 ms, p90 141.7 vs 141.5 ms, max 191.4 vs 192.5 ms. Neither arm had a stall
  of 0.5 s or more.
- Server CPU per session (harness `cpu.jsonl`, process tree): A 9.84 s and 30.81 s, B 10.12 s and 31.27 s, so B
  used 0.28 s and 0.46 s more. That matches two monitors at ~2.5% each over the sessions' wall time.

## Why the rewritten waits are not exercised

`Exl3RamMissRowBackend.post` (`python/sglang/srt/layers/moe/exl3_ram_miss.py:550-608` at `565a76c9fc`) calls
`rest_wait` only when `piece_stream` is off. The recipe sets `SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE=1`,
`..._PIECE_STREAM=1`, `..._COPY_ENGINE=1` and `..._LEASES=1`, so both prefill and decode take the piece-stream
branch, which uses `hit_wait`, `stream` and `copy_wait`. The branch's plan keeps "bounded hit-stage polling and piece
streaming unchanged". The one-phase `wait`/`lease_wait` path runs only without two-phase.

To measure the new protocol on its critical path, rerun both commits with
`SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM=0`, which makes W2 (`lease_rest_wait`) the wait. That is a non-production
shape, so it is a separate experiment and was not run here.

## Monitor-thread CPU (B)

The per-thread `/proc/<pid>/task/*/stat` sampler ran every 2 s across the server tree. The timed window was
28.7 s, and the whole tree used 42.8 s of CPU (1.49 cores) in that window. Two threads show the monitor's
signature:

| tid | first seen | CPU over timed window | utime / stime | Rate while idle (pre-health) |
|---|---|---|---|---|
| 423611 | with the RAM-miss service thread | 0.68 s (2.4%) | 0.31 / 0.37 s | 2.1% |
| 425041 | ~35 s later | 0.83 s (2.9%) | 0.41 / 0.42 s | n/a (not yet started) |

Both run at a flat 2–3% whether the server is idle or decoding, and split evenly between user and system time,
which fits a `sleep_for(25us)` loop. The monitor thread is unnamed, so it shows the scheduler's comm; the
identification is by this signature, not by name. The threads exist only in B, because A has no wait-completion
module. Together they cost about 5% of one core. For scale, the RAM-miss service thread took 25.4% and the copy
engine 34.7% in the same window. There is one monitor per `ExpertStreamDevice`, so two threads suggests two devices were constructed (inferred, not counted).

The A sampler recorded nothing: it attached to the dead server pid of the failed first attempt (below), so there
is no A per-thread baseline. The harness's per-session `cpu_s` gives the A/B comparison above.

## Device counters

The server logs no kWaits, kTimeouts or kPolls in either arm: nothing in the serving path logs
`ExpertStreamDevice.stats()`. The host service's shutdown counters are comparable: served 2904 vs 2913, rows_read
4080 vs 4144, read_errors 0/0, late_after_fatal 0/0, copy_errors 0/0, copy_fallbacks 0/0,
copy_generation_mismatches 0/0, slots_quarantined 0/0, spin_cpu 48 vs 38.

## Timeouts, fatals, refusals, validity

- No lease timeouts, fatals or tracebacks in either server log. No compile events during the timed set.
- **The first A attempt died at startup.** `run-20260928-022439` hit `RuntimeError: The DSV4 SWA pool cap (10752
  tokens, 0.23 GB) leaves no room for the full KV pool within the available 0.22 GB`. That process saw 29.24 GB free
  at weight-load start, against 29.41 GB on the retry (and 29.38 GB for B). GPU tenancy was the same 62 MiB each
  time. The recipe's KV budget has only tens of MB of slack, so ~170 MB of run-to-run variation in pre-load free
  VRAM is enough to refuse a launch. The retry, unchanged, succeeded.
- Verdict for both arms: Task 1's strict definition is False because of the acknowledged gaps (no engine-side
  step latency, provenance `sglang_file` unresolved, `FILE_READER`/`GRAPH_GATHER` resolve notes). "Valid except the
  acknowledged step-latency gap" is True for both. CONTENDED notes: the scheduler at ~100% on its own core (expected),
  and an `htop` at 62% on core 55, outside the server's cores, during one session of each arm.
- Before each arm, the driver was 615.71.09 per both `nvidia-smi` and `/proc/driver/nvidia/version`. Port 7867 was
  down. No foreign pytest, cc1plus, nvcc, cicc or ptxas was running (matched by comm, exe or exact argv element).

## Commands

```bash
# Sanity check for B, in wt-sf-cand: 5 passed (sglang from wt-sf-cand/python)
export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 MAX_JOBS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python \
  -m pytest test/manual/dsv41/test_expert_stream_sleep_free_cuda.py -q -p no:randomly

# Each arm, via /mnt/nvme1/sf-ab/arm.sh (driver/port/foreign-process/GPU-lock checks, then the thread sampler):
cd /data/models/slang/nvfp4-work/wt-sf-{base,cand}
EXPECT_SHA=$(git rev-parse HEAD) OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/rowimg-disk.lock \
  taskset -c 0-63 bash benchmarks/dsv41_baseline/run_arm.sh sleepfree-{A-base,B-cand} 30031

# Metrics (from wt-sf-cand)
python analysis/dsv41-drive/final-arms/arm_metrics.py A=<A run dir> B=<B run dir>
(cd benchmarks/dsv41_baseline && python paired.py <A run dir> <B run dir>)
python /mnt/nvme1/sf-ab/threads_report.py /mnt/nvme1/sf-ab/sleepfree-B-cand-threads.jsonl 2 25
```

The driver logs, thread samples and `arm_metrics.json` are in `divix01:/mnt/nvme1/sf-ab/`.
