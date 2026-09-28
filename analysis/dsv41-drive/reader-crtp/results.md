# Reader CRTP split: refactor-only decode pair (plan 2026-09-28-reader-crtp-uring-registration, Task 5)

**Verdict: PASS.** Output is byte-identical on every timed turn. Pooled ms/token is 102.40 (base) against 102.56
(split), so the split is +0.16 ms/token, well inside the ±1.5 bar. Both arms show `read_errors` = 0.

## What ran

| | base | split |
|---|---|---|
| arm | `crtp-base` | `crtp-split` |
| worktree (divix01) | `/data/models/slang/nvfp4-work/wt-reader-crtp-base` (detached) | `/data/models/slang/nvfp4-work/wt-reader-crtp` (detached) |
| commit | `4fe0c37a41` | `0ab87b35cc` (python/ as at `d2e459d421`) |
| python tree | `c2de63027560943a1bb4f33056b9c4b50126578c` | `00092c03d5d5c6c0fbcd93b8d8890261cac922fe` |
| generation label | `mirror3-base-4fe0c37a41` | `reader-crtp-split` |
| run dir | `servers/crtp-base/run-20260928-143215` | `servers/crtp-split/run-20260928-143642` |
| run_arm.sh rc / verdict | 0 / valid except the acknowledged step-latency gap | 0 / valid except the acknowledged step-latency gap |

- Run dirs are under `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/`. The driver's output is in
  `divix01:/mnt/nvme1/reader-crtp/`: `driver.log`, `pair-report.json`, `node0-gate.jsonl` and the clock CSVs.
- Both arms ran through the split worktree's `run_arm.sh` and recipe (`arm_env`), with `DSV41_WORKTREE` and
  `EXPECT_SHA` set to each arm's own worktree and commit.
- Overrides were identical in both arms. The nine io_uring knobs were set to their S0 values:
  `SGLANG_EXPERT_STREAM_URING_{MODE=default, QUEUE_DEPTH=0, FIXED_FILES=0, READ_MODE=normal, WAIT_MODE=block,
  SQ_THREAD_IDLE_MS=1000, SQ_THREAD_CPU=-1, SLAB_ARENA=0, DIAGNOSTICS=1}`. The pinned tier was
  `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:40960` and `SGLANG_MOE_PINNED_HOST_MB=98304`.
- Each server ran its own tree. The split server logged its `expert stream io_uring: mode=default read_mode=normal ...
  fixed_files=0 fixed_buffers=0` line once, and the base server (whose tree has no such knobs) logged none.
- **EXL3 gate.** Before each arm, the driver resolved `expert_stream_requirements_for` with that arm's own tree against
  `arm_env.MODEL_PATH`. Both resolved to `EXL3`. The same check on a nonexistent model path resolves to `NVFP4`, so it
  does discriminate.
- **Timing.** The arms ran back to back on 2026-09-28: base from 14:32:15 to 14:36:33, split from 14:36:42 to 14:41:35.

## Node-0 gate

No cut was needed. Both arms ran the reduced tier (0:57344,1:40960), set as a driver override.

| arm | MemFree MiB | page cache MiB | available MiB | need MiB (57344 + 4096 + 15360) | ok |
|---|---|---|---|---|---|
| crtp-base | 75315 | 10444 | 85759 | 76800 | true |
| crtp-split | 78263 | 8803 | 87066 | 76800 | true |

## Decode: per session and pooled

Pooled ms/token is total decode seconds over total decode tokens (`mirror3_report.decode`).

| session | base ms/token | split ms/token | Δ | tokens | base TTFT s | split TTFT s | SM clock at start (base / split) MHz |
|---|---|---|---|---|---|---|---|
| cfq-train-Single_CDW/2015/page_35.pdf-2 | 126.07 | 126.68 | +0.61 | 6 | 8.376 | 8.382 | 2955 / 2947 |
| cfq-train-Single_ETR/2004/page_261.pdf-1 | 100.71 | 100.84 | +0.13 | 84 | 7.886 | 7.886 | 2970 / 2970 |
| **pooled** | **102.40** | **102.56** | **+0.16** | 90 | median 8.131 | median 8.134 | |

SM clock over the timed window, for both arms: min 2940, median 2947, max 2970 MHz (maximum 3135).

## Output identity

Every timed turn's `(reasoning, content)` is byte-identical between the arms (`mirror3_report.identity`):
2 of 2 turns, `all_identical: true`.

## RAM-miss service counters (server lifetime: 3 warm-up rounds, prefill and the timed set)

| | served | rows_read | read_errors | copy_errors | copy_fallbacks | overruns | piece_stream_refused | piece_publish_refused | ram_misses | read_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3342 | 4797 | 0 | 0 | 0 | 0 | 0 | 0 | 2571 | 3524.5 |
| split | 3364 | 4795 | 0 | 0 | 0 | 0 | 0 | 0 | 2560 | 3507.7 |

## Limits

- The timed set is the harness default: 2 sessions, 90 decode tokens, one turn each. The short session (6 tokens)
  moves by 0.6 ms/token on its own. The pooled number is dominated by the 84-token session.
- There is one pair and no repeated base arm, so drift between the arms is not separated from an effect. The ±1.5 bar
  is the 2026-09-28 campaign's noise threshold.
- The verdicts' acknowledged problems are the same as the mirror3 reference run's
  (`servers/mirror3/run-20260928-122035`): provenance `sglang_file` unavailable, `SGLANG_MOE_EXPERT_FILE_READER`
  resolved to None, `GRAPH_GATHER` not true, and no engine-side step latency.

## Reproduce

```bash
# divix01. The driver takes rowimg-disk.lock itself; do not wrap it in `flock rowimg-disk.lock` (self-deadlock)
cd /data/models/slang/nvfp4-work/wt-reader-crtp && B=/data/models/slang/nvfp4-work/wt-reader-crtp-base \
 && OMP_NUM_THREADS=8 setsid nohup taskset -c 0-63 bash analysis/dsv41-drive/reader-crtp/drive_reader_crtp_pair.sh \
   $B $(git -C $B rev-parse HEAD) $PWD $(git rev-parse HEAD) /mnt/nvme1/reader-crtp 30031 \
   > /mnt/nvme1/reader-crtp/driver.log 2>&1 < /dev/null &
```
