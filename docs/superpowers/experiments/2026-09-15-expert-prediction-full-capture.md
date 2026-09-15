# Expert Prediction Full Capture (2026-09-15)

**Stopped early by explicit user request**, well short of the 500 GB byte cap or
all sessions finishing. Numbers below reflect what was actually captured, not
a completed run.

## Session set

- Built with `build_sessions.py --convfinqa-all PATH=SPLIT` (repeatable) and
  `--financebench-all`, added in `ccaa4e39c4` for this run.
- ConvFinQA train: all 2222 eligible dialogues from `train.json`, split=train.
- ConvFinQA val: all 302 eligible dialogues from `dev.json`, split=val.
- FinanceBench: all 150 questions (84 unique docs, all PDFs downloaded, 0
  download/extract failures), split=holdout.
- Total: 2674 sessions, 10662 turns (train 9126, val 1236, holdout 300).
- Deterministic seed-0 interleave (per-group shuffle + proportional merge)
  so the mix stays representative from the start of the file, given the file
  was very unlikely to finish before a byte cap or manual stop.
- File: `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`.

## Run

- Commits: `ccaa4e39c4` (session builder), `41dff41f14` (adds
  `--stop-when-capture-stopped` to `run_capture_sessions.py`, unused here
  since the stop was manual, not cap-triggered).
- Server: `HOT_GPU_MB=12288 CAPTURE=1 run-shadow-server.sh capture-full 31030 off radix`,
  port 31030, predictors off, radix cache on. Started 05:44:38, healthy by 05:47:59.
- Capture dir: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438`.
- Driver: `run_capture_sessions.py --port 31030 --sessions .../sessions.jsonl --results .../results.jsonl --stop-when-capture-stopped <capture_dir> --print-stream`.
- Sanity check (`--max-sessions 1`) passed before the full run: reader reported
  `rows=4908, violations=[], stopped_reason=null`.
- No crashes, no server restarts, no errors in `results.jsonl` (0 of 530 turns).
- Stopped by user request at ~11:56 (driver killed, ~30 s to flush, then server
  killed via the documented `pkill` patterns). Total wall time ~6h11m.
- df -h /mnt/nvme2 at launch: 786 GB free, so the default 500 GB byte cap was
  kept (never reached).

## Reader report

```json
{"shards":162,"rows":698644,"prefill_rows":394709,"decode_rows":303935,"forwards":304485,"ran_rows":710259,"requests":532,"bytes":360409061741,"bytes_per_row":515869.4,"stopped_reason":null,"violation_count":0,"violations":[]}
```

- `violations: []`.
- `stopped_reason: null` -- confirms this was a manual stop, not a byte-cap stop.
- `ran_rows (710259) > rows (698644)`: an 11615-row gap from the in-flight
  forward that was killed mid-decode before its shard flush; the last shard
  (`shard-000161.safetensors`, ~1.68 GB) is a partial shard from that flush
  and was left in place, not deleted.
- 360.4 GB captured, 515.9 KB/row -- in line with the smoke test's 534.2 KB/row.

## Rows per split

Joined via `row.request_index` -> `request_ids[idx]` -> `rid.rsplit("-t",1)[0]`
-> `session_id` -> split (from `sessions.jsonl`), read per-shard from safetensors
metadata plus the `row.request_index` tensor (no large tensors loaded):

- train: 539850 rows
- val: 65678 rows
- holdout: 93115 rows
- unmatched: 1 row (out of 698644)

## Gate agreement

Ran on a 5-shard subset (`shard-000000` through `shard-000004`, ~5 GB, the
same rows already checked in the pre-run sanity check) rather than all 162
shards, since a full-capture gate check would need to load ~360 GB of tensors
against 147 GB available RAM on divix01.

- Per-layer agreement range: **0.9751 - 0.9886** across layers 0-47 (mean ~0.982),
  consistent with the 2026-09-14 smoke test's 0.982-0.992.

## Results (results.jsonl, 530 completed turns, 0 errors)

| split | sessions | turns | truncated | scored | correct | accuracy |
|---|---|---|---|---|---|---|
| train | 111 | 453 | 9 | 453 | 333 | 0.7351 |
| val | 15 | 63 | 0 | 63 | 56 | 0.8889 |
| holdout | 7 | 14 | 0 | 0 | 0 | n/a (FinanceBench turns are not auto-scored) |

- Mean decode throughput: 14.44 tok/s over 530 turns.
- Truncation rate (finish_reason=length): 9/530 = 1.7%, all in train.

## Notes

- The run was ended well before any split finished or the byte cap was hit;
  rows-per-split above are proportional to elapsed time under the seed-0
  interleave, not representative of the full dataset's eventual mix.
- No production server was touched; only the local `capture-full` server on
  port 31030 was started and stopped.
