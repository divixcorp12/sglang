# Layer 14 Engram graph replay implementation results

## CPU checks

Ran on divix01 in the isolated checkout `/data/models/slang/nvfp4-work/cc-layer14-graph-cpu`, based on local commit `6205f090ea` with the candidate layer-14 edits overlaid:

- `test_engram_lookup_break.py` and `test_engram_file_table.py` with CUDA-only cases deselected: 14 passed, 5 deselected.

Both runs used divix01's `/data/models/slang/nvfp4-work/uring-test-deps` environment. Pytest emitted the existing `asyncio_mode` configuration warning and PyTorch JIT deprecation warnings.

The initial local attempt to run the GPU graph regression did not collect: the system Python lacked `sglang`, and the repository venv lacked `transformers`. The graph regressions were subsequently verified in the focused divix01 suite below.

## Focused divix01 suite

After the GPU lock became available, ran the full focused suite in the same isolated checkout under `flock` on GPU 0 (NVIDIA GeForce RTX 5090), with `CUDA_VISIBLE_DEVICES=0`:

```text
test_engram_file_table.py test_engram_lookup_break.py test_engram_row_cache.py
34 passed, 15 warnings in 19.76s
```

The first GPU run exposed that this PyTorch version reports an async device assertion as `device-side assert triggered`; the test now accepts that runtime text as well as the custom Engram assertion text, then verifies nonzero callback status and cleared pinned rows in the isolated child process.

## Real-shard capture/replay smoke

Ran a bounded two-layer lookup smoke in the isolated checkout under the GPU lock on GPU 0 (RTX 5090), using config metadata from `/mnt/nvme0/dsv41_flash/config.json` and the Engram shards in `/mnt/nvme2/DeepSeek-V4.1-Flash`:

- Config layers: 1 and 14; `engram_head_dim`: 256.
- Layer 1 table rows: 384,006,168; layer 14 table rows: 384,016,682.
- Captured one batch-1 lookup for each layer into one segment with zero breaks and two retained contexts.
- Replayed two changing ID sets, including numerically identical IDs across the two layer tags. Both outputs matched each table's eager lookup exactly (`max_abs_error=0.0` for both layers on both replays).
- Native store counters: 24 submitted SQEs, 24 completed CQEs, 12 unique misses, 16 hits, and 0 failures.

This smoke exercises actual shard rows and the Engram lookup graph path; it does not run a full model forward or serving benchmark.

## GPU and serving checks

Graph-level tests and real-shard lookup output comparison ran as described above. A full real-model forward, trace review, and serving A/B were not run. The existing `cc-engram-host-node-poc` checkout was left untouched; it is detached at `be8c899908`, while this candidate is based on `6205f090ea`.
