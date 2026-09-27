# DSV4.1 long-prompt prefill OOM (phase 0b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a 131,072-token prompt finish chunked prefill on divix01. Today it hits a CUDA OOM at a 61k prefix. Output must stay bit-identical.

**Architecture:** Layer 20 is the candidate source. It publishes a `[T, P]` bool candidate mask for every query row: 240 MiB at P = 61,440, and 1 GiB at 262k. Only the late-layer tail ever reads it, and the tail reads only its last `tail_len` rows. After this change the torch prefill indexer builds and publishes mask rows only for those tail rows when a tail exists. It publishes nothing on a layer-major pass's non-final chunks. `select_candidate_blocks` is row-independent, so the published rows equal the old full mask's tail slice.

**Tech Stack:** Python 3.13, PyTorch, sglang fork (`python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `.../dsv4/candidate_indexer.py`), pytest, divix01 (RTX 5090, 31.4 GiB).

**Spec:** the diagnosis `.superpowers/sdd/2026-09-27-dsv41-layer-major-prefill-phase1/oom-diagnosis.md`, fix A, with the Task 0 decision gate in `docs/superpowers/plans/2026-09-27-dsv41-layer-major-prefill-phase1.md` (Task 0, Step 6). The parent design is `docs/superpowers/specs/2026-09-26-dsv41-layer-major-prefill-design.md`.

## Global Constraints

- Work on branch `cc/layer-major-prefill`, after the controller has merged Task 0's analysis commit `ad337fddda` (`cc/lm-task-0`). Layer-major stays off by default (`SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0`), so every run here is plain chunked prefill.
- Output equality: the tail layers must see bit-identical candidate masks, and generated text must be identical before and after the fix.
- No new env vars. Nothing changes in the dense fp4 indexer path (`_low_ratio_index_topk_dense` / `_publish_or_consume_candidates`). The recipe runs the torch indexer (`SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`).
- `python/sglang/srt/model_executor/model_runner.py` is frozen. Use `msgspec.Struct`, not dataclasses. No defensive `getattr`. Comments are 1-2 lines, ASCII, and state non-obvious facts.
- Unit tests are CPU-only (`CUDA_VISIBLE_DEVICES=`). Only Task 2 uses the GPU, under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`.
- divix01: scratch goes under `/mnt/nvme1/` only. Use a private worktree `wt-oom-0b`, never `dsv41-direct-prod`/`-live`. Read `PIPESTATUS`.
- Commit trailer:

      Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
      Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF

## Review Focus

1. A batch of several requests, each with its own tail length. Each request's published mask must hold exactly its last `tail_len` rows, in request order, because `enter_late_layer_tail` zips masks with tail lengths by position.
2. A request whose extend is shorter than the 128-row window. `tail_len` equals its extend length, so all its rows are kept.
3. Bounded SWA replay off (`tail_forward_metadata is None`). Full masks are published exactly as before, because consumers then run all rows.
4. A tail crossing row-chunk boundaries. `rows_per_chunk` is 34 at P = 61k, so a 128-row tail spans 4-5 score chunks, and the kept rows of each chunk must be concatenated in order.
5. A layer-major non-final chunk (`layer_major_skip_candidates=True`) on the torch path. The source publishes empty masks, the same way the dense path already does at `_publish_or_consume_candidates`.

Task 1's tests pin 1, 2 and 4 through `keep_row_slice` and `candidate_publish_rows`, and 3 and 5 through the backend helper tests. Task 2 pins the end-to-end effect on the GPU.

---

### Task 1: Tail-only candidate masks in the torch prefill indexer

**Files:**
- Modify: `python/sglang/srt/layers/attention/dsv4/candidate_indexer.py` (add `keep_row_slice` after `select_candidate_blocks`, ~line 129)
- Modify: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`: `_low_ratio_index_topk_torch` (~3633-3708), `enter_late_layer_tail` (~1763-1790), and a new module-level `candidate_publish_rows` next to `_tail_rows`
- Test: `test/registered/unit/layers/attention/test_dsv4_candidate_indexer.py` (append)

**Interfaces:**
- Consumes: `select_candidate_blocks(logits, compress_lens, topk_blocks, block_size) -> Tensor[rows, width] bool` (row-independent); `LateLayerTail.extend_seq_lens_cpu`, `.local_lens_cpu`, `.cp_metadata`; `DSV4Metadata.layer_major_skip_candidates`
- Produces: `keep_row_slice(chunk_start: int, chunk_rows: int, keep_from: int) -> Optional[slice]` in `candidate_indexer`; `candidate_publish_rows(tail_metadata: Optional[DSV4Metadata]) -> Optional[list[int]]` in `deepseek_v4_backend`. With these, the source publishes `CandidateMasks(request_masks=[mask_b])`, where `mask_b.shape[0] == tail_len_b` when a tail exists.

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/unit/layers/attention/test_dsv4_candidate_indexer.py`:

```python
def _tail_masks_by_chunks(logits, lens, keep_from, rows_per_chunk, topk_blocks, block_size):
    kept = []
    for start in range(0, logits.shape[0], rows_per_chunk):
        s = logits[start : start + rows_per_chunk]
        sl = ci.keep_row_slice(start, s.shape[0], keep_from)
        if sl is not None:
            kept.append(
                ci.select_candidate_blocks(
                    s[sl], lens[start : start + rows_per_chunk][sl][:, None],
                    topk_blocks=topk_blocks, block_size=block_size,
                )
            )
    return torch.cat(kept) if kept else torch.zeros(0, logits.shape[1], dtype=torch.bool)


@pytest.mark.parametrize("keep_from", [0, 1, 33, 34, 35, 250, 299, 300])
def test_tail_rows_built_by_chunks_equal_the_full_mask_tail(keep_from):
    g = torch.Generator().manual_seed(keep_from)
    rows, width = 300, 97
    logits = torch.randn(rows, width, generator=g)
    lens = torch.randint(1, width + 1, (rows,), generator=g)
    logits = logits.masked_fill(torch.arange(width)[None, :] >= lens[:, None], -INF)
    full = ci.select_candidate_blocks(logits, lens[:, None], topk_blocks=4, block_size=8)
    got = _tail_masks_by_chunks(logits, lens, keep_from, 34, 4, 8)
    assert torch.equal(got, full[keep_from:])


def test_keep_row_slice_bounds():
    assert ci.keep_row_slice(0, 34, 0) == slice(0, 34)
    assert ci.keep_row_slice(0, 34, 34) is None
    assert ci.keep_row_slice(34, 34, 40) == slice(6, 34)
    assert ci.keep_row_slice(68, 10, 40) == slice(0, 10)


def _tail_meta(extend_lens_cpu, local_lens_cpu=None):
    tail = types.SimpleNamespace(
        extend_seq_lens_cpu=extend_lens_cpu,
        local_lens_cpu=local_lens_cpu,
        cp_metadata=object() if local_lens_cpu is not None else None,
    )
    return types.SimpleNamespace(late_layer_tail=tail)


def test_candidate_publish_rows_without_tail_keeps_every_row():
    assert backend_mod.candidate_publish_rows(None) is None


def test_candidate_publish_rows_per_request_and_short_extends():
    assert backend_mod.candidate_publish_rows(_tail_meta([128, 57, 128])) == [128, 57, 128]


def test_candidate_publish_rows_uses_local_lens_under_cp():
    assert backend_mod.candidate_publish_rows(_tail_meta([128], local_lens_cpu=[64])) == [64]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run on divix01 in `wt-oom-0b` (CPU):
`PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/layers/attention/test_dsv4_candidate_indexer.py -q -p no:randomly --basetemp=/mnt/nvme1/pytest-tmp/oom-0b 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}`

Expected: the new tests FAIL with `AttributeError: module ... has no attribute 'keep_row_slice'` / `'candidate_publish_rows'`, and the existing tests pass. If the file fails collection on the pyarrow break, record that and use `python -m pytest <file>::<test>` per test. If it still fails, report BLOCKED with the error.

- [ ] **Step 3: Add `keep_row_slice`**

In `candidate_indexer.py`, after `select_candidate_blocks`:

```python
def keep_row_slice(chunk_start: int, chunk_rows: int, keep_from: int) -> Optional[slice]:
    """Rows of the query-row chunk [chunk_start, chunk_start + chunk_rows) at or after
    keep_from, as a slice local to the chunk; None when the chunk ends before it."""
    lo = max(keep_from - chunk_start, 0)
    return slice(lo, chunk_rows) if lo < chunk_rows else None
```

Add `Optional` to the file's `typing` import if it is missing.

- [ ] **Step 4: Add `candidate_publish_rows` and use it in `enter_late_layer_tail`**

In `deepseek_v4_backend.py`, at module level next to `_tail_rows`:

```python
def candidate_publish_rows(tail_metadata: Optional["DSV4Metadata"]) -> Optional[list[int]]:
    """Rows per request the late layers run on this rank, or None when no tail runs and
    candidate consumers see every row."""
    if tail_metadata is None:
        return None
    tail = tail_metadata.late_layer_tail
    return tail.local_lens_cpu if tail.cp_metadata is not None else tail.extend_seq_lens_cpu
```

In `enter_late_layer_tail`, replace the `tail_lens_cpu = (...)` expression with `tail_lens_cpu = candidate_publish_rows(tail_metadata)`. Replace the two-line `TODO(candidate)` comment above `full_masks` with:

```python
        # The torch source publishes only these rows (the slice is then whole); the dense
        # source still publishes every row.
```

Leave the slicing code as it is: `mask[mask.shape[0] - t:]` on a `t`-row mask returns the whole mask.

- [ ] **Step 5: Publish only tail rows in `_low_ratio_index_topk_torch`**

Make these edits in `_low_ratio_index_topk_torch`:

(a) Right after `publish = [] if indexer.is_candidate_source else None`, add:

```python
        if publish is not None and self.forward_metadata.layer_major_skip_candidates:
            # A layer-major non-final chunk has no tail to read the masks.
            self.forward_metadata.candidate_metadata = CandidateMasks(request_masks=[])
            publish = None
        keep_rows = candidate_publish_rows(self.tail_forward_metadata)
```

(b) Inside the per-request loop, right after `masks = [] if publish is not None else None`, add:

```python
            keep_from = 0 if keep_rows is None else tok.numel() - keep_rows[b]
```

(c) Replace the `if masks is not None:` mask-append block with:

```python
                keep = (
                    keep_row_slice(start, tok_c.numel(), keep_from)
                    if masks is not None
                    else None
                )
                if keep is not None:
                    masks.append(
                        select_candidate_blocks(
                            s[keep],
                            lens_c[keep][:, None],
                            topk_blocks=indexer.candidate_topk_blocks,
                            block_size=indexer.candidate_block_size,
                        )
                    )
                elif consume is not None and masks is None:
                    s = s.masked_fill(~consume[b][rows], -torch.inf)
```

The `elif` keeps today's rule that a source never consumes. Its condition adds `masks is None` because a source chunk before the tail now gives `keep is None`.

(d) Replace `publish.append(torch.cat(masks) if len(masks) > 1 else masks[0])` with:

```python
                publish.append(
                    torch.cat(masks)
                    if len(masks) > 1
                    else masks[0]
                    if masks
                    else torch.zeros(0, lc, dtype=torch.bool, device=pos.device)
                )
```

(e) Import `keep_row_slice` from `sglang.srt.layers.attention.dsv4.candidate_indexer`, next to the existing `select_candidate_blocks` import.

The skip in (a) happens before `consume` is read. A source layer has `uses_candidates` false (`dsv41_sparse.py:190-191`), so its `consume` is None either way.

- [ ] **Step 6: Run the tests to verify they pass**

Run the Step 2 command. Expected: all tests pass, EXIT=0. Then run `py_compile` on both modified modules and `python -c "import sglang.srt.layers.attention.deepseek_v4_backend"` (CPU). Expected: no error.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/layers/attention/dsv4/candidate_indexer.py python/sglang/srt/layers/attention/deepseek_v4_backend.py test/registered/unit/layers/attention/test_dsv4_candidate_indexer.py
git commit -m "fix(dsv41): publish candidate masks only for the late-layer tail rows

Layer 20 kept a [T, P] bool mask alive through the late layers (240 MiB at a 61k
prefix, 1 GiB at 262k); only the last tail_len rows are read. Also honour
layer_major_skip_candidates on the torch indexer path.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
```

---

### Task 2: GPU verification at 16k (equality) and 128k (the OOM), and the reference entry

**Files:**
- Modify: `DSV41_REFERENCE.md`: correct §27.17's headroom line (~5115) and add `### 27.18` after §27.17

**Interfaces:**
- Consumes: Task 1's commit on `cc/layer-major-prefill`; `analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh <tag> <wt> <chunk> <hot_mb> <long_tokens>` (as in Task 0; it takes both locks itself, writes `/mnt/nvme1/prefill-chunk/<tag>/`, and ends `driver.log` with `long rc=<n>` and `DONE`)
- Produces: run directories `oom0b-base-16k`, `oom0b-fix-16k` and `oom0b-fix-128k`, the §27.18 text, and a gate verdict in the ledger

- [ ] **Step 1: Worktrees at the base and at the fix**

```bash
git -C /data/models/slang/sglang fetch -q origin
git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-oom-0b-base "$TASK1_BASE"   # the BASE the ledger records for Task 1
git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-oom-0b origin/cc/layer-major-prefill
```

Expected: `git -C <wt> log -1 --oneline` shows the two intended commits.

- [ ] **Step 2: Equality at 16k (base, then fix)**

```bash
cd /mnt/nvme1/prefill-chunk
LONG_MAX_NEW=32 bash /data/models/slang/nvfp4-work/wt-oom-0b-base/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh oom0b-base-16k /data/models/slang/nvfp4-work/wt-oom-0b-base 4096 16100 16384 > oom0b-base-16k.nohup 2>&1
LONG_MAX_NEW=32 bash /data/models/slang/nvfp4-work/wt-oom-0b/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh oom0b-fix-16k /data/models/slang/nvfp4-work/wt-oom-0b 4096 16100 16384 > oom0b-fix-16k.nohup 2>&1
diff <(python3 -c "import json;print(json.load(open('oom0b-base-16k/long.json'))['text'])") <(python3 -c "import json;print(json.load(open('oom0b-fix-16k/long.json'))['text'])"); echo DIFF_EXIT=$?
```

First confirm in `chunk_smoke.sh`'s request driver that the long request is greedy (`temperature` 0). If it is not, report that before running.
Expected: both runs end with `long rc=0` and `DONE`, and `DIFF_EXIT=0`, meaning 32 greedy tokens are identical. A difference is a stop: report it with both texts.

- [ ] **Step 3: The 128k run with the fix**

```bash
cd /mnt/nvme1/prefill-chunk
LONG_MAX_NEW=8 setsid nohup bash /data/models/slang/nvfp4-work/wt-oom-0b/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh \
  oom0b-fix-128k /data/models/slang/nvfp4-work/wt-oom-0b 4096 16100 131072 > oom0b-fix-128k.nohup 2>&1 < /dev/null &
```

Expected: after ~20 minutes, `driver.log` ends with `long rc=0` and `DONE`. Read the results:

```bash
/data/models/slang/.venv/bin/python /data/models/slang/nvfp4-work/wt-oom-0b/analysis/dsv41-drive/prefill-chunk/chunk_times.py oom0b-fix-128k | head -4
grep -c "memory allocation failed with OOM" oom0b-fix-128k/server.log
sed -E "s/.* 1=([0-9]+).*/\1/" oom0b-fix-128k/numa.log | sort -n | head -1
sort -t, -k2 -n oom0b-fix-128k/vram.csv | tail -1
```

Record these in the ledger: TTFT; the first, median and last chunk seconds; peak VRAM; OOM retries; and the minimum node-1 free.

- [ ] **Step 4: Decision gate**

| Result | Action |
|---|---|
| `long rc=0`, 0 OOM retries, last chunk within 1.5x the first | Done. Resume layer-major Task 12. |
| `long rc=0` but OOM retries > 0 | Done, but record in the ledger that the margin is thin. Task 12 keeps prompts ≤ 32k. The 262k phase must add fix B/C from the diagnosis. |
| OOM again | Stop. Run the zero-code MEM profile from the diagnosis (Q4, `/start_profile` with `activities=["MEM"]` around a 57,344-token prompt) and ask the user. |

- [ ] **Step 5: Reference entry**

In `DSV41_REFERENCE.md` §27.17, replace `- Headroom at the 4096 arm's peak is ~470 MiB.` with:

```markdown
- Headroom at the 4096 arm's peak was recorded as ~470 MiB, but that is nvidia-smi's 32,607 MiB total minus the peak.
  CUDA can use only 32,150 MiB, so the real margin was ~40 MiB (§27.18).
```

After §27.17, add `### 27.18 128k prefill: the layer-20 candidate mask OOM (result, 2026-09-27)`. It holds:
- the Task 0 failure: chunk 15, prefix 61,440, `flash_mla_sm120.py:251`, and 519 MiB reserved but free;
- the cause, with the `[T, P]` mask sizes at 16k, 61k, 131k and 262k;
- the fix and why it is exact (row independence);
- the Step 2 equality result and the Step 3 numbers;
- the run directories.

Write it in the style of §27.17: short bullets with numbers, and evidence paths at the end.

- [ ] **Step 6: Commit**

```bash
git add DSV41_REFERENCE.md
git commit -m "docs(dsv41): 128k prefill OOM and the tail-only candidate mask fix (27.18)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
```
