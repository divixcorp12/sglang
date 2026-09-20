# Expert row striping across NVMe drives — Spec and Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the per-expert-row read time on the decode critical path by splitting each expert's single O_DIRECT read into K fragments issued in parallel to K NVMe drives, sized in proportion to each drive's measured bandwidth.

**Architecture:** A new row source reads from a repacked *stripe set*: K directories, one per drive, each holding fixed-stride, page-aligned row fragments. An expert row is the byte-wise concatenation of its K fragments, in stripe order, so the destination slot assembles itself from K contiguous writes. The existing shard row source stays the default and the fallback. Nothing above the row source changes: the pinned tier, the gather kernel and the RAM-miss protocol all keep treating a row as opaque bytes.

**Tech Stack:** Python (layout, repack tool), the existing `UringFileReader` io_uring batch submit, safetensors (source only), pytest.

**Spec:** this document. Evidence: `DSV41_REFERENCE.md` §18.2 (step breakdown), §18.6 (burst frequency), and the drive measurements in `analysis/dsv41-drive/REPORT.md`.

## Global Constraints

- Branch `dsv41`, worktree `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41`. Push only to `shared`. No rebase, amend, force-push or `git stash`; stage by name. Commit trailer:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF`.
- CPU jobs on divix01: `taskset -c 0-63`, `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`. GPU only via `$ANA/gpu-run.sh`. Never start production.
- Read `.claude/skills/env-var-conventions/SKILL.md` before adding any `SGLANG_*` variable.
- The owner schedules every bulk copy; the repack reads 205 GB from nvme2 once.
- O_DIRECT everywhere: every fragment offset, length and destination address is a multiple of 4096.

## Measurements this design rests on

13.3 MB O_DIRECT random reads, io_uring, QD1 (`analysis/dsv41-drive/`):

| Drive | Link | fio BW | fio p50 | Our reader p50 |
|---|---|---:|---:|---:|
| nvme2 (experts today) | Gen3 x2 | 1,641 MB/s | 7.70 ms | 9.19 ms |
| nvme0 | Gen3 x4 | 3,218 MB/s | 3.92 ms | 5.29 ms |
| nvme1 | Gen3 x4 | 3,091 MB/s | 4.01 ms | — |

Our reader costs ~1.4 ms per row beyond the transfer (Python, torch, submit/complete). **That fixed cost does not stripe**, and it sets the floor:

| Configuration | Transfer | + overhead | Rows/step × saving |
|---|---:|---:|---|
| nvme2 today | 7.8 ms | 9.19 ms | baseline |
| nvme0 alone | 3.9 ms | ~5.3 ms | ~81 ms/step |
| **nvme0+nvme1, 2-way** | ~2.1 ms | **~3.5 ms** | ~34 ms/step beyond the move |
| + a third x4 drive, 3-way | ~1.4 ms | ~2.8 ms | ~13 ms/step beyond 2-way |
| + nvme2 weighted, 4-way | ~1.2 ms | ~2.6 ms | ~4 ms/step beyond 3-way |

**Ruling: build K-way, deploy 2-way.** Returns collapse after the second drive because of the 1.4 ms floor, while each extra drive adds a completion to wait on — a row's latency is the *max* over its fragments, so tail risk grows with K. The format and reader take K from a manifest, so a third drive is a repack and a config change, never a code change.

## Risk this design accepts

Every row needs every drive, so a write burst on any striped drive stalls all rows. Today `op-reth` bursts 50–1,800 IOPS on nvme1 and two chain nodes live on nvme0. Task 6 measures p99 per row under a deliberate burst; if it regresses past the gate there, drop to 2-way on the quietest pair, or abandon striping for plain placement on the single best drive (already worth ~81 ms/step, no code).

## File structure

- `python/sglang/srt/layers/moe/exl3_stripe_layout.py` — new. The manifest dataclass, its JSON (de)serializer, and fragment geometry.
- `python/sglang/srt/layers/moe/exl3_stripe_row_source.py` — new. `Exl3StripeRowSource`, an `ExpertRowSource` sibling of `Exl3ShardRowSource`.
- `python/sglang/srt/layers/moe/exl3_row_reader.py` — modified. `read()` gains a fragment-aware path.
- `python/sglang/srt/layers/moe/exl3_expert_format.py` — modified, minimally: choose the row source from the new env var.
- `scripts/dsv41/make_expert_stripe_set.py` — new. The offline repack.
- `python/sglang/srt/environ.py` — modified. Two new env vars (read the skill first).
- Tests: `test/registered/unit/layers/moe/test_exl3_stripe_layout.py`, `test_exl3_stripe_row_source.py`, `test/manual/dsv41/test_stripe_set_roundtrip.py`.

## On-disk format (the core decision)

Each drive `i` of K holds one directory. Inside it, one file per layer: `layer-<L>.bin`. A layer file is `num_experts` slots of identical stride:

```
fragment_i_bytes = align4096(ceil(row_bytes * w_i / sum(w)))   # per-drive share, page-aligned
stride_i         = fragment_i_bytes                            # slot n starts at n * stride_i
```

The last drive's fragment absorbs the rounding so the K fragments sum to exactly `row_bytes`. The read for expert `n` on drive `i` is therefore `(layer-<L>.bin, n * stride_i, fragment_i_bytes)` — no per-record offset table, O(1) lookup, page-aligned by construction. This is why the format is a repack and not a byte-split of the safetensors: a split of the originals would put fragments at arbitrary offsets and force per-fragment alignment padding through `row_start_in_buffer`.

A row's bytes keep the exact order `Exl3ExpertFormat` already defines (the concatenated streamed tensors), so fragment `i` is simply row bytes `[start_i, start_i + fragment_i_bytes)`. Assembly into a pinned slot is K contiguous writes at `dest + start_i`, and the gather kernel is untouched.

`manifest.json`, written identically into every stripe directory:

```json
{
  "version": 1,
  "source": "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw",
  "num_layers": 40, "num_experts": 384, "row_bytes": 13271040,
  "tensor_order": ["w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"],
  "stripes": [
    {"index": 0, "weight": 3218, "fragment_bytes": 6639616, "dir_hint": "/mnt/nvme0/dsv41-stripe-0"},
    {"index": 1, "weight": 3091, "fragment_bytes": 6631424, "dir_hint": "/mnt/nvme1/dsv41-stripe-1"}
  ],
  "row_sha256_sample": {"0:0": "…", "19:200": "…"}
}
```

`dir_hint` is provenance only; the runtime uses the env var's order. Every manifest must agree on everything except its own `index`, and the row source refuses a mismatched set.

## Task 1: The stripe manifest and geometry

**Files:** create `python/sglang/srt/layers/moe/exl3_stripe_layout.py`; test `test/registered/unit/layers/moe/test_exl3_stripe_layout.py`.

**Interfaces:**
- Produces `StripeGeometry(row_bytes: int, weights: Sequence[float])` with `.fragment_bytes -> tuple[int, ...]`, `.starts -> tuple[int, ...]`, and `.slot_offset(stripe: int, expert: int) -> int`.
- Produces `StripeManifest` (msgspec.Struct or dataclass, matching the file's neighbours) with `to_json()` / `from_json()` and `validate_set(manifests: Sequence[StripeManifest]) -> None`.

- [ ] **Step 1: Write the failing tests.**

```python
import pytest
from sglang.srt.layers.moe.exl3_stripe_layout import StripeGeometry

def test_fragments_are_page_aligned_and_sum_to_the_row():
    g = StripeGeometry(row_bytes=13271040, weights=(3218.0, 3091.0))
    assert sum(g.fragment_bytes) == 13271040
    assert all(n % 4096 == 0 for n in g.fragment_bytes[:-1])
    assert g.starts == (0, g.fragment_bytes[0])

def test_last_fragment_absorbs_the_remainder():
    g = StripeGeometry(row_bytes=4096 * 10 + 7, weights=(1.0, 1.0))
    assert sum(g.fragment_bytes) == 4096 * 10 + 7
    assert g.fragment_bytes[0] % 4096 == 0

def test_weights_shape_the_split():
    g = StripeGeometry(row_bytes=4096 * 100, weights=(3.0, 1.0))
    assert g.fragment_bytes[0] > 2 * g.fragment_bytes[1]

def test_single_stripe_is_the_whole_row():
    g = StripeGeometry(row_bytes=4096 * 5, weights=(1.0,))
    assert g.fragment_bytes == (4096 * 5,) and g.slot_offset(0, 3) == 3 * 4096 * 5

def test_slot_offset_is_stride_times_expert():
    g = StripeGeometry(row_bytes=4096 * 8, weights=(1.0, 1.0))
    assert g.slot_offset(1, 5) == 5 * g.fragment_bytes[1]

@pytest.mark.parametrize("weights", [(), (1.0, 0.0), (1.0, -1.0)])
def test_rejects_degenerate_weights(weights):
    with pytest.raises(ValueError):
        StripeGeometry(row_bytes=4096, weights=weights)
```

  Add manifest tests: a round trip through `to_json`/`from_json`; `validate_set` accepting two manifests differing only in `index`; and it raising when `row_bytes`, `num_experts`, `tensor_order` or the stripe list disagree, when indices are not `0..K-1`, or when a manifest is missing.

- [ ] **Step 2: Run them, confirm they fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** Keep it pure: no file I/O, no torch. The last fragment is `row_bytes - sum(previous)`; every earlier fragment is `align4096(row_bytes * w_i / sum(w))`. Raise if any earlier fragment is 0 or if the last is ≤ 0 (too many stripes for the row).
- [ ] **Step 4: Tests pass. Step 5: Commit.**

## Task 2: The offline repack tool

**Files:** create `scripts/dsv41/make_expert_stripe_set.py`; test `test/manual/dsv41/test_stripe_set_roundtrip.py`.

**Interfaces:** consumes `Exl3ExpertLayout` (`exl3_expert_layout.py`) for source records and `StripeGeometry` from Task 1. Produces `write_stripe_set(source_dir, out_dirs, weights, layers=None) -> StripeManifest`.

- [ ] **Step 1: Failing round-trip test** on a synthetic 2-layer, 4-expert source built in `tmp_path` (fabricate an `Exl3ExpertLayout` rather than a real checkpoint): write a 2-stripe set, then read every row back by concatenating its fragments and assert byte equality with the source rows, for both stripes and every expert. Include one odd `row_bytes` that is not a multiple of 4096.
- [ ] **Step 2: Run it, confirm it fails.**
- [ ] **Step 3: Implement.** Read each source row once (`Exl3RowReader`, `direct=True`), slice it per `StripeGeometry.starts`, and append to each layer file at `slot_offset`. Write in layer order, streaming — never hold a layer in RAM. Pad the final fragment of the final slot so the file length is a whole number of strides. Record `row_sha256_sample` for a handful of (layer, expert) pairs. `--layers` limits the work for a pilot. Print bytes written per stripe and the elapsed time.
- [ ] **Step 4: Test passes. Step 5: Commit.**
- [ ] **Step 6: Pilot on the real checkpoint (owner schedules; reads from nvme2).** One layer only, to `/mnt/nvme0/dsv41-stripe-0` and `/mnt/nvme1/dsv41-stripe-1`. Verify with `verify_stripe_set` from Task 3's CLI, then delete or keep as the pilot for Task 5.

## Task 3: Fragment-aware reads

**Files:** modify `python/sglang/srt/layers/moe/exl3_row_reader.py`; create `python/sglang/srt/layers/moe/exl3_stripe_row_source.py`; tests `test/registered/unit/layers/moe/test_exl3_stripe_row_source.py`.

**Interfaces:**
- `Exl3RowReader.read_fragmented(keys, destinations, *, geometry, files) -> list[int]` — or, if it reads more naturally, extend `read()` to take fragment lists. Decide from the code and say which in the report; the constraint is that **all fragments of all requested rows go into ONE `UringFileReader` submit**, because per-row latency is the max over its fragments and serialising them would defeat the design.
- `Exl3StripeRowSource(...)` mirrors `Exl3ShardRowSource`'s public surface (`covers`, `register_destinations`, `read`, `close`, `preferred_batch_rows`, `file_bytes_per_expert`) so `ExpertStreamer` needs no change.

- [ ] **Step 1: Failing tests** against a fake `UringFileReader` that records submissions: a 3-row batch over 2 stripes issues 6 reads in one submit; each read's `(file_id, offset, length)` matches `slot_offset`/`fragment_bytes`; destinations are `dest + start_i`; a row counts as served only when all its fragments complete; a short read or a per-fragment error fails that row (and only that row) the same way the shard source fails one.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Open each layer file once per stripe and cache the file id, as `_file` already does. Keep `buffer_bytes` at the row stride so pinned-slot sizing is unchanged.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Add `verify_stripe_set(stripe_dirs, source_dir, layers, experts)`** to the repack script's CLI: read rows through the stripe source and through the shard source and compare bytes. This is the tool Task 2 Step 6 and Task 5 use.
- [ ] **Step 6: Commit.**

## Task 4: Selecting the stripe set at runtime

**Files:** modify `python/sglang/srt/environ.py` and `python/sglang/srt/layers/moe/exl3_expert_format.py`; tests in the existing expert-format/env tests.

Read `.claude/skills/env-var-conventions/SKILL.md` first; follow its naming and defaulting rules.

- New: `SGLANG_MOE_EXPERT_STRIPE_DIRS` (str, default `""`) — os.pathsep-separated stripe directories, **in stripe order**. Non-empty selects the stripe row source.
- New: `SGLANG_MOE_EXPERT_STRIPE_WEIGHTS` (str, default `""`) — optional, colon-separated floats, used only by the repack tool; the runtime takes geometry from the manifest. Put it in the repack script's CLI instead if the skill's rules discourage a tool-only env var — decide and justify.

- [ ] **Step 1: Failing tests.** With the var set to two dirs, the format builds an `Exl3StripeRowSource`; unset, it still builds `Exl3ShardRowSource`. A set whose manifests disagree, or whose directory count differs from the manifests' `K`, raises a message naming the offending directory. `SGLANG_MOE_EXPERT_ROW_SOURCE=shards` together with stripe dirs is refused as contradictory.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement. Step 4: Tests pass. Step 5: Commit.**

## Task 5: Build the real stripe set and measure (owner schedules; heavy I/O)

- [ ] **Step 1: Capacity check.** 205 GB splits to ~103 GB per stripe. nvme0 has 517 GB free, nvme1 1.7 TB. Confirm before starting, and confirm production's state with the owner.
- [ ] **Step 2: Full repack** to `/mnt/nvme0/dsv41-stripe-0` and `/mnt/nvme1/dsv41-stripe-1`. Expect ~2 minutes of reading at nvme2's line rate plus write time; measure it.
- [ ] **Step 3: Verify** every expert of 3 layers (0, 19, 39) byte-for-byte against the shard source with `verify_stripe_set`, plus the manifest's `row_sha256_sample`.
- [ ] **Step 4: Per-row latency**, reusing `analysis/dsv41-drive/bench_drive_rows.py`: 3 × 48 rows, alternating between the stripe source and the nvme2 shard source. Report p50/p90/p99 and mean per row for both. **Gate: stripe p50 ≤ 4.0 ms** (the model predicts ~3.5).
- [ ] **Step 5: Tail under load.** Repeat Step 4 while a deliberate write burst runs on one striped drive (`fio --rw=randwrite --bs=128k --numjobs=2 --runtime=60` against a scratch file on that drive, **not** into production data; confirm the target path with the owner first). Report p99 with and without. **Gate: stripe p99 under burst ≤ 2× its quiet p99**; if it fails, recommend plain nvme0 placement instead and stop.
- [ ] **Step 6: End-to-end arm.** `trace_corpus.py --graphs`, sessions 0–3, 256-token prompts, 128 new tokens, the standard `c32` settings, with `SGLANG_MOE_EXPERT_STRIPE_DIRS` set. Compare tok/s against the recorded `c32` baseline of 2.823 and against a same-day nvme0-only arm, so the striping gain is separated from the drive-move gain.
- [ ] **Step 7: Record** the three arms, the per-row table and the ruling in `DSV41_REFERENCE.md` (a new §19), and commit.

## Verification summary

Correctness rests on Task 2's round trip and Task 5 Step 3's byte comparison against the shard source — the stripe set is a pure repack, so any mismatch is a bug, not a tolerance. Performance rests on the Step 4 and Step 5 gates, and the end-to-end arm is what decides whether the format stays.

## What this plan deliberately does not do

- No RAID0 or `dm-stripe`: both drives carry production filesystems, and a block-layer stripe would need new partitions or a loop-device stack under a production-adjacent workload.
- No expert-interleave (even/odd experts per drive): it only helps when a layer misses ≥2 experts at once, and the average is well under one row per layer.
- No change to the pinned tier, the gather kernel, or the RAM-miss protocol: a row stays opaque bytes above the row source.
