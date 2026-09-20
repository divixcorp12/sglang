# Expert row mirroring across NVMe drives — Spec and Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Supersedes:** `2026-09-19-expert-row-striping.md`. That plan split each expert row across drives, so every row needed every drive and a burst on either stalled all of them. Mirroring does everything striping did and adds the escape route, at the price of a second copy. Task 1's geometry survives; the repack tool does not (Task 0).

**Goal:** Cut the per-expert-row read time on the decode critical path by keeping a **full copy of the expert checkpoint on each of two NVMe drives** and splitting each row's single O_DIRECT read into per-drive sub-ranges issued in parallel — with the freedom to send any byte to either drive, so a slow drive can be routed around instead of waited on.

**Architecture:** No new on-disk format. Each mirror root is a byte-identical copy of the existing EXL3 checkpoint directory, so `Exl3ExpertLayout` already describes it. A new row source opens the same relative file on K roots and turns each record's existing `aligned_read()` into K page-aligned sub-ranges, submitted together. Because every root holds every byte, the split ratio is a runtime decision, not a property of the disk layout.

**Tech Stack:** Python, the existing `UringFileReader` io_uring batch submit, `Exl3ExpertLayout`/`Exl3RowReader`, pytest.

**Spec:** this document. Evidence: `DSV41_REFERENCE.md` §18.2, §18.6, and `analysis/dsv41-drive/REPORT.md`.

## Global Constraints

- Branch `dsv41`, worktree `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41`. Push only to `shared`. No rebase, amend, force-push or `git stash`; stage by name. Commit trailer:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF`.
- CPU jobs: `taskset -c 0-63`, `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`. GPU only via `$ANA/gpu-run.sh`. Never start production.
- Read `.claude/skills/env-var-conventions/SKILL.md` before adding any `SGLANG_*` variable.
- **Mirror roots (owner-supplied):** `/mnt/nvme0/dsv41_flash` and `/mnt/nvme4/dsv41_flash`. `/mnt/nvme4` is device **`nvme3n1`**, and also holds `eth-mainnet-reth` and the 64 GB swapfile (3.1 GB used). nvme0 holds bnb-mainnet and eth-mainnet-nimbus. Free: 961 GB and 625 GB; the copy needs 205 GB each.
- Tasks 1–4 touch no drive. Only Tasks 5–7 do, and the owner schedules each.
- O_DIRECT throughout: every sub-range offset and length is a multiple of 4096.

## Measurements this design rests on

13.3 MB O_DIRECT random reads, io_uring, QD1 (`analysis/dsv41-drive/`):

| Drive | Link | fio p50 | Our reader p50 |
|---|---|---:|---:|
| nvme2 (experts today) | Gen3 x2 | 7.70 ms | 9.19 ms |
| nvme0 | Gen3 x4 | 3.92 ms | 5.29 ms |
| nvme1 | Gen3 x4 | 4.01 ms | — |
| **nvme4 (`nvme3n1`)** | unknown | **unmeasured — Task 5** | — |

Our reader costs ~1.4 ms per row beyond the transfer, and that does not split. Expected per row with a 50/50 split across two x4 drives: ~2.1 ms transfer + ~1.4 ms ≈ **3.5 ms**, against 5.29 ms single-drive and 9.19 ms today. Over 18.7 rows/step that is ~34 ms beyond the ~81 ms the move off nvme2 already buys.

**Mirroring's real advantage is the bad case, not the good one.** Under a burst on one drive, a striped row waits for the slow half; a mirrored row can be read wholly from the quiet drive for ~5.3 ms. Task 6 measures exactly this.

## Design decisions

1. **Copies are byte-identical checkpoint directories.** No repack, no manifest, no slot geometry. A mirror root is valid if `Exl3ExpertLayout` builds from it and its files match the source's sizes; Task 4 verifies content.
2. **The split is per read, at a 4096 boundary.** For a record's `aligned_read() -> (offset, length, row_start)`, a K-way plan is K sub-ranges of `length`, each 4096-aligned, summing to `length`, each targeting one root. `row_start` is unchanged: the destination buffer is filled exactly as the single-drive path fills it.
3. **The split ratio is policy, not layout.** A `SplitPolicy` object returns the sub-range sizes for a read. Task 3 ships `StaticSplitPolicy` (fixed weights, default equal). Task 7 adds `AdaptiveSplitPolicy` (per-root EWMA of observed ms/byte, with a floor of 0 so a sick drive can be dropped entirely) only if Task 6's numbers justify it.
4. **One submit per batch.** Every sub-range of every requested row goes into a single `UringFileReader` submit; per-row latency is the max over its sub-ranges, so serialising them defeats the design.
5. **Degradation is automatic.** With K roots configured and one unreadable, the policy may route 100% to the survivors. A root that fails mid-read fails that row exactly as the shard source fails one; the streamer's existing retry/fatal path is unchanged.

## File structure

- `python/sglang/srt/layers/moe/exl3_stripe_layout.py` — **renamed** to `exl3_read_split.py` and reduced to the split-plan maths (Task 1's `StripeGeometry` becomes `ReadSplit`). The manifest classes go away with the repack (Task 0).
- `python/sglang/srt/layers/moe/exl3_mirror_row_source.py` — new. `Exl3MirrorRowSource`, an `ExpertRowSource` sibling of `Exl3ShardRowSource`.
- `python/sglang/srt/layers/moe/exl3_row_reader.py` — modified: read a row as K sub-ranges across K file ids.
- `python/sglang/srt/layers/moe/exl3_expert_format.py` — modified: pick the mirror row source when mirror roots are configured.
- `python/sglang/srt/environ.py` — modified: one new env var.
- `scripts/dsv41/verify_expert_mirror.py` — new: compare mirror roots against the source, byte for byte.
- Deleted: `scripts/dsv41/make_expert_stripe_set.py`, `test/manual/dsv41/test_stripe_set_roundtrip.py`.

---

### Task 0: Retire the striping artefacts

**Files:** delete `scripts/dsv41/make_expert_stripe_set.py` and `test/manual/dsv41/test_stripe_set_roundtrip.py`; rename `exl3_stripe_layout.py` → `exl3_read_split.py` and `test_exl3_stripe_layout.py` → `test_exl3_read_split.py`.

- [ ] **Step 1: Confirm nothing imports the deleted modules** (`grep -rn "make_expert_stripe_set\|StripeManifest\|StripeInfo\|validate_set" python/ scripts/ test/`). The stripe row source was never committed, so the manifest classes should have no consumers.
- [ ] **Step 2: Delete the two files with `git rm`.** Their history stays in the branch; the striping plan document stays too, with a superseded banner added in Task 8.
- [ ] **Step 3: Rename the module and its test** with `git mv`, then strip the manifest classes (`StripeManifest`, `StripeInfo`, `validate_set`) and their tests, keeping only the split maths. Rename `StripeGeometry` → `ReadSplit`, `fragment_bytes` → `part_bytes`, `strides`/`slot_offset` → **removed** (they were slot-layout concepts; mirroring has no slots). Keep `starts`.
- [ ] **Step 4: Run the renamed test file; it must pass unchanged apart from the renames.**
- [ ] **Step 5: Commit.**

### Task 1: The read-split plan

**Files:** modify `python/sglang/srt/layers/moe/exl3_read_split.py`; test `test/registered/unit/layers/moe/test_exl3_read_split.py`.

**Interfaces:** produces `ReadSplit(length: int, weights: Sequence[float]) -> .part_bytes: tuple[int, ...]`, `.starts: tuple[int, ...]`, where every part is a multiple of 4096 except that the parts sum exactly to `length`, and `SplitPolicy` with `plan(length: int) -> ReadSplit` plus `StaticSplitPolicy(weights)`.

The alignment rule differs from the striping plan and this is the point of the task: **`length` here is the already-page-aligned read length from `aligned_read()`**, so every part including the last can be a multiple of 4096. A part of 0 is legal (that root is skipped for this read) — that is how a drive gets dropped.

- [ ] **Step 1: Write the failing tests.**

```python
import pytest
from sglang.srt.layers.moe.exl3_read_split import ReadSplit, StaticSplitPolicy

@pytest.mark.parametrize("length", [4096, 4096 * 10, 13271040])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,), (2.0, 1.0, 1.0)])
def test_parts_are_page_aligned_and_sum_to_length(length, weights):
    s = ReadSplit(length=length, weights=weights)
    assert sum(s.part_bytes) == length
    assert all(n % 4096 == 0 for n in s.part_bytes)
    assert s.starts[0] == 0
    assert all(s.starts[i] + s.part_bytes[i] == s.starts[i + 1] for i in range(len(weights) - 1))

def test_a_zero_weight_drops_that_root():
    s = ReadSplit(length=4096 * 8, weights=(1.0, 0.0))
    assert s.part_bytes == (4096 * 8, 0)

def test_every_weight_zero_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4096, weights=(0.0, 0.0))

def test_a_length_that_is_not_page_aligned_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4097, weights=(1.0,))

def test_more_roots_than_pages_gives_empty_parts_not_an_error():
    s = ReadSplit(length=4096, weights=(1.0, 1.0, 1.0))
    assert sum(s.part_bytes) == 4096 and s.part_bytes.count(0) == 2

def test_static_policy_plans_the_same_split_every_time():
    p = StaticSplitPolicy(weights=(1.0, 1.0))
    assert p.plan(4096 * 4).part_bytes == p.plan(4096 * 4).part_bytes == (8192, 8192)
```

- [ ] **Step 2: Run them, confirm they fail. Step 3: Implement** (round each part down to a page, give the remainder pages to the largest-weight root). **Step 4: Tests pass. Step 5: Commit.**

### Task 2: Reading one row from K roots

**Files:** modify `python/sglang/srt/layers/moe/exl3_row_reader.py`; tests alongside the existing reader tests.

**Interfaces:** `Exl3RowReader.read_split(keys, destinations, *, roots, policy) -> list[int]`, or an extension of `read()` — choose from the code and justify it. Constraint: **all sub-ranges of all rows in one submit.**

- [ ] **Step 1: Failing tests with a recording fake `UringFileReader`:** 3 rows × 2 roots issues 6 reads in ONE submit; each read is `(file_id_of_root, record_offset + starts[i], part_bytes[i])`; destinations are `dest + starts[i]`; a 0-byte part issues no read at all; a row is served only when every part completes; a short read or per-root error fails that row exactly as the single-root path does (mirror `Exl3ShardRowSource.read`'s `RowReadStats` accounting and error behaviour).
- [ ] **Step 2: Run, confirm failure. Step 3: Implement.** Open each root's copy of a path once and cache the file id per (root, path). **Step 4: Tests pass. Step 5: Commit.**

### Task 3: The mirror row source

**Files:** create `python/sglang/srt/layers/moe/exl3_mirror_row_source.py`; test `test/registered/unit/layers/moe/test_exl3_mirror_row_source.py`.

**Interfaces:** `Exl3MirrorRowSource(reader, layer_id, segments, *, roots, policy)` mirroring `Exl3ShardRowSource`'s public surface exactly (`host_layouts`, `requires_page_aligned_destinations`, `covers`, `register_destinations`, `read`, `close`, `preferred_batch_rows`, `slot_bytes`, `bounce`, `file_bytes_per_expert`) so `ExpertStreamer` needs no change.

- [ ] **Step 1: Failing tests** over small synthetic checkpoints in `tmp_path`: two roots holding identical copies; reading rows returns bytes identical to the single-root shard source; `file_bytes_per_expert` matches; one root missing a file raises a message naming the root and the path.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement.** Resolve a record's path to each root by its path **relative to the source root**, so the roots may have different absolute prefixes. **Step 4: Tests pass. Step 5: Commit.**

### Task 4: Configuration and verification tooling

**Files:** modify `python/sglang/srt/environ.py` and `exl3_expert_format.py`; create `scripts/dsv41/verify_expert_mirror.py`; tests in the existing expert-format tests.

Read `.claude/skills/env-var-conventions/SKILL.md` first.

- New: `SGLANG_MOE_EXPERT_MIRROR_DIRS` (str, default `""`) — os.pathsep-separated mirror roots. Non-empty selects the mirror row source; one entry is legal and means "read everything from this root" (useful for the nvme0-only arm in Task 6).
- Optional: `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS` (str, default `""`) — colon-separated floats for `StaticSplitPolicy`; empty means equal weights.

- [ ] **Step 1: Failing tests.** Mirror dirs set → mirror row source with a `StaticSplitPolicy` of the parsed weights; unset → the shard source, unchanged; a weights list whose length differs from the dirs list is refused, naming both counts; a root that is not a readable directory is refused, naming it.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement.**
- [ ] **Step 4: `verify_expert_mirror.py`:** for given roots and a source dir, compare every expert row of chosen layers byte for byte (read through the mirror source and through the shard source), and independently compare file sizes for every file in the layout. Report the first mismatching (layer, expert, byte offset), or the first size mismatch. Precise output matters: this is what the 205 GB copies are trusted on.
- [ ] **Step 5: Tests pass. Step 6: Commit.**

### Task 5: Benchmark nvme4 (owner schedules; read-only, small)

nvme4 has never been measured and its bandwidth sets the default weights.

- [ ] **Step 1: Wait for the owner's go** — they were copying data off nvme4, and a benchmark during that is meaningless.
- [ ] **Step 2: Link width** from sysfs for device `nvme3n1`, as `analysis/dsv41-drive/` did for the others.
- [ ] **Step 3: fio**, the same recipe as `analysis/dsv41-drive/REPORT.md`: 13 MB O_DIRECT randread through io_uring, QD1 and QD6, 20 s, `--readonly` against an existing large file on nvme4 (`eth-mainnet-reth/static_files/...`). Report BW and clat p50/p99.
- [ ] **Step 4: Record** the row in the table above and pick the default weights as the ratio of measured bandwidths. **If nvme4 is materially slower than nvme0** (say below 2.5 GB/s), say so plainly: the weights compensate, but the mirror's best case is then bounded by the slower drive's share.

### Task 6: Copy, verify, measure (owner schedules; heavy I/O)

- [ ] **Step 1: Capacity and scheduling check** with the owner; 205 GB per root, and the source read comes off nvme2 which production's PLE cache also uses.
- [ ] **Step 2: Copy** the expert checkpoint to `/mnt/nvme0/dsv41_flash` and `/mnt/nvme4/dsv41_flash`. A plain `cp`/`rsync` per root; run them **sequentially**, not in parallel, so the nvme2 read is not the shared bottleneck twice over. Record wall time per root.
- [ ] **Step 3: Verify** with `verify_expert_mirror.py`: all files' sizes on both roots, plus every expert of layers 0, 19 and 39 byte for byte against nvme2. A mirror that differs anywhere is not usable.
- [ ] **Step 4: Per-row latency**, reusing `analysis/dsv41-drive/bench_drive_rows.py`, over four configurations: nvme2 (baseline), nvme0 only, nvme4 only, and both mirrored 50/50 (or the Task 5 weights). 3 × 48 rows each, alternating. Report p50/p90/p99. **Gate: mirrored p50 ≤ 4.0 ms.**
- [ ] **Step 5: The contention test, which is the point of mirroring.** Repeat the mirrored and single-root measurements while a deliberate write burst runs on ONE root (`fio --rw=randwrite --bs=128k --numjobs=2 --runtime=60` against a scratch file on that drive — confirm the path with the owner, never into production data). Report p50/p99 for: mirrored 50/50 under burst, and mirrored with the bursting root weighted to 0. **The second must be close to the quiet drive's solo number** — that is the escape route striping lacked, and it is what Task 7 automates.
- [ ] **Step 6: End-to-end arm.** `trace_corpus.py --graphs`, sessions 0–3, 256-token prompts, 128 new tokens, standard `c32` settings, with `SGLANG_MOE_EXPERT_MIRROR_DIRS` set to both roots. Compare against the recorded `c32` baseline (2.823 tok/s) and a same-day nvme0-only arm, so the mirroring gain is separated from the drive-move gain.
- [ ] **Step 7: Record** the arms, the per-row table and the ruling in `DSV41_REFERENCE.md` (new §19), and commit.

### Task 7: Adaptive split (only if Task 6 Step 5 justifies it)

- [ ] **Step 1: Decide from Task 6's numbers.** If the burst barely moves mirrored p99, keep the static policy and stop — record why.
- [ ] **Step 2: Failing tests** for `AdaptiveSplitPolicy`: it starts at the static weights; after feeding it observations where root B is 3× slower per byte, its next plan gives B a smaller part; a root whose observations time out repeatedly drops to a 0 part; a recovered root climbs back. Use an injected clock and fed observations — no sleeps, no real I/O.
- [ ] **Step 3: Implement** an EWMA of ms/byte per root, updated from the completion times the reader already has. Keep it allocation-free on the hot path and lock-free (single reader thread). Document the half-life choice and why in the docstring.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Re-run Task 6 Steps 5 and 6** with the adaptive policy; keep it only if p99 under burst improves against the static 50/50 arm. **Step 6: Commit and record.**

### Task 8: Mark the striping plan superseded

- [ ] **Step 1:** Add a banner at the top of `docs/superpowers/plans/2026-09-19-expert-row-striping.md`: superseded by this plan, with one sentence on why (every striped row needs every drive; mirroring buys the same latency and can route around a slow drive for one extra copy). Do not delete the file — the geometry reasoning and the drive measurements in it are the record of how we got here.
- [ ] **Step 2: Commit.**

## Verification summary

Correctness rests on Task 4's byte-for-byte verifier and Task 6 Step 3: the mirrors are plain copies, so any difference is a bug, not a tolerance. Performance rests on the Step 4 gate, and the design's real claim — surviving a contended drive — is settled by Step 5.

## What this plan deliberately does not do

- No striped layout: superseded, for the reason in the banner.
- No RAID1: the block layer would mirror writes we never make and could not split a single read by policy, which is the whole point.
- No change to the pinned tier, the gather kernel, or the RAM-miss protocol: a row stays opaque bytes above the row source.
