# Native mirror extents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the native (CUDA-graph) RAM-miss reader serve each expert row
from the configured mirror roots, so mirroring benefits decode and not only
prefill.

**Architecture:** The Python table builder already knows the layout and the
mirror policy. It will emit a per-(row, expert, part) **extent** table — file,
source offset, length, destination offset inside the row's bounce slot — and
the C++ reader will submit every extent of a batch in one io_uring batch with
per-extent completion accounting. With one root and one part the tables and the
behaviour are byte-for-byte what they are today.

**Tech Stack:** Python 3.13, PyTorch, C++17, liburing, pytest.

**Spec:** `EXL3_COPY_PIPELINE_HANDOFF.md` item 1 ("Unify validated native/eager
mirror extent plans and counters"), and the measurement that motivates it in
`DSV41_REFERENCE.md` §19.

## Why this is worth doing

§19 measured it: with `SGLANG_MOE_EXPERT_MIRROR_DIRS` set and CUDA graphs on,
decode throughput is unchanged (2.8286 vs 2.8277 tok/s) while nvme2 still
carries 127 GiB. `exl3_ram_miss_tables` builds `paths` from
`layout.records[(layer, expert)].path` and never consults the row source, so
every in-graph decode miss reads the source checkpoint. Prefill, which goes
through the Python row source, is 1.84x faster. This plan closes that gap.

## Global Constraints

- `PAGE_BYTES = 4096`. Every extent offset, length and destination offset is a
  multiple of it; a row's parts sum exactly to the row's aligned length.
- A zero-length part is legal and means "this root serves none of this row".
  It must issue **no** read and must not be counted as pending.
- **Never publish a row until every one of its extents has completed
  successfully.** Partial rows are corruption.
- Per-extent CQE identity. Two extents of one row must be distinguishable in
  completion handling; do not collapse them into one row completion.
- EOF clamp is per extent, against the file that extent reads.
- Mirrors are byte-identical whole-checkpoint copies, so an extent's offset
  inside a mirror equals its offset inside the source. Size equality is checked
  at open; it detects truncation, not different same-size content.
- Preserve the existing short-read resubmit, soft-error retry, drain-on-error
  and ring-empty-on-return invariants (I1, M3 in the source comments).
- One root and one part must reproduce today's behaviour exactly.
- Commit with `git commit -F <file>`, never `-m`, no backticks or `$(...)` in
  the body. Stage by name. Trailers:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
```

## File Structure

- `python/sglang/srt/layers/moe/exl3_ram_miss.py` — `exl3_ram_miss_tables`
  gains the extent table and the mirror roots. Owns the split decision, reusing
  `exl3_read_split.SplitPolicy` so native and eager cannot drift apart.
- `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` — `Tables` gains
  `parts`; `RowReader::read` submits and accounts per extent; `open()` size-checks
  every root's copy of a shard.
- `python/sglang/srt/layers/moe/exl3_expert_format.py` — passes the configured
  roots and policy into the table builder.
- Tests alongside the existing ones:
  `test/registered/unit/kernels/test_exl3_ram_miss_split.py` (fault paths),
  `test/registered/unit/layers/moe/test_exl3_ram_miss*.py` (tables).

---

### Task 1: The extent table (pure Python)

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`exl3_ram_miss_tables`, `Exl3RamMissTables`)
- Test: the existing `exl3_ram_miss` table tests

**Interfaces:**
- Consumes: `exl3_read_split.SplitPolicy.plan(length) -> ReadSplit` (already exists;
  `ReadSplit` carries `part_bytes` and `starts`).
- Produces: `Exl3RamMissTables` gains
  `extents: torch.int64 tensor of shape (layers, experts, parts, 4)` where the
  last dim is `[file, offset, length, dest_offset]`, and `parts: int`. `reads`
  is replaced; `starts` (the segment-copy base per row/expert) stays available
  to the C++ side as it is today via the existing `reads[..., 3]` column — keep
  it as its own `starts` tensor of shape (layers, experts) so the extent table
  has a single clear meaning.
- `paths` ordering becomes shard-major, root-minor: the file index of part `p`
  of a shard whose source index is `s` is `s * parts + p`. With no mirrors,
  `parts == 1` and the ordering is unchanged.

- [ ] **Step 1: Write the failing tests.** With `parts == 1` and no roots, the
  extent table must equal today's reads: `extents[l, e, 0] == [file, offset,
  length, 0]` and `starts[l, e] == start`, for a layout fixture already used by
  the existing tests. With two roots and equal weights, for every (layer,
  expert): the two parts' lengths sum to the row's aligned length, both are
  multiples of 4096, `dest_offset` of part 0 is 0 and of part 1 is part 0's
  length, offsets are `record.offset + dest_offset`, and the file indices are
  `s * 2` and `s * 2 + 1`. Add a weights `(1, 0)` case: part 1 has length 0 and
  part 0 covers the whole row.
- [ ] **Step 2: Run, confirm failure.**
- [ ] **Step 3: Implement.** Build `paths` by iterating source shard paths and,
  for each, appending that shard's path under every root (the source checkpoint
  itself when there are no roots). Compute each row's split once per distinct
  aligned length via the policy and cache it — rows share lengths, so this is a
  small dict, not 15,360 policy calls.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 2: Per-extent submission and accounting (C++)

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_split.py`

**Interfaces:**
- Consumes Task 1's `extents` (layers, experts, parts, 4) and `starts`.
- `Read` becomes `{file, offset, length, dest}`; `Tables` gains `int64_t parts`
  and `std::vector<int64_t> starts`.

- [ ] **Step 1: Write the failing tests** against the existing fault-injection
  harness: a two-part row where the SECOND part's CQE errors must fail the row
  (not publish a half-filled slot); a two-part row where the first part returns
  a short positive read must resubmit only that part, at the right offset and
  destination, and still complete; a zero-length part must issue no read at all
  (assert the reader's CQE count); reversed completion order must still
  complete the row.
- [ ] **Step 2: Run, confirm failure.**
- [ ] **Step 3: Implement.** Index extents as `j = i * parts + p`. Size `done`,
  `expected`, `retries` by `count * parts`. Skip zero-length extents entirely
  (no SQE, not counted in `pending`). Submit with destination
  `bounce_ + i * slot_bytes + ext.dest + done[j]`, offset `ext.offset + done[j]`,
  length `ext.length - done[j]`. Set user data to `j`. `expected[j] =
  min(ext.length, file_sizes[ext.file] - ext.offset)`. Keep the segment copy
  keyed on `starts[row, expert]`, unchanged. Keep batch-level publication: the
  copy loop already runs only after the whole batch succeeded, which satisfies
  the never-publish-partial rule.
- [ ] **Step 4: Tests pass**, including the pre-existing fault and thread tests.
- [ ] **Step 5: Commit.**

---

### Task 3: Size-check every root at open, and wire the env through

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (`open()`)
- Modify: `python/sglang/srt/layers/moe/exl3_expert_format.py`
- Test: alongside the existing format-selection tests

- [ ] **Step 1: Write the failing tests.** `open()` must fail with a message
  naming both paths and both sizes when a root's copy of a shard differs in size
  from the source's. The format must pass the configured roots and weights into
  the table builder, and must pass none when the env var is unset.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement. Step 4: Tests pass.
      Step 5: Commit.**

---

### Task 4: Prove it on the GPU

- [ ] **Step 1:** Re-run §19's `g-mirror` arm (`analysis/dsv41-drive/run-mirror-arms.sh`,
  graphs on, `GRAPH_GATHER=1`, mirrors set) and read the per-drive byte deltas.
  **Gate: nvme2's read volume during the arm falls to roughly zero** (the layout
  is still built from it, so a small residual is expected), and nvme0 and nvme4
  each carry about half the total.
- [ ] **Step 2:** Compare mean decode tok/s against §19's 2.8277 and against the
  2.8286 baseline. Record the number whatever it is, including no change.
- [ ] **Step 3:** Byte-parity check: total bytes read must match §19's `g-base`
  total to within a percent, as it did for prefill.
- [ ] **Step 4:** Record in `DSV41_REFERENCE.md` §19 as a follow-up subsection,
  and commit.

## Self-review notes

- Spec coverage: handoff item 1's "native-consumable extent plan" is Task 1,
  "both paths consume the same validated policy" is Task 1 reusing
  `SplitPolicy`, "two parts per row need independent CQE accounting ... do not
  map both parts to an indistinguishable row completion" is Task 2, "extend the
  same first-open size validation to descriptors actually used by native I/O"
  is Task 3, and the acceptance evidence in its table row 1 ("Both native graph
  demand and advisory reads use the intended drives; byte parity") is Task 4.
- Deliberately NOT in this plan: the handoff's items 3-7 (overlapped packing,
  raw-row pinned cache, leases, DMA, lookahead). Item 2, stage timing, is worth
  doing before those, but it is independent of this change and does not block it.
- The advisory/prefetch path shares `RowReader::read`, so it inherits mirroring
  for free. That is intended; Task 4 Step 1 will see its bytes too.
