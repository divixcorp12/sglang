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

### Task 2b: Unify EOF behaviour between the two readers, with a test

Handoff §3A: "the general reader's short-positive-read retry loop and the
native reader's expected-existing-bytes termination differ. Add a focused
O_DIRECT test for an aligned request crossing non-block-aligned EOF before
reusing either behavior." The page-aligned superset of a shard's LAST row
overruns EOF by construction, so this is a real case on every shard, not a
hypothetical. The eager path already clamps against the source size; the native
path clamps against `file_sizes`. They must agree.

**Files:**
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_split.py`
- Modify only if the test shows they disagree:
  `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`

- [ ] **Step 1: Write the test.** A real temp file whose size is NOT a multiple
  of the logical block size, an O_DIRECT aligned request whose offset+length
  crosses its EOF, read through the native reader and through
  `Exl3RowReader`/`read_split`. Assert both return the same bytes, the same
  completion status, and that the bytes past EOF are left untouched in the
  destination. Split the request across two roots as well, so the clamp is
  exercised on a part that lies entirely past EOF (length becomes 0 after
  clamping) and on a part that straddles it.
- [ ] **Step 2: Run.** If they already agree, record that as the finding and
  commit the test as a regression guard — do NOT change behaviour to match a
  preference. If they disagree, the native reader moves to the eager path's
  semantics, because that one is what the mirror verifier already validated
  205 GB of data against.
- [ ] **Step 3: Commit.**

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
  **Predicted before measuring, so the result can falsify it:** the ceiling is
  1.38x (~3.90 tok/s), derived above from §18.2's 190 ms NVMe share and the
  measured per-row ratio. Anything from 1.0x to 1.38x is plausible; below ~1.1x
  means the NVMe portion is not what §18.2's attribution says it is, or
  within-row splitting is losing at production queue depth (§3D).
- [ ] **Step 3:** Byte-parity check: total bytes read must match §19's `g-base`
  total to within a percent, as it did for prefill.
- [ ] **Step 4:** Record in `DSV41_REFERENCE.md` §19 as a follow-up subsection,
  and commit.

## The handoff's six main conclusions (§1), one by one

| # | Conclusion | Where it lands |
|---|---|---|
| 1 | Wire mirrors into the native graph reader first | **This plan, Tasks 1-4.** Independently confirmed by §19's byte accounting before planning. |
| 2 | The pipeline is still serial at important boundaries | Not here. Handoff §3B; needs generations (see below) first. |
| 3 | Biggest structural wins are earlier requests, incremental completion, overlap; queue flags follow measurement | Not here, and the ordering clause is honoured: stage timing before any of it. |
| 4 | A cold exact-demand dependency cannot be made latency-free | **Not a task — a bound on this plan.** See "What this plan cannot achieve" below. |
| 5 | Generic async controls are not sufficient for EXL3 | Not a task; a standing warning. Already borne out in this repo: `f5e5b33bfd` made EXL3 refuse the inert `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` flag rather than appear to support it. Nothing in this plan enables a generic flag and calls it async EXL3 promotion. |
| 6 | Slot leases and completion generations before increasing concurrency | Half addressed: generations are recorded as a precondition on §3B rather than retrofitted later. Leases are not in scope, and belong with §3B/§4C. |

### What this plan cannot achieve (conclusion 4)

Conclusion 4 is the one that bounds the whole effort, so it is worth stating the
ceiling before Task 4 measures anything.

From §18.2's step: NVMe 190 ms of a 391 ms step. Scaling the NVMe portion by the
measured per-row ratio (4.164/9.623) gives 82 ms, a 283 ms step, i.e. a **1.38x
step speedup**. Applied to the 2.823 tok/s c32 baseline that is **~3.90 tok/s**.

That is an upper bound, and a generous one: it assumes the entire NVMe portion
scales with a QD1 per-row ratio, that nothing else changes, and that the native
path realises the same ratio the eager path did. **If Task 4 lands well short of
1.38x, that is information about the pipeline, not a failure of the extent
table** — and §3D's whole-row-versus-split question, plus the §19 eager anomaly,
are the first places to look.

Beyond that ceiling, conclusion 4 says the remaining cold-miss latency cannot be
removed by faster reads at all: until routing is known the right expert cannot
be fetched, so the next lever is prediction, independent requests, or computing
hit experts while cold ones load — handoff §4B and §4D. This plan deliberately
buys the 1.38x and stops.

## What this plan takes from the handoff, and what it leaves

### §2 (the copy and dependency path) — used as design input, not as tasks

§2 is descriptive, and three of its observations shaped the design above rather
than becoming work items. Recording them so the reasoning is auditable:

- "The mirror subclass overrides only the read hook: its parts land directly in
  their final positions within the same bounce row. There is no extra
  mirror-assembly copy and no duplicated whole-row read." This is why Task 1's
  extent table carries a `dest_offset` into the row's existing bounce slot
  instead of introducing per-part buffers. The native path copies the eager
  path's shape deliberately.
- "One application-level reader call can still require multiple io_uring kernel
  submissions for queue limits, short reads, or retries." This is why Task 2
  keeps the existing resubmit loop rather than assuming one SQE per extent.
- "The inherited bounce-to-tensor scatter is the CPU copy to optimize." Agreed,
  and independently measured: a fixed 1.58 ms/row, ~8.4 GB/s, 38% of mirrored
  per-row time (`analysis/dsv41-drive/MIRROR_ROWS.md`). That is handoff §3C and
  is NOT in this plan; see below.

### §3A — this plan, with three named omissions

Covered: the native-consumable extent plan (Task 1), both paths consuming the
same validated policy (Task 1 reuses `SplitPolicy`), per-extent CQE accounting
that does not collapse parts into one row completion (Task 2), first-open size
validation on the descriptors native I/O actually uses (Task 3), unified EOF
handling with its own O_DIRECT test (Task 2b), and the acceptance evidence from
the handoff's own table row 1 (Task 4).

Deliberately omitted from §3A, each with a reason:

- **Generation in the CQE identity.** §3A asks for "request, row, part, and
  generation". This plan carries request+row+part but not generation, because
  `RowReader::read` returns with the ring empty (invariant I1) and publishes per
  batch, so no completion can outlive its request and be mistaken for a later
  one. Generation becomes load-bearing the moment Task 2's successor overlaps
  batches — handoff §3B — and must be added there, not retrofitted after a
  concurrency bug. Recorded as a precondition on §3B.
- **Content identity beyond size.** §3A: "Size equality detects truncation, not
  different same-size contents." Task 3 checks sizes only. The existing offline
  checker (`scripts/dsv41/verify_expert_mirror.py`) does byte comparison and
  passed 205 GB per root, so today's mirrors are known good; what is missing is
  a cheap *runtime* identity, e.g. a manifest of per-shard digests written at
  copy time and checked at open. Worth doing before mirrors are ever refreshed
  in place. Not blocking, so not in this plan.
- **Alternate-root retry on I/O error.** §3A notes it "would require
  completion-safe destination ownership". Out of scope: a mirror error still
  fails the batch, exactly as today.

### §3B, §3C, §3D, §3E — not in this plan, and why

- **§3B (bounded completion-driven pipeline).** The next structural step, and
  the natural successor to Task 2. Needs generations first (above). Deferred
  because it changes the concurrency model, and doing it on top of an unproven
  extent path would confuse two sources of risk.
- **§3C (remove the full-row CPU copy / raw-row pinned cache).** The largest
  single remaining per-row cost by our own measurement (1.58 ms fixed). Also the
  most invasive: it changes the segment copier's source addressing, and §3C
  itself warns that moving packing to the GPU can worsen load alignment because
  packed EXL3 data is not uniformly 16-byte aligned. Needs §3's stage timing
  first to confirm the scatter is on the critical path end to end, not just in
  the per-row bench.
- **§3D (drive scheduling: whole-row assignment vs within-row splitting).**
  **This one is directly relevant to an open anomaly and should be the next
  measurement, ahead of §3B and §3C.** §19 records that eager + mirrors ran 25%
  slower than eager alone while reading 1.54x more bytes. §3D predicts exactly
  this shape: "Whole-row assignment avoids waiting for two drives for every row
  and can improve tail latency; splitting can improve single-row latency.
  Neither is a universal winner." Our per-row bench measured one row per call
  (QD1), where splitting wins; production issues many rows per call, where every
  row waiting on its slowest of two drives may lose. Follow-up, not in this
  plan: measure whole-row assignment against within-row splitting at active
  counts 1/2/4/6/8, which is also what settles the §19 anomaly.
- **§3E (registered buffers, SQPOLL/IOPOLL, NUMA).** Secondary by the handoff's
  own ordering. One correction worth carrying into Task 2: the general reader
  already implements registered buffers and READ_FIXED and attempts
  SINGLE_ISSUER/DEFER_TASKRUN, while the native reader implements neither, so
  any future work here reuses proven code rather than starting fresh. The NUMA
  locality check it asks for is cheap and independent; worth doing alongside the
  §3D measurement.

### Ordering note

The handoff's own implementation order puts stage timing (its item 2) second,
before everything structural. This plan is its item 1. Item 2 does not block
Task 1-3 and is independent, but it should land before §3B or §3C are committed
to, because both are justified only by where the time actually goes.

## Self-review notes

- The advisory/prefetch path shares `RowReader::read`, so it inherits mirroring
  for free. That is intended; Task 4 Step 1 will see its bytes too.
- Tasks 1-3 are testable without a GPU. Task 4 is the only one that needs the
  lock, and its gate is falsifiable: nvme2 falls to roughly zero read volume, or
  the change did not work.
- Risk not yet retired: this plan assumes the native reader is the ONLY decode
  miss path. §19's byte accounting supports that (nvme2 carried 127 GiB with
  mirrors set, and the arithmetic closes against the baseline), but if some
  third path reads the source, Task 4 Step 1 will show nvme2 above zero and the
  cause must be found before the gate is called met.
