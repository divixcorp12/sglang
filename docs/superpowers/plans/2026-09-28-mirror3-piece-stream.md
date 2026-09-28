# Mirror3 Piece Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let RAM-miss piece streaming (with and without row images) read rows over 3 mirror roots, and in general any
N ≤ kPieces = 8. Rows over 1 or 2 roots must be cut, read and published byte for byte as they are at `1732aff4ba`.

**Architecture:** Today every nonzero part of a row is cut into up to `kSubReads = 4` sub-reads. Piece j of the row is
sub-read j, so more than 2 parts would need more than 8 pieces, and the reader refuses at startup. The plan makes the
per-part cut a function of the row: `sub_reads_per_part(reading) = min(kSubReads, kPieces / reading)`, where `reading` is
how many of that row's parts are nonzero. The function lives in `row_geometry`. Both the reader and the device's piece
table are built from that one function, so they cannot disagree. Pieces past the row's sub-read count get no bytes and
no dependencies. The reader already vets such pieces at admission and publishes them at once. That is exactly what a
1-part row does with pieces 4..7 today. So their readiness bits read as landed, and the device kernel, its 8-bit mask
and `kAllPieces` stay unchanged.

**Tech Stack:** C++20 host headers under `python/sglang/kernels/jit/csrc/moe/expert_stream/host/` (JIT-built through
tvm-ffi), CUDA device kernels (unchanged), pytest (CPU suite `test/registered/unit/kernels`, GPU suite `test/manual/dsv41`),
bash run drivers under `analysis/dsv41-drive/`.

**Spec:** The team-lead brief of 2026-09-28 ("RAM-miss piece streaming with 3 mirror roots", chosen design: keep
kPieces = 8 and the device unchanged, sub-reads per part = kPieces / parts capped at 4, empty pieces pre-landed). There is
no separate spec file. This plan argues from that brief and from the code at `1732aff4ba`, cited by line below.

### Verified against the code (read this before Task 1)

**The device needs no change.** Each claim below was checked against the code at `1732aff4ba`:

1. **Empty pieces already exist and already read as landed.** `row_geometry` (`host/piece_geometry.h:66-67`) makes the cut
   of any piece `j >= g.subs` equal to `s.bytes` in every segment. So piece j's run is `[bytes, bytes)`, which is empty,
   and `deps[j] == 0`. `queue_sub_reads` (`host/row_reader.h:735`, then `vet_pieces` at `:758`) vets every piece whose
   deps have all landed. For deps == 0 that happens at admission. `dispatch_ready_pieces` (`:1300-1303`) and the direct
   mode's `publish_landed` (`:1146-1154`) then publish a piece with no bytes at once via `publish_collected`, which sets
   its bit on every readiness word naming the row with the generation-checked CAS (`publish_piece`,
   `piece_geometry.h:95-104`). A 1-part row today has 4 sub-reads and 4 empty pieces (4..7), and the whole CUDA suite
   (`test_exl3_piece_stream_cuda.py`, built without mirror roots, so 1 part) streams such rows end to end. This plan
   reuses that mechanism. It does not add a second one.
2. **The device treats a bit as "copy this piece's runs".** It copies nothing for an empty run. `stream_copy_piece`
   (`row_copy_kernels.cuh:92-121`) skips a run with `hi <= lo`. The runs come from `ExpertStreamHost.piece_runs()`, which
   is `ffi_exports.h:525-570` calling the same `row_geometry`. Termination is `done[lane] == kAllPieces` (`:434`) after
   `served`, and the served judgement is `bits != kAllPieces` (`:415-416`). Both need all 8 bits, and the host publishes
   all 8 for any row it cuts. A READY lane (a hit) uses `kAllPieces` directly (`:346`, `:415`) and never reads a mask.
3. **Whether the bits are "pre-set".** They are not written at reservation: the tier initialises each word to
   `piece_word(generation)` with no bits. They are published at admission, before any of the row's sub-reads can land.
   The device cannot tell the difference, because its only exit needs every bit and every bit is published. The brief
   asked for pre-set bits, and this plan does not add them. Setting them at reservation would need the tier to know
   each row's geometry. That is a new code path with no behavioural gain.
4. **The real limit is `parts > kPieces`,** not `parts * kSubReads > kPieces` (`row_reader.h:106`). Every reading part
   needs at least one sub-read, and every sub-read owns one piece. `sub_reads_per_part` gives `reading * per_part <= 8`
   for every `reading` in 1..8 (static_assert in Task 2).

**One deliberate refinement of the brief:** the brief put sub-reads per part on the reader, as `kPieces / parts`. This
plan puts them on the row, as `kPieces / reading`, where `reading` counts the row's nonzero parts. The two agree on every
row when all of a table's parts are nonzero. They differ when a mirror weight is 0, which
`SGLANG_MOE_EXPERT_MIRROR_WEIGHTS` supports: `1:0:1` gives every row a zero-length part. Per reader, that row would get 2
sub-reads on each of its 2 reading parts, so 4 pieces, half the parallelism of today's 2-root recipe. Per row, it gets 4
each, which is today's 2-root cut exactly. The value must be computed in `row_geometry` anyway, because the device's
piece table (`ffi_exports.h:554`) has no reader to ask. Open question 1 asks the user to confirm this refinement.

### Every consumer of kSubReads / kPieces, and what each becomes

| Consumer | Where | Decision |
|---|---|---|
| `split_part` cut size and cap | `piece_geometry.h:13-21` | **per row:** takes `per_part` (`sub_reads_per_part(reading)`) |
| `row_geometry` split buffer `Read split[kSubReads]` | `piece_geometry.h:56` | **max:** buffer sized by the compile-time maximum |
| `row_geometry` guard `g.subs + n > kPieces` | `piece_geometry.h:58` | kept as a defensive check. It cannot fire any more for `parts <= 8` |
| `set_piece_stream` refusal | `row_reader.h:106-109` | **becomes `t_.parts > kPieces`**, with a message naming 8 and the count |
| `subs_ = on ? kSubReads : 1` (descriptor sizing) | `row_reader.h:121`, `size_extents` `:524` | **max:** descriptors stay `kBounceSlots * parts * kSubReads` |
| descriptor index `(slot*parts+part)*kSubReads + k` | `row_reader.h:737-738` | **max:** stride stays `kSubReads`. `k < per_part <= kSubReads`, so the index is unique |
| `RowGeometry` arrays `[kPieces]`, `BounceRow` deps/sub_dest/sub_done `[kPieces]`, `uint8_t` masks | `piece_geometry.h:36-39`, `row_reader.h:453-462` | unchanged: sub-reads per row stay ≤ 8 |
| `jobs_[kBounceSlots * kPieces]`, `size_jobs`, `piece_runs_` | `row_reader.h:537,564,1503` | unchanged (per piece) |
| stage trace `sub_land_seq/piece_*[kTraceRows][kPieces]` | `reader_base.h:259-262` | unchanged: indexed by sub-read ordinal (< 8) and piece |
| `extent_id = ordinal<<16 \| k<<8 \| part` | `row_reader.h:750-751` | unchanged: k < 4, part < 8 |
| `kMaxDrives = 4` trace columns | `reader_base.h:92`, `row_reader.h:229-233` | unchanged. 3 drives fit. More than 4 fold into the last column (trace only) |
| `queue_depth() = kQueueDepth * parts` | `row_reader.h:497-499` | unchanged: 48 credits for 3 parts, 6 SQEs a row |
| FFI tables `[kPieces, …]`: `piece_geometry`, `piece_runs`, checker | `ffi_exports.h:394-417,494-515,548-563` | unchanged (8 pieces, empty ones zero-length) |
| device `kRowPieces = 8`, `kAllPieces = 255u` | `row_copy_kernels.cuh:19-20` | unchanged |
| Python `STAGE_PIECES = 8` | `ops/moe/expert_stream_transport.py:423-424` | unchanged. Its comment ("the most sub-reads a row issues") stays true |
| Python `ALL_PIECES = 0xFF` | `python/sglang/test/dsv41_lease_sim.py:21` | unchanged |
| test `SUB_READS = 4` and its model `_expected_sub_reads` | `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py:48,59-65` | `SUB_READS` stays the max. The model gains `per_part` (Task 2) |

## Global Constraints

- The device kernels and their constants (`row_copy_kernels.cuh`, `lease_kernels.cuh`, `lease_device.cuh`) are not modified.
- `kPieces = 8`, `kSubReads = 4`, `kAllPieces = 0xFF` keep their values. The masks stay one byte.
- 1- and 2-part rows are cut, read (SQE stream, descriptor count, credit) and published exactly as at `1732aff4ba`, pinned by Task 1's goldens.
- The 2-root production recipe (`benchmarks/dsv41_baseline/arm_env.py`, `EXPERT_MIRROR_DIRS`) is not edited.
- Above 8 parts, piece streaming refuses at startup with an error that names the limit (8) and the count.
- Code reaches divix01 only by commit, `git push origin <branch>` and a private worktree (`.claude/rules/divix01-run-protocol.md`). No rsync or scp.
- CPU tests: `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest … -q -p no:randomly`, with pytest's status read via `echo "EXIT=${PIPESTATUS[0]}"`.
- GPU tests: `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -m pytest …`.
- Before trusting any result, print `sglang.__file__` in the worktree and read it.
- Lock order for the arms: `rowimg-disk.lock` first, then poll `cc-gpu.lock`, which `run_arm.sh` takes itself.
- Scratch and outputs on divix01 go under `/mnt/nvme1`, never `/tmp` (the root volume is ~88% full).
- Branch `cc/mirror3-piece-stream` from `origin/master` at `1732aff4ba`. Fix-up commits only: do not amend or rebase.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A mirror weight with a 0 share (`1:0:1`).** The expectation: the row reads its 2 nonzero parts with 4 sub-reads each (8 pieces), as today's 2-root recipe does, not 2 each. Pinned in Task 2 (`test_a_zero_part_gives_its_pieces_to_the_parts_that_read`) and in Tasks 3 and 4 (the `zero_middle` ids).
2. **A row whose third part is zero bytes or one page.** A row's parts can be uneven, or one can round down to nothing. The expectation: that part issues 0 or 1 sub-reads, the other pieces stay contiguous, and the rest are empty and published at admission. Pinned in Task 2 (`_one_row_tables((8, 1, 8))`, `((8, 8, 0))`, and the random 3/4/8-part layouts that zero a part in every third row).
3. **Uneven part sizes across roots (`3:1:2`).** Sub-reads differ in size from part to part, and a piece may straddle the end of one part and the start of the next, so its deps name two sub-reads on two drives. The expectation: deps are exact and the piece is vetted only when both have landed. Pinned in Task 2 (the geometry deps check over `(3,1,2)`) and Task 3 (`uneven` publish order).
4. **Prefill-sized versus decode-sized requests.** Decode asks for 1-6 rows. A prefill request can carry 16 rows over both banks: 96 SQEs against the 3-part ring's 48 credits. The expectation: credit binds without overflowing the descriptor queue, every row lands and every piece is published once. Pinned in Task 3 (`test_three_parts_a_prefill_sized_read_over_both_banks_lands_every_row`).
5. **A failed or short sub-read on the third drive after the empty pieces were published.** The word then holds some bits, including 6 and 7, but not all of them. The expectation: the device reports Failed (keep 0, go_2 0), not served, because `done != kAllPieces` and the status is not kServed. Pinned in Task 3 (`test_three_parts_a_failed_sub_read_publishes_none_of_its_pieces`) and Task 4 (G3 parametrized over three roots).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py` (create) | goldens pinning 1/2-part geometry and SQEs to `1732aff4ba` | 1 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_base.h` (modify `:55-68`) | `sub_reads_per_part`, its static_asserts, updated comment | 2 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/piece_geometry.h` (modify `:9-21`, `:42-62`) | per-row cut | 2 |
| `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py` (modify) | model gains `per_part`, N-part geometry tests (T2), 3-part reader and tier tests (T3), new refusal (T3) | 2, 3 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h` (modify `:96-122`, `:725-727`, `:1505-1508`) | refusal `parts > kPieces`, comments | 3 |
| `test/registered/unit/kernels/test_exl3_ram_miss_row_images.py` (modify `:161`, `:226-228`) | NOT_REUSED reason, 3-root direct-mode byte equality | 3 |
| `test/manual/dsv41/test_exl3_piece_stream_cuda.py` (modify) | `StreamService(mirror_weights=…)`, G12, G2/G3 over three roots | 4 |
| `test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py` (modify `:30-56`) | image fixture builds one image root per mirror | 4 |
| `analysis/dsv41-drive/nvme-load/nvme_sampler.py` (modify) | devices from argv | 6 |
| `analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh` (create) | the two arms, under the lock order, with samplers | 6 |
| `analysis/dsv41-drive/mirror3/mirror3_report.py` + `test_mirror3_report.py` (create) | ms/token, byte identity, per-drive split | 6 |

---

### Task 1: Goldens for 1- and 2-part rows at the base commit

The Python model in `test_exl3_ram_miss_piece_stream.py` is edited by Task 2, so it cannot prove that Task 2 left 1- and
2-part rows alone. This task freezes what the C++ produces at `1732aff4ba` as digests: every row's geometry, and a
piece-streaming read's SQEs, descriptor count and credit. The digests depend only on file offsets and lengths, never on
file contents. The test passes before and after every later task.

**Files:**
- Create: `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py`

**Interfaces:**
- Consumes: `piece_geometry(tables, row, expert)`, `read_rows_sqes(tables, row, experts, slots, *, direct, **faults)` from `sglang.kernels.ops.moe.expert_stream_transport`, and `ram_miss_setup(tmp_path, *, capacity, experts, mirror_weights, hidden, inter, row_images)` from `sglang.test.dsv41_ram_miss_fixtures`.
- Produces: `GOLDEN: dict[str, list]` and `_measure(tmp_path, weights, images) -> list`, used only in this file.

- [ ] **Step 1: Branch and divix01 worktree**

```bash
# laptop
git -C /home/dimitri/data/divix/sglang-nvfp4 fetch origin
git -C /home/dimitri/data/divix/sglang-nvfp4 worktree add -b cc/mirror3-piece-stream \
  /home/dimitri/data/divix/sglang-nvfp4-mirror3 origin/master
git -C /home/dimitri/data/divix/sglang-nvfp4-mirror3 log -1 --oneline   # expect 1732aff4ba
```

- [ ] **Step 2: Write the test file with an empty GOLDEN**

```python
"""Piece streaming over N mirror parts (docs/superpowers/plans/2026-09-28-mirror3-piece-stream.md).

GOLDEN pins the geometry and the piece-streaming SQE stream of 1- and 2-part rows to what the C++ produced at
1732aff4ba, before sub-reads per part became a function of the row's reading parts. The independent Python model in
test_exl3_ram_miss_piece_stream is edited by that same change, so it cannot be what shows the change left these rows
alone. The digests hash offsets and lengths only: file contents never enter them.

Regenerate (only at a commit whose 1- and 2-part cut is known good): python test_exl3_ram_miss_piece_stream_parts.py
"""

import hashlib
import json
import pathlib
import sys
import tempfile

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import piece_geometry, read_rows_sqes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# (id, mirror_weights, row_images): every 1- and 2-root shape production or the suites run.
SHAPES = [
    ("one_part", None, False),
    ("halves", (1.0, 1.0), False),
    ("zero_first_part", (0.0, 1.0), False),
    ("three_to_one", (3.0, 1.0), False),
    ("images_one_root", None, True),
    ("images_halves", (1.0, 1.0), True),
    ("images_zero_second", (1.0, 0.0), True),
]

GOLDEN = {}


def _geometry_digest(tables):
    h = hashlib.sha256()
    layers, experts = tables.extents.shape[:2]
    for row in range(layers):
        for expert in range(experts):
            h.update(json.dumps(piece_geometry(tables, row, expert), sort_keys=True).encode())
    return h.hexdigest()


def _sqe_digest(tables):
    """A two-batch piece-streaming read of 11 rows: its SQEs (as a set: the order is the refill's, pinned elsewhere),
    its descriptor count and its credit."""
    experts = list(range(11))[::-1]
    slots = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
    result, log, info, _ = read_rows_sqes(tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=2)
    assert result == 1
    return [hashlib.sha256(json.dumps(sorted(log)).encode()).hexdigest(), info["descriptors"], info["credit"]]


def _measure(tmp_path, weights, images):
    """Geometry digest, then (bounce path only) the SQE digest, descriptors and credit. The direct mode's SQEs depend
    on whether the filesystem takes O_DIRECT, so only its geometry is pinned."""
    dims = {} if images else dict(hidden=256, inter=512)
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)
    out = [_geometry_digest(s.tables)]
    if not images:
        out += _sqe_digest(s.tables)
    return out


@pytest.mark.parametrize("name, weights, images", SHAPES, ids=[shape[0] for shape in SHAPES])
def test_one_and_two_part_rows_are_cut_and_read_as_at_the_base_commit(tmp_path, name, weights, images):
    assert _measure(tmp_path, weights, images) == GOLDEN[name]


if __name__ == "__main__":
    golden = {}
    for name, weights, images in SHAPES:
        with tempfile.TemporaryDirectory(dir=sys.argv[1] if len(sys.argv) > 1 else None) as d:
            path = pathlib.Path(d) / "ckpt"
            path.mkdir()
            golden[name] = _measure(path, weights, images)
    print("GOLDEN = " + json.dumps(golden, indent=4))
```

- [ ] **Step 3: Commit, push, set up the divix01 worktree, and generate GOLDEN at the base commit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-mirror3
git add test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py
git commit -m "test(expert-stream): golden digests of 1- and 2-part piece geometry and SQEs (generator)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/mirror3-piece-stream
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-mirror3 origin/cc/mirror3-piece-stream \
  && git -C /data/models/slang/nvfp4-work/wt-mirror3 log -1 --oneline'
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && mkdir -p /mnt/nvme1/mirror3-scratch \
  && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && cd test/registered/unit/kernels && PYTHONPATH=/data/models/slang/nvfp4-work/wt-mirror3/python OMP_NUM_THREADS=8 \
     taskset -c 0-63 /data/models/slang/.venv/bin/python test_exl3_ram_miss_piece_stream_parts.py /mnt/nvme1/mirror3-scratch'
```

Expected: `sglang.__file__` is under `/data/models/slang/nvfp4-work/wt-mirror3/python`, then a `GOLDEN = {...}` block with 7
entries: 4 of length 4, `[geometry_sha, sqe_sha, descriptors, credit]`, and 3 of length 1. The `descriptors`/`credit`
pairs are `64/16` for `one_part` and `128/32` for the 2-part shapes (U10's `16*parts*4` and `16*parts`). Run the
generator a second time: it must print identical digests, which shows they do not depend on contents or temp paths.

- [ ] **Step 4: Paste the printed block over `GOLDEN = {}` and run the test at the base commit**

Replace the line `GOLDEN = {}` with the printed `GOLDEN = {...}` block verbatim.

```bash
git add -u && git commit -m "test(expert-stream): pin 1- and 2-part piece geometry and SQEs to 1732aff4ba

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `7 passed`, `EXIT=0`. The C++ is still `1732aff4ba`'s, so this is the baseline the later tasks must keep green.

---

### Task 2: Sub-reads per part as a function of the row's reading parts

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_base.h:55-68`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/piece_geometry.h:9-21,42-62`
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py` (model at `:48-65`, `_assert_geometry` at `:81-84`, `_synthetic_tables` at `:131-172`, U1 tests at `:175-197`)

**Interfaces:**
- Consumes: `Tables` (`host/row_tables.h`: `parts`, `extents[row*parts + p]` as `Read{file, offset, length, dest}`).
- Produces (C++): `constexpr int sub_reads_per_part(int reading)` in `reader_base.h`, and `inline int split_part(const Read& e, int per_part, Read* out)` in `piece_geometry.h`. `row_geometry(t, row_index, g, runs)` keeps its signature.
- Produces (Python test helpers, used by Task 3): `_per_part(reading: int) -> int`, `_row_sub_reads(tables, row, expert) -> list[tuple[int, int, tuple]]`, `_one_row_tables(part_pages: tuple[int, ...]) -> SimpleNamespace`.

- [ ] **Step 1: Write the failing tests**

In `test_exl3_ram_miss_piece_stream.py`, replace `SUB_READS = 4` through the end of `_row_sub_reads` (`:48-72`) with:

```python
PAGE = 4096
SUB_READS = 4  # the most sub-reads a part is cut into (the C++ kSubReads)
PIECES = ops.STAGE_PIECES
ALIGN = 128


def _round_up(value, to):
    return -(-value // to) * to


def _per_part(reading):
    """Sub-reads per reading part of a row that reads ``reading`` nonzero parts (the C++ sub_reads_per_part): the
    pieces shared out, at most SUB_READS. 1 or 2 -> 4 (the cut before N parts), 3 or 4 -> 2, 5 to 8 -> 1."""
    return SUB_READS if reading <= 0 else min(SUB_READS, PIECES // reading)


def _expected_sub_reads(part, per_part=SUB_READS):
    """Plan section 4.1, written out independently of the C++: len_k = round_up(ceil(len / per_part), page)."""
    file, offset, length, dest = part
    if length <= 0:
        return []
    len_k = _round_up(-(-length // per_part), PAGE)
    return [(file, offset + at, min(len_k, length - at), dest + at) for at in range(0, length, len_k)]


def _row_parts(tables, row, expert):
    return [tables.extents[row, expert, p].tolist() for p in range(tables.extents.shape[2])]


def _row_sub_reads(tables, row, expert):
    parts = _row_parts(tables, row, expert)
    per_part = _per_part(sum(length > 0 for _, _, length, _ in parts))
    out = []
    for p, part in enumerate(parts):
        out += [(p, k, sub) for k, sub in enumerate(_expected_sub_reads(part, per_part))]
    return out
```

(The file already defines `PAGE`, `PIECES`, `ALIGN` and `_round_up` at `:47-54`. Keep one copy of each.)

In `_assert_geometry`, replace

```python
    for p in range(tables.extents.shape[2]):
        file, offset, length, dest = tables.extents[row, expert, p].tolist()
        mine = [s for s in sub_reads if s["part"] == p]
        assert len(mine) <= SUB_READS
```

with

```python
    parts = _row_parts(tables, row, expert)
    per_part = _per_part(sum(length > 0 for _, _, length, _ in parts))
    assert len(sub_reads) <= PIECES
    for p, (file, offset, length, dest) in enumerate(parts):
        mine = [s for s in sub_reads if s["part"] == p]
        assert len(mine) <= per_part
```

In `_synthetic_tables`, replace the split block

```python
        if parts == 1:
            split_pages = [pages]
        else:
            # Row 0 is split evenly, so the layout always has a row with every sub-read.
            first = pages // 2 if r == 0 else rng.choice([0, pages, 1, pages - 1, rng.randrange(pages + 1)])
            split_pages = [first, pages - first]
```

with (the 1- and 2-part branches are unchanged, so their random layouts, and U1's seeds, stay what they were):

```python
        if parts == 1:
            split_pages = [pages]
        elif parts == 2:
            # Row 0 is split evenly, so the layout always has a row with every sub-read.
            first = pages // 2 if r == 0 else rng.choice([0, pages, 1, pages - 1, rng.randrange(pages + 1)])
            split_pages = [first, pages - first]
        elif r == 0:
            split_pages = [pages // parts] * (parts - 1) + [pages - pages // parts * (parts - 1)]
        else:
            # Any cut, and every third row gives one part's pages to its neighbour (a 0 share, or a rounding).
            cuts = sorted(rng.randrange(pages + 1) for _ in range(parts - 1))
            split_pages = [b - a for a, b in zip([0] + cuts, cuts + [pages])]
            if r % 3 == 1:
                z = rng.randrange(parts)
                split_pages[(z + 1) % parts] += split_pages[z]
                split_pages[z] = 0
```

Replace the two U1 layout tests (`:175-197`) with:

```python
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("parts", [1, 2, 3, 4, 8])
def test_u1_geometry_of_every_row_of_a_random_layout(seed, parts):
    tables = _synthetic_tables(seed, parts=parts)
    counts = set()
    for row in range(tables.extents.shape[0]):
        for expert in range(tables.extents.shape[1]):
            sub_reads, _ = _assert_geometry(tables, row, expert)
            counts.add(len(sub_reads))
    # Row 0 is split evenly and long enough for every sub-read: 4, 8, 6, 8, 8 of them.
    assert max(counts) == _per_part(parts) * parts


@pytest.mark.parametrize(
    "weights, dims",
    [(None, {}), ((1.0, 1.0), {}), ((0.0, 1.0), {}), ((1.0, 1.0), dict(hidden=256, inter=512)),
     ((3.0, 1.0), dict(hidden=256, inter=512)), (None, dict(hidden=256, inter=512)),
     ((1.0, 1.0, 1.0), dict(hidden=256, inter=512)), ((3.0, 1.0, 2.0), dict(hidden=256, inter=512)),
     ((1.0, 0.0, 1.0), dict(hidden=256, inter=512)), ((1.0,) * 4, dict(hidden=256, inter=512)),
     ((1.0,) * 8, dict(hidden=256, inter=512))],
    ids=["one_part", "halves", "zero_first_part", "large_halves", "small_last_part", "large_one_part",
         "thirds", "uneven_thirds", "zero_middle_third", "quarters", "eighths"],
)
def test_u1_geometry_of_every_row_of_a_real_layout(tmp_path, weights, dims):
    s = ram_miss_setup(tmp_path, capacity=2, mirror_weights=weights, **dims)
    for row in range(s.tables.extents.shape[0]):
        for expert in range(s.tables.extents.shape[1]):
            _assert_geometry(s.tables, row, expert)
```

Add after them:

```python
def _one_row_tables(part_pages):
    """One layer, one expert, one segment spanning the row: part p reads part_pages[p] pages of file p, in order."""
    parts = len(part_pages)
    total = sum(part_pages) * PAGE
    extents = torch.zeros((1, 1, parts, 4), dtype=torch.int64)
    dest = 0
    for p, pages in enumerate(part_pages):
        extents[0, 0, p] = torch.tensor([p, dest, pages * PAGE, dest])
        dest += pages * PAGE
    return SimpleNamespace(
        extents=extents,
        starts=torch.zeros((1, 1), dtype=torch.int64),
        file_sizes=torch.tensor([total] * parts, dtype=torch.int64),
        segments=torch.tensor([(0, 0, 0, total)], dtype=torch.int64),
        slabs=torch.zeros((1, 1), dtype=torch.int64),
        row_bytes=torch.tensor([total], dtype=torch.int64),
        paths=[f"/nonexistent/{p}" for p in range(parts)],
        source_paths=["/nonexistent/source"] * parts,
        slot_bytes=total,
    )


def _assert_empty_past(pieces, n):
    """Pieces 0..n-1 have bytes and dependencies; pieces n..7 have neither, so the reader publishes them at admission."""
    assert all(piece["deps"] != 0 for piece in pieces[:n])
    assert all(piece["deps"] == 0 and all(lo == hi for lo, hi in piece["runs"]) for piece in pieces[n:])


@pytest.mark.parametrize("parts, per_part", [(1, 4), (2, 4), (3, 2), (4, 2), (5, 1), (8, 1)])
def test_a_row_reading_every_part_cuts_each_into_its_share_of_the_pieces(parts, per_part):
    sub_reads, pieces = piece_geometry(_one_row_tables((8,) * parts), 0, 0)
    assert [sum(s["part"] == p for s in sub_reads) for p in range(parts)] == [per_part] * parts
    assert [(s["part"], s["k"]) for s in sub_reads] == [(p, k) for p in range(parts) for k in range(per_part)]
    _assert_empty_past(pieces, parts * per_part)


@pytest.mark.parametrize(
    "part_pages, per_part",
    [((8, 0, 8), 4), ((0, 8, 8), 4), ((8, 8, 0), 4), ((8, 0, 0), 4), ((8, 8, 8), 2), ((8, 1, 8), 2), ((1, 1, 1), 2),
     ((8, 8, 8, 0), 2), ((8, 0, 8, 0), 4)],
)
def test_a_zero_part_gives_its_pieces_to_the_parts_that_read(part_pages, per_part):
    """A zero-length part (a 0 mirror weight, or a rounding) reads nothing and does not count: the reading parts share
    all eight pieces. A part shorter than its share in pages is cut into fewer sub-reads, never an empty one."""
    tables = _one_row_tables(part_pages)
    sub_reads, pieces = piece_geometry(tables, 0, 0)
    counts = [sum(s["part"] == p for s in sub_reads) for p in range(len(part_pages))]
    assert counts == [min(per_part, pages) for pages in part_pages]
    _assert_empty_past(pieces, len(sub_reads))
    _assert_geometry(tables, 0, 0)


def test_nine_parts_cannot_be_cut():
    assert piece_geometry(_one_row_tables((1,) * 9), 0, 0) is None
```

- [ ] **Step 2: Commit, push, run on divix01, and see it fail**

```bash
git add -u && git commit -m "test(expert-stream): N-part piece geometry, sub-reads per reading part

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py -q -p no:randomly \
     -k "u1_geometry or share_of_the_pieces or zero_part or nine_parts" 2>&1 | tail -15; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: FAIL. The 3-, 4- and 8-part cases fail: `piece_geometry` returns `None`, because `g.subs + n > kPieces`
rejects 3 × 4 sub-reads, or the counts come back as `[4, 4, 4]`-style. The 1- and 2-part cases, and
`test_nine_parts_cannot_be_cut`, pass. `EXIT=1`.

- [ ] **Step 3: Implement `sub_reads_per_part` in `reader_base.h`**

Replace `reader_base.h:55-68` (the piece-streaming comment through the static_assert) with:

```cpp
// Piece streaming (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM, plan 2026-09-24-dsv41-piece-streaming §4.1-4.3; N mirror
// parts, plan 2026-09-28-mirror3-piece-stream). With it on, each nonzero part of a row is read as up to
// sub_reads_per_part(reading) page-aligned sub-reads, `reading` being how many of the row's parts are nonzero, and the
// row's needed bytes are cut into kPieces pieces: piece j is sub-read j of the row (in file order) mapped into segment
// destination coordinates, its inner cuts rounded down to kPieceAlign. Pieces past the row's sub-reads have no bytes
// and are published at admission. kPieces is fixed, not a knob: the device's readiness word carries one bit per piece.
// kSubReads is the most sub-reads a part is cut into; it also sizes the descriptors (one per slot, part and sub-read)
// and strides their index. With the flag off none of this is used and the reader issues one read per part.
constexpr int kSubReads = 4;
constexpr int kPieces = 8;
constexpr uint8_t kAllPieces = 0xFF;
constexpr int64_t kPieceAlign = 128;
// Row images (direct mode): O_DIRECT's file offset and segment length alignment on the mirror drives (XFS and ext4,
// 512 B logical blocks; exl3_row_image.IO_ALIGN). Slab rows are held to it too, which covers dio_mem_align (4).
constexpr int64_t kImageAlign = 512;
static_assert(kPieces <= 8, "a row's piece and sub-read masks are one byte each");

// Sub-reads per reading part of a row that reads `reading` nonzero parts: the pieces shared out, at most kSubReads.
// 1 or 2 reading parts give kSubReads (the cut before N parts), 3 or 4 give 2, 5 to 8 give 1. Past kPieces a part
// would get none: the reader refuses such tables (set_piece_stream) and row_geometry refuses such a row. 0 (a row with
// nothing to read, which admission refuses) gives kSubReads.
constexpr int sub_reads_per_part(int reading) {
  return reading <= 0 ? kSubReads : std::min(kSubReads, kPieces / reading);
}
constexpr bool every_part_count_fits() {
  for (int reading = 1; reading <= kPieces; ++reading) {
    const int per_part = sub_reads_per_part(reading);
    if (per_part < 1 || per_part > kSubReads || reading * per_part > kPieces) return false;
  }
  return true;
}
static_assert(every_part_count_fits(), "1..kPieces reading parts each get at least one sub-read and fit the pieces");
static_assert(
    sub_reads_per_part(1) == kSubReads && sub_reads_per_part(2) == kSubReads,
    "rows of one or two reading parts keep the cut they had before N parts");
static_assert(sub_reads_per_part(3) == 2 && sub_reads_per_part(4) == 2 && sub_reads_per_part(8) == 1, "N-part cut");
```

(`<algorithm>` is already included at `reader_base.h:10`. `std::min` is constexpr in C++14 and later.)

- [ ] **Step 4: Make `split_part` and `row_geometry` use it**

In `piece_geometry.h`, replace `:9-21` with:

```cpp
// Piece streaming, sub-reads (plan §4.1): part `e` as its sub-reads, in file order, into `out` (at most `per_part`
// entries, per_part <= kSubReads); returns how many. Each is len_k = round_up(ceil(length / per_part), kPage) bytes and
// the last takes what is left, so a part tiles exactly, a small part gives fewer than per_part and no sub-read is
// empty. A zero-length part gives none. The part's offset, dest and length are whole pages (the builder's), so every
// sub-read's are too.
inline int split_part(const Read& e, int per_part, Read* out) {
  if (e.length <= 0) return 0;
  const int64_t len_k = ((e.length + per_part - 1) / per_part + kPage - 1) / kPage * kPage;
  int n = 0;
  for (int64_t at = 0; at < e.length && n < per_part; at += len_k) {
    out[n++] = Read{e.file, e.offset + at, std::min(len_k, e.length - at), e.dest + at};
  }
  return n;
}
```

In `row_geometry`, replace

```cpp
  for (size_t p = 0; p < parts; ++p) {
    Read split[kSubReads];
    const int n = split_part(t.extents[base + p], split);
```

with

```cpp
  // The row's reading parts share the pieces (sub_reads_per_part); a zero-length part (a 0 mirror weight) reads nothing
  // and takes none, so 1:0:1 cuts like a 2-part row.
  int reading = 0;
  for (size_t p = 0; p < parts; ++p)
    reading += t.extents[base + p].length > 0 ? 1 : 0;
  if (reading > kPieces) return false;
  const int per_part = sub_reads_per_part(reading);
  for (size_t p = 0; p < parts; ++p) {
    Read split[kSubReads];
    const int n = split_part(t.extents[base + p], per_part, split);
```

Also change the doc comment's last sentence (`piece_geometry.h:48-49`) from "Returns false for a row that cannot be cut:
more sub-reads than pieces, or a piece with bytes that no sub-read reads." to "Returns false for a row that cannot be
cut: more reading parts than pieces, or a piece with bytes that no sub-read reads."

- [ ] **Step 5: Commit, push, run the geometry tests and Task 1's goldens**

```bash
git add -u && git commit -m "feat(expert-stream): cut each reading part into its share of the 8 pieces

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py \
     -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: everything passes except `test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name`. That test
still expects a 3-part refusal, and Task 3 replaces it. Task 1's 7 goldens pass unchanged. `EXIT=1` for that one
failure. If `test_u1_geometry_of_every_row_of_a_random_layout[...-8-...]` fails only on the `max(counts)` line, row 0 of
that seed is shorter than 8 pages. Print `pages` for row 0: the fix is to skip that seed for 8 parts, not to loosen the
geometry asserts.

---

### Task 3: The reader accepts up to 8 parts and publishes correct masks over 3

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h:96-122,725-727,1505-1508`
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py` (`:200-212` U1 reader test, `:534-537` refusal test, `_host` at `:563`, new tests)
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_row_images.py:161,226-228`

**Interfaces:**
- Consumes: Task 2's `_per_part`, `_row_sub_reads`, and `_sub_read(s, row, expert, part, k)` (already at `:487`). Also `read_rows_pieces(tables, row, experts, slots, *, direct, generation, **faults) -> (result, record, masks, info)`, `read_rows_sqes`, `read_rows_traced`, `piece_word`, `split._sentinel`, `split._assert_rows`, and `LeaseSim`.
- Produces: `RowReader::set_piece_stream` refusing only `t_.parts > kPieces`, with message `"piece streaming reads at most 8 mirror parts (a reading part needs a piece of its own), not <n>"`. `_host(tmp_path, workers, *, lease_mode=True, two_phase=True, piece_stream=True, weights=(1.0, 1.0))`.

- [ ] **Step 1: Write the failing tests**

Replace `test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name` (`:534-537`) with the version below. Keep
the name: `test_exl3_ram_miss_row_images.NOT_REUSED` keys on it.

```python
def test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name(tmp_path):
    s = ram_miss_setup(tmp_path, mirror_weights=(1.0,) * 9)
    with pytest.raises(RuntimeError, match="at most 8 mirror parts .* not 9"):
        read_rows_traced(s.tables, 1, [0], [0], direct=False, piece_stream=True, pack_workers=1)


@pytest.mark.parametrize("weights", [(1.0,) * 3, (1.0,) * 4, (1.0,) * 8], ids=["three", "four", "eight"])
def test_the_reader_streams_pieces_over_three_to_eight_mirror_parts(tmp_path, weights):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, piece_stream=True, pack_workers=2, poison=True
    )
    assert result == 1 and info["refused"] == 0 and record["piece_publish_refused"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    assert record["pieces_published"] == PIECES * len(experts)
    split._assert_rows(s, 1, experts, slots)
```

Parametrize `test_u1_the_geometry_is_what_the_reader_reads` (`:200`) over weights, keeping 2 parts first:

```python
@pytest.mark.parametrize(
    "weights", [(1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (3.0, 1.0, 2.0)],
    ids=["halves", "thirds", "zero_middle", "uneven"],
)
def test_u1_the_geometry_is_what_the_reader_reads(tmp_path, weights):
    """The exported geometry is the reader's own: a piece-streaming read issues exactly those sub-reads."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
```

(The rest of its body is unchanged.)

Add these tests after `test_a_sub_read_that_ends_short_leaves_its_pieces_unvetted_unpublished_and_fails_the_read`:

```python
THREE = [(1.0, 1.0, 1.0), (3.0, 1.0, 2.0), (1.0, 0.0, 1.0)]
THREE_IDS = ["thirds", "uneven", "zero_middle"]


@pytest.mark.parametrize("weights", THREE, ids=THREE_IDS)
def test_three_parts_publish_each_piece_after_its_sub_reads_and_the_empty_ones_at_admission(tmp_path, weights):
    """Three mirror parts, completions reversed and part 2's second sub-read of row 0 held back. Every piece with bytes
    is vetted after its last dependency lands and published after it is vetted. Every piece past the row's sub-reads
    (6 and 7 when all three parts read) has no dependencies and is vetted at admission, before any of the row's
    sub-reads landed, so the device's all-eight-bits exit is reached with no device change. Rows land exact."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    for slot in slots:
        split._sentinel(s, 1, slot)
    assert _sub_read(s, 1, experts[0], 2, 1)["length"] > 0  # the held sub-read exists for every weighting here
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, piece_stream=True, pack_workers=2,
        reverse_cqes=True, hold_ordinal=0, part=2, sub=1, poison=True,
    )
    assert result == 1 and info["refused"] == 0 and record["piece_publish_refused"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    assert record["pieces_published"] == PIECES * len(experts)
    split._assert_rows(s, 1, experts, slots)
    for row, expert in zip(record["pieces"], experts):
        sub_reads, pieces = piece_geometry(s.tables, 1, expert)
        n = len(sub_reads)
        assert n == len(_row_sub_reads(s.tables, 1, expert))
        assert n == (8 if 0.0 in weights else 6), (expert, n)  # 1:0:1 cuts like two parts
        first_landing = min(row["sub_seq"][k] for k in range(n))
        for j, piece in enumerate(pieces):
            assert row["publish"][j] > row["seq"][j] > 0, (row["row"], j)
            deps = [k for k in range(n) if piece["deps"] >> k & 1]
            if j >= n:
                assert not deps and row["seq"][j] < first_landing, (row["row"], j)
            else:
                assert deps and row["seq"][j] > max(row["sub_seq"][k] for k in deps), (row["row"], j)


@pytest.mark.parametrize("weights", THREE, ids=THREE_IDS)
def test_three_parts_every_bit_a_reader_can_see_names_bytes_already_stored(tmp_path, weights):
    """U3 over three parts: a thread polls the words while the read runs and checks each bit it sees against a
    reference copy, the empty pieces' bits included (they name no bytes, so they can never differ)."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=weights, hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    assert read_rows_traced(s.tables, 1, experts, ref_slots, direct=False)[0] == 1
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, reference=s.tables.slabs, ref_slots=ref_slots,
        piece_stream=True, pack_workers=2, pack_split=1, pack_delay_ns=10_000_000, poison=True,
    )
    assert result == 1 and info["refused"] == 0
    assert info["checked"] == PIECES * len(experts) and info["differed"] == 0
    assert _words(masks) == [piece_word(GEN, FULL)] * len(experts)
    split._assert_rows(s, 1, experts, slots)


@pytest.mark.parametrize(
    "fault", [dict(part_error=5), dict(part_short=PAGE, short_is_eof=True)], ids=["eio", "short_at_eof"]
)
def test_three_parts_a_failed_sub_read_publishes_none_of_its_pieces(tmp_path, fault):
    """Part 2's second sub-read of row 1 fails (EIO) or ends a page in, as at end of file. The read fails, no piece that
    depends on it is vetted or published, and nothing is published under another generation. The row's empty pieces
    may already be published: the device still sees a word short of all eight bits and a request not served."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots = [4, 1, 2], [0, 1, 2]
    sub_reads, pieces = piece_geometry(s.tables, 1, experts[1])
    bad = next(k for k, sub in enumerate(sub_reads) if (sub["part"], sub["k"]) == (2, 1))
    assert sub_reads[bad]["length"] > PAGE  # the short fault can fire
    result, record, masks, info = read_rows_pieces(
        s.tables, 1, experts, slots, direct=False, generation=GEN, piece_stream=True, pack_workers=2,
        part=2, sub=1, ordinal=1, poison=True, **fault,
    )
    assert result == 0 and info["refused"] == 0
    row1, bits = record["pieces"][1], _words(masks)[1] & FULL
    dependent = [j for j, piece in enumerate(pieces) if piece["deps"] >> bad & 1]
    assert dependent and all(row1["seq"][j] == 0 and not bits >> j & 1 for j in dependent)
    assert bits != FULL
    assert all(word >> 8 == GEN for word in _words(masks))


@pytest.mark.parametrize("credit", [1, 5, 48])
def test_three_parts_a_prefill_sized_read_over_both_banks_lands_every_row(tmp_path, credit):
    """16 rows, both banks full (the most a request carries), of three parts: 6 sub-reads a row, 96 SQEs, against the
    3-part ring's 48 credits or fewer. Every sub-read is issued once, every piece published once, every row exact."""
    s = ram_miss_setup(tmp_path, capacity=16, experts=16, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots = list(range(16))[::-1], list(range(16))
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, log, info, record = read_rows_sqes(
        s.tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=2, step=8, max_outstanding=credit
    )
    assert result == 1
    reads = sum(len(piece_geometry(s.tables, 1, e)[0]) for e in experts)
    assert reads == 16 * 6
    assert info["sqes"] == info["cqes"] == record["extents"] == reads
    assert info["descriptors"] == 16 * 3 * SUB_READS and info["credit"] == 16 * 3
    assert record["pending_max"] <= credit
    if credit < 16 * 3:
        assert record["pending_max"] == credit
    assert record["pieces_published"] == PIECES * len(experts) and record["piece_publish_refused"] == 0
    split._assert_rows(s, 1, experts, slots)
```

Change `_host` (`:563`) so a tier can be built over three roots:

```python
def _host(tmp_path, workers, *, lease_mode=True, two_phase=True, piece_stream=True, weights=(1.0, 1.0)):
    """A tier as the service builds it for piece streaming: lease mode, two-phase, packing workers, the flag."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
```

(The rest of `_host` is unchanged.) Then add after `test_the_tier_publishes_every_piece_of_a_read_row_into_each_lane_that_names_it`:

```python
def test_a_three_root_tier_publishes_all_eight_bits_into_each_lane_that_names_a_read_row(tmp_path):
    """The tier over three mirror parts: the same lanes as the two-part test, and every lane naming a row read ends
    with all eight bits (six with bytes, two empty) under the request's generation; the hit lane's word is untouched."""
    s, page, host, sim = _host(tmp_path, 2, weights=(1.0, 1.0, 1.0))
    try:
        assert s.tables.parts == 3
        _serve(host, sim, [3])
        host.enable_trace()
        req, waited = _serve(host, sim, [3, 4, 5, 4])
        assert waited.status == 1 and waited.go == 4
        assert [_piece_word_of(sim, req, lane) for lane in range(4)] == [0] + [piece_word(req.gen, FULL)] * 3
        (record,) = host.drain_trace()
        assert record["pieces_published"] == 2 * PIECES and record["piece_publish_refused"] == 0
        assert record["extents"] == 2 * 3 * 2  # two rows, three parts, two sub-reads each
        _assert_mapped(s, host, [4, 5])
    finally:
        host.stop()
```

In `test_exl3_ram_miss_row_images.py`, change the NOT_REUSED reason at `:161` to
`"test_the_reader_refuses_more_mirror_parts_than_the_pieces_can_name": "nine mirror roots of shards; the refusal is the reader's, not the mode's",`.
Also extend the direct-mode byte-equality parametrization (`:226-228`) to:

```python
@pytest.mark.parametrize(
    "weights",
    [None, (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (3.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (3.0, 1.0, 2.0)],
    ids=["one", "halves", "first", "second", "3to1", "thirds", "zero_middle", "uneven_thirds"],
)
```

- [ ] **Step 2: Commit, push, run, and see it fail**

```bash
git add -u && git commit -m "test(expert-stream): piece streaming over three to eight mirror parts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py -q -p no:randomly \
     -k "mirror_parts or three or reader_reads" 2>&1 | tail -20; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: FAIL with `RuntimeError: ... piece streaming reads at most kPieces / kSubReads mirror parts` for every 3-, 4-
and 8-part case, and the 9-part refusal test failing its `match`. `halves` passes. `EXIT=1`.

- [ ] **Step 3: Change the refusal and the comments in `row_reader.h`**

Replace `row_reader.h:96-110` (the comment and the first two checks of `set_piece_stream`) with:

```cpp
  // Piece streaming (sub_reads_per_part sub-reads per reading part, per-piece vetting, packing and publishing); off by
  // default. Before open(), or on an idle reader after it (the tier sets it before its service thread starts), since it
  // resizes the descriptor arrays and the packing queue. Refused without packing workers (the inline path has no piece
  // publisher), with more mirror parts than pieces (a reading part needs a piece of its own), or when a slab row base
  // is not kPieceAlign-aligned (a piece's cuts are aligned in the row).
  void set_piece_stream(bool on) {
    if (on) {
      if (pack_workers_ == 0 && !t_.images) {
        throw std::runtime_error(error_prefix<Layout>() + "piece streaming needs packing workers");
      }
      if (t_.parts > kPieces) {
        throw std::runtime_error(
            error_prefix<Layout>() + "piece streaming reads at most " + std::to_string(kPieces) +
            " mirror parts (a reading part needs a piece of its own), not " + std::to_string(t_.parts));
      }
```

Leave `subs_ = on ? kSubReads : 1;` (`:121`) as it is: `subs_` is now the most sub-reads per part, which sizes the
descriptors. Replace the comment at `:725-727` with:

```cpp
  // Piece streaming: queue the row's sub-reads, one descriptor each, (slot, part, k) -> (slot * parts + part) *
  // kSubReads + k: the stride is the most sub-reads a part can have, and k < sub_reads_per_part <= kSubReads, so the
  // index is unique whatever the row's cut. Credit is untouched: refill() takes it per SQE, so a sub-read costs one
  // like a part did. Pieces with no bytes (past the row's sub-reads) are vetted here, at admission.
```

Replace the comment at `:1505-1508` with:

```cpp
  // Piece streaming (set_piece_stream; off by default). subs_ is the most sub-reads per part: 1 with the flag off,
  // which makes descriptor (slot, part, sub) the old (slot, part), and kSubReads with it on whatever a row's cut
  // (row_geometry cuts each reading part into sub_reads_per_part <= kSubReads). sub_reads_ holds each live sub-read's
  // Read (a descriptor points into it), piece_runs_ each slot's piece runs, geometry_ a batch's rows between
  // validation and admission. All sized at open() or set_piece_stream(), and empty with the flag off.
```

Check that `<string>` is included in `reader_base.h` (`:35`), for `std::to_string`.

- [ ] **Step 4: Commit, push, run the piece-stream, row-images, parts and split suites**

```bash
git add -u && git commit -m "feat(expert-stream): piece streaming refuses only more than 8 mirror parts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py \
     test/registered/unit/kernels/test_exl3_ram_miss_row_images.py test/registered/unit/kernels/test_exl3_ram_miss_split.py \
     test/registered/unit/kernels/test_exl3_ram_miss_pack_workers.py -q -p no:randomly 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: all pass, `EXIT=0`. This includes Task 1's goldens (1- and 2-part rows unchanged), the direct-mode clones of
the new 3-part tests (`row_images_with_pieces`), and `test_the_direct_mode_leaves_every_slab_byte_as_the_bounce_path_does[pieces-thirds]`.

---

### Task 4: The stream kernel end to end over three roots (GPU)

**Files:**
- Modify: `test/manual/dsv41/test_exl3_piece_stream_cuda.py` (`StreamService.__init__` `:141-165`, `test_g2_…` `:418`, `test_g3_…` `:440`, new G12)
- Modify: `test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py:30-56`

**Interfaces:**
- Consumes: `exl3_ram_miss_tables(layout, segments, slabs, *, roots, policy, source_root, row_images=None)`, `StaticSplitPolicy(weights)` (`sglang.srt.layers.moe.exl3_read_split`), `ExpertStreamHost.piece_runs() -> int32[layers, experts, 8, segments, 2]`, and `piece_geometry`.
- Produces: `StreamService(tmp_path, *, …, mirror_weights: Optional[tuple[float, ...]] = None)`. The mirror roots are `tmp_path.parent / f"{tmp_path.name}_mirror{i}"`, copies of the fake checkpoint. `self.tables.parts == len(mirror_weights)`.

- [ ] **Step 1: Write the failing GPU tests**

In `StreamService.__init__`, add `mirror_weights=None` to the keyword arguments (after `sm_small=False`) and replace

```python
            self.tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs)
```

with

```python
            mirrors = {}
            if mirror_weights is not None:
                import shutil

                from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy

                roots = tuple(str(tmp_path.parent / f"{tmp_path.name}_mirror{i}") for i in range(len(mirror_weights)))
                for root in roots:
                    shutil.copytree(tmp_path, root)
                mirrors = dict(roots=roots, policy=StaticSplitPolicy(mirror_weights), source_root=str(tmp_path))
            self.tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs, **mirrors)
```

Add `piece_geometry` to the `expert_stream_transport` import list at the top. Parametrize G2 and G3 over roots. Only
their first lines change:

```python
@pytest.mark.parametrize("weights", [None, (1.0, 1.0, 1.0)], ids=["one_root", "three_roots"])
def test_g2_s_copies_a_piece_before_the_rest_are_published(tmp_path, weights):
    s = StreamService(tmp_path, timeout_ms=1000, mirror_weights=weights)
```

```python
@pytest.mark.parametrize("weights", [None, (1.0, 1.0, 1.0)], ids=["one_root", "three_roots"])
def test_g3_a_failed_read_fails_as_failed_well_under_the_deadline(tmp_path, weights):
    """keep 0, go_2 0 (so the DIRECT commit, which masks lanes below go_total by keep, commits nothing of S's), the
    terminal names the S lane, and the reason is Failed at the read's end, not Timeout at the deadline (M14). Over three
    roots, a row the reader admitted before failing has already published its empty pieces 6 and 7. S may see those
    bits, and it must still end Failed."""
    s = StreamService(tmp_path, timeout_ms=2000, mirror_weights=weights)
```

Add a new section before `# G3, T4 (device half) and T5`:

```python
# ---------------------------------------------------------------------------------------------------------------
# G12: three mirror roots (plan 2026-09-28-mirror3-piece-stream). Each reading part is cut into 2 sub-reads (4 when a
# part is empty), so pieces 6 and 7 of a row reading all three have no runs in S's table and are published at
# admission. S's exit (done == all eight bits) and its served judgement must hold with no device change.
# ---------------------------------------------------------------------------------------------------------------
THREE_ROOTS = [(1.0, 1.0, 1.0), (3.0, 1.0, 2.0), (1.0, 0.0, 1.0)]


@pytest.mark.parametrize("weights", THREE_ROOTS, ids=["thirds", "uneven", "zero_middle"])
def test_g12_three_mirror_roots_stream_every_piece_byte_exact(tmp_path, weights):
    experts, later = [3, 5, 9, 12], [3, 7, 11]  # later: 3 is resident by then (a READY lane), 7 and 11 are misses
    s = StreamService(tmp_path, mirror_weights=weights)
    try:
        assert s.tables.parts == 3
        runs = s.host.piece_runs()
        for expert in experts + later:
            subs = len(piece_geometry(s.tables, s.row, expert)[0])
            assert subs == (8 if 0.0 in weights else 6), (expert, subs)
            empty = runs[s.row, expert, subs:]
            assert bool((empty[..., 0] == empty[..., 1]).all()), expert  # S copies nothing for pieces past subs
        s.host.inject_fault(pack_delay_ns=PIECE_DELAY_NS, poison=True)
        acked = s.counters()["leases_acked"]
        s.plan(experts)
        s.step()
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        assert int(s.dev.go_2.item()) == len(experts) and s.stats()["stream_pieces"] > 0
        assert s.delivered(experts)
        assert s.until(lambda: s.counters()["leases_acked"] == acked + len(experts)), s.counters()
        s.plan(later)
        s.step()
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        assert s.delivered(later)
    finally:
        s.quiet()
        s.close()
```

In `test_exl3_piece_stream_row_images_cuda.py`, replace `image_tables` inside the `row_images` fixture (`:37-48`) with
a version that builds one image root per mirror root. Every test of the CUDA suite is cloned into this file, G12
included:

```python
    def image_tables(layout, segments, slabs, **mirrors):
        # G1's real service passes row_images=None (the flag is off in its environment). A StreamService built with
        # mirror_weights passes roots, policy and source_root: images then go on one root per mirror, split alike.
        mirrors = {key: value for key, value in mirrors.items() if value is not None}
        source = os.path.dirname(next(iter(layout.records.values())).path)
        if mirrors:
            assert set(mirrors) == {"roots", "policy", "source_root"}, sorted(mirrors)
            roots = [source.rstrip("/") + f"_images{i}" for i in range(len(mirrors["roots"]))]
            policy = mirrors["policy"]
        else:
            roots, policy = [source.rstrip("/") + "_images"], StaticSplitPolicy((1.0,))
        write_row_images(layout, segments, source, roots, sorted(slabs))
        images = open_row_images(roots, layout, segments, source, sorted(slabs))
        return tables_of(
            layout, segments, slabs, roots=roots, policy=policy, source_root=source, row_images=images,
        )
```

- [ ] **Step 2: Commit, push, run under the GPU lock**

```bash
git add -u && git commit -m "test(expert-stream): the stream kernel over three mirror roots (G12; G2 and G3 over three)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
     /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_exl3_piece_stream_cuda.py \
     test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py -q -p no:randomly 2>&1 | tail -12; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: every test passes, `EXIT=0`. That covers the G12 × 3 weightings on both the bounce path and row images,
G2/G3 `three_roots`, and every pre-existing test. Task 3 already landed the C++, so these tests do not have a red
phase. To show they are not vacuous, run the mutant in Step 3.

- [ ] **Step 3: Mutant: prove G12 needs Task 2's per-row cut**

Apply this in the divix01 worktree only, and never commit it. Put back the fixed 4 sub-reads per part while leaving
Task 3's relaxed refusal in place. In `row_geometry`, change `const int per_part = sub_reads_per_part(reading);` to
`const int per_part = kSubReads;`. A row reading three parts then needs 12 sub-reads, so `row_geometry` refuses it,
admission fails and the request must end Failed.

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 \
  && sed -i "s/const int per_part = sub_reads_per_part(reading);/const int per_part = kSubReads;/" \
     python/sglang/kernels/jit/csrc/moe/expert_stream/host/piece_geometry.h && git diff --stat \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
     /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_exl3_piece_stream_cuda.py -q -p no:randomly -k g12 2>&1 | tail -5; \
  echo "EXIT=${PIPESTATUS[0]}"; git checkout -- python/sglang/kernels/jit/csrc/moe/expert_stream/host/piece_geometry.h && git status --short'
```

Expected: G12 `thirds` and `uneven` fail (`EXIT=1`): keep is 0, or the piece-runs assert fails first. `zero_middle`
passes, because it reads two parts. After the checkout, `git status --short` is empty. Re-run Step 2's command and
record that it is green again.

---

### Task 5: Whole-suite comparison against the base

**Files:** none changed. This is the verification gate before any run on the production recipe.

**Interfaces:**
- Consumes: the branch head from Task 4, and `1732aff4ba`.
- Produces: two recorded suite counts, with their commands, quoted in Task 6's report.

- [ ] **Step 1: Run the registered kernels suite at the merge base, in a private worktree**

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-mirror3-base 1732aff4ba \
  && cd /data/models/slang/nvfp4-work/wt-mirror3-base \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

- [ ] **Step 2: The same at the branch head**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git status --short \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: both `EXIT=0`. The head's passed count equals the base's plus the tests Tasks 1-3 added, and its failed
count is 0. Record both lines with their command. Then remove the base worktree:
`ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-mirror3-base'`.

---

### Task 6: The 3-root decode arm against the 2-root reference on divix01

**Files:**
- Modify: `analysis/dsv41-drive/nvme-load/nvme_sampler.py` (devices from argv)
- Create: `analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh`
- Create: `analysis/dsv41-drive/mirror3/mirror3_report.py`
- Create: `analysis/dsv41-drive/mirror3/test_mirror3_report.py`

**Interfaces:**
- Consumes: `benchmarks/dsv41_baseline/run_arm.sh <arm> <port> [KEY=VAL …]` (writes `$DSV41_RUN_ROOT/servers/<arm>/run-<ts>/` holding `results.jsonl` and `boundary-samples.jsonl`, whose rows carry `label` = `server_ready` / `session_<id>` and `monotonic`), and `generations.register(tree, label)`.
- Produces: `mirror3_report.py <ref_run_dir> <ref_diskstats> <new_run_dir> <new_diskstats> <dev>...`, which prints JSON `{"ref": {...}, "new": {...}, "identical": {...}}`. Also `drive_split(samples, start, end, devices) -> {"seconds": float, "drives": {dev: {read_MB, MBps, iops, req_kB, util_pct, share_pct}}}`.

- [ ] **Step 1: Write the failing report test**

`analysis/dsv41-drive/mirror3/test_mirror3_report.py`:

```python
import json

import mirror3_report as report


def _sample(mono, per_dev):
    fields = dict(reads=0, sectors_read=0, io_ticks_ms=0)
    return {"mono": mono, **{d: {**fields, **v} for d, v in per_dev.items()}}


def test_drive_split_takes_the_deltas_inside_the_window_and_shares_the_bytes():
    samples = [
        _sample(0.0, {"a": dict(reads=0, sectors_read=0), "b": dict(reads=0, sectors_read=0)}),
        _sample(10.0, {"a": dict(reads=100, sectors_read=2000), "b": dict(reads=50, sectors_read=2000)}),
        _sample(20.0, {"a": dict(reads=300, sectors_read=6000, io_ticks_ms=5000), "b": dict(reads=150, sectors_read=4000)}),
        _sample(99.0, {"a": dict(reads=999, sectors_read=99999), "b": dict(reads=999, sectors_read=99999)}),
    ]
    out = report.drive_split(samples, 5.0, 25.0, ["a", "b"])  # the window holds the samples at 10 and 20
    assert out["seconds"] == 10.0
    a, b = out["drives"]["a"], out["drives"]["b"]
    assert a["read_MB"] == 4000 * 512 / 1e6 and b["read_MB"] == 2000 * 512 / 1e6
    assert a["iops"] == 20.0 and a["req_kB"] == 4000 * 512 / 1e3 / 200
    assert round(a["share_pct"] + b["share_pct"], 9) == 100.0 and a["share_pct"] > b["share_pct"]
    assert a["util_pct"] == 50.0


def test_identity_compares_reasoning_and_content_per_turn(tmp_path):
    for name, text in (("ref", "x"), ("new", "y")):
        d = tmp_path / name
        d.mkdir()
        rows = [dict(session_id="s1", turn=0, reasoning="r", content="c", decode_tokens_per_sec=10.0),
                dict(session_id="s2", turn=0, reasoning="r", content=text, decode_tokens_per_sec=8.0)]
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert report.identity(tmp_path / "ref", tmp_path / "new") == {"s1/t0": True, "s2/t0": False}
    per_turn, median = report.ms_per_token(tmp_path / "ref")
    assert per_turn == {"s1/t0": 100.0, "s2/t0": 125.0} and median == 112.5


def test_timed_window_runs_from_server_ready_to_the_last_session(tmp_path):
    rows = [dict(label="before_server", monotonic=1.0), dict(label="server_ready", monotonic=5.0),
            dict(label="session_a", monotonic=9.0), dict(label="session_b", monotonic=12.0)]
    (tmp_path / "boundary-samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert report.timed_window(tmp_path) == (5.0, 12.0)
```

- [ ] **Step 2: Run it and see it fail**

```bash
git add analysis/dsv41-drive/mirror3/test_mirror3_report.py && git commit -m "test(mirror3): report arithmetic

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && cd analysis/dsv41-drive/mirror3 && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly . 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `ModuleNotFoundError: No module named 'mirror3_report'`, `EXIT=2` (a collection error).

- [ ] **Step 3: Write `mirror3_report.py`**

```python
"""The 3-root piece-streaming arm against its 2-root reference (plan 2026-09-28-mirror3-piece-stream Task 6).

ms/token per turn (1000 / decode_tokens_per_sec), byte identity of every turn's reasoning and content, and each mirror
drive's read split over the arm's timed window: server_ready to the last session's boundary sample, both on
CLOCK_MONOTONIC (boundary-samples.jsonl's "monotonic", nvme_sampler.py's "mono"). The window includes each session's
prefill as well as its decode.

Usage: mirror3_report.py <ref_run_dir> <ref_diskstats.jsonl> <new_run_dir> <new_diskstats.jsonl> <device>...
"""

import json
import statistics
import sys
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def timed_window(run_dir):
    rows = load_jsonl(Path(run_dir) / "boundary-samples.jsonl")
    start = next(r["monotonic"] for r in rows if r["label"] == "server_ready")
    ends = [r["monotonic"] for r in rows if r["label"].startswith("session_")]
    if not ends:
        raise ValueError(f"{run_dir}: no timed session finished")
    return start, max(ends)


def drive_split(samples, start, end, devices):
    inside = [s for s in samples if start <= s["mono"] <= end]
    if len(inside) < 2:
        raise ValueError(f"fewer than two diskstats samples in [{start}, {end}]")
    a, b = inside[0], inside[-1]
    dt = b["mono"] - a["mono"]
    drives = {}
    for d in devices:
        reads = b[d]["reads"] - a[d]["reads"]
        read_bytes = (b[d]["sectors_read"] - a[d]["sectors_read"]) * 512
        drives[d] = {
            "read_MB": read_bytes / 1e6,
            "MBps": read_bytes / 1e6 / dt,
            "iops": reads / dt,
            "req_kB": read_bytes / 1e3 / reads if reads else 0.0,
            "util_pct": 100 * (b[d]["io_ticks_ms"] - a[d]["io_ticks_ms"]) / (dt * 1e3),
        }
    total = sum(v["read_MB"] for v in drives.values())
    for v in drives.values():
        v["share_pct"] = 100 * v["read_MB"] / total if total else 0.0
    return {"seconds": dt, "drives": drives}


def _turns(run_dir):
    return {f'{r["session_id"]}/t{r["turn"]}': r for r in load_jsonl(Path(run_dir) / "results.jsonl") if "error" not in r}


def ms_per_token(run_dir):
    per_turn = {k: 1000.0 / r["decode_tokens_per_sec"] for k, r in _turns(run_dir).items() if r.get("decode_tokens_per_sec")}
    return per_turn, statistics.median(per_turn.values())


def identity(ref_dir, new_dir):
    ref, new = _turns(ref_dir), _turns(new_dir)
    if ref.keys() != new.keys():
        raise ValueError(f"the arms ran different turns: {sorted(ref)} vs {sorted(new)}")
    return {k: (ref[k].get("reasoning"), ref[k].get("content")) == (new[k].get("reasoning"), new[k].get("content")) for k in sorted(ref)}


def arm(run_dir, diskstats, devices):
    per_turn, median = ms_per_token(run_dir)
    start, end = timed_window(run_dir)
    return {"run_dir": str(run_dir), "ms_per_token": per_turn, "median_ms_per_token": median,
            "disk": drive_split(load_jsonl(diskstats), start, end, devices)}


def main(argv):
    ref_dir, ref_disk, new_dir, new_disk, *devices = argv
    out = {"ref": arm(ref_dir, ref_disk, devices), "new": arm(new_dir, new_disk, devices),
           "identical": identity(ref_dir, new_dir)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Let the sampler take its devices from argv**

In `analysis/dsv41-drive/nvme-load/nvme_sampler.py`, replace the docstring's first line and the `DEVICES = …` and
`out_path, done_log = …` lines with:

```python
"""Samples /proc/diskstats for the given NVMe namespaces every 100 ms until the arm driver logs DRIVER DONE.

Usage: nvme_sampler.py <out.jsonl> <done_log> [device ...]  (default: nvme0n1 nvme1n1 nvme2n1 nvme3n1)
```

```python
out_path, done_log = sys.argv[1], sys.argv[2]
DEVICES = tuple(sys.argv[3:]) or ("nvme0n1", "nvme1n1", "nvme2n1", "nvme3n1")
```

After the `for line in f:` loop that fills `sample` (before `out.write`), add:

```python
        missing = [d for d in DEVICES if d not in sample]
        if missing:
            sys.exit(f"not in /proc/diskstats: {missing}")
```

- [ ] **Step 5: Write the driver `drive_mirror3_arms.sh`**

```bash
#!/usr/bin/env bash
# Plan 2026-09-28-mirror3-piece-stream Task 6: the production recipe's decode arm over arm_env's 2 mirror roots
# (reference) and over 3, at one commit, each with a /proc/diskstats sampler on the three mirror drives.
# Lock order: rowimg-disk.lock is held across both arms; cc-gpu.lock is polled here and taken by run_arm.sh itself.
# Usage: drive_mirror3_arms.sh <worktree> <sha> <out_dir under /mnt/nvme1> [port]
set -u
WT=${1:?worktree}; SHA=${2:?commit}; OUT=${3:?out dir}; PORT=${4:-30021}
PY=/data/models/slang/.venv/bin/python
ROOTS3=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash:/mnt/nvme2/dsv41_flash
say() { echo "$(date +%T) $*"; }
case $OUT in /mnt/nvme1/*) ;; *) say "out dir must be under /mnt/nvme1"; exit 1 ;; esac
cd "$WT" || exit 1
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
[ -z "$(git status --porcelain)" ] || { say "worktree dirty"; exit 1; }
PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)"
mkdir -p "$OUT"
DEVS=$(for r in ${ROOTS3//:/ }; do basename "$(findmnt -no SOURCE --target "$r")"; done | sort -u | tr '\n' ' ')
say "mirror drives: $DEVS"
[ "$(echo $DEVS | wc -w)" = 3 ] || { say "the three roots are not on three devices: $DEVS"; exit 1; }
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"

run_one() {  # <arm> [KEY=VAL ...]
    local arm=$1; shift
    local done_log=$OUT/$arm.done
    : > "$done_log"
    while ! flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true; do say "cc-gpu.lock held; waiting"; sleep 60; done
    setsid nohup taskset -c 20-23 $PY "$WT/analysis/dsv41-drive/nvme-load/nvme_sampler.py" \
        "$OUT/$arm-diskstats.jsonl" "$done_log" $DEVS > "$OUT/$arm-sampler.log" 2>&1 < /dev/null &
    say "arm $arm start"
    EXPECT_SHA=$SHA DSV41_WORKTREE=$WT bash "$WT/benchmarks/dsv41_baseline/run_arm.sh" "$arm" "$PORT" "$@"
    local rc=$?
    echo "DRIVER DONE rc=$rc" >> "$done_log"
    say "arm $arm rc=$rc"
}

run_one mirror2-ref
run_one mirror3 "SGLANG_MOE_EXPERT_MIRROR_DIRS=$ROOTS3"
root=${DSV41_RUN_ROOT:-/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline}
say "ref run: $(ls -td "$root"/servers/mirror2-ref/run-* | head -1)"
say "new run: $(ls -td "$root"/servers/mirror3/run-* | head -1)"
say "devices: $DEVS"
say "DRIVER DONE"
```

- [ ] **Step 6: Run the report test, commit and push**

```bash
chmod +x analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh
git add analysis/dsv41-drive/mirror3 analysis/dsv41-drive/nvme-load/nvme_sampler.py
git commit -m "feat(mirror3): 2- vs 3-root arm driver, diskstats sampler devices, report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin cc/mirror3-piece-stream
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-mirror3 && git fetch origin && git checkout --detach origin/cc/mirror3-piece-stream \
  && cd analysis/dsv41-drive/mirror3 && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly . 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `3 passed`, `EXIT=0`.

- [ ] **Step 7: Preflight on divix01 (no GPU, no locks)**

```bash
ssh divix01 'for r in /mnt/nvme0/dsv41_flash /mnt/nvme4/dsv41_flash /mnt/nvme2/dsv41_flash; do \
    echo "$r -> $(findmnt -no SOURCE --target $r)  $(du -sh $r | cut -f1)"; done; \
  df -h /mnt/nvme1 /; \
  cd /data/models/slang/nvfp4-work/wt-mirror3/benchmarks/dsv41_baseline \
  && PYTHONPATH=. /data/models/slang/.venv/bin/python -c "import generations; print(generations.register(\"/data/models/slang/nvfp4-work/wt-mirror3/python\", \"mirror3-$(git rev-parse --short HEAD)\"))"'
```

Expected: three distinct devices, three roots of matching size, and the tree registered. If `/mnt/nvme2/dsv41_flash`'s
size differs from the other two by more than the row images, stop and ask (open question 3). Server startup refuses a
root whose images or shard copies do not match, and that refusal would cost a whole arm.

- [ ] **Step 8: Run both arms**

```bash
ssh divix01 'OUT=/mnt/nvme1/mirror3-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cd /data/models/slang/nvfp4-work/wt-mirror3 \
  && setsid nohup bash analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh $PWD $(git rev-parse HEAD) $OUT \
     > $OUT/drive.log 2>&1 < /dev/null & echo $OUT'
```

Poll `tail -5 $OUT/drive.log` about every 10 minutes: each arm takes ~200 s to start, then warm-up and 2 sessions. The
run is over when the log shows `DRIVER DONE` and both `rc=0`. If an arm aborts, read its `run.log` before re-running.
For the 3-root arm, check `server.log` for a mirror or row-image refusal first.

- [ ] **Step 9: Report**

```bash
ssh divix01 'OUT=<the out dir>; REF=<ref run line>; NEW=<new run line>; DEVS="<devices line>"; \
  cd /data/models/slang/nvfp4-work/wt-mirror3/analysis/dsv41-drive/mirror3 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python mirror3_report.py $REF $OUT/mirror2-ref-diskstats.jsonl \
     $NEW $OUT/mirror3-diskstats.jsonl $DEVS | tee $OUT/report.json; \
  cd ../../../benchmarks/dsv41_baseline && /data/models/slang/.venv/bin/python paired.py $REF $NEW | tee $OUT/paired.txt'
```

Report to the lead:
- ms/token per turn and the median for each arm, and `paired.py`'s ratio and sign-test p. The p is at best 0.25 over 2 sessions, so the result is directional.
- Byte identity: `identical` must be all `true` (2 of 2). A `false` is a correctness failure. Stop and debug it; do not quote the speed.
- Per drive and per arm: `read_MB`, `MBps`, `iops`, `req_kB`, `util_pct`, `share_pct`. With 3 roots at equal weights, expect about 33% each, and `req_kB` about 4/3 of the 2-root arm's, because 6 sub-reads replace 8 per row.
- Task 5's two suite lines, with their commands.
- `nvme2` also holds the checkpoint (`EXPERT_DIR`) and the benchmark corpus. Its share in the 2-root arm is the non-mirror baseline to subtract.

---

## Self-review

1. **Spec coverage.**
   - Refusal `parts > kPieces` with a clear error: Task 3.
   - 1- and 2-part byte-for-byte equivalence: Task 1 goldens, re-run in Tasks 2, 3 and 5.
   - The 2-root recipe is untouched: arm_env is not edited, and Task 6 runs it as the reference.
   - CPU geometry for parts 1, 2, 3, 4 and 8 (pieces, sub-reads, deps, empty bits): Task 2.
   - Reader masks with 3 parts, including short and failed reads: Task 3.
   - Tier lease-page words: Task 3's `test_a_three_root_tier_…`.
   - GPU end to end over 3 roots, with byte compare: Task 4, bounce and row images.
   - The 3-root and 2-root runs with ms/token, byte identity and diskstats splits under the lock order: Task 6.
   - Every kSubReads consumer and Python mirror: the table in the header.
   - Whether the device must change: answered in the header, and Task 4's mutant proves G12 needs the per-row cut.
2. **Placeholder scan.** GOLDEN is generated by an exact command at the base commit (Task 1 Step 3), not invented. Step 9's `<the out dir>` and the run lines are values Step 8's driver prints. No TBD or TODO remains.
3. **Type consistency.**
   - `sub_reads_per_part(int)` and `split_part(const Read&, int, Read*)` are used only inside `piece_geometry.h`.
   - `_per_part`, `_row_sub_reads`, `_one_row_tables`, `THREE`, `THREE_IDS` and `_host(..., weights=)` are defined in Tasks 2 and 3 before they are used.
   - `StreamService(mirror_weights=)` is used by G2, G3 and G12, and by the image fixture through the `roots`, `policy` and `source_root` keys.
   - `drive_split`, `ms_per_token`, `identity` and `timed_window` match the report test.
4. **Review Focus.** Each of the five has a named test in its owning task: a 0 share, a zero or one-page third part, `3:1:2`, a prefill-sized request, and a failure after the empty pieces were published.

## Open questions for the user

1. **Per-row or per-reader sub-reads.** This plan cuts per row, by the row's nonzero parts, so `1:0:1` keeps 4 sub-reads per part. The brief said per reader, by `parts`. The two differ only when a mirror weight is 0. Confirm per-row is acceptable.
2. **Empty pieces are published at admission, not pre-set at reservation.** The device cannot tell the difference, and this is today's 1-part path. Pre-setting them would need the tier to compute geometry at reservation. Confirm there is no other reason, such as trace or bookkeeping, to want the bits set at reservation.
3. **`/mnt/nvme2` carries three loads:** the checkpoint (`SGLANG_DSV41_EXPERT_DIR`), the benchmark corpus, and now a mirror. Should the 3-root arm also be compared against a weight that favours the other two drives (for example `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=2:2:1`)? And does `/mnt/nvme2/dsv41_flash` hold the shard copies as well as the row images? Startup checks both.
4. **Should the 3-root recipe become production** (`arm_env.EXPERT_MIRROR_DIRS`) if Task 6 shows a win? This plan leaves that as a separate change.
