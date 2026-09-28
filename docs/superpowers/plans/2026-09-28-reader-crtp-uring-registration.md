# Reader CRTP Split and io_uring Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the expert-stream host `RowReader` into a CRTP base holding the read pipeline, plus two derived readers (`PackReader` for the bounce-and-pack path, and `RowReader` for the direct row-image path), without changing behavior. Then make `io_uring_register_buffers` and `io_uring_register_files` work for the tier's real slab sizes, which exceed io_uring's 1 GiB per-buffer limit.

**Architecture:** `ReaderCore<Derived, Layout, Reader>` owns everything that is the same for both paths: descriptors, generations, ring credit, admission, sub-reads and piece vetting, reap/process/retire, faults and trace. The derived readers answer only "where does this read land", "what happens once bytes land", and "what memory and threads to open". The tier keeps one `Source` type: `AnyReader`, a `std::variant` of the two readers chosen by `Tables::images`.

Registration reuses the chunked logic that `io/uring_file_reader.cpp` already has (a sparse table plus `update_tag` slots), moved into a shared `io/registered_buffers.h`. It adds **row-aligned** cutting: one contiguous slab per name per layer stays exactly as allocated and CUDA-pinned, and it is *registered* in chunks of the largest multiple of its `row_bytes` that is ≤ 1 GiB, with the remainder last. So every slab row, and therefore every iovec, lies in exactly one registered buffer, and nothing is ever clipped.

The bounce path reads one slot per SQE and never needs more. A row-image read whose iovecs span several named slabs (k registered buffers) **fans out**: it becomes k legs, one fixed SQE per run of iovecs sharing a buffer, submitted together, and it completes when its last leg lands.
- Credit still counts SQEs, and a read's legs are reserved all-or-nothing.
- A short leg resubmits alone.
- A failing leg fails the call once, after `drain` has reaped every leg.
- A piece bit is published only after every leg of its sub-read has landed.
- Default reads are a single leg, which is exactly today's code.

**Tech Stack:** C++20 header-only JIT host module (tvm-ffi), liburing 2.12, Linux io_uring (`READ_FIXED`, `READV_FIXED`, `IOSQE_FIXED_FILE`), pytest, the dsv41 decode-arm harness (`benchmarks/dsv41_baseline/run_arm.sh`).

**Spec:** No separate spec doc. The requirements are the user's request and the team-lead brief of 2026-09-28, both quoted in **Requirements** below. Supporting evidence is on `origin/arm/sleepfree-cand`, in `analysis/dsv41-drive/uring-config/HANDOFF.md` (design of the Codex option layer) and `results.md` (the 2026-09-28 option campaign). Read both.

**Base and dependency (read first):**
- **Execution waits until `mirror3-piece-stream` is merged into master** (user decision, 2026-09-28). Do not start Task 0 before that merge. At that point, cut `cc/reader-crtp-uring` from the merge's master commit and use it wherever this plan says `b3f887aaa9` or `origin/mirror3-piece-stream`: as the base, as the merge-base for suite counts, and as the golden's generation commit. Re-run the Task 2 cherry-pick dry run there first.
- Branch `cc/reader-crtp-uring` is cut from `origin/mirror3-piece-stream` at `b3f887aaa9`, not from master. That branch changes `row_reader.h`/`reader_base.h`/`piece_geometry.h` (per-row `sub_reads_per_part`, up to 8 parts for 3 mirror roots). A separate agent is finishing it, and the user has not yet decided to merge it into master. This plan must not touch `mirror3-piece-stream`. If that branch moves before this one lands, **merge** `origin/mirror3-piece-stream` (or master, once it holds it) into `cc/reader-crtp-uring`. Never rebase. If `mirror3-piece-stream` is abandoned, re-base the work by merging this branch's commits onto master in a new branch. That is a user decision, recorded at the end.
- The io_uring knobs (`SGLANG_EXPERT_STREAM_URING_*`) exist **only** on `origin/arm/sleepfree-cand` and `origin/codex/sleep-free-lease-wait` (verified 2026-09-28: master and mirror3 have no fixed-buffer or fixed-file code in `expert_stream/`; master's only io_uring env var is `SGLANG_URING_FILE_READER_QUEUE_DEPTH`) (Codex commits `a5b4c87bb7 9dad1512e9 e0e79ddfd6 ba58e8a5a1 49d3c6b840`), not on master or mirror3. Task 2 cherry-picks those five commits. **The plan ports those knobs as they are and adds none.** Only a test-only fault word lowers the chunk cap. A dry run on 2026-09-28 applied all five cleanly onto `b3f887aaa9`. The rest of `arm/sleepfree-cand` (sleep-free lease wait, stream-wait kernels) is **not** taken.

## Requirements

User, verbatim: "i want to make a couple of changes to the row_reader class. I want to first make two versions, a crtp base class with the base logic, then a derived pack_reader and a derived (non packed) row_reader. I also want to get io_uring_register_buffers/io_uring_register_files working."

What the code says "packed" means (mapped 2026-09-28 on `b3f887aaa9`): today one class, `RowReader<Layout, Reader>` (`host/row_reader.h`, 1534 lines), runs two paths chosen at runtime by `Tables::images`:
- **Bounce path (`images == false`, "packed"):** it reads a row's page-aligned superset into a 16-slot bounce (`bounce_`, `kBounceSlots = kBanks * kBounceRows`), then *packs* it, meaning it splits it per segment into the pinned slabs. It does this inline (`pack_one`), or on the `PackPool` workers (`dispatch_ready_rows`), or piece by piece with piece streaming (`dispatch_ready_pieces`/`collect_pieces`).
- **Direct path (`images == true`, row images, `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1`, "non packed"):** each read is one `readv` whose iovecs are the destination slab rows (`image_iovecs`). No bounce and no pool. "Packing" is publishing (`publish_landed`). **This is the production recipe's path** (the 2026-09-28 campaign used row images with piece streaming).

Deliverables, in order:
1. A behavior-preserving CRTP refactor, proven by the existing suites plus a new golden characterization test (byte-identical slab bytes, identical SQE sets, identical deterministic trace fields).
2. Working fixed buffers (`READ_MODE=fixed|readv_fixed`) at real tier sizes, registered in row-aligned ≤ 1 GiB chunks over the unchanged CUDA-pinned slabs, plus fixed files (`FIXED_FILES=1`) verified end to end with mirror roots. Both stay behind the existing knobs, and the defaults do not change.
3. A refactor-only decode pair (base commit vs the split, default knobs: byte-identical output, unchanged ms/token), and a decode A/B on divix01 for each registration mode against the default at the same commit, reporting ms/token and byte identity. **Expectation, stated up front:** the 2026-09-28 campaign found every working io_uring option within ±0.7 ms/token of the defaults (~104.2 ms/token) and IOPOLL worse (114.4). Registration may well not move ms/token. The arm measures this; nothing in this plan assumes a win.

## Global Constraints

- Code reaches divix01 only by commit → `git push origin cc/reader-crtp-uring` → fetch into the private worktree `/data/models/slang/nvfp4-work/wt-reader-crtp`. No rsync, scp or `git archive`. Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- Every run on divix01 uses `PYTHONPATH=$PWD/python` and first prints `sglang.__file__`, which must be under `wt-reader-crtp/python`. Any piped pytest reads `${PIPESTATUS[0]}`; a result that came through a pipe without it is unverified.
- The registered-suite target is `test/registered/unit/kernels` with `-p no:randomly`, and its counts are compared to the same command at the merge-base (`b3f887aaa9`). Record the exact command next to every count quoted.
- CPU work runs under `taskset -c 0-63` with `OMP_NUM_THREADS=8`. GPU work (and any suite that touches CUDA) runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. Cores 64-71 stay free.
- Lock order is `rowimg-disk.lock` first, then `cc-gpu.lock`. `run_arm.sh` takes `cc-gpu.lock` itself (non-blocking), so an arm driver holds `rowimg-disk.lock` and polls for the GPU. It never holds the GPU lock while waiting for the disk.
- Mutants go only in a private worktree (`git worktree add --detach /data/models/slang/nvfp4-work/wt-<name> <commit>`). Revert with `git checkout --`, re-run green, and record both results. Never commit a mutant.
- Do not add or rename environment variables. The chunk cap for tests is a test-only fault word (Task 7), not a knob. If a knob ever becomes necessary, read `.claude/skills/env-var-conventions/SKILL.md` first. The Codex knobs are read with `getenv` in C++ and `os.environ` in Python, outside `environ.py`; migrating them is a follow-up, not this plan (recorded decision 5).
- `UringFileReader` (`io/uring_file_reader.cpp`) keeps its behavior when its registration logic moves to the shared table: 1 GiB chunks from base, silent 0 on failure. `test_uring_file_reader.py` and `test_exl3_row_reader.py` guard this.
- Slab allocation is not changed: one contiguous `allocate_host_slab` tensor per name per layer, NUMA placement and `cudaHostRegister` as today. Registration chunks are views over it.
- Default behavior must not change. With every `SGLANG_EXPERT_STREAM_URING_*` unset, the reader prepares the same opcodes (bounce path: `IORING_OP_READ`, direct path: `IORING_OP_READV`), the same SQE set, and writes the same bytes as at `b3f887aaa9`.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No amend, no rebase, no force push. Never push to or merge into `master`. Do not push to the retired `shared` remote.
- NUMA for arms: node 0 is short of memory because of an uncapped ZFS ARC. Arms run with `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:40960` and `SGLANG_MOE_PINNED_HOST_MB=98304`, after a node-0 pre-launch check that `MemFree + Active(file) + Inactive(file)` ≥ 76800 MiB (share 57344 + headroom 4096 + footprint 15360). Every arm of the A/B uses the same tier.
- A foreign-process gate matches executable names (`pgrep -x`), never argv substrings. For pytest, match a python exe whose argv contains exactly the tokens `-m` `pytest`.
- The divix01 kernel is `6.12.0-211.60.1.el10_2.x86_64` (checked 2026-09-28, not 7.0), with liburing 2.12, `ulimit -l` unlimited, `io_uring_disabled=0` and THP `always`. The laptop runs 7.0. Probe capabilities at runtime; never infer them from the version.

Run template, used by every task (defined once here; each step names its files):

```bash
# SYNC: laptop -> divix01 private worktree
git push origin cc/reader-crtp-uring
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && git fetch origin \
  && git checkout --detach origin/cc/reader-crtp-uring && git log -1 --oneline && git status --short'

# CPU-TEST <files...>: CPU-only test files
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && export PYTHONPATH=$PWD/python \
  OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <files...> -q -rs -p no:randomly 2>&1 | tail -25; \
  echo "EXIT=${PIPESTATUS[0]}"'

# SUITE: the registered kernels suite (touches CUDA in one tier test, so it takes the GPU lock)
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && export PYTHONPATH=$PWD/python \
  OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -5; \
  echo "EXIT=${PIPESTATUS[0]}"'
```

## Review Focus

1. **A row on a chunk boundary, and the non-divisible last chunk.** The last row of one registered chunk and the first row of the next must each be read by one fixed SQE into the right buffer, and the remainder chunk must hold whole rows. Pinned in Task 6 Step 1 (every row in exactly one chunk; 306/306/156 rows of 3,501,056 B) and in Task 8 Step 1 (rows 305 and 306 of a real 1.5 GiB slab).
2. **An image read spanning several named slabs (fan-out).** Its legs may complete in any order. One short leg resubmits alone. One failing leg fails the read once, only after every leg is reaped. A held leg keeps the read unretired and its dependent pieces unpublished. Pinned in Task 7 Step 1 by `test_legs_completing_out_of_order`, `test_one_short_leg_resubmits_only_that_leg`, `test_one_failing_leg_fails_the_read_once_after_every_leg_is_reaped`, `test_a_held_leg_keeps_its_read_unretired_and_its_pieces_unpublished` and `test_ring_reset_mid_fan_out`, plus the retire-on-first-completion mutant.
3. **Registration refused.** A row larger than the cap, more than 16384 chunks, an unsupported sparse table, or a kernel refusal (ENOMEM from `RLIMIT_MEMLOCK`) must refuse `open()` with regions, chunks, cap, memlock and the reason. There must be no silent normal-mode read. Pinned in Task 7 Step 1 (`test_registration_refusal_is_a_clear_error_not_a_fallback`).
4. **Ring reset after a submit failure in a fixed mode.** `UringReader::drain` resets the ring when prepared SQEs were never consumed. It must re-create the sparse table, re-add every chunk and re-register the files. Pinned in Task 7 Step 1 (`test_ring_reset_mid_fan_out`).
5. **Fixed files with mirror roots, and default opcodes after the split.** The fixed file table is `fds_` in `Tables::paths` order across 3 roots, including a zero-weight root; this is pinned in Task 7 Step 1 (`test_fixed_files_with_three_mirror_roots`). With default knobs the bounce path still prepares `IORING_OP_READ` via `prep_read`, and an image reader still traces `pack_split` as the old `set_pack` did; this is pinned in Task 1's golden and Task 4's structural test.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_core.h` | Create (Task 4) | `ReaderCore<Derived, Layout, Reader>`: the pipeline shared by both paths, plus `SqeRecord` at namespace scope |
| `.../host/pack_reader.h` | Create (Task 4) | `PackReader<Layout, Reader>`: bounce, `PackPool`, inline/worker/piece packing |
| `.../host/row_reader.h` | Rewrite (Tasks 3-4) | `RowReader<Layout, Reader>`: the direct row-image path only (`image_iovecs`, `publish_landed`, O_DIRECT alignment) |
| `.../host/any_reader.h` | Create (Task 4) | `AnyReader<Layout, Reader>`: a `std::variant` of the two, chosen by `Tables::images`; the tier's `Source` |
| `python/sglang/kernels/jit/csrc/io/registered_buffers.h` | Create (Task 6) | `plan_chunks` (row-aligned ≤ 1 GiB cuts) and `RegisteredBufferTable` (sparse slots, `update_tag`, lookup, release), moved out of `uring_file_reader.cpp` |
| `python/sglang/kernels/jit/csrc/io/uring_file_reader.cpp` | Modify (Task 6) | Uses the shared table; behavior unchanged |
| `.../host/uring_reader.h` | Modify (Tasks 2, 7) | Codex option layer (Task 2); registration through the table, `fixed_legs`/`prep_readv_fixed`/`sq_space`, O(1) fixed-file map, diagnostics (Task 7) |
| `.../host/faulty_reader.h` | Modify (Tasks 2, 7) | Forward the new optional `UringReader` members |
| `.../host/row_tables.h` | Modify (Tasks 2, 7) | `RegisteredRegion {base, bytes, row_bytes}`; `buffer_regions` becomes `(N, 3)` |
| `.../host/ffi_exports.h` | Modify (Tasks 2, 4, 7) | `using Source = AnyReader<Layout, Reader>`; the `fixed_chunk_cap` fault word applied before `open()`; `fixed_cuts` in `read_rows_sqes` info |
| `.../host/read_fault.h` | Modify (Task 7) | `kFaultWords` 28 → 30; word 28 = `fixed_chunk_cap`, word 29 = `leg` |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | Modify (Tasks 2, 7) | `_table_buffer_regions` gives one row per slab with `row_bytes`; `_fault_tensor(fixed_chunk_cap=0)`; `fixed_cuts` |
| `test/registered/unit/kernels/test_expert_stream_reader_golden.py` | Create (Task 1) | Golden characterization of both paths |
| `test/registered/unit/kernels/test_expert_stream_reader_split.py` | Create (Task 4) | Structural checks of the split |
| `test/registered/unit/kernels/test_registered_buffer_table.py` | Create (Task 6) | Native tests: every row in one chunk, the non-divisible last chunk, the table on a real ring |
| `test/registered/unit/kernels/test_expert_stream_fixed_buffers.py` | Create (Task 7) | Fixed buffers and files through the FFI with real io_uring and a lowered cap |
| `test/manual/dsv41/test_expert_stream_fixed_buffers_big.py` | Create (Task 8) | divix01 only: a real 1.5 GiB slab registered as row-aligned chunks |
| `analysis/dsv41-drive/reader-crtp/*` | Create (Task 5) | Refactor-only decode pair (base vs split) |
| `analysis/dsv41-drive/uring-reg/*` | Create (Task 9) | Registration decode arms and their write-up |

---

### Task 0: Workspace and baselines

**Files:** none changed.

**Interfaces:** Produces the branch `cc/reader-crtp-uring` at `b3f887aaa9` on origin; the divix01 worktree `wt-reader-crtp`; and the baseline suite counts that every later suite result is compared to.

- [ ] **Step 1: Create the branch and laptop worktree** (superpowers:using-git-worktrees)

```bash
git fetch origin mirror3-piece-stream arm/sleepfree-cand
git worktree add -b cc/reader-crtp-uring ../sglang-nvfp4-reader-crtp origin/mirror3-piece-stream
cd ../sglang-nvfp4-reader-crtp && git log -1 --oneline   # expect b3f887aaa9
git push -u origin cc/reader-crtp-uring
```

- [ ] **Step 2: Create the divix01 worktree**

```bash
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-reader-crtp origin/cc/reader-crtp-uring \
  && git -C /data/models/slang/nvfp4-work/wt-reader-crtp log -1 --oneline'
```

- [ ] **Step 3: Record the baseline suite at `b3f887aaa9`.** Run SUITE and write the passed, failed, skipped and error counts, with the command, into the task log (for example `N passed, M skipped`). Every later SUITE result is compared to these counts. The expected delta is exactly the tests this plan adds.

- [ ] **Step 4: Record the baseline of the reader suites** (the files the refactor must keep green):

CPU-TEST `test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_pack_workers.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py test/registered/unit/kernels/test_exl3_ram_miss_row_images.py test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py`

Expected: EXIT=0. Record the counts. (`test_exl3_ram_miss_tier.py` runs inside SUITE, which holds the GPU lock.)

---

### Task 1: Golden characterization of both reader paths

The existing suites are thick (~166 reader tests), but they assert properties, not a fingerprint of everything the reader emits. This task pins, per shape and mode: the result, a digest of every slab byte, the sorted SQE log, the descriptor count and credit, and the deterministic stage-record fields. It is generated **at the base commit**, so it describes the reader before any change.

**Files:**
- Create: `test/registered/unit/kernels/test_expert_stream_reader_golden.py`

**Interfaces:**
- Consumes: `read_rows_sqes(tables, row, experts, slots, *, direct, step=BOUNCE_ROWS, max_sqes=4096, layout="exl3", **faults) -> (result, log, info, record)` and `ram_miss_setup(tmp_path, *, capacity, layers, experts, mirror_weights, hidden, inter, row_images)` from `sglang.test.dsv41_ram_miss_fixtures`, both as on `b3f887aaa9`.
- Produces: `GOLDEN`, a dict keyed `"<shape>/<mode>"`. Tasks 2, 3, 4 and 6 re-run this file unchanged; it must stay green without edits.

- [ ] **Step 1: Write the test with an empty `GOLDEN`**

```python
"""Golden characterization of the expert-stream reader (plan 2026-09-28-reader-crtp-uring-registration Task 1).

Generated at b3f887aaa9 (origin/mirror3-piece-stream) BEFORE the CRTP split. Every later task must keep it green
unedited: it is the proof that the split and the io_uring option layer leave the default reader byte-for-byte what it
was. Per shape and mode it pins the read's result, a digest of every slab byte, the SQE log as a sorted set (the
refill order depends on completion timing, so only the set is deterministic), the descriptor count, the ring credit
and the stage record's deterministic fields. Regenerate only with the user's approval:
    PYTHONPATH=python python test/registered/unit/kernels/test_expert_stream_reader_golden.py [tmp_dir]
"""

import hashlib
import json
import pathlib
import sys
import tempfile

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# (id, mirror_weights, row_images)
SHAPES = [
    ("one_root", None, False),
    ("halves", (1.0, 1.0), False),
    ("three_roots", (1.0, 1.0, 1.0), False),
    ("three_roots_zero_mid", (1.0, 0.0, 1.0), False),
    ("images_one_root", None, True),
    ("images_halves", (1.0, 1.0), True),
    ("images_three_roots", (1.0, 1.0, 1.0), True),
]
# (id, faults): the bounce path's three packers and the direct path with and without piece streaming.
BOUNCE_MODES = [
    ("inline", {}),
    ("workers", {"pack_workers": 2, "pack_split": 3}),
    ("pieces", {"pack_workers": 2, "piece_stream": True}),
]
IMAGE_MODES = [
    ("direct", {}),
    ("direct_split_traced", {"pack_split": 3}),  # images ignore workers but trace pack_split (old set_pack)
    ("direct_pieces", {"piece_stream": True}),
]
# Stage-record fields that do not depend on time or on completion order.
STAGE_KEYS = [
    "rows", "batches", "bytes", "extents", "rows_asked", "useful_bytes", "submitted_bytes", "retried_bytes",
    "cancelled_bytes", "pack_workers", "pack_split", "piece_stream", "pieces_vetted", "pieces_published",
    "piece_publish_refused",
]
EXPERTS = [10, 3, 7, 0, 11, 5, 1, 8, 2, 9, 4]
SLOTS = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]

GOLDEN = {}


def _slab_digest(obj, h):
    if isinstance(obj, torch.Tensor):
        h.update(obj.contiguous().view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            h.update(str(key).encode())
            _slab_digest(obj[key], h)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _slab_digest(item, h)


def _measure(root, weights, images, faults):
    dims = {} if images else dict(hidden=256, inter=512)
    s = ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)
    for slabs in (s.slabs.values() if isinstance(s.slabs, dict) else s.slabs):
        for t in (slabs.values() if isinstance(slabs, dict) else [slabs]):
            t.view(torch.uint8).fill_(0x5A)  # a fixed prior content, so unread bytes are pinned too
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, direct=False, **faults)
    h = hashlib.sha256()
    _slab_digest(s.slabs, h)
    return {
        "result": result,
        "slabs": h.hexdigest(),
        "sqes": hashlib.sha256(json.dumps(sorted(log)).encode()).hexdigest(),
        "sqe_count": info["sqes"],
        "descriptors": info["descriptors"],
        "credit": info["credit"],
        "stage": {key: int(record[key]) for key in STAGE_KEYS},
    }


def _cases():
    for shape, weights, images in SHAPES:
        for mode, faults in IMAGE_MODES if images else BOUNCE_MODES:
            yield f"{shape}/{mode}", weights, images, faults


@pytest.mark.parametrize("key, weights, images, faults", list(_cases()), ids=[c[0] for c in _cases()])
def test_reader_matches_the_base_commit(tmp_path, key, weights, images, faults):
    root = tmp_path / "ckpt"
    root.mkdir()
    assert _measure(root, weights, images, faults) == GOLDEN[key]


if __name__ == "__main__":
    golden = {}
    for key, weights, images, faults in _cases():
        with tempfile.TemporaryDirectory(dir=sys.argv[1] if len(sys.argv) > 1 else None) as d:
            root = pathlib.Path(d) / "ckpt"
            root.mkdir()
            golden[key] = _measure(root, weights, images, faults)
    print("GOLDEN = " + json.dumps(golden, indent=4, sort_keys=True))
```

- [ ] **Step 2: Check the fixture's field names before generating.** `RamMissSetup.slabs` is a `dict`. `_measure` handles `{layer: {name: tensor}}` and `{name: tensor}`. If `record` from `read_rows_sqes` is not indexable by the `STAGE_KEYS` names, read `STAGE_FIELDS` in `python/sglang/kernels/ops/moe/expert_stream_transport.py` and use its exact spellings. If `ram_miss_setup` refuses the 3-root image shape, drop only that shape and say so in the commit message. Commit, SYNC, then on divix01:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 && for i in 1 2 3; do taskset -c 0-63 /data/models/slang/.venv/bin/python \
  test/registered/unit/kernels/test_expert_stream_reader_golden.py /mnt/nvme1/nvfp4-work/golden-tmp > /tmp/golden-$i.txt; \
  echo rc=$?; done; md5sum /tmp/golden-*.txt'
```

Expected: rc=0 three times, and three identical md5s. If any key differs between runs, the differing field is nondeterministic. Remove it from that mode's pinned dict and record why in the module docstring. Do not widen tolerance any other way.

- [ ] **Step 3: Paste the printed `GOLDEN` into the file, commit, SYNC, and run it**

CPU-TEST `test/registered/unit/kernels/test_expert_stream_reader_golden.py`
Expected: every case PASS, EXIT=0.

- [ ] **Step 4: Mutant check (private worktree).** In `wt-golden-mutant`, in `row_reader.h::image_iovecs` change `s.dst + (lo - s.src)` to `s.dst + (lo - s.src) + 512`, and in `pack_one` swap the `memcpy` source to `base + segment.src + 1`. Run the golden file; expected FAIL on the `images_*` and `*/inline` keys respectively. `git checkout --` both files, re-run, and expect PASS. Record both results and remove the worktree.

- [ ] **Step 5: Commit**

```bash
git add test/registered/unit/kernels/test_expert_stream_reader_golden.py
git commit -m "test(expert-stream): golden characterization of the bounce and direct reader paths

Pins slab bytes, the SQE set, descriptors, credit and deterministic stage fields per mirror shape and
packing mode at b3f887aaa9, before the CRTP split.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/reader-crtp-uring
```

---

### Task 2: Port the Codex io_uring option layer

**Files:** from the cherry-picks: `host/uring_options.h` (new), `host/uring_reader.h`, `host/faulty_reader.h`, `host/row_reader.h` (17 lines), `host/row_tables.h`, `host/ffi_exports.h`, `expert_stream_transport.py`, `srt/layers/moe/expert_host_tier.py`, `srt/layers/moe/expert_stream.py`, the tests `test_expert_stream_uring_{native,options,integration}.py`, `test_expert_stream_buffer_regions.py`, `test/registered/unit/layers/moe/test_expert_host_slab_arena.py`, and `analysis/dsv41-drive/uring-config/HANDOFF.md`.

**Interfaces:**
- Produces (used by Tasks 4 and 6): `UringOptions::from_env()`; `UringReader::configure_resources(const std::vector<int>& fds, const std::vector<iovec>& buffers, bool direct)`; `UringReader::close()`; `Tables::buffer_regions` (`std::vector<iovec>`, from Python's `_table_buffer_regions(tables)`); `tables_from(..., TensorView buffer_regions, ...)`; and in `RowReader`, `configured_queue_depth_`, the `configure_resources` call in `open()` and `io_.close()` in the destructor.

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick -x a5b4c87bb7 9dad1512e9 e0e79ddfd6 ba58e8a5a1 49d3c6b840
```

Expected: all five apply without conflict, as they did in the 2026-09-28 dry run. If one conflicts, stop and report it; do not resolve by hand without re-reading the conflicting hunk against `b3f887aaa9`. `-x` keeps the provenance. These commits carry Codex's authorship, and adding the trailer would require amending, which is forbidden.

- [ ] **Step 2: SYNC and run the golden plus the ported suites (CPU)**

CPU-TEST `test/registered/unit/kernels/test_expert_stream_reader_golden.py test/registered/unit/kernels/test_expert_stream_uring_options.py test/registered/unit/kernels/test_expert_stream_buffer_regions.py test/registered/unit/layers/moe/test_expert_host_slab_arena.py`

Expected: EXIT=0. A passing golden here is the evidence that the option layer is default-neutral on the mirror3 reader.

- [ ] **Step 3: Run the real-kernel matrix on NVMe under the disk lock**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_expert_stream_uring_native.py \
  test/registered/unit/kernels/test_expert_stream_uring_integration.py \
  -q -rs -p no:randomly --basetemp=/mnt/nvme2/nvfp4-work/reader-crtp-native 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: EXIT=0 with 0 capability skips. The campaign on this kernel got 45 passed and 0 skipped.

- [ ] **Step 4: SUITE.** Expected: the Task 0 counts plus the tests the five commits add, with no new failures. Record the command and the counts.

- [ ] **Step 5: Push** (the cherry-picks are the commits): `git push origin cc/reader-crtp-uring`.

---

### Task 3: CRTP prepare. Route every path branch through named hooks, still one class

A pure move (Task 4) is only checkable when the code being moved already has its final shape. This task gives it that shape inside the existing `RowReader`. Every `t_.images` / `pool_` / `bounce_` decision becomes a call to a hook with a `pack_` or `row_` implementation. After this task, Task 4 moves bodies without editing them.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h`

**Interfaces:**
- Produces (private members of `RowReader`, each renamed in Task 4 by dropping the prefix into the derived class). Here `X` is a hook in the table below: `pack_X` goes to `PackReader::X` and `row_X` goes to `RowReader::X`.

| Hook `X` (signature) | `pack_X` body (from) | `row_X` body (from) |
|---|---|---|
| `void check_piece_stream_support() const` | throw `"piece streaming needs packing workers"` when `pack_workers_ == 0` (from `set_piece_stream` and `open`) | empty |
| `void on_piece_stream_set()` | `if (pool_) size_jobs();` | empty |
| `bool open_memory()` | the `posix_memalign` of `bounce_` (from `open`) | `check_image_alignment(); return true;` |
| `void open_workers(cpu_set_t inherited)` | the piece-stream refusal plus the `if (pack_workers_ > 0) {...}` pool block (from `open`) | empty |
| `std::vector<iovec> registered_regions() const` | one region per bounce slot (from the cherry-picked `open`) | `t_.buffer_regions` |
| `size_t max_iovecs() const` | `1` | `t_.segments.size()` |
| `unsigned destination(const ExtentDesc& d, iovec* out) const` | `out[0] = {bounce_slot(d.slot) + d.read->dest + d.done, size_t(d.read->length - d.done)}; return 1;` | `return image_iovecs(size_t(d.slot), d.read->dest + d.done, d.read->dest + d.read->length, out);` |
| `bool advance()` | old `pack_one` minus its first line (`if (t_.images) return publish_landed();`) | `return publish_landed();` |
| `void collect()` | old `collect_packed` (which calls `collect_pieces`) | see Step 3 |
| `void quiesce()` | old `quiesce` | empty (with images `c.packing` is always 0, so the old body returned at once) |
| `void poison_slot(size_t slot, uint8_t fill)` | the `memset(bounce_slot...)` branch | the slab-row `memset` branch |
| `void after_finish(size_t slot)` | `if (fault_.poison) poison_slot(slot, kPoisonFill ^ 0xFF);` | empty |
| `int64_t unfinished_jobs() const` | old body | `return 0;` |
| `unsigned pack_workers() const`, `unsigned pack_split() const` | `pack_workers_`, `pack_split_` | `0`, `pack_split_` (**kept**: the old `set_pack` stored `split` even for images, and `read()` traces it) |

- [ ] **Step 1: Add the dispatchers.** For each hook `X`, add a private dispatcher `X(...)` that calls `t_.images ? row_X(...) : pack_X(...)`, and move the corresponding code into `pack_X`/`row_X` exactly as the table says. Call sites change like this:
  - `read()`: `collect_packed()` → `collect()`; `pack_one()` → `advance()`; `Quiesce::~Quiesce` calls `reader->quiesce()` (now the dispatcher).
  - `set_piece_stream`: the inline workers check → `check_piece_stream_support()`; `if (pool_) size_jobs();` → `on_piece_stream_set()`.
  - `open()`: the images/bounce `if` → `if (!open_memory()) return false;`; the configure block uses `registered_regions()`; the tail after the owner pin → `open_workers(inherited);`.
  - `size_extents()`: `iovecs_.assign(extents * max_iovecs(), iovec{})` for **both** paths (the bounce path now owns one scratch iovec per descriptor).
  - `refill()`: replace the `if (t_.images) {...} else {...}` preparation with:

```cpp
      iovec* iov = &iovecs_[static_cast<size_t>(index) * max_iovecs()];
      const unsigned count = destination(d, iov);
      const int fd = fds_[d.read->file];
      // The bounce path keeps IORING_OP_READ (prep_read), the direct path IORING_OP_READV: the default opcodes.
      const bool prepared_one = t_.images ? io_.prep_readv(fd, iov, count, offset, tag)
                                          : io_.prep_read(fd, iov[0].iov_base, static_cast<unsigned>(iov[0].iov_len), offset, tag);
```

  - `finish_row`: `if (fault_.poison && !t_.images) poison_slot(best, kPoisonFill ^ 0xFF);` → `after_finish(best);`.
  - `admit_batch`'s `if (fault_.poison) poison_slot(slot, kPoisonFill);` stays and calls the dispatcher.

- [ ] **Step 2: Keep `set_pack` semantics.** `set_pack` stays as is: `pack_workers_ = t_.images ? 0 : max(0, workers); pack_split_ = split > 0 ? split : pack_workers_;`. `read()` traces `pack_workers()`/`pack_split()` (the dispatchers) instead of the members.

- [ ] **Step 3: Write `row_collect()`.** On the direct path the old `collect_packed()` did something only with piece streaming. It called `collect_pieces()`, and there no piece is ever held by a job (`publish_landed` publishes on dispatch), so its only effect was finishing rows:

```cpp
  void row_collect() {
    if (!piece_stream_) return;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      BounceRow& r = rows_[s];
      if (r.state == RowState::Ready && r.published == kAllPieces) {
        finish_row(s, r.pack_first == INT64_MAX ? 0 : r.pack_first, r.pack_last);
      }
    }
  }
```

- [ ] **Step 4: Check that no path branch is left outside a hook.** From the worktree root:

```bash
grep -n 't_\.images\|bounce_\b\|pool_\b\|jobs_\[' python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h
```

Expected: matches only inside `pack_*`/`row_*` bodies, the dispatchers, the constructor, `set_pack`, `bounce_slot`, `size_jobs`, `dispatch_ready_rows`, `dispatch_ready_pieces`, `collect_pieces`, and the member declarations. Any other hit is a branch still in shared code. Move it into a hook.

- [ ] **Step 5: Commit, SYNC, then run the golden plus the reader suites**

CPU-TEST `test/registered/unit/kernels/test_expert_stream_reader_golden.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_pack_workers.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py test/registered/unit/kernels/test_exl3_ram_miss_row_images.py test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py test/registered/unit/kernels/test_expert_stream_uring_integration.py`

Expected: EXIT=0, with counts equal to Task 0 Step 4 plus the Task 1 and Task 2 additions.

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h
git commit -m "refactor(expert-stream): route every bounce/direct branch of RowReader through named hooks

Prepares the CRTP split: shared pipeline code now calls check_piece_stream_support, on_piece_stream_set,
open_memory, open_workers, registered_regions, max_iovecs, destination, advance, collect, quiesce,
poison_slot, after_finish, unfinished_jobs, pack_workers and pack_split, each with a pack_ and a row_
body. Default opcodes unchanged (prep_read for the bounce, prep_readv for images); golden green.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/reader-crtp-uring
```

---

### Task 4: CRTP split into `ReaderCore`, `PackReader`, `RowReader` and `AnyReader`

**Files:**
- Create: `host/reader_core.h`, `host/pack_reader.h`, `host/any_reader.h`
- Rewrite: `host/row_reader.h` (direct only)
- Modify: `host/ffi_exports.h:18` (`using Source`), plus every `#include "row_reader.h"` that needs `any_reader.h` (`ram_tier.h`/`ram_thread.h`: include whichever header they include today and let `any_reader.h` pull in the rest)
- Create: `test/registered/unit/kernels/test_expert_stream_reader_split.py`

**Interfaces:**
- Consumes: the hook table of Task 3.
- Produces:

```cpp
// reader_core.h
struct SqeRecord { int64_t file, offset, length, bounce; };  // moved out of the class; members unchanged

template <class Derived, ExpertRowLayout Layout, AsyncFileReader Reader>
class ReaderCore {
 public:
  using LayoutType = Layout;
  using SqeRecord = expert_stream::SqeRecord;
  ReaderCore(const ReaderCore&) = delete;
  ReaderCore& operator=(const ReaderCore&) = delete;
  const Tables& tables() const;
  void set_piece_stream(bool on);
  bool piece_stream() const;
  int64_t publish_refused() const;
  size_t descriptors() const;
  unsigned credit() const;
  void set_sqe_log(std::vector<SqeRecord>* log);
  void set_owner_core(int64_t core);
  void set_fault(const ReadFault& fault);
  int64_t cqes() const;
  int64_t stale_cqes() const;
  int64_t generation_wraps() const;
  bool open();
  int read(int64_t layer, const std::vector<int32_t>& experts, const std::vector<int64_t>& slots, size_t step,
           const std::function<bool(size_t)>& abandon, StageRecord* trace = nullptr,
           std::vector<uint8_t>* packed = nullptr, size_t max_reading_rows = SIZE_MAX,
           const std::function<void()>& progress = nullptr, const PiecePublish* publish = nullptr);
 protected:
  ReaderCore(Tables tables, bool direct);
  ~ReaderCore();            // non-virtual: never deleted through the base
  void close_io();          // io_.close() when Reader has it; idempotent
  Derived& derived();
  const Derived& derived() const;
  // ... every shared member function and member named in Step 2
};

// pack_reader.h
template <ExpertRowLayout Layout, AsyncFileReader Reader>
class PackReader : public ReaderCore<PackReader<Layout, Reader>, Layout, Reader> {
 public:
  PackReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0);  // throws if tables.images
  ~PackReader();            // pool_.reset(); this->close_io(); std::free(bounce_);
  void set_pack(int64_t workers, int64_t split);
  unsigned pack_workers() const;
  unsigned pack_split() const;
  std::vector<int> packing_cpus() const;
  int64_t unfinished_jobs() const;
  // hooks (friend ReaderCore): every X of Task 3's table, with the pack_X body
};

// row_reader.h
template <ExpertRowLayout Layout, AsyncFileReader Reader>
class RowReader : public ReaderCore<RowReader<Layout, Reader>, Layout, Reader> {
 public:
  RowReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0);  // throws if !tables.images
  void set_pack(int64_t workers, int64_t split);  // pack_split_ = split > 0 ? split : 0 (old semantics for images)
  unsigned pack_workers() const;                  // 0
  unsigned pack_split() const;                    // pack_split_
  std::vector<int> packing_cpus() const;          // {}
  int64_t unfinished_jobs() const;                // 0
  // hooks: every X of Task 3's table, with the row_X body; plus image_iovecs, check_image_alignment, publish_landed
};

// any_reader.h
template <ExpertRowLayout Layout, AsyncFileReader Reader>
class AnyReader {
 public:
  using LayoutType = Layout;
  using SqeRecord = expert_stream::SqeRecord;
  AnyReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0);
  // forwards: tables, set_pack, pack_workers, pack_split, packing_cpus, set_piece_stream, piece_stream,
  // publish_refused, descriptors, credit, set_sqe_log, set_owner_core, unfinished_jobs, set_fault, cqes,
  // stale_cqes, generation_wraps, open, read
  bool is_pack() const;
};
```

- [ ] **Step 1: Write the structural test (it fails now: the files do not exist)**

```python
"""The CRTP split's shape (plan 2026-09-28-reader-crtp-uring-registration Task 4): the shared pipeline names no
bounce, pool or image mechanism; each derived reader holds only its own; the tier reads through AnyReader."""

import pathlib

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOST = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/kernels/jit/csrc/moe/expert_stream/host"


def _code(name):
    """The header without // comments, so prose naming a mechanism does not count."""
    return "\n".join(line.split("//", 1)[0] for line in (HOST / name).read_text().splitlines())


def test_the_core_names_no_path_specific_mechanism():
    core = _code("reader_core.h")
    for word in ("t_.images", "bounce_", "pool_", "jobs_", "PackPool", "image_iovecs", "publish_landed",
                 "posix_memalign", "check_image_alignment"):
        assert word not in core, word
    assert "template <class Derived, ExpertRowLayout Layout, AsyncFileReader Reader>" in core
    assert "class ReaderCore" in core


def test_each_derived_reader_holds_only_its_own_mechanism():
    pack, row = _code("pack_reader.h"), _code("row_reader.h")
    assert "class PackReader : public ReaderCore<PackReader<Layout, Reader>, Layout, Reader>" in pack
    assert "class RowReader : public ReaderCore<RowReader<Layout, Reader>, Layout, Reader>" in row
    for word in ("image_iovecs", "publish_landed", "check_image_alignment"):
        assert word not in pack, word
    for word in ("bounce_", "pool_", "PackPool", "PackJob", "dispatch_ready"):
        assert word not in row, word


def test_the_bounce_path_keeps_the_scalar_read_opcode():
    core = _code("reader_core.h")
    assert "io_.prep_read(" in core and "io_.prep_readv(" in core


def test_the_tier_and_ffi_read_through_any_reader():
    assert "using Source = AnyReader<Layout, Reader>;" in _code("ffi_exports.h")
    any_reader = _code("any_reader.h")
    assert "std::variant<std::monostate, PackReader<Layout, Reader>, RowReader<Layout, Reader>>" in any_reader
```

- [ ] **Step 2: Move the code.** Use the `mechanical-refactor-verify` skill for this step. The move is: every method body below goes from `row_reader.h` (Task 3 state) to its new home **byte-identical**, except for the listed edits.
  - **To `ReaderCore`:** the constants `kMaxSoftErrors kMaxRetries kPoisonFill kPoisonSlot kProgressIntervalNs`; the types `RowState ExtentDesc BounceRow Completion Call`; the public methods in the interface block; and `queue_depth next_generation size_extents reset_pipeline has_ready queue_push admit admit_batch plan_pieces queue_sub_reads land_sub_read vet_pieces piece_delivered refill reap process fault_matches_sub retire take_ready_row publish_collected holding_for_probe finish_row account_unfinished submit drain`.
    - Also the members `t_ direct_ fds_ devs_ file_drive_ drive_dev_ fault_ cqes_ part_fired_ c_ descs_ queue_ completions_ held_ again_ rows_ owner_core_ owner_pinned_ unpinned_affinity_ io_ configured_queue_depth_ piece_stream_ subs_ sub_reads_ piece_runs_ iovecs_ geometry_ sqe_log_ publishes_ publish_refused_ rows_busy_ bank_live_ generation_ generation_wraps_ stale_cqes_ retired_ stale_ stale_index_ stale_waiting_ stale_armed_`.
    - The only edits: each dispatcher call `X(...)` becomes `derived().X(...)`; `read()`'s trace lines use `derived().pack_workers()`/`derived().pack_split()`; `Quiesce` calls `reader->derived().quiesce()`; and in `refill` the `t_.images ?` test becomes `Derived::kScatter ?`, with `static constexpr bool kScatter` false in `PackReader` and true in `RowReader`.
    - The destructor: `close_io(); for (int fd : fds_) ::close(fd); if (owner_pinned_) pthread_setaffinity_np(...)`. It keeps the old affinity-restore comment.
  - **To `PackReader`:** every `pack_X` body renamed `X`, plus `bounce_slot size_jobs dispatch_ready_rows dispatch_ready_pieces collect_pieces set_pack pack_workers pack_split packing_cpus unfinished_jobs`, and the members `bounce_ pack_workers_ pack_split_ pool_ jobs_ runs_`. The destructor order is `pool_.reset()` (joins workers), then `this->close_io()` (unregisters the bounce), then `std::free(bounce_)`. Keep the old member-order comment about `io_`, reworded to say `io_` now lives in the base and is closed explicitly before the bounce is freed. Base members use `this->` or a `using Base::...;` block. Pick one style and use it throughout.
  - **To `RowReader`:** every `row_X` body renamed `X`, plus `image_iovecs check_image_alignment publish_landed set_pack pack_workers pack_split packing_cpus unfinished_jobs`, and the member `pack_split_`.
  - Both derived classes declare `friend class ReaderCore<...>;` so the hooks can stay private.
  - The dispatchers from Task 3 are deleted.

- [ ] **Step 3: Write `AnyReader`**

```cpp
// The tier's reader: a PackReader (bounce and pack) or a RowReader (direct row images), chosen once by
// Tables::images. RamTier and HostExports hold one Source type; the variant keeps them unchanged.
#pragma once

#include <variant>

#include "pack_reader.h"
#include "row_reader.h"

namespace sglang::expert_stream {

template <ExpertRowLayout Layout, AsyncFileReader Reader>
class AnyReader {
 public:
  using LayoutType = Layout;
  using SqeRecord = expert_stream::SqeRecord;

  AnyReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0) {
    if (tables.images) {
      impl_.template emplace<RowReader<Layout, Reader>>(std::move(tables), direct, pack_workers, pack_split);
    } else {
      impl_.template emplace<PackReader<Layout, Reader>>(std::move(tables), direct, pack_workers, pack_split);
    }
  }
  AnyReader(const AnyReader&) = delete;
  AnyReader& operator=(const AnyReader&) = delete;

  bool is_pack() const { return std::holds_alternative<PackReader<Layout, Reader>>(impl_); }
  const Tables& tables() const { return visit([](auto& r) -> const Tables& { return r.tables(); }); }
  void set_pack(int64_t w, int64_t s) { visit([&](auto& r) { r.set_pack(w, s); }); }
  unsigned pack_workers() const { return visit([](auto& r) { return r.pack_workers(); }); }
  unsigned pack_split() const { return visit([](auto& r) { return r.pack_split(); }); }
  std::vector<int> packing_cpus() const { return visit([](auto& r) { return r.packing_cpus(); }); }
  void set_piece_stream(bool on) { visit([&](auto& r) { r.set_piece_stream(on); }); }
  bool piece_stream() const { return visit([](auto& r) { return r.piece_stream(); }); }
  int64_t publish_refused() const { return visit([](auto& r) { return r.publish_refused(); }); }
  size_t descriptors() const { return visit([](auto& r) { return r.descriptors(); }); }
  unsigned credit() const { return visit([](auto& r) { return r.credit(); }); }
  void set_sqe_log(std::vector<SqeRecord>* log) { visit([&](auto& r) { r.set_sqe_log(log); }); }
  void set_owner_core(int64_t core) { visit([&](auto& r) { r.set_owner_core(core); }); }
  int64_t unfinished_jobs() const { return visit([](auto& r) { return r.unfinished_jobs(); }); }
  void set_fault(const ReadFault& f) { visit([&](auto& r) { r.set_fault(f); }); }
  int64_t cqes() const { return visit([](auto& r) { return r.cqes(); }); }
  int64_t stale_cqes() const { return visit([](auto& r) { return r.stale_cqes(); }); }
  int64_t generation_wraps() const { return visit([](auto& r) { return r.generation_wraps(); }); }
  bool open() { return visit([](auto& r) { return r.open(); }); }
  template <class... Args>
  int read(Args&&... args) { return visit([&](auto& r) { return r.read(std::forward<Args>(args)...); }); }

 private:
  template <class F>
  decltype(auto) visit(F&& f) {
    if (auto* p = std::get_if<PackReader<Layout, Reader>>(&impl_)) return f(*p);
    return f(std::get<RowReader<Layout, Reader>>(impl_));
  }
  template <class F>
  decltype(auto) visit(F&& f) const {
    if (auto* p = std::get_if<PackReader<Layout, Reader>>(&impl_)) return f(*p);
    return f(std::get<RowReader<Layout, Reader>>(impl_));
  }
  std::variant<std::monostate, PackReader<Layout, Reader>, RowReader<Layout, Reader>> impl_;
};

}  // namespace sglang::expert_stream
```

If the compiler finds a `reader.`/`reader_.` use in `ffi_exports.h`, `ram_tier.h` or `ram_thread.h` that is missing from this list, add a forwarder of the same shape. Do not change the caller.

- [ ] **Step 4: Switch the tier.** In `ffi_exports.h`, replace `using Source = RowReader<Layout, Reader>;` with `using Source = AnyReader<Layout, Reader>;` and include `any_reader.h`. `RamTier<Source>` is unchanged.

- [ ] **Step 5: Verify the move is pure.** With the `mechanical-refactor-verify` skill's body-diff approach: extract each moved method's body from `git show HEAD~1:.../row_reader.h` (the Task 3 commit) and from its new file, then diff them. Allowed differences are only the edits listed in Step 2, plus the removed `pack_`/`row_` prefixes. Put the script in the scratchpad, not the repo, and paste its "N bodies identical, M with listed edits only" summary into the commit message.

- [ ] **Step 6: Commit, SYNC, run the structural test, the golden and the reader suites, then SUITE**

CPU-TEST `test/registered/unit/kernels/test_expert_stream_reader_split.py test/registered/unit/kernels/test_expert_stream_reader_golden.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_pack_workers.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream_parts.py test/registered/unit/kernels/test_exl3_ram_miss_row_images.py test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py test/registered/unit/kernels/test_expert_stream_uring_integration.py`

Expected: EXIT=0. Then SUITE: counts equal Task 2's plus the 4 structural tests, with no new failures or errors. Record the command and the counts.

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/{reader_core.h,pack_reader.h,row_reader.h,any_reader.h,ffi_exports.h} \
        test/registered/unit/kernels/test_expert_stream_reader_split.py
git commit -m "refactor(expert-stream): split RowReader into ReaderCore (CRTP) + PackReader + RowReader

ReaderCore<Derived, Layout, Reader> holds the pipeline; PackReader is the bounce-and-pack path,
RowReader the direct row-image path; AnyReader (a variant chosen by Tables::images) is the tier's Source.
Method bodies moved byte-identical except the listed derived() edits (<body-diff summary>). Golden,
reader suites and the registered suite green.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/reader-crtp-uring
```

---

### Task 5: Refactor-only decode pair on divix01 (base vs split, default knobs)

The split changes the production reader (`RowReader`, row images, piece streaming), so it gets one decode pair before any registration work lands on top.

**Files:**
- Create: `analysis/dsv41-drive/reader-crtp/drive_reader_crtp_pair.sh`, `analysis/dsv41-drive/reader-crtp/results.md`
- Modify: `benchmarks/dsv41_baseline/generations.json`

**Interfaces:** Consumes `run_arm.sh <arm> <port> KEY=VAL...`, `generations.register(tree_sha, label)`, and `analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh` as the template.

- [ ] **Step 1: Register both trees.** On the laptop, in `cc/reader-crtp-uring`:

```bash
BASE_TREE=$(git rev-parse b3f887aaa9:python); SPLIT_TREE=$(git rev-parse HEAD:python)
PYTHONPATH=benchmarks/dsv41_baseline python -c "import generations as g; \
  g.register('$BASE_TREE', 'mirror3-base-b3f887aaa9'); g.register('$SPLIT_TREE', 'reader-crtp-split')"
git add benchmarks/dsv41_baseline/generations.json
git commit -m "bench(dsv41): register the mirror3 base and reader-crtp split trees for the refactor pair

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/reader-crtp-uring
```

`run_arm.sh` checks the tree of the worktree it runs in, so the base arm needs its own worktree at a commit whose `python/` tree is `b3f887aaa9`'s, and whose `generations.json` registers it. Create a branch `cc/reader-crtp-base-arm` from `b3f887aaa9` and cherry-pick only the `generations.json` commit onto it (`python/` stays identical). Push it, then add worktree `/data/models/slang/nvfp4-work/wt-reader-crtp-base` at `origin/cc/reader-crtp-base-arm`. Confirm there that `git rev-parse HEAD:python` equals `BASE_TREE`.

- [ ] **Step 2: Write the pair driver.** Copy `drive_mirror3_arms.sh` and change only the following:
  - (a) It takes two worktrees and two SHAs: `base` and `split`.
  - (b) Drop `ROOTS3`, the three-device check and the diskstats sampler; keep the SM-clock sampler.
  - (c) Keep unchanged: the driver-version check, the checks on ports 7867 and `$PORT`, the exe-name foreign-process gate with exact `-m` `pytest` argv tokens, `cc-gpu.lock` polling while holding `rowimg-disk.lock`, and PID tracking and cleanup.
  - (d) Keep `node0_gate` with the same numbers (`0:57344,1:40960`, need 76800 MiB). Run it before each arm. The one permitted cut happens before the first arm only and applies to both arms.
  - (e) Run `crtp-base` from the base worktree, then `crtp-split` from the split worktree. Both use `$(tier_overrides)` and all nine `SGLANG_EXPERT_STREAM_URING_*` at their S0 defaults. The base tree does not read them, and they are harmless there.
  - (f) Stop if either arm has rc≠0 or read errors > 0.
  - Commit and push.

- [ ] **Step 3: Launch under the disk lock**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && mkdir -p /mnt/nvme1/reader-crtp \
  && OMP_NUM_THREADS=8 nohup flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 \
     bash analysis/dsv41-drive/reader-crtp/drive_reader_crtp_pair.sh \
       /data/models/slang/nvfp4-work/wt-reader-crtp-base $(git -C /data/models/slang/nvfp4-work/wt-reader-crtp-base rev-parse HEAD) \
       $PWD $(git rev-parse HEAD) /mnt/nvme1/reader-crtp 30031 \
     > /mnt/nvme1/reader-crtp/driver.log 2>&1 < /dev/null &'
```

- [ ] **Step 4: Report** in `analysis/dsv41-drive/reader-crtp/results.md`:
  - per session and pooled: ms/token and TTFT;
  - output byte-identical (yes/no);
  - `rows_read/served/read_errors`;
  - the SM clock at the start of each session;
  - node-0 gate values.
  - Pass: byte-identical output, and pooled ms/token within ±1.5 ms/token, the 2026-09-28 campaign's noise threshold. Output that differs is a blocker: stop and investigate before Task 6.
  - Commit with the trailer and push.

---

### Task 6: Extract the chunked-registration logic from `uring_file_reader.cpp` into a shared, row-aligned table

`python/sglang/kernels/jit/csrc/io/uring_file_reader.cpp` already registers memory in chunks of at most 1 GiB. It uses a sparse table (`io_uring_register_buffers_sparse`, :96), fills slots with `io_uring_register_buffers_update_tag` (`register_buffer`, :141-176), looks up with `find_buffer_` (:624), and releases with `unregister_range_`/`rollback_` (:639-663). This task moves that logic into a header both readers use, and adds **row-aligned** cutting. Each chunk is the largest multiple of the region's `row_bytes` that fits in 1 GiB, and the last chunk takes the remainder. Then every slab row lies in exactly one registered buffer, and no single iovec (which is always inside one slab row) can straddle two.

**Files:**
- Create: `python/sglang/kernels/jit/csrc/io/registered_buffers.h`
- Modify: `python/sglang/kernels/jit/csrc/io/uring_file_reader.cpp` (use the table; behavior unchanged)
- Create: `test/registered/unit/kernels/test_registered_buffer_table.py`
- Regression: `test/registered/unit/kernels/test_uring_file_reader.py`, `test/registered/unit/layers/moe/test_exl3_row_reader.py`

**Interfaces:**
- Produces (used by Task 7):

```cpp
namespace sglang::io {
constexpr uint64_t kMaxRegisteredBufferBytes = 1ULL << 30;  // io_buffer_validate refuses more (-EFAULT)
constexpr unsigned kMaxRegisteredBufferSlots = 1U << 14;    // IORING_MAX_REG_BUFFERS

struct ChunkPlan { uint64_t base, length; };
// Cut [base, base + bytes) into chunks of at most `cap`. row_bytes == 0: cap-sized chunks from base (the
// UringFileReader behavior). row_bytes > 0: each chunk is floor(cap / row_bytes) whole rows, the last the
// remainder, so every row [base + k*row_bytes, +row_bytes) lies in exactly one chunk. Throws
// std::invalid_argument when row_bytes > cap, or bytes is not a multiple of row_bytes.
std::vector<ChunkPlan> plan_chunks(uint64_t base, uint64_t bytes, uint64_t row_bytes, uint64_t cap);

class RegisteredBufferTable {
 public:
  // Sparse-registers `slots` empty entries on `ring`. False: the ring does not support it (errno in last_error()).
  bool init(io_uring* ring, unsigned slots);
  bool supported() const;
  // Registers plan_chunks(base, bytes, row_bytes, cap) into free slots. Returns the chunk count; 0 on overlap, no
  // free slot or a kernel refusal, after rolling back every chunk of this call (last_error() says which).
  int64_t add(uint64_t base, uint64_t bytes, uint64_t row_bytes, uint64_t cap = kMaxRegisteredBufferBytes);
  int find(uint64_t destination, uint64_t length) const;  // slot holding the range whole, or -1
  int64_t remove_range(uint64_t low, uint64_t high);       // unregister every chunk inside; returns the count
  void clear();                                            // forget every slot (after the ring was torn down)
  size_t chunks() const; uint64_t bytes() const; uint64_t largest() const;
  int last_error() const;                                  // positive errno of the last failed add, or 0
  std::string last_error_context() const;                  // "overlap", "no free slot", or the failing update
};
}  // namespace sglang::io
```

- [ ] **Step 1: Write the failing native test.** A C++ harness compiled like `test_expert_stream_uring_native.py`'s fixture (`c++ -std=c++20 -O1 -UNDEBUG -I python/sglang/kernels/jit/csrc ... -luring`). `plan_chunks` is tested on fake addresses (no memory). The table is tested against a real ring on small `mmap`ed buffers.

```python
"""Row-aligned registered-buffer chunks shared by UringFileReader and the expert-stream reader (plan
2026-09-28-reader-crtp-uring-registration Task 6). plan_chunks is address math (fake addresses, so tier-sized slabs cost
nothing); RegisteredBufferTable runs against a real ring on small buffers."""

import shutil
import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SOURCE = r'''
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <sys/mman.h>
#include "io/registered_buffers.h"
using namespace sglang::io;
constexpr uint64_t G = 1ULL << 30;
template <class F> static bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

// Every row [base + k*row, +row) lies in exactly one chunk, and the chunks tile the slab in order.
static void every_row_in_one_chunk(uint64_t base, uint64_t bytes, uint64_t row, uint64_t cap) {
  const auto plan = plan_chunks(base, bytes, row, cap);
  uint64_t at = base;
  for (const auto& c : plan) {
    assert(c.base == at && c.length > 0 && c.length <= cap && c.length % row == 0);
    at += c.length;
  }
  assert(at == base + bytes);
  for (uint64_t k = 0; k < bytes / row; ++k) {
    const uint64_t lo = base + k * row, hi = lo + row;
    int holders = 0;
    for (const auto& c : plan) holders += c.base <= lo && hi <= c.base + c.length;
    assert(holders == 1);
  }
}

int main() {
  // A dsv41-sized named slab: 2.69 GB of 3,501,056 B rows (not a divisor of 1 GiB). Chunks are 306 rows
  // (1,071,323,136 B) and the last takes the remainder: the non-divisible last chunk.
  const uint64_t row = 3501056, rows = 768, base = 0x7f0000001000ull;
  auto plan = plan_chunks(base, rows * row, row, G);
  assert(plan.size() == 3);
  assert(plan[0].length == (G / row) * row && plan[1].length == plan[0].length);
  assert(plan[2].length == rows * row - 2 * plan[0].length && plan[2].length % row == 0);
  every_row_in_one_chunk(base, rows * row, row, G);
  // Divisible: 1 GiB of 4 KiB rows is exactly one chunk; one more row makes a 4 KiB last chunk.
  assert(plan_chunks(base, G, 4096, G).size() == 1);
  auto two = plan_chunks(base, G + 4096, 4096, G);
  assert(two.size() == 2 && two[1].length == 4096);
  // A lowered test cap over a small slab: 49152 B rows, 64 KiB cap -> one row per chunk.
  every_row_in_one_chunk(0x10000, 12 * 49152, 49152, 65536);
  assert(plan_chunks(0x10000, 12 * 49152, 49152, 65536).size() == 12);
  // row_bytes == 0 keeps UringFileReader's plain chunks from base.
  auto plain = plan_chunks(base, 2 * G + 5, 0, G);
  assert(plain.size() == 3 && plain[0].length == G && plain[2].length == 5);
  // Refusals: a row larger than the cap, a slab that is not whole rows.
  assert(throws([] { plan_chunks(0x1000, 4 * (G + 4096), G + 4096, G); }));
  assert(throws([] { plan_chunks(0x1000, 10000, 4096, G); }));

  // The table on a real ring: rows of 16 KiB, cap 64 KiB -> 4 rows per chunk; 10 rows -> 3 chunks (4, 4, 2).
  io_uring ring{};
  assert(io_uring_queue_init(8, &ring, 0) == 0);
  RegisteredBufferTable t;
  assert(t.init(&ring, 16));
  const size_t n = 10 * 16384;
  auto* mem = static_cast<uint8_t*>(mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  const uint64_t b = reinterpret_cast<uint64_t>(mem);
  assert(t.add(b, n, 16384, 65536) == 3 && t.chunks() == 3 && t.bytes() == n && t.largest() == 65536);
  for (uint64_t k = 0; k < 10; ++k) assert(t.find(b + k * 16384, 16384) >= 0);
  assert(t.find(b + 3 * 16384, 2 * 16384) == -1);  // rows 3 and 4 straddle chunks 0 and 1
  assert(t.add(b + 4096, 4096, 4096, 65536) == 0 && t.last_error_context() == "overlap");
  assert(t.remove_range(b, b + n) == 3 && t.chunks() == 0 && t.find(b, 16384) == -1);
  // No free slot: 16 slots, 17 chunks requested -> 0 and nothing left registered.
  auto* big = static_cast<uint8_t*>(mmap(nullptr, 17 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(t.add(reinterpret_cast<uint64_t>(big), 17 * 4096, 4096, 4096) == 0 && t.chunks() == 0);
  assert(t.last_error_context() == "no free slot");
  io_uring_queue_exit(&ring);
  std::puts("PASS registered buffer table");
  return 0;
}
'''


def test_registered_buffer_table(tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / "table.cpp"
    source.write_text(_SOURCE)
    binary = tmp_path / "table"
    built = subprocess.run(
        [compiler, "-std=c++20", "-O1", "-UNDEBUG", "-I", str(root / "python/sglang/kernels/jit/csrc"), str(source),
         "-luring", "-o", str(binary)], capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30, check=False)
    assert run.returncode == 0 and "PASS registered buffer table" in run.stdout, run.stdout + run.stderr
```

Commit, SYNC, then CPU-TEST `test/registered/unit/kernels/test_registered_buffer_table.py`. Expected: FAIL (`io/registered_buffers.h: No such file`).

- [ ] **Step 2: Write `io/registered_buffers.h`.** Move the logic out of `UringFileReaderObj`. Its `Buffer {base, length, slot}` struct, `slot_used_`, the `register_buffer` loop, `find_buffer_`, `unregister_range_`, `rollback_` and `sort_buffers_` become the table's members. The bodies are kept except for:
  - The loop's `std::min(remaining, kMaxRegisteredBufferBytes)` becomes iteration over `plan_chunks(base, bytes, row_bytes, cap)`.
  - Every `return 0` now records `last_error_`/`last_error_context_` first: `"overlap"`, `"no free slot"`, or `"update slot N: <strerror>"` with the negated rc of `io_uring_register_buffers_update_tag`.
  - `find` returns the **slot**, as `find_buffer_` does.
  - `plan_chunks`:

```cpp
inline std::vector<ChunkPlan> plan_chunks(uint64_t base, uint64_t bytes, uint64_t row_bytes, uint64_t cap) {
  if (cap == 0 || cap > kMaxRegisteredBufferBytes) throw std::invalid_argument("registered chunk cap must be in (0, 1 GiB]");
  uint64_t step = cap;
  if (row_bytes > 0) {
    if (row_bytes > cap)
      throw std::invalid_argument("a " + std::to_string(row_bytes) + " B row does not fit one registered buffer of " + std::to_string(cap) + " B");
    if (bytes % row_bytes != 0) throw std::invalid_argument("a registered slab must be whole rows");
    step = (cap / row_bytes) * row_bytes;
  }
  std::vector<ChunkPlan> plan;
  for (uint64_t at = 0; at < bytes; at += step) plan.push_back({base + at, std::min(step, bytes - at)});
  return plan;
}
```

- [ ] **Step 3: Make `UringFileReaderObj` use the table.**
  - The constructor's `buffers_supported_ = io_uring_register_buffers_sparse(...) == 0` becomes `buffers_supported_ = table_.init(&ring_, kRegisteredBufferSlots);`.
  - `register_buffer(address, nbytes)` becomes `return buffers_supported_ && nbytes > 0 ? table_.add(address, nbytes, 0) : 0;`. Its public silent-zero contract is unchanged: this reader's callers already fall back to per-read pinning.
  - `unregister_buffer`/`unregister_range_` becomes `table_.remove_range`; `find_buffer_` becomes `table_.find`; `shutdown_` calls `table_.clear()` after `io_uring_queue_exit`.
  - Delete the moved members.

- [ ] **Step 4: Commit, SYNC, run**

CPU-TEST `test/registered/unit/kernels/test_registered_buffer_table.py test/registered/unit/kernels/test_uring_file_reader.py test/registered/unit/layers/moe/test_exl3_row_reader.py`

Expected: EXIT=0, and the two regression files pass with Task 0's counts for them.

```bash
git add python/sglang/kernels/jit/csrc/io/registered_buffers.h python/sglang/kernels/jit/csrc/io/uring_file_reader.cpp \
        test/registered/unit/kernels/test_registered_buffer_table.py
git commit -m "refactor(io): extract UringFileReader's chunked buffer registration into a row-aligned table

RegisteredBufferTable (sparse slots, update_tag, lookup, range release) is UringFileReader's logic moved
into io/registered_buffers.h; plan_chunks adds row alignment (floor(1 GiB / row_bytes) rows per chunk, the
remainder last) so every slab row lies in one registered buffer. UringFileReader passes row_bytes 0 and
keeps its behavior; its suites are green.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/reader-crtp-uring
```

---

### Task 7: Registered buffers and files in the expert-stream reader, with parallel fan-out

The Codex layer (Task 2) registers whole regions with dense `io_uring_register_buffers`. That is exactly what failed on 2026-09-28: 40 per-layer arenas of up to 2.69 GB, refused with EFAULT. This task replaces it with Task 6's table over **one region per named slab and one for the bounce**, each carrying its `row_bytes`. The slab allocation itself is untouched: still one contiguous `allocate_host_slab` per name per layer, CUDA-pinned with `cudaHostRegister`. Only the registration is cut into chunks.

**Why a read fans out, and how (recorded decision 3: parallel fan-out).** Every iovec the reader builds lies inside one slab row: `image_iovecs` emits one iovec per segment, within that segment's destination row, and the bounce path reads one bounce slot. Row-aligned chunks therefore guarantee that no iovec straddles two registered buffers, so nothing is ever clipped. But a row-image read (a part, or a piece-streaming sub-read) covering k segments has iovecs in k *named* slabs, which are k registered buffers, and `READV_FIXED` takes one buffer index per SQE. Such a read is prepared as **k legs**, and all of them are submitted together:
- A **leg** is a run of consecutive iovecs sharing one buffer, prepared as one `READV_FIXED` SQE (for `READ_MODE=fixed`, one iovec per leg, `READ_FIXED`).
- The logical read, meaning the descriptor, completes when its last leg lands.

The bounce path always has one leg. Default (non-fixed) reads are one leg covering the whole descriptor, which reduces exactly to today's code; the golden test pins that.

**Invariants, before and after:**

| | Today | With legs |
|---|---|---|
| Unit of state | descriptor (one SQE in flight at most) | descriptor with `legs` (≤ `kMaxLegs = 16`) and `legs_inflight`, plus a per-leg `Leg {start, bytes, expected, done, first_iov, iov_count, buffer, state}` |
| SQE tag | `generation << 32 \| index` | `generation << 32 \| leg << 24 \| index` (`index < 2^24`; `size_extents` refuses more) |
| Ring credit `pending` | SQEs prepared and not reaped, ≤ `capacity` | unchanged: **SQEs**, not logical reads, so `pending == UringReader::outstanding_` stays exact for `drain(pending)` |
| Preparation | one SQE per queued descriptor | **all-or-nothing**: the descriptor's Idle legs are prepared together only if `pending + n <= capacity` (or `pending == 0`) **and** the SQ has `n` free entries; otherwise nothing is prepared and the descriptor stays at the queue head |
| Queue | holds each descriptor at most once | unchanged, and enforced by a `queued` flag: a descriptor is re-queued once however many of its legs need resubmission |
| Retire | on the completion that brings `done` to `expected` | only when `legs_inflight == 0` **and** every leg has `done == expected`. `retire()` throws `logic_error` if `legs_inflight != 0` |
| Short read | resubmit the descriptor from `done` | resubmit **that leg only**, from its own `done`: its iovecs are advanced in place, since the kernel no longer holds them after the reap |
| `-EINTR/-EAGAIN` | requeue, `retries++` (≤ `kMaxRetries` per descriptor) | the same, per leg; the retry budget stays per descriptor |
| Error on one leg | `c.failed = true` | unchanged: `c.failed` is set once, no leg of any read is prepared again, and `read()` then runs `drain(c.pending)`, which reaps **every** SQE still in flight before returning. The caller releases the slots only after that, so no destination is freed while a leg can still write it, and a failed read is never retired |
| Piece-stream publish | `land_sub_read` at retire | unchanged. Retire now waits for the last leg, so a sub-read's piece bit is published only after every leg of that sub-read has landed |
| Minimum depth | none | fixed modes refuse `open()` when `queue_depth() < max_iovecs()` (clear error). A lowered `max_outstanding` fault can still prepare an over-wide descriptor, but only when `pending == 0` |

**Knobs.** No new environment variables. The ported Codex knobs are `SGLANG_EXPERT_STREAM_URING_READ_MODE={normal,fixed,readv_fixed}`, `FIXED_FILES={0,1}` and `SLAB_ARENA={0,1}`. Test-only fault words: 28 `fixed_chunk_cap`; 29 `leg`, which narrows `part_short`, `part_error` and `hold_ordinal` to leg `leg` (-1: any).

**Files:**
- Modify: `host/uring_reader.h`, `host/faulty_reader.h`, `host/row_tables.h`, `host/reader_core.h`, `host/pack_reader.h`, `host/row_reader.h`, `host/any_reader.h`, `host/read_fault.h`, `host/ffi_exports.h`, `python/sglang/kernels/ops/moe/expert_stream_transport.py`
- Create: `test/registered/unit/kernels/test_expert_stream_fixed_buffers.py`
- Modify: `test/registered/unit/kernels/test_expert_stream_buffer_regions.py`, and the region-shape and fixed-prep call sites in `test_expert_stream_uring_native.py`, `test_expert_stream_uring_integration.py` and `test_expert_stream_uring_options.py`

**Interfaces:**
- Consumes: `sglang::io::RegisteredBufferTable`, `plan_chunks` (Task 6); `configure_resources` (Task 2); `ReaderCore::refill`/`reap`/`process`/`retire`/`account_unfinished` and the hook `registered_regions()` (Task 4).
- Produces:
  - `struct RegisteredRegion { void* base; size_t bytes; size_t row_bytes; };` in `row_tables.h`. `Tables::buffer_regions` becomes `std::vector<RegisteredRegion>`, and the tensor becomes `(N, 3)`. `registered_regions()` returns it: `PackReader` gives `{bounce_, kBounceSlots * slot_bytes, slot_bytes}`, and `RowReader` gives `t_.buffer_regions`.
  - Python `_table_buffer_regions(tables) -> (N, 3)`: one row per distinct slab tensor, as `(data_ptr, nbytes, nbytes // shape[0])`. Arena owners are ignored.
  - `struct FixedLeg { unsigned first, count; int buffer; size_t bytes; };`
  - `UringReader`:
    - `configure_resources(const std::vector<int>&, const std::vector<RegisteredRegion>&, bool direct)`
    - `set_fixed_chunk_cap(size_t)`
    - `bool fixed_reads() const`
    - `unsigned fixed_legs(const iovec* iov, unsigned count, FixedLeg* out) const`, which groups consecutive iovecs by registered buffer (one per leg for `READ_MODE=fixed`) and throws if an iovec lies in no buffer
    - `bool prep_readv_fixed(int fd, const iovec* iov, unsigned count, uint64_t off, int buffer, uint64_t tag)`
    - `unsigned sq_space() const` (`io_uring_sq_space_left`)
    - `void note_fanout(unsigned legs)`, which only counts, for diagnostics
    - `uint64_t fixed_cuts() const`, `uint64_t fanout_sqes() const`
  - `FaultyReader` forwards all of these behind `requires` clauses.
  - `ReaderCore`: `set_fixed_chunk_cap(int64_t)`; `int64_t fixed_cuts() const` (logical reads prepared as more than one leg, first attempts); `int64_t fanout_sqes() const` (the SQEs those reads issued, first attempts); `static constexpr unsigned kMaxLegs = 16`. `AnyReader` forwards the three.
  - The `read_rows_sqes` info tensor becomes 7 words, `[..., fixed_cuts, fanout_sqes]`. Its dict gains `fixed_cuts` and `fanout_sqes`. `read_rows_with_fault`'s `stats` also gains both, so its results tensor becomes 10 words.
  - Refusal policy (recorded decision 2): a clear error from `open()`, never a fallback. It fires when `READ_MODE != normal` or `FIXED_FILES=1` and any registration step fails, or when a fixed mode's `queue_depth() < max_iovecs()`. The message names regions, chunks, bytes, largest chunk, cap, `RLIMIT_MEMLOCK` and `last_error_context()`. `UringFileReader` keeps its own silent fallback.

- [ ] **Step 1: Write the failing tests**

```python
"""Registered buffers and files in the expert-stream reader, with parallel fan-out (plan
2026-09-28-reader-crtp-uring-registration Task 7). A test-only cap (fault word fixed_chunk_cap) makes the fixture's
small slabs register as many row-aligned chunks; fault word `leg` aims the existing short/error/hold faults at one
leg of a fanned-out read. The real >1 GiB slab is the manual test of Task 8."""

import errno
import os
import subprocess
import sys

import pytest

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes, read_rows_with_fault
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PREFIX = "SGLANG_EXPERT_STREAM_URING_"
EXPERTS = [10, 3, 7, 0, 11, 5, 1, 8, 2, 9, 4]
SLOTS = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
CAP = 64 * 1024


@pytest.fixture
def uring_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(PREFIX):
            monkeypatch.delenv(key)

    def set_(**values):
        for key, value in values.items():
            monkeypatch.setenv(PREFIX + key, str(value))

    return set_


def _setup(tmp_path, images, weights=None):
    root = tmp_path / "ckpt"
    root.mkdir()
    dims = {} if images else dict(hidden=256, inter=512)
    return ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)


def _snapshot(slabs):
    if isinstance(slabs, dict):
        return {k: _snapshot(v) for k, v in slabs.items()}
    return slabs.clone()


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return same_bytes(a, b)


def _read(s, **faults):
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, direct=False, **faults)
    return result, log, info, record, _snapshot(s.slabs)


def _supported(fn):
    try:
        return fn()
    except RuntimeError as e:
        if "unsupported by the running kernel" in str(e) or "requires liburing 2.10" in str(e):
            pytest.skip(str(e))
        raise


def _pieces(images, on):
    return ({"piece_stream": True} | ({} if images else {"pack_workers": 2})) if on else {}


def test_regions_are_one_per_slab_with_its_row_bytes(tmp_path):
    s = _setup(tmp_path, True)
    regions = ops._table_buffer_regions(s.tables).tolist()
    assert regions and all(nbytes % row == 0 and row > 0 for _, nbytes, row in regions)
    assert len({base for base, _, _ in regions}) == len(regions)


@pytest.mark.parametrize("images", [False, True], ids=["bounce", "images"])
@pytest.mark.parametrize("read_mode", ["fixed", "readv_fixed"])
@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_fanned_out_reads_are_byte_identical(tmp_path, uring_env, images, read_mode, pieces):
    s = _setup(tmp_path, images)
    base_result, base_log, _, base_rec, base_bytes = _read(s, **_pieces(images, pieces))
    uring_env(READ_MODE=read_mode)
    result, log, info, rec, fixed_bytes = _supported(lambda: _read(s, fixed_chunk_cap=CAP, **_pieces(images, pieces)))
    assert base_result == result == 1 and _equal(fixed_bytes, base_bytes)
    assert rec["retried_bytes"] == 0 and rec["submitted_bytes"] == base_rec["submitted_bytes"]
    assert rec["bytes"] == base_rec["bytes"] and rec["useful_bytes"] == base_rec["useful_bytes"]
    assert sum(entry[2] for entry in log) == rec["submitted_bytes"]
    if images:  # a multi-slab image read fans out: more SQEs than logical reads, the same byte ranges in the file
        assert info["fixed_cuts"] > 0 and info["fanout_sqes"] > info["fixed_cuts"]
        assert len(log) == len(base_log) - info["fixed_cuts"] + info["fanout_sqes"]
    else:       # one bounce slot is one row: never fanned out
        assert info["fixed_cuts"] == info["fanout_sqes"] == 0 and sorted(log) == sorted(base_log)


def test_normal_mode_never_fans_out(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    result, _, info, _, _ = _read(s, fixed_chunk_cap=CAP)
    assert result == 1 and info["fixed_cuts"] == info["fanout_sqes"] == 0


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_legs_completing_out_of_order(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    uring_env(READ_MODE="readv_fixed")
    # reverse_cqes: every reaped batch (waiting for all in flight) is processed back to front, so each read's legs
    # land last-first.
    result, _, info, _, fixed_bytes = _supported(
        lambda: _read(s, fixed_chunk_cap=CAP, reverse_cqes=True, **_pieces(True, pieces)))
    assert result == 1 and info["fixed_cuts"] > 0 and _equal(fixed_bytes, base_bytes)


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_a_held_leg_keeps_its_read_unretired_and_its_pieces_unpublished(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    uring_env(READ_MODE="readv_fixed")
    # Leg 1 of every read of row 0 is withheld until nothing else is pending: the row may neither retire nor (piece
    # streaming) publish a piece that depends on it before it lands. A reader that retired on the first completion
    # would vet row 0 short and fail the read.
    result, _, _, rec, fixed_bytes = _supported(lambda: _read(
        s, fixed_chunk_cap=CAP, hold_ordinal=0, leg=1, **_pieces(True, pieces)))
    assert result == 1 and _equal(fixed_bytes, base_bytes)
    if pieces:
        assert rec["pieces_published"] == len(EXPERTS) * 8 and rec["piece_publish_refused"] == 0


def test_one_short_leg_resubmits_only_that_leg(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE="readv_fixed")
    _, clean_log, clean_info, _, base_bytes = _supported(lambda: _read(s, fixed_chunk_cap=CAP))
    result, log, info, rec, fixed_bytes = _read(s, fixed_chunk_cap=CAP, part=0, part_short=512, leg=1)
    assert result == 1 and _equal(fixed_bytes, base_bytes)
    assert len(log) == len(clean_log) + 1                    # exactly one extra SQE: the short leg's remainder
    extra = sorted(set(log) - set(clean_log))
    assert len(extra) == 1 and rec["retried_bytes"] == extra[0][2]
    assert info["fanout_sqes"] == clean_info["fanout_sqes"]  # first attempts only


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_one_failing_leg_fails_the_read_once_after_every_leg_is_reaped(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE="readv_fixed")
    stats, cqes = {}, []
    # Leg 1 of row 0 fails with EIO while its other legs succeed. The read fails once; read() drains every SQE still
    # in flight before returning (unfinished_jobs 0, and the same reader's next read succeeds on a clean ring).
    first, then = _supported(lambda: read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=False, part=0, part_error=errno.EIO,
        ordinal=0, leg=1, fixed_chunk_cap=CAP, stats=stats, cqes=cqes, **_pieces(True, pieces)))
    assert (first, then) == (0, 1)
    assert stats["unfinished_jobs"] == 0 and stats["fixed_cuts"] > 0


@pytest.mark.parametrize("submit_first", [False, True], ids=["unconsumed", "in_flight"])
@pytest.mark.parametrize("read_mode", ["fixed", "readv_fixed"])
def test_ring_reset_mid_fan_out(tmp_path, uring_env, read_mode, submit_first):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE=read_mode, FIXED_FILES=1)
    # The first submit fails with a fanned-out read's legs prepared: either none reached the kernel (drain resets the
    # ring and must re-create the sparse table, re-add every chunk and re-register the files) or all did (drain waits
    # for every leg). Either way the clean second read on the same reader succeeds.
    first, then = _supported(lambda: read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=False,
        submit_error=errno.EIO, submit_call=1, submit_first=submit_first, fixed_chunk_cap=CAP))
    assert (first, then) == (0, 1)


@pytest.mark.parametrize("images", [False, True], ids=["bounce", "images"])
@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0), (1.0, 0.0, 1.0)], ids=["three", "zero_mid"])
def test_fixed_files_with_three_mirror_roots(tmp_path, uring_env, images, weights):
    s = _setup(tmp_path, images, weights)
    base_result, base_log, _, _, base_bytes = _read(s)
    uring_env(FIXED_FILES=1)
    result, log, _, _, fixed_bytes = _read(s)
    assert base_result == result == 1 and sorted(log) == sorted(base_log) and _equal(fixed_bytes, base_bytes)
    roots_read = {str(r) for f, *_ in log for r in s.roots if s.tables.paths[f].startswith(str(r))}
    assert len(roots_read) == sum(1 for w in weights if w > 0)
    uring_env(READ_MODE="readv_fixed")
    result, _, _, _, both_bytes = _supported(lambda: _read(s, fixed_chunk_cap=CAP))
    assert result == 1 and _equal(both_bytes, base_bytes)


def test_registration_refusal_is_a_clear_error_not_a_fallback(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE="readv_fixed")
    with pytest.raises(RuntimeError, match="does not fit one registered buffer"):
        _read(s, fixed_chunk_cap=4096)  # the fixture's 49152 B rows exceed a 4 KiB cap
    uring_env(QUEUE_DEPTH=2)
    with pytest.raises(RuntimeError, match="queue depth"):
        _read(s, fixed_chunk_cap=CAP)  # a fanned-out read of up to 6 legs cannot be reserved in a depth-2 ring
    uring_env(QUEUE_DEPTH=0)
    if os.geteuid() == 0:
        pytest.skip("root has CAP_IPC_LOCK: the memlock limit does not bind")
    child = subprocess.run(
        [sys.executable, "-c", _MEMLOCK_CHILD, str(tmp_path / "child")],
        env=dict(os.environ, **{PREFIX + "READ_MODE": "readv_fixed"}), capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "REFUSED" in child.stdout and "RLIMIT_MEMLOCK" in child.stdout, child.stdout


_MEMLOCK_CHILD = r'''
import pathlib, resource, sys
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True)
s = ram_miss_setup(root, capacity=12, experts=12, row_images=True)
resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
try:
    read_rows_sqes(s.tables, 1, [0, 1], [0, 1], direct=False)
    print("READ WITHOUT REGISTRATION")  # a silent fallback: the missing REFUSED fails the test
except RuntimeError as e:
    print("REFUSED", e)
'''
```

The fixture's image rows span 6 segments. If `ram_miss_setup`'s image segments turn out to be fewer than 2 per sub-read with piece streaming, so that no piece-streaming read fans out, raise `capacity`/`experts` or drop `pieces=True` from the fan-out assertions only, and say so in the commit message. Commit, SYNC, then CPU-TEST `test/registered/unit/kernels/test_expert_stream_fixed_buffers.py`. Expected: FAIL (`_table_buffer_regions` returns `(N, 2)`, or `unexpected keyword argument 'fixed_chunk_cap'`).

- [ ] **Step 2: Regions carry `row_bytes`.**
  - Python `_table_buffer_regions`:

```python
def _table_buffer_regions(tables) -> torch.Tensor:
    """One registration region per slab tensor, with its row size: the reader cuts registration on row boundaries
    (plan_chunks), so the slab allocation itself stays one contiguous tensor. Arena owners are not registered whole: a
    2.69 GB arena is past io_uring's 1 GiB per-buffer limit (the 2026-09-28 S3 refusal)."""
    regions: dict[int, tuple[int, int, int]] = {}
    for slab in getattr(tables, "keepalive", ()):
        if not isinstance(slab, torch.Tensor):
            raise ValueError("I/O buffer owners must be tensors")
        if slab.device.type != "cpu" or not slab.is_contiguous():
            raise ValueError("I/O buffer slabs must be contiguous CPU tensors")
        nbytes = slab.numel() * slab.element_size()
        if nbytes and slab.dim() >= 1 and slab.shape[0] > 0:
            regions.setdefault(slab.data_ptr(), (slab.data_ptr(), nbytes, nbytes // slab.shape[0]))
    return torch.tensor(list(regions.values()), dtype=torch.int64).reshape(-1, 3)
```

  - `row_tables.h`: add `RegisteredRegion`. `tables_from` reads `(N, 3)` and refuses `base <= 0`, `bytes <= 0`, `row_bytes <= 0`, `bytes % row_bytes != 0`, or overflow.
  - `ffi_exports.h`: the matcher becomes `{-1, 3}`.
  - Update the Codex tests listed under **Files**: region shape `(N, 3)`. Their native harness now calls `fixed_legs` + `prep_readv_fixed` wherever it prepared fixed reads, and its `configure_resources` gets `RegisteredRegion{base, bytes, bytes}`.

- [ ] **Step 3: `UringReader` registers through the shared table.** Edit the Task 2 file:
  - Add `#include "../../io/registered_buffers.h"`.
  - Members: `sglang::io::RegisteredBufferTable table_; std::vector<RegisteredRegion> regions_; size_t chunk_cap_ = sglang::io::kMaxRegisteredBufferBytes; std::vector<int> fd_index_; uint64_t fixed_reads_ = 0, fixed_cuts_ = 0, fanout_sqes_ = 0, next_report_ = uint64_t{1} << 16; double register_ms_ = 0;`.
  - `configure_resources` keeps the Codex checks, stores `regions_` (fixed modes) and builds `fd_index_` (fixed files: size max fd + 1, filled with -1, then `fd_index_[fds[i]] = i`).
  - `register_resources()` covers buffers and files:

```cpp
    if (!regions_.empty()) {
      size_t slots = 0;
      for (const auto& r : regions_) slots += sglang::io::plan_chunks(
          reinterpret_cast<uint64_t>(r.base), r.bytes, r.row_bytes, chunk_cap_).size();  // throws: row > cap
      if (slots > sglang::io::kMaxRegisteredBufferSlots) refuse(slots, "more chunks than the kernel's 16384 slots");
      const auto t0 = std::chrono::steady_clock::now();
      if (!table_.init(&ring_, static_cast<unsigned>(slots))) refuse(slots, "sparse buffer table unsupported");
      for (const auto& r : regions_) {
        if (table_.add(reinterpret_cast<uint64_t>(r.base), r.bytes, r.row_bytes, chunk_cap_) == 0) refuse(slots, table_.last_error_context());
      }
      register_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      buffers_registered_ = true;
    }
```

    `refuse(slots, why)` tears down whatever registration exists: `io_uring_unregister_buffers` if the table was initialised, and the files if registered. It then throws `std::runtime_error("expert stream registering fixed buffers (regions=R, chunks=C, bytes=B, largest=L, cap=K, RLIMIT_MEMLOCK=<value|unlimited>): " + why)`, where the memlock value comes from `getrlimit(RLIMIT_MEMLOCK)`.
  - `close_ring()`: `if (buffers_registered_) { io_uring_unregister_buffers(&ring_); table_.clear(); }`. `drain()`'s reset path already calls `register_resources()`, which re-creates the sparse table and re-adds every chunk.
  - Legs and fixed preparation (replacing Codex's `buffer_index` and the scalar `readv_fixed` map):

```cpp
  bool fixed_reads() const { return options_.read_mode != UringReadMode::Normal; }
  unsigned sq_space() const { return io_uring_sq_space_left(&ring_); }
  void set_fixed_chunk_cap(size_t cap) {
    if (configured_) throw std::logic_error("the fixed-buffer chunk cap is set before resources are configured");
    chunk_cap_ = cap == 0 ? sglang::io::kMaxRegisteredBufferBytes : cap;
  }
  // Consecutive iovecs sharing one registered buffer form a leg (READ_FIXED: one iovec per leg). Rows never straddle
  // buffers (plan_chunks), so every iovec lies in exactly one buffer and nothing is clipped.
  unsigned fixed_legs(const iovec* iov, unsigned count, FixedLeg* out) const {
    unsigned legs = 0;
    for (unsigned i = 0; i < count; ++i) {
      const int b = table_.find(reinterpret_cast<uint64_t>(iov[i].iov_base), iov[i].iov_len);
      if (b < 0) throw std::invalid_argument("a fixed read's destination lies in no registered buffer");
      if (legs > 0 && out[legs - 1].buffer == b && options_.read_mode == UringReadMode::ReadvFixed) {
        ++out[legs - 1].count;
        out[legs - 1].bytes += iov[i].iov_len;
      } else {
        out[legs++] = FixedLeg{i, 1, b, iov[i].iov_len};
      }
    }
    return legs;
  }
  void note_fanout(unsigned legs) {
    ++fixed_reads_;
    if (legs > 1) { ++fixed_cuts_; fanout_sqes_ += legs; }
    if (options_.diagnostics && fixed_reads_ >= next_report_) { next_report_ <<= 1; report_fixed(); }
  }
  uint64_t fixed_cuts() const { return fixed_cuts_; }
  uint64_t fanout_sqes() const { return fanout_sqes_; }
  bool prep_readv_fixed(int fd, const iovec* iov, unsigned count, uint64_t off, int buffer, uint64_t tag) {
    require_ready();
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    if (count == 1 && options_.read_mode == UringReadMode::Fixed)
      io_uring_prep_read_fixed(sqe, mapped_fd(fd), iov[0].iov_base, static_cast<unsigned>(iov[0].iov_len), off, buffer);
#if SGLANG_URING_HAS_READV_FIXED
    else
      io_uring_prep_readv_fixed(sqe, mapped_fd(fd), iov, count, off, 0, buffer);
#endif
    finish_prep(sqe, tag);
    return true;
  }
```

  - `prep_read`/`prep_readv` in a fixed read mode throw `std::logic_error("fixed read modes prepare through prep_readv_fixed")`.
  - `mapped_fd` becomes the O(1) `fd_index_` lookup, throwing when the fd is absent.
  - `diagnostics()` appends `regions=%zu chunks=%zu largest_chunk=%llu chunk_cap=%zu register_ms=%.1f`. `report_fixed()` prints `expert stream io_uring: fixed_reads=%llu fixed_cuts=%llu fanout_sqes=%llu`, and is also called from `close()` when diagnostics are on.
  - `faulty_reader.h` forwards `set_fixed_chunk_cap`, `fixed_reads`, `sq_space`, `fixed_legs`, `prep_readv_fixed`, `note_fanout`, `fixed_cuts` and `fanout_sqes` behind `requires` clauses, and updates the parameter type of its `configure_resources`.

- [ ] **Step 4: `ReaderCore` legs.** Apply the invariant table above:
  1. **Tags.** Add `static uint64_t make_tag(uint32_t generation, uint32_t index, unsigned leg)`, `static uint32_t tag_index(uint64_t)` (`& 0xFFFFFF`) and `static unsigned tag_leg(uint64_t)` (`>> 24 & 0xFF`). Replace **every** decode of `completion.data` (in `process`, in `reap`'s `hold_ordinal` filter, and in the stale-CQE fault) with them. `size_extents` refuses `extents > 0xFFFFFF`, and `open()` refuses `max_iovecs() > kMaxLegs`.
  2. **Descriptor state.** `ExtentDesc` gains `uint8_t legs = 0, legs_inflight = 0; bool queued = false;`. A member `std::vector<Leg> legs_` is sized `descs_.size() * kMaxLegs` in `size_extents`, with:

```cpp
  enum class LegState : uint8_t { Idle, Inflight, Done };
  struct Leg {
    int64_t start = 0;     // the leg's first byte, from the read's start (d.read->offset / dest)
    int64_t bytes = 0;     // what the leg covers
    int64_t expected = 0;  // bytes, clamped at the read's end-of-file expectation (d.expected)
    int64_t done = 0;
    unsigned first_iov = 0, iov_count = 0;
    int buffer = -1;       // registered buffer; -1 outside fixed modes
    LegState state = LegState::Idle;
  };
```

  3. **Planning a descriptor's legs**, `plan_legs(index)`, on its first preparation. It builds the iovecs with `derived().destination(d, iov)` from `done = 0`. In fixed modes it takes `io_.fixed_legs(...)`; otherwise it makes one leg covering everything (`first_iov 0, iov_count count, buffer -1`). Each leg's `start` is the prefix sum of the earlier legs' bytes, and `expected = clamp(d.expected - start, 0, bytes)`. A leg with `expected == 0` is marked `Done` at once (it lies past end of file). In fixed modes it calls `io_.note_fanout(legs)`, and adds to `fixed_cuts_`/`fanout_sqes_` when `legs > 1`.
  4. **`refill()`.** Pop the head descriptor only if its Idle legs fit. Let `n` be the count of `Idle` legs. The descriptor is prepared only when `(c.pending + n <= c.capacity || c.pending == 0) && (!fixed_reads() || io_.sq_space() >= n)`. Otherwise break without popping. For each Idle leg:
     - advance its iovec slice in place past `leg.done`: skip whole iovecs, then trim the partial one's base and length;
     - prepare it at file offset `d.read->offset + leg.start + leg.done`, with `prep_readv_fixed` (fixed modes), `prep_readv` (`Derived::kScatter`) or `prep_read` (bounce, non-fixed; one leg, one iovec), tagged `make_tag(d.generation, index, l)`;
     - set `state = Inflight`, `++d.legs_inflight`, `++c.pending`.
     Because space was checked first, `prep_*` returning false is now a `logic_error`. Clear `d.queued`. Trace and `sqe_log_` record one entry per leg with the leg's offset and length. `submitted_bytes += leg remaining`. `retried_bytes += leg remaining` only when `leg.done > 0 || d.retries > 0`. `extent_submit` is stamped on the descriptor's first leg; `extent_attempts` counts each resubmitted leg.
  5. **`process()`.** Decode index, leg and generation. A completion is **stale** unless the generation matches, `leg < d.legs` and that leg is `Inflight`: then `++stale_cqes_` and `c.failed = true`, as today. Otherwise set `leg.state = Idle` and `--d.legs_inflight`. Existing faults (`part_error`, `part_short`, `short_is_eof`, `cqe_error`) additionally match `fault_.leg < 0 || fault_.leg == l`. Then:
     - `-EINTR/-EAGAIN`: `++d.retries`; if the budget is exceeded, fail; else queue the descriptor if not `queued`.
     - Error, or `res == 0` with `leg.done < leg.expected`: `c.failed = true`.
     - Otherwise `leg.done += res`, `d.done += res`. With `eof`, `leg.expected = leg.done` and `d.expected -= (old leg.expected - leg.done)`. If `leg.done < leg.expected`, queue the descriptor if not `queued` (only this leg is Idle and unfinished); else `leg.state = Done`.
     - Finally, if `d.legs_inflight == 0` and every leg is `Done`, call `retire(index, completion, returned)`.
     - Queueing goes through `again_` as today, and `queue_push` sets `queued`.
  6. **`retire()`** starts with `if (d.legs_inflight != 0) throw std::logic_error(...)`. It is otherwise unchanged; `d.done` is the sum of the legs.
  7. **`account_unfinished()`** is unchanged: a live descriptor's `expected - done` is cancelled. The failure path is unchanged: `read()` breaks on `c.failed`, `drain(c.pending)` reaps every leg still in flight, and only then does `read()` return 0.
  8. **Fault word 29 `leg`.** It also narrows the `hold_ordinal` filter in `reap` to completions whose `tag_leg == fault_.leg` when `fault_.leg >= 0`.
  9. **`open()`** refuses fixed modes when `queue_depth() < derived().max_iovecs()`, with `"fixed reads need a queue depth of at least N (the widest fanned-out read); SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=M"`.

- [ ] **Step 5: Fault words and FFI info.**
  - `read_fault.h`: `kFaultWords = 30`. `fault_from` reads `fault.leg = f[29]`; word 28 (`fixed_chunk_cap`) is applied before `open()`.
  - `ffi_exports.h`: `reader.set_fixed_chunk_cap(f[28]);` before `reader.open()` in `read_rows_traced`, `read_rows_faulted`, `read_rows_sqes` and `read_rows_pieces`.
    - `read_rows_sqes`: info `{7}`, with `info[5] = fixed_cuts()` and `info[6] = fanout_sqes()`.
    - `read_rows_faulted`: results `{10}`, with `[8] = fixed_cuts()` and `[9] = fanout_sqes()`.
  - Python: `_fault_tensor(..., fixed_chunk_cap: int = 0, leg: int = -1)` is appended last. `read_rows_sqes` returns `fixed_cuts` and `fanout_sqes`; `read_rows_with_fault` allocates 10 results and adds both to `stats`.

- [ ] **Step 6: Commit, SYNC, run**

CPU-TEST `test/registered/unit/kernels/test_expert_stream_fixed_buffers.py test/registered/unit/kernels/test_registered_buffer_table.py test/registered/unit/kernels/test_expert_stream_reader_golden.py test/registered/unit/kernels/test_expert_stream_reader_split.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py test/registered/unit/kernels/test_exl3_ram_miss_row_images.py test/registered/unit/kernels/test_expert_stream_uring_options.py test/registered/unit/kernels/test_expert_stream_buffer_regions.py test/registered/unit/kernels/test_uring_file_reader.py`

Expected: EXIT=0, with no skips on divix01 (6.12 EL10 advertises `READV_FIXED`). The golden and the reader suites passing here show that the one-leg default path is unchanged. Then run the native/integration matrix (Task 2 Step 3's command); expected EXIT=0, 0 skips.

- [ ] **Step 7: Mutants** (private worktree `wt-fanout-mutant`). Each must fail the named test; revert it and re-run green.
  1. **Retire on the first completion, not the last.** In `process()`, replace the retire condition with `if (true)` right after the first successful leg (drop the `legs_inflight == 0 && all Done` check, and the `retire()` guard). Must fail: `test_a_held_leg_keeps_its_read_unretired_and_its_pieces_unpublished` (a short vetting fails the read) and `test_legs_completing_out_of_order`.
  2. Resubmit **every** leg on one leg's short read: in `refill()`, reset all legs to Idle. Must fail: `test_one_short_leg_resubmits_only_that_leg`.
  3. Return from `read()` on a leg error without draining: skip `drain(c.pending)` on the failure path. Must fail: `test_one_failing_leg_fails_the_read_once_after_every_leg_is_reaped` (the next read meets the stale completions).
  4. Skip the SQ-space and credit reservation: prepare legs one at a time as space allows. Must fail: `test_registration_refusal_is_a_clear_error_not_a_fallback`'s depth-2 case no longer refuses. Run it with the depth check also removed; expect `prep_*` false followed by the `logic_error`.
  5. `plan_chunks`: `step = cap` even when `row_bytes > 0`. Must fail: `test_registered_buffer_table`.
  6. `UringReader::drain` reset path: skip `register_resources()`. Must fail: `test_ring_reset_mid_fan_out[unconsumed-*]`.
  7. `mapped_fd`: `fd_index_[fd] + 1`. Must fail: `test_fixed_files_with_three_mirror_roots`.

- [ ] **Step 8: SUITE**, compared with Task 5's counts plus this task's and Task 6's tests. Record the command and the counts. Then:

```bash
git add -A python/sglang/kernels/jit/csrc/moe/expert_stream/host python/sglang/kernels/ops/moe/expert_stream_transport.py \
        test/registered/unit/kernels/test_expert_stream_fixed_buffers.py test/registered/unit/kernels/test_expert_stream_buffer_regions.py \
        test/registered/unit/kernels/test_expert_stream_uring_native.py test/registered/unit/kernels/test_expert_stream_uring_integration.py \
        test/registered/unit/kernels/test_expert_stream_uring_options.py
git commit -m "feat(expert-stream): register row-aligned <=1 GiB chunks; fan multi-slab fixed reads out in parallel

The reader registers one region per named slab (and the bounce) through io::RegisteredBufferTable, cut on
row boundaries, instead of whole arenas the kernel refuses past 1 GiB. A fixed read whose iovecs span k
registered buffers is k legs submitted together: credit counts SQEs and a read's legs are reserved
all-or-nothing, a short leg resubmits alone, the read retires on its last leg, and a failing leg fails
the call once, after drain reaps every leg. Default reads are one leg (golden unchanged). Refusals are
explicit errors; fixed files map fd -> index in O(1). Test-only fault words fixed_chunk_cap and leg.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin cc/reader-crtp-uring
```

---

### Task 8: divix01 real-kernel verification, including a real slab over 1 GiB

**Files:**
- Create: `test/manual/dsv41/test_expert_stream_fixed_buffers_big.py`

**Interfaces:** Consumes `UringReader::{configure_resources, fixed_legs, prep_readv_fixed, sq_space}` (Task 7), driven from a native harness as `test_expert_stream_uring_native.py` does.

- [ ] **Step 1: Write the manual test.** The native harness does the following, in the same compile form as Task 6's test, with `-luring`:
  - Map a 1.5 GiB slab of 3,501,056 B rows (460 rows, 1,610,485,760 B), anonymous and populated (THP is `always` on divix01). Map a second 64-row slab of 49,152 B rows.
  - Write a 16 MiB file of known bytes.
  - Configure a `UringReader` from env (`READ_MODE`, `FIXED_FILES=1`) with both slabs as `RegisteredRegion`s and the default 1 GiB cap.
  - Assert the table holds 2 chunks for the big slab (306 rows, then 154) plus 1 for the small one.
  - Read row 305 (the last row of chunk 0) and row 306 (the first row of chunk 1) from the file. `fixed_legs` must give one leg each, in buffers 0 and 1. Prepare each as one SQE and assert the bytes equal the file's.
  - Read one two-iovec destination (a row of the big slab plus a row of the small slab) at consecutive file offsets. `fixed_legs` must give 2 legs. Prepare both at their offsets, submit them together, `submit(2)`, reap both, and assert correct bytes.
  - Print `PASS big fixed slab chunks=N register_ms=X`.
  - The Python wrapper is `@pytest.mark.skipif(os.uname().nodename != "divix01", reason="1.5 GiB pinned: divix01 only")`.

- [ ] **Step 2: Commit, SYNC, run on node 1 memory under the disk lock**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && for mode in fixed readv_fixed; do SGLANG_EXPERT_STREAM_URING_READ_MODE=$mode SGLANG_EXPERT_STREAM_URING_FIXED_FILES=1 \
     SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS=1 flock /data/models/slang/nvfp4-work/rowimg-disk.lock \
     numactl --membind=1 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/manual/dsv41/test_expert_stream_fixed_buffers_big.py -q -s -p no:randomly \
     --basetemp=/mnt/nvme2/nvfp4-work/reader-crtp-big 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"; done'
```

Expected: EXIT=0 twice, printing `PASS big fixed slab chunks=3`. Record `register_ms` (the pin cost of 1.5 GiB), which Task 9 extrapolates to ~100 GB.

- [ ] **Step 3: GPU regression with fixed modes on, then off** (GPU lock only; this reads no NVMe row images):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  MAX_JOBS=8 CUDA_HOME=/usr/local/cuda-13.4 SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build \
  SGLANG_EXPERT_STREAM_URING_FIXED_FILES=1 SGLANG_EXPERT_STREAM_URING_READ_MODE=readv_fixed \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py \
     -q -x -p no:randomly 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: EXIT=0. The same command without the two `SGLANG_EXPERT_STREAM_URING_*` exports should also give EXIT=0 with the same counts. This covers registration over real `cudaHostRegister`-pinned slabs.

- [ ] **Step 4: Commit the manual test** with the trailer and push.

---

### Task 9: Registration decode arms on divix01 (each mode against the default, one commit)

**Files:**
- Create: `analysis/dsv41-drive/uring-reg/drive_uring_reg_arms.sh`, `analysis/dsv41-drive/uring-reg/results.md`
- Modify: `benchmarks/dsv41_baseline/generations.json`

**Interfaces:** Consumes Task 5's pair driver as the template, plus `run_arm.sh` and `generations.register`.

**What this measures, honestly.** On 2026-09-28 every working io_uring option landed within ±0.7 ms/token of the defaults (104.2 ms/token), with identical output. Fixed buffers were unsupported at tier size, and fixed files alone were never run. The expected answer is "no change". These arms are the first end-to-end measurement of both, and they report whatever they find.

Two arms only (user decision, 2026-09-28): the baseline and everything on.

| Arm | Overrides (all nine `SGLANG_EXPERT_STREAM_URING_*` set explicitly) |
|---|---|
| `R0` | S0 defaults: `MODE=default QUEUE_DEPTH=0 FIXED_FILES=0 READ_MODE=normal WAIT_MODE=block SQ_THREAD_IDLE_MS=1000 SQ_THREAD_CPU=-1 SLAB_ARENA=0 DIAGNOSTICS=1` |
| `R3` | as R0 but `SLAB_ARENA=1 READ_MODE=readv_fixed FIXED_FILES=1` (arena, fixed buffers with parallel fan-out, and fixed files together) |

With `SLAB_ARENA=1` the named slabs share one arena allocation. Registration is still one row-aligned region per named slab (`_table_buffer_regions` ignores the arena owner), so the arena changes only the allocation layout, not the chunking.

**Limits of this design, stated in `results.md`:**
- **Drift is not controlled.** There is no repeated baseline arm (no R0b), so a change in machine state between R0 and R3 (clocks, page cache, ARC, drive temperature) is indistinguishable from an effect. The 2026-09-28 campaign's S0 vs S0b differed by 0.3 ms/token, which gives a rough scale only.
- **A win cannot be attributed to a single mode.** R3 changes the arena layout, fixed buffers and fixed files at once. If R3 passes the bar below, which of the three produced the gain is unknown, and a confirmation plan must separate them.

- [ ] **Step 1: Register the tree** (`generations.register('$(git rev-parse HEAD:python)', 'reader-crtp-uring-reg')`), commit with the trailer, push and SYNC. Confirm on divix01 that HEAD is the pushed commit and the worktree is clean.

- [ ] **Step 2: Write the driver** by copying Task 5's pair driver, with one worktree and exactly the two arms R0 then R3 (no others). Keep every guard and the per-arm node-0 gate with the same tier (`0:57344,1:40960`; the one permitted cut happens before R0 only, then applies to all arms). Add after each arm's "ready" line:
  - (a) append node 0 and node 1 `MemFree` plus `VmallocUsed` and `Slab` from `/proc/meminfo` to `$OUT/memory.jsonl`, since registration pins ~100 GB and allocates kernel page tables, and node 0 is tight;
  - (b) copy the server log's `expert stream io_uring:` lines to `$OUT/<arm>-uring.txt`. If the `fixed_reads=… fixed_cuts=… fanout_sqes=…` line is absent (the server was killed before `close()`), write `not captured`.
  - (c) Stop on rc≠0, a registration refusal, or read errors > 0. Report output that differs from R0 as a failure.
  - Commit and push.

- [ ] **Step 3: Launch under the disk lock**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && mkdir -p /mnt/nvme1/uring-reg \
  && OMP_NUM_THREADS=8 nohup flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 \
     bash analysis/dsv41-drive/uring-reg/drive_uring_reg_arms.sh $PWD $(git rev-parse HEAD) /mnt/nvme1/uring-reg 30031 \
     > /mnt/nvme1/uring-reg/driver.log 2>&1 < /dev/null &'
```

Poll `driver.log` no more often than every 10 minutes. Start no other GPU or disk work until it finishes.

- [ ] **Step 4: Report** in `analysis/dsv41-drive/uring-reg/results.md`, in the shape of `uring-config/results.md`:
  - the verdict, commit, tree and host facts;
  - per arm: session and pooled ms/token and TTFT, R3's Δ vs R0, whether R3's output is byte-identical to R0's, `rows_read/served/read_errors`, startup to "fired up", `register_ms`, regions and chunks, largest chunk, `fixed_reads`, `fixed_cuts` and `fanout_sqes` (or "not captured"), node-0/node-1 `MemFree`, and `VmallocUsed`/`Slab` deltas vs R0.
  - The rule: R3 goes to a confirmation plan (≥3 alternating pairs, and arms that separate arena, fixed buffers and fixed files) only if it beats R0 by ≥1.5 ms/token with identical output; otherwise the defaults stay. Repeat the two limits above: drift is uncontrolled, and a win cannot be attributed to a single mode. Say it plainly if registration moved nothing.
  - Commit with the trailer and push.

---

## Self-review notes (for the executor)

- **Spec coverage:**
  - CRTP base and two derived readers: Tasks 3-4.
  - Behavior preserved: the Task 1 golden (unedited in Tasks 2, 3, 4 and 7), the reader suites, SUITE count diffs, and the Task 5 decode pair with byte identity.
  - Chunked registration over one contiguous slab per name, cut on row boundaries: Tasks 6-7. It reuses `uring_file_reader.cpp`'s logic via `io/registered_buffers.h`, with no second implementation.
  - Every row in one chunk and the non-divisible last chunk: Task 6.
  - A real slab over 1 GiB: Task 8.
  - Refusal as a clear error: Task 7, recorded decision 2.
  - Fixed files with 3 roots: Task 7.
  - Registration decode arms: Task 9.
- **Names used across tasks:**
  - Reader classes: `ReaderCore`, `PackReader`, `RowReader`, `AnyReader`, `SqeRecord`.
  - Registration: `sglang::io::{plan_chunks, ChunkPlan, RegisteredBufferTable, kMaxRegisteredBufferBytes, kMaxRegisteredBufferSlots}`, `RegisteredRegion`.
  - `UringReader::{set_fixed_chunk_cap, fixed_reads, sq_space, fixed_legs, prep_readv_fixed, note_fanout, fixed_cuts, fanout_sqes}`, `FixedLeg`.
  - `ReaderCore::{kMaxLegs, Leg, LegState, make_tag, tag_index, tag_leg, fixed_cuts, fanout_sqes}` and `ExtentDesc::{legs, legs_inflight, queued}`.
  - Fault words 28 `fixed_chunk_cap` and 29 `leg` (`kFaultWords = 30`); info keys `fixed_cuts` and `fanout_sqes`.

## Decisions recorded (2026-09-28)

1. **Codex knobs:** cherry-pick the five commits now (Task 2). This plan adds no knobs.
2. **Registration failure:** a clear error from `open()`, with no silent fallback in the expert reader. `UringFileReader` keeps its own fallback.
3. **Base:** `origin/mirror3-piece-stream`, and execution waits for it to merge into master (see the header).
4. **Decode arms:** two only, R0 and R3 (Task 9). Drift is uncontrolled, and a win cannot be attributed to a single mode.
5. **Team-lead rulings:**
   - **Parallel fan-out** (the user's decision 3, reversing the earlier serial-continuation ruling). A fixed read spanning k registered buffers is k legs submitted together, and it completes on the last one. Design and tests are in Task 7.
   - Migrating the Codex knobs to `environ.py` is a follow-up, not this plan.
   - IOPOLL io-wq punting is out of scope.

## Open decisions for the user

None outstanding.
