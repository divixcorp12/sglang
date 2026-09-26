# Expert-stream transport split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the CPU/NVMe-to-GPU expert-streaming machinery (request page, lease protocol, row reader, RAM tier,
service thread, device kernels, and their Python wrappers) from the EXL3 quantization format. The format and the file
reader become template arguments, and the code follows the safety and style of `route_radix.cuh` /
`route_quant_fused.cuh`. Behaviour is unchanged throughout.

**Architecture:** The transport already receives the format only as runtime tables (`Segment{name, dst, src, bytes}`,
extents, slabs, `row_bytes`, the device copy table). So the EXL3 template argument is a thin trait, `Exl3RowLayout`:
its name, its ordered tensor names, and which of them are small enough for the copy wait's SM reads. The io_uring
calls already live in six places inside `RowReader`, and they become an `AsyncFileReader` type parameter
(`UringReader`, wrapped by `FaultyReader` for the test-only submit faults). The work runs in five phases, each
independently shippable:
1. **Relocations** (Tasks 1-3): pure moves into `expert_stream/`, each proven by a byte-level reconstruction script.
2. **Single wire layout** (Task 4): one header replaces four copies of the page and lease-block constants.
3. **Templates** (Tasks 5-7): the namespace rename, then `RowReader<Layout, Reader>`, `RamTier<Source>` and
   `RamThread<Tier>`.
4. **Safety at the boundaries** (Tasks 8-9): typed `__grid_constant__` kernel params, `TensorMatcher` /
   `RuntimeCheck` in every launcher and FFI entry.
5. **Python rename** (Tasks 10-11): generic ops modules, generic export names, and JIT modules keyed by layout.

**Tech Stack:** C++20 (g++ for the host module, nvcc for the device module), CUDA (SM120, RTX 5090), liburing, TVM FFI
(`TVM_FFI_DLL_EXPORT_TYPED_FUNC`, `load_jit`), Python 3 + PyTorch, pytest, `msgspec`.

**Spec:** No spec file. The spec is the read-only analysis from the 2026-09-26 session (the reply that begins "The main
finding: the format is already out of the C++"). Its binding points are restated under Design below. Also read
`analysis/dsv41-drive/LEASE_PROTOCOL.md` sections 4, 6-8 and 20; this plan must not change any contract written
there.

## Design (the spec, restated)

- **Transport vs format.** No C++ or CUDA file mentions trellis, suh/svh, mcg, bit widths or tile shapes. The only
  format facts the transport needs at compile time are:
  - the layout's name, which becomes the error prefix and today is the literal `exl3 RAM miss:`;
  - the ordered tensor names, whose count today is the magic 6 of `EXL3_STREAMED_NAMES`;
  - the small-tensor mask, which today is `not name.endswith("_trellis")` in `sm_copy_mask`.

  Per-layer byte sizes, the safetensors row schema and row-image layouts stay runtime data built in Python. Moving
  them into C++ would force one instantiation per bit width.
- **Layers.**
  - Wire layout: one `constexpr` header, host- and device-includable.
  - Device primitives.
  - Lease-protocol kernels: format-free.
  - Row-copy kernels (stream, copy wait): templated on the layout.
  - Host: row tables and pieces; `RowReader<Layout, Reader>`; pack pool; `RamTier<Source>`; `CopyEngine` (already
    generic, with virtual backends); `RamThread<Tier>`.
  - EXL3 instantiation files (`exl3_ram_miss_host.cpp`, `exl3_ram_miss.cuh`), which bind the templates and export.
- **Reader seam.** `AsyncFileReader` provides `init`, `ready`, `prep_read`, `prep_readv`, `submit`, `reap` and
  `drain`. `UringReader` keeps today's `io_uring_queue_init(depth, &ring, 0)` flags. **Do not** reuse `init_ring` from
  `csrc/io/uring_file_reader.cpp`: its `IORING_SETUP_SINGLE_ISSUER | DEFER_TASKRUN` breaks this reader. The ring is
  created on the Python thread (`RamTier::open`) and driven by the service thread and the fill thread.
- **Style targets.** Taken from `route_radix.cuh` and `route_quant_fused.cuh`:
  - a `struct XKernel { static void run(...) }` entry per launch;
  - `__grid_constant__` params built with designated initializers;
  - `TensorMatcher` / `SymbolicSize` / `RuntimeCheck` before launch;
  - `SGL_DEVICE`;
  - namespaces under `sglang::`.

## Global Constraints

- **Behaviour is frozen:**
  - every byte offset of the request page, lease block and prefetch page;
  - every kernel's memory ordering and control flow;
  - every env var name and default;
  - every error message's text.

  A task that changes one of these is wrong.
- **FFI export names and signatures** stay exactly as they are until Task 11. Python wrappers change only where a
  task says so.
- **Out of scope** (do not touch):
  - `expert_doorbell.cuh` / `expert_doorbell.py`;
  - `csrc/io/uring_file_reader.cpp`;
  - `srt/layers/engram_host_node.cpp`;
  - `Exl3RamMissService` beyond the lines Task 7 names;
  - every `SGLANG_*` env var (renames would need the `env-var-conventions` skill);
  - `LEASE_PROTOCOL.md` semantics (only its file paths change, in Task 11).
- **Base commit:** `db6c99db56` (master = origin/master on 2026-09-26). Every line number in this plan refers to it.
  Task 0 checks that nothing has moved since.
- **Commit classification** (`.claude/skills/mechanical-refactor-verify/guide-split.md` section 1.2): every commit
  message contains exactly one of `mechanical_provable` or `non_mechanical_provable`. Pure relocations are
  `mechanical_provable` and carry a proof. Everything else is `non_mechanical_provable`.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Repo rules:**
  - `.claude/rules/comment-style.md`: one or two lines, cross-boundary constraints only; Doxygen `\brief` on exported
    C++ entities.
  - `.claude/rules/no-dataclasses.md`: `msgspec.Struct` for new Python structs.
  - `.claude/rules/general-code-style.md`: keyword arguments.
  - `.claude/rules/unit-test-admission.md`: every new test case names the diff that would turn it red.
- **Running code** (`.claude/rules/divix01-run-protocol.md`): commit, push, and run in a private divix01 worktree.
  - Never rsync or scp a tree.
  - Set `PYTHONPATH=$PWD/python` and print `sglang.__file__` once per worktree.
  - Read `${PIPESTATUS[0]}` after any piped pytest.
  - GPU work (all suites below) runs under `cc-gpu.lock` on cores 32-63.
- **Suites**, defined once and referenced by name. Run them on divix01 from the worktree root, where
  `PY=/data/models/slang/.venv/bin/python` and `LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock`:

  ```bash
  # SUITE_UNIT: every registered test of this subsystem (CPU + GPU-capable), under the GPU lock
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock $LOCK taskset -c 32-63 $PY -m pytest -q -p no:randomly \
    test/registered/unit/kernels \
    test/registered/unit/layers/moe/test_exl3_ram_miss_service.py \
    test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py \
    test/registered/unit/layers/moe/test_exl3_ram_miss_tables.py \
    test/registered/unit/layers/moe/test_exl3_native_prefetch.py \
    test/registered/unit/layers/moe/test_exl3_expert_format.py \
    test/registered/unit/layers/moe/test_expert_host_tier.py \
    test/registered/unit/layers/moe/test_expert_stream.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"

  # SUITE_GPU: the manual device tests (real kernels, CUDA graphs)
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock $LOCK taskset -c 32-63 $PY -m pytest -q -p no:randomly \
    test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
    test/manual/dsv41/test_exl3_native_prefetch_cuda.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py \
    test/manual/dsv41/test_exl3_piece_stream_cuda.py test/manual/dsv41/test_exl3_task5_item4_gpu.py \
    2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
  ```

  A suite passes when `EXIT=0` and its pass/skip counts equal Task 0's baseline, plus exactly the tests the task adds.
  Record the counts and the command in the task's commit message body.

## Review Focus

The inputs and conditions below are implied by the design but exercised by no existing test. The task that owns each
one adds its test.

1. **A layout constant re-added to one source instead of the wire header.** It would compile, because an ambiguous
   name only errors where it is used, and then drift silently. Expected: a CPU test fails naming the file (Task 4).
2. **The "absent" sentinels passed to device launchers:** an empty `dst_slots`, `hot_slots` replaced by a dummy tensor,
   and a zero lease address. Expected: the new `TensorMatcher` checks accept every sentinel the Python wrapper sends
   today, and reject a wrong dtype (Task 8).
3. **Pinned host tensors** (`page`, `slot_map`, `hot_page`) handed to device launchers. Expected: accepted as
   `kDLCPU`/`kDLCUDAHost`; a CUDA tensor in their place is refused (Task 8).
4. **The Python name order and the C++ trait disagree** (someone reorders `EXL3_STREAMED_NAMES`). Expected: opening
   the host refuses with both orders in the message, instead of copying `suh` bytes into a `trellis` slab (Task 7).
5. **An SM mask naming a non-small tensor**, which would move a trellis off the DMA onto SM reads. Expected:
   `set_copy_table` refuses it (Task 7).

---

## File structure (end state)

Paths are relative to `python/sglang/kernels/jit/csrc/moe/`. "moved" means relocated verbatim in Tasks 2-3.

| File | Responsibility |
|---|---|
| `expert_stream/lease_layout.h` | **new (T4).** Request page, lease block and prefetch page constants: the one C++ home. Pure `constexpr`, no CUDA. |
| `expert_stream/row_layout.h` | **new (T7).** `ExpertRowLayout` concept, `kNumNames<L>`, `error_prefix<L>()`. |
| `expert_stream/lease_device.cuh` | moved (T3). Device constants (block sizes, state words) and helpers (acquire/release, deadlines, `lane_result_valid`). |
| `expert_stream/lease_kernels.cuh` | moved (T3). Post, wait, lease_wait, hit_wait, stream_hit_wait, rest_wait, stage_ack, finalize, ack. From T8 also `LeaseProtocolKernel` launchers. |
| `expert_stream/row_copy_kernels.cuh` | moved (T3). Stream kernel, copy wait. From T8 also `RowCopyKernel<L>` launchers. |
| `expert_stream/host/reader_base.h` | moved (T2). Reader constants, clocks, `StageRecord`. |
| `expert_stream/host/row_tables.h` | moved (T2). `Segment`, `Read`, `Tables`, `tables_from`, row-image checks. |
| `expert_stream/host/read_fault.h` | moved (T2). `ReadFault`, fault decoding. |
| `expert_stream/host/piece_geometry.h` | moved (T2). Sub-reads, pieces, readiness words. |
| `expert_stream/host/pack_pool.h` | `git mv` of `exl3_ram_miss_pack_pool.h` (T2). |
| `expert_stream/host/file_reader.h` | **new (T6).** `ReadCompletion`, `AsyncFileReader` concept. |
| `expert_stream/host/uring_reader.h` | **new (T6).** `UringReader`. |
| `expert_stream/host/faulty_reader.h` | **new (T6).** `SubmitFault`, `FaultyReader<Inner>`. |
| `expert_stream/host/row_reader.h` | moved (T2). `RowReader`, templated in T6/T7. |
| `expert_stream/host/tier_protocol.h` | moved (T2). Slot states, counters, request records, `StageRing`. |
| `expert_stream/host/copy_engine.h` | moved (T2). `CopyBackend`s, `CopyEngine`. |
| `expert_stream/host/ram_tier.h` | moved (T2). `Tier`, `RamTier`, templated in T6. |
| `expert_stream/host/ram_thread.h` | moved (T2). `RamThread`, templated in T6. |
| `exl3/exl3_row_layout.h` | **new (T7).** `sglang::exl3::Exl3RowLayout`. |
| `exl3_ram_miss_host.cpp` | Shrinks to the EXL3 host instantiation: aliases, handle registries, FFI exports. |
| `exl3_ram_miss.cuh` | Shrinks to the EXL3 device instantiation: includes plus (until T8) the launchers. |

Python:
- `python/sglang/test/expert_stream_sources.py` (**new**, T2): the lists of C++ sources that source-reading tests
  parse.
- `analysis/expert-stream-split/cxx_move_proof.py` (**new**, T1): the C++ relocation proof.
- T10 renames `kernels/ops/moe/exl3_lease_block.py` → `expert_lease_block.py` and `kernels/ops/moe/exl3_ram_miss.py`
  → `expert_stream_transport.py`.

---

### Task 0: Branch, worktree, baseline

**Files:** none.

- [ ] **Step 1: Create the branch and a laptop worktree** (superpowers:using-git-worktrees):

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4 worktree add -b expert-stream-transport \
  /home/dimitri/data/divix/sglang-nvfp4-worktrees/expert-stream-transport master
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/expert-stream-transport && git log -1 --oneline
```
Expected: `db6c99db56 dni`, or a later master. If it is later, run Step 2.

- [ ] **Step 2: Prove the line numbers still hold**

```bash
git diff --stat db6c99db56 HEAD -- python/sglang/kernels/jit/csrc/moe/ python/sglang/kernels/ops/moe/ \
  python/sglang/srt/layers/moe/exl3_ram_miss.py python/sglang/srt/layers/moe/exl3_expert_format.py test/
```
Expected: empty. If it is not empty, stop and re-derive every line range in Tasks 2, 3 and 4 from the anchors given
there before going on.

- [ ] **Step 3: Check the one branch that edited these files**

`origin/prefill-evict-3` is 10 commits ahead of master and touches `exl3_ram_miss*`. Its prefill-share test is
already on master, so it is believed to be superseded:
```bash
git log --oneline master..origin/prefill-evict-3 | head
git diff --stat master...origin/prefill-evict-3 -- python/sglang/kernels
```
If any of its C++ hunks are missing from master, ask the user whether it lands first. This plan relocates every line
it touches.

- [ ] **Step 4: Baseline on divix01**

```bash
git push origin expert-stream-transport
ssh divix01 'git -C /data/models/slang/sglang fetch origin && git -C /data/models/slang/sglang worktree add --detach \
  /data/models/slang/nvfp4-work/wt-ests origin/expert-stream-transport && cd /data/models/slang/nvfp4-work/wt-ests && \
  PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"'
```
Expected: the printed path lies under `/data/models/slang/nvfp4-work/wt-ests/python/`. Then run `SUITE_UNIT` and
`SUITE_GPU` there and write their pass/skip/fail counts into `/mnt/nvme1/expert-stream-split/baseline.txt`, together
with the commit. Every later task compares against these counts.

---

### Task 1: The C++ relocation proof

The repo's `mechanical-refactor-verify` primitives are Python-AST based and cannot certify a C++ move. This script
certifies one. It rebuilds each destination file from line ranges of the base files and requires an exact match,
ignoring only structural lines (blank lines, `#include`, `#pragma once`, namespace open/close, the `using` of
`TensorView`, and a new header's leading comment).

**Files:**
- Create: `analysis/expert-stream-split/cxx_move_proof.py`

**Interfaces:**
- Produces: `python3 analysis/expert-stream-split/cxx_move_proof.py MANIFEST.json`. It prints `PASS` and exits 0, or
  prints each mismatch and exits 1. Manifest format:
  `{"base": "<commit>", "sources": [path, ...], "files": {dst_path: [[src_path, first, last], ...]}}`, where lines are
  1-based and inclusive, and paths are repo-relative.
- Contract for new files: the authored header comment comes first and is followed by `#pragma once` before any moved
  line. Only the comment and blank lines before the first other line are exempt, so a moved block that opens with a
  comment is still compared.

- [ ] **Step 1: Write the script**

```python
"""Certify that a C++ commit is a pure relocation of line blocks (expert-stream split plan, Task 1).

Every non-structural line of every source at the base commit must appear in exactly one manifest block. Every
destination at HEAD, with structural lines removed, must equal its blocks concatenated in manifest order.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

STRUCTURAL = re.compile(
    r"^\s*($|#include\b|#pragma once\b|namespace [\w:]+ \{$|\}\s*// namespace\b|using tvm::ffi::TensorView;$)"
)


def _base_lines(base: str, path: str) -> list[str]:
    text = subprocess.run(["git", "show", f"{base}:{path}"], check=True, capture_output=True, text=True).stdout
    return text.split("\n")


def _body(lines: list[str], new_file: bool) -> list[str]:
    """Drop structural lines, and a new file's leading comment (authored). Only comment and blank lines before the
    first other line count as leading: a moved block that opens with a comment sits after `#pragma once`."""
    start = 0
    if new_file:
        while start < len(lines) and (lines[start].startswith("//") or not lines[start].strip()):
            start += 1
    return [line.rstrip() for line in lines[start:] if not STRUCTURAL.match(line)]


def main(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    base = manifest["base"]
    sources = {path: _base_lines(base, path) for path in manifest["sources"]}
    owner: dict[tuple[str, int], str] = {}
    errors = []
    for dst, blocks in manifest["files"].items():
        expected = []
        for src, first, last in blocks:
            for number in range(first, last + 1):
                key = (src, number)
                if key in owner:
                    errors.append(f"{src}:{number} is in two blocks ({owner[key]} and {dst})")
                owner[key] = dst
            expected += _body(sources[src][first - 1 : last], new_file=False)
        new_file = dst not in sources
        actual = _body(Path(dst).read_text().split("\n"), new_file=new_file)
        if actual != expected:
            for i, (a, e) in enumerate(zip(actual + [""] * len(expected), expected + [""] * len(actual))):
                if a != e:
                    errors.append(f"{dst}: body line {i + 1}: expected {e!r}, found {a!r}")
                    break
    for src, lines in sources.items():
        for number, line in enumerate(lines, start=1):
            if (src, number) not in owner and not STRUCTURAL.match(line):
                errors.append(f"{src}:{number} is in no block: {line.strip()[:80]!r}")
    for error in errors:
        print(error)
    print("PASS" if not errors else f"FAIL ({len(errors)})")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
```

- [ ] **Step 2: Show it can fail.** Write a one-file manifest that "moves" `exl3_ram_miss_pack_pool.h` onto itself.
  Run it (expect `PASS`), then change one character of a code line in the working-tree file and run it again (expect
  `FAIL`, naming the line). Restore the file with `git checkout -- <file>`.

```bash
cat > /tmp/claude-1000/pp.json <<'EOF'
{"base": "HEAD", "sources": ["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_pack_pool.h"],
 "files": {"python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_pack_pool.h":
   [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_pack_pool.h", 1, 309]]}}
EOF
python3 analysis/expert-stream-split/cxx_move_proof.py /tmp/claude-1000/pp.json
```
Use the session scratchpad for the manifest if `/tmp/claude-1000` is not yours.

- [ ] **Step 3: Commit**

```bash
git add analysis/expert-stream-split/cxx_move_proof.py
git commit -m "$(cat <<'EOF'
analysis(expert-stream): a line-block reconstruction proof for C++ relocations

non_mechanical_provable

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Split the host module into headers (pure move)

**Files:**
- Create: `python/sglang/test/expert_stream_sources.py`
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`,
  `test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py`
- Move: `exl3_ram_miss_host.cpp` lines → `expert_stream/host/*.h`; `git mv exl3_ram_miss_pack_pool.h
  expert_stream/host/pack_pool.h`
- Create: `analysis/expert-stream-split/manifest-host.json`

**Interfaces:**
- Produces: `sglang.test.expert_stream_sources.host_sources() -> tuple[Path, ...]`, `device_sources() -> tuple[Path,
  ...]`, `wire_header() -> Path`, `native_prefetch_source() -> Path`, `joined_text(paths) -> str`. Later tasks and
  tests read source text only through these.

- [ ] **Step 1 (prepare): Point the source-reading tests at a source list**

Create `python/sglang/test/expert_stream_sources.py`:
```python
"""The C++ sources of the expert-stream transport, for tests that read source text rather than run it."""

from pathlib import Path

MOE = Path(__file__).resolve().parents[1] / "kernels" / "jit" / "csrc" / "moe"


def host_sources() -> tuple[Path, ...]:
    """The EXL3 host instantiation first, then every transport host header."""
    return (MOE / "exl3_ram_miss_host.cpp", *sorted((MOE / "expert_stream" / "host").glob("*.h")))


def device_sources() -> tuple[Path, ...]:
    return (MOE / "exl3_ram_miss.cuh", *sorted((MOE / "expert_stream").glob("*.cuh")))


def wire_header() -> Path:
    return MOE / "expert_stream" / "lease_layout.h"


def native_prefetch_source() -> Path:
    return MOE / "exl3_native_prefetch.cuh"


def joined_text(paths) -> str:
    return "\n".join(path.read_text() for path in paths)
```

In `test_exl3_ram_miss_device_args.py`, change `_constants` to take several paths and an optional seed, parsing their
joined text once:
```python
def _constants(*paths: Path, known: dict[str, int] | None = None) -> dict[str, int]:
    """Every ``constexpr <type> kName = <integer expression>;`` of the given C++ sources, read as one text; a name
    defined twice fails. ``known`` seeds names defined elsewhere (the wire header)."""
    seeded = dict(known or {})
    found: dict[str, int] = {}
    pattern = r"^\s*(?:static\s+)?constexpr\s+[\w:]+\s+(k\w+)\s*=\s*([^;]+);"
    for name, expression in re.findall(pattern, joined_text(paths), re.MULTILINE):
        assert name not in found, f"{name} is defined twice: the layout check cannot tell which one applies"
        expression = re.sub(r"(?<=\d)[uU][lL]*\b", "", expression.strip())
        found[name] = seeded[name] = _evaluate(ast.parse(expression, mode="eval").body, seeded)
    return found
```

Replace every use of the old single-file form:
- `_constants(CSRC / "exl3_ram_miss.cuh")` → `_constants(*device_sources())`
- `_constants(CSRC / "exl3_ram_miss_host.cpp")` → `_constants(*host_sources())`
- The `files = {...}` dict in `test_the_device_kernels_speak_the_host_page_layout` and
  `test_hot_sidecar_layout_and_384_expert_size_match_the_native_abi` → `{"device": _constants(*device_sources()),
  "host": _constants(*host_sources())}`, with the `reference` key renamed to `"host"`.
- The `enum Counter` regex reads `joined_text(host_sources())` instead of `(CSRC / "exl3_ram_miss_host.cpp").read_text()`.
- `test_the_host_source_agrees_with_the_lease_layout_once_it_defines_it` passes `joined_text(host_sources())` and
  `_constants(*host_sources())`.
- Import: `from sglang.test.expert_stream_sources import device_sources, host_sources, joined_text`.

In `test_exl3_ram_miss_stage_trace_causal.py`, `test_no_clock_read_bypasses_the_trace_gate` reads
`joined_text(host_sources()).splitlines()` instead of `CPP.read_text().splitlines()`. Delete the `CPP` constant.

- [ ] **Step 2: Run the two changed files locally, then commit and push the prepare step**

```bash
python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_device_args.py \
  test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py -q -p no:randomly; echo "EXIT=$?"
```
Expected: the same pass/skip counts as before the edit. If the laptop lacks a dependency, run them on divix01 per
Task 0 Step 4. Commit as `test(expert-stream): read transport sources through one list` with
`non_mechanical_provable`.

- [ ] **Step 3 (move): Create the headers from these exact base ranges**

Each new header starts with a one-line `//` description, `#pragma once`, its includes, and `namespace sglang {
namespace exl3_ram_miss {`. It ends with the matching closers. Keep the namespace name `exl3_ram_miss` in this task;
Task 5 renames it.

| Destination (`expert_stream/host/`) | Base lines of `exl3_ram_miss_host.cpp` | Includes |
|---|---|---|
| `reader_base.h` | 55-282 (from `// A bounce bank holds` to `kStageTouch`) | base lines 9-46 verbatim, minus the pack-pool include |
| `row_tables.h` | 283-468 (`struct Segment` .. end of `tables_from`) | `"reader_base.h"` |
| `read_fault.h` | 469-580 (`// Test-only fault injection` .. `check_fault_words`) | `"row_tables.h"` |
| `piece_geometry.h` | 581-697 (`split_part` .. `PiecePublish`) | `"read_fault.h"` |
| `row_reader.h` | 698-2166 (the reader comment and `class RowReader`) | `"piece_geometry.h"`, `"pack_pool.h"` |
| `tier_protocol.h` | 2644-2968 (page and lease constants .. `StageRing`) | `"row_reader.h"` |
| `copy_engine.h` | 2969-3484 (`CopyLane` .. `CopyEngine`) | `"tier_protocol.h"` |
| `ram_tier.h` | 3485-5327 (`Tier` .. `find`) | `"copy_engine.h"` |
| `ram_thread.h` | 5760-5960 (`RamThread` .. `find_thread`) | `"ram_tier.h"` |

`exl3_ram_miss_host.cpp` keeps base lines 1-54, 2167-2643, 5328-5759 and 5961-6045. Replace its system includes and
the pack-pool include with the single line `#include "expert_stream/host/ram_thread.h"`.

`git mv python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_pack_pool.h
python/sglang/kernels/jit/csrc/moe/expert_stream/host/pack_pool.h`. Its content is unchanged.

The chain of includes is deliberate. It is the base file's own declaration order, so no definition can move ahead of
something it uses. Task 6 trims it.

- [ ] **Step 4: Write the manifest and run the proof**

`analysis/expert-stream-split/manifest-host.json`:
```json
{"base": "<the Task 2 prepare commit>",
 "sources": ["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp"],
 "files": {
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/reader_base.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 55, 282]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_tables.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 283, 468]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/read_fault.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 469, 580]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/piece_geometry.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 581, 697]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/row_reader.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 698, 2166]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 2644, 2968]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 2969, 3484]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 3485, 5327]],
  "python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h": [["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 5760, 5960]],
  "python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp": [
    ["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 1, 54],
    ["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 2167, 2643],
    ["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 5328, 5759],
    ["python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp", 5961, 6045]]}}
```

Run `python3 analysis/expert-stream-split/cxx_move_proof.py analysis/expert-stream-split/manifest-host.json`.
Expected: `PASS`. On `FAIL`, adjust a range boundary, never the code.

The `.cpp`'s leading comment is kept (it is in block 1-54), so its "new file" exemption does not apply. Edit that
comment only in Task 7.

- [ ] **Step 5: Format and compile on divix01**

Run `pre-commit run clang-format --files <the new headers and the .cpp>`. It must change nothing in moved lines; if it
does, the proof re-run shows it. Commit (`refactor(expert-stream): split the host module into headers`, with
`mechanical_provable` and the manifest path in the body), push, pull into `wt-ests`, and run `SUITE_UNIT`.
Expected: `EXIT=0`, with counts equal to the baseline.

---

### Task 3: Split the device header (pure move)

**Files:**
- Move: `exl3_ram_miss.cuh` lines → `expert_stream/lease_device.cuh`, `lease_kernels.cuh`, `row_copy_kernels.cuh`
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_lanes.py`
- Create: `analysis/expert-stream-split/manifest-device.json`

- [ ] **Step 1 (prepare):** In `test_exl3_ram_miss_stage_trace_lanes.py`, make
  `test_the_device_writes_the_lane_count_into_the_record_it_posts` read `joined_text(device_sources())` instead of
  `CUH.read_text()`, and delete `CUH`. Run the file and commit it (`non_mechanical_provable`).

- [ ] **Step 2 (move): Create the device headers from these base ranges of `exl3_ram_miss.cuh`**

| Destination (`expert_stream/`) | Base lines | Why together |
|---|---|---|
| `lease_device.cuh` | 41-352, 551-564, 717-848 | Constants and helpers. `lane_result_valid` (333), `publish_terminal` and `lease_hit_wait_body` are used by kernels in both kernel files. |
| `lease_kernels.cuh` | 354-550, 565-716, 849-1043, 1466-1575, 1732-1768 | post, wait, lease_wait, hit_wait, stream_hit_wait, rest_wait, stage_ack, finalize, ack |
| `row_copy_kernels.cuh` | 1044-1465, 1576-1731 | stream helpers and stream kernel; `copy_wait_read` and copy wait (which uses `stream_copy16`) |
| `exl3_ram_miss.cuh` (stays) | 1-40, 1770-2156 | the file comment, then the launchers |

The dependencies were checked on the base: nothing in `lease_kernels.cuh`'s ranges uses a name defined in 1044-1204.

Headers:
- `lease_device.cuh` gets base includes 28-37 and wraps its blocks in `namespace sglang { namespace
  exl3_ram_miss_device {`.
- The two kernel headers include `"lease_device.cuh"` and wrap in `namespace sglang {`.
- `exl3_ram_miss.cuh` replaces its includes with `#include "expert_stream/lease_kernels.cuh"` and `#include
  "expert_stream/row_copy_kernels.cuh"`.

- [ ] **Step 3: Manifest and proof.** `manifest-device.json` has the same shape as Task 2's, with the source
  `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` and the ranges above (each multi-range destination lists
  its blocks in table order). Expected: `PASS`.

- [ ] **Step 4: Commit and verify.** Commit `refactor(expert-stream): split the device header` with
  `mechanical_provable`, then run `SUITE_UNIT` and `SUITE_GPU` on divix01. Expected: `EXIT=0` both, with baseline
  counts. The `EXL3_RAM_MISS_TEST_*` test builds must still compile: `device_module_with_hooks` is exercised by the
  GPU suite.

---

### Task 4: One wire layout

Today the request page and lease block constants are written in `tier_protocol.h` (host), `lease_device.cuh` (device)
and `exl3_native_prefetch.cuh`, and checked against Python by `test_exl3_ram_miss_device_args.py`. After this task
they are written once in C++.

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_layout.h`
- Modify: `expert_stream/host/tier_protocol.h`, `expert_stream/lease_device.cuh`, `exl3_native_prefetch.cuh`
- Modify: `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`

**Interfaces:**
- Produces: `namespace sglang::expert_stream::wire`, holding every constant listed in Step 2, plus two new ones:
  `kPageBytes = kAdviseRing + kAdviseRecords * kRecordBytes` and `kLeaseBlockAlign = 4096`. Consumers add `using
  namespace ::sglang::expert_stream::wire;` inside their own namespace.

- [ ] **Step 1: Write the failing test (Review Focus 1)**

Replace `test_the_device_kernels_speak_the_host_page_layout`, `test_the_lease_block_layout_is_written_once_in_python_and_in_the_device_source`,
`test_the_host_source_agrees_with_the_lease_layout_once_it_defines_it` and `test_the_host_lease_guard_can_fail`
(together with `_check_host_lease_layout` and `_LEASE_CONCEPTS`) with the code below. The deleted guard existed to
catch lease code under non-`kLease` names in a file that defined no layout. With one header and the one-home test
below, no C++ source other than the header may define a layout name at all, which is strictly stronger.

```python
def _wire():
    return _constants(wire_header())


_NAME = re.compile(r"^\s*(?:static\s+)?constexpr\s+[\w:]+\s+(k\w+)\s*=", re.MULTILINE)


def test_the_wire_header_is_the_python_layout():
    """The page, lease block and prefetch page (LEASE_PROTOCOL.md section 4): one C++ home, equal to Python."""
    wire = _wire()
    python = {
        "kDemandHead": ram_miss.WORDS["demand_head"], "kDemandDone": ram_miss.WORDS["demand_done"],
        "kFatal": ram_miss.WORDS["fatal"], "kAdviseHead": ram_miss.WORDS["advise_head"],
        "kAdviseDone": ram_miss.WORDS["advise_done"], "kBusySeq": ram_miss.WORDS["busy_seq"],
        "kHeartbeat": ram_miss.WORDS["heartbeat"], "kRecordBytes": ram_miss.RECORD_BYTES,
        "kDemandRing": ram_miss.DEMAND_RING, "kDemandRecords": ram_miss.DEMAND_RECORDS,
        "kAdviseRing": ram_miss.ADVISE_RING, "kAdviseRecords": ram_miss.ADVISE_RECORDS, "kMaxIds": ram_miss.MAX_IDS,
        "kServed": ram_miss.STATUS["served"], "kPageBytes": PAGE_BYTES,
        "kHotHeaderBytes": ram_miss.HOT_HEADER_BYTES, "kHotAlignment": ram_miss.HOT_ALIGNMENT,
        "kHotRecords": ram_miss.HOT_RECORDS,
        "kPfReqGen": ram_miss.PREFETCH_FIELDS["req_gen"], "kPfReqRow": ram_miss.PREFETCH_FIELDS["req_row"],
        "kPfReqExpert": ram_miss.PREFETCH_FIELDS["req_expert"], "kPfReqDst": ram_miss.PREFETCH_FIELDS["req_dst"],
        "kPfDoneGen": ram_miss.PREFETCH_FIELDS["done_gen"], "kPfDoneReason": ram_miss.PREFETCH_FIELDS["done_reason"],
        "kPrefetchPageBytes": ram_miss.PREFETCH_PAGE_BYTES, "kPfTagRequest": ram_miss.PREFETCH_TAG_REQUEST,
        "kPfTagCopied": ram_miss.PREFETCH_TAG_COPIED, "kPfTagSkipped": ram_miss.PREFETCH_TAG_SKIPPED,
        "kPfSkipUnarmed": ram_miss.PREFETCH_SKIP_REASONS["unarmed"],
        "kPfSkipNotReady": ram_miss.PREFETCH_SKIP_REASONS["not_ready"],
        "kPfSkipInvalid": ram_miss.PREFETCH_SKIP_REASONS["invalid"],
        "kLeaseBlockAlign": exl3_lease_block.BLOCK_ALIGN,
        **_lease_python_constants(), **_lease_device_only_constants(),
    }
    assert {name: wire.get(name) for name in python} == python
    assert wire["kLeaseRing"] == wire["kDemandRecords"] and wire["kLeaseLanes"] == wire["kMaxIds"]


def test_no_other_source_defines_a_wire_constant():
    """A layout constant re-added beside its user compiles (an ambiguous name errors only where it is used) and then
    drifts; this names the file that re-added it."""
    wire = set(_wire())
    for path in (*host_sources(), *device_sources(), native_prefetch_source()):
        clash = wire & set(_NAME.findall(path.read_text()))
        assert not clash, f"{path.name} redefines wire constants {sorted(clash)}: define them only in lease_layout.h"
```

Keep `test_hot_sidecar_layout_and_384_expert_size_match_the_native_abi`, but replace its `files` loop with a single
check against `_wire()`. Keep the state-word and counter checks from the old page test as their own test,
`test_the_device_state_words_are_the_python_state_words`, which reads `_constants(*device_sources(), known=_wire())`.
Add the imports `from sglang.kernels.ops.moe import exl3_lease_block` and `wire_header, native_prefetch_source` to
the `expert_stream_sources` import.

- [ ] **Step 2: Run it to see it fail**

```bash
python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -q -p no:randomly -k "wire"; echo "EXIT=$?"
```
Expected: FAIL. `_wire()` raises `FileNotFoundError` for `lease_layout.h`.

- [ ] **Step 3: Create the header and delete the copies**

`expert_stream/lease_layout.h`:
```cpp
// Wire layout of the expert-stream request page, lease block and prefetch page (LEASE_PROTOCOL.md section 4).
// Mirrored by ops/moe/exl3_ram_miss.py and ops/moe/exl3_lease_block.py; test_exl3_ram_miss_device_args checks both.
// Only `constexpr <type> kName = <integer expression>;` lines: that test parses them.
#pragma once

#include <cstdint>

namespace sglang::expert_stream::wire {

// ... constants, moved as listed below ...

}  // namespace sglang::expert_stream::wire
```

Move into it, in this order, deleting each from its old home:
1. **Request page** from host `tier_protocol.h` (base host lines 2645-2677): `kDemandHead` .. `kFailed`, with their
   comments.
2. **Lease block** from host `tier_protocol.h` (base host lines 2679-2741), taking the device's comments where the
   two differ.
3. **The device-only tags and reasons** (base `.cuh` lines 136-153: `kLeaseTagDemand` .. `kLeaseReasonCount`).
   Delete the host's duplicate tag lines (base host lines 2748-2755).
4. **Prefetch page** from host `copy_engine.h` (base host lines 2993-3009: `kPfReqGen` .. `kPfSkipInvalid`, plus
   `kPrefetchPageBytes`).
5. **The two new constants:**
   ```cpp
   constexpr int64_t kPageBytes = kAdviseRing + kAdviseRecords * kRecordBytes;
   constexpr int64_t kLeaseBlockAlign = 4096;  // lease block, and each area offset inside it
   ```

Then:
- Delete the same names from `lease_device.cuh` (base `.cuh` lines 45-153, **except** `kBlock` and
  `kCopyWaitThreads`, which are launch configuration and stay).
- Delete `kFatal`, `kLeaseHeaderShutdown` and every `kPf*` line from `exl3_native_prefetch.cuh`.
- Keep in `tier_protocol.h` the `static_assert(kPieceTargets >= kLeaseLanes, ...)` line, which is host-only.
- Keep the two `static_assert`s that are pure layout facts in the header (`LaneRequest` order, "area D fits one
  page"), once each.
- Add `#include "../lease_layout.h"` plus `using namespace ::sglang::expert_stream::wire;` inside `namespace
  exl3_ram_miss` in `tier_protocol.h`. Add `#include "lease_layout.h"` and the same `using` inside `namespace
  exl3_ram_miss_device` in `lease_device.cuh`. Add `#include "expert_stream/lease_layout.h"` and the same `using`
  inside `namespace exl3_native_prefetch_device`.

- [ ] **Step 4: Run the test and the suites**

The local test expects PASS. On divix01, run `SUITE_UNIT` and `SUITE_GPU`: `EXIT=0`, baseline counts minus the four
deleted tests plus the three new ones. Then prove the guard can fail: add `constexpr int64_t kFatal = 8;` to
`row_copy_kernels.cuh`, run `-k no_other_source` and expect FAIL naming `row_copy_kernels.cuh`, then revert with
`git checkout --`.

- [ ] **Step 5: Commit** `refactor(expert-stream): one wire layout header for page, lease block and prefetch page`
  with `non_mechanical_provable`.

---

### Task 5: Generic namespaces and device style

**Files:** every file under `expert_stream/`, plus `exl3_ram_miss_host.cpp` and `exl3_ram_miss.cuh`.

- [ ] **Step 1: Rename the namespaces**

```bash
cd python/sglang/kernels/jit/csrc/moe
sed -i 's/namespace exl3_ram_miss_device\b/namespace device::expert_stream/; s/exl3_ram_miss_device::/device::expert_stream::/g; s/}  \/\/ namespace exl3_ram_miss_device/}  \/\/ namespace device::expert_stream/' \
  expert_stream/*.cuh exl3_ram_miss.cuh
sed -i 's/namespace exl3_ram_miss\b/namespace expert_stream/; s/}  \/\/ namespace exl3_ram_miss$/}  \/\/ namespace expert_stream/; s/exl3_ram_miss::/expert_stream::/g' \
  expert_stream/host/*.h exl3_ram_miss_host.cpp
sed -i 's/using namespace exl3_ram_miss_device;/using namespace device::expert_stream;/; s/using namespace exl3_ram_miss;/using namespace expert_stream;/' \
  expert_stream/*.cuh expert_stream/host/*.h exl3_ram_miss.cuh exl3_ram_miss_host.cpp
```
Also apply the second `sed` to `expert_stream/host/pack_pool.h`.

Check what is left:
```bash
grep -rn 'exl3_ram_miss_device\|namespace exl3_ram_miss\b\|exl3_ram_miss::' expert_stream exl3_ram_miss.cuh exl3_ram_miss_host.cpp
```
Expected: no output.

- [ ] **Step 2: Use the house device macro, and reuse the shared acquire load**

In `lease_device.cuh` and `row_copy_kernels.cuh`, replace `__device__ __forceinline__` with `SGL_DEVICE` (from
`<sgl_kernel/utils.cuh>`, already included). Make `ld_acquire_sys` delegate to the existing primitive, and add
`#include <sgl_kernel/distributed/ptx.cuh>`:
```cpp
SGL_DEVICE uint32_t ld_acquire_sys(const uint8_t* address) {
  return ::sglang::device::ptx::load_acquire_sys(reinterpret_cast<const uint32_t*>(address));
}
```
The 64-bit loads and the stores stay local, because `ptx.cuh` has no equivalents. The PTX instruction is the same
`ld.acquire.sys.global.u32`, so codegen is unchanged. Confirm this in Step 3.

- [ ] **Step 3: Verify the generated code did not change**

On divix01, compile the device module at the Task 4 commit and at this one, with `SGLANG_JIT_LOG_RESOURCE_USAGE=1`.
Diff the `ptxas` register/shared-memory lines per kernel. Expected: identical. Then run `SUITE_UNIT` and
`SUITE_GPU`, and expect baseline counts.

- [ ] **Step 4: Commit** `refactor(expert-stream): generic namespaces and SGL_DEVICE` with `non_mechanical_provable`.

---

### Task 6: The file reader as a template parameter

**Files:**
- Create: `expert_stream/host/file_reader.h`, `expert_stream/host/uring_reader.h`, `expert_stream/host/faulty_reader.h`
- Modify: `expert_stream/host/row_reader.h`, `ram_tier.h`, `ram_thread.h`, `exl3_ram_miss_host.cpp`

**Interfaces:**
- Produces:
  - `sglang::expert_stream::ReadCompletion{uint64_t data; int res;}`;
  - `concept AsyncFileReader<R>`;
  - `class UringReader`;
  - `struct SubmitFault{int error; int64_t call; bool submit_first; int64_t short_call;}`;
  - `template <AsyncFileReader Inner> class FaultyReader` with `set_submit_fault(const SubmitFault&)`;
  - `template <AsyncFileReader Reader> class RowReader`;
  - `template <class Source> class RamTier`;
  - `template <class Tier> class RamThread`.
- In `exl3_ram_miss_host.cpp`: `using Exl3Source = RowReader<FaultyReader<UringReader>>;`, `using Exl3Tier =
  RamTier<Exl3Source>;`, `using Exl3Thread = RamThread<Exl3Tier>;`.

- [ ] **Step 1: Write the concept and the io_uring reader**

`file_reader.h`:
```cpp
// The asynchronous positional reader a RowReader drives: prepare, submit, reap, and settle what is in flight.
#pragma once

#include <sys/uio.h>

#include <concepts>
#include <cstdint>
#include <vector>

namespace sglang::expert_stream {

// One reaped read: the tag it was prepared with (RowReader: generation << 32 | descriptor) and bytes or -errno.
struct ReadCompletion {
  uint64_t data;
  int res;
};

// One thread drives a reader at a time, but not always the thread that built it (the service opens on the Python
// thread and reads on its own), so an implementation must not bind itself to its creating thread.
// Buffers and iovec arrays passed to prep_* stay the caller's and must outlive the read's completion.
template <typename R>
concept AsyncFileReader =
    requires(R& r, const R& cr, int fd, void* buf, const iovec* iov, unsigned n, uint64_t off, uint64_t tag,
             std::vector<ReadCompletion>& out) {
      { r.init(n) } -> std::same_as<bool>;                         // n: reads prepared and not yet reaped, at most
      { cr.ready() } -> std::same_as<bool>;
      { r.prep_read(fd, buf, n, off, tag) } -> std::same_as<bool>;  // false: no room, reap first
      { r.prep_readv(fd, iov, n, off, tag) } -> std::same_as<bool>;
      { r.submit(n) } -> std::same_as<int>;                         // wait for n completions (0: none); <0 is -errno
      { r.reap(out) } -> std::same_as<unsigned>;                    // appends, never blocks
      { r.drain(n) } -> std::same_as<void>;                         // n prepared-or-in-flight reads: settle all
    };

}  // namespace sglang::expert_stream
```

`uring_reader.h`. The bodies are today's calls from `RowReader::open`/`refill`/`reap`/`submit`/`drain`, moved:
```cpp
// AsyncFileReader over one io_uring ring.
#pragma once

#include <liburing.h>

#include <cerrno>
#include <cstdio>
#include <exception>

#include "file_reader.h"

namespace sglang::expert_stream {

class UringReader {
 public:
  UringReader() = default;
  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;
  ~UringReader() {
    if (ready_) io_uring_queue_exit(&ring_);
  }

  // Flags 0, deliberately: SINGLE_ISSUER / DEFER_TASKRUN would bind the ring to the opening thread (see file_reader.h).
  bool init(unsigned depth) {
    depth_ = depth;
    ready_ = io_uring_queue_init(depth, &ring_, 0) == 0;
    return ready_;
  }
  bool ready() const { return ready_; }

  bool prep_read(int fd, void* buf, unsigned len, uint64_t off, uint64_t tag) {
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    io_uring_prep_read(sqe, fd, buf, len, off);
    io_uring_sqe_set_data64(sqe, tag);
    return true;
  }
  bool prep_readv(int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag) {
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    io_uring_prep_readv(sqe, fd, iov, count, off);
    io_uring_sqe_set_data64(sqe, tag);
    return true;
  }

  // (move here, verbatim, the comment block of RowReader::submit about a submit that consumes nothing)
  int submit(unsigned wait_nr) { return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_); }

  unsigned reap(std::vector<ReadCompletion>& out) {
    io_uring_cqe* cqe;
    unsigned head;
    unsigned seen = 0;
    io_uring_for_each_cqe(&ring_, head, cqe) {
      ++seen;
      out.push_back(ReadCompletion{io_uring_cqe_get_data64(cqe), cqe->res});
    }
    io_uring_cq_advance(&ring_, seen);
    return seen;
  }

  // (move here, verbatim, the comment and body of RowReader::drain, with ring_ready_ -> ready_ and
  // queue_depth() -> depth_)
  void drain(unsigned pending);

 private:
  io_uring ring_{};
  unsigned depth_ = 0;
  bool ready_ = false;
};

static_assert(AsyncFileReader<UringReader>);

}  // namespace sglang::expert_stream
```
Write `drain` out of line in the same header as `inline void UringReader::drain(unsigned pending) { ... }`, using the
moved body. Its `stderr` text `ERROR exl3 RAM miss: io_uring ring reset failed` stays byte-identical; Task 7 derives
the prefix.

`faulty_reader.h`:
```cpp
// Test-only submit faults (ReadFault's submit_* words) around any reader; zero faults forward every call unchanged.
#pragma once

#include "file_reader.h"

namespace sglang::expert_stream {

struct SubmitFault {
  int error = 0;              // errno the call-th submit returns (0: none)
  int64_t call = 0;           // 1-based count of submits over the reader's life
  bool submit_first = false;  // submit the prepared reads before failing (reads are then in flight)
  int64_t short_call = 0;     // that submit consumes nothing and reports success
};

template <AsyncFileReader Inner>
class FaultyReader {
 public:
  void set_submit_fault(const SubmitFault& fault) { fault_ = fault; }
  bool init(unsigned depth) { return inner_.init(depth); }
  bool ready() const { return inner_.ready(); }
  bool prep_read(int fd, void* buf, unsigned len, uint64_t off, uint64_t tag) {
    return inner_.prep_read(fd, buf, len, off, tag);
  }
  bool prep_readv(int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag) {
    return inner_.prep_readv(fd, iov, count, off, tag);
  }
  int submit(unsigned wait_nr) {
    ++submits_;
    if (fault_.error != 0 && submits_ == fault_.call) {
      if (fault_.submit_first) inner_.submit(0);
      return -fault_.error;
    }
    if (fault_.short_call != 0 && submits_ == fault_.short_call) return 0;
    return inner_.submit(wait_nr);
  }
  unsigned reap(std::vector<ReadCompletion>& out) { return inner_.reap(out); }
  void drain(unsigned pending) { inner_.drain(pending); }

 private:
  Inner inner_;
  SubmitFault fault_{};
  int64_t submits_ = 0;
};

}  // namespace sglang::expert_stream
```

- [ ] **Step 2: Make `RowReader` drive `Reader io_`**

In `row_reader.h`, `template <AsyncFileReader Reader> class RowReader`, with these edits and no others:
- Members: delete `io_uring ring_{}`, `bool ring_ready_`, `int64_t submits_`; add `Reader io_;`. Delete the private
  `struct Completion` and write `using Completion = ReadCompletion;`.
- Destructor: delete `if (ring_ready_) io_uring_queue_exit(&ring_);` (the member's destructor does it). Member order
  matters: `io_` must be declared **after** `pool_`, so the ring is torn down after the workers are joined, as today.
- `open()`: `if (io_uring_queue_init(queue_depth(), &ring_, 0) != 0) return false; ring_ready_ = true;` → `if
  (!io_.init(queue_depth())) return false;`.
- `set_piece_stream` and `read()`: `ring_ready_` → `io_.ready()`.
- `set_fault`: add `if constexpr (requires { io_.set_submit_fault(SubmitFault{}); }) io_.set_submit_fault(SubmitFault{fault.submit_error, fault.submit_call, fault.submit_first, fault.submit_short_call});`.
- `refill()`: the loop head becomes
  ```cpp
  while (c.queue_count > 0 && c.pending < c.capacity) {
    const uint32_t index = queue_[c.queue_head];
    ExtentDesc& d = descs_[index];
    const int64_t remaining = d.read->length - d.done;
    const uint64_t tag = (static_cast<uint64_t>(d.generation) << 32) | index;
    const uint64_t offset = static_cast<uint64_t>(d.read->offset + d.done);
    bool prepared_one;
    if (t_.images) {
      // (keep the existing two-line comment here)
      iovec* iov = &iovecs_[static_cast<size_t>(index) * t_.segments.size()];
      const unsigned count =
          image_iovecs(static_cast<size_t>(d.slot), d.read->dest + d.done, d.read->dest + d.read->length, iov);
      prepared_one = io_.prep_readv(fds_[d.read->file], iov, count, offset, tag);
    } else {
      prepared_one = io_.prep_read(
          fds_[d.read->file], bounce_slot(static_cast<size_t>(d.slot)) + d.read->dest + d.done,
          static_cast<unsigned>(remaining), offset, tag);
    }
    if (!prepared_one) break;
    c.queue_head = (c.queue_head + 1) % queue_.size();
    --c.queue_count;
    ++c.pending;
    // (the sqe_log_ and trace blocks follow unchanged)
  ```
  Today the SQE is taken before the queue is popped and the loop breaks on a null SQE, so the queue is untouched on a
  full SQ either way. The rebuilt iovecs of a refused read are rebuilt again on its next attempt.
- `reap()`: replace the `io_uring_cqe*` / `for_each` / `cq_advance` block with `completions_.clear(); again_.clear();
  const unsigned seen = io_.reap(completions_);`. `c.pending -= seen;` and everything after stay.
- `submit()`: its body becomes `return io_.submit(wait_nr);`. Its fault lines and comment moved in Step 1.
- `drain()`: its body becomes `io_.drain(pending);`.
- `#include "uring_reader.h"` and `"faulty_reader.h"` are **not** included here: `row_reader.h` includes only
  `"file_reader.h"`. The instantiation file includes the concrete readers.

Also add, above the class, the static-member definitions that a class template needs. Today's `static constexpr`
members are implicitly inline in C++17+, so none are expected; if the compiler reports a missing definition, add it
there.

- [ ] **Step 3: Template the tier, the thread and the registries**

- `ram_tier.h`: `template <class Source> class RamTier`. The member `RowReader reader_;` becomes `Source reader_;`.
  Delete `registry_mutex`, `registry` and `find` from this header.
- `ram_thread.h`: `template <class Tier> class RamThread`, with `std::shared_ptr<RamTier>` → `std::shared_ptr<Tier>`.
  Delete `thread_registry` and `find_thread`.
- Inside the class bodies, qualify any call the compiler reports as depending on the template (`this->` is not
  needed: neither class has a dependent base). Enum and struct names from `tier_protocol.h` are non-dependent and
  unchanged.
- `exl3_ram_miss_host.cpp`: after the includes, add
  ```cpp
  #include "expert_stream/host/faulty_reader.h"
  #include "expert_stream/host/uring_reader.h"

  namespace sglang {
  namespace {
  using namespace expert_stream;
  using Exl3Source = RowReader<FaultyReader<UringReader>>;
  using Exl3Tier = RamTier<Exl3Source>;
  using Exl3Thread = RamThread<Exl3Tier>;
  // (move registry_mutex/registry/find and thread_registry/find_thread here, verbatim, with RamTier -> Exl3Tier and
  // RamThread -> Exl3Thread; an anonymous namespace so a second layout's module can never resolve to these)
  }  // namespace
  ```
  In every export, replace `RowReader reader(` with `Exl3Source reader(`, `std::make_shared<RamTier>` with
  `std::make_shared<Exl3Tier>`, and `std::make_shared<RamThread>` with `std::make_shared<Exl3Thread>`.

- [ ] **Step 4: Verify.** `SUITE_UNIT` on divix01 must match the baseline counts. The tests that pin this task are the
  `submit_error`/`submit_first` rows of `test_exl3_ram_miss_split.py` (lines 130-134), the `submit_short_call` tests
  in `test_exl3_ram_miss_row_images.py` and `_split.py`, `read_rows_sqes` (the SQE log order), and
  `test_exl3_ram_miss_thread.py` (the ring is opened on one thread and read on another). If any of them changes
  outcome, the decorator or the refill order is wrong. Do not edit the test.

- [ ] **Step 5: Commit** `refactor(expert-stream): the file reader is a template parameter (UringReader)` with
  `non_mechanical_provable`.

---

### Task 7: The EXL3 row layout as a template parameter

**Files:**
- Create: `expert_stream/row_layout.h`, `exl3/exl3_row_layout.h`
- Modify:
  - `expert_stream/host/row_tables.h`, `row_reader.h`, `ram_tier.h`, `copy_engine.h`, `uring_reader.h`;
  - `exl3_ram_miss_host.cpp`;
  - `python/sglang/kernels/ops/moe/exl3_ram_miss.py`;
  - `python/sglang/srt/layers/moe/exl3_ram_miss.py`.
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py`,
  `test/registered/unit/kernels/test_exl3_ram_miss_tier.py`.

**Interfaces:**
- Consumes: Task 6's `RowReader<Reader>` and `RamTier<Source>`.
- Produces:
  - `concept ExpertRowLayout<L>` (`L::kName`, `L::kNames`, `L::kSmallMask`);
  - `kNumNames<L>`;
  - `std::string error_prefix<L>()`, which returns `std::string(L::kName) + " RAM miss: "` so every EXL3 message
    stays byte-identical;
  - `RowReader<Layout, Reader>`, with `using Layout = ...;` exposed as a member type;
  - `tables_from<Layout>(...)`;
  - exports `exl3_ram_miss_layout_names() -> std::string` (newline-joined) and `exl3_ram_miss_layout_small_mask() ->
    int64_t`;
  - Python `Exl3RamMissHost.layout_names: tuple[str, ...]`, `Exl3RamMissHost.small_mask: int` and
    `ops.moe.exl3_ram_miss.host_layout() -> tuple[tuple[str, ...], int]`.

- [ ] **Step 1: Write the failing tests (Review Focus 4 and 5)**

In `test_exl3_ram_miss_copy_engine.py`:
```python
def test_the_python_names_are_the_host_modules_layout():
    """EXL3_STREAMED_NAMES orders the slab table and the copy table; the C++ trait orders the SM mask. A reorder on
    one side would SM-copy a trellis and DMA a scale vector."""
    from sglang.kernels.ops.moe.exl3_ram_miss import host_layout
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES

    names, small_mask = host_layout()
    assert names == EXL3_STREAMED_NAMES
    assert small_mask == 0b110110


def test_a_copy_table_sm_mask_naming_a_trellis_is_refused(tmp_path):
    """set_copy_table refuses an SM mask outside the layout's small tensors: SM-reading a 13 MB trellis in the copy
    wait would stall the chain instead of using the DMA engine."""
    s, _page, host, _sim = _host(tmp_path)
    host.enable_copy_engine(-1, spin_us=200)
    table = torch.zeros((6, 3), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="exl3 RAM miss: .*small"):
        host.set_copy_table(ROW, table, DST_ROWS, sm_mask=0b000001)
```
`_host` (line 25), `ROW` and `DST_ROWS` are the file's existing helpers. Also make the existing
`test_the_sm_mask_leaves_exactly_the_trellis_tensors_on_the_copy_engine` keep passing unchanged:
`sm_copy_mask(EXL3_STREAMED_NAMES) == 0b110110`.

In `test_exl3_ram_miss_tier.py`, add:
```python
def test_a_slab_table_narrower_than_the_layout_is_refused(tmp_path):
    """tables_from indexes slabs[row][name] for every layout name; a 5-wide table would read past each row."""
    s = ram_miss_setup(tmp_path)
    narrow = dataclasses.replace(s.tables, slabs=s.tables.slabs[:, :5].contiguous(), row_bytes=s.tables.row_bytes[:5])
    with pytest.raises(RuntimeError, match="exl3 RAM miss: .*6 names"):
        exl3_ram_miss.read_rows_once(narrow, row=0, experts=[0], slots=[0], direct=False)
```
The file already imports `ram_miss_setup` and `exl3_ram_miss`; add `import dataclasses`. `Exl3RamMissTables` is a
frozen dataclass, so `dataclasses.replace` builds the narrow copy. `read_rows_once(tables, row, experts, slots, *,
direct, step=BOUNCE_ROWS)` is at `ops/moe/exl3_ram_miss.py:74`.

- [ ] **Step 2: Run them to see them fail**

On divix01: `-k "layout or trellis_is_refused or narrower"`. Expected: the first fails on the `host_layout` import,
the second on `DID NOT RAISE`, the third on `DID NOT RAISE` or a crash-free wrong read.

- [ ] **Step 3: Write the trait and the concept**

`exl3/exl3_row_layout.h`:
```cpp
// The EXL3 streamed expert row as the expert-stream transport sees it: six tensors in EXL3_STREAMED_NAMES order
// (srt/layers/moe/exl3_expert_format.py); test_exl3_ram_miss_copy_engine checks the two agree.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace sglang::exl3 {

struct Exl3RowLayout {
  static constexpr std::string_view kName = "exl3";
  static constexpr std::array<std::string_view, 6> kNames = {
      "w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"};
  // suh/svh: 44.5 KB of a 13.3 MB DSV4.1 row, read by the copy wait's SMs rather than the DMA engine.
  static constexpr uint32_t kSmallMask = 0b110110;
};

}  // namespace sglang::exl3
```

`expert_stream/row_layout.h`:
```cpp
// What the transport needs to know about a streamed row's format at compile time; everything else is a runtime table.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace sglang::expert_stream {

template <typename L>
concept ExpertRowLayout = requires {
  { L::kName } -> std::convertible_to<std::string_view>;
  { L::kSmallMask } -> std::convertible_to<uint32_t>;
  L::kNames.size();
} && (L::kNames.size() >= 1) && (L::kNames.size() <= 32) && ((uint64_t{L::kSmallMask} >> L::kNames.size()) == 0);

template <ExpertRowLayout L>
inline constexpr int64_t kNumNames = static_cast<int64_t>(L::kNames.size());

// "<name> RAM miss: ": the prefix every service error carries, so EXL3's messages are unchanged.
template <ExpertRowLayout L>
std::string error_prefix() {
  return std::string(L::kName) + " RAM miss: ";
}

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: Thread the layout through**

- `row_tables.h`: `template <ExpertRowLayout Layout> Tables tables_from(...)`. At its top:
  ```cpp
  const std::string prefix = error_prefix<Layout>();
  if (slabs.size(1) != kNumNames<Layout> || row_bytes.size(0) != kNumNames<Layout>) {
    throw std::runtime_error(prefix + "slabs and row_bytes must have " + std::to_string(kNumNames<Layout>) +
                             " names (the layout's), got " + std::to_string(slabs.size(1)) + " and " +
                             std::to_string(row_bytes.size(0)));
  }
  ```
  After the segment loop, add a check that each `segment.name` lies in `[0, kNumNames<Layout>)`, using the same
  prefix. `check_image_tables` becomes `template <ExpertRowLayout Layout>`. Replace each literal `"exl3 RAM miss: "`
  in both functions with `prefix +` (or `error_prefix<Layout>() +`).
- `row_reader.h`: `template <ExpertRowLayout Layout, AsyncFileReader Reader> class RowReader` with `using LayoutType =
  Layout;` (named `LayoutType` so it cannot shadow the template parameter), and the same literal-to-prefix
  replacement.
- `ram_tier.h`: inside `RamTier<Source>`, `using Layout = typename Source::LayoutType;`, and the same replacement. In
  `set_copy_table(row, entries, dst_rows, sm_mask)`, first:
  ```cpp
  if ((static_cast<uint64_t>(sm_mask) & ~static_cast<uint64_t>(Layout::kSmallMask)) != 0) {
    throw std::runtime_error(error_prefix<Layout>() + "sm_mask names a tensor that is not one of the layout's small ones");
  }
  ```
- `copy_engine.h` and `uring_reader.h` are not templated on the layout. Give `CopyEngine` a `std::string prefix`
  constructor argument (the tier passes `error_prefix<Layout>()`) and use it for its messages. `UringReader::drain`'s
  `stderr` line changes to the fixed text `"ERROR expert stream: io_uring ring reset failed\n"`. That is the one
  message this task changes; it goes to `stderr` only, and `grep -rn "ring reset failed" test/` must be empty (check
  it).
- Guard. The following must print nothing:
  ```bash
  grep -rn 'exl3' python/sglang/kernels/jit/csrc/moe/expert_stream/
  ```
  Comments that point at EXL3 files as the example instantiation are the only acceptable hits. Reword them to name
  `exl3_ram_miss_host.cpp` as "the EXL3 instantiation" and re-run.
- `exl3_ram_miss_host.cpp`: `#include "exl3/exl3_row_layout.h"`, then `using Exl3Source = RowReader<exl3::Exl3RowLayout,
  FaultyReader<UringReader>>;`. `tables_from(` calls become `tables_from<exl3::Exl3RowLayout>(`. Add:
  ```cpp
  /// \brief The layout this module was built for: its tensor names in copy-table order, newline-joined.
  std::string exl3_ram_miss_layout_names() {
    std::string out;
    for (const auto name : exl3::Exl3RowLayout::kNames) out += std::string(name) + "\n";
    out.pop_back();
    return out;
  }
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_layout_names, exl3_ram_miss_layout_names);

  /// \brief Bit i: name i may be read by the copy wait's SMs (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES).
  int64_t exl3_ram_miss_layout_small_mask() { return exl3::Exl3RowLayout::kSmallMask; }
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_layout_small_mask, exl3_ram_miss_layout_small_mask);
  ```
  Update the file's opening comment to one line: "The EXL3 instantiation of the expert-stream host transport: type
  aliases, handle registries and the FFI exports."

- [ ] **Step 5: Python reads the layout from the module**

`ops/moe/exl3_ram_miss.py`:
```python
@cache_once
def host_layout() -> tuple[tuple[str, ...], int]:
    """The host module's row layout: its tensor names in copy-table order and the SM-readable ones as a bit mask."""
    module = _host_module()
    return tuple(str(module.exl3_ram_miss_layout_names()).split("\n")), int(module.exl3_ram_miss_layout_small_mask())
```

`srt/layers/moe/exl3_ram_miss.py`, `sm_copy_mask`:
```python
def sm_copy_mask(names: Sequence[str]) -> int:
    """The copy-table entries the copy wait reads itself (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES): the host
    layout's small tensors, wherever they sit in ``names``."""
    layout_names, small = host_layout()
    small_names = {name for i, name in enumerate(layout_names) if small >> i & 1}
    return sum(1 << i for i, name in enumerate(names) if name in small_names)
```
It imports `host_layout` from `sglang.kernels.ops.moe.exl3_ram_miss`.

In `Exl3RamMissService.ensure_started`, right after `tables = exl3_ram_miss_tables(...)`, refuse a name-order mismatch
before opening the host:
```python
layout_names, _ = host_layout()
if layout_names != EXL3_STREAMED_NAMES:
    raise RuntimeError(f"exl3 RAM miss: the host module's layout {layout_names} is not EXL3_STREAMED_NAMES {EXL3_STREAMED_NAMES}")
```

- [ ] **Step 6: Check fixtures narrower than six names**

```bash
grep -rn "slabs" test/registered/unit/kernels test/manual/dsv41 | grep -v "EXL3_STREAMED_NAMES\|s\.slabs\|tables\.slabs" | head -30
```
Every hit that builds a C++ `Tables` with fewer than six names now fails by design. Pure-Python hits such as
`stream_segment_map`'s `SimpleNamespace` are unaffected. If a C++ one exists, widen it to the six names, and say so in
the commit body.

- [ ] **Step 7: Run the three new tests, then the suites.** On divix01, the new tests expect PASS. `SUITE_UNIT` and
  `SUITE_GPU` expect `EXIT=0` and baseline counts plus the three new tests.

- [ ] **Step 8: Commit** `refactor(expert-stream): the row layout is a template parameter (Exl3RowLayout)` with
  `non_mechanical_provable`.

---

### Task 8: Typed kernel params and checked device launchers

**Files:**
- Modify: `expert_stream/lease_kernels.cuh`, `expert_stream/row_copy_kernels.cuh`, `exl3_ram_miss.cuh`
- Modify: `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (the `cuda_wrappers` strings and the `hot_slots` dummy)
- Test: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`

**Interfaces:**
- Produces, in `namespace sglang`:
  - `struct LeaseProtocolKernel` with static `post`, `wait`, `lease_wait`, `lease_ack`, `lease_hit_wait`,
    `lease_rest_wait`, `lease_stage_ack`, `lease_finalize`, `lease_stream_hit_wait`;
  - `template <expert_stream::ExpertRowLayout L> struct RowCopyKernel` with static `lease_stream` and
    `lease_copy_wait`.

  The static methods have exactly today's FFI parameter lists, so Python call sites do not change.

- [ ] **Step 1: Write the failing test (Review Focus 2 and 3)**

Append to `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`:
```python
def test_the_launchers_refuse_a_wrong_dtype_and_accept_every_sentinel_the_wrapper_sends():
    """The launchers check what they cast: a state tensor of int64 would have its words read as halves of int32.
    The sentinels the wrapper sends for 'absent' (empty dst_slots, the no-hot-slots tensor, a zero lease address)
    and the pinned host page must still pass."""
    import sglang.kernels.ops.moe.exl3_ram_miss as ram_miss

    page = ram_miss.new_page(pin=True)
    slot_map = torch.full((1, 16), -1, dtype=torch.int32).pin_memory()
    dev = ram_miss.Exl3RamMissDevice(page, slot_map, device="cuda", layers=1, timeout_ms=5, advise=False)
    planned = torch.zeros(8, dtype=torch.int64, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    routes = torch.zeros(8, dtype=torch.int64, device="cuda")
    dev.post(0, planned, count, routes, next_row=-1)  # hot_slots, dst_slots absent; no lease block
    torch.cuda.synchronize()
    bad_state = torch.zeros(len(ram_miss.STATE_WORDS), dtype=torch.int64, device="cuda")
    with pytest.raises(Exception, match="state"):
        dev._kernels().exl3_ram_miss_post(
            page, bad_state, slot_map, planned, count, routes, 0, 0, dev.last_routes, -1, 0, 0,
            dev.timeout_ns, 0, 0, dev._no_hot_slots, 0, dev.state[:0], 0,
        )
    with pytest.raises(Exception, match="page"):
        dev._kernels().exl3_ram_miss_post(
            page.cuda(), dev.state, slot_map, planned, count, routes, 0, 0, dev.last_routes, -1, 0, 0,
            dev.timeout_ns, 0, 0, dev._no_hot_slots, 0, dev.state[:0], 0,
        )
```
`Exl3RamMissDevice.post(row, planned, count, routes, next_row, hot_slots=None, hot_capacity=0, dst_slots=None,
copy_engine=False)` is at `ops/moe/exl3_ram_miss.py:1348`. The direct calls repeat the positional order the wrapper
uses at line 1365.

- [ ] **Step 2: Run it to see it fail.** Run it on divix01 under the GPU lock. Expected: FAIL on the missing
  `_no_hot_slots` attribute. With that attribute in place, the `DID NOT RAISE` failure shows that nothing checks
  today.

- [ ] **Step 3: Give every kernel a params struct**

The rule, applied to all 11 kernels:
- The struct's fields are the kernel's current parameter list: same order, names and types, with `__restrict__`
  dropped. The lists are recorded below from the base.
- The kernel takes `const __grid_constant__ XParams p`.
- The kernel's first lines rebind each field to a local of the old name, restoring `__restrict__` on pointers, so
  the body below stays byte-identical.

For example, the ack kernel:
```cpp
struct LeaseAckParams {
  uint8_t* page;
  uint8_t* lease;
  int64_t lease_d;
  const int32_t* go_count;
  const int64_t* lane_ctx;
  float* keep;
};

__global__ __launch_bounds__(device::expert_stream::kLeaseLanes, 1) void exl3_ram_miss_lease_ack_kernel(
    const __grid_constant__ LeaseAckParams p) {
  uint8_t* __restrict__ const page = p.page;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  const int32_t* __restrict__ const go_count = p.go_count;
  const int64_t* __restrict__ const lane_ctx = p.lane_ctx;
  float* __restrict__ const keep = p.keep;
  // (unchanged body)
```

The structs and their fields, one per kernel:

| Struct | Fields (in order) |
|---|---|
| `PostParams` | `page, state, slot_map, planned, count, routes, route_count, row, experts, advise, last_routes, next_row, lease, lease_d, timeout_ns, hot_page, hot_stride, hot_slots, hot_capacity, dst_slots, dst_count, copy_engine` |
| `WaitParams` | `page, state, slot_map, planned, count, row, experts, lanes, host_rows, keep, ram_miss, timeout_ns` |
| `LeaseWaitParams` | `page, state, planned, count, row, lanes, host_rows, keep, ram_miss, timeout_ns, lease, lease_d, go_count, lane_ctx` |
| `HitWaitParams` | `page, state, planned, count, dst_slots, row, lanes, host_rows_1, dst_slots_1, lease, lease_d, go_1, lane_ctx_1, origin_1, claimed, violated, budget_ns` |
| `StreamHitWaitParams` | `HitWaitParams`'s fields, then `go_2, stream_count, stream_abort` |
| `RestWaitParams` | `page, state, planned, count, dst_slots, row, lanes, host_rows_2, dst_slots_2, ram_miss, lease, lease_d, claimed, go_2, lane_ctx_2, origin_2` |
| `StreamParams` | `page, state, planned, count, dst_slots, row, lanes, experts, host_rows_2, dst_slots_2, ram_miss, lease, lease_d, lease_p, claimed, go_2, lane_ctx_2, origin_2, stream_count, stream_abort, segments, segment_count, segment_map, row_segments, piece_runs, fault` |
| `StageAckParams` | `page, lease, lease_d, go_count, lane_ctx, origin, violated` |
| `FinalizeParams` | `page, state, count, go_1, go_2, go_ce, violated, keep, lease, lease_d` (keep the kernel's comment about the missing `lanes` above the struct) |
| `CopyWaitParams` | `page, state, count, lease, lease_c, lease_d, sm_table, sm_count, go_ce` |
| `LeaseAckParams` | as shown above |

Give `StreamHitWaitParams` its fields written out in full, not by inheritance, so `__grid_constant__` stays a plain
aggregate. The field types are the parameter types from the base signatures (`lease_hit_wait` at `.cuh:850`, and so
on).

- [ ] **Step 4: Move the launchers into checked kernel structs**

Move each free launcher from `exl3_ram_miss.cuh` (base lines 1770-2156) into a static method. Protocol kernels go in
`struct LeaseProtocolKernel` in `lease_kernels.cuh`; stream and copy wait go in `template <ExpertRowLayout L> struct
RowCopyKernel` in `row_copy_kernels.cuh`, which includes `"row_layout.h"`. Each method:

1. Declares the matchers:
   - `auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();` for device tensors;
   - `auto host = SymbolicDevice{}; host.set_options<kDLCPU, kDLCUDAHost>();` for pinned tensors.
2. Verifies every tensor argument with `TensorMatcher`, per the table below.
3. Checks every address/offset argument with `RuntimeCheck`: `lease_address` is 0 or a multiple of
   `kLeaseBlockAlign`; `lease_d`, `lease_p` and `lease_c` are multiples of `kLeaseBlockAlign` whenever
   `lease_address != 0`.
4. Builds the params with designated initializers (`.page = ..., .lease_d = lease_d, .lease_p = lease_p, ...`) and
   launches with `host::LaunchKernel(grid, block, device.unwrap())(kernel, params)`. The launch shapes are today's.

| Tensor | Matcher |
|---|---|
| `page` | `TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device(host)` |
| `state` | `TensorMatcher({kStateWords}).with_dtype<int32_t>().with_device(device)` |
| `slot_map` | `TensorMatcher({L_, E_}).with_dtype<int32_t>().with_device(host)` (`L_`, `E_` symbolic; `E_` feeds `experts`) |
| `planned` | `TensorMatcher({P_}).with_dtype<int64_t>().with_device(device)`; for `wait`/`lease_wait`/`hit_wait`/`rest_wait`/`stream`, `RuntimeCheck(P_ >= lanes)` |
| `count`, `go_*`, `violated`, `stream_count`, `stream_abort` | `TensorMatcher({1}).with_dtype<int32_t>().with_device(device)` |
| `routes` | `TensorMatcher({R_}).with_dtype<int64_t>().with_device(device)` |
| `last_routes` | `TensorMatcher({L_, kMaxIds}).with_dtype<int32_t>().with_device(device)` |
| `hot_slots` | `TensorMatcher({-1}).with_dtype<int64_t>().with_device(device)`; when `hot_address != 0`, `RuntimeCheck(size >= hot_capacity)` |
| `dst_slots`, `dst_slots_*`, `origin_*`, `claimed` | `TensorMatcher({-1}).with_dtype<int32_t>().with_device(device)` (empty allowed) |
| `host_rows*`, `ram_miss` | `TensorMatcher({-1}).with_dtype<int64_t>().with_device(device)` |
| `keep` | `TensorMatcher({-1}).with_dtype<float>().with_device(device)` |
| `lane_ctx*` | `TensorMatcher({-1, 4}).with_dtype<int64_t>().with_device(device)` |
| `segments` (stream) | `TensorMatcher({kNumNames<L>, 3}).with_dtype<int64_t>().with_device(device)` |
| `segment_map` | `TensorMatcher({-1}).with_dtype<int32_t>().with_device(device)`; `RuntimeCheck(size == row_segments + kNumNames<L>)` |
| `piece_runs` | `TensorMatcher({-1, -1, kRowPieces, row_segments, 2}).with_dtype<int32_t>().with_device(device)` |
| `fault` | `TensorMatcher({kStreamFaultWords}).with_dtype<int32_t>().with_device(device)` |
| `sm_table` (copy wait, passed as an address today) | `RuntimeCheck(sm_count <= std::popcount(L::kSmallMask))`, and a nonzero address when `sm_count > 0` |

- Put the `-1` wildcard extents where the wrapper legitimately varies (read `Exl3RamMissDevice.__init__` allocations,
  `ops/moe/exl3_ram_miss.py:1234-1329`).
- Name each matcher's error by its tensor. The `TensorMatcher` messages already carry the dimension; prefix each
  `RuntimeCheck` message with the tensor name.
- `exl3_ram_miss.cuh` shrinks to its comment, the two includes, and `#include "exl3/exl3_row_layout.h"`.

- [ ] **Step 5: Point the Python wrappers at the structs, and give `hot_slots` a real "absent" tensor**

In `ops/moe/exl3_ram_miss.py`:
```python
_LEASE_METHODS = {
    "exl3_ram_miss_post": "post", "exl3_ram_miss_wait": "wait", "exl3_ram_miss_lease_wait": "lease_wait",
    "exl3_ram_miss_lease_ack": "lease_ack", "exl3_ram_miss_lease_hit_wait": "lease_hit_wait",
    "exl3_ram_miss_lease_rest_wait": "lease_rest_wait", "exl3_ram_miss_lease_stage_ack": "lease_stage_ack",
    "exl3_ram_miss_lease_finalize": "lease_finalize", "exl3_ram_miss_lease_stream_hit_wait": "lease_stream_hit_wait",
}
_ROW_COPY_METHODS = {"exl3_ram_miss_lease_stream": "lease_stream", "exl3_ram_miss_lease_copy_wait": "lease_copy_wait"}
_LAYOUT = "sglang::exl3::Exl3RowLayout"


def _device_wrappers() -> list[tuple[str, str]]:
    return [(name, f"LeaseProtocolKernel::{method}") for name, method in _LEASE_METHODS.items()] + [
        (name, f"RowCopyKernel<{_LAYOUT}>::{method}") for name, method in _ROW_COPY_METHODS.items()
    ]
```
- `_device_module` and `device_module_with_hooks` pass `cuda_wrappers=_device_wrappers()`.
- `_DEVICE_KERNELS` is deleted; update any reference with `grep -rn _DEVICE_KERNELS`.
- In `Exl3RamMissDevice.__init__`, add `self._no_hot_slots = torch.empty(0, dtype=torch.int64, device=device)`. It
  is stable for graph capture. In `post`, pass `hot_slots if hot_slots is not None else self._no_hot_slots` instead
  of `self.state`, which was int32 and would now fail the int64 matcher.

- [ ] **Step 6: Verify.** On divix01, run the new test (expect PASS), then `SUITE_UNIT` and `SUITE_GPU` (expect
  `EXIT=0`, baseline counts plus one). `test_exl3_ram_miss_graph_gpu.py` is the capture-and-replay check: params are
  captured by value, and the pointers inside them are the same stable tensors as before.

- [ ] **Step 7: Commit** `refactor(expert-stream): typed __grid_constant__ params and checked launchers` with
  `non_mechanical_provable`.

---

### Task 9: Checked host FFI boundary

**Files:**
- Modify: `exl3_ram_miss_host.cpp` (the entries that take tensors: `open`, `read_rows*`, `set_copy_table`,
  `victim_census`, `slot_info`, `mapping`, `slot_to_expert`, `lru_order`, `set_hot`, `counters`, `layer_rows`,
  `trace_drain`, `enable_native_prefetch`, `prefetch_lease`, `inject_fault`, `piece_geometry`, `piece_runs`)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_tier.py`

- [ ] **Step 1: Write the failing test**

```python
def test_open_refuses_an_extent_table_of_the_wrong_dtype(tmp_path):
    """open() reads extents as int64 [L, E, parts, 4] through a raw pointer; int32 would be read as packed pairs and
    name files and offsets that were never written."""
    s = ram_miss_setup(tmp_path)
    bad = dataclasses.replace(s.tables, extents=s.tables.extents.to(torch.int32))
    with pytest.raises(Exception, match="extents"):
        exl3_ram_miss.read_rows_once(bad, row=0, experts=[0], slots=[0], direct=False)
```

- [ ] **Step 2: Run it to see it fail.** Expected: `DID NOT RAISE`, or a crash. If it crashes the interpreter, run
  it under `pytest --forked` for the red run only.

- [ ] **Step 3: Add the matchers**

Add a helper in `exl3_ram_miss_host.cpp` and call it first in `exl3_ram_miss_open` and every `exl3_ram_miss_read_rows*`:
```cpp
// The table tensors every reader entry takes, checked once: tables_from reads them through raw pointers.
void check_table_tensors(TensorView extents, TensorView starts, TensorView file_sizes, TensorView segments,
                         TensorView slabs, TensorView row_bytes) {
  using namespace host;
  auto L_ = SymbolicSize{"layers"}, E_ = SymbolicSize{"experts"}, P_ = SymbolicSize{"parts"};
  auto cpu = SymbolicDevice{};
  cpu.set_options<kDLCPU>();
  TensorMatcher({L_, E_, P_, 4}).with_dtype<int64_t>().with_device(cpu).verify(extents);
  TensorMatcher({L_, E_}).with_dtype<int64_t>().with_device(cpu).verify(starts);
  TensorMatcher({-1}).with_dtype<int64_t>().with_device(cpu).verify(file_sizes);
  TensorMatcher({-1, 4}).with_dtype<int64_t>().with_device(cpu).verify(segments);
  TensorMatcher({L_, kNumNames<exl3::Exl3RowLayout>}).with_dtype<int64_t>().with_device(cpu).verify(slabs);
  TensorMatcher({kNumNames<exl3::Exl3RowLayout>}).with_dtype<int64_t>().with_device(cpu).verify(row_bytes);
}
```
Add `#include <sgl_kernel/tensor.h>`. `uring_file_reader.cpp` shows it builds in a host-only module. For the other
entries, verify each tensor argument's dtype and rank from its `static_cast<...*>(x.data_ptr())` and the
`.size(...)` reads in its body. Each tensor gets one `TensorMatcher` line; do not add checks for tensors an entry
does not dereference.

`TensorMatcher` failures throw `host::PanicError`, which the FFI surfaces as a Python exception. Check that no
existing `pytest.raises(RuntimeError, match=...)` in `SUITE_UNIT` now meets a different exception type. The Task 7
narrow-slab test fires in `tables_from` first only if these checks run after it, so call `check_table_tensors`
**before** `tables_from` and change that test's `match` to `"slabs"`.

- [ ] **Step 4: Verify.** The new test expects PASS. `SUITE_UNIT` expects `EXIT=0` and baseline counts plus four new
  tests.

- [ ] **Step 5: Commit** `refactor(expert-stream): check every tensor at the host FFI boundary` with
  `non_mechanical_provable`.

---

### Task 10: Generic Python ops modules

**Files:**
- Move: `python/sglang/kernels/ops/moe/exl3_lease_block.py` → `expert_lease_block.py`;
  `python/sglang/kernels/ops/moe/exl3_ram_miss.py` → `expert_stream_transport.py`
- Modify: every importer. There are 64 files at base; list them with
  `grep -rln 'ops.moe.exl3_ram_miss\|ops.moe import exl3_ram_miss\|exl3_lease_block' python test benchmark* analysis scripts`.
- Rename: `Exl3RamMissHost` → `ExpertStreamHost`, `Exl3RamMissDevice` → `ExpertStreamDevice`.

- [ ] **Step 1 (move, `mechanical_provable`):** `git mv` both modules and repoint every import, changing nothing else.
  Produce the proof with the skill (`/mechanical-refactor-verify construct <commit>`). If the generator reports
  `UNSUPPORTED` for a whole-module rename, write a `Repro` per `guide-construct-proof.md` section 2. The expected
  verdict is `PASS`. Also update `python/sglang/test/expert_stream_sources.py`, which has no import of these modules;
  confirm with grep.

- [ ] **Step 2 (rename, `non_mechanical_provable`):** Rename the two classes everywhere:
  ```bash
  grep -rl 'Exl3RamMissHost\|Exl3RamMissDevice' python test benchmark* analysis scripts | \
    xargs sed -i 's/\bExl3RamMissHost\b/ExpertStreamHost/g; s/\bExl3RamMissDevice\b/ExpertStreamDevice/g'
  grep -rn 'Exl3RamMissHost\|Exl3RamMissDevice' python test benchmark* analysis scripts
  ```
  The second grep expects no output. `Exl3RamMissService`, `Exl3RamMissTables` and `Exl3RamMissRowBackend` keep their
  names: they are the EXL3 service, which stays EXL3.

- [ ] **Step 3: Verify.** `SUITE_UNIT` and `SUITE_GPU` expect baseline counts.

- [ ] **Step 4: Commit** the two commits above separately, in order.

---

### Task 11: Generic export names and layout-keyed modules

**Files:**
- Modify: `exl3_ram_miss_host.cpp` (every `TVM_FFI_DLL_EXPORT_TYPED_FUNC` name), `ops/moe/expert_stream_transport.py`,
  and every direct `module.exl3_ram_miss_*` caller (`grep -rn '\.exl3_ram_miss_' python test`)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (file paths only)

**Interfaces:**
- Produces:
  - FFI names `expert_stream_*` (was `exl3_ram_miss_*`, suffixes unchanged);
  - `ops.moe.expert_stream_transport.LAYOUTS: dict[str, TransportBuild]`, where `TransportBuild` is a frozen
    `msgspec.Struct` of `host_source: str`, `device_layout: str`;
  - `ExpertStreamHost(..., layout: str = "exl3")`, `ExpertStreamDevice(..., layout: str = "exl3")`;
  - `host_layout(layout: str = "exl3")`.

- [ ] **Step 1: Rename the exports**

In `exl3_ram_miss_host.cpp`, rename each exported C++ function and its export macro from `exl3_ram_miss_<x>` to
`expert_stream_<x>`. On the device side, rename the export names in `_LEASE_METHODS`/`_ROW_COPY_METHODS` the same way
(the struct method names are already generic). Then:
```bash
grep -rl '\.exl3_ram_miss_' python test | xargs sed -i 's/\.exl3_ram_miss_/.expert_stream_/g'
grep -rn '\.exl3_ram_miss_' python test
```
The second grep expects no output.

- [ ] **Step 2: Key modules by layout**

```python
class TransportBuild(msgspec.Struct, frozen=True):
    """One instantiation of the transport: the host translation unit that binds its C++ layout and file reader, and
    the device layout type RowCopyKernel is instantiated with."""

    host_source: str
    device_layout: str


# One row per format the transport is built for; adding a format adds a row and its instantiation files.
LAYOUTS = {"exl3": TransportBuild(host_source="moe/exl3_ram_miss_host.cpp", device_layout="sglang::exl3::Exl3RowLayout")}


@cache_once
def _host_module(layout: str = "exl3") -> Module:
    return load_jit(f"expert_stream_host_{layout}", cpp_files=[LAYOUTS[layout].host_source],
                    extra_ldflags=["-luring", "-lpthread", "-ldl"], header_only=False)
```
- Do the same for `_device_module(layout)` and `device_module_with_hooks(defines, layout="exl3")`, with module names
  `expert_stream_{layout}` and `cuda_files=["moe/exl3_ram_miss.cuh"]`.
- `_device_wrappers(layout)` uses `LAYOUTS[layout].device_layout` in place of `_LAYOUT`.
- `ExpertStreamHost` and `ExpertStreamDevice` take `layout` by keyword and pass it through.
- `host_layout(layout)` does the same.

The JIT module names change, so the first load compiles cold. That is expected once per machine.

- [ ] **Step 3: Update `LEASE_PROTOCOL.md`'s file references**

```bash
grep -n 'exl3_ram_miss\(_host\)\?\.\(cuh\|cpp\)\|exl3_lease_block\|ops/moe/exl3_ram_miss' analysis/dsv41-drive/LEASE_PROTOCOL.md
```
Point each hit at its new file: layout → `expert_stream/lease_layout.h`; kernels → `expert_stream/lease_kernels.cuh`
or `row_copy_kernels.cuh`; service → `expert_stream/host/ram_tier.h`; Python → `expert_lease_block.py` /
`expert_stream_transport.py`. Do not edit anything else in the document.

- [ ] **Step 4: Verify.** `SUITE_UNIT` and `SUITE_GPU` expect baseline counts plus the tests added in Tasks 4, 7, 8
  and 9.

- [ ] **Step 5: Commit** `refactor(expert-stream): generic export names; JIT modules keyed by layout` with
  `non_mechanical_provable`.

---

### Task 12: Whole-branch verification

**Files:** none.

- [ ] **Step 1: Chain proof.** Run the skill's chain verifier over the branch:
  `/mechanical-refactor-verify verify --base db6c99db56 --branch expert-stream-transport --proof <folder>`.
  - The folder holds the Task 10 Python proof.
  - The Task 2 and 3 C++ proofs are re-run by hand: `python3 analysis/expert-stream-split/cxx_move_proof.py` on each
    manifest, checked out at its move commit.

  Expected: every `mechanical_provable` commit passes.

- [ ] **Step 2: A decode arm on each side.** Following `benchmarks/dsv41_baseline/README.md`, register both trees and
  run one untraced arm each with `benchmarks/dsv41_baseline/run_arm.sh`:
  - `EXPECT_SHA=db6c99db56`;
  - `EXPECT_SHA=<branch tip>`.

  Use a port other than 7867, and take the GPU lock via `run_arm.sh` itself. Expected: ms/token within the arms'
  usual session-to-session spread, and no new fail-closed counters (`timeouts`, `failures`, `sticky`) in either
  run's stats. A difference outside the spread is a blocker: bisect it by task.

- [ ] **Step 3: Final review.** Dispatch a reviewer (superpowers:requesting-code-review) on `db6c99db56..HEAD`. Point
  it at Global Constraints and Review Focus, and at the Task 6 decorator and Task 8 checks as the highest-risk diffs.

---

## Self-review notes

- **Coverage of the spec:**

  | Spec point | Task |
  |---|---|
  | Generic transport separated from EXL3 | 2, 3, 5 |
  | EXL3 detail passed as a template argument | 7 (host), 8 (device) |
  | File reader templated, with an io_uring reader type passed in | 6 |
  | Lease protocol and Python wrappers generic | 4, 10, 11 |
  | Safety and style of `route_radix.cuh` / `route_quant_fused.cuh` | 5, 8, 9 |

- **Deliberately not done:**
  - Converting the host's existing `throw std::runtime_error` sites to `RuntimeCheck`. It would change exception
    types that tests match, for no safety gain on already-validated paths.
  - A second reader implementation such as `PreadReader`.
  - Generalizing `Exl3RamMissService` to other formats.
  - Unifying `expert_doorbell`.

  Each is a separate plan if wanted.
- **Names across tasks:**
  - `host_sources` / `device_sources` / `wire_header` / `native_prefetch_source` / `joined_text`: Task 2 onward.
  - `Exl3Source` / `Exl3Tier` / `Exl3Thread`: Task 6, extended in 7.
  - `LayoutType`: Task 7, read by `RamTier`.
  - `LeaseProtocolKernel` / `RowCopyKernel<L>`: Task 8, renamed exports in 11.
  - `_no_hot_slots`: Task 8, used by the Task 8 test.
