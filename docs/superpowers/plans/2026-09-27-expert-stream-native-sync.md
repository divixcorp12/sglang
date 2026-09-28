# Expert-stream native sync primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-written PTX and `volatile` accesses in the three expert_stream device headers with CUDA
intrinsics and libcu++ (`__ldcv`/`__stcg`, `cuda::ptx::get_sreg_globaltimer`, `cuda::atomic_thread_fence`) and
with named relaxed/acquire/release helpers. The protocol does not change; the machine code changes only in the
ways each task names, and a SASS gate checks that. (`cuda::atomic_ref` was tried in Task 2 and rejected on its
SASS; see Task 2.)

**Architecture:** There are three commits, each with its own proof, so a reviewer can reject one and keep the others.
1. **Intrinsics (Task 1).** Replacements that produce identical SASS. Gate: SASS identical per function.
2. **Named helpers (Task 2, revised).** Every raw `volatile` cast and the stray `ld.relaxed.gpu` asm become calls
   to named helpers in `lease_device.cuh`: `ld/st_relaxed_sys<T>` (a volatile access), the existing
   `ld_acquire_sys{,64}`/`st_release_sys{,64}`, and `ld_relaxed_gpu`. Gate: SASS identical per function.
3. **Fences (Task 3).** A deliberate weakening: the five seqlock fences go from `__threadfence_system()`
   (`MEMBAR.SC.SYS`) to acquire/release fences (`MEMBAR.ALL.SYS`). Each has a host-side partner of the matching
   Boehm-seqlock shape, and the task records the argument site by site. The gate allows only that substitution.

**Tech Stack:** CUDA 13.4 (`/usr/local/cuda-13.4`, the toolkit the sglang JIT resolves on divix01), target
`sm_120f` (RTX 5090), libcu++/CCCL headers shipped with that toolkit, tvm-ffi JIT (`load_jit`), pytest. All builds and
tests run on divix01.

**Spec:** No spec file. The spec is the user's 2026-09-27 request: move the expert_stream kernels off hand-rolled
sync primitives onto native CUDA/libcu++ APIs. The user chose the "mechanical cleanup" first, `cuda::atomic_ref`
over typed PTX wrappers, and the fence weakening as a separate final task. After `atomic_ref` failed Task 2's gate,
the user chose to revert it and use named wrappers instead (Task 2, revised). The probe results that shaped the plan
are under "Evidence" below. Also read `.claude/rules/divix01-run-protocol.md` (how code reaches divix01, the lock
rules, `PIPESTATUS`) and `analysis/dsv41-drive/LEASE_PROTOCOL.md` sections 6.2-6.3 and 11.4 (the seqlocks).

## Global Constraints

- No protocol change: no wire-layout edit (`lease_layout.h` untouched), no host-side edit (`expert_stream/host/**`
  untouched), no Python edit other than the new test file.
- No toolchain upgrade. CUDA 13.4 on divix01 already provides every API used here.
- Scope is exactly `python/sglang/kernels/jit/csrc/moe/expert_stream/{lease_device,lease_kernels,row_copy_kernels}.cuh`.
  `expert_cache_transfer.cuh` is out of scope: its asm carries different clobbers from the intrinsics
  (`st.global.cg.u32` with no `"memory"`), so a swap there is not a mechanical proof.
- `sgl_kernel/distributed/ptx.cuh` is not edited (communicator code uses it). This plan only stops using it here.
- Memory scopes stay as they are. A `volatile` access is relaxed at system scope under the PTX memory model, and
  the relaxed helpers keep it a volatile access, so nothing narrows, even for device-only `state[]` words.
- No `cuda::atomic_ref` (Task 2's first attempt, reverted). With CUDA 13.4's libcu++, an atomic access through a
  `__grid_constant__` parameter gains a run-time local-pointer check (`QSPC`) and a byte-copy fallback, and
  adjacent relaxed accesses are merged and reordered.
- `__threadfence()` (device scope, used by the stream kernel's block counting) and `atomicCAS`/`atomicAdd` are
  already native intrinsics and stay unchanged.
- The SmAck fence (`row_copy_kernels.cuh:593`) stays `__threadfence_system()`. It orders other threads' loads,
  through `__syncthreads()`, before thread 0's release, which is not the seqlock shape Task 3 argues.
- Every divix01 command follows the run protocol: code arrives by `git push` then `git fetch` into a private
  worktree; `PYTHONPATH=$PWD/python`; CPU jobs under `taskset -c 0-63`; GPU jobs under
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`; suite status read from
  `${PIPESTATUS[0]}`, never from the pipeline.
- Record the exact command next to every suite count you quote.

## Review Focus

1. **A helper that is not what it wraps.** If a named helper were not inlined, or its body differed from the cast
   it replaced (a lost `volatile`, a changed width), the compiler could hoist or merge accesses. Expected: no
   instruction changes. Pinned by the Task 2 gate (`--exact`) and its mutant (a helper with its `volatile`
   removed must fail the gate).
2. **`cuda::atomic_ref` coming back.** Expected: none in these headers. Its local-pointer check and merging are
   invisible in review and visible only in SASS. Pinned by the Task 2 source tests (`volatile` lives only in the
   helpers; inline PTX only in `lease_device.cuh`) and by this plan's Task 2 note in `LEASE_PROTOCOL.md` 6.2.
3. **Building the wrong tree (the interpreter trap).** Expected: every SASS dump comes from the worktree under
   test. Pinned by the gate's build step, which refuses to write a dump unless `sglang.__file__` lies inside the
   worktree it was given.
4. **The test-hook build (`device_module_with_hooks`, `EXL3_RAM_MISS_TEST_*`).** It compiles `#ifdef` blocks the
   production gate never builds. Expected: they still compile and pass. Pinned by the Task 1-3 suite runs, which
   include `test_exl3_ram_miss_piece_stream.py` and `test_exl3_ram_miss_copy_engine.py` (the hook builds).
5. **Captured decode graph.** Expected: the chain still captures and replays. Pinned by the Task 4 run of
   `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`.

## Evidence (probes run 2026-09-27 on divix01, CUDA 13.4, `-arch=sm_120f`)

Probe sources are in `divix01:/data/models/slang/nvfp4-work/scratch-atomic-probe/probe{2,3,4}.cu`.

| Form | SASS |
|---|---|
| asm `ld.global.cv.v2.b64` + `st.global.cg.v2.b64` vs `__stcg(__ldcv(longlong2*))` | identical: `LDG.E.128.STRONG.SYS` / `STG.E.128.STRONG.GPU` |
| asm `ld.global.cv.u8` vs `__ldcv(const unsigned char*)` | identical: `LDG.E.U8.STRONG.SYS` |
| asm `mov.u64 %globaltimer` vs `cuda::ptx::get_sreg_globaltimer()` | identical: `CS2R SR_GLOBALTIMERLO` |
| asm `ld.acquire.sys.global.u32/u64` vs `atomic_ref<..., system>.load(acquire)` | `LDG.E[.64].STRONG.SYS` → `LD.E[.64].STRONG.SYS`, same `CCTL.IVALL` |
| `volatile int32_t` load / store vs `atomic_ref` relaxed | `LDG/STG.E.STRONG.SYS` → `LD/ST.E.STRONG.SYS` |
| asm `ld.relaxed.gpu.global.s32` vs `atomic_ref<int, device>.load(relaxed)` | `LDG.E.STRONG.GPU` → `LD.E.STRONG.GPU` |
| `volatile uint16_t` store vs `atomic_ref<uint16_t>` relaxed store | `STG.E.U16.STRONG.SYS` → `ST.E.U16.STRONG.SYS` |
| `volatile uint8_t` store vs `atomic_ref<uint8_t>` relaxed store | `STG.E.U8.STRONG.SYS` → **`LD` + `ATOM.E.CAS.STRONG.SYS` loop** |
| `__threadfence()` vs `atomic_thread_fence(seq_cst, device)` | identical: `MEMBAR.SC.GPU` |
| `__threadfence_system()` vs `atomic_thread_fence(acquire or release, system)` | `MEMBAR.SC.SYS` → `MEMBAR.ALL.SYS` |

`sm_120` behaves differently from `sm_90` here: nvcc 12.4 for `sm_90` lowers both fences to `MEMBAR.ALL.SYS`, and
it rejects `atomic_ref<uint16_t>`. Only divix01's toolchain counts.

The `atomic_ref` rows above come from kernels with plain pointer parameters. Through a `const __grid_constant__`
parameter, as every kernel in this module takes, the same accesses also gain a run-time local-pointer check
(`QSPC.E.L`) and a byte-copy fallback (`probe5.cu`, Task 2's first gate). That is why Task 2 does not use
`atomic_ref`.

---

### Task 0: Worktrees, the SASS gate, and the baselines

**Files:**
- Create (divix01, not committed): `/data/models/slang/nvfp4-work/scratch-atomic-probe/sass_gate.py`
- No repository file changes.

**Interfaces:**
- Produces: `sass_gate.py build <worktree> <out.sass>` and
  `sass_gate.py diff <before.sass> <after.sass> [--exact] [--allow OLD=NEW ...]` (exit 0 = gate passes).
- Produces: `/data/models/slang/nvfp4-work/scratch-atomic-probe/base.sass` and the baseline suite counts, used by
  every later task.

- [ ] **Step 1: Create the branch on the laptop**

Use superpowers:using-git-worktrees to create a worktree for branch `expert-stream-native-sync` from `master` at
`4d8a9444cd`. Push it so divix01 can fetch it:

```bash
git push -u origin expert-stream-native-sync
```

- [ ] **Step 2: Create the two divix01 worktrees**

```bash
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-native-sync origin/expert-stream-native-sync \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-native-sync-base 4d8a9444cd \
  && git -C /data/models/slang/nvfp4-work/wt-native-sync log -1 --oneline \
  && git -C /data/models/slang/nvfp4-work/wt-native-sync-base log -1 --oneline'
```

Expected: both print `4d8a9444cd merge(expert-stream): follow-ups ...`.

- [ ] **Step 3: Write the gate tool on divix01**

Write this file to `/data/models/slang/nvfp4-work/scratch-atomic-probe/sass_gate.py`:

```python
"""SASS gate for the expert_stream device module (sm_120f).

  build <worktree> <out.sass>          build the module from <worktree> into a fresh JIT cache, dump its SASS
  diff <before> <after> [--exact] [--allow OLD=NEW ...]
      --exact   every function's full instruction list must match (operands included, addresses stripped)
      default   every function's sequence of memory/fence/barrier opcodes must match, except position-for-position
                substitutions an --allow rule names (OLD is an opcode prefix replaced by NEW)
"""
import argparse
import collections
import difflib
import os
import pathlib
import re
import subprocess
import sys
import tempfile

SCRATCH = "/data/models/slang/nvfp4-work/scratch-atomic-probe"
PYTHON = "/data/models/slang/.venv/bin/python"
CUOBJDUMP = "/usr/local/cuda-13.4/bin/cuobjdump"
FUNC = re.compile(r"Function : (\S+)")
INSTR = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
PREDICATE = re.compile(r"^@!?\w+\s+")
MEMORY = re.compile(r"^(LD|ST|ATOM|RED|MEMBAR|CCTL|FENCE|BAR|ERRBAR)")
CONSTANT_LOADS = re.compile(r"^(LDC|ULDC)")

BUILD = """
import sys, sglang
worktree = sys.argv[1]
if not sglang.__file__.startswith(worktree + "/"):
    sys.exit(f"sglang imported from {sglang.__file__}, not from {worktree}")
from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.moe import expert_stream_transport as transport
with override_jit_cuda_arch(12, 0):
    transport._device_module("exl3")
"""


def build(worktree: str, out: str) -> None:
    worktree = os.path.realpath(worktree)
    cache = tempfile.mkdtemp(prefix="jit-", dir=SCRATCH)
    env = dict(os.environ, PYTHONPATH=f"{worktree}/python", SGLANG_JIT_CACHE_DIR=cache, CUDA_VISIBLE_DEVICES="")
    subprocess.run(["taskset", "-c", "0-63", PYTHON, "-c", BUILD, worktree], env=env, check=True)
    modules = sorted(pathlib.Path(cache).rglob("*.so"))
    if len(modules) != 1:
        sys.exit(f"expected exactly one built module under {cache}, found {modules}")
    sass = subprocess.run([CUOBJDUMP, "-sass", str(modules[0])], check=True, capture_output=True, text=True).stdout
    pathlib.Path(out).write_text(sass)
    print(f"{out}: {len(parse(sass))} functions from {modules[0]}")


def parse(sass: str) -> dict[str, list[str]]:
    functions: dict[str, list[str]] = {}
    name = None
    for line in sass.splitlines():
        if match := FUNC.search(line):
            name = match.group(1)
            functions[name] = []
        elif name is not None and (match := INSTR.match(line)):
            functions[name].append(PREDICATE.sub("", match.group(1)))
    return functions


def memory_opcodes(instructions: list[str]) -> list[str]:
    opcodes = (instruction.split()[0] for instruction in instructions)
    return [op for op in opcodes if MEMORY.match(op) and not CONSTANT_LOADS.match(op)]


def diff(before: str, after: str, exact: bool, allow: list[tuple[str, str]]) -> int:
    a = parse(pathlib.Path(before).read_text())
    b = parse(pathlib.Path(after).read_text())
    failures: list[str] = []
    used: collections.Counter[str] = collections.Counter()
    if a.keys() != b.keys():
        failures.append(f"function sets differ: {sorted(a.keys() ^ b.keys())}")
    for name in sorted(a.keys() & b.keys()):
        if exact:
            if a[name] != b[name]:
                body = "\n".join(difflib.unified_diff(a[name], b[name], lineterm="", n=2))
                failures.append(f"{name}: SASS differs\n{body}")
            continue
        ma, mb = memory_opcodes(a[name]), memory_opcodes(b[name])
        matcher = difflib.SequenceMatcher(a=ma, b=mb, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace" and i2 - i1 == j2 - j1:
                for old_op, new_op in zip(ma[i1:i2], mb[j1:j2]):
                    if any(old_op.startswith(o) and n + old_op[len(o):] == new_op for o, n in allow):
                        used[f"{old_op} -> {new_op}"] += 1
                    else:
                        failures.append(f"{name}: {old_op} -> {new_op} is not an allowed substitution")
            else:
                failures.append(f"{name}: {tag} {ma[i1:i2]} -> {mb[j1:j2]}")
        if len(a[name]) != len(b[name]):
            print(f"info {name}: {len(a[name])} -> {len(b[name])} instructions")
    for substitution, count in sorted(used.items()):
        print(f"allowed {count:4d}x {substitution}")
    for failure in failures:
        print(f"FAIL {failure}")
    print("GATE PASS" if not failures else f"GATE FAIL ({len(failures)})")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("worktree")
    b.add_argument("out")
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--exact", action="store_true")
    d.add_argument("--allow", action="append", default=[])
    args = parser.parse_args()
    if args.command == "build":
        build(args.worktree, args.out)
        return 0
    allow = [tuple(rule.split("=", 1)) for rule in args.allow]
    return diff(args.before, args.after, args.exact, allow)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Build the baseline SASS, and self-check the gate**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync-base base.sass \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync-base base2.sass \
  && python3 sass_gate.py diff base.sass base2.sass --exact; echo EXIT=$?'
```

Expected: the build prints `base.sass: N functions from .../expert_stream_exl3...so` with N >= 11 (the 11 kernels in
the three headers, plus any from `exl3_ram_miss.cuh`), then `GATE PASS` and `EXIT=0`. Two builds of the same tree
must be identical, or `--exact` means nothing later. If the build step fails because `CUDA_VISIBLE_DEVICES=""`
blocks it, stop and report: do not drop the variable, since that would put an unlocked GPU context on the box.

- [ ] **Step 5: Record the baseline suite counts**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
       test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
       test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py \
       test/registered/unit/layers/moe/test_exl3_ram_miss_tables.py test/registered/unit/layers/moe/test_expert_stream.py \
       test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py \
       -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: the first line prints a path under `wt-native-sync-base/python/`. Write the two `N passed / M skipped /
K failed` lines, with both commands, into the task report: they are the comparison for every later run. If the second
command hits pyarrow collection errors (see the run protocol), drop the files that fail collection, record which,
and use the same file list at every later step.

---

### Task 1: Cache-hint intrinsics and the global timer (identical SASS)

**Files:**
- Create: `test/registered/unit/kernels/test_expert_stream_sync_primitives.py`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh:53-63` (`stream_copy16`, `stream_copy1`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh:483-508` (`copy_wait_read`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh:79-83` (`global_ns`) and its includes

**Interfaces:**
- Consumes: `sass_gate.py`, `base.sass` (Task 0).
- Produces: the test file, whose helpers `code_lines(name)` and `matches(pattern)` Tasks 2 and 3 extend;
  `after1.sass`.

- [ ] **Step 1: Write the failing source test**

```python
"""The expert_stream device headers use CUDA intrinsics and libcu++ for synchronization, not hand-written PTX (CPU).

Each rule here pins a convention whose machine-code equivalence was proven by a SASS gate when it was adopted
(docs/superpowers/plans/2026-09-27-expert-stream-native-sync.md), so a later edit cannot quietly reintroduce the
old form.
"""

import re

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import device_sources

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

HEADERS = {path.name: path for path in device_sources() if path.parent.name == "expert_stream"}


def code_lines(name: str) -> list[tuple[int, str]]:
    """(line number, code) for every line of header `name`, with `//` comments removed."""
    lines = []
    for number, line in enumerate(HEADERS[name].read_text().splitlines(), 1):
        code = line.split("//", 1)[0]
        if code.strip():
            lines.append((number, code))
    return lines


def matches(pattern: str) -> list[tuple[str, int, str]]:
    return [
        (name, number, code.strip())
        for name in sorted(HEADERS)
        for number, code in code_lines(name)
        if re.search(pattern, code)
    ]


def test_the_three_device_headers_are_found():
    assert sorted(HEADERS) == ["lease_device.cuh", "lease_kernels.cuh", "row_copy_kernels.cuh"]


def test_cache_hinted_copies_and_the_timer_use_intrinsics_not_ptx():
    assert matches(r"ld\.global\.cv|st\.global\.cg|%globaltimer") == []
```

- [ ] **Step 2: Commit the test, push, and prove it fails against the current sources**

```bash
git add test/registered/unit/kernels/test_expert_stream_sync_primitives.py
git commit -m "test(expert-stream): pin intrinsic cache-hinted copies and timer in the device headers"
git push
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync && git fetch origin && git checkout --detach origin/expert-stream-native-sync \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_expert_stream_sync_primitives.py -q -p no:randomly 2>&1 | tail -15; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `test_cache_hinted_copies_and_the_timer_use_intrinsics_not_ptx` FAILS, listing lines 55, 56, 61, 492 and
499 of `row_copy_kernels.cuh` and line 81 of `lease_device.cuh`. `EXIT=1`.

- [ ] **Step 3: Replace the copy asm**

In `row_copy_kernels.cuh`, replace `stream_copy16` and `stream_copy1` (lines 51-63) with:

```cpp
// ld.global.cv, never .nc: a tag-2 lane's host bytes are written while the kernel runs, and .nc may serve a line
// cached before its piece was published (LEASE_PROTOCOL.md E1 amendment). __ldcv/__stcg emit ld.global.cv/st.global.cg
// with a "memory" clobber, so they stay ordered after the acquire that admitted the lane.
SGL_DEVICE void stream_copy16(const uint8_t* src, uint8_t* dst) {
  __stcg(reinterpret_cast<longlong2*>(dst), __ldcv(reinterpret_cast<const longlong2*>(src)));
}

SGL_DEVICE void stream_copy1(const uint8_t* src, uint8_t* dst) {
  *dst = __ldcv(src);
}
```

Replace the unrolled loop body of `copy_wait_read` (lines 488-503) with:

```cpp
  for (; u + 3 * step < units; u += 4 * step) {
    longlong2 v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      v[k] = __ldcv(reinterpret_cast<const longlong2*>(src + 16 * (u + k * step)));
#pragma unroll
    for (int k = 0; k < 4; ++k)
      __stcg(reinterpret_cast<longlong2*>(dst + 16 * (u + k * step)), v[k]);
  }
```

- [ ] **Step 4: Replace the timer asm**

In `lease_device.cuh`, add `#include <cuda/ptx>` beside the other system includes (after `#include <algorithm>`),
and replace `global_ns` with:

```cpp
SGL_DEVICE uint64_t global_ns() {
  return cuda::ptx::get_sreg_globaltimer();
}
```

- [ ] **Step 5: Format, commit, push**

```bash
pre-commit run clang-format --files python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh
git add python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh
git commit -m "refactor(expert-stream): __ldcv/__stcg and cuda::ptx globaltimer in place of inline PTX (SASS identical)"
git push
```

- [ ] **Step 6: Gate: identical SASS**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync && git fetch origin && git checkout --detach origin/expert-stream-native-sync \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync after1.sass \
  && python3 sass_gate.py diff base.sass after1.sass --exact; echo EXIT=$?'
```

Expected: `GATE PASS`, `EXIT=0`. If `--exact` fails only in register numbering, fall back to
`python3 sass_gate.py diff base.sass after1.sass` (memory-sequence mode, no `--allow`), which must pass. Paste both
outputs into the task report. Any memory-opcode difference means the intrinsics are not equivalent: stop and report.

- [ ] **Step 7: Mutant: the gate catches a non-coherent load**

In a private worktree (the run protocol's mutant rule), change `__ldcv` in `stream_copy16` to `__ldg`, build, gate,
revert:

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-native-sync-mutant origin/expert-stream-native-sync \
  && cd /data/models/slang/nvfp4-work/wt-native-sync-mutant \
  && sed -i "0,/__ldcv(reinterpret_cast<const longlong2\*>(src))/s//__ldg(reinterpret_cast<const longlong2*>(src))/" python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh \
  && git diff --stat \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync-mutant mutant1.sass \
  ; python3 sass_gate.py diff base.sass mutant1.sass; echo EXIT=$? \
  ; git -C /data/models/slang/nvfp4-work/wt-native-sync-mutant checkout -- . \
  && git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-native-sync-mutant'
```

Expected: `git diff --stat` shows 1 file changed; the diff reports `FAIL ... LDG.E.128.STRONG.SYS -> LDG.E.128.CONSTANT
is not an allowed substitution` (or similar `.CONSTANT` wording) and `EXIT=1`. Then the worktree is removed.

- [ ] **Step 8: Run the source test and the suites**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
       test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: the path is under `wt-native-sync/python/`; counts equal the Task 0 baseline plus 2 passed (the two new
tests); `EXIT` matches the baseline's. Record the command and counts.

---

### Task 2 (revised 2026-09-27): named relaxed/acquire/release helpers, identical SASS

**Why this replaced the `cuda::atomic_ref` version.** The first Task 2 (`02b7422947`) failed its gate: GATE FAIL
(138). It failed for two reasons, and both are libcu++ behaviour in CUDA 13.4, not our call sites:
- **A local-pointer check on every access.** `cuda/std/__atomic/functions/cuda_local.h` wraps each atomic load,
  store, CAS and exchange in a run-time test of whether the generic pointer is local (`QSPC.E.L`), with a byte-copy
  fallback (`LDL.U8`/`STL.U8`, `__nanosleep(0)`). The test is compiled in when the pointer comes from a
  `const __grid_constant__` parameter, which every kernel here takes. That added 132 `LDL.U8`, 224 `STL.U8` and
  23 `QSPC` across 11 kernels.
- **Merging and reordering, even with the check off.** With `_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE` defined, nvcc
  still merges adjacent relaxed system-scope accesses into wider ones and moves memory operations around them
  (GATE FAIL (48)).

Evidence: `divix01:/data/models/slang/nvfp4-work/scratch-atomic-probe/{gate2.txt,gate2b.txt,probe5.cu}`.

So the helpers below keep today's exact instructions. Relaxed is a `volatile` access inside a named function (the
PTX memory model treats `ld/st.volatile` as relaxed at system scope). Acquire and release stay the `.global` asm
they already are. What changes is that every call site says which ordering it uses, instead of repeating a cast.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (the helpers; `write_record`;
  `lane_result_read`/`lane_result_reread`; `publish_terminal`; `lease_hit_wait_body`'s capacity)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh`
- Modify: `test/registered/unit/kernels/test_expert_stream_sync_primitives.py`
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (section 6.2)

Line numbers below are those of the base commit `4d8a9444cd`. Task 1 shifted some of them, so find each site by its
"Replace" text, which occurs exactly once at its site.

**Interfaces:**
- Consumes: `code_lines`, `matches` (Task 1); `after1.sass` (Task 1).
- Produces (in `sglang::device::expert_stream`, `lease_device.cuh`):
  - `template <typename T> T ld_relaxed_sys(const T* word)` and
    `template <typename T> void st_relaxed_sys(T* word, std::type_identity_t<T> value)`: a typed word; `T` is deduced.
  - `template <typename T> T ld_relaxed_sys(const uint8_t* address)` and
    `template <typename T> void st_relaxed_sys(uint8_t* address, std::type_identity_t<T> value)`: a wire field at a
    byte address; `T` is explicit. Call it with `T` of 2 bytes or more. For a `uint8_t` field use the typed form
    with no explicit `T`, because `st_relaxed_sys<uint8_t>(p, v)` is ambiguous.
  - `SGL_DEVICE int32_t ld_relaxed_gpu(const int32_t* word)`.
  - `ld_acquire_sys`, `st_release_sys`, `ld_acquire_sys64`, `st_release_sys64`: unchanged.
  - `ld_volatile` is deleted; its three callers use `ld_relaxed_sys`.

- [ ] **Step 1: Revert the failed attempt**

In the laptop worktree, add two revert commits (newest first). Do not amend or reset:

```bash
git revert --no-edit 02b7422947
git revert --no-edit 4f35ca9432
git push
```

Expected: two new commits. `git diff 8e97b0bd31 --stat` is empty (the tree equals Task 1's).

- [ ] **Step 2: The source tests (failing)**

Append to `test_expert_stream_sync_primitives.py`:

```python
def test_inline_ptx_lives_only_in_the_lease_device_helpers():
    assert sorted({name for name, _, _ in matches(r"\basm\b")}) == ["lease_device.cuh"]


def test_volatile_lives_only_in_the_relaxed_helpers():
    # Every concurrent access goes through ld/st_relaxed_sys, ld_acquire_sys{,64}, st_release_sys{,64} or
    # ld_relaxed_gpu; a raw volatile cast at a call site hides which ordering it relies on.
    assert [(name, code) for name, _, code in matches(r"\bvolatile\b") if not code.startswith("asm")] == [
        ("lease_device.cuh", "return *reinterpret_cast<const volatile T*>(word);"),
        ("lease_device.cuh", "*reinterpret_cast<volatile T*>(word) = value;"),
    ]
```

Commit (`test(expert-stream): pin named sync helpers in the device headers`), push, and run it as in Task 1 Step 2.
Expected: both new tests FAIL (the first lists `row_copy_kernels.cuh`'s `ld.relaxed.gpu` asm; the second lists
every raw `volatile` site), `EXIT=1`.

- [ ] **Step 3: The helpers**

In `lease_device.cuh`, add `#include <type_traits>` beside `<cstdint>`. Replace `ld_volatile` (the function
`SGL_DEVICE int32_t ld_volatile(const int32_t* address) {...}`) with:

```cpp
// Every word the host or another kernel accesses concurrently goes through one of these, so each call site states
// its ordering. Relaxed is a volatile access: the PTX memory model treats ld/st.volatile as relaxed at system scope,
// and nvcc emits LDG/STG.E.STRONG.SYS for it. Not cuda::atomic_ref: with CUDA 13.4's libcu++, an access through a
// __grid_constant__ parameter gains a run-time local-pointer check with a byte-copy fallback, and adjacent relaxed
// accesses are merged and reordered (plan 2026-09-27-expert-stream-native-sync, Task 2).
template <typename T>
SGL_DEVICE T ld_relaxed_sys(const T* word) {
  return *reinterpret_cast<const volatile T*>(word);
}

template <typename T>
SGL_DEVICE void st_relaxed_sys(T* word, std::type_identity_t<T> value) {
  *reinterpret_cast<volatile T*>(word) = value;
}

// A wire field at a byte address (lease_layout.h offsets); T names the field's type.
template <typename T>
SGL_DEVICE T ld_relaxed_sys(const uint8_t* address) {
  return ld_relaxed_sys(reinterpret_cast<const T*>(address));
}

template <typename T>
SGL_DEVICE void st_relaxed_sys(uint8_t* address, std::type_identity_t<T> value) {
  st_relaxed_sys<T>(reinterpret_cast<T*>(address), value);
}

// Device scope, for the stream kernel's abort word, which only its own blocks write.
SGL_DEVICE int32_t ld_relaxed_gpu(const int32_t* word) {
  int32_t value;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"(word) : "memory");
  return value;
}
```

- [ ] **Step 4: `lease_device.cuh` call sites**

`write_record`: replace the two pointer declarations and every store through them. The body from the first comment
through the release store becomes:

```cpp
  // Seqlock writer: invalidate seq before touching the payload, so a lapped record that
  // is half rewritten never passes the thread's read_record seq re-check.
  st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);
  __threadfence_system();
  st_relaxed_sys<uint16_t>(record + kRecRow, static_cast<uint16_t>(row));
  st_relaxed_sys<uint16_t>(record + kRecNeedCount, static_cast<uint16_t>(need_count));
  st_relaxed_sys<uint16_t>(record + kRecProtectCount, static_cast<uint16_t>(protect_count));
  st_relaxed_sys<uint16_t>(record + kRecStatus, 0);
  st_relaxed_sys<uint32_t>(record + kRecAfter, after);
  st_relaxed_sys<uint32_t>(record + kRecArmed, armed);
  st_relaxed_sys<uint32_t>(record + kRecLanes, lanes);
  for (int i = 0; i < kMaxIds; ++i) {
    st_relaxed_sys<int32_t>(record + kRecNeed + 4 * i, i < need_count ? need[i] : -1);
    st_relaxed_sys<int32_t>(record + kRecProtect + 4 * i, i < protect_count ? protect[i] : -1);
  }
  // The seqlock order the thread's read_record relies on: seq=0, fence, payload, seq last. The release store is the
  // second fence: it orders every payload store above before the seq.
  st_release_sys(record + kRecSeq, seq);
```

`lane_result_read` / `lane_result_reread`:

```cpp
SGL_DEVICE LaneRead lane_result_read(const uint8_t* result) {
  LaneRead r;
  r.ready = ld_acquire_sys64(result + kLeaseRrReady);
  r.slot_generation = ld_relaxed_sys<uint32_t>(result + kLeaseRrSlotGeneration);
  r.host_slot = ld_relaxed_sys<int32_t>(result + kLeaseRrHostSlot);
  r.expert = ld_relaxed_sys<int32_t>(result + kLeaseRrExpert);
  return r;
}

// Relaxed: the caller's fence already orders it after the payload loads, and nothing after it depends on it.
SGL_DEVICE uint64_t lane_result_reread(const uint8_t* result) {
  return ld_relaxed_sys<uint64_t>(result + kLeaseRrReady);
}
```

`publish_terminal`:

```cpp
  st_relaxed_sys<uint32_t>(terminal + kLeaseTermSkippedMask, skipped_mask);
  st_relaxed_sys<uint32_t>(terminal + kLeaseTermReason, reason);
```

`lease_hit_wait_body`'s capacity:

```cpp
  const uint32_t capacity = ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
```

- [ ] **Step 5: `lease_kernels.cuh` call sites**

| Line (base) | Replace | With |
|---|---|---|
| 77, 158, 238 | `ld_volatile(X)` | `ld_relaxed_sys(X)` |
| 98 | `*reinterpret_cast<volatile uint32_t*>(hot) = 0;` | `st_relaxed_sys<uint32_t>(hot, 0u);` |
| 100 | `*reinterpret_cast<volatile uint32_t*>(hot + 4) = static_cast<uint32_t>(experts);` | `st_relaxed_sys<uint32_t>(hot + 4, static_cast<uint32_t>(experts));` |
| 124 | `*reinterpret_cast<volatile uint64_t*>(request + kLeaseLrGen) = 0ull;` | `st_relaxed_sys<uint64_t>(request + kLeaseLrGen, 0ull);` |
| 126 | `*reinterpret_cast<volatile uint32_t*>(request + kLeaseLrCount) = static_cast<uint32_t>(planned_count);` | `st_relaxed_sys<uint32_t>(request + kLeaseLrCount, static_cast<uint32_t>(planned_count));` |
| 127 | `*reinterpret_cast<volatile uint32_t*>(request + kLeaseLrRow) = static_cast<uint32_t>(row);` | `st_relaxed_sys<uint32_t>(request + kLeaseLrRow, static_cast<uint32_t>(row));` |
| 135 | `*reinterpret_cast<volatile uint32_t*>(request + kLeaseLrFlags) = copy_engine != 0 ? kLeaseLrFlagCopyEngine : 0u;` | `st_relaxed_sys<uint32_t>(request + kLeaseLrFlags, copy_engine != 0 ? kLeaseLrFlagCopyEngine : 0u);` |
| 222, 342, 668 | `*reinterpret_cast<const volatile uint16_t*>(record + kRecStatus)` | `ld_relaxed_sys<uint16_t>(record + kRecStatus)` |
| 365-366, 692-693 | `*reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4)` | `ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4)` |
| 769-770, 898-899 | `*reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes)` | `ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes)` |

The hot bitmap (base lines 101-107): replace `volatile uint8_t* bits = reinterpret_cast<volatile uint8_t*>(hot + kHotHeaderBytes);`
with `uint8_t* bits = hot + kHotHeaderBytes;`, and the store `bits[byte] = mask;` with `st_relaxed_sys(bits + byte, mask);`
(the typed form, `T` deduced as `uint8_t`).

The lane arrays (base lines 128-133) become:

```cpp
    for (int i = 0; i < kMaxIds; ++i)
      st_relaxed_sys<int32_t>(request + kLeaseLrExpert + 4 * i, i < planned_count ? static_cast<int32_t>(planned[i]) : -1);
    for (int i = 0; i < kMaxIds; ++i) {
      st_relaxed_sys<int32_t>(
          request + kLeaseLrDst + 4 * i, dst_slots != nullptr && i < planned_count && i < dst_count ? dst_slots[i] : -1);
    }
```

- [ ] **Step 6: `row_copy_kernels.cuh` call sites**

| Line (base) | Replace | With |
|---|---|---|
| 34 | `int32_t old = *reinterpret_cast<volatile int32_t*>(word);` | `int32_t old = ld_relaxed_sys(word);` |
| 302 | `? *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4)` | `? ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4)` |
| 383 | `*reinterpret_cast<const volatile uint16_t*>(record + kRecStatus)` | `ld_relaxed_sys<uint16_t>(record + kRecStatus)` |
| 437 | `*reinterpret_cast<volatile int32_t*>(&state[kReqFailed]) = 1;` | `st_relaxed_sys(&state[kReqFailed], 1);` |
| 443 | `*reinterpret_cast<volatile int32_t*>(stream_abort) = 1;` | `st_relaxed_sys(stream_abort, 1);` |
| 569 | `*reinterpret_cast<const volatile int32_t*>(result + kLeaseRrHostSlot)` | `ld_relaxed_sys<int32_t>(result + kLeaseRrHostSlot)` |
| 570 | `*reinterpret_cast<const volatile int32_t*>(request + kLeaseLrDst + 4 * lane)` | `ld_relaxed_sys<int32_t>(request + kLeaseLrDst + 4 * lane)` |
| 641 | `*reinterpret_cast<const volatile uint32_t*>(done + kLeaseCdMask)` | `ld_relaxed_sys<uint32_t>(done + kLeaseCdMask)` |

The last block's abort read (base lines 456-457, `int32_t aborted;` followed by the `ld.relaxed.gpu` asm) becomes:

```cpp
  const int32_t aborted = ld_relaxed_gpu(stream_abort);
```

After the edits, `grep -n "volatile\|asm" ` over the three headers must print only lines inside
`lease_device.cuh`'s helpers, and comments.

- [ ] **Step 7: The doc note**

In `analysis/dsv41-drive/LEASE_PROTOCOL.md`, at the end of section 6.2 (after "No other primitives are
introduced."), add:

```markdown
**2026-09-27:** every concurrent device access in `expert_stream/*.cuh` now goes through a named helper in
`lease_device.cuh`: `ld_relaxed_sys<T>` / `st_relaxed_sys<T>` (a volatile access, i.e. relaxed at system scope),
`ld_acquire_sys{,64}` / `st_release_sys{,64}`, and `ld_relaxed_gpu`. `ld_volatile` is gone. The SASS is
unchanged. `cuda::atomic_ref` was tried and rejected. Under CUDA 13.4's libcu++, every access through a
`__grid_constant__` parameter gains a run-time local-pointer check with a byte-copy fallback, and adjacent relaxed
accesses are merged and reordered.
```

- [ ] **Step 8: Format, commit, push**

Format with clang-format 20.1.7, the rev pinned in `.pre-commit-config.yaml` (`pre-commit run clang-format --files ...`,
or the 20.1.7 wheel if pre-commit is absent), then:

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/row_copy_kernels.cuh analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "refactor(expert-stream): named relaxed/acquire/release helpers for every concurrent access (SASS identical)"
git push
```

- [ ] **Step 9: Gate: identical SASS**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync && git fetch origin && git checkout --detach origin/expert-stream-native-sync \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync after2.sass \
  && python3 sass_gate.py diff after1.sass after2.sass --exact; echo EXIT=$?'
```

Expected: `GATE PASS`, `EXIT=0`. The register-numbering fallback of Task 1 Step 6 applies here the same way, and no
other one. Any other difference is a STOP.

- [ ] **Step 10: Mutant: the gate catches a lost volatile**

In `wt-native-sync-mutant` at the pushed commit, make the pointer-form `ld_relaxed_sys` a plain load
(`return *word;`), build, gate against `after1.sass` with `--exact`, then revert and remove the worktree:

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-native-sync-mutant origin/expert-stream-native-sync \
  && cd /data/models/slang/nvfp4-work/wt-native-sync-mutant \
  && python3 -c "
import pathlib
p = pathlib.Path(\"python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh\")
old = \"return *reinterpret_cast<const volatile T*>(word);\"
t = p.read_text(); assert t.count(old) == 1
p.write_text(t.replace(old, \"return *word;\"))
" && git diff --stat \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync-mutant mutant2.sass \
  ; python3 sass_gate.py diff after1.sass mutant2.sass --exact; echo EXIT=$? \
  ; git -C /data/models/slang/nvfp4-work/wt-native-sync-mutant checkout -- . \
  && git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-native-sync-mutant'
```

Expected: `git diff --stat` shows 1 file changed; `GATE FAIL`, `EXIT=1` (the slot-map and `state[]` reads lose
`.STRONG.SYS`, or are hoisted out of polling loops); then the worktree is removed.

- [ ] **Step 11: Source test and suites**

Run Task 1 Step 8's command. Expected: the Task 0 baseline plus 4 passed; `EXIT` matches the baseline's. Then run
Task 0 Step 5's second command in `wt-native-sync` with the same file list. Expected: the same counts as its
baseline. Record both commands and counts.

---

### Task 3: Acquire/release fences for the five seqlocks

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (fences in `write_record`,
  `lane_result_valid`, `lease_hit_wait_body`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (the hot-page and LaneRequest
  invalidating fences in `exl3_ram_miss_post_kernel`)
- Modify: `test/registered/unit/kernels/test_expert_stream_sync_primitives.py`
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (section 6.3)

**Interfaces:**
- Consumes: `matches` (Task 1); `after2.sass` (Task 2).
- Produces: `after3.sass`.

**The argument, site by site.** Each device fence pairs with a host counterpart that already has the other half of
Boehm's seqlock ("Can seqlocks get along with programming language memory models?", 2012): the writer does
invalidate, *release* fence, payload, release store; the reader does acquire load, payload, *acquire* fence,
re-read. A sequentially consistent fence is stronger than either half needs.

| Device site | Role | Host partner | Partner's shape |
|---|---|---|---|
| `write_record` (`lease_device.cuh`, after `seq = 0`) | writer, also used for advisories | `read_record` (`host/tier_protocol.h:161-179`) | acquire seq, payload, `atomic_thread_fence(acquire)`, acquire seq |
| post kernel, hot page (`lease_kernels.cuh:99`) | writer | `read_gpu_hot` (`host/ram_tier.h:430-440`) | acquire seq, payload, `atomic_thread_fence(acquire)`, acquire seq |
| post kernel, LaneRequest (`lease_kernels.cuh:125`) | writer | `read_lane_request` (`host/ram_tier.h:647-662`) | acquire gen, payload, `atomic_thread_fence(acquire)`, acquire gen |
| `lane_result_valid` (`lease_device.cuh`, between read and re-read) | reader | `grant_lane_group_locked` (`host/ram_tier.h:832-858`) | release-store 0, `_mm_sfence`, payload, `_mm_sfence`, release-store tag |
| `lease_hit_wait_body` (one per pass) | reader | same | same |

Before editing, open each host partner at the cited lines and confirm the shape. If any partner differs, leave that
device fence unchanged, drop its row, and say so in the report.

- [ ] **Step 1: Extend the source test (failing)**

Append to `test_expert_stream_sync_primitives.py`:

```python
def test_only_the_smack_publish_keeps_a_seq_cst_system_fence():
    # The five seqlock fences are acquire/release (Boehm's shape, paired with the host's); SmAck orders other
    # threads' loads through __syncthreads before thread 0's release, a different argument, so it stays seq_cst.
    assert [(name, code) for name, _, code in matches(r"__threadfence_system\(\)")] == [
        ("row_copy_kernels.cuh", "__threadfence_system();"),
    ]


def test_the_seqlock_fences_are_two_acquires_and_three_releases():
    assert [(name, code) for name, _, code in matches(r"atomic_thread_fence")] == [
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);"),
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);"),
        ("lease_kernels.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
        ("lease_kernels.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
    ]
```

Commit, push, run as in Task 1 Step 2. Expected: both FAIL, `EXIT=1`.

- [ ] **Step 2: The two reader fences**

In `lease_device.cuh`, add `#include <cuda/atomic>` beside `#include <cuda/ptx>` (for `cuda::atomic_thread_fence`;
a fence takes no pointer, so libcu++'s local-pointer check does not apply to it). Then:
- In `lane_result_valid`, replace `__threadfence_system();` with
  `cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);`.
- In `lease_hit_wait_body`, replace `__threadfence_system();  // one per pass: every lane's payload loads before any lane's re-read`
  with `cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);  // one per pass: every lane's payload loads before any lane's re-read`.

In the comment block above `struct LaneRead`, replace the sentence "The reader must order its payload loads before
the re-read with a system fence: an acquire orders only what follows it, so without the fence the re-read may be
served before the payload and a rewrite caught half-way passes." with "The reader must order its payload loads
before the re-read with an acquire fence (Boehm's seqlock reader): an acquire *load* orders only what follows it, so
without the fence the re-read may be served before the payload and a rewrite caught half-way passes."

- [ ] **Step 3: The three writer fences**

- In `write_record`, replace the `__threadfence_system();` after `st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);`
  with `cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);`.
- In `exl3_ram_miss_post_kernel` (`lease_kernels.cuh`), replace the `__threadfence_system();` after
  `st_relaxed_sys<uint32_t>(hot, 0u);` and the one after `st_relaxed_sys<uint64_t>(request + kLeaseLrGen, 0ull);`
  with the same release fence.

- [ ] **Step 4: The doc note**

In `analysis/dsv41-drive/LEASE_PROTOCOL.md` section 6.3, after the paragraph ending "Cost per fence still
unmeasured **[OPEN 5]**.", add:

```markdown
**2026-09-27:** the three invalidating fences are now `atomic_thread_fence(release, system)` and the device
readers' fences in `lane_result_valid` and stage 1 are `atomic_thread_fence(acquire, system)`: Boehm's seqlock
halves, each paired with a host side of the other half (`read_record`, `read_gpu_hot`, `read_lane_request`;
`grant_lane_group_locked`). On sm_120 this is `MEMBAR.SC.SYS` -> `MEMBAR.ALL.SYS`. The SmAck publish keeps
`__threadfence_system()`. The saving is still unmeasured **[OPEN 5]**.
```

- [ ] **Step 5: Format, commit, push**

```bash
pre-commit run clang-format --files python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  analysis/dsv41-drive/LEASE_PROTOCOL.md
git commit -m "perf(expert-stream): acquire/release fences for the five seqlocks in place of seq_cst system fences"
git push
```

- [ ] **Step 6: Gate: only the fence substitution**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync && git fetch origin && git checkout --detach origin/expert-stream-native-sync \
  && cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py build /data/models/slang/nvfp4-work/wt-native-sync after3.sass \
  && python3 sass_gate.py diff after2.sass after3.sass --allow MEMBAR.SC.SYS=MEMBAR.ALL.SYS; echo EXIT=$? \
  && grep -c "MEMBAR.SC.SYS" after3.sass'
```

Expected: only `allowed Nx MEMBAR.SC.SYS -> MEMBAR.ALL.SYS`, `GATE PASS`, `EXIT=0`. The substituting functions are
the post kernel and the kernels that inline `lane_result_valid` or `lease_hit_wait_body`. The final `grep -c`
must be nonzero: the copy-wait kernel's SmAck fence is still `MEMBAR.SC.SYS`. Paste the output.

- [ ] **Step 7: Suites**

Run Task 1 Step 8's command and Task 0 Step 5's second command in `wt-native-sync`. Expected: Task 0 baseline plus
6 passed for the first, the baseline for the second; `EXIT` matches the baseline. Record commands and counts.

- [ ] **Step 8: Fence mutants (recorded, not a gate)**

Two mutants, each in `wt-native-sync-mutant` at the pushed commit, each reverted with `git checkout -- .` and the
worktree removed afterwards:
- M1: delete the release fence line in `write_record`.
- M2: delete the acquire fence line in `lane_result_valid`.

For each, run the tests that exercise those seqlocks under the GPU lock:

```bash
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_two_phase.py \
  test/registered/unit/kernels/test_exl3_ram_miss_lease_service.py test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Record for each mutant whether it was killed (EXIT != 0) or survived. **Survival is the expected outcome, not a
failure of this task.** x86 stores are totally ordered and the GPU's posted PCIe writes arrive in order, so a
missing fence may never tear a record on this box; the ordering argument above is the proof, and the mutants
measure what the suite can see. After both reverts, rerun the same three files in `wt-native-sync` and record that
they are green again.

---

### Task 4: Whole-branch verification and cleanup

**Files:** none changed.

- [ ] **Step 1: Branch against base, in one gate run**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/scratch-atomic-probe \
  && python3 sass_gate.py diff base.sass after3.sass --allow MEMBAR.SC.SYS=MEMBAR.ALL.SYS; echo EXIT=$?'
```

Expected: `GATE PASS`, `EXIT=0`, with only Task 3's `MEMBAR.SC.SYS -> MEMBAR.ALL.SYS` substitutions.

- [ ] **Step 2: The manual GPU tests (captured graph)**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-native-sync && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
       test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: the same counts as these files had in Task 0 Step 5 (they are in that file list).

- [ ] **Step 3: Scan for unfinished work**

```bash
git diff 4d8a9444cd --stat
git diff 4d8a9444cd | grep -nE "TODO|FIXME|XXX|\.skip|\.only|pytest\.mark\.skip" ; echo "matches: $?"
```

Expected: the stat lists exactly the three headers, the test file, `LEASE_PROTOCOL.md`, and this plan if it was
committed; the grep prints nothing (`matches: 1`).

- [ ] **Step 4: Remove the divix01 worktrees and scratch**

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-native-sync-base \
  && git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-native-sync \
  && rm -rf /data/models/slang/nvfp4-work/scratch-atomic-probe/jit-*'
```

Keep `scratch-atomic-probe/*.sass`, `sass_gate.py` and the probes until the branch is reviewed: they are the
evidence the review reads.
