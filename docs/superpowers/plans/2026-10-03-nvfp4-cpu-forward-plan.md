# NVFP4 CPU kernel: EXL3-format layout (experts, shapes, ForwardPlan, OpenMP, Python build) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the NVFP4 CPU expert kernel (`python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp`) the same structure as the optimized EXL3 kernel — compile-time layer/expert structs (`experts.hpp`), shapes (`shapes.hpp`), a `ForwardPlan<Shape, Isa>` with `PlanTraits`, and an OpenMP team in place of the hand-written `Workers` pool — built from Python instead of CMake, with every output bit-identical to today's.

**Architecture:** A forward becomes `ForwardPlan<Shape, Isa>::run`, dispatched once per call: `ForwardPlan<Dsv41Shape, Isa::Avx2>` when the layer is DeepSeek V4.1's routed expert (H=5120, I=2304, SwiGLU limit 10) on an AVX2 build, else `ForwardPlan<GenericShape, kBuildIsa>`. The plan reads slots through `StridedExperts<Shape>` (the descriptor's seven slab bases and byte strides) and runs one OpenMP team per forward with four phases separated by barriers: quantize the input, gate/up + SwiGLU for every routed expert, quantize every intermediate, down + routing-weighted sum into `out`. Workers are pinned exactly as EXL3's are (`freeze_compute_cores` / `pin_compute_worker`). `optimized/build.py` compiles the library (and test harnesses); `nvfp4_cpu_ext.py` builds it on first use and caches it by content hash.

**Tech Stack:** C++20 + C11 (GCC, `-O3 -ffp-contract=off -march=native -fopenmp`), the vendored GGML subset under `nvfp4_cpu/upstream/`, Python 3 + ctypes + numpy + filelock, pytest, Google Benchmark bench under `nvfp4_cpu/bench/`.

**Spec:** the user's request of 2026-10-03 (no separate spec file), quoted:
> get it setup in the same format as the exl3 code, including removing `nvfp4_cpu/CMakeLists.txt` and adding the proper python builds; 1: compile time structs, and proper experts definition like `exl3_cpu/optimized/experts.hpp`; 2: a proper forwardplan class including plan traits; 3: `exl3_cpu/optimized/shapes.hpp` shapes def; 4: remove the manual works def from `nvfp4_cpu/optimized/moe_mul1.cpp` and use the exl3 format with openmp.

The EXL3 reference implementation for every shape here: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/{experts.hpp,shapes.hpp,forward_plan.hpp,moe_mul1.cpp,build.py}` and its plan `docs/superpowers/plans/2026-10-02-exl3-cpu-forward-plan.md`.

## Amendments during execution (2026-10-03)

- **The specialized shape is MiMo V2.6 Pro's, not DeepSeek V4.1's** (user: there is no DSV4.1 NVFP4 checkpoint; the
  first model is `/mnt/nvme4/mimi-v26-pro`). Wherever this plan says `Dsv41Shape` / "the DSV4.1 plan", read
  `MimoV26ProShape` (hidden 6144, intermediate 2048, SiLU with no clamp: `act_limit` 0) and
  `ForwardPlan<MimoV26ProShape, Isa::Avx2>`. The A/B harness's large configs are `mimo_v26_pro`,
  `mimo_v26_pro_l2_up_scaled` (the specialized plan), `mimo_v26_pro_limit10`, `h5120_n2304_lim10`,
  `h5120_n2304_l2_up_scaled` (the generic plan): 9 configs, 288 cases. Task 6 benches `--hidden=6144 --intermediate=2048`.
- **"Portable" pins `-march=x86-64 -mtune=generic`.** divix01's GCC defaults to x86-64-v3 (AVX2), so omitting
  `-march=native` did not select the scalar dot product. The A/B baseline is `/mnt/nvme1/nvfp4-plan/base3`.

## Global Constraints

- **Bit-exact or it does not ship.** From Task 3 on, every task's gate includes `run_nvfp4_cpu_forward_checks.sh WT OUT BASE_OUT` (Task 2) printing `PASS native bit-exact`, `PASS baseline bit-exact`, `PASS portable bit-exact` and exiting 0. A refactor task whose behavior must not change has no RED step of its own: the A/B baseline captured in Task 2 is its test.
- **The C ABI in `cpu_experts_cabi.h` does not change**: same four functions, same descriptor (ABI v1), same status codes (0 ok, 1 internal error, 2 invalid arguments, 3 concurrent use). `set_cores` after the first forward stays status **2** (not EXL3's 1). The one deliberate relaxation: a later forward may use more workers than the first (up to the configured cores), which the old pool refused with 1.
- Do not change arithmetic or its order: the dot product (`dot_nvfp4.h`, the upstream-baseline `dot` body), Q8_0 quantization, SwiGLU with its clamp, `alpha * inv_input_scale [* route]` and the per-output sum over routes in routing order starting from `0.f`. Changes are to scheduling, data access and structure only.
- Compiler flags stay the kernel's: `-O3 -ffp-contract=off` (never `-Ofast`/`-ffast-math`: the bench's Q8 reference tolerance and bit-exactness both depend on it), `-march=native` for native builds. Adds `-std=c++20 -fopenmp -pthread`. `nvfp4.c` is compiled as C (`-x c -std=c11`): it uses implicit `void*` conversions C++ rejects.
- The kernel becomes Linux + OpenMP only, as EXL3's is (`#error` otherwise). The laptop (macOS) cannot build it; every build and test runs on divix01.
- Keep the upstream-baseline build (`NVFP4_CPU_UPSTREAM_BASELINE`, the bench's `baseline` backend) and the portable (no `-march=native`, scalar dot) build working; both are A/B-gated.
- No new `SGLANG_*` environment variable. Compilers come from `--cxx`/`CXX`.
- Code is edited on the laptop, committed, pushed to `origin`, and run on divix01 in a pulled worktree (`.claude/rules/divix01-run-protocol.md`). Never rsync/scp a tree. CPU jobs under `taskset -c 0-63`. Read pytest's own exit status (`PIPESTATUS[0]`), never a pipe's.
- No server may be running during the latency A/B (Task 6): check with `pgrep -f '[s]glang.launch_server'` (bracketed so it cannot match itself).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft`. Stage files by name.

## Non-goals (say so in the final report; ask before doing)

- An `Nvfp4CpuQuantTrait` for `python/sglang/srt/layers/moe/cpu_experts/` (the service still runs EXL3 only; the README says runtime selection is not wired).
- A `sglang_nvfp4_cpu_experts_keep_warm` ABI like EXL3's.
- Any kernel tuning beyond the `kRowUnit` measurement in Task 6.

## Review Focus

1. **Stale scratch across layers.** The scratch arena is reused by every layer a thread forwards. A layer whose hidden or intermediate size is not a multiple of 64 must still read zeros in the padded tail of its input and intermediate. The A/B config order in Task 2 interleaves 80-wide layers after 5120-wide ones for exactly this, and `prepare_scratch` zeroes the tails on every call. Reviewers check that every tail range is written on every call.
2. **A barrier skipped by some workers.** The "invalid" early-out is read after a barrier. Every worker must take the same branch, or the team deadlocks. Task 4's `test_an_intermediate_q8_cannot_represent_returns_2_and_leaves_out_untouched` covers the Middle-phase path, and the A/B nonfinite-input case covers the PrepareInput path. Reviewers check that no other read of `invalid` sits between a write and its barrier.
3. **An OpenMP environment that shrinks the team.** If `OMP_THREAD_LIMIT` or `OMP_DYNAMIC` yields fewer threads than requested, the forward must fail with status 1 and leave `out` untouched, never silently compute on fewer workers. Covered by Task 4's `test_a_team_smaller_than_requested_fails_the_forward_and_leaves_out_untouched`.
4. **Routing order and skipped lanes.** The kernel must preserve the order of `-1` slots, zero weights (including `-0.0`), duplicate slots, and the routing order of the sum. The A/B routings `k5_skip_dup_zero` and `k8_negzero` cover these.
5. **Concurrent use.** A forward, free or `set_cores` issued during another forward must return 3, not race the scratch. Nothing exercises this, because the timing is hard to force. Reviewers check that all three still `try_lock` the same `forward_mutex` before touching state.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `python/sglang/srt/layers/quantization/nvfp4_cpu/CMakeLists.txt` | **deleted** | 1 |
| `.../nvfp4_cpu/optimized/build.py` (new) | compile the library or a harness executable; CLI like EXL3's `build.py` | 1 |
| `python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py` (new) | build-on-first-use loader, cached by content hash, returns `ctypes.CDLL` | 1 |
| `test/registered/unit/kernels/test_nvfp4_cpu_build.py` (new) | build.py / loader tests; runs the two native harnesses CTest used to run | 1 |
| `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp` (new) | A/B harness: fixed forwards through the C ABI, every output and status dumped | 2 |
| `test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh` (new) | divix01 driver: build each variant, dump, compare bitwise | 2 |
| `.../nvfp4_cpu/optimized/layout.h` | `rounded` becomes `constexpr` | 3 |
| `.../nvfp4_cpu/optimized/experts.hpp` (new) | `LayerInfo`, `SlabName`, `SlabRowBytes`, `Projection`, `w13_rows`, `StridedExperts<Shape>` | 3 |
| `.../nvfp4_cpu/optimized/shapes.hpp` (new) | `GenericShape` (3), `Dsv41Shape` (5) | 3, 5 |
| `.../nvfp4_cpu/optimized/forward_plan.hpp` (new) | `Phase`, `PlanTraits`, `share`, `ForwardPlan` | 4, 5 |
| `.../nvfp4_cpu/optimized/moe_mul1.cpp` | registry, cores, arithmetic helpers, ctx/arena, dispatch, C ABI; `Workers`/`Work`/`calculate` removed | 3, 4, 5 |
| `test/registered/unit/kernels/test_nvfp4_cpu_experts.py` (new) | C ABI behavior under the OpenMP team (subprocess per case) | 4 |
| `.../nvfp4_cpu/bench/CMakeLists.txt` | standalone; OpenMP; C++20 | 4 |
| `.../nvfp4_cpu/bench/run.sh` | exports the OpenMP wait policy; records it | 6 |
| `.../nvfp4_cpu/optimized/README.md`, `bench/README.md`, `upstream/README.md` | build, threading, code layout | 1, 6 |

The `.hpp` files follow EXL3's pattern: included mid-file inside `moe_mul1.cpp`'s anonymous namespace, not standalone.

## Working environment

- Laptop worktree: `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-nvfp4-plan`, branch `nvfp4-cpu-forward-plan` from `origin/master` (`278b3c5fc8` or later):
  `git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4 worktree add -b nvfp4-cpu-forward-plan ../sglang-nvfp4-nvfp4-plan origin/master`
- divix01 worktrees (create in Task 1 / Task 2):
  - branch: `/data/models/slang/nvfp4-work/wt-nvfp4-plan`, detached at `origin/nvfp4-cpu-forward-plan`, refreshed per task with `git -C <wt> fetch -q origin && git -C <wt> checkout -q --detach origin/nvfp4-cpu-forward-plan`.
  - base: `/data/models/slang/nvfp4-work/wt-nvfp4-plan-base`, at Task 2's commit (kernel identical to master).
- Compiler for every build, test and A/B: `GCC15=/opt/rh/gcc-toolset-15/root/usr/bin/g++` (divix01's default `g++` is GCC 14).
- Scratch output: `/mnt/nvme1/nvfp4-plan/` (not `/tmp`, which is on the 88%-full root volume).
- **`divix_pytest FILES...`** below means, on divix01 in the branch worktree:
  ```bash
  cd /data/models/slang/nvfp4-work/wt-nvfp4-plan && CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
    /data/models/slang/.venv/bin/python -m pytest FILES -q -p no:randomly 2>&1 | tail -15; echo "EXIT=${PIPESTATUS[0]}"
  ```
- **`ab_check`** below means, on divix01:
  ```bash
  bash /data/models/slang/nvfp4-work/wt-nvfp4-plan/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh \
    /data/models/slang/nvfp4-work/wt-nvfp4-plan /mnt/nvme1/nvfp4-plan/task<N> /mnt/nvme1/nvfp4-plan/base; echo EXIT=$?
  ```
  Expected: `PASS native bit-exact`, `PASS baseline bit-exact`, `PASS portable bit-exact`, `EXIT=0`.

---

### Task 1: Python build, build-on-first-use loader, CMake removed

**Files:**
- Create: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py`
- Create: `python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py`
- Create: `test/registered/unit/kernels/test_nvfp4_cpu_build.py`
- Delete: `python/sglang/srt/layers/quantization/nvfp4_cpu/CMakeLists.txt`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/README.md` (Build section), `.../nvfp4_cpu/bench/README.md` (Build section)

**Interfaces:**
- Produces: `build.build(output: Path, *, cxx: str, native: bool = True, upstream_baseline: bool = False, main: Path | None = None, extra_flags: Sequence[str] = ()) -> Path`; module constants `CXX_FLAGS`, `C_FLAGS`; CLI `build.py --output PATH [--cxx CXX] [--portable] [--upstream-baseline] [--main HARNESS.cpp]`.
- Produces: `nvfp4_cpu_ext.nvfp4_cpu_library(build_dir: str | None = None) -> ctypes.CDLL` (cached per process; `.cache_clear()` available).

- [ ] **Step 1: Create the worktree and write the failing test**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4 fetch origin
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4 worktree add -b nvfp4-cpu-forward-plan ../sglang-nvfp4-nvfp4-plan origin/master
```

`test/registered/unit/kernels/test_nvfp4_cpu_build.py`:

```python
"""The NVFP4 CPU expert library builds from Python and passes its native harnesses (Linux, GCC with OpenMP).

``nvfp4_cpu/optimized/build.py`` compiles the kernel into a shared library or, with a harness ``main``, into an
executable; ``nvfp4_cpu_ext.nvfp4_cpu_library`` builds the library once per content hash and loads it. The two native
harnesses are the ones the removed CMake build ran under CTest.
"""

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
OPTIMIZED = REPO / "python/sglang/srt/layers/quantization/nvfp4_cpu/optimized"
CXX = os.environ.get("CXX") or shutil.which("g++")
C_ABI = (
    "sglang_nvfp4_cpu_experts_register_slabs",
    "sglang_nvfp4_cpu_experts_free_layer",
    "sglang_nvfp4_cpu_experts_forward",
    "sglang_nvfp4_cpu_experts_set_cores",
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", OPTIMIZED / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_py_makes_a_library_exporting_the_c_abi(tmp_path):
    library = ctypes.CDLL(str(_build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)))
    for name in C_ABI:
        getattr(library, name)


def test_the_loader_builds_once_per_content_and_reuses_the_library(tmp_path):
    from sglang.srt.layers.quantization import nvfp4_cpu_ext

    nvfp4_cpu_ext.nvfp4_cpu_library.cache_clear()
    library = nvfp4_cpu_ext.nvfp4_cpu_library(str(tmp_path))
    built = list(tmp_path.glob("*.so"))
    assert len(built) == 1
    stamp = built[0].stat().st_mtime_ns
    nvfp4_cpu_ext.nvfp4_cpu_library.cache_clear()
    nvfp4_cpu_ext.nvfp4_cpu_library(str(tmp_path))
    assert [p.stat().st_mtime_ns for p in tmp_path.glob("*.so")] == [stamp]
    for name in C_ABI:
        getattr(library, name)


@pytest.mark.parametrize(
    "harness, flags",
    [
        ("nvfp4_cpu_sanitizer.cpp", ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]),
        ("nvfp4_cpu_ggml_check.cpp", []),
    ],
)
def test_the_native_harness_passes(tmp_path, harness, flags):
    exe = _build_module().build(
        tmp_path / Path(harness).stem, cxx=CXX, main=REPO / "test/registered/unit/kernels" / harness, extra_flags=flags
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Commit the test, push, run it on divix01 to see it fail**

```bash
cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-nvfp4-plan
git add test/registered/unit/kernels/test_nvfp4_cpu_build.py
git commit -m "$(cat <<'EOF'
test(nvfp4-cpu): the kernel builds from Python and passes its native harnesses (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push -u origin nvfp4-cpu-forward-plan
ssh divix01 'git -C /data/models/slang/sglang fetch origin && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-nvfp4-plan origin/nvfp4-cpu-forward-plan && mkdir -p /mnt/nvme1/nvfp4-plan'
```

Run: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_build.py`
Expected: 4 failed: `FileNotFoundError` for `optimized/build.py`; `ImportError` for `nvfp4_cpu_ext`. `EXIT=1`.

- [ ] **Step 3: Write `build.py`**

`python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py`:

```python
"""Build the NVFP4 CPU expert library (or a native harness linked with it) without CMake.

The kernel is optimized/moe_mul1.cpp plus the vendored GGML C subset (../upstream/nvfp4.c, compiled as C: it relies on
C's implicit void* conversions). -ffp-contract=off is part of the arithmetic contract: never build it with -Ofast.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

SRC = Path(__file__).resolve().parent
UPSTREAM = SRC.parent / "upstream"
CXX_FLAGS = ["-std=c++20", "-O3", "-ffp-contract=off", "-fPIC", "-pthread", "-fopenmp"]
C_FLAGS = ["-x", "c", "-std=c11", "-O3", "-ffp-contract=off", "-fPIC"]


def build(
    output: Path,
    *,
    cxx: str,
    native: bool = True,
    upstream_baseline: bool = False,
    main: Path | None = None,
    extra_flags: Sequence[str] = (),
) -> Path:
    """Compile the kernel into ``output``: a shared library, or with ``main``, that harness's executable.

    ``native`` adds -march=native (the AVX2 dot product on an AVX2 host; without it, the scalar loop).
    ``upstream_baseline`` builds the bench's baseline: each row converted to GGML blocks before GGML's own dot product.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    common = (["-march=native"] if native else []) + list(extra_flags)
    if upstream_baseline:
        common.append("-DNVFP4_CPU_UPSTREAM_BASELINE=1")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        c_object = Path(tmp) / "nvfp4.o"
        subprocess.run([cxx, *C_FLAGS, *common, "-c", str(UPSTREAM / "nvfp4.c"), "-o", str(c_object)], check=True)
        sources = [str(SRC / "moe_mul1.cpp")] + ([str(Path(main).resolve())] if main else [])
        link = [] if main else ["-shared"]
        subprocess.run(
            [cxx, *CXX_FLAGS, *common, "-I", str(SRC), *sources, str(c_object), *link, "-o", str(output)], check=True
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination; keep it outside the source tree")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--portable", action="store_true", help="omit -march=native (the scalar dot product)")
    parser.add_argument("--upstream-baseline", action="store_true", help="the bench's GGML-conversion baseline")
    parser.add_argument("--main", type=Path, help="link this harness source into an executable instead")
    args = parser.parse_args()
    print(
        build(
            args.output,
            cxx=args.cxx,
            native=not args.portable,
            upstream_baseline=args.upstream_baseline,
            main=args.main,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the loader**

`python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py`:

```python
"""The NVFP4 CPU expert library, built on first use by nvfp4_cpu/optimized/build.py.

The library is cached under ``build_dir`` by a hash of its sources, the build flags, the compiler's version and what
-march=native means on this host, so an edit, a compiler change or a different CPU builds a new one; a file lock keeps
concurrent processes from building the same one twice. The compiler is $CXX, else g++.
"""

import ctypes
import functools
import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Optional

from filelock import FileLock

_KERNEL = Path(__file__).resolve().parent / "nvfp4_cpu"
_DEFAULT_BUILD_DIR = "~/.cache/sglang/nvfp4_cpu"


def _build_module():
    spec = importlib.util.spec_from_file_location("nvfp4_cpu_build", _KERNEL / "optimized" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources() -> list[Path]:
    optimized, upstream = _KERNEL / "optimized", _KERNEL / "upstream"
    return sorted(
        [*optimized.glob("*.cpp"), *optimized.glob("*.hpp"), *optimized.glob("*.h"), *optimized.glob("*.py")]
        + [*upstream.glob("*.c"), *upstream.glob("*.h")]
    )


def library_path(build_dir: Path, cxx: str) -> Path:
    """Where the library for the current sources, flags, compiler and host CPU lives."""
    build = _build_module()
    digest = hashlib.sha256()
    digest.update(subprocess.check_output([cxx, "--version"]))
    digest.update(subprocess.check_output([cxx, "-march=native", "-Q", "--help=target"]))
    digest.update(" ".join(build.CXX_FLAGS + build.C_FLAGS).encode())
    for path in _sources():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return build_dir / f"libsglang_nvfp4_cpu_{digest.hexdigest()[:16]}.so"


@functools.cache
def nvfp4_cpu_library(build_dir: Optional[str] = None) -> ctypes.CDLL:
    """The native library (built here first if needed), exporting the cpu_experts_cabi.h functions."""
    cxx = os.environ.get("CXX", "g++")
    root = Path(os.path.expanduser(build_dir or _DEFAULT_BUILD_DIR))
    root.mkdir(parents=True, exist_ok=True)
    path = library_path(root, cxx)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.so")
            _build_module().build(partial, cxx=cxx)
            os.replace(partial, path)
    return ctypes.CDLL(str(path))
```

- [ ] **Step 5: Delete the CMake build and update the two Build sections**

```bash
git rm python/sglang/srt/layers/quantization/nvfp4_cpu/CMakeLists.txt
```

In `nvfp4_cpu/optimized/README.md`, replace everything from `Build from the checkout root:` through the paragraph ending `rebuild before moving to a different ISA.` with:

````markdown
Build from the checkout root, on the Linux machine that will run it (GCC with OpenMP):

```sh
python python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py \
  --cxx /opt/rh/gcc-toolset-15/root/usr/bin/g++ --output /absolute/path/libsglang_nvfp4_cpu.so
```

`--portable` omits `-march=native` (the scalar dot product); `--upstream-baseline` builds the bench's GGML-conversion
baseline; `--main HARNESS.cpp` links a native harness into an executable instead. Inside SGLang,
`sglang.srt.layers.quantization.nvfp4_cpu_ext.nvfp4_cpu_library()` builds the library on first use with `$CXX`
and caches it under `~/.cache/sglang/nvfp4_cpu` by a hash of the sources, flags, compiler and host CPU. A native build
targets the build machine's ISA. The arithmetic needs `-ffp-contract=off` and must never be built with `-Ofast`.

The native harnesses (`test/registered/unit/kernels/nvfp4_cpu_{sanitizer,ggml_check}.cpp`) build and run under
`test/registered/unit/kernels/test_nvfp4_cpu_build.py`.
````

In `nvfp4_cpu/bench/README.md`, replace the `## Build` section's code block and the paragraph after it (through `benchmark executables.`) with:

````markdown
```sh
src=python/sglang/srt/layers/quantization/nvfp4_cpu
build=/absolute/path/nvfp4-cpu-build
cmake -S "$src/bench" -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j4
```

This builds both benchmark executables in BUILD_DIR. Google Benchmark v1.9.4 uses the same pinned commit as the EXL3
benchmark; it is fetched if not installed. For offline builds, install its CMake package and set
`NVFP4_FETCH_BENCHMARK=OFF`, or point `FETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK` to a local checkout of that version.
The library itself and the correctness harnesses build from Python (`../optimized/README.md`).
````

- [ ] **Step 6: Commit, push, run the test on divix01**

```bash
git add python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py \
        python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/README.md \
        python/sglang/srt/layers/quantization/nvfp4_cpu/bench/README.md
git commit -m "$(cat <<'EOF'
build(nvfp4-cpu): build the kernel from Python (build.py, nvfp4_cpu_ext) and drop its CMake build

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
ssh divix01 'git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan fetch -q origin && git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan checkout -q --detach origin/nvfp4-cpu-forward-plan'
```

Run: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_build.py`
Expected: `4 passed`, `EXIT=0`. If the sanitizer case fails only on LeakSanitizer reports whose every frame is inside `libgomp`, ledger a ruling and add `env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}` for that case alone; any report with a frame in `moe_mul1.cpp`, `dot_nvfp4.h` or `nvfp4.c` is a real failure.

---

### Task 2: Bit-exact A/B harness, driver, and the baseline

**Files:**
- Create: `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp`
- Create: `test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh`

**Interfaces:**
- Consumes: `build.py --main`, `--portable`, `--upstream-baseline` (Task 1).
- Produces: `run_nvfp4_cpu_forward_checks.sh WORKTREE OUT [BASE_OUT]`, which prints `DUMPED <variant>` or `PASS <variant> bit-exact` / `FAIL ...` for variants `native`, `baseline`, `portable` and exits nonzero on any failure; the baseline dumps at `/mnt/nvme1/nvfp4-plan/base/{native,baseline,portable}.bin`.

- [ ] **Step 1: Write the harness**

`test/manual/dsv41/nvfp4_cpu_forward_ab.cpp`:

```cpp
// Bit-exact A/B harness for the NVFP4 CPU expert kernel; run_nvfp4_cpu_forward_checks.sh builds and runs it.
// It calls only the C ABI (cpu_experts_cabi.h), so the same source runs against any kernel revision. Every output and
// status is written to OUT in a fixed order: two revisions agree when their files are byte-identical.
//   nvfp4_cpu_forward_ab OUT CORE [CORE...]     worker i runs on CORE i; the caller is worker 0
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
struct Config {
    const char* name;
    int hidden, intermediate, layout;
    float limit, inv13, inv2;
    bool up_alpha;
};
// Ordered so each small layer reuses scratch a 5120-wide layer dirtied: the 80/144-wide layers' 64-column tails
// must still read zeros. dsv41* are DeepSeek V4.1's routed expert (the DSV4.1 plan); dsv41_nolimit is its shape
// without the clamp (the generic plan).
const Config kConfigs[] = {
    {"dsv41", 5120, 2304, 0, 10.f, 1.f, 1.f, false},
    {"h80_n80_l0", 80, 80, 0, 0.f, 1.f, 1.f, false},
    {"dsv41_l2_up_scaled", 5120, 2304, 2, 10.f, .5f, .25f, true},
    {"h80_n80_l1_lim", 80, 80, 1, 2.5f, .5f, .25f, true},
    {"dsv41_nolimit", 5120, 2304, 0, 0.f, 1.f, 1.f, false},
    {"h256_n192_l2", 256, 192, 2, 0.f, 1.f, 1.f, true},
    {"h144_n128_l0_lim", 144, 128, 0, 10.f, .75f, 1.f, false},
};
struct Routing {
    const char* name;
    std::vector<int32_t> slots;
    std::vector<float> weights;
};
const Routing kRoutings[] = {
    {"k0", {}, {}},
    {"k1", {3}, {.7f}},
    {"k3", {0, 5, 2}, {.5f, .3f, .2f}},
    {"k5_skip_dup_zero", {1, -1, 4, 6, 1}, {.4f, .9f, 0.f, .3f, .2f}},
    {"k8_negzero", {0, 1, 2, 3, 4, 5, 6, 7}, {.125f, -.25f, .5f, -0.f, .0625f, 1.f, .3f, .2f}},
};
constexpr int kCapacity = 8;
constexpr uint64_t kPad = 64;           // every weight/scale stride is its minimum plus this: strides are honored
constexpr uint64_t kAlphaStride = 8;    // fp32 alpha plus 4 bytes of padding

uint64_t rounded(uint64_t x, uint64_t n) { return (x + n - 1) / n * n; }

// A finite signed E4M3 scale in [2^-2, 2^1].
uint8_t scale(std::mt19937& rng) {
    const unsigned e = 5 + rng() % 4, m = rng() % 8, s = rng() & 1;
    return uint8_t(s << 7 | e << 3 | m);
}

struct Slabs {
    std::vector<uint8_t> bytes[7];
    uint64_t stride[7]{};
};

Slabs make_slabs(const Config& c, std::mt19937& rng) {
    const uint64_t h = c.hidden, n = c.intermediate;
    const uint64_t minimum[4] = {n * h, h * n / 2, rounded(2 * n, 128) * rounded(h / 16, 4),
                                 rounded(h, 128) * rounded(n / 16, 4)};
    Slabs s;
    for (int i = 0; i < 4; ++i) {
        s.stride[i] = minimum[i] + kPad;
        s.bytes[i].resize(s.stride[i] * kCapacity);
        for (auto& b : s.bytes[i]) b = i < 2 ? uint8_t(rng()) : scale(rng);
    }
    const float base[3] = {.01f, .015f, .02f};  // gate, down, up
    for (int i = 4; i < 7; ++i) {
        if (i == 6 && !c.up_alpha) continue;
        s.stride[i] = kAlphaStride;
        s.bytes[i].assign(kAlphaStride * kCapacity, 0);
        for (int slot = 0; slot < kCapacity; ++slot) {
            const float a = base[i - 4] * float(1 + slot);
            std::memcpy(s.bytes[i].data() + slot * kAlphaStride, &a, sizeof(a));
        }
    }
    return s;
}

void put(FILE* f, const std::string& name, int status, const std::vector<float>& out) {
    const uint32_t length = uint32_t(name.size()), count = uint32_t(out.size());
    std::fwrite(&length, 4, 1, f);
    std::fwrite(name.data(), 1, length, f);
    std::fwrite(&status, 4, 1, f);
    std::fwrite(&count, 4, 1, f);
    std::fwrite(out.data(), 4, count, f);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s OUT CORE [CORE...]\n", argv[0]); return 2; }
    std::vector<int32_t> cores;
    for (int i = 2; i < argc; ++i) cores.push_back(std::atoi(argv[i]));
    if (sglang_nvfp4_cpu_experts_set_cores(cores.data(), int32_t(cores.size())) != 0) {
        std::fprintf(stderr, "set_cores refused\n"); return 1;
    }
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) return 1;
    // Descending: the pre-OpenMP pool fixes its size at the first forward and refuses a larger later one.
    std::vector<int> teams = {int(cores.size())};
    for (int t : {3, 1}) if (t < teams.back()) teams.push_back(t);
    std::mt19937 rng(20261003);
    int cases = 0, unexpected = 0;
    for (const Config& c : kConfigs) {
        Slabs s = make_slabs(c, rng);
        SglangNvfp4CpuLayer d{};
        d.abi_version = 1; d.capacity = kCapacity; d.hidden = c.hidden; d.intermediate = c.intermediate;
        d.w13_layout = c.layout; d.activation = 0; d.act_limit = c.limit;
        d.inv_input_scale13 = c.inv13; d.inv_input_scale2 = c.inv2;
        for (int i = 0; i < 7; ++i) {
            d.slabs[i] = s.bytes[i].empty() ? nullptr : s.bytes[i].data();
            d.slot_bytes[i] = s.stride[i];
        }
        int64_t handle = -1;
        if (sglang_nvfp4_cpu_experts_register_slabs(&d, &handle) != 0) {
            std::fprintf(stderr, "%s: registration refused\n", c.name); return 1;
        }
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        std::vector<uint16_t> x(c.hidden);
        for (auto& v : x) v = ggml_compute_fp32_to_fp16(unit(rng));
        for (int threads : teams)
            for (const Routing& r : kRoutings)
                for (int accumulate : {0, 1}) {
                    std::vector<float> out(c.hidden);
                    for (int i = 0; i < c.hidden; ++i) out[i] = accumulate ? .001f * float(i) - .04f : 777.f;
                    const int status = sglang_nvfp4_cpu_experts_forward(
                        handle, x.data(), r.slots.empty() ? nullptr : r.slots.data(),
                        r.weights.empty() ? nullptr : r.weights.data(), int32_t(r.slots.size()), out.data(),
                        threads, accumulate);
                    unexpected += status != 0;
                    put(f, std::string(c.name) + "/t" + std::to_string(threads) + "/" + r.name + "/acc" +
                               std::to_string(accumulate), status, out);
                    ++cases;
                }
        // Refusals: an infinite input element, and a slot past the capacity. Both must leave out untouched.
        std::vector<uint16_t> bad = x;
        bad[7] = 0x7c00;
        const int32_t slot = 2, past = kCapacity;
        const float weight = .5f;
        std::vector<float> out(c.hidden, 777.f);
        int status = sglang_nvfp4_cpu_experts_forward(handle, bad.data(), &slot, &weight, 1, out.data(), teams[0], 0);
        unexpected += status != 2;
        put(f, std::string(c.name) + "/nonfinite_x", status, out);
        out.assign(c.hidden, 777.f);
        status = sglang_nvfp4_cpu_experts_forward(handle, x.data(), &past, &weight, 1, out.data(), teams[0], 0);
        unexpected += status != 2;
        put(f, std::string(c.name) + "/slot_past_capacity", status, out);
        cases += 2;
        if (sglang_nvfp4_cpu_experts_free_layer(handle) != 0) { std::fprintf(stderr, "free refused\n"); return 1; }
    }
    std::fclose(f);
    std::printf("%d cases, %d unexpected statuses\n", cases, unexpected);
    return unexpected ? 1 : 0;
}
```

- [ ] **Step 2: Write the driver**

`test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh`:

```bash
#!/usr/bin/env bash
# Bit-exact gate for a change to the NVFP4 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT              build every variant from WORKTREE and dump into OUT
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT BASE_OUT     ... then compare each dump bitwise with BASE_OUT's
# Variants: native (-march=native, the AVX2 dot), baseline (GGML conversion, the bench's baseline backend),
# portable (no -march=native, the scalar dot). CXX defaults to GCC 15; NVFP4_AB_CORES to 0-7 (NUMA node 0).
set -uo pipefail
wt=$(realpath "$1"); out=$2; base=${3:-}
cxx=${CXX:-/opt/rh/gcc-toolset-15/root/usr/bin/g++}
py=/data/models/slang/.venv/bin/python
cores=${NVFP4_AB_CORES:-0,1,2,3,4,5,6,7}
build=$wt/python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py
harness=$wt/test/manual/dsv41/nvfp4_cpu_forward_ab.cpp
mkdir -p "$out"
fail=0
for variant in native baseline portable; do
  flags=()
  case $variant in baseline) flags=(--upstream-baseline) ;; portable) flags=(--portable) ;; esac
  exe=$out/ab-$variant
  if ! taskset -c 0-63 "$py" "$build" --cxx "$cxx" --output "$exe" --main "$harness" "${flags[@]}" \
      > "$out/build-$variant.log" 2>&1; then
    echo "FAIL $variant: build (see $out/build-$variant.log)"; fail=1; continue
  fi
  if ! OMP_WAIT_POLICY=ACTIVE taskset -c "$cores" "$exe" "$out/$variant.bin" ${cores//,/ } \
      > "$out/run-$variant.log" 2>&1; then
    echo "FAIL $variant: run (see $out/run-$variant.log)"; fail=1; continue
  fi
  if [[ -z $base ]]; then
    echo "DUMPED $variant: $(cat "$out/run-$variant.log")"
  elif cmp -s "$out/$variant.bin" "$base/$variant.bin"; then
    echo "PASS $variant bit-exact"
  else
    echo "FAIL $variant differs from $base/$variant.bin"; fail=1
  fi
done
exit $fail
```

- [ ] **Step 3: Commit, push, create the base worktree, capture the baseline**

```bash
chmod +x test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh
git add test/manual/dsv41/nvfp4_cpu_forward_ab.cpp test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh
git commit -m "$(cat <<'EOF'
test(nvfp4-cpu): bit-exact A/B harness and driver for the CPU expert kernel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-nvfp4-plan-base origin/nvfp4-cpu-forward-plan \
  && git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan-base log -1 --oneline \
  && bash /data/models/slang/nvfp4-work/wt-nvfp4-plan-base/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh \
       /data/models/slang/nvfp4-work/wt-nvfp4-plan-base /mnt/nvme1/nvfp4-plan/base; echo EXIT=$?'
```

Expected: `DUMPED native: 224 cases, 0 unexpected statuses`, the same for `baseline` and `portable`, and `EXIT=0`. With 8 cores the teams are {8, 3, 1}, so each config runs 3 × 5 routings × 2 accumulate modes = 30 forwards plus 2 refusals, and 7 configs × 32 = 224.

- [ ] **Step 4: Prove the baseline is deterministic**

```bash
ssh divix01 'bash /data/models/slang/nvfp4-work/wt-nvfp4-plan-base/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh \
  /data/models/slang/nvfp4-work/wt-nvfp4-plan-base /mnt/nvme1/nvfp4-plan/base-rerun /mnt/nvme1/nvfp4-plan/base; echo EXIT=$?'
```

Expected: `PASS native bit-exact`, `PASS baseline bit-exact`, `PASS portable bit-exact`, `EXIT=0`. (A FAIL here means the old kernel is nondeterministic and the gate cannot work: stop and report.)

---

### Task 3: `LayerInfo`, `StridedExperts<Shape>`, `GenericShape` — the registry reads layers through them

Behavior-preserving: the gate is `ab_check`. The old `Workers` pool still runs the forward in this task.

**Files:**
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/layout.h`
- Create: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/experts.hpp`
- Create: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/shapes.hpp`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp` (`Layer`, `calculate`, `valid`, registration, forward's slot checks)

**Interfaces:**
- Produces (in `moe_mul1.cpp`'s anonymous namespace): `struct LayerInfo { int capacity, hidden, intermediate, w13_layout; float act_limit, inv_input_scale13, inv_input_scale2; bool up_alpha; }`; `enum SlabName { kW13, kW2, kSf13, kSf2, kGateAlpha, kDownAlpha, kUpAlpha, kSlabNames }`; `struct SlabRowBytes { uint64_t bytes[kSlabNames]; static constexpr SlabRowBytes of(int hidden, int intermediate); }`; `struct Projection { const uint8_t* w; const uint8_t* sf; float alpha; }`; `void w13_rows(int layout, int n, int i, int& gate, int& up)`; `template <class Shape> struct StridedExperts { const uint8_t* base[kSlabNames]; uint64_t stride[kSlabNames]; Projection gate(int) const; Projection up(int) const; Projection down(int) const; float alpha(SlabName, int) const; template <class Other> StridedExperts<Other> as() const; }`; `struct GenericShape { static int hidden(const LayerInfo&); static int intermediate(const LayerInfo&); static float act_limit(const LayerInfo&); }`; `LayerInfo info_of(const SglangNvfp4CpuLayer&)`; `StridedExperts<GenericShape> strided_of(const SglangNvfp4CpuLayer&)`; `float swiglu(float g, float u, float limit)`.

- [ ] **Step 1: Confirm the gate is green before the edit**

Run: `ab_check` with `<N>`=`3pre` (the branch worktree is at Task 2's commit).
Expected: three `PASS ... bit-exact`, `EXIT=0`.

- [ ] **Step 2: `rounded` becomes constexpr**

In `layout.h`, replace

```cpp
inline size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
```

with

```cpp
constexpr size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
```

- [ ] **Step 3: Write `experts.hpp`**

```cpp
// Included by moe_mul1.cpp inside its anonymous namespace, after cpu_experts_cabi.h and layout.h (rounded).
// The forward reads a layer through these: LayerInfo for the facts, StridedExperts for each slot's projections.
// Accessors are cheap to copy and allocate nothing.

// What a forward needs to know about a layer besides its slabs: SglangNvfp4CpuLayer's scalar fields.
struct LayerInfo
{
    int capacity;
    int hidden;               // columns of gate/up, rows of down
    int intermediate;         // rows of gate and of up, columns of down
    int w13_layout;           // 0 [gate, up], 1 [up, gate], 2 alternating 64-row [up, gate] chunks
    float act_limit;          // 0: no clamp
    float inv_input_scale13;  // cancels an activation scale folded into the GPU gate/up alphas
    float inv_input_scale2;   // the same for down
    bool up_alpha;            // slab kUpAlpha is registered; else up shares the gate alpha
};

// The descriptor's slabs, in cpu_experts_cabi.h order.
enum SlabName { kW13, kW2, kSf13, kSf2, kGateAlpha, kDownAlpha, kUpAlpha, kSlabNames };
static_assert(kW13 == 0 && kSf13 == 2 && kGateAlpha == 4 && kUpAlpha == 6 && kSlabNames == 7,
              "SlabName indexes SglangNvfp4CpuLayer::slabs");

// The fewest bytes one slot's row of each slab holds: packed E2M1 weights (two per byte), E4M3 scales in the GPU's
// 128x4 swizzle (rows padded to 128, scale groups to 4), one fp32 alpha. Registration refuses a smaller stride.
struct SlabRowBytes
{
    uint64_t bytes[kSlabNames];

    static constexpr SlabRowBytes of(int hidden, int intermediate)
    {
        const uint64_t h = uint64_t(hidden), n = uint64_t(intermediate);
        return {{n * h, h * n / 2, rounded(2 * n, 128) * rounded(h / 16, 4), rounded(h, 128) * rounded(n / 16, 4),
                 4, 4, 4}};
    }
};
// The sanitizer harness's 80 x 80 layer (test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp) registers exactly these.
static_assert(SlabRowBytes::of(80, 80).bytes[kW13] == 80 * 80 && SlabRowBytes::of(80, 80).bytes[kW2] == 80 * 80 / 2);
static_assert(SlabRowBytes::of(80, 80).bytes[kSf13] == 256 * 8 && SlabRowBytes::of(80, 80).bytes[kSf2] == 128 * 8);

// One projection of one slot: packed E2M1 rows, their swizzled E4M3 scales, and the slot's fp32 GPU GEMM alpha. Gate
// and up share the w13 rows (w13_rows maps an output to its row); down's rows are its own.
struct Projection
{
    const uint8_t* w;
    const uint8_t* sf;
    float alpha;
};

// The w13 rows holding gate and up output i of n, per LayerInfo::w13_layout.
inline void w13_rows(int layout, int n, int i, int& gate, int& up)
{
    gate = i;
    up = i + n;
    if (layout == 1) std::swap(gate, up);
    if (layout == 2) { up = (i / 64) * 128 + i % 64; gate = up + 64; }
}

// The descriptor registration: slot s of every slab at base + s * stride, nothing stored per slot. Strides are the
// registrant's (at least SlabRowBytes) under any Shape. Shape names the plan this view is checked for: run_plan makes
// a StridedExperts<Dsv41Shape> only after Dsv41Shape::accepts, and ForwardPlan<Shape, I> takes only its own Shape's
// view. The kernel keeps no reference to the slabs: the registrant keeps them alive.
template <class Shape>
struct StridedExperts
{
    const uint8_t* base[kSlabNames];  // base[kUpAlpha] may be null
    uint64_t stride[kSlabNames];

    Projection gate(int slot) const { return {at(kW13, slot), at(kSf13, slot), alpha(kGateAlpha, slot)}; }
    Projection up(int slot) const
    {
        return {at(kW13, slot), at(kSf13, slot), alpha(base[kUpAlpha] ? kUpAlpha : kGateAlpha, slot)};
    }
    Projection down(int slot) const { return {at(kW2, slot), at(kSf2, slot), alpha(kDownAlpha, slot)}; }

    // A slot's alpha, read on every call: slab rows change when a slot is reused.
    float alpha(SlabName name, int slot) const
    {
        float v;
        std::memcpy(&v, at(name, slot), sizeof(v));
        return v;
    }

    // The same slabs under another shape's assumptions (the caller has checked they hold).
    template <class Other>
    StridedExperts<Other> as() const
    {
        StridedExperts<Other> o;
        std::copy(std::begin(base), std::end(base), std::begin(o.base));
        std::copy(std::begin(stride), std::end(stride), std::begin(o.stride));
        return o;
    }

private:
    const uint8_t* at(SlabName name, int slot) const { return base[name] + size_t(slot) * stride[name]; }
};
```

- [ ] **Step 4: Write `shapes.hpp`**

```cpp
// Included by moe_mul1.cpp inside its anonymous namespace, after experts.hpp.
//
// The shapes a forward plan can be specialized for. A plan reads every layer fact a shape may fix through its Shape:
// GenericShape takes each from the layer's LayerInfo. The w13 row order and the alphas' input scales stay runtime
// facts of the descriptor under every shape.

struct GenericShape
{
    static int hidden(const LayerInfo& info) { return info.hidden; }
    static int intermediate(const LayerInfo& info) { return info.intermediate; }
    static float act_limit(const LayerInfo& info) { return info.act_limit; }
};
```

- [ ] **Step 5: Read layers through them in `moe_mul1.cpp`**

Add `#include <algorithm>`/`<iterator>` if not present (`<algorithm>` is). Directly after `#include "dot_nvfp4.h"` inside the namespace, add:

```cpp
#include "experts.hpp"
#include "shapes.hpp"
```

Replace the whole `struct Layer { ... };` with:

```cpp
LayerInfo info_of(const SglangNvfp4CpuLayer& d)
{
    return {d.capacity, d.hidden, d.intermediate, d.w13_layout, d.act_limit, d.inv_input_scale13,
            d.inv_input_scale2, d.slabs[kUpAlpha] != nullptr};
}

StridedExperts<GenericShape> strided_of(const SglangNvfp4CpuLayer& d)
{
    StridedExperts<GenericShape> e{};
    for (int i = 0; i < kSlabNames; ++i) {
        e.base[i] = static_cast<const uint8_t*>(d.slabs[i]);
        e.stride[i] = d.slot_bytes[i];
    }
    return e;
}

struct Layer {
    LayerInfo info;
    StridedExperts<GenericShape> strided;
    // Activations only. Sized once; mutable slab contents are read each job.
    std::vector<float> x, intermediate, result;
    std::vector<block_q8_0> qx, qi;
    std::vector<std::vector<block_nvfp4>> row_scratch;
    explicit Layer(const SglangNvfp4CpuLayer& d) : info(info_of(d)), strided(strided_of(d)), x(d.hidden),
        intermediate(d.intermediate), result(d.hidden),
        qx(rounded(d.hidden,64)/32), qi(rounded(d.intermediate,64)/32) {
        x.resize(rounded(d.hidden,64)); intermediate.resize(rounded(d.intermediate,64));
    }
};
```

Add, just above `struct Work`:

```cpp
// Gate/up output of the gated SiLU: optional pre-SiLU clamp (gate from above, up both ways), stable SiLU including
// large negative gates, times up.
inline float swiglu(float g, float u, float limit)
{
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
    return silu * u;
}
```

Replace `calculate` with:

```cpp
void calculate(void* p, int rank, int n) {
    auto& c = *static_cast<Work*>(p); auto& l = *c.layer; const LayerInfo& d = l.info;
    if (c.stage == 0) {
        const Projection gate = l.strided.gate(c.slot), up = l.strided.up(c.slot);
        const float gate_alpha = gate.alpha * d.inv_input_scale13;
        const float up_alpha = up.alpha * d.inv_input_scale13;
        for (int i = rank; i < d.intermediate; i += n) {
            int gate_row, up_row;
            w13_rows(d.w13_layout, d.intermediate, i, gate_row, up_row);
            const float g = dot(gate.w, gate.sf, gate_row, d.hidden, l.qx.data(), l.row_scratch[rank].data()) * gate_alpha;
            const float u = dot(up.w, up.sf, up_row, d.hidden, l.qx.data(), l.row_scratch[rank].data()) * up_alpha;
            l.intermediate[i] = swiglu(g, u, d.act_limit);
        }
    } else {
        const Projection down = l.strided.down(c.slot);
        const float alpha = down.alpha * d.inv_input_scale2 * c.route;
        for (int i = rank; i < d.hidden; i += n)
            l.result[i] += dot(down.w, down.sf, i, d.intermediate, l.qi.data(), l.row_scratch[rank].data()) * alpha;
    }
}
```

In `valid`, replace the `minimum` array and its loop with:

```cpp
    const SlabRowBytes minimum = SlabRowBytes::of(d.hidden, d.intermediate);
    for (int i = 0; i < kSlabNames; ++i) {
        if (i == kUpAlpha && !d.slabs[i]) continue;
        if (!d.slabs[i] || d.slot_bytes[i] < minimum.bytes[i]
            || d.slot_bytes[i] > SIZE_MAX / uint64_t(d.capacity)) return false;
    }
    return true;
```

In `sglang_nvfp4_cpu_experts_forward`, replace the per-slot check loop with:

```cpp
        const auto& E = l->strided;
        for (int j = 0; j < k; ++j) {
            if (slots[j] < -1 || slots[j] >= l->info.capacity || !std::isfinite(weights[j])) return 2;
            if (slots[j] >= 0 && (!std::isfinite(E.alpha(kGateAlpha, slots[j]))
                || !std::isfinite(E.alpha(kDownAlpha, slots[j]))
                || (l->info.up_alpha && !std::isfinite(E.alpha(kUpAlpha, slots[j]))))) return 2;
        }
```

and replace every remaining `l->d.hidden` / `l->d.intermediate` in that function with `l->info.hidden` / `l->info.intermediate`.

- [ ] **Step 6: Commit, push, run both gates**

```bash
git add python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/layout.h \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/experts.hpp \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/shapes.hpp \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp
git commit -m "$(cat <<'EOF'
refactor(nvfp4-cpu): read layers through LayerInfo and StridedExperts<Shape> (experts.hpp, shapes.hpp)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
ssh divix01 'git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan fetch -q origin && git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan checkout -q --detach origin/nvfp4-cpu-forward-plan'
```

Run: `ab_check` with `<N>`=`3`. Expected: three `PASS ... bit-exact`, `EXIT=0`.
Run: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_build.py`. Expected: `4 passed`, `EXIT=0`.

---

### Task 4: `ForwardPlan<GenericShape, kBuildIsa>` on an OpenMP team; the `Workers` pool removed

**Files:**
- Create: `test/registered/unit/kernels/test_nvfp4_cpu_experts.py`
- Create: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/forward_plan.hpp`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp` (rewritten around the plan; full text below)
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/bench/CMakeLists.txt`

**Interfaces:**
- Consumes: everything Task 3 produces.
- Produces: `enum class Isa { Scalar, Avx2 }`, `constexpr Isa kBuildIsa`; `enum class Phase { PrepareInput, GateUp, Middle, Down }`; `template <class Shape, Isa I> struct PlanTraits { static constexpr int kRowUnit; }`; `template <int Unit> std::pair<int64_t,int64_t> share(int64_t total, int worker, int workers)`; `template <class Shape, Isa I> struct ForwardPlan { static int run(ForwardCtx&, const StridedExperts<Shape>&, ForwardArena&, int threads); }` returning 0 or 2, throwing on team/pin failure; `struct Route { int slot; float weight; }`; `struct ForwardCtx`, `struct ForwardArena`; `int run_plan(ForwardCtx&, const RegisteredLayer&, int threads)`; `freeze_compute_cores()`, `pin_compute_worker(int, std::atomic<int>&)`.

- [ ] **Step 1: Write the failing tests**

`test/registered/unit/kernels/test_nvfp4_cpu_experts.py`:

```python
"""The NVFP4 CPU expert C ABI under its OpenMP team (Linux, GCC with OpenMP).

Each case runs in its own process: core configuration freezes at a process's first forward, and the OpenMP
environment is read when the team first forms. The child registers the sanitizer harness's 80 x 80 layer (every
weight nibble 1.0, every scale 1.0) and prints one line per call.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CXX = os.environ.get("CXX") or shutil.which("g++")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or CXX is None, reason="the kernel needs Linux and a GCC with OpenMP"
)

CHILD = r"""
import ctypes, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1])
calls = [int(t) for t in sys.argv[2].split(",")]
cores = [int(c) for c in sys.argv[3].split(",")] if sys.argv[3] else []
alpha_value = float(sys.argv[4])


class Layer(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("capacity", ctypes.c_int32), ("hidden", ctypes.c_int32),
        ("intermediate", ctypes.c_int32), ("w13_layout", ctypes.c_int32), ("activation", ctypes.c_int32),
        ("act_limit", ctypes.c_float), ("inv_input_scale13", ctypes.c_float), ("inv_input_scale2", ctypes.c_float),
        ("slabs", ctypes.c_void_p * 7), ("slot_bytes", ctypes.c_uint64 * 7),
    ]


H = N = 80
CAP = 2
w13 = np.full(CAP * N * H, 0x22, np.uint8)
w2 = np.full(CAP * H * N // 2, 0x22, np.uint8)
sf13 = np.full(CAP * 256 * 8, 56, np.uint8)
sf2 = np.full(CAP * 128 * 8, 56, np.uint8)
alpha = np.full(CAP, alpha_value, np.float32)
d = Layer(abi_version=1, capacity=CAP, hidden=H, intermediate=N, inv_input_scale13=1.0, inv_input_scale2=1.0)
for i, (slab, stride) in enumerate([(w13, N * H), (w2, H * N // 2), (sf13, 256 * 8), (sf2, 128 * 8), (alpha, 4), (alpha, 4)]):
    d.slabs[i] = slab.ctypes.data
    d.slot_bytes[i] = stride
handle = ctypes.c_int64(-1)
assert lib.sglang_nvfp4_cpu_experts_register_slabs(ctypes.byref(d), ctypes.byref(handle)) == 0
core_array = (ctypes.c_int32 * max(len(cores), 1))(*cores)
if cores:
    print("cores", lib.sglang_nvfp4_cpu_experts_set_cores(core_array, len(cores)))
x = np.full(H, 0x3C00, np.uint16)  # fp16 1.0
slots = np.zeros(1, np.int32)
weights = np.ones(1, np.float32)
for threads in calls:
    out = np.full(H, 123.0, np.float32)
    status = lib.sglang_nvfp4_cpu_experts_forward(
        handle, x.ctypes.data_as(ctypes.c_void_p), slots.ctypes.data_as(ctypes.c_void_p),
        weights.ctypes.data_as(ctypes.c_void_p), 1, out.ctypes.data_as(ctypes.c_void_p), threads, 0)
    print("forward", threads, status, "untouched" if (out == 123.0).all() else "written")
if cores:
    print("cores-after", lib.sglang_nvfp4_cpu_experts_set_cores(core_array, len(cores)))
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nvfp4_cpu_build", REPO / "python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build.build(tmp_path_factory.mktemp("nvfp4") / "libnvfp4.so", cxx=CXX)


def _run(library, calls, cores=(), alpha=1.0, **env):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(library), calls, ",".join(map(str, cores)), str(alpha)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.split("\n")[:-1]


def _allowed(count):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < count:
        pytest.skip(f"needs {count} allowed CPUs")
    return allowed[:count]


def test_a_later_forward_may_use_more_workers_than_the_first(library):
    assert _run(library, "1,3") == ["forward 1 0 written", "forward 3 0 written"]


def test_a_team_smaller_than_requested_fails_the_forward_and_leaves_out_untouched(library):
    assert _run(library, "4", OMP_THREAD_LIMIT="2") == ["forward 4 1 untouched"]


def test_more_workers_than_configured_cores_fails_the_forward(library):
    # Only the first two lines: whether cores can still change after this refused forward differs between the old
    # pool (it refused before starting, so yes) and the OpenMP team (the cores froze at the attempt), and neither is
    # a contract.
    assert _run(library, "2", cores=_allowed(1))[:2] == ["cores 0", "forward 2 1 untouched"]


def test_cores_cannot_change_after_the_first_forward(library):
    assert _run(library, "2", cores=_allowed(2)) == ["cores 0", "forward 2 0 written", "cores-after 2"]


def test_an_intermediate_q8_cannot_represent_returns_2_and_leaves_out_untouched(library):
    # gate = up = 80 * 1e30, so SiLU(gate) * up overflows to inf before the down projection's quantization.
    assert _run(library, "2", alpha=1e30) == ["forward 2 2 untouched"]
```

- [ ] **Step 2: Commit, push, run them to see the two new behaviors fail**

```bash
git add test/registered/unit/kernels/test_nvfp4_cpu_experts.py
git commit -m "$(cat <<'EOF'
test(nvfp4-cpu): the C ABI under an OpenMP team: any team size up to the cores, a short team fails (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
ssh divix01 'git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan fetch -q origin && git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan checkout -q --detach origin/nvfp4-cpu-forward-plan'
```

Run: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_experts.py`
Expected: `2 failed, 3 passed`, `EXIT=1`. The failures: `test_a_later_forward_may_use_more_workers_than_the_first` (old pool: `forward 3 1 untouched`) and `test_a_team_smaller_than_requested_fails_the_forward_and_leaves_out_untouched` (old pool ignores OpenMP: `forward 4 0 written`). The three passing tests are characterization: they pin today's behavior that the rewrite must keep.

- [ ] **Step 3: Write `forward_plan.hpp`**

```cpp
// Included by moe_mul1.cpp inside its anonymous namespace, after the arithmetic (dot, swiglu, q8_representable,
// quantize_block), the worker-core helpers (freeze_compute_cores, pin_compute_worker) and ForwardCtx/ForwardArena.
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is the
// dot product's tier, fixed when the library is compiled (kBuildIsa). The primary PlanTraits is the generic plan's.

// The forward's phases, in team order; a barrier separates each from the next.
enum class Phase : int
{
    PrepareInput = 0,  // fp16 input to fp32, then its Q8_0 blocks
    GateUp = 1,        // every route's gate/up rows, gated SiLU into its intermediate
    Middle = 2,        // every route's intermediate to Q8_0 blocks
    Down = 3,          // every route's down rows, routing-weighted sum into out
};

// What a (Shape, ISA) plan sets. The primary template is the generic plan.
template <class Shape, Isa I>
struct PlanTraits
{
    static constexpr int kRowUnit = 16;  // output rows per split unit: 16 fp32 outputs fill one cache line
};

// Worker `worker`'s contiguous share [first, last) of `total` items, cut on multiples of Unit.
template <int Unit>
std::pair<int64_t, int64_t> share(int64_t total, int worker, int workers)
{
    const int64_t units = (total + Unit - 1) / Unit;
    return {std::min(total, units * worker / workers * Unit), std::min(total, units * (worker + 1) / workers * Unit)};
}

template <class Shape, Isa I>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape, I>;

    // Runs ctx's routes through E on `threads` workers (the caller is worker 0). Returns 0, or 2 when Q8_0 cannot
    // represent the input or an intermediate (out is then untouched). Throws when the team is short or a worker
    // cannot be pinned.
    static int run(ForwardCtx& ctx, const StridedExperts<Shape>& E, ForwardArena& ar, int threads)
    {
        bind_routes(ctx, E);
        prepare_scratch(ctx, ar, threads);
        run_team(ctx, threads);
        return ctx.invalid.load(std::memory_order_relaxed) ? 2 : 0;
    }

private:
    // Each route's projections and scaled alphas, in routing order; the team reads only these.
    static void bind_routes(ForwardCtx& c, const StridedExperts<Shape>& E)
    {
        for (int r = 0; r < c.routes; ++r) {
            const int slot = c.route[r].slot;
            c.gate[r] = E.gate(slot);
            c.up[r] = E.up(slot);
            c.down[r] = E.down(slot);
            c.gate_alpha[r] = c.gate[r].alpha * c.info.inv_input_scale13;
            c.up_alpha[r] = c.up[r].alpha * c.info.inv_input_scale13;
            c.down_alpha[r] = c.down[r].alpha * c.info.inv_input_scale2 * c.route[r].weight;
        }
    }

    // Sizes this call's scratch from the arena. The padded tails are zeroed on every call: the arena is shared by
    // every layer this thread forwards, and Q8_0 blocks and the dot product read whole 64-value blocks.
    static void prepare_scratch(ForwardCtx& c, ForwardArena& ar, int threads)
    {
        const size_t H = size_t(Shape::hidden(c.info)), N = size_t(Shape::intermediate(c.info));
        const size_t Hp = rounded(H, 64), Np = rounded(N, 64), routes = size_t(c.routes);
        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(ar.xf, Hp);
        grow(ar.qx, Hp / 32);
        grow(ar.inter, routes * Np);
        grow(ar.qi, routes * Np / 32);
        std::fill(ar.xf.begin() + H, ar.xf.begin() + Hp, 0.f);
        for (size_t r = 0; r < routes; ++r)
            std::fill(ar.inter.begin() + r * Np + N, ar.inter.begin() + (r + 1) * Np, 0.f);
        c.xf = ar.xf.data();
        c.qx = ar.qx.data();
        c.inter = ar.inter.data();
        c.qi = ar.qi.data();
#if defined(NVFP4_CPU_UPSTREAM_BASELINE)
        c.row_scratch_stride = std::max(Hp, Np) / 64;
        grow(ar.row_scratch, size_t(threads) * c.row_scratch_stride);
        c.row_scratch = ar.row_scratch.data();
#else
        (void)threads;
        c.row_scratch = nullptr;
        c.row_scratch_stride = 0;
#endif
    }

    static void run_team(ForwardCtx& ctx, int count)
    {
        freeze_compute_cores();
        if (!g_compute_cores.empty() && size_t(count) > g_compute_cores.size())
            throw std::runtime_error("CPU expert worker count exceeds configured cores");
        std::atomic<int> pin_error{0};
        std::atomic<int> actual_workers{0};
        #pragma omp parallel num_threads(count) shared(ctx, pin_error, actual_workers)
        {
            const int worker = omp_get_thread_num(), n = omp_get_num_threads();
            if (worker == 0) actual_workers.store(n, std::memory_order_relaxed);
            pin_compute_worker(worker, pin_error);
            if (n == count) {
                step<Phase::PrepareInput>(ctx, worker, n);
                // `invalid` is read only after a barrier, so every worker takes the same branch.
                if (!ctx.invalid.load(std::memory_order_relaxed)) {
                    step<Phase::GateUp>(ctx, worker, n);
                    step<Phase::Middle>(ctx, worker, n);
                    if (!ctx.invalid.load(std::memory_order_relaxed)) phase<Phase::Down>(ctx, worker, n);
                }
            }
        }
        if (pin_error.load()) throw std::runtime_error("cannot pin CPU expert worker to its configured core");
        if (actual_workers.load() != count)
            throw std::runtime_error("OpenMP returned fewer CPU expert workers than requested");
    }

    // One phase, then wait for the whole team. Called inside run_team's parallel region (an orphaned barrier binds to
    // that team).
    template <Phase P>
    static void step(ForwardCtx& c, int worker, int workers)
    {
        phase<P>(c, worker, workers);
        #pragma omp barrier
    }

    // One phase for this worker; P picks the phase at compile time.
    template <Phase P>
    static void phase(ForwardCtx& c, int worker, int workers)
    {
        const int H = Shape::hidden(c.info), N = Shape::intermediate(c.info);
        const size_t Hp = rounded(size_t(H), 64), Np = rounded(size_t(N), 64);
        [[maybe_unused]] block_nvfp4* scratch =
            c.row_scratch ? c.row_scratch + size_t(worker) * c.row_scratch_stride : nullptr;

        if constexpr (P == Phase::PrepareInput) {
            const auto [b0, b1] = share<1>(int64_t(Hp / 32), worker, workers);
            for (int64_t b = b0; b < b1; ++b) {
                for (int64_t i = b * 32; i < std::min<int64_t>(b * 32 + 32, H); ++i) {
                    uint16_t v;
                    std::memcpy(&v, c.x + 2 * i, 2);
                    c.xf[i] = ggml_compute_fp16_to_fp32(v);
                }
                if (!q8_representable(c.xf + b * 32)) c.invalid.store(true, std::memory_order_relaxed);
                else quantize_block(c.xf + b * 32, c.qx[b]);
            }
        } else if constexpr (P == Phase::GateUp) {
            const auto [r0, r1] = share<Traits::kRowUnit>(int64_t(c.routes) * N, worker, workers);
            for (int64_t row = r0; row < r1; ++row) {
                const int r = int(row / N), i = int(row % N);
                int gate_row, up_row;
                w13_rows(c.info.w13_layout, N, i, gate_row, up_row);
                const float g = dot(c.gate[r].w, c.gate[r].sf, gate_row, H, c.qx, scratch) * c.gate_alpha[r];
                const float u = dot(c.up[r].w, c.up[r].sf, up_row, H, c.qx, scratch) * c.up_alpha[r];
                c.inter[size_t(r) * Np + size_t(i)] = swiglu(g, u, Shape::act_limit(c.info));
            }
        } else if constexpr (P == Phase::Middle) {
            // Route r's intermediate is Np floats at r * Np, so block b of the flat range is route b / (Np / 32)'s.
            const auto [b0, b1] = share<1>(int64_t(c.routes) * int64_t(Np / 32), worker, workers);
            for (int64_t b = b0; b < b1; ++b) {
                const float* v = c.inter + b * 32;
                if (!q8_representable(v)) c.invalid.store(true, std::memory_order_relaxed);
                else quantize_block(v, c.qi[b]);
            }
        } else {
            static_assert(P == Phase::Down);
            const auto [h0, h1] = share<Traits::kRowUnit>(H, worker, workers);
            for (int64_t h = h0; h < h1; ++h) {
                float sum = 0.f;
                for (int r = 0; r < c.routes; ++r)
                    sum += dot(c.down[r].w, c.down[r].sf, int(h), N, c.qi + size_t(r) * (Np / 32), scratch)
                           * c.down_alpha[r];
                c.out[h] = c.accumulate ? c.out[h] + sum : sum;
            }
        }
    }
};
```

- [ ] **Step 4: Rewrite `moe_mul1.cpp` around the plan**

Replace the whole file with:

```cpp
// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is ForwardPlan<Shape, Isa>::run (forward_plan.hpp) on one OpenMP team per call; layers are read through
// LayerInfo and StridedExperts (experts.hpp) under a Shape (shapes.hpp).
#if !defined(__linux__) || !defined(_OPENMP)
#error The NVFP4 CPU expert kernel requires Linux and OpenMP.
#endif
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__AVX2__) && !defined(NVFP4_CPU_FORCE_SCALAR)
#include <immintrin.h>
#endif

namespace {
#include "dot_nvfp4.h"

// The dot product's tier is fixed when the library is compiled (dot_nvfp4.h's #if chain): AVX2 under -march=native
// on an AVX2 host, else the scalar loop.
enum class Isa { Scalar, Avx2 };
#if defined(__AVX2__)
constexpr Isa kBuildIsa = Isa::Avx2;
#else
constexpr Isa kBuildIsa = Isa::Scalar;
#endif

#include "experts.hpp"
#include "shapes.hpp"

// -------------------------------------------------------------------------------------------
//   Layer registry
// -------------------------------------------------------------------------------------------

LayerInfo info_of(const SglangNvfp4CpuLayer& d)
{
    return {d.capacity, d.hidden, d.intermediate, d.w13_layout, d.act_limit, d.inv_input_scale13,
            d.inv_input_scale2, d.slabs[kUpAlpha] != nullptr};
}

StridedExperts<GenericShape> strided_of(const SglangNvfp4CpuLayer& d)
{
    StridedExperts<GenericShape> e{};
    for (int i = 0; i < kSlabNames; ++i) {
        e.base[i] = static_cast<const uint8_t*>(d.slabs[i]);
        e.stride[i] = d.slot_bytes[i];
    }
    return e;
}

struct RegisteredLayer
{
    LayerInfo info;
    StridedExperts<GenericShape> strided;
};
std::mutex registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<const RegisteredLayer>> layers;
int64_t next_handle = 1;
std::shared_ptr<const RegisteredLayer> lookup(int64_t h) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = layers.find(h); return it == layers.end() ? nullptr : it->second;
}

// One forward, core configuration or free at a time: the others return 3 rather than race the forward's scratch.
std::mutex forward_mutex;

// -------------------------------------------------------------------------------------------
//   Worker cores
// -------------------------------------------------------------------------------------------

// Worker cores set by sglang_nvfp4_cpu_experts_set_cores; copied into g_compute_cores at the first forward.
std::mutex g_cores_mutex;
std::vector<int> g_configured_cores;
std::atomic<bool> g_compute_started{false};
std::vector<int> g_compute_cores;  // Immutable after release publication at first forward.

// Freezes the configured cores into g_compute_cores once, at the first forward. Steady-state calls acquire no mutex.
inline void freeze_compute_cores()
{
    if (g_compute_started.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_cores_mutex);
    if (!g_compute_started.load(std::memory_order_relaxed)) {
        g_compute_cores = g_configured_cores;
        g_compute_started.store(true, std::memory_order_release);
    }
}

// Inside a parallel region: pins OpenMP worker `worker` to its compute core (none configured: no-op), setting
// pin_error if it cannot.
inline void pin_compute_worker(int worker, std::atomic<int>& pin_error)
{
    if (g_compute_cores.empty()) return;
    const int core = g_compute_cores[worker];
    static thread_local int pinned_core = -1;
    if (pinned_core != core || sched_getcpu() != core) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
            pin_error.store(1, std::memory_order_relaxed);
        else pinned_core = core;
    }
}

// -------------------------------------------------------------------------------------------
//   Arithmetic (unchanged from the pre-OpenMP kernel)
// -------------------------------------------------------------------------------------------

float dot(const uint8_t* w, const uint8_t* sf, int row, int k,
          const block_q8_0* x, block_nvfp4* scratch) {
    GpuRow view(w,sf,row,k);
    const int padded=int(rounded(k,64));
#if defined(NVFP4_CPU_UPSTREAM_BASELINE)
    // Convert into worker-local scratch on every row; the source slabs may mutate.
    for (int ib=0;ib<padded/64;++ib) {
        auto& block=scratch[ib];
        for (int g=0;g<4;++g) {
            int group=ib*4+g;
            uint8_t scale=group<k/16?sf[sf_index(row,group,k/16)]:0;
            block.d[g]=scale&127;
            const auto q=view.bytes(ib)+g*8;
            const uint8_t flip=scale&128?8:0;
            for (int j=0;j<8;++j) {
                // GGML puts columns j and j+8 in a byte; GPU puts 2j and 2j+1.
                uint8_t lo=(q[j/2]>>(4*(j%2)))&15;
                uint8_t hi=(q[(j+8)/2]>>(4*((j+8)%2)))&15;
                block.qs[g*8+j]=(lo^flip)|((hi^flip)<<4);
            }
        }
    }
    float result;
    ggml_vec_dot_nvfp4_q8_0(padded,&result,0,scratch,0,x,0,1);
    return result;
#else
    (void)scratch;
    return dot_gpu(padded,view,x);
#endif
}

// Whether Q8_0 represents the 32 values of a block: finite, with a delta within FP16 range.
bool q8_representable(const float* v)
{
    for (int j = 0; j < 32; ++j)
        if (!std::isfinite(v[j]) || std::abs(v[j]) > 65504.f * 127.f) return false;
    return true;
}

// One Q8_0 block. A zero FP16 delta contributes zero; this also avoids overflowing the reciprocal for a subnormal
// FP32 amax in the upstream reference quantizer.
void quantize_block(const float* v, block_q8_0& out)
{
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(v[j]));
    if (!ggml_compute_fp32_to_fp16(amax / 127.f)) out = block_q8_0{};
    else quantize_row_q8_0_ref(v, &out, 32);
}

// Gate/up output of the gated SiLU: optional pre-SiLU clamp (gate from above, up both ways), stable SiLU including
// large negative gates, times up.
inline float swiglu(float g, float u, float limit)
{
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
    return silu * u;
}

// -------------------------------------------------------------------------------------------
//   Forward context and scratch
// -------------------------------------------------------------------------------------------

constexpr int kMaxRoutes = 8;  // the C ABI's k limit

// 64-byte-aligned storage, so each kRowUnit share of fp32 outputs owns whole cache lines.
template <class T>
struct CacheAligned
{
    using value_type = T;
    CacheAligned() = default;
    template <class U> CacheAligned(const CacheAligned<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t{64}); }
    template <class U> bool operator==(const CacheAligned<U>&) const { return true; }
};

struct Route
{
    int slot;
    float weight;
};

struct ForwardCtx
{
    LayerInfo info;
    const uint8_t* x;  // fp16 [hidden], possibly unaligned
    float* out;        // fp32 [hidden]
    bool accumulate;
    Route route[kMaxRoutes];  // live routes (no -1 slot, no zero weight), in routing order
    int routes = 0;
    // Bound by the plan from the layer's experts, per route.
    Projection gate[kMaxRoutes], up[kMaxRoutes], down[kMaxRoutes];
    float gate_alpha[kMaxRoutes], up_alpha[kMaxRoutes], down_alpha[kMaxRoutes];
    // Scratch from the calling thread's ForwardArena.
    float* xf;                  // [rounded(hidden, 64)]
    block_q8_0* qx;             // [rounded(hidden, 64) / 32]
    float* inter;               // [routes][rounded(intermediate, 64)]
    block_q8_0* qi;             // [routes][rounded(intermediate, 64) / 32]
    block_nvfp4* row_scratch;   // [workers][row_scratch_stride], upstream-baseline builds only
    size_t row_scratch_stride;
    // Q8_0 cannot represent the input or an intermediate: the forward returns 2 and leaves out untouched.
    std::atomic<bool> invalid{false};
};

struct ForwardArena
{
    std::vector<float, CacheAligned<float>> xf, inter;
    std::vector<block_q8_0> qx, qi;
    std::vector<block_nvfp4> row_scratch;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

#include "forward_plan.hpp"

// Runs the call's plan.
int run_plan(ForwardCtx& ctx, const RegisteredLayer& layer, int threads)
{
    return ForwardPlan<GenericShape, kBuildIsa>::run(ctx, layer.strided, ForwardArena::get(), threads);
}

bool valid(const SglangNvfp4CpuLayer& d) {
    if (d.abi_version != 1 || d.capacity < 1 || d.hidden < 16 || d.intermediate < 16
        || d.hidden > (1 << 20) || d.intermediate > (1 << 20)
        || d.hidden % 16 || d.intermediate % 16 || d.w13_layout < 0 || d.w13_layout > 2
        || (d.w13_layout == 2 && d.intermediate % 64) || d.activation != 0
        || !std::isfinite(d.act_limit) || d.act_limit < 0
        || !std::isfinite(d.inv_input_scale13) || d.inv_input_scale13 <= 0
        || !std::isfinite(d.inv_input_scale2) || d.inv_input_scale2 <= 0) return false;
    const SlabRowBytes minimum = SlabRowBytes::of(d.hidden, d.intermediate);
    for (int i = 0; i < kSlabNames; ++i) {
        if (i == kUpAlpha && !d.slabs[i]) continue;
        if (!d.slabs[i] || d.slot_bytes[i] < minimum.bytes[i]
            || d.slot_bytes[i] > SIZE_MAX / uint64_t(d.capacity)) return false;
    }
    return true;
}
} // namespace

extern "C" int sglang_nvfp4_cpu_experts_register_slabs(const SglangNvfp4CpuLayer* d, int64_t* h) noexcept {
    if (!d || !h || !valid(*d)) return 2;
    try {
        auto l = std::make_shared<const RegisteredLayer>(RegisteredLayer{info_of(*d), strided_of(*d)});
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (next_handle == INT64_MAX) return 1;
        const auto id = next_handle++; layers.emplace(id, std::move(l)); *h = id; return 0;
    } catch (...) { return 1; }
}

extern "C" int sglang_nvfp4_cpu_experts_free_layer(int64_t h) noexcept {
    try {
        std::unique_lock<std::mutex> forward_lock(forward_mutex, std::try_to_lock);
        if (!forward_lock.owns_lock()) return 3;
        std::lock_guard<std::mutex> lock(registry_mutex); return layers.erase(h) ? 0 : 2;
    }
    catch (...) { return 1; }
}

// Worker i on cores[i], the caller as worker 0. Cores must be distinct and allowed by this thread's affinity; refused
// (2) once the first forward has frozen them.
extern "C" int sglang_nvfp4_cpu_experts_set_cores(const int32_t* c, int32_t n) noexcept {
    try {
        std::unique_lock<std::mutex> forward_lock(forward_mutex, std::try_to_lock);
        if (!forward_lock.owns_lock()) return 3;
        if (!c || n < 1 || n > 4096) return 2;
        cpu_set_t allowed; CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed)) return 2;
        for (int i = 0; i < n; ++i) {
            if (c[i] < 0 || c[i] >= CPU_SETSIZE || !CPU_ISSET(c[i], &allowed)) return 2;
            for (int j = 0; j < i; ++j) if (c[j] == c[i]) return 2;
        }
        std::lock_guard<std::mutex> lock(g_cores_mutex);
        if (g_compute_started.load(std::memory_order_relaxed)) return 2;
        g_configured_cores.assign(c, c + n);
        return 0;
    } catch (...) { return 1; }
}

extern "C" int sglang_nvfp4_cpu_experts_forward(int64_t h, const void* x,
    const int32_t* slots, const float* weights, int32_t k, float* out, int32_t threads, int32_t accumulate) noexcept {
    try {
        if (!x || !out || k < 0 || k > kMaxRoutes || threads < 1 || threads > 4096
            || (k && (!slots || !weights)) || (accumulate != 0 && accumulate != 1)) return 2;
        std::unique_lock<std::mutex> lock(forward_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return 3;
        const auto l = lookup(h); if (!l) return 2;
        const auto& E = l->strided;
        for (int j = 0; j < k; ++j) {
            if (slots[j] < -1 || slots[j] >= l->info.capacity || !std::isfinite(weights[j])) return 2;
            if (slots[j] >= 0 && (!std::isfinite(E.alpha(kGateAlpha, slots[j]))
                || !std::isfinite(E.alpha(kDownAlpha, slots[j]))
                || (l->info.up_alpha && !std::isfinite(E.alpha(kUpAlpha, slots[j]))))) return 2;
        }
        ForwardCtx ctx;
        ctx.info = l->info;
        ctx.x = static_cast<const uint8_t*>(x);
        ctx.out = out;
        ctx.accumulate = accumulate != 0;
        for (int j = 0; j < k; ++j) {
            if (slots[j] == -1 || weights[j] == 0) continue;
            ctx.route[ctx.routes++] = {slots[j], weights[j]};
        }
        return run_plan(ctx, *l, threads);
    } catch (...) { return 1; }
}
```

In `cpu_experts_cabi.h`, change the comment `// Configure once before first forward. Worker 0 is the calling engine thread.` to `// Configure before the first forward (refused with 2 after it). Worker i runs on cores[i]; worker 0 is the calling engine thread.` (no signature change).

- [ ] **Step 5: The bench builds against OpenMP**

In `nvfp4_cpu/bench/CMakeLists.txt`:
- Change `set(CMAKE_CXX_STANDARD 17)` to `set(CMAKE_CXX_STANDARD 20)`.
- After `find_package(Threads REQUIRED)`, add `find_package(OpenMP REQUIRED COMPONENTS CXX)`.
- Inside the `foreach`, after the `add_library(... OBJECT ...)` line, add `target_link_libraries(nvfp4_kernel_${backend} PRIVATE OpenMP::OpenMP_CXX)`.
- Change the executable's link line to `target_link_libraries(nvfp4_cpu_${backend} PRIVATE benchmark::benchmark Threads::Threads OpenMP::OpenMP_CXX)`.

`nvfp4.c` needs nothing further: the OBJECT library already compiles it with the project's C compiler (`CMAKE_C_STANDARD 11`).

- [ ] **Step 6: Commit, push, run every gate**

```bash
git add python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/forward_plan.hpp \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/cpu_experts_cabi.h \
        python/sglang/srt/layers/quantization/nvfp4_cpu/bench/CMakeLists.txt
git commit -m "$(cat <<'EOF'
refactor(nvfp4-cpu): run the forward as ForwardPlan<Shape, Isa> on one OpenMP team; drop the Workers pool

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
ssh divix01 'git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan fetch -q origin && git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan checkout -q --detach origin/nvfp4-cpu-forward-plan'
```

Run: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_nvfp4_cpu_build.py`
Expected: `9 passed`, `EXIT=0`.
Run: `ab_check` with `<N>`=`4`. Expected: three `PASS ... bit-exact`, `EXIT=0`.
Run (the bench still configures, builds and validates):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-nvfp4-plan && b=/mnt/nvme1/nvfp4-plan/bench-task4 \
  && taskset -c 0-63 cmake -S python/sglang/srt/layers/quantization/nvfp4_cpu/bench -B $b -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ -DCMAKE_C_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/gcc > $b.configure.log 2>&1 \
  && taskset -c 0-63 cmake --build $b -j16 > $b.build.log 2>&1 \
  && for be in baseline optimized; do taskset -c 0-7 $b/nvfp4_cpu_$be --cpus=0-7 --numa-node=0 --workers=8 --validate-only; echo "$be EXIT=$?"; done'
```

Expected: each prints `Q8 dense reference verified; individually pinned CPUs: 0 1 2 3 4 5 6 7 ...` and `EXIT=0`.

---

### Task 5: `Dsv41Shape` and the DSV4.1 plan `ForwardPlan<Dsv41Shape, Isa::Avx2>`

Behavior-preserving: the gate is `ab_check`, whose `dsv41` and `dsv41_l2_up_scaled` configs take the new plan on the native build, and whose `dsv41_nolimit` config, the portable build, and every small config keep the generic plan.

**Files:**
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/shapes.hpp`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/forward_plan.hpp`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp` (`run_plan`)

**Interfaces:**
- Consumes: `ForwardPlan`, `PlanTraits`, `StridedExperts::as<Other>()`, `kBuildIsa`.
- Produces: `struct Dsv41Shape { static constexpr int kHidden = 5120, kIntermediate = 2304; static constexpr float kActLimit = 10.0f; hidden/intermediate/act_limit; static bool accepts(const LayerInfo&); }`; `template <> struct PlanTraits<Dsv41Shape, Isa::Avx2>`.

- [ ] **Step 1: Add `Dsv41Shape` to `shapes.hpp`**

Change the file's header comment's last sentence to: `GenericShape takes each from the layer's LayerInfo; Dsv41Shape fixes DeepSeek V4.1's routed expert at compile time: hidden 5120, intermediate 2304, gated SiLU clamped at its swiglu_limit of 10. The w13 row order and the alphas' input scales stay runtime facts of the descriptor under every shape.` Then append:

```cpp
struct Dsv41Shape
{
    static constexpr int kHidden = 5120, kIntermediate = 2304;
    static constexpr float kActLimit = 10.0f;
    static constexpr int hidden(const LayerInfo&) { return kHidden; }
    static constexpr int intermediate(const LayerInfo&) { return kIntermediate; }
    static constexpr float act_limit(const LayerInfo&) { return kActLimit; }

    // Whether this layer may take the DSV4.1 plan: it has every fact above.
    static bool accepts(const LayerInfo& info)
    {
        return info.hidden == kHidden && info.intermediate == kIntermediate && info.act_limit == kActLimit;
    }
};
// No 64-column tail, so GpuRow never copies one, and whole Q8_0 blocks: what the plan's constants rely on.
static_assert(Dsv41Shape::kHidden % 64 == 0 && Dsv41Shape::kIntermediate % 64 == 0);
static_assert(SlabRowBytes::of(Dsv41Shape::kHidden, Dsv41Shape::kIntermediate).bytes[kSf13] == 4608ull * 320);
```

- [ ] **Step 2: Specialize `PlanTraits` for DSV4.1 on AVX2**

In `forward_plan.hpp`, change the header comment's last sentence to `The primary PlanTraits is the generic plan's; PlanTraits<Dsv41Shape, Isa::Avx2> is DeepSeek V4.1's on an AVX2 build.` and, after the primary `PlanTraits`, add:

```cpp
// DeepSeek V4.1 on AVX2: the shape's constants make every loop bound, sf_index group count and row stride a
// compile-time value. kRowUnit 16 splits 2304 gate/up rows into 144 units and 5120 down rows into 320, both even over
// 16 workers (Task 6 measures 32 and 48).
template <>
struct PlanTraits<Dsv41Shape, Isa::Avx2>
{
    static constexpr int kRowUnit = 16;
};
```

- [ ] **Step 3: Dispatch in `run_plan`**

Replace `run_plan` in `moe_mul1.cpp` with:

```cpp
// Runs the call's plan: the DSV4.1 plan when the layer is DeepSeek V4.1's routed expert on an AVX2 build, else the
// generic plan for this build's tier. Both read the same slabs; the DSV4.1 plan through the view checked for it.
int run_plan(ForwardCtx& ctx, const RegisteredLayer& layer, int threads)
{
    if constexpr (kBuildIsa == Isa::Avx2) {
        if (Dsv41Shape::accepts(ctx.info))
            return ForwardPlan<Dsv41Shape, Isa::Avx2>::run(ctx, layer.strided.as<Dsv41Shape>(), ForwardArena::get(),
                                                           threads);
    }
    return ForwardPlan<GenericShape, kBuildIsa>::run(ctx, layer.strided, ForwardArena::get(), threads);
}
```

- [ ] **Step 4: Commit, push, run the gates**

```bash
git add python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/shapes.hpp \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/forward_plan.hpp \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp
git commit -m "$(cat <<'EOF'
feat(nvfp4-cpu): DeepSeek V4.1 plan, ForwardPlan<Dsv41Shape, Isa::Avx2>, with the shape fixed at compile time

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
ssh divix01 'git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan fetch -q origin && git -C /data/models/slang/nvfp4-work/wt-nvfp4-plan checkout -q --detach origin/nvfp4-cpu-forward-plan'
```

Run: `ab_check` with `<N>`=`5`. Expected: three `PASS ... bit-exact`, `EXIT=0`.

- [ ] **Step 5: Prove the native build instantiates the DSV4.1 plan and the portable build does not**

```bash
ssh divix01 'nm -C /mnt/nvme1/nvfp4-plan/task5/ab-native | grep -c "Dsv41Shape"; nm -C /mnt/nvme1/nvfp4-plan/task5/ab-portable | grep -c "Dsv41Shape"'
```

Expected: the first count is ≥ 1 and the second is `0`. If the first is also 0 because GCC inlined every instantiation, look instead for `$0x900` (2304) or `$0x1400` (5120) immediates in `objdump -d --no-show-raw-insn /mnt/nvme1/nvfp4-plan/task5/ab-native`, compared with `ab-portable`, and ledger which check you used.
Run: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_nvfp4_cpu_build.py`. Expected: `9 passed`, `EXIT=0`.

---

### Task 6: Bench OpenMP policy, READMEs, latency A/B against master

**Files:**
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/bench/run.sh`
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/README.md` (Attach, Threading and lifetime, new Code layout section)
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/bench/README.md` (intro sentence about OpenMP; Run section)
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu/upstream/README.md` (last paragraph)
- Possibly modify: `.../optimized/forward_plan.hpp` (`PlanTraits<Dsv41Shape, Isa::Avx2>::kRowUnit`, only per Step 4's rule)

- [ ] **Step 1: `run.sh` sets and records the OpenMP wait policy**

After `shift 2` in `run.sh`, add:

```bash
# The kernel's workers are one OpenMP team per forward: spin between forwards rather than sleep, never let the
# runtime shrink the team, and never let OpenMP bind threads (the kernel pins each worker to its configured core).
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-ACTIVE} GOMP_SPINCOUNT=${GOMP_SPINCOUNT:-INFINITE} OMP_DYNAMIC=FALSE
unset OMP_PROC_BIND OMP_PLACES
```

and in the `environment.txt` block, after the `printf 'Rounds: ...` line, add:

```bash
  printf 'OMP_WAIT_POLICY=%s GOMP_SPINCOUNT=%s OMP_DYNAMIC=%s OMP_THREAD_LIMIT=%s\n' \
    "$OMP_WAIT_POLICY" "$GOMP_SPINCOUNT" "$OMP_DYNAMIC" "${OMP_THREAD_LIMIT:-unset}"
```

- [ ] **Step 2: Update the READMEs**

In `optimized/README.md`:

- Attach section: change `Include \`cpu_experts_cabi.h\` and link the shared library or compile \`moe_mul1.cpp\` into the same native module, as with EXL3.` to `Include \`cpu_experts_cabi.h\` and load the library \`build.py\` or \`nvfp4_cpu_ext.nvfp4_cpu_library()\` builds.`, and remove the sentence `Compile/link \`../upstream/nvfp4.c\` alongside \`moe_mul1.cpp\`.`
- Replace the whole `## Threading and lifetime` section's first paragraph with:

```markdown
Each forward runs one OpenMP team of `threads` workers, the calling engine thread as worker 0, in four phases
separated by barriers: input to Q8_0; every routed expert's gate/up rows and SiLU; every intermediate to Q8_0; every
expert's down rows summed in routing order into `out`. Configure distinct allowed Linux cores before the first
forward; worker i is pinned to core i (once, then re-checked cheaply). A forward may use any team size up to the
configured cores. If OpenMP forms a smaller team (`OMP_THREAD_LIMIT`, `OMP_DYNAMIC`), the forward returns 1 and
leaves `out` untouched. For latency set `OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE OMP_DYNAMIC=FALSE` and leave
`OMP_PROC_BIND` unset. Concurrent forward/free/configuration is rejected (3). Stop/join the engine before freeing
handles or slab storage; do not unload the library while callbacks are in use. The kernel requires Linux and OpenMP.
```

- Add a section before `## Validation boundaries`:

```markdown
## Code layout

`moe_mul1.cpp` holds the registry, worker cores, the arithmetic and the C ABI. A forward is
`ForwardPlan<Shape, Isa>::run` (`forward_plan.hpp`), picked once per call in `run_plan`:
`ForwardPlan<Dsv41Shape, Isa::Avx2>` when `Dsv41Shape::accepts` the layer on an AVX2 build, else
`ForwardPlan<GenericShape, kBuildIsa>`. The tier is the build's (`-march=native` AVX2, or the portable scalar loop).
`PlanTraits<Shape, Isa>` holds the plan's knobs (`kRowUnit`, the split unit). A plan reads every layer fact it may fix
through its Shape (`shapes.hpp`): `GenericShape` from the layer's `LayerInfo`, `Dsv41Shape` as compile-time constants
(5120/2304, SiLU limit 10). Plans read slots through `StridedExperts<Shape>` (`experts.hpp`) over the descriptor's
slab bases and strides; `SlabRowBytes` is each slab's minimum stride.

Bit-exact checks for any change here: `test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh` (native, upstream-baseline
and portable builds against a baseline worktree's dumps), and `test/registered/unit/kernels/test_nvfp4_cpu_{build,experts}.py`.
```

In `bench/README.md`: change `records. No Python, PyTorch, CUDA or OpenMP is required.` to `records. No Python, PyTorch or CUDA is required; the kernel needs OpenMP, and \`run.sh\` sets its wait policy (\`OMP_WAIT_POLICY=ACTIVE\`, \`GOMP_SPINCOUNT=INFINITE\`, \`OMP_DYNAMIC=FALSE\`) unless the caller already did.` and change `use identical Q8_0 activation quantization, FP32 projections, SiLU, routing, thread pool and` to `use identical Q8_0 activation quantization, FP32 projections, SiLU, routing, OpenMP team and`.

In `upstream/README.md`, replace the last paragraph (`` `../optimized/moe_mul1.cpp` is SGLang glue: ... ``) with:

```markdown
`../optimized/moe_mul1.cpp` and its headers (`experts.hpp`, `shapes.hpp`, `forward_plan.hpp`) are SGLang glue:
registration, the OpenMP forward plan, GPU scale addressing, global alphas, FP16 input, SiLU, routing and callback ABI.
```

- [ ] **Step 3: Latency A/B, master's bench against the branch's (16 workers, NUMA node 1)**

Check that nothing else is running, then build both bench binaries:

```bash
ssh divix01 'pgrep -fa "[s]glang.launch_server" && echo "SERVER RUNNING: stop" ; systemctl is-active exl3bench.service'
ssh divix01 'set -e; gcc15=/opt/rh/gcc-toolset-15/root/usr/bin; o=/mnt/nvme1/nvfp4-plan
  taskset -c 0-63 cmake -S /data/models/slang/nvfp4-work/wt-nvfp4-plan-base/python/sglang/srt/layers/quantization/nvfp4_cpu/bench \
    -B $o/old-build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$gcc15/g++ -DCMAKE_C_COMPILER=$gcc15/gcc > $o/old.configure.log 2>&1
  taskset -c 0-63 cmake --build $o/old-build -j16 --target nvfp4_cpu_optimized > $o/old.build.log 2>&1
  cd /data/models/slang/nvfp4-work/wt-nvfp4-plan && git fetch -q origin && git checkout -q --detach origin/nvfp4-cpu-forward-plan
  taskset -c 0-63 cmake -S python/sglang/srt/layers/quantization/nvfp4_cpu/bench -B $o/new-build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$gcc15/g++ -DCMAKE_C_COMPILER=$gcc15/gcc > $o/new.configure.log 2>&1
  taskset -c 0-63 cmake --build $o/new-build -j16 --target nvfp4_cpu_optimized > $o/new.build.log 2>&1
  ls -l $o/old-build/nvfp4_cpu_optimized $o/new-build/nvfp4_cpu_optimized'
```

The old build comes from the base worktree's `bench/` directory. The top-level `CMakeLists.txt` is already gone there, because Task 1 deleted it. That base bench CMake is still the pre-OpenMP one, and it compiles the pre-OpenMP kernel with `-march=native`, so it is the right thing to compare against.

Run the A/B through `exl3bench.service` (isolated CPUs 16–33 and siblings 52–69). Save the current command file, write the A/B's, start the service the way the 2026-10-03 keep-warm A/B did, wait until it is inactive, then restore:

```bash
ssh divix01 'cp /data/models/exl3_exp/google_benchmark/service-command.txt /mnt/nvme1/nvfp4-plan/service-command.txt.before
  o=/mnt/nvme1/nvfp4-plan; flags=@--cpus=18-33@--numa-node=1@--workers=16@--experts=1,3,5
  printf "%s\n" /bin/bash /data/models/slang/nvfp4-work/exl3-gemv/ab.sh 8 \
    "old=$o/old-build/nvfp4_cpu_optimized$flags" "new=$o/new-build/nvfp4_cpu_optimized$flags" \
    > /data/models/exl3_exp/google_benchmark/service-command.txt'
```

After the run: restore with `cp /data/models/slang/nvfp4-work/exl3-gemv/service-command.txt.saved /data/models/exl3_exp/google_benchmark/service-command.txt` and confirm with `cat`. Report with `python3 /data/models/slang/nvfp4-work/exl3-gemv/ab-report.py <results dir> old`.

Expected: a table of median p50 (µs) for experts 1, 3 and 5. **Gate:** `new` is no slower than `old` by more than 3% at any expert count. If it is slower, stop: report the table and do not tune further in this plan.

- [ ] **Step 4: Measure the DSV4.1 plan's `kRowUnit` (32 and 48 against 16)**

Make two mutant builds in a private worktree, built from the branch with `PlanTraits<Dsv41Shape, Isa::Avx2>::kRowUnit` changed to 32 and 48, and never commit them. Run them with `ab.sh 8` against `new` exactly as in Step 3. Then revert the mutant (`git checkout -- forward_plan.hpp`) and confirm `git status` is clean.

Rule: change the committed `kRowUnit` only if one value beats 16 by more than 2% at every expert count in the median p50 and does not lose at any. Otherwise keep 16, and record the table in the ledger. If it changes, edit the specialization's value and its comment (cite the medians). Then re-run `ab_check` with `<N>`=`6`, expecting three `PASS ... bit-exact` (row assignment does not change arithmetic), and commit.

- [ ] **Step 5: Commit and push**

```bash
git add python/sglang/srt/layers/quantization/nvfp4_cpu/bench/run.sh \
        python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/README.md \
        python/sglang/srt/layers/quantization/nvfp4_cpu/bench/README.md \
        python/sglang/srt/layers/quantization/nvfp4_cpu/upstream/README.md
git commit -m "$(cat <<'EOF'
docs(nvfp4-cpu): OpenMP threading, code layout and Python build; the bench sets the OpenMP wait policy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft
EOF
)"
git push
```

Final gates on the pushed head: `divix_pytest test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_nvfp4_cpu_build.py` gives `9 passed`, `EXIT=0`. `ab_check` with `<N>`=`final` gives three `PASS`, `EXIT=0`. Record both commands and their output in the ledger next to the A/B table.
