# EXL3 CPU kernel: ForwardPlan<Shape, Isa> and strided expert views — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the optimized EXL3 CPU expert kernel so the ISA tier and the DeepSeek V4.1 shape are compile-time parameters of one forward plan, and let sglang register a layer as six slab base pointers instead of `capacity × 9` tensor views — with every output bit-identical to today's.

**Architecture:** The forward becomes `ForwardPlan<Shape, Isa>`, dispatched once per call. `GenericShape` (the primary template) reads dimensions from the layer; `Dsv41Shape` fixes H=5120, I=2304, 3-bit, and its `PlanTraits<Dsv41Shape, Isa::Bw>` specialization turns on today's DSV4.1 fast path (compact scratch, grouped traversal, 512-wide single-expert quantization). The plan reads experts through an accessor: `TableExperts` over `make_layer`'s per-expert tables (unchanged, still used by upstream's bindings, the bench and the tests) or `StridedExperts<Shape>` over the pinned tier's slabs (new C ABI `sglang_exl3_cpu_experts_register_slabs`, used by `Exl3CpuQuantTrait`).

**Tech Stack:** C++20 (GCC 15, `-Ofast -march=native`, OpenMP), ATen/c10, Python 3.13 + ctypes, pytest, Google Benchmark bench under `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/`.

**Spec:** this conversation's agreed design (no separate spec file): ForwardPlan<Shape, Isa>; two expert accessors (TableExperts / StridedExperts<Shape>); the DSV4.1 plan stays AVX-512BW-only; behavior-preserving and bit-exact; keep compile-time-dead branches (other `EXL3_MOE_CPU_ACT_*` builds, other ISA tiers) because the kernel may run on another CPU later.

## Global Constraints

- Kernel file: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`. The older kernel `exl3_cpu/moe_mul1.cpp` is NOT touched.
- **Bit-exact or it does not ship.** Every task's gate is `run_exl3_cpu_forward_checks.sh check` (Task 1) green: all A/B dumps bitwise equal to the merge-base's, `exl3_cpu_optimized --validate-only` ("Verified 24 bit-exact layer outputs"), `exl3_full_stack_prod --validate-only` ("Verified 48 bit-exact layer outputs"), and both CPU-expert pool test files passing.
- The DSV4.1 plan exists only as `ForwardPlan<Dsv41Shape, Isa::Bw>` (today's `g_isa == Isa::Bw` gate, `==` not `>=`). VNNI/VBMI keep the generic plan.
- Keep every compile-time-dead branch: `ACT_ROWS == 1`, `EXL3_MOE_CPU_ACT_BLOCK != 128`, scalar/AVX2/VNNI/VBMI tiers. Only the runtime-dead phase 4 (whole-row down transform, skipped by `if(phase==4)continue;` today) is removed.
- Public symbols upstream links against keep their exact signatures: `exl3_moe_cpu_make_layer`, `_free_layer`, `_forward`, `_forward_raw`, `_stage_experts`, `_set_prof`, `_pool_stress`, `_has_*` (`exllamav3/exllamav3_ext/bindings.cpp` and `cpu/moe_handoff.cu` at the pinned commit `02aef45` compile against upstream's header).
- Do not change arithmetic, operation order, `noipa`/`optimize("no-associative-math")` attributes, or target attributes of any function body. Changes are to signatures, dispatch and data access only.
- Code is edited on the laptop, committed, pushed to `origin`, and run on divix01 in a pulled worktree (`.claude/rules/divix01-run-protocol.md`). Never rsync/scp a tree. CPU jobs under `taskset -c 0-63`. Read pytest's own exit status, never a pipe's.
- No server may be running when the benches run: they pin CPUs 16–33 and 52. The driver refuses if `pgrep -f sglang.launch_server` matches.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage files by name.

## Review Focus

1. **A non-BW tier silently changing.** divix01 is AVX-512BW-only, so the frozen references cover only BW. Scalar and AVX2 are covered by the A/B harness through `EXL3_MOE_CPU_MAX_ISA`; VNNI/VBMI cannot run here at all. Task 1 dumps scalar/avx2/bw; reviewers must check by reading that the VNNI/VBMI arms of each `if constexpr` chain are textually the old `case` bodies.
2. **A multi-token (prefill) call taking the DSV4.1 plan.** A chunk with `m == 2` must fall back to the generic plan exactly as `compact_forward` did. The harness's `t2k3` route sends two tokens to a shared expert at DSV4.1 dims; Task 4's tests depend on it.
3. **A strided handle addressing the wrong bytes.** An off-by-a-part offset (gate vs up within a w13 row) still produces finite numbers. Task 5 compares strided registration against `make_layer` bitwise at both shapes and every tier, and against the frozen references through the full-stack bench.
4. **Slabs freed while registered.** The kernel no longer holds tensor references for strided layers. Task 6's trait keeps the slab tensors alive until `free_layer`; its test checks that the references are dropped only then.
5. **A slab with the wrong row size or layout reaching the kernel.** The C ABI cannot see tensor shapes. Task 6's trait validates dtype, contiguity, row count and per-row element count before calling; its test pins each refusal.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `test/manual/dsv41/exl3_cpu_forward_ab.py` (new) | A/B harness: dump fixed forwards per ISA tier and registration, compare dumps bitwise | 1, 5 |
| `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` (new) | divix01 driver: baseline dumps, or every bit-exact gate for a change | 1 |
| `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp` | kernel: ISA templates, registry, forward dispatch, new C ABI | 2–5 |
| `.../optimized/experts.hpp` (new) | `LayerInfo`, `TableExperts`, `SlabRowBytes`, `StridedExperts<Shape>` | 3, 5 |
| `.../optimized/shapes.hpp` (new) | `GenericShape`, `Dsv41Shape` (the static DSV4.1 config) | 4 |
| `.../optimized/forward_plan.hpp` (new) | `PlanTraits`, `ForwardPlan` | 4 |
| `.../optimized/register.hpp`, `traversal.hpp` | `grouped` becomes a parameter instead of a `thread_local` | 4 |
| `.../optimized/cpu_experts_cabi.h` | declares `sglang_exl3_cpu_experts_register_slabs` | 5 |
| `.../optimized/README.txt` | code layout and the slab registration API | 5, 7 |
| `python/.../expert_stream/bench/src/stack_fixture.cpp` | registers through the slab ABI | 5 |
| `python/sglang/srt/layers/moe/cpu_experts/exl3.py` | trait registers by base pointer, keeps slabs alive | 6 |
| `test/registered/unit/kernels/test_cpu_expert_pool.py` | trait registration test rewritten for the slab ABI | 6 |
| `test/manual/dsv41/test_cpu_expert_pool_exl3.py` | pool vs `make_layer` at the DSV4.1 shape too | 6 |

The header split follows the file's existing pattern (`register.hpp`, `traversal.hpp` are included mid-file inside the anonymous namespace). The `.hpp` files are not standalone: each relies on what the `.cpp` has declared at its include point.

## Working environment

- Laptop worktree: `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-forward-plan`, branch `exl3-cpu-forward-plan`, based on `origin/master` = `44401799ac`.
- divix01 worktrees (create once, Task 1):
  - base: `/data/models/slang/nvfp4-work/wt-forward-plan-base` at Task 1's commit (kernel identical to `44401799ac`)
  - branch: `/data/models/slang/nvfp4-work/wt-forward-plan` at `origin/exl3-cpu-forward-plan`
- Output root on divix01: `/data/models/slang/nvfp4-work/exl3-forward-plan/` (`base/`, then `task2/`, `task3/`, …).
- To run a task's gate after pushing:

```bash
ssh divix01 'git -C /data/models/slang/sglang fetch -q origin \
  && git -C /data/models/slang/nvfp4-work/wt-forward-plan checkout -q --detach origin/exl3-cpu-forward-plan \
  && git -C /data/models/slang/nvfp4-work/wt-forward-plan log -1 --oneline \
  && /data/models/slang/nvfp4-work/wt-forward-plan/test/manual/dsv41/run_exl3_cpu_forward_checks.sh check \
       /data/models/slang/nvfp4-work/wt-forward-plan /data/models/slang/nvfp4-work/exl3-forward-plan/taskN \
       /data/models/slang/nvfp4-work/exl3-forward-plan/base'
```

  Run it with `run_in_background`; a cold run builds the extension's `moe_mul1.o` and both bench targets (several minutes). Add `slabs` as a fifth argument from Task 5 on.

---

### Task 1: Bit-exact A/B harness, driver, and the merge-base baseline

**Files:**
- Create: `test/manual/dsv41/exl3_cpu_forward_ab.py`
- Create: `test/manual/dsv41/run_exl3_cpu_forward_checks.sh`

**Interfaces:**
- Produces: `exl3_cpu_forward_ab.py dump --isa {scalar,avx2,bw} --registration {table,slabs} --out FILE` and `exl3_cpu_forward_ab.py compare WANT GOT` (exit 0 iff bitwise equal). `--registration slabs` calls `sglang_exl3_cpu_experts_register_slabs` with the exact C signature Task 5 implements:
  `int sglang_exl3_cpu_experts_register_slabs(const void* const* slabs, int32_t capacity, int32_t hidden, int32_t intermediate, int32_t bits, int32_t swizzled, float act_limit, int64_t* handle)`.
- Produces: `run_exl3_cpu_forward_checks.sh baseline WT OUT` and `run_exl3_cpu_forward_checks.sh check WT OUT BASE [slabs]`.

- [ ] **Step 1: Write the harness**

`test/manual/dsv41/exl3_cpu_forward_ab.py`:

```python
"""Bit-exact A/B harness for the optimized EXL3 CPU expert kernel (exl3_cpu/optimized/moe_mul1.cpp).

``dump`` runs a fixed set of forwards through the extension ``exl3_ext()`` builds and saves every output; ``compare``
checks two dumps for bitwise equality. A dump made at the merge-base is the reference a kernel refactor must reproduce
exactly, on every ISA tier the host can run. The kernel reads EXL3_MOE_CPU_MAX_ISA once, at load, so each tier is its
own process.

Run on divix01 through run_exl3_cpu_forward_checks.sh, which sets the build environment.
"""

import argparse
import ctypes
import os
import sys

# (name, hidden, intermediate): a generic shape, and DeepSeek V4.1's, which takes the DSV4.1 plan on AVX-512BW.
SHAPES = (("generic", 512, 256), ("dsv41", 5120, 2304))
CAP = 6
LIMIT = 10.0
THREADS = 4
# (tokens, experts per token) -> routes. Two tokens sharing expert 0 and 4 make m=2 chunks, which the DSV4.1 plan
# refuses: at the DSV4.1 shape that case runs the generic plan.
ROUTES = {
    (1, 1): [[2]],
    (1, 3): [[4, 0, 5]],
    (1, 5): [[1, 3, 0, 5, 2]],
    (2, 3): [[4, 0, 5], [0, 2, 4]],
}
WEIGHTS = (0.5, 0.3, 0.2, 0.15, 0.1)
SCALES = (1.0, 8.0)
# exl3_expert_format.EXL3_STREAMED_NAMES, the slab order of sglang_exl3_cpu_experts_register_slabs.
NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
TIERS = {"scalar": (False, False), "avx2": (True, False), "bw": (True, True)}


def random_slabs(torch, hidden, inter, seed):
    """The pinned tier's slab rows ([CAP, parts, ...], 3-bit), filled with random codes and signs."""
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    def codes(*shape):
        return torch.randint(-32768, 32767, shape, generator=g, dtype=torch.int16)

    return {
        "w13_trellis": codes(CAP, 2, hidden // 16, inter // 16, 48),
        "w13_suh": signs(CAP, 2, hidden),
        "w13_svh": signs(CAP, 2, inter),
        "w2_trellis": codes(CAP, 1, inter // 16, hidden // 16, 48),
        "w2_suh": signs(CAP, 1, inter),
        "w2_svh": signs(CAP, 1, hidden),
    }


def register_table(ext, s):
    """make_layer over one view per slot: gate is w13 part 0, up part 1, down w2 part 0."""
    rows = range(CAP)
    return ext.exl3_moe_cpu_make_layer(
        [s["w13_trellis"][i, 0] for i in rows],
        [s["w13_suh"][i, 0] for i in rows],
        [s["w13_svh"][i, 0] for i in rows],
        [s["w13_trellis"][i, 1] for i in rows],
        [s["w13_suh"][i, 1] for i in rows],
        [s["w13_svh"][i, 1] for i in rows],
        [s["w2_trellis"][i, 0] for i in rows],
        [s["w2_suh"][i, 0] for i in rows],
        [s["w2_svh"][i, 0] for i in rows],
        [],
        [],
        [],
        0,
        LIMIT,
        0,
    )


def register_slabs(ext, s, hidden, inter):
    """The slab ABI over the same tensors: six base pointers, slot s at base + s rows."""
    fn = ctypes.CDLL(ext.__file__).sglang_exl3_cpu_experts_register_slabs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int64),
    ]
    bases = (ctypes.c_void_p * len(NAMES))(*(s[n].data_ptr() for n in NAMES))
    handle = ctypes.c_int64(-1)
    status = fn(bases, CAP, hidden, inter, 3, 0, LIMIT, ctypes.byref(handle))
    if status != 0:
        sys.exit(f"register_slabs refused the {hidden}x{inter} slabs: status {status}")
    return handle.value


def dump(args):
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = args.isa  # read at the kernel's load: before the extension imports
    os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")
    import torch

    import sglang
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    print(f"sglang {sglang.__file__}")
    ext = exl3_ext()
    tier = (bool(ext.exl3_moe_cpu_has_avx2()), bool(ext.exl3_moe_cpu_has_avx512_bw()))
    if tier != TIERS[args.isa]:
        sys.exit(f"asked for {args.isa}, the kernel reports avx2={tier[0]} bw={tier[1]}")
    torch.set_num_threads(1)
    outputs = {}
    for name, hidden, inter in SHAPES:
        slabs = random_slabs(torch, hidden, inter, seed=hidden)
        if args.registration == "table":
            handle = register_table(ext, slabs)
        else:
            handle = register_slabs(ext, slabs, hidden, inter)
        try:
            g = torch.Generator().manual_seed(7)
            for (tokens, k), route in ROUTES.items():
                sel = torch.tensor(route, dtype=torch.int64)
                w = torch.tensor([list(WEIGHTS[:k])] * tokens).half()
                for scale in SCALES:
                    x = (torch.randn(tokens, hidden, generator=g) * scale).half()
                    out = torch.full((tokens, hidden), float("nan"))
                    ext.exl3_moe_cpu_forward(handle, x, sel, w, out, THREADS)
                    case = f"{name}/t{tokens}k{k}/s{scale}"
                    if not torch.isfinite(out).all():
                        sys.exit(f"{case}: non-finite output")
                    outputs[case] = out
        finally:
            ext.exl3_moe_cpu_free_layer(handle)
    torch.save({"isa": args.isa, "registration": args.registration, "outputs": outputs}, args.out)
    print(f"{len(outputs)} outputs ({args.isa}, {args.registration}) -> {args.out}")


def compare(args):
    import torch

    want, got = torch.load(args.want), torch.load(args.got)
    if want["outputs"].keys() != got["outputs"].keys():
        sys.exit(f"the dumps hold different cases: {sorted(want['outputs'])} vs {sorted(got['outputs'])}")
    bad = [c for c in want["outputs"] if not torch.equal(want["outputs"][c], got["outputs"][c])]
    for c in bad:
        diff = (want["outputs"][c] - got["outputs"][c]).abs().max().item()
        print(f"MISMATCH {c}: max |diff| {diff:.3e}")
    total = len(want["outputs"])
    print(
        f"{total - len(bad)}/{total} bit-exact: {want['isa']}/{want['registration']} vs "
        f"{got['isa']}/{got['registration']}"
    )
    sys.exit(1 if bad else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--isa", choices=sorted(TIERS), required=True)
    d.add_argument("--registration", choices=("table", "slabs"), default="table")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("want")
    c.add_argument("got")
    args = parser.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the divix01 driver**

`test/manual/dsv41/run_exl3_cpu_forward_checks.sh` (mode `0755`):

```bash
#!/usr/bin/env bash
# Bit-exact gates for a change to the optimized EXL3 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#
#   run_exl3_cpu_forward_checks.sh baseline WORKTREE OUT
#       At the merge-base: exl3_cpu_forward_ab.py dumps per ISA tier (scalar, avx2, bw), through make_layer.
#   run_exl3_cpu_forward_checks.sh check WORKTREE OUT BASELINE_OUT [slabs]
#       (1) the same dumps, compared bitwise with BASELINE_OUT's; with `slabs`, also through the slab registration,
#       compared with the same make_layer baseline; (2) the bare-forward bench's 24 frozen DSV4.1 outputs;
#       (3) the full-stack bench's 48; (4) the CPU expert pool tests.
#
# Builds into OUT: a private copy of the extension's build directory (~670 MB) and the bench. Exits nonzero when any
# step fails; each step's log is OUT/<step>.log.
set -uo pipefail
mode=${1:?mode}
wt=$(realpath "${2:?worktree}")
out=${3:?output dir}
base=${4:-}
slabs=${5:-}
mkdir -p "$out/tmp"
out=$(realpath "$out")

if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the benches pin CPUs 16-33 and 52" >&2
  exit 2
fi

PY=/data/models/slang/.venv/bin/python
GXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
export TMPDIR=$out/tmp
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=$out/exl3-build
export SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_EXL3_CPU_CXX=$GXX CUDA_HOME=/usr/local/cuda-13.4
export PYTHONPATH=$wt/python OMP_NUM_THREADS=8 EXL3_MOE_CPU_PIN=0
if [[ ! -d $SGLANG_EXL3_BUILD_DIR/resid_b128_cpu_v1 ]]; then
  mkdir -p "$SGLANG_EXL3_BUILD_DIR"
  cp -a ~/.cache/sglang/exl3_ext/resid_b128_cpu_v1 "$SGLANG_EXL3_BUILD_DIR/"
fi
cd "$wt"
echo "worktree $wt at $(git rev-parse --short HEAD)"

failed=()
step() {
  local name=$1
  shift
  "$@" > "$out/$name.log" 2>&1
  local rc=$?
  echo "$name EXIT=$rc  ($(tail -1 "$out/$name.log"))"
  [[ $rc -eq 0 ]] || failed+=("$name")
}

ab=$wt/test/manual/dsv41/exl3_cpu_forward_ab.py
registrations=(table)
[[ $slabs == slabs ]] && registrations+=(slabs)
for isa in bw avx2 scalar; do
  for reg in "${registrations[@]}"; do
    [[ $mode == baseline && $reg != table ]] && continue
    step "dump-$isa-$reg" taskset -c 0-63 "$PY" "$ab" dump --isa "$isa" --registration "$reg" --out "$out/ab-$isa-$reg.pt"
    if [[ $mode == check ]]; then
      step "compare-$isa-$reg" "$PY" "$ab" compare "$base/ab-$isa-table.pt" "$out/ab-$isa-$reg.pt"
    fi
  done
done

if [[ $mode == check ]]; then
  bench=$wt/python/sglang/kernels/jit/csrc/moe/expert_stream/bench
  build=$out/bench-build
  configure=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=$GXX" "-DEXL3_TORCH_ROOT=$SITE/torch"
    -DEXL3_CXX11_ABI=1 "-DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi")
  gbench=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src
  [[ -d $gbench ]] && configure+=("-DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=$gbench")
  step bench-configure taskset -c 0-63 cmake -S "$bench" -B "$build" "${configure[@]}"
  step bench-build taskset -c 0-63 cmake --build "$build" -j16 --target exl3_cpu_optimized exl3_full_stack_prod
  export EXL3_MOE_CPU_MAX_ISA=bw OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  step bare-validate "$build/exl3_cpu_optimized" --validate-only
  mkdir -p "$out/images"
  step full-stack-validate "$build/exl3_full_stack_prod" --validate-only "--image-dir=$out/images"
  unset EXL3_MOE_CPU_MAX_ISA
  step pytest taskset -c 0-63 "$PY" -m pytest -q -p no:randomly \
    test/manual/dsv41/test_cpu_expert_pool_exl3.py test/registered/unit/kernels/test_cpu_expert_pool.py
fi

if ((${#failed[@]})); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "ALL GREEN ($mode)"
```

- [ ] **Step 3: Commit and push**

```bash
cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-forward-plan
chmod +x test/manual/dsv41/run_exl3_cpu_forward_checks.sh
git add test/manual/dsv41/exl3_cpu_forward_ab.py test/manual/dsv41/run_exl3_cpu_forward_checks.sh docs/superpowers/plans/2026-10-02-exl3-cpu-forward-plan.md
git commit -m "$(cat <<'EOF'
test(exl3-cpu): bit-exact A/B harness and divix01 checks for the optimized CPU kernel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push -u origin exl3-cpu-forward-plan
```

- [ ] **Step 4: Create the divix01 worktrees and dump the baseline**

The baseline worktree is the merge-base plus only this task's two new files, so check it out at the pushed branch head (Task 1 changes no kernel code):

```bash
ssh divix01 'git -C /data/models/slang/sglang fetch -q origin \
  && git -C /data/models/slang/sglang worktree add -q --detach /data/models/slang/nvfp4-work/wt-forward-plan-base origin/exl3-cpu-forward-plan \
  && git -C /data/models/slang/sglang worktree add -q --detach /data/models/slang/nvfp4-work/wt-forward-plan origin/exl3-cpu-forward-plan \
  && /data/models/slang/nvfp4-work/wt-forward-plan-base/test/manual/dsv41/run_exl3_cpu_forward_checks.sh baseline \
       /data/models/slang/nvfp4-work/wt-forward-plan-base /data/models/slang/nvfp4-work/exl3-forward-plan/base'
```

Expected: `dump-bw-table EXIT=0 (16 outputs (bw, table) -> …)` (2 shapes × 4 routes × 2 scales), the same for `avx2` and `scalar`, then `ALL GREEN (baseline)`. Any other count: stop and read the log.

- [ ] **Step 5: Prove the harness is deterministic (the A/B is meaningless otherwise)**

Run `check` on the unchanged base worktree against its own baseline:

```bash
ssh divix01 '/data/models/slang/nvfp4-work/wt-forward-plan-base/test/manual/dsv41/run_exl3_cpu_forward_checks.sh check \
  /data/models/slang/nvfp4-work/wt-forward-plan-base /data/models/slang/nvfp4-work/exl3-forward-plan/base-recheck \
  /data/models/slang/nvfp4-work/exl3-forward-plan/base'
```

Expected: every `compare-*` prints `16/16 bit-exact`, `bare-validate` ends `Verified 24 bit-exact layer outputs…`, `full-stack-validate` ends `Verified 48 bit-exact layer outputs…`, `pytest` passes, then `ALL GREEN (check)`. If a compare fails here, the harness itself is nondeterministic: stop and report; do not proceed to Task 2.

---

### Task 2: ISA as a template parameter, dispatched once per forward

Mechanical: every runtime `g_isa` test inside the forward becomes a compile-time test on a template parameter `I`, and the forward picks the instantiation once. `g_isa` stays for `detect_isa`, the `exl3_moe_cpu_has_*` queries, and that one dispatch.

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`

**Interfaces:**
- Produces (used by Tasks 3–4): `template <Isa I> constexpr bool kAvx512`; templates `hadamard_128_avx2<I>`, `hadamard_128<I>`, `prepare_block_avx2<I>`, `prepare_rows<I>`, `run_tiles_raw<I>`, `run_tiles<I>`, `transform_out_avx2<I>`, `transform_out<I>`, `middle_blocks<I, Wide>`, `prepare_gu_blocks<I, Wide>`, `transform_owned_blocks<I>`, `forward_phase_index<I>`, `run_team<I>(ForwardCtx&, int count, bool grouped)`.

- [ ] **Step 1: Add the tier predicate after `extern const Isa g_isa;`**

```cpp
enum class Isa { Scalar, Avx2, Bw, Vnni, Vbmi };
extern const Isa g_isa;
// The forward is instantiated per tier; g_isa picks the instantiation once per call (forward_raw).
template <Isa I> constexpr bool kAvx512 = I == Isa::Bw || I == Isa::Vnni || I == Isa::Vbmi;
```

- [ ] **Step 2: Template the Hadamard selectors**

Replace `hadamard_128_avx2` and `hadamard_128`:

```cpp
template <Isa I>
M1_TARGET_AVX2 void hadamard_128_avx2(float* v) {
    if constexpr (kAvx512<I>) hadamard_512(v);
    else hadamard_128_current(v);
}


template <Isa I>
inline void hadamard_128(float* v)
{
    if constexpr (I != Isa::Scalar) hadamard_128_avx2<I>(v);
    else                            hadamard_128_scalar(v);
}
```

- [ ] **Step 3: Template `prepare_block_avx2` and `prepare_rows`**

`prepare_block_avx2` gains `template <Isa I>` before `M1_TARGET_AVX2` and calls `hadamard_128_avx2<I>(dst + block);`. Its body is otherwise unchanged.

`prepare_rows` gains `template <Isa I>`. The compact branch calls `prepare_block_avx2<I>(...)`. The per-row loop becomes (the scalar arithmetic is copied unchanged):

```cpp
    for (int r = 0; r < m; ++r)
    {
        float* dst = p.tin + static_cast<size_t>(r) * k;
        const size_t src_off = static_cast<size_t>(token_idx[r]) * src_stride;
        if constexpr (I != Isa::Scalar)
        {
            prepare_block_avx2<I>(src_f16 ? reinterpret_cast<const void*>(src_f16 + src_off)
                                          : reinterpret_cast<const void*>(src_f32 + src_off),
                                  src_f16 != nullptr, mat.suh, dst, k);
        }
        else
        {
            for (int block = 0; block < k; block += 128)
            {
                float vals[128];
                for (int i = 0; i < 128; ++i)
                {
                    const float xv = src_f16 ? half_to_float(src_f16[src_off + block + i])
                                             : src_f32[src_off + block + i];
                    vals[i] = xv * half_to_float(mat.suh[block + i]);
                }
                hadamard_128<I>(vals);
                for (int i = 0; i < 128; ++i)
                    dst[block + i] = vals[i] * HAD_SCALE;
            }
        }

        // int8 quantization, one scale per row
        int32_t* splat = p.splat32 + static_cast<size_t>(r) * k;
        // dup is only read by the AVX2/BW maddubs kernels; skip the stores on the VNNI/VBMI tiers
        int32_t* splat_dup = (p.splat_dup && (I == Isa::Avx2 || I == Isa::Bw))
            ? p.splat_dup + static_cast<size_t>(r) * k : nullptr;
        float q;
        int32_t s;
        if constexpr (I != Isa::Scalar)
        {
            if (ACT_ROWS > 1 || act_blocked(k))
            {
                quantize_act(p, r, m, k, dst, splat_dup != nullptr);
                continue;
            }
            quantize_row_avx2(dst, splat, splat_dup, k, q, s);
        }
        else
        {
            float amax = 0.0f;
            for (int i = 0; i < k; ++i) amax = std::max(amax, std::fabs(dst[i]));
            q = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float rq = 1.0f / q;
            s = 0;
            for (int i = 0; i < k; ++i)
            {
                int v = static_cast<int>(std::lround(dst[i] * rq));
                v = std::clamp(v, -127, 127);
                s += v;
                splat[i] = static_cast<int32_t>(static_cast<uint8_t>(static_cast<int8_t>(v))) * 0x01010101;
            }
        }
        p.q[r] = q;
        p.sum_x8[r] = s;
    }
```

Equivalence: the old code ran `quantize_act` iff `g_isa != Scalar && (ACT_ROWS > 1 || act_blocked(k))`, else `quantize_row_avx2` iff `g_isa != Scalar`, else the scalar loop. The new nesting is the same decision tree.

- [ ] **Step 4: Template `run_tiles_raw`**

Add `template <Isa I>`. Replace the outer `switch (g_isa) { case Isa::Vbmi: { … } case Isa::Vnni: { … } case Isa::Bw: { … } case Isa::Avx2: { … } case Isa::Scalar: { … } }` with an `if constexpr` chain whose arms are the **verbatim** case bodies (including each inner `switch` and the trailing `return;` where one exists):

```cpp
template <Isa I>
void run_tiles_raw(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Vbmi)
    {
        switch (mat.bits * 4 + m - 1)
        {
            // ... the old `case Isa::Vbmi:` inner switch, unchanged ...
        }
        return;
    }
    else if constexpr (I == Isa::Vnni)
    {
        switch (mat.bits * 4 + m - 1)
        {
            // ... the old `case Isa::Vnni:` inner switch, unchanged ...
        }
        return;
    }
    else if constexpr (I == Isa::Bw)
    {
        switch (mat.bits * 4 + m - 1)
        {
            // ... the old `case Isa::Bw:` inner switch, unchanged ...
        }
        return;
    }
    else if constexpr (I == Isa::Avx2)
    {
        switch (mat.bits)
        {
            // ... the old `case Isa::Avx2:` inner switch, unchanged ...
        }
    }
    else
    {
        switch (mat.bits)
        {
            // ... the old `case Isa::Scalar:` inner switch, unchanged ...
        }
    }
}
```

The `// ...` lines above stand for text you move, not text you write: cut each old inner `switch` block and paste it into its arm without editing a character. Afterwards `git diff -w` on this function must show only the outer-dispatch lines changing.

- [ ] **Step 5: Template `run_tiles`**

```cpp
template <Isa I>
void run_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Bw && ACT_ROWS == 2 && EXL3_MOE_CPU_ACT_BLOCK == 128)
    {
        if (m == 1 && mat.bits == 3 && act_blocked(mat.k))
        {
            register_tiles(mat,in,tout,tn0,tn1);
            return;
        }
    }
    if (I == Isa::Scalar || (ACT_ROWS == 1 && !act_blocked(mat.k)))
    {
        run_tiles_raw<I>(mat, in, tout, m, tn0, tn1);
        return;
    }
    // ... rest of the old body unchanged, except both `run_tiles_raw(` calls become `run_tiles_raw<I>(` ...
}
```

- [ ] **Step 6: Template the output transforms**

`transform_out_avx2` gains `template <Isa I>` before `M1_TARGET_AVX2` and calls `hadamard_128_avx2<I>(v);`. `transform_out` becomes:

```cpp
template <Isa I>
__attribute__((noipa)) void transform_out(const MoeCpuMatrix& mat, float* tout, int m)
{
    if constexpr (I != Isa::Scalar) { transform_out_avx2<I>(mat, tout, m); return; }
    else
    {
        for (int r = 0; r < m; ++r)
            for (int block = 0; block < mat.n; block += 128)
            {
                float* v = tout + static_cast<size_t>(r) * mat.n + block;
                hadamard_128<I>(v);
                if (mat.bias)
                    for (int i = 0; i < 128; ++i)
                        v[i] = v[i] * HAD_SCALE * half_to_float(mat.svh[block + i])
                               + half_to_float(mat.bias[block + i]);
                else
                    for (int i = 0; i < 128; ++i)
                        v[i] *= HAD_SCALE * half_to_float(mat.svh[block + i]);
            }
    }
}
```

- [ ] **Step 7: Template the phase helpers**

`middle_blocks` and `forward_phase_index` each declare a local `const int I = …interm_size;`, which would shadow the new template parameter `I` (a hard error in GCC). In both, rename that local to `I_` and every use of it in the function body (`nb=I/128`, `MAX_M*I`, `size_t(r)*I`, `H / 16`-style sizes that use it, `I / 16`). Do this first, then:

- `template<bool Wide = false> void middle_blocks(...)` → `template<Isa I, bool Wide = false> void middle_blocks(...)`; inside, `transform_out(` → `transform_out<I>(` (twice) and `prepare_block_avx2(` → `prepare_block_avx2<I>(`.
- `prepare_gu_blocks`: same template change; `prepare_block_avx2(` → `prepare_block_avx2<I>(`.
- `transform_owned_blocks` gains `template <Isa I>`; `transform_out(` → `transform_out<I>(`.
- `forward_phase_index`: delete the stray `inline` line above it and give it `template <Isa I>`. Inside:
  - `g_isa==Isa::Bw` → `I==Isa::Bw`; `g_isa!=Isa::Scalar` → `I!=Isa::Scalar` (plain `if`: both arms compile for every tier).
  - `prepare_gu_blocks<true>(` / `<false>(` → `prepare_gu_blocks<I, true>(` / `<I, false>(`; the same for `middle_blocks`.
  - `prepare_rows(` → `prepare_rows<I>(`, `run_tiles(` → `run_tiles<I>(`, `transform_out(` → `transform_out<I>(`, `transform_owned_blocks(` → `transform_owned_blocks<I>(`.

- [ ] **Step 8: Move the team into `run_team<I>` and dispatch once**

In `forward_raw`, everything from `// Freeze the configured cores once.` to the final `printf` moves into a new function placed just above `forward_raw` (inside the anonymous namespace is not required; place it after `forward_phase_index`, before `} // namespace`):

```cpp
// The forward's OpenMP team for tier I: pins worker i to the configured core i, runs the phases with a barrier after
// each, and checks the team. Phase 4 (the whole-row down transform) is folded into phase 3's owned blocks.
template <Isa I>
void run_team(ForwardCtx& ctx, int count, bool grouped)
{
    // Freeze the configured cores once. Steady-state forwards acquire no pool mutex.
    if (!g_compute_started.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_cores_mutex);
        if (!g_compute_started.load(std::memory_order_relaxed)) {
            g_compute_cores = g_configured_cores;
            g_compute_started.store(true, std::memory_order_release);
        }
    }
    TORCH_CHECK(g_compute_cores.empty() || size_t(count)<=g_compute_cores.size(),
                "CPU expert worker count exceeds configured cores");
    const bool prof=g_prof_enabled.load(std::memory_order_relaxed);
    double phase_us[6]{};
    std::atomic<int> pin_error{0};
    std::atomic<int> actual_workers{0};
    #pragma omp parallel num_threads(count) shared(ctx,pin_error,actual_workers,phase_us)
    {
        // ... the old parallel-region body verbatim, with `forward_phase_index(&ctx,worker,n,phase);`
        //     replaced by `forward_phase_index<I>(&ctx,worker,n,phase);` ...
    }
    TORCH_CHECK(!pin_error.load(),"cannot pin CPU expert worker to its configured core");
    TORCH_CHECK(actual_workers.load()==count,"OpenMP returned fewer CPU expert workers than requested");
    if(prof)printf("moe_cpu phases(us): %.1f %.1f %.1f %.1f %.1f %.1f\n",phase_us[0],phase_us[1],phase_us[2],phase_us[3],phase_us[4],phase_us[5]);
}

using TeamFn = void (*)(ForwardCtx&, int, bool);

TeamFn team_for(Isa isa)
{
    switch (isa) {
        case Isa::Scalar: return run_team<Isa::Scalar>;
        case Isa::Avx2:   return run_team<Isa::Avx2>;
        case Isa::Bw:     return run_team<Isa::Bw>;
        case Isa::Vnni:   return run_team<Isa::Vnni>;
        case Isa::Vbmi:   return run_team<Isa::Vbmi>;
    }
    return run_team<Isa::Scalar>;
}
```

`grouped_traversal=grouped;` stays as the first line of the parallel body for now (Task 4 removes the `thread_local`). `forward_raw` keeps its `compact_forward` computation (still `g_isa==Isa::Bw && …`) and its arena setup, computes `const int count=threads>0?threads:1;` and `const bool grouped=compact_forward && nc==1;` as before, then ends with:

```cpp
    team_for(g_isa)(ctx, count, grouped);
    give_back();
}
```

Because `forward_raw` is `static` and outside the anonymous namespace while `run_team` is declared inside it, `run_team`/`team_for` must be declared before `} // namespace` (they are file-local either way).

- [ ] **Step 9: Check that no forward code still reads `g_isa`**

Run: `grep -n "g_isa" python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`
Expected: only the declaration, `detect_isa`'s definition line (`const Isa g_isa = …`), the four `exl3_moe_cpu_has_*` lines, `compact_forward`'s `g_isa==Isa::Bw` in `forward_raw`, and `team_for(g_isa)`.

- [ ] **Step 10: Commit, push, run the gate**

```bash
git add python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp
git commit -m "$(cat <<'EOF'
refactor(exl3-cpu): the forward's ISA tier is a template parameter, chosen once per call

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push
```

Then run the gate (Working environment) with `OUT=…/task2`. Expected: `ALL GREEN (check)`. A compile error in `bench-build` or `dump-bw-table` names the missed call site; fix it and push a new commit (do not amend).

---

### Task 3: Layer facts and the `TableExperts` accessor

The forward stops reaching into `MoeCpuLayer`: it reads layer facts from `LayerInfo` and matrices through an accessor, so Task 5 can add a second accessor without touching the phases.

**Files:**
- Create: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/experts.hpp`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`

**Interfaces:**
- Consumes: Task 2's templates.
- Produces: `struct LayerInfo { int num_experts, hidden, intermediate; bool gated; int activation; float act_limit; }`; `struct TableExperts { const MoeCpuLayer* layer; const MoeCpuMatrix& gate(int) const; …up…; …down…; }`; `struct RegisteredLayer { LayerInfo info; std::unique_ptr<MoeCpuLayer> table; }`; `ForwardCtx::info` (replaces `ForwardCtx::layer`); every phase function gains a trailing-template `class Experts` and a `const Experts& E` parameter right after the context; `run_team<I, Experts>(ForwardCtx&, const Experts&, int count, bool grouped)`.

- [ ] **Step 1: Create `experts.hpp`**

```cpp
// Included by moe_mul1.cpp inside its anonymous namespace, after MoeCpuMatrix/MoeCpuLayer are declared.
// The forward reads a layer through these: LayerInfo for the facts, an Experts accessor for each expert's
// gate/up/down matrix. Accessors are cheap to copy and allocate nothing.

// What a forward needs to know about a layer besides its matrices.
struct LayerInfo
{
    int num_experts;
    int hidden;        // k of gate/up, n of down
    int intermediate;  // n of gate/up, k of down
    bool gated;
    int activation;    // 0 silu, 1 gelu, 2 relu2 (gateless), 3 swiglu_oai
    float act_limit;
};

// make_layer's registration: one MoeCpuMatrix per expert and projection, wherever each tensor lives.
struct TableExperts
{
    const MoeCpuLayer* layer;
    const MoeCpuMatrix& gate(int e) const { return layer->gates[e]; }
    const MoeCpuMatrix& up(int e) const { return layer->ups[e]; }
    const MoeCpuMatrix& down(int e) const { return layer->downs[e]; }
};
```

- [ ] **Step 2: Include it and replace the registry**

Move the `// MoE Layer registry` block (its comment, `g_layers`, `g_layers_mutex`) to just after `struct Chunk { … };` under `// Forward driver`: Task 4's `Dsv41Shape` names `Chunk`, and Task 5's registry entry names `Dsv41Shape`'s sibling `GenericShape`, so the order is fixed now as `Chunk` → `experts.hpp` → (Task 4: `shapes.hpp`) → registry → `ForwardCtx`. Add `#include "experts.hpp"` right after `struct Chunk`, then replace the registry globals:

```cpp
// A registered layer: make_layer's per-expert tables (table), or, from Task 5, the slab registration's view.
struct RegisteredLayer
{
    LayerInfo info;
    std::unique_ptr<MoeCpuLayer> table;
};

std::vector<std::unique_ptr<RegisteredLayer>> g_layers;
std::mutex g_layers_mutex;
```

`exl3_moe_cpu_make_layer`: keep building `layer` exactly as today (still `new MoeCpuLayer`, wrap it immediately: `auto table = std::unique_ptr<MoeCpuLayer>(new MoeCpuLayer);` and use `table->` for every `layer->`), then replace the final three lines with:

```cpp
    auto entry = std::make_unique<RegisteredLayer>();
    entry->info = {table->num_experts, table->hidden_size, table->interm_size, !table->gates.empty(),
                   table->activation, table->act_limit};
    entry->table = std::move(table);
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    g_layers.push_back(std::move(entry));
    return static_cast<int64_t>(g_layers.size() - 1);
```

`exl3_moe_cpu_free_layer`: `delete g_layers[handle]; g_layers[handle] = nullptr;` → `g_layers[handle].reset();`.

`get_layer`:

```cpp
static const RegisteredLayer& get_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(g_layers.size()) && g_layers[handle], "invalid CPU MoE layer handle");
    return *g_layers[handle];
}
```

- [ ] **Step 3: `ForwardCtx` carries `LayerInfo`**

In `struct ForwardCtx`, `const MoeCpuLayer* layer;` → `LayerInfo info;`.

- [ ] **Step 4: Thread the accessor through the phases**

Apply these rewrites in `middle_blocks`, `prepare_gu_blocks`, `forward_phase_index` and `run_team` (and nowhere else):

| old | new |
|---|---|
| `template<Isa I, bool Wide = false> void middle_blocks(ForwardCtx& c,int worker,int num_workers)` | `template<Isa I, bool Wide = false, class Experts> void middle_blocks(ForwardCtx& c,const Experts& E,int worker,int num_workers)` |
| same for `prepare_gu_blocks` | same shape |
| `template <Isa I> void forward_phase_index(void* vctx, int worker, int num_workers, int phase)` | `template <Isa I, class Experts> void forward_phase_index(ForwardCtx& c, const Experts& E, int worker, int num_workers, int phase)` — delete its first line `ForwardCtx& c = *static_cast<ForwardCtx*>(vctx);` and `const MoeCpuLayer& L = *c.layer;` |
| `const auto& L=*c.layer;` (helpers) | delete |
| `L.hidden_size` | `c.info.hidden` |
| `L.interm_size` | `c.info.intermediate` |
| `!L.gates.empty()` / `L.gates.empty()` | `c.info.gated` / `!c.info.gated` |
| `L.activation`, `L.act_limit` | `c.info.activation`, `c.info.act_limit` |
| `L.gates[x]`, `L.ups[x]`, `L.downs[x]` | `E.gate(x)`, `E.up(x)`, `E.down(x)` |
| `prepare_gu_blocks<I, true>(c,worker,num_workers)` etc. | `prepare_gu_blocks<I, true>(c,E,worker,num_workers)` |
| in `run_team`: `forward_phase_index<I>(&ctx,worker,n,phase);` | `forward_phase_index<I>(ctx,E,worker,n,phase);` |

Two expressions bind a reference to a conditional; keep them as `const MoeCpuMatrix& mat = up ? E.up(ch.expert) : E.gate(ch.expert);` (with `TableExperts` both arms are lvalues; with Task 5's by-value accessor the reference extends the temporary's lifetime). The `auto mat=E.gate(ch.expert);mat.n=128;…` copies in `middle_blocks` are already copies.

`run_team` becomes `template <Isa I, class Experts> void run_team(ForwardCtx& ctx, const Experts& E, int count, bool grouped)`. Do not add `E` to the `shared(...)` clause: the region has no `default(none)`, so `E` is shared by default, and listing a reference there is not portable across GCC versions. `TeamFn`/`team_for` become templates on `Experts`:

```cpp
template <class Experts>
using TeamFn = void (*)(ForwardCtx&, const Experts&, int, bool);

template <class Experts>
TeamFn<Experts> team_for(Isa isa)
{
    switch (isa) {
        case Isa::Scalar: return run_team<Isa::Scalar, Experts>;
        case Isa::Avx2:   return run_team<Isa::Avx2, Experts>;
        case Isa::Bw:     return run_team<Isa::Bw, Experts>;
        case Isa::Vnni:   return run_team<Isa::Vnni, Experts>;
        case Isa::Vbmi:   return run_team<Isa::Vbmi, Experts>;
    }
    return run_team<Isa::Scalar, Experts>;
}
```

- [ ] **Step 5: `forward_raw` uses the entry**

At the top of `forward_raw`:

```cpp
    const RegisteredLayer& layer = get_layer(handle);
    const LayerInfo& info = layer.info;
    const TableExperts E{layer.table.get()};
```

Then `ctx.layer = layer;` → `ctx.info = info;`; `layer->hidden_size` → `info.hidden`; `layer->interm_size` → `info.intermediate`; `layer->num_experts` → `info.num_experts`; `!layer->gates.empty()` → `info.gated`; the `compact_forward` loop's `layer->gates[ch.expert]` / `ups` / `downs` → `E.gate(ch.expert)` / `E.up(…)` / `E.down(…)`. The call becomes `team_for<TableExperts>(g_isa)(ctx, E, count, grouped);`.

- [ ] **Step 6: Check nothing in the forward still touches `MoeCpuLayer`**

Run: `grep -n "MoeCpuLayer\|->gates\|->ups\|->downs\|\.gates\[\|\.ups\[\|\.downs\[" python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`
Expected: only `exl3_moe_cpu_make_layer`'s body (building `table`), `RegisteredLayer`, and `experts.hpp`'s `TableExperts`.

- [ ] **Step 7: Commit, push, run the gate**

```bash
git add python/sglang/srt/layers/quantization/exl3_cpu/optimized/experts.hpp python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp
git commit -m "$(cat <<'EOF'
refactor(exl3-cpu): the forward reads layers through LayerInfo and an experts accessor

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push
```

Gate with `OUT=…/task3`. Expected: `ALL GREEN (check)`.

---

### Task 4: `ForwardPlan<Shape, Isa>`, `PlanTraits`, and the DSV4.1 specialization

Replaces the runtime `compact_forward` flag, the `thread_local grouped_traversal`, and `single_expert_quant512` with a plan chosen per call: `ForwardPlan<Dsv41Shape, Isa::Bw>` when `Dsv41Shape::accepts` the call on a BW host, else `ForwardPlan<GenericShape, I>`.

**Files:**
- Create: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/shapes.hpp`
- Create: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/forward_plan.hpp`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/register.hpp`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/traversal.hpp`

**Interfaces:**
- Consumes: Task 3's `LayerInfo`, `TableExperts`, `ForwardCtx`, `Chunk`, `ForwardArena`.
- Produces: `GenericShape`, `Dsv41Shape { kHidden=5120, kIntermediate=2304, kBits=3; static bool accepts(const LayerInfo&, const Experts&, const std::vector<Chunk>&); }`, `PlanTraits<Shape, I>`, `ForwardPlan<Shape, I>::run(ForwardCtx&, const Experts&, ForwardArena&, int threads)`; `run_tiles<I>(…, bool grouped)`, `register_tiles(…, bool grouped)`, `traversal_tiles(…, bool grouped)`. Task 5 reuses `Shape::kFixed`, `Shape::kHidden/kIntermediate/kBits`.

- [ ] **Step 1: `grouped` becomes a parameter**

- `moe_mul1.cpp`: delete `thread_local bool grouped_traversal = false;` and its comment line above it.
- `traversal.hpp`: `M1_TARGET_BW void traversal_tiles(const MoeCpuMatrix& mat,const PreparedIn& in, float* tout,int t0,int t1)` gains a last parameter `bool grouped`; inside, `grouped_traversal ?` → `grouped ?`.
- `register.hpp`: `register_tiles(const MoeCpuMatrix& mat,const PreparedIn& in,float* tout,int t0,int t1)` gains `bool grouped` and passes it: `traversal_tiles(mat,in,tout,t0,t1,grouped);`.
- `moe_mul1.cpp`: `run_tiles<I>(…, int tn0, int tn1)` gains `bool grouped = false` and calls `register_tiles(mat,in,tout,tn0,tn1,grouped);`.

- [ ] **Step 2: Create `shapes.hpp` and include it**

Include it with `#include "shapes.hpp"` directly after `#include "experts.hpp"` (which Task 3 placed after `struct Chunk`). The file:

```cpp
// Included by moe_mul1.cpp inside its anonymous namespace, after struct Chunk and experts.hpp.
//
// The shapes a forward plan can be specialized for. GenericShape takes every dimension from the layer. Dsv41Shape is
// DeepSeek V4.1's routed expert: hidden 5120, intermediate 2304, 3-bit, unswizzled, gated.

struct GenericShape
{
    static constexpr bool kFixed = false;
    static int hidden(const LayerInfo& info) { return info.hidden; }
    static int intermediate(const LayerInfo& info) { return info.intermediate; }
};

struct Dsv41Shape
{
    static constexpr bool kFixed = true;
    static constexpr int kHidden = 5120, kIntermediate = 2304, kBits = 3;
    static constexpr int hidden(const LayerInfo&) { return kHidden; }
    static constexpr int intermediate(const LayerInfo&) { return kIntermediate; }

    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // the shape and is gated, every routed expert is unswizzled 3-bit, and every chunk holds one token (a prefill
    // chunk of two tokens takes the generic plan).
    template <class Experts>
    static bool accepts(const LayerInfo& info, const Experts& E, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        if (info.hidden != kHidden || info.intermediate != kIntermediate || !info.gated) return false;
        for (const auto& ch : chunks) {
            const MoeCpuMatrix& g = E.gate(ch.expert);
            const MoeCpuMatrix& u = E.up(ch.expert);
            const MoeCpuMatrix& d = E.down(ch.expert);
            if (ch.m != 1 || g.bits != kBits || u.bits != kBits || d.bits != kBits || g.swz || u.swz || d.swz)
                return false;
        }
        return true;
    }
};
```

- [ ] **Step 3: Create `forward_plan.hpp`**

The phase bodies move here from `forward_phase_index` unchanged except for the edits listed below the code. Write the file as:

```cpp
// Included by moe_mul1.cpp inside its anonymous namespace, after the phase helpers (prepare_rows, run_tiles,
// transform_out, middle_blocks, prepare_gu_blocks, transform_owned_blocks, assign_gemvs) and ForwardCtx/ForwardArena.
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is
// the ISA tier. The primary template is the generic plan. PlanTraits<Dsv41Shape, Isa::Bw> turns on the fast path that
// was measured and validated bit-exact on AVX-512BW (exl3_cpu/optimized/README.txt); every other (Shape, I) pair runs
// the generic plan.

// What a (Shape, ISA) plan turns on. The primary template is the generic plan.
template <class Shape, Isa I>
struct PlanTraits
{
    static constexpr bool kCompactScratch = false;    // int16 compact activations instead of the int32 splats
    static constexpr bool kGroupedTraversal = false;  // range-aware multi-band traversal when one expert is routed
    static constexpr bool kWideSingleExpert = false;  // 512-wide quantization for one token through one expert
};

// DeepSeek V4.1 on AVX-512BW: the validated fast path.
template <>
struct PlanTraits<Dsv41Shape, Isa::Bw>
{
    static constexpr bool kCompactScratch = true;
    static constexpr bool kGroupedTraversal = true;
    static constexpr bool kWideSingleExpert = true;
};

template <class Shape, Isa I>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape, I>;

    // Sizes this call's scratch from the arena, runs the team, and returns. ctx.chunks must be non-empty.
    template <class Experts>
    static void run(ForwardCtx& ctx, const Experts& E, ForwardArena& ar, int threads)
    {
        prepare_scratch(ctx, ar);
        const int nc = static_cast<int>(ctx.chunks.size());
        const bool grouped = Traits::kGroupedTraversal && nc == 1;
        const bool wide = Traits::kWideSingleExpert && ctx.m_total == 1 && nc == 1;
        run_team(ctx, E, threads > 0 ? threads : 1, grouped, wide);
    }

private:
    static void prepare_scratch(ForwardCtx& ctx, ForwardArena& ar)
    {
        constexpr bool compact = Traits::kCompactScratch;
        const int nc = static_cast<int>(ctx.chunks.size());
        const int H = Shape::hidden(ctx.info);
        const int I_ = Shape::intermediate(ctx.info);
        // ... the old arena block of forward_raw, from `auto grow = [](auto& v, size_t n) …` through the end of the
        //     `if (EXL3_MOE_CPU_ACT_BLOCK) { … }` block, verbatim, with `compact_forward` replaced by `compact` and
        //     the local `I` renamed `I_` (the template parameter is named I) ...
    }

    template <class Experts>
    static void run_team(ForwardCtx& ctx, const Experts& E, int count, bool grouped, bool wide)
    {
        // ... Task 3's run_team body verbatim, except:
        //     - drop the `grouped_traversal=grouped;` line;
        //     - the phase loop calls `phase(ctx,E,worker,n,p,grouped,wide);` ...
    }

    // One phase for this worker. Phases: 0 prepare gate/up inputs, 1 gate/up GEMVs, 2 output transform + activation +
    // down input, 3 down GEMVs with their output transform, 5 routing-weighted accumulate. Phase 4, the old whole-row
    // down transform, no longer exists: phase 3's owned blocks include it.
    template <class Experts>
    static void phase(ForwardCtx& c, const Experts& E, int worker, int num_workers, int phase, bool grouped, bool wide)
    {
        // ... Task 3's forward_phase_index<I, Experts> body verbatim, except the edits listed below ...
    }
};
```

The `// ...` comments mark text you move, not text you write. Edits while moving:

- In `prepare_scratch`: `compact_forward` → `compact` (now `constexpr`, so `if(!compact)` / `compact?nullptr:…` fold; leave the expressions as they are). In `prepare_scratch` and `phase`, the locals become `const int H = Shape::hidden(ctx.info);` (`c.info` in `phase`) and `const int I_ = Shape::intermediate(…);` (Task 2 already renamed the old local `I` to `I_`; `forward_raw`'s arena block still says `I`, so rename it while moving).
- In `phase`: `single_expert_quant512(c)` (two places) → `wide`; delete the whole `case 4:` block; `run_tiles<I>(…, t0, t1)` → `run_tiles<I>(…, t0, t1, grouped)` in phase 1 and phase 3; the helper calls `prepare_gu_blocks<I, …>`, `middle_blocks<I, …>`, `prepare_rows<I>`, `transform_out<I>`, `transform_owned_blocks<I>` keep `I` (the template parameter).
- In `run_team`'s phase loop keep the old skeleton exactly — `for(int phase=0;phase<6;++phase) { if(phase==4)continue; …; if(phase<5) { #pragma omp barrier } … }` — so the barrier count and `phase_us` indices do not change; rename the loop variable to `p` to avoid shadowing the member function `phase`.

- [ ] **Step 4: Remove the old pieces from `moe_mul1.cpp`**

Delete `single_expert_quant512`, `forward_phase_index`, `run_team`, `TeamFn` and `team_for`. Add `#include "forward_plan.hpp"` where `forward_phase_index` was (after `transform_owned_blocks`, before `} // namespace`).

- [ ] **Step 5: `forward_raw` picks the plan**

Delete `compact_forward`, its loop, the whole arena-sizing block, and the `count`/`grouped` locals. Keep the chunking and the `if (!nc) { give_back(); return; }` early return. After it:

```cpp
    if (g_isa == Isa::Bw && Dsv41Shape::accepts(info, E, ctx.chunks))
        ForwardPlan<Dsv41Shape, Isa::Bw>::run(ctx, E, ar, threads);
    else
        switch (g_isa) {
            case Isa::Scalar: ForwardPlan<GenericShape, Isa::Scalar>::run(ctx, E, ar, threads); break;
            case Isa::Avx2:   ForwardPlan<GenericShape, Isa::Avx2>::run(ctx, E, ar, threads); break;
            case Isa::Bw:     ForwardPlan<GenericShape, Isa::Bw>::run(ctx, E, ar, threads); break;
            case Isa::Vnni:   ForwardPlan<GenericShape, Isa::Vnni>::run(ctx, E, ar, threads); break;
            case Isa::Vbmi:   ForwardPlan<GenericShape, Isa::Vbmi>::run(ctx, E, ar, threads); break;
        }
    give_back();
```

Equivalence with the old gate: `compact_forward` was `g_isa==Bw && ACT_ROWS==2 && BLOCK==128 && H==5120 && I==2304 && gated && ∀chunk (m==1 && 3-bit ×3 && unswizzled ×3)`; `accepts` checks the same conjunction. `grouped` was `compact_forward && nc==1`; it is now `kGroupedTraversal (true only for the DSV4.1 plan) && nc==1`. `single_expert_quant512` was `m_total==1 && nc==1 && prep_d[0].compact != nullptr` (non-null iff `compact_forward`); `wide` is `kWideSingleExpert && m_total==1 && nc==1`.

- [ ] **Step 6: Commit, push, run the gate**

```bash
git add python/sglang/srt/layers/quantization/exl3_cpu/optimized/shapes.hpp \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/forward_plan.hpp \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/register.hpp \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/traversal.hpp
git commit -m "$(cat <<'EOF'
refactor(exl3-cpu): ForwardPlan<Shape, Isa> with the DSV4.1 fast path as a PlanTraits specialization

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push
```

Gate with `OUT=…/task4`. Expected: `ALL GREEN (check)`. The `dsv41/t2k3` cases in every `compare-*` exercise the fallback from the DSV4.1 plan; `bare-validate` exercises the DSV4.1 plan itself.

---

### Task 5: `StridedExperts<Shape>` and the slab registration C ABI

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/experts.hpp`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/cpu_experts_cabi.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack_fixture.cpp:144-166`
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/README.txt`

**Interfaces:**
- Consumes: Task 4's `GenericShape`/`Dsv41Shape` (`kFixed`, `kHidden`, `kIntermediate`, `kBits`), `ForwardPlan`, `RegisteredLayer`.
- Produces: `int sglang_exl3_cpu_experts_register_slabs(const void* const* slabs, int32_t capacity, int32_t hidden, int32_t intermediate, int32_t bits, int32_t swizzled, float act_limit, int64_t* handle) noexcept` — 0 and `*handle` on success, 1 on a kernel error, 2 on invalid arguments; the handle is freed by `exl3_moe_cpu_free_layer` and runs through `exl3_moe_cpu_forward`/`_forward_raw`/`sglang_exl3_cpu_experts_forward` like any other. Task 6's trait calls it.

- [ ] **Step 1: Add the strided accessor to `experts.hpp`**

`experts.hpp` is included before `forward_plan.hpp`, but `StridedExperts` is a template, so it may name `Shape::kFixed` etc. without the shapes being declared yet. Append:

```cpp
// The pinned tier's slabs, in exl3_expert_format.EXL3_STREAMED_NAMES order: one row per expert slot.
enum SlabName { kW13Trellis, kW13Suh, kW13Svh, kW2Trellis, kW2Suh, kW2Svh, kSlabNames };

// Bytes of one slot's row of each slab. w13 rows hold gate (part 0) then up (part 1); w2 rows hold down.
// A trellis is [k/16][n/16][16 * bits] uint16, i.e. k * n * bits / 8 bytes; sign vectors are fp16.
struct SlabRowBytes
{
    size_t bytes[kSlabNames];

    static constexpr SlabRowBytes of(int hidden, int intermediate, int bits)
    {
        const size_t trellis = size_t(hidden) * size_t(intermediate) * size_t(bits) / 8;
        return {{2 * trellis, 2 * 2 * size_t(hidden), 2 * 2 * size_t(intermediate),
                 trellis, 2 * size_t(intermediate), 2 * size_t(hidden)}};
    }
};

// The slab registration: expert e of every slab at base + e * row bytes, nothing stored per expert. A fixed Shape
// (Dsv41Shape) makes the dimensions, and so every row size, compile-time constants; GenericShape reads them from the
// registered values. The kernel keeps no reference to the slabs: the registrant keeps them alive.
template <class Shape>
struct StridedExperts
{
    const uint8_t* base[kSlabNames];
    int hidden, intermediate, bits;  // as registered; a fixed Shape's constants take their place
    int swz;

    int H() const { if constexpr (Shape::kFixed) return Shape::kHidden; else return hidden; }
    int I() const { if constexpr (Shape::kFixed) return Shape::kIntermediate; else return intermediate; }
    int B() const { if constexpr (Shape::kFixed) return Shape::kBits; else return bits; }

    MoeCpuMatrix gate(int e) const { return w13_part(e, 0); }
    MoeCpuMatrix up(int e) const { return w13_part(e, 1); }
    MoeCpuMatrix down(int e) const
    {
        const SlabRowBytes r = SlabRowBytes::of(H(), I(), B());
        return matrix(at(kW2Trellis, e, r), at(kW2Suh, e, r), at(kW2Svh, e, r), I(), H());
    }

    // The same slabs under another shape's assumptions (the caller has checked they hold).
    template <class Other>
    StridedExperts<Other> as() const
    {
        StridedExperts<Other> o;
        std::copy(std::begin(base), std::end(base), std::begin(o.base));
        o.hidden = hidden; o.intermediate = intermediate; o.bits = bits; o.swz = swz;
        return o;
    }

private:
    const uint8_t* at(SlabName name, int e, const SlabRowBytes& r) const
    {
        return base[name] + size_t(e) * r.bytes[name];
    }

    MoeCpuMatrix w13_part(int e, int part) const
    {
        const SlabRowBytes r = SlabRowBytes::of(H(), I(), B());
        return matrix(at(kW13Trellis, e, r) + part * (r.bytes[kW13Trellis] / 2),
                      at(kW13Suh, e, r) + part * (r.bytes[kW13Suh] / 2),
                      at(kW13Svh, e, r) + part * (r.bytes[kW13Svh] / 2), H(), I());
    }

    MoeCpuMatrix matrix(const uint8_t* trellis, const uint8_t* suh, const uint8_t* svh, int k, int n) const
    {
        MoeCpuMatrix m;
        m.trellis = reinterpret_cast<const uint16_t*>(trellis);
        m.suh = reinterpret_cast<const at::Half*>(suh);
        m.svh = reinterpret_cast<const at::Half*>(svh);
        m.bias = nullptr;
        m.k = k;
        m.n = n;
        m.bits = B();
        m.swz = swz;
        return m;
    }
};
```

Its data members are public (the private section holds only functions), so `StridedExperts<GenericShape> strided{}` value-initializes it and `as<Other>()` needs no friendship.

- [ ] **Step 2: The registry holds either kind; `forward_raw` dispatches on it**

The include order from Tasks 3–4 (`Chunk` → `experts.hpp` → `shapes.hpp` → registry) already declares `GenericShape` before the registry. Give the entry the strided view:

```cpp
// A registered layer: make_layer's per-expert tables, or the slab registration's view (table == nullptr).
struct RegisteredLayer
{
    LayerInfo info;
    std::unique_ptr<MoeCpuLayer> table;
    StridedExperts<GenericShape> strided{};
};
```

In `forward_raw`, replace `const TableExperts E{layer.table.get()};` and the Task 4 plan dispatch with a helper declared above `forward_raw` (inside the anonymous namespace, after `#include "forward_plan.hpp"`):

```cpp
// Runs the call's plan. E reads the layer's experts under the generic plan; D reads the same experts under the
// DSV4.1 plan's assumptions (for a strided layer, the compile-time-shaped view of the same slabs).
template <class Experts, class Dsv41Experts>
void run_plan(ForwardCtx& ctx, const Experts& E, const Dsv41Experts& D, ForwardArena& ar, int threads)
{
    if (g_isa == Isa::Bw && Dsv41Shape::accepts(ctx.info, E, ctx.chunks)) {
        ForwardPlan<Dsv41Shape, Isa::Bw>::run(ctx, D, ar, threads);
        return;
    }
    switch (g_isa) {
        case Isa::Scalar: ForwardPlan<GenericShape, Isa::Scalar>::run(ctx, E, ar, threads); return;
        case Isa::Avx2:   ForwardPlan<GenericShape, Isa::Avx2>::run(ctx, E, ar, threads); return;
        case Isa::Bw:     ForwardPlan<GenericShape, Isa::Bw>::run(ctx, E, ar, threads); return;
        case Isa::Vnni:   ForwardPlan<GenericShape, Isa::Vnni>::run(ctx, E, ar, threads); return;
        case Isa::Vbmi:   ForwardPlan<GenericShape, Isa::Vbmi>::run(ctx, E, ar, threads); return;
    }
}
```

and in `forward_raw`, where Task 4's dispatch was:

```cpp
    if (layer.table) {
        const TableExperts t{layer.table.get()};
        run_plan(ctx, t, t, ar, threads);
    } else {
        run_plan(ctx, layer.strided, layer.strided.as<Dsv41Shape>(), ar, threads);
    }
    give_back();
```

`Dsv41Shape::accepts` must also guard the strided view's registered dimensions: it already compares `info.hidden/intermediate` and each expert's `bits`/`swz`, and for a strided layer `info` and the matrices come from the registered values, so `as<Dsv41Shape>()` is used only when they equal the constants.

- [ ] **Step 3: The C ABI**

`cpu_experts_cabi.h`, after `sglang_exl3_cpu_experts_set_cores`:

```c
// Register `capacity` expert slots laid out as the pinned tier's slab rows. slabs[6] are the bases of w13_trellis,
// w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts
// s rows in, w13 rows holding gate then up. Gated SiLU clamped at act_limit. The kernel stores only the pointers:
// keep the slabs alive until exl3_moe_cpu_free_layer(*handle). Returns 0 and the layer handle, 1 on a kernel error,
// 2 on invalid arguments.
int sglang_exl3_cpu_experts_register_slabs(const void* const* slabs, int32_t capacity, int32_t hidden,
    int32_t intermediate, int32_t bits, int32_t swizzled, float act_limit, int64_t* handle) EXL3_CPU_NOEXCEPT;
```

`moe_mul1.cpp`, after `sglang_exl3_cpu_experts_set_cores`:

```cpp
extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_register_slabs(
    const void* const* slabs, int32_t capacity, int32_t hidden, int32_t intermediate, int32_t bits, int32_t swizzled,
    float act_limit, int64_t* handle) noexcept
{
    if (!slabs || !handle || capacity < 1 || bits < 1 || bits > 8 || (swizzled != 0 && swizzled != 1)) return 2;
    // make_matrix's limits: 128-element blocks, and k (hidden for gate/up, intermediate for down) <= 8192 for the
    // int32 accumulators.
    if (hidden < 128 || intermediate < 128 || hidden % 128 || intermediate % 128 || hidden > 8192 || intermediate > 8192)
        return 2;
    if (!std::isfinite(act_limit) || act_limit < 0.0f) return 2;
    for (int i = 0; i < kSlabNames; ++i)
        if (!slabs[i]) return 2;
    try {
        auto entry = std::make_unique<RegisteredLayer>();
        entry->info = {capacity, hidden, intermediate, true, 0, act_limit};
        for (int i = 0; i < kSlabNames; ++i)
            entry->strided.base[i] = static_cast<const uint8_t*>(slabs[i]);
        entry->strided.hidden = hidden;
        entry->strided.intermediate = intermediate;
        entry->strided.bits = bits;
        entry->strided.swz = swizzled && bits != 8 ? 1 : 0;  // make_matrix's rule: K8 is never swizzled
        std::lock_guard<std::mutex> lock(g_layers_mutex);
        g_layers.push_back(std::move(entry));
        *handle = static_cast<int64_t>(g_layers.size() - 1);
        return 0;
    } catch (...) {
        return 1;
    }
}
```

`RegisteredLayer`, `g_layers`, `g_layers_mutex` and `kSlabNames` live in the anonymous namespace of the same translation unit, so this function sees them.

- [ ] **Step 4: The full-stack fixture registers through the slab ABI**

Replace `StackFixture::register_layer` in `bench/src/stack_fixture.cpp`:

```cpp
// Mirrors cpu_experts/exl3.py::register_layer: the row's six slabs by base pointer (kNames is EXL3_STREAMED_NAMES
// order), activation 0 (silu) with cpu_forward.cpp's limit 10, unswizzled 3-bit.
int64_t StackFixture::register_layer(int64_t row) const {
  const Impl& f = *impl_;
  const void* slabs[kNames];
  for (int n = 0; n < kNames; ++n) slabs[n] = f.set.slabs[row][n];
  int64_t handle = -1;
  const int status = sglang_exl3_cpu_experts_register_slabs(
      slabs, static_cast<int32_t>(kCapacity), static_cast<int32_t>(f.hidden), static_cast<int32_t>(f.intermediate), 3,
      0, 10.0f, &handle);
  if (status != 0)
    throw std::runtime_error("the kernel refused row " + std::to_string(row) + "'s slabs: status " +
                             std::to_string(status));
  return handle;
}
```

The fixture's slab row sizes (`row_bytes` in the constructor) are exactly `SlabRowBytes::of(hidden, intermediate, 3)`: `2 * nbytes(gate trellis)` = `2 * H*I*3/8`, `2 * nbytes(gate suh)` = `2 * 2H`, and so on. Leave `free_layer` as is.

- [ ] **Step 5: README**

In `optimized/README.txt`, under "Build and link", replace the sentence beginning `Use moe_mul1.h for the ATen layer-registration API` with:

```
Use moe_mul1.h for the ATen layer-registration API (exl3_moe_cpu_make_layer, one tensor per expert and projection)
and cpu_experts_cabi.h for the service API: forward, core configuration, and
sglang_exl3_cpu_experts_register_slabs, which registers a layer as the pinned tier's six slab base pointers (the
kernel keeps no reference: the caller keeps the slabs alive until exl3_moe_cpu_free_layer).
```

- [ ] **Step 6: Commit, push, run the gate with `slabs`**

```bash
git add python/sglang/srt/layers/quantization/exl3_cpu/optimized/experts.hpp \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/moe_mul1.cpp \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/cpu_experts_cabi.h \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/README.txt \
  python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack_fixture.cpp
git commit -m "$(cat <<'EOF'
feat(exl3-cpu): register a layer as slab base pointers (StridedExperts<Shape>)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push
```

Gate with `OUT=…/task5` and the fifth argument `slabs`. Expected: `compare-{bw,avx2,scalar}-slabs` each `16/16 bit-exact` against the **make_layer** baseline (strided registration reproduces the table registration on every tier, both shapes, including `dsv41/t*` on the DSV4.1 plan), `full-stack-validate` `Verified 48 bit-exact layer outputs` (the strided DSV4.1 plan against the frozen references), and `ALL GREEN (check)`.

---

### Task 6: `Exl3CpuQuantTrait` registers by base pointer

**Files:**
- Modify: `python/sglang/srt/layers/moe/cpu_experts/exl3.py`
- Modify: `test/registered/unit/kernels/test_cpu_expert_pool.py:224-275`
- Modify: `test/manual/dsv41/test_cpu_expert_pool_exl3.py`

**Interfaces:**
- Consumes: Task 5's `sglang_exl3_cpu_experts_register_slabs`.
- Produces: `Exl3CpuQuantTrait.register_layer(slabs, capacity) -> int` (unchanged signature; the pool and service are untouched) and `free_layer(handle)`, which now also drops the trait's reference to the slabs.

- [ ] **Step 1: Write the failing registered test**

Replace `FakeExt`, `_exl3_slabs` and `test_exl3_trait_registers_each_slot_as_the_right_slab_views` (lines 224–275) with:

```python
class FakeExt:
    def __init__(self):
        self.freed = []

    def exl3_moe_cpu_free_layer(self, handle):
        self.freed.append(handle)


def _exl3_slabs():
    i16, f16 = torch.int16, torch.float16
    n = CAP * 2 * (H // 16) * (INTER // 16) * 48
    return {
        "w13_trellis": torch.arange(n, dtype=i16).view(CAP, 2, H // 16, INTER // 16, 48),
        "w13_suh": torch.zeros(CAP, 2, H, dtype=f16),
        "w13_svh": torch.zeros(CAP, 2, INTER, dtype=f16),
        "w2_trellis": torch.zeros(CAP, INTER // 16, H // 16, 48, dtype=i16),
        "w2_suh": torch.zeros(CAP, INTER, dtype=f16),
        "w2_svh": torch.zeros(CAP, H, dtype=f16),
    }


class FakeRegisterSlabs:
    """Stands in for the kernel's sglang_exl3_cpu_experts_register_slabs; records each call's values."""

    def __init__(self, status=0):
        self.status, self.calls = status, []

    def __call__(self, bases, capacity, hidden, intermediate, bits, swizzled, act_limit, handle):
        self.calls.append(([bases[i] for i in range(6)], capacity, hidden, intermediate, bits, swizzled, act_limit))
        handle._obj.value = 40 + len(self.calls)
        return self.status


def _trait_with(monkeypatch, fake, **kw):
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0, **kw)
    monkeypatch.setattr(trait, "_native", lambda name: fake if name == "sglang_exl3_cpu_experts_register_slabs" else None)
    return trait


@pytest.mark.parametrize("tier_layout", [False, True], ids=["flat_w2", "tier_w2"])
def test_exl3_trait_registers_the_six_slab_bases(monkeypatch, tier_layout):
    """The CPU expert id is the host slot: the kernel addresses slot s at each slab's base plus s rows, gate and up as
    w13 parts 0 and 1. The pinned tier's w2 slabs carry a one-part axis ([slot, 1, ...]), which changes no row."""
    slabs = _exl3_slabs()
    if tier_layout:
        slabs = {n: (t.unsqueeze(1) if n.startswith("w2_") else t) for n, t in slabs.items()}
    fake = FakeRegisterSlabs()
    trait = _trait_with(monkeypatch, fake, swizzled=True)
    handle = trait.register_layer(slabs, CAP)
    names = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
    assert handle == 41
    assert fake.calls == [([slabs[n].data_ptr() for n in names], CAP, H, INTER, 3, 1, 10.0)]


def test_exl3_trait_keeps_the_slabs_alive_until_free(monkeypatch):
    """The kernel keeps only pointers: the trait holds the tensors until the layer is freed."""
    slabs = _exl3_slabs()
    trait = _trait_with(monkeypatch, FakeRegisterSlabs())
    handle = trait.register_layer(slabs, CAP)
    probe = weakref.ref(slabs["w2_svh"])
    del slabs
    assert probe() is not None
    trait.free_layer(handle)
    assert trait.ext.freed == [handle]
    assert probe() is None


@pytest.mark.parametrize(
    "name, bad",
    [
        ("w13_suh", lambda t: t.transpose(1, 2).contiguous().transpose(1, 2)),  # not contiguous
        ("w2_svh", lambda t: t.float()),  # wrong dtype
        ("w13_svh", lambda t: t[:, :, : INTER // 2]),  # wrong row size (and not contiguous)
        ("w2_trellis", lambda t: t[: CAP - 1]),  # fewer rows than the capacity
    ],
    ids=["noncontiguous", "dtype", "row_size", "rows"],
)
def test_exl3_trait_refuses_slabs_the_kernel_would_misaddress(monkeypatch, name, bad):
    slabs = _exl3_slabs()
    slabs[name] = bad(slabs[name])
    fake = FakeRegisterSlabs()
    with pytest.raises(ValueError, match=name):
        _trait_with(monkeypatch, fake).register_layer(slabs, CAP)
    assert fake.calls == []


def test_exl3_trait_reports_a_refused_registration(monkeypatch):
    trait = _trait_with(monkeypatch, FakeRegisterSlabs(status=2))
    with pytest.raises(RuntimeError, match="status 2"):
        trait.register_layer(_exl3_slabs(), CAP)
```

Add `import weakref` to the module's imports. `test_exl3_trait_refuses_a_kernel_that_would_pin_its_own_workers` keeps using `FakeExt()` unchanged.

`handle._obj.value` works because the trait passes `ctypes.byref(handle)`; `byref` objects expose the referenced object as `_obj`.

- [ ] **Step 2: Run it to verify it fails**

On divix01 (the registered suite needs the venv's torch):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-forward-plan && git fetch -q origin && git checkout -q --detach origin/exl3-cpu-forward-plan \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
     test/registered/unit/kernels/test_cpu_expert_pool.py -k exl3_trait; echo "EXIT=$?"'
```

(Push the test-only commit first, or apply this check after Step 3's push; the protocol forbids copying an uncommitted tree.) Expected before Step 3: the four new tests FAIL (the trait still calls `make_layer`, which `FakeExt` no longer has: `AttributeError`).

- [ ] **Step 3: Rewrite the trait's registration**

In `python/sglang/srt/layers/moe/cpu_experts/exl3.py`:

- Module docstring: replace "registers each streamed layer's pinned slab rows with it as views" with "registers each streamed layer's pinned slabs with it by base pointer".
- Delete `_one_part` (no longer used).
- `__init__` adds `self._slabs: dict[int, list[torch.Tensor]] = {}`.
- Replace `register_layer` and `free_layer`:

```python
    def register_layer(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> int:
        """Register one layer's first ``capacity`` slab rows with the kernel, by base pointer.

        The kernel addresses slot ``s`` of each slab at its base plus ``s`` rows, so each slab must be a contiguous
        CPU tensor of at least ``capacity`` rows of the format's row size; the trait keeps the tensors alive until
        ``free_layer``. Returns the kernel's layer handle. Raises if the activation limit is not known yet, if a slab
        would be misaddressed, or if the kernel refuses the registration.
        """
        if self.act_limit is None:
            raise ValueError(
                "the EXL3 CPU kernel needs the layers' activation limit before a layer registers"
            )
        hidden = int(slabs["w13_suh"].shape[-1])
        intermediate = int(slabs["w13_svh"].shape[-1])
        bits = int(slabs["w13_trellis"].shape[-1]) // 16
        trellis = hidden * intermediate * bits // 16  # int16 elements of one [k/16, n/16, 16 * bits] trellis
        row = {  # elements per slot row: w13 rows hold gate then up, w2 rows hold down
            "w13_trellis": 2 * trellis,
            "w13_suh": 2 * hidden,
            "w13_svh": 2 * intermediate,
            "w2_trellis": trellis,
            "w2_suh": intermediate,
            "w2_svh": hidden,
        }
        for name in self.slab_names:
            slab = slabs[name]
            dtype = torch.int16 if name.endswith("_trellis") else torch.float16
            if (
                slab.device.type != "cpu"
                or not slab.is_contiguous()
                or slab.dtype != dtype
                or slab.shape[0] < capacity
                or slab[0].numel() != row[name]
            ):
                raise ValueError(
                    f"EXL3 slab {name} {tuple(slab.shape)} {slab.dtype} is not {capacity} contiguous CPU rows of "
                    f"{row[name]} {dtype} elements"
                )
        import ctypes

        fn = self._native("sglang_exl3_cpu_experts_register_slabs")
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int64),
        ]
        bases = (ctypes.c_void_p * len(self.slab_names))(
            *(slabs[name].data_ptr() for name in self.slab_names)
        )
        handle = ctypes.c_int64(-1)
        status = fn(
            bases, capacity, hidden, intermediate, bits, int(self.swizzled), self.act_limit, ctypes.byref(handle)
        )
        if status != 0:
            raise RuntimeError(f"the EXL3 CPU kernel refused the layer's slabs: status {status}")
        self._slabs[handle.value] = [slabs[name] for name in self.slab_names]
        return handle.value

    def free_layer(self, handle) -> None:
        """Release the kernel's layer ``handle`` and the slabs it addressed."""
        self.ext.exl3_moe_cpu_free_layer(handle)
        self._slabs.pop(handle, None)
```

In the test, `FakeRegisterSlabs` ignores `restype`/`argtypes` assignments (plain attribute sets on a Python object). `_native`'s error message: replace "which a build flavor selects (SGLANG_EXL3_CPU_ACT_RESIDUAL=1 SGLANG_EXL3_CPU_ACT_BLOCK=128)" with "which SGLANG_DSV41_CPU_EXPERTS=1 builds (exl3_cpu/optimized)" — only the optimized kernel defines the slab ABI.

- [ ] **Step 4: The manual pool test covers the DSV4.1 shape**

In `test/manual/dsv41/test_cpu_expert_pool_exl3.py`:

- Make `_random_slabs(seed)` take `(seed, hidden=H, inter=INTER)` and use those in place of the module constants `H`/`INTER` inside it; `_direct_layer` is unchanged.
- Parametrize `test_pool_matches_direct_kernel_calls_bit_for_bit` over shapes, replacing its `H` uses with the parameter:

```python
@pytest.mark.parametrize("hidden, inter", [(H, INTER), (5120, 2304)], ids=["generic", "dsv41"])
def test_pool_matches_direct_kernel_calls_bit_for_bit(monkeypatch, hidden, inter):
```

  Inside: `slabs = _random_slabs(20260929, hidden, inter)`; `x = (torch.randn(1, hidden, generator=g) * scale).half()`; `want = torch.zeros(1, hidden)`; `got = torch.zeros(1, hidden)`; `changed = torch.zeros(1, hidden)`. Update the docstring: "the pool's slab registration runs bit-identically to make_layer's per-expert registration over the same slabs, at a generic shape and at DeepSeek V4.1's, where both take the DSV4.1 plan on AVX-512BW."
- `test_the_c_abi_forward_overwrites_or_accumulates` keeps `make_layer`; leave it.
- Its `build_flavor` skip: the trait now needs the optimized kernel; change the skip condition to `if not optimized_cpu(cpu_act_defines()):` with `from sglang.srt.layers.quantization.exl3_ext import cpu_act_defines, exl3_ext, optimized_cpu` and message `"the slab ABI is the optimized kernel's: set SGLANG_DSV41_CPU_EXPERTS=1"`, and add the same skip at the top of `test_pool_matches_direct_kernel_calls_bit_for_bit`.

- [ ] **Step 5: Commit, push, run the gate**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/exl3.py test/registered/unit/kernels/test_cpu_expert_pool.py \
  test/manual/dsv41/test_cpu_expert_pool_exl3.py
git commit -m "$(cat <<'EOF'
feat(cpu-experts): the EXL3 trait registers pinned slabs by base pointer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push
```

Gate with `OUT=…/task6` and `slabs`. Expected: `ALL GREEN (check)`; `pytest.log` shows the new `exl3_trait` tests and `test_pool_matches_direct_kernel_calls_bit_for_bit[generic]` and `[dsv41]` passing.

---

### Task 7: Latency A/B, README layout note, and the branch review

The refactor is bit-exact by construction, but templating changes inlining; this task measures that it did not cost time.

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3_cpu/optimized/README.txt`

- [ ] **Step 1: README code layout**

Append to `optimized/README.txt`, before "Provenance and validation":

```
Code layout
-----------
moe_mul1.cpp holds the kernels and the public API. A forward is ForwardPlan<Shape, Isa>::run (forward_plan.hpp),
picked once per call in forward_raw: ForwardPlan<Dsv41Shape, Isa::Bw> when Dsv41Shape::accepts the call on an
AVX-512BW host, else ForwardPlan<GenericShape, I> for the host's tier. PlanTraits<Dsv41Shape, Isa::Bw> is the one
specialization: compact scratch, grouped traversal, wide single-expert quantization. shapes.hpp fixes DeepSeek
V4.1's dimensions. Plans read experts through an accessor (experts.hpp): TableExperts over make_layer's per-expert
tables, or StridedExperts<Shape> over sglang_exl3_cpu_experts_register_slabs's slab bases.

Bit-exact checks for any change here: test/manual/dsv41/run_exl3_cpu_forward_checks.sh (A/B dumps per ISA tier
against the merge-base, the bare and full-stack benches' frozen references, the CPU expert pool tests).
```

- [ ] **Step 2: Build both bench binaries**

The base worktree's bench was built in `base-recheck/bench-build` by Task 1 Step 5; the branch's in `task6/bench-build`. Confirm both exist:

```bash
ssh divix01 'ls -l /data/models/slang/nvfp4-work/exl3-forward-plan/{base-recheck,task6}/bench-build/exl3_cpu_optimized'
```

- [ ] **Step 3: Write the A/B timing job (on divix01, not committed)**

`/data/models/slang/nvfp4-work/exl3-forward-plan/latency-ab.sh`:

```bash
#!/bin/bash
# Alternating process rounds of the bare-forward bench: merge-base vs branch, experts 1/3/5, 512 forwards each.
set -euo pipefail
root=/data/models/slang/nvfp4-work/exl3-forward-plan
results=${EXL3BENCH_RESULTS:-$root/latency-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$results"
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
export EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for round in 0 1 2 3 4 5 6 7; do
  order=(base-recheck task6)
  (( round % 2 )) && order=(task6 base-recheck)
  for side in "${order[@]}"; do
    "$root/$side/bench-build/exl3_cpu_optimized" --benchmark_min_time=512x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$side.json" --benchmark_out_format=json \
      > "$results/round-$round-$side.log" 2>&1
  done
done
echo "Results: $results"
```

- [ ] **Step 4: Run it on the isolated partition**

The isolation service runs any user-owned command file (`bench/service/README.txt`, "Any other job"). Only when no server runs:

```bash
ssh divix01 'pgrep -f sglang.launch_server && exit 2
  chmod +x /data/models/slang/nvfp4-work/exl3-forward-plan/latency-ab.sh
  printf "%s\n" /bin/bash /data/models/slang/nvfp4-work/exl3-forward-plan/latency-ab.sh \
    > /data/models/exl3_exp/google_benchmark/service-command.txt
  systemctl --no-ask-password start exl3bench.service'
```

Wait for it (`systemctl show exl3bench.service -p ActiveState` → `inactive`, ~5–10 minutes), then remove the command file so the service returns to its default job:

```bash
ssh divix01 'rm -f /data/models/exl3_exp/google_benchmark/service-command.txt; systemctl show exl3bench.service -p Result -p ExecMainStatus; journalctl -u exl3bench.service -n 5 --no-pager'
```

Expected: `Result=success`, `ExecMainStatus=0`; the journal names the results directory.

- [ ] **Step 5: Compare the medians of p50**

```bash
ssh divix01 '/data/models/slang/.venv/bin/python - <<"EOF"
import glob, json, statistics
d = sorted(glob.glob("/data/models/exl3_exp/google_benchmark/service-*"))[-1]
p50 = {}
for f in glob.glob(f"{d}/round-*.json"):
    side = "branch" if f.endswith("-task6.json") else "base"
    for b in json.load(open(f))["benchmarks"]:
        p50.setdefault(b["name"], {}).setdefault(side, []).append(b["p50_us"])
print("results:", d)
for name, sides in sorted(p50.items()):
    base, branch = statistics.median(sides["base"]), statistics.median(sides["branch"])
    print(f"{name:40s} base {base:9.2f} us  branch {branch:9.2f} us  {100 * (branch / base - 1):+6.2f}%  ({len(sides['base'])} rounds)")
EOF'
```

Expected: for each expert count, the branch's median p50 within ±2% of the base's. A larger regression is a finding: report the numbers and the expert counts; do not tune in this branch.

- [ ] **Step 6: Commit and push the README**

```bash
git add python/sglang/srt/layers/quantization/exl3_cpu/optimized/README.txt
git commit -m "$(cat <<'EOF'
docs(exl3-cpu): the optimized kernel's code layout and bit-exact checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
git push
```

- [ ] **Step 7: Clean up divix01**

Ask before deleting; when approved:

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-forward-plan-base \
  && rm -rf /data/models/slang/nvfp4-work/exl3-forward-plan/{task2,task3,task4,task5}'
```

Keep `base/` (the reference dumps), `task6/` and the latency results until the branch merges.
