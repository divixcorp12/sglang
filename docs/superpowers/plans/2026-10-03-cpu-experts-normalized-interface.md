# CPU Experts Normalized Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every CPU expert quant (today EXL3 and NVFP4) the same shape: one C ABI, one registration descriptor, one
`ExpertForward<Quant>` front end with the layer registry, one `MoeBufferRow<Quant>` view of a pinned-RAM expert slot,
runtime ISA dispatch to `ForwardPlan<Shape, Isa>` specializations, and the same file set per quant, so a new quant is
a new directory that fills in the same files.

**Architecture:** A header-only framework in `python/sglang/srt/layers/quantization/cpu_experts_common/` owns
everything that is not arithmetic: ISA detection, the worker team and its cores, keep-warm, the layer registry,
argument and route validation, and the C ABI wrappers. Each quant supplies a `Quant` type (its slabs, registration
parameters, layer type and plan dispatch), its math per ISA tier, its shapes and its forward plan. Each quant stays
one translation unit, so no build gains a source file: torch's `load()` for EXL3, `build.py` for NVFP4 and both bench
CMake files keep compiling one `.cpp`.

**Tech Stack:** C++20 (GCC 15 on divix01), OpenMP, x86 intrinsics with per-function `target` attributes, ctypes, pytest.

**Spec:** this plan carries its own spec, the user's request of 2026-10-03 and the decisions in **Decisions** below.

## Decisions

The user asked for six things; four questions were settled before planning.

1. Normalized format for all CPU experts: a standard forward call, forward plan and file layout. Not a common quant format.
2. `template <typename MoeQuant> struct MoeBufferRow`: the wrapper for one expert slot in the pinned-RAM buffer.
3. A central `ExpertForward` with a static member for the layer array, filled by the registration the CPU experts
   service performs. **Deviation, on purpose:** the user sketched `template <typename ExpertPlan> struct ExpertForward`.
   One registered layer runs under several plans (one per ISA tier and shape), and a static member of a class
   template is one array per instantiation, so `ExpertForward<ExpertPlan>` would give each plan its own layer array and
   a handle registered once would be missing from the other plans. The registry is therefore keyed by quant:
   `ExpertForward<Quant>`, whose `run()` picks the `ForwardPlan<Shape, Isa>`.
4. A standard file set per quant (math, forward plan, shapes, ...).
5. The `extern "C"` functions are one-line wrappers.
6. NVFP4 gets per-ISA specializations like EXL3.

Answers:

- ISA dispatch: **runtime, like EXL3.** One binary per quant; tiers are template instantiations under `target`
  attributes; the tier is detected once at load. NVFP4 gets Scalar and Avx2 now (its existing loops); no new AVX-512
  NVFP4 arithmetic in this plan.
- Registration: **one common descriptor** (`SglangCpuExpertsLayer`) plus a small quant-specific parameter struct. Every
  quant exports the same five C functions: `register_layer`, `free_layer`, `forward`, `keep_warm`, `set_cores`.
- EXL3 torch paths (`exl3_moe_cpu_make_layer`, `exl3_moe_cpu_forward`, `exl3_moe_cpu_forward_raw`, `has_avx*`):
  **kept, as thin wrappers** over the same `ExpertForward<Exl3Quant>`. The legacy `exl3_cpu/moe_mul1.cpp` is untouched
  except for what its C ABI must keep compiling against (it is the bench's frozen baseline).
- Branch: **stay on `numa-node-distributor`** in the main worktree. Another session commits there too: stage by name,
  commit with `git commit --only <paths>`, and re-check `git status --short --branch` before every commit.

## Global Constraints

- Arithmetic is frozen. Every task ends bit-exact against the pre-refactor base commit (recorded in Task 0) for both
  quants, on every tier each quant has: NVFP4 Scalar and Avx2; EXL3 Scalar, Avx2, Bw, Vnni, Vbmi.
- NVFP4 builds with `-ffp-contract=off` and never `-Ofast`. EXL3 builds as today (`-Ofast -march=native`, GCC 15).
- Each quant remains one translation unit; `cpu_experts_common/` is header-only.
- Code runs on divix01 only from a pushed commit, in a private worktree
  (`.claude/rules/divix01-run-protocol.md`). Pushing needs the user's go-ahead at each checkpoint (Tasks 4, 7, 9).
- Read `PIPESTATUS[0]` for any suite run through a pipe; record the exact command next to every result.
- C ABI status codes, for every quant: 0 success, 1 internal error, 2 invalid arguments, 3 concurrent use.
- Comments follow `.claude/rules/comment-style.md`; no new `.md` files beyond this plan.

## Review Focus

1. **A handle registered through one quant's library used with another's.** Each library has its own
   `ExpertForward<Quant>::layers`, so the lookup must return 2, never read the wrong layout. Test: Task 2's
   `unknown_handle_is_refused`.
2. **EXL3 inputs that were silently accepted and are now refused.** The common validation refuses slots outside
   `[-1, capacity)` and nonfinite weights with 2. EXL3 skipped an out-of-range expert silently. The pool and service
   only pass valid slots, so this is a tightening, not a break. Test: Task 6 adds `slot_past_capacity_is_refused` to
   the EXL3 C ABI test.
3. **A concurrent EXL3 forward now returns 3.** The torch path (`exl3_moe_cpu_forward`) and the engine's C ABI share
   `ExpertForward<Exl3Quant>`'s try-lock. A second concurrent caller used to race the per-thread arena; now the C ABI
   returns 3 and the torch wrapper raises. Test: Task 2's `concurrent_forward_returns_3` (framework level).
4. **`set_cores` after the first forward.** NVFP4 returned 2 and EXL3 returned 1. The common answer is 2.
   `exl3.py`'s `native_set_cores` message and any EXL3 test asserting 1 change with it. Test: Task 6 asserts 2.
5. **The ISA cap environment variables still work.** `EXL3_MOE_CPU_MAX_ISA` keeps its name and values. NVFP4 gains
   `NVFP4_CPU_MAX_ISA` (scalar|avx2), which replaces the `--portable` build as the way to run the scalar tier. Test:
   Task 5's `test_the_scalar_cap_runs_the_scalar_tier`.

---

## File Structure

### Common framework (new, header-only)

`python/sglang/srt/layers/quantization/cpu_experts_common/`

| File | Responsibility |
| --- | --- |
| `isa.hpp` | `enum class Isa`, `detect_isa(top, cap_env)`, the `TARGET_*` attribute macros |
| `team.hpp` | Worker cores (`Cores`): configure, freeze, pin; `run_team(threads, body)` |
| `keep_warm.hpp` | Register-only keep-warm loops per tier and the team that runs them |
| `buffer_row.hpp` | `MoeBufferRow<Quant>` and `MoeBufferRows<Quant>` (the strided slab view of a layer) |
| `routes.hpp` | `Route`, `RouteTable`: a call's live routes per token, validated |
| `expert_forward.hpp` | `ExpertForward<Quant>`: the layer registry, argument validation, `register_layer`, `free_layer`, `run` |
| `cabi.hpp` | `SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant)`: the five `extern "C"` one-liners |

The C descriptor lives beside the engine's forward struct, in
`python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_expert_forward_abi.h` (renamed in Task 1 to
`cpu_experts_abi.h`, since it now holds the registration descriptor too).

### Per quant (the standard file set)

`python/sglang/srt/layers/quantization/<quant>_cpu/optimized/`

| File | Responsibility |
| --- | --- |
| `cpu_experts_cabi.h` | The quant's parameter struct and the five C declarations (`sglang_<quant>_cpu_experts_*`) |
| `quant.hpp` | `struct <Quant>Quant`: slab names, minimum row bytes, parameter validation, `Layer`, `Projection`, `dispatch` |
| `math.hpp` | ISA-independent arithmetic: activation quantization, activations |
| `math_scalar.hpp`, `math_avx2.hpp`, `math_avx512.hpp` | The dot product or GEMV kernels of each tier (`math_avx512.hpp` only where the quant has that tier) |
| `shapes.hpp` | `GenericShape` and the fixed model shapes |
| `forward_plan.hpp` | `ForwardPlan<Shape, Isa>`, its `PlanTraits`, `ForwardArena` |
| `kernel.cpp` | The translation unit: includes, `extern "C"` via `cabi.hpp`, quant-only extras (EXL3 torch wrappers) |

EXL3 keeps `moe_mul1.h` (its public ATen API) and folds `register.hpp` and `traversal.hpp` into `math_avx512.hpp`.
NVFP4's `moe_mul1.h`, `experts.hpp` and `dot_nvfp4.h` are absorbed into `quant.hpp`, `math*.hpp` and
`buffer_row.hpp`. `moe_mul1.cpp` becomes `kernel.cpp` in Task 9, the last task, so earlier diffs stay reviewable.

---

### Task 0: Record the baseline

**Files:** none (records only).

- [ ] **Step 1: Pin the base commit.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4 status --short --branch
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4 rev-parse --short HEAD   # call it BASE
```

- [ ] **Step 2: Produce the baseline dumps on divix01** (BASE is already pushed).

```bash
ssh divix01 'R=/data/models/slang/sglang; W=/data/models/slang/nvfp4-work; D=$W/normalize-base
git -C $R fetch -q origin && git -C $R worktree add -q --detach $W/wt-normalize-base BASE
bash $W/wt-normalize-base/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh $W/wt-normalize-base $D/nvfp4
bash $W/wt-normalize-base/test/manual/dsv41/run_exl3_cpu_forward_checks.sh baseline $W/wt-normalize-base $D/exl3'
```

Expected: `DUMPED native`, `DUMPED portable` and the EXL3 baseline's `EXIT=0` lines. Keep `$D` for the whole plan.

---

### Task 1: The common registration descriptor

**Files:**
- Rename: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_expert_forward_abi.h` to `cpu_experts_abi.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h` (the include)
- Modify: `python/sglang/srt/layers/quantization/{exl3_cpu,nvfp4_cpu}/optimized/cpu_experts_cabi.h` (the include path)
- Modify: `python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py` (`_FORWARD_ABI` path)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/pool.py` (ctypes mirror)
- Test: `test/registered/unit/kernels/test_cpu_experts_abi.py` (new)

**Interfaces:**
- Produces: `SglangCpuExpertsLayer`, `SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION`, `SGLANG_CPU_EXPERTS_MAX_SLABS` (C);
  `CpuExpertsLayer`, `CPU_EXPERTS_LAYER_ABI_VERSION` (Python, `pool.py`).

- [ ] **Step 1: Write the failing layout test.** It compiles a probe printing `offsetof` for every field and compares
  it with the ctypes mirror. Field drift between C and ctypes is the bug this pins.

```python
"""SglangCpuExpertsLayer and SglangCpuExpertsForward match their ctypes mirrors in pool.py, field by field."""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
ABI = REPO / "python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"
CXX = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")

PROBE = r"""
#include <cstddef>
#include <cstdio>
#include "cpu_experts_abi.h"
#define F(T, f) std::printf(#T " " #f " %zu\n", offsetof(T, f));
int main() {
    F(SglangCpuExpertsLayer, abi_version) F(SglangCpuExpertsLayer, capacity) F(SglangCpuExpertsLayer, hidden)
    F(SglangCpuExpertsLayer, intermediate) F(SglangCpuExpertsLayer, activation) F(SglangCpuExpertsLayer, act_limit)
    F(SglangCpuExpertsLayer, slab_count) F(SglangCpuExpertsLayer, slabs) F(SglangCpuExpertsLayer, slot_bytes)
    F(SglangCpuExpertsLayer, params)
    F(SglangCpuExpertsForward, abi_version) F(SglangCpuExpertsForward, rows) F(SglangCpuExpertsForward, layer)
    F(SglangCpuExpertsForward, x) F(SglangCpuExpertsForward, slots) F(SglangCpuExpertsForward, weights)
    F(SglangCpuExpertsForward, out) F(SglangCpuExpertsForward, k) F(SglangCpuExpertsForward, threads)
    F(SglangCpuExpertsForward, accumulate)
    std::printf("SglangCpuExpertsLayer sizeof %zu\nSglangCpuExpertsForward sizeof %zu\n",
                sizeof(SglangCpuExpertsLayer), sizeof(SglangCpuExpertsForward));
}
"""


@pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")
def test_the_ctypes_mirrors_match_the_c_layout(tmp_path):
    from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertsForwardCall, CpuExpertsLayer

    (tmp_path / "probe.cpp").write_text(PROBE)
    exe = tmp_path / "probe"
    subprocess.run([CXX, "-I", str(ABI.parent), str(tmp_path / "probe.cpp"), "-o", str(exe)], check=True)
    c = {}
    for line in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines():
        struct, field, value = line.split()
        c[(struct, field)] = int(value)
    for struct, mirror in (("SglangCpuExpertsLayer", CpuExpertsLayer), ("SglangCpuExpertsForward", CpuExpertsForwardCall)):
        for name, _ in mirror._fields_:
            assert c[(struct, name)] == getattr(mirror, name).offset, (struct, name)
        assert c[(struct, "sizeof")] == ctypes.sizeof(mirror), struct
```

- [ ] **Step 2: Run it; expect failure** (`ImportError: cannot import name 'CpuExpertsLayer'`).

```bash
PYTHONPATH=$PWD/python python -m pytest -q test/registered/unit/kernels/test_cpu_experts_abi.py; echo "EXIT=${PIPESTATUS[0]}"
```

- [ ] **Step 3: Rename the header and add the descriptor.**

```bash
git mv python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_expert_forward_abi.h \
       python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h
```

Append to `cpu_experts_abi.h`:

```c
#define SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION 1u
#define SGLANG_CPU_EXPERTS_MAX_SLABS 8

// One layer's pinned host tier, for every quant's register_layer: slot s of slab i starts at slabs[i] + s *
// slot_bytes[i]. slab_count is the quant's own count (a slab the quant marks optional may be null); params points at
// the quant's parameter struct (cpu_experts_cabi.h), or is null when the quant has none. activation 0 is gated SiLU
// with the pre-SiLU clamp act_limit (0: none); a quant refuses (2) an activation it does not implement. The kernel
// stores the pointers only: the registrant keeps every slab and params alive until free_layer.
typedef struct SglangCpuExpertsLayer {
    uint32_t abi_version;
    int32_t capacity;
    int32_t hidden;
    int32_t intermediate;
    int32_t activation;
    float act_limit;
    int32_t slab_count;
    const void* slabs[SGLANG_CPU_EXPERTS_MAX_SLABS];
    uint64_t slot_bytes[SGLANG_CPU_EXPERTS_MAX_SLABS];
    const void* params;
} SglangCpuExpertsLayer;
```

Update the four includes (`cpu_experts.h`, both `cpu_experts_cabi.h`, `nvfp4_cpu_ext.py`'s `_FORWARD_ABI`) to the new name.

- [ ] **Step 4: Add the ctypes mirror** in `pool.py`, beside `CpuExpertsForwardCall`:

```python
CPU_EXPERTS_LAYER_ABI_VERSION = 1
CPU_EXPERTS_MAX_SLABS = 8


class CpuExpertsLayer(ctypes.Structure):
    """``SglangCpuExpertsLayer`` (``expert_stream/host/cpu_experts_abi.h``); the field order is the C struct's."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("capacity", ctypes.c_int32),
        ("hidden", ctypes.c_int32),
        ("intermediate", ctypes.c_int32),
        ("activation", ctypes.c_int32),
        ("act_limit", ctypes.c_float),
        ("slab_count", ctypes.c_int32),
        ("slabs", ctypes.c_void_p * CPU_EXPERTS_MAX_SLABS),
        ("slot_bytes", ctypes.c_uint64 * CPU_EXPERTS_MAX_SLABS),
        ("params", ctypes.c_void_p),
    ]
```

- [ ] **Step 5: Run the test; expect PASS.** Also grep that nothing names the old header:
  `git grep -n cpu_expert_forward_abi -- ':!docs'` prints nothing.

- [ ] **Step 6: Commit.**

```bash
git add test/registered/unit/kernels/test_cpu_experts_abi.py python/sglang/srt/layers/moe/cpu_experts/pool.py \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/cpu_experts_cabi.h \
  python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/cpu_experts_cabi.h \
  python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py
H=python/sglang/kernels/jit/csrc/moe/expert_stream/host
git commit --only $H/cpu_expert_forward_abi.h $H/cpu_experts_abi.h $H/cpu_experts.h \
  test/registered/unit/kernels/test_cpu_experts_abi.py python/sglang/srt/layers/moe/cpu_experts/pool.py \
  python/sglang/srt/layers/quantization/exl3_cpu/optimized/cpu_experts_cabi.h \
  python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/cpu_experts_cabi.h \
  python/sglang/srt/layers/quantization/nvfp4_cpu_ext.py \
  -m "feat(cpu-experts): SglangCpuExpertsLayer, the common registration descriptor"
```

(`--only` with explicit paths, both sides of the rename included, keeps the other session's staged files out.)

---

### Task 2: The common framework

**Files:**
- Create: `python/sglang/srt/layers/quantization/cpu_experts_common/{isa,team,keep_warm,buffer_row,routes,expert_forward,cabi}.hpp`
- Test: `test/registered/unit/kernels/cpu_experts_common_check.cpp` (native harness, new)
- Test: `test/registered/unit/kernels/test_cpu_experts_common.py` (builds and runs it, new)

**Interfaces:**
- Consumes: `SglangCpuExpertsLayer`, `SglangCpuExpertsForward` (Task 1).
- Produces, all in `namespace sglang::cpu_experts`:
  - `enum class Isa { Scalar, Avx2, Bw, Vnni, Vbmi }`; `Isa detect_isa(Isa top, const char* cap_env)`.
  - `struct Cores` with `static int configure(const int32_t* cores, int32_t n)`, `static void freeze()`,
    `static const std::vector<int>& frozen()`, `static void pin(int worker, std::atomic<int>& error)`.
  - `template <class Body> void run_team(int threads, Body&& body)`: `body(worker, workers)` on a pinned team; throws
    `std::runtime_error` when the team is short or a pin fails.
  - `int keep_warm(Isa isa, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns)`.
  - `template <class Quant> struct MoeBufferRow` and `template <class Quant> struct MoeBufferRows`.
  - `struct Route { int32_t slot; float weight; }`; `struct RouteTable` (per token: `count[t]`,
    `route(t, i)`, `rows`, `k`).
  - `template <class Quant> struct ExpertForward` with `static int register_layer(const SglangCpuExpertsLayer*,
    int64_t*)`, `static int free_layer(int64_t)`, `static int forward(const SglangCpuExpertsForward*)`,
    `static std::shared_ptr<const typename Quant::Layer> lookup(int64_t)`, and the registry
    `static inline std::vector<std::shared_ptr<const typename Quant::Layer>> layers`.
  - `SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant)`.

**The Quant contract** (what every quant type must provide; Tasks 3 and 6 implement it):

```cpp
struct SomeQuant {
    static constexpr const char* kName;        // "nvfp4", "exl3": error messages only
    static constexpr int kSlabs;               // slabs the quant reads (<= SGLANG_CPU_EXPERTS_MAX_SLABS)
    static constexpr uint32_t kOptionalSlabs;  // bit i: slab i may be null
    static constexpr int kMaxRoutes;           // per-token k limit of the forward
    static constexpr int kMaxRows;             // rows limit of the forward
    static constexpr Isa kTopIsa;              // the highest tier the quant implements
    static constexpr const char* kIsaCapEnv;   // "NVFP4_CPU_MAX_ISA", "EXL3_MOE_CPU_MAX_ISA"
    using Params;                              // the quant's C parameter struct (may be an empty struct)
    using Layer;                               // what register_layer stores: holds an MoeBufferRows<SomeQuant>
    using Row;                                 // what MoeBufferRow<SomeQuant> decodes a slot into (gate/up/down)
    // Minimum slot bytes per slab for these dimensions and params; register_layer refuses a smaller stride.
    static std::array<uint64_t, kSlabs> min_slot_bytes(const SglangCpuExpertsLayer&, const Params&);
    // 0, or 2 when the descriptor's scalars or params are outside what the quant implements.
    static int validate(const SglangCpuExpertsLayer&, const Params*);
    static Layer make_layer(const SglangCpuExpertsLayer&, const Params*);
    // 0, or 2 when slot's row cannot run (NVFP4: a nonfinite alpha). Read on every call: slots are reused.
    static int check_slot(const Layer&, int slot);
    // Runs one call after the framework validated it: picks ForwardPlan<Shape, Isa> and returns its status.
    static int dispatch(const Layer&, const SglangCpuExpertsForward&, const RouteTable&, Isa);
    // The row decoded from slot s's bytes; MoeBufferRow<SomeQuant>::row() calls it.
    static Row decode(const uint8_t* const* slot_bases, const Layer&);
};
```

- [ ] **Step 1: Write the failing harness.** A toy quant (`ToyQuant`: one slab of `float[hidden]` per slot; the plan
  writes `out[t][h] (+)= sum over routes of weight * slab[slot][h]`) drives every framework path:

```cpp
// Standalone harness for cpu_experts_common: a toy quant through ExpertForward. Built by test_cpu_experts_common.py.
#include "../../../python/sglang/srt/layers/quantization/cpu_experts_common/expert_forward.hpp"
#include "../../../python/sglang/srt/layers/quantization/cpu_experts_common/cabi.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace toy {
using namespace sglang::cpu_experts;
struct ToyParams { float scale; };
struct ToyQuant {
    static constexpr const char* kName = "toy";
    static constexpr int kSlabs = 1;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 8;
    static constexpr int kMaxRows = 64;
    static constexpr Isa kTopIsa = Isa::Avx2;
    static constexpr const char* kIsaCapEnv = "TOY_CPU_MAX_ISA";
    using Params = ToyParams;
    struct Row { const float* v; };
    struct Layer { int hidden; float scale; MoeBufferRows<ToyQuant> rows; };
    static std::array<uint64_t, 1> min_slot_bytes(const SglangCpuExpertsLayer& d, const Params&)
    { return {uint64_t(d.hidden) * 4}; }
    static int validate(const SglangCpuExpertsLayer& d, const Params* p)
    { return d.activation == 0 && p && std::isfinite(p->scale) ? 0 : 2; }
    static Layer make_layer(const SglangCpuExpertsLayer& d, const Params* p)
    { return {d.hidden, p->scale, MoeBufferRows<ToyQuant>::of(d)}; }
    static int check_slot(const Layer&, int) { return 0; }
    static Row decode(const uint8_t* const* base, const Layer&) { return {reinterpret_cast<const float*>(base[0])}; }
    static int dispatch(const Layer& l, const SglangCpuExpertsForward& c, const RouteTable& r, Isa isa)
    {
        last_isa = isa;
        run_team(c.threads, [&](int worker, int workers) {
            for (int h = worker; h < l.hidden; h += workers)
                for (int t = 0; t < c.rows; ++t) {
                    float s = 0;
                    for (int i = 0; i < r.count[t]; ++i)
                        s += r.route(t, i).weight * l.scale
                             * decode(l.rows.slot(r.route(t, i).slot).base.data(), l).v[h];
                    float& o = c.out[size_t(t) * l.hidden + h];
                    o = c.accumulate ? o + s : s;
                }
        });
        return 0;
    }
    static inline Isa last_isa = Isa::Scalar;
};
}  // namespace toy

SGLANG_CPU_EXPERTS_DEFINE_CABI(toy, toy::ToyQuant)

// ... main(): one block per check below, each an assert; prints "ok <name>" per check.
```

`main()` checks, each named so a failure points at its contract:
  - `register_then_forward_overwrites_and_accumulates`: capacity 3, hidden 16, slab values `slot + h`, rows 2,
    k 2 with one `-1` slot; exact float results for both `accumulate` values.
  - `abi_versions_are_checked`: a layer or call with `abi_version + 1` returns 2.
  - `slot_bytes_below_the_minimum_are_refused`: `slot_bytes[0] = hidden * 4 - 4` returns 2.
  - `unknown_handle_is_refused` (Review Focus 1): forward with handle 99 returns 2; after `free_layer(h)`, h returns 2.
  - `routes_are_validated`: slot `capacity`, slot `-2`, weight NaN, `k = kMaxRoutes + 1`, `rows = kMaxRows + 1`,
    `rows = 0` each return 2 with `out` untouched.
  - `concurrent_forward_returns_3` (Review Focus 3): a second thread calls forward while the first is inside
    `dispatch` (a `std::atomic<bool>` gate in `ToyQuant::dispatch`, set by a test-only flag); it gets 3.
  - `isa_cap_env_lowers_the_tier`: with `TOY_CPU_MAX_ISA=scalar` set before the first call (`setenv` in `main`),
    `last_isa == Isa::Scalar`; detection never exceeds `kTopIsa`.
  - `set_cores_after_the_first_forward_returns_2` (Review Focus 4).
  - `keep_warm_returns_when_the_word_moves`: `word` changed by another thread after 1 ms; returns 0 within 1 s.

- [ ] **Step 2: Write `test_cpu_experts_common.py`** (Linux + GCC with OpenMP, like `test_nvfp4_cpu_build.py`): it
  compiles `cpu_experts_common_check.cpp` with `-std=c++20 -O2 -fopenmp -pthread` (and once more under
  `-fsanitize=address,undefined` when `_links_sanitizers` holds), runs it, asserts returncode 0 and that every
  check name above appears as `ok <name>` in stdout.

- [ ] **Step 3: Run; expect a compile failure** (`expert_forward.hpp: No such file`).

- [ ] **Step 4: Implement the headers.** The moves come from code that exists today. Copy each body and change only
  what the table says.

| Header | Built from | Change |
| --- | --- | --- |
| `isa.hpp` | EXL3 `moe_mul1.cpp` `enum class Isa`, `detect_isa`, `M1_TARGET_*` macros | `detect_isa(Isa top, const char* cap_env)`: result is `min(hw, top, cap)`; macros renamed `SGLANG_TARGET_AVX2/BW/VNNI/VBMI` |
| `team.hpp` | NVFP4 `moe_mul1.cpp` worker-core block (`g_configured_cores`, `freeze_compute_cores`, `pin_compute_worker`, `set_cores` body) and NVFP4 `ForwardPlan::run_team` | globals become `Cores`' `static inline` members; `configure` returns 2 after freeze; `run_team` keeps the checks (team size, pin error) and throws `std::runtime_error` |
| `keep_warm.hpp` | EXL3 `keep_warm_done`, `keep_warm_bw/avx2/scalar`, the `sglang_exl3_cpu_experts_keep_warm` body | tier chosen by the `Isa` argument; Vnni/Vbmi use the Bw loop |
| `buffer_row.hpp` | NVFP4 `StridedExperts` (bases and strides) | see below |
| `routes.hpp` | NVFP4 `sglang_nvfp4_cpu_experts_forward` route loop | keeps every route, -1 slots and zero weights dropped, routing order kept |
| `expert_forward.hpp` | NVFP4 registry (`layers`, `lookup`, `forward_mutex`, `register_slabs`, `free_layer`) and the forward's validation | generic over `Quant`; handles are indices into `layers`, freed entries reset (EXL3's scheme), never reused |

`buffer_row.hpp` in full:

```cpp
// One expert slot of a layer's pinned host tier, and the layer's slots. Slot s of slab i is at base[i] + s *
// stride[i]; the quant decodes a slot's bytes into its Row. Views only: nothing is copied or owned.
#pragma once
#include "cpu_experts_abi.h"
#include <array>
#include <cstdint>

namespace sglang::cpu_experts {

template <class Quant>
struct MoeBufferRow
{
    std::array<const uint8_t*, Quant::kSlabs> base;  // this slot's first byte in each slab (null: optional, absent)

    const uint8_t* slab(int i) const { return base[i]; }
};

template <class Quant>
struct MoeBufferRows
{
    std::array<const uint8_t*, Quant::kSlabs> base;
    std::array<uint64_t, Quant::kSlabs> stride;

    static MoeBufferRows of(const SglangCpuExpertsLayer& d)
    {
        MoeBufferRows r{};
        for (int i = 0; i < Quant::kSlabs; ++i) {
            r.base[i] = static_cast<const uint8_t*>(d.slabs[i]);
            r.stride[i] = d.slot_bytes[i];
        }
        return r;
    }

    MoeBufferRow<Quant> slot(int s) const
    {
        MoeBufferRow<Quant> row;
        for (int i = 0; i < Quant::kSlabs; ++i) row.base[i] = base[i] ? base[i] + size_t(s) * stride[i] : nullptr;
        return row;
    }
};

}  // namespace sglang::cpu_experts
```

A plan reads slot `s` as `Quant::decode(layer.rows.slot(s).base.data(), layer)`: `MoeBufferRows` only addresses
bytes, and the quant alone knows what they mean.

`cabi.hpp` in full:

```cpp
// The five C functions every CPU expert quant exports, as one-line wrappers over ExpertForward<Quant>.
#pragma once
#include "expert_forward.hpp"
#include "keep_warm.hpp"
#include "team.hpp"

#define SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant)                                                             \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_register_layer(          \
        const SglangCpuExpertsLayer* d, int64_t* handle) noexcept                                               \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::register_layer(d, handle); }                          \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_free_layer(              \
        int64_t handle) noexcept                                                                                \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::free_layer(handle); }                                 \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_forward(                 \
        const SglangCpuExpertsForward* call) noexcept                                                           \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::forward(call); }                                      \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_keep_warm(               \
        int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept                     \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::keep_warm(threads, word, seen, deadline_ns); }         \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_set_cores(               \
        const int32_t* cores, int32_t n) noexcept                                                               \
    { return ::sglang::cpu_experts::Cores::configure(cores, n); }
```

`ExpertForward<Quant>::forward` order (it is NVFP4's today, generalized): null and `abi_version` check; `rows` in
`[1, kMaxRows]`, `k` in `[0, kMaxRoutes]`, `threads` in `[1, 4096]`, `accumulate` in {0, 1}, non-null `x`/`out` and
(when `k`) `slots`/`weights` (else 2); try-lock (else 3); `lookup` (else 2); every slot in `[-1, capacity)`, every
weight finite, `Quant::check_slot` for every live slot (else 2); build the `RouteTable` in a `thread_local` arena;
`Quant::dispatch(layer, call, routes, isa())`; any exception returns 1. `isa()` is
`static const Isa isa = detect_isa(Quant::kTopIsa, Quant::kIsaCapEnv)`.

- [ ] **Step 5: Run the harness test; expect PASS** (both the plain and the ASan/UBSan build).

```bash
PYTHONPATH=$PWD/python python -m pytest -q test/registered/unit/kernels/test_cpu_experts_common.py; echo "EXIT=${PIPESTATUS[0]}"
```

Locally this runs in the amd64 container (macOS cannot build OpenMP/Linux code): mount the repo and run the same
command with the image's `python3`.

- [ ] **Step 6: Commit** the seven headers and two test files by name (`git commit --only ...`), message
  `feat(cpu-experts): header-only framework: ISA, team, keep-warm, buffer rows, routes, ExpertForward, C ABI macro`.

---

### Task 3: NVFP4 onto ExpertForward (same tier selection as today)

**Files:**
- Create: `nvfp4_cpu/optimized/quant.hpp` (from `moe_mul1.h`'s `RegisteredLayer`, `LayerInfo`, `SlabRowBytes`,
  `Projection`, `w13_rows`; and `experts.hpp`)
- Modify: `nvfp4_cpu/optimized/moe_mul1.cpp` (registry, cores and C ABI bodies deleted; `SGLANG_CPU_EXPERTS_DEFINE_CABI(nvfp4, Nvfp4Quant)`)
- Modify: `nvfp4_cpu/optimized/forward_plan.hpp` (takes `const RouteTable&` instead of filling routes itself; uses
  `run_team` from `team.hpp`)
- Modify: `nvfp4_cpu/optimized/cpu_experts_cabi.h` (`SglangNvfp4CpuParams`, the five declarations)
- Delete: `nvfp4_cpu/optimized/experts.hpp`, `nvfp4_cpu/optimized/moe_mul1.h`
- Modify callers: `test/registered/unit/kernels/{nvfp4_cpu_sanitizer.cpp,test_nvfp4_cpu_build.py,test_nvfp4_cpu_experts.py}`,
  `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp`, `nvfp4_cpu/bench/src/cpu_forward.cpp`, `nvfp4_cpu/optimized/README.md`

**Interfaces:**
- Consumes: everything Task 2 produces.
- Produces: `struct Nvfp4Quant` (the Quant contract); `typedef struct SglangNvfp4CpuParams { int32_t w13_layout;
  float inv_input_scale13, inv_input_scale2; } SglangNvfp4CpuParams;`; C names `sglang_nvfp4_cpu_experts_register_layer`
  (replaces `_register_slabs`), `_free_layer`, `_forward`, `_keep_warm` (new), `_set_cores`.

The old `SglangNvfp4CpuLayer` maps onto the new pair like this: `capacity`, `hidden`, `intermediate`, `activation`,
`act_limit`, `slabs[0..6]`, `slot_bytes[0..6]` move to `SglangCpuExpertsLayer` with `slab_count = 7` and
`kOptionalSlabs = 1u << 6` (the up alpha); `w13_layout` and the two inverse scales move to `SglangNvfp4CpuParams`.
Registration validation is `valid()` from today's `moe_mul1.cpp`, split into `Nvfp4Quant::validate` (scalars,
params) and the framework's stride check against `min_slot_bytes` (today's `SlabRowBytes::of`).

- [ ] **Step 1: Port the tests first** to the new registration (they fail to link until Step 3): the sanitizer
  harness, the A/B harness, `test_nvfp4_cpu_experts.py`'s child (ctypes `CpuExpertsLayer` + a
  `SglangNvfp4CpuParams` ctypes struct defined in the child), the bench's `Layer` registration, and
  `test_nvfp4_cpu_build.py`'s `C_ABI` tuple (now the five names). The A/B harness keeps its dump format and case
  order, so its dumps stay byte-comparable with Task 0's.

- [ ] **Step 2: Confirm they fail** in the container: `nvfp4_cpu_sanitizer` fails to link
  (`undefined reference to sglang_nvfp4_cpu_experts_register_layer`).

- [ ] **Step 3: Implement `Nvfp4Quant`** in `quant.hpp` and route the C ABI through the macro. `kTopIsa` is
  `kBuildIsa` here (Avx2 when compiled with AVX2, else Scalar), so tier selection is unchanged until Task 5.
  `dispatch` keeps `run_plan`'s rule: `ForwardPlan<MimoV26ProShape, Isa::Avx2>` when `MimoV26ProShape::accepts`,
  else `ForwardPlan<GenericShape, kBuildIsa>`. `check_slot` is today's alpha-finiteness check. `keep_warm` uses
  `keep_warm.hpp`.

- [ ] **Step 4: Gate.** In the container: sanitizer and ggml_check pass; A/B native and portable dumps
  byte-identical to Task 0's (copy Task 0's `nvfp4/*.bin` down, or regenerate them from BASE in the container's
  base worktree as before). Expected: `PASS native bit-exact vs HEAD`, `PASS portable bit-exact vs HEAD`.

- [ ] **Step 5: Commit** by name; message `refactor(nvfp4-cpu): register_layer/forward/keep_warm through ExpertForward<Nvfp4Quant>`.

---

### Task 4: Checkpoint: NVFP4 on divix01

- [ ] **Step 1: Ask the user to push** (AskUserQuestion: push now / later). On yes:
  `git push origin numa-node-distributor`.
- [ ] **Step 2: Run the NVFP4 gate** against Task 0's dumps, in a private worktree at the pushed commit:

```bash
ssh divix01 'R=/data/models/slang/sglang; W=/data/models/slang/nvfp4-work; D=$W/normalize-base
git -C $R fetch -q origin && git -C $R worktree add -q --detach $W/wt-normalize origin/numa-node-distributor
bash $W/wt-normalize/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh $W/wt-normalize $W/normalize-t4 $D/nvfp4
cd $W/wt-normalize && PYTHONPATH=$PWD/python CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ OMP_NUM_THREADS=8 \
  taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
  test/registered/unit/kernels/test_nvfp4_cpu_build.py test/registered/unit/kernels/test_nvfp4_cpu_experts.py \
  test/registered/unit/kernels/test_cpu_experts_common.py test/registered/unit/kernels/test_cpu_experts_abi.py
echo "PYTEST_EXIT=$?"'
```

Expected: `PASS native bit-exact`, `PASS portable bit-exact`, `PYTEST_EXIT=0`. Record both lines with the command.

---

### Task 5: NVFP4 runtime ISA tiers and the standard math files

**Files:**
- Create: `nvfp4_cpu/optimized/math.hpp` (`q8_representable`, `quantize_block`, `swiglu`, `GpuRow`)
- Create: `nvfp4_cpu/optimized/math_scalar.hpp` (`dot_rows<Isa::Scalar, M>`: today's scalar tail loop)
- Create: `nvfp4_cpu/optimized/math_avx2.hpp` (`dot_rows<Isa::Avx2, M>`: today's `__AVX2__` branch under
  `SGLANG_TARGET_AVX2`)
- Delete: `nvfp4_cpu/optimized/dot_nvfp4.h` (its `__AVX__`-only branch is dropped: no tier selects it, and no
  supported host is AVX-without-AVX2)
- Modify: `quant.hpp` (`kTopIsa = Isa::Avx2`, `kIsaCapEnv = "NVFP4_CPU_MAX_ISA"`; `dispatch` switches on the
  detected tier), `forward_plan.hpp` (calls `dot_rows<I, M>`), `optimized/build.py` (drops `-march=native`;
  `--portable` is removed), `test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh` (variants: `avx2` and `scalar`, both
  from one build, the second with `NVFP4_CPU_MAX_ISA=scalar`), `test_nvfp4_cpu_build.py`,
  `test_nvfp4_cpu_experts.py`, `nvfp4_cpu_ggml_check.cpp` (calls both tiers), `optimized/README.md`,
  `nvfp4_cpu/bench/CMakeLists.txt` (drops `-march=native`)

**Interfaces:**
- Consumes: `Isa`, `detect_isa`, `SGLANG_TARGET_AVX2` (Task 2); `Nvfp4Quant` (Task 3).
- Produces: `template <Isa I, int M> void dot_rows(int n, const GpuRow& x, const block_q8_0* const* ys, float* out)`.

Bit-exactness argument, to check while porting: the AVX2 tier's per-token operation order is today's
`dot_gpu_rows` AVX2 branch exactly. The scalar tier is today's tail loop, which a portable build ran for every block.
Under `target("avx2,fma,f16c")` the compiler may not contract `mul`+`add` because `-ffp-contract=off` holds per
function. `_mm256_fmadd_ps` is explicit in the source and stays.

- [ ] **Step 1: Write the failing tests.** In `test_nvfp4_cpu_build.py`, replace
  `test_a_portable_build_uses_no_avx_registers` with:

```python
def test_the_scalar_cap_runs_the_scalar_tier(tmp_path):
    """NVFP4_CPU_MAX_ISA=scalar must select the scalar tier in a library that also holds the AVX2 one."""
    exe = _build_module().build(
        tmp_path / "ab", cxx=CXX, main=REPO / "test/manual/dsv41/nvfp4_cpu_forward_ab.cpp"
    )
    cores = sorted(os.sched_getaffinity(0))[:1]
    capped = subprocess.run(
        [str(exe), str(tmp_path / "scalar.bin"), str(cores[0])],
        env={**os.environ, "NVFP4_CPU_MAX_ISA": "scalar", "NVFP4_CPU_REPORT_ISA": "1"},
        capture_output=True, text=True, timeout=900,
    )
    assert capped.returncode == 0, capped.stdout + capped.stderr
    assert "nvfp4 isa scalar" in capped.stderr
```

`NVFP4_CPU_REPORT_ISA=1` makes `ExpertForward::isa()` print `"<kName> isa <tier>"` to stderr once; it lives in
`isa.hpp` for every quant (EXL3 gets `EXL3_MOE_CPU_REPORT_ISA` the same way in Task 7).
Extend `nvfp4_cpu_ggml_check.cpp` to run its whole loop for `dot_rows<Isa::Scalar, 1>` and, when
`__builtin_cpu_supports("avx2")`, `dot_rows<Isa::Avx2, 1>`, both against the same gold value and against each other
within the existing tolerance.

- [ ] **Step 2: Run; expect failure** (no `NVFP4_CPU_MAX_ISA` handling, `dot_rows<Isa::Scalar, 1>` undefined).

- [ ] **Step 3: Split the math** as the Files list says, move `dot_gpu_rows`'s branches into the two tier
  specializations, and make `Nvfp4Quant::dispatch` switch on the tier:

```cpp
static int dispatch(const Layer& l, const SglangCpuExpertsForward& c, const RouteTable& r, Isa isa)
{
    if (isa >= Isa::Avx2 && MimoV26ProShape::accepts(l.info))
        return ForwardPlan<MimoV26ProShape, Isa::Avx2>::run(l, c, r);
    return isa >= Isa::Avx2 ? ForwardPlan<GenericShape, Isa::Avx2>::run(l, c, r)
                            : ForwardPlan<GenericShape, Isa::Scalar>::run(l, c, r);
}
```

- [ ] **Step 4: Gate.** Container and (at the next push) divix01: the new `avx2` dump byte-identical to Task 0's
  `native` dump, the new `scalar` dump byte-identical to Task 0's `portable` dump. `run_nvfp4_cpu_forward_checks.sh`
  takes a `NVFP4_AB_BASE_NAMES=native,portable` mapping for this one comparison, so the old dump names still compare.
  Sanitizer and ggml_check pass.

- [ ] **Step 5: Commit** by name; message `feat(nvfp4-cpu): runtime ISA tiers (scalar, avx2) in math_scalar/math_avx2`.

---

### Task 6: EXL3 onto ExpertForward, torch paths as wrappers

**Files:**
- Create: `exl3_cpu/optimized/quant.hpp` (`Exl3Quant`; from `experts.hpp` and `moe_mul1.cpp`'s `RegisteredLayer`,
  `LayerInfo`, `SlabRowBytes`, `StridedExperts`, `TableExperts`)
- Modify: `exl3_cpu/optimized/moe_mul1.cpp`: the registry (`g_layers`), cores block, `detect_isa`, keep-warm and
  the C ABI bodies are deleted in favour of the framework; `forward_raw` becomes the plan entry `Exl3Quant::dispatch`
  calls; `exl3_moe_cpu_make_layer` registers a table layer into `ExpertForward<Exl3Quant>::layers`;
  `exl3_moe_cpu_forward(_raw)` build an `SglangCpuExpertsForward` and call `ExpertForward<Exl3Quant>::forward`,
  turning a nonzero status into `TORCH_CHECK(false, ...)`
- Modify: `exl3_cpu/optimized/cpu_experts_cabi.h` (`SglangExl3CpuParams { int32_t bits, swizzled; }`, five names)
- Delete: `exl3_cpu/optimized/experts.hpp`
- Modify: `python/sglang/srt/layers/moe/cpu_experts/exl3.py` (`register_layer` with `CpuExpertsLayer`; strides from
  each slab's `stride(0) * element_size()`; `native_set_cores` message for status 2)
- Modify callers: `kernels/jit/csrc/moe/expert_stream/bench/src/{cpu_forward,full_stack,stack_fixture}.cpp` (if
  they register), `test/manual/dsv41/test_cpu_expert_pool_exl3.py`, `test/manual/dsv41/exl3_cpu_forward_ab.py`
  (slab registration), `exl3_cpu/optimized/README.txt`
- Legacy `exl3_cpu/moe_mul1.cpp`: only `register_slabs` was never there; its `forward` and `set_cores` keep compiling
  against the new header. If the five-name header no longer declares `register_slabs`, nothing to change there.

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `struct Exl3Quant` (Quant contract; `Layer` holds either `MoeBufferRows<Exl3Quant>` or a
  `std::unique_ptr<MoeCpuLayer>` table, as `RegisteredLayer` does today); `kTopIsa = Isa::Vbmi`;
  `kIsaCapEnv = "EXL3_MOE_CPU_MAX_ISA"`; `kMaxRoutes = 32`; `kMaxRows = 65536`.

EXL3's layer row (`Row`) is today's `MoeCpuMatrix` triple: `Exl3Quant::decode` is `StridedExperts::gate/up/down`
re-expressed over `MoeBufferRow` bases. Strides now come from the descriptor. Registration refuses a stride below
`SlabRowBytes::of(hidden, intermediate, bits)`; at that minimum the addresses equal today's.

- [ ] **Step 1: Write the failing tests.** In `test/manual/dsv41/test_cpu_expert_pool_exl3.py`, add (they run in
  Task 7's divix01 checkpoint):
  - `test_the_c_abi_refuses_a_slot_past_capacity` (Review Focus 2): forward with slot `capacity` returns 2 and leaves
    `out` untouched.
  - `test_set_cores_after_the_first_forward_returns_2` (Review Focus 4).
  - `test_the_exl3_library_exports_the_five_c_names`: `getattr` on the extension's CDLL for each of
    `sglang_exl3_cpu_experts_{register_layer,free_layer,forward,keep_warm,set_cores}`.
  Port the existing `test_the_c_abi_forward_overwrites_or_accumulates` and the registration in
  `exl3_cpu_forward_ab.py` to `register_layer`.

- [ ] **Step 2: Implement** as the Files list says. `Exl3Quant::dispatch` is today's `run_plan` with `g_isa`
  replaced by the `isa` argument; `forward_raw`'s chunking stays in the EXL3 plan entry, unchanged (it decides EXL3's
  accumulation order; moving it into the common `RouteTable` would change its numerics).

- [ ] **Step 3: Gate (no local build: needs torch).** Commit, then Task 7.

- [ ] **Step 4: Commit** by name; message `refactor(exl3-cpu): register_layer/forward/keep_warm through ExpertForward<Exl3Quant>; torch paths wrap it`.

---

### Task 7: Checkpoint: EXL3 on divix01

- [ ] **Step 1: Ask the user to push.** On yes, push, then in a fresh private worktree:

```bash
ssh divix01 'W=/data/models/slang/nvfp4-work; D=$W/normalize-base
# ... worktree add at origin/numa-node-distributor as wt-normalize-t7 ...
bash $W/wt-normalize-t7/test/manual/dsv41/run_exl3_cpu_forward_checks.sh check $W/wt-normalize-t7 $W/normalize-t7 $D/exl3 slabs'
```

Expected: `ALL GREEN (check)`: 32/32 bit-exact per tier for both table and slab registrations, bare 24 and
full-stack 48 frozen outputs bit-exact, pool tests passing (now including Task 6's three new tests). Also re-run the
Task 4 NVFP4 pytest line and `test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py`. Known flake:
`test_a_failed_forward_aborts_the_process` also fails at BASE; record it, do not count it.

- [ ] **Step 2: If any EXL3 tier differs**, stop and bisect within Task 6's diff before continuing. Arithmetic is
  frozen, so a difference is a refactor bug, never a new baseline.

---

### Task 8: EXL3 standard math files

**Files:**
- Create: `exl3_cpu/optimized/math.hpp` (activation quantization: `PreparedIn`, `quantize_act`,
  `compact_quantize_*`, `prepare_rows`, `prepare_block_*`, Hadamard, `half_to_float`, decode helpers)
- Create: `exl3_cpu/optimized/math_scalar.hpp`, `math_avx2.hpp`, `math_avx512.hpp` (the GEMV tiles of each tier;
  `math_avx512.hpp` absorbs `register.hpp`, `traversal.hpp` and the Bw/Vnni/Vbmi band kernels)
- Delete: `exl3_cpu/optimized/register.hpp`, `exl3_cpu/optimized/traversal.hpp`
- Modify: `exl3_cpu/optimized/moe_mul1.cpp` (keeps only includes, the torch wrappers and the C ABI macro)

This is a relocation, nothing else. **REQUIRED SUB-SKILL:** `mechanical-refactor-verify`: produce the split as
prepare (any rename needed so blocks can move verbatim), move (byte-for-byte relocation, reproducible from the
primitives), postpare (includes), each its own commit, checked by the skill's reproduction.

- [ ] **Step 1: Prepare commit**, **Step 2: Move commit**, **Step 3: Postpare commit**, as the skill prescribes.
- [ ] **Step 4: Gate.** The move commit's reproduction check passes. EXL3 cannot build locally, so the build gate is
  Task 9's checkpoint.

---

### Task 9: Same names, same files, and the final checkpoint

**Files:**
- Rename: `{exl3_cpu,nvfp4_cpu}/optimized/moe_mul1.cpp` to `kernel.cpp`
- Modify every reference: `nvfp4_cpu/optimized/build.py`, `exl3_cpu/optimized/build.py`, `exl3_ext.py`
  (`OPTIMIZED_CPU_KERNEL`), `kernels/jit/csrc/moe/expert_stream/bench/CMakeLists.txt` (both `optimized/moe_mul1.cpp`
  uses), `nvfp4_cpu/bench/CMakeLists.txt`, both READMEs, `docs` are left alone
- Test: `test/registered/unit/kernels/test_cpu_experts_layout.py` (new)

- [ ] **Step 1: Write the failing shape test.** It asserts the standard file set and nothing else is required:

```python
"""Every CPU expert quant has the standard file set, so a new quant is a new directory filling in the same files."""

from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

QUANT = Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/quantization"
STANDARD = {"cpu_experts_cabi.h", "quant.hpp", "math.hpp", "math_scalar.hpp", "math_avx2.hpp", "shapes.hpp",
            "forward_plan.hpp", "kernel.cpp"}
QUANTS = ["exl3_cpu", "nvfp4_cpu"]


def test_every_quant_has_the_standard_files():
    for quant in QUANTS:
        present = {p.name for p in (QUANT / quant / "optimized").iterdir()}
        assert STANDARD <= present, (quant, sorted(STANDARD - present))


def test_every_quant_declares_the_five_c_names():
    for quant in QUANTS:
        prefix = quant.removesuffix("_cpu")
        header = (QUANT / quant / "optimized/cpu_experts_cabi.h").read_text()
        for name in ("register_layer", "free_layer", "forward", "keep_warm", "set_cores"):
            assert f"sglang_{prefix}_cpu_experts_{name}(" in header, (quant, name)
```

- [ ] **Step 2: Run; expect failure** (`kernel.cpp` missing).
- [ ] **Step 3: `git mv` both files and update every reference** (`git grep -n "optimized/moe_mul1.cpp\|\"moe_mul1.cpp\"" -- ':!docs'` prints nothing afterwards).
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Final checkpoint.** Ask to push; then on divix01 run Task 4's NVFP4 gate, Task 7's EXL3 gate, and the
  registered selection: `test_cpu_experts_{abi,common,layout}.py`, `test_nvfp4_cpu_{build,experts}.py`,
  `test_exl3_ram_miss_cpu_experts.py`. Record each command with its result. Remove the private worktrees afterwards.
- [ ] **Step 6: Commit** by name; message `refactor(cpu-experts): kernel.cpp per quant; the standard file set is tested`.

---

## Self-Review

1. **Coverage.** (1) normalized interface: Tasks 1, 2, 9. (2) `MoeBufferRow<Quant>`: Task 2, used by Tasks 3 and 6.
   (3) `ExpertForward` with a static layer array: Task 2 (keyed by quant; see Decisions 3). (4) standard files:
   Tasks 3, 5, 6, 8, 9. (5) C externs as wrappers: Task 2's macro, used in Tasks 3 and 6. (6) NVFP4 ISA
   specializations: Task 5.
2. **Placeholders.** None open. Task 2's slot access is fixed as `rows.slot(s)` plus `Quant::decode`.
3. **Types.** `SglangCpuExpertsLayer`, `SglangCpuExpertsForward`, `MoeBufferRows<Quant>::slot`,
   `Quant::{validate,make_layer,check_slot,dispatch,decode,min_slot_bytes}`, `RouteTable::{count,route}`,
   `ExpertForward<Quant>::{register_layer,free_layer,forward,keep_warm,lookup}` are used with the same names
   throughout.
4. **Review Focus.** Each item has its test: 1 and 3 in Task 2, 2 and 4 in Task 6, 5 in Task 5.
