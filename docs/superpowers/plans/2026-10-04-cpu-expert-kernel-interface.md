# CPU Expert Kernel Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-written CPU expert C ABI (six `extern "C"` functions per kernel library, ctypes glue, handle and engine registries) with one C++ virtual interface, `CpuExpertKernel`, that the expert-stream host calls directly through a kernel address each quant's own binding hands out.

**Architecture:** A new header `host/cpu_experts/kernel.hpp` defines `LayerSlabs`, `ForwardCall`, `CpuExpertLayer` and `CpuExpertKernel`. `ExpertForward<Quant>` becomes a `final` implementation of it, with no registry and no lock; each kernel library defines one hidden-visibility accessor (`exl3_cpu_kernel()`, `nvfp4_cpu_kernel()`). The EXL3 torch extension exposes the accessor's address as the op `sglang_exl3_cpu::kernel_address`, the NVFP4 library as the tvm-ffi export `nvfp4_cpu_kernel_address`. The host's `CpuExpertEngine` calls `kernel->forward` / `kernel->keep_warm`; a per-`RamTier` `CpuExpertLayers` owns every row's layer and is shared by every group's engine. The C ABI survives as a transitional shim (`cabi.hpp` re-implemented over the kernel) until every caller has moved, then is deleted.

**Tech Stack:** C++20 (GCC 15, OpenMP/libgomp), TVM-FFI JIT modules (`sglang.kernels.jit.utils.load_jit`, `tvm_ffi.load_module`), PyTorch C++ extension (`torch.utils.cpp_extension.load`, `TORCH_LIBRARY`), Python 3.13, pytest, CMake (benches).

**Spec:** `docs/superpowers/specs/2026-10-04-cpu-expert-kernel-interface-design.md` (binding; its design decisions are settled: a virtual interface, not templates; each quant binding returns its kernel address; `CpuExpertPool` is deleted).

## Global Constraints

- Code reaches divix01 only by commit, `git push origin cpu-kernel-iface`, and a pulled private worktree `/data/models/slang/nvfp4-work/wt-kiface` (`.claude/rules/divix01-run-protocol.md`). Never rsync/scp/git-archive. Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- CPU jobs: `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest ... -q -p no:randomly`, status read from `${PIPESTATUS[0]}`, with `CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++` exported. GPU work only through `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh` (exit 75 = the lock timed out, not a test failure).
- Bitwise numerics of both kernels unchanged: the EXL3 dumps (`CPU_CHECKS` against `nlane-cpu-base`), the NVFP4 dumps (`NVFP4_AB` against `kiface-nvfp4-base`) and the benches' frozen references must stay byte-identical.
- Commits: never amend, rebase or force-push; stage files by name (`git add <path>`; `git rm <path>` for deletions); run `git diff --cached --stat` before each commit and check it lists exactly the task's files. Every message ends with the two lines
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft`.
- Env vars: this plan adds none. Any new `SGLANG_*` variable would follow `.claude/skills/env-var-conventions/SKILL.md`; the existing kernel tier variables (`EXL3_MOE_CPU_MAX_ISA`, `NVFP4_CPU_MAX_ISA`, ...) are unchanged.
- The interface's signatures use only plain structs, `std::span`, `std::array` and `std::unique_ptr` (no `std::string`, no `std::vector`), so the host and kernels may differ in `_GLIBCXX_USE_CXX11_ABI` and visibility. No RTTI crosses a library: `static_cast` after the `&layer.kernel() == this` check, never `dynamic_cast`/`typeid`.
- Each kernel accessor is `__attribute__((visibility("hidden")))` so two libraries that define it in one process never interpose each other's. A kernel library must stay loaded for as long as a host holds its kernel or layers: the loaders that hand out addresses are cached for the process (`functools.cache` / the torch extension), never unloaded.
- Comments follow `.claude/rules/comment-style.md`. `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` edits touch only the CPU-experts section and the member list.

### Run templates (used by every task)

`SYNC` (laptop, after committing):

```bash
git push origin cpu-kernel-iface
ssh divix01 'set -e; R=/data/models/slang/sglang; W=/data/models/slang/nvfp4-work/wt-kiface;
  git -C $R fetch origin;
  if [ -d $W ]; then git -C $W checkout --detach origin/cpu-kernel-iface;
  else git -C $R worktree add --detach $W origin/cpu-kernel-iface; fi;
  git -C $W log -1 --oneline'
```

Before trusting any run in a fresh worktree: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-kiface && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang;print(sglang.__file__)"'` must print a path under `wt-kiface/python/`.

`RUN_CPU <files...>` (divix01, the narrow selection a task names):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-kiface && export CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++;
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <files...> \
  -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

`KIFACE_CPU [extra files...]` (laptop; the full CPU suite, `p2-cpu-par.sh` pointed at `wt-kiface`). The runner hard-codes `wt-nlane` and lists `test_cpu_expert_pool.py`; `POOL` is that file until Task 7 deletes it, `test/registered/unit/kernels/test_cpu_expert_service.py` from Task 7 on:

```bash
S=/private/tmp/claude-501/-Users-dnikolaidis-Desktop-divix-sglang-nvfp4/12ab61b9-282a-4593-a29c-35d2fcbda06a/scratchpad/p2-cpu-par.sh
POOL=${POOL:-test/registered/unit/kernels/test_cpu_expert_pool.py}
sed -e 's#nvfp4-work/wt-nlane#nvfp4-work/wt-kiface#' \
    -e "s# test/registered/unit/kernels/test_cpu_expert_pool.py# $POOL#" "$S" \
  | ssh divix01 'CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ bash -s -- test/registered/unit/kernels/test_cpu_experts_common.py test/registered/unit/kernels/test_cpu_experts_layout.py <extra files...>'
```

Expected: the runner's last line `EXIT=0` (its first line is the worktree's `git log -1`, check it is the task's commit).

`RUN_GPU <files...>` (divix01, under the GPU lock):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-kiface && CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh \
  /data/models/slang/.venv/bin/python -m pytest <files...> -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

`RUN_EXT <files...>` (divix01; manual tests needing the EXL3 extension built with the optimized kernel, from a private build directory; refuse to run while `pgrep -f sglang.launch_server` finds a server). Task 3 adds a source to the extension, so the first `RUN_EXT` after it rebuilds the flavor in `kiface-exl3-build` (minutes):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-kiface && B=/data/models/slang/nvfp4-work/kiface-exl3-build;
  [ -d $B/resid_b128_cpu_v1 ] || { mkdir -p $B && cp -a ~/.cache/sglang/exl3_ext/resid_b128_cpu_v1 $B/; };
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 EXL3_MOE_CPU_PIN=0 SGLANG_DSV41_CPU_EXPERTS=1 \
  SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 SGLANG_EXL3_BUILD_DIR=$B \
  SGLANG_EXL3_CPU_CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
  CUDA_HOME=/usr/local/cuda-13.4 taskset -c 18-21 \
  /data/models/slang/.venv/bin/python -m pytest <files...> -q -p no:randomly 2>&1 | tail -15; echo EXIT=${PIPESTATUS[0]}'
```

`RUN_EXT`, `RUN_GPU`, `RUN_CPU` and (from Task 1) `run_exl3_cpu_forward_checks.sh` all export `CXX` as GCC 15: the expert-stream host module is JIT-built with `$CXX` (else `c++`, `jit/utils/compile/toolchain.py`), and it must share GCC 15's libstdc++ and C++ ABI with both kernel libraries.

`SUITE_EXT`: `RUN_EXT test/manual/dsv41/test_cpu_expert_engines_exl3.py test/manual/dsv41/test_cpu_expert_pool_exl3.py` (from Task 7 on, without `test_cpu_expert_pool_exl3.py`, which that task deletes).

`CPU_CHECKS <task>` (divix01; the EXL3 bit-exact gate: per-tier dumps through the table and the slab registration compared with the existing make_layer baseline `nlane-cpu-base`, the bare-forward bench's 24 frozen outputs, the full-stack bench's 48, and the script's pytest step; no server may run):

```bash
ssh divix01 'bash /data/models/slang/nvfp4-work/wt-kiface/test/manual/dsv41/run_exl3_cpu_forward_checks.sh check \
  /data/models/slang/nvfp4-work/wt-kiface /data/models/slang/nvfp4-work/kiface-cpu-<task> \
  /data/models/slang/nvfp4-work/nlane-cpu-base slabs 2>&1 | tail -20; echo EXIT=${PIPESTATUS[0]}'
```

Expected: `ALL GREEN (check)`, `EXIT=0`.

`NVFP4_AB <task>` (divix01; the NVFP4 bit-exact gate: harness and kernel built from the worktree, every one-row output and status byte-compared with Task 0's dump):

```bash
ssh divix01 'bash /data/models/slang/nvfp4-work/wt-kiface/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh \
  /data/models/slang/nvfp4-work/wt-kiface /data/models/slang/nvfp4-work/kiface-nvfp4-<task> \
  /data/models/slang/nvfp4-work/kiface-nvfp4-base; echo EXIT=$?'
```

Expected: `PASS avx2 bit-exact vs avx2`, `PASS scalar bit-exact vs scalar`, `EXIT=0`.

`BENCH <dir> <nodes>` (divix01; the C++ benches outside the JIT, then the full stack's self-test on synthetic rows with a fake kernel):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-kiface && B=/data/models/slang/nvfp4-work/<dir>; SITE=/data/models/slang/.venv/lib/python3.13/site-packages;
  taskset -c 0-63 cmake -S python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $B -DEXPERT_STREAM_NODES=<nodes> \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src >/dev/null \
  && taskset -c 0-63 cmake --build $B -j 16 2>&1 | tail -3; echo BUILD=${PIPESTATUS[0]};
  mkdir -p $B/images && taskset -c 0-63 $B/exl3_full_stack_prod --self-test --image-dir=$B/images 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'
```

Expected: `BUILD=0`, `EXIT=0`. Run as `BENCH kiface-bench 1` and `BENCH kiface-bench-n2 2`.

## Review Focus

1. A kernel library unloaded (a garbage-collected tvm-ffi `Module`) while the host still holds its kernel or a layer would crash at the next forward or at teardown in the layer's destructor; the loader must be process-lifetime cached. Pinned by Task 3's `test_the_module_is_loaded_once_and_kept`.
2. A layer made by one kernel reaching another: two groups enabled with different kernels, or a foreign layer handed to `set_cpu_layer`. Expect a refusal naming both, never a misread layer. Pinned by Task 1's `a_layer_of_another_kernel_is_refused`, Task 2's cross-library harness and Task 5's `test_groups_must_share_one_kernel` self-test.
3. Exceptions crossing a `.so` boundary built with `-fvisibility=hidden` (the host module's flags): a kernel's `std::invalid_argument` must reach the host's `catch (const std::exception&)`, with the layer's deleting destructor running in the library that made it. Pinned by Task 2's cross-library harness.
4. Bad registration input through the FFI: a second `set_cpu_layer` for a row, one before `enable_cpu_experts`, a malformed slab table, params of the wrong size. Expect an error naming the cause, nothing installed. Pinned by Task 4's `test_a_rows_layer_is_made_once_by_the_enabled_kernel` and Task 6's `test_the_kernel_refuses_params_of_the_wrong_size`.
5. A forward that throws on the CPU expert thread (a refused input, a failed pin) must fail-stop naming the row and the kernel's message, and leave the output part untouched. Pinned by Task 4's `test_a_failed_forward_aborts_the_process` (message now carries the exception text) and Task 1's `a_failed_pin_throws_runtime_error_and_leaves_out_untouched`.

## Resolutions of spec gaps (decided here, binding on the tasks)

- **The C ABI is kept, re-implemented over the kernel, until Task 9.** `cabi.hpp` becomes a transitional shim with its own handle table, engine table, `last_error()` and status mapping, so every existing caller works at every boundary while callers move one by one.
- **`set_cpu_layer`'s "slab tensors" are an int64 `[n, 2]` table of `{address, slot bytes}`** (address 0 for an absent optional slab) plus scalars and a uint8 params tensor, the way `open()` already takes slab addresses. Python builds it from the slab tensors (`CpuExpertLayerSpec`, Task 3).
- **Core lists are validated by the kernel on every call** (`check_cores`: each in `[0, CPU_SETSIZE)`, no repeat), replacing `Engines::create`'s check; `CPU_SET` past `CPU_SETSIZE` would be undefined.
- **Quant contract:** `Quant::validate` and `Quant::dispatch` keep returning `int` status; `ExpertForward` turns a nonzero `validate` and a `dispatch` status 2 into `std::invalid_argument`, any other nonzero status into `std::runtime_error`. Harnesses that write statuses map them back (0 / 1 / 2) so the NVFP4 dumps stay byte-identical.
- **EXL3's upstream link-compat API** (`exl3_moe_cpu_make_layer` / `exl3_moe_cpu_free_layer` / `exl3_moe_cpu_forward[_raw]`, which upstream `bindings.cpp` references and sglang cannot edit) stays, with its own handle table in `kernel.cpp` (shared_ptr, so no status 3). The success-criteria grep uses `git grep -w`, so these names do not match `free_layer`.
- **The vendored `exl3/moe_mul1.cpp`'s sglang C ABI** (`sglang_exl3_cpu_experts_forward`, `..._set_cores`) becomes two C++-linkage functions the bench's baseline backend calls (`exl3_moe_cpu_baseline_forward`, `exl3_moe_cpu_baseline_set_cores`).
- **The FakeKernel replaces both native fakes and the Python CFUNCTYPE fakes.** One kernel per tier means per-group fakes are gone: a call records `cores.front()` (the group), a hold (`test_kernel_hold(core)`) gates one group, `fail` makes every forward throw, `zero` makes it write a zero partial (the GPU lane-order test). `test_kernel_address(ns_per_expert, fail, zero)` therefore takes two more arguments than the spec's one-argument sketch; `test_keep_warm_engine` becomes `test_keep_warm_core`.
- **Kernel-correctness tests call the kernel through the host module's test exports** (`expert_stream_kernel_layer` / `_kernel_forward` / `_kernel_drop`, instr build only), generic over any kernel address. The module-level Python wrappers are named `kernel_layer` / `kernel_forward` / `kernel_drop` (not `test_*`: pytest would collect an imported `test_*` function).
- **`test_cpu_expert_pool.py` is split, not dropped wholesale**: its policy, service, calibration, groups and EXL3-trait tests move to `test_cpu_expert_service.py` (ported to the new trait API); its `CpuExpertPool` tests and its `free_layer` tests are deleted (Task 7).
- **The trait module is renamed** `cpu_experts/pool.py` -> `cpu_experts/trait.py` (Task 3 creates `trait.py` with `CpuExpertLayerSpec`; Task 7 moves the Protocol there and deletes `pool.py`).

---

### Task 0: Baseline

**Files:** none.

- [ ] **Step 1:** `SYNC` (the branch head is the plan commit `c69f0dca2d` or a later plan-only commit; none changes code). Check `sglang.__file__`.
- [ ] **Step 2:** Confirm the EXL3 baseline exists: `ssh divix01 'ls /data/models/slang/nvfp4-work/nlane-cpu-base/ab-{bw,avx2,scalar}-table.pt'`. If any is missing, make it at `53963aa379`:
  `ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-kiface-base 53963aa379 && bash /data/models/slang/nvfp4-work/wt-kiface-base/test/manual/dsv41/run_exl3_cpu_forward_checks.sh baseline /data/models/slang/nvfp4-work/wt-kiface-base /data/models/slang/nvfp4-work/nlane-cpu-base 2>&1 | tail -5'` -- expect `ALL GREEN (baseline)`.
- [ ] **Step 3:** Make the NVFP4 baseline dump: `ssh divix01 'bash /data/models/slang/nvfp4-work/wt-kiface/test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh /data/models/slang/nvfp4-work/wt-kiface /data/models/slang/nvfp4-work/kiface-nvfp4-base; echo EXIT=$?'` -- expect `DUMPED avx2: ...`, `DUMPED scalar: ...`, `EXIT=0`.
- [ ] **Step 4:** `KIFACE_CPU test/registered/unit/kernels/test_cpu_experts_abi.py`, `SUITE_EXT`, `CPU_CHECKS base`, and `BENCH kiface-bench 1`. Record in the ledger as `Baseline:` the parallel and serial counts, the `SUITE_EXT` counts, the `CPU_CHECKS` final line and the BENCH lines, each with its command.

---

### Task 1: The kernel interface, `ExpertForward` as a kernel, the C ABI as a shim

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/kernel.hpp`
- Create: `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h`
- Create: `python/sglang/kernels/jit/csrc/nvfp4/optimized/kernel.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/expert_forward.hpp` (rewrite)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/team.hpp` (span cores, `check_cores`; `last_error`/`fail`/`Engines` move out)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/keep_warm.hpp` (span cores, throws)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/buffer_row.hpp` (`of(const LayerSlabs&)`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/cabi.hpp` (rewrite as the transitional shim)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/quant.hpp`, `forward_plan.hpp:895-905`, `kernel.cpp`
- Modify: `python/sglang/kernels/jit/csrc/nvfp4/optimized/quant.hpp`, `forward_plan.hpp:127-145`, `kernel.cpp`
- Modify: `test/registered/unit/kernels/cpu_experts_common_toy.hpp`, `cpu_experts_common_toy_lib.cpp`, `cpu_experts_common_check.cpp` (rewrite), `test_cpu_experts_common.py` (`CHECKS`)
- Modify: `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` (one line: `export CXX`)

**Interfaces:**
- Produces (`kernel.hpp`, namespace `sglang::cpu_experts`, external linkage, NOT in an anonymous namespace):

```cpp
inline constexpr int kMaxSlabs = 8;
struct LayerSlabs { int32_t capacity = 0, hidden = 0, intermediate = 0, activation = 0; float act_limit = 0;
                    int32_t slab_count = 0; std::array<const void*, kMaxSlabs> slabs{}; std::array<uint64_t, kMaxSlabs> slot_bytes{}; };
struct ForwardCall { int32_t rows = 0, k = 0, threads = 1; const void* x = nullptr; const int32_t* slots = nullptr;
                     const float* weights = nullptr; float* out = nullptr; bool accumulate = false; std::span<const int> cores; };
class CpuExpertLayer;   // virtual ~; const CpuExpertKernel& kernel() const; protected explicit CpuExpertLayer(const CpuExpertKernel&)
class CpuExpertKernel { virtual const char* name() const noexcept = 0;
  virtual std::unique_ptr<CpuExpertLayer> make_layer(const LayerSlabs&, std::span<const std::byte> params) const = 0;
  virtual void forward(const CpuExpertLayer&, const ForwardCall&) const = 0;
  virtual void keep_warm(std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) const = 0; };
```

- Produces (framework, anonymous namespace inside `sglang::cpu_experts`): `template <class Quant> class ExpertForward final : public CpuExpertKernel` with nested `class Layer final : public CpuExpertLayer { const typename Quant::Layer quant; }`, `static Isa isa()`, `std::unique_ptr<CpuExpertLayer> wrap(typename Quant::Layer) const`; `inline void check_cores(std::span<const int>)`; `template <Isa Top> void keep_warm(Isa, std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns)`; `CallCores(std::span<const int>)`.
- Produces (accessors): `namespace sglang::exl3_cpu { __attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel(); }`, `namespace sglang::nvfp4_cpu { ... nvfp4_cpu_kernel(); }`, `namespace toy { const ::sglang::cpu_experts::CpuExpertKernel& TOY_KERNEL(); }` (`TOY_KERNEL` defaults to `toy_kernel`).
- Produces (shim): `SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant, accessor)` -- the same six `extern "C"` functions as today, now over `accessor()`; `CabiLayers::add(std::shared_ptr<const CpuExpertLayer>) -> int64_t`; `last_error()`, `fail()` and `Engines` now live in `cabi.hpp`.
- Quant contract changes (every quant): `min_slot_bytes(const LayerSlabs&, const Params&)`, `validate(const LayerSlabs&, const Params*) -> int`, `make_layer(const LayerSlabs&, const Params*) -> Layer`, `dispatch(const Layer&, const ForwardCall&, const RouteTable&, Isa) -> int`. `Params` stays where it is (`cpu_experts_cabi.h`) until Task 9.

- [ ] **Step 1: Write the failing test.** Replace `test/registered/unit/kernels/cpu_experts_common_check.cpp` with the interface harness below; it does not compile until the interface exists. The C ABI checks (`abi_versions_are_checked`, `unknown_handle_is_refused`, the free racing a forward, engine create/free, `last_error`) are deleted, not ported (spec, "Benches and tests").

```cpp
// Standalone harness for host/cpu_experts: a toy quant through ExpertForward, as a CpuExpertKernel. Built by
// test_cpu_experts_common.py.
#include "cpu_experts_common_toy.hpp"
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace toy {
const sglang::cpu_experts::CpuExpertKernel& toy_kernel()
{
    static const sglang::cpu_experts::ExpertForward<ToyQuant> kernel{};
    return kernel;
}
}  // namespace toy

namespace {
using sglang::cpu_experts::CpuExpertKernel;
using sglang::cpu_experts::CpuExpertLayer;
using sglang::cpu_experts::ForwardCall;
using sglang::cpu_experts::Isa;
using sglang::cpu_experts::LayerSlabs;

constexpr int kCapacity = 3;
constexpr int kHidden = 16;

int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// What a call threw: 0 nothing, 2 std::invalid_argument, 1 any other std::exception.
template <class F>
int status_of(F&& f)
{
    try {
        f();
        return 0;
    } catch (const std::invalid_argument&) {
        return 2;
    } catch (const std::exception&) {
        return 1;
    }
}

struct Fixture {
    std::vector<float> slab = std::vector<float>(size_t(kCapacity) * kHidden);
    toy::ToyParams params{2.0f};

    Fixture()
    {
        for (int s = 0; s < kCapacity; ++s)
            for (int h = 0; h < kHidden; ++h) slab[size_t(s) * kHidden + h] = float(s + h);
    }

    LayerSlabs layer() const
    {
        LayerSlabs d;
        d.capacity = kCapacity;
        d.hidden = kHidden;
        d.intermediate = kHidden;
        d.slab_count = 1;
        d.slabs[0] = slab.data();
        d.slot_bytes[0] = uint64_t(kHidden) * 4;
        return d;
    }

    std::span<const std::byte> params_bytes() const { return std::as_bytes(std::span<const toy::ToyParams>(&params, 1)); }
};

std::unique_ptr<CpuExpertLayer> make_toy(const Fixture& f) { return toy::toy_kernel().make_layer(f.layer(), f.params_bytes()); }

// One forward's buffers; out starts at 7 so a refused call is seen to leave it untouched.
struct Call {
    const CpuExpertLayer* layer;
    std::vector<uint16_t> x;
    std::vector<int32_t> slots;
    std::vector<float> weights;
    std::vector<float> out;
    std::vector<int> cores;
    ForwardCall c;

    Call(const CpuExpertLayer& l, int rows, int k, std::vector<int32_t> s, std::vector<float> w, int threads = 2,
         std::vector<int> on = {})
        : layer(&l), x(size_t(rows) * kHidden), slots(std::move(s)), weights(std::move(w)),
          out(size_t(rows) * kHidden, 7.0f), cores(std::move(on))
    {
        c.rows = rows;
        c.k = k;
        c.threads = threads;
        c.x = x.data();
        c.slots = slots.data();
        c.weights = weights.data();
        c.out = out.data();
        c.cores = cores;
    }
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;

    int run(const CpuExpertKernel& kernel = toy::toy_kernel()) { return status_of([&] { kernel.forward(*layer, c); }); }
    bool untouched() const
    {
        for (float v : out)
            if (v != 7.0f) return false;
        return true;
    }
};

std::vector<int> allowed_cores()
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CHECK(sched_getaffinity(0, sizeof(set), &set) == 0);
    std::vector<int> cores;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) cores.push_back(c);
    return cores;
}

void ok(const char* name) { std::printf("ok %s\n", name); std::fflush(stdout); }

}  // namespace

int main()
{
    // isa() is computed once, at the first forward: the cap and the report switch must be in the environment before
    // it. The Python test reads the report ("toy isa scalar") from stderr.
    CHECK(setenv("TOY_CPU_MAX_ISA", "scalar", 1) == 0);
    CHECK(setenv("TOY_CPU_REPORT_ISA", "1", 1) == 0);
    Fixture f;
    const CpuExpertKernel& kernel = toy::toy_kernel();
    CHECK(std::string(kernel.name()) == "toy");
    const std::vector<int> allowed = allowed_cores();
    const int team = allowed.size() >= 2 ? 2 : 1;
    const std::vector<int> cores(allowed.begin(), allowed.begin() + team);

    {
        const auto layer = make_toy(f);
        CHECK(&layer->kernel() == &kernel);
        // Token 0: slots 0 and 2; token 1: slot 1 and a skipped -1.
        Call call(*layer, 2, 2, {0, 2, 1, -1}, {0.5f, 0.25f, 2.0f, 1.0f}, team, cores);
        for (int accumulate = 0; accumulate < 2; ++accumulate) {
            call.c.accumulate = accumulate != 0;
            CHECK(call.run() == 0);
            for (int hh = 0; hh < kHidden; ++hh) {
                const float base = accumulate ? 1.0f : 0.0f;
                const float t0 = 0.5f * 2.0f * float(0 + hh) + 0.25f * 2.0f * float(2 + hh);
                const float t1 = 2.0f * 2.0f * float(1 + hh);
                CHECK(call.out[hh] == base + t0);
                CHECK(call.out[kHidden + hh] == base + t1);
            }
            std::fill(call.out.begin(), call.out.end(), 1.0f);
        }
        ok("make_layer_then_forward_overwrites_and_accumulates");
    }

    {
        const LayerSlabs good = f.layer();
        auto refused = [&](const LayerSlabs& d, std::span<const std::byte> p) {
            return status_of([&] { kernel.make_layer(d, p); }) == 2;
        };
        LayerSlabs d = good;
        d.slot_bytes[0] = uint64_t(kHidden) * 4 - 4;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.slabs[0] = nullptr;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.slab_count = 2;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.activation = 1;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.capacity = 0;
        CHECK(refused(d, f.params_bytes()));
        CHECK(refused(good, {}));  // the toy needs its params
        const std::byte wide[8]{};
        CHECK(refused(good, wide));  // not sizeof(ToyParams)
        ok("params_and_slabs_are_validated");
    }

    {
        // A second kernel object of the same quant (another library's, in production) refuses this one's layer.
        const sglang::cpu_experts::ExpertForward<toy::ToyQuant> other{};
        const auto layer = make_toy(f);
        Call call(*layer, 1, 1, {0}, {1.0f});
        CHECK(call.run(other) == 2);
        CHECK(call.untouched());
        CHECK(call.run() == 0);
        ok("a_layer_of_another_kernel_is_refused");
    }

    {
        const auto layer = make_toy(f);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const int max_routes = toy::ToyQuant::kMaxRoutes, max_rows = toy::ToyQuant::kMaxRows;
        Call bad_slot(*layer, 1, 1, {kCapacity}, {1.0f});
        Call negative_slot(*layer, 1, 1, {-2}, {1.0f});
        Call nan_weight(*layer, 1, 1, {0}, {nan});
        Call wide(*layer, 1, max_routes + 1, std::vector<int32_t>(max_routes + 1, 0),
                  std::vector<float>(max_routes + 1, 1.0f));
        Call tall(*layer, max_rows + 1, 1, std::vector<int32_t>(max_rows + 1, 0),
                  std::vector<float>(max_rows + 1, 1.0f));
        Call empty(*layer, 1, 1, {0}, {1.0f});
        empty.c.rows = 0;
        Call no_threads(*layer, 1, 1, {0}, {1.0f});
        no_threads.c.threads = 0;
        Call null_slots(*layer, 1, 1, {0}, {1.0f});
        null_slots.c.slots = nullptr;
        Call null_out(*layer, 1, 1, {0}, {1.0f});
        null_out.c.out = nullptr;
        for (Call* c : {&bad_slot, &negative_slot, &nan_weight, &wide, &tall, &empty, &no_threads, &null_slots}) {
            CHECK(c->run() == 2);
            CHECK(c->untouched());
        }
        CHECK(null_out.run() == 2);
        // k = 0 has no routes to read: the output is overwritten with zeros.
        Call no_routes(*layer, 1, 0, {}, {});
        no_routes.c.slots = nullptr;
        no_routes.c.weights = nullptr;
        CHECK(no_routes.run() == 0);
        for (float v : no_routes.out) CHECK(v == 0.0f);
        // -1 slots and zero weights are dropped, the routing order kept.
        Call sparse(*layer, 1, 4, {1, -1, 2, 0}, {0.0f, 1.0f, 0.5f, 0.25f});
        CHECK(sparse.run() == 0);
        const auto& kept = toy::ToyQuant::last_routes;
        CHECK(kept.size() == 2 && kept[0].slot == 2 && kept[0].weight == 0.5f && kept[1].slot == 0
              && kept[1].weight == 0.25f);
        ok("routes_are_validated");
    }

    {
        // No lock: a forward parked inside its dispatch does not stop another on the same layer.
        const auto layer = make_toy(f);
        toy::ToyQuant::inside.store(false);
        toy::ToyQuant::hold.store(true);
        int first = -1;
        Call held(*layer, 1, 1, {0}, {1.0f}, 1, cores);
        std::thread runner([&] {
            toy::ToyQuant::park_here = true;
            first = held.run();
        });
        while (!toy::ToyQuant::inside.load()) std::this_thread::yield();
        Call second(*layer, 1, 1, {0}, {1.0f}, 1);
        CHECK(second.run() == 0);
        CHECK(!second.untouched());
        toy::ToyQuant::hold.store(false);
        runner.join();
        CHECK(first == 0);
        ok("forwards_run_at_once");
    }

    {
        CHECK(toy::ToyQuant::last_isa == Isa::Scalar);
        CHECK(sglang::cpu_experts::ExpertForward<toy::ToyQuant>::isa() == Isa::Scalar);
        const Isa hw = sglang::cpu_experts::detect_isa(Isa::Vbmi, nullptr);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Avx2, nullptr) <= Isa::Avx2);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Scalar, nullptr) == Isa::Scalar);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Vbmi, "TOY_CPU_MAX_ISA") == Isa::Scalar);
        CHECK(setenv("TOY_CPU_BOGUS_ISA", "avx9000", 1) == 0);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Vbmi, "TOY_CPU_BOGUS_ISA") == hw);
        // A cap above the hardware never raises the tier.
        CHECK(setenv("TOY_CPU_HIGH_ISA", "VBMI", 1) == 0);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Vbmi, "TOY_CPU_HIGH_ISA") == hw);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Avx2, "TOY_CPU_HIGH_ISA") <= Isa::Avx2);
        std::printf("detected %d\n", int(hw));
        ok("isa_cap_env_lowers_the_tier");
    }

    if (allowed.size() >= 2) {
        // The engine thread may run under a mask that excludes the expert cores (the server runs under taskset): a
        // call accepts a core outside the caller's affinity; only the workers' own pins must succeed.
        const auto layer = make_toy(f);
        cpu_set_t saved, one;
        CHECK(pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved) == 0);
        CPU_ZERO(&one);
        CPU_SET(allowed[0], &one);
        CHECK(pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0);
        Call outside(*layer, 1, 1, {0}, {1.0f}, 1, {allowed[1]});
        CHECK(outside.run() == 0);
        CHECK(toy::ToyQuant::last_cpus == std::vector<int>{allowed[1]});
        CHECK(pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved) == 0);
        ok("a_core_outside_the_callers_affinity_is_pinned");
    }

    {
        // More workers than cores, a repeated core and a core outside [0, CPU_SETSIZE) are refused, out untouched;
        // without cores the team is unpinned and has no core limit.
        const auto layer = make_toy(f);
        Call too_many(*layer, 1, 1, {0}, {1.0f}, team + 1, cores);
        Call repeated(*layer, 1, 1, {0}, {1.0f}, 1, {cores[0], cores[0]});
        Call past(*layer, 1, 1, {0}, {1.0f}, 1, {CPU_SETSIZE});
        Call negative(*layer, 1, 1, {0}, {1.0f}, 1, {-1});
        for (Call* c : {&too_many, &repeated, &past, &negative}) {
            CHECK(c->run() == 2);
            CHECK(c->untouched());
        }
        Call unpinned(*layer, 1, 1, {0}, {1.0f}, team + 1);
        CHECK(unpinned.run() == 0);
        ok("cores_are_validated_and_bound_the_team");
    }

    {
        // Each call's team runs on its own cores, also while another call's team runs at once from another thread
        // (the two-team half needs four allowed CPUs).
        const auto layer = make_toy(f);
        Call one(*layer, 1, 1, {0}, {1.0f}, team, cores);
        CHECK(one.run() == 0);
        CHECK(toy::ToyQuant::last_cpus == cores);
        if (allowed.size() >= 4) {
            const std::vector<int> a = {allowed[0], allowed[1]}, b = {allowed[2], allowed[3]};
            std::vector<int> seen_a, seen_b;
            std::atomic<int> bad{0};
            auto run = [&](const std::vector<int>& on, std::vector<int>* seen) {
                for (int i = 0; i < 200; ++i) {
                    Call call(*layer, 1, 1, {0}, {1.0f}, 2, on);
                    if (call.run() != 0) bad.fetch_add(1);
                    seen->insert(seen->end(), toy::ToyQuant::last_cpus.begin(), toy::ToyQuant::last_cpus.end());
                }
            };
            std::thread ta(run, std::cref(a), &seen_a), tb(run, std::cref(b), &seen_b);
            ta.join();
            tb.join();
            CHECK(bad.load() == 0);
            CHECK(seen_a.size() == 400 && seen_b.size() == 400);
            for (int cpu : seen_a) CHECK(cpu == a[0] || cpu == a[1]);
            for (int cpu : seen_b) CHECK(cpu == b[0] || cpu == b[1]);
        }
        ok("each_calls_team_runs_on_its_own_cores");
    }

    {
        // A core in range but past this machine's CPUs passes validation and fails the worker's pin: a
        // std::runtime_error, out untouched; the next call runs.
        const auto layer = make_toy(f);
        Call fails(*layer, 1, 1, {0}, {1.0f}, 1, {CPU_SETSIZE - 1});
        CHECK(fails.run() == 1);
        CHECK(fails.untouched());
        Call fits(*layer, 1, 1, {0}, {1.0f}, team, cores);
        CHECK(fits.run() == 0);
        ok("a_failed_pin_throws_runtime_error_and_leaves_out_untouched");
    }

    {
        uint32_t word = 5;
        auto warm = [&](std::span<const int> on, int32_t threads, const uint32_t* w, int64_t deadline) {
            return status_of([&] { kernel.keep_warm(on, threads, w, 5, deadline); });
        };
        CHECK(warm(cores, 0, &word, now_ns() + 1000000000) == 2);
        CHECK(warm(cores, 1, nullptr, now_ns() + 1000000000) == 2);
        CHECK(warm(cores, team + 1, &word, now_ns() + 1000000000) == 2);
        const std::vector<int> repeated = {cores[0], cores[0]};
        CHECK(warm(repeated, 1, &word, now_ns() - 1) == 2);
        CHECK(warm({}, team + 1, &word, now_ns() - 1) == 0);  // no cores: no core limit
        // An expired deadline returns at the first clock poll.
        CHECK(warm(cores, team, &word, now_ns() - 1) == 0);
        const int64_t start = now_ns();
        std::thread mover([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            __atomic_store_n(&word, 6u, __ATOMIC_RELEASE);
        });
        const int r = warm(cores, team, &word, start + 60LL * 1000000000);
        mover.join();
        CHECK(r == 0);
        CHECK(now_ns() - start < 1000000000);
        // Every tier's loop, as far as this CPU runs them, through the free function.
        for (Isa tier : {Isa::Scalar, Isa::Avx2, Isa::Bw, Isa::Vnni, Isa::Vbmi}) {
            if (tier > sglang::cpu_experts::detect_isa(Isa::Vbmi, nullptr)) break;
            sglang::cpu_experts::keep_warm<Isa::Vbmi>(tier, {}, team, &word, 6, now_ns() + 2000000);
            sglang::cpu_experts::keep_warm<Isa::Vbmi>(tier, {}, team, &word, 5, now_ns() + 1000000000);
        }
        ok("keep_warm_returns_when_the_word_moves");
    }

    std::printf("all ok\n");
    return 0;
}
```

In `test/registered/unit/kernels/test_cpu_experts_common.py`, replace the module docstring's second paragraph with "``cpu_experts_common_check.cpp`` drives a toy quant through ``ExpertForward`` as a ``CpuExpertKernel`` and prints ``ok <check>`` per contract it holds." and `CHECKS` with:

```python
CHECKS = (
    "make_layer_then_forward_overwrites_and_accumulates",
    "params_and_slabs_are_validated",
    "a_layer_of_another_kernel_is_refused",
    "routes_are_validated",
    "forwards_run_at_once",
    "isa_cap_env_lowers_the_tier",
    "cores_are_validated_and_bound_the_team",
    "each_calls_team_runs_on_its_own_cores",
    "a_failed_pin_throws_runtime_error_and_leaves_out_untouched",
    "keep_warm_returns_when_the_word_moves",
)
```

(`a_core_outside_the_callers_affinity_is_pinned` prints only on hosts with two allowed CPUs, so it is not in the required list.)

- [ ] **Step 2: Run it to verify it fails.** Commit nothing yet; on the laptop, `cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-kernel-iface && g++ -std=c++20 -fopenmp -fsyntax-only test/registered/unit/kernels/cpu_experts_common_check.cpp` fails if the laptop has a Linux GCC; otherwise this check happens on divix01 in Step 9 (the macOS toolchain cannot build the framework: `isa.hpp` refuses non-Linux). Expected failure: `'LayerSlabs' does not name a type` / `kernel.hpp: No such file`.

- [ ] **Step 3: Create `kernel.hpp`.**

```cpp
// The CPU expert kernel interface: the one C++ boundary between the expert-stream host (CpuExpertEngine) and each
// quant's CPU kernel library (EXL3's torch extension, NVFP4's tvm-ffi library). Each library keeps its own build flags;
// the host calls through the vtable. Signatures use plain structs, std::span, std::array and std::unique_ptr only
// (no std::string or std::vector), so the two sides may differ in _GLIBCXX_USE_CXX11_ABI. Not in an anonymous
// namespace: every library must name the same classes. No RTTI crosses a library (a kernel static_casts its own
// layers after the kernel() check).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace sglang::cpu_experts {
inline constexpr int kMaxSlabs = 8;

// One layer's pinned host tier: slot s of slab i starts at slabs[i] + s * slot_bytes[i]. Views only: the caller keeps
// every slab alive for the layer's lifetime.
struct LayerSlabs {
  int32_t capacity = 0, hidden = 0, intermediate = 0, activation = 0;
  float act_limit = 0;
  int32_t slab_count = 0;
  std::array<const void*, kMaxSlabs> slabs{};
  std::array<uint64_t, kMaxSlabs> slot_bytes{};
};

// One forward: `rows` token rows; row t's experts are slots[t*k+i] weighted by weights[t*k+i], -1 skipped. out (fp32
// [rows][hidden]) is overwritten, or added to when accumulate. Worker i runs on cores[i]; empty: unpinned workers.
struct ForwardCall {
  int32_t rows = 0, k = 0, threads = 1;
  const void* x = nullptr;
  const int32_t* slots = nullptr;
  const float* weights = nullptr;
  float* out = nullptr;
  bool accumulate = false;
  std::span<const int> cores;
};

class CpuExpertKernel;

// A registered layer, opaque to the host. It names the kernel that made it, so a layer passed to another kernel is
// refused rather than misread.
class CpuExpertLayer {
 public:
  virtual ~CpuExpertLayer() = default;
  const CpuExpertKernel& kernel() const { return *kernel_; }

 protected:
  explicit CpuExpertLayer(const CpuExpertKernel& k) : kernel_(&k) {}

 private:
  const CpuExpertKernel* kernel_;
};

class CpuExpertKernel {
 public:
  virtual ~CpuExpertKernel() = default;
  virtual const char* name() const noexcept = 0;
  // Validates (the old C ABI's registration checks) and stores views; throws std::invalid_argument. `params` is the quant's
  // own parameter struct, size-checked against it (empty when the quant has none).
  virtual std::unique_ptr<CpuExpertLayer> make_layer(const LayerSlabs&, std::span<const std::byte> params) const = 0;
  // Validates (today's forward checks) and runs. Throws std::invalid_argument for a bad call, std::runtime_error for a
  // failure; out is untouched when it throws. Pins the calling thread and its workers to cores, as today's engine did.
  virtual void forward(const CpuExpertLayer&, const ForwardCall&) const = 0;
  // Register-only work at the forward's vector width on `threads` workers pinned to cores, until *word != seen or
  // CLOCK_MONOTONIC reaches deadline_ns (keep_warm.hpp). Throws on a bad argument.
  virtual void keep_warm(std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
                         int64_t deadline_ns) const = 0;
};
}  // namespace sglang::cpu_experts
```

- [ ] **Step 4: Rework `team.hpp` and `keep_warm.hpp`.** In `team.hpp`: delete `last_error()`, `fail()` and `struct Engines` (they move verbatim into `cabi.hpp`, Step 6); remove the `<mutex>`, `<memory>`, `<string>`, `<cstdio>` includes no longer used and add `<span>`; change the file comment's first line to "The team every forward runs: one OpenMP team per call, the calling thread as worker 0, worker i pinned to core i of the call's cores (none: unpinned)." Then replace `call_cores`, `CallCores`, `pin` and `run_team`'s core handling with:

```cpp
// The cores of the call this thread is running (empty: unpinned), which run_team pins its workers to. ExpertForward
// sets it around a forward's dispatch (CallCores), so a quant's plan calls run_team without passing the cores down.
inline std::span<const int>& call_cores()
{
    static thread_local std::span<const int> cores;
    return cores;
}

struct CallCores
{
    explicit CallCores(std::span<const int> cores) : saved(call_cores()) { call_cores() = cores; }
    ~CallCores() { call_cores() = saved; }
    CallCores(const CallCores&) = delete;
    CallCores& operator=(const CallCores&) = delete;
    std::span<const int> saved;
};

// Throws std::invalid_argument unless every core is in [0, CPU_SETSIZE) and none repeats. Not checked against the
// caller's affinity: the engine thread may run under a mask that excludes the expert cores, and the workers pin
// themselves outside it; a core that cannot be pinned fails the call's pin (std::runtime_error).
inline void check_cores(std::span<const int> cores)
{
    for (size_t i = 0; i < cores.size(); ++i) {
        if (cores[i] < 0 || cores[i] >= CPU_SETSIZE)
            throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " is outside [0, CPU_SETSIZE)");
        for (size_t j = 0; j < i; ++j)
            if (cores[j] == cores[i])
                throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " repeats");
    }
}

// Inside a team: pins worker `worker` to cores[worker] (empty cores: no-op), setting `error` when it cannot; the caller
// checks that worker < cores.size(). Each thread remembers its last core, so a team on the same cores repins nothing.
inline void pin(int worker, std::span<const int> cores, std::atomic<int>& error)
{
    if (cores.empty()) return;
    const int core = cores[size_t(worker)];
    // ... body unchanged from today's pin() ...
}
```

In `run_template run_team`, read `const std::span<const int> cores = call_cores();`, test `!cores.empty() && size_t(threads) > cores.size()`, keep the messages, and change the pin error text to "cannot pin CPU expert worker to its core". (`<string>` stays included for `std::to_string`; add `<stdexcept>`.)

In `keep_warm.hpp`, replace the bottom function with:

```cpp
// Holds `threads` workers (the caller as worker 0, each pinned to `cores` as the forward pins them; empty: unpinned) in
// register-only work of tier min(isa, Top) until *word != seen or CLOCK_MONOTONIC reaches deadline_ns. Throws
// std::invalid_argument for no worker, no word or more threads than `cores`, std::runtime_error for a failed pin.
template <Isa Top>
void keep_warm(Isa isa, std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
               int64_t deadline_ns)
{
    if (threads < 1 || word == nullptr || (!cores.empty() && size_t(threads) > cores.size()))
        throw std::invalid_argument("CPU expert keep-warm needs a worker, a word and no more workers than its cores");
    std::atomic<int> pin_error{0};
    #pragma omp parallel num_threads(threads) shared(cores, pin_error)
    {
        pin(omp_get_thread_num(), cores, pin_error);
        keep_warm_detail::sink.fetch_add(keep_warm_loop<Top>(isa, word, seen, deadline_ns), std::memory_order_relaxed);
    }
    if (pin_error.load(std::memory_order_relaxed)) throw std::runtime_error("cannot pin CPU expert worker to its core");
}
```

(add `<span>` and `<stdexcept>`; `<vector>` is no longer needed).

In `buffer_row.hpp`, include `kernel.hpp` instead of `../cpu_experts_abi.h` and change `of` to `static MoeBufferRows of(const LayerSlabs& d)` (body unchanged: `d.slabs[i]`, `d.slot_bytes[i]` index `std::array`).

- [ ] **Step 5: Rewrite `expert_forward.hpp`.** Replace the whole file comment (today's first three lines name a registry and per-library state, which Task 9's grep must not find) with:

```cpp
// The validation and dispatch every CPU expert quant shares, generic over the quant (the Quant contract: kName,
// kSlabs, kOptionalSlabs, kMaxRoutes, kMaxRows, kTopIsa, kIsaCapEnv, kIsaReportEnv, Params, Layer, Row,
// min_slot_bytes, validate, make_layer, check_slot, dispatch, decode). ExpertForward<Quant> is the quant's
// CpuExpertKernel (kernel.hpp): each library holds one behind its accessor; it holds no layer table and takes no lock.
// A Quant's dispatch may ignore the RouteTable and read the request directly: EXL3 does, to keep its frozen
// accumulation order, so it runs the zero-weight routes that RouteTable drops.
```

Body:

```cpp
#pragma once
#include "buffer_row.hpp"
#include "isa.hpp"
#include "keep_warm.hpp"
#include "kernel.hpp"
#include "routes.hpp"
#include "team.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

template <class Quant>
class ExpertForward final : public CpuExpertKernel
{
public:
    static_assert(Quant::kSlabs >= 1 && Quant::kSlabs <= kMaxSlabs);
    static_assert(Quant::kMaxRows >= 1 && Quant::kMaxRoutes >= 1);
    using Params = typename Quant::Params;
    static_assert(std::is_trivially_copyable_v<Params>, "params cross the interface as bytes");

    // A layer this kernel made: the quant's own, tagged with the kernel.
    class Layer final : public CpuExpertLayer
    {
    public:
        Layer(const CpuExpertKernel& k, typename Quant::Layer q) : CpuExpertLayer(k), quant(std::move(q)) {}
        const typename Quant::Layer quant;
    };

    // Computed at the first call, so a test may set the cap variable before it. Then, when Quant::kIsaReportEnv (may
    // be null) is "1", prints "<kName> isa <tier>" to stderr, once.
    static Isa isa()
    {
        static const Isa isa = [] {
            const Isa detected = detect_isa(Quant::kTopIsa, Quant::kIsaCapEnv);
            report_isa(Quant::kName, detected, Quant::kIsaReportEnv);
            return detected;
        }();
        return isa;
    }

    const char* name() const noexcept override { return Quant::kName; }

    std::unique_ptr<CpuExpertLayer> make_layer(const LayerSlabs& d, std::span<const std::byte> params) const override
    {
        if (!params.empty() && params.size() != sizeof(Params))
            refuse("params hold " + std::to_string(params.size()) + " bytes, the quant's hold "
                   + std::to_string(sizeof(Params)));
        Params p{};
        if (!params.empty()) std::memcpy(&p, params.data(), sizeof(Params));
        const Params* given = params.empty() ? nullptr : &p;
        if (d.capacity < 1 || d.slab_count != Quant::kSlabs)
            refuse("a layer needs capacity >= 1 and " + std::to_string(Quant::kSlabs) + " slabs");
        if (Quant::validate(d, given) != 0) refuse("the layer's shape, activation or parameters");
        const std::array<uint64_t, Quant::kSlabs> minimum = Quant::min_slot_bytes(d, p);
        for (int i = 0; i < Quant::kSlabs; ++i) {
            if (!d.slabs[i]) {
                if (Quant::kOptionalSlabs >> i & 1u) continue;
                refuse("slab " + std::to_string(i) + " is required");
            }
            if (d.slot_bytes[i] < minimum[i] || d.slot_bytes[i] > SIZE_MAX / uint64_t(d.capacity))
                refuse("slab " + std::to_string(i) + "'s slots hold " + std::to_string(d.slot_bytes[i])
                       + " bytes, at least " + std::to_string(minimum[i]));
        }
        return std::make_unique<Layer>(*this, Quant::make_layer(d, given));
    }

    // Wraps a quant layer built without LayerSlabs (EXL3's per-expert table layers), as make_layer would; the caller
    // has validated it.
    std::unique_ptr<CpuExpertLayer> wrap(typename Quant::Layer quant) const
    {
        return std::make_unique<Layer>(*this, std::move(quant));
    }

    void forward(const CpuExpertLayer& layer, const ForwardCall& c) const override
    {
        if (&layer.kernel() != this)
            refuse(std::string("a layer of kernel ") + layer.kernel().name() + " (another library's or object's)");
        if (!c.x || !c.out || c.rows < 1 || c.rows > Quant::kMaxRows || c.k < 0 || c.k > Quant::kMaxRoutes
            || c.threads < 1 || c.threads > 4096 || (c.k && (!c.slots || !c.weights)))
            refuse("rows, k, threads or a buffer out of range");
        if (!c.cores.empty() && size_t(c.threads) > c.cores.size())
            refuse(std::to_string(c.threads) + " workers on " + std::to_string(c.cores.size()) + " cores");
        check_cores(c.cores);
        const typename Quant::Layer& l = static_cast<const Layer&>(layer).quant;
        const int capacity = l.rows.capacity;
        const size_t n = size_t(c.rows) * size_t(c.k);
        for (size_t j = 0; j < n; ++j) {
            const int32_t slot = c.slots[j];
            if (slot < -1 || slot >= capacity || !std::isfinite(c.weights[j]))
                refuse("slot " + std::to_string(slot) + " outside the layer's " + std::to_string(capacity)
                       + " or a non-finite weight");
            if (slot >= 0 && Quant::check_slot(l, slot) != 0) refuse("slot " + std::to_string(slot) + " is unusable");
        }
        const RouteTable routes = RouteTable::build(c.slots, c.weights, c.rows, c.k);
        const CallCores on_cores(c.cores);
        const int status = Quant::dispatch(l, c, routes, isa());
        if (status == 2) refuse("the forward refused its input (status 2)");
        if (status != 0)
            throw std::runtime_error(std::string(Quant::kName) + " CPU experts: forward failed (status "
                                     + std::to_string(status) + ")");
    }

    // keep_warm (keep_warm.hpp) at this quant's tier, compiling only the loops up to kTopIsa.
    void keep_warm(std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
                   int64_t deadline_ns) const override
    {
        check_cores(cores);
        ::sglang::cpu_experts::keep_warm<Quant::kTopIsa>(isa(), cores, threads, word, seen, deadline_ns);
    }

private:
    [[noreturn]] static void refuse(const std::string& why)
    {
        throw std::invalid_argument(std::string(Quant::kName) + " CPU experts: " + why);
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
```

- [ ] **Step 6: Rewrite `cabi.hpp` as the transitional shim.** It is the only place left with handles, engines and statuses; Task 9 deletes it.

```cpp
// Transitional, deleted by plan 2026-10-04-cpu-expert-kernel-interface Task 9: the six C functions of the old CPU
// expert C ABI, over a quant library's CpuExpertKernel (kernel.hpp), with the ABI's layer handles, engines, statuses
// (0 ok, 1 internal error, 2 invalid arguments, 3 a free_layer while a forward runs) and last_error().
#pragma once
#include "../cpu_experts_abi.h"
#include "kernel.hpp"
#include <cstdio>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state.
namespace {

// [last_error(), fail() and struct Engines: moved here verbatim from team.hpp, with fail()'s comment updated to
//  "it runs in the C functions' catch blocks".]

// The C ABI's layer handles: index h is layers[h]; a freed entry is reset and its index never reused. Forwards hold
// layer_mutex shared; free_layer takes it exclusively and returns 3 while a forward runs.
struct CabiLayers
{
    static inline std::vector<std::shared_ptr<const CpuExpertLayer>> layers;
    static inline std::mutex registry_mutex;
    static inline std::shared_mutex layer_mutex;

    static int64_t add(std::shared_ptr<const CpuExpertLayer> layer)
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        layers.push_back(std::move(layer));
        return int64_t(layers.size() - 1);
    }

    static std::shared_ptr<const CpuExpertLayer> lookup(int64_t handle)
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (handle < 0 || handle >= int64_t(layers.size())) return nullptr;
        return layers[size_t(handle)];
    }
};

// Runs `f`, mapping what it throws to a C status: std::invalid_argument 2, anything else 1 (printed by fail()).
template <class F>
int cabi_status(F&& f) noexcept
{
    last_error().clear();
    try {
        f();
        return 0;
    } catch (const std::invalid_argument& e) {
        try { last_error() = e.what(); } catch (...) {}
        return 2;
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("unknown exception");
    }
}

inline LayerSlabs cabi_slabs(const SglangCpuExpertsLayer& d)
{
    LayerSlabs s;
    s.capacity = d.capacity;
    s.hidden = d.hidden;
    s.intermediate = d.intermediate;
    s.activation = d.activation;
    s.act_limit = d.act_limit;
    s.slab_count = d.slab_count;
    for (int i = 0; i < SGLANG_CPU_EXPERTS_MAX_SLABS; ++i) {
        s.slabs[i] = d.slabs[i];
        s.slot_bytes[i] = d.slot_bytes[i];
    }
    return s;
}

template <class Quant>
int cabi_register(const CpuExpertKernel& kernel, const SglangCpuExpertsLayer* d, int64_t* handle) noexcept
{
    return cabi_status([&] {
        if (!d || !handle || d->abi_version != SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION)
            throw std::invalid_argument("bad layer descriptor");
        const std::span<const std::byte> params =
            d->params ? std::span<const std::byte>(static_cast<const std::byte*>(d->params), sizeof(typename Quant::Params))
                      : std::span<const std::byte>();
        *handle = CabiLayers::add(kernel.make_layer(cabi_slabs(*d), params));
    });
}

inline int cabi_free(int64_t handle) noexcept
{
    last_error().clear();
    std::unique_lock<std::shared_mutex> layer_lock(CabiLayers::layer_mutex, std::try_to_lock);
    if (!layer_lock.owns_lock()) return 3;
    std::lock_guard<std::mutex> lock(CabiLayers::registry_mutex);
    if (handle < 0 || handle >= int64_t(CabiLayers::layers.size()) || !CabiLayers::layers[size_t(handle)]) return 2;
    CabiLayers::layers[size_t(handle)].reset();
    return 0;
}

inline int cabi_forward(const CpuExpertKernel& kernel, const SglangCpuExpertsForward* call) noexcept
{
    return cabi_status([&] {
        if (!call || call->abi_version != SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION
            || (call->accumulate != 0 && call->accumulate != 1))
            throw std::invalid_argument("bad forward call");
        bool found = false;
        const Engines::Cores cores = Engines::find(call->engine, &found);
        if (!found) throw std::invalid_argument("unknown engine");
        const std::shared_lock<std::shared_mutex> lock(CabiLayers::layer_mutex);
        const std::shared_ptr<const CpuExpertLayer> layer = CabiLayers::lookup(call->layer);
        if (!layer) throw std::invalid_argument("unknown layer handle");
        ForwardCall c;
        c.rows = call->rows;
        c.k = call->k;
        c.threads = call->threads;
        c.x = call->x;
        c.slots = call->slots;
        c.weights = call->weights;
        c.out = call->out;
        c.accumulate = call->accumulate == 1;
        if (cores) c.cores = *cores;
        kernel.forward(*layer, c);
    });
}

inline int cabi_keep_warm(const CpuExpertKernel& kernel, int64_t engine, int32_t threads, const uint32_t* word,
                          uint32_t seen, int64_t deadline_ns) noexcept
{
    return cabi_status([&] {
        bool found = false;
        const Engines::Cores cores = Engines::find(engine, &found);
        if (!found) throw std::invalid_argument("unknown engine");
        kernel.keep_warm(cores ? std::span<const int>(*cores) : std::span<const int>(), threads, word, seen, deadline_ns);
    });
}

}  // namespace
}  // namespace sglang::cpu_experts

#define SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant, accessor)                                                       \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_register_layer(              \
        const SglangCpuExpertsLayer* d, int64_t* handle) noexcept                                                   \
    { return ::sglang::cpu_experts::cabi_register<Quant>(accessor(), d, handle); }                                  \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_free_layer(                  \
        int64_t handle) noexcept                                                                                    \
    { return ::sglang::cpu_experts::cabi_free(handle); }                                                            \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_forward(                     \
        const SglangCpuExpertsForward* call) noexcept                                                               \
    { return ::sglang::cpu_experts::cabi_forward(accessor(), call); }                                               \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_keep_warm(                   \
        int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept         \
    { return ::sglang::cpu_experts::cabi_keep_warm(accessor(), engine, threads, word, seen, deadline_ns); }         \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_engine_create(               \
        const int32_t* cores, int32_t n, int64_t* engine) noexcept                                                  \
    { return ::sglang::cpu_experts::Engines::create(cores, n, engine); }                                            \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_engine_free(                 \
        int64_t engine) noexcept                                                                                    \
    { return ::sglang::cpu_experts::Engines::destroy(engine); }
```

- [ ] **Step 7: Port the three quants (mechanical).** In `cpu_experts_common_toy.hpp`, make the quant's name overridable so two toy libraries can be told apart (Task 2): before `namespace toy`, `#ifndef TOY_NAME` / `#define TOY_NAME "toy"` / `#endif`, and `static constexpr const char* kName = TOY_NAME;`. In each of `cpu_experts_common_toy.hpp`, `exl3/optimized/quant.hpp`, `nvfp4/optimized/quant.hpp`: replace `SglangCpuExpertsLayer` with `LayerSlabs` and `SglangCpuExpertsForward` with `ForwardCall` in `min_slot_bytes`, `validate`, `make_layer` and `dispatch` (bodies unchanged: `d.slabs[kUpAlpha]` indexes `std::array`; `c.accumulate != 0` compiles on a `bool`). Change `exl3/optimized/forward_plan.hpp:897` and `nvfp4/optimized/forward_plan.hpp:130` and `nvfp4/optimized/kernel.cpp:23` the same way. Fix the NVFP4 `static_assert` message to "SlabName indexes LayerSlabs::slabs as cpu_experts_cabi.h orders them". The toy header now includes only `expert_forward.hpp` (drop `cabi.hpp`) and, after its anonymous namespace, declares the accessor:

```cpp
#ifndef TOY_KERNEL
#define TOY_KERNEL toy_kernel
#endif
namespace toy {
// The toy kernel of this library or harness: each defines it (test_cpu_experts_common.py builds libraries with
// distinct names, so a harness links two side by side).
const sglang::cpu_experts::CpuExpertKernel& TOY_KERNEL();
}  // namespace toy
```

- [ ] **Step 8: The accessors.** Create `exl3/optimized/kernel.h`:

```cpp
// The EXL3 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the torch extension's
// sglang_exl3_cpu::kernel_address op hands to Python, and the one the benches link.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"

namespace sglang::exl3_cpu {
// Hidden: another library defining it in the same process (the extension and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel();
}  // namespace sglang::exl3_cpu
```

and `nvfp4/optimized/kernel.h` the same with "NVFP4", `nvfp4_cpu_kernel()`, "the tvm-ffi export nvfp4_cpu_kernel_address" and `namespace sglang::nvfp4_cpu`. In `exl3/optimized/kernel.cpp`, include `kernel.h`, and after `#include "forward_plan.hpp"`:

```cpp
namespace sglang::exl3_cpu {
const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel()
{
    static const ::sglang::cpu_experts::ExpertForward<Exl3Quant> kernel{};
    return kernel;
}
}  // namespace sglang::exl3_cpu

SGLANG_CPU_EXPERTS_DEFINE_CABI(exl3, ::sglang::exl3_cpu::Exl3Quant, ::sglang::exl3_cpu::exl3_cpu_kernel)
```

and in its torch wrappers: `Exl3Forward::isa()` stays (a static); `exl3_moe_cpu_make_layer` ends with

```cpp
    const auto& kernel = static_cast<const Exl3Forward&>(::sglang::exl3_cpu::exl3_cpu_kernel());
    return ::sglang::cpu_experts::CabiLayers::add(kernel.wrap(std::move(layer)));
```

(drop its `std::make_shared<const Exl3Quant::Layer>` and registry lines); `exl3_moe_cpu_free_layer` calls `sglang_exl3_cpu_experts_free_layer(handle)`; `exl3_moe_cpu_forward_raw` calls `sglang_exl3_cpu_experts_forward(&call)`. In `nvfp4/optimized/kernel.cpp`, include `kernel.h`, define `nvfp4_cpu_kernel()` the same way after `Nvfp4Quant::dispatch` (inside `namespace sglang::nvfp4_cpu`, outside the anonymous one), and change the macro line to `SGLANG_CPU_EXPERTS_DEFINE_CABI(nvfp4, ::sglang::nvfp4_cpu::Nvfp4Quant, ::sglang::nvfp4_cpu::nvfp4_cpu_kernel)`. In `cpu_experts_common_toy_lib.cpp`:

```cpp
// The toy quant as a shared library, for test_cpu_experts_common.py's per-library and portable-build checks.
#include "cpu_experts_common_toy.hpp"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/cabi.hpp"

namespace toy {
__attribute__((visibility("default"))) const sglang::cpu_experts::CpuExpertKernel& TOY_KERNEL()
{
    static const sglang::cpu_experts::ExpertForward<ToyQuant> kernel{};
    return kernel;
}
}  // namespace toy

SGLANG_CPU_EXPERTS_DEFINE_CABI(toy, toy::ToyQuant, toy::TOY_KERNEL)
```

- [ ] **Step 8b: GCC 15 for the host JIT in the EXL3 gate.** In `test/manual/dsv41/run_exl3_cpu_forward_checks.sh`, after the line `export SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_EXL3_CPU_CXX=$GXX CUDA_HOME=/usr/local/cuda-13.4`, add:

```bash
export CXX=$GXX  # the host module's JIT build (kernel_layer/kernel_forward from Task 6): the kernels' GCC 15
```

- [ ] **Step 9: Commit, sync, run.**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/{kernel.hpp,expert_forward.hpp,team.hpp,keep_warm.hpp,buffer_row.hpp,cabi.hpp} \
  python/sglang/kernels/jit/csrc/exl3/optimized/{kernel.h,quant.hpp,forward_plan.hpp,kernel.cpp} \
  python/sglang/kernels/jit/csrc/nvfp4/optimized/{kernel.h,quant.hpp,forward_plan.hpp,kernel.cpp} \
  test/registered/unit/kernels/{cpu_experts_common_toy.hpp,cpu_experts_common_toy_lib.cpp,cpu_experts_common_check.cpp,test_cpu_experts_common.py} \
  test/manual/dsv41/run_exl3_cpu_forward_checks.sh
git commit -m "feat(cpu-experts): CpuExpertKernel interface; ExpertForward is a kernel, the C ABI a shim over it" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FVzWUFXT8SqY4pbyJn21Ft"
```

`SYNC`, then `RUN_CPU test/registered/unit/kernels/test_cpu_experts_common.py test/registered/unit/kernels/test_cpu_experts_abi.py test/registered/unit/kernels/test_nvfp4_cpu_build.py test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_cpu_experts_layout.py`. Expected: all pass (the plain and asan-ubsan harness runs report every `CHECKS` name). Then `NVFP4_AB t1` (both PASS) and `CPU_CHECKS t1` (`ALL GREEN (check)`). A red harness check is fixed with a follow-up commit, never an amend.

---

### Task 2: The interface across two libraries

**Files:**
- Create: `test/registered/unit/kernels/cpu_experts_cross_library_check.cpp`
- Modify: `test/registered/unit/kernels/cpu_experts_common_toy_lib.cpp` (drop the C ABI), `test/registered/unit/kernels/test_cpu_experts_common.py`

**Interfaces:**
- Consumes: `CpuExpertKernel`, `LayerSlabs`, `ForwardCall` (Task 1); `toy::TOY_KERNEL()` (Task 1).
- Produces: nothing later tasks call.

- [ ] **Step 1: Write the failing test.** Create the harness:

```cpp
// Two toy quant libraries in one process, built as the expert-stream host module is (-fvisibility=hidden), each with
// its own accessor (TOY_KERNEL): the CpuExpertKernel interface crosses the .so boundary. Built by
// test_cpu_experts_common.py.
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace toy {
const sglang::cpu_experts::CpuExpertKernel& toy_kernel_a();
const sglang::cpu_experts::CpuExpertKernel& toy_kernel_b();
}  // namespace toy

int main()
{
    using namespace sglang::cpu_experts;
    const CpuExpertKernel& a = toy::toy_kernel_a();
    const CpuExpertKernel& b = toy::toy_kernel_b();
    CHECK(&a != &b && std::strcmp(a.name(), "toy_a") == 0 && std::strcmp(b.name(), "toy_b") == 0);

    constexpr int hidden = 16, capacity = 2;
    std::vector<float> slab(hidden * capacity);
    for (int i = 0; i < hidden * capacity; ++i) slab[i] = float(i);
    const float scale = 1.0f;
    LayerSlabs d;
    d.capacity = capacity;
    d.hidden = hidden;
    d.intermediate = hidden;
    d.slab_count = 1;
    d.slabs[0] = slab.data();
    d.slot_bytes[0] = hidden * 4;
    const auto params = std::as_bytes(std::span<const float>(&scale, 1));
    std::unique_ptr<CpuExpertLayer> layer = a.make_layer(d, params);
    CHECK(&layer->kernel() == &a);

    std::vector<uint16_t> x(hidden);
    const int32_t slot = 1;
    const float weight = 1.0f;
    std::vector<float> out(hidden, 7.0f);
    ForwardCall c;
    c.rows = 1;
    c.k = 1;
    c.threads = 1;
    c.x = x.data();
    c.slots = &slot;
    c.weights = &weight;
    c.out = out.data();
    a.forward(*layer, c);
    for (int h = 0; h < hidden; ++h) CHECK(out[h] == float(hidden + h));

    // Library b refuses library a's layer: its std::invalid_argument reaches this executable's catch.
    std::fill(out.begin(), out.end(), 7.0f);
    bool refused = false;
    try {
        b.forward(*layer, c);
    } catch (const std::invalid_argument& e) {
        // b's refusal names both kernels: itself ("toy_b CPU experts: ...") and the layer's ("kernel toy_a").
        refused = std::strstr(e.what(), "toy_b") != nullptr && std::strstr(e.what(), "kernel toy_a") != nullptr;
    }
    CHECK(refused);
    for (float v : out) CHECK(v == 7.0f);
    refused = false;
    try {
        b.make_layer(d, {});
    } catch (const std::exception&) {
        refused = true;
    }
    CHECK(refused);
    layer.reset();  // the deleting destructor runs in library a
    std::printf("ok cross_library\n");
    return 0;
}
```

In `test_cpu_experts_common.py`, delete `_Layer`, `_Forward` and `test_each_library_keeps_its_own_registry_and_engines` (and the now-unused `ctypes` import), and add:

```python
CROSS = Path(__file__).resolve().parent / "cpu_experts_cross_library_check.cpp"
# As the expert-stream host module builds (expert_stream_transport._host_module_cached): the interface must cross a
# hidden-visibility library.
HIDDEN = ["-fvisibility=hidden", "-fvisibility-inlines-hidden"]


def test_two_libraries_refuse_each_others_layers_across_the_so_boundary(tmp_path):
    for name in ("a", "b"):
        _build_toy_library(
            tmp_path / f"libtoy_{name}.so", *HIDDEN, f"-DTOY_KERNEL=toy_kernel_{name}", f'-DTOY_NAME="toy_{name}"'
        )
    exe = tmp_path / "cross_library"
    subprocess.run(
        [CXX, *CXX_FLAGS, str(CROSS), f"-L{tmp_path}", "-ltoy_a", "-ltoy_b", f"-Wl,-rpath,{tmp_path}", "-o", str(exe)],
        check=True,
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "ok cross_library" in result.stdout, result.stdout + result.stderr
```

Also update the module docstring: "... and ``cpu_experts_cross_library_check.cpp`` links two toy libraries and passes a layer between them."

- [ ] **Step 2: Run it.** Commit the test (`test(cpu-experts): the kernel interface crosses two hidden-visibility libraries`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_cpu_experts_common.py`. This test pins what Task 1 built, so it is expected to PASS already; a failure here is a real cross-library defect (a hidden typeinfo the catch cannot match, an interposed accessor) and is fixed in Task 1's files with a fix commit before going on.
- [ ] **Step 3: Implement.** Drop the C ABI from `cpu_experts_common_toy_lib.cpp` (remove the `cabi.hpp` include and the macro line; update its comment to "The toy quant as a shared library, for test_cpu_experts_common.py: its kernel behind the accessor TOY_KERNEL, and the portable-build check.").
- [ ] **Step 4: Run.** `RUN_CPU test/registered/unit/kernels/test_cpu_experts_common.py` -- expected all pass, including `test_a_scalar_quant_library_uses_no_avx_registers` (the library still instantiates the kernel's vector code).
- [ ] **Step 5: Commit** (`git add test/registered/unit/kernels/cpu_experts_common_toy_lib.cpp`; message `test(cpu-experts): the toy library exports no C ABI`), `SYNC`.

---

### Task 3: Quant bindings hand out their kernel; traits describe their layers

**Files:**
- Create: `python/sglang/kernels/jit/csrc/exl3/optimized/torch_ops.cpp`
- Create: `python/sglang/kernels/jit/csrc/nvfp4/optimized/ffi.cpp`
- Create: `python/sglang/srt/layers/moe/cpu_experts/trait.py`
- Modify: `python/sglang/srt/layers/quantization/exl3/ext.py:31-36,155-168`
- Modify: `python/sglang/srt/layers/quantization/nvfp4/build.py`, `python/sglang/srt/layers/quantization/nvfp4/ext.py`
- Modify: `python/sglang/srt/layers/quantization/exl3/schemes/exl3_cpu_experts.py`, `python/sglang/srt/layers/quantization/nvfp4/schemes/nvfp4_cpu_experts.py`
- Modify: `python/sglang/srt/layers/moe/cpu_experts/pool.py` (Protocol gains two methods)
- Test: `test/registered/unit/kernels/test_nvfp4_cpu_build.py`, `test/registered/unit/kernels/test_nvfp4_cpu_experts.py`, `test/registered/unit/kernels/test_cpu_expert_pool.py`, `test/manual/dsv41/test_cpu_expert_engines_exl3.py`

**Interfaces:**
- Consumes: `exl3_cpu_kernel()`, `nvfp4_cpu_kernel()` (Task 1).
- Produces:
  - torch op `torch.ops.sglang_exl3_cpu.kernel_address() -> int` (optimized extension only).
  - tvm-ffi export `nvfp4_cpu_kernel_address() -> int` in the NVFP4 shared library.
  - `nvfp4/build.py`: `tvm_ffi_flags() -> list[str]`; `build(output, *, cxx, main=None, extra_flags=())` links `ffi.cpp` and tvm-ffi when `main is None`.
  - `nvfp4/ext.py`: `nvfp4_cpu_library_path(build_dir: Optional[str] = None) -> Path`, `nvfp4_cpu_module(build_dir: Optional[str] = None) -> tvm_ffi.Module` (cached), `nvfp4_cpu_kernel_address(build_dir: Optional[str] = None) -> int`; `nvfp4_cpu_library()` (ctypes) stays until Task 7.
  - `cpu_experts/trait.py`: `@dataclass(frozen=True) class CpuExpertLayerSpec(capacity: int, hidden: int, intermediate: int, act_limit: float, slabs: tuple[tuple[int, int], ...], params: bytes, keep: tuple = (), activation: int = 0)`.
  - Traits: `kernel_address() -> int`, `layer_spec(slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec`. `Nvfp4CpuQuantTrait.__init__` gains `module=None` (a loaded tvm-ffi module; default `nvfp4_cpu_module()`).

- [ ] **Step 1: Write the failing tests.** In `test_nvfp4_cpu_build.py` add:

```python
def test_the_library_hands_out_its_kernel_address(tmp_path):
    from tvm_ffi import load_module

    module = load_module(str(_build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)))
    address = int(module.nvfp4_cpu_kernel_address())
    assert address != 0 and int(module.nvfp4_cpu_kernel_address()) == address


def test_the_module_is_loaded_once_and_kept(tmp_path):
    """Review Focus 1: the host holds the kernel's address, so the module that owns it must outlive every host: the
    loader is cached for the process and hands out one module."""
    from sglang.srt.layers.quantization.nvfp4 import ext as nvfp4_cpu_ext

    nvfp4_cpu_ext.nvfp4_cpu_module.cache_clear()
    first = nvfp4_cpu_ext.nvfp4_cpu_module(str(tmp_path))
    assert nvfp4_cpu_ext.nvfp4_cpu_module(str(tmp_path)) is first
    assert nvfp4_cpu_ext.nvfp4_cpu_kernel_address(str(tmp_path)) == int(first.nvfp4_cpu_kernel_address())
```

In `test_nvfp4_cpu_experts.py` add:

```python
def test_the_scheme_describes_a_layer_for_make_layer(built):
    import struct

    import torch
    from tvm_ffi import load_module

    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, w13_layout=2, inv_input_scale13=0.5,
                               module=load_module(str(built)))
    slabs = {
        "w13": torch.zeros((2, 80 * 80), dtype=torch.uint8),
        "w2": torch.zeros((2, 80 * 80 // 2), dtype=torch.uint8),
        "sf13": torch.zeros((2, 256 * 8), dtype=torch.uint8),
        "sf2": torch.zeros((2, 128 * 8), dtype=torch.uint8),
        "gate_alpha": torch.ones(2, 1),
        "down_alpha": torch.ones(2, 1),
    }
    spec = trait.layer_spec(slabs, capacity=2)
    assert (spec.capacity, spec.hidden, spec.intermediate, spec.act_limit) == (2, 80, 80, 0.0)
    names = ("w13", "w2", "sf13", "sf2", "gate_alpha", "down_alpha")
    assert spec.slabs == tuple((slabs[n].data_ptr(), slabs[n][0].numel() * slabs[n].element_size()) for n in names) + ((0, 0),)
    assert spec.params == struct.pack("<iff", 2, 0.5, 1.0)
    assert all(any(k is slabs[n] for k in spec.keep) for n in names)
    assert trait.kernel_address() != 0
    with pytest.raises(ValueError, match="w2"):
        trait.layer_spec({**slabs, "w2": slabs["w2"][:1]}, capacity=2)
```

In `test_cpu_expert_pool.py` add, beside the EXL3 trait tests (it reuses that file's `FakeExt`, `_exl3_slabs`, `CAP`, `H`, `INTER`):

```python
@pytest.mark.parametrize("tier_layout", [False, True], ids=["flat_w2", "tier_w2"])
def test_exl3_trait_describes_the_six_slabs_for_make_layer(tier_layout):
    """layer_spec gives the kernel's make_layer what the C ABI's registration took: each slab's base and row size in
    EXL3_STREAMED_NAMES order, the shape, the clamp and SglangExl3CpuParams {bits, swizzled}."""
    import struct

    slabs = _exl3_slabs()
    if tier_layout:
        slabs = {n: (t.unsqueeze(1) if n.startswith("w2_") else t) for n, t in slabs.items()}
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0, swizzled=True)
    spec = trait.layer_spec(slabs, CAP)
    names = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
    trellis = H * INTER * 3 // 8  # bytes of one 3-bit [k/16, n/16, 48] trellis
    row_bytes = [2 * trellis, 2 * 2 * H, 2 * 2 * INTER, trellis, 2 * INTER, 2 * H]  # quant.hpp's SlabRowBytes
    assert spec.slabs == tuple(zip([slabs[n].data_ptr() for n in names], row_bytes))
    assert (spec.capacity, spec.hidden, spec.intermediate, spec.act_limit, spec.activation) == (CAP, H, INTER, 10.0, 0)
    assert spec.params == struct.pack("<ii", 3, 1)
    assert all(any(k is slabs[n] for k in spec.keep) for n in names)
```

In `test/manual/dsv41/test_cpu_expert_engines_exl3.py` add (`_kernel()` is that file's helper returning `(Exl3CpuQuantTrait, cores)`; it takes no arguments):

```python
def test_the_extension_hands_out_one_kernel_address():
    trait, _ = _kernel()
    address = trait.kernel_address()
    assert address != 0 and trait.kernel_address() == address
```

- [ ] **Step 2: Run to verify they fail.** Commit the tests (`test(cpu-experts): kernel addresses and layer specs`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_nvfp4_cpu_build.py test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_pool.py`. Expected: the four new tests fail (`has no attribute 'nvfp4_cpu_kernel_address'`, `nvfp4_cpu_module`, `unexpected keyword argument 'module'`, `has no attribute 'layer_spec'`).

- [ ] **Step 3: The bindings.** Create `exl3/optimized/torch_ops.cpp`:

```cpp
// The EXL3 CPU expert kernel's address as the torch op sglang_exl3_cpu::kernel_address: how Python hands the kernel to
// the expert-stream host (enable_cpu_experts) and to the host's test exports. Built into the optimized extension only
// (quantization/exl3/ext.py), not into benches or the standalone build.
#include "kernel.h"
#include <torch/library.h>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::exl3_cpu::exl3_cpu_kernel()); }
}  // namespace

TORCH_LIBRARY(sglang_exl3_cpu, m) { m.def("kernel_address() -> int", &kernel_address); }
```

In `exl3/ext.py` add `OPTIMIZED_TORCH_OPS = os.path.join(_CSRC, "optimized", "torch_ops.cpp")` beside `OPTIMIZED_CPU_KERNEL`, and in `exl3_ext()` pass `sources=extension_sources(...) + ([OPTIMIZED_TORCH_OPS] if optimized else [])`. Create `nvfp4/optimized/ffi.cpp`:

```cpp
// The NVFP4 CPU expert library's one tvm-ffi export: its kernel's address (nvfp4_cpu_kernel), which Python hands to the
// expert-stream host. Linked into the shared library only; a harness executable calls nvfp4_cpu_kernel() itself.
#include "kernel.h"
#include <tvm/ffi/function.h>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::nvfp4_cpu::nvfp4_cpu_kernel()); }
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cpu_kernel_address, kernel_address);
```

Confirm the macro's header first: `ssh divix01 'grep -rl "define TVM_FFI_DLL_EXPORT_TYPED_FUNC" $(/data/models/slang/.venv/bin/python -c "import tvm_ffi.libinfo as l; print(l.find_include_path())")'` and include that header if it is not `tvm/ffi/function.h`.

In `nvfp4/build.py`, document in the module docstring that the shared library also links tvm-ffi for its one export, and add:

```python
def tvm_ffi_flags() -> list[str]:
    """Include and link flags for tvm-ffi, located as the JIT locates them (jit/utils/compile/toolchain.tvm_ffi_paths)."""
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi

    lib = Path(find_libtvm_ffi())
    includes = dict.fromkeys([find_include_path(), find_dlpack_include_path()])
    return [f"-I{p}" for p in includes] + [f"-L{lib.parent}", f"-l{lib.stem.removeprefix('lib')}"]
```

and in `build()`: `sources = [str(SRC / "kernel.cpp")] + ([str(Path(main).resolve())] if main else [str(SRC / "ffi.cpp")])` and `link = [] if main else ["-shared", *tvm_ffi_flags()]` (the includes ride in `link`, which sits in the same compile-and-link command).

In `nvfp4/ext.py`: add `import tvm_ffi` to the digest (`digest.update(tvm_ffi.__version__.encode())`, imported inside `library_path`), update the module docstring ("... loaded with tvm-ffi (its one export hands out the kernel's address) and, until the C ABI is gone, with ctypes"), and refactor:

```python
def nvfp4_cpu_library_path(build_dir: Optional[str] = None) -> Path:
    """The built library's path (built here first if needed)."""
    cxx = os.environ.get("CXX", "g++")
    root = Path(os.path.expanduser(build_dir or _DEFAULT_BUILD_DIR))
    root.mkdir(parents=True, exist_ok=True)
    path = library_path(root, cxx)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.so")
            _build_module().build(partial, cxx=cxx)
            os.replace(partial, path)
    return path


@functools.cache
def nvfp4_cpu_module(build_dir: Optional[str] = None):
    """The library as a tvm-ffi module. Cached for the process: a host holds its kernel's address, so the module is
    never unloaded."""
    from tvm_ffi import load_module

    return load_module(str(nvfp4_cpu_library_path(build_dir)))


def nvfp4_cpu_kernel_address(build_dir: Optional[str] = None) -> int:
    """The address of the library's CpuExpertKernel, for ExpertStreamHost.enable_cpu_experts."""
    return int(nvfp4_cpu_module(build_dir).nvfp4_cpu_kernel_address())


@functools.cache
def nvfp4_cpu_library(build_dir: Optional[str] = None) -> ctypes.CDLL:
    """The native library through ctypes, exporting the cpu_experts_cabi.h functions (until Task 9 of plan
    2026-10-04-cpu-expert-kernel-interface)."""
    return ctypes.CDLL(str(nvfp4_cpu_library_path(build_dir)))
```

- [ ] **Step 4: The layer spec and the traits.** Create `cpu_experts/trait.py`:

```python
"""What an expert format's CPU kernel gives the RAM-miss service: its kernel's address and each layer's slabs.

The native half is ``expert_stream/host/cpu_experts/kernel.hpp``: the host calls the kernel's ``make_layer`` with a
``CpuExpertLayerSpec`` (``ExpertStreamHost.set_cpu_layer``) and runs its forwards on the CPU expert threads.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CpuExpertLayerSpec:
    """One layer's pinned host tier as the kernel's ``make_layer`` reads it (``LayerSlabs``), and the quant's parameter
    bytes.

    ``slabs`` holds one ``(address, slot bytes)`` pair per slab in the quant's order, ``(0, 0)`` for an absent optional
    slab; slot ``s`` of slab ``i`` starts at ``address + s * slot_bytes``. ``keep`` holds the tensors those addresses
    point into: whoever registers the spec keeps them alive for as long as the host may read them.
    """

    capacity: int
    hidden: int
    intermediate: int
    act_limit: float
    slabs: tuple[tuple[int, int], ...]
    params: bytes
    keep: tuple = ()
    activation: int = 0
```

In `cpu_experts/pool.py`'s `CpuExpertQuantTrait` Protocol add:

```python
    def kernel_address(self) -> int:
        """The address of the format's ``CpuExpertKernel`` (``expert_stream/host/cpu_experts/kernel.hpp``)."""
        ...

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> "CpuExpertLayerSpec":
        """One layer's first ``capacity`` slab rows, checked, as the kernel's ``make_layer`` takes them."""
        ...
```

In `Exl3CpuQuantTrait`: move the checks at the top of `register_layer` (the `act_limit` check through the slab loop) into `def _dims(self, slabs, capacity) -> tuple[int, int, int]` returning `(hidden, intermediate, bits)`; `register_layer` calls it. Add:

```python
    def kernel_address(self) -> int:
        """The address of the extension's EXL3 CpuExpertKernel (its torch op ``sglang_exl3_cpu::kernel_address``)."""
        try:
            return int(torch.ops.sglang_exl3_cpu.kernel_address())
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                f"the EXL3 extension {self.ext.__file__} has no sglang_exl3_cpu::kernel_address: CPU experts need the "
                "optimized CPU kernel, which SGLANG_DSV41_CPU_EXPERTS=1 builds (csrc/exl3/optimized)"
            ) from error

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """The layer's six slabs by base pointer and row size, and ``SglangExl3CpuParams``' bytes ({bits, swizzled})."""
        hidden, intermediate, bits = self._dims(slabs, capacity)
        views = [slabs[name] for name in self.slab_names]
        return CpuExpertLayerSpec(
            capacity=capacity,
            hidden=hidden,
            intermediate=intermediate,
            act_limit=float(self.act_limit),
            # One slot's row: the slab is contiguous (checked), so this is its stride(0), which PyTorch does not keep
            # meaningful for a one-slot slab.
            slabs=tuple((v.data_ptr(), v[0].numel() * v.element_size()) for v in views),
            params=struct.pack("<ii", bits, int(self.swizzled)),
            keep=tuple(views),
        )
```

In `Nvfp4CpuQuantTrait`: add the `module=None` keyword (store it; `kernel_address()` uses `self.module` when given, else `nvfp4_cpu_kernel_address()`); make the ctypes library lazy, so a trait built with `module=` never builds or loads the default-directory library (until Task 7 deletes it): `__init__` stores `self._library = library` instead of calling `nvfp4_cpu_library()`, and

```python
    @property
    def library(self) -> ctypes.CDLL:
        """The kernel's C ABI through ctypes, loaded on first use (only register_layer, forward, free_layer and the
        native_* methods use it)."""
        if self._library is None:
            from sglang.srt.layers.quantization.nvfp4.ext import nvfp4_cpu_library

            self._library = nvfp4_cpu_library()
        return self._library
```

(the methods keep reading `self.library`); move `register_layer`'s checks into `def _checked(self, slabs, capacity) -> tuple[str, ...]` returning the slab names present (with `up_alpha` when given); and add:

```python
    def kernel_address(self) -> int:
        """The address of the library's NVFP4 CpuExpertKernel (its tvm-ffi export ``nvfp4_cpu_kernel_address``)."""
        if self.module is not None:
            return int(self.module.nvfp4_cpu_kernel_address())
        from sglang.srt.layers.quantization.nvfp4.ext import nvfp4_cpu_kernel_address

        return nvfp4_cpu_kernel_address()

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """The seven slabs (up_alpha (0, 0) when absent) and ``SglangNvfp4CpuParams``' bytes."""
        names = self._checked(slabs, capacity)
        views = [slabs[name] for name in names]
        pairs = [(v.data_ptr(), v[0].numel() * v.element_size()) for v in views]
        return CpuExpertLayerSpec(
            capacity=capacity,
            hidden=self.hidden,
            intermediate=self.intermediate,
            act_limit=float(self.act_limit),
            slabs=tuple(pairs + [(0, 0)] * (len(self.slab_names) + 1 - len(pairs))),
            params=struct.pack("<iff", self.w13_layout, self.inv_input_scale13, self.inv_input_scale2),
            keep=tuple(views),
        )
```

`_checked` raises `ValueError` naming the slab (the existing message `NVFP4 slab {name} {shape} is not {capacity} contiguous CPU rows`) and the activation-limit `ValueError`. Both traits import `struct` and `CpuExpertLayerSpec` from `sglang.srt.layers.moe.cpu_experts.trait`.

- [ ] **Step 5: Run.** Commit (`feat(cpu-experts): quant bindings hand out their kernel; traits describe layers for make_layer`, staging the files of this task by name), `SYNC`, rerun Step 2's selection -- expected all pass. `NVFP4_AB t3` (both PASS: the library build changed). `RUN_EXT test/manual/dsv41/test_cpu_expert_engines_exl3.py` -- expected pass (first run rebuilds the extension with `torch_ops.cpp`). `CPU_CHECKS t3` -- `ALL GREEN (check)`.

---

### Task 4: The host runs CPU lanes through the kernel

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (CPU-experts section `:667-760`, `calibrate_cpu_split`'s `eligible` use is unchanged, members `:1932-1971`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h:384-450,762-763`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h:1-15,642-706,941-946`
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`TEST_ONLY_EXPORTS`, `enable_cpu_experts`, `set_cpu_layer`, the fake helpers)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py`
- Modify: `python/sglang/test/dsv41_ram_miss_fixtures.py` (add `fake_cpu_layer`)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py`, `test_exl3_ram_miss_numa_completion.py`, `test_cpu_expert_keep_warm.py`, `test_exl3_cpu_split_calibration.py`, `test_cpu_expert_pool.py` (service fakes), `test_expert_stream_build_variants.py`; `test/manual/dsv41/test_cpu_split_calibration_cuda.py`, `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`

**Interfaces:**
- Consumes: `CpuExpertKernel`, `CpuExpertLayer`, `LayerSlabs`, `ForwardCall` (Task 1); `CpuExpertLayerSpec`, `trait.kernel_address()`, `trait.layer_spec()` (Task 3).
- Produces (C++, `sglang::expert_stream`):
  - `class CpuExpertLayers { explicit CpuExpertLayers(int64_t rows); int64_t rows() const; bool set(int64_t row, std::unique_ptr<cpu_experts::CpuExpertLayer>); const cpu_experts::CpuExpertLayer* get(int64_t row) const; }`
  - `struct CpuExpertConfig { const cpu_experts::CpuExpertKernel* kernel; const CpuExpertLayers* layers; x_base; x_stride; out_base; out_stride; out_part_stride; hidden; threads; cores; spin_ns; keep_warm_ns; }` (no `forward`, `keep_warm`, `engine`, `rows`).
  - `RamTier::set_cpu_layer(int64_t row, std::unique_ptr<cpu_experts::CpuExpertLayer> layer)`, `RamTier::make_cpu_layer(int64_t row, const cpu_experts::LayerSlabs&, std::span<const std::byte> params)`, `RamTier::cpu_kernel() const -> const cpu_experts::CpuExpertKernel*`.
  - FFI: `expert_stream_enable_cpu_experts(handle, group, kernel, split, cores, x_rows, out_rows, hidden, parts, threads, spin_ns, keep_warm_ns)`; `expert_stream_set_cpu_layer(handle, row, slabs int64[n,2], capacity, hidden, intermediate, activation, act_limit double, params uint8[m])`; test-only `expert_stream_test_kernel_address(ns_per_expert, fail, zero) -> int64`, `expert_stream_test_kernel_calls(out float64[n, 5 + 2 * kLanes]) -> int64`, `expert_stream_test_kernel_hold(core, on)`, `expert_stream_test_keep_warm_calls() -> int64`, `expert_stream_test_keep_warm_core() -> int64`.
- Produces (Python): `ExpertStreamHost.enable_cpu_experts(kernel: int, split, cores, x_rows, out_rows, *, threads, group=0, spin_us=50_000, keep_warm_us=0)`; `ExpertStreamHost.set_cpu_layer(row: int, spec: CpuExpertLayerSpec)`; `ExpertStreamHost.test_kernel_address(ns_per_expert: int = 0, *, fail: int = 0, zero: bool = False) -> int`; `test_kernel_calls() -> list[dict]` (keys `core`, `affinity`, `threads`, `accumulate`, `slots`, `weights`); `test_kernel_hold(core: int, on: bool = True)`; `test_keep_warm_calls() -> int`; `test_keep_warm_core() -> int`; `dsv41_ram_miss_fixtures.fake_cpu_layer(hidden: int = 8) -> CpuExpertLayerSpec`. Removed: `test_forward_address`, `test_keep_warm_address`, `test_keep_warm_engine`, the `engine`/`keep_warm` arguments.

- [ ] **Step 1: Write the failing tests (port the fakes).** Every test that injected a fake now enables `host.test_kernel_address(...)` and registers rows with `host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))`. Add to `python/sglang/test/dsv41_ram_miss_fixtures.py`:

```python
def fake_cpu_layer(hidden: int = 8):
    """A layer for the instr build's fake CPU expert kernel (ExpertStreamHost.test_kernel_address), which reads only
    its hidden size."""
    from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec

    return CpuExpertLayerSpec(capacity=1, hidden=hidden, intermediate=0, act_limit=0.0, slabs=(), params=b"")
```

`test_exl3_ram_miss_cpu_experts.py` (the pattern every other file follows): delete `FakeForward`, the `CpuExpertForward` import, `ctypes` and `HANDLE`; `_host(tmp_path, *, split, copy_engine=True, parts=2, fail=0)` calls `host.enable_cpu_experts(host.test_kernel_address(fail=fail), split, _cores(), x_rows, out_rows, threads=2, spin_us=200)`; add

```python
def _calls(host):
    """The fake kernel's calls as (slots, weights, threads)."""
    return [(c["slots"], c["weights"], c["threads"]) for c in host.test_kernel_calls()]
```

and rewrite each assertion: `forward.calls == [(HANDLE, s, w, 2)]` -> `_calls(host) == [(s, w, 2)]`; `len(forward.calls)` -> `len(host.test_kernel_calls())`; `forward.accumulates` -> `[c["accumulate"] for c in host.test_kernel_calls()]`; `forward.affinities == [{_cores()[0]}]` -> `[c["affinity"] for c in host.test_kernel_calls()] == [_cores()[0]]`. The fake writes the same `out[j] = (out[j] if accumulate else j) + sum(w * (s + 1))`, so the `out_rows` assertions stay. Replace `test_the_kernel_engine_reaches_every_cpu_forward` with:

```python
def test_the_groups_cores_reach_every_cpu_forward(tmp_path):
    """The cores enable_cpu_experts takes are the ones every forward of the CPU expert thread carries, so the kernel
    runs its team there. Mutant: leave ForwardCall.cores empty in CpuExpertEngine::run -- red (core -1)."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=_split(n1=1))
    try:
        _load(sim, host, [2])
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        req = _post(sim, [2])
        assert req.kinds == [LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: len(host.test_kernel_calls()) == 1)
        assert sim.copy_wait(req)
        assert [c["core"] for c in host.test_kernel_calls()] == [_cores()[0]]
    finally:
        host.stop()


def test_a_rows_layer_is_made_once_by_the_enabled_kernel(tmp_path):
    """Review Focus 4: a second layer for a row, a layer before any group is enabled, a row past the tier and a
    malformed slab table are refused, naming why."""
    s, page, host, sim, dst, out_rows = _host(tmp_path, split=NO_SPLIT, copy_engine=False)
    try:
        with pytest.raises(Exception, match="not enabled"):
            host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    finally:
        host.stop()
    s, page, host, sim, dst, out_rows = _host(tmp_path / "on", split=NO_SPLIT)
    try:
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        with pytest.raises(Exception, match="registered once"):
            host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
        with pytest.raises(Exception, match=f"CPU expert layer for row {ROWS + 5} of {ROWS}"):
            host.set_cpu_layer(ROWS + 5, fake_cpu_layer(HIDDEN))
        bad = dataclasses.replace(fake_cpu_layer(HIDDEN), slabs=((1, 2),) * 9)
        with pytest.raises(Exception, match="has at most 8 slabs"):
            host.set_cpu_layer(0, bad)
    finally:
        host.stop()
```

(`import dataclasses`, `import pytest`; `_host`'s tmp path argument must be a fresh directory -- create `tmp_path / "on"` with `mkdir()` first if `ram_miss_setup` needs it to exist.) In `_SCRIPT_HOST`, drop the `FakeForward`/`HANDLE` import, enable with `host.test_kernel_address(fail={result})` and register with `host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))` (import `fake_cpu_layer` from the fixtures module); `test_a_failed_forward_aborts_the_process` expects `f"CPU expert forward of row {ROW} failed: fake CPU expert forward failed (-5)"`; the miss-read test prints `len(host.test_kernel_calls())`. `test_cpu_experts_need_the_copy_engine` enables with `host.test_kernel_address()`.

`test_exl3_ram_miss_numa_completion.py`: one kernel per tier, so `_host(tmp_path, *, split, arm=True, ns_per_expert=0)` enables `host.test_kernel_address(ns_per_expert)` on both groups; the per-group checks read `core`: `a.calls == [([slot2], 1)] and b.calls == [([slot3], 2)]` -> `sorted((c["core"], c["slots"]) for c in host.test_kernel_calls()) == [(cores[0], [slot2]), (cores[2], [slot3])]`. The gated test holds group 1's first core with `host.test_kernel_hold(cores[2])` in place of the `threading.Event` gate and releases it with `host.test_kernel_hold(cores[2], False)` (in a `finally` before `host.stop()`, so a failure cannot leave the engine spinning at teardown). The "native" calibration case uses `ns_per_expert=1000`. In `_SCRIPT`, the stuck group 1 becomes a hold: enable both groups with `host.test_kernel_address(0)`, then `host.test_kernel_hold(cores[2])` before the post; the expected abort text `"group 1: its CPU job"` is unchanged.

`test_cpu_expert_keep_warm.py`: `_host(tmp_path, request, keep_warm_us)` enables `host.test_kernel_address(FORWARD_NS)` with `keep_warm_us=keep_warm_us` (no `keep_warm=`, no `engine=`), registers `fake_cpu_layer(HIDDEN)`; replace the last test with:

```python
def test_the_idle_thread_keeps_its_own_cores_warm(tmp_path, request):
    """The keep-warm pins its workers to the group's cores, which must be the ones enable_cpu_experts took. Mutant:
    pass empty cores to keep_warm in CpuExpertEngine::run -- red (core -1)."""
    _, host, _keep = _host(tmp_path, request, keep_warm_us=500_000)
    _run_jobs(host)
    time.sleep(0.02)
    assert host.test_keep_warm_calls() >= 1
    assert host.test_keep_warm_core() == sorted(os.sched_getaffinity(0))[0]
```

and update its docstring's "native fakes" sentence to "the instr build's fake kernel; its keep-warm counts its calls (from 0 at each test_kernel_address) ...". `test_exl3_cpu_split_calibration.py`: `host.test_forward_address(ns)` -> `host.test_kernel_address(ns)`, `host.set_cpu_layer(ROW, 7)` -> `host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))` (use the file's hidden size, 8). `test/manual/dsv41/test_cpu_split_calibration_cuda.py`: the same two substitutions. `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`: delete `_Forward`; enable `c.host.test_kernel_address(zero=True)` (the old fake wrote a zero partial), register `fake_cpu_layer(HIDDEN)`, and read `[(c["slots"], c["weights"]) for c in c.host.test_kernel_calls()]` where it read `(HANDLE, slots, weights)` tuples (drop `HANDLE` from the expected tuples).

`test_cpu_expert_pool.py`, the service fakes only: `FakeHost.enable_cpu_experts(self, kernel, split, cores, x_rows, out_rows, *, threads, group=0, spin_us=50_000, keep_warm_us=0)` records `kernel` and `keep_warm_us`; `FakeHost.set_cpu_layer(self, row, spec)` records `(row, spec)`; `FakeServiceTrait` gains `kernel_address()` returning `0x1234` and `layer_spec(slabs, capacity)` returning `CpuExpertLayerSpec(capacity, 8, 8, self.act_limit, (), b"")` and drops `native_*`. Assertions about engines (`native_create_engine` calls, `engine=`) become assertions that `enable_cpu_experts` got `kernel == 0x1234`; `keep_warm=` assertions become `keep_warm_us` ones (2000 by default, 0 at 0). `test_cpu_expert_groups_run_one_engine_per_node_and_register_each_layer_once` checks one `set_cpu_layer` per row across groups.

`test_expert_stream_build_variants.py`: add to `RAW_EXPORTS`

```python
    "test_kernel_address": lambda m, h: m.expert_stream_test_kernel_address(0, 0, 0),
    "test_kernel_calls": lambda m, h: m.expert_stream_test_kernel_calls(torch.zeros((0, 5 + 2 * 8), dtype=torch.float64)),
    "test_kernel_hold": lambda m, h: m.expert_stream_test_kernel_hold(0, 0),
    "test_keep_warm_calls": lambda m, h: m.expert_stream_test_keep_warm_calls(),
    "test_keep_warm_core": lambda m, h: m.expert_stream_test_keep_warm_core(),
```

- [ ] **Step 2: Run to verify they fail.** Commit the test changes (`test(expert-stream): CPU lanes through a fake CpuExpertKernel`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py test/registered/unit/kernels/test_expert_stream_build_variants.py`. Expected: FAIL with `has no attribute 'test_kernel_address'` / `cannot import name 'fake_cpu_layer'` (the fixture is in this commit, so the latter only if staged wrong).

- [ ] **Step 3: `cpu_experts.h`.** Include `"cpu_experts/kernel.hpp"` instead of `"cpu_experts_abi.h"`; add `<span>`, `<stdexcept>`. Rewrite the file comment's middle: "A CPU lane is a RAM-tier expert the device typed with kLeaseTagCpu. The engine runs it from its host slot through the format's CpuExpertKernel (cpu_experts/kernel.hpp) ... // CpuExpertLayers every row's layer, shared by the tier's engines // CpuJob ... // CpuExpertConfig the pinned input/output tables, the thread count, the cores and the kernel". Delete `CpuExpertForward` and `CpuExpertKeepWarm`. Add before `CpuJob`:

```cpp
// Every row's CPU expert layer, one per RamTier and shared by every group's engine: a row's layer addresses the row's
// whole slab, so one serves every group. Set once per row from any thread; read lock-free by the engines. Owns the
// layers, and is destroyed only after every engine has stopped (RamTier's member order), so no forward reads a freed
// one.
class CpuExpertLayers {
 public:
  explicit CpuExpertLayers(int64_t rows)
      : rows_(rows), layers_(std::make_unique<std::atomic<cpu_experts::CpuExpertLayer*>[]>(static_cast<size_t>(rows))) {
    for (int64_t r = 0; r < rows_; ++r)
      layers_[r].store(nullptr, std::memory_order_relaxed);
  }
  ~CpuExpertLayers() {
    for (int64_t r = 0; r < rows_; ++r)
      delete layers_[r].load(std::memory_order_relaxed);
  }
  CpuExpertLayers(const CpuExpertLayers&) = delete;
  CpuExpertLayers& operator=(const CpuExpertLayers&) = delete;

  int64_t rows() const {
    return rows_;
  }

  // Installs `layer` as `row`'s (the caller checks both); false, destroying `layer`, when the row already has one. The
  // release pairs with get()'s acquire, so an engine that sees the layer sees it constructed.
  bool set(int64_t row, std::unique_ptr<cpu_experts::CpuExpertLayer> layer) {
    cpu_experts::CpuExpertLayer* unset = nullptr;
    if (!layers_[row].compare_exchange_strong(unset, layer.get(), std::memory_order_acq_rel)) return false;
    layer.release();
    return true;
  }

  // `row`'s layer; nullptr outside [0, rows) or before set().
  const cpu_experts::CpuExpertLayer* get(int64_t row) const {
    return row >= 0 && row < rows_ ? layers_[row].load(std::memory_order_acquire) : nullptr;
  }

 private:
  int64_t rows_;
  std::unique_ptr<std::atomic<cpu_experts::CpuExpertLayer*>[]> layers_;
};
```

`CpuExpertConfig`: replace `forward`, `engine`, `rows` with

```cpp
  const cpu_experts::CpuExpertKernel* kernel = nullptr;  // the format's kernel, for every forward and keep-warm
  const CpuExpertLayers* layers = nullptr;  // the tier's rows' layers (RamTier sets it); a row without one is skipped
```

replace `keep_warm`/`keep_warm_ns` with `int64_t keep_warm_ns = 0;  // > 0: the kernel's keep-warm runs while idle for this long after each job`, and make the `cores` comment "worker 0 uses the first CPU; every forward and keep-warm pins worker i to cores[i]". The constructor checks `config_.kernel == nullptr` ("no CPU expert kernel") and `config_.layers == nullptr` ("no CPU expert layers") in place of the forward and rows checks, and drops `handles_` and `set_layer`. `eligible(row)` returns `config_.layers->get(row) != nullptr`. In `run()`: `const bool warm = config_.keep_warm_ns > 0;`; the keep-warm call becomes

```cpp
          try {
            config_.kernel->keep_warm(
                config_.cores, config_.threads, reinterpret_cast<const uint32_t*>(&kick_), kick, warm_until);
          } catch (const std::exception& e) {
            fail_stop(prefix_ + "CPU expert keep-warm failed: " + e.what());
          }
```

and the forward becomes

```cpp
      cpu_experts::ForwardCall call;
      call.rows = 1;
      call.k = job.k;
      call.threads = config_.threads;
      call.x = config_.x_base + job.row * config_.x_stride;
      call.slots = job.slots;
      call.weights = job.weights;
      call.out =
          reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride);
      call.accumulate = job.accumulate;
      call.cores = config_.cores;
      try {
        const cpu_experts::CpuExpertLayer* layer = config_.layers->get(job.row);
        if (layer == nullptr) throw std::invalid_argument("the row has no registered layer");
        config_.kernel->forward(*layer, call);
      } catch (const std::exception& e) {
        fail_stop(prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed: " + e.what());
      }
```

- [ ] **Step 4: `ram_tier.h`.** Add members, declared before `dist_` so they are destroyed after it (after every engine):

```cpp
  // Every row's CPU expert layer, shared by every group's engine; declared before dist_, so destroyed after the engines.
  std::unique_ptr<CpuExpertLayers> cpu_layers_;
  const cpu_experts::CpuExpertKernel* cpu_kernel_ = nullptr;  // every group's engine runs this one kernel
```

and in the constructor body `cpu_layers_ = std::make_unique<CpuExpertLayers>(layers_);`. In `enable_cpu_experts`: replace `config.rows = layers_;` with

```cpp
    if (config.kernel == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts need the format's kernel");
    if (cpu_kernel_ != nullptr && cpu_kernel_ != config.kernel)
      throw std::runtime_error(
          error_prefix<Layout>() + "every group's CPU experts run one kernel: group " + std::to_string(g) +
          " names " + config.kernel->name() + ", another group " + cpu_kernel_->name());
    cpu_kernel_ = config.kernel;
    config.layers = cpu_layers_.get();
```

Replace `set_cpu_layer(int64_t row, int64_t handle)` with:

```cpp
  // Installs `row`'s layer, made by the enabled kernel, for every group's engine: the layer addresses the whole slab,
  // so one serves them all. Any time, once per row; until then no post types a CPU lane for the row.
  void set_cpu_layer(int64_t row, std::unique_ptr<cpu_experts::CpuExpertLayer> layer) {
    if (cpu_kernel_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    if (row < 0 || row >= layers_)
      throw std::runtime_error(error_prefix<Layout>() + "CPU expert layer for row " + std::to_string(row) + " of " +
                               std::to_string(layers_));
    if (layer == nullptr || &layer->kernel() != cpu_kernel_)
      throw std::runtime_error(error_prefix<Layout>() + "a CPU expert layer must come from the enabled kernel");
    if (!cpu_layers_->set(row, std::move(layer)))
      throw std::runtime_error(error_prefix<Layout>() + "a CPU expert layer is registered once");
  }

  // set_cpu_layer of the enabled kernel's make_layer over `d` and `params` (its std::invalid_argument propagates).
  void make_cpu_layer(int64_t row, const cpu_experts::LayerSlabs& d, std::span<const std::byte> params) {
    if (cpu_kernel_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    set_cpu_layer(row, cpu_kernel_->make_layer(d, params));
  }

  const cpu_experts::CpuExpertKernel* cpu_kernel() const {
    return cpu_kernel_;
  }
```

(`make_cpu_layer` repeats `set_cpu_layer`'s row check, with the same "CPU expert layer for row R of N" text, before calling `make_layer`, so an out-of-range row never reaches the kernel. `ROWS` in the test is the tier's row count: confirm `ram_miss_setup` builds that many rows, and use its count in the `match` if it differs.)

- [ ] **Step 5: `ffi_exports.h`.** First two helpers in `HostExports`, which `set_cpu_layer` here and Task 6's `kernel_layer` both call:

```cpp
  // A slab table and a layer's scalars as LayerSlabs (set_cpu_layer, the test export kernel_layer): `slabs` int64
  // [n, 2] of {address, slot bytes} in the format's slab order (address 0: an absent optional slab), n <= kMaxSlabs.
  static cpu_experts::LayerSlabs layer_slabs(
      TensorView slabs, int64_t capacity, int64_t hidden, int64_t intermediate, int64_t activation, double act_limit) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("slabs", TensorMatcher({-1, 2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slabs);
    if (slabs.size(0) > cpu_experts::kMaxSlabs)
      throw std::runtime_error(error_prefix<Layout>() + "a CPU expert layer has at most " +
                               std::to_string(cpu_experts::kMaxSlabs) + " slabs");
    cpu_experts::LayerSlabs d;
    d.capacity = static_cast<int32_t>(capacity);
    d.hidden = static_cast<int32_t>(hidden);
    d.intermediate = static_cast<int32_t>(intermediate);
    d.activation = static_cast<int32_t>(activation);
    d.act_limit = static_cast<float>(act_limit);
    d.slab_count = static_cast<int32_t>(slabs.size(0));
    const auto* s = static_cast<const int64_t*>(slabs.data_ptr());
    for (int i = 0; i < d.slab_count; ++i) {
      d.slabs[i] = reinterpret_cast<const void*>(static_cast<intptr_t>(s[2 * i]));
      d.slot_bytes[i] = static_cast<uint64_t>(s[2 * i + 1]);
    }
    return d;
  }

  // A layer's params tensor (uint8 [m], the format's parameter struct) as make_layer's bytes; empty for m = 0.
  static std::span<const std::byte> params_bytes(TensorView params) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("params", TensorMatcher({-1}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), params);
    return {static_cast<const std::byte*>(params.data_ptr()), static_cast<size_t>(params.size(0))};
  }
```

Then `enable_cpu_experts(int64_t handle, int64_t group, int64_t kernel, TensorView split, TensorView cores, TensorView x_rows, TensorView out_rows, int64_t hidden, int64_t parts, int64_t threads, int64_t spin_ns, int64_t keep_warm_ns)`: doc "`kernel` is the format's CpuExpertKernel's address (the trait's kernel_address()), which must outlive the service; rows join once their layer is made (set_cpu_layer). ... `keep_warm_ns` > 0 runs the kernel's keep-warm while idle for that long after each job." Body: `if (kernel == 0) throw ... "CPU experts need the format's kernel";` `config.kernel = reinterpret_cast<const cpu_experts::CpuExpertKernel*>(static_cast<intptr_t>(kernel));` drop `config.engine`/`config.forward`/`config.keep_warm`; `if (keep_warm_ns < 0) throw ...` stays; `config.keep_warm_ns = keep_warm_ns;`. Replace `set_cpu_layer`:

```cpp
  // CPU experts: `row`'s layer, made here by the enabled kernel's make_layer from the row's pinned slabs. `slabs` is
  // int64 [n, 2] of {address, slot bytes} in the format's slab order (address 0: an absent optional slab), `params`
  // uint8 [bytes] the format's parameter struct. Once per row, at any time; the slabs must outlive the service.
  static void set_cpu_layer(
      int64_t handle,
      int64_t row,
      TensorView slabs,
      int64_t capacity,
      int64_t hidden,
      int64_t intermediate,
      int64_t activation,
      double act_limit,
      TensorView params) {
    find(handle)->make_cpu_layer(
        row, layer_slabs(slabs, capacity, hidden, intermediate, activation, act_limit), params_bytes(params));
  }
```

(`fake_cpu_layer`'s table has zero rows: the Python side builds it as an int64 `[0, 2]` tensor (`_layer_tensors`, Step 7), and every Task 4 test that registers a fake layer exercises `TensorMatcher({-1, 2})` on it.) The export lines keep their names.

- [ ] **Step 6: The fake kernel (`ffi_test_exports.h`).** Replace `test_forward_ns` through `test_keep_warm_engine` with the fake below and its exports, and update the header's `misc` line to `test_kernel_address, test_kernel_calls, test_kernel_hold, test_keep_warm_calls, test_keep_warm_core, pause_ns`:

```cpp
  // Test only: the CPU expert kernel tests enable in place of a format's (test_kernel_address). Its layers keep only the
  // LayerSlabs' hidden. A forward spins k * ns_per_expert, waits while its worker-0 core is held (test_kernel_hold),
  // throws when made failing, else writes out[j] for j < max(hidden, 1) -- (accumulate ? out[j] : j) + sum_i weights[i]
  // * (slots[i] + 1), or a zero partial (accumulate ? out[j] : 0) when made zeroing -- and records the call. A
  // keep-warm counts its calls and records its first core, then spins until its word moves or its deadline passes.
  class FakeKernel final : public cpu_experts::CpuExpertKernel {
   public:
    struct Call {
      int32_t core, affinity, threads, accumulate, k;
      std::array<int32_t, Wire::kLanes> slots;
      std::array<float, Wire::kLanes> weights;
    };
    struct Layer final : cpu_experts::CpuExpertLayer {
      Layer(const CpuExpertKernel& k, int32_t h) : CpuExpertLayer(k), hidden(h) {}
      const int32_t hidden;
    };

    void reset(int64_t ns_per_expert, int64_t fail, bool zero) {
      std::lock_guard<std::mutex> lock(mutex_);
      calls_.clear();
      ns_.store(ns_per_expert, std::memory_order_relaxed);
      fail_.store(fail, std::memory_order_relaxed);
      zero_.store(zero, std::memory_order_relaxed);
      held_core_.store(-1, std::memory_order_release);
      warm_calls_.store(0, std::memory_order_relaxed);
      warm_core_.store(-1, std::memory_order_relaxed);
    }

    const char* name() const noexcept override {
      return "fake";
    }
    std::unique_ptr<cpu_experts::CpuExpertLayer> make_layer(
        const cpu_experts::LayerSlabs& d, std::span<const std::byte>) const override {
      return std::make_unique<Layer>(*this, d.hidden);
    }
    void forward(const cpu_experts::CpuExpertLayer& layer, const cpu_experts::ForwardCall& c) const override {
      if (&layer.kernel() != this) throw std::invalid_argument("fake CPU expert kernel: another kernel's layer");
      const int32_t core = c.cores.empty() ? -1 : c.cores.front();
      const int64_t until = now_ns() + c.k * ns_.load(std::memory_order_relaxed);
      while (now_ns() < until)
        _mm_pause();
      while (core >= 0 && held_core_.load(std::memory_order_acquire) == core)
        _mm_pause();
      if (const int64_t f = fail_.load(std::memory_order_relaxed); f != 0)
        throw std::runtime_error("fake CPU expert forward failed (" + std::to_string(f) + ")");
      double sum = 0;
      for (int32_t i = 0; i < c.k; ++i)
        sum += static_cast<double>(c.weights[i]) * (c.slots[i] + 1);
      const int32_t hidden = std::max(static_cast<const Layer&>(layer).hidden, 1);
      const bool zero = zero_.load(std::memory_order_relaxed);
      for (int32_t j = 0; j < hidden; ++j) {
        const double base = c.accumulate ? static_cast<double>(c.out[j]) : (zero ? 0.0 : static_cast<double>(j));
        c.out[j] = static_cast<float>(zero ? base : base + sum);
      }
      cpu_set_t mask;
      CPU_ZERO(&mask);
      int32_t affinity = -1;
      if (sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) == 1)
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
          if (CPU_ISSET(cpu, &mask)) affinity = cpu;
      Call call{core, affinity, c.threads, c.accumulate ? 1 : 0, std::min<int32_t>(c.k, Wire::kLanes), {}, {}};
      for (int32_t i = 0; i < call.k; ++i) {
        call.slots[i] = c.slots[i];
        call.weights[i] = c.weights[i];
      }
      std::lock_guard<std::mutex> lock(mutex_);
      calls_.push_back(call);
    }
    void keep_warm(std::span<const int> cores, int32_t, const uint32_t* word, uint32_t seen, int64_t deadline_ns)
        const override {
      warm_calls_.fetch_add(1, std::memory_order_relaxed);
      warm_core_.store(cores.empty() ? -1 : cores.front(), std::memory_order_relaxed);
      while (__atomic_load_n(word, __ATOMIC_ACQUIRE) == seen && now_ns() < deadline_ns)
        _mm_pause();
    }

    std::vector<Call> calls() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return calls_;
    }
    void hold(int64_t core, bool on) {
      held_core_.store(on ? core : -1, std::memory_order_release);
    }
    int64_t warm_calls() const {
      return warm_calls_.load(std::memory_order_relaxed);
    }
    int64_t warm_core() const {
      return warm_core_.load(std::memory_order_relaxed);
    }

   private:
    mutable std::mutex mutex_;
    mutable std::vector<Call> calls_;
    std::atomic<int64_t> ns_{0}, fail_{0}, held_core_{-1};
    std::atomic<bool> zero_{false};
    mutable std::atomic<int64_t> warm_calls_{0}, warm_core_{-1};
  };
  static FakeKernel& fake_kernel() {
    static FakeKernel kernel;
    return kernel;
  }

  // Test only: the fake kernel's address, its calls and keep-warm counts reset; `fail` nonzero makes every forward
  // throw, `zero` makes it write a zero partial.
  static int64_t test_kernel_address(int64_t ns_per_expert, int64_t fail, int64_t zero) {
    if constexpr (!Build::kFaults) {
      test_only("test_kernel_address");
    } else {
      fake_kernel().reset(ns_per_expert, fail, zero != 0);
      return static_cast<int64_t>(reinterpret_cast<intptr_t>(&fake_kernel()));
    }
  }
  // Test only: the fake's calls since test_kernel_address, as float64 rows {core, affinity, threads, accumulate, k,
  // slots[kLanes], weights[kLanes]} into `out` (as many as fit); returns how many there are.
  static int64_t test_kernel_calls(TensorView out) {
    if constexpr (!Build::kFaults) {
      test_only("test_kernel_calls");
    } else {
      using namespace host;
      auto cpu = SymbolicDevice{};
      expert_stream::verify_named(
          "out", TensorMatcher({-1, 5 + 2 * Wire::kLanes}).with_dtype<double>().with_device<kDLCPU>(cpu), out);
      const std::vector<FakeKernel::Call> calls = fake_kernel().calls();
      auto* o = static_cast<double*>(out.data_ptr());
      const int64_t width = 5 + 2 * Wire::kLanes;
      for (int64_t r = 0; r < std::min<int64_t>(out.size(0), static_cast<int64_t>(calls.size())); ++r) {
        const FakeKernel::Call& c = calls[r];
        double* row = o + r * width;
        row[0] = c.core;
        row[1] = c.affinity;
        row[2] = c.threads;
        row[3] = c.accumulate;
        row[4] = c.k;
        for (int i = 0; i < Wire::kLanes; ++i) {
          row[5 + i] = c.slots[i];
          row[5 + Wire::kLanes + i] = c.weights[i];
        }
      }
      return static_cast<int64_t>(calls.size());
    }
  }
  // Test only: while on, a fake forward whose worker-0 core is `core` waits (one held core at a time).
  static void test_kernel_hold(int64_t core, int64_t on) {
    if constexpr (!Build::kFaults) {
      test_only("test_kernel_hold");
    } else {
      fake_kernel().hold(core, on != 0);
    }
  }
  static int64_t test_keep_warm_calls() {
    if constexpr (!Build::kFaults) {
      test_only("test_keep_warm_calls");
    } else {
      return fake_kernel().warm_calls();
    }
  }
  // Test only: the first core the fake keep-warm's last call took (-1 before any call since test_kernel_address).
  static int64_t test_keep_warm_core() {
    if constexpr (!Build::kFaults) {
      test_only("test_keep_warm_core");
    } else {
      return fake_kernel().warm_core();
    }
  }
```

(Check how the existing `test_only` branches return in a non-void function -- they rely on `test_only` being `[[noreturn]]`; the new ones follow the same shape.) Export lines replace the four old ones:

```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_kernel_address, Exports::test_kernel_address);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_kernel_calls, Exports::test_kernel_calls);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_kernel_hold, Exports::test_kernel_hold);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_keep_warm_calls, Exports::test_keep_warm_calls); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_keep_warm_core, Exports::test_keep_warm_core);   \
```

- [ ] **Step 7: The transport.** Add one module-level helper, used by `set_cpu_layer` here and by Task 6's `kernel_layer`:

```python
def _layer_tensors(spec) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``CpuExpertLayerSpec``'s slab table (int64 ``[n, 2]`` of {address, slot bytes}) and params (uint8 ``[m]``),
    as the host's set_cpu_layer and kernel_layer take them. ``reshape(n, 2)``: torch refuses ``reshape(-1, 2)`` of a
    zero-slab table (the fake kernel's layers have none)."""
    flat = [int(v) for pair in spec.slabs for v in pair]
    slabs = torch.tensor(flat, dtype=torch.int64).reshape(len(spec.slabs), 2)
    params = torch.tensor(list(spec.params), dtype=torch.uint8)
    return slabs, params
```

Add `"test_kernel_address", "test_kernel_calls", "test_kernel_hold", "test_keep_warm_calls", "test_keep_warm_core"` to `TEST_ONLY_EXPORTS`. `enable_cpu_experts(self, kernel: int, split, cores, x_rows, out_rows, *, threads, group=0, spin_us=50_000, keep_warm_us=0)`: docstring's `forward`/`engine`/`keep_warm` sentences become "``kernel`` is the format's ``CpuExpertKernel`` address (the trait's ``kernel_address()``); its library must stay loaded while the host runs. Each row joins once its layer is set (:meth:`set_cpu_layer`). ... For ``keep_warm_us`` after each job the idle thread runs the kernel's keep-warm on its workers (0: off)."; the FFI call passes `int(kernel)` and drops `engine`/`keep_warm`. Then:

```python
    def set_cpu_layer(self, row: int, spec) -> None:
        """Make ``row``'s layer with the enabled kernel from ``spec`` (a ``CpuExpertLayerSpec``).

        Once per row, at any time. The host keeps ``spec.keep`` (the slabs the layer reads) alive.
        """
        slabs, params = _layer_tensors(spec)
        self._module.expert_stream_set_cpu_layer(
            self.handle, row, slabs, int(spec.capacity), int(spec.hidden), int(spec.intermediate),
            int(spec.activation), float(spec.act_limit), params,
        )
        self._cpu_layer_keep[row] = spec.keep
```

(initialise `self._cpu_layer_keep: dict[int, tuple] = {}` in `__init__`. No Python row check: `RamTier::make_cpu_layer`'s "CPU expert layer for row" refusal is the one the test pins.) Replace the four fake wrappers with:

```python
    def test_kernel_address(self, ns_per_expert: int = 0, *, fail: int = 0, zero: bool = False) -> int:
        """Test only: the instr build's fake ``CpuExpertKernel`` (``FakeKernel``, ffi_test_exports.h), reset.

        A forward spins ``ns_per_expert`` per expert and writes ``out[j] = (out[j] if accumulate else j) + sum(w * (s +
        1))`` (``zero``: a zero partial); ``fail`` nonzero makes every forward throw. Instrumented build only.
        """
        _refuse_test_only("test_kernel_address", self.variant)
        return int(self._module.expert_stream_test_kernel_address(int(ns_per_expert), int(fail), int(bool(zero))))

    def test_kernel_calls(self) -> list[dict]:
        """Test only: the fake kernel's forwards since :meth:`test_kernel_address`, in order."""
        _refuse_test_only("test_kernel_calls", self.variant)
        lanes = self.wire.lanes
        width = 5 + 2 * lanes
        count = int(self._module.expert_stream_test_kernel_calls(torch.zeros((0, width), dtype=torch.float64)))
        out = torch.zeros((count, width), dtype=torch.float64)
        self._module.expert_stream_test_kernel_calls(out)
        calls = []
        for row in out.tolist()[:count]:
            k = int(row[4])
            calls.append({
                "core": int(row[0]), "affinity": int(row[1]), "threads": int(row[2]), "accumulate": bool(row[3]),
                "slots": [int(s) for s in row[5 : 5 + k]], "weights": row[5 + lanes : 5 + lanes + k],
            })
        return calls

    def test_kernel_hold(self, core: int, on: bool = True) -> None:
        """Test only: hold (or release) the fake forwards whose worker-0 core is ``core``."""
        _refuse_test_only("test_kernel_hold", self.variant)
        self._module.expert_stream_test_kernel_hold(int(core), int(bool(on)))

    def test_keep_warm_calls(self) -> int:
        """Test only: calls of the fake kernel's keep-warm since :meth:`test_kernel_address`."""
        _refuse_test_only("test_keep_warm_calls", self.variant)
        return int(self._module.expert_stream_test_keep_warm_calls())

    def test_keep_warm_core(self) -> int:
        """Test only: the first core the fake keep-warm's last call took (-1 before any call)."""
        _refuse_test_only("test_keep_warm_core", self.variant)
        return int(self._module.expert_stream_test_keep_warm_core())
```

(The calls count may grow between the two FFI calls; `[:count]` keeps the first sizing's rows.)

- [ ] **Step 8: The service.** In `CpuExpertService.__init__`: drop `self.engine = trait.native_create_engine(...)`; `shared` copies `shared.layers` instead of `shared.handles`; `self.layers: dict[int, object] = {}`; the enable call is

```python
        host.enable_cpu_experts(
            trait.kernel_address(),
            self.split,
            self.cores,
            self.x_rows,
            self.out_rows,
            threads=self.threads,
            group=self.group,
            keep_warm_us=max(keep_warm_us, 0),
        )
```

`registered`, `attach_device` and `calibrate` read `self.layers`; `register` builds `spec = self.trait.layer_spec({name: slabs[name] for name in self.trait.slab_names}, capacity)`, stores `self.layers[row] = spec` and calls `self.host.set_cpu_layer(row, spec)`. Update the class docstring's "layer handles" to "layer specs" and `CpuExpertGroups`' docstring "one layer handle addresses a whole slab, so every group's engine gets the same handle" to "one layer addresses a whole slab, so the host shares each row's layer among the groups' engines".

- [ ] **Step 9: Run.** Commit (`feat(expert-stream): CPU lanes run through CpuExpertKernel; one layer table per tier`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_expert_stream_build_variants.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py` -- all pass. Then `KIFACE_CPU test/registered/unit/kernels/test_cpu_experts_abi.py` -- `EXIT=0`. The C++ benches do not build at this commit (they still use `CpuExpertConfig::forward`); Task 5 is the next commit and runs `BENCH`, so do not run `CPU_CHECKS` here.

---

### Task 5: The full-stack bench on the kernel interface

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack.h:90-110,165-190,236-240`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/stack_fixture.h:50-60`, `stack_fixture.cpp:1-10,155-205`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/full_stack.cpp` (includes, `stack_config`, `LayerHandles`, `bare_into`, `main`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/self_test.cpp:495-800`, `self_test.h` if it declares the fake

**Interfaces:**
- Consumes: `CpuExpertConfig{kernel, keep_warm_ns}`, `RamTier::set_cpu_layer(row, std::unique_ptr<CpuExpertLayer>)` (Task 4); `exl3_cpu_kernel()` (Task 1).
- Produces: `StackConfig::kernel` (`const cpu_experts::CpuExpertKernel*`), `StackConfig::keep_warm_ns`; `Stack::set_cpu_layer(int64_t row, std::unique_ptr<cpu_experts::CpuExpertLayer>)`; `StackFixture::make_layer(int64_t row) const -> std::unique_ptr<cpu_experts::CpuExpertLayer>`. Removed: `StackConfig::forward`, `::keep_warm`, `Group::engine`, `Group::forward`, `StackFixture::register_layer`, `StackFixture::free_layer`.

- [ ] **Step 1: Write the failing test.** In `self_test.cpp`, replace `FakeCall`, `fake_calls`, `fake_mutex` and `fake_forward` with a bench-local fake kernel, and add a self-test that two groups on different kernels are refused:

```cpp
// One call of the fake kernel, recorded for the test to inspect.
struct FakeCall {
  std::vector<int32_t> slots;
  std::vector<float> weights;
  int32_t threads;
  bool accumulate;
  int core;  // worker 0's core: the group's first
};

// The kernel's stand-in: out[h] = h + x[0] + sum_i weights[i] * (slots[i] + 1), x[0] read as an integer, so the output
// proves the row's x, the slots and the weights reached it.
class FakeKernel final : public es::cpu_experts::CpuExpertKernel {
 public:
  struct Layer final : es::cpu_experts::CpuExpertLayer {
    explicit Layer(const CpuExpertKernel& k) : CpuExpertLayer(k) {}
  };
  const char* name() const noexcept override { return "bench-fake"; }
  std::unique_ptr<es::cpu_experts::CpuExpertLayer> make_layer(
      const es::cpu_experts::LayerSlabs&, std::span<const std::byte>) const override {
    return std::make_unique<Layer>(*this);
  }
  void forward(const es::cpu_experts::CpuExpertLayer& layer, const es::cpu_experts::ForwardCall& c) const override {
    if (&layer.kernel() != this) throw std::invalid_argument("bench fake: another kernel's layer");
    uint16_t x0;
    std::memcpy(&x0, c.x, 2);
    float sum = 0.0f;
    for (int32_t i = 0; i < c.k; ++i)
      sum += c.weights[i] * static_cast<float>(c.slots[i] + 1);
    for (int64_t h = 0; h < kSelfHidden; ++h)
      c.out[h] = (c.accumulate ? c.out[h] : 0.0f) + static_cast<float>(h) + static_cast<float>(x0) + sum;
    std::lock_guard<std::mutex> guard(mutex);
    calls.push_back({std::vector<int32_t>(c.slots, c.slots + c.k), std::vector<float>(c.weights, c.weights + c.k),
                     c.threads, c.accumulate, c.cores.empty() ? -1 : c.cores.front()});
  }
  void keep_warm(std::span<const int>, int32_t, const uint32_t*, uint32_t, int64_t) const override {}

  mutable std::mutex mutex;
  mutable std::vector<FakeCall> calls;
};
```

(`es` is the bench's alias for `sglang::expert_stream`; `cpu_experts` is `sglang::cpu_experts`, so write `::sglang::cpu_experts::` if `es::cpu_experts` does not name it.) `test_stack` and `test_two_groups` create `FakeKernel fake;` before the `StackConfig`, set `config.kernel = &fake;`, register with `stack.set_cpu_layer(0, fake.make_layer({}, {}))`, read `fake.calls` under `fake.mutex`, and `test_two_groups` tells the groups apart by `call.core == placement.groups[g].workers.front()` where it read `call.engine - 1` (drop `group.engine = g + 1`). Give the bench fake a name, so a refusal can be shown to name both kernels: `explicit FakeKernel(const char* name = "bench-fake") : name_(name) {}`, `const char* name() const noexcept override { return name_; }`, member `const char* name_;`.

Factor `test_two_groups`' rows, images and buffers into a rig both two-group tests use (its body is `test_two_groups`' current setup, lines 701-743 of `self_test.cpp`, moved):

```cpp
// Two NUMA groups' synthetic rows, row images and pinned buffers (test_two_groups, test_groups_must_share_one_kernel):
// per group 3 staging slots and 4 mappable; config() sends every eligible lane to the CPU.
struct TwoGroupRig {
  static constexpr int64_t kGroup = 7;
  std::vector<std::array<AlignedBuffer, kNames>> slabs;
  RowSet set;
  AlignedBuffer x;
  AlignedBuffer out;

  TwoGroupRig(const std::filesystem::path& dir, const std::string& name)
      : slabs(kSelfRows),
        x(aligned_zeroed(kSelfRows * 2 * kSelfHidden)),
        out(aligned_zeroed(kSelfRows * 4 * kSelfHidden * 4)) {  // two parts per group
    const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
    set.layout = layout;
    set.experts = kSelfExperts;
    set.capacity = 2 * kGroup;
    for (int64_t row = 0; row < kSelfRows; ++row) {
      std::array<uint8_t*, kNames> bases{};
      for (int n = 0; n < kNames; ++n) {
        slabs[row][n] = aligned_zeroed(2 * kGroup * 512);
        bases[n] = slabs[row][n].get();
      }
      set.slabs.push_back(bases);
      const auto path = dir / (name + "-layer-" + std::to_string(row) + ".rows");
      write_row_image(
          path,
          layout,
          kSelfExperts,
          [&](int64_t e, uint8_t* image) {
            for (int n = 0; n < kNames; ++n)
              std::memset(image + layout.name_offsets[n], pattern(row, e, n), 512);
          },
          "");
      set.paths.push_back(path.string());
    }
  }

  StackConfig config(const Placement& placement, const ::sglang::cpu_experts::CpuExpertKernel& kernel) const {
    StackConfig c;
    c.rows = set;
    c.staging = 3;
    c.kernel = &kernel;
    c.x_base = x.get();
    c.x_stride = 2 * kSelfHidden;
    c.out_base = out.get();
    c.out_stride = 4 * kSelfHidden * 4;
    c.hidden = kSelfHidden;
    c.copy_cpu = placement.copy;
    c.ranges = {{0, kGroup}, {kGroup, 2 * kGroup}};
    for (int g = 0; g < 2; ++g) {
      StackConfig::Group group;
      group.service_cpu = placement.groups[g].service;
      group.cores.assign(placement.groups[g].workers.begin(), placement.groups[g].workers.end());
      group.split = {0, 1, 2, 3, 4, 5, 6, 7, 8};
      c.groups.push_back(group);
    }
    return c;
  }
};
```

`test_two_groups` becomes `TwoGroupRig rig(dir, "selftest2"); FakeKernel fake; StackConfig config = rig.config(placement, fake);` followed by its unchanged body from `PinScope writer` on, with `kGroup` read as `TwoGroupRig::kGroup` and `out` as `rig.out`. Add, and call it from `run_self_test` after `test_two_groups`:

```cpp
// Review Focus 2: one tier runs one kernel. Group 1 naming a second kernel is refused while the stack is built, the
// message naming both kernels. Runs only in the two-node build: BENCH kiface-bench-n2 gates it, kiface-bench (one node)
// returns at once.
void test_groups_must_share_one_kernel(const Placement& placement, const std::filesystem::path& dir) {
  if constexpr (w::Wire::kNodes != 2) {
    return;
  } else {
    TwoGroupRig rig(dir, "selftest-kernels");
    FakeKernel first("fake-a"), second("fake-b");
    StackConfig config = rig.config(placement, first);
    config.groups[1].kernel = &second;
    PinScope writer(placement.writer);
    CHECK_THROWS(Stack<BenchBuild> stack(std::move(config)), "group 1 names fake-b, another group fake-a");
  }
}
```

`StackConfig::Group` gains one field for this test, `const ::sglang::cpu_experts::CpuExpertKernel* kernel = nullptr;  // null: the stack's kernel (set only to provoke the one-kernel refusal)`. The needle is `RamTier::enable_cpu_experts`' message (Task 4 Step 4): "every group's CPU experts run one kernel: group 1 names fake-b, another group fake-a".

- [ ] **Step 2: Run to verify it fails.** Commit, `SYNC`, `BENCH kiface-bench 1` -- expected `BUILD` nonzero (`StackConfig` has no member `kernel`).

- [ ] **Step 3: Implement.** `stack.h`: `StackConfig` gets `const cpu_experts::CpuExpertKernel* kernel = nullptr;  // the format's kernel, every group's` and `int64_t keep_warm_ns = 0;  // > 0: the kernel's keep-warm runs while idle after each job`, losing `forward`, `keep_warm`, `Group::engine`, `Group::forward`; the enable loop sets `cpu.kernel = group.kernel != nullptr ? group.kernel : config_.kernel;` and `cpu.keep_warm_ns = config_.keep_warm_ns;`; `set_cpu_layer`:

```cpp
  // Installs the row's CPU layer (StackFixture::make_layer), made by the stack's kernel.
  void set_cpu_layer(int64_t row, std::unique_ptr<::sglang::cpu_experts::CpuExpertLayer> layer) {
    tier_->set_cpu_layer(row, std::move(layer));
  }
```

`stack_fixture.h/.cpp`: include `kernel.h` (on the bench's include path `${QUANT}/optimized`) and `<cstring>`; replace `register_layer` and `free_layer` with

```cpp
// Mirrors Exl3CpuQuantTrait.layer_spec: the row's six slabs by base pointer and row stride (kNames is
// EXL3_STREAMED_NAMES' order), made into a layer by the EXL3 kernel. Views only: the fixture's slabs outlive it.
std::unique_ptr<::sglang::cpu_experts::CpuExpertLayer> StackFixture::make_layer(int64_t row) const {
  const int32_t params[2] = {3, 0};  // SglangExl3CpuParams {bits, swizzled}
  ::sglang::cpu_experts::LayerSlabs d;
  // ... the capacity/hidden/intermediate/activation/act_limit/slab_count/slabs/slot_bytes assignments of today's
  //     register_layer, unchanged, into d ...
  return ::sglang::exl3_cpu::exl3_cpu_kernel().make_layer(d, std::as_bytes(std::span<const int32_t>(params, 2)));
}
```

`full_stack.cpp`: drop the `cpu_experts_cabi.h` include (include `kernel.h`); `stack_config` sets `c.kernel = &::sglang::exl3_cpu::exl3_cpu_kernel();` and `c.keep_warm_ns`, drops `engines` (parameter and `group.engine`); `LayerHandles` becomes

```cpp
// The bare forwards' layers, made before the stack exists (BM_bare runs first); the stack gets its own per row.
struct BareLayers {
  std::vector<std::unique_ptr<::sglang::cpu_experts::CpuExpertLayer>> layers;
};
```

`bare_into` builds a `ForwardCall` (`rows 1`, `k`, `threads = workers.size()`, `x`, `slots`, `weights`, `out`, `accumulate false`, `cores = placement_.groups[group].workers`) and calls `::sglang::exl3_cpu::exl3_cpu_kernel().forward(*bare_[row], call)` inside `try`, rethrowing as `std::runtime_error(std::string("the bare CPU forward failed: ") + e.what())`; wherever the bench passed `handles_[row]` to `stack_->set_cpu_layer`, pass `fixture_.make_layer(row)`; `main` drops `engine_create` and fills `BareLayers` with `fixture->make_layer(row)`. Read each touched function before editing; every `engines`/`handles` member and parameter goes.

- [ ] **Step 4: Run.** Commit (`feat(bench): the full-stack bench runs the kernel interface`), `SYNC`, `BENCH kiface-bench 1` and `BENCH kiface-bench-n2 2` -- `BUILD=0 EXIT=0` both. `CPU_CHECKS t5` -- `ALL GREEN (check)` (bare-forward bench still on the shim, full-stack bench on the interface, 48 frozen outputs bit-exact).

---

### Task 6: Kernel-correctness tests through the host's kernel test exports

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (three exports)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`TEST_ONLY_EXPORTS`, module functions `kernel_layer`, `kernel_forward`, `kernel_drop`)
- Modify: `test/registered/unit/kernels/test_nvfp4_cpu_experts.py`, `test/registered/unit/kernels/test_expert_stream_build_variants.py`
- Modify: `test/manual/dsv41/exl3_cpu_forward_ab.py`, `test/manual/dsv41/test_cpu_expert_engines_exl3.py`

**Interfaces:**
- Consumes: `CpuExpertLayerSpec`, `trait.layer_spec`, `trait.kernel_address` (Task 3).
- Produces (C++ test-only): `expert_stream_kernel_layer(kernel, slabs int64[n,2], capacity, hidden, intermediate, activation, act_limit, params uint8[m]) -> int64 id`; `expert_stream_kernel_forward(id, x, slots int32[rows,k], weights float32[rows,k], out float32[rows,hidden], threads, cores int64[n], accumulate) -> int64` (0, 2 for `std::invalid_argument`, 1 for any other exception); `expert_stream_kernel_error() -> string` (this thread's last message); `expert_stream_kernel_drop(id)`.
- Produces (Python, `expert_stream_transport`): `kernel_layer(kernel: int, spec, *, layout="exl3", variant=None) -> int`; `kernel_forward(layer: int, x, slots, weights, out, *, threads: int, cores=(), accumulate=False, layout="exl3", variant=None) -> tuple[int, str]`; `kernel_drop(layer: int, *, layout="exl3", variant=None) -> None`.

- [ ] **Step 1: Write the failing tests.** Rewrite `test_nvfp4_cpu_experts.py`'s `CHILD` to drive the kernel through the host module:

```python
CHILD = r"""
import struct, sys, threading
import numpy as np
import torch
from tvm_ffi import load_module
from sglang.kernels.ops.moe import expert_stream_transport as es
from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec

kernel = int(load_module(sys.argv[1]).nvfp4_cpu_kernel_address())
calls = [int(t) for t in sys.argv[2].split(",")]
cores = [int(c) for c in sys.argv[3].split(",")] if sys.argv[3] else []
alpha_value = float(sys.argv[4])
groups = int(sys.argv[5])

H = N = 80
CAP = 2
w13 = np.full(CAP * N * H, 0x22, np.uint8)
w2 = np.full(CAP * H * N // 2, 0x22, np.uint8)
sf13 = np.full(CAP * 256 * 8, 56, np.uint8)
sf2 = np.full(CAP * 128 * 8, 56, np.uint8)
alpha = np.full(CAP, alpha_value, np.float32)
slabs = [(w13, N * H), (w2, H * N // 2), (sf13, 256 * 8), (sf2, 128 * 8), (alpha, 4), (alpha, 4)]
spec = CpuExpertLayerSpec(capacity=CAP, hidden=H, intermediate=N, act_limit=0.0,
                          slabs=tuple((a.ctypes.data, s) for a, s in slabs) + ((0, 0),),
                          params=struct.pack("<iff", 0, 1.0, 1.0))
layer = es.kernel_layer(kernel, spec, variant="instr")
x = torch.full((1, H), 1.0, dtype=torch.float16)
slots = torch.zeros((1, 1), dtype=torch.int32)
weights = torch.ones((1, 1), dtype=torch.float32)


def forward(threads, on):
    out = torch.full((1, H), 123.0)
    status, _ = es.kernel_forward(layer, x, slots, weights, out, threads=threads, cores=on, variant="instr")
    return status, "untouched" if bool((out == 123.0).all()) else "written", out


if groups < 2:
    for threads in calls:
        status, state, _ = forward(threads, cores)
        print("forward", threads, status, state)
else:
    width = len(cores) // groups
    parts = [cores[i * width:(i + 1) * width] for i in range(groups)]
    results = {i: [] for i in range(groups)}

    def run(i):
        for _ in range(20):
            for threads in calls:
                results[i].append((threads,) + forward(threads, parts[i]))

    runners = [threading.Thread(target=run, args=(i,)) for i in range(groups)]
    for runner in runners:
        runner.start()
    for runner in runners:
        runner.join()
    for i in range(groups):
        for threads, status, state, _ in results[i]:
            print("forward", i, threads, status, state)
    first, second = results[0], results[1]
    print("same" if all(torch.equal(p[3], q[3]) for p, q in zip(first, second)) else "different")
"""
```

`_run(library, calls, cores=(), alpha=1.0, groups=1, **env)` passes `str(groups)`. Expected lines: `test_more_workers_than_the_engines_cores_is_refused_and_leaves_out_untouched` -> rename `test_more_workers_than_cores_is_refused_and_leaves_out_untouched`, `== ["forward 2 2 untouched"]`; delete `test_a_freed_engine_is_refused` (engine registry semantics); the unpinnable test asserts `lines[:1] == ["forward 2 1 untouched"]`; `test_two_engines_forward_at_once_without_a_busy_status` -> `test_two_core_groups_forward_at_once` with `groups=2` and no `"engine 0"` lines; the others are unchanged. Replace `test_the_scheme_registers_a_layer_and_runs_it` and `test_the_scheme_creates_and_frees_engines` with:

```python
def test_the_scheme_layer_runs_through_the_kernel(built):
    import torch
    from tvm_ffi import load_module

    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, module=load_module(str(built)))
    slabs = {  # the CHILD's layer: H = N = 80, two slots
        "w13": torch.full((2, 80 * 80), 0x22, dtype=torch.uint8),
        "w2": torch.full((2, 80 * 80 // 2), 0x22, dtype=torch.uint8),
        "sf13": torch.full((2, 256 * 8), 56, dtype=torch.uint8),
        "sf2": torch.full((2, 128 * 8), 56, dtype=torch.uint8),
        "gate_alpha": torch.ones(2, 1),
        "down_alpha": torch.ones(2, 1),
    }
    layer = es.kernel_layer(trait.kernel_address(), trait.layer_spec(slabs, capacity=2), variant="instr")
    saved = os.sched_getaffinity(0)
    try:
        x = torch.ones(1, 80, dtype=torch.float16)
        out = torch.full((1, 80), 123.0)
        assert es.kernel_forward(layer, x, torch.zeros(1, 1), torch.ones(1, 1), out, threads=1, variant="instr")[0] == 0
        assert torch.isfinite(out).all() and not (out == 123.0).any()
        refused = torch.full((1, 80), 123.0)
        status, why = es.kernel_forward(layer, x, torch.full((1, 1), 2), torch.ones(1, 1), refused, threads=1, variant="instr")
        assert status == 2 and "nvfp4" in why and (refused == 123.0).all()  # slot 2 is past the capacity
    finally:
        es.kernel_drop(layer, variant="instr")
        os.sched_setaffinity(0, saved)


def test_the_kernel_refuses_params_of_the_wrong_size(built):
    """Review Focus 4: make_layer checks the params' size against the quant's struct and names both."""
    import dataclasses

    import torch
    from tvm_ffi import load_module

    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

    trait = Nvfp4CpuQuantTrait(hidden=80, intermediate=80, act_limit=0.0, module=load_module(str(built)))
    slabs = {n: torch.zeros((2, b), dtype=torch.uint8) for n, b in
             (("w13", 6400), ("w2", 3200), ("sf13", 2048), ("sf2", 1024))}
    slabs |= {"gate_alpha": torch.ones(2, 1), "down_alpha": torch.ones(2, 1)}
    spec = dataclasses.replace(trait.layer_spec(slabs, capacity=2), params=b"\0" * 8)
    with pytest.raises(Exception, match="params hold 8 bytes, the quant's hold 12"):
        es.kernel_layer(trait.kernel_address(), spec, variant="instr")
```

Update the module docstring: "The NVFP4 CPU expert kernel under its OpenMP team, through the expert-stream host's kernel test exports (Linux, GCC with OpenMP). Each case runs in its own process ...". In `test_expert_stream_build_variants.py` add to `RAW_EXPORTS`:

```python
    "kernel_layer": lambda m, h: m.expert_stream_kernel_layer(
        0, torch.zeros((0, 2), dtype=torch.int64), 0, 0, 0, 0, 0.0, torch.zeros(0, dtype=torch.uint8)
    ),
    "kernel_forward": lambda m, h: m.expert_stream_kernel_forward(
        0, torch.zeros(1, dtype=torch.uint8), torch.zeros((1, 1), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.float32), torch.zeros((1, 1), dtype=torch.float32), 1,
        torch.zeros(0, dtype=torch.int64), 0,
    ),
    "kernel_drop": lambda m, h: m.expert_stream_kernel_drop(0),
```

In `test_the_module_level_test_only_helpers_refuse_on_prod`, the literal parametrize tuple becomes
`("read_rows_with_fault", "read_rows_sqes", "seqlock_stress", "pause_ns", "read_record_fields", "kernel_layer", "kernel_forward", "kernel_drop")`
and `calls` gains (with `from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer`):

```python
        "kernel_layer": lambda: ops.kernel_layer(0, fake_cpu_layer(), variant="prod"),
        "kernel_forward": lambda: ops.kernel_forward(
            0, torch.zeros((1, 8), dtype=torch.float16), torch.zeros((1, 1)), torch.zeros((1, 1)),
            torch.zeros((1, 8)), threads=1, variant="prod",
        ),
        "kernel_drop": lambda: ops.kernel_drop(0, variant="prod"),
```

`test/manual/dsv41/exl3_cpu_forward_ab.py`: `register_slabs(ext, s, limit)` returns `(es.kernel_layer(trait.kernel_address(), trait.layer_spec(s, CAP)), trait)` (exit on `RuntimeError` as today); the `slabs` registration's forward is `status, why = es.kernel_forward(handle, x, sel, w.float(), out, threads=THREADS)` (exit naming the case unless `status == 0`); `engines` mode splits `--cores` into two halves and runs each half's cases on its own thread through `kernel_forward(..., cores=half)` (rename it `--registration cores`, keeping `engines` as an accepted alias so the check script's history reads); cleanup is `es.kernel_drop(handle)` for slab layers, `ext.exl3_moe_cpu_free_layer(handle)` for table layers. Fp16 weights widen exactly to fp32 and the kernel narrows them back, so the dumps stay bit-exact. Update its docstring to name `kernel_forward` instead of "the C ABI's forward". `test/manual/dsv41/test_cpu_expert_engines_exl3.py`: `_forward(trait, layer, x, slots, weights, cores, threads)` returns `(status, out)` from `es.kernel_forward(..., cores=cores)`; the test creates the layer with `kernel_layer(trait.kernel_address(), trait.layer_spec(_random_slabs(...), CAP))` and runs on `cores[:2]` and `cores[2:4]`; its docstring's "engines" become "core groups".

- [ ] **Step 2: Run to verify they fail.** Commit, `SYNC`, `RUN_CPU test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_expert_stream_build_variants.py` -- expected FAIL (`module ... has no attribute 'kernel_layer'`).

- [ ] **Step 3: Implement the exports** in `HostTestExports` (instr only; prod refuses through `test_only`):

```cpp
  // Test only: layers a test made with any kernel's make_layer (kernel_layer), by id, for kernel_forward.
  static std::mutex& kernel_layers_mutex() {
    static std::mutex mutex;
    return mutex;
  }
  // A test layer and the hidden size it was made with, which bounds kernel_forward's x and out.
  struct KernelLayer {
    std::shared_ptr<cpu_experts::CpuExpertLayer> layer;
    int64_t hidden = 0;
  };
  static std::vector<KernelLayer>& kernel_layers() {
    static std::vector<KernelLayer> layers;
    return layers;
  }
  static std::string& kernel_error_text() {
    static thread_local std::string text;
    return text;
  }

  // Test only: kernel `kernel`'s make_layer over a slab table as set_cpu_layer takes it; returns the layer's id. The
  // kernel's std::invalid_argument propagates.
  static int64_t kernel_layer(int64_t kernel, TensorView slabs, int64_t capacity, int64_t hidden, int64_t intermediate,
                              int64_t activation, double act_limit, TensorView params) {
    if constexpr (!Build::kFaults) {
      test_only("kernel_layer");
    } else {
      const auto* k = reinterpret_cast<const cpu_experts::CpuExpertKernel*>(static_cast<intptr_t>(kernel));
      std::shared_ptr<cpu_experts::CpuExpertLayer> layer =
          k->make_layer(Base::layer_slabs(slabs, capacity, hidden, intermediate, activation, act_limit),
                        Base::params_bytes(params));
      std::lock_guard<std::mutex> lock(kernel_layers_mutex());
      kernel_layers().push_back({std::move(layer), hidden});
      return static_cast<int64_t>(kernel_layers().size() - 1);
    }
  }

  // Test only: one forward of layer `id` with its own kernel: x fp16 [rows, hidden] (every format's input today),
  // slots int32 and weights float32 [rows, k], out float32 [rows, hidden], hidden the layer's, all contiguous CPU;
  // cores int64 [n] (empty: unpinned). The shapes are checked against the layer, so a short x or out is refused here
  // rather than read or written past its end. Returns 0, 2 for std::invalid_argument, 1 for any other exception, its
  // message in kernel_error().
  static int64_t kernel_forward(int64_t id, TensorView x, TensorView slots, TensorView weights, TensorView out,
                                int64_t threads, TensorView cores, int64_t accumulate) {
    if constexpr (!Build::kFaults) {
      test_only("kernel_forward");
    } else {
      KernelLayer entry;
      {
        std::lock_guard<std::mutex> lock(kernel_layers_mutex());
        if (id < 0 || id >= static_cast<int64_t>(kernel_layers().size()) || !kernel_layers()[id].layer)
          throw std::runtime_error("kernel_forward: no layer " + std::to_string(id));
        entry = kernel_layers()[id];
      }
      using namespace host;
      auto cpu = SymbolicDevice{};
      auto rows = SymbolicSize{"rows"};
      auto k = SymbolicSize{"k"};
      const int64_t hidden = entry.hidden;
      expert_stream::verify_named("x", TensorMatcher({rows, hidden}).with_dtype<uint16_t>().with_device<kDLCPU>(cpu), x);
      expert_stream::verify_named("slots", TensorMatcher({rows, k}).with_dtype<int32_t>().with_device<kDLCPU>(cpu), slots);
      expert_stream::verify_named("weights", TensorMatcher({rows, k}).with_dtype<float>().with_device<kDLCPU>(cpu), weights);
      expert_stream::verify_named("out", TensorMatcher({rows, hidden}).with_dtype<float>().with_device<kDLCPU>(cpu), out);
      expert_stream::verify_named("cores", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cores);
      const std::shared_ptr<cpu_experts::CpuExpertLayer>& layer = entry.layer;
      std::vector<int> on;
      const auto* c = static_cast<const int64_t*>(cores.data_ptr());
      for (int64_t i = 0; i < cores.size(0); ++i)
        on.push_back(static_cast<int>(c[i]));
      cpu_experts::ForwardCall call;
      call.rows = static_cast<int32_t>(slots.size(0));
      call.k = static_cast<int32_t>(slots.size(1));
      call.threads = static_cast<int32_t>(threads);
      call.x = x.data_ptr();
      call.slots = static_cast<const int32_t*>(slots.data_ptr());
      call.weights = static_cast<const float*>(weights.data_ptr());
      call.out = static_cast<float*>(out.data_ptr());
      call.accumulate = accumulate != 0;
      call.cores = on;
      kernel_error_text().clear();
      try {
        layer->kernel().forward(*layer, call);
        return 0;
      } catch (const std::invalid_argument& e) {
        kernel_error_text() = e.what();
        return 2;
      } catch (const std::exception& e) {
        kernel_error_text() = e.what();
        return 1;
      }
    }
  }

  static std::string kernel_error() {
    if constexpr (!Build::kFaults) {
      test_only("kernel_error");
    } else {
      return kernel_error_text();
    }
  }

  static void kernel_drop(int64_t id) {
    if constexpr (!Build::kFaults) {
      test_only("kernel_drop");
    } else {
      std::lock_guard<std::mutex> lock(kernel_layers_mutex());
      if (id >= 0 && id < static_cast<int64_t>(kernel_layers().size())) kernel_layers()[id] = {};
    }
  }
```

(`TensorMatcher`'s dtype for fp16 is whatever the codebase's matchers use for half -- read `sgl_kernel/tensor.h` and use that type in place of `uint16_t`; the Python wrapper passes x as `float16`. Return the message string the way `build_name` returns one -- read its signature and use the same type.) Export the four with `expert_stream_` prefixes and add `"kernel_layer", "kernel_forward", "kernel_error", "kernel_drop"` to `TEST_ONLY_EXPORTS` (and `"kernel_error": lambda m, h: m.expert_stream_kernel_error()` to `RAW_EXPORTS`). In the transport:

```python
def kernel_layer(kernel: int, spec, *, layout: str = "exl3", variant: Optional[str] = None) -> int:
    """Test only: kernel ``kernel``'s make_layer over ``spec`` (a ``CpuExpertLayerSpec``); returns the layer's id.

    The caller keeps ``spec.keep`` alive until :func:`kernel_drop`. Instrumented build only.
    """
    _refuse_test_only("kernel_layer", variant)
    slabs, params = _layer_tensors(spec)
    return int(_host_module(layout, variant).expert_stream_kernel_layer(
        int(kernel), slabs, int(spec.capacity), int(spec.hidden), int(spec.intermediate), int(spec.activation),
        float(spec.act_limit), params,
    ))


def kernel_forward(layer: int, x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, out: torch.Tensor, *,
                   threads: int, cores: Sequence[int] = (), accumulate: bool = False, layout: str = "exl3",
                   variant: Optional[str] = None) -> tuple[int, str]:
    """Test only: one forward of :func:`kernel_layer`'s ``layer``. Returns (0, "") or the kernel's refusal: (2, why)
    for a bad call, (1, why) for a failure; ``out`` is untouched then. Pins the calling thread to ``cores[0]``.
    Instrumented build only."""
    _refuse_test_only("kernel_forward", variant)
    module = _host_module(layout, variant)
    status = int(module.expert_stream_kernel_forward(
        int(layer), x.to(torch.float16).contiguous(), slots.to(torch.int32).contiguous(), weights.to(torch.float32).contiguous(), out,
        int(threads), torch.tensor(list(cores), dtype=torch.int64), int(bool(accumulate)),
    ))
    return status, (str(module.expert_stream_kernel_error()) if status else "")


def kernel_drop(layer: int, *, layout: str = "exl3", variant: Optional[str] = None) -> None:
    """Test only: release :func:`kernel_layer`'s ``layer``. Instrumented build only."""
    _refuse_test_only("kernel_drop", variant)
    _host_module(layout, variant).expert_stream_kernel_drop(int(layer))
```

(`_host_module(layout, variant)` resolves `variant=None` to the environment's build; the tests pass `variant="instr"` where the default would be prod. The error string is read on the same thread, so the thread-local holds it.)

- [ ] **Step 4: Run.** Commit (`test(cpu-experts): kernel correctness through the host's kernel test exports`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_expert_stream_build_variants.py` -- all pass. `SUITE_EXT` -- pass. `CPU_CHECKS t6` -- `ALL GREEN (check)` (its `slabs` dumps now run through `kernel_forward`, compared with the make_layer baseline).

---

### Task 7: Delete the pool and the ctypes halves of the traits

**Files:**
- Delete: `test/registered/unit/kernels/test_cpu_expert_pool.py`, `test/registered/unit/kernels/test_cpu_experts_abi.py`, `test/manual/dsv41/test_cpu_expert_pool_exl3.py`, `python/sglang/srt/layers/moe/cpu_experts/pool.py`
- Create: `test/registered/unit/kernels/test_cpu_expert_service.py`
- Modify: `python/sglang/srt/layers/moe/cpu_experts/trait.py` (gains the Protocol), `service.py` (imports)
- Modify: `python/sglang/srt/layers/quantization/exl3/schemes/exl3_cpu_experts.py`, `python/sglang/srt/layers/quantization/nvfp4/schemes/nvfp4_cpu_experts.py`, `python/sglang/srt/layers/quantization/nvfp4/ext.py`
- Modify: `test/registered/unit/kernels/test_nvfp4_cpu_build.py` (`test_the_loader_builds_once_per_content_and_reuses_the_library`), `test/manual/dsv41/test_cpu_expert_engines_exl3.py` (helpers move in), `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` (pytest step)

**Interfaces:**
- Consumes: everything above.
- Produces: `sglang.srt.layers.moe.cpu_experts.trait.CpuExpertQuantTrait` (Protocol: `name`, `slab_names`, `act_limit`, `x_dtype`, `weights_dtype`, `out_dtype`, `check_environment()`, `hidden_size(slabs)`, `kernel_address()`, `layer_spec(slabs, capacity)`). Removed: `CpuExpertPool`, `CpuExpertsForwardCall`, `CpuExpertsLayer`, `CpuExpertForward`, `CPU_EXPERTS_*_ABI_VERSION`, `CPU_EXPERTS_MAX_SLABS`; trait methods `register_layer`, `free_layer`, `forward`, `_native`, `native_forward`, `native_keep_warm`, `native_create_engine`, `native_free_engine`; `Exl3CpuParams`, `Nvfp4CpuParams`; `Nvfp4CpuQuantTrait(library=...)`; `nvfp4_cpu_library()`.

- [ ] **Step 1: Split the tests (mechanical move).** Create `test/registered/unit/kernels/test_cpu_expert_service.py` with `register_cpu_ci(est_time=<the old file's>, suite="base-a-test-cpu")`, the old file's imports minus `CpuExpertPool`, and these tests moved verbatim from `test_cpu_expert_pool.py` with the helpers they use (`FakeTrait`, which `FakeServiceTrait` subclasses, minus its `register_layer`, `forward` and `free_layer` methods and the state only they set; `RecordingSlabs` if a moved test still uses it; `FakeServiceTrait`, `FakeHost`, `FakeExt`, the slab helpers, fixtures): every `test_k_star_*`, `test_split_table_*`, `test_service_*`, `test_configured_split_*`, `test_split_from_grid_*`, `test_format_calibration_*`, `test_calibration_*`, `test_a_16_lane_service_*`, `test_the_configured_split_lists_*`, `test_failed_calibration_*`, `test_log_stats_*`, `test_cpu_expert_groups_*`, `test_each_group_*`, `test_exl3_trait_refuses_slabs_the_kernel_would_misaddress`, `test_exl3_trait_reads_a_one_slot_slabs_row_size_not_its_stride`, `test_exl3_trait_refuses_a_kernel_that_would_pin_its_own_workers`, `test_exl3_trait_describes_the_six_slabs_for_make_layer`. Port the moved EXL3 trait tests from `register_layer` to `layer_spec` (they assert on `spec.slabs` / the `ValueError`, not on a ctypes call; `FakeRegisterLayer` goes). Delete, not port: the pool tests (`test_pool_*`, `test_compute_*`, `test_capacity_zero_layer_*`, `test_single_core_pool_*`, `test_failed_registration_frees_*`, `test_close_frees_*`), `test_exl3_trait_registers_the_six_slab_bases`, `test_exl3_trait_keeps_the_slabs_alive_until_free`, `test_exl3_trait_reports_a_refused_registration`. Move `CAP`, `LIMIT` and `_random_slabs` from `test/manual/dsv41/test_cpu_expert_pool_exl3.py` into `test_cpu_expert_engines_exl3.py` (drop its import of them) and delete the former. Every `from sglang.srt.layers.moe.cpu_experts.pool import ...` in the tree becomes `...cpu_experts.trait import ...` (`git grep -l "cpu_experts.pool"` lists them; expect the two scheme files, `service.py`'s docstring and the moved tests).
  Add to the new file the test that pins the deletion:

```python
def test_the_pool_and_the_c_abi_mirrors_are_gone():
    import importlib

    from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertQuantTrait

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sglang.srt.layers.moe.cpu_experts.pool")
    assert {"kernel_address", "layer_spec"} <= set(dir(CpuExpertQuantTrait))
    assert not {"register_layer", "free_layer", "forward", "native_forward"} & set(dir(CpuExpertQuantTrait))
```

- [ ] **Step 2: Run to verify it fails.** `git rm test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_cpu_experts_abi.py test/manual/dsv41/test_cpu_expert_pool_exl3.py`, `git add` the new file and the moved helpers, commit (`test(cpu-experts): service, policy and trait tests leave the pool's file`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_cpu_expert_service.py`. Expected: `test_the_pool_and_the_c_abi_mirrors_are_gone` FAILS (`cannot import name 'CpuExpertQuantTrait'`); every moved test passes.
- [ ] **Step 3: Implement.** Move the `CpuExpertQuantTrait` Protocol into `trait.py` with only the members listed under Produces (its docstring: "One expert format's CPU kernel. ``slab_names`` are the pinned-tier tensors it reads; ``kernel_address`` and ``layer_spec`` are what the RAM-miss service gives the host (``expert_stream/host/cpu_experts/kernel.hpp``); its CPU expert threads run the kernel without Python.") and `git rm python/sglang/srt/layers/moe/cpu_experts/pool.py`. In both scheme files delete the removed methods and ctypes structs (the params are `struct.pack` already) and rewrite the module docstrings: EXL3 "The kernel is the optimized build of exllamav3's CPU MoE kernel, ``csrc/exl3/optimized/kernel.cpp`` (built by SGLANG_DSV41_CPU_EXPERTS=1). ``Exl3CpuQuantTrait`` describes each streamed layer's pinned slabs for the kernel's ``make_layer`` (``layer_spec``) and hands out the kernel's address (the extension's torch op ``sglang_exl3_cpu::kernel_address``)."; NVFP4 likewise with "loaded by ``nvfp4.ext.nvfp4_cpu_module``" and "its tvm-ffi export ``nvfp4_cpu_kernel_address``". `Nvfp4CpuQuantTrait` loses `library` (keeps `module`). In `nvfp4/ext.py` delete `nvfp4_cpu_library` and the `ctypes` import, and update the docstring. `test_the_loader_builds_once_per_content_and_reuses_the_library` uses `nvfp4_cpu_ext.nvfp4_cpu_module.cache_clear()` / `nvfp4_cpu_library_path(str(tmp_path))`. In `run_exl3_cpu_forward_checks.sh`, the pytest step runs `test/manual/dsv41/test_cpu_expert_engines_exl3.py test/registered/unit/kernels/test_cpu_expert_service.py`, and its header comment's "(4) the CPU expert pool tests" becomes "(4) the CPU expert engine and service tests".
- [ ] **Step 4: Run.** `git grep -nE '\b(CpuExpertPool|CpuExpertForward|CpuExpertsForwardCall|CpuExpertsLayer)\b|\b(native_forward|native_keep_warm|native_create_engine|native_free_engine|register_layer|free_layer)\s*\(' -- '*.py'` -- expected: no output (C++ keeps the C ABI until Task 9; the deletion test names the methods only as strings). Commit (`refactor(cpu-experts): delete CpuExpertPool and the traits' ctypes halves`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_cpu_expert_service.py test/registered/unit/kernels/test_nvfp4_cpu_build.py test/registered/unit/kernels/test_nvfp4_cpu_experts.py`, then `POOL=test/registered/unit/kernels/test_cpu_expert_service.py KIFACE_CPU` -- `EXIT=0`. `SUITE_EXT` (now without the deleted file) -- pass. `CPU_CHECKS t7` -- `ALL GREEN (check)`.

---

### Task 8: Benches and native harnesses on the interface; EXL3's upstream API keeps its own table

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.cpp` (table, wrappers), `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h` (`exl3_cpu_table_layer`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/moe_mul1.cpp:20-22,2665-2704`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/cpu_forward.cpp`, `bench/CMakeLists.txt:76-77` (comment)
- Modify: `python/sglang/kernels/jit/csrc/nvfp4/bench/src/cpu_forward.cpp`
- Modify: `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp`, `test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp`

**Interfaces:**
- Consumes: the accessors (Task 1).
- Produces: `namespace sglang::exl3_cpu { __attribute__((visibility("hidden"))) std::shared_ptr<const ::sglang::cpu_experts::CpuExpertLayer> exl3_cpu_table_layer(int64_t handle); }` (throws `std::invalid_argument` for an unknown or freed handle); C++-linkage `void exl3_moe_cpu_baseline_forward(int64_t handle, const at::Half* x, const int32_t* slots, const float* weights, float* out, int rows, int k, int threads, bool accumulate)` and `void exl3_moe_cpu_baseline_set_cores(std::span<const int> cores)` in the vendored kernel (both throw `std::runtime_error`).

- [ ] **Step 1: Port the harnesses (the bit-exact gates are the tests).** The harnesses compare against frozen outputs (`NVFP4_AB`, the bare bench's `--validate-only`, the sanitizer's asserts), so the failing-first step is building them against the interface with the shim still present: do Steps 3-5, then Step 6 runs every gate.
- [ ] **Step 2: EXL3 `kernel.cpp`'s own table.** Replace the shim calls with:

```cpp
namespace {
// Upstream's handle API (exl3_moe_cpu_make_layer / free_layer / forward), which bindings.cpp links: its own table of
// this kernel's layers. A freed entry is reset and its index never reused; a forward holds its layer's shared_ptr, so a
// free racing it only drops the table's reference.
std::mutex g_table_mutex;
std::vector<std::shared_ptr<const ::sglang::cpu_experts::CpuExpertLayer>> g_table;
}  // namespace

namespace sglang::exl3_cpu {
std::shared_ptr<const ::sglang::cpu_experts::CpuExpertLayer> exl3_cpu_table_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_table_mutex);
    if (handle < 0 || handle >= int64_t(g_table.size()) || !g_table[size_t(handle)])
        throw std::invalid_argument("exl3 CPU experts: no table layer " + std::to_string(handle));
    return g_table[size_t(handle)];
}
}  // namespace sglang::exl3_cpu
```

`exl3_moe_cpu_make_layer` pushes `kernel.wrap(std::move(layer))` into `g_table` under the lock and returns its index; `exl3_moe_cpu_free_layer` resets the entry (unknown: no-op, as upstream's free); `exl3_moe_cpu_forward_raw` builds a `ForwardCall` (`accumulate false`, no cores) and calls

```cpp
    try {
        ::sglang::exl3_cpu::exl3_cpu_kernel().forward(*::sglang::exl3_cpu::exl3_cpu_table_layer(handle), call);
    } catch (const std::exception& e) {
        TORCH_CHECK(false, "exl3_moe_cpu_forward: ", e.what());
    }
```

Delete `check_status` and the `#include "../../moe/expert_stream/host/cpu_experts/cabi.hpp"` dependency on `last_error` (the macro line stays until Task 9, so keep the include for now). Declare `exl3_cpu_table_layer` in `kernel.h` (with `<memory>`), hidden, with the comment "The layer exl3_moe_cpu_make_layer registered as `handle` (upstream's per-expert tensor API); valid until exl3_moe_cpu_free_layer(handle) and after it for the holder of the returned reference."
- [ ] **Step 3: The vendored baseline.** In `exl3/moe_mul1.cpp`, drop `#include "optimized/cpu_experts_cabi.h"`, add `#include <span>`, update the header's "Local change" line to "two C++ entry points for sglang's bare-forward bench baseline (the end of this file)", and replace the `extern "C"` section with:

```cpp
// ---- sglang: the bench baseline's entry points (expert_stream/bench/src/cpu_forward.cpp, EXL3_BENCH_BASELINE) ----

// forward_raw over `rows` token rows (x fp16 [rows][hidden]) through layer `handle`, into out (fp32 [rows][hidden]):
// overwritten, or added to when `accumulate`. Routing weights narrow to fp16 as the optimized kernel narrows them. The
// calling thread is the pool's worker 0. Throws std::runtime_error when the forward fails.
void exl3_moe_cpu_baseline_forward(int64_t handle, const at::Half* x, const int32_t* slots, const float* weights,
                                   float* out, int rows, int k, int threads, bool accumulate)
{
    if (rows < 1 || k < 0 || k > 32) throw std::runtime_error("baseline forward: rows or k out of range");
    static thread_local std::vector<at::Half> wts;
    wts.resize(static_cast<size_t>(rows) * k);
    for (size_t i = 0; i < wts.size(); ++i) wts[i] = at::Half(weights[i]);
    forward_raw(handle, x, slots, wts.data(), out, rows, k, threads, accumulate);
}

// The pool's cores, worker i on cores[i % n] (worker 0 is the thread that calls the forward). Before the first
// forward: once the pool has spawned, its workers are already placed, and this throws.
void exl3_moe_cpu_baseline_set_cores(std::span<const int> cores)
{
    if (cores.empty()) throw std::runtime_error("baseline set_cores: no cores");
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_pool.spawned > 0) throw std::runtime_error("baseline set_cores: the pool has spawned");
    g_pool.core_order.assign(cores.begin(), cores.end());
}
```

(check `forward_raw`'s exact parameter list at `moe_mul1.cpp:2470` and match it; it may throw a `c10::Error`, which derives from `std::exception`).
- [ ] **Step 4: The bare EXL3 bench (`bench/src/cpu_forward.cpp`).** Include `kernel.h` instead of `cpu_experts_cabi.h`; replace `g_engine` with `std::vector<int> g_cores;  // the bench's worker cores, worker 0 first`; under `#ifdef EXL3_BENCH_BASELINE` declare the two baseline functions (with `<span>` and `<c10/util/Half.h>`); `Workload::forward(layer)`:

```cpp
  // One full forward on `layer`, writing `output`. Throws if the call fails.
  void forward(size_t layer) {
#ifdef EXL3_BENCH_BASELINE
    exl3_moe_cpu_baseline_forward(handles[layer], static_cast<const at::Half*>(fixture.layers[layer].input.data_ptr()),
                                  slots.data(), weights.data(), output.data(), 1, experts, options.workers, false);
#else
    ::sglang::cpu_experts::ForwardCall call;
    call.rows = 1;
    call.k = experts;
    call.threads = options.workers;
    call.x = fixture.layers[layer].input.data_ptr();
    call.slots = slots.data();
    call.weights = weights.data();
    call.out = output.data();
    call.cores = g_cores;
    ::sglang::exl3_cpu::exl3_cpu_kernel().forward(*::sglang::exl3_cpu::exl3_cpu_table_layer(handles[layer]), call);
#endif
  }
```

and in `main`, `g_cores = cores;` then `#ifdef EXL3_BENCH_BASELINE exl3_moe_cpu_baseline_set_cores(cores); #endif` (the optimized backend needs no setup: the cores ride on each call). Update the file comment ("one full forward through the kernel interface, per backend ... timed from the forward call alone") and the CMake comment at `CMakeLists.txt:76` ("the baseline copy includes the vendored kernel's own headers beside the source").
- [ ] **Step 5: NVFP4 bench and harnesses.** In each of `nvfp4/bench/src/cpu_forward.cpp`, `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp` and `test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp`: include `"kernel.h"` and `"quant.hpp"` (for `SglangNvfp4CpuParams`; the latter replaces `"cpu_experts_cabi.h"`), make layers with `::sglang::nvfp4_cpu::nvfp4_cpu_kernel().make_layer(d, std::as_bytes(std::span<const SglangNvfp4CpuParams>(&params, 1)))` from a `LayerSlabs d` filled with today's descriptor values, and run forwards through `kernel.forward(*layer, call)` with `call.cores` = the harness's core list (where it created an engine). Statuses the harnesses write or assert come from one helper, so the A/B dump stays byte-identical:

```cpp
// The C ABI's status for what a kernel call threw: 0 none, 2 std::invalid_argument, 1 any other exception.
template <class F>
int status_of(F&& f)
{
    try {
        f();
        return 0;
    } catch (const std::invalid_argument&) {
        return 2;
    } catch (const std::exception&) {
        return 1;
    }
}
```

In the sanitizer, `register_layer(...) == 2` asserts become `status_of([&] { kernel.make_layer(d, params); }) == 2`; `forward(nullptr) == 2` is deleted (no null call exists); the keep-warm asserts become `status_of([&] { kernel.keep_warm({}, 2, &word, 0, INT64_MAX); }) == 0`, `... keep_warm({}, 0, ...) == 2`, and the never-created-engine case becomes a repeated core: `const int twice[2] = {0, 0}; assert(status_of([&] { kernel.keep_warm(twice, 1, &word, 0, INT64_MAX); }) == 2);`; `free_layer` calls go (the `unique_ptr` frees). In the A/B harness, `engine_create` failing printed "engine_create refused" -- replace it with a `check_cores`-equivalent `status_of` around the first forward, keeping the dump's format and order unchanged (read the harness's write order before editing; only the means of calling changes).
- [ ] **Step 6: Run.** Commit (`refactor(cpu-experts): benches and harnesses call the kernel interface; upstream's handle API keeps its own table`), `SYNC`. `NVFP4_AB t8` -- both PASS (the harness at this revision against Task 0's dump). `RUN_CPU test/registered/unit/kernels/test_nvfp4_cpu_build.py` -- pass (it builds and runs the sanitizer, plain and asan-ubsan, scalar and avx2). `CPU_CHECKS t8` -- `ALL GREEN (check)` (`bare-validate` now runs the interface; 24 frozen outputs). `BENCH kiface-bench 1` -- `BUILD=0` (it builds `exl3_cpu_baseline` too) and `EXIT=0`. `SUITE_EXT` -- pass (the table path through `exl3_moe_cpu_forward` changed).

---

### Task 9: Delete the C ABI

**Files:**
- Delete: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/cabi.hpp`, `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h`, `python/sglang/kernels/jit/csrc/exl3/optimized/cpu_experts_cabi.h`, `python/sglang/kernels/jit/csrc/nvfp4/optimized/cpu_experts_cabi.h`
- Modify: `exl3/optimized/quant.hpp`, `exl3/optimized/kernel.cpp`, `exl3/optimized/moe_mul1.h:116`, `nvfp4/optimized/quant.hpp`, `nvfp4/optimized/kernel.cpp`
- Modify: `python/sglang/srt/layers/quantization/nvfp4/ext.py:22-24,40`
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/README.txt`, `python/sglang/kernels/jit/csrc/nvfp4/optimized/README.md`, `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/README.txt` (where it names the C ABI), `python/sglang/kernels/jit/csrc/nvfp4/bench/README.md` (same)
- Test: `test/registered/unit/kernels/test_cpu_experts_layout.py`, `test/registered/unit/kernels/test_nvfp4_cpu_build.py`

**Interfaces:**
- Produces: `SglangExl3CpuParams` defined in `exl3/optimized/quant.hpp`, `SglangNvfp4CpuParams` in `nvfp4/optimized/quant.hpp` (global namespace, plain structs, the same fields and comments as today).

- [ ] **Step 1: Write the failing tests.** Replace `test_cpu_experts_layout.py`'s body after `QUANTS`:

```python
STANDARD = {"kernel.h", "quant.hpp", "math.hpp", "math_scalar.hpp", "math_avx2.hpp", "shapes.hpp", "forward_plan.hpp",
            "kernel.cpp"}
FRAMEWORK = CSRC / "moe/expert_stream/host/cpu_experts"


def test_every_quant_has_the_standard_files():
    for quant in QUANTS:
        present = {p.name for p in (CSRC / quant / "optimized").iterdir()}
        assert STANDARD <= present, (quant, sorted(STANDARD - present))


def test_every_quant_declares_its_kernel_accessor():
    for quant in QUANTS:
        header = (CSRC / quant / "optimized/kernel.h").read_text()
        assert f"CpuExpertKernel& {quant}_cpu_kernel();" in header, quant
        assert 'visibility("hidden")' in header, quant


def test_no_cpu_expert_library_exports_a_c_function():
    """The C ABI is gone: no quant source and no framework header declares an extern "C" function."""
    sources = [p for quant in QUANTS for p in (CSRC / quant / "optimized").iterdir()] + list(FRAMEWORK.iterdir())
    offenders = [p.name for p in sources if p.is_file() and 'extern "C"' in p.read_text()]
    assert not offenders, offenders
    assert not (CSRC / "moe/expert_stream/host/cpu_experts_abi.h").exists()
```

Update the module docstring to "Every CPU expert quant has the standard file set and exposes its kernel only through its accessor (kernel.h), so a new quant is a new directory filling in the same files." In `test_nvfp4_cpu_build.py`, delete `C_ABI` and replace `test_build_py_makes_a_library_exporting_the_c_abi` with:

```python
def test_build_py_makes_a_library_exporting_no_c_abi(tmp_path):
    library = _build_module().build(tmp_path / "libnvfp4.so", cxx=CXX)
    symbols = subprocess.run(["nm", "-D", "--defined-only", str(library)], capture_output=True, text=True, check=True).stdout
    assert "sglang_nvfp4_cpu_experts_" not in symbols
    assert "__tvm_ffi_nvfp4_cpu_kernel_address" in symbols
```

(Read the export's actual symbol name from `nm -D` on a Task 3 build first; tvm-ffi prefixes it.)
- [ ] **Step 2: Run to verify they fail.** Commit the tests (`test(cpu-experts): no C ABI remains`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_cpu_experts_layout.py test/registered/unit/kernels/test_nvfp4_cpu_build.py` -- expected FAIL (`cpu_experts_cabi.h` holds `extern "C"`; the library exports `sglang_nvfp4_cpu_experts_*`).
- [ ] **Step 3: Delete.** Move `SglangExl3CpuParams` (with its comment block, minus the sentence about `SglangCpuExpertsLayer`: "A layer's params: make_layer's `params` bytes are this struct.") into `exl3/optimized/quant.hpp` before the `sglang::exl3_cpu` namespace, and `SglangNvfp4CpuParams` likewise into `nvfp4/optimized/quant.hpp`; drop both quant headers' `#include "cpu_experts_cabi.h"` and `moe_mul1.h`'s last line; drop the macro lines and the `cabi.hpp` includes from both `kernel.cpp`s; fix the NVFP4 `static_assert` message to "SlabName indexes LayerSlabs::slabs in the quant's slab order" and quant.hpp's "The descriptor's slabs, in cpu_experts_cabi.h order" to "in the slab order of SglangNvfp4CpuParams' comment"; update `kernel.cpp`'s EXL3 file comment ("This file holds the kernel's accessor and the torch wrappers."). `git rm` the four files. In `nvfp4/ext.py`, drop `_FORWARD_ABI` and its comment and remove it from `_sources()` (the framework glob already covers `kernel.hpp`). Rewrite the README sections that describe the C ABI: each kernel README states that the library's kernel is `CpuExpertKernel` (`host/cpu_experts/kernel.hpp`) behind its accessor, reached from Python through the torch op / tvm-ffi export, layers made by the host's `set_cpu_layer` from `CpuExpertLayerSpec`; keep every measured-provenance section unchanged.
- [ ] **Step 4: The success-criteria greps.** On the laptop:

```bash
git grep -nE '#include .*(cpu_experts_cabi\.h|cabi\.hpp|cpu_experts_abi\.h)|SGLANG_CPU_EXPERTS_DEFINE_CABI\(' -- python test benchmarks
git grep -nE '\b(register_layer|free_layer|engine_create|engine_free)\s*\(|\bSglangCpuExperts(Forward|Layer)\b' -- python test benchmarks
git grep -nE 'mutex|registry' -- python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/expert_forward.hpp
git grep -n 'extern "C"' -- python/sglang/kernels/jit/csrc/exl3 python/sglang/kernels/jit/csrc/nvfp4/optimized python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts
```

Expected: no output from any of the four. The patterns match code (an include, a declaration or call, a type), not prose; `\b` keeps upstream's `exl3_moe_cpu_free_layer(` and the unrelated `register_layer_transfer_counter(` out. If a comment or docstring still matches, reword it (it describes the deleted ABI) rather than widening the pattern; docs/ and analysis/ are history and are not searched.
- [ ] **Step 5: Run.** Commit (`refactor(cpu-experts): delete the C ABI`), `SYNC`, `RUN_CPU test/registered/unit/kernels/test_cpu_experts_layout.py test/registered/unit/kernels/test_nvfp4_cpu_build.py test/registered/unit/kernels/test_nvfp4_cpu_experts.py test/registered/unit/kernels/test_cpu_experts_common.py` -- all pass. `NVFP4_AB t9`, `CPU_CHECKS t9`, `BENCH kiface-bench 1`, `SUITE_EXT` -- green.

---

### Task 10: Final gates

**Files:** none (a red gate gets a fix commit in the task that owns the code, then this task reruns).

- [ ] **Step 1:** `SYNC` at the branch head; check `sglang.__file__`.
- [ ] **Step 2:** `POOL=test/registered/unit/kernels/test_cpu_expert_service.py KIFACE_CPU test/registered/unit/kernels/test_expert_stream_build_variants.py` -- `EXIT=0`. Compare the counts with Task 0's: the deltas are the deleted files' tests (`test_cpu_expert_pool.py`'s pool and free tests, `test_cpu_experts_abi.py`, the NVFP4 engine tests) and this plan's new tests; list both in the ledger.
- [ ] **Step 3:** `CPU_CHECKS final`, `NVFP4_AB final`, `SUITE_EXT`, `BENCH kiface-bench 1`, `BENCH kiface-bench-n2 2` -- all green.
- [ ] **Step 4:** `RUN_GPU test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_cpu_split_calibration_cuda.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_dsv41_layer_fusion_gpu.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py` -- `EXIT=0` (75 is a lock timeout: rerun).
- [ ] **Step 5:** Record every gate's final line with its command in the ledger, then remove the scratch outputs `kiface-cpu-*`, `kiface-nvfp4-t*` (keep `kiface-nvfp4-base` and `nlane-cpu-base`) and the worktree `wt-kiface-base` if Task 0 made it.
