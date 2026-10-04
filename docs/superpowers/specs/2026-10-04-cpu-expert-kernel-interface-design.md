# CPU expert kernel interface: replace the C ABI with a C++ virtual interface

Status: approved in conversation 2026-10-04. Branch `cpu-kernel-iface`, cut from `numa-p2` at 53963aa379 (Phase 2 of
the NUMA node distributor plus the GCC 15 / resolved `-march` JIT change).

## Why

Three separately built libraries meet at a hand-written C ABI:

- the expert-stream host (JIT tvm-ffi module, `moe/expert_stream/host/`), which runs CPU lanes on `CpuExpertEngine`;
- the EXL3 CPU kernel (torch extension, GCC 15, `-Ofast -march=native`, `quantization/exl3/ext.py`);
- the NVFP4 CPU kernel (ctypes `.so`, baseline x86-64 with runtime ISA dispatch, `-ffp-contract=off` as part of its
  bitwise contract, `quantization/nvfp4/ext.py` + `build.py`).

Each kernel library exports six `extern "C"` functions (`SGLANG_CPU_EXPERTS_DEFINE_CABI`, `cabi.hpp`), declared in
`exl3/optimized/cpu_experts_cabi.h` and `nvfp4/optimized/cpu_experts_cabi.h`, over the C structs of
`expert_stream/host/cpu_experts_abi.h`. Python glues them with ctypes (`moe/cpu_experts/pool.py` and the two scheme
traits) and hands raw function-pointer addresses to the host.

Since CPU lanes now run from `CpuExpertEngine` (one per NUMA group, Phase 2) rather than from Python, the C ABI's
extras are dead weight in production:

- `ExpertForward<Quant>` keeps a layer registry (`std::vector<std::shared_ptr<const Layer>>`, `registry_mutex`) and a
  `std::shared_mutex layer_mutex` so `free_layer` cannot free a layer a forward reads (status 3). Production registers
  each row once (`CpuExpertService.register`) and never frees one.
- `Engines::create` / `engine_free` (`team.hpp`) keep a per-library registry of core lists, though the host already
  holds each group's cores (`CpuExpertConfig::cores`).
- Each group's `CpuExpertEngine` keeps its own copy of every row's integer handle (`handles_`).

## Decision: a non-template virtual interface (not `CpuExpertEngine<Quant>`)

Templating the engine on the quant would compile both kernels into the expert-stream module: one flag set for EXL3
(`-Ofast`) and NVFP4 (`-ffp-contract=off`, baseline arch), both kernels rebuilt for every Layout / node-count / variant
build, and the template parameter propagated through `RamTier`, `NumaGroup` and `CopyEngine` (which hold
`CpuExpertEngine*`). A virtual interface keeps each kernel in its own library with its own flags. It crosses `.so`
boundaries, which needs one C++ ABI and the shared libstdc++ on both sides; the GCC 15 JIT change provides that, and
the interface's signatures use only plain structs, `std::span`, `std::array` and `std::unique_ptr` (no `std::string`,
no `std::vector`).

## The interface

New header `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/kernel.hpp`, replacing
`cpu_experts_abi.h` and `cpu_experts/cabi.hpp`:

```cpp
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
  // Validates (today's register_layer checks) and stores views; throws std::invalid_argument. `params` is the quant's
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

Behaviour carried over unchanged: every validation today's `register_layer` and `forward` perform (capacity, slab
count, optional slabs, minimum slot bytes, rows <= kMaxRows, k <= kMaxRoutes, threads in [1, 4096] and not above
`cores.size()` when cores are given, slot range, `check_slot`, finite weights), the ISA detection and report
(`isa()`), the RouteTable, the pinning semantics of `CallCores`, and each quant's numerics (bitwise).

## The quant side

`cpu_experts/expert_forward.hpp`: `ExpertForward<Quant>` becomes `final : CpuExpertKernel`. Its nested
`Layer final : CpuExpertLayer` holds `typename Quant::Layer`. `forward` checks `&layer.kernel() == this`, then
`static_cast`s (no RTTI across libraries). Deleted: the `layers` vector, `registry_mutex`, `layer_mutex`, status 3,
integer handles, `free_layer`, `last_error()`/`fail()` status plumbing, `Engines` (create/destroy/find) and
`cabi.hpp`. `Quant::validate`, `min_slot_bytes`, `make_layer` and `dispatch` take `LayerSlabs` / `ForwardCall`
instead of the C structs; `Quant::Params` stays the quant's own struct (`SglangExl3CpuParams`, `SglangNvfp4CpuParams`),
moved out of the deleted `cpu_experts_cabi.h` headers into each quant's `quant.hpp`. Non-zero `dispatch` statuses
become exceptions in `ExpertForward::forward`.

Each kernel library defines one accessor, `const CpuExpertKernel& exl3_cpu_kernel()` / `nvfp4_cpu_kernel()`, a
function-local static `ExpertForward<Quant>`.

## Reaching the kernel from Python

No `extern "C"` functions remain. Each quant's own binding returns its kernel object's address as an int64:

- EXL3: the torch extension gains an op returning `reinterpret_cast<int64_t>(&exl3_cpu_kernel())`. The extension's
  other entry points that used handles (`exl3_moe_cpu_make_layer` / `exl3_moe_cpu_free_layer` style ops in
  `exl3/moe_mul1.cpp`, `exl3/optimized/kernel.cpp`, `moe_mul1.h`) are ported or removed with their callers.
- NVFP4: the ctypes library becomes a tvm-ffi module with one typed export doing the same (the build in
  `quantization/nvfp4/build.py` links tvm-ffi as the expert-stream module does; `nvfp4/ext.py` loads it with tvm-ffi).

## The host side

`expert_stream/host/cpu_experts.h`:

- `CpuExpertConfig` loses `forward`, `keep_warm` (function pointers) and `engine`; gains
  `const cpu_experts::CpuExpertKernel* kernel`. A keep-warm is on when `keep_warm_ns > 0`.
- New `CpuExpertLayers`, one per `RamTier`, shared by every group's engine (replaces each engine's `handles_`):
  `set(row, std::unique_ptr<CpuExpertLayer>)` once per row from any thread (release), `get(row)` (acquire; nullptr
  means not eligible). It owns the layers and is destroyed after every group's engine has stopped. No lock on the
  forward path: a layer is never freed while an engine runs.
- `run()` calls `kernel->forward(*layer, call)` (cores = the config's cores) and `kernel->keep_warm(...)`, catching
  `std::exception` into `fail_stop(prefix + e.what())`.

`ffi_exports.h`: `enable_cpu_experts` takes the kernel address instead of `forward`/`engine`/`keep_warm` addresses;
`set_cpu_layer(row, ...)` takes the row's slab tensors plus the params bytes and calls `kernel->make_layer` itself
(once per row; every group's engine sees it through the shared `CpuExpertLayers`).

Test-only (`ffi_test_exports.h`): the fake forward and fake keep-warm (`test_forward_address`,
`test_keep_warm_address`, `test_keep_warm_calls`, `test_keep_warm_engine`) become one `FakeKernel : CpuExpertKernel`
behind `test_kernel_address(ns_per_expert)`, keeping the fakes' observable behaviour, and stay refused by the prod
build like every test-only export.

## Python

- `moe/cpu_experts/pool.py`: delete `CpuExpertPool`, `CpuExpertsForwardCall`, `CpuExpertsLayer`, the
  `CpuExpertForward` CFUNCTYPE and the trait methods `register_layer`/`free_layer`/`forward`. What remains of the trait
  (slab names, act_limit, the kernel address, the params bytes) stays or moves to where the service needs it.
- `quantization/exl3/schemes/exl3_cpu_experts.py`, `quantization/nvfp4/schemes/nvfp4_cpu_experts.py`: no ctypes
  symbol lookups; the trait yields the kernel address and its params bytes.
- `moe/cpu_experts/service.py` / `CpuExpertGroups` / `threading_config.py`: no `engine_create`/`engine_free`; pass
  the kernel address to `enable_cpu_experts` and the slabs + params to `set_cpu_layer`.

## Benches and tests

- C++ benches link the kernel library and use the interface directly: `expert_stream/bench/src/{cpu_forward,
  full_stack,self_test,stack_fixture}.{cpp,h}` and `CMakeLists.txt`, `nvfp4/bench/src/cpu_forward.cpp`,
  `test/manual/dsv41/nvfp4_cpu_forward_ab.cpp`, `test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp`.
- The shared-framework toy (`cpu_experts_common_toy.hpp`, `cpu_experts_common_toy_lib.cpp`,
  `cpu_experts_common_check.cpp`, `test_cpu_experts_common.py`) ports to the interface: it is the framework's test.
- Deleted with the ABI: `test_cpu_experts_abi.py`, `test_cpu_expert_pool.py`, `test/manual/dsv41/
  test_cpu_expert_pool_exl3.py`; tests of free_layer/status-3/engine registry semantics are deleted, not ported.
- Kernel-correctness tests (`test_nvfp4_cpu_experts.py`, `test_nvfp4_cpu_build.py`, `test_cpu_experts_layout.py`,
  `test/manual/dsv41/{exl3_cpu_forward_ab.py,test_cpu_expert_engines_exl3.py,test_exl3_cpu_lane_order_cuda.py}`)
  call the kernel through a test export (the host module's or the kernel library's) instead of ctypes.
- RAM-miss tests that inject a Python CFUNCTYPE forward (`test_exl3_ram_miss_cpu_experts.py`,
  `test_exl3_ram_miss_numa_completion.py`, others found by grep for `CpuExpertForward` / `test_forward_address`) use
  `test_kernel_address`.
- Docs that describe the C ABI (`exl3/optimized/README.txt`, `nvfp4/optimized/README.md`) are updated.

## Success criteria

- No `extern "C"` CPU-expert function, `cpu_experts_cabi.h`, `cabi.hpp` or `cpu_experts_abi.h` remains; `git grep`
  for `register_layer`, `free_layer`, `engine_create`, `engine_free`, `SglangCpuExpertsForward`,
  `SglangCpuExpertsLayer` in code finds nothing.
- `expert_forward.hpp` has no mutex and no registry.
- Both kernels' outputs are bitwise unchanged (existing reference tests and the bench self-tests).
- The CPU suite (`p2-cpu-par.sh` selection, with `CXX` = GCC 15), CPU_CHECKS against `nlane-cpu-base`, SUITE_EXT and
  BENCH at 1 and 2 nodes are green; GPU suite once at the end.
