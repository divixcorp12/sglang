# Expert-stream full-stack CPU-expert benchmark: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Google Benchmark executable, in a `prod` and an `instr` build, in which a C++ writer posts CPU-hit requests
into the lease lanes. The real `RamTier`, `RamThread` and `CpuExpertEngine`, with the optimized EXL3 kernel, serve
those requests. The executable reports post→CopyDone latency next to the bare kernel call, as `overhead_p50_us` per
expert count.

**Architecture:**
- `DeviceSim` is a C++ port of `ChainSim`'s device role. It writes seqlocked records into the request page, applies
  the host's map deltas, types lanes like `ram_slot_map.type_lanes`, closes the copy gate and spins on CopyDone.
- `Stack<Build>` owns the unmodified host stack, with the copy engine on its host backend, so CUDA is never loaded.
- `StackFixture` builds real EXL3 slabs and row-image files from the existing eight-layer fixture.
- `full_stack.cpp` validates every output bit-exactly against the frozen references, then times `BM_bare` and
  `BM_stack`.

**Tech Stack:** C++20, GCC 15, CMake, Google Benchmark 1.9.4, ATen/c10 (fixture tensors only), the tvm-ffi headers
(needed by the host headers), liburing, and bash for the runner and the systemd service files.

**Spec:** `docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md`

## Global Constraints

- All new code lives under `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/`. No production header changes
  (spec, "Non-goals"). If a needed hook is missing, stop and ask.
- Build with GCC 15 (`/opt/rh/gcc-toolset-15/root/usr/bin/g++`), C++20, Linux only. That is the existing
  `CMakeLists.txt` gate.
- Placement defaults:
  - writer `16`;
  - `RamThread` service `17`, busy-poll;
  - copy thread `52`;
  - CPU experts `18-33`;
  - host node `0`, worker node `1`.
- The isolated partition becomes `16-33,52-69` (spec, "Partition change").
- One request in flight, CPU hits only (`kKindHitCpu`), `split[n] = n`. Experts `0..k-1` for `k ∈ {1, 3, 5}`,
  weights `0.071234 + 0.23*i/(k-1)` (`cpu_forward.cpp`'s), act limit `10.0`, 8 layers rotated.
- Before timing, every output is checked bit-exactly against `reference-e{k}.bin` through the stack and bare, 48 in
  all. After each timed benchmark, its 8 outputs are checked again.
- Code is written on the laptop, committed, pushed, and built and run in the divix01 worktree `wt-fullstack`. Never
  build or run in the production checkout. Never copy trees to divix01.
- Commits stage files by name. No amend, rebase or force-push. Each commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
  ```
- Run nothing on cores 16-33 outside `exl3bench.service`. Off-service runs use the node-0 placement below.
- Read every remote command's own exit status (`echo EXIT=$?` right after it). Never read it through a pipe.

## Spec rulings made in this plan (review these)

1. **Copy-thread placement.** The spec doesn't place the copy thread, which inherits the affinity of the thread that
   enables it. The `RamThread` watchdog inherits the same way. The plan adds `--copy-cpu`, default 52: the writer's
   SMT sibling, node 0, inside the partition. `Stack` enables the copy engine and starts `RamThread` while pinned
   there. Production's copy thread inherits the server's affinity.
2. **`BM_bare` runs on worker 0's core, not the writer's.** The kernel's caller is worker 0. Called from the writer
   core it would compute on node 0 against node-1 slabs, which differs from the stack's engine thread on 18.
3. **The writer opens the gate itself once it sees CopyDone.** This is the closed→open CAS that CW does
   (`row_copy_kernels.cuh`). The spec says the writer does not. But when CopyDone lands before the close, nothing else
   opens the gate, and the watchdog would abort the process 2 s later.
4. **`DeviceSim` never types `kKindHitCopy`.** The bench has no copy table, so non-CPU hits are `kKindHitSm`, as with
   `hit_copy="sm"`.
5. **Extra options `--copy-cpu`, `--host-node` and `--worker-node`.** These let validation run on node 0 off the
   production cores (Tasks 4 and 8).
6. **The writer's deadline is half of `--wait-timeout-ms`.** The watchdog's copy-wait timeout is the full value, so the
   writer prints its diagnostics before the watchdog can abort.
7. **Row images are reused when a stamp sidecar matches** (fixture path, size, mtime, image size). The spec says they
   are written at setup; rewriting 533 MB in each of the 16 processes per run is avoidable. Bit-exact checks back the
   reuse up.
8. **Mutants.** A wrong slot is refused by the host's own check, as a fail-stop rather than a bit-exact failure.
   Task 8 records that, and adds a weight-swap mutant for the bit-exact path. The "drop the head store" mutant drops
   it for captured posts only, so the loads still work and the copy-wait deadline is what fires.
9. **`--self-test` pins its threads.** The real `RamThread` busy-polls a pinned core. Defaults are writer 0, service
   1, copy 2, workers `3`; run it under `taskset -c 0-15`.
10. **Extra files.** `placement.{h,cpp}`, `row_images.{h,cpp}`, `self_test.{h,cpp}` and `aligned.h` are added for
    focused units. `stack_fixture.h` exposes no ATen type, so no translation unit includes both ATen and the tvm-ffi
    headers.
11. **The full-stack targets build only when `EXL3_TVM_FFI_ROOT` is set**, so the existing bench's configure is
    unchanged.
12. **`service/install.sh` is unchanged.** It names no CPUs; the partition lives in `exl3bench-isolation`,
    `exl3bench-run`, `exl3bench.service` and `service/README.txt`.

## Review Focus

1. **Stale row images.** Images of the right size left by another fixture or layout must be rewritten, not reused.
   Tested by `test_image_stamp` (Task 3).
2. **CopyDone before the close.** When CopyDone is stored before the writer closes the gate, the writer must open the
   gate itself; otherwise the watchdog aborts. Tested by `test_copy_wait_gate` (Task 2).
3. **A post while the row's delta is still owed.** It must wait, then fail at its deadline with a message naming the
   delta, never type lanes from a stale map. Tested by `test_owed_delta` (Task 2).
4. **Bad placements.** Duplicate roles, a CPU outside the allowed set, or a role on the service core's SMT sibling must
   be refused before any setup. Tested by `test_placement` (Task 1).
5. **A failed process in one round.** The runner must still run every round, record each exit status, exit nonzero,
   and refuse to reuse a results directory. Tested by the stub-binary run in Task 6.

---

## Conventions used by every task

Paths:

```
WT=/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-wt-cpubench      # laptop worktree, branch expert-stream-cpu-bench
BENCH=python/sglang/kernels/jit/csrc/moe/expert_stream/bench      # relative to the repo root
R=/data/models/slang/nvfp4-work/wt-fullstack                       # divix01 private worktree (created in Task 1)
B=/data/models/slang/nvfp4-work/fullstack-build                    # divix01 build directory (created in Task 1)
```

**SYNC+BUILD** (push, move the divix01 worktree to the pushed commit, build both binaries):

```bash
git -C $WT push origin expert-stream-cpu-bench
ssh divix01 'R=/data/models/slang/nvfp4-work/wt-fullstack; B=/data/models/slang/nvfp4-work/fullstack-build
  git -C $R fetch -q origin && git -C $R checkout -q --detach origin/expert-stream-cpu-bench && git -C $R log -1 --oneline
  taskset -c 0-63 cmake --build $B -j16 --target exl3_full_stack_prod exl3_full_stack_instr > $B/last-build.log 2>&1
  echo BUILD=$?; tail -25 $B/last-build.log'
```

**SELFTEST** (both builds; no fixture; synthetic rows):

```bash
ssh divix01 'B=/data/models/slang/nvfp4-work/fullstack-build
  for v in prod instr; do
    taskset -c 0-15 $B/exl3_full_stack_$v --self-test --image-dir=/data/models/slang/nvfp4-work/fullstack-selftest
    echo "$v EXIT=$?"
  done'
```

**VALIDATE** (both builds; real fixture; node-0 placement, off the production cores; no timing):

```bash
ssh divix01 'B=/data/models/slang/nvfp4-work/fullstack-build
  export EXL3_MOE_CPU_MAX_ISA=bw OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  for v in prod instr; do
    taskset -c 0-15,36 $B/exl3_full_stack_$v --validate-only --writer-cpu=0 --service-cpu=1 --copy-cpu=36 \
      --cpus=2-15 --host-node=0 --worker-node=0 --image-dir=/data/models/slang/nvfp4-work/fullstack-images
    echo "$v EXIT=$?"
  done'
```

Commit template (stage by name, HEREDOC message):

```bash
git -C $WT add <files...>
git -C $WT commit -F - <<'EOF'
<subject>

<body>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

C++ "tests" are checks in `--self-test` (`src/self_test.cpp`). Each failed check prints
`FAIL file:line: expression`. The run ends with `self-test (<build>): N checks, M failed` and exits nonzero when
M > 0. A red step that cannot link (functions declared, not yet defined) is the expected first failure for new units.

## File structure

All under `$BENCH`:

| File | Responsibility |
|---|---|
| `CMakeLists.txt` (modify) | `exl3_full_stack_{prod,instr}` when `EXL3_TVM_FFI_ROOT` is set |
| `src/aligned.h` | `AlignedBuffer`, `aligned_zeroed`, `round_up` (first-touch allocations) |
| `src/placement.{h,cpp}` | CPU-list parsing, placement validation, pinning, thread-affinity verification |
| `src/device_sim.{h,cpp}` | The device's side of the lease protocol; `load_experts` |
| `src/row_images.{h,cpp}` | Row-image layout, image files (stamp-gated), `RowSet`, the O_DIRECT probe |
| `src/stack.h` | `image_tables`, `StackConfig`, `Stack<Build>` (the real tier and thread) |
| `src/stack_fixture.{h,cpp}` | EXL3 slabs, images, x/out rows and layer registration from `Fixture`; no ATen in the header |
| `src/self_test.{h,cpp}` | `--self-test`: placement, `DeviceSim`, images, and the real stack with a fake forward |
| `src/full_stack.cpp` | Options, setup, 48 bit-exact checks, `BM_bare` / `BM_stack`, counters |
| `run_full_stack.sh` | Process rounds, the results record, exit statuses |
| `README.txt` (modify) | "Full-stack bench" section |
| `service/exl3bench-isolation`, `service/exl3bench-run`, `service/exl3bench.service`, `service/README.txt` (modify) | Partition `16-33,52-69` |

---

### Task 1: Build scaffold, placement, and the self-test harness

**Files:**
- Modify: `$BENCH/CMakeLists.txt` (append after the existing `foreach`/`endforeach`)
- Create: `$BENCH/src/aligned.h`, `$BENCH/src/placement.h`, `$BENCH/src/placement.cpp`, `$BENCH/src/self_test.h`,
  `$BENCH/src/self_test.cpp`, `$BENCH/src/full_stack.cpp`

**Interfaces:**
- Consumes: `sglang::expert_stream::core_siblings(int)` from `expert_stream/host/core_topology.h` (plain header, no
  tvm-ffi).
- Produces (namespace `fullstack`):
  - `int number(const std::string&)`;
  - `std::vector<int32_t> parse_cpus(const std::string&)`;
  - `struct Placement { int writer, service, copy; std::vector<int32_t> workers; int host_node, worker_node; }`;
  - `struct Topology { std::function<int(int)> node_of; std::function<std::vector<int>(int)> siblings_of; cpu_set_t allowed; }`;
  - `Topology system_topology()`;
  - `void validate_placement(const Placement&, const Topology&, bool check_nodes)`;
  - `void pin_self(int cpu)`;
  - `class PinScope { explicit PinScope(int cpu); }`;
  - `std::set<int> task_ids()`;
  - `std::vector<int> expected_threads(const Placement&)`;
  - `void verify_threads(const std::set<int>& before, std::vector<int> expected)`;
  - `std::string cpu_list(std::vector<int>)`;
  - `AlignedBuffer aligned_zeroed(int64_t bytes, int64_t align = 4096)`;
  - `constexpr int64_t round_up(int64_t, int64_t)`;
  - `int run_self_test(const Placement&, const std::filesystem::path& image_dir)`.
  - Self-test harness: `CHECK(cond)` and `CHECK_THROWS(expr, "needle")` in `self_test.cpp`.

- [ ] **Step 1: Write the CMake targets**

Append to `$BENCH/CMakeLists.txt`, after the final `endforeach()`:

```cmake
# The full-stack bench (docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md). The expert-stream
# host headers need tvm-ffi's headers (reader_base.h) and liburing; nothing loads CUDA (the copy engine's host
# backend). Built only when EXL3_TVM_FFI_ROOT names the tvm_ffi package, so the CPU-forward bench configures as before.
set(EXL3_TVM_FFI_ROOT "" CACHE PATH "The tvm_ffi Python package directory (include/ and lib/), for the full-stack bench")
if(EXL3_TVM_FFI_ROOT)
  find_path(TVM_FFI_INCLUDE tvm/ffi/function.h PATHS "${EXL3_TVM_FFI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_path(DLPACK_INCLUDE dlpack/dlpack.h PATHS "${EXL3_TVM_FFI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(TVM_FFI tvm_ffi PATHS "${EXL3_TVM_FFI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(URING uring REQUIRED)
  get_filename_component(MOE_CSRC "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  if(NOT EXISTS "${MOE_CSRC}/expert_stream/host/ram_thread.h")
    message(FATAL_ERROR "Cannot locate the expert-stream host headers: ${MOE_CSRC}")
  endif()
  set(FULL_STACK_SOURCES src/full_stack.cpp src/self_test.cpp)
  foreach(VARIANT prod instr)
    set(TARGET exl3_full_stack_${VARIANT})
    if(VARIANT STREQUAL "instr")
      set(INSTR 1)
    else()
      set(INSTR 0)
    endif()
    add_executable(${TARGET} ${FULL_STACK_SOURCES})
    target_include_directories(${TARGET} PRIVATE "${QUANT}/optimized" "${MOE_CSRC}")
    # tvm-ffi's dlpack first: the host headers need its DLPack, not any copy under the torch include tree.
    target_include_directories(${TARGET} SYSTEM PRIVATE "${TVM_FFI_INCLUDE}" "${DLPACK_INCLUDE}"
      "${ATEN_INCLUDE}" "${ATEN_INCLUDE}/torch/csrc/api/include")
    target_compile_definitions(${TARGET} PRIVATE
      EXL3_MOE_CPU_ACT_RESIDUAL=1 EXL3_MOE_CPU_ACT_BLOCK=128
      EXL3_FULL_STACK_INSTR=${INSTR} EXL3_BENCH_BACKEND="full-stack-${VARIANT}")
    target_compile_options(${TARGET} PRIVATE -O3 -g)
    target_link_libraries(${TARGET} PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX Threads::Threads
      "${TORCH_CPU}" "${C10}" "${TVM_FFI}" "${URING}")
    set_target_properties(${TARGET} PROPERTIES BUILD_RPATH "${EXL3_TORCH_ROOT}/lib;${EXL3_TVM_FFI_ROOT}/lib")
  endforeach()
else()
  message(STATUS "Full-stack bench skipped: set EXL3_TVM_FFI_ROOT to build exl3_full_stack_{prod,instr}")
endif()
```

- [ ] **Step 2: Write `src/aligned.h`**

```cpp
// First-touch host allocations: zeroed by the allocating thread, so its NUMA node backs the pages.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace fullstack {

struct FreeDeleter {
  void operator()(uint8_t* p) const {
    std::free(p);
  }
};
using AlignedBuffer = std::unique_ptr<uint8_t[], FreeDeleter>;

constexpr int64_t round_up(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

inline AlignedBuffer aligned_zeroed(int64_t bytes, int64_t align = 4096) {
  const auto size = static_cast<size_t>(round_up(bytes, align));
  void* p = std::aligned_alloc(static_cast<size_t>(align), size);
  if (p == nullptr) throw std::bad_alloc();
  std::memset(p, 0, size);
  return AlignedBuffer(static_cast<uint8_t*>(p));
}

}  // namespace fullstack
```

- [ ] **Step 3: Write `src/placement.h`**

```cpp
// Where the bench's threads run (spec, "Placement"), the refusals before setup, and the affinity check after it.
#pragma once

#include <sched.h>

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace fullstack {

int number(const std::string& text);
std::vector<int32_t> parse_cpus(const std::string& text);  // "16-17,52": ranges and singles, no duplicates

struct Placement {
  int writer = -1;   // the GPU stand-in: posts records, spins on CopyDone
  int service = -1;  // RamThread, busy-polling: a physical core of its own
  int copy = -1;     // the copy engine thread and RamThread's watchdog (both inherit the enabling thread's affinity)
  std::vector<int32_t> workers;  // CPU experts: worker 0 (the CPU expert thread, the kernel's caller), then helpers
  int host_node = 0;    // writer, service and copy
  int worker_node = 1;  // every worker
};

struct Topology {
  std::function<int(int)> node_of;                  // a CPU's NUMA node, -1 when unknown
  std::function<std::vector<int>(int)> siblings_of; // a CPU's SMT siblings, itself included
  cpu_set_t allowed;                                // the process's CPUs
};

Topology system_topology();  // sysfs and sched_getaffinity

// Throws, naming the CPU and the rule: duplicate roles, a CPU outside `allowed`, a role on the service CPU's SMT
// sibling, and (check_nodes) a role off its node.
void validate_placement(const Placement& placement, const Topology& topology, bool check_nodes);

void pin_self(int cpu);  // the calling thread only

// Pins the calling thread to `cpu` and restores its previous affinity on destruction. Threads created inside the
// scope inherit `cpu`.
class PinScope {
 public:
  explicit PinScope(int cpu);
  ~PinScope();
  PinScope(const PinScope&) = delete;
  PinScope& operator=(const PinScope&) = delete;

 private:
  cpu_set_t saved_;
};

std::set<int> task_ids();
// The CPUs the threads created during setup must be pinned to, one each: the service, the copy thread and the
// watchdog (both on `copy`), the CPU expert thread (workers[0]) and the kernel's helpers (workers[1..]).
std::vector<int> expected_threads(const Placement& placement);
// Every thread not in `before`, except io_uring's kernel workers ("iou-*"), must be pinned to exactly one CPU, and
// those CPUs, sorted, must equal `expected` sorted. Throws otherwise.
void verify_threads(const std::set<int>& before, std::vector<int> expected);
std::string cpu_list(std::vector<int> cpus);  // "16,17,52"

}  // namespace fullstack
```

- [ ] **Step 4: Write the failing self-test: `src/self_test.h` and `src/self_test.cpp` (placement checks)**

`src/self_test.h`:

```cpp
#pragma once

#include <filesystem>

#include "placement.h"

namespace fullstack {

// --self-test: placement rules, DeviceSim's records and typing on a standalone page, row images, then the real stack
// (this binary's build) on synthetic rows with a fake forward. Returns the number of failed checks.
int run_self_test(const Placement& placement, const std::filesystem::path& image_dir);

}  // namespace fullstack
```

`src/self_test.cpp`:

```cpp
#include "self_test.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace fullstack {
namespace {

int checks = 0;
int failures = 0;

void check(bool ok, const char* what, const char* file, int line) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
  }
}

template <class F>
void check_throws(F&& f, const std::string& needle, const char* what, const char* file, int line) {
  try {
    f();
  } catch (const std::exception& error) {
    const bool found = std::string(error.what()).find(needle) != std::string::npos;
    check(found, what, file, line);
    if (!found) std::fprintf(stderr, "  message: %s\n", error.what());
    return;
  }
  check(false, what, file, line);
}

#define CHECK(cond) check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_THROWS(expr, needle) check_throws([&] { expr; }, needle, #expr " throws " needle, __FILE__, __LINE__)

// ---- placement ----

// divix01: node 0 = 0-17,36-53; node 1 = 18-35,54-71; c and c + 36 are SMT siblings. Allowed: the partition.
Topology fake_topology() {
  Topology t;
  t.node_of = [](int cpu) { return cpu % 36 < 18 ? 0 : 1; };
  t.siblings_of = [](int cpu) { return std::vector<int>{cpu % 36, cpu % 36 + 36}; };
  CPU_ZERO(&t.allowed);
  for (int cpu = 16; cpu <= 33; ++cpu) {
    CPU_SET(cpu, &t.allowed);
    CPU_SET(cpu + 36, &t.allowed);
  }
  return t;
}

Placement production_placement() {
  Placement p;
  p.writer = 16;
  p.service = 17;
  p.copy = 52;
  p.workers = parse_cpus("18-33");
  return p;
}

bool passes(const Placement& p, const Topology& t, bool check_nodes) {
  try {
    validate_placement(p, t, check_nodes);
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "  refused: %s\n", error.what());
    return false;
  }
}

void test_placement() {
  CHECK(parse_cpus("16-17,52") == std::vector<int32_t>({16, 17, 52}));
  CHECK_THROWS(parse_cpus("3,3"), "Duplicate CPU");
  CHECK_THROWS(parse_cpus("5-3"), "Invalid CPU range");
  CHECK_THROWS(parse_cpus("x"), "Invalid integer");
  const Topology t = fake_topology();
  CHECK(passes(production_placement(), t, true));

  Placement twice = production_placement();
  twice.service = 16;
  CHECK_THROWS(validate_placement(twice, t, true), "two roles");

  Placement sibling = production_placement();
  sibling.copy = 53;  // 17's SMT sibling
  CHECK_THROWS(validate_placement(sibling, t, true), "physical core of the service CPU 17");

  Placement outside = production_placement();
  outside.copy = 34;
  CHECK_THROWS(validate_placement(outside, t, true), "outside the process's allowed CPUs");

  Placement writer_node = production_placement();
  writer_node.writer = 19;
  writer_node.workers = parse_cpus("18,20-33");
  CHECK_THROWS(validate_placement(writer_node, t, true), "writer CPU 19 is on NUMA node 1, not node 0");
  CHECK(passes(writer_node, t, false));  // the self-test's mode: no node rules

  Topology wider = fake_topology();
  CPU_SET(10, &wider.allowed);
  Placement worker_node = production_placement();
  worker_node.workers.back() = 10;
  CHECK_THROWS(validate_placement(worker_node, wider, true), "worker CPU 10 is on NUMA node 0, not node 1");

  std::vector<int> expected = expected_threads(production_placement());
  std::vector<int> want = {17};  // sorted: service, workers 18-33, then the copy CPU twice (copy thread, watchdog)
  for (int cpu = 18; cpu <= 33; ++cpu) want.push_back(cpu);
  want.push_back(52);
  want.push_back(52);
  CHECK(expected == want);
  CHECK(cpu_list({52, 16, 17}) == "16,17,52");
}

}  // namespace

int run_self_test(const Placement& placement, const std::filesystem::path& image_dir) {
  (void)placement;
  (void)image_dir;
  test_placement();
  std::fprintf(stderr, "self-test: %d checks, %d failed\n", checks, failures);
  return failures;
}

}  // namespace fullstack
```

- [ ] **Step 5: Write `src/full_stack.cpp` (options, placement, self-test dispatch)**

```cpp
// The full-stack CPU-expert benchmark (docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md):
// a writer thread posts CPU-hit requests into the lease lanes; the real RamTier, RamThread and CpuExpertEngine serve
// them with the optimized EXL3 kernel; the writer times post -> CopyDone against the bare kernel call.
#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "placement.h"
#include "self_test.h"

namespace {
namespace fs = std::filesystem;
using namespace fullstack;

struct Options {
  fs::path fixture = "/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin";
  fs::path references = "/data/models/exl3_exp/threading";
  fs::path image_dir = "/data/models/exl3_exp/google_benchmark/full-stack-images";
  std::optional<int> writer_cpu, service_cpu, copy_cpu;
  std::optional<std::string> cpus;
  int host_node = 0;
  int worker_node = 1;
  int warmup = 128;
  int gap_us = 0;
  int wait_timeout_ms = 2000;
  bool validate_only = false;
  bool self_test = false;
};

Options parse_options(int& argc, char** argv) {
  Options opt;
  int remaining = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
    if (arg.starts_with("--fixture=")) opt.fixture = value("--fixture=");
    else if (arg.starts_with("--reference-dir=")) opt.references = value("--reference-dir=");
    else if (arg.starts_with("--image-dir=")) opt.image_dir = value("--image-dir=");
    else if (arg.starts_with("--writer-cpu=")) opt.writer_cpu = number(value("--writer-cpu="));
    else if (arg.starts_with("--service-cpu=")) opt.service_cpu = number(value("--service-cpu="));
    else if (arg.starts_with("--copy-cpu=")) opt.copy_cpu = number(value("--copy-cpu="));
    else if (arg.starts_with("--cpus=")) opt.cpus = value("--cpus=");
    else if (arg.starts_with("--host-node=")) opt.host_node = number(value("--host-node="));
    else if (arg.starts_with("--worker-node=")) opt.worker_node = number(value("--worker-node="));
    else if (arg.starts_with("--warmup-forwards=")) opt.warmup = number(value("--warmup-forwards="));
    else if (arg.starts_with("--gap-us=")) opt.gap_us = number(value("--gap-us="));
    else if (arg.starts_with("--wait-timeout-ms=")) opt.wait_timeout_ms = number(value("--wait-timeout-ms="));
    else if (arg == "--validate-only") opt.validate_only = true;
    else if (arg == "--self-test") opt.self_test = true;
    else if (arg == "--help") {
      std::cout << "Full-stack CPU-expert benchmark (" EXL3_BENCH_BACKEND ")\n"
        "--fixture=FILE --reference-dir=DIR --image-dir=DIR (O_DIRECT-capable; row images are written there)\n"
        "--writer-cpu=16 --service-cpu=17 --copy-cpu=52 --cpus=18-33 --host-node=0 --worker-node=1\n"
        "--warmup-forwards=128 --gap-us=0 --wait-timeout-ms=2000 --validate-only\n"
        "--self-test: synthetic rows and a fake forward; defaults --writer-cpu=0 --service-cpu=1 --copy-cpu=2 --cpus=3\n"
        "Google Benchmark flags are also accepted.\n";
      argv[remaining++] = argv[i];
    } else argv[remaining++] = argv[i];
  }
  argc = remaining;
  argv[remaining] = nullptr;
  if (opt.wait_timeout_ms < 2) throw std::runtime_error("--wait-timeout-ms must be at least 2");
  return opt;
}

// The bench's defaults are production's placement; the self-test's fit any four CPUs (run it under taskset -c 0-15).
Placement resolve_placement(const Options& o) {
  Placement p;
  p.writer = o.writer_cpu.value_or(o.self_test ? 0 : 16);
  p.service = o.service_cpu.value_or(o.self_test ? 1 : 17);
  p.copy = o.copy_cpu.value_or(o.self_test ? 2 : 52);
  p.workers = parse_cpus(o.cpus.value_or(o.self_test ? "3" : "18-33"));
  p.host_node = o.host_node;
  p.worker_node = o.worker_node;
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const Placement placement = resolve_placement(options);
    validate_placement(placement, system_topology(), !options.self_test);
    if (options.self_test) return run_self_test(placement, options.image_dir) == 0 ? 0 : 1;
    throw std::runtime_error("only --self-test is built yet");
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
```

- [ ] **Step 6: Commit the failing state, create the divix01 worktree and build directory, build (expect a link failure)**

```bash
git -C $WT add $BENCH/CMakeLists.txt $BENCH/src/aligned.h $BENCH/src/placement.h $BENCH/src/self_test.h \
  $BENCH/src/self_test.cpp $BENCH/src/full_stack.cpp
git -C $WT commit -F - <<'EOF'
bench(full-stack): build targets, placement interface and its self-test checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
git -C $WT push origin expert-stream-cpu-bench
ssh divix01 'set -e; git -C /data/models/slang/sglang fetch -q origin
  git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-fullstack origin/expert-stream-cpu-bench
  R=/data/models/slang/nvfp4-work/wt-fullstack; B=/data/models/slang/nvfp4-work/fullstack-build
  V=/data/models/slang/.venv/lib/python3.13/site-packages; mkdir -p $B
  taskset -c 0-63 cmake -S $R/python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $B \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    -DEXL3_TORCH_ROOT=$V/torch -DEXL3_TVM_FFI_ROOT=$V/tvm_ffi \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src \
    > $B/configure.log 2>&1; echo CONFIGURE=$?; tail -3 $B/configure.log'
```

Expected: `CONFIGURE=0`. Then run **SYNC+BUILD**.
Expected: `BUILD=` nonzero, with `undefined reference to 'fullstack::parse_cpus` (and `validate_placement`, …) in the
tail. The checks compile against the interface, which has no definitions yet.

- [ ] **Step 7: Implement `src/placement.cpp`**

```cpp
#include "placement.h"

#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "expert_stream/host/core_topology.h"

namespace fullstack {
namespace fs = std::filesystem;

int number(const std::string& text) {
  size_t end = 0;
  int n = 0;
  try {
    n = std::stoi(text, &end);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid integer: " + text);
  }
  if (end != text.size() || n < 0) throw std::runtime_error("Invalid integer: " + text);
  return n;
}

std::vector<int32_t> parse_cpus(const std::string& text) {
  std::vector<int32_t> cores;
  std::stringstream list(text);
  std::string item;
  while (std::getline(list, item, ',')) {
    const auto dash = item.find('-');
    const int first = number(item.substr(0, dash));
    const int last = dash == std::string::npos ? first : number(item.substr(dash + 1));
    if (first > last || last >= CPU_SETSIZE) throw std::runtime_error("Invalid CPU range: " + item);
    for (int cpu = first; cpu <= last; ++cpu) {
      if (std::find(cores.begin(), cores.end(), cpu) != cores.end())
        throw std::runtime_error("Duplicate CPU: " + std::to_string(cpu));
      cores.push_back(cpu);
    }
  }
  if (cores.empty()) throw std::runtime_error("Empty CPU list");
  return cores;
}

Topology system_topology() {
  Topology t;
  t.node_of = [](int cpu) {
    for (int node = 0; node < 64; ++node) {
      if (fs::exists("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(node))) return node;
    }
    return -1;
  };
  t.siblings_of = [](int cpu) { return sglang::expert_stream::core_siblings(cpu); };
  CPU_ZERO(&t.allowed);
  if (sched_getaffinity(0, sizeof(t.allowed), &t.allowed) != 0)
    throw std::runtime_error("Cannot read the process's CPU affinity");
  return t;
}

void validate_placement(const Placement& p, const Topology& t, bool check_nodes) {
  if (p.workers.empty()) throw std::runtime_error("placement: no CPU expert cores");
  struct Role {
    const char* name;
    int cpu;
  };
  std::vector<Role> roles = {{"writer", p.writer}, {"service", p.service}, {"copy", p.copy}};
  for (int32_t cpu : p.workers) roles.push_back({"worker", cpu});
  std::set<int> seen;
  for (const Role& role : roles) {
    if (role.cpu < 0 || role.cpu >= CPU_SETSIZE)
      throw std::runtime_error(std::string("placement: the ") + role.name + " CPU " + std::to_string(role.cpu) + " is out of range");
    if (!seen.insert(role.cpu).second)
      throw std::runtime_error("placement: CPU " + std::to_string(role.cpu) + " has two roles");
    if (!CPU_ISSET(role.cpu, &t.allowed))
      throw std::runtime_error(std::string("placement: the ") + role.name + " CPU " + std::to_string(role.cpu) +
                               " is outside the process's allowed CPUs");
  }
  // A busy-polling service never yields: no other role may share its physical core (check_dedicated_core's rule).
  for (int sibling : t.siblings_of(p.service)) {
    if (sibling != p.service && seen.contains(sibling))
      throw std::runtime_error("placement: CPU " + std::to_string(sibling) + " shares the physical core of the service CPU " +
                               std::to_string(p.service));
  }
  if (!check_nodes) return;
  for (const Role& role : roles) {
    const int want = std::string(role.name) == "worker" ? p.worker_node : p.host_node;
    const int node = t.node_of(role.cpu);
    if (node != want)
      throw std::runtime_error(std::string("placement: the ") + role.name + " CPU " + std::to_string(role.cpu) +
                               " is on NUMA node " + std::to_string(node) + ", not node " + std::to_string(want));
  }
}

void pin_self(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
    throw std::runtime_error("Cannot pin the calling thread to CPU " + std::to_string(cpu));
}

PinScope::PinScope(int cpu) {
  CPU_ZERO(&saved_);
  if (sched_getaffinity(0, sizeof(saved_), &saved_) != 0) throw std::runtime_error("Cannot read the thread's affinity");
  pin_self(cpu);
}

PinScope::~PinScope() {
  sched_setaffinity(0, sizeof(saved_), &saved_);
}

std::set<int> task_ids() {
  std::set<int> result;
  for (const auto& entry : fs::directory_iterator("/proc/self/task"))
    result.insert(number(entry.path().filename().string()));
  return result;
}

std::vector<int> expected_threads(const Placement& p) {
  std::vector<int> cpus = {p.service, p.copy, p.copy};
  cpus.insert(cpus.end(), p.workers.begin(), p.workers.end());
  std::sort(cpus.begin(), cpus.end());
  return cpus;
}

std::string cpu_list(std::vector<int> cpus) {
  std::sort(cpus.begin(), cpus.end());
  std::string text;
  for (int cpu : cpus) text += (text.empty() ? "" : ",") + std::to_string(cpu);
  return text;
}

void verify_threads(const std::set<int>& before, std::vector<int> expected) {
  std::vector<int> pinned;
  for (int tid : task_ids()) {
    if (before.contains(tid)) continue;
    std::ifstream comm_file("/proc/self/task/" + std::to_string(tid) + "/comm");
    std::string comm;
    std::getline(comm_file, comm);
    if (comm.starts_with("iou-")) continue;  // io_uring's kernel workers
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(tid, sizeof(mask), &mask) != 0 || CPU_COUNT(&mask) != 1)
      throw std::runtime_error("thread " + std::to_string(tid) + " (" + comm + ") is not pinned to one CPU");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &mask)) pinned.push_back(cpu);
  }
  std::sort(pinned.begin(), pinned.end());
  std::sort(expected.begin(), expected.end());
  if (pinned != expected)
    throw std::runtime_error("threads are pinned to {" + cpu_list(pinned) + "}, expected {" + cpu_list(expected) + "}");
}

}  // namespace fullstack
```

In `CMakeLists.txt`, add the new source:

```cmake
  set(FULL_STACK_SOURCES src/full_stack.cpp src/self_test.cpp src/placement.cpp)
```

- [ ] **Step 8: Commit, SYNC+BUILD, SELFTEST**

```bash
git -C $WT add $BENCH/src/placement.cpp $BENCH/CMakeLists.txt
git -C $WT commit -F - <<'EOF'
bench(full-stack): placement rules, pinning and thread-affinity verification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD**, expecting `BUILD=0`. Then run **SELFTEST**, expecting `self-test: 13 checks, 0 failed` and
`prod EXIT=0`, `instr EXIT=0`. The `refused:` lines for the expected refusals are informational.

- [ ] **Step 9: Show the existing bench still configures without tvm-ffi**

```bash
ssh divix01 'R=/data/models/slang/nvfp4-work/wt-fullstack; V=/data/models/slang/.venv/lib/python3.13/site-packages
  T=/data/models/slang/nvfp4-work/fullstack-noffi-configure
  taskset -c 0-63 cmake -S $R/python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $T \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ -DEXL3_TORCH_ROOT=$V/torch \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src \
    > $T.out 2>&1; echo CONFIGURE=$?; grep -c "Full-stack bench skipped" $T.out; rm -rf $T $T.out'
```

Expected: `CONFIGURE=0`, then `1`.

---

### Task 2: `DeviceSim`, the device's side of the protocol

**Files:**
- Create: `$BENCH/src/device_sim.h`, `$BENCH/src/device_sim.cpp`
- Modify: `$BENCH/src/self_test.cpp` (add tests and their calls), `$BENCH/CMakeLists.txt` (add `src/device_sim.cpp`)

**Interfaces:**
- Consumes: `sglang::expert_stream::wire` constants (`expert_stream/lease_layout.h`); `es::read_record`,
  `es::Request`, `es::RecordRead` (`expert_stream/host/tier_protocol.h`, self-test only); `aligned_zeroed`,
  `round_up` (Task 1).
- Produces (namespace `fullstack`):
  - `constexpr int kLanes = 8`;
  - `int64_t monotonic_ns()`: CLOCK_MONOTONIC, the clock the host's `now_ns` and the stage trace use;
  - `struct SimRequest { uint32_t seq; uint64_t gen; int64_t idx, row; int count; std::array<int32_t,8> experts, kinds, slots; uint64_t chain; }`;
  - `using PostHook = std::function<void(const uint8_t* record)>`.
  - `class DeviceSim`:
    - `DeviceSim(uint8_t* page, uint8_t* lease, int64_t rows, int64_t experts, uint32_t epoch = 0)`;
    - `void set_row_cpu(int64_t row)`;
    - `SimRequest post(int64_t row, std::span<const int32_t> experts, std::span<const float> weights, bool captured, int64_t deadline_ns, const PostHook& before_publish = {})`;
    - `void sync_row(int64_t row, int64_t deadline_ns)`;
    - `bool copy_wait(const SimRequest&, int64_t deadline_ns)`;
    - `bool wait_pieces(const SimRequest&, int lane, int64_t deadline_ns) const`;
    - `static bool needs_copy_wait(const SimRequest&)`;
    - `int32_t ram_slot(int64_t row, int32_t expert) const`;
    - `std::array<int32_t,8> staging(int64_t row) const`;
    - `uint64_t map_chain(int64_t row) const`;
    - `uint64_t copy_done(const SimRequest&) const`;
    - `uint32_t copy_gate() const`;
    - `uint32_t epoch() const`.
  - `uint32_t load_experts(DeviceSim&, int64_t row, std::span<const int32_t> experts, int staging, int64_t timeout_ns)`,
    which returns the last load's seq.

- [ ] **Step 1: Write `src/device_sim.h`**

```cpp
// The device's side of the expert-stream lease protocol, for the bench: a C++ port of python/sglang/test/
// dsv41_chain_sim.py's ChainSim (analysis/dsv41-drive/LEASE_PROTOCOL.md). It stands in for the post kernel, S and CW;
// it is not evidence about them. Lanes are typed as ram_slot_map.type_lanes types them, with no copy table
// (hit_copy="sm": a hit the CPU does not take is kKindHitSm) and CPU hits only (cpu_misses=false).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace fullstack {

constexpr int kLanes = 8;  // wire::kLeaseLanes

int64_t monotonic_ns();  // CLOCK_MONOTONIC: the host's now_ns() and its stage trace read the same clock

struct SimRequest {
  uint32_t seq = 0;
  uint64_t gen = 0;  // epoch << 32 | seq
  int64_t idx = 0;   // (seq - 1) % 16: the ring slot, the CopyDone and PieceMask index
  int64_t row = 0;
  int count = 0;
  std::array<int32_t, kLanes> experts{};
  std::array<int32_t, kLanes> kinds{};  // wire::kKind*
  std::array<int32_t, kLanes> slots{};  // a hit's RAM slot, a miss's staging slot
  uint64_t chain = 0;                   // the row's map-chain number when a lane misses, else 0
};

// Called with the record after its payload is written and before its seq is published (the self-test's seqlock check).
using PostHook = std::function<void(const uint8_t* record)>;

class DeviceSim {
 public:
  DeviceSim(uint8_t* page, uint8_t* lease, int64_t rows, int64_t experts, uint32_t epoch = 0);

  // The row's CPU layer is registered: its hits become CPU-eligible (the device side's set_row_cpu).
  void set_row_cpu(int64_t row);

  // The post kernel: apply the row's pending delta (waiting for it until deadline_ns, then throwing), type the lanes,
  // write the record (seq = 0, payload, seq with a release) and demand_head (release). Protect ids are the experts;
  // destinations are 0..count-1.
  SimRequest post(int64_t row, std::span<const int32_t> experts, std::span<const float> weights, bool captured,
                  int64_t deadline_ns, const PostHook& before_publish = {});

  // Apply the row's delta now, waiting for the host to publish it; throws at deadline_ns.
  void sync_row(int64_t row, int64_t deadline_ns);

  // CW: close the gate for G (seq_cst), spin until CopyDone == G, then open the gate if it is still closed for G
  // (CW's own open, for a CopyDone that landed before the close). False at deadline_ns, the gate left closed.
  // True at once for a request with no host lane.
  bool copy_wait(const SimRequest& request, int64_t deadline_ns);

  // S: lane `lane`'s PieceMask word reads piece_word(G) with all 8 piece bits. False at deadline_ns.
  bool wait_pieces(const SimRequest& request, int lane, int64_t deadline_ns) const;

  static bool needs_copy_wait(const SimRequest& request);  // a kKindHitCopy, kKindHitCpu or kKindMissCpu lane

  int32_t ram_slot(int64_t row, int32_t expert) const;
  std::array<int32_t, kLanes> staging(int64_t row) const;
  uint64_t map_chain(int64_t row) const;
  uint64_t copy_done(const SimRequest& request) const;
  uint32_t copy_gate() const;
  uint32_t epoch() const;

 private:
  bool apply_pending(int64_t row);

  uint8_t* page_;
  uint8_t* lease_;
  int64_t rows_;
  int64_t experts_;
  uint32_t epoch_;
  std::vector<int32_t> ram_slot_;                    // [rows][experts]
  std::vector<std::array<int32_t, kLanes>> staging_;  // [rows]
  std::vector<uint64_t> map_chain_;                  // starts at 1: the attach delta's tag
  std::vector<uint64_t> map_applied_;
  std::vector<uint8_t> row_cpu_;
};

// Make `experts` resident in `row` through uncaptured posts, as a plain request does. A post misses at most `staging`
// experts (the row's staging slots), so they go in groups; every lane's pieces are awaited, then the row's delta is
// applied. Throws on a lane that is not a miss, a late piece, or an expert left unmapped. Returns the last post's seq.
uint32_t load_experts(DeviceSim& sim, int64_t row, std::span<const int32_t> experts, int staging, int64_t timeout_ns);

}  // namespace fullstack
```

- [ ] **Step 2: Write the failing DeviceSim checks in `src/self_test.cpp`**

Add these includes at the top of `self_test.cpp`, after `#include "self_test.h"`:

```cpp
#include <array>
#include <cstring>
#include <utility>

#include "aligned.h"
#include "device_sim.h"
#include "expert_stream/host/tier_protocol.h"
```

Add these declarations right after `int failures = 0;`:

```cpp
namespace w = ::sglang::expert_stream::wire;
namespace es = ::sglang::expert_stream;
```

Insert after `test_placement()`, still inside the anonymous namespace:

```cpp
// ---- DeviceSim, on a standalone page and lease block (no service: the host's words are written by hand) ----

uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & w::kLeaseGateSeqMask) << w::kLeaseGateSeqShift) | low;
}

int64_t soon() {
  return monotonic_ns() + 100'000'000;
}

constexpr std::array<int16_t, 8> kStaging012 = {0, 1, 2, -1, -1, -1, -1, -1};
constexpr std::array<int32_t, 9> kAllToCpu = {0, 1, 2, 3, 4, 5, 6, 7, 8};

struct Blocks {
  explicit Blocks(int64_t rows)
      : page(aligned_zeroed(w::kPageBytes)),
        lease_bytes(w::kLeaseBlockBytes + round_up(rows * w::kDeltaStride, 4096)),
        lease(aligned_zeroed(lease_bytes)) {}

  // The host's delta record for `row`: payload, then the tag with a release (RamTier::publish_delta_locked).
  void delta(int64_t row, uint64_t tag, std::array<int16_t, 8> staging, std::vector<std::pair<int16_t, int16_t>> entries) {
    uint8_t* d = lease.get() + w::kDeltaBase + row * w::kDeltaStride;
    const auto count = static_cast<uint32_t>(entries.size());
    std::memcpy(d + w::kDeltaCount, &count, 4);
    std::memcpy(d + w::kDeltaStaging, staging.data(), 16);
    for (size_t i = 0; i < entries.size(); ++i) {
      const int16_t entry[2] = {entries[i].first, entries[i].second};
      std::memcpy(d + w::kDeltaEntries + 4 * i, entry, 4);
    }
    __atomic_store_n(reinterpret_cast<uint64_t*>(d + w::kDeltaTag), tag, __ATOMIC_RELEASE);
  }
  void split(std::array<int32_t, 9> table) {
    std::memcpy(lease.get() + w::kSplit, table.data(), sizeof(table));
  }
  void armed(bool on) {
    const uint32_t value = on ? 1 : 0;
    std::memcpy(lease.get() + w::kCopyArmed, &value, 4);
  }
  template <class T>
  T at(int64_t offset) const {
    T value;
    std::memcpy(&value, page.get() + offset, sizeof(value));
    return value;
  }

  AlignedBuffer page;
  int64_t lease_bytes;
  AlignedBuffer lease;
};

void test_record_bytes() {
  Blocks b(2);
  b.delta(0, 1, kStaging012, {{3, 4}});  // expert 3 resident in slot 4
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 2, 8);
  sim.set_row_cpu(0);
  const int32_t experts[] = {3, 5};
  const float weights[] = {0.5f, 0.25f};
  const SimRequest r = sim.post(0, experts, weights, true, soon());
  // type_lanes: expert 3 hits slot 4 and is the CPU's (split[1] = 1); expert 5 misses into staging[0] = 0.
  CHECK(r.kinds[0] == int32_t(w::kKindHitCpu) && r.kinds[1] == int32_t(w::kKindMissGpu));
  CHECK(r.slots[0] == 4 && r.slots[1] == 0);
  CHECK(r.seq == 1 && r.gen == 1 && r.idx == 0 && r.chain == 2 && sim.map_chain(0) == 2);
  const int64_t rec = w::kDemandRing;
  CHECK(b.at<uint32_t>(w::kDemandHead) == 1);
  CHECK(b.at<uint32_t>(rec + w::kRecSeq) == 1);
  CHECK(b.at<uint16_t>(rec + w::kRecRow) == 0);
  CHECK(b.at<uint8_t>(rec + w::kRecCounts) == (2 | 2 << 4));
  CHECK(b.at<uint8_t>(rec + w::kRecFlags) == w::kRecFlagCaptured);
  CHECK(b.at<uint64_t>(rec + w::kRecChain) == 2);
  CHECK(b.at<uint32_t>(rec + w::kRecEpoch) == 0);
  CHECK(b.at<uint32_t>(rec + w::kRecKinds) == (3u | 4u << 4));
  const int16_t ids[8] = {3, 5, -1, -1, -1, -1, -1, -1};
  const int16_t slots[8] = {4, 0, -1, -1, -1, -1, -1, -1};
  const int16_t dst[8] = {0, 1, -1, -1, -1, -1, -1, -1};
  const float lane_weights[8] = {0.5f, 0.25f, 0, 0, 0, 0, 0, 0};
  CHECK(std::memcmp(b.page.get() + rec + w::kRecProtect, ids, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneExpert, ids, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneSlot, slots, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneDst, dst, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneWeight, lane_weights, 32) == 0);
  // The production parser reads it back as the device meant it.
  es::Request req;
  CHECK(es::read_record(b.page.get() + rec, 1, &req) == es::RecordRead::kOk);
  CHECK(req.gen == 1 && req.row == 0 && req.captured && req.chain == 2);
  CHECK(req.lanes.size() == 2 && req.protect.size() == 2);
  CHECK(req.lanes[0].expert == 3 && req.lanes[0].slot == 4 && req.lanes[0].dst == 0 && req.lanes[0].weight == 0.5f &&
        req.lanes[0].kind == w::kKindHitCpu);
  CHECK(req.lanes[1].expert == 5 && req.lanes[1].slot == 0 && req.lanes[1].kind == w::kKindMissGpu);
}

void test_seqlock_order() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 5}});
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {0.75f};
  bool hooked = false;
  sim.post(0, e, wt, true, soon(), [&](const uint8_t* record) {
    hooked = true;
    uint32_t seq, head, kinds;
    float weight;
    std::memcpy(&seq, record + w::kRecSeq, 4);
    std::memcpy(&head, b.page.get() + w::kDemandHead, 4);
    std::memcpy(&kinds, record + w::kRecKinds, 4);
    std::memcpy(&weight, record + w::kRecLaneWeight, 4);
    CHECK(seq == 0);   // the seqlock word is 0 while the payload is in place
    CHECK(head == 0);  // demand_head moves only after the seq
    CHECK(kinds == w::kKindHitCpu && weight == 0.75f);
  });
  CHECK(hooked);
  CHECK(b.at<uint32_t>(w::kDemandRing + w::kRecSeq) == 1 && b.at<uint32_t>(w::kDemandHead) == 1);
  // A rewrite of a ring slot that held an earlier record also zeroes its seq first: seq 17 reuses slot 0.
  for (int i = 0; i < 15; ++i) sim.post(0, e, wt, true, soon());
  bool rehooked = false;
  sim.post(0, e, wt, true, soon(), [&](const uint8_t* record) {
    rehooked = true;
    uint32_t seq;
    std::memcpy(&seq, record + w::kRecSeq, 4);
    CHECK(record == b.page.get() + w::kDemandRing);
    CHECK(seq == 0);
  });
  CHECK(rehooked && b.at<uint32_t>(w::kDemandRing + w::kRecSeq) == 17);
}

void test_owed_delta() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {});
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t miss[] = {5};
  const float one[] = {1.0f};
  const SimRequest first = sim.post(0, miss, one, false, soon());
  CHECK(first.kinds[0] == int32_t(w::kKindMissGpu) && first.slots[0] == 0 && first.chain == 2);
  // The host has not published delta 2: the row's next post waits for it, then refuses at its deadline.
  CHECK_THROWS(sim.post(0, miss, one, true, monotonic_ns() + 1'000'000), "delta 2");
  CHECK(b.at<uint32_t>(w::kDemandHead) == 1);  // the refused post wrote nothing
  // Delta 2, as the host publishes it: expert 5 in slot 0, the victim slot 3 now staging[0].
  b.delta(0, 2, {3, 1, 2, -1, -1, -1, -1, -1}, {{5, 0}});
  const SimRequest hit = sim.post(0, miss, one, true, soon());
  CHECK(hit.kinds[0] == int32_t(w::kKindHitCpu) && hit.slots[0] == 0 && hit.chain == 0 && sim.map_chain(0) == 2);
  CHECK(sim.staging(0)[0] == 3);
  // The next miss takes the new staging[0] and the next chain number.
  const int32_t other[] = {6};
  const SimRequest second = sim.post(0, other, one, true, soon());
  CHECK(second.kinds[0] == int32_t(w::kKindMissGpu) && second.slots[0] == 3 && second.chain == 3);
}

void test_typing() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}, {1, 5}});
  b.armed(true);
  b.split({0, 0, 0, 0, 0, 0, 0, 0, 0});
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t two[] = {0, 1};
  const float halves[] = {0.5f, 0.5f};
  SimRequest r = sim.post(0, two, halves, true, soon());  // split[2] = 0: no CPU lane
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm) && !DeviceSim::needs_copy_wait(r));
  b.split({0, 1, 1, 1, 1, 1, 1, 1, 1});  // split[2] = 1: the last eligible lane only
  r = sim.post(0, two, halves, true, soon());
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitCpu) && DeviceSim::needs_copy_wait(r));
  b.split(kAllToCpu);
  r = sim.post(0, two, halves, false, soon());  // uncaptured: no host lanes
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
  b.armed(false);
  r = sim.post(0, two, halves, true, soon());  // the copy engine is not armed: no host lanes
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
  Blocks c(1);
  c.delta(0, 1, kStaging012, {{0, 4}});
  c.split(kAllToCpu);
  c.armed(true);
  DeviceSim cold(c.page.get(), c.lease.get(), 1, 8);  // no set_row_cpu: the row's layer is not registered
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  CHECK(cold.post(0, e, wt, true, soon()).kinds[0] == int32_t(w::kKindHitSm));
  CHECK_THROWS(cold.post(0, two, wt, true, soon()), "weights");
  const int32_t dup[] = {0, 0};
  CHECK_THROWS(cold.post(0, dup, halves, true, soon()), "twice");
}

void test_epoch_wrap() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}});
  b.split(kAllToCpu);
  b.armed(true);
  const uint32_t last = 0xFFFFFFFFu;
  std::memcpy(b.page.get() + w::kDemandHead, &last, 4);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8, /*epoch=*/7);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  const SimRequest r = sim.post(0, e, wt, true, soon());
  CHECK(r.seq == 1 && r.idx == 0 && sim.epoch() == 8 && r.gen == (uint64_t{8} << 32 | 1));
  CHECK(b.at<uint32_t>(w::kDemandRing + w::kRecEpoch) == 8 && b.at<uint32_t>(w::kDemandHead) == 1);
}

void test_copy_wait_gate() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}});
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  const SimRequest r = sim.post(0, e, wt, true, soon());
  // No CopyDone: the wait ends at its deadline with the gate closed for G (the watchdog's evidence).
  CHECK(!sim.copy_wait(r, monotonic_ns() + 2'000'000));
  CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateClosed));
  // CopyDone stored before the close (the host's open found nothing closed): CW opens the gate itself.
  __atomic_store_n(reinterpret_cast<uint64_t*>(b.lease.get() + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes),
                   r.gen, __ATOMIC_RELEASE);
  CHECK(sim.copy_wait(r, soon()));
  CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateOpen));
  CHECK(sim.copy_done(r) == r.gen);
}
```

In `run_self_test`, after `test_placement();`, add:

```cpp
  test_record_bytes();
  test_seqlock_order();
  test_owed_delta();
  test_typing();
  test_epoch_wrap();
  test_copy_wait_gate();
```

In `CMakeLists.txt`, change the sources line to:

```cmake
  set(FULL_STACK_SOURCES src/full_stack.cpp src/self_test.cpp src/placement.cpp src/device_sim.cpp)
```

- [ ] **Step 3: Commit, SYNC+BUILD (expect a failure: `device_sim.cpp` does not exist)**

```bash
git -C $WT add $BENCH/src/device_sim.h $BENCH/src/self_test.cpp $BENCH/CMakeLists.txt
git -C $WT commit -F - <<'EOF'
bench(full-stack): DeviceSim interface and its record, seqlock, delta, typing, epoch and gate checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD**. Expected: `BUILD=` nonzero, with `Cannot find source file: src/device_sim.cpp` (CMake
regenerates) in the tail.

- [ ] **Step 4: Implement `src/device_sim.cpp`**

```cpp
#include "device_sim.h"

#include <immintrin.h>
#include <time.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "expert_stream/lease_layout.h"

namespace fullstack {
namespace w = ::sglang::expert_stream::wire;
static_assert(kLanes == w::kLeaseLanes && kLanes == w::kMaxIds);

int64_t monotonic_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

namespace {

template <class T>
T load_acquire(const uint8_t* p) {
  return __atomic_load_n(reinterpret_cast<const T*>(p), __ATOMIC_ACQUIRE);
}

template <class T>
void store_release(uint8_t* p, T value) {
  __atomic_store_n(reinterpret_cast<T*>(p), value, __ATOMIC_RELEASE);
}

template <class T>
void put(uint8_t* p, T value) {
  std::memcpy(p, &value, sizeof(value));
}

uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & w::kLeaseGateSeqMask) << w::kLeaseGateSeqShift) | low;
}

}  // namespace

DeviceSim::DeviceSim(uint8_t* page, uint8_t* lease, int64_t rows, int64_t experts, uint32_t epoch)
    : page_(page),
      lease_(lease),
      rows_(rows),
      experts_(experts),
      epoch_(epoch),
      ram_slot_(static_cast<size_t>(rows * experts), -1),
      staging_(static_cast<size_t>(rows)),
      map_chain_(static_cast<size_t>(rows), 1),
      map_applied_(static_cast<size_t>(rows), 0),
      row_cpu_(static_cast<size_t>(rows), 0) {
  for (auto& s : staging_) s.fill(-1);
}

void DeviceSim::set_row_cpu(int64_t row) {
  row_cpu_.at(static_cast<size_t>(row)) = 1;
}

// The post's delta apply (ChainSim.apply_pending): the tag is acquired, then the payload read.
bool DeviceSim::apply_pending(int64_t row) {
  const uint8_t* d = lease_ + w::kDeltaBase + row * w::kDeltaStride;
  const uint64_t tag = load_acquire<uint64_t>(d + w::kDeltaTag);
  if (tag != map_chain_[row]) return false;
  if (map_applied_[row] == tag) return true;
  uint32_t count;
  std::memcpy(&count, d + w::kDeltaCount, 4);
  if (count > static_cast<uint32_t>(w::kDeltaMaxEntries))
    throw std::runtime_error("row " + std::to_string(row) + ": delta " + std::to_string(tag) + " has " +
                             std::to_string(count) + " entries");
  for (uint32_t i = 0; i < count; ++i) {
    int16_t entry[2];
    std::memcpy(entry, d + w::kDeltaEntries + 4 * i, 4);
    if (entry[0] < 0 || entry[0] >= experts_) throw std::runtime_error("a delta entry names expert " + std::to_string(entry[0]));
    ram_slot_[row * experts_ + entry[0]] = entry[1];
  }
  for (int k = 0; k < kLanes; ++k) {
    int16_t slot;
    std::memcpy(&slot, d + w::kDeltaStaging + 2 * k, 2);
    staging_[row][k] = slot;
  }
  map_applied_[row] = tag;
  return true;
}

void DeviceSim::sync_row(int64_t row, int64_t deadline_ns) {
  if (row < 0 || row >= rows_) throw std::runtime_error("row " + std::to_string(row) + " is out of range");
  for (uint32_t spin = 0; !apply_pending(row); ++spin) {
    if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns)
      throw std::runtime_error("row " + std::to_string(row) + ": the host has not published delta " +
                               std::to_string(map_chain_[row]));
    _mm_pause();
  }
}

SimRequest DeviceSim::post(int64_t row, std::span<const int32_t> experts, std::span<const float> weights, bool captured,
                           int64_t deadline_ns, const PostHook& before_publish) {
  const int count = static_cast<int>(experts.size());
  if (row < 0 || row >= rows_) throw std::runtime_error("row " + std::to_string(row) + " is out of range");
  if (count < 1 || count > kLanes) throw std::runtime_error("a post has 1.." + std::to_string(kLanes) + " lanes");
  if (weights.size() != experts.size()) throw std::runtime_error("a post needs one of its weights per lane");
  for (int j = 0; j < count; ++j) {
    if (experts[j] < 0 || experts[j] >= experts_) throw std::runtime_error("expert " + std::to_string(experts[j]) + " is out of range");
    for (int i = 0; i < j; ++i)
      if (experts[i] == experts[j]) throw std::runtime_error("a post names expert " + std::to_string(experts[j]) + " twice");
  }
  sync_row(row, deadline_ns);

  // ram_slot_map.type_lanes, hit_copy="sm", cpu_misses=false, cpu_ok=ce_ok=true.
  SimRequest r;
  r.row = row;
  r.count = count;
  bool hit[kLanes] = {};
  int m = 0;
  for (int j = 0; j < count; ++j) {
    r.experts[j] = experts[j];
    const int32_t slot = ram_slot(row, experts[j]);
    if (slot >= 0) {
      r.slots[j] = slot;
      hit[j] = true;
      continue;
    }
    if (m >= kLanes || staging_[row][m] < 0) throw std::runtime_error("a miss lane has no staging slot");
    r.slots[j] = staging_[row][m++];
  }
  const bool host_lanes = captured && load_acquire<uint32_t>(lease_ + w::kCopyArmed) != 0;
  bool eligible[kLanes] = {};
  int n = 0;
  for (int j = 0; j < count; ++j) {
    eligible[j] = host_lanes && row_cpu_[row] != 0 && hit[j];
    n += eligible[j] ? 1 : 0;
  }
  int take = n > 0 ? __atomic_load_n(reinterpret_cast<const int32_t*>(lease_ + w::kSplit) + n, __ATOMIC_RELAXED) : 0;
  bool cpu[kLanes] = {};
  for (int j = count - 1; j >= 0 && take > 0; --j) {
    if (eligible[j]) {
      cpu[j] = true;
      --take;
    }
  }
  bool miss = false;
  for (int j = 0; j < count; ++j) {
    r.kinds[j] = static_cast<int32_t>(cpu[j] ? w::kKindHitCpu : hit[j] ? w::kKindHitSm : w::kKindMissGpu);
    miss = miss || !hit[j];
  }
  if (miss) r.chain = ++map_chain_[row];

  const uint32_t head = __atomic_load_n(reinterpret_cast<const uint32_t*>(page_ + w::kDemandHead), __ATOMIC_RELAXED);
  uint32_t seq = head + 1;
  if (seq == 0) {  // the post kernel never posts 0: it wraps to 1 in the next epoch
    seq = 1;
    ++epoch_;
  }
  r.seq = seq;
  r.gen = static_cast<uint64_t>(epoch_) << 32 | seq;
  r.idx = static_cast<int64_t>((seq - 1) % w::kDemandRecords);
  uint8_t* rec = page_ + w::kDemandRing + r.idx * w::kRecordBytes;

  // The seqlock: seq = 0 first. x86 keeps stores in order; the barrier keeps the compiler from hoisting the payload.
  __atomic_store_n(reinterpret_cast<uint32_t*>(rec + w::kRecSeq), 0u, __ATOMIC_RELAXED);
  asm volatile("" ::: "memory");
  put<uint16_t>(rec + w::kRecRow, static_cast<uint16_t>(row));
  put<uint8_t>(rec + w::kRecCounts, static_cast<uint8_t>(count | count << 4));  // protect ids = the lane experts
  put<uint8_t>(rec + w::kRecFlags, static_cast<uint8_t>(captured ? w::kRecFlagCaptured : 0));
  put<uint64_t>(rec + w::kRecChain, r.chain);
  put<uint32_t>(rec + w::kRecEpoch, epoch_);
  uint32_t kinds = 0;
  for (int j = 0; j < count; ++j) kinds |= (static_cast<uint32_t>(r.kinds[j]) & 0xFu) << (4 * j);
  put<uint32_t>(rec + w::kRecKinds, kinds);
  for (int j = 0; j < kLanes; ++j) {
    const bool lane = j < count;
    put<int16_t>(rec + w::kRecProtect + 2 * j, static_cast<int16_t>(lane ? experts[j] : -1));
    put<int16_t>(rec + w::kRecLaneExpert + 2 * j, static_cast<int16_t>(lane ? experts[j] : -1));
    put<int16_t>(rec + w::kRecLaneSlot + 2 * j, static_cast<int16_t>(lane ? r.slots[j] : -1));
    put<int16_t>(rec + w::kRecLaneDst + 2 * j, static_cast<int16_t>(lane ? j : -1));
    put<float>(rec + w::kRecLaneWeight + 4 * j, lane ? weights[j] : 0.0f);
  }
  if (before_publish) before_publish(rec);
  store_release<uint32_t>(rec + w::kRecSeq, seq);
  store_release<uint32_t>(page_ + w::kDemandHead, seq);
  return r;
}

bool DeviceSim::needs_copy_wait(const SimRequest& r) {
  for (int j = 0; j < r.count; ++j) {
    const auto kind = static_cast<uint32_t>(r.kinds[j]);
    if (kind == w::kKindHitCopy || kind == w::kKindHitCpu || kind == w::kKindMissCpu) return true;
  }
  return false;
}

bool DeviceSim::copy_wait(const SimRequest& r, int64_t deadline_ns) {
  if (!needs_copy_wait(r)) return true;
  auto* gate = reinterpret_cast<uint32_t*>(lease_ + w::kLeaseCopyGate);
  const uint32_t closed = gate_word(r.seq, w::kLeaseGateClosed);
  __atomic_store_n(gate, closed, __ATOMIC_SEQ_CST);  // CW: close, then (fence.sc) read CopyDone
  const uint8_t* done = lease_ + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes;
  for (uint32_t spin = 0; load_acquire<uint64_t>(done) != r.gen; ++spin) {
    if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns) return false;
    _mm_pause();
  }
  // CW's own open, from G's closed word only (the host's CAS rule): nothing else opens a gate closed after CopyDone.
  uint32_t expected = closed;
  __atomic_compare_exchange_n(gate, &expected, gate_word(r.seq, w::kLeaseGateOpen), false, __ATOMIC_SEQ_CST,
                              __ATOMIC_ACQUIRE);
  return true;
}

bool DeviceSim::wait_pieces(const SimRequest& r, int lane, int64_t deadline_ns) const {
  const uint8_t* word = lease_ + w::kLeasePieceMask + (r.idx * w::kLeaseLanes + lane) * w::kLeasePieceMaskLineBytes;
  const uint64_t want = ((r.gen & ((uint64_t{1} << 56) - 1)) << 8) | 0xFFu;  // piece_word(G, every piece)
  for (uint32_t spin = 0; load_acquire<uint64_t>(word) != want; ++spin) {
    if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns) return false;
    _mm_pause();
  }
  return true;
}

int32_t DeviceSim::ram_slot(int64_t row, int32_t expert) const {
  return ram_slot_.at(static_cast<size_t>(row * experts_ + expert));
}

std::array<int32_t, kLanes> DeviceSim::staging(int64_t row) const {
  return staging_.at(static_cast<size_t>(row));
}

uint64_t DeviceSim::map_chain(int64_t row) const {
  return map_chain_.at(static_cast<size_t>(row));
}

uint64_t DeviceSim::copy_done(const SimRequest& r) const {
  return load_acquire<uint64_t>(lease_ + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes);
}

uint32_t DeviceSim::copy_gate() const {
  return load_acquire<uint32_t>(lease_ + w::kLeaseCopyGate);
}

uint32_t DeviceSim::epoch() const {
  return epoch_;
}

uint32_t load_experts(DeviceSim& sim, int64_t row, std::span<const int32_t> experts, int staging, int64_t timeout_ns) {
  if (staging < 1) throw std::runtime_error("load_experts needs at least one staging slot");
  uint32_t last = 0;
  for (size_t first = 0; first < experts.size(); first += static_cast<size_t>(staging)) {
    const auto group = experts.subspan(first, std::min<size_t>(static_cast<size_t>(staging), experts.size() - first));
    const std::vector<float> weights(group.size(), 1.0f);
    const int64_t deadline = monotonic_ns() + timeout_ns;
    const SimRequest r = sim.post(row, group, weights, /*captured=*/false, deadline);
    for (int j = 0; j < r.count; ++j) {
      if (r.kinds[j] != static_cast<int32_t>(w::kKindMissGpu))
        throw std::runtime_error("row " + std::to_string(row) + ": loading expert " + std::to_string(group[j]) +
                                 ", which is already resident");
      if (!sim.wait_pieces(r, j, deadline))
        throw std::runtime_error("row " + std::to_string(row) + ": expert " + std::to_string(group[j]) +
                                 "'s pieces did not land in time (gen " + std::to_string(r.gen) + ")");
    }
    last = r.seq;
  }
  sim.sync_row(row, monotonic_ns() + timeout_ns);
  for (int32_t e : experts)
    if (sim.ram_slot(row, e) < 0)
      throw std::runtime_error("row " + std::to_string(row) + ": expert " + std::to_string(e) + " is not resident after its load");
  return last;
}

}  // namespace fullstack
```

- [ ] **Step 5: Commit, SYNC+BUILD, SELFTEST**

```bash
git -C $WT add $BENCH/src/device_sim.cpp
git -C $WT commit -F - <<'EOF'
bench(full-stack): DeviceSim, the device's side of the lease protocol (a C++ ChainSim)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD**, expecting `BUILD=0`. Then run **SELFTEST**, expecting `self-test: N checks, 0 failed`, and
`prod EXIT=0`, `instr EXIT=0`. If `test_record_bytes`'s `read_record` checks fail while the byte checks pass, the
record layout disagrees with the production parser. Fix `device_sim.cpp`, never the checks.

---

### Task 3: Row images, `Stack<Build>`, and the real stack under the self-test

**Files:**
- Create: `$BENCH/src/row_images.h`, `$BENCH/src/row_images.cpp`, `$BENCH/src/stack.h`
- Modify: `$BENCH/src/self_test.cpp` (stamp and stack tests), `$BENCH/CMakeLists.txt` (add `src/row_images.cpp`)

**Interfaces:**
- Consumes:
  - `DeviceSim`, `load_experts` and `monotonic_ns` (Task 2);
  - `PinScope` and `Placement` (Task 1);
  - `aligned_zeroed` (Task 1);
  - host: `RamTier`, `RamThread`, `RowReader`, `UringReader`/`InstrUringReader`, `Tables`, `Read`, `Segment`,
    `RegisteredRegion`, `check_image_tables`, `check_dedicated_core`, `CpuExpertConfig`, `CpuExpertForward`,
    `StageRecord`, `reached`.
- Produces (namespace `fullstack`):
  - row images:
    - `constexpr int kNames = 6`;
    - `struct ImageLayout { std::array<int64_t,6> row_bytes, name_offsets; int64_t image_bytes, row_stride; }`;
    - `ImageLayout image_layout(const std::array<int64_t,6>&)`;
    - `struct RowSet { ImageLayout layout; int64_t experts, capacity; std::vector<std::string> paths; std::vector<std::array<uint8_t*,6>> slabs; }`;
    - `void require_o_direct(const std::filesystem::path&)`;
    - `bool write_row_image(path, const ImageLayout&, int64_t experts, const std::function<void(int64_t, uint8_t*)>& fill, const std::string& stamp)`;
  - in `stack.h`:
    - `using BenchBuild`;
    - `es::Tables image_tables(const RowSet&)`;
    - `struct StackConfig`.
  - `template <class Build> class Stack`:
    - `explicit Stack(StackConfig)`;
    - `uint8_t* page()`;
    - `uint8_t* lease()`;
    - `int32_t mirror(int64_t row, int32_t expert) const`;
    - `bool wait_handled(uint32_t seq, int64_t deadline_ns) const`;
    - `void set_cpu_layer(int64_t row, int64_t handle)`;
    - `void set_split(const std::array<int64_t, 9>&)`;
    - `std::array<int64_t,3> cpu_stats() const`;
    - `std::array<int64_t, es::kCounterCount> counters() const`;
    - `Tier& tier()`;
    - `void drain_all()` (InstrBuild);
    - `bool drain_stage(es::StageRecord&, uint32_t seq, int64_t deadline_ns)` (InstrBuild).

- [ ] **Step 1: Write `src/row_images.h`**

```cpp
// Row images (python/sglang/srt/layers/moe/exl3_row_image.py): what the tier's RowReader reads with O_DIRECT, straight
// into the slab rows, and the files the bench writes for it. No host header and no ATen here.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fullstack {

constexpr int kNames = 6;  // Exl3RowLayout::kNames: w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh
constexpr int64_t kImageAlign = 512;

struct ImageLayout {
  std::array<int64_t, kNames> row_bytes{};     // each name's slab row
  std::array<int64_t, kNames> name_offsets{};  // where it sits in the image
  int64_t image_bytes = 0;
  int64_t row_stride = 0;  // image_bytes rounded up to a page: expert e's image is at e * row_stride
};

// row_image_layout: the names' slab rows back to back; each a positive multiple of 512 bytes (O_DIRECT).
ImageLayout image_layout(const std::array<int64_t, kNames>& row_bytes);

// What the tier reads: one image file per streamed row and that row's six slabs, `capacity` slots each.
struct RowSet {
  ImageLayout layout;
  int64_t experts = 0;
  int64_t capacity = 0;
  std::vector<std::string> paths;                   // [row]
  std::vector<std::array<uint8_t*, kNames>> slabs;  // [row][name]: slot s at slabs[row][n] + s * row_bytes[n]
};

// Creates `dir` and refuses it unless a file there opens with O_DIRECT (tmpfs does not).
void require_o_direct(const std::filesystem::path& dir);

// Writes `path` as a row-image layer file: expert e's image (fill(e, image) writes image_bytes bytes into a zeroed
// row) at e * row_stride, zero padded, through "<path>.tmp", fsync and rename. A non-empty `stamp` lets an existing
// file of the right size be kept when "<path>.stamp" holds the same stamp. Returns true when the file was written.
bool write_row_image(const std::filesystem::path& path, const ImageLayout& layout, int64_t experts,
                     const std::function<void(int64_t expert, uint8_t* image)>& fill, const std::string& stamp);

}  // namespace fullstack
```

- [ ] **Step 2: Write `src/stack.h`**

```cpp
// The real host stack the bench drives: RamTier (its RowReader, the copy engine on its host backend, the CPU expert
// engine) and its RamThread, set up and torn down in the CPU-experts test's order
// (test_exl3_ram_miss_cpu_experts.py::_host). Production headers, unmodified.
#pragma once

#ifndef EXL3_FULL_STACK_INSTR
#error "EXL3_FULL_STACK_INSTR selects the host build: 0 ProdBuild, 1 InstrBuild"
#endif

#include <immintrin.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "aligned.h"
#include "device_sim.h"
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/core_topology.h"
#include "expert_stream/host/row_reader.h"
#include "expert_stream/host/ram_thread.h"
#include "expert_stream/host/uring_reader.h"
#include "placement.h"
#include "row_images.h"

namespace fullstack {

namespace es = ::sglang::expert_stream;
using BenchBuild = std::conditional_t<EXL3_FULL_STACK_INSTR != 0, es::InstrBuild, es::ProdBuild>;
static_assert(kNames == static_cast<int>(::sglang::exl3::Exl3RowLayout::kNames.size()));

// exl3_ram_miss_host.cpp / exl3_ram_miss_host_instr.cpp's readers, minus the instrumented build's FaultyReader.
template <class Build>
struct ReaderFor;
template <>
struct ReaderFor<es::ProdBuild> {
  using type = es::UringReader;
};
template <>
struct ReaderFor<es::InstrBuild> {
  using type = es::InstrUringReader;
};

// exl3_ram_miss.py::_row_image_tables for one root: file `row` is the row's image, expert e's at e * row_stride,
// identity segments; then the reader's own image-table check.
inline es::Tables image_tables(const RowSet& set) {
  const ImageLayout& layout = set.layout;
  const auto rows = static_cast<int64_t>(set.paths.size());
  es::Tables t;
  t.layers = rows;
  t.experts = set.experts;
  t.parts = 1;
  t.slot_bytes = layout.image_bytes;
  t.paths = set.paths;
  t.source_paths = set.paths;
  t.file_sizes.assign(static_cast<size_t>(rows), set.experts * layout.row_stride);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t e = 0; e < set.experts; ++e) t.extents.push_back(es::Read{row, e * layout.row_stride, layout.image_bytes, 0});
  t.starts.assign(static_cast<size_t>(rows * set.experts), 0);
  for (int n = 0; n < kNames; ++n) t.segments.push_back(es::Segment{n, 0, layout.name_offsets[n], layout.row_bytes[n]});
  t.need_end = layout.image_bytes;
  for (int64_t row = 0; row < rows; ++row) {
    t.slabs.emplace_back(set.slabs[row].begin(), set.slabs[row].end());
    for (int n = 0; n < kNames; ++n)
      t.buffer_regions.push_back(es::RegisteredRegion{set.slabs[row][n],
                                                      static_cast<size_t>(set.capacity * layout.row_bytes[n]),
                                                      static_cast<size_t>(layout.row_bytes[n])});
  }
  t.row_bytes.assign(layout.row_bytes.begin(), layout.row_bytes.end());
  t.images = true;
  es::check_image_tables<::sglang::exl3::Exl3RowLayout>(t);
  return t;
}

struct StackConfig {
  RowSet rows;
  int64_t staging = 3;
  es::CpuExpertForward forward = nullptr;
  int threads = 1;
  std::vector<int> cores;  // worker 0 first: the CPU expert thread pins itself there
  uint8_t* x_base = nullptr;
  int64_t x_stride = 0;
  uint8_t* out_base = nullptr;
  int64_t out_stride = 0;  // bytes; two parts of `hidden` floats per row
  int64_t hidden = 0;
  int service_cpu = -1;
  int copy_cpu = -1;
  int64_t wait_timeout_ns = 2'000'000'000;   // the watchdog's copy-wait deadline (SGLANG_DSV41_RAM_MISS_TIMEOUT_MS)
  int64_t fatal_wait_ns = 30'000'000'000;    // the watchdog's hung-request deadline
  std::array<int64_t, es::kLeaseLanes + 1> split{};
  size_t trace_capacity = 0;  // InstrBuild: the stage trace's ring, 0 off
};

template <class Build>
class Stack {
 public:
  using Source = es::RowReader<::sglang::exl3::Exl3RowLayout, typename ReaderFor<Build>::type, Build>;
  using Tier = es::RamTier<Source>;
  using Thread = es::RamThread<Tier>;
  static constexpr int64_t kCopySpinNs = 5'000'000;  // the transport's enable_copy_engine default (spin_us=5000)
  static constexpr int64_t kCpuSpinNs = 50'000'000;  // enable_cpu_experts' default (spin_us=50_000)

  // Construct on the writer's thread: the request page and lease block are first-touched here.
  explicit Stack(StackConfig config) : config_(std::move(config)) {
    rows_ = static_cast<int64_t>(config_.rows.paths.size());
    experts_ = config_.rows.experts;
    page_ = aligned_zeroed(es::kPageBytes);
    lease_bytes_ = es::kLeaseBlockBytes + round_up(rows_ * es::kDeltaStride, 4096);
    lease_ = aligned_zeroed(lease_bytes_);
    slot_map_.assign(static_cast<size_t>(rows_ * experts_), -1);
    tier_ = std::make_shared<Tier>(page_.get(), slot_map_.data(), lease_.get(), lease_bytes_, image_tables(config_.rows),
                                   std::vector<int64_t>(static_cast<size_t>(rows_), config_.rows.capacity),
                                   /*direct=*/true, /*hot_page=*/nullptr, 0);
    if (!tier_->open()) throw std::runtime_error("the tier's reader did not open (its error is on stderr)");
    tier_->reserve_staging(config_.staging);
    // The copy engine's thread and RamThread's watchdog inherit this thread's affinity: the copy CPU.
    PinScope copy(config_.copy_cpu);
    tier_->enable_copy_engine(-1, kCopySpinNs, config_.wait_timeout_ns);
    tier_->arm_copy_engine(true);
    es::CpuExpertConfig cpu;
    cpu.forward = config_.forward;
    cpu.x_base = config_.x_base;
    cpu.x_stride = config_.x_stride;
    cpu.out_base = config_.out_base;
    cpu.out_stride = config_.out_stride;
    cpu.out_part_stride = config_.hidden * static_cast<int64_t>(sizeof(float));
    cpu.hidden = config_.hidden;
    cpu.threads = config_.threads;
    cpu.cores = config_.cores;
    cpu.spin_ns = kCpuSpinNs;
    tier_->enable_cpu_experts(std::move(cpu), std::vector<int64_t>(config_.split.begin(), config_.split.end()));
    if constexpr (Build::kMetrics) {
      if (config_.trace_capacity > 0) tier_->enable_trace(config_.trace_capacity);
    }
    es::check_dedicated_core(config_.service_cpu, tier_->cpu_cores(), "full-stack bench: ");
    thread_ = std::make_unique<Thread>(tier_, config_.service_cpu, config_.fatal_wait_ns, /*spin_ns=*/0, /*busy_poll=*/true);
    thread_->start();
  }

  // The FFI's stop_thread order: open a gate left closed, stop the service, settle; the tier's destructor then stops
  // the copy and CPU expert threads.
  ~Stack() {
    if (tier_) tier_->open_closed_gate();
    if (thread_) thread_->stop();
    if (tier_) tier_->final_settle();
    thread_.reset();
    tier_.reset();
  }

  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;

  uint8_t* page() const {
    return page_.get();
  }
  uint8_t* lease() const {
    return lease_.get();
  }
  Tier& tier() {
    return *tier_;
  }
  const Tier& tier() const {
    return *tier_;
  }

  // The host's slot-map mirror (published after each read).
  int32_t mirror(int64_t row, int32_t expert) const {
    return __atomic_load_n(&slot_map_[static_cast<size_t>(row * experts_ + expert)], __ATOMIC_ACQUIRE);
  }

  bool wait_handled(uint32_t seq, int64_t deadline_ns) const {
    for (uint32_t spin = 0; !es::reached(tier_->handled_through(), seq); ++spin) {
      if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns) return false;
      _mm_pause();
    }
    return true;
  }

  void set_cpu_layer(int64_t row, int64_t handle) {
    tier_->set_cpu_layer(row, handle);
  }

  void set_split(const std::array<int64_t, es::kLeaseLanes + 1>& split) {
    tier_->set_cpu_split(split.data(), static_cast<int64_t>(split.size()));
  }

  std::array<int64_t, 3> cpu_stats() const {  // {jobs, lanes, forward ns}
    std::array<int64_t, 3> out{};
    tier_->cpu_stats(out.data());
    return out;
  }

  std::array<int64_t, es::kCounterCount> counters() const {
    std::array<int64_t, es::kCounterCount> out{};
    tier_->counters(out.data());
    return out;
  }

  // InstrBuild only (ProdBuild's drain_trace throws): discard every stage record so far. Unconstrained on purpose: the
  // callers' `if constexpr (BenchBuild::kMetrics)` sits in non-template functions, whose discarded branches are still
  // checked, so a constrained member would not compile in the prod build.
  void drain_all() {
    while (tier_->drain_trace(scratch_.get(), 1) == 1) {
    }
  }

  // InstrBuild: the next stage record, which must be request `seq`'s. The service pushes it as it finishes the request,
  // which can trail the CopyDone the writer saw: polled until deadline_ns.
  bool drain_stage(es::StageRecord& out, uint32_t seq, int64_t deadline_ns) {
    while (tier_->drain_trace(&out, 1) != 1) {
      if (monotonic_ns() > deadline_ns) return false;
      _mm_pause();
    }
    return out.seq == static_cast<int64_t>(seq);
  }

 private:
  StackConfig config_;
  int64_t rows_ = 0;
  int64_t experts_ = 0;
  // Declared before tier_ and thread_, so destroyed after them.
  AlignedBuffer page_;
  int64_t lease_bytes_ = 0;
  AlignedBuffer lease_;
  std::vector<int32_t> slot_map_;
  std::unique_ptr<es::StageRecord> scratch_ = std::make_unique<es::StageRecord>();
  std::shared_ptr<Tier> tier_;
  std::unique_ptr<Thread> thread_;
};

}  // namespace fullstack
```

- [ ] **Step 3: Write the failing stamp and stack checks in `src/self_test.cpp`**

Add these includes after the Task 2 includes:

```cpp
#include <filesystem>
#include <mutex>

#include "row_images.h"
#include "stack.h"
```

Insert after `test_copy_wait_gate()` (inside the anonymous namespace):

```cpp
// ---- row images ----

void test_image_stamp(const std::filesystem::path& dir) {
  const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
  CHECK(layout.image_bytes == 3072 && layout.row_stride == 4096 && layout.name_offsets[5] == 2560);
  CHECK_THROWS(image_layout({512, 512, 512, 512, 512, 100}), "multiple of 512");
  const std::filesystem::path path = dir / "selftest-stamp.rows";
  int fills = 0;
  auto fill = [&](int64_t, uint8_t* image) {
    ++fills;
    image[0] = 1;
  };
  CHECK(write_row_image(path, layout, 2, fill, "fixture A"));
  CHECK(!write_row_image(path, layout, 2, fill, "fixture A"));  // same stamp, right size: kept
  CHECK(write_row_image(path, layout, 2, fill, "fixture B"));   // another fixture's images: rewritten
  CHECK(write_row_image(path, layout, 2, fill, ""));            // no stamp: always written
  CHECK(fills == 6 && std::filesystem::file_size(path) == 2 * 4096);
}

// ---- the real stack, synthetic rows, a fake forward ----

constexpr int64_t kSelfRows = 2;
constexpr int64_t kSelfExperts = 8;
constexpr int64_t kSelfCapacity = 7;  // 3 staging slots, 4 mappable
constexpr int64_t kSelfHidden = 64;
constexpr int64_t kFakeHandle = 7;

struct FakeCall {
  int64_t layer;
  std::vector<int32_t> slots;
  std::vector<float> weights;
  int32_t threads;
  int32_t accumulate;
};
std::mutex fake_mutex;
std::vector<FakeCall> fake_calls;

// The kernel's stand-in: out[h] = h + x[0] + sum_i weights[i] * (slots[i] + 1), x[0] read as an integer, so the output
// proves the row's x, the slots and the weights reached it.
int fake_forward(int64_t layer, const void* x, const int32_t* slots, const float* weights, int32_t k, float* out,
                 int32_t threads, int32_t accumulate) {
  uint16_t x0;
  std::memcpy(&x0, x, 2);
  float sum = 0.0f;
  for (int32_t i = 0; i < k; ++i) sum += weights[i] * static_cast<float>(slots[i] + 1);
  for (int64_t h = 0; h < kSelfHidden; ++h)
    out[h] = (accumulate != 0 ? out[h] : 0.0f) + static_cast<float>(h) + static_cast<float>(x0) + sum;
  std::lock_guard<std::mutex> guard(fake_mutex);
  fake_calls.push_back({layer, std::vector<int32_t>(slots, slots + k), std::vector<float>(weights, weights + k), threads,
                        accumulate});
  return 0;
}

uint8_t pattern(int64_t row, int64_t expert, int name) {
  return static_cast<uint8_t>(1 + row * 64 + expert * 6 + name);
}

bool part0_is(const float* part0, float offset) {
  for (int64_t h = 0; h < kSelfHidden; ++h)
    if (part0[h] != static_cast<float>(h) + offset) return false;
  return true;
}

void test_stack(const Placement& placement, const std::filesystem::path& dir) {
  const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
  std::vector<std::array<AlignedBuffer, kNames>> slabs(kSelfRows);
  RowSet set;
  set.layout = layout;
  set.experts = kSelfExperts;
  set.capacity = kSelfCapacity;
  for (int64_t row = 0; row < kSelfRows; ++row) {
    std::array<uint8_t*, kNames> bases{};
    for (int n = 0; n < kNames; ++n) {
      slabs[row][n] = aligned_zeroed(kSelfCapacity * 512);
      bases[n] = slabs[row][n].get();
    }
    set.slabs.push_back(bases);
    const auto path = dir / ("selftest-layer-" + std::to_string(row) + ".rows");
    write_row_image(path, layout, kSelfExperts, [&](int64_t e, uint8_t* image) {
      for (int n = 0; n < kNames; ++n) std::memset(image + layout.name_offsets[n], pattern(row, e, n), 512);
    }, "");
    set.paths.push_back(path.string());
  }
  AlignedBuffer x = aligned_zeroed(kSelfRows * 2 * kSelfHidden);
  AlignedBuffer out = aligned_zeroed(kSelfRows * 2 * kSelfHidden * 4);
  auto* part0 = reinterpret_cast<float*>(out.get());  // row 0, part 0
  StackConfig config;
  config.rows = set;
  config.staging = 3;
  config.forward = &fake_forward;
  config.threads = static_cast<int>(placement.workers.size());
  config.cores.assign(placement.workers.begin(), placement.workers.end());
  config.x_base = x.get();
  config.x_stride = 2 * kSelfHidden;
  config.out_base = out.get();
  config.out_stride = 2 * kSelfHidden * 4;
  config.hidden = kSelfHidden;
  config.service_cpu = placement.service;
  config.copy_cpu = placement.copy;
  config.split = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  config.trace_capacity = 256;
  fake_calls.clear();
  const int64_t timeout = 1'000'000'000;
  {
    PinScope writer(placement.writer);
    Stack<BenchBuild> stack(std::move(config));
    DeviceSim sim(stack.page(), stack.lease(), kSelfRows, kSelfExperts);

    // Load three experts: one uncaptured post of three misses, into staging slots 0, 1, 2; the victims 3, 4, 5 (the
    // lowest free slots) become the staging slots.
    const int32_t three[] = {0, 1, 2};
    const uint32_t loaded = load_experts(sim, 0, three, 3, timeout);
    CHECK(sim.ram_slot(0, 0) == 0 && sim.ram_slot(0, 1) == 1 && sim.ram_slot(0, 2) == 2);
    CHECK(sim.staging(0)[0] == 3 && sim.staging(0)[1] == 4 && sim.staging(0)[2] == 5 && sim.staging(0)[3] == -1);
    CHECK(sim.map_chain(0) == 2);
    CHECK(stack.wait_handled(loaded, monotonic_ns() + timeout));
    CHECK(stack.mirror(0, 0) == 0 && stack.mirror(0, 2) == 2 && stack.mirror(0, 3) == -1);
    bool landed = true;  // the real reader read each expert's image into its slot
    for (int64_t e = 0; e < 3; ++e)
      for (int n = 0; n < kNames; ++n)
        landed = landed && slabs[0][n][e * 512] == pattern(0, e, n) && slabs[0][n][e * 512 + 511] == pattern(0, e, n);
    CHECK(landed);

    // Row 0's layer is not registered yet: a captured post types SM hits, which nothing waits for.
    const int32_t two[] = {0, 1};
    const float halves[] = {0.5f, 0.5f};
    SimRequest r = sim.post(0, two, halves, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
    CHECK(stack.wait_handled(r.seq, monotonic_ns() + timeout));
    CHECK(fake_calls.empty());

    // Registered: split[3] = 3, every lane is the CPU's; one CPU job computes slots 2, 0, 1 into part 0.
    stack.set_cpu_layer(0, kFakeHandle);
    sim.set_row_cpu(0);
    uint16_t mark = 100;
    std::memcpy(x.get(), &mark, 2);  // the post kernel's x store, before the record
    const int32_t order[] = {2, 0, 1};
    const float weights[] = {0.5f, 0.25f, 0.125f};
    if constexpr (BenchBuild::kMetrics) stack.drain_all();
    r = sim.post(0, order, weights, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::kKindHitCpu) && r.kinds[1] == int32_t(w::kKindHitCpu) &&
          r.kinds[2] == int32_t(w::kKindHitCpu));
    CHECK(sim.copy_wait(r, monotonic_ns() + timeout));
    {
      std::lock_guard<std::mutex> guard(fake_mutex);
      CHECK(fake_calls.size() == 1);
      if (fake_calls.size() == 1) {
        const FakeCall& call = fake_calls[0];
        CHECK(call.layer == kFakeHandle && call.slots == std::vector<int32_t>({2, 0, 1}));
        CHECK(call.weights == std::vector<float>({0.5f, 0.25f, 0.125f}));
        CHECK(call.threads == static_cast<int32_t>(placement.workers.size()) && call.accumulate == 0);
      }
    }
    CHECK(part0_is(part0, 100.0f + 2.0f));  // 0.5 * 3 + 0.25 * 1 + 0.125 * 2
    auto cpu = stack.cpu_stats();
    CHECK(cpu[0] == 1 && cpu[1] == 3 && cpu[2] > 0);
    CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateOpen));
    if constexpr (BenchBuild::kMetrics) {
      auto stage = std::make_unique<es::StageRecord>();
      CHECK(stack.drain_stage(*stage, r.seq, monotonic_ns() + timeout));
      CHECK(stage->observed > 0 && stage->done >= stage->observed && stage->lanes == 3);
    }

    // split[n] = 0: no CPU lane, nothing for the CPU.
    stack.set_split({0, 0, 0, 0, 0, 0, 0, 0, 0});
    r = sim.post(0, two, halves, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
    CHECK(stack.wait_handled(r.seq, monotonic_ns() + timeout));
    CHECK(stack.cpu_stats()[0] == 1);
    stack.set_split({0, 1, 2, 3, 4, 5, 6, 7, 8});

    // A hit and a miss: the hit is the CPU's; the miss is read into staging[0] = 3 for the device (cpu_misses off).
    const int32_t mixed[] = {0, 6};
    const float mixed_weights[] = {0.75f, 0.5f};
    mark = 101;
    std::memcpy(x.get(), &mark, 2);
    r = sim.post(0, mixed, mixed_weights, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::kKindHitCpu) && r.kinds[1] == int32_t(w::kKindMissGpu));
    CHECK(r.slots[1] == 3 && r.chain == 3);
    CHECK(sim.copy_wait(r, monotonic_ns() + timeout));
    CHECK(sim.wait_pieces(r, 1, monotonic_ns() + timeout));
    CHECK(part0_is(part0, 101.0f + 0.75f));
    landed = true;
    for (int n = 0; n < kNames; ++n) landed = landed && slabs[0][n][3 * 512] == pattern(0, 6, n);
    CHECK(landed);
    cpu = stack.cpu_stats();
    CHECK(cpu[0] == 2 && cpu[1] == 4);
    sim.sync_row(0, monotonic_ns() + timeout);  // delta 3: expert 6 in slot 3; the victim, slot 6, now staging[0]
    CHECK(sim.ram_slot(0, 6) == 3 && sim.staging(0)[0] == 6);
    const float* row1 = part0 + 2 * kSelfHidden;
    bool untouched = true;
    for (int64_t i = 0; i < 2 * kSelfHidden; ++i) untouched = untouched && row1[i] == 0.0f;
    CHECK(untouched);
  }  // teardown: open the gate, stop the service, settle, stop the copy and CPU threads
  CHECK(fake_calls.size() == 2);
}
```

Replace `run_self_test` with:

```cpp
int run_self_test(const Placement& placement, const std::filesystem::path& image_dir) {
  test_placement();
  test_record_bytes();
  test_seqlock_order();
  test_owed_delta();
  test_typing();
  test_epoch_wrap();
  test_copy_wait_gate();
  require_o_direct(image_dir);
  test_image_stamp(image_dir);
  test_stack(placement, image_dir);
  std::fprintf(stderr, "self-test (%s): %d checks, %d failed\n", std::string(BenchBuild::kName).c_str(), checks,
               failures);
  return failures;
}
```

In `CMakeLists.txt`:

```cmake
  set(FULL_STACK_SOURCES src/full_stack.cpp src/self_test.cpp src/placement.cpp src/device_sim.cpp src/row_images.cpp)
```

- [ ] **Step 4: Commit, SYNC+BUILD (expect a failure: `row_images.cpp` does not exist)**

```bash
git -C $WT add $BENCH/src/row_images.h $BENCH/src/stack.h $BENCH/src/self_test.cpp $BENCH/CMakeLists.txt
git -C $WT commit -F - <<'EOF'
bench(full-stack): Stack over the real tier and thread, row-image interface, and the stack self-test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD**. Expected: nonzero, `Cannot find source file: src/row_images.cpp`. If the failure is instead a
compile error inside `stack.h`, it is a real finding, for example a host header that does not build in this
configuration. Debug it before moving on. If a fix would need a production-header change, stop and ask (Global
Constraints).

- [ ] **Step 5: Implement `src/row_images.cpp`**

```cpp
#include "row_images.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "aligned.h"

namespace fullstack {
namespace fs = std::filesystem;

ImageLayout image_layout(const std::array<int64_t, kNames>& row_bytes) {
  ImageLayout layout;
  int64_t cursor = 0;
  for (int n = 0; n < kNames; ++n) {
    if (row_bytes[n] <= 0 || row_bytes[n] % kImageAlign != 0)
      throw std::runtime_error("row images: name " + std::to_string(n) + "'s slab row is " + std::to_string(row_bytes[n]) +
                               " B, not a positive multiple of 512");
    layout.row_bytes[n] = row_bytes[n];
    layout.name_offsets[n] = cursor;
    cursor += row_bytes[n];
  }
  layout.image_bytes = cursor;
  layout.row_stride = round_up(cursor, 4096);
  return layout;
}

void require_o_direct(const fs::path& dir) {
  fs::create_directories(dir);
  const fs::path probe = dir / ".o_direct_probe";
  const int fd = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0644);
  if (fd < 0)
    throw std::runtime_error("the image directory " + dir.string() + " does not accept O_DIRECT (" + std::strerror(errno) +
                             "): use a disk filesystem, not tmpfs");
  ::close(fd);
  fs::remove(probe);
}

namespace {

std::string read_text(const fs::path& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

void write_all(int fd, const uint8_t* data, size_t bytes, const fs::path& path) {
  while (bytes > 0) {
    const ssize_t n = ::write(fd, data, bytes);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write " + path.string() + ": " + std::strerror(errno));
    }
    data += n;
    bytes -= static_cast<size_t>(n);
  }
}

}  // namespace

bool write_row_image(const fs::path& path, const ImageLayout& layout, int64_t experts,
                     const std::function<void(int64_t, uint8_t*)>& fill, const std::string& stamp) {
  const auto file_bytes = static_cast<uintmax_t>(experts * layout.row_stride);
  const fs::path stamp_path = path.string() + ".stamp";
  std::error_code size_error;
  const uintmax_t size = fs::file_size(path, size_error);
  if (!stamp.empty() && !size_error && size == file_bytes && read_text(stamp_path) == stamp) return false;
  std::error_code ignored;
  fs::remove(stamp_path, ignored);  // a file without its stamp is never reused
  const fs::path tmp = path.string() + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) throw std::runtime_error("open " + tmp.string() + ": " + std::strerror(errno));
  std::vector<uint8_t> row(static_cast<size_t>(layout.row_stride));
  try {
    for (int64_t e = 0; e < experts; ++e) {
      std::fill(row.begin(), row.end(), 0);
      fill(e, row.data());
      write_all(fd, row.data(), row.size(), tmp);
    }
    if (::fsync(fd) != 0) throw std::runtime_error("fsync " + tmp.string() + ": " + std::strerror(errno));
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  fs::rename(tmp, path);
  if (!stamp.empty()) {
    std::ofstream out(stamp_path);
    out << stamp;
    if (!out) throw std::runtime_error("cannot write " + stamp_path.string());
  }
  return true;
}

}  // namespace fullstack
```

- [ ] **Step 6: Commit, SYNC+BUILD, SELFTEST**

```bash
git -C $WT add $BENCH/src/row_images.cpp
git -C $WT commit -F - <<'EOF'
bench(full-stack): row-image files with stamp-gated reuse and the O_DIRECT probe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD**, expecting `BUILD=0`. Then run **SELFTEST**. Expected: `self-test (prod): N checks, 0 failed` and
`self-test (instr): M checks, 0 failed` (instr has 2 more), both `EXIT=0`. A `FATAL exl3 RAM miss: …` line means the
host fail-stopped on something `DeviceSim` wrote; the message names the rule. Fix `device_sim.cpp`, or the test's
expectation if it contradicts `type_lanes`.

---

### Task 4: `StackFixture` and `--validate-only` (48 bit-exact checks)

**Files:**
- Create: `$BENCH/src/stack_fixture.h`, `$BENCH/src/stack_fixture.cpp`
- Modify: `$BENCH/src/full_stack.cpp`, `$BENCH/CMakeLists.txt`

**Interfaces:**
- Consumes:
  - `Fixture` and `compare_reference` (`src/fixture.h`);
  - `exl3_moe_cpu_make_layer`, `exl3_moe_cpu_free_layer`, `exl3_moe_cpu_has_avx512_*` (`moe_mul1.h`);
  - `sglang_exl3_cpu_experts_forward` and `sglang_exl3_cpu_experts_set_cores` (`cpu_experts_cabi.h`);
  - Tasks 1-3.
- Produces:
  - `class StackFixture`:
    - `kCapacity = 8`, `kStaging = 3`;
    - `StackFixture(const fs::path& fixture, const fs::path& image_dir)`;
    - `rows()`, `experts()`, `hidden()`;
    - `const RowSet& row_set()`;
    - `uint8_t* x_row(int64_t)`, `int64_t x_stride()`;
    - `float* out_row(int64_t)`, `int64_t out_stride()`;
    - `void write_x(int64_t)`;
    - `int64_t register_layer(int64_t)`;
    - `static void free_layer(int64_t)`.
  - `void configure_cpu_kernel_runtime()`.
  - `void check_reference(const fs::path&, const std::vector<float>&)`.
  - In `full_stack.cpp`:
    - `class Bench`, with `stack_call(row, k)`, `bare_call(row, k)`, `validate(k, via_stack)`, `describe(request)`,
      `bare_p50_us`;
    - `struct StackCall { int64_t t0, t1; SimRequest request; }`;
    - `class LayerHandles`.

- [ ] **Step 1: Write `src/stack_fixture.h`**

```cpp
// The real workload for the stack, from the eight-layer DSV4.1 fixture (fixture.h): each layer's six EXL3 slabs of
// kCapacity slots, its row-image file, and the CPU experts' x and output rows. No ATen type in this interface, so the
// translation units that include the host headers (and tvm-ffi's) never include ATen. Construct it on a worker-node
// thread: the slabs, x and output rows are first-touched there.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "row_images.h"

namespace fullstack {

class StackFixture {
 public:
  static constexpr int64_t kCapacity = 8;  // 5 experts + 3 staging slots
  static constexpr int64_t kStaging = 3;

  StackFixture(const std::filesystem::path& fixture, const std::filesystem::path& image_dir);
  ~StackFixture();
  StackFixture(const StackFixture&) = delete;
  StackFixture& operator=(const StackFixture&) = delete;

  int64_t rows() const;
  int64_t experts() const;
  int64_t hidden() const;
  const RowSet& row_set() const;
  uint8_t* x_row(int64_t row) const;  // the row's FP16 input, padded to 16 bytes, as the post kernel stages it
  int64_t x_stride() const;
  float* out_row(int64_t row) const;  // part 0 (CPU hits) at [0, hidden), part 1 (CPU misses) at [hidden, 2 hidden)
  int64_t out_stride() const;         // bytes
  void write_x(int64_t row) const;    // the post's x store: the layer's fixture input into the row's x
  int64_t register_layer(int64_t row) const;  // exl3_moe_cpu_make_layer over the row's kCapacity slot views
  static void free_layer(int64_t handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// cpu_forward.cpp's kernel runtime: AVX512BW exactly (EXL3_MOE_CPU_MAX_ISA=bw set at launch), single-threaded ATen,
// no dynamic OpenMP. Throws.
void configure_cpu_kernel_runtime();

// fixture.cpp's compare_reference: bit-exact, or throws naming the file.
void check_reference(const std::filesystem::path& path, const std::vector<float>& actual);

}  // namespace fullstack
```

- [ ] **Step 2: Write `src/stack_fixture.cpp`**

```cpp
#include "stack_fixture.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <omp.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "aligned.h"
#include "fixture.h"
#include "moe_mul1.h"

namespace fullstack {
namespace fs = std::filesystem;

struct StackFixture::Impl {
  int64_t rows = 0;
  int64_t experts = 0;
  int64_t hidden = 0;
  int64_t intermediate = 0;
  RowSet set;
  std::vector<std::array<AlignedBuffer, kNames>> slabs;
  AlignedBuffer x;
  int64_t x_stride = 0;
  AlignedBuffer out;
  int64_t out_stride = 0;
  std::vector<uint8_t> inputs;  // [rows][2 * hidden]: each layer's FP16 fixture input
};

namespace {

int64_t nbytes(const at::Tensor& t) {
  return static_cast<int64_t>(t.nbytes());
}

// Each name's slab row from the fixture's nine matrices per expert (gate, up, down x trellis, suh, svh), in the pinned
// tier's row shape: w13_* hold gate then up ([slot, 2, ...]), w2_* hold down ([slot, 1, ...]).
constexpr std::array<std::array<int, 2>, kNames> kSources = {{{0, 3}, {1, 4}, {2, 5}, {6, -1}, {7, -1}, {8, -1}}};

std::string fixture_stamp(const fs::path& fixture, const ImageLayout& layout) {
  const fs::path absolute = fs::absolute(fixture);
  return absolute.string() + "\n" + std::to_string(fs::file_size(absolute)) + "\n" +
         std::to_string(fs::last_write_time(absolute).time_since_epoch().count()) + "\nimage_bytes " +
         std::to_string(layout.image_bytes) + "\n";
}

}  // namespace

StackFixture::StackFixture(const fs::path& fixture_path, const fs::path& image_dir) : impl_(std::make_unique<Impl>()) {
  Impl& f = *impl_;
  require_o_direct(image_dir);
  const Fixture fixture(fixture_path);
  f.rows = static_cast<int64_t>(fixture.layers.size());
  f.experts = fixture.experts;
  f.hidden = fixture.hidden;
  const auto& m = fixture.layers[0].matrices;
  f.intermediate = m[2][0].size(0);  // gate svh: [n] = I
  const std::array<int64_t, kNames> row_bytes = {2 * nbytes(m[0][0]), 2 * nbytes(m[1][0]), 2 * nbytes(m[2][0]),
                                                 nbytes(m[6][0]),     nbytes(m[7][0]),     nbytes(m[8][0])};
  f.set.layout = image_layout(row_bytes);
  f.set.experts = f.experts;
  f.set.capacity = kCapacity;
  const ImageLayout& layout = f.set.layout;
  const std::string stamp = fixture_stamp(fixture_path, layout);
  f.slabs.resize(static_cast<size_t>(f.rows));
  for (int64_t row = 0; row < f.rows; ++row) {
    std::array<uint8_t*, kNames> bases{};
    for (int n = 0; n < kNames; ++n) {
      f.slabs[row][n] = aligned_zeroed(kCapacity * row_bytes[n]);
      bases[n] = f.slabs[row][n].get();
    }
    f.set.slabs.push_back(bases);
    char name[40];
    std::snprintf(name, sizeof(name), "fixture-layer-%03lld.rows", static_cast<long long>(row));
    const fs::path path = image_dir / name;
    const auto& matrices = fixture.layers[row].matrices;
    write_row_image(path, layout, f.experts, [&](int64_t e, uint8_t* image) {
      for (int n = 0; n < kNames; ++n) {
        uint8_t* cursor = image + layout.name_offsets[n];
        for (int source : kSources[n]) {
          if (source < 0) continue;
          const at::Tensor& t = matrices[source][e];
          std::memcpy(cursor, t.data_ptr(), t.nbytes());
          cursor += t.nbytes();
        }
      }
    }, stamp);
    f.set.paths.push_back(path.string());
  }
  f.x_stride = round_up(2 * f.hidden, 16);  // cpu_experts/service.py: FP16 rows padded to 16 bytes
  f.x = aligned_zeroed(f.rows * f.x_stride);
  f.inputs.resize(static_cast<size_t>(f.rows * 2 * f.hidden));
  for (int64_t row = 0; row < f.rows; ++row) {
    std::memcpy(f.inputs.data() + row * 2 * f.hidden, fixture.layers[row].input.data_ptr(), 2 * f.hidden);
    std::memcpy(f.x.get() + row * f.x_stride, fixture.layers[row].input.data_ptr(), 2 * f.hidden);
  }
  f.out_stride = 2 * f.hidden * static_cast<int64_t>(sizeof(float));
  f.out = aligned_zeroed(f.rows * f.out_stride);
}

StackFixture::~StackFixture() = default;

int64_t StackFixture::rows() const { return impl_->rows; }
int64_t StackFixture::experts() const { return impl_->experts; }
int64_t StackFixture::hidden() const { return impl_->hidden; }
const RowSet& StackFixture::row_set() const { return impl_->set; }
uint8_t* StackFixture::x_row(int64_t row) const { return impl_->x.get() + row * impl_->x_stride; }
int64_t StackFixture::x_stride() const { return impl_->x_stride; }
float* StackFixture::out_row(int64_t row) const {
  return reinterpret_cast<float*>(impl_->out.get() + row * impl_->out_stride);
}
int64_t StackFixture::out_stride() const { return impl_->out_stride; }

void StackFixture::write_x(int64_t row) const {
  std::memcpy(x_row(row), impl_->inputs.data() + row * 2 * impl_->hidden, 2 * impl_->hidden);
}

// cpu_experts/exl3.py::register_layer over the row's slots: gate is w13 part 0, up part 1, down w2's one part; each
// view is contiguous. Activation 0 (silu) with cpu_forward.cpp's limit 10, unswizzled.
int64_t StackFixture::register_layer(int64_t row) const {
  const Impl& f = *impl_;
  const int64_t H = f.hidden, I = f.intermediate;
  const auto& rb = f.set.layout.row_bytes;
  auto view = [&](int name, int64_t slot, int64_t offset, at::IntArrayRef shape, at::ScalarType dtype) {
    return at::from_blob(f.set.slabs[row][name] + slot * rb[name] + offset, shape, at::TensorOptions().dtype(dtype));
  };
  std::array<std::vector<at::Tensor>, 9> m;
  for (int64_t s = 0; s < kCapacity; ++s) {
    for (int part = 0; part < 2; ++part) {
      m[3 * part + 0].push_back(view(0, s, part * rb[0] / 2, {H / 16, I / 16, 48}, at::kShort));
      m[3 * part + 1].push_back(view(1, s, part * rb[1] / 2, {H}, at::kHalf));
      m[3 * part + 2].push_back(view(2, s, part * rb[2] / 2, {I}, at::kHalf));
    }
    m[6].push_back(view(3, s, 0, {I / 16, H / 16, 48}, at::kShort));
    m[7].push_back(view(4, s, 0, {I}, at::kHalf));
    m[8].push_back(view(5, s, 0, {H}, at::kHalf));
  }
  return exl3_moe_cpu_make_layer(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], {}, {}, {}, 0, 10.0, 0);
}

void StackFixture::free_layer(int64_t handle) {
  exl3_moe_cpu_free_layer(handle);
}

void configure_cpu_kernel_runtime() {
  // ISA detection occurs during kernel static initialization, before main.
  if (!exl3_moe_cpu_has_avx512_bw() || exl3_moe_cpu_has_avx512_vnni() || exl3_moe_cpu_has_avx512_vbmi())
    throw std::runtime_error("This study requires AVX512BW; set EXL3_MOE_CPU_MAX_ISA=bw before launch");
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  omp_set_dynamic(0);
}

void check_reference(const fs::path& path, const std::vector<float>& actual) {
  compare_reference(path, actual);
}

}  // namespace fullstack
```

In `CMakeLists.txt`:

```cmake
  set(FULL_STACK_SOURCES src/full_stack.cpp src/self_test.cpp src/placement.cpp src/device_sim.cpp src/row_images.cpp
    src/stack_fixture.cpp src/fixture.cpp "${QUANT}/optimized/moe_mul1.cpp")
```

The kernel source keeps its `-Ofast;-march=native` source property from the existing `foreach(BACKEND …)` loop. That
loop sets it on the same file path.

- [ ] **Step 3: Add the setup, `Bench` and validation to `src/full_stack.cpp`**

Add these includes to `full_stack.cpp`, after `#include <benchmark/benchmark.h>`:

```cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include "cpu_experts_cabi.h"
#include "device_sim.h"
#include "stack.h"
#include "stack_fixture.h"
```

Add `namespace es = ::sglang::expert_stream;` after `namespace fs = std::filesystem;`.

Insert after `resolve_placement` (inside the anonymous namespace):

```cpp
StackConfig stack_config(const StackFixture& f, const Placement& p, const Options& o) {
  StackConfig c;
  c.rows = f.row_set();
  c.staging = StackFixture::kStaging;
  c.forward = &sglang_exl3_cpu_experts_forward;
  c.threads = static_cast<int>(p.workers.size());
  c.cores.assign(p.workers.begin(), p.workers.end());
  c.x_base = f.x_row(0);
  c.x_stride = f.x_stride();
  c.out_base = reinterpret_cast<uint8_t*>(f.out_row(0));
  c.out_stride = f.out_stride();
  c.hidden = f.hidden();
  c.service_cpu = p.service;
  c.copy_cpu = p.copy;
  c.wait_timeout_ns = int64_t{o.wait_timeout_ms} * 1'000'000;
  for (int n = 0; n <= es::kLeaseLanes; ++n) c.split[n] = n;  // every eligible lane is the CPU's
  if constexpr (BenchBuild::kMetrics) c.trace_capacity = 4096;
  return c;
}

// Freed after the stack, whose CPU expert thread uses them: declare before the Stack.
struct LayerHandles {
  std::vector<int64_t> handles;
  ~LayerHandles() {
    for (int64_t handle : handles) StackFixture::free_layer(handle);
  }
};

struct StackCall {
  int64_t t0 = 0;  // before the x store
  int64_t t1 = 0;  // CopyDone seen
  SimRequest request;
};

class Bench {
 public:
  Bench(const Options& options, Placement placement, StackFixture& fixture, Stack<BenchBuild>& stack, DeviceSim& sim,
        std::vector<int64_t> handles)
      : options_(options),
        placement_(std::move(placement)),
        fixture_(fixture),
        stack_(stack),
        sim_(sim),
        handles_(std::move(handles)),
        deadline_ns_(int64_t{options.wait_timeout_ms} * 1'000'000 / 2) {
    for (int k : {1, 3, 5}) {
      for (int i = 0; i < k; ++i) {
        experts_[k].push_back(i);
        // cpu_forward.cpp's routing coefficients, which the frozen references were computed with
        weights_[k].push_back(0.071234f + (k == 1 ? 0.0f : 0.23f * i / (k - 1)));
      }
      for (int64_t row = 0; row < fixture.rows(); ++row) {
        std::vector<int32_t> slots;
        for (int32_t e : experts_[k]) slots.push_back(sim.ram_slot(row, e));
        slots_[k].push_back(std::move(slots));
      }
    }
  }

  const Options& options() const { return options_; }
  const Placement& placement() const { return placement_; }
  Stack<BenchBuild>& stack() { return stack_; }
  int64_t rows() const { return fixture_.rows(); }

  // The device's part of one request: x, the record, the copy wait. t0..t1 is the timed interval.
  StackCall stack_call(int64_t row, int k) {
    StackCall call;
    call.t0 = monotonic_ns();
    fixture_.write_x(row);
    call.request = sim_.post(row, experts_[k], weights_[k], /*captured=*/true, call.t0 + deadline_ns_);
    const bool done = sim_.copy_wait(call.request, call.t0 + deadline_ns_);
    call.t1 = monotonic_ns();
    if (!done) throw std::runtime_error("the copy wait passed its deadline: " + describe(call.request));
    for (int j = 0; j < k; ++j) {
      if (call.request.kinds[j] != static_cast<int32_t>(es::kKindHitCpu))
        throw std::runtime_error("lane " + std::to_string(j) + " was not typed HIT_CPU: " + describe(call.request));
    }
    return call;
  }

  // The C ABI forward on the same layer handle, slots, x and output memory, from the calling thread.
  void bare_call(int64_t row, int k) {
    if (sglang_exl3_cpu_experts_forward(handles_[row], fixture_.x_row(row), slots_[k][row].data(), weights_[k].data(), k,
                                        fixture_.out_row(row), static_cast<int32_t>(placement_.workers.size()), 0) != 0)
      throw std::runtime_error("the bare CPU forward failed");
  }

  // Every layer's output for k experts, through the stack or bare, against reference-e{k}.bin, bit-exact. Each output
  // is NaN-poisoned first, so a forward that did not run fails.
  void validate(int k, bool via_stack) {
    const int64_t hidden = fixture_.hidden();
    std::vector<float> results(static_cast<size_t>(rows() * hidden));
    for (int64_t row = 0; row < rows(); ++row) {
      float* out = fixture_.out_row(row);
      std::fill(out, out + hidden, std::numeric_limits<float>::quiet_NaN());
      if (via_stack) {
        stack_call(row, k);
      } else {
        bare_call(row, k);
      }
      for (int64_t h = 0; h < hidden; ++h)
        if (!std::isfinite(out[h])) throw std::runtime_error("non-finite output in layer " + std::to_string(row));
      std::copy(out, out + hidden, results.begin() + row * hidden);
    }
    check_reference(options_.references / ("reference-e" + std::to_string(k) + ".bin"), results);
  }

  std::string describe(const SimRequest& r) {
    std::ostringstream s;
    s << "gen " << r.gen << " (seq " << r.seq << ") row " << r.row << ", " << r.count << " lanes, kinds";
    for (int j = 0; j < r.count; ++j) s << ' ' << r.kinds[j];
    const auto c = stack_.counters();
    const auto cpu = stack_.cpu_stats();
    s << "; served " << c[es::kServedRequests] << " touch_only " << c[es::kTouchOnly] << " rows_read "
      << c[es::kRowsRead] << " overruns " << c[es::kOverruns] << "; cpu jobs " << cpu[0] << " lanes " << cpu[1]
      << "; CopyDone " << sim_.copy_done(r) << ", gate 0x" << std::hex << sim_.copy_gate() << std::dec
      << ", handled through " << stack_.tier().handled_through();
    return s.str();
  }

  std::map<int, double> bare_p50_us;  // BM_bare's p50 per k, for BM_stack's overhead counter

 private:
  const Options& options_;
  Placement placement_;
  StackFixture& fixture_;
  Stack<BenchBuild>& stack_;
  DeviceSim& sim_;
  std::vector<int64_t> handles_;
  int64_t deadline_ns_;
  std::map<int, std::vector<int32_t>> experts_;
  std::map<int, std::vector<float>> weights_;
  std::map<int, std::vector<std::vector<int32_t>>> slots_;  // [k][row]
};
```

Replace `main` with:

```cpp
int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const Placement placement = resolve_placement(options);
    validate_placement(placement, system_topology(), !options.self_test);
    if (options.self_test) return run_self_test(placement, options.image_dir) == 0 ? 0 : 1;

    setenv("EXL3_MOE_CPU_PIN", "0", 1);
    setenv("EXL3_MOE_CPU_SMALL_WORKERS", "0", 1);
    configure_cpu_kernel_runtime();
    if (sglang_exl3_cpu_experts_set_cores(placement.workers.data(), static_cast<int32_t>(placement.workers.size())) != 0)
      throw std::runtime_error("Cannot configure kernel cores");
    const auto before = task_ids();
    std::unique_ptr<StackFixture> fixture;
    {
      PinScope first_touch(placement.workers.front());  // slabs, x and outputs on the workers' node
      fixture = std::make_unique<StackFixture>(options.fixture, options.image_dir);
    }
    pin_self(placement.writer);  // the writer's stores and clock; the page and lease are first-touched here
    LayerHandles layers;
    Stack<BenchBuild> stack(stack_config(*fixture, placement, options));
    DeviceSim sim(stack.page(), stack.lease(), fixture->rows(), fixture->experts());
    const int64_t timeout_ns = int64_t{options.wait_timeout_ms} * 1'000'000;
    std::vector<int32_t> all(static_cast<size_t>(fixture->experts()));
    std::iota(all.begin(), all.end(), 0);
    for (int64_t row = 0; row < fixture->rows(); ++row) {
      load_experts(sim, row, all, static_cast<int>(StackFixture::kStaging), timeout_ns);
      layers.handles.push_back(fixture->register_layer(row));
      stack.set_cpu_layer(row, layers.handles.back());
      sim.set_row_cpu(row);
    }
    Bench bench(options, placement, *fixture, stack, sim, layers.handles);
    for (int k : {1, 3, 5}) {
      bench.validate(k, /*via_stack=*/true);
      PinScope caller(placement.workers.front());  // the kernel's caller is worker 0, as on the CPU expert thread
      std::this_thread::sleep_for(std::chrono::milliseconds(100));  // that thread's 50 ms idle spin ends first
      bench.validate(k, /*via_stack=*/false);
    }
    const std::vector<int> expected = expected_threads(placement);
    verify_threads(before, expected);
    std::cerr << "Verified 48 bit-exact layer outputs (24 through the stack, 24 bare); threads pinned to {"
              << cpu_list(expected) << "}; writer CPU " << placement.writer << " (" << BenchBuild::kName << " build)\n";
    if (!options.validate_only) throw std::runtime_error("timing is not built yet: run with --validate-only");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
```

- [ ] **Step 4: Commit, SYNC+BUILD, SELFTEST, VALIDATE**

```bash
git -C $WT add $BENCH/src/stack_fixture.h $BENCH/src/stack_fixture.cpp $BENCH/src/full_stack.cpp $BENCH/CMakeLists.txt
git -C $WT commit -F - <<'EOF'
bench(full-stack): StackFixture over the eight-layer fixture and --validate-only's 48 bit-exact checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD**, expecting `BUILD=0`. Run **SELFTEST**, expecting both `0 failed`, `EXIT=0`. Then run
**VALIDATE**. Expected for each of prod and instr: `Verified 48 bit-exact layer outputs …; threads pinned to
{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,36,36}`, then `EXIT=0`. The first run writes 8 image files (~533 MB) into
`fullstack-images`; the second reuses them.

`Bit-exact reference check failed` on the stack path only, with the bare path passing, implicates the stack's slots,
weights, x or out wiring. On both paths, it implicates the slab views or image layout. Debug with
superpowers:systematic-debugging. Never loosen the comparison.

---

### Task 5: `BM_bare`, `BM_stack` and the instrumented breakdown

**Files:**
- Modify: `$BENCH/src/full_stack.cpp`

**Interfaces:**
- Consumes:
  - `Bench` (Task 4);
  - `Stack::drain_all` and `Stack::drain_stage` (Task 3);
  - `es::StageRecord::observed` and `es::StageRecord::done`.
- Produces:
  - benchmarks `BM_bare/experts:{1,3,5}` and `BM_stack/experts:{1,3,5}`;
  - counters:
    - `p50_us`, `p95_us`, `p99_us`, `experts`, `workers`, `layers`;
    - `overhead_p50_us` (stack only);
    - instr only: `pickup_p50_us`, `pickup_p95_us`, `service_p50_us`, `service_p95_us`, `forward_p50_us`,
      `forward_p95_us`, `handoff_p50_us`, `handoff_p95_us`.

- [ ] **Step 1: Add the benchmark functions**

Insert after `class Bench { … };` (inside the anonymous namespace):

```cpp
bool benchmark_failed = false;

double quantile_us(std::vector<double> seconds, double fraction) {
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(seconds.size() - 1));
  std::nth_element(seconds.begin(), seconds.begin() + static_cast<std::ptrdiff_t>(index), seconds.end());
  return seconds[index] * 1e6;
}

void gap(const Options& options) {
  if (options.gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
}

void bm_bare(benchmark::State& state, Bench& bench, int k) {
  try {
    PinScope caller(bench.placement().workers.front());  // the kernel's caller is worker 0 (spec ruling 2)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // the CPU expert thread on this core goes to sleep
    bench.validate(k, false);
    for (int i = 0; i < bench.options().warmup; ++i) bench.bare_call(i % bench.rows(), k);
    std::vector<double> samples;
    int64_t row = 0;
    for (auto _ : state) {
      gap(bench.options());
      const int64_t t0 = monotonic_ns();
      bench.bare_call(row, k);
      const int64_t t1 = monotonic_ns();
      const double seconds = static_cast<double>(t1 - t0) * 1e-9;
      state.SetIterationTime(seconds);
      samples.push_back(seconds);  // sample storage and layer selection are outside the timed interval
      row = (row + 1) % bench.rows();
    }
    bench.validate(k, false);
    if (!samples.empty()) {
      const double p50 = quantile_us(samples, 0.50);
      state.counters["p50_us"] = p50;
      state.counters["p95_us"] = quantile_us(samples, 0.95);
      state.counters["p99_us"] = quantile_us(samples, 0.99);
      bench.bare_p50_us[k] = p50;
    }
    state.counters["experts"] = k;
    state.counters["workers"] = static_cast<double>(bench.placement().workers.size());
    state.counters["layers"] = static_cast<double>(bench.rows());
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}

void bm_stack(benchmark::State& state, Bench& bench, int k) {
  try {
    Stack<BenchBuild>& stack = bench.stack();
    bench.validate(k, true);
    for (int i = 0; i < bench.options().warmup; ++i) bench.stack_call(i % bench.rows(), k);
    std::vector<double> total, pickup, service, forward, handoff;
    [[maybe_unused]] auto stage = std::make_unique<es::StageRecord>();
    if constexpr (BenchBuild::kMetrics) stack.drain_all();
    const auto counters0 = stack.counters();
    const auto cpu0 = stack.cpu_stats();
    int64_t calls = 0;
    int64_t row = 0;
    for (auto _ : state) {
      gap(bench.options());
      const auto cpu_before = stack.cpu_stats();
      const StackCall call = bench.stack_call(row, k);
      const double seconds = static_cast<double>(call.t1 - call.t0) * 1e-9;
      state.SetIterationTime(seconds);
      total.push_back(seconds);
      if constexpr (BenchBuild::kMetrics) {
        // The breakdown, from the stage trace and the CPU expert thread's forward time (exact: one request in flight).
        if (!stack.drain_stage(*stage, call.request.seq, call.t1 + 1'000'000'000))
          throw std::runtime_error("no stage record for request " + std::to_string(call.request.seq));
        const auto cpu_after = stack.cpu_stats();
        const double forward_s = static_cast<double>(cpu_after[2] - cpu_before[2]) * 1e-9;
        pickup.push_back(static_cast<double>(stage->observed - call.t0) * 1e-9);
        service.push_back(static_cast<double>(stage->done - stage->observed) * 1e-9);
        forward.push_back(forward_s);
        handoff.push_back(static_cast<double>(call.t1 - stage->done) * 1e-9 - forward_s);
      }
      ++calls;
      row = (row + 1) % bench.rows();
    }
    // Reconcile: one CPU job of k lanes per call, no read, no overrun.
    const auto counters1 = stack.counters();
    const auto cpu1 = stack.cpu_stats();
    if (cpu1[0] - cpu0[0] != calls || cpu1[1] - cpu0[1] != calls * k)
      throw std::runtime_error("CPU jobs/lanes " + std::to_string(cpu1[0] - cpu0[0]) + "/" +
                               std::to_string(cpu1[1] - cpu0[1]) + " for " + std::to_string(calls) + " calls of " +
                               std::to_string(k) + " lanes");
    if (counters1[es::kRowsRead] != counters0[es::kRowsRead] || counters1[es::kOverruns] != counters0[es::kOverruns])
      throw std::runtime_error("a row read or an overrun during timing");
    if constexpr (BenchBuild::kMetrics) {
      if (stack.tier().trace_dropped() != 0) throw std::runtime_error("the stage trace dropped records");
    }
    bench.validate(k, true);
    if (!total.empty()) {
      const double p50 = quantile_us(total, 0.50);
      state.counters["p50_us"] = p50;
      state.counters["p95_us"] = quantile_us(total, 0.95);
      state.counters["p99_us"] = quantile_us(total, 0.99);
      if (bench.bare_p50_us.contains(k)) state.counters["overhead_p50_us"] = p50 - bench.bare_p50_us[k];
      if constexpr (BenchBuild::kMetrics) {
        state.counters["pickup_p50_us"] = quantile_us(pickup, 0.50);
        state.counters["pickup_p95_us"] = quantile_us(pickup, 0.95);
        state.counters["service_p50_us"] = quantile_us(service, 0.50);
        state.counters["service_p95_us"] = quantile_us(service, 0.95);
        state.counters["forward_p50_us"] = quantile_us(forward, 0.50);
        state.counters["forward_p95_us"] = quantile_us(forward, 0.95);
        state.counters["handoff_p50_us"] = quantile_us(handoff, 0.50);
        state.counters["handoff_p95_us"] = quantile_us(handoff, 0.95);
      }
    }
    state.counters["experts"] = k;
    state.counters["workers"] = static_cast<double>(bench.placement().workers.size());
    state.counters["layers"] = static_cast<double>(bench.rows());
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}
```

- [ ] **Step 2: Replace the validate-only tail of `main` with the benchmark run**

Replace these two lines in `main`:

```cpp
    if (!options.validate_only) throw std::runtime_error("timing is not built yet: run with --validate-only");
    return 0;
```

with:

```cpp
    if (options.validate_only) return 0;
    benchmark::AddCustomContext("backend", EXL3_BENCH_BACKEND);
    benchmark::AddCustomContext("host_build", std::string(BenchBuild::kName));
    benchmark::AddCustomContext("fixture", options.fixture.string());
    benchmark::AddCustomContext("image_dir", options.image_dir.string());
    benchmark::AddCustomContext("writer_cpu", std::to_string(placement.writer));
    benchmark::AddCustomContext("service_cpu", std::to_string(placement.service));
    benchmark::AddCustomContext("copy_cpu", std::to_string(placement.copy));
    benchmark::AddCustomContext("worker_cpus", cpu_list(std::vector<int>(placement.workers.begin(), placement.workers.end())));
    std::ifstream cgroup_file("/proc/self/cgroup");
    benchmark::AddCustomContext("cgroup", std::string((std::istreambuf_iterator<char>(cgroup_file)), {}));
    benchmark::AddCustomContext("gap_us", std::to_string(options.gap_us));
    benchmark::AddCustomContext("compiler", __VERSION__);
    for (int k : {1, 3, 5}) {
      benchmark::RegisterBenchmark("BM_bare/experts:" + std::to_string(k),
                                   [&bench, k](benchmark::State& state) { bm_bare(state, bench, k); })
          ->UseManualTime()
          ->Unit(benchmark::kMicrosecond);
      benchmark::RegisterBenchmark("BM_stack/experts:" + std::to_string(k),
                                   [&bench, k](benchmark::State& state) { bm_stack(state, bench, k); })
          ->UseManualTime()
          ->Unit(benchmark::kMicrosecond);
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    verify_threads(before, expected);
    return benchmark_failed ? 1 : 0;
```
- [ ] **Step 3: Commit, SYNC+BUILD, VALIDATE, a short timing smoke**

```bash
git -C $WT add $BENCH/src/full_stack.cpp
git -C $WT commit -F - <<'EOF'
bench(full-stack): BM_bare and BM_stack, overhead_p50_us, and the instrumented breakdown

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
```

Run **SYNC+BUILD** (`BUILD=0`) and **VALIDATE** (both `EXIT=0`). Then run the smoke test. It checks shape and
counters only; it is not a measurement, since it runs off the isolated partition:

```bash
ssh divix01 'B=/data/models/slang/nvfp4-work/fullstack-build
  export EXL3_MOE_CPU_MAX_ISA=bw OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  for v in prod instr; do
    taskset -c 0-15,36 $B/exl3_full_stack_$v --benchmark_min_time=16x --warmup-forwards=8 --writer-cpu=0 --service-cpu=1 \
      --copy-cpu=36 --cpus=2-15 --host-node=0 --worker-node=0 --image-dir=/data/models/slang/nvfp4-work/fullstack-images
    echo "$v EXIT=$?"
  done'
```

Expected:
- Six rows per build: `BM_bare/experts:1`, `BM_stack/experts:1`, … `:5`.
- Each `BM_stack` row has `overhead_p50_us`.
- The instr rows also have `pickup_p50_us`, `service_p50_us`, `forward_p50_us` and `handoff_p50_us`.
- `forward_p50_us` is close to the same k's `BM_bare` p50.
- Both builds end `EXIT=0`.

---

### Task 6: `run_full_stack.sh`

**Files:**
- Create: `$BENCH/run_full_stack.sh`

**Interfaces:**
- Consumes: `exl3_full_stack_{prod,instr}` (Task 5); under the service, `EXL3BENCH_RESULTS` (set by
  `exl3bench-run`).
- Produces: `BUILD_DIR/RESULTS_DIR` records:
  - `environment.txt`;
  - `round-N-{prod,instr}.{json,log}`;
  - `status.txt`, one `round N VARIANT exit S` line per process;
  - the script's exit status is 1 if any process failed.

- [ ] **Step 1: Write the failing stub test (no script yet)**

```bash
ssh divix01 'R=/data/models/slang/nvfp4-work/wt-fullstack; T=$(mktemp -d -p /data/models/slang/nvfp4-work runner-stub.XXXX)
  mkdir $T/build $T/refs; touch $T/fixture.bin $T/refs/reference-e{1,3,5}.bin
  printf "#!/bin/bash\necho prod \"\$@\"\n" > $T/build/exl3_full_stack_prod
  printf "#!/bin/bash\necho instr; exit 3\n" > $T/build/exl3_full_stack_instr
  chmod +x $T/build/*; echo $T > /data/models/slang/nvfp4-work/runner-stub.path
  EXL3_BENCH_ROUNDS=2 bash $R/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/run_full_stack.sh $T/build $T/out \
    --fixture=$T/fixture.bin --reference-dir=$T/refs --image-dir=$T/img; echo EXIT=$?'
```

Expected: `EXIT=127`, or bash's "No such file or directory". The script does not exist yet.

- [ ] **Step 2: Write `run_full_stack.sh`**

```bash
#!/usr/bin/env bash
# The full-stack CPU-expert bench (README.txt, "Full-stack bench"): alternating prod/instr process rounds, a results
# record like run.sh's, and each process's exit status. Under exl3bench.service, RESULTS_DIR defaults to the fresh
# directory exl3bench-run made (EXL3BENCH_RESULTS). No Python driver.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run_full_stack.sh BUILD_DIR [NEW_RESULTS_DIR] [benchmark options...]' >&2
  exit 2
fi
build=$(realpath "$1")
shift
if [[ $# -gt 0 && $1 != --* ]]; then
  results=$1
  shift
  mkdir "$results" # Refuse to overwrite a prior record.
elif [[ -n ${EXL3BENCH_RESULTS:-} ]]; then
  results=$EXL3BENCH_RESULTS
  if [[ ! -d $results || -n $(ls -A "$results") ]]; then
    echo "EXL3BENCH_RESULTS=$results is not an empty directory" >&2
    exit 2
  fi
else
  echo 'No results directory: name one, or run under exl3bench.service (EXL3BENCH_RESULTS)' >&2
  exit 2
fi
results=$(realpath "$results")
rounds=${EXL3_BENCH_ROUNDS:-8}
if [[ ! $rounds =~ ^[1-9][0-9]*$ ]]; then
  echo 'EXL3_BENCH_ROUNDS must be a positive integer' >&2
  exit 2
fi
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
export EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw EXL3_MOE_CPU_SMALL_WORKERS=0
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fixture=/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin
references=/data/models/exl3_exp/threading
images=/data/models/exl3_exp/google_benchmark/full-stack-images
for arg in "$@"; do
  case $arg in
    --fixture=*) fixture=${arg#--fixture=} ;;
    --reference-dir=*) references=${arg#--reference-dir=} ;;
    --image-dir=*) images=${arg#--image-dir=} ;;
  esac
done
binaries=("$build/exl3_full_stack_prod" "$build/exl3_full_stack_instr")
if [[ -f $build/compile_commands.json ]]; then
  cp "$build/compile_commands.json" "$results/compile_commands.json"
fi
{
  date -Is
  uname -a
  cat /proc/self/cgroup
  sed -n '/Cpus_allowed_list/p;/Mems_allowed_list/p' /proc/self/status
  printf 'Arguments:'
  printf ' %q' "$@"
  printf '\n'
  sha256sum "${binaries[@]}"
  sha256sum "$fixture" "$references"/reference-e{1,3,5}.bin
  echo "Images: $images"
  ls -l "$images" 2>/dev/null || true
  ldd "${binaries[@]}" || true
} > "$results/environment.txt"
failed=0
for ((round=0; round<rounds; ++round)); do
  order=(prod instr)
  if (( round % 2 )); then order=(instr prod); fi
  for variant in "${order[@]}"; do
    status=0
    "$build/exl3_full_stack_$variant" \
      --benchmark_min_time=512x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$variant.json" \
      --benchmark_out_format=json "$@" \
      > "$results/round-$round-$variant.log" 2>&1 || status=$?
    cat "$results/round-$round-$variant.log"
    echo "round $round $variant exit $status" | tee -a "$results/status.txt"
    if (( status != 0 )); then failed=1; fi
  done
done
echo "Results: $results"
exit $failed
```

- [ ] **Step 3: Commit, push, and run the stub checks**

```bash
git -C $WT add $BENCH/run_full_stack.sh
git -C $WT commit -F - <<'EOF'
bench(full-stack): run_full_stack.sh, alternating prod/instr rounds with a results record and exit statuses

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
git -C $WT push origin expert-stream-cpu-bench
ssh divix01 'R=/data/models/slang/nvfp4-work/wt-fullstack; T=$(cat /data/models/slang/nvfp4-work/runner-stub.path)
  S=$R/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/run_full_stack.sh
  git -C $R fetch -q origin && git -C $R checkout -q --detach origin/expert-stream-cpu-bench
  A="--fixture=$T/fixture.bin --reference-dir=$T/refs --image-dir=$T/img"
  EXL3_BENCH_ROUNDS=2 bash $S $T/build $T/out $A > $T/run1.log 2>&1; echo "RUN1 EXIT=$?"; cat $T/out/status.txt
  EXL3_BENCH_ROUNDS=2 bash $S $T/build $T/out $A > $T/run2.log 2>&1; echo "RUN2 EXIT=$?"; grep -c "File exists" $T/run2.log
  mkdir $T/svc; EXL3BENCH_RESULTS=$T/svc EXL3_BENCH_ROUNDS=1 bash $S $T/build $A > $T/run3.log 2>&1; echo "RUN3 EXIT=$?"; cat $T/svc/status.txt
  EXL3BENCH_RESULTS=$T/svc bash $S $T/build $A > $T/run4.log 2>&1; echo "RUN4 EXIT=$?"; cat $T/run4.log
  grep -c "^Arguments: --fixture" $T/out/environment.txt
  rm -rf $T /data/models/slang/nvfp4-work/runner-stub.path'
```

Expected:
- `RUN1 EXIT=1`, with four status lines in this order:
  ```
  round 0 prod exit 0
  round 0 instr exit 3
  round 1 instr exit 3
  round 1 prod exit 0
  ```
- `RUN2 EXIT=1` and `1`: `mkdir` refuses the existing record.
- `RUN3 EXIT=1`, with two status lines written into `$T/svc`.
- `RUN4 EXIT=2` and `EXL3BENCH_RESULTS=… is not an empty directory`.
- The final `grep` prints `1`.

---

### Task 7: The isolated partition widens to `16-33,52-69`, and the README

**Files:**
- Modify:
  - `$BENCH/service/exl3bench-isolation:8`;
  - `$BENCH/service/exl3bench-run:10`;
  - `$BENCH/service/exl3bench.service:2,10`;
  - `$BENCH/service/README.txt:102-103,125`;
  - `$BENCH/README.txt` (append a section).

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the partition `16-33,52-69` in all four places, agreeing.

- [ ] **Step 1: Write the failing agreement check**

```bash
cd $WT/$BENCH/service && grep -c '16-33,52-69' exl3bench-isolation exl3bench-run; grep -c 'AllowedCPUs=16-33 52-69' exl3bench.service; grep -c '18-33,54-69\|18-33 54-69' exl3bench-isolation exl3bench-run exl3bench.service README.txt
```

Expected now: `0`, `0`, `0`, then nonzero counts of the old set. That is the failing state.

- [ ] **Step 2: Edit the four service files**

- `exl3bench-isolation` line 8: `target=18-33,54-69` → `target=16-33,52-69`.
- `exl3bench-run` line 10: `cpus=18-33,54-69` → `cpus=16-33,52-69`.
- `exl3bench.service`:
  - line 2: `Description=Benchmarks on an isolated CPU partition (CPUs 16-33 and siblings 52-69)`;
  - line 10: `AllowedCPUs=16-33 52-69`.
- `service/README.txt`:
  - line 102 becomes `The job runs as dnikolaidis in the isolated partition (CPUs 16-33 and siblings`;
  - line 103 becomes `52-69; 16-17 on NUMA node 0, 18-33 on node 1; memory nodes 0-1), with PATH=/usr/bin:/bin and working`
    (line 104, `directory /data/…`, is unchanged);
  - line 125 becomes `Expected: isolated, 16-33,52-69, 0-1. The cgroup may disappear after completion.`

Add this paragraph to `service/README.txt` right after the line-125 paragraph:

```
The partition holds CPUs 16-17 (and siblings 52-53) since the full-stack bench: its writer and RamThread service run
there, on node 0, as production's service does. Starting the service moves every other thread off 16-17 and 52-53,
including a production server's service thread pinned to 17: do not start it while the server runs. run.sh still pins
itself to 18-33. A changed partition needs one reinstall, with the service stopped:
  sudo bash service/install.sh
```

- [ ] **Step 3: Append the "Full-stack bench" section to `$BENCH/README.txt`**

```
Full-stack bench
----------------
exl3_full_stack_prod and exl3_full_stack_instr time the CPU-expert path above the kernel: a writer thread (the GPU's
stand-in) posts CPU-hit requests into the lease lanes; the real RamTier, RamThread (busy-polling), copy engine (host
backend) and CPU expert engine serve them with the optimized kernel; the writer spins on CopyDone. Design:
docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md.

Build (needs the tvm_ffi package's headers and liburing; the targets exist only with EXL3_TVM_FFI_ROOT):
  cmake -S "$bench" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    -DEXL3_TORCH_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/torch \
    -DEXL3_TVM_FFI_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/tvm_ffi
  cmake --build "$build" -j16 --target exl3_full_stack_prod exl3_full_stack_instr

Placement (defaults; all options):
  --writer-cpu=16    the writer, node 0; the request page and lease block are first-touched there
  --service-cpu=17   RamThread, busy-polling, its physical core to itself
  --copy-cpu=52      the copy thread and the watchdog (the writer's SMT sibling)
  --cpus=18-33       CPU expert worker 0 (the CPU expert thread) and the kernel's 15 helpers, node 1; slabs, x and
                     outputs are first-touched there
  --host-node=0 --worker-node=1   the nodes the runner requires; anything else is refused before setup
Run inside exl3bench.service (partition 16-33,52-69, service/README.txt). Off the partition, use another placement,
for example --writer-cpu=0 --service-cpu=1 --copy-cpu=36 --cpus=2-15 --host-node=0 --worker-node=0 under
taskset -c 0-15,36: correct, but not a measurement.

Row images: --image-dir (default /data/models/exl3_exp/google_benchmark/full-stack-images) must accept O_DIRECT.
The first run writes eight layer files (~533 MB) and a .stamp each; later runs reuse files whose stamp names the same
fixture (path, size, mtime) and image size.

Checks: before timing, all 24 layer outputs through the stack and all 24 bare are compared bit-exactly with
reference-e{1,3,5}.bin, and every thread created during setup must be pinned to its CPU. After each timed benchmark,
its 8 outputs are compared again; BM_stack also requires one CPU job of k lanes per call, no row read and no overrun.
--validate-only stops after the 48 checks. --self-test (no fixture; run under taskset -c 0-15, defaults
--writer-cpu=0 --service-cpu=1 --copy-cpu=2 --cpus=3) checks the writer's records and lane typing against
ram_slot_map.type_lanes and drives the real stack with a fake forward.

Benchmarks, per k in 1, 3, 5 (experts 0..k-1, cpu_forward.cpp's weights, 8 layers rotated):
  BM_bare/experts:k   the C ABI forward, called from worker 0's CPU, on the stack's handles, slots, x and output
  BM_stack/experts:k  x store, record, gate close, spin until CopyDone == G; t0 before the x store, t1 at CopyDone
Counters: p50_us, p95_us, p99_us per call; BM_stack adds overhead_p50_us = its p50 - BM_bare's p50 (same process,
same k). The instr build adds, as p50/p95:
  pickup_us   observed - t0           (the record reaching the service)
  service_us  done - observed         (the service handling the record and submitting the CPU job)
  forward_us  the CPU expert thread's forward time for the call
  handoff_us  (t1 - done) - forward   (CPU job queue, done word, copy thread, CopyDone, the writer's poll)
Fidelity: the writer's stores reach the service by coherence between two node-0 cores, not by PCIe/DDIO; pickup is a
lower bound on the GPU path's. The prod build's numbers are the headline; instr's carry the trace's cost.

Run (from the service: write the job, then start the service):
  printf '%s\n' /bin/bash "$bench/run_full_stack.sh" \
    /data/models/exl3_exp/google_benchmark/full-stack-build \
    > /data/models/exl3_exp/google_benchmark/service-command.txt
run_full_stack.sh BUILD_DIR [NEW_RESULTS_DIR] [options...] alternates prod/instr processes over EXL3_BENCH_ROUNDS
rounds (default 8), 512 calls per benchmark, writes environment.txt, logs, Google JSON and status.txt, and exits 1
when any process failed. Without NEW_RESULTS_DIR it writes to EXL3BENCH_RESULTS (the service's directory).
Do not run it while a production server uses CPUs 16-33.
```

- [ ] **Step 4: Re-run the agreement check, a syntax check, and commit**

```bash
cd $WT/$BENCH/service && grep -c '16-33,52-69' exl3bench-isolation exl3bench-run; grep -c 'AllowedCPUs=16-33 52-69' exl3bench.service; grep -c '18-33,54-69\|18-33 54-69' exl3bench-isolation exl3bench-run exl3bench.service README.txt; bash -n exl3bench-isolation; echo ISO=$?; bash -n exl3bench-run; echo RUN=$?
```

Expected: `1` for each of `exl3bench-isolation`, `exl3bench-run` and `exl3bench.service`; `0` old-set matches in
every file; `ISO=0`, `RUN=0`.

```bash
git -C $WT add $BENCH/service/exl3bench-isolation $BENCH/service/exl3bench-run $BENCH/service/exl3bench.service \
  $BENCH/service/README.txt $BENCH/README.txt
git -C $WT commit -F - <<'EOF'
bench(service): the isolated partition widens to 16-33,52-69 for the full-stack bench; README section

Installing it needs one `sudo bash service/install.sh` with the service stopped.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF
EOF
git -C $WT push origin expert-stream-cpu-bench
```

---

### Task 8: Mutants and the recorded validation

**Files:**
- None committed. Mutants are applied in `$R` on divix01 and reverted (run protocol, "Mutants and concurrent
  lanes").

**Interfaces:**
- Consumes: the built binaries and `src/device_sim.cpp` as written in Task 2.
- Produces: a recorded result per mutant, with the restored baseline green next to it.

Each mutant follows the same shape:
1. Apply it with `git -C $R apply`.
2. Rebuild prod only.
3. Run the prod **VALIDATE** line.
4. Read the message and exit status.
5. Restore with `git -C $R checkout -- <file>` and rebuild.

Use this shell helper on divix01 (paste into each ssh command):

```bash
R=/data/models/slang/nvfp4-work/wt-fullstack; B=/data/models/slang/nvfp4-work/fullstack-build
F=python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/device_sim.cpp
export EXL3_MOE_CPU_MAX_ISA=bw OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
build() { taskset -c 0-63 cmake --build $B -j16 --target exl3_full_stack_prod > $B/mutant-build.log 2>&1; echo BUILD=$?; }
validate() { taskset -c 0-15,36 $B/exl3_full_stack_prod --validate-only --writer-cpu=0 --service-cpu=1 --copy-cpu=36 \
  --cpus=2-15 --host-node=0 --worker-node=0 --image-dir=/data/models/slang/nvfp4-work/fullstack-images > $B/mutant.log 2>&1
  echo EXIT=$?; tail -3 $B/mutant.log; }
```

- [ ] **Step 1: Mutant A, lane weights reversed.** Expected: the bit-exact check fails.

A throwaway one-line edit on divix01, never committed (run protocol). With the helper's variables set:

```bash
sed -i 's/lane ? weights\[j\] : 0.0f/lane ? weights[count - 1 - j] : 0.0f/' $R/$F
git -C $R diff --numstat   # expect: 1	1	…/device_sim.cpp (the mutation applied, once)
build; validate
```

Expected: `BUILD=0`; `EXIT=1`, and the tail shows
`Error: Bit-exact reference check failed: /data/models/exl3_exp/threading/reference-e3.bin`. k = 1 has one lane and
is unchanged; k = 3 is the first to fail.

Restore: `git -C $R checkout -- $F && build`.

- [ ] **Step 2: Mutant B, lane 0's slot off by one.** Expected: the host's own check refuses it.

Captured posts only, so the setup loads (uncaptured misses into staging slots) are unaffected:

```bash
sed -i 's/lane ? r.slots\[j\] : -1/lane ? r.slots[j] + (captured \&\& j == 0 ? 1 : 0) : -1/' $R/$F
git -C $R diff --numstat   # expect: 1	1	…/device_sim.cpp
build; validate
```

Expected: `EXIT=134`, `SIGABRT` from `fail_stop`. The tail shows `FATAL exl3 RAM miss: request … of row 0: lane 0
(expert 0, slot 1): the device maps it there, the tier does not`. This is ruling 8: the host refuses before the kernel
runs, so the bit-exact check is never reached.

Restore: `git -C $R checkout -- $F && build`.

- [ ] **Step 3: Mutant C, demand_head not published for captured posts.** Expected: the copy-wait deadline fires,
  with diagnostics.

```bash
sed -i 's/^  store_release<uint32_t>(page_ + w::kDemandHead, seq);/  if (!captured) store_release<uint32_t>(page_ + w::kDemandHead, seq);/' $R/$F
git -C $R diff --numstat   # expect: 1	1	…/device_sim.cpp
build; validate
```

Expected: `EXIT=1` after about 1 s (half of the 2000 ms timeout). The tail shows `Error: the copy wait passed its
deadline: gen … row 0, 1 lanes, kinds 3; served … cpu jobs 0 …`. The watchdog must not abort first; it would print
`FATAL … a copy wait held the decode stream`.

Restore: `git -C $R checkout -- $F && build`.

- [ ] **Step 4: Restored baseline.** Run `git -C $R status --short` (expect empty), then SYNC+BUILD, SELFTEST and
  VALIDATE.

Expected: a clean tree; `BUILD=0`; both self-tests `0 failed`; both validations `Verified 48 bit-exact layer
outputs`, `EXIT=0`. Record the four results (A, B, C, baseline), with commands and the tail lines, in the ledger.
These mutant runs are only meaningful next to that green baseline.

- [ ] **Step 5: Hand-off for the real measurement (the user's action).** Report to the user:
  - The service files changed. Install them once with `exl3bench.service` stopped and no production server running:
    `sudo bash python/sglang/kernels/jit/csrc/moe/expert_stream/bench/service/install.sh`.
  - Build into `/data/models/exl3_exp/google_benchmark/full-stack-build` with the README's cmake lines.
  - Write the `service-command.txt` line from the README, then start the service.
  - The results land in the service's `EXL3BENCH_RESULTS` directory, which the journal prints.

Do not start the service yourself: it moves threads off cores 16-17 and needs the installed partition.
