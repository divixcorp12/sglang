// Device side of the option C RAM-miss service (DSV41 Phase 3b plan, D10-D15, D22).
//
// post: one block; thread 0 builds the layer's request (need = planned VRAM misses
// whose host-mapped slot map entry is -1; protect = every routed expert), writes it
// into the page's demand ring with volatile stores, fences system-wide and
// release-stores demand_head. The record is posted for every MoE layer (the thread
// uses touch-only records for LRU recency); the wait is armed only when something is
// needed or advisories are on. With `advise`, it also remembers this token's routes
// for `row` and posts the previous token's routes of `next_row` that are not in RAM
// as an advisory record.
// wait: one block; thread 0 polls demand_done with ld.acquire.sys and __nanosleep
// until it reaches the armed sequence or `timeout_ns` of %globaltimer passes, then
// translates the planned experts to pinned slots from the slot map. A timeout, a
// failed request, or a planned row still not in RAM raises the page's fatal word
// (sticky: later posts post nothing and later waits return at once) and sets keep
// to 0, which drops the layer's routed output for this forward.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace exl3_ram_miss_device {

constexpr int kBlock = 32;
// The page layout mirrors exl3_ram_miss_host.cpp and the Python constants;
// test_exl3_ram_miss_device_args checks all three agree.
constexpr int64_t kDemandHead = 0;
constexpr int64_t kDemandDone = 4;
constexpr int64_t kFatal = 8;
constexpr int64_t kAdviseHead = 16;
constexpr int64_t kRecordBytes = 128;
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kAdviseRing = kDemandRing + kDemandRecords * kRecordBytes;
constexpr uint32_t kAdviseRecords = 64;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;
constexpr int64_t kRecRow = 4;
constexpr int64_t kRecNeedCount = 6;
constexpr int64_t kRecProtectCount = 8;
constexpr int64_t kRecStatus = 10;
constexpr int64_t kRecAfter = 12;
constexpr int64_t kRecNeed = 16;
constexpr int64_t kRecProtect = 48;
constexpr uint16_t kServed = 1;

constexpr int kPosted = 0;
constexpr int kPending = 1;
constexpr int kTimeouts = 2;
constexpr int kFailures = 3;
constexpr int kWaits = 4;
constexpr int kPolls = 5;
constexpr int kSticky = 6;
constexpr int kAdvised = 7;
constexpr int kUnservedMisses = 8;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint8_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys(uint8_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ int32_t ld_volatile(const int32_t* address) {
  return *reinterpret_cast<const volatile int32_t*>(address);
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

__device__ __forceinline__ bool listed(const int32_t* ids, int count, int32_t id) {
  for (int i = 0; i < count; ++i) {
    if (ids[i] == id) return true;
  }
  return false;
}

__device__ __forceinline__ void write_record(
    uint8_t* record, uint32_t seq, int64_t row, const int32_t* need, int need_count, const int32_t* protect,
    int protect_count, uint32_t after) {
  volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(record);
  volatile uint16_t* halves = reinterpret_cast<volatile uint16_t*>(record);
  // Seqlock writer: invalidate seq before touching the payload, so a lapped record that
  // is half rewritten never passes the thread's read_record seq re-check.
  words[kRecSeq / 4] = 0u;
  __threadfence_system();
  halves[kRecRow / 2] = static_cast<uint16_t>(row);
  halves[kRecNeedCount / 2] = static_cast<uint16_t>(need_count);
  halves[kRecProtectCount / 2] = static_cast<uint16_t>(protect_count);
  halves[kRecStatus / 2] = 0;
  words[kRecAfter / 4] = after;
  volatile int32_t* need_out = reinterpret_cast<volatile int32_t*>(record + kRecNeed);
  volatile int32_t* protect_out = reinterpret_cast<volatile int32_t*>(record + kRecProtect);
  for (int i = 0; i < kMaxIds; ++i) {
    need_out[i] = i < need_count ? need[i] : -1;
    protect_out[i] = i < protect_count ? protect[i] : -1;
  }
  // The seqlock order the thread's read_record relies on: seq=0, fence, payload, fence, seq last.
  __threadfence_system();
  words[kRecSeq / 4] = seq;
}

__device__ __forceinline__ void raise_fatal(uint8_t* page, uint32_t seq) {
  if (ld_acquire_sys(page + kFatal) == 0) st_release_sys(page + kFatal, seq);
}

}  // namespace exl3_ram_miss_device

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_post_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ slot_map,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int64_t* __restrict__ routes,
    int64_t route_count,
    int64_t row,
    int64_t experts,
    int64_t advise,
    int32_t* __restrict__ last_routes,
    int64_t next_row) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  if (state[kSticky] != 0 || ld_acquire_sys(page + kFatal) != 0) {
    state[kSticky] = 1;
    state[kPending] = 0;
    return;
  }
  const int32_t* map_row = slot_map + row * experts;
  int32_t need[kMaxIds];
  int32_t protect[kMaxIds];
  int need_count = 0;
  int protect_count = 0;
  const int64_t planned_count = min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), static_cast<int64_t>(kMaxIds));
  for (int64_t i = 0; i < planned_count; ++i) {
    const int32_t expert = static_cast<int32_t>(planned[i]);
    if (expert >= 0 && expert < experts && ld_volatile(map_row + expert) < 0 && !listed(need, need_count, expert)) {
      need[need_count++] = expert;
    }
  }
  for (int64_t i = 0; i < route_count && protect_count < kMaxIds; ++i) {
    const int32_t expert = static_cast<int32_t>(routes[i]);
    if (expert >= 0 && expert < experts && !listed(protect, protect_count, expert)) protect[protect_count++] = expert;
  }
  for (int i = 0; i < need_count && protect_count < kMaxIds; ++i) {
    if (!listed(protect, protect_count, need[i])) protect[protect_count++] = need[i];
  }
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) seq = 1;
  state[kPosted] = static_cast<int32_t>(seq);
  uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
  write_record(record, seq, row, need, need_count, protect, protect_count, 0);
  __threadfence_system();
  st_release_sys(page + kDemandHead, seq);
  state[kPending] = (need_count > 0 || advise != 0) ? static_cast<int32_t>(seq) : 0;
  if (advise == 0) return;
  for (int i = 0; i < kMaxIds; ++i) last_routes[row * kMaxIds + i] = i < protect_count ? protect[i] : -1;
  if (next_row < 0) return;
  const int32_t* next_map = slot_map + next_row * experts;
  int32_t ahead[kMaxIds];
  int ahead_count = 0;
  for (int i = 0; i < kMaxIds; ++i) {
    const int32_t expert = last_routes[next_row * kMaxIds + i];
    if (expert >= 0 && expert < experts && ld_volatile(next_map + expert) < 0) ahead[ahead_count++] = expert;
  }
  if (ahead_count == 0) return;
  uint32_t advice = static_cast<uint32_t>(state[kAdvised]) + 1u;
  if (advice == 0) advice = 1;
  state[kAdvised] = static_cast<int32_t>(advice);
  uint8_t* advice_record = page + kAdviseRing + static_cast<int64_t>((advice - 1u) % kAdviseRecords) * kRecordBytes;
  write_record(advice_record, advice, next_row, ahead, ahead_count, ahead, ahead_count, seq);
  __threadfence_system();
  st_release_sys(page + kAdviseHead, advice);
}

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ slot_map,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    int64_t row,
    int64_t experts,
    int64_t lanes,
    int64_t* __restrict__ host_rows,
    float* __restrict__ keep,
    int64_t* __restrict__ ram_miss,
    int64_t timeout_ns) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  bool ok = state[kSticky] == 0;
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  if (ok && seq != 0) {
    state[kWaits] += 1;
    const uint64_t start = global_ns();
    int64_t polls = 0;
    uint32_t done = ld_acquire_sys(page + kDemandDone);
    while (!reached(done, seq) && static_cast<int64_t>(global_ns() - start) < timeout_ns) {
      __nanosleep(256);
      ++polls;
      done = ld_acquire_sys(page + kDemandDone);
    }
    const int64_t total = static_cast<int64_t>(state[kPolls]) + polls;
    state[kPolls] = static_cast<int32_t>(total < 0x7fffffffLL ? total : 0x7fffffffLL);
    if (!reached(done, seq)) {
      state[kTimeouts] += 1;
      raise_fatal(page, seq);
      ok = false;
    } else {
      __threadfence_system();
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
      if (status != kServed) {
        state[kFailures] += 1;
        raise_fatal(page, seq);
        ok = false;
      }
    }
  }
  state[kPending] = 0;
  const int32_t* map_row = slot_map + row * experts;
  const int64_t planned_count = min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), lanes);
  int64_t misses = 0;
  for (int64_t i = 0; i < lanes; ++i) {
    int64_t slot = 0;
    if (i < planned_count) {
      const int64_t expert = planned[i];
      slot = expert >= 0 && expert < experts ? ld_volatile(map_row + expert) : -1;
      if (slot < 0) {
        ++misses;
        slot = 0;
      }
    }
    host_rows[i] = slot;
  }
  if (misses > 0 && ok) {
    // A served (or unarmed) request left a planned row out of RAM: never expected.
    state[kUnservedMisses] += static_cast<int32_t>(misses);
    raise_fatal(page, 0xFFFFFFFFu);
    ok = false;
  }
  if (!ok) state[kSticky] = 1;
  ram_miss[0] += misses;
  keep[0] = ok ? 1.0f : 0.0f;
}

void exl3_ram_miss_post(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView routes,
    int64_t row,
    int64_t advise,
    tvm::ffi::TensorView last_routes,
    int64_t next_row) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_post_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(slot_map.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int64_t*>(routes.data_ptr()),
      routes.size(0),
      row,
      slot_map.size(1),
      advise,
      static_cast<int32_t*>(last_routes.data_ptr()),
      next_row);
}

void exl3_ram_miss_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    int64_t row,
    tvm::ffi::TensorView host_rows,
    tvm::ffi::TensorView keep,
    tvm::ffi::TensorView ram_miss,
    int64_t timeout_ns) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(slot_map.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      row,
      slot_map.size(1),
      host_rows.size(0),
      static_cast<int64_t*>(host_rows.data_ptr()),
      static_cast<float*>(keep.data_ptr()),
      static_cast<int64_t*>(ram_miss.data_ptr()),
      timeout_ns);
}

}  // namespace sglang
