// The device's memory accessors for words shared with the host or another kernel: each call site states its ordering.
// Moved verbatim from lease_device.cuh so the lease channel (lease_channel.cuh) and the target's kernels share them.
#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {
namespace device::expert_stream {

SGL_DEVICE uint32_t ld_acquire_sys(const uint8_t* address) {
  return ::sglang::device::ptx::load_acquire_sys(reinterpret_cast<const uint32_t*>(address));
}

SGL_DEVICE void st_release_sys(uint8_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

SGL_DEVICE uint64_t ld_acquire_sys64(const uint8_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

SGL_DEVICE void st_release_sys64(uint8_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

SGL_DEVICE void st_relaxed_sys_v2(uint8_t* address, uint32_t x, uint32_t y) {
  asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(address), "r"(x), "r"(y) : "memory");
}

SGL_DEVICE void st_relaxed_sys_v4(uint8_t* address, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address), "r"(x), "r"(y), "r"(z), "r"(w)
               : "memory");
}

SGL_DEVICE uint4 ld_relaxed_sys_v4(const uint8_t* address) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(address)
               : "memory");
  return v;
}

// Relaxed system-scope accessors. Every word the host or another kernel accesses concurrently goes through these or
// the acquire/release forms above, so each call site states its ordering.
//
// A relaxed access is a volatile access: the PTX memory model treats ld/st.volatile as relaxed at system scope, and
// nvcc emits LDG/STG.E.STRONG.SYS for it. cuda::atomic_ref is avoided: with CUDA 13.4's libcu++, an access through a
// __grid_constant__ parameter gains a run-time local-pointer check with a byte-copy fallback, and adjacent relaxed
// accesses are merged and reordered.
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

}  // namespace device::expert_stream
}  // namespace sglang
