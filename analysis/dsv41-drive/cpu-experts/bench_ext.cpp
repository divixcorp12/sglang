// Standalone binding for exllamav3's CPU mul1 MoE kernel (cpu/moe_mul1.cpp) plus a DRAM read probe.
#include <torch/extension.h>
#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "cpu/moe_mul1.h"

__attribute__((target("avx512f"))) static uint64_t read_chunk(const uint8_t* p, size_t bytes) {
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
  for (size_t i = 0; i + 256 <= bytes; i += 256) {
    acc0 = _mm512_xor_si512(acc0, _mm512_load_si512(p + i));
    acc1 = _mm512_xor_si512(acc1, _mm512_load_si512(p + i + 64));
    acc2 = _mm512_xor_si512(acc2, _mm512_load_si512(p + i + 128));
    acc3 = _mm512_xor_si512(acc3, _mm512_load_si512(p + i + 192));
  }
  acc0 = _mm512_xor_si512(_mm512_xor_si512(acc0, acc1), _mm512_xor_si512(acc2, acc3));
  alignas(64) uint64_t lanes[8];
  _mm512_store_si512(lanes, acc0);
  uint64_t r = 0;
  for (auto v : lanes) r ^= v;
  return r;
}

// Seconds for `threads` threads to each stream-read a disjoint share of `buf` once.
double read_seconds(const at::Tensor& buf, int64_t threads) {
  const auto* base = static_cast<const uint8_t*>(buf.data_ptr());
  const size_t total = static_cast<size_t>(buf.numel()) * buf.element_size();
  const size_t share = (total / threads) & ~size_t(255);
  std::atomic<uint64_t> sink{0};
  std::vector<std::thread> pool;
  const auto t0 = std::chrono::steady_clock::now();
  for (int64_t t = 0; t < threads; ++t)
    pool.emplace_back([&, t] { sink.fetch_xor(read_chunk(base + t * share, share)); });
  for (auto& th : pool) th.join();
  const auto t1 = std::chrono::steady_clock::now();
  if (sink.load() == 0x12345) std::printf(" ");
  return std::chrono::duration<double>(t1 - t0).count();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_layer", &exl3_moe_cpu_make_layer);
  m.def("free_layer", &exl3_moe_cpu_free_layer);
  m.def("forward", &exl3_moe_cpu_forward, py::call_guard<py::gil_scoped_release>());
  m.def("has_avx512_bw", &exl3_moe_cpu_has_avx512_bw);
  m.def("has_avx512_vnni", &exl3_moe_cpu_has_avx512_vnni);
  m.def("read_seconds", &read_seconds, py::call_guard<py::gil_scoped_release>());
}
