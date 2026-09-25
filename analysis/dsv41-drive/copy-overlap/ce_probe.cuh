// Probe (ii) for copy-engine C1: a host service thread issues cudaMemcpyAsync on its own stream while the main
// thread is blocked in cudaGraphLaunch of a graph that spins on the completion word those copies publish.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

namespace ce_probe {

inline int64_t realtime_ns() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// counter[0]: the request sequence, advanced by the post kernel. The post word lives in mapped host memory.
__global__ void post_kernel(uint32_t* post_word, uint32_t* counter, uint64_t* stamps, int64_t stamp_cap) {
  const uint32_t seq = counter[0] + 1;
  counter[0] = seq;
  const int64_t i = static_cast<int64_t>(seq - 1) % stamp_cap;
  stamps[2 * i] = global_ns();
  __threadfence_system();
  asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(post_word), "r"(seq) : "memory");
}

// Spins until the device done word reaches the current sequence, or times out and latches fail.
__global__ void wait_kernel(const uint32_t* done_word, const uint32_t* counter, uint64_t* stamps, int64_t stamp_cap,
                            uint32_t* fail, uint64_t timeout_ns) {
  const uint32_t seq = counter[0];
  const uint64_t start = global_ns();
  uint32_t seen;
  while (true) {
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(done_word) : "memory");
    if (seen >= seq) break;
    if (global_ns() - start > timeout_ns) {
      atomicAdd(fail, 1u);
      break;
    }
  }
  stamps[2 * (static_cast<int64_t>(seq - 1) % stamp_cap) + 1] = global_ns();
}

// A host node standing in for the decode graph's Engram host callbacks: about 50 us of host work.
void CUDART_CB host_node_fn(void*) {
  const int64_t end = realtime_ns() + 50000;
  while (realtime_ns() < end) {
  }
}

__global__ void filler_kernel(float* x) { x[threadIdx.x] = x[threadIdx.x] * 0.999f + 1.0f; }

struct Probe {
  std::thread thread;
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> served{0};
  std::atomic<int> error{0};
  std::vector<int64_t> seen_ns, issued_ns;  // host CLOCK_REALTIME per request
  std::vector<int64_t> api_ns;              // host time spent in CUDA API calls per request
};

Probe* g_probe = nullptr;


void service_loop(Probe* p, int device, volatile uint32_t* post_word, uint32_t* done_word_dev, uint32_t* gen_ring_host,
                  std::vector<int64_t> src, std::vector<int64_t> dst, std::vector<int64_t> bytes, int64_t src_rows,
                  int64_t rows_per_request, int64_t dst_slots) {
  if (cudaSetDevice(device) != cudaSuccess) {
    p->error = 1;
    return;
  }
  cudaStream_t stream;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    p->error = 2;
    return;
  }
  uint32_t last = 0;
  int64_t cursor = 0;
  while (!p->stop.load(std::memory_order_relaxed)) {
    const uint32_t seq = __atomic_load_n(const_cast<uint32_t*>(post_word), __ATOMIC_ACQUIRE);
    if (seq == last) continue;
    const int64_t seen = realtime_ns();
    last = seq;
    int64_t api = 0;
    for (int64_t r = 0; r < rows_per_request; ++r) {
      const int64_t row = cursor++ % src_rows;
      const int64_t slot = r % dst_slots;
      for (size_t s = 0; s < src.size(); ++s) {
        const int64_t t0 = realtime_ns();
        if (cudaMemcpyAsync(reinterpret_cast<void*>(dst[s] + slot * bytes[s]), reinterpret_cast<void*>(src[s] + row * bytes[s]),
                            bytes[s], cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          p->error = 3;
          return;
        }
        api += realtime_ns() - t0;
      }
    }
    uint32_t* gen = gen_ring_host + (seq % 1024);
    *gen = seq;
    const int64_t t0 = realtime_ns();
    if (cudaMemcpyAsync(done_word_dev, gen, 4, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      p->error = 4;
      return;
    }
    api += realtime_ns() - t0;
    p->seen_ns.push_back(seen);
    p->issued_ns.push_back(realtime_ns());
    p->api_ns.push_back(api);
    p->served.fetch_add(1);
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

}  // namespace ce_probe

namespace sglang {

void ce_probe_start(tvm::ffi::TensorView post_word, tvm::ffi::TensorView done_word, tvm::ffi::TensorView gen_ring,
                    tvm::ffi::TensorView src_ptrs, tvm::ffi::TensorView dst_ptrs, tvm::ffi::TensorView seg_bytes,
                    int64_t src_rows, int64_t rows_per_request, int64_t dst_slots, int64_t device) {
  host::RuntimeCheck(ce_probe::g_probe == nullptr, "probe already running");
  const auto n = static_cast<size_t>(seg_bytes.size(0));
  std::vector<int64_t> src(n), dst(n), bytes(n);
  for (size_t i = 0; i < n; ++i) {
    src[i] = static_cast<const int64_t*>(src_ptrs.data_ptr())[i];
    dst[i] = static_cast<const int64_t*>(dst_ptrs.data_ptr())[i];
    bytes[i] = static_cast<const int64_t*>(seg_bytes.data_ptr())[i];
  }
  auto* p = new ce_probe::Probe();
  ce_probe::g_probe = p;
  p->thread = std::thread(ce_probe::service_loop, p, static_cast<int>(device),
                          static_cast<volatile uint32_t*>(post_word.data_ptr()), static_cast<uint32_t*>(done_word.data_ptr()),
                          static_cast<uint32_t*>(gen_ring.data_ptr()), src, dst, bytes, src_rows, rows_per_request, dst_slots);
}

// Stops the service and writes [served, error, then per request: seen_ns, issued_ns, api_ns] into out (int64, CPU).
int64_t ce_probe_stop(tvm::ffi::TensorView out) {
  auto* p = ce_probe::g_probe;
  host::RuntimeCheck(p != nullptr, "probe not running");
  p->stop = true;
  p->thread.join();
  auto* o = static_cast<int64_t*>(out.data_ptr());
  const int64_t cap = (out.size(0) - 2) / 3;
  const int64_t n = std::min<int64_t>(static_cast<int64_t>(p->seen_ns.size()), cap);
  o[0] = static_cast<int64_t>(p->served.load());
  o[1] = p->error.load();
  for (int64_t i = 0; i < n; ++i) {
    o[2 + 3 * i] = p->seen_ns[i];
    o[3 + 3 * i] = p->issued_ns[i];
    o[4 + 3 * i] = p->api_ns[i];
  }
  delete p;
  ce_probe::g_probe = nullptr;
  return n;
}

void ce_probe_post(tvm::ffi::TensorView post_word, tvm::ffi::TensorView counter, tvm::ffi::TensorView stamps) {
  const auto device = host::LaunchKernel::resolve_device(counter.device());
  host::LaunchKernel(1, 1, device)(ce_probe::post_kernel, static_cast<uint32_t*>(post_word.data_ptr()),
                                    static_cast<uint32_t*>(counter.data_ptr()), static_cast<uint64_t*>(stamps.data_ptr()),
                                    static_cast<int64_t>(stamps.size(0) / 2));
}

void ce_probe_wait(tvm::ffi::TensorView done_word, tvm::ffi::TensorView counter, tvm::ffi::TensorView stamps,
                   tvm::ffi::TensorView fail, int64_t timeout_ns) {
  const auto device = host::LaunchKernel::resolve_device(counter.device());
  host::LaunchKernel(1, 1, device)(ce_probe::wait_kernel, static_cast<const uint32_t*>(done_word.data_ptr()),
                                    static_cast<const uint32_t*>(counter.data_ptr()), static_cast<uint64_t*>(stamps.data_ptr()),
                                    static_cast<int64_t>(stamps.size(0) / 2), static_cast<uint32_t*>(fail.data_ptr()),
                                    static_cast<uint64_t>(timeout_ns));
}

void ce_probe_filler(tvm::ffi::TensorView x) {
  const auto device = host::LaunchKernel::resolve_device(x.device());
  host::LaunchKernel(1, 32, device)(ce_probe::filler_kernel, static_cast<float*>(x.data_ptr()));
}

void ce_probe_host_node(tvm::ffi::TensorView x) {
  const auto device = x.device();
  const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
  host::RuntimeCheck(cudaLaunchHostFunc(stream, ce_probe::host_node_fn, nullptr) == cudaSuccess, "cudaLaunchHostFunc failed");
}

}  // namespace sglang
