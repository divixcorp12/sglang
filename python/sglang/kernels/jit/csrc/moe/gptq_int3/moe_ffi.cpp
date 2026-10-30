// Synchronous GPTQ INT3 CPU experts for a model layer that holds no expert-stream engine (the experimental MiMo
// hybrid checkpoint). One job at a time through the same typed kernel, Team and JobWindow the engine uses.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "../expert_stream/host/cpu_experts/single_job.hpp"
#include "entry.h"

namespace sglang::gptq_int3 {
namespace {

namespace ce = ::sglang::cpu_experts;
using K = GptqInt3CpuKernel;

void check_cpu(const tvm::ffi::TensorView& t, const char* name, int64_t ndim, DLDataTypeCode code, int bits) {
  if (t.device().device_type != kDLCPU || t.ndim() != ndim || t.dtype().code != code || t.dtype().bits != bits ||
      !t.IsContiguous())
    throw std::invalid_argument(std::string("GPTQ INT3 MoE: invalid ") + name);
}

struct GptqInt3MoeObj : public tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.GptqInt3Moe", GptqInt3MoeObj, tvm::ffi::Object);

  // Unpinned: a Team given cores pins its caller, which is the scheduler thread here.
  explicit GptqInt3MoeObj(int64_t threads)
      : team_(std::make_unique<ce::Team>(std::span<const int>{}, static_cast<int>(threads), &K::warm, 0, "int3-moe")),
        runner_(std::make_unique<ce::JobRunner<K>>(*team_)) {}

  ~GptqInt3MoeObj() {
    runner_.reset();
    team_.reset();
  }

  /// slab_ptrs/slab_bytes: int64 [6], each slab's address and its per-slot stride in bytes (CpuExpertLayerSpec.slabs).
  /// x fp16 [rows, hidden]; slots int32 [rows, k]; weights fp32 [rows, k]; out fp32 [rows, hidden], overwritten.
  void forward(int64_t hidden, int64_t intermediate, int64_t capacity, double act_limit,
               const tvm::ffi::TensorView slab_ptrs, const tvm::ffi::TensorView slab_bytes,
               const tvm::ffi::TensorView x, const tvm::ffi::TensorView slots, const tvm::ffi::TensorView weights,
               const tvm::ffi::TensorView out) {
    check_cpu(slab_ptrs, "slab pointers", 1, kDLInt, 64);
    check_cpu(slab_bytes, "slab strides", 1, kDLInt, 64);
    check_cpu(x, "x", 2, kDLFloat, 16);
    check_cpu(slots, "slots", 2, kDLInt, 32);
    check_cpu(weights, "weights", 2, kDLFloat, 32);
    check_cpu(out, "out", 2, kDLFloat, 32);
    const int64_t rows = x.size(0);
    if (slab_ptrs.size(0) != 6 || slab_bytes.size(0) != 6 || x.size(1) != hidden || slots.size(0) != rows ||
        weights.size(0) != rows || weights.size(1) != slots.size(1) || out.size(0) != rows || out.size(1) != hidden)
      throw std::invalid_argument("GPTQ INT3 MoE: inconsistent shapes");
    if (rows == 0) return;
    K::Layer shape;
    shape.capacity = static_cast<int32_t>(capacity);
    shape.hidden = static_cast<int32_t>(hidden);
    shape.intermediate = static_cast<int32_t>(intermediate);
    shape.slab_count = 6;
    shape.act_limit = static_cast<float>(act_limit);
    const auto* ptrs = static_cast<const int64_t*>(slab_ptrs.data_ptr());
    const auto* bytes = static_cast<const int64_t*>(slab_bytes.data_ptr());
    for (int s = 0; s < 6; ++s) {
      shape.slabs[s] = reinterpret_cast<const void*>(ptrs[s]);
      shape.slot_bytes[s] = static_cast<uint64_t>(bytes[s]);
    }
    const auto layer = K::make_layer(shape);
    ce::ForwardCall call;
    call.rows = static_cast<int32_t>(rows);
    call.k = static_cast<int32_t>(slots.size(1));
    call.x = x.data_ptr();
    call.slots = static_cast<const int32_t*>(slots.data_ptr());
    call.weights = static_cast<const float*>(weights.data_ptr());
    call.out = static_cast<float*>(out.data_ptr());
    call.out_mode = ce::OutMode::kOverwrite;
    runner_->run(layer, call);
  }

  /// The lossless adapter's bit repacking (cpu_experts/gptq_int3.py prepare_projection): canonical qweight int32
  /// [K/32*3, N], each column a 3-bit code stream, into low2 uint8 [N, K/4] and high1 uint8 [N, K/8] planes.
  void adapt(const tvm::ffi::TensorView qweight, const tvm::ffi::TensorView low, const tvm::ffi::TensorView high) {
    check_cpu(qweight, "qweight", 2, kDLInt, 32);
    check_cpu(low, "low2 plane", 2, kDLUInt, 8);
    check_cpu(high, "high1 plane", 2, kDLUInt, 8);
    const int64_t n = qweight.size(1), k = qweight.size(0) / 3 * 32;
    if (qweight.size(0) % 3 || k % 128 || low.size(0) != n || low.size(1) != k / 4 || high.size(0) != n ||
        high.size(1) != k / 8)
      throw std::invalid_argument("GPTQ INT3 MoE: inconsistent adapter shapes");
    const auto* q = static_cast<const uint32_t*>(qweight.data_ptr());
    auto* lo = static_cast<uint8_t*>(low.data_ptr());
    auto* hi = static_cast<uint8_t*>(high.data_ptr());
#pragma omp parallel for schedule(static)
    for (int64_t col = 0; col < n; ++col) {
      for (int64_t k8 = 0; k8 < k; k8 += 8) {
        uint8_t h = 0, l0 = 0, l1 = 0;
        for (int j = 0; j < 8; ++j) {
          const int64_t bit = (k8 + j) * 3, word = bit >> 5;
          const int shift = static_cast<int>(bit & 31);
          uint32_t v = q[word * n + col] >> shift;
          if (shift > 29) v |= q[(word + 1) * n + col] << (32 - shift);
          v &= 7;
          h |= static_cast<uint8_t>((v >> 2) << j);
          if (j < 4)
            l0 |= static_cast<uint8_t>((v & 3) << (2 * j));
          else
            l1 |= static_cast<uint8_t>((v & 3) << (2 * (j - 4)));
        }
        hi[col * (k / 8) + k8 / 8] = h;
        lo[col * (k / 4) + k8 / 4] = l0;
        lo[col * (k / 4) + k8 / 4 + 1] = l1;
      }
    }
  }

  int64_t max_rows() const { return K::max_rows(); }
  int64_t max_routes() const { return K::max_routes(); }

 private:
  std::unique_ptr<ce::Team> team_;
  std::unique_ptr<ce::JobRunner<K>> runner_;
};

void register_gptq_int3_moe() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<GptqInt3MoeObj>()
      .def(refl::init<int64_t>(), "__init__")
      .def("forward", &GptqInt3MoeObj::forward)
      .def("adapt", &GptqInt3MoeObj::adapt)
      .def("max_rows", &GptqInt3MoeObj::max_rows)
      .def("max_routes", &GptqInt3MoeObj::max_routes);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_once, register_gptq_int3_moe);

}  // namespace sglang::gptq_int3
