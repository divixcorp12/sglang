// One expert slot of a layer's pinned host tier, and the layer's slots. Slot s of slab i is at base[i] + s *
// stride[i]; the quant decodes a slot's bytes into its Row. Views only: nothing is copied or owned.
#pragma once
#include "../../../../kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

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
    int32_t capacity;  // slots per slab: ExpertForward refuses a routed slot outside [0, capacity)

    static MoeBufferRows of(const SglangCpuExpertsLayer& d)
    {
        MoeBufferRows r{};
        r.capacity = d.capacity;
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

}  // namespace
}  // namespace sglang::cpu_experts
