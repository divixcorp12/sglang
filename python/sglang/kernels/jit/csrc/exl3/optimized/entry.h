// What the expert-stream host sees of the EXL3 CPU kernel: its concrete kernel type (cpu_experts/binding.hpp lists what
// a kernel type offers), declared here and defined in kernel.cpp, the kernel's own translation unit, which the host
// module compiles with the kernel flags. Plain std types and no kernel internals, so the host and the device
// translation units can include it through formats/exl3/format.h. kernel.h adds the kernel's own typed interface.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// A layer's params: the host builds this struct from the int64 list Python passes ([bits_w13, bits_w2, swizzled,
// row_weighted, plan]; Exl3CpuKernel::params), so Python never restates its layout. slabs: w13_trellis, w13_suh,
// w13_svh, w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts s *
// slot_bytes[i] in, w13 rows holding gate then up. hidden and intermediate are multiples of 128 in [128, 8192];
// activation 0 (gated SiLU, clamped at act_limit when it is nonzero). A layer stores views.
struct SglangExl3CpuParams {
    int16_t bits_w13;      // gate and up trellis bits per weight, 1..8
    int16_t bits_w2;       // down trellis bits per weight, 1..8
    int16_t swizzled;      // 0 or 1: band-contiguous trellis layout; an 8-bit projection is never swizzled
    int16_t row_weighted;  // 0 or 1: the layer's gate/up and down tiles are split by chunk row count
                           // (tile_assignment.hpp), the same arithmetic in another worker's hands
    int32_t plan;          // a SglangExl3CpuPlan (kernel.h); make_layer refuses one that does not accept the layer
};

namespace sglang::exl3_cpu {
// Hidden: the host module's other symbols stay private to its .so, and a standalone build never interposes.
struct __attribute__((visibility("hidden"))) Exl3CpuKernel {
    using Params = SglangExl3CpuParams;
    using Layer = ::sglang::cpu_experts::ExpertLayer<Params>;

    static constexpr bool usable() noexcept { return true; }
    static const char* name() noexcept;
    // std::invalid_argument for a wrong count or a value its field cannot hold; make_layer checks the values.
    static Params params(std::span<const int64_t> values);
    static Layer make_layer(const Layer& shape);
    static int32_t max_routes() noexcept;
    static int32_t max_rows() noexcept;
    static void check(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
    static void forward(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call, ::sglang::cpu_experts::Team& team);
    static int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns);

    // The tier this build runs at (isa.hpp's isa_name): scalar, avx2, bw, vnni or vbmi.
    static std::string isa();
    // The compiled plans (kernel.h's SglangExl3CpuPlanInfo) as a JSON array of objects, the generic plan first;
    // `available` follows the build's -march.
    static std::string plans_json();
    // The most tokens a chunk holds in this build (math.hpp's CHUNK_M = MAX_M / kActRows).
    static int64_t chunk_m();
    // The kernel's options from Python's config, before its first forward: the worker trace (SGLANG_EXPERT_STREAM_TRACE's
    // cpu_worker and gate_path; worker_trace.hpp), which only a -DEXL3_MOE_CPU_WORKER_TRACE=1 build records.
    static void configure(
        std::string worker_trace_prefix, int64_t worker_trace_capacity, int64_t worker_trace_min_us,
        std::string scratch_trace_prefix, std::string trace_gate_path);
};
}  // namespace sglang::exl3_cpu
