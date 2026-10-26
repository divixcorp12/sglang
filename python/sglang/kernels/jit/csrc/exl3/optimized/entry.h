// What the expert-stream host sees of the EXL3 CPU kernel: its concrete kernel type (cpu_experts/binding.hpp lists what
// a kernel type offers), declared here and defined in kernel.cpp, the kernel's own translation unit, which the host
// module compiles with the kernel flags. Plain std types and no kernel internals, so the host and the device
// translation units can include it through formats/exl3/format.h. It also carries the kernel's typed plan interface.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

// A layer's params: the host builds this struct from the int64 list Python passes ([bits_w13, bits_w2, swizzled,
// plan]; Exl3CpuKernel::params), so Python never restates its layout. slabs: w13_trellis, w13_suh,
// w13_svh, w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts s *
// slot_bytes[i] in, w13 rows holding gate then up. hidden and intermediate are multiples of 128 in [128, 8192];
// activation 0 (gated SiLU, clamped at act_limit when it is nonzero). A layer stores views.
struct SglangExl3CpuParams {
    int16_t bits_w13;      // gate and up trellis bits per weight, 1..8
    int16_t bits_w2;       // down trellis bits per weight, 1..8
    int16_t swizzled;      // 0 or 1: band-contiguous trellis layout; an 8-bit projection is never swizzled
    int32_t plan;          // a SglangExl3CpuPlan; make_layer refuses one that does not accept the layer
};

// The forward plans this library compiles (forward_plan.hpp's run_plan), by the id a layer's params name. The generic
// plan runs any layer make_layer accepts; a fixed plan runs only the layer facts its Shape (shapes.hpp) fixes.
enum SglangExl3CpuPlan : int32_t {
    kSglangExl3CpuPlanGeneric = 0,  // ForwardPlan<GenericShape>
    kSglangExl3CpuPlanDsv41 = 1,    // ForwardPlan<Dsv41Shape>
};

// One compiled plan as exl3_cpu_plans lists it (and the host export expert_stream_cpu_kernel_plans as JSON). A fixed
// plan accepts exactly the layer facts below; the generic plan (fixed false, facts 0) accepts any layer. available:
// min_isa is within the build's tier; make_layer refuses an unavailable plan.
struct SglangExl3CpuPlanInfo {
    int32_t id;
    const char* name;
    bool fixed;
    int32_t hidden, intermediate, bits_w13, bits_w2, swizzled;
    float act_limit;
    const char* min_isa;
    bool available;
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
    static int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns);

    // The job interface (cpu_experts/binding.hpp's CpuExpertJobKernel): one call as five phases of tasks (forward_plan.hpp
    // lays them out), PrepareGateUp, GateUp, Middle, DownTiles and DownOut. Middle is the first phase that reads w2
    // (kGatePhase) and DownOut the one that writes `out` (kOutPhase). A Job is one slot's state, with its scratch.
    struct __attribute__((visibility("hidden"))) Job {
        struct Impl;
        Job();
        ~Job();
        Job(const Job&) = delete;
        Job& operator=(const Job&) = delete;
        std::unique_ptr<Impl> impl;
    };
    static constexpr int kPhases = 5;
    static constexpr int kGatePhase = 2;
    static constexpr int kOutPhase = 4;
    // Throws std::invalid_argument for a plan this build does not compile.
    static void prepare(Job& job, const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
    static ::sglang::cpu_experts::Tasks tasks(const Job& job, int phase);
    static void run(Job& job, int phase, int t0, int t1, int worker);
    static int status(const Job& job);

    // The tier this build runs at (isa.hpp's isa_name): scalar, avx2, bw, vnni or vbmi.
    static std::string isa();
    // The compiled plans (SglangExl3CpuPlanInfo) as a JSON array of objects, the generic plan first;
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

// The compiled plans, the generic plan first; `available` follows the build's -march.
__attribute__((visibility("hidden"))) std::vector<SglangExl3CpuPlanInfo> exl3_cpu_plans();
// The first available fixed plan that accepts these layer facts, else the generic plan: for native harnesses (the
// benches); Python chooses from the plans' list itself.
__attribute__((visibility("hidden"))) int32_t exl3_cpu_match_plan(
    int32_t hidden, int32_t intermediate, int32_t bits_w13, int32_t bits_w2, int32_t swizzled, float act_limit);
}  // namespace sglang::exl3_cpu
