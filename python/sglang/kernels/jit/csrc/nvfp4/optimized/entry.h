// What the expert-stream host sees of the NVFP4 CPU kernel: its concrete kernel type (cpu_experts/binding.hpp lists what
// a kernel type offers), declared here and defined in kernel.cpp, the kernel's own translation unit (with the vendored
// upstream/nvfp4.c, compiled as C), which the host module compiles with the kernel flags. Plain std types and no kernel
// internals, so formats/nvfp4/format.h can include it from the host and device translation units.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

// A layer's params: Nvfp4CpuKernel::params builds this struct from the int64 list Python passes, so Python never
// restates its layout. Packed row-major E2M1 weights (low nibble first), GPU-ready 128x4-swizzled E4M3 scales. Slab
// strides are BYTES per host slot. slabs: w13, w2, sf13, sf2, gate_alpha, down_alpha, optional up_alpha (null shares
// gate_alpha). Alphas are FP32 scalars per slot. activation 0 is ordinary SiLU(gate)*up; act_limit L > 0 clamps
// gate=min(gate,L), up=clamp(up,-L,L) before SiLU. A layer stores views, never repacks.
struct SglangNvfp4CpuParams {
    // 0: [gate,up], 1: [up,gate], 2: alternating 64-row [up,gate] chunks.
    int32_t w13_layout;
    // Cancel activation scales folded into GPU GEMM alphas; use 1 for weight-only alphas.
    float inv_input_scale13, inv_input_scale2;
};

namespace sglang::nvfp4_cpu {
// Hidden: the host module's other symbols stay private to its .so, and a standalone build never interposes.
struct __attribute__((visibility("hidden"))) Nvfp4CpuKernel {
    using Params = SglangNvfp4CpuParams;
    using Layer = ::sglang::cpu_experts::ExpertLayer<Params>;

    static constexpr bool usable() noexcept { return true; }
    static const char* name() noexcept;
    // From [w13_layout, inv_input_scale13 bits, inv_input_scale2 bits]: the scales are IEEE binary32 bit patterns in
    // the low 32 bits. std::invalid_argument for a wrong count or a value its field cannot hold; make_layer checks the
    // values themselves.
    static Params params(std::span<const int64_t> values);
    static Layer make_layer(const Layer& shape);
    static int32_t max_routes() noexcept;
    static int32_t max_rows() noexcept;
    static void check(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
    // Runs the call on `team` by the job interface below with a static split of each phase and a barrier between the
    // phases: the engine's path until it admits jobs to a JobWindow, when this goes.
    static void forward(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call, ::sglang::cpu_experts::Team& team);
    static int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns);

    // The job interface (cpu_experts/binding.hpp's CpuExpertJobKernel): one call as four phases of tasks (forward_plan.hpp
    // lays them out), PrepareInput (32-value blocks), GateUp (16-row units), Middle (Q8_0 blocks) and Down (16-row units).
    // Down is the first phase that reads the second stage's slabs (kGatePhase) and the one that writes `out`. A Job is
    // one slot's state, with its scratch. After a Q8_0 check fails, the phases that follow have no tasks and status is 2.
    struct __attribute__((visibility("hidden"))) Job {
        struct Impl;
        Job();
        ~Job();
        Job(const Job&) = delete;
        Job& operator=(const Job&) = delete;
        std::unique_ptr<Impl> impl;
    };
    static constexpr int kPhases = 4;
    static constexpr int kGatePhase = 3;
    // Throws std::invalid_argument for a call with more tasks in a phase than an int counts.
    static void prepare(Job& job, const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
    static ::sglang::cpu_experts::Tasks tasks(const Job& job, int phase);
    static void run(Job& job, int phase, int t0, int t1, int worker);
    static int status(const Job& job);
};
}  // namespace sglang::nvfp4_cpu
