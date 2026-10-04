// A toy CPU expert quant for cpu_experts_common's tests: one slab of float[hidden] per slot; a forward writes
// out[t][h] (+)= sum over routes of weight * scale * slab[slot][h]. TOY_TOP_ISA picks kTopIsa (default Avx2).
#pragma once
#include "../../../../python/sglang/srt/layers/quantization/cpu_experts_common/expert_forward.hpp"
#include "../../../../python/sglang/srt/layers/quantization/cpu_experts_common/cabi.hpp"
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#ifndef TOY_TOP_ISA
#define TOY_TOP_ISA Avx2
#endif

namespace toy {
namespace {
using namespace sglang::cpu_experts;
struct ToyParams { float scale; };
struct ToyQuant {
    static constexpr const char* kName = "toy";
    static constexpr int kSlabs = 1;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 8;
    static constexpr int kMaxRows = 64;
    static constexpr Isa kTopIsa = Isa::TOY_TOP_ISA;
    static constexpr const char* kIsaCapEnv = "TOY_CPU_MAX_ISA";
    static constexpr const char* kIsaReportEnv = "TOY_CPU_REPORT_ISA";
    using Params = ToyParams;
    struct Row { const float* v; };
    struct Layer { int hidden; float scale; MoeBufferRows<ToyQuant> rows; };
    static std::array<uint64_t, 1> min_slot_bytes(const SglangCpuExpertsLayer& d, const Params&)
    { return {uint64_t(d.hidden) * 4}; }
    static int validate(const SglangCpuExpertsLayer& d, const Params* p)
    { return d.activation == 0 && p && std::isfinite(p->scale) ? 0 : 2; }
    static Layer make_layer(const SglangCpuExpertsLayer& d, const Params* p)
    { return {d.hidden, p->scale, MoeBufferRows<ToyQuant>::of(d)}; }
    static int check_slot(const Layer&, int) { return 0; }
    static Row decode(const uint8_t* const* base, const Layer&) { return {reinterpret_cast<const float*>(base[0])}; }
    static int dispatch(const Layer& l, const SglangCpuExpertsForward& c, const RouteTable& r, Isa isa)
    {
        last_isa = isa;
        last_routes.clear();
        for (int i = 0; i < r.count[0]; ++i) last_routes.push_back(r.route(0, i));
        if (park_here && hold.load()) {
            inside.store(true);
            while (hold.load()) std::this_thread::yield();
        }
        // The caller's copy: a worker naming last_cpus would reach its own thread's.
        std::vector<int>& cpus = last_cpus;
        cpus.assign(size_t(c.threads), -1);
        run_team(c.threads, [&](int worker, int workers) {
            cpus[size_t(worker)] = sched_getcpu();
            for (int h = worker; h < l.hidden; h += workers)
                for (int t = 0; t < c.rows; ++t) {
                    float s = 0;
                    for (int i = 0; i < r.count[t]; ++i)
                        s += r.route(t, i).weight * l.scale
                             * decode(l.rows.slot(r.route(t, i).slot).base.data(), l).v[h];
                    float& o = c.out[size_t(t) * l.hidden + h];
                    o = c.accumulate ? o + s : s;
                }
        });
        return 0;
    }
    // Per calling thread, so forwards running at once from two threads each keep their own.
    static inline thread_local Isa last_isa = Isa::Scalar;
    static inline thread_local std::vector<Route> last_routes;  // token 0's, from this thread's last dispatch
    static inline thread_local std::vector<int> last_cpus;      // the CPU each worker of this thread's last team ran on
    // Test-only gate: while `hold` is set, a dispatch on a thread that set park_here parks after setting `inside`,
    // inside its forward.
    static inline std::atomic<bool> hold{false};
    static inline std::atomic<bool> inside{false};
    static inline thread_local bool park_here = false;
};
}  // namespace
}  // namespace toy
