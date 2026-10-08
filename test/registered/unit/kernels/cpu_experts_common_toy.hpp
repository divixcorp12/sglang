// A toy CPU expert quant for the CPU experts framework's tests: one slab of float[hidden] per slot; a forward writes
// out[t][h] (+)= sum over routes of weight * scale * slab[slot][h]. TOY_TOP_ISA picks kTopIsa (default Avx2).
#pragma once
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/expert_forward.hpp"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/routes.hpp"
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#ifndef TOY_TOP_ISA
#define TOY_TOP_ISA Avx2
#endif

#ifndef TOY_NAME
#define TOY_NAME "toy"
#endif

namespace toy {
namespace {
using namespace sglang::cpu_experts;
struct ToyParams { float scale; };
struct ToyQuant {
    static constexpr const char* kName = TOY_NAME;
    static constexpr int kSlabs = 1;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 8;
    static constexpr int kMaxRows = 64;
    static constexpr Isa kTopIsa = Isa::TOY_TOP_ISA;
    static constexpr const char* kIsaCapEnv = "TOY_CPU_MAX_ISA";
    static constexpr const char* kIsaReportEnv = "TOY_CPU_REPORT_ISA";
    using Params = ToyParams;
    static std::array<uint64_t, 1> row_bytes(const ExpertLayer& l, const Params&) { return {uint64_t(l.hidden) * 4}; }
    static const char* validate(const ExpertLayer& l, Params& p)
    { return l.activation == 0 && std::isfinite(p.scale) ? nullptr : "the toy takes activation 0 and a finite scale"; }
    static bool usable(const ExpertLayer&, const Params&, int) { return true; }
    // The toy's typed view of a slot: its row of floats.
    static const float* expert(const ExpertRow& r) { return reinterpret_cast<const float*>(r.slab[0]); }
    static int dispatch(const ExpertLayer& l, const Params& p, const ForwardCall& c, Isa isa, Team& team)
    {
        const RouteTable r = RouteTable::build(c.slots, c.weights, c.rows, c.k);
        last_isa = isa;
        last_routes.clear();
        for (int i = 0; i < r.count[0]; ++i) last_routes.push_back(r.route(0, i));
        if (park_here && hold.load()) {
            inside.store(true);
            while (hold.load()) std::this_thread::yield();
        }
        // The caller's copy: a worker naming last_cpus would reach its own thread's.
        std::vector<int>& cpus = last_cpus;
        cpus.assign(size_t(team.workers()), -1);
        team.run([&](int worker, int workers) {
            cpus[size_t(worker)] = sched_getcpu();
            for (int h = worker; h < l.hidden; h += workers)
                for (int t = 0; t < c.rows; ++t) {
                    float s = 0;
                    for (int i = 0; i < r.count[t]; ++i)
                        s += r.route(t, i).weight * p.scale * expert(l[r.route(t, i).slot])[h];
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

#ifndef TOY_KERNEL
#define TOY_KERNEL toy_kernel
#endif
namespace toy {
// The toy kernel of this library or harness: each defines it (test_cpu_experts_common.py builds libraries with
// distinct names, so a harness links two side by side).
const sglang::cpu_experts::CpuExpertKernel& TOY_KERNEL();
}  // namespace toy
