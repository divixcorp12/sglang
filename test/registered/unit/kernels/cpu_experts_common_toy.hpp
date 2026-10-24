// A toy CPU expert quant for the CPU experts framework's tests: one slab of float[hidden] per slot; a job writes
// out[t][h] (+)= sum over routes of weight * scale * slab[slot][h]. TOY_TOP_ISA picks kTopIsa (default Avx2).
#pragma once
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/expert_forward.hpp"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/routes.hpp"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/single_job.hpp"
#include <sched.h>
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
using ToyLayer = ExpertLayer<ToyParams>;
struct ToyQuant {
    static constexpr const char* kName = TOY_NAME;
    static constexpr int kSlabs = 1;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 8;
    static constexpr int kMaxRows = 64;
    static constexpr Isa kTopIsa = Isa::TOY_TOP_ISA;
    using Params = ToyParams;
    static std::array<uint64_t, 1> row_bytes(const ToyLayer& l, const Params&) { return {uint64_t(l.hidden) * 4}; }
    static const char* validate(const ToyLayer& l, Params& p)
    { return l.activation == 0 && std::isfinite(p.scale) ? nullptr : "the toy takes activation 0 and a finite scale"; }
    static bool usable(const ToyLayer&, const Params&, int) { return true; }
    // The toy's typed view of a slot: its row of floats.
    static const float* expert(const ExpertRow& r) { return reinterpret_cast<const float*>(r.slab[0]); }
    // Per admitting thread, so jobs admitted at once from two threads each keep their own.
    static inline thread_local Isa last_isa = Isa::Scalar;
    static inline thread_local std::vector<Route> last_routes;  // token 0's, from this thread's last prepare
    static inline thread_local std::vector<int> last_cpus;      // the CPU each worker of this thread's last job ran on
    // Test-only gate: while `hold` is set, a prepare on a thread that set park_here parks after setting `inside`.
    static inline std::atomic<bool> hold{false};
    static inline std::atomic<bool> inside{false};
    static inline thread_local bool park_here = false;
};
// The toy's kernel type: ExpertForward's static functions (binding.hpp's list) and a job interface of one phase of one
// task per worker. Each task waits for every worker to have taken one, so each worker runs exactly one and the CPU it
// ran on is recorded under its own index.
struct Toy : ExpertForward<ToyQuant> {
    struct Job {
        const ToyLayer* layer = nullptr;
        ForwardCall call{};
        RouteStore store;
        RouteTable table{};
        std::vector<int>* cpus = nullptr;
        std::atomic<int> arrived{0};
    };
    static constexpr int kPhases = 1;
    static constexpr int kGatePhase = -1;
    static void prepare(Job& job, const ToyLayer& l, const ForwardCall& c)
    {
        job.layer = &l;
        job.call = c;
        job.table = RouteTable::build(c.slots, c.weights, c.rows, c.k, job.store);
        ToyQuant::last_isa = isa();
        ToyQuant::last_routes.clear();
        for (int i = 0; i < job.table.count[0]; ++i) ToyQuant::last_routes.push_back(job.table.route(0, i));
        if (ToyQuant::park_here && ToyQuant::hold.load()) {
            ToyQuant::inside.store(true);
            while (ToyQuant::hold.load()) std::this_thread::yield();
        }
        ToyQuant::last_cpus.assign(size_t(c.team_workers), -1);
        job.cpus = &ToyQuant::last_cpus;
        job.arrived.store(0, std::memory_order_relaxed);
    }
    static Tasks tasks(const Job& job, int) { return {job.call.team_workers, 1}; }
    static void run(Job& job, int, int t0, int t1, int worker)
    {
        const int n = job.call.team_workers;
        const ToyLayer& l = *job.layer;
        const ForwardCall& c = job.call;
        for (int task = t0; task < t1; ++task) {
            (*job.cpus)[size_t(worker)] = sched_getcpu();
            job.arrived.fetch_add(1);
            while (job.arrived.load() < n) std::this_thread::yield();
            for (int h = task; h < l.hidden; h += n)
                for (int t = 0; t < c.rows; ++t) {
                    float s = 0;
                    for (int i = 0; i < job.table.count[t]; ++i)
                        s += job.table.route(t, i).weight * l.params.scale * ToyQuant::expert(l[job.table.route(t, i).slot])[h];
                    float& o = c.out[size_t(t) * l.hidden + h];
                    o = c.accumulate() ? o + s : s;
                }
        }
    }
    static int status(const Job&) { return 0; }
};
}  // namespace
}  // namespace toy
