// A call's live routes per token: the -1 slots and zero weights of the call's [rows][k] slots/weights dropped, the
// routing order kept. A quant that groups its work per token builds it in its dispatch (NVFP4 does; EXL3 groups by
// expert itself, reading the call directly, and so runs zero-weight routes).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

struct Route
{
    int32_t slot;
    float weight;
};

struct RouteTable
{
    const int32_t* count;  // [rows]: live routes of token t, at most k
    const Route* routes;   // [rows][k]: token t's live routes first
    int rows;
    int k;

    const Route& route(int t, int i) const { return routes[size_t(t) * k + i]; }

    // The table points into this thread's arena: valid until the next build on the same thread.
    static RouteTable build(const int32_t* slots, const float* weights, int rows, int k)
    {
        struct Arena
        {
            std::vector<Route> routes;
            std::vector<int32_t> count;
        };
        static thread_local Arena arena;
        if (arena.routes.size() < size_t(rows) * k) arena.routes.resize(size_t(rows) * k);
        if (arena.count.size() < size_t(rows)) arena.count.resize(size_t(rows));
        for (int t = 0; t < rows; ++t) {
            int live = 0;
            for (int j = 0; j < k; ++j) {
                const int32_t slot = slots[size_t(t) * k + j];
                const float weight = weights[size_t(t) * k + j];
                if (slot == -1 || weight == 0) continue;
                arena.routes[size_t(t) * k + live++] = {slot, weight};
            }
            arena.count[t] = live;
        }
        return {arena.count.data(), arena.routes.data(), rows, k};
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
