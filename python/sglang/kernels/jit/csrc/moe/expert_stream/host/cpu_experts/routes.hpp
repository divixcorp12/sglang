// A call's live routes per token: the -1 slots and zero weights of the call's [rows][k] slots/weights dropped, the
// routing order kept. A quant that groups its work per token builds it (NVFP4 does, into its job's store at prepare; EXL3
// groups by expert itself, reading the call directly, and so runs zero-weight routes).
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

// The storage a RouteTable points into.
struct RouteStore
{
    std::vector<Route> routes;
    std::vector<int32_t> count;
};

struct RouteTable
{
    const int32_t* count;  // [rows]: live routes of token t, at most k
    const Route* routes;   // [rows][k]: token t's live routes first
    int rows;
    int k;

    const Route& route(int t, int i) const { return routes[size_t(t) * k + i]; }

    // The table points into `store`: valid until the next build into it.
    static RouteTable build(const int32_t* slots, const float* weights, int rows, int k, RouteStore& store)
    {
        if (store.routes.size() < size_t(rows) * k) store.routes.resize(size_t(rows) * k);
        if (store.count.size() < size_t(rows)) store.count.resize(size_t(rows));
        for (int t = 0; t < rows; ++t) {
            int live = 0;
            for (int j = 0; j < k; ++j) {
                const int32_t slot = slots[size_t(t) * k + j];
                const float weight = weights[size_t(t) * k + j];
                if (slot == -1 || weight == 0) continue;
                store.routes[size_t(t) * k + live++] = {slot, weight};
            }
            store.count[t] = live;
        }
        return {store.count.data(), store.routes.data(), rows, k};
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
