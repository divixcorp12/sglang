// Output-tile ownership only: changing the owner never changes a tile's arithmetic.
#pragma once
#include <algorithm>
#include <cstdint>

#ifndef EXL3_MOE_CPU_ROW_WEIGHTED_ASSIGNMENT
#define EXL3_MOE_CPU_ROW_WEIGHTED_ASSIGNMENT 0
#endif

namespace sglang::exl3_cpu {
namespace {
constexpr int FLAT_MAX_GEMVS_PER_WORKER = 4;

// Original assignment. Many GEMVs retain whole-GEMV striding for locality.
template <int Unit = 8, typename Gemv>
inline void assign_gemvs(int worker, int num_workers, int total, int tiles_n, Gemv gemv)
{
    static_assert(Unit == 2 || Unit == 8);
    if (total > FLAT_MAX_GEMVS_PER_WORKER * num_workers) {
        for (int j = worker; j < total; j += num_workers) gemv(j, 0, tiles_n);
        return;
    }
    const int64_t groups = static_cast<int64_t>(total) * (tiles_n / Unit);
    const int f0 = static_cast<int>(groups * worker / num_workers) * Unit;
    const int f1 = static_cast<int>(groups * (worker + 1) / num_workers) * Unit;
    for (int j = f0 / tiles_n; j * tiles_n < f1; ++j)
        gemv(j, std::max(f0 - j * tiles_n, 0), std::min(f1 - j * tiles_n, tiles_n));
}

// A group costs rows(j) units. Round shared boundaries up to whole aligned
// groups; adjacent workers use the same boundary, so every tile has one owner.
template <int Unit = 8, typename Rows, typename Gemv>
inline void assign_row_weighted_gemvs(int worker, int num_workers, int total, int tiles_n, Rows rows, Gemv gemv)
{
    static_assert(Unit == 2 || Unit == 8);
    if (total == 0) return;
    int64_t sum_rows = 0;
    bool uniform = true;
    const int first_rows = rows(0);
    for (int j = 0; j < total; ++j) {
        sum_rows += rows(j);
        uniform &= rows(j) == first_rows;
    }
    // Keep the original mapping, including its large-batch locality strategy,
    // when weighting cannot distinguish any of the GEMVs.
    if (uniform) { assign_gemvs<Unit>(worker, num_workers, total, tiles_n, gemv); return; }
    const int groups = tiles_n / Unit;
    const int64_t cost = sum_rows * groups;
    const int64_t begin = cost * worker / num_workers;
    const int64_t end = cost * (worker + 1) / num_workers;
    int64_t prefix = 0;
    for (int j = 0; j < total; ++j) {
        const int m = rows(j);  // ForwardCtx chunks always hold at least one row.
        auto boundary = [&](int64_t target) {
            const int64_t local = std::clamp(target - prefix, int64_t(0), int64_t(groups) * m);
            return int((local + m - 1) / m) * Unit;
        };
        const int t0 = boundary(begin), t1 = boundary(end);
        if (t0 < t1) gemv(j, t0, t1);
        prefix += int64_t(groups) * m;
    }
}

template <int Unit = 8, typename Rows, typename Gemv>
inline void assign_plan_gemvs(int worker, int num_workers, int total, int tiles_n, Rows rows, Gemv gemv)
{
    if constexpr (EXL3_MOE_CPU_ROW_WEIGHTED_ASSIGNMENT != 0)
        assign_row_weighted_gemvs<Unit>(worker, num_workers, total, tiles_n, rows, gemv);
    else
        assign_gemvs<Unit>(worker, num_workers, total, tiles_n, gemv);
}
} // namespace
} // namespace sglang::exl3_cpu
