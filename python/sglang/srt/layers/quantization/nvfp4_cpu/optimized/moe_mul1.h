// moe_mul1.cpp's types and functions. Only moe_mul1.cpp includes this: everything is in its anonymous namespace, as
// are dot_nvfp4.h, experts.hpp and shapes.hpp, included here. A forward is ForwardPlan<Shape, Isa>::run
// (forward_plan.hpp) on one OpenMP team per call.
#pragma once
#if !defined(__linux__) || !defined(_OPENMP)
#error The NVFP4 CPU expert kernel requires Linux and OpenMP.
#endif
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__AVX2__) && !defined(NVFP4_CPU_FORCE_SCALAR)
#include <immintrin.h>
#endif

namespace {
constexpr size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
// Inverse address of utils.swizzle_blockscale's reshape/permute.
inline size_t sf_index(int row, int group, int groups) {
    const size_t tiles_k = rounded(groups, 4) / 4;
    return (((size_t(row / 128) * tiles_k + group / 4) * 32 + row % 32) * 4
            + (row % 128) / 32) * 4 + group % 4;
}

#include "dot_nvfp4.h"

// The dot product's tier is fixed when the library is compiled (dot_nvfp4.h's #if chain): AVX2 under -march=native
// on an AVX2 host, else the scalar loop.
enum class Isa { Scalar, Avx2 };
#if defined(__AVX2__)
constexpr Isa kBuildIsa = Isa::Avx2;
#else
constexpr Isa kBuildIsa = Isa::Scalar;
#endif

#include "experts.hpp"
#include "shapes.hpp"

constexpr int kMaxRoutes = 8;      // the C ABI's k limit
constexpr int kMaxRows = 1 << 16;  // the C ABI's rows limit; arbitrary, it keeps every scratch index in range
// Tokens one weight-row decode serves: dot_rows's accumulators, one AVX2 register each, sit beside the decoded row.
constexpr int kChunkRows = 4;

// -------------------------------------------------------------------------------------------
//   Layer registry
// -------------------------------------------------------------------------------------------

struct RegisteredLayer
{
    LayerInfo info;
    StridedExperts<GenericShape> strided;
};

// -------------------------------------------------------------------------------------------
//   Forward context (its scratch, ForwardArena, is forward_plan.hpp's)
// -------------------------------------------------------------------------------------------

// 64-byte-aligned storage, so each kRowUnit share of fp32 outputs owns whole cache lines.
template <class T>
struct CacheAligned
{
    using value_type = T;
    CacheAligned() = default;
    template <class U> CacheAligned(const CacheAligned<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t{64}); }
    template <class U> bool operator==(const CacheAligned<U>&) const { return true; }
};

// One live route of one token (no -1 slot, no zero weight). The C ABI sets slot and weight; the plan binds the rest.
struct Route
{
    int slot;
    float weight;
    int unit;          // the (token, slot) unit computing this route's expert
    float down_alpha;  // the slot's down alpha * inv_input_scale2 * weight
};

// A route found while grouping a call's routes by slot: token `token`'s route number `index`.
struct RouteRef
{
    int slot, token, index;
};

// Up to kChunkRows units of one slot, run together so each weight row is decoded once for all of them. A unit is one
// (token, slot) pair: one intermediate row and one down result per output, however many of the token's routes name
// the slot.
struct Chunk
{
    Projection gate, up, down;
    float gate_alpha, up_alpha;  // the projection's alpha * inv_input_scale13
    int unit0, units;            // units [unit0, unit0 + units)
};

struct ForwardCtx
{
    LayerInfo info;
    const uint8_t* x;  // fp16 [rows][hidden], possibly unaligned
    float* out;        // fp32 [rows][hidden]
    int rows;
    bool accumulate;
    // From the calling thread's ForwardArena; the plan binds unit, down_alpha and everything below.
    Route* route;            // [rows][kMaxRoutes], each token's live routes in routing order
    const int* route_count;  // [rows]
    const Chunk* chunk;      // [chunks]
    int chunks;
    const int* unit_token;   // [units], the token each unit reads
    int units;
    // Scratch from the calling thread's ForwardArena.
    float* xf;                  // [rows][rounded(hidden, 64)]
    block_q8_0* qx;             // [rows][rounded(hidden, 64) / 32]
    float* inter;               // [units][rounded(intermediate, 64)]
    block_q8_0* qi;             // [units][rounded(intermediate, 64) / 32]
    float* partial;             // [workers][partial_stride]: one down output of every unit
    size_t partial_stride;
    // Q8_0 cannot represent the input or an intermediate: the forward returns 2 and leaves out untouched.
    std::atomic<bool> invalid{false};
};

// -------------------------------------------------------------------------------------------
//   Functions (moe_mul1.cpp)
// -------------------------------------------------------------------------------------------

LayerInfo info_of(const SglangNvfp4CpuLayer& d);
StridedExperts<GenericShape> strided_of(const SglangNvfp4CpuLayer& d);
std::shared_ptr<const RegisteredLayer> lookup(int64_t h);

// Freezes the configured cores into g_compute_cores once, at the first forward. Steady-state calls acquire no mutex.
inline void freeze_compute_cores();
// Inside a parallel region: pins OpenMP worker `worker` to its compute core (none configured: no-op), setting
// pin_error if it cannot.
inline void pin_compute_worker(int worker, std::atomic<int>& pin_error);

// Row `row` of a packed E2M1 matrix of k columns (scales sf) against the m <= kChunkRows Q8_0 vectors xs[0..m), into
// out[0..m).
void dot_rows(const uint8_t* w, const uint8_t* sf, int row, int k, const block_q8_0* const* xs, int m, float* out);
// Whether Q8_0 represents the 32 values of a block: finite, with a delta within FP16 range.
bool q8_representable(const float* v);
// One Q8_0 block. A zero FP16 delta contributes zero.
void quantize_block(const float* v, block_q8_0& out);
// Gate/up output of the gated SiLU: optional pre-SiLU clamp (gate from above, up both ways), stable SiLU including
// large negative gates, times up.
inline float swiglu(float g, float u, float limit);

// Runs the call's plan: the MiMo V2.6 Pro plan when the layer is that model's routed expert on an AVX2 build, else the
// generic plan for this build's tier.
template <Isa I = kBuildIsa>
int run_plan(ForwardCtx& ctx, const RegisteredLayer& layer, int threads);
bool valid(const SglangNvfp4CpuLayer& d);
} // namespace
