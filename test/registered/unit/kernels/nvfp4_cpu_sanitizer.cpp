// Standalone ASan/UBSan harness: no Python runtime or CUDA dependencies.
#include "kernel.h"
#include "quant.hpp"
#include "../upstream/kernels.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>

using Layer = ::sglang::cpu_experts::CpuExpertLayer;
const ::sglang::cpu_experts::CpuExpertKernel& kernel = ::sglang::nvfp4_cpu::nvfp4_cpu_kernel();

// The C ABI's status for what a kernel call threw: 0 none, 2 std::invalid_argument, 1 any other exception.
template <class F>
int status_of(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::invalid_argument&) {
        return 2;
    } catch (const std::exception&) {
        return 1;
    }
}

// One kernel forward over `rows` token rows (x [rows][hidden], slots/weights [rows][k], out [rows][hidden]).
int forward(const Layer& layer, const void* x, const int32_t* slots, const float* weights, int32_t k, float* out,
            int32_t threads, int32_t accumulate, int32_t rows = 1) {
    ::sglang::cpu_experts::ForwardCall call;
    call.rows = rows; call.x = x; call.slots = slots; call.weights = weights;
    call.out = out; call.k = k; call.threads = threads; call.accumulate = accumulate != 0;
    return status_of([&] {
        kernel.check(layer, call);
        kernel.forward(layer, call);
    });
}

// A batched call's rows are bitwise the one-row calls': distinct inputs, a token with no live route, a slot named twice by
// one token, and five tokens on slot 0 (more than one chunk holds).
void check_batch(const Layer& layer, int h, int threads) {
    constexpr int rows = 7, k = 3;
    std::vector<unsigned short> x(size_t(rows) * h);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ggml_compute_fp32_to_fp16(float(int(i * 37 % 61) - 30) / 64.f);
    const int32_t slots[rows * k] = {0, 1, -1,  0, 0, 1,  1, -1, -1,  0, -1, 1,  -1, -1, -1,  0, 1, 0,  1, 0, -1};
    const float weights[rows * k] = {.5f, .25f, 1,  .125f, .75f, -.5f,  1, 1, 1,  .3f, 1, .6f,  1, 1, 1,
                                     .2f, .4f, .1f,  -.7f, .9f, 1};
    for (int accumulate : {0, 1}) {
        std::vector<float> batched(size_t(rows) * h), single(size_t(rows) * h);
        for (size_t i = 0; i < batched.size(); ++i) batched[i] = single[i] = accumulate ? float(i % 13) - 6 : 123;
        assert(forward(layer, x.data(), slots, weights, k, batched.data(), threads, accumulate, rows) == 0);
        for (int t = 0; t < rows; ++t)
            assert(forward(layer, x.data() + size_t(t) * h, slots + t * k, weights + t * k, k,
                           single.data() + size_t(t) * h, threads, accumulate) == 0);
        assert(std::memcmp(batched.data(), single.data(), batched.size() * sizeof(float)) == 0);
    }
}

double quantized_constant(double value) {
    float d=std::abs(float(value))/127.f;
    double delta=ggml_compute_fp16_to_fp32(ggml_compute_fp32_to_fp16(d));
    if (!delta) return 0;
    return std::round(float(value)*(d?1.f/d:0.f))*delta;
}
int main() {
    constexpr int h = 80, n = 80, capacity = 2;
    std::vector<unsigned char> w13(capacity * n*h, 0x22), w2(capacity*h*n/2, 0x22);
    std::vector<unsigned char> sf13(capacity*256*8, 56), sf2(capacity*128*8, 56);
    std::vector<float> alpha(capacity, 1), out(h, 123);
    std::vector<unsigned short> x(h, 0x3c00);
    SglangNvfp4CpuParams params{};
    params.inv_input_scale13 = .5f; params.inv_input_scale2 = .25f;
    const auto params_bytes = std::as_bytes(std::span<const SglangNvfp4CpuParams>(&params, 1));
    ::sglang::cpu_experts::LayerSlabs d;
    d.capacity = capacity; d.hidden = h; d.intermediate = n;
    d.slab_count = 7;
    d.slabs[0] = w13.data(); d.slabs[1] = w2.data();
    d.slabs[2] = sf13.data(); d.slabs[3] = sf2.data();
    d.slabs[4] = alpha.data(); d.slabs[5] = alpha.data();
    d.slot_bytes[0] = n*h; d.slot_bytes[1] = h*n/2;
    d.slot_bytes[2] = 256*8; d.slot_bytes[3] = 128*8;
    d.slot_bytes[4] = 4; d.slot_bytes[5] = 4;
    std::unique_ptr<Layer> owned;
    assert(status_of([&] { owned = kernel.make_layer(d, params_bytes); }) == 0);
    const Layer* layer_ptr = owned.get();
    const int32_t slots[] = {1,-1,0}; const float weights[] = {.5f,1,.25f};
    const double g0=h*.5*quantized_constant(1);
    const double mid0=float(float(g0/(1+std::exp(-g0)))*float(g0));
    const double expected0=quantized_constant(mid0)*n*.25*.75;
    for (int trial = 0; trial < 100; ++trial) {
        const int threads = trial == 0 ? 3 : trial % 3 + 1;
        assert(forward(*layer_ptr, x.data(), slots, weights, 3, out.data(), threads, 0) == 0);
        for (float v : out) assert(std::abs(v - expected0) < .01f);
        assert(forward(*layer_ptr, x.data(), slots, weights, 3, out.data(), threads, 1) == 0);
        for (float v : out) assert(std::abs(v - 2*expected0) < .01f);
    }
    for (int threads : {1, 3}) check_batch(*layer_ptr, h, threads);
    assert(forward(*layer_ptr, x.data(), slots, weights, 3, out.data(), 3, 0, 0) == 2);  // rows 0
    owned.reset();

    d.act_limit=8; // Keep Q8 intermediate deltas representable at large scales.
    // Every finite E4M3 encoding, including signed zeros, subnormals and
    // max-normal scales. Compare with an independent double scalar oracle.
    assert(status_of([&] { owned = kernel.make_layer(d, params_bytes); }) == 0);
    layer_ptr = owned.get();
    for (int code = 0; code < 256; ++code) {
        if ((code & 127) == 127) continue;
        std::fill(sf13.begin(), sf13.end(), static_cast<unsigned char>(code));
        const int exponent = (code >> 3) & 15, fraction = code & 7;
        double scale = exponent ? std::ldexp(1.0 + fraction / 8.0, exponent - 7)
                                : std::ldexp(double(fraction), -9);
        if (code & 128) scale = -scale;
        const double g = std::min(h * .5 * scale * quantized_constant(1),8.0);
        const double u = std::clamp(h * .5 * scale * quantized_constant(1),-8.0,8.0);
        const double sigmoid = g >= 0 ? 1.0 / (1.0 + std::exp(-g)) : std::exp(g) / (1.0 + std::exp(g));
        const double expected = quantized_constant(float(float(g*sigmoid)*float(u))) * n * .25 * .75;
        assert(forward(*layer_ptr, x.data(), slots, weights, 3, out.data(), 3, 0) == 0);
        for (float v : out) { if (!(std::isfinite(v) && std::abs(v - expected) <= 1e-4 + std::abs(expected) * 1e-4)) std::fprintf(stderr,"code=%d actual=%.9g expected=%.9g\n",code,v,expected); assert(std::isfinite(v) && std::abs(v - expected) <= 1e-4 + std::abs(expected) * 1e-4); }
    }
    std::fill(sf13.begin(), sf13.end(), 56);
    const int32_t skipped = -1;
    assert(forward(*layer_ptr, x.data(), &skipped, weights, 1, out.data(), 3, 0) == 0);
    for (float v : out) assert(v == 0);
    assert(forward(*layer_ptr, x.data(), nullptr, nullptr, 0, out.data(), 3, 1) == 0);
    const int32_t invalid = capacity;
    std::fill(out.begin(), out.end(), 123);
    assert(forward(*layer_ptr, x.data(), &invalid, weights, 1, out.data(), 3, 0) == 2);
    for (float v : out) assert(v == 123);
    owned.reset();
    d.act_limit=0;
    std::fill(sf13.begin(),sf13.end(),126);
    std::fill(out.begin(),out.end(),123);
    assert(status_of([&] { owned = kernel.make_layer(d, params_bytes); }) == 0);
    layer_ptr = owned.get();
    // An overflowing Q8 FP16 delta rejects the job without publishing output.
    assert(forward(*layer_ptr,x.data(),slots,weights,3,out.data(),3,0)==2);
    for (float v:out) assert(v==123);
    x[0]=0x7e00;
    assert(forward(*layer_ptr,x.data(),slots,weights,3,out.data(),3,0)==2);
    for (float v:out) assert(v==123);
    owned.reset();
    assert(status_of([&] { kernel.make_layer(d, {}); }) == 2);  // no params
    d.slab_count = 6;
    assert(status_of([&] { kernel.make_layer(d, params_bytes); }) == 2);
    d.slab_count = 7;
    // Keep-warm returns at once when the word has already moved past `seen`.
    const uint32_t word = 1;
    assert(status_of([&] { kernel.keep_warm({}, 2, &word, 0, INT64_MAX); }) == 0);
    assert(status_of([&] { kernel.keep_warm({}, 0, &word, 0, INT64_MAX); }) == 2);
}
