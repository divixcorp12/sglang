// Standalone ASan/UBSan harness: no Python runtime or CUDA dependencies.
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstdio>

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
    SglangNvfp4CpuLayer d{};
    d.abi_version = 1; d.capacity = capacity; d.hidden = h; d.intermediate = n;
    d.inv_input_scale13 = .5f; d.inv_input_scale2 = .25f;
    d.slabs[0] = w13.data(); d.slabs[1] = w2.data();
    d.slabs[2] = sf13.data(); d.slabs[3] = sf2.data();
    d.slabs[4] = alpha.data(); d.slabs[5] = alpha.data();
    d.slot_bytes[0] = n*h; d.slot_bytes[1] = h*n/2;
    d.slot_bytes[2] = 256*8; d.slot_bytes[3] = 128*8;
    d.slot_bytes[4] = 4; d.slot_bytes[5] = 4;
    int64_t layer = -1;
    assert(sglang_nvfp4_cpu_experts_register_slabs(&d, &layer) == 0);
    const int32_t slots[] = {1,-1,0}; const float weights[] = {.5f,1,.25f};
    const double g0=h*.5*quantized_constant(1);
    const double mid0=float(float(g0/(1+std::exp(-g0)))*float(g0));
    const double expected0=quantized_constant(mid0)*n*.25*.75;
    for (int trial = 0; trial < 100; ++trial) {
        const int threads = trial == 0 ? 3 : trial % 3 + 1;
        assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), threads, 0) == 0);
        for (float v : out) assert(std::abs(v - expected0) < .01f);
        assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), threads, 1) == 0);
        for (float v : out) assert(std::abs(v - 2*expected0) < .01f);
    }
    assert(sglang_nvfp4_cpu_experts_free_layer(layer) == 0);
    assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), 3, 0) == 2);

    d.act_limit=8; // Keep Q8 intermediate deltas representable at large scales.
    // Every finite E4M3 encoding, including signed zeros, subnormals and
    // max-normal scales. Compare with an independent double scalar oracle.
    assert(sglang_nvfp4_cpu_experts_register_slabs(&d, &layer) == 0);
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
        assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), 3, 0) == 0);
        for (float v : out) { if (!(std::isfinite(v) && std::abs(v - expected) <= 1e-4 + std::abs(expected) * 1e-4)) std::fprintf(stderr,"code=%d actual=%.9g expected=%.9g\n",code,v,expected); assert(std::isfinite(v) && std::abs(v - expected) <= 1e-4 + std::abs(expected) * 1e-4); }
    }
    std::fill(sf13.begin(), sf13.end(), 56);
    const int32_t skipped = -1;
    assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), &skipped, weights, 1, out.data(), 3, 0) == 0);
    for (float v : out) assert(v == 0);
    assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), nullptr, nullptr, 0, out.data(), 3, 1) == 0);
    const int32_t invalid = capacity;
    std::fill(out.begin(), out.end(), 123);
    assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), &invalid, weights, 1, out.data(), 3, 0) == 2);
    for (float v : out) assert(v == 123);
    assert(sglang_nvfp4_cpu_experts_free_layer(layer) == 0);
    d.act_limit=0;
    std::fill(sf13.begin(),sf13.end(),126);
    std::fill(out.begin(),out.end(),123);
    assert(sglang_nvfp4_cpu_experts_register_slabs(&d,&layer)==0);
    // An overflowing Q8 FP16 delta rejects the job without publishing output.
    assert(sglang_nvfp4_cpu_experts_forward(layer,x.data(),slots,weights,3,out.data(),3,0)==2);
    for (float v:out) assert(v==123);
    x[0]=0x7e00;
    assert(sglang_nvfp4_cpu_experts_forward(layer,x.data(),slots,weights,3,out.data(),3,0)==2);
    for (float v:out) assert(v==123);
    assert(sglang_nvfp4_cpu_experts_free_layer(layer)==0);
    d.abi_version = 2;
    assert(sglang_nvfp4_cpu_experts_register_slabs(&d, &layer) == 2);
}
