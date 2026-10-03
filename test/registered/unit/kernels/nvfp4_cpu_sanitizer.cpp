// Standalone ASan/UBSan harness: no Python runtime or CUDA dependencies.
#include "cpu_experts_cabi.h"
#include <cassert>
#include <cmath>
#include <vector>

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
    for (int trial = 0; trial < 100; ++trial) {
        const int threads = trial == 0 ? 3 : trial % 3 + 1;
        assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), threads, 0) == 0);
        for (float v : out) assert(std::abs(v - 24000.f) < .01f);
        assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), threads, 1) == 0);
        for (float v : out) assert(std::abs(v - 48000.f) < .01f);
    }
    assert(sglang_nvfp4_cpu_experts_free_layer(layer) == 0);
    assert(sglang_nvfp4_cpu_experts_forward(layer, x.data(), slots, weights, 3, out.data(), 3, 0) == 2);
}
