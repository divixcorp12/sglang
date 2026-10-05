// Two toy quant libraries in one process, built as the expert-stream host module is (-fvisibility=hidden), each with
// its own accessor (TOY_KERNEL): the CpuExpertKernel interface crosses the .so boundary. Built by
// test_cpu_experts_common.py.
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace toy {
const sglang::cpu_experts::CpuExpertKernel& toy_kernel_a();
const sglang::cpu_experts::CpuExpertKernel& toy_kernel_b();
}  // namespace toy

int main()
{
    using namespace sglang::cpu_experts;
    const CpuExpertKernel& a = toy::toy_kernel_a();
    const CpuExpertKernel& b = toy::toy_kernel_b();
    CHECK(&a != &b && std::strcmp(a.name(), "toy_a") == 0 && std::strcmp(b.name(), "toy_b") == 0);

    constexpr int hidden = 16, capacity = 2;
    std::vector<float> slab(hidden * capacity);
    for (int i = 0; i < hidden * capacity; ++i) slab[i] = float(i);
    const float scale = 1.0f;
    LayerSlabs d;
    d.capacity = capacity;
    d.hidden = hidden;
    d.intermediate = hidden;
    d.slab_count = 1;
    d.slabs[0] = slab.data();
    d.slot_bytes[0] = hidden * 4;
    const auto params = std::as_bytes(std::span<const float>(&scale, 1));
    std::unique_ptr<CpuExpertLayer> layer = a.make_layer(d, params);
    CHECK(&layer->kernel() == &a);

    std::vector<uint16_t> x(hidden);
    const int32_t slot = 1;
    const float weight = 1.0f;
    std::vector<float> out(hidden, 7.0f);
    ForwardCall c;
    c.rows = 1;
    c.k = 1;
    c.threads = 1;
    c.x = x.data();
    c.slots = &slot;
    c.weights = &weight;
    c.out = out.data();
    a.forward(*layer, c);
    for (int h = 0; h < hidden; ++h) CHECK(out[h] == float(hidden + h));

    // Library b refuses library a's layer: its std::invalid_argument reaches this executable's catch.
    std::fill(out.begin(), out.end(), 7.0f);
    bool refused = false;
    try {
        b.forward(*layer, c);
    } catch (const std::invalid_argument& e) {
        // b's refusal names both kernels: itself ("toy_b CPU experts: ...") and the layer's ("kernel toy_a").
        refused = std::strstr(e.what(), "toy_b") != nullptr && std::strstr(e.what(), "kernel toy_a") != nullptr;
    }
    CHECK(refused);
    for (float v : out) CHECK(v == 7.0f);
    refused = false;
    try {
        b.make_layer(d, {});
    } catch (const std::exception&) {
        refused = true;
    }
    CHECK(refused);
    layer.reset();  // the deleting destructor runs in library a
    std::printf("ok cross_library\n");
    return 0;
}
