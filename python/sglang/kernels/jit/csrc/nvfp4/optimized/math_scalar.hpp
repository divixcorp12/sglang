// dot_rows<Isa::Scalar, M>: the portable dot product, GGML's scalar loop.
// Derived from GGML x86 quants.c, llama.cpp 889edf43ddae0cfe9a4564a882764dc879759870.
// MIT, ggml authors; see ../upstream/LICENSE.llama.cpp.
// Changes: GPU row/scale access, adjacent-nibble interleave, return signature, several Q8_0 vectors per weight row.
#pragma once
#include "math.hpp"

namespace sglang::nvfp4_cpu {
namespace {

template <>
struct DotRows<Isa::Scalar>
{
    template <int M>
    static void rows(int n, const GpuRow& x, const block_q8_0* const* ys, float* out)
    {
        assert(n % QK_NVFP4 == 0);
        const int nb = n / QK_NVFP4;
        float sumf[M] = {};
        for (int ib = 0; ib < nb; ++ib) {
            for (int s_idx = 0; s_idx < 4; ++s_idx) {
                const float d = x.scale(ib, s_idx);
                const int q8_block = s_idx / 2;
                const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
                for (int t = 0; t < M; ++t) {
                    const block_q8_0* y = ys[t];
                    const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

                    int sumi_lo = 0, sumi_hi = 0;
                    for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                        const uint8_t qv = x.bytes(ib)[s_idx*(QK_NVFP4_SUB/2) + j];
                        sumi_lo += y[2*ib + q8_block].qs[q8_off + 2*j] * kvalues_fp4[qv & 0xf];
                        sumi_hi += y[2*ib + q8_block].qs[q8_off + 2*j + 1] * kvalues_fp4[qv >>  4];
                    }

                    sumf[t] += dy * d * (sumi_lo + sumi_hi);
                }
            }
        }
        for (int t = 0; t < M; ++t) out[t] = sumf[t];
    }
};

}  // namespace
}  // namespace sglang::nvfp4_cpu
