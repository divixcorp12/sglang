// Differential GGML/layout check; C++ only, no CUDA or Python runtime.
#include "dot_nvfp4.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(42);
    constexpr float fp4[]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    for (int k:{16,32,48,64,80,144,512}) for (int row:{0,31,32,127,128}) {
        const int padded=int(rounded(k,64));
        std::vector<uint8_t> w(size_t(row+1)*k/2), sf(rounded(row+1,128)*rounded(k/16,4));
        std::vector<block_nvfp4> blocks(padded/64);
        std::vector<block_q8_0> x(padded/32);
        for (auto& b:x) { b.d=ggml_compute_fp32_to_fp16(.025f); for (auto& q:b.qs) q=int(rng()%255)-127; }
        // Exhaust every finite signed scale with random nibble/activation pairs.
        for (int code=0;code<256;++code) {
            if ((code&127)==127) continue;
            for (int j=0;j<k/2;++j) w[size_t(row)*k/2+j]=rng()%256;
            for (int g=0;g<k/16;++g) sf[sf_index(row,g,k/16)]=uint8_t((code+g)%127 | (code&128));
            for (auto& b:blocks) b=block_nvfp4{};
            double gold=0, magnitude=0;
            for (int g=0;g<k/16;++g) {
                const uint8_t scale=sf[sf_index(row,g,k/16)];
                const uint8_t* q=w.data()+size_t(row)*k/2+g*8;
                auto& b=blocks[g/4]; b.d[g%4]=scale&127;
                for (int j=0;j<8;++j) {
                    uint8_t lo=(q[j/2]>>(4*(j%2)))&15;
                    uint8_t hi=(q[(j+8)/2]>>(4*((j+8)%2)))&15;
                    uint8_t sign=scale&128?8:0;
                    b.qs[(g%4)*8+j]=(lo^sign)|((hi^sign)<<4);
                }
                int exp=(scale>>3)&15, man=scale&7;
                double value=exp?std::ldexp(1+man/8.0,exp-7):std::ldexp(double(man),-9);
                if (scale&128) value=-value;
                for (int j=0;j<16;++j) {
                    int c=g*16+j, id=(q[j/2]>>(4*(j%2)))&15;
                    double term=fp4[id]*value*x[c/32].qs[c%32]*ggml_compute_fp16_to_fp32(x[c/32].d);
                    gold+=term; magnitude+=std::abs(term);
                }
            }
            // Match padded columns to zero weights, without reading beyond the GPU row.
            GpuRow view(w.data(),sf.data(),row,k);
            float actual=dot_gpu(padded,view,x.data()), upstream;
            ggml_vec_dot_nvfp4_q8_0(padded,&upstream,0,blocks.data(),0,x.data(),0,1);
            const double tolerance=1e-5+1e-6*magnitude;
            assert(std::isfinite(actual) && std::abs(actual-gold)<=tolerance);
            assert(std::isfinite(upstream) && std::abs(upstream-gold)<=tolerance);
            assert(std::abs(actual-upstream)<=tolerance);
        }
    }
}
