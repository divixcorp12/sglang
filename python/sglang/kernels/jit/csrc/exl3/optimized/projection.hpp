// One EXL3 projection's matrix as the optimized CPU kernel reads it: views over a slot's trellis and sign vectors.
#pragma once
#include "half.hpp"

namespace sglang::exl3_cpu {

struct Exl3Projection
{
    const uint16_t* trellis;
    const Fp16* suh;
    const Fp16* svh;
    const Fp16* bias;  // nullable; added after the output transform
    int k;
    int n;
    int bits;
    // Band-contiguous ("swizzled") trellis layout: tile (kt, nt) stored at group nt/8, then kt, then member nt%8, so
    // each 8-tile output band reads as one sequential k-stream.
    int swz = 0;
    // The whole matrix's k-tiles: a swizzled group's stride, which a k-block view (run_tiles) keeps.
    int group_tiles_k = 0;
};

}  // namespace sglang::exl3_cpu
