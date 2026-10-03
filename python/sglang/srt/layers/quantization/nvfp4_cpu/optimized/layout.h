#pragma once
#include <cstddef>
inline size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
// Inverse address of utils.swizzle_blockscale's reshape/permute.
inline size_t sf_index(int row, int group, int groups) {
    const size_t tiles_k = rounded(groups, 4) / 4;
    return (((size_t(row / 128) * tiles_k + group / 4) * 32 + row % 32) * 4
            + (row % 128) / 32) * 4 + group % 4;
}
