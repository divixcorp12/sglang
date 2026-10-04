// -------------------------------------------------------------------------------------------
//   AVX-512 VNNI banded kernel
// -------------------------------------------------------------------------------------------

// Gather the two 32-bit word vectors covering row `row`'s 16-bit states (the permute stage of
// the extraction, split out so a row pair can share it -- see vnni_band_rows)
template <int bits, int row>
M1_TARGET_BW
inline void dword_gather(__m512i p0, __m512i p1, __m512i p2, __m512i p3, __m512i& a, __m512i& b)
{
    alignas(64) static constexpr auto i0d = make_row_indices<bits, row, false>();
    alignas(64) static constexpr auto i1d = make_row_indices<bits, row, true>();
    const __m512i i0 = _mm512_load_si512(i0d.data());
    const __m512i i1 = _mm512_load_si512(i1d.data());
    a = _mm512_permutex2var_epi32(p0, i0, p1);
    b = _mm512_permutex2var_epi32(p0, i1, p1);
    if constexpr (bits > 4)
    {
        // Up to 64 packed words: indices >= 32 select from the second register pair. vpermt2var
        // uses index bits [4:0], so the same index vectors address both pairs; constexpr masks
        // choose per lane
        constexpr __mmask16 hm0 = make_row_himask<bits, row, false>();
        constexpr __mmask16 hm1 = make_row_himask<bits, row, true>();
        if constexpr (hm0 != 0)
            a = _mm512_mask_blend_epi32(hm0, a, _mm512_permutex2var_epi32(p2, i0, p3));
        if constexpr (hm1 != 0)
            b = _mm512_mask_blend_epi32(hm1, b, _mm512_permutex2var_epi32(p2, i1, p3));
    }
}

// Per-lane shift counts for the funnel merge: cols 0-7 use s0, cols 8-15 use s1
template <int s0, int s1>
constexpr std::array<int32_t, 16> make_lane_shifts()
{
    std::array<int32_t, 16> v{};
    for (int i = 0; i < 16; ++i) v[i] = i < 8 ? s0 : s1;
    return v;
}

// Shift-merge codes for `row` out of its gathered word vectors; delta = bits extracts row+1
// from row's own gather (valid when word_pair_ok). Vector shifts by >= 32 are well-defined
// zero, so the s' == 0 case needs no special path. The two half-rows generally need different
// shifts: one per-lane variable funnel shift (vpsrlvd/vpsllvd against compile-time count
// vectors) merges both in 4 uops (GCC fuses the or+and into vpternlogd), where two immediate
// funnel shifts plus a lane blend took 8.
template <int bits, int row, int delta>
M1_TARGET_BW
inline __m512i dword_codes(__m512i a, __m512i b)
{
    constexpr int s0 = row_shift<bits, row>(0) - delta;
    constexpr int s1 = row_shift<bits, row>(8) - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds shift headroom");
    if constexpr (s0 == s1)
    {
        const __m512i c = _mm512_or_si512(_mm512_srli_epi32(b, s0), _mm512_slli_epi32(a, 32 - s0));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    }
    else
    {
        alignas(64) static constexpr auto sh = make_lane_shifts<s0, s1>();
        alignas(64) static constexpr auto shc = make_lane_shifts<32 - s0, 32 - s1>();
        const __m512i c = _mm512_or_si512(_mm512_srlv_epi32(b, _mm512_load_si512(sh.data())),
                                          _mm512_sllv_epi32(a, _mm512_load_si512(shc.data())));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    }
}

template <int bits, int row>
M1_TARGET_BW
inline __m512i extract_row(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    __m512i a, b;
    dword_gather<bits, row>(p0, p1, p2, p3, a, b);
    return dword_codes<bits, row, 0>(a, b);
}

template <int bits, int rows, int band, int R>
M1_TARGET_VNNI
inline void vnni_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (dword_pair_wins<bits>())
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            __m512i a, wb;
            dword_gather<bits, R>(p0, p1, p2, p3, a, wb);
            const __m512i code0 = dword_codes<bits, R, 0>(a, wb);
            __m512i code1;
            if constexpr (word_pair_ok<bits, R>())
            {
                code1 = dword_codes<bits, R, bits>(a, wb);
            }
            else
            {
                dword_gather<bits, R + 1>(p0, p1, p2, p3, a, wb);
                code1 = dword_codes<bits, R + 1, 0>(a, wb);
            }
            const __m512i prod0 = _mm512_mullo_epi32(code0, mult);
            const __m512i prod1 = _mm512_mullo_epi32(code1, mult);
            for (int i = 0; i < rows; ++i)
            {
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R + 1]));
            }
            vnni_band_rows<bits, rows, band, R + 2>(p0, p1, p2, p3, b, splat, k, acc);
        }
    }
    else
    {
        if constexpr (R < 16) {
            const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
            const __m512i prod = _mm512_mullo_epi32(code, _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT)));
            for (int i = 0; i < rows; ++i)
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
            vnni_band_rows<bits, rows, band, R + 1>(p0, p1, p2, p3, b, splat, k, acc);
        }
    }
}

template <int bits, int rows, int band>
M1_TARGET_VNNI
void vnni_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    // The remaining-word count is computed at the call sites: MSVC rejects reading even a
    // constexpr local inside a capture-less lambda (C3493), unlike GCC/clang
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    // Swizzled (band-contiguous) trellis layout: tile (kt, nt) lives at group nt/8, then kt,
    // then member nt%8, so a band's k-stream is (near-)sequential instead of packed_size-sized
    // runs strided by the full row. Requires band-aligned tile ranges (divisors of 8 within one
    // group; enforced by the band tables in *_tiles and the group-aligned splits in
    // forward_phase). Prefetch step differs accordingly.
    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * (tl_swz_tiles_k ? tl_swz_tiles_k : tiles_k) * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                // Whole-group band on the swizzled layout: the k-stream is sequential, one line
                // one step ahead is enough and the HW prefetcher follows the run (wider/farther
                // measured neutral on K4, 7960X)
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                // Strided stream: the native layout (K8, or VNNI-only CPUs where nothing is
                // swizzled) strides row_stride per step, and a partial-group band on the
                // swizzled layout (rows 3-4 at band 4) reads half a group then skips half; both
                // outrun the HW prefetcher, so touch every line of the tile row a few steps
                // ahead, as the AVX2 tier does (PR #331). +11% decode on K8 (7960X, four
                // order-alternated pairs, 60.9 -> 67.9 tok/s), prefill unchanged; bits == 6 keeps
                // the shorter distance that tier found necessary for its 96-byte rows
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            vnni_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_VNNI
void vnni_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // m = 1 supports band widths up to 16 (16 zmm accumulators). Measured on the 7960X: 16 is
    // not better than 8 for decode-shape jobs (medians 1.01 vs 0.98 ms, interleaved A/B) -- the
    // prefetcher already covers 8-tile bursts and the extra accumulators cost load-scheduling
    // registers -- so 8 is fixed (was a runtime switch via EXL3_MOE_CPU_BAND during that
    // investigation; no case remained for deviating from 8, so removed ahead of further
    // microoptimization work that wants less runtime branching in this path)
    constexpr int band_cap = 8;

    // Swizzled layout requires bands that are divisors of 8 (whole or partial groups); the
    // VBMI tier widens these (see vbmi_tiles) but the dword scheme's extra live temporaries
    // don't leave the register headroom for that here
    const int max_band = mat.swz ? (rows == 1 ? 8 : rows <= 3 ? 4 : 2)
                                 : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: vnni_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: vnni_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: vnni_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: vnni_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: vnni_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: vnni_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: vnni_band<bits, rows, 7>(mat, in, tout, n0); break;
            case 8: vnni_band<bits, rows, 8>(mat, in, tout, n0); break;
            default:
                if constexpr (rows == 1)
                {
                    switch (band)
                    {
                        case 9: vnni_band<bits, 1, 9>(mat, in, tout, n0); break;
                        case 10: vnni_band<bits, 1, 10>(mat, in, tout, n0); break;
                        case 11: vnni_band<bits, 1, 11>(mat, in, tout, n0); break;
                        case 12: vnni_band<bits, 1, 12>(mat, in, tout, n0); break;
                        case 13: vnni_band<bits, 1, 13>(mat, in, tout, n0); break;
                        case 14: vnni_band<bits, 1, 14>(mat, in, tout, n0); break;
                        case 15: vnni_band<bits, 1, 15>(mat, in, tout, n0); break;
                        default: vnni_band<bits, 1, 16>(mat, in, tout, n0); break;
                    }
                }
                break;
        }
        n0 += band;
    }
}

// -------------------------------------------------------------------------------------------
//   AVX-512 BW banded kernel
//
//   AVX-512F/BW/VL without VNNI (Skylake-SP/X), which otherwise fell through to the AVX2 tier.
//   The VNNI kernel's dword extraction and k-major band structure (pure AVX-512F) with the AVX2
//   tier's accumulate in place of vpdpbusd: vpmaddubsw of the product bytes against 0x01 pairs
//   gives (b0+b1), (b2+b3) as i16 lanes (<= 510, cannot saturate), then one vpmaddwd per token
//   row against splat_dup (x8 in both 16-bit slots). Bit-exact with the AVX2 and VNNI tiers.
//   Separate functions rather than a template flag on the VNNI kernel: a function carries one
//   target attribute, and compiling this accumulate under the VNNI target would permit
//   contracting vpmaddwd+vpaddd into vpdpwssd.
// -------------------------------------------------------------------------------------------

// Force-inlined: GCC otherwise outlines the 16-row chain behind a call on every tile step, and
// with every zmm register caller-saved the band loop then reloads all of its live constants
// (index tables, multiplier, shift vectors) after each call (+8-9% inlined, Skylake-SP)
template <int bits, int rows, int band, int R>
M1_TARGET_BW
M1_ALWAYS_INLINE void bw_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat_dup, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (dword_pair_wins<bits>())
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            const __m512i ones = _mm512_set1_epi32(0x01010101);
            __m512i a, wb;
            dword_gather<bits, R>(p0, p1, p2, p3, a, wb);
            const __m512i code0 = dword_codes<bits, R, 0>(a, wb);
            __m512i code1;
            if constexpr (word_pair_ok<bits, R>())
            {
                code1 = dword_codes<bits, R, bits>(a, wb);
            }
            else
            {
                dword_gather<bits, R + 1>(p0, p1, p2, p3, a, wb);
                code1 = dword_codes<bits, R + 1, 0>(a, wb);
            }
            const __m512i ps0 = _mm512_maddubs_epi16(_mm512_mullo_epi32(code0, mult), ones);
            const __m512i ps1 = _mm512_maddubs_epi16(_mm512_mullo_epi32(code1, mult), ones);
            for (int i = 0; i < rows; ++i)
            {
                const __m512i x0 = _mm512_set1_epi32(splat_dup[static_cast<size_t>(i) * k + R]);
                const __m512i x1 = _mm512_set1_epi32(splat_dup[static_cast<size_t>(i) * k + R + 1]);
                acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps0, x0));
                acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps1, x1));
            }
            bw_band_rows<bits, rows, band, R + 2>(p0, p1, p2, p3, b, splat_dup, k, acc);
        }
    }
    else
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            const __m512i ones = _mm512_set1_epi32(0x01010101);
            const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
            const __m512i ps = _mm512_maddubs_epi16(_mm512_mullo_epi32(code, mult), ones);
            for (int i = 0; i < rows; ++i)
                acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps,
                    _mm512_set1_epi32(splat_dup[static_cast<size_t>(i) * k + R])));
            bw_band_rows<bits, rows, band, R + 1>(p0, p1, p2, p3, b, splat_dup, k, acc);
        }
    }
}

// Three-bit batched BW kernel. Decode two adjacent input rows at once into 32
// interleaved 16-bit lanes. For a 16-bit state s and M = (M_hi << 16) + M_lo,
// the low/high product halves are mullo(s, M_lo) and mulhi_unsigned(s, M_lo)
// + mullo(s, M_hi), modulo 2^16. Summing their bytes produces the same mul1
// byte-sum as the dword kernel; vpmaddwd combines both rows without saturation.
// Three-bit tiles occupy 96 bytes, so both halfword gathers fit in p0/p1.
// Used below by the block-128 residual path: one token has two activation rows.
//
// Extract two adjacent rows into interleaved 16-bit lanes. Each lane needs the two
// halfwords enclosing its state in the format's (previous dword : current dword) window.
template <int row, bool high>
constexpr std::array<uint16_t, 32> bw3_word_indices()
{
    std::array<uint16_t, 32> idx{};
    constexpr auto inv = make_tc_perm_inv();
    for (int col = 0; col < 16; ++col)
        for (int r = 0; r < 2; ++r)
        {
            const int t = inv[(row + r) * 16 + col];
            const int b0 = t * 3 + 3 - 16 + 768;
            const int b1 = b0 + 16;
            const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
            const int half = shift / 16 + int(high);
            idx[col * 2 + r] = half < 2 ? (((b1 - 1) / 32) % 24) * 2 + half
                                                  : ((b0 / 32) % 24) * 2 + half - 2;
        }
    return idx;
}

template <int row, bool left>
constexpr std::array<uint16_t, 32> bw3_word_shifts()
{
    std::array<uint16_t, 32> shifts{};
    for (int col = 0; col < 16; ++col)
    {
        const int s0 = row_shift<3, row>(col) % 16;
        const int s1 = row_shift<3, row + 1>(col) % 16;
        shifts[col * 2] = left ? 16 - s0 : s0;
        shifts[col * 2 + 1] = left ? 16 - s1 : s1;
    }
    return shifts;
}

template <int rows, int band, int P>
M1_TARGET_BW
M1_ALWAYS_INLINE void bw3_band_rows(__m512i p0, __m512i p1, int b,
    const int32_t* splat_dup, int k, __m512i (&acc)[band][MAX_M], const char* future_weight)
{
    if constexpr (P < 8)
    {
        // Prefetch a future tile while decoding the current tile's register data.
        // P is compile-time: issue these once per tile, not once per row pair.
        if constexpr (P == 0)
        {
            _mm_prefetch(future_weight, _MM_HINT_T0);
            _mm_prefetch(future_weight + 64, _MM_HINT_T0);
        }
        constexpr int R = 2 * P;
        alignas(64) static constexpr auto il = bw3_word_indices<R, false>();
        alignas(64) static constexpr auto ih = bw3_word_indices<R, true>();
        alignas(64) static constexpr auto sr = bw3_word_shifts<R, false>();
        alignas(64) static constexpr auto sl = bw3_word_shifts<R, true>();
        const __m512i lo = _mm512_permutex2var_epi16(p0, _mm512_load_si512(il.data()), p1);
        const __m512i hi = _mm512_permutex2var_epi16(p0, _mm512_load_si512(ih.data()), p1);
        const __m512i state = _mm512_or_si512(
            _mm512_srlv_epi16(lo, _mm512_load_si512(sr.data())),
            _mm512_sllv_epi16(hi, _mm512_load_si512(sl.data())));
        const __m512i ml = _mm512_set1_epi16(static_cast<int16_t>(MUL1_MULT & 0xffff));
        const __m512i mh = _mm512_set1_epi16(static_cast<int16_t>(MUL1_MULT >> 16));
        const __m512i prod_lo = _mm512_mullo_epi16(state, ml);
        const __m512i prod_hi = _mm512_add_epi16(_mm512_mulhi_epu16(state, ml),
                                                _mm512_mullo_epi16(state, mh));
        const __m512i ones = _mm512_set1_epi8(1);
        // Sum each product's four unsigned bytes. The result is in [0, 1020],
        // safely representable as signed i16 for the adjacent-row dot product.
        const __m512i sum = _mm512_add_epi16(_mm512_maddubs_epi16(prod_lo, ones),
                                            _mm512_maddubs_epi16(prod_hi, ones));
        for (int i = 0; i < rows; ++i)
        {
            const size_t off = static_cast<size_t>(i) * k + R;
            // Each i32 holds two copies of a signed i16 activation. On x86,
            // the middle four bytes of two adjacent entries hold the needed pair.
            // memcpy permits the unaligned load without violating aliasing rules.
            uint32_t pair;
            std::memcpy(&pair, reinterpret_cast<const char*>(splat_dup + off) + 2, sizeof(pair));
            acc[b][i] = _mm512_add_epi32(acc[b][i],
                _mm512_madd_epi16(sum, _mm512_set1_epi32(static_cast<int32_t>(pair))));
        }
        bw3_band_rows<rows, band, P + 1>(p0, p1, b, splat_dup, k, acc, future_weight);
    }
}

template <int bits, int rows, int band>
M1_TARGET_BW
void bw_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    // Layout and prefetch handling as in vnni_band (see the comments there); the host hands
    // this tier the swizzled layout too (moe_cpu_host gates it on has_avx512_bw)
    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat_dup = in.splat_dup + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * (tl_swz_tiles_k ? tl_swz_tiles_k : tiles_k) * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                // GCC uses a different spelling; at most four cache lines per tile.
                #if defined(__GNUC__) && !defined(__clang__)
                #pragma GCC unroll 4
                #else
                #pragma unroll
                #endif
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            bw_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat_dup, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_BW
void bw_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // Same band widths as vnni_tiles (one more live temporary per band step, same budget)
    constexpr int band_cap = 8;
    const int max_band = mat.swz ? (rows == 1 ? 8 : rows <= 3 ? 4 : 2)
                                 : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: bw_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: bw_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: bw_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: bw_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: bw_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: bw_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: bw_band<bits, rows, 7>(mat, in, tout, n0); break;
            default: bw_band<bits, rows, 8>(mat, in, tout, n0); break;
        }
        n0 += band;
    }
}

// -------------------------------------------------------------------------------------------
//   AVX-512 VBMI banded kernel
//
//   Replaces the dword extraction's two 32-bit cross permutes + shift-merge with a single
//   byte-level permute (vpermb / vpermt2b): the 3 bytes covering each column's 16-bit state
//   are gathered directly into the dword lane, then one (even K) or two blended (odd K)
//   sub-byte right-shifts + mask produce the same codes bit-exactly.
//
//   Layout facts this relies on (from make_tc_perm; validated bit-exact against the dword
//   scheme for K1-8 x m1-4 in benchmarks/moe_mul1_bench):
//   - within a half-row (cols 0-7 / 8-15) the inverse-permutation index steps by 32 per
//     column, so bit offsets step by 32*bits and shift % 8 is uniform per half-row at every K
//   - the two half-rows differ by 4*bits bits: same shift % 8 for even K, +/-4 for odd K
//   - rows (2p, 2p+1) differ by exactly `bits` bits, so when shift % 8 >= bits for every
//     column of the even row, one gather serves both rows ("byte pairing": K1/K2/K4 all 8
//     pairs, K3/K6 rows 8-15 only, K5/K7/K8 never)
// -------------------------------------------------------------------------------------------

// For each column, byte indices (into the tile's 32*bits packed bytes) of the 3 bytes covering
// bits [shift, shift+16) of the (w0:w1) combined word window; 4th lane byte unused (index 0,
// never observed: shift%8 + 16 <= 23 keeps the value inside the low 3 bytes)
template <int bits, int row>
constexpr std::array<uint8_t, 64> make_row_byte_indices()
{
    std::array<uint8_t, 64> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col)
    {
        const int t = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        const int w0 = (b0 / 32) % words32;          // high (earlier) word
        const int w1 = ((b1 - 1) / 32) % words32;    // low (later) word
        const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
        const int fb = shift / 8;
        for (int byte = 0; byte < 3; ++byte)
        {
            const int mb = fb + byte;
            const int src = mb < 4 ? w1 * 4 + mb : w0 * 4 + (mb - 4);
            idx[col * 4 + byte] = static_cast<uint8_t>(src);
        }
        idx[col * 4 + 3] = 0;
    }
    return idx;
}

// For bits > 4 (tile spans 4 zmms): which gathered bytes come from the (p2,p3) pair.
// vpermt2b consumes idx bits [6:0], so raw indices >= 128 address the high pair directly.
template <int bits, int row>
constexpr uint64_t make_row_byte_himask()
{
    const auto idx = make_row_byte_indices<bits, row>();
    uint64_t m = 0;
    for (int i = 0; i < 64; ++i)
        if (idx[i] >= 128) m |= uint64_t(1) << i;
    return m;
}

// Byte-level pairing is valid iff the odd row's value stays inside the even row's gathered
// byte window for every column, i.e. shift % 8 >= bits everywhere (stricter than the dword
// path's word_pair_ok, which only needs the full shift's headroom)
template <int bits, int row>
constexpr bool byte_pair_ok()
{
    for (int col = 0; col < 16; ++col)
        if (row_shift<bits, row>(col) % 8 < bits) return false;
    return true;
}

template <int bits, int row>
M1_TARGET_VBMI
inline __m512i gather_row_bytes(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    alignas(64) static constexpr auto bidx = make_row_byte_indices<bits, row>();
    const __m512i idx = _mm512_load_si512(bidx.data());
    if constexpr (bits <= 2)
    {
        (void) p1; (void) p2; (void) p3;
        return _mm512_permutexvar_epi8(idx, p0);
    }
    else if constexpr (bits <= 4)
    {
        (void) p2; (void) p3;
        return _mm512_permutex2var_epi8(p0, idx, p1);
    }
    else
    {
        constexpr uint64_t hm = make_row_byte_himask<bits, row>();
        if constexpr (hm == 0)
            return _mm512_permutex2var_epi8(p0, idx, p1);
        else if constexpr (hm == ~uint64_t(0))
            return _mm512_permutex2var_epi8(p2, idx, p3);
        else
            return _mm512_mask_blend_epi8(static_cast<__mmask64>(hm),
                _mm512_permutex2var_epi8(p0, idx, p1),
                _mm512_permutex2var_epi8(p2, idx, p3));
    }
}

// delta = 0 extracts `row` itself; delta = bits extracts row+1 from row's gathered bytes
template <int bits, int row, int delta>
M1_TARGET_VBMI
inline __m512i shift_mask_row(__m512i g)
{
    constexpr int s0 = row_shift<bits, row>(0) % 8 - delta;
    constexpr int s1 = row_shift<bits, row>(8) % 8 - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds sub-byte shift headroom");
    if constexpr (s0 == s1)
        return _mm512_and_si512(_mm512_srli_epi32(g, s0), _mm512_set1_epi32(0xffff));
    else
        return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00,
            _mm512_srli_epi32(g, s0), _mm512_srli_epi32(g, s1)), _mm512_set1_epi32(0xffff));
}

template <int bits, int rows, int band, int P>
M1_TARGET_VBMI
inline void vbmi_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (P < 8)
    {
        constexpr int R = P * 2;
        const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
        __m512i c0, c1;
        if constexpr (byte_pair_ok<bits, R>())
        {
            const __m512i g = gather_row_bytes<bits, R>(p0, p1, p2, p3);
            c0 = shift_mask_row<bits, R, 0>(g);
            c1 = shift_mask_row<bits, R, bits>(g);
        }
        else
        {
            c0 = shift_mask_row<bits, R, 0>(gather_row_bytes<bits, R>(p0, p1, p2, p3));
            c1 = shift_mask_row<bits, R + 1, 0>(gather_row_bytes<bits, R + 1>(p0, p1, p2, p3));
        }
        const __m512i prod0 = _mm512_mullo_epi32(c0, mult);
        const __m512i prod1 = _mm512_mullo_epi32(c1, mult);
        for (int i = 0; i < rows; ++i)
        {
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R + 1]));
        }
        vbmi_band_rows<bits, rows, band, P + 1>(p0, p1, p2, p3, b, splat, k, acc);
    }
}

template <int bits, int rows, int band>
M1_TARGET_VBMI
void vbmi_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    // Same MSVC-safe form as vnni_band (C3493: no constexpr locals read inside the lambda)
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * (tl_swz_tiles_k ? tl_swz_tiles_k : tiles_k) * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                // Whole-group band on the swizzled layout: the k-stream is sequential, one line
                // one step ahead is enough and the HW prefetcher follows the run (wider/farther
                // measured neutral on K4, 7960X)
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                // Strided stream: the native layout (K8, or VNNI-only CPUs where nothing is
                // swizzled) strides row_stride per step, and a partial-group band on the
                // swizzled layout (rows 3-4 at band 4) reads half a group then skips half; both
                // outrun the HW prefetcher, so touch every line of the tile row a few steps
                // ahead, as the AVX2 tier does (PR #331). +11% decode on K8 (7960X, four
                // order-alternated pairs, 60.9 -> 67.9 tok/s), prefill unchanged; bits == 6 keeps
                // the shorter distance that tier found necessary for its 96-byte rows
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            vbmi_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_VBMI
void vbmi_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    constexpr int band_cap = 8;

    // Swizzled layout: bands must be divisors of 8 (whole or partial groups). Unlike the
    // dword scheme, byte-gather extraction needs few temporaries, so wider bands (up to 16
    // zmm accumulators: rows2 x band8, rows3/4 x band4) fit the register budget and keep the
    // swizzled stream at full duty. Narrow divisor bands (read-N-skip-N) measured BELOW
    // native layout at m>1. K2 rows4 prefers band 2 (measured 216 vs 197 Gw/s at band 4).
    const int max_band = mat.swz
        ? (rows <= 2 ? 8 : (rows == 4 && bits == 2 ? 2 : 4))
        : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: vbmi_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: vbmi_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: vbmi_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: vbmi_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: vbmi_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: vbmi_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: vbmi_band<bits, rows, 7>(mat, in, tout, n0); break;
            default: vbmi_band<bits, rows, 8>(mat, in, tout, n0); break;
        }
        n0 += band;
    }
}

// K3/BW block-128 residual decode. Keep each output band's fp32 sums across input
// blocks, avoiding repeated dispatch/setup and the full-size intermediate output.
// Walking the input blocks within a band also follows the swizzled weight layout.
// Preserve the generic path's rounding: scale each int32 block with its own FMA,
// sum blocks separately for base/residual rows, then add the residual exactly once.
template <int band>
M1_TARGET_BW
void bw3_blocked_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    constexpr int B = 128, rows = 2, packed_size = 48;
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const size_t step = static_cast<size_t>(mat.swz ? 8 : tiles_n) * packed_size;
    __m512 sums[band][rows];
    for (int kb = 0; kb < mat.k / B; ++kb)
    {
        __m512i acc[band][MAX_M];
        for (int b = 0; b < band; ++b)
            for (int i = 0; i < rows; ++i)
                acc[b][i] = _mm512_setzero_si512();
        for (int kt = 0; kt < B / 16; ++kt)
        {
            const int tile_k = kb * (B / 16) + kt;
            const int32_t* splat = in.splat_dup + static_cast<size_t>(kb) * rows * B + kt * 16;
            for (int b = 0; b < band; ++b)
            {
                const size_t tile = mat.swz
                    ? static_cast<size_t>(n0 / 8) * tiles_k * 8 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b
                    : static_cast<size_t>(tile_k) * tiles_n + n0 + b;
                const uint16_t* packed = mat.trellis + tile * packed_size;
                // Lead by two K tiles (step is in uint16_t elements). The prefetch
                // may point beyond the final tile; form its address
                // as an integer instead of an out-of-object C++ array pointer.
                const char* pf = reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(packed) + 4 * step);
                const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
                const __m512i p0 = _mm512_loadu_si512(pw);
                // The tile tail is exactly 32 bytes. A 256-bit load avoids a
                // masked 512-bit memory operation and keeps the upper lanes zero.
                const __m512i p1 = _mm512_zextsi256_si512(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pw + 16)));
                bw3_band_rows<rows, band, 0>(p0, p1, b, splat, B, acc, pf);
            }
        }
        for (int b = 0; b < band; ++b)
            for (int i = 0; i < rows; ++i)
            {
                const float scale = mul1_k_inv() * in.bq[kb * MAX_M + i];
                const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.bsum[kb * MAX_M + i]) * scale);
                const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
                if (kb == 0) sums[b][i] = v;
                else sums[b][i] = _mm512_add_ps(sums[b][i], v);
            }
    }
    for (int b = 0; b < band; ++b)
    {
        _mm512_storeu_ps(tout + (n0 + b) * 16, _mm512_add_ps(sums[b][0], sums[b][1]));
        _mm512_storeu_ps(tout + mat.n + (n0 + b) * 16, sums[b][1]);
    }
}

M1_TARGET_BW
void bw3_blocked_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // Match the eight output tiles stored together in the swizzled layout.
    // Keep each output's fp32 block accumulation order unchanged.
    constexpr int cap = 8;
    for (int n0 = tn0; n0 < tn1;)
    {
        const int band = std::min({cap, tn1 - n0, 8 - n0 % 8});
        switch (band)
        {
            case 1: bw3_blocked_band<1>(mat, in, tout, n0); break;
            case 2: bw3_blocked_band<2>(mat, in, tout, n0); break;
            case 3: bw3_blocked_band<3>(mat, in, tout, n0); break;
            case 4: bw3_blocked_band<4>(mat, in, tout, n0); break;
            case 5: bw3_blocked_band<5>(mat, in, tout, n0); break;
            case 6: bw3_blocked_band<6>(mat, in, tout, n0); break;
            case 7: bw3_blocked_band<7>(mat, in, tout, n0); break;
            case 8: bw3_blocked_band<8>(mat, in, tout, n0); break;
        }
        n0 += band;
    }
}


M1_TARGET_BW M1_ALWAYS_INLINE __m512i register_bytesum(__m512i state) {
    const __m512i ml = _mm512_set1_epi16(int16_t(MUL1_MULT & 0xffff));
    const __m512i mh = _mm512_set1_epi16(int16_t(MUL1_MULT >> 16));
    const __m512i lo = _mm512_mullo_epi16(state, ml);
    const __m512i hi = _mm512_add_epi16(_mm512_mulhi_epu16(state, ml), _mm512_mullo_epi16(state, mh));
    const __m512i ones = _mm512_set1_epi8(1);
    return _mm512_add_epi16(_mm512_maddubs_epi16(lo, ones), _mm512_maddubs_epi16(hi, ones));
}
// Read the original two 96-byte tiles. No persistent CPU layout or extra weights.
// Each column pair's 32 three-bit codes occupies three consecutive uint32 words.
// Transpose those triplets across sixteen SIMD lanes (eight per output tile).
// The preceding lane's final word supplies the circular trellis's initial state.
template<int Q>
constexpr std::array<int32_t,16> register_word_indices() {
    std::array<int32_t,16> indices{};
    for(int lane=0;lane<16;++lane) indices[lane]=lane*3+Q;
    return indices;
}

template<int Q>
M1_TARGET_BW M1_ALWAYS_INLINE __m512i register_word(__m512i p0,__m512i p1,__m512i p2) {
    alignas(64) static constexpr auto idx=register_word_indices<Q>();
    constexpr __mmask16 mask=Q==2?0xfc00:0xf800;
    const __m512i index=_mm512_load_si512(idx.data());
    const __m512i low=_mm512_permutex2var_epi32(p0,index,p1);
    // vpermd uses only the low four index bits: indices 32..47 address p2.
    return _mm512_mask_permutexvar_epi32(low,mask,index,p2);
}

template<int J,int Pairs,bool Compact=false>
M1_TARGET_BW M1_ALWAYS_INLINE void register_rows(__m512i prev,__m512i a,__m512i b,__m512i c,
    const std::conditional_t<Compact,int16_t,int32_t>* splat,__m512i (&acc)[Pairs][2][2],int band) {
    if constexpr(J<16) {
        constexpr int offset=19+6*J,q=offset/32,s=offset%32;
        __m512i lo,hi;
        if constexpr(q==0){lo=prev;hi=a;}
        else if constexpr(q==1){lo=a;hi=b;}
        else if constexpr(q==2){lo=b;hi=c;}
        else {lo=c;hi=_mm512_setzero_si512();}
        __m512i state;
    if constexpr(s<=13) {
        // Only v[31:13] survives the halfword selection. hi contributes to
        // v[s-1:0], so it is irrelevant in these seven row positions.
        state=_mm512_mask_blend_epi16(0xaaaaaaaaU,
            _mm512_srli_epi32(lo,16-s),_mm512_slli_epi32(lo,s+3));
    } else if constexpr(s==15) {
        state=_mm512_mask_blend_epi16(0xaaaaaaaaU,_mm512_srli_epi32(lo,1),
            _mm512_or_si512(_mm512_slli_epi32(lo,18),_mm512_srli_epi32(hi,14)));
    } else if constexpr(s==29) {
        state=_mm512_mask_blend_epi16(0xaaaaaaaaU,
            _mm512_or_si512(_mm512_slli_epi32(lo,13),_mm512_srli_epi32(hi,19)),hi);
    } else if constexpr(s==31) {
        state=_mm512_mask_blend_epi16(0xaaaaaaaaU,
            _mm512_or_si512(_mm512_slli_epi32(lo,15),_mm512_srli_epi32(hi,17)),_mm512_slli_epi32(hi,2));
    } else {
        const __m512i v=_mm512_or_si512(_mm512_slli_epi32(lo,s),_mm512_srli_epi32(hi,32-s));
        state=_mm512_mask_blend_epi16(0xaaaaaaaaU,_mm512_srli_epi32(v,16),_mm512_slli_epi32(v,3));
    }
        const __m512i sum=register_bytesum(state);
        constexpr int row=(J/4)*2+(J%2)*8,half=(J/2)%2;
        for(int i=0;i<2;++i) {
            uint32_t pair;std::memcpy(&pair,reinterpret_cast<const char*>(splat+i*128+row)+(Compact?0:2),4);
            acc[band][half][i]=_mm512_add_epi32(acc[band][half][i],_mm512_madd_epi16(sum,_mm512_set1_epi32(int32_t(pair))));
        }
        register_rows<J+1,Pairs,Compact>(prev,a,b,c,splat,acc,band);
    }
}

template<int Pairs,int FixedK=0,int FixedN=0,bool Compact=false>
M1_TARGET_BW void register_band(const MoeCpuMatrix& mat,const PreparedIn& in,float* tout,int n0) {
    const int k=FixedK?FixedK:mat.k,n=FixedN?FixedN:mat.n;
    const bool swz=FixedK?false:mat.swz;
    const int tiles_k=k/16,tiles_n=n/16;
    const size_t step=size_t(swz?8:tiles_n)*96;
    __m512 sums[Pairs][2][2];
    alignas(64) static constexpr int previous[16]={7,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14};
    for(int kb=0;kb<k/128;++kb) {
        __m512i acc[Pairs][2][2];
        for(int b=0;b<Pairs;++b)for(int h=0;h<2;++h)for(int i=0;i<2;++i)acc[b][h][i]=_mm512_setzero_si512();
        for(int kt=0;kt<8;++kt) {
            const auto* splat=[&]() {
                    if constexpr(Compact) return in.compact+size_t(kb)*256+kt*16;
                    else return in.splat_dup+size_t(kb)*256+kt*16;
                }();
            for(int band=0;band<Pairs;++band) {
                const int nt=n0+2*band,tile_k=kb*8+kt;
                const size_t tile=swz?size_t(nt/8)*tiles_k*8+size_t(tile_k)*8+nt%8:size_t(tile_k)*tiles_n+nt;
                const uint32_t* packed=reinterpret_cast<const uint32_t*>(mat.trellis+tile*48);
                const uintptr_t future=reinterpret_cast<uintptr_t>(packed)+step*2;
                #pragma GCC unroll 3
                for(int line=0;line<3;++line)_mm_prefetch(reinterpret_cast<const char*>(future+line*64),_MM_HINT_T0);
                const __m512i p0=_mm512_loadu_si512(packed),p1=_mm512_loadu_si512(packed+16),p2=_mm512_loadu_si512(packed+32);
                const __m512i a=register_word<0>(p0,p1,p2),b=register_word<1>(p0,p1,p2),c=register_word<2>(p0,p1,p2);
                const __m512i prev=_mm512_permutexvar_epi32(_mm512_load_si512(previous),c);
                register_rows<0,Pairs,Compact>(prev,a,b,c,splat,acc,band);
            }
        }
        __m512 scales[2],corrections[2];
        for(int i=0;i<2;++i) {
            const float scale=0x1.bb8p-8f*in.bq[kb*MAX_M+i];
            scales[i]=_mm512_set1_ps(scale);
            corrections[i]=_mm512_set1_ps(-510.0f*float(in.bsum[kb*MAX_M+i])*scale);
        }
        for(int b=0;b<Pairs;++b)for(int h=0;h<2;++h)for(int i=0;i<2;++i) {
            const __m512 v=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][h][i]),scales[i],corrections[i]);
            if(kb==0)sums[b][h][i]=v;else sums[b][h][i]=_mm512_add_ps(sums[b][h][i],v);
        }
    }
    for(int b=0;b<Pairs;++b) {
        const __m512 lo=_mm512_add_ps(sums[b][0][0],sums[b][0][1]);
        const __m512 hi=_mm512_add_ps(sums[b][1][0],sums[b][1][1]);
        _mm512_storeu_ps(tout+(n0+2*b)*16,_mm512_shuffle_f32x4(lo,hi,0x44));
        _mm512_storeu_ps(tout+(n0+2*b+1)*16,_mm512_shuffle_f32x4(lo,hi,0xee));
        _mm512_storeu_ps(tout+n+(n0+2*b)*16,_mm512_shuffle_f32x4(sums[b][0][1],sums[b][1][1],0x44));
        _mm512_storeu_ps(tout+n+(n0+2*b+1)*16,_mm512_shuffle_f32x4(sums[b][0][1],sums[b][1][1],0xee));
    }
}

// Visit neighboring 128-output bands before advancing to the next 128-input block.
// Original packed weights and each output's increasing-kb FP32 addition order stay
// unchanged. One integer band is live at a time; FP32 partials reside in private
// aligned scratch. The one-band variant controls for this factoring/scratch cost.

struct IntegerAccum { __m512i lanes[4][2][2]; };
M1_TARGET_BW inline void integer_acc_zero(IntegerAccum& a) {
    for(int b=0;b<4;++b)for(int h=0;h<2;++h)for(int i=0;i<2;++i)a.lanes[b][h][i]=_mm512_setzero_si512();
}

M1_TARGET_BW __attribute__((noinline))
void traversal_kblock(const MoeCpuMatrix& mat, const PreparedIn& in,
                      __m512 (&sums)[4][2][2], int n0, int kb) {
    const int tiles_k=mat.k/16, tiles_n=mat.n/16;
    const bool swz=mat.swz;
    const size_t step=size_t(swz?8:tiles_n)*96;
    alignas(64) static constexpr int previous[16]={7,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14};
    IntegerAccum first;integer_acc_zero(first);
    for(int kt=0;kt<8;++kt) {
        auto& acc=first.lanes;
        const int16_t* splat=in.compact+size_t(kb)*256+kt*16;
        #pragma omp unroll full
        for(int band=0;band<4;++band) {
            const int nt=n0+2*band, tile_k=kb*8+kt;
            const size_t tile=swz?size_t(nt/8)*tiles_k*8+size_t(tile_k)*8+nt%8
                                 :size_t(tile_k)*tiles_n+nt;
            const uint32_t* packed=reinterpret_cast<const uint32_t*>(mat.trellis+tile*48);
            const uintptr_t future=reinterpret_cast<uintptr_t>(packed)+step*2;
            #pragma GCC unroll 3
            for(int line=0;line<3;++line)
                _mm_prefetch(reinterpret_cast<const char*>(future+line*64),_MM_HINT_T0);
            const __m512i p0=_mm512_loadu_si512(packed);
            const __m512i p1=_mm512_loadu_si512(packed+16);
            const __m512i p2=_mm512_loadu_si512(packed+32);
            const __m512i a=register_word<0>(p0,p1,p2);
            const __m512i b=register_word<1>(p0,p1,p2);
            const __m512i c=register_word<2>(p0,p1,p2);
            const __m512i prev=_mm512_permutexvar_epi32(_mm512_load_si512(previous),c);
            register_rows<0,4,true>(prev,a,b,c,splat,acc,band);
        }
    }
    auto& acc=first.lanes;
    __m512 scales[2],corrections[2];
    for(int i=0;i<2;++i) {
        const float scale=0x1.bb8p-8f*in.bq[kb*MAX_M+i];
        scales[i]=_mm512_set1_ps(scale);
        corrections[i]=_mm512_set1_ps(-510.0f*float(in.bsum[kb*MAX_M+i])*scale);
    }
    for(int b=0;b<4;++b) for(int h=0;h<2;++h) for(int i=0;i<2;++i) {
        const __m512 v=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][h][i]),scales[i],corrections[i]);
        if(kb==0) sums[b][h][i]=v;
        else sums[b][h][i]=_mm512_add_ps(sums[b][h][i],v);
    }
}

template<int Groups>
M1_TARGET_BW void traversal_group(const MoeCpuMatrix& mat, const PreparedIn& in,
                                 float* tout, int n0) {
    static_assert(Groups>=1 && Groups<=4);
    // Each group has 16 ZMM partial sums: 1024 bytes per 128 outputs.
    alignas(64) __m512 sums[Groups][4][2][2];
    for(int kb=0;kb<mat.k/128;++kb)
        for(int group=0;group<Groups;++group)
            traversal_kblock(mat,in,sums[group],n0+group*8,kb);
    for(int group=0;group<Groups;++group) for(int b=0;b<4;++b) {
        const int tile=n0+group*8+2*b;
        const __m512 lo=_mm512_add_ps(sums[group][b][0][0],sums[group][b][0][1]);
        const __m512 hi=_mm512_add_ps(sums[group][b][1][0],sums[group][b][1][1]);
        _mm512_storeu_ps(tout+tile*16,_mm512_shuffle_f32x4(lo,hi,0x44));
        _mm512_storeu_ps(tout+(tile+1)*16,_mm512_shuffle_f32x4(lo,hi,0xee));
        _mm512_storeu_ps(tout+mat.n+tile*16,
            _mm512_shuffle_f32x4(sums[group][b][0][1],sums[group][b][1][1],0x44));
        _mm512_storeu_ps(tout+mat.n+(tile+1)*16,
            _mm512_shuffle_f32x4(sums[group][b][0][1],sums[group][b][1][1],0xee));
    }
}

M1_TARGET_BW void traversal_tiles(const MoeCpuMatrix& mat,const PreparedIn& in,
                                 float* tout,int t0,int t1,bool grouped) {
    for(int t=t0;t<t1;) {
        const int remaining=(t1-t)/8;
        // Range-aware: preserve three-band E1 ranges, 9=3+3+3 and 10=3+3+4.
        const int cap=grouped ? (remaining>4 ? 3 : 4) : 1;
        const int groups=std::min(cap,remaining);
        switch(groups) {
            case 4: traversal_group<4>(mat,in,tout,t);break;
            case 3: traversal_group<3>(mat,in,tout,t);break;
            case 2: traversal_group<2>(mat,in,tout,t);break;
            default: traversal_group<1>(mat,in,tout,t);break;
        }
        t+=groups*8;
    }
}

// Compact tile pairs [t0, t1) inside one 128-output group, t1 - t0 in {0, 2, 4, 6}: the same per-output arithmetic as
// traversal_kblock, which takes whole groups.
M1_TARGET_BW void compact_pairs(const MoeCpuMatrix& mat,const PreparedIn& in,float* tout,int t0,int t1) {
    switch((t1-t0)/2) {
        case 1:register_band<1,0,0,true>(mat,in,tout,t0);break;
        case 2:register_band<2,0,0,true>(mat,in,tout,t0);break;
        case 3:register_band<3,0,0,true>(mat,in,tout,t0);break;
        default:break;
    }
}

M1_TARGET_BW void register_tiles(const MoeCpuMatrix& mat,const PreparedIn& in,float* tout,int t0,int t1,bool grouped) {
    if(in.compact) {
        if(mat.swz) {
            // The swizzled layout stores a 128-output group's eight tiles together.
            TORCH_CHECK(t0%8==0 && t1%8==0,"compact swizzled input requires whole output blocks");
            for(int t=t0;t<t1;t+=8)register_band<4,0,0,true>(mat,in,tout,t);
            return;
        }
        // Unswizzled: whole groups through the traversal, a partial group at either end as tile pairs.
        TORCH_CHECK(t0%2==0 && t1%2==0,"compact input requires whole tile pairs");
        const int a0=std::min(t1,(t0+7)/8*8), a1=std::max(a0,t1/8*8);
        compact_pairs(mat,in,tout,t0,a0);
        if(a1>a0)traversal_tiles(mat,in,tout,a0,a1,grouped);
        compact_pairs(mat,in,tout,a1,t1);
        return;
    }
    for(int n0=t0;n0<t1;) {
        if((n0%2)||t1-n0<2){bw3_blocked_band<1>(mat,in,tout,n0++);continue;}
        const int pairs=std::min({4,(t1-n0)/2,(8-n0%8)/2});
        switch(pairs) {
            case 1:register_band<1>(mat,in,tout,n0);break;
            case 2:register_band<2>(mat,in,tout,n0);break;
            case 3:register_band<3>(mat,in,tout,n0);break;
            case 4:register_band<4>(mat,in,tout,n0);break;
        }
        n0+=pairs*2;
    }
}
