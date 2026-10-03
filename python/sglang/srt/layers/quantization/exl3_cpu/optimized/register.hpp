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

#include "traversal.hpp"

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
