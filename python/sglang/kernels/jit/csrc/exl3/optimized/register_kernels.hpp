// The DSV4.1 plan's register kernels (register_rows ... register_tiles<M>), one copy per integer accumulate.
// math_avx512.hpp includes this file twice, with no include guard: once in namespace reg_bw (M1_TARGET_BW,
// vpmaddwd + vpaddd) and once in reg_vnni (M1_TARGET_VNNI, vpdpwssd). A function carries one target attribute, so the
// two tiers cannot share a function; including one body keeps them from drifting. The includer defines, around each
// include:
//   M1_REG_TARGET             the target attribute of every function here
//   M1_REG_ACC(acc, sum, pair) acc + sum . pair, the pairwise 16-bit dot of the byte-sum vector with a broadcast
//                             activation pair: wraps mod 2^32 in both forms, so the two tiers are bit-identical.
//   M1_REG_BYTES              1: one-token compact chunks skip the byte sum and accumulate the product bytes
//                             themselves with vpdpbusd (register_rows' byte path, byte_dup_rows); 0: today's kernels.
// Everything the two copies share (register_bytesum, register_word, kRegisterBudget) is defined before the includes.

// The byte path's activations: each compact int16 x (|x| <= 127) as the byte pair (x, x), so a broadcast activation
// pair reads as the bytes (xE, xE, xO, xO). rows quantized rows of one 128-input block, 128 apart, as in compact.
M1_REG_TARGET M1_ALWAYS_INLINE void byte_dup_rows(const int16_t* src,int16_t* dst,int rows) {
    const __m512i low=_mm512_set1_epi16(0x00ff);
    for(int i=0;i<rows*128;i+=32) {
        const __m512i x=_mm512_and_si512(_mm512_loadu_si512(src+i),low);
        _mm512_store_si512(dst+i,_mm512_or_si512(x,_mm512_slli_epi16(x,8)));
    }
}

// Decoded state J (0..15) of a tile pair: 32 16-bit trellis states, from the transposed words a, b, c and the
// preceding lane's last word (prev).
template<int J>
M1_REG_TARGET M1_ALWAYS_INLINE __m512i register_state(__m512i prev,__m512i a,__m512i b,__m512i c) {
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
        return state;
}

template<int J,int Pairs,bool Compact=false,int M=1>
M1_REG_TARGET M1_ALWAYS_INLINE void register_rows(__m512i prev,__m512i a,__m512i b,__m512i c,
    const std::conditional_t<Compact,int16_t,int32_t>* splat,__m512i (&acc)[Pairs][2][2*M],int band) {
    if constexpr(J<16) {
        const __m512i state=register_state<J>(prev,a,b,c);
        constexpr int row=(J/4)*2+(J%2)*8,half=(J/2)%2;
        if constexpr(M1_REG_BYTES && Compact && M==1) {
            // The byte sum's 32-bit product, its low and high halves (register_bytesum), each dotted bytewise with
            // (xE, xE, xO, xO): the sum of all four product bytes times x per state, the integer the byte sum gives.
            const __m512i ml=_mm512_set1_epi16(int16_t(MUL1_MULT&0xffff));
            const __m512i mh=_mm512_set1_epi16(int16_t(MUL1_MULT>>16));
            const __m512i lo=_mm512_mullo_epi16(state,ml);
            const __m512i hi=_mm512_add_epi16(_mm512_mulhi_epu16(state,ml),_mm512_mullo_epi16(state,mh));
            for(int i=0;i<2;++i) {
                uint32_t pair;std::memcpy(&pair,splat+i*128+row,4);
                const __m512i x=_mm512_set1_epi32(int32_t(pair));
                acc[band][half][i]=_mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc[band][half][i],lo,x),hi,x);
            }
            register_rows<J+1,Pairs,Compact,M>(prev,a,b,c,splat,acc,band);
            return;
        }
        const __m512i sum=register_bytesum(state);
        // 2 * M quantized rows, 128 apart: M token rows, then their M residual rows (compact_quantize_block).
        for(int i=0;i<2*M;++i) {
            uint32_t pair;std::memcpy(&pair,reinterpret_cast<const char*>(splat+i*128+row)+(Compact?0:2),4);
            acc[band][half][i]=M1_REG_ACC(acc[band][half][i],sum,_mm512_set1_epi32(int32_t(pair)));
        }
        register_rows<J+1,Pairs,Compact,M>(prev,a,b,c,splat,acc,band);
    }
}

// M token rows (the tokens of one chunk) share each decoded tile. Per output: the int32 sum of each (k-block,
// quantized row), one FMA scale each, fp32 sums in increasing k-block order, then token + residual once: the generic
// path's rounding (bw3_blocked_band).
template<int Pairs,int FixedK=0,int FixedN=0,bool Compact=false,int M=1>
M1_REG_TARGET void register_band(const Exl3Projection& mat,const PreparedIn& in,float* tout,int n0) {
    constexpr int R=2*M;  // quantized rows: M token rows, then their M residual rows
    const int k=FixedK?FixedK:mat.k,n=FixedN?FixedN:mat.n;
    const bool swz=FixedK?false:mat.swz;
    const int tiles_k=k/16,tiles_n=n/16;
    const size_t step=size_t(swz?8:tiles_n)*96;
    __m512 sums[Pairs][2][R];
    alignas(64) static constexpr int previous[16]={7,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14};
    constexpr bool bytes=M1_REG_BYTES && Compact && M==1;  // register_rows' byte path
    [[maybe_unused]] alignas(64) int16_t dup[bytes?R*128:1];
    for(int kb=0;kb<k/128;++kb) {
        __m512i acc[Pairs][2][R];
        for(int b=0;b<Pairs;++b)for(int h=0;h<2;++h)for(int i=0;i<R;++i)acc[b][h][i]=_mm512_setzero_si512();
        if constexpr(bytes) byte_dup_rows(in.compact+size_t(kb)*R*128,dup,R);
        for(int kt=0;kt<8;++kt) {
            const auto* splat=[&]() {
                    if constexpr(bytes) return static_cast<const int16_t*>(dup)+kt*16;
                    else if constexpr(Compact) return in.compact+size_t(kb)*R*128+kt*16;
                    else return in.splat_dup+size_t(kb)*R*128+kt*16;
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
                register_rows<0,Pairs,Compact,M>(prev,a,b,c,splat,acc,band);
            }
        }
        __m512 scales[R],corrections[R];
        for(int i=0;i<R;++i) {
            const float scale=0x1.bb8p-8f*in.bq[kb*MAX_M+i];
            scales[i]=_mm512_set1_ps(scale);
            corrections[i]=_mm512_set1_ps(-510.0f*float(in.bsum[kb*MAX_M+i])*scale);
        }
        for(int b=0;b<Pairs;++b)for(int h=0;h<2;++h)for(int i=0;i<R;++i) {
            const __m512 v=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][h][i]),scales[i],corrections[i]);
            if(kb==0)sums[b][h][i]=v;else sums[b][h][i]=_mm512_add_ps(sums[b][h][i],v);
        }
    }
    // Token row t at tout + t * n gets token + residual; its residual row stays at tout + (M + t) * n, as the generic
    // path leaves it.
    for(int b=0;b<Pairs;++b)for(int t=0;t<M;++t) {
        float* row=tout+size_t(t)*n;
        float* res=tout+size_t(M+t)*n;
        const __m512 lo=_mm512_add_ps(sums[b][0][t],sums[b][0][M+t]);
        const __m512 hi=_mm512_add_ps(sums[b][1][t],sums[b][1][M+t]);
        _mm512_storeu_ps(row+(n0+2*b)*16,_mm512_shuffle_f32x4(lo,hi,0x44));
        _mm512_storeu_ps(row+(n0+2*b+1)*16,_mm512_shuffle_f32x4(lo,hi,0xee));
        _mm512_storeu_ps(res+(n0+2*b)*16,_mm512_shuffle_f32x4(sums[b][0][M+t],sums[b][1][M+t],0x44));
        _mm512_storeu_ps(res+(n0+2*b+1)*16,_mm512_shuffle_f32x4(sums[b][0][M+t],sums[b][1][M+t],0xee));
    }
}

// register_band<P, 0, 0, true, M> for a runtime P in 1..sizeof...(P): one direct call per P.
template<int M,int... P>
M1_REG_TARGET M1_ALWAYS_INLINE void register_band_for(std::integer_sequence<int,P...>,const Exl3Projection& mat,
                                                     const PreparedIn& in,float* tout,int n0,int pairs) {
    (void)((pairs==P+1 && (register_band<P+1,0,0,true,M>(mat,in,tout,n0),true)) || ...);
}

// register_rows for one output half: the states J with (J / 2) % 2 == Half, each decoded once, into that half's 2 * M
// rows (register_band_halves).
template<int J,int Pairs,int M,int Half>
M1_REG_TARGET M1_ALWAYS_INLINE void register_rows_half(__m512i prev,__m512i a,__m512i b,__m512i c,
    const int16_t* splat,__m512i (&acc)[Pairs][2*M],int band) {
    if constexpr(J<16) {
        if constexpr((J/2)%2==Half) {
            const __m512i sum=register_bytesum(register_state<J>(prev,a,b,c));
            constexpr int row=(J/4)*2+(J%2)*8;
            for(int i=0;i<2*M;++i) {
                uint32_t pair;std::memcpy(&pair,splat+i*128+row,4);
                acc[band][i]=M1_REG_ACC(acc[band][i],sum,_mm512_set1_epi32(int32_t(pair)));
            }
        }
        register_rows_half<J+1,Pairs,M,Half>(prev,a,b,c,splat,acc,band);
    }
}

// register_band<Pairs, 0, 0, true, M> for chunks whose 4 * M accumulators per tile pair do not fit the registers
// (kRegisterBudget's split entries). Each 128-input block runs its eight k-tiles once per output half, holding only
// that half's 2 * M accumulators; the second pass reloads the tiles from L1 and decodes the other half's states, so
// every state is still decoded once. Each output's integer sums are the same exact int32 values, and its per-block
// scale, FP32 sums in increasing k-block order and token + residual are register_band's operations in its order.
template<int Pairs,int M>
M1_REG_TARGET void register_band_halves(const Exl3Projection& mat,const PreparedIn& in,float* tout,int n0) {
    constexpr int R=2*M;
    const int k=mat.k,n=mat.n,tiles_n=n/16;
    const size_t step=size_t(tiles_n)*96;
    __m512 sums[Pairs][2][R];
    alignas(64) static constexpr int previous[16]={7,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14};
    for(int kb=0;kb<k/128;++kb) {
        __m512 scales[R],corrections[R];
        for(int i=0;i<R;++i) {
            const float scale=0x1.bb8p-8f*in.bq[kb*MAX_M+i];
            scales[i]=_mm512_set1_ps(scale);
            corrections[i]=_mm512_set1_ps(-510.0f*float(in.bsum[kb*MAX_M+i])*scale);
        }
        auto half=[&]<int Half>() {
            __m512i acc[Pairs][R];
            for(int b=0;b<Pairs;++b)for(int i=0;i<R;++i)acc[b][i]=_mm512_setzero_si512();
            for(int kt=0;kt<8;++kt) {
                const int16_t* splat=in.compact+size_t(kb)*R*128+kt*16;
                for(int band=0;band<Pairs;++band) {
                    const int nt=n0+2*band,tile_k=kb*8+kt;
                    const uint32_t* packed=reinterpret_cast<const uint32_t*>(mat.trellis+(size_t(tile_k)*tiles_n+nt)*48);
                    if constexpr(Half==0) {
                        const uintptr_t future=reinterpret_cast<uintptr_t>(packed)+step*2;
                        #pragma GCC unroll 3
                        for(int line=0;line<3;++line)_mm_prefetch(reinterpret_cast<const char*>(future+line*64),_MM_HINT_T0);
                    }
                    const __m512i p0=_mm512_loadu_si512(packed),p1=_mm512_loadu_si512(packed+16),p2=_mm512_loadu_si512(packed+32);
                    const __m512i a=register_word<0>(p0,p1,p2),b=register_word<1>(p0,p1,p2),c=register_word<2>(p0,p1,p2);
                    const __m512i prev=_mm512_permutexvar_epi32(_mm512_load_si512(previous),c);
                    register_rows_half<0,Pairs,M,Half>(prev,a,b,c,splat,acc,band);
                }
            }
            for(int b=0;b<Pairs;++b)for(int i=0;i<R;++i) {
                const __m512 v=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]),scales[i],corrections[i]);
                if(kb==0)sums[b][Half][i]=v;else sums[b][Half][i]=_mm512_add_ps(sums[b][Half][i],v);
            }
        };
        half.template operator()<0>();
        half.template operator()<1>();
    }
    for(int b=0;b<Pairs;++b)for(int t=0;t<M;++t) {
        float* row=tout+size_t(t)*n;
        float* res=tout+size_t(M+t)*n;
        const __m512 lo=_mm512_add_ps(sums[b][0][t],sums[b][0][M+t]);
        const __m512 hi=_mm512_add_ps(sums[b][1][t],sums[b][1][M+t]);
        _mm512_storeu_ps(row+(n0+2*b)*16,_mm512_shuffle_f32x4(lo,hi,0x44));
        _mm512_storeu_ps(row+(n0+2*b+1)*16,_mm512_shuffle_f32x4(lo,hi,0xee));
        _mm512_storeu_ps(res+(n0+2*b)*16,_mm512_shuffle_f32x4(sums[b][0][M+t],sums[b][1][M+t],0x44));
        _mm512_storeu_ps(res+(n0+2*b+1)*16,_mm512_shuffle_f32x4(sums[b][0][M+t],sums[b][1][M+t],0xee));
    }
}

// Visit neighboring 128-output bands before advancing to the next 128-input block.
// Original packed weights and each output's increasing-kb FP32 addition order stay
// unchanged. One integer band is live at a time; FP32 partials reside in private
// aligned scratch. The one-band variant controls for this factoring/scratch cost.

struct IntegerAccum { __m512i lanes[4][2][2]; };
M1_REG_TARGET inline void integer_acc_zero(IntegerAccum& a) {
    for(int b=0;b<4;++b)for(int h=0;h<2;++h)for(int i=0;i<2;++i)a.lanes[b][h][i]=_mm512_setzero_si512();
}

M1_REG_TARGET __attribute__((noinline))
void traversal_kblock(const Exl3Projection& mat, const PreparedIn& in,
                      __m512 (&sums)[4][2][2], int n0, int kb) {
    const int tiles_k=mat.k/16, tiles_n=mat.n/16;
    const bool swz=mat.swz;
    const size_t step=size_t(swz?8:tiles_n)*96;
    alignas(64) static constexpr int previous[16]={7,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14};
    IntegerAccum first;integer_acc_zero(first);
    [[maybe_unused]] alignas(64) int16_t dup[M1_REG_BYTES?256:1];  // register_rows' byte path
    if constexpr(M1_REG_BYTES) byte_dup_rows(in.compact+size_t(kb)*256,dup,2);
    for(int kt=0;kt<8;++kt) {
        auto& acc=first.lanes;
        const int16_t* splat=(M1_REG_BYTES?static_cast<const int16_t*>(dup):in.compact+size_t(kb)*256)+kt*16;
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
M1_REG_TARGET void traversal_group(const Exl3Projection& mat, const PreparedIn& in,
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

M1_REG_TARGET void traversal_tiles(const Exl3Projection& mat,const PreparedIn& in,
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
M1_REG_TARGET void compact_pairs(const Exl3Projection& mat,const PreparedIn& in,float* tout,int t0,int t1) {
    switch((t1-t0)/2) {
        case 1:register_band<1,0,0,true>(mat,in,tout,t0);break;
        case 2:register_band<2,0,0,true>(mat,in,tout,t0);break;
        case 3:register_band<3,0,0,true>(mat,in,tout,t0);break;
        default:break;
    }
}

// The tokens of one chunk, M = 1..CHUNK_M. M = 1 is the one-token kernel either plan calls (compact or not, swizzled
// or not; the grouped traversal: kRegisterBudget[0]). M > 1 is the DSV4.1 plan's compact, unswizzled input, in runs of
// up to kRegisterBudget[M - 1].pairs tile pairs; M = 5..8 (halves entries) through register_band_halves.
template<int M>
M1_REG_TARGET void register_tiles(const Exl3Projection& mat,const PreparedIn& in,float* tout,int t0,int t1,
                                 [[maybe_unused]] bool grouped) {
    [[maybe_unused]] constexpr RegisterBudget budget=kRegisterBudget[M-1];
    if constexpr (M==1) {
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
    } else {
        static_assert(!budget.traversal,"the grouped traversal holds one token's rows (IntegerAccum)");
        TORCH_CHECK(in.compact && !mat.swz && t0%2==0 && t1%2==0,
                    "a chunk of several tokens needs compact, unswizzled tile pairs");
        if constexpr (budget.halves) {
            static_assert(budget.pairs==1,"register_band_halves runs one tile pair per call here");
            for(int n0=t0;n0<t1;n0+=2)register_band_halves<1,M>(mat,in,tout,n0);
            return;
        }
        for(int n0=t0;n0<t1;) {
            const int pairs=std::min(budget.pairs,(t1-n0)/2);
            register_band_for<M>(std::make_integer_sequence<int,budget.pairs>{},mat,in,tout,n0,pairs);
            n0+=pairs*2;
        }
    }
}
