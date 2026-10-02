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
