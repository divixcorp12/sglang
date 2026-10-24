// The x86 vector tiers a CPU expert quant may implement, and the tier a build runs at: min(the build's tier, the quant's
// top tier). The build's tier comes from the compiler's macros under the build's -march (kBuildIsa); nothing is
// detected or configured at run time, and a lower tier is tested by building at a lower -march.
#pragma once

#if !defined(__GNUC__) || !defined(__linux__) || !defined(__x86_64__)
#error "the CPU experts framework needs GCC or Clang on x86-64 Linux"
#endif

// The build's tiers as preprocessor flags, for code that holds a tier's instructions and so must not be compiled
// below it. Each tier needs every feature its code is written for, fma and f16c included; kBuildIsa is the same test.
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define SGLANG_CPU_EXPERTS_AVX2 1
#endif
#if defined(SGLANG_CPU_EXPERTS_AVX2) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define SGLANG_CPU_EXPERTS_AVX512 1
#if defined(__AVX512VNNI__)
#define SGLANG_CPU_EXPERTS_VNNI 1
#if defined(__AVX512VBMI__)
#define SGLANG_CPU_EXPERTS_VBMI 1
#endif
#endif
#endif

// For code that must compile its tier's instructions whatever the build's -march (keep_warm.hpp's loops).
#define SGLANG_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define SGLANG_TARGET_BW __attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

// Ordered: a lower tier's code runs on every higher tier's hardware.
enum class Isa { Scalar, Avx2, Bw, Vnni, Vbmi };
template <Isa I> constexpr bool kAvx512 = I == Isa::Bw || I == Isa::Vnni || I == Isa::Vbmi;

constexpr Isa kBuildIsa =
#if defined(SGLANG_CPU_EXPERTS_VBMI)
    Isa::Vbmi;
#elif defined(SGLANG_CPU_EXPERTS_VNNI)
    Isa::Vnni;
#elif defined(SGLANG_CPU_EXPERTS_AVX512)
    Isa::Bw;
#elif defined(SGLANG_CPU_EXPERTS_AVX2)
    Isa::Avx2;
#else
    Isa::Scalar;
#endif

// The tier's name: scalar, avx2, bw, vnni or vbmi.
constexpr const char* isa_name(Isa isa)
{
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Avx2: return "avx2";
        case Isa::Bw: return "bw";
        case Isa::Vnni: return "vnni";
        case Isa::Vbmi: return "vbmi";
    }
    return "unknown";
}

}  // namespace
}  // namespace sglang::cpu_experts
