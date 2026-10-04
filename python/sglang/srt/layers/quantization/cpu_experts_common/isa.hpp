// The x86 vector tiers a CPU expert quant may implement, and the tier one runs at: min(hardware, the quant's top
// tier, the quant's cap environment variable).
#pragma once
#include <cctype>
#include <cstdlib>
#include <string>

#if !defined(__GNUC__) || !defined(__linux__) || !defined(__x86_64__)
#error "cpu_experts_common needs GCC or Clang on x86-64 Linux"
#endif

#define SGLANG_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define SGLANG_TARGET_BW __attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
#define SGLANG_TARGET_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#define SGLANG_TARGET_VBMI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

// Ordered: a lower tier's code runs on every higher tier's hardware.
enum class Isa { Scalar, Avx2, Bw, Vnni, Vbmi };
template <Isa I> constexpr bool kAvx512 = I == Isa::Bw || I == Isa::Vnni || I == Isa::Vbmi;

// cap_env (may be null) names a variable holding scalar|avx2|bw|avx512bw|vnni|avx512|vbmi, case-insensitive, that
// caps the tier for testing; it never raises the tier, and an unrecognized value is ignored.
inline Isa detect_isa(Isa top, const char* cap_env)
{
    __builtin_cpu_init();
    Isa hw;
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("fma")) {
        if (__builtin_cpu_supports("avx512vnni"))
            hw = __builtin_cpu_supports("avx512vbmi") ? Isa::Vbmi : Isa::Vnni;
        else
            hw = Isa::Bw;
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        hw = Isa::Avx2;
    else
        hw = Isa::Scalar;
    if (top < hw) hw = top;

    if (const char* e = cap_env ? std::getenv(cap_env) : nullptr) {
        std::string s(e);
        for (char& c : s) c = (char) std::tolower((unsigned char) c);
        Isa cap;
        if (s == "scalar") cap = Isa::Scalar;
        else if (s == "avx2") cap = Isa::Avx2;
        else if (s == "bw" || s == "avx512bw") cap = Isa::Bw;
        else if (s == "vnni" || s == "avx512") cap = Isa::Vnni;
        else if (s == "vbmi") cap = Isa::Vbmi;
        else return hw;
        if (cap < hw) hw = cap;
    }
    return hw;
}

}  // namespace
}  // namespace sglang::cpu_experts
