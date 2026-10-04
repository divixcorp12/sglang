// The toy quant as a shared library, for test_cpu_experts_common.py's per-library state and portable-build checks.
#include "cpu_experts_common_toy.hpp"

SGLANG_CPU_EXPERTS_DEFINE_CABI(toy, toy::ToyQuant)
