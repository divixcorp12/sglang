// The toy quant as a shared library, for test_cpu_experts_common.py's portable-build check: the toy's warm loop is the
// build tier's, and so is the only keep_warm_detail function the library holds.
#include "cpu_experts_common_toy.hpp"

namespace toy {
__attribute__((visibility("default"))) int32_t toy_warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    return Toy::warm(word, seen, deadline_ns);
}
}  // namespace toy
