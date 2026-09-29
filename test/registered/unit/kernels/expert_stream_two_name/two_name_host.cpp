// Test only: the TwoNameLayout instantiation of the expert-stream host
// transport, the EXL3 instrumented file (InstrBuild) with its layout swapped.
// Relative includes: this file is outside csrc/, and load_jit adds no include
// path for it.
#include "../../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/build_policy.h"
#include "../../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/faulty_reader.h"
#include "../../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h"
#include "../../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/uring_reader.h"
#include "two_name_layout.h"

namespace sglang {

using TwoNameReader = expert_stream::FaultyReader<expert_stream::UringReader>;
using TwoNameHostExports =
    expert_stream::HostExports<expert_stream::testing::TwoNameLayout,
                               TwoNameReader, expert_stream::InstrBuild>;

EXPERT_STREAM_HOST_EXPORTS(TwoNameHostExports)

} // namespace sglang
