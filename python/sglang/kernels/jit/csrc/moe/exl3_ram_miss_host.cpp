// exl3_ram_miss_host.cpp -- the production build.
// The EXL3 instantiation of the expert-stream host transport: its row layout, file reader and build policy
// (ProdBuild, build_policy.h), and the FFI exports. exl3_ram_miss_host_instr.cpp is the instrumented build.
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/faulty_reader.h"
#include "expert_stream/host/ffi_exports.h"
#include "expert_stream/host/uring_reader.h"

namespace sglang {

using Exl3Reader = expert_stream::FaultyReader<expert_stream::UringReader>;
static_assert(expert_stream::AsyncFileReader<Exl3Reader>);
using Exl3HostExports = expert_stream::HostExports<exl3::Exl3RowLayout, Exl3Reader, expert_stream::ProdBuild>;

EXPERT_STREAM_HOST_EXPORTS(Exl3HostExports)

}  // namespace sglang
