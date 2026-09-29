// exl3_ram_miss_host.cpp -- the production build.
// The EXL3 instantiation of the expert-stream host transport: its row layout, file reader and build policy
// (ProdBuild, build_policy.h), and the FFI exports. exl3_ram_miss_host_instr.cpp is the instrumented build.
// Production reads through the bare UringReader: no FaultyReader, no fault state (plan 2026-09-29-hotpath-zero-overhead
// Task 10); every test-only export refuses, naming the instrumented build.
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/ffi_exports.h"
#include "expert_stream/host/uring_reader.h"

namespace sglang {

using Exl3Reader = expert_stream::UringReader;
static_assert(expert_stream::AsyncFileReader<Exl3Reader>);
using Exl3HostExports = expert_stream::HostExports<exl3::Exl3RowLayout, Exl3Reader, expert_stream::ProdBuild>;

static_assert(!expert_stream::HasRingFaultHooks<Exl3Reader>, "production reads through a reader with no fault hooks");
static_assert(
    !Exl3HostExports::kReaderFaults && !Exl3HostExports::kSqeLog && !Exl3HostExports::kBallast,
    "the production build carries no fault state");

EXPERT_STREAM_HOST_EXPORTS(Exl3HostExports)

}  // namespace sglang
