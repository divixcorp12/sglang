// exl3_ram_miss_host_instr.cpp -- the instrumented build: stage trace, full counters and test-only faults.
// The same EXL3 instantiation as exl3_ram_miss_host.cpp with InstrBuild (build_policy.h). A service loads it when
// the process writes a stream trace or injects a RAM-miss fault (expert_stream_transport.host_variant).
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/faulty_reader.h"
#include "expert_stream/host/ffi_exports.h"
#include "expert_stream/host/ffi_test_exports.h"
#include "expert_stream/host/uring_reader.h"

namespace sglang {

using Exl3Reader = expert_stream::FaultyReader<expert_stream::InstrUringReader>;
static_assert(expert_stream::AsyncFileReader<Exl3Reader>);
using Exl3HostExports = expert_stream::HostExports<exl3::Exl3RowLayout, Exl3Reader, expert_stream::InstrBuild>;

static_assert(
    Exl3HostExports::kReaderFaults && Exl3HostExports::kSqeLog && Exl3HostExports::kBallast,
    "the instrumented build keeps every fault");

EXPERT_STREAM_HOST_EXPORTS(Exl3HostExports)
EXPERT_STREAM_HOST_TEST_EXPORTS(Exl3HostExports)

}  // namespace sglang
