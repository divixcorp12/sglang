// engine_host.cpp -- the expert-stream engine's host module, one translation unit for every format and router.
//
// The format (-DSGLANG_EXPERT_FORMAT) and router (-DSGLANG_EXPERT_ROUTER) are picked by engine_select.h; this file binds
// that format's row layout and CPU-kernel binding, the file reader and the build policy (build_policy.h) into
// HostExports and expands the FFI exports. Two builds, as before the engine was generic:
//   production (default)            the bare UringReader and ProdBuild: no FaultyReader, no fault state (plan
//                                   2026-09-29-hotpath-zero-overhead Task 10); every test-only export refuses, naming
//                                   the instrumented build.
//   instrumented (-DSGLANG_EXPERT_STREAM_INSTR=1)   InstrBuild with stage trace, full counters and test-only faults. A
//                                   service loads it when the process writes a stream trace or injects a RAM-miss fault
//                                   (expert_stream_transport.host_variant).
// The module's host gate scorer is the engine's router: SGLANG_EXPERT_STREAM_GATE_ROUTER (host/gate_scorer.h).
#define SGLANG_EXPERT_STREAM_GATE_ROUTER ::sglang::expert_stream::routers::SGLANG_EXPERT_ROUTER

#include "engine_select.h"
#include "host/build_policy.h"
#if defined(SGLANG_EXPERT_STREAM_INSTR) && SGLANG_EXPERT_STREAM_INSTR
#include "host/faulty_reader.h"
#endif
#include "host/ffi_exports.h"
#include "host/ffi_test_exports.h"
#include "host/uring_reader.h"

namespace sglang {

#if defined(SGLANG_EXPERT_STREAM_INSTR) && SGLANG_EXPERT_STREAM_INSTR
using EngineReader = expert_stream::FaultyReader<expert_stream::InstrUringReader>;
using EngineBuild = expert_stream::InstrBuild;
#else
using EngineReader = expert_stream::UringReader;
using EngineBuild = expert_stream::ProdBuild;
#endif
static_assert(expert_stream::AsyncFileReader<EngineReader>);
using EngineHostExports = expert_stream::
    HostExports<expert_stream::EngineLayout, EngineReader, EngineBuild, expert_stream::EngineFormat::CpuKernel>;

#if defined(SGLANG_EXPERT_STREAM_INSTR) && SGLANG_EXPERT_STREAM_INSTR
static_assert(
    EngineHostExports::kReaderFaults && EngineHostExports::kSqeLog && EngineHostExports::kBallast,
    "the instrumented build keeps every fault");
#else
static_assert(!expert_stream::HasRingFaultHooks<EngineReader>, "production reads through a reader with no fault hooks");
static_assert(
    !EngineHostExports::kReaderFaults && !EngineHostExports::kSqeLog && !EngineHostExports::kBallast,
    "the production build carries no fault state");
#endif

EXPERT_STREAM_HOST_EXPORTS(EngineHostExports)
EXPERT_STREAM_HOST_TEST_EXPORTS(EngineHostExports)

}  // namespace sglang
