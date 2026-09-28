"""Run the real enqueue helper against a CPU driver double; no CUDA device is needed."""

import shutil
import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_wait_enqueue_preserves_capture_dependencies_and_driver_errors(tmp_path: Path):
    # Regressions caught: enqueuing eagerly during capture, losing incoming edges,
    # failing to make subsequent kernels wait, or continuing after a driver error.
    compiler = shutil.which("c++")
    assert compiler is not None, "the CPU kernel tests require a C++ compiler"
    root = Path(__file__).resolve().parents[4]
    include = tmp_path / "include"
    (include / "sgl_kernel").mkdir(parents=True)
    (include / "cuda_runtime.h").write_text(
        '#pragma once\n#include "cuda.h"\nusing cudaStream_t = CUstream;\n'
    )
    (include / "sgl_kernel" / "utils.h").write_text(
        """#pragma once
#include <sstream>
#include <stdexcept>
namespace sglang::host {
template <typename... Args> void RuntimeCheck(bool ok, Args&&... args) {
  if (ok) return;
  std::ostringstream message;
  (message << ... << args);
  throw std::runtime_error(message.str());
}
}
"""
    )
    (include / "cuda.h").write_text(
        """#pragma once
#include <cstddef>
#include <cstdint>
using CUstream = struct CUstream_st*;
using CUcontext = struct CUctx_st*;
using CUgraph = struct CUgraph_st*;
using CUgraphNode = struct CUgraphNode_st*;
using CUdeviceptr = std::uintptr_t;
using cuuint64_t = std::uint64_t;
enum CUresult { CUDA_SUCCESS, CUDA_ERROR_NOT_SUPPORTED };
enum CUstreamCaptureStatus {
  CU_STREAM_CAPTURE_STATUS_NONE, CU_STREAM_CAPTURE_STATUS_ACTIVE, CU_STREAM_CAPTURE_STATUS_INVALIDATED
};
enum CUgraphNodeType { CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12 };
enum CUstreamBatchMemOpType { CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1 };
constexpr unsigned CU_STREAM_WAIT_VALUE_EQ = 1;
constexpr unsigned CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1;
struct CUgraphEdgeData { unsigned char from_port, to_port, type, reserved[5]; };
union CUstreamBatchMemOpParams {
  CUstreamBatchMemOpType operation;
  struct {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union { std::uint32_t value; std::uint64_t value64; };
    unsigned flags;
    CUdeviceptr alias;
  } waitValue;
  std::uint64_t pad[6];
};
struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 {
  CUcontext ctx;
  unsigned count;
  CUstreamBatchMemOpParams* paramArray;
  unsigned flags;
};
struct CUgraphNodeParams {
  CUgraphNodeType type;
  int reserved0[3];
  union { long long reserved1[29]; CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 memOp; char asBytes[232]; };
  long long reserved2;
};
CUresult cuGetErrorName(CUresult, const char**);
CUresult cuStreamIsCapturing(CUstream, CUstreamCaptureStatus*);
CUresult cuStreamGetCtx(CUstream, CUcontext*);
CUresult cuStreamGetCaptureInfo(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*,
                              const CUgraphNode**, const CUgraphEdgeData**, std::size_t*);
CUresult cuStreamWaitValue32(CUstream, CUdeviceptr, std::uint32_t, unsigned);
CUresult cuGraphAddNode(CUgraphNode*, CUgraph, const CUgraphNode*, const CUgraphEdgeData*,
                      std::size_t, CUgraphNodeParams*);
CUresult cuStreamUpdateCaptureDependencies(CUstream, CUgraphNode*, const CUgraphEdgeData*,
                                         std::size_t, unsigned);
"""
    )
    source = tmp_path / "wait_enqueue.cpp"
    source.write_text(
        r"""#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "moe/expert_stream/stream_wait.cuh"

static CUstream stream = reinterpret_cast<CUstream>(0x10);
static CUcontext context = reinterpret_cast<CUcontext>(0x20);
static CUgraph graph = reinterpret_cast<CUgraph>(0x30);
static CUgraphNode incoming[] = {reinterpret_cast<CUgraphNode>(0x40), reinterpret_cast<CUgraphNode>(0x50)};
static CUgraphNode gate = reinterpret_cast<CUgraphNode>(0x60);
static CUgraphEdgeData edges[] = {{2, 0, 1, {}}, {0, 0, 0, {}}};
static std::vector<std::string> calls;
static std::string mode, fail_at;
static bool captured = false, frontier_set = false;
static CUresult call(const char* name) {
  calls.emplace_back(name);
  return fail_at == name ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
}
CUresult cuGetErrorName(CUresult error, const char** name) {
  assert(error == CUDA_ERROR_NOT_SUPPORTED);
  *name = "CUDA_ERROR_NOT_SUPPORTED";
  return CUDA_SUCCESS;
}
CUresult cuStreamIsCapturing(CUstream s, CUstreamCaptureStatus* status) {
  assert(s == stream);
  *status = mode == "eager" ? CU_STREAM_CAPTURE_STATUS_NONE :
            mode == "invalid" ? CU_STREAM_CAPTURE_STATUS_INVALIDATED : CU_STREAM_CAPTURE_STATUS_ACTIVE;
  return call("query");
}
CUresult cuStreamGetCtx(CUstream s, CUcontext* out) {
  assert(s == stream);
  assert(!captured); // Would invalidate the driver-owned capture dependency arrays.
  *out = context;
  return call("context");
}
CUresult cuStreamGetCaptureInfo(CUstream s, CUstreamCaptureStatus* status, cuuint64_t*, CUgraph* out,
                              const CUgraphNode** deps, const CUgraphEdgeData** data, std::size_t* count) {
  assert(s == stream);
  captured = true;
  *status = mode == "invalid_after_query" ? CU_STREAM_CAPTURE_STATUS_INVALIDATED : CU_STREAM_CAPTURE_STATUS_ACTIVE;
  *out = graph;
  *deps = incoming;
  *data = edges;
  *count = mode == "empty" ? 0 : 2;
  return call("capture");
}
CUresult cuStreamWaitValue32(CUstream s, CUdeviceptr address, std::uint32_t value, unsigned flags) {
  assert(s == stream && address == 0x1000 && value == 17 && flags == CU_STREAM_WAIT_VALUE_EQ);
  assert(mode == "eager");
  return call("wait");
}
CUresult cuGraphAddNode(CUgraphNode* out, CUgraph g, const CUgraphNode* deps,
                      const CUgraphEdgeData* data, std::size_t count, CUgraphNodeParams* params) {
  assert(captured && g == graph);
  assert(count == (mode == "empty" ? 0 : 2));
  for (std::size_t i = 0; i < count; ++i) {
    assert(deps[i] == incoming[i]);
    assert(data[i].from_port == edges[i].from_port && data[i].type == edges[i].type);
  }
  assert(params->type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP);
  assert(params->memOp.ctx == context && params->memOp.count == 1 && params->memOp.flags == 0);
  const auto& op = params->memOp.paramArray[0];
  assert(op.operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
  assert(op.waitValue.address == 0x1000 && op.waitValue.value == 17 && op.waitValue.flags == CU_STREAM_WAIT_VALUE_EQ);
  assert(params->reserved2 == 0);
  for (int reserved : params->reserved0) assert(reserved == 0);
  *out = gate;
  return call("add");
}
CUresult cuStreamUpdateCaptureDependencies(CUstream s, CUgraphNode* deps, const CUgraphEdgeData* data,
                                         std::size_t count, unsigned flags) {
  assert(s == stream && count == 1 && deps[0] == gate);
  assert(flags == CU_STREAM_SET_CAPTURE_DEPENDENCIES);
  assert(data == nullptr || (data[0].from_port == 0 && data[0].to_port == 0 && data[0].type == 0));
  frontier_set = true;
  return call("frontier");
}
int main(int argc, char** argv) {
  assert(argc == 3);
  mode = argv[1];
  fail_at = argv[2];
  bool threw = false;
  try {
    sglang::host::enqueue_expert_stream_wait(stream, 0x1000, 17, CU_STREAM_WAIT_VALUE_EQ);
  } catch (const std::runtime_error& error) {
    threw = true;
    if (!fail_at.empty()) assert(std::string(error.what()).find("CUDA_ERROR_NOT_SUPPORTED") != std::string::npos);
  }
  assert(threw == (!fail_at.empty() || mode == "invalid" || mode == "invalid_after_query"));
  if (!threw && mode != "eager") assert(frontier_set);
  for (const auto& name : calls) std::cout << name << '\n';
}
"""
    )
    executable = tmp_path / "wait_enqueue"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{include}",
            f"-I{root / 'python/sglang/kernels/jit/csrc'}",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    scenarios = [
        ("eager", "", ["query", "wait"]),
        ("capture", "", ["query", "context", "capture", "add", "frontier"]),
        ("empty", "", ["query", "context", "capture", "add", "frontier"]),
        ("invalid", "", ["query"]),
        ("invalid_after_query", "", ["query", "context", "capture"]),
        ("eager", "query", ["query"]),
        ("eager", "wait", ["query", "wait"]),
        ("capture", "context", ["query", "context"]),
        ("capture", "capture", ["query", "context", "capture"]),
        ("capture", "add", ["query", "context", "capture", "add"]),
        ("capture", "frontier", ["query", "context", "capture", "add", "frontier"]),
    ]
    for mode, failure, expected in scenarios:
        result = subprocess.run(
            [str(executable), mode, failure],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (mode, failure, result.stderr)
        assert result.stdout.splitlines() == expected, (mode, failure, result.stdout)
