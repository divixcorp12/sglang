#pragma once

#include <sgl_kernel/utils.h>

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace sglang::host {

inline void expert_stream_driver_check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  RuntimeCheck(false, operation, " failed: ", name ? name : "unknown CUDA driver error");
}

// The caller owns the aligned, device-addressable completion word for the entire
// graph lifetime. Its producer must also complete failed/cancelled requests: a
// stream memory wait has no timeout. A later kernel acquires and validates the
// completion record before consuming any payload.
inline void enqueue_expert_stream_wait(cudaStream_t stream, CUdeviceptr address, uint32_t value, unsigned flags) {
  CUstreamCaptureStatus status;
  expert_stream_driver_check(cuStreamIsCapturing(stream, &status), "cuStreamIsCapturing");
  if (status == CU_STREAM_CAPTURE_STATUS_NONE) {
    expert_stream_driver_check(cuStreamWaitValue32(stream, address, value, flags), "cuStreamWaitValue32");
    return;
  }
  RuntimeCheck(status == CU_STREAM_CAPTURE_STATUS_ACTIVE, "expert-stream wait: CUDA stream capture is invalidated");

  // Get the context before the capture frontier: the driver-owned dependency
  // arrays remain valid only until the next API call operating on the stream.
  CUcontext context;
  expert_stream_driver_check(cuStreamGetCtx(stream, &context), "cuStreamGetCtx");
  CUgraph graph;
  const CUgraphNode* dependencies = nullptr;
  const CUgraphEdgeData* edge_data = nullptr;
  size_t dependency_count = 0;
  expert_stream_driver_check(
      cuStreamGetCaptureInfo(stream, &status, nullptr, &graph, &dependencies, &edge_data, &dependency_count),
      "cuStreamGetCaptureInfo");
  RuntimeCheck(status == CU_STREAM_CAPTURE_STATUS_ACTIVE, "expert-stream wait: CUDA stream capture is invalidated");

  CUstreamBatchMemOpParams operation{};
  operation.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
  operation.waitValue.address = address;
  operation.waitValue.value = value;
  operation.waitValue.flags = flags;
  CUgraphNodeParams params{};
  params.type = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
  params.memOp.ctx = context;
  params.memOp.count = 1;
  params.memOp.paramArray = &operation;

  CUgraphNode wait_node;
  expert_stream_driver_check(
      cuGraphAddNode(&wait_node, graph, dependencies, edge_data, dependency_count, &params), "cuGraphAddNode");
  // Merely adding the node does not make subsequent captured work depend on it.
  expert_stream_driver_check(
      cuStreamUpdateCaptureDependencies(stream, &wait_node, nullptr, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES),
      "cuStreamUpdateCaptureDependencies");
}

}  // namespace sglang::host
