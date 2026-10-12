// The edges of a captured CUDA graph, for the chain-PDL probe: did PDL survive stream capture as programmatic
// edges, or was every kernel serialized behind a plain full dependency? One line per edge, written into `out`
// (uint8, CPU): "<from kernel>\t<to kernel>\t<edge type>\t<from port>\n", where type 1 is
// cudaGraphDependencyTypeProgrammatic and port 1 / 2 are cudaGraphKernelNodePortProgrammatic / LaunchCompletion.
// A node that is not a kernel is named "node<type>". Returns the bytes written.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sglang {

namespace chain_graph {

inline std::string node_name(cudaGraphNode_t node) {
  cudaGraphNodeType type;
  CHECK_CUDA(cudaGraphNodeGetType(node, &type)) << "cudaGraphNodeGetType";
  if (type != cudaGraphNodeTypeKernel) return "node" + std::to_string(static_cast<int>(type));
  cudaKernelNodeParams params{};
  CHECK_CUDA(cudaGraphKernelNodeGetParams(node, &params)) << "cudaGraphKernelNodeGetParams";
  const char* name = nullptr;
  if (cudaFuncGetName(&name, params.func) != cudaSuccess || name == nullptr) return "kernel?";
  return name;
}

}  // namespace chain_graph

int64_t chain_graph_edges(int64_t graph, tvm::ffi::TensorView out) {
  const auto g = reinterpret_cast<cudaGraph_t>(graph);
  size_t n = 0;
  CHECK_CUDA(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &n)) << "cudaGraphGetEdges (count)";
  std::vector<cudaGraphNode_t> from(n), to(n);
  std::vector<cudaGraphEdgeData> data(n);
  CHECK_CUDA(cudaGraphGetEdges(g, from.data(), to.data(), data.data(), &n)) << "cudaGraphGetEdges";
  std::string text;
  for (size_t i = 0; i < n; ++i) {
    text += chain_graph::node_name(from[i]) + "\t" + chain_graph::node_name(to[i]) + "\t" +
            std::to_string(static_cast<int>(data[i].type)) + "\t" + std::to_string(static_cast<int>(data[i].from_port)) +
            "\n";
  }
  host::RuntimeCheck(static_cast<int64_t>(text.size()) <= out.size(0), "chain_graph_edges: out too small for ",
                     text.size(), " bytes");
  std::memcpy(out.data_ptr(), text.data(), text.size());
  return static_cast<int64_t>(text.size());
}

}  // namespace sglang
