# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA runtime binding utilities."""

try:
    from cuda.bindings import driver as drv
    from cuda.bindings import runtime as rt
except ImportError:
    drv = None
    rt = None


def _cudaGetErrorString(error):
    if rt is None:
        return "<cuda.bindings not available>"
    err, msg = rt.cudaGetErrorString(error)
    if err != rt.cudaError_t.cudaSuccess:
        return "<unknown>"
    if isinstance(msg, bytes):
        return msg.decode("utf-8", "replace")
    return str(msg)


def checkCudaErrors(result):
    if rt is None:
        raise RuntimeError(
            "cuda.bindings is not available. Install it with: pip install cuda-python"
        )
    if result[0] != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA error {int(result[0])}({_cudaGetErrorString(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def _count_host_nodes(graph) -> int:
    _, count = checkCudaErrors(drv.cuGraphGetNodes(graph, 0))
    nodes, _ = checkCudaErrors(drv.cuGraphGetNodes(graph, count))
    host = 0
    for node in nodes:
        node_type = checkCudaErrors(drv.cuGraphNodeGetType(node))
        if node_type == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_HOST:
            host += 1
        elif node_type == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
            host += _count_host_nodes(
                checkCudaErrors(drv.cuGraphChildGraphNodeGetGraph(node))
            )
    return host


def capturing_host_node_count(stream_handle: int) -> int:
    """Host nodes (child graphs included) in the graph ``stream_handle`` is capturing."""
    if drv is None:
        raise RuntimeError(
            "cuda.bindings is not available. Install it with: pip install cuda-python"
        )
    info = checkCudaErrors(drv.cuStreamGetCaptureInfo(drv.CUstream(stream_handle)))
    status, graph = info[0], info[2]
    if status != drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise RuntimeError(f"stream {stream_handle:#x} is not capturing ({status})")
    return _count_host_nodes(graph)
