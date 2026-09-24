"""T8 and T9 (docs/superpowers/plans/task6-v1-checklist.md section 5): the V1 two-phase device
chain's graph topology and output parity, against real CUDA kernels on a real GPU.

T8: the graph D7 (``Exl3RamMissRowBackend.post``, two-phase branch) captures is the linear chain
``post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F`` -- checked with ``cudaGraphGetEdges``, not assumed
(PER_ROW_TRANSFER.md section 5.2 item 1). The harness for reading a captured graph's node/edge
structure is copied from ``cuda_graph_dedup_mixin.py``'s ``graph_signature``, not derived: kernel
identity comes from ``cuGraphKernelNodeGetParams(node).func``, a stable per-``__global__``-function
handle, and edges come from ``cuGraphGetEdges`` on ``torch.cuda.CUDAGraph(keep_graph=True)``'s
``raw_cuda_graph()``. A second test captures the real production apply (the "-> fused" clause the
checklist's row also names) and checks F is comparable to -- an ancestor or descendant of, never
incomparable with -- every other node in that larger graph, so nothing can race it on a fork the
stream-ordering happens to hide today.

T9: two-phase's output -- both the destination rows the copy kernels write and the fused MoE
consumer's bytes over them -- is bitwise identical to **M1**, the Task 5 lease-mode batched arm
(``exl3_ram_miss_lease_wait_kernel`` + one copy + one ack, ``two_phase=False``), on fixed routes
and seeds, eager and graph-captured. "M1" here is that measurement arm; it is not to be confused
with the "A1"/"A2" kernel *stages* the chain above names (task6-v1-checklist.md section 6).

Run on divix01 under ``gpu-run.sh`` (it holds cc-gpu.lock) with PYTHONPATH pointing at the tree
under test. Recipes copied rather than derived, per the plan's instruction:
  - GPU harness shape (pytestmark skip, hand-driven lease block, real-service end-to-end):
    test/manual/dsv41/test_exl3_lease_kernels_cuda.py
  - The graph-capture / fused-apply / byte-equality pattern for T9:
    test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py
  - The raw-graph node/edge enumeration (T8):
    python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py (graph_signature)
"""

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NEEDS_EXL3_SRC = pytest.mark.skipif(
    not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC (an exllamav3 checkout)"
)

try:
    from cuda.bindings import driver as cuda_drv  # noqa: E402
except ImportError:  # pragma: no cover - exercised only when cuda-python is missing
    cuda_drv = None

from sglang.kernels.ops.moe import exl3_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe.exl3_ram_miss import (  # noqa: E402
    Exl3RamMissDevice,
    Exl3RamMissHost,
    new_page,
)
from sglang.kernels.ops.moe.expert_cache_transfer import (  # noqa: E402
    copy_expert_row_segments_gpu,
    expert_row_segments,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (  # noqa: E402
    checkCudaErrors,
)

LAYERS, EXPERTS, CAPACITY = 1, 16, 8
TOP_K = 4


def _cuda_ready():
    torch.cuda.synchronize()


def _close(host, slabs):
    from sglang.srt.layers.moe.expert_host_tier import release_host_slabs

    try:
        if host is not None:
            host.stop()
    finally:
        release_host_slabs([slab for names in slabs.values() for slab in names.values()])


# ---------------------------------------------------------------------------------------------------------------
# T8's low-level harness: exactly the kernel chain, nothing else. Copied in shape from
# test_exl3_lease_kernels_cuda.py's Service, trimmed to what topology inspection needs, and extended
# with the two-phase step chain D7 builds.
# ---------------------------------------------------------------------------------------------------------------
class Service:
    def __init__(self, tmp_path, *, timeout_ms=2000):
        from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
        from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
        from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
        from sglang.test.dsv41_fake_exl3 import write_fake_exl3

        write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS, hidden=1024, inter=512, finite=True)
        self.layout = build_exl3_expert_layout(str(tmp_path))
        self.fmt = Exl3ExpertFormat(self.layout, 0, direct=False)
        self.specs = {s.name: s for s in self.fmt.tensor_specs(None)}
        self.names = EXL3_STREAMED_NAMES
        self.slabs = {lid: {} for lid in range(LAYERS)}
        self.host = None
        try:
            for lid in range(LAYERS):
                for n in self.names:
                    self.slabs[lid][n] = allocate_host_slab(CAPACITY, self.specs[n].row_shape, self.specs[n].dtype, register=True)
            tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs)
            self.page = new_page(pin=True)
            slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
            self.slot_map = slot_map
            self.host = Exl3RamMissHost(tables, page=self.page, slot_map=slot_map, direct=False)
            self.host.enable_lease_mode()
            self.host.enable_two_phase()
            self.host.start_thread(fatal_wait_s=60.0)
            self.dev = Exl3RamMissDevice(
                self.page, slot_map, device="cuda", layers=LAYERS, timeout_ms=timeout_ms, advise=False,
                lease_block=self.host.lease_block, lease_layout=self.host.lease_layout,
            )
        except BaseException:
            _close(self.host, self.slabs)
            raise
        self.planned = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.host_rows = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest = {
            n: torch.zeros((TOP_K,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda") for n in self.names
        }
        self.segments = expert_row_segments([(self.slabs[0][n], self.dest[n]) for n in self.names])
        self.dest_slots = torch.arange(TOP_K, dtype=torch.int32, device="cuda")

    def close(self):
        _close(self.host, self.slabs)

    def plan(self, experts):
        self.planned.fill_(-1)
        self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1)
        self.routes[: len(experts)] = torch.tensor(experts, dtype=torch.int64)

    def make_backend(self, row=0, poll_bound=64):
        """The real D7 orchestration (``Exl3RamMissRowBackend``), built once and reused across warm-up and
        capture -- exactly how ``post`` is used in production (constructed at attach time, replayed every
        step). Building a fresh one per call, inside a capture, allocates tensors mid-capture and is not
        what D7 does; T8's mutant is applied to this class, so T8 must run through it, unmodified, to test it."""
        from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissRowBackend

        return Exl3RamMissRowBackend(
            segments={0: self.segments},
            host_row_map=self.slot_map[0].to("cuda"),
            device_side=self.dev,
            row=row,
            next_row=-1,
            capacity=TOP_K,
            two_phase=True,
            poll_bound=poll_bound,
        )

    def make_plan(self):
        from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

        return ExpertRowPlan(expert_ids=self.planned[:TOP_K], slots=self.dest_slots, count=self.count)

    def until(self, predicate, timeout_s=10.0):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if predicate():
                return True
            time.sleep(0.002)
        return False


@pytest.fixture
def service(tmp_path):
    s = Service(tmp_path)
    try:
        yield s
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------------------------
# T8. Raw-graph node/edge enumeration, copied in method (not code) from cuda_graph_dedup_mixin.graph_signature:
# cuGraphGetNodes for the node list, cuGraphKernelNodeGetParams(node).func for identity, cuGraphGetEdges for the
# dependency edges. A stage's kernel identity is learned once, from a graph that captures only that one call.
# ---------------------------------------------------------------------------------------------------------------
def _kernel_func(fn):
    """Capture exactly one call of ``fn`` in its own graph and return its sole kernel node's ``func`` handle.

    ``fn`` must issue exactly one kernel launch when called (true of every ``Exl3RamMissDevice`` stage method
    and of ``copy_expert_row_segments_gpu``); more than one is a harness bug, caught by the assertion below
    rather than silently mis-tagging a stage.
    """
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=stream):
        fn()
    raw = g.raw_cuda_graph()
    _, num_nodes = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw, 0))
    nodes, _ = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw, num_nodes))
    kernel_funcs = []
    for node in nodes:
        node_type = checkCudaErrors(cuda_drv.cuGraphNodeGetType(node))
        if node_type == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            params = checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(node))
            kernel_funcs.append(int(params.func))
    assert len(kernel_funcs) == 1, f"{fn} captured {len(kernel_funcs)} kernel nodes, expected exactly 1"
    del g
    return kernel_funcs[0]


def _graph_kernel_chain(raw_graph):
    """The captured graph's nodes, in direct-edge order, as ``(node_type, func_or_None)`` pairs.

    Walks direct edges from the unique in-degree-0 node, asserting the graph is a simple path over *every*
    node (not just the kernel ones): a fork anywhere (an extra predecessor or successor, e.g. an event node
    inserted by a second-stream capture) fails here rather than being silently skipped. ``func`` is the
    kernel's identity for a KERNEL node and ``None`` otherwise (``post``'s planned-buffer refresh is a
    device-to-device MEMCPY node, not a kernel launch, and is expected to lead the chain).
    """
    _, num_nodes = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw_graph, 0))
    nodes, _ = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw_graph, num_nodes))
    node_list = list(nodes)
    node_index = {int(n): i for i, n in enumerate(node_list)}

    _, _, _, num_edges = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw_graph, 0))
    from_nodes, to_nodes, _, _ = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw_graph, num_edges))
    children = [[] for _ in node_list]
    parents = [[] for _ in node_list]
    for src, dst in zip(from_nodes, to_nodes):
        si, di = node_index[int(src)], node_index[int(dst)]
        children[si].append(di)
        parents[di].append(si)

    n = len(node_list)
    assert len(from_nodes) == n - 1, f"{n} nodes but {len(from_nodes)} edges: not a simple chain"
    roots = [i for i in range(n) if not parents[i]]
    assert len(roots) == 1, f"expected one root, found {len(roots)}: the graph forks"
    order = [roots[0]]
    while len(order) < n:
        cur = order[-1]
        assert len(children[cur]) == 1, f"node {cur} has {len(children[cur])} successors: the graph forks"
        nxt = children[cur][0]
        assert len(parents[nxt]) == 1, f"node {nxt} has {len(parents[nxt])} predecessors: the graph forks"
        order.append(nxt)

    typed = []
    for idx in order:
        node_type = checkCudaErrors(cuda_drv.cuGraphNodeGetType(node_list[idx]))
        if node_type == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            params = checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(node_list[idx]))
            typed.append((node_type, int(params.func)))
        else:
            typed.append((node_type, None))
    return typed, n, len(from_nodes)


def _assert_all_nodes_ordered_against(raw_graph, target_func):
    """Every node in ``raw_graph`` is comparable to the node whose kernel identity is ``target_func``:
    an ancestor of it, that node itself, or a descendant of it. Nothing is incomparable -- on neither a
    directed path to nor from it, i.e. able to run concurrently with it on a fork.

    This is the property T8's second test wants ("F precedes the fused consumer") without needing to
    name every fused-compute node individually: instead of asserting F is the *direct* predecessor of
    some specific downstream node (the internal shape of the fused kernel is not this test's business),
    it asserts nothing in the whole captured graph can race F. Returns the descendant count, so the
    caller can additionally assert the graph did not degenerate to nothing running after F at all.
    """
    _, num_nodes = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw_graph, 0))
    nodes, _ = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw_graph, num_nodes))
    node_list = list(nodes)
    node_index = {int(n): i for i, n in enumerate(node_list)}

    _, _, _, num_edges = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw_graph, 0))
    from_nodes, to_nodes, _, _ = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw_graph, num_edges))
    children = [[] for _ in node_list]
    parents = [[] for _ in node_list]
    for src, dst in zip(from_nodes, to_nodes):
        si, di = node_index[int(src)], node_index[int(dst)]
        children[si].append(di)
        parents[di].append(si)

    target_idx = None
    for i, node in enumerate(node_list):
        node_type = checkCudaErrors(cuda_drv.cuGraphNodeGetType(node))
        if node_type != cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        params = checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(node))
        if int(params.func) == target_func:
            assert target_idx is None, "the target kernel identity appears in more than one node"
            target_idx = i
    assert target_idx is not None, "the target kernel identity was not found in the captured graph"

    def _reach(start, adjacency):
        seen, stack = set(), list(adjacency[start])
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adjacency[cur])
        return seen

    descendants = _reach(target_idx, children)
    ancestors = _reach(target_idx, parents)
    comparable = ancestors | {target_idx} | descendants
    incomparable = [i for i in range(len(node_list)) if i not in comparable]
    assert not incomparable, f"{len(incomparable)} node(s) are neither before nor after the target kernel"
    return len(descendants)


@pytest.mark.skipif(cuda_drv is None, reason="needs cuda-python (cuda.bindings.driver)")
class TestGraphTopology:
    def test_the_captured_two_phase_chain_is_the_linear_post_w1_c1_a1_w2_c2_a2_f(self, service):
        """T8: cudaGraphGetEdges shows exactly the chain D7 builds, checked rather than assumed."""
        s = service
        s.plan([3, 5])  # both resident up front, so stage 1 claims something real, not just an empty commit
        s.dev.post(0, s.planned, s.count, s.routes, -1)
        s.dev.wait(0, s.planned, s.count, s.host_rows, s.keep, s.ram_miss)  # warms residency via the batched wait
        _cuda_ready()
        assert s.dev.go_count.item() == 2, "both experts must be resident before stage 1 has anything to claim"

        # Learn each stage's kernel identity from a graph that captures only that one call.
        # D sums the stages' copy counts into go_total, the count a DIRECT residency commit reads.
        expected_names = ["post", "W1", "C1", "A1", "W2", "C2", "A2", "F", "D"]
        expected_funcs = {
            "post": _kernel_func(lambda: s.dev.post(0, s.planned, s.count, s.routes, -1)),
            "W1": _kernel_func(lambda: s.dev.hit_wait(0, s.planned, s.count, s.dest_slots, 64)),
            "C1": _kernel_func(
                lambda: copy_expert_row_segments_gpu(s.segments, s.dev.host_rows_1, s.dev.dst_slots_1, s.dev.go_1)
            ),
            "A1": _kernel_func(lambda: s.dev.stage_ack(1)),
            "W2": _kernel_func(lambda: s.dev.rest_wait(0, s.planned, s.count, s.dest_slots, s.ram_miss)),
            "C2": _kernel_func(
                lambda: copy_expert_row_segments_gpu(s.segments, s.dev.host_rows_2, s.dev.dst_slots_2, s.dev.go_2)
            ),
            "A2": _kernel_func(lambda: s.dev.stage_ack(2)),
            "F": _kernel_func(lambda: s.dev.finalize(s.count, s.keep)),
            "D": _kernel_func(lambda: torch.add(s.dev.go_1, s.dev.go_2, out=s.dev.go_total)),
        }
        # C1 and C2 are the same kernel (copy_expert_row_segments_gpu_kernel): confirm that identity assumption
        # rather than let it silently make the chain assertion below vacuous.
        assert expected_funcs["C1"] == expected_funcs["C2"], "stage 1 and stage 2 use different copy kernels"

        s.plan([3, 5])
        backend = s.make_backend()
        plan = s.make_plan()
        with torch.cuda.stream(torch.cuda.Stream()):
            backend.post(0, plan)  # warm-up outside capture: JIT, first-touch, context
        _cuda_ready()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            backend.post(0, plan)
        raw = graph.raw_cuda_graph()
        typed, node_count, edge_count = _graph_kernel_chain(raw)
        del graph

        # post's _stage_planned is a device-to-device tensor copy (a MEMCPY node, not a kernel launch); it leads
        # the chain every replay refreshes it from the plan. Everything after it must be the 9-kernel chain.
        assert node_count == 10 and edge_count == 9, (node_count, edge_count)
        assert typed[0][0] == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY, typed[0]
        kernel_nodes = typed[1:]
        assert all(t == cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL for t, _ in kernel_nodes), typed
        funcs = [f for _, f in kernel_nodes]

        # Resolve each captured node to a stage name by its position among same-identity stages (C1 then C2).
        remaining = dict(expected_funcs)
        resolved = []
        for func in funcs:
            match = None
            for name in expected_names:
                if name in resolved:
                    continue
                if remaining.get(name) == func:
                    match = name
                    break
            assert match is not None, (func, resolved, remaining)
            resolved.append(match)
        assert resolved == expected_names, resolved

    @NEEDS_EXL3_SRC
    def test_finalize_precedes_the_fused_moe_consumer(self, tmp_path):
        """T8's "-> fused" clause: the checklist row is ``post -> ... -> F -> fused``, not ``... -> F``.

        The sibling test above captures ``backend.post`` alone and pins the 10-node RAM-miss chain
        exactly; it says nothing about the fused MoE kernel that reads the rows F's success gates,
        because that kernel is not in that capture. This test captures the real production apply
        (``Exl3MoEMethod._apply_graph``, the same shape T9 uses) and checks that nothing in that larger
        graph is incomparable with F -- neither an ancestor nor a descendant of it, i.e. able to run
        concurrently with it on a fork. Stream ordering gives this in practice today, which is exactly
        why a refactor that broke it would go unnoticed without this check.
        """
        from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
        from sglang.test.dsv41_fake_exl3 import write_fake_exl3

        write_fake_exl3(str(tmp_path), num_layers=1, num_experts=F_EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
        layer, streamer, service, checks = _fused_layer(tmp_path, two_phase=True)
        try:
            gen = torch.Generator(device="cpu").manual_seed(7)
            x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.softmax(torch.randn((1, F_TOP_K), generator=gen), -1).cuda()
            ids = torch.tensor([[0, 3, 5, 1, 7, 6]], device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)  # warm-up: JIT, residency
            _cuda_ready()

            # F's kernel identity, learned from the exact device_side instance the capture below drives
            # (not a separate harness): an isolated single-op capture of the same finalize() call.
            dev = streamer.row_backend.device_side
            probe_count = torch.zeros(1, dtype=torch.int32, device="cuda")  # allocated outside the probe: a fresh
            # allocation inside it would itself add a node (an allocator fill/memset), miscounting the kernel probe.
            f_func = _kernel_func(lambda: dev.finalize(probe_count, streamer.row_backend.keep))

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
            raw = graph.raw_cuda_graph()
            descendant_count = _assert_all_nodes_ordered_against(raw, f_func)
            del graph
            assert descendant_count > 0, "nothing in the captured graph runs after F: the fused consumer was not captured downstream of it"
            assert service.host.fatal_seq() == 0, service.host.counters()
        finally:
            service.shutdown()


# ---------------------------------------------------------------------------------------------------------------
# T9. The full fused-MoE harness, copied from test_exl3_ram_miss_graph_gpu.py's _layers / _source_rows, extended
# with a two_phase switch so the same shape builds both M1 (leases on, two-phase off) and M2 (two-phase on).
# ---------------------------------------------------------------------------------------------------------------
HIDDEN, INTER, F_EXPERTS, F_TOP_K = 1024, 512, 16, 6


def _source_rows(tmp_path, layer_id=0):
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, layer_id, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    source = {name: torch.empty((F_EXPERTS,) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    Exl3ShardRowSource.for_layer(layout, layer_id, fmt.segment_map(), direct=False).read(
        torch.arange(F_EXPERTS, dtype=torch.long), source
    )
    return source


def _fused_layer(tmp_path, *, two_phase: bool, timeout_ms=2000):
    """One streamed layer with the real service attached, leases on and ``two_phase`` as given.

    Copied in shape from test_exl3_ram_miss_graph_gpu.py's ``_layers`` (single-layer case), with the
    two-phase override added. The service is a process-wide singleton, so a caller must
    ``service.shutdown()`` this arm before building the other.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer

    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    with (
        envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
        envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
        envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(timeout_ms),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(two_phase),
    ):
        layer = torch.nn.Module()
        layer.layer_id = 0
        layer.top_k = F_TOP_K
        fmt = Exl3ExpertFormat(layout, 0, direct=False)
        streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
        layer._nvfp4_expert_streamer = streamer
        ExpertPinnedHostCache(streamer, 8, **fmt.pinned_tier_options(layer))
        hot = ExpertHotCache(streamer, 3, scratch_rows=F_TOP_K)
        hot.reassign([0, 1, 2])
        streamer.enable_graph_gather(F_TOP_K)
        checks = []
        manager = type(
            "M",
            (),
            {
                "register_fail_stop_check": lambda self, f: checks.append(f),
                "add_residency_listener": lambda self, listener: listener(0, list(hot.slot_to_expert)),
            },
        )()
        streamer.format.attach_hot_cache_manager(manager, streamer)
    service = service_module.Exl3RamMissService.get()
    assert service.lease_mode is True and service.two_phase is two_phase
    return layer, streamer, service, checks


@NEEDS_EXL3_SRC
class TestOutputParity:
    def test_two_phase_output_is_bitwise_equal_to_m1_eager_and_graph(self, tmp_path):
        """T9: destination rows byte-equal and fused output bitwise equal against M1, eager and graph.

        M1 = Task 5's lease-mode batched arm (two_phase=False); M2 = this mechanism (two_phase=True).
        Fixed seed, fixed routes: [0, 3, 5, 1, 7, 6] (mixed against the [0, 1, 2] hot set) and
        [9, 10, 11, 0, 1, 12] (12, 11, 10, 9 never touched before this test; 0, 1 are hot-set hits).

        Two more properties are pinned beyond output equality, because M1/M2 agreeing on output does
        not by itself prove stage 1 is doing anything (see the method below for why an exact go_1
        count is not safe to assert): a liveness check that stage 1 does claim a hit lane at least once
        given enough identical warm requests, and a per-lane check -- under an injected read delay that
        makes it deterministic rather than a second race -- that a lane whose expert has never been
        resident cannot be claimed by stage 1 while its read is still outstanding.
        """
        from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
        from sglang.test.dsv41_fake_exl3 import write_fake_exl3

        write_fake_exl3(str(tmp_path), num_layers=1, num_experts=F_EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
        source = _source_rows(tmp_path)
        gen = torch.Generator(device="cpu").manual_seed(3)
        x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
        weights = torch.softmax(torch.randn((1, F_TOP_K), generator=gen), -1).cuda()
        routes = ([0, 3, 5, 1, 7, 6], [9, 10, 11, 0, 1, 12])

        results = {}
        for arm, two_phase in (("M1", False), ("M2", True)):
            layer, streamer, service, checks = _fused_layer(tmp_path, two_phase=two_phase)
            try:
                ids = torch.tensor([routes[0]], device="cuda", dtype=torch.int32)
                Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)  # eager warm-up
                dev = streamer.row_backend.device_side

                if two_phase:
                    # Liveness, not an exact count. Stage 1's hit/miss split races real host-service
                    # scheduling latency against the poll bound (checklist O4: neither is measured), so
                    # a single call's go_1 is not safe to pin to a specific number -- verified
                    # empirically against this exact route and service: identical inputs gave go_1 in
                    # {0, 1, 3, 4} across independent runs on the same box. What a single flaky number
                    # cannot state, and what this loop does instead, is the actual regression T9's
                    # mutant papered over without a test statement: given enough attempts against warm
                    # residency, stage 1 eventually claims a hit lane. A stage 1 that permanently
                    # returns go_1 == 0 (a silent no-op) fails this within its deadline; a stage 1 that
                    # sometimes does and sometimes does not (today's real behaviour) passes.
                    live = False
                    deadline = time.perf_counter() + 10.0
                    while time.perf_counter() < deadline:
                        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
                        _cuda_ready()
                        if int(dev.go_1.item()) > 0:
                            live = True
                            break
                    assert live, "stage 1 never claimed a single hit lane across repeated identical warm requests"

                eager_outs, graph_outs, gathers = [], [], []
                for route in routes:
                    ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
                    inject_this_call = two_phase and route == routes[1]
                    if inject_this_call:
                        # Force this call's reads to take far longer than stage 1's poll bound (tens of
                        # us at the default poll_bound), so the claim check below is deterministic, not a
                        # race. Without an injected delay this is NOT a safe property to assert: verified
                        # empirically that a lane whose expert was never resident before (12, here) can
                        # still show claimed == 1 on its very first read, because stage 1 does not
                        # distinguish "resident before this request" from "published while I was still
                        # polling" (D1's own words) -- and this test's tiny fake checkpoint reads fast
                        # enough that the miss's publish sometimes lands inside the poll window anyway.
                        service.host.inject(delay_s=0.2)
                    try:
                        eager_outs.append(Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0).float().clone())
                    finally:
                        if inject_this_call:
                            service.host.inject(delay_s=0.0)
                    assert streamer.row_backend.keep.item() == 1.0, (arm, route, service.host.counters())
                    for check in checks:
                        check()

                    if inject_this_call:
                        # Now deterministic: experts 9, 10, 11, 12 were never resident anywhere before
                        # this exact call (route[0]'s two experts beyond the hot set are 3, 5, 6, 7; only
                        # route[1] ever names 9, 10, 11, 12), and the injected delay above guarantees
                        # their reads outlast stage 1's poll, so none of them can be claimed by stage 1.
                        planned = streamer.row_backend.planned.tolist()
                        claimed = dev.claimed.tolist()
                        for expert in (9, 10, 11, 12):
                            lane = planned.index(expert)
                            assert claimed[lane] == 0, (arm, route, expert, lane, planned, claimed)

                    remap, tensors = streamer.gather(ids)
                    slots = remap.reshape(-1).tolist()
                    for k, expert in enumerate(route):
                        for name, rows in tensors.items():
                            assert torch.equal(rows[slots[k]].cpu(), source[name][expert]), (arm, route, k, expert, name)
                    gathers.append((remap.clone(), {n: t.clone() for n, t in tensors.items()}))

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
                for route in routes:
                    ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
                    graph.replay()
                    _cuda_ready()
                    assert streamer.row_backend.keep.item() == 1.0, (arm, route, service.host.counters())
                    graph_outs.append(out.float().clone())
                results[arm] = {"eager": eager_outs, "graph": graph_outs, "gathers": gathers}
                assert service.host.fatal_seq() == 0, (arm, service.host.counters())
            finally:
                service.shutdown()

        for i, route in enumerate(routes):
            assert torch.equal(results["M1"]["eager"][i], results["M2"]["eager"][i]), (route, "eager")
            assert torch.equal(results["M1"]["graph"][i], results["M2"]["graph"][i]), (route, "graph")
            assert torch.equal(results["M1"]["eager"][i], results["M1"]["graph"][i]), (route, "M1 eager vs graph")
            assert torch.equal(results["M2"]["eager"][i], results["M2"]["graph"][i]), (route, "M2 eager vs graph")
            remap1, tensors1 = results["M1"]["gathers"][i]
            remap2, tensors2 = results["M2"]["gathers"][i]
            slots1, slots2 = remap1.reshape(-1).tolist(), remap2.reshape(-1).tolist()
            for k in range(len(route)):
                for name in tensors1:
                    assert torch.equal(
                        tensors1[name][slots1[k]].view(torch.uint8), tensors2[name][slots2[k]].view(torch.uint8)
                    ), (route, k, name)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
