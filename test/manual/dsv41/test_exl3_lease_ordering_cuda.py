"""The orderings the slot-map protocol rests on, on a real GPU against the real service (LEASE_PROTOCOL.md, "Victims"
and "Copy engine"): a record's victims are never slots its own delayed readers use, copy-engine hits typed while the
service is paused are waited for in CW, the captured chain waits in CW armed and unarmed, a first armed replay
completes, and a kernel's first load under an armed copy wait.

Each ordering test names the mutant it must fail under. Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at
the tree under test.
"""

import json
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

try:
    from cuda.bindings import driver as cuda_drv  # noqa: E402
except ImportError:  # pragma: no cover - only when cuda-python is missing
    cuda_drv = None

from lease_chain_rig import CAPACITY, STAGING, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe.expert_stream_transport import device_module_with_hooks  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

CW_DELAY_CYCLES = 600_000_000  # ~250 ms at the 5090's clock: long after the DMA of a copied lane has completed
SM_READ_DELAY_NS = 100_000_000
MAPPABLE = CAPACITY - STAGING


def _resident(c, experts, row=0):
    c.plan(experts, row)
    c.gather(row)
    torch.cuda.synchronize()
    assert c.handled(), c.host.counters()


class _RewriteOnUnmap:
    """A watcher thread: the instant the host's map drops one of ``experts``, write 0xAB over the slot it held. The
    host forgetting a slot is the earliest moment anything may rewrite it."""

    def __init__(self, c, experts, row=0):
        self.c, self.row = c, row
        self.slots = {e: c.host.mapping(row)[e] for e in experts}
        assert all(slot >= 0 for slot in self.slots.values()), self.slots
        self.unmapped, self.at_s = None, None
        self.t0 = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while time.perf_counter() - self.t0 < 10.0:
            mapping = self.c.host.mapping(self.row)
            gone = [e for e, slot in self.slots.items() if mapping[e] != slot]
            if gone:
                self.at_s = time.perf_counter() - self.t0
                self.unmapped = gone
                for e in gone:
                    for n in self.c.names:  # an int index is a view of the slab; a list index would fill a copy
                        self.c.slabs[self.row][n][self.slots[e]].view(torch.uint8).fill_(0xAB)
                return

    def join(self):
        self.thread.join()
        return self.unmapped, self.at_s


def _victim_under_a_delayed_reader(c, armed, **hooks):
    """Fill every mappable slot (0..5 the oldest, 6 and 7 the newest), then post five hits and one miss with the hits'
    reader delayed: the miss evicts the oldest slot the record does not route, 5, while the readers still wait."""
    _resident(c, list(range(TOP_K)))
    _resident(c, list(range(TOP_K, MAPPABLE)))
    if armed:
        c.host.arm_copy_engine()
    experts = [0, 1, 2, 3, 4, 9]
    watcher = _RewriteOnUnmap(c, range(MAPPABLE))
    c.plan(experts)
    c.chain(captured=armed, **hooks)
    snapshot = c.snapshot()
    torch.cuda.synchronize()
    unmapped, at_s = watcher.join()
    assert unmapped == [5], (unmapped, c.host.counters())
    c.check(experts, snapshot)  # 0xAB here: the victim was a slot a delayed reader of this record used
    want = [LaneKind.HIT_COPY if armed else LaneKind.HIT_SM] * 5 + [LaneKind.MISS_GPU]
    assert c.kinds(len(experts)) == want
    assert c.handled()
    return at_s


@pytest.mark.parametrize("where", ["before_c1", "before_cw"])
def test_a_records_victim_is_never_a_slot_its_own_delayed_readers_use(tmp_path, where):
    """A ~250 ms device sleep delays the record's reader of its hit slots: C1 (HIT_SM, engine unarmed), or CW's SM
    reads of the small tensors (HIT_COPY, their trellis already DMA'd). The watcher writes 0xAB over the record's victim
    the instant the host unmaps it, long before the reader runs; every destination row must still hold its checkpoint
    bytes. Mutant: take_victim_locked ignores the record's routes (it then evicts expert 0, a delayed hit)."""
    armed = where == "before_cw"
    c = Chain(tmp_path, copy_engine=armed, sm_small_copies=armed)
    try:
        at_s = _victim_under_a_delayed_reader(c, armed, **{where: lambda: torch.cuda._sleep(CW_DELAY_CYCLES)})
        assert at_s < 0.1, f"the victim was unmapped {at_s:.3f} s in: after the delayed reader, so it proved nothing"
    finally:
        c.close()


def test_a_records_victim_is_never_a_slot_cws_late_sm_reads_use(tmp_path):
    """The hook starts every CW warp but the first ~100 ms late on its small-tensor reads (warp 1 reads the 1024-byte
    rows' upper half), so the record's victim is unmapped and rewritten while its hits' SM reads are still pending.
    Mutant: as above, the victim ignores the routes -- red on 0xAB."""
    c = Chain(tmp_path, copy_engine=True, sm_small_copies=True)
    try:
        c.dev._module = device_module_with_hooks([f"EXL3_RAM_MISS_TEST_CW_SM_READ_DELAY_NS={SM_READ_DELAY_NS}"])
        c.plan([])
        c.chain(captured=True)  # compile and load the hooked build before the timed request
        torch.cuda.synchronize()
        at_s = _victim_under_a_delayed_reader(c, True)
        assert at_s < SM_READ_DELAY_NS / 2e9, f"the victim was unmapped {at_s:.3f} s in: the delay did not cover it"
    finally:
        c.close()


def test_copy_engine_hits_typed_while_the_service_is_paused_are_waited_for_in_cw(tmp_path):
    """The post types HIT_COPY from the device map alone, so lanes posted while the service is paused are typed before
    any copy job exists. C1 must leave them alone and CW must wait for the engine, which copies them on resume.
    Mutant: CW builds its mask from anything but lane_kind -- it waits for nothing and reads stale destination rows."""
    c = Chain(tmp_path, copy_engine=True)
    try:
        experts = [3, 4, 5]
        _resident(c, experts)
        c.host.arm_copy_engine()
        for _ in range(5):
            c.plan(experts)
            c.host.pause(5.0)
            c.chain(captured=True)
            snapshot = c.snapshot()
            time.sleep(0.01)
            c.host.resume()
            torch.cuda.synchronize()
            assert c.kinds(3) == [LaneKind.HIT_COPY] * 3
            assert int(c.dev.go_1.item()) == 0
            assert int(c.dev.ce_mask.item()) & 0xFF == 0b111, "CW did not wait for the three HIT_COPY lanes"
            c.check(experts, snapshot)
            assert c.handled(), c.host.counters()
    finally:
        c.close()


def _node_signature(node):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import checkCudaErrors

    node_type = checkCudaErrors(cuda_drv.cuGraphNodeGetType(node))
    if node_type != cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        return (int(node_type), None)
    return (int(node_type), int(checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(node)).func))


def _chain_nodes(raw):
    """The graph's nodes along its one path, asserting it is a simple chain over every node."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import checkCudaErrors

    _, num_nodes = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw, 0))
    nodes, _ = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw, num_nodes))
    nodes = list(nodes)
    index = {int(n): i for i, n in enumerate(nodes)}
    _, _, _, num_edges = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw, 0))
    sources, targets, _, _ = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw, num_edges))
    children = [[] for _ in nodes]
    parents = [[] for _ in nodes]
    for a, b in zip(sources, targets):
        children[index[int(a)]].append(index[int(b)])
        parents[index[int(b)]].append(index[int(a)])
    assert len(sources) == len(nodes) - 1, f"{len(nodes)} nodes but {len(sources)} edges: not a simple chain"
    order = [next(i for i in range(len(nodes)) if not parents[i])]
    while len(order) < len(nodes):
        assert len(children[order[-1]]) == 1, "the graph forks"
        order.append(children[order[-1]][0])
    return [_node_signature(nodes[i]) for i in order]


def _captured_nodes(fn):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph, _chain_nodes(graph.raw_cuda_graph())


@pytest.mark.skipif(cuda_drv is None, reason="needs cuda-python (cuda.bindings.driver)")
def test_the_captured_chain_waits_in_cw_and_replays_right_armed_and_unarmed(tmp_path):
    """The captured gather ends CW -> stream wait on the gate (a memory-op node, no kernel spinning) -> CC; its replays
    deliver the planned rows unarmed (HIT_SM through C1) and armed (HIT_COPY through the engine), and the engine
    copies exactly when armed."""
    c = Chain(tmp_path, copy_engine=True)
    try:
        c.plan([0, 1])
        graph, chain = _captured_nodes(c.gather)
        torch.cuda.synchronize()
        assert c.handled()
        _, tail = _captured_nodes(lambda: c.dev.copy_wait(c.plans[0].count, c.plans[0].slots))
        assert len(tail) == 3 and tail[1][0] == int(cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP), tail
        assert chain[-3:] == tail, chain
        rng = random.Random(11)
        for armed in (False, True, False, True):
            c.host.arm_copy_engine(armed)
            jobs = c.host.counters()["copy_jobs"]
            for _ in range(8):
                experts = rng.sample(range(MAPPABLE + 2), TOP_K)  # mostly resident after the first replays
                c.plan(experts)
                graph.replay()
                snapshot = c.snapshot()
                torch.cuda.synchronize()
                c.check(experts, snapshot)
                assert c.handled(), (armed, c.host.counters())
            assert (c.host.counters()["copy_jobs"] > jobs) == armed, (armed, c.host.counters())
    finally:
        c.close()


def _first_armed_replay(c, *, before, after, unarmed_first):
    """Capture ``before`` filler kernels, the gather, ``after`` fillers (the decode graph's size around one layer);
    optionally replay once unarmed; then arm and replay with every lane a hit. Returns the armed replay's seconds."""
    _resident(c, [0, 1, 2])
    x = torch.zeros(1, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        c.gather()  # eager warm-up: loads every kernel; an eager post never copies
        x.add_(1)
    torch.cuda.synchronize()
    assert c.handled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(before):
            x.add_(1)
        c.gather()
        for _ in range(after):
            x.add_(1)
    if unarmed_first:
        c.plan([1, 2, 0])
        graph.replay()
        torch.cuda.synchronize()
        assert c.handled()
    c.host.arm_copy_engine()
    jobs = c.host.counters()["copy_jobs"]
    experts = [2, 0, 1]
    c.plan(experts)
    t0 = time.perf_counter()
    graph.replay()
    snapshot = c.snapshot()
    torch.cuda.synchronize()
    replay_s = time.perf_counter() - t0
    assert c.host.counters()["copy_jobs"] == jobs + 1, c.host.counters()
    c.check(experts, snapshot)
    assert c.handled(), c.host.counters()
    return replay_s


@pytest.mark.parametrize(
    ("before", "after", "unarmed_first"), [(0, 0, False), (3000, 0, False), (3000, 3000, True)],
    ids=["bare", "long_head", "long_tail_after_an_unarmed_replay"],
)
def test_the_first_armed_replay_of_a_copy_engine_graph_completes(tmp_path, before, after, unarmed_first):
    """Both smokes that failed at startup (abba1-on, diag arm1-on) failed on a decode step that ran armed early in the
    server's life. A graph's first armed replay completes well inside the copy-wait timeout when nothing long follows
    the chain in its first launch, or when it has run once unarmed, which the service guarantees by arming only after
    COPY_ENGINE_ARM_DECODES decode forwards. Mutant: the copy thread never stores CopyDone -- CC traps."""
    c = Chain(tmp_path, copy_engine=True)
    try:
        replay_s = _first_armed_replay(c, before=before, after=after, unarmed_first=unarmed_first)
        assert replay_s < 1.0, replay_s
    finally:
        c.close()


_LONG_TAIL_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
from test_exl3_lease_ordering_cuda import _first_armed_replay
c = Chain(sys.argv[1], copy_engine=True)
_first_armed_replay(c, before=0, after=3000, unarmed_first=False)
print("reached", flush=True)
"""


def test_a_graphs_first_launch_armed_with_a_long_tail_aborts_at_the_copy_wait_timeout(tmp_path):
    """The hazard the arming delay exists for, pinned: a graph launched for the first time armed, with ~3000 kernels
    after the chain, never gets its copy (the same at 62afc40d7d, where it failed closed after 2 s). 300 or 1000
    kernels after, or 3000 before, complete; so does any graph that ran once unarmed. When this test goes red, a
    change made the first armed launch safe: then the arming delay may be revisited."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_LONG_TAIL_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=300,
    )
    assert "reached" not in result.stdout, result.stdout
    assert "FATAL" in result.stderr and "a copy wait held the decode stream" in result.stderr, result.stderr[-4000:]


_LAZY_SCRIPT = """
import json, sys, threading, time, uuid
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
from sglang.kernels.jit.utils import load_jit

tmp = Path(sys.argv[1])
salt = uuid.uuid4().hex[:12]
source = tmp / f"lazy_{salt}.cuh"
source.write_text(
    "#include <sgl_kernel/tensor.h>\\n#include <sgl_kernel/utils.h>\\n#include <sgl_kernel/utils.cuh>\\n"
    "#include <tvm/ffi/container/tensor.h>\\n"
    f"__global__ void lazy_{salt}_kernel(float* x) {{ x[threadIdx.x] += 1.0f; }}\\n"
    "namespace sglang {\\n"
    f"void lazy_{salt}(tvm::ffi::TensorView x) {{\\n"
    "  const auto stream = host::LaunchKernel::resolve_device(x.device());\\n"
    f"  host::LaunchKernel(1, 32, stream)(lazy_{salt}_kernel, static_cast<float*>(x.data_ptr()));\\n"
    "}\\n}\\n"
)
jit = load_jit(f"lazy_{salt}", cuda_files=[str(source)], cuda_wrappers=[(f"lazy_{salt}", f"lazy_{salt}")])
kernel = getattr(jit, f"lazy_{salt}")
x = torch.zeros(32, dtype=torch.float32, device="cuda")
side = torch.cuda.Stream()
(tmp / "svc").mkdir(exist_ok=True)
c = Chain(tmp / "svc", copy_engine=True, copy_wait_ms=1000)
src = torch.empty(256 << 20, dtype=torch.uint8).pin_memory()
dst = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
c.plan([0, 1, 2])
c.gather()
c.snapshot()
torch.cuda.synchronize()
assert c.handled()
c.host.copy_engine_ballast(dst, src)
c.host.arm_copy_engine()
c.host.pause(5.0)
c.chain(captured=True)
resume = threading.Timer(0.005, c.host.resume)
resume.start()
t0 = time.perf_counter()
with torch.cuda.stream(side):
    kernel(x)  # the kernel's first launch
launch_ms = (time.perf_counter() - t0) * 1e3
torch.cuda.synchronize()
resume.join()
jobs = c.host.counters()["copy_jobs"]
print(json.dumps({"launch_ms": round(launch_ms, 2), "copy_jobs": jobs, "x": float(x[0].item())}), flush=True)
c.host.copy_engine_ballast(None, None)
c.close()
"""


@pytest.mark.parametrize("mode", ["LAZY", "EAGER"])
def test_a_kernels_first_load_while_an_armed_copy_wait_holds_the_stream(tmp_path, mode):
    """The soak's fail-stop, reduced: a JIT kernel, built and loaded but never launched, is launched for the first time
    while the copy thread's copies are being issued and the chain waits for them. Under LAZY the launch loads the
    module, which waits for the device, which waits on the gate for copies that cannot be issued meanwhile: the
    watchdog aborts at the copy-wait timeout. Under EAGER the kernel loaded with its library and everything completes.
    This is why the service refuses the copy engine without EAGER. One process per mode: CUDA reads
    CUDA_MODULE_LOADING once."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_LAZY_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        env=dict(os.environ, CUDA_MODULE_LOADING=mode), capture_output=True, text=True, timeout=900,
    )
    if mode == "EAGER":
        assert result.returncode == 0, result.stderr[-4000:]
        row = json.loads([line for line in result.stdout.splitlines() if line.startswith("{")][-1])
        assert row["launch_ms"] < 100 and row["x"] == 1.0, row
    else:
        assert result.returncode != 0, result.stdout[-2000:]
        assert "FATAL" in result.stderr and "a copy wait held the decode stream" in result.stderr, result.stderr[-4000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
