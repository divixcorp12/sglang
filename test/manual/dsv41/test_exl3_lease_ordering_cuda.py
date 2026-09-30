"""The orderings the minimal protocol rests on, on a real GPU against the real service (LEASE_PROTOCOL.md, "Done" and
"Copy engine"): Done covers every read of a leased slot, S hands W1's unclaimed COPYING lanes to CW, the captured chain
waits in CW armed and unarmed, a first armed replay completes, and a kernel's first load under an armed copy wait.

Each ordering test names the mutant it must fail under. Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at
the tree under test.
"""

import json
import os
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

try:
    from cuda.bindings import driver as cuda_drv  # noqa: E402
except ImportError:  # pragma: no cover - only when cuda-python is missing
    cuda_drv = None

from lease_chain_rig import CAPACITY, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe.expert_stream_transport import device_module_with_hooks  # noqa: E402

CW_DELAY_CYCLES = 600_000_000  # ~250 ms at the 5090's clock: long after the DMA of a copied lane has completed
SM_READ_DELAY_NS = 100_000_000


def _resident(c, experts, row=0):
    c.plan(experts, row)
    c.gather(row)
    torch.cuda.synchronize()
    assert c.retired(), c.host.counters()


@pytest.mark.parametrize("where", ["before_c1", "before_cw"])
def test_a_slab_row_rewritten_the_moment_its_lease_is_released_never_reaches_the_destination(tmp_path, where):
    """A ~250 ms device sleep delays the chain's reader of the leased slots: C1 (hits READY, engine unarmed), or CW's
    SM reads of the small tensors (hits COPYING, their trellis already DMA'd). A watcher writes 0xAB over every leased
    slot the instant its lease drops; the destination must hold the checkpoint's bytes, so no lease dropped before
    Done. Mutants: retire a lease without Done == G (before_c1 red); hand a copied job back on its DMA alone, ignoring
    its SM entries (before_cw red)."""
    armed = where == "before_cw"
    c = Chain(tmp_path, copy_engine=armed, sm_small_copies=armed)
    try:
        experts = list(range(TOP_K))
        _resident(c, experts)
        if armed:
            c.host.arm_copy_engine()
        slots = [c.host.mapping(0)[e] for e in experts]
        copied = c.host.counters()["leases_copied"]
        c.plan(experts)
        c.chain(captured=armed, **{where: lambda: torch.cuda._sleep(CW_DELAY_CYCLES)})
        snapshot = c.snapshot()
        released_s = c.rewrite_on_release(slots)
        torch.cuda.synchronize()
        assert released_s is not None, c.host.counters()
        c.check(experts, snapshot)  # 0xAB here: a slot was rewritten under a read of it
        assert released_s > 0.1, f"the leases dropped {released_s:.3f} s after launch, before the delayed reader ran"
        assert (c.host.counters()["leases_copied"] - copied == TOP_K) == armed, c.host.counters()
    finally:
        c.close()


def test_every_sm_read_of_a_slot_happens_before_cws_done(tmp_path):
    """The hook starts every CW warp but the first ~100 ms late on its small-tensor reads (warp 1 reads the 1024-byte
    rows' upper half). A watcher writes 0xAB over each slot the instant its lease drops: a Done published before the
    late reads finish lets the sentinel into the destination. Mutants: store Done above the barrier that follows the
    reads, or above the reads -- each red on 0xAB."""
    c = Chain(tmp_path, copy_engine=True, sm_small_copies=True)
    try:
        experts = list(range(TOP_K))
        _resident(c, experts)
        c.host.arm_copy_engine()
        slots = [c.host.mapping(0)[e] for e in experts]
        c.dev._module = device_module_with_hooks([f"EXL3_RAM_MISS_TEST_CW_SM_READ_DELAY_NS={SM_READ_DELAY_NS}"])
        c.plan([])
        c.chain(captured=True)  # compile and load the hooked build before the timed request
        torch.cuda.synchronize()
        c.plan(experts)
        c.chain(captured=True)
        snapshot = c.snapshot()
        released_s = c.rewrite_on_release(slots)
        torch.cuda.synchronize()
        assert released_s is not None, c.host.counters()
        c.check(experts, snapshot)  # 0xAB here: Done was published before a read of the slot finished
        assert released_s > SM_READ_DELAY_NS / 2e9, f"the leases dropped {released_s:.3f} s in: the delay did not act"
    finally:
        c.close()


def test_s_hands_a_copying_lane_w1_did_not_claim_to_the_copy_wait(tmp_path):
    """The service is paused while the chain launches, so W1 (no budget) passes before any lane is published and
    claims nothing; on resume the lanes are published COPYING, and S must leave them to CW (copy_owned), which waits
    for the engine. Mutant: S treats a COPYING lane as not yet published -- it never admits it, and traps at its
    deadline once served."""
    c = Chain(tmp_path, copy_engine=True, hit_wait_ns=0)
    try:
        experts = [3, 4, 5]
        _resident(c, experts)
        c.host.arm_copy_engine()
        for _ in range(5):
            c.plan(experts)
            c.host.pause(5.0)
            c.chain(captured=True)
            snapshot = c.snapshot()
            time.sleep(0.01)  # W1 has long finished its one pass
            c.host.resume()
            torch.cuda.synchronize()
            assert c.dev.claimed[:3].tolist() == [0, 0, 0], "W1 claimed a lane published after it ran"
            assert int(c.dev.go_1.item()) == 0
            assert int(c.dev.ce_mask.item()) & 0xFF == 0b111, "CW did not wait for the three COPYING lanes"
            c.check(experts, snapshot)
            assert c.retired(), c.host.counters()
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
    deliver the planned rows unarmed (hits READY through C1) and armed (COPYING through the engine), and the engine
    copies exactly when armed."""
    c = Chain(tmp_path, copy_engine=True)
    try:
        c.plan([0, 1])
        graph, chain = _captured_nodes(c.gather)
        torch.cuda.synchronize()
        assert c.retired()
        _, tail = _captured_nodes(lambda: c.dev.copy_wait(c.plans[0].count))
        assert len(tail) == 3 and tail[1][0] == int(cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP), tail
        assert chain[-3:] == tail, chain
        rng = random.Random(11)
        for armed in (False, True, False, True):
            c.host.arm_copy_engine(armed)
            jobs = c.host.counters()["copy_jobs"]
            for _ in range(8):
                experts = rng.sample(range(CAPACITY), TOP_K)  # mostly resident after the first replays
                c.plan(experts)
                graph.replay()
                snapshot = c.snapshot()
                torch.cuda.synchronize()
                c.check(experts, snapshot)
                assert c.retired(), (armed, c.host.counters())
            assert (c.host.counters()["copy_jobs"] > jobs) == armed, (armed, c.host.counters())
    finally:
        c.close()


@pytest.mark.parametrize("fillers", [0, 3000])
def test_the_first_replay_of_a_copy_engine_graph_completes_armed(tmp_path, fillers):
    """Both smokes that failed at startup (abba1-on, diag arm1-on) failed on a decode step that ran armed early in the
    server's life, where a graph variant may run for the first time. A freshly captured graph, with ``fillers`` kernels
    on each side of the chain for the decode graph's size, is replayed for the first time armed with every lane a
    hit: its copy must complete well inside the copy-wait timeout."""
    c = Chain(tmp_path, copy_engine=True)
    try:
        _resident(c, [0, 1, 2])
        x = torch.zeros(1, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            c.gather()  # eager warm-up: loads every kernel; an eager post never copies
            x.add_(1)
        torch.cuda.synchronize()
        assert c.retired()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(fillers):
                x.add_(1)
            c.gather()
            for _ in range(fillers):
                x.add_(1)
        c.host.arm_copy_engine()
        jobs = c.host.counters()["copy_jobs"]
        experts = [2, 0, 1]
        c.plan(experts)
        t0 = time.perf_counter()
        graph.replay()
        snapshot = c.snapshot()
        torch.cuda.synchronize()
        replay_s = time.perf_counter() - t0
        assert replay_s < 1.0, replay_s
        assert c.host.counters()["copy_jobs"] == jobs + 1, c.host.counters()
        c.check(experts, snapshot)
        assert c.retired(), c.host.counters()
    finally:
        c.close()


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
assert c.retired()
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
