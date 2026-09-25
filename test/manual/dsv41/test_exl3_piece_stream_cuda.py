"""Piece streaming's device side (piece-streaming plan section 5): the stream kernel S against the real C++ service,
the real copy tables and, for G1, the real fused MoE consumer. GPU only.

- G1 (= T9) parity with the two-phase arm, with a per-piece pack delay that outlasts W1's budget.
- G2 S streams: the host withholds pieces 1..7 until S reports (StreamProbe) that it copied piece 0.
- G3 a failed read fails as Failed, promptly; T4 (device half) and T5: F's terminal names the S lane and voids the
  quarantined slot.
- G4 / T7 one request deadline, S included.
- G6 (= T8) the flag-on chain is post -> W1 -> C1 -> A1 -> S -> A2 -> F -> add, 9 nodes and 8 edges; the flag-off chain
  is today's, node for node.
- G7 replay freshness; G8 masks full but a protect row fails; G9 S identity; G10 abort/commit race; G11 late last
  publish; T6 S never writes keep; T10 all-miss; T11 all-hit; T12 W1 stops once every lane is claimed or LOADING.

Run on divix01 holding cc-gpu.lock, with PYTHONPATH pointing at the tree under test. Harness shapes are copied from
test_exl3_two_phase_timing_cuda.py (the real-service chain), test_exl3_two_phase_parity_cuda.py (graph topology and
the fused harness) and test_exl3_lease_kernels_cuda.py (the hand-driven lease block).
"""

import inspect
import os
import threading
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
    DEMAND_RECORDS,
    DEMAND_RING,
    RECORD_BYTES,
    STATUS,
    STREAM_FAULT_WORDS,
    WORDS,
    Exl3RamMissDevice,
    Exl3RamMissHost,
    new_page,
    piece_word,
    stream_segment_map,
)
from sglang.kernels.ops.moe.expert_cache_transfer import (  # noqa: E402
    copy_expert_row_segments_gpu,
    expert_row_segments,
)

LAYERS, EXPERTS, CAPACITY = 1, 16, 8
TOP_K = 6
LANES = lease.LANES
REASON = lease.TERMINAL_REASONS
WORD_MASK = (1 << 64) - 1
HIT_WAIT_NS = 100_000  # SGLANG_DSV41_RAM_MISS_HIT_WAIT_US's default
PIECE_DELAY_NS = 2 * HIT_WAIT_NS  # G1: per piece, so a read outlasts W1's budget (kills M7)
ALL = 0xFF


def _cuda_ready():
    torch.cuda.synchronize()


def _idx(seq):
    return (seq - 1) % DEMAND_RECORDS


def _tagged(tag, generation):
    return (tag << 56) | generation


def _set_page_word(page, name, value):
    offset = WORDS[name]
    page[offset : offset + 4].view(torch.int32)[0] = value - (1 << 32) if value >= (1 << 31) else value


class Block:
    """The lease block, read and written from Python (test_exl3_lease_kernels_cuda.py's Block, trimmed)."""

    def __init__(self, block, layout):
        self.block, self.layout = block, layout

    def u64(self, offset):
        return int(self.block[offset : offset + 8].view(torch.int64)[0]) & WORD_MASK

    def set_u64(self, offset, value):
        value &= WORD_MASK
        self.block[offset : offset + 8].view(torch.int64)[0] = value - (1 << 64) if value >= (1 << 63) else value

    def u32(self, offset):
        return int(self.block[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF

    def set_u32(self, offset, value):
        self.block[offset : offset + 4].view(torch.int32)[0] = value - (1 << 32) if value >= (1 << 31) else value

    def d(self, offset):
        return self.layout.d_offset + offset

    def rr(self, idx, lane):
        return lease.ROW_RESULT + (idx * LANES + lane) * lease.ROW_RESULT_BYTES

    def mask_offset(self, idx, lane):
        return self.layout.piece_offset + (idx * LANES + lane) * lease.PIECE_MASK_LINE_BYTES

    def mask(self, idx, lane):
        return self.u64(self.mask_offset(idx, lane))

    def probe(self, idx):
        return self.u64(self.d(lease.STREAM_PROBE + idx * lease.STREAM_PROBE_BYTES))

    def ack_word(self, idx, lane):
        return self.u64(self.d(lease.LANE_ACK + (idx * LANES + lane) * lease.LANE_ACK_BYTES))

    def terminal(self, idx):
        base = self.d(lease.TERMINAL + idx * lease.TERMINAL_BYTES)
        f = lease.TERMINAL_FIELDS
        return {"mask": self.u32(base + f["skipped_mask"]), "reason": self.u32(base + f["reason"]),
                "word": self.u64(base + f["gen"])}


def _close(host, slabs):
    from sglang.srt.layers.moe.expert_host_tier import release_host_slabs

    try:
        if host is not None:
            host.stop()
    finally:
        release_host_slabs([slab for names in slabs.values() for slab in names.values()])


class StreamService:
    """The real chain one call at a time against the real C++ service thread: lease mode, two-phase and (unless
    ``piece_stream`` is False) piece streaming with ``pack_workers`` packing workers. Flag on, the chain is
    post -> W1 -> C1 -> A1 -> S -> A2 -> F; flag off, today's post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F.
    ``layers`` streamed layers are built and every request goes to streamed row ``row``. ``copy_engine`` enables and
    arms the service's copy engine (LEASE_PROTOCOL.md 7.6), posts with its flag and adds the copy wait before F."""

    def __init__(
        self, tmp_path, *, timeout_ms=2000, hit_wait_ns=HIT_WAIT_NS, piece_stream=True, pack_workers=2, layers=LAYERS,
        row=0, copy_engine=False,
    ):
        from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
        from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
        from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
        from sglang.test.dsv41_fake_exl3 import write_fake_exl3

        self.layers, self.row = layers, row
        write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=EXPERTS, hidden=1024, inter=512, finite=True)
        self.layout = build_exl3_expert_layout(str(tmp_path))
        self.fmt = Exl3ExpertFormat(self.layout, row, direct=False)
        self.specs = {s.name: s for s in self.fmt.tensor_specs(None)}
        self.names = EXL3_STREAMED_NAMES
        self.slabs = {lid: {} for lid in range(layers)}
        self.host = None
        self.hit_wait_ns = hit_wait_ns
        self.timeout_ms = timeout_ms
        self.piece_stream = piece_stream
        self.copy_engine = copy_engine
        try:
            for lid in range(layers):
                for n in self.names:
                    self.slabs[lid][n] = allocate_host_slab(CAPACITY, self.specs[n].row_shape, self.specs[n].dtype, register=True)
            self.tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs)
            self.page = new_page(pin=True)
            self.slot_map = torch.full((layers, EXPERTS), -1, dtype=torch.int32).pin_memory()
            self.host = Exl3RamMissHost(
                self.tables, page=self.page, slot_map=self.slot_map, direct=False, pack_workers=pack_workers
            )
            self.host.enable_lease_mode()
            self.host.enable_two_phase()
            if piece_stream:
                self.host.enable_piece_stream()
            if copy_engine:
                self.host.enable_copy_engine(torch.cuda.current_device())
            self.host.start_thread(fatal_wait_s=60.0)
            self.dev = Exl3RamMissDevice(
                self.page, self.slot_map, device="cuda", layers=layers, timeout_ms=timeout_ms, advise=False,
                lease_block=self.host.lease_block, lease_layout=self.host.lease_layout,
                piece_stream=piece_stream, piece_runs=self.host.piece_runs() if piece_stream else None,
            )
        except BaseException:
            _close(self.host, self.slabs)
            raise
        self.block = Block(self.host.lease_block, self.host.lease_layout)
        self.planned = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest = {
            n: torch.zeros((TOP_K,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda") for n in self.names
        }
        self.dest_slots = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        self.segments = expert_row_segments([(self.slabs[row][n], self.dest[n]) for n in self.names])
        self.segment_map = stream_segment_map(self.segments, self.tables, row) if piece_stream else None
        if copy_engine:
            self.host.set_copy_table(row, self.segments.table, TOP_K)
            self.host.arm_copy_engine()
        if piece_stream:
            # One empty (unarmed) request through the whole chain: JIT-compiles every kernel before a test times one.
            self.plan([])
            self.step()

    def close(self):
        _close(self.host, self.slabs)

    def plan(self, experts, routes=None):
        routes = experts if routes is None else routes
        self.planned.fill_(-1)
        self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1)
        self.routes[: len(routes)] = torch.tensor(routes, dtype=torch.int64)

    # --- the stages ------------------------------------------------------------------------------------------
    def post(self):
        if self.copy_engine:
            self.dev.post(self.row, self.planned, self.count, self.routes, -1, dst_slots=self.dest_slots, copy_engine=True)
        else:
            self.dev.post(self.row, self.planned, self.count, self.routes, -1)

    def hit_wait(self):
        self.dev.hit_wait(self.row, self.planned, self.count, self.dest_slots, self.hit_wait_ns)

    def copy1(self):
        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_1, self.dev.dst_slots_1, self.dev.go_1)

    def ack1(self):
        self.dev.stage_ack(1)

    def stream(self):
        """S. Hands S this harness's ``keep`` only when stream() declares that parameter: production's does not, and
        the M10 mutant (S writes keep) adds it so there is a real pointer to write through."""
        args = (self.row, self.planned, self.count, self.dest_slots, self.ram_miss, self.segments, self.segment_map)
        if "keep" in inspect.signature(self.dev.stream).parameters:
            self.dev.stream(*args, keep=self.keep)
        else:
            self.dev.stream(*args)

    def rest_wait(self):
        self.dev.rest_wait(self.row, self.planned, self.count, self.dest_slots, self.ram_miss)

    def copy2(self):
        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_2, self.dev.dst_slots_2, self.dev.go_2)

    def ack2(self):
        self.dev.stage_ack(2)

    def finalize(self):
        self.dev.finalize(self.count, self.keep)

    def copy_wait(self):
        self.dev.copy_wait(self.count)

    def total(self):
        torch.add(self.dev.go_1, self.dev.go_2, out=self.dev.go_total)
        if self.copy_engine:
            self.dev.go_total.add_(self.dev.go_ce)

    def step(self):
        """The whole chain in production order (flag off: W2 and C2 in place of S). Returns the request's seq."""
        self.post()
        self.hit_wait()
        self.copy1()
        self.ack1()
        if self.piece_stream:
            self.stream()
        else:
            self.rest_wait()
            self.copy2()
        self.ack2()
        if self.copy_engine:
            self.copy_wait()
        self.finalize()
        self.total()
        _cuda_ready()
        return self.seq()

    # --- reading back --------------------------------------------------------------------------------------------
    def stats(self):
        return self.dev.stats()

    def seq(self):
        return int(self.stats()["posted"]) & 0xFFFFFFFF

    def generation(self, seq):
        return ((int(self.stats()["pending_epoch"]) & 0xFFFFFFFF) << 32) | seq

    def until(self, predicate, timeout_s=10.0):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if predicate():
                return True
            time.sleep(0.002)
        return False

    def counters(self):
        return self.host.counters()

    def expected(self, experts):
        from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

        source = Exl3ShardRowSource.for_layer(self.layout, self.row, self.fmt.segment_map(), direct=False)
        rows = {}
        for expert in sorted(set(experts)):
            rows[expert] = {n: torch.empty(self.specs[n].row_shape, dtype=self.specs[n].dtype) for n in self.names}
            source.read(torch.tensor([expert]), {n: t.unsqueeze(0) for n, t in rows[expert].items()})
        return rows

    def delivered(self, experts):
        want = self.expected(experts)
        for lane, expert in enumerate(experts):
            for n in self.names:
                if not torch.equal(self.dest[n][lane].cpu().view(torch.uint8), want[expert][n].view(torch.uint8)):
                    return False
        return True

    def make_backend(self):
        from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissRowBackend

        return Exl3RamMissRowBackend(
            segments={0: self.segments},
            host_row_map=self.slot_map[self.row].to("cuda"),
            device_side=self.dev,
            row=self.row,
            next_row=-1,
            capacity=TOP_K,
            two_phase=True,
            hit_wait_ns=self.hit_wait_ns,
            stream_maps={0: self.segment_map} if self.piece_stream else None,
            copy_engine=self.copy_engine,
        )

    def make_plan(self):
        from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

        return ExpertRowPlan(expert_ids=self.planned[:TOP_K], slots=self.dest_slots, count=self.count)

    def quiet(self):
        """Clear the faults and let the service finish whatever read a test left running."""
        self.host.inject(delay_s=0.0)
        self.host.inject_fault()
        assert self.until(lambda: self.host.busy_since_ns() == 0, timeout_s=15.0)


@pytest.fixture
def service(tmp_path):
    s = StreamService(tmp_path)
    try:
        yield s
    finally:
        s.close()


def _warm(s, experts):
    """Make ``experts`` resident through the flag-on chain itself, and wait for its acknowledgements."""
    acked = s.counters()["leases_acked"]
    s.plan(experts)
    s.step()
    assert s.keep.item() == 1.0, s.counters()
    assert s.until(lambda: s.counters()["leases_acked"] == acked + len(experts)), s.counters()


def _post_and_wait_for_hits(s, hits):
    """post, then hold W1 back until the service has granted ``hits`` READY lanes, so which stage takes a hit lane is
    not a race against the service's scheduling (the reason T11 of the two-phase suite does the same)."""
    before = s.counters()["hit_leases_granted"]
    s.post()
    assert s.until(lambda: s.counters()["hit_leases_granted"] == before + hits), s.counters()


# ---------------------------------------------------------------------------------------------------------------
# G2: S streams. The host withholds pieces 1..7 until it acquires StreamProbe == tagged(1, gen), which S stores after
# copying a piece. A wait-for-every-bit S (M3b) never copies, the probe never fires, and the request times out.
# ---------------------------------------------------------------------------------------------------------------
def test_g2_s_copies_a_piece_before_the_rest_are_published(tmp_path):
    s = StreamService(tmp_path, timeout_ms=1000)
    try:
        s.host.inject_fault(hold_until_probe_ms=3000)  # longer than D5: only S's probe can release the pieces early
        s.plan([4])
        start = time.perf_counter()
        seq = s.step()
        elapsed = time.perf_counter() - start
        gen = s.generation(seq)
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        assert s.block.probe(_idx(seq)) == _tagged(lease.STREAM_PROBE_TAG, gen)
        assert elapsed < 0.5, f"the pieces were released by the hold's timeout, not by the probe: {elapsed:.3f} s"
        assert int(s.dev.go_1.item()) == 0 and int(s.dev.go_2.item()) == 1
        assert s.delivered([4])
    finally:
        s.quiet()
        s.close()


# ---------------------------------------------------------------------------------------------------------------
# G3, T4 (device half) and T5: a failed read.
# ---------------------------------------------------------------------------------------------------------------
def test_g3_a_failed_read_fails_as_failed_well_under_the_deadline(tmp_path):
    """keep 0, go_2 0 (so the DIRECT commit, which masks lanes below go_total by keep, commits nothing of S's), the
    terminal names the S lane, and the reason is Failed at the read's end, not Timeout at the deadline (M14)."""
    s = StreamService(tmp_path, timeout_ms=2000)
    try:
        s.host.inject(delay_s=0.2, fail_reads=True)
        s.plan([5])
        start = time.perf_counter()
        seq = s.step()
        elapsed = time.perf_counter() - start
        assert s.keep.item() == 0.0
        assert int(s.dev.go_2.item()) == 0 and int(s.dev.go_total.item()) == 0
        terminal = s.block.terminal(_idx(seq))
        assert terminal["mask"] == 0b1, terminal
        assert terminal["reason"] == REASON["failed"], terminal
        assert elapsed < 0.5 * s.timeout_ms / 1000, elapsed
    finally:
        s.quiet()
        s.close()


def _failed_mixed_request(s):
    """[3 hit, 9 miss] whose read fails: W1 claims the hit (after its grant), S aborts on the failed read."""
    _warm(s, [3])
    s.host.inject(delay_s=0.2, fail_reads=True)
    s.plan([3, 9])
    _post_and_wait_for_hits(s, 1)
    s.hit_wait()
    s.copy1()
    s.ack1()
    s.stream()
    s.ack2()
    s.finalize()
    s.total()
    _cuda_ready()
    return s.seq()


def test_t5_the_terminal_names_the_s_lane_and_not_the_hit_lane(service):
    s = service
    seq = _failed_mixed_request(s)
    assert s.keep.item() == 0.0
    assert int(s.dev.go_1.item()) == 1 and int(s.dev.go_2.item()) == 0
    terminal = s.block.terminal(_idx(seq))
    assert terminal["mask"] & 0b01 == 0, f"the hit lane was acknowledged by A1: {terminal}"
    assert terminal["mask"] & 0b10 != 0, f"the S lane was never acknowledged, so the terminal must name it: {terminal}"
    assert terminal["reason"] == REASON["failed"], terminal


def test_t4_device_the_finalize_terminal_voids_the_quarantined_lane(service):
    """The failed read quarantined the miss lane's leased slot (task 4); F's terminal names that lane, the service
    voids its lease, and the slot returns to FREE with its mapping cleared."""
    s = service
    quarantined = s.counters()["slots_quarantined"]
    voided = s.counters()["leases_voided"]
    _failed_mixed_request(s)
    assert s.counters()["slots_quarantined"] == quarantined + 1, s.counters()
    assert s.until(lambda: s.counters()["leases_voided"] == voided + 1), s.counters()
    assert s.until(lambda: all(state != 3 for state, _, _, _ in s.host.slot_info(0))), s.host.slot_info(0)
    assert s.host.mapping(0)[9] == -1


# ---------------------------------------------------------------------------------------------------------------
# G4 and T7: one request deadline (D5), and S is inside it.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hit_wait_ns", [HIT_WAIT_NS, 10_000_000_000], ids=["g4", "t7"])
def test_g4_t7_the_request_times_out_once_at_the_shared_deadline(tmp_path, hit_wait_ns):
    """G4: S times out at about 1x the deadline, with reason Timeout. T7: with a W1 budget far past the deadline W1
    polls until the deadline itself, and S must not start a deadline of its own after it (that would be ~2x)."""
    timeout_ms = 400
    s = StreamService(tmp_path, timeout_ms=timeout_ms, hit_wait_ns=hit_wait_ns)
    try:
        s.host.inject(delay_s=5.0)
        s.plan([11])
        start = time.perf_counter()
        seq = s.step()
        elapsed = time.perf_counter() - start
        assert s.keep.item() == 0.0 and int(s.dev.go_2.item()) == 0
        assert s.block.terminal(_idx(seq))["reason"] == REASON["timeout"]
        assert 0.9 * timeout_ms / 1000 <= elapsed < 1.4 * timeout_ms / 1000, elapsed
    finally:
        s.quiet()
        s.close()


# ---------------------------------------------------------------------------------------------------------------
# G6 (= T8): the chain's graph, flag on and flag off.
# ---------------------------------------------------------------------------------------------------------------
def _kernel_nodes(fn):
    """Capture ``fn`` in its own graph and return every node as (type, func, grid, block, shared bytes)."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=stream):
        fn()
    nodes = _chain(g.raw_cuda_graph())
    del g
    return nodes


def _node_signature(node):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import checkCudaErrors

    node_type = checkCudaErrors(cuda_drv.cuGraphNodeGetType(node))
    if node_type != cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        return (int(node_type), None, None, None, None)
    p = checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(node))
    return (
        int(node_type), int(p.func), (p.gridDimX, p.gridDimY, p.gridDimZ), (p.blockDimX, p.blockDimY, p.blockDimZ),
        p.sharedMemBytes,
    )


def _chain(raw):
    """The graph's nodes along its one path, asserting it is a simple chain over every node (no fork)."""
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
    roots = [i for i in range(len(nodes)) if not parents[i]]
    assert len(roots) == 1, roots
    order = [roots[0]]
    while len(order) < len(nodes):
        assert len(children[order[-1]]) == 1, "the graph forks"
        order.append(children[order[-1]][0])
    return [_node_signature(nodes[i]) for i in order]


def _stage_signatures(s, *, stream):
    """Each stage's node signature, learned from a graph that captures only that call."""
    signatures = {
        "post": _kernel_nodes(lambda: s.dev.post(0, s.planned, s.count, s.routes, -1)),
        "W1": _kernel_nodes(lambda: s.dev.hit_wait(0, s.planned, s.count, s.dest_slots, s.hit_wait_ns)),
        "C1": _kernel_nodes(s.copy1),
        "A1": _kernel_nodes(s.ack1),
        "A2": _kernel_nodes(s.ack2),
        "F": _kernel_nodes(lambda: s.dev.finalize(s.count, s.keep)),
        "add": _kernel_nodes(s.total),
    }
    if stream:
        signatures["S"] = _kernel_nodes(s.stream)
    else:
        signatures["W2"] = _kernel_nodes(lambda: s.dev.rest_wait(0, s.planned, s.count, s.dest_slots, s.ram_miss))
        signatures["C2"] = _kernel_nodes(
            lambda: copy_expert_row_segments_gpu(s.segments, s.dev.host_rows_2, s.dev.dst_slots_2, s.dev.go_2)
        )
    for name, nodes in signatures.items():
        assert len(nodes) == 1, (name, nodes)
    return {name: nodes[0] for name, nodes in signatures.items()}


def _captured_backend_chain(s):
    s.plan([])  # count 0: every eager launch below (warm-up, stage learning) is an empty, served request
    backend = s.make_backend()
    plan = s.make_plan()
    with torch.cuda.stream(torch.cuda.Stream()):
        backend.post(0, plan)
    _cuda_ready()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        backend.post(0, plan)
    chain = _chain(graph.raw_cuda_graph())
    del graph
    return chain


@pytest.mark.skipif(cuda_drv is None, reason="needs cuda-python (cuda.bindings.driver)")
def test_g6_the_flag_on_chain_is_post_w1_c1_a1_s_a2_f_add(service):
    s = service
    chain = _captured_backend_chain(s)
    assert len(chain) == 9, chain  # 9 nodes; _chain already asserted 8 edges along one path
    assert chain[0][0] == int(cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY), chain[0]
    stages = _stage_signatures(s, stream=True)
    order = ["post", "W1", "C1", "A1", "S", "A2", "F", "add"]
    assert chain[1:] == [stages[name] for name in order], (chain, stages)
    assert stages["S"][2] == (8, 1, 1) and stages["S"][3] == (256, 1, 1), stages["S"]


@pytest.mark.skipif(cuda_drv is None, reason="needs cuda-python (cuda.bindings.driver)")
def test_g6_the_flag_off_chain_is_todays_node_for_node(tmp_path):
    """Flag off: 10 nodes and 9 edges, and every node's kernel, grid, block and shared memory is the two-phase
    stage's; none is the stream kernel or the stream W1 (learned from a flag-on device in the same process). It does
    not compare kernel arguments: that the flag-off kernels take today's parameters rests on their unchanged C++
    signatures and the flag-off API, which this learns the stage signatures from."""
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    on = StreamService(tmp_path / "on")
    try:
        on_stages = _stage_signatures(on, stream=True)
    finally:
        on.close()
    s = StreamService(tmp_path / "off", piece_stream=False)
    try:
        assert s.dev.piece_stream is False and s.dev.stream_count is None
        chain = _captured_backend_chain(s)
        assert len(chain) == 10, chain
        assert chain[0][0] == int(cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY), chain[0]
        stages = _stage_signatures(s, stream=False)
        order = ["post", "W1", "C1", "A1", "W2", "C2", "A2", "F", "add"]
        assert chain[1:] == [stages[name] for name in order], (chain, stages)
        funcs = {node[1] for node in chain}
        assert on_stages["S"][1] not in funcs and on_stages["W1"][1] not in funcs
        assert on_stages["W1"][1] != stages["W1"][1], "flag on must launch its own W1 (it resets S's words)"
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------------------------
# G7: replay freshness. Replay 1 is served; replay 2 aborts in S on a fatal word the host raises mid-read.
# ---------------------------------------------------------------------------------------------------------------
def test_g7_a_replay_that_aborts_in_s_reads_nothing_from_the_replay_before(service):
    s = service
    backend = s.make_backend()
    plan = s.make_plan()
    s.plan([6])
    with torch.cuda.stream(torch.cuda.Stream()):
        backend.post(0, plan)  # warm-up outside capture (a served miss); 6 is resident afterwards
    _cuda_ready()
    assert backend.keep.item() == 1.0, s.counters()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.post(0, plan)

    acked = s.counters()["leases_acked"]
    s.plan([7])  # replay 1: a miss, served by S
    graph.replay()
    _cuda_ready()
    assert backend.keep.item() == 1.0 and int(s.dev.go_2.item()) == 1, (s.counters(), s.stats())
    assert s.until(lambda: s.counters()["leases_acked"] == acked + 1), "replay 1's lease must retire acknowledged"

    s.host.inject(delay_s=0.3)  # replay 2's read is still running when the fatal word goes up
    s.plan([7, 8])  # lane 0 resident (C1 may copy it), lane 1 a miss for S
    raise_fatal = threading.Timer(0.1, lambda: _set_page_word(s.page, "fatal", 0x7FFF0001))
    raise_fatal.start()
    graph.replay()
    _cuda_ready()
    raise_fatal.join()
    seq = s.seq()
    gen = s.generation(seq)
    assert int(s.dev.go_2.item()) == 0, "go_2 must be W1's reset, never replay 1's commit"
    assert backend.keep.item() == 0.0
    assert s.stats()["req_failed"] == 1
    copied = {int(lane) for lane in s.dev.origin_1[: int(s.dev.go_1.item())].tolist()}
    acked_lanes = {lane for lane in range(2) if s.block.ack_word(_idx(seq), lane) & ((1 << 56) - 1) == gen}
    assert acked_lanes == copied, (acked_lanes, copied)
    s.quiet()


# ---------------------------------------------------------------------------------------------------------------
# G8: every mask full, but a protect row's read fails: the request is not served, so S must not commit (M8).
# ---------------------------------------------------------------------------------------------------------------
def test_g8_full_masks_do_not_commit_a_request_whose_protect_row_failed(service):
    s = service
    # Lane 0 is expert 2; the routes also protect 12, so the service reads both (2 is ordinal 0, 12 ordinal 1). 12's
    # completions are held until 2 has packed and published every piece, then its first one fails.
    s.host.inject_fault(part=0, part_error=5, ordinal=1, hold_ordinal=1)
    s.plan([2], routes=[2, 12])
    seq = s.step()
    gen = s.generation(seq)
    assert s.block.mask(_idx(seq), 0) == piece_word(gen, ALL), "the precondition: lane 0's mask is full"
    assert s.keep.item() == 0.0 and int(s.dev.go_2.item()) == 0
    assert s.block.terminal(_idx(seq))["reason"] == REASON["failed"]
    s.quiet()


# ---------------------------------------------------------------------------------------------------------------
# T6: S never writes keep.
# ---------------------------------------------------------------------------------------------------------------
def test_t6_s_never_writes_keep(service):
    s = service
    s.plan([10])
    s.keep.fill_(0.25)  # a sentinel neither S nor anything before F may change
    s.post()
    s.hit_wait()
    s.copy1()
    s.ack1()
    s.stream()
    s.ack2()
    _cuda_ready()
    assert int(s.dev.go_2.item()) == 1, "S must have taken its commit path"
    assert s.keep.item() == 0.25, "only the finalize kernel writes keep"
    s.finalize()
    _cuda_ready()
    assert s.keep.item() == 1.0


# ---------------------------------------------------------------------------------------------------------------
# T10: an all-miss request; W1 stays bounded by its budget.
# ---------------------------------------------------------------------------------------------------------------
def test_t10_all_miss_w1_stays_bounded(tmp_path):
    s = StreamService(tmp_path, timeout_ms=500)
    try:
        s.host.inject(delay_s=5.0)
        s.plan([13, 14])
        s.post()
        start = time.perf_counter()
        s.hit_wait()
        _cuda_ready()
        elapsed = time.perf_counter() - start
        assert int(s.dev.go_1.item()) == 0
        assert elapsed < 0.3, elapsed
        # The miss lanes were granted LOADING at reservation: close the request so the terminal voids them.
        s.copy1()
        s.ack1()
        s.stream()
        s.ack2()
        s.finalize()
        _cuda_ready()
        assert s.keep.item() == 0.0
    finally:
        s.quiet()
        s.close()


# ---------------------------------------------------------------------------------------------------------------
# T12: W1 stops once every lane is claimed or LOADING. A LOADING lane never turns READY, and W1 used to read it as
# unpublished, so it polled it to the end of its budget on every read layer (DSV41_REFERENCE.md 24.7, item 3).
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hits", [[], [1, 2]], ids=["all-miss", "mixed"])
def test_t12_w1_stops_once_every_lane_is_claimed_or_loading(tmp_path, hits):
    s = StreamService(tmp_path, timeout_ms=500, hit_wait_ns=50_000_000)  # a budget the old W1 ran out visibly
    try:
        if hits:
            _warm(s, hits)
        misses = [13, 14]
        s.host.inject(delay_s=5.0)  # the misses' read stays in flight: their lanes stay LOADING throughout
        s.plan(hits + misses)
        granted = s.counters()["leases_granted"]
        s.post()
        assert s.until(lambda: s.counters()["leases_granted"] == granted + len(hits) + len(misses)), s.counters()
        passes = s.stats()["w1_passes"]
        s.hit_wait()
        _cuda_ready()
        assert s.stats()["w1_passes"] - passes <= 2, s.stats()
        assert int(s.dev.go_1.item()) == len(hits), "W1 left before claiming a READY lane"
        # Close the request as T10 does: S times out on the held misses and F voids them.
        s.copy1()
        s.ack1()
        s.stream()
        s.ack2()
        s.finalize()
        _cuda_ready()
        assert s.keep.item() == 0.0
    finally:
        s.quiet()
        s.close()


# ---------------------------------------------------------------------------------------------------------------
# T11: an all-hit request: W1 takes every lane; S copies nothing and acknowledges nothing.
# ---------------------------------------------------------------------------------------------------------------
def test_t11_all_hit_s_copies_and_acknowledges_nothing(service):
    s = service
    experts = [1, 2, 3]
    _warm(s, experts)
    slots = [s.host.mapping(0)[e] for e in experts]
    spare = next(slot for slot in range(CAPACITY) if slot not in slots)
    for n in s.names:
        s.dest[n].zero_()
        s.slabs[0][n][spare].fill_(-1)  # poison a slot no lane reads
    s.plan(experts)
    pieces = s.stats()["stream_pieces"]
    _post_and_wait_for_hits(s, len(experts))
    s.hit_wait()
    s.copy1()
    s.ack1()
    _cuda_ready()
    assert int(s.dev.go_1.item()) == len(experts)
    seq = s.seq()
    acks = [s.block.ack_word(_idx(seq), lane) for lane in range(LANES)]
    s.stream()
    s.ack2()
    _cuda_ready()
    assert int(s.dev.go_2.item()) == 0
    assert s.stats()["stream_pieces"] == pieces, "S copied a piece of a request W1 took whole"
    assert [s.block.ack_word(_idx(seq), lane) for lane in range(LANES)] == acks
    assert s.delivered(experts)
    s.finalize()
    _cuda_ready()
    assert s.keep.item() == 1.0


# ---------------------------------------------------------------------------------------------------------------
# G11: the owner's last CAS lands just before kDemandDone while S's leader stalls between reading the masks and
# reading kDemandDone. Judged on the masks read before kDemandDone (M15), a complete request looks incomplete.
# ---------------------------------------------------------------------------------------------------------------
def test_g11_a_late_last_publish_is_judged_on_the_masks_reread_after_kdemanddone(service):
    s = service
    s.host.inject_fault(last_publish_delay_ns=2_000_000)
    s.dev.stream_fault[STREAM_FAULT_WORDS["stall_ns"]] = 1_000_000
    backend = s.make_backend()
    plan = s.make_plan()
    s.plan([0])
    with torch.cuda.stream(torch.cuda.Stream()):
        backend.post(0, plan)
    _cuda_ready()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.post(0, plan)
    for replay in range(1000):
        s.plan([(replay + 1) % EXPERTS])  # 16 experts over 8 slots: every replay's lane is a miss S streams
        graph.replay()
        _cuda_ready()
        assert backend.keep.item() == 1.0, (replay, s.counters(), s.stats())
    assert s.stats()["stream_pieces"] > 0
    s.dev.stream_fault.zero_()
    s.quiet()


# ---------------------------------------------------------------------------------------------------------------
# G1 (= T9): output parity with the two-phase arm, through the real fused MoE consumer.
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


def _fused_layer(tmp_path, *, piece_stream: bool, timeout_ms=2000):
    """test_exl3_two_phase_parity_cuda.py's single-layer fused harness, two-phase on, with the piece-stream switch
    and two packing workers (piece streaming needs them; the two-phase arm gets the same, for a like comparison)."""
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
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE.override(True),
        envs.SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM.override(piece_stream),
        envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.override(2),
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
    assert service.two_phase is True and service.piece_stream is piece_stream
    assert streamer.row_backend.piece_stream is piece_stream
    return layer, streamer, service, checks


@NEEDS_EXL3_SRC
def test_g1_piece_streaming_output_is_bitwise_equal_to_the_two_phase_arm(tmp_path):
    """G1 = T9 for piece streaming: rows and fused output bitwise equal to the two-phase arm, eager and graph, with a
    per-piece pack delay of 2x W1's budget (so a W1 that took a LOADING lane, M7, or an S that copied before a
    piece's bit, M3, would copy bytes the pack had not written yet) and a poisoned bounce."""
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    timeout_ms = 2000
    injected_ns = F_TOP_K * 8 * PIECE_DELAY_NS  # the most delay one request carries: every lane a miss, 8 pieces each
    assert injected_ns < 0.05 * timeout_ms * 1_000_000, "the injected delay must stay under 5% of the deadline"

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=F_EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path)
    gen = torch.Generator(device="cpu").manual_seed(3)
    x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
    weights = torch.softmax(torch.randn((1, F_TOP_K), generator=gen), -1).cuda()
    eager_routes = ([0, 3, 5, 1, 7, 6], [9, 10, 11, 0, 1, 12])
    graph_routes = eager_routes + ([13, 14, 15, 2, 4, 8],)  # the third is new: misses under the graph as well

    results = {}
    for arm, piece_stream in (("two_phase", False), ("piece_stream", True)):
        layer, streamer, service, checks = _fused_layer(tmp_path, piece_stream=piece_stream, timeout_ms=timeout_ms)
        try:
            service.host.inject_fault(pack_delay_ns=PIECE_DELAY_NS, poison=True)
            dev = streamer.row_backend.device_side
            ids = torch.tensor([eager_routes[0]], device="cuda", dtype=torch.int32)
            eager_outs, graph_outs, gathers = [], [], []
            for route in eager_routes:
                ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
                eager_outs.append(Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0).float().clone())
                assert streamer.row_backend.keep.item() == 1.0, (arm, route, service.host.counters())
                for check in checks:
                    check()
                if piece_stream and route == eager_routes[1]:
                    # 9..12 were never resident: their lanes were granted LOADING, which W1 must never claim.
                    planned = streamer.row_backend.planned.tolist()
                    claimed = dev.claimed.tolist()
                    for expert in (9, 10, 11, 12):
                        assert claimed[planned.index(expert)] == 0, (expert, planned, claimed)
                remap, tensors = streamer.gather(ids)
                slots = remap.reshape(-1).tolist()
                for k, expert in enumerate(route):
                    for name, rows in tensors.items():
                        assert torch.equal(rows[slots[k]].cpu(), source[name][expert]), (arm, route, k, expert, name)
                gathers.append((remap.clone(), {n: t.clone() for n, t in tensors.items()}))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
            for route in graph_routes:
                ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
                graph.replay()
                _cuda_ready()
                assert streamer.row_backend.keep.item() == 1.0, (arm, route, service.host.counters())
                graph_outs.append(out.float().clone())
            if piece_stream:
                assert dev.stats()["stream_pieces"] > 0, "S never copied a piece: nothing was streamed"
            results[arm] = {"eager": eager_outs, "graph": graph_outs, "gathers": gathers}
            assert service.host.fatal_seq() == 0, (arm, service.host.counters())
        finally:
            service.host.inject_fault()
            service.shutdown()

    for i, route in enumerate(graph_routes):
        assert torch.equal(results["two_phase"]["graph"][i], results["piece_stream"]["graph"][i]), (route, "graph")
    for i, route in enumerate(eager_routes):
        assert torch.equal(results["two_phase"]["eager"][i], results["piece_stream"]["eager"][i]), (route, "eager")
        assert torch.equal(results["piece_stream"]["eager"][i], results["piece_stream"]["graph"][i]), route
        remap1, tensors1 = results["two_phase"]["gathers"][i]
        remap2, tensors2 = results["piece_stream"]["gathers"][i]
        slots1, slots2 = remap1.reshape(-1).tolist(), remap2.reshape(-1).tolist()
        for k in range(len(route)):
            for name in tensors1:
                assert torch.equal(
                    tensors1[name][slots1[k]].view(torch.uint8), tensors2[name][slots2[k]].view(torch.uint8)
                ), (route, k, name)


def test_g1_a_second_layer_streams_its_own_rows_equal_to_the_flag_off_arm(tmp_path):
    """G1 through streamed row 1 of 2: S indexes its piece table by (row, expert), so a wrong row index copies row 0's
    cuts over row 1's bytes. The precondition checks the two rows' cuts differ for the planned experts, so the
    comparison can see that; the bytes must equal the flag-off (two-phase) arm and the checkpoint."""
    experts = [3, 5, 9, 12]
    delivered = {}
    for piece_stream in (False, True):
        directory = tmp_path / ("on" if piece_stream else "off")
        directory.mkdir()
        s = StreamService(directory, layers=2, row=1, piece_stream=piece_stream)
        try:
            if piece_stream:
                runs = s.dev.piece_runs.cpu()
                assert all(not torch.equal(runs[0, e], runs[1, e]) for e in experts), "rows 0 and 1 cut alike"
            s.host.inject_fault(pack_delay_ns=PIECE_DELAY_NS, poison=True)
            s.plan(experts)
            s.step()
            assert s.keep.item() == 1.0, (piece_stream, s.counters(), s.stats())
            if piece_stream:
                assert int(s.dev.go_2.item()) == len(experts) and s.stats()["stream_pieces"] > 0
            assert s.delivered(experts), piece_stream
            delivered[piece_stream] = {n: s.dest[n][: len(experts)].cpu().view(torch.uint8).clone() for n in s.names}
        finally:
            s.quiet()
            s.close()
    for name, rows in delivered[False].items():
        assert torch.equal(rows, delivered[True][name]), name


# ---------------------------------------------------------------------------------------------------------------
# G9 and G10: the hand-driven lease block (no service): the test writes the RowResults, the masks and the page.
# ---------------------------------------------------------------------------------------------------------------
REC_STATUS = 10  # kRecStatus
ROW_BYTES = 4096  # the hand-driven copy table's one segment


class Rig:
    """One flag-on device over a lease block this test writes by hand, with a one-segment copy table and a piece
    table that cuts the segment's 4096-byte row into 8 pieces of 512 bytes."""

    def __init__(self, *, timeout_ms=500):
        self.page = new_page(pin=True)
        self.slot_map = torch.full((1, EXPERTS), -1, dtype=torch.int32).pin_memory()
        self.layout = lease.lease_layout([CAPACITY])
        self.raw = lease.new_lease_block(self.layout, pin=True)
        self.block = Block(self.raw, self.layout)
        self.block.set_u32(lease.ROW_TABLE, self.layout.slot_gen_base[0])
        self.block.set_u32(lease.ROW_TABLE + 4, CAPACITY)
        runs = torch.zeros((1, EXPERTS, 8, 1, 2), dtype=torch.int32)
        for piece in range(8):
            runs[0, :, piece, 0, 0] = piece * ROW_BYTES // 8
            runs[0, :, piece, 0, 1] = (piece + 1) * ROW_BYTES // 8
        self.dev = Exl3RamMissDevice(
            self.page, self.slot_map, device="cuda", layers=1, timeout_ms=timeout_ms, advise=False,
            lease_block=self.raw, lease_layout=self.layout, piece_stream=True, piece_runs=runs,
        )
        self.slab = torch.arange(CAPACITY * ROW_BYTES, dtype=torch.int64).remainder(251).to(torch.uint8)
        self.slab = self.slab.reshape(CAPACITY, ROW_BYTES).pin_memory()
        self.dest = torch.zeros((TOP_K, ROW_BYTES), dtype=torch.uint8, device="cuda")
        self.segments = expert_row_segments([(self.slab, self.dest)])
        self.segment_map = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        self.planned = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest_slots = torch.arange(TOP_K, dtype=torch.int32, device="cuda")

    def plan(self, experts):
        self.planned.fill_(-1)
        self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1)
        self.routes[: len(experts)] = torch.tensor(experts, dtype=torch.int64)

    def post(self):
        self.dev.post(0, self.planned, self.count, self.routes, -1)
        _cuda_ready()
        seq = int(self.dev.stats()["posted"]) & 0xFFFFFFFF
        return seq, ((int(self.dev.stats()["pending_epoch"]) & 0xFFFFFFFF) << 32) | seq

    def serve_loading(self, seq, gen, lanes, *, bits=ALL):
        """Per lane (expert, host_slot): its mask word, then its RowResult under tag LOADING, then the request served."""
        idx = _idx(seq)
        for lane, (expert, slot) in enumerate(lanes):
            self.block.set_u64(self.block.mask_offset(idx, lane), piece_word(gen, bits))
            base = self.block.rr(idx, lane)
            f = lease.ROW_RESULT_FIELDS
            self.block.set_u32(base + f["slot_generation"], 0)
            self.block.set_u32(base + f["host_slot"], slot & 0xFFFFFFFF)
            self.block.set_u32(base + f["expert"], expert)
            self.block.set_u64(base + f["ready"], _tagged(lease.LOADING, gen))
        record = DEMAND_RING + idx * RECORD_BYTES
        self.page[record + REC_STATUS : record + REC_STATUS + 2].view(torch.int16)[0] = STATUS["served"]
        _set_page_word(self.page, "demand_done", seq)

    def rest(self):
        """Everything after post: W1 -> C1 -> A1 -> S -> A2 -> F."""
        self.dev.hit_wait(0, self.planned, self.count, self.dest_slots, HIT_WAIT_NS)
        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_1, self.dev.dst_slots_1, self.dev.go_1)
        self.dev.stage_ack(1)
        self.dev.stream(0, self.planned, self.count, self.dest_slots, self.ram_miss, self.segments, self.segment_map)
        self.dev.stage_ack(2)
        self.dev.finalize(self.count, self.keep)
        _cuda_ready()


@pytest.mark.parametrize(
    "lane", [(9, 1), (4, CAPACITY)], ids=["wrong_expert", "host_slot_past_capacity"]
)
def test_g9_a_loading_row_result_that_fails_validation_is_an_identity_violation(lane):
    rig = Rig()
    rig.plan([4])
    seq, gen = rig.post()
    rig.serve_loading(seq, gen, [lane])
    rig.rest()
    assert rig.keep.item() == 0.0 and int(rig.dev.go_2.item()) == 0
    assert rig.block.terminal(_idx(seq))["reason"] == REASON["identity"]


def test_g9_control_a_valid_loading_row_result_is_streamed_and_committed():
    rig = Rig()
    rig.plan([4])
    seq, gen = rig.post()
    rig.serve_loading(seq, gen, [(4, 3)])
    rig.rest()
    assert rig.keep.item() == 1.0 and int(rig.dev.go_2.item()) == 1
    assert torch.equal(rig.dest[0].cpu(), rig.slab[3])


@pytest.mark.parametrize("aborter", ["counts_last", "counts_first"])
def test_g10_one_aborting_block_leaves_go_2_zero_and_the_request_failed(aborter):
    """Block 2 takes the abort path. counts_last: it stores late, after the others counted, so the aborting block is
    itself the last one and its refusal rests on its own ``sh.aborting``, not on the counter decision (it is the
    race a missing fence would open, M13b). counts_first: the completing blocks count late, so a completing block
    is last; only this case tests the counter decision (completed count and abort word, M13a)."""
    rig = Rig()
    rig.dev.stream_fault[STREAM_FAULT_WORDS["abort_block"]] = 3
    delay = "abort_delay_ns" if aborter == "counts_last" else "count_delay_ns"
    rig.dev.stream_fault[STREAM_FAULT_WORDS[delay]] = 200_000
    rig.plan([4])
    seq, gen = rig.post()
    rig.serve_loading(seq, gen, [(4, 3)])
    rig.dev.hit_wait(0, rig.planned, rig.count, rig.dest_slots, HIT_WAIT_NS)
    rig.dev.stream(0, rig.planned, rig.count, rig.dest_slots, rig.ram_miss, rig.segments, rig.segment_map)
    _cuda_ready()
    assert int(rig.dev.go_2.item()) == 0
    assert rig.dev.stats()["req_failed"] == 1


def test_g10_without_an_abort_a_thousand_replays_all_commit():
    rig = Rig()
    for replay in range(1000):
        rig.plan([replay % EXPERTS, (replay + 5) % EXPERTS])
        seq, gen = rig.post()
        rig.serve_loading(seq, gen, [(replay % EXPERTS, 1), ((replay + 5) % EXPERTS, 2)])
        rig.rest()
        assert rig.keep.item() == 1.0 and int(rig.dev.go_2.item()) == 2, (replay, rig.dev.stats())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
