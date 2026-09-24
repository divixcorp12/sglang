"""Task 6 V1 two-phase, device chain: timing and degenerate shapes (task6-v1-checklist.md S5 T6, T7, T10, T11).

Real GPU kernels against the real C++ service thread in lease mode with two-phase enabled
(``Exl3RamMissHost.enable_two_phase()``), following the same shape as
``test_exl3_lease_kernels_cuda.py``'s ``TestServiceEndToEnd``. ``TwoPhaseService`` mirrors
``Exl3RamMissRowBackend.post``'s two-phase branch (post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F,
``exl3_ram_miss.py:345-360``) one call at a time, so a test can intervene between stages -- T6
needs to rewrite a slot's generation between the grant and its acknowledgement, and T7/T10 need
to measure one stage's wall time in isolation.

Run on divix01 under ``gpu-run.sh`` (it holds cc-gpu.lock) with PYTHONPATH pointing at the tree
under test.
"""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import exl3_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissDevice, Exl3RamMissHost, WORDS, new_page  # noqa: E402

LAYERS, EXPERTS, CAPACITY = 1, 16, 8
TOP_K = 6
LANES = lease.LANES
REASON = lease.TERMINAL_REASONS


def _cuda_ready():
    torch.cuda.synchronize()


def page_word(page, name):
    offset = WORDS[name]
    return int(page[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF


class Block:
    """The subset of test_exl3_lease_kernels_cuda.py's ``Block`` this file needs."""

    def __init__(self, block, layout):
        self.block, self.layout = block, layout

    def u32(self, offset):
        return int(self.block[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF

    def set_u32(self, offset, value):
        self.block[offset : offset + 4].view(torch.int32)[0] = value - (1 << 32) if value >= (1 << 31) else value

    def d(self, offset):
        return self.layout.d_offset + offset

    def slot_gen_offset(self, row, slot):
        return lease.SLOT_GEN + 4 * (self.layout.slot_gen_base[row] + slot)

    def ack_word(self, idx, lane):
        base = self.d(lease.LANE_ACK + (idx * LANES + lane) * lease.LANE_ACK_BYTES)
        return int(self.block[base : base + 8].view(torch.int64)[0]) & ((1 << 64) - 1)

    def terminal(self, idx):
        base = self.d(lease.TERMINAL + idx * lease.TERMINAL_BYTES)
        f = lease.TERMINAL_FIELDS
        return {"mask": self.u32(base + f["skipped_mask"]), "reason": self.u32(base + f["reason"])}


def _close(host, slabs):
    from sglang.srt.layers.moe.expert_host_tier import release_host_slabs

    try:
        if host is not None:
            host.stop()
    finally:
        release_host_slabs([slab for names in slabs.values() for slab in names.values()])


class TwoPhaseService:
    """The real post/hit_wait/copy/stage_ack/rest_wait/copy/stage_ack/finalize chain, one call at a
    time, against the real C++ service thread with two-phase enabled.
    """

    def __init__(self, tmp_path, *, timeout_ms=2000, hit_wait_ns=100_000, advise=False):
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
        self.hit_wait_ns = hit_wait_ns
        try:
            for lid in range(LAYERS):
                for n in self.names:
                    self.slabs[lid][n] = allocate_host_slab(CAPACITY, self.specs[n].row_shape, self.specs[n].dtype, register=True)
            tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs)
            self.page = new_page(pin=True)
            slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
            self.slot_map = slot_map
            self.host = Exl3RamMissHost(tables, page=self.page, slot_map=slot_map, direct=False)
            self.host.enable_lease_mode()  # two-phase is refused without lease mode
            self.host.enable_two_phase()
            self.host.start_thread(fatal_wait_s=60.0)
            self.dev = Exl3RamMissDevice(
                self.page, slot_map, device="cuda", layers=LAYERS, timeout_ms=timeout_ms, advise=advise,
                lease_block=self.host.lease_block, lease_layout=self.host.lease_layout,
            )
        except BaseException:
            _close(self.host, self.slabs)
            raise
        self.block = Block(self.host.lease_block, self.host.lease_layout)
        self.planned = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest = {
            n: torch.zeros((TOP_K,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda") for n in self.names
        }
        from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments

        self.segments = expert_row_segments([(self.slabs[0][n], self.dest[n]) for n in self.names])
        self.dest_slots = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        self.host_rows = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")

    def close(self):
        _close(self.host, self.slabs)

    def plan(self, experts):
        self.planned.fill_(-1)
        self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1)
        self.routes[: len(experts)] = torch.tensor(experts, dtype=torch.int64)

    def post(self, row=0):
        self.dev.post(row, self.planned, self.count, self.routes, -1)

    def hit_wait(self, row=0):
        self.dev.hit_wait(row, self.planned, self.count, self.dest_slots, self.hit_wait_ns)

    def copy1(self):
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_1, self.dev.dst_slots_1, self.dev.go_1)

    def ack1(self):
        self.dev.stage_ack(1)

    def rest_wait(self, row=0):
        """Stage 2. Tries a ``keep`` kwarg first (the T6 mutant's restored write needs a real
        pointer to write through); falls back to the production 5-argument call when that kwarg
        is not accepted, so this method -- and every test that calls it -- is unchanged whether
        or not that mutant is applied.
        """
        try:
            self.dev.rest_wait(row, self.planned, self.count, self.dest_slots, self.ram_miss, keep=self.keep)
        except TypeError:
            self.dev.rest_wait(row, self.planned, self.count, self.dest_slots, self.ram_miss)

    def copy2(self):
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        copy_expert_row_segments_gpu(self.segments, self.dev.host_rows_2, self.dev.dst_slots_2, self.dev.go_2)

    def ack2(self):
        self.dev.stage_ack(2)

    def finalize(self):
        self.dev.finalize(self.count, self.keep)

    def step_two_phase(self, row=0):
        """The whole chain, production order, one call: post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F."""
        self.post(row)
        self.hit_wait(row)
        self.copy1()
        self.ack1()
        self.rest_wait(row)
        self.copy2()
        self.ack2()
        self.finalize()

    def step_batched(self, row=0, next_row=-1):
        """The Task 5 batched (M1) chain, entirely separate kernels from every two-phase one
        (D2's resolution: M1 stays unchanged so both arms exist in one binary). Used to establish
        residency for a test without touching stage_ack/hit_wait/rest_wait/finalize at all, so a
        two-phase-kernel mutant cannot leak into a test's setup step.
        """
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        self.dev.post(row, self.planned, self.count, self.routes, next_row)
        self.dev.wait(row, self.planned, self.count, self.host_rows, self.keep, self.ram_miss)
        copy_expert_row_segments_gpu(self.segments, self.host_rows, self.dest_slots, self.dev.go_count)
        self.dev.ack(self.keep)

    def until(self, predicate, timeout_s=10.0):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if predicate():
                return True
            time.sleep(0.002)
        return False

    def expected(self, experts):
        """Ground truth per expert, read straight from the fake EXL3 files (test_exl3_lease_kernels_cuda.py)."""
        from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

        source = Exl3ShardRowSource.for_layer(self.layout, 0, self.fmt.segment_map(), direct=False)
        rows = {}
        for expert in sorted(set(experts)):
            rows[expert] = {n: torch.empty(self.specs[n].row_shape, dtype=self.specs[n].dtype) for n in self.names}
            source.read(torch.tensor([expert]), {n: t.unsqueeze(0) for n, t in rows[expert].items()})
        return rows

    def delivered(self, experts):
        """Every lane's destination row is byte-identical to ground truth: nothing else was read."""
        want = self.expected(experts)
        for lane, expert in enumerate(experts):
            for n in self.names:
                if not torch.equal(self.dest[n][lane].cpu().view(torch.uint8), want[expert][n].view(torch.uint8)):
                    return False
        return True


@pytest.fixture
def service(tmp_path):
    s = TwoPhaseService(tmp_path)
    try:
        yield s
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------------------------
def test_t6_stage2_wait_never_writes_keep_so_a_stage1_violation_is_not_overwritten(service):
    """T6: keep has exactly one writer, the finalize kernel.

    A mixed request (one hit, one miss). Stage 1 claims the hit lane; stage 2 -- run BEFORE stage
    1's acknowledgement, so its own wait genuinely reaches the success path rather than being
    poisoned by the fatal word stage 1's violation raises -- serves the miss lane cleanly. ``keep``
    is pre-set to a sentinel (0.0, never written by anything but F) so that stage 2's kernel taking
    its success path is exactly what would flip it if stage 2 wrote ``keep`` at all. Only after
    that is the hit lane's slot generation rewritten (what a recycle would do) and stage 1
    acknowledged, forcing VIOLATED; the finalize kernel must then be the one to decide `keep`.

    The miss lane's read is delayed (``inject(delay_s=...)``): the fake EXL3 corpus is tiny enough
    that an undelayed read can finish inside stage 1's own poll window, so the miss lane's
    RowResult can legitimately publish before stage 1 gives up and get claimed as a second hit --
    correct two-phase behavior, but it would make this mixed-request setup nondeterministic.
    """
    s = service
    s.plan([3])  # make expert 3 resident, so the mixed request's lane 0 is a hit
    s.step_two_phase()
    _cuda_ready()
    assert s.until(lambda: s.host.counters()["leases_acked"] == 1), s.host.counters()

    s.host.inject(delay_s=0.05)  # keep lane 1 (expert 9) a miss past stage 1's poll window
    s.plan([3, 9])  # lane 0: hit (resident); lane 1: miss (never loaded)
    s.keep.fill_(0.0)  # sentinel: nothing but F may change this
    s.post()
    s.hit_wait()
    _cuda_ready()
    assert int(s.dev.go_1[0]) == 1, "lane 0 (expert 3) must be claimed as a hit"
    s.copy1()

    # Stage 2 BEFORE stage 1's acknowledgement: the fatal word stage 1's violation will raise has
    # not fired yet, so stage 2's own wait can reach genuine success rather than the !ok branch.
    s.rest_wait()
    _cuda_ready()
    assert int(s.dev.go_2[0]) == 1, "lane 1 (expert 9) must be served and claimed by stage 2"
    s.copy2()
    s.ack2()
    _cuda_ready()
    # The mutant-killing assertion: stage 2's wait and stage 2's ack have both run, and nothing
    # but F may have touched `keep`. It must still read the sentinel.
    assert s.keep.item() == 0.0, "stage 2's wait must not write keep (D2); only the finalize kernel may"

    slot = int(s.dev.lane_ctx_1[0, 3])  # {generation, slot_generation, row, slot}
    offset = s.block.slot_gen_offset(0, slot)
    s.block.set_u32(offset, s.block.u32(offset) + 1)  # what a recycle of the slot would do
    s.ack1()
    _cuda_ready()

    s.finalize()
    _cuda_ready()
    assert s.keep.item() == 0.0, "F must fail the request: a VIOLATED lane makes it unserved"
    assert page_word(s.page, "fatal") != 0
    s.host.inject(delay_s=0.0)


def test_t7_one_request_deadline_not_one_per_stage(tmp_path):
    """T7: one absolute request deadline (D5), computed once by the post kernel, not one per stage.

    ``hit_wait_ns`` is set far above any real deployment value so stage 1's own poll loop is bounded
    by the deadline rather than by its own budget -- with a realistic budget (T10's concern)
    stage 1 always returns in microseconds regardless of the deadline, and this test would not
    exercise D5 at all.
    """
    timeout_ms = 400
    s = TwoPhaseService(tmp_path, timeout_ms=timeout_ms, hit_wait_ns=10_000_000_000)
    try:
        s.host.inject(delay_s=5.0)  # the demand read must not complete inside this test's window
        s.plan([11])  # never loaded: a miss in both stages
        start = time.perf_counter()
        s.post()
        s.hit_wait()
        _cuda_ready()
        s.copy1()
        s.ack1()
        s.rest_wait()
        _cuda_ready()
        elapsed = time.perf_counter() - start
        s.copy2()
        s.ack2()
        s.finalize()
        _cuda_ready()
        assert int(s.dev.go_1[0]) == 0 and int(s.dev.go_2[0]) == 0
        assert s.keep.item() == 0.0
        assert s.block.terminal(0)["reason"] == REASON["timeout"]
        # One shared deadline: the whole request fails at ~1x the timeout. Two independent
        # per-stage deadlines (the mutant) fail at ~2x. 1.4x sits with a wide margin on both
        # sides of that 1x/2x split, wide enough to survive scheduling jitter on a loaded box
        # (divix01 runs a standing QuestDB JVM and ethereum nodes).
        assert elapsed < 1.4 * (timeout_ms / 1000), elapsed
    finally:
        s.host.inject(delay_s=0.0)
        assert s.until(lambda: s.host.busy_since_ns() == 0, timeout_s=15.0)
        s.close()


def test_t10_all_miss_request_does_not_pay_the_read_wait_twice(tmp_path):
    """T10: stage 1's poll is bounded (D1), so an all-miss request commits go_1 == 0 promptly
    instead of also spinning through the full demand-read wait inside stage 1's own poll loop.

    ``hit_wait_ns`` is the real default (``SGLANG_DSV41_RAM_MISS_HIT_WAIT_US`` defaults to 100)
    rather than an inflated one, because this test is about that bound doing its job, not about
    the deadline sharing T7 covers.
    """
    timeout_ms = 2000
    s = TwoPhaseService(tmp_path, timeout_ms=timeout_ms, hit_wait_ns=100_000)
    try:
        s.host.inject(delay_s=5.0)  # the demand read must not complete inside this test's window
        s.plan([13])  # never loaded: a miss
        s.post()
        start = time.perf_counter()
        s.hit_wait()
        _cuda_ready()
        elapsed = time.perf_counter() - start
        assert int(s.dev.go_1[0]) == 0
        # Bounded poll (~64 iterations of a 256 ns nanosleep plus kernel-launch overhead, well
        # under a millisecond) versus spin-to-deadline (2 s): 300 ms sits nearly two orders of
        # magnitude above the bounded case and almost seven times below the spin case, wide
        # enough to survive scheduling jitter on a loaded box.
        assert elapsed < 0.3, elapsed
    finally:
        s.host.inject(delay_s=0.0)
        assert s.until(lambda: s.host.busy_since_ns() == 0, timeout_s=15.0)
        s.close()


def test_t11_all_hit_request_is_entirely_stage1_and_stage2_acknowledges_nothing(tmp_path):
    """T11: the batched path survives as stage 1 covering every lane.

    All k lanes hit: stage 1 claims and acknowledges every lane, stage 2 legitimately commits
    zero, and its acknowledgement kernel -- guarded by ``entry < go_2[0]`` -- must touch nothing.
    A poisoned source slab planted at a slot no planned lane uses, checked against ground truth
    after the chain runs, proves stage 2's (no-op) copy read nothing: a real int16 weight value
    can legitimately equal any sentinel, so only a full read-back-and-compare rules this out.

    Stage 1's own poll timing (O4, unmeasured) is a separate, already-flagged concern (see T10):
    a lane the poll has not yet observed by the time it gives up correctly and safely falls
    through to stage 2, so a race between ``post`` and ``hit_wait`` can legitimately split an
    all-hit request across both stages even at a generous poll bound (confirmed empirically: 1 of
    7 runs at a 10_000_000-poll bound read only 2 of 3 lanes as hits). That race is not what T11 is
    about, so the test waits on the host's own ``hit_leases_granted`` counter -- incremented
    synchronously inside the reservation hold that publishes the RowResults -- before launching
    stage 1's kernel at all, removing it from this test's scope entirely.

    Setup uses the batched (M1) chain, not the two-phase one: the residency-establishing request
    is an all-miss request, so its own stage 1 is empty, and stage_ack's mutant (an empty stage
    acknowledging lane 0 anyway) would corrupt it with a bogus, zero-generation acknowledgement --
    which mismatches the real slot generation and raises VIOLATED, sending the finalize kernel's
    ``keep = 0`` / sticky / fatal path during what is meant to be a clean setup step, before the
    all-hit request this test is actually about ever runs. Confirmed empirically: with the T11
    mutant applied and the two-phase chain used for setup, the all-hit request's own
    ``hit_leases_granted`` wait timed out completely (the device latched sticky during setup), not
    at the ``leases_acked`` assertion the checklist names. The batched chain shares no kernel with
    stage_ack, so it cannot trip that mutant.
    """
    s = TwoPhaseService(tmp_path, hit_wait_ns=100_000)
    try:
        experts = [1, 2, 3]
        s.plan(experts)
        s.step_batched()  # first pass: loads all three via the miss path, establishing residency
        _cuda_ready()
        assert s.until(lambda: s.host.counters()["leases_acked"] == len(experts)), s.host.counters()

        for n in s.names:
            s.dest[n].zero_()
            s.slabs[0][n][5].fill_(-1)  # poison a slot no planned lane below will occupy

        s.plan(experts)  # all three now resident: an all-hit request
        leases_acked_before = s.host.counters()["leases_acked"]
        hit_leases_before = s.host.counters()["hit_leases_granted"]
        s.post()
        # Wait for all three hit publishes to land before stage 1 polls at all (see the docstring).
        assert s.until(lambda: s.host.counters()["hit_leases_granted"] == hit_leases_before + len(experts))
        s.hit_wait()
        _cuda_ready()
        assert int(s.dev.go_1[0]) == len(experts), "every lane must be claimed by stage 1"
        s.copy1()
        s.ack1()
        _cuda_ready()
        assert s.until(lambda: s.host.counters()["leases_acked"] == leases_acked_before + len(experts))
        ack_words_after_stage1 = [s.block.ack_word(0, lane) for lane in range(len(experts))]

        s.rest_wait()
        _cuda_ready()
        assert int(s.dev.go_2[0]) == 0, "nothing is left for stage 2 to claim"
        s.copy2()
        s.ack2()
        _cuda_ready()

        assert [s.block.ack_word(0, lane) for lane in range(len(experts))] == ack_words_after_stage1, (
            "an empty stage must acknowledge nothing"
        )
        assert s.host.counters()["leases_acked"] == leases_acked_before + len(experts)
        assert s.delivered(experts), "stage 2 must not have touched the poisoned slot"

        s.finalize()
        _cuda_ready()
        assert s.keep.item() == 1.0
    finally:
        s.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
