"""Task 5 item 4, on a real GPU: failure before readiness, timeout before copy, failure after a subset packed,
dependent compute suppressed on a fatal demand error, and a skipped copy that emits no acknowledgement.

What the kernel-only file (``test_exl3_lease_kernels_cuda.py``) and the graph file
(``test_exl3_ram_miss_graph_gpu.py``) already pin, and what they do not:

* Kernel level, hand-driven: every refusal (timeout, failed status, six identity faults, shutdown, a fatal word,
  the sticky flag) leaves ``go_count`` zero and publishes a terminal. Acknowledgement absence with a *stale but
  committed-looking* ``lane_ctx`` is pinned for three of those (identity, timeout, sticky) only.
* Graph level: a forced timeout leaves ``keep`` at 0, ``go_count`` at 0 and the scratch rows untouched. It does not
  look at the layer's OUTPUT, at any LATER layer, or at the acknowledgement words.
* No test drives a REAL service read that fails after other rows of the same request have packed, through the
  real wait, copy and acknowledgement kernels. The tier test does it on the CPU and stops at publication.

Run on divix01 under ``gpu-run.sh`` with PYTHONPATH pointing at the tree under test, and SGLANG_EXL3_SRC and
SGLANG_EXL3_BUILD_DIR set for the graph tests (they build the exllamav3 extension).
"""

import errno
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_exl3_lease_kernels_cuda import (  # noqa: E402
    LANES,
    REASON,
    STATUS,
    Block,
    Rig,
    Service,
    _committed,
    _cuda_ready,
    _idx,
    _set_page_word,
    _tagged,
    lease,
    page_word,
)

NO_ACKS = bytes(lease.RING * LANES * lease.LANE_ACK_BYTES)


def _dest_untouched(s):
    return all(not s.dest[n].any().item() for n in s.names)


def _slab_holds(s, expert, want):
    """Whether some slot of row 0's slabs holds ``expert``'s bytes, every streamed tensor."""
    capacity = s.slabs[0][s.names[0]].shape[0]
    for slot in range(capacity):
        if all(
            torch.equal(s.slabs[0][n][slot].view(torch.uint8), want[expert][n].view(torch.uint8)) for n in s.names
        ):
            return True
    return False


# ---------------------------------------------------------------------------------------------------------------
# A refused request emits no acknowledgement, for EVERY reason the wait kernel refuses for.
# ---------------------------------------------------------------------------------------------------------------
class TestSkippedCopyEmitsNoAcknowledgement:
    @pytest.mark.parametrize("how", ["failed_status", "shutdown", "fatal_word", "count_over_lanes"])
    def test_a_refused_request_emits_no_acknowledgement_even_with_a_stale_lane_context(self, how):
        """The existing test of this name covers identity, timeout and sticky. These are the other refusals: the
        service reporting the request failed, the header's shutdown, a fatal word already on the page, and a plan
        of more lanes than the lease block names."""
        rig = Rig(timeout_ms=50, lanes=LANES + 3 if how == "count_over_lanes" else LANES)
        first = _committed(rig, [7, 2, 11], slots=[5, 0, 3], slot_generations=[4, 1, 6])
        rig.ack()
        assert rig.block.ack_word(_idx(first), 0) != 0
        start = rig.block.d(lease.LANE_ACK)
        rig.raw[start : start + lease.RING * LANES * lease.LANE_ACK_BYTES].zero_()
        stale = rig.dev.lane_ctx.clone()  # a committed-looking context is still in device memory
        assert rig.dev.lane_ctx.abs().sum().item() > 0
        experts = [7, 2, 11]
        if how == "count_over_lanes":
            experts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        rig.plan(experts)
        seq = rig.post()
        if how == "failed_status":
            rig.serve(seq, [], status=STATUS["failed"])
        elif how == "shutdown":
            rig.block.set_u32(lease.HEADER["shutdown"], 1)
        elif how == "fatal_word":
            _set_page_word(rig.page, "fatal", 99)
        rig.wait()
        assert rig.go() == 0 and rig.keep.item() == 0.0
        assert torch.equal(rig.dev.lane_ctx, stale), "the wait kernel left the old context in place"
        fatal_after_wait = page_word(rig.page, "fatal")
        rig.keep.fill_(1.0)  # so a keep write by the ack kernel would show
        rig.ack()
        assert rig.block.ack_area() == NO_ACKS, "no acknowledgement at all"
        assert page_word(rig.page, "fatal") == fatal_after_wait, "and no fatal word from the ack kernel"
        assert rig.keep.item() == 1.0, "and no keep write"


# ---------------------------------------------------------------------------------------------------------------
# Failure after a subset packed, timeout before copy: the real service, the real copy kernel.
# ---------------------------------------------------------------------------------------------------------------
class TestRealServiceFailures:
    def test_a_read_that_fails_after_other_rows_packed_copies_nothing_and_acknowledges_nothing(self, tmp_path):
        s = Service(tmp_path)
        try:
            experts = [3, 5, 7]
            want = s.expected(experts)
            # hold_ordinal withholds row 2's completions until rows 0 and 1 have packed, so the failure lands
            # AFTER a subset of the request is in the pinned slabs whatever order the kernel completes reads in.
            s.host.inject_fault(part=0, part_error=errno.EIO, ordinal=2, hold_ordinal=2)
            s.plan(experts)
            for n in s.names:
                s.dest[n].zero_()
            s.step()
            _cuda_ready()
            assert s.until(lambda: s.host.busy_since_ns() == 0, timeout_s=15.0)
            counters = s.host.counters()
            # Not vacuous: the two rows before the failure really did pack, and the failed row never did.
            assert _slab_holds(s, 3, want) and _slab_holds(s, 5, want), "rows 0 and 1 packed before row 2 failed"
            assert not _slab_holds(s, 7, want), "the failing row never packed"
            # The property: nothing published, so the device refused the whole request and nothing was copied.
            assert counters["read_errors"] == 1 and counters["rows_read"] == 0, counters
            assert s.keep.item() == 0.0 and s.dev.go_count.item() == 0
            assert s.host.fatal_seq() != 0
            assert _dest_untouched(s), "the copy kernel read none of the packed rows"
            terminal = s.block.terminal(0)
            assert terminal["word"] == _tagged(lease.TERMINAL_TAG, 1) and terminal["mask"] == 0b111
            assert terminal["reason"] == REASON["failed"]
            assert s.block.ack_area() == NO_ACKS, "a skipped copy emits no acknowledgement"
            assert counters["leases_granted"] == 0 and counters["leases_acked"] == 0, counters
            assert s.leases() == [0] * len(s.leases())
            assert not any(s.host.contains(0, e) for e in experts)
        finally:
            s.host.inject_fault()
            s.close()

    def test_the_control_without_the_fault_the_same_request_copies_and_is_acknowledged(self, tmp_path):
        """What the test above would show if the fault were not there: the same experts, delivered and acknowledged."""
        s = Service(tmp_path)
        try:
            experts = [3, 5, 7]
            s.plan(experts)
            s.step()
            _cuda_ready()
            assert s.keep.item() == 1.0 and s.dev.go_count.item() == 3
            assert s.until(lambda: s.host.counters()["leases_acked"] == 3), s.host.counters()
            want = s.expected(experts)
            for lane, expert in enumerate(experts):
                for n in s.names:
                    assert torch.equal(s.dest[n][lane].cpu().view(torch.uint8), want[expert][n].view(torch.uint8))
            assert s.block.ack_area() != NO_ACKS
        finally:
            s.close()

    def test_a_timed_out_request_stays_unacknowledged_when_the_service_finishes_late_and_the_next_one_is_refused(
        self, tmp_path
    ):
        s = Service(tmp_path, timeout_ms=50)
        try:
            s.host.inject(delay_s=1.0)
            s.plan([7, 9])
            s.step()
            _cuda_ready()
            assert s.keep.item() == 0.0 and s.dev.go_count.item() == 0 and s.host.fatal_seq() != 0
            # The service finishes its delayed read AFTER the device gave up. Whatever it then does, no
            # acknowledgement appears, no lease survives, and no lane is ever counted as consumed.
            assert s.until(lambda: s.host.busy_since_ns() == 0, timeout_s=15.0)
            time.sleep(0.2)  # let retire_leases run over whatever the late serve left
            counters = s.host.counters()
            print("late-serve counters", counters)
            assert s.block.ack_area() == NO_ACKS
            assert counters["leases_acked"] == 0 and s.leases() == [0] * len(s.leases()), counters
            assert counters["leases_granted"] == counters["leases_voided"], counters
            # A later request on the same page is refused without touching the tier or the destination.
            s.host.inject(delay_s=0.0)
            for n in s.names:
                s.dest[n].zero_()
            s.keep.fill_(1.0)
            s.plan([1, 2])
            s.step()
            _cuda_ready()
            assert s.keep.item() == 0.0 and s.dev.go_count.item() == 0
            assert _dest_untouched(s)
            assert s.block.ack_area() == NO_ACKS
            assert s.host.counters()["leases_acked"] == 0
        finally:
            s.host.inject(delay_s=0.0)
            s.close()


# ---------------------------------------------------------------------------------------------------------------
# Dependent compute: the fused MoE of the failed layer and of every later layer runs nothing.
# ---------------------------------------------------------------------------------------------------------------
def _run_two_layers_with_a_fatal_error(tmp_path, lease_on, cause):
    from test_exl3_ram_miss_graph_gpu import HIDDEN, TOP_K, _layers, _step_route

    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    pairs, service, checks = _layers(tmp_path, timeout_ms=100, lease=lease_on, num_layers=2)
    try:
        gen = torch.Generator(device="cpu").manual_seed(5)
        inputs = []
        for layer_id in range(2):
            x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
            ids = torch.tensor([_step_route(layer_id, -1)], device="cuda", dtype=torch.int32)
            inputs.append((x, weights, ids))

        def run():
            return [
                Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
                for (layer, streamer), (x, weights, ids) in zip(pairs, inputs)
            ]

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = run()
        # Baseline replay: both layers' routes are resident, nothing fails, the output is real.
        graph.replay()
        torch.cuda.synchronize()
        for layer_id, (_, streamer) in enumerate(pairs):
            assert streamer.row_backend.keep.item() == 1.0, (layer_id, service.host.counters())
        baseline = [o.clone() for o in outs]
        assert all(b.float().abs().sum().item() > 0 for b in baseline), "the baseline computes something"
        block = Block(service.host.lease_block, service.host.lease_layout) if lease_on else None
        if lease_on:
            deadline = time.perf_counter() + 10.0
            while time.perf_counter() < deadline:
                c = service.host.counters()
                if c["leases_granted"] > 0 and c["leases_acked"] == c["leases_granted"]:
                    break
                time.sleep(0.005)
            acked_before = service.host.counters()["leases_acked"]
            acks_before = block.ack_area()
        # Layer 0 now routes to experts the tier does not hold and the service fails to read; layer 1's route is
        # the same resident one as in the baseline.
        route0 = _step_route(0, 0)
        hot0, tier0 = pairs[0][1].hot_cache.slot_to_expert, pairs[0][1].pinned_host_cache._lru
        assert [e for e in route0 if e not in hot0 and e not in tier0], "layer 0 misses"
        inputs[0][2].copy_(torch.tensor([route0], device="cuda", dtype=torch.int32))
        if cause == "timeout":
            service.host.inject(delay_s=10.0)
        else:
            service.host.inject_fault(part=0, part_error=errno.EIO)
        started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        assert time.perf_counter() - started < 5.0, "the fatal error fails the replay, it does not hang it"
        result = {"outs": [o.clone() for o in outs], "baseline": baseline}
        result["keeps"] = [streamer.row_backend.keep.item() for _, streamer in pairs]
        result["expert_counts"] = [int(layer._exl3_fused_moe.expert_count.count_nonzero()) for layer, _ in pairs]
        result["fatal"] = service.host.fatal_seq()
        if lease_on:
            result["go_count"] = service.device_side.go_count.item()
            deadline = time.perf_counter() + 20.0
            while time.perf_counter() < deadline and service.host.busy_since_ns() != 0:
                time.sleep(0.01)
            time.sleep(0.2)
            result["acked_delta"] = service.host.counters()["leases_acked"] - acked_before
            result["acks_unchanged"] = block.ack_area() == acks_before
            result["leases_left"] = sum(sum(info[2] for info in service.host.slot_info(r)) for r in range(2))
        with pytest.raises(RuntimeError, match="exl3 RAM miss"):
            for check in checks:
                check()
        return result
    finally:
        service.host.inject(delay_s=0.0)
        service.host.inject_fault()
        service.shutdown()
        import sglang.srt.layers.moe.exl3_ram_miss as service_module

        service_module.Exl3RamMissService._instance = None


@pytest.mark.parametrize("cause", ["timeout", "read_fault"])
@pytest.mark.parametrize("lease_on", [False, True], ids=["leases_off", "leases_on"])
def test_a_fatal_demand_error_drops_the_failed_layer_and_every_later_layer_and_runs_no_expert(
    tmp_path, lease_on, cause
):
    r = _run_two_layers_with_a_fatal_error(tmp_path, lease_on, cause)
    assert r["fatal"] != 0
    assert r["keeps"] == [0.0, 0.0], "layer 0 failed, layer 1 was dropped behind it"
    # The dependent compute: no expert ran (the fused MoE's per-slot counts are all zero) and the output is
    # exactly zero, not a stale or partial result. Layer 1's route was resident and gave a real output a
    # replay earlier, so its zero is the fatal error's doing.
    assert r["expert_counts"] == [0, 0]
    for layer_id, (out, base) in enumerate(zip(r["outs"], r["baseline"])):
        assert int(out.count_nonzero()) == 0, layer_id
        assert int(base.count_nonzero()) > 0, layer_id
    if lease_on:
        assert r["go_count"] == 0
        assert r["acked_delta"] == 0 and r["acks_unchanged"], "a refused replay acknowledged nothing"
        assert r["leases_left"] == 0
