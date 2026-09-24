"""The lease protocol's device kernels on a real GPU (LEASE_PROTOCOL.md 6.3, 7.3, 7.4, 13).

Two layers of test, neither of which is the CPU LeaseSim:

* ``TestHandDriven``: the real post, wait and acknowledgement kernels against a service that is a few lines of
  Python in this file writing the lease block by hand. It can produce a stale generation, a wrong expert or a bad
  slot on purpose, which the real service never would.
* ``TestServiceEndToEnd``: the real post, wait, copy and acknowledgement kernels against the real C++ service
  thread in lease mode, with the real copy kernel between wait and acknowledgement.

What this file does NOT establish: the memory-ordering argument of 6.4 (that every load of the source bytes has
returned before the acknowledgement store) rests on CUDA stream/graph semantics, and no test can show a violation
of it absent. See the report that came with this file.

Run on divix01 under ``gpu-run.sh`` (it holds cc-gpu.lock) with PYTHONPATH pointing at the tree under test.
"""

import threading
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import exl3_lease_block as lease  # noqa: E402
from sglang.kernels.ops.moe.exl3_ram_miss import (  # noqa: E402
    DEMAND_RECORDS,
    DEMAND_RING,
    PAGE_BYTES,
    RECORD_BYTES,
    STATE_WORDS,
    STATUS,
    WORDS,
    Exl3RamMissDevice,
    Exl3RamMissHost,
    new_page,
    page_word,
)

REC_STATUS = 10  # the record's u16 status, kRecStatus
LAYERS, EXPERTS, CAPACITY = 2, 16, 8
TOP_K = 6
LANES = lease.LANES
REASON = lease.TERMINAL_REASONS
WORD_MASK = (1 << 64) - 1


def _cuda_ready():
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------------------------
# A block and a page read and written from Python. The lease block is one pinned tensor; the kernels see it through UVA.
# ---------------------------------------------------------------------------------------------------------------
class Block:
    def __init__(self, block, layout):
        self.block, self.layout = block, layout

    def u64(self, offset):
        return int(self.block[offset : offset + 8].view(torch.int64)[0]) & WORD_MASK

    def set_u64(self, offset, value):
        value &= WORD_MASK
        self.block[offset : offset + 8].view(torch.int64)[0] = value - (1 << 64) if value >= (1 << 63) else value

    def u32(self, offset):
        return int(self.block[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF

    def i32(self, offset):
        return int(self.block[offset : offset + 4].view(torch.int32)[0])

    def set_u32(self, offset, value):
        self.block[offset : offset + 4].view(torch.int32)[0] = value - (1 << 32) if value >= (1 << 31) else value

    def d(self, offset):
        return self.layout.d_offset + offset

    # area S, written by the service
    def rr(self, idx, lane):
        return lease.ROW_RESULT + (idx * LANES + lane) * lease.ROW_RESULT_BYTES

    def slot_gen_offset(self, row, slot):
        return lease.SLOT_GEN + 4 * (self.layout.slot_gen_base[row] + slot)

    # area D, written by the device
    def lane_request(self, idx):
        base = self.d(lease.LANE_REQUEST + idx * lease.LANE_REQUEST_BYTES)
        gen = self.u64(base)
        f = lease.LANE_REQUEST_FIELDS
        return {
            "word": gen,
            "count": self.u32(base + f["count"]),
            "row": self.u32(base + f["row"]),
            "experts": [self.i32(base + f["expert"] + 4 * i) for i in range(LANES)],
        }

    def ack_word(self, idx, lane):
        return self.u64(self.d(lease.LANE_ACK + (idx * LANES + lane) * lease.LANE_ACK_BYTES))

    def ack_area(self):
        start = self.d(lease.LANE_ACK)
        return bytes(self.block[start : start + lease.RING * LANES * lease.LANE_ACK_BYTES].tolist())

    def terminal(self, idx):
        base = self.d(lease.TERMINAL + idx * lease.TERMINAL_BYTES)
        f = lease.TERMINAL_FIELDS
        return {
            "mask": self.u32(base + f["skipped_mask"]),
            "reason": self.u32(base + f["reason"]),
            "word": self.u64(base + f["gen"]),
        }


def _tagged(tag, generation):
    return (tag << 56) | generation


def _set_page_word(page, name, value):
    offset = WORDS[name]
    page[offset : offset + 4].view(torch.int32)[0] = value - (1 << 32) if value >= (1 << 31) else value


def _idx(seq):
    return (seq - 1) % DEMAND_RECORDS


class Rig:
    """One device with a lease block, and the buffers the kernels use, on a page nobody serves."""

    def __init__(self, *, timeout_ms=200, lanes=TOP_K, demand_head=0):
        self.page = new_page(pin=True)
        if demand_head:
            _set_page_word(self.page, "demand_head", demand_head)
        self.slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
        self.layout = lease.lease_layout([CAPACITY] * LAYERS)
        self.raw = lease.new_lease_block(self.layout, pin=True)
        self.block = Block(self.raw, self.layout)
        # The row table is what the service writes at open; the wait kernel reads each row's capacity from it.
        for row in range(LAYERS):
            entry = lease.ROW_TABLE + row * lease.ROW_TABLE_ENTRY_BYTES
            self.block.set_u32(entry, self.layout.slot_gen_base[row])
            self.block.set_u32(entry + 4, self.layout.capacities[row])
        self.dev = Exl3RamMissDevice(
            self.page, self.slot_map, device="cuda", layers=LAYERS, timeout_ms=timeout_ms, advise=False,
            lease_block=self.raw, lease_layout=self.layout,
        )
        self.planned = torch.zeros(max(lanes, LANES), dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((LANES,), -1, dtype=torch.int64, device="cuda")
        self.host_rows = torch.zeros(lanes, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")

    def plan(self, experts):
        self.planned.fill_(-1)
        self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1)
        self.routes[: min(len(experts), LANES)] = torch.tensor(experts[:LANES], dtype=torch.int64)

    def post(self, row=0):
        self.dev.post(row, self.planned, self.count, self.routes, -1)
        _cuda_ready()
        return int(self.dev.stats()["posted"]) & 0xFFFFFFFF

    def state(self, name):
        return int(self.dev.state[STATE_WORDS[name]])

    def generation(self, seq):
        return ((self.state("pending_epoch") & 0xFFFFFFFF) << 32) | seq

    def write_lane(self, seq, lane, expert, slot, slot_generation, *, gen=None, tag=lease.READY):
        """One RowResult, payload first and the tagged ready word last, as the service writes it (tag 0: unpublished)."""
        gen = self.generation(seq) if gen is None else gen
        base = self.block.rr(_idx(seq), lane)
        f = lease.ROW_RESULT_FIELDS
        self.block.set_u32(base + f["slot_generation"], slot_generation)
        self.block.set_u32(base + f["host_slot"], slot & 0xFFFFFFFF)
        self.block.set_u32(base + f["expert"], expert)
        self.block.set_u64(base + f["ready"], _tagged(tag, gen) if tag else 0)

    def serve(self, seq, lanes, *, gen=None, tag=lease.READY, status=STATUS["served"], done=True):
        """What the real service does for request ``seq``: RowResults, then the record status and demand_done.

        ``lanes``: per lane ``(expert, host_slot, slot_generation)``."""
        for lane, (expert, slot, slot_generation) in enumerate(lanes):
            self.write_lane(seq, lane, expert, slot, slot_generation, gen=gen, tag=tag)
        self.finish(seq, status, done)

    def finish(self, seq, status=STATUS["served"], done=True):
        record = DEMAND_RING + _idx(seq) * RECORD_BYTES
        self.page[record + REC_STATUS : record + REC_STATUS + 2].view(torch.int16)[0] = status
        if done:
            _set_page_word(self.page, "demand_done", seq)

    def wait(self, row=0):
        self.dev.wait(row, self.planned, self.count, self.host_rows, self.keep, self.ram_miss)
        _cuda_ready()

    def ack(self):
        self.dev.ack(self.keep)
        _cuda_ready()

    def go(self):
        return int(self.dev.go_count[0])


def _committed(rig, experts, *, row=0, slots=None, slot_generations=None):
    rig.plan(experts)
    seq = rig.post(row)
    slots = list(range(len(experts))) if slots is None else slots
    slot_generations = [3] * len(experts) if slot_generations is None else slot_generations
    rig.serve(seq, list(zip(experts, slots, slot_generations)))
    for expert_slot, gen in zip(slots, slot_generations):
        rig.block.set_u32(rig.block.slot_gen_offset(row, expert_slot), gen)  # the mapped SlotGen the service mirrors
    rig.wait(row)
    return seq


# ---------------------------------------------------------------------------------------------------------------
class TestHandDriven:
    def test_post_writes_the_lane_request_and_arms_a_request_with_lanes_only_in_lease_mode(self):
        rig = Rig()
        rig.slot_map[0, 3] = 0  # both planned experts already in RAM: today's post would not arm
        rig.slot_map[0, 5] = 1
        rig.plan([3, 5])
        seq = rig.post()
        assert seq == 1 and rig.state("pending") == 1, "lease mode arms every request with lanes"
        request = rig.block.lane_request(0)
        assert request["word"] == _tagged(lease.DEMAND_TAG, 1)
        assert (request["count"], request["row"]) == (2, 0) and request["experts"] == [3, 5] + [-1] * 6
        assert rig.state("epoch") == 0 and rig.state("pending_epoch") == 0

        legacy_page = new_page(pin=True)
        legacy = Exl3RamMissDevice(legacy_page, rig.slot_map, device="cuda", layers=LAYERS, timeout_ms=50, advise=False)
        legacy.post(0, rig.planned, rig.count, rig.routes, -1)
        _cuda_ready()
        assert int(legacy.state[STATE_WORDS["pending"]]) == 0, "without a lease block the post arms as before"
        assert page_word(legacy_page, "demand_head") == 1

    def test_the_lane_request_of_a_plan_of_more_than_eight_lanes_is_clamped_and_the_wait_refuses_it(self):
        rig = Rig(lanes=9)
        rig.planned = torch.arange(9, dtype=torch.int64, device="cuda")
        rig.count.fill_(9)
        rig.routes.fill_(-1)
        seq = rig.post()
        assert rig.block.lane_request(0)["count"] == LANES
        rig.serve(seq, [(e, e % CAPACITY, 1) for e in range(LANES)])
        rig.wait()
        assert rig.go() == 0 and rig.keep.item() == 0.0
        terminal = rig.block.terminal(0)
        assert terminal["word"] == _tagged(lease.TERMINAL_TAG, 1) and terminal["mask"] == 0xFF
        assert terminal["reason"] == REASON["count"] and page_word(rig.page, "fatal") == 0xFFFFFFFF

    def test_the_epoch_advances_when_the_sequence_wraps_and_names_the_generation(self):
        rig = Rig(demand_head=0xFFFFFFFE)
        rig.plan([1])
        assert rig.post() == 0xFFFFFFFF
        assert rig.state("epoch") == 0 and rig.state("pending_epoch") == 0
        request = rig.block.lane_request(_idx(0xFFFFFFFF))
        assert request["word"] == _tagged(lease.DEMAND_TAG, 0xFFFFFFFF)
        assert rig.post() == 1
        assert rig.state("epoch") == 1 and rig.state("pending_epoch") == 1
        assert rig.block.lane_request(0)["word"] == _tagged(lease.DEMAND_TAG, (1 << 32) | 1)
        # The wait and the acknowledgement use the generation the request's own post stored.
        rig.serve(1, [(1, 4, 9)], gen=(1 << 32) | 1)
        rig.block.set_u32(rig.block.slot_gen_offset(0, 4), 9)
        rig.wait()
        assert rig.go() == 1 and rig.dev.lane_ctx[0, 0].item() == (1 << 32) | 1
        rig.ack()
        assert rig.block.ack_word(0, 0) == _tagged(lease.CONSUMED, (1 << 32) | 1)

    def test_a_valid_request_commits_go_count_host_rows_and_lane_context(self):
        rig = Rig()
        seq = _committed(rig, [7, 2, 11], slots=[5, 0, 3], slot_generations=[4, 1, 6])
        assert rig.go() == 3 and rig.keep.item() == 1.0 and rig.ram_miss.item() == 0
        assert rig.host_rows.tolist() == [5, 0, 3, 0, 0, 0]
        ctx = rig.dev.lane_ctx.cpu().tolist()
        gen = rig.generation(seq)
        assert ctx[:3] == [[gen, 4, 0, 5], [gen, 1, 0, 0], [gen, 6, 0, 3]]
        assert rig.block.terminal(_idx(seq))["word"] == 0 and rig.state("sticky") == 0
        assert page_word(rig.page, "fatal") == 0

    def test_a_request_without_lanes_commits_zero_and_keeps_the_layer(self):
        rig = Rig()
        rig.plan([])
        rig.post()
        assert rig.state("pending") == 0, "a touch-only record is not armed"
        rig.wait()
        assert rig.go() == 0 and rig.keep.item() == 1.0

    REFUSALS = {
        # name: the keyword arguments that make lane 1 wrong
        "stale_generation": dict(gen_delta=1 << 32),
        "failed_tag": dict(tag=lease.LOADING),
        "unpublished_ready": dict(tag=0),
        "wrong_expert": dict(expert=4),
        "negative_slot": dict(slot=-1),
        "slot_past_capacity": dict(slot=CAPACITY),
    }

    @pytest.mark.parametrize("name", sorted(REFUSALS))
    def test_a_lane_that_fails_validation_refuses_the_whole_request_and_publishes_a_terminal(self, name):
        bad = self.REFUSALS[name]
        rig = Rig()
        rig.plan([2, 6, 9])
        seq = rig.post()
        gen = rig.generation(seq)
        rig.write_lane(seq, 0, 2, 0, 1)
        rig.write_lane(
            seq, 1, bad.get("expert", 6), bad.get("slot", 1), 1, gen=gen + bad.get("gen_delta", 0), tag=bad.get("tag", lease.READY)
        )
        rig.write_lane(seq, 2, 9, 2, 1)
        rig.finish(seq)
        rig.wait()
        assert rig.go() == 0, "fail closed"
        assert rig.keep.item() == 0.0 and rig.state("sticky") == 1
        assert rig.host_rows.tolist() == [0] * TOP_K
        terminal = rig.block.terminal(_idx(seq))
        assert terminal["word"] == _tagged(lease.TERMINAL_TAG, gen), "the terminal names this request"
        assert terminal["mask"] == 0b111 and terminal["reason"] == REASON["identity"]
        assert page_word(rig.page, "fatal") == 0xFFFFFFFF

    def test_a_status_other_than_served_fails_closed_with_a_failed_terminal(self):
        rig = Rig()
        rig.plan([2, 6])
        seq = rig.post()
        rig.serve(seq, [], status=STATUS["failed"])
        rig.wait()
        assert rig.go() == 0 and rig.keep.item() == 0.0
        terminal = rig.block.terminal(_idx(seq))
        assert terminal["word"] == _tagged(lease.TERMINAL_TAG, rig.generation(seq)) and terminal["mask"] == 0b11
        assert terminal["reason"] == REASON["failed"] and page_word(rig.page, "fatal") == seq
        assert rig.dev.stats()["failures"] == 1

    def test_a_timeout_publishes_the_terminal_and_the_fatal_word(self):
        rig = Rig(timeout_ms=50)
        rig.plan([2, 6])
        seq = rig.post()
        start = time.perf_counter()
        rig.wait()
        assert time.perf_counter() - start < 2.0
        assert rig.go() == 0 and rig.keep.item() == 0.0
        terminal = rig.block.terminal(_idx(seq))
        assert terminal["word"] == _tagged(lease.TERMINAL_TAG, rig.generation(seq)) and terminal["mask"] == 0b11
        assert terminal["reason"] == REASON["timeout"] and page_word(rig.page, "fatal") == seq
        assert rig.dev.stats()["timeouts"] == 1

    def test_shutdown_ends_the_wait_promptly_without_a_fatal_word(self):
        rig = Rig(timeout_ms=5000)
        rig.plan([2, 6])
        seq = rig.post()
        threading.Timer(0.05, lambda: rig.block.set_u32(lease.HEADER["shutdown"], 1)).start()
        start = time.perf_counter()
        rig.wait()
        assert time.perf_counter() - start < 2.0, "the wait must not run to its 5 s timeout (D4)"
        assert rig.go() == 0 and rig.keep.item() == 0.0 and page_word(rig.page, "fatal") == 0
        terminal = rig.block.terminal(_idx(seq))
        assert terminal["reason"] == REASON["aborted"] and terminal["mask"] == 0b11
        assert rig.dev.stats()["timeouts"] == 0

    def test_a_fatal_word_raised_while_waiting_ends_the_wait_promptly(self):
        rig = Rig(timeout_ms=5000)
        rig.plan([2])
        seq = rig.post()
        threading.Timer(0.05, lambda: _set_page_word(rig.page, "fatal", 99)).start()
        start = time.perf_counter()
        rig.wait()
        assert time.perf_counter() - start < 2.0
        assert rig.go() == 0 and page_word(rig.page, "fatal") == 99
        assert rig.block.terminal(_idx(seq))["reason"] == REASON["aborted"]

    def test_a_request_posted_after_the_sticky_flag_names_no_generation_and_publishes_no_terminal(self):
        rig = Rig()
        rig.dev.state[STATE_WORDS["sticky"]] = 1
        rig.plan([2, 6])
        assert rig.post() == 0 or rig.state("pending") == 0
        rig.wait()
        assert rig.go() == 0 and rig.keep.item() == 0.0
        assert all(rig.block.terminal(i)["word"] == 0 for i in range(DEMAND_RECORDS))

    def test_lanes_with_nothing_armed_are_refused_and_nothing_is_named(self):
        rig = Rig()
        rig.plan([2])
        rig.post()
        rig.dev.state[STATE_WORDS["pending"]] = 0  # a post that did not arm, which the lease post never does
        rig.wait()
        assert rig.go() == 0 and rig.keep.item() == 0.0 and page_word(rig.page, "fatal") == 0xFFFFFFFF
        assert all(rig.block.terminal(i)["word"] == 0 for i in range(DEMAND_RECORDS))

    def test_go_count_is_zero_on_entry_even_when_the_previous_request_committed(self):
        rig = Rig()
        _committed(rig, [1, 2])
        assert rig.go() == 2
        rig.dev.state[STATE_WORDS["sticky"]] = 1
        rig.wait()
        assert rig.go() == 0

    # ---- the acknowledgement kernel ----

    def test_the_acknowledgement_kernel_acknowledges_exactly_the_committed_lanes(self):
        rig = Rig()
        seq = _committed(rig, [7, 2, 11], slots=[5, 0, 3], slot_generations=[4, 1, 6])
        idx = _idx(seq)
        assert rig.block.ack_area() == bytes(lease.RING * LANES * lease.LANE_ACK_BYTES), "nothing before the ack kernel"
        rig.ack()
        gen = rig.generation(seq)
        for lane in range(3):
            assert rig.block.ack_word(idx, lane) == _tagged(lease.CONSUMED, gen), lane
        for lane in range(3, LANES):
            assert rig.block.ack_word(idx, lane) == 0, lane
        assert sum(1 for i in range(DEMAND_RECORDS) for lane in range(LANES) if rig.block.ack_word(i, lane)) == 3
        assert rig.keep.item() == 1.0 and page_word(rig.page, "fatal") == 0

    def test_a_slot_generation_that_moved_is_acknowledged_violated_and_drops_the_layer(self):
        rig = Rig()
        seq = _committed(rig, [7, 2, 11], slots=[5, 0, 3], slot_generations=[4, 1, 6])
        rig.block.set_u32(rig.block.slot_gen_offset(0, 0), 2)  # lane 1's slot was rewritten after the grant
        rig.ack()
        idx, gen = _idx(seq), rig.generation(seq)
        assert rig.block.ack_word(idx, 0) == _tagged(lease.CONSUMED, gen)
        assert rig.block.ack_word(idx, 1) == _tagged(lease.VIOLATED, gen)
        assert rig.block.ack_word(idx, 2) == _tagged(lease.CONSUMED, gen)
        assert rig.keep.item() == 0.0 and page_word(rig.page, "fatal") == seq

    def test_the_slot_generation_is_read_from_the_lanes_own_row(self):
        rig = Rig()
        seq = _committed(rig, [7], row=1, slots=[2], slot_generations=[8])
        other_row = rig.block.slot_gen_offset(0, 2)
        rig.block.set_u32(other_row, 123)  # the same slot number in another row must not matter
        rig.ack()
        assert rig.block.ack_word(_idx(seq), 0) == _tagged(lease.CONSUMED, rig.generation(seq))

    @pytest.mark.parametrize("how", ["identity", "timeout", "sticky"])
    def test_a_refused_request_emits_no_acknowledgement_even_with_a_stale_lane_context(self, how):
        """The property 7.4 asks for by construction: one word gates the copy and the acknowledgement."""
        rig = Rig(timeout_ms=50)
        first = _committed(rig, [7, 2, 11], slots=[5, 0, 3], slot_generations=[4, 1, 6])
        rig.ack()
        assert rig.block.ack_word(_idx(first), 0) != 0
        # Wipe the acknowledgements, so any word after this point was written by the next ack kernel launch.
        start = rig.block.d(lease.LANE_ACK)
        rig.raw[start : start + lease.RING * LANES * lease.LANE_ACK_BYTES].zero_()
        stale = rig.dev.lane_ctx.clone()  # a plausible, committed-looking context is still in device memory
        keep_before = rig.keep.item()
        fatal_before = page_word(rig.page, "fatal")
        rig.plan([7, 2, 11])
        seq = rig.post()
        if how == "identity":
            rig.serve(seq, [(7, 5, 4), (99, 0, 1), (11, 3, 6)])
        elif how == "sticky":
            rig.dev.state[STATE_WORDS["sticky"]] = 1
        rig.wait()
        assert rig.go() == 0
        assert torch.equal(rig.dev.lane_ctx, stale), "the wait kernel left the old context in place"
        keep_after_wait, fatal_after_wait = rig.keep.item(), page_word(rig.page, "fatal")
        rig.keep.fill_(1.0 if how == "sticky" else keep_after_wait)
        rig.ack()
        assert rig.block.ack_area() == bytes(lease.RING * LANES * lease.LANE_ACK_BYTES), "no acknowledgement at all"
        assert page_word(rig.page, "fatal") == fatal_after_wait, "and no fatal word from the ack kernel"
        assert rig.keep.item() == (1.0 if how == "sticky" else keep_after_wait), "and no keep write"
        assert keep_before == 1.0 and fatal_before == 0

    def test_the_acknowledgement_kernel_follows_go_count_and_nothing_else(self):
        rig = Rig()
        seq = _committed(rig, [7, 2, 11], slots=[5, 0, 3], slot_generations=[4, 1, 6])
        rig.dev.go_count.fill_(1)  # a smaller commit: only lane 0 is in the set
        rig.ack()
        idx = _idx(seq)
        assert rig.block.ack_word(idx, 0) != 0 and rig.block.ack_word(idx, 1) == 0 and rig.block.ack_word(idx, 2) == 0

    def test_the_acknowledgement_kernel_needs_a_lease_device(self):
        page = new_page(pin=True)
        slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
        dev = Exl3RamMissDevice(page, slot_map, device="cuda", layers=LAYERS, timeout_ms=50, advise=False)
        with pytest.raises(RuntimeError, match="lease"):
            dev.ack(torch.ones(1, dtype=torch.float32, device="cuda"))


# ---------------------------------------------------------------------------------------------------------------
def _close(host, slabs):
    from sglang.srt.layers.moe.expert_host_tier import release_host_slabs

    try:
        if host is not None:
            host.stop()
    finally:
        release_host_slabs([slab for names in slabs.values() for slab in names.values()])


class Service:
    def __init__(self, tmp_path, *, timeout_ms=2000, advise=False):
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
        self.host_rows = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest = {
            n: torch.zeros((TOP_K,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda") for n in self.names
        }
        from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments

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

    def step(self, row=0, next_row=-1):
        """post, wait, the real copy kernel (count = go_count), acknowledgement: one layer, one stream."""
        self.step_with(self.dev, row=row, next_row=next_row)

    def step_with(self, dev, row=0, next_row=-1):
        """The same chain driven by ``dev``. ``go_count`` and ``lane_ctx`` belong to the device object, so a second
        device -- built after the first has posted, continuing the demand sequence from the page's head -- runs whole
        requests of its own while the first device's acknowledgement is still withheld."""
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        dev.post(row, self.planned, self.count, self.routes, next_row)
        dev.wait(row, self.planned, self.count, self.host_rows, self.keep, self.ram_miss)
        copy_expert_row_segments_gpu(self.segments, self.host_rows, self.dest_slots, dev.go_count)
        dev.ack(self.keep)

    def expected(self, experts):
        from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

        source = Exl3ShardRowSource.for_layer(self.layout, 0, self.fmt.segment_map(), direct=False)
        rows = {}
        for expert in sorted(set(experts)):
            rows[expert] = {n: torch.empty(self.specs[n].row_shape, dtype=self.specs[n].dtype) for n in self.names}
            source.read(torch.tensor([expert]), {n: t.unsqueeze(0) for n, t in rows[expert].items()})
        return rows

    def leases(self, row=0):
        return [info[2] for info in self.host.slot_info(row)]

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


def _delivered(s, experts):
    want = s.expected(experts)
    for lane, expert in enumerate(experts):
        for n in s.names:
            assert torch.equal(s.dest[n][lane].cpu().view(torch.uint8), want[expert][n].view(torch.uint8)), (lane, expert, n)


class TestServiceEndToEnd:
    def test_a_request_is_served_copied_and_acknowledged_and_the_service_retires_the_leases(self, service):
        s = service
        s.plan([3, 5])
        s.step()
        _cuda_ready()
        assert s.keep.item() == 1.0 and s.dev.go_count.item() == 2 and s.ram_miss.item() == 0
        _delivered(s, [3, 5])
        assert s.until(lambda: s.host.counters()["leases_acked"] == 2), s.host.counters()
        counters = s.host.counters()
        assert counters["leases_granted"] == 2 and counters["leases_voided"] == 0 and counters["lease_double_signal"] == 0
        assert s.leases() == [0] * CAPACITY and s.host.fatal_seq() == 0
        gen = (0 << 32) | 1
        assert s.block.ack_word(0, 0) == _tagged(lease.CONSUMED, gen) and s.block.ack_word(0, 2) == 0

    def test_a_request_whose_experts_are_all_resident_is_still_leased_and_acknowledged(self, service):
        s = service
        s.plan([3, 5])
        s.step()
        _cuda_ready()
        assert s.until(lambda: s.host.counters()["leases_acked"] == 2)
        rows_before = s.host.counters()["rows_read"]
        s.plan([5, 3, 5])  # hits, with a duplicate lane (LEASE_PROTOCOL.md 9: one lease per consuming lane)
        s.step()
        _cuda_ready()
        assert s.keep.item() == 1.0 and s.dev.go_count.item() == 3
        assert s.until(lambda: s.host.counters()["leases_acked"] == 5), s.host.counters()
        assert s.host.counters()["rows_read"] == rows_before, "hits read nothing"
        assert s.host.counters()["leases_granted"] == 5 and s.leases() == [0] * CAPACITY
        _delivered(s, [5, 3, 5])

    def test_the_whole_chain_captures_in_a_graph_and_every_replay_is_acknowledged(self, service):
        s = service
        s.plan([1, 2])
        s.step()  # warm up outside the capture
        _cuda_ready()
        assert s.until(lambda: s.host.counters()["leases_acked"] == 2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s.step()
        acked = 2
        for experts in ([4], [6, 8], [9, 10, 11], [4, 6, 9, 1]):
            s.plan(experts)
            graph.replay()
            _cuda_ready()
            acked += len(experts)
            assert s.keep.item() == 1.0 and s.dev.go_count.item() == len(experts), experts
            assert s.until(lambda: s.host.counters()["leases_acked"] == acked), (experts, s.host.counters())
            _delivered(s, experts)
        assert s.leases() == [0] * CAPACITY and s.host.fatal_seq() == 0
        assert s.host.counters()["lease_double_signal"] == 0

    def test_a_timeout_gives_the_copy_nothing_to_read_and_no_acknowledgement_and_the_lanes_are_retired(self, tmp_path):
        s = Service(tmp_path, timeout_ms=50)
        try:
            s.host.inject(delay_s=3.0)
            s.plan([7, 9])
            for n in s.names:
                s.dest[n].zero_()
            s.step()
            _cuda_ready()
            assert s.keep.item() == 0.0 and s.dev.go_count.item() == 0 and s.host.fatal_seq() != 0
            assert all(not s.dest[n].any().item() for n in s.names), "the copy kernel read nothing"
            terminal = s.block.terminal(0)
            assert terminal["word"] == _tagged(lease.TERMINAL_TAG, 1) and terminal["mask"] == 0b11
            assert terminal["reason"] == REASON["timeout"]
            assert s.block.ack_area() == bytes(lease.RING * LANES * lease.LANE_ACK_BYTES)
            # The service finishes its delayed read, meets the terminal and does not lease, or voids what it leased.
            assert s.until(lambda: s.host.busy_since_ns() == 0, timeout_s=15.0)
            counters = s.host.counters()
            assert counters["leases_acked"] == 0 and s.leases() == [0] * CAPACITY, counters
            assert counters["leases_granted"] == counters["leases_voided"], counters
        finally:
            s.host.inject(delay_s=0.0)
            s.close()

    def test_advisories_and_leases_coexist(self, tmp_path):
        """Advise on and lease on together (LEASE_PROTOCOL.md 15, and the arming rule the device and the service must
        agree on): an advisory for the next layer is read but never leased, every demand's lanes are leased and
        retired, and nothing is waited on that the service treats as touch-only."""
        s = Service(tmp_path, advise=True)
        try:
            s.plan([9, 10])
            s.step(row=1)  # last_routes[1] = [9, 10]; both are read into row 1's RAM
            _cuda_ready()
            assert s.until(lambda: s.host.counters()["leases_acked"] == 2), s.host.counters()
            for slot, (state, expert, leases, _gen) in enumerate(s.host.slot_info(1)):
                if expert in (9, 10) and state == 2:
                    s.host.release(1, slot)  # out of RAM again, so the next post advises them
            assert not s.host.contains(1, 9) and not s.host.contains(1, 10)
            rows_before = s.host.counters()["rows_read"]
            s.plan([3, 5])
            s.step(row=0, next_row=1)  # demand for row 0, and an advisory for row 1's [9, 10]
            _cuda_ready()
            assert s.keep.item() == 1.0 and s.dev.go_count.item() == 2
            _delivered(s, [3, 5])
            assert s.until(lambda: s.host.counters()["advisories"] >= 1 and s.host.contains(1, 9) and s.host.contains(1, 10))
            assert s.until(lambda: s.host.counters()["leases_acked"] == 4), s.host.counters()
            counters = s.host.counters()
            assert counters["advisory_rows"] == 2 and counters["leases_granted"] == 4, counters  # the advisory took no lease
            assert counters["leases_voided"] == 0 and counters["lease_double_signal"] == 0, counters
            assert s.host.fatal_seq() == 0 and s.leases(0) == [0] * CAPACITY and s.leases(1) == [0] * CAPACITY
            assert counters["rows_read"] - rows_before == 2 + 2  # demand 3, 5 (row 0) and advisory 9, 10 (row 1)
        finally:
            s.close()

    def test_a_slot_rewritten_under_a_committed_copy_is_acknowledged_violated(self, service):
        s = service
        s.plan([3, 5])
        s.dev.post(0, s.planned, s.count, s.routes, -1)
        s.dev.wait(0, s.planned, s.count, s.host_rows, s.keep, s.ram_miss)
        _cuda_ready()
        assert s.dev.go_count.item() == 2
        slot = int(s.dev.lane_ctx[1, 3])
        offset = s.block.slot_gen_offset(0, slot)
        s.block.set_u32(offset, s.block.u32(offset) + 1)  # what a recycle of the slot would do (6.5)
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        copy_expert_row_segments_gpu(s.segments, s.host_rows, s.dest_slots, s.dev.go_count)
        s.dev.ack(s.keep)
        _cuda_ready()
        assert s.keep.item() == 0.0 and s.host.fatal_seq() == 1
        assert s.block.ack_word(0, 0) == _tagged(lease.CONSUMED, 1) and s.block.ack_word(0, 1) == _tagged(lease.VIOLATED, 1)
        assert s.until(lambda: s.host.counters()["leases_acked"] == 2), "a VIOLATED word still retires the lease"


def _resident(s, row=0):
    """expert -> slot for every kReady slot of ``row``."""
    return {expert: slot for slot, (state, expert, _leases, _gen) in enumerate(s.host.slot_info(row)) if state == 2}


class TestDelayedConsumption:
    """Task 5 step 3: delayed GPU consumption under RAM admission pressure, with the real acknowledgement kernel as
    the consumer rather than a CPU stand-in written from the same specification.

    The first device posts, waits and copies, and then does not acknowledge. Its leases stay outstanding for as long
    as this test likes. A second device -- its own ``go_count`` and ``lane_ctx``, continuing the demand sequence from
    the page's head -- then runs six whole requests that fill the tier and force evictions out of it. The held
    experts carry the oldest ``stamp`` values in the tier, so ``take_slot_locked`` would choose them first; the only
    thing standing between them and the victim search is the ``leased_locked`` skip.
    """

    def test_a_leased_source_survives_admission_pressure_until_the_gpu_acknowledges(self, service):
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

        s = service
        held = [3, 4, 5]
        s.plan(held)
        s.dev.post(0, s.planned, s.count, s.routes, -1)
        s.dev.wait(0, s.planned, s.count, s.host_rows, s.keep, s.ram_miss)
        copy_expert_row_segments_gpu(s.segments, s.host_rows, s.dest_slots, s.dev.go_count)
        _cuda_ready()
        # Read, published, leased and copied -- and deliberately not acknowledged.
        assert s.dev.go_count.item() == len(held) and s.keep.item() == 1.0
        _delivered(s, held)
        slots = {expert: _resident(s)[expert] for expert in held}
        before = {expert: {n: s.slabs[0][n][slot].clone() for n in s.names} for expert, slot in slots.items()}
        assert all(s.leases()[slot] > 0 for slot in slots.values()), s.leases()
        assert s.host.counters()["leases_acked"] == 0, s.host.counters()

        newer = Exl3RamMissDevice(
            s.page, s.slot_map, device="cuda", layers=LAYERS, timeout_ms=2000, advise=False,
            lease_block=s.host.lease_block, lease_layout=s.host.lease_layout,
        )

        def press(experts):
            """A whole newer request on the second device: posted, served, copied and acknowledged."""
            s.plan(experts)
            s.step_with(newer)
            _cuda_ready()
            assert s.keep.item() == 1.0 and newer.go_count.item() == len(experts), experts
            assert s.until(lambda: all(s.host.contains(0, e) for e in experts)), (experts, _resident(s))

        # CAPACITY is 8 and the lease holds 3: the first two rounds fill the tier, the last two can only be served
        # by evicting, and every eviction has to walk past the three leased slots to find its victim.
        press([6, 7, 8])
        press([9, 10])
        assert len(_resident(s)) == CAPACITY, _resident(s)
        evicted_before = s.host.counters()["evictions"]
        press([11, 12, 13])
        press([14, 15, 0])
        counters = s.host.counters()
        assert counters["evictions"] - evicted_before == 6, counters
        assert counters["no_victim"] == 0, counters

        # The property: the leased sources are untouched -- same slot, still leased, byte for byte the same rows.
        resident = _resident(s)
        for expert, slot in slots.items():
            assert resident.get(expert) == slot, (expert, slot, resident)
            assert s.leases()[slot] > 0, (expert, slot, s.leases())
            for n in s.names:
                assert torch.equal(
                    s.slabs[0][n][slot].view(torch.uint8), before[expert][n].view(torch.uint8)
                ), (expert, n)
        assert s.host.counters()["leases_acked"] == sum(len(e) for e in ([6, 7, 8], [9, 10], [11, 12, 13], [14, 15, 0]))

        # The device's own verdict on the same question: every lane's slot generation is still the one it leased,
        # so the acknowledgement is CONSUMED and keep survives. A recycled source would have made this VIOLATED.
        keep_held = torch.ones(1, dtype=torch.float32, device="cuda")
        s.dev.ack(keep_held)
        _cuda_ready()
        assert keep_held.item() == 1.0 and s.host.fatal_seq() == 0
        for lane in range(len(held)):
            assert s.block.ack_word(0, lane) == _tagged(lease.CONSUMED, 1), lane
        assert s.until(lambda: all(s.leases()[slot] == 0 for slot in slots.values())), s.host.counters()
        counters = s.host.counters()
        assert counters["leases_voided"] == 0 and counters["lease_double_signal"] == 0, counters

        # The control, and the reason the assertions above are not vacuous: the same pressure, with the same slots
        # now retired, takes them first. Their stamps are the oldest in the tier, so two evictions are experts 3
        # and 4 in that order -- which is precisely the choice the lease was suppressing.
        evicted_before = s.host.counters()["evictions"]
        press([1, 2])
        assert s.host.counters()["evictions"] - evicted_before == 2, s.host.counters()
        gone = [expert for expert in held if not s.host.contains(0, expert)]
        assert gone == [3, 4], (gone, _resident(s))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
