"""The GPU drip's host half (CPU; LEASE_PROTOCOL.md, "The GPU drip"): the copy thread copies a RAM-resident expert's
row into a scratch slot while no demand copy is pending, publishes its scratch-table word once every chunk landed and
the source never changed, the post (ChainSim, typing with ram_slot_map.type_lanes) types the lane HIT_SCRATCH, and the
service checks it against the slot. The CPU copy backend lands a mark's bytes only when the test releases it, so a
test can hold a drip unlanded. The scratch here is host memory; the GPU rig is test_exl3_gpu_drip_cuda.py."""

import struct
import time

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import (
    CAND_DRIP_COUNT,
    CAND_DRIP_EXPERTS,
    CAND_DRIP_SEQ,
    candidate_offset,
    new_candidate_page,
    new_page,
)
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, attached_host, ram_miss_setup, run_host_script

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROW = 1
DST_ROWS = 6
CHUNK = 4096  # several chunks a row, so a drip is several marks


class Rig:
    def __init__(self, tmp_path, *, slots=2, targets=None, candidates=None):
        # Six slots, two staging: four mappable rows.
        self.s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
        self.page = new_page(pin=False, wire=wire_layout(8))
        self.host = attached_host(self.s, self.page, k=2, variant="instr")
        self.sim = ChainSim(self.host, self.page, self.s.slabs)
        self.slabs = self.s.slabs[ROW]
        try:
            self.host.enable_copy_engine(-1)
            self.host.enable_gpu_drip(slots, picks=4, chunk_bytes=CHUNK, targets=targets, candidates=candidates)
            self.scratch = {
                n: torch.zeros((slots,) + tuple(slab.shape[1:]), dtype=slab.dtype) for n, slab in self.slabs.items()
            }
            self.host.set_drip_scratch(list(self.scratch.values()))
            # The copy thread writes through these addresses until host.stop(): the rig keeps them.
            self.dst = {
                n: torch.zeros((DST_ROWS,) + tuple(slab.shape[1:]), dtype=slab.dtype) for n, slab in self.slabs.items()
            }
            table = torch.tensor(
                [
                    [slab.data_ptr(), self.dst[n].data_ptr(), slab[0].numel() * slab.element_size()]
                    for n, slab in self.slabs.items()
                ],
                dtype=torch.int64,
            )
            self.host.set_copy_table(ROW, table, DST_ROWS)
            self.host.arm_copy_engine()
        except BaseException:
            self.host.stop()
            raise

    def load(self, experts):
        """Make ``experts`` resident in row ROW through a plain (uncaptured) request of misses."""
        req = self.sim.post(ROW, experts)
        assert self.host.pump() == 1 and self.sim.wait_served(req)

    def state(self, slot=0):
        return self.host.drip_slots()[slot]

    def word(self, slot=0):
        return self.sim.scratch_words()[slot]

    def scratch_holds(self, slot, ram_slot):
        return all(
            torch.equal(self.scratch[n][slot].view(torch.uint8), self.slabs[n][ram_slot].view(torch.uint8))
            for n in self.slabs
        )


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def test_a_landed_drip_is_typed_hit_scratch_and_holds_the_ram_rows_bytes(tmp_path):
    """Published only once every chunk landed: the word names (ROW, 3), the scratch slot holds the RAM row's bytes, the
    captured post types the lane HIT_SCRATCH at the slot, and the service checks it and copies nothing for it.
    Mutation: land() before the last chunk -- red on the bytes."""
    rig = Rig(tmp_path)
    try:
        rig.load([3])
        ram_slot = rig.host.mapping(ROW)[3]
        rig.host.copy_engine_release(-1)
        rig.host.drip_offer(ROW, [3])
        assert _until(lambda: rig.state()["state"] == "published")
        assert rig.word() & 0xFFFFFFFF == ROW << 16 | 3 and rig.word() >> 32 != 0
        assert rig.scratch_holds(0, ram_slot)
        before = rig.host.counters()
        req = rig.sim.post(ROW, [3], dst=[2], captured=True, drip=True)
        assert req.kinds == [LaneKind.HIT_SCRATCH] and req.slots == [0]
        assert rig.host.pump() == 1
        after = rig.host.counters()
        assert (after["drip_issued"], after["drip_landed"], after["drip_used"]) == (1, 1, 1)
        assert after["copy_lanes"] == before["copy_lanes"], "the service copied a scratch lane"
    finally:
        rig.host.stop()


def test_an_unlanded_drip_is_never_published_and_its_target_post_abandons_it(tmp_path):
    """Marks held: the drip's first chunk is issued but the word stays 0, so the target's captured post types the
    resident lane a copy-engine hit; that post abandons the drip, and its chunks, released later, publish nothing.
    Mutation: publish at issue -- red on the HIT_COPY kind."""
    rig = Rig(tmp_path)
    try:
        rig.load([3])
        rig.host.drip_offer(ROW, [3])
        assert _until(lambda: rig.state()["state"] == "filling" and rig.host.copy_engine_marked() >= 1)
        req = rig.sim.post(ROW, [3], dst=[2], captured=True, drip=True)
        assert req.kinds == [LaneKind.HIT_COPY] and rig.word() == 0
        assert rig.host.pump() == 1
        assert _until(lambda: rig.host.counters()["drip_abandoned"] == 1)
        rig.host.copy_engine_release(-1)
        assert rig.sim.copy_wait(req)
        assert _until(lambda: rig.state()["state"] == "free")
        c = rig.host.counters()
        assert (c["drip_landed"], c["drip_used"]) == (0, 0) and rig.word() == 0
    finally:
        rig.host.stop()


def test_a_drip_is_issued_only_after_the_demand_copy_before_it_completed(tmp_path):
    """The copy engine's priority: with a demand copy's mark unreleased the drip issues nothing; once that mark
    completes, the drip's chunks follow. Mutation: drip_step ignoring demand_idle -- red on the mark count."""
    rig = Rig(tmp_path)
    try:
        rig.load([3, 4])
        req = rig.sim.post(ROW, [4], dst=[1], captured=True, drip=True)
        assert req.kinds == [LaneKind.HIT_COPY]
        assert rig.host.pump() == 1
        assert _until(lambda: rig.host.copy_engine_marked() == 1)
        rig.host.drip_offer(ROW, [3])
        assert _until(lambda: rig.state()["state"] == "filling")
        time.sleep(0.1)
        assert rig.host.copy_engine_marked() == 1, "a drip chunk was issued behind a pending demand copy"
        rig.host.copy_engine_release(1)
        assert rig.sim.copy_wait(req)
        assert _until(lambda: rig.host.copy_engine_marked() >= 2)
        rig.host.copy_engine_release(-1)
        assert _until(lambda: rig.state()["state"] == "published")
    finally:
        rig.host.stop()


def test_a_drip_whose_source_row_was_evicted_meanwhile_is_not_published(tmp_path):
    """The RAM row is released (unmapped, its slot free to rewrite) while the drip's chunks are in flight: the mirror
    generation moved, so the landed copy is dropped. Mutation: land() without source_unchanged -- red on the word."""
    rig = Rig(tmp_path)
    try:
        rig.load([3])
        rig.host.drip_offer(ROW, [3])
        assert _until(lambda: rig.host.copy_engine_marked() >= 1)
        rig.host.release(ROW, rig.host.mapping(ROW)[3])
        rig.host.copy_engine_release(-1)
        assert _until(lambda: rig.host.counters()["drip_abandoned"] == 1)
        assert rig.state()["state"] == "free" and rig.word() == 0 and rig.host.counters()["drip_landed"] == 0
    finally:
        rig.host.stop()


def test_a_retired_slot_is_kept_until_every_record_that_could_read_its_word_is_handled(tmp_path):
    """One slot. The target's post retires the slot (word 0), but a post that read the word before the clear may still
    run its chain: that record is accepted, and the slot is not reused (a new pick finds no slot) until the record two
    past the head read at the clear is handled. Then expert 4 drips into it. Mutation: release_quarantined freeing at
    once -- red on drip_no_slot and on the late record's check."""
    rig = Rig(tmp_path, slots=1)
    try:
        rig.load([3, 4])
        rig.host.copy_engine_release(-1)
        rig.host.drip_offer(ROW, [3])
        assert _until(lambda: rig.state()["state"] == "published")
        used = rig.sim.post(ROW, [3], dst=[2], captured=True, drip=True)
        assert used.kinds == [LaneKind.HIT_SCRATCH]
        assert _until(lambda: rig.state()["state"] == "retiring") and rig.word() == 0
        # A post that read the word before the clear: its lane still names slot 0.
        late = rig.sim.post(ROW, [3], dst=[2], captured=True, kinds=[int(LaneKind.HIT_SCRATCH)], slots=[0])
        rig.host.drip_offer(ROW, [4])
        assert _until(lambda: rig.host.counters()["drip_no_slot"] == 1)
        assert rig.host.pump() == 1 and rig.host.pump() == 1  # both checked against the retiring slot
        assert rig.host.counters()["drip_used"] == 2
        assert rig.state()["state"] == "retiring", "freed before the record after the late one was handled"
        rig.sim.post(0, [])
        assert rig.host.pump() == 1
        assert _until(lambda: rig.state()["state"] == "free")
        assert late.seq + 1 == rig.host.handled_through()
        rig.host.drip_offer(ROW, [4])
        assert _until(lambda: rig.state()["state"] == "published")
        assert rig.word() & 0xFFFFFFFF == ROW << 16 | 4 and rig.scratch_holds(0, rig.host.mapping(ROW)[4])
    finally:
        rig.host.stop()


def _write_drips(page, seq, experts, *, slot_seq=None):
    """The drip select kernel's publication for record ``seq``: count and experts, then its seq word."""
    off = candidate_offset(seq)
    payload = struct.pack("<H", len(experts)) + b"\0\0" + struct.pack(f"<{len(experts)}H", *experts)
    page[off + CAND_DRIP_COUNT : off + CAND_DRIP_COUNT + len(payload)] = torch.frombuffer(
        bytearray(payload), dtype=torch.uint8
    )
    assert CAND_DRIP_EXPERTS == CAND_DRIP_COUNT + 4
    word = (seq if slot_seq is None else slot_seq) & 0xFFFFFFFF
    page[off + CAND_DRIP_SEQ : off + CAND_DRIP_SEQ + 4] = torch.frombuffer(
        bytearray(struct.pack("<I", word)), dtype=torch.uint8
    )


def test_the_scorers_picks_drip_for_the_target_row_unless_it_posted_first(tmp_path):
    """Row 0 targets row 1: the drip picks published for a row-0 record are dripped into row 1's scratch and used by
    row 1's post. Picks published only after row 1 posted again are late: nothing is admitted. Mutation: admit()
    without the last-posted check -- red on the second drip_issued."""
    page = new_candidate_page(pin=False)
    rig = Rig(tmp_path, targets=[1, -1], candidates=page)
    try:
        rig.load([3, 4])
        rig.host.copy_engine_release(-1)
        source = rig.sim.post(0, [])
        assert rig.host.pump() == 1
        _write_drips(page, source.seq, [3])
        assert _until(lambda: rig.state()["state"] == "published")
        req = rig.sim.post(ROW, [3], dst=[2], captured=True, drip=True)
        assert req.kinds == [LaneKind.HIT_SCRATCH] and rig.host.pump() == 1
        assert _until(lambda: rig.state()["state"] == "retiring")
        late = rig.sim.post(0, [])
        rig.sim.post(ROW, [])
        assert rig.host.pump() == 1 and rig.host.pump() == 1
        _write_drips(page, late.seq, [4])
        time.sleep(0.1)
        assert rig.host.counters()["drip_issued"] == 1, "a pick published after its target posted was dripped"
    finally:
        rig.host.stop()


def test_a_scratch_lane_the_drip_does_not_hold_fails_stop(tmp_path):
    """A record whose HIT_SCRATCH lane names a slot that holds nothing (the device's word and the host's slot
    disagree) is a protocol error, like a hit at the wrong RAM slot."""
    result = run_host_script(
        tmp_path,
        """
        host.enable_copy_engine(-1)
        host.enable_gpu_drip(1, picks=1, chunk_bytes=4096)
        slabs = list(s.slabs[1].values())
        host.set_drip_scratch([torch.zeros((1,) + tuple(x.shape[1:]), dtype=x.dtype) for x in slabs])
        dst = [torch.zeros((6,) + tuple(x.shape[1:]), dtype=x.dtype) for x in slabs]
        table = [[x.data_ptr(), d.data_ptr(), x[0].numel() * x.element_size()] for x, d in zip(slabs, dst)]
        host.set_copy_table(1, torch.tensor(table, dtype=torch.int64), 6)
        host.arm_copy_engine()
        sim.post(1, [3], captured=True, kinds=[6], slots=[0])
        host.pump()
        print("reached", flush=True)
        """,
        capacity=6,
        staging=2,
    )
    assert_aborted(result, "a scratch lane whose drip slot does not hold the expert")
