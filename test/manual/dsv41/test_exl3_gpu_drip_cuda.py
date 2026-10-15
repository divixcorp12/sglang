"""The GPU drip on a real GPU against the real C++ service (LEASE_PROTOCOL.md, "The GPU drip"): the copy engine
drips a RAM-resident expert's row into a VRAM scratch slot, the real post types the target's lane HIT_SCRATCH and C1s
copies it from the scratch, byte for byte what a drip-off gather delivers; an unlanded drip is never used; a scratch
slot is not refilled while a delayed C1s may still read it; the drip select kernel publishes the best mapped,
not-hot experts.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import sys
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import Chain, until  # noqa: E402

from sglang.kernels.ops.moe import expert_stream_transport as ops  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

DELAY_CYCLES = 600_000_000  # ~250 ms at the 5090's clock: long enough for the host to retire and try to refill


def _drip_chain(tmp_path, drip_slots=2):
    c = Chain(tmp_path, copy_engine=True, start=False, drip_slots=drip_slots)
    c.host.start_thread(fatal_wait_s=60.0)
    c.host.arm_copy_engine()
    return c


def _resident(c, experts, row):
    c.plan(experts, row)
    c.gather(row)
    torch.cuda.synchronize()
    assert c.handled(), c.host.counters()


def _landed(c, row, expert):
    return any(s["state"] == "published" and (s["row"], s["expert"]) == (row, expert) for s in c.host.drip_slots())


def _captured(c, experts, row, **hooks):
    """One captured-flag chain of ``experts`` into destination rows 0.., its destinations snapshot on the stream."""
    c.plan(experts, row)
    c.chain(row, captured=True, **hooks)
    snapshot = c.snapshot(row)
    torch.cuda.synchronize()
    assert c.handled(), c.host.counters()
    return snapshot


@pytest.mark.parametrize("drip", [True, False], ids=["drip_on", "drip_off"])
def test_a_landed_drip_is_copied_from_vram_byte_for_byte(tmp_path, drip):
    """Experts 5 and 9 are resident in row 1 and 5 has landed in a scratch slot (on). The captured chain of [5, 9, 12]
    types 5 HIT_SCRATCH, 9 a copy-engine hit and 12 a GPU miss; every destination row is the reference's exact bytes,
    as with the drip off, where 5 is a copy-engine hit, and the copy engine copied one lane fewer. Mutation: C1s left
    out of the chain -- red on 5's bytes."""
    c = _drip_chain(tmp_path)
    try:
        _resident(c, [5, 9], 1)
        if drip:
            c.host.drip_offer(1, [5])
            assert until(lambda: _landed(c, 1, 5)), c.host.drip_slots()
        lanes = c.host.counters()["copy_lanes"]
        snapshot = _captured(c, [5, 9, 12], 1)
        hit_5 = LaneKind.HIT_SCRATCH if drip else LaneKind.HIT_COPY
        assert c.kinds(3) == [int(hit_5), int(LaneKind.HIT_COPY), int(LaneKind.MISS_GPU)]
        c.check([5, 9, 12], snapshot, 1)
        counters = c.host.counters()
        assert counters["copy_lanes"] - lanes == (1 if drip else 2)
        assert counters["drip_used"] == (1 if drip else 0)
    finally:
        c.close()


def test_an_unlanded_drip_is_never_used(tmp_path):
    """The drip of 5 is admitted but held before its first copy: its word stays 0, so the chain copies 5 on the copy
    engine, exactly; the post then abandons it and nothing is ever published. Mutation: publishing at admission --
    red on the kind (and the scratch's zero bytes)."""
    c = _drip_chain(tmp_path)
    try:
        _resident(c, [5], 1)
        c.host.drip_hold(True)
        c.host.drip_offer(1, [5])
        assert until(lambda: c.host.drip_slots()[0]["state"] == "filling")
        snapshot = _captured(c, [5], 1)
        assert c.kinds(1) == [int(LaneKind.HIT_COPY)]
        c.check([5], snapshot, 1)
        assert until(lambda: c.host.counters()["drip_abandoned"] == 1)
        c.host.drip_hold(False)
        assert until(lambda: c.host.drip_slots()[0]["state"] == "free")
        assert c.host.counters()["drip_landed"] == 0
    finally:
        c.close()


def test_a_scratch_slot_is_not_refilled_under_a_delayed_reader(tmp_path):
    """One slot holds 5. The chain using it sleeps before C1 (so C1s has not read the slot) while the host retires the
    slot and is offered 7: it must find no slot, and 5's bytes arrive exact. Two records later the slot is free, 7 drips
    into it and its chain delivers 7 exactly. Mutation: release_quarantined freeing at once -- 7 overwrites the slot
    under the sleeping C1s, red on 5's bytes."""
    c = _drip_chain(tmp_path, drip_slots=1)
    try:
        _resident(c, [5, 7], 1)
        c.host.drip_offer(1, [5])
        assert until(lambda: _landed(c, 1, 5))

        def offer_while_asleep():
            torch.cuda._sleep(DELAY_CYCLES)
            # The post ran before the sleep: once its record retired the slot, 7 is offered while C1s still waits.
            assert until(lambda: c.host.drip_slots()[0]["state"] == "retiring", 2.0)
            c.host.drip_offer(1, [7])
            until(lambda: c.host.counters()["drip_no_slot"] == 1 or c.host.drip_slots()[0]["expert"] == 7, 2.0)
            time.sleep(0.05)  # a refill, were one admitted, lands before C1s wakes

        snapshot = _captured(c, [5], 1, before_c1=offer_while_asleep)
        assert c.kinds(1) == [int(LaneKind.HIT_SCRATCH)]
        c.check([5], snapshot, 1)
        assert c.host.counters()["drip_no_slot"] == 1
        for _ in range(2):
            _resident(c, [9], 0)
        assert until(lambda: c.host.drip_slots()[0]["state"] == "free")
        c.host.drip_offer(1, [7])
        assert until(lambda: _landed(c, 1, 7))
        snapshot = _captured(c, [7], 1)
        assert c.kinds(1) == [int(LaneKind.HIT_SCRATCH)]
        c.check([7], snapshot, 1)
    finally:
        c.close()


def test_the_drip_select_publishes_the_best_mapped_experts_that_are_not_hot():
    """The drip select kernel against a reference: of the target row's experts mapped in RAM and not VRAM-hot, the
    `picks` best by their best score over the rows (ties to the lower id), under the drip seq; rows 0 publishes none.
    Mutation: the eligibility ignoring the hot set -- red on the picks."""
    experts, picks, seq = 64, 4, 7
    g = torch.Generator().manual_seed(3)
    scores = torch.randn((3, experts), generator=g).cuda()
    scores[1, 10] = scores[0, 11] = 99.0  # 10 hot, 11 mapped: 11 must lead, 10 must not appear
    ram_slot = torch.full((2, experts), -1, dtype=torch.int32)
    ram_slot[1, ::2] = 0
    ram_slot[1, 11] = 0
    ram_slot = ram_slot.cuda()
    hot = torch.tensor([10, 20, -1], dtype=torch.int64).cuda()
    state = torch.zeros(len(ops.STATE_WORDS), dtype=torch.int32).cuda()
    state[ops.STATE_WORDS["posted"]] = seq
    page = ops.new_candidate_page(pin=True)
    module = ops._device_module()
    ops.run_spec_drip(scores, 3, hot_slots=hot, hot_capacity=3, ram_slot=ram_slot, target=1, picks=picks,
                      state=state, candidates=page, module=module)
    torch.cuda.synchronize()
    best = scores.max(dim=0).values.cpu()
    eligible = [e for e in range(experts) if ram_slot[1, e] >= 0 and e not in (10, 20)]
    want = sorted(eligible, key=lambda e: (-float(best[e]), e))[:picks]
    assert ops.read_drip_picks(page, seq) == want and want[0] == 11
    ops.run_spec_drip(scores, 0, hot_slots=hot, hot_capacity=3, ram_slot=ram_slot, target=1, picks=picks,
                      state=state, candidates=page, module=module)
    torch.cuda.synchronize()
    assert ops.read_drip_picks(page, seq) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
