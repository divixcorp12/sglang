"""A planned RAM-hit lane that is NOT in the request's protect set is evicted by that same request.

This is the basis for withdrawing "fails loudly" from the plan for the early-read (kBusySeq-gated)
mechanism: a device that reads slot_map early sees such a lane as a hit (entry >= 0) and copies it,
while the service's reservation for the same request may evict it and overwrite its slot; nothing
re-reads the map afterwards, so the copy is silently the wrong bytes. Today's path re-reads the map
after demand_done and raises unserved_misses instead.

The service never sees which experts the device planned: a record carries only `need` (planned
experts missing from RAM) and `protect` (the routed experts). A planned hit that is not routed is
therefore invisible to it, which is why membership in protect is the whole protection.

Setup (CPU, host service over a simulated device page): a 3-slot tier holding experts 0, 1, 2 (0 the
least recently used). Request A plans lane 0 but protects only expert 4, which is missing.
  expected: resident after = [1, 2, 4]  (expert 0, resident and planned, was the victim)
Control: the same request with lane 0 also in protect (as the production router-miss producer does:
planned is a subset of routes, and routes are protected): resident after = [0, 2, 4] (expert 1 goes).

The mutant this must go red against (busy_seq_mutants.sh, victim_skips_resident_planned_lane): in
take_slot_locked, `if (listed(protect, expert)) {` becomes `if (listed(protect, expert) || expert == 0) {`,
i.e. victim selection skips the resident planned lane. The first test then fails. Run:
python unprotected_lane_eviction.py, or python -m pytest unprotected_lane_eviction.py.
"""

import pathlib
import tempfile

import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup


def resident_after(protect):
    """Experts resident in row 1 after a request needing expert 4 with the given protect set."""
    s = ram_miss_setup(pathlib.Path(tempfile.mkdtemp(prefix="lane_eviction_")), capacity=3, experts=6)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    try:
        for expert in (0, 1, 2):  # 0 is the least recently used
            seq = sim_post(page, 1, need=[expert], protect=[expert])
            host.pump()
            sim_wait(page, seq, 1.0)
        before = sorted(e for e in range(6) if host.contains(1, e))
        seq = sim_post(page, 1, need=[4], protect=protect, lanes=4)
        host.pump()
        sim_wait(page, seq, 1.0)
        return before, sorted(e for e in range(6) if host.contains(1, e))
    finally:
        host.stop()


def test_a_resident_planned_lane_outside_protect_is_evicted_by_its_own_request():
    before, after = resident_after(protect=[4])
    assert before == [0, 1, 2]
    assert after == [1, 2, 4]  # lane 0 was resident, planned, and the victim


def test_the_same_lane_survives_when_it_is_in_protect():
    before, after = resident_after(protect=[4, 0])
    assert before == [0, 1, 2]
    assert 0 in after and after == [0, 2, 4]


if __name__ == "__main__":
    print("protect=[4]   :", resident_after([4]))
    print("protect=[4, 0]:", resident_after([4, 0]))
