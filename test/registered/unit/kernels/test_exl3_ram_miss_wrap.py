"""The service walks the 32-bit request sequence through its wrap like the device does (CPU).

The device skips sequence 0 on wrap (post kernel and sim_post: `if (seq == 0) seq = 1`), so the
service must too. pump_demand and pump_advice both advance with `next += 1u`, so after 0xFFFFFFFF
each spends one iteration on a phantom seq 0: the done word steps back to 0 and the ring slot
`(0 - 1) % records` is read as if it held a request.

That phantom looks different on the two page states, so both are covered:
  used page   the slot holds an older sequence, the seqlock read fails, one overrun (demand) or one
              skipped advisory is counted and the done word is stored as 0.
  fresh page  the slot holds zeros, the read SUCCEEDS as an empty request, no overrun is counted, and
              the phantom is served: a touch-only demand or an empty advisory.
A test that only looked for the overrun counter would pass on a fresh page.
"""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import WORDS, Exl3RamMissHost, new_page, sim_post
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RECORD_BYTES = 128
DEMAND_RING = 64
DEMAND_RECORDS = 16
ADVISE_RING = DEMAND_RING + DEMAND_RECORDS * RECORD_BYTES
ADVISE_RECORDS = 64
STALE_SEQ = 0xFFFFFFF0  # what the slot before the wrap holds in a run that has been going a while


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _signed(value):
    return value - 2**32 if value >= 2**31 else value


def _set_word(page, offset, value):
    page[offset : offset + 4].view(torch.int32)[0] = _signed(value)


def _word(page, offset):
    return int(page[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF


def _drive(tmp_path, *, advisory, seed, used):
    """Seed both words at `seed`, post four records through the wrap, pump exactly four times.

    Returns the done word after every pump and the host's counters."""
    kind = "advise" if advisory else "demand"
    ring, records = (ADVISE_RING, ADVISE_RECORDS) if advisory else (DEMAND_RING, DEMAND_RECORDS)
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    _set_word(page, WORDS[f"{kind}_head"], seed)
    _set_word(page, WORDS[f"{kind}_done"], seed)
    if used:
        # The slot a phantom seq 0 would read: (0 - 1) % records, as the service computes it.
        _set_word(page, ring + (records - 1) * RECORD_BYTES, STALE_SEQ)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    try:
        posted = [sim_post(page, 1, need=[expert], protect=[expert], advisory=advisory) for expert in (0, 1, 2, 3)]
        done = []
        for _ in posted:
            assert host.pump() == (2 if advisory else 1)
            done.append(_word(page, WORDS[f"{kind}_done"]))
        idle = host.pump()
        return posted, done, idle, host.counters()
    finally:
        host.stop()


@pytest.mark.parametrize("used", [True, False], ids=["used_page", "fresh_page"])
@pytest.mark.parametrize("advisory", [False, True], ids=["demand", "advisory"])
def test_the_service_skips_sequence_zero_at_the_wrap(tmp_path, advisory, used):
    posted, done, idle, counters = _drive(tmp_path, advisory=advisory, seed=0xFFFFFFFD, used=used)
    assert posted == [0xFFFFFFFE, 0xFFFFFFFF, 1, 2]  # the device side skips 0
    # One pump per posted record, each leaving the done word at that record's sequence: no phantom
    # iteration, so the word never steps back to 0 and nothing is left over for a fifth pump.
    assert done == posted
    assert idle == 0
    if advisory:
        assert counters["advisories"] == 4 and counters["advisories_skipped"] == 0
    else:
        assert counters["served"] == 4
        assert counters["touch_only"] == 0  # a fresh page reads the phantom as an empty touch record
        assert counters["overruns"] == 0  # a used page fails the phantom's seqlock read


@pytest.mark.parametrize("advisory", [False, True], ids=["demand", "advisory"])
def test_the_control_a_service_opened_at_the_wrap_already_skips_zero(tmp_path, advisory):
    """open() derives its first sequence from the done word with the skip, so a page whose done word
    is already 0xFFFFFFFF is served correctly today. It shows the failure above is the per-request
    advance, not the test setup.

    If the test above fails, read this one first: when this control fails too, the setup is wrong
    (seeding, slot arithmetic, pump return codes), not the service."""
    posted, done, idle, counters = _drive(tmp_path, advisory=advisory, seed=0xFFFFFFFF, used=True)
    assert posted == [1, 2, 3, 4]
    assert done == posted and idle == 0
    assert counters["overruns"] == 0 and counters["advisories_skipped"] == 0
