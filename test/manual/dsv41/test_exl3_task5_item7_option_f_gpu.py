"""Task 5 item 7, on a real GPU: Option F's all-hit handshake is preserved in every configuration.

LEASE_PROTOCOL.md section 15 lists the orderings that must survive leases. Two of them are decided by one line of
the post kernel, ``armed = need_count > 0 || advise != 0 || (lease != nullptr && planned_count > 0)``:

* advise ON, leases OFF (Option F as it ran before leases): a request whose rows are all resident is still armed, so
  the device waits for the service even though nothing is read. The service's recency and eviction decisions for the
  row precede the gather (section 15, row 2).
* advise OFF, leases OFF (the plain path): an all-hit request is NOT armed and does not wait. Removing or adding
  the handshake here is a behaviour change, and the plan says any removal "is a separate optimization after
  equivalent protection is proven". This pins that today's shortcut is still today's shortcut.
* leases ON: every request with lanes is armed (the handshake is where the lease is granted); that arm is pinned by
  ``test_exl3_lease_kernels_cuda.py`` and is repeated here only as the third leg of the truth table.

Whether the wait really BLOCKS on an armed all-hit request is observed, not inferred from the ``pending`` word: with
no service running, an armed request times out and an unarmed one returns at once.

Run on divix01 under ``gpu-run.sh`` with PYTHONPATH pointing at the tree under test.
"""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe.exl3_ram_miss import STATE_WORDS, Exl3RamMissDevice, new_page, page_word  # noqa: E402

LAYERS, EXPERTS = 2, 16
TOP_K = 6


class Device:
    """A device with every planned expert already resident in row 0's slot map, and no service."""

    def __init__(self, *, advise, lease=False, timeout_ms=150):
        self.page = new_page(pin=True)
        self.slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
        self.experts = [2, 5, 9]
        for slot, expert in enumerate(self.experts):
            self.slot_map[0, expert] = slot
        kwargs = {}
        if lease:
            from sglang.kernels.ops.moe import exl3_lease_block as lease_block

            self.layout = lease_block.lease_layout([8] * LAYERS)
            self.raw = lease_block.new_lease_block(self.layout, pin=True)
            for row in range(LAYERS):
                entry = lease_block.ROW_TABLE + row * lease_block.ROW_TABLE_ENTRY_BYTES
                self.raw[entry : entry + 4].view(torch.int32)[0] = self.layout.slot_gen_base[row]
                self.raw[entry + 4 : entry + 8].view(torch.int32)[0] = self.layout.capacities[row]
            kwargs = dict(lease_block=self.raw, lease_layout=self.layout)
        self.dev = Exl3RamMissDevice(
            self.page, self.slot_map, device="cuda", layers=LAYERS, timeout_ms=timeout_ms, advise=advise, **kwargs
        )
        self.planned = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.planned[: len(self.experts)] = torch.tensor(self.experts, dtype=torch.int64)
        self.count = torch.full((1,), len(self.experts), dtype=torch.int32, device="cuda")
        self.routes = self.planned.clone()
        self.host_rows = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")

    def post(self):
        self.dev.post(0, self.planned, self.count, self.routes, -1)
        torch.cuda.synchronize()

    def pending(self):
        return int(self.dev.state[STATE_WORDS["pending"]])

    def wait(self):
        start = time.perf_counter()
        self.dev.wait(0, self.planned, self.count, self.host_rows, self.keep, self.ram_miss)
        torch.cuda.synchronize()
        return time.perf_counter() - start


def test_with_advise_on_an_all_hit_request_is_armed_and_the_device_waits_for_the_service():
    d = Device(advise=True)
    d.post()
    assert d.pending() != 0, "armed: Option F waits for demand_done even when nothing is missing"
    elapsed = d.wait()
    assert d.dev.stats()["timeouts"] == 1, "and it really waited: no service ran, so it timed out"
    assert elapsed >= 0.1 and d.keep.item() == 0.0


def test_with_advise_off_an_all_hit_request_is_not_armed_and_the_device_does_not_wait():
    d = Device(advise=False)
    d.post()
    assert d.pending() == 0, "unarmed: the plain path skips the service round trip when every row is resident"
    elapsed = d.wait()
    assert d.dev.stats()["timeouts"] == 0 and elapsed < 0.1, "and it really did not wait"
    assert d.keep.item() == 1.0 and d.host_rows[:3].tolist() == [0, 1, 2]
    assert page_word(d.page, "demand_head") != 0, "the touch record is still posted, so the service sees the hits"


def test_with_leases_on_an_all_hit_request_with_lanes_is_armed_whatever_advise_says():
    for advise in (False, True):
        d = Device(advise=advise, lease=True)
        d.post()
        assert d.pending() != 0, advise
