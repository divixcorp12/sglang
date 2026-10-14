"""RAM prefetch promotion on a real GPU against the real C++ service (LEASE_PROTOCOL.md, deferred publication).

A pool row of row r that landed before row r's record was published is mapped into row r's deferred delta when row
r-1's next record is served; the real post kernel then types it a hit and no NVMe row is read for it.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import EXPERTS, LAYERS, Chain  # noqa: E402

from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer, paused  # noqa: E402
from sglang.test.dsv41_ram_prefetch_fixtures import HIDDEN, enable, x_token_bytes  # noqa: E402

HITS = {int(LaneKind.HIT_COPY), int(LaneKind.HIT_SM), int(LaneKind.HIT_CPU)}


def _forward_row(c, row, experts):
    c.plan(experts, row)
    c.gather(row)
    torch.cuda.synchronize()
    assert c.handled()


@pytest.mark.parametrize("promote", [True, False], ids=["promote_on", "promote_off"])
def test_a_landed_pool_row_is_typed_a_hit_after_the_previous_rows_next_record(tmp_path, promote):
    """Row 1's miss record leaves its delta deferred; expert 7 landed in row 1's pool; row 0's next record publishes
    it with 7 mapped, and row 1's post types 7 a hit with no row read. Off: 7 is a GPU miss read from NVMe again.
    Mutations: promote at publish skips the landed entry (7 is MISS_GPU); the entry is mapped but the delta is
    published before row 0's record is claimed (the post sees the unpromoted map)."""
    c = Chain(tmp_path, copy_engine=True, start=False, spec_share=2)
    try:
        x_rows = torch.zeros((LAYERS, x_token_bytes()), dtype=torch.uint8).pin_memory()
        out_rows = torch.zeros((LAYERS, 2, HIDDEN), dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(c.host.test_kernel_address(zero=True), [0] * (c.lanes + 1), cores, x_rows, out_rows,
                                  threads=2)
        for row in range(LAYERS):
            c.host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
        enable(SimpleNamespace(host=c.host, x_rows=x_rows), [[30.0] * EXPERTS], promote=promote)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        for row in range(LAYERS):
            c.dev.set_row_cpu(row)
        c.host.arm_copy_engine()

        landed = 7
        with paused(c.host):
            c.host.spec_place(1, landed)
        _forward_row(c, 0, [0])
        _forward_row(c, 1, [0])  # a miss: row 1's delta waits for row 0's next record
        _forward_row(c, 0, [0])  # a hit; its serve publishes row 1's delta
        read = c.host.counters()["rows_read"]

        c.plan([landed], 1)
        c.chain(1)
        torch.cuda.synchronize()
        assert c.handled()
        if promote:
            assert c.kinds(1)[0] in HITS
            assert c.host.counters()["rows_read"] == read, "a promoted row was read from NVMe again"
            assert c.host.counters()["spec_promoted_early"] == 1
        else:
            assert c.kinds(1)[0] == int(LaneKind.MISS_GPU)
            assert c.host.counters()["spec_promoted_early"] == 0
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
