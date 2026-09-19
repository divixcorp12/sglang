"""A format can own the pinned tier's slot bookkeeping through a PinnedSlotTable (CPU)."""

import pytest
import torch

from sglang.srt.layers.moe.expert_host_tier import PinnedSlotLRU
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


class RecordingTable(PinnedSlotLRU):
    """A PinnedSlotLRU that records every host use, as an external owner would see it."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self.host_uses = 0
        self.depth = 0
        self.max_depth = 0

    def before_host_use(self, cache):
        self.host_uses += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

    def after_host_use(self, cache):
        self.depth -= 1


def _streamer():
    reference = {
        name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + i for i, name in enumerate(NAMES)
    }
    layer = torch.nn.Module()
    layer.layer_id = 0
    return ExpertStreamer(layer, NAMES, format=SpecOnlyFormat(reference)), reference


def test_the_cache_uses_a_supplied_slot_table_and_announces_host_use():
    streamer, reference = _streamer()
    table = RecordingTable(3)
    cache = ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=table)
    cache.ensure_rows(torch.tensor([5, 2]))
    assert table.host_uses >= 1
    assert 5 in table and 2 in table
    assert cache.expert_to_slot[5].item() == table.expert_to_slot[5]
    slot = table.expert_to_slot[5]
    assert torch.equal(cache.tensors["w2_suh"][slot], reference["w2_suh"][5])
    before = table.host_uses
    cache.lookup(torch.tensor([5]))
    assert table.host_uses == before + 1
    assert table.depth == 0  # every use was closed
    cache.gather_rows(torch.tensor([5]), {n: torch.empty((1,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in reference.items()})
    assert table.depth == 0 and table.max_depth >= 2  # gather_rows nests lookup/ensure_rows


def test_host_use_is_closed_when_the_call_raises():
    streamer, _ = _streamer()
    table = RecordingTable(1)
    cache = ExpertPinnedHostCache(streamer, 1, device="cpu", slot_table=table)
    with pytest.raises(Exception):
        cache.lookup(torch.tensor([5], device="meta"))  # wrong device: lookup raises
    assert table.depth == 0


def test_a_slot_table_of_another_capacity_is_refused():
    streamer, _ = _streamer()
    with pytest.raises(ValueError, match="capacity"):
        ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=RecordingTable(2))


def test_the_default_lru_needs_no_host_use_hook_behaviour():
    PinnedSlotLRU(2).before_host_use(object())  # no-ops
    PinnedSlotLRU(2).after_host_use(object())


def test_the_cache_binds_its_capacity_into_an_unbound_slot_table():
    streamer, _ = _streamer()

    class Unbound(RecordingTable):
        def __init__(self):
            super().__init__(3)
            self.capacity = None

        def bind_capacity(self, capacity):
            self.capacity = capacity

    table = Unbound()
    ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=table)
    assert table.capacity == 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
