"""A format can own the pinned tier's slot bookkeeping through a PinnedSlotTable (CPU)."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe import expert_hot_cache
from sglang.srt.layers.moe.expert_host_tier import PinnedSlotLRU
from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
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


class GuardedTable(RecordingTable):
    """Records every read of its slot map made outside a host use.

    An owner thread may move slots whenever no host use is open, so such a
    read can see a slot map that changes before the caller uses it.
    """

    def __init__(self, capacity):
        self.outside_reads = 0
        super().__init__(capacity)

    @property
    def expert_to_slot(self):
        if self.depth == 0:
            self.outside_reads += 1
        return self._slot_map

    @expert_to_slot.setter
    def expert_to_slot(self, value):
        self._slot_map = value


def _streamer():
    reference = {
        name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + i
        for i, name in enumerate(NAMES)
    }
    layer = torch.nn.Module()
    layer.layer_id = 0
    return ExpertStreamer(layer, NAMES, format=SpecOnlyFormat(reference)), reference


def test_the_cache_uses_a_supplied_slot_table_and_announces_host_use():
    streamer, reference = _streamer()
    table = RecordingTable(3)
    cache = ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=table)
    cache.ensure_rows(torch.tensor([5, 2]))
    assert table.host_uses == 1  # one top-level call is one host use
    assert 5 in table and 2 in table
    assert cache.expert_to_slot[5].item() == table.expert_to_slot[5]
    slot = table.expert_to_slot[5]
    assert torch.equal(cache.tensors["w2_suh"][slot], reference["w2_suh"][5])
    before = table.host_uses
    cache.lookup(torch.tensor([5]))
    assert table.host_uses == before + 1
    assert table.depth == 0  # every use was closed
    outputs = {
        n: torch.empty((1,) + tuple(t.shape[1:]), dtype=t.dtype)
        for n, t in reference.items()
    }
    before = table.host_uses
    cache.copy_rows(torch.tensor([5]), outputs)
    assert table.host_uses == before + 1 and table.depth == 0
    assert torch.equal(outputs["w2_suh"][0], reference["w2_suh"][5])
    cache.gather_rows(
        torch.tensor([5]),
        {
            n: torch.empty((1,) + tuple(t.shape[1:]), dtype=t.dtype)
            for n, t in reference.items()
        },
    )
    # gather_rows nests lookup/ensure_rows, each calling before/after again.
    assert table.depth == 0 and table.max_depth >= 2


def test_host_use_is_closed_when_the_call_raises():
    streamer, _ = _streamer()
    table = RecordingTable(1)
    cache = ExpertPinnedHostCache(streamer, 1, device="cpu", slot_table=table)
    with pytest.raises(ValueError, match="device"):
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


def test_host_use_is_a_nestable_context_that_closes_on_error():
    streamer, _ = _streamer()
    table = RecordingTable(2)
    cache = ExpertPinnedHostCache(streamer, 2, device="cpu", slot_table=table)
    with cache.host_use():
        assert table.depth == 1
        cache.ensure_rows(torch.tensor([1]))
        assert table.max_depth == 2 and table.depth == 1
    assert table.depth == 0
    with pytest.raises(KeyError):
        with cache.host_use():
            raise KeyError("boom")
    assert table.depth == 0 and table.host_uses == 3


def _tickets(experts):
    return [
        SimpleNamespace(expert_id=expert, slot=slot, generation=1)
        for slot, expert in enumerate(experts)
    ]


def _promoting_hot_cache(streamer, events, table):
    """The parts of ExpertHotCache that _prepare_promotion and
    _load_reserved_in_chunks touch; ExpertHotCache itself needs CUDA."""
    hot = SimpleNamespace(
        streamer=streamer,
        device=torch.device("cpu"),
        _transfer_plan=SimpleNamespace(
            set_rows=lambda *args, **kwargs: events.append(("set_rows", table.depth))
        ),
        _copy_routes_for=lambda sources, use_secondary: "routes",
        _slot_batch=lambda publish=True: contextlib.nullcontext(),
        begin_loading=lambda ticket: True,
        _cancel_tickets=lambda tickets: events.append(("cancel", len(tickets))),
        promotion_in_flight=None,
        wait_for_slot_publication=lambda: None,
        _transfer_executor=SimpleNamespace(
            wait=lambda ticket, stream: events.append(("wait", table.depth))
        ),
    )
    hot._prepare_promotion = lambda tickets: ExpertHotCache._prepare_promotion(
        hot, tickets
    )

    def complete(promotion):
        events.append(("complete", table.depth))
        hot.promotion_in_flight = None

    hot.complete_promotion = complete
    return hot


def test_a_promotion_reads_the_pinned_slot_map_inside_one_host_use():
    streamer, _ = _streamer()
    table = GuardedTable(3)
    cache = ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=table)
    hot = _promoting_hot_cache(streamer, [], table)
    promotion = ExpertHotCache._prepare_promotion(hot, _tickets([5, 2]))
    assert promotion.secondary_source_rows == [
        table._slot_map[5],
        table._slot_map[2],
    ]
    assert table.outside_reads == 0
    assert table.depth == 0 and table.max_depth >= 2
    assert promotion.routes == "routes" and cache.capacity == 3


def test_each_promotion_chunk_runs_inside_one_host_use():
    # is_pinned makes evictable_rows() read the table's slot map as well.
    streamer, _ = _streamer()
    table = GuardedTable(2)
    cache = ExpertPinnedHostCache(
        streamer, 2, device="cpu", slot_table=table, is_pinned=lambda expert: False
    )
    events = []
    hot = _promoting_hot_cache(streamer, events, table)
    stream = SimpleNamespace(
        synchronize=lambda: events.append(("synchronize", table.depth))
    )
    with (
        patch("torch.cuda.current_stream", return_value=stream),
        patch.object(
            expert_hot_cache,
            "submit_hot_cache_promotions",
            side_effect=lambda promotions, producer_stream: events.append(
                ("submit", table.depth)
            ),
        ),
    ):
        ExpertHotCache._load_reserved_in_chunks(hot, _tickets([5, 2, 7]), cache)
    assert table.outside_reads == 0
    kinds = [kind for kind, _ in events]
    assert kinds == ["set_rows", "submit", "wait", "synchronize", "complete"] * 2
    assert all(depth >= 1 for _, depth in events), events
    assert table.depth == 0
    assert table.host_uses >= 2  # at least one outer host use per chunk


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
