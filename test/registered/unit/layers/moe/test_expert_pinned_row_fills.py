"""The pinned tier's native fills (SGLANG_DSV41_ENABLE_PREFILL_FILLS) as ExpertPinnedHostCache drives them (CPU).

A fake PinnedRowFills stands in for the RAM-miss service: its rows land only when waited for or joined (or, to stand
in for the reader running ahead, ``progress`` rows per ``fill_landed`` poll), so a copy that ran before its row landed
would read the sentinel instead of the row.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_host_tier import PinnedSlotLRU
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
SENTINEL = -7


class FakeFills:
    """Claims through the tier's LRU and lands rows lazily, recording every call."""

    def __init__(self, table: PinnedSlotLRU, reference):
        self.table = table
        self.reference = reference
        self.cache = None
        self.events = []
        self.claimed = []
        self.landed = 0
        self.fail = False
        self.progress = 0

    def fill_begin(self, experts, protected, fallback):
        self.events.append(("begin", list(experts), sorted(protected), fallback))
        self.claimed = []
        for expert in experts:
            slot, _ = self.table.assign(expert, frozenset(protected))
            for name in NAMES:
                self.cache.tensors[name][slot].fill_(SENTINEL)
            self.claimed.append((expert, slot))
        self.landed = 0
        return [slot for _, slot in self.claimed], 0

    def _land(self, rows):
        for expert, slot in self.claimed[self.landed : rows]:
            for name in NAMES:
                self.cache.tensors[name][slot].copy_(self.reference[name][expert])
        self.landed = max(self.landed, rows)

    def fill_wait(self, rows):
        self.events.append(("wait", rows))
        self._land(rows)

    def fill_landed(self):
        self._land(min(len(self.claimed), self.landed + self.progress))
        return self.landed

    def fill_end(self):
        self.events.append(("end",))
        if self.fail:
            for _, slot in self.claimed[self.landed :]:
                self.table.release(slot)
            return False
        self._land(len(self.claimed))
        return True


def _setup(capacity=6, experts=8):
    reference = {
        name: torch.arange(experts * 4, dtype=torch.int16).reshape(experts, 4) + 100 * i
        for i, name in enumerate(NAMES)
    }
    layer = torch.nn.Module()
    layer.layer_id = 0
    streamer = ExpertStreamer(layer, NAMES, format=SpecOnlyFormat(reference))
    table = PinnedSlotLRU(capacity)
    fills = FakeFills(table, reference)
    cache = ExpertPinnedHostCache(streamer, capacity, device="cpu", slot_table=table, row_fills=fills)
    fills.cache = cache
    streamer.read_host_rows = lambda *args, **kwargs: pytest.fail("the row source read a row")
    return streamer, cache, fills, reference


def _outputs(reference, rows):
    return {n: torch.empty((rows,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in reference.items()}


def _check(outputs, reference, experts):
    for name in NAMES:
        assert torch.equal(outputs[name], reference[name][list(experts)]), name


def test_without_row_fills_the_row_source_reads_as_before():
    streamer, cache, fills, reference = _setup()
    cache.row_fills = None
    calls = []
    streamer.read_host_rows = lambda ids, destinations, slots: calls.append(ids.tolist()) or [
        destinations[n][slots].copy_(reference[n][ids]) for n in NAMES
    ]
    cache.ensure_rows(torch.tensor([3, 1]))
    assert calls == [[3, 1]] and fills.events == []


def test_ensure_rows_reads_through_the_fills_and_joins_them():
    streamer, cache, fills, reference = _setup()
    cache.ensure_rows(torch.tensor([3, 1, 3]), protected=[5])
    assert fills.events == [("begin", [3, 1], [1, 3, 5], True), ("end",)]
    outputs = _outputs(reference, 2)
    cache.copy_rows(torch.tensor([3, 1]), outputs)
    _check(outputs, reference, [3, 1])
    assert cache.stats.populated_rows == 2
    # Counted as reads in the stream trace (rows and blocked time), with no split.
    assert streamer.background_read_stats.rows == 2 and streamer.background_read_stats.split_ns == 0


def test_a_failed_fill_raises_from_ensure_rows():
    streamer, cache, fills, reference = _setup()
    fills.fail = True
    with pytest.raises(RuntimeError, match="failed"):
        cache.ensure_rows(torch.tensor([2]))
    assert 2 not in cache._lru and cache.expert_to_slot[2].item() == -1


def test_gather_rows_waits_for_each_chunks_own_prefetched_rows_before_copying():
    streamer, cache, fills, reference = _setup()
    cache.ensure_rows(torch.tensor([4]))
    fills.events.clear()
    with cache.host_use():
        assert cache.prefetch_rows([1, 2, 4, 6], protected=[1, 2, 4, 6]) == 3
        first = _outputs(reference, 2)
        cache.gather_rows(torch.tensor([1, 4]), first)
        assert fills.events[-1] == ("wait", 1)  # expert 1 is the first claimed; 4 was resident
        second = _outputs(reference, 2)
        cache.gather_rows(torch.tensor([2, 6]), second)
        assert fills.events[-1] == ("wait", 3)
        cache.finish_fills()
    assert fills.events[0] == ("begin", [1, 2, 6], [1, 2, 4, 6], False) and fills.events[-1] == ("end",)
    _check(first, reference, [1, 4])
    _check(second, reference, [2, 6])


@pytest.mark.parametrize("split", [False, True])
def test_a_chunk_miss_outside_the_prefetch_joins_it_before_admitting(split):
    with envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(split):
        streamer, cache, fills, reference = _setup()
    with cache.host_use():
        cache.prefetch_rows([1, 2], protected=[1, 2])
        outputs = _outputs(reference, 2)
        cache.gather_rows(torch.tensor([2, 7]), outputs)
        cache.finish_fills()
    assert [event[0] for event in fills.events] == ["begin", "end", "begin", "end"]
    assert fills.events[2][1] == [7]
    _check(outputs, reference, [2, 7])


def test_a_failed_prefetch_raises_when_joined():
    streamer, cache, fills, reference = _setup()
    fills.fail = True
    with cache.host_use():
        cache.prefetch_rows([1, 2], protected=[1, 2])
        with pytest.raises(RuntimeError, match="prefetch"):
            cache.finish_fills()
    assert 1 not in cache._lru and 2 not in cache._lru


def test_prefill_fills_prefetches_the_layers_vram_misses_in_ascending_order():
    streamer, cache, fills, reference = _setup()
    streamer.hot_cache = SimpleNamespace(
        capacity=2, expert_to_slot=torch.tensor([-1, -1, -1, 0, -1, -1, 1, -1])
    )
    flushes = []
    streamer.before_eager_gather = lambda: flushes.append(True)
    with streamer.prefill_fills(torch.tensor([6, 5, 3, 1])):
        assert fills.events == [("begin", [1, 5], [1, 3, 5, 6], False)]
    assert fills.events[-1] == ("end",) and flushes == [True]


def test_prefill_fills_does_nothing_without_row_fills():
    streamer, cache, fills, reference = _setup()
    cache.row_fills = None
    with streamer.prefill_fills(torch.tensor([1, 2])):
        pass
    assert fills.events == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def test_copy_rows_copies_only_the_named_rows_and_leaves_the_rest():
    streamer, cache, fills, reference = _setup()
    cache.ensure_rows(torch.tensor([3, 1, 5]))
    outputs = {n: torch.full_like(t, SENTINEL) for n, t in _outputs(reference, 3).items()}
    cache.copy_rows(torch.tensor([3, 1, 5]), outputs, rows=torch.tensor([2, 0]))
    for name in NAMES:
        assert torch.equal(outputs[name][0], reference[name][3])
        assert torch.equal(outputs[name][2], reference[name][5])
        assert (outputs[name][1] == SENTINEL).all()


def _split_setup(**kwargs):
    with envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(True):
        return _setup(**kwargs)


def _sentinel_outputs(reference, rows):
    return {n: torch.full_like(t, SENTINEL) for n, t in _outputs(reference, rows).items()}


def test_split_gather_copies_resident_rows_before_waiting_for_the_fills():
    """A chunk's resident rows must already be copied when the host starts waiting for its fills."""
    streamer, cache, fills, reference = _split_setup()
    cache.ensure_rows(torch.tensor([4]))
    seen = []
    wait = fills.fill_wait
    with cache.host_use():
        assert cache.prefetch_rows([1, 4, 6], protected=[1, 4, 6]) == 2  # claims 1 and 6; 4 is resident
        outputs = _sentinel_outputs(reference, 3)
        fills.fill_wait = lambda rows: (seen.append({n: o.clone() for n, o in outputs.items()}), wait(rows))
        cache.gather_rows(torch.tensor([1, 4, 6]), outputs)
        cache.finish_fills()
    for name in NAMES:
        assert torch.equal(seen[0][name][1], reference[name][4])  # the resident row, copied before the wait
        assert (seen[0][name][0] == SENTINEL).all() and (seen[0][name][2] == SENTINEL).all()
    _check(outputs, reference, [1, 4, 6])


def test_split_gather_places_every_chunks_rows_like_the_unsplit_gather():
    """Two chunks with filling rows at different positions: byte-identical to the flag-off gather."""
    got = None
    for split in (False, True):
        with envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(split):
            streamer, cache, fills, reference = _setup(capacity=6, experts=8)
        cache.ensure_rows(torch.tensor([0, 5]))
        with cache.host_use():
            cache.prefetch_rows([0, 2, 3, 5, 7], protected=[0, 2, 3, 5, 7])
            first, second = _sentinel_outputs(reference, 3), _sentinel_outputs(reference, 3)
            cache.gather_rows(torch.tensor([0, 2, 5]), first)
            cache.gather_rows(torch.tensor([7, 5, 3]), second)  # filling at 0 and 2, resident 5 between
            cache.finish_fills()
        _check(first, reference, [0, 2, 5])
        _check(second, reference, [7, 5, 3])
        if got is None:
            got = (first, second)
        else:
            for a, b in zip(got, (first, second)):
                for name in NAMES:
                    assert torch.equal(a[name], b[name])


def test_a_chunk_of_only_filling_rows_waits_then_copies_once():
    streamer, cache, fills, reference = _split_setup()
    copies = []
    copy = cache.copy_rows
    cache.copy_rows = lambda ids, outputs, rows=None: copies.append(rows) or copy(ids, outputs, rows=rows)
    with cache.host_use():
        cache.prefetch_rows([1, 6], protected=[1, 6])
        outputs = _sentinel_outputs(reference, 2)
        cache.gather_rows(torch.tensor([1, 6]), outputs)
        cache.finish_fills()
    assert copies == [None] and fills.events[1] == ("wait", 2)
    _check(outputs, reference, [1, 6])


def test_a_failed_fill_still_raises_with_the_split():
    streamer, cache, fills, reference = _split_setup()
    cache.ensure_rows(torch.tensor([4]))

    def failing_wait(rows):
        raise RuntimeError("a prefetch of pinned host rows failed")

    with cache.host_use():
        cache.prefetch_rows([1, 4], protected=[1, 4])
        fills.fill_wait = failing_wait
        with pytest.raises(RuntimeError, match="prefetch"):
            cache.gather_rows(torch.tensor([1, 4]), _sentinel_outputs(reference, 2))
        cache.finish_fills()


def _snapshot_at_waits(fills, outputs):
    """Record the outputs as they are each time the host starts waiting for a fill."""
    seen = []
    wait = fills.fill_wait
    fills.fill_wait = lambda rows: (seen.append((rows, {n: o.clone() for n, o in outputs.items()})), wait(rows))
    return seen


def _copied(snapshot, row, reference, expert):
    return all(torch.equal(snapshot[name][row], reference[name][expert]) for name in NAMES)


def _untouched(snapshot, row):
    return all((snapshot[name][row] == SENTINEL).all() for name in NAMES)


def test_a_row_that_landed_in_an_earlier_chunks_wait_is_copied_before_the_next_wait():
    streamer, cache, fills, reference = _split_setup()
    with cache.host_use():
        cache.prefetch_rows([1, 2, 5, 6], protected=[1, 2, 5, 6])
        cache.gather_rows(torch.tensor([1, 5]), _sentinel_outputs(reference, 2))  # waits for 1, 2 and 5
        outputs = _sentinel_outputs(reference, 2)
        seen = _snapshot_at_waits(fills, outputs)
        cache.gather_rows(torch.tensor([2, 6]), outputs)
        cache.finish_fills()
    assert [rows for rows, _ in seen] == [4]
    assert _copied(seen[0][1], 0, reference, 2) and _untouched(seen[0][1], 1)
    _check(outputs, reference, [2, 6])


def test_rows_the_reader_landed_before_the_chunk_are_copied_before_its_wait():
    streamer, cache, fills, reference = _split_setup()
    with cache.host_use():
        cache.prefetch_rows([1, 2, 3, 6], protected=[1, 2, 3, 6])
        fills._land(2)  # the fill thread landed 1 and 2 while the host was elsewhere
        outputs = _sentinel_outputs(reference, 4)
        seen = _snapshot_at_waits(fills, outputs)
        cache.gather_rows(torch.tensor([1, 2, 3, 6]), outputs)
        cache.finish_fills()
    assert [rows for rows, _ in seen] == [4]
    snapshot = seen[0][1]
    assert _copied(snapshot, 0, reference, 1) and _copied(snapshot, 1, reference, 2)
    assert _untouched(snapshot, 2) and _untouched(snapshot, 3)
    _check(outputs, reference, [1, 2, 3, 6])


def test_a_chunks_filling_rows_are_copied_batch_by_batch_as_they_land():
    """Twenty filling rows: the host waits for eight at a time and copies each batch before waiting for the next."""
    streamer, cache, fills, reference = _split_setup(capacity=24, experts=24)
    experts = list(range(20))
    with cache.host_use():
        assert cache.prefetch_rows(experts, protected=experts) == 20
        outputs = _sentinel_outputs(reference, 20)
        seen = _snapshot_at_waits(fills, outputs)
        cache.gather_rows(torch.tensor(experts), outputs)
        cache.finish_fills()
    assert [rows for rows, _ in seen] == [8, 16, 20]
    for (_, snapshot), done in zip(seen, (0, 8, 16)):
        assert all(_copied(snapshot, row, reference, row) for row in range(done))
        assert all(_untouched(snapshot, row) for row in range(done, 20))
    _check(outputs, reference, experts)


@pytest.mark.parametrize("progress", [0, 3])
def test_three_chunks_with_resident_landed_and_filling_rows_place_like_the_unsplit_gather(progress):
    got = None
    chunks = ([9, 0, 2, 11], [3, 5, 12, 4], [13, 7, 1, 10])
    for split in (False, True):
        with envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(split):
            streamer, cache, fills, reference = _setup(capacity=14, experts=14)
        cache.ensure_rows(torch.tensor([0, 5, 7]))
        fills.progress = progress
        with cache.host_use():
            prefetch = sorted({e for chunk in chunks for e in chunk} - {0, 5, 7})
            cache.prefetch_rows(prefetch, protected=prefetch + [0, 5, 7])
            outputs = [_sentinel_outputs(reference, len(chunk)) for chunk in chunks]
            for chunk, output in zip(chunks, outputs):
                cache.gather_rows(torch.tensor(chunk), output)
            cache.finish_fills()
        for chunk, output in zip(chunks, outputs):
            _check(output, reference, chunk)
        if got is None:
            got = outputs
        else:
            for a, b in zip(got, outputs):
                for name in NAMES:
                    assert torch.equal(a[name], b[name])
