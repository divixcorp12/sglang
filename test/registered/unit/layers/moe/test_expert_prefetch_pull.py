"""Stage C2 bullets 6-7: PrefetchPuller correctness and lifecycle against a real-shaped hot cache.

``ExpertHotCache`` (expert_hot_cache.py, out of C2's scope -- see stage-c2-report.md)
does not yet allocate ``DedicatedPrefetchSlot``'s trailing row: its tensors are
``(capacity + scratch_rows, ...)``, one row short of ``capacity + scratch_rows + 1``.
``_FakeHotCache`` below is a test-only stand-in shaped the way a hot cache with that
row *would* be, so ``PrefetchPuller`` itself -- construction, the setup assertions
against real allocation/real expert_to_slot, the captured post/join, and the
covered/residual remap -- is exercised end to end through the real production
classes. The one missing piece, confirmed separately, is that constructing a
``PrefetchPuller`` against the *actual* production ``ExpertHotCache`` raises at
setup (``assert_within_allocation``) rather than silently indexing out of bounds --
see ``test_prefetch_puller_rejects_a_hot_cache_allocated_without_the_trailing_row``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

EXPERTS, ROW_ELEMS = 32, 64
TENSOR_NAMES = ("w",)


class _FakeStreamer:
    def __init__(self, layer, tensor_names=TENSOR_NAMES):
        self.layer = layer
        self.tensor_names = tensor_names


class _FakeHotCache:
    """Shaped like ``ExpertHotCache`` but WITH the trailing dedicated-slot row."""

    def __init__(self, streamer, capacity, scratch_rows, device, dtype=torch.float32):
        self.streamer = streamer
        self.capacity = capacity
        self.scratch_rows = scratch_rows
        self.tensors = {
            name: torch.zeros((capacity + scratch_rows + 1, ROW_ELEMS), dtype=dtype, device=device)
            for name in streamer.tensor_names
        }
        self.expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)


def _source_layer(device):
    layer = torch.nn.Module()
    # Pinned host source: every row's bytes are its own expert id, broadcast, so a
    # byte-exact check against "the correct source row" needs no bookkeeping table.
    rows = torch.arange(EXPERTS, dtype=torch.float32).unsqueeze(1).expand(EXPERTS, ROW_ELEMS).contiguous()
    layer.w = rows.pin_memory()
    return layer


def _set_candidate(bank, hot_caches, target_layer, expert_id):
    """Make ``expert_id`` the bank's top (and only, width=1) candidate for ``target_layer``.

    Goes through the real ``PrefetchCandidateBank.write`` public path -- a one-hot
    score row -- rather than poking bank internals directly.
    """
    scores = torch.zeros(1, EXPERTS, device=bank.ids.device)
    scores[0, expert_id] = 1.0
    bank.write(target_layer, scores, expert_to_slot=hot_caches[target_layer].expert_to_slot)


def _build_puller(device, *, layer_ids=(1,), capacity=6, scratch_rows=2, enable_hits=(), pull_mode="always"):
    from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
    from sglang.srt.layers.moe.expert_prediction.serving.runtime import PrefetchPuller

    layer = _source_layer(device)
    streamer = _FakeStreamer(layer)
    hot_caches = {lid: _FakeHotCache(streamer, capacity, scratch_rows, device) for lid in layer_ids}
    for lid in enable_hits:
        hot_caches[lid].expert_to_slot[0] = 0
    bank = PrefetchCandidateBank(layer_ids=list(layer_ids), width=1, device=device)
    kwargs = dict(bank=bank, layer_ids=list(layer_ids), hot_caches=hot_caches, device=device)
    if pull_mode != "always":
        kwargs["pull_mode"] = pull_mode
    puller = PrefetchPuller(**kwargs)
    return puller, bank, hot_caches, layer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchPullerCorrectness(unittest.TestCase):
    def test_pulled_row_is_byte_exact_from_the_predicted_expert_not_a_neighbor(self):
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device)
        predicted_expert = 19
        _set_candidate(bank, hot_caches, 1, predicted_expert)
        puller.post_target(1)
        puller.join_target(
            1,
            flat_ids=torch.tensor([predicted_expert], device=device),
            missed_mask=torch.tensor([True], device=device),
            demand_remap=torch.tensor([0], dtype=torch.int64, device=device),
        )
        torch.cuda.synchronize()
        slot = puller._slots[1].index
        delivered = hot_caches[1].tensors["w"][slot]
        expected = torch.full((ROW_ELEMS,), float(predicted_expert), device=device)
        torch.testing.assert_close(delivered, expected)
        # A wrong-expert bug that preserves shape/dtype is invisible to anything
        # except this: assert it is NOT a neighboring expert's row either.
        for neighbor in (predicted_expert - 1, predicted_expert + 1):
            self.assertFalse(bool((delivered == float(neighbor)).all()))

    def test_demand_scratch_and_resident_rows_are_never_overwritten_by_the_pull(self):
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device, capacity=6, scratch_rows=2)
        cache = hot_caches[1]
        sentinel = -1.0
        cache.tensors["w"].fill_(sentinel)
        _set_candidate(bank, hot_caches, 1, 3)
        puller.post_target(1)
        puller.join_target(
            1,
            flat_ids=torch.tensor([3], device=device),
            missed_mask=torch.tensor([True], device=device),
            demand_remap=torch.tensor([0], dtype=torch.int64, device=device),
        )
        torch.cuda.synchronize()
        # Rows [0, capacity + scratch_rows) belong to residency slots and demand
        # scratch; the pull must only ever touch its own trailing row.
        untouched = cache.tensors["w"][: cache.capacity + cache.scratch_rows]
        self.assertTrue(bool((untouched == sentinel).all()))

    def test_covered_route_reads_the_dedicated_slot_not_the_ordinary_demand_remap(self):
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device)
        predicted_expert = 11
        _set_candidate(bank, hot_caches, 1, predicted_expert)
        puller.post_target(1)
        demand_remap = torch.tensor([777], dtype=torch.int64, device=device)
        remap = puller.join_target(
            1,
            flat_ids=torch.tensor([predicted_expert], device=device),
            missed_mask=torch.tensor([True], device=device),
            demand_remap=demand_remap,
        )
        self.assertEqual(remap.item(), puller._slots[1].index)
        self.assertNotEqual(remap.item(), 777)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchPullerLifecycle(unittest.TestCase):
    def test_graph_replay_resets_metadata_when_a_live_candidate_becomes_resident(self):
        # The candidate bank keeps an in-range ID for graph stability. Replays
        # must instead publish -1/count=0 every time the live map makes that
        # candidate ineligible; otherwise a prior offer can leak into a later
        # forward and suppress its demand route.
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device)
        candidate = 13
        _set_candidate(bank, hot_caches, 1, candidate)
        cache = hot_caches[1]
        flat_ids = torch.tensor([candidate], device=device)
        missed_mask = torch.tensor([True], device=device)
        demand_remap = torch.tensor([123], dtype=torch.int64, device=device)

        def step():
            puller.post_target(1)
            return puller.join_target(
                1,
                flat_ids=flat_ids,
                missed_mask=missed_mask,
                demand_remap=demand_remap,
            )

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            remap = step()

        for becomes_resident in (True, True, False, True):
            cache.expert_to_slot[candidate] = 0 if becomes_resident else -1
            graph.replay()
            torch.cuda.synchronize()
            plan = puller._plans[1]
            if becomes_resident:
                self.assertEqual((plan.expert_ids.item(), plan.count.item()), (-1, 0))
                self.assertEqual(remap.item(), demand_remap.item())
            else:
                self.assertEqual((plan.expert_ids.item(), plan.count.item()), (candidate, 1))
                self.assertEqual(remap.item(), puller.slot_for(1))

    def test_post_rechecks_residency_and_resets_a_former_offer_to_no_offer(self):
        # A candidate can become resident after scoring and before the graph's
        # posting node. It must publish a coherent no-offer plan, rather than
        # copying the now-resident fallback row or retaining an earlier ID.
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device)
        candidate = 13
        _set_candidate(bank, hot_caches, 1, candidate)
        hot_caches[1].expert_to_slot[candidate] = 0
        demand_remap = torch.tensor([123], dtype=torch.int64, device=device)
        # Calling post after the live residency update is the production order.
        puller.post_target(1)
        remap = puller.join_target(
            1,
            flat_ids=torch.tensor([candidate], device=device),
            missed_mask=torch.tensor([True], device=device),
            demand_remap=demand_remap,
        )
        torch.cuda.synchronize()
        plan = puller._plans[1]
        self.assertEqual((plan.expert_ids.item(), plan.count.item()), (-1, 0))
        self.assertEqual(remap.item(), demand_remap.item())

    def test_graph_replay_publishes_a_coherent_zero_payload_plan_without_touching_the_dedicated_slot(self):
        """A count-zero control replays the normal post/join graph with no delivered row."""
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device, pull_mode="count_zero")
        candidate = 13
        _set_candidate(bank, hot_caches, 1, candidate)
        cache = hot_caches[1]
        slot = puller.slot_for(1)
        cache.tensors["w"].fill_(-1.0)
        flat_ids = torch.tensor([candidate], device=device)
        missed_mask = torch.tensor([True], device=device)
        demand_remap = torch.tensor([0], dtype=torch.int64, device=device)

        def step():
            puller.post_target(1)
            return puller.join_target(
                1,
                flat_ids=flat_ids,
                missed_mask=missed_mask,
                demand_remap=demand_remap,
            )

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            remap = step()
        puller.stats[1].counts.zero_()  # Ignore the capture warmup replay.

        original_demand_rows = demand_remap.clone()
        expected_slot = cache.tensors["w"][slot].clone()
        for should_post in (False, True, False, True):
            puller._should_post[1].fill_(should_post)
            graph.replay()
            torch.cuda.synchronize()
            plan = puller._plans[1]
            if should_post:
                self.assertEqual((plan.expert_ids.item(), plan.count.item()), (candidate, 1))
                self.assertEqual(remap.item(), slot)
                expected_slot = cache.tensors["w"][slot].clone()
            else:
                self.assertEqual((plan.expert_ids.item(), plan.count.item()), (-1, 0))
                self.assertEqual(remap.item(), original_demand_rows.item())
                torch.testing.assert_close(cache.tensors["w"][slot], expected_slot)
        covered, residual, wasted, posted = puller.stats[1].snapshot()
        self.assertEqual((covered, residual, wasted, posted), (2, 2, 0, 2))

    def test_setup_rejects_a_hot_cache_allocated_without_the_trailing_row(self):
        # Reject unsupported concurrent-state sharing at setup, not at first
        # corrupted read: a hot cache with no reserved trailing row would have
        # `slot.index` land one row past the real allocation.
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.runtime import PrefetchPuller

        device = torch.device("cuda")
        layer = _source_layer(device)
        streamer = _FakeStreamer(layer)
        cache = _FakeHotCache(streamer, capacity=6, scratch_rows=2, device=device)
        # Truncate to exactly what production ExpertHotCache.__init__ allocates today:
        # (capacity + scratch_rows), one row short of the dedicated slot.
        cache.tensors["w"] = cache.tensors["w"][:-1].clone()
        bank = PrefetchCandidateBank(layer_ids=[1], width=1, device=device)
        with self.assertRaises(ValueError):
            PrefetchPuller(bank=bank, layer_ids=[1], hot_caches={1: cache}, device=device)

    def test_setup_rejects_a_permanent_mapping_that_already_reaches_the_reserved_row(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.runtime import PrefetchPuller

        device = torch.device("cuda")
        layer = _source_layer(device)
        streamer = _FakeStreamer(layer)
        cache = _FakeHotCache(streamer, capacity=6, scratch_rows=2, device=device)
        cache.expert_to_slot[0] = cache.capacity + cache.scratch_rows  # the reserved index
        bank = PrefetchCandidateBank(layer_ids=[1], width=1, device=device)
        with self.assertRaises(RuntimeError):
            PrefetchPuller(bank=bank, layer_ids=[1], hot_caches={1: cache}, device=device)

    def test_replay_picks_up_a_changed_candidate_without_python_between_replays(self):
        # Regression guard for a missing-hook staleness bug: a captured post/join must
        # read the bank's CURRENT row on every replay via the tensor's stable address,
        # never bake in the id that was live at capture time.
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device)
        flat_ids = torch.zeros(1, dtype=torch.long, device=device)
        missed_mask = torch.ones(1, dtype=torch.bool, device=device)
        demand_remap = torch.zeros(1, dtype=torch.int64, device=device)

        def step():
            flat_ids.copy_(bank.ids_for(1)[:1])
            puller.post_target(1)
            puller.join_target(1, flat_ids=flat_ids, missed_mask=missed_mask, demand_remap=demand_remap)

        _set_candidate(bank, hot_caches, 1, 5)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for expert_id in (5, 21, 5, 30):
            _set_candidate(bank, hot_caches, 1, expert_id)
            graph.replay()
            torch.cuda.synchronize()
            slot = puller._slots[1].index
            delivered = hot_caches[1].tensors["w"][slot]
            self.assertTrue(bool((delivered == float(expert_id)).all()))

    def test_eager_then_captured_replay_do_not_cross_contaminate_the_dedicated_slot(self):
        # Alternate an eager (uncaptured) pull with a captured-and-replayed one on the
        # same puller; each must deliver its own forward's prediction, not a leftover.
        device = torch.device("cuda")
        puller, bank, hot_caches, _ = _build_puller(device)
        slot = puller._slots[1].index

        _set_candidate(bank, hot_caches, 1, 4)
        puller.post_target(1)
        puller.join_target(
            1,
            flat_ids=torch.tensor([4], device=device),
            missed_mask=torch.tensor([True], device=device),
            demand_remap=torch.zeros(1, dtype=torch.int64, device=device),
        )
        torch.cuda.synchronize()
        self.assertTrue(bool((hot_caches[1].tensors["w"][slot] == 4.0).all()))

        flat_ids = torch.zeros(1, dtype=torch.long, device=device)
        missed_mask = torch.ones(1, dtype=torch.bool, device=device)
        demand_remap = torch.zeros(1, dtype=torch.int64, device=device)

        def step():
            flat_ids.copy_(bank.ids_for(1)[:1])
            puller.post_target(1)
            puller.join_target(1, flat_ids=flat_ids, missed_mask=missed_mask, demand_remap=demand_remap)

        _set_candidate(bank, hot_caches, 1, 17)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        _set_candidate(bank, hot_caches, 1, 22)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(bool((hot_caches[1].tensors["w"][slot] == 22.0).all()))


if __name__ == "__main__":
    unittest.main()
