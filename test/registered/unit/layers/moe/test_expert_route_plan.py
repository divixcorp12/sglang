"""CPU tests for multi-token route deduplication in streamed expert gathers."""

import random
import unittest
from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy
from sglang.srt.layers.moe.expert_route_plan import plan_graph_routes, should_dedup
from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPERTS = 16
TOP_K = 10


def _streamer():
    layer = torch.nn.Module()
    layer.rows = torch.arange(EXPERTS * 4, dtype=torch.uint8).reshape(EXPERTS, 4)
    return ExpertStreamer(layer, ("rows",))


def _verify_routes(seed, tokens=4):
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(tokens)]
    )


def _expert_to_slot(resident):
    mapping = torch.full((EXPERTS,), -1, dtype=torch.long)
    for slot, expert in enumerate(resident):
        mapping[expert] = slot
    return mapping


def _legacy_graph_plan(flat, expert_to_slot, capacity):
    """The graph gather's per-route plan before deduplication."""
    count = flat.numel()
    slots = expert_to_slot.index_select(0, flat)
    hit = slots >= 0
    scratch = torch.arange(capacity, capacity + count)
    order = torch.argsort(hit.to(torch.uint8), stable=True)
    return (
        torch.where(hit, slots, scratch),
        flat.index_select(0, order),
        int(count - hit.sum()),
    )


class _NoHostReads(TorchDispatchMode):
    FORBIDDEN = {
        "aten::_local_scalar_dense",
        "aten::item",
        "aten::nonzero",
        "aten::is_nonzero",
        "aten::equal",
    }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name in self.FORBIDDEN:
            raise AssertionError(f"host read {func._schema.name}")
        return func(*args, **(kwargs or {}))


class TestEagerRouteDedup(unittest.TestCase):
    def test_multi_token_verify_gathers_one_row_per_distinct_expert(self):
        streamer = _streamer()
        ids = _verify_routes(seed=3)
        self.assertLessEqual(ids.numel(), 64)

        source_ids, compact_ids = streamer._plan_eager_routes(ids)

        self.assertTrue(should_dedup(ids))
        self.assertEqual(source_ids.numel(), len(set(ids.flatten().tolist())))
        self.assertLess(source_ids.numel(), ids.numel())
        self.assertTrue(
            torch.equal(source_ids[compact_ids.long()].reshape(ids.shape), ids)
        )

    def test_single_token_decode_keeps_routes_without_dedup(self):
        streamer = _streamer()
        ids = _verify_routes(seed=4, tokens=1)

        source_ids, compact_ids = streamer._plan_eager_routes(ids)

        self.assertFalse(should_dedup(ids))
        self.assertFalse(should_dedup(ids.reshape(-1)))
        self.assertTrue(torch.equal(source_ids, ids.reshape(-1)))
        self.assertEqual(compact_ids.tolist(), list(range(TOP_K)))

    def test_residency_records_routed_multiplicity_after_dedup(self):
        streamer = _streamer()
        streamer.residency_policy = ExpertResidencyPolicy(EXPERTS, 4, device="cpu")
        ids = _verify_routes(seed=5)

        source_ids, _ = streamer._plan_eager_routes(ids)

        self.assertLess(source_ids.numel(), ids.numel())
        expected = torch.bincount(ids.reshape(-1), minlength=EXPERTS)
        policy = streamer.residency_policy
        torch.testing.assert_close(
            policy.pending_counts, expected.to(policy.pending_counts.dtype)
        )
        self.assertEqual(policy.snapshot_metrics()["recorded_routes"], ids.numel())

    def test_cached_gather_holds_the_kernel_expert_count_at_the_dedup_limit(self):
        """Deduplicated gathers must hand the fused-MoE kernel one expert count,
        not one that varies with distinct misses (4x slower decode, E10);
        single-token gathers keep their top-k rows unpadded."""
        experts = 80
        layer = torch.nn.Module()
        layer.rows = torch.arange(experts * 4, dtype=torch.int32).reshape(experts, 4)
        streamer = ExpertStreamer(layer, ("rows",))

        def all_miss(ids):
            return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)

        streamer.hot_cache = SimpleNamespace(lookup=all_miss, capacity=1)

        def copy_rows(source_ids, outputs):
            for name, output in outputs.items():
                torch.index_select(getattr(layer, name), 0, source_ids, out=output)
            return 0

        streamer._copy_source_rows = copy_rows
        for tokens, distinct, expected_rows in (
            (2, 3, 64),
            (2, 17, 64),
            (2, 40, 64),
            (2, 70, 70),
            (1, 10, 10),
        ):
            ids = torch.arange(distinct).repeat(tokens).reshape(tokens, distinct)
            source_ids, compact_ids = streamer._plan_eager_routes(ids)
            with self.subTest(tokens=tokens, distinct=distinct):
                compact, tensors = streamer._gather_cached(source_ids, compact_ids, ids)
                self.assertEqual(tensors["rows"].shape[0], expected_rows)
                self.assertEqual(streamer.last_gather_stats.requested_rows, distinct)
                self.assertTrue(
                    torch.equal(tensors["rows"][compact.long()], layer.rows[ids])
                )


class TestGraphRoutePlan(unittest.TestCase):
    def assert_plan_serves_routes(self, flat, resident, scratch_rows):
        capacity = len(resident)
        experts = torch.randn(EXPERTS, 3, generator=torch.Generator().manual_seed(1))
        plan = plan_graph_routes(flat, _expert_to_slot(resident), scratch_rows, capacity)
        misses = int(plan.miss_plan_rows)
        distinct_misses = set(flat.tolist()) - set(resident)

        self.assertEqual(int(plan.unique_miss_rows), len(distinct_misses))
        self.assertEqual(
            int(plan.unique_hit_rows), len(set(flat.tolist()) & set(resident))
        )
        self.assertEqual(
            int(plan.routed_miss_rows),
            sum(expert not in resident for expert in flat.tolist()),
        )
        for rows_written in (misses, plan.source_rows.numel()):
            cache = torch.zeros(capacity + scratch_rows, 3)
            if capacity:
                cache[:capacity] = experts[torch.tensor(resident)]
            cache[capacity : capacity + rows_written] = experts[
                plan.source_rows[:rows_written]
            ]
            self.assertTrue(torch.equal(cache[plan.remap], experts[flat]))
        scratch_used = plan.remap[plan.remap >= capacity]
        self.assertEqual(scratch_used.unique().numel(), len(distinct_misses))
        self.assertLess(int(plan.remap.max()), capacity + scratch_rows)

    def test_remap_resolves_every_route_with_one_scratch_row_per_distinct_miss(self):
        scratch_rows = 12
        for seed in range(200):
            rng = random.Random(seed)
            count = rng.randint(1, scratch_rows)
            flat = torch.tensor(
                [rng.randrange(EXPERTS) for _ in range(count)], dtype=torch.long
            )
            resident = rng.sample(range(EXPERTS), rng.randint(0, 8))
            with self.subTest(seed=seed):
                self.assert_plan_serves_routes(flat, resident, scratch_rows)
        flat = torch.tensor([3, 5, 3, 9, 5, 1], dtype=torch.long)
        for name, routes, resident in (
            ("all_hit", flat, [1, 3, 5, 9]),
            ("all_miss", flat, [0, 2]),
            ("all_same", torch.full((scratch_rows,), 7, dtype=torch.long), [0]),
            ("all_same_hit", torch.full((scratch_rows,), 7, dtype=torch.long), [7]),
        ):
            with self.subTest(name=name):
                self.assert_plan_serves_routes(routes, resident, scratch_rows)

    def test_plan_is_exact_when_distinct_misses_fill_every_scratch_row(self):
        routes = torch.tensor([4, 1, 3, 2], dtype=torch.long)
        for resident in ([], [0], [5, 6, 7]):
            with self.subTest(resident=resident):
                plan = plan_graph_routes(
                    routes, _expert_to_slot(resident), routes.numel(), len(resident)
                )
                self.assertEqual(int(plan.miss_plan_rows), routes.numel())
                self.assertEqual(
                    sorted(plan.remap.tolist()),
                    list(range(len(resident), len(resident) + routes.numel())),
                )
                self.assert_plan_serves_routes(routes, resident, routes.numel())

    def test_planning_reads_no_device_value_on_the_host(self):
        flat = torch.tensor([4, 2, 4, 9, 0, 2, 7, 7], dtype=torch.long)
        expert_to_slot = _expert_to_slot([2, 7])
        with self.assertRaises(AssertionError):
            with _NoHostReads():
                torch.tensor([1]).sum().item()

        with _NoHostReads():
            plan = plan_graph_routes(flat, expert_to_slot, 8, 2)

        self.assert_plan_serves_routes(flat, [2, 7], 8)
        self.assertEqual(int(plan.unique_miss_rows), 3)

    def test_single_token_decode_plans_same_rows_as_pre_dedup_gather(self):
        for seed in range(50):
            rng = random.Random(seed)
            flat = torch.tensor(rng.sample(range(EXPERTS), TOP_K), dtype=torch.long)
            resident = rng.sample(range(EXPERTS), rng.randint(0, 12))
            capacity = len(resident)
            expert_to_slot = _expert_to_slot(resident)
            legacy_remap, legacy_rows, legacy_misses = _legacy_graph_plan(
                flat, expert_to_slot, capacity
            )

            plan = plan_graph_routes(flat, expert_to_slot, TOP_K, capacity)

            hit = expert_to_slot[flat] >= 0
            with self.subTest(seed=seed):
                self.assertTrue(torch.equal(plan.source_rows, legacy_rows))
                self.assertEqual(int(plan.miss_plan_rows), legacy_misses)
                self.assertEqual(int(plan.unique_miss_rows), legacy_misses)
                self.assertEqual(int(plan.routed_miss_rows), legacy_misses)
                self.assertTrue(torch.equal(plan.remap[hit], legacy_remap[hit]))
                self.assertEqual(
                    sorted(plan.remap[~hit].tolist()),
                    list(range(capacity, capacity + legacy_misses)),
                )
                self.assertTrue(
                    torch.equal(
                        plan.source_rows[plan.remap[~hit] - capacity], flat[~hit]
                    )
                )


if __name__ == "__main__":
    unittest.main()
