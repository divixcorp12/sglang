"""CPU tests of the pinned tier's NUMA placement: parsing, row split, capacity check, binding, manager wiring."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_stream, host_numa
from sglang.srt.layers.moe.expert_host_tier import PAGE_BYTES, allocate_host_slab
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCacheManager, ExpertStreamer
from sglang.srt.layers.moe.host_numa import (
    MIB,
    allocate_bound,
    check_capacity,
    page_nodes,
    parse_placement,
    split_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HAS_NODE1 = os.path.exists("/sys/devices/system/node/node1")


class TestParsePlacement(unittest.TestCase):
    def test_empty_is_no_placement(self):
        self.assertEqual(parse_placement(""), ())
        self.assertEqual(parse_placement("  "), ())

    def test_node_mib_pairs_in_order(self):
        self.assertEqual(parse_placement("1:30, 0:65"), ((1, 30 * MIB), (0, 65 * MIB)))

    def test_refuses_malformed_zero_and_repeated_nodes(self):
        for value in ("0", "0:", "a:1", "0:-1", "0:1.5", "0:0", "0:1,0:2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_placement(value)


class TestSplitRows(unittest.TestCase):
    def test_runs_tile_the_rows_in_placement_order(self):
        runs = split_rows(10, ((0, 65), (1, 35)))
        self.assertEqual(runs, [(0, 0, 7), (1, 7, 3)])

    def test_counts_sum_exactly_for_every_row_count(self):
        placement = ((0, 3), (1, 3), (2, 1))
        for rows in range(0, 50):
            runs = split_rows(rows, placement)
            self.assertEqual(sum(count for _, _, count in runs), rows)
            starts = [start for _, start, _ in runs]
            self.assertEqual(starts, sorted(starts))

    def test_largest_remainder_ties_go_to_the_earlier_node(self):
        self.assertEqual(split_rows(1, ((0, 1), (1, 1))), [(0, 0, 1)])
        self.assertEqual(split_rows(3, ((0, 1), (1, 1))), [(0, 0, 2), (1, 2, 1)])

    def test_a_node_whose_share_rounds_to_nothing_gets_no_run(self):
        self.assertEqual(split_rows(2, ((0, 99), (1, 1))), [(0, 0, 2)])


class TestCheckCapacity(unittest.TestCase):
    def _root(self, nodes):
        root = tempfile.mkdtemp()
        for node, (free_kb, active_kb, inactive_kb) in nodes.items():
            os.makedirs(os.path.join(root, f"node{node}"))
            with open(os.path.join(root, f"node{node}", "meminfo"), "w") as f:
                f.write(f"Node {node} MemTotal:  99999999 kB\n")
                f.write(f"Node {node} MemFree:   {free_kb} kB\n")
                f.write(f"Node {node} Active(file):  {active_kb} kB\n")
                f.write(f"Node {node} Inactive(file): {inactive_kb} kB\n")
        return root

    def test_page_cache_counts_as_available(self):
        root = self._root({0: (1024, 0, 0), 1: (0, 1024, 1024)})
        check_capacity(((1, 2 * MIB),), headroom=0, root=root)

    def test_refuses_naming_the_short_node(self):
        root = self._root({0: (100 * 1024, 0, 0), 1: (1024, 0, 0)})
        with self.assertRaisesRegex(ValueError, r"node 1: asked 2 MiB"):
            check_capacity(((0, 2 * MIB), (1, 2 * MIB)), headroom=0, root=root)

    def test_headroom_is_required_on_top(self):
        root = self._root({0: (4 * 1024, 0, 0)})
        check_capacity(((0, 3 * MIB),), headroom=MIB, root=root)
        with self.assertRaises(ValueError):
            check_capacity(((0, 3 * MIB),), headroom=2 * MIB, root=root)

    def test_refuses_a_missing_node(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            check_capacity(((7, MIB),), root=self._root({0: (1, 0, 0)}))


class TestBinding(unittest.TestCase):
    def test_a_slab_bound_to_node0_lives_there(self):
        rows, row_bytes = 16, 3 * PAGE_BYTES
        slab = allocate_bound(rows * row_bytes, [(0, 0, rows)], row_bytes)
        self.assertEqual(slab.data_ptr() % PAGE_BYTES, 0)
        self.assertEqual(slab.numel(), rows * row_bytes)
        slab.fill_(1)
        self.assertEqual(page_nodes(slab, samples=32), {0: 32})

    def test_pages_are_not_touched_by_allocation(self):
        slab = allocate_bound(8 * PAGE_BYTES, [(0, 0, 8)], PAGE_BYTES)
        self.assertNotIn(0, page_nodes(slab, samples=8))

    @unittest.skipUnless(HAS_NODE1, "needs a second NUMA node")
    def test_row_runs_land_on_their_nodes(self):
        rows, row_bytes = 20, 5 * PAGE_BYTES
        runs = split_rows(rows, ((0, 3), (1, 1)))
        slab = allocate_bound(rows * row_bytes, runs, row_bytes)
        slab.fill_(1)
        (_, _, n0), (_, first1, _) = runs
        self.assertEqual(page_nodes(slab[: n0 * row_bytes], samples=16), {0: 16})
        self.assertEqual(page_nodes(slab[first1 * row_bytes :], samples=16), {1: 16})

    def test_allocate_host_slab_with_a_placement_keeps_shape_and_bytes(self):
        slab = allocate_host_slab(6, (5, 7), torch.int16, register=False, placement=((0, MIB),))
        self.assertEqual(tuple(slab.shape), (6, 5, 7))
        self.assertEqual(slab.dtype, torch.int16)
        self.assertEqual(slab.data_ptr() % PAGE_BYTES, 0)
        slab.copy_(torch.arange(6 * 35, dtype=torch.int16).view(6, 5, 7))
        self.assertEqual(int(slab[5, 4, 6]), 6 * 35 - 1)
        self.assertEqual(set(page_nodes(slab)), {0})


def _cpu_model():
    """Two spec-only EXL3-shaped layers whose pinned tiers stay on the CPU."""
    model = torch.nn.Module()
    for layer_id in range(2):
        generator = torch.Generator().manual_seed(layer_id)
        reference = {
            name: torch.randint(-(2**15), 2**15, (8, rows, 6), dtype=torch.int16, generator=generator)
            for name, rows in (("w13_trellis", 2), ("w2_trellis", 1))
        }
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer._nvfp4_expert_streamer = ExpertStreamer(
            layer, tuple(reference), format=SpecOnlyFormat(reference, tier_options={"device": "cpu"})
        )
        model.add_module(str(layer_id), layer)
    return model


class TestManagerPlacement(unittest.TestCase):
    def setUp(self):
        self.model = _cpu_model()

    def test_unset_keeps_first_touch_and_reports_no_placement(self):
        with envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override(""):
            self.assertEqual(expert_stream.pinned_host_placement(4 * MIB), ())

    def test_a_placement_that_disagrees_with_the_budget_is_refused(self):
        with envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("0:3"), self.assertRaisesRegex(ValueError, "must agree"):
            expert_stream.pinned_host_placement(4 * MIB)

    def test_the_capacity_check_runs_before_anything_is_allocated(self):
        with envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("0:4"), patch.object(
            host_numa, "check_capacity", side_effect=ValueError("short")
        ), patch.object(host_numa, "allocate_bound") as allocate:
            with self.assertRaisesRegex(ValueError, "short"):
                ExpertPinnedHostCacheManager.from_model(self.model, budget_bytes=4 * MIB)
            allocate.assert_not_called()

    def test_the_manager_binds_every_slab_and_logs_the_placement(self):
        with envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("0:4"), patch.object(host_numa, "check_capacity"):
            with self.assertLogs(expert_stream.logger, "INFO") as logs:
                manager = ExpertPinnedHostCacheManager.from_model(self.model, budget_bytes=4 * MIB)
        self.assertTrue(manager.caches)
        for cache in manager.caches.values():
            for slab in cache.tensors.values():
                slab.fill_(0)
                self.assertEqual(set(page_nodes(slab)), {0})
        self.assertIn('"numa": {"mib": {"0": 4}', "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
