"""CPU tests of the pinned tier's NUMA placement: parsing, row split, capacity check, binding, manager wiring."""

import errno
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
    HUGE_BYTES,
    MIB,
    address_policy,
    allocate_bound,
    check_capacity,
    group_ranges,
    page_nodes,
    parse_placement,
    plan_bindings,
    slot_nodes,
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


class TestPlanBindings(unittest.TestCase):
    """The mbind ranges: tile the span out to its 2 MiB end, change node only on 2 MiB boundaries, keep each total.

    Totals count the tail past ``nbytes`` (up to the 2 MiB end) as bound to the last range's node, as
    ``_numa_bound_bytes`` does: each node's bound bytes, tail included, stay within 2 MiB of its ask.
    """

    def _check_tiling(self, nbytes, bindings):
        self.assertEqual(bindings[0][1], 0)
        self.assertEqual(bindings[-1][2], -(-nbytes // HUGE_BYTES) * HUGE_BYTES)
        self.assertEqual(bindings[-1][2] % HUGE_BYTES, 0)  # the last range ends on a 2 MiB boundary of the base
        for (node, _, end), (following, start, _) in zip(bindings, bindings[1:]):
            self.assertEqual(end, start)  # no gap, no overlap
            self.assertNotEqual(node, following)  # same-node neighbours are merged
            self.assertEqual(start % HUGE_BYTES, 0)

    @staticmethod
    def _totals(ranges, nbytes=None):
        totals = {}
        for node, lo, hi in ranges:
            hi = hi if nbytes is None else min(hi, nbytes)
            totals[node] = totals.get(node, 0) + hi - lo
        return totals

    def test_a_single_node_binds_the_whole_span_out_to_its_2mib_end(self):
        # Replaces the page-rounded end (10 MiB + a page): the tail up to 12 MiB is bound too, to the same node.
        self.assertEqual(plan_bindings(10 * MIB + 5, [(1, 0, 10 * MIB + 5)]), [(1, 0, 12 * MIB)])
        self.assertEqual(plan_bindings(4 * MIB, [(0, 0, 4 * MIB)]), [(0, 0, 4 * MIB)])  # already aligned: no tail

    def test_the_last_range_ends_on_a_2mib_boundary_bound_to_the_last_runs_node(self):
        # A separate production slab (SLAB_ARENA=0): its page-rounded end is not 2 MiB aligned. Ending the last
        # binding there split the VMA and left the last huge page on 4 KiB folios, so the slab's last registered
        # chunk could not coalesce (final review, Important 1).
        row_bytes = 3_501_056
        for rows, share in ((460, ((0, 5), (1, 4))), (181, ((0, 60), (1, 40))), (182, ((1, 1), (0, 1))), (1, ((0, 1),))):
            runs = [(node, first * row_bytes, (first + count) * row_bytes) for node, first, count in split_rows(rows, share)]
            nbytes = rows * row_bytes
            with self.subTest(rows=rows, share=share):
                self.assertNotEqual(nbytes % HUGE_BYTES, 0)
                bindings = plan_bindings(nbytes, runs)
                self._check_tiling(nbytes, bindings)
                node, start, end = bindings[-1]
                self.assertEqual(node, runs[-1][0])
                self.assertEqual(end, -(-nbytes // HUGE_BYTES) * HUGE_BYTES)
                self.assertLess(end - nbytes, HUGE_BYTES)
                self.assertLessEqual(start, end - HUGE_BYTES)  # the last huge page lies wholly in this range
                bound, asked = self._totals(bindings), self._totals(runs)
                self.assertEqual(sum(bound.values()), end)  # the tail is counted, to the last node
                for n in asked:
                    self.assertLessEqual(abs(bound.get(n, 0) - asked[n]), HUGE_BYTES, (n, bound, asked))

    def test_nothing_to_bind_is_no_range(self):
        self.assertEqual(plan_bindings(0, []), [])
        self.assertEqual(plan_bindings(PAGE_BYTES, [(0, 0, 0)]), [])

    def test_unaligned_row_runs_change_node_on_a_2mib_boundary(self):
        rows, row_bytes = 40, 3_501_056  # dsv41-sized rows: the ideal boundary is not 2 MiB aligned
        runs = [(node, first * row_bytes, (first + count) * row_bytes) for node, first, count in split_rows(rows, ((0, 60), (1, 40)))]
        self.assertNotEqual(runs[1][1] % HUGE_BYTES, 0)
        bindings = plan_bindings(rows * row_bytes, runs)
        self._check_tiling(rows * row_bytes, bindings)
        self.assertEqual([node for node, _, _ in bindings], [0, 1])
        self.assertLessEqual(abs(bindings[0][2] - runs[1][1]), HUGE_BYTES)

    def test_adjacent_runs_on_one_node_merge_into_one_range(self):
        runs = [(0, 0, 3 * MIB), (0, 3 * MIB, 7 * MIB), (1, 7 * MIB, 12 * MIB), (1, 12 * MIB, 20 * MIB)]
        bindings = plan_bindings(20 * MIB, runs)
        self._check_tiling(20 * MIB, bindings)
        self.assertEqual([node for node, _, _ in bindings], [0, 1])

    def test_many_slab_totals_stay_within_2mib_where_nearest_rounding_drifts(self):
        # Node 0 asks 4.8 MiB then node 1 5.2 MiB, 30 times: every 0->1 change is ideally at 1.4 MiB past a 2 MiB
        # boundary (nearest: up) and every 1->0 change at 0.6 MiB past one (nearest: down), so rounding each to its
        # nearest gives node 0 1.2 MiB extra per slab, 36 MiB in all. The carried error must not drift.
        runs = []
        for k in range(30):
            base = k * 10 * MIB + 6 * MIB // 10
            runs += [(0, base, base + 48 * MIB // 10), (1, base + 48 * MIB // 10, base + 10 * MIB)]
        runs[0] = (0, 0, runs[0][2])
        nbytes = runs[-1][2]
        asked = self._totals(runs)
        span = -(-nbytes // PAGE_BYTES) * PAGE_BYTES
        cuts = [0] + [(lo + HUGE_BYTES // 2) // HUGE_BYTES * HUGE_BYTES for _, lo, _ in runs[1:]] + [span]
        naive = self._totals([(node, cuts[i], cuts[i + 1]) for i, (node, _, _) in enumerate(runs)])
        self.assertGreater(abs(naive[0] - asked[0]), 10 * HUGE_BYTES)  # the case does drift under nearest rounding
        bindings = plan_bindings(nbytes, runs)
        self._check_tiling(nbytes, bindings)
        bound = self._totals(bindings)
        for node in (0, 1):
            self.assertLessEqual(abs(bound[node] - asked[node]), HUGE_BYTES, (node, bound, asked))

    def test_arena_like_layouts_keep_every_node_within_2mib(self):
        rows_bytes = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)  # the dsv41 EXL3 slabs' rows
        for rows in (1, 7, 37, 181, 182):
            for share in ((0, 60), (1, 40)), ((0, 5), (1, 4)), ((1, 1), (0, 1)):
                runs, offset = [], 0
                for _ in range(3):  # three layers' worth of slab joins
                    for row_bytes in rows_bytes:
                        offset = -(-offset // PAGE_BYTES) * PAGE_BYTES
                        for node, first, count in split_rows(rows, share):
                            runs.append((node, offset + first * row_bytes, offset + (first + count) * row_bytes))
                        offset += rows * row_bytes
                with self.subTest(rows=rows, share=share):
                    bindings = plan_bindings(offset, runs)
                    self._check_tiling(offset, bindings)
                    bound, asked = self._totals(bindings), self._totals(runs)
                    for node in asked:
                        self.assertLessEqual(abs(bound.get(node, 0) - asked[node]), HUGE_BYTES)

    def test_a_run_shorter_than_2mib_can_round_away_to_its_neighbours_node(self):
        runs = [(0, 0, 8 * MIB), (1, 8 * MIB, 8 * MIB + 100_000), (0, 8 * MIB + 100_000, 16 * MIB)]
        bindings = plan_bindings(16 * MIB, runs)
        self._check_tiling(16 * MIB, bindings)
        bound = self._totals(bindings)
        self.assertLessEqual(abs(bound.get(1, 0) - 100_000), HUGE_BYTES)


def _mbind_permitted() -> bool:
    try:
        allocate_bound(PAGE_BYTES, [(0, 0, 1)], PAGE_BYTES)
    except OSError as error:
        if error.errno in (errno.EPERM, errno.ENOSYS):
            return False
        raise
    return True


MBIND_PERMITTED = _mbind_permitted()
MPOL_DEFAULT, MPOL_BIND = 0, 2


def _vm_flags(address: int) -> set[str]:
    """The kernel's VmFlags for the mapping holding ``address`` (/proc/self/smaps)."""
    inside = False
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            head = line.split()[0]
            if "-" in head and not head.endswith(":"):
                lo, hi = (int(part, 16) for part in head.split("-"))
                inside = lo <= address < hi
            elif inside and head == "VmFlags:":
                return set(line.split()[1:])
    raise AssertionError(f"no mapping holds {address:#x}")


@unittest.skipUnless(os.path.exists("/proc/self/smaps"), "needs Linux /proc/self/smaps")
class TestHugePages(unittest.TestCase):
    def test_the_slab_mapping_asks_for_transparent_huge_pages(self):
        """With THP in madvise mode, or defrag=madvise, only a madvised mapping is promised 2 MiB pages: the CPU
        expert kernel's traversal touches a new 4 KiB page on nearly every weight load, which costs 1-3% on 4 KiB
        pages (bench A/B on divix01, 2026-10-02)."""
        with patch.object(host_numa, "_mbind"):
            slab = allocate_bound(5 * MIB + 7, [(0, 0, 1)], 5 * MIB + 7)
        self.assertIn("hg", _vm_flags(slab.data_ptr()))
        self.assertIn("hg", _vm_flags(slab.data_ptr() + slab.numel() - 1))


@unittest.skipUnless(MBIND_PERMITTED, "mbind is not permitted here (seccomp without CAP_SYS_NICE)")
class TestBinding(unittest.TestCase):
    def test_every_page_of_a_run_is_bound_and_plain_memory_is_not(self):
        rows, row_bytes = 16, 3 * PAGE_BYTES
        slab = allocate_bound(rows * row_bytes, [(0, 0, rows)], row_bytes)
        for address in (slab.data_ptr(), slab.data_ptr() + rows * row_bytes - PAGE_BYTES):
            self.assertEqual(address_policy(address), (MPOL_BIND, frozenset({0})))
        plain = torch.empty(4 << 20, dtype=torch.uint8)
        self.assertEqual(address_policy(plain.data_ptr())[0], MPOL_DEFAULT)

    def test_the_mapping_base_is_2mib_aligned(self):
        for nbytes in (PAGE_BYTES, 3 * PAGE_BYTES + 100, 5 * MIB + 7):
            with self.subTest(nbytes=nbytes):
                slab = allocate_bound(nbytes, [(0, 0, 1)], nbytes)
                self.assertEqual(slab.data_ptr() % HUGE_BYTES, 0)
                self.assertEqual(slab.numel(), nbytes)

    def test_unaligned_row_runs_tile_the_slab_with_node_changes_on_2mib_boundaries(self):
        # Replaces the page-rounding contract (a page shared by two rows went to the later run): the node change
        # now sits on a 2 MiB boundary of the 2 MiB-aligned base, and the ranges still tile the page-rounded slab.
        rows, row_bytes = 10, 3 * MIB + 100
        runs = [(0, 0, 6), (1, 6, 4)]
        calls = []
        with patch.object(host_numa, "_mbind", side_effect=lambda a, n, node: calls.append((a, n, node))):
            slab = allocate_bound(rows * row_bytes, runs, row_bytes)
        base, nbytes = slab.data_ptr(), rows * row_bytes
        (a0, n0, node0), (a1, n1, node1) = calls
        self.assertEqual((node0, node1), (0, 1))
        self.assertEqual(a0, base)
        self.assertEqual(a0 + n0, a1)
        self.assertEqual((a1 - base) % HUGE_BYTES, 0)
        # The 2 MiB multiple just below or just above the ideal row boundary: here below, by almost 2 MiB, because the
        # tail past the slab (bound to node 1 and counted as its bytes) is compensated at this change.
        self.assertLessEqual(abs((a1 - base) - 6 * row_bytes), HUGE_BYTES)
        self.assertEqual(a1 + n1, base + -(-nbytes // HUGE_BYTES) * HUGE_BYTES)  # out to the 2 MiB end
        self.assertEqual((a1 + n1 - base) % HUGE_BYTES, 0)
        self.assertEqual(slab._numa_bound_bytes, {0: n0, 1: n1})

    def test_a_single_node_placement_binds_the_whole_span_once(self):
        calls = []
        with patch.object(host_numa, "_mbind", side_effect=lambda a, n, node: calls.append((a, n, node))):
            slab = allocate_bound(5 * PAGE_BYTES + 3, [(1, 0, 1)], 5 * PAGE_BYTES + 3)
        self.assertEqual(calls, [(slab.data_ptr(), HUGE_BYTES, 1)])
        self.assertEqual(slab._numa_bound_bytes, {1: HUGE_BYTES})  # the untouched tail is counted as bound

    def test_zero_bytes_is_an_empty_tensor_and_binds_nothing(self):
        with patch.object(host_numa, "_mbind") as bind:
            slab = allocate_bound(0, [], 1)
        bind.assert_not_called()
        self.assertEqual(slab.numel(), 0)
        self.assertEqual(slab._numa_bound_bytes, {})

    def test_pages_are_not_touched_by_allocation(self):
        slab = allocate_bound(8 * PAGE_BYTES, [(0, 0, 8)], PAGE_BYTES)
        self.assertNotIn(0, page_nodes(slab, samples=8))
        split = allocate_bound(16 * MIB, [(0, 0, 3), (1, 3, 1)], 4 * MIB) if HAS_NODE1 else slab
        self.assertFalse(set(page_nodes(split, samples=32)) & {0, 1})

    @unittest.skipUnless(HAS_NODE1, "needs a second NUMA node")
    def test_the_policy_changes_node_exactly_at_each_2mib_boundary(self):
        rows, row_bytes = 40, 3_501_056
        runs = split_rows(rows, ((0, 60), (1, 40)))
        slab = allocate_bound(rows * row_bytes, runs, row_bytes)
        base = slab.data_ptr()
        byte_runs = [(node, first * row_bytes, (first + count) * row_bytes) for node, first, count in runs]
        bindings = plan_bindings(rows * row_bytes, byte_runs)
        self.assertEqual(len(bindings), 2)
        for (node, _, end), (following, _, _) in zip(bindings, bindings[1:]):
            self.assertEqual(end % HUGE_BYTES, 0)
            self.assertEqual(address_policy(base + end - PAGE_BYTES), (MPOL_BIND, frozenset({node})))
            self.assertEqual(address_policy(base + end), (MPOL_BIND, frozenset({following})))
        self.assertEqual(slab._numa_bound_bytes, {node: hi - lo for node, lo, hi in bindings})

    @unittest.skipUnless(HAS_NODE1, "needs a second NUMA node")
    def test_the_tail_past_the_slab_is_bound_to_the_last_node_out_to_2mib(self):
        rows, row_bytes = 40, 3_501_056
        runs = split_rows(rows, ((0, 60), (1, 40)))
        nbytes = rows * row_bytes
        slab = allocate_bound(nbytes, runs, row_bytes)
        base, end = slab.data_ptr(), -(-nbytes // HUGE_BYTES) * HUGE_BYTES
        self.assertNotEqual(nbytes % HUGE_BYTES, 0)
        # The whole last huge page, tail included, is one binding to the last run's node; the slack past it is not.
        for address in (base + end - HUGE_BYTES, base + nbytes - 1, base + end - PAGE_BYTES):
            self.assertEqual(address_policy(address), (MPOL_BIND, frozenset({1})))
        self.assertEqual(address_policy(base + end)[0], MPOL_DEFAULT)
        self.assertEqual(sum(slab._numa_bound_bytes.values()), end)
        self.assertFalse(set(page_nodes(slab[nbytes - PAGE_BYTES :], samples=1)) & {0, 1})  # not touched

    @unittest.skipUnless(HAS_NODE1, "needs a second NUMA node")
    def test_row_runs_land_on_their_nodes(self):
        rows, row_bytes = 20, MIB + 5 * PAGE_BYTES
        runs = split_rows(rows, ((0, 3), (1, 1)))
        slab = allocate_bound(rows * row_bytes, runs, row_bytes)
        slab.fill_(1)
        (_, _, n0), (_, first1, _) = runs
        # Rows more than 2 MiB from the node change are on their own node; the rows around it may be on either.
        self.assertEqual(page_nodes(slab[: n0 * row_bytes - 2 * HUGE_BYTES], samples=16), {0: 16})
        self.assertEqual(page_nodes(slab[first1 * row_bytes + 2 * HUGE_BYTES :], samples=16), {1: 16})

    def test_allocate_host_slab_with_a_placement_keeps_shape_and_bytes(self):
        slab = allocate_host_slab(6, (5, 7), torch.int16, register=False, placement=((0, MIB),))
        self.assertEqual(tuple(slab.shape), (6, 5, 7))
        self.assertEqual(slab.data_ptr() % HUGE_BYTES, 0)
        self.assertEqual(address_policy(slab.data_ptr()), (MPOL_BIND, frozenset({0})))
        self.assertEqual(slab._numa_bound_bytes, {0: HUGE_BYTES})  # 210 B of rows, bound out to the 2 MiB end
        slab.copy_(torch.arange(6 * 35, dtype=torch.int16).view(6, 5, 7))
        self.assertEqual(int(slab[5, 4, 6]), 6 * 35 - 1)

    def test_allocate_host_slab_without_a_placement_keeps_its_page_aligned_path(self):
        with patch.object(host_numa, "allocate_bound") as allocate:
            slab = allocate_host_slab(6, (5, 7), torch.int16, register=False)
        allocate.assert_not_called()
        self.assertEqual(slab.data_ptr() % PAGE_BYTES, 0)
        self.assertFalse(hasattr(slab, "_numa_bound_bytes"))


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

    @unittest.skipUnless(MBIND_PERMITTED, "mbind is not permitted here")
    def test_the_manager_binds_every_slab_and_logs_the_placement(self):
        with envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("0:4"), patch.object(host_numa, "check_capacity"):
            with self.assertLogs(expert_stream.logger, "INFO") as logs:
                manager = ExpertPinnedHostCacheManager.from_model(self.model, budget_bytes=4 * MIB)
        self.assertTrue(manager.caches)
        for cache in manager.caches.values():
            for slab in cache.tensors.values():
                self.assertEqual(address_policy(slab.data_ptr()), (MPOL_BIND, frozenset({0})))
        output = "\n".join(logs.output)
        self.assertIn('"mib": {"0": 4}', output)
        # The bytes actually bound per node, once for the tier: here every slab's page-rounded span on node 0.
        owners = {}
        for cache in manager.caches.values():
            for slab in cache.tensors.values():
                owner = getattr(slab, "_expert_stream_slab_arena", slab)
                owners[id(owner)] = owner
        bound = sum(owner._numa_bound_bytes[0] for owner in owners.values())
        self.assertIn(f'"numa": {{"bound_bytes": {{"0": {bound}}}', output)
        self.assertEqual(output.count('"bound_bytes"'), 1)


if __name__ == "__main__":
    unittest.main()


def _bound(rows, row_bytes, seam, nodes=(0, 1)):
    """A uint8 [rows, row_bytes] slab as allocate_bound leaves it, its bytes bound to nodes[0] below ``seam`` and
    nodes[1] from it; no real mbind."""
    with patch.object(host_numa, "_mbind"):
        owner = allocate_bound(rows * row_bytes, [(nodes[0], 0, 1)], rows * row_bytes)
    slab = owner.view(rows, row_bytes)  # a new tensor object: the attribute goes on it, as allocate_host_slab does
    slab._numa_bindings = [(nodes[0], 0, seam), (nodes[1], seam, -(-rows * row_bytes // HUGE_BYTES) * HUGE_BYTES)]
    return slab


class TestSlotNodes(unittest.TestCase):
    def test_allocate_bound_records_exactly_the_bindings_it_applied(self):
        rows, row_bytes = 10, 3 * MIB + 100
        calls = []
        with patch.object(host_numa, "_mbind", side_effect=lambda a, n, node: calls.append((a, n, node))):
            slab = allocate_bound(rows * row_bytes, [(0, 0, 6), (1, 6, 4)], row_bytes)
        base = slab.data_ptr()
        self.assertEqual(slab._numa_bindings, [(node, a - base, a - base + n) for a, n, node in calls])

    def test_slots_take_the_node_that_holds_their_bytes(self):
        slab = _bound(10, MIB, 4 * MIB)
        self.assertEqual(slot_nodes({"w": slab}, 10), [0] * 4 + [1] * 6)

    def test_a_slot_across_a_seam_belongs_to_no_group(self):
        """Review Focus 5: 3 MiB rows with the node change at 8 MiB, which rounding put inside row 2. Row 2 belongs to
        no node and so to no group; rows 0-1 are node 0's and rows 3-5 node 1's."""
        slab = _bound(6, 3 * MIB, 8 * MIB)
        owners = slot_nodes({"w": slab}, 6)
        self.assertEqual(owners, [0, 0, None, 1, 1, 1])
        self.assertEqual(group_ranges([owners], [0, 1]), [[(0, 2)], [(3, 6)]])

    def test_a_small_slab_wholly_on_one_node_does_not_move_a_slot(self):
        """A layer's sign-vector slabs fit one 2 MiB page, so all of them sit on one node: under 1% of a slot."""
        big = _bound(6, 3 * MIB, 9 * MIB)
        small = _bound(6, 4096, 2 * MIB, nodes=(0, 0))
        self.assertEqual(slot_nodes({"trellis": big, "suh": small}, 6), [0, 0, 0, 1, 1, 1])

    def test_an_arena_slabs_offset_into_its_owner_counts(self):
        with patch.object(host_numa, "_mbind"):
            arena = allocate_bound(12 * MIB, [(0, 0, 1)], 12 * MIB)
        arena._numa_bindings = [(0, 0, 6 * MIB), (1, 6 * MIB, 12 * MIB)]
        first, second = arena[: 6 * MIB].view(3, 2 * MIB), arena[6 * MIB :].view(3, 2 * MIB)
        for slab in (first, second):
            slab._expert_stream_slab_arena = arena
        self.assertEqual(slot_nodes({"a": first}, 3), [0, 0, 0])
        self.assertEqual(slot_nodes({"b": second}, 3), [1, 1, 1])
        self.assertEqual(slot_nodes({"a": first, "b": second}, 3), [None, None, None])

    def test_a_slab_without_a_placement_is_refused(self):
        with self.assertRaisesRegex(ValueError, "without a placement"):
            slot_nodes({"w": torch.zeros(4, 16, dtype=torch.uint8)}, 4)

    def test_group_ranges_refuse_a_split_or_starved_node(self):
        with self.assertRaisesRegex(ValueError, "not one contiguous range"):
            group_ranges([[0, 0, 1, 1, 0]], [0, 1])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            group_ranges([[0, 1, 1, 1]], [0, 1])
        with self.assertRaisesRegex(ValueError, "node 2"):
            group_ranges([[0, 0, 2, 2]], [0, 1])
        self.assertEqual(group_ranges([[0, 0, 1, 1], [0, 0, 0, 1, 1]], [0, 1]), [[(0, 2), (0, 3)], [(2, 4), (3, 5)]])
