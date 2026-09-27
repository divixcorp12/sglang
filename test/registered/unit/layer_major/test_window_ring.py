"""Generic hybrid-SWA window-KV ring: admission budget, allocation, mapping,
and finalize/release (spec Sec 5.1, 6.4).

A whole-suffix layer-major extend reserves full KV for every token but window
KV for a ring of ``chunk + page`` slots only. After the pass, only the final
window's ring slots stay mapped; every other ring slot returns to the pool.
This code is generic to any hybrid-SWA model -- the DSV4 adapter only chooses
the ring size.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.allocation import _evict_for_ring
from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4
CHUNK = 16
RING = CHUNK + PAGE  # 5 pages


class _FakeExtendKernel:
    """Drop-in for the Triton alloc_extend_kernel[grid](...) call.

    PagedTokenToKVPoolAllocator.alloc_extend dispatches a Triton kernel
    unconditionally (no CPU/eager fallback), which needs an active GPU
    driver even for CPU tensors. These tests verify slot/page bookkeeping,
    not the kernel, so route the same call through the pure-torch
    alloc_extend_naive instead of spending the shared GPU on a unit test.
    """

    def __getitem__(self, grid):
        def _launch(prefix_lens, seq_lens, last_loc, free_pages, out_indices, bs_pow2, page_size):
            alloc_extend_naive(
                prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, out_indices.device
            )

        return _launch


def _allocator(size=256, size_swa=64, device="cpu"):
    kv = SWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device=device,
    )
    return SWATokenToKVPoolAllocator(
        size=size,
        size_swa=size_swa,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv,
        need_sort=False,
    )


def _alloc_whole(a, n, device="cpu"):
    # prefix_lens/seq_lens/last_loc mirror the batch's own device tensors (CUDA in production,
    # allocation.py:alloc_for_extend); prefix_lens_cpu/seq_lens_cpu are always host tensors
    # (get_num_new_pages asserts on this) regardless of the allocator's device.
    cpu = lambda x: torch.tensor(x)
    dev = lambda x: torch.tensor(x, device=device)
    full = a.alloc_extend_swa_tail(
        dev([0]), cpu([0]), dev([n]), cpu([n]), dev([-1]), n, swa_tail_len=RING
    )
    assert full is not None
    return full


@mock.patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel", new=_FakeExtendKernel())
class TestWindowRing(unittest.TestCase):
    def test_ring_admission_restores_available_size(self):
        a = _allocator()
        swa_before = a.swa_available_size()
        full = _alloc_whole(a, 64)
        ring = a.ring_slots(full[-RING:])
        self.assertEqual(ring.numel(), RING)
        self.assertEqual(swa_before - a.swa_available_size(), RING)
        a.finalize_ring(full, extend_start=0, keep_from=64 - 8, ring=ring)
        a.free(full)
        self.assertEqual(a.swa_available_size(), swa_before)

    def test_every_chunk_sees_its_predecessor_page_in_distinct_slots(self):
        a = _allocator()
        n = 64
        full = _alloc_whole(a, n)
        ring = a.ring_slots(full[-RING:])
        for start in range(0, n, CHUNK):
            pos = torch.arange(start, min(n, start + CHUNK))
            a.map_ring_positions(full[pos], pos, ring)
            window = torch.arange(max(0, start - PAGE), min(n, start + CHUNK))
            slots = a.full_to_swa_index_mapping[full[window]]
            self.assertEqual(slots.unique().numel(), window.numel())

    def test_release_after_finalize_returns_every_slot(self):
        a = _allocator()
        swa_before, full_before = a.swa_available_size(), a.full_available_size()
        n = 50  # final chunk of 2 tokens; PAGE=4 does not divide n
        full = _alloc_whole(a, n)
        ring = a.ring_slots(full[-RING:])
        for start in range(0, n, CHUNK):
            pos = torch.arange(start, min(n, start + CHUNK))
            a.map_ring_positions(full[pos], pos, ring)
        keep_from = (n - 8) // PAGE * PAGE
        a.finalize_ring(full, extend_start=0, keep_from=keep_from, ring=ring)
        kept = a.full_to_swa_index_mapping[full[keep_from:]]
        self.assertTrue(bool((kept > 0).all()))
        self.assertTrue(bool((a.full_to_swa_index_mapping[full[:keep_from]] == 0).all()))
        a.free(full)
        self.assertEqual(
            (a.swa_available_size(), a.full_available_size()),
            (swa_before, full_before),
        )

    def test_finalize_keeps_the_partially_kept_page_off_the_free_list(self):
        """Pins Critical-1: releasing "unused" ring slots page-at-a-time must
        not free a page that still backs a kept position. With n=50 and
        PAGE=4, the final kept page (positions 48-49) holds only 2 of its 4
        slots live; the other 2 must stay reserved with the request, not
        return to the pool, and the whole ring must come back exactly once
        when the request frees.
        """
        a = _allocator()
        swa_before, full_before = a.swa_available_size(), a.full_available_size()
        n = 50
        full = _alloc_whole(a, n)
        ring = a.ring_slots(full[-RING:])
        for start in range(0, n, CHUNK):
            pos = torch.arange(start, min(n, start + CHUNK))
            a.map_ring_positions(full[pos], pos, ring)
        keep_from = (n - 8) // PAGE * PAGE
        a.finalize_ring(full, extend_start=0, keep_from=keep_from, ring=ring)

        kept = a.full_to_swa_index_mapping[full[keep_from:]]
        kept_pages = (kept[kept > 0] // PAGE).unique().tolist()
        free_pages = a.swa_attn_allocator.get_all_free_pages().tolist()
        for p in kept_pages:
            self.assertNotIn(
                p, free_pages, f"page {p} still backs a kept position but was freed"
            )

        a.free(full)
        self.assertEqual(a.swa_available_size(), swa_before)
        self.assertEqual(a.full_available_size(), full_before)
        # Every physical SWA page came back exactly once: no duplicates, none
        # missing (available_size above already pins the count; this pins
        # that it is not e.g. one page short and one page double-counted).
        all_free = a.swa_attn_allocator.get_all_free_pages()
        self.assertEqual(all_free.numel(), all_free.unique().numel())


class TestRingBudget(unittest.TestCase):
    def _budget(self, remaining_total, remaining_swa):
        cls = type(
            "B",
            (SWAPrefillBudget,),
            {
                "remaining_total": property(lambda self: remaining_total),
                "remaining_swa": property(lambda self: remaining_swa),
            },
        )
        b = object.__new__(cls)
        b.page_size = PAGE
        b.req_ring = False
        return b

    def test_ring_budget(self):
        self.assertTrue(
            self._budget(1000, RING + PAGE + 1).check_prefill_ring(
                total_tokens=500, max_new_tokens=8, ring_tokens=RING
            )
        )
        self.assertFalse(
            self._budget(1000, RING + PAGE).check_prefill_ring(
                total_tokens=500, max_new_tokens=8, ring_tokens=RING
            )
        )
        self.assertFalse(
            self._budget(400, 1000).check_prefill_ring(
                total_tokens=500, max_new_tokens=8, ring_tokens=RING
            )
        )


class TestRingEvictionSizing(unittest.TestCase):
    """Pins Important-2: the ring branch must size its SWA evict target by
    the ring, not by the whole (potentially huge) layer-major extend."""

    class _FakeAllocator:
        def full_available_size(self):
            return 0

        def swa_available_size(self):
            return 0

    class _FakeTreeCache:
        def __init__(self):
            self.calls = []

        def is_chunk_cache(self):
            return False

        def evict_for_alloc(self, params):
            self.calls.append(params)

    def test_ring_branch_sizes_swa_by_ring_not_by_full_extend(self):
        tree_cache = self._FakeTreeCache()
        full_target = 250_000  # a long layer-major extend
        swa_target = RING + PAGE  # what the ring branch should ask for

        _evict_for_ring(
            tree_cache,
            self._FakeAllocator(),
            full_target=full_target,
            swa_target=swa_target,
        )

        self.assertEqual(len(tree_cache.calls), 1)
        params = tree_cache.calls[0]
        self.assertIsInstance(params, EvictParams)
        self.assertEqual(params.num_tokens, full_target)
        self.assertEqual(params.swa_num_tokens, swa_target)
        self.assertLess(params.swa_num_tokens, full_target)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestWindowRingCudaFreePath(unittest.TestCase):
    """TestWindowRing above mocks alloc_extend_kernel with the pure-torch alloc_extend_naive and runs
    on a CPU allocator, so free_index.is_cuda is always False there and _free_swa_pages always takes
    _free_swa_pages_none_cuda. This class runs on a real CUDA device with the real Triton
    alloc_extend_kernel, so free_index.is_cuda is True and _free_swa_pages dispatches to
    _free_swa_pages_cuda (get_and_clear_swa_pages), the path Task 12's GPU equivalence run actually
    exercises."""

    def test_finalize_keeps_the_partially_kept_page_off_the_free_list_on_cuda(self):
        # Same scenario as TestWindowRing.test_finalize_keeps_the_partially_kept_page_off_the_free_list
        # (Critical-1: a page that still backs a kept position must not be freed), but through the real
        # CUDA free kernel instead of the CPU fallback.
        a = _allocator(device="cuda")
        swa_before, full_before = a.swa_available_size(), a.full_available_size()
        n = 50  # PAGE=4 does not divide n; final kept page is only partially live.
        full = _alloc_whole(a, n, device="cuda")
        ring = a.ring_slots(full[-RING:])
        for start in range(0, n, CHUNK):
            pos = torch.arange(start, min(n, start + CHUNK), device="cuda")
            a.map_ring_positions(full[pos], pos, ring)
        keep_from = (n - 8) // PAGE * PAGE
        a.finalize_ring(full, extend_start=0, keep_from=keep_from, ring=ring)

        kept = a.full_to_swa_index_mapping[full[keep_from:]]
        kept_pages = (kept[kept > 0] // PAGE).unique().tolist()
        free_pages = a.swa_attn_allocator.get_all_free_pages().tolist()
        for p in kept_pages:
            self.assertNotIn(p, free_pages, f"page {p} still backs a kept position but was freed")

        # free_segment (not free()'s finalize_ring path) is what actually reaches _free_swa_pages_cuda: check (a).
        with mock.patch.object(a, "_free_swa_pages_cuda", wraps=a._free_swa_pages_cuda) as m:
            a.free_segment(full, start_pos=0)
            m.assert_called()
        self.assertEqual(a.swa_available_size(), swa_before)
        self.assertEqual(a.full_available_size(), full_before)
        all_free = a.swa_attn_allocator.get_all_free_pages()
        self.assertEqual(all_free.numel(), all_free.unique().numel())


if __name__ == "__main__":
    unittest.main()
