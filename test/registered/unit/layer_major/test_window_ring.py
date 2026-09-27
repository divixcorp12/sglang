"""Generic hybrid-SWA window-KV ring: admission budget, allocation, mapping,
and finalize/release (spec Sec 5.1, 6.4).

A whole-suffix layer-major extend reserves full KV for every token but window
KV for a ring of ``chunk + page`` slots only. After the pass, only the final
window's ring slots stay mapped; every other ring slot returns to the pool.
This code is generic to any hybrid-SWA model -- the DSV4 adapter only chooses
the ring size.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

PAGE = 4
CHUNK = 16
RING = CHUNK + PAGE  # 5 pages

# PagedTokenToKVPoolAllocator.alloc_extend dispatches a Triton kernel
# unconditionally, which needs an active GPU driver even for CPU tensors
# (page_size > 1 has no eager fallback here). Ring construction and
# allocation therefore cannot run without CUDA; TestRingBudget is pure
# Python and needs neither the allocator nor a GPU.
_HAS_CUDA = torch.cuda.is_available()


def _allocator(size=256, size_swa=64):
    kv = SWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )
    return SWATokenToKVPoolAllocator(
        size=size,
        size_swa=size_swa,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=False,
    )


def _alloc_whole(a, n):
    t = torch.tensor
    full = a.alloc_extend_swa_tail(
        t([0]), t([0]), t([n]), t([n]), t([-1]), n, swa_tail_len=RING
    )
    assert full is not None
    return full


@unittest.skipUnless(
    _HAS_CUDA,
    "alloc_extend_swa_tail's full-side alloc_extend dispatches a Triton "
    "kernel that needs an active GPU driver, even on CPU tensors",
)
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
        n = 50  # final chunk of 2 tokens
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


if __name__ == "__main__":
    unittest.main()
