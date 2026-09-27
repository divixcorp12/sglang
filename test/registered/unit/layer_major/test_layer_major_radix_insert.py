"""Layer-major ring finalize vs the radix insert that follows it (Task 12 crash, layer-major-8k arm).

cache_unfinished_req inserts a layer-major request up to the SWA branch point when its admission match set one,
else up to page_floor(seq_len), and then rematches that key: the rematch must reach the insert's prefix, so the
request has to keep live window KV up to max(window, page) before that end. A ring finalize that kept only the
last window before seq_len left the key ending on a tombstone and failed
``new_prefix_len=32512, len(new_indices)=16384`` in unified_radix_cache.cache_unfinished_req.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.models.deepseek_v4_layer_major import DSV4_WINDOW, DeepseekV4LayerMajorAdapter
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

PAGE = 256
CHUNK = 1024
KV = 16 * 1024


class _FakeExtendKernel:
    """Routes the Triton alloc_extend_kernel launch through alloc_extend_naive (see test_window_ring.py)."""

    def __getitem__(self, grid):
        def _launch(prefix_lens, seq_lens, last_loc, free_pages, out_indices, bs_pow2, page_size):
            alloc_extend_naive(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, out_indices.device)

        return _launch


@mock.patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel", new=_FakeExtendKernel())
class TestLayerMajorRadixInsert(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=PAGE))
        self.rtp = ReqToTokenPool(size=4, max_context_len=8192, device="cpu", enable_memory_saver=False)
        kv = SWAKVPool(size=KV, size_swa=KV, page_size=PAGE, dtype=torch.bfloat16, head_num=1, head_dim=8,
                       swa_attention_layer_ids=[1], full_attention_layer_ids=[0], device="cpu")
        self.alloc = SWATokenToKVPoolAllocator(size=KV, size_swa=KV, page_size=PAGE, dtype=torch.bfloat16,
                                               device="cpu", kvcache=kv, need_sort=False)
        self.cache = UnifiedRadixCache(params=CacheInitParams(
            req_to_token_pool=self.rtp, token_to_kv_pool_allocator=self.alloc, page_size=PAGE, disable=False,
            sliding_window_size=DSV4_WINDOW, tree_components=(ComponentType.FULL, ComponentType.SWA)))
        # The recipe's --enable-decoder-swa-bounded-replay: admission caps the match a window short of the input.
        override = get_context().override_server_args(enable_decoder_swa_bounded_replay=True)
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(setattr, env_gate, "is_unified_kv_triton", env_gate.is_unified_kv_triton)
        env_gate.is_unified_kv_triton = lambda: False
        self.adapter = DeepseekV4LayerMajorAdapter.__new__(DeepseekV4LayerMajorAdapter)
        self.adapter.runner = SimpleNamespace(token_to_kv_pool_allocator=self.alloc)
        self.adapter.page = PAGE
        self.adapter.chunk = CHUNK
        self.ids = list(range(1000, 9000))
        self.rid = 0

    def _layer_major_prefill(self, n: int) -> Req:
        """Admission match, ring allocation, ring finalize and the post-prefill insert, as the scheduler runs them."""
        req = Req(rid=self.rid, origin_input_text="", origin_input_ids=array("q", self.ids[:n]),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
        self.rid += 1
        self.rtp.alloc([req])
        req.init_next_round_input(self.cache)
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        prefix = len(req.prefix_indices)
        slot = req.kv.req_pool_idx
        self.rtp.write((slot, slice(0, prefix)), req.prefix_indices)
        last_loc = self.rtp.req_to_token[slot, prefix - 1 : prefix] if prefix else torch.tensor([-1])
        full = self.alloc.alloc_extend_swa_tail(
            torch.tensor([prefix]), torch.tensor([prefix]), torch.tensor([n]), torch.tensor([n]),
            last_loc.to(torch.int64), n - prefix, swa_tail_len=CHUNK + PAGE,
        ).to(torch.int64)
        self.rtp.write((slot, slice(prefix, n)), full)
        req.set_extend_range(prefix, n)
        ring = self.alloc.ring_slots(full[-(CHUNK + PAGE) :])
        self.alloc.map_ring_positions(full, torch.arange(prefix, n), ring)
        handle = SimpleNamespace(schedule_batch=SimpleNamespace(reqs=[req], prefix_lens=[prefix]),
                                 spans=[SimpleNamespace(end=n)], extend_full_locs=full, ring=ring, finalized=False)
        self.adapter._finalize_ring(handle)
        req.kv.kv_committed_len = n
        self.cache.cache_unfinished_req(req)
        return req

    def _finish(self, req: Req) -> None:
        self.cache.cache_finished_req(req, owned_kv_len=len(req.origin_input_ids))
        self.rtp.free(req)

    def test_prefix_hit_capped_at_a_tombstone_branch_point(self):
        # The crash's shape at 1/16 scale: a layer-major pass leaves [0, 1792) tombstoned and [1792, 2048) live;
        # a longer prompt's capped match (2148 - 128) reaches full KV 1792 but no live SWA, so its branch point
        # is 1792 and the insert stops on the tombstone.
        self._finish(self._layer_major_prefill(2048))
        req = self._layer_major_prefill(2148)
        self.assertGreaterEqual(req.kv.cache_protected_len, 1792)
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", self.ids[:1792]))))
        self.assertEqual(len(m.device_indices), 1792)
        self._finish(req)
        self.cache.sanity_check()

    def test_unaligned_tail_at_least_a_window_long(self):
        # seq_len % page >= window: the last window starts at page_floor(seq_len), where the insert ends, so a
        # finalize keeping only that window left the whole inserted leaf tombstoned and the tree matched nothing.
        req = self._layer_major_prefill(2048 + 200)
        self.assertEqual(req.kv.cache_protected_len, 2048)
        self._finish(req)
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
