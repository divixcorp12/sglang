"""Layer-major ring finalize vs the radix insert that follows it (Task 12 crash, layer-major-8k arm).

See radix-debug-report.md: a finalize that kept only the last window before seq_len left the insert key
ending on a tombstone (``new_prefix_len=32512, len(new_indices)=16384``)."""

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
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
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
        self.pool_baseline = self._pool_sizes()

    def _run_pass(self, n: int) -> Req:
        """Admission match, ring allocation and ring finalize, as the scheduler runs them mid-pass."""
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
        return req

    def _layer_major_prefill(self, n: int) -> Req:
        """A pass that finishes normally: finalize, then the post-prefill insert."""
        req = self._run_pass(n)
        req.kv.kv_committed_len = n
        self.cache.cache_unfinished_req(req)
        return req

    def _finish(self, req: Req) -> None:
        self.cache.cache_finished_req(req, owned_kv_len=len(req.origin_input_ids))
        self.rtp.free(req)

    def _abort(self, req: Req, *, owned_kv_len: int) -> None:
        # release_pass(failed=True): only _finalize_ring ran, so cache_protected_len is still the pre-pass prefix.
        self.cache.cache_finished_req(req, is_insert=False, owned_kv_len=owned_kv_len)
        self.rtp.free(req)

    def _pool_sizes(self) -> tuple[int, int]:
        return self.alloc.full_available_size(), self.alloc.swa_available_size()

    def _assert_pool_restored_after_evict(self) -> None:
        # evict() reclaims every unlocked node, seeded prefixes included, so the only stable baseline to
        # compare against is the pristine pool from setUp, not a snapshot taken after _seed().
        self.cache.evict(EvictParams(num_tokens=KV, swa_num_tokens=KV))
        self.assertEqual(self._pool_sizes(), self.pool_baseline)

    def _seed(self, n: int, swa_live_from: int) -> None:
        """Cache ids[:n] with window KV live only from swa_live_from, as a request whose window slid there leaves it."""
        full = self.alloc.alloc_extend(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([n]), torch.tensor([n]), torch.tensor([-1]), n
        ).to(torch.int64)
        self.alloc.free_swa_segment(full[:swa_live_from], start_pos=0)
        self.cache.insert(InsertParams(key=RadixKey(array("q", self.ids[:n])), value=full,
                                       swa_evicted_seqlen=swa_live_from))

    def _assert_matches(self, n: int) -> None:
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", self.ids[:n]))))
        self.assertEqual(len(m.device_indices), n)

    def test_prefix_hit_capped_at_a_tombstone_branch_point(self):
        # The crash's shape at 1/16 scale: [0, 1792) tombstoned, [1792, 2048) live. A 2148-token prompt's match,
        # capped at 2148 - 128, reaches full KV 1792 but no live window, so admission sets branch point 1792.
        # This is also the radix-prefix-hit case for the clamp (Radix M4): prefix_len=1792 here, and the raw
        # no-branch floor (1536) sits below it, so max(prefix_len, floor) must bind to keep the assertion below
        # true. A layer-major extend must be at least a ring long (see min_ring_len), and a ring is always many
        # pages wide, so the floor sits within ~2 pages of seq_len -- always past any prefix short of a
        # tombstone branch point like this one.
        self._seed(2048, swa_live_from=1792)
        req = self._layer_major_prefill(2148)
        self.assertEqual(req.kv.cache_protected_len, 1792)
        self._assert_matches(1792)
        self._finish(req)
        self.cache.sanity_check()
        self._assert_pool_restored_after_evict()

    def test_branch_point_older_than_the_ring_inserts_to_the_end(self):
        # Branch point 2048 lies more than a ring (5 pages) below seq_len 4000, so its window is gone from the
        # ring: finalize drops the branch and the insert runs to page_floor(seq_len).
        self._seed(2048, swa_live_from=2048)
        req = self._layer_major_prefill(4000)
        self.assertEqual(req.kv.cache_protected_len, 3840)
        self._assert_matches(3840)
        self._finish(req)
        self.cache.sanity_check()
        self._assert_pool_restored_after_evict()

    def test_unaligned_tail_at_least_a_window_long(self):
        # seq_len % page >= window: the last window starts at page_floor(seq_len), where the insert ends, so a
        # finalize keeping only that window left the whole inserted leaf tombstoned and the tree matched nothing.
        req = self._layer_major_prefill(2048 + 200)
        self.assertEqual(req.kv.cache_protected_len, 2048)
        self._assert_matches(2048)
        self._finish(req)
        self.cache.sanity_check()
        self._assert_pool_restored_after_evict()

    def test_abort_after_finalize_restores_the_pool(self):
        # release_pass(failed=True) runs _finalize_ring only; cache_finished_req(is_insert=False) then frees the
        # request's KV without inserting it: both the SWA and full pools must return to their starting sizes.
        self._seed(2048, swa_live_from=1792)
        req = self._run_pass(2148)
        self._abort(req, owned_kv_len=2148)
        self.cache.sanity_check()
        self._assert_pool_restored_after_evict()


if __name__ == "__main__":
    unittest.main()
