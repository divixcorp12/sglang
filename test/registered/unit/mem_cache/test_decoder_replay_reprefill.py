"""Decoder SWA bounded replay: a prefix hit must leave the trailing sliding window to be re-prefilled.

Under --enable-decoder-swa-bounded-replay a prefill writes late-layer SWA KV only for the last
min(window, extend) tokens of each extend. A match boundary inside an earlier extend therefore keeps
late-layer slots no forward wrote, and a short suffix's first decode steps read them (poisoned-pool
probe: first-decode logprob off by 0.1-0.5 against 0.01-0.1 noise). The device slots and their
HiCache host copy are equally stale, so the cap applies with or without a host SWA pool.
"""

import unittest
from types import SimpleNamespace

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _paged_swa_cache(*, has_swa_host_pool: bool) -> UnifiedRadixCache:
    component = SWAComponent.__new__(SWAComponent)
    component.sliding_window_size = 128
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.components = {ComponentType.SWA: component}
    cache.tree_core = SimpleNamespace(has_swa_host_pool=has_swa_host_pool)
    return cache


class TestDecoderReplayReprefill(unittest.TestCase):
    def setUp(self):
        original = env_gate.is_unified_kv_triton
        env_gate.is_unified_kv_triton = lambda: False
        self.addCleanup(setattr, env_gate, "is_unified_kv_triton", original)

    def _override(self, enabled: bool) -> None:
        override = get_context().override_server_args(enable_decoder_swa_bounded_replay=enabled)
        override.install()
        self.addCleanup(override.restore)

    def test_bounded_replay_reprefills_the_window_with_and_without_a_host_pool(self):
        self._override(True)
        for has_swa_host_pool in (False, True):
            with self.subTest(has_swa_host_pool=has_swa_host_pool):
                cache = _paged_swa_cache(has_swa_host_pool=has_swa_host_pool)
                self.assertEqual(cache.swa_reprefill_tail_tokens(), 128)

    def test_paged_swa_without_bounded_replay_reuses_the_whole_prefix(self):
        self._override(False)
        cache = _paged_swa_cache(has_swa_host_pool=False)
        self.assertEqual(cache.swa_reprefill_tail_tokens(), 0)


if __name__ == "__main__":
    unittest.main()
