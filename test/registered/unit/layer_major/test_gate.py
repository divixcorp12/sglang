"""Unit tests for the layer-major prefill admission gate and launch refusals."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest import mock

from sglang.srt.layer_major.gate import (
    LayerMajorGate,
    gate_from_env,
    launch_refusal,
    scheduler_layer_major_refusal,
)


class TestGate(unittest.TestCase):
    def test_threshold_and_capacity(self):
        g = LayerMajorGate(min_tokens=32768, max_tokens=262144)
        self.assertFalse(g.admits(extend_len=32767, wants_prompt_logprobs=False, wants_hidden_states=False))
        self.assertTrue(g.admits(extend_len=32768, wants_prompt_logprobs=False, wants_hidden_states=False))
        self.assertFalse(g.admits(extend_len=262145, wants_prompt_logprobs=False, wants_hidden_states=False))

    def test_refuses_prompt_logprobs_and_hidden_states(self):
        g = LayerMajorGate(min_tokens=10, max_tokens=100)
        self.assertFalse(g.admits(extend_len=50, wants_prompt_logprobs=True, wants_hidden_states=False))
        self.assertFalse(g.admits(extend_len=50, wants_prompt_logprobs=False, wants_hidden_states=True))

    def test_gate_from_env_off_by_default(self):
        with mock.patch.dict(os.environ, {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0"}):
            self.assertIsNone(gate_from_env(max_tokens=1000))
        with mock.patch.dict(os.environ, {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "64"}):
            self.assertEqual(gate_from_env(max_tokens=1000), LayerMajorGate(min_tokens=64, max_tokens=1000))

    def test_launch_refusals(self):
        ok = dict(max_running_requests=1, speculative_algorithm=None, enable_dp_attention=False, attn_cp_size=1,
                  enable_two_batch_overlap=False)
        self.assertIsNone(launch_refusal(**ok))
        self.assertIn("max-running-requests", launch_refusal(**{**ok, "max_running_requests": 2}))
        self.assertIn("speculative", launch_refusal(**{**ok, "speculative_algorithm": "EAGLE"}))
        self.assertIn("DP attention", launch_refusal(**{**ok, "enable_dp_attention": True}))
        self.assertIn("context parallelism", launch_refusal(**{**ok, "attn_cp_size": 2}))
        self.assertIn("two-batch overlap", launch_refusal(**{**ok, "enable_two_batch_overlap": True}))

    def test_scheduler_refusals(self):
        ok = dict(
            raw_chunked_prefill_size=4096,
            effective_chunked_prefill_size=4096,
            is_hybrid_swa_allocator=True,
            is_swa_req_ring=False,
            adapter_missing_model_name=None,
        )
        self.assertIsNone(scheduler_layer_major_refusal(**ok))
        self.assertIn(
            "chunked prefill to be enabled",
            scheduler_layer_major_refusal(**{**ok, "effective_chunked_prefill_size": None}),
        )
        self.assertIn(
            "differs from the launch value",
            scheduler_layer_major_refusal(**{**ok, "effective_chunked_prefill_size": 2048}),
        )
        self.assertIn(
            "hybrid-SWA",
            scheduler_layer_major_refusal(**{**ok, "is_hybrid_swa_allocator": False}),
        )
        self.assertIn(
            "per-request SWA ring",
            scheduler_layer_major_refusal(**{**ok, "is_swa_req_ring": True}),
        )
        self.assertIn(
            "MyModel has no layer-major adapter",
            scheduler_layer_major_refusal(**{**ok, "adapter_missing_model_name": "MyModel"}),
        )


if __name__ == "__main__":
    unittest.main()
