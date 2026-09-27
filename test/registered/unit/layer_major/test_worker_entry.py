from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layer_major import worker_entry
from sglang.srt.layer_major.heartbeat import pass_progress


class TestWorkerEntry(unittest.TestCase):
    def test_runtime_off_when_threshold_is_zero(self):
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=0):
            self.assertIsNone(worker_entry.layer_major_runtime(SimpleNamespace(model=object())))

    def test_runtime_refuses_a_model_without_an_adapter(self):
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=64):
            with self.assertRaisesRegex(ValueError, "no layer-major adapter"):
                worker_entry.layer_major_runtime(SimpleNamespace(model=object()))

    def test_seam_propagates_pass_failure_after_release(self):
        runtime = SimpleNamespace(adapter="a", store="s", residency="r")
        runner = SimpleNamespace(attn_backend="backend", forward_pass_id=0)
        with mock.patch.object(worker_entry, "run_pass", side_effect=RuntimeError("pass failed")):
            with self.assertRaisesRegex(RuntimeError, "pass failed"):
                worker_entry.run_layer_major_prefill(runtime, runner, "sb", SimpleNamespace())

    def test_run_pass_executes_in_the_calling_process(self):
        # The scheduler watchdog reads the module-global heartbeat in its own process; run_pass
        # must therefore advance it right there, not in a worker process the watchdog cannot see.
        before = pass_progress()
        runtime = SimpleNamespace(adapter="a", store="s", residency="r")
        runner = SimpleNamespace(attn_backend="backend", forward_pass_id=0)

        def fake_run_pass(adapter, residency, forward_batch, schedule_batch, store):
            from sglang.srt.layer_major.heartbeat import current_heartbeat

            current_heartbeat().tick()
            return "logits"

        with mock.patch.object(worker_entry, "run_pass", side_effect=fake_run_pass):
            out = worker_entry.run_layer_major_prefill(
                runtime, runner, SimpleNamespace(layer_major_ring_tokens=None), SimpleNamespace()
            )
        self.assertEqual(out.logits_output, "logits")
        self.assertGreater(pass_progress(), before)


if __name__ == "__main__":
    unittest.main()
