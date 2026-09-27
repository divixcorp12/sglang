from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt import runtime_context as rc
from sglang.srt.layer_major import worker_entry
from sglang.srt.layer_major.heartbeat import pass_progress
from sglang.srt.server_args import ServerArgs


def _runner(**overrides):
    defaults = dict(
        attn_backend="backend",
        forward_pass_id=0,
        eplb_manager=None,
        server_args=SimpleNamespace(chunked_prefill_size=4096),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestWorkerEntry(unittest.TestCase):
    def test_runtime_off_when_threshold_is_zero(self):
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=0):
            self.assertIsNone(worker_entry.layer_major_runtime(SimpleNamespace(model=object())))

    def test_runtime_refuses_a_model_without_an_adapter(self):
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=64):
            with self.assertRaisesRegex(ValueError, "no layer-major adapter"):
                worker_entry.layer_major_runtime(SimpleNamespace(model=object()))

    def test_runtime_refuses_when_expert_prediction_enabled(self):
        class _Model:
            @staticmethod
            def make_layer_major_adapter(model, model_runner):
                raise AssertionError("must not build the adapter once a refusal applies")

        runner = SimpleNamespace(model=_Model(), expert_prediction_runtime=object())
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=64):
            with self.assertRaisesRegex(ValueError, "expert-prediction"):
                worker_entry.layer_major_runtime(runner)

    def test_runtime_refuses_when_topk_capture_enabled(self):
        # Toggle the real launch flag the capturer factories consult
        # (RoutedExpertsCapturer.create / create_indexer_capturer), not the helper: the capturers
        # themselves (get_resources().experts_capturer/.indexer_capturer) are only installed later in
        # startup (ModelRunner._init_post_memory_pool_components, after TpModelWorker.__init__ already
        # ran), so reading them here would never catch a launch that enables the feature.
        class _Model:
            @staticmethod
            def make_layer_major_adapter(model, model_runner):
                raise AssertionError("must not build the adapter once a refusal applies")

        runner = SimpleNamespace(model=_Model(), expert_prediction_runtime=None)
        rc.reset_context()
        try:
            rc.publish(ServerArgs(model_path="dummy"), role="test")
            with rc.get_exec().features.override(enable_return_routed_experts=True):
                with mock.patch.object(
                    worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=64
                ):
                    with self.assertRaisesRegex(ValueError, "top-k capture"):
                        worker_entry.layer_major_runtime(runner)
        finally:
            rc.reset_context()

    def test_draft_worker_gets_no_layer_major_runtime(self):
        # A draft worker must short-circuit before layer_major_runtime ever inspects the model: this
        # model has no adapter, so reaching layer_major_runtime would raise instead of returning None.
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=64):
            result = worker_entry.layer_major_runtime_for_worker(
                SimpleNamespace(model=object()), is_draft_worker=True
            )
        self.assertIsNone(result)

    def test_seam_propagates_pass_failure_after_release(self):
        # Review Focus #3 (Task 11 half): the pass fails inside run_layer, and release_pass must still
        # run with failed=True before the exception reaches the caller.
        class _FailingAdapter:
            def __init__(self):
                self.release_calls = []

            def begin_pass(self, forward_batch, schedule_batch, store):
                return "handle"

            def layer_ids(self, handle):
                return range(1)

            def num_chunks(self, handle):
                return 1

            def run_layer(self, handle, layer_id, chunk, store):
                raise RuntimeError("pass failed")

            def finish_pass(self, handle, store):
                raise AssertionError("must not reach finish_pass after run_layer raised")

            def release_pass(self, handle, store, *, failed):
                self.release_calls.append(failed)

        adapter = _FailingAdapter()
        runtime = SimpleNamespace(adapter=adapter, store="s", residency=worker_entry.NullResidency())
        runner = _runner()
        with self.assertRaisesRegex(RuntimeError, "pass failed"):
            worker_entry.run_layer_major_prefill(runtime, runner, "sb", SimpleNamespace())
        self.assertEqual(adapter.release_calls, [True])

    def test_run_pass_executes_in_the_calling_process(self):
        # The scheduler watchdog reads the module-global heartbeat in its own process; run_pass
        # must therefore advance it right there, not in a worker process the watchdog cannot see.
        before = pass_progress()
        runtime = SimpleNamespace(adapter="a", store="s", residency="r")
        runner = _runner()
        schedule_batch = SimpleNamespace(layer_major_ring_tokens=4096)

        def fake_run_pass(adapter, residency, forward_batch, schedule_batch, store):
            from sglang.srt.layer_major.heartbeat import current_heartbeat

            current_heartbeat().tick()
            return "logits"

        with mock.patch.object(worker_entry, "run_pass", side_effect=fake_run_pass):
            out = worker_entry.run_layer_major_prefill(runtime, runner, schedule_batch, SimpleNamespace())
        self.assertEqual(out.logits_output, "logits")
        self.assertGreater(pass_progress(), before)

    def test_forward_pass_id_advances_once_per_pass(self):
        runtime = SimpleNamespace(adapter="a", store="s", residency="r")
        runner = _runner(forward_pass_id=5)
        schedule_batch = SimpleNamespace(layer_major_ring_tokens=100)
        with mock.patch.object(worker_entry, "run_pass", return_value="logits"):
            worker_entry.run_layer_major_prefill(runtime, runner, schedule_batch, SimpleNamespace())
        self.assertEqual(runner.forward_pass_id, 6)

    def test_eplb_manager_on_forward_pass_end_is_called(self):
        eplb_manager = mock.Mock()
        runtime = SimpleNamespace(adapter="a", store="s", residency="r")
        runner = _runner(eplb_manager=eplb_manager)
        schedule_batch = SimpleNamespace(layer_major_ring_tokens=100)
        with mock.patch.object(worker_entry, "run_pass", return_value="logits"):
            worker_entry.run_layer_major_prefill(runtime, runner, schedule_batch, SimpleNamespace())
        eplb_manager.on_forward_pass_end.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
