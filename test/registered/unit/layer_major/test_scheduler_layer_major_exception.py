"""A layer-major pass's forward exception must not reach run_scheduler_process's
top-level handler (which SIGQUITs the server): Scheduler.run_batch catches it for
exactly a batch whose layer_major_ring_tokens is not None, finishes the request
with an error, releases its KV the same way retraction does (no further forward
needed), and returns normally. A normal batch's exception is not caught here.
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers import scheduler as scheduler_mod
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeSendToTokenizer:
    def __init__(self):
        self.sent = []

    def send_output(self, msg, req):
        self.sent.append((msg, req))


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.finished_reason = None
        self.weight_version_events = []
        self.output_ids = []

    def finished(self):
        return self.finished_reason is not None


class _FakeBatch:
    def __init__(self, layer_major_ring_tokens, reqs):
        self.layer_major_ring_tokens = layer_major_ring_tokens
        self.reqs = reqs
        self.forward_mode = SimpleNamespace(is_extend=lambda: True)


def _make_scheduler():
    sched = Scheduler.__new__(Scheduler)
    sched.tree_cache = object()
    sched.ipc_channels = SimpleNamespace(send_to_tokenizer=_FakeSendToTokenizer())
    # run_batch/process_batch_result are @scheduler_stage_method-decorated;
    # None bypasses the (unrelated) stage-timing recorder.
    sched.scheduler_stage_metrics = None
    return sched


class TestRunBatchLayerMajorExceptionHandling(unittest.TestCase):
    def setUp(self):
        # _make_abort_req reads get_serving().weight_version.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_catches_exception_finishes_request_and_releases_kv(self):
        sched = _make_scheduler()
        req = _FakeReq("r0")
        batch = _FakeBatch(layer_major_ring_tokens=4352, reqs=[req])
        sched._run_batch_impl = mock.Mock(side_effect=RuntimeError("boom"))

        with mock.patch.object(scheduler_mod, "release_kv_cache") as mock_release:
            result = sched.run_batch(batch)

        self.assertIsInstance(result, GenerationBatchResult)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(batch.reqs, [])
        mock_release.assert_called_once_with(req, sched.tree_cache, is_insert=False)
        self.assertEqual(len(sched.ipc_channels.send_to_tokenizer.sent), 1)

    def test_does_not_catch_exception_for_a_normal_batch(self):
        sched = _make_scheduler()
        batch = _FakeBatch(layer_major_ring_tokens=None, reqs=[])
        sched._run_batch_impl = mock.Mock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            sched.run_batch(batch)

    def test_process_batch_result_is_a_noop_for_a_failed_layer_major_batch(self):
        sched = _make_scheduler()
        batch = _FakeBatch(layer_major_ring_tokens=4352, reqs=[])
        # A real GenerationBatchResult() placeholder should never be inspected:
        # process_batch_result must return before touching any of its fields.
        Scheduler.process_batch_result(sched, batch, GenerationBatchResult())


if __name__ == "__main__":
    unittest.main()
