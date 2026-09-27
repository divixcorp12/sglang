from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.layer_major.driver import run_pass
from sglang.srt.layer_major.heartbeat import pass_progress


class _Adapter:
    def __init__(self, layers=range(0, 3), chunks=2, fail_at=None):
        self.calls, self._layers, self._chunks, self._fail_at = [], layers, chunks, fail_at

    def field_specs(self):
        return []

    def begin_pass(self, forward_batch, schedule_batch, store):
        self.calls.append("begin_pass")
        return "handle"

    def layer_ids(self, handle):
        return self._layers

    def num_chunks(self, handle):
        return self._chunks

    def run_layer(self, handle, layer_id, chunk, store):
        if (layer_id, chunk) == self._fail_at:
            raise RuntimeError("boom")
        self.calls.append(("layer", layer_id, chunk))

    def finish_pass(self, handle, store):
        self.calls.append("finish_pass")
        return "logits"

    def release_pass(self, handle, store, *, failed):
        self.calls.append(("release", failed))


class _Residency:
    def __init__(self, calls, *, fail_restore=False, fail_end=False):
        self.calls = calls
        self._fail_restore = fail_restore
        self._fail_end = fail_end
        self.restore_calls = 0

    def begin(self, layer_ids):
        self.calls.append(("res.begin", tuple(layer_ids)))

    def make_resident(self, layer_id):
        self.calls.append(("res.make", layer_id))

    def restore(self):
        self.restore_calls += 1
        self.calls.append("res.restore")
        if self._fail_restore:
            raise RuntimeError("restore boom")

    def end(self):
        self.calls.append("res.end")
        if self._fail_end:
            raise RuntimeError("end boom")


class TestDriver(unittest.TestCase):
    def test_layer_outer_chunk_inner_order(self):
        a = _Adapter()
        out = run_pass(a, _Residency(a.calls), "fb", "sb", store=None)
        self.assertEqual(out, "logits")
        self.assertEqual(
            a.calls,
            ["begin_pass", ("res.begin", (0, 1, 2)),
             ("res.make", 0), ("layer", 0, 0), ("layer", 0, 1),
             ("res.make", 1), ("layer", 1, 0), ("layer", 1, 1),
             ("res.make", 2), ("layer", 2, 0), ("layer", 2, 1),
             "res.restore", "finish_pass", "res.end", ("release", False)],
        )

    def test_heartbeat_ticks_once_per_layer_chunk(self):
        before = pass_progress()
        a = _Adapter()
        run_pass(a, _Residency(a.calls), "fb", "sb", store=None)
        self.assertEqual(pass_progress() - before, 3 * 2)

    def test_exception_mid_pass_still_ends_residency(self):
        a = _Adapter(fail_at=(1, 1))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_pass(a, _Residency(a.calls), "fb", "sb", store=None)
        self.assertEqual(a.calls[-3:], ["res.restore", "res.end", ("release", True)])
        self.assertNotIn("finish_pass", a.calls)

    def test_restore_failure_still_ends_and_releases_and_is_not_retried(self):
        a = _Adapter()
        residency = _Residency(a.calls, fail_restore=True)
        with self.assertRaisesRegex(RuntimeError, "restore boom"):
            run_pass(a, residency, "fb", "sb", store=None)
        self.assertEqual(residency.restore_calls, 1)
        self.assertEqual(a.calls[-3:], ["res.restore", "res.end", ("release", True)])
        self.assertNotIn("finish_pass", a.calls)

    def test_end_failure_still_releases(self):
        a = _Adapter()
        residency = _Residency(a.calls, fail_end=True)
        with self.assertRaisesRegex(RuntimeError, "end boom"):
            run_pass(a, residency, "fb", "sb", store=None)
        self.assertEqual(a.calls[-3:], ["finish_pass", "res.end", ("release", False)])


if __name__ == "__main__":
    unittest.main()
