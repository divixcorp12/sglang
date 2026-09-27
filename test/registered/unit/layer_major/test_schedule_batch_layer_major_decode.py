"""A layer-major extend's ring size must not survive into its own decode batch.

At max_running_requests=1, Scheduler.update_running_batch reuses the same
ScheduleBatch object across the extend-to-decode transition (running_batch =
last_batch when running_batch was empty), so without an explicit reset,
batch.layer_major_ring_tokens would still be set on the very next decode step
and the tp_worker seam would wrongly run the layer-major pass on it.
"""

import types
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLayerMajorRingClearedOnDecode(unittest.TestCase):
    def test_layer_major_ring_tokens_cleared_by_prepare_for_decode(self):
        batch = ScheduleBatch(reqs=[])
        batch.layer_major_ring_tokens = 4352
        # Spec branch returns immediately after the reset this test pins, so
        # the rest of decode-preparation's heavier fixture is not needed.
        batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)

        with patch("sglang.srt.speculative.spec_utils.spec_prepare_for_decode"):
            batch.prepare_for_decode()

        self.assertIsNone(batch.layer_major_ring_tokens)

    def test_non_layer_major_batch_stays_none(self):
        batch = ScheduleBatch(reqs=[])
        batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)

        with patch("sglang.srt.speculative.spec_utils.spec_prepare_for_decode"):
            batch.prepare_for_decode()

        self.assertIsNone(batch.layer_major_ring_tokens)


if __name__ == "__main__":
    unittest.main()
