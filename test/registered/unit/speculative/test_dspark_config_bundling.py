import unittest
from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_config import (
    checkpoint_bundles_dspark_draft,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_dspark_keys_without_draft_layers_do_not_bundle(self):
        """A truncated export keeps the dspark_* keys but drops the draft
        (num_nextn_predict_layers: 0, no draft tensors)."""
        self.assertFalse(
            checkpoint_bundles_dspark_draft(
                {"dspark_block_size": 5, "num_nextn_predict_layers": 0}
            )
        )

    def test_dspark_keys_with_draft_layers_bundle(self):
        self.assertTrue(
            checkpoint_bundles_dspark_draft(
                {"dspark_block_size": 5, "num_nextn_predict_layers": 3}
            )
        )
        self.assertTrue(
            checkpoint_bundles_dspark_draft(
                SimpleNamespace(dspark_block_size=5, num_nextn_predict_layers=3)
            )
        )

    def test_missing_nextn_key_keeps_old_behaviour(self):
        self.assertTrue(checkpoint_bundles_dspark_draft({"dspark_block_size": 5}))

    def test_no_dspark_keys_never_bundles(self):
        self.assertFalse(
            checkpoint_bundles_dspark_draft({"num_nextn_predict_layers": 3})
        )


if __name__ == "__main__":
    unittest.main()
