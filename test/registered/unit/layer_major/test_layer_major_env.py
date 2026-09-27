import os
import unittest
from unittest import mock

from sglang.srt.environ import envs


class TestLayerMajorEnv(unittest.TestCase):
    def test_defaults_leave_the_path_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in (
                "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS",
                "SGLANG_LAYER_MAJOR_STATE_NUMA_NODE",
            ):
                os.environ.pop(name, None)
            self.assertEqual(envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get(), 0)
            self.assertEqual(envs.SGLANG_LAYER_MAJOR_STATE_NUMA_NODE.get(), 1)

    def test_threshold_reads_from_the_environment(self):
        with mock.patch.dict(os.environ, {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "32768"}):
            self.assertEqual(envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get(), 32768)


if __name__ == "__main__":
    unittest.main()
