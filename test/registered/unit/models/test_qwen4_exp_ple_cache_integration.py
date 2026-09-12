import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    _qwen4_exp_ple_cache_identity,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
from sglang.srt.models.qwen4_exp_ple_table import allocate_ple_host_table
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4ExpPleCacheIntegration(unittest.TestCase):
    def test_cache_identity_canonicalizes_checkpoint_and_records_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "checkpoint")
            alias = os.path.join(directory, "alias")
            os.mkdir(checkpoint)
            os.symlink(checkpoint, alias)
            model_config = SimpleNamespace(
                model_path=alias,
                revision="model-revision",
                hf_config=SimpleNamespace(_commit_hash="commit-hash"),
                hf_text_config=SimpleNamespace(),
            )
            server_args = SimpleNamespace(model_path=alias, revision="server-revision")

            identity = json.loads(
                _qwen4_exp_ple_cache_identity(
                    model_config=model_config, server_args=server_args
                )
            )

            self.assertEqual(identity["model_path"], os.path.realpath(checkpoint))
            self.assertEqual(identity["revision"], "model-revision")
            self.assertEqual(identity["commit_hash"], "commit-hash")

    def test_cold_cache_completion_starts_trimmer_after_exact_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            table = allocate_ple_host_table(
                (4, 2),
                torch.bfloat16,
                "file",
                directory,
                cache_identity="checkpoint\0module",
            )
            embedding = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
            embedding._ple_file_cache_table = table
            embedding._ple_file_cache = table._sglang_ple_file_cache
            embedding._ple_file_cache_seen_shards = set()
            embedding._file_rss_trimmer = None
            trimmer = object()

            embedding.record_ple_file_cache_shard(0)
            embedding.record_ple_file_cache_shard(1)
            with mock.patch.object(
                qwen4_exp_module,
                "make_ple_file_rss_trimmer",
                return_value=trimmer,
            ):
                embedding.complete_ple_file_cache(2)

            self.assertTrue(
                os.path.exists(f"{embedding._ple_file_cache.path}.manifest.json")
            )
            self.assertIs(embedding._file_rss_trimmer, trimmer)


if __name__ == "__main__":
    import unittest

    unittest.main()
