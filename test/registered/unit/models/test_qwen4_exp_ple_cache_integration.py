import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    _checkpoint_cache_identity,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
)
from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table,
    get_ple_file_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ObservingVocabParallelEmbedding(torch.nn.Module):
    """Small stand-in that records the device used for PLE table construction."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        *,
        params_dtype,
        output_dtype,
        use_attn_tp_group,
    ):
        super().__init__()
        self.quant_config = None
        self.enable_tp = True
        self.use_attn_tp_group = use_attn_tp_group
        self.tp_size = 1
        self.num_embeddings = num_embeddings
        self.org_vocab_size = num_embeddings
        self.padding_size = 1
        self.num_added_embeddings = 0
        self.use_presharded_weights = False
        self.org_vocab_size_padded = num_embeddings
        self.num_embeddings_padded = num_embeddings
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=num_embeddings,
        )
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = num_embeddings
        self.num_org_embeddings_per_partition = num_embeddings
        self.num_added_embeddings_per_partition = 0
        self.quant_method = UnquantizedEmbeddingMethod()
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=params_dtype),
            requires_grad=False,
        )


def _ngram_config(*, ple_offload_embedding):
    return SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size=31,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=1,
        eos_token_id=2,
        ple_embedding_dtype=None,
        ple_offload_embedding=ple_offload_embedding,
        seed=1234,
    )


class TestQwen4ExpPleCacheIntegration(unittest.TestCase):
    def test_offload_builds_meta_placeholder_and_wraps_it_for_host_backends(self):
        """Catches a CUDA allocation before PLE offload can replace the table."""
        with mock.patch.object(
            qwen4_exp_module,
            "VocabParallelEmbedding",
            _ObservingVocabParallelEmbedding,
        ):
            regular = Qwen4ExpNGramEmbedding(
                _ngram_config(ple_offload_embedding=False), embedding_dim=8
            )
            offloaded = Qwen4ExpNGramEmbedding(
                _ngram_config(ple_offload_embedding=True), embedding_dim=8
            )

        self.assertNotEqual(regular.ngram_embedding.weight.device.type, "meta")
        self.assertEqual(offloaded.ngram_embedding.weight.device.type, "meta")
        self.assertNotEqual(offloaded.ngram_embedding.weight_scale.device.type, "meta")

        for backend in ("pinned", "file"):
            with self.subTest(backend=backend):
                with mock.patch.object(
                    qwen4_exp_module,
                    "VocabParallelEmbedding",
                    _ObservingVocabParallelEmbedding,
                ):
                    source = Qwen4ExpNGramEmbedding(
                        _ngram_config(ple_offload_embedding=True), embedding_dim=8
                    )
                host_table = torch.empty_like(
                    source.ngram_embedding.weight, device="cpu"
                )
                with (
                    mock.patch.object(
                        qwen4_exp_module,
                        "allocate_ple_host_table",
                        return_value=host_table,
                    ) as allocate,
                    mock.patch.object(
                        qwen4_exp_module,
                        "get_ple_file_cache",
                        return_value=None,
                    ),
                    mock.patch.object(
                        qwen4_exp_module,
                        "make_ple_file_prefetcher",
                        return_value=None,
                    ),
                    mock.patch.object(
                        qwen4_exp_module,
                        "make_ple_file_rss_trimmer",
                        return_value=None,
                    ),
                    mock.patch.object(
                        qwen4_exp_module,
                        "check_file_backend_supported",
                        return_value=True,
                    ),
                ):
                    wrapped = Qwen4ExpPinnedHostEmbedding(
                        source.ngram_embedding,
                        backend=backend,
                    )

                self.assertEqual(wrapped.weight.device.type, "cpu")
                self.assertIs(
                    wrapped.weight_scale, source.ngram_embedding.weight_scale
                )
                self.assertEqual(allocate.call_args.kwargs["shape"], (18, 4))

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

            identity = _checkpoint_cache_identity(
                model_config=model_config, server_args=server_args
            )

            self.assertEqual(
                identity,
                {
                    "model_path": os.path.realpath(checkpoint),
                    "revision": "model-revision",
                    "commit_hash": "commit-hash",
                },
            )
            with self.assertRaises(TypeError):
                identity["revision"] = "changed"

    def test_cache_identity_is_stable_and_distinguishes_checkpoint_inputs(self):
        def identity(model_path, revision, commit_hash):
            return _checkpoint_cache_identity(
                model_config=SimpleNamespace(
                    model_path=model_path,
                    revision=revision,
                    hf_config=SimpleNamespace(_commit_hash=commit_hash),
                    hf_text_config=SimpleNamespace(),
                ),
                server_args=SimpleNamespace(
                    model_path=model_path, revision="server-revision"
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "checkpoint")
            other_checkpoint = os.path.join(directory, "other-checkpoint")
            os.mkdir(checkpoint)
            os.mkdir(other_checkpoint)

            baseline = identity(checkpoint, "revision-a", "commit-a")

            self.assertEqual(
                baseline,
                identity(checkpoint, "revision-a", "commit-a"),
            )
            self.assertNotEqual(
                baseline,
                identity(other_checkpoint, "revision-a", "commit-a"),
            )
            self.assertNotEqual(
                baseline,
                identity(checkpoint, "revision-b", "commit-a"),
            )
            self.assertNotEqual(
                baseline,
                identity(checkpoint, "revision-a", "commit-b"),
            )

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

            reopened = allocate_ple_host_table(
                (4, 2),
                torch.bfloat16,
                "file",
                directory,
                cache_identity="checkpoint\0module",
            )
            reopened_cache = get_ple_file_cache(reopened)
            self.assertIsNotNone(reopened_cache)
            self.assertTrue(reopened_cache.cache_hit)
            reopened_cache.close()
            self.assertIs(embedding._file_rss_trimmer, trimmer)


if __name__ == "__main__":
    import unittest

    unittest.main()
