"""Prefetch checkpoints load only when complete, unmodified and shaped like the live MoE layers."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.training import apex, llapor

HIDDEN, EXPERTS, TOP_K, RANK = 16, 32, 4, 8


def _specs(hidden=HIDDEN):
    return [MoeLayerSpec(layer_id=i, num_experts=EXPERTS, top_k=TOP_K, hidden_size=hidden) for i in (0, 1)]


def _write_llapor(root: Path, *, checksum_override=None, done=True) -> Path:
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import state_dict_sha256

    torch.manual_seed(0)
    directory = root / "llapor" / "pair-00"
    directory.mkdir(parents=True)
    model = llapor.build_predictor("middle", pca_rank=RANK, num_experts=EXPERTS)
    state = model.state_dict()
    torch.save(state, directory / "model.pt")
    torch.save({"mean": torch.randn(HIDDEN), "components": torch.randn(RANK, HIDDEN)}, directory / "pca.pt")
    manifest = {
        "architecture": {"group": "middle", "pca_rank": RANK, "num_experts": EXPERTS},
        "grouping": {"source_layer": 0, "target_layer": 1, "group": "middle"},
        "pca_stats": {"explained_variance": [1.0] * RANK},
        "tensor_checksums": {"model": checksum_override or state_dict_sha256(state)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if done:
        (directory / "DONE").write_text("ok")
    return directory


def _write_apex(root: Path) -> Path:
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import state_dict_sha256

    directory = root / "apex" / "layer-01"
    directory.mkdir(parents=True)
    ranker = apex.Ranker(HIDDEN, EXPERTS)
    cdf = apex.OrdinalCDF(HIDDEN, EXPERTS - TOP_K + 1)
    torch.save(ranker.state_dict(), directory / "ranker.pt")
    torch.save(cdf.state_dict(), directory / "cdf.pt")
    manifest = {
        "layer_id": 1,
        "architecture": {"hidden_size": HIDDEN, "num_experts": EXPERTS, "top_k": TOP_K},
        "tensor_checksums": {"ranker": state_dict_sha256(ranker.state_dict())},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "DONE").write_text("ok")
    return directory


class TestPrefetchCheckpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_llapor_loads_keyed_by_target_layer(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root)
        loaded = load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())
        self.assertEqual(sorted(loaded), [1])
        self.assertEqual((loaded[1].source_layer, loaded[1].target_layer), (0, 1))
        self.assertFalse(loaded[1].model.training)

    def test_apex_loads_with_cdf(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_apex(self.root)
        loaded = load_prefetch_checkpoints(self.root, predictor="apex", specs=_specs())
        self.assertEqual(loaded[1].top_k, TOP_K)
        self.assertEqual(loaded[1].cdf.thresholds().numel(), EXPERTS - TOP_K + 1)

    def test_checksum_mismatch_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root, checksum_override="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())

    def test_incomplete_checkpoint_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root, done=False)
        with self.assertRaisesRegex(ValueError, "DONE"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())

    def test_hidden_size_mismatch_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root)
        with self.assertRaisesRegex(ValueError, "hidden"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs(hidden=HIDDEN * 2))

    def test_unknown_layer_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_apex(self.root)
        with self.assertRaisesRegex(ValueError, "lacks"):
            load_prefetch_checkpoints(self.root, predictor="apex", specs=_specs()[:1])

    def test_unknown_predictor_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        with self.assertRaisesRegex(ValueError, "unknown prefetch predictor"):
            load_prefetch_checkpoints(self.root, predictor="affinity", specs=_specs())


if __name__ == "__main__":
    unittest.main()
