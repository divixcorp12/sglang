"""b1_cache.py against the uncached path: cached rows equal make_batch, the device recall@1 and the cached scorer
equal b1_score's, and cached training learns a planted rule deterministically. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_cache  # noqa: E402
import b1_head  # noqa: E402
import b1_score  # noqa: E402
import b1_train  # noqa: E402
from test_b1_head import RANK, _cfg, _model, _x_of  # noqa: E402
from test_b1_rows import _group  # noqa: E402
from test_b1_train import _planted  # noqa: E402
from test_b1_train import _x_of as _planted_x_of  # noqa: E402

E = b1_head.HeadConfig().num_experts


def _mixed():
    model, groups = _model("E5")
    groups = groups + [_group(seq=30, source_row=1, split="train", n=1, has_bias=False)]
    vocab = b1_head.Vocab.from_rows(groups)
    return model.pca, vocab, groups


@pytest.mark.parametrize("variant", ["E2", "E5"])
def test_cached_rows_equal_make_batch(variant):
    pca, vocab, groups = _mixed()
    cache = b1_cache.build(groups, x_of=_x_of, pca=pca, vocab=vocab, num_experts=E)
    want = b1_head.make_batch(groups, x_of=_x_of, pca=pca, cfg=_cfg(variant), vocab=vocab)
    got = b1_cache.take(cache, torch.arange(cache.rows), variant=variant)
    for name in b1_head.HeadBatch.__struct_fields__:
        assert torch.equal(getattr(got, name), getattr(want, name)), name
    sub = torch.tensor([3, 0, 5])
    part = b1_cache.take(cache, sub, variant=variant)
    assert torch.equal(part.u, want.u[sub]) and torch.equal(part.y, want.y[sub])


def test_device_recall_and_cached_scorer_match_b1_score():
    pca, vocab, groups = _mixed()
    groups = [g for g in groups]
    torch.manual_seed(1)
    model = b1_head.B1Head(_cfg("E5"), pca=pca, vocab=vocab).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)  # spread the scores so no near-ties decide the top read
    cache = b1_cache.build(groups, x_of=_x_of, pca=pca, vocab=vocab, num_experts=E)
    cached = b1_cache.CachedHeadScorer(model, cache=cache, name="h")
    exact = b1_score.evaluate(groups, b1_head.HeadScorer(model, x_of=_x_of, name="h"), reads=b1_score.READS)
    via_cache = b1_score.evaluate(groups, cached, reads=b1_score.READS)
    for budget, keyed in exact.cells.items():
        for key, c in keyed.items():
            assert via_cache.cells[budget][key] == c, (budget, key)
    recall = b1_cache.recall_at_1(cache, b1_cache.predict(model, cache))
    assert recall == pytest.approx(exact.cells["reads=1"]["pooled"].recall)


def _planted_caches():
    train = [_planted(i, split="train") for i in range(400)]
    dev = [_planted(1000 + i, split="dev") for i in range(100)]
    pca = b1_head.fit_train_pca(train, x_of=_planted_x_of, rank=RANK, max_rows=1000, seed=0)
    vocab = b1_head.Vocab.from_rows(train + dev)
    build = lambda gs: b1_cache.build(gs, x_of=_planted_x_of, pca=pca, vocab=vocab, num_experts=E)  # noqa: E731
    return build(train), build(dev), pca, vocab, dev


def _cached_run(seed):
    train_cache, dev_cache, pca, vocab, dev = _planted_caches()
    cfg = b1_head.HeadConfig(pca_rank=RANK, hidden=32, emb_dim=4, dropout=0.0)
    tcfg = b1_train.TrainConfig(epochs=25, lr=1e-2, batch_rows=32, patience=5, seed=seed)
    model, hist = b1_train.train_head_cached(cfg, tcfg, train_cache=train_cache, dev_cache=dev_cache, pca=pca,
                                             vocab=vocab)
    return model, hist, b1_train.dev_table(model, dev_cache=dev_cache, dev_groups=dev, name="h")


def test_cached_training_learns_the_planted_rule_and_is_seeded():
    a, ha, table = _cached_run(seed=3)
    assert ha["best_dev_recall_at_1"] > 0.9
    assert table.cells["reads=1"]["pooled"].recall == pytest.approx(ha["best_dev_recall_at_1"])
    b, hb, _ = _cached_run(seed=3)
    assert ha["history"] == hb["history"]
    for (name, ta), (_, tb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(ta, tb), name
    assert not a.training and np.isfinite([h["loss"] for h in ha["history"]]).all()
