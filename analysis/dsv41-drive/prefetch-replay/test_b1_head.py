"""b1_head.py on synthetic groups: batch shapes and labels, embeddings, train-only PCA, scorer and save/load. CPU only."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_head  # noqa: E402
import b1_rows  # noqa: E402
from test_b1_rows import _group  # noqa: E402

DIM = 8  # fit_pca oversamples to min(rank + 10, DIM), which must not exceed the fitted row count
RANK = 4
E = b1_rows.NUM_EXPERTS


def _x_of(g):
    rng = np.random.default_rng(g.seq * 7 + g.source_row)
    return rng.standard_normal((len(g.tokens), DIM)).astype(np.float32)


def _train_groups(count=6):
    return [_group(seq=i, source_row=i % 3, split="train", n=2) for i in range(count)]


def _cfg(variant="E2"):
    return b1_head.HeadConfig(variant=variant, pca_rank=RANK, hidden=8, emb_dim=4, hard_neg_weight=3.0)


def _model(variant="E2"):
    groups = _train_groups()
    pca = b1_head.fit_train_pca(groups, x_of=_x_of, rank=RANK, max_rows=100, seed=0)
    return b1_head.B1Head(_cfg(variant), pca=pca, vocab=b1_head.Vocab.from_rows(groups)), groups


def test_pca_refuses_a_non_train_group():
    groups = _train_groups(2) + [_group(seq=9, split="dev")]
    with pytest.raises(ValueError):
        b1_head.fit_train_pca(groups, x_of=_x_of, rank=RANK, seed=0)


def test_pca_subsample_is_seeded_and_capped():
    groups = _train_groups(6)  # 12 rows
    a = b1_head.fit_train_pca(groups, x_of=_x_of, rank=RANK, max_rows=8, seed=3)
    b = b1_head.fit_train_pca(groups, x_of=_x_of, rank=RANK, max_rows=8, seed=3)
    assert a.components.shape == (RANK, DIM)
    assert torch.equal(a.mean, b.mean) and torch.equal(a.components, b.components)


@pytest.mark.parametrize("variant,dense", [("E2", 2), ("E5", 4)])
def test_batch_shapes_and_labels(variant, dense):
    model, groups = _model(variant)
    batch = b1_head.make_batch(groups[:2], x_of=_x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
    assert batch.u.shape == (4, RANK + dense * E)
    assert batch.y.shape == batch.hard_neg.shape == (4, E)
    # Token 0 routes 7,3,9,1,2,0 with kinds MISS_GPU,HIT,0,0,0,0: kind 0 (VRAM hot) is never a label.
    assert set(torch.nonzero(batch.y[0]).flatten().tolist()) == {7, 3}
    # Gate top-12 is experts 0..11; every routed id (hot or not) is excluded from the hard negatives.
    assert set(torch.nonzero(batch.hard_neg[0]).flatten().tolist()) == set(range(12)) - {7, 3, 9, 1, 2, 0}
    assert not (batch.y * batch.hard_neg).any()


def test_embeddings_follow_the_vocab_and_logits_shape():
    model, groups = _model()
    assert model.source_emb.num_embeddings == len(model.vocab.layers)
    assert model.stratum_emb.num_embeddings == len(model.vocab.strata) == 2
    batch = b1_head.make_batch(groups, x_of=_x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
    logits = model.eval()(batch)
    assert logits.shape == (12, E)
    loss = b1_head.head_loss(logits, batch, cfg=model.cfg)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_hard_negative_weight_raises_the_loss():
    model, groups = _model()
    batch = b1_head.make_batch(groups, x_of=_x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
    logits = model.eval()(batch)
    plain = b1_head.head_loss(logits, batch, cfg=b1_head.HeadConfig(hard_neg_weight=1.0))
    heavy = b1_head.head_loss(logits, batch, cfg=b1_head.HeadConfig(hard_neg_weight=5.0))
    assert heavy > plain


def test_scorer_returns_noisy_or_probabilities():
    model, groups = _model()
    ts = b1_head.HeadScorer(model, x_of=_x_of, name="e2")(groups[0])
    assert ts.rule == "noisy_or" and ts.scores.shape == (2, E) and ts.scores.dtype == np.float32
    assert ((ts.scores >= 0) & (ts.scores <= 1)).all()


def test_save_load_round_trip_and_checksum(tmp_path):
    model, groups = _model("E5")
    path = tmp_path / "head.pt"
    sha = b1_head.save_head(path, model)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, sha2 = b1_head.load_head(path)
    assert sha2 == sha and loaded.cfg == model.cfg and loaded.vocab == model.vocab
    batch = b1_head.make_batch(groups, x_of=_x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
    assert torch.allclose(model.eval()(batch), loaded(batch))
    assert b1_head.head_bytes_bf16(model) == 2 * (
        sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers()))
