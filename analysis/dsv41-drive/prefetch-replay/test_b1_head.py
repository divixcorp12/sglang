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


def _reference_batch(groups, *, model):
    """The original per-group make_batch, kept as the oracle for the vectorised one."""
    from sglang.srt.layers.moe.expert_prediction.training import llapor
    from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT

    cfg, pca, vocab, e = model.cfg, model.pca, model.vocab, model.cfg.num_experts
    us, ys, negs = [], [], []
    for g in groups:
        feats = llapor.encode_features(torch.from_numpy(_x_of(g)), torch.from_numpy(g.source_ids.astype(np.int64)),
                                       torch.from_numpy(g.source_weights.astype(np.float32)), pca=pca, num_experts=e)
        ids = torch.from_numpy(g.gate_ids.astype(np.int64))
        scores = torch.from_numpy(g.gate_scores.astype(np.float32))
        valid = (ids >= 0) & torch.isfinite(scores)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        margin = torch.where(valid, scores - scores[:, 5:6], torch.zeros_like(scores))
        dense = []
        for src in (scores, margin, valid.float()):
            d = torch.zeros((len(ids), e))
            d.scatter_add_(1, ids.clamp_min(0), src * valid.float())
            dense.append(d)
        tids = torch.from_numpy(g.target_ids.astype(np.int64))
        y = torch.zeros((len(tids), e))
        y.scatter_add_(1, tids.clamp_min(0), torch.from_numpy((g.target_ids >= 0) & (g.target_kinds != KIND_VRAM_HOT)).float())
        routed = torch.zeros((len(tids), e))
        routed.scatter_add_(1, tids.clamp_min(0), torch.from_numpy(g.target_ids >= 0).float())
        us.append(torch.cat([feats, dense[0], dense[1]], -1) if cfg.variant == "E5" else feats)
        ys.append(y.clamp(max=1.0))
        negs.append(dense[2] * (1.0 - routed.clamp(max=1.0)))
    layer_ix = {layer: i for i, layer in enumerate(vocab.layers)}
    counts = [len(g.tokens) for g in groups]
    rep = lambda vals: torch.tensor(np.repeat(np.array(vals, dtype=np.int64), counts))  # noqa: E731
    return dict(u=torch.cat(us), y=torch.cat(ys), hard_neg=torch.cat(negs),
                source_layer=rep([layer_ix[g.source_layer] for g in groups]),
                target_layer=rep([layer_ix[g.target_layer] for g in groups]),
                stratum=rep([vocab.strata.index(g.target_stratum) for g in groups]))


@pytest.mark.parametrize("variant", ["E2", "E5"])
def test_vectorised_batch_matches_the_per_group_reference(variant):
    model, groups = _model(variant)
    mixed = groups + [_group(seq=20, source_row=1, split="train", n=1, has_bias=False),
                      _group(seq=21, source_row=2, split="train", n=2, has_bias=False)]
    vocab = b1_head.Vocab.from_rows(mixed)
    model = b1_head.B1Head(model.cfg, pca=model.pca, vocab=vocab)
    got = b1_head.make_batch(mixed, x_of=_x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
    want = _reference_batch(mixed, model=model)
    for name, tensor in want.items():
        assert torch.allclose(getattr(got, name).float(), tensor.float(), atol=1e-5), name


def test_score_many_matches_one_group_at_a_time():
    model, groups = _model("E5")
    scorer = b1_head.HeadScorer(model, x_of=_x_of, name="e5")
    many = scorer.score_many(groups)
    assert len(many) == len(groups)
    for g, ts in zip(groups, many, strict=True):
        np.testing.assert_allclose(ts.scores, scorer(g).scores, atol=1e-6)


def test_pca_from_shards_reads_only_train_and_is_seeded(tmp_path):
    rng = np.random.default_rng(0)
    for split, session, rows in (("train", "a/1", 7), ("train", "b", 9), ("dev", "c", 5)):
        x = (rng.standard_normal((rows, DIM)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        b1_rows.write_x(tmp_path, split=split, session=session, x=x)
    a = b1_head.fit_train_pca_from_shards(tmp_path, sessions=["a/1", "b"], rank=RANK, max_rows=12, seed=1)
    b = b1_head.fit_train_pca_from_shards(tmp_path, sessions=["a/1", "b"], rank=RANK, max_rows=12, seed=1)
    assert a.components.shape == (RANK, DIM) and torch.equal(a.components, b.components)
    with pytest.raises(FileNotFoundError):
        b1_head.fit_train_pca_from_shards(tmp_path, sessions=["c"], rank=RANK, max_rows=12, seed=1)
    # With max_rows above the 16 train rows every row is used: the mean is the mean of the train x shards only.
    every = b1_head.fit_train_pca_from_shards(tmp_path, sessions=["a/1", "b"], rank=RANK, max_rows=100, seed=1)
    import router_score
    rows = np.concatenate([router_score.bf16_to_f32(np.asarray(b1_rows.read_x(tmp_path, split="train", session=s)))
                           for s in ("a/1", "b")])
    assert torch.allclose(every.mean, torch.from_numpy(rows).mean(0), atol=1e-5)


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
