"""Tests for r1_train: dev metric vs the shared policy, determinism, checkpoints, T0's cap-1 keep, F0, early stop."""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_cache  # noqa: E402
import r1_models  # noqa: E402
import r1_policy  # noqa: E402
import r1_train  # noqa: E402
from r1_groups import GroupBatch, GroupKey, GroupLabels, ModelInput  # noqa: E402
from r1_plan import ArmSpec  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E, RANK, HID, W = 12, 4, 8, 6
VOCAB = b1_head.Vocab(layers=(1, 2, 3), strata=("a", "b"))
HEAD_CFG = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=HID, emb_dim=2, num_experts=E)
F_DIM = RANK + 4 * E


def _spec(name="T1", *, objective="block", epochs=3, warmup=1, route_weight=0.0) -> ArmSpec:
    return ArmSpec(name=name, parent=None, objective=objective, route_weight=route_weight, epochs=epochs,
                   warmup=warmup)


def _model(*, aggregator=False) -> r1_models.R1Model:
    torch.manual_seed(0)
    pca = PCAStats(mean=torch.zeros(10), components=torch.randn(RANK, 10), explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(HEAD_CFG, pca=pca, vocab=VOCAB)
    return r1_models.R1Model(r1_models.R1Config(head=HEAD_CFG, aggregator=aggregator), head=head)


def _split(n: int, *, seed: int):
    rng = np.random.default_rng(seed)
    widths = rng.integers(1, W + 1, size=n)
    live = np.arange(W)[None, :] < widths[:, None]
    miss = rng.random((n, E)) < 0.25
    available = rng.random((n, E)) < 0.8
    route = rng.random((n, W, E)) < 0.2
    u = rng.standard_normal((n, W, F_DIM)).astype(np.float32) * live[..., None]
    u[..., 0] += miss[:, :1] * 2.0  # a learnable signal
    batch = GroupBatch(
        live=live, position=np.tile(np.arange(W, dtype=np.int64), (n, 1)), width=widths.astype(np.float32),
        source_layer=np.zeros(n, np.int64), target_layer=rng.integers(0, 3, n).astype(np.int64),
        stratum=rng.integers(0, 2, n).astype(np.int64), miss=miss.astype(np.float32),
        available=available.astype(np.float32), route=(route * live[..., None]).astype(np.float32),
        token_y=(route * live[..., None]).astype(np.float32), hard_neg=np.zeros((n, W, E), np.float32))
    inp = ModelInput(batch=batch, u=torch.from_numpy(u), x=None, s_native=None)
    keys = [GroupKey(session="s", block="b", seq=i, source_layer=0, target_layer=int(batch.target_layer[i]),
                     horizon=1) for i in range(n)]
    groups = []
    for i, k in enumerate(keys):
        lab = GroupLabels(
            miss=miss[i], available=available[i], route=route[i, : widths[i]], miss_gpu=miss[i],
            miss_cpu=np.zeros(E, bool), eligible_all=np.nonzero(miss[i])[0].astype(np.int16),
            eligible_addr=np.nonzero(miss[i] & available[i])[0].astype(np.int16))
        groups.append(SimpleNamespace(key=k, labels=lab))
    return r1_cache.GroupCache(inp=inp, keys=keys, groups=n), groups


def _train(spec, *, tmp_path, seed=3, aggregator=False, patience=8, name="run"):
    train, _ = _split(40, seed=1)
    dev, dev_groups = _split(24, seed=2)
    model = _model(aggregator=aggregator)
    res = r1_train.train_arm(spec, model=model, new_params={"head.predictor.fc_out"}, train_cache=train,
                             dev_cache=dev, dev_groups=dev_groups, seed=seed, out_dir=str(tmp_path / name),
                             groups_per_step=16, patience=patience, device="cpu", parent_sha="p" * 64)
    return model, res, dev, dev_groups


def test_dev_metric_matches_the_policy_sweep():
    dev, groups = _split(24, seed=2)
    model = _model()
    point = r1_train.dev_metric(model, dev_cache=dev, dev_groups=groups, cap=2)
    probs = r1_train._block_probs(model, cache=dev)
    cands = [r1_policy.candidates(row, skip=~g.labels.available) for row, g in zip(probs, groups)]
    pts = r1_policy.sweep(cands, [g.labels for g in groups], cap=2)
    assert point == r1_policy.best_under_budget(pts)
    assert point.mean_reads <= 1.0


def test_dev_metric_refuses_misordered_groups():
    dev, groups = _split(6, seed=2)
    with pytest.raises(ValueError):
        r1_train.dev_metric(_model(), dev_cache=dev, dev_groups=groups[::-1])


@pytest.mark.parametrize("objective,aggregator", [("b1_token", False), ("block", True), ("block+route", True)])
def test_objectives_train_and_save(tmp_path, objective, aggregator):
    spec = _spec(objective=objective, route_weight=0.25 if objective == "block+route" else 0.0)
    _, res, _, _ = _train(spec, tmp_path=tmp_path, aggregator=aggregator)
    assert [c["epoch"] for c in res.curve] == [0, 1, 2]
    assert all(math.isfinite(c["loss"]) for c in res.curve)
    assert res.parent_sha == "p" * 64
    loaded, sha, parent = r1_models.load_r1(res.ckpt_path)
    assert sha == res.ckpt_sha and parent == "p" * 64
    assert res.params == sum(p.numel() for p in loaded.parameters())


def test_seeded_runs_are_identical(tmp_path):
    _, a, _, _ = _train(_spec(), tmp_path=tmp_path, name="a")
    _, b, _, _ = _train(_spec(), tmp_path=tmp_path, name="b")
    assert a.curve == b.curve and a.best_epoch == b.best_epoch and a.ckpt_sha == b.ckpt_sha


def test_best_epoch_weights_are_restored(tmp_path):
    model, res, dev, groups = _train(_spec(), tmp_path=tmp_path)
    best = max(c["recall"] for c in res.curve)
    assert res.curve[res.best_epoch]["recall"] == best
    assert r1_train.dev_metric(model, dev_cache=dev, dev_groups=groups).recall == best


def test_only_t0_keeps_a_cap1_checkpoint(tmp_path):
    _, t0, _, _ = _train(_spec("T0", objective="b1_token"), tmp_path=tmp_path, name="t0")
    extra = t0.extra_ckpts["best_cap1"]
    assert os.path.exists(extra["path"]) and "recall_cap1" in t0.curve[0]
    assert r1_models.load_r1(extra["path"])[1] == extra["sha"]
    _, t1, _, _ = _train(_spec("T1"), tmp_path=tmp_path, name="t1")
    assert t1.extra_ckpts == {} and "recall_cap1" not in t1.curve[0]


def test_zero_epoch_arm_is_evaluated_and_saved_unchanged(tmp_path):
    before = _model().state_dict()
    model, res, _, _ = _train(_spec("F0", objective="b1_token", epochs=0, warmup=0), tmp_path=tmp_path)
    assert res.best_epoch == -1 and len(res.curve) == 1
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_early_stop_after_patience(tmp_path):
    spec = _spec(epochs=30)
    _, res, _, _ = _train(spec, tmp_path=tmp_path, patience=2)
    assert len(res.curve) <= res.best_epoch + 3
    assert len(res.curve) < 30


def test_frozen_encoder_does_not_move(tmp_path):
    spec = ArmSpec(name="T1", parent="T0", objective="block", route_weight=0.0, epochs=2, warmup=1,
                   freeze_encoder=True)
    before = {k: v.clone() for k, v in _model(aggregator=True).head.predictor.state_dict().items()}
    model, _, _, _ = _train(spec, tmp_path=tmp_path, aggregator=True)
    for k, v in model.head.predictor.state_dict().items():
        assert torch.equal(v, before[k])
