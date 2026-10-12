"""Tests for r2_arms: budget refusal, T5/G0 training on tiny synthetic caches, parent checks and stage records."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_arms  # noqa: E402
import r1_cache  # noqa: E402
import r1_challengers  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r1_stage  # noqa: E402
import r1_train  # noqa: E402
import r2_arms  # noqa: E402
import r2_plan  # noqa: E402
from r1_groups import GroupBatch, GroupKey, GroupLabels, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E, RANK, HID, W, D = 12, 4, 8, 6, 10
VOCAB = b1_head.Vocab(layers=(1, 2, 3), strata=("a", "b"))
HEAD_CFG = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=HID, emb_dim=2, num_experts=E)
F_DIM = RANK + 4 * E
SEEDS = (0, 1, 2)
G0_CFG = r1_challengers.G0Config(rank=5, pos_dim=5, emb_dim=2, hidden=D, num_experts=E)


def _split(n: int, *, seed: int):
    rng = np.random.default_rng(seed)
    widths = rng.integers(1, W + 1, size=n)
    live = np.arange(W)[None, :] < widths[:, None]
    miss = rng.random((n, E)) < 0.25
    available = rng.random((n, E)) < 0.8
    route = rng.random((n, W, E)) < 0.2
    u = rng.standard_normal((n, W, F_DIM)).astype(np.float32) * live[..., None]
    x = torch.from_numpy(rng.standard_normal((n, W, D)).astype(np.float32) * live[..., None]).bfloat16()
    s_native = torch.from_numpy(rng.standard_normal((n, W, E)).astype(np.float32) * live[..., None])
    batch = GroupBatch(
        live=live, position=np.tile(np.arange(W, dtype=np.int64), (n, 1)), width=widths.astype(np.float32),
        source_layer=np.zeros(n, np.int64), target_layer=rng.integers(0, 3, n).astype(np.int64),
        stratum=rng.integers(0, 2, n).astype(np.int64), miss=miss.astype(np.float32),
        available=available.astype(np.float32), route=(route * live[..., None]).astype(np.float32),
        token_y=(route * live[..., None]).astype(np.float32), hard_neg=np.zeros((n, W, E), np.float32))
    inp = ModelInput(batch=batch, u=torch.from_numpy(u), x=x, s_native=s_native)
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


def _data() -> r1_arms._Data:
    train, tg = _split(32, seed=1)
    dev, dg = _split(20, seed=2)
    return r1_arms._Data(groups={"train": tg, "dev": dg}, caches={"train": train, "dev": dev})


def _tiny_arms() -> tuple[r1_plan.ArmSpec, ...]:
    fast = {"epochs": 6, "warmup": 1, "lr_new": 1e-2, "lr_inherited": 1e-2}
    t5, g0 = r2_plan._arms()
    return (msgspec.structs.replace(t5, lr_projection=1e-2, **fast), msgspec.structs.replace(g0, **fast))


@pytest.fixture
def env(tmp_path, monkeypatch):
    torch.manual_seed(0)
    pca = PCAStats(mean=torch.zeros(D), components=torch.randn(RANK, D), explained_variance=torch.ones(RANK))
    head_path = str(tmp_path / "head.pt")
    b1_head.save_head(head_path, b1_head.B1Head(HEAD_CFG, pca=pca, vocab=VOCAB))
    r1_run = tmp_path / "r1"
    (r1_run / "stages" / "audit").mkdir(parents=True)
    (r1_run / r1_arms.EXCLUSIONS_REL).write_text('{"cells": []}')
    t3 = r1_models.R1Model(r1_models.R1Config(head=HEAD_CFG, position=True, adapter_rank=0, aggregator=True),
                           head=b1_head.B1Head(HEAD_CFG, pca=pca, vocab=VOCAB))
    t3_path = str(tmp_path / "T3_seed0.pt")
    t3_sha = r1_models.save_r1(t3_path, t3, parent_sha="", arm="T3")
    out = tmp_path / "out"
    out.mkdir()
    ctx = r1_stage.StageCtx(out=str(out), dataset="d", corpus="c", model_path="m", b1_head=head_path, device="cpu",
                            smoke=False, plan_sha="a" * 64)
    base = r1_plan.default_plan(dataset="d", corpus="c", b1_head=head_path, out_dir=str(out), inputs={})
    plan = r2_plan.default_plan(r1_run=str(r1_run), r1_plan=base, r1_plan_sha="b" * 64, inputs={}, dataset="d",
                                corpus="c", b1_head=head_path, out_dir=str(out))
    plan = msgspec.structs.replace(plan, arms=_tiny_arms(), groups_per_step=8)
    seen = []

    def build(ctx, **kw):
        seen.append(kw)
        return _data()

    monkeypatch.setattr(r2_arms, "build_data", build)
    monkeypatch.setattr(r2_arms, "_g0_config", lambda: G0_CFG)
    return SimpleNamespace(ctx=ctx, plan=plan, parents={"T5": (t3_path, t3_sha)}, seen=seen, head_path=head_path,
                           f0=b1_head.load_head(head_path)[0])


def _train(env, *, seeds=SEEDS, parents=None):
    return r2_arms.train_arms(env.ctx, plan=env.plan, data=_data(), f0=env.f0, parents=parents or env.parents,
                              seeds=seeds, horizon=1, out_dir=os.path.join(env.ctx.out, "ckpt"))


def test_budget_refuses_one_group_past_and_accepts_at_budget():
    n = 7
    need = r1_cache.x_cache_bytes(n, x_dtype=torch.bfloat16)
    base = SimpleNamespace(x_dtype="bfloat16", x_budget_bytes=need, x_margin_bytes=0)
    assert r2_arms.check_x_budget(n_groups=n, plan=base, free_bytes=need) == need
    with pytest.raises(MemoryError, match="budget"):
        r2_arms.check_x_budget(n_groups=n + 1, plan=base, free_bytes=10 * need)
    with pytest.raises(MemoryError, match="free"):
        r2_arms.check_x_budget(n_groups=n, plan=base, free_bytes=need - 1)
    margin = SimpleNamespace(x_dtype="bfloat16", x_budget_bytes=need, x_margin_bytes=5)
    with pytest.raises(MemoryError):
        r2_arms.check_x_budget(n_groups=n, plan=margin, free_bytes=need + 4)
    assert r2_arms.check_x_budget(n_groups=n, plan=margin, free_bytes=need + 5) == need


def test_three_seeds_per_arm_with_parent_hashes(env):
    res = _train(env)
    assert set(res) == {"T5", "G0"}
    for arm in res:
        assert set(res[arm]) == set(SEEDS)
    parent_sha = env.parents["T5"][1]
    assert all(r.parent_sha == parent_sha for r in res["T5"].values())
    assert all(r.parent_sha == "" for r in res["G0"].values())
    assert len({r.ckpt_sha for arm in res.values() for r in arm.values()}) == 6


def test_arms_train_finitely_and_the_loss_falls(env):
    res = _train(env, seeds=(0,))
    for arm, by_seed in res.items():
        losses = [r["loss"] for r in by_seed[0].curve]
        assert all(np.isfinite(losses)), arm
        assert min(losses[1:]) < losses[0], arm
    data = _data()
    model, _ = r2_arms._fresh_g0(f0=env.f0, seed=0)
    spec = env.plan.arms[1]
    loss = r1_train._loss(model, r1_cache.take(data.caches["train"], torch.arange(8)), spec=spec, hard_neg_weight=1.0)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_saved_checkpoints_reload_to_the_same_scores(env):
    res = _train(env, seeds=(0,))
    data = _data()
    loaders = {"T5": lambda p: r1_challengers.load_with_projection(p)[0], "G0": lambda p: r1_challengers.load_g0(p)[0]}
    for arm, by_seed in res.items():
        r = by_seed[0]
        model = loaders[arm](r.ckpt_path)
        point = r1_train.dev_metric(model, dev_cache=data.caches["dev"], dev_groups=data.groups["dev"])
        assert point.recall == pytest.approx(r.curve[r.best_epoch]["recall"], abs=1e-9)
        assert r1_stage.sha256_file(r.ckpt_path) == r.ckpt_sha


def test_g0_at_step_zero_reproduces_the_native_scores(env):
    model, _ = r2_arms._fresh_g0(f0=env.f0, seed=0)
    inp = r1_cache.take(_data().caches["dev"], torch.arange(10))
    model.eval()
    with torch.no_grad():
        z = model.token_logits(inp)
    assert torch.equal(z, inp.s_native)
    live = torch.from_numpy(inp.batch.live)
    for g in range(3):
        row = inp.s_native[g][live[g]]
        assert torch.equal(z[g][live[g]].argsort(dim=-1), row.argsort(dim=-1))


def test_t5_parent_sha_mismatch_raises(env):
    path, sha = env.parents["T5"]
    bad = {"T5": (path, "0" * 64)}
    with pytest.raises(ValueError, match="sha256"):
        _train(env, seeds=(0,), parents=bad)


def test_stage_train_records_curves_and_hashes_and_is_cached(env):
    r = r2_arms.stage_train(env.ctx, plan=env.plan, stage="train", horizon=1, seeds=SEEDS, parents=env.parents,
                            arms=("T5", "G0"))
    assert r["label"] == r1_plan.LABEL
    for arm in ("T5", "G0"):
        assert set(r["arms"][arm]) == {"0", "1", "2"} and set(r["dev_recall"][arm]) == {"0", "1", "2"}
        for s in ("0", "1", "2"):
            assert r["ckpts"][arm][s]["sha"] == r["arms"][arm][s]["ckpt_sha"]
            assert r["arms"][arm][s]["curve"]
    assert r["ckpts"]["T5"]["0"]["parent_sha"] == env.parents["T5"][1]
    assert os.path.exists(os.path.join(r1_stage.stage_dir(env.ctx, "train"), "DONE.json"))
    calls = len(env.seen)
    assert r2_arms.stage_train(env.ctx, plan=env.plan, stage="train", horizon=1, seeds=SEEDS, parents=env.parents,
                               arms=("T5", "G0")) == r
    assert len(env.seen) == calls
    assert env.seen[0]["full"] is True and env.seen[0]["horizon"] == 1


def test_l2_stage_uses_its_own_parent_and_horizon(env):
    r = r2_arms.stage_train(env.ctx, plan=env.plan, stage="l2", horizon=2, seeds=(0,), parents=env.parents,
                            arms=("T5",))
    assert r["horizon"] == 2 and set(r["arms"]) == {"T5"} and set(r["arms"]["T5"]) == {"0"}
    assert env.seen[-1]["horizon"] == 2
    with pytest.raises(ValueError):
        r2_arms.stage_train(env.ctx, plan=env.plan, stage="calib", horizon=1, seeds=(0,), parents=env.parents,
                            arms=("T5",))
