"""Tests for r1_arms: ladder keep rule, ancestry, seed median, stage records, caching, smoke caps and calibration."""

from __future__ import annotations

import math
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
import r1_plan  # noqa: E402
import r1_policy  # noqa: E402
import r1_stage  # noqa: E402
from r1_groups import GroupBatch, GroupKey, GroupLabels, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E, RANK, HID, W = 12, 4, 8, 6
VOCAB = b1_head.Vocab(layers=(1, 2, 3), strata=("a", "b"))
HEAD_CFG = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=HID, emb_dim=2, num_experts=E)
F_DIM = RANK + 4 * E


def _split(n: int, *, seed: int):
    rng = np.random.default_rng(seed)
    widths = rng.integers(1, W + 1, size=n)
    live = np.arange(W)[None, :] < widths[:, None]
    miss = rng.random((n, E)) < 0.25
    available = rng.random((n, E)) < 0.8
    route = rng.random((n, W, E)) < 0.2
    u = rng.standard_normal((n, W, F_DIM)).astype(np.float32) * live[..., None]
    u[..., 0] += miss[:, None, :1] * 2.0
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


def _data():
    train, tg = _split(32, seed=1)
    dev, dg = _split(20, seed=2)
    return r1_arms._Data(groups={"train": tg, "dev": dg}, caches={"train": train, "dev": dev})


def _arms():
    base = {"route_weight": 0.0, "epochs": 2, "warmup": 1}
    return (
        r1_plan.ArmSpec(name="F0", parent=None, objective="b1_token", route_weight=0.0, epochs=0, warmup=0),
        r1_plan.ArmSpec(name="T0", parent=None, objective="b1_token", **base),
        r1_plan.ArmSpec(name="T1", parent="T0", objective="block", freeze_encoder=True, **base),
        r1_plan.ArmSpec(name="T2", parent="T1", objective="block+route", **{**base, "route_weight": 0.25}),
        r1_plan.ArmSpec(name="G0", parent=None, objective="block+route", g0=True, **base),
    )


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    torch.manual_seed(0)
    pca = PCAStats(mean=torch.zeros(10), components=torch.randn(RANK, 10), explained_variance=torch.ones(RANK))
    head_path = str(tmp_path / "head.pt")
    b1_head.save_head(head_path, b1_head.B1Head(HEAD_CFG, pca=pca, vocab=VOCAB))
    out = tmp_path / "out"
    (out / "stages" / "audit").mkdir(parents=True)
    (out / r1_arms.EXCLUSIONS_REL).write_text('{"cells": []}')
    c = r1_stage.StageCtx(out=str(out), dataset="d", corpus="c", model_path="m", b1_head=head_path, device="cpu",
                          smoke=False, plan_sha="a" * 64)
    seen = []

    def build(ctx, *, horizon, full, f0):
        seen.append((horizon, full))
        if full:
            raise MemoryError("x cache over budget")
        return _data()

    monkeypatch.setattr(r1_arms, "_build_data", build)
    plan = msgspec.structs.replace(r1_plan.default_plan(dataset="d", corpus="c", b1_head=head_path,
                                                        out_dir=str(out), inputs={}), arms=_arms())
    return SimpleNamespace(ctx=c, plan=plan, seen=seen)


def test_stage_train_walks_the_ladder_and_skips_g0_on_memory_refusal(ctx):
    r = r1_arms.stage_train(ctx.ctx, plan=ctx.plan)
    assert list(r["arms"]) == ["F0", "T0", "T1", "T2"]
    assert r["skipped"].keys() == {"G0"} and "MemoryError" in r["skipped"]["G0"]
    assert r["arms"]["T0"]["parent_sha"] == "" and r["arms"]["F0"]["best_epoch"] == -1
    for name in ("T1", "T2"):
        parent = r["parent_used"][name]
        assert r["arms"][name]["parent_sha"] == r["arms"][parent]["ckpt_sha"]
    assert (ctx.seen[0], ctx.seen[-1]) == ((1, False), (1, True))
    assert r["ancestry"][0] == "T0" and r["ancestry"][-1] == r["candidate"]
    assert os.path.exists(os.path.join(r1_stage.stage_dir(ctx.ctx, "train"), "DONE.json"))


def test_an_arm_that_adds_nothing_keeps_its_parent(ctx):
    st = r1_arms._Ladder()
    res = lambda arm, recall, params=10: r1_arms.r1_train.ArmResult(  # noqa: E731
        arm=arm, seed=0, curve=[{"epoch": 0, "recall": recall}], best_epoch=0, ckpt_path="", ckpt_sha="",
        parent_sha="", extra_ckpts={}, params=params)
    specs = {s.name: s for s in _arms()}
    r1_arms._record(st, spec=specs["T0"], res=res("T0", 0.5), parent_name=None)
    r1_arms._record(st, spec=specs["T1"], res=res("T1", 0.5), parent_name="T0")  # equal: adds nothing
    r1_arms._record(st, spec=specs["T2"], res=res("T2", 0.6), parent_name=st.kept["T1"])
    assert st.kept == {"T0": "T0", "T1": "T0", "T2": "T2"}
    assert r1_arms._ancestry(st, "T2") == ["T0", "T2"]


def test_rerun_is_cached_and_changed_inputs_rerun(ctx):
    a = r1_arms.stage_train(ctx.ctx, plan=ctx.plan)
    calls = len(ctx.seen)
    b = r1_arms.stage_train(ctx.ctx, plan=ctx.plan)
    assert b == a and len(ctx.seen) == calls
    with open(r1_arms._exclusions_path(ctx.ctx), "w") as f:
        f.write('{"cells": [], "changed": 1}')
    r1_arms.stage_train(ctx.ctx, plan=ctx.plan)
    assert len(ctx.seen) > calls


def test_training_needs_a_frozen_plan(ctx):
    bad = msgspec.structs.replace(ctx.ctx, plan_sha=None)
    with pytest.raises(ValueError):
        r1_arms.stage_train(bad, plan=ctx.plan)


def test_seeds_report_median_and_keep_seed0_candidate(ctx):
    train = r1_arms.stage_train(ctx.ctx, plan=ctx.plan)
    s = r1_arms.stage_seeds(ctx.ctx, plan=ctx.plan, train=train)
    sel = s["selection"]
    assert sel["candidate"] == train["candidate"] and sel["ancestry"] == train["ancestry"]
    assert set(s["repeats"]) == {"1", "2"}
    scores = sel["dev_by_seed"][sel["candidate"]]
    assert set(scores) == {"0", "1", "2"} and scores["0"] == train["dev_recall"][sel["candidate"]]
    assert sel["median_dev"] == sorted(scores.values())[1]
    assert sel["ckpts"][sel["candidate"]]["path"] == train["arms"][sel["candidate"]]["ckpt_path"]
    assert "T0" in sel["dev_by_seed"] and "F0" in sel["ckpts"]


def test_l2_trains_ancestry_and_control_at_horizon_2(ctx):
    train = r1_arms.stage_train(ctx.ctx, plan=ctx.plan)
    sel = r1_arms.stage_seeds(ctx.ctx, plan=ctx.plan, train=train)["selection"]
    ctx.seen.clear()
    r = r1_arms.stage_l2(ctx.ctx, plan=ctx.plan, selection=sel)
    assert ctx.seen == [(2, False)] and r["horizon"] == 2 and r["control"] == "T0"
    assert set(r["arms"]) == set(sel["ancestry"]) | {"T0"}


def test_smoke_caps_epochs():
    spec = _arms()[1]
    eff = r1_arms._effective(spec, horizon=2, smoke=True)
    assert eff.epochs == 1 and eff.warmup == 1 and eff.horizon == 2
    assert r1_arms._effective(spec, horizon=1, smoke=False).epochs == spec.epochs


def test_calibrate_takes_the_lowest_threshold_within_budget():
    _, groups = _split(10, seed=5)
    rng = np.random.default_rng(0)
    cands = {"s": [r1_policy.candidates(rng.random(E).astype(np.float32), skip=~g.labels.available)
                   for g in groups]}
    out = r1_arms.calibrate(cands, labels=[g.labels for g in groups], caps=(1, 2), mean_reads=1.0)
    for cap in (1, 2):
        pts = r1_policy.sweep(cands["s"], [g.labels for g in groups], cap=cap)
        thr = r1_policy.calibrate_threshold(pts, mean_reads=1.0)
        cell = out["s"][cap]
        assert cell["threshold"] == (thr if math.isfinite(thr) else None)
        assert cell["point"]["mean_reads"] <= 1.0


def test_calibrate_reads_nothing_when_no_threshold_fits():
    _, groups = _split(4, seed=6)
    c = [r1_policy.candidates(np.ones(E, np.float32), skip=~g.labels.available) for g in groups]
    out = r1_arms.calibrate({"s": c}, labels=[g.labels for g in groups], caps=(1,), mean_reads=0.0)
    assert out["s"][1]["threshold"] is None and out["s"][1]["point"]["reads"] == 0
