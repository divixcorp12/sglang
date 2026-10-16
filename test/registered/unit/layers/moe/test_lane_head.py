"""The gather-time lane head (SGLANG_OPT_LANE_HEAD_PATH): its file, its Python definition against the training
pipeline (analysis/dsv41-drive/dspark/next_verify_head.py), and its device state (CPU).

The definition must serve what training saw. ``LaneHeadReference`` replays a verify sequence and is compared, row for
row, with next_verify_head's own extraction (its features) and predict (its probabilities), on a synthetic sequence
here and on the held-out C2 capture when divix01's files are present (``LANE_HEAD_C2_*``).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.srt.layers.moe import lane_head as lh
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[5]
DSPARK = REPO / "analysis" / "dsv41-drive" / "dspark"
EXPERTS = 384  # next_verify_head.NUM_EXPERTS: its model's embedding width


def _load_script(name: str):
    """Import an analysis script by path. next_verify_head blanks CUDA_VISIBLE_DEVICES at import (it trains on the
    CPU); restore it so the rest of the session keeps its devices."""
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    spec = importlib.util.spec_from_file_location(f"lane_head_test_{name}", DSPARK / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return mod


@pytest.fixture(scope="module")
def nvh():
    return _load_script("next_verify_head")


def _export(head_dir: Path, data_dir: Path, out: Path, proj_dtype: str, layers: int) -> lh.LaneHeadFile:
    exporter = _load_script("export_lane_head")
    argv = sys.argv
    sys.argv = ["export_lane_head.py", "--head", str(head_dir), "--data", str(data_dir), "--out", str(out),
                "--proj-dtype", proj_dtype, "--layer-ids", ",".join(str(i) for i in range(layers))]
    try:
        assert exporter.main() == 0
    finally:
        sys.argv = argv
    return lh.read_lane_head_file(str(out))


# ---- a synthetic verify sequence in next_verify_head's input shape -------------------------------------------------


class FakeSource:
    """next_verify_head.Source's fields over generated routes: two requests, a skewed expert pool (so experts recur
    within and across verifies), tied gate weights, verifies with a dead tail (live < tokens) and residents."""

    def __init__(self, layers=2, tokens=6, topk=6, hidden=64, verifies=(9, 7), seed=3):
        rng = np.random.default_rng(seed)
        n = sum(verifies)
        self.layers, self.tokens, self.topk, self.hidden = layers, tokens, topk, hidden
        pool = rng.choice(EXPERTS, size=40, replace=False)
        weights = rng.dirichlet(np.full(len(pool), 0.3))
        self.ids = np.zeros((n, layers, tokens, topk), dtype=np.int32)
        for idx in np.ndindex(n, layers, tokens):
            self.ids[idx] = rng.choice(pool, size=topk, replace=False, p=weights)
        self.w = rng.uniform(0.01, 0.4, size=(n, layers, tokens, topk)).astype(np.float32)
        self.w[:, :, 1, 0] = self.w[:, :, 0, 0]  # ties: the earliest token wins
        self.ids[:, :, 1, 0] = self.ids[:, :, 0, 0]
        xs = rng.standard_normal((n, layers, tokens, hidden)).astype(np.float32)
        self.x = (torch.from_numpy(xs).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
        self.recs = []
        router = 0
        for rid, count in enumerate(verifies):
            for _ in range(count):
                live = tokens if rng.random() < 0.7 else int(rng.integers(1, tokens))
                self.recs.append({
                    "rid": f"r{rid}", "live": live, "router": router, "conf": None,
                    "routes": [self.ids[router, L, :live].reshape(-1).tolist() for L in range(layers)],
                    "hot": [sorted(rng.choice(pool, size=12, replace=False).tolist()) for _ in range(layers)],
                })
                router += 1

    def x_rows(self, router: int, layer: int, live: int) -> np.ndarray:
        return (self.x[router, layer, :live, :].astype(np.uint32) << 16).view(np.float32)

    def serving_inputs(self, rec: dict, layer: int):
        """What the gather sees: every row of the verify graph, dead rows as mask_dead_routes leaves them (row 0's
        routes at weight 0), the bf16 gate input, the resident map and the dead mask."""
        ri, live = rec["router"], rec["live"]
        ids = torch.from_numpy(self.ids[ri, layer].copy())
        w = torch.from_numpy(self.w[ri, layer].copy())
        dead = torch.arange(self.tokens).view(-1, 1) >= live
        ids = torch.where(dead, ids[:1], ids)
        w = torch.where(dead, 0.0, w)
        x = torch.from_numpy(self.x[ri, layer].view(np.int16).copy()).view(torch.bfloat16)
        resident = torch.zeros(EXPERTS, dtype=torch.bool)
        resident[rec["hot"][layer]] = True
        return ids, w, x, resident, dead


def _write_training_dirs(nvh, src: FakeSource, root: Path, rank=16, seed=5):
    """next_verify_head's extract output for ``src`` (shards, pca.npz, manifest.json) and a random variant-B head in
    its train output format (L<LL>.npz, summary.json), with the draft odds constant as in the B0 corpus."""
    rng = np.random.default_rng(seed)
    data, head = root / "data", root / "head"
    data.mkdir()
    head.mkdir()
    mean = rng.standard_normal((src.layers, src.hidden)).astype(np.float32) * 0.1
    proj = rng.standard_normal((src.layers, src.hidden, rank)).astype(np.float32) / np.sqrt(src.hidden)
    np.savez(data / "pca.npz", mean=mean, proj=proj)
    info = nvh.extract_source(src, mean, proj, str(data), "syn")
    manifest = {"train": {"train-syn": {}}, "eval": {"syn": info}, "pca_rank": rank, "dense": list(nvh.DENSE)}
    (data / "manifest.json").write_text(json.dumps(manifest))
    features = len(nvh.DENSE)
    for L in range(src.layers):
        fm = rng.uniform(0.0, 1.0, features).astype(np.float32)
        fs = rng.uniform(0.2, 1.0, features).astype(np.float32)
        fm[15], fs[15] = 0.9999, 0.0042  # live: near-constant in training, as in B0
        fm[16], fs[16] = 1.0, 1e-6  # odds: constant, zeroed by constant_safe
        np.savez(
            head / f"L{L:02d}.npz", fm=fm, fs=fs, zs=rng.uniform(0.5, 2.0, rank).astype(np.float32),
            **{"state_bias.weight": rng.standard_normal((EXPERTS, 1)).astype(np.float32) * 0.5,
               "state_dense.weight": rng.standard_normal((1, features)).astype(np.float32) * 0.5,
               "state_dense.bias": rng.standard_normal(1).astype(np.float32),
               "state_z.weight": rng.standard_normal((1, rank)).astype(np.float32) * 0.3,
               "state_u.weight": rng.standard_normal((EXPERTS, rank)).astype(np.float32) * 0.3},
        )
    summary = {"variant": "B", "epochs": 1, "layers": {str(L): {"dev_logloss": 0.5} for L in range(src.layers)}}
    (head / "summary.json").write_text(json.dumps(summary))
    return data, head


def _replay(src, recs, layers: int, head: lh.LaneHeadFile, inputs) -> dict:
    """Run the reference over ``recs`` (a reset at each new request, as a prefill does); per (router, layer) the
    features, keys and probabilities."""
    ref = lh.LaneHeadReference(head.tensors, head.meta)
    out = {}
    rid = None
    for rec in recs:
        if rec["rid"] != rid:
            rid = rec["rid"]
            ref.reset()
        for L in range(layers):
            ids, w, x, resident, dead = inputs(rec, L)
            feats = ref.features(L, ids, w, x, resident, dead)
            keys, p = ref.gather(L, ids, w, x, resident, dead)
            out[(rec["router"], L)] = (feats, keys, p, resident)
    return out


def _check_against_training(nvh, data: Path, head_dir: Path, split: str, replay: dict, layers: int,
                            tol: tuple[float, float]) -> dict:
    """Every extracted (router, layer, expert) row: identical raw features, the same PCA coordinates (relative
    ``tol[0]``: numpy's BLAS against torch's), predict's probability (``tol[1]``), and per (router, layer) the same
    lane order over the misses."""
    pred = nvh.predict_split(str(data), str(head_dir), split)
    stats = {"rows": 0, "max_dp": 0.0, "max_dz": 0.0, "orders": 0}
    for L in range(layers):
        shard = np.load(data / split / f"L{L:02d}.npz")
        by_router: dict[int, list[int]] = {}
        for i, r in enumerate(shard["router"].tolist()):
            by_router.setdefault(r, []).append(i)
        for router, rows in by_router.items():
            if (router, L) not in replay:
                continue
            feats, keys, p, resident = replay[(router, L)]
            index = {e: j for j, e in enumerate(feats["e"])}
            for i in rows:
                e = int(shard["e"][i])
                j = index[e]
                assert np.array_equal(feats["f"][j].numpy(), shard["f"][i]), (router, L, e, feats["f"][j], shard["f"][i])
                dz = float(np.abs(feats["z"][j].numpy() - shard["z"][i]).max() / max(1.0, np.abs(shard["z"][i]).max()))
                stats["max_dz"] = max(stats["max_dz"], dz)
                dp = abs(float(p[e]) - float(pred[L]["p"][i]))
                stats["max_dp"] = max(stats["max_dp"], dp)
                stats["rows"] += 1
            misses = [int(shard["e"][i]) for i in rows if shard["miss"][i]]
            assert sorted(misses) == sorted(e for e in feats["e"] if not resident[e])
            want = [int(shard["e"][i]) for i in sorted((i for i in rows if shard["miss"][i]),
                                                      key=lambda i: (-float(pred[L]["p"][i]), int(shard["e"][i])))]
            got = sorted(misses, key=lambda e: -int(keys[e]))
            assert got == want, (router, L)
            stats["orders"] += 1
    assert stats["rows"] > 0 and stats["orders"] > 0
    assert stats["max_dz"] <= tol[0] and stats["max_dp"] <= tol[1], stats
    return stats


def test_the_reference_reproduces_next_verify_heads_features_probabilities_and_lane_order(nvh, tmp_path):
    """Synthetic sequence through next_verify_head's extract and predict, against the exported file through the
    reference with the serving inputs (dead rows included). Mutations: the score not reset per request, prefill or
    dead-row routes counted, the age off by one, the odds fed 0, a tie broken toward the later token, or the file's
    standardization folded differently all change a feature or a probability."""
    src = FakeSource()
    data, head_dir = _write_training_dirs(nvh, src, tmp_path)
    head = _export(head_dir, data, tmp_path / "lane.safetensors", "fp32", src.layers)
    replay = _replay(src, src.recs, src.layers, head, src.serving_inputs)
    stats = _check_against_training(nvh, data, head_dir, "syn", replay, src.layers, tol=(1e-5, 1e-6))
    print(stats)


def test_the_export_records_the_training_value_of_the_constant_odds_and_the_kernel_is_fed_it(nvh, tmp_path, monkeypatch):
    """Condition 1: the draft odds were constant (1.0) in training and zeroed by constant_safe; the file records 1.0,
    the scale zeroes the term, and the device launch passes exactly the metadata's value."""
    src = FakeSource(verifies=(3,))
    data, head_dir = _write_training_dirs(nvh, src, tmp_path)
    head = _export(head_dir, data, tmp_path / "lane.safetensors", "bf16", src.layers)
    assert head.meta.constant_features == {"odds": 1.0}
    assert head.meta.odds == 1.0
    assert torch.all(head.tensors["feature_scale"][:, lh.ODDS] == lh.CONSTANT_SCALE)
    assert torch.all(head.tensors["feature_mean"][:, lh.ODDS] == 1.0)
    calls = []
    import sglang.kernels.ops.moe.lane_head as ops

    monkeypatch.setattr(ops, "lane_head_keys", lambda *args: calls.append(args))
    device_head = lh.LaneHead(head, "cpu")
    ids, w, x, resident, dead = src.serving_inputs(src.recs[0], 0)
    device_head.launch(0, ids, w, x, torch.where(resident, 0, -1), dead)
    assert len(calls) == 1 and calls[0][-1] == head.meta.odds
    assert calls[0][3] is dead


def test_device_bytes_are_what_the_device_head_allocates(nvh, tmp_path):
    """The hot-cache budget is reduced by lane_head_device_bytes(meta) before sizing: it must equal the bytes the
    head then allocates. Mutation: dropping the per-request state from the count."""
    src = FakeSource(verifies=(2,))
    data, head_dir = _write_training_dirs(nvh, src, tmp_path)
    for dtype in ("bf16", "fp32"):
        head = _export(head_dir, data, tmp_path / f"lane-{dtype}.safetensors", dtype, src.layers)
        device_head = lh.LaneHead(head, "cpu")
        allocated = sum(t.numel() * t.element_size() for t in device_head.weights.values()) + sum(
            t.numel() * t.element_size()
            for t in (device_head.keys, device_head.score, device_head.last, device_head.verify_ct,
                      device_head.z_partial, device_head.decay_pow)
        )
        assert lh.lane_head_device_bytes(lh.read_lane_head_meta(str(tmp_path / f"lane-{dtype}.safetensors"))) == allocated
        assert device_head.device_bytes == allocated


def test_the_state_counts_live_verify_routes_decays_by_live_tokens_and_resets():
    """Score: decays by score_decay ** live, plus the verify's route counts; age: verifies since last routed; a dead
    row's routes count for nothing; reset is a new request."""
    meta = lh.LaneHeadMeta(
        format_version=1, layer_ids=(0,), hidden=16, rank=16, experts=8, features=lh.FEATURES, pca_proj_dtype="fp32",
        score_decay=0.5, constant_features={"odds": 1.0}, training={},
    )
    tensors = _tiny_tensors(meta)
    ref = lh.LaneHeadReference(tensors, meta)
    x = torch.zeros(2, 16, dtype=torch.bfloat16)
    resident = torch.zeros(8, dtype=torch.bool)
    ids = torch.tensor([[1, 2], [1, 3]])
    w = torch.tensor([[0.5, 0.25], [0.5, 0.75]])
    ref.gather(0, ids, w, x, resident)
    assert ref.score[0].tolist() == [0, 2, 1, 1, 0, 0, 0, 0]
    assert ref.last[0].tolist() == [-1, 0, 0, 0, -1, -1, -1, -1]
    feats = ref.features(0, torch.tensor([[1, 4], [3, 2]]), w, x, resident, dead=torch.tensor([[False], [True]]))
    assert feats["live"] == 1 and feats["e"] == [1, 4]
    f1 = feats["f"][0]
    assert f1[0] == 1 and f1[3] == 1 and f1[9] == 0.5  # one live route at position 0, its weight
    assert f1[10] == np.float32(np.log1p(2.0)) and f1[12] == np.float32(np.log1p(1)) and f1[13] == 0
    assert feats["f"][1][13] == 1 and feats["f"][1][11] == 1  # expert 4: never routed, no score
    ref.gather(0, torch.tensor([[1, 4], [3, 2]]), w, x, resident, dead=torch.tensor([[False], [True]]))
    assert ref.score[0].tolist() == [0, 2 * 0.5 + 1, 0.5, 0.5, 1, 0, 0, 0]
    assert ref.verify_ct[0] == 2
    ref.reset()
    assert not ref.score.any() and (ref.last == -1).all() and ref.verify_ct[0] == 0


def test_every_prefill_starts_a_request_for_the_heads_state():
    """The residency updater resets the head at every prefill it observes (boundary or not, graph-served or not),
    never at a verify or a draft. Mutation: resetting only at a qualifying prefill boundary."""
    from types import SimpleNamespace

    from sglang.srt.layers.moe.expert_residency_clock import ForwardKind
    from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

    resets = []
    updater = SimpleNamespace(
        lane_head=SimpleNamespace(reset=lambda: resets.append(1)), host_pending=False, flush=lambda: None,
        _count=lambda *args, **kwargs: None, _apply=lambda *args: None, enabled=torch.ones(1, dtype=torch.bool),
        prefill_promotions=0, decode_forwards=torch.zeros(1), boundary_pending=torch.zeros(1, dtype=torch.bool),
        _decode_boundary_reached=lambda: torch.zeros(1, dtype=torch.bool),
    )
    for kind, boundary, graph_served, expected in [
        (ForwardKind.PREFILL, False, False, 1),
        (ForwardKind.PREFILL, True, False, 1),
        (ForwardKind.PREFILL, False, True, 1),
        (ForwardKind.VERIFY, True, True, 0),
        (ForwardKind.DECODE, False, False, 0),
        (ForwardKind.DRAFT, False, False, 0),
    ]:
        resets.clear()
        GpuResidencyUpdater.observe_forward(updater, kind, 8, boundary, graph_served)
        assert len(resets) == expected, kind


def _tiny_tensors(meta: lh.LaneHeadMeta, proj_dtype=torch.float32) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    L, H, R, E, F = len(meta.layer_ids), meta.hidden, meta.rank, meta.experts, len(lh.FEATURES)
    scale = torch.ones(L, F)
    scale[:, lh.ODDS] = lh.CONSTANT_SCALE
    mean = torch.zeros(L, F)
    mean[:, lh.ODDS] = 1.0
    return {
        "pca_mean": torch.zeros(L, H), "pca_proj": torch.randn(L, H, R, generator=gen).to(proj_dtype),
        "feature_mean": mean, "feature_scale": scale, "z_scale": torch.ones(L, R),
        "expert_bias": torch.randn(L, E, generator=gen), "dense_w": torch.randn(L, F, generator=gen),
        "dense_b": torch.zeros(L), "z_w": torch.randn(L, R, generator=gen), "expert_u": torch.randn(L, E, R, generator=gen),
    }  # fmt: skip


@pytest.mark.parametrize(
    "defect, match",
    [
        ("missing", "missing or unreadable"),
        ("layers", "covers layers"),
        ("experts", "experts, the layers have"),
        ("hidden", "the router input is"),
        ("shape", "expert_u is"),
        ("odds_value", "training's mean differs"),
        ("odds_scale", "not zeroed by its scale"),
        ("odds_varies", "were not constant in training"),
        ("proj_dtype", "the metadata says"),
        ("nan", "non-finite"),
    ],
)
def test_a_bad_file_is_a_value_error_naming_the_path(tmp_path, defect, match):
    meta = lh.LaneHeadMeta(
        format_version=1, layer_ids=(0, 1), hidden=16, rank=16, experts=8, features=lh.FEATURES, pca_proj_dtype="bf16",
        score_decay=0.98, constant_features={"odds": 1.0}, training={},
    )
    tensors = _tiny_tensors(meta, torch.bfloat16)
    path = tmp_path / "lane.safetensors"
    kwargs = {}
    if defect == "layers":
        kwargs["layer_ids"] = [0, 2]
    elif defect == "experts":
        kwargs["experts"] = 9
    elif defect == "hidden":
        kwargs["hidden"] = 32
    elif defect == "shape":
        tensors["expert_u"] = tensors["expert_u"][:, :, :8].contiguous()
    elif defect == "odds_value":
        meta = lh.LaneHeadMeta(**{**_fields(meta), "constant_features": {"odds": 0.5}})
    elif defect == "odds_scale":
        tensors["feature_scale"][:, lh.ODDS] = 1.0
    elif defect == "odds_varies":
        meta = lh.LaneHeadMeta(**{**_fields(meta), "constant_features": {}})
    elif defect == "proj_dtype":
        tensors["pca_proj"] = tensors["pca_proj"].float()
    elif defect == "nan":
        tensors["z_w"][0, 0] = float("nan")
    if defect != "missing":
        lh.write_lane_head_file(str(path), tensors=tensors, meta=meta)
    with pytest.raises(ValueError, match=match) as err:
        lh.read_lane_head_file(str(path), **kwargs)
    assert str(path) in str(err.value)


def _fields(meta):
    return {name: getattr(meta, name) for name in meta.__struct_fields__}


# ---- the held-out C2 capture (divix01) ------------------------------------------------------------------------------

C2_DATA = Path(os.environ.get("LANE_HEAD_C2_DATA", "/mnt/nvme1/cache-tiers-tmp/head-data"))
C2_HEAD = Path(os.environ.get("LANE_HEAD_C2_HEAD", "/data/models/slang/nvfp4-work/ram-prefetch/cache-tiers-20261015/head/B"))
C2_VERIFIES = int(os.environ.get("LANE_HEAD_C2_VERIFIES", "16"))


@pytest.mark.skipif(not (C2_DATA / "c2").is_dir() or not C2_HEAD.is_dir(), reason="the C2 head data is on divix01")
def test_on_captured_c2_verifies_the_reference_reproduces_the_trained_heads_predictions(nvh, tmp_path):
    """The trained head B, exported with its fp32 projection, on the first verifies of C2 (held out): the router
    capture's routes and gate inputs through the reference give next_verify_head's features exactly, its predictions
    and its lane order. The bf16 export is measured against the same predictions."""
    manifest = json.loads((C2_DATA / "manifest.json").read_text())
    src = nvh.Source(manifest["eval"]["c2"]["dir"])
    recs = src.recs[:C2_VERIFIES]
    layers = src.layers

    def inputs(rec, L):
        ri, live = rec["router"], rec["live"]
        ids = torch.from_numpy(src.ids[ri, L].astype(np.int64))
        w = torch.from_numpy(src.w[ri, L].copy())
        dead = torch.arange(ids.shape[0]).view(-1, 1) >= live
        ids = torch.where(dead, ids[:1], ids)
        w = torch.where(dead, 0.0, w)
        x = torch.from_numpy(np.asarray(src.x[ri, L]).view(np.int16).copy()).view(torch.bfloat16)
        resident = torch.zeros(EXPERTS, dtype=torch.bool)
        resident[list(rec["hot"][L])] = True
        return ids, w, x, resident, dead

    for proj_dtype in ("fp32", "bf16"):
        head = _export(C2_HEAD, C2_DATA, tmp_path / f"c2-{proj_dtype}.safetensors", proj_dtype, layers)
        assert head.meta.odds == 1.0
        replay = _replay(src, recs, layers, head, inputs)
        if proj_dtype == "fp32":
            stats = _check_against_training(nvh, C2_DATA, C2_HEAD, "c2", replay, layers, tol=(1e-5, 1e-5))
        else:
            stats = _bf16_drift(nvh, C2_DATA, replay, layers)
        print(proj_dtype, stats)


def _bf16_drift(nvh, data: Path, replay: dict, layers: int) -> dict:
    """The bf16 projection's effect: the largest probability change and the share of (verify, layer) lane orders that
    still match the fp32 predictions."""
    pred = nvh.predict_split(str(data), str(C2_HEAD), "c2")
    max_dp, same, total = 0.0, 0, 0
    for L in range(layers):
        shard = np.load(data / "c2" / f"L{L:02d}.npz")
        rows_of: dict[int, list[int]] = {}
        for i, r in enumerate(shard["router"].tolist()):
            rows_of.setdefault(r, []).append(i)
        for router, rows in rows_of.items():
            if (router, L) not in replay:
                continue
            _, keys, p, _ = replay[(router, L)]
            for i in rows:
                max_dp = max(max_dp, abs(float(p[int(shard["e"][i])]) - float(pred[L]["p"][i])))
            miss = [i for i in rows if shard["miss"][i]]
            want = [int(shard["e"][i]) for i in sorted(miss, key=lambda i: (-float(pred[L]["p"][i]), int(shard["e"][i])))]
            got = sorted((int(shard["e"][i]) for i in miss), key=lambda e: -int(keys[e]))
            same += got == want
            total += 1
    assert max_dp <= 1e-3, max_dp
    return {"max_dp": max_dp, "orders_same": same, "orders": total}
