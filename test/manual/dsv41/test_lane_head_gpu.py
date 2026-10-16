"""The lane head's kernels (kernels/ops/moe/lane_head.py) on a real GPU against its Python definition
(srt/layers/moe/lane_head.LaneHeadReference): the plan keys order every gather's routed experts as the reference's
probabilities do, P agrees within 1e-3, the per-request state (score, last verify, counter) advances as the reference's,
a captured launch replays with new inputs, and the draft-odds argument is the value the kernel uses. Random inputs at
DSV4.1's sizes, and the trained head on C2's captured verifies when divix01's files are present.

The microbenchmark (``test_microbenchmark``, run with ``-s``) reports µs per layer and per 40-layer verify, eager and
graph-replayed, with the 40 layers' weights in L2 and with a 256 MB buffer touched before every layer's launch.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "test" / "registered" / "unit" / "layers" / "moe"))

import test_lane_head as cpu  # noqa: E402

from sglang.kernels.ops.moe import lane_head as ops  # noqa: E402
from sglang.srt.layers.moe import lane_head as lh  # noqa: E402

HIDDEN, RANK, EXPERTS, TOP_K, TOKENS = 5120, 64, 384, 6, 6


def _meta(layers: int, proj_dtype: str = "bf16", odds: float = 1.0) -> lh.LaneHeadMeta:
    return lh.LaneHeadMeta(
        format_version=1, layer_ids=tuple(range(layers)), hidden=HIDDEN, rank=RANK, experts=EXPERTS,
        features=lh.FEATURES, pca_proj_dtype=proj_dtype, score_decay=0.98, constant_features={"odds": odds},
        training={},
    )


def _random_head(layers: int, proj_dtype=torch.bfloat16, seed=0) -> dict[str, torch.Tensor]:
    """A head at DSV4.1's sizes with logits of a trained head's spread (P mostly in 0.05..0.95)."""
    g = torch.Generator().manual_seed(seed)
    F = len(lh.FEATURES)
    scale = torch.rand(layers, F, generator=g) * 0.8 + 0.2
    scale[:, lh.ODDS] = lh.CONSTANT_SCALE
    mean = torch.rand(layers, F, generator=g)
    mean[:, lh.ODDS] = 1.0
    return {
        "pca_mean": torch.randn(layers, HIDDEN, generator=g) * 0.05,
        "pca_proj": (torch.randn(layers, HIDDEN, RANK, generator=g) / HIDDEN**0.5).to(proj_dtype),
        "feature_mean": mean, "feature_scale": scale, "z_scale": torch.rand(layers, RANK, generator=g) + 0.5,
        "expert_bias": torch.randn(layers, EXPERTS, generator=g) * 0.5,
        "dense_w": torch.randn(layers, F, generator=g) * 0.3, "dense_b": torch.randn(layers, generator=g) * 0.1,
        "z_w": torch.randn(layers, RANK, generator=g) * 0.1, "expert_u": torch.randn(layers, EXPERTS, RANK, generator=g) * 0.1,
    }  # fmt: skip


def _file(tmp_path, tensors, meta) -> lh.LaneHeadFile:
    path = tmp_path / "lane.safetensors"
    lh.write_lane_head_file(str(path), tensors=tensors, meta=meta)
    return lh.read_lane_head_file(str(path))


def _gather_inputs(rng: np.random.Generator, live: int, pool: np.ndarray):
    ids = np.stack([rng.choice(pool, size=TOP_K, replace=False) for _ in range(TOKENS)]).astype(np.int32)
    w = rng.uniform(0.01, 0.4, size=(TOKENS, TOP_K)).astype(np.float32)
    dead = torch.arange(TOKENS).view(-1, 1) >= live
    ids_t = torch.where(dead, torch.from_numpy(ids)[:1], torch.from_numpy(ids))
    w_t = torch.where(dead, 0.0, torch.from_numpy(w))
    x = (torch.randn(TOKENS, HIDDEN, generator=torch.Generator().manual_seed(int(rng.integers(1 << 30)))) * 2).to(
        torch.bfloat16
    )
    resident = torch.from_numpy(rng.random(EXPERTS) < 0.3)
    return ids_t, w_t, x, resident, dead


def _p_of_keys(keys: torch.Tensor) -> torch.Tensor:
    return (keys >> 16).to(torch.int32).view(torch.float32)


def _compare(keys_dev, keys_ref, p_ref, routed) -> float:
    keys_dev = keys_dev.cpu()
    e = torch.tensor(sorted(routed), dtype=torch.int64)
    p_dev = _p_of_keys(keys_dev[e])
    dp = float((p_dev - p_ref[e]).abs().max())
    order_dev = e[torch.argsort(keys_dev[e], descending=True)].tolist()
    order_ref = e[torch.argsort(keys_ref[e], descending=True)].tolist()
    assert order_dev == order_ref, (order_dev, order_ref, p_dev, p_ref[e])
    return dp


@pytest.mark.parametrize("proj_dtype", ["bf16", "fp32"])
def test_kernel_keys_and_state_match_the_reference_on_random_gathers(tmp_path, proj_dtype):
    layers = 3
    head = _file(tmp_path, _random_head(layers, lh.PROJ_DTYPES[proj_dtype]), _meta(layers, proj_dtype))
    dev = lh.LaneHead(head, "cuda")
    ref = lh.LaneHeadReference(head.tensors, head.meta)
    rng = np.random.default_rng(1)
    pool = rng.choice(EXPERTS, size=48, replace=False)
    max_dp = 0.0
    for verify in range(40):
        if verify in (0, 23):  # a prefill
            dev.reset()
            ref.reset()
        for row in range(layers):
            live = TOKENS if rng.random() < 0.7 else int(rng.integers(1, TOKENS))
            ids, w, x, resident, dead = _gather_inputs(rng, live, pool)
            slots = torch.where(resident, 0, -1).to(torch.int64)
            dev.launch(row, ids.cuda(), w.cuda(), x.cuda(), slots.cuda(), dead.cuda())
            keys_ref, p_ref = ref.gather(row, ids, w, x, resident, dead)
            routed = {int(e) for e in ids[~dead.view(-1)].reshape(-1).tolist()}
            max_dp = max(max_dp, _compare(dev.keys[row], keys_ref, p_ref, routed))
    torch.cuda.synchronize()
    assert max_dp <= 1e-3, max_dp
    assert torch.equal(dev.last.cpu().long(), torch.from_numpy(ref.last))
    assert torch.equal(dev.verify_ct.cpu().long(), torch.from_numpy(ref.verify_ct))
    torch.testing.assert_close(dev.score.cpu().double(), torch.from_numpy(ref.score), rtol=1e-5, atol=1e-5)
    print(f"{proj_dtype}: max |dP| {max_dp:.2e}")


def test_the_kernel_uses_the_odds_it_is_given():
    """Condition 1's other half: with the odds term live (scale 1), the kernel's P is the reference's at that odds
    value, so the launch's argument is what the feature reads."""
    layers = 1
    tensors = _random_head(layers, torch.float32)
    tensors["feature_scale"][:, lh.ODDS] = 1.0
    tensors["feature_mean"][:, lh.ODDS] = 0.0
    tensors["dense_w"][:, lh.ODDS] = 2.0
    rng = np.random.default_rng(2)
    pool = rng.choice(EXPERTS, size=24, replace=False)
    ids, w, x, resident, dead = _gather_inputs(rng, TOKENS, pool)
    slots = torch.where(resident, 0, -1).to(torch.int64)
    routed = {int(e) for e in ids.reshape(-1).tolist()}
    for odds in (0.25, 1.0):
        meta = _meta(layers, "fp32", odds)
        ref = lh.LaneHeadReference(tensors, meta)
        keys_ref, p_ref = ref.gather(0, ids, w, x, resident, dead)
        dev = lh.LaneHead(lh.LaneHeadFile(path="mem", tensors=tensors, meta=meta), "cuda")
        dev.launch(0, ids.cuda(), w.cuda(), x.cuda(), slots.cuda(), dead.cuda())
        assert _compare(dev.keys[0], keys_ref, p_ref, routed) <= 1e-3


def test_a_captured_launch_replays_with_new_inputs(tmp_path):
    layers = 2
    head = _file(tmp_path, _random_head(layers), _meta(layers))
    dev = lh.LaneHead(head, "cuda")
    ref = lh.LaneHeadReference(head.tensors, head.meta)
    rng = np.random.default_rng(3)
    pool = rng.choice(EXPERTS, size=48, replace=False)
    ids_b = torch.zeros(TOKENS, TOP_K, dtype=torch.int32, device="cuda")
    w_b = torch.zeros(TOKENS, TOP_K, dtype=torch.float32, device="cuda")
    x_b = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    slots_b = torch.full((EXPERTS,), -1, dtype=torch.int64, device="cuda")
    dead_b = torch.zeros(TOKENS, 1, dtype=torch.bool, device="cuda")
    ids_b.copy_(torch.from_numpy(pool[: TOKENS * TOP_K].reshape(TOKENS, TOP_K).astype(np.int32)))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dev.launch(1, ids_b, w_b, x_b, slots_b, dead_b)  # warm-up: builds the module
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dev.launch(1, ids_b, w_b, x_b, slots_b, dead_b)
    dev.reset()
    for _ in range(6):
        live = int(rng.integers(1, TOKENS + 1))
        ids, w, x, resident, dead = _gather_inputs(rng, live, pool)
        ids_b.copy_(ids)
        w_b.copy_(w)
        x_b.copy_(x)
        slots_b.copy_(torch.where(resident, 0, -1))
        dead_b.copy_(dead)
        graph.replay()
        keys_ref, p_ref = ref.gather(1, ids, w, x, resident, dead)
        routed = {int(e) for e in ids[~dead.view(-1)].reshape(-1).tolist()}
        assert _compare(dev.keys[1], keys_ref, p_ref, routed) <= 1e-3
    assert int(dev.verify_ct[1]) == 6 and int(dev.verify_ct[0]) == 0


@pytest.mark.skipif(not (cpu.C2_DATA / "c2").is_dir() or not cpu.C2_HEAD.is_dir(), reason="the C2 head data is on divix01")
@pytest.mark.parametrize("proj_dtype", ["bf16", "fp32"])
def test_the_trained_head_on_captured_c2_verifies(tmp_path, proj_dtype):
    """The exported head B on C2's first verifies: the kernel's keys order every (verify, layer) as the reference's
    with the same file, P within 1e-3."""
    nvh = cpu._load_script("next_verify_head")
    manifest = json.loads((cpu.C2_DATA / "manifest.json").read_text())
    src = nvh.Source(manifest["eval"]["c2"]["dir"])
    head = cpu._export(cpu.C2_HEAD, cpu.C2_DATA, tmp_path / "c2.safetensors", proj_dtype, src.layers)
    dev = lh.LaneHead(head, "cuda")
    ref = lh.LaneHeadReference(head.tensors, head.meta)
    max_dp, gathers = 0.0, 0
    rid = None
    for rec in src.recs[: cpu.C2_VERIFIES]:
        if rec["rid"] != rid:
            rid = rec["rid"]
            dev.reset()
            ref.reset()
        ri, live = rec["router"], rec["live"]
        for L in range(src.layers):
            ids = torch.from_numpy(src.ids[ri, L].copy())
            w = torch.from_numpy(src.w[ri, L].copy())
            dead = torch.arange(ids.shape[0]).view(-1, 1) >= live
            ids = torch.where(dead, ids[:1], ids)
            w = torch.where(dead, 0.0, w)
            x = torch.from_numpy(np.asarray(src.x[ri, L]).view(np.int16).copy()).view(torch.bfloat16)
            resident = torch.zeros(EXPERTS, dtype=torch.bool)
            resident[list(rec["hot"][L])] = True
            dev.launch(L, ids.cuda(), w.cuda(), x.cuda(), torch.where(resident, 0, -1).cuda(), dead.cuda())
            keys_ref, p_ref = ref.gather(L, ids, w, x, resident, dead)
            routed = {int(e) for e in ids[~dead.view(-1)].reshape(-1).tolist()}
            max_dp = max(max_dp, _compare(dev.keys[L], keys_ref, p_ref, routed))
            gathers += 1
    assert max_dp <= 1e-3, max_dp
    print(f"C2 {proj_dtype}: {gathers} gathers, max |dP| {max_dp:.2e}")


def test_microbenchmark(tmp_path):
    """µs per layer launch (PCA + keys), 40 distinct layers, tokens 6, top_k 6. Warm: back to back, the weights
    (~29 MB) stay in the 96 MB L2. Cold: a 256 MB buffer is rewritten before each layer's launch, as the MoE work
    between two layers' gathers would evict it. Each timed with CUDA events around the launch alone, and the warm
    case also as one captured 40-layer graph replay."""
    layers = 40
    head = _file(tmp_path, _random_head(layers), _meta(layers))
    dev = lh.LaneHead(head, "cuda")
    rng = np.random.default_rng(4)
    pool = rng.choice(EXPERTS, size=64, replace=False)
    inputs = []
    for _ in range(layers):
        ids, w, x, resident, dead = _gather_inputs(rng, TOKENS, pool)
        inputs.append((ids.cuda(), w.cuda(), x.cuda(), torch.where(resident, 0, -1).cuda(), dead.cuda()))
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for row in range(layers):
        dev.launch(row, *inputs[row])
    torch.cuda.synchronize()

    def per_launch(cold: bool, reps: int = 20) -> list[float]:
        times = []
        for _ in range(reps):
            for row in range(layers):
                if cold:
                    flush.add_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                dev.launch(row, *inputs[row])
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1e3)
        return times

    warm = per_launch(False)
    cold = per_launch(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for row in range(layers):
            dev.launch(row, *inputs[row])
    graph.replay()
    torch.cuda.synchronize()
    replays = []
    for _ in range(50):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        replays.append(start.elapsed_time(end) * 1e3)
    report = {
        "eager_warm_us_per_layer": statistics.median(warm),
        "eager_cold_us_per_layer": statistics.median(cold),
        "eager_cold_p90_us_per_layer": sorted(cold)[int(0.9 * len(cold))],
        "graph_warm_us_per_verify": statistics.median(replays),
        "graph_warm_us_per_layer": statistics.median(replays) / layers,
        "device_bytes": dev.device_bytes,
    }
    report["cold_ms_per_verify_40_layers"] = report["eager_cold_us_per_layer"] * layers / 1e3
    print("lane head microbenchmark", json.dumps(report))
