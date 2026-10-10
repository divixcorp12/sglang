"""Layer-0 experts predicted from the verify's token ids: id recovery, predictors, per-verify scoring, leads (CPU)."""

import importlib.util
import os

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _module():
    spec = importlib.util.spec_from_file_location(
        "layer0_draft_predict", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "layer0_draft_predict.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bf16_bits_widen_to_the_same_float32():
    m = _module()
    x = np.array([1.0, -2.5, 0.15625], dtype=np.float32)
    bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(m.bf16_to_f32(bits), x)


def test_the_nearest_normed_embedding_names_the_token_and_special_ids_are_never_chosen():
    m = _module()
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 16)).astype(np.float32)
    w = np.ones(16, np.float32)
    want = np.array([3, 17, 42, 3])
    x = m.rms_norm(emb[want], w) + 0.3 * rng.standard_normal((4, 16)).astype(np.float32)
    ids, c1, c2 = m.nearest_ids(x, emb, w, exclude=[], chunk=7)
    assert ids.tolist() == want.tolist()
    assert np.all(c1 >= c2)
    # Excluding the true id moves the answer to another token, never the excluded one.
    ids, _, _ = m.nearest_ids(x, emb, w, exclude=[3], chunk=7)
    assert 3 not in ids.tolist()


def test_gate_scores_are_sqrt_softplus_plus_bias_and_topk_takes_the_largest():
    m = _module()
    W = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    b = np.array([0.0, 0.5, 0.0], np.float32)
    s = m.gate_scores(np.array([[2.0, 0.0]], np.float32), W, b)
    assert np.allclose(s[0], np.sqrt(np.log1p(np.exp([2.0, 0.0, -2.0]))) + b)
    assert m.topk(s, 2).tolist() == [[0, 1]]


def test_the_table_shrinks_an_unseen_token_to_its_prior_and_a_frequent_one_to_its_counts():
    m = _module()
    tokens = np.array([7] * 100 + [8])
    routes = np.array([[0, 1]] * 100 + [[2, 3]])
    table = m.TokenTable.fit(tokens, routes, n_experts=5)
    prior = np.full((2, 5), 0.2, np.float32)
    s = table.scores(np.array([7, 9]), prior, beta=1.0)
    assert s[0, 0] > 0.98 and s[0, 1] > 0.98 and s[0, 4] < 0.01
    assert np.allclose(s[1], 0.4)  # k * prior: two routings of a 0.2 share each
    assert table.seen(np.array([7, 8, 9])).tolist() == [100, 1, 0]


def test_a_bias_fit_raises_an_expert_the_scores_under_predict():
    m = _module()
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((400, 6)).astype(np.float32)
    scores[:, 5] -= 3.0  # expert 5 almost never reaches the top 2
    routes = np.stack([np.full(400, 5), scores[:, :5].argmax(1)], 1)  # but it is always routed
    bias = m.fit_bias(scores, routes, k=2, iters=200, step=0.05)
    assert bias[5] > 1.0
    assert (m.topk(scores + bias, 2) == 5).any(1).mean() > 0.9


def test_union_keeps_each_experts_best_confidence_over_the_live_tokens():
    m = _module()
    scores = np.array([[0.9, 0.1, 0.5, 0.0], [0.2, 0.8, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]], np.float32)
    conf = scores  # a probability-like predictor is its own confidence
    pred = m.union_topk(scores, conf, k=2, live=2)  # token 2 is padding
    assert pred == {0: 0.9, 1: 0.8, 2: 0.6}


def test_a_verify_scores_useful_reads_on_nvme_rows_and_wasted_reads_on_unrouted_nvme_rows():
    m = _module()
    pred = {1: 0.9, 2: 0.9, 3: 0.5, 4: 0.2}
    routed = {1, 2, 5, 6}
    nvme = {2, 5}  # routed experts read from NVMe
    unrouted_nvme = {3, 7}  # tier of experts the verify does not route (approximate)
    r = m.score_verify(pred, routed, nvme, unrouted_nvme)
    assert r == {"predicted": 4, "routed": 4, "nvme": 2, "hit_routed": 2, "hit_nvme": 1, "wasted_nvme": 1,
                 "wasted_any": 2}


def test_confidence_bins_count_candidates_and_their_useful_and_wasted_reads():
    m = _module()
    pred = {1: 0.95, 2: 0.9, 3: 0.5, 4: 0.1}
    bins = m.bin_verify(pred, routed={1, 4}, nvme={1}, unrouted_nvme={2, 3}, edges=(0.0, 0.4, 0.8, 1.0001))
    assert bins == [{"candidates": 1, "routed": 1, "useful": 0, "wasted": 0},
                    {"candidates": 1, "routed": 0, "useful": 0, "wasted": 1},
                    {"candidates": 2, "routed": 1, "useful": 1, "wasted": 1}]


def test_the_recency_tier_calls_hot_experts_vram_recent_ones_ram_and_the_rest_nvme():
    m = _module()
    history = [{1, 2}, {3}, {4}]  # layer-0 experts routed by the three forwards before this one
    nvme = m.recency_nvme(history, window=2, hot={9}, n_experts=10)
    assert nvme == {0, 1, 2, 5, 6, 7, 8}


def test_the_lead_runs_from_the_last_draft_end_and_from_the_previous_verifys_last_gate_to_row_0s_demand():
    m = _module()
    ev = [
        {"event": "gate_open", "row": 39, "gen": 10, "ns": 1_000_000},
        {"event": "draft_observed", "row": 0, "gen": 0, "ns": 50_000_000},
        {"event": "draft_end", "row": 0, "gen": 0, "ns": 52_000_000},
        {"event": "copy_submit", "row": 0, "gen": 11, "ns": 54_500_000},
        {"event": "copy_submit", "row": 0, "gen": 11, "ns": 54_600_000},
        {"event": "copy_submit", "row": 1, "gen": 12, "ns": 60_000_000},
    ]
    leads = m.leads(ev)
    assert leads == {11: {"draft_end_ms": 2.5, "draft_observed_ms": 4.5, "prev_verify_end_ms": 53.5}}
