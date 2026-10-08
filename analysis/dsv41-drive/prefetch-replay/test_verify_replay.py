"""verify_replay.py: the per-group gate chain, the two-group tier and the predictors. CPU only."""

import sys
from pathlib import Path

from pytest import approx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify_replay as vr  # noqa: E402

LAYERS = [0, 1, 2]


def _args(**over):
    a = vr.parse(["trace.jsonl"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _loaded(forwards):
    return {"layer_ids": LAYERS, "forwards": forwards}


def _verify(seq, routes, hot=None, tokens=2):
    return {"kind": "graph", "seq": seq, "phase": "target_verify", "verify": True, "tokens": tokens, "rids": ["a"],
            "router": seq, "routes": {l: routes.get(l, []) for l in LAYERS}, "hot": {l: (hot or {}).get(l, []) for l in LAYERS},
            "misses": {}}


def _run(forwards, **over):
    a = _args(**over)
    return vr.run_one(a, _loaded(forwards), None)


ROWS = 40  # 14/13/13 rows over the three layers, 8/7/7 of them for group 0: room for every scenario below


def test_a_layer_with_one_forced_miss_waits_for_its_landing_and_its_job():
    """One expert of group 0, not in RAM: gate = row landing (2.2) + miss job (1.0); the layer ends gpu_ms later,
    and the two layers with no lanes cost gpu_ms each."""
    out = _run([_verify(0, {0: [2]})], ram_rows=ROWS, nvme_row_ms=2.2, miss_job_ms=1.0, gpu_ms=1.0, step_ms=0.0)
    assert out["gate_ms_per_step"] == approx(3.2) and out["step_ms_per_step"] == approx(3.2 + 1.0 + 2 * 1.0)
    assert out["ram_misses_per_step"] == 1


def test_hits_cost_the_split_cpu_job_or_the_dma_whichever_is_later():
    """Four hits of group 0 all in RAM: SPLIT gives 3 CPU lanes (0.3 + 3 * 0.6 = 2.1) against one DMA lane (1.2)."""
    first = _verify(0, {0: [0, 2, 4, 6]})
    second = _verify(1, {0: [0, 2, 4, 6]})
    out = _run([first, second], ram_rows=ROWS, split="0 1 2 2 3 3 4 5 5", dma_ms=1.2, cpu_fixed_ms=0.3, cpu_lane_ms=0.6, gpu_ms=0.0, step_ms=0.0)
    # The first verify reads the four rows (misses); the second hits all four.
    assert out["per_step_gate_ms"][1] == approx(2.1)


def test_gate_is_the_slower_group():
    """Three forced misses issued in route order share one queue: group 0's rows (experts 0, 2) land at 2.2 and 4.4,
    group 1's (expert 1) at 6.6. Group 0's serial jobs end at 5.4, group 1's at 7.6: the layer waits for the slower
    group, and the groups' gates are reported apart."""
    out = _run([_verify(0, {0: [0, 2, 1]})], ram_rows=ROWS, nvme_row_ms=2.2, miss_job_ms=1.0, gpu_ms=0.0, step_ms=0.0)
    assert out["gate_ms_per_step"] == approx(7.6)
    assert out["per_group_gate_ms"] == approx([5.4, 7.6])


def test_an_oracle_prefetch_turns_the_next_layers_miss_into_a_hit():
    """Layer 0 routes expert 0; layer 1 will route expert 2. With oracle h=1 the read for 2 is issued at layer 0's
    post, queues behind layer 0's demand row (lands at 4.4), and layer 0's chain (3.2) plus its GPU work (3.0)
    outlasts it, so layer 1's gate is one CPU hit lane, not a landing."""
    routes = {0: [0], 1: [2]}
    base = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=3.0, step_ms=0.0)
    arm = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=3.0, step_ms=0.0, predictor="oracle", h=1, k=1)
    assert base["per_layer_exposed_ms"][1] == approx(3.2)
    assert arm["per_layer_exposed_ms"][1] == approx(0.3 + 0.6)
    assert arm["spec_rows_per_step"] == 1 and arm["precision_target"] == 1.0
    assert arm["step_ms_per_step"] < base["step_ms_per_step"]


def test_a_demand_on_a_filling_row_waits_and_is_late():
    """A speculative read still in flight when its row is demanded is promoted, not read again, and counts late."""
    routes = {0: [0], 1: [2]}
    arm = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=0.0, step_ms=0.0, predictor="oracle", h=1, k=1,
               nvme_row_ms=50.0, miss_job_ms=1.0)
    assert arm["late_per_step"] == 1 and arm["ram_misses_per_step"] == 1  # layer 0's own miss only
    assert arm["spec_rows_per_step"] == 1


def test_demand_reads_outrank_speculative_ones_in_prio_mode():
    """With a wrong speculative row queued ahead, a demand row waits at most one piece under prio, a whole row under fifo."""
    routes = {0: [0], 1: [4]}  # the oracle is wrong on purpose: noisy with p=0 issues a random other expert
    prio = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=0.0, step_ms=0.0, predictor="noisy", p=0.0, h=1, k=1, seed=3, queue="prio")
    fifo = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=0.0, step_ms=0.0, predictor="noisy", p=0.0, h=1, k=1, seed=3, queue="fifo")
    assert prio["demand_delay_mean_ms"] <= 2.2 / 8 + 1e-9
    assert fifo["demand_delay_mean_ms"] >= prio["demand_delay_mean_ms"]


def test_the_tier_splits_rows_between_the_two_groups_by_node_share():
    a = _args(ram_rows=10, node_share=0.6)
    tiers = vr.make_tiers(a, LAYERS)
    assert [[t.capacity for t in pair] for pair in tiers] == [[2, 2], [2, 1], [2, 1]]  # rows per layer 4, 3, 3
