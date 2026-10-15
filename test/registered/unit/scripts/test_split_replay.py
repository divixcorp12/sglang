import importlib.util
import os
import sys

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark")
sys.path.insert(0, _DIR)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolves string annotations through the module
    spec.loader.exec_module(mod)
    return mod


mr = _load("misscut_replay")
sr = _load("split_replay")

FLAT = {(t, p): sr.Cost(0.0, 1.0) for t in (0, 1) for p in (0, 1)}  # 1 ms per expert, hits and misses


def _group(**kw):
    base = dict(m=0, hits=0, dma=0, forced_hits=0, take=0, cut=0, issue=0.0, dma_end=0.0, hit=None, misses=[],
                free=0.0, cpu_end=0.0)
    base.update(kw)
    return mr.Group(**base)


def _node(team, toks, today, forced=(), misses=(), g=None):
    elig = list(enumerate(toks))
    return sr.Node(team, elig, frozenset(today), list(forced), list(misses), g or _group())


def test_cost_is_zero_without_experts_and_linear_in_experts_and_pairs():
    c = sr.Cost(0.1, 0.2, 0.4)
    assert c(0, 0) == 0.0
    assert abs(c(2, 3) - (0.1 + 0.4 + 1.2)) < 1e-12


def test_node_finish_runs_the_hit_job_then_each_miss_after_it_lands():
    p = sr.Params(first_byte=0.0, per_expert=2.0, copy_lane=1.0)
    n = _node(0, [1, 1, 1], [], misses=[1])
    # 3 hits end at 3; the miss lands at 2 and runs after the hit job: 4.
    assert sr.node_finish(n, {0: 1, 1: 1, 2: 1}, FLAT, p, 1.0) == 4.0
    # One hit ends at 1; the miss waits for its landing: 3.
    assert sr.node_finish(n, {0: 1}, FLAT, p, 1.0) == 3.0


def test_exhaustive_balances_both_cpus_against_the_shared_copy_fifo():
    p = sr.Params(first_byte=0.0, per_expert=2.0, copy_lane=1.0)
    nodes = [_node(0, [1] * 6, []), _node(1, [1] * 3, [])]
    sets = sr.rule_exhaustive(nodes, FLAT, p, 1.0, tokens=False)
    takes = [len(s) for s in sets]
    # 9 hits over three sides of 1 ms each: the gate cannot beat 3.
    assert max(takes[0], takes[1], 9 - sum(takes)) == 3


def test_exhaustive_with_tokens_puts_the_lightest_hits_on_the_cpu():
    cost = {(t, p): sr.Cost(0.0, 0.0, 1.0) for t in (0, 1) for p in (0, 1)}  # 1 ms per (token, expert) pair
    p = sr.Params(copy_lane=2.0)
    nodes = [_node(0, [3, 1, 1], [])]
    (s,) = sr.rule_exhaustive(nodes, cost, p, 1.0, tokens=True)
    assert set(s) == {1, 2}  # the CPU takes the two 1-token hits (2 ms), the copy engine the 3-token one (2 ms)


def test_greedy_sends_each_hit_to_the_side_that_finishes_first():
    p = sr.Params(copy_lane=1.5)
    nodes = [_node(0, [1, 1, 1], [])]
    (s,) = sr.rule_greedy(nodes, FLAT, p, 1.0, tokens=False)
    # Plan order 0, 1, 2: CPU 1 <= copy 1.5, CPU 2 > 1.5 goes to copy, then CPU 2 <= copy 3.
    assert set(s) == {0, 2}


def test_rule_table_takes_the_last_eligible_hits_less_the_flat_cut():
    nodes = [_node(0, [1] * 4, []), _node(1, [1] * 4, [], misses=[1])]
    table = [0, 1, 2, 2, 3]
    sets = sr.rule_table(nodes, table)
    assert set(sets[0]) == {1, 2, 3}
    assert set(sets[1]) == {2, 3}


def test_physics_moves_a_lane_from_the_hit_job_to_the_copy_fifo():
    g = _group(hits=2, dma=1, hit=(0.0, 2.0), issue=0.0)
    n = _node(0, [1, 1, 1], [1, 2], g=g)
    today = {1: 1, 2: 1}
    assert sr.physics({}, [n], today, FLAT, 1.0, 1.0) == 2.0
    assert sr.physics({}, [n], {2: 1}, FLAT, 1.0, 1.0) == 2.0  # CPU 1, copy 2
    assert sr.physics({}, [n], {0: 1, 1: 1, 2: 1}, FLAT, 1.0, 1.0) == 3.0


def test_layer_effect_delays_the_gpu_only_when_its_stream_kernel_runs():
    r = {"wait_start": 10.0, "stream": 2.0}
    # The gate moves 1 ms earlier; with coupling 1 the GPU arrives 1 ms later (one added copy lane).
    saving, arrive, wait = mr.layer_effect(r, 12.0, -1.0, 1, 1.0, 1.0)
    assert (saving, arrive, wait) == (1.0, 1.0, -2.0)
    assert mr.layer_effect(dict(r, stream=0.0), 12.0, -1.0, 1, 1.0, 1.0)[0] == 1.0
    assert mr.layer_effect(r, 12.0, -1.0, 1, 1.0, 0.0) == (1.0, 0.0, -1.0)
