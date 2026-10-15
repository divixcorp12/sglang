"""The model-based CPU split (SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL): its cost words, the greedy, the Python lane typing
the device transcribes, and the launch gate (CPU). The device parity is test/manual/dsv41/test_exl3_lease_kernels_cuda.py
test_the_split_model_*."""

import random

import pytest

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.srt.environ import envs
from sglang.srt.layers.moe.ram_slot_map import (
    SPLIT_MODEL_DEFAULTS_MS,
    LaneKind,
    model_cpu_lanes,
    model_node_finish,
    split_model_stats_words,
    split_model_words,
    type_lanes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# One node: hit job a + b*k + c*pairs, miss ma + mc*tokens; copy_lane, first_byte, per_expert.
ONE = [0, 1000, 0, 0, 500] + [1500, 1000, 1000, 0, 0, 0]  # no stream side, no flags


def test_the_default_words_are_the_untraced_fits_in_ns_node_by_node():
    words = split_model_words(2)
    assert len(words) == 5 * 2 + 6
    assert words[:5] == [16_000, 154_000, 395_000, 331_000, 193_000]
    assert words[5:10] == [-50_000, 179_000, 390_000, 291_000, 245_000]
    assert words[10:] == [1_050_000, 800_000, 1_600_000, 650_000, 0, 0]
    assert split_model_words(2, residency_tiebreak=True)[-1] == 1
    assert split_model_words(1)[:5] == words[:5] and split_model_words(3)[10:15] == words[5:10]
    assert split_model_stats_words(2) == 5


def test_overrides_set_one_node_every_node_or_a_shared_cost():
    words = split_model_words(2, "hit_c.1=0.5, miss_a=0.25,copy_lane=1.3")
    assert words[2] == 395_000 and words[7] == 500_000
    assert words[3] == words[8] == 250_000
    assert words[10] == 1_300_000


@pytest.mark.parametrize("bad", ["hit_c", "nope=1", "copy_lane.0=1", "hit_c.2=1", "hit_c.x=1", "per_expert=-1",
                                 "hit_a=abc"])
def test_an_override_the_model_cannot_read_is_refused(bad):
    with pytest.raises(ValueError):
        split_model_words(2, bad)


def test_node_finish_runs_the_hit_job_then_each_miss_after_its_landing():
    # 2 hits: 2000 ns; misses land at 2000 and 3000 and cost 500 a token: 2000 + 500, then 3000 + 1000.
    assert model_node_finish(ONE, 0, 2, 3, [1, 2], 1) == 4000
    assert model_node_finish(ONE, 0, 0, 0, [], 1) == 0
    assert model_node_finish(ONE, 0, 0, 0, [1], 1) == 2000 + 500


def _greedy(tokens, *, forced=None, hit=None, eligible=None, copyable=None, node_of=None, words=ONE, nodes=1):
    n = len(tokens)
    return model_cpu_lanes(
        node_of or [0] * n, tokens, hit or [True] * n, eligible or [True] * n, forced or [False] * n,
        copyable or [True] * n, words, nodes)


def test_the_greedy_balances_the_cpu_against_the_copy_fifo():
    # 1 us an expert on the CPU, 1.5 us a copy lane: 1 CPU (1 <= 1.5), 2nd CPU 2 > 1.5 -> copy, 3rd CPU 2 <= 3.
    assert _greedy([1, 1, 1]) == [True, False, True]


def test_the_greedy_takes_the_heaviest_first_and_breaks_ties_by_plan_order():
    words = [0, 0, 1000, 0, 0] + [2500, 0, 0, 0, 0, 0]  # 1 us a (token, expert) pair
    # Lane 1 (3 tokens) first: CPU 3000 > 2500 -> copy; then lanes 0 and 2 (1 token each, plan order): CPU 1000 and
    # 2000 <= 5000.
    assert _greedy([1, 3, 1], words=words) == [True, False, True]
    # A tie at the copy FIFO goes to the CPU (finish <= copy): one 1-token lane at 1000 vs a 1000-ns copy lane.
    assert _greedy([1], words=[0, 0, 1000, 0, 0, 1000, 0, 0, 0, 0, 0]) == [True]


def test_forced_hits_and_uncopyable_hits_load_the_cpu_before_the_greedy():
    # A forced hit and an uncopyable eligible hit make the CPU 2 us before the candidates: the first candidate would
    # end at 3 > 1.5 -> copy, the second at 3 <= 3 -> CPU.
    got = _greedy([1, 1, 1, 1], forced=[False, False, False, True], copyable=[True, True, False, True])
    assert got == [False, True, True, False]  # forced lanes are the caller's: never set here


def test_a_forced_miss_after_the_hit_job_sets_the_cpus_floor():
    # Without the miss, 3 hits split CPU, copy, CPU. A forced 1-token miss lands at 2000 and costs 500, so the node
    # ends at 2500 with one or two hits: the first hit goes to the copy engine (2500 > 1500), the next two fit under
    # the FIFO's 3000.
    assert _greedy([1, 1, 1]) == [True, False, True]
    with_miss = _greedy([1, 1, 1, 1], forced=[False, False, False, True], hit=[True, True, True, False])
    assert with_miss == [False, True, True, False]
    assert model_node_finish(ONE, 0, 2, 2, [1], 1) == 2500


def test_both_nodes_share_one_copy_fifo():
    words = [0, 1000, 0, 0, 0] * 2 + [1500, 0, 0, 0, 0, 0]
    # Each node alone would put lanes 0 and 2 of its own on the CPU; together the shared FIFO fills: plan order
    # 0 (node 0, CPU 1000), 1 (node 1, CPU 1000), 2 (node 0, 2000 > 1500 -> copy), 3 (node 1, 2000 <= 3000 -> CPU).
    got = _greedy([1, 1, 1, 1], node_of=[0, 1, 0, 1], words=words, nodes=2)
    assert got == [True, True, False, True]


STREAM = [0, 1000, 0, 0, 0] + [1500, 1000, 1000, 2000, 0, 0]  # the stream kernel: 2 us a copy lane past 2 us


def test_the_stream_side_charges_copy_lanes_only_with_streamed_misses():
    # No GPU-streamed miss: CPU, copy, CPU as without the stream side. One streamed miss: the stream kernel would end at
    # 2000 + 2000 per copy lane, so every hit is cheaper on the CPU (1000, 2000, 3000 <= 4000).
    assert _greedy([1, 1, 1], words=STREAM) == [True, False, True]
    n = 3
    got = model_cpu_lanes([0] * n, [1] * n, [True] * n, [True] * n, [False] * n, [True] * n, STREAM, 1, streamed=1)
    assert got == [True, True, True]
    off = STREAM[:8] + [0, 0, 0]  # stream_lane 0: no stream side even with a streamed miss
    got = model_cpu_lanes([0] * n, [1] * n, [True] * n, [True] * n, [False] * n, [True] * n, off, 1, streamed=1)
    assert got == [True, False, True]


def test_type_lanes_counts_an_unforced_miss_as_streamed():
    ram = list(range(16)) + [-1] * 16
    kinds, _ = _typing([0, 1, 2, 20], ram, split_model=STREAM, tokens=[1] * 4)
    assert kinds == [LaneKind.HIT_CPU] * 3 + [LaneKind.MISS_GPU]
    # The same miss forced is the CPU's, not streamed: the copy side is the FIFO alone.
    kinds, _ = _typing([0, 1, 2, 20], ram, forced_from=3, split_model=STREAM, tokens=[1] * 4)
    assert kinds[3] == LaneKind.MISS_CPU and kinds[:3].count(LaneKind.HIT_CPU) < 3


def test_the_residency_tiebreak_gives_the_copy_engine_the_highest_keys_among_equal_tokens():
    # Without it, one-token lanes split CPU, copy, CPU in plan order: the CPU holds lane 0, the highest key. With it the
    # count stays (2 on the CPU) but the CPU takes the lowest keys, lanes 1 and 2, and the copy engine lane 0.
    off = _greedy([1, 1, 1])
    tied = _greedy([1, 1, 1], words=ONE[:-1] + [1])
    assert off == [True, False, True] and tied == [False, True, True]


def test_the_tiebreak_moves_lanes_only_within_a_node_and_a_token_count():
    words = [0, 1000, 0, 0, 0] * 2 + [1500, 0, 0, 0, 0, 1]
    tokens, node_of = [2, 1, 1, 2, 1, 1], [0, 0, 1, 0, 0, 1]
    plain = _greedy(tokens, node_of=node_of, words=words[:-1] + [0], nodes=2)
    tied = _greedy(tokens, node_of=node_of, words=words, nodes=2)
    for node in (0, 1):
        for tok in (1, 2):
            group = [j for j in range(6) if node_of[j] == node and tokens[j] == tok]
            assert sum(plain[j] for j in group) == sum(tied[j] for j in group)
            on = [j for j in group if tied[j]]
            off = [j for j in group if not tied[j]]
            assert all(c > x for c in on for x in off)  # every CPU lane below every copy lane in residency key


def test_ineligible_lanes_are_never_the_models():
    assert _greedy([1, 1], eligible=[False, True]) == [False, True]


def _typing(experts, ram, *, forced_from=None, split_model=None, tokens=None, hit_copy="ce", ce_ok=True, stats=None,
            scratch=None, nodes=1):
    lanes = 8
    w = wire_layout(lanes, nodes)
    split = [max(0, (2 * n) // 3) for _ in range(nodes) for n in range(w.lanes + 1)]
    staging = [100 * (n + 1) + k for n in range(nodes) for k in range(w.lanes)]
    return type_lanes(experts, ram, staging, split, lanes=lanes, captured=True, copy_armed=True, hit_copy=hit_copy,
                      cpu_on=True, cpu_misses=False, ce_ok=ce_ok, nodes=nodes, forced_from=forced_from, miss_cut=1,
                      miss_cut_max=3, split_model=split_model, tokens=tokens, model_stats=stats, scratch=scratch)


def test_off_the_typing_is_the_table_and_the_cut():
    rng = random.Random(3)
    for _ in range(200):
        experts = rng.sample(range(32), rng.randint(1, 8))
        ram = [rng.randrange(16) if rng.random() < 0.6 else -1 for _ in range(32)]
        forced_from = rng.randint(0, len(experts))
        assert _typing(experts, ram, forced_from=forced_from) == _typing(experts, ram, forced_from=forced_from,
                                                                         split_model=None, tokens=[1] * len(experts))


def test_on_forced_lanes_stay_cpu_and_the_counters_compare_with_the_table():
    ram = list(range(16))
    experts = [0, 1, 2, 3, 4]  # all hits; lane 4 forced
    stats = [0, 0, 0]
    kinds, _ = _typing(experts, ram, forced_from=4, split_model=ONE, tokens=[1] * 5, stats=stats)
    assert kinds[4] == LaneKind.HIT_CPU
    # The forced hit puts the CPU at 1 us; candidates 0..3 against a 1.5 us copy lane: 2 > 1.5 copy, 2 <= 3 CPU,
    # 3 <= 3 CPU (a tie is the CPU's), 4 > 3 copy.
    assert [k == LaneKind.HIT_CPU for k in kinds[:4]] == [False, True, True, False]
    table, _ = _typing(experts, ram, forced_from=4)
    on = [k == LaneKind.HIT_CPU for k in kinds[:4]]
    off = [k == LaneKind.HIT_CPU for k in table[:4]]
    assert stats == [1, sum(a and not b for a, b in zip(on, off)), sum(b and not a for a, b in zip(on, off))]


def test_without_the_copy_engine_the_table_types_and_nothing_counts():
    ram = list(range(16))
    stats = [0, 0, 0]
    for hit_copy, ce_ok in (("sm", True), ("ce", False)):
        got = _typing([0, 1, 2], ram, split_model=ONE, tokens=[1, 1, 1], hit_copy=hit_copy, ce_ok=ce_ok, stats=stats)
        assert got == _typing([0, 1, 2], ram, hit_copy=hit_copy, ce_ok=ce_ok)
    assert stats == [0, 0, 0]


def test_scratch_lanes_are_not_the_models():
    ram = list(range(16))
    kinds, slots = _typing([0, 1], ram, split_model=ONE, tokens=[1, 1], scratch=[5, -1])
    assert kinds[0] == LaneKind.HIT_SCRATCH and slots[0] == 5


def test_the_model_needs_each_lanes_tokens():
    with pytest.raises(ValueError, match="tokens"):
        _typing([0, 1], list(range(16)), split_model=ONE)
    with pytest.raises(ValueError, match="tokens"):
        _typing([0, 1], list(range(16)), split_model=ONE, tokens=[1])


def _check_slot_map():
    from sglang.srt.arg_groups import expert_stream_requirements_exl3 as req

    req._check_slot_map()


@pytest.mark.parametrize("overrides, match", [
    ({"SGLANG_DSV41_CPU_EXPERTS": False}, "CPU_EXPERTS=1"),
    ({"SGLANG_DSV41_CPU_EXPERTS": True, "SGLANG_DSV41_RAM_HIT_COPY": "sm"}, "RAM_HIT_COPY=ce"),
    ({"SGLANG_DSV41_CPU_EXPERTS": True, "SGLANG_DSV41_CPU_EXPERTS_MISSES": True}, "MISSES"),
    ({"SGLANG_DSV41_CPU_EXPERTS": True, "SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS": "nope=1"}, "cannot read"),
])
def test_the_gate_refuses_a_split_model_it_cannot_run(overrides, match):
    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL.override(True))
        for name, value in overrides.items():
            stack.enter_context(getattr(envs, name).override(value))
        with pytest.raises(ValueError, match=match):
            _check_slot_map()


def test_the_gate_accepts_the_split_model_and_refuses_costs_without_it():
    with envs.SGLANG_DSV41_CPU_EXPERTS.override(True), envs.SGLANG_DSV41_RAM_HIT_COPY.override("ce"):
        with envs.SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL.override(True):
            _check_slot_map()
        with envs.SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS.override("copy_lane=1"):
            with pytest.raises(ValueError, match="ENABLE_CPU_SPLIT_MODEL=1"):
                _check_slot_map()


def test_the_defaults_cover_every_key():
    from sglang.srt.layers.moe.ram_slot_map import SPLIT_MODEL_KEYS, SPLIT_MODEL_NODE_KEYS

    assert set(SPLIT_MODEL_DEFAULTS_MS) == set(SPLIT_MODEL_NODE_KEYS + SPLIT_MODEL_KEYS)
    assert not envs.SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL.get()  # off by default
