import importlib.util
import os
import sys

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark", "misscut_replay.py"
)
_spec = importlib.util.spec_from_file_location("misscut_replay", _PATH)
mr = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.dirname(_PATH))
_spec.loader.exec_module(mr)

SPLIT = [0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8]


def _group(**kw):
    base = dict(m=0, hits=0, dma=0, forced_hits=0, take=0, cut=0, issue=0.0, dma_end=0.0, hit=None, misses=[],
                free=0.0, cpu_end=0.0)
    base.update(kw)
    return mr.Group(**base)


def test_flat_cut_applies_only_to_one_to_cut_max_misses():
    assert [mr.flat_cut(m, 4) for m in range(6)] == [0, 1, 1, 1, 0, 0]
    assert mr.flat_cut(2, 0) == 0
    assert mr.flat_cut(2, 4, cut=3) == 3


def test_infer_forced_hits_finds_the_fewest_consistent_with_the_split():
    # 4 CPU hits, 2 DMA lanes, no misses: split[6] = 4, so none forced.
    assert mr.infer_forced_hits(4, 2, 0, SPLIT) == 0
    # 5 CPU hits, 1 DMA lane, one miss (cut 1): u unforced hits need split[u + 1] - 1 == u, first true at u = 1
    # (split[2] = 2), so 4 are forced.
    assert mr.infer_forced_hits(5, 1, 1, SPLIT) == 4


def test_run_team_queues_misses_behind_the_hit_job():
    # The hit job ends at 3; the first miss lands at 2 and waits, the second lands at 5 after the first ends at 3.5.
    assert mr.run_team(0.0, (0.0, 3.0), [(2.0, 0.5), (5.0, 0.5)]) == 5.5
    assert mr.run_team(0.0, None, [(2.0, 0.5)]) == 2.5


def test_dma_fifo_serializes_groups_in_issue_order():
    assert mr.dma_fifo([0.1, 0.0], [1, 2], 1.0) == [3.0, 2.0]
    assert mr.dma_fifo([0.0, 0.0], [0, 1], 1.0) == [0.0, 1.0]


def test_graded_cut_ends_the_hit_job_near_the_first_landing():
    rule = mr.Rule(first_byte=0.0, per_expert=2.0, cpu_lane=1.0, miss_lane=0.5, copy_lane=1.0, cap=3)
    # One miss landing at 2: with 6 lanes the hit job ends at 6; cutting 4 ends it at 2, and the copy side (no other
    # DMA lanes) then ends at 4 > 2.5, so the best is a cut that balances: cut 3 gives max(3.5, 3) = 3.5.
    assert mr.graded_cut(1, 6, 0, 0, rule) == 3
    assert mr.graded_cut(1, 6, 0, 0, rule, copy_side=False) == 4
    assert mr.graded_cut(4, 6, 0, 0, rule) == 0
    assert mr.graded_cut(0, 6, 0, 0, rule) == 0


def test_first_landing_cut_counts_forced_hits_in_the_hit_job():
    rule = mr.Rule(first_byte=1.0, per_expert=2.0, cpu_lane=1.0, cap=3)
    assert mr.first_landing_cut(1, 6, 0, rule) == 3
    assert mr.first_landing_cut(1, 6, 2, rule) == 5
    assert mr.first_landing_cut(5, 6, 0, rule) == 0


def test_replay_record_moves_a_lane_from_the_hit_job_to_the_copy_engine():
    g = _group(m=1, hits=4, dma=1, take=5, cut=1, hit=(0.0, 4.0), misses=[(2.0, 0.5)], issue=0.0)
    cpu_lane = {0: 1.0}
    assert mr.replay_record({0: g}, {0: 1}, 1.0, cpu_lane) == 4.5
    # Cut 3: the hit job ends at 2, the miss at 2.5, the DMA's 3 lanes at 3.
    assert mr.replay_record({0: g}, {0: 3}, 1.0, cpu_lane) == 3.0
    # No cut: 5 hits end at 5, the miss at 5.5.
    assert mr.replay_record({0: g}, {0: 0}, 1.0, cpu_lane) == 5.5
