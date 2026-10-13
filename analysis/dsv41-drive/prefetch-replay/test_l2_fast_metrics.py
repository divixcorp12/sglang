"""l2_fast_metrics on synthetic R1Groups: agreement, near-threshold listing, abstention and availability. CPU only."""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import l2_fast_metrics  # noqa: E402
import r1_groups  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

E = r1_groups.NUM_EXPERTS
MISS_GPU, HIT = int(LaneKind.MISS_GPU), int(LaneKind.HIT_COPY)
THR = 0.5


def _group(*, seq):
    n = 2
    target_ids = np.array([[7, 3, 9, 1, 2, 0], [10, 11, 12, 13, 14, -1]], dtype=np.int16)
    kinds = np.array([[MISS_GPU, MISS_GPU, 0, 0, 0, 0], [MISS_GPU, HIT, HIT, HIT, HIT, 0]], dtype=np.uint8)
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 3)
    return b1_rows.Group(
        session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=3, source_stratum="a", gate_has_bias=True, target_row=1,
        target_layer=4, target_stratum="b", tokens=np.array([0, 1], np.int16), x_offset=n * seq,
        source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6), source_weights=np.full((n, 6), 0.25, np.float32),
        target_ids=target_ids, target_kinds=kinds, ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


def _groups(tmp_path):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=[_group(seq=0), _group(seq=1)])
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def _scores(**by_id):
    s = np.full(E, -np.inf, np.float32)
    for k, v in by_id.items():
        s[int(k[1:])] = v
    return s


def test_identical_scores_agree_everywhere(tmp_path):
    # Catches a table that reports spurious diffs or disagreements for equal inputs.
    groups = _groups(tmp_path)
    s = [_scores(e7=0.9, e10=0.4), _scores(e7=0.2, e10=0.6)]
    t = l2_fast_metrics.parity_table(groups, ref=s, got=[x.copy() for x in s], threshold=THR)
    assert t["groups"] == 2 and t["max_abs_diff"] == 0.0 and t["p99_abs_diff"] == 0.0
    assert t["top1_agree"] == 1.0 and t["read_agree"] == 1.0
    assert t["disagreements"] == 0 and t["near_threshold"] == [] and t["far_disagreements"] == 0


def test_disagreement_within_near_is_listed_and_not_far(tmp_path):
    # Catches a near-threshold flip being counted as a real (far) read-set change.
    groups = _groups(tmp_path)
    ref = [_scores(e7=0.5005), _scores(e7=0.9)]
    got = [_scores(e7=0.4995), _scores(e7=0.9)]
    t = l2_fast_metrics.parity_table(groups, ref=ref, got=got, threshold=THR)
    assert t["disagreements"] == 1 and t["far_disagreements"] == 0 and t["read_agree"] == 0.5
    (d,) = t["near_threshold"]
    assert (d["group"], d["session"], d["ref_read"], d["got_read"]) == (0, "s1", 7, None)
    assert d["ref_score"] == np.float32(0.5005) and d["got_score"] == np.float32(0.4995)


def test_disagreement_outside_near_is_far(tmp_path):
    # Catches a clear-cut read change hiding in the near-threshold list.
    groups = _groups(tmp_path)
    ref = [_scores(e7=0.9), _scores(e7=0.9)]
    got = [_scores(e7=0.1, e10=0.8), _scores(e7=0.9)]
    t = l2_fast_metrics.parity_table(groups, ref=ref, got=got, threshold=THR)
    assert t["disagreements"] == 1 and t["far_disagreements"] == 1 and t["near_threshold"] == []
    assert t["top1_agree"] == 0.5


def test_both_abstaining_is_agreement(tmp_path):
    # Catches treating (None, None) as a mismatch, which would flood the table with abstentions.
    groups = _groups(tmp_path)
    ref = [_scores(e7=0.1), _scores(e7=0.2)]
    got = [_scores(e10=0.3), _scores(e7=0.1)]
    t = l2_fast_metrics.parity_table(groups, ref=ref, got=got, threshold=THR)
    assert t["read_agree"] == 1.0 and t["disagreements"] == 0


def test_unavailable_experts_never_enter_the_diff(tmp_path):
    # Expert 0 is VRAM-hot and 3 is in the RAM snapshot; catches skipped rows leaking into diffs or top-1.
    groups = _groups(tmp_path)
    assert not groups[0].labels.available[[0, 3]].any()
    ref = [_scores(e7=0.9, e0=0.1, e3=0.1), _scores(e7=0.9)]
    got = [_scores(e7=0.9, e0=50.0, e3=-7.0), _scores(e7=0.9)]
    t = l2_fast_metrics.parity_table(groups, ref=ref, got=got, threshold=THR)
    assert t["max_abs_diff"] == 0.0 and t["top1_agree"] == 1.0 and t["disagreements"] == 0


def test_recall_change_counts_useful_reads_against_all_eligible(tmp_path):
    # Catches a recall change computed on a different denominator than the F109 policy report.
    groups = _groups(tmp_path)
    ref = [_scores(e7=0.9), _scores(e7=0.9)]
    got = [_scores(e7=0.1), _scores(e7=0.9)]
    c = l2_fast_metrics.recall_change(groups, ref=ref, got=got, threshold=THR)
    assert c["ref_useful"] == 2 and c["got_useful"] == 1 and c["eligible"] > 0
    assert c["ref_recall"] == 2 / c["eligible"]
    assert math.isclose(c["diff"], -1 / c["eligible"])


def test_json_finite_maps_non_finite_to_none():
    # Catches inf leaking into result.json, which strict JSON parsers refuse.
    assert l2_fast_metrics.json_finite(math.inf) is None
    assert l2_fast_metrics.json_finite(-math.inf) is None
    assert l2_fast_metrics.json_finite(0.25) == 0.25
