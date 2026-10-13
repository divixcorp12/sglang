"""l2_cap1_metrics.py on the hand-built group of test_r1_metrics.py: abstention and coverage counts, the gate-top-12
split of useful reads, and the DSpark extract with and without width keys. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import l2_cap1_metrics  # noqa: E402
import r1_groups  # noqa: E402
import r1_metrics  # noqa: E402
import r1_policy  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

E = r1_groups.NUM_EXPERTS
GPU, CPU, HIT, HOT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY), int(KIND_VRAM_HOT)

# Eligible rows 10, 11, 20, 30. Gate top-12 is 0..11, so 20 and 30 are outside it.
_IDS = [[10, 11, 12, 0, -1, -1], [10, 20, 30, 2, -1, -1]]
_KINDS = [[GPU, CPU, HIT, HOT, 0, 0], [GPU, GPU, GPU, HOT, 0, 0]]


def _group(*, seq=0, target_layer=4):
    n = 2
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 30)
    return b1_rows.Group(
        session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=target_layer - 1, source_stratum="a", gate_has_bias=True,
        target_row=1, target_layer=target_layer, target_stratum="b", tokens=np.array([0, 1], np.int16),
        x_offset=n * seq, source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=np.array(_IDS, np.int16),
        target_kinds=np.array(_KINDS, np.uint8), ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


def _groups(tmp_path, n=3):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1",
                         groups=[_group(seq=i, target_layer=4 + i) for i in range(n)])
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def _ids(*ids):
    return np.array(ids, dtype=np.int64)


def _cands(r, scores):
    s = np.full(E, -np.inf, np.float32)
    for e, v in scores.items():
        s[e] = v
    return r1_policy.candidates(s, skip=~r.labels.available)


def test_coverage_counts_add_up(tmp_path):
    groups = _groups(tmp_path)
    reads = [_ids(), _ids(10, 11), _ids(10, 11, 20, 30)]
    c = l2_cap1_metrics.coverage_counts(groups, reads)
    assert (c["forward_layers"], c["abstained"], c["layers_with_miss"]) == (3, 1, 3)
    assert c["abstention_rate"] == pytest.approx(1 / 3)
    assert (c["fully_covered"], c["partly_covered"], c["not_covered"]) == (1, 1, 1)
    assert c["abstained"] + sum(1 for x in reads if len(x)) == c["forward_layers"]
    assert c["fully_covered"] + c["partly_covered"] + c["not_covered"] == c["layers_with_miss"]


def test_candidate_split_sums_to_useful_rows(tmp_path):
    groups = _groups(tmp_path, n=1)
    s = l2_cap1_metrics.candidate_split(groups, [_ids(10, 20, 5, 30)])
    assert s == {"inside_top12": 1, "outside_top12": 2, "eligible_inside": 2, "eligible_outside": 2}
    assert s["inside_top12"] + s["outside_top12"] == int(groups[0].labels.miss[_ids(10, 20, 5, 30)].sum())


def test_gain_split_differences(tmp_path):
    groups = _groups(tmp_path, n=1)
    g = l2_cap1_metrics.gain_split(groups, cand_reads=[_ids(10, 20)], gate_reads=[_ids(10, 11)])
    assert g["inside"] == {"cand": 1, "gate": 2, "diff": -1}
    assert g["outside"] == {"cand": 1, "gate": 0, "diff": 1}
    with pytest.raises(ValueError):
        l2_cap1_metrics.gain_split(groups, cand_reads=[], gate_reads=[_ids()])


def test_dspark_json_counts_shared_once_and_reads_width_keys(tmp_path):
    groups = _groups(tmp_path, n=1)
    cands = [_cands(groups[0], {10: 0.95, 20: 0.8, 11: 0.75})]
    rep = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.7, widths=True)
    d = l2_cap1_metrics.dspark_json(r1_metrics.report_to_json(rep))
    assert sum(v["eligible_all"] for v in d["rowsets"].values()) == rep.cells["pooled"].eligible_all
    assert d["rowsets"]["shared"]["eligible_all"] == 1 and "0" in d["positions"]
    assert d["widths"] == {"2": {"useful": 2, "eligible_all": 4, "recall": pytest.approx(0.5)}}


def test_dspark_json_without_widths_is_empty(tmp_path):
    groups = _groups(tmp_path, n=1)
    rep = r1_metrics.policy_report(groups, [_cands(groups[0], {10: 0.9})], scorer="x", cap=2, threshold=0.5)
    assert l2_cap1_metrics.dspark_json(r1_metrics.report_to_json(rep))["widths"] == {}
