"""r1_metrics.policy_report(widths=True): live-block-width cells and histograms, with default output unchanged.
Synthetic groups are built as test_r1_metrics.py builds them. CPU only."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import r1_groups  # noqa: E402
import r1_metrics  # noqa: E402
import r1_policy  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

E = r1_groups.NUM_EXPERTS
GPU = int(LaneKind.MISS_GPU)
_ROWS = [[10, 11, 12, -1, -1, -1], [10, 20, 30, -1, -1, -1], [11, 12, 13, -1, -1, -1]]
_SCORES = {10: 0.95, 11: 0.75, 12: 0.7, 13: 0.6, 20: 0.8, 5: 0.5}


def _group(*, width, seq, session="s1"):
    mask = np.zeros(6, np.uint64)
    kinds = np.zeros((width, 6), np.uint8)
    ids = np.array(_ROWS[:width], np.int16)
    kinds[ids >= 0] = GPU
    return b1_rows.Group(
        session=session, split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=3, source_stratum="a", gate_has_bias=True,
        target_row=1, target_layer=4, target_stratum="b", tokens=np.arange(width, dtype=np.int16),
        x_offset=width * seq, source_ids=np.arange(6 * width, dtype=np.int16).reshape(width, 6),
        source_weights=np.full((width, 6), 0.25, np.float32), target_ids=ids, target_kinds=kinds,
        ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (width, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(width, 0))


def _read(tmp_path):
    by_session = {"s1": [_group(width=1, seq=0), _group(width=3, seq=1)], "s2": [_group(width=3, seq=2, session="s2")]}
    for session, groups in by_session.items():
        b1_rows.write_groups(tmp_path, split="dev", horizon=1, session=session, groups=groups)
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def _cands(r):
    s = np.full(E, -np.inf, np.float32)
    for e, v in _SCORES.items():
        s[e] = v
    return r1_policy.candidates(s, skip=~r.labels.available)


def _reports(tmp_path):
    groups = _read(tmp_path)
    cands = [_cands(r) for r in groups]
    return groups, cands


def test_default_report_json_is_unchanged(tmp_path):
    groups, cands = _reports(tmp_path)
    base = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.62)
    off = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.62, widths=False)
    assert r1_metrics.report_to_json(base) == r1_metrics.report_to_json(off)
    assert not any(k.startswith("width:") for k in base.cells) and not any(k.startswith("width:") for k in base.hist)


def test_width_cells_partition_the_pooled_cell(tmp_path):
    groups, cands = _reports(tmp_path)
    base = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.62)
    rep = r1_metrics.policy_report(groups, cands, scorer="x", cap=2, threshold=0.62, widths=True)
    assert {k for k in rep.cells if k.startswith("width:")} == {"width:1", "width:3"}
    assert {k for k in rep.hist if k.startswith("width:")} == {"width:1", "width:3"}
    assert rep.cells["pooled"] == base.cells["pooled"]
    assert rep.cells["width:1"].forwards == 1 and rep.cells["width:3"].forwards == 2
    for field in ("forwards", "eligible_all", "eligible_addr", "read", "useful", "wasted", "cap_dropped_useful"):
        total = sum(getattr(c, field) for k, c in rep.cells.items() if k.startswith("width:"))
        assert total == getattr(rep.cells["pooled"], field), field
