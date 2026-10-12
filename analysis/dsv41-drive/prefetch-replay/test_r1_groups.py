"""r1_groups.py on synthetic groups: the locked-test refusal, D4 labels and both denominators, the D14 exclusion
manifest and the skip mask shared with b1_score. CPU only."""

import json
import sys
from pathlib import Path

import msgspec
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import b1_score  # noqa: E402
import r1_groups  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

HIDDEN = 4
MISS_GPU, MISS_CPU, HIT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY)


def _group(*, seq=0, source_row=0, x_offset=0, session="s1", split="dev", horizon=1, block="b000"):
    n = 3
    target_ids = np.array([[7, 3, 9, 1, 2, 0], [7, 300, 3, 4, 5, 6], [10, 11, 12, 13, 14, -1]], dtype=np.int16)
    # 7: GPU miss on token 0, CPU miss on token 1; 300 CPU miss; 3 a hit on token 0 and a GPU miss on token 1;
    # 10 a GPU miss on token 2 only.
    kinds = np.array([[MISS_GPU, HIT, 0, 0, 0, 0], [MISS_CPU, MISS_CPU, MISS_GPU, 0, 0, 0],
                      [MISS_GPU, HIT, HIT, HIT, HIT, 0]], dtype=np.uint8)
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64((1 << 3) | (1 << 63))
    mask[300 // 64] |= np.uint64(1 << (300 % 64))
    return b1_rows.Group(
        session=session, split=split, block=block, seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=horizon, source_row=source_row, source_layer=source_row + 3, source_stratum="a",
        gate_has_bias=True, target_row=source_row + horizon, target_layer=source_row + 3 + horizon,
        target_stratum="b", tokens=np.array([0, 1, 2], np.int16), x_offset=x_offset,
        source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=target_ids, target_kinds=kinds,
        ram_mask=mask, hot=np.array([0, 1], np.int16), gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


def _write(tmp_path, groups, *, split="dev", session="s1", horizon=1):
    b1_rows.write_groups(tmp_path, split=split, horizon=horizon, session=session, groups=groups)


def _manifest(tmp_path, cells):
    path = tmp_path / "exclusions.json"
    rows = [{"block": b, "seq": s, "layer_id": layer, "token": t, "reason": "capture mismatch",
             "source_groups": [], "target_groups": []} for b, s, layer, t in cells]
    path.write_text(json.dumps({"cells": rows}))
    return path


def test_locked_test_is_refused_at_the_call_with_no_override(tmp_path):
    _write(tmp_path, [_group(split="locked_test")], split="locked_test")
    with pytest.raises(PermissionError):
        r1_groups.read_r1_groups(tmp_path, split="locked_test", horizon=1, exclusions=r1_groups.Exclusions.empty())
    with pytest.raises(ValueError):
        r1_groups.read_r1_groups(tmp_path, split="test", horizon=1, exclusions=r1_groups.Exclusions.empty())
    assert r1_groups.R1_SPLITS == ("train", "dev", "calib")


def test_labels_are_the_miss_union_with_both_denominators(tmp_path):
    _write(tmp_path, [_group()])
    (r,) = r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty())
    lab = r.labels
    np.testing.assert_array_equal(np.nonzero(lab.miss_gpu)[0], [3, 7, 10])
    np.testing.assert_array_equal(np.nonzero(lab.miss_cpu)[0], [7, 300])
    np.testing.assert_array_equal(np.nonzero(lab.miss)[0], [3, 7, 10, 300])
    np.testing.assert_array_equal(lab.eligible_all, b1_rows.eligible(r.g))
    assert lab.eligible_all.dtype == np.int16 and lab.eligible_addr.dtype == np.int16
    # 3 and 300 are in the RAM snapshot; 0 and 1 are VRAM-hot.
    np.testing.assert_array_equal(lab.eligible_addr, [7, 10])
    assert not lab.available[[0, 1, 3, 63, 300]].any() and lab.available[[2, 7, 10, 383]].all()
    assert lab.route.shape == (3, 384) and lab.route.sum() == 6 + 6 + 5
    assert lab.route[2, 10] and not lab.route[2, 0] and lab.route[0, 0]
    np.testing.assert_array_equal(r.positions, [0, 1, 2])
    assert r.positions.dtype == np.int64
    assert r.key == r1_groups.GroupKey(session="s1", block="b000", seq=0, source_layer=3, target_layer=4, horizon=1)


def test_skip_mask_matches_b1_score():
    g = _group()
    np.testing.assert_array_equal(r1_groups.skip_mask(g), b1_score._skip_mask(g, num_experts=384))
    assert r1_groups.skip_mask(g).shape == (384,)


def test_exclusions_drop_the_token_from_source_and_target_groups_only(tmp_path):
    # Layer 4 is the target of the row-0 group and the source of the row-1 group; the row-5 group never touches it.
    groups = [_group(seq=7, source_row=0, x_offset=0), _group(seq=7, source_row=1, x_offset=3),
              _group(seq=7, source_row=5, x_offset=6), _group(seq=8, source_row=0, x_offset=9)]
    _write(tmp_path, groups)
    exclusions = r1_groups.load_exclusions(_manifest(tmp_path, [("b000", 7, 4, 1)]))
    assert exclusions.cells == frozenset({("b000", 7, 4, 1)})
    got = list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=exclusions))
    assert [(r.key.seq, r.key.source_layer) for r in got] == [(7, 3), (7, 4), (7, 8), (8, 3)]
    for r, want_pos, want_x in zip(got, ([0, 2], [0, 2], [0, 1, 2], [0, 1, 2]), ([0, 2], [3, 5], [6, 7, 8],
                                                                                [9, 10, 11]), strict=True):
        np.testing.assert_array_equal(r.positions, want_pos)
        np.testing.assert_array_equal(r.x_index, want_x)
        assert len(r.g.tokens) == len(r.g.target_ids) == len(r.g.gate_ids) == r.labels.route.shape[0]
    # Token 1 carried the only CPU miss on 7 and 300 and the GPU miss on 3.
    np.testing.assert_array_equal(np.nonzero(got[0].labels.miss)[0], [7, 10])
    assert not got[0].labels.miss_cpu.any()


def test_a_group_with_every_token_excluded_is_dropped(tmp_path):
    _write(tmp_path, [_group(seq=1), _group(seq=2)])
    exclusions = r1_groups.Exclusions(cells=frozenset({("b000", 1, 3, t) for t in range(3)}))
    got = list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=exclusions))
    assert [r.key.seq for r in got] == [2]


def test_x_rows_follow_x_index_after_a_middle_token_is_dropped(tmp_path):
    values = np.arange(3 * HIDDEN, dtype=np.float32).reshape(3, HIDDEN) - 1.5
    b1_rows.write_x(tmp_path, split="dev", session="s1", x=(values.view(np.uint32) >> 16).astype(np.uint16))
    _write(tmp_path, [_group()])
    exclusions = r1_groups.Exclusions(cells=frozenset({("b000", 0, 3, 1)}))
    (r,) = r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=exclusions)
    xmap = b1_rows.read_x(tmp_path, split="dev", session="s1")
    np.testing.assert_array_equal(r1_groups.x_rows(xmap, r), values[[0, 2]])


def test_a_manifest_missing_a_field_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cells": [{"block": "b000", "seq": 1, "layer_id": 2}]}))
    with pytest.raises(msgspec.ValidationError, match="token"):
        r1_groups.load_exclusions(path)
