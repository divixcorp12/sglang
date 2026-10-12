"""b1_rows.py on synthetic groups: shard round-trips, eligibility counted once per expert, never-candidates (D10),
the RAM snapshot bits and the locked-test refusal. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

HIDDEN = 4
MISS_GPU, MISS_CPU, HIT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY)


def _group(*, seq=0, source_row=0, n=2, x_offset=0, session="s1", split="dev", horizon=1, has_bias=True):
    target_ids = np.array([[7, 3, 9, 1, 2, 0], [7, 300, 3, 4, 5, 6]][:n], dtype=np.int16)
    # 7 is a miss on both tokens, 300 a CPU miss, 3 a hit on one token and a miss on the other.
    kinds = np.array([[MISS_GPU, HIT, 0, 0, 0, 0], [MISS_CPU, MISS_CPU, MISS_GPU, 0, 0, 0]][:n], dtype=np.uint8)
    gate_ids = np.tile(np.arange(12, dtype=np.int16), (n, 1)) if has_bias else np.full((n, 12), -1, np.int16)
    gate_scores = np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0)
    if not has_bias:
        gate_scores = np.full((n, 12), -np.inf, np.float32)
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64((1 << 3) | (1 << 63))
    mask[300 // 64] |= np.uint64(1 << (300 % 64))
    return b1_rows.Group(
        session=session, split=split, block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=True,
        horizon=horizon, source_row=source_row, source_layer=source_row + 3, source_stratum="a",
        gate_has_bias=has_bias, target_row=source_row + horizon, target_layer=source_row + 3 + horizon,
        target_stratum="b", tokens=np.arange(n, dtype=np.int16), x_offset=x_offset,
        source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=target_ids, target_kinds=kinds,
        ram_mask=mask, hot=np.array([0, 1], np.int16), gate_ids=gate_ids, gate_scores=gate_scores)


def _assert_same(a, b):
    for name in a.__struct_fields__:
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, np.ndarray):
            assert va.dtype == vb.dtype and np.array_equal(va, vb), name
        else:
            assert type(va) is type(vb) and va == vb, name


def test_groups_round_trip_in_session_then_seq_order(tmp_path):
    s1 = [_group(seq=2, x_offset=4), _group(seq=1, source_row=1, n=1, x_offset=2), _group(seq=1, x_offset=0)]
    s0 = [_group(session="s0", seq=5, has_bias=False)]
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=s1)
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s0", groups=s0)
    got = list(b1_rows.read_groups(tmp_path, split="dev", horizon=1))
    for want, have in zip([s0[0], s1[2], s1[1], s1[0]], got, strict=True):
        _assert_same(want, have)
    only = list(b1_rows.read_groups(tmp_path, split="dev", horizon=1, sessions=["s1"]))
    assert [g.session for g in only] == ["s1"] * 3


def test_session_ids_with_slashes_round_trip_as_one_flat_shard_each(tmp_path):
    """Real B0 session ids look like 'cfq-train-Double_AAL/2014/page_89.pdf'."""
    sid = "cfq-train-Double_AAL/2014/page_89.pdf-1"
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session=sid, groups=[_group(session=sid)])
    b1_rows.write_x(tmp_path, split="dev", session=sid, x=np.zeros((2, HIDDEN), np.uint16))
    assert [g.session for g in b1_rows.read_groups(tmp_path, split="dev", horizon=1)] == [sid]
    assert b1_rows.read_x(tmp_path, split="dev", session=sid).shape == (2, HIDDEN)
    assert all(p.parent.name in ("x", "h1") for p in tmp_path.rglob("*") if p.is_file())


def test_an_empty_session_shard_round_trips(tmp_path):
    b1_rows.write_groups(tmp_path, split="train", horizon=4, session="s9", groups=[])
    assert list(b1_rows.read_groups(tmp_path, split="train", horizon=4)) == []


def test_a_group_from_another_shard_is_refused(tmp_path):
    with pytest.raises(ValueError):
        b1_rows.write_groups(tmp_path, split="dev", horizon=2, session="s1", groups=[_group()])


def test_x_round_trips_through_a_memmap_and_decodes_bf16(tmp_path):
    values = np.arange(6 * HIDDEN, dtype=np.float32).reshape(6, HIDDEN) - 3.5
    words = (values.view(np.uint32) >> 16).astype(np.uint16)
    b1_rows.write_x(tmp_path, split="dev", session="s1", x=words)
    xmap = b1_rows.read_x(tmp_path, split="dev", session="s1")
    assert isinstance(xmap, np.memmap) and xmap.shape == (6, HIDDEN)
    g = _group(n=2, x_offset=3)
    np.testing.assert_array_equal(b1_rows.x_rows(xmap, g), values[3:5])


def test_eligibility_counts_each_miss_expert_once_and_never_a_hit_only_one():
    g = _group()
    np.testing.assert_array_equal(b1_rows.eligible(g), [3, 7, 300])
    assert b1_rows.eligible(g).dtype == np.int16
    one = _group(n=1)
    np.testing.assert_array_equal(b1_rows.eligible(one), [7])


def test_never_candidates_are_eligible_experts_outside_every_tokens_gate_top12():
    g = _group()
    np.testing.assert_array_equal(b1_rows.never_candidates(g), [300])
    np.testing.assert_array_equal(b1_rows.never_candidates(_group(has_bias=False)), [3, 7, 300])


def test_in_ram_reads_the_384_bit_snapshot():
    g = _group()
    np.testing.assert_array_equal(b1_rows.in_ram(g, np.array([3, 7, 63, 300, 383])),
                                  [True, False, True, True, False])


def test_locked_test_is_refused_without_a_frozen_selection(tmp_path):
    b1_rows.write_groups(tmp_path, split="locked_test", horizon=1, session="s1",
                         groups=[_group(split="locked_test")])
    b1_rows.write_x(tmp_path, split="locked_test", session="s1", x=np.zeros((2, HIDDEN), np.uint16))
    with pytest.raises(PermissionError):
        b1_rows.read_groups(tmp_path, split="locked_test", horizon=1)
    with pytest.raises(PermissionError):
        b1_rows.read_x(tmp_path, split="locked_test", session="s1")
    sha = "0" * 64
    assert len(list(b1_rows.read_groups(tmp_path, split="locked_test", horizon=1, frozen_sha256=sha))) == 1
    assert b1_rows.read_x(tmp_path, split="locked_test", session="s1", frozen_sha256=sha).shape == (2, HIDDEN)
