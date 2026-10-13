"""l2_cap1_scorers.py on synthetic groups and temp files: the D2 top-1 gate block, the coverage split, sha
verification and the checkpoint key map. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import l2_cap1_scorers  # noqa: E402
import r1_groups  # noqa: E402
import r1_stage  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

E = r1_groups.NUM_EXPERTS
_IDS = [[10, 11, 12, 0, -1, -1], [10, 20, 30, 2, -1, -1]]
_KINDS = [[int(LaneKind.MISS_GPU)] * 3 + [0] * 3] * 2
_TOKEN_IDS = [list(range(12)), [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11]]


def _group(*, seq=0, biased=True):
    n = 2
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 30)
    ids = np.array(_TOKEN_IDS, np.int16) if biased else np.full((n, 12), -1, np.int16)
    scores = np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0)
    if not biased:
        scores = np.full((n, 12), -np.inf, np.float32)
    return b1_rows.Group(
        session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=3, source_stratum="a", gate_has_bias=biased,
        target_row=1, target_layer=4, target_stratum="b", tokens=np.array([0, 1], np.int16),
        x_offset=n * seq, source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=np.array(_IDS, np.int16),
        target_kinds=np.array(_KINDS, np.uint8), ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=ids, gate_scores=scores)


def _read(tmp_path, groups):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=groups)
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


def test_gate_top1_keeps_each_tokens_best_available_expert_only(tmp_path):
    groups = _read(tmp_path, [_group()])
    block = l2_cap1_scorers.GateTop1Scorer().score_groups(groups)[0]
    assert block.dtype == np.float32 and block.shape == (E,)
    # Experts 0 and 1 are hot, so token 0 falls to expert 2 (col 2); token 1's best is expert 5 (col 0).
    assert block[2] == pytest.approx(3 / 11) and block[5] == pytest.approx(5 / 11)
    assert np.isfinite(block).sum() == 2
    assert l2_cap1_scorers.GateTop1Scorer().name == "gate_cap1"


def test_split_covered_moves_unbiased_groups_to_uncovered(tmp_path):
    groups = _read(tmp_path, [_group(seq=0), _group(seq=1, biased=False), _group(seq=2)])
    covered, uncovered = l2_cap1_scorers.split_covered(groups)
    assert [r.g.seq for r in covered] == [0, 2]
    assert [r.g.seq for r in uncovered] == [1]


def test_verify_ckpt_refuses_a_tampered_file(tmp_path):
    f = tmp_path / "c.pt"
    f.write_bytes(b"weights")
    sha = r1_stage.sha256_file(str(f))
    assert l2_cap1_scorers.verify_ckpt(str(f), sha=sha) == sha
    f.write_bytes(b"weights!")
    with pytest.raises(ValueError):
        l2_cap1_scorers.verify_ckpt(str(f), sha=sha)


def test_learned_ckpts_reads_the_documented_keys():
    l2 = {"arms": {
        "T0": {"ckpt_path": "/c/T0.pt", "ckpt_sha": "a" * 64,
               "extra_ckpts": {"best_cap1": {"path": "/c/T0_best.pt", "sha": "b" * 64, "epoch": 3}}},
        "T3": {"ckpt_path": "/c/T3.pt", "ckpt_sha": "c" * 64, "extra_ckpts": {}}}}
    assert l2_cap1_scorers.learned_ckpts(l2) == {
        "L2_T0": ("/c/T0.pt", "a" * 64), "L2_T3": ("/c/T3.pt", "c" * 64),
        "L2_T0_best_cap1": ("/c/T0_best.pt", "b" * 64)}
