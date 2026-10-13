"""Tests for l2_cap1_sanity: exact reproduction of spike points."""
from __future__ import annotations

import copy
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2_cap1_sanity as S  # noqa: E402


def _point(i: int) -> dict:
    return {"reads": 100 + i, "useful": 40 + i, "eligible": 90 + i, "recall": 0.4}


def _fixture():
    names = ("gate", "L2_T0", "L2_T3")
    pts = {s: {n: _point(i + j) for j, n in enumerate(names)} for i, s in enumerate(("dev", "calib"))}
    spike = {"points": copy.deepcopy(pts)}
    spike["points"]["dev"]["L2_T1"] = _point(7)
    evals = {s: {n: {2: {"point": copy.deepcopy(p)}} for n, p in d.items()} for s, d in pts.items()}
    return evals, spike


def test_match_passes():
    evals, spike = _fixture()
    out = S.check_spike(evals, spike=spike)
    assert out["status"] == "reproduced"
    assert out["label"] == "observed offline"
    assert len(out["checked"]) == 6


def test_one_off_useful_raises_and_names_it():
    evals, spike = _fixture()
    evals["calib"]["L2_T3"][2]["point"]["useful"] += 1
    with pytest.raises(S.SanityError, match=r"calib/L2_T3\.useful"):
        S.check_spike(evals, spike=spike)


def test_missing_scorer_raises():
    evals, spike = _fixture()
    del evals["dev"]["L2_T0"]
    with pytest.raises(S.SanityError, match=r"dev/L2_T0"):
        S.check_spike(evals, spike=spike)


def test_extra_spike_arms_ignored():
    evals, spike = _fixture()
    spike["points"]["calib"]["L2_T1"] = _point(9)
    assert S.check_spike(evals, spike=spike)["status"] == "reproduced"
