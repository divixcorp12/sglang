"""wc_report: write-combined vs ordinary pinned slab, paired by cell shape, median over interleaved rounds (CPU)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import wc_report as report  # noqa: E402


def _cell(slab, method, gbs, rnd, grid=8, a=4, b=16):
    return {"kind": "cell", "method": method, "grid": grid, "a": a, "b": b, "slab": slab, "round": rnd, "gbs": gbs}


def test_pairs_take_the_median_over_rounds_and_ratio_against_the_ordinary_slab():
    records = [
        _cell("pinned", "sm_cv16", 12.30, 0), _cell("wc", "sm_cv16", 12.90, 0), _cell("hostalloc", "sm_cv16", 12.31, 0),
        _cell("pinned", "sm_cv16", 12.28, 1), _cell("wc", "sm_cv16", 13.10, 1), _cell("hostalloc", "sm_cv16", 12.29, 1),
        _cell("pinned", "sm_cv16", 12.32, 2), _cell("wc", "sm_cv16", 13.00, 2), _cell("hostalloc", "sm_cv16", 12.30, 2),
        _cell("pinned", "ce_each", 13.6, 0, grid=0, a=4, b=0), _cell("wc", "ce_each", 13.7, 0, grid=0, a=4, b=0),
    ]
    rows = {(r["method"], r["grid"], r["a"], r["b"]): r for r in report.pairs(records)}
    sm = rows[("sm_cv16", 8, 4, 16)]
    assert sm["gbs"] == {"pinned": 12.30, "hostalloc": 12.30, "wc": 13.00}
    assert sm["wc_ratio"] == pytest.approx(13.00 / 12.30)
    assert sm["hostalloc_ratio"] == pytest.approx(1.0)
    assert rows[("ce_each", 0, 4, 0)]["wc_ratio"] == pytest.approx(13.7 / 13.6)


def test_a_shape_missing_the_ordinary_slab_has_no_ratio():
    rows = report.pairs([_cell("wc", "sm_cv16", 13.0, 0)])
    assert rows[0]["wc_ratio"] is None
