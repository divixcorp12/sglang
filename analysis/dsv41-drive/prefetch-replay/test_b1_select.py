"""b1_select.py on hand-built Tables: the grid, the dev pick, calib-only cuts, the frozen file and the locked-test guard,
bootstrap determinism, the learning-curve rule and the verdict at and around each threshold. CPU only."""

import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_select  # noqa: E402
from b1_score import Cell, Table  # noqa: E402


def _table(*, split="dev", name="s", cells=None):
    return Table(scorer=name, split=split, horizon=1, cells=cells or {})


def _precision_table(precision, *, split="dev", name="s"):
    return _table(split=split, name=name,
                  cells={"reads=1": {"pooled": Cell(forwards=10, eligible=20, read=100, useful=int(precision * 100))}})


def _selection():
    h = b1_select.HorizonSelection(variant="E5", config={"variant": "E5", "pca_rank": 256, "hidden": 512,
                                                          "hard_neg_weight": 2.0},
                                   cuts={"reads=1": 0.5, "reads=2": 0.3, "reads=3": 0.2},
                                   model_path="/m/h1.pt", model_sha256="ab" * 32)
    return b1_select.Selection(horizons={1: h}, pop_path="/m/pop.json", pop_sha256="cd" * 32,
                               created="2026-10-12T00:00:00Z")


def test_grid_sizes_and_variants():
    e2, e5 = b1_select.grid(variant="E2"), b1_select.grid(variant="E5")
    assert len(e2) == 4 and len(e5) == 12
    assert {(c["pca_rank"], c["hidden"]) for c in e2} == {(128, 256), (128, 512), (256, 256), (256, 512)}
    assert {c["hard_neg_weight"] for c in e5} == {1.0, 2.0, 4.0}
    assert all(c["variant"] == "E5" for c in e5)
    with pytest.raises(ValueError):
        b1_select.grid(variant="E3")


def test_pick_best_by_dev_precision_and_refuses_other_splits():
    tables = {"a": _precision_table(0.30), "b": _precision_table(0.45), "c": _precision_table(0.45)}
    assert b1_select.pick_best(tables) == "b"
    with pytest.raises(ValueError):
        b1_select.pick_best({"a": _precision_table(0.3, split="locked_test")})


def test_calibrate_cuts_matches_mean_reads_and_refuses_non_calib():
    # Mean reads per forward: cut 0.9 -> 0.8, 0.6 -> 1.1, 0.4 -> 2.0, 0.2 -> 3.2, 0.1 -> 5.0.
    sweep = {f"cut={c!r}": {"pooled": Cell(forwards=10, read=r)}
             for c, r in [(0.9, 8), (0.6, 11), (0.4, 20), (0.2, 32), (0.1, 50)]}
    cuts = b1_select.calibrate_cuts(_table(split="calib", cells=sweep))
    assert cuts == {"reads=1": 0.6, "reads=2": 0.4, "reads=3": 0.2}
    with pytest.raises(ValueError):
        b1_select.calibrate_cuts(_table(split="dev", cells=sweep))
    with pytest.raises(ValueError):
        b1_select.calibrate_cuts(_table(split="calib", cells={"reads=1": {"pooled": Cell(forwards=1, read=1)}}))


def test_freeze_round_trip_sha_and_no_overwrite(tmp_path):
    path = tmp_path / "run" / "selection.json"
    sha = b1_select.freeze_selection(str(path), _selection())
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    sel, sha2 = b1_select.load_frozen(str(path))
    assert sha2 == sha and sel == _selection()
    with pytest.raises(FileExistsError):
        b1_select.freeze_selection(str(path), _selection())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_require_frozen_guards_locked_test(tmp_path):
    path = tmp_path / "selection.json"
    with pytest.raises(PermissionError):
        b1_select.require_frozen("locked_test", str(path))
    with pytest.raises(PermissionError):
        b1_select.require_frozen("locked_test", None)
    assert b1_select.require_frozen("dev", str(path)) is None
    assert b1_select.require_frozen("calib", None) is None
    sha = b1_select.freeze_selection(str(path), _selection())
    assert b1_select.require_frozen("locked_test", str(path)) == sha


def test_require_frozen_refuses_a_file_that_is_not_a_selection(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text("{}")
    with pytest.raises(PermissionError):
        b1_select.require_frozen("locked_test", str(path))


def test_paired_diffs_by_session():
    head = _table(cells={"reads=1": {"session:a": Cell(read=10, useful=6), "session:b": Cell(read=10, useful=4),
                                     "pooled": Cell(read=20, useful=10)}})
    gate = _table(cells={"reads=1": {"session:a": Cell(read=10, useful=5), "session:b": Cell(read=10, useful=4)}})
    diffs = b1_select.paired_diffs(head, gate)
    assert diffs.keys() == {"a", "b"}
    assert diffs["a"] == pytest.approx(0.1) and diffs["b"] == pytest.approx(0.0)


def test_session_bootstrap_is_seeded_and_brackets_median():
    diffs = {f"s{i:02d}": 0.01 * i - 0.03 for i in range(15)}
    a = b1_select.session_bootstrap(diffs, n=2000, seed=7)
    b = b1_select.session_bootstrap(dict(reversed(list(diffs.items()))), n=2000, seed=7)
    assert a == b
    median, lo, hi = a
    assert median == pytest.approx(0.04)
    assert lo <= median <= hi and lo < hi
    assert b1_select.session_bootstrap(diffs, n=2000, seed=8) != a
    assert b1_select.session_bootstrap({"s": 0.2}, n=100, seed=0) == (0.2, 0.2, 0.2)
    with pytest.raises(ValueError):
        b1_select.session_bootstrap({})


@pytest.mark.parametrize("p15,p30,p60,expected", [
    # Binary-exact values, so the at-threshold case is not decided by rounding.
    (0.25, 0.5, 0.75, True),
    (0.25, 0.5, 0.625, False),  # exactly half the earlier step is not more than half
    (0.25, 0.5, 0.6251, True),
    (0.25, 0.5, 0.5, False),
    (0.5, 0.5, 0.515625, True),
    (0.5, 0.5, 0.5, False),
])
def test_still_improving(p15, p30, p60, expected):
    assert b1_select.still_improving(p15, p30, p60) is expected


def _verdict(**kw):
    base = dict(gate_floor_precision_r1=0.50, head_precision_r1=0.50, gate_floor_recall_at_p=0.40,
                head_recall_at_p=0.40, ci_lo=0.01, cost_ms_per_layer=0.05)
    return b1_select.verdict(**{**base, **kw})


@pytest.mark.parametrize("head_p,expected", [(0.65, True), (0.66, True), (0.649, False), (0.50, False)])
def test_verdict_precision_threshold(head_p, expected):
    v = _verdict(head_precision_r1=head_p)
    assert v["arms"]["precision"] is expected and v["advance"] is expected


@pytest.mark.parametrize("head_r,expected", [(0.50, True), (0.51, True), (0.499, False), (None, False)])
def test_verdict_recall_threshold(head_r, expected):
    v = _verdict(head_recall_at_p=head_r)
    assert v["arms"]["recall"] is expected and v["advance"] is expected


def test_verdict_recall_needs_a_gate_recall():
    assert _verdict(gate_floor_recall_at_p=None, head_recall_at_p=0.9)["arms"]["recall"] is False


@pytest.mark.parametrize("ci_lo,expected", [(0.001, True), (0.0, False), (-0.01, False)])
def test_verdict_ci_threshold(ci_lo, expected):
    v = _verdict(head_precision_r1=0.70, ci_lo=ci_lo)
    assert v["arms"]["ci"] is expected and v["advance"] is expected


@pytest.mark.parametrize("cost,expected", [(0.1, True), (0.0999, True), (0.1001, False)])
def test_verdict_cost_threshold(cost, expected):
    v = _verdict(head_precision_r1=0.70, cost_ms_per_layer=cost)
    assert v["arms"]["cost"] is expected and v["advance"] is expected


def test_verdict_reasons_name_the_ceiling_on_failure():
    fail, ok = _verdict(), _verdict(head_precision_r1=0.70)
    assert fail["advance"] is False and any(b1_select.CEILING in r for r in fail["reasons"])
    assert ok["advance"] is True and not any(b1_select.CEILING in r for r in ok["reasons"])
