"""r1_metrics.session_bootstrap with useful_ci: default output unchanged, recall keys unchanged when the useful-read
CI is requested, interval ordering, zero-width CI on identical reports, and the default n and seed. CPU only."""

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_metrics  # noqa: E402
from r1_metrics import PolicyCell, PolicyReport  # noqa: E402

_RECALL_KEYS = ("diff", "lo", "hi", "ratio", "precision_diff", "cand_recall", "ctrl_recall", "sessions", "n", "seed",
                "denominator")
_USEFUL_KEYS = ("useful_diff", "useful_lo", "useful_hi", "cand_useful", "ctrl_useful", "cand_reads", "ctrl_reads")


def _rep(*, sessions: dict) -> PolicyReport:
    cells = {f"session:{s}": PolicyCell(useful=u, eligible_all=e, eligible_addr=e, read=r)
             for s, (u, e, r) in sessions.items()}
    return PolicyReport(scorer="x", split="dev", cap=1, threshold=0.5, cells=cells, hist={}, positions={},
                        rowsets={}, reliability=[], forward_totals={})


_CAND = {"a": (5, 10, 8), "b": (2, 6, 4), "c": (7, 9, 9), "d": (0, 4, 1), "e": (3, 5, 5)}
_CTRL = {"a": (3, 10, 6), "b": (2, 6, 5), "c": (4, 9, 7), "d": (1, 4, 2), "e": (1, 5, 3)}


def test_default_has_only_the_original_keys_and_is_deterministic():
    cand, ctrl = _rep(sessions=_CAND), _rep(sessions=_CTRL)
    out = r1_metrics.session_bootstrap(cand, ctrl, n=500, seed=3)
    assert tuple(out) == _RECALL_KEYS
    assert out == r1_metrics.session_bootstrap(cand, ctrl, n=500, seed=3)


def test_useful_ci_keeps_recall_keys_and_orders_the_interval():
    cand, ctrl = _rep(sessions=_CAND), _rep(sessions=_CTRL)
    base = r1_metrics.session_bootstrap(cand, ctrl, n=500, seed=3)
    out = r1_metrics.session_bootstrap(cand, ctrl, n=500, seed=3, useful_ci=True)
    assert {k: out[k] for k in _RECALL_KEYS} == base
    assert set(out) == set(_RECALL_KEYS) | set(_USEFUL_KEYS)
    assert out["useful_lo"] <= out["useful_diff"] <= out["useful_hi"]
    assert out["useful_diff"] == out["cand_useful"] - out["ctrl_useful"] == 17 - 11
    assert (out["cand_reads"], out["ctrl_reads"]) == (27, 23)


def test_identical_reports_give_a_zero_width_useful_ci_at_zero():
    cand, ctrl = _rep(sessions=_CAND), _rep(sessions=_CAND)
    out = r1_metrics.session_bootstrap(cand, ctrl, n=200, seed=0, useful_ci=True)
    assert out["useful_diff"] == out["useful_lo"] == out["useful_hi"] == 0.0


def test_defaults_are_ten_thousand_resamples_and_seed_zero():
    params = inspect.signature(r1_metrics.session_bootstrap).parameters
    assert params["n"].default == 10_000
    assert params["seed"].default == 0
    assert params["useful_ci"].default is False
