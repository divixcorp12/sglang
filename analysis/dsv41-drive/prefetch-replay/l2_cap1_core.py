"""L+2 cap-1 evaluation core: every cap-1 number of every scorer on one shared, covered group list per split.

Per split and scorer: the cap-1 sweep curve, the budget thresholds (dev: best recall within budget; calib: lowest
feasible threshold on calib's own sweep, D5), and per budget the pooled point, abstention and coverage counts, the
DSpark breakdown and the gate-top-12 split. Per learned scorer and budget: a paired session bootstrap against the gate
at the same budget and at the gate's matched achieved reads, with the D3 gain split. Recall and useful reads are
compared; precision is reported but never compared alone. Uncovered groups are never passed here.

Inputs: {split: list[R1Group]} (covered groups) and {split: {scorer: list[Candidates]}} aligned with them.
Output: (result JSON dict, reads {split: {scorer: {budget: list[np.ndarray]}}} for the timing replay).

Usage: result, reads = l2_cap1_core.evaluate_cap1(groups, cands, gate="gate_cap1")
"""

from __future__ import annotations

import math
import os
import sys

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_cap1_metrics  # noqa: E402
import l2_cap1_policy  # noqa: E402
import r1_metrics  # noqa: E402
import r1_policy  # noqa: E402
from l2_cap1_policy import BUDGETS, CAP  # noqa: E402
from r1_groups import R1Group  # noqa: E402
from r1_metrics import PolicyReport  # noqa: E402
from r1_policy import Candidates, SweepPoint  # noqa: E402

_THRESHOLD_RULES = {"dev": l2_cap1_policy.dev_thresholds, "calib": l2_cap1_policy.calib_thresholds}


class _BudgetEval(msgspec.Struct, frozen=True):
    threshold: float
    report: PolicyReport
    reads: list[np.ndarray]
    summary: dict


class _ScorerEval(msgspec.Struct, frozen=True):
    points: list[SweepPoint]
    curve: list[dict]
    budgets: dict[float, _BudgetEval]


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def assert_identical(groups: dict[str, list[R1Group]], cands: dict[str, dict[str, list[Candidates]]]) -> None:
    """Every scorer of every split has exactly one candidate list per group of that split's one shared list."""
    if set(groups) != set(cands):
        raise ValueError(f"splits differ: groups {sorted(groups)} vs candidates {sorted(cands)}")
    for split, gs in groups.items():
        if not gs:
            raise ValueError(f"{split}: no groups")
        if not cands[split]:
            raise ValueError(f"{split}: no scorers")
        for name, cs in cands[split].items():
            if len(cs) != len(gs):
                raise ValueError(f"{split}/{name}: {len(cs)} candidate lists for {len(gs)} groups")


def _point_json(*, budget: float | None, threshold: float, rep: PolicyReport) -> dict:
    c = rep.cells["pooled"]
    return {"budget": budget, "threshold": _finite(threshold), "reads": c.read, "forwards": c.forwards,
            "mean_reads": c.mean_reads, "useful": c.useful, "wasted": c.read - c.useful,
            "eligible": c.eligible_all, "recall": c.recall, "precision": c.precision}


def _at_threshold(groups: list[R1Group], cs: list[Candidates], *, scorer: str,
                  threshold: float) -> tuple[PolicyReport, list[np.ndarray]]:
    rep = r1_metrics.policy_report(groups, cs, scorer=scorer, cap=CAP, threshold=threshold, widths=True)
    return rep, [r1_policy.select(c, threshold=threshold, cap=CAP) for c in cs]


def _budget_eval(groups: list[R1Group], cs: list[Candidates], *, scorer: str, budget: float,
                 threshold: float) -> _BudgetEval:
    rep, reads = _at_threshold(groups, cs, scorer=scorer, threshold=threshold)
    summary = {"point": _point_json(budget=budget, threshold=threshold, rep=rep),
               "coverage": l2_cap1_metrics.coverage_counts(groups, reads),
               "dspark": l2_cap1_metrics.dspark_json(r1_metrics.report_to_json(rep)),
               "candidate_split": l2_cap1_metrics.candidate_split(groups, reads)}
    return _BudgetEval(threshold=threshold, report=rep, reads=reads, summary=summary)


def _scorer_eval(groups: list[R1Group], cs: list[Candidates], *, split: str, scorer: str,
                 budgets: tuple[float, ...]) -> _ScorerEval:
    if split not in _THRESHOLD_RULES:
        raise ValueError(f"no budget rule for split {split!r}; expected one of {sorted(_THRESHOLD_RULES)}")
    points = r1_policy.sweep(cs, [r.labels for r in groups], cap=CAP)
    thresholds = _THRESHOLD_RULES[split](points, budgets=budgets)
    return _ScorerEval(points=points, curve=l2_cap1_policy.curve_json(points),
                       budgets={b: _budget_eval(groups, cs, scorer=scorer, budget=b, threshold=thresholds[b])
                                for b in budgets})


def _bootstrap(cand: PolicyReport, ctrl: PolicyReport) -> dict:
    return r1_metrics.session_bootstrap(cand, ctrl, n=r1_metrics.BOOTSTRAP_RESAMPLES, seed=0, denominator="all",
                                        useful_ci=True)


def _matched(groups: list[R1Group], gate_cs: list[Candidates], *, gate: str, gate_points: list[SweepPoint],
             learned: _BudgetEval) -> dict:
    """The gate at its curve point nearest the learned head's achieved reads, and the paired comparison there."""
    target = learned.summary["point"]["reads"]
    m = l2_cap1_policy.matched_point(gate_points, target_reads=target)
    gate_rep, gate_reads = _at_threshold(groups, gate_cs, scorer=gate, threshold=m["threshold"])
    return {"target_reads": m["target_reads"], "achieved_reads": m["achieved_reads"],
            "learned_reads": target, "gate_reads": gate_rep.cells["pooled"].read,
            "gate_point": _point_json(budget=None, threshold=m["threshold"], rep=gate_rep),
            "bootstrap": _bootstrap(learned.report, gate_rep),
            "gain_split": l2_cap1_metrics.gain_split(groups, cand_reads=learned.reads, gate_reads=gate_reads)}


def _comparisons(groups: list[R1Group], gate_cs: list[Candidates], *, gate: str, gate_eval: _ScorerEval,
                 learned: _ScorerEval) -> dict[float, dict]:
    return {b: {"same_budget": _bootstrap(e.report, gate_eval.budgets[b].report),
                "matched": _matched(groups, gate_cs, gate=gate, gate_points=gate_eval.points, learned=e)}
            for b, e in learned.budgets.items()}


def _scorer_json(ev: _ScorerEval, *, comparisons: dict[float, dict] | None) -> dict:
    return {"curve": ev.curve,
            "budgets": {str(b): {**e.summary, **(comparisons[b] if comparisons is not None else {})}
                        for b, e in ev.budgets.items()}}


def _evaluate_split(groups: list[R1Group], cands: dict[str, list[Candidates]], *, split: str, gate: str,
                    budgets: tuple[float, ...]) -> tuple[dict, dict]:
    if gate not in cands:
        raise ValueError(f"{split}: gate scorer {gate!r} missing from {sorted(cands)}")
    evals = {name: _scorer_eval(groups, cs, split=split, scorer=name, budgets=budgets) for name, cs in cands.items()}
    result = {name: _scorer_json(ev, comparisons=None if name == gate else _comparisons(
        groups, cands[gate], gate=gate, gate_eval=evals[gate], learned=ev)) for name, ev in evals.items()}
    reads = {name: {b: e.reads for b, e in ev.budgets.items()} for name, ev in evals.items()}
    return result, reads


def evaluate_cap1(groups: dict[str, list[R1Group]], cands: dict[str, dict[str, list[Candidates]]], *,
                  gate: str = "gate_cap1", budgets: tuple[float, ...] = BUDGETS) -> tuple[dict, dict]:
    """All cap-1 numbers per split and scorer on identical groups; reads per budget are kept out of the JSON."""
    assert_identical(groups, cands)
    result: dict[str, dict] = {}
    reads: dict[str, dict] = {}
    for split, gs in groups.items():
        result[split], reads[split] = _evaluate_split(gs, cands[split], split=split, gate=gate, budgets=budgets)
    return result, reads
