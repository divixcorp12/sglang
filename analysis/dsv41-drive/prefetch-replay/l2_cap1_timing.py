"""Cap-1 timing replay for the L+2 comparison: each scorer's own measured cap-1 delay, a zero-delay sensitivity run
and the owner's frozen-candidate rule (C13, D6, N4).

Inputs: the loaded RunPlan, {split: [R1Group]}, the stamp windows, {split: {scorer: {1: {"reads": [...]}}}} from
r1_stage1.evaluate_split, and r1_cost --cap 1 result dicts for L2_T0, L2_T3 and L2_T0_best_cap1.
Output: {"measured": r1_stage1._timing output, "zero_delay": the same with every delay 0.0}, and the chosen budget.
Usage: delays = delays_from_costs(costs); out = replay_budgets(evals, groups=g, plan=plan, windows=w, delays=delays)
       pick = frozen_candidate(on_time_by_budget)
The RunPlan is only read; the view below carries the two fields r1_stage1._timing uses.
"""

from __future__ import annotations

import os
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_plan  # noqa: E402
import r1_stage1  # noqa: E402
from r1_groups import R1Group  # noqa: E402

LABEL = "replay estimate"
CAP = 1
GATE = "gate_cap1"
_COST_SCORERS = ("L2_T0", "L2_T3", "L2_T0_best_cap1")
_GATE_COST_FROM = "L2_T0"
RULE = ("the smallest mean-read budget whose dev nominal on-time useful rows are at least (1 - tolerance) x the "
        "1.00 budget's; otherwise 1.00")


class TimingView(msgspec.Struct, frozen=True):
    """The two RunPlan fields r1_stage1._timing reads, with the gate cap replaced."""

    timing: r1_plan.TimingAssumptions
    stage2_gate: r1_plan.Stage2Gate


def timing_view(plan: r1_plan.RunPlan, *, cap: int = CAP) -> TimingView:
    spec = plan.stage2_gate
    gate = r1_plan.Stage2Gate(split=spec.split, cap=cap, mean_reads=spec.mean_reads)
    return TimingView(timing=plan.timing, stage2_gate=gate)


def delays_from_costs(costs: dict[str, dict]) -> dict[str, float]:
    """Per-scorer delay in ms: a learned head's complete bf16 median, the cap-1 gate's gate-only time."""
    for name in _COST_SCORERS:
        if name not in costs:
            raise ValueError(f"no cap-1 cost for scorer {name!r}")
        if costs[name].get("cap") != CAP:
            raise ValueError(f"cost for {name!r} is cap {costs[name].get('cap')!r}, not {CAP}")
    delays = {name: float(costs[name]["gate_ms"]) for name in _COST_SCORERS}
    delays[GATE] = float(costs[_GATE_COST_FROM]["gate_only_ms"])
    return delays


def replay_budgets(evals: dict[str, dict], *, groups: dict[str, list[R1Group]], plan: r1_plan.RunPlan, windows: dict,
    view = timing_view(plan, cap=CAP)
    measured = r1_stage1._timing(evals, groups=groups, plan=view, windows=windows, scorer_delay_ms=dict(delays))
    zero = {name: 0.0 for name in delays}
    zero_delay = r1_stage1._timing(evals, groups=groups, plan=view, windows=windows, scorer_delay_ms=zero)
    return {"measured": _relabel(measured), "zero_delay": _relabel(zero_delay)}


def _relabel(out: dict) -> dict:
    """r1_timing labels its output 'model'; the report vocabulary calls that a replay estimate."""
    return {**out, "assumptions": {**out["assumptions"], "label": LABEL}}


def frozen_candidate(on_time_by_budget: dict[float, int], *, budgets: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                     tolerance: float = 0.05) -> dict:
    reference = on_time_by_budget[max(budgets)]
    floor = (1.0 - tolerance) * reference
    chosen = next((b for b in sorted(budgets) if on_time_by_budget[b] >= floor), max(budgets))
    return {"budget": chosen, "on_time_useful_by_budget": {b: on_time_by_budget[b] for b in sorted(budgets)},
            "reference": reference, "tolerance": tolerance, "rule": RULE}
