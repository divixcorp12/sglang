"""R2 evaluation: T5, G0, R1's T3 and F0 and the native gate scored on the same groups, the median-seed winner, calib
thresholds, the timing replay and the L+2 table (C3, C4, C6, D6-D8, D11).

scorers builds the GroupScorers of one split; evaluate runs r1_stage1.evaluate_split with the plan's caps and the
select (dev) or calibrate (calib) rule; assert_identical refuses a comparison whose scorers did not share group keys,
skip masks and eligible denominators; select picks the winner by median dev recall across seeds; replay and l2_table
run r1_stage1._timing with a per-scorer delay. Only dev and calib are scored (R4, R5).

Inputs: a StageCtx-like ctx (dataset, device, model_path), the loaded R2Plan, the B1 head, (spec, checkpoint path)
pairs for the learned arms, train results {"arms": {arm: {seed: ArmResult dict}}} and R1's seeds result.
Usage: ev = r2_eval.evaluate(groups, r2_eval.scorers(ctx, plan=plan, split="dev", groups=groups, learned=l, f0=f0),
       plan=plan, split="dev"); r2_eval.assert_identical(ev, groups)
"""

from __future__ import annotations

import os
import statistics
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_arms  # noqa: E402
import r1_cache  # noqa: E402
import r1_controls  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r1_run  # noqa: E402
import r1_stage1  # noqa: E402
import r2_plan  # noqa: E402
from r1_groups import R1Group  # noqa: E402

L2_SUFFIX = "_L2"
_T3 = "T3"
_SEED0 = "0"


class _TimingView(msgspec.Struct, frozen=True):
    """The two RunPlan fields r1_stage1._timing reads, taken from an R2 plan."""

    timing: r1_plan.TimingAssumptions
    stage2_gate: r1_plan.Stage2Gate


def _timing_view(plan: r2_plan.R2Plan) -> _TimingView:
    gate = r1_plan.Stage2Gate(split=plan.selection_metric.split, cap=plan.cap_primary, mean_reads=plan.mean_reads)
    return _TimingView(timing=plan.timing, stage2_gate=gate)


def _check_split(split: str, *, plan: r2_plan.R2Plan) -> str:
    """The threshold rule of a split: dev selects, calib calibrates; train and locked_test are refused (R4, R5)."""
    if split == plan.selection_metric.split:
        return "select"
    if split == plan.calibration_rule.split:
        return "calibrate"
    raise ValueError(f"split {split!r} is neither the selection split nor the calibration split")


# ---- scorers and evaluation --------------------------------------------------------------------------------------


def scorers(ctx, *, plan: r2_plan.R2Plan, split: str, groups: list[R1Group],
            learned: dict[str, tuple[r1_plan.ArmSpec, str]], f0: b1_head.B1Head) -> list:
    """The native gate (no floors) and each learned checkpoint, all chunked, on one split's groups.

    learned maps scorer name -> (ArmSpec, checkpoint path); T5 and G0 read raw x, G0 also the native gate scores.
    """
    _check_split(split, plan=plan)
    x_of = r1_cache.make_x_of(ctx.dataset, split=split)
    needs_gates = any(spec.g0 for spec, _ in learned.values())
    gates = r1_arms._target_gates(ctx, groups=groups) if needs_gates else None
    out: list = [r1_controls.GateGroupScorer(floors=None)]
    for name, (spec, path) in learned.items():
        model = r1_arms._load_model(spec, path=path).to(ctx.device)
        collate = r1_arms._collate_fn(ctx, f0=f0, x_of=x_of, full=spec.projection or spec.g0,
                                      gates=gates if spec.g0 else None)
        scorer = r1_models.R1GroupScorer(model, collate_fn=collate, name=name)
        out.append(r1_arms._Chunked(scorer, chunk=r1_arms._CALIB_CHUNK))
    return out


def evaluate(groups: list[R1Group], scorers: list, *, plan: r2_plan.R2Plan, split: str) -> dict:
    """{scorer: {cap: {point, report_json, reads}}} at the plan's caps; dev takes the D8 point, calib the D9 threshold."""
    rule = _check_split(split, plan=plan)
    return r1_stage1.evaluate_split(groups, scorers, caps=(plan.cap_secondary, plan.cap_primary),
                                    mean_reads=plan.mean_reads, rule=rule)


# ---- identical-group check ---------------------------------------------------------------------------------------


def _eligible(groups: list[R1Group]) -> tuple[int, int]:
    return (sum(len(r.labels.eligible_all) for r in groups), sum(len(r.labels.eligible_addr) for r in groups))


def _same_groups(name: str, *, a: list[R1Group], b: list[R1Group], ref: str) -> None:
    if [r.key for r in a] != [r.key for r in b]:
        raise ValueError(f"{name!r} was scored on different group keys than {ref!r} ({len(a)} vs {len(b)} groups)")
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if (x.labels.available != y.labels.available).any():
            raise ValueError(f"{name!r} and {ref!r} disagree on the skip mask of group {i} ({x.key})")


def _check_eval(name: str, *, caps: dict, groups: list[R1Group]) -> None:
    eligible = _eligible(groups)
    for cap, v in caps.items():
        if len(v["reads"]) != len(groups):
            raise ValueError(f"{name!r} cap {cap}: {len(v['reads'])} read lists for {len(groups)} groups")
        p = v["point"]
        if (p["eligible"], p["eligible_addr"]) != eligible:
            raise ValueError(f"{name!r} cap {cap}: eligible denominators {(p['eligible'], p['eligible_addr'])} "
                             f"differ from the groups' {eligible}")
        for r, ids in zip(groups, v["reads"], strict=True):
            if len(ids) and not r.labels.available[ids].all():
                raise ValueError(f"{name!r} cap {cap}: a read targets a skipped row in group {r.key}")
    forwards = {v["point"]["forwards"] for v in caps.values()}
    if len(forwards) > 1:
        raise ValueError(f"{name!r}: forward counts differ across caps {sorted(forwards)}")


def assert_identical(evals: dict, groups: list[R1Group] | dict[str, list[R1Group]]) -> None:
    """Raise unless every scorer shares group keys, skip masks and the eligible denominator.

    groups is the one shared list, or {scorer: its own list} when each scorer was scored separately.
    """
    per = groups if isinstance(groups, dict) else {n: groups for n in evals}
    if set(per) != set(evals):
        raise ValueError(f"scorers {sorted(evals)} and group lists {sorted(per)} differ")
    ref = sorted(evals)[0]
    for name in sorted(evals):
        _same_groups(name, a=per[name], b=per[ref], ref=ref)
        _check_eval(name, caps=evals[name], groups=per[name])
    forwards = {n: next(iter(c.values()))["point"]["forwards"] for n, c in evals.items()}
    if len(set(forwards.values())) > 1:
        raise ValueError(f"forward counts differ across scorers: {forwards}")


# ---- selection ---------------------------------------------------------------------------------------------------


def _dev_recall(arm: dict) -> float:
    """The dev recall of the arm's restored (best) epoch, as r1_arms._dev_recall."""
    return next(r["recall"] for r in arm["curve"] if r["epoch"] == arm["best_epoch"])


def select(train_result: dict, *, r1_seeds: dict) -> dict:
    """Median dev recall per arm across seeds, the winner (ties to fewer parameters) and its seed-0 checkpoint."""
    arms = train_result["arms"]
    by_seed = {n: {str(s): _dev_recall(a) for s, a in seeds.items()} for n, seeds in arms.items()}
    medians = {n: statistics.median(v.values()) for n, v in by_seed.items()}
    params = {n: seeds[_SEED0]["params"] for n, seeds in arms.items()}
    winner = max(sorted(medians), key=lambda n: (medians[n], -params[n]))
    seed0 = arms[winner][_SEED0]
    return {"median_by_arm": medians, "dev_by_seed": by_seed, "params": params, "winner": winner,
            "candidate": {"path": seed0["ckpt_path"], "sha": seed0["ckpt_sha"]},
            "t3_median": float(r1_seeds["selection"]["median_by_arm"][_T3]),
            "rule": "median dev recall across seeds; ties to fewer parameters; the seed-0 checkpoint is the candidate"}


# ---- timing replay -----------------------------------------------------------------------------------------------


def scorer_delays(*, r1_cost: dict, arm_costs: dict[str, dict]) -> dict[str, float]:
    """Replay delay per scorer: gate-only for the native gate, R1's complete median for T3 and F0 ("*"), and each
    R2 arm's own bf16 complete median."""
    return {**r1_run._scorer_delays(r1_cost), **{n: float(c["complete_ms"]) for n, c in arm_costs.items()}}


def replay(evals: dict[str, dict], *, groups: dict[str, list[R1Group]], plan: r2_plan.R2Plan, windows: dict,
           delays: dict[str, float], flagged: set[str]) -> dict:
    """Cap-2 reads under zero / nominal / adverse contention with each scorer's delay, plus no-flag recall.

    evals and groups are keyed by split; delays maps scorer name -> ms with "*" for the unlisted.
    """
    dev, cap = plan.selection_metric.split, plan.cap_primary
    timing = r1_stage1._timing(evals, groups=groups, plan=_timing_view(plan), windows=windows,
                               scorer_delay_ms=delays)
    points = {split: {n: caps[cap]["point"] for n, caps in ev.items()} for split, ev in evals.items()}
    return {"timing": timing, "dev": points[dev], "calib": points.get(plan.calibration_rule.split, {}),
            "noflag": r1_run._noflag(evals[dev], groups=groups[dev], flagged=flagged, cap=cap),
            "scorer_delay_ms": dict(delays), "label": plan.label}


# ---- L+2 ---------------------------------------------------------------------------------------------------------


def _l2_delays(delays: dict[str, float], *, names: list[str]) -> dict[str, float]:
    return {**delays, **{f"{n}{L2_SUFFIX}": delays.get(n, delays["*"]) for n in names}}


def l2_table(ctx, *, plan: r2_plan.R2Plan, f0: b1_head.B1Head, arms: dict[str, tuple[r1_plan.ArmSpec, str]],
             groups: dict[str, list[R1Group]], windows: dict, delays: dict[str, float]) -> dict:
    """The L+2 native gate beside each L+2 arm (the R2 winner and R1's T3), cap-2 points and the horizon-2 replay.

    arms maps arm name -> (spec, L+2 checkpoint path); scorers are named <arm>_L2. groups holds the horizon-2 dev and
    calib groups.
    """
    learned = {f"{n}{L2_SUFFIX}": sp for n, sp in arms.items()}
    evals: dict[str, dict] = {}
    for split, gs in groups.items():
        sc = scorers(ctx, plan=plan, split=split, groups=gs, learned=learned, f0=f0)
        evals[split] = evaluate(gs, sc, plan=plan, split=split)
        assert_identical(evals[split], gs)
    cap = plan.cap_primary
    timing = r1_stage1._timing(evals, groups=groups, plan=_timing_view(plan), windows=windows,
                               scorer_delay_ms=_l2_delays(delays, names=list(arms)))
    return {"horizon": plan.l2_horizon, "label": plan.label,
            "points": {s: {n: caps[cap]["point"] for n, caps in ev.items()} for s, ev in evals.items()},
            "timing": timing}
