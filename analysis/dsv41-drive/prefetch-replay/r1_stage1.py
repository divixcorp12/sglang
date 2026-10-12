"""R1 Stage 1 (D3, no training): F0 and the native controls under caps, read histograms, timing and the Stage-2 gate.

On dev and calib, every scorer (F0 = the frozen B1 head, gate, gate + floors, max and noisy-or poolings of F0, the
hybrid, prev and pop) goes through r1_policy at cap 1 and cap 2 with mean reads <= 1 per forward-layer. Dev takes the
D8 point (best recall within budget); calib takes the D9 threshold (lowest within budget). Each policy becomes an
r1_metrics report and its ordered reads; the cap-2 reads are replayed under the plan's D16 contention cases
(r1_timing), and r1_plan.stage2_gate compares F0 with the best native control on dev.

Inputs: a StageCtx (dataset, B1 head, device), the frozen RunPlan, the D14 exclusions and r1_stamps windows.
Output: {stage1, timing, stage2_gate, read_records_paths} in r1_report's results layout; per-policy report JSON and
read records under stages/stage1/.

Usage: out = r1_stage1.run_stage1(ctx, plan=plan, exclusions=r1_groups.load_exclusions(p), windows=windows)
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Literal

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_baselines  # noqa: E402
import b1_dataset  # noqa: E402
import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import r1_cache  # noqa: E402
import r1_controls  # noqa: E402
import r1_groups  # noqa: E402
import r1_metrics  # noqa: E402
import r1_plan  # noqa: E402
import r1_policy  # noqa: E402
import r1_stage  # noqa: E402
import r1_timing  # noqa: E402
from r1_groups import Exclusions, GroupScorer, R1Group  # noqa: E402

STAGE = "stage1"
SPLITS = ("dev", "calib")
F0_HEAD = "F0"
F0 = "F0_noisy_or"  # B1 pooled its head's token probabilities by noisy-or
NATIVE_CONTROLS = ("gate", "gate_floors")
HORIZON = 1
SMOKE_SESSIONS = 2  # sessions per split in a --smoke run
_CHUNK = 512  # groups per F0 forward, as b1_score.evaluate_many


class _ChunkedHead:
    """F0's HeadScorer scored a chunk of groups per forward, so a whole split never builds one batch."""

    def __init__(self, scorer: b1_head.HeadScorer):
        self._scorer = scorer
        self.name = scorer.name

    def __call__(self, g: b1_rows.Group) -> b1_rows.TokenScores:
        return self._scorer(g)

    def score_many(self, groups: list[b1_rows.Group]) -> list[b1_rows.TokenScores]:
        out: list[b1_rows.TokenScores] = []
        for a in range(0, len(groups), _CHUNK):
            out += self._scorer.score_many(groups[a:a + _CHUNK])
        return out


def _threshold(cands: list[r1_policy.Candidates], *, labels: list, cap: int, mean_reads: float,
               rule: Literal["select", "calibrate"]) -> float:
    points = r1_policy.sweep(cands, labels, cap=cap)
    if not points:
        return math.inf
    if rule == "select":
        return r1_policy.best_under_budget(points, mean_reads=mean_reads).threshold
    if rule == "calibrate":
        return r1_policy.calibrate_threshold(points, mean_reads=mean_reads)
    raise ValueError(f"unknown threshold rule {rule!r}")


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _point(rep: r1_metrics.PolicyReport, *, mean_reads: float) -> dict:
    c = rep.cells["pooled"]
    return {"threshold": _finite(rep.threshold), "reads": c.read, "useful": c.useful, "wasted": c.wasted,
            "eligible": c.eligible_all, "eligible_addr": c.eligible_addr, "forwards": c.forwards,
            "recall": c.recall, "recall_addr": c.recall_addr, "precision": c.precision,
            "mean_reads": c.mean_reads, "feasible": c.mean_reads <= mean_reads}


def evaluate_split(groups: list[R1Group], scorers: list[GroupScorer], *, caps=r1_policy.CAPS,
                   mean_reads: float = r1_policy.MEAN_READS_BUDGET,
                   thresholds: dict[str, dict[int, float]] | None = None,
                   rule: Literal["select", "calibrate"] = "select") -> dict:
    """{scorer: {cap: {point, report_json, reads}}}; one shared scoring path, skip mask and denominator for all.

    The threshold is thresholds[scorer][cap] when given, else this split's D8 point (rule='select') or its D9
    threshold (rule='calibrate'). reads holds each group's ordered read ids.
    """
    cands = r1_policy.score_split(groups, scorers)
    labels = [r.labels for r in groups]
    out: dict[str, dict[int, dict]] = {}
    for name, cs in cands.items():
        out[name] = {}
        for cap in caps:
            t = (thresholds[name][cap] if thresholds is not None else
                 _threshold(cs, labels=labels, cap=cap, mean_reads=mean_reads, rule=rule))
            rep = r1_metrics.policy_report(groups, cs, scorer=name, cap=cap, threshold=t)
            out[name][cap] = {"point": _point(rep, mean_reads=mean_reads), "report_json": r1_metrics.report_to_json(rep),
                              "reads": [r1_policy.select(c, threshold=t, cap=cap) for c in cs]}
    return out


def write_read_records(path: str, groups: list[R1Group], reads: list[np.ndarray]) -> None:
    """One npz per policy: group keys, CSR offsets [G+1] and the ordered read ids (int16), written atomically."""
    if len(groups) != len(reads):
        raise ValueError(f"{len(reads)} read lists for {len(groups)} groups")
    ids = [np.asarray(x, dtype=np.int16) for x in reads]
    k = [r.key for r in groups]
    arrays = {
        "session": np.array([x.session for x in k], dtype=str), "block": np.array([x.block for x in k], dtype=str),
        "seq": np.array([x.seq for x in k], dtype=np.int64),
        "source_layer": np.array([x.source_layer for x in k], dtype=np.int64),
        "target_layer": np.array([x.target_layer for x in k], dtype=np.int64),
        "horizon": np.array([x.horizon for x in k], dtype=np.int64),
        "offsets": np.cumsum([0] + [len(x) for x in ids]).astype(np.int64),
        "ids": np.concatenate(ids) if ids else np.zeros(0, dtype=np.int16),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_HIST_STATS = ("p50", "p90", "p95", "p99", "max", "max_burst")


def _stage1_tables(evals: dict[str, dict]) -> dict:
    """r1_report layout: stage1[split][scorer]['capN'] cells, and pre-cap histograms keyed capN/split/key."""
    out: dict = {split: {name: {f"cap{cap}": dict(v["point"]) for cap, v in caps.items()}
                         for name, caps in ev.items()} for split, ev in evals.items()}
    hists: dict[str, dict] = {}
    for split, ev in evals.items():
        for name, caps in ev.items():
            for cap, v in caps.items():
                for key, h in v["report_json"]["hist"].items():
                    hists.setdefault(name, {})[f"cap{cap}/{split}/{key}"] = {
                        **h["bins"], **{q: h[q] for q in _HIST_STATS}}
    out["histograms"] = hists
    return out


def _timing_cell(res: r1_timing.TimingResult, *, eligible: int) -> dict:
    reads = res.reads_admitted + res.reads_dropped_window
    return {"on_time_useful": res.on_time_useful, "late_useful": res.late_useful, "useful": res.useful,
            "wasted": reads - res.useful, "reads": reads, "reads_admitted": res.reads_admitted,
            "recall": res.on_time_useful / eligible if eligible else 0.0,
            "partial_saved_ms": res.partial_saved_ms, "missing_windows": res.missing_windows}


def _delay_for(scorer_delay_ms: float | dict[str, float], *, name: str) -> float:
    if isinstance(scorer_delay_ms, dict):
        return scorer_delay_ms.get(name, scorer_delay_ms["*"])  # "*" is the default for scorers not listed
    return scorer_delay_ms


def _timing(evals: dict[str, dict], *, groups: dict[str, list[R1Group]], plan: r1_plan.RunPlan, windows: dict,
            scorer_delay_ms: float | dict[str, float]) -> dict:
    """Cap-2 reads replayed per contention case; 'cases' is dev (the Stage-2 split), 'calib' holds the same for calib.

    A float is one delay for every scorer; a dict maps scorer name -> delay with "*" as the default.
    """
    cap = plan.stage2_gate.cap
    by_split: dict[str, dict] = {}
    for split, ev in evals.items():
        eligible = sum(len(r.labels.eligible_all) for r in groups[split])
        by_split[split] = {c: {} for c in plan.timing.contention}
        for name, caps in ev.items():
            delay = _delay_for(scorer_delay_ms, name=name)
            for case in r1_timing.cases_from_plan(plan.timing, scorer_delay_ms=delay):
                res = r1_timing.replay(groups[split], caps[cap]["reads"], windows=windows, case=case)
                by_split[split][case.name][name] = _timing_cell(res, eligible=eligible)
    assumptions = {"model": r1_timing.MODEL_TEXT, "label": r1_timing.LABEL, "cap": cap,
                   "row_read_ms": plan.timing.row_read_ms, "row_read_source": plan.timing.row_read_source,
                   "contention": dict(plan.timing.contention), "contention_source": plan.timing.contention_source,
                   "scorer_delay_ms": scorer_delay_ms, "scorer_delay_source": plan.timing.scorer_delay_source,
                   "scorer_delay_scope": ("per scorer: gate-only for the native gate controls, complete otherwise"
                                          if isinstance(scorer_delay_ms, dict)
                                          else "the same delay for every scorer")}
    return {"assumptions": assumptions, "cases": by_split.get(plan.stage2_gate.split, {}),
            "calib": by_split.get("calib", {})}


def _gate_inputs(evals: dict[str, dict], timing: dict, *, name: str, spec: r1_plan.Stage2Gate) -> dict:
    return {"recall_cap2": evals[spec.split][name][spec.cap]["point"]["recall"],
            "on_time_useful_nominal": timing["cases"][spec.timing_case][name]["on_time_useful"]}


def summarize(evals: dict[str, dict], *, groups: dict[str, list[R1Group]], plan: r1_plan.RunPlan, windows: dict,
              scorer_delay_ms: float | dict[str, float]) -> dict:
    """Stage-1 tables, timing cases and the D3 gate (F0 vs the best native control on dev) from evaluate_split runs."""
    timing = _timing(evals, groups=groups, plan=plan, windows=windows, scorer_delay_ms=scorer_delay_ms)
    spec = plan.stage2_gate
    controls = {n: _gate_inputs(evals, timing, name=n, spec=spec) for n in NATIVE_CONTROLS if n in evals[spec.split]}
    gate = r1_plan.stage2_gate(f0=_gate_inputs(evals, timing, name=F0, spec=spec), controls=controls, spec=spec)
    return {"stage1": _stage1_tables(evals), "timing": timing, "stage2_gate": msgspec.to_builtins(gate)}


def smoke_sessions(ctx: r1_stage.StageCtx, *, split: str) -> list[str] | None:
    if not ctx.smoke:
        return None
    with open(os.path.join(ctx.dataset, b1_dataset.DATASET_NAME)) as f:
        info = json.load(f)
    built = set(info["built_sessions"])
    return [s for s in info["sessions_by_split"][split] if s in built][:SMOKE_SESSIONS]


def _groups(ctx: r1_stage.StageCtx, *, split: str, exclusions: Exclusions) -> list[R1Group]:
    return list(r1_groups.read_r1_groups(ctx.dataset, split=split, horizon=HORIZON, exclusions=exclusions,
                                         sessions=smoke_sessions(ctx, split=split)))


def f0_token_scorer(ctx: r1_stage.StageCtx, *, groups: list[R1Group], split: str) -> _ChunkedHead:
    """The frozen B1 head reading x through r1_groups.x_rows, so dropped tokens stay aligned with r.g."""
    model, _ = b1_head.load_head(ctx.b1_head)
    x_of_group = r1_cache.make_x_of(ctx.dataset, split=split)
    by_id = {id(r.g): r for r in groups}
    head = b1_head.HeadScorer(model.to(ctx.device), x_of=lambda g: x_of_group(by_id[id(g)]), name=F0_HEAD)
    return _ChunkedHead(head)


def _write_policy_files(ctx: r1_stage.StageCtx, *, split: str, groups: list[R1Group], ev: dict) -> dict[str, str]:
    root = r1_stage.stage_dir(ctx, STAGE)
    paths: dict[str, str] = {}
    for name, caps in ev.items():
        for cap, v in caps.items():
            stem = f"{split}_{name}_cap{cap}"
            r1_stage.write_json_atomic(os.path.join(root, "reports", f"{stem}.json"), v["report_json"])
            path = os.path.join(root, "reads", f"{stem}.npz")
            write_read_records(path, groups, v["reads"])
            paths[f"{split}/{name}/cap{cap}"] = path
    return paths


def run_stage1(ctx: r1_stage.StageCtx, *, plan: r1_plan.RunPlan, exclusions: Exclusions, windows: dict,
               scorer_delay_ms: float | dict[str, float] = 0.0) -> dict:
    """Stage 1 on dev and calib; pop is fit on train only (R5), F0 is the frozen B1 head."""
    pop = b1_baselines.PopScorer.fit(r.g for r in _groups(ctx, split="train", exclusions=exclusions))
    floors = b1_baselines.load_floors()
    groups: dict[str, list[R1Group]] = {}
    evals: dict[str, dict] = {}
    paths: dict[str, str] = {}
    for split in SPLITS:
        groups[split] = _groups(ctx, split=split, exclusions=exclusions)
        scorers = r1_controls.control_scorers(
            floors=floors, pop=pop, f0_token_scorer=f0_token_scorer(ctx, groups=groups[split], split=split),
            learned=None)
        rule = "select" if split == plan.selection_metric.split else "calibrate"
        evals[split] = evaluate_split(groups[split], scorers, caps=(plan.cap_secondary, plan.cap_primary),
                                      mean_reads=plan.mean_reads, rule=rule)
        paths.update(_write_policy_files(ctx, split=split, groups=groups[split], ev=evals[split]))
    out = summarize(evals, groups=groups, plan=plan, windows=windows, scorer_delay_ms=scorer_delay_ms)
    return {**out, "read_records_paths": paths}
