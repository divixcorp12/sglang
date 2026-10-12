"""R2 run driver: plan, train, select, calib, cost, replay, l2 and report for T5 and G0, resumable (S1.2).

Reads the R1 run read-only (run_plan.json, selection.json, the exclusion manifest, the seeds/calib/cost/l2 results
and the T3, F0 and L+2 T3 checkpoints), hashes them into a frozen r2_plan.json before any training, and writes only
under --out. Each stage writes ``stages/<name>/result.json`` and DONE.json with its input sha256s and is reused while
they match; the plan stage refuses to refreeze a changed plan. Every R2 stage after plan re-verifies the R1 inputs
against the plan's hashes. A raw-x budget refusal (r2_arms.check_x_budget) exits 3 with its message. --out must lie
under /mnt/nvme1/nvfp4-work/, outside b1-run-20261012, b1-r1-20261012 and --r1-run; every accepted invocation is
appended to <out>/invocations.jsonl. results.json, REPORT.md and selection.json land in --out.

Inputs: MODEL_PATH (gates), the R1 run dir, the B1 dataset dir, the B0 corpus dir and the frozen B1 head.
The train, select, calib, cost, replay and l2 stages need the GPU: run them under cc-gpu.lock on cores 32-63.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python3 r2_run.py MODEL_PATH --r1-run R1
       --dataset D --corpus C --b1-head H --out /mnt/nvme1/nvfp4-work/b1-r2-<date> [--stages ...] [--device cuda]
       [--smoke]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
from collections.abc import Callable

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_arms  # noqa: E402
import r1_audit  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
import r1_plan  # noqa: E402
import r1_stage  # noqa: E402
import r1_stage1  # noqa: E402
import r1_stamps  # noqa: E402
import r2_arms  # noqa: E402
import r2_cost  # noqa: E402
import r2_eval  # noqa: E402
import r2_plan  # noqa: E402
import r2_report  # noqa: E402
from r1_groups import R1Group  # noqa: E402
from r1_stage import StageCtx  # noqa: E402

STAGES = ("plan", "train", "select", "calib", "cost", "replay", "l2", "report")
FORBIDDEN = ("b1-run-20261012", "b1-r1-20261012")
RESULT_NAME = r1_arms.RESULT_NAME
SELECTION_NAME = "selection.json"
RESULTS_NAME = "results.json"
REPORT_NAME = "REPORT.md"
L2_TABLE = "l2_table"  # the L+2 comparison's own stage dir; stages/l2 belongs to r2_arms.stage_train
HORIZON = 1
ARMS = ("T5", "G0")
T3, F0 = "T3", "F0"
GATE = "gate"  # b1_baselines.GateScorer's name without floors
COST_RUNS, COST_WARMUP = 1000, 50  # as r1_run
SMOKE_COST_RUNS, SMOKE_COST_WARMUP = 50, 5
BUDGET_EXIT = 3
_R1_RESULTS = ("seeds", "calib", "cost", "l2")
_R1_FILES = {"run_plan": r1_plan.PLAN_NAME, "selection": SELECTION_NAME, "exclusions": r1_arms.EXCLUSIONS_REL,
             **{n: os.path.join("stages", n, RESULT_NAME) for n in _R1_RESULTS}}
_R1_CKPTS = ("t3_seed0", "f0", "t3_l2")


class _Run(msgspec.Struct, frozen=True):
    ctx: StageCtx
    r1_run: str
    plan: r2_plan.R2Plan | None


# ---- records (r1_run._owned's pattern, kept local) ---------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _write_text_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _result_path(ctx: StageCtx, name: str) -> str:
    return os.path.join(r1_stage.stage_dir(ctx, name), RESULT_NAME)


def _has_result(ctx: StageCtx, name: str) -> bool:
    done = os.path.join(r1_stage.stage_dir(ctx, name), r1_stage.DONE_NAME)
    return os.path.exists(done) and os.path.exists(_result_path(ctx, name))


def _load_result(ctx: StageCtx, name: str) -> dict:
    if not _has_result(ctx, name):
        raise RuntimeError(f"stage {name!r} has no DONE record under {ctx.out}; run it first")
    return _read_json(_result_path(ctx, name))


def _maybe(ctx: StageCtx, name: str) -> dict | None:
    return _read_json(_result_path(ctx, name)) if _has_result(ctx, name) else None


def _result_sha(ctx: StageCtx, name: str) -> str:
    return r1_stage.sha256_file(_result_path(ctx, name)) if _has_result(ctx, name) else "none"


def _dir_hashes(path: str) -> dict[str, str]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for name in sorted(filenames):
            if name == r1_stage.DONE_NAME or ".tmp." in name:
                continue
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, path)] = r1_stage.sha256_file(full)
    return out


def _owned(ctx: StageCtx, name: str, *, inputs: dict[str, str], compute: Callable[[], dict]) -> dict:
    """The stage's result.json when its recorded inputs match; otherwise compute, write it, then DONE.json."""
    state = r1_stage.done_state(ctx, name, inputs=inputs)
    if state == "match" and os.path.exists(_result_path(ctx, name)):
        return _load_result(ctx, name)
    r1_stage.write_json_atomic(_result_path(ctx, name), compute())
    r1_stage.write_done(ctx, name, inputs=inputs, outputs=_dir_hashes(r1_stage.stage_dir(ctx, name)),
                        info={"previous": state, "smoke": ctx.smoke})
    return _load_result(ctx, name)


def _common_inputs(run: _Run) -> dict[str, str]:
    return {"plan_sha": str(run.ctx.plan_sha), "smoke": str(run.ctx.smoke)}


# ---- R1 inputs (read-only) ---------------------------------------------------------------------------------------


def _r1_path(r1_run: str, key: str) -> str:
    return os.path.join(r1_run, _R1_FILES[key])


def _r1_result(r1_run: str, key: str) -> dict:
    return _read_json(_r1_path(r1_run, key))


def _r1_ckpts(r1_run: str) -> dict[str, tuple[str, str]]:
    """{key: (path, recorded sha256)}: T3 and F0 at seed 0 from R1's selection, T3 at L+2 from R1's l2 stage."""
    ckpts = _r1_result(r1_run, "seeds")["selection"]["ckpts"]
    l2 = _r1_result(r1_run, "l2")["arms"][T3]
    raw = {"t3_seed0": (ckpts[T3]["path"], ckpts[T3]["sha"]), "f0": (ckpts[F0]["path"], ckpts[F0]["sha"]),
           "t3_l2": (l2["ckpt_path"], l2["ckpt_sha"])}
    return {k: (os.path.join(r1_run, p), sha) for k, (p, sha) in raw.items()}


def _r1_hashes(r1_run: str) -> dict[str, str]:
    """sha256 of every R1 input; a checkpoint whose file differs from R1's recorded sha256 refuses."""
    out = {key: r1_stage.sha256_file(_r1_path(r1_run, key)) for key in _R1_FILES}
    for key, (path, recorded) in _r1_ckpts(r1_run).items():
        actual = r1_stage.sha256_file(path)
        if actual != recorded:
            raise ValueError(f"R1 checkpoint {path}: sha256 {actual} differs from R1's recorded {recorded}")
        out[key] = actual
    return out


def _verify_r1(plan: r2_plan.R2Plan) -> None:
    current = _r1_hashes(plan.r1_run)
    changed = sorted(k for k, sha in current.items() if plan.inputs.get(k) != sha)
    if changed:
        raise RuntimeError(f"R1 inputs {changed} under {plan.r1_run} differ from the frozen R2 plan's hashes")


def _r1_specs(r1_run: str) -> dict[str, r1_plan.ArmSpec]:
    plan, _ = r1_plan.load_plan(_r1_path(r1_run, "run_plan"))
    return {s.name: s for s in plan.arms}


# ---- data helpers ------------------------------------------------------------------------------------------------


def _f0(ctx: StageCtx) -> b1_head.B1Head:
    return b1_head.load_head(ctx.b1_head)[0]


def _groups(run: _Run, *, split: str, horizon: int) -> list[R1Group]:
    """One split's groups under R1's exclusion manifest, read from the R1 run (the smoke subset in --smoke)."""
    exclusions = r1_groups.load_exclusions(_r1_path(run.r1_run, "exclusions"))
    return list(r1_groups.read_r1_groups(run.ctx.dataset, split=split, horizon=horizon, exclusions=exclusions,
                                         sessions=r1_stage1.smoke_sessions(run.ctx, split=split)))


def _corpus_manifest(ctx: StageCtx) -> dict:
    return _read_json(os.path.join(ctx.corpus, r1_audit.CORPUS_MANIFEST))


def _windows(ctx: StageCtx, *, horizon: int) -> dict:
    blocks = sorted(set(_corpus_manifest(ctx)["session_block"].values()))
    return r1_stamps.corpus_windows(ctx.corpus, blocks=blocks, horizons=(horizon,))[0]


def _flagged(ctx: StageCtx) -> set[str]:
    return {f["session_id"] for f in _corpus_manifest(ctx)["flagged"]}


def _spec(plan: r2_plan.R2Plan, name: str) -> r1_plan.ArmSpec:
    return next(s for s in plan.arms if s.name == name)


def _learned(run: _Run, *, names: tuple[str, ...]) -> dict[str, tuple[r1_plan.ArmSpec, str]]:
    """(spec, seed-0 checkpoint) per scorer: R2 arms from this run's train stage, T3 and F0 from R1."""
    train = _load_result(run.ctx, "train")
    r1_specs, r1_ckpts = _r1_specs(run.r1_run), _r1_ckpts(run.r1_run)
    r1_paths = {T3: r1_ckpts["t3_seed0"][0], F0: r1_ckpts["f0"][0]}
    out = {}
    for name in names:
        if name in ARMS:
            out[name] = (_spec(run.plan, name), train["ckpts"][name]["0"]["path"])
        else:
            out[name] = (r1_specs[name], r1_paths[name])
    return out


def _evaluate(run: _Run, *, split: str, names: tuple[str, ...]) -> tuple[dict, list[R1Group]]:
    """The native gate and ``names`` on one split's identical groups, checked by r2_eval.assert_identical."""
    groups = _groups(run, split=split, horizon=HORIZON)
    scorers = r2_eval.scorers(run.ctx, plan=run.plan, split=split, groups=groups,
                              learned=_learned(run, names=names), f0=_f0(run.ctx))
    evals = r2_eval.evaluate(groups, scorers, plan=run.plan, split=split)
    r2_eval.assert_identical(evals, groups)
    return evals, groups


def _points(evals: dict, *, cap: int) -> dict:
    return {name: caps[cap]["point"] for name, caps in evals.items()}


def _winner(ctx: StageCtx) -> str:
    return _load_result(ctx, "select")["winner"]


# ---- plan and train ----------------------------------------------------------------------------------------------


def _stage_plan(run: _Run) -> dict:
    """Freeze r2_plan.json over the R1 input hashes (files and checkpoints) and the B1 head's."""
    ctx = run.ctx
    inputs = {**_r1_hashes(run.r1_run), "b1_head": r1_stage.sha256_file(ctx.b1_head)}

    def compute() -> dict:
        r1, r1_sha = r1_plan.load_plan(_r1_path(run.r1_run, "run_plan"))
        plan = r2_plan.default_plan(r1_run=run.r1_run, r1_plan=r1, r1_plan_sha=r1_sha, inputs=inputs,
                                    dataset=ctx.dataset, corpus=ctx.corpus, b1_head=ctx.b1_head, out_dir=ctx.out)
        path = os.path.join(ctx.out, r2_plan.PLAN_NAME)
        return {"plan": path, "plan_sha": r2_plan.freeze_plan(path, plan), "label": plan.label}

    return _owned(ctx, "plan", inputs=inputs, compute=compute)


def _stage_train(run: _Run) -> dict:
    """T5 (from a verified copy of R1's T3 seed 0) and G0 at L+1, seeds 0/1/2; r2_arms keeps its own DONE record."""
    path, _ = _r1_ckpts(run.r1_run)["t3_seed0"]
    return r2_arms.stage_train(run.ctx, plan=run.plan, stage="train", horizon=HORIZON, seeds=run.plan.seeds,
                               parents={"T5": (path, run.plan.inputs["t3_seed0"])}, arms=ARMS)


# ---- select and calib --------------------------------------------------------------------------------------------


def _stage_select(run: _Run) -> dict:
    """Median-seed winner and the dev comparison beside T3, F0 and the gate; also written to <out>/selection.json."""
    ctx, plan = run.ctx, run.plan
    inputs = {**_common_inputs(run), "train": _result_sha(ctx, "train"), "r1_seeds": plan.inputs["seeds"]}

    def compute() -> dict:
        selection = r2_eval.select(_load_result(ctx, "train"), r1_seeds=_r1_result(run.r1_run, "seeds"))
        evals, _ = _evaluate(run, split=plan.selection_metric.split, names=(*ARMS, T3, F0))
        comparison = {f"cap{c}": _points(evals, cap=c) for c in (plan.cap_secondary, plan.cap_primary)}
        return {**selection, "comparison": comparison, "label": plan.label}

    result = _owned(ctx, "select", inputs=inputs, compute=compute)
    r1_stage.write_json_atomic(os.path.join(ctx.out, SELECTION_NAME),
                               {k: v for k, v in result.items() if k != "comparison"})
    return result


def _stage_calib(run: _Run) -> dict:
    """Calib thresholds (lowest within budget) for the winner, T3, F0 and the gate at both caps."""
    ctx, plan = run.ctx, run.plan
    inputs = {**_common_inputs(run), "select": _result_sha(ctx, "select")}

    def compute() -> dict:
        evals, _ = _evaluate(run, split=plan.calibration_rule.split, names=(_winner(ctx), T3, F0))
        caps = (plan.cap_secondary, plan.cap_primary)
        return {"scorers": {n: {str(c): v[c]["point"] for c in caps} for n, v in evals.items()},
                "label": plan.label}

    return _owned(ctx, "calib", inputs=inputs, compute=compute)


# ---- cost --------------------------------------------------------------------------------------------------------


def _require_cuda(ctx: StageCtx) -> None:
    if not (ctx.device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError(f"the cost stage needs CUDA (got device {ctx.device!r}): "
                           "run it with --device cuda under cc-gpu.lock on cores 32-63")


def _measure(run: _Run, *, name: str, path: str, layer: int) -> dict:
    ctx, plan = run.ctx, run.plan
    spec = _spec(plan, name)
    W, bias = r1_gate.load_target_gates(ctx.model_path, layer_ids=[layer])[layer]
    runs, warmup = (SMOKE_COST_RUNS, SMOKE_COST_WARMUP) if ctx.smoke else (COST_RUNS, COST_WARMUP)
    model = r1_arms._load_model(spec, path=path).to(ctx.device)
    measure = r2_cost.measure_g0 if spec.g0 else r2_cost.measure_t5
    return measure(model, W=W, bias=bias, cost=plan.cost, tokens=plan.cost.live_tokens, runs=runs, warmup=warmup,
                   device=ctx.device)


def _stage_cost(run: _Run) -> dict:
    """T5 and G0 seed-0 scorers under a CUDA graph at the plan's live tokens, bf16 and fp32, on R1's cost layer."""
    ctx, plan = run.ctx, run.plan
    inputs = {**_common_inputs(run), "train": _result_sha(ctx, "train"), "r1_cost": plan.inputs["cost"]}

    def compute() -> dict:
        _require_cuda(ctx)
        layer = int(_r1_result(run.r1_run, "cost")["layer"])
        ckpts = _load_result(ctx, "train")["ckpts"]
        arms = {n: _measure(run, name=n, path=ckpts[n]["0"]["path"], layer=layer) for n in ARMS}
        return {"arms": arms, "layer": layer, "gpu": torch.cuda.get_device_name(), "label": plan.label}

    return _owned(ctx, "cost", inputs=inputs, compute=compute)


def _delays(run: _Run) -> dict[str, float]:
    winner = _winner(run.ctx)
    arms = _load_result(run.ctx, "cost")["arms"]
    return r2_eval.scorer_delays(r1_cost=_r1_result(run.r1_run, "cost"), arm_costs={winner: arms[winner]})


# ---- replay and L+2 ----------------------------------------------------------------------------------------------


def _stage_replay(run: _Run) -> dict:
    """Timing replay (zero / nominal / adverse) of the winner, T3, F0 and the gate with each one's own delay."""
    ctx, plan = run.ctx, run.plan
    inputs = {**_common_inputs(run), "select": _result_sha(ctx, "select"), "cost": _result_sha(ctx, "cost"),
              "r1_cost": plan.inputs["cost"]}

    def compute() -> dict:
        names = (_winner(ctx), T3, F0)
        splits = (plan.selection_metric.split, plan.calibration_rule.split)
        scored = {s: _evaluate(run, split=s, names=names) for s in splits}
        return r2_eval.replay({s: ev for s, (ev, _) in scored.items()},
                              groups={s: gs for s, (_, gs) in scored.items()}, plan=plan,
                              windows=_windows(ctx, horizon=HORIZON), delays=_delays(run), flagged=_flagged(ctx))

    return _owned(ctx, "replay", inputs=inputs, compute=compute)


def _l2_parents(run: _Run, *, winner: str) -> dict[str, tuple[str, str]]:
    if winner != "T5":
        return {}
    path, _ = _r1_ckpts(run.r1_run)["t3_l2"]
    return {"T5": (path, run.plan.inputs["t3_l2"])}


def _l2_table(run: _Run, *, winner: str, train: dict) -> dict:
    ctx, plan = run.ctx, run.plan
    inputs = {**_common_inputs(run), "l2": _result_sha(ctx, "l2"), "replay": _result_sha(ctx, "replay"),
              "r1_t3_l2": plan.inputs["t3_l2"]}

    def compute() -> dict:
        arms = {winner: (_spec(plan, winner), train["ckpts"][winner]["0"]["path"]),
                T3: (_r1_specs(run.r1_run)[T3], _r1_ckpts(run.r1_run)["t3_l2"][0])}
        splits = (plan.selection_metric.split, plan.calibration_rule.split)
        groups = {s: _groups(run, split=s, horizon=plan.l2_horizon) for s in splits}
        delays = _load_result(ctx, "replay")["scorer_delay_ms"]
        return r2_eval.l2_table(ctx, plan=plan, f0=_f0(ctx), arms=arms, groups=groups,
                                windows=_windows(ctx, horizon=plan.l2_horizon), delays=delays)

    return _owned(ctx, L2_TABLE, inputs=inputs, compute=compute)


def _stage_l2(run: _Run) -> dict:
    """The winner (even if it does not beat T3) at L+2, seed 0, beside the L+2 gate and R1's L+2 T3 (D11)."""
    winner = _winner(run.ctx)
    train = r2_arms.stage_train(run.ctx, plan=run.plan, stage="l2", horizon=run.plan.l2_horizon, seeds=(0,),
                                parents=_l2_parents(run, winner=winner), arms=(winner,))
    return _l2_table(run, winner=winner, train=train)


# ---- report ------------------------------------------------------------------------------------------------------


def _arms(train: dict | None) -> dict:
    out: dict = {}
    for arm, by_seed in (train or {}).get("arms", {}).items():
        out[arm] = {seed: {"curve": [r["recall"] for r in a["curve"]],
                           "epochs": sum(1 for r in a["curve"] if r["epoch"] >= 0),
                           "dev_recall_cap2": train["dev_recall"][arm][seed], "checkpoint_sha256": a["ckpt_sha"],
                           "parent_sha256": a["parent_sha"] or None}
                    for seed, a in by_seed.items()}
    return out


def _selection(select: dict | None) -> dict:
    if select is None:
        return {}
    return {"winner": select["winner"], "median_dev_recall": {**select["median_by_arm"], T3: select["t3_median"]},
            "t3_median": select["t3_median"], "dev_by_seed": select["dev_by_seed"], "candidate": select["candidate"]}


def _calib(calib: dict | None, *, cap: int) -> dict:
    out = {}
    for name, caps in (calib or {}).get("scorers", {}).items():
        p = caps[str(cap)]
        out[name] = {"threshold": p["threshold"], "recall_cap2": p["recall"], "precision": p["precision"],
                     "reads": p["reads"]}
    return out


def _cost(cost: dict | None) -> dict:
    """{arm: {dtype: {path: timing + param/buffer bytes}}}, the r2_report shape."""
    out: dict = {}
    for arm, m in (cost or {}).get("arms", {}).items():
        out[arm] = {}
        for dtype in ("bf16", "fp32"):
            row = m[dtype]
            sizes = {"param_bytes": row["param_bytes"], "buffer_bytes": row["buffer_bytes"]}
            out[arm][dtype] = {p: {**row[p], **sizes} for p in r2_report.COST_PATHS if p in row}
    return out


def _increments(cost: dict | None) -> dict:
    return {arm: {"increment_ms": m["increment_ms"], "complete_ms": m["complete_ms"]}
            for arm, m in (cost or {}).get("arms", {}).items()}


def _l2(table: dict | None) -> dict:
    points = (table or {}).get("points", {}).get("dev", {})
    return {n: {"recall_cap2": p["recall"], "precision": p["precision"], "reads": p["reads"]}
            for n, p in points.items()}


def _not_evaluated(reason: str) -> dict:
    # passed None means the gate was never evaluated, not that it failed.
    return {"passed": None, "reasons": [f"not evaluated: {reason}"], "values": {}}


def _continuation(plan: r2_plan.R2Plan, *, select: dict, calib: dict, cost: dict, replay: dict) -> dict:
    """D10 on the winner's seed-0 candidate against the native gate and T3's R1 median."""
    winner = select["winner"]
    if winner not in replay["dev"]:
        raise RuntimeError(f"replay did not score the winner {winner!r}: rerun replay")
    cap = str(plan.continuation_gate.cap)

    def arm(name: str) -> dict:
        p = calib["scorers"][name][cap]
        return {"recall_cap2": replay["dev"][name]["recall"], "precision_calib": p["precision"],
                "reads_calib": p["reads"]}

    nominal = replay["timing"]["cases"][plan.continuation_gate.timing_case]
    outcome = r2_plan.continuation_gate(
        candidate=arm(winner), gate=arm(GATE), t3_median=float(select["t3_median"]),
        cost_ms=float(cost["arms"][winner]["increment_ms"]), seeds=select["dev_by_seed"][winner],
        noflag={"candidate": replay["noflag"][winner], "control": replay["noflag"][GATE]},
        nominal={"candidate": nominal[winner]["on_time_useful"], "control": nominal[GATE]["on_time_useful"]},
        spec=plan.continuation_gate)
    return msgspec.to_builtins(outcome)


def _gate_result(plan: r2_plan.R2Plan, *, parts: dict) -> dict:
    missing = [n for n in ("select", "calib", "cost", "replay") if parts[n] is None]
    if missing:
        return _not_evaluated(f"stages {missing} have not run")
    return _continuation(plan, select=parts["select"], calib=parts["calib"], cost=parts["cost"],
                         replay=parts["replay"])


def _results(run: _Run) -> dict:
    ctx, plan = run.ctx, run.plan
    parts = {n: _maybe(ctx, n) for n in ("train", "select", "calib", "cost", "replay", L2_TABLE)}
    gate = _gate_result(plan, parts=parts)
    return {
        "label": plan.label, "plan_sha": ctx.plan_sha, "smoke": ctx.smoke, "r1_run": run.r1_run,
        "r1_plan_sha": plan.r1_plan_sha, "arms": _arms(parts["train"]), "selection": _selection(parts["select"]),
        "comparison": (parts["select"] or {}).get("comparison", {}),
        "calib": _calib(parts["calib"], cap=plan.cap_primary),
        "timing": (parts["replay"] or {}).get("timing", {}), "cost": _cost(parts["cost"]),
        "cost_increment": _increments(parts["cost"]),
        "l2": _l2(parts[L2_TABLE]), "l2_timing": (parts[L2_TABLE] or {}).get("timing", {}),
        "continuation_gate": gate, "adds_over_t3": None if gate["passed"] is None else bool(gate["passed"]),
        "paths": {"out": ctx.out, "dataset": ctx.dataset, "corpus": ctx.corpus, "model_path": ctx.model_path,
                  "b1_head": ctx.b1_head},
    }


def _stage_report(run: _Run) -> dict:
    """results.json and REPORT.md in --out with the R2 continuation-gate outcome."""
    ctx = run.ctx
    inputs = {**_common_inputs(run), **{n: _result_sha(ctx, n) for n in (*STAGES[1:-1], L2_TABLE)}}

    def compute() -> dict:
        results = _results(run)
        results_path, report_path = os.path.join(ctx.out, RESULTS_NAME), os.path.join(ctx.out, REPORT_NAME)
        r1_stage.write_json_atomic(results_path, results)
        _write_text_atomic(report_path, r2_report.render(_read_json(results_path)))
        return {"results": results_path, "results_sha256": r1_stage.sha256_file(results_path),
                "report": report_path, "report_sha256": r1_stage.sha256_file(report_path),
                "continuation_gate_passed": results["continuation_gate"]["passed"],
                "adds_over_t3": results["adds_over_t3"], "label": run.plan.label}

    return _owned(ctx, "report", inputs=inputs, compute=compute)


_STAGE_FNS: dict[str, Callable[[_Run], dict]] = {
    "plan": _stage_plan, "train": _stage_train, "select": _stage_select, "calib": _stage_calib,
    "cost": _stage_cost, "replay": _stage_replay, "l2": _stage_l2, "report": _stage_report,
}


# ---- orchestration -----------------------------------------------------------------------------------------------


def _with_plan(run: _Run) -> _Run:
    """Load the frozen plan (refusing when it is missing) and re-verify the R1 inputs against its hashes."""
    if run.plan is not None:
        return run
    path = os.path.join(run.ctx.out, r2_plan.PLAN_NAME)
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is not frozen yet: run the plan stage first")
    plan, sha = r2_plan.load_plan(path)
    if os.path.realpath(plan.r1_run) != run.r1_run:
        raise RuntimeError(f"the frozen plan reads R1 run {plan.r1_run!r}, not --r1-run {run.r1_run!r}")
    _verify_r1(plan)
    return msgspec.structs.replace(run, plan=plan, ctx=msgspec.structs.replace(run.ctx, plan_sha=sha))


def _run_stages(run: _Run, *, selected: set[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for name in STAGES:
        if name not in selected:
            continue
        if name != "plan":
            run = _with_plan(run)
        _STAGE_FNS[name](run)
        status[name] = "done"
    return status


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_path")
    p.add_argument("--r1-run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--b1-head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stages", nargs="+", choices=STAGES, help="default: all, in order")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _guard_out(out: str, *, r1_run: str, root: str) -> str:
    """The real --out; ValueError outside the work root, in a protected run dir, or equal to or inside --r1-run."""
    real = r1_stage.guard_out_root(out, root=root, forbidden=FORBIDDEN)
    if real == r1_run or real.startswith(r1_run + os.sep):
        raise ValueError(f"--out {out!r} is or lies inside --r1-run {r1_run!r}, which is read-only")
    return real


def run(args: argparse.Namespace, *, out: str) -> dict[str, str]:
    ctx = StageCtx(out=out, dataset=os.path.abspath(args.dataset), corpus=os.path.abspath(args.corpus),
                   model_path=args.model_path, b1_head=os.path.abspath(args.b1_head), device=args.device,
                   smoke=args.smoke, plan_sha=None)
    run_ = _Run(ctx=ctx, r1_run=os.path.realpath(args.r1_run), plan=None)
    return _run_stages(run_, selected=set(args.stages or STAGES))


def main(argv: list[str] | None = None, *, out_root: str = r1_stage.OUT_ROOT) -> int:
    """2 for a refused --out (not logged), 3 for the raw-x budget refusal, 1 for another failure, 0 when done."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse(argv)
    try:
        out = _guard_out(args.out, r1_run=os.path.realpath(args.r1_run), root=out_root)
    except ValueError as e:
        print(f"r2_run: {e}", file=sys.stderr)
        return 2
    started, code = _now(), 1
    try:
        print(json.dumps(run(args, out=out), indent=1))
        code = 0
    except MemoryError as e:
        print(f"r2_run: refused: {e}", file=sys.stderr)
        code = BUDGET_EXIT
    except Exception:
        traceback.print_exc()
    finally:
        r1_stage.log_invocation(out, argv=argv, exit_code=code, started=started, ended=_now())
    return code


if __name__ == "__main__":
    sys.exit(main())
