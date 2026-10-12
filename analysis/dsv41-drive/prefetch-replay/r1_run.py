"""R1 run driver: plan, audit, stage1, train, seeds, l2, calib, cost, replay and report, resumable (S1.5).

Each stage writes ``stages/<name>/result.json`` and an atomic DONE.json holding its input and output sha256s; a stage
is reused only while its recorded inputs match and reruns when they do not (the plan stage refuses instead, through
r1_plan.freeze_plan). run_plan.json is frozen by the plan stage before any training stage can run (C1). When the
Stage-2 gate (D3) fails, train, seeds, l2 and calib are skipped and the run reports after Stage 1; cost and replay
still time F0, so Stage 1's timing replay carries a measured scorer delay. ``--smoke`` (a few sessions, 1 epoch) runs
every stage whatever the gate says. --out must lie under /mnt/nvme1/nvfp4-work/ and outside b1-run-20261012; every
invocation is appended with its exit code to <out>/invocations.jsonl. REPORT.md and results.json land in --out.

Inputs: MODEL_PATH (gates), the B1 dataset dir, the B0 corpus dir, its integrity report and the frozen B1 head.
The train, cost and replay stages need the GPU: run the driver under cc-gpu.lock on cores 32-63.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python3 r1_run.py MODEL_PATH --dataset D
       --corpus C --integrity R --b1-head H --out /mnt/nvme1/nvfp4-work/b1-r1-<date> [--stages ...] [--device cuda]
       [--smoke]
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import traceback
from collections.abc import Callable

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_baselines  # noqa: E402
import b1_head  # noqa: E402
import r1_arms  # noqa: E402
import r1_audit  # noqa: E402
import r1_cache  # noqa: E402
import r1_challengers  # noqa: E402
import r1_controls  # noqa: E402
import r1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r1_report  # noqa: E402
import r1_stage  # noqa: E402
import r1_stage1  # noqa: E402
import r1_stamps  # noqa: E402
from r1_groups import R1Group  # noqa: E402
from r1_plan import RunPlan  # noqa: E402
from r1_stage import StageCtx  # noqa: E402

STAGES = ("plan", "audit", "stage1", "train", "seeds", "l2", "calib", "cost", "replay", "report")
STAGE2 = ("train", "seeds", "l2", "calib")
RESULT_NAME = r1_arms.RESULT_NAME
SELECTION_NAME = "selection.json"
RESULTS_NAME = "results.json"
REPORT_NAME = "REPORT.md"
HORIZON = 1
PARITY_GROUPS, SMOKE_PARITY_GROUPS = 512, 16  # arbitrary sample sizes for the gate parity check
COST_RUNS, COST_WARMUP = 1000, 50
SMOKE_COST_RUNS, SMOKE_COST_WARMUP = 50, 5
_CHUNK = 512  # groups per learned-scorer forward; only bounds memory


class _Run(msgspec.Struct, frozen=True):
    ctx: StageCtx
    integrity: str
    plan: RunPlan | None


# ---- records -----------------------------------------------------------------------------------------------------


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


def _plan_path(ctx: StageCtx) -> str:
    return os.path.join(ctx.out, r1_plan.PLAN_NAME)


def _exclusions_path(ctx: StageCtx) -> str:
    return os.path.join(ctx.out, r1_arms.EXCLUSIONS_REL)


def _input_files(run: _Run) -> dict[str, str]:
    ctx = run.ctx
    return {"dataset_json": os.path.join(ctx.dataset, r1_audit.DATASET_JSON),
            "corpus_manifest": os.path.join(ctx.corpus, r1_audit.CORPUS_MANIFEST),
            "integrity": run.integrity, "b1_head": ctx.b1_head}


def _common_inputs(run: _Run) -> dict[str, str]:
    return {"plan_sha": str(run.ctx.plan_sha), "smoke": str(run.ctx.smoke)}


# ---- data helpers ------------------------------------------------------------------------------------------------


def _sessions(ctx: StageCtx, *, split: str) -> list[str] | None:
    """All sessions, or the first few built ones of the split in a --smoke run (as r1_stage1 picks them)."""
    if not ctx.smoke:
        return None
    info = _read_json(os.path.join(ctx.dataset, r1_audit.DATASET_JSON))
    built = set(info["built_sessions"])
    return [s for s in info["sessions_by_split"][split] if s in built][:r1_stage1.SMOKE_SESSIONS]


def _groups(ctx: StageCtx, *, split: str) -> list[R1Group]:
    exclusions = r1_groups.load_exclusions(_exclusions_path(ctx))
    return list(r1_groups.read_r1_groups(ctx.dataset, split=split, horizon=HORIZON, exclusions=exclusions,
                                         sessions=_sessions(ctx, split=split)))


def _pop(ctx: StageCtx) -> b1_baselines.PopScorer:
    return b1_baselines.PopScorer.fit(r.g for r in _groups(ctx, split="train"))  # train only (R5)


def _windows(ctx: StageCtx) -> tuple[dict, r1_stamps.StampAudit]:
    manifest = _read_json(os.path.join(ctx.corpus, r1_audit.CORPUS_MANIFEST))
    blocks = sorted(set(manifest["session_block"].values()))
    return r1_stamps.corpus_windows(ctx.corpus, blocks=blocks, horizons=(HORIZON,))


def _target_gates(ctx: StageCtx, *, groups: list[R1Group]) -> dict:
    layers = sorted({r.key.target_layer for r in groups if r.g.gate_has_bias})
    return r1_gate.load_target_gates(ctx.model_path, layer_ids=layers) if layers else {}


def _load_arm(spec: r1_plan.ArmSpec, *, path: str):
    if spec.g0:
        return r1_challengers.load_g0(path)[0]
    if spec.projection:
        return r1_challengers.load_with_projection(path)[0]
    return r1_models.load_r1(path)[0]


class _Chunked:
    """GroupScorer that scores _CHUNK groups per forward; the output equals the inner scorer's."""

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for a in range(0, len(groups), _CHUNK):
            out.extend(self._inner.score_groups(groups[a:a + _CHUNK]))
        return out


def _arm_scorer(ctx: StageCtx, *, plan: RunPlan, name: str, path: str, split: str,
                groups: list[R1Group]) -> _Chunked:
    spec = next(s for s in plan.arms if s.name == name)
    f0 = b1_head.load_head(ctx.b1_head)[0]
    model = _load_arm(spec, path=path).to(ctx.device)
    gates = _target_gates(ctx, groups=groups) if spec.g0 else None
    x_of = r1_cache.make_x_of(ctx.dataset, split=split)

    def collate(gs: list[R1Group]):
        return r1_cache.collate(gs, x_of=x_of, pca=f0.pca, cfg=f0.cfg, vocab=f0.vocab, device=ctx.device,
                                with_x=spec.projection or spec.g0, gates=gates)

    return _Chunked(r1_models.R1GroupScorer(model, collate_fn=collate, name=name))


def _gate_passed(ctx: StageCtx) -> bool:
    return bool(_load_result(ctx, "stage1")["stage2_gate"]["passed"])


def _stage2_allowed(ctx: StageCtx) -> bool:
    return _gate_passed(ctx) or ctx.smoke


# ---- plan, audit, stage1 ------------------------------------------------------------------------------------------


def _stage_plan(run: _Run) -> dict:
    """Freeze run_plan.json (arms, seeds, epochs, objectives, rules, both gates, D16/D17) over the input hashes."""
    ctx = run.ctx
    inputs = {os.path.abspath(p): r1_stage.sha256_file(p) for p in _input_files(run).values()}

    def compute() -> dict:
        plan = r1_plan.default_plan(dataset=ctx.dataset, corpus=ctx.corpus, b1_head=ctx.b1_head, out_dir=ctx.out,
                                    inputs=inputs)
        sha = r1_plan.freeze_plan(_plan_path(ctx), plan)
        return {"plan": _plan_path(ctx), "plan_sha": sha, "label": plan.label}

    return _owned(ctx, "plan", inputs=inputs, compute=compute)


def _stage_audit(run: _Run) -> dict:
    """D14 exclusion manifest, the input manifest and a seeded gate-parity sample on dev."""
    ctx, plan = run.ctx, run.plan
    files = _input_files(run)
    inputs = {**_common_inputs(run), "integrity": r1_stage.sha256_file(files["integrity"]),
              "dataset_json": r1_stage.sha256_file(files["dataset_json"])}

    def compute() -> dict:
        exclusions = r1_audit.build_exclusions(integrity=_read_json(run.integrity), dataset_dir=ctx.dataset,
                                               corpus_dir=ctx.corpus)
        r1_stage.write_json_atomic(_exclusions_path(ctx), exclusions)
        manifest = r1_audit.input_manifest(dataset_dir=ctx.dataset, corpus_dir=ctx.corpus,
                                           integrity_path=run.integrity, b1_head_path=ctx.b1_head, repo=REPO)
        manifest_path = os.path.join(r1_stage.stage_dir(ctx, "audit"), "manifest.json")
        r1_stage.write_json_atomic(manifest_path, manifest)
        parity = r1_audit.gate_parity_sample(dataset_dir=ctx.dataset, model_path=ctx.model_path, split="dev",
                                             horizon=HORIZON, seed=plan.screen_seed,
                                             n_groups=SMOKE_PARITY_GROUPS if ctx.smoke else PARITY_GROUPS)
        return {"exclusions": _exclusions_path(ctx), "manifest": manifest_path, "parity": parity,
                "exclusion_cells": len(exclusions["cells"]), "commit": manifest["commit"],
                "dirty": manifest["dirty"], "label": plan.label}

    return _owned(ctx, "audit", inputs=inputs, compute=compute)


def _stage_stage1(run: _Run) -> dict:
    """F0 and the native controls under caps on dev and calib, histograms, timing (no delay) and the D3 gate."""
    ctx, plan = run.ctx, run.plan
    inputs = {**_common_inputs(run), "exclusions": r1_stage.sha256_file(_exclusions_path(ctx)),
              "b1_head": r1_stage.sha256_file(ctx.b1_head), "audit": _result_sha(ctx, "audit")}

    def compute() -> dict:
        windows, stamp_audit = _windows(ctx)
        out = r1_stage1.run_stage1(ctx, plan=plan, exclusions=r1_groups.load_exclusions(_exclusions_path(ctx)),
                                   windows=windows)
        return {**out, "stamp_audit": msgspec.to_builtins(stamp_audit), "label": plan.label}

    return _owned(ctx, "stage1", inputs=inputs, compute=compute)


# ---- stage 2 (r1_arms keeps its own DONE records) ----------------------------------------------------------------


def _stage_train(run: _Run) -> dict:
    r1_arms.stage_train(run.ctx, plan=run.plan)
    return _load_result(run.ctx, "train")


def _stage_seeds(run: _Run) -> dict:
    """Seed repeats; the selection is also written to <out>/selection.json."""
    r1_arms.stage_seeds(run.ctx, plan=run.plan, train=_load_result(run.ctx, "train"))
    seeds = _load_result(run.ctx, "seeds")
    r1_stage.write_json_atomic(os.path.join(run.ctx.out, SELECTION_NAME), seeds["selection"])
    return seeds


def _stage_l2(run: _Run) -> dict:
    r1_arms.stage_l2(run.ctx, plan=run.plan, selection=_load_result(run.ctx, "seeds")["selection"])
    return _load_result(run.ctx, "l2")


def _native_scorers(ctx: StageCtx) -> dict:
    """The controls stage_calib re-thresholds beside the learned arms; F0 and the hybrid come from Stage 1."""
    floors = b1_baselines.load_floors()
    scorers = [r1_controls.GateGroupScorer(), r1_controls.GateGroupScorer(floors=floors),
               r1_controls.SanityScorer(b1_baselines.PrevScorer()), r1_controls.SanityScorer(_pop(ctx))]
    return {s.name: s for s in scorers}


def _stage_calib(run: _Run) -> dict:
    selection = _load_result(run.ctx, "seeds")["selection"]
    r1_arms.stage_calib(run.ctx, plan=run.plan, selection=selection, stage1_scorers=_native_scorers(run.ctx),
                        stage1_sha=_result_sha(run.ctx, "stage1"))
    return _load_result(run.ctx, "calib")


# ---- cost --------------------------------------------------------------------------------------------------------


class _Subject(msgspec.Struct, frozen=True):
    name: str
    path: str | None  # None: the frozen B1 head wrapped as an R1 model
    threshold: float
    note: str


def _thr(v) -> float:
    return math.inf if v is None else float(v)


def _cost_subject(run: _Run) -> _Subject:
    """The candidate when it is a plain R1 arm; F0 otherwise, or when the run stopped after Stage 1."""
    ctx, plan = run.ctx, run.plan
    cap = str(plan.cap_primary)
    if not _stage2_allowed(ctx):
        thr = _load_result(ctx, "stage1")["stage1"]["calib"][r1_stage1.F0][f"cap{cap}"]["threshold"]
        return _Subject(name=r1_stage1.F0_HEAD, path=None, threshold=_thr(thr), note="stopped after Stage 1")
    selection = _load_result(ctx, "seeds")["selection"]
    calib = _load_result(ctx, "calib")["scorers"]
    name = selection["candidate"]
    spec = next(s for s in plan.arms if s.name == name)
    note = "candidate"
    if spec.g0 or spec.projection:
        name, note = r1_stage1.F0_HEAD, f"candidate {spec.name} reads raw x; r1_cost times R1 heads, so F0 is timed"
    return _Subject(name=name, path=selection["ckpts"][name]["path"], threshold=_thr(calib[name][cap]["threshold"]),
                    note=note)


def _subject_model(ctx: StageCtx, subject: _Subject) -> r1_models.R1Model:
    if subject.path is not None:
        return r1_models.load_r1(subject.path)[0]
    f0 = b1_head.load_head(ctx.b1_head)[0]
    return r1_models.R1Model(r1_models.R1Config(head=f0.cfg), head=f0)


def _cost_layer(ctx: StageCtx) -> int:
    manifest = _read_json(_load_result(ctx, "audit")["manifest"])
    return next(lid for lid, b in zip(manifest["layer_ids"], manifest["gate_has_bias"], strict=True) if b)


def _read_set_diff(ctx: StageCtx, model: r1_models.R1Model, *, threshold: float, cap: int) -> float:
    """D17 over every calib group, collated lazily per chunk so the inputs never sit on the GPU at once."""
    groups = _groups(ctx, split="calib")
    f0 = b1_head.load_head(ctx.b1_head)[0]
    x_of = r1_cache.make_x_of(ctx.dataset, split="calib")
    chunks = [groups[a:a + _CHUNK] for a in range(0, len(groups), _CHUNK)]
    inputs = (r1_cache.collate(c, x_of=x_of, pca=f0.pca, cfg=f0.cfg, vocab=f0.vocab, device=ctx.device)
              for c in chunks)
    skips = (np.stack([r1_groups.skip_mask(r.g) for r in c]) for c in chunks)
    return r1_cost.read_set_diff(model, inputs=inputs, skips=skips, threshold=threshold, cap=cap)


def _stage_cost(run: _Run) -> dict:
    """Complete offline scorer cost under a CUDA graph at the plan's live tokens, bf16 beside fp32 (D13, D17)."""
    ctx, plan = run.ctx, run.plan
    subject = _cost_subject(run)
    ckpt = subject.path or ctx.b1_head
    inputs = {**_common_inputs(run), "subject": subject.name, "checkpoint": r1_stage.sha256_file(ckpt),
              "threshold": repr(subject.threshold), "audit": _result_sha(ctx, "audit")}

    def compute() -> dict:
        if not (ctx.device.startswith("cuda") and torch.cuda.is_available()):
            raise RuntimeError(f"the cost stage needs CUDA (got device {ctx.device!r}): "
                               "run it with --device cuda under cc-gpu.lock on cores 32-63")
        layer = _cost_layer(ctx)
        W, bias = r1_gate.load_target_gates(ctx.model_path, layer_ids=[layer])[layer]
        model = _subject_model(ctx, subject).to(ctx.device)
        runs, warmup = (SMOKE_COST_RUNS, SMOKE_COST_WARMUP) if ctx.smoke else (COST_RUNS, COST_WARMUP)
        result = r1_cost.measure_complete(model, W=W, bias=bias, cost=plan.cost, tokens=plan.cost.live_tokens,
                                          runs=runs, warmup=warmup, device=ctx.device)
        diff = _read_set_diff(ctx, model, threshold=subject.threshold, cap=plan.cap_primary)
        threshold = None if math.isinf(subject.threshold) else subject.threshold
        return {**result, "read_set_diff_fraction": diff, "subject": subject.name, "subject_note": subject.note,
                "checkpoint_sha256": inputs["checkpoint"], "threshold": threshold, "layer": layer,
                "gpu": torch.cuda.get_device_name(), "label": plan.label}

    return _owned(ctx, "cost", inputs=inputs, compute=compute)


# ---- replay ------------------------------------------------------------------------------------------------------


def _candidate(run: _Run) -> tuple[str, str] | None:
    if not _stage2_allowed(run.ctx):
        return None
    selection = _load_result(run.ctx, "seeds")["selection"]
    name = selection["candidate"]
    return name, selection["ckpts"][name]["path"]


def _evaluate(run: _Run, *, split: str, groups: list[R1Group], pop, candidate: tuple[str, str] | None) -> dict:
    ctx, plan = run.ctx, run.plan
    learned = None if candidate is None else _arm_scorer(ctx, plan=plan, name=candidate[0], path=candidate[1],
                                                         split=split, groups=groups)
    scorers = r1_controls.control_scorers(
        floors=b1_baselines.load_floors(), pop=pop, learned=learned,
        f0_token_scorer=r1_stage1.f0_token_scorer(ctx, groups=groups, split=split))
    rule = "select" if split == plan.selection_metric.split else "calibrate"
    return r1_stage1.evaluate_split(groups, scorers, caps=(plan.cap_secondary, plan.cap_primary),
                                    mean_reads=plan.mean_reads, rule=rule)


def _noflag(ev: dict, *, groups: list[R1Group], flagged: set[str], cap: int) -> dict[str, float]:
    """Pooled eligible-row recall per scorer at its own threshold, over groups of sessions the corpus did not flag."""
    keep = [i for i, r in enumerate(groups) if r.key.session not in flagged]
    eligible = sum(len(groups[i].labels.eligible_all) for i in keep)
    out = {}
    for name, caps in ev.items():
        reads = caps[cap]["reads"]
        useful = sum(int(np.isin(reads[i], groups[i].labels.eligible_all).sum()) for i in keep)
        out[name] = useful / eligible if eligible else 0.0
    return out


def _stage_replay(run: _Run) -> dict:
    """Timing replay with the measured scorer delay over F0, the controls and the candidate, plus no-flag recall."""
    ctx, plan = run.ctx, run.plan
    candidate = _candidate(run)
    inputs = {**_common_inputs(run), "stage1": _result_sha(ctx, "stage1"), "cost": _result_sha(ctx, "cost"),
              "exclusions": r1_stage.sha256_file(_exclusions_path(ctx)), "candidate": str(candidate),
              "candidate_ckpt": r1_stage.sha256_file(candidate[1]) if candidate else "none"}

    def compute() -> dict:
        delay = float(_load_result(ctx, "cost")["gate_ms"])
        windows, _ = _windows(ctx)
        pop = _pop(ctx)
        groups = {s: _groups(ctx, split=s) for s in r1_stage1.SPLITS}
        evals = {s: _evaluate(run, split=s, groups=groups[s], pop=pop, candidate=candidate) for s in groups}
        summary = r1_stage1.summarize(evals, groups=groups, plan=plan, windows=windows, scorer_delay_ms=delay)
        dev = plan.selection_metric.split
        manifest = _read_json(os.path.join(ctx.corpus, r1_audit.CORPUS_MANIFEST))
        flagged = {f["session_id"] for f in manifest["flagged"]}
        cap = plan.continuation_gate.cap
        return {"scorer_delay_ms": delay, "candidate": None if candidate is None else candidate[0],
                "timing": summary["timing"], "stage2_gate_with_delay": summary["stage2_gate"],
                "dev": {n: caps[cap]["point"] for n, caps in evals[dev].items()},
                "noflag": _noflag(evals[dev], groups=groups[dev], flagged=flagged, cap=cap),
                "flagged_sessions": sorted(flagged), "label": plan.label}

    return _owned(ctx, "replay", inputs=inputs, compute=compute)


# ---- report ------------------------------------------------------------------------------------------------------


def _calibration(stage1: dict, calib: dict | None, *, cap: int) -> dict:
    """Calib operating points at cap: Stage 1's D9 cells (F0 = its noisy-or), replaced by stage_calib's when run."""
    out = {}
    for name, caps in stage1["stage1"].get("calib", {}).items():
        c = caps[f"cap{cap}"]
        out[name] = {"threshold": c["threshold"], "recall_cap2": c["recall"], "precision": c["precision"],
                     "reads": c["reads"], "source": "stage1"}
    if r1_stage1.F0 in out:
        out[r1_stage1.F0_HEAD] = {**out[r1_stage1.F0], "source": f"stage1 {r1_stage1.F0}"}
    for name, caps in (calib or {}).get("scorers", {}).items():
        p = caps[str(cap)]
        out[name] = {"threshold": p["threshold"], "recall_cap2": p["point"]["recall"],
                     "precision": p["point"]["precision"], "reads": p["point"]["reads"], "source": "calib"}
    return out


def _arms(train: dict | None) -> dict:
    out = {}
    for name, a in (train or {}).get("arms", {}).items():
        out[name] = {"parent": train["parent_used"].get(name), "kept_as": train["kept"].get(name), "seed": a["seed"],
                     "epochs": sum(1 for r in a["curve"] if r["epoch"] >= 0), "best_epoch": a["best_epoch"],
                     "params": a["params"], "checkpoint_sha256": a["ckpt_sha"],
                     "parent_sha256": a["parent_sha"] or None, "curve": [r["recall"] for r in a["curve"]]}
    return out


def _selection(train: dict | None, seeds: dict | None) -> dict:
    medians = (seeds or {}).get("selection", {}).get("median_by_arm", {})
    return {n: {"dev_recall_cap2": r, "median_dev_recall_cap2": medians.get(n)}
            for n, r in (train or {}).get("dev_recall", {}).items()}


def _best_row(arm: dict) -> dict:
    return next(r for r in arm["curve"] if r["epoch"] == arm["best_epoch"])


def _l2(l2: dict | None) -> dict:
    if l2 is None:
        return {}

    def cell(name: str) -> dict:
        r = _best_row(l2["arms"][name])
        return {"arm": name, "recall_cap2": r["recall"], "precision": r["precision"], "reads": r["reads"],
                "split": "dev"}

    return {l2["candidate"]: cell(l2["candidate"]), "control": cell(l2["control"])}


def _continuation(plan: RunPlan, *, seeds: dict, calib: dict, cost: dict, replay: dict) -> dict:
    """D10 on the candidate: dev recall at the D8 point (seeds compare on the same split), calib precision and reads."""
    candidate = seeds["selection"]["candidate"]
    if replay["candidate"] != candidate:
        raise RuntimeError(f"replay scored {replay['candidate']!r}, the selection is {candidate!r}: rerun replay")
    cap = str(plan.continuation_gate.cap)

    def arm(name: str) -> dict:
        p = calib["scorers"][name][cap]["point"]
        return {"recall_cap2": replay["dev"][name]["recall"], "precision_calib": p["precision"],
                "reads_calib": p["reads"]}

    controls = {n: arm(n) for n in r1_stage1.NATIVE_CONTROLS if n in replay["dev"] and n in calib["scorers"]}
    best = min(controls, key=lambda n: (-controls[n]["recall_cap2"], n))  # r1_plan's best-control rule
    nominal = replay["timing"]["cases"][plan.continuation_gate.timing_case]
    outcome = r1_plan.continuation_gate(
        candidate=arm(candidate), controls=controls, cost_ms=float(cost["gate_ms"]),
        seeds=seeds["selection"]["dev_by_seed"][candidate],
        noflag={"candidate": replay["noflag"][candidate], "control": replay["noflag"][best]},
        nominal={"candidate": nominal[candidate]["on_time_useful"], "control": nominal[best]["on_time_useful"]},
        spec=plan.continuation_gate)
    return msgspec.to_builtins(outcome)


def _not_evaluated(reason: str) -> dict:
    return {"passed": False, "reasons": [f"not evaluated: {reason}"], "values": {}}


def _gate_result(run: _Run, *, stopped: bool, parts: dict) -> dict:
    if stopped:
        return _not_evaluated("stopped after Stage 1 (D3)")
    missing = [n for n in ("seeds", "calib", "cost", "replay") if parts[n] is None]
    if missing:
        return _not_evaluated(f"stages {missing} have not run")
    return _continuation(run.plan, seeds=parts["seeds"], calib=parts["calib"], cost=parts["cost"],
                         replay=parts["replay"])


def _results(run: _Run) -> dict:
    ctx, plan = run.ctx, run.plan
    stage1 = _load_result(ctx, "stage1")
    stopped = not _stage2_allowed(ctx)
    parts = {n: None if stopped and n in STAGE2 else _maybe(ctx, n) for n in ("audit", *STAGE2, "cost", "replay")}
    replay, cost = parts["replay"], parts["cost"]
    results = {
        "label": plan.label, "plan_sha": ctx.plan_sha, "smoke": ctx.smoke,
        "stage1": stage1["stage1"],
        "timing": replay["timing"] if replay else stage1["timing"],
        "timing_without_scorer_delay": stage1["timing"],
        "stage2_gate": stage1["stage2_gate"],
        "stage2_gate_with_delay": replay["stage2_gate_with_delay"] if replay else None,
        "stage2_forced_by_smoke": ctx.smoke and not stage1["stage2_gate"]["passed"],
        "stopped_after_stage1": stopped,
        "arms": _arms(parts["train"]), "selection": _selection(parts["train"], parts["seeds"]),
        "candidate": (parts["seeds"] or {}).get("selection"),
        "calibration": _calibration(stage1, parts["calib"], cap=plan.cap_primary),
        "cost": cost or {}, "l2": _l2(parts["l2"]),
        "continuation_gate": _gate_result(run, stopped=stopped, parts=parts),
        "audit": parts["audit"], "stamp_audit": stage1.get("stamp_audit"),
        "paths": {"out": ctx.out, "dataset": ctx.dataset, "corpus": ctx.corpus, "model_path": ctx.model_path,
                  "b1_head": ctx.b1_head, "integrity": run.integrity},
    }
    results["attribution"] = r1_report.attribution(results)
    return results


def _stage_report(run: _Run) -> dict:
    """results.json (r1_report schema) and REPORT.md in --out, with the continuation-gate outcome (C7)."""
    ctx = run.ctx
    inputs = {**_common_inputs(run), **{n: _result_sha(ctx, n) for n in STAGES[:-1]}}

    def compute() -> dict:
        results = _results(run)
        results_path, report_path = os.path.join(ctx.out, RESULTS_NAME), os.path.join(ctx.out, REPORT_NAME)
        r1_stage.write_json_atomic(results_path, results)
        _write_text_atomic(report_path, r1_report.render(_read_json(results_path)))
        return {"results": results_path, "results_sha256": r1_stage.sha256_file(results_path),
                "report": report_path, "report_sha256": r1_stage.sha256_file(report_path),
                "stopped_after_stage1": results["stopped_after_stage1"],
                "continuation_gate_passed": results["continuation_gate"]["passed"], "label": run.plan.label}

    return _owned(ctx, "report", inputs=inputs, compute=compute)


_STAGE_FNS: dict[str, Callable[[_Run], dict]] = {
    "plan": _stage_plan, "audit": _stage_audit, "stage1": _stage_stage1, "train": _stage_train,
    "seeds": _stage_seeds, "l2": _stage_l2, "calib": _stage_calib, "cost": _stage_cost, "replay": _stage_replay,
    "report": _stage_report,
}


# ---- orchestration -----------------------------------------------------------------------------------------------


def _with_plan(run: _Run) -> _Run:
    if run.plan is not None:
        return run
    path = _plan_path(run.ctx)
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is not frozen yet: run the plan stage first")
    plan, sha = r1_plan.load_plan(path)
    return msgspec.structs.replace(run, plan=plan, ctx=msgspec.structs.replace(run.ctx, plan_sha=sha))


def _run_stages(run: _Run, *, selected: set[str]) -> dict[str, str]:
    """Run the selected stages in STAGES order; Stage 2 is skipped when the D3 gate failed (outside --smoke)."""
    status: dict[str, str] = {}
    for name in STAGES:
        if name not in selected:
            continue
        if name != "plan":
            run = _with_plan(run)
        if name in STAGE2 and not _stage2_allowed(run.ctx):
            status[name] = "skipped: the Stage-2 gate failed (D3)"
            continue
        _STAGE_FNS[name](run)
        if name == "plan":
            run = _with_plan(run)
        status[name] = "done"
    return status


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_path")
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--integrity", required=True)
    p.add_argument("--b1-head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stages", nargs="+", choices=STAGES, help="default: all, in order")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace, *, out: str) -> dict[str, str]:
    ctx = StageCtx(out=out, dataset=os.path.abspath(args.dataset), corpus=os.path.abspath(args.corpus),
                   model_path=args.model_path, b1_head=os.path.abspath(args.b1_head), device=args.device,
                   smoke=args.smoke, plan_sha=None)
    run_ = _Run(ctx=ctx, integrity=os.path.abspath(args.integrity), plan=None)
    return _run_stages(run_, selected=set(args.stages or STAGES))


def main(argv: list[str] | None = None, *, out_root: str = r1_stage.OUT_ROOT) -> int:
    """Run the selected stages; 2 for a refused --out (not logged), 1 for a failed stage, 0 when done."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse(argv)
    try:
        out = r1_stage.guard_out_root(args.out, root=out_root)
    except ValueError as e:
        print(f"r1_run: {e}", file=sys.stderr)
        return 2
    started, code = _now(), 1
    try:
        print(json.dumps(run(args, out=out), indent=1))
        code = 0
    except Exception:
        traceback.print_exc()
    finally:
        r1_stage.log_invocation(out, argv=argv, exit_code=code, started=started, ended=_now())
    return code


if __name__ == "__main__":
    sys.exit(main())
