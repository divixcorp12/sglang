"""R1 training stages: the arm ladder, seed repeats, L+2 and calibration (D5, D7, D8, D9, D11, C4).

stage_train walks the frozen ladder F0, T0..T4 and then T5/G0 at seed 0, each arm from a copy of its parent's checkpoint;
an arm whose dev recall (the D8 point) does not beat its parent's is dropped and the next arm builds on the parent.
stage_seeds repeats T0 and the selected candidate's ancestry at the plan's repeat seeds and reports the median dev
recall; the seed-0 checkpoint stays the candidate. stage_l2 trains the same ancestry and T0 at horizon 2. stage_calib
picks one global threshold per scorer and cap on calib with the shared r1_policy sweep. Every stage writes
``stages/<name>/result.json`` and an atomic DONE.json holding the input and output sha256 values.

Inputs: a StageCtx, the frozen RunPlan, the audit stage's exclusion manifest (stages/audit/exclusions.json), the B1
dataset, the frozen B1 head and (for G0) the model path. Only train, dev and calib splits are ever read (R4).
Usage: r = r1_arms.stage_train(ctx, plan=plan); s = r1_arms.stage_seeds(ctx, plan=plan, train=r)
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import shutil
import statistics
import sys

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_cache  # noqa: E402
import r1_challengers  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
import r1_policy  # noqa: E402
import r1_stage  # noqa: E402
import r1_train  # noqa: E402
from r1_groups import R1Group  # noqa: E402
from r1_plan import ArmSpec, RunPlan  # noqa: E402
from r1_stage import StageCtx  # noqa: E402

# Written by the audit stage; the driver and this module must agree on the name.
EXCLUSIONS_REL = os.path.join("stages", "audit", "exclusions.json")
RESULT_NAME = "result.json"
_SMOKE_GROUPS = 256  # arbitrary; keeps --smoke runs to seconds
_SMOKE_EPOCHS = 1
_CALIB_CHUNK = 1024  # groups per scoring forward; arbitrary, only bounds memory
_CHAIN_ROOT = "T0"
_REFERENCE = "F0"
_CHALLENGER_LINE = "G0"
_CAP1_KEY = "best_cap1"


class _Data(msgspec.Struct):
    groups: dict[str, list[R1Group]]
    caches: dict[str, r1_cache.GroupCache]


class _Run(msgspec.Struct, frozen=True):
    ctx: StageCtx
    plan: RunPlan
    f0: b1_head.B1Head
    data: _Data
    out_dir: str
    horizon: int


class _Ladder(msgspec.Struct):
    results: dict[str, r1_train.ArmResult] = {}
    kept: dict[str, str] = {}  # arm -> the arm whose weights represent it on the ladder
    parent_used: dict[str, str | None] = {}
    skipped: dict[str, str] = {}


# ---- stage bookkeeping -------------------------------------------------------------------------------------------


def _exclusions_path(ctx: StageCtx) -> str:
    return os.path.join(ctx.out, EXCLUSIONS_REL)


def _base_inputs(ctx: StageCtx, *, plan: RunPlan) -> dict[str, str]:
    if ctx.plan_sha is None:
        raise ValueError("training stages need the frozen plan's sha256 in the StageCtx (freeze run_plan.json first)")
    return {"plan_sha": ctx.plan_sha, "plan_content": r1_stage.sha256_json(plan),
            "b1_head": r1_stage.sha256_file(ctx.b1_head),
            "exclusions": r1_stage.sha256_file(_exclusions_path(ctx)), "smoke": str(ctx.smoke)}


def _result_path(ctx: StageCtx, name: str) -> str:
    return os.path.join(r1_stage.stage_dir(ctx, name), RESULT_NAME)


def _ckpt_dir(ctx: StageCtx, name: str) -> str:
    return os.path.join(r1_stage.stage_dir(ctx, name), "ckpt")


def _cached(ctx: StageCtx, name: str, *, inputs: dict[str, str]) -> dict | None:
    if r1_stage.done_state(ctx, name, inputs=inputs) != "match" or not os.path.exists(_result_path(ctx, name)):
        return None
    with open(_result_path(ctx, name)) as f:
        return json.load(f)


def _finish(ctx: StageCtx, name: str, *, inputs: dict[str, str], result: dict, ckpts: dict[str, str]) -> dict:
    """Write result.json, then DONE.json; ``ckpts`` maps checkpoint path -> sha256."""
    path = _result_path(ctx, name)
    r1_stage.write_json_atomic(path, result)
    base = r1_stage.stage_dir(ctx, name)
    outputs = {os.path.relpath(path, base): r1_stage.sha256_file(path)}
    outputs.update({os.path.relpath(p, base): sha for p, sha in sorted(ckpts.items())})
    r1_stage.write_done(ctx, name, inputs=inputs, outputs=outputs,
                        info={"label": result["label"], "checkpoints": len(ckpts)})
    return result


def _ckpt_hashes(results: dict[str, r1_train.ArmResult]) -> dict[str, str]:
    out = {}
    for res in results.values():
        out[res.ckpt_path] = res.ckpt_sha
        out.update({e["path"]: e["sha"] for e in res.extra_ckpts.values()})
    return out


# ---- data and models ---------------------------------------------------------------------------------------------


def _read_groups(ctx: StageCtx, *, split: str, horizon: int) -> list[R1Group]:
    exclusions = r1_groups.load_exclusions(_exclusions_path(ctx))
    stream = r1_groups.read_r1_groups(ctx.dataset, split=split, horizon=horizon, exclusions=exclusions)
    if ctx.smoke:
        stream = itertools.islice(stream, _SMOKE_GROUPS)
    return list(stream)


def _target_gates(ctx: StageCtx, *, groups: list[R1Group]) -> dict:
    layers = sorted({r.key.target_layer for r in groups if r.g.gate_has_bias})
    return r1_gate.load_target_gates(ctx.model_path, layer_ids=layers) if layers else {}


def _build_data(ctx: StageCtx, *, horizon: int, full: bool, f0: b1_head.B1Head) -> _Data:
    """Train and dev caches; ``full`` adds raw x and the native gate scores that T5 and G0 read."""
    groups = {s: _read_groups(ctx, split=s, horizon=horizon) for s in ("train", "dev")}
    gates = _target_gates(ctx, groups=groups["train"] + groups["dev"]) if full else None
    caches = {s: r1_cache.build_cache(groups[s], x_of=r1_cache.make_x_of(ctx.dataset, split=s), pca=f0.pca,
                                      cfg=f0.cfg, vocab=f0.vocab, device=ctx.device, with_x=full, gates=gates)
              for s in groups}
    return _Data(groups=groups, caches=caches)


def _load_f0(ctx: StageCtx) -> b1_head.B1Head:
    return b1_head.load_head(ctx.b1_head)[0]


def _spec(plan: RunPlan, name: str) -> ArmSpec:
    return next(s for s in plan.arms if s.name == name)


def _effective(spec: ArmSpec, *, horizon: int, smoke: bool) -> ArmSpec:
    if not smoke:
        return msgspec.structs.replace(spec, horizon=horizon)
    epochs = min(spec.epochs, _SMOKE_EPOCHS)
    return msgspec.structs.replace(spec, horizon=horizon, epochs=epochs, warmup=min(spec.warmup, epochs))


def _config(spec: ArmSpec, *, head_cfg: b1_head.HeadConfig) -> r1_models.R1Config:
    return r1_models.R1Config(head=head_cfg, position=spec.position, adapter_rank=spec.adapter_rank,
                              aggregator=spec.objective != "b1_token")


def _fresh(spec: ArmSpec, *, f0: b1_head.B1Head, seed: int):
    """F0 (the frozen head), T0 (E5 weights re-initialised on F0's PCA and vocab) or G0; no parent."""
    torch.manual_seed(seed)
    if spec.name == _REFERENCE:
        return r1_models.R1Model(_config(spec, head_cfg=f0.cfg), head=copy.deepcopy(f0)), set()
    if spec.g0:
        model = r1_challengers.G0Model(r1_challengers.G0Config(), n_layers=len(f0.vocab.layers),
                                       n_strata=len(f0.vocab.strata))
    else:
        head = b1_head.B1Head(f0.cfg, pca=f0.pca, vocab=f0.vocab)
        model = r1_models.R1Model(_config(spec, head_cfg=f0.cfg), head=head)
    return model, {n for n, _ in model.named_parameters()}


def _from_parent(spec: ArmSpec, *, parent: r1_models.R1Model, seed: int):
    """The child model and the names of its parameters the parent did not have."""
    torch.manual_seed(seed)
    model = r1_models.R1Model.from_parent(_config(spec, head_cfg=parent.cfg.head), parent=parent)
    own = {k: tuple(v.shape) for k, v in parent.state_dict().items()}
    new = {n for n, p in model.named_parameters() if own.get(n) != tuple(p.shape)}
    if spec.projection:
        model = r1_challengers.with_projection(model)
        new |= r1_challengers.projection_params(model)
    return model, new


def _load_model(spec: ArmSpec, *, path: str):
    if spec.g0:
        return r1_challengers.load_g0(path)[0]
    if spec.projection:
        return r1_challengers.load_with_projection(path)[0]
    return r1_models.load_r1(path)[0]


def _copy_parent(run: _Run, *, parent: r1_train.ArmResult, child: str) -> str:
    dest = os.path.join(run.out_dir, f"parent_{parent.arm}_seed{parent.seed}_for_{child}.pt")
    os.makedirs(run.out_dir, exist_ok=True)
    shutil.copy2(parent.ckpt_path, dest)
    if r1_stage.sha256_file(dest) != parent.ckpt_sha:
        raise ValueError(f"{parent.ckpt_path}: sha256 differs from the recorded {parent.ckpt_sha}")
    return dest


def _train(run: _Run, *, spec: ArmSpec, parent: r1_train.ArmResult | None, seed: int) -> r1_train.ArmResult:
    eff = _effective(spec, horizon=run.horizon, smoke=run.ctx.smoke)
    if parent is None:
        model, new = _fresh(eff, f0=run.f0, seed=seed)
    else:
        copied = _copy_parent(run, parent=parent, child=spec.name)
        model, new = _from_parent(eff, parent=_load_model(_spec(run.plan, parent.arm), path=copied), seed=seed)
    caches = run.data.caches
    return r1_train.train_arm(eff, model=model, new_params=new, train_cache=caches["train"],
                              dev_cache=caches["dev"], dev_groups=run.data.groups["dev"], seed=seed,
                              out_dir=run.out_dir, groups_per_step=run.plan.groups_per_step,
                              patience=run.plan.patience, device=run.ctx.device,
                              parent_sha=parent.ckpt_sha if parent else "")


def _dev_recall(res: r1_train.ArmResult) -> float:
    """The D8 dev recall of the arm's restored (best) epoch."""
    return next(r["recall"] for r in res.curve if r["epoch"] == res.best_epoch)


# ---- the ladder --------------------------------------------------------------------------------------------------


def _record(state: _Ladder, *, spec: ArmSpec, res: r1_train.ArmResult, parent_name: str | None) -> None:
    state.results[spec.name] = res
    state.parent_used[spec.name] = parent_name
    adds = parent_name is None or _dev_recall(res) > _dev_recall(state.results[parent_name])
    state.kept[spec.name] = spec.name if adds else parent_name


def _run_phase(run: _Run, *, specs: list[ArmSpec], state: _Ladder) -> None:
    for spec in specs:
        parent_name = state.kept[spec.parent] if spec.parent else None
        parent = state.results[parent_name] if parent_name else None
        res = _train(run, spec=spec, parent=parent, seed=run.plan.screen_seed)
        _record(state, spec=spec, res=res, parent_name=parent_name)


def _phase(ctx: StageCtx, plan: RunPlan, *, specs: list[ArmSpec], full: bool, f0: b1_head.B1Head, out_dir: str,
           state: _Ladder) -> None:
    """Build the caches, train ``specs`` in order and release the caches; a memory refusal skips what is left."""
    if not specs:
        return
    try:
        data = _build_data(ctx, horizon=1, full=full, f0=f0)
        _run_phase(_Run(ctx=ctx, plan=plan, f0=f0, data=data, out_dir=out_dir, horizon=1), specs=specs, state=state)
    except (MemoryError, torch.cuda.OutOfMemoryError) as e:
        if not full:
            raise
        for s in specs:
            if s.name not in state.results:
                state.skipped[s.name] = f"{type(e).__name__}: {e}"


def _ancestry(state: _Ladder, name: str) -> list[str]:
    out: list[str] = []
    cur: str | None = name
    while cur is not None:
        out.append(cur)
        cur = state.parent_used[cur]
    return out[::-1]


def _choose_candidate(state: _Ladder, *, plan: RunPlan) -> str:
    """D8 over the ladder's final arm and G0: best dev recall, ties to fewer parameters."""
    chain = [s.name for s in plan.arms if s.name != _REFERENCE and not s.g0 and s.name in state.kept]
    finals = {state.kept[chain[-1]]} if chain else set()
    if _CHALLENGER_LINE in state.kept:
        finals.add(state.kept[_CHALLENGER_LINE])
    if not finals:
        raise ValueError("no trained arm to select a candidate from")
    return max(sorted(finals), key=lambda n: (_dev_recall(state.results[n]), -state.results[n].params))


def _arm_records(results: dict[str, r1_train.ArmResult]) -> dict[str, dict]:
    return {n: msgspec.to_builtins(r) for n, r in results.items()}


def stage_train(ctx: StageCtx, *, plan: RunPlan) -> dict:
    """Seed-0 ladder at L+1: {arms, kept, parent_used, skipped, candidate, ancestry, dev_recall}."""
    inputs = _base_inputs(ctx, plan=plan)
    hit = _cached(ctx, "train", inputs=inputs)
    if hit is not None:
        return hit
    f0, out, state = _load_f0(ctx), _ckpt_dir(ctx, "train"), _Ladder()
    plain = [s for s in plan.arms if not (s.projection or s.g0)]
    late = [s for s in plan.arms if s.projection or s.g0]
    _phase(ctx, plan, specs=plain, full=False, f0=f0, out_dir=out, state=state)
    _phase(ctx, plan, specs=late, full=True, f0=f0, out_dir=out, state=state)
    candidate = _choose_candidate(state, plan=plan)
    result = {"horizon": 1, "seed": plan.screen_seed, "label": plan.label,
              "arms": _arm_records(state.results), "kept": state.kept, "parent_used": state.parent_used,
              "skipped": state.skipped, "candidate": candidate, "ancestry": _ancestry(state, candidate),
              "dev_recall": {n: _dev_recall(r) for n, r in state.results.items()}}
    return _finish(ctx, "train", inputs=inputs, result=result, ckpts=_ckpt_hashes(state.results))


# ---- seeds -------------------------------------------------------------------------------------------------------


def _chains(train: dict) -> list[list[str]]:
    """T0 and the candidate's ancestry; one chain when the ancestry starts at T0."""
    ancestry = list(train["ancestry"])
    return [ancestry] if ancestry[0] == _CHAIN_ROOT else [ancestry, [_CHAIN_ROOT]]


def _needs_full(plan: RunPlan, *, names: set[str]) -> bool:
    return any(s.projection or s.g0 for s in plan.arms if s.name in names)


def _run_chain(run: _Run, *, names: list[str], seed: int) -> dict[str, r1_train.ArmResult]:
    """Each arm from the same-seed result of its predecessor in ``names``; the first starts from no parent."""
    out: dict[str, r1_train.ArmResult] = {}
    prev: r1_train.ArmResult | None = None
    for name in names:
        prev = _train(run, spec=_spec(run.plan, name), parent=prev, seed=seed)
        out[name] = prev
    return out


def _ckpt_record(arm: dict) -> dict:
    return {"path": arm["ckpt_path"], "sha": arm["ckpt_sha"], "parent_sha": arm["parent_sha"],
            "extra": arm["extra_ckpts"]}


def _selection(train: dict, *, repeats: dict[str, dict[str, dict]]) -> dict:
    ancestry, candidate = list(train["ancestry"]), train["candidate"]
    arms = sorted({n for chain in _chains(train) for n in chain})
    by_seed: dict[str, dict[str, float]] = {n: {str(train["seed"]): train["dev_recall"][n]} for n in arms}
    for seed, results in repeats.items():
        for n in arms:
            by_seed[n][seed] = _dev_recall(msgspec.convert(results[n], r1_train.ArmResult))
    names = sorted({*arms, _REFERENCE})
    return {"candidate": candidate, "ancestry": ancestry,
            "median_dev": statistics.median(by_seed[candidate].values()), "dev_by_seed": by_seed,
            "median_by_arm": {n: statistics.median(v.values()) for n, v in by_seed.items()},
            "ckpts": {n: _ckpt_record(train["arms"][n]) for n in names}, "horizon": 1, "seed": train["seed"]}


def stage_seeds(ctx: StageCtx, *, plan: RunPlan, train: dict) -> dict:
    """{repeats: {seed: {arm: ArmResult}}, selection: {candidate, ancestry, median_dev, ...}} (D7)."""
    inputs = {**_base_inputs(ctx, plan=plan), "train": r1_stage.sha256_json(train)}
    hit = _cached(ctx, "seeds", inputs=inputs)
    if hit is not None:
        return hit
    chains = _chains(train)
    f0, out = _load_f0(ctx), _ckpt_dir(ctx, "seeds")
    data = _build_data(ctx, horizon=1, full=_needs_full(plan, names={n for c in chains for n in c}), f0=f0)
    run = _Run(ctx=ctx, plan=plan, f0=f0, data=data, out_dir=out, horizon=1)
    repeats: dict[str, dict[str, dict]] = {}
    hashes: dict[str, str] = {}
    for seed in plan.repeat_seeds:
        results: dict[str, r1_train.ArmResult] = {}
        for chain in chains:
            results.update(_run_chain(run, names=chain, seed=seed))
        repeats[str(seed)] = _arm_records(results)
        hashes.update(_ckpt_hashes(results))
    result = {"repeats": repeats, "selection": _selection(train, repeats=repeats), "label": plan.label}
    return _finish(ctx, "seeds", inputs=inputs, result=result, ckpts=hashes)


# ---- L+2 ---------------------------------------------------------------------------------------------------------


def stage_l2(ctx: StageCtx, *, plan: RunPlan, selection: dict) -> dict:
    """The selected ancestry and its matched control T0 trained at L+2 with the same recipe (D11); seed 0."""
    inputs = {**_base_inputs(ctx, plan=plan), "selection": r1_stage.sha256_json(selection)}
    hit = _cached(ctx, "l2", inputs=inputs)
    if hit is not None:
        return hit
    ancestry = list(selection["ancestry"])
    chains = _chains({"ancestry": ancestry})
    f0 = _load_f0(ctx)
    data = _build_data(ctx, horizon=2, full=_needs_full(plan, names=set(itertools.chain(*chains))), f0=f0)
    run = _Run(ctx=ctx, plan=plan, f0=f0, data=data, out_dir=_ckpt_dir(ctx, "l2"), horizon=2)
    results: dict[str, r1_train.ArmResult] = {}
    for chain in chains:
        results.update(_run_chain(run, names=chain, seed=plan.screen_seed))
    candidate = selection["candidate"]
    result = {"horizon": 2, "seed": plan.screen_seed, "label": plan.label, "candidate": candidate,
              "ancestry": ancestry, "control": _CHAIN_ROOT, "arms": _arm_records(results),
              "dev_recall": {n: _dev_recall(r) for n, r in results.items()},
              "l4": "not run: only if L+2 shows useful extra lead (D11)"}
    return _finish(ctx, "l2", inputs=inputs, result=result, ckpts=_ckpt_hashes(results))


# ---- calibration -------------------------------------------------------------------------------------------------


class _Chunked:
    """GroupScorer that scores ``chunk`` groups per forward; the output equals the inner scorer's."""

    def __init__(self, inner, *, chunk: int):
        self.inner = inner
        self.chunk = chunk
        self.name = inner.name

    def score_groups(self, groups: list[R1Group]) -> list:
        out: list = []
        for a in range(0, len(groups), self.chunk):
            out.extend(self.inner.score_groups(groups[a:a + self.chunk]))
        return out


def _learned_scorers(ctx: StageCtx, *, plan: RunPlan, selection: dict, groups: list[R1Group],
                     taken: set[str]) -> list[_Chunked]:
    """F0, the matched control T0, the candidate (and T0's cap-1 checkpoint) as shared-path GroupScorers."""
    f0 = _load_f0(ctx)
    x_of = r1_cache.make_x_of(ctx.dataset, split="calib")
    wanted = {_REFERENCE: ("F0", selection["ckpts"][_REFERENCE]["path"]),
              _CHAIN_ROOT: ("T0", selection["ckpts"][_CHAIN_ROOT]["path"]),
              selection["candidate"]: (selection["candidate"], selection["ckpts"][selection["candidate"]]["path"])}
    extra = selection["ckpts"][_CHAIN_ROOT]["extra"].get(_CAP1_KEY)
    if extra is not None:
        wanted[f"{_CHAIN_ROOT}_{_CAP1_KEY}"] = (_CHAIN_ROOT, extra["path"])
    todo = {n: (_spec(plan, arm), path) for n, (arm, path) in wanted.items() if n not in taken}
    gates = _target_gates(ctx, groups=groups) if any(s.g0 for s, _ in todo.values()) else None
    out = []
    for name, (spec, path) in todo.items():
        model = _load_model(spec, path=path).to(ctx.device)
        collate = _collate_fn(ctx, f0=f0, x_of=x_of, full=spec.projection or spec.g0,
                              gates=gates if spec.g0 else None)
        out.append(_Chunked(r1_models.R1GroupScorer(model, collate_fn=collate, name=name), chunk=_CALIB_CHUNK))
    return out


def _collate_fn(ctx: StageCtx, *, f0: b1_head.B1Head, x_of, full: bool, gates: dict | None):
    def collate(gs: list[R1Group]):
        return r1_cache.collate(gs, x_of=x_of, pca=f0.pca, cfg=f0.cfg, vocab=f0.vocab, device=ctx.device,
                                with_x=full, gates=gates)

    return collate


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _point_at(points: list[r1_policy.SweepPoint], *, threshold: float) -> r1_policy.SweepPoint:
    if math.isinf(threshold):
        p = points[0]
        return r1_policy.SweepPoint(threshold=math.inf, reads=0, useful=0, eligible=p.eligible, forwards=p.forwards,
                                    recall=0.0, precision=0.0, mean_reads=0.0)
    return next(p for p in points if p.threshold == threshold)


def _point_dict(p: r1_policy.SweepPoint) -> dict:
    return {**msgspec.to_builtins(p), "threshold": _finite(p.threshold)}


def calibrate(cands: dict[str, list[r1_policy.Candidates]], *, labels: list, caps: tuple[int, ...],
              mean_reads: float) -> dict[str, dict[int, dict]]:
    """D9: per scorer and cap, the lowest threshold within the read budget and the point it gives (calib only)."""
    out: dict[str, dict[int, dict]] = {}
    for name, c in cands.items():
        out[name] = {}
        for cap in caps:
            pts = r1_policy.sweep(c, labels, cap=cap)
            thr = r1_policy.calibrate_threshold(pts, mean_reads=mean_reads)
            out[name][cap] = {"threshold": _finite(thr), "point": _point_dict(_point_at(pts, threshold=thr))}
    return out


def stage_calib(ctx: StageCtx, *, plan: RunPlan, selection: dict, stage1_scorers: dict) -> dict:
    """{scorer: {cap: {threshold, point}}} on calib at L+1; stage1_scorers maps name -> GroupScorer (native controls)."""
    inputs = {**_base_inputs(ctx, plan=plan), "selection": r1_stage.sha256_json(selection),
              "scorers": r1_stage.sha256_json(sorted(stage1_scorers))}
    hit = _cached(ctx, "calib", inputs=inputs)
    if hit is not None:
        return {name: {int(c): v for c, v in caps.items()} for name, caps in hit["scorers"].items()}
    groups = _read_groups(ctx, split="calib", horizon=1)
    learned = _learned_scorers(ctx, plan=plan, selection=selection, groups=groups, taken=set(stage1_scorers))
    scorers = [*stage1_scorers.values(), *learned]
    cands = r1_policy.score_split(groups, scorers)
    caps = (plan.cap_secondary, plan.cap_primary)
    table = calibrate(cands, labels=[r.labels for r in groups], caps=caps, mean_reads=plan.mean_reads)
    sources = {s.name: ("stage1" if s.name in stage1_scorers else "learned") for s in scorers}
    result = {"horizon": 1, "split": "calib", "groups": len(groups), "label": plan.label, "sources": sources,
              "candidate": selection["candidate"], "scorers": table}
    _finish(ctx, "calib", inputs=inputs, result=result, ckpts={})
    return table
