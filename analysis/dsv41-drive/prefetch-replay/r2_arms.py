"""R2 training: T5 (trainable projection from T3) and G0 (full-score correction) on bf16 device caches (D3-D6, D11).

build_data reads train and dev with R1's exclusion manifest and builds caches whose raw x is held as bf16 under the
plan's budget, refused with a MemoryError when it does not fit (checked against free GPU memory first). train_arms
trains each arm at every seed through r1_train.train_arm; T5 starts from a verified copy of its parent checkpoint
(R1's T3, or R1's L+2 T3 at horizon 2) and G0 from scratch. stage_train wraps this in a DONE-recorded stage that
writes ``stages/<train|l2>/result.json`` with per-seed curves, checkpoint hashes and parent hashes.

Inputs: a StageCtx, the frozen R2Plan, the R1 run (exclusions), parent checkpoints {arm: (path, sha256)}, the B1
dataset and head and (for G0) the model path. Only train and dev are read (R4).
Usage: r = r2_arms.stage_train(ctx, plan=plan, stage="train", horizon=1, seeds=(0, 1, 2),
                               parents={"T5": (t3_path, t3_sha)}, arms=("T5", "G0"))
"""

from __future__ import annotations

import os
import shutil
import sys

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_arms  # noqa: E402
import r1_cache  # noqa: E402
import r1_challengers  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
import r1_stage  # noqa: E402
import r1_stage1  # noqa: E402
import r1_train  # noqa: E402
import r2_plan  # noqa: E402
from r1_groups import R1Group  # noqa: E402
from r1_plan import ArmSpec  # noqa: E402
from r1_stage import StageCtx  # noqa: E402

_X_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_SPLITS = ("train", "dev")
_STAGES = ("train", "l2")


# ---- data --------------------------------------------------------------------------------------------------------


def _x_dtype(plan: r2_plan.R2Plan) -> torch.dtype:
    return _X_DTYPES[plan.x_dtype]


def check_x_budget(*, n_groups: int, plan: r2_plan.R2Plan, free_bytes: int, cfg: b1_head.HeadConfig) -> int:
    """Device bytes the raw-x caches of ``n_groups`` need; MemoryError past the plan budget or free memory.

    The plan budget bounds raw x alone; free memory must hold the whole cache (u, x, s_native) plus the margin.
    """
    need = r1_cache.x_cache_bytes(n_groups, x_dtype=_x_dtype(plan))
    device = r1_cache.cache_bytes(n_groups, cfg=cfg, x_dtype=_x_dtype(plan), with_s_native=True)
    if need > plan.x_budget_bytes or device + plan.x_margin_bytes > free_bytes:
        raise MemoryError(
            f"raw x for {n_groups} groups needs {need} bytes ({plan.x_dtype}) and the caches {device} bytes with "
            f"u and s_native; the plan budget is {plan.x_budget_bytes} bytes and {free_bytes} bytes are free with "
            f"a {plan.x_margin_bytes} byte margin; raise x_budget_bytes in a new plan or free the GPU")
    return need


def _free_bytes(device: str) -> int:
    if torch.device(device).type != "cuda":
        return sys.maxsize
    return torch.cuda.mem_get_info(torch.device(device))[0]


def _read_split(ctx: StageCtx, *, split: str, horizon: int, exclusions) -> list[R1Group]:
    return list(r1_groups.read_r1_groups(ctx.dataset, split=split, horizon=horizon, exclusions=exclusions,
                                         sessions=r1_stage1.smoke_sessions(ctx, split=split)))


def build_data(ctx: StageCtx, *, plan: r2_plan.R2Plan, exclusions_path: str, horizon: int, f0: b1_head.B1Head,
               full: bool) -> r1_arms._Data:
    """Train and dev groups and caches; ``full`` adds bf16 raw x and the native scores, checked against the budget."""
    exclusions = r1_groups.load_exclusions(exclusions_path)
    groups = {s: _read_split(ctx, split=s, horizon=horizon, exclusions=exclusions) for s in _SPLITS}
    if full:
        check_x_budget(n_groups=sum(len(g) for g in groups.values()), plan=plan, free_bytes=_free_bytes(ctx.device),
                       cfg=f0.cfg)
    gates = r1_arms._target_gates(ctx, groups=groups["train"] + groups["dev"]) if full else None
    extra = {"x_dtype": _x_dtype(plan), "x_budget_bytes": plan.x_budget_bytes} if full else {}
    caches = {s: r1_cache.build_cache(groups[s], x_of=r1_cache.make_x_of(ctx.dataset, split=s), pca=f0.pca,
                                      cfg=f0.cfg, vocab=f0.vocab, device=ctx.device, with_x=full, gates=gates,
                                      **extra)
              for s in groups}
    return r1_arms._Data(groups=groups, caches=caches)


# ---- training ----------------------------------------------------------------------------------------------------


def _spec(plan: r2_plan.R2Plan, name: str) -> ArmSpec:
    return next(s for s in plan.arms if s.name == name)


def _g0_config() -> r1_challengers.G0Config:
    return r1_challengers.G0Config()


def _fresh_g0(*, f0: b1_head.B1Head, seed: int) -> tuple[r1_challengers.G0Model, set[str]]:
    torch.manual_seed(seed)
    model = r1_challengers.G0Model(_g0_config(), n_layers=len(f0.vocab.layers), n_strata=len(f0.vocab.strata))
    return model, {n for n, _ in model.named_parameters()}


def _copy_parent(*, path: str, sha: str, dest: str) -> str:
    """Copy a parent checkpoint into the run and verify its sha256 before and after the copy."""
    actual = r1_stage.sha256_file(path)
    if actual != sha:
        raise ValueError(f"{path}: sha256 {actual} differs from the recorded {sha}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)
    if r1_stage.sha256_file(dest) != sha:
        raise ValueError(f"{dest}: copy differs from the recorded sha256 {sha}")
    return dest


def _train_one(ctx: StageCtx, *, plan: r2_plan.R2Plan, spec: ArmSpec, model, new: set[str],
               data: r1_arms._Data, seed: int, horizon: int, out_dir: str, parent_sha: str) -> r1_train.ArmResult:
    eff = r1_arms._effective(spec, horizon=horizon, smoke=ctx.smoke)
    caches = data.caches
    return r1_train.train_arm(eff, model=model, new_params=new, train_cache=caches["train"],
                              dev_cache=caches["dev"], dev_groups=data.groups["dev"], seed=seed, out_dir=out_dir,
                              groups_per_step=plan.groups_per_step, patience=plan.patience, device=ctx.device,
                              parent_sha=parent_sha)


def _train_t5(ctx: StageCtx, *, plan: r2_plan.R2Plan, data: r1_arms._Data, parent: tuple[str, str],
              seeds: tuple[int, ...], horizon: int, out_dir: str) -> dict[int, r1_train.ArmResult]:
    spec = _spec(plan, "T5")
    path, sha = parent
    copied = _copy_parent(path=path, sha=sha, dest=os.path.join(out_dir, f"parent_T5_for_T5_{sha[:12]}.pt"))
    out = {}
    for seed in seeds:
        eff = r1_arms._effective(spec, horizon=horizon, smoke=ctx.smoke)
        model, new = r1_arms._from_parent(eff, parent=r1_models.load_r1(copied)[0], seed=seed)
        out[seed] = _train_one(ctx, plan=plan, spec=spec, model=model, new=new, data=data, seed=seed,
                               horizon=horizon, out_dir=out_dir, parent_sha=sha)
    return out


def _train_g0(ctx: StageCtx, *, plan: r2_plan.R2Plan, data: r1_arms._Data, f0: b1_head.B1Head,
              seeds: tuple[int, ...], horizon: int, out_dir: str) -> dict[int, r1_train.ArmResult]:
    spec = _spec(plan, "G0")
    out = {}
    for seed in seeds:
        model, new = _fresh_g0(f0=f0, seed=seed)
        out[seed] = _train_one(ctx, plan=plan, spec=spec, model=model, new=new, data=data, seed=seed,
                               horizon=horizon, out_dir=out_dir, parent_sha="")
    return out


def train_arms(ctx: StageCtx, *, plan: r2_plan.R2Plan, data: r1_arms._Data, f0: b1_head.B1Head,
               parents: dict[str, tuple[str, str]], seeds: tuple[int, ...], horizon: int, out_dir: str,
               arms: tuple[str, ...] = ("T5", "G0")) -> dict[str, dict[int, r1_train.ArmResult]]:
    """Every arm in ``arms`` at every seed; T5 from its verified parent copy, G0 from scratch."""
    out: dict[str, dict[int, r1_train.ArmResult]] = {}
    if "T5" in arms:
        out["T5"] = _train_t5(ctx, plan=plan, data=data, parent=parents["T5"], seeds=seeds, horizon=horizon,
                              out_dir=out_dir)
    if "G0" in arms:
        out["G0"] = _train_g0(ctx, plan=plan, data=data, f0=f0, seeds=seeds, horizon=horizon, out_dir=out_dir)
    return out


# ---- stage -------------------------------------------------------------------------------------------------------


def _exclusions_path(plan: r2_plan.R2Plan) -> str:
    """R1's manifest, read from the R1 run and never from this run's directory."""
    return os.path.join(plan.r1_run, r1_arms.EXCLUSIONS_REL)


def _stage_inputs(ctx: StageCtx, *, plan: r2_plan.R2Plan, stage: str, horizon: int, seeds: tuple[int, ...],
                  parents: dict[str, tuple[str, str]], arms: tuple[str, ...]) -> dict[str, str]:
    if ctx.plan_sha is None:
        raise ValueError("R2 training stages need the frozen plan's sha256 in the StageCtx (freeze r2_plan.json first)")
    exclusions = r1_stage.sha256_file(_exclusions_path(plan))
    recorded = plan.inputs.get("exclusions")
    if recorded is not None and recorded != exclusions:
        raise ValueError(f"{_exclusions_path(plan)}: sha256 {exclusions} differs from the plan's {recorded}")
    inputs = {"plan_sha": ctx.plan_sha, "plan_content": r1_stage.sha256_json(plan),
              "b1_head": r1_stage.sha256_file(ctx.b1_head), "exclusions": exclusions, "smoke": str(ctx.smoke),
              "stage": stage, "horizon": str(horizon), "seeds": ",".join(str(s) for s in seeds),
              "arms": ",".join(arms)}
    inputs.update({f"parent_{arm}": sha for arm, (_, sha) in sorted(parents.items())})
    return inputs


def _seed_keyed(results: dict[str, dict[int, r1_train.ArmResult]]) -> dict[str, dict[str, dict]]:
    return {arm: {str(s): msgspec.to_builtins(r) for s, r in by_seed.items()} for arm, by_seed in results.items()}


def _ckpt_records(results: dict[str, dict[int, r1_train.ArmResult]]) -> dict[str, dict[str, dict]]:
    return {arm: {str(s): {"path": r.ckpt_path, "sha": r.ckpt_sha, "parent_sha": r.parent_sha}
                  for s, r in by_seed.items()} for arm, by_seed in results.items()}


def _all_hashes(results: dict[str, dict[int, r1_train.ArmResult]]) -> dict[str, str]:
    return r1_arms._ckpt_hashes({f"{arm}:{s}": r for arm, by_seed in results.items() for s, r in by_seed.items()})


def stage_train(ctx: StageCtx, *, plan: r2_plan.R2Plan, stage: str, horizon: int, seeds: tuple[int, ...],
                parents: dict[str, tuple[str, str]], arms: tuple[str, ...]) -> dict:
    """{arms, dev_recall, ckpts} per arm and seed (str keys) for ``arms`` at ``horizon``; stage is 'train' or 'l2'."""
    if stage not in _STAGES:
        raise ValueError(f"stage must be one of {_STAGES}, not {stage!r}")
    inputs = _stage_inputs(ctx, plan=plan, stage=stage, horizon=horizon, seeds=seeds, parents=parents, arms=arms)
    hit = r1_arms._cached(ctx, stage, inputs=inputs)
    if hit is not None:
        return hit
    f0 = r1_arms._load_f0(ctx)
    data = build_data(ctx, plan=plan, exclusions_path=_exclusions_path(plan), horizon=horizon, f0=f0, full=True)
    results = train_arms(ctx, plan=plan, data=data, f0=f0, parents=parents, seeds=seeds, horizon=horizon,
                         out_dir=r1_arms._ckpt_dir(ctx, stage), arms=arms)
    result = {"stage": stage, "horizon": horizon, "seeds": list(seeds), "label": plan.label,
              "arms": _seed_keyed(results), "ckpts": _ckpt_records(results),
              "parents": {arm: {"path": p, "sha": sha} for arm, (p, sha) in parents.items() if arm in arms},
              "dev_recall": {arm: {str(s): r1_arms._dev_recall(r) for s, r in by_seed.items()}
                             for arm, by_seed in results.items()}}
    return r1_arms._finish(ctx, stage, inputs=inputs, result=result, ckpts=_all_hashes(results))
