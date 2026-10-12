"""R1 training loop: one arm from an ArmSpec over device-resident whole-group caches, selected by the capped policy.

Steps are 512 whole groups (split into micro-batches with gradient accumulation if asked), AdamW with a new/inherited
learning-rate split, linear warmup then cosine, and early stop after `patience` epochs without a better dev point.
Every epoch is scored on dev as the D8 metric: best eligible-row recall at mean reads <= 1 per forward-layer under
cap 2, through r1_policy (the same candidate/sweep path every control uses). T0 also keeps its best epoch at cap 1.

Inputs: an R1Model (or G0Model) already built from its parent, train/dev GroupCaches and the dev R1Groups, in order.
Output: ArmResult with curves, checkpoint path and sha256 values, the parent's sha256 and the parameter count.
Usage: res = r1_train.train_arm(spec, model=m, new_params=new, train_cache=tr, dev_cache=dv, dev_groups=dg, seed=0,
                                out_dir=out, device="cuda", parent_sha=sha)
"""

from __future__ import annotations

import copy
import math
import os
import sys

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_aggregator  # noqa: E402
import r1_cache  # noqa: E402
import r1_challengers  # noqa: E402
import r1_models  # noqa: E402
import r1_policy  # noqa: E402
from r1_groups import ModelInput, R1Group  # noqa: E402
from r1_plan import ArmSpec  # noqa: E402

_EVAL_CHUNK = 512  # groups per dev forward; arbitrary, only bounds memory
_CAP1_ARM = "T0"  # D7/C4: this arm also keeps its best checkpoint at one read per group


class ArmResult(msgspec.Struct):
    arm: str
    seed: int
    curve: list[dict]
    best_epoch: int  # -1 for an arm with no epochs (F0)
    ckpt_path: str
    ckpt_sha: str
    parent_sha: str
    extra_ckpts: dict[str, dict]  # name -> {"path", "sha", "epoch"}
    params: int  # all parameters, frozen included


def _block_probs(model, *, cache: r1_cache.GroupCache) -> np.ndarray:
    """f32 [G, 384] sigmoid block scores of the cached groups, in cache order."""
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, cache.groups, _EVAL_CHUNK):
            idx = torch.arange(a, min(a + _EVAL_CHUNK, cache.groups), device=cache.inp.u.device)
            out.append(torch.sigmoid(model(r1_cache.take(cache, idx))).float().cpu().numpy())
    return np.concatenate(out)


def dev_metric(model, *, dev_cache: r1_cache.GroupCache, dev_groups: list[R1Group], cap: int = 2,
               mean_reads: float = r1_policy.MEAN_READS_BUDGET) -> r1_policy.SweepPoint:
    """D8: the best-recall sweep point with mean reads <= budget under ``cap`` on the cached dev groups."""
    if [r.key for r in dev_groups] != dev_cache.keys:
        raise ValueError("dev_groups do not match the dev cache's group order")
    probs = _block_probs(model, cache=dev_cache)
    cands = [r1_policy.candidates(row, skip=~r.labels.available) for row, r in zip(probs, dev_groups)]
    points = r1_policy.sweep(cands, [r.labels for r in dev_groups], cap=cap)
    return r1_policy.best_under_budget(points, mean_reads=mean_reads)


def _tensor(values: np.ndarray, *, device) -> torch.Tensor:
    return torch.as_tensor(values, device=device)


def _block_logits(model, *, token_logits: torch.Tensor, inp: ModelInput) -> torch.Tensor:
    if model.aggregator is not None:
        return model.aggregator(token_logits, batch=inp.batch)
    live = _tensor(inp.batch.live, device=token_logits.device).bool()
    return r1_models._noisy_or_logit(token_logits, live=live)


def _loss(model, inp: ModelInput, *, spec: ArmSpec, hard_neg_weight: float) -> torch.Tensor:
    b, dev = inp.batch, inp.u.device
    live = _tensor(b.live, device=dev)
    logits = model.token_logits(inp)
    if spec.objective == "b1_token":
        return r1_aggregator.b1_token_loss(logits, token_y=_tensor(b.token_y, device=dev),
                                           hard_neg=_tensor(b.hard_neg, device=dev), live=live,
                                           hard_neg_weight=hard_neg_weight)
    block = _block_logits(model, token_logits=logits, inp=inp)
    loss = r1_aggregator.block_loss(block, miss=_tensor(b.miss, device=dev),
                                    available=_tensor(b.available, device=dev))
    if spec.objective == "block+route":
        loss = loss + spec.route_weight * r1_aggregator.route_loss(
            logits, route=_tensor(b.route, device=dev), live=live)
    return loss


def _lr_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    done = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, done)))


def _optimizer(model, *, spec: ArmSpec, new_params: set[str]) -> torch.optim.AdamW:
    proj = set()
    if spec.projection and spec.lr_projection is not None:
        proj = new_params & r1_challengers.projection_params(model)
    groups = r1_models.param_groups(model, new=new_params - proj, lr_new=spec.lr_new,
                                    lr_inherited=spec.lr_inherited, weight_decay=spec.weight_decay)
    if proj:
        named = dict(model.named_parameters())
        moved = [named[n] for n in sorted(proj) if named[n].requires_grad]
        ids = {id(p) for p in moved}
        groups = [{**g, "params": [p for p in g["params"] if id(p) not in ids]} for g in groups]
        groups = [g for g in groups if g["params"]]
        groups.append({"params": moved, "lr": spec.lr_projection, "weight_decay": spec.weight_decay})
    return torch.optim.AdamW(groups)


def _prepare(model, *, spec: ArmSpec) -> None:
    if isinstance(model, r1_models.R1Model):
        for m in model.head.predictor.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = spec.dropout
    if spec.freeze_encoder or spec.freeze_embeddings:
        r1_models.freeze(model, encoder=spec.freeze_encoder, embeddings=spec.freeze_embeddings)


def _train_epoch(model, opt, sched, *, spec: ArmSpec, cache: r1_cache.GroupCache, groups_per_step: int,
                 micro_groups: int, seed: int, epoch: int) -> float:
    model.train()
    hard_neg_weight = model.cfg.head.hard_neg_weight if isinstance(model, r1_models.R1Model) else 1.0
    total, count = 0.0, 0
    for batch in r1_cache.epoch_batches(cache, groups_per_step=groups_per_step, seed=seed, epoch=epoch):
        opt.zero_grad()
        n = len(batch)
        for a in range(0, n, micro_groups):
            part = batch[a:a + micro_groups]
            loss = _loss(model, r1_cache.take(cache, part), spec=spec, hard_neg_weight=hard_neg_weight)
            (loss * (len(part) / n)).backward()
            total += float(loss.detach()) * len(part)
        count += n
        opt.step()
        sched.step()
    mean = total / max(1, count)
    if not math.isfinite(mean):
        raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
    return mean


def _save(path: str, model, *, parent_sha: str, arm: str) -> str:
    if isinstance(model, r1_models.R1Model):
        return r1_models.save_r1(path, model, parent_sha=parent_sha, arm=arm)
    return r1_challengers.save_g0(path, model, parent_sha=parent_sha, arm=arm)


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _row(*, epoch: int, loss: float | None, lr: float, p: r1_policy.SweepPoint,
         p1: r1_policy.SweepPoint | None) -> dict:
    row = {"epoch": epoch, "loss": loss, "lr": lr, "recall": p.recall, "precision": p.precision,
           "mean_reads": p.mean_reads, "reads": p.reads, "useful": p.useful, "eligible": p.eligible,
           "threshold": _finite(p.threshold)}
    if p1 is not None:
        row.update({"recall_cap1": p1.recall, "mean_reads_cap1": p1.mean_reads,
                    "threshold_cap1": _finite(p1.threshold)})
    return row


def train_arm(spec: ArmSpec, *, model, new_params: set[str], train_cache: r1_cache.GroupCache,
              dev_cache: r1_cache.GroupCache, dev_groups: list[R1Group], seed: int, out_dir: str,
              groups_per_step: int = 512, patience: int = 8, device, parent_sha: str = "",
              micro_groups: int | None = None) -> ArmResult:
    """Train ``model`` for spec.epochs, restore the best dev epoch and save it under ``out_dir``.

    ``parent_sha`` is the sha256 of the parent checkpoint (empty for an arm with no parent).
    """
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    _prepare(model, spec=spec)
    steps = math.ceil(train_cache.groups / groups_per_step)
    warmup, total = spec.warmup * steps, spec.epochs * steps
    opt = sched = None
    if spec.epochs > 0:
        opt = _optimizer(model, spec=spec, new_params=new_params)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: _lr_factor(s, warmup_steps=warmup, total_steps=total))
    keep_cap1 = spec.name == _CAP1_ARM
    micro = micro_groups or groups_per_step

    curve: list[dict] = []
    best_recall, best_epoch, best_state = -1.0, -1, None
    best1_recall, best1_epoch, best1_state = -1.0, -1, None
    if spec.epochs == 0:
        p = dev_metric(model, dev_cache=dev_cache, dev_groups=dev_groups)
        curve.append(_row(epoch=-1, loss=None, lr=0.0, p=p, p1=None))
    for epoch in range(spec.epochs):
        lr = opt.param_groups[0]["lr"]
        loss = _train_epoch(model, opt, sched, spec=spec, cache=train_cache, groups_per_step=groups_per_step,
                            micro_groups=micro, seed=seed, epoch=epoch)
        p = dev_metric(model, dev_cache=dev_cache, dev_groups=dev_groups)
        p1 = dev_metric(model, dev_cache=dev_cache, dev_groups=dev_groups, cap=1) if keep_cap1 else None
        curve.append(_row(epoch=epoch, loss=loss, lr=lr, p=p, p1=p1))
        if p.recall > best_recall:
            best_recall, best_epoch, best_state = p.recall, epoch, copy.deepcopy(model.state_dict())
        if p1 is not None and p1.recall > best1_recall:
            best1_recall, best1_epoch, best1_state = p1.recall, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience and (p1 is None or epoch - best1_epoch >= patience):
            break
    extra: dict[str, dict] = {}
    if best1_state is not None:
        model.load_state_dict(best1_state)
        path = os.path.join(out_dir, f"{spec.name}_seed{seed}_best_cap1.pt")
        extra["best_cap1"] = {"path": path, "epoch": best1_epoch,
                              "sha": _save(path, model, parent_sha=parent_sha, arm=spec.name)}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    path = os.path.join(out_dir, f"{spec.name}_seed{seed}.pt")
    sha = _save(path, model, parent_sha=parent_sha, arm=spec.name)
    return ArmResult(arm=spec.name, seed=seed, curve=curve, best_epoch=best_epoch, ckpt_path=path, ckpt_sha=sha,
                     parent_sha=parent_sha, extra_ckpts=extra,
                     params=sum(p.numel() for p in model.parameters()))
