"""Bounded cached-feature training primitive. No execution on import and no model loader.

A caller supplies a validated native head/capture bundle. This cannot silently train
from old top-1 traces or substitute reference-prefix probabilities for rollout labels.
"""

import math
import random
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .model import fixed_future_targets, planning_loss
from .sampling import Sampling, probabilities


def label_projection(width, context_width, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.randn(width, context_width, generator=generator, dtype=torch.float32)
    return p / p.norm(dim=1, keepdim=True)


class Trainer:
    def __init__(self, head, config, projection, eos_id):
        if head.config.arm == "B0":
            raise ValueError("stock head is frozen")
        self.head, self.config, self.projection, self.eos_id = (
            head,
            config,
            projection,
            eos_id,
        )
        groups = []
        for dependency in (False, True):
            for decay in (False, True):
                params = [
                    p
                    for n, p in head.named_parameters()
                    if p.requires_grad
                    and n.startswith("dep_") == dependency
                    and (p.ndim >= 2) == decay
                ]
                if params:
                    lr = config["learning_rates"]["dependency" if dependency else "new"]
                    groups.append(
                        {
                            "params": params,
                            "lr": lr,
                            "peak_lr": lr,
                            "weight_decay": config["weight_decay"] if decay else 0.0,
                        }
                    )
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(config["betas"]))
        self.rng = random.Random(config["seed"])

    def _record(self, record):
        meta, tensors = record
        if meta["split"] != "train":
            raise ValueError(
                "training cannot access selection/screen/calibration/test data"
            )
        device = next(self.head.parameters()).device
        values = {k: v.to(device) for k, v in tensors.items()}
        plan = self.head.prepare(values["hidden"], values["history"], meta["anchor"])
        return meta, values, plan

    def reference_loss(self, record):
        meta, t, plan = self._record(record)
        if meta["kind"] != "reference":
            raise ValueError("CE/plan labels require reference-prefix record")
        logits = self.head.teacher_logits(plan, meta["anchor"], t["tokens"])
        ce = F.cross_entropy(logits, t["tokens"])
        aux = ce * 0
        if self.head.has_aux:
            targets = fixed_future_targets(
                t["future_features"],
                t["tokens"],
                self.projection.to(logits.device),
                self.head.config.gamma,
                self.eos_id,
            )
            aux = planning_loss(plan, targets, self.head.config.gamma)
        return ce, aux

    def matching_loss(self, record):
        meta, t, plan = self._record(record)
        if meta["kind"] != "rollout" or meta["producer"]["arm"] not in {"C", "F"}:
            raise ValueError("TV requires common C/F rollout bank")
        logits = self.head.teacher_logits(plan, meta["anchor"], t["tokens"])
        q = probabilities(logits, Sampling(**meta["proposal_sampling"]))
        p = t["target_probabilities"].detach()
        if p.shape != q.shape:
            raise ValueError("target/proposal vocabulary or prefix shape mismatch")
        return 0.5 * (p - q).abs().sum(-1).mean()

    def run_stage(self, references, rollouts, stage, *, on_update=None):
        if stage not in self.config["updates"]:
            raise ValueError("unknown stage")
        count = self.config["updates"][stage]
        warm = stage == "warmstart"
        if not references or not warm and not rollouts:
            raise ValueError("missing stage data")
        if not warm:
            producers = {
                a: sum(r[0]["producer"]["arm"] == a for r in rollouts)
                for a in ("C", "F")
            }
            if not producers["C"] or producers["C"] != producers["F"]:
                raise ValueError("producer banks must be nonempty and equally weighted")
        batch = self.config["effective_batch"]
        if count < 1 or batch < 1:
            raise ValueError("invalid update/batch budget")
        self.head.train()
        logs = []
        for update in range(count):
            warmup = max(1, math.ceil(0.05 * count))
            ratio = (
                (update + 1) / warmup
                if update < warmup
                else 0.1
                + 0.9
                * 0.5
                * (
                    1
                    + math.cos(math.pi * (update - warmup + 1) / max(1, count - warmup))
                )
            )
            for group in self.optimizer.param_groups:
                group["lr"] = group["peak_lr"] * ratio
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            selected = [self.rng.choice(references) for _ in range(batch)]
            valid_aux = sum(
                self.head.has_aux
                and (
                    len(t["tokens"]) == self.head.config.gamma
                    or int(t["tokens"][-1]) == self.eos_id
                )
                for _, t in selected
            )
            for reference in selected:
                # GPU BF16 reference; CPU correctness tests remain FP32.
                device = next(self.head.parameters()).device.type
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else nullcontext()
                )
                with context:
                    ce, aux = self.reference_loss(reference)
                    loss = (ce if warm else 0.1 * ce) + self.config[
                        "auxiliary_weight"
                    ] * aux * batch / max(1, valid_aux)
                    if not warm:
                        loss = loss + 0.9 * self.matching_loss(
                            self.rng.choice(rollouts)
                        )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training objective")
                (loss / batch).backward()
                total += float(loss.detach()) / batch
            gradients = {
                n: float(p.grad.norm())
                for n, p in self.head.named_parameters()
                if p.grad is not None
            }
            norm = torch.nn.utils.clip_grad_norm_(
                self.head.parameters(),
                self.config["gradient_clip"],
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            row = {
                "stage": stage,
                "update": update + 1,
                "loss": total,
                "gradient_norm": float(norm),
                "branch_gradient_norms": gradients,
            }
            logs.append(row)
            if on_update:
                on_update(row)
        return logs
