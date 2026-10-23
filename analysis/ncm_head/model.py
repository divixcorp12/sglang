"""CPU-testable revision-3 reference head. No SGLang or CUDA import side effects.

Inputs are target-projected committed features and POST-HC, PRE-NORM draft rows.
The projector supplied by the native bridge owns the original norm/LM projection.
This is an eager correctness/training reference, not an optimized serving kernel.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def rms(x, eps=1e-6):
    return (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    ).to(x.dtype)


def mlp(n, r, hidden=None):
    return nn.Sequential(
        nn.Linear(n * r, hidden or 2 * r, bias=False),
        nn.SiLU(),
        nn.Linear(hidden or 2 * r, r, bias=False),
    )


def segments(gamma, device=None):
    if gamma < 1:
        raise ValueError("gamma must be positive")
    ids = torch.arange(gamma, device=device) * min(4, gamma) // gamma
    weights = torch.bincount(ids, minlength=min(4, gamma)).float() / gamma
    return ids, weights


@dataclass
class MemoryState:
    accumulator: torch.Tensor
    mass: torch.Tensor
    latest: torch.Tensor
    processed: int = 0
    generation: str = ""

    @property
    def slots(self):
        return self.accumulator / self.mass[:, None].clamp_min(
            torch.finfo(torch.float32).tiny
        )


class Memory(nn.Module):
    def __init__(self, context_width, r):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_width, r, bias=False),
            nn.Linear(r, 2 * r, bias=False),
            nn.SiLU(),
            nn.Linear(2 * r, r, bias=False),
        )
        self.register_buffer(
            "alpha", torch.exp(-1 / torch.tensor([8.0, 32.0, 128.0, 512.0]))
        )
        self.r = r

    def _apply(self, fn):
        super()._apply(fn)
        # Module.to(bfloat16) must not quantize the slow EMA coefficients.
        self.alpha = torch.exp(
            -1
            / torch.tensor(
                [8.0, 32.0, 128.0, 512.0], device=self.alpha.device, dtype=torch.float32
            )
        )
        return self

    def empty(self, generation=""):
        device = self.alpha.device
        return MemoryState(
            torch.zeros(4, self.r, device=device),
            torch.zeros(4, device=device),
            torch.zeros(self.r, device=device),
            0,
            generation,
        )

    def ingest(self, state, features, *, start, generation):
        # The caller must pass ONLY committed, target-processed rows, contiguous in token order.
        if generation != state.generation or start != state.processed:
            raise ValueError("memory generation/watermark mismatch")
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("features must be finite [processed_tokens, d_context]")
        if features.shape[0] == 0:
            return state
        u = self.encoder(rms(features.to(self.encoder[0].weight.dtype))).float()
        a, b = state.accumulator, state.mass
        for row in u:
            a = self.alpha[:, None] * a + (1 - self.alpha[:, None]) * row
            b = self.alpha * b + (1 - self.alpha)
        return MemoryState(a, b, u[-1], start + len(u), generation)


@dataclass(frozen=True)
class HeadConfig:
    context_width: int
    draft_width: int
    gamma: int = 5
    width: int = 128
    arm: str = "F"
    flat_hidden: int = 256

    def __post_init__(self):
        if (
            min(
                self.context_width,
                self.draft_width,
                self.gamma,
                self.width,
                self.flat_hidden,
            )
            < 1
        ):
            raise ValueError("dimensions must be positive")
        if self.arm not in {
            "B0",
            "C",
            "P",
            "PM",
            "PH",
            "F",
            "F-noaux",
            "F-flat",
            "F-flat-aux",
        }:
            raise ValueError("unknown arm")


@dataclass
class Plan:
    d: torch.Tensor
    anchor: torch.Tensor
    coarse: torch.Tensor
    residual: torch.Tensor
    position: torch.Tensor
    adapter: torch.Tensor
    base_logits: torch.Tensor
    memory_present: bool


class Head(nn.Module):
    def __init__(
        self, config, embedding, projector, dependency_embedding, dependency_output
    ):
        super().__init__()
        self.config = config
        r, g, arm = config.width, config.gamma, config.arm
        self.embedding = nn.Embedding.from_pretrained(embedding.detach(), freeze=True)
        self.projector = projector.requires_grad_(False)
        self.dep_embedding = nn.Embedding.from_pretrained(
            dependency_embedding.detach().float().clone(), freeze=arm == "B0"
        )
        self.dep_output = nn.Linear(
            dependency_output.shape[1], dependency_output.shape[0], bias=False
        )
        self.dep_output.weight = nn.Parameter(
            dependency_output.detach().float().clone(), requires_grad=arm != "B0"
        )
        self.has_memory = arm in {"PM", "F", "F-noaux", "F-flat", "F-flat-aux"}
        self.has_prefix = arm in {"PH", "F", "F-noaux", "F-flat", "F-flat-aux"}
        self.has_aux = arm in {"P", "PM", "PH", "F", "F-flat-aux"}
        self.flat = arm in {"F-flat", "F-flat-aux"}
        if arm == "B0":
            return
        self.memory = Memory(config.context_width, r)
        self.down = nn.Linear(config.draft_width, r, bias=False)
        self.embed_down = nn.Linear(embedding.shape[1], r, bias=False)
        self.position_embedding = nn.Embedding(g, r)
        self.up = nn.Linear(r, config.draft_width, bias=False)
        nn.init.zeros_(self.up.weight)
        if arm == "C":
            self.control = mlp(5, r, config.flat_hidden)
        else:
            j = min(4, g)
            self.segment_embedding = nn.Embedding(j, r)
            if self.has_memory:
                self.scale_embedding = nn.Parameter(torch.randn(4, r) * 0.02)
                self.key, self.value = (
                    nn.Linear(r, r, bias=False),
                    nn.Linear(r, r, bias=False),
                )
                self.q0, self.qfine = (
                    nn.Linear(3 * r, r, bias=False),
                    nn.Linear(3 * r, r, bias=False),
                )
            if self.flat:
                self.flat_predictor = mlp(7, r, config.flat_hidden)
            else:
                self.coarse = mlp(4, r)
                self.fine = mlp(6, r)
            self.adapter = mlp(3, r)
        if self.has_prefix:
            self.state_init = mlp(2, r)
            self.state_to_dep = nn.Linear(r, dependency_embedding.shape[1], bias=False)
            nn.init.zeros_(self.state_to_dep.weight)
            self.joint = nn.Linear(4 * r, 2 * r, bias=False)

    def _read(self, query, state):
        if not self.has_memory or not state.processed:
            return torch.zeros_like(query)
        m = state.slots.to(query.dtype)
        keys = self.key(rms(m + self.scale_embedding))
        values = self.value(rms(m))
        score = (query.float() @ keys.float().T) / math.sqrt(self.config.width)
        return score.softmax(-1).to(values.dtype) @ values

    def prepare(self, hidden, history, anchor_id):
        c, r = self.config, self.config.width
        if (
            hidden.shape != (c.gamma, c.draft_width)
            or history.ndim != 2
            or history.shape[1] != c.context_width
        ):
            raise ValueError("incorrect feature boundary/shape")
        if not torch.isfinite(hidden).all() or not torch.isfinite(history).all():
            raise ValueError("nonfinite input features")
        if c.arm == "B0":
            z = hidden.new_zeros(c.gamma, r)
            return Plan(
                z,
                z[0],
                z[0],
                z[: min(4, c.gamma)],
                z,
                z,
                self.projector(hidden),
                bool(len(history)),
            )
        # Memory-off controls still encode the latest processed feature; do not replay unused history.
        used_history = history if self.has_memory else history[-1:]
        state = self.memory.ingest(
            self.memory.empty(), used_history, start=0, generation=""
        )
        d = self.down(rms(hidden.to(self.down.weight.dtype)))
        anchor = self.embed_down(
            self.embedding(torch.as_tensor(anchor_id, device=hidden.device)).to(d.dtype)
        )
        latest = state.latest.to(d.dtype)
        pool = d.mean(0)
        pos = self.position_embedding.weight
        ids, weights = segments(c.gamma, hidden.device)
        if c.arm == "C":
            a = self.control(
                torch.cat(
                    [
                        d,
                        anchor.expand_as(d),
                        latest.expand_as(d),
                        pool.expand_as(d),
                        pos,
                    ],
                    -1,
                )
            )
            z0, delta, z = pool * 0, d[: len(weights)] * 0, d * 0
        else:
            seg = torch.stack([d[ids == i].mean(0) for i in range(len(weights))])
            se = self.segment_embedding.weight
            query0 = torch.cat([anchor, latest, pool])
            read0 = (
                self._read(self.q0(query0), state)
                if self.has_memory
                else torch.zeros_like(pool)
            )
            if self.flat:
                # Same observed information, four segments, and global/segment memory access.
                q = torch.cat([pool.expand_as(seg), seg, se], -1)
                readj = (
                    self._read(self.qfine(q), state)
                    if self.has_memory
                    else torch.zeros_like(seg)
                )
                means = self.flat_predictor(
                    torch.cat(
                        [
                            anchor.expand_as(seg),
                            latest.expand_as(seg),
                            pool.expand_as(seg),
                            seg,
                            se,
                            read0.expand_as(seg),
                            readj,
                        ],
                        -1,
                    )
                )
                z0 = (weights[:, None] * means).sum(0)
                delta = means - z0
            else:
                z0 = self.coarse(torch.cat([anchor, latest, pool, read0]))
                q = torch.cat([z0.expand_as(seg), seg, se], -1)
                readj = (
                    self._read(self.qfine(q), state)
                    if self.has_memory
                    else torch.zeros_like(seg)
                )
                raw = self.fine(
                    torch.cat(
                        [
                            z0.expand_as(seg),
                            seg,
                            anchor.expand_as(seg),
                            latest.expand_as(seg),
                            readj,
                            se,
                        ],
                        -1,
                    )
                )
                delta = raw - (weights[:, None] * raw).sum(0)
            z = z0 + delta[ids]
            a = self.adapter(torch.cat([d, z, pos], -1))
        logits = self.projector(hidden + self.up(a).to(hidden.dtype))
        return Plan(d, anchor, z0, delta, z, a, logits, bool(len(history)))

    def initial_state(self, plan):
        return (
            self.state_init(torch.cat([plan.coarse, plan.anchor]))
            if self.has_prefix
            else torch.zeros_like(plan.coarse)
        )

    def step_logits(self, plan, state, previous_id, position):
        v = self.dep_embedding(torch.as_tensor(previous_id, device=plan.d.device))
        if self.has_prefix:
            v = v + self.state_to_dep(state)
        return plan.base_logits[position].float() + self.dep_output(v.float()), v

    def observe(self, plan, state, token_id, position):
        if not self.has_prefix:
            return state
        e = self.embed_down(
            self.embedding(torch.as_tensor(token_id, device=state.device)).to(
                state.dtype
            )
        )
        gate, candidate = self.joint(
            torch.cat([e, plan.d[position], plan.position[position], state])
        ).chunk(2)
        gate = gate.sigmoid()
        return gate * state + (1 - gate) * candidate.tanh()

    def teacher_logits(self, plan, anchor_id, tokens):
        state, previous, logits = self.initial_state(plan), anchor_id, []
        for i, token in enumerate(tokens):
            value, _ = self.step_logits(plan, state, previous, i)
            logits.append(value)
            if i + 1 < len(tokens):
                state = self.observe(plan, state, token, i)
            previous = token
        return torch.stack(logits)


class Confidence(nn.Module):
    """All token/planner inputs detached; only this module receives confidence gradients."""

    def __init__(self, width, rank, sampling_features):
        super().__init__()
        self.vproj = nn.Linear(rank, width, bias=False)
        self.hidden = nn.Linear(6 * width + sampling_features, width, bias=False)
        self.out = nn.Linear(width, 1, bias=True)

    def forward(self, plan, state, correction, position_embedding, sampling, i):
        fixed = torch.cat(
            [plan.d[i], plan.adapter[i], plan.position[i], position_embedding, state]
        ).detach()
        x = torch.cat([fixed, self.vproj(correction.detach()), sampling.detach()])
        return self.out(F.silu(self.hidden(x))).squeeze(-1)


class DenseProjector(nn.Module):
    """Frozen reference projector; native quantized forward must be validated separately."""

    def __init__(self, weight, norm_weight, eps):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("norm_weight", norm_weight.detach().clone())
        self.eps = eps

    def forward(self, hidden):
        normalized = rms(hidden, self.eps) * self.norm_weight.to(hidden.dtype)
        return F.linear(normalized.to(self.weight.dtype), self.weight).float()


def fixed_future_targets(features, token_ids, projection, gamma, eos_id):
    """Return None for unknown truncated futures. EOS feature included; only later rows zero."""
    if len(features) != len(token_ids) or len(features) > gamma:
        raise ValueError("future feature/token alignment")
    if projection.requires_grad:
        raise ValueError("label projection must be frozen")
    with torch.no_grad():
        rows = []
        terminal = False
        for f, t in zip(features, token_ids):
            if terminal:
                raise ValueError("actual rows after EOS are forbidden")
            rows.append(F.linear(rms(f.float()), projection.float()))
            terminal = int(t) == eos_id
        if len(rows) < gamma and not terminal:
            return None
        while len(rows) < gamma:
            rows.append(projection.new_zeros(projection.shape[0]))
        t = torch.stack(rows)
        ids, w = segments(gamma, t.device)
        coarse = t.mean(0)
        means = torch.stack([t[ids == j].mean(0) for j in range(len(w))])
        return coarse, means - coarse


def planning_loss(plan, targets, gamma):
    if targets is None:
        return plan.coarse.sum() * 0
    coarse, residual = targets
    _, w = segments(gamma, plan.coarse.device)
    return (plan.coarse - coarse).square().mean() + (
        w * (plan.residual - residual).square().mean(-1)
    ).sum()


def proposal_parameter_count(head):
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def active_nonvocab_count(head):
    return sum(
        p.numel()
        for n, p in head.named_parameters()
        if p.requires_grad and not n.startswith("dep_")
    )


def capacity_width(target_count, current_count, current_width, latent_width, arm):
    """Nearest multiple-of-16 hidden width; verify final actual count after constructing."""
    coefficient = (6 if arm == "C" else 8) * latent_width
    if arm not in {"C", "F-flat", "F-flat-aux"}:
        raise ValueError("only flat controls have adjustable width")
    return max(
        16,
        16 * round((current_width + (target_count - current_count) / coefficient) / 16),
    )
