"""R1 model family: the B1 E5 head over group batches, with optional position, adapters and block aggregator.

R1Model wraps a B1Head (copied, never shared) and runs it per token over a ModelInput [G,6,F]. Optional parts, each
zero-effect at step 0 so a child starts at its parent's outputs: an 8-wide position embedding plus the live width
scalar (through a zero-initialised projection), a rank-r target-layer residual output adapter (U and b zero), and
the BlockAggregator. With the aggregator off the block logit is the noisy-or of the live tokens' logits.

Inputs: ModelInput from r1_cache.collate (padding rows are zeros with live=False); checkpoints written by save_r1.
Usage: m = r1_models.R1Model.from_parent(cfg, parent=f0); loss = ...; r1_models.save_r1(p, m, parent_sha=s, arm="T3")
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys
from collections.abc import Callable

import msgspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
from r1_aggregator import BlockAggregator  # noqa: E402
from r1_groups import MAX_WIDTH, GroupBatch, ModelInput, R1Group  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

_TINY = 1e-12


class R1Config(msgspec.Struct, frozen=True):
    head: b1_head.HeadConfig
    position: bool = False
    adapter_rank: int = 0  # 0 = no adapter
    aggregator: bool = False
    pos_dim: int = 8


def _noisy_or_logit(token_logits: torch.Tensor, *, live: torch.Tensor) -> torch.Tensor:
    """logit(1 - prod(1 - sigmoid(l))) over live positions = s + log(1 - exp(-s)), s = sum softplus(l)."""
    liv = live.unsqueeze(-1)
    s = torch.where(liv, F.softplus(token_logits), torch.zeros_like(token_logits)).sum(dim=1).clamp_min(_TINY)
    return s + torch.log(-torch.expm1(-s))


class R1Model(nn.Module):
    def __init__(self, cfg: R1Config, *, head: b1_head.B1Head):
        super().__init__()
        if head.cfg != cfg.head:
            raise ValueError("head config differs from R1Config.head")
        self.cfg = cfg
        self.head = head
        hidden = head.cfg.hidden
        e = head.cfg.num_experts
        self.pos_emb: nn.Embedding | None = None
        self.pos_proj: nn.Linear | None = None
        if cfg.position:
            self.pos_emb = nn.Embedding(MAX_WIDTH, cfg.pos_dim)
            self.pos_proj = nn.Linear(cfg.pos_dim + 1, hidden, bias=False)
            nn.init.zeros_(self.pos_proj.weight)
        self.adapter_a: nn.Parameter | None = None
        self.adapter_u: nn.Parameter | None = None
        self.adapter_b: nn.Parameter | None = None
        if cfg.adapter_rank > 0:
            n_layers = len(head.vocab.layers)
            self.adapter_a = nn.Parameter(torch.randn(hidden, cfg.adapter_rank) / hidden**0.5)
            self.adapter_u = nn.Parameter(torch.zeros(n_layers, e, cfg.adapter_rank))
            self.adapter_b = nn.Parameter(torch.zeros(n_layers, e))
        self.aggregator: BlockAggregator | None = None
        if cfg.aggregator:
            self.aggregator = BlockAggregator(n_layers=len(head.vocab.layers), n_strata=len(head.vocab.strata),
                                              emb_dim=head.cfg.emb_dim)

    @classmethod
    def from_parent(cls, cfg: R1Config, *, parent: R1Model) -> R1Model:
        """A deep copy of the parent's head and every part the child shares with it, plus the child's new parts."""
        model = cls(cfg, head=copy.deepcopy(parent.head))
        own = model.state_dict()
        shared = {k: v.detach().clone() for k, v in parent.state_dict().items()
                  if k in own and own[k].shape == v.shape}
        model.load_state_dict(shared, strict=False)
        return model

    @property
    def device(self) -> torch.device:
        return self.head.pca_mean.device

    def _index(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, device=self.device)

    def _encoder_input(self, inp: ModelInput) -> torch.Tensor:
        """[G,6,F + 3 emb]: E5 features then the source, target and stratum embeddings of the group."""
        b = inp.batch
        emb = torch.cat([self.head.source_emb(self._index(b.source_layer)),
                         self.head.target_emb(self._index(b.target_layer)),
                         self.head.stratum_emb(self._index(b.stratum))], dim=-1)
        return torch.cat([inp.u, emb.unsqueeze(1).expand(-1, inp.u.shape[1], -1)], dim=-1)

    def _position_features(self, batch: GroupBatch) -> torch.Tensor:
        pos = self._index(batch.position).clamp(0, MAX_WIDTH - 1)
        width = self._index(batch.width).unsqueeze(-1).expand(-1, pos.shape[1]) / MAX_WIDTH
        return torch.cat([self.pos_emb(pos), width.unsqueeze(-1)], dim=-1)

    def shared_repr(self, inp: ModelInput) -> torch.Tensor:
        """The predictor's pre-output hidden state v + g * r, f32 [G,6,hidden]."""
        p = self.head.predictor
        x = self._encoder_input(inp)
        pre = p.fc_in(x)
        if self.pos_proj is not None:
            pre = pre + self.pos_proj(self._position_features(inp.batch))
        v = F.gelu(pre)
        r = p.fc_r2(p.dropout(F.gelu(p.fc_r1(v))))
        g = torch.sigmoid(p.fc_gate(x[..., : p.pca_dim]))
        return v + g * r

    def token_logits(self, inp: ModelInput) -> torch.Tensor:
        """Per-token expert logits f32 [G,6,E]; padding rows hold values nobody may read (mask by live)."""
        h = self.shared_repr(inp)
        out = self.head.predictor.fc_out(self.head.predictor.dropout(h))
        if self.adapter_a is not None:
            layer = self._index(inp.batch.target_layer)
            delta = torch.bmm(h @ self.adapter_a, self.adapter_u[layer].transpose(1, 2))
            out = out + delta + self.adapter_b[layer].unsqueeze(1)
        return out

    def forward(self, inp: ModelInput) -> torch.Tensor:
        """Block logits f32 [G,E]."""
        logits = self.token_logits(inp)
        if self.aggregator is not None:
            return self.aggregator(logits, batch=inp.batch)
        return _noisy_or_logit(logits, live=self._index(inp.batch.live).bool())


def _matches(name: str, prefixes: set[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def param_groups(model: R1Model, *, new: set[str], lr_new: float, lr_inherited: float, weight_decay: float
                 ) -> list[dict]:
    """AdamW groups of the trainable parameters: new (by name or name prefix) vs inherited; no decay on biases."""
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets.setdefault((_matches(name, new), p.ndim >= 2), []).append(p)
    groups = []
    for (is_new, decay), params in sorted(buckets.items()):
        groups.append({"params": params, "lr": lr_new if is_new else lr_inherited,
                       "weight_decay": weight_decay if decay else 0.0})
    return groups


def freeze(model: R1Model, *, encoder: bool, embeddings: bool) -> None:
    """encoder: the head's predictor, position projection and adapter; embeddings: layer/stratum/position tables.

    The block aggregator is never frozen here.
    """
    parts: list[nn.Module | nn.Parameter | None] = []
    if encoder:
        parts += [model.head.predictor, model.pos_proj, model.adapter_a, model.adapter_u, model.adapter_b]
    if embeddings:
        parts += [model.head.source_emb, model.head.target_emb, model.head.stratum_emb, model.pos_emb]
    for part in parts:
        if part is not None:
            part.requires_grad_(False)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_r1(path, model: R1Model, *, parent_sha: str, arm: str) -> str:
    """Write config, vocab, state and the parent's sha256 to ``path``; returns the file's sha256."""
    path = os.fspath(path)
    payload = {
        "cfg": msgspec.to_builtins(model.cfg),
        "vocab": msgspec.to_builtins(model.head.vocab),
        "parent_sha": parent_sha,
        "arm": arm,
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return _file_sha256(path)


def load_r1(path) -> tuple[R1Model, str, str]:
    """(model in eval mode, file sha256, parent sha256)."""
    path = os.fspath(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = msgspec.convert(payload["cfg"], R1Config)
    vocab = msgspec.convert(payload["vocab"], b1_head.Vocab)
    state = payload["state"]
    pca = PCAStats(mean=state["head.pca_mean"], components=state["head.pca_components"],
                   explained_variance=state["head.pca_variance"])
    model = R1Model(cfg, head=b1_head.B1Head(cfg.head, pca=pca, vocab=vocab))
    model.load_state_dict(state)
    return model.eval(), _file_sha256(path), payload["parent_sha"]


def _live_counts(groups: list[R1Group]) -> list[int]:
    return [len(r.g.tokens) for r in groups]


class R1GroupScorer:
    """GroupScorer: sigmoid of the block logit where the expert is available, -inf elsewhere."""

    def __init__(self, model: R1Model, *, collate_fn: Callable[[list[R1Group]], ModelInput], name: str):
        self.model = model.eval()
        self.collate_fn = collate_fn
        self.name = name

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        with torch.no_grad():
            p = torch.sigmoid(self.model(self.collate_fn(groups))).float().cpu().numpy()
        out = []
        for row, r in zip(p, groups):
            out.append(np.where(r.labels.available, row, -np.inf).astype(np.float32))
        return out


class R1TokenScorer:
    """Per-token sigmoid scores f32 [n,384] for the token-pooling controls (pooled once, in r1_policy)."""

    def __init__(self, model: R1Model, *, collate_fn: Callable[[list[R1Group]], ModelInput], name: str):
        self.model = model.eval()
        self.collate_fn = collate_fn
        self.name = name

    def score_tokens(self, groups: list[R1Group]) -> list[np.ndarray]:
        with torch.no_grad():
            p = torch.sigmoid(self.model.token_logits(self.collate_fn(groups))).float().cpu().numpy()
        # Live rows are the group's tokens in order, packed first.
        return [p[i, :n] for i, n in enumerate(_live_counts(groups))]
