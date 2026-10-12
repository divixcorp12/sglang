"""B1 learned head: the E2 and E5 expert predictors (S1.2, D3).

One MiddlePredictor-shaped residual MLP per horizon, shared over source layers, with learned embeddings for source
layer, target layer and stratum. E2 reads PCA(x), the source route mask and weights; E5 adds the target gate's top-12
biased scores and their margins to the 6th, scattered dense. Labels are the target routes that are not VRAM-hot; the
hard-negative mask marks gate top-12 entries that were not routed. The PCA is fit on training-split rows only (R2).

Inputs: b1_rows.Group streams and an ``x_of(g)`` callable giving f32 [n, 5120].
Usage: import b1_head; pca = b1_head.fit_train_pca(train_groups, x_of=x_of, rank=256, seed=0)
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable, Iterable
from typing import Literal

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
import b1_rows  # noqa: E402
import router_score  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training import llapor  # noqa: E402
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats, fit_pca  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

TRAIN_SPLIT = "train"
_GATE_REF = 5  # the token's 6th gate score is the margin reference


class HeadConfig(msgspec.Struct, frozen=True):
    variant: Literal["E2", "E5"] = "E2"
    pca_rank: int = 256
    hidden: int = 512
    emb_dim: int = 16
    hard_neg_weight: float = 1.0
    num_experts: int = 384
    dropout: float = 0.1


class Vocab(msgspec.Struct, frozen=True):
    layers: tuple[int, ...]
    strata: tuple[str, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[b1_rows.Group]) -> Vocab:
        layers: set[int] = set()
        strata: set[str] = set()
        for g in rows:
            layers.update((g.source_layer, g.target_layer))
            strata.update((g.source_stratum, g.target_stratum))
        return cls(layers=tuple(sorted(layers)), strata=tuple(sorted(strata)))


class HeadBatch(msgspec.Struct):
    u: torch.Tensor  # f32 [n, feature_dim]: pca | mask | route (| gate score | gate margin for E5)
    source_layer: torch.Tensor  # int64 [n], index into Vocab.layers
    target_layer: torch.Tensor  # int64 [n]
    stratum: torch.Tensor  # int64 [n], the target stratum's index into Vocab.strata
    y: torch.Tensor  # f32 [n, E], 1 where the target routed a non-VRAM-hot expert
    hard_neg: torch.Tensor  # f32 [n, E], 1 where the gate top-12 holds an expert the target did not route


def _feature_dim(cfg: HeadConfig) -> int:
    dense = 4 if cfg.variant == "E5" else 2
    return cfg.pca_rank + dense * cfg.num_experts


def _take_priority(*, buf: torch.Tensor, keys: np.ndarray, filled: int, rows: torch.Tensor,
                   new_keys: np.ndarray) -> int:
    """Keep the max_rows rows with the largest random keys (a uniform sample without replacement)."""
    cap = buf.shape[0]
    room = min(cap - filled, len(new_keys))
    buf[filled:filled + room] = rows[:room]
    keys[filled:filled + room] = new_keys[:room]
    filled += room
    rows, new_keys = rows[room:], new_keys[room:]
    if len(new_keys) == 0:
        return filled
    # The buffer is full: the new rows replace the slots holding the smallest keys, where they beat them.
    all_keys = np.concatenate([keys[:cap], new_keys])
    keep = np.argpartition(-all_keys, cap - 1)[:cap]
    old_keep = np.sort(keep[keep < cap])
    new_keep = np.sort(keep[keep >= cap]) - cap
    free = np.setdiff1d(np.arange(cap), old_keep, assume_unique=True)
    buf[torch.from_numpy(free)] = rows[torch.from_numpy(new_keep)]
    keys[free] = new_keys[new_keep]
    return filled


def fit_train_pca(groups: Iterable[b1_rows.Group], *, x_of: Callable[[b1_rows.Group], np.ndarray], rank: int,
                  max_rows: int = 200_000, seed: int) -> PCAStats:
    """PCA over at most ``max_rows`` uniformly sampled training-split x rows; refuses any other split (C4)."""
    rng = np.random.default_rng(seed)
    buf: torch.Tensor | None = None
    keys = np.empty(max_rows, dtype=np.float64)
    filled = 0
    for g in groups:
        if g.split != TRAIN_SPLIT:
            raise ValueError(f"PCA is fit on train rows only, got a {g.split!r} group ({g.session})")
        x = np.asarray(x_of(g), dtype=np.float32)
        if len(x) == 0:
            continue
        if buf is None:
            # Stored as bf16: x came from bf16 words, so this is lossless and halves the buffer.
            buf = torch.empty((max_rows, x.shape[1]), dtype=torch.bfloat16)
        filled = _take_priority(buf=buf, keys=keys, filled=filled, rows=torch.from_numpy(x).to(torch.bfloat16),
                                new_keys=rng.random(len(x)))
    if buf is None or filled == 0:
        raise ValueError("no training rows to fit the PCA on")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return fit_pca(buf[:filled].float(), rank)


def fit_train_pca_from_shards(dataset_dir, *, sessions: list[str], rank: int, max_rows: int, seed: int) -> PCAStats:
    """PCA over at most ``max_rows`` x rows drawn uniformly without replacement from these sessions' train x shards.

    Reads only the train split (R2); x is stored once per session, so one fit serves every horizon.
    """
    maps = [b1_rows.read_x(dataset_dir, split=TRAIN_SPLIT, session=s) for s in sorted(sessions)]
    sizes = np.array([len(m) for m in maps], dtype=np.int64)
    total = int(sizes.sum())
    if total == 0:
        raise ValueError("no training rows to fit the PCA on")
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(total, size=min(max_rows, total), replace=False))
    starts = np.concatenate([[0], np.cumsum(sizes)])
    words = np.concatenate([np.asarray(m[picked[(picked >= a) & (picked < b)] - a])
                            for m, a, b in zip(maps, starts[:-1], starts[1:])])
    x = torch.from_numpy(router_score.bf16_to_f32(words))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return fit_pca(x, rank)


def _dense(ids: torch.Tensor, values: torch.Tensor, *, num_experts: int) -> torch.Tensor:
    # Padding ids (-1) land on expert 0 with value 0, so add rather than overwrite.
    out = torch.zeros((ids.shape[0], num_experts), dtype=torch.float32, device=ids.device)
    return out.scatter_add_(1, ids.clamp_min(0), values)


def _gate_dense(ids: torch.Tensor, scores: torch.Tensor, *, num_experts: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense gate score, margin to the 6th and top-12 membership; zeros where the target has no biased gate."""
    valid = (ids >= 0) & torch.isfinite(scores)
    zero = torch.zeros_like(scores)
    scores = torch.where(valid, scores, zero)
    margin = torch.where(valid, scores - scores[:, _GATE_REF:_GATE_REF + 1], zero)
    v = valid.float()
    return (_dense(ids, scores * v, num_experts=num_experts), _dense(ids, margin * v, num_experts=num_experts),
            _dense(ids, v, num_experts=num_experts))


def _stack(groups: list[b1_rows.Group], name: str, dtype, device) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([getattr(g, name) for g in groups]).astype(dtype)).to(device)


def make_batch(groups: list[b1_rows.Group], *, x_of: Callable[[b1_rows.Group], np.ndarray], pca: PCAStats,
               cfg: HeadConfig, vocab: Vocab) -> HeadBatch:
    """All groups' tokens as one batch; every tensor op runs once over the batch, on the PCA's device."""
    device, e = pca.mean.device, cfg.num_experts
    x = torch.from_numpy(np.concatenate([np.asarray(x_of(g), dtype=np.float32) for g in groups])).to(device)
    u = llapor.encode_features(x, _stack(groups, "source_ids", np.int64, device),
                               _stack(groups, "source_weights", np.float32, device), pca=pca, num_experts=e)
    score, margin, in_top = _gate_dense(_stack(groups, "gate_ids", np.int64, device),
                                        _stack(groups, "gate_scores", np.float32, device), num_experts=e)
    if cfg.variant == "E5":
        u = torch.cat([u, score, margin], dim=-1)
    target_ids = np.concatenate([g.target_ids for g in groups])
    kinds = np.concatenate([g.target_kinds for g in groups])
    tids = torch.from_numpy(target_ids.astype(np.int64)).to(device)
    labelled = torch.from_numpy((target_ids >= 0) & (kinds != KIND_VRAM_HOT)).float().to(device)
    routed = _dense(tids, torch.from_numpy(target_ids >= 0).float().to(device), num_experts=e).clamp_(max=1.0)

    counts = np.array([len(g.tokens) for g in groups], dtype=np.int64)
    layer_ix = {layer: i for i, layer in enumerate(vocab.layers)}
    stratum_ix = {s: i for i, s in enumerate(vocab.strata)}

    def index(values: list[int]) -> torch.Tensor:
        return torch.from_numpy(np.repeat(np.array(values, dtype=np.int64), counts)).to(device)

    return HeadBatch(
        u=u,
        source_layer=index([layer_ix[g.source_layer] for g in groups]),
        target_layer=index([layer_ix[g.target_layer] for g in groups]),
        stratum=index([stratum_ix[g.target_stratum] for g in groups]),
        y=_dense(tids, labelled, num_experts=e).clamp_(max=1.0),
        # Hard negatives are gate top-12 entries the target did not route at all (hot-routed ones are not).
        hard_neg=in_top * (1.0 - routed))


class B1Head(nn.Module):
    def __init__(self, cfg: HeadConfig, *, pca: PCAStats, vocab: Vocab):
        super().__init__()
        if pca.components.shape[0] != cfg.pca_rank:
            raise ValueError(f"PCA rank {pca.components.shape[0]} != cfg.pca_rank {cfg.pca_rank}")
        self.cfg = cfg
        self.vocab = vocab
        self.register_buffer("pca_mean", pca.mean.detach().clone())
        self.register_buffer("pca_components", pca.components.detach().clone())
        self.register_buffer("pca_variance", pca.explained_variance.detach().clone())
        self.source_emb = nn.Embedding(len(vocab.layers), cfg.emb_dim)
        self.target_emb = nn.Embedding(len(vocab.layers), cfg.emb_dim)
        self.stratum_emb = nn.Embedding(len(vocab.strata), cfg.emb_dim)
        self.predictor = llapor.MiddlePredictor(
            _feature_dim(cfg) + 3 * cfg.emb_dim, cfg.pca_rank, cfg.num_experts, hidden=cfg.hidden,
            dropout=cfg.dropout)

    @property
    def pca(self) -> PCAStats:
        return PCAStats(mean=self.pca_mean, components=self.pca_components, explained_variance=self.pca_variance)

    def inputs(self, batch: HeadBatch) -> torch.Tensor:
        return torch.cat([batch.u, self.source_emb(batch.source_layer), self.target_emb(batch.target_layer),
                          self.stratum_emb(batch.stratum)], dim=-1)

    def forward(self, batch: HeadBatch) -> torch.Tensor:
        return self.predictor(self.inputs(batch))


def head_loss(logits: torch.Tensor, batch: HeadBatch, *, cfg: HeadConfig) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, batch.y, reduction="none")
    weight = 1.0 + (cfg.hard_neg_weight - 1.0) * batch.hard_neg
    return (bce * weight).mean()


class HeadScorer:
    """Per-token sigmoid probabilities for the shared scoring function, aggregated there by noisy-or."""

    def __init__(self, model: B1Head, *, x_of: Callable[[b1_rows.Group], np.ndarray], name: str):
        self.model = model.eval()
        self.x_of = x_of
        self.name = name

    def __call__(self, g: b1_rows.Group) -> b1_rows.TokenScores:
        return self.score_many([g])[0]

    def score_many(self, groups: list[b1_rows.Group]) -> list[b1_rows.TokenScores]:
        """One forward over all ``groups``' tokens, split back per group in order."""
        batch = make_batch(groups, x_of=self.x_of, pca=self.model.pca, cfg=self.model.cfg, vocab=self.model.vocab)
        with torch.no_grad():
            p = torch.sigmoid(self.model(batch)).float().cpu().numpy()
        bounds = np.cumsum([0] + [len(g.tokens) for g in groups])
        return [b1_rows.TokenScores(scores=p[a:b], rule="noisy_or") for a, b in zip(bounds[:-1], bounds[1:])]


def head_bytes_bf16(model: B1Head) -> int:
    """Parameters plus the PCA buffers, at 2 bytes each."""
    count = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    return 2 * count


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_head(path, model: B1Head) -> str:
    """Write config, vocab and state to ``path``; returns the file's sha256."""
    path = os.fspath(path)
    payload = {
        "cfg": msgspec.to_builtins(model.cfg),
        "vocab": msgspec.to_builtins(model.vocab),
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return _file_sha256(path)


def load_head(path) -> tuple[B1Head, str]:
    path = os.fspath(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = msgspec.convert(payload["cfg"], HeadConfig)
    vocab = msgspec.convert(payload["vocab"], Vocab)
    state = payload["state"]
    pca = PCAStats(mean=state["pca_mean"], components=state["pca_components"],
                   explained_variance=state["pca_variance"])
    model = B1Head(cfg, pca=pca, vocab=vocab)
    model.load_state_dict(state)
    return model.eval(), _file_sha256(path)
