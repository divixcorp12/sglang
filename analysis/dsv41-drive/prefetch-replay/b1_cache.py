"""B1 split cache: one split's head inputs built once and held on the device, for training and dev scoring.

Training re-reading shards and rebuilding every batch each epoch cost minutes per epoch; here make_batch runs once
per (split, horizon, PCA rank) and epochs only index the cached rows. The cache holds the E5 layout
(pca | mask | route | gate score | gate margin); E2 is its first pca_rank + 2 * E columns. Per group it also holds
the skip mask (hot or in RAM) and eligible rows, so dev recall at 1 read is computed on the device for early stopping.
The selection tables stay b1_score's (CachedHeadScorer feeds it the cached probabilities).

Inputs: a b1_rows.Group stream, x_of, a fitted PCA and the Vocab the model uses.
Usage: cache = b1_cache.build(groups, x_of=x_of, pca=pca, vocab=vocab, num_experts=384)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import b1_score  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

_BUILD_ROWS = 16_384
_PREDICT_ROWS = 65_536


class SplitCache(msgspec.Struct):
    batch: b1_head.HeadBatch  # every row, E5 layout; y and hard_neg stored as bool
    pca_rank: int
    num_experts: int
    row_group: torch.Tensor  # int64 [rows], the row's group index
    group_start: np.ndarray  # int64 [groups + 1], each group's first row
    keys: dict[tuple[str, int, int], int]  # (session, seq, source_row) -> group index
    skip: torch.Tensor  # bool [groups, E]: VRAM-hot or in RAM, never read
    eligible: torch.Tensor  # bool [groups, E]

    @property
    def rows(self) -> int:
        return int(self.row_group.shape[0])


def _key(g: b1_rows.Group) -> tuple[str, int, int]:
    return (g.session, g.seq, g.source_row)


def _group_masks(g: b1_rows.Group, *, num_experts: int) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.zeros(num_experts, dtype=bool)
    eligible[b1_rows.eligible(g)] = True
    return b1_score._skip_mask(g, num_experts=num_experts), eligible


def _chunked(groups: Iterable[b1_rows.Group], *, rows: int) -> Iterable[list[b1_rows.Group]]:
    chunk, n = [], 0
    for g in groups:
        chunk.append(g)
        n += len(g.tokens)
        if n >= rows:
            yield chunk
            chunk, n = [], 0
    if chunk:
        yield chunk


def build(groups: Iterable[b1_rows.Group], *, x_of: Callable[[b1_rows.Group], np.ndarray], pca: PCAStats,
          vocab: b1_head.Vocab, num_experts: int) -> SplitCache:
    """make_batch over every group (E5 layout), on the PCA's device; one pass over the shards."""
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=int(pca.components.shape[0]), num_experts=num_experts)
    parts, sizes, keys, skips, eligibles = [], [], {}, [], []
    for chunk in _chunked(groups, rows=_BUILD_ROWS):
        b = b1_head.make_batch(chunk, x_of=x_of, pca=pca, cfg=cfg, vocab=vocab)
        parts.append(msgspec.structs.replace(b, y=b.y.bool(), hard_neg=b.hard_neg.bool()))
        for g in chunk:
            keys[_key(g)] = len(sizes)
            sizes.append(len(g.tokens))
            skip, eligible = _group_masks(g, num_experts=num_experts)
            skips.append(skip)
            eligibles.append(eligible)
    if not parts:
        raise ValueError("no groups to cache")
    device = pca.mean.device
    batch = b1_head.HeadBatch(**{f: torch.cat([getattr(p, f) for p in parts]) for f in b1_head.HeadBatch.__struct_fields__})
    counts = np.array(sizes, dtype=np.int64)
    return SplitCache(
        batch=batch, pca_rank=cfg.pca_rank, num_experts=num_experts,
        row_group=torch.from_numpy(np.repeat(np.arange(len(counts)), counts)).to(device),
        group_start=np.concatenate([[0], np.cumsum(counts)]), keys=keys,
        skip=torch.from_numpy(np.stack(skips)).to(device), eligible=torch.from_numpy(np.stack(eligibles)).to(device))


def take(cache: SplitCache, idx: torch.Tensor, *, variant: str) -> b1_head.HeadBatch:
    """Rows ``idx`` as a training batch for ``variant``; E2 drops the gate score and margin columns."""
    b = cache.batch
    u = b.u[idx]
    if variant == "E2":
        u = u[:, :cache.pca_rank + 2 * cache.num_experts]
    return b1_head.HeadBatch(u=u, source_layer=b.source_layer[idx], target_layer=b.target_layer[idx],
                             stratum=b.stratum[idx], y=b.y[idx].float(), hard_neg=b.hard_neg[idx].float())


def predict(model: b1_head.B1Head, cache: SplitCache) -> torch.Tensor:
    """Per-row sigmoid probabilities f32 [rows, E], on the cache's device."""
    out = []
    with torch.no_grad():
        for a in range(0, cache.rows, _PREDICT_ROWS):
            idx = torch.arange(a, min(a + _PREDICT_ROWS, cache.rows), device=cache.row_group.device)
            out.append(torch.sigmoid(model(take(cache, idx, variant=model.cfg.variant))).float())
    return torch.cat(out)


def recall_at_1(cache: SplitCache, p: torch.Tensor) -> float:
    """Pooled eligible-row recall at 1 read, b1_score's rule: noisy-or over the group's tokens, skip, then top 1.

    float64 like b1_score: saturated p == 1 gives log1p(-1) = -inf and a value of exactly 1.0, so ties at 1.0 go to
    the lowest id there too; a float32 1 - exp(.) would split them.
    """
    log_miss = torch.zeros((len(cache.group_start) - 1, cache.num_experts), dtype=torch.float64, device=p.device)
    log_miss.index_add_(0, cache.row_group, torch.log1p(-p.double().clamp(0.0, 1.0)))
    value = torch.where(cache.skip, torch.tensor(-torch.inf, dtype=torch.float64, device=p.device),
                        1.0 - torch.exp(log_miss))
    best, top = value.max(dim=1)  # max returns the first maximum: ties go to the lowest expert id
    read = torch.isfinite(best)
    useful = cache.eligible.gather(1, top[:, None]).squeeze(1) & read
    total = int(cache.eligible.sum())
    return float(useful.sum()) / total if total else 0.0


class CachedHeadScorer:
    """b1_score scorer whose per-group probabilities come from one predict() over the cache."""

    def __init__(self, model: b1_head.B1Head, *, cache: SplitCache, name: str):
        self.name = name
        self.cache = cache
        self.p = predict(model, cache).cpu().numpy()

    def __call__(self, g: b1_rows.Group) -> b1_rows.TokenScores:
        i = self.cache.keys[_key(g)]
        a, b = self.cache.group_start[i], self.cache.group_start[i + 1]
        return b1_rows.TokenScores(scores=self.p[a:b], rule="noisy_or")

    def score_many(self, groups: list[b1_rows.Group]) -> list[b1_rows.TokenScores]:
        return [self(g) for g in groups]
