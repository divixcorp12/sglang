"""L+2 fast path, folded torch form: T0 / T3 (r1_models.R1Model, E5) scored from serving's precomputed gate scores.

fold() turns an R1Model into a FoldedHead: the PCA mean folded into fc_in / fc_gate biases, fc_in's sparse E5 column
blocks (route mask, route weight, gate score, gate margin) transposed so one expert is one row, and the source /
target / stratum embedding contributions precomputed into one bias row per serving (source, target) pair (D4). For T3
the position projection becomes a [width, position] bias table and the aggregator's embedding context a per-row
bias. score() is mathematically the reference forward: sigmoid of the block logit, from biased fp32 gate scores
(never the gate weight, D2), x and the source routes, with E5 applied as row gathers rather than dense scatters.

Inputs: an R1Model (r1_models.load_r1) and the serving rows (serving_rows over R1Groups at horizon 2).
Output: FoldedHead; score() gives f32 [E] block scores; select_cap1() the cap-1 read.
Usage: f = l2_fast_torch.fold(model, rows=l2_fast_torch.serving_rows(groups))
       p = l2_fast_torch.score(f, scores=s, x=x, src_ids=ids, src_w=w, positions=pos, row=l2_fast_torch.row_index(
           f, target_layer=22))
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

import msgspec
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_aggregator  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
from b1_rows import GATE_DEPTH, TOPK  # noqa: E402
from r1_groups import MAX_WIDTH, R1Group  # noqa: E402

N_SPARSE = 4  # E5 column blocks after PCA: route mask, route weight, gate score, gate margin


class FoldRow(msgspec.Struct, frozen=True):
    source_layer: int
    target_layer: int
    stratum: str  # the target stratum, as b1_head.make_batch indexes it


class FoldedHead(msgspec.Struct, frozen=True):
    w_pca: torch.Tensor  # [rank, 5120]
    w_in_pca: torch.Tensor  # [hidden, rank]
    b_in: torch.Tensor  # [hidden], PCA mean folded in
    w_in_sparse: torch.Tensor  # [4, E, hidden]: mask, route, score, margin
    row_bias: torch.Tensor  # [R, hidden], embedding columns of fc_in per row
    pos_bias: torch.Tensor  # [6 widths, 6 positions, hidden] for T3, else empty
    w_r1: torch.Tensor
    b_r1: torch.Tensor
    w_r2: torch.Tensor
    b_r2: torch.Tensor
    w_gate: torch.Tensor  # [hidden, rank]
    b_gate: torch.Tensor  # [hidden], PCA mean folded in
    w_out: torch.Tensor  # [E, hidden]
    b_out: torch.Tensor
    agg_w1: torch.Tensor  # [32, 10] for T3, else empty
    agg_row_bias: torch.Tensor  # [R, 32] for T3, else empty
    agg_w2: torch.Tensor  # [32] for T3, else empty
    agg_b2: torch.Tensor  # [1] for T3, else empty
    rows: tuple[FoldRow, ...]


def serving_rows(groups: Iterable[R1Group]) -> tuple[FoldRow, ...]:
    """Unique (source, target, stratum) rows sorted by target; one source and stratum per target (D4)."""
    by_target: dict[int, FoldRow] = {}
    for r in groups:
        row = FoldRow(source_layer=int(r.g.source_layer), target_layer=int(r.g.target_layer),
                      stratum=str(r.g.target_stratum))
        seen = by_target.setdefault(row.target_layer, row)
        if seen != row:
            raise ValueError(f"target layer {row.target_layer} has two rows: {seen} and {row}")
    return tuple(by_target[t] for t in sorted(by_target))


def row_index(folded: FoldedHead, *, target_layer: int) -> int:
    for i, row in enumerate(folded.rows):
        if row.target_layer == target_layer:
            return i
    raise KeyError(f"target layer {target_layer} is not a folded row")


def _check_foldable(model: r1_models.R1Model) -> None:
    if model.cfg.adapter_rank > 0:
        raise ValueError("fold does not support the target-layer adapter (adapter_rank > 0)")
    if model.cfg.head.variant != "E5":
        raise ValueError(f"fold needs an E5 head, got {model.cfg.head.variant!r}")


def _row_bias(model: r1_models.R1Model, *, rows: tuple[FoldRow, ...], w_emb: torch.Tensor) -> torch.Tensor:
    head = model.head
    layer_ix = {layer: i for i, layer in enumerate(head.vocab.layers)}
    stratum_ix = {s: i for i, s in enumerate(head.vocab.strata)}
    emb = torch.stack([torch.cat([head.source_emb.weight[layer_ix[r.source_layer]],
                                  head.target_emb.weight[layer_ix[r.target_layer]],
                                  head.stratum_emb.weight[stratum_ix[r.stratum]]]) for r in rows]).double()
    return emb @ w_emb.T


def _pos_bias(model: r1_models.R1Model) -> torch.Tensor:
    pos = torch.arange(MAX_WIDTH, device=model.device)
    feats = [torch.cat([model.pos_emb(pos).double(),
                        torch.full((MAX_WIDTH, 1), w / MAX_WIDTH, dtype=torch.float64, device=model.device)], dim=-1)
             for w in range(1, MAX_WIDTH + 1)]
    return torch.stack(feats) @ model.pos_proj.weight.double().T


def _agg_parts(model: r1_models.R1Model, *, rows: tuple[FoldRow, ...]) -> tuple[torch.Tensor, ...]:
    agg = model.aggregator
    n = r1_aggregator.N_BLOCK_FEATURES
    w1, b1 = agg.mlp[0].weight.double(), agg.mlp[0].bias.double()
    layer_ix = {layer: i for i, layer in enumerate(model.head.vocab.layers)}
    stratum_ix = {s: i for i, s in enumerate(model.head.vocab.strata)}
    ctx = torch.stack([torch.cat([agg.layer_emb.weight[layer_ix[r.target_layer]],
                                  agg.stratum_emb.weight[stratum_ix[r.stratum]]]) for r in rows]).double()
    return w1[:, :n], b1 + ctx @ w1[:, n:].T, agg.mlp[2].weight.double()[0], agg.mlp[2].bias.double()


def fold(model: r1_models.R1Model, *, rows: tuple[FoldRow, ...], dtype: torch.dtype = torch.float32) -> FoldedHead:
    """The folded head of ``model`` for ``rows``; folds are computed in fp64, then stored as ``dtype``."""
    _check_foldable(model)
    head, p = model.head, model.head.predictor
    rank, e = head.cfg.pca_rank, head.cfg.num_experts
    with torch.no_grad():
        w_pca = head.pca_components.double()
        mu_proj = w_pca @ head.pca_mean.double()
        w_in = p.fc_in.weight.double()
        w_in_pca = w_in[:, :rank]
        w_in_sparse = w_in[:, rank:rank + N_SPARSE * e].reshape(-1, N_SPARSE, e).permute(1, 2, 0)
        w_gate = p.fc_gate.weight.double()
        empty = torch.empty(0, dtype=torch.float64, device=model.device)
        pos_bias = _pos_bias(model) if model.pos_proj is not None else empty
        agg = _agg_parts(model, rows=rows) if model.aggregator is not None else (empty,) * 4
        parts = dict(
            w_pca=w_pca, w_in_pca=w_in_pca, b_in=p.fc_in.bias.double() - w_in_pca @ mu_proj,
            w_in_sparse=w_in_sparse, row_bias=_row_bias(model, rows=rows, w_emb=w_in[:, rank + N_SPARSE * e:]),
            pos_bias=pos_bias, w_r1=p.fc_r1.weight, b_r1=p.fc_r1.bias, w_r2=p.fc_r2.weight, b_r2=p.fc_r2.bias,
            w_gate=w_gate, b_gate=p.fc_gate.bias.double() - w_gate @ mu_proj, w_out=p.fc_out.weight,
            b_out=p.fc_out.bias, agg_w1=agg[0], agg_row_bias=agg[1], agg_w2=agg[2], agg_b2=agg[3])
        return FoldedHead(rows=tuple(rows), **{k: v.detach().to(dtype=dtype, copy=True).contiguous()
                                                     for k, v in parts.items()})


def _sparse_pre(folded: FoldedHead, *, scores: torch.Tensor, src_ids: torch.Tensor,
                src_w: torch.Tensor) -> torch.Tensor:
    """E5's sparse fc_in terms [T, hidden] as row gathers: route mask and weight, then the gate top-12 and margins."""
    w = folded.w_in_sparse.float()
    top = torch.sort(scores, dim=-1, descending=True, stable=True)
    top_vals, top_ids = top.values[:, :GATE_DEPTH], top.indices[:, :GATE_DEPTH]
    margin = top_vals - top_vals[:, TOPK - 1:TOPK]
    return (w[0][src_ids].sum(dim=1) + (w[1][src_ids] * src_w.float().unsqueeze(-1)).sum(dim=1)
            + (w[2][top_ids] * top_vals.unsqueeze(-1)).sum(dim=1) + (w[3][top_ids] * margin.unsqueeze(-1)).sum(dim=1))


def _token_logits(folded: FoldedHead, *, h: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    v = F.gelu(pre)
    r = F.linear(F.gelu(F.linear(v, folded.w_r1.float(), folded.b_r1.float())), folded.w_r2.float(),
                 folded.b_r2.float())
    g = torch.sigmoid(F.linear(h, folded.w_gate.float(), folded.b_gate.float()))
    return F.linear(v + g * r, folded.w_out.float(), folded.b_out.float())


def _block_logit(folded: FoldedHead, *, logits: torch.Tensor, positions: torch.Tensor,
                 row: int | torch.Tensor) -> torch.Tensor:
    live = torch.ones((1, logits.shape[0]), dtype=torch.bool, device=logits.device)
    if folded.agg_w1.numel() == 0:
        return r1_models._noisy_or_logit(logits.unsqueeze(0), live=live)[0]
    feats = r1_aggregator.block_features(logits.unsqueeze(0), live=live, position=positions.unsqueeze(0))[0]
    hid = F.gelu(F.linear(feats, folded.agg_w1.float(), folded.agg_row_bias[row].float()))
    return hid @ folded.agg_w2.float() + folded.agg_b2.float()


def score(folded: FoldedHead, *, scores: torch.Tensor, x: torch.Tensor, src_ids: torch.Tensor, src_w: torch.Tensor,
          positions: torch.Tensor, row: int | torch.Tensor) -> torch.Tensor:
    """f32 [E] sigmoid block scores of one group of x.shape[0] live tokens; graph-capturable, no host syncs.

    scores: biased fp32 gate scores [>= T, E] (only the first T rows are read); x [T, 5120]; src_ids int64 [T, k];
    src_w [T, k]; positions int64 [T] (0 = anchor); row: index into folded.rows.
    """
    tokens = x.shape[0]
    h = x.float() @ folded.w_pca.float().T
    pre = F.linear(h, folded.w_in_pca.float(), folded.b_in.float()) + folded.row_bias[row].float()
    if folded.pos_bias.numel() > 0:
        pre = pre + folded.pos_bias[tokens - 1][positions.clamp(0, MAX_WIDTH - 1)].float()
    pre = pre + _sparse_pre(folded, scores=scores[:tokens].float(), src_ids=src_ids, src_w=src_w)
    logits = _token_logits(folded, h=h, pre=pre)
    return torch.sigmoid(_block_logit(folded, logits=logits, positions=positions, row=row)).float()


def select_cap1(block_scores: torch.Tensor, *, skip: torch.Tensor, threshold: float | torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(count int32[1], expert int64[1], value f32[1]): the best non-skipped score >= threshold, ties to the lowest id.

    No read gives count 0, expert -1 and value 0. Graph-capturable.
    """
    expert = r1_cost.select_torch(block_scores, skip=skip, threshold=threshold, cap=1)
    hit = expert >= 0
    value = torch.where(hit, block_scores[expert.clamp_min(0)].float(), torch.zeros_like(block_scores[:1]).float())
    return hit.to(torch.int32), expert, value
