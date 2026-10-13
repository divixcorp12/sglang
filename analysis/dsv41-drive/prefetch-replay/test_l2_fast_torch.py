"""Tests for l2_fast_torch: the folded T0/T3 head equals R1Model's forward, and select_cap1 matches r1_policy."""

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import b1_head  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_models  # noqa: E402
import r1_policy  # noqa: E402
from b1_rows import TOPK  # noqa: E402
from r1_groups import MAX_WIDTH, GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E = 16
RANK = 8
LAYERS = (19, 20, 21, 22)
STRATA = ("a", "b")
ROWS = (l2_fast_torch.FoldRow(source_layer=19, target_layer=21, stratum="b"),
        l2_fast_torch.FoldRow(source_layer=20, target_layer=22, stratum="a"))


def _model(*, t3: bool, adapter_rank: int = 0) -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=16, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.randn(r1_cost.X_DIM, generator=g) * 0.5,
                   components=torch.randn((RANK, r1_cost.X_DIM), generator=g) / r1_cost.X_DIM ** 0.5,
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=LAYERS, strata=STRATA))
    model = r1_models.R1Model(r1_models.R1Config(head=cfg, position=t3, aggregator=t3, adapter_rank=adapter_rank),
                              head=head)
    # Zero-initialised parts (pos_proj) and default embeddings would leave the folds unexercised.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model.eval()


def _gate() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return torch.randn((E, r1_cost.X_DIM), generator=g) * 0.02, torch.randn(E, generator=g) * 0.1


def _case(*, tokens: int, positions: list[int], seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((tokens, r1_cost.X_DIM), generator=g)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(tokens)])
    w = torch.rand((tokens, TOPK), generator=g)
    return x, ids, w, torch.tensor(positions, dtype=torch.int64)


def _reference(model, *, x, ids, w, positions, row: l2_fast_torch.FoldRow, W, bias) -> torch.Tensor:
    tokens = x.shape[0]
    with torch.no_grad():
        u = r1_cost._gate_features(x, ids, w, ref=model, W=W, bias=bias)
    u_pad = torch.zeros((1, MAX_WIDTH, u.shape[1]))
    u_pad[0, :tokens] = u
    live = np.zeros((1, MAX_WIDTH), dtype=bool)
    live[0, :tokens] = True
    pos = np.zeros((1, MAX_WIDTH), dtype=np.int64)
    pos[0, :tokens] = positions.numpy()
    vocab = model.head.vocab
    batch = GroupBatch(
        live=live, position=pos, width=np.array([tokens], dtype=np.float32),
        source_layer=np.array([vocab.layers.index(row.source_layer)]),
        target_layer=np.array([vocab.layers.index(row.target_layer)]),
        stratum=np.array([vocab.strata.index(row.stratum)]),
        miss=np.zeros((1, E), np.float32), available=np.ones((1, E), np.float32),
        route=np.zeros((1, MAX_WIDTH, E), np.float32), token_y=np.zeros((1, MAX_WIDTH, E), np.float32),
        hard_neg=np.zeros((1, MAX_WIDTH, E), np.float32))
    with torch.no_grad():
        return torch.sigmoid(model(ModelInput(batch=batch, u=u_pad, x=None, s_native=None)))[0]


@pytest.mark.parametrize("t3", [False, True], ids=["T0", "T3"])
@pytest.mark.parametrize("positions", [[0], [3], [0, 1, 2, 3, 4, 5], [0, 2, 5], [1, 4]],
                         ids=["w1-anchor", "w1-draft", "w6", "gaps", "no-anchor"])
def test_score_equals_reference_forward(t3, positions):
    """A fold that drops the PCA mean, an embedding, a position bias or a sparse block changes the block scores."""
    model = _model(t3=t3)
    W, bias = _gate()
    folded = l2_fast_torch.fold(model, rows=ROWS)
    x, ids, w, pos = _case(tokens=len(positions), positions=positions, seed=len(positions) + 10 * positions[0])
    scores = torch.full((MAX_WIDTH, E), 50.0)  # rows past the live width must never be read
    scores[:len(positions)] = r1_gate.full_gate_scores(x, W=W, bias=bias)
    for row in ROWS:
        want = _reference(model, x=x, ids=ids, w=w, positions=pos, row=row, W=W, bias=bias)
        with torch.no_grad():
            got = l2_fast_torch.score(folded, scores=scores, x=x, src_ids=ids, src_w=w, positions=pos,
                                      row=l2_fast_torch.row_index(folded, target_layer=row.target_layer))
        assert got.shape == (E,) and got.dtype == torch.float32
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


def test_fold_shapes_and_empty_t0_parts():
    """The kernel task reads these shapes; a T0 fold must leave the aggregator and position tables empty."""
    t0 = l2_fast_torch.fold(_model(t3=False), rows=ROWS)
    t3 = l2_fast_torch.fold(_model(t3=True), rows=ROWS)
    assert t3.w_in_sparse.shape == (4, E, 16) and t3.row_bias.shape == (2, 16)
    assert t3.pos_bias.shape == (MAX_WIDTH, MAX_WIDTH, 16)
    assert t3.agg_w1.shape == (32, 10) and t3.agg_row_bias.shape == (2, 32) and t3.agg_w2.shape == (32,)
    assert t3.agg_b2.shape == (1,)
    for empty in (t0.pos_bias, t0.agg_w1, t0.agg_row_bias, t0.agg_w2, t0.agg_b2):
        assert empty.numel() == 0
    assert l2_fast_torch.fold(_model(t3=False), rows=ROWS, dtype=torch.bfloat16).w_out.dtype == torch.bfloat16


def test_fold_refuses_adapter():
    """The adapter adds a per-target output term the fold does not carry; folding it would silently drop it."""
    with pytest.raises(ValueError, match="adapter"):
        l2_fast_torch.fold(_model(t3=False, adapter_rank=2), rows=ROWS)


def test_row_index_unknown_target_raises():
    """A target without a folded row must not fall back to another row's bias."""
    with pytest.raises(KeyError):
        l2_fast_torch.row_index(l2_fast_torch.fold(_model(t3=False), rows=ROWS), target_layer=19)


def _group(source: int, target: int, stratum: str):
    return SimpleNamespace(g=SimpleNamespace(source_layer=source, target_layer=target, target_stratum=stratum))


def test_serving_rows_sorted_unique_and_refuses_conflict():
    """D4 fixes one source and stratum per target row; a second one would make the precomputed bias wrong."""
    rows = l2_fast_torch.serving_rows([_group(20, 22, "a"), _group(19, 21, "b"), _group(20, 22, "a")])
    assert rows == ROWS
    with pytest.raises(ValueError):
        l2_fast_torch.serving_rows([_group(20, 22, "a"), _group(19, 22, "a")])
    with pytest.raises(ValueError):
        l2_fast_torch.serving_rows([_group(20, 22, "a"), _group(20, 22, "b")])


def test_select_cap1_tie_goes_to_lowest_id():
    """Equal best scores must read the lower expert id, as r1_policy orders (-score, id)."""
    scores = torch.tensor([0.1, 0.7, 0.3, 0.7])
    count, expert, value = l2_fast_torch.select_cap1(scores, skip=torch.zeros(4, dtype=torch.bool), threshold=0.5)
    assert (count.item(), expert.item(), value.item()) == (1, 1, pytest.approx(0.7))
    assert count.dtype == torch.int32 and expert.dtype == torch.int64 and value.dtype == torch.float32


def test_select_cap1_all_skipped_reads_nothing():
    """A skipped (hot or RAM-resident) expert is never read, even when it scores best."""
    count, expert, _ = l2_fast_torch.select_cap1(torch.ones(4), skip=torch.ones(4, dtype=torch.bool), threshold=0.0)
    assert (count.item(), expert.item()) == (0, -1)


def test_select_cap1_threshold_above_every_score_reads_nothing():
    """Below-threshold scores abstain rather than read the best one."""
    count, expert, value = l2_fast_torch.select_cap1(torch.full((4,), 0.4), skip=torch.zeros(4, dtype=torch.bool),
                                                     threshold=0.5)
    assert (count.item(), expert.item(), value.item()) == (0, -1, 0.0)


def test_select_cap1_agrees_with_policy():
    """select_cap1 is the read r1_policy.candidates + select(cap=1) makes, on rows with ties and skips."""
    g = torch.Generator().manual_seed(3)
    for _ in range(200):
        scores = torch.rand(E, generator=g).round(decimals=1)
        skip = torch.rand(E, generator=g) < 0.3
        thr = float(torch.rand(1, generator=g))
        count, expert, value = l2_fast_torch.select_cap1(scores, skip=skip, threshold=torch.tensor(thr))
        want = r1_policy.select(r1_policy.candidates(scores.numpy(), skip=skip.numpy()), threshold=thr, cap=1)
        assert count.item() == len(want)
        if len(want):
            assert expert.item() == int(want[0]) and value.item() == pytest.approx(float(scores[want[0]]))
