"""GPU tests for l2_fast_kernels: the fused spec_head kernels agree with the folded torch path on T0-like / T3-like heads.

The reference is l2_fast_torch.score on a fold whose matrices are rounded to bf16, the kernels' storage precision (D3).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import msgspec  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import b1_head  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_aggregator  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
from b1_rows import TOPK  # noqa: E402
from r1_groups import MAX_WIDTH  # noqa: E402

from sglang.kernels.ops.moe import spec_head  # noqa: E402
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="spec_head kernels need CUDA")

E = 384
RANK = 128
LAYERS = (19, 20, 21, 22)
STRATA = ("a", "b")
ROWS = (l2_fast_torch.FoldRow(source_layer=19, target_layer=21, stratum="b"),
        l2_fast_torch.FoldRow(source_layer=20, target_layer=22, stratum="a"))
DEV = "cuda"
T0_T3 = pytest.mark.parametrize("t3", [False, True], ids=["T0", "T3"])


def _model(*, t3: bool, tie: tuple[int, int] | None = None) -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=512, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.randn(r1_cost.X_DIM, generator=g) * 0.5,
                   components=torch.randn((RANK, r1_cost.X_DIM), generator=g) / r1_cost.X_DIM ** 0.5,
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=LAYERS, strata=STRATA))
    model = r1_models.R1Model(r1_models.R1Config(head=cfg, position=t3, aggregator=t3), head=head)
    # Nonzero everywhere (pos_proj and the embeddings start at zero) so every fold is exercised.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        if tie is not None:
            fc_out = model.head.predictor.fc_out
            fc_out.weight[tie[1]] = fc_out.weight[tie[0]]
            fc_out.bias[list(tie)] = 4.0
    return model.eval()


def _folded(*, t3: bool, tie: tuple[int, int] | None = None) -> l2_fast_torch.FoldedHead:
    return l2_fast_torch.fold(_model(t3=t3, tie=tie), rows=ROWS)


def _rounded(folded: l2_fast_torch.FoldedHead) -> l2_fast_torch.FoldedHead:
    """``folded`` on the device with its matrices rounded through bf16, as the kernels store them."""
    parts = {}
    for name in folded.__struct_fields__:
        if name == "rows":
            continue
        t = getattr(folded, name)
        parts[name] = (t.bfloat16().float() if name in l2_fast_kernels.MATRIX_FIELDS else t).to(DEV)
    return msgspec.structs.replace(folded, **parts)


class _Case(msgspec.Struct):
    scores: torch.Tensor
    x: torch.Tensor
    ids: torch.Tensor
    w: torch.Tensor
    positions: torch.Tensor


def _case(positions: list[int], seed: int = 1) -> _Case:
    tokens = len(positions)
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(tokens)])
    scores = torch.randn((MAX_WIDTH, E), generator=g)
    return _Case(scores=scores.to(DEV), x=torch.randn((tokens, r1_cost.X_DIM), generator=g).bfloat16().to(DEV),
                 ids=ids.to(DEV), w=torch.rand((tokens, TOPK), generator=g).to(DEV),
                 positions=torch.tensor(positions, dtype=torch.int64, device=DEV))


def _ref(ref: l2_fast_torch.FoldedHead, c: _Case, *, row: int) -> torch.Tensor:
    with torch.no_grad():
        return l2_fast_torch.score(ref, scores=c.scores, x=c.x, src_ids=c.ids, src_w=c.w, positions=c.positions,
                                   row=row)


def _kernel_args(c: _Case, *, row: int, skip: torch.Tensor, threshold) -> dict:
    return dict(scores=c.scores, x=c.x, src_ids=c.ids, src_w=c.w, positions=c.positions, row=row, skip=skip,
                threshold=threshold)


def _no_skip() -> torch.Tensor:
    return torch.zeros(E, dtype=torch.bool, device=DEV)


@T0_T3
@pytest.mark.parametrize("positions", [[0], [0, 1, 2, 3, 4, 5], [0, 2, 5]], ids=["w1", "w6", "gaps"])
def test_block_scores_and_pick_match_folded_torch(t3, positions):
    """A kernel that drops a fold term, a tail stage or the width padding moves block scores off the torch path."""
    folded = _folded(t3=t3)
    head = l2_fast_kernels.KernelHead(folded, device=DEV)
    ref_folded = _rounded(folded)
    c = _case(positions, seed=len(positions))
    for row in (0, 1):
        want = _ref(ref_folded, c, row=row)
        count, expert, value, got = head.score(**_kernel_args(c, row=row, skip=_no_skip(), threshold=0.0))
        torch.testing.assert_close(got, want, atol=1e-3, rtol=0)
        top = torch.topk(want, 2)
        assert count == 1
        if float(top.values[0] - top.values[1]) > 1e-3:
            assert expert == int(top.indices[0])
        assert value == pytest.approx(float(got[expert]), abs=1e-6)


@T0_T3
def test_tie_goes_to_lowest_expert_id(t3):
    """Two experts with identical output rows and biases score equal; the read must be the lower id."""
    head = l2_fast_kernels.KernelHead(_folded(t3=t3, tie=(300, 17)), device=DEV)
    count, expert, _, got = head.score(**_kernel_args(_case([0, 1, 2]), row=0, skip=_no_skip(), threshold=0.0))
    if got[17] == got[300]:
        assert (count, expert) == (1, 17)
    else:  # T3's block features may rank the two apart; the pick must still be the best score
        assert expert == int(got.argmax())


@T0_T3
def test_all_skipped_reads_nothing(t3):
    """Skipped experts are never read, even when they score best; count 0 must carry expert -1."""
    head = l2_fast_kernels.KernelHead(_folded(t3=t3), device=DEV)
    skip = torch.ones(E, dtype=torch.bool, device=DEV)
    count, expert, value, _ = head.score(**_kernel_args(_case([0, 1]), row=0, skip=skip, threshold=0.0))
    assert (count, expert, value) == (0, -1, 0.0)


@T0_T3
def test_threshold_above_every_score_reads_nothing(t3):
    """A best score below the threshold abstains rather than reading the best expert."""
    head = l2_fast_kernels.KernelHead(_folded(t3=t3), device=DEV)
    count, expert, _, _ = head.score(**_kernel_args(_case([0]), row=0, skip=_no_skip(), threshold=2.0))
    assert (count, expert) == (0, -1)


@T0_T3
@pytest.mark.parametrize("mask", ["hot_slots", "ram_slot"])
def test_hot_and_ram_masks_move_the_pick_to_the_next_expert(t3, mask):
    """Either skip source alone must hide the winner; wiring one to the wrong buffer would still read it."""
    head = l2_fast_kernels.KernelHead(_folded(t3=t3), device=DEV)
    c = _case([0, 1, 2])
    args = _kernel_args(c, row=0, skip=_no_skip(), threshold=0.0)
    _, winner, _, block = head.score(**args)
    order = torch.argsort(block, descending=True, stable=True)
    assert int(order[0]) == winner
    hot = torch.full((4,), -1, dtype=torch.int64, device=DEV)
    ram = torch.full((3, E), -1, dtype=torch.int32, device=DEV)
    if mask == "hot_slots":
        hot[2] = winner
    else:
        ram[2, winner] = 5
    threshold = torch.zeros(1, device=DEV)
    spec_head.run_spec_head(weights=head.weights, workspace=head.workspace, x=c.x, scores=c.scores,
                            src_ids=c.ids, src_w=c.w, positions=c.positions, row=0, hot_slots=hot, hot_capacity=4,
                            ram_slot=ram[2:], target=0, threshold=threshold)
    count, expert, _ = spec_head.read_record(head.workspace)
    assert (count, expert) == (1, int(order[1]))


def test_tail_features_equal_block_features():
    """T3's exported block features are the aggregator's input; a layout drift here shifts every T3 score."""
    folded = _folded(t3=True)
    head = l2_fast_kernels.KernelHead(folded, device=DEV, features=True)
    g = torch.Generator().manual_seed(5)
    positions = torch.tensor([0, 1, 3, 4], dtype=torch.int64, device=DEV)
    logits = (torch.randn((4, E), generator=g) * 2).to(DEV)
    threshold = torch.zeros(1, device=DEV)
    spec_head.run_spec_head_tail(
        weights=head.weights, workspace=head.workspace, logits=logits, positions=positions, row=0,
        hot_slots=head.hot_slots, hot_capacity=0, ram_slot=head.ram_slot, target=0, threshold=threshold)
    live = torch.ones((1, 4), dtype=torch.bool, device=DEV)
    want = r1_aggregator.block_features(logits.unsqueeze(0), live=live, position=positions.unsqueeze(0))[0]
    torch.testing.assert_close(head.workspace.features, want, atol=1e-4, rtol=1e-4)


@T0_T3
def test_graph_replay_reproduces_the_record(t3):
    """The launch must hold no host read or per-call allocation address a captured graph would freeze wrongly."""
    head = l2_fast_kernels.KernelHead(_folded(t3=t3), device=DEV)
    c = _case([0, 1, 2, 3])
    skip = _no_skip()
    threshold = torch.zeros(1, device=DEV)
    args = _kernel_args(c, row=1, skip=skip, threshold=threshold)
    head.launch(**args)
    want = spec_head.read_record(head.workspace)
    want_block = head.workspace.block_scores.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head.launch(**args)
    head.workspace.record.zero_()
    head.workspace.block_scores.zero_()
    graph.replay()
    assert spec_head.read_record(head.workspace) == want
    torch.testing.assert_close(head.workspace.block_scores, want_block, atol=0, rtol=0)
