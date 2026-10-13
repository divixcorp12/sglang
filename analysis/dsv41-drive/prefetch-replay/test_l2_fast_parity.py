"""l2_fast_parity: frozen-threshold extraction, refusals (locked_test, guarded --out) on CPU, and on CUDA the reference,
folded torch and fused kernel paths agreeing on every read of synthetic groups at a threshold away from the scores."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import l2_fast_parity  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_cache  # noqa: E402
import r1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
from b1_rows import GATE_DEPTH, TOPK  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="spec_score / spec_head kernels need CUDA")

E = r1_groups.NUM_EXPERTS
RANK = 128
LAYERS = (19, 20, 21, 22)
STRATA = ("a", "b")
SOURCE, TARGET = 19, 21
WINNER = 5
THR = 0.75
DEV = "cuda"
ROOT = "/mnt/nvme1/nvfp4-work"


def _f109(**calib_thr):
    def budgets(thr):
        return {"budgets": {b: {"point": {"threshold": thr[i]}} for i, b in enumerate(("0.25", "0.5", "0.75", "1.0"))}}

    calib = {name: budgets(t) for name, t in calib_thr.items()}
    dev = {name: budgets([9.0] * 4) for name in calib_thr}
    return {"cap1": {"splits": {"dev": dev, "calib": calib}}}


def test_frozen_thresholds_take_calib_budgets_as_recorded():
    # Catches reading dev's thresholds, the wrong budgets, or turning an infeasible (None) budget into a number.
    f109 = _f109(L2_T0=[0.1, 0.2, 0.3, 0.4], L2_T3=[0.5, 0.6, 0.7, None], L2_T0_best_cap1=[0.0] * 4)
    assert l2_fast_parity.frozen_thresholds(f109) == {"L2_T0": {"0.75": 0.3, "1.0": 0.4},
                                                      "L2_T3": {"0.75": 0.7, "1.0": None}}


def test_locked_test_is_refused_before_any_read(tmp_path):
    # Catches a split loop that could reach the locked test split (R2); the dataset dir need not exist.
    with pytest.raises(PermissionError):
        l2_fast_parity._read_split(tmp_path / "missing", split=b1_rows.LOCKED_SPLIT,
                                   exclusions=r1_groups.Exclusions.empty(), limit=None)


@pytest.mark.parametrize("out, run_dir", [
    (f"{ROOT}/f109-l2-cap1-20261013/parity.json", f"{ROOT}/b1-r1-20261012"),
    (f"{ROOT}/some-run/f123/parity.json", f"{ROOT}/some-run"),
    ("/tmp/f123/parity.json", f"{ROOT}/b1-r1-20261012"),
], ids=["f109-dir", "inside-run-dir", "outside-root"])
def test_guarded_out_is_refused_with_exit_2(out, run_dir):
    # Catches a parity run writing into F109's frozen outputs, the R1 run dir, or outside the work root.
    argv = [run_dir, "--dataset", "d", "--corpus", "c", "--b1-head", "h", "--model-path", "m",
            "--f109", f"{ROOT}/f109-l2-cap1-20261013", "--out", out]
    assert l2_fast_parity.main(argv) == 2


# ===== CUDA: synthetic groups and models =====


def _model(*, t3: bool) -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=512, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.randn(r1_cost.X_DIM, generator=g) * 0.5,
                   components=torch.randn((RANK, r1_cost.X_DIM), generator=g) / r1_cost.X_DIM ** 0.5,
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=LAYERS, strata=STRATA))
    model = r1_models.R1Model(r1_models.R1Config(head=cfg, position=t3, aggregator=t3), head=head)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        # WINNER's token logit is 6 and every other's about -6, so block scores sit far from THR.
        fc_out = model.head.predictor.fc_out
        fc_out.weight[WINNER] = 0.0
        fc_out.bias.fill_(-6.0)
        fc_out.bias[WINNER] = 6.0
        if t3:  # aggregator = gelu(mean token logit): monotone, so WINNER stays on top
            agg = model.aggregator.mlp
            for layer in (agg[0], agg[2]):
                layer.weight.zero_()
                layer.bias.zero_()
            agg[0].weight[0, 5] = 1.0
            agg[2].weight[0, 0] = 1.0
    return model.eval().to(DEV)


def _gate() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return ((torch.randn((E, r1_cost.X_DIM), generator=g) / r1_cost.X_DIM ** 0.5).to(DEV),
            (torch.randn(E, generator=g) * 0.1).to(DEV))


def _top12(scores: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    top = torch.sort(scores, dim=-1, descending=True, stable=True)
    return (top.indices[:, :GATE_DEPTH].cpu().numpy().astype(np.int16),
            top.values[:, :GATE_DEPTH].cpu().numpy().astype(np.float32))


def _write_dataset(root: Path, *, W: torch.Tensor, bias: torch.Tensor) -> None:
    """Six dev groups of widths 1-6 (one without an anchor); odd groups hold WINNER in the hot set."""
    token_sets = [[0], [0, 1], [0, 2, 5], [0, 1, 2, 3], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    g = torch.Generator().manual_seed(3)
    groups, words, offset = [], [], 0
    for seq, tokens in enumerate(token_sets):
        n = len(tokens)
        x = torch.randn((n, r1_cost.X_DIM), generator=g).bfloat16()
        words.append(x.view(torch.int16).numpy().view(np.uint16))
        gate_ids, gate_scores = _top12(r1_gate.full_gate_scores(x.float().to(DEV), W=W, bias=bias))
        target_ids = np.tile(np.arange(10, 10 + TOPK, dtype=np.int16), (n, 1))
        kinds = np.zeros((n, TOPK), np.uint8)
        kinds[:, 0] = int(LaneKind.MISS_GPU)
        groups.append(b1_rows.Group(
            session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
            horizon=l2_fast_parity.HORIZON, source_row=0, source_layer=SOURCE, source_stratum="a",
            gate_has_bias=True, target_row=1, target_layer=TARGET, target_stratum="b",
            tokens=np.array(tokens, np.int16), x_offset=offset,
            source_ids=torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(n)]).numpy().astype(np.int16),
            source_weights=torch.rand((n, TOPK), generator=g).numpy(), target_ids=target_ids, target_kinds=kinds,
            ram_mask=np.zeros(6, np.uint64), hot=np.array([0, WINNER] if seq % 2 else [0, 1], np.int16),
            gate_ids=gate_ids, gate_scores=gate_scores))
        offset += n
    b1_rows.write_x(root, split="dev", session="s1", x=np.concatenate(words))
    b1_rows.write_groups(root, split="dev", horizon=l2_fast_parity.HORIZON, session="s1", groups=groups)


@needs_cuda
@pytest.mark.parametrize("case, dtype", [(l2_fast_parity.FP32_GATE, torch.float32),
                                         (l2_fast_parity.SPEC_SCORE, torch.bfloat16)], ids=["fp32_gate", "spec_score"])
def test_torch_and_kernel_paths_agree_with_the_reference_on_every_read(tmp_path, case, dtype):
    """A fold, gather, gate-score feed or kernel record that drifts from the F109 reference changes some read."""
    W, bias = _gate()
    _write_dataset(tmp_path, W=W, bias=bias)
    groups = l2_fast_parity._read_split(tmp_path, split="dev", exclusions=r1_groups.Exclusions.empty(), limit=None)
    assert len(groups) == 6
    x_of = r1_cache.make_x_of(tmp_path, split="dev")
    models = {"L2_T0": _model(t3=False), "L2_T3": _model(t3=True)}
    heads = l2_fast_parity._heads(models, rows=l2_fast_torch.serving_rows(groups), device=DEV)
    thresholds = {name: dict.fromkeys(l2_fast_parity.BUDGETS, THR) for name in models}
    ref = l2_fast_parity.reference_scores(groups, models=models, f0=models["L2_T0"].head, x_of=x_of, device=DEV)
    gates = {TARGET: (W.to(dtype).contiguous(), bias)}
    got, mismatch = l2_fast_parity.case_scores(groups, case=case, heads=heads, gates=gates, x_of=x_of, device=DEV,
                                               thresholds=thresholds)
    t = l2_fast_parity.tables(groups, ref=ref, got=got, mismatch=mismatch, thresholds=thresholds, case=case)
    for name in models:
        reads = [l2_fast_parity._policy_read(r, s, threshold=THR) for r, s in zip(groups, ref[name])]
        assert reads == [WINNER, None] * 3
        for path in l2_fast_parity.PATHS:
            table = t[name][path]["0.75"]
            assert table["read_agree"] == 1.0 and table["disagreements"] == 0, (name, path, table)
        assert t[name]["kernel"]["0.75"]["kernel_record_mismatch"] == 0
        if case == l2_fast_parity.FP32_GATE:
            assert t[name]["torch"]["0.75"]["max_abs_diff"] <= 1e-4
        else:
            assert t[name]["torch"]["0.75"]["recall_change"]["diff"] == 0.0
