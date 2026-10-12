"""b1_train.py on a planted rule: early-stopped training reaches high dev recall, seeded CPU runs match, subsets nest."""

import sys
from pathlib import Path

import msgspec
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import b1_train  # noqa: E402
from test_b1_rows import _group  # noqa: E402

DIM = 8
RANK = 4
EXPERT_POS, EXPERT_NEG = 5, 9


def _x_of(g):
    rng = np.random.default_rng(g.seq)
    x = rng.standard_normal((len(g.tokens), DIM)).astype(np.float32)
    x[:, 0] *= 5.0  # dominant direction so the top PCA components keep the planted signal
    return x


def _planted(seq, *, split):
    g = _group(seq=seq, n=1, split=split)
    ids = np.full((1, 6), -1, dtype=np.int16)
    ids[0, 0] = EXPERT_POS if _x_of(g)[0, 0] > 0 else EXPERT_NEG
    kinds = np.zeros((1, 6), dtype=np.uint8)
    kinds[0, 0] = b1_rows.LaneKind.MISS_GPU
    return msgspec.structs.replace(g, target_ids=ids, target_kinds=kinds)


def _run(seed=0, **train_kwargs):
    train = [_planted(i, split="train") for i in range(400)]
    dev = [_planted(1000 + i, split="dev") for i in range(100)]
    cfg = b1_head.HeadConfig(pca_rank=RANK, hidden=32, emb_dim=4, dropout=0.0)
    tcfg = b1_train.TrainConfig(epochs=25, lr=1e-2, batch_rows=32, patience=5, seed=seed, **train_kwargs)
    return b1_train.train_head(cfg, tcfg, train_groups=lambda: train, dev_groups=lambda: dev, x_of=_x_of)


def test_compiled_training_matches_eager_and_keeps_state_dict_keys():
    # The eager backend runs dynamo's graph capture on the real path without CPU codegen.
    a, ha = _run(seed=3)
    b, hb = _run(seed=3, compile=True, compile_backend="eager")
    assert [h["dev_recall_at_1"] for h in ha["history"]] == [h["dev_recall_at_1"] for h in hb["history"]]
    np.testing.assert_allclose([h["loss"] for h in ha["history"]], [h["loss"] for h in hb["history"]], rtol=1e-5)
    assert list(a.state_dict()) == list(b.state_dict())


def test_planted_rule_is_learned_to_high_recall():
    model, hist = _run()
    assert hist["best_dev_recall_at_1"] > 0.9
    assert hist["history"][hist["best_epoch"]]["dev_recall_at_1"] == hist["best_dev_recall_at_1"]
    assert hist["best_dev_table"]["cells"]["reads=1"]["pooled"]["eligible"] == 100
    assert not model.training


def test_seeded_cpu_runs_are_bit_identical():
    a, ha = _run(seed=3)
    b, hb = _run(seed=3)
    assert ha["history"] == hb["history"]
    for (name, ta), (_, tb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(ta, tb), name


def test_nested_subsets_are_nested_seeded_and_sized():
    sessions = [f"s{i:03d}" for i in range(60)]
    a = b1_train.nested_subsets(sessions, seed=1)
    assert [len(s) for s in a] == [15, 30, 60]
    assert set(a[0]) <= set(a[1]) <= set(a[2]) == set(sessions)
    assert a == b1_train.nested_subsets(list(reversed(sessions)), seed=1)
    assert a[0] != b1_train.nested_subsets(sessions, seed=2)[0]
