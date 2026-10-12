"""r1_cache.py on synthetic groups: padded collation equals make_batch per group, exclusions keep x aligned, native
gate scores, seeded whole-group batching and the locked_test refusal. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import r1_cache  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
from test_b1_head import DIM, RANK, _cfg, _model, _x_of  # noqa: E402
from test_b1_rows import _group  # noqa: E402

E = b1_head.HeadConfig().num_experts


def _r1(g, *, drop=None):
    keep = None
    if drop is not None:
        keep = np.ones(len(g.tokens), dtype=bool)
        keep[drop] = False
    return r1_groups._r1_group(g, keep=keep)


def _setup(variant="E5"):
    model, groups = _model(variant)
    return model, [_r1(g) for g in groups]


def _x(r):
    return _x_of(r.g)


def test_collate_rows_equal_make_batch_and_padding_is_dead():
    model, rs = _setup()
    inp = r1_cache.collate(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu")
    want = b1_head.make_batch([r.g for r in rs], x_of=_x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
    b = inp.batch
    assert inp.u.shape == (len(rs), 6, want.u.shape[1]) and inp.x is None and inp.s_native is None
    n = len(rs[0].g.tokens)
    assert b.live[:, :n].all() and not b.live[:, n:].any()
    assert torch.equal(inp.u[0, :n], want.u[:n])
    assert not inp.u[:, n:].any() and not b.token_y[:, n:].any() and not b.route[:, n:].any()
    np.testing.assert_array_equal(b.token_y[1, :n], want.y[n:2 * n].numpy())
    np.testing.assert_array_equal(b.hard_neg[1, :n], want.hard_neg[n:2 * n].numpy())
    np.testing.assert_array_equal(b.width, [n] * len(rs))
    np.testing.assert_array_equal(b.source_layer, want.source_layer[::n].numpy())
    np.testing.assert_array_equal(b.route[0, :n], rs[0].labels.route.astype(np.float32))
    np.testing.assert_array_equal(b.miss[0], rs[0].labels.miss.astype(np.float32))
    assert b.available.shape == (len(rs), E)


def test_excluded_middle_token_keeps_features_aligned():
    model, rs = _setup()
    full = rs[0]
    cut = _r1(full.g, drop=0)
    inp = r1_cache.collate([cut], x_of=lambda r: _x_of(full.g)[r.x_index], pca=model.pca, cfg=model.cfg,
                           vocab=model.vocab, device="cpu")
    ref = r1_cache.collate([full], x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu")
    assert inp.batch.live[0].tolist() == [True, False, False, False, False, False]
    assert torch.equal(inp.u[0, 0], ref.u[0, 1])
    np.testing.assert_array_equal(inp.batch.position[0, :1], [1])


def test_s_native_and_x_follow_the_gate_formula():
    model, rs = _setup()
    g0 = rs[0].g
    rng = np.random.default_rng(0)
    W = torch.from_numpy(rng.standard_normal((E, DIM)).astype(np.float32))
    bias = torch.from_numpy(rng.standard_normal(E).astype(np.float32))
    gates = {r.g.target_layer: (W, bias) for r in rs}
    inp = r1_cache.collate(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu", with_x=True,
                           gates=gates)
    n = len(g0.tokens)
    x = torch.from_numpy(_x_of(g0))
    assert torch.equal(inp.x[0, :n], x) and not inp.x[0, n:].any()
    assert torch.allclose(inp.s_native[0, :n], r1_gate.full_gate_scores(x, W=W, bias=bias))
    assert not inp.s_native[0, n:].any()
    only_gates = r1_cache.collate(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu",
                                  gates=gates)
    assert only_gates.x is None and only_gates.s_native is not None


def test_cache_take_and_seeded_epoch_batches():
    model, rs = _setup()
    cache = r1_cache.build_cache(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu", chunk=4)
    whole = r1_cache.collate(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu")
    assert cache.groups == len(rs) == len(cache.keys) and torch.equal(cache.inp.u, whole.u)
    np.testing.assert_array_equal(cache.inp.batch.route, whole.batch.route)
    sub = r1_cache.take(cache, torch.tensor([3, 0]))
    assert torch.equal(sub.u, whole.u[[3, 0]])
    np.testing.assert_array_equal(sub.batch.miss, whole.batch.miss[[3, 0]])
    a = [b.tolist() for b in r1_cache.epoch_batches(cache, groups_per_step=4, seed=5, epoch=0)]
    again = [b.tolist() for b in r1_cache.epoch_batches(cache, groups_per_step=4, seed=5, epoch=0)]
    other = [b.tolist() for b in r1_cache.epoch_batches(cache, groups_per_step=4, seed=5, epoch=1)]
    assert a == again and a != other
    assert sorted(i for batch in a for i in batch) == list(range(cache.groups))
    assert [len(batch) for batch in a] == [4, 2]


def test_x_budget_is_enforced():
    model, rs = _setup()
    with pytest.raises(MemoryError):
        r1_cache.build_cache(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu",
                             with_x=True, x_budget_bytes=1024)


def test_make_x_of_reads_the_kept_rows_and_refuses_locked_test(tmp_path):
    with pytest.raises(PermissionError):
        r1_cache.make_x_of(tmp_path, split="locked_test")
    g = _group(seq=1, source_row=0, split="dev", n=2)
    values = np.arange(2 * DIM, dtype=np.float32).reshape(2, DIM)
    b1_rows.write_x(tmp_path, split="dev", session=g.session, x=(values.view(np.uint32) >> 16).astype(np.uint16))
    x_of = r1_cache.make_x_of(tmp_path, split="dev")
    np.testing.assert_array_equal(x_of(_r1(g)), values)
    np.testing.assert_array_equal(x_of(_r1(g, drop=0)), values[[1]])
    with pytest.raises(ValueError):
        r1_cache.make_x_of(tmp_path, split="train")(_r1(g))


def test_head_config_dim_matches_cache_features():
    model, rs = _setup("E2")
    inp = r1_cache.collate(rs[:1], x_of=_x, pca=model.pca, cfg=_cfg("E2"), vocab=model.vocab, device="cpu")
    assert inp.u.shape[-1] == RANK + 2 * E
