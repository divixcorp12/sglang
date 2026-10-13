"""r1_cache.py's R2 additions on synthetic groups: bf16 raw-x storage handed out as fp32, the dtype-aware x budget,
and R1's fp32 defaults unchanged. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_cache  # noqa: E402
from r1_groups import MAX_WIDTH  # noqa: E402
from test_b1_head import DIM  # noqa: E402
from test_r1_cache import E, _setup, _x  # noqa: E402

DTYPES = (torch.float32, torch.bfloat16)


def _build(model, rs, **kw):
    return r1_cache.build_cache(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu",
                                with_x=True, chunk=4, **kw)


def _gates(rs):
    rng = np.random.default_rng(0)
    W = torch.from_numpy(rng.standard_normal((E, DIM)).astype(np.float32))
    bias = torch.from_numpy(rng.standard_normal(E).astype(np.float32))
    return {r.g.target_layer: (W, bias) for r in rs}


def test_x_cache_bytes_uses_the_dtype_itemsize():
    assert r1_cache.x_cache_bytes(3, x_dtype=torch.float32) == 3 * MAX_WIDTH * r1_cache._HIDDEN * 4
    assert r1_cache.x_cache_bytes(3, x_dtype=torch.bfloat16) == 3 * MAX_WIDTH * r1_cache._HIDDEN * 2


def test_bf16_cache_takes_fp32_x_within_bf16_rounding():
    model, rs = _setup()
    ref = _build(model, rs)
    half = _build(model, rs, x_dtype=torch.bfloat16)
    assert half.inp.x.dtype == torch.bfloat16
    idx = torch.tensor([3, 0, 5])
    got, want = r1_cache.take(half, idx).x, r1_cache.take(ref, idx).x
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-5)
    assert torch.equal(got, want.to(torch.bfloat16).float())


def test_default_cache_matches_r1():
    model, rs = _setup()
    cache = _build(model, rs)
    assert cache.inp.x.dtype == torch.float32
    idx = torch.tensor([1, 4])
    assert torch.equal(r1_cache.take(cache, idx).x, cache.inp.x[idx])
    assert r1_cache._X_BUDGET_BYTES == 8 << 30
    with pytest.raises(MemoryError, match="raw x cache needs more than 1024 bytes; stream it instead"):
        _build(model, rs, x_budget_bytes=1024)


@pytest.mark.parametrize("x_dtype", DTYPES)
def test_budget_accepts_exactly_and_refuses_one_chunk_past(x_dtype):
    model, rs = _setup()
    total = r1_cache.x_cache_bytes(len(rs), x_dtype=x_dtype)
    assert _build(model, rs, x_dtype=x_dtype, x_budget_bytes=total).groups == len(rs)
    # chunk=4 over more than 4 groups: a budget that holds only the first chunk fails on the second.
    with pytest.raises(MemoryError):
        _build(model, rs, x_dtype=x_dtype, x_budget_bytes=r1_cache.x_cache_bytes(4, x_dtype=x_dtype))
    with pytest.raises(MemoryError):
        _build(model, rs, x_dtype=x_dtype, x_budget_bytes=total - 1)


def test_s_native_is_independent_of_x_dtype():
    model, rs = _setup()
    gates = _gates(rs)
    kw = dict(x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu", with_x=True, gates=gates)
    full = r1_cache.collate(rs, **kw)
    half = r1_cache.collate(rs, x_dtype=torch.bfloat16, **kw)
    assert half.s_native.dtype == torch.float32 and torch.equal(half.s_native, full.s_native)
    assert half.x.dtype == torch.bfloat16 and torch.equal(half.u, full.u)


def _nbytes(inp) -> int:
    return sum(t.numel() * t.element_size() for t in (inp.u, inp.x, inp.s_native) if t is not None)


@pytest.mark.parametrize("x_dtype", DTYPES)
def test_cache_bytes_counts_u_x_and_s_native_of_a_built_cache(x_dtype):
    model, rs = _setup()
    cache = _build(model, rs, x_dtype=x_dtype, gates=_gates(rs))
    assert _nbytes(cache.inp) == r1_cache.cache_bytes(len(rs), cfg=model.cfg, x_dtype=x_dtype, with_s_native=True,
                                                      hidden=DIM)
    plain = _build(model, rs, x_dtype=x_dtype)
    assert _nbytes(plain.inp) == r1_cache.cache_bytes(len(rs), cfg=model.cfg, x_dtype=x_dtype, with_s_native=False,
                                                      hidden=DIM)


def test_chunked_cache_equals_one_collate():
    model, rs = _setup()
    gates = _gates(rs)
    cache = _build(model, rs, x_dtype=torch.bfloat16, gates=gates)
    whole = r1_cache.collate(rs, x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cpu",
                             with_x=True, gates=gates, x_dtype=torch.bfloat16)
    assert len(rs) > 4 and cache.groups == len(rs)
    for name in ("u", "x", "s_native"):
        assert torch.equal(getattr(cache.inp, name), getattr(whole, name)), name
    for name in whole.batch.__struct_fields__:
        np.testing.assert_array_equal(getattr(cache.inp.batch, name), getattr(whole.batch, name), err_msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_build_peak_stays_near_the_cache_size():
    # The real train split died joining chunks: chunks plus the joined copy held every tensor twice.
    model, rs = _setup()
    rs = rs * 40
    gates = {k: (W.cuda(), b.cuda()) for k, (W, b) in _gates(rs).items()}
    kw = dict(x_of=_x, pca=model.pca, cfg=model.cfg, vocab=model.vocab, device="cuda", with_x=True, gates=gates,
              x_dtype=torch.bfloat16)
    # The first gate matmul allocates cuBLAS's 32 MiB workspace for the process; keep it out of the baseline.
    r1_cache.collate(rs[:1], **kw)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    cache = r1_cache.build_cache(rs, chunk=4, **kw)
    size = _nbytes(cache.inp)
    assert torch.cuda.max_memory_allocated() - base < 1.5 * size
