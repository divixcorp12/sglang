"""Per-call (num_heads, top_k) FlashInfer dispatch check for SM120 sparse decode.

FlashInfer's DSV4 decode kernel is instantiated for a fixed set of
(num_heads, top_k) pairs, keyed on the main index width ``indices.shape[-1]``.
Other widths (e.g. a wider verify index list) must fall back to Triton per call
unless FlashInfer was forced explicitly, in which case a clear error is raised.
"""

import pytest
import torch

from sglang.kernels.ops.attention import flash_mla_sm120 as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TABLE = frozenset({(16, 128), (16, 512), (64, 128)})


@pytest.fixture(autouse=True)
def fake_capabilities(monkeypatch):
    mod._flashinfer_dsv4_decode_capabilities.cache_clear()
    mod.flashinfer_dsv4_decode_supports.cache_clear()
    mod._log_flashinfer_fallback_once.cache_clear()
    monkeypatch.setattr(
        mod,
        "_flashinfer_dsv4_decode_capabilities",
        lambda: (64, frozenset(h for h, _ in _TABLE), _TABLE),
    )
    calls = []
    monkeypatch.setattr(
        mod, "_flash_mla_flashinfer", lambda *a, **k: calls.append("fi") or "fi"
    )
    monkeypatch.setattr(
        mod, "_flash_mla_sm120_prefill", lambda *a, **k: calls.append("prefill")
    )
    import sglang.kernels.ops.attention.flash_mla_sm120_triton as tri

    monkeypatch.setattr(
        tri,
        "flash_mla_sparse_decode_triton",
        lambda *a, **k: calls.append("triton") or ("tri", None),
    )
    yield calls
    mod.flashinfer_dsv4_decode_supports.cache_clear()
    mod._log_flashinfer_fallback_once.cache_clear()


def _call(heads, topk, tokens=1):
    q = torch.zeros(tokens, 1, heads, 512)
    idx = torch.zeros(tokens, 1, topk, dtype=torch.int32)
    return mod.flash_mla_with_kvcache_sm120(
        q=q,
        k_cache=torch.zeros(1),
        indices=idx,
        head_dim_v=512,
        softmax_scale=1.0,
    )


def test_supports_pair_checks_topk():
    assert mod.flashinfer_dsv4_decode_supports(16, 128)
    assert not mod.flashinfer_dsv4_decode_supports(16, 192)
    assert not mod.flashinfer_dsv4_decode_supports(8, 128)


def test_num_heads_api_unchanged():
    assert mod.flashinfer_dsv4_decode_supports_num_heads(16, 64)
    assert not mod.flashinfer_dsv4_decode_supports_num_heads(16, 65)
    assert not mod.flashinfer_dsv4_decode_supports_num_heads(8, 1)


def test_supported_pair_uses_flashinfer(monkeypatch, fake_capabilities):
    monkeypatch.setattr(mod, "_sm120_default_backend", "flashinfer")
    monkeypatch.setattr(mod, "_sm120_backend_forced", False)
    assert _call(16, 128) == "fi"
    assert fake_capabilities == ["fi"]


def test_unsupported_pair_falls_back_to_triton_once_logged(
    monkeypatch, fake_capabilities, caplog
):
    monkeypatch.setattr(mod, "_sm120_default_backend", "flashinfer")
    monkeypatch.setattr(mod, "_sm120_backend_forced", False)
    with caplog.at_level("WARNING", logger=mod.logger.name):
        assert _call(16, 192) == ("tri", None)
        assert _call(16, 192) == ("tri", None)
    assert fake_capabilities == ["triton", "triton"]
    assert len([r for r in caplog.records if "192" in r.getMessage()]) == 1


def test_forced_flashinfer_unsupported_pair_raises(monkeypatch):
    monkeypatch.setattr(mod, "_sm120_default_backend", "flashinfer")
    monkeypatch.setattr(mod, "_sm120_backend_forced", True)
    with pytest.raises(RuntimeError, match=r"num_heads=16.*top_k=192"):
        _call(16, 192)


def test_forced_flashinfer_supported_pair_ok(monkeypatch):
    monkeypatch.setattr(mod, "_sm120_default_backend", "flashinfer")
    monkeypatch.setattr(mod, "_sm120_backend_forced", True)
    assert _call(64, 128) == "fi"


def test_prefill_sized_batch_is_not_gated(monkeypatch, fake_capabilities):
    monkeypatch.setattr(mod, "_sm120_default_backend", "flashinfer")
    monkeypatch.setattr(mod, "_sm120_backend_forced", False)
    _call(16, 192, tokens=65)
    assert fake_capabilities == ["prefill"]


def test_forced_triton_ignores_table(monkeypatch, fake_capabilities):
    monkeypatch.setattr(mod, "_sm120_default_backend", "triton")
    monkeypatch.setattr(mod, "_sm120_backend_forced", True)
    _call(16, 128)
    assert fake_capabilities == ["triton"]


def test_old_flashinfer_without_table_falls_back(monkeypatch, fake_capabilities):
    monkeypatch.setattr(
        mod, "_flashinfer_dsv4_decode_capabilities", lambda: (0, frozenset(), frozenset())
    )
    mod.flashinfer_dsv4_decode_supports.cache_clear()
    monkeypatch.setattr(mod, "_sm120_default_backend", "flashinfer")
    monkeypatch.setattr(mod, "_sm120_backend_forced", False)
    _call(16, 128)
    assert fake_capabilities == ["triton"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
