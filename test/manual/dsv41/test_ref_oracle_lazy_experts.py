"""The oracle's disk-backed experts equal its eagerly loaded ones (CPU)."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41"))

import ref_oracle  # noqa: E402

from sglang.test.dsv41_fake_exl3 import HIDDEN, INTER, write_fake_exl3  # noqa: E402

_WEIGHTS = ("w1", "w2", "w3")


def _cpu_dense(t):
    """CPU stand-in for the CUDA-only `exl3_dense_weight`: [in, out] fp16, a deterministic
    function of trellis, suh, svh and mul1, so a wrong tensor anywhere changes the output."""
    g = t.trellis.float().sum(-1).repeat_interleave(16, 0).repeat_interleave(16, 1) / 1e6
    g = g * (2.0 if t.mul1 else 1.0)
    return (g * t.suh.float()[:, None] * t.svh.float()[None, :]).half()


@pytest.fixture
def cpu_dense(monkeypatch):
    monkeypatch.setattr(ref_oracle, "exl3_dense_weight", _cpu_dense)


def _tensors_in(obj):
    return [k for k, v in vars(obj).items() if isinstance(v, torch.Tensor)]


def _inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(5, HIDDEN, generator=g).bfloat16(), torch.rand(5, 1, generator=g)


def test_lazy_routed_experts_match_eager(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=3)
    _state, eager, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu")
    _state, lazy, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu", lazy_routed=True)
    assert set(lazy) == set(eager)
    for key, disk in lazy.items():
        loaded = disk.load("cpu")
        for w in _WEIGHTS:
            assert torch.equal(loaded[w].trellis, eager[key][w].trellis)
            assert torch.equal(loaded[w].suh.view(torch.int16), eager[key][w].suh.view(torch.int16))
            assert torch.equal(loaded[w].svh.view(torch.int16), eager[key][w].svh.view(torch.int16))
            assert loaded[w].mul1 == eager[key][w].mul1


def test_lazy_mode_holds_no_expert_tensor(tmp_path, cpu_dense):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=2, finite=True)
    _state, lazy, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu", lazy_routed=True)
    x, w = _inputs()
    for disk in lazy.values():
        before = dict(vars(disk))
        assert not _tensors_in(disk)
        disk.load("cpu")
        assert not _tensors_in(disk)
        assert vars(disk) == before
        expert = ref_oracle._make_lazy_expert()(HIDDEN, INTER, swiglu_limit=10.0)
        expert.bind_disk(disk, "cpu")
        expert(x, w)
        assert not _tensors_in(disk)
        assert expert._tensors is None and expert._dense == ()
        assert not _tensors_in(expert)
        assert not any(True for _ in expert.parameters())


def test_lazy_expert_disk_matches_eager_bind(tmp_path, cpu_dense):
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=3, finite=True)
    _state, eager, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu")
    _state, lazy, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu", lazy_routed=True)
    Lazy = ref_oracle._make_lazy_expert()
    x, w = _inputs()
    for key in eager:
        a = Lazy(HIDDEN, INTER, swiglu_limit=10.0)
        b = Lazy(HIDDEN, INTER, swiglu_limit=10.0)
        a.bind(eager[key]["w1"], eager[key]["w2"], eager[key]["w3"], keep_dense=False)
        b.bind_disk(lazy[key], "cpu")
        for da, db in zip(a._dequant(), b._dequant()):
            assert torch.equal(da, db)
        out_a, out_b = a(x, w), b(x, w)
        assert torch.isfinite(out_a.float()).all() and out_a.abs().sum() > 0
        assert torch.equal(out_a, out_b)
        assert torch.equal(a(x, w), b(x, w))  # the second call re-reads the shard
        assert b._tensors is None and b._dense == ()


def test_bind_routed_lazy_matches_eager(tmp_path, cpu_dense):
    num_layers, num_experts = 2, 3
    write_fake_exl3(str(tmp_path), num_layers=num_layers, num_experts=num_experts, finite=True)
    Lazy = ref_oracle._make_lazy_expert()

    def stub():
        layers = [
            SimpleNamespace(
                ffn=SimpleNamespace(
                    experts=torch.nn.ModuleList(Lazy(HIDDEN, INTER, swiglu_limit=10.0) for _ in range(num_experts))
                )
            )
            for _ in range(num_layers)
        ]
        return SimpleNamespace(layers=layers)

    def run(model, x, topk, weights):
        outs = []
        for layer in model.layers:
            y = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32)
            for t in range(x.shape[0]):
                for j in range(topk.shape[1]):
                    e = int(topk[t, j])
                    y[t] += layer.ffn.experts[e](x[t : t + 1], weights[t : t + 1, j : j + 1])[0].float()
            outs.append(y)
        return torch.stack(outs)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(6, HIDDEN, generator=g).bfloat16()
    topk = torch.stack([torch.randperm(num_experts, generator=g)[:2] for _ in range(6)])
    weights = torch.rand(6, 2, generator=g)

    results = {}
    for lazy_routed in (False, True):
        _state, routed, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu", lazy_routed=lazy_routed)
        model = stub()
        ref_oracle._bind_routed(model, routed, lazy_routed, "cpu")
        results[lazy_routed] = run(model, x, topk, weights)
        for layer in model.layers:
            for expert in layer.ffn.experts:
                assert (expert._tensors is None) == lazy_routed
                assert expert._dense == ()
    assert torch.isfinite(results[False]).all() and results[False].abs().sum() > 0
    assert torch.equal(results[False], results[True])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
