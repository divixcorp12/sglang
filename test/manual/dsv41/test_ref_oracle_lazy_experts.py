"""The oracle's disk-backed experts equal its eagerly loaded ones (CPU)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41"))

import ref_oracle  # noqa: E402

from sglang.test.dsv41_fake_exl3 import write_fake_exl3  # noqa: E402


def test_lazy_routed_experts_match_eager(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=3)
    _state, eager, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu")
    _state, lazy, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu", lazy_routed=True)
    assert set(lazy) == set(eager)
    for key, disk in lazy.items():
        loaded = disk.load("cpu")
        for w in ("w1", "w2", "w3"):
            assert torch.equal(loaded[w].trellis, eager[key][w].trellis)
            assert torch.equal(loaded[w].suh.view(torch.int16), eager[key][w].suh.view(torch.int16))
            assert torch.equal(loaded[w].svh.view(torch.int16), eager[key][w].svh.view(torch.int16))


def test_lazy_mode_holds_no_expert_tensor(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=2)
    _state, lazy, _shared = ref_oracle._load_checkpoint(str(tmp_path), device="cpu", lazy_routed=True)
    for disk in lazy.values():
        assert not any(isinstance(v, torch.Tensor) for v in vars(disk).values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
