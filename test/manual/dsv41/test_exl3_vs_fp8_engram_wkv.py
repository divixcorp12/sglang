"""EXL3 Engram wkv against the official FP8 copy: ground-truth check for orientation.

layers.1.engram.wkv is quantized (EXL3) in the 3.0bpw export but also ships, unquantized,
as an F8_E4M3 tensor plus an F8_E8M0 block scale in the official DeepSeek-V4.1-Flash
checkpoint. Reconstructing the EXL3 tensors and dequantizing the FP8 copy should agree
up to quantization noise -- and only in one orientation, since exl3_dense_weight returns
[in, out] while the official Linear weight is stored [out, in].
"""

import json
import os

import pytest
import torch

from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_dense_weight

EXL3_DIR = os.environ.get(
    "DSV41_EXL3_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
)
ENGRAM_DIR = os.environ.get("DSV41_ENGRAM_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash")
ENGRAM_SHARD = "model-00047-of-00048.safetensors"
TENSOR_PREFIX = "layers.1.engram.wkv"

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and os.environ.get("SGLANG_EXL3_SRC")
        and os.path.exists(os.path.join(EXL3_DIR, "model.safetensors.index.json"))
        and os.path.exists(os.path.join(ENGRAM_DIR, ENGRAM_SHARD))
    ),
    reason="needs a GPU, SGLANG_EXL3_SRC, and the divix01 EXL3 + Engram checkpoints",
)


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _find_shard(model_dir: str, tensor_name: str) -> str:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    return os.path.join(model_dir, weight_map[tensor_name])


def _load_exl3_wkv(device: str = "cuda") -> Exl3Tensors:
    from safetensors import safe_open

    shard = _find_shard(EXL3_DIR, f"{TENSOR_PREFIX}.trellis")
    with safe_open(shard, framework="pt", device=device) as f:
        trellis = f.get_tensor(f"{TENSOR_PREFIX}.trellis")
        suh = f.get_tensor(f"{TENSOR_PREFIX}.suh")
        svh = f.get_tensor(f"{TENSOR_PREFIX}.svh")
        mul1 = f.get_tensor(f"{TENSOR_PREFIX}.mul1")
    return Exl3Tensors(trellis=trellis, suh=suh, svh=svh, mul1=bool(mul1.item()))


def _load_fp8_wkv_dense() -> torch.Tensor:
    from safetensors import safe_open

    path = os.path.join(ENGRAM_DIR, ENGRAM_SHARD)
    with safe_open(path, framework="pt", device="cuda") as f:
        w = f.get_tensor(f"{TENSOR_PREFIX}.weight")
        s = f.get_tensor(f"{TENSOR_PREFIX}.scale")
    # F8_E4M3 / F8_E8M0 upcast to float32 before arithmetic.
    w = w.float()
    s = s.float()
    return w * s.repeat_interleave(32, 0).repeat_interleave(32, 1)


def test_exl3_matches_fp8_only_when_transposed():
    t = _load_exl3_wkv()
    exl3_dense = exl3_dense_weight(t).float()  # [in, out] = [6144, 25600]
    fp8_dense = _load_fp8_wkv_dense()  # [out, in] = [25600, 6144]

    assert exl3_dense.shape == (6144, 25600)
    assert fp8_dense.shape == (25600, 6144)

    correct = exl3_dense.t()  # -> [25600, 6144], matches fp8_dense's orientation
    rel_rms = _rel_rms(correct, fp8_dense)
    cos = float(
        torch.nn.functional.cosine_similarity(
            correct.reshape(-1), fp8_dense.reshape(-1), dim=0
        )
    )
    print(f"rel_rms(exl3.T, fp8) = {rel_rms:.6f}")
    print(f"cos(exl3.T, fp8) = {cos:.6f}")
    assert rel_rms < 0.08
    assert cos > 0.99

    # Orientation pin: skipping the required .t() (row-major reinterpretation into the
    # same shape, rather than an algebraic transpose) must be far worse. A consistent
    # transpose of both sides leaves rel-RMS unchanged, so this is the only way to make
    # a same-shape "wrong orientation" comparison that a real bug (dropping the .t())
    # would actually produce.
    wrong = exl3_dense.reshape(fp8_dense.shape)
    wrong_rel_rms = _rel_rms(wrong, fp8_dense)
    print(f"rel_rms(exl3 reshaped without transpose, fp8) = {wrong_rel_rms:.6f}")
    assert wrong_rel_rms > 10 * rel_rms


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
