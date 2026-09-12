import unittest

import torch

from sglang.srt.layers.quantization.utils import swizzle_blockscale


def _reference_swizzle(scale: torch.Tensor) -> torch.Tensor:
    batch, rows, cols = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    padded = torch.zeros(
        (batch, padded_rows, padded_cols),
        dtype=scale.dtype,
        device=scale.device,
    )
    padded[:, :rows, :cols] = scale
    return (
        padded.reshape(
            batch,
            padded_rows // 128,
            4,
            32,
            padded_cols // 4,
            4,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(batch, padded_rows, padded_cols)
    )


class TestNvfp4Swizzle(unittest.TestCase):
    def test_can_remain_on_cpu_without_padding(self):
        scale = torch.arange(2 * 128 * 8, dtype=torch.float32).reshape(2, 128, 8)
        scale = scale.to(torch.float8_e4m3fn)

        actual = swizzle_blockscale(scale, target_device="cpu")

        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(actual, _reference_swizzle(scale)))

    def test_can_remain_on_cpu_with_padding(self):
        scale = torch.arange(3 * 129 * 5, dtype=torch.float32).reshape(3, 129, 5)
        scale = (scale.remainder(16) - 8).to(torch.float8_e4m3fn)

        actual = swizzle_blockscale(scale, target_device=torch.device("cpu"))

        self.assertEqual(actual.shape, (3, 256, 8))
        self.assertTrue(torch.equal(actual, _reference_swizzle(scale)))

    def test_none_preserves_source_device(self):
        scale = torch.ones((1, 128, 4), dtype=torch.float8_e4m3fn)

        actual = swizzle_blockscale(scale, target_device=None)

        self.assertEqual(actual.device, scale.device)
        self.assertTrue(torch.equal(actual, _reference_swizzle(scale)))


if __name__ == "__main__":
    unittest.main()
