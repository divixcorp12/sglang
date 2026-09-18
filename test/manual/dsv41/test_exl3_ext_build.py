"""exllamav3's extension builds for sm_120 and exposes the kernels the exl3 method calls."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC"
)

NEEDED = ("exl3_gemm", "reconstruct", "reconstruct_had_slice", "had_r_128", "exl3_moe")


def test_builds_and_exposes_kernels():
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    ext = exl3_ext()
    missing = [name for name in NEEDED if not hasattr(ext, name)]
    assert not missing, missing


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
