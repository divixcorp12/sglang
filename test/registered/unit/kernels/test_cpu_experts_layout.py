"""Every CPU expert quant has the standard file set, so a new quant is a new directory filling in the same files."""

from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

QUANT = Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/quantization"
STANDARD = {"cpu_experts_cabi.h", "quant.hpp", "math.hpp", "math_scalar.hpp", "math_avx2.hpp", "shapes.hpp",
            "forward_plan.hpp", "kernel.cpp"}
QUANTS = ["exl3_cpu", "nvfp4_cpu"]


def test_every_quant_has_the_standard_files():
    for quant in QUANTS:
        present = {p.name for p in (QUANT / quant / "optimized").iterdir()}
        assert STANDARD <= present, (quant, sorted(STANDARD - present))


def test_every_quant_declares_the_five_c_names():
    for quant in QUANTS:
        prefix = quant.removesuffix("_cpu")
        header = (QUANT / quant / "optimized/cpu_experts_cabi.h").read_text()
        for name in ("register_layer", "free_layer", "forward", "keep_warm", "set_cores"):
            assert f"sglang_{prefix}_cpu_experts_{name}(" in header, (quant, name)
