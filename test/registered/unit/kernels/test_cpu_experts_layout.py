"""Every CPU expert quant has the standard file set and exposes its kernel only through its accessor (kernel.h), so a new
quant is a new directory filling in the same files."""

from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CSRC = Path(__file__).resolve().parents[4] / "python/sglang/kernels/jit/csrc"
QUANTS = ["exl3", "nvfp4"]
STANDARD = {"kernel.h", "quant.hpp", "math.hpp", "math_scalar.hpp", "math_avx2.hpp", "shapes.hpp", "forward_plan.hpp",
            "kernel.cpp"}
FRAMEWORK = CSRC / "moe/expert_stream/host/cpu_experts"


def test_every_quant_has_the_standard_files():
    for quant in QUANTS:
        present = {p.name for p in (CSRC / quant / "optimized").iterdir()}
        assert STANDARD <= present, (quant, sorted(STANDARD - present))


def test_every_quant_declares_its_kernel_accessor():
    for quant in QUANTS:
        header = (CSRC / quant / "optimized/kernel.h").read_text()
        assert f"CpuExpertKernel& {quant}_cpu_kernel();" in header, quant
        assert 'visibility("hidden")' in header, quant


def test_no_cpu_expert_library_exports_a_c_function():
    """The C ABI is gone: no quant source and no framework header declares an extern "C" function."""
    sources = [p for quant in QUANTS for p in (CSRC / quant / "optimized").iterdir()] + list(FRAMEWORK.iterdir())
    offenders = [p.name for p in sources if p.is_file() and 'extern "C"' in p.read_text()]
    assert not offenders, offenders
    assert not (CSRC / "moe/expert_stream/host/cpu_experts_abi.h").exists()
