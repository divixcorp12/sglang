"""The CRTP split's shape (plan 2026-09-28-reader-crtp-uring-registration Task 4): the shared pipeline names no
bounce, pool or image mechanism, and the derived reader holds only its own. Since plan 2026-09-29-hotpath-zero-overhead
D4 the one derived reader is RowReader, and the tier reads through it directly."""

import pathlib

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOST = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/kernels/jit/csrc/moe/expert_stream/host"


def _code(name):
    """The header without // comments, so prose naming a mechanism does not count."""
    return "\n".join(line.split("//", 1)[0] for line in (HOST / name).read_text().splitlines())


def test_the_core_names_no_path_specific_mechanism():
    core = _code("reader_core.h")
    for word in ("t_.images", "bounce_", "pool_", "jobs_", "PackPool", "image_iovecs", "publish_landed",
                 "posix_memalign", "check_image_alignment"):
        assert word not in core, word
    assert "template <class Derived, ExpertRowLayout Layout, AsyncFileReader Reader>" in core
    assert "class ReaderCore" in core


def test_each_derived_reader_holds_only_its_own_mechanism():
    row = _code("row_reader.h")
    assert "class RowReader : public ReaderCore<RowReader<Layout, Reader>, Layout, Reader>" in row
    for word in ("bounce_", "pool_", "PackPool", "PackJob", "dispatch_ready", "set_pack", "packing_cpus"):
        assert word not in row, word


def test_the_bounce_path_keeps_the_scalar_read_opcode():
    core = _code("reader_core.h")
    assert "io_.prep_read(" in core and "io_.prep_readv(" in core


def test_the_tier_and_ffi_read_through_the_row_reader():
    assert "using Source = RowReader<Layout, Reader>;" in _code("ffi_exports.h")
    for gone in ("pack_reader.h", "pack_pool.h", "any_reader.h"):
        assert not (HOST / gone).exists(), gone
