"""The engine's transport builds are keyed by (format, router): the registry, the JIT keys, the format traits' C++
headers (CPU only; nothing here builds a JIT module or needs CUDA).

  * LAYOUTS holds the four pairs (exl3, sqrt_softplus), (nvfp4, sqrt_softplus), (nvfp4, sigmoid: MiMo once converted to
    NVFP4) and (mxfp4, sigmoid), each with its own
    host and device JIT key and the -D macros engine_select.h switches on;
  * an unregistered pair, or a bare format name, raises, and the message names the JIT key looked for;
  * the row layouts of formats/<name>/row_layout.h agree with their Python tensor tables;
  * the format traits compile, satisfy ExpertFormatTraits, and UnsupportedCpuKernel aborts naming its format (a child
    process with core dumps off), when a C++ compiler is on the machine. The engine TUs themselves (tvm-ffi, CUDA) are
    compile-checked on divix01 only."""

import re
import resource
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

KEYS = [ops.EXL3_BUILD, ops.NVFP4_BUILD, ops.NVFP4_MIMO_BUILD, ops.MXFP4_BUILD, ops.FAKE_BUILD]
UNREGISTERED = ("mxfp4", "sqrt_softplus")  # a real format with a router it is not built for


def test_the_builds_are_registered_as_format_router_pairs():
    assert set(ops.LAYOUTS) == {
        ("exl3", "sqrt_softplus"),
        ("nvfp4", "sqrt_softplus"),
        ("nvfp4", "sigmoid"),
        ("mxfp4", "sigmoid"),
        ("fake", "sqrt_softplus"),
    }
    assert ops.NVFP4_MIMO_BUILD == ("nvfp4", "sigmoid")


def test_the_two_nvfp4_builds_share_their_sources_and_differ_only_in_the_router_macro():
    dsv41, mimo = ops.LAYOUTS[ops.NVFP4_BUILD], ops.LAYOUTS[ops.NVFP4_MIMO_BUILD]
    assert dsv41.host_sources == mimo.host_sources and dsv41.device_source == mimo.device_source
    assert dsv41.defines != mimo.defines
    assert set(dsv41.defines) ^ set(mimo.defines) == {
        "-DSGLANG_EXPERT_ROUTER=sqrt_softplus",
        "-DSGLANG_EXPERT_ROUTER=sigmoid",
    }
    assert ops.device_jit_key(ops.NVFP4_MIMO_BUILD) == "expert_stream_nvfp4_sigmoid_l8"
    assert ops.host_jit_key(ops.NVFP4_MIMO_BUILD, "prod") == "expert_stream_host_nvfp4_sigmoid_prod_l8"


def test_each_build_has_distinct_host_and_device_jit_keys_naming_both_halves():
    host = [ops.host_jit_key(key, variant) for key in KEYS for variant in ops.VARIANTS]
    device = [ops.device_jit_key(key) for key in KEYS]
    assert len(set(host)) == len(host) and len(set(device)) == len(device)
    for (fmt, router), key in zip(KEYS, device):
        assert fmt in key and router in key, key
    assert ops.host_jit_key(ops.EXL3_BUILD, "prod") == "expert_stream_host_exl3_sqrt_softplus_prod_l8"
    assert ops.host_jit_key(ops.MXFP4_BUILD, "instr", 16, 2) == "expert_stream_host_mxfp4_sigmoid_instr_l16_n2"
    assert ops.device_jit_key(ops.NVFP4_BUILD) == "expert_stream_nvfp4_sqrt_softplus_l8"


@pytest.mark.parametrize("key", KEYS)
def test_a_build_selects_its_format_and_router_by_macro_in_the_shared_engine_files(key):
    fmt, router = key
    build = ops.LAYOUTS[key]
    assert f"-DSGLANG_EXPERT_FORMAT={fmt}" in build.defines and f"-DSGLANG_EXPERT_ROUTER={router}" in build.defines
    assert build.host_sources == {
        "prod": "moe/expert_stream/engine_host.cpp",
        "instr": "moe/expert_stream/engine_host.cpp",
    }
    assert build.device_source == "moe/expert_stream/engine_device.cuh"
    for source in (*build.host_sources.values(), build.device_source):
        assert (MOE.parent / source).is_file(), source
    assert (MOE / "formats" / fmt / "format.h").is_file()
    wrappers = dict(ops._device_wrappers(key))
    assert wrappers["expert_stream_spec_score"] == "SpecScoreKernelFor<sglang::expert_stream::EngineRouter>::score"
    assert wrappers["expert_stream_lease_copy_wait"] == (
        "RowCopyKernel<sglang::expert_stream::EngineLayout>::lease_copy_wait"
    )


def test_an_unregistered_host_build_raises_with_the_jit_key_it_looked_for():
    with pytest.raises(ValueError, match="expert_stream_host_mxfp4_sqrt_softplus_prod_l8") as error:
        ops._host_module(UNREGISTERED, "prod")
    assert "(format, router)" in str(error.value) and "mxfp4', 'sigmoid" in str(error.value)  # what is registered
    with pytest.raises(ValueError, match="expert_stream_host_exl3_sigmoid_instr_l8_n2"):
        ops._host_module(("exl3", "sigmoid"), "instr", 8, 2)


def test_an_unregistered_device_build_raises_with_the_jit_key_it_looked_for():
    with pytest.raises(ValueError, match="expert_stream_mxfp4_sqrt_softplus_l8"):
        ops._device_module(UNREGISTERED)
    with pytest.raises(ValueError, match="expert_stream_exl3_sigmoid_l16"):
        ops.device_module_with_hooks(["EXPERT_STREAM_TEST_X"], ("exl3", "sigmoid"), lanes=16)


def test_a_bare_format_name_is_not_a_build_key():
    with pytest.raises(ValueError, match=r"\(format, router\) key"):
        ops._host_module("exl3", "prod")
    with pytest.raises(ValueError, match=r"\(format, router\) key"):
        ops.host_layout("exl3")


def _names(header: str) -> list[str]:
    text = (MOE / "formats" / header / "row_layout.h").read_text()
    block = re.search(r"kNames = \{(.*?)\};", text, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def _mask(header: str, field: str) -> int:
    text = (MOE / "formats" / header / "row_layout.h").read_text()
    return int(re.search(rf"{field} = 0b([01]+);", text).group(1), 2)


def test_the_nvfp4_row_is_the_streamed_tensor_tuple_in_order():
    from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS

    names = _names("nvfp4")
    assert tuple(names) == NVFP4_STREAM_TENSORS
    small, second = _mask("nvfp4", "kSmallMask"), _mask("nvfp4", "kSecondStageMask")
    assert {n for i, n in enumerate(names) if small >> i & 1} == {"g1_alphas", "g2_alphas"}
    assert {n for i, n in enumerate(names) if second >> i & 1} == {"w2_weight", "w2_blockscale_swizzled"}


def test_the_exl3_row_is_unchanged_by_its_move():
    from sglang.srt.layers.moe.expert_engine.formats.exl3 import EXL3_STREAMED_NAMES

    assert tuple(_names("exl3")) == EXL3_STREAMED_NAMES
    assert _mask("exl3", "kSmallMask") == 0b110110 and _mask("exl3", "kSecondStageMask") == 0b111000


def test_the_mxfp4_row_holds_the_flashinfer_weights_and_the_two_per_expert_side_tensors():
    names = _names("mxfp4")
    assert names == [
        "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale", "mxfp4_weight_global_scale", "swiglu_limit",
    ]
    small, second = _mask("mxfp4", "kSmallMask"), _mask("mxfp4", "kSecondStageMask")
    assert {n for i, n in enumerate(names) if small >> i & 1} == {"mxfp4_weight_global_scale", "swiglu_limit"}
    assert {n for i, n in enumerate(names) if second >> i & 1} == {"w2_weight", "w2_weight_scale"}
    # The method indexes these two per expert (Mxfp4FlashinferCutlassMoEMethod).
    method = Path(ops.__file__).resolve().parents[3] / "srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py"
    text = method.read_text()
    assert "_mxfp4_weight_global_scale_tensor" in text and "_swiglu_limit_tensor" in text


def test_the_generic_engine_files_name_no_format():
    for path in ("engine_host.cpp", "engine_device.cuh", "engine_select.h", "format.h", "router.h"):
        text = (MOE / "expert_stream" / path).read_text().lower()
        assert not re.search(r"exl3|nvfp4|mxfp4", text), path


CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")


def _compile(tmp_path: Path, source: str, *flags: str) -> Path:
    src, out = tmp_path / "probe.cpp", tmp_path / "probe"
    src.write_text(textwrap.dedent(source))
    subprocess.run([CXX, "-std=c++20", f"-I{MOE}", *flags, str(src), "-o", str(out)], check=True, capture_output=True)
    return out


@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
@pytest.mark.parametrize("key", KEYS)
def test_the_engine_select_header_compiles_for_every_registered_pair(tmp_path, key):
    """engine_select.h picks the format and router by macro and static_asserts both concepts."""
    fmt, router = key
    _compile(
        tmp_path,
        """
        #include "expert_stream/engine_select.h"
        int main() { return sglang::expert_stream::EngineLayout::kNames.size() == 0; }
        """,
        f"-DSGLANG_EXPERT_FORMAT={fmt}",
        f"-DSGLANG_EXPERT_ROUTER={router}",
    )


@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_unsupported_cpu_kernel_is_not_usable_and_aborts_naming_its_format(tmp_path):
    probe = _compile(
        tmp_path,
        """
        #include "formats/exl3/format.h"
        #include "formats/mxfp4/format.h"
        #include <cstring>
        using namespace sglang;
        int main(int argc, char** argv) {
          using Stub = formats::mxfp4::Format::CpuKernel;
          static_assert(!Stub::usable());
          static_assert(formats::exl3::Format::CpuKernel::usable());
          if (std::strcmp(Stub::name(), "mxfp4") != 0 || Stub::max_routes() != 0) return 3;
          Stub::Layer layer;
          if (argc > 1 && !std::strcmp(argv[1], "make_layer")) Stub::make_layer(layer);
          if (argc > 1 && !std::strcmp(argv[1], "check")) Stub::check(layer, cpu_experts::ForwardCall{});
          return 0;
        }
        """,
    )

    def run(*args):
        return subprocess.run(
            [str(probe), *args],
            capture_output=True,
            text=True,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),  # an abort must not write a core file
        )

    assert run().returncode == 0
    for call in ("make_layer", "check"):
        result = run(call)
        assert result.returncode < 0, result  # SIGABRT
        assert "mxfp4 CPU experts" in result.stderr and call in result.stderr, result.stderr


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
