"""The cast-fusion launchers refuse a tensor that is not on a CUDA device, even when called without the wrapper.

Both launchers dereference ``input.data_ptr()`` in a kernel. A CPU input would be read as a device address. Each case
passes a CPU input straight to the JIT module's ``run``, with a valid CUDA output, and expects the launcher's
``TensorMatcher`` to raise before it launches. Writing ``.with_device(device)`` without ``<kDLCUDA>`` in either
launcher turns its case red: the untemplated call resets the allowed devices to "any" (the ``bdcf769eb8`` footgun), so
the CPU tensor is accepted.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_silu_mul_clamp_half_launcher_refuses_a_cpu_input():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _silu_module

    gate_up = torch.zeros(1, 16, dtype=torch.float16)
    out = torch.empty(1, 8, dtype=torch.float16, device="cuda")
    with pytest.raises(Exception, match="not in the allowed options"):
        _silu_module().run(gate_up, out, 7.0)


def test_scale_to_bf16_launcher_refuses_a_cpu_input():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _scale_module

    routed = torch.zeros(8, dtype=torch.float32)
    out = torch.empty(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception, match="not in the allowed options"):
        _scale_module().run(routed, out, 1.5)
