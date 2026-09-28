"""The cast-fusion launchers refuse a tensor that is not on a CUDA device, even when called without the wrapper.

Both launchers dereference ``input.data_ptr()`` in a kernel, so a CPU input would be read as a device address. Each case
calls the JIT module's ``run`` directly and expects the launcher's ``TensorMatcher`` to raise the device refusal
("Device value [...] not in the allowed options") on ``input`` before it launches.

Writing ``.with_device(device)`` without ``<kDLCUDA>`` in either launcher turns its cases red: the untemplated call
resets the allowed devices to "any" (the ``bdcf769eb8`` footgun), so the CPU input is accepted and binds the symbolic
device to cpu. With a CUDA output, only the output's "Device mismatch" then stops the launch, which the ``match`` does
not accept. With a CPU output too (the all-CPU cases), nothing stops it, and the kernel launches on host pointers.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEVICE_REFUSAL = r"^input: [\s\S]*Device value .* not in the allowed options"


def test_silu_mul_clamp_half_launcher_refuses_a_cpu_input():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _silu_module

    gate_up = torch.zeros(1, 16, dtype=torch.float16)
    out = torch.empty(1, 8, dtype=torch.float16, device="cuda")
    with pytest.raises(Exception, match=DEVICE_REFUSAL):
        _silu_module().run(gate_up, out, 7.0)


def test_silu_mul_clamp_half_launcher_refuses_all_cpu_tensors():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _silu_module

    gate_up = torch.zeros(1, 16, dtype=torch.float16)
    out = torch.empty(1, 8, dtype=torch.float16)
    with pytest.raises(Exception, match=DEVICE_REFUSAL):
        _silu_module().run(gate_up, out, 7.0)


def test_scale_to_bf16_launcher_refuses_a_cpu_input():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _scale_module

    routed = torch.zeros(8, dtype=torch.float32)
    out = torch.empty(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception, match=DEVICE_REFUSAL):
        _scale_module().run(routed, out, 1.5)


def test_scale_to_bf16_launcher_refuses_all_cpu_tensors():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _scale_module

    routed = torch.zeros(8, dtype=torch.float32)
    out = torch.empty(8, dtype=torch.bfloat16)
    with pytest.raises(Exception, match=DEVICE_REFUSAL):
        _scale_module().run(routed, out, 1.5)
