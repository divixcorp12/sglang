"""The split model's startup calibration end to end on a small fixture (GPU; split_model_calibration.py).

The host times the fake CPU kernel's jobs (k * FORWARD_NS each, whatever their tokens), DMAs through the real CUDA
copy backend, NVMe reads through the fixture's reader, and the stream pull (calibration_pull) against copy-engine
lanes; calibrate() fits and clamps them. This checks the plumbing and the fit's shape, not the real model's costs.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_lease_block import cpu_row_bytes, wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.srt.layers.moe import split_model_calibration as calib
from sglang.test.dsv41_ram_miss_fixtures import attached_host, fake_cpu_layer, ram_miss_setup

ROW, ROWS, HIDDEN, TOKENS = 1, 2, 8, 6
FORWARD_NS = 100_000

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_calibrate_fits_every_cost_on_a_small_fixture(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, mirror_weights=(1.0, 1.0), hidden=2048, inter=4096)
    host = attached_host(s, new_page(pin=False, wire=wire_layout(8)), k=4)
    host.enable_copy_engine(-1)
    pinned = {n: t.pin_memory() for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), t.data_ptr(), t[0].numel() * t.element_size()] for t in pinned.values()], dtype=torch.int64
    )
    host.set_copy_table(ROW, table, 1)
    x_rows = torch.zeros((ROWS, cpu_row_bytes(HIDDEN, TOKENS, host.wire.lanes)), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, TOKENS, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(host.test_kernel_address(FORWARD_NS), [0] * 9, cores, x_rows, out_rows, threads=2)
    host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    service = SimpleNamespace(
        group=0, layers={ROW: None}, tokens=TOKENS, _group_slots=lambda r: 12, slabs_by_row={ROW: pinned},
        trait=SimpleNamespace(slab_names=list(pinned)),
    )
    module = ops._device_module(ops.EXL3_BUILD, 8, 1)
    device_side = SimpleNamespace(calibration_pull=module.expert_stream_calibration_pull)
    costs, clamped, seconds = calib.calibrate(
        host, [service], device_side, device=torch.cuda.current_device(), reps=5, staging_width=4
    )
    print(f"calibrated in {seconds:.2f} s, clamped {clamped}: {costs.line()}")
    # The fake kernel: k * FORWARD_NS a job, so b is FORWARD_NS and c is near 0.
    assert math.isclose(costs.hit_b[0], FORWARD_NS / 1e6, rel_tol=0.25), costs.hit_b
    assert abs(costs.hit_c[0]) < 0.02, costs.hit_c
    assert costs.copy_lane > 0 and costs.per_expert > 0 and costs.stream_base > 0
    assert 0 <= costs.stream_lane <= calib.RANGES_MS["stream_lane"][1]
    assert seconds < 10
