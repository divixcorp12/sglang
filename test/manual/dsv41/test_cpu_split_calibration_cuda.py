"""Startup split calibration with the real CUDA copy backend (GPU; spec 2026-10-01-cpu-split-calibration).

The DMA runs on the calibration's own stream from pinned host rows into a VRAM scratch buffer; the CPU side is the
instr build's native fake forward, so this checks the link measurement, not the kernel.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

ROW, ROWS, HIDDEN, LANES = 1, 2, 8, 8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_calibration_measures_a_link_that_grows_with_the_experts(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, mirror_weights=(1.0, 1.0), hidden=2048, inter=4096)
    host = attached_host(s, new_page(pin=False), k=3)
    host.enable_copy_engine(-1, spin_us=200)
    pinned = {n: t.pin_memory() for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), t.data_ptr(), t[0].numel() * t.element_size()] for t in pinned.values()], dtype=torch.int64
    )
    host.set_copy_table(ROW, table, 1)
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(host.test_forward_address(100_000), [0] * 9, cores, x_rows, out_rows, threads=2, spin_us=200)
    host.set_cpu_layer(ROW, 7)
    scratch = torch.empty(LANES * host.copy_expert_bytes(ROW), dtype=torch.uint8, device="cuda")
    grid = host.calibrate_cpu_split(ROW, device=torch.cuda.current_device(), reps=5, scratch=scratch)
    link = grid[1, 1:].tolist()
    print("link ms m=1..8:", " ".join(f"{v:.3f}" for v in link))
    assert all(v > 0 for v in link)
    assert link[7] > link[0], link
    for n in range(1, LANES + 1):
        assert all(grid[1 + n, k] > 0 for k in range(n + 1))
