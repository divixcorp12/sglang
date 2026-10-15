"""SGLANG_DSV41_CPU_EXPERTS_STORE_OUT on the CPU expert engine (cpu_experts.h run_job): a verify's per-token job leaves
its overwrite to the kernel instead of zeroing its token rows and then accumulating.

The kernel is the instr build's fake CpuExpertKernel (ExpertStreamHost.test_kernel_address), driven by the split
calibration's per-token jobs (every job an overwrite of part 0, each lane routed by about 1.25 of the tokens, so one
token routes none). The fake marks an overwrite: it writes j + sum to column j of every row, where an accumulate adds
sum to what the row held. So the rows tell which the engine did: with the flag off, zeros plus sum (the engine zeroed
them, then accumulated); with it on, j + sum (the kernel overwrote them, the unrouted token's row too). That the real
EXL3 kernel's overwrite gives the bits of the old zero-then-accumulate is test/manual/dsv41/test_exl3_cpu_store_out.py.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import fake_cpu_layer, ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROW, ROWS, HIDDEN, DST_ROWS, EXPERTS, LANES = 1, 2, 8, 6, 8, 4


def _host(tmp_path, *, nodes, tokens, store_out):
    """Groups 0..nodes-1, each with its own two cores and half of the row's 20 slots, writing into out_rows."""
    s = ram_miss_setup(tmp_path, capacity=20, experts=EXPERTS, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    node_ranges = [[(10 * g, 10 * g + 10)] * 2 for g in range(nodes)] if nodes > 1 else None
    host = ExpertStreamHost(
        s.tables, page=new_page(pin=False, wire=lease.wire_layout(8, nodes)),
        slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), variant="instr", node_ranges=node_ranges,
    )
    host.reserve_staging(2)
    host.enable_copy_engine(-1)
    host.copy_engine_release(-1)  # the CPU test backend completes only released marks
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    if tokens == 1:
        x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
        out_rows = torch.full((ROWS, 2 * nodes, HIDDEN), 7.0, dtype=torch.float32)
    else:  # a verify's rows: the tokens' inputs and the token table (cpu_token_table.h)
        x_rows = torch.zeros((ROWS, lease.cpu_row_bytes(HIDDEN, tokens, host.wire.lanes)), dtype=torch.uint8)
        out_rows = torch.full((ROWS, 2 * nodes, tokens, HIDDEN), 7.0, dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < 2 * nodes:
        pytest.skip(f"needs {2 * nodes} cores in the affinity mask")
    kernel = host.test_kernel_address(1000)  # the fake's spin keeps the blocking calibration from deadlocking
    for group in range(nodes):
        host.enable_cpu_experts(
            kernel, [0] * (host.wire.lanes + 1), cores[2 * group : 2 * group + 2], x_rows, out_rows, threads=2,
            group=group, store_out=store_out,
        )
    host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    return host, (s, dst, x_rows, out_rows)


def _calibrate(host, group, tokens):
    scratch = torch.zeros(LANES * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    before = len(host.test_kernel_calls())
    host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch, group=group, lanes=LANES, tokens=tokens)
    return host.test_kernel_calls()[before:]


@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 6])
@pytest.mark.parametrize("store_out", [False, True], ids=["zeroed", "stored"])
def test_a_per_token_job_leaves_its_overwrite_to_the_kernel_only_with_store_out(tmp_path, nodes, tokens, store_out):
    """Mutations: run_job ignores store_out (the stored case reads zeros + sum and accumulate calls); run_job skips the
    zeroing without store_out (the zeroed case reads 7 + sum); the FFI drops store_out (as the first)."""
    host, keep = _host(tmp_path, nodes=nodes, tokens=tokens, store_out=store_out)
    out_rows = keep[-1]
    try:
        for group in range(nodes):
            calls = _calibrate(host, group, tokens)
            assert calls and len(calls) % tokens == 0
            per_token = tokens > 1
            # A one-token row's job is an overwrite either way; a verify's is one only when the kernel owns it.
            assert {c["accumulate"] for c in calls} == {per_token and not store_out}
            last = calls[-tokens:]  # the last job's rows, in token order; it is what part 0 holds now
            part = out_rows[ROW, 2 * group]
            rows = part.reshape(tokens, HIDDEN)
            for t, call in enumerate(last):
                total = sum(w * (s + 1) for s, w in zip(call["slots"], call["weights"]))
                base = [0.0] * HIDDEN if per_token and not store_out else [float(j) for j in range(HIDDEN)]
                assert rows[t].tolist() == pytest.approx([b + total for b in base]), (group, t)
            if tokens == 6:  # of 4 lanes, lane % 6 routes tokens 0-3 and lane 3 token 4 too: token 5 routes none
                assert all(s < 0 for s in last[5]["slots"]) and rows[5].tolist() == (
                    [0.0] * HIDDEN if not store_out else [float(j) for j in range(HIDDEN)]
                )
            other = out_rows[ROW, 2 * (1 - group)] if nodes == 2 else None
            if other is not None and group == 0:
                assert (other == 7.0).all(), "group 0's jobs write only its own parts"
    finally:
        host.stop()
