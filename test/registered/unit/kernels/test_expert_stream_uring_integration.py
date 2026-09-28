"""Python tables -> native FFI -> RowReader with real registered-file/buffer I/O."""

import errno
import os
from dataclasses import replace

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab_arena
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

from test_expert_stream_uring_native import _run_native, uring_native_binary  # noqa: F401

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _use_shared_slab_arenas(setup):
    """Retain one complete allocation per layer, containing every readv destination."""
    capacity = int(setup.tables.capacity[0])
    specs = {
        name: (setup.specs[name].row_shape, setup.specs[name].dtype)
        for name in EXL3_STREAMED_NAMES
    }
    setup.slabs = {
        layer: allocate_host_slab_arena(capacity, specs, register=False)
        for layer in setup.tables.layer_ids
    }
    owners = tuple(
        setup.slabs[layer][name]
        for layer in setup.tables.layer_ids
        for name in EXL3_STREAMED_NAMES
    )
    addresses = torch.tensor(
        [[setup.slabs[layer][name].data_ptr() for name in EXL3_STREAMED_NAMES]
         for layer in setup.tables.layer_ids],
        dtype=torch.int64,
    )
    setup.tables = replace(setup.tables, slabs=addresses, keepalive=owners)


def _assert_rows(setup, layer, experts, slots):
    reference = setup.reference(layer, experts)
    for lane, slot in enumerate(slots):
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(setup.slabs[layer][name][slot], reference[name][lane]), (name, experts[lane], slot)


@pytest.mark.parametrize("mode", ["default", "iopoll", "sqpoll", "sqpoll_iopoll"])
@pytest.mark.parametrize("row_images", [False, True], ids=["fixed-shard-bounce", "readv-fixed-shared-arena"])
def test_registered_io_through_tables_and_ffi_recovers_after_request_failure(
    tmp_path, monkeypatch, uring_native_binary, mode, row_images
):
    read_mode = "readv_fixed" if row_images else "fixed"
    # An optional feature may skip only after the native harness probes the
    # running kernel and verifies production refusal; FFI failures below fail.
    _run_native(uring_native_binary, tmp_path, mode=mode, read_mode=read_mode, fixed_files=True)
    for name in list(os.environ):
        if name.startswith("SGLANG_EXPERT_STREAM_URING_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_MODE", mode)
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_READ_MODE", read_mode)
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_FIXED_FILES", "1")
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH", "2")
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS", "1")
    setup = ram_miss_setup(
        tmp_path / "checkpoint", capacity=3, experts=6, mirror_weights=(1.0, 1.0),
        hidden=256, inter=256, row_images=row_images,
    )
    if row_images:
        _use_shared_slab_arenas(setup)
        # The metadata crossing FFI must describe two complete arenas, rather
        # than six separately registered named slabs per layer.
        regions = ops._table_buffer_regions(setup.tables)
        assert regions.shape == (len(setup.tables.layer_ids), 2)
    assert setup.tables.row_images == row_images

    experts, slots = [5, 0, 3], [2, 0, 1]
    result, trace = ops.read_rows_traced(setup.tables, 1, experts, slots, direct=True)
    assert result == 1
    assert 0 < trace["pending_max"] <= 2
    assert trace["bytes"] > 0
    _assert_rows(setup, 1, experts, slots)

    # The same native reader sees a failed request and then a clean one. Cover
    # both SQEs still in userspace and submitted reads whose buffers stay live
    # until drain completes; SQPOLL may consume either set asynchronously.
    for submitted_first in (False, True):
        for slab in setup.slabs[1].values():
            slab.view(torch.uint8).fill_(0xA7)
        recovered_experts, recovered_slots = [4, 1, 2], [1, 2, 0]
        stats = {}
        failed, recovered = ops.read_rows_with_fault(
            setup.tables, 1, [0, 5, 3], [0, 1, 2], recovered_experts, recovered_slots,
            direct=True, submit_error=errno.EIO, submit_call=1, submit_first=submitted_first,
            stats=stats,
        )
        assert (failed, recovered) == (0, 1)
        assert stats["stale_cqes"] == 0
        assert stats["unfinished_jobs"] == 0
        _assert_rows(setup, 1, recovered_experts, recovered_slots)
