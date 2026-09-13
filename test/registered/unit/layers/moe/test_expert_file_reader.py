"""io_uring expert-row reads against the verified NVFP4 expert file cache."""

import os
import tempfile

import pytest
import torch

from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader
from sglang.srt.model_loader.file_tensor_cache import (
    FileTensorCacheGroup,
    FileTensorSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/include/liburing.h"),
    reason="io_uring expert reader tests require liburing headers.",
)

PAGE = 4096


def _bound_layer(directory):
    specs = (
        FileTensorSpec("w13_weight", (6, 2, PAGE), (2 * PAGE, PAGE, 1), torch.uint8),
        FileTensorSpec(
            "w13_blockscale_swizzled", (6, PAGE), (PAGE, 1), torch.float8_e4m3fn
        ),
    )
    group = FileTensorCacheGroup.open(directory, "expert_reader_test", {"k": 1}, specs)
    generator = torch.Generator().manual_seed(11)
    layer = torch.nn.Module()
    for spec in specs:
        tensor = group.tensors[spec.tag]
        tensor.view(torch.uint8).copy_(
            torch.randint(
                0,
                256,
                tensor.view(torch.uint8).shape,
                dtype=torch.uint8,
                generator=generator,
            )
        )
        parameter = torch.nn.Parameter(tensor, requires_grad=False)
        parameter._sglang_file_cache_group = group
        parameter._sglang_file_cache_tag = spec.tag
        setattr(layer, spec.tag, parameter)
    return group, layer, tuple(spec.tag for spec in specs)


def _aligned_like(tensor, rows):
    row_bytes = tensor[0].numel() * tensor.element_size()
    storage = torch.empty(rows * row_bytes + PAGE, dtype=torch.uint8)
    start = (-storage.data_ptr()) % PAGE
    return (
        storage[start : start + rows * row_bytes]
        .view(tensor.dtype)
        .view((rows,) + tuple(tensor.shape[1:]))
    )


@pytest.mark.parametrize("mode", ["uring", "uring_direct"])
def test_reads_expert_rows_into_slots_in_one_batch(mode):
    with tempfile.TemporaryDirectory() as directory:
        group, layer, names = _bound_layer(directory)
        try:
            reader = ExpertFileRowReader.from_layer(layer, names, mode=mode)
            assert reader is not None and reader.names == names
            destinations = {
                name: _aligned_like(getattr(layer, name).data, 4) for name in names
            }
            reader.register_destinations(destinations.values())
            rows = torch.tensor([5, 0, 3])
            slots = torch.tensor([1, 3, 0])

            reader.read(rows, destinations, slots)

            for name in names:
                source = getattr(layer, name).data
                assert torch.equal(
                    destinations[name][slots].view(torch.uint8),
                    source[rows].view(torch.uint8),
                )
        finally:
            group.close()


def test_mmap_mode_builds_no_reader_and_unbound_tensors_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        group, layer, names = _bound_layer(directory)
        try:
            assert ExpertFileRowReader.from_layer(layer, names, mode="mmap") is None
            layer.w2_weight = torch.nn.Parameter(
                torch.zeros(6, PAGE, dtype=torch.uint8), requires_grad=False
            )
            with pytest.raises(ValueError, match="no expert file"):
                ExpertFileRowReader.from_layer(
                    layer, names + ("w2_weight",), mode="uring"
                )
            reader = ExpertFileRowReader.from_layer(layer, names, mode="uring")
            with pytest.raises(ValueError, match="does not cover"):
                reader.read(torch.tensor([0]), {"w2_weight": layer.w2_weight.data[:1]})
        finally:
            group.close()


def test_tensor_not_at_file_offset_zero_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        group, layer, names = _bound_layer(directory)
        try:
            shifted = torch.nn.Parameter(layer.w13_weight.data[1:], requires_grad=False)
            shifted._sglang_file_cache_group = group
            shifted._sglang_file_cache_tag = "w13_weight"
            layer.w13_weight = shifted
            with pytest.raises(ValueError, match="offset zero"):
                ExpertFileRowReader.from_layer(layer, names, mode="uring")
        finally:
            group.close()
