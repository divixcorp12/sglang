"""The expert host arena moves host rows off file mappings into registered memory."""

import os
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.srt.model_loader.file_tensor_cache import (
    FileTensorCacheGroup,
    FileTensorSpec,
)
from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")

PAGE = 4096
_FILE_TAGS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertHostArena(unittest.TestCase):
    def _file_backed_model(self, directory):
        specs = (
            FileTensorSpec(_FILE_TAGS[0], (4, PAGE), (PAGE, 1), torch.uint8),
            FileTensorSpec(_FILE_TAGS[1], (4, 2 * PAGE), (2 * PAGE, 1), torch.uint8),
            FileTensorSpec(_FILE_TAGS[2], (4, PAGE), (PAGE, 1), torch.float8_e4m3fn),
            FileTensorSpec(_FILE_TAGS[3], (4, PAGE), (PAGE, 1), torch.float8_e4m3fn),
        )
        group = FileTensorCacheGroup.open(directory, "arena_test", {"k": 1}, specs)
        generator = torch.Generator().manual_seed(9)
        layer = torch.nn.Module()
        for name, spec in zip(NVFP4_STREAM_TENSORS[:4], specs):
            mapped = group.tensors[spec.tag]
            mapped.view(torch.uint8).copy_(
                torch.randint(
                    0,
                    256,
                    mapped.view(torch.uint8).shape,
                    dtype=torch.uint8,
                    generator=generator,
                )
            )
            parameter = torch.nn.Parameter(mapped, requires_grad=False)
            parameter._sglang_file_cache_group = group
            parameter._sglang_file_cache_tag = spec.tag
            setattr(layer, name, parameter)
        for name in NVFP4_STREAM_TENSORS[4:]:
            setattr(
                layer,
                name,
                torch.nn.Parameter(torch.rand(4, device="cuda"), requires_grad=False),
            )
        layer._nvfp4_file_source_bytes_per_expert = 5 * PAGE
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        model = torch.nn.Module()
        model.add_module("0", layer)
        return group, model, layer

    def test_arena_pins_all_host_rows_off_the_file_mapping(self):
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena

        for mode in ("mmap", "uring_direct"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, {"SGLANG_MOE_EXPERT_FILE_READER": mode}),
            ):
                group, model, layer = self._file_backed_model(directory)
                streamer = layer._nvfp4_expert_streamer
                try:
                    host_names = NVFP4_STREAM_TENSORS[:4]
                    expected = {
                        name: getattr(layer, name).data.clone() for name in host_names
                    }
                    mapped = {
                        name: getattr(layer, name).data_ptr() for name in host_names
                    }
                    reader = streamer.file_row_reader
                    self.assertEqual(reader is None, mode == "mmap")
                    read = (
                        patch.object(reader, "read", wraps=reader.read)
                        if reader is not None
                        else nullcontext()
                    )
                    with read as spy:
                        arena = ExpertHostArena.from_model(model)
                    try:
                        if reader is not None:
                            self.assertEqual(spy.call_count, 1)
                        for name in host_names:
                            data = getattr(layer, name).data
                            self.assertTrue(is_gpu_readable_host_tensor(data), name)
                            self.assertNotEqual(data.data_ptr(), mapped[name])
                            self.assertEqual(data.data_ptr() % PAGE, 0)
                            self.assertTrue(
                                torch.equal(
                                    data.view(torch.uint8),
                                    expected[name].view(torch.uint8),
                                ),
                                name,
                            )
                        self.assertIsNone(streamer.file_row_reader)
                        self.assertFalse(
                            hasattr(layer, "_nvfp4_file_source_bytes_per_expert")
                        )
                        self.assertEqual(
                            arena.nbytes,
                            sum(
                                tensor.numel() * tensor.element_size()
                                for tensor in expected.values()
                            ),
                        )
                        ids = torch.tensor(
                            [[3, 0, 3]], device="cuda", dtype=torch.int32
                        )
                        compact, tensors = streamer.gather(ids)
                        for name in NVFP4_STREAM_TENSORS:
                            source = getattr(layer, name).data
                            self.assertTrue(
                                torch.equal(
                                    tensors[name][compact.long()]
                                    .view(torch.uint8)
                                    .cpu(),
                                    source[ids.long().to(source.device)]
                                    .view(torch.uint8)
                                    .cpu(),
                                ),
                                name,
                            )
                    finally:
                        arena.close()
                    self.assertFalse(is_gpu_readable_host_tensor(layer.w13_weight.data))
                finally:
                    group.close()

    def test_registry_accepts_tensors_spanning_adjacent_registrations(self):
        from sglang.srt.utils.cuda_host_registry import (
            forget_cuda_host_registration,
            is_cuda_host_registered,
            record_cuda_host_registration,
        )

        tensor = torch.empty(3 * PAGE, dtype=torch.uint8)
        base = tensor.data_ptr()
        record_cuda_host_registration(base, PAGE)
        self.addCleanup(forget_cuda_host_registration, base)
        self.assertTrue(is_cuda_host_registered(tensor[:PAGE]))
        self.assertFalse(is_cuda_host_registered(tensor))
        record_cuda_host_registration(base + 2 * PAGE, PAGE)
        self.addCleanup(forget_cuda_host_registration, base + 2 * PAGE)
        self.assertFalse(is_cuda_host_registered(tensor))
        record_cuda_host_registration(base + PAGE, PAGE)
        self.addCleanup(forget_cuda_host_registration, base + PAGE)
        self.assertTrue(is_cuda_host_registered(tensor))
        self.assertTrue(is_cuda_host_registered(tensor[PAGE // 2 : 5 * PAGE // 2]))
        forget_cuda_host_registration(base + PAGE)
        self.assertFalse(is_cuda_host_registered(tensor))
        self.assertFalse(is_cuda_host_registered(tensor[PAGE:]))

    def test_copy_ranges_stop_at_registration_ends(self):
        from sglang.srt.layers.moe.expert_dma import _registration_run_end
        from sglang.srt.utils.cuda_host_registry import (
            cuda_host_registration_end,
            forget_cuda_host_registration,
            record_cuda_host_registration,
        )

        tensor = torch.empty(2 * PAGE, dtype=torch.uint8)
        base = tensor.data_ptr()
        for offset in (0, PAGE):
            record_cuda_host_registration(base + offset, PAGE)
            self.addCleanup(forget_cuda_host_registration, base + offset)

        self.assertIsNone(cuda_host_registration_end(base - 1))
        self.assertEqual(cuda_host_registration_end(base), base + PAGE)
        self.assertEqual(cuda_host_registration_end(base + PAGE - 1), base + PAGE)
        self.assertEqual(cuda_host_registration_end(base + PAGE), base + 2 * PAGE)
        self.assertIsNone(cuda_host_registration_end(base + 2 * PAGE))

        run_end = _registration_run_end(tensor.view(8, PAGE // 4))
        self.assertEqual([run_end(row) for row in (0, 3, 4, 7)], [4, 4, 8, 8])

    def test_model_without_streamers_has_no_arena(self):
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena

        self.assertIsNone(ExpertHostArena.from_model(torch.nn.Linear(2, 2)))
