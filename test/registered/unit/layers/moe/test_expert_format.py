"""CPU tests for the expert format seam: specs, dense sources and streamer discovery."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertTensorSpec,
    expert_streamer_of,
    iter_expert_streamers,
)
from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXPERTS = 4
HIDDEN = 512
INTERMEDIATE = 256


def _nvfp4_layer(experts=EXPERTS):
    """CPU parameters with the production NVFP4 streamed shapes and dtypes."""
    shapes = {
        "w13_weight": ((experts, 2 * INTERMEDIATE, HIDDEN // 2), torch.uint8),
        "w2_weight": ((experts, HIDDEN, INTERMEDIATE // 2), torch.uint8),
        "w13_blockscale_swizzled": (
            (experts, 2 * INTERMEDIATE, HIDDEN // 16),
            torch.float8_e4m3fn,
        ),
        "w2_blockscale_swizzled": (
            (experts, HIDDEN, INTERMEDIATE // 16),
            torch.float8_e4m3fn,
        ),
        "g1_alphas": ((experts,), torch.float32),
        "g2_alphas": ((experts,), torch.float32),
    }
    generator = torch.Generator().manual_seed(3)
    layer = torch.nn.Module()
    for name, (shape, dtype) in shapes.items():
        if dtype is torch.float32:
            values = torch.rand(shape, generator=generator)
        else:
            values = torch.randint(
                0, 256, shape, dtype=torch.uint8, generator=generator
            ).view(dtype)
        setattr(layer, name, torch.nn.Parameter(values, requires_grad=False))
    return layer


def _legacy_bytes(layer, experts, host_only):
    """ExpertStreamer's byte counts as computed before formats existed."""
    total = 0
    for name in NVFP4_STREAM_TENSORS:
        tensor = getattr(layer, name).data
        if host_only and tensor.device.type != "cpu":
            continue
        total += tensor.numel() * tensor.element_size() // experts
    return total


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


def _as_bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


class TestDenseLayerFormat(unittest.TestCase):
    def test_specs_equal_source_row_shapes_on_an_nvfp4_layer(self):
        layer = _nvfp4_layer()
        specs = DenseLayerFormat(NVFP4_STREAM_TENSORS).tensor_specs(layer)
        self.assertEqual(tuple(spec.name for spec in specs), NVFP4_STREAM_TENSORS)
        for spec in specs:
            source = getattr(layer, spec.name).data
            self.assertEqual(spec.row_shape, tuple(source.shape[1:]))
            self.assertEqual(spec.dtype, source.dtype)
            self.assertEqual(spec.residence, "host")
            self.assertEqual(
                spec.row_bytes, source.numel() * source.element_size() // EXPERTS
            )

    def test_streamer_byte_counts_match_the_legacy_formula(self):
        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        self.assertIsInstance(streamer.format, DenseLayerFormat)
        self.assertEqual(streamer.num_experts, EXPERTS)
        self.assertEqual(
            streamer.bytes_per_expert, _legacy_bytes(layer, EXPERTS, host_only=False)
        )
        self.assertEqual(
            streamer.host_bytes_per_expert,
            _legacy_bytes(layer, EXPERTS, host_only=True),
        )
        self.assertEqual(
            streamer.specs, DenseLayerFormat(NVFP4_STREAM_TENSORS).tensor_specs(layer)
        )
        self.assertEqual(streamer.spec("g1_alphas").row_shape, ())

    def test_sources_are_looked_up_on_every_call(self):
        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        replacement = layer.w13_weight.data.clone()
        # ExpertHostArena.bind rebinds parameters exactly like this.
        layer.w13_weight.data = replacement
        self.assertEqual(
            streamer.source("w13_weight").data_ptr(), replacement.data_ptr()
        )
        plain = torch.nn.Module()
        plain.rows = torch.zeros(4, 3)
        streamer = ExpertStreamer(plain, ("rows",))
        plain.rows = torch.ones(4, 3)
        self.assertTrue(torch.equal(streamer.source("rows"), torch.ones(4, 3)))

    def test_file_source_bytes_follow_the_layer_attribute(self):
        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        self.assertIsNone(streamer.file_source_bytes_per_expert)
        layer._nvfp4_file_source_bytes_per_expert = 12
        self.assertEqual(streamer.file_source_bytes_per_expert, 12)
        # ExpertHostArena.bind drops the attribute like this.
        layer.__dict__.pop("_nvfp4_file_source_bytes_per_expert")
        self.assertIsNone(streamer.file_source_bytes_per_expert)

    def test_invalid_dense_sources_keep_their_errors(self):
        cases = (
            ({"a": torch.zeros(4, 2)}, ("a", "b"), "'b' is missing"),
            ({"a": torch.tensor(1.0)}, ("a",), "has no expert dimension"),
            ({"a": torch.zeros(0, 4)}, ("a",), "has no expert rows"),
            ({"a": torch.zeros(4, 6)[:, ::2]}, ("a",), "must be contiguous"),
            ({"a": torch.zeros(4, 2, device="meta")}, ("a",), "unsupported device"),
            (
                {"a": torch.zeros(4, 2), "b": torch.zeros(5, 2)},
                ("a", "b"),
                "expert count mismatch",
            ),
        )
        for tensors, names, message in cases:
            layer = torch.nn.Module()
            for name, tensor in tensors.items():
                setattr(layer, name, tensor)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ExpertStreamer(layer, names)

    def test_a_format_whose_specs_disagree_with_its_source_is_rejected(self):
        class WrongShape(DenseLayerFormat):
            def tensor_specs(self, layer):
                return tuple(
                    ExpertTensorSpec(
                        spec.name, spec.row_shape + (1,), spec.dtype, spec.residence
                    )
                    for spec in super().tensor_specs(layer)
                )

        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, "does not match its spec"):
            ExpertStreamer(layer, ("rows",), format=WrongShape(("rows",)))

    def test_format_names_must_match_the_streamer_names(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        layer.other = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, "do not match tensor names"):
            ExpertStreamer(layer, ("rows",), format=DenseLayerFormat(("other",)))

    def test_default_row_source_kinds(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        dense = DenseLayerFormat(("rows",))
        specs = dense.tensor_specs(layer)
        with envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"):
            self.assertIsNone(dense.default_row_source(layer, specs, "auto"))
            self.assertIsNone(dense.default_row_source(layer, specs, "tensor"))
            with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_FILE_READER"):
                dense.default_row_source(layer, specs, "files")
        with self.assertRaisesRegex(ValueError, "no row source kind 'shards'"):
            dense.default_row_source(layer, specs, "shards")


class TestStreamerDiscovery(unittest.TestCase):
    def test_helpers_find_streamers_in_module_order(self):
        model = torch.nn.Module()
        first, plain, second = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
        first._nvfp4_expert_streamer = "first"
        second._nvfp4_expert_streamer = "second"
        model.add_module("a", first)
        model.add_module("b", plain)
        model.add_module("c", second)
        self.assertEqual(list(iter_expert_streamers(model)), ["first", "second"])
        self.assertEqual(expert_streamer_of(first), "first")
        self.assertIsNone(expert_streamer_of(plain))


class TestSpecStagingShapes(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()

    def test_cached_gather_stages_rows_in_their_source_shapes(self):
        layer = _nvfp4_layer(experts=8)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)

        def copy_rows(source_ids, outputs):
            for name, output in outputs.items():
                torch.index_select(getattr(layer, name).data, 0, source_ids, out=output)
            return 0

        streamer._copy_source_rows = copy_rows
        ids = torch.tensor([[1, 5], [5, 2]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_cached(source_ids, compact_ids, ids)
        for name in NVFP4_STREAM_TENSORS:
            source = getattr(layer, name).data
            self.assertEqual(tuple(tensors[name].shape), (64,) + tuple(source.shape[1:]))
            self.assertEqual(tensors[name].dtype, source.dtype)
            self.assertTrue(
                torch.equal(
                    _as_bytes(tensors[name][compact.long()]), _as_bytes(source[ids])
                ),
                name,
            )


class TestRowSourceKnob(unittest.TestCase):
    def _layer(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        return layer

    def test_auto_keeps_the_mmap_default_of_no_reader(self):
        with (
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("auto"),
            envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"),
        ):
            self.assertIsNone(ExpertStreamer(self._layer(), ("rows",)).row_source)

    def test_tensor_kind_builds_no_reader(self):
        with (
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("tensor"),
            envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"),
        ):
            self.assertIsNone(ExpertStreamer(self._layer(), ("rows",)).row_source)

    def test_files_kind_needs_an_io_uring_reader_and_expert_files(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("files"):
            with envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"):
                with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_FILE_READER"):
                    ExpertStreamer(self._layer(), ("rows",))
            with envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"):
                with self.assertRaisesRegex(ValueError, "has no expert file"):
                    ExpertStreamer(self._layer(), ("rows",))

    def test_unknown_kind_is_refused(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
            with self.assertRaisesRegex(ValueError, "no row source kind 'shards'"):
                ExpertStreamer(self._layer(), ("rows",))

    def test_an_explicit_row_source_ignores_the_knob(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
            streamer = ExpertStreamer(self._layer(), ("rows",), row_source=None)
        self.assertIsNone(streamer.row_source)


if __name__ == "__main__":
    unittest.main()
