"""CPU tests for the expert format seam: specs, dense sources and streamer discovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertTensorSpec,
    expert_streamer_of,
    iter_expert_streamers,
)
from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache, HotCacheSlotTicket
from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import CountingRowSource, SpecOnlyFormat

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


def _spec_only_reference(experts=8):
    generator = torch.Generator().manual_seed(9)
    return {
        "w13_trellis": torch.randint(
            -(2**15), 2**15, (experts, 2, 6), dtype=torch.int16, generator=generator
        ),
        "w13_suh": torch.randn(experts, 2, 4, generator=generator).half(),
        "w2_trellis": torch.randint(
            -(2**15), 2**15, (experts, 1, 6), dtype=torch.int16, generator=generator
        ),
    }


class TestSpecOnlyFormat(unittest.TestCase):
    def _streamer(self, row_source=expert_stream._DEFAULT_ROW_SOURCE):
        reference = _spec_only_reference()
        layer = torch.nn.Module()
        layer.layer_id = 0
        streamer = ExpertStreamer(
            layer,
            tuple(reference),
            format=SpecOnlyFormat(reference),
            row_source=row_source,
        )
        return reference, layer, streamer

    def test_specs_describe_the_rows_and_the_row_source_reads_them(self):
        reference, _, streamer = self._streamer()
        row_bytes = sum(t[0].numel() * t.element_size() for t in reference.values())
        self.assertTrue(streamer.has_spec_only_tensors)
        self.assertIsNone(streamer.source("w13_trellis"))
        self.assertIsInstance(streamer.row_source, CountingRowSource)
        self.assertEqual(streamer.num_experts, 8)
        self.assertEqual(streamer.bytes_per_expert, row_bytes)
        self.assertEqual(streamer.host_bytes_per_expert, row_bytes)
        self.assertEqual(streamer.file_source_bytes_per_expert, row_bytes)
        destinations = {
            name: torch.zeros((2,) + tuple(t.shape[1:]), dtype=t.dtype)
            for name, t in reference.items()
        }
        streamer.read_host_rows(torch.tensor([6, 1]), destinations)
        self.assertEqual(len(streamer.row_source.calls), 1)
        for name, tensor in reference.items():
            self.assertTrue(torch.equal(destinations[name], tensor[[6, 1]]), name)

    def test_dense_formats_have_no_spec_only_tensors(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        self.assertFalse(ExpertStreamer(layer, ("rows",)).has_spec_only_tensors)

    def test_the_tensor_kind_is_refused(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("tensor"):
            with self.assertRaisesRegex(ValueError, "no row source kind 'tensor'"):
                self._streamer()

    def test_without_a_row_source_host_reads_are_refused(self):
        _, _, streamer = self._streamer(row_source=None)
        with self.assertRaisesRegex(ValueError, "no row source covers"):
            streamer.read_host_rows(
                torch.tensor([0]), {"w13_trellis": torch.zeros(1, 2, 6, dtype=torch.int16)}
            )

    def test_the_host_arena_refuses_the_format(self):
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena

        _, layer, streamer = self._streamer()
        layer._nvfp4_expert_streamer = streamer
        with self.assertRaisesRegex(ValueError, "does not support the host arena"):
            ExpertHostArena.from_model(torch.nn.Sequential(layer))

    def test_graph_gather_is_refused_before_any_cuda_work(self):
        _, _, streamer = self._streamer()
        with self.assertRaisesRegex(ValueError, "does not support graph gather"):
            streamer.enable_graph_gather(4)

    def test_the_hot_cache_manager_refuses_every_graph_flag(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

        _, layer, streamer = self._streamer()
        layer._nvfp4_expert_streamer = streamer
        model = torch.nn.Sequential(layer)
        common = dict(
            budget_bytes=1 << 20,
            seed_path=None,
            dynamic=False,
            update_prefill_tokens=16,
            min_residence_forwards=0,
            benefit_ratio=1.0,
        )
        for flags in (
            dict(graph_gather_batch_size=1),
            dict(gpu_residency_update=True),
            dict(expert_doorbell=True),
        ):
            with self.subTest(**flags):
                with self.assertRaisesRegex(ValueError, "does not support graph gather"):
                    ExpertHotCacheManager.from_model(model, **common, **flags)

    def test_a_cpu_pinned_tier_admits_rows_through_the_row_source(self):
        reference, _, streamer = self._streamer()
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        self.assertEqual(cache.cached_names, tuple(reference))
        outputs = {
            name: torch.zeros((5,) + tuple(t.shape[1:]), dtype=t.dtype)
            for name, t in reference.items()
        }
        ids = torch.tensor([7, 0, 3, 5, 2])
        result = cache.gather_rows(ids, outputs)
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 5))
        for name, tensor in reference.items():
            self.assertTrue(torch.equal(outputs[name], tensor[ids]), name)
        self.assertTrue(
            all(call.names == tuple(reference) for call in streamer.row_source.calls)
        )

    def test_cached_gather_serves_spec_only_misses_from_the_pinned_tier(self):
        reference, _, streamer = self._streamer()
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)
        ExpertPinnedHostCache(streamer, 2, device="cpu")
        expert_stream._STAGING.clear()
        ids = torch.tensor([[4, 1], [6, 4]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_eager_rows(source_ids, compact_ids, ids)
        expert_stream._STAGING.clear()
        for name, tensor in reference.items():
            self.assertTrue(torch.equal(tensors[name][compact.long()], tensor[ids]), name)
        self.assertEqual(streamer.last_gather_stats.pinned_host_miss_rows, 3)
        self.assertEqual(streamer.last_gather_stats.host_read_rows, 3)


_SIX_NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


class _FixedRoom:
    """A pinned tier stand-in with a fixed number of evictable rows."""

    def __init__(self, rows):
        self.rows = rows

    def evictable_rows(self):
        return self.rows


class TestSpecOnlyPromotion(unittest.TestCase):
    """CPU checks of the hot cache's spec-only promotion control flow.

    ExpertHotCache needs CUDA to construct, so these tests build a bare instance
    with ``__new__`` and set only the attributes the method under test reads.
    """

    def _tickets(self, experts):
        return tuple(
            HotCacheSlotTicket(slot, expert, 1) for slot, expert in enumerate(experts)
        )

    def test_six_spec_only_tensors_promote_in_evictable_chunks(self):
        reference = {
            name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + position
            for position, name in enumerate(_SIX_NAMES)
        }
        layer = torch.nn.Module()
        layer.layer_id = 0
        streamer = ExpertStreamer(layer, _SIX_NAMES, format=SpecOnlyFormat(reference))
        pinned = ExpertPinnedHostCache(
            streamer, 3, device="cpu", is_pinned=lambda expert: expert == 1
        )
        pinned.ensure_rows(torch.tensor([1]))
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.streamer = streamer
        cache._transfer_executor = object()
        calls = []
        cache._load_reserved_in_chunks = lambda tickets, tier: calls.append(
            (tuple(tickets), tier)
        )
        tickets = self._tickets((5, 7, 0))
        cache._load_reserved(tickets)
        self.assertEqual(calls, [(tickets, pinned)])
        self.assertEqual(pinned.evictable_rows(), 2)

    def test_a_failed_chunk_aborts_its_promotion_and_cancels_the_rest(self):
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.device = torch.device("cpu")
        cache.promotion_in_flight = None
        promotion = SimpleNamespace(name="first chunk")

        def prepare(tickets):
            cache.promotion_in_flight = promotion
            return promotion

        aborted, cancelled = [], []

        def abort(staged):
            aborted.append(staged)
            cache.promotion_in_flight = None

        cache._prepare_promotion = prepare
        cache.abort_promotion = abort
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        cache._transfer_executor = SimpleNamespace(
            wait=lambda ticket, stream: (_ for _ in ()).throw(RuntimeError("copy failed"))
        )
        tickets = self._tickets((5, 7, 0))
        drained = []
        with (
            patch("torch.cuda.current_stream", return_value=None),
            patch("torch.cuda.synchronize", side_effect=drained.append),
            patch(
                "sglang.srt.layers.moe.expert_hot_cache.submit_hot_cache_promotions",
                return_value="ticket",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                cache._load_reserved_in_chunks(tickets, _FixedRoom(2))
        # The copies were submitted, so the device is drained before the slots are freed.
        self.assertEqual(drained, [cache.device])
        self.assertEqual(aborted, [promotion])
        self.assertIsNone(cache.promotion_in_flight)
        self.assertEqual(cancelled, [tickets[2:]])

    def test_an_undrainable_device_keeps_the_failed_promotion_in_flight(self):
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.device = torch.device("cpu")
        cache.promotion_in_flight = None
        promotion = SimpleNamespace(name="first chunk")

        def prepare(tickets):
            cache.promotion_in_flight = promotion
            return promotion

        aborted, cancelled = [], []
        cache._prepare_promotion = prepare
        cache.abort_promotion = aborted.append
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        cache._transfer_executor = SimpleNamespace(
            wait=lambda ticket, stream: (_ for _ in ()).throw(RuntimeError("copy failed"))
        )
        tickets = self._tickets((5, 7, 0))
        with (
            patch("torch.cuda.current_stream", return_value=None),
            patch("torch.cuda.synchronize", side_effect=RuntimeError("device lost")),
            patch(
                "sglang.srt.layers.moe.expert_hot_cache.submit_hot_cache_promotions",
                return_value="ticket",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                cache._load_reserved_in_chunks(tickets, _FixedRoom(2))
        # Its slots stay LOADING: no later reservation can reuse them.
        self.assertEqual(aborted, [])
        self.assertIs(cache.promotion_in_flight, promotion)
        self.assertEqual(cancelled, [tickets[2:]])

    def test_no_evictable_slots_cancel_every_ticket(self):
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cancelled = []
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        tickets = self._tickets((5, 7))
        with self.assertRaisesRegex(RuntimeError, "no evictable slots"):
            cache._load_reserved_in_chunks(tickets, _FixedRoom(0))
        self.assertEqual(cancelled, [tickets])

    def _promoting_cache(self, pinned_tier, reserved):
        """A bare hot cache whose promotion steps mirror the real pinned-tier use.

        ``prepare`` admits the chunk to the tier and requires every row there,
        as ``_prepare_promotion`` does; it then marks the chunk reserved, which
        an inclusive ``is_pinned`` turns into protection.
        """
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.device = torch.device("cpu")
        cache.promotion_in_flight = None
        completed, cancelled = [], []

        def prepare(tickets):
            experts = [ticket.expert_id for ticket in tickets]
            pinned_tier.ensure_rows(torch.tensor(experts))
            if any(pinned_tier._expert_to_slot.get(e, -1) < 0 for e in experts):
                raise RuntimeError("promotion needs every row in the pinned host tier")
            reserved.update(experts)
            promotion = SimpleNamespace(experts=tuple(experts))
            cache.promotion_in_flight = promotion
            return promotion

        def complete(promotion):
            cache.promotion_in_flight = None
            completed.append(promotion.experts)

        cache._prepare_promotion = prepare
        cache.complete_promotion = complete
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        cache.wait_for_slot_publication = lambda: None
        cache._transfer_executor = SimpleNamespace(wait=lambda ticket, stream: None)
        return cache, completed, cancelled

    def _promote(self, cache, tickets, pinned_tier):
        with (
            patch(
                "torch.cuda.current_stream",
                return_value=SimpleNamespace(synchronize=lambda: None),
            ),
            patch(
                "sglang.srt.layers.moe.expert_hot_cache.submit_hot_cache_promotions",
                return_value="ticket",
            ),
        ):
            cache._load_reserved_in_chunks(tickets, pinned_tier)

    def _tier(self, capacity, is_pinned):
        reference = {
            name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + position
            for position, name in enumerate(_SIX_NAMES)
        }
        layer = torch.nn.Module()
        layer.layer_id = 0
        streamer = ExpertStreamer(layer, _SIX_NAMES, format=SpecOnlyFormat(reference))
        return reference, ExpertPinnedHostCache(
            streamer, capacity, device="cpu", is_pinned=is_pinned
        )

    def test_promotion_chunks_shrink_as_promoted_rows_become_protected(self):
        # Capacity 3; only 5 becomes protected once reserved. A chunk size fixed at
        # the start (3) would make the second chunk [3, 6, 2] evict one of its own
        # rows; per-chunk sizing gives [5, 7, 0], [3, 6], [2].
        reserved = set()
        reference, tier = self._tier(3, lambda expert: expert in reserved and expert == 5)
        cache, completed, cancelled = self._promoting_cache(tier, reserved)
        self._promote(cache, self._tickets((5, 7, 0, 3, 6, 2)), tier)
        self.assertEqual(completed, [(5, 7, 0), (3, 6), (2,)])
        self.assertEqual(cancelled, [])
        for expert, slot in tier._expert_to_slot.items():
            for name, tensor in reference.items():
                self.assertTrue(torch.equal(tier.tensors[name][slot], tensor[expert]))

    def test_an_inclusive_tier_smaller_than_the_promotion_fails_cleanly(self):
        # Every reserved expert is protected (the inclusive hierarchy), so a tier
        # of 4 rows can hold one chunk of 4; the other 2 tickets are cancelled.
        reserved = set()
        reference, tier = self._tier(4, lambda expert: expert in reserved)
        cache, completed, cancelled = self._promoting_cache(tier, reserved)
        tickets = self._tickets((5, 7, 0, 3, 6, 2))
        with self.assertRaisesRegex(RuntimeError, "no evictable slots"):
            self._promote(cache, tickets, tier)
        self.assertEqual(completed, [(5, 7, 0, 3)])
        self.assertEqual(cancelled, [tickets[4:]])
        self.assertIsNone(cache.promotion_in_flight)
        self.assertEqual(sorted(tier._expert_to_slot), [0, 3, 5, 7])
        for expert, slot in tier._expert_to_slot.items():
            for name, tensor in reference.items():
                self.assertTrue(torch.equal(tier.tensors[name][slot], tensor[expert]))


if __name__ == "__main__":
    unittest.main()
