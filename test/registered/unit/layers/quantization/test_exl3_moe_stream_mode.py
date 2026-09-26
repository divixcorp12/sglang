"""Exl3MoEMethod in streaming mode: no expert parameters, an attached expert
streamer, and an apply that runs gathered chunks through the eager loop (CPU)."""

import contextlib
import functools
import json
import types

import pytest
import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import (
    EXL3_STREAMED_NAMES,
    exl3_expert_layout_for,
)
from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace
from sglang.srt.layers.moe.expert_row_source import RowReadStats
from sglang.srt.layers.quantization import exl3 as exl3_mod
from sglang.srt.layers.quantization import exl3_ops
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod, Exl3RowViews
from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import HIDDEN, INTER, write_fake_exl3

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
NUM_EXPERTS = 6


def _fake_linear(x, t, out_dtype=None):
    """A CPU stand-in for exl3_linear that depends on all three tensors of its expert."""
    scale = 1.0 + t.trellis.float().mean() / 32768 + t.suh.float().mean()
    y = x.float().sum(-1, keepdim=True) * t.svh.float() * scale
    return y.to(out_dtype or x.dtype)


def _reference(ckpt, layer):
    """({name: [experts, parts, ...]}, resident w13, resident w2) read with safetensors."""
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    def get(expert, w, kind):
        name = f"layers.{layer}.ffn.experts.{expert}.{w}.{kind}"
        with safe_open(str(ckpt / weight_map[name]), "pt") as f:
            return f.get_tensor(name)

    rows = {}
    for prefix, linears in (("w13", ("w1", "w3")), ("w2", ("w2",))):
        for kind in ("trellis", "suh", "svh"):
            rows[f"{prefix}_{kind}"] = torch.stack(
                [torch.stack([get(e, w, kind) for w in linears]) for e in range(NUM_EXPERTS)]
            )

    def tensors(e, w):
        return Exl3Tensors(trellis=get(e, w, "trellis"), suh=get(e, w, "suh"), svh=get(e, w, "svh"), mul1=True)

    w13 = [(tensors(e, "w1"), tensors(e, "w3")) for e in range(NUM_EXPERTS)]
    w2 = [tensors(e, "w2") for e in range(NUM_EXPERTS)]
    return rows, w13, w2


class FakeStreamer:
    """The streamer interface Exl3MoEMethod uses, over in-memory reference rows.

    Each chunk's rows sit in reverse order behind three padding rows, so the
    method must follow ``row_of_source``.
    """

    def __init__(self, reference, chunk_rows):
        self.reference = reference
        self.chunk_rows = chunk_rows
        self.recorded = []
        self.chunks = []
        self.last_gather_stats = None
        self.background_read_stats = RowReadStats()
        self.used_host_iterator = False

    def serves_graph_gather(self, topk_output):
        return False  # no graph gather: apply takes the eager streamed path

    def record_routes(self, topk_ids):
        self.recorded.append(topk_ids.clone())

    def prefill_fills(self, source_ids):
        return contextlib.nullcontext()  # no pinned tier to fill

    def iter_gather_experts(self, source_ids, chunk_rows=None):
        ids = source_ids.tolist()
        for start in range(0, len(ids), self.chunk_rows):
            chunk = ids[start : start + self.chunk_rows]
            n = len(chunk)
            row_of_source = [3 + n - 1 - i for i in range(n)]
            rows = {
                name: torch.zeros((3 + n,) + t.shape[1:], dtype=t.dtype)
                for name, t in self.reference.items()
            }
            for expert, row in zip(chunk, row_of_source):
                for name, t in self.reference.items():
                    rows[name][row] = t[expert]
            self.chunks.append(chunk)
            yield torch.tensor(chunk), torch.tensor(row_of_source), rows
        self.last_gather_stats = types.SimpleNamespace(
            miss_rows=len(ids), host_read_rows=len(ids), host_read_ns=0, host_split_ns=0
        )

    def iter_gather_experts_host(self, source_ids, experts, chunk_rows=None):
        assert experts == source_ids.tolist()
        self.used_host_iterator = True
        for chunk, row_of_source, rows in self.iter_gather_experts(source_ids, chunk_rows):
            yield chunk.tolist(), row_of_source.tolist(), rows


@pytest.fixture
def ckpt(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=NUM_EXPERTS, finite=True)
    exl3_expert_layout_for.cache_clear()
    return tmp_path


def _layer(method, num_experts=NUM_EXPERTS, hidden=HIDDEN, inter=INTER):
    layer = torch.nn.Module()
    layer.layer_id = 1
    method.create_weights(layer, num_experts, hidden, inter, torch.bfloat16)
    return layer


def _streaming_env(ckpt):
    return (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_EXPERT_DIR.override(str(ckpt)),
        envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("auto"),
        envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"),
    )


def test_stream_mode_registers_no_expert_parameters():
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True):
        layer = _layer(Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True))
    assert list(layer.named_parameters()) == []
    assert layer.exl3_streamed is True


def test_process_attaches_an_exl3_streamer(ckpt):
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
    from sglang.srt.layers.moe.expert_format import expert_streamer_of

    a, b, c, d = _streaming_env(ckpt)
    with a, b, c, d:
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
        method.process_weights_after_loading(layer)
    streamer = expert_streamer_of(layer)
    assert streamer.format.key == "exl3"
    assert streamer.layer_id == 1 and streamer.num_experts == NUM_EXPERTS
    assert isinstance(streamer.row_source, Exl3ShardRowSource)
    assert streamer.row_source.layer_id == 1


@pytest.mark.parametrize(
    "kwargs, env_dir, match",
    [
        ({"num_experts": NUM_EXPERTS + 1}, True, "disable_shared_experts_fusion"),
        ({"hidden": 2 * HIDDEN}, True, "do not match the layer"),
        ({}, False, "needs SGLANG_DSV41_EXPERT_DIR"),
    ],
)
def test_process_rejects_a_mismatched_layer(ckpt, kwargs, env_dir, match):
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True), envs.SGLANG_DSV41_EXPERT_DIR.override(
        str(ckpt) if env_dir else ""
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method, **kwargs)
        with pytest.raises(ValueError, match=match):
            method.process_weights_after_loading(layer)


def test_a_launch_without_a_pinned_tier_warns_once(ckpt, monkeypatch, caplog):
    from sglang.srt.layers.moe import exl3_expert_format

    monkeypatch.setattr(exl3_expert_format, "_WARNED_WITHOUT_PINNED_TIER", False)
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True), envs.SGLANG_DSV41_EXPERT_DIR.override(
        str(ckpt)
    ), envs.SGLANG_MOE_PINNED_HOST_MB.override(0):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        with caplog.at_level("WARNING", logger="sglang.srt.layers.moe.exl3_expert_format"):
            for _ in range(2):
                layer = _layer(method, hidden=2 * HIDDEN)  # fails after the warning
                with pytest.raises(ValueError, match="do not match the layer"):
                    method.process_weights_after_loading(layer)
    warnings = [r for r in caplog.records if "SGLANG_MOE_PINNED_HOST_MB" in r.getMessage()]
    assert len(warnings) == 1


def _fake_accumulates(monkeypatch):
    monkeypatch.setattr(
        exl3_mod, "exl3_moe_accumulate",
        functools.partial(exl3_ops.exl3_moe_accumulate, linear=_fake_linear),
    )
    monkeypatch.setattr(
        exl3_mod, "exl3_moe_accumulate_planned",
        functools.partial(exl3_ops.exl3_moe_accumulate_planned, linear=_fake_linear),
    )


@pytest.mark.parametrize("route_plan", [False, True])
@pytest.mark.parametrize("chunk_rows", [2, 64])
def test_streamed_apply_matches_the_resident_loop(ckpt, monkeypatch, chunk_rows, route_plan):
    reference, w13, w2 = _reference(ckpt, 1)
    _fake_accumulates(monkeypatch)
    trace = Exl3StreamTrace()
    monkeypatch.setattr(exl3_mod, "get_exl3_stream_trace", lambda: trace)
    with (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan),
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
    streamer = FakeStreamer(reference, chunk_rows)
    layer._nvfp4_expert_streamer = streamer
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, HIDDEN, generator=generator).to(torch.bfloat16)
    topk_ids = torch.tensor([[5, 0, 3], [3, 1, 5], [0, 4, 1], [5, 3, 4]], dtype=torch.int32)
    topk_weights = torch.rand(4, 3, generator=generator)
    topk = types.SimpleNamespace(topk_weights=topk_weights, topk_ids=topk_ids)
    dispatch = types.SimpleNamespace(hidden_states=x, topk_output=topk)

    got = method.apply(layer, dispatch).hidden_states
    want = exl3_ops.exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, 10.0, linear=_fake_linear)
    assert torch.equal(got, want)
    assert len(streamer.recorded) == 1 and torch.equal(streamer.recorded[0], topk_ids.reshape(-1))
    assert [e for chunk in streamer.chunks for e in chunk] == [0, 1, 3, 4, 5]
    assert all(len(chunk) <= chunk_rows for chunk in streamer.chunks)
    assert trace.stats()["vram_misses"] == 5
    assert streamer.used_host_iterator is route_plan


def test_apply_runs_graph_gathered_routes_in_graph(monkeypatch):
    """A route set the streamer's graph gather serves goes to _apply_graph, without the capture guard."""
    calls = []

    def fake_apply_graph(layer, streamer, x, topk_weights, topk_ids, swiglu_limit):
        calls.append((layer, streamer, x, topk_weights, topk_ids, swiglu_limit))
        return torch.ones_like(x)

    def refuse(name):
        raise AssertionError(f"{name} guarded against capture on the in-graph path")

    monkeypatch.setattr(Exl3MoEMethod, "_apply_graph", staticmethod(fake_apply_graph))
    monkeypatch.setattr(exl3_mod, "assert_not_capturing", refuse)
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
    streamer = types.SimpleNamespace(serves_graph_gather=lambda topk: topk.topk_ids.numel() <= 6)
    layer._nvfp4_expert_streamer = streamer
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=1.5
    )
    topk_ids = torch.tensor([[5, 0, 3, 1, 2, 4]], dtype=torch.int32)
    x, topk_weights, dispatch = _routed_inputs(topk_ids)

    got = method.apply(layer, dispatch).hidden_states
    assert len(calls) == 1
    assert all(got_arg is want for got_arg, want in zip(calls[0], (layer, streamer, x, topk_weights, topk_ids)))
    assert calls[0][5] == 10.0
    assert torch.equal(got, torch.full_like(x, 1.5))  # the routed scale still applies


def _select(x, logits, cfg):
    from sglang.srt.layers.moe.topk import select_experts

    return select_experts(hidden_states=x, router_logits=logits, topk_config=cfg, layer_id=1)


@pytest.mark.parametrize("kind", ["standard", "packed", "bypassed"])
def test_apply_routes_every_topk_format_alike(monkeypatch, kind):
    """The draft path hands over a BypassedTopKOutput (hidden_states/router_logits/topk_config, no
    topk_ids); apply must materialize it once, and pass the same routing to the streamer check and
    the kernel as it does for the equivalent standard output."""
    from sglang.srt.layers.moe.topk import (
        BypassedTopKOutput,
        StandardTopKOutputPacked,
        TopKConfig,
    )

    calls, inspected = [], []

    def fake_apply_graph(layer, streamer, x, topk_weights, topk_ids, swiglu_limit):
        calls.append((topk_weights, topk_ids))
        return torch.ones_like(x)

    monkeypatch.setattr(Exl3MoEMethod, "_apply_graph", staticmethod(fake_apply_graph))
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)

    def serves_graph_gather(topk):
        inspected.append(topk)
        return isinstance(getattr(topk, "topk_ids", None), torch.Tensor)  # a bypassed output has no ids

    layer._nvfp4_expert_streamer = types.SimpleNamespace(serves_graph_gather=serves_graph_gather)
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, HIDDEN, generator=generator).to(torch.bfloat16)
    logits = torch.randn(1, NUM_EXPERTS, generator=generator)
    cfg = TopKConfig(top_k=3, renormalize=True, torch_native=True)  # the fused router needs CUDA
    want = _select(x, logits, cfg)
    if kind == "bypassed":
        topk = BypassedTopKOutput(hidden_states=x, router_logits=logits, topk_config=cfg)
    elif kind == "packed":
        topk = StandardTopKOutputPacked(*want, want.topk_ids)
    else:
        topk = want

    method.apply(layer, types.SimpleNamespace(hidden_states=x, topk_output=topk))

    assert len(calls) == 1, "a bypassed output must reach the in-graph path, not fail before it"
    assert torch.equal(calls[0][0], want.topk_weights) and torch.equal(calls[0][1], want.topk_ids)
    assert len(inspected) == 1 and hasattr(inspected[0], "topk_ids")  # the converted value, once
    if kind != "bypassed":
        assert inspected[0] is topk  # standard formats are passed through untouched


def _routed_inputs(topk_ids, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(topk_ids.shape[0], HIDDEN, generator=generator).to(torch.bfloat16)
    topk_weights = torch.rand(topk_ids.shape, generator=generator)
    topk = types.SimpleNamespace(topk_weights=topk_weights, topk_ids=topk_ids)
    return x, topk_weights, types.SimpleNamespace(hidden_states=x, topk_output=topk)


@pytest.mark.parametrize("route_plan", [False, True])
@pytest.mark.parametrize("pinned_rows", [3, 6])  # evicting, and holding every expert
def test_a_real_streamer_spanning_chunks_matches_the_resident_loop(ckpt, monkeypatch, pinned_rows, route_plan):
    """Three chunks of at most 2 experts reuse one staging set, and a 3-row pinned
    tier evicts between them: each chunk's rows must be read through row_of_source.
    (No uncached case: that path pins host memory, which needs CUDA. The hot cache
    is CUDA-only, so the all-hit path runs only in the GPU file.)"""
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

    _, w13, w2 = _reference(ckpt, 1)
    _fake_accumulates(monkeypatch)
    a, b, c, d = _streaming_env(ckpt)
    with a, b, c, d, envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
        method.process_weights_after_loading(layer)
    streamer = layer._nvfp4_expert_streamer
    streamer.format.max_gather_rows = 2
    ExpertPinnedHostCache(streamer, pinned_rows, device="cpu", **streamer.format.pinned_tier_options(layer))
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    topk_ids = torch.tensor([[5, 0, 3], [3, 1, 5], [0, 4, 1], [5, 3, 4]], dtype=torch.int32)
    x, topk_weights, dispatch = _routed_inputs(topk_ids)
    chunks = []
    iterate = streamer.iter_gather_experts
    iterate_host = streamer.iter_gather_experts_host

    def recording(ids, **kwargs):
        for chunk, row_of_source, rows in iterate(ids, **kwargs):
            chunks.append(("device", chunk.tolist()))
            yield chunk, row_of_source, rows

    def recording_host(ids, experts, **kwargs):
        for chunk, row_of_source, rows in iterate_host(ids, experts, **kwargs):
            chunks.append(("host", chunk))
            yield chunk, row_of_source, rows

    streamer.iter_gather_experts = recording
    streamer.iter_gather_experts_host = recording_host

    got = method.apply(layer, dispatch).hidden_states
    want = exl3_ops.exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, 10.0, linear=_fake_linear)
    path = "host" if route_plan else "device"
    assert chunks == [(path, [0, 1]), (path, [3, 4]), (path, [5])]
    assert torch.equal(got, want)



def _constant_linear(x, t, out_dtype=None):
    """Ignores its input: every row is svh[0] * 2**14, so each expert adds an exact constant through w2."""
    value = float(t.svh.reshape(-1)[0]) * 2**14
    return torch.full((x.shape[0], t.svh.numel()), value, dtype=out_dtype or x.dtype)


@pytest.mark.parametrize("route_plan", [False, True])
def test_streamed_apply_accumulates_in_ascending_expert_order(ckpt, monkeypatch, route_plan):
    """Experts 0, 1, 2 add 2**24, 1 and -2**24 in fp32: ascending order gives 0, any other order gives 1 or more,
    which survives the bf16 cast. Every other parity test is blind to the order at bf16."""
    reference, w13, w2 = _reference(ckpt, 1)
    for expert, svh0 in ((0, 1024.0), (1, 2.0**-14), (2, -1024.0)):
        reference["w2_svh"][expert, 0, 0] = svh0
        w2[expert].svh[0] = svh0
    monkeypatch.setattr(
        exl3_mod, "exl3_moe_accumulate",
        functools.partial(exl3_ops.exl3_moe_accumulate, linear=_constant_linear),
    )
    monkeypatch.setattr(
        exl3_mod, "exl3_moe_accumulate_planned",
        functools.partial(exl3_ops.exl3_moe_accumulate_planned, linear=_constant_linear),
    )
    trace = Exl3StreamTrace()
    monkeypatch.setattr(exl3_mod, "get_exl3_stream_trace", lambda: trace)
    with (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan),
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
    streamer = FakeStreamer(reference, 64)
    layer._nvfp4_expert_streamer = streamer
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    topk_ids = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    x, topk_weights, dispatch = _routed_inputs(topk_ids)
    got = method.apply(layer, dispatch).hidden_states
    want = exl3_ops.exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, 10.0, linear=_constant_linear)
    assert float(got[0, 0]) == 0.0
    assert torch.equal(got, want)

def test_route_plan_all_dropped_routes_give_zeros(ckpt, monkeypatch):
    """Every route -1: no chunk is gathered and the output is zeros, as with the flag off."""
    reference, _, _ = _reference(ckpt, 1)
    _fake_accumulates(monkeypatch)
    trace = Exl3StreamTrace()
    monkeypatch.setattr(exl3_mod, "get_exl3_stream_trace", lambda: trace)
    with (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(True),
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
    streamer = FakeStreamer(reference, 2)
    layer._nvfp4_expert_streamer = streamer
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    topk_ids = torch.full((3, 3), -1, dtype=torch.int32)
    x, _, dispatch = _routed_inputs(topk_ids)
    got = method.apply(layer, dispatch).hidden_states
    assert torch.equal(got, torch.zeros_like(x)) and streamer.chunks == []


def test_row_views_are_cached_per_buffer_and_row():
    shapes = {
        "w13_trellis": ((2, 1, 1, 48), torch.int16),
        "w13_suh": ((2, 16), torch.float16),
        "w13_svh": ((2, 16), torch.float16),
        "w2_trellis": ((1, 1, 1, 48), torch.int16),
        "w2_suh": ((1, 16), torch.float16),
        "w2_svh": ((1, 16), torch.float16),
    }
    rows = {name: torch.zeros((3,) + shape, dtype=dtype) for name, (shape, dtype) in shapes.items()}
    views = Exl3RowViews(max_buffers=1)
    w13, w2 = views.select(rows, [7, 2], [2, 0])
    again, _ = views.select(rows, [2], [0])
    assert again[2][0] is w13[2][0] and again[2][1] is w13[2][1]
    assert w13[7][1].trellis.data_ptr() == rows["w13_trellis"][2, 1].data_ptr()
    assert w2[2].svh.data_ptr() == rows["w2_svh"][0, 0].data_ptr()
    other = {name: t.clone() for name, t in rows.items()}
    fresh, _ = views.select(other, [2], [0])
    assert fresh[2][0] is not w13[2][0]
    assert tuple(EXL3_STREAMED_NAMES) == tuple(shapes)


@pytest.mark.parametrize("route_plan", [False, True])
def test_streamed_apply_skips_route_recording_while_capturing(monkeypatch, route_plan):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.srt.model_executor.runner_utils import capture_mode

    recorded = []

    class _Streamer:
        background_read_stats = type("S", (), {"rows": 0})()
        last_gather_stats = None

        def record_routes(self, routed):
            recorded.append(routed.tolist())

        def iter_gather_experts(self, source_ids):
            return iter(())

        def iter_gather_experts_host(self, source_ids, experts):
            return iter(())

        def prefill_fills(self, source_ids):
            return contextlib.nullcontext()

    layer = type("L", (), {"layer_id": 0})()
    x = torch.zeros((1, 8), dtype=torch.float16)
    weights = torch.ones((1, 2), dtype=torch.float32)
    ids = torch.tensor([[1, 2]])
    monkeypatch.setattr(capture_mode, "is_capture_mode", True)
    Exl3MoEMethod._apply_streamed(layer, _Streamer(), x, weights, ids, None, route_plan=route_plan)
    assert recorded == []
    monkeypatch.setattr(capture_mode, "is_capture_mode", False)
    Exl3MoEMethod._apply_streamed(layer, _Streamer(), x, weights, ids, None, route_plan=route_plan)
    assert recorded == [[1, 2]]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
