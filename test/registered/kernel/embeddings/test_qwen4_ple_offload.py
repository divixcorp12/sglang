from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpModel,
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLELayer,
)
from sglang.srt.utils import set_weight_attrs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _make_source_embedding(
    *,
    dtype=torch.bfloat16,
    embedding_dim=7,
    vocab_start=0,
    vocab_end=8,
    org_vocab_size=8,
    tp_size=1,
    num_added_embeddings=0,
):
    local_rows = vocab_end - vocab_start
    weight = nn.Parameter(
        torch.empty((local_rows, embedding_dim), dtype=dtype, device="cuda"),
        requires_grad=False,
    )
    set_weight_attrs(
        weight,
        {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": lambda *_args, **_kwargs: None,
        },
    )
    shard_indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=vocab_start,
        padded_org_vocab_end_index=vocab_end,
        padded_added_vocab_start_index=org_vocab_size,
        padded_added_vocab_end_index=org_vocab_size,
        org_vocab_start_index=vocab_start,
        org_vocab_end_index=vocab_end,
        added_vocab_start_index=org_vocab_size,
        added_vocab_end_index=org_vocab_size,
    )
    return SimpleNamespace(
        weight=weight,
        quant_config=None,
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=tp_size,
        num_embeddings=org_vocab_size + num_added_embeddings,
        org_vocab_size=org_vocab_size,
        padding_size=1,
        num_added_embeddings=num_added_embeddings,
        use_presharded_weights=False,
        org_vocab_size_padded=org_vocab_size,
        num_embeddings_padded=org_vocab_size + num_added_embeddings,
        shard_indices=shard_indices,
        embedding_dim=embedding_dim,
        weight_scale=None,
        quant_method=UnquantizedEmbeddingMethod(),
        num_embeddings_per_partition=local_rows,
        num_org_embeddings_per_partition=local_rows,
        num_added_embeddings_per_partition=0,
    )


def _load_rows(offloaded, rows):
    pointer = offloaded.weight.data_ptr()
    offloaded.weight_loader(offloaded.weight, rows)
    assert offloaded.weight.data_ptr() == pointer
    assert offloaded.weight.is_pinned()
    assert offloaded.weight.weight_loader.__self__ is offloaded
    assert offloaded.quant_method is None


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("embedding_dim", [7, 64, 257])
def test_qwen4_ple_pinned_gather_tp1(input_dtype, embedding_dim):
    source = _make_source_embedding(embedding_dim=embedding_dim)
    offloaded = Qwen4ExpPinnedHostEmbedding(source)
    rows = torch.arange(8 * embedding_dim, dtype=torch.bfloat16, device="cuda").reshape(
        8, embedding_dim
    )
    _load_rows(offloaded, rows)

    ids = torch.tensor([[0, 7, 3], [4, 1, 6]], dtype=input_dtype, device="cuda")
    expected = rows.index_select(0, ids.long().flatten()).reshape(
        *ids.shape, embedding_dim
    )
    actual = offloaded(ids)

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_ple_pinned_gather_shard_boundaries_and_out_buffer():
    embedding_dim = 13
    source = _make_source_embedding(
        embedding_dim=embedding_dim,
        vocab_start=4,
        vocab_end=8,
        org_vocab_size=8,
        tp_size=2,
    )
    offloaded = Qwen4ExpPinnedHostEmbedding(source)
    rows = torch.arange(8 * embedding_dim, dtype=torch.bfloat16, device="cuda").reshape(
        8, embedding_dim
    )
    _load_rows(offloaded, rows)

    ids = torch.tensor([[-1, 3, 4], [7, 8, 100]], device="cuda")
    output = torch.full(
        (*ids.shape, embedding_dim),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual = offloaded.gather(ids, out=output)
    expected = torch.zeros_like(output)
    expected[0, 2] = rows[4]
    expected[1, 0] = rows[7]

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_ple_pinned_gather_empty_input():
    offloaded = Qwen4ExpPinnedHostEmbedding(_make_source_embedding())
    _load_rows(offloaded, torch.zeros((8, 7), dtype=torch.bfloat16, device="cuda"))
    ids = torch.empty((0, 3), dtype=torch.int64, device="cuda")
    actual = offloaded.gather(ids)
    assert actual.shape == (0, 3, 7)
    assert actual.numel() == 0


def test_qwen4_ple_pinned_embedding_rejects_unsupported_weights():
    with pytest.raises(TypeError, match="requires bfloat16"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(num_added_embeddings=1))



def test_qwen4_pinned_embedding_binds_an_offered_host_table():
    table = torch.empty((8, 7), dtype=torch.bfloat16, pin_memory=True)
    offloaded = Qwen4ExpPinnedHostEmbedding(_make_source_embedding(), host_table=table)
    rows = torch.arange(8 * 7, dtype=torch.bfloat16).reshape(8, 7)
    table.copy_(rows)

    ids = torch.tensor([5, 0, 7], device="cuda")
    assert offloaded.weight.data_ptr() == table.data_ptr()
    torch.testing.assert_close(
        offloaded(ids).cpu(), rows.index_select(0, ids.cpu()), rtol=0, atol=0
    )


def test_qwen4_pinned_embedding_rejects_an_unusable_host_table():
    for table in (
        torch.empty((8, 7), dtype=torch.bfloat16),
        torch.empty((9, 7), dtype=torch.bfloat16, pin_memory=True),
        torch.empty((8, 7), dtype=torch.float32, pin_memory=True),
    ):
        with pytest.raises(ValueError, match="bound host table"):
            Qwen4ExpPinnedHostEmbedding(_make_source_embedding(), host_table=table)


def _build_qwen4_embed_tokens(*, host, offer=None):
    from contextlib import ExitStack
    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.models import qwen3_5
    from sglang.srt.speculative.draft_shared_weights import (
        draft_shares_target_embed_and_head,
    )

    parallel = SimpleNamespace(tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1)
    pp_group = SimpleNamespace(
        is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
    )
    config = SimpleNamespace(
        vocab_size=64,
        hidden_size=16,
        rms_norm_eps=1e-6,
        hc_count=2,
        hc_lowrank=2,
        ple_layer_ids=[],
        eos_token_id=0,
        num_hidden_layers=1,
        full_attention_interval=1,
        tie_word_embeddings=False,
        model_type="qwen3_5_text",
    )
    with ExitStack() as stack:
        for owner, name, value in (
            (qwen3_5, "get_pp_group", lambda: pp_group),
            (qwen3_5, "make_layers", lambda *a, **k: (nn.ModuleList(), 0, 0)),
            (qwen3_5, "get_stream", lambda name: None),
            (qwen3_5, "is_dp_attention_enabled", lambda: False),
            (qwen4_exp_module, "is_dp_attention_enabled", lambda: False),
            (qwen4_exp_module, "GatedResidual", lambda *a, **k: nn.Module()),
        ):
            stack.enter_context(patch.object(owner, name, value))
        stack.enter_context(
            patch(
                "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
                lambda: parallel,
            )
        )
        stack.enter_context(envs.SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING.override(host))
        stack.enter_context(torch.device("cuda"))
        stack.enter_context(torch.inference_mode(False))
        # The model loader builds under the checkpoint dtype.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        stack.callback(torch.set_default_dtype, default_dtype)
        if offer is not None:
            stack.enter_context(draft_shares_target_embed_and_head(offer, None))
        model = Qwen4ExpModel(config, None, "mtp" if offer is not None else "", offer is not None)
    return model.embed_tokens


def test_qwen4_token_embedding_stays_on_the_gpu_by_default():
    embed = _build_qwen4_embed_tokens(host=False)
    assert not isinstance(embed, Qwen4ExpPinnedHostEmbedding)
    assert embed.weight.is_cuda


def test_qwen4_host_token_embedding_serves_target_and_draft_from_one_table():
    target = _build_qwen4_embed_tokens(host=True)
    assert isinstance(target, Qwen4ExpPinnedHostEmbedding)
    assert target.weight.device.type == "cpu" and target.weight.is_pinned()
    rows = torch.randn(64, 16, dtype=torch.bfloat16)
    target.weight_loader(target.weight, rows)

    draft = _build_qwen4_embed_tokens(host=True, offer=target.weight)
    assert isinstance(draft, Qwen4ExpPinnedHostEmbedding)
    assert draft.weight.data_ptr() == target.weight.data_ptr()
    assert draft.shares_target_weight
    assert [name for name, _ in draft.named_parameters()] == ["weight"]

    ids = torch.tensor([3, 63, 0, 17], device="cuda")
    expected = rows.index_select(0, ids.cpu())
    for embed in (target, draft):
        torch.testing.assert_close(embed(ids).cpu(), expected, rtol=0, atol=0)

def test_qwen4_ple_prefetch_buffer_lifecycle(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer.ple_embedding = SimpleNamespace(
        ngram_embedding=Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=layer.ple_embed_dim)
        )
    )
    layer._graph_prefetch_buffers = {}
    layer._eager_prefetch_buffer = None
    lookup_ids = torch.empty((0,), dtype=torch.int64, device="cuda")

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: False)
    eager_large = layer._get_prefetch_buffer(8, lookup_ids)
    eager_small = layer._get_prefetch_buffer(3, lookup_ids)
    assert eager_small.data_ptr() == eager_large.data_ptr()
    assert layer._eager_prefetch_buffer.shape == (8, layer.ple_embed_dim)

    eager_grown = layer._get_prefetch_buffer(12, lookup_ids)
    eager_grown_small = layer._get_prefetch_buffer(4, lookup_ids)
    assert eager_grown_small.data_ptr() == eager_grown.data_ptr()
    assert layer._eager_prefetch_buffer.shape == (12, layer.ple_embed_dim)

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: True)
    graph_three = layer._get_prefetch_buffer(3, lookup_ids)
    graph_five = layer._get_prefetch_buffer(5, lookup_ids)
    graph_three_reused = layer._get_prefetch_buffer(3, lookup_ids)
    assert graph_three_reused.data_ptr() == graph_three.data_ptr()
    assert graph_five.data_ptr() != graph_three.data_ptr()
    assert set(layer._graph_prefetch_buffers) == {3, 5}


def test_qwen4_ple_staged_prefetch_stays_on_primary_stream_during_bcg(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer._prefetch_state = None
    layer._prefetch_stream = object()

    lookup_ids = torch.tensor([[1], [3]], dtype=torch.int64, device="cuda")
    prefetched = torch.empty(
        (2, layer.ple_embed_dim), dtype=torch.bfloat16, device="cuda"
    )
    gathered_on_stream = []
    primary_stream = torch.cuda.current_stream().cuda_stream

    def gather(input_ids, out):
        gathered_on_stream.append(torch.cuda.current_stream().cuda_stream)
        out.fill_(4)
        return out

    layer.ple_embedding = SimpleNamespace(
        gather_dp_tokens=False,
        ngram_heads=1,
        compute_ngram_ids=lambda _: lookup_ids,
        _prepare_embedding_lookup=lambda *_: (lookup_ids, "semantic_tokens"),
        ngram_embedding=SimpleNamespace(
            _file_row_stager=object(),
            gather=gather,
        ),
    )
    monkeypatch.setattr(layer, "_get_prefetch_buffer", lambda *_: prefetched)
    monkeypatch.setattr(
        qwen4_exp_module, "is_in_breakable_cuda_graph", lambda: True, raising=False
    )

    layer.start_prefetch(
        SimpleNamespace(physical_tokens=2),
        SimpleNamespace(input_ids=lookup_ids.reshape(-1)),
    )

    assert gathered_on_stream == [primary_stream]
    assert layer._prefetch_state == (prefetched, "semantic_tokens", 2)
    torch.testing.assert_close(prefetched, torch.full_like(prefetched, 4))


def test_qwen4_ple_non_file_prefetch_keeps_side_stream_in_bcg(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer._prefetch_state = None
    layer._prefetch_stream = torch.cuda.Stream()

    lookup_ids = torch.tensor([[1], [3]], dtype=torch.int64, device="cuda")
    prefetched = torch.empty(
        (2, layer.ple_embed_dim), dtype=torch.bfloat16, device="cuda"
    )

    def gather(input_ids, out):
        out.fill_(4)
        return out

    layer.ple_embedding = SimpleNamespace(
        gather_dp_tokens=False,
        ngram_heads=1,
        compute_ngram_ids=lambda _: lookup_ids,
        _prepare_embedding_lookup=lambda *_: (lookup_ids, "semantic_tokens"),
        ngram_embedding=SimpleNamespace(gather=gather),
    )
    monkeypatch.setattr(layer, "_get_prefetch_buffer", lambda *_: prefetched)
    monkeypatch.setattr(qwen4_exp_module, "is_in_breakable_cuda_graph", lambda: True)

    layer.start_prefetch(
        SimpleNamespace(physical_tokens=2),
        SimpleNamespace(input_ids=lookup_ids.reshape(-1)),
    )
    torch.cuda.current_stream().wait_stream(layer._prefetch_stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(prefetched, torch.full_like(prefetched, 4))


def test_qwen4_ple_staged_prefetch_replays_with_fresh_ids():
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer._prefetch_state = None
    layer._prefetch_stream = object()
    layer._graph_prefetch_buffers = {}
    layer._eager_prefetch_buffer = None

    source_ids = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    prefetched = torch.empty(
        (2, layer.ple_embed_dim), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty_like(prefetched)

    def gather(input_ids, out):
        out.copy_(input_ids.to(out.dtype).unsqueeze(-1).expand_as(out))
        return out

    layer.ple_embedding = SimpleNamespace(
        gather_dp_tokens=False,
        ngram_heads=1,
        compute_ngram_ids=lambda _: source_ids.reshape(-1, 1) + 0,
        _prepare_embedding_lookup=lambda ids, *_: (ids, "semantic_tokens"),
        ngram_embedding=SimpleNamespace(
            _file_row_stager=object(),
            gather=gather,
        ),
    )
    layer._get_prefetch_buffer = lambda *_: prefetched

    graph = BreakableCUDAGraph()
    capture_stream = torch.cuda.Stream()
    with (
        enable_breakable_cuda_graph(),
        BreakableCUDAGraphCapture(graph, stream=capture_stream),
    ):
        layer.start_prefetch(
            SimpleNamespace(physical_tokens=2),
            SimpleNamespace(input_ids=source_ids),
        )
        output.copy_(prefetched)

    source_ids.copy_(torch.tensor([5, 7], dtype=torch.int64, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    assert len(graph._break_fns) == 1
    expected = torch.tensor([[5] * 7, [7] * 7], dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(output, expected)


def _staged_before_replay_layer(source_ids, gathered):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer._prefetch_state = None
    layer._prefetch_stream = object()
    layer._graph_prefetch_buffers = {}
    layer._eager_prefetch_buffer = None
    layer.stage_before_replay = True

    def gather(input_ids, out):
        gathered.append(input_ids.clone())
        out.copy_(input_ids.to(out.dtype).unsqueeze(-1).expand_as(out))
        return out

    layer.ple_embedding = SimpleNamespace(
        gather_dp_tokens=False,
        ngram_heads=1,
        compute_ngram_ids=lambda _: source_ids.reshape(-1, 1) + 0,
        _prepare_embedding_lookup=lambda ids, *_: (ids, "semantic_tokens"),
        ngram_embedding=SimpleNamespace(
            _file_row_stager=object(),
            gather=gather,
            allocate_output=lambda shape, device: torch.empty(
                shape, dtype=torch.bfloat16, device=device
            ),
        ),
    )
    return layer


def test_qwen4_ple_rows_staged_before_replay_leave_no_graph_break(monkeypatch):
    source_ids = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    gathered = []
    layer = _staged_before_replay_layer(source_ids, gathered)
    output = torch.empty((2, layer.ple_embed_dim), dtype=torch.bfloat16, device="cuda")
    batch = SimpleNamespace(physical_tokens=2)
    forward_batch = SimpleNamespace(input_ids=source_ids)

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: True)
    graph = BreakableCUDAGraph()
    with (
        enable_breakable_cuda_graph(),
        BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()),
    ):
        layer.start_prefetch(batch, forward_batch)
        output.copy_(layer._prefetch_state[0])
        layer._prefetch_state = None
    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: False)

    assert gathered == []
    assert len(graph._break_fns) == 0

    for routes in ([5, 7], [2, 2]):
        source_ids.copy_(torch.tensor(routes, dtype=torch.int64, device="cuda"))
        layer.stage_rows_for_replay(batch, forward_batch, graph_tokens=2)
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.tensor(
            [[routes[0]] * 7, [routes[1]] * 7], dtype=torch.bfloat16, device="cuda"
        )
        torch.testing.assert_close(output, expected)
    assert layer._prefetch_state is None

    padded = torch.full((3, layer.ple_embed_dim), 9, dtype=torch.bfloat16, device="cuda")
    layer._graph_prefetch_buffers[3] = padded
    layer.stage_rows_for_replay(batch, forward_batch, graph_tokens=3)
    torch.cuda.synchronize()
    torch.testing.assert_close(padded[:2], expected)
    torch.testing.assert_close(padded[2], torch.zeros_like(padded[2]))

    with pytest.raises(RuntimeError, match="no captured PLE buffer"):
        layer.stage_rows_for_replay(batch, forward_batch, graph_tokens=4)


def test_qwen4_model_stages_only_prefetched_ple_layers_before_replay(monkeypatch):
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    staged = []

    def ple(layer_id, stages):
        return SimpleNamespace(
            stage_before_replay=stages,
            stage_rows_for_replay=lambda batch, forward_batch, graph_tokens: (
                staged.append((layer_id, batch, forward_batch, graph_tokens))
            ),
        )

    model.layers = [
        SimpleNamespace(ple=ple(0, True)),
        SimpleNamespace(ple=None),
        SimpleNamespace(ple=ple(2, True)),
        SimpleNamespace(ple=ple(3, False)),
    ]
    model._start_layer, model._end_layer = 0, 4
    model.has_ple = True
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 9
    forward_batch = SimpleNamespace(input_ids="ids")
    prepared = []

    def prepare(input_ids, batch, *, ngram_size, ngram_eos_token_id):
        prepared.append((input_ids, batch, ngram_size, ngram_eos_token_id))
        return "ple_batch"

    monkeypatch.setattr(qwen4_exp_module, "_prepare_ple_batch", prepare)

    model.prepare_decode_graph_replay(forward_batch, graph_tokens=4)

    assert prepared == [("ids", forward_batch, 3, 9)]
    assert staged == [(2, "ple_batch", forward_batch, 4)]


def test_qwen4_target_verify_replay_stages_under_the_captured_token_key(
    monkeypatch, tmp_path
):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    heads, draft_tokens, requests = 2, 4, 1
    layer, table = _uring_file_layer(
        monkeypatch,
        tmp_path,
        "verify",
        torch.bfloat16,
        80,
        (0, 5000),
        heads=heads,
        graph=requests * draft_tokens,
    )
    ids = torch.arange(requests * draft_tokens * heads, device="cuda") * 7 + 3
    layer.lookup = ids.reshape(requests * draft_tokens, heads)
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    model.layers = [SimpleNamespace(ple=None), SimpleNamespace(ple=layer)]
    model._start_layer, model._end_layer = 0, 2
    model.has_ple = True
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 9
    monkeypatch.setattr(
        qwen4_exp_module,
        "_prepare_ple_batch",
        lambda *_a, **_k: SimpleNamespace(physical_tokens=requests * draft_tokens),
    )
    runner = SimpleNamespace(
        capture_forward_mode=ForwardMode.TARGET_VERIFY,
        captured_req_width=draft_tokens,
        ragged_verify_mode=False,
    )

    graph_tokens = DecodeCudaGraphRunner.capture_output_rows(runner, requests)
    assert graph_tokens == requests * draft_tokens
    with pytest.raises(RuntimeError, match="no captured PLE buffer"):
        model.prepare_decode_graph_replay(SimpleNamespace(input_ids="ids"), requests)

    layer._graph_prefetch_buffers[graph_tokens].fill_(9)
    model.prepare_decode_graph_replay(SimpleNamespace(input_ids="ids"), graph_tokens)
    torch.cuda.synchronize()

    byte_rows = table.view(torch.uint8).view(table.shape[0], -1)
    expected = (
        byte_rows[ids.cpu()].to("cuda").view(table.dtype).reshape(graph_tokens, -1)
    )
    staged = layer._graph_prefetch_buffers[graph_tokens]
    assert torch.equal(staged.view(torch.uint8), expected.view(torch.uint8))


def _uring_file_layer(monkeypatch, tmp_path, name, dtype, dim, vocab, heads, graph):
    from sglang.srt.models.qwen4_exp_ple_table import PleFileRowStager

    monkeypatch.setattr(
        qwen4_exp_module, "check_file_backend_supported", lambda *_: False
    )
    start, end = vocab
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(
            dtype=dtype,
            embedding_dim=dim,
            vocab_start=start,
            vocab_end=end,
            org_vocab_size=end,
            tp_size=2 if start else 1,
        ),
        backend="file",
        table_dir=str(tmp_path / name),
    )
    table = offloaded._ple_file_cache_table
    generator = torch.Generator().manual_seed(len(name) * 31 + dim)
    table.view(torch.uint8).view(-1).copy_(
        torch.randint(0, 256, (table.numel() * table.element_size(),), generator=generator, dtype=torch.uint8)
    )
    offloaded._file_row_stager = PleFileRowStager(table, reader_mode="uring")

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = heads * dim
    layer.stage_before_replay = True
    layer._graph_prefetch_buffers = {
        graph: torch.full((graph, heads * dim), 9, dtype=torch.bfloat16, device="cuda")
    }
    ids = torch.zeros((0, heads), dtype=torch.int64, device="cuda")
    layer.lookup = ids
    layer.ple_embedding = SimpleNamespace(
        gather_dp_tokens=False,
        ngram_heads=heads,
        compute_ngram_ids=lambda _: layer.lookup,
        _prepare_embedding_lookup=lambda ids, *_: (ids, ids.shape[0]),
        ngram_embedding=offloaded,
    )
    return layer, table


def test_qwen4_model_stages_every_ple_layer_with_one_native_read(monkeypatch, tmp_path):
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader

    heads, graph = 2, 5
    specs = [
        ("bf16", torch.bfloat16, 80, (0, 5000)),
        ("fp8", torch.float8_e4m3fn, 160, (0, 5000)),
        ("shard", torch.bfloat16, 80, (4, 3004)),
    ]
    layers = [
        _uring_file_layer(monkeypatch, tmp_path, *spec, heads=heads, graph=graph)
        for spec in specs
    ]
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    model.layers = [SimpleNamespace(ple=None)] + [
        SimpleNamespace(ple=layer) for layer, _ in layers
    ]
    model._start_layer, model._end_layer = 0, len(model.layers)
    model.has_ple = True
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 9
    forward_batch = SimpleNamespace(input_ids="ids")

    calls = {"read": 0, "read_paged_rows": 0}
    for method in calls:
        original = getattr(UringFileReader, method)

        def counted(self, *args, _original=original, _method=method):
            calls[_method] += 1
            return _original(self, *args)

        monkeypatch.setattr(UringFileReader, method, counted)

    def expected_rows(table, ids, vocab):
        start, end = vocab
        flat = ids.reshape(-1).cpu()
        inside = (flat >= start) & (flat < end)
        byte_rows = table.view(torch.uint8).view(table.shape[0], -1)
        rows = byte_rows[torch.where(inside, flat - start, 0)].clone()
        rows[~inside] = 0
        return (
            rows.to("cuda")
            .view(table.dtype)
            .to(torch.bfloat16)
            .reshape(ids.shape[0], -1)
        )

    def captured(layer):
        return layer._graph_prefetch_buffers[graph].clone()

    generator = torch.Generator().manual_seed(11)
    for tokens in (3, 3, 1, 3):
        monkeypatch.setattr(
            qwen4_exp_module,
            "_prepare_ple_batch",
            lambda *_a, _tokens=tokens, **_k: SimpleNamespace(physical_tokens=_tokens),
        )
        for (layer, _), (_, _, _, (start, end)) in zip(layers, specs):
            local = torch.randint(0, end - start, (tokens * heads,), generator=generator)
            local[0] = 25
            local[-1] = end - start - 1
            ids = local + start
            ids[1] = -1
            if tokens * heads > 2:
                ids[2] = start + 25
                ids[3] = start + 26
                ids[4] = end
            layer.lookup = ids.reshape(tokens, heads).to("cuda")
            layer._graph_prefetch_buffers[graph].fill_(9)

        calls.update(read=0, read_paged_rows=0)
        model.prepare_decode_graph_replay(forward_batch, graph_tokens=graph)
        torch.cuda.synchronize()
        assert calls == {"read": 0, "read_paged_rows": 1}
        batched = [captured(layer) for layer, _ in layers]

        for (layer, table), (_, _, _, vocab), rows in zip(layers, specs, batched):
            expected = torch.zeros_like(rows)
            expected[:tokens] = expected_rows(table, layer.lookup, vocab)
            assert torch.equal(rows.view(torch.uint8), expected.view(torch.uint8))

        calls.update(read=0, read_paged_rows=0)
        for layer, _ in layers:
            layer._graph_prefetch_buffers[graph].fill_(9)
            layer.stage_rows_for_replay(
                SimpleNamespace(physical_tokens=tokens), forward_batch, graph
            )
        torch.cuda.synchronize()
        assert calls == {"read": len(layers), "read_paged_rows": 0}
        for (layer, _), rows in zip(layers, batched):
            assert torch.equal(captured(layer).view(torch.uint8), rows.view(torch.uint8))


def test_qwen4_replay_staging_waits_for_ids_pending_on_the_forward_stream(
    monkeypatch, tmp_path
):
    """Under the overlap scheduler a replay's decode ids are still being written by the
    previous forward when staging starts; the staged rows must be the final ids' rows."""
    heads, graph = 2, 3
    vocab = (0, 5000)
    layer, table = _uring_file_layer(
        monkeypatch, tmp_path, "bf16", torch.bfloat16, 80, vocab, heads=heads, graph=graph
    )
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    model.layers = [SimpleNamespace(ple=None), SimpleNamespace(ple=layer)]
    model._start_layer, model._end_layer = 0, len(model.layers)
    model.has_ple = True
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 9
    monkeypatch.setattr(
        qwen4_exp_module,
        "_prepare_ple_batch",
        lambda *_a, **_k: SimpleNamespace(physical_tokens=1),
    )
    forward_batch = SimpleNamespace(input_ids="ids")
    forward_stream = torch.cuda.Stream()
    final = torch.tensor([[4001, 4002]], dtype=torch.int64, device="cuda")
    with torch.cuda.stream(forward_stream):
        layer.lookup = torch.tensor([[11, 12]], dtype=torch.int64, device="cuda")
        model.prepare_decode_graph_replay(forward_batch, graph_tokens=graph)
        pending = torch.zeros_like(final)
        # The previous forward is still running when the next batch is staged.
        torch.cuda._sleep(200_000_000)
        pending.copy_(final)
        layer.lookup = pending
        model.prepare_decode_graph_replay(forward_batch, graph_tokens=graph)
    torch.cuda.synchronize()

    byte_rows = table.view(torch.uint8).view(table.shape[0], -1)
    expected = torch.zeros_like(layer._graph_prefetch_buffers[graph])
    expected[:1] = (
        byte_rows[final.reshape(-1).cpu()].to("cuda").view(table.dtype).reshape(1, -1)
    )
    staged = layer._graph_prefetch_buffers[graph]
    assert torch.equal(staged.view(torch.uint8), expected.view(torch.uint8))

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
