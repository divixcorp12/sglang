"""Candidate indexer: level-one block selection, score masking, and factory gating."""

import sys
import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as backend_mod
from sglang.srt.layers.attention.dsv4 import candidate_indexer as ci
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INF = float("inf")


def test_keeps_top_blocks_and_forces_the_newest():
    logits = torch.tensor([[1.0, 5.0, -INF, -INF, 3.0, 0.0, 2.0, 2.0]])
    keep = ci.select_candidate_blocks(logits, compress_lens=8, topk_blocks=2, block_size=2)
    assert keep.tolist() == [[True, True, False, False, False, False, True, True]]


def test_underfilled_topk_drops_unreachable_blocks():
    logits = torch.tensor([[-INF, -INF, -INF, -INF, 1.0, 2.0]])
    keep = ci.select_candidate_blocks(logits, compress_lens=2, topk_blocks=3, block_size=2)
    assert keep.tolist() == [[True, True, False, False, True, True]]


def test_ragged_width_is_padded_then_trimmed():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    keep = ci.select_candidate_blocks(logits, compress_lens=3, topk_blocks=1, block_size=2)
    assert keep.tolist() == [[False, False, True]]


def test_mask_topk_scores_invalidates_masked_and_out_of_range():
    scores = torch.tensor([[0.0, -INF, 5.0]])
    indices = torch.tensor([[2, 1, 7, 0]])
    assert ci.mask_topk_scores(scores, indices).tolist() == [[2, -1, -1, 0]]


def test_mask_topk_scores_with_offsets():
    scores = torch.tensor([[0.0, 1.0, 5.0]])
    indices = torch.tensor([[3, 1]])
    assert ci.mask_topk_scores(scores, indices, torch.tensor([1])).tolist() == [[3, 1]]


@pytest.fixture
def sm(monkeypatch):
    def set_sm(value):
        monkeypatch.setattr(ci, "get_platform", lambda: types.SimpleNamespace(device_sm=value))

    return set_sm


def test_factory_is_off_without_a_source_layer_or_blocks(sm):
    sm(120)
    assert ci.make_candidate_indexer(topk_blocks=0, block_size=128, source_layer_id=20) is None
    assert ci.make_candidate_indexer(topk_blocks=8, block_size=128, source_layer_id=-1) is None


def test_factory_is_off_on_hopper(sm):
    sm(90)
    assert ci.make_candidate_indexer(topk_blocks=8, block_size=128, source_layer_id=20) is None


def test_factory_needs_paged_sparse_logits_on_sm120(sm, monkeypatch):
    sm(120)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.deep_gemm_wrapper.configurer",
        types.SimpleNamespace(DEEPGEMM_PAGED_SPARSE_MQA_LOGITS=False),
    )
    with pytest.raises(RuntimeError, match="paged sparse MQA logits"):
        ci.make_candidate_indexer(topk_blocks=8, block_size=128, source_layer_id=20)


def _paged_sparse_logits_configurer(monkeypatch, available):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.deep_gemm_wrapper.configurer",
        types.SimpleNamespace(DEEPGEMM_PAGED_SPARSE_MQA_LOGITS=available),
    )


def test_factory_is_off_when_topk_v2_is_off(sm, monkeypatch):
    # sm_120 cannot run topk_v2 (model_hook turns it off), and the DeepGEMM indexer's
    # publish/select both call it, so no such indexer may be built there. The
    # configurer stub raises if the factory gets past the gate.
    sm(120)
    _paged_sparse_logits_configurer(monkeypatch, False)
    with envs.SGLANG_OPT_USE_TOPK_V2.override(False):
        assert (
            ci.make_candidate_indexer(topk_blocks=8, block_size=8, source_layer_id=20)
            is None
        )


def test_factory_builds_the_deep_gemm_indexer_on_sm100_with_topk_v2(sm, monkeypatch):
    sm(100)
    _paged_sparse_logits_configurer(monkeypatch, True)

    class Stub:
        def __init__(self, topk_blocks, block_size):
            self.args = (topk_blocks, block_size)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm",
        types.SimpleNamespace(DeepGemmCandidateIndexer=Stub),
    )
    with envs.SGLANG_OPT_USE_TOPK_V2.override(True):
        built = ci.make_candidate_indexer(topk_blocks=8, block_size=8, source_layer_id=20)
    assert isinstance(built, Stub) and built.args == (8, 8)


# --- decode dispatch when no candidate indexer exists (sm_120) --------------------


def _indexer(*, source=False, uses=False):
    return types.SimpleNamespace(
        is_candidate_source=source,
        uses_candidates=uses,
        candidate_topk_blocks=1,
        candidate_block_size=4,
        index_topk=2,
    )


def _dispatch(monkeypatch, *, candidate_indexer, indexer, variant=None):
    """Which decode function `_low_ratio_index_topk` picks on an sm_100+ device."""
    from sglang.srt.model_executor.runner_utils import capture_mode

    calls = []
    cls = backend_mod.DeepseekV4AttnBackend
    monkeypatch.setattr(backend_mod, "_is_sm100_or_newer", lambda: True)
    monkeypatch.setattr(
        cls, "_low_ratio_index_topk_decode", lambda self, *a, **k: calls.append("paged")
    )
    monkeypatch.setattr(
        cls, "_low_ratio_index_topk_sm90_decode", lambda self, *a, **k: calls.append("masks")
    )
    monkeypatch.setattr(capture_mode, "_capture_attention_variant", variant)
    backend = object.__new__(cls)
    backend.candidate_indexer = candidate_indexer
    layer = types.SimpleNamespace(indexer=indexer, compress_ratio=1)
    mode = types.SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False)
    cls._low_ratio_index_topk(
        backend, layer, None, None, None, None, types.SimpleNamespace(forward_mode=mode)
    )
    return calls


@pytest.mark.parametrize("role", ["source", "consumer"])
def test_candidate_layers_select_through_masks_without_a_candidate_indexer(monkeypatch, role):
    indexer = _indexer(source=role == "source", uses=role == "consumer")
    calls = _dispatch(monkeypatch, candidate_indexer=None, indexer=indexer)
    assert calls == ["masks"]


def test_plain_layers_keep_the_paged_decode_without_a_candidate_indexer(monkeypatch):
    assert _dispatch(monkeypatch, candidate_indexer=None, indexer=_indexer()) == ["paged"]


def test_candidate_layers_keep_the_paged_decode_with_a_candidate_indexer(monkeypatch):
    calls = _dispatch(
        monkeypatch, candidate_indexer=object(), indexer=_indexer(source=True)
    )
    assert calls == ["paged"]


@pytest.mark.parametrize("variant", ["candidate_all", "candidate_c2_all", "candidate_unfiltered"])
def test_graphs_where_every_request_fits_keep_the_paged_decode(monkeypatch, variant):
    calls = _dispatch(
        monkeypatch, candidate_indexer=None, indexer=_indexer(uses=True), variant=variant
    )
    assert calls == ["paged"]


def test_mask_decode_never_reaches_topk_v2_and_consumers_stay_inside_the_mask(monkeypatch):
    def poison(*args, **kwargs):
        raise AssertionError("topk_v2 / paged top-k must not run on the mask path")

    for name in ("topk_transform_paged_v2", "topk_transform_paged_from_metadata"):
        if hasattr(backend_mod, name):
            monkeypatch.setattr(backend_mod, name, poison)
    width = 16

    def logits(q, weights, slots, lens, table, page_size):
        j = torch.arange(slots.shape[1])[None, :]
        scores = ((j * 7) % 16).float().expand(slots.shape[0], -1).clone()
        return scores.masked_fill(j >= lens[:, None], -INF)

    monkeypatch.setattr(backend_mod, "fp4_index_logits_decode", logits)
    cls = backend_mod.DeepseekV4AttnBackend
    bs, ratio = 2, 1
    backend = object.__new__(cls)
    backend.req_to_token = torch.arange(bs * 64, dtype=torch.int32).view(bs, 64)
    backend.forward_metadata = types.SimpleNamespace(
        c1_indexer_metadata=types.SimpleNamespace(max_compressed_seq_len=width),
        c2_indexer_metadata=None,
        candidate_metadata=None,
    )
    page = {}
    core = types.SimpleNamespace(
        sparse_page_indices=lambda r: page.setdefault("p", torch.zeros(bs, 4, dtype=torch.int32)),
        sparse_raw_indices=lambda r: page.setdefault("r", torch.zeros(bs, 4, dtype=torch.int32)),
    )
    backend.forward_metadata.core_metadata = core
    backend.token_to_kv_pool = types.SimpleNamespace(
        get_index_k_with_scale_buffer=lambda layer_id: torch.zeros(4, 68 * 4, dtype=torch.uint8)
    )
    req = torch.tensor([0, 1])
    pos = torch.tensor([15, 9])  # visible lengths 16 and 10

    def layer(indexer):
        indexer.queries = lambda q_lora, freqs: torch.zeros(bs, 1, 128)
        indexer.head_weights = lambda x: torch.zeros(bs, 1)
        return types.SimpleNamespace(
            indexer=indexer,
            compress_ratio=ratio,
            layer_id=0,
            freqs_cis=torch.zeros(16, 1),
        )

    cls._low_ratio_index_topk_sm90_decode(
        backend, layer(_indexer(source=True)), None, None, req, pos
    )
    mask = backend.forward_metadata.candidate_metadata.mask
    # one best block of four positions plus the newest block; row 1 spans blocks 0-2
    assert mask.shape == (bs, width)
    assert mask[0].sum() <= 8 and mask[0, 12:].all()

    page["r"].zero_()
    cls._low_ratio_index_topk_sm90_decode(
        backend, layer(_indexer(uses=True)), None, None, req, pos
    )
    chosen = page["r"]
    for b in range(bs):
        picked = chosen[b][chosen[b] >= 0]
        assert picked.numel() > 0
        assert mask[b, picked.long()].all()


@pytest.mark.parametrize("topk_v2, expected", [(False, False), (True, True)])
def test_dense_fp4_prefill_indexer_needs_topk_v2(monkeypatch, topk_v2, expected):
    # The dense prefill selects with topk_transform_ragged_v2, which sm_120 cannot run
    # (model_hook turns topk_v2 off there); prefill then takes the torch indexer.
    monkeypatch.setattr(backend_mod, "_has_dense_fp4_indexer", lambda: True)
    batch = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_extend=lambda: True),
        seq_lens_cpu=[8],
        extend_seq_lens_cpu=[8],
    )
    use = backend_mod.DeepseekV4AttnBackend._use_dense_fp4_prefill_indexer
    with envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False):
        with envs.SGLANG_OPT_USE_TOPK_V2.override(topk_v2):
            assert use(batch) is expected


def _tail_masks_by_chunks(logits, lens, keep_from, rows_per_chunk, topk_blocks, block_size):
    kept = []
    for start in range(0, logits.shape[0], rows_per_chunk):
        s = logits[start : start + rows_per_chunk]
        sl = ci.keep_row_slice(start, s.shape[0], keep_from)
        if sl is not None:
            kept.append(
                ci.select_candidate_blocks(
                    s[sl], lens[start : start + rows_per_chunk][sl][:, None],
                    topk_blocks=topk_blocks, block_size=block_size,
                )
            )
    if kept:
        return torch.cat(kept)
    return torch.zeros(0, logits.shape[1], dtype=torch.bool)


@pytest.mark.parametrize("keep_from", [0, 1, 33, 34, 35, 250, 299, 300])
def test_tail_rows_built_by_chunks_equal_the_full_mask_tail(keep_from):
    g = torch.Generator().manual_seed(keep_from)
    rows, width = 300, 97
    logits = torch.randn(rows, width, generator=g)
    lens = torch.randint(1, width + 1, (rows,), generator=g)
    logits = logits.masked_fill(torch.arange(width)[None, :] >= lens[:, None], -INF)
    full = ci.select_candidate_blocks(logits, lens[:, None], topk_blocks=4, block_size=8)
    got = _tail_masks_by_chunks(logits, lens, keep_from, 34, 4, 8)
    assert torch.equal(got, full[keep_from:])


def test_keep_row_slice_bounds():
    assert ci.keep_row_slice(0, 34, 0) == slice(0, 34)
    assert ci.keep_row_slice(0, 34, 34) is None
    assert ci.keep_row_slice(34, 34, 40) == slice(6, 34)
    assert ci.keep_row_slice(68, 10, 40) == slice(0, 10)


def _tail_meta(extend_lens_cpu, local_lens_cpu=None):
    tail = types.SimpleNamespace(
        extend_seq_lens_cpu=extend_lens_cpu,
        local_lens_cpu=local_lens_cpu,
        cp_metadata=object() if local_lens_cpu is not None else None,
    )
    return types.SimpleNamespace(late_layer_tail=tail)


def test_candidate_publish_rows_without_tail_keeps_every_row():
    assert backend_mod.candidate_publish_rows(None) is None


def test_candidate_publish_rows_per_request_and_short_extends():
    assert backend_mod.candidate_publish_rows(_tail_meta([128, 57, 128])) == [128, 57, 128]


def test_candidate_publish_rows_uses_local_lens_under_cp():
    assert backend_mod.candidate_publish_rows(_tail_meta([128], local_lens_cpu=[64])) == [64]


# --- driving _low_ratio_index_topk_torch on a fake backend ------------------------


def _fake_two_request_setup(monkeypatch, *, rows_per_chunk):
    """Two requests (5 and 4 compressed positions, ratio 1) sharing one fake backend's
    dependencies for `_low_ratio_index_topk_torch`."""
    monkeypatch.setattr(
        backend_mod,
        "_torch_indexer_rows_per_chunk",
        lambda num_heads, lc: rows_per_chunk,
    )
    req = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    pos = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3])
    req_to_token = torch.arange(20, dtype=torch.int64).view(2, 10)
    layer = types.SimpleNamespace(
        compress_ratio=1, indexer=None, layer_id=0, freqs_cis=torch.zeros(10, 1)
    )
    return req, pos, req_to_token, layer


def _fake_scores(q_sub, index_k, weights_sub):
    # Each row's score depends only on its own query value: real-scores rows are
    # independent, so a fake reproducing that lets chunk-vs-whole runs agree exactly.
    lc = index_k.shape[0]
    qv = q_sub.squeeze(-1)
    cols = torch.arange(lc, dtype=torch.float32)
    return torch.sin(qv[:, None] * 0.37 + cols[None, :] * 0.11) * 5


def _fake_indexer(*, source, uses):
    return types.SimpleNamespace(
        is_candidate_source=source,
        uses_candidates=uses,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        index_topk=3,
        queries=lambda q_lora, freqs: q_lora,
        head_weights=lambda x: x,
        scores=_fake_scores,
    )


def _run_torch_indexer(
    layer,
    indexer,
    req,
    pos,
    req_to_token,
    *,
    tail_metadata,
    candidate_tail_only,
    skip=False,
):
    layer.indexer = indexer
    page_indices = torch.zeros(req.numel(), 3, dtype=torch.int32)
    core = types.SimpleNamespace(
        sparse_page_indices=lambda ratio: page_indices,
        sparse_raw_indices=lambda ratio: None,
    )
    forward_metadata = types.SimpleNamespace(
        core_metadata=core, candidate_metadata=None, layer_major_skip_candidates=skip
    )
    backend = types.SimpleNamespace(
        token_to_kv_pool=types.SimpleNamespace(
            get_low_ratio_index_k_dequant=(
                lambda layer_id, slots_j: torch.zeros(slots_j.shape[0], 1)
            )
        ),
        forward_metadata=forward_metadata,
        tail_forward_metadata=tail_metadata,
        candidate_tail_only=candidate_tail_only,
        req_to_token=req_to_token,
    )
    # Identity queries()/head_weights() make x and q_lora interchangeable here.
    q_lora = torch.arange(req.numel(), dtype=torch.float32).unsqueeze(-1)
    backend_mod.DeepseekV4AttnBackend._low_ratio_index_topk_torch(
        backend, layer, q_lora, q_lora, req, pos
    )
    return forward_metadata.candidate_metadata, page_indices


_TAIL_METADATA = types.SimpleNamespace(
    late_layer_tail=types.SimpleNamespace(
        extend_seq_lens_cpu=[3, 2], local_lens_cpu=None, cp_metadata=None
    )
)


def test_torch_indexer_publishes_tail_rows_matching_the_full_mask_tail(monkeypatch):
    req, pos, req_to_token, layer = _fake_two_request_setup(
        monkeypatch, rows_per_chunk=3
    )
    source = _fake_indexer(source=True, uses=False)
    full_meta, full_pages = _run_torch_indexer(
        layer,
        source,
        req,
        pos,
        req_to_token,
        tail_metadata=None,
        candidate_tail_only=True,
    )
    tail_meta, tail_pages = _run_torch_indexer(
        layer,
        source,
        req,
        pos,
        req_to_token,
        tail_metadata=_TAIL_METADATA,
        candidate_tail_only=True,
    )
    assert torch.equal(tail_meta.request_masks[0], full_meta.request_masks[0][-3:])
    assert torch.equal(tail_meta.request_masks[1], full_meta.request_masks[1][-2:])
    # The source's own top-k (page_indices) does not depend on tail-only publishing.
    assert torch.equal(full_pages, tail_pages)


def test_torch_indexer_falls_back_to_full_rows_when_a_consumer_precedes_the_tail(
    monkeypatch,
):
    req, pos, req_to_token, layer = _fake_two_request_setup(
        monkeypatch, rows_per_chunk=3
    )
    source = _fake_indexer(source=True, uses=False)
    full_meta, _ = _run_torch_indexer(
        layer,
        source,
        req,
        pos,
        req_to_token,
        tail_metadata=None,
        candidate_tail_only=True,
    )
    unsafe_meta, _ = _run_torch_indexer(
        layer,
        source,
        req,
        pos,
        req_to_token,
        tail_metadata=_TAIL_METADATA,
        candidate_tail_only=False,
    )
    assert len(full_meta.request_masks) == len(unsafe_meta.request_masks)
    for full_b, unsafe_b in zip(full_meta.request_masks, unsafe_meta.request_masks):
        assert torch.equal(full_b, unsafe_b)


def test_torch_indexer_skip_flag_publishes_no_masks(monkeypatch):
    req, pos, req_to_token, layer = _fake_two_request_setup(
        monkeypatch, rows_per_chunk=3
    )
    source = _fake_indexer(source=True, uses=False)
    meta, _ = _run_torch_indexer(
        layer,
        source,
        req,
        pos,
        req_to_token,
        tail_metadata=None,
        candidate_tail_only=True,
        skip=True,
    )
    assert isinstance(meta, backend_mod.CandidateMasks)
    assert meta.request_masks == []


def test_torch_indexer_raises_when_a_tail_length_exceeds_its_request_rows(monkeypatch):
    req, pos, req_to_token, layer = _fake_two_request_setup(
        monkeypatch, rows_per_chunk=3
    )
    source = _fake_indexer(source=True, uses=False)
    oversized_tail = types.SimpleNamespace(
        late_layer_tail=types.SimpleNamespace(
            extend_seq_lens_cpu=[9, 2], local_lens_cpu=None, cp_metadata=None
        )
    )
    with pytest.raises(ValueError, match="exceeds request"):
        _run_torch_indexer(
            layer,
            source,
            req,
            pos,
            req_to_token,
            tail_metadata=oversized_tail,
            candidate_tail_only=True,
        )


# --- candidate_tail_only_of: config-only, no model_runner.model (fix round 2) -----


def _exec_with_bounded_replay(enabled):
    return types.SimpleNamespace(
        features=types.SimpleNamespace(enable_decoder_swa_bounded_replay=enabled)
    )


def test_candidate_tail_only_true_for_a_target_shaped_config(monkeypatch):
    # source layer 20, kv sources ending at 20 -> late_layer_start 21, 20 >= 21-1: tail
    # only.
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: _exec_with_bounded_replay(True),
    )
    config = types.SimpleNamespace(
        candidate_source_layer_id=20, kv_source_layer_ids=[20]
    )
    assert backend_mod.candidate_tail_only_of(config) is True


def test_candidate_tail_only_false_when_a_consumer_precedes_the_tail(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: _exec_with_bounded_replay(True),
    )
    config = types.SimpleNamespace(
        candidate_source_layer_id=10, kv_source_layer_ids=[20]
    )
    assert backend_mod.candidate_tail_only_of(config) is False


def test_candidate_tail_only_true_when_bounded_replay_is_off(monkeypatch):
    # No tail ever forms, so nothing is unsafe to publish in full: same as before this
    # fix.
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: _exec_with_bounded_replay(False),
    )
    config = types.SimpleNamespace(
        candidate_source_layer_id=20, kv_source_layer_ids=[20]
    )
    assert backend_mod.candidate_tail_only_of(config) is True


def test_candidate_tail_only_of_never_needs_a_model_for_a_draft_shaped_config(
    monkeypatch,
):
    # NextN/DSpark draft configs default candidate_source_layer_id to -1 (drafts run
    # no candidate indexing) and never populate kv_source_layer_ids. late_layer_start_of
    # would raise on an empty list, so the short-circuit on candidate_source_layer_id
    # < 0 must skip it even when bounded replay happens to be on globally. The backend
    # used to read model_runner.model.model.late_layer_start instead: NextN never sets
    # that attribute and DSpark's model has no `.model` at all, so every MTP/DSpark
    # launch crashed at backend construction. candidate_tail_only_of takes only a
    # config, never a model.
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: _exec_with_bounded_replay(True),
    )
    draft_config = types.SimpleNamespace(
        candidate_source_layer_id=-1, kv_source_layer_ids=[]
    )
    assert backend_mod.candidate_tail_only_of(draft_config) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
