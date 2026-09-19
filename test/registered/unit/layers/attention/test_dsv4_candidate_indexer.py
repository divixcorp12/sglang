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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
