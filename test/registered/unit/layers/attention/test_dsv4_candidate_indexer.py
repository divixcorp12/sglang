"""Candidate indexer: level-one block selection, score masking, and factory gating."""

import sys
import types

import pytest
import torch

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
