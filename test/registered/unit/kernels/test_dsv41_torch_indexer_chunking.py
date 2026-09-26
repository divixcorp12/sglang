"""DSV4.1 torch prefill indexer: a score byte budget changes only how rows are chunked.

`_low_ratio_index_topk_torch` scores, masks and selects each query row on its own, so
running the rows in budget-sized chunks must give bitwise the same logits, top-k page
indices, raw indices and candidate masks as one pass. The cases below force chunk
sizes that do not divide the row count, include ties (integer-valued scores, and
rows whose relu-summed score is exactly 0), a request shorter than index_topk, and
on CUDA the production shape (512 rows, 32 heads, a 30k-position ratio-1 context).
"""

import functools
import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as backend_mod
from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateMasks
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

BUDGET = envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_scores = functools.partial(DeepseekV41Indexer.scores, None)


def _mib_for_rows(rows: int, heads: int, lc: int) -> int:
    """The smallest whole-MiB budget that gives exactly `rows` rows per chunk."""
    per_row = heads * lc * 2
    mb = -(-rows * per_row // (1 << 20))
    assert (mb << 20) // per_row == rows, (rows, heads, lc)
    return mb


def test_default_is_the_builtin_one_gib():
    with BUDGET.override(0):
        assert backend_mod._torch_indexer_rows_per_chunk(32, 30000) == (1 << 30) // (
            32 * 30000 * 2
        )
        # a context too wide for even one row still makes progress
        assert backend_mod._torch_indexer_rows_per_chunk(32, 1 << 30) == 1
    with BUDGET.override(256):
        assert backend_mod._torch_indexer_rows_per_chunk(32, 30000) == (256 << 20) // (
            32 * 30000 * 2
        )


def _case(device, *, heads, lc0, rows0, rows1, ties, seed=0):
    """Two requests in one extend batch: request 0 scores the last `rows0` positions
    of an `lc0`-position ratio-1 context, request 1 is a fresh `rows1`-token prompt."""
    g = torch.Generator().manual_seed(seed)
    d = 128

    def rand(*shape):
        if ties:  # small integers: exact products, many equal scores, many relu zeros
            return torch.randint(-2, 3, shape, generator=g).to(torch.bfloat16)
        return torch.randn(shape, generator=g).to(torch.bfloat16)

    width = max(lc0, rows1)
    req = torch.cat([torch.zeros(rows0, dtype=torch.int64), torch.ones(rows1, dtype=torch.int64)])
    pos = torch.cat([torch.arange(lc0 - rows0, lc0), torch.arange(rows1)])
    n = pos.numel()
    return types.SimpleNamespace(
        heads=heads,
        req=req.to(device),
        pos=pos.to(device),
        q=rand(n, heads, d).to(device),
        w=rand(n, heads).to(device),
        index_k=rand(2 * width, d).to(device),
        # request r's position j lives in pool slot r * width + j
        req_to_token=(
            torch.arange(2)[:, None] * width + torch.arange(width)[None, :]
        ).to(torch.int32).to(device),
        n=n,
        width=width,
        rows0=rows0,
    )


def _run(case, budget_mb, *, source=False, uses=False, published=None, topk=512):
    """One `_low_ratio_index_topk_torch` call on a stand-in backend; returns the
    page and raw indices it wrote and, for a candidate source, the masks it published."""
    device = case.q.device
    page = torch.zeros(case.n, topk, dtype=torch.int32, device=device)
    raw = torch.zeros(case.n, topk, dtype=torch.int32, device=device)
    backend = object.__new__(backend_mod.DeepseekV4AttnBackend)
    backend.req_to_token = case.req_to_token
    backend.token_to_kv_pool = types.SimpleNamespace(
        get_low_ratio_index_k_dequant=lambda layer_id, slots: case.index_k[slots]
    )
    backend.forward_metadata = types.SimpleNamespace(
        core_metadata=types.SimpleNamespace(
            sparse_page_indices=lambda ratio: page,
            sparse_raw_indices=lambda ratio: raw,
        ),
        candidate_metadata=published,
    )
    indexer = types.SimpleNamespace(
        queries=lambda q_lora, freqs: case.q,
        head_weights=lambda x: case.w,
        scores=_scores,
        index_topk=topk,
        is_candidate_source=source,
        uses_candidates=uses,
        candidate_topk_blocks=64,
        candidate_block_size=8,
    )
    layer = types.SimpleNamespace(
        indexer=indexer,
        compress_ratio=1,
        layer_id=0,
        freqs_cis=torch.zeros(case.width + 1, 1, device=device),
    )
    with BUDGET.override(budget_mb):
        backend_mod.DeepseekV4AttnBackend._low_ratio_index_topk_torch(
            backend, layer, None, None, case.req, case.pos
        )
    masks = backend.forward_metadata.candidate_metadata if source else None
    return page, raw, masks


def _assert_same(a, b):
    page_a, raw_a, masks_a = a
    page_b, raw_b, masks_b = b
    assert torch.equal(page_a, page_b)
    assert torch.equal(raw_a, raw_b)
    if masks_a is not None:
        assert len(masks_a.request_masks) == len(masks_b.request_masks)
        for x, y in zip(masks_a.request_masks, masks_b.request_masks):
            assert torch.equal(x, y)


def _check_all_roles(case, budgets):
    """Plain, candidate-source and candidate-consumer layers, each chunked at every
    budget, against the same layer run in one pass (budget 0 = the built-in 1 GiB,
    which holds every row here)."""
    lc = case.width
    rows = backend_mod._torch_indexer_rows_per_chunk
    with BUDGET.override(0):
        assert rows(case.heads, lc) >= case.rows0  # one pass per request
    whole = {
        "plain": _run(case, 0),
        "source": _run(case, 0, source=True),
    }
    consumer_masks = whole["source"][2]
    whole["consumer"] = _run(case, 0, uses=True, published=consumer_masks)
    # the one-pass run actually selected something, so equality is not vacuous
    assert (whole["plain"][0] >= 0).sum() > 0
    for mb in budgets:
        with BUDGET.override(mb):
            assert rows(case.heads, lc) < case.rows0  # the budget really splits the rows
        _assert_same(_run(case, mb), whole["plain"])
        _assert_same(_run(case, mb, source=True), whole["source"])
        _assert_same(
            _run(case, mb, uses=True, published=consumer_masks), whole["consumer"]
        )


@pytest.mark.parametrize("ties", [False, True])
def test_chunked_rows_match_one_pass_cpu(ties):
    # per row 32 heads x 16400 positions x 2 bytes; 41 + 13 rows
    case = _case("cpu", heads=32, lc0=16400, rows0=41, rows1=13, ties=ties)
    budgets = [_mib_for_rows(r, 32, case.width) for r in (1, 3, 7, 40)]
    _check_all_roles(case, budgets)


def test_chunked_scores_are_bitwise_the_one_pass_rows_cpu():
    case = _case("cpu", heads=32, lc0=5000, rows0=37, rows1=0, ties=False)
    k = case.index_k[: case.width]
    whole = _scores(case.q, k, case.w)
    for step in (1, 5, 36):
        parts = [
            _scores(case.q[s : s + step], k, case.w[s : s + step])
            for s in range(0, case.n, step)
        ]
        assert torch.equal(torch.cat(parts), whole)


@CUDA
@pytest.mark.parametrize("ties", [False, True])
def test_chunked_rows_match_one_pass_cuda_production_shape(ties):
    # A 512-row prefill chunk at the end of a 30k-token prompt, the shape Track A
    # saw (DSV41_REFERENCE.md 25.4), plus a short request under index_topk.
    case = _case("cuda", heads=32, lc0=30000, rows0=512, rows1=100, ties=ties)
    budgets = [_mib_for_rows(r, 32, case.width) for r in (1, 7, 100, 333)] + [64, 256]
    _check_all_roles(case, budgets)


@CUDA
def test_chunked_scores_are_bitwise_the_one_pass_rows_cuda():
    case = _case("cuda", heads=32, lc0=30000, rows0=512, rows1=0, ties=False)
    k = case.index_k[: case.width]
    whole = _scores(case.q, k, case.w)
    for step in (1, 7, 100, 333):
        parts = [
            _scores(case.q[s : s + step], k, case.w[s : s + step])
            for s in range(0, case.n, step)
        ]
        assert torch.equal(torch.cat(parts), whole)
