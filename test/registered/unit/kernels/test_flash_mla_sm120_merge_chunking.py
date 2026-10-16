"""SM120 triton FlashMLA: a merge budget changes only how the merge and sink are chunked.

`flash_mla_sparse_decode_triton` merges the SWA and extra (c4/c128) partial outputs and
applies the attention sink in fp32, which at a 4096-token prefill chunk (64 heads, 512
dims) holds 3 fp32 copies of the output, 1.5 GiB. Under
SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB the same elementwise ops run over token
chunks written in place, so the output and LSE must be bitwise the one-pass path's.
The cases cover chunk sizes that do not divide the token count, a single-token chunk,
rows whose LSE is -inf on one or both sides, sink-only layers (no extra cache), and on
CUDA the real kernel at the production shape with its peak memory.
"""

import pytest
import torch

from sglang.kernels.ops.attention import flash_mla_sm120_triton as tmod
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

BUDGET = envs.SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SM120 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="needs an SM120 GPU for the triton kernel",
)


def _partials(tokens: int, heads: int, dim: int, device: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def out():
        x = torch.randn(tokens, 1, heads, dim, generator=g)
        return x.to(device, torch.bfloat16)

    def lse():
        x = (torch.randn(tokens, 1, heads, generator=g) * 4).to(device)
        # Rows with nothing to attend on one side (the kernel leaves -inf).
        x[::5, :, ::3] = float("-inf")
        return x

    out1, lse1, out2, lse2 = out(), lse(), out(), lse()
    # Both sides empty: the merge's lonely rows.
    lse2[::10, :, ::3] = float("-inf")
    sink = torch.randn(heads, generator=g).to(device)
    return out1, lse1, out2, lse2, sink


def _one_pass(out1, lse1, out2, lse2, sink):
    out, lse = out1, lse1
    if out2 is not None:
        out, lse = tmod._merge_partial_attn(out, lse, out2, lse2)
    if sink is not None:
        out, lse = tmod._apply_attn_sink(out, lse, sink)
    return out, lse


def test_chunk_tokens_from_budget():
    # 3 fp32 copies of 64 x 512 per token = 384 KiB a token.
    assert tmod.merge_chunk_tokens(64, 512, 64) == (64 << 20) // (3 * 64 * 512 * 4)
    assert tmod.merge_chunk_tokens(64, 512, 64) == 170
    assert tmod.merge_chunk_tokens(64, 512, 1536) == 4096
    assert tmod.merge_chunk_tokens(64, 512, 0) == 1
    assert tmod.merge_chunk_tokens(4, 16, 1) == (1 << 20) // (3 * 4 * 16 * 4)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("extra", [True, False])
@pytest.mark.parametrize("sink", [True, False])
@pytest.mark.parametrize(
    "tokens,chunk", [(1, 1), (7, 3), (64, 64), (65, 64), (300, 17), (1000, 1)]
)
def test_chunked_merge_is_bitwise_one_pass(device, extra, sink, tokens, chunk):
    if not extra and not sink:
        pytest.skip("nothing to merge")
    out1, lse1, out2, lse2, s = _partials(tokens, 8, 32, device, seed=tokens)
    out2, lse2 = (out2, lse2) if extra else (None, None)
    s = s if sink else None
    ref_out, ref_lse = _one_pass(out1.clone(), lse1.clone(), out2, lse2, s)
    got_out, got_lse = tmod._merge_and_sink_chunked(out1, lse1, out2, lse2, s, chunk)
    assert got_out.data_ptr() == out1.data_ptr()  # written in place
    assert torch.equal(got_out, ref_out)
    assert torch.equal(got_lse, ref_lse)


def _kv_cache(num_pages: int, page_size: int, seed: int) -> torch.Tensor:
    """A random DSv4 paged cache (fp8 nope, bf16 rope, UE8M0 scales), no NaN/Inf."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (num_pages, page_size)
    data = torch.zeros(*shape, tmod._TOKEN_DATA_STRIDE, dtype=torch.uint8)
    data[..., : tmod._NOPE_DIM] = torch.randint(
        0, 0x70, (*shape, tmod._NOPE_DIM), generator=g, dtype=torch.uint8
    )
    rope = torch.randn(*shape, tmod._ROPE_DIM, generator=g).clamp(-2, 2)
    data[..., tmod._NOPE_DIM :] = rope.to(torch.bfloat16).view(torch.uint8)
    scales = torch.zeros(*shape, tmod._SCALE_STRIDE, dtype=torch.uint8)
    scales[..., :7] = torch.randint(
        120, 131, (*shape, 7), generator=g, dtype=torch.uint8
    )
    flat = torch.cat(
        [data.reshape(num_pages, -1), scales.reshape(num_pages, -1)], dim=1
    )
    bpt = tmod._TOKEN_DATA_STRIDE + tmod._SCALE_STRIDE
    return flat.cuda().view(*shape, 1, bpt).view(torch.float8_e4m3fn)


def _call(tokens: int, extra: bool, sink: bool, seed: int = 0):
    heads, page_size = 64, 64
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(tokens, 1, heads, tmod._D, generator=g) * 0.5
    q = q.cuda().to(torch.bfloat16)
    kv = _kv_cache(16, page_size, seed)
    idx = torch.randint(
        0, 16 * page_size, (tokens, 1, 128), generator=g, dtype=torch.int32
    ).cuda()
    idx[::7, :, 64:] = -1
    kwargs = {}
    if extra:
        kwargs["extra_k_cache"] = _kv_cache(32, page_size, seed + 1)
        kwargs["extra_indices"] = torch.randint(
            0, 32 * page_size, (tokens, 1, 512), generator=g, dtype=torch.int32
        ).cuda()
        kwargs["extra_topk_length"] = torch.randint(
            0, 513, (tokens,), generator=g, dtype=torch.int32
        ).cuda()
    attn_sink = torch.randn(heads, generator=g).cuda() if sink else None
    return lambda: tmod.flash_mla_sparse_decode_triton(
        q, kv, idx, None, attn_sink, tmod._D, tmod._D**-0.5, **kwargs
    )


@SM120
@pytest.mark.parametrize("extra", [True, False])
@pytest.mark.parametrize("tokens", [4096, 2048 + 17, 171])
@pytest.mark.parametrize("budget_mb", [64, 1])
def test_kernel_output_is_bitwise_unchunked(extra, tokens, budget_mb):
    run = _call(tokens, extra, sink=True)
    run()  # autotune once, so both calls below use the same kernel config
    with BUDGET.override(0):
        ref_out, ref_lse = run()
    with BUDGET.override(budget_mb):
        assert tmod.merge_chunk_tokens(64, tmod._D, budget_mb) < tokens
        got_out, got_lse = run()
    assert torch.equal(got_out, ref_out)
    assert torch.equal(got_lse, ref_lse)


CHUNK_KERNELS = envs.SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS


def test_kernel_chunk_tokens_from_budget():
    # Per token: two bf16 partials and three fp32 merge copies of 64 x 512 = 1 MiB a token.
    assert tmod.kernel_chunk_tokens(64, 512, 32) == 32
    assert tmod.kernel_chunk_tokens(64, 512, 128) == 128
    assert tmod.kernel_chunk_tokens(64, 512, 0) == 1


@SM120
@pytest.mark.parametrize("extra", [True, False])
@pytest.mark.parametrize("tokens", [4096, 2048 + 17, 171])
@pytest.mark.parametrize("budget_mb", [32, 1])
def test_chunked_kernels_are_bitwise_unchunked(extra, tokens, budget_mb):
    run = _call(tokens, extra, sink=True)
    run()  # autotune once
    with BUDGET.override(0), CHUNK_KERNELS.override(False):
        ref_out, ref_lse = run()
    with BUDGET.override(budget_mb), CHUNK_KERNELS.override(True):
        assert tmod.kernel_chunk_tokens(64, tmod._D, budget_mb) < tokens
        got_out, got_lse = run()
    assert torch.equal(got_out, ref_out)
    assert torch.equal(got_lse, ref_lse)


@SM120
def test_chunked_kernels_hold_one_full_output():
    """With the kernels chunked, only the final bf16 output (and its LSE) is full size; the partials and merge
    temporaries stay within the budget."""
    tokens, heads, budget_mb = 4096, 64, 32
    bf16_out = tokens * heads * tmod._D * 2
    run = _call(tokens, extra=True, sink=True)
    run()
    with BUDGET.override(budget_mb), CHUNK_KERNELS.override(True):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = run()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del out
    assert peak <= bf16_out + (budget_mb << 20) + (32 << 20), peak
    print(f"peak with chunked kernels at {budget_mb} MiB: {peak / 2**20:.0f} MiB")


@SM120
def test_budget_bounds_the_merge_peak():
    """At 4096 tokens the one-pass merge holds the two bf16 partials plus 3 fp32 copies
    (8 bf16 outputs in all); chunked, only the two partials plus the budget."""
    tokens, heads, budget_mb = 4096, 64, 64
    bf16_out = tokens * heads * tmod._D * 2
    run = _call(tokens, extra=True, sink=True)
    run()

    def peak(budget):
        with BUDGET.override(budget):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = run()
            torch.cuda.synchronize()
            del out
            return torch.cuda.max_memory_allocated() - base

    one_pass, chunked = peak(0), peak(budget_mb)
    assert one_pass >= 8 * bf16_out
    # The two partials, the budget, and the small [B, 1, H] and index tensors.
    bound = 2 * bf16_out + (budget_mb << 20) + (32 << 20)
    assert chunked <= bound, (chunked, one_pass)
    print(
        f"peak one-pass {one_pass / 2**20:.0f} MiB, "
        f"chunked at {budget_mb} MiB {chunked / 2**20:.0f} MiB"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
