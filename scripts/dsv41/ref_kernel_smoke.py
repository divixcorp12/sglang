"""Smoke-test the reference's six tilelang kernel entry points on the smallest legal shapes.

Imports only `inference/kernel.py` from the official snapshot -- never `model.py`, which would
build a model. Shapes/dtypes for each call are taken from the kernel's own signature/asserts and
from how `inference/model.py` calls it (cited below by line number in the snapshot copied to
/tmp/model.py.txt and /tmp/kernel.py.txt while writing this script; re-read the live files on
divix01 if the snapshot has moved).

Only `sparse_attn` and `hc_split_sinkhorn` are required by the bf16 reference oracle (Task 8's
`ref_oracle.py`); the other four are reported here for the DSV41_REFERENCE.md sm_120 kernel
matrix (Sec. 6). Needs a GPU: run only inside a granted GPU window (Task 10).
"""

import argparse
import sys
import traceback


def _act_quant_case(kernel):
    # kernel.py: act_quant(x, block_size, scale_fmt, scale_dtype, inplace=False) -> (y, s)
    # Called at model.py:707 as act_quant(kv, fp8_block_size=32, scale_fmt, scale_dtype, True)
    # and model.py:1042/1062 the same way. block_size must divide the last dim (N % block_size == 0).
    import torch

    x = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")
    kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, inplace=False)


def _fp4_act_quant_case(kernel):
    # kernel.py: fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=e8m0) -> (y, s)
    # Called at model.py:546/552/760 with block_size 16 or 32, inplace=True; here out-of-place so
    # the return value is exercised too. N % block_size == 0.
    import torch

    x = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")
    kernel.fp4_act_quant(x, 32, inplace=False, scale_dtype=torch.float8_e8m0fnu)


def _fp8_gemm_case(kernel):
    # kernel.py: fp8_gemm(a, a_s, b, b_s, scale_dtype, block_size) -> c[M,N]
    # a: fp8 [M,K], a_s: [M, K/block]; b: fp8 [N,K], b_s: [ceil(N/block), K/block].
    # Called via model.py's linear()/model.py:196-205 with block_size=fp8_block_size=32.
    import torch

    m, n, k, block = 1, 1, 32, 32
    a = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda")
    a_s = torch.ones(m, k // block, dtype=torch.float32, device="cuda")
    b_s = torch.ones((n + block - 1) // block, k // block, dtype=torch.float32, device="cuda")
    kernel.fp8_gemm(a, a_s, b, b_s, torch.float32, block)


def _sparse_attn_case(kernel):
    # kernel.py: sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale) -> o
    # q: [b,m,h,d] bf16, kv: [b,n,d] bf16, attn_sink: [h] fp32, topk_idxs: [b,m,topk] int32.
    # Called at model.py:780 and model.py:1067; the wrapper itself pads h < 16 to 16.
    import torch

    b, m, h, d, n, topk = 1, 1, 1, 16, 1, 1
    q = torch.zeros(b, m, h, d, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(b, n, d, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.zeros(h, dtype=torch.float32, device="cuda")
    topk_idxs = torch.zeros(b, m, topk, dtype=torch.int32, device="cuda")
    kernel.sparse_attn(q, kv, attn_sink, topk_idxs, d**-0.5)


def _hc_split_sinkhorn_case(kernel):
    # kernel.py: hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6)
    #   -> (pre, post, comb)
    # mixes: [b,s,(2+hc)*hc] fp32, hc_scale: [3] fp32, hc_base: [(2+hc)*hc] fp32.
    # Called at model.py:955 with hc_mult=args.hc_mult (4 in the released config); hc_mult=1 here
    # is the smallest legal value the kernel's own shape math allows.
    import torch

    hc_mult = 1
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.zeros(1, 1, mix_hc, dtype=torch.float32, device="cuda")
    hc_scale = torch.ones(3, dtype=torch.float32, device="cuda")
    hc_base = torch.zeros(mix_hc, dtype=torch.float32, device="cuda")
    kernel.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, 2, 1e-6)


def _fp4_gemm_case(kernel):
    # kernel.py: fp4_gemm(a, a_s, b, b_s, scale_dtype, act_block_size) -> c[M,N]
    # a: fp8 [M,K], a_s: [M, K/act_block]; b: fp4-packed [N, K/2] (float4_e2m1fn_x2), b_s: [N, K/32].
    # Called via model.py's linear()/model.py:186-195 with act_block_size=fp8_block_size=32.
    import torch

    m, n, k, act_block = 1, 1, 32, 32
    a = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.zeros(n, k // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    a_s = torch.ones(m, k // act_block, dtype=torch.float32, device="cuda")
    b_s = torch.ones(n, k // 32, dtype=torch.float32, device="cuda")
    kernel.fp4_gemm(a, a_s, b, b_s, torch.float32, act_block)


CASES = {
    "act_quant": _act_quant_case,
    "fp4_act_quant": _fp4_act_quant_case,
    "fp8_gemm": _fp8_gemm_case,
    "sparse_attn": _sparse_attn_case,
    "hc_split_sinkhorn": _hc_split_sinkhorn_case,
    "fp4_gemm": _fp4_gemm_case,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="official DeepSeek-V4.1-Flash snapshot dir")
    args = parser.parse_args()

    inference_dir = f"{args.snapshot}/inference"
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    import kernel  # the reference's own kernel.py; never import model.py here

    failed = []
    for name, case in CASES.items():
        try:
            case(kernel)
        except Exception:  # noqa: BLE001 -- report every kernel's failure, don't stop early
            print(f"FAIL {name}")
            traceback.print_exc()
            failed.append(name)
        else:
            print(f"ok {name}")

    if failed:
        print(f"failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
