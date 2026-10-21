"""Diagnostic for parity_prefill.py's decode failures: FlashInfer's <= 64-token path differs from the reference by
relL2 0.09 to 0.16 while its > 64-token path agrees to 5e-4 on the same builders. Which input feature moves it?

Varies, on a 6-token verify and a 1-token decode: sink on/off, extra pool on/off, ragged on/off, key magnitude
(the builder's nope bytes reach 0x6F, i.e. large fp8 values; a 'small' variant caps them at 0x30), q scale.
Prints relL2 of FlashInfer and Triton against the fp32 reference for each. Read-only diagnostic, no verdict.

Run like parity_prefill.py (GPU lock, cores 32-63, PYTHONPATH=$PWD/python).
"""

from __future__ import annotations

import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import parity_prefill as pp  # noqa: E402


def main():
    from sglang.kernels.ops.attention import flash_mla_sm120 as fmod

    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo, "test", "registered", "kernels", "ops", "attention"))
    import test_flash_mla_backends as t

    helpers = (t._build_kvcache, t._build_q_indices)
    device = torch.device("cuda")
    print("sglang", fmod.__file__)

    def run(label, tokens, extra_pbs, extra_topk, sink, ragged, qscale=1.0, cap=None):
        kw = pp.build_case(fmod, helpers, tokens, extra_pbs, extra_topk, sink, ragged, 11, device)
        kw["q"] = (kw["q"].float() * qscale).to(torch.bfloat16)
        if cap is not None:
            # clamp the fp8 nope bytes of every pool to <= cap (keeps the sign bit clear: positive small values)
            for k in ("k_cache", "extra_k_cache"):
                if k in kw:
                    v = kw[k].view(torch.uint8)
                    pbs = v.shape[1]
                    flat = v.reshape(v.shape[0], -1)
                    data = flat[:, : pbs * 576].view(v.shape[0], pbs, 576)
                    data[:, :, :448] = torch.clamp(data[:, :, :448], max=cap)
        ref_out, _ = pp.run_reference(fmod, kw)
        tri, _ = pp.run_entry(fmod, "triton", kw)
        fi, _ = pp.run_entry(fmod, "flashinfer", kw)
        keep = torch.ones_like(ref_out, dtype=torch.bool)
        print(
            f"{label:44s} T={tokens:3d} relL2 triton {pp.rel_l2(tri, ref_out, keep):.2e} "
            f"flashinfer {pp.rel_l2(fi, ref_out, keep):.2e} |ref|max {float(ref_out.float().abs().max()):.1f}",
            flush=True,
        )

    for tokens in (1, 6, 64, 65):
        run("swa only, sink, regular", tokens, None, 0, True, False)
        run("swa only, no sink, regular", tokens, None, 0, False, False)
        run("swa only, no sink, regular, q x0.1", tokens, None, 0, False, False, qscale=0.1)
        run("swa only, no sink, regular, nope<=0x30", tokens, None, 0, False, False, cap=0x30)
        run("swa only, no sink, regular, nope<=0x30, q x0.1", tokens, None, 0, False, False, qscale=0.1, cap=0x30)
        run("c4, sink, ragged", tokens, 64, 512, True, True)
        run("c4, no sink, regular", tokens, 64, 512, False, False)
        run("c4, no sink, regular, nope<=0x30, q x0.1", tokens, 64, 512, False, False, qscale=0.1, cap=0x30)


if __name__ == "__main__":
    main()
