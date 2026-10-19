"""Step 2 of the FlashInfer-prefill check: FlashInfer against Triton on the DSV4.1 SM120 FlashMLA shapes.

Question: can SGLANG_SM120_FLASHMLA_BACKEND=triton (arm_env.dspark_env, line "Every DSpark text result so far ran
triton attention") be dropped, so prefill (> 64 tokens) runs FlashInfer's _sparse_mla_sm120_paged_attention and the
<= 64-token DSpark verify runs its decode path?

Each case is built from synthetic caches in the production layout (DSV4.1: 64 heads, D 512, SWA pool pbs 256 with a
128-token window, an extra pool of c4 pbs 64 / c2 pbs 128 / c128 pbs 2, sink on or off, ragged topk_length with -1
padding), and run through flash_mla_with_kvcache_sm120 with `_sm120_default_backend` patched, i.e. the code path the
server takes. Three outputs are compared per case:

  ref       the pure-PyTorch fp32 reference (_sm120_sparse_decode_fwd), run in token chunks: the ground truth
  triton    flash_mla_sparse_decode_triton through the entry point
  flashinfer  the entry point with the flashinfer backend (> 64 tokens: _flash_mla_sm120_prefill; else the decode path)

Verdict per case (a case that raises is reported as ERROR with its message and does not stop the others):
  out   relL2(x, ref) = ||x - ref||_2 / ||ref||_2 over the float32 output; PASS when
        relL2(flashinfer, ref) <= max(REL_L2_FLOOR, REL_L2_FACTOR * relL2(triton, ref)), and no non-finite value.
        REL_L2_FLOOR 2e-2 (bf16 output with an fp8 KV: 2^-8 is 4e-3 per element), REL_L2_FACTOR 1.5.
        Also reported, not part of the verdict: max |flashinfer - triton| and the count of elements outside the
        existing tests' atol 5e-2 / rtol 5e-2 (test_flash_mla_backends.py).
  lse   the entry point returns None for lse on the flashinfer path, so FlashInfer's lse comes from calling
        _sparse_mla_sm120_paged_attention directly on the same split caches (a copy of _flash_mla_sm120_prefill's
        body for > 64 tokens, of _flash_mla_flashinfer's for fewer). PASS when max |flashinfer - ref| <= 2e-2 + 1e-3 |ref|
        over finite ref entries, and -inf agrees (lonely rows).
  The token rows with no valid key (topk_length 0 on both pools) are excluded from `out` and listed as `lonely`.

Also measured for the 4096-token cases: median kernel ms over REPS after a warm-up, and the peak extra allocation
(torch.cuda.max_memory_allocated above the memory held before the call) for triton at merge budget 0 and 128 MiB
and for flashinfer. This is the "+1.75 GiB at 4096" figure of DSV_OPTIMIZATION.md 3.7, measured on the installed stack.

Run on divix01, in a worktree at the pushed commit, under the GPU lock (never a bare command):

  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \\
    flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \\
    /data/models/slang/.venv-upstream/bin/python analysis/dsv41-drive/fi-prefill/parity_prefill.py \\
      --out /mnt/nvme1/fi-prefill/parity.json

It prints sglang.__file__ and flashinfer.__version__ first; read them (the interpreter trap in divix01-run-protocol.md).
Exit status is 0 only when every case passes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from unittest import mock

import torch

REL_L2_FLOOR = 2e-2
REL_L2_FACTOR = 1.5
LSE_ATOL = 2e-2
LSE_RTOL = 1e-3
CLOSE_ATOL = 5e-2
CLOSE_RTOL = 5e-2
REPS = 5
HEADS = 64
SWA_PBS = 256
SWA_WINDOW = 128
POOL_TOKENS = 1 << 18  # 262,144 SWA tokens: 1,024 pages of 256, ~150 MiB

# (name, tokens, extra pool page size or None, extra topk, sink, ragged)
CASES = [
    # the DSpark verify and decode: <= 64 tokens (FlashInfer decode), a 6-token verify being the served shape
    ("decode-t1-swa", 1, None, 0, True, False),
    ("verify-t6-c4", 6, 64, 512, True, True),
    ("verify-t6-c2", 6, 128, 512, True, True),
    ("verify-t6-c128", 6, 2, 32, True, True),
    ("decode-t64-c4", 64, 64, 512, True, True),
    # prefill: > 64 tokens (FlashInfer's prefill)
    ("prefill-t65-c4", 65, 64, 512, True, True),
    ("prefill-t4096-swa-only", 4096, None, 0, True, False),
    ("prefill-t4096-swa-nosink", 4096, None, 0, False, False),
    ("prefill-t4096-c4", 4096, 64, 512, True, True),
    ("prefill-t4096-c2", 4096, 128, 512, True, True),
    ("prefill-t4096-c128", 4096, 2, 32, True, True),
    ("prefill-t2065-c4", 2065, 64, 512, True, True),
]
PERF_CASES = ("prefill-t4096-swa-only", "prefill-t4096-c4", "prefill-t4096-c128")


def build_case(fmod, helpers, tokens, extra_pbs, extra_topk, sink, ragged, seed, device):
    build_kvcache, _ = helpers
    g = torch.Generator(device="cpu").manual_seed(seed)
    swa_pages = POOL_TOKENS // SWA_PBS
    k_cache, _ = build_kvcache(swa_pages, SWA_PBS, device=device, seed=seed)
    q = (
        torch.randn((tokens, 1, HEADS, fmod._D), generator=g, dtype=torch.float32).clamp(-1.5, 1.5).to(torch.bfloat16)
    ).to(device)

    def draw(pool_tokens, topk):
        # per-token distinct random valid indices; a rank-1 trick avoids tokens x randperm(pool) in Python
        idx = torch.randint(0, pool_tokens, (tokens, topk), generator=g, dtype=torch.int64)
        return idx.to(torch.int32)

    idx = draw(POOL_TOKENS, SWA_WINDOW)
    swa_len = torch.full((tokens,), SWA_WINDOW, dtype=torch.int32)
    if ragged:
        # early tokens see a short window, as in a prompt's first 128 tokens
        swa_len = torch.minimum(swa_len, torch.arange(1, tokens + 1, dtype=torch.int32))
        swa_len = torch.where(torch.rand(tokens, generator=g) < 0.25, torch.randint(1, SWA_WINDOW + 1, (tokens,), generator=g, dtype=torch.int32), swa_len)
    pos = torch.arange(SWA_WINDOW).unsqueeze(0)
    idx = torch.where(pos < swa_len.unsqueeze(1), idx, torch.full_like(idx, -1))
    kwargs = dict(
        q=q,
        k_cache=k_cache,
        indices=idx.unsqueeze(1).to(device),
        topk_length=swa_len.to(device),
        attn_sink=(torch.linspace(-1.0, 1.0, HEADS, dtype=torch.float32, device=device) if sink else None),
        head_dim_v=fmod._D,
        softmax_scale=fmod._D**-0.5,
    )
    if extra_pbs is not None:
        extra_pool = 1 << 15  # tokens
        extra_pages = extra_pool // extra_pbs
        extra_cache, _ = build_kvcache(extra_pages, extra_pbs, device=device, seed=seed + 1)
        eidx = draw(extra_pool, extra_topk)
        elen = torch.full((tokens,), extra_topk, dtype=torch.int32)
        if ragged:
            # compressed entries grow with the position: none for the first tokens (a lonely extra branch)
            elen = torch.minimum(elen, (torch.arange(tokens, dtype=torch.int32) * extra_topk // max(tokens, 1)))
        epos = torch.arange(extra_topk).unsqueeze(0)
        eidx = torch.where(epos < elen.unsqueeze(1), eidx, torch.full_like(eidx, -1))
        kwargs.update(
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=eidx.unsqueeze(1).to(device),
            extra_topk_length=elen.to(device),
        )
    return kwargs


def run_entry(fmod, backend, kwargs):
    with mock.patch.object(fmod, "_sm120_default_backend", backend):
        out, lse = fmod.flash_mla_with_kvcache_sm120(**kwargs)
    return out, lse


def run_reference(fmod, kwargs, chunk=256):
    """The fp32 PyTorch reference over token chunks (its gathered KV is tokens x topk x 512 floats)."""
    tokens = kwargs["q"].shape[0]
    outs, lses = [], []
    for s in range(0, tokens, chunk):
        e = min(s + chunk, tokens)
        kw = dict(
            q=kwargs["q"][s:e],
            k_cache=kwargs["k_cache"],
            indices=kwargs["indices"][s:e],
            topk_length=kwargs["topk_length"][s:e],
            attn_sink=kwargs["attn_sink"],
            head_dim_v=kwargs["head_dim_v"],
            softmax_scale=kwargs["softmax_scale"],
        )
        if "extra_k_cache" in kwargs:
            kw.update(
                extra_k_cache=kwargs["extra_k_cache"],
                extra_indices=kwargs["extra_indices_in_kvcache"][s:e],
                extra_topk_length=kwargs["extra_topk_length"][s:e],
            )
        out, lse = fmod._sm120_sparse_decode_fwd(**kw)
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, 0), torch.cat(lses, 0)


def flashinfer_lse(fmod, kwargs):
    """FlashInfer's out and lse through the dispatcher directly (the entry point drops lse)."""
    from flashinfer.mla._sparse_mla_sm120 import _DECODE_MAX_TOKENS, _sparse_mla_sm120_paged_attention

    q = kwargs["q"]
    q2 = q.squeeze(1)
    tokens, heads, _ = q2.shape
    dev = q.device
    idx = kwargs["indices"].squeeze(1)
    k = kwargs["k_cache"]
    kv_u8 = k.view(torch.uint8) if k.dtype != torch.uint8 else k
    kv_64 = fmod._split_kv_pages_to_64(kv_u8, k.shape[1], touched_indices=idx) if k.shape[1] != fmod._PBS_DST else kv_u8
    ek = kwargs.get("extra_k_cache")
    ek_u8 = ek.view(torch.uint8) if ek is not None and ek.dtype != torch.uint8 else ek
    eidx = kwargs.get("extra_indices_in_kvcache")
    eidx = eidx.squeeze(1) if eidx is not None else None
    ek_64 = fmod._extra_kv_to_64(ek_u8, eidx)
    out = torch.empty((tokens, heads, kwargs["head_dim_v"]), dtype=torch.bfloat16, device=dev)
    lse = torch.empty((tokens, heads), dtype=torch.float32, device=dev)
    mid_out = mid_lse = None
    extra = dict(d_v=kwargs["head_dim_v"])
    if tokens <= _DECODE_MAX_TOKENS:
        topk = idx.shape[-1]
        etopk = eidx.shape[-1] if eidx is not None else 0
        splits = (topk + 63) // 64 + ((etopk + 63) // 64 if etopk else 0)
        mid_out = torch.empty(tokens, heads, splits, kwargs["head_dim_v"], dtype=torch.bfloat16, device=dev)
        mid_lse = torch.empty(tokens, heads, splits, dtype=torch.float32, device=dev)
    else:
        extra = {}  # _flash_mla_sm120_prefill passes no d_v
    _sparse_mla_sm120_paged_attention(
        q2, kv_64, idx, out, lse, kwargs["softmax_scale"],
        topk_length=kwargs["topk_length"], attn_sink=kwargs["attn_sink"],
        extra_kv_cache=ek_64, extra_indices=eidx, extra_topk_length=kwargs.get("extra_topk_length"),
        mid_out=mid_out, mid_lse=mid_lse, **extra,
    )
    return out, lse


def rel_l2(x, ref, keep):
    x = x.float()[keep]
    ref = ref.float()[keep]
    return float((x - ref).norm() / ref.norm().clamp_min(1e-30))


def compare_case(fmod, name, kwargs):
    res = {"case": name, "tokens": int(kwargs["q"].shape[0]), "status": "PASS", "notes": []}
    ref_out, ref_lse = run_reference(fmod, kwargs)  # (T,1,H,D), (T,H,1)
    tri_out, tri_lse = run_entry(fmod, "triton", kwargs)
    fi_out, _ = run_entry(fmod, "flashinfer", kwargs)
    fi_out2, fi_lse = flashinfer_lse(fmod, kwargs)
    torch.cuda.synchronize()

    tokens = kwargs["q"].shape[0]
    have = kwargs["topk_length"] > 0
    if "extra_topk_length" in kwargs:
        have = have | (kwargs["extra_topk_length"] > 0)
    keep = have.view(tokens, 1, 1, 1).expand_as(ref_out)
    res["lonely"] = int((~have).sum())

    for tag, o in (("triton", tri_out), ("flashinfer", fi_out)):
        if not torch.isfinite(o.float()[keep]).all():
            res["status"] = "FAIL"
            res["notes"].append(f"{tag} out has non-finite values")
    e_tri = rel_l2(tri_out, ref_out, keep)
    e_fi = rel_l2(fi_out, ref_out, keep)
    res.update(relL2_triton=e_tri, relL2_flashinfer=e_fi, relL2_flashinfer_direct=rel_l2(fi_out2.unsqueeze(1), ref_out, keep))
    limit = max(REL_L2_FLOOR, REL_L2_FACTOR * e_tri)
    res["relL2_limit"] = limit
    if not e_fi <= limit:
        res["status"] = "FAIL"
        res["notes"].append(f"out relL2 flashinfer {e_fi:.3e} > limit {limit:.3e}")
    d = (fi_out.float() - tri_out.float()).abs()
    res["max_abs_fi_vs_triton"] = float(d[keep].max()) if keep.any() else 0.0
    tol = CLOSE_ATOL + CLOSE_RTOL * tri_out.float().abs()
    res["outside_5e-2_fi_vs_triton"] = int(((d > tol) & keep).sum())

    # lse, (T, H) each: the reference and Triton return (T, H, 1)
    r = ref_lse.reshape(tokens, HEADS)
    t = tri_lse.reshape(tokens, HEADS)
    f = fi_lse.reshape(tokens, HEADS)
    fin = torch.isfinite(r)
    res["lse_lonely_agree"] = bool((torch.isfinite(f) == fin).all())
    if fin.any():
        err_f = (f - r).abs()[fin]
        err_t = (t - r).abs()[fin]
        res["lse_max_abs_flashinfer"] = float(err_f.max())
        res["lse_max_abs_triton"] = float(err_t.max())
        bad = err_f > (LSE_ATOL + LSE_RTOL * r.abs()[fin])
        res["lse_outside_tol_flashinfer"] = int(bad.sum())
        if bad.any():
            res["status"] = "FAIL"
            res["notes"].append(f"lse flashinfer outside tol on {int(bad.sum())} entries")
    if not res["lse_lonely_agree"]:
        res["status"] = "FAIL"
        res["notes"].append("lse -inf (lonely row) pattern differs between flashinfer and the reference")
    return res


def measure(fmod, name, kwargs, device):
    """Median ms and peak extra allocation of one call per backend / merge budget."""
    rows = {}
    for label, backend, budget in (("triton-budget0", "triton", "0"), ("triton-budget128", "triton", "128"), ("flashinfer", "flashinfer", "0")):
        os.environ["SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB"] = budget
        run_entry(fmod, backend, kwargs)  # warm-up (JIT, split buffers)
        torch.cuda.synchronize()
        times = []
        peak_extra = 0
        for _ in range(REPS):
            torch.cuda.empty_cache()
            held = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            out, _ = run_entry(fmod, backend, kwargs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            peak_extra = max(peak_extra, torch.cuda.max_memory_allocated(device) - held)
            del out
        rows[label] = {"median_ms": statistics.median(times), "peak_extra_mib": peak_extra / 2**20}
    os.environ.pop("SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB", None)
    return {"case": name, "perf": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="JSON results path")
    ap.add_argument("--only", help="comma-separated case names")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    import flashinfer
    import sglang

    print("sglang", sglang.__file__)
    print("flashinfer", flashinfer.__version__, flashinfer.__file__)
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    if torch.cuda.get_device_capability(0) != (12, 0):
        print("not SM120")
        return 2

    from sglang.kernels.ops.attention import flash_mla_sm120 as fmod

    # the test file's synthetic cache builders, by path (test/ is not a package)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo, "test", "registered", "kernels", "ops", "attention"))
    import test_flash_mla_backends as t

    helpers = (t._build_kvcache, t._build_q_indices)
    device = torch.device("cuda")

    only = set(a.only.split(",")) if a.only else None
    results = []
    perf = []
    for i, (name, tokens, epbs, etopk, sink, ragged) in enumerate(CASES):
        if only and name not in only:
            continue
        print(f"== {name}", flush=True)
        try:
            kw = build_case(fmod, helpers, tokens, epbs, etopk, sink, ragged, a.seed + 10 * i, device)
            r = compare_case(fmod, name, kw)
            if not a.no_perf and name in PERF_CASES:
                perf.append(measure(fmod, name, kw, device))
        except Exception as exc:  # a case FlashInfer refuses is a finding, not a crash
            r = {"case": name, "status": "ERROR", "notes": [f"{type(exc).__name__}: {exc}"]}
        results.append(r)
        print(json.dumps(r, indent=1), flush=True)
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"sglang": sglang.__file__, "flashinfer": flashinfer.__version__, "cases": results, "perf": perf}, f, indent=1)
    bad = [r for r in results if r["status"] != "PASS"]
    print(f"{len(results) - len(bad)}/{len(results)} cases PASS" + (f"; not passing: {[r['case'] for r in bad]}" if bad else ""))
    for p in perf:
        print(p["case"], {k: (round(v["median_ms"], 2), round(v["peak_extra_mib"])) for k, v in p["perf"].items()}, "(ms, peak-extra MiB)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
