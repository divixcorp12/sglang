"""The oracle's head-grouped `sparse_attn` against a float32 torch version of the kernel's math.

The reference tilelang kernel cannot launch on sm_120 at the released 64 heads x 512 (141312 B
of shared memory); scripts/dsv41/ref_oracle.py runs it 16 heads at a time instead.
"""

import os
import sys

import pytest
import torch

SNAPSHOT = os.environ.get(
    "DSV41_SNAPSHOT",
    "/mnt/nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
    "dba1be0a40aa45a94ad051997016db3960a90277",
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(os.path.join(SNAPSHOT, "inference"))),
    reason="needs a GPU and the official DeepSeek-V4.1-Flash snapshot",
)


def _torch_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    # o[b,m,h] = sum_j p_j v_j / (sum_j p_j + exp(sink_h - max)), p_j = exp(s_j - max) over the
    # valid (!= -1) indices; the sink enters the denominator only.
    qf, kvf = q.float(), kv.float()
    idx = topk_idxs.long()
    valid = idx >= 0
    g = torch.stack([kvf[b][idx[b].clamp(min=0)] for b in range(kv.size(0))])  # [b,m,k,d]
    s = torch.einsum("bmhd,bmkd->bmhk", qf, g) * scale
    s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
    m = s.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(s - m)
    den = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float()[None, None, :, None] - m)
    return torch.einsum("bmhk,bmkd->bmhd", p, g) / den


def test_head_split_matches_torch():
    sys.path.insert(0, os.path.join(SNAPSHOT, "inference"))
    sys.path.insert(0, os.path.join(ROOT, "scripts", "dsv41"))
    import kernel
    from ref_oracle import make_head_split_sparse_attn

    torch.manual_seed(0)
    b, m, h, d, n, topk = 1, 6, 64, 512, 300, 192
    q = torch.randn(b, m, h, d, device="cuda").bfloat16()
    kv = torch.randn(b, n, d, device="cuda").bfloat16()
    sink = torch.randn(h, device="cuda")
    idx = torch.randint(0, n, (b, m, topk), device="cuda", dtype=torch.int32)
    idx[:, :, -40:] = -1  # padded slots
    idx[:, 0, :] = -1  # a row with no valid key -> all-zero output
    scale = d**-0.5

    got = make_head_split_sparse_attn(kernel)(q, kv, sink, idx, scale).float()
    ref = _torch_sparse_attn(q, kv, sink, idx, scale)
    assert got.shape == (b, m, h, d)
    assert torch.all(got[:, 0] == 0)
    err = (got - ref).abs().max() / ref.abs().max()
    print(f"max err / max |ref| = {err:.3e}")
    assert err < 2e-2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
