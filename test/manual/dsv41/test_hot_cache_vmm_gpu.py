"""VmmHotCacheBacking on the GPU: tensors on per-tensor VMM reservations, and release_tail frees trailing granules.

Checks that the tensors take the requested shapes and dtypes and are writable; that after release_tail the kept rows
keep their addresses and contents; that cudaMemGetInfo's free memory rises by the released bytes; and that the same
holds in a child process with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True while torch allocates around it (two
VMM users in one process). Small sizes: a few hundred MiB at most.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SPECS = [("w13", 26, (2, 320, 144, 48), torch.int16), ("suh", 26, (2, 5120), torch.float16)]


def _check(keep: int) -> dict:
    from sglang.srt.layers.moe.hot_cache_vmm import VmmHotCacheBacking, tail_release

    dev = torch.device("cuda", torch.cuda.current_device())
    backing = VmmHotCacheBacking(dev, SPECS)
    out = {}
    for name, rows, shape, dtype in SPECS:
        t = backing.tensors[name]
        assert t.shape == (rows,) + shape and t.dtype == dtype and t.is_cuda
        for i in range(rows):
            t[i].fill_(i + 1)
    before = {n: (t.data_ptr(), t[:keep].clone()) for n, t in backing.tensors.items()}
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    freed = backing.release_tail(keep)
    free1 = torch.cuda.mem_get_info()[0]
    expect = sum(
        tail_release(row_bytes=backing.row_bytes[n], rows=r, keep_rows=keep, granularity=backing.granularity).nbytes
        for n, r, _, _ in SPECS
    )
    assert freed == expect > 0
    for n, t in backing.tensors.items():
        assert t.data_ptr() == before[n][0]
        assert torch.equal(t[:keep], before[n][1])  # kept rows still backed, contents intact
    out["freed"], out["free_rise"] = freed, free1 - free0
    backing.close()
    return out


def test_release_frees_trailing_granules_and_keeps_the_rest():
    out = _check(keep=20)
    # Free memory rises by the released granules (other processes may move it a little).
    assert abs(out["free_rise"] - out["freed"]) <= 64 << 20, out


def test_works_beside_expandable_segments():
    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from test_hot_cache_vmm_gpu import _check
        pad = [torch.empty(64 << 20, dtype=torch.uint8, device="cuda") for _ in range(3)]
        out = _check(keep=13)
        more = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        more.fill_(1); torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["num_alloc_retries"] == 0
        print("OK", out)
        """
    )
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
