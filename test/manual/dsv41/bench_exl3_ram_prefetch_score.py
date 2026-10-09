"""The GPU scorer's kernels at DSV4's shape (384 experts, hidden 7168, bf16 gate), per launch.

Each launch scores a different one of 40 gates (220 MB, past the 96 MiB L2), as decode scores a different layer's gate
after each post. Times are CUDA-event averages over back-to-back launches, so they include no host gap.

Run on divix01 under cc-gpu.lock from the repository root, with PYTHONPATH pointing at the tree under test.
"""

import argparse

import torch

from sglang.kernels.ops.moe import expert_stream_transport as es

STATE_POSTED = es.STATE_WORDS["posted"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--iters", type=int, default=400)
    a = p.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    gates = [
        torch.randn((a.experts, a.hidden), generator=g, device="cuda").mul_(0.02).to(torch.bfloat16)
        for _ in range(a.layers)
    ]
    bias = torch.randn(a.experts, generator=g, device="cuda") * 0.1
    scores = torch.empty((32, a.experts), dtype=torch.float32, device="cuda")
    page = es.new_candidate_page(pin=True)
    state = torch.zeros(len(es.STATE_WORDS), dtype=torch.int32, device="cuda")
    state[STATE_POSTED] = 1
    hot = torch.full((16,), -1, dtype=torch.int64, device="cuda")
    ram_slot = torch.full((2, a.experts), -1, dtype=torch.int32, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for tokens in (1, 3, 6):
        x = torch.randn((tokens, a.hidden), generator=g, device="cuda").to(torch.bfloat16)

        def score(i):
            es.run_spec_score(x, gates[i % a.layers], bias, scores)

        def select(i):
            es.run_spec_select(
                scores, tokens, top_k=6, per_token=1, top_k_only=False, hot_slots=hot, hot_capacity=16,
                ram_slot=ram_slot, target=1, state=state, candidates=page,
            )

        for name, fn in (("score", score), ("select", select)):
            for i in range(20):
                fn(i)
            torch.cuda.synchronize()
            start.record()
            for i in range(a.iters):
                fn(i)
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000 / a.iters
            gbps = a.experts * a.hidden * 2 / (us * 1e-6) / 1e9 if name == "score" else 0.0
            print(f"tokens {tokens} {name:6s} {us:8.2f} us/launch" + (f"  {gbps:7.1f} GB/s of gate" if gbps else ""))


if __name__ == "__main__":
    main()
