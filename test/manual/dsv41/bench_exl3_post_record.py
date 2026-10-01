"""Eager post-kernel cost with SM-hit lanes only: records of SM hits lap the ring, so nothing has to serve them.

--mode hits: the row's delta is already applied, so the post reads only the delta's tag.
--mode delta: before each post the row's applied word is reset, so every post re-applies the same full delta
(DELTA_MAX_ENTRIES entries that leave the map as it is). This is the path a post takes after any miss.

Run on divix01 under cc-gpu.lock from test/manual/dsv41, with PYTHONPATH pointing at the tree under test. For the
kernel's own duration, run it under nsys and read exl3_ram_miss_post_kernel's row in cuda_gpu_kern_sum.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from lease_chain_rig import CAPACITY, EXPERTS, TOP_K, Chain  # noqa: E402

import sglang  # noqa: E402
from sglang.kernels.ops.moe import expert_lease_block as lease  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402


def write_delta(block: torch.Tensor, row: int, tag: int, staging, entries) -> None:
    """The host's delta publication, done here: payload, then the tag (x86 keeps tensor stores in order)."""
    base = lease.DELTA_BASE + row * lease.DELTA_STRIDE
    f = lease.DELTA_FIELDS
    flat = [v for pair in entries for v in pair]
    block[base + f["count"] : base + f["count"] + 4].view(torch.int32)[0] = len(entries)
    block[base + f["staging"] : base + f["staging"] + 2 * lease.LANES].view(torch.int16)[:] = torch.tensor(
        staging, dtype=torch.int16)
    block[base + f["entries"] : base + f["entries"] + 2 * len(flat)].view(torch.int16)[:] = torch.tensor(
        flat, dtype=torch.int16)
    block[base + f["tag"] : base + f["tag"] + 8].view(torch.int64)[0] = tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hits", "delta"], default="hits")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=2000)
    args = parser.parse_args()
    print("sglang:", sglang.__file__)
    with tempfile.TemporaryDirectory() as tmp:
        c = Chain(Path(tmp), start=False)
        try:
            row = 0
            experts = list(range(TOP_K))
            c.dev.map_bulk_apply(torch.tensor([[row, e, e] for e in experts], dtype=torch.int32))
            c.plan(experts, row)
            backend, plan = c.backends[row], c.plans[row]
            backend._stage_planned(plan)
            applied = c.dev.map_bank["map_applied"]
            if args.mode == "delta":
                torch.cuda.synchronize()
                tag = int(c.dev.map_bank["map_chain"][row])
                staging = c.dev.map_bank["staging"][row].tolist()
                entries = [(e, e) for e in range(CAPACITY)] + [(e, -1) for e in range(CAPACITY, EXPERTS)]
                assert len(entries) == lease.DELTA_MAX_ENTRIES
                write_delta(c.host.lease_block, row, tag, staging, entries)

            def post() -> None:
                if args.mode == "delta":
                    applied[row] = 0
                c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots)

            for _ in range(args.warmup):
                post()
            torch.cuda.synchronize()
            assert c.kinds(TOP_K) == [LaneKind.HIT_SM] * TOP_K, c.kinds(TOP_K)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                post()
            end.record()
            end.synchronize()
            print(f"post ({args.mode}): {start.elapsed_time(end) * 1000 / args.iters:.2f} us/post over {args.iters} "
                  f"({TOP_K} SM-hit lanes)")
        finally:
            c.close()


if __name__ == "__main__":
    main()
