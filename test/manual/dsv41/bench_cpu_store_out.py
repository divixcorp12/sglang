"""Per-job cost of SGLANG_DSV41_CPU_EXPERTS_STORE_OUT: the split calibration's CPU cells, real EXL3 kernel.

The calibration times per-token jobs of a verify's 6 tokens through the CPU expert engine's run_job (each lane routed
by about 1.25 tokens, the served jobs' shape), so cpu[k] is the job the flag changes: with it off the engine zeroes 6
token rows of 5120 fp32 and the forward accumulates; with it on the kernel stores. Arms alternate in one process, a
fresh host per round. Prints the mean ms of cpu[1..lanes] per arm and round, and the per-k difference.

Needs SGLANG_EXL3_SRC, SGLANG_DSV41_CPU_EXPERTS=1, the bw tier and a quiet box. Run on divix01, e.g.
  numactl --cpunodebind=0 --membind=0 python test/manual/dsv41/bench_cpu_store_out.py --cores 6-15 --rounds 6
"""

import argparse
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))

from test_exl3_cpu_dsv41_plan_multitoken import H, _slabs  # noqa: E402

ROW, ROWS, TOKENS, LIMIT, CAP = 1, 2, 6, 10.0, 20


def _cores(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _arm(trait, slabs, cores, *, store_out, lanes, reps, tmp):
    from sglang.kernels.ops.moe import expert_lease_block as lease
    from sglang.kernels.ops.moe.expert_stream_transport import new_page
    from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

    s = ram_miss_setup(Path(tmp), capacity=CAP, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, new_page(pin=False, wire=lease.wire_layout(16)), k=3, lanes=16)
    try:
        host.enable_copy_engine(-1)
        dst = {n: torch.zeros((6,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
        table = torch.tensor(
            [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
            dtype=torch.int64,
        )
        host.set_copy_table(ROW, table, 6)
        x_rows = torch.zeros((ROWS, lease.cpu_row_bytes(H, TOKENS, host.wire.lanes)), dtype=torch.uint8).pin_memory()
        x_rows.view(torch.float16)[:, : TOKENS * H].normal_()  # the staged inputs; the token table follows them
        out_rows = torch.zeros((ROWS, 2, TOKENS, H), dtype=torch.float32).pin_memory()
        host.enable_cpu_experts(
            trait.kernel_address(), [0] * (host.wire.lanes + 1), cores, x_rows, out_rows, threads=len(cores),
            keep_warm_us=2000, store_out=store_out,
        )
        host.set_cpu_layer(ROW, trait.layer_spec(slabs, CAP))
        scratch = torch.zeros(lanes * host.copy_expert_bytes(ROW), dtype=torch.uint8)
        grid = host.calibrate_cpu_split(ROW, device=-1, reps=reps, scratch=scratch, lanes=lanes, tokens=TOKENS,
                                        timeout_s=5.0)
        return grid[0, 1 : lanes + 1].tolist()
    finally:
        host.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", default="6-15")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--lanes", type=int, default=4)
    args = ap.parse_args()
    from sglang.srt.layers.quantization.exl3.ext import exl3_ext
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    import sglang

    print("sglang", sglang.__file__)
    trait = Exl3CpuQuantTrait(exl3_ext(), act_limit=LIMIT)
    slabs = {n: t.pin_memory() for n, t in _slabs(20261015).items()}  # CAP=12 slots: extended below
    slabs = {n: torch.cat([t, t[: CAP - t.shape[0]]]).pin_memory() for n, t in slabs.items()}
    cores = _cores(args.cores)
    ms = {False: [], True: []}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        for r in range(args.rounds):
            for store_out in ((False, True) if r % 2 == 0 else (True, False)):
                sub = os.path.join(tmp, f"r{r}-{int(store_out)}")
                os.makedirs(sub)
                cells = _arm(trait, slabs, cores, store_out=store_out, lanes=args.lanes, reps=args.reps, tmp=sub)
                ms[store_out].append(cells)
                print(f"round {r} store_out={int(store_out)} cpu[1..{args.lanes}] ms:",
                      " ".join(f"{v:.4f}" for v in cells), flush=True)
    for k in range(args.lanes):
        off = statistics.median(c[k] for c in ms[False])
        on = statistics.median(c[k] for c in ms[True])
        print(f"k={k + 1}: median off {off * 1e3:.1f} us, on {on * 1e3:.1f} us, saved {(off - on) * 1e3:.1f} us/job")


if __name__ == "__main__":
    main()
