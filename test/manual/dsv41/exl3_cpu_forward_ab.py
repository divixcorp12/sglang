"""Bit-exact A/B harness for the optimized EXL3 CPU expert kernel (exl3_cpu/optimized/moe_mul1.cpp).

``dump`` runs a fixed set of forwards through the extension ``exl3_ext()`` builds and saves every output; ``compare``
checks two dumps for bitwise equality. A dump made at the merge-base is the reference a kernel refactor must reproduce
exactly, on every ISA tier the host can run. The kernel reads EXL3_MOE_CPU_MAX_ISA once, at load, so each tier is its
own process.

Run on divix01 through run_exl3_cpu_forward_checks.sh, which sets the build environment.
"""

import argparse
import ctypes
import os
import sys

# (name, hidden, intermediate): a generic shape, and DeepSeek V4.1's, which takes the DSV4.1 plan on AVX-512BW.
SHAPES = (("generic", 512, 256), ("dsv41", 5120, 2304))
CAP = 6
LIMIT = 10.0
THREADS = 4
# (tokens, experts per token) -> routes. Two tokens sharing expert 0 and 4 make m=2 chunks, which the DSV4.1 plan
# refuses: at the DSV4.1 shape that case runs the generic plan.
ROUTES = {
    (1, 1): [[2]],
    (1, 3): [[4, 0, 5]],
    (1, 5): [[1, 3, 0, 5, 2]],
    (2, 3): [[4, 0, 5], [0, 2, 4]],
}
WEIGHTS = (0.5, 0.3, 0.2, 0.15, 0.1)
SCALES = (1.0, 8.0)
# exl3_expert_format.EXL3_STREAMED_NAMES, the slab order of sglang_exl3_cpu_experts_register_slabs.
NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
TIERS = {"scalar": (False, False), "avx2": (True, False), "bw": (True, True)}


def random_slabs(torch, hidden, inter, seed):
    """The pinned tier's slab rows ([CAP, parts, ...], 3-bit), filled with random codes and signs."""
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    def codes(*shape):
        return torch.randint(-32768, 32767, shape, generator=g, dtype=torch.int16)

    return {
        "w13_trellis": codes(CAP, 2, hidden // 16, inter // 16, 48),
        "w13_suh": signs(CAP, 2, hidden),
        "w13_svh": signs(CAP, 2, inter),
        "w2_trellis": codes(CAP, 1, inter // 16, hidden // 16, 48),
        "w2_suh": signs(CAP, 1, inter),
        "w2_svh": signs(CAP, 1, hidden),
    }


def register_table(ext, s):
    """make_layer over one view per slot: gate is w13 part 0, up part 1, down w2 part 0."""
    rows = range(CAP)
    return ext.exl3_moe_cpu_make_layer(
        [s["w13_trellis"][i, 0] for i in rows],
        [s["w13_suh"][i, 0] for i in rows],
        [s["w13_svh"][i, 0] for i in rows],
        [s["w13_trellis"][i, 1] for i in rows],
        [s["w13_suh"][i, 1] for i in rows],
        [s["w13_svh"][i, 1] for i in rows],
        [s["w2_trellis"][i, 0] for i in rows],
        [s["w2_suh"][i, 0] for i in rows],
        [s["w2_svh"][i, 0] for i in rows],
        [],
        [],
        [],
        0,
        LIMIT,
        0,
    )


def register_slabs(ext, s, hidden, inter):
    """The slab ABI over the same tensors: six base pointers, slot s at base + s rows."""
    fn = ctypes.CDLL(ext.__file__).sglang_exl3_cpu_experts_register_slabs
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int64),
    ]
    bases = (ctypes.c_void_p * len(NAMES))(*(s[n].data_ptr() for n in NAMES))
    handle = ctypes.c_int64(-1)
    status = fn(bases, CAP, hidden, inter, 3, 0, LIMIT, ctypes.byref(handle))
    if status != 0:
        sys.exit(f"register_slabs refused the {hidden}x{inter} slabs: status {status}")
    return handle.value


def dump(args):
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = args.isa  # read at the kernel's load: before the extension imports
    os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")
    import torch

    import sglang
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    print(f"sglang {sglang.__file__}")
    ext = exl3_ext()
    tier = (bool(ext.exl3_moe_cpu_has_avx2()), bool(ext.exl3_moe_cpu_has_avx512_bw()))
    if tier != TIERS[args.isa]:
        sys.exit(f"asked for {args.isa}, the kernel reports avx2={tier[0]} bw={tier[1]}")
    torch.set_num_threads(1)
    outputs = {}
    for name, hidden, inter in SHAPES:
        slabs = random_slabs(torch, hidden, inter, seed=hidden)
        if args.registration == "table":
            handle = register_table(ext, slabs)
        else:
            handle = register_slabs(ext, slabs, hidden, inter)
        try:
            g = torch.Generator().manual_seed(7)
            for (tokens, k), route in ROUTES.items():
                sel = torch.tensor(route, dtype=torch.int64)
                w = torch.tensor([list(WEIGHTS[:k])] * tokens).half()
                for scale in SCALES:
                    x = (torch.randn(tokens, hidden, generator=g) * scale).half()
                    out = torch.full((tokens, hidden), float("nan"))
                    ext.exl3_moe_cpu_forward(handle, x, sel, w, out, THREADS)
                    case = f"{name}/t{tokens}k{k}/s{scale}"
                    if not torch.isfinite(out).all():
                        sys.exit(f"{case}: non-finite output")
                    outputs[case] = out
        finally:
            ext.exl3_moe_cpu_free_layer(handle)
    torch.save({"isa": args.isa, "registration": args.registration, "outputs": outputs}, args.out)
    print(f"{len(outputs)} outputs ({args.isa}, {args.registration}) -> {args.out}")


def compare(args):
    import torch

    want, got = torch.load(args.want), torch.load(args.got)
    if want["outputs"].keys() != got["outputs"].keys():
        sys.exit(f"the dumps hold different cases: {sorted(want['outputs'])} vs {sorted(got['outputs'])}")
    bad = [c for c in want["outputs"] if not torch.equal(want["outputs"][c], got["outputs"][c])]
    for c in bad:
        diff = (want["outputs"][c] - got["outputs"][c]).abs().max().item()
        print(f"MISMATCH {c}: max |diff| {diff:.3e}")
    total = len(want["outputs"])
    print(
        f"{total - len(bad)}/{total} bit-exact: {want['isa']}/{want['registration']} vs "
        f"{got['isa']}/{got['registration']}"
    )
    sys.exit(1 if bad else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--isa", choices=sorted(TIERS), required=True)
    d.add_argument("--registration", choices=("table", "slabs"), default="table")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("want")
    c.add_argument("got")
    args = parser.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
