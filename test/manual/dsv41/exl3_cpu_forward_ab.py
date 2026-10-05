"""Bit-exact A/B harness for the optimized EXL3 CPU expert kernel (csrc/exl3/optimized/kernel.cpp).

``dump`` runs a fixed set of forwards through the extension ``exl3_ext()`` builds, on slab layers made by its kernel's
make_layer as the RAM-miss service makes them, and saves every output; ``compare`` checks two dumps for bitwise
equality. (Revisions before 2026-10-05 also took ``--registration table``, upstream's per-expert tensor layers, which
run the same plans: a ``table`` dump there equals its ``slabs`` dump.) ``--registration cores`` (``engines`` is accepted
as an alias) runs the host's ``kernel_forward`` on two core groups at once (the first and second half of ``--cores``),
exits 1 naming the case when they differ, and saves the first group's outputs. A dump made at the merge-base is the
reference a kernel refactor must reproduce exactly, on every ISA tier the host can run. The kernel reads
EXL3_MOE_CPU_MAX_ISA once, at its first forward or tier query, so each tier is its own process.

Run on divix01 through run_exl3_cpu_forward_checks.sh, which sets the build environment.
"""

import argparse
import itertools
import os
import sys
import threading

# (name, hidden, intermediate): a generic shape, and DeepSeek V4.1's, which takes the DSV4.1 plan on AVX-512BW.
SHAPES = (("generic", 512, 256), ("dsv41", 5120, 2304))
CAP = 6
# Activation limits: DeepSeek V4.1's swiglu_limit (10), which its plan requires, and another, which takes the
# generic plan at the DSV4.1 shape.
LIMITS = (10.0, 7.0)
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


def register_slabs(ext, s, limit):
    """The kernel's make_layer over the same tensors, as the RAM-miss service makes it: six base pointers, slot s at
    base + s rows. Returns the layer handle (freed by ``kernel_drop``) and the trait."""
    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    trait = Exl3CpuQuantTrait(ext, act_limit=limit)
    try:
        return es.kernel_layer(trait.kernel_address(), trait.layer_spec(s, CAP), variant="instr"), trait
    except RuntimeError as error:
        sys.exit(f"kernel_layer refused the slabs: {error}")


def cases_for(torch, name, hidden, limit):
    """(case, tokens, k, x, sel, w) in the order the dump's RNG draws them: one generator per shape and limit."""
    g = torch.Generator().manual_seed(7)
    for (tokens, k), route in ROUTES.items():
        sel = torch.tensor(route, dtype=torch.int64)
        w = torch.tensor([list(WEIGHTS[:k])] * tokens).half()
        for scale in SCALES:
            x = (torch.randn(tokens, hidden, generator=g) * scale).half()
            yield f"{name}/l{limit:g}/t{tokens}k{k}/s{scale}", tokens, k, x, sel, w


def parse_cores(text):
    """A taskset list such as "18-25" or "18-21,26"."""
    cores = []
    for item in text.split(","):
        first, _, last = item.partition("-")
        cores += range(int(first), int(last or first) + 1)
    return cores


def kernel_forward(torch, handle, hidden, case, tokens, k, x, sel, w, cores=()):
    """One case through the host's ``kernel_forward``; exits naming the case unless it returns 0."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    out = torch.full((tokens, hidden), float("nan"))
    status, why = es.kernel_forward(
        handle, x, sel.to(torch.int32).contiguous(), w.float().contiguous(), out, threads=THREADS, cores=cores,
        variant="instr",
    )
    if status != 0:
        sys.exit(f"{case}: kernel_forward returned {status}: {why}")
    return out


def run_on_cores(torch, handle, hidden, cases, group, cores, results, errors):
    """Every case through ``kernel_forward`` on ``cores``, on this thread: a forward pins its calling thread to its
    worker 0 core, so each core group gets a thread of its own."""
    try:
        for case, tokens, k, x, sel, w in cases:
            results[group][case] = kernel_forward(torch, handle, hidden, case, tokens, k, x, sel, w, cores)
    except BaseException as error:  # noqa: BLE001 -- reported by the caller, naming the group (a failed case exits)
        errors[group] = error


def run_two_core_groups(torch, handle, hidden, cases, halves):
    """Runs every case on both core groups at once and returns group 0's outputs; exits 1 naming the first case the
    two groups disagree on."""
    results = {g: {} for g in range(2)}
    errors = {}
    threads = [
        threading.Thread(target=run_on_cores, args=(torch, handle, hidden, cases, g, halves[g], results, errors))
        for g in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        sys.exit(f"a core group failed: {errors}")
    for case, _, _, _, _, _ in cases:
        if not torch.isfinite(results[0][case]).all():
            sys.exit(f"{case}: non-finite output")
        if not torch.equal(results[0][case], results[1][case]):
            sys.exit(f"{case}: core group A and core group B differ")
    return results[0]


def dump(args):
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = args.isa  # read once, at the kernel's first use
    os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")
    import torch

    import sglang
    from sglang.srt.layers.quantization.exl3.ext import exl3_ext

    print(f"sglang {sglang.__file__}")
    ext = exl3_ext()
    tier = (bool(ext.exl3_moe_cpu_has_avx2()), bool(ext.exl3_moe_cpu_has_avx512_bw()))
    if tier != TIERS[args.isa]:
        sys.exit(f"asked for {args.isa}, the kernel reports avx2={tier[0]} bw={tier[1]}")
    torch.set_num_threads(1)
    halves = None
    if args.registration == "cores":
        cores = parse_cores(args.cores)
        if len(cores) != 2 * THREADS:
            sys.exit(f"--cores must list {2 * THREADS} cores (two groups of {THREADS} workers), got {len(cores)}")
        halves = (cores[:THREADS], cores[THREADS : 2 * THREADS])
    outputs = {}
    for (name, hidden, inter), limit in itertools.product(SHAPES, LIMITS):
        slabs = random_slabs(torch, hidden, inter, seed=hidden)
        handle, _ = register_slabs(ext, slabs, limit)
        try:
            cases = list(cases_for(torch, name, hidden, limit))
            if halves is not None:
                outputs.update(run_two_core_groups(torch, handle, hidden, cases, halves))
                continue
            for case, tokens, k, x, sel, w in cases:
                out = kernel_forward(torch, handle, hidden, case, tokens, k, x, sel, w)
                if not torch.isfinite(out).all():
                    sys.exit(f"{case}: non-finite output")
                outputs[case] = out
        finally:
            from sglang.kernels.ops.moe import expert_stream_transport as es

            es.kernel_drop(handle, variant="instr")
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
    d.add_argument("--registration", choices=("slabs", "cores", "engines"), default="slabs")
    d.add_argument("--cores", help="cores: a taskset list of 2 * THREADS cores, one core group on each half")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("want")
    c.add_argument("got")
    args = parser.parse_args()
    if args.cmd == "dump" and args.registration == "engines":
        args.registration = "cores"  # the old name, accepted so the check script's history reads
    if args.cmd == "dump" and args.registration == "cores" and not args.cores:
        parser.error(f"--registration cores needs --cores: a taskset list of {2 * THREADS} cores")
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
