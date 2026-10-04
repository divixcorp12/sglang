"""Bit-exact A/B harness for the optimized EXL3 CPU expert kernel (csrc/exl3/optimized/kernel.cpp).

``dump`` runs a fixed set of forwards through the extension ``exl3_ext()`` builds and saves every output; ``compare``
checks two dumps for bitwise equality. ``--registration engines`` runs the C ABI's forward on two engines at once (the
first and second half of ``--cores``), exits 1 naming the case when they differ, and saves the first engine's outputs. A dump made at the merge-base is the reference a kernel refactor must reproduce
exactly, on every ISA tier the host can run. The kernel reads EXL3_MOE_CPU_MAX_ISA once, at its first forward or tier
query, so each tier is its own process.

Run on divix01 through run_exl3_cpu_forward_checks.sh, which sets the build environment.
"""

import argparse
import ctypes
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


def register_table(ext, s, limit):
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
        limit,
        0,
    )


def register_slabs(ext, s, limit):
    """The C ABI's register_layer over the same tensors, as the RAM-miss service registers them: six base pointers,
    slot s at base + s rows. Returns the handle and the trait that frees it."""
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    trait = Exl3CpuQuantTrait(ext, act_limit=limit)
    try:
        return trait.register_layer(s, CAP), trait
    except RuntimeError as error:
        sys.exit(f"register_layer refused the slabs: {error}")


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


def run_on_engine(torch, trait, handle, hidden, cases, engine, results, errors):
    """Every case through the C ABI's forward on ``engine``, on this thread: a forward pins its calling thread to the
    engine's worker 0 core, so each engine gets a thread of its own."""
    from sglang.srt.layers.moe.cpu_experts.pool import (
        CPU_EXPERTS_FORWARD_ABI_VERSION,
        CpuExpertForward,
        CpuExpertsForwardCall,
    )

    forward = CpuExpertForward(trait.native_forward())
    try:
        for case, tokens, k, x, sel, w in cases:
            slots = sel.to(torch.int32).contiguous()
            weights = w.float().contiguous()
            out = torch.full((tokens, hidden), float("nan"))
            call = CpuExpertsForwardCall(
                abi_version=CPU_EXPERTS_FORWARD_ABI_VERSION,
                rows=tokens,
                layer=handle,
                x=x.data_ptr(),
                slots=ctypes.cast(slots.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
                weights=ctypes.cast(weights.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                out=ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                k=k,
                threads=THREADS,
                accumulate=0,
                engine=engine,
            )
            status = forward(ctypes.byref(call))
            if status != 0:
                raise RuntimeError(f"{case}: the forward on engine {engine} returned {status}")
            results[engine][case] = out
    except Exception as error:  # noqa: BLE001 -- reported by the caller, naming the engine
        errors[engine] = error


def run_two_engines(torch, trait, handle, hidden, cases, engines):
    """Runs every case on both engines at once and returns engine A's outputs; exits 1 naming the first case the two
    engines disagree on."""
    results = {e: {} for e in engines}
    errors = {}
    threads = [
        threading.Thread(target=run_on_engine, args=(torch, trait, handle, hidden, cases, e, results, errors))
        for e in engines
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        sys.exit(f"an engine failed: {errors}")
    first, second = engines
    for case, _, _, _, _, _ in cases:
        if not torch.isfinite(results[first][case]).all():
            sys.exit(f"{case}: non-finite output")
        if not torch.equal(results[first][case], results[second][case]):
            sys.exit(f"{case}: engine A and engine B differ")
    return results[first]


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
    engines, engine_trait = None, None
    if args.registration == "engines":
        from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

        cores = parse_cores(args.cores)
        if len(cores) != 2 * THREADS:
            sys.exit(f"--cores must list {2 * THREADS} cores (two engines of {THREADS} workers), got {len(cores)}")
        engine_trait = Exl3CpuQuantTrait(ext, act_limit=LIMITS[0])
        engines = (
            engine_trait.native_create_engine(cores[:THREADS]),
            engine_trait.native_create_engine(cores[THREADS : 2 * THREADS]),
        )
    outputs = {}
    try:
        for (name, hidden, inter), limit in itertools.product(SHAPES, LIMITS):
            slabs = random_slabs(torch, hidden, inter, seed=hidden)
            if args.registration == "table":
                handle, trait = register_table(ext, slabs, limit), None
            else:
                handle, trait = register_slabs(ext, slabs, limit)
            try:
                cases = list(cases_for(torch, name, hidden, limit))
                if engines is not None:
                    outputs.update(run_two_engines(torch, trait, handle, hidden, cases, engines))
                    continue
                for case, tokens, k, x, sel, w in cases:
                    out = torch.full((tokens, hidden), float("nan"))
                    ext.exl3_moe_cpu_forward(handle, x, sel, w, out, THREADS)
                    if not torch.isfinite(out).all():
                        sys.exit(f"{case}: non-finite output")
                    outputs[case] = out
            finally:
                if trait is None:
                    ext.exl3_moe_cpu_free_layer(handle)
                else:
                    trait.free_layer(handle)
    finally:
        for engine in engines or ():
            engine_trait.native_free_engine(engine)
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
    d.add_argument("--registration", choices=("table", "slabs", "engines"), default="table")
    d.add_argument("--cores", help="engines: a taskset list of 2 * THREADS cores, one engine on each half")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("want")
    c.add_argument("got")
    args = parser.parse_args()
    if args.cmd == "dump" and args.registration == "engines" and not args.cores:
        parser.error(f"--registration engines needs --cores: a taskset list of {2 * THREADS} cores")
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
