"""Splits a HOTPATH_SHIM_STACKS dump's copy-thread calls by the libcuda entry point that made them (phase 2 Task P1).

Usage: attribute_entry.py <callsites.txt> <shim.json> <out.json> <every> [first]

final-fix/attribute_stacks.py groups records by their innermost frames, which inside the stripped libcuda are its
internal functions (named after the nearest exported symbol, so one entry point shows up as many "sites"). This script
instead walks each record outward to the first frame in the host module (HOST) and classifies the record by it:

  CudaCopyBackend::issue    -> cuMemcpyAsync   (steady state, per copy)
  CudaCopyBackend::mark     -> cuEventRecord   (steady state, once per job)
  CudaCopyBackend::query    -> cuEventQuery    (steady state, completion polling)
  CudaCopyBackend::init     -> init            (start-up)
  CudaCopyBackend::shutdown -> shutdown        (exit)
  another HOST function whose callee is an exported cu* entry (a devirtualized, inlined backend call: at -O2
  CudaCopyBackend::issue and ::mark are inlined into CopyEngine::run) -> that entry point's class
  any other HOST function   -> "other: <function>" (e.g. CopyEngine::run's start handshake)
  no HOST frame             -> "other: no host frame" (thread start/exit, libcuda TLS destructors)

The copy thread makes no cuLaunchKernel call; a record under one would be classified "launch" (and none is).

Estimates: the records among the first `first` calls (default 256) are exact. Past them, every `every`-th call is
recorded; a class's share of those sampled records times the kind's whole-run count estimates its steady-state calls,
with a 95% Wilson interval on the share. Run it where the dump's modules live (divix01, the JIT cache and libcuda)."""

import collections
import json
import math
import re
import sys

from sglang.test.hotpath_shim import Symbolizer, read_stacks

CLASSES = {
    "issue": "cuMemcpyAsync",
    "mark": "cuEventRecord",
    "query": "cuEventQuery",
    "init": "init",
    "shutdown": "shutdown",
}


ENTRY = {"cuMemcpyAsync": "cuMemcpyAsync", "cuEventRecord": "cuEventRecord", "cuEventQuery": "cuEventQuery",
         "cuLaunchKernel": "launch"}


def classify(frames: list[str]) -> tuple[str, str]:
    """(class, outermost libcuda frame below the host frame)."""
    last_cuda = ""
    for f in frames:
        if f.startswith("hotpath_shim.so"):
            continue
        if f.startswith("libcuda"):
            last_cuda = re.sub(r"\+0x[0-9a-f]+$", "", f)
            continue
        if f.startswith("sgl_kernel_jit_expert_stream_host") or "expert_stream_host" in f.split("!")[0]:
            m = re.search(r"CudaCopyBackend::(\w+)(\[abi:cxx11\])?\(", f)
            if m:
                return CLASSES.get(m.group(1), "other: CudaCopyBackend::" + m.group(1)), last_cuda
            # A backend call the compiler devirtualized and inlined into its caller (issue/mark inside
            # CopyEngine::run): the outermost libcuda frame is then the exported entry point the host called.
            e = re.match(r"libcuda[^!]*!(cu\w+?)(_v\d+)?$", last_cuda)
            if e:
                return ENTRY.get(e.group(1), "other entry: " + e.group(1)), last_cuda
            if "launch" in f.lower() and "Kernel" in f:
                return "launch", last_cuda
            fn = f.split("!", 1)[-1]
            fn = re.sub(r"\(.*", "", fn)
            fn = re.sub(r"sglang::expert_stream::", "", fn)
            while True:
                shorter = re.sub(r"<[^<>]*>", "", fn)
                if shorter == fn:
                    break
                fn = shorter
            return "other: " + fn, last_cuda
    return "other: no host frame", last_cuda


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def main(argv):
    stacks, shim_json, out_json, every = argv[1], argv[2], argv[3], int(argv[4])
    first_n = int(argv[5]) if len(argv) > 5 else 256
    records, maps = read_stacks(stacks)
    counts = json.loads(open(shim_json).read())
    sym = Symbolizer(maps)
    groups = collections.defaultdict(lambda: {"first": 0, "sampled": 0, "cuda_frames": collections.Counter()})
    for r in records:
        if r["thread"] != "copy":
            continue
        cls, cuda = classify([sym(a) for a in r["frames"]])
        g = groups[(r["kind"], cls)]
        g["first" if r["seq"] < first_n else "sampled"] += 1
        g["cuda_frames"][cuda] += 1
    sampled = collections.Counter()
    for (kind, _), g in groups.items():
        sampled[kind] += g["sampled"]
    rows = []
    for (kind, cls), g in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[1]["sampled"], -kv[1]["first"])):
        total = counts["copy"][kind]
        n = sampled[kind]
        share = g["sampled"] / n if n else None
        lo, hi = wilson(g["sampled"], n)
        rows.append({
            "kind": kind, "class": cls, "total": total, "first": g["first"], "sampled": g["sampled"],
            "sampled_of_kind": n, "share": share,
            "est_calls": round(share * total) if share is not None and total > first_n else g["first"],
            "est_lo": round(lo * total) if n and total > first_n else g["first"],
            "est_hi": round(hi * total) if n and total > first_n else g["first"],
            "outermost_libcuda_frames": dict(g["cuda_frames"].most_common(4)),
        })
    json.dump({"counts": counts, "every": every, "first": first_n, "rows": rows}, open(out_json, "w"), indent=2)
    print(f"copy-thread counts: {json.dumps(counts['copy'])}")
    print("| kind | class | first (exact) | sampled | share | est. calls [95% CI] |")
    print("|---|---|---:|---:|---:|---|")
    for r in rows:
        share = "--" if r["share"] is None else f"{r['share']:.4f}"
        print(f"| {r['kind']} | {r['class']} | {r['first']} | {r['sampled']}/{r['sampled_of_kind']} | {share} | "
              f"{r['est_calls']:,} [{r['est_lo']:,}, {r['est_hi']:,}] |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
