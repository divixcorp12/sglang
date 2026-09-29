"""Attributes a HOTPATH_SHIM_STACKS dump's calls to call sites (final-fix round, hotpath-zero-overhead item 5).

Usage: attribute_stacks.py <stacks.txt> <shim.json> <out.json> <every> [frames]

Each record is symbolized through the dump's own maps (sglang.test.hotpath_shim.Symbolizer: the modules' PT_LOAD
segments and nm symbols, nm -D for a stripped module such as libcuda). A record's site is its first `frames` (default 6)
frames below the shim's own (hotpath_shim.so), innermost first. Per (thread, kind, site): `first` = records among the
first-N calls, `sampled` = records among the every-Nth calls; `share` = the site's share of the sampled records, which
estimates its share of the whole-run count when the count far exceeds N (est. calls = share * total); for a kind whose
count is at most N every call is a `first` record and the table is exact. Run it where the dump's modules live (the
JIT .so cache, libcuda) -- on divix01, right after the arm."""

import collections
import json
import sys

from sglang.test.hotpath_shim import Symbolizer, read_stacks


def main(argv):
    stacks, shim_json, out_json, every = argv[1], argv[2], argv[3], int(argv[4])
    depth = int(argv[5]) if len(argv) > 5 else 6
    records, maps = read_stacks(stacks)
    counts = json.loads(open(shim_json).read())
    sym = Symbolizer(maps)
    groups = collections.defaultdict(lambda: {"first": 0, "sampled": 0, "seqs": []})
    # The first-N records of a (thread, kind) fill slots 0..N-1 in call order; N is the highest seq < every observed.
    for r in records:
        frames = [sym(a) for a in r["frames"]]
        frames = [f for f in frames if not f.startswith("hotpath_shim.so")]
        site = " <- ".join(frames[:depth])
        g = groups[(r["thread"], r["kind"], site)]
        first = r["seq"] < 256  # HOTPATH_SHIM_STACKS_FIRST as the driver sets it
        g["first" if first else "sampled"] += 1
        if len(g["seqs"]) < 8:
            g["seqs"].append(r["seq"])
    per_kind = collections.Counter()
    for (th, kind, _), g in groups.items():
        per_kind[(th, kind)] += g["sampled"]
    rows = []
    for (th, kind, site), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1]["sampled"], -kv[1]["first"])):
        total = counts[th][kind]
        share = g["sampled"] / per_kind[(th, kind)] if per_kind[(th, kind)] else None
        rows.append({"thread": th, "kind": kind, "total": total, "site": site, "first": g["first"],
                     "sampled": g["sampled"], "share_of_sampled": share,
                     "est_calls": round(share * total) if share is not None and total > 256 else g["first"],
                     "example_seqs": g["seqs"]})
    json.dump({"counts": counts, "every": every, "depth": depth, "rows": rows}, open(out_json, "w"), indent=2)
    print(f"counts: {json.dumps({th: counts[th] for th in ('service', 'copy')})}")
    print("| thread | kind | total | first | sampled | est. calls | site (innermost first) |")
    print("|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r['thread']} | {r['kind']} | {r['total']} | {r['first']} | {r['sampled']} | {r['est_calls']} | "
              f"{r['site']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
