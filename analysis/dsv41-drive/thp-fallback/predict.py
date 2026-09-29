"""Predict registration time of non-clone strategies from recorded 100 GiB layouts (results.md, "Options").

    python predict.py natural.jsonl strategies.jsonl

Each thp_probe.py record carries per_chunk = [ms, coalesced, 4 KiB pages, THP heads, visits] for the 315 planned chunks
of the production tier (0:61440,1:40960, 45 layers; the chunk lengths are rebuilt here from the same geometry). The
per-chunk visit model of thp_probe.py is calibrated per layout on its measured direct registration (ns per visit),
then applied to reordered or re-cut chunk lists. "split" assumes each mixed chunk's 4 KiB pages are one contiguous run
(fallback happens in runs), cut out on row boundaries.
"""

import json
import math
import sys

GIB, HUGE, PAGE = 1 << 30, 2 << 20, 4096
EXL3_ROWS = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)


def geometry(total_mib: int) -> list[tuple[int, int]]:
    rows = total_mib * (1 << 20) // sum(EXL3_ROWS)
    layers = math.ceil(rows / 181)
    out = []
    for layer in range(layers):
        count = rows // layers + (layer < rows % layers)
        for row in EXL3_ROWS:
            n, step = count * row, (GIB // row) * row
            out += [(min(step, n - at), row) for at in range(0, n, step)]
    return out


def visits(seq) -> int:
    total = earlier = 0
    for c in seq:
        bvecs = c["H"] if c["co"] else c["P"]
        own = c["H"] * (c["H"] - 1) // 2 if c["co"] else c["H"] * c["P"] // 2
        total += own + c["H"] * earlier
        earlier += bvecs
    return total


def split(c) -> list[dict]:
    n, row_pages = c["L"] // c["row"], -(-c["row"] // PAGE)
    small_rows = min(n, -(-c["S"] // row_pages) + 1)
    heads_small = round(max(0, small_rows * c["H"] / n - c["S"] / 512))
    small = dict(P=small_rows * row_pages, H=heads_small, co=False, L=small_rows * c["row"], row=c["row"], S=c["S"])
    rest = dict(P=(n - small_rows) * row_pages, H=c["H"] - heads_small, co=True, L=(n - small_rows) * c["row"],
                row=c["row"], S=0)
    return [p for p in (rest, small) if p["P"]]


def main() -> None:
    print("| layout | measured direct s | ns/visit | coalesced first, mixed last | two rings | mixed unregistered "
          "| split 4 KiB rows out, last | mixed first | all THP (0 % 4 KiB) |")
    print("|" + "---|" * 9)
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            total = sum(mib for _, mib in r["placement_mib"])
            chunks = [dict(P=-(-length // PAGE), H=h, S=s, co=co, L=length, row=row)
                      for (length, row), (_, co, s, h, _) in zip(geometry(total), r["per_chunk"])]
            assert len(chunks) == len(r["per_chunk"]) == r["chunks"]
            measured = (r.get("strategies") or {}).get("prod", {}).get("s") or r["register_s"]
            ns = measured * 1e9 / visits(chunks)
            mixed = sorted([c for c in chunks if not c["co"]], key=lambda c: -c["H"])
            coal = [c for c in chunks if c["co"]]
            pieces = [p for c in chunks for p in (split(c) if not c["co"] and c["S"] else [c])]
            cut = [p for p in pieces if p["co"]] + sorted([p for p in pieces if not p["co"]], key=lambda c: -c["H"])
            clean = []
            for c in chunks:
                h = max(c["H"], -(-c["L"] // HUGE))
                clean.append(dict(c, H=h, co=h > 1, S=0))
            s = lambda v: v * ns / 1e9  # noqa: E731
            print(f"| {r['label']} | {measured:.1f} | {ns:.1f} | {s(visits(coal + mixed)):.1f} | "
                  f"{s(visits(coal) + visits(mixed)):.1f} | {s(visits(coal)):.1f} | {s(visits(cut)):.1f} | "
                  f"{s(visits(mixed + coal)):.0f} | {s(visits(clean)):.1f} |")


if __name__ == "__main__":
    main()
