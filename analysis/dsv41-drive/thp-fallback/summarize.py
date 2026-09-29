"""One table row per thp_probe.py result: python summarize.py results.jsonl [...]"""

import json
import sys

COLUMNS = ("label", "GiB", "4K GiB", "4K %", "mixed", "uncoal.", "fault s", "reg s", "ms mixed", "ms coal.",
           "Gvisits", "ns/visit", "r2", "repair")


def row(r: dict) -> list:
    reg = r.get("after_register") or r.get("after_fault") or {}
    fit = r.get("fit") or {}
    repair = r.get("repair")
    return [
        r["label"], f"{reg.get('tier_gib', 0):.1f}", f"{reg.get('small_gib', 0):.2f}",
        f"{100 * reg.get('small_fraction', 0):.2f}", r.get("mixed_chunks", "-"), r.get("uncoalesced_chunks", "-"),
        f"{r.get('fault_s', 0):.1f}", f"{r.get('register_s', 0):.2f}", f"{r.get('ms_mixed_chunks', 0):.0f}",
        f"{r.get('ms_coalesced_chunks', 0):.0f}", f"{r.get('visits_total_g', 0):.2f}",
        f"{fit.get('ms_per_gvisit', 0) / 1e3:.2f}" if fit else "-", f"{fit.get('r2', 0):.3f}" if fit else "-",
        (f"{repair['frames']} frames, {repair['failed_calls']} failed, {repair['s']:.1f} s, "
         f"4K {repair['after']['small_gib']:.2f} GiB -> mixed {repair['after']['mixed_chunks']}") if repair else "",
    ]


def main() -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for path in sys.argv[1:]:
        for line in open(path):
            if line.strip():
                print("| " + " | ".join(str(x) for x in row(json.loads(line))) + " |")


if __name__ == "__main__":
    main()
