"""Per-forward counters from port_counters.sh: each count's difference between the N2- and N1-iteration runs over
N2 - N1, as uops dispatched per core cycle by port, stall fractions, core over reference cycles, and the share of
cycles at each AVX frequency license."""

import csv
import glob
import os
import sys
from collections import defaultdict


def main(results: str) -> None:
    n1, n2 = (int(v) for v in open(f"{results}/iterations.txt").read().split())
    counts = defaultdict(dict)
    for path in glob.glob(f"{results}/*.csv"):
        tag, workers, _, n = os.path.basename(path)[:-4].rsplit("-", 3)
        with open(path) as f:
            for row in csv.reader(line for line in f if line.strip() and not line.startswith("#")):
                try:
                    value = float(row[0])
                except ValueError:  # <not counted> or <not supported>
                    continue
                counts[(tag, workers)][(row[2].removesuffix(":u"), int(n))] = value
    print("bench   workers  port0 port1 port5 port6 load  memstall stalls cycles/ref lic0 lic1 lic2  IPC")
    for (tag, workers), c in sorted(counts.items()):
        per = {ev: (v - c[(ev, n1)]) / (n2 - n1) for (ev, n), v in c.items() if n == n2 and (ev, n1) in c}
        cycles = per["cycles"]

        def share(event: str) -> float:
            return per.get(event, 0.0) / cycles

        port = [share(f"uops_dispatched_port.port_{p}") for p in range(8)]
        lic = [share(f"core_power.lvl{i}_turbo_license") for i in range(3)]
        print(
            f"{tag:8s} {workers:4s}  {port[0]:.2f}  {port[1]:.2f}  {port[5]:.2f}  {port[6]:.2f}  {port[2] + port[3]:.2f}"
            f"  {share('cycle_activity.stalls_mem_any'):.2f}     {share('cycle_activity.stalls_total'):.2f}"
            f"   {cycles / per['ref-cycles']:.2f}      {lic[0]:.2f} {lic[1]:.2f} {lic[2]:.2f}"
            f"  {per['instructions'] / cycles:.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
