"""Split a capture's NVMe demand reads into forced CPU misses and the rest ("other reads").

other reads = rows_read - (forced CPU misses - pool-served ones). The rest are unforced misses (a staging slot,
GPU-lane or CPU-with-staging), which skip the prefetch pool and re-read NVMe (DSV41_REFERENCE.md §33.36).
Needs an InstrBuild capture: `miss_expert` events exist only for forced CPU misses.

Usage: read_split.py RUN_DIR...   (a spec_margin_capture.py arm dir: server.log plus events.*.exl3-tier*.jsonl)
"""

import glob
import json
import os
import sys

for run in sys.argv[1:]:
    misses = sum(
        1 for f in glob.glob(run + "/events.*.exl3-tier*.jsonl") for line in open(f) if "miss_expert" in line
    )
    line = [l for l in open(run + "/server.log") if "exl3 RAM miss thread counters" in l][-1]
    c = json.loads(line[line.index("{") :])
    used = c.get("spec_used", 0)
    forced = misses - used
    other = c["rows_read"] - forced
    name = os.path.join(*os.path.normpath(run).split(os.sep)[-2:])
    print(
        "%-26s rows_read %6d  forced CPU misses %6d  pool-served %5d  forced reads %6d  other reads %5d (%2.0f%%)"
        "  landed %5d  dropped %5d"
        % (name, c["rows_read"], misses, used, forced, other, 100.0 * other / c["rows_read"],
           c.get("spec_landed", 0), c.get("spec_dropped", 0))
    )
