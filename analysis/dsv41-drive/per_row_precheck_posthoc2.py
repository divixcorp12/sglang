"""Post-hoc, NOT registered: is a request's pack order (row_pack_ns end order) its ordinal order? Read-only over a trace."""
import json, sys, collections
for path in sys.argv[1:]:
    tot = same = 0
    by_m = collections.Counter(); by_m_same = collections.Counter()
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("kind") != "ram_miss_request" or d["request"]["type"] != "demand" or d["status"] != "served":
                continue
            rows = d["row_pack_ns"]
            m = len(rows)
            if m < 2 or d["untraced"]["rows"]:
                continue
            order = [r["row"] for r in sorted(rows, key=lambda r: r["end"])]
            ok = order == sorted(order)
            tot += 1; same += ok; by_m[m] += 1; by_m_same[m] += ok
    print(path.rsplit("/", 1)[-1], "m>=2 requests", tot, "pack order == ordinal order:", round(same / tot, 4), {m: round(by_m_same[m] / by_m[m], 3) for m in sorted(by_m)})
