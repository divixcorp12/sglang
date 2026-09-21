import sys, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
for name in ON[:1] + OFF:
    steps, used, eager = analyse(name)
    rows = share = reqs = reqs_share = 0
    for d, st in used:
        m = d["rows_asked"]
        if m < 2: continue
        whole = collections.defaultdict(int)
        for e in d["extent_cqe_ns"]: whole[e["row"]] = max(whole[e["row"]], e["cqe"])
        w = [whole[j] for j in range(m)]
        reqs += 1; rows += m - 1
        s = sum(1 for j in range(1, m) if w[j] == w[j-1])
        share += s; reqs_share += (s > 0)
    print("%s: m>=2 requests %d; requests with >=1 pair of rows completing in the same reap %d (%.1f%%); adjacent row pairs sharing a reap %d of %d (%.1f%%)" %
          (name, reqs, reqs_share, 100*reqs_share/reqs, share, rows, 100*share/rows))
