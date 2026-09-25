"""Per-layer RAM-tier split from a decode route log (DSV41_REFERENCE.md 27.4 item 5).

Rebuilds each layer's VRAM-miss stream from `graph_routes` (routes minus that forward's hot set), runs Mattson's
LRU stack-distance pass per layer to get RAM misses for every capacity, validates the even split against the
measured `graph_step.layer_ram_rows`, then allocates the row budget greedily by marginal misses saved.

Usage: ram_split.py STAGES_JSONL TOTAL_ROWS [--min-rows N] [--out WEIGHTS_JSON]

The weights file is what SGLANG_MOE_PINNED_HOST_LAYER_WEIGHTS reads.
"""
import argparse
import heapq
import json

ap = argparse.ArgumentParser()
ap.add_argument("stages")
ap.add_argument("total_rows", type=int)
ap.add_argument("--min-rows", type=int, default=64, help="floor per layer (the eager gather needs up to 64 rows)")
ap.add_argument("--out", help="write {\"layer_rows\": [...]} here")
a = ap.parse_args()

streams = None
held = None
measured = None
tokens = held_tokens = 0
rid_index = {}
for line in open(a.stages):
    r = json.loads(line)
    kind = r.get("kind")
    if kind == "graph_routes" and r.get("phase") == "decode":
        routes, hot = r["routes"], r["hot"]
        if streams is None:
            streams = [[] for _ in routes]
            held = [[] for _ in routes]
        rid = (r.get("rids") or ["?"])[0]
        rid_index.setdefault(rid, len(rid_index))
        test = rid_index[rid] % 2 == 1  # odd requests held out, as in the prefetch study
        target = held if test else streams
        for layer, (rt, h) in enumerate(zip(routes, hot)):
            hs = set(h)
            target[layer].extend(e for e in rt if e not in hs)
        if test:
            held_tokens += int(r.get("forward_tokens", 1))
        else:
            tokens += int(r.get("forward_tokens", 1))
    elif kind == "graph_step" and r.get("layer_ram_rows"):
        rows = r["layer_ram_rows"]
        measured = measured or [0] * len(rows)
        for i, v in enumerate(rows):
            measured[i] += v

L = len(streams)
E = 384
# curves[l][c] = misses of an LRU of capacity c over layer l's stream (c = 0..E).
def lru_curves(streams_):
    out = []
    for s in streams_:
        stack, hist, cold = [], [0] * (E + 1), 0
        for e in s:
            try:
                d = stack.index(e)
                hist[d] += 1
                stack.pop(d)
            except ValueError:
                cold += 1
            stack.insert(0, e)
        # An access at stack distance d hits iff capacity > d.
        tail = [0] * (E + 2)
        for d in range(E, -1, -1):
            tail[d] = tail[d + 1] + hist[d]
        out.append([cold + tail[c] for c in range(E + 1)])
    return out

curves = lru_curves(streams)
held_curves = lru_curves(held)

even = [a.total_rows // L + (1 if l < a.total_rows % L else 0) for l in range(L)]
sim_even = [curves[l][min(even[l], E)] for l in range(L)]
print(f"decode tokens {tokens}; VRAM-miss accesses/token {sum(len(s) for s in streams)/tokens:.2f}")
if measured:
    print(f"RAM misses/token: simulated even split {sum(sim_even)/tokens:.2f}, measured {sum(measured)/tokens:.2f}")
    top = sorted(range(L), key=lambda l: -measured[l])[:8]
    print("measured top layers:", ", ".join(f"L{l}:{measured[l]/tokens:.2f}" for l in top))
    print("simulated top layers:", ", ".join(f"L{l}:{sim_even[l]/tokens:.2f}" for l in sorted(range(L), key=lambda l: -sim_even[l])[:8]))

# Greedy allocation: start every layer at the floor, then give each next row to the largest marginal saving.
alloc = [a.min_rows] * L
left = a.total_rows - sum(alloc)
heap = [(-(curves[l][alloc[l]] - curves[l][alloc[l] + 1]), l) for l in range(L)]
heapq.heapify(heap)
while left > 0 and heap:
    gain, l = heapq.heappop(heap)
    alloc[l] += 1
    left -= 1
    if alloc[l] < E:
        heapq.heappush(heap, (-(curves[l][alloc[l]] - curves[l][alloc[l] + 1]), l))
sim_opt = [curves[l][alloc[l]] for l in range(L)]
print(f"greedy split: simulated RAM misses/token {sum(sim_opt)/tokens:.2f} "
      f"(even {sum(sim_even)/tokens:.2f}, saving {(sum(sim_even)-sum(sim_opt))/tokens:.2f}/token)")
print("greedy rows per layer:", alloc)
he = sum(held_curves[l][min(even[l], E)] for l in range(L)) / held_tokens
ho = sum(held_curves[l][alloc[l]] for l in range(L)) / held_tokens
print(f"HELD-OUT ({len(rid_index)//2} requests, {held_tokens} tokens): even {he:.2f}, greedy-from-train {ho:.2f}, "
      f"saving {he-ho:.2f}/token ({(he-ho)/he*100:.1f}%)")
print(json.dumps({"layer_rows": alloc}))
if a.out:
    with open(a.out, "w") as f:
        json.dump({"layer_rows": alloc}, f)
