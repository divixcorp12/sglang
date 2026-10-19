"""Per-verify table of a served long-prompt capture, for the cost-aware cut study (costcut_study.py).

One row per verify of ``RUN_DIR`` (spec_margin_capture's output directory), from the job trace (CLOCK_MONOTONIC):
  cost_ms         verify_end - verify_begin (the whole step: draft, verify, host work);
  tokens          completion tokens that the client's streamed chunks gained at this verify (chunks are matched to the
                  last verify that ended before them; 0 when no chunk arrived for it);
  routed[l]       svc_routed b of layer l (group 0): the experts the verify's live rows routed there;
  lanes[l]        svc_routed c of layer l: the lanes (VRAM misses) of that layer;
  read_ms         summed svc_record c (ns inside read_rows) over both groups, the NVMe read wait the host spent;
  svc_ms          summed svc_record a (service ns);
  forced          miss_expert events (MISS_CPU lanes), evict events (demand reads in) and pool_point events;
  begin_ns, end_ns.
The tier files are complete (one svc_routed per layer per verify); the copy-engine file overflows in the cut arms, so
nothing from it is used here.

Usage: costcut_extract.py RUN_DIR OUT.npz
"""

from __future__ import annotations

import bisect
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_compare  # noqa: E402

LAYERS = 40


def main(run_dir: str, out: str) -> int:
    verifies, _ = serve_compare.load_verifies(run_dir)
    n = len(verifies)
    begins = [b for b, _ in verifies]
    routed = np.zeros((n, LAYERS), np.int16)
    lanes = np.zeros((n, LAYERS), np.int16)
    seen = np.zeros((n, LAYERS), bool)
    read_ns = np.zeros(n)
    svc_ns = np.zeros(n)
    forced = np.zeros(n, np.int32)
    evict = np.zeros(n, np.int32)
    pool = np.zeros(n, np.int32)
    wanted = {"svc_routed", "svc_record", "miss_expert", "evict", "pool_point"}
    for path in sorted(glob.glob(os.path.join(run_dir, "events.*.exl3-tier*.jsonl"))):
        with open(path) as f:
            f.readline()
            for line in f:
                head = line[:44]
                if not any(f'"{w}"' in head for w in wanted):
                    continue
                e = json.loads(line)
                k = bisect.bisect_right(begins, e["ns"]) - 1
                if k < 0 or e["ns"] > verifies[k][1]:
                    continue
                ev = e["event"]
                if ev == "svc_routed":
                    if e["group"] == 0:
                        routed[k, e["row"]] = e["b"]
                        lanes[k, e["row"]] = e["c"]
                        seen[k, e["row"]] = True
                elif ev == "svc_record":
                    read_ns[k] += e["c"]
                    svc_ns[k] += e["a"]
                elif ev == "miss_expert":
                    forced[k] += 1
                elif ev == "evict":
                    evict[k] += 1
                elif ev == "pool_point":
                    pool[k] += 1
    with open(os.path.join(run_dir, "long_prompt.json")) as f:
        request = json.load(f)["requests"][0]
    ends = [e for _, e in verifies]
    tokens = np.zeros(n, np.int32)
    seen_tokens = 0
    for c in request["chunks"]:
        t = c["completion_tokens"]
        if t is None or t <= seen_tokens:
            continue
        k = bisect.bisect_right(ends, c["ns"]) - 1
        if k >= 0:
            tokens[k] += t - seen_tokens
        seen_tokens = t
    np.savez_compressed(
        out, begin_ns=np.array(begins), end_ns=np.array(ends), cost_ms=(np.array(ends) - np.array(begins)) / 1e6,
        tokens=tokens, routed=routed, lanes=lanes, complete=seen.all(axis=1), read_ms=read_ns / 1e6, svc_ms=svc_ns / 1e6,
        forced=forced, evict=evict, pool=pool, first_token_ns=request["first_token_ns"], end_request_ns=request["end_ns"])
    print(f"{run_dir}: {n} verifies, {int(seen.all(axis=1).sum())} with all {LAYERS} layers, {int(tokens.sum())} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
