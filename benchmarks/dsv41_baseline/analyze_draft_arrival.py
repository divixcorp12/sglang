"""Join draft arrival stages; GPU correlation includes a measured offset interval."""
import argparse
import json
from pathlib import Path


def summarize(root):
    clocks = {}
    for p in root.glob("events.*.draft-clock*.json"):
        pid = int(p.name.split(".")[1])
        clocks.setdefault(pid, []).append(json.loads(p.read_text()))
    requests, drops = [], {}
    for path in root.glob("events.*.jsonl"):
        jobs = {}
        pid = int(path.name.split(".")[1])
        footer = False
        for line in path.open():
            e = json.loads(line)
            if "dropped" in e:
                drops[path.name] = e["dropped"]; footer = True
            if e.get("event", "").startswith("draft_"):
                jobs.setdefault((e["gen"],e["seq"]), {})[e["event"]] = e
        if not footer:
            raise ValueError(f"missing footer: {path}")
        for (epoch,seq), es in jobs.items():
            if "draft_observed" not in es:
                continue
            needed = ["draft_observed","draft_selected","draft_record_ready","draft_payload_ready","draft_start","draft_end"]
            if any(k not in es for k in needed):
                raise ValueError(f"incomplete request {path}:{epoch}:{seq}")
            observed = es["draft_observed"]
            row = dict(pid=pid, epoch=epoch, seq=seq, stage=observed["row"], tid=observed["b"],
                       observed_ns=observed["ns"], gpu_publish_ns=observed["a"],
                       select_us=(es["draft_selected"]["ns"]-observed["ns"])/1000,
                       record_us=(es["draft_record_ready"]["ns"]-es["draft_selected"]["ns"])/1000,
                       prepare_us=(es["draft_payload_ready"]["ns"]-es["draft_record_ready"]["ns"])/1000,
                       start_overhead_us=(es["draft_start"]["ns"]-es["draft_payload_ready"]["ns"])/1000,
                       forward_us=(es["draft_end"]["ns"]-es["draft_start"]["ns"])/1000,
                       trigger=es.get("draft_trigger"))
            if pid in clocks and observed["a"]:
                lo = max(a["offset_low"] for a in clocks[pid])
                hi = min(a["offset_high"] for a in clocks[pid])
                if lo > hi:
                    raise ValueError("clock anchor intervals disagree; cannot correlate GPU and CPU")
                row["publication_to_observe_us"] = [(observed["ns"]-observed["a"]-hi)/1000,
                                                     (observed["ns"]-observed["a"]-lo)/1000]
                row["clock_anchors"] = len(clocks[pid])
            requests.append(row)
    if any(drops.values()):
        raise ValueError(f"trace overflow: {drops}")
    def stats(key):
        xs = sorted(r[key] for r in requests)
        return dict(n=len(xs),p50=xs[len(xs)//2],p95=xs[int((len(xs)-1)*.95)],max=xs[-1]) if xs else dict(n=0)
    return dict(requests=requests, drops=drops,
                stats={k:stats(k) for k in ("select_us","record_us","prepare_us","start_overhead_us","forward_us")},
                clock_note="Publication marker precedes record/head release. GPU correlation assumes stable ns offset; startup-only anchors do not bound later clock drift.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    result=summarize(args.root)
    (args.root/"draft-arrival-analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["stats"],indent=2))
