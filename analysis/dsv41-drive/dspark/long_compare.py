"""Compare long-prompt captures (spec_margin_capture.py --long-prompt output dirs) by elapsed decode, or read one alone
as a diagnostic.

Per run, from long_prompt.json and the job trace (CLOCK_MONOTONIC on both, so the client's chunks and the server's
verifies share a clock):
  TTFT of each request (the --ttft-repeats samples and the long one), and its prompt tokens served from cache;
  windows of elapsed decode from the long request's first streamed token: 0-30 s, 30-60 s, then each 60 s to the cut;
  per window, from the client: ms/token (the window's time over the tokens streamed in it); from the verifies that
    began in it (lane_tier_replay.Run's records): tokens per verify, stream and wait per verify, forced CPU misses
    (the CPU's forced-miss jobs) and cold GPU lanes (NVMe-read MISS_GPU lanes) per verify;
  flags per window, serve_compare.py's with the window in the prompt's place: STALL when its ms per stream layer is over
    --stall-factor x the run's median window, corroborated by the memsample's worst drive await; CPUSLOW when either
    team's CPU ms per lane is over --cpu-factor x the median, corroborated by swap-in and major fault rate;
  PREFILL-STALL per request (share_ab_report.py's 2x rule): the nvme3n1 read await over its prefill (sent to first
    token) over 2x the median prefill's, pooled over every run given (a run holds only K+1 prefills); with under 3
    prefills nothing is flagged and the awaits are reported for reading by eye;
  trace truncation: the job trace keeps each file's first CAPACITY events, so a window past a file's last event when
    it dropped any is marked TRUNC and its server columns undercount.
Pair (first run vs each other): greedy decoding makes the two texts equal until they diverge. The divergence token is
the token count before the first chunk whose text differs, exact to within a chunk (a verify's accepted tokens).
Over that common prefix, both arms' ms/token on the same tokens: overall, and per window of the first run's elapsed
decode (the token range the first run streamed in it). The server columns are paired by elapsed window.

Usage: long_compare.py RUN_DIR [RUN_DIR ...] [--stall-factor 1.3] [--cpu-factor 1.2] [--json OUT]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_tier_replay as ltr  # noqa: E402
import serve_compare  # noqa: E402
import share_ab_report  # noqa: E402

FIRST_WINDOWS_S = (30.0, 60.0)
WINDOW_S = 60.0
MIN_PREFILLS = 3


def window_edges(duration_s: float) -> list[float]:
    edges = [0.0]
    for e in FIRST_WINDOWS_S:
        if e < duration_s:
            edges.append(e)
    while edges[-1] + WINDOW_S < duration_s:
        edges.append(edges[-1] + WINDOW_S)
    edges.append(duration_s)
    return edges


def bucket(t_s: float, edges: list[float]) -> int | None:
    """The window holding elapsed ``t_s``; None before the first token or past the last edge."""
    if t_s < 0 or t_s > edges[-1]:
        return None
    return min(bisect.bisect_right(edges, t_s) - 1, len(edges) - 2)


def counted(chunks: list[dict]) -> list[dict]:
    """The chunks that advanced the completion count, with their cumulative text length."""
    out, seen, chars = [], 0, 0
    for c in chunks:
        chars += len(c["reasoning"]) + len(c["content"])
        n = c["completion_tokens"]
        if n is not None and n > seen:
            out.append({"ns": c["ns"], "tokens": n, "chars": chars})
            seen = n
    return out


def tokens_at(points: list[dict], ns: int) -> int:
    k = bisect.bisect_right([p["ns"] for p in points], ns) - 1
    return points[k]["tokens"] if k >= 0 else 0


def ns_at_token(points: list[dict], n: int) -> int | None:
    """Arrival of the chunk that brought the count to ``n``."""
    for p in points:
        if p["tokens"] >= n:
            return p["ns"]
    return None


def client_windows(points: list[dict], t0: int, edges: list[float]) -> list[dict]:
    out = []
    for a, b in zip(edges, edges[1:]):
        lo, hi = tokens_at(points, t0 + int(a * 1e9)), tokens_at(points, t0 + int(b * 1e9))
        out.append({"t0_s": a, "t1_s": b, "tokens": hi - lo, "token_range": [lo, hi],
                    "ms_per_token": (b - a) * 1e3 / (hi - lo) if hi > lo else None})
    return out


def divergence(a: list[dict], b: list[dict]) -> dict:
    """The two texts' common prefix, in characters and in tokens of both arms (the smaller)."""
    ta = "".join(c["reasoning"] + c["content"] for c in a)
    tb = "".join(c["reasoning"] + c["content"] for c in b)
    chars = len(os.path.commonprefix([ta, tb]))
    diverged = chars < min(len(ta), len(tb))

    def within(chunks):
        pts = counted(chunks)
        return max((p["tokens"] for p in pts if p["chars"] <= chars), default=0)

    tokens = min(within(a), within(b))
    return {"common_chars": chars, "common_tokens": tokens, "diverged": diverged,
            "divergence_token": tokens if diverged else None}


def ms_per_token_over(points: list[dict], lo: int, hi: int) -> float | None:
    t_lo, t_hi = ns_at_token(points, lo), ns_at_token(points, hi)
    if t_lo is None or t_hi is None or hi <= lo:
        return None
    return (t_hi - t_lo) / 1e6 / (hi - lo)


def trace_ends(d: str) -> dict[str, int]:
    """Per events file that dropped events, the ns of its last kept one."""
    out = {}
    for path in glob.glob(os.path.join(d, "events.*.jsonl")):
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = [json.loads(line) for line in f.read().decode(errors="replace").splitlines()[1:] if line.strip()]
        if tail and tail[-1].get("dropped"):
            last = [e["ns"] for e in tail if "ns" in e]
            if last:
                out[os.path.basename(path)] = last[-1]
    return out


def server_windows(d: str, t0: int, cut: int, edges: list[float], stall_factor: float, cpu_factor: float,
                   run: ltr.Run | None = None) -> list[dict]:
    run = run or ltr.Run(d)
    n = len(edges) - 1
    win = [collections.Counter() for _ in range(n)]
    cpu = [collections.defaultdict(list) for _ in range(n)]
    of_verify = {}
    for k, (b, _) in enumerate(run.verifies):
        if b <= cut:
            i = bucket((b - t0) / 1e9, edges)
            if i is not None:
                of_verify[k] = i
                win[i]["verifies"] += 1
    for r in run.records:
        i = of_verify.get(r.verify)
        if i is None:
            continue
        w = win[i]
        w["stream_ms"] += r.stream
        w["wait_ms"] += r.wait
        w["gpu_nvme"] += r.gpu_nvme
        w["cpu_forced"] += ltr.cpu_misses(r)
        if r.stream > serve_compare.STREAM_LAYER_NS / 1e6:
            w["stream_layers"] += 1
        for j in r.jobs:
            cpu[i][j["team"]].append((j["end"] - j["start"]) / 1e6 / max(j["lanes"], 1))
    teams = sorted({t for c in cpu for t in c})
    out = []
    for i, w in enumerate(win):
        v = w["verifies"]
        out.append({
            "verifies": v,
            "stream_per_verify": w["stream_ms"] / v if v else None,
            "wait_per_verify": w["wait_ms"] / v if v else None,
            "cpu_forced_per_verify": w["cpu_forced"] / v if v else None,
            "gpu_nvme_per_verify": w["gpu_nvme"] / v if v else None,
            "stream_layers_per_verify": w["stream_layers"] / v if v else None,
            "ms_per_stream_layer": w["stream_ms"] / w["stream_layers"] if w["stream_layers"] else 0.0,
            "cpu_ms_per_lane": {t: statistics.mean(cpu[i][t]) if cpu[i][t] else 0.0 for t in teams},
        })
    flag_windows(out, teams, stall_factor, cpu_factor)
    return out


def flag_windows(windows: list[dict], teams: list[int], stall_factor: float, cpu_factor: float) -> None:
    """serve_compare.prompts' STALL and CPUSLOW rules, each window against the run's median window."""
    live = [w for w in windows if w["verifies"]]
    for w in windows:
        w["stall"] = bool(w["verifies"]) and serve_compare._median_ratio(
            w["ms_per_stream_layer"], [x["ms_per_stream_layer"] for x in live]) > stall_factor
        w["cpu_ratio"] = max((serve_compare._median_ratio(w["cpu_ms_per_lane"][t], [x["cpu_ms_per_lane"][t] for x in live])
                              for t in teams), default=0.0) if w["verifies"] else 0.0
        w["cpu_slow"] = w["cpu_ratio"] > cpu_factor


def corroborate(d: str, windows: list[dict], t0_epoch_s: float, edges: list[float]) -> None:
    """serve_compare's memsample corroboration over each window's epoch span."""
    intervals = serve_compare.memsample_intervals(serve_compare.load_memsample(d))
    dev_await = collections.defaultdict(list)
    for iv in intervals:
        for dev, aw in iv["await"].items():
            if aw is not None:
                dev_await[dev].append(aw)
    med_await = {dev: statistics.median(xs) for dev, xs in dev_await.items()}
    med_fault = statistics.median(iv["fault"] for iv in intervals) if intervals else 0.0
    for w, a, b in zip(windows, edges, edges[1:]):
        lo, hi = t0_epoch_s + a, t0_epoch_s + b
        over = [iv for iv in intervals if iv["t0"] < hi and lo < iv["t1"]]
        w["disk_ratio"] = max((aw / med_await[dev] for iv in over for dev, aw in iv["await"].items()
                               if aw is not None and med_await.get(dev)), default=None)
        fault = max((iv["fault"] for iv in over), default=None)
        w["fault_ratio"] = None if fault is None else fault / max(med_fault, 1.0)
        w["node0_free_mib"] = min((iv["node0_free_mib"] for iv in over), default=None)


def prefill_awaits(d: str, requests: list[dict]) -> list[dict]:
    """Each request's nvme3n1 read await from its send to its first token."""
    samples = share_ab_report.nvme3n1_samples(d)
    out = []
    for r in requests:
        first = epoch_of_first_token(r)
        aw, reads = (share_ab_report.window_await(samples, r["sent_epoch_ns"] / 1e9, first)
                     if first is not None else (None, 0))
        out.append({"rid": r["rid"], "nvme3n1_await_ms": aw, "nvme3n1_reads": reads, "prefill_stall": None})
    return out


def flag_prefills(results: list[dict]) -> float | None:
    """share_ab_report's PREFILL-STALL: a prefill's await over 2x the median prefill's, pooled over every run given,
    since a run holds one prefill per request. Prefill reads queue deeper than decode's, so decode is no reference.
    Under MIN_PREFILLS awaits there is no median to trust and nothing is flagged."""
    awaits = [p["nvme3n1_await_ms"] for r in results for p in r["prefill"] if p["nvme3n1_await_ms"] is not None]
    if len(awaits) < MIN_PREFILLS:
        return None
    ref = statistics.median(awaits)
    for r in results:
        for p in r["prefill"]:
            p["prefill_stall"] = bool(p["nvme3n1_await_ms"] and p["nvme3n1_await_ms"] > share_ab_report.STALL_FACTOR * ref)
    return ref


def epoch_of_first_token(r: dict) -> float | None:
    for c in r["chunks"]:
        if c["ns"] == r["first_token_ns"]:
            return c["epoch_ns"] / 1e9
    return None


def long_request(d: str) -> tuple[dict, dict]:
    with open(os.path.join(d, "long_prompt.json")) as f:
        record = json.load(f)
    longs = [r for r in record["requests"] if r["kind"] == "long"]
    if len(longs) != 1 or longs[0]["first_token_ns"] is None:
        raise SystemExit(f"{d}: no long request with a first token in long_prompt.json")
    return record, longs[0]


def analyse(d: str, stall_factor: float, cpu_factor: float, run: ltr.Run | None = None) -> dict:
    record, long = long_request(d)
    t0 = long["first_token_ns"]
    points = counted(long["chunks"])
    cut = points[-1]["ns"] if points else t0
    # The cut lands on the first chunk at or past decode_seconds; its overshoot would make a sliver window.
    duration_s = min((cut - t0) / 1e9, long["decode_seconds"]) if long["cut"] else (cut - t0) / 1e9
    edges = window_edges(duration_s)
    cut = t0 + int(duration_s * 1e9)
    windows = client_windows(points, t0, edges)
    for w, s in zip(windows, server_windows(d, t0, cut, edges, stall_factor, cpu_factor, run)):
        w.update(s)
        w["tokens_per_verify"] = w["tokens"] / w["verifies"] if w["verifies"] else None
    t0_epoch = epoch_of_first_token(long)
    corroborate(d, windows, t0_epoch, edges)
    ends = trace_ends(d)
    complete_s = (min(ends.values()) - t0) / 1e9 if ends else None
    for w in windows:
        w["truncated"] = complete_s is not None and w["t1_s"] > complete_s
    requests = record["requests"]
    return {
        "dir": d,
        "prompt_sha256": record["prompt_sha256"],
        "ttft_s": [{"rid": r["rid"], "kind": r["kind"], "ttft_s": r["ttft_s"],
                    "cached_tokens": next((c["cached_tokens"] for c in r["chunks"] if c["cached_tokens"] is not None),
                                          None)} for r in requests],
        "long": {"ttft_s": long["ttft_s"], "decode_s": (cut - t0) / 1e9, "tokens": tokens_at(points, cut),
                 "cut": long["cut"], "finish_reason": long["finish_reason"]},
        "windows": windows,
        "prefill": prefill_awaits(d, requests),
        "trace_dropped_files": sorted(ends),
        "trace_complete_through_s": complete_s,
        "drive_busy_s": serve_compare.drive_load(d),
    }


def pair(a_dir: str, b_dir: str, a: dict, b: dict) -> dict:
    _, la = long_request(a_dir)
    _, lb = long_request(b_dir)
    div = divergence(la["chunks"], lb["chunks"])
    pa, pb = counted(la["chunks"]), counted(lb["chunks"])
    start = max(pa[0]["tokens"], pb[0]["tokens"]) if pa and pb else 0
    end = div["common_tokens"]
    by_window = []
    for w in a["windows"]:
        lo, hi = max(w["token_range"][0], start), min(w["token_range"][1], end)
        ma, mb = ms_per_token_over(pa, lo, hi), ms_per_token_over(pb, lo, hi)
        by_window.append({"t0_s": w["t0_s"], "t1_s": w["t1_s"], "tokens": [lo, hi] if hi > lo else None,
                          "ms_per_token": [ma, mb], "delta": mb - ma if ma is not None and mb is not None else None})
    server = []
    for wa, wb in zip(a["windows"], b["windows"]):
        server.append({k: (wb[k] - wa[k] if wa.get(k) is not None and wb.get(k) is not None else None)
                       for k in ("stream_per_verify", "wait_per_verify", "cpu_forced_per_verify", "gpu_nvme_per_verify",
                                 "tokens_per_verify", "ms_per_token")}
                      | {"t0_s": wa["t0_s"], "t1_s": wa["t1_s"],
                         "flags": [window_flags(wa), window_flags(wb)]})
    ma, mb = ms_per_token_over(pa, start, end), ms_per_token_over(pb, start, end)
    return {
        "against": a_dir,
        "same_prompt": a["prompt_sha256"] == b["prompt_sha256"],
        "divergence": div,
        "common_prefix": {"tokens": [start, end], "ms_per_token": [ma, mb],
                          "delta": mb - ma if ma is not None and mb is not None else None},
        "by_window_same_tokens": by_window,
        "by_window_server": server,
        "ttft_delta_s": b["long"]["ttft_s"] - a["long"]["ttft_s"],
    }


def window_flags(w: dict) -> str:
    out = serve_compare.flags(w)
    return (out + " TRUNC").strip() if w.get("truncated") else out


def _f(v, fmt: str) -> str:
    return f"{v:{fmt}}" if v is not None else "-"


def print_run(res: dict) -> None:
    print(f"== {os.path.basename(os.path.normpath(res['dir']))}")
    for t in res["ttft_s"]:
        print(f"   {t['kind']:4s} {t['rid']}: TTFT {_f(t['ttft_s'], '.2f')} s, cached prompt tokens {t['cached_tokens']}")
    lg = res["long"]
    print(f"   long: {lg['tokens']} tokens over {lg['decode_s']:.1f} s of decode (cut {lg['cut']}, finish {lg['finish_reason']})")
    for p in res["prefill"]:
        print(f"   prefill {p['rid']}: nvme3n1 await {_f(p['nvme3n1_await_ms'], '.1f')} ms over {p['nvme3n1_reads']} reads"
              + ("  PREFILL-STALL" if p["prefill_stall"] else ""))
    print("   window(s)  tokens ms/token verifies tok/v stream/v  wait/v cpu-forced/v gpu-cold/v ms/layer  disk x swap x  flags")
    for w in res["windows"]:
        print(f"   {w['t0_s']:4.0f}-{w['t1_s']:<4.0f} {w['tokens']:7d} {_f(w['ms_per_token'], '8.2f')} {w['verifies']:8d}"
              f" {_f(w['tokens_per_verify'], '5.2f')} {_f(w['stream_per_verify'], '8.1f')} {_f(w['wait_per_verify'], '7.1f')}"
              f" {_f(w['cpu_forced_per_verify'], '12.2f')} {_f(w['gpu_nvme_per_verify'], '10.2f')}"
              f" {w['ms_per_stream_layer']:8.2f} {_f(w['disk_ratio'], '6.1f')} {_f(w['fault_ratio'], '6.1f')}  {window_flags(w)}")
    if res["trace_dropped_files"]:
        print(f"   job trace dropped events in {', '.join(res['trace_dropped_files'])}: complete through "
              f"{res['trace_complete_through_s']:.1f} s of decode")


def print_pair(name: str, p: dict) -> None:
    div = p["divergence"]
    print(f"== {name} - {os.path.basename(os.path.normpath(p['against']))}")
    if not p["same_prompt"]:
        print("   !! the two runs served different prompts")
    print(f"   TTFT delta {p['ttft_delta_s']:+.2f} s")
    print(f"   common prefix {div['common_tokens']} tokens ({div['common_chars']} chars); "
          + (f"diverged at token {div['divergence_token']}" if div["diverged"] else "no divergence in the shorter text"))
    c = p["common_prefix"]
    print(f"   ms/token on tokens {c['tokens'][0]}..{c['tokens'][1]}: {_f(c['ms_per_token'][0], '.2f')} -> "
          f"{_f(c['ms_per_token'][1], '.2f')} ({_f(c['delta'], '+.2f')})")
    print("   window(s)  tokens        ms/token a -> b (same tokens)  d stream/v d wait/v d cpu-forced/v d gpu-cold/v  flags (a | b)")
    for w, s in zip(p["by_window_same_tokens"], p["by_window_server"]):
        tok = f"{w['tokens'][0]}..{w['tokens'][1]}" if w["tokens"] else "-"
        print(f"   {w['t0_s']:4.0f}-{w['t1_s']:<4.0f} {tok:>13s} {_f(w['ms_per_token'][0], '7.2f')} -> "
              f"{_f(w['ms_per_token'][1], '7.2f')} ({_f(w['delta'], '+6.2f')})"
              f" {_f(s['stream_per_verify'], '+10.1f')} {_f(s['wait_per_verify'], '+8.1f')}"
              f" {_f(s['cpu_forced_per_verify'], '+14.2f')} {_f(s['gpu_nvme_per_verify'], '+12.2f')}"
              f"  {s['flags'][0]} | {s['flags'][1]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stall-factor", type=float, default=1.3)
    ap.add_argument("--cpu-factor", type=float, default=1.2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    out = {d: analyse(d, a.stall_factor, a.cpu_factor) for d in a.runs}
    ref = flag_prefills(list(out.values()))
    print(f"prefill nvme3n1 await median over {sum(len(r['prefill']) for r in out.values())} prefills: "
          + (f"{ref:.1f} ms (PREFILL-STALL over {share_ab_report.STALL_FACTOR:g}x)" if ref is not None
             else f"under {MIN_PREFILLS} prefills with reads, none flagged"))
    for d in a.runs:
        out[d]["prefill_median_await_ms"] = ref
        print_run(out[d])
    first = a.runs[0]
    for d in a.runs[1:]:
        out[d]["pair"] = pair(first, d, out[first], out[d])
        print_pair(os.path.basename(os.path.normpath(d)), out[d]["pair"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
