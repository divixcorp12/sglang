"""The DSpark sampled-finish A/B report (spec F142, dspark-sampled-finish, S2.3): -prod, the sampled confidence cut and the
cut plus block verification, one long-prompt capture each (run_sampled_finish_ab.sh), compared.

Per arm, from long_compare.analyse (the client's stream and the server's verifies on one clock):
  ms/token over the whole decode and over 60-360 s of elapsed decode, ms per verify (window time over the verifies that
  began in it), tokens per verify, the long request's TTFT, finish reason;
and from the arm's server.log:
  KV (max_total_num_tokens), the confidence cut's decode-log summaries ("conf cut: verifies N, w* ..., cut rows R
  (total T), widths [...]": verify-weighted w*, summed width counts, total cut rows), the T(w) calibration line
  (a, b, r2 and, when it logs them, the per-width sample counts), the count of "device-loaded after serving started"
  warnings, "dropped an uncommitted verify" lines, and error lines (Traceback, CUDA error, out of memory, KvPinMissed, a
  disabled cut).

Ship rule (owner, 2026-10-18), printed as ADVISORY: the owner gives the final yes in every case. An arm clears the rule
against -prod when its 60-360 s ms/token and ms per verify are both >= 5% lower, it logged no errors and its tokens per
verify are not lower; the block arm ships over the cut-only arm only if it also beats cut-only by >= 5% on both. When no
arm clears it the result goes to the owner.

Usage: sampled_finish_report.py --prod RUN_DIR --cut RUN_DIR --block RUN_DIR [--json OUT]
(RUN_DIR is the arm's directory holding server.log and long_prompt.json, e.g. OUT/prod/prod.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_compare  # noqa: E402

SHIP_FRACTION = 0.05
STEADY_WINDOW_S = (60.0, 360.0)
KV_PIN = 381184
ARMS = ("prod", "cut", "block")

CONF_CUT_RE = re.compile(
    r"conf cut: verifies (?P<n>\d+), w\* (?P<w>[\d.]+), E\(w\*\) (?P<e>[\d.]+), accept len (?P<acc>[\d.]+), "
    r"cut rows (?P<rows>\d+) \(total (?P<total>\d+)\)(?:, widths \[(?P<widths>[\d, ]*)\])?(?:, sampled (?P<sampled>\d+))?"
)
TW_RE = re.compile(
    r"dspark confidence cut: T\(w\) ms = \[(?P<ms>[^\]]*)\] \(a=(?P<a>[-\d.eE+]+), b=(?P<b>[-\d.eE+]+)/row, "
    r"r2=(?P<r2>[-\d.eE+]+)(?:;[^)]*?counts \[(?P<counts>[\d, ]*)\])?"
)
KV_RE = re.compile(r"max_total_num_tokens=(\d+)")
LATE_LOAD = "device-loaded after serving started"
DROPPED = "dropped an uncommitted verify"
ERROR_RE = re.compile(
    r"Traceback \(most recent|CUDA error|OutOfMemoryError|CUDA out of memory|KvPinMissed|confidence cut disabled"
    r"|RemoteDisconnected"
)


def _ints(text: str | None) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()] if text else []


def parse_server_log(text: str) -> dict:
    """The log lines the report reads. ``cut`` aggregates every decode-log "conf cut:" summary; ``calibration`` is the
    last T(w) line (None when the cut never calibrated)."""
    kv, late, dropped, errors = None, 0, 0, []
    verifies, w_sum, e_sum, acc_sum, total_rows, widths, sampled = 0, 0.0, 0.0, 0.0, 0, [], 0
    summaries, calibration = 0, None
    for line in text.splitlines():
        if (m := KV_RE.search(line)) and kv is None:
            kv = int(m.group(1))
        late += LATE_LOAD in line
        dropped += DROPPED in line
        if ERROR_RE.search(line):
            errors.append(line.strip()[:200])
        if m := CONF_CUT_RE.search(line):
            n = int(m["n"])
            summaries += 1
            verifies += n
            w_sum += float(m["w"]) * n
            e_sum += float(m["e"]) * n
            acc_sum += float(m["acc"]) * n
            total_rows = max(total_rows, int(m["total"]))
            sampled += int(m["sampled"] or 0)
            counts = _ints(m["widths"])
            size = max(len(widths), len(counts))
            widths = [(widths + [0] * size)[k] + (counts + [0] * size)[k] for k in range(size)]
        if m := TW_RE.search(line):
            counts = _ints(m["counts"])
            calibration = {"a": float(m["a"]), "b": float(m["b"]), "r2": float(m["r2"]),
                           "ms": [float(x) for x in m["ms"].split(",") if x.strip()],
                           "counts": counts or None, "samples": sum(counts) if counts else None}
    cut = None
    if verifies:
        cut = {"summaries": summaries, "verifies": verifies, "w_star": w_sum / verifies, "e_w_star": e_sum / verifies,
               "accept_len": acc_sum / verifies, "cut_rows": total_rows, "width_counts": widths or None,
               "sampled_verifies": sampled}
    return {"kv": kv, "late_loads": late, "dropped_uncommitted": dropped, "errors": errors, "cut": cut,
            "calibration": calibration}


def aggregate(res: dict) -> dict:
    """One arm's headline numbers from long_compare.analyse's result: whole decode and the 60-360 s windows."""
    lo, hi = STEADY_WINDOW_S
    counted_files_dropped = any("verify" in f or "device-stamps" in f for f in res.get("trace_dropped_files", []))

    def over(windows: list[dict]) -> dict:
        tokens = sum(w["tokens"] for w in windows)
        verifies = sum(w["verifies"] for w in windows)
        seconds = sum(w["t1_s"] - w["t0_s"] for w in windows)
        return {
            "seconds": seconds, "tokens": tokens, "verifies": verifies,
            "ms_per_token": seconds * 1e3 / tokens if tokens else None,
            "ms_per_verify": seconds * 1e3 / verifies if verifies else None,
            "tokens_per_verify": tokens / verifies if verifies else None,
            # Verifies and their times come from the verify and device-stamp files; a drop in another file (the copy
            # engine's, say) undercounts only the diagnostic columns, so it does not make ms per verify untrustworthy.
            "truncated": counted_files_dropped and any(w.get("truncated") for w in windows),
        }

    steady = [w for w in res["windows"] if w["t0_s"] >= lo and w["t1_s"] <= hi + 1e-6]
    return {"whole": over(res["windows"]), "steady": over(steady), "steady_complete": bool(steady) and steady[-1]["t1_s"] >= hi - 1e-6,
            "ttft_s": res["long"]["ttft_s"], "finish_reason": res["long"]["finish_reason"], "tokens_total": res["long"]["tokens"]}


def _gain(base: float | None, other: float | None) -> float | None:
    """The fraction by which ``other`` is lower than ``base``."""
    return (base - other) / base if base and other is not None else None


def compare(base: dict, other: dict) -> dict:
    g_tok = _gain(base["steady"]["ms_per_token"], other["steady"]["ms_per_token"])
    g_ver = _gain(base["steady"]["ms_per_verify"], other["steady"]["ms_per_verify"])
    return {"ms_per_token_gain": g_tok, "ms_per_verify_gain": g_ver,
            "both_clear": g_tok is not None and g_ver is not None and g_tok >= SHIP_FRACTION and g_ver >= SHIP_FRACTION}


def verdict(arms: dict[str, dict]) -> dict:
    """The ship rule applied to {"prod"|"cut"|"block": summary}. A summary needs ``agg`` (aggregate), ``log``
    (parse_server_log) and ``kv_ok``. Advisory: the owner decides."""
    prod = arms["prod"]
    out: dict = {"advisory": True, "arms": {}}
    clears = {}
    for name in ("cut", "block"):
        a = arms[name]
        vs = compare(prod["agg"], a["agg"])
        tpv_p, tpv_a = prod["agg"]["steady"]["tokens_per_verify"], a["agg"]["steady"]["tokens_per_verify"]
        reasons = []
        if not vs["both_clear"]:
            reasons.append("60-360 s ms/token and ms per verify are not both >= 5% below -prod")
        if a["log"]["errors"]:
            reasons.append(f"{len(a['log']['errors'])} error line(s) in server.log")
        if prod["log"]["errors"]:
            reasons.append("-prod's server.log has error lines")
        if tpv_a is None or tpv_p is None or tpv_a < tpv_p:
            reasons.append("tokens per verify lower than -prod")
        if not a["agg"]["steady_complete"]:
            reasons.append("the 60-360 s window is incomplete")
        if a["agg"]["steady"]["truncated"]:
            reasons.append("job trace truncated in 60-360 s: ms per verify undercounts verifies")
        if not a["kv_ok"]:
            reasons.append(f"KV is not {KV_PIN}")
        clears[name] = not reasons
        out["arms"][name] = {"vs_prod": vs, "clears_rule_vs_prod": not reasons, "reasons": reasons}
    block_vs_cut = compare(arms["cut"]["agg"], arms["block"]["agg"])
    out["block_vs_cut"] = block_vs_cut
    if clears["block"] and block_vs_cut["both_clear"]:
        out["suggestion"] = "block"
    elif clears["cut"]:
        out["suggestion"] = "cut"
    elif clears["block"]:
        # Block clears against -prod, cut-only does not, and block is not 5% over cut-only: the rule does not decide.
        out["suggestion"] = "owner: block clears against -prod but cut-only does not and block is not >= 5% over it"
    else:
        out["suggestion"] = "none"
    if out["suggestion"] == "none" and any(
        (v["vs_prod"]["ms_per_token_gain"] or 0) > 0 or (v["vs_prod"]["ms_per_verify_gain"] or 0) > 0 for v in out["arms"].values()
    ):
        out["suggestion"] = "owner: no arm clears the rule; closer results go to the owner"
    return out


def _f(v, fmt: str) -> str:
    return f"{v:{fmt}}" if v is not None else "-"


def _pct(v) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "-"


def render(arms: dict[str, dict], ver: dict) -> str:
    names = {"prod": "-prod", "cut": "-confcut-sampled", "block": "-confcut-sampled-block"}
    lines = [f"{'':34s}" + "".join(f"{names[n]:>26s}" for n in ARMS)]

    def row(label: str, get, fmt: str = ".2f") -> None:
        lines.append(f"{label:34s}" + "".join(f"{_f(get(arms[n]), fmt):>26s}" for n in ARMS))

    row("ms/token, whole decode", lambda a: a["agg"]["whole"]["ms_per_token"])
    row("ms/token, 60-360 s", lambda a: a["agg"]["steady"]["ms_per_token"])
    row("ms per verify, whole", lambda a: a["agg"]["whole"]["ms_per_verify"], ".1f")
    row("ms per verify, 60-360 s", lambda a: a["agg"]["steady"]["ms_per_verify"], ".1f")
    row("tokens per verify, whole", lambda a: a["agg"]["whole"]["tokens_per_verify"])
    row("tokens per verify, 60-360 s", lambda a: a["agg"]["steady"]["tokens_per_verify"])
    row("tokens / verifies, 60-360 s", lambda a: f"{a['agg']['steady']['tokens']}/{a['agg']['steady']['verifies']}", "s")
    row("long TTFT (s)", lambda a: a["agg"]["ttft_s"])
    row("finish reason", lambda a: a["agg"]["finish_reason"], "s")
    row("KV (max_total_num_tokens)", lambda a: a["log"]["kv"], "d")
    row("late device-loads (after serving)", lambda a: a["log"]["late_loads"], "d")
    row("dropped uncommitted verify lines", lambda a: a["log"]["dropped_uncommitted"], "d")
    row("error lines", lambda a: len(a["log"]["errors"]), "d")
    row("conf cut verifies logged", lambda a: a["log"]["cut"]["verifies"] if a["log"]["cut"] else None, "d")
    row("conf cut w* (verify-weighted)", lambda a: a["log"]["cut"]["w_star"] if a["log"]["cut"] else None)
    row("conf cut E(w*)", lambda a: a["log"]["cut"]["e_w_star"] if a["log"]["cut"] else None)
    row("conf cut accept len", lambda a: a["log"]["cut"]["accept_len"] if a["log"]["cut"] else None)
    row("conf cut rows (total)", lambda a: a["log"]["cut"]["cut_rows"] if a["log"]["cut"] else None, "d")
    row("conf cut sampled verifies", lambda a: a["log"]["cut"]["sampled_verifies"] if a["log"]["cut"] else None, "d")
    row("conf cut width counts 1..", lambda a: str(a["log"]["cut"]["width_counts"]) if a["log"]["cut"] else None, "s")
    lines.append("")
    for n in ARMS:
        c = arms[n]["log"]["calibration"]
        lines.append(f"T(w) {names[n]}: " + (
            f"a={c['a']:.4f} ms, b={c['b']:.4f} ms/row, r2={c['r2']:.4f}, samples "
            f"{c['samples'] if c['samples'] is not None else 'not logged'}"
            + (f" (per width {c['counts']})" if c["counts"] else "") if c else "no calibration line"))
    for n in ARMS:
        for e in arms[n]["log"]["errors"][:3]:
            lines.append(f"ERROR {names[n]}: {e}")
        if arms[n]["agg"]["steady"]["truncated"]:
            lines.append(f"TRUNC {names[n]}: job trace dropped events inside 60-360 s; its ms per verify undercounts verifies")
        if arms[n]["log"]["kv"] != KV_PIN:
            lines.append(f"KV {names[n]}: {arms[n]['log']['kv']}, expected {KV_PIN}")
    lines += ["", "SHIP RULE (ADVISORY; the owner decides): >= 5% on both 60-360 s ms/token and ms per verify against -prod,",
              "no errors, tokens per verify not lower; block over cut-only needs >= 5% over cut-only on both as well."]
    for n in ("cut", "block"):
        v = ver["arms"][n]
        lines.append(f"  {names[n]} vs -prod: ms/token {_pct(v['vs_prod']['ms_per_token_gain'])} faster, ms per verify "
                     f"{_pct(v['vs_prod']['ms_per_verify_gain'])} faster -> "
                     + ("clears the rule" if v["clears_rule_vs_prod"] else "does not clear: " + "; ".join(v["reasons"])))
    b = ver["block_vs_cut"]
    lines.append(f"  -confcut-sampled-block vs -confcut-sampled: ms/token {_pct(b['ms_per_token_gain'])} faster, ms per verify "
                 f"{_pct(b['ms_per_verify_gain'])} faster -> " + ("over 5% on both" if b["both_clear"] else "not >= 5% on both"))
    lines.append(f"  advisory suggestion: {ver['suggestion']}")
    return "\n".join(lines)


def summarise(d: str, stall_factor: float = 1.3, cpu_factor: float = 1.2) -> dict:
    res = long_compare.analyse(d, stall_factor, cpu_factor)
    with open(os.path.join(d, "server.log"), errors="replace") as f:
        log = parse_server_log(f.read())
    return {"dir": d, "agg": aggregate(res), "log": log, "kv_ok": log["kv"] == KV_PIN, "analysis": res}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n in ARMS:
        ap.add_argument(f"--{n}", required=True, metavar="RUN_DIR")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    arms = {n: summarise(getattr(a, n)) for n in ARMS}
    ver = verdict(arms)
    print(render(arms, ver))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"arms": arms, "verdict": ver}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
