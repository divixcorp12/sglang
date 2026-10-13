"""L+2 cap-1 report: assembles result.json with a label on every section and renders REPORT.md (C12, C14, D4, N4).

Inputs: the pieces the driver produced (sanity, l2_cap1_core result, uncovered groups, l2_cap1_timing replay, cost
dicts, frozen candidates, group counts) plus commands, checkpoint hashes and paths. A missing piece renders `pending`.
Output: result.json and REPORT.md under one output directory, written atomically.
Every section is labelled 'observed offline', 'replay estimate' or 'hypothesis'. No ordinary-decode dataset exists,
so nothing in the report measures ordinary decode.

Usage: result = build_result(run_dir=..., out=..., commands=..., plan_sha=..., ckpts=..., sanity=..., cap1=...,
                             uncovered=..., timing=..., costs=..., frozen=..., group_counts=...)
       write_outputs(out, result)
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_cap1_timing  # noqa: E402

OBSERVED = "observed offline"
REPLAY = l2_cap1_timing.LABEL
HYPOTHESIS = "hypothesis"
LABELS = (OBSERVED, REPLAY, HYPOTHESIS)
PENDING = "pending"
READ_NOTHING = "inf (read nothing)"
NO_ORDINARY_DECODE = ("No ordinary-decode dataset exists, so nothing in this report measures ordinary decode; every "
                      "row is a DSpark draft-verify forward.")
COVERED_TEXT = ("Headline numbers use only the target layers the serving path can cover (biased target gate, "
                "layer+2 streamed): 38 of 40 rows. Other groups are reported separately and enter no headline number.")
COST_SCOPE = ("six live tokens, CUDA graph; complete = head plus gate (bf16 median) on the scorer's own L+2 "
              "checkpoint; gate-only = the native gate alone")
D4_TEXT = ("Per source record the serving path publishes at most one entry: the block head's top available expert, "
           "with its block score in the margin field. The calibrated global threshold is expressed in that block "
           "score. Not yet implemented in serving.")
_RESULT_NAME = "result.json"
_REPORT_NAME = "REPORT.md"
_SCOPE_TEXT = "L+2 learned heads (T0, T3) versus the native gate under the serving read policy: cap 1 per layer."


def build_result(*, run_dir: str, out: str, commands: list[str], plan_sha: str, ckpts: dict[str, dict],
                 sanity: dict, cap1: dict, uncovered: dict, timing: dict, costs: dict, frozen: dict,
                 group_counts: dict) -> dict:
    """Every top-level value is a dict carrying a `label` from LABELS."""
    artifacts = {"result_json": os.path.join(out, _RESULT_NAME), "report_md": os.path.join(out, _REPORT_NAME),
                 "run_dir": run_dir, "out": out}
    return {
        "scope": {"label": OBSERVED, "text": _SCOPE_TEXT, "covered": COVERED_TEXT,
                  "ordinary_decode": NO_ORDINARY_DECODE},
        "provenance": {"label": OBSERVED, "commands": list(commands), "plan_sha256": plan_sha,
                       "checkpoints": ckpts, "artifacts": artifacts},
        "sanity": {"label": OBSERVED, "result": sanity},
        "groups": {"label": OBSERVED, "counts": group_counts},
        "cap1": {"label": OBSERVED, "splits": cap1},
        "uncovered": {"label": OBSERVED, "splits": uncovered},
        "costs": {"label": OBSERVED, "scope": COST_SCOPE, "scorers": costs},
        "timing": {"label": REPLAY, "measured": timing.get("measured"), "zero_delay": timing.get("zero_delay")},
        "frozen_candidate": {"label": REPLAY, "scorers": frozen},
        "serving_interface": {"label": HYPOTHESIS, "text": D4_TEXT},
    }


def _fmt(value: object) -> str:
    if value is None:
        return PENDING
    if isinstance(value, float):
        if value == math.inf:
            return READ_NOTHING
        return f"{value:.4f}" if math.isfinite(value) else PENDING
    return str(value)


def _table(headers: list[str], rows: list[list[object]], *, label: str) -> list[str]:
    head = [*headers, "label"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([*(_fmt(c) for c in row), label]) + " |" for row in rows]
    return [*lines, ""]


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body]


def _scope_lines(result: dict) -> list[str]:
    s = result["scope"]
    labels = ", ".join(f"`{x}`" for x in LABELS)
    return _section("Scope and labels", [s["text"], "", f"Labels: {labels}.", "", s["ordinary_decode"], ""])


def _sanity_lines(result: dict) -> list[str]:
    s = result["sanity"]["result"]
    if not s:
        return _section("C2 sanity (cap 2 against spike-l2)", [PENDING, ""])
    rows = [[k, json.dumps(_jsonable(v), sort_keys=True)] for k, v in s.items()]
    return _section("C2 sanity (cap 2 against spike-l2)", _table(["field", "value"], rows, label=OBSERVED))


def _group_lines(result: dict) -> list[str]:
    counts = result["groups"]["counts"]
    rows = [[sp, c.get("covered"), c.get("uncovered"),
             c["covered"] + c["uncovered"] if c.get("covered") is not None and c.get("uncovered") is not None
             else None] for sp, c in counts.items()]
    body = [COVERED_TEXT, "", *_table(["split", "covered", "uncovered", "total"], rows, label=OBSERVED)]
    unc = result["uncovered"]["splits"]
    body += ["Uncovered groups (separate; in no headline number):", "", "```json",
             json.dumps(_jsonable(unc), indent=2, sort_keys=True) if unc else PENDING, "```", ""]
    return _section("Group counts", body)


def _budget_rows(scorer: str, budgets: dict) -> list[list[object]]:
    rows = []
    for key, b in budgets.items():
        p, c = b["point"], b["coverage"]
        # A produced point stores threshold inf as None (the _finite convention): read nothing, not pending.
        threshold = math.inf if p["threshold"] is None else p["threshold"]
        rows.append([scorer, key, threshold, p["mean_reads"], p["reads"], p["useful"], p["wasted"],
                     p["recall"], p["precision"], c["abstention_rate"], c["fully_covered"], c["partly_covered"]])
    return rows


def _split_lines(result: dict) -> list[str]:
    head = ["scorer", "budget (target mean reads)", "threshold", "achieved mean reads", "reads", "useful", "wasted",
            "recall", "precision", "abstention", "fully covered", "partly covered"]
    splits = result["cap1"]["splits"]
    body: list[str] = []
    for split, scorers in splits.items():
        rows = [r for name, ev in scorers.items() for r in _budget_rows(name, ev["budgets"])]
        body += [f"### {split} - cap 1 operating points", "", *_table(head, rows, label=OBSERVED)]
    return _section("Operating points per scorer", body or [PENDING, ""])


def _matched_rows(scorers: dict) -> list[list[object]]:
    rows = []
    for name, ev in scorers.items():
        for key, b in ev["budgets"].items():
            if "matched" not in b:
                continue
            m, bt = b["matched"], b["matched"]["bootstrap"]
            rows.append([name, key, m["target_reads"], m["achieved_reads"], bt["diff"], bt["lo"], bt["hi"],
                         bt.get("useful_diff"), bt.get("useful_lo"), bt.get("useful_hi"), bt["precision_diff"]])
    return rows


def _matched_lines(result: dict) -> list[str]:
    head = ["scorer", "budget", "learned reads (target)", "gate achieved reads", "recall diff", "recall lo",
            "recall hi", "useful diff", "useful lo", "useful hi", "precision diff (never alone)"]
    body: list[str] = ["Learned minus gate at the gate's curve point nearest the learned head's achieved reads; "
                       "95% paired whole-session bootstrap CIs, 10,000 resamples.", ""]
    for split, scorers in result["cap1"]["splits"].items():
        body += [f"### {split}", "", *_table(head, _matched_rows(scorers), label=OBSERVED)]
    return _section("Learned vs gate at matched reads", body)


def _dspark_rows(scorers: dict, *, kind: str, cols: tuple[str, ...]) -> list[list[object]]:
    rows = []
    for name, ev in scorers.items():
        for key, b in ev["budgets"].items():
            for k, v in b["dspark"][kind].items():
                rows.append([name, key, k, *(v[c] for c in cols)])
    return rows


def _dspark_lines(result: dict) -> list[str]:
    kinds = {"rowsets": ("anchor/draft/shared row sets", "rowset", ("eligible_all", "useful", "recall")),
             "positions": ("coverage by draft position (0 = anchor)", "position",
                           ("miss_rows", "covered", "coverage")),
             "widths": ("coverage by live block width", "width", ("eligible_all", "useful", "recall"))}
    body = ["Shared experts count once.", ""]
    for split, scorers in result["cap1"]["splits"].items():
        for kind, (title, key, cols) in kinds.items():
            rows = _dspark_rows(scorers, kind=kind, cols=cols)
            body += [f"### {split} - {title}", "", *_table(["scorer", "budget", key, *cols], rows, label=OBSERVED)]
    return _section("DSpark breakdown", body)


def _split_d3_lines(result: dict) -> list[str]:
    head = ["scorer", "budget", "side", "learned useful", "gate useful", "diff"]
    body = ["Useful reads of each learned head and of the matched gate, inside and outside the gate's top 12.", ""]
    for split, scorers in result["cap1"]["splits"].items():
        rows = [[n, k, side, g["cand"], g["gate"], g["diff"]]
                for n, ev in scorers.items() for k, b in ev["budgets"].items() if "matched" in b
                for side, g in b["matched"]["gain_split"].items()]
        body += [f"### {split}", "", *_table(head, rows, label=OBSERVED)]
    return _section("D3 matched-candidate split", body)


def _cost_lines(result: dict) -> list[str]:
    c = result["costs"]
    head = ["scorer", "cap", "tokens", "complete bf16 median ms", "gate-only ms", "param bytes", "buffer bytes",
            "selection matches policy"]
    rows = [[n, d.get("cap"), d.get("tokens"), d.get("gate_ms"), d.get("gate_only_ms"), d.get("param_bytes"),
             d.get("buffer_bytes"), d.get("selection_matches_policy")] for n, d in c["scorers"].items()]
    body = [f"Scope: {c['scope']}.", "", *_table(head, rows, label=OBSERVED)] if rows else [PENDING, ""]
    return _section("Scorer cost at cap 1", body)


def _timing_rows(replay: dict) -> list[list[object]]:
    rows = []
    for part in ("cases", "calib"):
        for case, scorers in replay[part].items():
            for name, cell in scorers.items():
                rows.append([part, case, name, cell["on_time_useful"], cell["late_useful"], cell["useful"],
                             cell["wasted"], cell["reads"], cell["recall"]])
    return rows


def _timing_lines(result: dict) -> list[str]:
    head = ["split", "contention", "scorer", "on-time useful", "late useful", "useful", "wasted", "reads", "recall"]
    body: list[str] = []
    for variant in ("measured", "zero_delay"):
        replay = result["timing"][variant]
        body.append(f"### {variant.replace('_', ' ')} scorer delay")
        body.append("")
        if replay is None:
            body += [PENDING, ""]
            continue
        delays = replay["assumptions"]["scorer_delay_ms"]
        body += [f"Scorer delay ms: `{json.dumps(_jsonable(delays), sort_keys=True)}`", "",
                 *_table(head, _timing_rows(replay), label=REPLAY)]
    return _section("Timing replay at cap 1", body)


def _frozen_lines(result: dict) -> list[str]:
    scorers = result["frozen_candidate"]["scorers"]
    if not scorers:
        return _section("Frozen candidate per scorer", [PENDING, ""])
    rows = [[n, f["budget"], f["reference"], f["tolerance"],
             json.dumps(_jsonable(f["on_time_useful_by_budget"]), sort_keys=True), f["rule"]]
            for n, f in scorers.items()]
    head = ["scorer", "budget", "reference on-time useful", "tolerance", "dev on-time useful by budget", "rule"]
    note = "Chosen on dev nominal on-time useful rows; precision is not a criterion."
    return _section("Frozen candidate per scorer", [note, "", *_table(head, rows, label=REPLAY)])


def _provenance_lines(result: dict) -> list[str]:
    p = result["provenance"]
    ck = [[n, c.get("path"), c.get("sha")] for n, c in p["checkpoints"].items()]
    art = [[k, v] for k, v in p["artifacts"].items()]
    body = ["```", *p["commands"], "```", "", f"plan sha256: `{_fmt(p['plan_sha256'])}`", "",
            *_table(["checkpoint", "path", "sha256"], ck, label=OBSERVED),
            *_table(["artifact", "path"], art, label=OBSERVED)]
    return _section("Commands, hashes and artifacts", body)


def render_report(result: dict) -> str:
    d4 = _section("D4 Stage-3 interface", [f"Label: {HYPOTHESIS}. {result['serving_interface']['text']}", ""])
    parts = [["# L+2 learned heads vs native gate at cap 1 (Stage 1)", ""], _scope_lines(result),
             _sanity_lines(result), _group_lines(result), _split_lines(result), _matched_lines(result),
             _dspark_lines(result), _split_d3_lines(result), _cost_lines(result), _timing_lines(result),
             _frozen_lines(result), d4, _provenance_lines(result)]
    return "\n".join(line for part in parts for line in part) + "\n"


def _jsonable(value: object) -> object:
    """Plain JSON types; non-finite floats become null and int keys become strings."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_outputs(out: str, result: dict) -> tuple[str, str]:
    os.makedirs(out, exist_ok=True)
    result_path, report_path = os.path.join(out, _RESULT_NAME), os.path.join(out, _REPORT_NAME)
    _atomic_write(result_path, json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(report_path, render_report(result))
    return result_path, report_path
