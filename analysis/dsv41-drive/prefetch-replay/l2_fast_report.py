"""L+2 fast-path report: result.json and REPORT.md from the profile, cost, interference and parity artifacts (C27).

Inputs: OUT/{profile,cost,interference,parity,parity-fp32}.json and OUT/commands.txt, whichever exist. A missing artifact renders
`pending`; nothing is estimated. Checkpoint sha256s are read from the artifacts, git_head and sglang.__file__ from
commands.txt.
Output: OUT/result.json and OUT/REPORT.md, written atomically. Every dict holding a number also holds a `label`
('observed offline' or 'hypothesis') and the measurement `scope`.
Nothing here measures serving: the numbers are offline, under CUDA graphs, on one RTX 5090.

Usage: python3 l2_fast_report.py --out /mnt/nvme1/nvfp4-work/f123-l2-fast-<date>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_fast_common as common  # noqa: E402

OBSERVED = common.OBSERVED
HYPOTHESIS = common.HYPOTHESIS
LABELS = (OBSERVED, HYPOTHESIS)
PENDING = "pending"
EXPECTED_PREFIX = {"L2_T0": "e6f7661d", "L2_T3": "e675e2bd"}
MODELS = tuple(EXPECTED_PREFIX)
COMPONENTS = ("gate", "features", "head_mlp", "aggregate_sigmoid", "select", "complete")
PATHS = ("native", "fused", "torch")
STATS = ("median_ms", "p90_ms", "p99_ms", "runs", "nodes")
ARTIFACTS = {"profile": "profile.json", "cost": "cost.json", "interference": "interference.json",
             "parity": "parity.json", "parity_router_fp32": "parity-fp32.json"}
BUDGET_KEY = "0.75"
_RESULT_NAME = "result.json"
_REPORT_NAME = "REPORT.md"
_META_KEYS = ("label", "scope", "scope_detail", "status")
_SCOPES = {
    "profile": "T0/T3 complete r1_cost path, six tokens, CUDA graph, one layer's inputs, eager-built reference",
    "cost": "widths 1/2/4/6, CUDA graph per layer slot replayed round-robin over 38 slots, one stream",
    "interference": "fused or native scorer replayed on a side stream under a fixed main-stream bf16 GEMM load",
    "parity": "every covered dev and calib group, frozen F109 calib thresholds, budgets 0.75 and 1.0",
    "parity_router_fp32": "same groups and thresholds as parity, spec_score case with a float32 gate weight",
}
_TEXT = ("Stage 2 offline measurements of the L+2 learned heads T0 and T3 as a serving-shaped fast path. "
         "Nothing is wired into serving; no number here measures ordinary decode.")


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stamp(node: object, *, label: str, scope: str) -> object:
    """Copy of node where every dict with a number (or its own label) carries label and scope."""
    if isinstance(node, dict):
        own = node.get("scope")
        if own is not None and not isinstance(own, str):
            node = {**node, "scope_detail": own, "scope": json.dumps(own, sort_keys=True)}
        if isinstance(node.get("scope"), str):
            scope = node["scope"]
        if node.get("label") in LABELS:
            label = node["label"]
        out = {k: _stamp(v, label=label, scope=scope) for k, v in node.items()}
        if "label" in node or any(_is_num(v) for v in node.values()):
            out["label"], out["scope"] = label, scope
        return out
    if isinstance(node, (list, tuple)):
        if node and all(_is_num(v) for v in node):
            return {"values": [_finite(v) for v in node], "label": label, "scope": scope}
        return [_stamp(v, label=label, scope=scope) for v in node]
    return _finite(node)


def _finite(v: object) -> object:
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _get(node: object, *keys: str) -> object:
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return PENDING
        node = node[k]
    return node


def _val(v: object) -> object:
    """A stats dict stands for its median; anything else is shown as is."""
    return v.get("median_ms", PENDING) if isinstance(v, dict) else v


def _parse_commands(text: str | None) -> dict:
    if text is None:
        return {"git_head": PENDING, "sglang_file": PENDING, "commands": []}
    git_head, sglang_file, commands, pending_cmd = PENDING, PENDING, [], None
    for line in text.splitlines():
        if line.startswith("git_head="):
            git_head = line.split("=", 1)[1].strip()
        elif line.startswith("sglang.__file__="):
            sglang_file = line.split("=", 1)[1].strip()
        elif line.startswith("CMD:"):
            pending_cmd = line[4:].strip()
        elif line.startswith("EXIT=") and pending_cmd is not None:
            commands.append(f"{pending_cmd}  [EXIT={line[5:].strip()}]")
            pending_cmd = None
    if pending_cmd is not None:
        commands.append(f"{pending_cmd}  [EXIT=pending]")
    return {"git_head": git_head, "sglang_file": sglang_file, "commands": commands}


def _section(data: dict | None, *, name: str) -> dict:
    if data is None:
        return {"label": OBSERVED, "status": PENDING, "scope": _SCOPES[name]}
    return {"label": OBSERVED, "scope": _SCOPES[name], **data, "status": "produced"}


def _checkpoints(*, profile, cost, interference, parity) -> dict:
    sources = {"profile": profile, "cost": cost, "interference": interference}
    out = {}
    for model in MODELS:
        found = {n: _get(d, "models", model, "checkpoint_sha256") for n, d in sources.items()}
        found["parity"] = _get(parity, "ckpts", model, "sha256")
        real = sorted({v for v in found.values() if v != PENDING})
        out[model] = {"sha256": real[0] if len(real) == 1 else (PENDING if not real else "inconsistent"),
                      "sources": found, "expected_prefix": EXPECTED_PREFIX[model],
                      "prefix_matches": (real[0].startswith(EXPECTED_PREFIX[model]) if len(real) == 1 else PENDING)}
    return out


def _incremental(cost: dict | None) -> dict:
    out = {}
    for model in MODELS:
        inc = _get(cost, "models", model, "incremental_ms")
        if inc == PENDING:
            continue
        out[model] = {w: {"incremental_ms": v, "bar_ms_per_layer": common.BAR_MS_PER_LAYER,
                          "within_bar": v <= common.BAR_MS_PER_LAYER if _is_num(v) else PENDING}
                      for w, v in inc.items()}
    return {"label": OBSERVED, "scope": _SCOPES["cost"], **out}


def _tradeoff(cost: dict | None, parity: dict | None) -> dict:
    width = str(common.WIDTHS[-1])
    rows = {}
    for model in MODELS:
        recall = _get(parity, "cases", "spec_score", "calib", model, "kernel", BUDGET_KEY, "recall_change",
                      "ref_recall")
        inc = _get(cost, "models", model, "incremental_ms", width)
        rows[model] = {"ref_recall_calib_budget_0_75": recall, f"incremental_ms_width_{width}": inc,
                       "bar_ms_per_layer": common.BAR_MS_PER_LAYER}
    best = {k: _best(rows, key=k, high=high) for k, high in
            (("ref_recall_calib_budget_0_75", True), (f"incremental_ms_width_{width}", False))}
    scope = ("F109 reference recall on calib at budget 0.75 (from the spec_score parity case) against fused "
             f"incremental ms at width {width}")
    return {"rows": {"label": OBSERVED, "scope": scope, **rows},
            "choice": {"label": HYPOTHESIS, "scope": scope, "higher_recall": best["ref_recall_calib_budget_0_75"],
                       "lower_incremental_cost": best[f"incremental_ms_width_{width}"],
                       "text": "Not decided here; D1 picks after both are measured."}}


def _best(rows: dict, *, key: str, high: bool) -> str:
    vals = {m: r[key] for m, r in rows.items() if _is_num(r[key])}
    if len(vals) != len(rows):
        return PENDING
    return (max if high else min)(vals, key=vals.get)


def build_result(*, out: str, commands: str | None, profile: dict | None, cost: dict | None,
                 interference: dict | None, parity: dict | None, parity_router_fp32: dict | None = None) -> dict:
    """result.json content; `commands` is the text of commands.txt. A None artifact is `pending`."""
    meta = _parse_commands(commands)
    cost_section = {**_section(cost, name="cost"), "incremental_vs_bar": _incremental(cost)}
    result = {
        "label": OBSERVED,
        "scope": _TEXT,
        "bar_ms_per_layer": common.BAR_MS_PER_LAYER,
        "provenance": {"label": OBSERVED, "scope": "read from commands.txt and the artifacts' own fields",
                       **meta, "checkpoints": _checkpoints(profile=profile, cost=cost, interference=interference,
                                                            parity=parity),
                       "artifacts": {n: os.path.join(out, f) for n, f in ARTIFACTS.items()}
                       | {"result_json": os.path.join(out, _RESULT_NAME),
                          "report_md": os.path.join(out, _REPORT_NAME)}},
        "profile": _section(profile, name="profile"),
        "cost": cost_section,
        "interference": _section(interference, name="interference"),
        "parity": _section(parity, name="parity"),
        "parity_router_fp32": _section(parity_router_fp32, name="parity_router_fp32"),
        "tradeoff": _tradeoff(cost, parity),
    }
    return _stamp(result, label=OBSERVED, scope=_TEXT)


def _fmt(v: object) -> str:
    if v is None:
        return PENDING
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _table(headers: list[str], rows: list[list[object]], *, label: str) -> list[str]:
    head = [*headers, "label"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([*(_fmt(c) for c in row), label]) + " |" for row in rows]
    return [*lines, ""]


def _section_lines(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body]


def _scope_lines(result: dict) -> list[str]:
    body = [result["scope"], "", f"Labels: `{OBSERVED}` (measured by a run of this harness) and `{HYPOTHESIS}` "
            "(a choice or expectation that no run has shown). `pending` marks a number no run has produced.", "",
            f"Historical bar: {_fmt(result['bar_ms_per_layer'])} ms per layer.", ""]
    return _section_lines("Scope and labels", body)


def _profile_lines(result: dict) -> list[str]:
    p = result["profile"]
    rows = [[m, c, *(_get(p, "models", m, "components", c, s) for s in STATS)]
            for m in MODELS for c in COMPONENTS]
    meta = f"Scope: {p['scope']}. Layer {_fmt(_get(p, 'layer'))}, tokens {_fmt(_get(p, 'tokens'))}."
    return _section_lines("C1 reference path profile (r1_cost complete path)",
                          [meta, "", *_table(["model", "component", *STATS], rows, label=OBSERVED)])


def _cost_lines(result: dict) -> list[str]:
    c = result["cost"]
    rows = [[m, path, w, *(_get(c, "models", m, path, str(w), s) for s in STATS)]
            for m in MODELS for path in PATHS for w in common.WIDTHS]
    inc = [[m, w, _get(c, "incremental_vs_bar", m, str(w), "incremental_ms"), common.BAR_MS_PER_LAYER,
            _get(c, "incremental_vs_bar", m, str(w), "within_bar")] for m in MODELS for w in common.WIDTHS]
    bw = [[w, _get(c, "native_bandwidth_gbps", str(w))] for w in common.WIDTHS]
    body = [f"Scope: {c['scope']}. Medians over {_fmt(_get(c, 'runs'))} replays; inputs rotate over "
            f"{_fmt(_get(c, 'slots'))} layer slots so the runs do not sit in L2.", "",
            *_table(["model", "path", "width", *STATS], rows, label=OBSERVED), "### Incremental cost over native", "",
            *_table(["model", "width", "incremental ms (fused - native)", "bar ms/layer", "within bar"], inc,
                    label=OBSERVED),
            "### Implied native bandwidth (lower bound; includes select)", "",
            *_table(["width", "GB/s"], bw, label=OBSERVED)]
    return _section_lines("C21-C25 cost by live width", body)


def _interference_lines(result: dict) -> list[str]:
    i = result["interference"]
    rows = [[m, _val(_get(i, "models", m, "fused_side")), _get(i, "models", m, "slowdown_vs_native"),
             _get(i, "models", m, "slowdown_vs_none")] for m in MODELS]
    body = [f"Scope: {i['scope']}. Width {_fmt(_get(i, 'width'))}, {_fmt(_get(i, 'runs'))} replays.", "",
            f"Main stream alone median ms: {_fmt(_val(_get(i, 'main_alone')))}; with the native scorer on the side "
            f"stream: {_fmt(_val(_get(i, 'native_side')))}.", "",
            *_table(["model", "main median ms with fused side", "slowdown vs native", "slowdown vs none"], rows,
                    label=OBSERVED)]
    main_load = _get(i, "main_load", "label")
    if main_load != PENDING:
        body += [f"Main load sizing label: `{main_load}`.", ""]
    return _section_lines("C26 side-stream interference", body)


def _cells(node: object, path: tuple[str, ...] = ()):
    """(path, parity_table) for every parity table below a parity `cases` dict."""
    if not isinstance(node, dict):
        return
    if "max_abs_diff" in node:
        yield path, node
        return
    for k, v in node.items():
        if k not in _META_KEYS:
            yield from _cells(v, (*path, k))


def _parity_rows(cells: list) -> list[list[object]]:
    rows = []
    for path, c in cells:
        rc = c.get("recall_change", {})
        rows.append([" / ".join(path), c["groups"], c["max_abs_diff"], c["p99_abs_diff"], c["top1_agree"],
                     c["read_agree"], c["disagreements"], len(c["near_threshold"]), c["far_disagreements"],
                     c.get("kernel_record_mismatch", "n/a"),
                     rc.get("ref_recall", "n/a"), rc.get("got_recall", "n/a"), rc.get("diff", "n/a")])
    return rows


def _parity_lines(result: dict, *, key: str = "parity",
                  title: str = "C13-C20 parity against the F109 reference") -> list[str]:
    p = result[key]
    cells = list(_cells(_get(p, "cases")))
    head = ["case / split / scorer / path / budget", "groups", "max |diff|", "p99 |diff|", "top-1 agree",
            "read agree", "disagreements", "near threshold", "far", "kernel record mismatch", "ref recall",
            "got recall", "recall change"]
    body = [f"Scope: {p['scope']}.", "",
            *_table(["spec_score gate weight dtype", "dtype label"],
                    [[_get(p, "gate_dtype"), _get(p, "gate_dtype_label")]], label=OBSERVED)]
    body += _table(head, _parity_rows(cells), label=OBSERVED) if cells else [PENDING, ""]
    body += _near_lines(cells) + _far_lines(cells) + _candidate_lines(cells)
    return _section_lines(title, body)


def _near_lines(cells: list) -> list[str]:
    rows = [[" / ".join(path), d.get("group"), d.get("session"), d.get("ref_read"), d.get("got_read"),
             d.get("ref_score"), d.get("got_score")]
            for path, c in cells for d in c["near_threshold"]]
    body = ["### Near-threshold disagreements (|score - threshold| <= 1e-3)", ""]
    return body + (_table(["where", "group", "session", "ref read", "got read", "ref score", "got score"], rows,
                          label=OBSERVED) if rows else ["none", ""])


def _far_lines(cells: list) -> list[str]:
    rows = [[" / ".join(path), c["far_disagreements"]] for path, c in cells if c["far_disagreements"]]
    body = ["### Far disagreements (more than 1e-3 from the threshold)", ""]
    return body + (_table(["where", "count"], rows, label=OBSERVED) if rows else ["none", ""])


def _candidate_lines(cells: list) -> list[str]:
    rows = [[" / ".join(path), c["disagreements"], c["recall_change"].get("ref_recall"),
             c["recall_change"].get("got_recall"), c["recall_change"].get("diff")]
            for path, c in cells
            if path[0].startswith("spec_score") and c["disagreements"] and "recall_change" in c]
    body = ["### Precision candidates (spec_score gate scores that change a read)", ""]
    return body + (_table(["where", "read changes", "ref recall", "got recall", "recall change"], rows,
                          label=OBSERVED) if rows else ["none reported", ""])


def _tradeoff_lines(result: dict) -> list[str]:
    t = result["tradeoff"]
    width = common.WIDTHS[-1]
    rows = [[m, _get(t, "rows", m, "ref_recall_calib_budget_0_75"), _get(t, "rows", m, f"incremental_ms_width_{width}"),
             common.BAR_MS_PER_LAYER] for m in MODELS]
    c = t["choice"]
    body = [f"Scope: {t['rows']['scope']}.", "",
            *_table(["model", "F109 reference recall (calib, budget 0.75)", f"fused incremental ms (width {width})",
                     "bar ms/layer"], rows, label=OBSERVED),
            f"Choice (label: `{HYPOTHESIS}`): higher recall {c['higher_recall']}; lower incremental cost "
            f"{c['lower_incremental_cost']}. {c['text']}", ""]
    return _section_lines("D1 trade-off: accuracy against incremental cost", body)


def _provenance_lines(result: dict) -> list[str]:
    p = result["provenance"]
    ck = [[m, p["checkpoints"][m]["sha256"], p["checkpoints"][m]["expected_prefix"],
           p["checkpoints"][m]["prefix_matches"]] for m in MODELS]
    art = [[k, v] for k, v in p["artifacts"].items()]
    cmds = p["commands"] or [PENDING]
    body = [f"git_head: `{p['git_head']}`", "", f"sglang.__file__: `{p['sglang_file']}`", "", "```", *cmds, "```", "",
            *_table(["checkpoint", "sha256", "expected prefix", "prefix matches"], ck, label=OBSERVED),
            *_table(["artifact", "path"], art, label=OBSERVED)]
    return _section_lines("Commands, hashes and artifacts", body)


def render_report(result: dict) -> str:
    parts = [["# L+2 learned heads: serving-shaped fast path (Stage 2)", ""], _scope_lines(result),
             _profile_lines(result), _cost_lines(result), _interference_lines(result), _parity_lines(result),
             _parity_lines(result, key="parity_router_fp32",
                           title="Q1 parity with float32 gate weight (router_fp32 follow-up run)"),
             _tradeoff_lines(result), _provenance_lines(result)]
    return "\n".join(line for part in parts for line in part) + "\n"


def _write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_outputs(out: str, result: dict) -> tuple[str, str]:
    os.makedirs(out, exist_ok=True)
    result_path, report_path = os.path.join(out, _RESULT_NAME), os.path.join(out, _REPORT_NAME)
    common.write_json(result_path, result)
    _write_text(report_path, render_report(result))
    return result_path, report_path


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_text(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = common.guard_out(args.out)
    result = build_result(out=out, commands=_read_text(os.path.join(out, "commands.txt")),
                          **{n: _read_json(os.path.join(out, f)) for n, f in ARTIFACTS.items()})
    result_path, report_path = write_outputs(out, result)
    print(result_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
