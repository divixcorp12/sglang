"""Does storage read overlap CPU packing? Per-request causal timeline from RAM-miss stage traces.

Task 4's gate is "timeline proves I/O and packing overlap". The claim is per row, not a sorted
list of stages: row A packing before the last completion of row B is the expected overlap.

Input is the JSONL the stream trace writes (SGLANG_DSV41_EXPERT_TRACE_PATH); only its
``kind == "ram_miss_request"`` lines are read. Nothing is written back.

What a schema allows (Exl3StreamTrace.RAM_MISS_TRACE_SCHEMA; spans are NOT comparable across it):
  1 (no ``schema`` field): stamps cover the first io_uring batch only and packing follows the last
    completion by construction. It has no per-row or per-extent stamps, so overlap cannot be
    shown from it and the file is refused, not approximated.
  2: ``row_pack_ns`` (row, start, end) and ``extent_cqe_ns`` (row, part, cqe). Overlap and the
    per-row chain ``extent cqe <= row pack start`` can be checked.
  3: adds ``row_pack_ns[].admit`` and ``extent_cqe_ns[].submit/attempts``: the chain admit <=
    extent submit <= extent cqe <= pack start, and per-row queueing.
  4: adds ``request.lanes``. 5: adds ``request.pack_workers`` / ``request.pack_split``, the packing mode.
  6: adds ``request.piece_stream``, ``extent_cqe_ns[].sub`` and ``pieces``; with piece streaming an extent is one
    sub-read of a part, and a row is still ready at its last extent's cqe.

Packing mode. With packing workers (SGLANG_DSV41_RAM_MISS_PACK_WORKERS) a row's pack start is when a worker
woke and took a chunk, and the rows' spans overlap, so five inline metrics mean something else and are
REFUSED for a trace that has a worker-mode request (``result["refused"]`` names them, the report prints
REFUSED, main exits 2): ready_to_pack (the lag is the workers' wake-up, not a busy packer),
rows_queued_behind_the_packer (counts that wake-up), serial and saved_fraction (serial = window +
pack_total, and pack_total sums overlapping spans), saved_us (the numerator of that comparison) and
hidden_fraction_of_pack (same pack_total). Coverage and tail use the union of the intervals and the last
end, and are still emitted. A trace of schema < 5 does not record its mode: it is read as inline and the
report says it assumed so. A schema-5 line without ``request.pack_workers`` is refused, not assumed.

``cqe`` is when the reaping wait returned, not a per-completion time (io_uring has none), so two
rows whose extents were reaped together share it and cannot be ordered against each other.

Definitions, for one request with rows r (each a set of extents):
  ready_r      max cqe over row r's extents: the row's last extent was reaped
  packing_r    [row_pack start, end]
  window       [stages_ns.submit, max ready_r]: the reads' lifetime
  overlapped   row A packed while some other row B was still outstanding: ready_B > start_A
  hidden pack  the part of the packing inside the window; the rest ("exposed tail") is after the
               last completion, the only packing that a fully serial reader would not also pay
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Iterable, Optional


# A row whose packing began this long after its last extent was reaped waited for the packer, not
# for storage. Arbitrary; the lag is bimodal (sub-microsecond, or about a row's packing time).
QUEUED_BEHIND_PACKER_NS = 50_000


# Named for the keys they take out of the result; see "Packing mode" above.
WORKER_MODE_REFUSED = (
    "ready_to_pack_us",
    "rows_queued_behind_the_packer",
    "saved_fraction_of_serial",
    "saved_us_vs_pack_after_last_cqe",
    "hidden_fraction_of_pack",
)


class UnsupportedSchema(ValueError):
    pass


def pack_mode(record: dict) -> tuple:
    """(``"inline"`` or ``"workers"``, whether that was assumed) for one request line.

    Only the schema-5 field says which reader packed. A line of an older schema has none and is inline by
    assumption: the workers were barred from any run that writes a stage trace. A schema-5 line without
    the field is a broken writer, and guessing there is the bug this exists to remove."""
    request = record.get("request") or {}
    if "pack_workers" in request:
        return ("workers" if request["pack_workers"] > 0 else "inline"), False
    if record.get("schema", 1) >= 5:
        raise UnsupportedSchema(f"schema {record['schema']} line without request.pack_workers: packing mode unknown")
    return "inline", True


def _percentile(values: list, q: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, -(-len(ordered) * q // 100) - 1))]


def _summary(values: list) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p10": _percentile(values, 10),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _union_ns(intervals: list, lo: int, hi: int) -> int:
    """Total length of the union of ``intervals`` clipped to [lo, hi]."""
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if min(b, hi) > max(a, lo))
    total, end = 0, lo
    for a, b in clipped:
        a = max(a, end)
        if b > a:
            total += b - a
            end = b
    return total


def analyse_request(record: dict) -> Optional[dict]:
    """One request's overlap numbers, or ``{"skipped": reason}`` when it cannot be judged.

    A worker-mode request (see ``pack_mode``) has no ``hidden_fraction_of_pack``, ``saved_ns``,
    ``saved_fraction`` or ``ready_to_pack_ns`` key: asking for one is a KeyError, not a wrong number.

    A request is judged only when every row it read has a stamp for each of its extents and a
    packing interval: bounded arrays (``untraced``), a failed request or a touch never are."""
    rows = record.get("row_pack_ns") or []
    extents = record.get("extent_cqe_ns") or []
    untraced = record.get("untraced") or {}
    if record.get("status") not in ("served",):
        return {"skipped": f"status {record.get('status')}"}
    if untraced.get("rows") or untraced.get("extents"):
        return {"skipped": "rows or extents beyond the trace's bounds"}
    packed = [r for r in rows if r["end"] > 0]
    if len(packed) < len(rows) or not packed:
        return {"skipped": "a row never packed"}
    if any(e["cqe"] <= 0 for e in extents):
        return {"skipped": "an extent has no completion stamp"}
    submit = (record.get("stages_ns") or {}).get("submit", 0)
    if submit <= 0:
        return {"skipped": "no submit stamp"}
    mode, _ = pack_mode(record)

    ready = {}
    for extent in extents:
        ready[extent["row"]] = max(ready.get(extent["row"], 0), extent["cqe"])
    if set(ready) != {r["row"] for r in packed}:
        return {"skipped": "rows and extents do not match"}

    chain_violations = sum(1 for r in packed if ready[r["row"]] > r["start"])
    schema3_violations = 0
    queue_ns = []
    if any("admit" in r for r in packed):
        submits = {}
        for extent in extents:
            if "submit" in extent:
                submits.setdefault(extent["row"], []).append(extent["submit"])
        for r in packed:
            rows_submits = submits.get(r["row"], [])
            if not r.get("admit") or not rows_submits or min(rows_submits) < r["admit"]:
                schema3_violations += 1
            elif max(e["cqe"] for e in extents if e["row"] == r["row"]) < min(rows_submits):
                schema3_violations += 1
            else:
                queue_ns.append(min(rows_submits) - r["admit"])

    last_ready = max(ready.values())
    window = last_ready - submit
    pack_total = sum(r["end"] - r["start"] for r in packed)
    intervals = [(r["start"], r["end"]) for r in packed]
    hidden = _union_ns(intervals, submit, last_ready)
    overlapped_rows = sum(
        1 for r in packed if any(ready[other] > r["start"] for other in ready if other != r["row"])
    )
    # A reader that packed only after the last completion would pay window + pack_total; the packing
    # inside the window is what this one did not pay in series. It is not "finish time saved": a packer
    # that started late (lag) can still finish later than that baseline.
    serial = window + pack_total
    lag_ns = [r["start"] - ready[r["row"]] for r in packed]
    request_end = max(r["end"] for r in packed)
    tail = max(0, request_end - last_ready)
    result = {
        "rows": len(packed),
        "pack_mode": mode,
        "window_ns": window,
        "pack_ns": pack_total,
        "hidden_ns": hidden,
        "overlapped_rows": overlapped_rows,
        "coverage": hidden / window if window > 0 else None,
        "exposed_tail_ns": tail,
        "tail_share": tail / (request_end - submit) if request_end > submit else None,
        "chain_violations": chain_violations,
        "schema3_violations": schema3_violations,
        "queue_ns": queue_ns,
    }
    if mode == "inline":
        result.update(
            hidden_fraction_of_pack=hidden / pack_total if pack_total > 0 else None,
            saved_ns=hidden,
            saved_fraction=hidden / serial if serial > 0 else None,
            ready_to_pack_ns=lag_ns,
        )
    else:
        result["refused"] = "worker-mode packing: hidden_fraction_of_pack, saved_ns, saved_fraction, ready_to_pack_ns"
    return result


def _schema(record: dict) -> int:
    return record.get("schema", 1)


def iter_requests(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            if "ram_miss_request" not in line:
                continue
            record = json.loads(line)
            if record.get("kind") == "ram_miss_request":
                yield record


def _bucket(rows: int) -> str:
    return "2" if rows == 2 else "3-4" if rows <= 4 else "5-8" if rows <= 8 else "9+"


def analyse_file(path: str, *, example_rows: int = 4) -> dict:
    """The overlap picture of one trace file. Raises UnsupportedSchema for a schema 1 file."""
    schemas = Counter()
    modes = Counter()
    statuses = Counter()
    skipped = Counter()
    per_request = []
    chain_violations = schema3_violations = 0
    example = None
    for record in iter_requests(path):
        schemas[_schema(record)] += 1
        mode, assumed = pack_mode(record)
        modes["inline (assumed: schema < 5)" if assumed else mode] += 1
        statuses[record.get("status")] += 1
        if _schema(record) < 2:
            continue
        result = analyse_request(record)
        if "skipped" in result:
            skipped[result["skipped"]] += 1
            continue
        chain_violations += result["chain_violations"]
        schema3_violations += result["schema3_violations"]
        per_request.append(result)
        if result["rows"] >= example_rows and result["overlapped_rows"] and example is None:
            example = record
    if set(schemas) == {1}:
        raise UnsupportedSchema(
            f"{path}: schema 1 has no per-row or per-extent stamps ({sum(schemas.values())} requests); "
            "overlap cannot be shown from it"
        )
    workers = modes["workers"] > 0
    multi = [r for r in per_request if r["rows"] >= 2]
    overlapping = [r for r in multi if r["overlapped_rows"] > 0]
    by_rows = {}
    for bucket in ("2", "3-4", "5-8", "9+"):
        group = [r for r in multi if _bucket(r["rows"]) == bucket]
        if group:
            by_rows[bucket] = {
                "requests": len(group),
                "with_overlap": sum(1 for r in group if r["overlapped_rows"]),
                "coverage": _summary([r["coverage"] for r in group if r["coverage"] is not None]),
            }
            if not workers:
                by_rows[bucket]["hidden_fraction_of_pack"] = _summary([r["hidden_fraction_of_pack"] for r in group])
                by_rows[bucket]["saved_us"] = _summary([r["saved_ns"] / 1e3 for r in group])
    all_rows = sum(r["rows"] for r in multi)
    result = {
        "path": path,
        "schemas": dict(schemas),
        "pack_mode": dict(modes),
        "statuses": dict(statuses),
        "judged_requests": len(per_request),
        "single_row_requests": len(per_request) - len(multi),
        "multi_row_requests": len(multi),
        "skipped": dict(skipped),
        "requests_with_overlap": len(overlapping),
        "fraction_with_overlap": len(overlapping) / len(multi) if multi else None,
        "rows_packed_while_another_outstanding": sum(r["overlapped_rows"] for r in multi),
        "rows_in_multi_row_requests": all_rows,
        "chain_violations_cqe_after_pack_start": chain_violations,
        "schema3_chain_violations": schema3_violations,
        "rows_per_request": _summary([r["rows"] for r in multi]),
        "window_us": _summary([r["window_ns"] / 1e3 for r in multi]),
        "pack_us": _summary([r["pack_ns"] / 1e3 for r in multi]),
        "coverage_of_window_by_pack": _summary([r["coverage"] for r in multi if r["coverage"] is not None]),
        "exposed_tail_us": _summary([r["exposed_tail_ns"] / 1e3 for r in multi]),
        "tail_share_of_request": _summary([r["tail_share"] for r in multi if r["tail_share"] is not None]),
        "queue_admit_to_submit_us": _summary([x / 1e3 for r in multi for x in r["queue_ns"]]),
        "by_rows_per_request": by_rows,
        "example": example,
    }
    if workers:
        result["refused"] = {
            "metrics": list(WORKER_MODE_REFUSED),
            "reason": f"{modes['workers']} of {sum(modes.values())} requests were packed by workers "
            f"(request.pack_workers > 0): rows pack concurrently and a row's pack start follows a worker's "
            f"wake-up, so these are not comparable with an inline trace's",
        }
    else:
        result.update(
            hidden_fraction_of_pack=_summary([r["hidden_fraction_of_pack"] for r in multi]),
            saved_us_vs_pack_after_last_cqe=_summary([r["saved_ns"] / 1e3 for r in multi]),
            saved_fraction_of_serial=_summary([r["saved_fraction"] for r in multi if r["saved_fraction"] is not None]),
            ready_to_pack_us=_summary([x / 1e3 for r in multi for x in r["ready_to_pack_ns"]]),
            rows_queued_behind_the_packer=sum(
                1 for r in multi for x in r["ready_to_pack_ns"] if x > QUEUED_BEHIND_PACKER_NS
            ),
        )
    return result


def render_timeline(record: dict, width: int = 72) -> str:
    """ASCII Gantt of one request, in microseconds after its first submit: '.' a row's extents are
    outstanding (from submit to its last completion), '#' the row is packing."""
    submit = record["stages_ns"]["submit"]
    ready, packs = {}, {}
    for e in record["extent_cqe_ns"]:
        ready[e["row"]] = max(ready.get(e["row"], 0), e["cqe"])
    for r in record["row_pack_ns"]:
        packs[r["row"]] = (r["start"], r["end"])
    end = max(max(b for _, b in packs.values()), max(ready.values()))
    span = max(1, end - submit)

    def col(t):
        return min(width - 1, max(0, int((t - submit) * (width - 1) / span)))

    lines = [f"seq {record['request']['seq']}  layer {record['layer']}  schema {_schema(record)}  pack {pack_mode(record)[0]}  "
             f"span {span / 1e3:.0f} us  ('.' read outstanding, '#' packing)"]
    for row in sorted(packs):
        cells = [" "] * width
        for c in range(0, col(ready[row]) + 1):
            cells[c] = "."
        for c in range(col(packs[row][0]), col(packs[row][1]) + 1):
            cells[c] = "#"
        lines.append(
            f"row {row:>2} |{''.join(cells)}| ready +{(ready[row] - submit) / 1e3:.0f} "
            f"pack +{(packs[row][0] - submit) / 1e3:.0f}..+{(packs[row][1] - submit) / 1e3:.0f} us"
        )
    return "\n".join(lines)


def format_report(result: dict) -> str:
    def s(d, key="p50", scale=1.0, fmt="{:.1f}"):
        return "n/a" if d.get(key) is None else fmt.format(d[key] * scale)

    refused = result.get("refused")
    lines = [
        f"{result['path']}",
        f"  schema {result['schemas']}  status {result['statuses']}  pack mode {result['pack_mode']}",
        f"  judged {result['judged_requests']}  single-row {result['single_row_requests']}  "
        f"multi-row {result['multi_row_requests']}  skipped {result['skipped']}",
        f"  requests with overlap: {result['requests_with_overlap']}/{result['multi_row_requests']}"
        f" = {(result['fraction_with_overlap'] or 0):.1%}   "
        f"rows packed while another was outstanding: {result['rows_packed_while_another_outstanding']}"
        f"/{result['rows_in_multi_row_requests']}",
        f"  per-row chain violations (cqe > pack start): {result['chain_violations_cqe_after_pack_start']}",
        f"  rows/request p50 {s(result['rows_per_request'])}  window us p50 {s(result['window_us'])} "
        f"p90 {s(result['window_us'], 'p90')}  {'pack us (SUM of overlapping spans)' if refused else 'pack us'} "
        f"p50 {s(result['pack_us'])}",
        f"  window covered by packing:  p50 {s(result['coverage_of_window_by_pack'], 'p50', 100, '{:.1f}%')} "
        f"p90 {s(result['coverage_of_window_by_pack'], 'p90', 100, '{:.1f}%')}",
    ]
    if refused:
        lines.append(f"  REFUSED {', '.join(refused['metrics'])}: {refused['reason']}")
    else:
        lines += [
            f"  pack hidden inside the read window: p10 {s(result['hidden_fraction_of_pack'], 'p10', 100, '{:.0f}%')} "
            f"p50 {s(result['hidden_fraction_of_pack'], 'p50', 100, '{:.0f}%')} "
            f"p90 {s(result['hidden_fraction_of_pack'], 'p90', 100, '{:.0f}%')}",
            f"  packing not paid in series (= hidden): p50 {s(result['saved_us_vs_pack_after_last_cqe'])} us "
            f"p90 {s(result['saved_us_vs_pack_after_last_cqe'], 'p90')} us   "
            f"({s(result['saved_fraction_of_serial'], 'p50', 100, '{:.1f}%')} of the serial time at p50)",
            f"  ready -> pack start lag: p50 {s(result['ready_to_pack_us'])} us p90 {s(result['ready_to_pack_us'], 'p90')} us"
            f"   rows that waited for the packer (> {QUEUED_BEHIND_PACKER_NS // 1000} us): "
            f"{result['rows_queued_behind_the_packer']}/{result['rows_in_multi_row_requests']}",
        ]
    lines.append(
        f"  packing after the last completion (exposed tail): p50 {s(result['exposed_tail_us'])} us "
        f"p90 {s(result['exposed_tail_us'], 'p90')} us = p50 {s(result['tail_share_of_request'], 'p50', 100, '{:.0f}%')} "
        f"of submit..pack end"
    )
    if result["queue_admit_to_submit_us"].get("n"):
        lines.append(
            f"  admit -> extent submit: p50 {s(result['queue_admit_to_submit_us'])} us "
            f"p90 {s(result['queue_admit_to_submit_us'], 'p90')} us   schema-3 chain violations "
            f"{result['schema3_chain_violations']}"
        )
    for bucket, g in result["by_rows_per_request"].items():
        if refused:
            lines.append(
                f"    rows {bucket:>3}: {g['requests']:>5} requests, {g['with_overlap']:>5} overlap, "
                f"coverage p50 {s(g['coverage'], 'p50', 100, '{:.1f}%')}"
            )
        else:
            lines.append(
                f"    rows {bucket:>3}: {g['requests']:>5} requests, {g['with_overlap']:>5} overlap, "
                f"pack hidden p50 {s(g['hidden_fraction_of_pack'], 'p50', 100, '{:.0f}%')}, "
                f"saved p50 {s(g['saved_us'])} us"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", nargs="+")  # exit 0: complete; 1: nothing analysed; 2: some metrics refused
    parser.add_argument("--json", metavar="PATH", help="also write the results here")
    parser.add_argument("--example", action="store_true", help="draw one overlapping request per file")
    args = parser.parse_args(argv)
    results, refused = [], []
    for path in args.traces:
        try:
            results.append(analyse_file(path))
        except UnsupportedSchema as error:
            refused.append(str(error))
    for result in results:
        print(format_report(result))
        if args.example and result["example"] is not None:
            print(render_timeline(result["example"]))
        print()
    for message in refused:
        print("REFUSED:", message)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "example"} for r in results], f, indent=1)
    if not results:
        return 1
    # 2: something was printed but not all of it: a worker-mode trace's refused metrics are absent, not zero.
    return 2 if any("refused" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
