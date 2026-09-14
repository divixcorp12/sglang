"""Debug trace of speculative verify outcomes, for graph-vs-eager parity checks.

Off unless ``SGLANG_SPEC_VERIFY_TRACE`` names a JSONL file; with more than one
distributed rank each rank writes ``<file>.rank<N>``. Each target verify then
appends one row per request: ``rid``, the request's ``verify_index``, the
``draft`` tokens it verified, the raw ``argmax`` token of every verify row,
``accept_len`` (bonus token included), the ``committed`` tokens the sampler
accepted, each row's top-1 minus top-2 logit ``gap``, and the batch sampling
flags ``all_greedy``, ``penalties``, ``logit_bias`` and ``grammar``.
``python -m sglang.srt.speculative.verify_trace REFERENCE CANDIDATE`` aligns two
traces request by request and reports each request's first divergence of any
kind and its first committed divergence.
"""

from __future__ import annotations

import json
import sys
from functools import cache
from typing import Any, Optional, Sequence

import torch

from sglang.srt.environ import envs

_FLAG_FIELDS = ("penalties", "logit_bias", "grammar")


@cache
def verify_trace_path() -> str:
    """The trace file, read once per process; empty when tracing is off."""
    path = envs.SGLANG_SPEC_VERIFY_TRACE.get()
    if (
        path
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return f"{path}.rank{torch.distributed.get_rank()}"
    return path


def verify_logit_summary(
    next_token_logits: torch.Tensor, draft_token_num: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax token and top-1 minus top-2 logit gap per verify row, shaped [bs, draft_token_num]."""
    top = torch.topk(next_token_logits.float(), 2, dim=-1)
    gaps = top.values[:, 0] - top.values[:, 1]
    return (
        top.indices[:, 0].view(-1, draft_token_num),
        gaps.view(-1, draft_token_num),
    )


def sampling_flags(sampling_info: Any, has_grammar: bool) -> dict[str, bool]:
    """Batch sampling settings under which the raw argmax need not equal the committed tokens."""
    orchestrator = getattr(sampling_info, "penalizer_orchestrator", None)
    return {
        "all_greedy": bool(sampling_info.is_all_greedy),
        "penalties": sampling_info.acc_additive_penalties is not None
        or sampling_info.acc_scaling_penalties is not None
        or bool(orchestrator is not None and orchestrator.is_required),
        "logit_bias": sampling_info.logit_bias is not None,
        "grammar": bool(has_grammar),
    }


class VerifyTraceWriter:
    """Appends verify rows to a JSONL file and numbers each request's verifies."""

    def __init__(self, path: str):
        self.path = path
        self._verify_counts: dict[str, int] = {}

    def rows(
        self,
        rids: Sequence[str],
        draft: Sequence[Sequence[int]],
        argmax: Sequence[Sequence[int]],
        gaps: Sequence[Sequence[float]],
        accept_lens: Sequence[int],
        committed: Sequence[Sequence[int]],
        flags: dict[str, bool],
    ) -> list[dict[str, Any]]:
        """One JSON-ready row per request, from host values; ``committed`` rows are -1 padded."""
        rows = []
        for rid, draft_row, argmax_row, gap_row, accept_len, committed_row in zip(
            rids, draft, argmax, gaps, accept_lens, committed
        ):
            index = self._verify_counts.get(rid, 0)
            self._verify_counts[rid] = index + 1
            rows.append(
                {
                    "rid": rid,
                    "verify_index": index,
                    "draft": [int(token) for token in draft_row],
                    "argmax": [int(token) for token in argmax_row],
                    "accept_len": int(accept_len),
                    "committed": [int(token) for token in committed_row if token >= 0],
                    "gap": [round(float(gap), 6) for gap in gap_row],
                    **flags,
                }
            )
        return rows

    def write(self, rows: Sequence[dict[str, Any]]) -> None:
        with open(self.path, "a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


@cache
def _writer(path: str) -> VerifyTraceWriter:
    return VerifyTraceWriter(path)


def record_verify_trace(
    path: str,
    reqs: Sequence[Any],
    draft_tokens: torch.Tensor,
    summary: tuple[torch.Tensor, torch.Tensor],
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    flags: dict[str, bool],
) -> None:
    """Append this verify's rows, copying everything to the host in one transfer.

    ``predict`` and ``accept_index`` are the sampler's outputs; the committed
    tokens are ``predict[accept_index]`` over the non-negative indices.
    """
    argmax, gaps = summary
    width = argmax.shape[1]
    committed = torch.where(
        accept_index >= 0, predict.reshape(-1)[accept_index.long().clamp(min=0)], -1
    )
    committed_width = committed.shape[1]
    host = torch.cat(
        [
            draft_tokens.reshape(-1, width).double(),
            argmax.double(),
            gaps.double(),
            accept_lens.reshape(-1, 1).double(),
            committed.double(),
        ],
        dim=1,
    ).tolist()
    accept_column = 3 * width
    writer = _writer(path)
    writer.write(
        writer.rows(
            [req.rid for req in reqs],
            [row[:width] for row in host],
            [row[width : 2 * width] for row in host],
            [row[2 * width : accept_column] for row in host],
            [row[accept_column] for row in host],
            [
                row[accept_column + 1 : accept_column + 1 + committed_width]
                for row in host
            ],
            flags,
        )
    )


def requests_in_order(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows by ``rid`` in order of each request's first verify, sorted by ``verify_index``.

    Request ids differ between runs; with one running request at a time the
    first-verify order identifies the same workload request in both traces.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["rid"], []).append(row)
    return [
        sorted(request_rows, key=lambda row: row["verify_index"])
        for request_rows in grouped.values()
    ]


def _comparable(rows: Sequence[dict[str, Any]]) -> bool:
    return all(
        row.get("all_greedy", False) and not any(row.get(f, True) for f in _FLAG_FIELDS)
        for row in rows
    )


def _first_difference(a: Sequence[Any], b: Sequence[Any]) -> Optional[int]:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def _divergence(
    ref: dict[str, Any], cand: dict[str, Any], field: str, row: Optional[int]
) -> dict[str, Any]:
    def gap(trace_row: dict[str, Any]) -> Optional[float]:
        return trace_row["gap"][row] if row is not None and row < len(trace_row["gap"]) else None

    return {
        "verify_index": ref["verify_index"],
        "field": field,
        "row": row,
        "reference_gap": gap(ref),
        "candidate_gap": gap(cand),
    }


def _verify_count_divergence(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if len(reference) == len(candidate):
        return None
    return {
        "verify_index": min(len(reference), len(candidate)),
        "field": "verify_count",
        "row": None,
        "reference_gap": None,
        "candidate_gap": None,
    }


def first_any_divergence(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """First verify where ``draft``, ``argmax`` or ``committed`` differ, including rows past both accept lengths.

    ``verify_count`` when the shared prefix matches but one trace has more
    verifies; ``None`` when the traces are identical.
    """
    for ref, cand in zip(reference, candidate):
        for field in ("draft", "argmax", "committed"):
            row = _first_difference(ref[field], cand[field])
            if row is not None:
                return _divergence(ref, cand, field, row)
    return _verify_count_divergence(reference, candidate)


def first_committed_divergence(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """First verify where the committed tokens or ``accept_len`` differ, scanning the whole request.

    ``field`` is ``committed`` at a differing token (``row`` is its position)
    and ``accept_len`` when only the lengths differ; ``verify_count`` when every
    shared verify commits the same tokens but the verify counts differ.
    """
    for ref, cand in zip(reference, candidate):
        row = _first_difference(ref["committed"], cand["committed"])
        shared = min(len(ref["committed"]), len(cand["committed"]))
        if row is not None and row < shared:
            return _divergence(ref, cand, "committed", row)
        if row is not None or ref["accept_len"] != cand["accept_len"]:
            return _divergence(ref, cand, "accept_len", row)
    return _verify_count_divergence(reference, candidate)


def compare_request(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Both first divergences of one request; ``comparable`` is false when any row in either trace was non-greedy or flagged."""
    return {
        "comparable": _comparable(reference) and _comparable(candidate),
        "first_any_divergence": first_any_divergence(reference, candidate),
        "first_committed_divergence": first_committed_divergence(reference, candidate),
    }


def compare_traces(
    reference_rows: Sequence[dict[str, Any]], candidate_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per workload request: verify counts in both traces and both first divergences."""
    reference = requests_in_order(reference_rows)
    candidate = requests_in_order(candidate_rows)
    report = []
    for index in range(max(len(reference), len(candidate))):
        ref = reference[index] if index < len(reference) else []
        cand = candidate[index] if index < len(candidate) else []
        report.append(
            {
                "request": index,
                "reference_verifies": len(ref),
                "candidate_verifies": len(cand),
                **compare_request(ref, cand),
            }
        )
    return report


def _load(path: str) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m sglang.srt.speculative.verify_trace REFERENCE CANDIDATE")
        return 2
    for entry in compare_traces(_load(args[0]), _load(args[1])):
        print(json.dumps(entry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
