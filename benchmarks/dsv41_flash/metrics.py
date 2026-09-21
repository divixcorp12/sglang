"""Percentiles and per-request records for one arm's workload."""

from __future__ import annotations

import hashlib

from arm_config import import_provenance


def nearest_rank(values: list, q: float):
    """Nearest-rank percentile, q in [0, 100]; with n samples p99 is a real observation, not an interpolation."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, -(-len(ordered) * q // 100) - 1))]


def summarize(values: list) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p50": nearest_rank(values, 50),
        "p95": nearest_rank(values, 95),
        "p99": nearest_rank(values, 99),
        "max": max(values),
    }


def request_record(
    *,
    index: int,
    prompt_tokens: int,
    submit_t: float,
    log: list,
    end_t: float,
    text: str,
) -> dict:
    """One finished request from its chunk log of (arrival time, cumulative completion_tokens)."""
    if not log:
        raise RuntimeError(f"request {index} yielded no chunks")
    first_t = log[0][0]
    completion_tokens = log[-1][1]
    steps = import_provenance().step_latency(log)
    decode_s = end_t - first_t
    return {
        "index": index,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "submit_t": submit_t,
        "first_token_t": first_t,
        "end_t": end_t,
        "ttft_s": first_t - submit_t,
        "e2e_s": end_t - submit_t,
        "decode_tok_s": (completion_tokens - 1) / decode_s
        if decode_s > 0 and completion_tokens > 1
        else 0.0,
        "step_s": steps.get("step_s"),
        "multi_token_chunks": steps.get("multi_token_chunks"),
        "step_unavailable": steps.get("unavailable"),
        "output_sha1": hashlib.sha1(text.encode()).hexdigest(),
    }


def aggregate(requests: list, *, wall_s: float) -> dict:
    """Throughput and latency percentiles over the measured requests.

    tokens_per_s is completion tokens over the wall time of the whole window (first submit to last finish), so
    prefill and any queueing count; decode_tok_s_mean is the per-request rate after the first token.
    """
    if not requests:
        raise ValueError("no measured requests")
    steps = [s for r in requests if r["step_s"] for s in r["step_s"]]
    total_tokens = sum(r["completion_tokens"] for r in requests)
    return {
        "requests": len(requests),
        "completion_tokens": total_tokens,
        "wall_s": wall_s,
        "tokens_per_s": total_tokens / wall_s,
        "decode_tok_s_mean": sum(r["decode_tok_s"] for r in requests) / len(requests),
        "e2e_s": summarize([r["e2e_s"] for r in requests]),
        "ttft_s": summarize([r["ttft_s"] for r in requests]),
        "step_s": summarize(steps),
        "multi_token_chunks": sum(r["multi_token_chunks"] or 0 for r in requests),
        "steps_unavailable": sum(1 for r in requests if r["step_unavailable"]),
    }
