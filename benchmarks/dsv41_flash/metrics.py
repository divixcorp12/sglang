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


def block_spread(samples: list, blocks: int) -> dict:
    """p50 and min of each of `blocks` consecutive slices of the samples, and their relative range.

    The within-arm spread: if the blocks of one arm disagree by more than the effect looked for, the window
    cannot resolve it. Consecutive slices, so a slow drift shows up as spread.
    """
    if blocks < 2 or len(samples) < 2 * blocks:
        return {
            "blocks": 0,
            "reason": f"{len(samples)} samples cannot fill {blocks} blocks",
        }
    size = len(samples) // blocks
    parts = [samples[i * size : (i + 1) * size] for i in range(blocks)]
    p50s, mins = [nearest_rank(b, 50) for b in parts], [min(b) for b in parts]

    def rel(values: list) -> float:
        return (max(values) - min(values)) / (sum(values) / len(values))

    return {
        "blocks": blocks,
        "block_size": size,
        "p50s": p50s,
        "mins": mins,
        "p50_rel_range": rel(p50s),
        "min_rel_range": rel(mins),
    }


def aggregate(
    requests: list, *, wall_s: float, discard_steps: int, blocks: int
) -> dict:
    """Throughput and per-token decode latency over the measured requests.

    The first `discard_steps` steps of every request are dropped (the hot cache is still settling after prefill)
    and the rest are pooled in request order. tokens_per_s is completion tokens over the whole window's wall
    time, prefill included.
    """
    if not requests:
        raise ValueError("no measured requests")
    steps = []
    for r in requests:
        if r["step_s"]:
            steps += r["step_s"][discard_steps:]
    total_tokens = sum(r["completion_tokens"] for r in requests)
    return {
        "requests": len(requests),
        "completion_tokens": total_tokens,
        "wall_s": wall_s,
        "tokens_per_s": total_tokens / wall_s,
        "decode_tok_s_mean": sum(r["decode_tok_s"] for r in requests) / len(requests),
        "e2e_s_mean": sum(r["e2e_s"] for r in requests) / len(requests),
        "ttft_s_mean": sum(r["ttft_s"] for r in requests) / len(requests),
        "discarded_steps_per_request": discard_steps,
        "step_s": summarize(steps),
        "step_blocks": block_spread(steps, blocks),
        "multi_token_chunks": sum(r["multi_token_chunks"] or 0 for r in requests),
        "steps_unavailable": sum(1 for r in requests if r["step_unavailable"]),
    }
