"""Client-observed inter-token latency, from `run_capture_sessions.py`'s `chunk_times`.

**This is not `step_latency` and must never be written into that field or compared
against Task 1's `check_arm` thresholds.** `chunk_times` are HTTP client-side arrival
timestamps (`time.monotonic()` in `_stream_chat`, one per streamed reasoning/content
delta): detokenizer, JSON serialization, the socket, the client's own event loop, and
client-process scheduling are all inside every gap computed here. Task 1's
`step_latency` (`provenance.step_latency`) is engine-side, computed from the native
`/generate` endpoint's per-chunk `completion_tokens`, and `check_arm`'s thresholds are
calibrated against that engine-side quantity. Substituting a differently-defined
number into a gate with tuned thresholds is a real category of error (see README
"One harness, not two"), not a shortcut.
"""

from __future__ import annotations


def _percentile(sorted_values: list, q: float) -> float:
    # Nearest-rank, matching provenance.step_latency's own convention (not interpolated).
    return sorted_values[min(len(sorted_values) - 1, max(0, -(-len(sorted_values) * q // 100) - 1))]


def client_inter_token_latency_s(chunk_times: list) -> dict:
    """Percentiles of the gaps between successive chunk arrivals, in seconds.

    `{"unavailable": reason}` with fewer than 2 chunks (no gap to measure).
    """
    if len(chunk_times) < 2:
        return {"unavailable": f"fewer than two chunks ({len(chunk_times)})"}
    gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    ordered = sorted(gaps)
    return {
        "n": len(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }
