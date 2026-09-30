"""How many of a layer's RAM-tier experts go to the CPU (plan 2026-09-29-dsv41-cpu-experts, section 2)."""

import torch


def parse_core_list(spec: str) -> list[int]:
    """Cores from a taskset-style list such as "36-47,50"."""
    cores: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        cores.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return sorted(set(cores))


def k_star(n: int, c_cpu_ms: float, c_link_ms: float, handoff_ms: float) -> int:
    """How many of a layer's ``n`` RAM-tier experts go to the CPU.

    Minimizes max(handoff + k * c_cpu, (n - k) * c_link) over k = 0..n. A tie goes to the larger k:
    equal layer time, and the link's bandwidth stays free for the GPU's own copies.
    """
    best_k, best = 0, max(handoff_ms, n * c_link_ms)
    for k in range(1, n + 1):
        cost = max(handoff_ms + k * c_cpu_ms, (n - k) * c_link_ms)
        if cost <= best:
            best_k, best = k, cost
    return best_k


def split_table(
    max_n: int, c_cpu_ms: float, c_link_ms: float, handoff_ms: float
) -> torch.Tensor:
    """``k_star(n)`` for n = 0..max_n, int32 ``[max_n + 1]``, for a device-side lookup."""
    return torch.tensor(
        [k_star(n, c_cpu_ms, c_link_ms, handoff_ms) for n in range(max_n + 1)],
        dtype=torch.int32,
    )
