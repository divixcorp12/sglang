"""How many of a layer's RAM-tier experts go to the CPU (plan 2026-09-29-dsv41-cpu-experts, section 2)."""

from typing import Sequence

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


def split_from_grid(grid: Sequence[Sequence[float]], tie: float = 0.02) -> list[int]:
    """``split[n]`` from the startup calibration's grid: per n, the k with the least measured layer time.

    ``grid[1 + n][k]`` is the ms for n lanes with k on the CPU, measured with both paths running together. A k within
    ``tie`` of the best wins over a smaller one: equal layer time, and the link stays free for the GPU's own misses.
    """
    lanes = len(grid) - 2
    split = [0]
    for n in range(1, lanes + 1):
        times = list(grid[1 + n][: n + 1])
        best = min(times)
        split.append(max(k for k, t in enumerate(times) if t <= best * (1 + tie)))
    return split


def format_calibration(
    grid: Sequence[Sequence[float]],
    split: Sequence[int],
    *,
    row: int,
    expert_bytes: int,
    reps: int,
) -> str:
    """The calibration's report: the CPU and link tables, each n's layer time at its chosen k, and the split."""
    lanes = len(grid) - 2

    def ms(values) -> str:
        return " ".join(f"{v:.2f}" for v in values)

    layer = [grid[1 + n][split[n]] for n in range(1, lanes + 1)]
    return "\n".join(
        [
            f"CPU experts calibration: row {row}, expert {expert_bytes / 2**20:.1f} MiB, {reps} reps",
            f"  cpu  ms k=1..{lanes}: {ms(grid[0][1:])}",
            f"  link ms m=1..{lanes}: {ms(grid[1][1:])}",
            f"  layer ms n=1..{lanes} at chosen k: {ms(layer)}",
            f"  split n=0..{lanes}: " + " ".join(str(k) for k in split),
        ]
    )
