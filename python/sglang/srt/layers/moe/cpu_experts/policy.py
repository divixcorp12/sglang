"""Split policy for CPU experts: how many of a layer's RAM-tier experts the CPU runs.

A decode step has ``n`` RAM-tier experts to serve. The CPU computes ``k`` of them
(cost ``handoff + k * c_cpu``) while the PCIe link copies the other ``n - k`` to the GPU
(cost ``(n - k) * c_link``); the two paths run concurrently, so the layer takes the
larger of the two. This module picks ``k`` per ``n``, either from configured per-expert
costs (``k_star``, ``split_table``) or from the startup calibration's measured grid
(``split_from_grid``), and formats the calibration report.

The calibration itself runs in ``cpu_experts/service.py``; the native side is
``expert_stream/host/cpu_experts.h``.
"""

from typing import Sequence

import torch


def k_star(n: int, c_cpu_ms: float, c_link_ms: float, handoff_ms: float) -> int:
    """How many of a layer's ``n`` RAM-tier experts go to the CPU.

    Minimizes ``max(handoff + k * c_cpu, (n - k) * c_link)`` over ``k = 0..n``. A tie
    goes to the larger ``k``: the layer time is equal and the link's bandwidth stays
    free for the GPU's own copies.
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
    """``k_star(n)`` for ``n = 0..max_n`` as int32 ``[max_n + 1]``, for device use."""
    return torch.tensor(
        [k_star(n, c_cpu_ms, c_link_ms, handoff_ms) for n in range(max_n + 1)],
        dtype=torch.int32,
    )


def split_from_grid(grid: Sequence[Sequence[float]], tie: float = 0.02) -> list[int]:
    """``split[n]`` from the startup calibration's grid: per ``n``, the best ``k``.

    ``grid[1 + n][k]`` is the measured ms for ``n`` lanes with ``k`` on the CPU, both
    paths running together. A ``k`` within ``tie`` (relative) of the best wins over a
    smaller one: the layer time is equal and the link stays free for the GPU's own
    misses.
    """
    lanes = len(grid) - 2
    split = [0]
    for n in range(1, lanes + 1):
        times = list(grid[1 + n][: n + 1])
        best = min(times)
        split.append(max(k for k, t in enumerate(times) if t <= best * (1 + tie)))
    return split


def capped_split(grid, width: int, configured) -> list[int]:
    """The split from a grid measured up to `width` lanes: split_from_grid's entries 0..width, then the configured
    entries above, which a split capped at the live lanes never reads."""
    return split_from_grid([row[: width + 1] for row in grid[: width + 2]]) + list(configured[width + 1 :])


def format_calibration(
    grid: Sequence[Sequence[float]],
    split: Sequence[int],
    *,
    row: int,
    expert_bytes: int,
    reps: int,
) -> str:
    """The calibration report: CPU and link tables, layer times, the chosen split."""
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
