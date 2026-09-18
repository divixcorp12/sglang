"""Routing-skew summaries from per-layer expert counts."""

import numpy as np

FRACS = (0.05, 0.10, 0.20, 0.35, 0.50)


def _gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n, total = x.size, x.sum()
    if total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def skew_summary(counts: np.ndarray) -> list[dict]:
    out = []
    for row in counts:
        ordered = np.sort(row)[::-1].astype(np.float64)
        total = ordered.sum()
        out.append(
            {
                "gini": _gini(row),
                "unused_experts": int((row == 0).sum()),
                "top_frac_mass": {
                    f"{int(round(f * 100))}%": float(ordered[: max(1, int(round(f * row.size)))].sum() / total)
                    for f in FRACS
                },
            }
        )
    return out


def cache_hit_rate(counts: np.ndarray, resident_frac: float) -> list[float]:
    rates = []
    for row in counts:
        ordered = np.sort(row)[::-1].astype(np.float64)
        k = max(1, int(round(resident_frac * row.size)))
        rates.append(float(ordered[:k].sum() / ordered.sum()))
    return rates
