"""SM clock stability, not a ramp target, and per-arm clock-profile compatibility.

An earlier version of this module gated warm-up completion on the SM clock reaching
0.95 x `clocks.max.sm` (3135 MHz on this card), modeled on a GPU-bound workload that
sustains a high clock under load. Measured smoke data disproved the premise for this
one: `clocks.sm` reads ~2947-2970 MHz at rest and **~2572 MHz during active decode** —
the 190 ms NVMe wait per step, roughly half of it, showing up in the clock domain as
the GPU sitting mostly idle. This workload never sustains a high clock, so a
ramp-fraction gate would either abort every arm or, loosened enough to pass, certify
nothing.

What actually ruins a paired comparison is two arms sitting at *different* points on
whatever the workload's clock behavior is, not both sitting low. So this module checks
**stability** (successive samples agreeing with each other) during warm-up, and
**compatibility** (two arms' recorded samples agreeing with each other) at pairing
time — never a fixed target.
"""

from __future__ import annotations

import subprocess

# Provisional: from the smoke's spread (2947-2970 at rest, 2572 under decode — a much
# larger swing than either band alone). Both bands should shrink once the noise-floor
# pair (running the same config twice) measures the real per-session spread; until
# then, treat this constant as a placeholder, not a calibrated value.
STABILITY_TOLERANCE_FRACTION = 0.03
CLOCK_TOLERANCE_FRACTION = 0.03


def sample_sm_clock_mhz() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def sample_sm_clock_limit_mhz() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def is_stable(
    recent_samples_mhz: list[int], *, tolerance_fraction: float = STABILITY_TOLERANCE_FRACTION
) -> bool:
    """Whether the most recent clock samples agree with each other.

    True only once there are at least 2 samples and their spread (max - min) is within
    `tolerance_fraction` of their mean; a single sample is never "stable" on its own.
    """
    if len(recent_samples_mhz) < 2:
        return False
    mean = sum(recent_samples_mhz) / len(recent_samples_mhz)
    if mean == 0:
        return True
    spread = max(recent_samples_mhz) - min(recent_samples_mhz)
    return spread / mean <= tolerance_fraction


class ClockStabilityTimeoutError(RuntimeError):
    pass


def clock_profiles_compatible(
    a_clocks_mhz: list[int],
    b_clocks_mhz: list[int],
    *,
    tolerance_fraction: float = CLOCK_TOLERANCE_FRACTION,
) -> bool:
    """Refuse to pair arms whose per-session clock samples differ materially.

    Compares the mean sampled SM clock across each arm's sessions; a relative
    difference beyond `tolerance_fraction` means the two arms sat at different points
    of this workload's clock behavior, not just ordinary jitter under load.
    """
    if not a_clocks_mhz or not b_clocks_mhz:
        raise ValueError("no clock samples to compare")
    a_mean = sum(a_clocks_mhz) / len(a_clocks_mhz)
    b_mean = sum(b_clocks_mhz) / len(b_clocks_mhz)
    baseline = max(a_mean, b_mean)
    if baseline == 0:
        return True
    return abs(a_mean - b_mean) / baseline <= tolerance_fraction
