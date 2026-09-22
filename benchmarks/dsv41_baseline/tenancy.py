"""Card tenancy: capture at run start/end, and refuse to pair arms taken under different tenancy.

Today the card is empty, which the campaign itself calls unusual (rule 9). Two arms
recorded under different tenancy are not comparable, so `tenancy_compatible` is a hard
gate in `paired.py`, not an advisory note.
"""

from __future__ import annotations

import re
import subprocess

import msgspec

PRODUCTION_PID = 7867

# Tolerance for "the same tenancy": memory noise from unrelated small allocations,
# never enough to hide a real production launch (multiple GiB).
MEMORY_TOLERANCE_MIB = 512


class Tenancy(msgspec.Struct, frozen=True, kw_only=True):
    memory_used_mib: int
    production_running: bool
    sm_clock_mhz: int
    sm_clock_limit_mhz: int


def production_running(*, port: int = PRODUCTION_PID) -> bool:
    result = subprocess.run(
        ["ss", "-ltn", f"sport = :{port}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return "LISTEN" in result.stdout


def capture_tenancy() -> Tenancy:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,clocks.sm,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    mem_str, sm_str, sm_limit_str = (part.strip() for part in query.stdout.strip().split(","))
    return Tenancy(
        memory_used_mib=int(mem_str),
        production_running=production_running(),
        sm_clock_mhz=int(sm_str),
        sm_clock_limit_mhz=int(sm_limit_str),
    )


def parse_tenancy(fields: dict) -> Tenancy:
    return msgspec.convert(fields, Tenancy)


def tenancy_compatible(a: Tenancy, b: Tenancy) -> bool:
    if a.production_running != b.production_running:
        return False
    if a.sm_clock_limit_mhz != b.sm_clock_limit_mhz:
        return False
    return abs(a.memory_used_mib - b.memory_used_mib) <= MEMORY_TOLERANCE_MIB


_ENVIRON_KV = re.compile(r"([^=]+)=(.*)")


def parse_proc_environ(raw: bytes) -> dict[str, str]:
    """Parse `/proc/<pid>/environ`'s NUL-separated `KEY=VALUE` records."""
    out: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        match = _ENVIRON_KV.match(entry.decode("utf-8", errors="replace"))
        if match is None:
            continue
        out[match.group(1)] = match.group(2)
    return out


class EnvVerificationError(RuntimeError):
    pass


def verify_env(*, actual: dict[str, str], expected: dict[str, str]) -> None:
    """String-compare each expected var against the real process environ (rule 4).

    Verifies against `/proc/<pid>/environ`, not what the harness thinks it set: two
    flags were once silently never set for whole arms and this is the check that
    would have caught it.
    """
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise EnvVerificationError(f"env mismatch against /proc/<pid>/environ: {mismatches}")
