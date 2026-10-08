"""Which CPU expert sources run the EXL3 kernel's row-weighted tile assignment (SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT).

Imported by the launch gate, so it reads only the environment."""

from sglang.srt.environ import envs

SOURCES = {
    "": frozenset(),
    "draft": frozenset({"draft"}),
    "target": frozenset({"target"}),
    "both": frozenset({"draft", "target"}),
}


def row_weighted_sources() -> frozenset[str]:
    """The sources ("draft", "target") whose layers the option weights; empty when it is unset."""
    value = envs.SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT.get()
    if value not in SOURCES:
        raise ValueError(
            f"SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT must be draft, target, both or empty, got {value!r}"
        )
    return SOURCES[value]
