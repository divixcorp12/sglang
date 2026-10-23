"""Which CPU expert sources run the EXL3 kernel's row-weighted tile assignment (SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT).

Imported by the launch gate, so it reads only the resolved config, never the model."""

from typing import Optional

from sglang.srt.layers.moe.expert_engine.config import Exl3CpuConfig, resolved_config

SOURCES = {
    "": frozenset(),
    "draft": frozenset({"draft"}),
    "target": frozenset({"target"}),
    "both": frozenset({"draft", "target"}),
}


def row_weighted_sources(config: Optional[Exl3CpuConfig] = None) -> frozenset[str]:
    """The sources ("draft", "target") whose layers the option weights; empty when it is unset.

    ``config`` is the EXL3 CPU config; None resolves the process's.
    """
    value = (resolved_config().exl3.cpu if config is None else config).row_weighted
    if value not in SOURCES:
        raise ValueError(
            f"SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT must be draft, target, both or empty, got {value!r}"
        )
    return SOURCES[value]
