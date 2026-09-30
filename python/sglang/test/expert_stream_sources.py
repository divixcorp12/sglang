"""The C++ sources of the expert-stream transport, for tests that read source text rather than run it."""

from pathlib import Path

MOE = Path(__file__).resolve().parents[1] / "kernels" / "jit" / "csrc" / "moe"


def host_sources() -> tuple[Path, ...]:
    """The EXL3 host instantiations (production, instrumented) first, then every transport host header."""
    return (
        MOE / "exl3_ram_miss_host.cpp",
        MOE / "exl3_ram_miss_host_instr.cpp",
        *sorted((MOE / "expert_stream" / "host").glob("*.h")),
    )


def device_sources() -> tuple[Path, ...]:
    return (MOE / "exl3_ram_miss.cuh", *sorted((MOE / "expert_stream").glob("*.cuh")))


def wire_header() -> Path:
    return MOE / "expert_stream" / "lease_layout.h"


def joined_text(paths) -> str:
    return "\n".join(path.read_text() for path in paths)
