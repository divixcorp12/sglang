"""The C++ sources of the expert-stream transport, for tests that read source text rather than run it."""

from pathlib import Path

MOE = Path(__file__).resolve().parents[1] / "kernels" / "jit" / "csrc" / "moe"


def host_sources() -> tuple[Path, ...]:
    """The engine's host translation unit (one file for the production and instrumented builds) first, then every
    transport host header."""
    return (
        MOE / "expert_stream" / "engine_host.cpp",
        *sorted((MOE / "expert_stream" / "host").glob("*.h")),
    )


def device_sources() -> tuple[Path, ...]:
    """Every device header of the transport and the engine's device translation unit (engine_device.cuh)."""
    return tuple(sorted((MOE / "expert_stream").glob("*.cuh")))


def wire_header() -> Path:
    return MOE / "expert_stream" / "lease_layout.h"


def joined_text(paths) -> str:
    return "\n".join(path.read_text() for path in paths)
