"""The DeepSeek-V4.1 runtime knobs, resolved from the environment into one frozen struct."""

from __future__ import annotations

import msgspec

from sglang.srt.environ import envs


class Dsv41Config(msgspec.Struct, frozen=True):
    reasoning_effort: str | None
    engram_host_table: bool
    engram_host_table_layout: str
    engram_table_dir: str
    engram_ram_gib: float
    expert_stream: bool
    expert_dir: str
    expert_trace_path: str
    ram_miss_timeout_ms: int
    ram_miss_fault: str
    enable_expert_prefetch: bool
    torch_prefill_indexer: bool
    fused_wo_a: bool

    @classmethod
    def from_envs(cls) -> "Dsv41Config":
        # Never cache the result: envs.X.override(...) in tests must be visible to the
        # next call, so every call re-reads the environment.
        return cls(
            reasoning_effort=envs.SGLANG_DSV41_REASONING_EFFORT.get(),
            engram_host_table=envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get(),
            engram_host_table_layout=envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get(),
            engram_table_dir=envs.SGLANG_DSV41_ENGRAM_TABLE_DIR.get(),
            engram_ram_gib=envs.SGLANG_DSV41_ENGRAM_RAM_GIB.get(),
            expert_stream=envs.SGLANG_DSV41_EXPERT_STREAM.get(),
            expert_dir=envs.SGLANG_DSV41_EXPERT_DIR.get(),
            expert_trace_path=envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get(),
            ram_miss_timeout_ms=envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
            ram_miss_fault=envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get(),
            enable_expert_prefetch=envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH.get(),
            torch_prefill_indexer=envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get(),
            fused_wo_a=envs.SGLANG_DSV41_FUSED_WO_A.get(),
        )
