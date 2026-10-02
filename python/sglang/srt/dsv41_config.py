"""The DeepSeek-V4.1 runtime knobs, resolved from the environment into one value.

``Dsv41Config.from_envs`` reads every ``SGLANG_*DSV41*`` environment variable once
and returns a frozen struct, so a component reads a field instead of the process
environment. The variables are documented where they are declared, in
``sglang.srt.environ``.
"""

from __future__ import annotations

from typing import Optional

import msgspec

from sglang.srt.environ import envs


class Dsv41Config(msgspec.Struct, frozen=True):
    """Every ``SGLANG_*DSV41*`` knob, de-prefixed.

    The meaning of a field is documented on the matching ``EnvField`` in
    ``sglang.srt.environ``.
    """

    reasoning_effort: Optional[str]
    engram_host_table: bool
    engram_host_table_layout: str
    engram_table_dir: str
    engram_ram_gib: float
    engram_host_node_cache_uring: bool
    enable_engram_device_wait: bool
    expert_stream: bool
    expert_dir: str
    expert_trace_path: str
    router_capture_path: str
    ram_miss_timeout_ms: int
    ram_miss_fault: str
    enable_ram_miss_copy_engine: bool
    enable_ram_miss_sm_small_copies: bool
    enable_prefill_share: bool
    enable_moe_side_stream: bool
    torch_prefill_indexer: bool
    fused_wo_a: bool

    @classmethod
    def from_envs(cls) -> Dsv41Config:
        """Read the environment now. Never cached, so ``envs.X.override(...)`` in tests
        and late launcher edits are observed."""
        return cls(
            reasoning_effort=envs.SGLANG_DSV41_REASONING_EFFORT.get(),
            engram_host_table=envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get(),
            engram_host_table_layout=envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get(),
            engram_table_dir=envs.SGLANG_DSV41_ENGRAM_TABLE_DIR.get(),
            engram_ram_gib=envs.SGLANG_DSV41_ENGRAM_RAM_GIB.get(),
            engram_host_node_cache_uring=envs.SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING.get(),
            enable_engram_device_wait=envs.SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT.get(),
            expert_stream=envs.SGLANG_DSV41_EXPERT_STREAM.get(),
            expert_dir=envs.SGLANG_DSV41_EXPERT_DIR.get(),
            expert_trace_path=envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get(),
            router_capture_path=envs.SGLANG_DSV41_ROUTER_CAPTURE_PATH.get(),
            ram_miss_timeout_ms=envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
            ram_miss_fault=envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get(),
            enable_ram_miss_copy_engine=envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.get(),
            enable_ram_miss_sm_small_copies=envs.SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES.get(),
            enable_prefill_share=envs.SGLANG_DSV41_ENABLE_PREFILL_SHARE.get(),
            enable_moe_side_stream=envs.SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM.get(),
            torch_prefill_indexer=envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get(),
            fused_wo_a=envs.SGLANG_DSV41_FUSED_WO_A.get(),
        )
