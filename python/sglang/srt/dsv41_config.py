"""The DeepSeek-V4.1 runtime knobs outside expert streaming, resolved from the environment into one value.

``Dsv41Config.from_envs`` reads the model's own ``SGLANG_*DSV41*`` environment variables (reasoning effort, Engram,
the fused kernels, the prefill indexer, the RAM-miss fault test hook) once and returns a frozen struct, so a component
reads a field instead of the process environment. The variables are documented where they are declared, in
``sglang.srt.environ``.

The expert-streaming knobs are not here: the engine's live in ``ExpertStreamConfig`` and the DSV4.1 profile's
(MoE side stream, draft graph, prefetch margin floors and learned head, draft top-prob log) in
``Dsv41ProfileConfig``, both in ``sglang.srt.layers.moe.expert_engine.config``.
"""

from __future__ import annotations

from typing import Optional

import msgspec

from sglang.srt.environ import envs


class Dsv41Config(msgspec.Struct, frozen=True):
    """Every non-streaming ``SGLANG_*DSV41*`` knob, de-prefixed.

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
    engram_verify_in_graph: bool
    ram_miss_fault: str
    enable_layer_fusion: bool
    enable_exl3_cast_fusion: bool
    torch_prefill_indexer: bool
    torch_prefill_indexer_score_budget_mb: int
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
            engram_verify_in_graph=envs.SGLANG_DSV41_ENGRAM_VERIFY_IN_GRAPH.get(),
            ram_miss_fault=envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get(),
            enable_layer_fusion=envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.get(),
            enable_exl3_cast_fusion=envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.get(),
            torch_prefill_indexer=envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get(),
            torch_prefill_indexer_score_budget_mb=envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB.get(),
            fused_wo_a=envs.SGLANG_DSV41_FUSED_WO_A.get(),
        )
