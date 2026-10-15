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
    engram_verify_in_graph: bool
    expert_stream: bool
    expert_dir: str
    expert_trace_path: str
    router_capture_path: str
    residency_capture: bool
    verify_accept_log_path: str
    ram_miss_timeout_ms: int
    ram_miss_idle_spin_us: int
    ram_miss_copy_idle_spin_us: int
    ram_miss_spin_core: Optional[int]
    ram_miss_fault: str
    enable_ram_miss_copy_engine: bool
    enable_ram_miss_sm_small_copies: bool
    ram_hit_copy: str
    enable_lease_pdl: bool
    cpu_experts: bool
    cpu_experts_cores: str
    cpu_experts_threads: int
    cpu_experts_split: str
    cpu_experts_cpu_ms: float
    cpu_experts_link_ms: float
    cpu_experts_handoff_ms: float
    cpu_experts_retune_batches: int
    enable_cpu_experts_calibration: bool
    cpu_experts_calibration_reps: int
    cpu_experts_misses: bool
    cpu_split_miss_cut: int
    cpu_split_miss_cut_max: int
    cpu_two_stage: bool
    cpu_two_stage_w13_weights: str
    cpu_two_stage_w2_weights: str
    enable_dspark_cpu_experts: bool
    dspark_cpu_experts_cores: str
    dspark_cpu_experts_threads: int
    disable_dspark_draft_graph: bool
    dspark_draft_resident_path: str
    cpu_experts_keep_warm_us: int
    ram_prefetch: bool
    ram_prefetch_per_token: int
    ram_prefetch_per_layer: int
    ram_prefetch_spec_share: int
    ram_prefetch_top_k_only: bool
    ram_prefetch_scorer: str
    ram_prefetch_margin_floors: Optional[str]
    ram_prefetch_idle_drive: bool
    ram_prefetch_idle_deadline_us: int
    ram_prefetch_idle_spread: bool
    ram_prefetch_idle_pieces: int
    ram_prefetch_promote: bool
    ram_prefetch_horizon: int
    enable_prefill_fills: bool
    enable_prefill_share: bool
    enable_prefill_route_plan: bool
    enable_prefill_split_gather: bool
    enable_moe_side_stream: bool
    enable_layer_fusion: bool
    enable_exl3_cast_fusion: bool
    torch_prefill_indexer: bool
    torch_prefill_indexer_score_budget_mb: int
    fused_wo_a: bool
    ram_prefetch_learned_head: Optional[str] = None
    gpu_drip_slots: int = 0
    gpu_drip_picks: int = 4
    gpu_drip_chunk_kb: int = 1024

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
            expert_stream=envs.SGLANG_DSV41_EXPERT_STREAM.get(),
            expert_dir=envs.SGLANG_DSV41_EXPERT_DIR.get(),
            expert_trace_path=envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get(),
            router_capture_path=envs.SGLANG_DSV41_ROUTER_CAPTURE_PATH.get(),
            residency_capture=envs.SGLANG_DSV41_RESIDENCY_CAPTURE.get(),
            verify_accept_log_path=envs.SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH.get(),
            ram_miss_timeout_ms=envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
            ram_miss_idle_spin_us=envs.SGLANG_DSV41_RAM_MISS_IDLE_SPIN_US.get(),
            ram_miss_copy_idle_spin_us=envs.SGLANG_DSV41_RAM_MISS_COPY_IDLE_SPIN_US.get(),
            ram_miss_spin_core=envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.get(),
            ram_miss_fault=envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get(),
            enable_ram_miss_copy_engine=envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.get(),
            enable_ram_miss_sm_small_copies=envs.SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES.get(),
            ram_hit_copy=envs.SGLANG_DSV41_RAM_HIT_COPY.get(),
            enable_lease_pdl=envs.SGLANG_DSV41_ENABLE_LEASE_PDL.get(),
            cpu_experts=envs.SGLANG_DSV41_CPU_EXPERTS.get(),
            cpu_experts_cores=envs.SGLANG_DSV41_CPU_EXPERTS_CORES.get(),
            cpu_experts_threads=envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.get(),
            cpu_experts_split=envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.get(),
            cpu_experts_cpu_ms=envs.SGLANG_DSV41_CPU_EXPERTS_CPU_MS.get(),
            cpu_experts_link_ms=envs.SGLANG_DSV41_CPU_EXPERTS_LINK_MS.get(),
            cpu_experts_handoff_ms=envs.SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS.get(),
            cpu_experts_retune_batches=envs.SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES.get(),
            enable_cpu_experts_calibration=envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.get(),
            cpu_experts_calibration_reps=envs.SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS.get(),
            cpu_experts_misses=envs.SGLANG_DSV41_CPU_EXPERTS_MISSES.get(),
            cpu_split_miss_cut=envs.SGLANG_DSV41_CPU_SPLIT_MISS_CUT.get(),
            cpu_split_miss_cut_max=envs.SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX.get(),
            cpu_two_stage=envs.SGLANG_DSV41_CPU_TWO_STAGE.get(),
            cpu_two_stage_w13_weights=envs.SGLANG_DSV41_CPU_TWO_STAGE_W13_WEIGHTS.get(),
            cpu_two_stage_w2_weights=envs.SGLANG_DSV41_CPU_TWO_STAGE_W2_WEIGHTS.get(),
            enable_dspark_cpu_experts=envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS.get(),
            dspark_cpu_experts_cores=envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get(),
            dspark_cpu_experts_threads=envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.get(),
            disable_dspark_draft_graph=envs.SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH.get(),
            dspark_draft_resident_path=envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get(),
            cpu_experts_keep_warm_us=envs.SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US.get(),
            ram_prefetch=envs.SGLANG_DSV41_RAM_PREFETCH.get(),
            ram_prefetch_per_token=envs.SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN.get(),
            ram_prefetch_per_layer=envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.get(),
            ram_prefetch_spec_share=envs.SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE.get(),
            ram_prefetch_top_k_only=envs.SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY.get(),
            ram_prefetch_scorer=envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get(),
            ram_prefetch_margin_floors=envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.get(),
            ram_prefetch_idle_drive=envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE.get(),
            ram_prefetch_idle_deadline_us=envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US.get(),
            ram_prefetch_idle_spread=envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD.get(),
            ram_prefetch_idle_pieces=envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES.get(),
            ram_prefetch_promote=envs.SGLANG_DSV41_RAM_PREFETCH_PROMOTE.get(),
            ram_prefetch_horizon=envs.SGLANG_DSV41_RAM_PREFETCH_HORIZON.get(),
            enable_prefill_fills=envs.SGLANG_DSV41_ENABLE_PREFILL_FILLS.get(),
            enable_prefill_share=envs.SGLANG_DSV41_ENABLE_PREFILL_SHARE.get(),
            enable_prefill_route_plan=envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.get(),
            enable_prefill_split_gather=envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.get(),
            enable_moe_side_stream=envs.SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM.get(),
            enable_layer_fusion=envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.get(),
            enable_exl3_cast_fusion=envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.get(),
            torch_prefill_indexer=envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get(),
            torch_prefill_indexer_score_budget_mb=envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB.get(),
            fused_wo_a=envs.SGLANG_DSV41_FUSED_WO_A.get(),
            ram_prefetch_learned_head=envs.SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD.get(),
            gpu_drip_slots=envs.SGLANG_DSV41_GPU_DRIP_SLOTS.get(),
            gpu_drip_picks=envs.SGLANG_DSV41_GPU_DRIP_PICKS.get(),
            gpu_drip_chunk_kb=envs.SGLANG_DSV41_GPU_DRIP_CHUNK_KB.get(),
        )
