"""Server-argument requirements of EXL3 expert streaming (DeepSeek V4.1).

The MoE expert-caching gate imports this module the first time a launch's
quantization method resolves to ``exl3`` (the checkpoint's ``config.json``
names it). EXL3 experts stream with prefill eager and decode eager or a
breakable CUDA graph at max batch size 1 (a ``full`` decode graph is refused);
graph gather only with that breakable decode graph, reading missed rows from the
pinned host tier (never the host arena); and a ``stat`` or ``per_pass`` recorder
under dynamic residency. Speculative decoding (DSpark or otherwise) is refused with
any decode CUDA graph: it must run with the decode graph disabled, since a spec verify
step runs more than one token through scratch and RAM-miss posting sized for one.
Nothing else is required; ``--max-running-requests`` and the
overlap schedule stay free. This module runs during server-args processing,
so it imports only the gate module, ``sglang.srt.environ``, the graph-config enum
and the standard library.
"""

import dataclasses

from sglang.srt.arg_groups.expert_stream_requirements import (
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    register_expert_stream_requirements,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig

_EAGER = eager_expert_stream_requirements(
    "EXL3",
    enabled=lambda: envs.SGLANG_DSV41_EXPERT_STREAM.get(),
    enable_hint="SGLANG_DSV41_EXPERT_STREAM=1",
    # The shared eager checks still apply to EXL3. Its sole GPU residency
    # exception is the captured DIRECT path, with the graph gather as source.
    allow_gpu_residency_update=lambda: (
        envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get() == 2
        and envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.get()
    ),
)


class _EagerGraphView:
    """``cfg`` with no graph config, so the shared eager checks skip their graph rule."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def __getattr__(self, name):
        if name == "cuda_graph_config":
            return None
        return getattr(self._cfg, name)


def _check(cfg, budgets) -> None:
    """The eager checks, with decode allowed as a breakable CUDA graph at max batch size 1.

    Everything that still needs the host (the eager MoE fallback, the Engram
    file-table lookup) runs as an eager break; prefill stays eager. Decode
    ``full`` cannot work: the Engram lookup reads its ids on the host. A breakable decode
    graph also turns DSV4's alt-stream overlap off (below).
    """
    if envs.SGLANG_MOE_HOT_ASYNC_PROMOTIONS.get():
        # With SGLANG_MOE_GPU_RESIDENCY_UPDATE the in-graph updater owns the hot slots: a boundary only ranks victims
        # and the gather writes missed rows straight into them, so there is no promotion copy to defer. Without it,
        # EXL3 rows come only from the row source and promote through the pinned tier synchronously
        # (ExpertHotCache._load_reserved_in_chunks). Either way the flag would be ignored.
        raise ValueError(
            "SGLANG_MOE_HOT_ASYNC_PROMOTIONS has no effect on EXL3 experts: the in-graph GPU residency "
            "updater writes missed rows into hot slots during the gather, or, without it, promotions run "
            "synchronously through the pinned host tier; unset it"
        )
    graph = cfg.cuda_graph_config
    if not isinstance(graph, CudaGraphConfig):
        # run_resolution_pipeline's first offload pass runs before parse_cuda_graph_config,
        # while this is still the raw CLI value: the decode backend is not known yet, and
        # the pass after parsing runs every check below.
        return
    if (
        getattr(cfg, "speculative_algorithm", None) is not None
        and graph.decode.backend != Backend.DISABLED
    ):
        # Option C's graph-gather scratch and RAM-miss posting are sized for one token per
        # step; a DSpark verify runs up to block_size + 1 tokens (Phase D2).
        raise ValueError(
            "EXL3 expert caching runs DSpark verify eagerly only; pass "
            "--cuda-graph-backend-decode disabled (or --disable-cuda-graph)"
        )
    if graph.decode.backend == Backend.DISABLED:
        if envs.SGLANG_DSV41_CPU_EXPERTS.get():
            raise ValueError(
                "SGLANG_DSV41_CPU_EXPERTS computes experts inside the captured decode graph's copy wait; "
                "pass --cuda-graph-backend-decode breakable"
            )
        _EAGER.check(cfg, budgets)
        return
    if graph.decode.backend != Backend.BREAKABLE:
        raise ValueError(
            "EXL3 expert caching needs --cuda-graph-backend-decode breakable "
            "(or disabled); full decode graphs cannot run the Engram file-table lookup"
        )
    if (graph.decode.max_bs or 0) != 1:
        raise ValueError(
            "EXL3 expert caching captures decode graphs at max batch size 1 only; "
            "pass --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1"
        )
    if graph.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "EXL3 expert caching runs prefill eagerly; pass --cuda-graph-backend-prefill disabled"
        )
    if budgets.graph_gather and not budgets.pinned_budget_mb:
        raise ValueError(
            "EXL3 graph gathers read missed rows from the pinned host tier; "
            "set SGLANG_MOE_PINNED_HOST_MB"
        )
    if budgets.graph_gather and not budgets.hot_budget_mb:
        raise ValueError("EXL3 graph gathers need SGLANG_MOE_HOT_GPU_MB")
    # The shared eager check refuses graph gather; decode graphs may use it.
    _EAGER.check(_EagerGraphView(cfg), dataclasses.replace(budgets, graph_gather=False))
    _check_cpu_experts(budgets)
    # DSV4's alt-stream overlap still gives wrong output when captured in the breakable
    # decode graph, and this gate turns it off. Its mHC stats side stream was one cause
    # (forked before the MoE break, launched on after it; fixed by _refork_stats_stream:
    # the NaNs went away). A second defect remains in the first segment: the MQA alt-stream
    # prepare (MQALayer, capture-only) leaves layer 0's MoE input different from eager's,
    # cause unknown. The stats stream also runs in eager decode, so eager decode of this
    # launch loses that overlap too.
    overlap = envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP
    if overlap.get():
        if overlap.is_set():
            raise ValueError(
                "EXL3 expert caching's breakable decode graph gives wrong output with "
                "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1; unset it or set it to 0"
            )
        overlap.set(False)


def _check_cpu_experts(budgets) -> None:
    """SGLANG_DSV41_CPU_EXPERTS (plan 2026-09-29-dsv41-cpu-experts, Step B): the RAM-miss service's grant sends
    resident lanes to the CPU, the copy engine's copy wait completes them, and layer fusion's route tables and DIRECT
    commit leave them out of the fused MoE and the residency."""
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        return
    needs = [
        ("SGLANG_MOE_EXPERT_GRAPH_GATHER=1", budgets.graph_gather),
        ("SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1", envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.get()),
        ("SGLANG_DSV41_ENABLE_LAYER_FUSION=1", envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.get()),
        ("SGLANG_DSV41_CPU_EXPERTS_CORES (a taskset list)", bool(envs.SGLANG_DSV41_CPU_EXPERTS_CORES.get())),
    ]
    missing = [name for name, ok in needs if not ok]
    if missing:
        raise ValueError("SGLANG_DSV41_CPU_EXPERTS needs " + ", ".join(missing))
    if envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off":
        # The pull join rewrites a route's slot after planning; a CPU lane's route would then match no plan slot, and
        # the fused MoE would compute it on top of the CPU's partial.
        raise ValueError("SGLANG_DSV41_CPU_EXPERTS cannot run with SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE; set it to off")


exl3_expert_stream_requirements = ExpertStreamRequirements(
    "EXL3", _check, graph_gather_host_source="pinned_tier"
)
register_expert_stream_requirements(("exl3",), exl3_expert_stream_requirements)
