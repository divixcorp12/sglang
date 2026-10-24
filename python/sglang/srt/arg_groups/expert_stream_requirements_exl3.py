"""Server-argument requirements of EXL3 expert streaming (DeepSeek V4.1).

The MoE expert-caching gate imports this module the first time a launch's
quantization method resolves to ``exl3`` (the checkpoint's ``config.json`` names it).
It registers :data:`exl3_expert_stream_requirements`, which requires:

* Prefill eager; decode eager or a breakable CUDA graph at max batch size 1. A
  ``full`` decode graph is refused, because the Engram file-table lookup reads its
  ids on the host.
* Graph gather only with that breakable decode graph, reading missed rows from the
  pinned host tier (never the host arena). It needs DIRECT residency: the GPU
  residency update with ``SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE=2``.
* A ``stat`` or ``per_pass`` recorder under dynamic residency.
* Speculative decoding as DSpark: its verify eager (decode graphs disabled), or in the breakable decode graph on DIRECT
  residency at 1-64 miss lanes with a static verify (§33.8). The target's CPU experts serve that graphed verify; with
  SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES (and a lane per route: MISS_LANES unset) the lanes past the victims
  are theirs (spill).
* A copy mode the native engine has: ``sm_copy`` or ``copy_engine``, never the legacy streamer's ``dma``.

Nothing else is required; ``--max-running-requests`` and the overlap schedule stay
free. The shared eager checks come from ``eager_expert_stream_requirements``. The
settings come from the process's resolved config (``ExpertStreamConfig``, the DSV4.1
profile and the EXL3 format config). This module runs during server-args processing,
so it imports only the gate module, ``sglang.srt.environ``, the engine config, the
graph-config enum and the standard library.
"""

import dataclasses
import json
import os

from sglang.srt.arg_groups.expert_stream_requirements import (
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    register_expert_stream_requirements,
)
from sglang.srt.environ import InsertOnMissStage, envs
from sglang.srt.layers.moe.cpu_experts.assignment import row_weighted_sources
from sglang.srt.layers.moe.cpu_experts.draft_resident import load_resident_set
from sglang.srt.layers.moe.cpu_experts.threading_config import (
    check_not_reserved,
    parse_cpu_list,
)
from sglang.srt.layers.moe.expert_engine.config import (
    CopyMode,
    PullMode,
    resolved_config,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig

_DIRECT = InsertOnMissStage.DIRECT


def _direct_residency(stream) -> bool:
    """DIRECT residency: the GPU residency update at insert-on-miss stage 2."""
    return stream.residency.gpu_update and stream.residency.insert_on_miss_stage == _DIRECT


_EAGER = eager_expert_stream_requirements(
    "EXL3",
    enabled=lambda: resolved_config().stream.enabled,
    enable_hint="SGLANG_EXPERT_STREAM_ENABLE=1",
    # The shared eager checks still apply to EXL3. Its sole GPU residency
    # exception is the captured DIRECT path, with the graph gather as source.
    allow_gpu_residency_update=lambda: (
        resolved_config().stream.residency.insert_on_miss_stage == _DIRECT
        and resolved_config().stream.copy.graph_gather
    ),
)


def _group_sets(env_name: str, key: str) -> bool:
    """Whether group env ``env_name``'s JSON object names ``key``, which the decoded struct cannot tell from a default.

    The config's resolution has already decoded the object, so it is valid JSON here.
    """
    raw = getattr(envs, env_name).get().strip()
    return bool(raw) and key in json.loads(raw)


class _EagerGraphView:
    """``cfg`` with no graph config, so the shared eager checks skip their graph rule.

    Used when a breakable decode graph is allowed: every other attribute passes
    through to the wrapped ``cfg``.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def __getattr__(self, name):
        if name == "cuda_graph_config":
            return None
        return getattr(self._cfg, name)


def check_dspark_cpu_experts(cfg) -> None:
    """The draft CPU experts' launch rules, so a bad core list or resident file fails here, not at the first draft
    call."""
    resolved = resolved_config()
    stream, draft = resolved.stream, resolved.stream.draft
    if getattr(cfg, "speculative_algorithm", None) != "DSPARK":
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS computes the DSpark draft's routed experts on the CPU; "
            "pass --speculative-algorithm DSPARK or unset it"
        )
    if stream.cpu_experts.enabled and (draft.cores or draft.threads):
        raise ValueError(
            "the DSpark draft shares the GPU node's CPU expert team under SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS; "
            "unset SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_CORES and SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_THREADS"
        )
    # Unset cores are derived by ThreadingConfig when the draft starts; named ones are checked here.
    cores = parse_cpu_list(draft.cores)
    for core in cores:
        check_not_reserved(core)
    if len(cores) == 1:
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS needs SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_CORES with at "
            "least two cores (one spinning worker per core), or unset to derive them"
        )
    threads = draft.threads
    if cores and not 0 <= threads <= len(cores):
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_THREADS={threads} on {len(cores)} cores: use 0 (one per core) "
            f"up to {len(cores)}"
        )
    exl3_cpu = resolved.exl3.cpu
    if exl3_cpu.pin != "0":
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS needs SGLANG_EXL3_MOE_CPU_PIN=0: the kernel would otherwise "
            "pin its workers to the first cores"
        )
    # The kernel is the optimized residual/128 build. The recipe sets these two in either case, and
    # kernels/ops/moe/cpu_kernel_build.py refuses any other value.
    if not (exl3_cpu.act_residual and exl3_cpu.act_block == 128):
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS runs the optimized EXL3 CPU kernel: set "
            "SGLANG_EXL3_CPU_ACT_RESIDUAL=1 and SGLANG_EXL3_CPU_ACT_BLOCK=128"
        )
    if draft.resident_path:
        load_resident_set(draft.resident_path)


_EAGER_VERIFY_REMEDY = "or pass --cuda-graph-backend-decode disabled to run the DSpark verify eagerly"
# The target's CPU experts compute in the captured graph's copy wait, so its verify cannot go eager.
_CPU_EXPERTS_VERIFY_REMEDY = "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS serves the DSpark verify only in the decode graph"


def _check_row_weighted_assignment() -> None:
    """A named source's CPU experts must be on: the option would otherwise log a change no kernel runs."""
    resolved = resolved_config()
    needs = {
        "draft": ("SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS", resolved.stream.draft.cpu_experts),
        "target": ("SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS", resolved.stream.cpu_experts.enabled),
    }
    for source in sorted(row_weighted_sources(resolved.exl3.cpu)):
        name, on = needs[source]
        if not on:
            raise ValueError(
                f"SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT={resolved.exl3.cpu.row_weighted} needs "
                f"{name}=1: it weights the {source}'s EXL3 CPU expert layers"
            )


def _check_graphed_verify(cfg, remedy: str = _EAGER_VERIFY_REMEDY) -> None:
    """A speculative verify in the breakable decode graph (DSV41_REFERENCE.md §33.7-§33.8).

    Only DSpark's static verify, on DIRECT residency at W miss lanes: a verify routes more than a few lanes,
    and only DIRECT's gather flags the misses it cannot serve, which the DSpark worker re-runs eagerly.
    """
    algorithm = cfg.speculative_algorithm
    if str(algorithm).upper() != "DSPARK":
        raise ValueError(
            f"EXL3 expert caching graphs the verify of DSpark only, not {algorithm}; {remedy}"
        )
    stream = resolved_config().stream
    copy = stream.copy
    lanes = copy.graph_gather_miss_lanes
    if not (
        copy.graph_gather
        and _direct_residency(stream)
        and (1 <= lanes <= 64 or (lanes == 0 and copy.graph_gather_victim_lanes > 0))
    ):
        raise ValueError(
            "EXL3 expert caching runs a DSpark verify in the decode graph only with "
            "SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER=1, SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE=1, "
            "SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE=2 and SGLANG_EXPERT_STREAM_GRAPH_GATHER_MISS_LANES=1-64, "
            f"or unset with SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES (got {lanes}): a lane per route needs "
            f"spill's CPU lanes; {remedy}"
        )
    if envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
        raise ValueError(
            "EXL3 expert caching runs a DSpark verify in the decode graph with SGLANG_RAGGED_VERIFY_MODE=static only "
            f"(compact mode reads the host); {remedy}"
        )
    if stream.cpu_experts.enabled and not copy.graph_gather_victim_lanes:
        # The tested shape is a lane per route plus VICTIM_LANES; a narrowed MISS_LANES verify with CPU experts and no
        # spill has never run on a GPU.
        raise ValueError(
            "a graphed DSpark verify with SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS needs "
            "SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES (a lane per route: unset "
            "SGLANG_EXPERT_STREAM_GRAPH_GATHER_MISS_LANES)"
        )


def _verify_routes(cfg) -> "int | None":
    """Routes a DSpark verify gathers a layer (tokens x top-k), or None when the gate cannot know them.

    The verify has ``--speculative-num-draft-tokens``, else block size + 1, tokens; top-k is the checkpoint's
    ``num_experts_per_tok`` in a local ``config.json``. A block size left to the draft's config or a remote model path
    gives None, and the bound is left to ``GpuResidencyUpdater._init_insert_direct`` at startup.
    """
    tokens = getattr(cfg, "speculative_num_draft_tokens", None)
    if tokens is None and getattr(cfg, "speculative_dspark_block_size", None) is not None:
        tokens = int(cfg.speculative_dspark_block_size) + 1
    model_path = getattr(cfg, "model_path", None)
    if tokens is None or not isinstance(model_path, str):
        return None
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as stream:
            top_k = json.load(stream).get("num_experts_per_tok")
    except (OSError, ValueError, AttributeError):
        return None
    return int(tokens) * int(top_k) if isinstance(top_k, int) and top_k > 0 else None


def _check_victim_lanes(cfg) -> None:
    """SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES (spill): the CPU takes the lanes past the victims, and every
    route has a lane, so a verify's distinct misses never outnumber its lanes."""
    stream = resolved_config().stream
    victims = stream.copy.graph_gather_victim_lanes
    if not victims:
        return
    if str(getattr(cfg, "speculative_algorithm", None)).upper() != "DSPARK":
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES={victims} spills the graphed DSpark verify; "
            "pass --speculative-algorithm DSPARK or unset it"
        )
    if not stream.cpu_experts.enabled:
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES={victims} needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1"
        )
    lanes = stream.copy.graph_gather_miss_lanes
    if lanes:
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES gives every route a lane: unset "
            f"SGLANG_EXPERT_STREAM_GRAPH_GATHER_MISS_LANES (got {lanes})"
        )
    routes = _verify_routes(cfg)
    if routes is not None and victims >= routes:
        # GpuResidencyUpdater._init_insert_direct refuses 1 <= V < miss lanes after the 7-15 minute load.
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES={victims} must be below the {routes} routes of the "
            "DSpark verify (its tokens x the model's top-k)"
        )


def _check_copy_mode() -> None:
    """The native engine copies with the SM copy kernel or its own copy engine; ``dma`` belongs to the legacy
    streamer, which an EXL3 launch never runs."""
    if resolved_config().stream.copy.mode is CopyMode.DMA:
        raise ValueError(
            "SGLANG_EXPERT_STREAM_COPY_MODE=dma: dma is the legacy Qwen streamer's copy mode; the native engine "
            "takes sm_copy or copy_engine"
        )


def _check(cfg, budgets) -> None:
    """The eager checks, with decode allowed as a breakable CUDA graph at max batch size 1.

    Everything that still needs the host (the eager MoE fallback, the Engram
    file-table lookup) runs as an eager break; prefill stays eager. Decode ``full``
    cannot work: the Engram lookup reads its ids on the host. A breakable decode graph
    also turns DSV4's alt-stream overlap off (see the comment at the end).
    """
    stream = resolved_config().stream
    if stream.residency.async_promotions:
        # With the GPU residency update the in-graph updater owns the hot
        # slots: a boundary only ranks victims and the gather writes missed rows
        # straight into them, so there is no promotion copy to defer. Without it,
        # EXL3 rows come only from the row source and promote through the pinned
        # tier synchronously (ExpertHotCache._load_reserved_in_chunks). Either way
        # the flag would be ignored.
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_HOT_ASYNC_PROMOTIONS has no effect on EXL3 experts: the in-graph GPU "
            "residency updater writes missed rows into hot slots during the gather, or, without it, promotions run "
            "synchronously through the pinned host tier; unset it"
        )
    _check_copy_mode()
    graph = cfg.cuda_graph_config
    if not isinstance(graph, CudaGraphConfig):
        # run_resolution_pipeline's first offload pass runs before
        # parse_cuda_graph_config, while this is still the raw CLI value: the decode
        # backend is not known yet, and the pass after parsing runs every check below.
        return
    if stream.draft.cpu_experts:
        check_dspark_cpu_experts(cfg)
    _check_row_weighted_assignment()
    _check_ram_prefetch()
    _check_mirror_caps()
    cpu_experts = stream.cpu_experts.enabled
    speculative = getattr(cfg, "speculative_algorithm", None) is not None
    if cpu_experts and graph.decode.backend != Backend.BREAKABLE:
        # Checked before the speculative and backend rules: each would point a CPU-experts launch at the other's
        # decode backend.
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS computes experts inside the captured decode graph's copy wait; "
            "pass --cuda-graph-backend-decode breakable"
        )
    if speculative and graph.decode.backend != Backend.DISABLED:
        _check_graphed_verify(cfg, _CPU_EXPERTS_VERIFY_REMEDY if cpu_experts else _EAGER_VERIFY_REMEDY)
    _check_victim_lanes(cfg)
    if graph.decode.backend == Backend.DISABLED:
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
            "set SGLANG_EXPERT_STREAM_PINNED_HOST_MB"
        )
    if budgets.graph_gather and not budgets.hot_budget_mb:
        raise ValueError("EXL3 graph gathers need SGLANG_EXPERT_STREAM_HOT_GPU_MB")
    # Before the shared eager checks: their residency-update rule would tell a
    # CPU-experts launch without DIRECT to turn the update off, but CPU experts need
    # it on, at insert-on-miss stage 2.
    _check_slot_map()
    _check_cpu_experts(budgets)
    _check_direct_residency(budgets)
    # The shared eager check refuses graph gather; decode graphs may use it.
    _EAGER.check(_EagerGraphView(cfg), dataclasses.replace(budgets, graph_gather=False))
    # DSV4's alt-stream overlap still gives wrong output when captured in the
    # breakable decode graph, so this gate turns it off. One cause, the mHC stats side
    # stream (forked before the MoE break, launched on after it), is fixed by
    # _refork_stats_stream. A second defect remains in the first segment: the MQA
    # alt-stream prepare (MQALayer, capture-only) leaves layer 0's MoE input different
    # from eager's, cause unknown. The stats stream also runs in eager decode, so
    # eager decode of this launch loses that overlap too.
    overlap = envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP
    if overlap.get():
        if overlap.is_set():
            raise ValueError(
                "EXL3 expert caching's breakable decode graph gives wrong output with "
                "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1; unset it or set it to 0"
            )
        overlap.set(False)


def _check_slot_map() -> None:
    """The slot-map chain's switches.

    ``copy.ram_hit_copy`` types how the post copies RAM hits;
    ``cpu_experts.misses`` lets the CPU take NVMe misses;
    ``cpu_experts.split.miss_cut``/``miss_cut_max`` trim the CPU's split on nodes with forced misses;
    ``cpu_experts.split.model`` (and ``model_costs``) replaces the split and the cut with the cost model;
    ``cpu_experts.two_stage`` overlaps a CPU miss's gate/up GEMVs with its w2 read.
    """
    stream = resolved_config().stream
    cpu, split, two_stage = stream.cpu_experts, stream.cpu_experts.split, stream.cpu_experts.two_stage
    hit_copy = stream.copy.ram_hit_copy
    if hit_copy not in ("ce", "sm"):
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_RAM_HIT_COPY must be ce or sm, got {hit_copy!r}"
        )
    if cpu.misses and not cpu.enabled:
        # The device would type misses kMissCpu that no CPU thread computes.
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_MISSES needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1"
        )
    for key, value in (("miss_cut", split.miss_cut), ("miss_cut_max", split.miss_cut_max)):
        if value < 0:
            raise ValueError(f"SGLANG_EXPERT_STREAM_CPU_SPLIT: {key} must be >= 0, got {value}")
    if split.miss_cut and _group_sets("SGLANG_EXPERT_STREAM_CPU_SPLIT", "miss_cut") and not cpu.enabled:
        # Without CPU experts there is no split to cut: an explicitly set cut is a launch typo (the default is on).
        raise ValueError("SGLANG_EXPERT_STREAM_CPU_SPLIT: miss_cut needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1")
    if split.model:
        # The model splits RAM hits between the CPU and the copy engine; it does not price CPU-computed NVMe misses.
        if not cpu.enabled:
            raise ValueError("SGLANG_EXPERT_STREAM_CPU_SPLIT: model needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1")
        if hit_copy != "ce":
            raise ValueError("SGLANG_EXPERT_STREAM_CPU_SPLIT: model needs SGLANG_EXPERT_STREAM_RAM_HIT_COPY=ce")
        if cpu.misses:
            raise ValueError(
                "SGLANG_EXPERT_STREAM_CPU_SPLIT: model refuses SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_MISSES=1"
            )
        from sglang.srt.layers.moe.ram_slot_map import split_model_words

        split_model_words(2, split.model_costs)
    else:
        if split.model_costs.strip():
            raise ValueError("SGLANG_EXPERT_STREAM_CPU_SPLIT: model_costs needs model=true")
        if split.model_residency_tiebreak:
            raise ValueError("SGLANG_EXPERT_STREAM_CPU_SPLIT: model_residency_tiebreak needs model=true")
    if two_stage.enabled and not cpu.enabled:
        # Without CPU experts there is no CPU miss to split: a set flag is a launch typo.
        raise ValueError(
            "SGLANG_EXPERT_STREAM_CPU_TWO_STAGE: cpu_experts.two_stage needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1"
        )
    for key, value in (("w13_weights", two_stage.w13_weights), ("w2_weights", two_stage.w2_weights)):
        if value.strip() and not two_stage.enabled:
            raise ValueError(f"SGLANG_EXPERT_STREAM_CPU_TWO_STAGE: {key} needs two_stage.enabled")
    if two_stage.enabled:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import two_stage_span_weights

        two_stage_span_weights()


def _check_ram_prefetch() -> None:
    """``prefetch.ram``'s prerequisite and its options' bounds, which the host refuses at enable too.

    Runs on the post-parse pass only (the pass before ``parse_cuda_graph_config`` returns first), ahead of the backend
    rules: a disabled decode graph returns early, and the option must not pass silently.
    """
    from sglang.srt.layers.moe import ram_prefetch

    resolved = resolved_config()
    stream, prefetch = resolved.stream, resolved.stream.prefetch.ram
    floors_setting = resolved.dsv41.ram_prefetch.margin_floors_path
    scorer = prefetch.scorer
    if scorer not in ram_prefetch.SCORERS:
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER must be one of {ram_prefetch.SCORERS}, got {scorer!r}"
        )
    # The gpu default and its packaged floors apply to a prefetching launch only; a value set explicitly is checked
    # either way, so it cannot pass unused.
    explicit_scorer = envs.SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER.is_set()
    explicit_floors = floors_setting
    if explicit_floors and scorer != "gpu":
        # The floors filter the GPU scorer's candidate page; the CPU scorer would run without them, silently.
        raise ValueError("SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS needs SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu")
    deadline = prefetch.idle_deadline_us
    if not 1 <= deadline <= ram_prefetch.MAX_IDLE_DEADLINE_US:
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_DEADLINE_US must be in [1, {ram_prefetch.MAX_IDLE_DEADLINE_US}], "
            f"got {deadline}"
        )
    pieces = prefetch.idle_pieces
    if not 1 <= pieces <= ram_prefetch.MAX_IDLE_PIECES:
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_PIECES must be in [1, {ram_prefetch.MAX_IDLE_PIECES}], "
            f"got {pieces}"
        )
    horizon = prefetch.horizon
    if not 1 <= horizon <= ram_prefetch.MAX_HORIZON:
        raise ValueError(
            f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_HORIZON must be in [1, {ram_prefetch.MAX_HORIZON}], got {horizon}"
        )
    needs = "needs SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH=1"
    if not prefetch.enabled:
        if horizon != 1:
            raise ValueError(f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_HORIZON {needs}")
        if explicit_scorer and scorer == "gpu":
            raise ValueError(f"SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu {needs}")
        if explicit_floors:
            raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS {needs}")
        if prefetch.idle_drive:
            raise ValueError(f"SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE {needs}")
        if resolved.dsv41.ram_prefetch.learned_head_path:
            raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD {needs}")
        if prefetch.promote:
            raise ValueError(f"SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_PROMOTE {needs}")
        if prefetch.pool_serve:
            raise ValueError(f"SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_POOL_SERVE {needs}")
        return
    if not stream.cpu_experts.enabled:
        # Only a record with a CPU lane stages the input the scorer reads, and only a forced CPU miss uses the pool.
        raise ValueError("SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH needs SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1")
    if scorer == "gpu" and stream.copy.mode is not CopyMode.COPY_ENGINE:
        # Refused, not switched to the CPU scorer: the CPU experts the prefetch needs refuse this launch either way.
        raise ValueError(
            "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu runs after the copy-engine post captured in the decode "
            "graph; set SGLANG_EXPERT_STREAM_COPY_MODE=copy_engine"
            + (
                ""
                if explicit_scorer
                else " (gpu is the default scorer; SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS needs the copy engine too)"
            )
        )
    floors = ram_prefetch.margin_floors_path(scorer)
    if horizon > 1 and scorer == "gpu" and floors_setting is None:
        raise ValueError(
            "the packaged margin floors were fit for horizon 1; set SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS "
            "to '' or a horizon-matched table"
        )
    if floors:
        try:
            ram_prefetch.read_margin_floors(floors)
        except (OSError, ValueError) as error:
            raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS: {error}") from error
    for name, value, high in (
        ("SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_TOKEN", prefetch.per_token, ram_prefetch.MAX_PER_TOKEN),
        ("SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_LAYER", prefetch.per_layer, ram_prefetch.MAX_PER_LAYER),
        ("SGLANG_EXPERT_STREAM_RAM_PREFETCH_SPEC_SHARE", prefetch.spec_share, ram_prefetch.MAX_SPEC_SHARE),
    ):
        if not 1 <= value <= high:
            raise ValueError(f"{name} must be in [1, {high}], got {value}")
    _check_learned_head(scorer=scorer, horizon=horizon)


def _check_learned_head(*, scorer: str, horizon: int) -> None:
    """``SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD``'s prerequisites and its file (a prefetching launch only; the
    prefetch-off case is refused in ``_check_ram_prefetch``'s early return). Row coverage is checked at enable."""
    resolved = resolved_config()
    path = resolved.dsv41.ram_prefetch.learned_head_path
    if not path:
        return
    name = "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD"
    if scorer != "gpu":
        raise ValueError(f"{name} needs SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER=gpu")
    if horizon != 2:
        raise ValueError(f"{name} needs SGLANG_EXPERT_STREAM_RAM_PREFETCH_HORIZON=2, got {horizon}")
    if resolved.dsv41.ram_prefetch.margin_floors_path != "":
        raise ValueError(f"{name} needs SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS='' (no floors table)")
    from sglang.kernels.ops.moe.spec_head import HEAD_MAX_PICKS
    from sglang.srt.layers.moe import ram_prefetch_learned_head

    per_layer = resolved.stream.prefetch.ram.per_layer
    if not 1 <= per_layer <= HEAD_MAX_PICKS:
        raise ValueError(
            f"{name} needs SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_LAYER in 1..{HEAD_MAX_PICKS}, got {per_layer}"
        )

    try:
        ram_prefetch_learned_head.read_head_file(path)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error


def _check_mirror_caps() -> None:
    """``tiers.mirror.dynamic``'s caps, one positive in-flight cap per mirror root, refused here rather than when the
    RAM-miss service starts (exl3_read_split.mirror_caps, which the service calls too)."""
    from sglang.srt.layers.moe.exl3_read_split import mirror_caps

    mirror = resolved_config().stream.tiers.mirror
    mirror_caps(mirror.dynamic, mirror.caps, mirror.dirs)


def _check_direct_residency(budgets) -> None:
    """Require DIRECT residency for a graph gather.

    The graph gather runs the RAM-miss service, whose VRAM-hot set is the one the
    DIRECT updater writes into every record: no other residency mode feeds it.
    """
    if budgets.graph_gather and not _direct_residency(resolved_config().stream):
        raise ValueError(
            "EXL3 graph gather (the RAM-miss service) needs DIRECT residency "
            "(SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE=1, SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE=2)"
        )


def _check_cpu_experts(budgets) -> None:
    """Check the prerequisites of ``SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS``.

    The RAM-miss service's grant sends resident lanes to the CPU, the copy engine's
    copy wait completes them, and layer fusion's route tables and the DIRECT commit
    leave them out of the fused MoE and the residency. Each piece must be on, so this
    names every missing switch at once.
    """
    stream = resolved_config().stream
    if not stream.cpu_experts.enabled:
        return
    needs = [
        ("SGLANG_EXPERT_STREAM_ENABLE_GRAPH_GATHER=1", budgets.graph_gather),
        ("SGLANG_EXPERT_STREAM_COPY_MODE=copy_engine", stream.copy.mode is CopyMode.COPY_ENGINE),
        (
            "SGLANG_DSV41_ENABLE_LAYER_FUSION=1",
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.get(),
        ),
        # DIRECT residency ranks the keys the fused plan sorts the miss lanes by,
        # so the CPU takes the coldest ones.
        ("SGLANG_EXPERT_STREAM_ENABLE_GPU_RESIDENCY_UPDATE=1", stream.residency.gpu_update),
        (
            "SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_STAGE=2",
            stream.residency.insert_on_miss_stage == _DIRECT,
        ),
        ("SGLANG_EXPERT_STREAM_ENABLE_FUSED_PLAN=1", stream.copy.fused_plan),
    ]
    missing = [name for name, ok in needs if not ok]
    if missing:
        raise ValueError("SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS needs " + ", ".join(missing))
    if stream.prefetch.shadow.pull_mode is not PullMode.OFF:
        # The pull join rewrites a route's slot after planning; a CPU lane's route
        # would then match no plan slot, and the fused MoE would compute it on top of
        # the CPU's partial.
        raise ValueError(
            "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS cannot run with a shadow prefetch pull; set "
            "SGLANG_EXPERT_STREAM_SHADOW_PREFETCH's pull_mode to off"
        )


exl3_expert_stream_requirements = ExpertStreamRequirements(
    "EXL3", _check, graph_gather_host_source="pinned_tier"
)
register_expert_stream_requirements(("exl3",), exl3_expert_stream_requirements)
