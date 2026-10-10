"""Server-argument requirements of EXL3 expert streaming (DeepSeek V4.1).

The MoE expert-caching gate imports this module the first time a launch's
quantization method resolves to ``exl3`` (the checkpoint's ``config.json`` names it).
It registers :data:`exl3_expert_stream_requirements`, which requires:

* Prefill eager; decode eager or a breakable CUDA graph at max batch size 1. A
  ``full`` decode graph is refused, because the Engram file-table lookup reads its
  ids on the host.
* Graph gather only with that breakable decode graph, reading missed rows from the
  pinned host tier (never the host arena). It needs DIRECT residency: the GPU
  residency update with ``SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2``.
* A ``stat`` or ``per_pass`` recorder under dynamic residency.
* Speculative decoding as DSpark: its verify eager (decode graphs disabled), or in the breakable decode graph on DIRECT
  residency at 1-64 miss lanes with a static verify (§33.8). The target's CPU experts serve that graphed verify; with
  SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (and a lane per route: MISS_LANES unset) the lanes past the victims are
  theirs (spill).

Nothing else is required; ``--max-running-requests`` and the overlap schedule stay
free. The shared eager checks come from ``eager_expert_stream_requirements``. This
module runs during server-args processing, so it imports only the gate module,
``sglang.srt.environ``, the graph-config enum and the standard library.
"""

import dataclasses
import json
import os

from sglang.srt.arg_groups.expert_stream_requirements import (
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    register_expert_stream_requirements,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.assignment import row_weighted_sources
from sglang.srt.layers.moe.cpu_experts.draft_resident import load_resident_set
from sglang.srt.layers.moe.cpu_experts.threading_config import (
    check_not_reserved,
    parse_cpu_list,
)
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


def _check_dspark_cpu_experts(cfg) -> None:
    """The draft CPU experts' launch rules, so a bad core list or resident file fails here, not at the first draft
    call."""
    if getattr(cfg, "speculative_algorithm", None) != "DSPARK":
        raise ValueError(
            "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS computes the DSpark draft's routed experts on the CPU; "
            "pass --speculative-algorithm DSPARK or unset it"
        )
    if envs.SGLANG_DSV41_CPU_EXPERTS.get() and (
        envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get() or envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.get()
    ):
        raise ValueError(
            "the DSpark draft shares the GPU node's CPU expert team under SGLANG_DSV41_CPU_EXPERTS; unset "
            "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES and SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS"
        )
    # Unset cores are derived by ThreadingConfig when the draft starts; named ones are checked here.
    cores = parse_cpu_list(envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get())
    for core in cores:
        check_not_reserved(core)
    if len(cores) == 1:
        raise ValueError(
            "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS needs SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES with at "
            "least two cores (one spinning worker per core), or unset to derive them"
        )
    threads = envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.get()
    if cores and not 0 <= threads <= len(cores):
        raise ValueError(
            f"SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS={threads} on {len(cores)} cores: use 0 (one per core) "
            f"up to {len(cores)}"
        )
    if os.environ.get("EXL3_MOE_CPU_PIN") != "0":
        raise ValueError(
            "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS needs EXL3_MOE_CPU_PIN=0: the kernel would otherwise pin "
            "its workers to the first cores"
        )
    # Only the optimized build exports a CpuExpertKernel. SGLANG_DSV41_CPU_EXPERTS also selects it, but a draft-only
    # launch has it off, so these two defines are the way in. The recipe sets them in either case (exl3/ext.py,
    # cpu_act_defines and optimized_cpu).
    if not (envs.SGLANG_EXL3_CPU_ACT_RESIDUAL.get() and envs.SGLANG_EXL3_CPU_ACT_BLOCK.get() == 128):
        raise ValueError(
            "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS runs the optimized EXL3 CPU kernel: set "
            "SGLANG_EXL3_CPU_ACT_RESIDUAL=1 and SGLANG_EXL3_CPU_ACT_BLOCK=128"
        )
    resident = envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get()
    if resident:
        load_resident_set(resident)


_EAGER_VERIFY_REMEDY = "or pass --cuda-graph-backend-decode disabled to run the DSpark verify eagerly"
# SGLANG_DSV41_CPU_EXPERTS computes in the captured graph's copy wait, so its verify cannot go eager.
_CPU_EXPERTS_VERIFY_REMEDY = "SGLANG_DSV41_CPU_EXPERTS serves the DSpark verify only in the decode graph"


def _check_row_weighted_assignment() -> None:
    """A named source's CPU experts must be on: the option would otherwise log a change no kernel runs."""
    needs = {
        "draft": ("SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS", envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS),
        "target": ("SGLANG_DSV41_CPU_EXPERTS", envs.SGLANG_DSV41_CPU_EXPERTS),
    }
    for source in sorted(row_weighted_sources()):
        name, flag = needs[source]
        if not flag.get():
            raise ValueError(
                f"SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT={envs.SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT.get()} needs "
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
    lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.get()
    if not (
        envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.get()
        and envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get()
        and envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get() == 2
        and (1 <= lanes <= 64 or (lanes == 0 and envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get() > 0))
    ):
        raise ValueError(
            "EXL3 expert caching runs a DSpark verify in the decode graph only with SGLANG_MOE_EXPERT_GRAPH_GATHER=1, "
            "SGLANG_MOE_GPU_RESIDENCY_UPDATE=1, SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 and "
            "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES=1-64, or unset with "
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (got {lanes}): a lane per route needs spill's CPU lanes; "
            f"{remedy}"
        )
    if envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
        raise ValueError(
            "EXL3 expert caching runs a DSpark verify in the decode graph with SGLANG_RAGGED_VERIFY_MODE=static only "
            f"(compact mode reads the host); {remedy}"
        )
    if envs.SGLANG_DSV41_CPU_EXPERTS.get() and not envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get():
        # The tested shape is a lane per route plus VICTIM_LANES; a narrowed MISS_LANES verify with CPU experts and no
        # spill has never run on a GPU.
        raise ValueError(
            "a graphed DSpark verify with SGLANG_DSV41_CPU_EXPERTS needs SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES "
            "(a lane per route: unset SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES)"
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
    """SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (spill): the CPU takes the lanes past the victims, and every route has
    a lane, so a verify's distinct misses never outnumber its lanes."""
    victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
    if not victims:
        return
    if str(getattr(cfg, "speculative_algorithm", None)).upper() != "DSPARK":
        raise ValueError(
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} spills the graphed DSpark verify; "
            "pass --speculative-algorithm DSPARK or unset it"
        )
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        raise ValueError(f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} needs SGLANG_DSV41_CPU_EXPERTS=1")
    lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.get()
    if lanes:
        raise ValueError(
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES gives every route a lane: unset "
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES (got {lanes})"
        )
    routes = _verify_routes(cfg)
    if routes is not None and victims >= routes:
        # GpuResidencyUpdater._init_insert_direct refuses 1 <= V < miss lanes after the 7-15 minute load.
        raise ValueError(
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} must be below the {routes} routes of the DSpark "
            "verify (its tokens x the model's top-k)"
        )


def _check(cfg, budgets) -> None:
    """The eager checks, with decode allowed as a breakable CUDA graph at max batch size 1.

    Everything that still needs the host (the eager MoE fallback, the Engram
    file-table lookup) runs as an eager break; prefill stays eager. Decode ``full``
    cannot work: the Engram lookup reads its ids on the host. A breakable decode graph
    also turns DSV4's alt-stream overlap off (see the comment at the end).
    """
    if envs.SGLANG_MOE_HOT_ASYNC_PROMOTIONS.get():
        # With SGLANG_MOE_GPU_RESIDENCY_UPDATE the in-graph updater owns the hot
        # slots: a boundary only ranks victims and the gather writes missed rows
        # straight into them, so there is no promotion copy to defer. Without it,
        # EXL3 rows come only from the row source and promote through the pinned
        # tier synchronously (ExpertHotCache._load_reserved_in_chunks). Either way
        # the flag would be ignored.
        raise ValueError(
            "SGLANG_MOE_HOT_ASYNC_PROMOTIONS has no effect on EXL3 experts: the in-graph GPU residency "
            "updater writes missed rows into hot slots during the gather, or, without it, promotions run "
            "synchronously through the pinned host tier; unset it"
        )
    graph = cfg.cuda_graph_config
    if not isinstance(graph, CudaGraphConfig):
        # run_resolution_pipeline's first offload pass runs before
        # parse_cuda_graph_config, while this is still the raw CLI value: the decode
        # backend is not known yet, and the pass after parsing runs every check below.
        return
    if envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS.get():
        _check_dspark_cpu_experts(cfg)
    _check_row_weighted_assignment()
    _check_ram_prefetch()
    _check_mirror_caps()
    cpu_experts = envs.SGLANG_DSV41_CPU_EXPERTS.get()
    speculative = getattr(cfg, "speculative_algorithm", None) is not None
    if cpu_experts and graph.decode.backend != Backend.BREAKABLE:
        # Checked before the speculative and backend rules: each would point a CPU-experts launch at the other's
        # decode backend.
        raise ValueError(
            "SGLANG_DSV41_CPU_EXPERTS computes experts inside the captured decode graph's copy wait; "
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
            "set SGLANG_MOE_PINNED_HOST_MB"
        )
    if budgets.graph_gather and not budgets.hot_budget_mb:
        raise ValueError("EXL3 graph gathers need SGLANG_MOE_HOT_GPU_MB")
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

    ``SGLANG_DSV41_RAM_HIT_COPY`` types how the post copies RAM hits;
    ``SGLANG_DSV41_CPU_EXPERTS_MISSES`` lets the CPU take NVMe misses;
    ``SGLANG_DSV41_CPU_SPLIT_MISS_CUT``/``_MAX`` trim the CPU's split on nodes with forced misses.
    """
    hit_copy = envs.SGLANG_DSV41_RAM_HIT_COPY.get()
    if hit_copy not in ("ce", "sm"):
        raise ValueError(
            f"SGLANG_DSV41_RAM_HIT_COPY must be ce or sm, got {hit_copy!r}"
        )
    if (
        envs.SGLANG_DSV41_CPU_EXPERTS_MISSES.get()
        and not envs.SGLANG_DSV41_CPU_EXPERTS.get()
    ):
        # The device would type misses kMissCpu that no CPU thread computes.
        raise ValueError(
            "SGLANG_DSV41_CPU_EXPERTS_MISSES needs SGLANG_DSV41_CPU_EXPERTS=1"
        )
    for name in ("SGLANG_DSV41_CPU_SPLIT_MISS_CUT", "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX"):
        value = getattr(envs, name).get()
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if envs.SGLANG_DSV41_CPU_SPLIT_MISS_CUT.get() and not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        # Without CPU experts there is no split to cut: a set cut is a launch typo.
        raise ValueError("SGLANG_DSV41_CPU_SPLIT_MISS_CUT needs SGLANG_DSV41_CPU_EXPERTS=1")


def _check_ram_prefetch() -> None:
    """``SGLANG_DSV41_RAM_PREFETCH``'s prerequisite and its options' bounds, which the host refuses at enable too.

    Runs on the post-parse pass only (the pass before ``parse_cuda_graph_config`` returns first), ahead of the backend
    rules: a disabled decode graph returns early, and the option must not pass silently.
    """
    from sglang.srt.layers.moe import ram_prefetch

    scorer = envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get()
    if scorer not in ram_prefetch.SCORERS:
        raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_SCORER must be one of {ram_prefetch.SCORERS}, got {scorer!r}")
    floors = envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.get()
    if floors:
        if scorer != "gpu":
            # The floors filter the GPU scorer's candidate page; the CPU scorer would run without them, silently.
            raise ValueError("SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS needs SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu")
        try:
            ram_prefetch.read_margin_floors(floors)
        except (OSError, ValueError) as error:
            raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS: {error}") from error
    deadline = envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US.get()
    if not 1 <= deadline <= ram_prefetch.MAX_IDLE_DEADLINE_US:
        raise ValueError(
            f"SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US must be in [1, {ram_prefetch.MAX_IDLE_DEADLINE_US}], "
            f"got {deadline}"
        )
    if not envs.SGLANG_DSV41_RAM_PREFETCH.get():
        if scorer == "gpu":
            raise ValueError("SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu needs SGLANG_DSV41_RAM_PREFETCH=1")
        if envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE.get():
            raise ValueError("SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE needs SGLANG_DSV41_RAM_PREFETCH=1")
        return
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        # Only a record with a CPU lane stages the input the scorer reads, and only a forced CPU miss uses the pool.
        raise ValueError("SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS=1")
    if scorer == "gpu" and not envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.get():
        raise ValueError(
            "SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu runs after the copy-engine post captured in the decode graph; "
            "set SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1"
        )
    for name, high in (
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", ram_prefetch.MAX_PER_TOKEN),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", ram_prefetch.MAX_PER_LAYER),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", ram_prefetch.MAX_SPEC_SHARE),
    ):
        value = getattr(envs, name).get()
        if not 1 <= value <= high:
            raise ValueError(f"{name} must be in [1, {high}], got {value}")


def _check_mirror_caps() -> None:
    """``SGLANG_MOE_EXPERT_MIRROR_DYNAMIC``'s caps, one positive in-flight cap per mirror root, refused here rather
    than when the RAM-miss service starts (exl3_read_split.mirror_caps, which the service calls too)."""
    from sglang.srt.layers.moe.exl3_read_split import mirror_caps

    mirror_caps(
        envs.SGLANG_MOE_EXPERT_MIRROR_DYNAMIC.get(),
        envs.SGLANG_MOE_EXPERT_MIRROR_CAPS.get(),
        envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.get(),
    )


def _check_direct_residency(budgets) -> None:
    """Require DIRECT residency for a graph gather.

    The graph gather runs the RAM-miss service, whose VRAM-hot set is the one the
    DIRECT updater writes into every record: no other residency mode feeds it.
    """
    if budgets.graph_gather and not (
        envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get()
        and envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get() == 2
    ):
        raise ValueError(
            "EXL3 graph gather (the RAM-miss service) needs DIRECT residency "
            "(SGLANG_MOE_GPU_RESIDENCY_UPDATE=1, SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2)"
        )


def _check_cpu_experts(budgets) -> None:
    """Check the prerequisites of ``SGLANG_DSV41_CPU_EXPERTS``.

    The RAM-miss service's grant sends resident lanes to the CPU, the copy engine's
    copy wait completes them, and layer fusion's route tables and the DIRECT commit
    leave them out of the fused MoE and the residency. Each piece must be on, so this
    names every missing switch at once.
    """
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        return
    needs = [
        ("SGLANG_MOE_EXPERT_GRAPH_GATHER=1", budgets.graph_gather),
        (
            "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1",
            envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.get(),
        ),
        (
            "SGLANG_DSV41_ENABLE_LAYER_FUSION=1",
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.get(),
        ),
        # DIRECT residency ranks the keys the fused plan sorts the miss lanes by,
        # so the CPU takes the coldest ones.
        (
            "SGLANG_MOE_GPU_RESIDENCY_UPDATE=1",
            envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get(),
        ),
        (
            "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2",
            envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get() == 2,
        ),
        ("SGLANG_MOE_EXPERT_FUSED_PLAN=1", envs.SGLANG_MOE_EXPERT_FUSED_PLAN.get()),
    ]
    missing = [name for name, ok in needs if not ok]
    if missing:
        raise ValueError("SGLANG_DSV41_CPU_EXPERTS needs " + ", ".join(missing))
    if envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.get() != "off":
        # The pull join rewrites a route's slot after planning; a CPU lane's route
        # would then match no plan slot, and the fused MoE would compute it on top of
        # the CPU's partial.
        raise ValueError(
            "SGLANG_DSV41_CPU_EXPERTS cannot run with SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE; set it to off"
        )


exl3_expert_stream_requirements = ExpertStreamRequirements(
    "EXL3", _check, graph_gather_host_source="pinned_tier"
)
register_expert_stream_requirements(("exl3",), exl3_expert_stream_requirements)
