"""Server-argument requirements of EXL3 expert streaming (DeepSeek V4.1).

The MoE expert-caching gate imports this module the first time a launch's
quantization method resolves to ``exl3`` (the checkpoint's ``config.json``
names it). EXL3 experts stream with prefill eager and decode eager or a
breakable CUDA graph at max batch size 1 (a ``full`` decode graph is refused);
no graph gather, no host arena, and a ``stat`` or ``per_pass`` recorder under
dynamic residency. Nothing else is required; ``--max-running-requests`` and the
overlap schedule stay free. This module runs during server-args processing,
so it imports only the gate module, ``sglang.srt.environ`` and the graph-config enum.
"""

from sglang.srt.arg_groups.expert_stream_requirements import (
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    register_expert_stream_requirements,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend

_EAGER = eager_expert_stream_requirements(
    "EXL3",
    enabled=lambda: envs.SGLANG_DSV41_EXPERT_STREAM.get(),
    enable_hint="SGLANG_DSV41_EXPERT_STREAM=1",
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
    ``full`` cannot work: the Engram lookup reads its ids on the host.
    """
    graph = cfg.cuda_graph_config
    if graph is None or graph.decode.backend == Backend.DISABLED:
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
    _EAGER.check(_EagerGraphView(cfg), budgets)


exl3_expert_stream_requirements = ExpertStreamRequirements("EXL3", _check)
register_expert_stream_requirements(("exl3",), exl3_expert_stream_requirements)
