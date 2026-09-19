"""Server-argument requirements of EXL3 expert streaming (DeepSeek V4.1).

The MoE expert-caching gate imports this module the first time a launch's
quantization method resolves to ``exl3`` (the checkpoint's ``config.json``
names it). EXL3 experts stream eagerly only: no CUDA graphs, no graph gather,
no host arena, and a ``stat`` or ``per_pass`` recorder under dynamic
residency. Nothing else is required; ``--max-running-requests`` and the
overlap schedule stay free. This module runs during server-args processing,
so it imports only the gate module and ``sglang.srt.environ``.
"""

from sglang.srt.arg_groups.expert_stream_requirements import (
    eager_expert_stream_requirements,
    register_expert_stream_requirements,
)
from sglang.srt.environ import envs

register_expert_stream_requirements(
    ("exl3",),
    eager_expert_stream_requirements(
        "EXL3",
        enabled=lambda: envs.SGLANG_DSV41_EXPERT_STREAM.get(),
        enable_hint="SGLANG_DSV41_EXPERT_STREAM=1",
    ),
)
