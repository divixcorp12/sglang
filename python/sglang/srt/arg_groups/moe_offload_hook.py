# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for ``--moe-offload-preset``."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.environ import envs
from sglang.srt.layers.moe import offload_presets
from sglang.srt.model_executor.cuda_graph_config import Backend

logger = logging.getLogger(__name__)


def handle_moe_offload_preset(server_args: Any) -> None:
    """Fill unset offload variables from the preset, derive forced ones, refuse invalid setups.

    Runs in the launcher before the scheduler and workers are spawned, so they
    inherit every variable set here.
    """
    cfg = resolving_view(server_args)
    name = cfg.moe_offload_preset
    resolved = offload_presets.resolve_offload_env(
        offload_presets.PRESETS[name], offload_presets.explicit_offload_env(os.environ)
    )
    for env_name, value in resolved.filled.items():
        getattr(envs, env_name).set(value)
        logger.info("MoE offload preset %s sets %s=%s", name, env_name, value)
    for env_name, value in resolved.overridden.items():
        logger.info(
            "MoE offload preset %s: %s=%s set explicitly, preset value %s",
            name,
            env_name,
            os.environ[env_name],
            value,
        )
    if offload_presets.needs_overlap_off(resolved.effective) and not cfg.disable_overlap_schedule:
        declare_resolution(server_args, "handle_moe_offload_preset", disable_overlap_schedule=True)
        logger.info("MoE offload preset %s turns overlap scheduling off", name)
    graph = cfg.cuda_graph_config
    try:
        offload_presets.check_offload_config(
            resolved.effective,
            speculative=cfg.speculative_algorithm is not None,
            decode_graphs_disabled=graph is not None and graph.decode.backend == Backend.DISABLED,
            decode_max_bs=None if graph is None else graph.decode.max_bs,
            tp_size=cfg.tp_size,
            pp_size=cfg.pp_size,
            dp_size=cfg.dp_size,
            dp_attention=cfg.enable_dp_attention,
            allowed_cpus=os.sched_getaffinity(0),
        )
    except ValueError as error:
        raise ValueError(f"--moe-offload-preset {name}: {error}") from error
