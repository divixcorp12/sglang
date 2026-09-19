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
    """Fill unset offload variables from the preset, derive forced ones.

    Runs in the launcher before the scheduler and workers are spawned, so they
    inherit every variable set here. Validation is separate (see
    ``check_moe_offload_config``): cuda_graph_config, speculative_algorithm,
    and the parallelism fields the checks need are not final yet at this
    point in the pipeline.
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


def check_moe_offload_config(server_args: Any) -> None:
    """Refuse invalid offload combinations once the fields it reads are final.

    Runs last in the resolution pipeline. cuda_graph_config keeps changing
    through handle_speculative_decoding, handle_data_parallelism,
    handle_dllm_inference and handle_other_validations; speculative_algorithm
    only settles at handle_speculative_decoding; tp/pp/dp_size and
    enable_dp_attention can still move through handle_data_parallelism and
    the DeepSeek-family model overrides. By this point every offload
    variable the preset fills is already in os.environ.
    """
    cfg = resolving_view(server_args)
    name = cfg.moe_offload_preset
    graph = cfg.cuda_graph_config
    try:
        offload_presets.check_offload_config(
            offload_presets.explicit_offload_env(os.environ),
            speculative=cfg.speculative_algorithm is not None,
            decode_graphs_disabled=graph.decode.backend == Backend.DISABLED,
            decode_max_bs=graph.decode.max_bs,
            tp_size=cfg.tp_size,
            pp_size=cfg.pp_size,
            dp_size=cfg.dp_size,
            dp_attention=cfg.enable_dp_attention,
            allowed_cpus=os.sched_getaffinity(0),
        )
    except ValueError as error:
        raise ValueError(f"--moe-offload-preset {name}: {error}") from error
