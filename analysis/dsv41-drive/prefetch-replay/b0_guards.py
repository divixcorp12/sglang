"""Pure refusals of the B0 capture driver: each raises RuntimeError naming what it refused.

``refuse_launch`` runs them all before the driver takes any lock or starts a server.
"""

import posixpath
import types

CORPUS_ROOT = "/mnt/nvme1"
ARM = "dspark-both-misscut"
PREFETCH_ENV = "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH"
# Substrings of `ps -eo cmd` lines that mean an A/B or a server is using the drives (R6).
BUSY_MARKERS = ("launch_server", "both_cpu_ab", "run_arm.sh")


def refuse_output_path(*, out_dir: str) -> None:
    """Refuse an output directory outside /mnt/nvme1 (R3, D7); the root volume and nvme0/2/4 fail here."""
    path = posixpath.normpath(out_dir)
    if not posixpath.isabs(path) or not (
        path == CORPUS_ROOT or path.startswith(CORPUS_ROOT + "/")
    ):
        raise RuntimeError(
            f"refused output path {out_dir!r}: captures go under {CORPUS_ROOT} only"
        )


def refuse_prefetch(*, env: dict) -> None:
    """Refuse a recipe whose SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH is not '0' (D6)."""
    value = env.get(PREFETCH_ENV)
    if value != "0":
        raise RuntimeError(
            f"refused {PREFETCH_ENV}={value!r}: the capture runs with RAM prefetch off ('0')"
        )


def refuse_arm(*, arm: str) -> None:
    """Refuse any arm but dspark-both-misscut."""
    if arm != ARM:
        raise RuntimeError(f"refused arm {arm!r}: the capture recipe is {ARM!r}")


def refuse_running_ab(*, ps_lines: list[str]) -> None:
    """Refuse while a launch_server, both_cpu_ab or run_arm.sh process is running (R6)."""
    busy = [
        line.strip()
        for line in ps_lines
        if "grep" not in line and any(marker in line for marker in BUSY_MARKERS)
    ]
    if busy:
        raise RuntimeError(
            f"refused: {len(busy)} serving or A/B process(es) running, first: {busy[0]!r}"
        )


def refuse_non_exl3(*, model_path: str) -> None:
    """Refuse a model whose resolved expert-stream requirements are not EXL3, asserted as check_gate does.

    A non-directory model_path silently resolves to NVFP4, so the label is checked, not just the call.
    """
    from sglang.srt.arg_groups.expert_stream_requirements import (
        expert_stream_requirements_for,
    )

    cfg = types.SimpleNamespace(quantization=None, model_path=model_path)
    label = expert_stream_requirements_for(cfg, cfg).label
    if label != "EXL3":
        raise RuntimeError(
            f"refused model {model_path!r}: the gate resolved {label!r} requirements, not EXL3"
        )


def refuse_launch(
    *, out_dir: str, env: dict, arm: str, model_path: str, ps_lines: list[str]
) -> None:
    """Run every refusal; raises RuntimeError naming the first one that fires."""
    refuse_output_path(out_dir=out_dir)
    refuse_prefetch(env=env)
    refuse_arm(arm=arm)
    refuse_running_ab(ps_lines=ps_lines)
    refuse_non_exl3(model_path=model_path)
