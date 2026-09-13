"""Qwen4-Exp may run with --language-model-only (no vision tower)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups import model_hook
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(architecture):
    args = SimpleNamespace(
        language_model_only=True,
        encoder_only=False,
        language_only=False,
        enable_prefix_mm_cache=False,
        enable_broadcast_mm_inputs_process=False,
        mm_enable_dp_encoder=False,
        disaggregation_mode="null",
        LANGUAGE_MODEL_ONLY_ARCHITECTURES=ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture])
    )
    return args, model_config


def _handle(architecture):
    args, model_config = _args(architecture)
    with patch.object(model_hook, "resolving_view", lambda a: a), patch.object(
        model_hook, "model_config_of", lambda a: model_config
    ):
        model_hook.handle_language_model_only(args)


def test_qwen4_exp_is_accepted():
    _handle("Qwen4ExpForConditionalGeneration")


def test_unlisted_architecture_is_still_rejected():
    with pytest.raises(ValueError, match="does not support"):
        _handle("Qwen3VLForConditionalGeneration")
