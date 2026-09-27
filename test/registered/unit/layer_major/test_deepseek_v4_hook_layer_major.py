"""CPU unit tests for the DSV4.1 layer-major launch refusals in deepseek_v4_hook.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig, PhaseConfig


class TestDsv41LayerMajorRefusal(unittest.TestCase):
    """The pure helper the reviewer asked for: isolates the two DSV4.1-only conditions."""

    def test_refuses_without_decoder_replay(self):
        reason = deepseek_v4_hook.dsv41_layer_major_refusal(decoder_replay=False, encoder_replay=False)
        self.assertIsNotNone(reason)
        self.assertIn("--enable-decoder-swa-bounded-replay", reason)

    def test_refuses_with_encoder_replay(self):
        reason = deepseek_v4_hook.dsv41_layer_major_refusal(decoder_replay=True, encoder_replay=True)
        self.assertIsNotNone(reason)
        self.assertIn("--enable-encoder-swa-bounded-replay", reason)

    def test_accepts_decoder_replay_without_encoder_replay(self):
        self.assertIsNone(
            deepseek_v4_hook.dsv41_layer_major_refusal(decoder_replay=True, encoder_replay=False)
        )


def _cfg(**overrides):
    """A minimal DSV4.1 config namespace: every field validate_deepseek_v41_features reads
    on a non-encoder-replay, non-CP, non-PD path, defaulted to an accepting launch."""
    values = dict(
        enable_encoder_swa_bounded_replay=False,
        speculative_algorithm=None,
        enable_hisparse=False,
        dsv4_attn_backend="fa3",
        enable_two_batch_overlap=False,
        pp_size=1,
        disaggregation_mode="null",
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled"),
            prefill=PhaseConfig(backend="disabled"),
        ),
        enable_decoder_swa_bounded_replay=True,
        enable_dp_attention=False,
        attn_cp_size=1,
        max_running_requests=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run(cfg):
    """Calls validate_deepseek_v41_features(cfg) as if resolving_view/model_config_of had
    already resolved a DeepSeek-V4.1 launch; server_args itself is unused past those two calls."""
    fake_model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
    with mock.patch.object(deepseek_v4_hook, "resolving_view", lambda server_args: server_args):
        with mock.patch.object(deepseek_v4_hook, "model_config_of", lambda server_args: fake_model_config):
            deepseek_v4_hook.validate_deepseek_v41_features(cfg)


class TestValidateDeepseekV41FeaturesLayerMajorGate(unittest.TestCase):
    """Exercises the hook block itself: it must only fire when the env threshold is set,
    and it must raise using the two DSV4.1-only refusals."""

    def test_block_does_not_run_when_env_is_off(self):
        # decoder_replay is False here, which would refuse if the block ran; with the env
        # threshold off, the block must not run at all and the launch must be accepted.
        with envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.override(0):
            _run(_cfg(enable_decoder_swa_bounded_replay=False))

    def test_accepts_a_valid_layer_major_launch(self):
        with envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.override(64):
            _run(_cfg())

    def test_refuses_without_decoder_replay_through_the_hook(self):
        with envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.override(64):
            with self.assertRaisesRegex(ValueError, "--enable-decoder-swa-bounded-replay"):
                _run(_cfg(enable_decoder_swa_bounded_replay=False))

    # The encoder-replay refusal (dsv41_layer_major_refusal's other branch) is not exercised
    # through the full hook here: setting cfg.enable_encoder_swa_bounded_replay=True also
    # enters validate_deepseek_v41_features's own, unrelated encoder-replay incompatibility
    # block (chunked_prefill_size, LoRA, radix sessions, external cache linker, unified
    # memory, ...), which would need a much larger fake cfg to satisfy just to fall through
    # to our block. It is covered directly above, on the pure helper
    # (test_refuses_with_encoder_replay).


if __name__ == "__main__":
    unittest.main()
