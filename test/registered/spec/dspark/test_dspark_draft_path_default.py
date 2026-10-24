import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
    handle_speculative_decoding,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args._model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            _BUNDLED_MODEL_PATH,
        )
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), 6
        )

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            "deepseek-ai/some-other-dspark-draft",
        )


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(self, *, moe_a2a_backend: str) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.enable_dp_lm_head = True
        server_args.attn_dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        return server_args

    def test_only_megamoe_is_admitted(self):
        """Both sides of the allowlist: megamoe passes, others raise by name."""
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(self._dp_server_args(moe_a2a_backend="megamoe"))
            for backend in ("deepep", "pplx"):
                with self.assertRaisesRegex(ValueError, backend):
                    _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        server_args = self._dp_server_args(moe_a2a_backend="megamoe")
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "static"):
                _handle_dspark(server_args)


class TestDsparkConfidenceCutRefusals(CustomTestCase):
    """SGLANG_DSPARK_CONFIDENCE_CUT's launch contract: one cut source, static, DSpark only."""

    def _cut_server_args(self) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_dspark_confidence_sts_path = "/fitted/sts.json"
        return server_args

    def test_flag_off_checks_nothing(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        server_args.speculative_algorithm = None
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(False):
            handle_speculative_decoding(server_args)

    def test_a_served_launch_is_admitted(self):
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(
            True
        ), envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            handle_speculative_decoding(self._cut_server_args())

    def test_a_cut_with_no_sts_file_is_admitted(self):
        """Production cuts at T=1 with no STS file; a gate demanding one refused its launch."""
        server_args = self._cut_server_args()
        server_args.speculative_dspark_confidence_sts_path = None
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(
            True
        ), envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            handle_speculative_decoding(server_args)

    def test_each_refusal_names_its_fix(self):
        refusals = {
            "speculative-algorithm DSPARK": dict(speculative_algorithm=None),
            "SGLANG_DSPARK_TOP_PROB_GATE=0": dict(top_prob_gate=0.4),
            "SGLANG_RAGGED_VERIFY_MODE=static": dict(mode="compact"),
        }
        for fix, change in refusals.items():
            server_args = self._cut_server_args()
            if "speculative_algorithm" in change:
                server_args.speculative_algorithm = change["speculative_algorithm"]
            with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(
                True
            ), envs.SGLANG_DSPARK_TOP_PROB_GATE.override(
                change.get("top_prob_gate", 0.0)
            ), envs.SGLANG_RAGGED_VERIFY_MODE.override(
                change.get("mode", "static")
            ):
                with self.subTest(fix=fix), self.assertRaisesRegex(ValueError, fix):
                    handle_speculative_decoding(server_args)


if __name__ == "__main__":
    unittest.main()
