"""A request for prompt-token logprobs is refused at admission on paths whose forward cannot produce them.

Found by the DSV4.1 copy-engine soak (docs/superpowers/plans/2026-09-25-dsv41-copy-engine-soak.md): one
``/generate`` request with ``logprob_start_len=0`` under ``--enable-decoder-swa-bounded-replay`` raised in
``DeepseekV4Model._check_late_layer_tail_readers`` during the prefill, which stopped the scheduler and so the server.
"""

import unittest

from sglang.srt.managers.scheduler import prompt_logprob_refusal
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def refusal(*, mlx=False, replay=False, return_logprob=True, start=0, n=10):
    return prompt_logprob_refusal(
        mlx_sampling=mlx,
        decoder_swa_bounded_replay=replay,
        return_logprob=return_logprob,
        logprob_start_len=start,
        input_len=n,
    )


class TestPromptLogprobRefusal(CustomTestCase):
    def test_bounded_replay_refuses_prompt_logprobs(self):
        for start in (0, 5, 9):
            msg = refusal(replay=True, start=start)
            self.assertIsNotNone(msg)
            self.assertIn("--enable-decoder-swa-bounded-replay", msg)

    def test_bounded_replay_admits_output_logprobs(self):
        # The scheduler resolves an unset start to the prompt length: output logprobs only.
        self.assertIsNone(refusal(replay=True, start=10))
        self.assertIsNone(refusal(replay=True, start=-1))
        self.assertIsNone(refusal(replay=True, return_logprob=False, start=0))

    def test_mlx_refusal_unchanged(self):
        self.assertIn("MLX", refusal(mlx=True, start=0))
        self.assertIsNone(refusal(mlx=True, start=10))

    def test_default_path_admits_prompt_logprobs(self):
        self.assertIsNone(refusal(start=0))


if __name__ == "__main__":
    unittest.main()
