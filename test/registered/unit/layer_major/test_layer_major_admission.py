"""CPU unit tests for scheduler admission of a whole-suffix layer-major extend."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.layer_major.gate import LayerMajorGate
from sglang.srt.managers.schedule_policy import layer_major_admission


def _budget(fits):
    return SimpleNamespace(check_prefill_ring=lambda **kw: fits)


class TestLayerMajorAdmission(unittest.TestCase):
    ARGS = dict(
        req_wants_prompt_logprobs=False,
        req_wants_hidden=False,
        prefix_len=512,
        extend_len=40000,
        total_tokens=41000,
        max_new_tokens=64,
        ring_tokens=4352,
    )

    def test_admits_whole_suffix_when_gate_and_ring_fit(self):
        adm = layer_major_admission(
            gate=LayerMajorGate(min_tokens=32768, max_tokens=262144),
            budget=_budget(True),
            **self.ARGS,
        )
        self.assertEqual(
            (adm.prefix_len, adm.extend_len, adm.max_new_tokens, adm.is_chunked),
            (512, 40000, 64, False),
        )

    def test_falls_back_when_ring_does_not_fit_or_gate_refuses(self):
        gate = LayerMajorGate(min_tokens=32768, max_tokens=262144)
        self.assertIsNone(layer_major_admission(gate=gate, budget=_budget(False), **self.ARGS))
        self.assertIsNone(
            layer_major_admission(gate=gate, budget=_budget(True), **{**self.ARGS, "extend_len": 1000})
        )
        self.assertIsNone(layer_major_admission(gate=None, budget=_budget(True), **self.ARGS))


if __name__ == "__main__":
    unittest.main()
