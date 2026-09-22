import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="stage-a")


class TestExpertDistributionObserver(unittest.TestCase):
    def setUp(self):
        from sglang.srt.eplb import expert_distribution as distribution

        self.distribution = distribution
        config = SimpleNamespace(
            moe=SimpleNamespace(
                expert_distribution_recorder_mode="stat",
                expert_distribution_recorder_buffer_size=-1,
                moe_a2a_backend="none",
            )
        )
        for name, value in (
            ("get_exec", config),
            ("reports_expert_balancedness", False),
            ("get_device", "cpu"),
            ("get_device_namespace", SimpleNamespace(device="cpu")),
        ):
            mocked = patch.object(distribution, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.recorder = distribution._ExpertDistributionRecorderReal(
            SimpleNamespace(num_layers=2, num_physical_experts=4), rank=0
        )
        self.batch = SimpleNamespace()

    def forward(self, step=0):
        with self.recorder.with_forward_pass(step, self.batch):
            with self.recorder.with_current_layer(1):
                self.recorder.on_select_experts(torch.tensor([[0, 1, 1, -1]]))

    def test_observer_and_user_accumulator_share_one_collected_matrix(self):
        recorder = self.recorder
        recorder.start_record()
        observed = []
        recorder.register_forward_observer(
            lambda batch, data: observed.append((batch, data))
        )
        gatherer = next(iter(recorder._single_pass_gatherers.values()))
        with (
            patch.object(gatherer, "collect", wraps=gatherer.collect) as collect,
            patch.object(
                recorder._accumulator, "append", wraps=recorder._accumulator.append
            ) as append,
        ):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                self.forward()
        self.assertEqual(collect.call_count, 1)
        self.assertIs(observed[0][0], self.batch)
        self.assertIs(observed[0][1], append.call_args.args[2])
        self.assertEqual(
            observed[0][1]["global_physical_count"].tolist(), [[0] * 4, [1, 2, 0, 0]]
        )
        self.assertEqual(
            sum(
                event.count
                for event in profile.key_averages()
                if event.key == "aten::scatter_add_"
            ),
            1,
        )
        self.assertEqual(
            recorder._accumulator._global_physical_count_of_buffered_step.get_all().tolist(),
            [[[0] * 4, [1, 2, 0, 0]]],
        )

    def test_observer_keeps_collecting_after_stop_without_consuming_user_output(self):
        recorder = self.recorder
        recorder.start_record()
        self.forward()
        recorder.stop_record()
        before = recorder._accumulator._global_physical_count_of_buffered_step.get_all().clone()
        observed = []
        recorder.register_forward_observer(
            lambda batch, data: observed.append(data["global_physical_count"].clone())
        )
        self.forward(1)
        self.forward(2)
        self.assertFalse(recorder.recording)
        self.assertEqual(len(observed), 2)
        self.assertTrue(torch.equal(observed[0], observed[1]))
        self.assertTrue(
            torch.equal(
                recorder._accumulator._global_physical_count_of_buffered_step.get_all(),
                before,
            )
        )

    def test_noop_registration_is_harmless(self):
        self.distribution._ExpertDistributionRecorderNoop().register_forward_observer(
            lambda batch, data: self.fail("noop must not call observer")
        )

    def test_select_experts_outside_layer_scope_is_ignored(self):
        """A speculative-decoding draft worker (e.g. DSpark) runs its own
        per-stage forward loop and never enters `with_current_layer` -- its
        stage ids (0..N) would otherwise collide with the target's layer-id
        space. A `on_select_experts` call reached with no current layer must
        be dropped, not attributed to some layer or allowed to crash the
        layer-keyed gatherer's tensor indexing.
        """
        recorder = self.recorder
        recorder.start_record()
        gatherer = next(iter(recorder._single_pass_gatherers.values()))
        before = gatherer._data.clone()
        with recorder.with_forward_pass(0, self.batch):
            # No with_current_layer() scope -- this is what an
            # uninstrumented draft-model forward loop does today.
            recorder.on_select_experts(torch.tensor([[0, 1, 1, -1]]))
        self.assertTrue(torch.equal(gatherer._data, before))
        # A legitimate, in-scope call on a real target layer still records.
        self.forward(step=1)
        self.assertEqual(gatherer._data[1].tolist(), [1, 2, 0, 0])
        self.assertEqual(gatherer._data[0].tolist(), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
