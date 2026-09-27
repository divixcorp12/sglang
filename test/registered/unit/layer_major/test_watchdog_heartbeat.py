from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layer_major.heartbeat import current_heartbeat
from sglang.srt.managers.scheduler_components import invariant_checker


class TestWatchdogHeartbeat(unittest.TestCase):
    def test_watchdog_counter_advances_with_heartbeat(self):
        captured = {}

        def fake_watchdog(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        scheduler = SimpleNamespace(forward_ct=7, is_initializing=False, cur_batch_for_debug=object())
        with mock.patch.object(invariant_checker, "WatchdogRaw", side_effect=fake_watchdog):
            invariant_checker.create_scheduler_watchdog(scheduler, watchdog_timeout=300)
        before = captured["get_counter"]()
        current_heartbeat().tick()
        self.assertEqual(captured["get_counter"](), before + 1)
        scheduler.forward_ct += 1
        self.assertEqual(captured["get_counter"](), before + 2)


if __name__ == "__main__":
    unittest.main()
