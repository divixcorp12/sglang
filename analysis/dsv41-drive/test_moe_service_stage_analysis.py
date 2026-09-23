"""The analysis must not mistake trace-drain time or prefill for decode service."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from moe_service_stage_analysis import analyze, request_intervals, select_decode_suffix, summarize


def record(
    seq: int, layer: int, observed: int, done: int, *, request_type: str = "demand", trace_time: float = 999.0
) -> dict:
    return {
        "kind": "ram_miss_request",
        "schema": 5,
        "layer": layer,
        "request": {"seq": seq, "type": request_type, "ok": 1, "lanes": 2, "backlog": 0},
        "status": "served" if request_type == "demand" else "touch",
        "rows_asked": 1 if request_type == "demand" else 0,
        "stages_ns": {
            "observed": observed,
            "reserved": observed + 10,
            "submit": observed + 20,
            "first_cqe": observed + 45,
            "last_cqe": observed + 70,
            "pack_start": observed + 50,
            "pack_end": observed + 80,
            "mapped": observed + 90,
            "done": done,
        },
        "untraced": {"rows": 0, "extents": 0},
        "t": trace_time,
    }


class ServiceStageAnalysisTest(unittest.TestCase):
    def test_layer_summary_and_coverage_flags(self) -> None:
        rows = [record(1, 0, 100, 200), record(2, 1, 300, 400, request_type="touch")]
        rows[0]["untraced"]["rows"] = 1
        summary = summarize(rows)
        self.assertEqual(summary["layers"]["0"]["groups"]["demand_read"]["n"], 1)
        self.assertEqual(summary["layers"]["1"]["groups"]["touch"]["n"], 1)
        self.assertFalse(summary["validity"]["complete_coverage"])

    def test_successful_no_read_demand_is_valid(self) -> None:
        row = record(1, 0, 100, 200)
        row["status"] = "no_read"
        row["rows_asked"] = 0
        self.assertTrue(summarize([row])["validity"]["all_served"])

    def test_duplicate_session_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "trace").write_text(json.dumps(record(1, 0, 1_100_000_000, 1_200_000_000)) + "\n")
            (root / "results").write_text('''{"session_id":"same","completion_tokens":1}\n{"session_id":"same","completion_tokens":1}\n''')
            (root / "boundaries").write_text('''{"label":"server_ready","monotonic":1.0}\n{"label":"session_same","monotonic":2.0}\n''')
            with self.assertRaisesRegex(ValueError, "duplicate session_id"):
                analyze(root / "trace", root / "results", root / "boundaries", (0,))

    def test_service_tail_after_result_boundary_stays_with_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [
                record(1, 0, 1_100_000_000, 1_200_000_000),
                record(2, 1, 1_400_000_000, 1_500_000_000),
                record(3, 0, 3_100_000_000, 3_200_000_000),
                record(4, 1, 3_400_000_000, 3_500_000_000),
            ]
            (root / "trace").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "results").write_text('''{"session_id":"one","completion_tokens":1}\n{"session_id":"two","completion_tokens":1}\n''')
            (root / "boundaries").write_text('''{"label":"server_ready","monotonic":1.0}\n{"label":"session_one","monotonic":1.35}\n{"label":"session_two","monotonic":3.35}\n''')
            report, selected = analyze(root / "trace", root / "results", root / "boundaries", (0, 1))
            self.assertEqual([row["seq"] for row in selected], [1, 2, 3, 4])
            self.assertGreater(report["sessions"][0]["service_tail_after_boundary_ms"], 0)

    def test_rejects_overlapping_session_selection(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [record(seq, (seq - 1) % 2, int(t * 1e9), int((t + 0.05) * 1e9))
                    for seq, t in enumerate((1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9), 1)]
            (root / "trace").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "results").write_text('''{"session_id":"one","completion_tokens":2}\n{"session_id":"two","completion_tokens":2}\n''')
            (root / "boundaries").write_text('''{"label":"server_ready","monotonic":1.0}\n{"label":"session_one","monotonic":1.45}\n{"label":"session_two","monotonic":1.95}\n''')
            with self.assertRaisesRegex(ValueError, "selected requests begin after session boundary|overlapping session selection"):
                analyze(root / "trace", root / "results", root / "boundaries", (0, 1))

    def test_selects_decode_suffix_by_native_stamps_and_rejects_prefill(self) -> None:
        rows = [
            record(1, 0, 110, 190),
            record(2, 1, 200, 280),
            record(3, 0, 310, 390),
            record(4, 1, 400, 480, request_type="touch"),
            record(5, 0, 510, 590),
            record(6, 1, 600, 680, request_type="touch"),
            record(7, 0, 1010, 1090),  # outside the window even though drained earlier
        ]
        selected = select_decode_suffix(rows, start_ns=100, end_ns=900, replays=2, layers=(0, 1))
        self.assertEqual([r["request"]["seq"] for r in selected], [3, 4, 5, 6])

    def test_rejects_a_missing_stage_record_in_decode_suffix(self) -> None:
        rows = [record(3, 0, 310, 390), record(5, 0, 510, 590), record(6, 1, 600, 680)]
        with self.assertRaisesRegex(ValueError, "expected 4 request records"):
            select_decode_suffix(rows, start_ns=100, end_ns=900, replays=2, layers=(0, 1))

    def test_read_intervals_preserve_read_pack_overlap(self) -> None:
        r = record(3, 0, 1000, 1100)
        intervals = request_intervals(r)
        self.assertEqual(intervals["cpu_total_ns"], 100)
        self.assertEqual(intervals["reserved_ns"], 10)
        self.assertEqual(intervals["reader_setup_ns"], 10)
        self.assertEqual(intervals["read_window_ns"], 50)
        self.assertEqual(intervals["pack_exposed_tail_ns"], 10)
        self.assertEqual(intervals["publish_signal_ns"], 10)

    def test_no_read_does_not_subtract_zero_io_stamps(self) -> None:
        r = record(3, 0, 1000, 1100)
        r["rows_asked"] = 0
        for key in ("submit", "first_cqe", "last_cqe", "pack_start", "pack_end"):
            r["stages_ns"][key] = 0
        intervals = request_intervals(r)
        self.assertEqual(intervals["cpu_total_ns"], 100)
        self.assertIsNone(intervals["read_window_ns"])
        self.assertIsNone(intervals["pack_exposed_tail_ns"])


if __name__ == "__main__":
    unittest.main()
