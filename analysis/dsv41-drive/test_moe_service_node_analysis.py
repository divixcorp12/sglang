"""Contract checks for ordered Nsight graph-node joins."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from moe_service_node_analysis import analyze


class NodeJoinTest(unittest.TestCase):
    def test_joins_replay_with_host_clock_validation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selected = root / "selected.jsonl"
            selected.write_text("".join(json.dumps(row) + "\n" for row in (
                {"seq": 10, "layer": 0, "type": "demand", "stages_ns": {"observed": 1_000_000_250, "done": 1_000_000_450}},
                {"seq": 11, "layer": 1, "type": "touch", "stages_ns": {"observed": 1_000_000_810, "done": 1_000_000_850}},
            )))
            db_path = root / "nodes.sqlite"
            with sqlite3.connect(db_path) as db:
                db.executescript("""
                    CREATE TABLE ANALYSIS_DETAILS(startTime INTEGER);
                    INSERT INTO ANALYSIS_DETAILS VALUES(1000000000);
                    CREATE TABLE StringIds(id INTEGER, value TEXT);
                    INSERT INTO StringIds VALUES(1,'exl3_ram_miss_post_kernel');
                    INSERT INTO StringIds VALUES(2,'exl3_ram_miss_wait_kernel');
                    INSERT INTO StringIds VALUES(3,'copy_expert_row_segments_gpu_kernel');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(correlationId INTEGER, graphNodeId INTEGER,
                        start INTEGER, end INTEGER, shortName INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,10,100,200,1);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,11,200,500,2);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,12,500,600,3);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,20,700,800,1);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,21,800,900,2);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(5,22,900,1000,3);
                """)
            report = analyze(selected, db_path, 2)
            self.assertEqual(report["replays"], 1)
            self.assertEqual(report["clock_checks"]["demand_observed_inside_post_wait"], 1)
            self.assertEqual(report["joined"][0]["seq"], 10)
            self.assertEqual(report["joined"][1]["seq"], 11)
            self.assertTrue(report["clock_checks"]["all_demand_joins_plausible"])
            self.assertAlmostEqual(report["demand_wait_minus_cpu_service_ms"], 0.0001)

            # A bad per-demand timestamp must not enter any cross-clock mean.
            lines = selected.read_text().splitlines()
            bad = json.loads(lines[0])
            bad["stages_ns"]["observed"] = 1_000_000_550
            selected.write_text(json.dumps(bad) + "\n" + lines[1] + "\n")
            report = analyze(selected, db_path, 2)
            self.assertFalse(report["clock_checks"]["all_demand_joins_plausible"])
            self.assertIsNone(report["exploratory_cross_clock_means_us"])


if __name__ == "__main__":
    unittest.main()
