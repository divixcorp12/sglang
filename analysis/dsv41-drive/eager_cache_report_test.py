"""eager_cache_report cuts trace lines and counter snapshots at session boundaries."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eager_cache_report as report  # noqa: E402


def _write(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write('{"kind": "pinned_tier", "t"')  # a line cut by SIGKILL


def test_sessions_are_cut_from_cumulative_snapshots(tmp_path):
    prefix = str(tmp_path / "x-base")
    sessions = [
        {"session": 0, "start_t": 10.0, "end_t": 20.0, "ttft_s": 4.0, "decode_tok_s": 2.0,
         "output_sha1": "a", "disk_bytes": {"nvme0": 0, "nvme2": 2 << 30, "nvme4": 0}},
        {"session": 1, "start_t": 20.5, "end_t": 30.0, "ttft_s": 3.0, "decode_tok_s": 3.0,
         "output_sha1": "b", "disk_bytes": {"nvme0": 0, "nvme2": 1 << 30, "nvme4": 0}},
    ]
    json.dump({"per_session": sessions, "mean_decode_tok_s": 2.5, "startup_disk_bytes": {"nvme0": 0}},
              open(prefix + ".json", "w"))
    trace = [
        {"forward": 1, "layer": 0, "tokens": 256, "vram_miss": 5, "ram_miss": 4, "read_ms": 1000.0, "background_rows": 1, "t": 12.0},
        {"forward": 2, "layer": 0, "tokens": 1, "vram_miss": 2, "ram_miss": 1, "read_ms": 500.0, "background_rows": 0, "t": 15.0},
        {"forward": 3, "layer": 0, "tokens": 256, "vram_miss": 7, "ram_miss": 3, "read_ms": 0.0, "background_rows": 0, "t": 22.0},
        {"kind": "graph_step", "t": 25.0},
    ]
    _write(prefix + ".trace", trace)
    tier = lambda t, **k: {"kind": "pinned_tier", "t": t, "occupancy": 0, "capacity": 100, **k}  # noqa: E731
    _write(prefix + ".trace.cache-stats", [
        tier(9.0, hits=0, admissions=0, evictions=0, lookup_hits=0, lookup_misses=0, populated_bytes=0),
        tier(19.9, hits=10, admissions=6, evictions=1, lookup_hits=10, lookup_misses=20, populated_bytes=600, occupancy=5),
        tier(29.9, hits=40, admissions=9, evictions=4, lookup_hits=40, lookup_misses=25, populated_bytes=900, occupancy=5),
        {"kind": "engram", "t": 9.0, "accesses": 0, "hits": 0, "misses": 0, "evictions": 0},
        {"kind": "engram", "t": 19.9, "accesses": 100, "hits": 25, "misses": 50, "evictions": 0, "filled_rows": 50, "capacity_rows": 200},
    ])
    _, rows = report.arm_rows(prefix)
    first, second = rows
    assert (first["vram_miss_rows"], first["ram_miss_rows"], first["background_rows"]) == (7, 5, 1)
    assert first["ram_miss_rows_prefill"] == 4 and first["forwards"] == 2 and first["decode_calls"] == 1
    assert first["read_time_s"] == 1.5
    assert first["tier"]["admissions"] == 6 and first["tier"]["lookup_misses"] == 20
    assert first["tier_occupancy"] == 5 and first["tier_capacity"] == 100
    assert first["engram_hit_rate"] == 0.25
    assert first["mib_per_read_row"] == 2 * 1024 / 6
    # Session 1 is what accrued between the two snapshots; it has no engram snapshot after its start.
    assert (second["vram_miss_rows"], second["ram_miss_rows"]) == (7, 3)
    assert second["tier"]["admissions"] == 3 and second["tier"]["evictions"] == 3
    assert second["engram_hit_rate"] is None
