"""The hot path's observable behavior, pinned at ba01695c35 (plan 2026-09-29-hotpath-zero-overhead Task 2): the tier's
slots and map, the lease entries, the row results, piece words and CopyDone the device reads, the functional
counters, the bytes of every READY row (and their identity with the checkpoint), and the SQEs a row-image read
prepares. Every later phase of the plan (packed-path deletion, the Build policy, allocation and lock removal) must
leave this file green without editing the golden. Regenerate only at master's code:
``python test/registered/unit/kernels/test_expert_stream_hotpath_golden.py --regen``."""

import json
import sys
from pathlib import Path

import torch

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test import hotpath_script as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

GOLDEN = Path(__file__).parent / "golden" / "hotpath_golden.json"
SQE_REQUESTS = [(0, [0], [0]), (0, [1, 2, 3], [1, 2, 3]), (1, [7, 0, 5, 2], [0, 1, 2, 3])]


def scenario(tmp_path):
    s, page, host, sim, dst = hp.build_host(tmp_path)
    try:
        return hp.run_script(s, page, host, sim, dst)
    finally:
        host.stop()


def sqe_golden(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, layers=2, experts=8, row_images=True, mirror_weights=(1.0, 1.0))
    out = []
    for row, experts, slots in SQE_REQUESTS:
        result, sqes, info, _record = read_rows_sqes(s.tables, row, experts, slots, direct=True)
        oracle = s.reference(s.tables.layer_ids[row], experts)
        exact = all(same_bytes(s.slabs[row][n][slot], oracle[n][i])
                    for n in EXL3_STREAMED_NAMES for i, slot in enumerate(slots))
        out.append({"result": result, "sqes": [list(q) for q in sqes], "exact": exact,
                    "info": {k: info[k] for k in ("sqes", "descriptors", "credit")}})
    return out


def test_the_scripted_scenario_matches_the_golden(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    snaps = scenario(tmp_path)
    for snap in snaps:
        for row in snap["rows"].values():
            assert all(entry["exact"] for entry in row["ready"].values()), "a READY row differs from the checkpoint"
        assert all(copy["exact"] for copy in snap["copies"]), "a copy-engine destination row differs from the checkpoint"
        assert snap["page"]["fatal"] == 0
    assert snaps[-1]["copies"], "the scenario completed no copy-engine copy, so no destination bytes were checked"
    records = snaps[-1]["page"]["records"]
    assert len(records) == sum(kind == "post" for kind, *_ in hp.SCRIPT)
    assert all(r["ring_seq"] == r["seq"] for r in records), "a posted record's sequence word was overwritten"
    # The byte and ring checks above hold whatever the golden says; then every snapshot must match it.
    assert len(snaps) == len(golden["scenario"])
    for step, (got, want) in enumerate(zip(snaps, golden["scenario"])):
        assert json.loads(json.dumps(got)) == want, f"step {step} ({hp.SCRIPT[step][0]}) diverged"


def test_row_image_reads_prepare_the_golden_sqes_and_land_exact_bytes(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    got = sqe_golden(tmp_path)
    assert json.loads(json.dumps(got)) == golden["sqes"]
    assert all(entry["exact"] and entry["result"] == 1 for entry in got)


if __name__ == "__main__" and "--regen" in sys.argv:
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path.home()) as a, tempfile.TemporaryDirectory(dir=Path.home()) as b:
        # write_fake_exl3 writes into an existing directory (pytest's tmp_path is one), and ram_miss_setup puts the
        # row images beside it, so each fixture gets a fresh subdirectory of its own temporary directory.
        (Path(a) / "s").mkdir()
        (Path(b) / "q").mkdir()
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps({"scenario": scenario(Path(a) / "s"), "sqes": sqe_golden(Path(b) / "q")},
                                     indent=1, sort_keys=True))
    print("wrote", GOLDEN)
