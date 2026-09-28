"""nvme_report: pattern words, O_DIRECT region rotation and the stale-word summary of the WC NVMe check (CPU)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import nvme_report as report  # noqa: E402


def test_pattern_words_differ_per_trial_and_fit_a_signed_int64():
    words = [report.pattern_word(t) for t in range(1000)]
    assert len(set(words)) == 1000
    assert all(-(1 << 63) <= w < (1 << 63) for w in words)


def test_regions_are_aligned_in_bounds_distinct_and_rotate_over_files():
    files = [f"layer-{i:03d}.rows" for i in range(40)]
    regs = report.regions(300, 1_662_976, file_bytes=5_113_380_864, files=files)
    assert len(regs) == 300 and len(set(regs)) == 300
    for path, offset in regs:
        assert offset % 4096 == 0 and 0 <= offset <= 5_113_380_864 - 1_662_976
    assert len({path for path, _ in regs}) == 40
    with pytest.raises(ValueError):
        report.regions(1, 1000, file_bytes=1 << 20, files=files)  # O_DIRECT needs a 4 KiB multiple


def _trial(slab, size, stale=(0, 0, 0), wrong=(0, 0, 0)):
    return {"kind": "trial", "slab": slab, "size": size, "stale": dict(zip(report.METHODS, stale)),
            "wrong": dict(zip(report.METHODS, wrong))}


def test_summary_counts_stale_and_wrong_words_and_trials_per_slab_method_and_size():
    records = [_trial("wc", 65536), _trial("wc", 65536, stale=(3, 0, 0), wrong=(3, 1, 0)), _trial("pinned", 65536)]
    s = report.summarize(records)
    assert s[("wc", "sm_cv16", 65536)] == {"trials": 2, "stale_words": 3, "wrong_words": 3, "bad_trials": 1}
    assert s[("wc", "weak", 65536)] == {"trials": 2, "stale_words": 0, "wrong_words": 1, "bad_trials": 1}
    assert s[("pinned", "ce", 65536)] == {"trials": 1, "stale_words": 0, "wrong_words": 0, "bad_trials": 0}
