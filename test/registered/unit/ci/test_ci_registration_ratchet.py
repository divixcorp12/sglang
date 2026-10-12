"""A ratchet on the CI registration rules: no NEW registered test file may break test collection.

test/run_suite.py globs every file under test/registered and calls collect_tests(sanity_check=True), which raises at
the FIRST file that has no CI registration, has a malformed one, or is registered but has no `if __name__ == "__main__":`
block (CI runs a file as `python3 <file> -f`, so without the block it exits 0 having run nothing). One such file stops
every suite from collecting, and the defect spreads by imitation: a new test written on the shape of a neighbouring file
inherits its missing block. This test parses every file with the project's own parser (ut_parse_one_file, the function
collect_tests calls) and fails when a file outside the lists below has any of the three defects.

The lists are the files that already had a defect when this was written. They are meant to shrink: fixing a file makes
test_the_known_lists_carry_no_fixed_files fail until its line is deleted, so a list can only get shorter. Do not add a
file to fix a red test here; add the block (unittest.main() or sys.exit(pytest.main([__file__]))) or register the file.
"""

import os
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci, ut_parse_one_file

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

REGISTERED = Path(__file__).resolve().parents[2]  # test/registered
SKIPPED_BASENAMES = ("conftest.py", "__init__.py")  # run_suite skips these

# Registered and enabled, but no `if __name__ == "__main__":` block.
KNOWN_NO_MAIN = frozenset(
    {
    "unit/layers/quantization/test_exl3_stream_scope.py",
    "unit/models/test_dspark_exl3_names.py",
    "unit/models/test_qwen2_moe_bcg_streamer_dispatch.py",
    "unit/models/test_qwen4_exp_language_model_only.py",
    "unit/models/test_qwen4_exp_route_trace.py",
    "unit/scripts/test_task1_arm_verdict.py",
    "unit/spec/test_verify_trace.py",
    }
)

# No register_*_ci call at all: collect_tests raises "No CI registry found" on the first such file,
# so a single one stops every suite from collecting. Register the file or move it to test/manual.
KNOWN_NO_REGISTRY = frozenset()

# A registration that is present but malformed: the parser itself raises.
KNOWN_MALFORMED = frozenset(
    {
        "unit/eplb/test_expert_distribution_observer.py",
    }
)


def _files():
    for path in sorted(REGISTERED.rglob("*.py")):
        if path.name not in SKIPPED_BASENAMES:
            yield path


def classify(path):
    """'malformed', 'no_registry', 'no_main' or None, as collect_tests would judge the file."""
    try:
        registries, has_main = ut_parse_one_file(str(path))
    except ValueError:
        return "malformed"
    if not registries:
        return "no_registry"
    if any(r.disabled is None for r in registries) and not has_main:
        return "no_main"
    return None


def _offenders(files=None):
    found = {"malformed": set(), "no_registry": set(), "no_main": set()}
    for path in files if files is not None else _files():
        kind = classify(path)
        if kind:
            found[kind].add(path.relative_to(REGISTERED).as_posix())
    return found


KNOWN = {"malformed": KNOWN_MALFORMED, "no_registry": KNOWN_NO_REGISTRY, "no_main": KNOWN_NO_MAIN}


class TestCiRegistrationRatchet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = _offenders()

    def test_no_new_file_breaks_test_collection(self):
        new = {kind: sorted(cls_files - KNOWN[kind]) for kind, cls_files in self.found.items()}
        self.assertEqual(
            {kind: files for kind, files in new.items() if files},
            {},
            "run_suite's collect_tests would raise on these; give each an `if __name__ == \"__main__\":` block or a "
            "register_*_ci call, and do not add it to the KNOWN lists",
        )

    def test_the_known_lists_carry_no_fixed_files(self):
        stale = {kind: sorted(KNOWN[kind] - files) for kind, files in self.found.items()}
        self.assertEqual(
            {kind: files for kind, files in stale.items() if files},
            {},
            "these files no longer have the defect (or no longer exist): delete their lines from the KNOWN lists",
        )

    def test_this_file_is_itself_collectable(self):
        self.assertIsNone(classify(Path(__file__).resolve()))

    def test_the_scan_covers_the_tree(self):
        # A scan that silently found no files would pass everything above.
        self.assertGreater(len(list(_files())), 1500)


if __name__ == "__main__":
    unittest.main()
