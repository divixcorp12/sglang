#!/usr/bin/env python3
"""Check the files in this directory against the sha256 table in section 9 of C_MEASUREMENT_PREREG.md (a copy of which sits two levels up
in the staged run directory, or pass --prereg). Prints OK/MISMATCH per file and exits 1 on any mismatch or on a file missing from the table.
Run it immediately before the window; it reads only files.  python3 verify_hashes.py [--prereg PATH]"""
import argparse, hashlib, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
FILES = ("c_harness.py", "c_analysis.py", "nvme_load_reader.py", "quiet_check.py", "prebuild_jit.py", "test_c_harness.py")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prereg", default=str(HERE.parent / "C_MEASUREMENT_PREREG.md")); a = ap.parse_args()
    text = Path(a.prereg).read_text()
    sec = text[text.index("## 9. Frozen artefacts"):text.index("## 10.")]
    bad = 0
    for name in FILES:
        m = re.search(r"\| `c_measurement/%s`[^|]*\| `([0-9a-f]{64})`" % re.escape(name), sec)
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest() if (HERE / name).exists() else None
        if not m: print("NOT IN TABLE  %s" % name); bad += 1
        elif got == m.group(1): print("OK            %s  %s" % (name, got[:16]))
        else: print("MISMATCH      %s  file %s  registered %s" % (name, (got or "missing")[:16], m.group(1)[:16])); bad += 1
    print("ALL MATCH" if not bad else "%d PROBLEM(S): do not run" % bad)
    return 1 if bad else 0
if __name__ == "__main__":
    sys.exit(main())
