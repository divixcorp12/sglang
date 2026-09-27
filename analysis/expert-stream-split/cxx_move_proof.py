"""Certify that a C++ commit is a pure relocation of line blocks (expert-stream split plan, Task 1).

Every non-structural line of every source at the base commit must appear in exactly one manifest block. Every
destination at HEAD, with structural lines removed, must equal its blocks concatenated in manifest order.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

STRUCTURAL = re.compile(
    r"^\s*($|#include\b|#pragma once\b|namespace [\w:]+ \{$|\}\s*// namespace\b|using tvm::ffi::TensorView;$)"
)


def _base_lines(base: str, path: str) -> list[str]:
    text = subprocess.run(["git", "show", f"{base}:{path}"], check=True, capture_output=True, text=True).stdout
    return text.split("\n")


def _body(lines: list[str], new_file: bool) -> list[str]:
    """Drop structural lines, and a new file's leading comment (authored). Only comment and blank lines before the
    first other line count as leading: a moved block that opens with a comment sits after `#pragma once`."""
    start = 0
    if new_file:
        while start < len(lines) and (lines[start].startswith("//") or not lines[start].strip()):
            start += 1
    return [line.rstrip() for line in lines[start:] if not STRUCTURAL.match(line)]


def main(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    base = manifest["base"]
    sources = {path: _base_lines(base, path) for path in manifest["sources"]}
    owner: dict[tuple[str, int], str] = {}
    errors = []
    for dst, blocks in manifest["files"].items():
        expected = []
        for src, first, last in blocks:
            for number in range(first, last + 1):
                key = (src, number)
                if key in owner:
                    errors.append(f"{src}:{number} is in two blocks ({owner[key]} and {dst})")
                owner[key] = dst
            expected += _body(sources[src][first - 1 : last], new_file=False)
        new_file = dst not in sources
        actual = _body(Path(dst).read_text().split("\n"), new_file=new_file)
        if actual != expected:
            for i, (a, e) in enumerate(zip(actual + [""] * len(expected), expected + [""] * len(actual))):
                if a != e:
                    errors.append(f"{dst}: body line {i + 1}: expected {e!r}, found {a!r}")
                    break
    for src, lines in sources.items():
        for number, line in enumerate(lines, start=1):
            if (src, number) not in owner and not STRUCTURAL.match(line):
                errors.append(f"{src}:{number} is in no block: {line.strip()[:80]!r}")
    for error in errors:
        print(error)
    print("PASS" if not errors else f"FAIL ({len(errors)})")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
