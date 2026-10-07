#!/usr/bin/env python3
"""Narrow v1.2.4 adapter: retain raw perf; canonicalize trace-start jump only."""
import os
import re
import subprocess
import sys

START = re.compile(r"(^\s*\d+/\d+\s+\d+\.\d+:\s+\d+\s+branches[^ ]*:\s+)tr strt jmp(?=\s+[0-9a-f]+\s)")


def normalize(line):
    return START.sub(r"\1tr strt    ", line)


def main():
    real = os.environ.get("MAGIC_TRACE_REAL_PERF", "/usr/bin/perf")
    if os.path.realpath(real) == os.path.realpath(__file__):
        raise SystemExit("real perf must not point to this adapter")
    command = [real, *sys.argv[1:]]
    if len(sys.argv) < 2 or sys.argv[1] != "script":
        os.execv(real, command)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        for line in child.stdout:
            sys.stdout.write(normalize(line))
        sys.stdout.flush()
    except BrokenPipeError:
        child.terminate()
    finally:
        child.wait()
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
