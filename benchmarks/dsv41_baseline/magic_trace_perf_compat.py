#!/usr/bin/python3
"""perf adapter for magic-trace v1.2.4's old trace-start grammar.

Upstream master accepts `tr strt jmp` and strips its `jmp` suffix before decoding:
https://github.com/janestreet/magic-trace/blob/master/src/perf_decode.ml
Only canonicalize that alias, preserving every timestamp/address/event. Recording
and all non-script commands exec the real perf unchanged. Keep original perf.data.
"""
import os
import re
import subprocess
import sys

START = re.compile(r"(^\s*\d+/\d+\s+\d+\.\d+:\s+\d+\s+branches[^ ]*:\s+)tr strt jmp(?=\s+[0-9a-f]+\s)")


def normalize(line):
    return START.sub(r"\1tr strt    ", line)


def main():
    command = ["/usr/bin/perf", *sys.argv[1:]]
    if len(sys.argv) < 2 or sys.argv[1] != "script":
        os.execv(command[0], command)
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
