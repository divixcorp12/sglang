#!/usr/bin/env python3
"""PreToolUse hook: when an agent rewrites a repo file with a script or a shell redirect, remind it to use Edit (or Write
for a new file). Reading and editing code through agent-lsp is enforced elsewhere: by ~/.claude/hooks/lsp-enforce.py
in your own sessions, and by hx itself for its agents."""

import json
import re
import sys

# A shell command that rewrites a file in place: python's open(..., "w"), sed -i, perl -pi, a redirect or tee.
REWRITE = re.compile(r"open\([^)]*,\s*['\"][wa]['\"]|\.write_text\(|\bsed\s+(-[a-zA-Z]*\s+)*-i|\bperl\s+-[a-z]*p[a-z]*i|\btee\s+(-a\s+)?[\w./~-]+\.\w+|(^|[^>&0-9])>{1,2}\s*[\w./~-]+\.(md|py|cpp|cc|cu|cuh|h|hpp|rs|json|toml|yaml|yml|txt)\b")
SCRATCH = re.compile(r"/tmp/|/private/tmp/|scratchpad|\$TMPDIR|mktemp")


def rewrites_file(tool: str, inp: dict) -> bool:
    cmd = str(inp.get("command") or "") if tool == "Bash" else ""
    return bool(cmd and REWRITE.search(cmd) and not SCRATCH.search(cmd))


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    tool, inp = payload.get("tool_name", ""), payload.get("tool_input") or {}
    if rewrites_file(tool, inp):
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext":
            "Change files in this repo with Edit (or Write for a new file), not a python/sed/perl script or a shell "
            "redirect: Edit works on Markdown and every other text file, shows the change, and keeps the diff reviewable. "
            "For code, use agent-lsp's symbol edits. Scripts are fine for scratch files under /tmp."}}))


if __name__ == "__main__":
    main()
