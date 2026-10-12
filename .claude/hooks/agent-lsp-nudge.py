#!/usr/bin/env python3
"""PreToolUse hook: when an agent reads or edits code (Python, C++, CUDA, Rust) through grep/sed/cat, Read or Edit, remind it to use
agent-lsp, with the one ToolSearch call that loads its tools. A nudge, never a block: it says so the first three
times in a session, then every tenth time."""

import json
import os
import re
import sys
import tempfile

TOOLS = ",".join(f"mcp__agent-lsp__{t}" for t in (
    "start_lsp", "list_symbols", "find_symbol", "get_symbol_source", "inspect_symbol", "find_references",
    "find_callers", "go_to_definition", "get_diagnostics", "blast_radius", "replace_symbol_body",
    "insert_after_symbol", "insert_before_symbol", "safe_delete_symbol", "rename_symbol"))
READING = re.compile(r"(^|[|;&(]\s*)(grep|rg|sed|cat|head|tail|awk|less|nl)\b")
PY = re.compile(r"\.(py|cpp|cc|cu|cuh|h|hpp|rs)\b|--include[= ]['\"]?\*\.(py|cpp|cc|cu|cuh|h|hpp|rs)|-t\s*(py|cpp|cuda|rust)\b|--type[= ](py|cpp|cuda|rust)\b")
CODE = (".py", ".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp", ".rs")


def python_code(tool: str, inp: dict) -> bool:
    if tool == "Bash":
        cmd = str(inp.get("command") or "")
        return bool(READING.search(cmd) and PY.search(cmd))
    if tool in ("Read", "Edit", "MultiEdit"):
        return str(inp.get("file_path") or "").endswith(CODE)
    if tool == "Grep":
        return str(inp.get("path") or "").endswith(CODE) or str(inp.get("type") or "") in ("py", "cpp", "cuda", "rust") or bool(PY.search(str(inp.get("glob") or "")))
    return False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    tool, inp = payload.get("tool_name", ""), payload.get("tool_input") or {}
    if not python_code(tool, inp):
        return
    state = os.path.join(tempfile.gettempdir(), f"agent-lsp-nudge-{payload.get('session_id', 'x')}")
    try:
        n = int(open(state).read() or 0) + 1
    except (OSError, ValueError):
        n = 1
    try:
        open(state, "w").write(str(n))
    except OSError:
        pass
    if n > 3 and n % 10:
        return
    editing = tool in ("Edit", "MultiEdit")
    say = ("This repo's code (Python, C++/CUDA, Rust) goes through agent-lsp, not "
           + ("Edit: change a function or class with replace_symbol_body, insert_after_symbol or insert_before_symbol, "
              "and keep Edit for imports, module constants and non-code files. " if editing else
              "grep/sed/cat or a whole-file Read: list_symbols for a file's outline, get_symbol_source or inspect_symbol "
              "for one symbol, find_symbol to locate a name, find_references / find_callers for usages. Keep grep for "
              "docs, logs and config. ")
           + f"If its tools aren't loaded yet, load them in one call: ToolSearch with query \"select:{TOOLS}\"; then "
           "call start_lsp once with root_dir = the project folder (the repo root, or python/ for the Python package).")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": say}}))


if __name__ == "__main__":
    main()
