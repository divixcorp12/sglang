"""Regenerates prompts.jsonl, the fixed parity prompt set, from repo text and arithmetic; no network, no randomness.

prompts.jsonl is checked in and is the source of truth: run this only to regenerate it from the same files at the
same commit. Each row is {id, kind, prompt, continuation, continuation_tokens}. dump_logits.py tokenizes `prompt`
and the first `continuation_tokens` tokens of `continuation`, and teacher-forces the pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("prompts.jsonl")

CONTINUATION_TOKENS = 256
# Enough characters for 256 tokens of dense code (about 2.5 chars/token) with margin.
CONTINUATION_CHARS = 1800
SHORT_PROMPT_CHARS = 1500
# About 4k tokens of English or markdown (3.5-4 chars/token).
LONG_PROMPT_CHARS = 15000

CODE_FILES = ("python/sglang/srt/environ.py", "benchmarks/dsv41_baseline/arm_env.py")
PROSE_FILES = ("DSV_OPTIMIZATION.md", "MOE_EXPERT_TRANSFER.md", "DSV41_REFERENCE.md")

TOOLS = (
    ("get_weather", {"city": "Lisbon", "unit": "celsius"}),
    ("search_docs", {"query": "expert streaming copy engine", "limit": 5}),
    ("run_sql", {"query": "SELECT id, total FROM orders WHERE total > 100 ORDER BY total DESC LIMIT 10"}),
    ("create_event", {"title": "Design review", "start": "2026-11-03T14:00:00Z", "duration_min": 45}),
    ("read_file", {"path": "src/main.rs", "start_line": 1, "end_line": 80}),
    ("convert_currency", {"amount": 250.0, "from": "EUR", "to": "JPY"}),
)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _line_start_at_or_after(text: str, offset: int) -> int:
    newline = text.find("\n", offset)
    return len(text) if newline < 0 else newline + 1


def _split_file(rel: str, *, fraction: float, prompt_chars: int) -> tuple[str, str]:
    """Prompt is the prompt_chars before a line-aligned cut at `fraction` of the file; continuation follows it."""
    text = _read(rel)
    cut = _line_start_at_or_after(text, int(len(text) * fraction))
    return text[max(0, cut - prompt_chars) : cut], text[cut : cut + CONTINUATION_CHARS]


def _row(*, row_id: str, kind: str, prompt: str, continuation: str) -> dict:
    assert len(continuation) >= CONTINUATION_CHARS // 2, row_id
    return {
        "id": row_id,
        "kind": kind,
        "prompt": prompt,
        "continuation": continuation,
        "continuation_tokens": CONTINUATION_TOKENS,
    }


def _file_rows(
    *, kind: str, files: tuple[str, ...], fractions: tuple[float, ...], count: int, prompt_chars: int
) -> list[dict]:
    rows = []
    for index in range(count):
        prompt, continuation = _split_file(
            files[index % len(files)], fraction=fractions[index % len(fractions)], prompt_chars=prompt_chars
        )
        rows.append(_row(row_id=f"{kind}-{index:02d}", kind=kind, prompt=prompt, continuation=continuation))
    return rows


def _gcd_steps(a: int, b: int) -> str:
    lines = []
    while b:
        lines.append(f"{a} = {a // b} * {b} + {a % b}")
        a, b = b, a % b
    lines.append(f"The last nonzero remainder is {a}, so the gcd is {a}.")
    return "\n".join(lines)


def _math_rows() -> list[dict]:
    rows = []
    for index, seed in enumerate((1071, 4620, 9973, 30030, 65537, 123456)):
        pairs = [(seed * (k + 3) + 17 * k, seed + 29 * (k + 1)) for k in range(1, 7)]
        prompt = (
            "Problem: Use the Euclidean algorithm to find the greatest common divisor of "
            f"{pairs[0][0]} and {pairs[0][1]}, showing every division step.\n\nSolution:\n"
        )
        continuation = "\n\n".join(
            f"gcd({max(a, b)}, {min(a, b)}):\n{_gcd_steps(max(a, b), min(a, b))}" for a, b in pairs
        )
        while len(continuation) < CONTINUATION_CHARS:
            continuation += "\n\nCheck: the gcd divides both inputs, and no larger integer does."
        rows.append(_row(row_id=f"math-{index:02d}", kind="math", prompt=prompt, continuation=continuation))
    return rows


def _tool_rows() -> list[dict]:
    schema = json.dumps(
        [{"name": name, "parameters": {key: type(value).__name__ for key, value in args.items()}} for name, args in TOOLS],
        indent=2,
    )
    rows = []
    for index, (name, args) in enumerate(TOOLS):
        prompt = (
            "You may call these tools by emitting a JSON object on its own line with keys `tool` and `arguments`.\n"
            f"Tools:\n{schema}\n\nUser: Please use {name} for me with the details I gave earlier.\nAssistant:"
        )
        calls = [
            f' {json.dumps({"tool": name, "arguments": args})}\nTool result {step + 1}: '
            f'{{"status": "ok", "rows": {step * 3 + 2}, "note": "completed {name} call {step + 1}"}}\n'
            f"Assistant: The call {step + 1} succeeded; I will verify the output and continue."
            for step in range(4)
        ]
        continuation = "\n".join(calls)
        while len(continuation) < CONTINUATION_CHARS:
            continuation += f"\nAssistant: Done with {name}; nothing further is needed."
        rows.append(_row(row_id=f"tool-{index:02d}", kind="tool", prompt=prompt, continuation=continuation))
    return rows


def _long_rows() -> list[dict]:
    specs = (
        ("DSV_OPTIMIZATION.md", 0.35),
        ("DSV41_REFERENCE.md", 0.5),
        ("python/sglang/srt/server_args.py", 0.6),
        ("MOE_EXPERT_TRANSFER.md", 0.7),
    )
    rows = []
    for index, (rel, fraction) in enumerate(specs):
        prompt, continuation = _split_file(rel, fraction=fraction, prompt_chars=LONG_PROMPT_CHARS)
        rows.append(_row(row_id=f"long-{index:02d}", kind="long", prompt=prompt, continuation=continuation))
    return rows


def build_rows() -> list[dict]:
    rows = _file_rows(
        kind="code", files=CODE_FILES, fractions=(0.2, 0.45, 0.7, 0.9), count=8, prompt_chars=SHORT_PROMPT_CHARS
    )
    rows += _math_rows()
    rows += _file_rows(
        kind="prose", files=PROSE_FILES, fractions=(0.15, 0.4, 0.65, 0.85), count=8, prompt_chars=SHORT_PROMPT_CHARS
    )
    rows += _tool_rows()
    rows += _long_rows()
    return rows


def main() -> int:
    rows = build_rows()
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"wrote {len(rows)} prompts to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
