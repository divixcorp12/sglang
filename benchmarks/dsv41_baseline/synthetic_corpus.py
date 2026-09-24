"""Build the timed (+1 warm-up) synthetic single-turn sessions the HTTP driver runs.

Each real corpus session's first-turn text is truncated to `PROMPT_TOKENS` by the
DSV4.1 tokenizer and re-decoded to a string, then wrapped as a one-turn session in
`run_capture_sessions.py`'s own schema (`session_id`, `domain`, `split`, `turns`,
`expected`). This is not new text: it is the same truncation `trace_corpus.py` applied
before feeding it straight to an Engine as token ids; here it goes through the
tokenizer twice (encode then decode) so it survives as a normal chat message over
`/v1/chat/completions` instead. `run_capture_sessions.py` itself is unmodified.

`assert_fits_context` is the explicit check the campaign brief asked for: fail before
sending, not discover a rejected request from a server error.
"""

from __future__ import annotations

import json
from typing import Callable

from session_subset import CONTEXT_LENGTH, NEW_TOKENS, PROMPT_TOKENS

TokenizeFn = Callable[[str], list]
DetokenizeFn = Callable[[list], str]


class ContextBudgetError(RuntimeError):
    pass


def assert_fits_context(
    prompt_tokens: int, *, new_tokens: int = NEW_TOKENS, context_length: int = CONTEXT_LENGTH
) -> None:
    total = prompt_tokens + new_tokens
    if total > context_length:
        raise ContextBudgetError(
            f"{prompt_tokens} prompt tokens + {new_tokens} generation tokens = {total} "
            f"exceeds context_length={context_length}"
        )


def build_synthetic_session(
    session: dict,
    *,
    tokenize: TokenizeFn,
    detokenize: DetokenizeFn,
    prompt_tokens: int = PROMPT_TOKENS,
    new_tokens: int = NEW_TOKENS,
    context_length: int = CONTEXT_LENGTH,
) -> dict:
    raw_text = session["turns"][0]
    ids = tokenize(raw_text)[:prompt_tokens]
    assert_fits_context(len(ids), new_tokens=new_tokens, context_length=context_length)
    truncated_text = detokenize(ids)
    return {
        "session_id": session["session_id"],
        "domain": "dsv41-baseline-synthetic",
        "split": session.get("split", "unknown"),
        "turns": [truncated_text],
        "expected": [None],
        "context_chars": len(truncated_text),
        "source_prompt_tokens": len(ids),
    }


def build_synthetic_sessions(sessions: list[dict], **kwargs) -> list[dict]:
    return [build_synthetic_session(s, **kwargs) for s in sessions]


def write_jsonl(path: str, sessions: list[dict]) -> None:
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
