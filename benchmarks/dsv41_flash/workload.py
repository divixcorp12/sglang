"""The load: prompts from the corpus, and a closed-loop driver over an Engine's async_generate."""

from __future__ import annotations

import asyncio
import json
import time

from metrics import request_record


def select_prompts(
    *, sessions_path: str, tokenizer, skip: int, count: int, input_tokens: int
) -> list[dict]:
    """First turns of the corpus sessions, tokenised and cut to exactly input_tokens; shorter sessions are skipped."""
    out = []
    with open(sessions_path) as f:
        for line_no, line in enumerate(f):
            if line_no < skip:
                continue
            ids = tokenizer(json.loads(line)["turns"][0]).input_ids
            if len(ids) >= input_tokens:
                out.append({"session_line": line_no, "input_ids": ids[:input_tokens]})
                if len(out) == count:
                    return out
    raise RuntimeError(
        f"only {len(out)} of {count} sessions from {sessions_path} (skip {skip}) have {input_tokens}+ tokens"
    )


def sampling_params(*, output_tokens: int) -> dict:
    # ignore_eos makes every request decode exactly output_tokens, so tokens/s is not inflated by early stops.
    return {"max_new_tokens": output_tokens, "temperature": 0, "ignore_eos": True}


async def _one(generate, *, index: int, input_ids: list, params: dict, clock) -> dict:
    from sglang.benchmark.stream_metrics import validate_finish_reason

    submit_t = clock()
    log, text, meta = [], "", None
    stream = await generate(input_ids=input_ids, sampling_params=params, stream=True)
    async for chunk in stream:
        meta = chunk["meta_info"]
        log.append((clock(), meta["completion_tokens"]))
        text = chunk["text"]
    end_t = clock()
    validate_finish_reason(meta["finish_reason"], ignore_eos=True)
    return request_record(
        index=index,
        prompt_tokens=len(input_ids),
        submit_t=submit_t,
        log=log,
        end_t=end_t,
        text=text,
    )


async def run_closed_loop(
    generate, *, prompts: list, params: dict, concurrency: int, clock=time.perf_counter
) -> tuple[list, float]:
    """``concurrency`` requests in flight until every prompt has run; returns the records in prompt order and the wall time."""
    gate = asyncio.Semaphore(concurrency)

    async def run(index: int, prompt: dict) -> dict:
        async with gate:
            return await _one(
                generate,
                index=index,
                input_ids=prompt["input_ids"],
                params=params,
                clock=clock,
            )

    started = clock()
    records = await asyncio.gather(*(run(i, p) for i, p in enumerate(prompts)))
    return list(records), clock() - started
