"""Run corpus first turns through the streamed full model and time each session.

The caller sets the streaming environment: SGLANG_DSV41_EXPERT_STREAM / _DIR /
_TRACE_PATH, the framework's SGLANG_MOE_* tier budgets, and the Engram knobs.
This script drives the Engine and records timings. It turns on the per-pass
expert distribution recorder, which the hot cache's dynamic residency needs.

Sessions run strictly one at a time: the stream trace counts a decode token as a
pass with tokens == 1, so two concurrent generate streams would corrupt it.
"""

from __future__ import annotations

import argparse
import json
import time


def _first_turns(path: str, n: int, skip: int = 0):
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if i >= skip + n:
                return
            yield json.loads(line)["turns"][0]


# Decode as a breakable CUDA graph at batch size 1, prefill eager (the EXL3 gate's only
# graph shape). The MoE runs in-graph once graph gather serves it, else as an eager break.
GRAPH_KWARGS = dict(
    cuda_graph_backend_decode="breakable",
    cuda_graph_backend_prefill="disabled",
    cuda_graph_bs_decode=[1],
    cuda_graph_max_bs_decode=1,
)


def engine_kwargs(args) -> dict:
    """The Engine of every corpus run (it satisfies the EXL3 expert-caching gate)."""
    kwargs = dict(
        model_path=args.model,
        tp_size=1,
        disable_shared_experts_fusion=True,
        context_length=4096,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        # BS1 decode is what Phase 3 measures; DSV4 reserves SWA slots per request.
        max_running_requests=4,
        # The hot cache's residency counts routes through the recorder's forward
        # observer; without it dynamic residency never updates (MOE_EXPERT_TRANSFER.md).
        expert_distribution_recorder_mode="per_pass",
        # A cached prefix would turn a later prompt into a one-token extend, which
        # the stream trace would count as a decode token.
        disable_radix_cache=True,
    )
    if getattr(args, "graphs", False):
        # The capture, RAM-miss thread and hot cache startup lines are info logs.
        kwargs.update(GRAPH_KWARGS, log_level="info")
    else:
        kwargs["disable_cuda_graph"] = True
    return kwargs


def sampling_params(args) -> dict:
    # Every session decodes exactly new_tokens tokens, so tok/s is not inflated by EOS.
    return {"max_new_tokens": args.new_tokens, "temperature": 0, "ignore_eos": True}


def time_stream(stream, new_tokens: int, clock=time.perf_counter) -> dict:
    """Consume one generate stream; return its TTFT and decode tok/s.

    The first chunk carries the first token (end of prefill), so the remaining
    new_tokens - 1 tokens are the decode.
    """
    started = clock()
    first = None
    for _chunk in stream:
        if first is None:
            first = clock()
    if first is None:
        raise RuntimeError("generate stream yielded no chunks; cannot time the session")
    decode_s = clock() - first
    return {
        "ttft_s": first - started,
        "decode_tok_s": (new_tokens - 1) / decode_s if decode_s > 0 else 0.0,
    }


def mean_decode_tok_s(sessions: list) -> float:
    if not sessions:
        raise ValueError("no sessions to average; check --sessions, --n and --skip")
    return sum(s["decode_tok_s"] for s in sessions) / len(sessions)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--skip", type=int, default=0, help="start after this many sessions")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--chunked-prefill-size", type=int, default=512)
    p.add_argument("--graphs", action="store_true", help="breakable decode graphs at batch size 1")
    args = p.parse_args()

    texts = list(_first_turns(args.sessions, args.n, args.skip))
    if not texts:
        # Fail before the Engine launch, which costs minutes of the GPU window.
        raise SystemExit(
            f"no sessions selected from {args.sessions} (--n {args.n}, --skip {args.skip})"
        )

    import sglang
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = sglang.Engine(**engine_kwargs(args))
    sessions = []
    for text in texts:
        ids = tokenizer(text).input_ids[: args.prompt_tokens]
        timing = time_stream(
            engine.generate(input_ids=ids, sampling_params=sampling_params(args), stream=True),
            args.new_tokens,
        )
        sessions.append({"prompt_tokens": len(ids), "new_tokens": args.new_tokens, **timing})
        print(json.dumps(sessions[-1]), flush=True)
    engine.shutdown()
    report = {"per_session": sessions, "mean_decode_tok_s": mean_decode_tok_s(sessions)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
