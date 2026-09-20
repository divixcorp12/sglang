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
import os
import sys
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
    dspark_draft = getattr(args, "dspark", None)
    if getattr(args, "graphs", False) and not dspark_draft:
        # The capture, RAM-miss thread and hot cache startup lines are info logs.
        kwargs.update(GRAPH_KWARGS, log_level="info")
    else:
        # The EXL3 expert-caching gate refuses speculation under a decode CUDA
        # graph, so a --dspark run is always eager regardless of --graphs.
        kwargs["disable_cuda_graph"] = True
    if dspark_draft:
        kwargs.update(
            speculative_algorithm="DSPARK",
            speculative_draft_model_path=dspark_draft,
            speculative_dspark_block_size=5,
        )
    return kwargs


def sampling_params(args) -> dict:
    # Every session decodes exactly new_tokens tokens, so tok/s is not inflated by EOS.
    # --stop-at-eos turns that off for an acceptance-rate run, where tokens past
    # EOS would bias the accept length in either direction.
    return {
        "max_new_tokens": args.new_tokens,
        "temperature": 0,
        "ignore_eos": not getattr(args, "stop_at_eos", False),
    }


def time_stream(stream, new_tokens: int, clock=time.perf_counter) -> dict:
    """Consume one generate stream; return its TTFT and decode tok/s.

    The first chunk carries the first token (end of prefill), so the remaining
    new_tokens - 1 tokens are the decode.

    When a chunk is a dict with a "meta_info" field (the Engine's normal output
    shape), the last chunk's completion_tokens and spec_verify_ct (present once
    speculative decoding is active) are carried into the result. A later slice
    divides them to get the accept length; this only captures the raw fields.
    """
    started = clock()
    first = None
    last_meta_info = None
    last_text = None
    for chunk in stream:
        if first is None:
            first = clock()
        if isinstance(chunk, dict):
            last_meta_info = chunk.get("meta_info", last_meta_info)
            # The Engine streams cumulative text, so the last chunk carries the
            # whole completion. Greedy parity compares these across arms.
            if chunk.get("text") is not None:
                last_text = chunk["text"]
    if first is None:
        raise RuntimeError("generate stream yielded no chunks; cannot time the session")
    decode_s = clock() - first
    # With EOS honoured the session can stop early, so the decoded count is the
    # server's completion_tokens when it reports one, not the requested ceiling.
    decoded = new_tokens
    if last_meta_info and last_meta_info.get("completion_tokens"):
        decoded = int(last_meta_info["completion_tokens"])
    result = {
        "ttft_s": first - started,
        "decode_tok_s": (decoded - 1) / decode_s if decode_s > 0 and decoded > 1 else 0.0,
    }
    if last_text is not None:
        result["output_text"] = last_text
    if last_meta_info:
        if "completion_tokens" in last_meta_info:
            result["completion_tokens"] = last_meta_info["completion_tokens"]
        if "spec_verify_ct" in last_meta_info:
            result["spec_verify_ct"] = last_meta_info["spec_verify_ct"]
    return result


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
    p.add_argument(
        "--dspark",
        metavar="DRAFT_DIR",
        help="run DSpark speculative decoding with this draft checkpoint dir "
        "(forces eager decode; the EXL3 gate refuses speculation under a decode graph)",
    )
    p.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="honour EOS instead of decoding exactly --new-tokens; use for an "
        "acceptance-rate run, where tokens past EOS would bias the accept length",
    )
    args = p.parse_args()

    texts = list(_first_turns(args.sessions, args.n, args.skip))
    if not texts:
        # Fail before the Engine launch, which costs minutes of the GPU window.
        raise SystemExit(
            f"no sessions selected from {args.sessions} (--n {args.n}, --skip {args.skip})"
        )

    import sglang
    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import provenance

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prov = provenance.capture({"trace_corpus": os.path.abspath(__file__)})
    prov["drive_idle_check"] = provenance.drive_idle_check()
    engine = sglang.Engine(**engine_kwargs(args))
    prov["sglang_env_drift_at_engine_ready"] = provenance.env_drift(prov["sglang_env"], provenance.process_env())
    sessions = []
    for text in texts:
        ids = tokenizer(text).input_ids[: args.prompt_tokens]
        chunk_log = []
        cpu_before = provenance.process_tree_cpu_s()
        timing = time_stream(
            provenance.timed_chunks(
                engine.generate(input_ids=ids, sampling_params=sampling_params(args), stream=True),
                chunk_log,
            ),
            args.new_tokens,
        )
        cpu_after = provenance.process_tree_cpu_s()
        sessions.append(
            {
                "prompt_tokens": len(ids),
                "new_tokens": args.new_tokens,
                "cpu_s": None if None in (cpu_before, cpu_after) else cpu_after - cpu_before,
                "step_latency": provenance.step_latency(chunk_log),
                **timing,
            }
        )
        print(json.dumps(sessions[-1]), flush=True)
    engine.shutdown()
    report = {"provenance": prov, "per_session": sessions, "mean_decode_tok_s": mean_decode_tok_s(sessions)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
