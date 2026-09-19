"""R4 on the truncated model: greedy decode in up to four Engine runs, compared.

Arms, each a fresh Engine (shut down before the next), with its own environment
(the scheduler subprocess inherits os.environ, so each run sets it first):
  eager   - no CUDA graphs, SGLANG_MOE_EXPERT_GRAPH_GATHER=0 (the exl3_moe_loop path);
  graph   - breakable decode graph at bs 1; graph gather as --graph-gather says;
  debug   - (--debug-arm) the graph arm with --debug-cuda-graph: the same in-graph
            path run eagerly through the capture machinery (graph gather on);
  control - (--control) a second eager run, for run-to-run nondeterminism.
Reports eager_vs_graph and debug_vs_eager against R4's bar (1e-3), graph_vs_debug
bitwise (tolerance 0.0: the capture-correctness gate), eager_vs_eager. Exit status 1
when the capture gate fails: graph_vs_debug with --debug-arm, else eager_vs_graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trace_corpus import GRAPH_KWARGS  # noqa: E402

LOGPROB_TOL = 1e-3
ARMS = ("eager", "graph", "debug", "control")


def compare(eager: dict, graph: dict, *, logprob_tol: float = LOGPROB_TOL) -> dict:
    """``{"tokens": [...], "logprobs": [...]}`` runs -> mismatch step, max |dlogprob|, pass."""
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(eager["tokens"], graph["tokens"])) if a != b),
        None,
    )
    if mismatch is None and len(eager["tokens"]) != len(graph["tokens"]):
        mismatch = min(len(eager["tokens"]), len(graph["tokens"]))
    common = len(eager["tokens"]) if mismatch is None else mismatch + 1
    deltas = [abs(a - b) for a, b in zip(eager["logprobs"][:common], graph["logprobs"][:common])]
    worst = max(deltas, default=0.0)
    return {
        "first_token_mismatch": mismatch,
        "max_abs_dlogprob": worst,
        "pass": mismatch is None and worst <= logprob_tol,
    }


def run_env(arm: str, graph_gather: bool) -> dict[str, str]:
    """Environment overrides of one arm's Engine."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    on = arm == "debug" or (arm == "graph" and graph_gather)
    return {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1" if on else "0"}


def engine_kwargs(model: str, arm: str, mem_fraction: float) -> dict:
    kwargs = dict(
        model_path=model,
        tp_size=1,
        disable_shared_experts_fusion=True,
        context_length=4096,
        mem_fraction_static=mem_fraction,
        max_running_requests=4,
        expert_distribution_recorder_mode="per_pass",
        disable_radix_cache=True,
    )
    if arm in ("graph", "debug"):
        # Engine's default log level is error; the capture line ("Breakable CUDA graph
        # captured: ... breaks=N") is an info log and is what a run checks first.
        kwargs.update(GRAPH_KWARGS, log_level="info")
    else:
        kwargs["disable_cuda_graph"] = True
    if arm == "debug":
        kwargs["debug_cuda_graph"] = True
    return kwargs


def _decode(model: str, ids: list[int], new_tokens: int, arm: str, graph_gather: bool, mem_fraction: float) -> dict:
    import sglang

    env = run_env(arm, graph_gather)
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        engine = sglang.Engine(**engine_kwargs(model, arm, mem_fraction))
        try:
            out = engine.generate(
                input_ids=ids,
                sampling_params={"max_new_tokens": new_tokens, "temperature": 0, "ignore_eos": True},
                return_logprob=True,
            )
        finally:
            engine.shutdown()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    entries = out["meta_info"]["output_token_logprobs"]
    return {"tokens": [e[1] for e in entries], "logprobs": [float(e[0]) for e in entries]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True, help="a text file; its first --prompt-tokens tokens")
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=32)
    # 0.8 fails on the truncated model: its KV pool grows to fill the budget and the sm120
    # FlashMLA page-split buffer (sized to the pool) then asks for another 16.6 GiB.
    p.add_argument("--mem-fraction-static", type=float, default=0.5)
    p.add_argument("--graph-gather", action="store_true", help="the graph arm serves the MoE in-graph")
    p.add_argument("--debug-arm", action="store_true")
    p.add_argument("--control", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.prompt_file) as f:
        ids = tokenizer(f.read()).input_ids[: args.prompt_tokens]
    arms = ["eager", "graph"] + (["debug"] if args.debug_arm else []) + (["control"] if args.control else [])
    runs = {
        arm: _decode(args.model, ids, args.new_tokens, arm, args.graph_gather, args.mem_fraction_static)
        for arm in arms
    }
    report = dict(runs)
    report["eager_vs_graph"] = compare(runs["eager"], runs["graph"])
    if "debug" in runs:
        report["graph_vs_debug"] = compare(runs["debug"], runs["graph"], logprob_tol=0.0)
        report["debug_vs_eager"] = compare(runs["eager"], runs["debug"])
    if "control" in runs:
        report["eager_vs_eager"] = compare(runs["eager"], runs["control"])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if "_vs_" in k}))
    gate = report["graph_vs_debug"] if "debug" in runs else report["eager_vs_graph"]
    sys.exit(0 if gate["pass"] else 1)


if __name__ == "__main__":
    main()
