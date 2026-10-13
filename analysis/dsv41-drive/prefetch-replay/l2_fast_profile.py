"""Per-component CUDA-graph profile of r1_cost's complete cap-1 path for the L+2 learned heads T0 and T3.

Splits r1_cost's complete step (fp32 gate, E5 feature build, bf16 head MLP, block aggregation + sigmoid, mask-threshold
select) into components. Each is captured as its own CUDA graph on fixed precomputed inputs and timed over --runs
replays, with its graph node count. "complete" is the unsplit step. Every label is observed offline, at 6 live tokens.

Inputs: an R1 run dir (stages/l2/result.json names the L2_T0 and L2_T3 checkpoints, verified by sha256), run_plan.json
(CostSpec), the model path (target gate) and the target layer id.
Output: --out profile.json {label, scope, tokens, runs, layer, gpu, plan_sha256, models: {L2_T0|L2_T3: {...}}}.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python l2_fast_profile.py --run-dir RUN_DIR
       --plan RUN_DIR/run_plan.json --model-path MODEL --layer 20 --out /mnt/nvme1/nvfp4-work/f123-.../profile.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Callable

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_cap1_scorers  # noqa: E402
import l2_fast_common as common  # noqa: E402
import r1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
from b1_rows import GATE_DEPTH, TOPK  # noqa: E402
from r1_groups import ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training import llapor  # noqa: E402

COMPONENTS = ("gate", "features", "head_mlp", "aggregate_sigmoid", "select", "complete")
PROFILED = ("L2_T0", "L2_T3")
CAP = 1
SCOPE = ("one CUDA graph per component, 6 live tokens, fixed inputs; no L2-defeating rotation, inputs "
         "L2-resident; replays queued behind a GPU sleep so host launch latency is excluded; RTX 5090")


class Steps(msgspec.Struct):
    """Graph-capturable component steps on fixed inputs, plus an eager gate-to-select chain."""

    steps: dict[str, Callable[[], torch.Tensor]]
    chain: Callable[[], torch.Tensor]


def _build_features(scores: torch.Tensor, *, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                    model: r1_models.R1Model) -> torch.Tensor:
    """f32 [n, F]: PCA | mask | route, then for E5 the top-12 gate score and margin (as r1_cost._gate_features)."""
    u = llapor.encode_features(x, ids, weights, pca=model.head.pca, num_experts=model.head.cfg.num_experts)
    if model.head.cfg.variant != "E5":
        return u
    top = torch.topk(scores, GATE_DEPTH, dim=-1)
    in_top = torch.zeros_like(scores).scatter_(1, top.indices, 1.0)
    margin = scores - top.values[:, TOPK - 1:TOPK]
    return torch.cat([u, scores * in_top, margin * in_top], dim=-1)


def component_steps(model: r1_models.R1Model, *, W: torch.Tensor, bias: torch.Tensor, cost: r1_plan.CostSpec,
                    device: str) -> Steps:
    """Component steps of the intended (fp32 gate and features, bf16 head) path, each reading fixed tensors."""
    r1_cost._check_cost_spec(cost)
    dtype = r1_cost._dtype(cost.head_dtype)
    ref = copy.deepcopy(model).float().to(device).eval()
    head = ref if dtype == torch.float32 else copy.deepcopy(ref).to(dtype).eval()
    W, bias = W.float().to(device), bias.float().to(device)
    tokens, experts = cost.live_tokens, ref.head.cfg.num_experts
    st = r1_cost._make_static(tokens=tokens, experts=experts, device=device)
    batch = r1_cost._static_batch(tokens=tokens, dtype=dtype, device=device, experts=experts)

    def gate() -> torch.Tensor:
        return r1_gate.full_gate_scores(st.x, W=W, bias=bias)

    def features_of(scores: torch.Tensor) -> torch.Tensor:
        return _build_features(scores, x=st.x, ids=st.ids, weights=st.weights, model=ref)

    def head_mlp_of(u: torch.Tensor) -> torch.Tensor:
        return head.token_logits(ModelInput(batch=batch, u=u.to(dtype).unsqueeze(0), x=None, s_native=None))

    def aggregate_sigmoid_of(logits: torch.Tensor) -> torch.Tensor:
        if head.aggregator is not None:
            block = head.aggregator(logits, batch=batch)
        else:
            block = r1_models._noisy_or_logit(logits, live=batch.live)
        return torch.sigmoid(block[0].float())

    def select_of(probs: torch.Tensor) -> torch.Tensor:
        return r1_cost.select_torch(probs, skip=st.skip, threshold=st.threshold, cap=CAP)

    with torch.no_grad():
        scores = gate()
        u = features_of(scores)
        logits = head_mlp_of(u)
        probs = aggregate_sigmoid_of(logits)

    def chain() -> torch.Tensor:
        return select_of(aggregate_sigmoid_of(head_mlp_of(features_of(gate()))))

    steps = {"gate": gate, "features": lambda: features_of(scores), "head_mlp": lambda: head_mlp_of(u),
             "aggregate_sigmoid": lambda: aggregate_sigmoid_of(logits), "select": lambda: select_of(probs),
             "complete": chain}
    return Steps(steps=steps, chain=chain)


def profile_model(model: r1_models.R1Model, *, W: torch.Tensor, bias: torch.Tensor, cost: r1_plan.CostSpec,
                  runs: int, warmup: int = 50, device: str = "cuda") -> dict:
    """{component: {median_ms, p90_ms, p99_ms, runs, nodes}} for one head."""
    built = component_steps(model, W=W, bias=bias, cost=cost, device=device)
    out = {}
    for name in COMPONENTS:
        step = built.steps[name]
        graph = r1_cost._capture(step)
        timing = common.replay_stats([graph], runs=runs, warmup=warmup)
        out[name] = {**timing, "nodes": common.graph_nodes(step)}
    return out


def build_profile(*, checkpoints: dict[str, tuple[str, str]], W: torch.Tensor, bias: torch.Tensor,
                  cost: r1_plan.CostSpec, runs: int, device: str = "cuda") -> dict:
    """Verify each checkpoint's sha256, then profile it; keys are the scorer names."""
    models = {}
    for name in PROFILED:
        path, sha = checkpoints[name]
        actual = l2_cap1_scorers.verify_ckpt(path, sha=sha)
        model = r1_models.load_r1(path)[0]
        components = profile_model(model, W=W, bias=bias, cost=cost, runs=runs, device=device)
        models[name] = {"checkpoint_sha256": actual, "components": components}
    return models


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--layer", type=int, required=True, help="target layer id whose gate is timed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=1000)
    args = ap.parse_args(argv)
    out = common.guard_out(args.out, run_dir=args.run_dir)
    plan, plan_sha = r1_plan.load_plan(args.plan)
    if plan.cost.live_tokens != 6:
        raise SystemExit(f"the F123 profile is fixed at 6 live tokens; the plan says {plan.cost.live_tokens}")
    with open(os.path.join(args.run_dir, "stages", "l2", "result.json")) as f:
        checkpoints = l2_cap1_scorers.learned_ckpts(json.load(f))
    W, bias = r1_gate.load_target_gates(args.model_path, layer_ids=[args.layer])[args.layer]
    result = {
        "label": common.OBSERVED, "scope": SCOPE, "tokens": plan.cost.live_tokens, "runs": args.runs,
        "layer": args.layer, "gpu": torch.cuda.get_device_name(), "plan_sha256": plan_sha,
        "models": build_profile(checkpoints=checkpoints, W=W, bias=bias, cost=plan.cost, runs=args.runs),
    }
    common.write_json(out, result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
