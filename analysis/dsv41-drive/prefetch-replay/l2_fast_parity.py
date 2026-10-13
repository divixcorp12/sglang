"""L+2 fast-path parity: the F109 reference (R1GroupScorer), the folded torch path and the fused spec_head kernels on
every serving-covered dev and calib group, at the F109 frozen calib thresholds (budgets 0.75 and 1.0; D7, never
refit). Dev and calib only (never locked_test); trains nothing.

Cases: 'fp32_gate' feeds both fast paths the reference's own fp32 gate scores (r1_gate.full_gate_scores);
'spec_score' feeds them serving's spec_score kernel scores from bf16 x with a bf16 gate weight (production's default
router dtype, router_fp32 unset) and 'spec_score_float32' with an fp32 gate weight (the router_fp32 variant). A
spec_score* read-set change is reported as a precision candidate with its recall change, never hidden (D2, D3).

Inputs: a finished R1 run (exclusions, stages/l2/result.json), the B1 dataset, the B0 corpus, the F0 head, the model
directory (target gates) and the F109 output directory (result.json, read only).
Output: OUT/parity.json only. OUT may not be RUN_DIR, inside it, or inside any protected run directory.

Usage: l2_fast_parity.py RUN_DIR --dataset D --corpus C --b1-head H --model-path M --f109 F109_DIR
       --out OUT/parity.json [--gate-dtype bfloat16] [--gate-dtype float32] [--device cuda] [--limit N]
Exit: 0 done, 1 failure, 2 refused --out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_rows  # noqa: E402
import l2_cap1_eval  # noqa: E402
import l2_cap1_scorers  # noqa: E402
import l2_fast_common as common  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_metrics  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_arms  # noqa: E402
import r1_cache  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
import r1_policy  # noqa: E402
from r1_groups import MAX_WIDTH, NUM_EXPERTS, R1Group  # noqa: E402

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402

HORIZON = 2
SPLITS = ("dev", "calib")
SCORERS = ("L2_T0", "L2_T3")
BUDGETS = ("0.75", "1.0")
PATHS = ("torch", "kernel")
FP32_GATE = "fp32_gate"
SPEC_SCORE = "spec_score"
PRECISION = "precision candidate"
# Case key and label per spec_score gate-weight dtype (Q1: both are run and reported separately).
GATE_DTYPES = {
    "bfloat16": (torch.bfloat16, SPEC_SCORE, "production default router dtype (router_fp32 unset)"),
    "float32": (torch.float32, f"{SPEC_SCORE}_float32", "router_fp32 variant"),
}
SCOPE = ("eager (no CUDA graph), current stream, one group per launch, live widths 1-6 as the groups have them; "
         "serving-covered dev and calib groups at horizon 2 with R1 exclusions")


class _Head(msgspec.Struct):
    folded: l2_fast_torch.FoldedHead  # fp32, on the device
    kernel: l2_fast_kernels.KernelHead  # bf16 matrices, fp32 biases (D3)


class _GroupInput(msgspec.Struct, frozen=True):
    x: torch.Tensor  # f32 [n, 5120], bf16-exact
    src_ids: torch.Tensor  # int64 [n, 6]
    src_w: torch.Tensor  # f32 [n, 6]
    positions: torch.Tensor  # int64 [n], 0 = anchor
    skip: torch.Tensor  # bool [E], ~available


# ===== Inputs =====


def frozen_thresholds(f109: dict) -> dict[str, dict[str, float | None]]:
    """F109's calib thresholds per learned scorer and budget, as recorded (None = no feasible threshold)."""
    calib = f109["cap1"]["splits"]["calib"]
    return {name: {b: calib[name]["budgets"][b]["point"]["threshold"] for b in BUDGETS} for name in SCORERS}


def _threshold(value: float | None) -> float:
    return math.inf if value is None else float(value)


def _read_split(dataset: str, *, split: str, exclusions: r1_groups.Exclusions, limit: int | None) -> list[R1Group]:
    """The split's serving-covered groups in dataset order; the first ``limit`` only for smoke runs."""
    if split == b1_rows.LOCKED_SPLIT:
        raise PermissionError("l2_fast_parity never reads locked_test (R2); there is no override")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    groups = r1_groups.read_r1_groups(dataset, split=split, horizon=HORIZON, exclusions=exclusions)
    covered = l2_cap1_scorers.split_covered(list(groups))[0]
    return covered if limit is None else covered[:limit]


def _verified_ckpts(l2: dict) -> dict[str, dict]:
    ckpts = l2_cap1_scorers.learned_ckpts(l2)
    return {name: {"path": ckpts[name][0], "sha256": l2_cap1_scorers.verify_ckpt(ckpts[name][0], sha=ckpts[name][1])}
            for name in SCORERS}


def _gate_table(raw: dict[int, tuple[torch.Tensor, torch.Tensor]], *, device: str,
                dtype: torch.dtype) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Per target layer (W in ``dtype``, fp32 bias), on the device."""
    return {layer: (W.to(device=device, dtype=dtype).contiguous(), bias.to(device=device, dtype=torch.float32))
            for layer, (W, bias) in raw.items()}


def _heads(models: dict[str, r1_models.R1Model], *, rows: tuple[l2_fast_torch.FoldRow, ...],
           device: str) -> dict[str, _Head]:
    out = {}
    for name, model in models.items():
        folded = l2_fast_torch.fold(model, rows=rows)
        out[name] = _Head(folded=folded, kernel=l2_fast_kernels.KernelHead(folded, device=device))
    return out


# ===== Scoring =====


def reference_scores(groups: list[R1Group], *, models: dict[str, r1_models.R1Model], f0, x_of,
                     device: str) -> dict[str, list[np.ndarray]]:
    """F109's block scores: R1GroupScorer over r1_cache.collate, chunked as l2_cap1_scorers.cap1_scorers does."""

    def collate(gs):
        return r1_cache.collate(gs, x_of=x_of, pca=f0.pca, cfg=f0.cfg, vocab=f0.vocab, device=device)

    out = {}
    for name, model in models.items():
        scorer = r1_models.R1GroupScorer(model, collate_fn=collate, name=name)
        out[name] = r1_arms._Chunked(scorer, chunk=r1_arms._CALIB_CHUNK).score_groups(groups)
    return out


def _group_input(r: R1Group, *, x_of, device: str) -> _GroupInput:
    return _GroupInput(
        x=torch.from_numpy(np.asarray(x_of(r), dtype=np.float32)).to(device),
        src_ids=torch.from_numpy(r.g.source_ids.astype(np.int64)).to(device),
        src_w=torch.from_numpy(r.g.source_weights.astype(np.float32)).to(device),
        positions=torch.from_numpy(r.positions.astype(np.int64)).to(device),
        skip=torch.from_numpy(~r.labels.available).to(device))


def _gate_scores(inp: _GroupInput, *, case: str, W: torch.Tensor, bias: torch.Tensor,
                 out: torch.Tensor) -> torch.Tensor:
    """Biased fp32 gate scores in rows 0..n-1 of ``out`` [MAX_WIDTH, E]."""
    n = inp.x.shape[0]
    if case == FP32_GATE:
        out[:n] = r1_gate.full_gate_scores(inp.x, W=W, bias=bias)
    else:
        es.run_spec_score(inp.x.to(torch.bfloat16), W, bias, out)
    return out


def _masked(block: torch.Tensor, r: R1Group) -> np.ndarray:
    """As R1GroupScorer reports: the block score where the expert is available, -inf elsewhere."""
    return np.where(r.labels.available, block.float().cpu().numpy(), -np.inf).astype(np.float32)


def _policy_read(r: R1Group, block: np.ndarray, *, threshold: float) -> int | None:
    ids = r1_policy.select(r1_policy.candidates(block, skip=~r.labels.available), threshold=threshold, cap=1)
    return int(ids[0]) if len(ids) else None


def _kernel_group(head: _Head, r: R1Group, *, inp: _GroupInput, scores: torch.Tensor, row: int,
                  thresholds: dict[str, float]) -> tuple[np.ndarray, dict[str, int]]:
    """Block scores, and per budget 1 when the kernel's own cap-1 record differs from the policy read of them."""
    block, mismatch = None, {}
    for budget, thr in thresholds.items():
        count, expert, _, got = head.kernel.score(scores=scores, x=inp.x, src_ids=inp.src_ids, src_w=inp.src_w,
                                                  positions=inp.positions, row=row, skip=inp.skip, threshold=thr)
        block = _masked(got, r) if block is None else block
        mismatch[budget] = int((expert if count else None) != _policy_read(r, block, threshold=thr))
    return block, mismatch


def case_scores(groups: list[R1Group], *, case: str, heads: dict[str, _Head],
                gates: dict[int, tuple[torch.Tensor, torch.Tensor]], x_of, device: str,
                thresholds: dict[str, dict[str, float]]) -> tuple[dict, dict]:
    """({scorer: {torch|kernel: [block scores per group]}}, {scorer: {budget: kernel record mismatches}})."""
    buf = torch.zeros((MAX_WIDTH, NUM_EXPERTS), dtype=torch.float32, device=device)
    got = {name: {path: [] for path in PATHS} for name in heads}
    mismatch = {name: dict.fromkeys(BUDGETS, 0) for name in heads}
    with torch.no_grad():
        for r in groups:
            inp = _group_input(r, x_of=x_of, device=device)
            W, bias = gates[r.g.target_layer]
            scores = _gate_scores(inp, case=case, W=W, bias=bias, out=buf)
            for name, head in heads.items():
                row = l2_fast_torch.row_index(head.folded, target_layer=r.g.target_layer)
                block = l2_fast_torch.score(head.folded, scores=scores, x=inp.x, src_ids=inp.src_ids,
                                            src_w=inp.src_w, positions=inp.positions, row=row)
                got[name]["torch"].append(_masked(block, r))
                kblock, miss = _kernel_group(head, r, inp=inp, scores=scores, row=row, thresholds=thresholds[name])
                got[name]["kernel"].append(kblock)
                for budget, m in miss.items():
                    mismatch[name][budget] += m
    return got, mismatch


def _stored_top12(groups: list[R1Group], *, gates: dict[int, tuple[torch.Tensor, torch.Tensor]], x_of,
                  device: str) -> dict[str, int]:
    """Groups whose fp32 recompute's top-12 differs from the dataset's stored gate ids (tie-only counted apart)."""
    mismatched = tie_only = 0
    with torch.no_grad():
        for r in groups:
            x = torch.from_numpy(np.asarray(x_of(r), dtype=np.float32)).to(device)
            W, bias = gates[r.g.target_layer]
            p = r1_gate.top12_parity(r.g, scores=r1_gate.full_gate_scores(x, W=W, bias=bias).cpu())
            mismatched += int(p.mismatched > 0)
            tie_only += int(p.mismatched == 0 and p.tie_only > 0)
    return {"mismatched": mismatched, "tie_only": tie_only}


# ===== Tables =====


def _table(groups: list[R1Group], *, ref: list[np.ndarray], got: list[np.ndarray], threshold: float,
           case: str, record_mismatch: int | None) -> dict:
    t = l2_fast_metrics.parity_table(groups, ref=ref, got=got, threshold=threshold)
    if case != FP32_GATE:
        t["recall_change"] = {**l2_fast_metrics.recall_change(groups, ref=ref, got=got, threshold=threshold),
                              "label": PRECISION}
    if record_mismatch is not None:
        t["kernel_record_mismatch"] = record_mismatch
    return t


def tables(groups: list[R1Group], *, ref: dict[str, list[np.ndarray]], got: dict, mismatch: dict,
           thresholds: dict[str, dict[str, float]], case: str) -> dict:
    """{scorer: {torch|kernel: {budget: parity_table (+ recall_change, + kernel_record_mismatch)}}}."""
    return {name: {path: {b: _table(groups, ref=ref[name], got=got[name][path], threshold=thresholds[name][b],
                                    case=case, record_mismatch=mismatch[name][b] if path == "kernel" else None)
                          for b in BUDGETS}
                   for path in PATHS}
            for name in got}


# ===== Orchestration =====


def _split_cases(groups: list[R1Group], *, split: str, dataset: str, models: dict, heads: dict, gates: dict,
                 f0, thresholds: dict, device: str) -> tuple[dict, dict]:
    """({case: tables}, stored top-12 counts) for one split."""
    x_of = r1_cache.make_x_of(dataset, split=split)
    ref = reference_scores(groups, models=models, f0=f0, x_of=x_of, device=device)
    top12 = _stored_top12(groups, gates=gates[FP32_GATE], x_of=x_of, device=device)
    out = {}
    for case, table in gates.items():
        got, mismatch = case_scores(groups, case=case, heads=heads, gates=table, x_of=x_of, device=device,
                                    thresholds=thresholds)
        out[case] = tables(groups, ref=ref, got=got, mismatch=mismatch, thresholds=thresholds, case=case)
    return out, top12


def run(*, run_dir: str, dataset: str, corpus: str, b1_head_path: str, model_path: str, f109_dir: str,
        gate_dtypes: tuple[str, ...], device: str, limit: int | None, commands: list[str]) -> dict:
    inputs = l2_cap1_eval._load_inputs(run_dir=run_dir, corpus=corpus, b1_head_path=b1_head_path)
    recorded = frozen_thresholds(l2_cap1_eval._read(os.path.join(f109_dir, "result.json")))
    thresholds = {name: {b: _threshold(v) for b, v in t.items()} for name, t in recorded.items()}
    ckpts = _verified_ckpts(inputs["l2"])
    groups = {s: _read_split(dataset, split=s, exclusions=inputs["exclusions"], limit=limit) for s in SPLITS}
    every = [r for gs in groups.values() for r in gs]
    models = {name: r1_models.load_r1(c["path"])[0].to(device) for name, c in ckpts.items()}
    heads = _heads(models, rows=l2_fast_torch.serving_rows(every), device=device)
    raw = r1_gate.load_target_gates(model_path, layer_ids=sorted({r.g.target_layer for r in every}))
    gates = {FP32_GATE: _gate_table(raw, device=device, dtype=torch.float32),
             **{GATE_DTYPES[d][1]: _gate_table(raw, device=device, dtype=GATE_DTYPES[d][0]) for d in gate_dtypes}}

    cases: dict = {case: {} for case in gates}
    top12: dict = {}
    for split, gs in groups.items():
        per_case, top12[split] = _split_cases(gs, split=split, dataset=dataset, models=models, heads=heads,
                                              gates=gates, f0=inputs["f0"], thresholds=thresholds, device=device)
        for case, t in per_case.items():
            cases[case][split] = t
    return _result(recorded=recorded, ckpts=ckpts, groups=groups, top12=top12, cases=cases,
                   gate_dtypes=gate_dtypes, limit=limit, commands=commands, f109_dir=f109_dir)


def _result(*, recorded: dict, ckpts: dict, groups: dict, top12: dict, cases: dict, gate_dtypes: tuple[str, ...],
            limit: int | None, commands: list[str], f109_dir: str) -> dict:
    return {
        "label": common.OBSERVED,
        "scope": {"text": SCOPE, "limit": limit, "smoke": limit is not None},
        "gate_dtypes": {FP32_GATE: {"dtype": "float32", "label": "reference fp32 gate (r1_gate.full_gate_scores)"},
                        **{GATE_DTYPES[d][1]: {"dtype": d, "label": GATE_DTYPES[d][2]} for d in gate_dtypes}},
        "thresholds": recorded,
        "thresholds_source": {"path": os.path.join(f109_dir, "result.json"), "split": "calib",
                              "rule": "frozen, read only, never refit (D7)"},
        "ckpts": ckpts,
        "group_counts": {s: len(gs) for s, gs in groups.items()},
        "stored_top12_mismatch": {s: c["mismatched"] for s, c in top12.items()},
        "stored_top12_tie_only": {s: c["tie_only"] for s, c in top12.items()},
        "commands": list(commands),
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--b1-head", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--f109", required=True, help="F109 output directory; only its result.json is read")
    p.add_argument("--out", required=True, help="OUT/parity.json")
    p.add_argument("--gate-dtype", action="append", choices=tuple(GATE_DTYPES), default=None,
                   help="spec_score gate-weight dtype; repeatable (default: bfloat16 and float32)")
    p.add_argument("--device", default="cuda", choices=("cuda",), help="the spec_score / spec_head kernels need CUDA")
    p.add_argument("--limit", type=int, default=None, help="first N covered groups per split; smoke tests only")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        out = common.guard_out(a.out, run_dir=a.run_dir)
    except common.RefusedOut as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    commands = [shlex.join([sys.executable, *(sys.argv if argv is None else [__file__, *argv])])]
    gate_dtypes = tuple(dict.fromkeys(a.gate_dtype or GATE_DTYPES))
    result = run(run_dir=a.run_dir, dataset=a.dataset, corpus=a.corpus, b1_head_path=a.b1_head,
                 model_path=a.model_path, f109_dir=a.f109, gate_dtypes=gate_dtypes, device=a.device, limit=a.limit,
                 commands=commands)
    common.write_json(out, result)
    print(json.dumps({"parity": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
