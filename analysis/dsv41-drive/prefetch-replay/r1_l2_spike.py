"""L+2 spike: the native gate applied two layers early against R1's L+2 heads, cap 2 at mean reads <= 1, plus the
horizon-2 timing replay. Reads a finished R1 run (its frozen plan, exclusions, cost and l2 checkpoints) and writes only
<run>/spike-l2/result.json. Exploratory; dev and calib only (never locked_test).

Usage: r1_l2_spike.py RUN_DIR --dataset DATASET --corpus CORPUS --b1-head HEAD [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_arms  # noqa: E402
import r1_cache  # noqa: E402
import r1_controls  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r1_run  # noqa: E402
import r1_stage1  # noqa: E402
import r1_stamps  # noqa: E402

HORIZON = 2
SPLITS = ("dev", "calib")


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _scorers(*, l2: dict, f0: b1_head.B1Head, dataset: str, split: str, device: str) -> list:
    x_of = r1_cache.make_x_of(dataset, split=split)

    def collate(gs):
        return r1_cache.collate(gs, x_of=x_of, pca=f0.pca, cfg=f0.cfg, vocab=f0.vocab, device=device)

    out = [r1_controls.GateGroupScorer(floors=None)]
    for name, arm in sorted(l2["arms"].items()):
        model = r1_models.load_r1(arm["ckpt_path"])[0].to(device)
        scorer = r1_models.R1GroupScorer(model, collate_fn=collate, name=f"L2_{name}")
        out.append(r1_arms._Chunked(scorer, chunk=r1_arms._CALIB_CHUNK))
    return out


def run(*, run_dir: str, dataset: str, corpus: str, b1_head_path: str, device: str) -> dict:
    plan, plan_sha = r1_plan.load_plan(os.path.join(run_dir, "run_plan.json"))
    exclusions = r1_groups.load_exclusions(os.path.join(run_dir, r1_arms.EXCLUSIONS_REL))
    l2 = _read(os.path.join(run_dir, "stages", "l2", "result.json"))
    delays = r1_run._scorer_delays(_read(os.path.join(run_dir, "stages", "cost", "result.json")))
    f0 = b1_head.load_head(b1_head_path)[0]
    manifest = _read(os.path.join(corpus, "manifest.json"))
    windows, audit = r1_stamps.corpus_windows(corpus, blocks=sorted(set(manifest["session_block"].values())),
                                              horizons=(HORIZON,))
    groups: dict[str, list] = {}
    evals: dict[str, dict] = {}
    for split in SPLITS:
        groups[split] = list(r1_groups.read_r1_groups(dataset, split=split, horizon=HORIZON, exclusions=exclusions))
        scorers = _scorers(l2=l2, f0=f0, dataset=dataset, split=split, device=device)
        rule = "select" if split == plan.selection_metric.split else "calibrate"
        evals[split] = r1_stage1.evaluate_split(groups[split], scorers, caps=(plan.cap_primary,),
                                                mean_reads=plan.mean_reads, rule=rule)
    timing = r1_stage1._timing(evals, groups=groups, plan=plan, windows=windows, scorer_delay_ms=delays)
    cap = plan.cap_primary
    return {"label": "spike, exploratory on reused data; L+2 only", "plan_sha256": plan_sha, "horizon": HORIZON,
            "scorer_delay_ms": delays, "stamp_audit": msgspec.to_builtins(audit),
            "points": {s: {n: caps[cap]["point"] for n, caps in ev.items()} for s, ev in evals.items()},
            "timing": timing}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--b1-head", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    a = p.parse_args(argv)
    result = run(run_dir=a.run_dir, dataset=a.dataset, corpus=a.corpus, b1_head_path=a.b1_head, device=a.device)
    out = os.path.join(a.run_dir, "spike-l2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({s: {n: (round(v["recall"], 4), round(v["precision"], 4), v["reads"]) for n, v in pts.items()}
                      for s, pts in result["points"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
