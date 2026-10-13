"""L+2 cap-1 evaluation driver: the native gate, T0, T3 and T0 best_cap1 at L+2 under the serving read policy (cap 1,
mean-read budgets 0 to 1), after a cap-2 reproduction of the saved spike-l2 points. Dev and calib only (never
locked_test); trains nothing.

Inputs: a finished R1 run (its frozen plan, exclusions, stages/l2/result.json, spike-l2/result.json), the B1 dataset,
the B0 corpus (stamp windows), the F0 head and r1_cost --cap 1 result files for L2_T0, L2_T3 and L2_T0_best_cap1.
Output: OUT/sanity.json, OUT/scores/<split>/<scorer>.npz (block-score cache), OUT/result.json and OUT/REPORT.md.
Nothing is written outside OUT, and OUT may not be RUN_DIR, inside it, or inside any protected run directory.

Usage: l2_cap1_eval.py RUN_DIR --dataset D --corpus C --b1-head H --out OUT --cost L2_T0=P --cost L2_T3=P
       --cost L2_T0_best_cap1=P [--device cuda]
Exit: 0 done, 1 failure or failed sanity, 2 refused --out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import l2_cap1_core  # noqa: E402
import l2_cap1_metrics  # noqa: E402
import l2_cap1_report  # noqa: E402
import l2_cap1_sanity  # noqa: E402
import l2_cap1_scorers  # noqa: E402
import l2_cap1_timing  # noqa: E402
import r1_arms  # noqa: E402
import r1_groups  # noqa: E402
import r1_metrics  # noqa: E402
import r1_plan  # noqa: E402
import r1_policy  # noqa: E402
import r1_stage  # noqa: E402
import r1_stage1  # noqa: E402
import r1_stamps  # noqa: E402
from l2_cap1_policy import BUDGETS, CAP  # noqa: E402
from r1_groups import GroupScorer, R1Group  # noqa: E402
from r1_policy import Candidates  # noqa: E402

OUT_ROOT = r1_stage.OUT_ROOT
# b1 / R1 / R2 run dirs and the F108 served A/B outputs (RAM_PREFETCH_B1_RESULTS_HANDOFF.md, ROUND 4).
FORBIDDEN = ("b1-run-20261012", "b1-r1-20261012", "b1-r2-20261012", "l2-ab-20261012",
             "l2-ab-20261012-failed-generation")
HORIZON = 2
SPLITS = ("dev", "calib")
SANITY_CAP = 2
SANITY_MEAN_READS = 1.0
SANITY_RULES = {"dev": "select", "calib": "calibrate"}
SANITY_ARMS = ("T0", "T3")
GATE_CACHE_SHA = "native-gate"
FROZEN_BUDGETS = (0.25, 0.5, 0.75, 1.0)
SANITY_NAME = "sanity.json"
SCORES_DIR = "scores"
OBSERVED = "observed offline"


class RefusedOut(ValueError):
    pass


def guard_out(out: str, *, run_dir: str, root: str = OUT_ROOT) -> str:
    """The real --out; refuses a path outside root, in a protected run dir, or equal to or inside run_dir."""
    try:
        real = r1_stage.guard_out_root(out, root=root, forbidden=FORBIDDEN)
    except ValueError as e:
        raise RefusedOut(str(e)) from e
    real_run = os.path.realpath(run_dir)
    if real == real_run or real.startswith(real_run + os.sep):
        raise RefusedOut(f"--out {out!r} is or lies inside RUN_DIR {run_dir!r}, which is read-only")
    return real


def _cost_arg(text: str) -> tuple[str, str]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--cost expects NAME=PATH, got {text!r}")
    return name, path


def _cost_paths(pairs: list[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, path in pairs:
        if name not in l2_cap1_scorers.LEARNED:
            raise ValueError(f"--cost name {name!r} is not one of {l2_cap1_scorers.LEARNED}")
        if name in out:
            raise ValueError(f"--cost {name!r} given twice")
        out[name] = path
    return out


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ===== Block-score cache =====


def group_ids(groups: list[R1Group]) -> list[str]:
    return [f"{k.session}|{k.block}|{k.seq}|{k.source_layer}|{k.target_layer}|{k.horizon}"
            for k in (r.key for r in groups)]


def save_scores(path: str, *, keys: list[str], sha: str, scores: list[np.ndarray]) -> None:
    """One npz per scorer and split: group keys, the checkpoint sha and [G, E] f32 block scores, atomically."""
    block = np.stack([np.asarray(s, dtype=np.float32) for s in scores]) if scores else np.zeros((0, 0), np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, keys=np.array(keys, dtype=str), sha=np.array(sha), scores=block)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_scores(path: str, *, keys: list[str], sha: str) -> list[np.ndarray] | None:
    """The cached block scores, or None when absent or when the group keys or checkpoint sha differ."""
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["sha"]) != sha or z["keys"].tolist() != list(keys):
            return None
        return list(z["scores"])


class _CachedScorer:
    """A GroupScorer whose block scores are read from, or written to, OUT/scores/<split>/<name>.npz."""

    def __init__(self, inner: GroupScorer, *, path: str, sha: str) -> None:
        self._inner = inner
        self._path = path
        self._sha = sha
        self.name = inner.name

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        keys = group_ids(groups)
        cached = load_scores(self._path, keys=keys, sha=self._sha)
        if cached is not None:
            return cached
        scores = [np.asarray(s, dtype=np.float32) for s in self._inner.score_groups(groups)]
        save_scores(self._path, keys=keys, sha=self._sha, scores=scores)
        return scores


# ===== Inputs =====


def _verified_ckpts(l2: dict) -> dict[str, dict]:
    return {name: {"path": path, "sha": l2_cap1_scorers.verify_ckpt(path, sha=sha)}
            for name, (path, sha) in l2_cap1_scorers.learned_ckpts(l2).items()}


def _load_inputs(*, run_dir: str, corpus: str, b1_head_path: str) -> dict:
    plan, plan_sha = r1_plan.load_plan(os.path.join(run_dir, "run_plan.json"))
    manifest = _read(os.path.join(corpus, "manifest.json"))
    windows, audit = r1_stamps.corpus_windows(corpus, blocks=sorted(set(manifest["session_block"].values())),
                                              horizons=(HORIZON,))
    return {"plan": plan, "plan_sha": plan_sha, "windows": windows,
            "exclusions": r1_groups.load_exclusions(os.path.join(run_dir, r1_arms.EXCLUSIONS_REL)),
            "l2": _read(os.path.join(run_dir, "stages", "l2", "result.json")),
            "spike": _read(os.path.join(run_dir, "spike-l2", "result.json")),
            "f0": b1_head.load_head(b1_head_path)[0]}


def _read_groups(dataset: str, *, exclusions) -> dict[str, list[R1Group]]:
    return {split: list(r1_groups.read_r1_groups(dataset, split=split, horizon=HORIZON, exclusions=exclusions))
            for split in SPLITS}


def _load_costs(paths: dict[str, str], *, ckpts: dict[str, dict]) -> dict[str, dict]:
    """Each cost file with its path and sha256; a cost must come from its scorer's own checkpoint (or T0's for
    best_cap1, which the record then says)."""
    out: dict[str, dict] = {}
    for name, path in sorted(paths.items()):
        cost = _read(path)
        own, t0 = ckpts[name]["sha"], ckpts["L2_T0"]["sha"]
        measured_on = cost.get("checkpoint_sha256")
        if measured_on != own and not (name == "L2_T0_best_cap1" and measured_on == t0):
            raise ValueError(f"cost {path} for {name} was measured on {measured_on}, not {own}")
        out[name] = {**cost, "path": path, "file_sha256": r1_stage.sha256_file(path),
                     "cost_source": "own checkpoint" if measured_on == own else "L2_T0 checkpoint (T0_seed0)"}
    return out


# ===== Sanity (C2) =====


def _sanity(groups: dict[str, list[R1Group]], *, inputs: dict, dataset: str, device: str) -> dict:
    """Cap 2 at mean budget 1.0 on the unfiltered groups; SanityError when spike-l2 is not reproduced."""
    l2 = {**inputs["l2"], "arms": {a: inputs["l2"]["arms"][a] for a in SANITY_ARMS}}
    evals = {split: r1_stage1.evaluate_split(
        gs, l2_cap1_sanity.spike_scorers(l2=l2, f0=inputs["f0"], dataset=dataset, split=split, device=device),
        caps=(SANITY_CAP,), mean_reads=SANITY_MEAN_READS, rule=SANITY_RULES[split]) for split, gs in groups.items()}
    return l2_cap1_sanity.check_spike(evals, spike=inputs["spike"], cap=SANITY_CAP)


# ===== Scoring =====


def _cache_shas(ckpts: dict[str, dict]) -> dict[str, str]:
    return {l2_cap1_scorers.GATE: GATE_CACHE_SHA, **{n: c["sha"] for n, c in ckpts.items()}}


def _score(groups: list[R1Group], *, split: str, out: str, shas: dict[str, str], inputs: dict, dataset: str,
           device: str) -> dict[str, list[Candidates]]:
    scorers = l2_cap1_scorers.cap1_scorers(l2_result=inputs["l2"], f0=inputs["f0"], dataset=dataset, split=split,
                                           device=device)
    cached = [_CachedScorer(s, path=os.path.join(out, SCORES_DIR, split, f"{s.name}.npz"), sha=shas[s.name])
              for s in scorers]
    return r1_policy.score_split(groups, cached)


def _partition(groups: list[R1Group], cands: dict[str, list[Candidates]]) -> tuple[dict, dict]:
    """(covered, uncovered), each {"groups": [...], "cands": {scorer: [...]}}, with the C7 filter applied once."""
    covered_idx = [i for i, r in enumerate(groups) if r.g.gate_has_bias]
    uncovered_idx = [i for i, r in enumerate(groups) if not r.g.gate_has_bias]

    def take(idx: list[int]) -> dict:
        return {"groups": [groups[i] for i in idx], "cands": {n: [cs[i] for i in idx] for n, cs in cands.items()}}

    return take(covered_idx), take(uncovered_idx)


# ===== Uncovered (separate, never headline) =====


def _threshold(value: float | None) -> float:
    return math.inf if value is None else float(value)


def _uncovered_point(groups: list[R1Group], cs: list[Candidates], *, scorer: str, threshold: float) -> dict:
    rep = r1_metrics.policy_report(groups, cs, scorer=scorer, cap=CAP, threshold=threshold)
    reads = [r1_policy.select(c, threshold=threshold, cap=CAP) for c in cs]
    c = rep.cells["pooled"]
    point = {"threshold": None if math.isinf(threshold) else threshold, "reads": c.read, "useful": c.useful,
             "wasted": c.read - c.useful, "eligible": c.eligible_all, "forwards": c.forwards,
             "mean_reads": c.mean_reads, "recall": c.recall, "precision": c.precision}
    return {"point": point, "coverage": l2_cap1_metrics.coverage_counts(groups, reads)}


def _uncovered(part: dict, *, covered: dict) -> dict:
    """Each scorer at each budget's covered threshold, on the groups outside the serving-covered set."""
    out: dict = {"groups": len(part["groups"]), "thresholds_from": "covered groups of the same split"}
    if not part["groups"]:
        return out
    out["scorers"] = {name: {key: _uncovered_point(part["groups"], cs, scorer=name,
                                                   threshold=_threshold(b["point"]["threshold"]))
                             for key, b in covered[name]["budgets"].items()}
                      for name, cs in part["cands"].items()}
    return out


# ===== Timing and frozen candidate =====


def _budget_name(scorer: str, budget: float) -> str:
    return f"{scorer}@{budget}"


def _replay(reads: dict, *, groups: dict[str, list[R1Group]], inputs: dict, delays: dict[str, float]) -> dict:
    """One replay row per (scorer, budget), named scorer@budget, each charged its own scorer's delay."""
    evals = {split: {_budget_name(n, b): {CAP: {"reads": r}} for n, by_b in sc.items() for b, r in by_b.items()}
             for split, sc in reads.items()}
    per_row = {row: delays[row.split("@")[0]] for ev in evals.values() for row in ev}
    return l2_cap1_timing.replay_budgets(evals, groups=groups, plan=inputs["plan"], windows=inputs["windows"],
                                         delays=per_row)


def _frozen(cap1: dict, *, timing: dict, case: str) -> dict:
    """Per scorer: the frozen-candidate budget on dev nominal on-time useful, and that budget's calib threshold."""
    cells = timing["measured"]["cases"][case]
    out: dict = {}
    for name in cap1["calib"]:
        pick = l2_cap1_timing.frozen_candidate(
            {b: cells[_budget_name(name, b)]["on_time_useful"] for b in FROZEN_BUDGETS}, budgets=FROZEN_BUDGETS)
        point = cap1["calib"][name]["budgets"][str(pick["budget"])]["point"]
        out[name] = {**pick, "calib_threshold": point["threshold"], "calib_point": point, "case": case}
    return out


# ===== Orchestration =====


def _fail_sanity(out: str, err: Exception) -> int:
    _write_json(os.path.join(out, SANITY_NAME), {"label": OBSERVED, "status": "failed", "error": str(err)})
    print(f"sanity failed: {err}", file=sys.stderr)
    return 1


def run(*, run_dir: str, dataset: str, corpus: str, b1_head_path: str, out: str, cost_paths: dict[str, str],
        device: str, commands: list[str]) -> dict:
    inputs = _load_inputs(run_dir=run_dir, corpus=corpus, b1_head_path=b1_head_path)
    ckpts = _verified_ckpts(inputs["l2"])
    costs = _load_costs(cost_paths, ckpts=ckpts)
    delays = l2_cap1_timing.delays_from_costs(costs)
    all_groups = _read_groups(dataset, exclusions=inputs["exclusions"])

    sanity = _sanity(all_groups, inputs=inputs, dataset=dataset, device=device)
    _write_json(os.path.join(out, SANITY_NAME), sanity)

    covered, uncovered = {}, {}
    for split, gs in all_groups.items():
        cands = _score(gs, split=split, out=out, shas=_cache_shas(ckpts), inputs=inputs, dataset=dataset,
                       device=device)
        covered[split], uncovered[split] = _partition(gs, cands)
    groups = {s: p["groups"] for s, p in covered.items()}
    cap1, reads = l2_cap1_core.evaluate_cap1(groups, {s: p["cands"] for s, p in covered.items()},
                                             gate=l2_cap1_scorers.GATE, budgets=BUDGETS)
    unc = {s: _uncovered(p, covered=cap1[s]) for s, p in uncovered.items()}

    timing = _replay(reads, groups=groups, inputs=inputs, delays=delays)
    frozen = _frozen(cap1, timing=timing, case=inputs["plan"].stage2_gate.timing_case)
    counts = {s: {"covered": len(covered[s]["groups"]), "uncovered": len(uncovered[s]["groups"])} for s in SPLITS}
    return l2_cap1_report.build_result(run_dir=run_dir, out=out, commands=commands, plan_sha=inputs["plan_sha"],
                                       ckpts=ckpts, sanity=sanity, cap1=cap1, uncovered=unc, timing=timing,
                                       costs=costs, frozen=frozen, group_counts=counts)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--b1-head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cost", type=_cost_arg, action="append", default=[], metavar="NAME=PATH",
                   help=f"r1_cost --cap 1 result for each of {', '.join(l2_cap1_scorers.LEARNED)}")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        out = guard_out(a.out, run_dir=a.run_dir)
    except RefusedOut as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    commands = [shlex.join([sys.executable, *(sys.argv if argv is None else [__file__, *argv])])]
    try:
        result = run(run_dir=a.run_dir, dataset=a.dataset, corpus=a.corpus, b1_head_path=a.b1_head, out=out,
                     cost_paths=_cost_paths(a.cost), device=a.device, commands=commands)
    except l2_cap1_sanity.SanityError as e:
        return _fail_sanity(out, e)
    paths = l2_cap1_report.write_outputs(out, result)
    print(json.dumps({"result": paths[0], "report": paths[1]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
