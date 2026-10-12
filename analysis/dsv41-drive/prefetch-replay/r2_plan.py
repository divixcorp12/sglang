"""R2 run plan: T5 and G0 specs, seeds, x budget, the cost-increment rule and the R2 continuation gate.

default_plan builds the r2_plan.json content from a loaded R1 plan (D4-D11); freeze_plan writes it once beside a
sha256 file before any training stage runs and load_plan reads it back. continuation_gate (D10) evaluates plain result
dicts against the plan's thresholds.

Inputs: the R1 run path, its plan and plan sha256, input sha256s (R1 checkpoints included), dataset, corpus, b1 head.
Usage: import r2_plan; sha = r2_plan.freeze_plan(f"{out}/r2_plan.json", r2_plan.default_plan(r1_run=r1, r1_plan=p,
       r1_plan_sha=s, inputs=hashes, dataset=d, corpus=c, b1_head=h, out_dir=out))
"""

from __future__ import annotations

import hashlib
import os
import statistics
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)

import r1_plan  # noqa: E402

PLAN_NAME = "r2_plan.json"
SHA_NAME = "r2_plan.sha256"
_EPS = 1e-9  # thresholds are inclusive, as in r1_plan


class R2ContinuationGate(msgspec.Struct, frozen=True, kw_only=True):
    cap: int = 2
    mean_reads: float = 1.0
    relative_recall: float = 0.10
    extra_reads: float = 0.05
    cost_ms: float = 0.1
    min_seeds: int = 3
    timing_case: str = "nominal"
    cost_path: str = ("G0 is charged its correction path given native scores (block aggregation, filtering and "
                      "selection included); T5 and T3 their head-only path")
    rule: str = ("candidate recall_cap2 >= (1 + relative_recall) x the native gate's and above T3's median; at calib "
                 "no lower precision and reads <= (1 + extra_reads) x the gate's; incremental cost <= cost_ms; every "
                 "seed beats the gate and the seed median meets the bar; the bar holds without flagged sessions; "
                 "on-time useful rows under nominal contention above the gate's. It does not rewrite B2.")


class R2Plan(msgspec.Struct, frozen=True, kw_only=True):
    r1_run: str
    r1_plan_sha: str
    inputs: dict[str, str]
    dataset: str
    corpus: str
    b1_head: str
    out_dir: str
    arms: tuple[r1_plan.ArmSpec, ...]
    timing: r1_plan.TimingAssumptions
    cost: r1_plan.CostSpec
    selection_metric: r1_plan.SelectionMetric
    calibration_rule: r1_plan.CalibrationRule
    groups_per_step: int
    patience: int
    seeds: tuple[int, ...] = (0, 1, 2)
    x_dtype: str = "bfloat16"
    x_budget_bytes: int = 16 << 30
    # Arbitrary headroom kept free on the GPU beyond the x budget; only bounds out-of-memory risk.
    x_margin_bytes: int = 2 << 30
    cap_primary: int = 2
    cap_secondary: int = 1
    mean_reads: float = 1.0
    continuation_gate: R2ContinuationGate = R2ContinuationGate()
    l2_horizon: int = 2
    l2_rule: str = ("train the better of T5 and G0 at L+2 with seed 0, even if it does not beat T3, and compare it to "
                    "the L+2 native gate and R1's L+2 T3 ladder checkpoint")
    parent_rule: str = "T5 starts from T3's seed-0 R1 checkpoint copied into the run; G0 trains from scratch"
    splits: tuple[str, ...] = ("train", "dev", "calib")
    label: str = r1_plan.LABEL


def _arms() -> tuple[r1_plan.ArmSpec, ...]:
    # T3's recipe (R1 ladder): block+route 0.25 with position features and no adapter.
    t5 = r1_plan.ArmSpec(name="T5", parent="T3", objective="block+route", route_weight=0.25, epochs=30, warmup=3,
                         lr_new=1e-3, lr_inherited=1e-4, lr_projection=1e-4, position=True, adapter_rank=0,
                         projection=True)
    g0 = r1_plan.ArmSpec(name="G0", parent=None, objective="block+route", route_weight=0.25, epochs=60, warmup=5,
                         g0=True)
    return (t5, g0)


def default_plan(*, r1_run: str, r1_plan: r1_plan.RunPlan, r1_plan_sha: str, inputs: dict[str, str], dataset: str,
                 corpus: str, b1_head: str, out_dir: str) -> R2Plan:
    """Timing, cost and selection constants are copied from the loaded R1 plan (D7, D8)."""
    return R2Plan(
        r1_run=r1_run,
        r1_plan_sha=r1_plan_sha,
        inputs=dict(sorted(inputs.items())),
        dataset=dataset,
        corpus=corpus,
        b1_head=b1_head,
        out_dir=out_dir,
        arms=_arms(),
        timing=r1_plan.timing,
        cost=r1_plan.cost,
        selection_metric=r1_plan.selection_metric,
        calibration_rule=r1_plan.calibration_rule,
        groups_per_step=r1_plan.groups_per_step,
        patience=r1_plan.patience,
        cap_primary=r1_plan.cap_primary,
        cap_secondary=r1_plan.cap_secondary,
        mean_reads=r1_plan.mean_reads,
    )


def _encode(plan: R2Plan) -> bytes:
    return msgspec.json.format(msgspec.json.encode(plan), indent=1) + b"\n"


def _sha_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), SHA_NAME)


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def freeze_plan(path: str, plan: R2Plan) -> str:
    """Write the plan at ``path`` and r2_plan.sha256 beside it; an identical plan is kept, a different one refuses."""
    data = _encode(plan)
    sha = hashlib.sha256(data).hexdigest()
    if os.path.exists(path):
        with open(path, "rb") as f:
            existing = hashlib.sha256(f.read()).hexdigest()
        if existing != sha:
            raise FileExistsError(f"{path} holds a different frozen plan (sha256 {existing}, new {sha})")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        _write_atomic(path, data)
    _write_atomic(_sha_path(path), f"{sha}  {PLAN_NAME}\n".encode())
    return sha


def load_plan(path: str) -> tuple[R2Plan, str]:
    """The plan and its sha256; raises when the sha256 file beside it disagrees."""
    with open(path, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    sha_path = _sha_path(path)
    if os.path.exists(sha_path):
        with open(sha_path) as f:
            recorded = f.read().split()[0]
        if recorded != sha:
            raise ValueError(f"{path} sha256 {sha} does not match {sha_path} ({recorded})")
    return msgspec.json.decode(data, type=R2Plan), sha


def _verdict(met: bool) -> str:
    return "met" if met else "not met"


def _meets_ratio(value: float, base: float, ratio: float) -> bool:
    return value > 0 and value >= (1.0 + ratio) * base - _EPS


def _seed_check(seeds: dict, *, base: float, spec: R2ContinuationGate) -> tuple[bool, str]:
    recalls = [float(seeds[k]) for k in sorted(seeds, key=str)]
    if len(recalls) < spec.min_seeds:
        return False, f"seeds: {len(recalls)} given, needs {spec.min_seeds}: not met"
    median = statistics.median(recalls)
    ok = _meets_ratio(median, base, spec.relative_recall) and all(r > base for r in recalls)
    return ok, f"seed median recall {median:.4f}, min {min(recalls):.4f} vs gate {base:.4f}: {_verdict(ok)}"


def continuation_gate(*, candidate: dict, gate: dict, t3_median: float, cost_ms: float, seeds: dict, noflag: dict,
                      nominal: dict, spec: R2ContinuationGate = R2ContinuationGate()) -> r1_plan.GateOutcome:
    """D10: candidate against the native gate and T3's median over recall, calib precision/reads, cost, seeds, flags, timing.

    candidate and gate hold recall_cap2, precision_calib and reads_calib; seeds maps seed -> recall_cap2; noflag and
    nominal hold candidate/control values (recall without flagged sessions; on-time useful rows).
    """
    arms = {
        "recall": _meets_ratio(candidate["recall_cap2"], gate["recall_cap2"], spec.relative_recall),
        "t3": candidate["recall_cap2"] > t3_median,
        "precision": candidate["precision_calib"] >= gate["precision_calib"] - _EPS,
        "reads": candidate["reads_calib"] <= (1.0 + spec.extra_reads) * gate["reads_calib"] + _EPS,
        "cost": cost_ms <= spec.cost_ms + _EPS,
        "noflag": _meets_ratio(noflag["candidate"], noflag["control"], spec.relative_recall),
        "nominal": nominal["candidate"] > nominal["control"],
    }
    arms["seeds"], seed_reason = _seed_check(seeds, base=gate["recall_cap2"], spec=spec)
    reasons = (
        f"recall at cap {spec.cap} {candidate['recall_cap2']:.4f} vs gate {gate['recall_cap2']:.4f} "
        f"(needs +{spec.relative_recall:.0%} relative): {_verdict(arms['recall'])}",
        f"recall vs T3 median {t3_median:.4f} (needs above): {_verdict(arms['t3'])}",
        f"calib precision {candidate['precision_calib']:.4f} vs {gate['precision_calib']:.4f} (needs no lower): "
        f"{_verdict(arms['precision'])}",
        f"calib reads {candidate['reads_calib']} vs {gate['reads_calib']} (needs <= +{spec.extra_reads:.0%}): "
        f"{_verdict(arms['reads'])}",
        f"incremental scorer cost {cost_ms:.4f} ms per layer (needs <= {spec.cost_ms}): {_verdict(arms['cost'])}",
        seed_reason,
        f"without flagged sessions {noflag['candidate']:.4f} vs {noflag['control']:.4f}: {_verdict(arms['noflag'])}",
        f"on-time useful rows ({spec.timing_case}) {nominal['candidate']} vs {nominal['control']} (needs above): "
        f"{_verdict(arms['nominal'])}",
    )
    values = {"candidate": dict(candidate), "gate": dict(gate), "t3_median": t3_median, "cost_ms": cost_ms,
              "seeds": {str(k): v for k, v in seeds.items()}, "noflag": dict(noflag), "nominal": dict(nominal),
              "arms": arms}
    return r1_plan.GateOutcome(passed=all(arms.values()), reasons=reasons, values=values)
