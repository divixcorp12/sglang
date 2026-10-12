"""R1 run plan: the arm ladder, recipe, selection and calibration rules, both gates and the frozen timing/cost constants.

default_plan builds the run_plan.json content (D3, D5, D7-D10, D16, D17); freeze_plan writes it once beside a sha256
file before any training stage runs, and load_plan reads it back with that sha. stage2_gate (D3) and
continuation_gate (D10) evaluate plain result dicts against the plan's thresholds.

Inputs: the dataset, corpus and B1 head paths plus their sha256s; result dicts from Stage 1, training, cost and replay.
Usage: import r1_plan; sha = r1_plan.freeze_plan(f"{out}/run_plan.json", r1_plan.default_plan(dataset=d, corpus=c,
       b1_head=h, out_dir=out, inputs=hashes))
"""

from __future__ import annotations

import hashlib
import os
import statistics
import sys
from typing import Literal

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)

LABEL = "exploratory on reused data; independent confirmation pending"
PLAN_NAME = "run_plan.json"
SHA_NAME = "run_plan.sha256"
_EPS = 1e-9  # thresholds are inclusive; absorbs float error as b1_select does


class ArmSpec(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    parent: str | None
    objective: Literal["b1_token", "block", "block+route"]
    route_weight: float
    epochs: int
    warmup: int
    lr_new: float = 1e-3
    lr_inherited: float = 1e-4
    lr_projection: float | None = None
    weight_decay: float = 1e-2
    dropout: float = 0.1
    freeze_encoder: bool = False
    freeze_embeddings: bool = False
    position: bool = False
    adapter_rank: int = 0
    projection: bool = False
    g0: bool = False
    horizon: int = 1


class TimingAssumptions(msgspec.Struct, frozen=True, kw_only=True):
    row_read_ms: float
    # Owner's D16 multipliers on row-read time, DSV41_REFERENCE section 33.16 (reads slow while others run).
    contention: dict[str, float] = msgspec.field(
        default_factory=lambda: {"zero": 1.0, "nominal": 2.0, "adverse": 3.0})
    contention_source: str = "DSV41_REFERENCE 33.16"
    row_read_source: str = "DSV41_REFERENCE 33.19: one 2.2-2.4 ms mirrored row read"
    scorer_delay_source: str = "stages/cost: complete scorer median on the bf16 path"


class CostSpec(msgspec.Struct, frozen=True, kw_only=True):
    # Owner's D17: bf16 head and aggregator over an fp32 gate, with the all-fp32 path timed beside it.
    gate_dtype: str = "float32"
    head_dtype: str = "bfloat16"
    aggregator_dtype: str = "bfloat16"
    reference_dtype: str = "float32"
    gate_path: str = "bf16"
    diff_metric: str = "fraction of groups whose capped read set differs vs fp32"
    live_tokens: int = 6


class SelectionMetric(msgspec.Struct, frozen=True, kw_only=True):
    split: str = "dev"
    metric: str = "best eligible-row recall along an exact sorted-threshold sweep"
    cap: int = 2
    mean_reads: float = 1.0
    # D8 v2: scorer cost is reported but not used as a tie-break.
    ties: tuple[str, ...] = ("fewer parameters",)
    cap1: str = "reported as its own lane; never vetoes or rescues cap 2"
    seeds: str = "median dev score across seeds; the seed-0 checkpoint is the candidate"
    early_stop: str = "this metric on dev, patience epochs without improvement"


class CalibrationRule(msgspec.Struct, frozen=True, kw_only=True):
    split: str = "calib"
    scope: str = "one global threshold per model and horizon, capped policy"
    cap: int = 2
    mean_reads: float = 1.0
    rule: str = "lowest threshold with mean reads per forward-layer <= mean_reads"
    ties: str = "equal reads go to the higher threshold"


class Stage2Gate(msgspec.Struct, frozen=True, kw_only=True):
    split: str = "dev"
    cap: int = 2
    mean_reads: float = 1.0
    relative_recall: float = 0.10
    timing_case: str = "nominal"
    rule: str = ("F0 recall_cap2 >= (1 + relative_recall) x the best native control's, and F0's on-time useful rows "
                 "under nominal contention above that control's; otherwise stop after Stage 1")


class ContinuationGate(msgspec.Struct, frozen=True, kw_only=True):
    cap: int = 2
    mean_reads: float = 1.0
    relative_recall: float = 0.10
    extra_reads: float = 0.05
    cost_ms: float = 0.1
    timing_case: str = "nominal"
    min_seeds: int = 3
    rule: str = ("recall_cap2 >= (1 + relative_recall) x the best native control's; at calib no lower precision and "
                 "reads <= (1 + extra_reads) x; bf16 scorer median <= cost_ms per layer; the seed median meets the "
                 "recall bar and every seed beats the control; the bar holds without flagged sessions; on-time useful "
                 "rows under nominal contention above the control's. It does not rewrite B2.")


class RunPlan(msgspec.Struct, frozen=True, kw_only=True):
    arms: tuple[ArmSpec, ...]
    timing: TimingAssumptions
    cost: CostSpec
    inputs: dict[str, str]
    out_dir: str
    dataset: str
    corpus: str
    b1_head: str
    screen_seed: int = 0
    repeat_seeds: tuple[int, ...] = (1, 2)
    groups_per_step: int = 512
    patience: int = 8
    cap_primary: int = 2
    cap_secondary: int = 1
    mean_reads: float = 1.0
    selection_metric: SelectionMetric = SelectionMetric()
    calibration_rule: CalibrationRule = CalibrationRule()
    stage2_gate: Stage2Gate = Stage2Gate()
    continuation_gate: ContinuationGate = ContinuationGate()
    late_arms: tuple[str, ...] = ("T5", "G0")
    ladder_rule: str = ("each arm starts from its parent's copied checkpoint; "
                        "if an arm adds nothing, the next builds on its parent")
    followup_horizons: tuple[int, ...] = (2,)
    followup_rule: str = "L+2 trains the selected architecture and its matched control; L+4 only if L+2 adds useful lead"
    splits: tuple[str, ...] = ("train", "dev", "calib")
    label: str = LABEL


class GateOutcome(msgspec.Struct, frozen=True, kw_only=True):
    passed: bool
    reasons: tuple[str, ...]
    values: dict


def _arms() -> tuple[ArmSpec, ...]:
    # Epochs and warmups: RAM_PREFETCH_B1_RESULTS_HANDOFF.md R1.8 (60 from scratch, 30 fine-tuning).
    joint = {"objective": "block+route", "route_weight": 0.25, "epochs": 30, "warmup": 3}
    return (
        ArmSpec(name="F0", parent=None, objective="b1_token", route_weight=0.0, epochs=0, warmup=0,
                freeze_encoder=True, freeze_embeddings=True),
        ArmSpec(name="T0", parent=None, objective="b1_token", route_weight=0.0, epochs=60, warmup=5),
        ArmSpec(name="T1", parent="T0", objective="block", route_weight=0.0, epochs=30, warmup=3,
                freeze_encoder=True, freeze_embeddings=True),
        ArmSpec(name="T2", parent="T1", **joint),
        ArmSpec(name="T3", parent="T2", position=True, **joint),
        ArmSpec(name="T4", parent="T3", position=True, adapter_rank=16, **joint),
        ArmSpec(name="T5", parent="T4", position=True, adapter_rank=16, projection=True, lr_projection=1e-4, **joint),
        ArmSpec(name="G0", parent=None, objective="block+route", route_weight=0.25, epochs=60, warmup=5, g0=True),
    )


def default_plan(*, dataset: str, corpus: str, b1_head: str, out_dir: str, inputs: dict[str, str]) -> RunPlan:
    """The frozen R1 plan; every frozen constant (D16 contention, D17 dtypes, row-read time) is set here."""
    return RunPlan(
        arms=_arms(),
        # D16 contention defaults on TimingAssumptions (DSV41_REFERENCE 33.16); row read from 33.19's 2.2-2.4 ms.
        timing=TimingAssumptions(row_read_ms=2.2),
        # D17: bf16 head+aggregator, fp32 gate, fp32 reference path timed beside it.
        cost=CostSpec(),
        inputs=dict(sorted(inputs.items())),
        out_dir=out_dir,
        dataset=dataset,
        corpus=corpus,
        b1_head=b1_head,
    )


def _encode(plan: RunPlan) -> bytes:
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


def freeze_plan(path: str, plan: RunPlan) -> str:
    """Write the plan at ``path`` and run_plan.sha256 beside it; an identical plan is kept, a different one refuses."""
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


def load_plan(path: str) -> tuple[RunPlan, str]:
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
    return msgspec.json.decode(data, type=RunPlan), sha


def _best_control(controls: dict[str, dict]) -> str:
    if not controls:
        raise ValueError("no native controls to compare against")
    return min(controls, key=lambda name: (-controls[name]["recall_cap2"], name))


def _meets_ratio(value: float, base: float, ratio: float) -> bool:
    return value > 0 and value >= (1.0 + ratio) * base - _EPS


def _verdict(met: bool) -> str:
    return "met" if met else "not met"


def stage2_gate(*, f0: dict, controls: dict[str, dict], spec: Stage2Gate = Stage2Gate()) -> GateOutcome:
    """D3: Stage 2 runs only when F0 beats the best native control on dev recall_cap2 and on-time useful rows."""
    best = _best_control(controls)
    ctl = controls[best]
    recall_ok = _meets_ratio(f0["recall_cap2"], ctl["recall_cap2"], spec.relative_recall)
    timing_ok = f0["on_time_useful_nominal"] > ctl["on_time_useful_nominal"]
    reasons = (
        f"dev recall at cap {spec.cap} {f0['recall_cap2']:.4f} vs {best} {ctl['recall_cap2']:.4f} "
        f"(needs +{spec.relative_recall:.0%} relative): {_verdict(recall_ok)}",
        f"on-time useful rows ({spec.timing_case}) {f0['on_time_useful_nominal']} vs {best} "
        f"{ctl['on_time_useful_nominal']} (needs above): {_verdict(timing_ok)}",
    )
    values = {"best_control": best, "f0": dict(f0), "control": dict(ctl), "recall": recall_ok, "timing": timing_ok}
    return GateOutcome(passed=recall_ok and timing_ok, reasons=reasons, values=values)


def _seed_checks(seeds: dict, *, base: float, spec: ContinuationGate) -> tuple[bool, str]:
    recalls = [float(seeds[k]) for k in sorted(seeds, key=str)]
    if len(recalls) < spec.min_seeds:
        return False, f"seeds: {len(recalls)} given, needs {spec.min_seeds}: not met"
    median = statistics.median(recalls)
    ok = _meets_ratio(median, base, spec.relative_recall) and all(r > base for r in recalls)
    return ok, f"seed median recall {median:.4f}, min {min(recalls):.4f} vs control {base:.4f}: {_verdict(ok)}"


def continuation_gate(*, candidate: dict, controls: dict[str, dict], cost_ms: float, seeds: dict, noflag: dict,
                      nominal: dict, spec: ContinuationGate = ContinuationGate()) -> GateOutcome:
    """D10 over the best native control: recall, calib precision and reads, bf16 cost, seeds, no flags, nominal timing.

    candidate and each control hold recall_cap2, precision_calib and reads_calib; seeds maps seed -> recall_cap2;
    noflag holds candidate/control recall_cap2 without flagged sessions; nominal holds their on-time useful rows.
    """
    best = _best_control(controls)
    ctl = controls[best]
    arms = {
        "recall": _meets_ratio(candidate["recall_cap2"], ctl["recall_cap2"], spec.relative_recall),
        "precision": candidate["precision_calib"] >= ctl["precision_calib"] - _EPS,
        "reads": candidate["reads_calib"] <= (1.0 + spec.extra_reads) * ctl["reads_calib"] + _EPS,
        "cost": cost_ms <= spec.cost_ms + _EPS,
        "noflag": _meets_ratio(noflag["candidate"], noflag["control"], spec.relative_recall),
        "nominal": nominal["candidate"] > nominal["control"],
    }
    arms["seeds"], seed_reason = _seed_checks(seeds, base=ctl["recall_cap2"], spec=spec)
    reasons = (
        f"recall at cap {spec.cap} {candidate['recall_cap2']:.4f} vs {best} {ctl['recall_cap2']:.4f} "
        f"(needs +{spec.relative_recall:.0%} relative): {_verdict(arms['recall'])}",
        f"calib precision {candidate['precision_calib']:.4f} vs {ctl['precision_calib']:.4f} (needs no lower): "
        f"{_verdict(arms['precision'])}",
        f"calib reads {candidate['reads_calib']} vs {ctl['reads_calib']} (needs <= +{spec.extra_reads:.0%}): "
        f"{_verdict(arms['reads'])}",
        f"bf16 scorer median {cost_ms:.4f} ms per layer (needs <= {spec.cost_ms}): {_verdict(arms['cost'])}",
        seed_reason,
        f"without flagged sessions {noflag['candidate']:.4f} vs {noflag['control']:.4f}: {_verdict(arms['noflag'])}",
        f"on-time useful rows ({spec.timing_case}) {nominal['candidate']} vs {nominal['control']} (needs above): "
        f"{_verdict(arms['nominal'])}",
    )
    values = {"best_control": best, "candidate": dict(candidate), "control": dict(ctl), "cost_ms": cost_ms,
              "seeds": {str(k): v for k, v in seeds.items()}, "noflag": dict(noflag), "nominal": dict(nominal),
              "arms": arms}
    return GateOutcome(passed=all(arms.values()), reasons=reasons, values=values)
