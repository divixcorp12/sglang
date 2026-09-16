"""Select provenance-locked BS1 pull gates from separate profiling histograms.

This is deliberately an offline decision tool.  It never starts a server or
changes a runtime flag: it validates two completed Task 2 profiling artifacts,
fits thresholds on the training session set, and retains only gates whose exact
feature/threshold remains net-positive on the held-out session set.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
COST_SCHEMA_VERSION = 1
BIN_COUNT = 256
BIN_EDGES = "uniform [0,1], lower-inclusive; 1.0 is in bin 255"
FEATURES = {"score": "top_score", "margin": "score_margin"}
COUNTERS = (
    "opportunity",
    "source_eligible",
    "target_useful",
    "target_wasted",
    "physical_demand_rows",
)
PROVENANCE_KEYS = ("commit", "predictor", "checkpoint_checksum", "cache_size")
COST_KEYS = (
    "useful_row_value",
    "posted_row_cost",
    "physical_demand_row_cost",
    "scorer_plus_selection_cost",
    "count_zero_control_cost",
    "join_exposure_cost",
)


def _read_json(path: Path) -> dict[str, Any]:
    """Load a completed calibration through the serving validator."""
    try:
        from sglang.srt.layers.moe.expert_prediction.serving.calibration import PullCalibrationHistogram
    except ModuleNotFoundError:
        # This script is also intentionally usable as a standalone offline
        # inspection tool.  Its completion check exactly mirrors the serving
        # validator when the serving package is not importable.
        payload = json.loads(path.read_text())
        if not payload.get("complete", False):
            raise ValueError("incomplete pull calibration artifact is invalid for decisions")
        return payload
    return PullCalibrationHistogram.require_complete(path)


def _as_payload(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        if not payload.get("complete", False):
            raise ValueError("incomplete pull calibration artifact is invalid for decisions")
        return payload
    return _read_json(Path(value))


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _validate_histogram(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported pull calibration schema_version")
    if payload.get("bin_count") != BIN_COUNT:
        raise ValueError("pull calibration must contain exactly 256 bins")
    if payload.get("range") != [0.0, 1.0]:
        raise ValueError("pull calibration score range must be [0, 1]")
    if payload.get("bin_edges") != BIN_EDGES:
        raise ValueError("pull calibration bin_edges must use the canonical uniform [0,1] contract")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("pull calibration is missing provenance")
    for key in PROVENANCE_KEYS:
        if provenance.get(key) in (None, ""):
            raise ValueError(f"pull calibration provenance is missing {key}")
    if isinstance(provenance["cache_size"], bool) or not isinstance(provenance["cache_size"], int) or provenance["cache_size"] < 0:
        raise ValueError("pull calibration provenance cache_size must be a nonnegative integer")
    _validate_scope_metadata(provenance)

    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("pull calibration needs at least one layer")
    for layer, record in layers.items():
        try:
            int(layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid calibration layer {layer!r}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"layer {layer} calibration must be an object")
        target_observations = record.get("target_observations")
        if isinstance(target_observations, bool) or not isinstance(target_observations, int) or target_observations < 0:
            raise ValueError(f"layer {layer} target_observations must be a nonnegative integer")
        totals: dict[str, dict[str, int]] = {}
        for feature in FEATURES:
            counters = record.get(feature)
            if not isinstance(counters, Mapping):
                raise ValueError(f"layer {layer} is missing {feature} histogram")
            totals[feature] = _validate_counters(layer, feature, counters)
        if totals["score"] != totals["margin"]:
            raise ValueError(f"layer {layer} score/margin aggregate totals differ")
        for field in ("source_eligible", "opportunity", "target_useful", "target_wasted"):
            if target_observations < totals["score"][field]:
                raise ValueError(f"layer {layer} target_observations is below {field} total")


def _validate_scope_metadata(provenance: Mapping[str, Any]) -> None:
    """Require a typed, self-consistent BS1/top-k-unique profiling scope."""
    batch_size = provenance.get("batch_size")
    top_k = provenance.get("top_k")
    top_k_unique = provenance.get("top_k_unique")
    shape = provenance.get("shape_provenance")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
        raise ValueError("calibration artifact is outside BS1 scope")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("calibration artifact top_k must be a positive integer")
    if top_k_unique is not True:
        raise ValueError("calibration artifact is outside top-k-unique scope")
    if not isinstance(shape, Mapping):
        raise ValueError("calibration artifact shape_provenance must be an object")
    shape_batch_size = shape.get("batch_size")
    shape_top_k = shape.get("top_k")
    if (
        isinstance(shape_batch_size, bool)
        or not isinstance(shape_batch_size, int)
        or isinstance(shape_top_k, bool)
        or not isinstance(shape_top_k, int)
        or shape_batch_size != batch_size
        or shape_top_k != top_k
        or shape.get("top_k_unique") is not True
    ):
        raise ValueError("calibration artifact shape_provenance contradicts canonical scope metadata")


def _validate_counters(layer: str, feature: str, counters: Mapping[str, Any]) -> dict[str, int]:
    values: dict[str, list[int]] = {}
    for field in COUNTERS:
        raw = counters.get(field)
        if not isinstance(raw, list) or len(raw) != BIN_COUNT:
            raise ValueError(f"layer {layer} {feature} needs 256 {field} bins")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw):
            raise ValueError(f"layer {layer} {feature} {field} bins must be nonnegative integers")
        values[field] = raw
    for index, (posted, useful, wasted, eligible) in enumerate(zip(
        values["opportunity"], values["target_useful"], values["target_wasted"], values["source_eligible"], strict=True
    )):
        if posted != useful + wasted:
            raise ValueError(f"layer {layer} {feature} bin {index} has inconsistent posted outcomes")
        if eligible < posted:
            raise ValueError(f"layer {layer} {feature} bin {index} has fewer source-eligible than posted rows")
    return {field: sum(raw) for field, raw in values.items()}


def _validate_pair(train: Mapping[str, Any], heldout: Mapping[str, Any]) -> None:
    train_provenance = train["provenance"]
    heldout_provenance = heldout["provenance"]
    for key in PROVENANCE_KEYS:
        if train_provenance[key] != heldout_provenance[key]:
            raise ValueError(f"calibration provenance mismatch for {key}")
    for key in ("batch_size", "top_k", "top_k_unique", "shape_provenance"):
        if train_provenance.get(key) != heldout_provenance.get(key):
            raise ValueError(f"calibration provenance mismatch for {key}")
    if train["bin_edges"] != heldout["bin_edges"]:
        raise ValueError("calibration provenance mismatch for bin_edges")

    train_checksum = train_provenance.get("session_set_checksum")
    heldout_checksum = heldout_provenance.get("session_set_checksum")
    if not train_checksum or not heldout_checksum or train_checksum == heldout_checksum:
        raise ValueError("training and held-out calibration session sets must be distinct")
    train_sessions = train_provenance.get("session_ids")
    heldout_sessions = heldout_provenance.get("session_ids")
    _validate_session_ids(train_sessions, "training")
    _validate_session_ids(heldout_sessions, "held-out")
    if set(train_sessions) & set(heldout_sessions):
        raise ValueError("training and held-out calibration sessions overlap")


def _validate_session_ids(session_ids: Any, label: str) -> None:
    if not isinstance(session_ids, list) or not session_ids:
        raise ValueError(f"{label} session_ids must be a nonempty list")
    if any(not isinstance(session_id, str) or not session_id.strip() for session_id in session_ids):
        raise ValueError(f"{label} session_ids must contain nonempty strings")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"{label} session_ids must not contain duplicates")


def _validate_costs(costs: Mapping[str, Any]) -> Mapping[str, Any]:
    if costs.get("schema_version") != COST_SCHEMA_VERSION:
        raise ValueError("unsupported measured-cost schema_version")
    layers = costs.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("measured costs need a layers object")
    for layer, values in layers.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"measured costs for layer {layer} must be an object")
        for key in COST_KEYS:
            if _require_number(values.get(key), f"layer {layer} {key}") < 0:
                raise ValueError(f"layer {layer} {key} must be nonnegative")
    return layers


def _sum_from_threshold(counters: Mapping[str, list[int]], threshold_bin: int) -> dict[str, int]:
    return {field: sum(counters[field][threshold_bin:]) for field in COUNTERS}


def _target_observations(record: Mapping[str, Any]) -> int:
    """Return the graph-counted total scorer/control denominator for one target."""
    return record["target_observations"]


def _net_value(selected: Mapping[str, int], target_observations: int, costs: Mapping[str, Any]) -> float:
    """Price one threshold using physical, rather than residual-route, traffic.

    Scoring and the count-zero control are paid for every graph-counted target
    observation in an enabled layer. Posting/join and demand-copy costs are
    paid only after the selected threshold. This prevents a precision-only gate
    from hiding the fixed cost of keeping a layer enabled.
    """
    useful_value = selected["target_useful"] * _require_number(costs["useful_row_value"], "useful_row_value")
    fixed = target_observations * (
        _require_number(costs["scorer_plus_selection_cost"], "scorer_plus_selection_cost")
        + _require_number(costs["count_zero_control_cost"], "count_zero_control_cost")
    )
    posted = selected["opportunity"] * (
        _require_number(costs["posted_row_cost"], "posted_row_cost")
        + _require_number(costs["join_exposure_cost"], "join_exposure_cost")
    )
    demand = selected["physical_demand_rows"] * _require_number(
        costs["physical_demand_row_cost"], "physical_demand_row_cost"
    )
    net_value = useful_value - fixed - posted - demand
    if not math.isfinite(net_value):
        raise ValueError("calibration net value is not finite")
    return net_value


def _select_layer(train: Mapping[str, Any], heldout: Mapping[str, Any], costs: Mapping[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for histogram_feature, gate_feature in FEATURES.items():
        train_counters = train[histogram_feature]
        heldout_counters = heldout[histogram_feature]
        train_observations = _target_observations(train)
        heldout_observations = _target_observations(heldout)
        for threshold_bin in range(BIN_COUNT):
            training_net = _net_value(_sum_from_threshold(train_counters, threshold_bin), train_observations, costs)
            if training_net <= 0:
                continue
            heldout_net = _net_value(_sum_from_threshold(heldout_counters, threshold_bin), heldout_observations, costs)
            if heldout_net <= 0:
                continue
            candidate = {
                "feature": gate_feature,
                "threshold": threshold_bin / BIN_COUNT,
                "training_net_value": training_net,
                "heldout_net_value": heldout_net,
                "training_physical_rows": _sum_from_threshold(train_counters, threshold_bin)["opportunity"]
                + _sum_from_threshold(train_counters, threshold_bin)["physical_demand_rows"],
                "heldout_physical_rows": _sum_from_threshold(heldout_counters, threshold_bin)["opportunity"]
                + _sum_from_threshold(heldout_counters, threshold_bin)["physical_demand_rows"],
            }
            if best is None or (candidate["training_net_value"], candidate["heldout_net_value"], candidate["threshold"]) > (
                best["training_net_value"], best["heldout_net_value"], best["threshold"]
            ):
                best = candidate
    return best


def calibrate_gate(
    train: Mapping[str, Any] | str | Path,
    heldout: Mapping[str, Any] | str | Path,
    costs: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return a versioned gate artifact, without accepting a same-set fit."""
    train_payload = _as_payload(train)
    heldout_payload = _as_payload(heldout)
    costs_payload = _as_payload_json(costs)
    _validate_histogram(train_payload)
    _validate_histogram(heldout_payload)
    _validate_pair(train_payload, heldout_payload)
    cost_layers = _validate_costs(costs_payload)

    if set(train_payload["layers"]) != set(heldout_payload["layers"]):
        raise ValueError("training and held-out calibration layer sets differ")

    selected: dict[str, Any] = {}
    for layer in sorted(set(train_payload["layers"]) & set(heldout_payload["layers"]), key=int):
        if layer not in cost_layers:
            raise ValueError(f"missing measured costs for calibration layer {layer}")
        result = _select_layer(train_payload["layers"][layer], heldout_payload["layers"][layer], cost_layers[layer])
        if result is not None:
            selected[layer] = result

    if not selected:
        raise ValueError("no positive training threshold remains positive on held-out sessions")

    provenance = train_payload["provenance"]
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "predictor": provenance["predictor"],
        "checkpoint_checksum": provenance["checkpoint_checksum"],
        "source_commit": provenance["commit"],
        "cache_size": provenance["cache_size"],
        "calibration_miss_regime": {
            "batch_size": 1,
            "shape_provenance": provenance.get("shape_provenance"),
            "top_k": provenance.get("top_k"),
            "bin_edges": train_payload["bin_edges"],
            "physical_row_field": "physical_demand_rows",
        },
        "source_histogram_schema_version": train_payload["schema_version"],
        "training_session_set_checksum": provenance["session_set_checksum"],
        "heldout_session_set_checksum": heldout_payload["provenance"]["session_set_checksum"],
        "target_layers": [int(layer) for layer in sorted(selected, key=int)],
        "layers": selected,
    }


def _as_payload_json(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="completed training pull-calibration.json")
    parser.add_argument("--heldout", type=Path, required=True, help="completed held-out pull-calibration.json")
    parser.add_argument("--costs", type=Path, required=True, help="versioned measured scorer/copy costs JSON")
    parser.add_argument("--out", type=Path, required=True, help="gate artifact output path")
    args = parser.parse_args(argv)
    artifact = calibrate_gate(args.train, args.heldout, args.costs)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
