"""Metadata-only readiness checks and cluster-aware analysis; no model launch."""

import hashlib
import json
import math
import struct
from pathlib import Path
from statistics import NormalDist


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    config_path = root / "config.json"
    c = json.loads(config_path.read_text())
    text = c.get("text_config", c)
    index = root / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    selected = {
        k: v
        for k, v in weight_map.items()
        if k in {"embed.weight", "head.weight"}
        or k.startswith("head.")
        or "markov_head" in k
        or k.endswith(("main_norm.weight", "mtp.2.norm.weight"))
    }
    headers = {}
    for shard in set(selected.values()):
        with (root / shard).open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if size > 64 * 1024 * 1024:
                raise ValueError("unreasonable safetensors header")
            headers[shard] = json.loads(f.read(size))
    return {
        "path": str(root.resolve()),
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index),
        "weights_fully_hashed": False,
        "dimensions": {
            k: text.get(k)
            for k in (
                "hidden_size",
                "vocab_size",
                "rms_norm_eps",
                "dspark_block_size",
                "dspark_markov_rank",
                "dspark_target_layer_ids",
            )
        },
        "quantization": c.get("quantization_config"),
        "tensors": {
            k: {"shard": v, **{a: headers[v][k][a] for a in ("shape", "dtype")}}
            for k, v in selected.items()
        },
    }


def validate_groups(rows):
    seen, content = {}, {}
    allowed = {"train", "select", "screen", "calibrate", "threshold", "test"}
    for row in rows:
        for key in ("id", "group", "split", "domain", "length_bin", "source", "text"):
            if not row.get(key):
                raise ValueError(f"missing {key}")
        if row["split"] not in allowed:
            raise ValueError("unknown split")
        if row["id"] in seen:
            raise ValueError("duplicate prompt id")
        seen[row["id"]] = row
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if digest in content:
            raise ValueError("duplicate prompt text")
        content[digest] = row["split"]
    groups = {}
    for row in rows:
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("prompt-family split leakage")
        groups[row["group"]] = row["split"]
    return {
        split: len({x["group"] for x in rows if x["split"] == split})
        for split in sorted(allowed)
    }


def screen_budget(paired_sd, useful_gain, minimum=64, maximum=1024):
    if (
        not math.isfinite(paired_sd)
        or paired_sd < 0
        or not math.isfinite(useful_gain)
        or useful_gain <= 0
    ):
        raise ValueError("invalid variance/effect")
    n = math.ceil(
        (
            (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8))
            * paired_sd
            / useful_gain
        )
        ** 2
    )
    rounded = max(minimum, 8 * math.ceil(n / 8))
    return {
        "required_prompt_groups": rounded,
        "cap": maximum,
        "within_cap": rounded <= maximum,
        "interval": "two-sided central 95%",
        "power_assumption": 0.8,
        "status": "planning approximation; not measured power; no automatic experiment launch",
    }


def useful_gain(baseline_reward, overhead_fraction, latency_reduction):
    if baseline_reward <= 0 or overhead_fraction < 0 or not 0 <= latency_reduction < 1:
        raise ValueError("invalid cost assumptions")
    return baseline_reward * ((1 + overhead_fraction) / (1 - latency_reduction) - 1)


def paired_report(rows, draws=5000, seed=0):
    """Rows are prompt-level means. Crossed seed/prompt resampling; equal domains, equal prompts within domain.

    Same prompt IDs must occur for every seed and both C/F. No endpoint selection here.
    """
    import numpy as np

    by = {}
    for row in rows:
        key = (str(row["seed"]), str(row["prompt_id"]), row["arm"])
        if (
            key in by
            or row["arm"] not in {"C", "F"}
            or not math.isfinite(row["accepted"])
        ):
            raise ValueError("invalid/duplicate prompt-level row")
        by[key] = row
    seeds = sorted({k[0] for k in by})
    ids = sorted({k[1] for k in by})
    if len(ids) < 2 or not seeds:
        raise ValueError("at least two prompt groups required")
    strata = {}
    domains = {}
    values = np.zeros((len(seeds), len(ids)))
    for i, s in enumerate(seeds):
        for j, p in enumerate(ids):
            if (s, p, "C") not in by or (s, p, "F") not in by:
                raise ValueError("unpaired/crossed-incomplete result grid")
            c, f = by[s, p, "C"], by[s, p, "F"]
            if (
                c["stratum"] != f["stratum"]
                or p in strata
                and strata[p] != c["stratum"]
            ):
                raise ValueError("stratum mismatch")
            strata[p] = c["stratum"]
            domain = c.get("domain", c["stratum"].rsplit("-", 1)[0])
            if f.get("domain", f["stratum"].rsplit("-", 1)[0]) != domain or (
                p in domains and domains[p] != domain
            ):
                raise ValueError("domain mismatch")
            domains[p] = domain
            values[i, j] = f["accepted"] - c["accepted"]
    cells = [
        np.array([j for j, p in enumerate(ids) if strata[p] == s])
        for s in sorted(set(strata.values()))
    ]
    unique_domains = set(domains.values())
    cell_weights = []
    for cell in cells:
        domain = domains[ids[cell[0]]]
        if any(domains[ids[j]] != domain for j in cell):
            raise ValueError("stratum crosses domains")
        total = sum(d == domain for d in domains.values())
        cell_weights.append(len(cell) / total / len(unique_domains))

    def average(means):
        return np.dot(cell_weights, means)

    rng = np.random.default_rng(seed)
    boot = []
    conditional = []
    for _ in range(draws):
        selected = [rng.choice(cell, len(cell), replace=True) for cell in cells]
        seed_ids = rng.integers(0, len(seeds), len(seeds))
        boot.append(
            average([values[np.ix_(seed_ids, cell)].mean() for cell in selected])
        )
        conditional.append(average([values[:, cell].mean() for cell in selected]))
    return {
        "prompt_groups": len(ids),
        "seeds": seeds,
        "mean_gain": float(average([values[:, cell].mean() for cell in cells])),
        "per_seed": [
            float(average([values[i, cell].mean() for cell in cells]))
            for i in range(len(seeds))
        ],
        "prompt_sd": float(values.mean(0).std(ddof=1)),
        "conditional_95": np.quantile(conditional, [0.025, 0.975]).tolist(),
        "crossed_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "disposition": "statistically preliminary; no architecture rejection",
        "warning": "Three or fewer seeds weakly estimate retraining uncertainty."
        if len(seeds) <= 3
        else None,
    }


def preflight(config, corpus=None, bundle=None):
    blockers = []
    if config.get("proposal_sampling") is None:
        blockers.append("proposal sampling transform unresolved")
    if config["checkpoint_rule"] != "fixed_final" or config["producer_arms"] != [
        "C",
        "F",
    ]:
        blockers.append("protocol invariant mismatch")
    if corpus is None or not Path(corpus).exists():
        blockers.append("real grouped prompt corpus absent")
    else:
        rows = [
            json.loads(x) for x in Path(corpus).read_text().splitlines() if x.strip()
        ]
        try:
            counts = validate_groups(rows)
            for split, required in config["split_counts"].items():
                if counts[split] < required:
                    blockers.append(f"{split}: {counts[split]} < {required} groups")
        except ValueError as e:
            blockers.append(str(e))
    if bundle is None or not Path(bundle).exists():
        blockers.append("native captured feature/weight bundle absent")
    else:
        manifest = json.loads((Path(bundle) / "manifest.json").read_text())
        for key in (
            "feature_alignment_validated",
            "native_logits_parity_validated",
            "input_gradient_validated",
            "checkpoint_hashes_complete",
        ):
            if manifest.get(key) is not True:
                blockers.append(key)
        if manifest.get("synthetic", False):
            blockers.append("synthetic fixtures cannot authorize native experiment")
        if not manifest.get("records"):
            blockers.append("capture contains no records")
        root = Path(bundle).resolve()
        for record in manifest.get("records", []):
            path = (root / record["file"]).resolve()
            if (
                path.parent != root
                or not path.is_file()
                or sha256(path) != record.get("sha256")
            ):
                blockers.append("capture record missing or hash mismatch")
                break
        weights = root / "weights.safetensors"
        if not weights.is_file() or sha256(weights) != manifest.get("weights_sha256"):
            blockers.append("native frozen weights missing or hash mismatch")
        for name, expected in config.get("native_config_sha256", {}).items():
            if (
                manifest.get("identity", {}).get("config_sha256", {}).get(name)
                != expected
            ):
                blockers.append(f"{name} configuration identity mismatch")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "experiment_started": False,
        "scope": "artifact checks only; GPU locks/resources must be checked at a future authorized launch",
    }
