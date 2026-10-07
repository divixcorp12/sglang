"""Serial, counters-off native A/B on reconstructed chunk multiplicities.

The binaries must come from the same commit, with only EXL3_ROW_WEIGHTED_ASSIGNMENT
different. This measures kernel latency, not end-to-end decode or request polling.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


CASES = [
    "6:2:counts2-1-1-1-1-1-1-1-1-1",
    "6:2:counts1-1-1-1-1-2-1-1-1-1",
    "6:2:counts1-1-1-1-1-1-1-1-1-2",
    "6:2:shared1",
    "6:2:shared2",
    "1:3:shared1",
    "5:3:random",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin"))
    parser.add_argument("--reference-dir", type=Path, default=Path("/data/models/exl3_exp/threading"))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args()
    if args.rounds < 2 or args.iterations < 16:
        parser.error("At least two rounds and sixteen iterations are required")
    args.output.mkdir(parents=True, exist_ok=False)
    references = args.output / "routed-references"
    env = dict(os.environ, OMP_NUM_THREADS="10", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", EXL3_MOE_CPU_MAX_ISA="bw")
    for key in list(env):
        if "TRACE" in key and (key.startswith("EXL3_") or key.startswith("SGLANG_")):
            del env[key]
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "binaries": {arm: {"path": str(path.resolve()), "sha256": sha256(path)} for arm, path in [("a", args.a), ("b", args.b)]},
        "fixture_sha256": sha256(args.fixture),
        "fixture": str(args.fixture),
        "cases": CASES,
        "rounds": args.rounds,
        "iterations": args.iterations,
        "routed_layers": 8,
        "routed_slots": 12,
        "memory_policy": "numactl --membind per node; all workers pinned to the same node",
        "limitations": "Reconstructed chunk multiplicities with synthetic activations and repeated fixture experts; no live serving, GPU transfers, or concurrent CPU groups.",
        "runs": [],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results = []
    for node, cpus in [(0, "6-15"), (1, "18-27")]:
        for round_id in range(args.rounds):
            # AB BA AB BA controls slow order drift; each process has fresh OpenMP state.
            order = [("a", args.a), ("b", args.b)] if round_id % 2 == 0 else [("b", args.b), ("a", args.a)]
            for arm, binary in order:
                label = f"node{node}-round{round_id}-{arm}"
                target = args.output / f"{label}.json"
                command = ["taskset", "-c", "0-63", "numactl", f"--membind={node}", str(binary.resolve()),
                           f"--fixture={args.fixture}", f"--reference-dir={args.reference_dir}",
                           f"--cpus={cpus}", "--workers=10", f"--numa-node={node}",
                           "--warmup-forwards=64", "--routed-layers=8", "--routed-slots=12",
                           "--routed=" + ",".join(CASES), f"--routed-reference-dir={references}",
                           "--benchmark_filter=optimized/rows:", f"--benchmark_min_time={args.iterations}x",
                           f"--benchmark_out={target}", "--benchmark_out_format=json"]
                if not references.exists():
                    if arm != "a":
                        raise RuntimeError("A must create the cross-arm references first")
                    command.append("--write-routed-references")
                started = time.time()
                print(f"Starting {label}", flush=True)
                with open(args.output / f"{label}.log", "w") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(target.read_text())
                if report["context"]["row_weighted_assignment"] != ("0" if arm == "a" else "1"):
                    raise RuntimeError("Wrong compile-time assignment in binary")
                if len(report["benchmarks"]) != len(CASES) or any(x.get("error_occurred") for x in report["benchmarks"]):
                    raise RuntimeError("Incomplete or failed benchmark")
                for row in report["benchmarks"]:
                    results.append(dict(node=node, round=round_id, arm=arm, **row))
                metadata["runs"].append(dict(label=label, command=command, seconds=time.time() - started))
                (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    summary = []
    for node in [0, 1]:
        for name in sorted({row["name"] for row in results}):
            record = dict(node=node, name=name)
            for metric in ["real_time", "p50_us", "p95_us", "p99_us"]:
                for arm in ["a", "b"]:
                    values = [row[metric] for row in results if row["node"] == node and row["name"] == name and row["arm"] == arm]
                    record[f"{arm}_{metric}"] = statistics.median(values)
                    record[f"{arm}_{metric}_range"] = [min(values), max(values)]
                record[f"{metric}_change_pct"] = 100 * (record[f"b_{metric}"] / record[f"a_{metric}"] - 1)
            summary.append(record)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
