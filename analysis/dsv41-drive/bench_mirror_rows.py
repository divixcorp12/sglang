"""Per-row EXL3 read latency: one drive vs the same row split across two mirrors.

Sibling of ``bench_drive_rows.py`` (same reader, same per-row timing, same
sampling), extended to four arms read in rotating order within every rep:

  1. baseline  -- the source checkpoint, ``Exl3ShardRowSource`` (single drive)
  2. one arm per mirror root -- ``Exl3MirrorRowSource`` with that root alone
  3. mirrored  -- ``Exl3MirrorRowSource`` over every root, ``StaticSplitPolicy``

Method, and why (see REPORT.md next to this file):

* O_DIRECT everywhere (``direct=True``), so no read can be served from the page
  cache. The per-rep medians are printed so contamination would still show up
  as later reps getting faster.
* Every ``.read()`` of one expert row is timed on its own with
  ``time.perf_counter_ns()``. The timed region is exactly the ``source.read``
  call, CPU split copies included, as in ``bench_drive_rows.py``; the reader's
  own read/split breakdown is reported next to it.
* 3 reps x 48 experts drawn by ``random.Random(1234).sample`` (no repeats
  within a rep). All arms read the same experts in a rep. The arm order
  rotates by one every rep so no arm is always the last to run.
* The first read of each (arm, shard file) is reported separately and left out
  of the statistics: the earlier run found a one-time 35-45 ms file-open
  outlier there.
* The destination buffers are zeroed before every arm and their CRCs are
  compared across arms afterwards (outside the timed region). A "fast" arm that
  read nothing, or the wrong bytes, would otherwise look like a win.

CPU only: no CUDA is touched. Run under ``taskset -c 0-63`` with
``OMP_NUM_THREADS=16 MKL_NUM_THREADS=16``. Point it only at drives that are
free of other I/O: any concurrent read or write is part of the measurement.

    python bench_mirror_rows.py                       # the four arms of Task 6 Step 4
    python bench_mirror_rows.py --weights 1:0 \\
        --arms single mirrored                        # Step 5: one root weighted to 0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import statistics
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# The worktree this file lives in (analysis/dsv41-drive/ -> repo root/python),
# unless sglang is already importable.
_PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
if (_PYTHON_DIR / "sglang").is_dir() and str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import torch  # noqa: E402

from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat  # noqa: E402
from sglang.srt.layers.moe.exl3_expert_layout import (  # noqa: E402
    Exl3ExpertLayout,
    build_exl3_expert_layout,
)
from sglang.srt.layers.moe.exl3_mirror_row_source import (  # noqa: E402
    Exl3MirrorRowSource,
)
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy  # noqa: E402
from sglang.srt.layers.moe.exl3_shard_row_source import (  # noqa: E402
    Exl3ShardRowSource,
)
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES  # noqa: E402

DEFAULT_SOURCE = "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
DEFAULT_ROOTS = ("/mnt/nvme0/dsv41_flash", "/mnt/nvme4/dsv41_flash")
DEFAULT_LAYER = 19  # all 384 experts in one shard in the earlier run; re-checked here
DEFAULT_SEED = 1234
GATE_P50_MS = 4.0  # Task 6 Step 4: mirrored p50 must not exceed this
# A later-rep median this much below rep 0's is flagged as possible cache warming.
CACHE_SUSPECT_RATIO = 0.85

ARM_KINDS = ("baseline", "single", "mirrored")


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "baseline" | "single" | "mirrored"
    roots: tuple[str, ...] = ()  # mirror roots; empty for the baseline
    weights: tuple[float, ...] = ()


@dataclass
class Sample:
    arm: str
    layer: int
    rep: int
    index: int
    expert: int
    ns: int
    file_bytes: int
    read_ns: int
    split_ns: int
    shard: str
    first: bool  # first read of this arm from this shard file: reported, not counted


@dataclass
class RunResult:
    samples: list[Sample] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- statistics


def pct(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, as in ``bench_drive_rows.py``."""
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize_arm(samples: Sequence[Sample], reps: int) -> dict:
    """Statistics over the counted reads of one arm; first reads listed apart."""
    counted = [s for s in samples if not s.first]
    firsts = [s for s in samples if s.first]
    ms = [s.ns / 1e6 for s in counted]
    out: dict = {
        "n": len(counted),
        "first_reads": [
            {"layer": s.layer, "shard": s.shard, "ms": s.ns / 1e6} for s in firsts
        ],
    }
    if not counted:
        return out
    total_bytes = sum(s.file_bytes for s in counted)
    out.update(
        p50_ms=pct(ms, 50),
        p90_ms=pct(ms, 90),
        p99_ms=pct(ms, 99),
        mean_ms=statistics.mean(ms),
        min_ms=min(ms),
        max_ms=max(ms),
        bytes_per_row=total_bytes / len(counted),
        mb_per_s=(total_bytes / 1e6) / (sum(s.ns for s in counted) / 1e9),
        read_ms_mean=statistics.mean(s.read_ns for s in counted) / 1e6,
        split_ms_mean=statistics.mean(s.split_ns for s in counted) / 1e6,
        rep_p50_ms=[
            statistics.median(s.ns / 1e6 for s in counted if s.rep == rep)
            if any(s.rep == rep for s in counted)
            else float("nan")
            for rep in range(reps)
        ],
    )
    return out


def cache_suspects(results: dict, reps: int) -> list[str]:
    """Arms whose last-rep median is far below rep 0's: what page-cache
    contamination (or a warming drive) would look like."""
    flagged = []
    for name, arm in results.items():
        medians = arm.get("rep_p50_ms") or []
        if reps >= 2 and len(medians) == reps and medians[0] > 0:
            if medians[-1] < CACHE_SUSPECT_RATIO * medians[0]:
                flagged.append(name)
    return flagged


# ------------------------------------------------------------------ set-up


def parse_weights(text: str, roots: int) -> tuple[float, ...]:
    parts = [p for p in re.split(r"[:,]", text) if p.strip()]
    try:
        weights = tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--weights {text!r}: not a list of numbers") from None
    if len(weights) != roots:
        raise SystemExit(f"--weights has {len(weights)} entries for {roots} roots")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise SystemExit("--weights must be non-negative and not all zero")
    return weights


def root_label(path: str, taken: set[str]) -> str:
    """``/mnt/nvme0/dsv41_flash`` -> ``nvme0``; otherwise the parent dir's name."""
    match = re.match(r"/mnt/([^/]+)/", path)
    label = match.group(1) if match else os.path.basename(os.path.dirname(path))
    if not label or label in taken:
        label = path
    taken.add(label)
    return label


def build_arms(
    source: str,
    roots: Sequence[str],
    weights: tuple[float, ...],
    kinds: Sequence[str],
) -> list[Arm]:
    taken: set[str] = set()
    labels = [root_label(r, taken) for r in roots]
    arms: list[Arm] = []
    if "baseline" in kinds:
        arms.append(Arm(root_label(source, taken.copy()) + " baseline", "baseline"))
    if "single" in kinds:
        arms += [
            Arm(f"{l} only", "single", (r,), (1.0,)) for l, r in zip(labels, roots)
        ]
    if "mirrored" in kinds:
        ratio = ":".join(f"{w:g}" for w in weights)
        arms.append(Arm(f"mirrored {ratio}", "mirrored", tuple(roots), weights))
    return arms


def layer_shards(layout: Exl3ExpertLayout, layer: int) -> list[str]:
    """Every distinct shard file that holds one of ``layer``'s expert rows."""
    return sorted({layout.records[(layer, e)].path for e in range(layout.num_experts)})


def build_source(arm: Arm, layout, layer: int, source: str):
    fmt = Exl3ExpertFormat(layout, layer, direct=True)
    segments = fmt.segment_map()
    if arm.kind == "baseline":
        return Exl3ShardRowSource.for_layer(layout, layer, segments, direct=True)
    return Exl3MirrorRowSource.for_mirrored_layer(
        layout,
        layer,
        segments,
        direct=True,
        roots=arm.roots,
        policy=StaticSplitPolicy(arm.weights),
        source_root=source,
    )


def make_destinations(source, rows: int) -> dict[str, torch.Tensor]:
    """Page-aligned CPU destinations, one ``[rows, row_bytes]`` tensor per name."""
    dests = {}
    for name in source.names:
        row_bytes = source.row_bytes[name]
        storage = torch.empty(rows * row_bytes + PAGE_BYTES, dtype=torch.uint8)
        start = (-storage.data_ptr()) % PAGE_BYTES
        dests[name] = storage[start : start + rows * row_bytes].view(rows, -1)
    return dests


def row_digests(dests: dict[str, torch.Tensor], rows: int) -> list[int]:
    """One CRC32 per row, chained over the names in a fixed order."""
    digests = []
    for i in range(rows):
        crc = 0
        for name in sorted(dests):
            crc = zlib.crc32(dests[name][i].numpy(), crc)
        digests.append(crc)
    return digests


# ------------------------------------------------------------------ timing


def time_rows(source, dests, experts: Sequence[int]):
    """Read one expert row at a time; return [(ns, RowReadStats)]."""
    out = []
    for i, expert in enumerate(experts):
        rows_t = torch.tensor([expert], dtype=torch.int64)
        dest_slice = {name: dests[name][i : i + 1] for name in dests}
        began = time.perf_counter_ns()
        stats = source.read(rows_t, dest_slice)
        ended = time.perf_counter_ns()
        out.append((ended - began, stats))
    return out


def arm_order(arms: Sequence[Arm], block: int) -> list[Arm]:
    """The arms rotated left by ``block``: each arm takes every position in turn."""
    shift = block % len(arms)
    return list(arms[shift:]) + list(arms[:shift])


def run(
    layout,
    source: str,
    arms: Sequence[Arm],
    layers: Sequence[int],
    reps: int,
    rows: int,
    seed: int,
    verbose: bool = True,
) -> RunResult:
    result = RunResult()
    rng = random.Random(seed)
    pool = list(range(layout.num_experts))
    sources = {
        (a.name, l): build_source(a, layout, l, source) for l in layers for a in arms
    }
    dests = {}
    for layer in layers:
        first = sources[(arms[0].name, layer)]
        dests[layer] = make_destinations(first, rows)
        for arm in arms:
            other = sources[(arm.name, layer)]
            if other.names != first.names or other.row_bytes != first.row_bytes:
                raise RuntimeError(
                    f"{arm.name} lays a layer-{layer} row out differently"
                )
    opened: set[tuple[str, str]] = set()  # (arm, shard) already touched once

    block = 0
    for rep in range(reps):
        for layer in layers:
            experts = rng.sample(pool, rows)
            digests: dict[str, list[int]] = {}
            for arm in arm_order(arms, block):
                for tensor in dests[layer].values():
                    tensor.zero_()  # a read that transfers nothing must not pass
                timings = time_rows(sources[(arm.name, layer)], dests[layer], experts)
                digests[arm.name] = row_digests(dests[layer], rows)
                arm_samples = []
                for i, (expert, (ns, stats)) in enumerate(zip(experts, timings)):
                    shard = os.path.basename(layout.records[(layer, expert)].path)
                    key = (arm.name, shard)
                    arm_samples.append(
                        Sample(
                            arm.name, layer, rep, i, expert, ns, stats.file_bytes,
                            stats.read_ns, stats.split_ns, shard, key not in opened,
                        )
                    )  # fmt: skip
                    opened.add(key)
                result.samples += arm_samples
                if verbose:
                    counted = [s.ns / 1e6 for s in arm_samples if not s.first] or [
                        float("nan")
                    ]
                    print(
                        f"rep {rep} layer {layer} {arm.name:<16} "
                        f"p50={statistics.median(counted):.3f}ms "
                        f"min={min(counted):.3f} max={max(counted):.3f}",
                        flush=True,
                    )
            reference = digests[arms[0].name]
            for arm in arms[1:]:
                for i, (a, b) in enumerate(zip(reference, digests[arm.name])):
                    if a != b:
                        result.mismatches.append(
                            {"arm": arm.name, "reference": arms[0].name, "layer": layer,
                             "rep": rep, "index": i, "expert": experts[i]}
                        )  # fmt: skip
            block += 1
    return result


# ------------------------------------------------------------------ report


def build_report(result: RunResult, arms: Sequence[Arm], meta: dict, reps: int) -> dict:
    by_arm = {a.name: [s for s in result.samples if s.arm == a.name] for a in arms}
    results = {name: summarize_arm(s, reps) for name, s in by_arm.items()}
    report = dict(meta)
    report["results"] = results
    report["cache_suspects"] = cache_suspects(results, reps)
    report["bytes_match"] = not result.mismatches
    report["mismatches"] = result.mismatches[:50]
    mirrored = next((a for a in arms if a.kind == "mirrored"), None)
    ratios = {}
    if mirrored and results[mirrored.name].get("p50_ms"):
        m = results[mirrored.name]["p50_ms"]
        for a in arms:
            if a is not mirrored and results[a.name].get("p50_ms"):
                ratios[f"{a.name} p50 / {mirrored.name} p50"] = (
                    results[a.name]["p50_ms"] / m
                )
    report["p50_ratios"] = ratios
    if mirrored and results[mirrored.name].get("p50_ms") is not None:
        p50 = results[mirrored.name]["p50_ms"]
        report["gate"] = {
            "rule": f"{mirrored.name} p50 <= {meta['gate_p50_ms']} ms",
            "p50_ms": p50,
            "pass": p50 <= meta["gate_p50_ms"],
        }
    report["samples_ns"] = {
        name: [
            {"layer": s.layer, "rep": s.rep, "expert": s.expert, "ns": s.ns, "first": s.first}
            for s in samples
        ]
        for name, samples in by_arm.items()
    }  # fmt: skip
    return report


def format_table(report: dict) -> str:
    res = report["results"]
    lines = []
    header = (
        f"{'arm':<18}{'n':>5}{'p50':>9}{'p90':>9}{'p99':>9}{'mean':>9}"
        f"{'MB/s':>9}  per-rep p50 (ms)"
    )
    lines += [header, "-" * len(header)]
    for name, r in res.items():
        if not r["n"]:
            lines.append(f"{name:<18}{0:>5}  (no counted reads)")
            continue
        reps = " ".join(f"{m:.2f}" for m in r["rep_p50_ms"])
        lines.append(
            f"{name:<18}{r['n']:>5}{r['p50_ms']:>9.3f}{r['p90_ms']:>9.3f}"
            f"{r['p99_ms']:>9.3f}{r['mean_ms']:>9.3f}{r['mb_per_s']:>9.0f}  {reps}"
        )
    lines.append("(ms except MB/s; first read of each arm per shard excluded, below)")
    lines.append("")
    lines.append("first read of each arm from each shard (excluded above):")
    for name, r in res.items():
        firsts = ", ".join(
            f"L{f['layer']} {f['shard']}: {f['ms']:.2f} ms" for f in r["first_reads"]
        )
        lines.append(f"  {name:<18}{firsts}")
    lines.append("")
    lines.append("read / split split of the mean (ms):")
    for name, r in res.items():
        if r["n"]:
            lines.append(
                f"  {name:<18}read {r['read_ms_mean']:.3f}  split {r['split_ms_mean']:.3f}"
            )
    if report["p50_ratios"]:
        lines.append("")
        for label, ratio in report["p50_ratios"].items():
            lines.append(f"  {label} = {ratio:.2f}x")
    lines.append("")
    if report["cache_suspects"]:
        lines.append(
            "WARNING: later reps are much faster than rep 0 for "
            + ", ".join(report["cache_suspects"])
            + " -- possible page-cache or warming effect."
        )
    else:
        lines.append("page-cache check: no arm's last-rep median is far below rep 0's.")
    if report["bytes_match"]:
        lines.append("byte check: every arm returned identical bytes for every row.")
    else:
        lines.append(
            f"BYTE MISMATCH: {len(report['mismatches'])} rows differ between arms "
            f"(first: {report['mismatches'][0]}). Timings are not trustworthy."
        )
    gate = report.get("gate")
    if gate:
        verdict = "PASS" if gate["pass"] else "FAIL"
        lines.append(
            f"GATE {verdict}: {gate['rule']} (measured {gate['p50_ms']:.3f} ms)"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default=DEFAULT_SOURCE, help="source checkpoint dir")
    p.add_argument(
        "--roots", nargs="+", default=list(DEFAULT_ROOTS), help="mirror roots"
    )
    p.add_argument("--layers", nargs="+", type=int, default=[DEFAULT_LAYER])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--rows", type=int, default=48, help="experts sampled per rep")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--weights", default=None,
        help="split weights for the mirrored arm, one per root, e.g. 1:1 (default) or 1:0",
    )  # fmt: skip
    p.add_argument(
        "--arms", nargs="+", choices=ARM_KINDS, default=list(ARM_KINDS),
        help="which arms to run; 'single' is one arm per root",
    )  # fmt: skip
    p.add_argument("--gate-p50-ms", type=float, default=GATE_P50_MS)
    p.add_argument("--allow-multi-shard", action="store_true",
                   help="accept a layer whose rows span several shard files")  # fmt: skip
    p.add_argument(
        "--output", default=None, help="results JSON (default: next to the script)"
    )
    args = p.parse_args(argv)
    if args.reps < 1 or args.rows < 1:
        p.error("--reps and --rows must be at least 1")
    return args


def check_setup(args, layout) -> list[dict]:
    """Validate the layers before any timed read; return each layer's facts."""
    facts = []
    for layer in args.layers:
        if not 0 <= layer < layout.num_layers:
            raise SystemExit(
                f"layer {layer} is outside the checkpoint's {layout.num_layers}"
            )
        shards = layer_shards(layout, layer)
        facts.append(
            {"layer": layer, "experts": layout.num_experts,
             "shards": [os.path.basename(s) for s in shards]}
        )  # fmt: skip
        if len(shards) != 1 and not args.allow_multi_shard:
            raise SystemExit(
                f"layer {layer}'s {layout.num_experts} experts span {len(shards)} shard "
                f"files ({', '.join(os.path.basename(s) for s in shards[:4])}...); the "
                "benchmark expects one so its file-open cost lands in one first read. "
                "Pick another layer or pass --allow-multi-shard."
            )
    if args.rows > layout.num_experts:
        raise SystemExit(f"--rows {args.rows} exceeds the {layout.num_experts} experts")
    return facts


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source = os.path.realpath(args.source)
    roots = [os.path.realpath(r) for r in args.roots]
    if len(set(roots)) != len(roots):
        raise SystemExit(f"--roots resolve to duplicates: {roots}")
    if source in roots:
        raise SystemExit("a mirror root is the source checkpoint itself")
    weights = parse_weights(args.weights or ":".join("1" for _ in roots), len(roots))
    devices = {os.stat(r).st_dev for r in roots}
    if len(devices) != len(roots):
        print(
            "WARNING: two mirror roots are on the same device; a 'mirror' of one drive."
        )

    layout = build_exl3_expert_layout(source)
    layer_facts = check_setup(args, layout)
    arms = build_arms(source, roots, weights, args.arms)
    if not arms:
        raise SystemExit("no arms selected")
    for fact in layer_facts:
        print(
            f"layer {fact['layer']}: {fact['experts']} experts in shards {fact['shards']}"
        )

    result = run(layout, source, arms, args.layers, args.reps, args.rows, args.seed)
    meta = {
        "when": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "argv": list(sys.argv if argv is None else argv),
        "source": source,
        "roots": roots,
        "weights": list(weights),
        "layers": layer_facts,
        "reps": args.reps,
        "rows": args.rows,
        "seed": args.seed,
        "direct": True,
        "arms": [a.name for a in arms],
        "gate_p50_ms": args.gate_p50_ms,
        "same_device_roots": len(devices) != len(roots),
    }
    report = build_report(result, arms, meta, args.reps)
    output = args.output or str(
        Path(__file__).with_name(
            "mirror-rows-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
    )
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(format_table(report))
    print(f"\nresults: {output}")
    return 0 if report["bytes_match"] else 2


if __name__ == "__main__":
    sys.exit(main())
