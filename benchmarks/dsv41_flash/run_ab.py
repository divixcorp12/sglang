"""Run both arms of the RAM-miss lease benchmark, as separate processes under gpu-run.sh, N repetitions, and compare.

    PYTHONPATH=$PWD/python python benchmarks/dsv41_flash/run_ab.py --reps 3 --out-dir RUNS/2026-09-22
    python benchmarks/dsv41_flash/run_ab.py --dry-run

Arms alternate off/on then on/off across repetitions, so a drift in host state (page cache, drive wear) does not
favour one arm. Each process holds cc-gpu.lock via gpu-run.sh for its whole life. --dry-run prints the commands
and runs bench_arm.py --dry-run under taskset -c 0-63: no lock, no GPU, no Engine.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_config as ac  # noqa: E402
import compare  # noqa: E402

VENV_PYTHON = "/data/models/slang/.venv/bin/python"
BENCH_ARM = str(Path(__file__).resolve().parent / "bench_arm.py")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__
        + "\nAny other flag (workload, resources) is passed through to bench_arm.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--out-dir",
        default=None,
        help="default: dsv41_flash_runs/<UTC stamp> under the current directory",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=VENV_PYTHON)
    p.add_argument("--gpu-run", default=ac.DEFAULT_GPU_RUN)
    p.add_argument(
        "--first-arm",
        choices=ac.ARMS,
        default="lease_off",
        help="arm that goes first in rep 0",
    )
    return p.parse_known_args(argv)


def child_env() -> dict:
    """PYTHONPATH names the tree under test first, then the pyarrow stand-in run-step.sh also appends."""
    tree = str(ac.REPO_ROOT / "python")
    parts = [tree] + ([ac.PYARROW_SHIM] if os.path.isdir(ac.PYARROW_SHIM) else [])
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(parts),
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def arm_order(reps: int, first: str) -> list[tuple[int, str]]:
    second = next(a for a in ac.ARMS if a != first)
    return [
        (rep, arm)
        for rep in range(reps)
        for arm in ((first, second) if rep % 2 == 0 else (second, first))
    ]


def arm_command(
    args, *, arm: str, rep: int, out_dir: Path, passthrough: list[str]
) -> list[str]:
    out = out_dir / f"{arm}_rep{rep}.json.gz"
    return [
        args.gpu_run,
        args.python,
        BENCH_ARM,
        "--arm",
        arm,
        "--rep",
        str(rep),
        "--out",
        str(out),
        "--gpu-run",
        args.gpu_run,
        *passthrough,
    ]


def main(argv=None) -> int:
    args, passthrough = parse_args(argv)
    env = child_env()
    if args.dry_run:
        plan = arm_order(args.reps, args.first_arm)
        out_dir = Path(args.out_dir or "dsv41_flash_runs/<stamp>")
        for rep, arm in plan:
            print(
                shlex.join(
                    arm_command(
                        args, arm=arm, rep=rep, out_dir=out_dir, passthrough=passthrough
                    )
                )
            )
        cmd = [
            "taskset",
            "-c",
            "0-63",
            args.python,
            BENCH_ARM,
            "--arm",
            "both",
            "--dry-run",
            *passthrough,
        ]
        print("\n$ " + shlex.join(cmd), flush=True)
        return subprocess.run(cmd, env=env).returncode
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"dsv41_flash_runs/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = []
    for rep, arm in arm_order(args.reps, args.first_arm):
        cmd = arm_command(
            args, arm=arm, rep=rep, out_dir=out_dir, passthrough=passthrough
        )
        print(
            f"\n[{datetime.datetime.now().isoformat(timespec='seconds')}] {shlex.join(cmd)}",
            flush=True,
        )
        rc = subprocess.run(cmd, env=env).returncode
        codes.append((arm, rep, rc))
        if rc == 75:
            print("cc-gpu.lock stayed held for 30 minutes; stopping", file=sys.stderr)
            break
    runs = compare.load_runs([str(out_dir)])
    table = compare.render(runs) if runs else "no results were written"
    (out_dir / "comparison.md").write_text(table + "\n")
    print("\n" + table)
    print("\nexit codes: " + ", ".join(f"{a} rep{r}={c}" for a, r, c in codes))
    return 0 if all(c == 0 for _, _, c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
