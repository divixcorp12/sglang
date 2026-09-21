"""One arm of the RAM-miss lease benchmark, in its own process: launch a real Engine, drive a load, write json.gz.

    PYTHONPATH=$PWD/python python benchmarks/dsv41_flash/bench_arm.py --arm lease_on --out arm.json.gz
    PYTHONPATH=$PWD/python python benchmarks/dsv41_flash/bench_arm.py --arm both --dry-run

--dry-run resolves both arms and runs the EXL3 gate on them; it loads nothing and never touches the GPU.
Real runs go through run_ab.py, which holds cc-gpu.lock via gpu-run.sh.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_config as ac  # noqa: E402

EXIT_BAD_LAUNCH = 2
EXIT_RUN_FAILED = 1
EXIT_LEASE_UNVERIFIED = 3


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--arm",
        choices=(*ac.ARMS, "both"),
        required=True,
        help="'both' is for --dry-run only",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved launch of each arm and assert the EXL3 gate accepts it",
    )
    p.add_argument("--out", help="result path, .json.gz (required unless --dry-run)")
    p.add_argument(
        "--rep", type=int, default=0, help="repetition index, recorded in the result"
    )
    w = p.add_argument_group("workload (recorded in the result)")
    w.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requests in flight; >1 needs --allow-eager-batches",
    )
    w.add_argument("--warmup-requests", type=int, default=4)
    w.add_argument("--requests", type=int, default=16, help="measured requests")
    w.add_argument("--input-tokens", type=int, default=256)
    w.add_argument("--output-tokens", type=int, default=128)
    w.add_argument(
        "--skip",
        type=int,
        default=0,
        help="corpus lines to skip before the first warmup prompt",
    )
    w.add_argument("--allow-eager-batches", action="store_true")
    r = p.add_argument_group("resources")
    r.add_argument("--model", default=ac.Paths().model)
    r.add_argument("--sessions", default=ac.DEFAULT_SESSIONS)
    r.add_argument("--engram-table-dir", default=ac.Paths().engram_table_dir)
    r.add_argument("--expert-dir", default=ac.Paths().expert_dir)
    r.add_argument(
        "--mirror-dirs",
        default="",
        help="SGLANG_MOE_EXPERT_MIRROR_DIRS; empty (default) runs without a mirror",
    )
    r.add_argument("--gpu-run", default=ac.DEFAULT_GPU_RUN)
    r.add_argument("--hot-gpu-mb", type=int, default=ac.Resources().hot_gpu_mb)
    r.add_argument("--pinned-host-mb", type=int, default=ac.Resources().pinned_host_mb)
    r.add_argument(
        "--mem-fraction-static", type=float, default=ac.Resources().mem_fraction_static
    )
    r.add_argument("--context-length", type=int, default=ac.Resources().context_length)
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="skip the stream trace: no counters, so the lease path cannot be verified",
    )
    p.add_argument(
        "--keep-trace",
        action="store_true",
        help="keep the stream trace next to --out, gzipped",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> tuple[ac.Paths, ac.Resources]:
    paths = ac.Paths(
        model=args.model,
        engram_table_dir=args.engram_table_dir,
        expert_dir=args.expert_dir,
        sessions=args.sessions,
        gpu_run=args.gpu_run,
        mirror_dirs=args.mirror_dirs,
    )
    res = ac.Resources(
        pinned_host_mb=args.pinned_host_mb,
        hot_gpu_mb=args.hot_gpu_mb,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
    )
    return paths, res


def workload_record(args: argparse.Namespace) -> dict:
    return {
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "measured_requests": args.requests,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "sessions": args.sessions,
        "skip": args.skip,
        "loop": "closed",
        "temperature": 0,
        "ignore_eos": True,
    }


def resolve_arms(args: argparse.Namespace) -> dict:
    paths, res = build_config(args)
    kwargs = ac.engine_kwargs(paths=paths, res=res)
    arms = ac.ARMS if args.arm == "both" else (args.arm,)
    out = {}
    for arm in arms:
        env = ac.arm_env(arm, paths=paths, res=res)
        ac.check_recipe(
            env=env,
            kwargs=kwargs,
            concurrency=args.concurrency,
            allow_eager_batches=args.allow_eager_batches,
        )
        out[arm] = {
            "env": env,
            "engine_kwargs": kwargs,
            "gate": ac.check_gate(env=env, kwargs=kwargs, model=paths.model),
        }
    return {"paths": paths, "resources": res, "arms": out}


def stray_sglang_env(env: dict) -> dict:
    return {
        k: v for k, v in os.environ.items() if k.startswith("SGLANG_") and k not in env
    }


def env_conflicts(env: dict) -> dict:
    """SGLANG_* variables the process already has at another value; those change what runs, so they are refused."""
    return {
        k: {"process": os.environ[k], "arm": v}
        for k, v in env.items()
        if k.startswith("SGLANG_") and k in os.environ and os.environ[k] != v
    }


def infra_overrides(env: dict) -> dict:
    """Non-SGLANG variables (CUDA_HOME, thread caps) the recipe replaces, as env.sh does; the box's shell has CUDA 13.4."""
    return {
        k: {"process": os.environ[k], "arm": v}
        for k, v in env.items()
        if not k.startswith("SGLANG_") and k in os.environ and os.environ[k] != v
    }


def dry_run(args: argparse.Namespace, sglang_file: str) -> int:
    import msgspec

    resolved = resolve_arms(args)
    path_report = ac.check_paths(resolved["paths"])
    problems = [
        f"missing path {label}: {v['path']}"
        for label, v in path_report.items()
        if not v["exists"]
    ]
    arms = resolved["arms"]
    report = {
        "dry_run": True,
        "sglang_file": sglang_file,
        "repo_root": str(ac.REPO_ROOT),
        "workload": workload_record(args),
        "paths": path_report,
        "resources": msgspec.to_builtins(resolved["resources"]),
        "arms": arms,
        "infra_env_overridden": infra_overrides(next(iter(arms.values()))["env"]),
    }
    if len(arms) == 2:
        report["env_differs_between_arms"] = ac.env_diff(
            arms["lease_off"]["env"], arms["lease_on"]["env"]
        )
        if list(report["env_differs_between_arms"]) != [ac.LEASE_ENV]:
            problems.append(
                f"the arms differ in more than {ac.LEASE_ENV}: {sorted(report['env_differs_between_arms'])}"
            )
    for arm, info in arms.items():
        for name, clash in env_conflicts(info["env"]).items():
            problems.append(
                f"{name} is already {clash['process']!r} in this process; {arm} needs {clash['arm']!r}"
            )
        if info["gate"]["unknown_sglang_env"]:
            problems.append(
                f"{arm} sets SGLANG_* names this tree does not define: {info['gate']['unknown_sglang_env']}"
            )
    stray = stray_sglang_env(next(iter(arms.values()))["env"])
    if stray:
        problems.append(
            f"SGLANG_* in the process environment but not in the recipe: {sorted(stray)}"
        )
    import torch

    report["cuda_initialized"] = torch.cuda.is_initialized()
    if report["cuda_initialized"]:
        problems.append("a dry run initialised CUDA")
    report["problems"] = problems
    print(json.dumps(report, indent=2, default=str))
    return 0 if not problems else EXIT_BAD_LAUNCH


def write_result(path: str, result: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(result, f, default=str)


def run_arm(args: argparse.Namespace, sglang_file: str) -> int:
    if not args.out or not args.out.endswith(".json.gz"):
        sys.exit("--out PATH.json.gz is required for a real run")
    resolved = resolve_arms(args)
    paths, res = resolved["paths"], resolved["resources"]
    missing = {
        k: v["path"] for k, v in ac.check_paths(paths).items() if not v["exists"]
    }
    if missing:
        sys.exit(f"refusing to launch, missing paths: {missing}")
    arm = args.arm
    env, kwargs = resolved["arms"][arm]["env"], resolved["arms"][arm]["engine_kwargs"]
    other_arm = next(a for a in ac.ARMS if a != arm)
    other = ac.arm_env(other_arm, paths=paths, res=res)
    if env_conflicts(env) or stray_sglang_env(env):
        sys.exit(
            f"process environment disagrees with the {arm} recipe: {env_conflicts(env)} stray: {sorted(stray_sglang_env(env))}"
        )
    trace_path = (
        None if args.no_trace else args.out[: -len(".json.gz")] + ".trace.jsonl"
    )
    if trace_path:
        Path(trace_path).unlink(missing_ok=True)
        env = {**env, ac.TRACE_ENV: trace_path}
    os.environ.update(env)

    from counters import TraceCursor, verify_lease, window
    from metrics import aggregate
    from transformers import AutoTokenizer
    from workload import run_closed_loop, sampling_params, select_prompts

    import sglang

    prov_mod = ac.import_provenance()
    tokenizer = AutoTokenizer.from_pretrained(paths.model)
    prompts = select_prompts(
        sessions_path=paths.sessions,
        tokenizer=tokenizer,
        skip=args.skip,
        count=args.warmup_requests + args.requests,
        input_tokens=args.input_tokens,
    )
    warm, measured = prompts[: args.warmup_requests], prompts[args.warmup_requests :]
    params = sampling_params(output_tokens=args.output_tokens)
    prov = prov_mod.capture({"bench_arm": os.path.abspath(__file__)})
    prov["drive_idle_check"] = prov_mod.drive_idle_check()

    result = {
        "schema": 1,
        "arm": arm,
        "rep": args.rep,
        "lease_switch": {"name": ac.LEASE_ENV, "value": env[ac.LEASE_ENV]},
        "env_arm": env,
        "env_differs_between_arms": ac.env_diff(
            env, other, a_name=arm, b_name=other_arm
        ),
        "workload": workload_record(args),
        "engine_kwargs": kwargs,
        "gate": resolved["arms"][arm]["gate"],
        "sglang_file": sglang_file,
        "prompts": [
            {"session_line": p["session_line"], "tokens": len(p["input_ids"])}
            for p in prompts
        ],
        "error": None,
    }
    cursor = TraceCursor(trace_path) if trace_path else None
    started = time.monotonic()
    engine = sglang.Engine(**kwargs)
    result["startup_s"] = time.monotonic() - started
    prov["sglang_env_drift_at_engine_ready"] = prov_mod.env_drift(
        prov["sglang_env"], prov_mod.process_env()
    )
    result["provenance"] = prov
    marks = {}
    try:
        run = lambda batch: engine.loop.run_until_complete(  # noqa: E731
            run_closed_loop(
                engine.async_generate,
                prompts=batch,
                params=params,
                concurrency=args.concurrency,
            )
        )
        warm_records, _ = run(warm) if warm else ([], 0.0)
        result["warmup"] = [
            {k: v for k, v in r.items() if k != "step_s"} for r in warm_records
        ]
        marks["start"] = cursor.mark() if cursor else None
        records, wall_s = run(measured)
        time.sleep(2.0)
        marks["end"] = cursor.mark() if cursor else None
        result["summary"] = aggregate(records, wall_s=wall_s)
        result["requests"] = records
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        engine.shutdown()
    if cursor:
        result["lease_check"] = verify_lease(arm=arm, cursor=cursor)
        result["counters_final"] = (
            marks["end"]["counters"] if marks.get("end") else None
        )
        if marks.get("start") and marks.get("end"):
            result["window"] = window(marks["start"], marks["end"])
            delta = result["window"]["counters_delta"]
            result["rows_read_window"] = None if delta is None else delta["rows_read"]
        result["rows_read_total"] = (
            result["counters_final"]["rows_read"] if result["counters_final"] else None
        )
    else:
        result["lease_check"] = {
            "ok": False,
            "reasons": ["--no-trace: counters unavailable, lease path not verified"],
        }
    result["trace_kept"] = bool(args.keep_trace and trace_path)
    write_result(args.out, result)
    if trace_path and Path(trace_path).exists():
        if args.keep_trace:
            with (
                open(trace_path, "rb") as src,
                gzip.open(trace_path + ".gz", "wb") as dst,
            ):
                dst.write(src.read())
        Path(trace_path).unlink()
    print(
        json.dumps(
            {
                "arm": arm,
                "rep": args.rep,
                "error": result["error"],
                "summary": result.get("summary"),
                "lease_check_ok": result["lease_check"]["ok"],
                "reasons": result["lease_check"]["reasons"],
            },
            default=str,
        ),
        flush=True,
    )
    if result["error"]:
        return EXIT_RUN_FAILED
    return 0 if result["lease_check"]["ok"] else EXIT_LEASE_UNVERIFIED


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.arm == "both" and not args.dry_run:
        sys.exit("--arm both is for --dry-run; a real run is one arm per process")
    try:
        sglang_file = ac.assert_sglang_from_repo()
        return (
            dry_run(args, sglang_file) if args.dry_run else run_arm(args, sglang_file)
        )
    except (ac.ImportPathError, ac.RecipeError, ValueError) as error:
        print(f"REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_BAD_LAUNCH


if __name__ == "__main__":
    sys.exit(main())
