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
    p.add_argument(
        "--max-projected-s",
        type=float,
        default=0.0,
        help="abort after warmup if the projected measured window exceeds this many seconds (0 = never)",
    )
    p.add_argument(
        "--expected-effect-ms",
        type=float,
        default=0.68,
        help="per-step lease cost being looked for (OPEN 11 re-take), for the after-warmup resolvability line",
    )
    p.add_argument(
        "--position",
        type=int,
        default=0,
        help="0 if this arm's process is the first of its pair, 1 if the second (order effect)",
    )
    w = p.add_argument_group("workload (recorded in the result)")
    w.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requests in flight; >1 needs --allow-eager-batches",
    )
    w.add_argument("--warmup-requests", type=int, default=1)
    w.add_argument("--warmup-output-tokens", type=int, default=128)
    w.add_argument(
        "--requests",
        type=int,
        default=4,
        help="measured requests, each a different prompt",
    )
    w.add_argument("--input-tokens", type=int, default=256)
    w.add_argument(
        "--output-tokens",
        type=int,
        default=512,
        help="per measured request; 4 x (512 - 1 - discard) = 1924 per-token samples",
    )
    w.add_argument(
        "--discard-steps",
        type=int,
        default=30,
        help="first decode steps of every measured request left out of the statistics",
    )
    w.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="consecutive slices of the samples, for the within-arm spread",
    )
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
        "warmup_output_tokens": args.warmup_output_tokens,
        "discard_steps": args.discard_steps,
        "blocks": args.blocks,
        "measured_requests": args.requests,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "sessions": args.sessions,
        "skip": args.skip,
        "loop": "closed",
        "temperature": 0,
        "ignore_eos": True,
    }


MIN_SAMPLES_PER_BLOCK = 20


def check_workload(args: argparse.Namespace) -> None:
    if args.input_tokens + args.output_tokens > args.context_length:
        raise ValueError(
            f"{args.input_tokens} + {args.output_tokens} tokens exceed --context-length {args.context_length}"
        )
    samples = args.requests * (args.output_tokens - 1 - args.discard_steps)
    if samples < args.blocks * MIN_SAMPLES_PER_BLOCK:
        raise ValueError(
            f"{samples} per-token samples after discarding {args.discard_steps} steps per request cannot fill "
            f"{args.blocks} blocks of {MIN_SAMPLES_PER_BLOCK}"
        )


def resolve_arms(args: argparse.Namespace) -> dict:
    check_workload(args)
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


def _resolvability_line(proj: dict) -> str:
    if "warmup_step_sd_s" not in proj:
        return "[resolvability] too few warmup steps after the discard to estimate the per-token sd"
    verdict = (
        "the window should resolve it"
        if proj["resolvable"]
        else "UNLIKELY TO RESOLVE the expected effect at this sd: consider aborting (the result would be an upper bound)"
    )
    return (
        "[resolvability] warmup per-token p50 {p:.1f} ms, sd {sd:.1f} ms over {n} steps; the planned window resolves "
        "~{r:.2f} ms ({rr:.2f}%) vs the {e:.2f} ms ({er:.2f}%) looked for: {v}. Warmup is the coldest request, so this "
        "leans pessimistic.".format(
            p=proj["warmup_step_p50_s"] * 1e3,
            sd=proj["warmup_step_sd_s"] * 1e3,
            n=proj["warmup_step_samples"],
            r=proj["resolvable_delta_s"] * 1e3,
            rr=proj["resolvable_delta_rel"] * 100,
            e=proj["expected_effect_s"] * 1e3,
            er=proj["expected_effect_rel"] * 100,
            v=verdict,
        )
    )


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

    from counters import TraceCursor, mix, verify_lease, window
    from metrics import aggregate, project
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
    warm_params = sampling_params(output_tokens=args.warmup_output_tokens)
    prov = prov_mod.capture({"bench_arm": os.path.abspath(__file__)})
    prov["drive_idle_check"] = prov_mod.drive_idle_check()

    result = {
        "schema": 1,
        "arm": arm,
        "rep": args.rep,
        "position": args.position,
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
        "trace_overhead": {"measured": False, "trace_enabled": trace_path is not None},
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
        run = lambda batch, sp: engine.loop.run_until_complete(  # noqa: E731
            run_closed_loop(
                engine.async_generate,
                prompts=batch,
                params=sp,
                concurrency=args.concurrency,
            )
        )
        warm_started = time.monotonic()
        warm_records, _ = run(warm, warm_params) if warm else ([], 0.0)
        warm_wall_s = time.monotonic() - warm_started
        result["warmup"] = [
            {k: v for k, v in r.items() if k != "step_s"} for r in warm_records
        ]
        if warm_records:
            result["projection"] = project(
                warm=warm_records[-1],
                requests=args.requests,
                output_tokens=args.output_tokens,
                startup_s=result["startup_s"],
                warm_wall_s=warm_wall_s,
                discard_steps=args.discard_steps,
                expected_effect_s=args.expected_effect_ms / 1e3,
            )
            print(
                "[projection] warmup TTFT {warmup_ttft_s:.1f} s, decode {warmup_decode_tok_s:.2f} tok/s "
                "({warmup_ms_per_token:.0f} ms/token): measured window ~{measured_window_s:.0f} s, "
                "this process ~{process_total_s:.0f} s in all".format(
                    **result["projection"]
                ),
                flush=True,
            )
            print(_resolvability_line(result["projection"]), flush=True)
            if (
                args.max_projected_s
                and result["projection"]["measured_window_s"] > args.max_projected_s
            ):
                raise RuntimeError(
                    f"projected measured window {result['projection']['measured_window_s']:.0f} s exceeds "
                    f"--max-projected-s {args.max_projected_s}"
                )
        marks["start"] = cursor.mark() if cursor else None
        records, wall_s = run(measured, params)
        time.sleep(2.0)
        marks["end"] = cursor.mark() if cursor else None
        result["summary"] = aggregate(
            records,
            wall_s=wall_s,
            discard_steps=args.discard_steps,
            blocks=args.blocks,
        )
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
            result["mix"] = mix(result["window"])
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
