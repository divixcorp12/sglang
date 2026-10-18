"""The DSpark mode of the recipe: both CPU-expert clients on, the graphed verify's configuration, and argv the server
parses as DSpark (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 15)."""

import argparse

import arm_env


def test_dspark_env_turns_both_cpu_expert_clients_on_with_spill():
    env = arm_env.arm_env(arm_env.dspark_env())
    assert env["SGLANG_DSV41_CPU_EXPERTS"] == "1"
    assert env["SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"] == "1"
    # One team per node: the draft runs on node 0's CPU expert team, so no draft cores are named (Task 12 refuses them).
    assert "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES" not in env and "SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS" not in env
    assert "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES" not in env  # a lane per route: 36 on a 40-lane wire
    assert env["SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES"] == "8"
    assert env["SGLANG_RAGGED_VERIFY_MODE"] == "static"
    assert env["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"] == "0"
    # Budget A (mem-budget-report.md L2): the DSpark arms trade hot cache for prefill headroom.
    assert env["SGLANG_MOE_HOT_GPU_MB"] == "10840"
    assert env["SGLANG_DSV41_ENABLE_PREFILL_FILLS"] == "1"  # the recipe's, kept


def test_prod_serves_the_dspark_mode_with_its_own_settings():
    assert arm_env.PROD_DSPARK is True
    assert arm_env.prod_env() == arm_env.arm_env({**arm_env.dspark_env(), **arm_env.prod_dspark_env()})
    argv = arm_env.ServerArgs.prod().argv()
    assert argv[argv.index("--speculative-algorithm") + 1] == "DSPARK"


# MERGE_PLAN_1016.md: production is the best served arm, RAM prefetch included, with both startup calibrations on.
PROD_DSPARK_SETTINGS = {
    "SGLANG_MOE_PINNED_HOST_FULL_LAYERS": "0,1",
    "SGLANG_MOE_PINNED_HOST_MB": "111530",
    "SGLANG_MOE_PINNED_HOST_NUMA_MB": "0:59656,1:51874",
    "SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL": "1",
    "SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS": "",
    "SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES": "0",
    "SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION": "1",
    "SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "1",
    "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3",
    "SGLANG_ENABLE_SCHEDULER_MLOCK": "1",
    "SGLANG_HICACHE_HOST_NUMA_NODE": "1",
    "SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB": "128",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_MOE_HOT_GPU_MB": "13144",
    "SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP": "1",
    "SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS": "0",
    "SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB": "128",
}


def test_prod_env_carries_the_best_arms_settings_and_both_startup_calibrations():
    env = arm_env.prod_env()
    assert {k: env.get(k) for k in PROD_DSPARK_SETTINGS} == PROD_DSPARK_SETTINGS


def test_prod_prefills_layer_major_under_dspark_and_the_arms_do_not():
    # Owner decision 2026-10-17: production turns layer-major prefill on under DSpark (threshold and the DSpark switch,
    # which layer_major/gate.py needs under speculative decoding), plus the lent residency rows. dspark_env() keeps it off for
    # every arm built on it.
    env = arm_env.prod_env()
    assert env["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"] == "8192"
    assert env["SGLANG_ENABLE_LAYER_MAJOR_PREFILL_DSPARK"] == "1"
    # Residency over lent hot-cache slots (owner, 2026-10-18): the rows are production's; the lend is the env default.
    assert env["SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS"] == "360"
    assert "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND" not in env
    from sglang.srt.environ import envs

    assert envs.SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND.get() is True
    # The lend's launch refusals: production sets neither prefetch pull mode nor GPU drip slots.
    assert env.get("SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE", "off") == "off"
    assert env.get("SGLANG_MOE_EXPERT_PREFETCH_PULL", "0") == "0"
    assert int(env.get("SGLANG_DSV41_GPU_DRIP_SLOTS", "0")) == 0
    assert arm_env.dspark_env()["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"] == "0"
    assert "SGLANG_ENABLE_LAYER_MAJOR_PREFILL_DSPARK" not in arm_env.dspark_env()
    # The gate's launch checks production meets: one running request and the DSpark algorithm.
    argv = arm_env.ServerArgs.prod().argv()
    assert argv[argv.index("--max-running-requests") + 1] == "1"
    assert argv[argv.index("--speculative-algorithm") + 1] == "DSPARK"
    assert int(env["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"]) > arm_env.CHUNKED_PREFILL_SIZE  # above the ring


def test_prod_leaves_the_pending_and_held_flags_off():
    env = arm_env.prod_env()
    # The side stream waits on its own A/B; the lane head is held; the rest measured neutral or not served.
    for held in ("SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM", "SGLANG_OPT_LANE_HEAD_PATH", "SGLANG_DSV41_GPU_DRIP_SLOTS",
                 "SGLANG_DSPARK_TOP_PROB_GATE", "SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK",
                 "SGLANG_DSV41_CPU_EXPERTS_STORE_OUT", "SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL",
                 "SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION"):
        assert held not in env, held
    # Measurement only: the instrumented build's traces and the hot cache's metrics file.
    assert not any("TRACE" in k for k in env) and "SGLANG_MOE_HOT_METRICS_FILE" not in env


def test_prod_leaves_the_confidence_cut_and_its_sts_collection_off():
    # docs/hx/compact-verify-exl3.draft.md: off by default; the cut and its STS file belong to both_cpu_ab's cut arm.
    env = arm_env.prod_env()
    assert "SGLANG_DSPARK_CONFIDENCE_CUT" not in env and "SGLANG_DSPARK_STS_COLLECT_PATH" not in env
    assert "--speculative-dspark-confidence-sts-path" not in arm_env.ServerArgs.prod().argv()
    assert not any("DSPARK_STS" in k or "CONFIDENCE_CUT" in k for k in arm_env.dspark_env())
    # The arm's fitted STS file: a production asset path with its digest pinned.
    assert arm_env.PROD_DSPARK_STS_PATH.startswith("/data/models/slang/prod-assets/dspark-sts/")
    assert len(arm_env.PROD_DSPARK_STS_SHA256) == 64


# The best arm's prefetch keys, from divix01:/mnt/nvme1/share64-ab-20261016/resolved-env.json "on" (its head was
# nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors, production's is the checksummed copy).
SERVED_PREFETCH = {
    "SGLANG_DSV41_RAM_PREFETCH": "1",
    "SGLANG_DSV41_RAM_PREFETCH_HORIZON": "2",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US": "4000",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE": "1",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES": "2",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD": "1",
    "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": "",
    "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "2",
    "SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN": "1",
    "SGLANG_DSV41_RAM_PREFETCH_POOL_SERVE": "1",
    "SGLANG_DSV41_RAM_PREFETCH_PROMOTE": "1",
    "SGLANG_DSV41_RAM_PREFETCH_SCORER": "gpu",
    "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": "2",
    "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "0",
    "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
}


def test_prod_serves_the_best_arms_prefetch_with_the_checksummed_head():
    env = arm_env.prod_env()
    prefetch = {k: v for k, v in env.items() if k.startswith("SGLANG_DSV41_RAM_PREFETCH")}
    head = prefetch.pop("SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD")
    assert {**prefetch, "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": env["SGLANG_MOE_PREFETCH_MAX_CANDIDATES"]} == SERVED_PREFETCH
    assert head == arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD
    assert head.startswith("/data/models/slang/prod-assets/") and len(arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD_SHA256) == 64
    # The arms keep their own: dspark_env() states no prefetch.
    assert not any(k.startswith("SGLANG_DSV41_RAM_PREFETCH") for k in arm_env.dspark_env())


def test_check_prod_assets_accepts_the_pinned_digest_and_refuses_any_other(tmp_path):
    import hashlib

    import pytest

    head = tmp_path / "head.safetensors"
    head.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    assert arm_env.check_prod_assets(str(head), sha) == f"prefetch head verified: {head} sha256 {sha}"
    with pytest.raises(SystemExit, match="not the pinned"):
        arm_env.check_prod_assets(str(head), "0" * 64)
    with pytest.raises(SystemExit, match="is missing"):
        arm_env.check_prod_assets(str(tmp_path / "none"), sha)


def test_prod_argv_carries_its_memory_and_the_arms_keep_theirs():
    argv = arm_env.ServerArgs.prod().argv()
    assert argv.count("--mem-fraction-static") == 1
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.8623"
    assert argv[argv.index("--max-total-tokens") + 1] == "381184"
    assert "--weight-loader-drop-cache-after-load" in argv
    # The arms' math (both_cpu_ab.fraction_keeping_kv) reads the OFF arm's measured init, unchanged.
    assert arm_env.DSPARK_HOT_GPU_MB == "10840" and arm_env.dspark_env()["SGLANG_MOE_HOT_GPU_MB"] == "10840"
    arm = arm_env.ServerArgs(port=1, dspark=True).argv()
    assert arm[arm.index("--mem-fraction-static") + 1] == "0.78"
    assert "--max-total-tokens" not in arm and "--weight-loader-drop-cache-after-load" not in arm
    for k in PROD_DSPARK_SETTINGS:
        if k not in arm_env.base_env() and k != "SGLANG_MOE_HOT_GPU_MB":
            assert k not in arm_env.dspark_env(), f"{k} would reach every arm"


def test_the_prod_argv_parses_with_its_memory_flags():
    from sglang.srt.server_args import ServerArgs

    argv = arm_env.ServerArgs.prod().argv()
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv[3:])
    assert ns.speculative_algorithm == "DSPARK"
    assert float(ns.mem_fraction_static) == 0.8623
    assert int(ns.max_total_tokens) == 381184
    assert ns.weight_loader_drop_cache_after_load is True


def test_prod_and_every_arm_pass_the_v41_reasoning_and_tool_call_parsers():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    for args in (arm_env.ServerArgs.prod(), arm_env.ServerArgs(port=1), arm_env.ServerArgs(port=1, dspark=True)):
        # parse_args rejects a name outside each flag's registered choices ("deepseek_v41" is not one).
        ns = parser.parse_args(args.argv()[3:])
        assert ns.reasoning_parser == "deepseek-v41"
        assert ns.tool_call_parser == "deepseekv41"
        assert ns.language_model_only is True


def test_prod_warms_decode_after_the_prefill_shapes_and_the_arms_keep_prefill_only():
    from sglang.srt.entrypoints.warmup import _warmup_registry

    argv = arm_env.ServerArgs.prod().argv()
    # Order matters: the decode warm-up's long prompt evicts the SWA KV that the prefill warm-up left cached.
    assert argv[argv.index("--warmups") + 1] == "dsv41_prefill_shapes,dsv41_decode_paths"
    assert {"dsv41_prefill_shapes", "dsv41_decode_paths"} <= set(_warmup_registry)
    for args in (arm_env.ServerArgs(port=1), arm_env.ServerArgs(port=1, dspark=True)):
        argv = args.argv()
        assert argv[argv.index("--warmups") + 1] == "dsv41_prefill_shapes"


def test_prod_and_every_arm_bound_the_prefill_share_with_the_recorder_it_needs():
    # The service refuses SGLANG_DSV41_ENABLE_PREFILL_SHARE under the no-op recorder, so the share is on only where
    # argv also names a recorder mode.
    assert arm_env.prod_env()["SGLANG_DSV41_ENABLE_PREFILL_SHARE"] == "1"
    assert arm_env.base_env()["SGLANG_DSV41_ENABLE_PREFILL_SHARE"] == "1"
    for args in (arm_env.ServerArgs.prod(), arm_env.ServerArgs(port=1), arm_env.ServerArgs(port=1, dspark=True)):
        argv = args.argv()
        assert argv[argv.index("--expert-distribution-recorder-mode") + 1] == "per_pass"


def test_dspark_server_args_use_the_dspark_mem_fraction():
    assert arm_env.DSPARK_MEM_FRACTION_STATIC == "0.78"
    argv = arm_env.ServerArgs(port=1, dspark=True).argv()
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.78"
    argv = arm_env.ServerArgs(port=1).argv()
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.875"


def test_the_dspark_argv_parses_as_dspark_at_block_size_5():
    from sglang.srt.server_args import ServerArgs

    argv = arm_env.ServerArgs(port=1, dspark=True).argv()
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv[3:])
    assert ns.speculative_algorithm == "DSPARK"
    assert ns.speculative_draft_model_path == arm_env.DSPARK_DRAFT
    assert int(ns.speculative_dspark_block_size) == 5
    assert ns.max_running_requests == 1
