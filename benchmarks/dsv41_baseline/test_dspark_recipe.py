"""The DSpark mode of the recipe: both CPU-expert clients on, the graphed verify's configuration, and argv the server
parses as DSpark (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 15)."""

import argparse
import json

import arm_env


def test_dspark_env_turns_both_cpu_expert_clients_on_with_spill():
    env = arm_env.arm_env(arm_env.dspark_env())
    assert env["SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS"] == "1"
    assert env["SGLANG_EXPERT_STREAM_ENABLE_DRAFT_CPU_EXPERTS"] == "1"
    # One team per node: the draft runs on node 0's CPU expert team, so no draft cores are named (Task 12 refuses them).
    assert "SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_CORES" not in env
    assert "SGLANG_EXPERT_STREAM_DRAFT_CPU_EXPERTS_THREADS" not in env
    assert "SGLANG_EXPERT_STREAM_GRAPH_GATHER_MISS_LANES" not in env  # a lane per route: 36 on a 40-lane wire
    assert env["SGLANG_EXPERT_STREAM_GRAPH_GATHER_VICTIM_LANES"] == "8"
    assert env["SGLANG_RAGGED_VERIFY_MODE"] == "static"
    assert env["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"] == "0"
    # Budget A (mem-budget-report.md L2): the DSpark arms trade hot cache for prefill headroom.
    assert env["SGLANG_EXPERT_STREAM_HOT_GPU_MB"] == "10840"
    assert env["SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS"] == "1"  # the recipe's, kept


def test_the_pinned_row_sizes_are_the_served_checkpoints():
    """arm_env records each layer's pinned row so the launch floor is known before the server starts; the server sizes
    its slabs from the checkpoint itself. A new export (other bit widths) must update the record, or the floor misjudges
    the tier. Runs where the checkpoint is (divix01)."""
    import os

    import pytest

    if not os.path.isdir(arm_env.EXPERT_DIR):
        pytest.skip(f"{arm_env.EXPERT_DIR} is not on this machine")
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_row_image import row_image_layouts

    layout = build_exl3_expert_layout(arm_env.EXPERT_DIR)
    images = row_image_layouts(layout, range(layout.num_layers))
    assert layout.num_layers == arm_env.PROD_PINNED_LAYERS
    assert tuple(images[layer].image_bytes for layer in range(layout.num_layers)) == arm_env.PROD_PINNED_ROW_BYTES_BY_LAYER


def test_prod_serves_the_dspark_mode_with_its_own_settings():
    assert arm_env.PROD_DSPARK is True
    assert arm_env.prod_env() == arm_env.arm_env({**arm_env.dspark_env(), **arm_env.prod_dspark_env()})
    argv = arm_env.ServerArgs.prod().argv()
    assert argv[argv.index("--speculative-algorithm") + 1] == "DSPARK"


# MERGE_PLAN_1016.md: production is the best served arm, RAM prefetch included, with both startup calibrations on.
PROD_DSPARK_SETTINGS = {
    "SGLANG_EXPERT_STREAM_PINNED_HOST_FULL_LAYERS": ",".join(str(layer) for layer in range(40)),
    # The 1017 checkpoint (2026-10-23): K2 layers 18-22 hold smaller rows (arm_env.PROD_PINNED_ROW_BYTES_BY_LAYER).
    "SGLANG_EXPERT_STREAM_PINNED_HOST_MB": "188900",
    "SGLANG_EXPERT_STREAM_PINNED_HOST_NUMA_MB": "0:94450,1:94450",
    # Master, 2026-10-22: the DIRECT insert score without eager routes, decay 0.999, newcomer probation.
    "SGLANG_EXPERT_STREAM_ENABLE_HOT_INSERT_EAGER_COUNTS": "0",
    "SGLANG_EXPERT_STREAM_HOT_INSERT_ON_MISS_DECAY": "0.999",
    "SGLANG_EXPERT_STREAM_ENABLE_HOT_INSERT_PROBATION": "1",
    "SGLANG_ENABLE_SCHEDULER_MLOCK": "1",
    "SGLANG_HICACHE_HOST_NUMA_NODE": "1",
    "SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB": "128",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_EXPERT_STREAM_HOT_GPU_MB": "12623",
    "SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP": "1",
    "SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS": "0",
    "SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB": "128",
    "SGLANG_DSPARK_CONFIDENCE_CUT": "1",
    "SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED": "1",
    "SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST": "1",
    "SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL": "0",
}


def test_prod_env_carries_the_best_arms_settings_and_both_startup_calibrations():
    env = arm_env.prod_env()
    assert {k: env.get(k) for k in PROD_DSPARK_SETTINGS} == PROD_DSPARK_SETTINGS
    # The CPU split's six settings are one JSON env (SGLANG_EXPERT_STREAM_CPU_SPLIT); a key left out keeps its default.
    split = json.loads(env["SGLANG_EXPERT_STREAM_CPU_SPLIT"])
    assert {"model": True, "model_refine_batches": 0, "calibration": True, "miss_cut": 1, "miss_cut_max": 3}.items() <= split.items()
    assert split.get("model_costs", "") == ""


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
    assert json.loads(env.get("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH") or "{}").get("pull_mode", "off") == "off"
    assert int(env.get("SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS", "0")) == 0
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
    for held in ("SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM", "SGLANG_OPT_LANE_HEAD_PATH", "SGLANG_EXPERT_STREAM_GPU_DRIP_SLOTS",
                 "SGLANG_DSPARK_TOP_PROB_GATE",
                 "SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT", "SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL",
                 "SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION"):
        assert held not in env, held
    assert not json.loads(env["SGLANG_EXPERT_STREAM_CPU_SPLIT"]).get("model_residency_tiebreak", False)
    # Measurement only: the instrumented build's traces and the hot cache's metrics file are one trace env, absent.
    assert not any("TRACE" in k for k in env)


def test_prod_runs_the_sampled_confidence_cut_without_block_verification_or_the_greedy_draft():
    # DSV_OPTIMIZATION.md 3.12, owner 2026-10-18: the cut for sampled requests with the draft's time in T(w) and no
    # online refit, on the fitted STS file. Block verification and the greedy draft stay off; STS collection is off.
    # Owner, 2026-10-20: T(w) is refit live every 32 timed cut verifies (DSV_OPTIMIZATION.md 3.12).
    env = arm_env.prod_env()
    assert {k: env[k] for k in arm_env.PROD_DSPARK_CUT_ENV} == {
        "SGLANG_DSPARK_CONFIDENCE_CUT": "1",
        "SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED": "1",
        "SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST": "1",
        "SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL": "0",
        "SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL": "32",
    }
    assert "SGLANG_DSPARK_STS_COLLECT_PATH" not in env and env.get("SGLANG_DSPARK_USE_GREEDY_DRAFT", "0") == "0"
    argv = arm_env.ServerArgs.prod().argv()
    assert argv[argv.index("--speculative-dspark-confidence-sts-path") + 1] == arm_env.PROD_DSPARK_STS_PATH
    assert "--speculative-use-block-verification" not in argv
    # The arms build on dspark_env(), which states none of it.
    assert not any("DSPARK_STS" in k or "CONFIDENCE_CUT" in k for k in arm_env.dspark_env())
    assert "--speculative-dspark-confidence-sts-path" not in arm_env.ServerArgs(port=1, dspark=True).argv()
    # The fitted STS file: a production asset path with its digest pinned.
    assert arm_env.PROD_DSPARK_STS_PATH.startswith("/data/models/slang/prod-assets/dspark-sts/")
    assert len(arm_env.PROD_DSPARK_STS_SHA256) == 64


# The best arm's prefetch keys, from divix01:/mnt/nvme1/share64-ab-20261016/resolved-env.json "on" (its head was
# nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors, production's is the checksummed copy).
SERVED_PREFETCH = {
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH": "1",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_HORIZON": "2",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_DEADLINE_US": "4000",
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE": "1",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_IDLE_PIECES": "2",
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_SPREAD": "1",
    "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": "",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_LAYER": "2",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_PER_TOKEN": "1",
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_POOL_SERVE": "1",
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_PROMOTE": "1",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER": "gpu",
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SPEC_SHARE": "2",
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_TOP_K_ONLY": "0",
    "SGLANG_EXPERT_STREAM_LEGACY_PREFETCH_MAX_CANDIDATES": "0",
}

# The served prefetch keys, by name prefix (the learned head and margin floors keep their SGLANG_DSV41 names).
PREFETCH_PREFIXES = (
    "SGLANG_EXPERT_STREAM_RAM_PREFETCH",
    "SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH",
    "SGLANG_DSV41_RAM_PREFETCH",
)


def test_prod_serves_the_best_arms_prefetch_with_the_checksummed_head():
    env = arm_env.prod_env()
    prefetch = {k: v for k, v in env.items() if k.startswith(PREFETCH_PREFIXES)}
    head = prefetch.pop("SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD")
    legacy = "SGLANG_EXPERT_STREAM_LEGACY_PREFETCH_MAX_CANDIDATES"
    assert {**prefetch, legacy: env[legacy]} == SERVED_PREFETCH
    assert head == arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD
    assert head.startswith("/data/models/slang/prod-assets/") and len(arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD_SHA256) == 64
    # The arms keep their own: dspark_env() states no prefetch.
    assert not any(k.startswith(PREFETCH_PREFIXES) for k in arm_env.dspark_env())


def test_check_prod_assets_accepts_the_pinned_digest_and_refuses_any_other(tmp_path):
    import hashlib

    import pytest

    head = tmp_path / "head.safetensors"
    head.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    sts = tmp_path / "sts.json"
    sts.write_bytes(b"temperatures")
    sts_sha = hashlib.sha256(b"temperatures").hexdigest()
    assert arm_env.check_prod_assets(str(head), sha, str(sts), sts_sha) == (
        f"prefetch head verified: {head} sha256 {sha}\nconfidence STS file verified: {sts} sha256 {sts_sha}"
    )
    with pytest.raises(SystemExit, match="not the pinned"):
        arm_env.check_prod_assets(str(head), "0" * 64, str(sts), sts_sha)
    with pytest.raises(SystemExit, match="is missing"):
        arm_env.check_prod_assets(str(tmp_path / "none"), sha, str(sts), sts_sha)
    # The cut's STS file is checked the same way: a wrong digest or a missing file refuses the launch.
    with pytest.raises(SystemExit, match="STS file .* has sha256 .*, not the pinned " + "0" * 64):
        arm_env.check_prod_assets(str(head), sha, str(sts), "0" * 64)
    with pytest.raises(SystemExit, match="STS file .*none.* is missing"):
        arm_env.check_prod_assets(str(head), sha, str(tmp_path / "none"), sts_sha)


def test_prod_argv_carries_its_memory_and_the_arms_keep_theirs():
    argv = arm_env.ServerArgs.prod().argv()
    assert argv.count("--mem-fraction-static") == 1
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.8623"
    assert argv[argv.index("--max-total-tokens") + 1] == "381184"
    assert "--weight-loader-drop-cache-after-load" in argv
    # The arms' math (both_cpu_ab.fraction_keeping_kv) reads the OFF arm's measured init, unchanged.
    assert arm_env.DSPARK_HOT_GPU_MB == "10840" and arm_env.dspark_env()["SGLANG_EXPERT_STREAM_HOT_GPU_MB"] == "10840"
    arm = arm_env.ServerArgs(port=1, dspark=True).argv()
    assert arm[arm.index("--mem-fraction-static") + 1] == "0.78"
    assert "--max-total-tokens" not in arm and "--weight-loader-drop-cache-after-load" not in arm
    for k in (*PROD_DSPARK_SETTINGS, "SGLANG_EXPERT_STREAM_CPU_SPLIT"):
        if k not in arm_env.base_env() and k != "SGLANG_EXPERT_STREAM_HOT_GPU_MB":
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
    # The service refuses SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE under the no-op recorder, so the share is on only where
    # argv also names a recorder mode.
    assert arm_env.prod_env()["SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE"] == "1"
    assert arm_env.base_env()["SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE"] == "1"
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
