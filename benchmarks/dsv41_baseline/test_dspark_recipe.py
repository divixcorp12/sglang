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


def test_prod_is_unchanged_until_the_switch():
    assert arm_env.PROD_DSPARK is False
    assert arm_env.prod_env() == arm_env.base_env()
    assert "--speculative-algorithm" not in arm_env.ServerArgs.prod().argv()
    assert arm_env.prod_env()["SGLANG_MOE_HOT_GPU_MB"] == "16080"
    prod = arm_env.ServerArgs.prod().argv()
    assert prod[prod.index("--mem-fraction-static") + 1] == "0.875"


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
