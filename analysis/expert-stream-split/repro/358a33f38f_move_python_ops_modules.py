import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanical_refactor_reproduction_utils import git_add_and_commit, verify_mechanical_refactor

BASE_COMMIT = "0c6e22b80dd1b9f92fcf14036d5c90a1061ebca3"
TARGET_COMMIT = "358a33f38faf02aaeb40865ac55b6165bf82a774"

FILES = """analysis/dsv41-drive/bench_pack_workers.py
analysis/dsv41-drive/busy_seq_protocol_probe.py
analysis/dsv41-drive/open11/open11_arming_cost.py
analysis/dsv41-drive/task6-microbench/g_harness.py
python/sglang/kernels/ops/moe/expert_stream_transport.py
python/sglang/srt/layers/moe/exl3_ram_miss.py
python/sglang/srt/layers/moe/exl3_stream_trace.py
python/sglang/test/dsv41_lease_sim.py
scripts/dsv41/exl3_stage_trace_overhead.py
test/manual/dsv41/test_exl3_copy_engine_cuda.py
test/manual/dsv41/test_exl3_lease_kernels_cuda.py
test/manual/dsv41/test_exl3_native_prefetch_cuda.py
test/manual/dsv41/test_exl3_piece_stream_cuda.py
test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py
test/manual/dsv41/test_exl3_ram_miss_cuda.py
test/manual/dsv41/test_exl3_task5_item7_option_f_gpu.py
test/manual/dsv41/test_exl3_two_phase_failure_cuda.py
test/manual/dsv41/test_exl3_two_phase_parity_cuda.py
test/manual/dsv41/test_exl3_two_phase_timing_cuda.py
test/registered/unit/kernels/test_exl3_lease_block.py
test/registered/unit/kernels/test_exl3_native_prefetch_service.py
test/registered/unit/kernels/test_exl3_ram_miss_advisory.py
test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py
test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py
test/registered/unit/kernels/test_exl3_ram_miss_device_args.py
test/registered/unit/kernels/test_exl3_ram_miss_lease_defer.py
test/registered/unit/kernels/test_exl3_ram_miss_lease_service.py
test/registered/unit/kernels/test_exl3_ram_miss_lease_thread.py
test/registered/unit/kernels/test_exl3_ram_miss_lease_wrap.py
test/registered/unit/kernels/test_exl3_ram_miss_leases.py
test/registered/unit/kernels/test_exl3_ram_miss_pack_workers.py
test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py
test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py
test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py
test/registered/unit/kernels/test_exl3_ram_miss_row_images.py
test/registered/unit/kernels/test_exl3_ram_miss_split.py
test/registered/unit/kernels/test_exl3_ram_miss_stage_trace.py
test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py
test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_lanes.py
test/registered/unit/kernels/test_exl3_ram_miss_task5_item5_ack_independence.py
test/registered/unit/kernels/test_exl3_ram_miss_thread.py
test/registered/unit/kernels/test_exl3_ram_miss_tier.py
test/registered/unit/kernels/test_exl3_ram_miss_trace_export.py
test/registered/unit/kernels/test_exl3_ram_miss_two_phase.py
test/registered/unit/kernels/test_exl3_ram_miss_two_phase_victim.py
test/registered/unit/kernels/test_exl3_ram_miss_wrap.py
test/registered/unit/layers/moe/test_exl3_ram_miss_service.py""".split()

# The two files that were themselves relocated (already-renamed names in FILES above).
OLD_MODULE_PATH = "python/sglang/kernels/ops/moe/exl3_ram_miss.py"
NEW_MODULE_PATH = "python/sglang/kernels/ops/moe/expert_stream_transport.py"
OLD_LEASE_PATH = "python/sglang/kernels/ops/moe/exl3_lease_block.py"
NEW_LEASE_PATH = "python/sglang/kernels/ops/moe/expert_lease_block.py"

# Ordered substitutions, exactly mirroring the sed passes used to author the commit.
# 1) fully-qualified dotted-attribute form: sglang.kernels.ops.moe.exl3_ram_miss(.anything)
# 2) fully-qualified dotted-attribute form for the lease module
# 3) "from sglang.kernels.ops.moe import exl3_ram_miss" (any suffix incl. " as X")
# 4) "from sglang.kernels.ops.moe import exl3_lease_block" (any suffix incl. " as X")
# 5) bare-name-dot-attribute cascade for the (rare) unaliased local bindings
# 6) revert the two places where pass 5 over-matched a C++ source-file string, not a
#    Python module-attribute access (the device .cuh file itself is not renamed).
SUBS = [
    (re.compile(r"sglang\.kernels\.ops\.moe\.exl3_ram_miss"), "sglang.kernels.ops.moe.expert_stream_transport"),
    (re.compile(r"sglang\.kernels\.ops\.moe\.exl3_lease_block"), "sglang.kernels.ops.moe.expert_lease_block"),
    (re.compile(r"sglang\.kernels\.ops\.moe import exl3_ram_miss"), "sglang.kernels.ops.moe import expert_stream_transport"),
    (re.compile(r"sglang\.kernels\.ops\.moe import exl3_lease_block"), "sglang.kernels.ops.moe import expert_lease_block"),
    (re.compile(r"\bexl3_ram_miss\."), "expert_stream_transport."),
    (re.compile(r"\bexl3_lease_block\."), "expert_lease_block."),
]

REVERT_CUH = [
    (re.compile(r"expert_stream_transport\.cuh"), "exl3_ram_miss.cuh"),
]


def transform(dir_root: Path) -> None:
    old_module = dir_root / OLD_MODULE_PATH
    new_module = dir_root / NEW_MODULE_PATH
    old_lease = dir_root / OLD_LEASE_PATH
    new_lease = dir_root / NEW_LEASE_PATH

    new_module.parent.mkdir(parents=True, exist_ok=True)
    old_module.rename(new_module)
    old_lease.rename(new_lease)

    for rel in FILES:
        path = dir_root / rel
        text = path.read_text()
        for pattern, repl in SUBS:
            text = pattern.sub(repl, text)
        for pattern, repl in REVERT_CUH:
            text = pattern.sub(repl, text)
        path.write_text(text)

    git_add_and_commit("refactor(expert-stream): generic Python ops module names", cwd=str(dir_root))


if __name__ == "__main__":
    verify_mechanical_refactor(BASE_COMMIT, TARGET_COMMIT, transform)
