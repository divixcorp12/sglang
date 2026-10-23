"""run_arm.sh KEY=VAL overrides for an io_uring arm, one per line.

The io_uring options are one group env, SGLANG_EXPERT_STREAM_URING, a JSON object keyed by the UringConfig fields
(python/sglang/srt/layers/moe/expert_engine/config.py); the SQPOLL thread's core is an ``sq=`` entry of
SGLANG_EXPERT_STREAM_NUMA_CORES. The drivers (drive_iopoll_cuts_arms.sh, drive_reader_crtp_pair.sh,
drive_uring_reg_arms.sh) keep their arms as ``key=value`` words and call this to get the env they pass:

    uring_overrides.py <benchmarks/dsv41_baseline dir> mode=default queue_depth=0 sq_thread_cpu=-1 ...

``sq_thread_cpu`` is not a UringConfig field: -1 (no pinned SQPOLL core) adds nothing, a core adds the NUMA_CORES
entry on top of production's RAM-thread core (arm_env.SPIN_CORE), on the core's node.
"""

import json
import sys

BOOL_KEYS = ("fixed_files", "diagnostics", "slab_arena")
INT_KEYS = ("queue_depth", "sq_thread_idle_ms")
STR_KEYS = ("mode", "read_mode", "wait_mode", "read_cuts")


def overrides(arm_env, words: list[str]) -> list[str]:
    group: dict[str, object] = {}
    sq_cpu = -1
    for word in words:
        key, _, value = word.partition("=")
        if key == "sq_thread_cpu":
            sq_cpu = int(value)
        elif key in BOOL_KEYS:
            if value not in ("0", "1"):
                raise ValueError(f"{word}: expected 0 or 1")
            group[key] = value == "1"
        elif key in INT_KEYS:
            group[key] = int(value)
        elif key in STR_KEYS:
            group[key] = value
        else:
            raise ValueError(f"{word}: not a UringConfig field")
    lines = ["SGLANG_EXPERT_STREAM_URING=" + json.dumps(group, separators=(",", ":"))]
    if sq_cpu >= 0:
        spin = arm_env.SPIN_CORE
        if arm_env.node_of_cpu(sq_cpu) != arm_env.node_of_cpu(spin):
            raise ValueError(f"sq_thread_cpu={sq_cpu} is on another node than the RAM thread's core {spin}")
        lines.append(f"SGLANG_EXPERT_STREAM_NUMA_CORES={arm_env.node_of_cpu(spin)}:ram={spin},sq={sq_cpu}")
    return lines


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    import arm_env

    print("\n".join(overrides(arm_env, sys.argv[2:])))
