# DSV4.1 launch options, 23 September 2026

This is the option ledger for the DSV4.1 EXL3 server. The shared environment recipe is
[`benchmarks/dsv41_baseline/arm_env.py`](../../benchmarks/dsv41_baseline/arm_env.py).
The saved production launcher on divix01 is
`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live/launch.sh`.
Its explicit overrides take precedence over the shared recipe. These are DSV4.1
launch defaults, not global defaults for all SGLang models. Changing a saved launch
file or recipe does not change an already running server.

## Enabled in the current DSV4.1 recipe

| Option | Value | Purpose |
| --- | ---: | --- |
| `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` | `1` | Coordinate RAM miss ownership. |
| `SGLANG_MOE_GPU_RESIDENCY_UPDATE` | `1` | Update GPU expert residency. |
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` | `2` | Insert fetched experts into the GPU cache with DIRECT stage 2. |
| `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` | `1` | Update the hot set each decode forward. |
| `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` | `8` | Use eight expert-row packing workers. |

The production launcher explicitly supplies these five values. The eight-worker
setting has not yet been compared with four workers under this new recipe, so its
throughput effect is unknown. Existing benchmark numbers were collected with other
launch options and are not a performance claim for this combination.

Previously enabled recipe settings include `uring_direct` expert reads, expert-row
mirrors, 50 GiB of pinned MoE host memory, a 14 GiB GPU hot cache, dynamic hot caching,
and expert graph gather. These are listed in `arm_env.base_env()` with their exact
values.

## Implemented or measured, still opt-in

| Option | Current value | Evidence and next decision |
| --- | ---: | --- |
| `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING` | off | The Engram graph host-node lookup with io_uring has been implemented and profiled for both Engram layers. It is not enabled by the recipe or saved production launcher. See [graph-node measurement](ENGRAM_DECODE_GRAPH_NODE_MEASUREMENT_20260923.md). Test the current launch combination before enabling it by default. |
| `SGLANG_MOE_ASYNC_RESIDENCY_SCORES` | `0` | One matched off/on/off HTTP run showed eight of eight session wins, with median paired gains of 4.07% and 5.20% against the two off arms. This used four pack workers and leases off, and the HTTP step-latency field was unavailable. See [throughput measurement](ASYNC_RESIDENCY_THROUGHPUT_20260923.md). Remeasure with the current DIRECT and eight-worker recipe before defaulting it on. |
| `SGLANG_DSV41_ENGRAM_RAM_GIB` | `5` | A 20 GiB Engram cache was proposed but not implemented as a pinned cache. This setting controls the Python row cache; the native host-node path hardcodes a 5 GiB budget in `engram_file_table.py`. Its cache slab uses ordinary `mmap`, while separate staging buffers are pinned. Changing this setting to 20 alone would neither enlarge nor pin the native cache. See [lookup plan](ENGRAM_LOOKUP_OPTIMIZATION_PLAN.md). |

## Off because of the current mode or limited applicability

| Option | Current value | Reason |
| --- | ---: | --- |
| `SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE` | off | DIRECT insertion rejects two-phase RAM misses. |
| `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` and `SGLANG_MOE_PREFETCH_MAX_CANDIDATES` | `0` | DIRECT insertion rejects the expert prefetch path. |
| `SGLANG_MOE_EXPERT_DOORBELL` | `0` | DIRECT stage 2 rejects doorbell mode. |
| `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` | `0` | The EXL3 path does not currently benefit from this mode. |
| `SGLANG_MOE_HOT_FUSED_INSERT` | off | Applies to stage 1 insertion, rather than the selected stage 2 mode. |
| `SGLANG_MOE_EXPERT_FUSED_PLAN` | off | Separate optional expert-planning path; not part of the selected recipe. |

Tracing-only knobs such as `SGLANG_DSV41_EXPERT_TRACE_PATH` and
`SGLANG_MOE_PREFORWARD_READINESS_DIAGNOSTIC` remain off for normal serving.

## Production and benchmark argument differences

The saved production launcher enables prefix caching and sets context length to
32,768 tokens. The benchmark `ServerArgs.argv()` still contains
`--disable-radix-cache` and a 4,096-token context, preserving its historical
measurement setup. Compare throughput only between runs with matching launch
arguments and environment values. The production server was shut down when this
ledger was written; this change does not start it.
