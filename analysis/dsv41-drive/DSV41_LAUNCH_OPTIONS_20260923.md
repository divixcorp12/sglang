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
| `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING` | `1` | Use Engram graph host-node lookup with io_uring for both Engram layers. |
| `SGLANG_MOE_EXPERT_FUSED_PLAN` | `1` | Enable the fused expert route planner with DIRECT stage 2. |

The production launcher explicitly supplies the first six values and inherits the
fused planner from `arm_env.base_env()`. The eight-worker
setting has not yet been compared with four workers under this new recipe, so its
throughput effect is unknown. The Engram host-node path was profiled separately;
see the
[graph-node measurement](ENGRAM_DECODE_GRAPH_NODE_MEASUREMENT_20260923.md).
A single two-session fused-plan-on arm completed on 2026-09-23, before this flag
became a default. It has no same-recipe fused-plan-off control, so its 3.579 tok/s
median is not an established speedup. Older benchmark numbers used other launch
options and are not a performance claim for the current recipe.

Previously enabled recipe settings include `uring_direct` expert reads, expert-row
mirrors, 50 GiB of pinned MoE host memory, a 14 GiB GPU hot cache, dynamic hot caching,
and expert graph gather. These are listed in `arm_env.base_env()` with their exact
values.

## Remaining cache-size work

| Option | Current value | Evidence and next decision |
| --- | ---: | --- |
| `SGLANG_DSV41_ENGRAM_RAM_GIB` | `5` | A 20 GiB Engram cache was proposed but not implemented as a pinned cache. This setting controls the Python row cache; the native host-node path hardcodes a 5 GiB budget in `engram_file_table.py`. Its cache slab uses ordinary `mmap`, while separate staging buffers are pinned. Changing this setting to 20 alone would neither enlarge nor pin the native cache. See [lookup plan](ENGRAM_LOOKUP_OPTIMIZATION_PLAN.md). |

## Off because of the current mode or limited applicability

| Option | Current value | Reason |
| --- | ---: | --- |
| `SGLANG_MOE_ASYNC_RESIDENCY_SCORES` | `0` | The current `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` path explicitly raises `ValueError` when async CPU residency scores are enabled (`expert_hot_cache.py`). A prior matched off/on/off HTTP run favored async scores with four pack workers and leases off, but that was a different residency mode; see [throughput measurement](ASYNC_RESIDENCY_THROUGHPUT_20260923.md). Enabling this requires changing or reconciling the residency update paths first. |
| `SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE` | off | DIRECT insertion rejects two-phase RAM misses. |
| `SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM` | off | Piece streaming (docs/superpowers/plans/2026-09-24-dsv41-piece-streaming.md): each RAM-miss row is read and published in 8 pieces, and the stream kernel S copies each piece as it lands, in place of W2 and C2. Refused at startup unless `SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE`, `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` and `SGLANG_DSV41_RAM_MISS_PACK_WORKERS > 0` all hold. Off until the plan's task 6 benchmark gives its verdict. |
| `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` and `SGLANG_MOE_PREFETCH_MAX_CANDIDATES` | `0` | DIRECT insertion rejects the expert prefetch path. |
| `SGLANG_MOE_EXPERT_DOORBELL` | `0` | DIRECT stage 2 rejects doorbell mode. |
| `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` | `0` | The EXL3 path does not currently benefit from this mode. |
| `SGLANG_MOE_HOT_FUSED_INSERT` | off | Applies to stage 1 insertion, rather than the selected stage 2 mode. |

Tracing-only knobs such as `SGLANG_DSV41_EXPERT_TRACE_PATH` and
`SGLANG_MOE_PREFORWARD_READINESS_DIAGNOSTIC` remain off for normal serving.

## Production and benchmark argument differences

The saved production launcher and current benchmark `ServerArgs.argv()` both enable
prefix caching and set context length to 32,768 tokens. Earlier benchmark arms used
`--disable-radix-cache` and a 4,096-token context. Compare throughput only between
runs with matching launch arguments and environment values. This benchmark change
does not start the production server.
