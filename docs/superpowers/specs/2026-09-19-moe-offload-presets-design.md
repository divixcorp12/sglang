# MoE offload presets — design

Date: 2026-09-19. Status: approved in chat, awaiting spec review.

## Problem

Serving Qwen3.8-Flash-Next-NVFP4 with expert offload on divix01 takes about 25
hand-set environment variables on top of 72 defined ones. The `Envs` defaults
encode the pre-campaign "off" behaviour, so every launcher repeats the winning
values. Many constraints between them surface only as startup errors, often
after the two-minute weight load, in the frozen `model_runner.py`. Some of
these could be derived instead of refused.

## Goals

1. One CLI flag selects a complete, documented offload configuration.
2. Two tuned presets: the current graph-gather approach and the doorbell
   approach.
3. Settings that another setting forces are derived, not hand-set.
4. Invalid combinations fail at argument resolution, in seconds, with one
   message naming the setting and the preset.
5. Each preset field documents what it does, its measured effect and its
   evidence row in `MOE_EXPERT_TRANSFER.md`.

## Non-goals

- Replacing the environment-variable reads with a config object threaded
  through consumers (approach B). The 36 reads in the frozen `model_runner.py`
  and the other read sites stay unchanged.
- Presets for non-offload server arguments (NEXTN, chunked prefill, graph
  backend). The resolver checks them against a preset's requirements and never
  sets them, except overlap scheduling, which it turns off when a rule forces
  that.
- Site-specific paths: the expert cache directory, the hot-cache seed, the
  metrics file and the PLE RSS budget stay in the launcher.
- Measuring the doorbell preset. It ships unmeasured and is labelled
  experimental.

## Surface

`--moe-offload-preset {off,graph-gather,doorbell}`, default `off`, declared
beside the PLE offload fields in `arg_groups/fields/exec_.py`. `off` is today's
behaviour exactly: no environment variable is touched and no server argument
is declared.

Precedence, highest first:

1. an environment variable the user set explicitly (`envs.X.is_set()`);
2. the selected preset's value;
3. the `Envs` default.

The resolver fills each unset variable with `envs.X.set(value)` in the launcher
process, before the scheduler and worker processes are spawned, so they inherit
it. It logs one line per preset-filled value and one line per explicit value
that overrides the preset, so a run's effective offload config can always be
reconstructed from its log.

## Components

### `python/sglang/srt/layers/moe/offload_presets.py` (new)

- `MoeOffloadPreset(msgspec.Struct, frozen=True, kw_only=True)`: one field per
  setting a preset controls, typed as the `Envs` descriptor parses it. Each
  field carries a comment with its effect and evidence row. `None` means the
  preset leaves that variable alone.
- A module-level mapping from field name to `Envs` descriptor name. A test
  checks that it is complete.
- Two constants, `GRAPH_GATHER_PRESET` and `DOORBELL_PRESET`. A docstring on
  each says when to use it, its measured speed (or "unmeasured") and its hard
  requirements.
- Pure functions, with no reads of process state:
  - `preset_env(preset) -> dict[str, str]`, the variables a preset sets;
  - `derive_env(values) -> dict[str, str]`, the variables forced by others
    (see **Auto-derivation**);
  - `check_offload_config(values, *, speculative, decode_graph_disabled,
    decode_max_bs, parallel, allowed_cpus) -> None`, which raises `ValueError`
    for invalid combinations (see **Validation**);
  - `needs_overlap_off(values) -> bool`.

### `python/sglang/srt/arg_groups/moe_offload_hook.py` (new)

`handle_moe_offload_preset(server_args)` is the only impure part:

1. read the preset and the currently set environment;
2. merge with the precedence above and apply `derive_env`;
3. raise if an explicitly set variable conflicts with a derived value;
4. `envs.X.set` every filled variable and log it;
5. `declare_resolution(server_args, "moe_offload_preset",
   disable_overlap_schedule=True)` when `needs_overlap_off` and overlap is on.
   If the user explicitly asked for overlap in a way that conflicts, raise;
6. run `check_offload_config`.

It is wired into `arg_groups/pipeline.py` immediately before
`handle_offload_compatibility`, so the existing validations read the filled
values.

### `environ.py`

Register `SGLANG_MOE_EXPERT_STREAM = EnvBool(False)`. Its two raw
`os.environ.get` reads (`expert_stream.py:366`, `memory_hook.py:103`) move to
`envs.SGLANG_MOE_EXPERT_STREAM.get()`, so the preset can set it through the
same path.

## Presets

Shared by both presets:

| Variable | Value | Evidence |
|---|---|---|
| `SGLANG_MOE_EXPERT_STREAM` | `1` | required for NVFP4 hot caching |
| `SGLANG_MOE_EXPERT_FILE_READER` | `uring_direct` | production reader |
| `SGLANG_QWEN4_PLE_FILE_READER` | `uring` | production reader |
| `SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY` | `1` | no graph break for PLE reads |
| `SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING` | `1` | #33, frees 1.18 GB |
| `SGLANG_MOE_HOT_GPU_MB` | `15360` | #34; tuned for 32 GB RTX 5090, override elsewhere |
| `SGLANG_MOE_PINNED_HOST_MB` | `0` | host arena replaces the pinned LRU |
| `SGLANG_MOE_EXPERT_HOST_ARENA` | `1` | production host path |
| `SGLANG_MOE_EXPERT_COPY_BACKEND` | `dma` | production copy path |
| `SGLANG_MOE_EXPERT_GRAPH_GATHER` | `1` | both approaches require it |
| `SGLANG_MOE_HOT_DYNAMIC` | `1` | dynamic residency |
| `SGLANG_MOE_HOT_DECAY_TOKENS` | `1` | tuned residency policy |
| `SGLANG_MOE_HOT_PROMOTION_SIGMAS` | `0` | tuned residency policy |
| `SGLANG_MOE_HOT_BENEFIT_RATIO` | `2` | tuned residency policy |
| `SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS` | `0` | tuned residency policy |
| `SGLANG_MOE_GPU_RESIDENCY_UPDATE` | `1` | in-graph residency update |
| `SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS` | `64` | production value |
| `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` | `1` | insert-on-miss needs a boundary every decode forward |
| `SGLANG_MOE_PREFETCH_MAX_CANDIDATES` | `0` | prefetch rejected (#5) |
| `SGLANG_MOE_EXPERT_FUSED_PLAN` | `1` | +6.4% |

`graph-gather` (the current best: 29.30 / 29.88 tok/s median at NEXTN-3 on
divix01, measured before `7b498893cd`):

| Variable | Value |
|---|---|
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` | `2` |
| `SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT` | `1` (no effect without a MIXED_PRECISION draft) |

Overlap scheduling stays on.

`doorbell` (experimental, unmeasured on this stack):

| Variable | Value |
|---|---|
| `SGLANG_MOE_EXPERT_DOORBELL` | `1` |
| `SGLANG_MOE_EXPERT_DOORBELL_MODE` | `current` |
| `SGLANG_MOE_EXPERT_DOORBELL_CPU` | `71` |
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` | `1` (stage 2 refuses the doorbell) |

Overlap scheduling is turned off. Speculative decoding is refused.

The doorbell preset's compatibility with stage 1 and the fused planner is
confirmed by a startup smoke test during implementation. A refused combination
is dropped from the preset, with the reason recorded in its docstring.

## Auto-derivation

Applied with or without a preset. Each rule is a pure function of the merged
values and fills a variable only if it is unset. An explicitly set variable
that contradicts a derived value raises, naming both variables.

| When | Derived |
|---|---|
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` >= 1 | `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`, `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1` |
| `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` | `SGLANG_MOE_HOT_DYNAMIC=1` |
| `SGLANG_MOE_EXPERT_DOORBELL=1` | overlap scheduling off |
| `SGLANG_MOE_HOT_GPU_MB > 0` and not both `SGLANG_MOE_EXPERT_GRAPH_GATHER=1` and `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` | overlap scheduling off |

Overlap is turned off through `declare_resolution`. The resolver never writes a
server-argument field directly.

## Validation

`check_offload_config` raises at argument resolution for:

- graph gather with the decode CUDA graph backend `disabled`;
- the residency update with a decode graph batch size above 1;
- the doorbell with speculative decoding, TP, PP, DP or DP attention, or a
  doorbell CPU outside the process's allowed set (`os.sched_getaffinity(0)`,
  which fixes the silent pinning failure recorded in
  `MOE_EXPERT_TRANSFER.md`);
- stage 2 with the doorbell;
- `SGLANG_MOE_HOT_GPU_MB > 0` without `SGLANG_MOE_EXPERT_STREAM=1`.

The existing checks in `model_runner.py`, `expert_hot_cache.py` and
`expert_residency_gpu.py` stay as a backstop. These checks only move the
failure ahead of the weight load.

## Testing

CPU unit tests, in one new file beside the other `arg_groups` tests:

1. Each preset's `preset_env` equals the offload variables of the current prod
   script, listed in the test as an external-source literal. This guards the
   preset against drifting from production.
2. An explicitly set variable wins over the preset. An explicit value that
   contradicts a derived one raises.
3. Each auto-derivation rule fires, and a contradicting explicit value raises.
4. The doorbell preset declares overlap off and refuses NEXTN.
5. `off` sets no variable and declares nothing.
6. The field-to-descriptor mapping covers every `MoeOffloadPreset` field, and
   every descriptor exists in `Envs`.

GPU: one startup smoke test per preset (healthy, one coherent reply, no timing).
The prod script then shrinks to `--moe-offload-preset graph-gather` plus its
site-specific paths, and the effective environment the resolver logs is
compared line for line with today's script.

## Rollout

1. Land presets, resolver and tests with default `off`. Behaviour does not
   change.
2. Smoke-test both presets on divix01.
3. Switch the prod script to the flag, keep a backup, and do not start prod
   unless asked.
4. Record the presets in `MOE_EXPERT_TRANSFER.md` ("Prod server run").

## Follow-up (not in this change)

Approach B, a typed config threaded through consumers, would remove the
environment reads. It touches the frozen `model_runner.py` and needs its own
design.
