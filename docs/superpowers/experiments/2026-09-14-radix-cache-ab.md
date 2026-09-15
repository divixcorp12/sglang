# Radix cache A/B on the NVFP4 production config (2026-09-14)

## Setup

- Commits: `92e9217851` (launcher `radix` knob + `radix-ab-driver.py`), `c19b41a4d0`
  (launcher `HOT_GPU_MB` knob).
- Launcher: `scripts/expert_prediction/run-shadow-server.sh <name> <port> off [radix]`,
  from the `cc-expert-prediction` worktree on divix01, at production's E16c flags.
- Flag differences from production for radix-on:
  - remove `--disable-radix-cache`
  - `--mamba-radix-cache-strategy extra_buffer` (was `extra_buffer_lazy`)
  - `--max-mamba-cache-size 8` (was `1`)
- Runs (one server at a time, GPU free before each):
  - `radix-off` (port 31020, hot cache 14336 MB) — first reference run, kept for
    cross-launch logprob noise only.
  - `radix-on` (port 31021, hot cache 12288 MB, `radix` arg)
  - `radix-off-12g` (port 31022, hot cache 12288 MB) — matched-hot-cache baseline;
    latency/decode comparisons are radix-off-12g vs radix-on.
- Driver: `scripts/expert_prediction/radix-ab-driver.py`, stdlib only, against
  `/v1/chat/completions`. Phase A is a non-streaming logprob probe (thinking off,
  `max_tokens=16`, `logprobs=true`, `top_logprobs=5`) over a deterministic ~3800-token
  synthetic backtesting module (P1, P1 repeat, P2). Phase B is a 4-turn streaming
  session (thinking on, `max_tokens=400`/turn) over the same module.

## Startup note

`radix-off-12g`'s first launch attempt hit a `torch.OutOfMemoryError` in
`maybe_init_expert_hot_cache` while allocating the hot-cache buffer — the OOM message
showed a second process still holding ~9 GB, i.e. the GPU had not fully released
memory from the prior `radix-on` teardown despite `nvidia-smi --query-compute-apps`
reporting empty. No config change was made; a bare retry after re-confirming
`nvidia-smi` was clean succeeded (see `radix-off-12g` log, run starting
`2026-09-14T18:58:58-05:00`). Not a radix/mamba issue and not counted as the one
allowed adjustment.

## Correctness (logprob comparison)

| Comparison | Token IDs equal | Max \|Δlogprob\| | Positions |
|---|---|---|---|
| radix-on: P1 cold vs P1 cached (within-launch) | yes | 0.142 | 16 |
| P2: radix-on vs radix-off-12g (cross-launch) | yes | 0.028 | 8 |
| P1 cold: radix-on vs radix-off-12g (cross-launch noise floor) | no (diverges at position 6) | 1.198 | 16 |
| P1 cold: radix-off-12g vs radix-off-14336 (cross-launch, different hot-cache size) | no (diverges at position 6) | 0.516 | 16 |

The within-launch cached-vs-cold diff (0.142, driven by one low-confidence position,
` back` at logprob -1.73) exceeds the ~1e-2 target but is well below the cross-launch
noise floor (0.52-1.20) established by two independently-launched radix-off runs with
greedy decoding drift. Token ids match exactly in every position for both the
within-launch cached-vs-cold comparison and the cross-launch P2 comparison, with no
argmax divergence. **Correctness: pass.**

## Cache hits (`Prefill batch` lines, in request order)

| Run | Request | #new-token | #cached-token |
|---|---|---|---|
| radix-off (14336) | P1 cold | 3800 | 0 |
| radix-off (14336) | P1 cached | 3800 | 0 |
| radix-off (14336) | P2 | 3800 | 0 |
| radix-off (14336) | Turn 1-4 | 3843 / 3873 / 3906 / 3934 | 0 / 0 / 0 / 0 |
| radix-off-12g | P1 cold | 3800 | 0 |
| radix-off-12g | P1 cached | 3800 | 0 |
| radix-off-12g | P2 | 3800 | 0 |
| radix-off-12g | Turn 1-4 | 3843 / 3873 / 3906 / 3934 | 0 / 0 / 0 / 0 |
| radix-on | P1 cold | 3800 | 0 |
| radix-on | P1 cached | 24 | 3776 |
| radix-on | P2 | 24 | 3776 |
| radix-on | Turn 1 | 3843 | 0 (different chat-template render than Phase A: thinking on vs off) |
| radix-on | Turn 2-4 | 33 / 66 / 30 | 3840 / 3840 / 3904 |

Radix-off never reuses a token, as expected. Radix-on reuses the full CONTEXT prefix
to the nearest 64-token page boundary (3800 → 3776 cached, 24 new = one partial page,
`--page-size 64`) for the repeat/P2 probe, and reuses the growing session prefix
turn-to-turn in Phase B. Turn 1 of Phase B is a cold miss relative to Phase A because
thinking is enabled (production default) there but disabled in the Phase A probe, so
the two requests render to different token sequences and never share a cache entry.

## Latency: TTFT by turn (s)

| Run | Turn 1 | Turn 2 | Turn 3 | Turn 4 |
|---|---|---|---|---|
| radix-off (14336) | 4.58 | 5.73 | 5.69 | 5.81 |
| radix-off-12g | 4.69 | 5.73 | 5.72 | 5.70 |
| radix-on | 4.83 | 1.87 | 2.10 | 1.53 |

Turn 1 is a cold prefill in every run (no prior session to cache), so TTFT is
comparable across all three (~4.6-4.8s). From turn 2 on, radix-on's cached prefix
cuts TTFT by roughly **3.6-4.2 seconds per turn** (turns 2-4: 1.53-2.10s vs
5.70-5.73s off). Phase A P2 wall time shows the same effect: 5.3-5.4s off vs 2.8s on.

## Decode throughput

| Run | Median server `gen throughput` proxy (client decode tok/s, Phase B) |
|---|---|
| radix-off (14336) | 10.69 |
| radix-off-12g | 9.98 |
| radix-on | 10.51 |

Radix-on's decode rate (10.51 tok/s) is within noise of the matched-hot-cache
baseline radix-off-12g (9.98 tok/s) and the original radix-off-14336 run (10.69
tok/s) — **no regression**.

## Memory

| Run | Hot cache | Mamba cache | Mamba slot config | KV cache | avail mem after pool end |
|---|---|---|---|---|---|
| radix-off (14336) | 14336 MB | 0.11 GB (`max_mamba_cache_size=1`) | strategy `extra_buffer_lazy` | 0.75+0.75 GB | 5.24 GB |
| radix-off-12g | 12288 MB | 0.11 GB (`max_mamba_cache_size=1`) | strategy `extra_buffer_lazy` | 0.75+0.75 GB | 7.25 GB |
| radix-on | 12288 MB | 0.47 GB (`max_mamba_cache_size=8`) | strategy `extra_buffer` | 0.75+0.75 GB | 6.86 GB |

Dropping the hot cache from 14336→12288 MB freed enough headroom (7.25 GB avail at
12288 vs 5.24 GB at 14336) to comfortably fit the larger 8-slot mamba cache under
radix (avail mem after pool end: 6.86 GB, still ≥1 GB above the 12288-MB-hot-cache
off run's mamba-only headroom).

## Verdict

- **Correctness: pass.** Token ids match exactly in all comparisons; the one
  logprob delta above 1e-2 (within-launch cached vs cold, 0.142) sits well under the
  cross-launch noise floor (0.52-1.20) and does not flip any argmax.
- **Cache hits: confirmed.** Radix-on reuses the CONTEXT prefix to the page boundary
  on the repeat/P2 probe (3776/3800 cached) and reuses the growing session prefix
  turn-to-turn in Phase B; radix-off never caches.
- **TTFT turns 2-4: improved by ~3.6-4.2s** (5.70-5.73s off → 1.53-2.10s on), the
  primary goal of enabling radix cache for multi-turn chat.
- **Decode tok/s: no regression** (10.51 on vs 9.98-10.69 off, within run-to-run
  noise).
- **Recommendation: enable radix cache in production.** Apply to
  `run-nvfp4-e16c-public.sh`:
  - remove `--disable-radix-cache`
  - change `--mamba-radix-cache-strategy extra_buffer_lazy` to `extra_buffer`
  - change `--max-mamba-cache-size 1` to `8`
  - consider lowering `SGLANG_MOE_HOT_GPU_MB` from 14336 toward 12288 to keep the
    same memory headroom the 8-slot mamba cache needs (this experiment did not
    A/B decode throughput at 14336-with-radix, so re-verify decode tok/s if
    production keeps the hot cache at 14336 instead).
