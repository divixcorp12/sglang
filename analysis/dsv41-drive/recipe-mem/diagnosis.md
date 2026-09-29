# Recipe GPU memory: why 0.90 stopped fitting, and why 0.91 OOMs a 16k prompt

2026-09-29, divix01. This is an investigation only; nothing in the recipe is changed here. `MEM_FRACTION_STATIC = 0.91`
on this branch is the change under review, and it is **not** safe (see Q3).

## Summary

| Question | Cause | Evidence |
|---|---|---|
| Q1: KV headroom drift | **Environment, not code.** The NVIDIA driver upgrade from 610.57.04 to 615.71.09 (dnf transaction 477, 2026-09-27 23:11, live after the 09-28 00:30 reboot) made the pre-load CUDA footprint ~0.5 GiB larger and jittery. | 09-27 commit and master measure the same today (table 2). The step lines up with the upgrade to the hour (table 1). |
| Q2: 25 late Triton loads | **Not new, and not the OOM.** Triton loads each specialization at first launch, and engine init never runs an eager prefill. The 1 GiB watcher threshold has only now been crossed. | The same kernels loaded late on 09-27. Free memory is flat across 12+ consecutive loads. Loads are guarded for the copy engine. |
| Q3: 16k at 0.90 before | **Yes, but with ~16 MiB to spare.** 0.91 on the new driver leaves 0.54 GiB less activation headroom than 0.90 on the old one. | §27.17/27.18: 16k chunked peak 32,134 MiB. §27.19: layer-major 16k/32k ran with 0 OOMs. |

## Q1: where the ~0.5 GiB went

**Table 1.** Every recipe launch in `dsv41-baseline/servers/*/run-2026092[6-9]*`, "Load weight begin" is free GPU
memory before weights, and "available" is KV `available_bytes`:

| window | driver | commits | Load weight begin (GB) | available at 0.90 (GB) | free after decode graph capture at 0.90 (GB) |
|---|---|---|---|---|---|
| 09-26 00:14 → 09-27 22:57, 40+ launches | 610.57.04 | many, through `2b5a183e1a` | **29.90–29.91** (no jitter) | 0.78–0.83 | 2.46–2.47 |
| 09-28 01:27 → 09-29, 50+ launches | 615.71.09 | `a2f0ae97b0` → master | **29.21–29.45** | 0.14–0.38 | 2.33–2.37 |

- **The step.** It falls between `svccpu-evensplit-notprod-A6` (09-27 22:57) and `lpdl-A` (09-28 01:27). Two things
  changed in that window:
  - 12 commits (`2b5a183e1a..a2f0ae97b0`: lease and row-copy kernels, `exl3_ram_miss.py`, env);
  - dnf transaction 477, which moved the driver and libcuda to 615.71.09, the kernel to 211.60.1 and system cuDNN to
    9.26, followed by the 00:30 reboot.
- **The weights did not grow.** "Weights +0.04 GB" is jitter: `mem usage` is 9.96–10.03 GB on both sides of the step.

**Table 2. Bisect, same driver today** (`launch_to_pool.sh`: each launch stops at decode graph capture and sends no
requests; interleaved; `/mnt/nvme1/recipe-mem/pool/`):

| launch | commit | Load weight begin | available (0.90) | KV tokens | Memory pool end | after capture |
|---|---|---|---|---|---|---|
| old-1 | `2b5a183e1a` (09-27, gave 29.91 / 0.80 then) | 29.36 | 0.30 | 42,752 | 3.17 | 2.35 |
| master-1 | `65754399e3` | 29.39 | 0.31 | 46,848 | 3.19 | 2.37 |
| old-2 | `2b5a183e1a` | 29.44 | 0.38 | 93,952 | 3.18 | 2.33 |
| master-2 | `65754399e3` | 29.45 | 0.37 | 92,160 | 3.19 | 2.34 |

The 09-27 code now behaves like master, so the code is cleared.

**Which part of the environment.**
- **Not cuDNN.** Torch loads the venv's own cuDNN (`site-packages/nvidia/cudnn/lib/libcudnn.so.9`, version 9.20),
  not the system 9.26. The venv was unchanged after 09-24.
- **libcuda is the one GPU-stack component that changed.** The process maps `/usr/lib64/libcuda.so.615.71.09`.
- **Bare-context probe** (`torch.zeros(1, device="cuda")`, then `mem_get_info`, no sglang):

| `CUDA_MODULE_LOADING` | used MiB (repeats) |
|---|---|
| EAGER (the recipe's) | 1961, 1973, 1903 |
| LAZY | 697, 697 |

- **What the probe shows.** Under 615.71.09 EAGER module loading alone takes ~1.9–2.0 GiB, and it jitters by ~70 MiB
  from process to process. LAZY does not jitter. That accounts for the server's pre-load footprint: 31.45 − 29.2..29.45
  = 2.0–2.24 GiB, against a steady 1.54 GiB on 610.57.04, where the arm_env comment put EAGER's cost at "~1 GiB".
- **Limit of this evidence.** The driver cannot be rolled back without root, so the old driver's bare-probe number is
  inferred from the server logs, not measured.

**Why a fraction change moves both budgets.**
- The KV budget is `free_at_profile − begin × (1 − fraction) − 0.16 GB` (`kv_cache_configurator.py:2209`).
- A larger pre-load footprint lowers `free_at_profile` one for one and barely moves the slack. At 0.90 the whole
  ~0.5 GiB came out of the KV pool: 0.80 → 0.27 GB against a 0.23 GB SWA floor.
- Raising the fraction to 0.91 restores the KV pool (0.52–0.66 GB), but it takes `begin × 0.01` ≈ 0.3 GB from
  activation headroom, and that is Q3's OOM.

## Q2: the late Triton loads

- **Where they come from.** `triton_load_watch` (since 08-07) is armed in `Scheduler.run_event_loop`
  (`scheduler.py:1953`). It warns only when a load finds < 1 GiB free (`SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB`).
  Triton loads each specialization's cubin at its first launch.
- **Why they load late.** Engine init captures only the BS1 decode graph and never runs an eager prefill or an eager
  scheduler step, so every prefill-path kernel first launches on the first request.
- **What loads.** The same list appears in the decode arm (22) and in the 16k run (25). The name is logged; the
  specialization key is not (the hook receives `hash` but does not print it). A name listed twice is two
  specializations.
  - First request (warm-up), prefill and eager path: `_gather_host_rows_to_kernel`, `_scatter_hot_rows_kernel`,
    `_rmsnorm_fp32_kernel`, `_rope_tail_fake_quant_fp4_kernel`, `_quantize_fp4_indexer_kernel`,
    `_store_fp4_index_k_cache_kernel`, `_gather_host_rows_kernel`, `_hc_mix_stats_partial_kernel`,
    `_hc_mix_reduce_sinkhorn_kernel`, `_router_triton_kernel`, `alloc_decode_kernel`,
    `get_and_clear_swa_pages_kernel`, `_expand_prefill_causally_kernel`, `_engram_hash_kernel`,
    `_page_table_positions_kernel`.
  - Only in the 16k layer-major prompt (long-sequence specializations): `_hc_combine_norm_prefill`, a second
    `_hc_mix_stats_partial_kernel` and a second `_page_table_positions_kernel`.
- **They are not new.** On the old driver at 0.90 (`/mnt/nvme1/layer-major/equiv-t12fix/layer-major-8k`, 09-27), the
  same long-prompt kernels warned at 0.37–0.69 GiB free. The first-request loads happened too, but above 1 GiB, so
  nothing was printed: 2.47 GB was free after capture then, against 1.92 now at 0.91.
- **Memory cost: below measurement, not the OOM.** In the decode arm, free is 0.89 GiB at 12 consecutive loads
  (03:49:07–03:49:10), and the watcher's resolution is 0.01 GiB, so each load is ≪ 10 MiB. The drop from 1.92 GiB
  after capture to ~0.9 GiB is the caching allocator's growth in the first eager forward. In the 16k run the fall to
  0.56 and then 0.06 GiB is layer-major activations (the failing allocation was a 256 MiB tensor).
- **Copy engine (LEASE_PROTOCOL 7.6).** Triton loads are guarded.
  - `_copy_engine_module_load_guard` wraps Triton's `load_binary` (and `tvm_ffi.load_module`)
    (`exl3_ram_miss.py:1113-1146`). Once the copy engine is armed, a load drains the device first, so it is not the
    fail-stop case.
  - `CUDA_MODULE_LOADING=EAGER` covers runtime-API modules only. They are 7.6's unguarded case, and EAGER is what
    closes it, which is why it cannot simply be switched to LAZY to recover the memory.
  - In the decode arm all loads but one came before the copy engine armed (03:49:24). The late one,
    `get_and_clear_swa_pages_kernel` at 03:49:39, ran under the guard.
- **Where a startup warm-up could compile them (not implemented).** Run one short eager prefill (and one eager
  scheduler decode step) before `triton_load_watch.mark_serving_started()` (`scheduler.py:1953`). Better still, run it
  before KV sizing (`KVCacheConfigurator._profile_available_bytes`, `kv_cache_configurator.py:2209`), so the cubins
  and the allocator's high-water mark are already in `free_at_profile`. A 16k-class shape would also pre-load the
  long-sequence specializations, but such a pass costs a 16k prefill at startup.

## Q3: did 16k fit at 0.90 with the 09-28-era headroom?

On the old driver, yes, but only just. Nothing has run a 16k prompt at 0.90 on the new driver.

| run (`/mnt/nvme1/prefill-chunk`, `layer-major`) | date | driver | fraction | path | free after capture | result | peak memory.used |
|---|---|---|---|---|---|---|---|
| `c4096-16k-h16100-m090` | 09-26 | 610 | 0.90 | chunked 4096 | 2.46 GB | TTFT 106.7 s, 0 retries | 32,134 MiB (**~16 MiB real margin**, §27.18: CUDA can use 32,150) |
| `oom0b-fix-16k` / `oom0b-fix-128k` | 09-27 | 610 | 0.90 | chunked 4096 | — | 106.3 s / 931.5 s, 0 retries | 32,114 / 32,136 MiB |
| `equiv-t12fix/layer-major-8k` (§27.19) | 09-27 | 610 | 0.90 | layer-major, 16k and 32k | 2.47 GB | 0 OOM retries | — |
| `c4096-16k-h16100` | 09-26 | 610 | 0.925 | chunked 4096 | — | **OOM** (3 retries) | 32,126 MiB |
| `mem091-lm16k` | 09-29 | 615 | 0.91 | layer-major | **1.92 GB** | **OOM**, 256 MiB alloc at 138 MiB free | 32,137 MiB |

- **16k is headroom-limited.** A 16k prefill fills the card whatever the fraction. At 0.90 on the old driver it fit
  with 2.46 GB of post-capture headroom and ~16 MiB to spare.
- **0.91 is not enough.** At 0.91 on the new driver there is 0.54 GB less headroom, so 0.91 is not a safe recipe.
- **0.90 on the new driver is probably not safe either.** It leaves 2.33–2.37 GB after capture, ~0.1 GB below the
  margin 16k needed. That is inferred, not run.

## Recommended fix (not implemented)

The driver took ~0.5 GiB that the recipe had allotted to the KV pool. That memory can only come back from the hot
cache, because the fraction trades KV against activations one for one:

1. **The change.** Set `MEM_FRACTION_STATIC = 0.895` and `SGLANG_MOE_HOT_GPU_MB` to about **15400** (−700 MiB) on the
   new driver.
   - Activation headroom returns to about the old 0.90 level: +0.147 GB, so ≈ 2.48 GB after capture against 2.46 then.
   - KV `available_bytes` comes back to ≈ 0.27 − 0.15 + 0.68 ≈ 0.8 GB, the old level. That clears the 0.23 GB floor
     by more than the ±0.12 GB launch jitter.
   - Cost: §27.7's slope puts the smaller hot cache at ~1.5–2 ms/token of decode.
   - Before merging, verify it with this branch's `run_checks.sh` (decode, 16k and 64k at the recipe and forced
     chunked).
2. **Separately, as a robustness item.** Add a startup eager-prefill warm-up before KV sizing (Q2), so the profile
   sees what serving will use. The ~16 MiB margin at 16k shows the recipe is sized with no slack for activations.
3. **Keep `CUDA_MODULE_LOADING=EAGER`.** LAZY would recover ~1.3 GiB, but it reopens 7.6's unguarded fail-stop.

## Reproduce

```bash
# divix01; worktrees of cc/recipe-mem-091 (script), 2b5a183e1a and 65754399e3
bash analysis/dsv41-drive/recipe-mem/launch_to_pool.sh <worktree> <tag>
CUDA_MODULE_LOADING=EAGER python -c 'import torch; torch.zeros(1, device="cuda"); f, t = torch.cuda.mem_get_info(); print((t - f) >> 20)'
dnf history info 477   # driver 610.57.04 -> 615.71.09, 2026-09-27 23:11
```
