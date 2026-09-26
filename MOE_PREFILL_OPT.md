# Task: cut DSV4.1 prefill TTFT by removing the host/GPU ping-pong in the eager streamed-MoE prefill path

You are continuing performance work on DeepSeek V4.1 Flash (EXL3 3.0 bpw) served by our SGLang fork on one RTX 5090 (SM120,
32 GB) on the host `divix01`. The experts do not fit in VRAM: they stream from a VRAM hot cache (1,128 rows), a 100 GiB
pinned host-RAM tier, and two NVMe mirrors. Batch size is 1.

Repo: `git@github.com:divixcorp12/sglang.git`, branch `master`. Read these first:
- `CLAUDE.md` (the Nsight rules, especially: bound trace-analysis memory, `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp`, never
  graph-mode nsys with the copy engine on).
- `.claude/rules/divix01-run-protocol.md`: laptop -> commit -> push -> private divix01 worktree; `PYTHONPATH=$PWD/python`
  and print `sglang.__file__`; read `PIPESTATUS`; CPU jobs under `taskset -c 0-63`; GPU work under `cc-gpu.lock` on
  cores 32-63; lock order `rowimg-disk.lock` then `cc-gpu.lock`; mutants only in private worktrees.
- `.claude/rules/` generally (env-var conventions, `msgspec.Struct` not dataclasses, comment style, no defensive getattr).
- `DSV41_REFERENCE.md` sections 27.2, 27.4 (item 7), 27.6, 27.9, 27.10. Section 27.10 is the measurement this task
  starts from.

## Where things stand (all measured, section 27.10)

The production recipe (`benchmarks/dsv41_baseline/arm_env.py`, `base_env()`, commit `003d82fb77`) now has
`SGLANG_DSV41_ENABLE_PREFILL_FILLS=1` and `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION=1`. Prefill fills (section 27.6) read
pinned-tier misses on a helper thread and issue a layer's reads up front; TTFT went from ~21/18 s to ~12/11 s.

Node-mode trace of the last timed session's prefill (260 prompt tokens), flags off vs on:

| | Flags off | Both on |
|---|---:|---:|
| Prefill wall | 19,454 ms | 12,882 ms |
| GPU busy (kernels + copies) | 6,588 ms | 6,571 ms |
| of which `_gather_host_rows_kernel` (726 launches) | 6,083 ms | 6,068 ms |
| GPU idle | 66% | 49% |
| D2H readbacks <=256 B / host time blocked in them | 8,380 / 6,114 ms | 8,174 / 6,102 ms |
| Eager kernels | ~142,000 | ~141,600 |
| PCIe RX mean | 13.7% | 20.6% (~5.9 GB/s, ~75 GB total) |

What it means:
- **The gather is capped by the link.** ~75 GB of expert rows (~5,600 rows x 13.3 MB; ~140 distinct experts per layer
  x 40 layers) cross PCIe Gen3 x16 at ~12.3 GB/s in ~6.1 s. That is the floor unless fewer rows are gathered.
- **The other ~6.3 s is the GPU idle, waiting for the host.** Split by NVMe activity in 100 ms bins
  (`analysis/dsv41-drive/pcie-trace/prefill_idle_vs_nvme.py`):
  - 1.4 s with a mirror >=50% busy (fill waits);
  - 2.7 s with 10-50% busy;
  - 2.2 s with the mirrors idle, i.e. pure host overhead.
- **The host is blocked ~6.1 s in ~8,200 tiny readbacks, and the GPU is idle ~6.3 s waiting for the host.** They take
  turns; together that is nearly the whole prefill.
- **The NVMe mirrors are not the limit in prefill:** each reads at link rate (~3.8 GB/s) while busy, but is busy only
  ~23% of the prefill (`analysis/dsv41-drive/nvme-load/`).
- **Target:** overlap the host work and fills with the gather, and take TTFT for a 260-token prompt from ~12.9 s toward
  the ~6-7 s link floor.

## The code path (eager prefill, per MoE layer)

- `python/sglang/srt/layers/quantization/exl3.py:528` `_apply_streamed`:
  1. `torch.unique(routed)`;
  2. `streamer.prefill_fills(source_ids)`;
  3. for each chunk from `streamer.iter_gather_experts(source_ids)`: `chunk.tolist()`, `row_of_source.tolist()`, then
     `exl3_moe_accumulate(...)`.
- `python/sglang/srt/layers/quantization/exl3_ops.py:196` `exl3_moe_accumulate`: a Python loop over experts. Each one
  does `torch.where(topk_ids == expert)`, which is a host sync (nonzero), plus about 6 small kernels: gate/up/down
  `exl3_linear`, clamp, silu, `index_add_`. With ~140 experts per layer this is probably most of both the ~5,600 syncs
  and the ~141,600 kernels. **Verify this attribution before building on it** (e.g. count nonzero/`aten::where`
  launches and D2H copies per layer in the trace, or add NVTX ranges).
- `python/sglang/srt/layers/moe/expert_stream.py`:
  - `:1832` `iter_gather_experts`: validates with `.any().item()` and `torch.unique` once, then gathers chunks of at
    most `EXL3_MAX_GATHER_ROWS = 64` experts (`exl3_expert_format.py:43`). Chunks share one staging set (~852 MB), so
    chunk k+1's gather waits for chunk k's consumers.
  - `:484` `gather_rows` (pinned tier): per chunk, `hit_mask.sum().item()` and `chunk.tolist()` (both syncs), then
    `ensure_rows`, `_await_fills`, `copy_rows`. With fills on, `_await_fills` waits for that chunk's rows only.
  - `:1643` `prefill_fills`: one `host_use()` per layer (one stream sync, one pause of the RAM-miss service thread),
    `torch.stack(...).tolist()` of the hot slots, `cache.prefetch_rows(...)`, then `finish_fills()` at the end.
  - `:1744`, `:1793`, `:1855`: further `.any().item()` range checks.
- The C++ side of fills is `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`
  (`RamTier::fill_begin` / `fill_wait` / `fill_end`; the `filling` slot flag; the prefill share of section 27.9 sits in
  `take_admit_slot_locked`).

## What to do

1. **Attribute first.** From the existing trace, measure how the ~8,200 syncs and ~141,600 kernels split across call
   sites: `torch.where` per expert, chunk `.item()` / `.tolist()`, range checks, and anything else. Write it down
   before changing code.
   - Traces on divix01: `/mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044.{nsys-rep,sqlite}` (both flags on) and
     `pevict-A-20260925-194303.{nsys-rep,sqlite}` (both off).
   - Analyse on divix01 under `taskset -c 0-31` with bounded memory.
   - `analysis/dsv41-drive/pcie-trace/compare_arms.py` finds the last session's prefill and decode windows.
2. **Remove the per-expert syncs** with a host-side plan built once per layer:
   - read `topk_ids` / `topk_weights` to the host in one D2H (260 x 6 values), or better, keep the whole plan on device;
   - compute every expert's token list and offsets once;
   - launch without further syncs.
   The accumulation order must stay ascending expert id in fp32, or outputs change (see the `exl3_moe_accumulate`
   docstring and `exl3_moe_loop`).
3. **Consider grouping the expert compute** so one launch covers several experts. A prefill-side grouped EXL3 gemm, or
   batching the existing `exl3_gemm_kernel` calls, would cut the ~112k small launches. Check what `exl3_ops.py` and
   `python/sglang/kernels/jit/csrc/moe/` already provide before writing a kernel.
4. **Let the host run ahead of the gather.** Once the syncs are gone, the host should issue chunk k+1's
   `ensure_rows`/fills and launches while chunk k's gather runs. Keep the staging-reuse contract (`iter_gather_experts`
   docstring): chunk k+1's gather must not start until chunk k's consumers are done. Stream order provides that
   without a host sync.
5. **Optional, only if steps 2-4 leave the gather exposed:** double-buffer the staging so the gather of chunk k+1
   overlaps chunk k's compute. That needs another ~852 MB of VRAM, which the 0.83 memory fraction does not have today.
   The on-hold prefill indexer cap (section 27.7, branch `cc/indexer-cap`) would free an estimated ~2.5 GB on long
   prompts. Coordinate before depending on it.

## Rules for the change

- **Behind a new default-off env var** in `python/sglang/srt/environ.py` (typed `EnvField`, `SGLANG_DSV41_*` name),
  with its field in `Dsv41Config` (`test_one_field_per_knob` must pass). Flag off must be exactly today's path.
- **Output must be byte-identical.** Greedy output flag off vs on, and bitwise parity tests of the MoE output against
  the current `exl3_moe_accumulate` path, including a case with more than 64 experts (several chunks), repeated
  experts per token, and dropped routes (`-1`).
- **Tests:** CPU tests where possible, GPU parity tests, mutants that the tests must catch, and the registered suite
  `test/registered/unit/kernels` compared against the merge base. Record every command next to its counts.
- **Arms:** `benchmarks/dsv41_baseline/run_arm.sh`, A (flag off) then B (flag on), once each, no ABBA, one port
  (30021). Hold `rowimg-disk.lock` and poll `cc-gpu.lock`: see `analysis/dsv41-drive/nvme-load/drive_traced_arm.sh`
  for a driver. Report TTFT for both sessions, pooled ms/token, and output identity.
- **Traces:** also run one node-mode traced arm of B (`NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node`; graph mode is
  refused with the copy engine on) and re-run `compare_arms.py` against `prod-flags-node` to show the change in GPU idle
  and readbacks.
- **Production (port 7867) is stopped and must stay stopped.** Never test in `cc-expert-prediction/dsv41-direct-prod`
  or `dsv41-direct-live`. Scratch goes on `/mnt/nvme1`, never `/` or `/tmp` on divix01.
- **Git:** never force-push; stage files by name; never commit `.omc/`. Other lanes may push to `master`, so fetch and
  rebase before pushing.

## Deliverables

- Code on `master` behind the default-off flag (or on a branch `cc/prefill-syncs` if the arm is negative).
- A new `DSV41_REFERENCE.md` subsection 27.x with:
  - the attribution;
  - what changed;
  - parity and test results, with commands;
  - the A/B arm table;
  - the before/after trace comparison.
- Mark section 27.4 item 7 accordingly.
- A final report: the flag name, commits, the TTFT change, the remaining prefill breakdown (link floor vs host vs
  NVMe), and anything unresolved.
