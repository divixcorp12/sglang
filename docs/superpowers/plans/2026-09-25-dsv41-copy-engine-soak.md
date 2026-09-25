# DSV4.1 copy-engine soak

Question: does the RAM-miss copy engine's known hang (LEASE_PROTOCOL.md 7.6, "Module loading"; copy-compute-overlap
plan section 10) fire under varied real traffic, and if so, why and how do we fix it. Branch `cc/ce-soak` from
`7bbc04e16e`. Evidence: `divix01:/mnt/nvme1/ce-soak/<run>/`.

## Handoff (mid-task)

Status: the hang fires, deterministically, and is fixed on the branch by requiring `CUDA_MODULE_LOADING=EAGER`.
The full ~2 h soak with the fix has **not** been run. Nothing is running; the GPU and both locks are free; the divix01
worktrees (`wt-ce-soak`, `wt-ce-soak-diag`) are removed; no server is up.

### Soak design

- Launcher `analysis/dsv41-drive/ce-soak/soak.sh <tag> <worktree> <seed>`: the `arm_env` recipe (copy engine on) on
  port 30021, stage-traced, holding `cc-gpu.lock` and `rowimg-disk.lock` for the whole run. One greedy priming request
  arms the copy engine; the script refuses to continue without the "copy engine armed" log line and a stage trace
  showing copy lanes with 0 fallbacks and 0 errors. A watcher captures py-spy, eu-stack, nvidia-smi and the log tail
  (`failstop/`) the moment the log reports a failed request.
- Driver `soak_driver.py`: a seeded plan (seed **20260925**, 320 requests, 265 items) of 20 kinds: greedy; sampled
  (temperature 0.3-1.2 with top_p/top_k/min_p); penalties; logit_bias; logprobs/top_logprobs; OpenAI `seed`; stop
  strings and stop_token_ids; max_tokens 1; json_schema; regex; EBNF; tools-style prompts (half with a `tools` field);
  system prompts; multi-turn conversations; long prompts to 28k tokens; bursts of 3-5 concurrent requests; client
  aborts mid-stream; n=2; `/v1/completions` with logprobs/echo; native `/generate` with input logprobs and
  json_schema. A greedy determinism control every 20 requests. 70% streamed. Records per request in
  `requests.jsonl`; a stalled stream (gap > 0.7 s) triggers a capture into `captures/`. `--start-item` and
  `--skip-items` select plan items; `SOAK_ENV_OVERRIDES`, `SOAK_EXTRA_ARGS`, `SOAK_CAPTURE_CUDA_GDB=1` are for diagnosis.
- Report `soak_report.py <out>`: survival, fail-stops, per-kind status, client gaps >= 0.5 s / 2 s, stage-trace
  multi-row demands > 10 ms, copy counters, grammar validity, determinism, ms/token per 10 minutes.

### Runs (all seed 20260925; "2 s" is the production RAM-miss deadline)

| Run | Purpose | Outcome |
|---|---|---|
| s1 | full soak, head `f160438589` | server died at request 13 (`/generate`, `logprob_start_len=0`): `ValueError` in `_check_late_layer_tail_readers` under decoder SWA bounded replay. Not the copy engine. Fixed (below). |
| s2 | full soak with the refusal | the refused request then crashed the TokenizerManager (`len(None)` in `convert_logprob_style`). Fixed (below). |
| s3 | full soak, both fixes | **copy-engine fail-stop**: RAM-miss request timed out on item 15 (sampled, min_p, 3,314-token prompt) at its 14th decode step, 7.5 min after arming. |
| d1-t60 | s3 with a 60 s deadline, stall capture + cuda-gdb | same item, same step. Scheduler in `cudaEventSynchronize`; copy thread polling `cuEventQuery`, not blocked; no thread in the driver loader. cuda-gdb hung. The later Engram assert is a consequence (its 10 s wait expired during the 60 s one). |
| d2-item15 | `--start-item 15`, 6 min | passed item 15 (and 12 more). |
| d3-probe | d1 on `cc/ce-soak-diag` (copy stall probe) | at 1 s stuck: fresh 4 KiB H2D copies on a new greatest- and a least-priority stream and a D2D copy **all** stayed incomplete for 500 ms, as did the job. No copy on the device progressed. |
| d4-item15-long | `--start-item 15`, 15 min | passed; rules out a time-based trigger. |
| e1-skip12 | skip item 12 | failed at item 15. |
| e2-skip1to7 | keep 0, 8-11, 13, 14 | failed at item 15. |
| e3-keep0-13-14 | keep 0, 13, 14 | passed. |
| e4-keep0-8-9 | keep 0, 8, 9, 13, 14 | passed. |
| e5-keep0-10-11 | keep 0, 10, 11, 13, 14 | failed at item 15. |
| e6-keep0-11 | keep 0, 11 | failed at item 15 (minimal: items 0, 11, 15). |
| e7-11-15-gdb | items 11, 15 only, 60 s deadline, cuda-gdb first | failed; cuda-gdb attach timed out after 120 s. |
| e8-eager | e6 with `CUDA_MODULE_LOADING=EAGER`, `--mem-fraction-static 0.83` | **passed** item 15 and 10 more requests. |
| e9-lazy083 | e6 with `--mem-fraction-static 0.83` only (control for e8) | **failed** at item 15, step 14. |

After e8 (coordinator's stop point) three GPU jobs ran, none a server: the hazard probes (below), the lazy-load
scenario, and the GPU test file.

### Hypothesis, and the evidence

A kernel loaded lazily (`CUDA_MODULE_LOADING=LAZY`, torch's default) for the first time while an armed decode step
waits in CW stops the copy thread's copies until the deadline.

- For: e8 (EAGER) passes the exact sequence e9 (LAZY, same memory fraction) fails. A reduced GPU scenario
  (`test/manual/dsv41/ce_lazy_load_scenario.py`: a built-but-never-launched JIT kernel launched for the first time
  while the chain's copies are issued) fails stop under LAZY (launch blocks 3,000 ms, timeout, fatal) and completes
  under EAGER (launch 0.13 ms), 2 of 2 each.
- Against / open: in the server no thread was seen inside the loader at the capture (0.7-1 s into the stall), the copy
  thread was not blocked on the driver lock, and copies issued after the stall began (the probe's) did not run either;
  the scenario's mechanism (loader holds the lock, copy thread blocked) is therefore not proven to be the server's.
  The kernel that loads is not identified: something first launched on item 15's 14th decode step (context
  3,314 + 14 = 3,328 = 13 KV pages of 256) and only when item 11 ran before it.

### Hazards ruled out (hazard probe `ce-soak/hazard_probe.py`, the real chain with a 20 ms ballast, 3 s deadline)

No deadlock, each 2 of 2, for a call made while CW spins: `cudaHostAlloc` via torch pinned alloc (delays the copy by
the call's 139-541 ms, no deadlock), `cudaHostRegister`, `cudaFreeHost` (host cache emptied), `cudaMalloc`,
`torch.cuda.empty_cache`, raw `cudaStreamCreateWithFlags`, `cudaEventCreate`, and H2D, D2H and D2D copies queued
behind the chain on its stream or on a pre-made side stream. One did deadlock: the first `torch.cuda.Stream()` of a
process (torch's pool initialisation) while CW spins; the server's pool is initialised long before arming. Also ruled
out: time since launch (d4), the refused request (e1), and hardware faults (no Xid besides the assert's 43).

### Reproduction

On divix01, in a worktree of `shared/cc/ce-soak` at `41966cd6f2` or later but **with `arm_env.py` reverted to LAZY**
(the current head sets EAGER and the service refuses the copy engine without it; override with
`SOAK_ENV_OVERRIDES="'CUDA_MODULE_LOADING': 'LAZY'"`, which the head's service will refuse, so use `41966cd6f2`):

    SOAK_MINUTES=6 SOAK_DRIVER_ARGS="--skip-items 1,2,3,4,5,6,7,8,9,10,12,13,14" \
      bash analysis/dsv41-drive/ce-soak/soak.sh <tag> $PWD 20260925

Item 11: tools prompt (525 tokens, `tools` field), streamed, temperature 1.037, min_p 0.062, max_tokens 128.
Item 15: 3,314-token notes prompt, streamed, temperature 0.781, min_p 0.035, max_tokens 38. The fail-stop comes
~2.5 min after arming. EAGER: add `SOAK_ENV_OVERRIDES="'CUDA_MODULE_LOADING': 'EAGER'"
SOAK_EXTRA_ARGS="--mem-fraction-static 0.83"`.

### Captures

`/mnt/nvme1/ce-soak/<run>/captures/*-stall.txt` (py-spy --native, eu-stack, nvidia-smi at the stall) and
`/<run>/failstop/` (the same after the log's fail-stop line) for s3, d1-t60, d3-probe, e1, e2, e5, e6, e7, e9. The
probe line is in `d3-probe/server.log` ("exl3 copy probe"). Hazard probe JSON: `/mnt/nvme1/ce-soak/probe/`.

### Fixed on the branch

- `fix(scheduler)` `c2f69ce864`, `83b1023a51`: a request for prompt-token logprobs under
  `--enable-decoder-swa-bounded-replay` is refused at admission (`managers/prompt_logprobs.py`) instead of raising
  inside the forward and killing the server.
- `fix(tokenizer_manager)` `d46a5f8e91` (test `1d9a7a2eab`): `convert_logprob_style` tolerates a batch output without logprob lists.
- `fix(dsv41)` `5108b7b90d`: the service refuses the copy engine unless `CUDA_MODULE_LOADING=EAGER`; the module-load
  guard also wraps `tvm_ffi.load_module` (under EAGER a library's kernels load at dlopen); `arm_env` sets EAGER and
  `MEM_FRACTION_STATIC = 0.83` (KV 204,288 tokens vs 209,408 at 0.80 LAZY).
- Tests: CPU `test_prompt_logprob_refusal.py`, `test_logprob_style_without_logprobs.py` (both mutants caught), two new
  tests in `test_exl3_ram_miss_service.py`; GPU `test_a_kernels_first_launch_while_cw_spins_fails_stop_under_lazy_
  loading_only[LAZY|EAGER]` in `test/manual/dsv41/test_exl3_copy_engine_cuda.py` (whole file: 13 passed).
- CPU suite at `7c0c9c1bc2`: **1408 passed, 414 skipped, 0 failed** —
  `CUDA_VISIBLE_DEVICES= PYTHONPATH=$WT/python OMP_NUM_THREADS=8 taskset -c 0-31 python -m pytest
  test/registered/unit/kernels test/registered/unit/layers/moe/test_exl3_ram_miss_{service,shutdown,tables}.py
  test/registered/unit/layers/moe/test_exl3_stream_trace.py test/registered/unit/layers/moe/test_exl3_native_prefetch.py
  test/registered/unit/test_dsv41_config.py test/registered/unit/managers/test_prompt_logprob_refusal.py
  test/registered/unit/managers/test_logprob_style_without_logprobs.py -q -p no:randomly -rfE --basetemp=...`
  (log `divix01:/mnt/nvme1/ce-soak/cpu-head.log`). Not yet run: the same at the merge-base, mutants of the EAGER
  refusal and the tvm-ffi guard.

### Next steps

1. Run the full soak at the head (`soak.sh s4 <wt> 20260925`, ~115 min) and write the results and the verdict here;
   then a second seed. Watch ms/token against the final-arms 110.8 (EAGER should not change it).
2. Mutants in a private worktree: drop `check_copy_engine_module_loading` from `ensure_started`, drop the tvm-ffi
   wrap; confirm the CPU tests fail; restore and re-run green. Run the CPU suite at `7bbc04e16e` for the comparison.
3. Optional, for the write-up: identify the lazily loaded kernel (a node-mode nsys capture of the e6 reproduction under
   LAZY, `--delay` to just before item 15, CUDA API trace for `cuModuleGetFunction`/`cuLibraryGetKernel`), and update
   LEASE_PROTOCOL.md 7.6 "Module loading" (EAGER is now required; the not-guarded list shrinks to libraries loaded
   after arming outside Triton and tvm-ffi).

Branch head at handoff: see the commit that adds this section (`git log -1 shared/cc/ce-soak`). `cc/ce-soak-diag`
(`82a84e496a`, the copy stall probe) is a diagnosis branch; do not merge it.
