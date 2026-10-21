# FlashInfer prefill check: steps 1 to 3 (branch `fi-prefill-check`)

Question: production runs `SGLANG_SM120_FLASHMLA_BACKEND=triton` (`benchmarks/dsv41_baseline/arm_env.py:386`, inside
`dspark_env()`, reached by `prod_env()`). The code default is `flashinfer`. Does FlashInfer 0.7.2 on `.venv-upstream`
serve the DSV4.1 prefill (> 64 tokens) and the 6-token DSpark verify correctly? Step 4 (served TTFT A/B at KV 381,184)
needs its own yes after these three are clean.

All runs: divix01, worktree `/data/models/slang/nvfp4-work/wt-fi-prefill` (detached at the pushed branch), CPU work under
`taskset -c 0-63`, GPU work under `cc-gpu.lock` on cores 32-63, `PYTHONPATH=$PWD/python`, interpreter
`/data/models/slang/.venv-upstream/bin/python`. Print `sglang.__file__` and read it.

## Before any GPU run

- `ss -ltn | grep :7867` empty (production down), `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` empty.
- Take `rowimg-disk.lock`, then `cc-gpu.lock` (steps 1 and 2 need only the GPU lock; step 3's capture takes both itself,
  so do not hold either when launching it).
- Never bypass the prelaunch floor (`PAGE_CACHE_OPT_OUT` stays 0). Do not touch `cc-expert-prediction/dsv41-direct-prod`.

## Step 1: the existing attention tests (cheap, 1 to 3 minutes)

```bash
cd /data/models/slang/nvfp4-work/wt-fi-prefill
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv-upstream/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_flash_mla_backends.py \
  test/registered/unit/kernels/test_flash_mla_sm120_extra_split.py \
  test/registered/unit/kernels/test_flash_mla_sm120_merge_chunking.py \
  test/registered/unit/test_flashinfer_sparse_mla.py -q -p no:randomly -rfE 2>&1 | tee /mnt/nvme1/fi-prefill/step1.log | tail -60; echo EXIT=${PIPESTATUS[0]}
```

The 8 failing subtests of the kernel GPU group may be elsewhere (that group is `test/registered/unit/kernels`
plus `test/manual/dsv41`); if these files are green, run the group's FlashInfer tests by `-k flashinfer` and match the
node IDs to the 8. Classify each: prefill route, decode route, or unrelated.

## Step 2: parity at the real shapes (cheap, about 5 minutes)

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv-upstream/bin/python analysis/dsv41-drive/fi-prefill/parity_prefill.py \
  --out /mnt/nvme1/fi-prefill/parity.json 2>&1 | tee /mnt/nvme1/fi-prefill/step2.log; echo EXIT=${PIPESTATUS[0]}
```

Twelve cases: decode and verify at 1, 6 and 64 tokens, prefill at 65, 2065 and 4096 tokens; SWA only, c4, c2 and c128
extra pools, sink on and off, ragged topk. Tolerances are in the script's docstring (out relL2 against the fp32
PyTorch reference, at most max(2e-2, 1.5x Triton's own error); lse within 2e-2 + 1e-3 |ref|). It also measures
median ms and peak extra allocation at 4096 tokens for Triton (merge budget 0 and 128) and FlashInfer: the VRAM
number step 4 needs. A case FlashInfer refuses prints ERROR with its message; the c128 pool (pbs 2) is the likeliest.

## Step 3: served parity on the 32k prompt (one pair of launches, about 25 to 30 minutes)

The arm is today's production arm, `PROD_ARM` of `analysis/dsv41-drive/dspark/both_cpu_ab.py`
(`...-mergechunk-expseg-prod`; confirmed it carries `SGLANG_SM120_FLASHMLA_BACKEND=triton`, merge budget 128, the
sampled confidence cut, KV pin). The override goes on the command line (`NAME=KEY=VAL`), so `arm_env.py` is not edited
and both arms differ in that one variable.

```bash
cd /data/models/slang/nvfp4-work/wt-fi-prefill
ARM=dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote-poolserve-fixedsplit-pin01-splitmodel-mem-hicache1-mergechunk-expseg-prod
OUT=/mnt/nvme1/fi-prefill/serve-$(date +%Y%m%d-%H%M)
nohup analysis/dsv41-drive/dspark/run_capture_pair.sh $OUT --arm $ARM --worktree $PWD \
  triton=SGLANG_SM120_FLASHMLA_BACKEND=triton flashinfer=SGLANG_SM120_FLASHMLA_BACKEND=flashinfer \
  -- --no-verify-split --long-prompt /mnt/nvme1/lm-lend/prompts/p32k.md --decode-seconds 90 --ttft-repeats 0 \
  > $OUT.log 2>&1 &
```

`--dry-run` was run (no lock, nothing launched): it prints each arm's env, prelaunch and capture command and
`resolved-env.json`. The prompt is `p32k.md`, 32,030 prompt tokens (the lm-lend 32k prompt, TTFT about 171 s on
production), greedy (`long_prompt.py` temperature 0), thinking on, cancelled 90 s after the first token. Triton runs
first, then FlashInfer, one server at a time. Watch `server.log` of each run dir with a background watcher.

Compare:

```bash
/data/models/slang/.venv-upstream/bin/python analysis/dsv41-drive/fi-prefill/compare_long.py \
  $OUT/triton $OUT/flashinfer --out $OUT/compare.json
```

- **Prefill (32k prompt).** Same prompt token count, same first token, and a common greedy prefix of at least 64 tokens;
  the first 600 characters of each are printed and must both read as coherent reasoning about the prompt. A flip later
  in the stream is expected (bf16 near-ties) and is reported as the first divergence token, not as a failure.
  A prefill defect in FlashInfer (wrong rows, a stale page split) shows as a first-token mismatch or garbage text.
- **The 6-token verify (<= 64 tokens, FlashInfer decode).** Every decode step of the FlashInfer arm runs it, so a
  coherent 90 s stream with a long common prefix is the check. Its accepted-length evidence: compare the two runs'
  `counters.json` verify and accepted counts (the sampled cut changes both arms the same way); a markedly lower
  acceptance on FlashInfer means a wrong verify. Confirm the arm really ran FlashInfer: `resolved-env.json` has
  `SGLANG_SM120_FLASHMLA_BACKEND` flashinfer for that arm only, and `/proc/<pid>/environ` of the live server agrees
  (run_arm-style verification). This is inferred from the environment: no kernel counter says which backend served a step.
- **Memory.** `<arm>.gpu-mem.csv` peak per arm gives the prefill VRAM difference on the real server (startup sets the
  peak; DSV_OPTIMIZATION.md 3.7). The hot cache is calibrated at startup from measured free memory, so the two arms
  may size it differently: read `SGLANG_MOE_HOT_GPU_MB` outcomes from each `server.log` before comparing.

TTFT from this pair is one sample per arm and is not the step 4 A/B (no repeats, no pairing, KV not yet checked
against 381,184: read the KV pool size of each `server.log` and stop if it differs from 381,184).

## What this does not settle

- Whether the explicit DSV4.1 modes in FlashInfer 0.7.2 (`cake_dsv4*`, FP8 prefill, full-BF16 prefill) beat the plain
  `_sparse_mla_sm120_paged_attention` path sglang calls. That is adapter work for after this check.
- Throughput and TTFT: step 4.
