# Handoff: DSV4.1 serves DSpark with both CPU-expert clients (2026-10-07)

You are picking up DSV4.1 (EXL3, RTX 5090 on divix01) right after DSpark speculative decoding went to production with
target CPU experts and draft CPU experts on together. Everything below is merged and running. Your job is the
follow-ups in "What is left". Read "How to work here" before touching divix01.

## State right now

- **master** = `473691871d` "Merge branch 'dsv41-dspark-both-cpu'", pushed to `origin`. `dsv41-dspark-both-cpu`,
  `dsv41-cpu-plan-m2` and `numa-node-distributor` have no commits off master.
- **Production is running** in DSpark mode, started 2026-10-07 00:13 on the owner's go:
  - checkout `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod` @ `473691871d`;
  - launched by `benchmarks/dsv41_baseline/launch_prod.sh` (port 7867), which holds `cc-gpu.lock` for its lifetime;
  - log `cc-expert-prediction/servers/prod-dspark-both-20261007-001351/server.log`, symlinked from
    `/data/models/slang/nvfp4-work/server.log`.
  - Its start: `/health` 200 at ~10 min (cold), KV pool 409,088 tokens, 5.23 GB headroom after init, both graphs
    captured, prefill warm-up 11/11, copy engine armed after 16 decode forwards, no Traceback/OOM/CUDA error.
- **Production's recipe**: `arm_env.PROD_DSPARK = True`, so `prod_env() = arm_env(dspark_env())`:
  - `SGLANG_DSV41_CPU_EXPERTS=1`, `SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1`;
  - `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES=8`;
  - `SGLANG_SM120_FLASHMLA_BACKEND=triton`;
  - budget A: `SGLANG_MOE_HOT_GPU_MB=10840`, `--mem-fraction-static 0.78`;
  - `--speculative-algorithm DSPARK`, draft `dsv41-dspark-draft`, block size 5.

  `DRY_RUN=1 launch_prod.sh` prints all of it and starts nothing.
- To go back to the non-DSpark recipe: set `PROD_DSPARK = False` (tests in `benchmarks/dsv41_baseline/` pin both
  modes), merge, pull the prod checkout, restart. Never start or stop production without the owner's OK.

## What was built (plan `docs/superpowers/plans/2026-10-06-dsv41-dspark-both-cpu-experts.md`)

The owner's requirement: "we absolutely need to support both cpu_experts and dspark cpu_experts at the same time".
Before this, `SGLANG_DSV41_CPU_EXPERTS` and DSpark's draft CPU experts were mutually refused. Now:

- **One CPU expert team per NUMA node.** The GPU node's `CpuExpertEngine` serves the target's lease records and the
  draft channel as a second job source (`keep_warm_either`, one thread named `exl3-cpu-exp`). `DraftCpuThread` is
  gone; in both-mode the draft attaches through the service's `draft_host` (`SharedDraftHost`).
- **A 6-token verify fits one record.** The wire carries 1..64 lanes (u64 masks, `lane_mask.cuh`); a DSpark verify
  uses 40 lanes, a lane per route (36), plus V=8 victim lanes. `Exl3Quant::kMaxRoutes` is 64.
- **Forced NVMe misses never need staging.** They get lane slot -1 and the host reads them into RAM victims
  (`RamTier::reserve_victims_locked`). A startup room check refuses a layer that can't hold them.
- **CPU experts run M tokens per job** (route tables take M tokens; per-token token tables; rows sized from
  `graph_gather_rows // top_k`). This sits on the m2 plan's MAX_M=4 CPU forward.
- **The eager re-verify** still exists, GPU-only, and runs only before the copy engine arms (the first 16 decode
  forwards). In steady state, no verify overflows.
- **Gate:** `expert_stream_requirements_exl3.py` admits CPU experts with DSpark only in the tested shape (graphed
  verify + `VICTIM_LANES`). It refuses `VICTIM_LANES` without DSpark, and V ≥ verify routes when it can read top-k from
  the model's `config.json`.
- **BS1 build digest** (`scripts/dsv41/bs1_build_digest.py`, golden `test/manual/dsv41/golden/bs1_build_digest.json`,
  test `test/manual/dsv41/test_bs1_build_digest.py`) pins the batch-1 device SASS and the CPU kernel's keep-warm and
  forward. `--permanent` leaves out the post kernel and host `.text` on purpose. A toolchain change now fails unless
  `SGLANG_TEST_BS1_DIGEST_TOOLCHAIN_ACKNOWLEDGED=1`.
- **A/B tooling:** `analysis/dsv41-drive/dspark/both_cpu_ab.py` (three arms), `scripts/dsv41/dspark_text_band.py`
  (1.4-nat near-tie text bar), the session driver records accept-length inputs, and `run_arm.sh` takes
  `DSV41_HEALTH_TIMEOUT_S` (DSpark arms get 2700 s).

## Results (DSV41_REFERENCE.md §33.11 and its addendum)

Same 8 sessions, budget A, all six bars pass:

| Arm | ms/token median | mean | Notes |
|---|---|---|---|
| prod (old recipe) | 88.30 | 113.5 (one 283 ms/token outlier session) | KV 270,848 |
| dspark-both | 95.05 | 106.3 | accept length 3.588, KV 402,944 |
| dspark-draft-only (first A/B) | 353.44 | | reverify rate 1.00 |

- Text: first divergences within 1.4 nats of prod's argmax (max gap 0.75 for dspark-both).
- `reverify_ct` 9, within the pre-arming verifies (16 + 1); `gather_overflow` flat after arming.
- Target CPU-expert jobs: 27,068 on group 0.

The owner flipped production knowing dspark-both is ~7.6% slower at the median.

**Why budget A.** The first smoke run went out of GPU memory in the prefill warm-up. Cause, found by the memory
investigation (`mem-budget-report.md` in the SDD workspace): the `triton` FlashMLA prefill merge runs in fp32 and
costs ~0.9 GiB more than prod's FlashInfer path at a 2048 chunk and ~1.75 GiB at 4096. Cutting the hot cache alone
doesn't help, because the freed VRAM goes to the KV pool, which then overflows HiCache's host pool (ratio 14).
`flashinfer` is not an alternative: it refuses the 5-token draft capture (`num_tokens > 64`).

## What is left (owner-prioritized order unknown; ask)

1. **Option C (the owner's TODO in `arm_env.py`):** chunk the triton fp32 prefill merge over tokens (bit-identical
   math), then return the DSpark arms to `MEM_FRACTION_STATIC 0.82` and the full 12040 MiB hot cache (Owner
   decision 4). Needs a GPU A/B.
2. **m2 plan follow-ups.** `dsv41-cpu-plan-m2` (MAX_M=4 multi-token CPU forward, 17 commits) went to master inside
   this merge without its own Opus final review, and its Task 7 (DSV41_REFERENCE.md §33.10) was never done. The owner
   approved merging anyway. Do both. The handoff `docs/superpowers/handoffs/2026-10-06-dsv41-cpu-plan-m34-spill.md`
   covers m2's context.
3. **M9: no timing check of the batch-1 path.** The branch changed the post kernel and host code that non-DSpark BS1
   runs, and no merge-base (`3c115b9fb5`) vs `473691871d` prod-arm A/B on the same sessions exists. Behaviour is
   pinned by the hotpath golden; timing is not.
4. **Measure what is still unmeasured:**
   - the draft CPU forward ms. M12 now logs the both-mode draft's stats at shutdown, but the line appears twice at
     exit (the registry's atexit close logs the cached stats again);
   - the decode cost of budget A's 1200 MiB smaller hot cache;
   - a longer A/B than 8 sessions (per-session spread was 61.5–149.5 ms/token).
5. **Deferred minors from the final review** (`final-review.md`, lines 100-203):
   - M2: the two-node `_check_spill_room` path has no unit test;
   - M5: the CPU digest doesn't pin the `ExpertForward::keep_warm` wrapper;
   - M6: the graph suite has no CPU-experts parametrisation;
   - M10: a draft-attached idle engine polls every 50 µs;
   - M11: the watchdog can mislabel a job (relaxed loads);
   - M13: 13 commits carry a Sonnet trailer (cosmetic);
   - M16: `Nvfp4Quant::kMaxRoutes` is 8 (refuses cleanly).
6. **Untraced:** the CPU split re-tune path (`SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES`) under multi-token jobs.

## Where the evidence is

- **SDD workspace (git-ignored, laptop):**
  `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-both-cpu/.superpowers/sdd/2026-10-06-dsv41-dspark-both-cpu-experts/`.
  - `progress.md` is the ledger, with every task's commits, tests, `Ruling:` lines and follow-ups.
  - Also there: `task-N-brief.md` and `task-N-report.md`, `final-review.md`, `fix-report.md`, `fix2-report.md`,
    `minors-report.md`, `mem-budget-report.md`, and `common.md` (run env).
- **Laptop worktree:** `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-both-cpu` (branch
  `dsv41-dspark-both-cpu`, equal to master's tree).
- **divix01 runs:** `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/both-cpu/`
  (`smoke-*`, `ab/`, `ab-fix/`, `mem-budget/`).
- **divix01 worktree** `/data/models/slang/nvfp4-work/wt-both-cpu` (detached, safe to reuse or remove), and helpers in
  `/data/models/slang/nvfp4-work/both-cpu-tmp/`.

## How to work here (rules and traps learned in this run)

- Follow `.claude/rules/divix01-run-protocol.md`:
  - push first, then run only pushed commits in a private divix01 worktree;
  - set `PYTHONPATH=$PWD/python` and print `sglang.__file__`;
  - read `PIPESTATUS[0]`;
  - CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS` capped;
  - take `rowimg-disk.lock` before `cc-gpu.lock`;
  - no ad-hoc copies;
  - mutants only in a private worktree, reverted afterwards.
- **GPU tests with CPU experts on** need `flock .../cc-gpu.lock taskset -c 0-5,36-41`. With `6-17,32-63` the team
  finds no free node-0 core.
- GPU tests also need `CUDA_MODULE_LOADING=EAGER`, `/usr/local/cuda-13.4/bin` on PATH, `SGLANG_EXL3_SRC` and the
  gcc-15 `CXX` (see `common.md`).
- CPU registered tests run with **no** EXL3 env.
- Production is not on the GPU lock's side of any test: while production runs it holds `cc-gpu.lock`, so GPU tests
  and arms wait or fail with exit 75 (`gpu-run.sh`). Plan GPU work with the owner.
- **`run_arm.sh` refuses an unregistered python tree.** It checks the untracked `generations.json` on divix01. A new
  tree must be registered there first, or the arm aborts before starting a server.
- **Cold start ≈ 10–15 min.** Weights read cold (attempt 1 took 435 s, a warm one 13 s), and every JIT module
  rebuilds after any C++/flag change (50–100 s each). Warm starts reach `Started server process` in ~5.5 min.
- **Root disk is tight.** `/` was 95% full; `/tmp` lives there. pytest leaves GBs under
  `/tmp/pytest-of-dnikolaidis`. Cleaned to 91% before the production start. The JIT cache is on `/home` (plenty).
- **Remote launches:** `ssh divix01 '... &'` hangs the tool; use `setsid nohup ... < /dev/null > log 2>&1 &`.
  Subagents that "wait on a monitor" stall for hours; tell them to poll logs in ≤10-minute stretches.
- **"Pre-existing failure" claims** must be checked at the merge-base, not at the branch HEAD. Two branch regressions
  were mislabelled this way; both were then fixed (`dd29a5e`).
- **Gate tests** must assert that the EXL3 requirements resolved. A missing model dir silently falls back to NVFP4.
- **Git and frozen files:** no amend, rebase, stash or force. Stage files by name. Don't edit `model_runner.py`
  (frozen). Never push master without the owner's explicit OK.
- **Reviewers and models:** reviewers run on opus; never use fable.
- **Commit trailer:**
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq`.

## Suite baselines (for comparison)

- `test/registered/unit/kernels` + `test/registered/unit/test_expert_stream_requirements_exl3.py`
  - Command: `OMP_NUM_THREADS=8 taskset -c 0-63 python -m pytest ... -q -p no:randomly`, no EXL3 env.
  - At `31b385b1de` (= master's tree): 1967 passed, 25 skipped, EXIT=0, 16.5 min.
- `benchmarks/dsv41_baseline`: 166 passed.
- BS1 digest: passes on a fresh build.
