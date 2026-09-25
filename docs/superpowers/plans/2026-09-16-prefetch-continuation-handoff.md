# Expert prefetch continuation handoff

## Objective

Use the recorded final-code baseline to start the pipeline changes that attack the measured current-system losses. The original symptom was lower tok/s with prefetch enabled on PCIe Gen3 x16. Correctness validation, a diagnostic Nsight trace, and a reduced timed baseline are now done at `57c842ae6e`. Do not claim a speedup until a change beats matched measurements.

## Authoritative references

- HiCache/production design and the appended `python in hot path` audit: [2026-09-16-hicache-expert-prefetch-optimization-handoff.md](2026-09-16-hicache-expert-prefetch-optimization-handoff.md)
- Measurement protocol and B/C/Cr/N/D arm definitions: [2026-09-16-prefetch-throughput-recovery.md](2026-09-16-prefetch-throughput-recovery.md)
- Earlier operational state/matrix handoff: [2026-09-16-prefetch-plan-completion-handoff.md](2026-09-16-prefetch-plan-completion-handoff.md)
- Persistent decision ledger (rulings, full result table, run directories): `.superpowers/sdd/2026-09-16-hicache-expert-prefetch-optimization-handoff/progress.md`
- Trace launcher and provenance contract: `scripts/expert_prediction/run-shadow-server.sh`; sudo environment relay: `scripts/expert_prediction/trace-env-relay.sh`

## Current code state

Branch: `master`, pushed to `shared` at `f2c6cd2cbc` (the 2026-09-17 optimizations and test fixes are merged; see Implementation results). Commits added after `0283eacb74`:

| Commit | Content |
| --- | --- |
| `cdc70383bc` | This handoff (first version). |
| `c27fc6f3f8` | Fix: JIT top-1 kernel used undefined `CUDART_INF_F` and never compiled; plus four CUDA test fixes (capture-writer frame release, sync-test warm-up, budget regex, physical-row assertion). |
| `57c842ae6e` | Launcher relays its environment through a sudo-prefixed `PREFETCH_TRACE_COMMAND` (`sudo` `env_reset` otherwise drops every `SGLANG_*` setting). |
| `9304aa4c9a` | Handoff update with validation and baseline. |
| `70100da7a4` | Project `CLAUDE.md`: use graph-mode Nsight tracing for timing. |

Earlier work in the branch:

| Area | Commits | Result |
| --- | --- | --- |
| No-offer semantics | `ef81a4d426` | Persistent validity; current-forward `id=-1,count=0`; live residency recheck. |
| Planner foundation | `f18b95794e` | Real posted count and optional `[covered,residual,wasted,posted]` outcomes. |
| Production path + Task C | `42a4d268e2`, `7bc343e5a3`, `50f860888f` | Count-safe fused/generic planning, CUDA JIT BS1 fp32 top-1 selector, reference fallback for shadow/calibration/wider diagnostics, physical-row accounting and cache-budget charge. |
| Async metrics/capture | `a9b7ac0b78`, `9128855359`, `3c0f5114bf`, `d879b685f8` | Owned pinned snapshots, event polling, bounded writers, background CPU formatting/I/O, bounded trace admission/order state. |
| Test determinism | `0283eacb74` | Fixes the CPU writer-queue test's scheduling race. |

Important design properties:

- The fast selector is a real JIT CUDA kernel. It is BS1/contiguous/fp32 only and is used only when a pull is actually posted (`pull_mode=always`, logged `serving_top1=True bank_width=1`). Pull modes `off` and `count_zero`, shadow recall, calibration and wider diagnostics use the width-16 reference bank (`serving_top1=False bank_width=16`).
- Planner coverage requires `posted_count == 1`; a stale nonnegative ID with count zero remains a demand route.
- The dedicated speculative row is charged against cache budget and is not a permanent residency slot.
- Optional telemetry does not wait on CUDA, serialize JSON, or write files on the inference path. Required cache-control and doorbell safety synchronization were not removed.

## Validation completed

- CPU: `test_async_telemetry.py` 6 passed on divix01.
- CUDA, 13-file suite from the shadow worktree under the GPU lock: **425 passed, 3 failed** at the fixed code. The three failures also fail at pre-work commit `0fe8d526df` and are not regressions: `test_forward_taps_never_synchronize` (test's own pageable `.to(cuda)` trips sync-debug) and two doorbell tests that are suite-order artifacts. Logs in `cc-expert-prediction/logs/cuda-tests-*.log`.
- Diagnostic trace (LLaPor D, `servers/prefetch-recovery-llapor-trace-d-final/run-20260916-190633`): relay kept configuration; `copy_expert_row_segments_gpu_kernel` is 78.4% of GPU kernel time, `select_prefetch_top1_kernel` 0.1%; GPU metrics captured; stats in `trace/stats/`.

## Baseline results at `57c842ae6e`

Reduced by user direction from the full matrix. Per arm: cold server on 31040, startup mode verified, fixed warm-up, 8 sessions / 29 turns at 768 tokens, fused plan 1 as *recorded* (false: the launcher printed a literal and never exported the flag, so these arms ran the generic planner; see `MOE_EXPERT_TRANSFER.md`, "Fused route planner arm"), candidates 16, budget 2, hot GPU 10240, calibration 0, not profiled. All arms 29 records, 0 errors. Run directories are listed in the ledger.

| Arm | Median decode tok/s per pass | Mean |
| --- | --- | ---: |
| B (old commit `0fe8d526df`) | 14.176 | 14.18 |
| LLaPor C | 13.384 | 13.38 |
| LLaPor Cr | 13.426, 13.412 | 13.42 |
| LLaPor N | 13.044, 12.897 | 12.97 |
| LLaPor D | 14.263, 14.105, 14.273 | 14.21 |
| APEX D | 13.944 | 13.94 |

What the baseline says about the current system (exploratory):

1. Prefetch is at break-even: LLaPor D ~= old-commit B; APEX D ~1.7% below it (one pass).
2. Scoring is the largest loss: C is ~0.8 tok/s (~5.6%) below B, and D's real pull only recovers it.
3. Empty pull N-C costs ~0.4 tok/s; shadow recall Cr-C is within noise (~0.15 tok/s pass spread).
4. Expert row copies dominate GPU time; selection cost is negligible under the top-1 kernel.
5. Confound: C/Cr/N use the width-16 reference bank, D uses top-1. N-to-D mixes payload and selector effects, and part of C's cost may be the wider bank.

Gaps: APEX has only one D pass. Final-code B was measured on 2026-09-17 (13.976 and 13.854 median, host-overhead matrix), confirming D ≈ B.

## Safety rules

- Do not touch production port 7867. Never stop/relaunch it without explicit user direction. It is currently down.
- `model_runner.py` is frozen. Do not edit it.
- Every GPU run: `/data/models/slang/nvfp4-work/cc-gpu.lock`, before/after GPU census, `taskset -c 0-63`, `OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=32`. Never profile timed arms.
- Do not reset, clean, stash, or alter unrelated `.omc`, untracked docs, `python/uv.lock`, or user files.
- Shadow worktree `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree` is detached at `57c842ae6e`. Deliver new commits to it by push to `shared` (after asking) or by git bundle.

## Decode-cost analysis (2026-09-17)

Full detail and rulings in the ledger. Per decode token (LLaPor D trace, node-mode, ~10% inflated): demand copies ~45 ms, model compute ~15 ms, pull copies ~17 ms.

- Pulls sit on demand copies: 62% of pull time overlaps a demand copy; only 26% overlaps compute.
- Copy vs compute contention is ~free (≤2.5% either side, GPU microbenchmark, bytes validated); copy vs copy shares the link (pull +97%, demand +~1 row).
- No inter-layer idle: the copy-free window between layers is ~0.34 ms, ~0.30 ms of it compute. A 1-row pull takes ~0.24 ms, so it fits a compute window.
- The ~9 ms step-tail idle is mostly an nsys `--cuda-graph-trace=node` artifact (~6 ms in `cudaGraphLaunch`); real host tail ~3–4 ms (batch prep 2.4 ms, output/IPC 0.55 ms). Use `--cuda-graph-trace=graph` for timing traces (project `CLAUDE.md`).
- Cross-token end-of-step prefetch is not worth it (best 8.7 misses covered at 22% precision with 40 rows).
- Misses concentrate in layers 0–2 (59–69%); wasted pulls concentrate in layers 15/31/39/40 (43–58% precision).

## Implementation results (2026-09-17)

> **Superseded headline.** Everything below stands as measured, but the campaign's
> final result is **stage 2 (DIRECT) at 19.360 tok/s**, accepted 2026-09-18 on an
> acceptance-grade paired arm. Where this section says "best known is 16.645",
> read that as "best known *as of 2026-09-17*". See **Next actions** and
> [`MOE_EXPERT_TRANSFER.md`](../../../MOE_EXPERT_TRANSFER.md).


Full detail, commits and run directories are in the ledger. The accepted work is merged to `master` @ `f2c6cd2cbc`; the rejected and in-flight branches remain separate. Compare arms only within the same matrix; CPU placement differs from the 2026-09-16 baseline.

| Work | Branch @ commit | Result | Status |
| --- | --- | --- | --- |
| Insert-on-miss residency (missed experts promoted D2D from scratch into slots each decode step) | `insert-on-miss` @ `679e1a7fb2` | B: 13.99 → **15.98 tok/s (+14.2%, −8.9 ms/token)**; misses 190 → 155 rows/token, decode H2D promotions 21 → 0; +190 MiB peak | Accepted; parity repeat and combined build running |
| Overlap scheduling with expert streaming | `host-decode-overhead` @ `90914e33ac` | B: +4.5% / +2.9% median over two pairs (~2.2–2.4 ms/token); base arms 13.98 / 13.85 are the final-code B | Accepted |
| Batch-prep reduction | — | With overlap on, ≤1.07 ms/step host work can still delay the GPU | No change needed |
| Top-1 selector in all pull modes | `scoring-cost` @ `3baf985d33` | C 13.33 → 13.81 (+2.6 ms/token); D unchanged (already top-1). Scoring breakdown: width-16 bank 2.37 ms, scorer 1.71 ms, shadow recall 1.88 ms | Accepted for C/N |
| Task 3 compute-window pull placement | `task3-compute-window-r2` @ `e08dd952ac` | Pull-over-demand overlap 62.9% → 0%, but D 14.1–14.3 → 13.5–13.75: pulls now stall `_hc_mix_persistent_kernel` (13 µs → 186 µs, 47×/step) | **Rejection reversed — see below** |
| Task 3 re-test on top of the hc_mix fix | `task3-on-hcmix` @ `ba362b5bd7` | Control (cap only) 14.903; both 15.375 / 15.280, mean 15.328 = **+0.425 tok/s, +2.9%** on top of the cap's own +4.4%. Matrix `t3hc-20260917-131434`, 29 records / 0 errors per arm. Suites 140 passed, 5 known failures | Directional win, not ABBA-clean (one control only; drift ~0.19 tok/s is ~45% of the effect) |
| `_hc_mix` copy-stall fix (cap the fused mix's grid so copies always find free SMs) | `hc-mix-stall` @ `732acac42f` | **D 14.089 → 14.713 (+4.4%)**; B 13.836 → 13.962 (noise, no pulls to stall behind). D's lead over B widens +1.8% → +5.4%. New `SGLANG_OPT_HC_MIX_MAX_CTAS` (default 128, `0` = old behaviour) | Accepted |
| Static per-layer pull gating | `pull-layer-gating` @ `fa5b873a34` | Estimated 0.1–0.5 ms/token, below noise | Timing cancelled; code kept |
| Layer-0 token-id table | — | Offline best ~0.85–0.91 ms/token (ceiling 1.41) | No-go |
| Strategy review / compression | — | Insert-on-miss and 12 GiB budget are the top levers; lossless NVFP4 compression ~0.95 ratio | Compression not viable |

Combined build (`combined-iom` @ `f727a001f7` = insert-on-miss + overlap + scoring + hc-mix), one arm per condition, adjacent in time, 29 records / 0 errors each:

| | combo-B (prefetch off) | combo-D (LLaPor, pull always) |
| --- | ---: | ---: |
| median tok/s | **16.645** | **15.928** |
| miss rows/token over the link | 159.8 | 181.0 |
| insertions = evictions per token | 159.5 | 133.9 |
| residency slots | 3403 | 3355 |
| pull posted / useful / wasted per token | 0 | 46.5 / 36.0 / 10.5 (77.4%) |
| peak GPU MiB | 27,621 | 27,663 |

**Ruling: prefetch does not earn its keep on top of the full stack — it costs 4.3%.** Best known configuration is the full stack with prefetch OFF at 16.645, i.e. baseline 13.96 -> 16.645 (**+19%**), entirely from non-predictive work (residency, scheduling, a kernel fix). Direction established; magnitude approximate (one arm per condition, so drift is not differenced out, but the 0.72 tok/s gap is well outside the ~0.3 noise floor and the arms ran 20 min apart).

D moves 21 more rows/token over the link. Three mechanisms, increasing size:

1. Pull-row slot cost, ~1.4 rows/token (48 slots, one per layer).
2. Wasted pulls, ~10.5 rows/token at 77.4% precision.
3. **Prefetch starves insert-on-miss of residency, ~9 rows/token.** A pull-covered route is served from the dedicated pull row, so it is correctly not a demand miss and is therefore never inserted; the expert stays nonresident and misses again later. Insertions fall 159.5 -> 133.9/token. Under insert-on-miss, every covered route is a learning opportunity removed from the cache. The two features actively fight each other. This was not anticipated.

Arithmetic check: +21 rows/token at 0.23 ms/row implies ~+4.9 ms/token, while the measured gap is ~+2.7 ms/token. The measurement is smaller than the row count predicts, consistent with pull copies partially overlapping compute — so the counters understate prefetch's disadvantage rather than overstate it.

Follow-up simulated and closed (2026-09-17, `analysis/strategy/sim_pull.py`, CPU-only; `sim_policy.py` left untouched). Policies over the same captured traces:

| policy | slots | link rows/token | insertions/token |
| --- | ---: | ---: | ---: |
| A no pull | 3403 | **147.5** | 147.4 |
| B pull, covered not inserted (today) | 3355 | **168.0** | 121.4 |
| C pull, covered inserted from the pull row | 3355 | **160.4** | 148.9 |
| C + pull row returned | 3403 | **159.0** | 147.4 |

Inserting covered experts is a real repair — it recovers ~9 rows/token over today's behaviour (~2 ms/token) — but it does **not** rescue prefetch: C is still +11.5 rows/token worse than not prefetching at all, extrapolating to ~16.4 tok/s against combo-B's 16.645.

**The calibration gate failed** (A −7.7%, B insertions −9.3%; the capture holds 13,685 decode rows for these sessions against 9,657 in the timed arm, so longer generations with better locality). The conclusion survives anyway because it rests on an identity rather than on the model's levels:

> **C − A = wasted rows = posted x (1 − precision)**

Once covered experts enter residency, every other prefetch cost cancels — each posted row either replaces a demand row or is waste. The simulator produced 11.5 rows/token for both quantities independently, as the algebra requires. At the measured 77.4% precision and 46.5 posted rows/token that is ~10.5 rows/token which no residency change can recover, because the byte has already crossed the link.

**Ruling: prefetch is unprofitable on this stack and the line is closed.** For it to win, precision would have to approach 100% (from 77.4%), or re-timing copies into compute windows would have to pay — and the measured arms already tested that, with D landing 0.72 tok/s behind B. Two independent lines of evidence agree, and the identity explains why the measurement came out as it did.

Caveat: the pull selector was a sampling model (measured per-layer posting rate and precision), not LLaPor, because the capture records hidden states but no pull decisions. So C's *benefit* may be understated with the real selector; the *cost* term is measured, not modelled, and it is the term that decides.

Prefetch cost accounting (as of 2026-09-17) — why prediction quality is not the lever:

- Precision is fine: 77.7% (396,419 useful of 510,326 posted, always-pull arm). The predictor is not guessing wrong.
- Coverage is small: pulls cover ~19% of misses (36.2 useful of 190 misses/token); the other 154 arrive as demand copies regardless.
- Prefetch cannot reduce bytes, only re-time them. At 2.76 MB/row, 190 misses is ~525 MB/token, ~46 ms of link time inside a ~68 ms token — the copy path is already ~2/3 saturated by demand traffic, so there is little idle link to hide pulls in. Wasted pulls (10.4 rows/token) consume ~2.5 ms of that scarce headroom.
- Enabling the pull row **costs one residency slot per layer**, shrinking the hot cache that insert-on-miss fills and raising miss pressure. Found 2026-09-17 while diagnosing the pull-row test.
- Consequence: byte-reducing levers (larger budget, insert-on-miss Stage B) beat better prediction. A predictor at 95% precision would still only re-time 19% of the traffic.

Speculative decoding / larger tokens-per-forward — measured and rejected (2026-09-17, `analysis/cross-token/spec_window.py`, CPU-only, 299,687 sliding windows over the 303,935-token capture with recorded residency). **N=1 reproduces the recorded miss rate to 0.2% (156.52 vs 156.8), so these numbers are calibrated, unlike the pull simulator's ~8% level bias.**

| N tokens/forward | routed reuse | miss reuse | miss rows/token | rows per accepted token @ a=0.7 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00x | 1.00x | 156.5 | 156.5 |
| 2 | 1.19x | 1.09x | 144.0 | 169.5 |
| 4 | 1.40x | 1.17x | 133.4 | **210.7 (+35%)** |
| 8 | 1.68x | 1.28x | 122.7 | 312.4 |

The mechanism: consecutive tokens do share experts (1.68x union reuse at N=8), but **the shared experts are the popular ones already resident**. Those repeats are cache hits, and hits are free; the rows that actually cross PCIe barely repeat (miss reuse only 1.28x). Batching tokens captures reuse exactly where it was already worthless.

Break-even acceptance is a=0.841 (N=2), 0.894 (N=4), 0.929 (N=8); a realistic MTP head is well below that, so speculation moves MORE bytes per useful token than plain decode (~13.8 tok/s estimated at a=0.7, N=4, against 16.645 today). Even the impossible best case — every draft accepted, no verification cost — only reaches 122.7 rows/token, and that ignores verification compute, the draft head, and a 981-row per-forward working set against 4,180 slots that would itself churn the cache. Benefit is uniform across layers (N=8/N=1 ratio 0.72-0.90, L0 worst at 0.897), so no subset of layers pays off disproportionately.

Note this is the second time an idea has failed for the same structural reason: **on a link that is the bottleneck, only reducing bytes helps.** Prefetch re-times bytes (fails); speculation re-groups bytes (fails). Residency policy, row compression, and reducing the 2.76 MB row size change bytes and are where effort should go.

Lessons:

- Serving-level greedy bit parity is not available on this stack: the base build is not repeatable against itself even with a frozen hot cache. Accept on unit-level byte-exact tests plus discrete-value logprob comparison and large-margin flip counts.
- "Copy overlapping compute is free" was false for `_hc_mix_persistent_kernel` because it launched one CTA per SM behind a device-wide barrier: a copy holding a few SMs kept the barrier closed. Fixed by capping the grid. A persistent kernel that both fills every SM and waits at a device-wide barrier cannot tolerate a concurrent copy — check both ingredients before writing one.
- A trace sweep found `_hc_mix` to be the only such kernel here: it outlives the copy it overlaps in 100% of 44,842 instances, while all 20 other kernels do so in 0%. Use "does the kernel outlive the copy" as the test; a with/without-overlap duration ratio is confounded by phase on an always-pull arm, where nearly every copy-free sample is prefill.
- Graph-mode Nsight traces hide kernels inside the decode graph; use node mode (attribution only) for copy-overlap questions.
- Read a red *new* test differently from a red *established* one. The pull-row test was written during combo prep and had never been green; its first failure looked like a combined-build regression and triggered a GPU hold, but the code was correct and the test asserted residency one forward too late (under the pull row's reduced capacity, normal eviction churn removed the freshly inserted row). Before calling a failure a regression, check `git branch --contains` on the test's commit and diff the implementation files across the branches — here they were byte-identical, which exonerated the merge immediately.

Timing protocol (user ruling, 2026-09-17): two tiers.

- Screening arm (~10 min): same cold-server, startup verification and warm-up, then 4 sessions (~15 turns) at max 384 tokens. Use only when a microbenchmark or model predicts ≥0.5 tok/s. Pair arms by session/turn; a same-day old-build control may be reused when the expected effect is well above the ~0.2 tok/s cross-matrix drift.
- Acceptance arm (~18 min): the full protocol (8 sessions / 29 turns / 768 tokens, ABBA order). Required before accepting a change, and for any hot-cache or residency change, whose steady state takes most of a session to reach.

Iteration-speed rulings (user, 2026-09-17). Arm cost breaks down as ~26 s process start, **129 s weight load**, 62 s graph capture + hot-cache fill, ~60 s warm-up, ~13 min timed (measured on `t3hc-llapor-d-both-p2`). Only the last line buys measurement; the first four are overhead.

- Keep the page cache between arms: the launcher runs `weight_loader_drop_cache_after_load=True`, which re-reads weights from NVMe every arm. Retaining it should save ~90 s/arm at no measurement cost (188 GB RAM, ~85 GB free). Verify the host expert arena is unaffected before trusting numbers from it, and never change this mid-matrix — all arms in one matrix must share the setting.
- Screening arms (15 turns / 384 tokens) are the default for go/no-go; full arms only for acceptance. Precision scales as 1/sqrt(n): spread ~0.15 tok/s at 29 turns, ~0.21 at 15, so the smallest reliably detectable effect moves from ~0.3 to ~0.42 tok/s (~3%). Task 3's +3.2% would have been marginal on a screening arm — screening filters, it does not accept.
- ABBA only when the expected effect is within ~5x drift. Observed drift is ~0.19 tok/s over a day (14.713 -> 14.903 for the same build). Above ~1 tok/s expected effect, two arms suffice.
- Microbenchmark first for kernel and copy-path work; spend serving arms only to confirm the winner. The hc_mix microbenchmark picked the CTA cap in minutes and its prediction held at +4.4% in the full matrix.

## Next actions

**Campaign landed 2026-09-18. Shipping config is insert-on-miss stage 2 (DIRECT)
plus the fused route planner at 20.695 tok/s, +48.2% over the 13.96 baseline**
(stage 2 alone: 19.360, +38.7%). Branch
`insert-on-miss-stage-b` @ `797be6f678`, fast-forwarded into
`master` 2026-09-18, **not pushed** — the repo owner pushes.
Full results, the winning config and every closed line live in
[`MOE_EXPERT_TRANSFER.md`](../../../MOE_EXPERT_TRANSFER.md), which is now the
authoritative document; this file is history plus what remains.

1. **Done and accepted.** Stage 2 beat stage 1 + fused kernel **19.360 vs 18.541
   median, faster on 27 of 29 paired turns (p < 1e-5)**, acceptance grade,
   `OVERLAP_SCHEDULE=1`. Mechanism: no scratch region -> 3403 -> 3883 slots ->
   +3.09 points hit rate -> 9.9% fewer miss rows -> 42.5 MB/token less over PCIe.
2. **Done, but off the shipping path.** The fused masked insert kernel cuts stage
   1's boundary ~45%, and stage 2 has no such boundary. Kept default-off,
   stage-1 only, by owner decision. Do not cite it as part of the shipped result.
3. **Closed since the last revision:** Stage B's in-graph penalty estimate (it was
   contaminated — the two stages did unequal work in the same harness); the shared
   10-row scratch pool (dominated by stage 2, and the indexing forbids it); the
   row-size lever (set by the model, not tunable); PCIe Gen4/5 (**the host board
   is Gen3 — confirmed hardware, not configuration**).
4. **Still open, in the order the bandwidth analysis implies:**
   - **12 GiB budget** — never measured. At 12 GiB stage 2 gives ~4,660 slots
     against stage 1's ~4,180. Same +480, and the only untried capacity lever.
   - **B2, now backlog #22** (fold stage 2's `gather_destinations` + `_commit_gather`
     into the fused planner). The decode trace bounds it at ~1.9–2.4 ms/token; the
     planner itself realized ~70% of its trace bound. Its original ~1% was scoped
     against stage 1 and is superseded.
   - **#20 offline blockscale re-coding** — +1.6% lossless / +3.2% lossy.
     **DEFERRED: do not start without asking the repo owner.** The lossy variant
     changes model numerics and that trade is theirs.
   - Fused scorer kernel — only if prefetch is ever revived, which the evidence
     says it should not be.
5. **Prod deployment: done in the script, not yet restarted.** The
   `main-port-probe-7bc4eb` worktree is detached at `797be6f678`, and
   `run-nvfp4-e16c-public.sh` carries stage 2, decode-forwards 1, overlap on and
   `SGLANG_MOE_EXPERT_FUSED_PLAN=1` (backups `.bak-pre-stage2-20260918`,
   `.bak-pre-fusedplan-20260918`). Stage 2 was validated under that script; the
   fused-planner line has not been started there yet. Prod was left down.
6. **Read before running any arm:** the `hot_update_decode_forwards` launcher
   warning in `MOE_EXPERT_TRANSFER.md`. Main-tip's `run-shadow-server.sh` tests
   `= 1`, which silently yields 4 under stage 2. That file has no test coverage,
   so the failure is silent and the arm still reports a number.
7. Optional cleanup on divix01 (ask first): nested `wt-task3/wt-task3-base`,
   `cc-expert-prediction/baseline-0fe8d526df` worktree, `trace-relay-proto/`,
   `fix-57c842ae6e*.bundle`, the 3.9 GB trace `report.sqlite`, and finished agent
   worktrees including `wt-stageb`.

## Operational patterns

SSH:

```bash
timeout 90 ssh -n -o BatchMode=yes divix01 '<command>'
```

**Do not pass `ControlMaster=no` or `ControlPath=none`** — an earlier revision of
this file recommended them and that guidance is withdrawn. Use the plain form
above.

Retry read-only status checks after transient SSH resets. Never use broad `pgrep` patterns that can match the checking shell itself. Stop only PIDs conclusively shown to be self-started shadow processes (the runner SIGTERMs the `sglang serve` child of its own `flock` launcher).

Privileged trace example (tokens only; no quotes or backslashes):

```bash
RUN_KIND=trace PREFETCH_TRACE_COMMAND="sudo -n /usr/local/sbin/nsys-profile profile -f true -o <run_dir>/trace/report --trace=cuda,nvtx,osrt --cuda-graph-trace=node --gpu-metrics-devices=all --delay=240 --duration=240" \
  scripts/expert_prediction/run-shadow-server.sh <name> 31040 off
```

Server startup takes ~200-220 s; choose `--delay` so the capture window opens after health and warm-up. nsys ends the server at the end of `--duration` (launcher rc 137 is expected).
