# sampled-accept at the upstream merge

Owner decision (2026-10-17): at the upstream merge, take upstream's greedy drafting (`SGLANG_DSPARK_USE_GREEDY_DRAFT`,
a65b776963, #43670) and block verification (`--speculative-use-block-verification`, e902207a4e, #43012; kernel from
57826a62ea, #42297) and drop this branch's versions. This file inventories everything on `origin/sampled-accept`
(base `origin/warmup-preload` 7b3fc4cd7b) and says what must survive.

Verdicts: **DROP** (upstream covers it equivalently), **PORT** (upstream lacks it and it matters; where it goes in
upstream's structure), **KEEP-AS-IS** (independent of the merge).

## Evidence behind the verdicts

- **The algorithms are equivalent.** Upstream's greedy draft materializes a one-hot q and runs the classic kernel.
  That accepts X with probability p(X), samples the residual max(p - 1[X], 0) (p with X removed), and draws the bonus
  from p after a full accept. Top-k/top-p apply to the target only, through the same builder. Upstream's block mode
  (`reject_sampling.py` `BLOCK_VERIFICATION`) is Sun et al. Algorithm 2: w_i = min(w_{i-1} p/q, 1),
  h_i = r_i / (r_i + 1 - w_i), h = w at the last step, the largest i with coin < h_i, and a residual of
  max(w_tau p - q, 0). This branch's kernel implements the same rule.
- **Distribution check on upstream's kernel.** This branch's harness (2^21 samples per config; joint and per-position
  chi-square; no filter, top-k 2, top-p 0.75, top-k 3 with top-p 0.8; sampled and one-hot q; token-by-token and
  block) was run against upstream's `reject_sampling.py` at `upstream/main`. Every config passes (min p 0.0033 against
  a 1e-6 floor) with no cut and with a static cut, where dead rows carry real candidates and finite garbage logits.
  Scratch: `divix01:/mnt/nvme1/sampled-accept/upstream_dist.py`, log `.../logs/upstream-dist.log`.
- **A cut ends a block-verified verify losslessly by capping.** With the chain run over all S rows and correct_len
  then capped at verify_len - 1, the first verify_len output tokens depend only on live rows. The dead rows act as
  any valid target there. This holds for upstream's block mode as well as for token-by-token.
- **Zero-filled dead candidates do bias the sampled accept** (classic and block; all p = 0 in the scratch check).
  The eager sampled path never builds them: `accept_and_finalize` passes `verify_ids_2d`, which holds the real
  drafts, and `run_compact` fills dead logit rows with 0. Only the folded greedy epilogue zero-fills candidates, and it
  never accepts sampled rows. So this is a constraint to keep, not a live bug.
- **Kernel cost at bs 1** (V = 129280, S = 6, each variant replayed 20 times in a CUDA graph). Upstream takes
  35-45 us token by token and 34-38 us block with a sampled q, and 48-54 us with one-hot q, which adds the zeros and
  scatter. This branch's kernel takes 7-10 us. Eager host time per verify is upstream 54 us (93 us one-hot) against
  114 us here, from two Triton launches and small allocations. Scratch `.../bench.py`, log `.../logs/bench2.log`.
  Served effect: not measured; the served a/b was cancelled by the owner.
- **Block verification adds nothing with a greedy (one-hot) draft.** Offline mean correct_len is equal within noise:
  0.0964 vs 0.0969, 1.6622 vs 1.6618, and so on. With a sampled draft it adds 3-11%: for example 0.694 -> 0.794,
  1.383 -> 1.452 and 1.351 -> 1.502. A served test of upstream's block verification must therefore draft by sampling,
  with `SGLANG_DSPARK_USE_GREEDY_DRAFT` off.

## Inventory

| # | Item (commit) | What it does | Upstream equivalent | Verdict |
|---|---|---|---|---|
| 1 | `SGLANG_DSPARK_SAMPLED_GREEDY_DRAFT` env (4b2834af74, moved in 62363c6fdc) | Flag: sampled requests draft by argmax | `SGLANG_DSPARK_USE_GREEDY_DRAFT` (a65b776963, `environ.py`) | DROP |
| 2 | `DraftBlockResult.greedy_draft` (4b2834af74) | Marks a one-hot draft so the accept needs no `corrected_logits` | Same field (a65b776963, `dspark_draft.py` `DraftBlockResult`) | DROP |
| 3 | `sample_draft_block(greedy_draft_sampled=)`, the proposer folding sampled batches into the greedy-only draft graph (4b2834af74) | Argmax draft through the eager sampler or the greedy fold | `draft_markov_block(..., greedy_draft=)`, `DsparkDraftSampler(greedy_draft=)`, `MarkovGreedyStep` for one row (a65b776963). Our `SampleStepTokens` / folded-sampling / `sample_block_greedy_fused` code is deleted by the merge | DROP |
| 4 | `_accept_sampled_rows` routing in `dspark_verify.accept_draft_tokens` (4b2834af74, 39c83cbb19) | Sends one-hot or block-verified sampled rows to `dspark_sampled_accept.py` | `accept_draft_tokens` builds a one-hot q and passes `block_verification` to `AcceptSampling` (merge branch `dspark_verify.py:977-1022`) | DROP |
| 5 | `SGLANG_DSPARK_SAMPLED_BLOCK_VERIFY` env and `TargetVerifyExecutor._sampled_block_verify` (39c83cbb19) | Flag for block verification | `--speculative-use-block-verification` -> `TargetVerifyExecutor(block_verification=)` (e902207a4e; merge branch `dspark_worker_v2.py:524`, `dspark_verify.py:122-186`) | DROP |
| 6 | `dspark_sampled_accept.py` kernels (`_chain_residual_partials_kernel`, `_chain_accept_kernel`; 4b2834af74, 39c83cbb19, 71383dbda6) | Token-by-token one-hot and block verification. Per-row vocab-block partials over S x 32 CTAs, a bonus search in one 4096 block, no q materialized, stops at verify_len - 1 | Classic kernel's `BLOCK_VERIFICATION` mode (57826a62ea, `reject_sampling.py`), one CTA, lazy residual scans | DROP now. Optional later adapter: behind upstream's flags in `accept_draft_tokens` when `draft_block.greedy_draft or block_verification`. Only if served ms/verify shows the accept's GPU time matters, and only after cutting its host overhead (cache the partial buffers and live-draft tensor) and restoring a cap_trim_lens counterfactual |
| 7 | `preload_sampled_accept_kernels` at worker init (4b2834af74, 39c83cbb19) | Loads the new kernels before serving | None. The merge branch has neither the cut-kernel preload (7b3fc4cd7b) nor warmup-preload's sampled decode warm-up, so `speculative_sampling_classic_kernel` (block variant and the plain one) device-loads on the first sampled verify after serving starts | PORT: (a) carry warmup-preload's `dsv41_decode_paths` sampled requests (temperature 0.6, and 0.6 / top_p 0.95 / top_k 20), already listed in the re-sync table; (b) add a worker-init preload next to `preload_cut_accept_kernels` in `dspark_accept.py` that runs `chain_speculative_sampling_triton(..., block_verification=<arg>)` once on bs-1 CUDA dummies with the real vocab and S (int32 buffers from `_get_or_create_chain_verify_buffers`). Call it from `TargetVerifyExecutor.__init__` when the arg is on, or always, since the token-by-token variant also loads late. Check: zero "device-loaded after serving started" lines in a sampled served run |
| 8 | Dead-row NaN caveat (no commit; analysis here) | Block verification reads rows past the cut. A NaN dead row L makes h_{L-1} NaN, so tau = L-1 is never chosen (bias). Token-by-token is immune because it stops reading at the first reject | None | PORT (small guard): in `dspark_accept._accept_sampling_core`, when `block_verification` and `cutoff_verify_lens` are set, replace target-prob rows at or past verify_len with any valid distribution (`torch.where(dead, uniform, p)`) before the kernel. Or assert finite dead rows in a GPU test of the static dead-row path |
| 9 | Ragged-cut finding (scratch `classic_cut.py`, `upstream_dist.py`) | Zero-filled dead candidates bias the sampled accept, classic and block | None needed today: the eager sampled path passes real candidates (`verify_ids_2d`) | PORT (guard plus test): keep sampled accepts on real candidates. Either assert in `accept_draft_tokens` that sampled rows never come from the epilogue's zero-filled candidates (`DsparkVerifyEpilogue.__call__` scatters with `fill_value=0`), or add a distribution test case with a ragged `cutoff_layout`, so a future fold of the sampled accept into the epilogue fails loudly |
| 10 | `cap_trim_lens` / block-accept estimator counterfactual | This branch's kernel stops at the cut and reports `cap_trim_lens = 0`, so `block_accept_lens = commit + trim` loses the would-have-accepted count | Upstream runs the full chain and caps (`CapCorrectLen`), keeping today's semantics (on static dead rows the counterfactual is computed against non-true rows, as it already is for greedy) | DROP. Upstream keeps the existing behaviour |
| 11 | `test/registered/spec/dspark/test_dspark_sampled_accept.py` (4b2834af74 .. 39c83cbb19): distribution harness (`_Chain`, `_chi2_pvalue`, `_filter`, `_pvalues`) and `TestSampledAcceptDistribution` (flags alone and together, cuts None/1..6/mixed, top-k/top-p, a 1% power self-check, a reference-filter check, block >= token-by-token mean length) | The correctness gate. Killed both mutants: acceptance + 0.05 (182 subtests failed) and block h + 0.05 (379 failed) | Upstream has `test_block_verification.py` and `test_dspark_greedy_draft_accept.py`: unit checks, not a distribution test with cuts and filters | PORT: retarget at upstream's path in `test/registered/spec/dspark/`. `accept_draft_tokens(..., block_verification=)` replaces `block_verify=`, and `DraftBlockResult` needs upstream's fields (`greedy_draft`, `top_probs`). Dead rows should carry real candidates plus garbage logits (the eager path's layout), not zeros. Keep the power self-check and the mean-length comparison (via `chain_speculative_sampling_triton(block_verification=)`). Rerun both mutants on upstream's kernel |
| 12 | `TestSampledChainParity` (triton vs torch reference of this branch's kernel at V = 129280, float64-CDF draw check) | Guards this branch's kernel only | n/a | DROP with the kernel. Bring back with the adapter (#6) |
| 13 | `TestSampledGreedyDraft` (`sample_draft_block` argmax with the flag on, sampling with it off; `_BlockHead` fixture) | Draft-side routing test for item 3 | `test_dspark_draft_block.py`, `test_dspark_draft_graph.py` (a65b776963) | DROP |
| 14 | `long_prompt.py` / `spec_margin_capture.py` `--temperature` / `--top-p`, `sampling` in `long_prompt.json`, tests in `test_long_prompt.py` (3b69a95037) | Sampled long-prompt served runs; greedy by default, so existing argv is unchanged | None (analysis scripts) | KEEP-AS-IS. Needed for any served test of upstream's flags at production sampling (0.6 / 0.95) |
| 15 | Fix-up commits c6cf6af1b1, 71383dbda6, 445da1cb6a | Test statistics (zero cells, single cell) and the float64 reference CDF | Part of #11/#12 | Follow #11/#12 |

## What would be lost by dropping ours

Nothing that affects correctness, provided items 7-9 and 11 are ported. What is lost by choice:

- **The faster accept kernel** (#6): about 30-45 us of GPU time per sampled verify at bs 1, against about 60 us more
  host time as it stands. It is optional; whether it matters is unmeasured.
- **Not materializing the one-hot q:** a 2.6 MB zeros-plus-scatter per verify. It is part of #6.

## Served runs

The served a/b (off / greedy draft / block verify, -prod arm, temperature 0.6, top_p 0.95, 240 s) never produced
data. Attempt 1 (`/mnt/nvme1/sampled-accept-served-20261017`) was aborted on the off arm by the node-0 load watchdog
(free 41074 pages against a min of 41172). That abort left an orphaned server, which was killed by PID. Attempt 2
(`/mnt/nvme1/sampled-accept-served2-20261017`) was cancelled by the owner while it waited for a quiet box, before
any server started.
