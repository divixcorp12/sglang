# Prefill admissions in the RAM-tier replay: capture plan, 2026-10-16

Branch `lane-tier-replay`. This is a plan only: no code changed and nothing run, apart from read-only reads of the
2026-10-16 A/B on divix01. It answers the two pending owner decisions that `LANE_TIER_REPLAY_HANDOFF.md` ends with:
1. a served capture that logs prefill admissions;
2. a prefill share for the RAM tier.

Code citations refer to the code the A/B ran: `lane-head-ab` `d269f7aa90` (laptop worktree `sglang-nvfp4-lane-head-ab`).
Host paths are under `python/sglang/kernels/jit/csrc/moe/expert_stream/host/`.

## 0. Finding: the A/B already logged every prefill's routes

**The route log already holds the prefill routing.** The replay reads only `graph_routes` lines with phase
`target_verify` (`lane_tier_replay.py:428`). The same `stages.jsonl` also holds one eager line per streamed MoE call,
written by `Exl3StreamTrace.record` (`srt/layers/moe/exl3_stream_trace.py:605`, called from `exl3.py:833`). For
phase `extend`, each line carries the layer's full `topk_ids` as `experts` and `counts`, plus `vram_miss`, `ram_miss`
(the gather's `host_read_rows`), `read_ms`, `t` (`time.monotonic()`, the job trace's `CLOCK_MONOTONIC`),
`forward_pass_id` and `rids`.

What `ab-off/stages.jsonl` holds (`ab-on` is the same shape):
- **18 prefills, 40 layer lines each:** 10 start-up warm-up prefills (fpid 7-31, 8 to 4,096 tokens) and the 8 serving
  prompts (fpid 40-163, 548 to 1,406 tokens).
- **Layers 0-20 carry every token** (fpid 40: 1,406 tokens, 8,436 routes). Layers 21-39 carry 128 tokens: the late
  layers run only the last window (`DSV4_WINDOW = 128`, `models/deepseek_v4_layer_major.py`, "the late-layer tail").
- **A serving prompt's prefill reads 2,200-3,530 RAM rows** (sum of `ram_miss`). Per layer this is up to 235 rows, out
  of about 190 usable rows per layer (103 + 87, handoff rule 2). Example: fpid 40, layer 2: 287 routed, 266 VRAM
  misses, 180 RAM reads.
  - So one long prompt rewrites most of layers 2-20's RAM tier. That is the departure the handoff could not see.
  - Layers 21-39 read 0-48 rows each.

**Consequence: decision 1 becomes a fallback.** Section 3 adds a stage 0: T with admissions simulated from these lines,
offline on the existing A/B, with no capture. A capture is needed only if stage 0 shows the rows read are right but the
victims are wrong (section 3). The event of section 1 is the smallest change that settles the victims exactly.

## 1. The event

### Where admissions happen

Every Python-side RAM admission goes through `take_admit_slot_locked` (`ram_tier.h:1982`), from exactly two entry
points:

| entry | `ram_tier.h` | Python callers |
|---|---|---|
| `assign` | `:399` | `ensure_rows` without fills (`expert_stream.py:422`), via `exl3_ram_miss.py:472` |
| `fill_begin` | `:456`, loop at `:476` | `prefetch_rows` (the layer's prefetch, `fallback=False`, stops at a victim-less claim or at the share) and `_fill_rows` (each gather chunk's leftover, `fallback=True`), `expert_stream.py:496`/`:469` |

The callers of `ensure_rows` are:
- prefill gathers (`prefill_fills`, `expert_stream.py:2118`; chunks at `:731`);
- eager verify forwards before graph capture;
- hot-cache promotions, which put the rows they copy into RAM first (`expert_hot_cache.py:671`);
- full layers (`exl3_ram_miss.py:1083`).

None of them emits an event. Promotion of a landed pool row (`promote_landed`, `ram_tier.h:2919`) takes its victim
through `take_victim_locked(..., count_eviction=false)` at `:2949`, also untraced. Every decode demand path is already
traced (`trace_evict` at `:3012` and `:3052`).

### The change

Three host emit sites and one flag, all InstrBuild-only. They go into the existing per-group job trace, in the existing
event format:
`{"event","ns","row","gen","seq","group","a","b","c"}`, written by `JobTrace<true>` (`job_trace.h:91`) to
`events.<pid>.exl3-tier<g>.<n>.jsonl`.

| event | site | row | gen / seq | group | a | b | c |
|---|---|---|---|---|---|---|---|
| `admit` | `assign`, after `take_admit_slot_locked` | layer row | 0 / 0 | expert's home group | expert | evicted expert (-1: free slot) | slot (-1: refused, none taken) |
| `fill_admit` | `fill_begin`, per claim, plus one for the claim that stops the loop | same | 0 / 0 | same | expert | evicted (-1) | slot (-1: claim stopped) |
| `promote_evict` | `promote_landed`, after a successful `take_victim_locked` | same | 0 / 0 | owner group `g` | victim expert (-1: free) | promoted expert | delta `chain` (joins the `delta_publish` event that follows it) |

- **Trace object.** `admit` and `fill_admit` emit into `dist_.home_group(expert).trace`; `promote_evict` into the
  claimer's `trace` with `group = g`, as `pool_point` already does (`:2397`). `emit_at` reserves slots with
  `fetch_add` (`job_trace.h`, "Writers reserve distinct slots"), so the owner thread may write a group's trace.
- **Request and timestamp.** The host does not know the forward, so no new plumbing. An admission of layer row L
  joins the route-log line for (fpid, L) whose `t` is the first one after the event's `ns`. The events of one layer
  lie between the previous layer's line and that layer's line. `rids` and `forward_pass_id` come from the line.
- **Not covered.** Admissions that fall outside every `extend` window are hot-cache promotions or eager verifies,
  told apart the same way.
- **Share ownership.** It is not logged. It is derivable from the share setting and is not needed by T.

### The flag

- **`SGLANG_DEBUG_DSV41_RAM_ADMIT_TRACE = EnvBool(False)`** in `environ.py`, next to `SGLANG_DSV41_EXPERT_TRACE_PATH`
  (`:1885`). It follows `env-var-conventions`: rule 1 (in `Envs`), rule 4 (`DEBUG_`, debug-only instrumentation) and
  rule 2 (an opt-in `EnvBool(False)`).
- **Read by the C++ trace.** `JobTrace<true>`'s constructor reads it, the way it already reads
  `SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE`, into `admits_`. Each site is
  `if constexpr (Build::kMetrics) if (trace.admits()) trace.emit(...)`.
  - `JobTrace<false>::admits()` is a constexpr `false`.
  - So ProdBuild compiles every site away, and the instrumented build with the flag unset takes one untaken branch per
    admission and writes nothing new.
- **Startup refusal.** At service start, `exl3_ram_miss.py` reads `envs.SGLANG_DEBUG_DSV41_RAM_ADMIT_TRACE.get()`. It
  refuses the flag without `SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX` or in ProdBuild, so a capture cannot log silently
  nothing. This is the same shape as `SGLANG_DSV41_ROUTER_CAPTURE_PATH`'s refusal without the trace.
- **Bit-identical when unset.** No tier decision reads the flag. Outputs, counters and the existing event files are
  unchanged.
- **Tests.**
  - `test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py` (its tier harness already drives `assign` and
    `fill_begin`). With the flag unset, the job-trace file has no admit events. With it set, there is one per claim,
    whose `b` equals the displaced expert and whose count matches the eviction counter. One test covers
    `promote_evict` against `own.promote_evicted`.
  - Then a C++ mutant per site: drop the emit, or swap a and b.
  - Selection, per the run protocol: those files, then `-k admit`.
  - `ram_tier.h` changes every host module's build key, so the first launch rebuilds them cold (50-100 s per
    variant). The tests warm them first through `spawn_child`.

### Cost per prefill and trace size

- **Events.** An admission is bounded by the layer's VRAM misses. A serving prompt has 6,960-8,290 VRAM misses
  (all layers) and 2,200-3,530 RAM reads, so expect about 2,200-3,500 `fill_admit`/`admit` events per prompt (at most
  8,300).
- **Time.** Each event costs a vDSO `clock_gettime` and one 64-byte store, about 30-60 ns. That is at most 0.5 ms per
  prompt, against a 13-17 s prefill (under 0.004%).
- **Size.** At about 125 bytes per JSON line, this is 0.3-1 MB per prompt.
- **Per arm.**
  - The A/B's prefills read 31,000 rows in total: about 4 MB, at most about 13 MB.
  - `promote_evict` adds the run's early promotions: about 3,200 events, 0.4 MB.
  - No new memory: each `JobTrace` already allocates 524,288 events (`SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY`). The
    tier files hold about 26,000-30,000 events today, so the admits fit about 8x over.

## 2. The capture run (only if stage 0, section 3, says it is needed)

**Code.**
- Branch `lane-tier-capture` = `lane-head-ab` `d269f7aa90` + the one event commit, so both arms run exactly the A/B's
  code plus the trace.
- Pushed, then a private divix01 worktree `wt-lane-tier-capture`, as in `.claude/rules/divix01-run-protocol.md`.

**Arms.**
- The A/B's pair: OFF `...-mergechunk-expseg` and ON `...-mergechunk-expseg-lanehead`, one sample each, OFF first.
- T's gate is per arm and on the delta, so it needs both.
- Driver: a copy of `/mnt/nvme1/lane-head-ab-20261016/run_ab.sh` writing to `/mnt/nvme1/lane-tier-capture-<date>`,
  with two changes:
  - `SGLANG_DEBUG_DSV41_RAM_ADMIT_TRACE=1` exported for both arms;
  - `TMPDIR` under `/mnt/nvme1`.
- Unchanged from `run_ab.sh`:
  - `spec_margin_capture.py <out> --arm <arm> --no-verify-split --route-log` (the route log with its `extend` lines,
    and the job trace);
  - the CPU split `A`;
  - the quiet-box wait, `prelaunch_evict.py`, the node-0 load watchdog;
  - the `nvme3n1` 2 s sampler, `memsample.py`, `nvidia-smi` sampling;
  - the disk lock, then the GPU lock, taken by `spec_margin_capture`'s `run_locks`.
- Never production, its checkout, or port 7867.

**Prompts and duration.**
- The same 8 serving prompts, after the server's 11 warm-up prefills (both are traced and both are used).
- About 9.5 min per arm (the A/B: start 03:58:14, ready 04:04:55, done 04:07:48), so about 20 min of box time plus
  the cold JIT rebuild at the first launch.
- Warm the host modules first, without the GPU (the new tests do it), so that launch time stays close to the A/B's.

**What makes it valid** (fixed before the run):
1. **Run health.** Both arms rc 0. No OOM lines. No node-0 abort. Every `events.*.jsonl` footer shows
   `"dropped":0`.
2. **Coverage.**
   - Every `extend` route-log (fpid, layer) with `vram_miss > 0` has at least one `fill_admit` or `admit` event inside
     its window.
   - Per (prefill, layer), the count of admissions with `c >= 0` is at least `ram_miss`. It equals `ram_miss` when the
     line's reads include the prefetch claims; settle which one from `gather_rows`' stats in step 0.1.
3. **Departures explained.** `lane_tier_replay.py departures` is re-run with the new events counted as departures:
   - the untraced class, 56-59% today, must fall to 2% or less in each arm;
   - "never seen" is allowed only for rows whose last insertion predates the trace (start-up seeding).
   - This is the decisive check. If it fails, there is a departure path this plan has not found: stop and report it.
4. **Neutrality.** OFF against the 2026-10-16 OFF on `serve_compare.py`:
   - clean ms/token 95% CI contains 0;
   - VRAM hits/verify and `rows_read` within 3%;
   - same output text per prompt where the sampling is greedy.
   - The flag cannot move a decision. This catches a launch that differs in something else.
5. **Stalls.** `serve_compare.py`'s STALL flag covers decode only. Prefill windows are covered by the
   `lane_head_ab_report.py` per-prompt `prefill_nvme3n1_await_ms` (`window_await` over `<run>.nvme3n1.samples`), and
   the route log gives a second view: per-prompt `read_ms / ram_miss`, about 1.35 ms/row in the A/B; ON prompt 6 had
   10.5 s for 3,225 rows, 3.3 ms/row. Fixed rule: a prompt whose prefill await, or prefill ms/row, exceeds 2x its
   arm's median is flagged PREFILL-STALL.
   - **T is evaluated on every prompt.** Tier decisions do not depend on time, except through which prefetch rows have
     landed, and `spec_land` and `promote_evict` record that.
   - **P and E use pair-clean prompts only:** neither STALL nor PREFILL-STALL in either arm.
   - If fewer than 4 prompts are pair-clean, rerun the stalled arm once. If it stalls again, report it and stop.

## 3. How T would use it, and its gate

**Stage 0 (offline, now, no capture): T-sim on the existing A/B.**
- **0.1.** Settle what the route-log `ram_miss` counts. It is the gather's `host_read_rows`. `read-amp` found prefill
  reads to be `host_read_rows + host_fill_rows` (`DSV_OPTIMIZATION.md`, the read-amplification entry). So check in
  `expert_stream.py` whether the layer's `prefetch_rows` claims are inside `ram_miss` or outside it. The check of 0.3
  compares like with like.
- **0.2.** T-sim adds the prefill rule to `tier_replay`. At each `extend` line (in `t` order among the verify records),
  for layer L:
  - Misses: the line's experts that are neither VRAM-hot (the hot set of the preceding verify's `graph_routes`) nor in
    RAM.
  - Claim them in ascending expert order (`prefill_fills` sorts them, `expert_stream.py:2147`). The protected set is
    every routed expert of the layer.
  - Each claim takes the home group's LRU READY slot that is not hot and not protected (`take_slot_locked`,
    `:2038`), until one finds none. That is the prefetch's stop.
  - The rest are admitted in 64-row chunks with `fallback` (protected victims allowed: the earlier chunks' rows).
  - Routed RAM hits are restamped.
  - The same rule applies to the eager `target_verify` lines (fpid 3-36).
- **0.3. Check before T.**
  - Per (prefill, layer), simulated reads (in the definition 0.1 settles) against `ram_miss`: each serving prompt's
    total within ±10%, and the per-layer correlation reported.
  - A miss here means the rows read are wrong: fix the rule from the code, not by tuning.
- **0.4. T's gate, unchanged.**
  - NVMe-cold GPU lanes/verify within ±15% per arm, ON-OFF delta within ±20%, on the same pair-clean prompts 0, 1, 4,
    5, 7.
  - Per-arm error and the delta reported even if they pass. Then E (±25%) with P as fitted.
- **Outcome.**
  - T passes: no capture is needed; go to the evaluation and section 4.
  - 0.3 passes and T fails: the rows are right, the victims wrong. Take the section 2 capture.
  - 0.3 fails: the prefill rule is wrong. The capture also settles it, since `fill_admit` gives the claim order and
    stop.

**With the capture: two modes.**
- **T-replay.** Admissions and promotion victims are applied as observed, in `ns` order, together with the decode
  `evict` events. This replaces the simulated promotion at landing (rate 0.51) and the trace's own evictions outside
  the verifies.
  - It tests the decode-side rules (1-7) alone.
  - Expected: lane-level precision and recall far above today's 0.38/0.61 and 0.25/0.26.
- **T-sim.** Admissions simulated as in stage 0. The capture is then only its ground truth, per (prefill, layer):
  - admitted count within ±5%;
  - victim-set Jaccard reported.
  - T-sim is the mode the evaluation needs: C2/base and the share sweep have no observed admissions.

**The gate T must then pass.** It is the same as above, and fixed now:
- NVMe-cold GPU lanes/verify within ±15% per arm (T-replay and T-sim each);
- ON-OFF delta within ±20%;
- E within ±25%;
- P re-validated out of sample on the capture's OFF, at its original tolerances (Δstream and Δwait ±20%, layers ±1.0,
  in-sample ±10%).
- If a part misses, report why and stop, as before.

## 4. The prefill-share replay

**The share already exists as a knob, at one size, off in every arm.**
- **Flag.** `SGLANG_DSV41_ENABLE_PREFILL_SHARE` (`environ.py:2005`, `EnvBool(False)`; `Dsv41Config`,
  `dsv41_config.py:166`).
- **Setting.** At start-up it registers `_set_prefill_share` as a pre-forward observer (`exl3_ram_miss.py:1357`,
  `:1975`). For `is_extend_without_speculative` forwards it calls `set_prefill_share(PREFILL_SHARE_ROWS)`, 0 for every
  other forward.
- **Size.** `PREFILL_SHARE_ROWS = EXL3_MAX_GATHER_ROWS = 64` per layer (`exl3_ram_miss.py:788`,
  `exl3_expert_format.py:52`), a constant. The host takes any non-negative value (`set_prefill_share`,
  `ram_tier.h:557`), so a served size sweep would need only an `EnvInt` in place of the constant.
- **Recorder.** It needs the expert-distribution recorder, and the arms pass `--expert-distribution-recorder-mode`
  (`arm_env.py:359`).
- **The A/B ran with it off.** `resolved-env.json` sets no `SGLANG_DSV41_ENABLE_PREFILL_SHARE`.

How it acts (`take_admit_slot_locked`, `ram_tier.h:1982-2017`):
- **When it engages.** Once the layer holds `share` prefill-owned rows (`owned_in_row`, summed over both groups) and
  the home group's range has no free slot, the victim is the LRU owned row of that range, under every exclusion
  (READY, not hot, not protected, not filling).
- **The fill's prefetch** (`stop_at_share`, from `fill_begin` at `:480`) then stops claiming. The chunk admissions
  after it evict the share's rows.
- **No qualifying owned row** in the home range means `take_slot_locked`'s decode LRU (`:2038`). So the bound is per
  layer, not strict per group.
- **Ownership ends** when decode uses the row: `stamp_routed_locked` and `touch` via `disown_*`, and `set_hot` at
  `:1347`.

**What was measured (DSV41_REFERENCE.md section 27.9, 2026-09-25, pre-DSpark arms):**
- the 64-row share cut S by about 3 ms/step and decode NVMe rows by 4.7%;
- it cost +0.5/+0.7 s TTFT with fills, because later chunks are no longer prefetched;
- the flag was left off.

That was before the verify graph, the pool and the GPU-lane split, and it was priced by S only. Today a cold GPU lane
costs 1.47 ms of stream (P), so the trade needs repricing.

**The sweep** (offline, T-sim, only after T passes):
- **Share sizes per layer:** 0 (today), 16, 32, 64 (the existing knob), 96, 128, and "no admission" (staging only, the
  section 27.9 worst case, as a bound).
- **Variant:** the share on layers 2-20 only. Layers 21-39 admit 0-48 rows per prompt, so a share there mostly costs
  prefetch.
- **Run on:** the A/B's two arms and C2/base.

**Per share, per arm:**
- **Decode.**
  - Per verify: NVMe-cold GPU lanes, RAM-warm GPU lanes, CPU forced misses, CPU hit lanes.
  - Priced with P: stream/verify, wait/verify, and net ms/verify = Δ of the later-of(stream, CPU) model P fits (wait
    ends 1.04 ms before max(stream, CPU end)).
  - All of it by verify position after the prefill (0-4, 5-14, 15+). Section 27.9 found the early excess mostly a cold
    start that no admission policy removes, so the split shows where any gain sits.
- **Rows.** Rows saved from eviction that decode then used, against prompt rows the share evicted that decode then
  missed.
- **Prefill cost.**
  - Prefill row reads per prompt. They should not change for prompts under 4,096 tokens: each expert is gathered once
    per layer per forward, and these prompts are one chunk. Re-reads come only from multi-chunk prompts, 28-32% in
    section 27.9's soaks.
  - Prefetch-stopped rows per layer: the rows read in a blocking `_fill_rows` per chunk instead of prefetched.
  - Priced two ways: as a bound, at the route log's measured prefill rate (about 1.35 ms/row); as a check, against
    section 27.9's +0.5..+0.7 s at share 64.
- **Net per prompt.** Δms/verify × verifies, minus ΔTTFT. Reported as ms/token with and without TTFT.

**Gate for serving any size:** net ms/verify better than -2 ms in both the A/B arms and C2/base, with the TTFT cost
stated. A served A/B of the chosen size would then use the existing flag (64) or the `EnvInt` (another size).

## 5. Coordination with the drip-hot agent

The GPU drip copies RAM-mapped, not-hot rows into the hot cache (`SGLANG_DSV41_GPU_DRIP_*`, "experts that are mapped
in RAM and not VRAM-hot"). Its candidate set is the RAM tier this lane models, under the same rules. What it would get:
- **Exact RAM membership at every verify from the capture** (T-replay): decode `evict`, `admit`/`fill_admit`,
  `promote_evict` and the insertions are all traced, so membership is reconstructed with no simulation. That is the
  drip's eligible set per (verify, layer).
- **The post-prefill collapse, measured.** A long prompt rewrites most of layers 2-20's RAM rows. So the drip's
  candidates right after a prefill are the prompt's experts, not decode's, and its scorer's picks there should be
  judged against that.
- **A validated T-sim** (stage 0 or the capture) for its own counterfactual. Its own interaction: a dripped row
  becomes hot, and a hot row keeps its RAM row and cannot be a victim (rule 3). So the drip shrinks the RAM LRU pool by
  the rows it lands, and its replay must hold them out of the victims.
- **The prefill-share result.** A share keeps decode's rows across a prefill. That keeps the drip's candidates alive
  across it, so the two compound, and the sweep should be rerun with the drip once it has a replay.

I will send drip-hot the event schema and stage 0's T-sim rule once team-lead approves this plan, so its replay uses
the same rule.

## Stage 0 result (2026-10-16): T passes; no capture needed

Approved by team-lead; run offline on the existing A/B. Script commit `5508b65381` (rule 10 in `TIER_RULES`;
`gate --prefill prev|next`).

**0.1: `ram_miss` excludes the prefetch's claims.**
- The line's `ram_miss` is `host_read_rows`, the rows `_record_read` adds inside the gather's window
  (`_gather_eager_rows`, `expert_stream.py:2431`).
- `_fill_rows` records `rows=len(slots)` (`:479`); `prefetch_rows` records no read; `_wait_fill` records time only
  (`:552`).
- So `ram_miss` counts the chunk (demand) admissions only. A prefill's reads are the prefetch claims plus `ram_miss`.
- Check 0.3 compares the replay's demand admissions with `ram_miss`.

**0.3: reads check, passes.**
- Per serving prompt, the replay's demand admissions against `ram_miss`: -0.4% to -4.7% (OFF), -0.1% to -6.3% (ON).
  All 16 are within ±10%.
- Per-layer correlation 0.999 in both arms.
- The replay's VRAM misses match the lines' `vram_miss` to within 0.5%, so the hot set is right.
- The warm-up prefills also agree (for example fpid 29: 2,747 against 2,810).
- The prefetch claims a further 1,000-1,830 rows per prompt.

**0.4: gate, all parts pass.** Run with `--bootstrap 200 --prefill prev`; P is unchanged from gate-5.

| check | predicted | observed | error | tolerance |
|---|---|---|---|---|
| T: NVMe-cold GPU lanes/verify, OFF | 13.79 | 15.10 | -8.7% | ±15% |
| T: same, ON | 30.61 | 33.22 | -7.9% | ±15% |
| T: ON - OFF | +16.82 | +18.12 | -7.2% | ±20% |
| E: Δstream/verify | +50.74 ms | +51.68 ms | -1.8% | ±25% |
| E: Δwait/verify | -53.19 ms | -46.92 ms | +13.4% | ±25% |

- Lane-level precision and recall: 0.94 / 0.86 (OFF) and 0.96 / 0.88 (ON), against 0.38 / 0.25 and 0.61 / 0.26
  before.
- The replay's promotions number 2,441 / 2,259, against the counters' 3,201 (OFF).
- The remaining under-prediction is about 8%, even across the verify's position after its prefill (age 0: 232 against
  270 OFF; age 8+: 695 against 780).
- The hot-set choice (`prev`/`next`) changes nothing at the reported precision.
- Without `--prefill` the script reproduces gate-5's T (9.94 / 14.07), so the old path is unchanged.
- Outputs: `/mnt/nvme1/lane-tier-replay/stage0-prev.json` (the gate), `stage0-{prev,next}-b0.json` and
  `stage0-noprefill-b0.json`.

## Owner constraint for head v1.1 (2026-10-16, via team-lead)

- **No retraining per RAM cache size.** The owner does not want a predictor that must be retrained whenever the RAM
  cache size changes. Values come from startup calibration, not constants.
- **So option (a) is the last resort.** Option (a) is a RAM-tier feature inside the learned head.
- **The preferred form is the product ranking.**
  - The head stays tier-agnostic and outputs P(recur next verify).
  - Serving ranks GPU lanes by P(recur) × the time saved for the expert's live tier (VRAM, RAM-warm or NVMe-cold).
  - The tier costs are measured at server startup.
  - Option (b) is the hard-cap special case of this form.
- **The rule for (a).** The evaluation includes the product ranking as an arm. Option (a) is justified only if it
  beats the product by a clear margin. Its ceiling here is `oracle-emp`, an oracle P under the product's tier costs.

## Evaluation (2026-10-16): lane keys on C2 and base, priced by tier

Script `analysis/dsv41-drive/dspark/lane_eval.py` (`55bec79464`); outputs in `/mnt/nvme1/lane-tier-replay/eval/`
(`{base,c2}.json`, `{base,c2}.abprices.json`). Offline, cores 56-63.

**What each arm replays.** One joint run per arm:
- The VRAM cache under today's victim rule. The arm decides which k misses take the GPU lanes, k being the capture's
  own count per (verify, layer). The resident set is resynchronized after each prefill.
- The RAM tier (rules 1-10) on the arm's own hot sets and misses.
- Each lane priced by its tier with P.

The head's P comes from `head/{c2,base}.pred-B.npz`. Base has 121 clean verifies and C2 has 700.

### Validation

- **The capture's own lanes, priced:**
  - against prices fit on the capture: 244.8 vs measured 231.0 ms/verify (base), 243.4 vs 217.2 (C2);
  - against prices fit on the A/B OFF: 216.9 (base) and 209.4 (C2).
  - So the capture's own fit runs 6-12% high, and the A/B prices 3-5% low.
- **T on these captures:** 12.6-13.3 NVMe-cold GPU lanes/verify predicted, against 14.3-15.0 observed (-12%). Forced
  CPU misses: 75.9 against 76.2 and 79.2 against 82.0.
- **The `today` simulation** has about 20 more hits per verify than the capture (382 vs 363). This is
  vram_cache_replay's known gap. Every net figure below is against `today`.

### The check: the current head should come out at about 0 net

| prices | head, Δ ms/verify before its cost | at 0.77 ms |
|---|---|---|
| fit on the capture | -3.1 / -3.2 | -2.4 |
| fit on the A/B OFF | -0.5 / -0.2 | +0.3 / +0.6 |
| capture's fit + A/B residual (`_ab`) | -0.3 / -0.2 | +0.4 / +0.6 |

- Figures are base / C2. Served A/B: +0.74 ms/token, 95% CI -3.1 .. +4.0; stream + wait per verify +4.8.
- **The capture's own fit fails the check.** The A/B prices, and the A/B-residual correction, pass it. So those two
  views are the ones to read below.

**Why P misses here.**
- P puts the CPU path on the critical path in 96-99% of records (stream-critical share 0.039 on base, 0.007 on C2).
  So moving an NVMe-cold lane from the CPU onto the GPU stream looks nearly free to P.
- The A/B says otherwise. It added 18.12 cold GPU lanes per verify and measured stream + wait +4.76 ms/verify, where P
  predicted -2.45. That is +0.40 ms per moved lane beyond P, and +0.26 ms per lane in all.
- P's E check passed at ±25% on stream and on wait separately. Their difference, the net of a cold move, is what an
  arm that trades one against the other depends on.
- So any arm that changes the cold GPU lanes a lot is read through that served number.

### Results

Δ ms/verify against `today`. Base / C2.

| arm | hits/verify | cold GPU lanes | A/B prices, before cost | A/B prices, at 0.77 | A/B prices, at 0.40 / 0.32 | own fit + `_ab` | T bias (`_bias`, A/B prices) |
|---|---|---|---|---|---|---|---|
| today | 372 / 383 | 11.9 / 10.7 | 0 | 0 | 0 | 0 | 0 |
| head (v1, served) | 391 / 398 | 19.0 / 18.4 | -0.5 / -0.2 | +0.3 / +0.6 | -0.1 / +0.2 ; -0.2 / +0.2 | -0.3 / -0.2 | +0.3 / +0.7 |
| capb (option b) | 386 / 394 | 0.5 / 0.6 | -1.4 / -0.9 | -0.6 / -0.1 | -1.0 / -0.5 ; -1.0 / -0.6 | -3.2 / -1.7 | -2.6 / -2.0 |
| product (P's tier costs) | 354 / 363 | 67 / 66 | +7.2 / +7.1 | +7.9 / +7.8 | | +8.3 / +3.8 | +12.9 / +12.9 |
| **product-emp** (served cold cost) | 380 / 387 | 0.5 / 0.6 | **-3.9 / -3.1** | **-3.1 / -2.3** | -3.5 / -2.7 ; -3.5 / -2.8 | -5.1 / -3.2 | -5.1 / -4.2 |
| oracle (next1-k, VRAM only) | 428 / 435 | 11.6 / 11.6 | -4.9 / -4.0 | -4.9 / -4.0 | | -5.7 / -4.5 | -5.0 / -3.9 |
| oracle-emp (ceiling of (a)) | 401 / 408 | 0.6 / 0.7 | -5.1 / -4.5 | | | -6.6 / -4.9 | -6.3 / -5.6 |

Notes on the table:
- **product with P's costs fails.** It puts 66 cold lanes on the GPU, because P makes a cold CPU lane cost about 2 ms
  and a GPU stream lane almost nothing. The A/B contradicts that.
  - With the capture's own fit and no correction it reads -13.8 / -18.4. That is the failure mode, not a gain.
  - So the startup-measured tier cost must be the served one: what a moved cold lane does to stream + wait.
  - The A/B's +0.26 ms/lane is that measurement (`product-emp`).
- **`_bias`** scales every arm's NVMe-cold GPU lanes by 15.10/13.79, T's under-prediction on the A/B.
  - It moves the head +0.8 to +0.9 ms worse, because the head adds cold lanes.
  - It moves capb and product-emp 1.1-1.3 ms better, because they remove them.
  - The oracle hardly moves.
  - On these captures T's under-prediction is about 12%, not 8%, so these shifts are a lower bound.
- **What changes the cold GPU lanes.** Only capb and product-emp take them out (about 0.5/verify, against 11-12 today).
  They move about 10 forced misses per verify onto the CPU.
  - capb keeps about 70% of the head's hit gain.
  - product-emp keeps 30-40%: it gives the warm lanes to the experts with the most CPU routes, and gains its time
    there.

### What it says

1. **The served head is neutral here too:** about 0 net at 0.77 ms in both credible views. The check passes.
2. **The owner's form, with the served cold-lane cost (`product-emp`), is the best head arm.**
   - -2.3 to -3.1 ms/verify net at 0.77 (A/B prices); -2.7 to -3.5 at the fused 0.32-0.40.
   - It beats the head by about 3 ms/verify, and capb by about 2-2.5 ms.
   - Its cost: one startup-measured constant per tier, and a tier-agnostic head. No retraining per RAM size.
3. **Option (a) is not justified.** Its ceiling, an oracle P under the same tier costs (`oracle-emp`), is only
   1.2-1.6 ms/verify better than product-emp, in every view. A learned (a) would recover part of that, against its retraining
   constraint. That is not a clear margin.
4. **The lane-key ceiling is small.** The VRAM-only oracle is worth 4-5 ms/verify. The best realizable arm is at about
   3 ms/verify before the head's cost.
5. **The conclusion rests on one served number.** product-emp's advantage comes from keeping cold rows off the GPU,
   priced by the A/B's +0.26 ms per moved lane (one sample per arm, CI including 0).
   - Before any served product-emp arm, the cold-move cost should be measured at startup, the way the owner's form
     needs anyway.
   - The capture's own P cannot rank these arms.

**Best head arm for the share sweep:** product-emp.

## Prefill-share sweep (2026-10-16)

`lane_eval.py --shares` (`8b2d081f67`, and the shadowing fix after it); outputs in
`/mnt/nvme1/lane-tier-replay/sweep/{ab-off,ab-on,base,c2}.json`.

**Setup.**
- **Which arms.** The A/B arms are replayed with their own served lanes (`actual`). C2 and base are replayed with
  `today` and with product-emp, the best head arm. All are priced with the A/B OFF prices.
- **How the share is modeled.** `eager_admit` follows take_admit_slot_locked:
  - every admission is prefill-owned;
  - past `share` owned rows with no free slot, the victim is the LRU owned row;
  - with none, the prefetch stops and a demand admission falls back to decode's LRU;
  - decode's use and set_hot end ownership.
- **"none"** is staging without admission.
- **TTFT bound.** The share's added demand reads per prompt, times the capture's demand read ms per row (1.35-1.62).

### Δ decode ms/verify against share 0 (same arm, A/B prices)

| share | A/B OFF | A/B ON | base today | base product-emp | C2 today | C2 product-emp |
|---|---|---|---|---|---|---|
| 16 | -10.7 | -11.8 | -18.2 | -19.6 | -23.4 | -25.0 |
| 32 | -10.4 | -11.6 | -17.7 | -18.9 | -22.1 | -23.7 |
| 64 (the existing flag) | -5.4 | -6.3 | -10.0 | -10.1 | -13.0 | -13.6 |
| 96 | -4.0 | -4.8 | -7.5 | -7.7 | -9.1 | -9.8 |
| 128 | -0.7 | -1.0 | -2.2 | -2.2 | -2.9 | -3.1 |
| none (no admission) | -9.5 | -10.1 | -21.3 | -22.9 | -31.6 | -33.7 |
| 64 on layers 2-20 | -5.6 | -6.5 | -8.3 | -8.6 | -10.6 | -11.3 |
| 16 on layers 2-20 | -8.6 | -10.0 | -13.2 | -14.2 | -16.5 | -17.6 |

- **Where it comes from.** NVMe-cold GPU lanes per verify fall from 13.8 to 9.3 (A/B OFF, share 16), from 29.9 to 17.9
  (ON) and from 10.7 to 3.3 (C2 today). Forced CPU misses fall by 4-19 per verify. The A/B-residual correction
  (`d_ms_ab`) adds 0.1-6 ms more gain.
- **By verify position after the prefill** (0-4 / 5-14 / 15+):
  - share 16: A/B OFF -4.9 / -10.9 / -19.6; C2 today -31.4 / -23.5 / -12.3;
  - share 64: A/B OFF -5.6 / -5.5 / -4.7; C2 today -22.9 / -11.2 / -3.1;
  - none: A/B OFF **+11.5** / -13.4 / -32.8; A/B ON **+8.5** / -18.3 / -29.6; C2 -37.7 / -32.5 / -21.4.
  - On the A/B, no admission costs the first verifies, as section 27.9 found, and gains later.
  - The 15+ bins hold only 19 / 11 verifies on the A/B.
- **With the best head arm.** product-emp and the share add up, slightly better than additively. On C2, product-emp at
  share 16 against today at share 0 is -28.1 ms/verify, before the head's cost. The parts are -3.1 (head arm) and
  -23.4 (share).
- **Layers 2-20 only** is no better at 64, and worse at 16. Keep the share on every layer.

### Prefill cost

- **Rows read.** Per serving prompt at share 0: 1,200-1,310 prefetch claims and 2,800-3,340 demand reads.
- **Share 64:**
  - The prefetch stops at the share, at about 27 claims per layer on average.
  - Demand reads fall, by 330-620 per prompt, because the decode rows the share keeps are also the prompt's hits.
  - So the bound gives -0.45 to -0.94 s of TTFT.
- **Share 16** gives +0.27 to +0.46 s on A/B and base, and -0.08 s on C2.
- **No admission:** +0.17 to +0.23 s on the A/B, -0.11 to -0.78 s on base and C2.

### Caveats, against section 27.9's served share-64 result

- **Section 27.9** (2026-09-25, pre-DSpark arms, one session each): S -3 ms/step, decode NVMe rows -4.7%, and TTFT
  **+0.5 to +1.2 s**.
- **This model at 64:** -8% cold reads (78 to 72 per verify on A/B OFF) and -5 to -14 ms/verify, through the CPU
  forced-miss path, which those arms did not have.
- **TTFT disagrees in sign.** This bound counts blocking reads only. It does not model what section 27.9 measured:
  chunks after the first are no longer prefetched, so their reads stop overlapping the gather.
- **The no-admission and small-share figures** go furthest from what T validated. T was checked on the A/B, where
  every prefill admits.

**What would settle it cheaply.** A served A/B of the existing flag, `SGLANG_DSV41_ENABLE_PREFILL_SHARE=1` (share 64,
no code), on the `-mergechunk-expseg` arm, with the route log. It reads:
- decode cold lanes per verify (predicted 13.8 to 10.2 on OFF);
- stream + wait per verify (predicted -5.4);
- TTFT.

Smaller shares need `PREFILL_SHARE_ROWS` as an `EnvInt`, a small code change.

## Decisions asked of team-lead and the owner

1. **Stage 0 now** (offline, the existing A/B, `taskset -c 56-63`, outputs under `/mnt/nvme1/lane-tier-replay/`). No
   capture until it says one is needed.
2. If stage 0 asks for it: **the event commit** (sections 1 and 2; host trace only, flag off by default) **and the
   two-arm capture**, about 20 min of box time plus a cold JIT build.
3. **The share sweep** (section 4) after T passes, offline.
