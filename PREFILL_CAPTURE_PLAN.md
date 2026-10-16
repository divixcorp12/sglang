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

## Decisions asked of team-lead and the owner

1. **Stage 0 now** (offline, the existing A/B, `taskset -c 56-63`, outputs under `/mnt/nvme1/lane-tier-replay/`). No
   capture until it says one is needed.
2. If stage 0 asks for it: **the event commit** (sections 1 and 2; host trace only, flag off by default) **and the
   two-arm capture**, about 20 min of box time plus a cold JIT build.
3. **The share sweep** (section 4) after T passes, offline.
