# DSV41 RAM prefetch, Phase 2: the gate scorer on the GPU

Status: approved in conversation 2026-10-09, pending review of this document.
Builds on: `docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md` (Phase 1) and branch
`codex/dsv41-ram-prefetch-margin` (top-k-only option, per-pick rank and margin).

## Why

Phase 1's speculative thread scores layer T+1's gate on the CPU when it handles layer T's record. The instrumented
capture `divix01:/data/models/slang/nvfp4-work/ram-prefetch/margin-20261009-002539` (8 sessions, CPU scorer) shows
that this arrives too late:

- Scoring costs ~1.8 ms per record (`spec_score_ns / spec_scored` = 18.3 s / 10,056) before any read is issued.
- A speculative read takes 2.0 ms (p50, issue to land); it lands a median of 0.4 ms before its demand.
- 992 of 2,373 used reads (42%) were still in flight when the demand arrived (`spec_promoted`).

On the GPU, layer T's MoE input is resident when its record is posted, and layer T+1's gate is a small GEMV
(384 experts x hidden x up to 6 tokens). Scoring there removes the CPU scoring time and starts the read as soon as
the host learns of the record.

## Decisions taken in conversation

- **Approach A:** a separate scoring step right after layer T's post, on the same stream, inside the captured graph.
  Not before the post (it would delay layer T's CPU demand work), not on a forked side stream (only if A's measured
  cost is not negligible).
- **The GPU also filters:** it skips experts VRAM-hot for layer T+1 or mapped in RAM (the device `ram_slot` map). The
  host keeps the pool check and re-checks the map.
- **Every record predicts:** in GPU mode a job is made for every record whose row has a target, not only records with
  a CPU lane.
- **The CPU scorer stays** the default and the fallback; an option selects the GPU scorer.

## Design

### The option

`SGLANG_DSV41_RAM_PREFETCH_SCORER` in `environ.py`: `cpu` (default) or `gpu`. `gpu` is refused at start without
`SGLANG_DSV41_RAM_PREFETCH`, and without the copy-engine path whose post is captured in the decode graph. All other
prefetch options (`PER_TOKEN`, `PER_LAYER`, `SPEC_SHARE`, `TOP_K_ONLY`) mean the same in both modes.

### The scoring kernels (`expert_stream/spec_score.cuh`, new)

Two launches after layer T's post, captured in the graph with it:

1. **Score**, multi-block. Each block takes a slice of the experts and computes `sqrt(softplus(w_e . x_t)) + b_e` for
   every live token, fp32 accumulation, into a device scratch `[tokens_max, experts]` fp32. `x` is layer T's MoE input
   (bf16, as the post reads it); `w` and `b` are layer T+1's live registry tensors (`ram_prefetch.registered_gates`).
   Templated on `w`'s dtype (bf16, or fp32 under `router_fp32`), so no device copy of the gates is made. A NaN score
   ranks below every other.
2. **Select**, one block. Builds the skip mask from layer T+1's hot-slot row and `ram_slot[T+1, e] >= 0`. Per live
   token: its top `kDepth` = 12 by score (ties to the lower id); walks them in order, all 12 or the first `top_k` under
   `TOP_K_ONLY`, passing over skipped experts, picking the first `per_token`, each with its margin to the token's
   `top_k`-th score (equal scores give 0). Merges the picks by best margin (ties to the lower id), each with the rank of
   the token that gave its best margin (the lowest on a tie), and writes up to `kMaxCandidates` = 8 to the candidate
   page slot of the record's seq.

The ranking rule is `GateScorer::choose`'s, so its tests are the reference. The live token count is the one the post
writes for the record. A record with more live tokens than `tokens_max` (prefill) publishes count 0.

### The candidate page (pinned host memory, new)

`kCandRecords` = 16 slots (= `Wire::kDemandRecords`, as the hot page), slot `(seq - 1) % 16`. Each slot: a header
(u32 seq, u16 count, u16 flags: 8 bytes) then `kMaxCandidates` 8-byte entries (u16 expert, u8 rank, a pad byte, fp32
margin): 72 bytes, padded to a 128-byte stride.
The select kernel stores the payload, a system release fence, then the seq word with a release store, as the post does
for the hot page. The host reads it as `read_gpu_hot` does: acquire-load the seq, compare to the job's, copy, acquire
fence, re-load and compare.

### The host in GPU mode (`ram_tier.h`)

- **Jobs.** `offer_spec` pushes `SpecJob{seq, row}` for every non-torn record whose row has a target; no staged lane or
  token table is needed. CPU mode is unchanged.
- **Waiting.** `serve_spec_job` reads the slot for `job.seq`; while the seq word does not match it spins briefly, then
  sleeps in short steps (the speculative thread can share a core with the polling RAM thread, node 1's core 35), up to
  `kCandWait` = 200 us. A slot not ready in time counts `spec_late` (a new core counter) and reads nothing. A lapped
  slot or a count of 0 counts `spec_dropped`.
- **Filtering and budget.** Both groups' threads read the same slot and apply the same filters: mapped in the host map
  now, or pooled before (`pooled_before`). Each takes the first `per_layer` that pass, as one list over both groups, and
  reads only its own group's (`Wire::home`). The layer's budget over both groups holds as in Phase 1.
- **Unchanged:** staleness (`spec_stale`), the reader's turn and demand priority, the pool, promotion, swap-on-use.
  `spec_read` gets the GPU's rank and margin for the `spec_submit` event's `gen`, as the CPU scorer's.
- **Metrics (InstrBuild):** in GPU mode `spec_scored` / `spec_score_ns` count the host's wait for the slot.

### Setup

At `_enable_ram_prefetch` with `gpu`: allocate the pinned candidate page and the device score scratch; build a per-row
device table (layer T+1's gate weight and bias pointers and dtype, target row, its hot-slot row, its `ram_slot` row);
pass the page to the host through `enable_ram_prefetch`. `Exl3RamMissRowBackend.post` launches the two kernels after
`side.post` with the row's table entry. Their parameters are frozen at capture, so the table is built before the decode
graph is captured; the plan verifies this ordering. New JIT modules are loaded in warm-up, never during capture.

## Testing

1. **Kernel parity (GPU):** on the integer-exact cases of `test_exl3_ram_prefetch_scorer.py`, the GPU candidates equal
   the CPU reference's (experts, ranks, margins) for every `top_k` / `per_token` / `per_layer` / `TOP_K_ONLY`
   combination, ties, and hot and mapped skips; on random bf16 data the top candidate agrees with a torch reference in
   >= 99% of cases.
2. **Publication (GPU):** the seq word is written last; an oversize record publishes count 0; 17 consecutive records
   show the seqlock detecting the lapped slot.
3. **Host consumption (CPU),** with a fixture that writes slots as the kernel does: a ready slot is read and filtered
   by the host map and the pool; a late slot counts `spec_late`; a lapped slot drops; both groups compute the same list
   and read their own; the budget holds; CPU mode is unchanged.
4. **Graph (GPU):** a captured post plus scoring, replayed, yields the eager candidates.
5. **Option and wiring:** `gpu` without the prefetch is refused; the service builds the table; the A/B driver's
   `dspark-both-prefetch-gpu` arm differs from `dspark-both-prefetch` in the scorer alone.

Regression: `test/registered/unit/kernels/test_*exl3*.py` and `test_*expert*.py`, each directory in its own pytest
run; the hot-path golden test pins the option-off path.

## Measurement (divix01, under the locks; production stays stopped)

1. A graph-mode Nsight trace of a short decode window: the two kernels' time per layer.
2. `spec_margin_capture.py` with the GPU scorer: hit rate by rank and margin, the host's wait, read, lead and slack,
   against the CPU-scored capture `margin-20261009-002539`.
3. An A/B with the arm order reversed: `dspark-both` against `dspark-both-prefetch-gpu` (and the top-k-only variant if
   time allows), judged by the median and by the paired per-session gain.

## Success criteria

- The two kernels cost <= 0.1 ms per layer.
- Fewer than 15% of used reads are still in flight at their demand (`spec_promoted / spec_used`, against 42%).
- The A/B is faster both by the median and paired per session, with the text within the near-tie band.

## Out of scope

Predicting further than one layer ahead; a side-stream fork; per-NUMA-group gate copies; a margin threshold option
(the margin table suggests one; it is a separate change).
