# NVFP4 expert offload: enhancements after the warp-aligned copy kernel

**Status (2026-09-15):** step 1 is done (experiment log E26). Steps 2–7 are not started, and each needs its own approval.

**Re-ranked order after step 1 (predicted gain over the post-E25 baseline of 15.83 tok/s on the E1 workload):**

1. Idea 3, FP8 slots: +7.3–8.6%. Implemented; blocked on the parked GPU accuracy session.
0. **New, from E28: the residency-update stall.** Every 4th decode step stalls the host ~60–80 ms beyond a normal step (38–47% of decode wall time without a profiler); ~70 ms is unattributed host CPU. Worth roughly +25–40% if mostly removed (estimate). E30 attributed it: 76 ms p50 per update, ≈85% Python bookkeeping and synchronous tiny transfers on the decode thread. Its fixes #1–#6 (vectorized decide, one mapping publish per layer, batched slot writes, pinned plan buffers, cached copy validation, async promotions) are estimated at ≈50 ms per update, ≈+20% tok/s. **Direction chosen 2026-09-15: move the whole update into the decode CUDA graph instead.** The steps: device-side decide (the swap loop is a sorted prefix, so it can be exactly equivalent), fixed-K promotions per layer, device scatter of mapping and slot state, promotion copies through the in-graph copy kernel into hot slots, and a device forward counter as the update trigger. It sits behind `SGLANG_MOE_GPU_RESIDENCY_UPDATE`, and the Python path stays as reference. Estimated ≈+30% tok/s, with ~14 ms of promotion PCIe per update remaining on the GPU stream. Branch `cc/residency-update-fast`. **Measured (E31, `252a888559`):** +22.3% tok/s (14.87 → 18.18), update-step p50 118 → 65 ms, hit rate unchanged; the Python fixes alone gave +15.8%. Merged `d43c59b58a` and deployed 2026-09-15; normal-step +1.7 ms is unresolved.
2. Idea 2, overlapping next-layer copies: E27 found no copy/compute contention, so the saving is min(copy, window)/copy per layer. E28 measured the real window at 0.268 ms, re-pricing the ceiling to ~10 ms/token (+15–17%). The CPU doorbell prototype (E29, `cc/doorbell-prototype` `d57db3e1a1`) was held behind later graph replays unless the host gated each layer (74 vs 40 ms/token). E32 (`696b8cc96b`) found the cause: the thread's own `cudaStreamCreate` stream. On a torch-created stream the hold is gone (0 timeouts, byte-exact) and the thread beats the in-graph kernel by 4–7% at 1–10 rows, but the estimated gain over the in-graph kernel is ≈2.4 ms/token, below the 3 ms go threshold.
3. Idea 4, skipping low-weight misses: +7.6–11.9% at 20–30% fewer misses. Measure the gate-weight distribution of misses first.
4. Idea 6, small items: about +4% each.
5. Idea 5, CPU experts: −7% at 4 threads; +14% at 8 threads only if scaling is linear.
6. Idea 7, hardware: Gen4 +27%, Gen5 +47%.

**Context:** `8473a28c88` made the in-graph miss copy 89% of the PCIe Gen3 link (E24) and saved 17.1–17.3 ms/token in serving, 13.4 → 17.4 tok/s median (E25). Every estimate in the experiment log before E25 priced a miss at the old kernel's cost, so the ranking of the remaining ideas is stale.

## Where a token goes after E25 (estimates, production settings, E25 workload)

| Part | Before E25 | After E25 |
|---|---|---|
| Expert miss copies (0.288 GiB/forward) | ~42 ms | ~25 ms; the link floor for these bytes is ~22 ms |
| Everything else (non-expert and expert compute, promotions) | ~33 ms | ~33 ms |
| Total | ~75 ms | ~58 ms |

A faster copy kernel can now gain about 3 ms at most. The remaining levers are fewer miss bytes, overlapping copies with compute, or a faster link.

## Ideas, in recommended order

### 1. Re-rank the other ideas at the post-E25 copy cost (offline)

- **What:** refit the tok/s model from E25's measured pairs and rerun the E20 replay outputs through it. Re-derive the expected gain of ideas 3–5 (and prefetch, E2) at the new cost.
- **Why first:** it is CPU-only and takes minutes, and it decides whether ideas 2–5 are still worth building.
- **Done when:** the experiment log has an entry with the new fit (with its derivation), a table of predicted tok/s per idea under the old and new models, and a re-ranked order.

### 2. Copy the next layer's predicted misses while the current layer computes

- **What:** predict layer L+1's routing from layer L's hidden state and launch the predicted misses' copy concurrently with layer L's compute. Correct mispredictions with the normal miss path.
- **Upside:** a large share of the ~25 ms of copies, if predictions are good and the copy doesn't slow compute (estimate).
- **Main risk:** the copy kernel runs on GPU threads that also run compute, so the two may contend. E22 never measured the kernel under compute load.
- **First step:** a microbenchmark of the copy kernel beside a real MoE decode load. If it contends heavily, drop this idea.
- **Also needs:** a layer-ahead routing prediction accuracy measurement, and whether a second stream works inside the decode CUDA graph.

### 3. Bigger GPU expert cache through FP8 non-expert weights (E20, parked)

- **What:** deploy `SGLANG_ONLINE_FP8_GROUPS` (committed in `fa66acb84f`) to free 1,279–1,509 cache slots at 14 GiB.
- **Upside:** E20 predicted +10.6–12.5% at the old copy cost. Step 1 re-derives it.
- **Blocker:** the GPU accuracy session (phase A, about 70 min) is parked.

### 4. Skip or defer low-weight routed experts

- **What:** for uncached experts whose gate weight is below a threshold, either drop them (renormalizing the rest) or use them one layer late, like kt-kernel's deferred experts.
- **Upside:** each avoided miss saves about 0.23 ms at the post-E25 cost (estimate), so 30% fewer misses is about 8 ms.
- **Risk:** it changes model output, so it needs a quality gate (logprob drift against unmodified output, then a small eval).
- **First step:** measure the gate-weight distribution of misses from a routing trace, to see how many misses carry little weight.

### 5. Run misses on the CPU (E21)

- **What:** compute cache misses with `kt_kernel`'s NVFP4 CPU kernel instead of copying them.
- **Status change:** E21's 25.8 ms for 130 misses at 4 threads beat ~44 ms of old copies. Against ~25 ms of post-E25 copies it is roughly a tie at 4 threads.
- **Condition:** it only wins with more cores (ik_llama scaled well at 18 threads), which conflicts with the ≤4-thread rule on divix01.

### 6. Small items

- **Last 11% to the link:** about 3 ms; where it goes has not been isolated.
- **Promotions (~7.6 ms, already on DMA):** batching or fewer promotions might save 1–3 ms.
- **Concurrent traffic:** the new kernel is untested with more than one request. More requests means more distinct experts per step, which uses the kernel's more-rows-than-warps branch.

### 7. Faster host link (hardware)

- **What:** divix01's Xeon Gold 6154 CPUs support only PCIe Gen3; the RTX 5090 supports Gen5.
- **Upside:** a Gen4/Gen5 host gives 2–4× the link, so copies would drop to roughly 6–11 ms (estimate). No software idea here comes close.

## Not speed, but open

- **Stock-server nondeterminism:** an unmodified-kernel run flipped a token by 6.5 nats two tokens after prefill (E25). It may affect output quality and has its own investigation owed.
