# DeepSeek V4.1 Flash — scoping reference

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B-FP8 \
    --tp-size 1 \
    --kv-cache-dtype nvfp4 \
    --prefill-kv-cache-dequant-dtype nvfp4 \
    --page-size 16

    --reasoning-parser auto \
    --tool-call-parser auto \
+   --enable-hierarchical-cache \
+   --hicache-ratio 2 \
+   --hicache-size 0 \
+   --hicache-write-policy write_through \

```
Production starts through a short wrapper on divix01, which runs the real launcher from the prod checkout.

- Wrapper (what you start, per run_server.md D5): divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live/launch.sh. It only execs the real launcher in the prod checkout. The old standalone version is saved beside it as launch-before-0925-consolidation.sh.
- Real launcher: benchmarks/dsv41_baseline/launch_prod.sh in the repo. On divix01 it runs from the prod checkout, /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod/benchmarks/dsv41_baseline/launch_prod.sh. It takes cc-gpu.lock, pins the server cores and starts sglang.launch_server.
- Settings: benchmarks/dsv41_baseline/arm_env.py. The launcher has no flags of its own: the environment comes from base_env() and the command-line arguments from ServerArgs.prod(). To change a production setting, edit arm_env.py, not the launcher.
```

**Current status (2026-09-23):** DSV4.1 serves on divix01 from
`master`; the latest measured code commit is `e36fa2530c`.
The saved production launcher uses
DIRECT stage-2 GPU expert insertion, RAM-miss leases, eight row-packing workers, the
fused expert planner, and io_uring Engram host nodes for **both** Engram layers.
Batch-1 decode has one captured
CUDA graph segment with zero Engram breaks (§22). The current HTTP benchmark uses a
32,768-token context, prefix caching, and **two timed sessions** (§23); older eight-session
and Engine-path results are historical measurements of different recipes. A node-level
decode profile found MoE wait and pinned-row copy dominant, with Engram callbacks small
(§23). The production server on port 7867 is currently stopped at the owner's request.

**Update (2026-09-24):** RAM-miss piece streaming and a NUMA-placed 100 GiB pinned tier
are merged into `master`. Two-phase and piece streaming are
recipe defaults, and the tier is 60 GiB on node 0 plus 40 GiB on node 1 (§23.1, §24.6).
- **Piece streaming:** with two-phase on in both arms, it won 8 of 8 paired sessions
  (p=0.0039) with byte-identical output, a gain of about **40 ms/token**.
- **The 100 GiB tier:** it moved the decode bottleneck to the host-to-GPU link (§24.6).
- §24 also lists the next decode work.

Sections 1 to 15 preserve the original September 18 scoping. Sections 16 to 22 record
dated experiments; **§23 is the current recipe and progress ledger**, with §24 its
2026-09-24 addendum. Together they supersede earlier present-tense plans or defaults
where they disagree.
This doc records what the model is, what it costs in bytes, what exists upstream / in
exllamav3 / in our fork, and what has to be built to serve it with our expert-streaming
stack on divix01. Companion to [`MOE_EXPERT_TRANSFER.md`](MOE_EXPERT_TRANSFER.md), whose
rule still governs: **on a saturated link, only bytes/token matter.**

Every number is tagged **[measured]** (read from checkpoint headers, files or sysfs),
**[source]** (quoted from code, docs or the tech report), or **[estimate]** (derived;
needs a measurement before anyone quotes it).

**Original owner decisions (2026-09-18; current layout is in §23):**
- EXL3 3.0 bpw experts.
- **DSpark in scope.**
- **~90 GB host RAM**, used as a *cache* tier.
- Expert weights and Engram tables read from NVMe with our io_uring reader.
- The original plan was to move experts to `/mnt/nvme1` (Gen3 x4) and keep Engram
  tables on `/mnt/nvme2` (Gen3 x2). The live recipe instead reads the EXL3 shards on
  `/mnt/nvme2` and uses expert-row mirrors on `/mnt/nvme0` and `/mnt/nvme4`.

The original design follows in §9 and its then-open decisions are in §12. Current
settings and remaining measurements are in §23.

---

## Running the benchmark server

The served-path arms go through `benchmarks/dsv41_baseline/run_arm.sh`, which takes
`cc-gpu.lock` itself. One arm includes cold server startup, a readiness gate (SM clock
stability + JIT settling), **two timed sessions** of up to 128 generated tokens, and a
discarded warm-up session. Usage is `run_arm.sh <arm_name> <port> [KEY=VAL ...]`,
where the `KEY=VAL` overrides are layered onto `arm_env.py`'s base recipe **and verified
afterwards against the live server's `/proc/<pid>/environ`**.

The saved production port is 7867; the server is currently stopped. Use a separate
benchmark port (7878 in the latest run).

**1. Laptop: commit and push.** The harness refuses a dirty tree, so unpushed work
cannot be run. Never copy a tree to divix01 by other means
(`.claude/rules/divix01-run-protocol.md`).

```bash
git push origin master
```

**2. divix01: move the serving checkout onto the new commit.** Since 2026-09-25 there is
one branch, `master`, on one remote, `origin` = GitHub `divixcorp12/sglang`; the divix01
bare repo (`remotes/sglang-nvfp4.git`, remote `shared`) and `codex/nvfp4-expert-stream-main`
are retired. The production and benchmark checkout is `dsv41-direct-prod`:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod \
  && git pull --ff-only origin master && git log -1 --oneline'
```

The `dsv41-direct-prod` checkout tracks `origin/master`. The full production procedure
(publish, check, update, dry-run, start, health, stop) is `run_server.md`, section
"DSV4.1 production".
Older linked diagnostic worktrees may be detached; inspect each checkout before updating
it. Do not assume the old `wt-p1bench` path is the benchmark target.

**3. Register the code generation.** Any change under `python/` is a new generation, and
the gate refuses an unregistered tree so a new generation can never be silently compared
against an old one. `generations.json` lives untracked in the worktree.

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod/benchmarks/dsv41_baseline \
  && PYTHONPATH=$PWD:$PWD/../../python:$PWD/../../scripts/dsv41 OMP_NUM_THREADS=8 taskset -c 0-63 \
     /data/models/slang/.venv/bin/python -c "
import generations, subprocess
tree = subprocess.check_output([\"git\",\"rev-parse\",\"HEAD:python\"], text=True).strip()
generations.register(tree, \"<short-label>\")
print(tree)"'
```

**4. Pre-build any JIT native extension before the arm, not during it.** `run_arm.sh`
aborts hard on a compile event landing mid-session, and `torch.utils.cpp_extension.load`
builds lazily on first call — i.e. inside decode. Force the build first. For the engram
host node (`SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING`):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod && OMP_NUM_THREADS=8 taskset -c 0-7,16-17,36-53 \
  PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "
from sglang.srt.layers.engram_host_node import native_engram_host_node
print(\"built\", native_engram_host_node())"'
```

A compile error for `liburing.h` indicates missing headers; a `-luring` link failure
indicates the library or link path is missing.

**5. Run the arm.**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod/benchmarks/dsv41_baseline \
  && EXPECT_SHA=$(git rev-parse HEAD) \
     ./run_arm.sh dsv41-current 7878'
```

`EXPECT_SHA` pins the tree: the arm refuses if the worktree is not at that exact commit.
Expert-row mirroring, Engram host-node io_uring, leases, DIRECT insertion, eight
pack workers, and the fused expert planner need no override — they are in
`base_env()`. Note that mirroring
applies to *expert rows* (`SGLANG_MOE_EXPERT_MIRROR_DIRS`), not to the Engram tables,
which are read from `SGLANG_DSV41_ENGRAM_TABLE_DIR` on `/mnt/nvme2` and are unmirrored.

### What to compare against

Section 20's served-path cells are historical comparators from an older launch. They
cannot serve as a matched baseline for the current recipe:

| arm shape | token-weighted | mean |
|---|---:|---:|
| default (mirrors on, leases off) | **2.741** | 2.775 |
| `SGLANG_MOE_EXPERT_MIRROR_DIRS=` (mirrors off) | **2.102** | 2.003 |

> **Both cells used `SGLANG_MOE_PINNED_HOST_MB=71680`; the current harness uses 51200
> and also changes DIRECT insertion, leases, row workers, context length, and prefix
> caching.** The old budget could no longer start on divix01 — the
> pinned buffer alone exceeded NUMA node 0's free memory, so the server exhausted node 0
> and spun in direct compaction until the 900s abort (three arms lost this way; see
> `analysis/engram-sync/HANG-FINDINGS.md`). Host pinning is a recorded constant of the
> recipe, so an arm run with the current settings is **not** directly comparable to
> these numbers. Re-baseline with matching settings before reading a delta.

Do **not** compare a current served-path arm against §19's 3.905-3.933 or §17's 2.781.
Those are Engine-path cells. The **older §20 recipe** measured served/Engine ratios of
0.69-0.71 with mirrors both on and off; that ratio has not been established for the
current DIRECT, eight-worker, 32,768-token recipe.

### Knobs worth knowing

| var | effect |
|---|---|
| `EXPECT_SHA` | refuse unless the worktree is at this commit |
| `DSV41_MAX_SESSIONS=N` | diagnostic cutoff before the default two sessions finish; `N=1` fails the two-record result gate by design |
| `NSYS_TRACE=1` | wrap the server in Nsight; always graph-mode (`--cuda-graph-trace=graph`), so its kernel table omits the graph body (`CLAUDE.md`) |
| `NSYS_SAMPLE=process-tree` | enable CPU sampling — required, or `--cudabacktrace`/`--python-backtrace` are silently inert |
| `NSYS_LAUNCH_ARGS=...` | extra args for `nsys launch`; application-scope flags go here, not on `nsys start` |

The decode flags the EXL3 path requires — `--cuda-graph-backend-decode breakable`,
`--cuda-graph-bs-decode 1`, `--cuda-graph-max-bs-decode 1`,
`--cuda-graph-backend-prefill disabled` — are already in `arm_env.py:134-141`. They are
not overrides and should not be passed on the command line.

---

## TL;DR — original scope, with current corrections

1. **The model is much bigger than Qwen3.8, in every byte dimension that matters.**
   552B backbone + 196B Engram params [source]. The EXL3 routed expert is
   **13,315,596 B** [measured], 4.8x the Qwen3.8 NVFP4 row. Routed experts total
   204.5 GB and the Engram tables 203 GB [measured].
2. **The original design was three tiers: NVMe → ~90 GB host RAM → VRAM hot cache**
   (§9). The current recipe allocates 50 GiB to the pinned MoE tier, 5 GiB to the
   native Engram row cache, and 14 GiB to the GPU expert hot cache (§23).
   - Experts are read **straight from the original EXL3 shards**, with one aligned read
     per expert and no on-disk re-layout. Every expert's 12 tensors are byte-contiguous,
     and none spans a file [measured].
   - **A raw row is not a kernel-ready slot.** `w1.trellis` starts at byte 14,852
     of the row, which is not 16 B-aligned, and exllamav3 loads trellis with 128-bit
     `cp.async`. The original slot-layout decision and its implementation are recorded
     in §§9.2 and 16–17.
3. **A RAM miss inside CUDA-graph decode was the main implementation problem.**
   - Four mechanisms were compared in §9.3. **Option C (CPU io_uring thread + device-side
     wait) was built and measured in Phase 3b (§17):** 2.781 tok/s under breakable CUDA
     graphs against 1.664 eager on the same four cold sessions. Caveats: R4 fails against
     the eager loop (fused-kernel numerics, §17.4), and graph gather's scratch cuts the hot
     cache to 888 slots against the eager run's 1,128 (§17.6). Current DIRECT insertion
     removes that scratch and provides 1,128 resident slots (§23).
   - Each needs a bounded wait and a defined failure path; a stuck NVMe read must not
     wedge the stream.
   - For experts the NVMe read is on the critical path whatever the mechanism (the
     route is only known at the layer). For Engram layer 1 it can overlap layer 0, but
     only with a split-submit or device-poll design.
4. **Drives.** `/mnt/nvme2` is Gen3 x2, ~1.9 GB/s, ~7 ms per expert [measured link].
   `/mnt/nvme1` is Gen3 x4, ~3.4 ms per expert on an *idle* drive [estimate]. It is not
   idle: an `op-reth` node's datadir lives on it, alongside other workloads, and the
   P310 is a DRAM-less QLC drive. The live EXL3 source remains on `/mnt/nvme2`, with
   row mirrors on `/mnt/nvme0` and `/mnt/nvme4` (§23).
5. **Five GB was enough in the original Engram corpus simulation.** Exact-LRU over
   1.5M tokens puts the ceiling at 71.70% (unique set 5.38 GB); 5 GB already reaches
   71.68%, 2 GB reaches 66.30%, 1 GB reaches 60.82%, and 35.18% of accesses hit within
   their own session. Larger budgets bought almost nothing **on that corpus** (§5).
   A proposed 20 GiB pinned native cache is still unimplemented and has no live
   throughput measurement (§23).
6. **DSpark changes the budget:**
   - Resident draft weights (6.75 GiB) cost ~544 target-expert slots, **~31% of the
     VRAM hot cache**.
   - Its 6-token verify makes **Stage B (#14) mandatory**: per-layer miss scratch would
     be 17.9 GiB. Stage B is on our mainline since 2026-09-18 (`797be6f678`, merged into
     `master` @ `b59edf2dc4`).
   - On the Qwen stand-in, speculation moves ~35% more expert bytes per accepted token at
     α=0.7, with break-even at α≈0.84–0.93.
   - **Rule: ship DSpark only if measured α clears the DSV4.1 break-even** (§10, §12).
   - **It did not run on the EXL3 stack in the September 19 assessment** (§18.5): the full-model dir
     has no draft, the draft loader has no EXL3 path, and the EXL3 streamer refuses the
     draft's 128-expert layers.
7. **Throughput envelope** [estimate]: ~3–8 tok/s, link plus NVMe only, before compute
   and slot-admission cost (§9.4). It depends on the VRAM miss rate `G` and the fraction `f`
   of those misses that also miss RAM. **Measured in Phase 3a (§16.8):** `G` ≈ 115 and
   `f` ≈ 18.5% on the full model in eager mode, which puts the I/O-only model at ~3.6
   tok/s on nvme2; the measured eager rate is 1.6 tok/s (§16.12).
8. **Upstream SGLang `dsv4.1` is a V4 extension, not a new model**, and touches none of
   our streaming files. Branch `dsv41` off the mainline and *merge* `dsv4.1` (a squashed
   cherry-pick leaves 29 of 110 files unapplied; the merge conflicts in 12)
   (§6). **Done**, twice over: the pre-squash PR branch merged first (`c64b2bd653`),
   then upstream's final squash `a6cf05817f` (#38798) landed via an `origin/main` merge
   (`c55f1572b0`, 2026-09-18, §6.1) — the streaming-file claim was reverified against
   both. The current fork now runs EXL3 in SGLang; exllamav3 (MIT) supplies its fused,
   sm_120-tuned EXL3 MoE kernel (§§17, 23).
9. **The September 19 option-C trace split a decode step into NVMe wait, PCIe gather,
   and compute** (§18.2; 391 ms/step under node-level tracing): 190 ms waiting for NVMe reads
   (10.2 ms per row), 128 ms gathering rows over PCIe (1.06 ms per row), 17 ms of
   compute and ~11 ms in the two Engram breaks, all serialized. The remaining ~41 ms/step
   in that window is one outlier residency-boundary stall (2.1 s). On the corpus sessions a
   boundary costs ~40–80 ms per 32 tokens (<1%) and its promotions pay for themselves:
   16% fewer VRAM misses, +3.4% tok/s against no decode boundaries (§18.6).
   Overlapping copies with compute is worth at most ~4%. Prefetch with the
   next layer's gate catches 48% of NVMe rows at top-6 but wastes 2.5 reads per useful
   one; a confidence-gated set is estimated at ~6% (one layer ahead) to ~10% (two) of the
   step (§18.4). The previous token's routes, the predictor 3b used, catch none.
   The newer graph-node profile and MoE service breakdown are in §23; use those to
   prioritize current decode work.
10. **A benchmark arm needs exclusive GPU use.** The harness checks GPU tenancy and
    takes `cc-gpu.lock`; it will not start while the production server is on port 7867.

---

## 1. What is on divix01

| Path | Contents | Size |
|---|---|---|
| `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/` | Complete EXL3 export, 51 files: 41 shards, `config.json`, `quantization_config.json`, `model.safetensors.index.json` (192,452 tensors), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, tech report, model card | Index `total_size` 219,307,470,008 B; files on disk 219,267,406,428 B (204.2 GiB) [measured] |
| `/mnt/nvme2/DeepSeek-V4.1-Flash/model-0004{7,8}-of-00048.safetensors` | **The two Engram tables**, one module per shard. Required by the EXL3 card: EXL3 only quantized the Engram `wkv` linears | 101.5 GB each [measured] |
| `/mnt/nvme2/DeepSeek-V4.1-Flash/.cache/huggingface/download/*.incomplete` | Two stale partial blobs from an earlier attempt | ~167 GB, **reclaimable** (not deleted) |
| `/mnt/nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be…/` | HF-cache snapshot: `config.json`, `README.md`, `inference/` (reference code incl. `engram.py`), `encoding/`, tech report, shards 1–2 | 2.2 GB |
| `/mnt/nvme2/nvfp4-work/benchmarks/full/{sessions,results}.jsonl` | Our benchmark corpus: 2,674 ConvFinQA/FinanceBench sessions with generated completions. Used for the Engram simulation | — |

The other 46 official shards (backbone FP8 + MXFP4 experts) are **not** downloaded.

### Host [measured 2026-09-18]

- 188 GB RAM. 64 GB swapfile on `/mnt/nvme4`, with 30–60 GB in use at different times
  that day.
- Co-tenants besides production include `op-reth` (datadir on nvme1), reth, op-node and
  QuestDB, so the DSV4.1 server gets a ~90 GB budget, not the whole machine.
- RTX 5090 32,607 MiB. The whole host is Gen3; see MOE_EXPERT_TRANSFER, "The link is at
  line rate".

### NVMe drives [measured, sysfs]

| Mount | Drive | Link now | Max | Free | Role / co-tenants |
|---|---|---|---|---|---|
| `/mnt/nvme2` | Samsung 990 EVO Plus 2 TB | **Gen3 x2** (~1.9 GB/s) | Gen5 x4 | 219 GB | **Engram tables.** Production's PLE cache also reads from it |
| `/mnt/nvme1` | Crucial P310 4 TB (DRAM-less QLC) | Gen3 x4 (~3.9 GB/s) | Gen4 x4 | 1.7 TB | **Experts (after the copy).** Shared with `op-reth` (write load bursty: 50–1,800 IOPS observed), `questdb-import`, `inductor_cache`, `okx_backfill_temp` |
| `/mnt/nvme0` | Samsung 990 EVO Plus 2 TB | Gen3 x4 | Gen5 x4 | 518 GB | — |
| `/mnt/nvme4` | SPCC 2 TB | Gen3 x4 | Gen4 x4 | 623 GB | Holds the swapfile |

The same Samsung model negotiates x4 in `nvme0`, so nvme2's x2 is a property of the slot
or its wiring.

**The expert copy.** Reading 205 GB off nvme2 takes ~2 min at line rate (1.9 GB/s). The P310's
sustained QLC write rate after its SLC cache fills is unknown and could make the copy
far slower. Schedule it when production is down, or throttle it, because it contends
with production's PLE reads.

---

## 2. Architecture

Reference: `inference/model.py`, `inference/engram.py` in the official repo, and the
tech report, *"Pushing the Limits of KV Cache Compression"*.

| Property | Value |
|---|---|
| Backbone layers | **40**, all MoE; there are no dense FFN layers (`model.py:929`, `get_moe_config` `:142-149`). All 15,360 routed experts are 3 bpw in the EXL3 export [measured] |
| DSpark layers | 3 draft stages (idx 40–42 of `compress_ratios`, which is why it has 43 entries), each with its own attention and a 128-expert / top-3 MoE |
| Routed experts | 384/layer, top-6, 1 shared expert (always on) |
| Hidden / moe_intermediate | 5120 / 2304 → 35,389,440 params per expert |
| Router | `sqrtsoftplus` scores, `noaux_tc` (bias on selection only), `routed_scaling_factor` 1.5, `norm_topk_prob`. A separate VL routing bias for image tokens. `swiglu_limit` 10 |
| Attention ("CSA2") | MLA-style: q LoRA 1280 → 64 heads x 512. **One 512-d latent per token (MQA)**. Grouped O-LoRA (8 x 1024). A 128-token sliding window plus, where `compress_ratio>0`, up to 512 compressed positions picked by a learned sparse indexer |
| `compress_ratios` | Layers 0–1: `0` (SWA only). Layers 2–19: `2`. Layers 20–39: `1`. A 20-layer causal *encoder* + 20-layer *decoder*, split at 20 |
| Cross-layer reuse | Only `kv_source_layer_ids` [2,8,14,20] compute compressed KV. Only `index_source_layer_ids` (8 layers) run the indexer. A two-level candidate filter at layer 20. The reference uses a global mutable singleton (`SharedAttentionRuntime`) |
| Engram | n-gram hash memory at layers **1 and 14** (§5) |
| Hyper-connections (mHC) | Residual carried as 4 streams, Sinkhorn-balanced mixing. Our fork's `hc_mix` already serves V4 |
| DSpark | A separately trained draft (§10). **Not EAGLE/MTP**; upstream's flag is `--speculative-algorithm DSPARK` |
| Vision | ViT (32 layers) + aligner, BF16, ~0.9 GB. `--language-model-only` skips it |
| Context | 1M (YaRN x16 over 64K) |
| Params | 552B backbone + 196B Engram; 8B activated/token prefill, 16B decode [source; the split is unexplained; possibly DSpark counted in decode] |

**Quantization in the official checkpoint** [source: `kernel.py`, `convert.py`]:
- Non-expert weights are FP8 E4M3, block 32x32, UE8M0 scales.
- Routed experts are **OCP MXFP4** (E2M1, one E8M0 scale per 32 along K).

---

## 3. Byte geometry, against Qwen3.8

| | Qwen3.8 NVFP4 (today) | DSV4.1 EXL3 3.0 bpw | DSV4.1 official MXFP4 |
|---|---:|---:|---:|
| MoE layers x experts, top-k | 48 x 512, top-10 | 40 x 384, top-6 | 40 x 384, top-6 |
| Bytes / routed expert | 2,764,800 | **13,315,596** [measured] | 18,800,640 [computed from shapes] |
| Routed rows / token, nothing cached | 480 | 240 | 240 |
| Routed bytes / token, nothing cached | 1.33 GB | 3.20 GB | 4.51 GB |
| All routed experts | 67.9 GB | **204.5 GB** [measured] | 288.8 GB |
| Link time / miss row at 12.02 GB/s | 0.23 ms | **1.11 ms** | 1.56 ms |
| NVMe time / miss row, nvme2 (x2) / idle x4 drive | — | ~7.0 / ~3.4 ms [estimate] | ~9.9 / ~4.8 ms |

**EXL3 expert layout on disk** [measured, all 41 shard headers + index]:
- 12 tensors per expert, stored in this order:
  - `w1.suh` F16[5120] (10,240 B)
  - `w1.svh` F16[2304] (4,608 B)
  - `w1.mul1` I32[] (4 B)
  - `w1.trellis` I16[320,144,48]
  - then the same four for `w2` and `w3`.
- 3.010 bits/param.
- **Byte-contiguous with zero gaps. No expert crosses a shard file** (all 15,360 checked).
- File start offsets are not 4 KiB-aligned. An O_DIRECT superset read wastes 500–4,596 B
  per 13.3 MB expert.
- **Inside the row, `w1.trellis` sits at byte 14,852 (≡ 4 mod 16).** It is misaligned
  for 128-bit loads (§9.2).

**EXL3 export breakdown** [measured]:

| Component | GiB | Bits |
|---|---:|---|
| Routed experts | 190.48 | 3 |
| DSpark (`mtp.*`, 4,836 tensors) | 6.75 | 4 |
| Attention, gates, Engram `wkv` | 3.32 | 5 (attention) |
| Embedding | 1.23 | — |
| Shared experts | 0.83 | higher |
| Vision + aligner | 0.90 | native BF16 |
| LM head | 0.46 | 6 |
| hc / norms | ~0.15 | — |

One DSpark draft expert is **17,739,276 B** at 4 bpw [measured, `mtp.0.ffn.experts.0.*`].
The `mtp.*` tensors are absent from `quantization_config.json`'s `tensor_storage` but
present in the shards.

---

## 4. VRAM budget (production stopped, Stage B merged)

Usable VRAM is taken as 31.8 GiB (32,607 MiB) [measured]. Stage B is assumed, because
§10 makes it mandatory, so there is no miss-scratch row.

| Item | GiB | Notes |
|---|---:|---|
| Attention + gates + Engram `wkv` + shared + head + hc | 4.76 | [measured bytes] |
| Embedding | 1.23 | Can move to host (+99 slots) |
| KV + indexer caches at 40K ctx | ≤0.87 | 584 B/token/layer on the sm_120 `v4` layout (upstream `deepseek_v4_memory_pool.py:132-142`). This is the upper bound if all 40 layers allocate; far less if only the SWA windows + 4 KV-source layers do. Confirm when porting. **Also confirm the pool-configurator interaction**: the pool configurator's request-cap SWA sizing (`pool_configurator.py:807`, present since the pre-squash tip, §6.1) reserves SWA slots from `max_running_requests` before sizing the full pool; our own `compare_oracle.py` needed `max_running_requests=4` to avoid a starved full pool at `mem_fraction_static=0.7` (`bacaf8c63b`) — the EXL3 serving config will need the same check once it exists |
| Activations, graphs, workspace | ~3 | [estimate] |
| Dequantized EXL3 modules (`wo_a`, compressor, indexer `wk`) | 2.8 | `wo_a`: 40 × 8,192×4,096 bf16 (8 groups × `o_lora_rank` 1,024; 64 heads × 512 / 8) = 2,684,354,560 B ≈ 2.50 GiB. Compressor (`DeepseekV41Compressor`, `layers/attention/dsv4/dsv41_sparse.py:98`): `wkv` 5,120×512 bf16 = 5.24 MB on the 20 ratio-1 layers, `wkv`+`wgate` = 10.49 MB on the 18 ratio-2 layers (config `compress_ratios`: 2 × 0, 20 × 1, 18 × 2) = 293.6 MB ≈ 0.27 GiB. Indexer `wk` 512×128 bf16 = 131 KB per `index_source_layer_ids` layer, negligible. Total ≈ 2.8 GiB. All come from `deepseek_v4_exl3_weights.adapt_exl3_weights` (`DEQUANT_PREFIX_RE`, `WO_A_SLICE_RE`; `wo_a`'s 8 slices concatenated to `[G·R, D]`), which dequantizes EXL3 trellis to dense bf16 at load and keeps it: a permanent VRAM cost. **Phase 3 must decide whether to keep this row** (dense `wo_a` stays resident) **or remove it** (e.g. by keeping `wo_a` EXL3 and reconstructing per-call like the other attention linears, at a decode-latency cost). **3a ruling: kept dense; revisit in 3b.** The 3a smoke loaded with 9.90 GB and peaked at 30,097 MiB with 1,128 hot slots (§16.5), and nothing in 3a measured the decode cost of removing it. Not the same number as §15.2's measured 18.5 GiB truncated-model load memory (3.4 GiB init + 3 × 4.74 GiB routed experts) — that figure is a 3-layer smoke-test load footprint, not a full 40-layer serving budget, and its "3.4 GiB init" already includes some of this row for those 3 layers. |
| **Hot cache, no DSpark** | **31.8 − 9.86 = ~21.9** | ≈ **1,770 slots** of 15,360 = 11.5% [estimate]. Does not yet subtract the dequantized-EXL3-modules row (2.8 GiB, pending Phase 3's keep/remove decision above); if kept, this drops to ~19.1 GiB ≈ 1,540 slots = 10.0% [estimate] |
| DSpark draft, fully resident | 6.75 | ≈ 544 slots, **31%** of the above. A draft-expert cache is the alternative (§10) |
| **Hot cache, DSpark resident** | **~15.2** | ≈ **1,220 slots** = 7.9% [estimate] |

Qwen today: 13.8% coverage. Before Stage B lands, subtract 2.98 GiB (240 slots) for the
non-speculative Stage A scratch, and 17.9 GiB for a 6-token verify, which does not fit.

---

## 5. Engram

[source: `inference/engram.py`, `model.py`; upstream `upstream-scope/dsv4.1`]

**Access pattern:**
- Tokens map to a compressed vocab of 99,092 (NFKC, lower-cased, accents stripped).
- For each position, 2-, 3- and 4-grams are rolling-XOR hashed with per-layer odd
  multipliers, into disjoint prime-sized bucket ranges.
- That gives **24 rows/layer/token and 48 rows/token**.
- A row is 256 B FP8 E4M3 plus an 8 B E8M0 scale row (block 32), so 264 B logical.
- **Row ids depend only on token ids.**

**On disk** [measured]: `embed.weight` `[~384M, 256]` F8_E4M3 at offset 0 of each shard's
data section, then `embed.scale` `[~384M, 8]` F8_E8M0 about 98.3 GB later. So:
- A row costs **two reads** unless the scales (6.1 GB for both layers) are held in RAM or
  the tables are re-laid out.
- Our reader works in 4 KiB pages, so a token on a full RAM miss touches 96 pages
  (384 KiB) against 12 KiB of payload: ~0.2 ms at 1.9 GB/s [estimate].

**Reader: already exists.** `read_paged_rows` / `PagedRowSource`
(`model_loader/file_row_reader.py`, `kernels/jit/csrc/io/uring_file_reader.cpp:236-334`)
was built for Qwen4 PLE's 160 B rows. It dedupes and coalesces pages and handles mixed
row sizes in one io_uring submission.

**Upstream implementation** (`layers/engram.py`, `kernels/ops/embeddings/engram_gather.py`):
- **Hashing runs on the GPU and inside the graph** for decode
  (`engram_hash_ids_and_commit`, `engram.py:315-330`) and for DSpark target-verify
  (whole draft block, `:294-299`; `commit_after_verify` `:438-459`). Prefill hashing is
  eager (`deepseek_v4.py:731-738`).
- **The gather is one Triton kernel** reading a raw pointer to VRAM or to a pinned host
  table (`_HostTable`, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `engram.py:549-706`:
  memfd/anon mmap, `cudaHostRegister`, hugepages). It dequantizes inline.
- **Upstream has no NVMe tier.** The whole 203 GB lives in VRAM (TP4–8) or host RAM.
- **Open:** can our RAM tier simply be upstream's `_HostTable` at reduced size plus a slot
  map, rather than a new cache?

**Cache behaviour** [measured, exact LRU, 1,500,000 tokens]:
- Exact-LRU simulation over both Engram layers sharing one cache, driven by upstream's
  own `EngramHasher`/`compute_engram_hash_ids` (bit-exact parity with DeepSeek's
  reference, `test/manual/dsv41/test_engram_parity.py`). Stack distances come from a
  Fenwick-tree reuse-distance scan (`scripts/dsv41/engram_cache_sim.py`), unit-tested
  against a brute-force `OrderedDict` LRU
  (`test/manual/dsv41/test_engram_cache_sim.py`, 9 passed).
- Corpus: our benchmark corpus (`/mnt/nvme2/nvfp4-work/benchmarks/full/{sessions,results}.jsonl`,
  795 sessions), prompts plus the recorded completions, DSV4.1-Flash tokenizer,
  no chat template, capped at 1,500,000 tokens (48 accesses/token, 72,000,000
  accesses total). Full JSON:
  `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-engram/cache_sim.json`.
- **hit_ceiling = 71.70%** (unique_rows = 20,376,046 of 72,000,000 accesses;
  unique_set_gb = 5.38 GB). **session_cold_hit_rate = 35.18%** (share of accesses whose
  previous touch of the same row was in the same session).

| RAM cache | hit_rate | misses/token (mean) | misses/token (p99) |
|---:|---:|---:|---:|
| 1 GB | 60.82% | 18.81 | 48.0 |
| 2 GB | 66.30% | 16.18 | 48.0 |
| 5 GB | 71.68% | 13.59 | 48.0 |
| 10 GB | 71.70% | 13.58 | 48.0 |
| 20 GB | 71.70% | 13.58 | 48.0 |

- `hit_rate` is non-decreasing with budget and never exceeds `hit_ceiling`, as expected
  of exact LRU; 5 GB already holds effectively the whole unique working set for this
  corpus, so 10 GB and 20 GB add nothing further.
- The corpus is repetitive financial text, so general traffic will reuse less; treat
  these numbers as an upper bound on real-traffic hit rate, not a general estimate.

**Timing is the constraint, not bandwidth.**
- Layer 1 needs its rows about one layer's compute after the token is sampled:
  ~2.5–4 ms [estimate].
- A RAM-miss fetch is ~0.2–1 ms [estimate].
- io_uring runs on the CPU and needs the ids. Under overlap scheduling the host learns
  the token one step late.
- Layer 14 has ~13 layers of slack.
- §9.3 covers which mechanisms can overlap the layer-1 fetch with layer 0.

**Historical scoping observation:** the fork had no Engram code at this point. The
current branch has native Engram host nodes for layers 1 and 14 (§§22–23).

---

## 6. Upstream SGLang `dsv4.1` branch

Fetched read-only as `upstream-scope/dsv4.1` @ `85e8eddc54` (and `upstream-scope/main`) at
Phase 0 scoping time. **Superseded 2026-09-18**: `origin/main` (`aed3fb1cdd`) has since
landed the same work's *final squash* — `a6cf05817f` ("dsv4.1: remaining model and
runtime integration", #38798) — plus ~45 other upstream commits, and we merged that into
`dsv41` (§6.1). The pre-squash file-level shape described immediately below is what Phase
0's research read; §6.1 records what changed on top of it.

**Shape of the change.** 197 commits, 110 files, +9,506/−719 against merge-base
`1f0c73e9bd`.
- **Config.** `configs/deepseek_v41.py` (88 lines) remaps `DeepseekV41ForCausalLM` onto
  `DeepseekV4ForCausalLM`.
- **Model.** `models/deepseek_v4.py` +1,646 (vision, mHC, Engram, candidate indexer).
- **Attention.** `layers/attention/deepseek_v4_backend.py` +2,239, `dsv4/dsv41_sparse.py`,
  Triton/Gluon kernels.
- **MoE.** `topk.py` gains `sqrtsoftplus` and packed top-k; `moe_fused_gate.py`.
- **Quant.** `fp8.py`/`fp8_utils.py` view block-FP8 as MXFP8;
  `mxfp4_flashinfer_trtllm_moe.py`.
- **Speculation.** DSpark (§10).
- **Chat/parsers.**
- **Env vars:** `SGLANG_DSV4_KV_LAYOUT`, `SGLANG_DSV41_TORCH_PREFILL_INDEXER`,
  `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `SGLANG_DSPARK_NVLINK_VOCAB_GATHER`.

**MoE.** `flashinfer_mxfp4` is the only runner upstream has used, through the generic
`FusedMoE`/`select_experts`. The recipes (`docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`)
cover H200/B200/B300/GB300/MI350X only, at TP4–TP8.

`validate_deepseek_v41_features()` (`arg_groups/deepseek_v4_hook.py:256+`):
- Allows DSpark as the only speculation.
- Rejects `--enable-hisparse`, two-batch overlap, PP, the unified KV layout, and
  `--dsv4-attn-backend trtllm`.
- **Does not reject overlap scheduling.**

### sm_120 (RTX 5090) matrix [source; none of it tested]

| Kernel / path | sm_120 status |
|---|---|
| FlashMLA decode | `flash_mla_sm120` path exists, and predates this branch |
| `swapab_attention` fast decode | sm_100 only; not taken |
| DeepGEMM `fp8_fp4` MQA logits (indexer) | Gated on capability ≥ 10, so *attempted*. **Coverage unverified.** Torch fallback for prefill (`SGLANG_DSV41_TORCH_PREFILL_INDEXER`). krasis's `deepseek_v4_compressor.cuh` has portable CUDA indexer-score kernels (V4 geometry) usable as a reference fallback |
| `sparse_prefill_fwd` | Explicitly not sm_120; dense prefill fallback |
| V4.1 compact KV layouts (528 / 288 B/token) | sm_100 only. sm_120 uses legacy `v4` 584 B/token/layer |
| `flashinfer_mxfp4` MoE | Unverified on sm_120. Irrelevant on the EXL3 path |

The `lmsysorg/sglang:dev-dsv41` image was not inspected.

### Divergence and porting strategy

- Our HEAD (`master`) is 256 ahead / 462 behind `dsv4.1`, from
  merge-base `7bc4eb3740`.
- `dsv4.1` touches **none** of `expert_stream.py`, `expert_hot_cache.py`,
  `expert_residency*.py`, `model_runner.py`, `scheduler.py` or
  `moe_runner/flashinfer_cutlass.py`.
- **Conflict hotspot:** `layers/moe/fused_moe_triton/layer.py` (+34/−19 upstream,
  +93/−0 ours). Re-validate `topk.py` against the streamer.
- **Base.** The mainline includes Stage B since 2026-09-18 (`SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE`,
  `797be6f678`, now under `master` @ `b59edf2dc4`).
- **Strategy (measured 2026-09-18): merge, don't cherry-pick.**
  - Applied to our base, the squashed 110-file `dsv4.1` diff leaves 29 files
    unapplied, because upstream's branch point is 270 commits newer than ours.
  - A real `git merge upstream-scope/dsv4.1` conflicts in only 12 files:
    `modelopt_quant.py`, `scheduler.py`, the Qwen4 PLE files, and a few config/test
    files.
  - Do it on a new branch `dsv41` off the mainline
    (plan: `docs/superpowers/plans/2026-09-18-dsv41-phase0.md`).
  - `dsv41` then carries ~467 upstream commits, so **never merge it wholesale into the
    production branch**, and **do not rebase** the expert-stream work onto `dsv4.1`.
- **Result (2026-09-18, Task 1).** Merge commit `c64b2bd653` (`merge upstream
  dsv4.1 (DeepSeek V4.1 support) into dsv41`, parents `86dac964b7` ours /
  `85e8eddc54` upstream). Conflicts landed exactly on the 12 predicted files.
  Judgment calls beyond the Step 4 table:
  - `arg_groups/fields/exec_.py:911-921` (`ple_offload_backend`/`ple_offload_dir`
    help text) — kept ours' wording rather than a textual union, because the
    file-backend behaviour those strings describe is ours (selected-row staging
    fallback), not upstream's fail-fast unified-memory-only design.
  - `models/qwen4_exp_ple_table.py` and its two paired tests
    (`test_qwen4_ple_offload.py`, `test_qwen4_exp_ple_table.py`) were add/add
    conflicts (the file didn't exist at the merge-base) where nearly the whole
    file is PLE-offload domain; took ours wholesale (796/685/475 lines) rather
    than hunk-by-hunk, since upstream's smaller versions encode a fundamentally
    different, superseded file-backend design (`check_file_backend_supported`
    raises on an unsupported device instead of falling back to selected-row
    staging).
  - `test_fp8_moe_runner_fallback.py` took upstream's `CustomTestCase` base
    class over ours' plain `unittest.TestCase`, matching house test convention;
    no behavior lost since the assertions are identical.
  - Self-review confirmed every expert-stream/doorbell/hot-cache identifier
    count in `scheduler.py` is preserved from ours, plus upstream's new
    `tree_cache.flush_pending_backups()` call.
  - GPU suite (Task 1 Step 8): run 2026-09-18: 860/861 vs baseline 861/861, the one failure a known flaky doorbell test (§15.2).
  - Follow-up merge of the upstream tip a5b84f11e5 (11 commits: block-fp8/mxfp8
    quant merge, V4.1-only gating, renames) — `9e5cc68bd8`, 0 conflicts, no fixes
    needed (semantic re-check of `environ.py`, `fused_moe_triton/layer.py`, and
    `decode_cuda_graph_runner.py` found only comment rewording; `modelopt_quant.py`
    and `memory_hook.py` were untouched by these 11 commits, so Task 1's fix
    stands).

### 6.1 2026-09-18 — Upstream #38798 merge

Before this, `dsv41` carried upstream's *pre-squash* `dsv4.1` PR branch (tip
`a5b84f11e5`, the state §6 above and §14 describe). We then merged `origin/main`
(`aed3fb1cdd`) into `dsv41` — merge commit **`c55f1572b0`**. That brought in upstream's
final squash of the same PR, **`a6cf05817f` ("dsv4.1: remaining model and runtime
integration", #38798)**, plus 45 other `main` commits, including `1b200ffaaa`
(block-FP8 served through FlashInfer MXFP8 GEMMs, #40039), `0be8a0af0e` (TileLang JIT
cache dir, #39364), `c46bf5e990` (FlashInfer fused finalize off by default for numerical
accuracy, #40105), `3ce3b4969f` (NPU `wo_a`), and `7bc9152447` (kernel tests
consolidated). A follow-up fix, **`bacaf8c63b`**, was needed in our own
`compare_oracle.py` tool (below).

**What #38798 changed against the pre-squash tip we already had:** little. Against its
parent on `main` the squash is 103 files, +8,802/−718, but almost all of that was already
in `dsv41` via the pre-squash merges. Against `a5b84f11e5` (restricted to the squash's
files) it is **27 files, +801/−127**, concentrated in `deepseek_v4.py`, `deepseek_v2.py`,
DSpark (`dspark_accept.py`, `dspark_verify.py`, `dspark_worker_v2.py`), `fp8.py`,
`flashinfer_comm_fusion.py`, `schedule_policy.py`/`tokenizer_manager.py`, and
pool/autotune tests. Merge conflicts (20 files) and their resolutions are listed in the
`c55f1572b0` message; our EXL3 no-fuse guard, sm_120 DeepGEMM metadata and
`candidate_source_layer_id` wiring were kept.
- **VL routing** (`srt/multimodal/dsv41/vl_routing.py`, `vision_topk`), the V4.1
  attention/DSpark runner files (`c2_decode_pool.py`, `decode_attention_sm100_gluon.py`,
  `dsv4/dsv41_sparse.py`, `dspark/commit_swa.py`) and request-cap SWA pool sizing
  (`compute_swa_request_cap()`, `pool_configurator.py:807`, `_resolve_swa_cap_tokens`
  at `:1196`) were **all already present at the pre-squash tip**, not new in the squash
  (`pool_configurator.py` is byte-identical before and after the merge). They are
  upstream-provided either way. The V4.1 path still sizes SWA from the request cap
  because `model_overrides/deepseek_v4.py` leaves `swa_full_tokens_ratio` unset for
  `deepseek_v41`.
- **`compare_oracle.py` needed `max_running_requests=4` after the merge** (`bacaf8c63b`):
  at `mem_fraction_static=0.7` the truncated model failed with "DSV4 SWA pool cap
  (598272 tokens, 0.98 GB) leaves no room for the full KV pool within the available
  0.83 GB". Pre-merge runs at the same settings loaded. Since the pool sizing code is
  unchanged, the trigger is elsewhere (a changed default or larger non-KV footprint);
  **root cause unverified**. With the cap, the merged tree reproduces the pre-merge
  oracle comparison exactly (top1 0.9206, mean |Δlp| 0.0839 vs `oracle-p4fw`). The
  EXL3 serving config must set `max_running_requests` deliberately for the same reason.
- **sm_120 coverage unchanged:** the V4.1 compact KV layouts, the indexer fast paths
  and `sparse_prefill_fwd` stay sm_100/sm_103-gated; `flash_mla_sm120` remains the
  sm_120 decode path. The §6 matrix holds as written (static re-read, not re-run).
- `layers/engram.py` has **zero diff** in `a6cf05817f` (it was already complete
  pre-squash) — §14.2's findings (allocation sized at `num_embeddings`, not a cache
  budget; the "owned" test is a contiguous shard-membership check, not a real
  cache-miss lookup) stand unchanged. STILL OURS.
- **Confirmed untouched, again:** `git show --stat a6cf05817f` and a scan of the other
  ~45 `main` commits touch none of `expert_stream.py`, `expert_hot_cache.py`,
  `expert_residency*.py`, `expert_host_arena.py`, `expert_doorbell.py*`,
  `model_runner.py`, `scheduler.py`, or `moe_runner/flashinfer_cutlass.py`. §6's and
  §8's "touches none of our streaming files" claim holds after the final merge, not
  just the pre-squash one.
- The block-FP8-via-MXFP8 and FlashInfer-fused-finalize commits (`1b200ffaaa`,
  `c46bf5e990`) land on the official MXFP4/FP8 checkpoint's quant path
  (`fp8.py`/`fp8_utils.py`/`mxfp4_flashinfer_trtllm_moe.py`), which §6 already called
  "Irrelevant on the EXL3 path" — still true; no plan change.

**Doc items this closes or updates:** §10's "`dspark_layers_to_capture` for V4.1 is not
yet checked" line was already stale before this merge — §14.3 (Task 2/Phase 0) had
already confirmed `[37, 38, 39]` from `config.json` directly; that finding is
unaffected by this merge and is now cross-referenced from §10. §11 Phase 0 item 3 ("no
diff of our quant files against `dsv4.1`'s `fp8.py`/etc. was run") is superseded: those
files are no longer a separate branch to diff against, they are merged code now (see
§11's updated note).

---

## 7. EXL3 and exllamav3

Repo: `turboderp-org/exllamav3`, **MIT**. The checkpoint was converted with v1.4.2,
codebook `mul1`, `out_scales=always`, `--hq`.

- **Format.** QTIP-style trellis: Hadamard-rotated weights with sign vectors `suh`/`svh`,
  int16 trellis tiles decoded through a procedural `mul1` codebook, no separate scales.
  Loader: `exllamav3/modules/linear.py:403-421`.
- **Kernels.**
  - `exllamav3_ext/quant/exl3_moe.cu` (+ `exl3_moe_coop.cu`) is a **fused MoE MLP**,
    bincount-driven, with bsz ≤ 8 decode tiles and sm_120 tile selection
    (`doc/env_vars.md:344-357`).
  - **It takes per-expert pointer arrays** (`ptrs_trellis/ptrs_suh/ptrs_svh`,
    `block_sparse_mlp.py`). So a slot-indexed hot cache can be exposed by rewriting
    pointer tables in-graph, which is cheaper than our NVFP4 remap path.
  - The trellis is loaded with 128-bit `cp.async` through `int4*` casts
    (`exl3_gemm_inner.cuh`), which needs at least 16 B-aligned trellis.
  - `exl3_gemv.cu` does dense decode.
- **DeepSeek V4.** `architecture/deepseek_v4{,_mtp,_vision}.py`.
  `_RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}` **raises `KeyError` on V4.1's
  ratios 1 and 2 today**. Engram support is not confirmed. exllamav3 is **not
  installed** anywhere on divix01.
- **Its own offload** (`model/moe_cpu_host.py`) assumes every expert fits in host RAM.
- **The card's runtime is vLLM** (`--quantization exl3`,
  `--hf-overrides '{"engram_table_dir": ...}'`, DSpark via `--speculative-config`). That
  vLLM EXL3 path is the nearest prior art for an SGLang EXL3 quant method. Find it.
- **SM-contention risk.** Trellis decode is real ALU work. The `hc_mix` lesson says
  measure it against in-flight copies before committing.

---

## 8. Our fork's integration surface

(`master` @ `557c1fbaec`; line numbers are jump targets.)

> **Historical code snapshot.** This section describes the fork at `557c1fbaec`,
> before the EXL3 and Engram implementation. Its present-tense gaps and line numbers
> do not describe current HEAD. See §§17 and 23 for the implemented path.

**Hook point: the quant method.** `ModelOptFp4MoEMethod.process_weights_after_loading`
sets `layer._nvfp4_expert_streamer` (`modelopt_quant.py:2963`). Consumers discover it by
`getattr` walk: `model_runner.py:718,745,747`, `qwen2_moe.py:772`,
`fused_moe_triton/layer.py:1572`, `expert_host_arena.py:65`, `expert_hot_cache.py:957`.
No other quant method attaches a streamer, and there is no EXL3 code anywhere.

**A row is a hard-coded NVFP4 tuple** (`NVFP4_STREAM_TENSORS`, `expert_stream.py:40-47`).
Expert count, bytes/row and top-k are derived from shapes (`_validate_sources`, `:820`).
**Per-tensor alignment inside a slot is not parametric** (§9.2).

**Pieces relevant to the three tiers:**

| Piece | State |
|---|---|
| `ExpertHostArena` (`expert_host_arena.py:42`) | Holds **100%** of rows. The in-graph gather (`expert_cache_transfer.cuh:17-71`) reads it by fixed address via `ld.global.nc`; there is **no miss path**. The gather plan assumes `expert_ids` name host source rows (`expert_row_plan.py:7,74`). It has to become a slot-indexed RAM cache |
| `ExpertPinnedHostCache` (`expert_stream.py:89-285`) | Bounded LRU of rows in pinned RAM, fed by io_uring. **Eager-only**: `ensure_rows` calls `.tolist()` and keeps Python bookkeeping. Disabled in production |
| `ExpertFileRowReader` / `uring_direct` (`expert_file_reader.py:25-140`) | Needs the offloader's re-laid-out per-tensor files (`utils/offloader.py:264-347`). An EXL3 variant can point `AlignedRowSource` at the **original shards** with a per-expert aligned offset table |
| `read_paged_rows` (`file_row_reader.py`) | Sub-page row reader (PLE). Reusable for Engram as-is |
| `copy_expert_row_segments_gpu` (`expert_cache_transfer.cuh:200`) | Already takes `{src, dst, bytes}` segment lists: a candidate place to pad a raw row into an aligned slot |
| Doorbell (`expert_doorbell.py/.cuh`) | Refuses overlap scheduling (`model_runner.py:797-801`), all speculative decoding (`:769-772`) and decode graph bs > 1 (`:773-775`). The overlap refusal is a *placement* constraint (fail-stop check after results), not physics |
| `SGLANG_MOE_GPU_RESIDENCY_UPDATE` (insert-on-miss lives here) | Refuses speculative decoding (`:812-815`, "verify commits do not reach the device clock") and decode graph bs > 1 (`:816-820`, "padded batches would add graph rows as tokens") |
| `breakable` decode graph (`runner_backend/breakable_cuda_graph_backend.py`) | Segments captured graphs with eager host code between them. Our `deepseek_v4.py`'s `eager_on_graph` breaks (attention, low-ratio sources, Engram hashing) fire only for **extend** batches (`forward_mode.is_extend()`, `models/deepseek_v4.py:2033-2038`, `:2337-2345`, `:4223-4228` at `dsv41` `4b906b3d4f`), so DSV4 **decode** captures with zero breaks of its own. The ~40 breaks per forward are a prefill property. The eager break functions run under overlap scheduling in the shipping Qwen config (corrected in §17) |
| Graph-gather scratch (`..._GRAPH_GATHER_SCRATCH_ROWS`) | Already multiplies by `verify_tokens` under speculation (`model_runner.py:740-750`) |
| CUDA host nodes / `cudaLaunchHostFunc` | **Not used anywhere** in `python/sglang/kernels/jit/csrc` or `sgl-kernel`. Would need a new C++ entry point |

**Also needed:**
- A pin primitive for the shared expert. The hot cache has none; residency is only
  emergent. Phase 3.
- Streaming wired into `deepseek_v4.py`'s MoE forward. It has no expert-stream
  references today.
- A check of grouped `noaux_tc` top-k (`deepseek_v2.py:657-671`) against `sqrtsoftplus`.

**Existing DeepSeek support:**
- `models/deepseek_v4.py` (4,039 lines), `deepseek_v4_nextn.py`, `deepseek_v4_dspark.py`
  and `layers/attention/dsv4/`.
- No V4.1 code: no Engram, no ratio-1/2 generalization, no vision.
- PLE offload and the Mamba flags drop out.

---

## 9. Three-tier design: NVMe → ~90 GB RAM → VRAM

> **Original design budget, not the current allocation.** Current values are 50 GiB
> pinned MoE host memory, 5 GiB native Engram cache, and 14 GiB GPU hot budget (§23).

### 9.1 RAM budget split [estimate]

| Item | GB | Notes |
|---|---:|---|
| Process, non-expert host state, io_uring staging | ~10 | Measure |
| Engram RAM cache (rows + scales, 264 B) | 5 | [measured, exact LRU, §5]. Rule: smallest budget within 5 points of `hit_ceiling` (71.70%) — 5 GB reaches 71.68%, within 0.02 points, so larger budgets are not worth it on this corpus |
| **Expert RAM cache** | **~75** | ≈ **5,630 experts, 36.7%** of 15,360 |
| VRAM hot cache | — | 1,220–1,770 slots (§4), all also held in RAM (inclusive). Distinct coverage = the RAM tier, ~36.7% |

**Pinning ~80 GB at startup.** `cudaHostRegister` on this much memory takes real time
and interacts with THP. Measure it. Upstream's `_HostTable` uses `MADV_HUGEPAGE` /
`MADV_COLLAPSE`.

**Hierarchy: inclusive, VRAM eviction = drop** (both decided 2026-09-18).
- Promotion RAM→VRAM **keeps** the RAM copy.
- An evicted VRAM row is **dropped**, with no D2H demote. It is still in RAM, so its
  next use is a RAM hit (1.11 ms), not an NVMe read.
- The cost: RAM duplicates the VRAM-resident set (1,220–1,770 rows, 16–24 GB), so
  distinct coverage is the RAM tier alone, ~5,630 experts (36.7%).
- The rejected alternative, exclusive with drop, would have covered ~45–48% of experts
  distinct, but every dropped row would have gone back to NVMe.
- The Phase 3 routing trace should confirm the trade.

### 9.2 Reading from NVMe, and the slot layout

**Experts:**
- One O_DIRECT read per expert, straight from the EXL3 shard (on nvme1 after the copy),
  using a per-expert `(file, aligned_offset, aligned_len)` table computed from headers.
  There is no on-disk re-layout.
- The reader takes the expert directory as a parameter (nvme2 now, nvme1 later).

**Slot layout (Phase 0 decision — resolved, §14.1):**
- A raw row copied byte-for-byte leaves `w1.trellis` at an offset ≡ 4 mod 16.
  exllamav3's trellis needs exactly **16 B alignment** (`cp.async.cg` on an `int4*`
  cast) and `suh`/`svh` need **8 B alignment** (a `half4` cast); nothing in
  exllamav3 needs wider than 16 B. `build_exl3_slot_layout`'s default is **16 B**
  (§14.1(a)-(b)), so the slot needs per-tensor padded offsets at that alignment, not
  128 B.
- The O_DIRECT superset read also leaves the row itself at a file-dependent offset
  inside its RAM buffer.
- **Padding site (decided, §14.1(c)): the RAM→VRAM segment copy**
  (`copy_expert_row_segments_gpu`, `expert_cache_transfer.cuh:200-213`), not a
  separate NVMe→RAM host memcpy. That kernel's per-row copy already degrades
  gracefully from 16-byte vectorized loads to 4-byte words to bytes when its source
  is unaligned (`copy_expert_row_lane`, `:53-78`), so an unaligned RAM-side row is
  still correct — only the VRAM destination, written at the slot's aligned offset,
  needs to be 16 B-aligned. RAM-tier rows stay raw/unpadded, so the ~75 GB RAM
  budget (§9.1) carries no padding tax. A separate NVMe→RAM host memcpy pass was
  considered and rejected: it would add a distinct ~1–2 ms/admission CPU copy on top
  of the link/NVMe time for no alignment benefit the segment copy doesn't already
  give the destination.

**Engram:**
- `read_paged_rows`: 48 weight pages plus 48 scale pages per token on a full RAM miss.
- Options if contention shows up: hold the 6.1 GB of scales in RAM (halving the reads),
  or re-lay out the tables shard by shard (peak +101.5 GB).

**GPUDirect Storage is not an option.** On GeForce, cuFile runs only in compatibility
mode, which bounces through host memory ([NVIDIA
forum](https://forums.developer.nvidia.com/t/gds-requirement-for-cards/184489),
[GDS O_DIRECT guide](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html)).

### 9.3 The shared hard problem: a RAM miss inside a CUDA-graph decode

The in-graph gather can read a row that is in RAM, through a slot index or pointer
table. **Nothing can wait for NVMe from inside the graph today.**

**What any mechanism must satisfy:**
- **Bounded wait plus a defined failure path.** A failed or stuck io_uring read must not
  wedge the stream and every `synchronize()` after it. Candidates:
  - a fail-stop flag read by the next kernel plus a watchdog (as the doorbell does);
  - drop-on-miss as a *safety valve* only, which is a narrower ask than option D as a
    policy.
- **Experts:** the route is only known at the layer, so the NVMe read is on the critical
  path in every option. §9.4 models it serially.
- **Engram layer 1:** the ids are known at graph start (GPU hash), so a design that
  *submits* early and *waits* before layer 1 can hide the read behind layer 0.

| Option | How | Overlap scheduling | DSpark | Engram overlap with layer 0 | Main costs / risks | Size |
|---|---|---|---|---|---|---|
| **A. `breakable`-graph host break** after each MoE router and before Engram layers | D2H ids → host checks the RAM map → io_uring misses → publish slots → resume. Either new segments per MoE layer, or router placement moved before the existing attention break (to decide) | Works under overlap (Qwen ships eager breaks with it), but DSV4 decode has no breaks today: option A would **add** ~40 per decode forward (§17). A D2H sync in a break stalls the single scheduler thread and re-exposes ~2.3 ms/token of host prep, **~1–2% at a 110–200 ms token** | OK: breaks see the verify-batch union | Only with a separate early submit | Scheduler-thread stall; more segments | L |
| **B. CUDA-graph host nodes, split** (`cudaLaunchHostFunc` captured as host nodes) | Node 1 at graph start: *submit* io_uring reads for known ids. Node 2 before the consumer: *wait* with a bounded timeout, then write the slot table | Likely compatible: no Python or scheduler in the loop (overlap uses a single scheduler thread plus `FutureMap`, `overlap_utils.py:248`) | OK | **Yes**, the submit/wait split | The stream is idle for the whole wait. Host functions may make no CUDA calls and may serialize with each other (a 7 ms callback blocks every other host callback). New C++ entry point. Capture with `torch.cuda.graph` and the breakable backend untested. **An unsplit, blocking B cannot hide the Engram read** | L |
| **C. CPU io_uring thread + device-side poll** | The GPU posts ids to mapped memory. A CPU thread reads NVMe into a RAM slot and writes a completion flag. A kernel polls the flag with a watchdog. Doorbell-*like*, but redesigned: not the doorbell's all-or-nothing tag semantics and not its current guards | Placement of the fail-stop check must be redesigned (the doorbell's refusal is placement, not physics) | Needs the same guard redesign §10 already requires | **Yes**, by posting at graph start | A polling kernel occupies SMs during the wait (the `hc_mix` lesson); a late-landing write needs the doorbell's race analysis | L |
| D. Drop missing experts from top-k (policy) | Compute with the resident subset | — | Muddies accept/reject | — | Lossy. **Owner sign-off plus accuracy eval** | M |

**Phase 0 prototype acceptance criteria** (whichever of A, B or C is tried first):
- Measured per-callback or per-break overhead at the RAM-hit path.
- Stream-idle time on a forced NVMe miss.
- A demonstrated timeout that fails stop without hanging the process.
- Capture under the breakable backend with overlap scheduling on.

### 9.4 Throughput model [estimate]

`ms/token ≈ G x 1.11 + f x G x t_nvme + admission + compute`, where:
- `G` = VRAM misses/token (240 routed rows/token, no speculation).
- `f` = fraction of those misses that also miss RAM.
- `t_nvme` = 7.0 ms on nvme2, 3.4 ms on an *idle* x4 drive.
- `admission` = any host-side padding copy (§9.2).

Engram adds <1 ms.

| G | f | nvme2 link+NVMe ms | tok/s | idle x4 (nvme1) ms | tok/s |
|---:|---:|---:|---:|---:|---:|
| 84 (35%) | 10% | 152 | 6.6 | 122 | 8.2 |
| 84 | 25% | 240 | 4.2 | 165 | 6.1 |
| 108 (45%) | 10% | 196 | 5.1 | 157 | 6.4 |
| 108 | 25% | 309 | 3.2 | 212 | 4.7 |

- The experts will be on nvme1, so the right-hand columns apply, but they assume an idle
  drive. nvme1 carries `op-reth` and is DRAM-less QLC, so measure it under load.
- Compute is excluded; EXL3 BS1 decode cost is unmeasured.
- A *uniform* router is the worst case: ~90% of routed rows miss VRAM (G≈216), and
  with 36.7% of experts in RAM, f≈60%. That is ~680 ms/token on an idle x4 drive and
  ~1.15 s on nvme2. The design rests on routing skew keeping f low.
- **At f ≥ 25% the system is NVMe-bound, and no software change in this doc fixes that.**

**Measured in Phase 3a (§16.8, eager mode, full 40-layer model, 1,128 hot slots, 5,644 RAM
rows).** These replace the proxy's `G` ≈ 107 and `f` ≈ 34% above:

| Arm | `G` | `f` | `f_after_warmup` | §9.4 model, nvme2 | x4 |
|---|---:|---:|---:|---:|---:|
| Cold dynamic (8 sessions, no seed) | 115.03 | 0.1853 | 0.1749 | 276.9 ms (3.6 tok/s) | 200.1 ms (5.0 tok/s) |
| Seeded dynamic (4 other sessions) | 111.73 | 0.1844 | 0.1754 | 268.2 ms (3.7 tok/s) | 194.1 ms (5.2 tok/s) |

- `G` and `f` sit between the table's 108/10% and 108/25% rows, so the model's link + NVMe
  envelope holds (196–309 ms on nvme2). `f` is **well under** the proxy's 34% and under the
  25% NVMe-bound line.
- **The model under-counts a miss.** Measured per decode RAM miss: 8.10 ms read + 2.18 ms
  CPU split = ~10.3 ms, against the model's 7.0 ms (nvme2). Its `t_nvme` has no split term.
- **The model is not the bottleneck.** Measured eager decode is **1.6 tok/s** (~0.62
  s/token), against ~0.28 s modelled. About 0.22 s of the 0.60 s median forward is RAM-miss
  I/O; the other ~0.38 s is not (§16.12, an observation to profile). The x4 column
  is still unmeasured (nvme1 fio pending, §16.3).
- The seeded arm ran on different sessions from the cold arm, and no static arm ran, so
  neither `G`/`f` pair isolates the seed's effect.

---

## 10. DSpark (in scope)

[source: `inference/model.py:1089-1157`, report §2.4.3, upstream
`srt/speculative/dspark_components/*` and `kernels/ops/speculative/dspark/*`]

**Mechanics.**
- Three draft stages with sliding-window attention and their own 128-expert/top-3 MoE,
  fed by target hidden states from layers **37–39** in the reference code. Upstream's
  `dspark_layers_to_capture` for V4.1 is confirmed **`[37, 38, 39]`**, with no separate
  draft checkpoint — see §14.3 for the full resolution chain (`config.json` keys through
  to `self.dspark_layers_to_capture`), unaffected by the 2026-09-18 #38798 merge (§6.1).
- `TargetHiddenKvInjector` (`dspark_kv_inject.py`) writes those hidden states into the
  draft KV every verify step.
- One draft forward proposes a 5-token block; a Markov head refines it and a confidence
  head scores it.
- **Verify processes up to 6 tokens** (anchor + 5; `dspark_config.py`).
- `DSparkVerifyPlanner` (`srt/speculative/dspark_components/dspark_planner.py:70`,
  `_dynamic_graph_tier` at `:135`) picks each request's verify length. It uses
  confidence-based survival (`kernels/ops/speculative/dspark/dspark_schedule.py`) and a
  profiled steps/s table (`dspark_components/dspark_sps.py`). So k is dynamic.
- The tech report gives **no acceptance-rate numbers**.

**Byte model.** On the Qwen stand-in (`analysis/cross-token/spec_window_summary.txt`):
- The union of experts grows sublinearly (1.68x reuse at N=8), but miss rows grow faster
  than accepted tokens unless α is high.
- **+35% rows per accepted token at α=0.7, N=4; break-even α≈0.84–0.93.**
- Expert NVMe cost is **bandwidth-bound** (13.3 MB per read), so "amortizing one stall
  over 6 tokens" does not rescue the byte model. It helps only Engram's small,
  latency-bound reads.
- DSV4.1's overlap curve (384 experts, top-6) is unmeasured.

**Costs specific to this stack:**
- **Draft residency.** 6.75 GiB ≈ 544 target slots, ~31% of the no-DSpark hot cache.
  The draft touches only 9 experts/step (~160 MB). The alternative is a small draft-expert
  cache, but it has its own catch: **a draft-expert miss sits on the critical path
  *before* every verify**, 17.7 MB = ~1.5 ms link + ~4.5 ms NVMe on an idle x4 drive,
  costlier than a target miss. Deciding needs draft routing skew. It also means two
  hot-cache managers sharing one VRAM budget.
- **Stage B is mandatory.** Stage A keeps per-layer miss scratch until the boundary: at
  k=6, up to 36 rows/layer x 40 = 1,440 rows = 17.9 GiB. Deduplicate the union before
  the gather.
- **Guards to redesign.** Both the GPU residency updater (insert-on-miss, +14.2% on Qwen)
  and the doorbell refuse:
  - `is_speculative()` (verify commits do not reach the device clock);
  - **decode graph bs > 1**, because their accounting treats graph rows as tokens.

  A verify forward is `bs x draft_token_num` tokens, so the row-as-token accounting must
  be redesigned, not just the speculation check. Insert-on-miss must learn batched verify
  misses and verify-then-commit ordering.
- **Graph shapes.** Decode, draft-block and verify (possibly per tier via
  `_dynamic_graph_tier`). Each needs its own scratch and gather sizing.

**Measurement gate, then a rule.** On the running model, measure:
1. DSV4.1's cross-token expert-union curve at N=2/4/6.
2. Draft routing skew.
3. Real α on representative traffic.

Then recompute the break-even. **DSpark ships only if measured α clears it** (§12).

---

## 11. Phases

**Phase 0 — no GPU:**
1. **Base branch — done.** Branch `dsv41` off the mainline (Stage B included), merged
   `upstream-scope/dsv4.1` into it (§6). Task 1: merge commit `c64b2bd653` (base
   `85e8eddc54`), then the upstream-tip follow-up merge `9e5cc68bd8` (`a5b84f11e5`,
   0 conflicts). The GPU suite (Task 1 Step 8) ran 2026-09-18: no merge
   regression (§15.2).
2. **Done — Task 5 / §5.** Bit-exact Engram hash parity test
   (`test/manual/dsv41/test_engram_parity.py`) and the exact-LRU re-run
   (`scripts/dsv41/engram_cache_sim.py`, unit-tested against a brute-force LRU) are
   both in §5, with the stated granularity (1,500,000 tokens, 48 accesses/token) and
   the RAM-budget table. Weight-row and scale-row hit rates were not separated (the
   simulation tracks 264 B combined rows, §5's "logical" row); if that split matters
   for Phase 3, re-run with the two counted independently.
3. **Done.** `git diff a5b84f11e5 HEAD --stat -- fp8.py fp8_utils.py mxfp4*.py` (the
   pre-squash tip we scoped Phase 0 against, vs current `dsv41`) is 1 file, +4/−10:
   `fp8.py`, all of it upstream's own follow-on work (`#39823`) landed by the
   `c55f1572b0` merge (§6.1), not a fork change. Since that merge, `a5b84f11e5` is no
   longer the right baseline — the meaningful check is against current upstream:
   `git diff origin/main HEAD --stat -- fp8.py fp8_utils.py mxfp4*.py` is **empty**.
   Our fork carries no changes of its own to the FP8/MXFP4 quant files, confirming
   §6.1's file-level finding that this path serves only the official MXFP4/FP8
   checkpoint and is irrelevant to the EXL3 path.
4. **Done — §14.2.** Upstream's Engram `_HostTable`/gather read in full; the gather
   takes a row index, not a raw offset, so a slot map can sit in front of it, but the
   allocation, addressing and miss semantics all need to change (§14.2).
5. **Done — Task 3 landed the layout, this task fixed its default — §14.1.** The
   per-expert aligned offset table is `Exl3ExpertLayout`
   (`python/sglang/srt/layers/moe/exl3_expert_layout.py`); the slot layout is
   `build_exl3_slot_layout` (`python/sglang/srt/layers/moe/exl3_slot_layout.py`),
   whose default alignment this task changed from 128 B to 16 B (§14.1). Padding site
   (NVMe→RAM host memcpy vs RAM→VRAM segment copy): §14.1(c) recommends the
   RAM→VRAM segment copy.
6. **Done — §14.5.** No per-expert load statistics or plots exist in the tech report;
   the calibration trace is confirmed not obtainable (not in the downloaded files).
   `G`/`f` remain unmeasured; Phase 1's truncated-model router statistics (§11 Phase 1)
   are the next, still-partial proxy.
7. **Done — §14.4.** No official or single canonical vLLM EXL3 implementation exists.
   Record: issue and repos found, §14.4.
8. **Still open (§12.1).** The go/no-go tok/s bar needs an owner decision; nothing in
   this task's research resolves it.

**GPU window items** (each needs crypto-c9 scheduling; a small-VRAM microbenchmark
beside production perturbs production latency, so decide explicitly):
- Prototype one miss mechanism (§9.3) against the acceptance criteria there. It needs a
  new C++ entry point for B.
- When production is down: fio on **nvme1 with `op-reth` running** (13 MB random reads
  at QD 6–36) and on nvme2 (4 KiB random reads at QD48). Measure host RAM free with
  production stopped, and `cudaHostRegister` time for ~80 GB.

**Phase 1+2 (EXL3-first bring-up) — done, 2026-09-18.** Kernel coverage: §15.1 (Window
A). Model bring-up, oracle and router skew: §15.2 (Window B). **Merge decision** (owner,
2026-09-18): bring up the EXL3-quantized checkpoint first rather than the original
Phase 1/Phase 2 split (model bring-up on dense weights, then a separate EXL3 quant-method
phase), because every backbone linear in this checkpoint is EXL3 and the official
(MXFP4/FP8) backbone shards were never downloaded — there is no non-EXL3 model to bring
up first. `Exl3MoEMethod` (trellis-resident routed experts) and the EXL3 linear method
therefore landed together with model bring-up in this window; the remaining EXL3 MoE
performance work (below) is the only piece deferred.

**Phase 2b — original EXL3 performance checklist:** the fused EXL3 MoE/GEMV path,
exllamav3 subset, and batch-1 CUDA-graph capture were implemented by Phase 3b (§17).
The proposed standalone BS1 microbenchmark with a concurrent copy has no result
recorded in this reference; the serving measurements in §§17–23 supersede the
earlier “open” implementation bullets.

**Phase 3a — EXL3 streaming on the expert framework — done, 2026-09-19 (§16).** Landed and
measured in eager mode on the full 40-layer model:
- The EXL3 format and shard row-source plugins on the framework (`Exl3ExpertFormat`,
  `Exl3ShardRowSource`), with the EXL3 server-args gate (§16.2).
- The per-name, inclusive pinned RAM tier with VRAM eviction = drop, and its hot-slot clamp
  (§16.2, §14.1). `Exl3RamExpertCache` was deleted.
- `G`, `f` and the hot-cache seed recorded (§16.8, §16.10), the oracle at 22 layers
  (§16.6), and split-versus-repack on nvme2 (§16.4).
- **Not done in 3a:** the full-depth oracle comparison, nvme1 fio, the Engram hit rate on
  the full model, and the go/no-go check (open, §12.1: the owner decides whether 3b is
  worth building).

**Phase 3b — graph-mode three-tier streaming (original checklist; implemented portions in §17):**
- **Option C built and measured (§17):** decode under breakable CUDA graphs at bs 1 with the
  MoE in-graph (pinned-tier graph gather + fused `exl3_moe`), NVMe misses served by the
  io_uring thread with a bounded in-graph wait and fail-stop; next-layer advisory prefetch
  behind `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`.
- The in-graph RAM tier — done, §17.
- The §9.3 miss mechanism and its failure path — done, §17.
- CUDA graphs and graph gather for EXL3 — done, §17.
- A prefill admission policy for the RAM tier, **only if** the simulation-only arm shows a
  large gap. At 256-token prompts it showed none (admitting prefill misses helped, §16.9);
  it was not run at 512+ tokens.
- A BLOB host layout, if the split numbers call for one (§16.4; the nvme2 ruling was to
  skip the repack).
- Async reads.
- Profile the ~0.38 s/token of eager decode that is not RAM-miss I/O (§16.12) before
  sizing any of the above — done, §17 (§17.7).

**Phase 4 — DSpark:**
- Run the measurement gate (§10), then apply the α rule.
- Build batched verify misses and the guard/accounting redesign.
- Decide draft residency vs a draft cache; build the verify graph shapes.

**Phase 5 — serving and quality:**
- Measure tok/s against §9.4.
- Run a quality check of EXL3 3.0 bpw with `scripts/fp8_accuracy/` or equivalent.

---

## 12. Open decisions

**Decided:**
- ~~Drive for the experts~~: they move to `/mnt/nvme1`; the owner does the copy later
  (schedule it, §1).
- ~~VRAM eviction~~: **drop**, with no D2H demote (§9.1).
- ~~RAM copy on promotion~~: **keep it**; the hierarchy is inclusive (§9.1).

**Open:**
1. **Go/no-go bar.** What tok/s justifies the downtime and the Phase 2–3 investment?
   The envelope is ~3–8 tok/s before compute, against 19.36 tok/s on Qwen today
   (MOE_EXPERT_TRANSFER).
2. **Miss mechanism**: A, B (split) or C (§9.3), decided after the prototype (option C
   built, §17; A and B not built).
3. **Overlap scheduling.** At DSV4.1 token times it is worth ~1–2%, not Qwen's ~4%.
   Keep it only if the chosen mechanism is compatible.
4. **Failure policy** for an NVMe read that fails or times out inside a graph: fail-stop,
   or drop-on-miss as a safety valve (§9.3).
5. **Drop-on-miss as a policy (option D).** Lossy. Only with sign-off plus an accuracy
   eval.
6. **DSpark ship rule**: ships only if measured α clears the DSV4.1 break-even (§10).
7. **Production downtime budget** for the GPU items and Phases 1–5.
8. **Disk**: reclaim the 167 GB of stale `.incomplete` blobs on nvme2.

---

## 13. Risks and open questions

- **Routing skew (`G`, `f`): measured** (§16.8), on the full 40-layer model in eager
  mode. Cold dynamic: `G` 115.0, `f` 18.5% (`f_after_warmup` 17.5%); seeded dynamic:
  `G` 111.7, `f` 18.4%. That is **better than the §15.2 proxy** (`G` ≈ 107, `f` ≈ 34%,
  static, 3 layers, prefill), so the envelope's pessimistic rows are not the operating
  point; it sits between §9.4's 108/10% and 108/25% rows. **Caveats that remain:**
  - 8 cold and 4 seeded sessions on one corpus, 256-token prompts and 128 new tokens; the
    `tier_sim` numbers replay the same cold-arm trace (sessions 0–7) whose routing they model,
    so they are fit and evaluated on the same 8 sessions (in-sample);
  - the live hot cache was 1,128 slots, below §4's 1,220–1,770, and the RAM tier 5,644 rows;
  - the model routes on its own EXL3 activations, whose 22-layer logits sit at the oracle
    noise floor (§16.6) and are unchecked at depth 40, so routing at depth 40 rests on
    unverified quality;
  - nvme1 is still unmeasured, so the x4 column is arithmetic;
  - the measured cost is not `G`/`f` alone: eager decode ran at 1.6 tok/s, about 0.38 s per
    token of it outside RAM-miss I/O (§16.12).
- nvme1's real throughput under `op-reth` write load (DRAM-less QLC) is unknown. So is
  its sustained write rate for the 205 GB copy.
- Option C was prototyped in Phase 3b (§17); A and B were not. Option B's host-callback
  serialization and stream-idle behaviour, and option C's SM cost while polling, remain
  unmeasured.
- ~~The EXL3 slot alignment~~: resolved (§14.1) — trellis needs 16 B, `suh`/`svh`
  need 8 B (checked in both the main and coop kernels), and padding happens in the
  RAM→VRAM segment copy, not as a separate admission-copy cost.
- The §4 KV figure: whether all 40 layers allocate 584 B/token.
- **sm_120 coverage of the indexer / FlashMLA**: measured on the truncated model
  (§15.2). FlashMLA works via `flash_mla_sm120` plus FlashInfer sparse MLA (64-token
  pages, after splitting the c2 extra KV pool to match); the c2 indexer's DeepGEMM
  logits work via the `lucifer1004/DeepGEMM-sm120` fork; the prefill indexer runs the
  torch path (`SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`). **Candidate indexer (3a,
  §16.7):** on the full model it does not run on sm_120. `SGLANG_OPT_USE_TOPK_V2` is
  forced off there (the kernel needs more than the 99 KB of shared memory), so layers 20
  and above take the mask decode path (`a0c5a6b81d`). **Still open:** CUDA-graph capture
  on sm_120 is untested (3a ran eager only).
- **EXL3 3.0 bpw quality on our workload**: measured on the truncated model (§15.2) —
  SGLang vs. reference oracle top-1 0.921, mean |Δlogprob| 0.084, below the plan's bar
  (≥0.98 / ≤0.05) but **accepted on a noise-floor ruling**: two legitimate oracles that
  differ only in KV rounding disagree more than SGLang does (top-1 0.887, mean |Δlp|
  0.111), and the 3-layer model's logits are flat enough (median top-1 probability
  0.092) that tiny numeric differences flip the argmax, so the absolute bar isn't a
  meaningful gate at this depth. Risk if the ruling is wrong: a sub-1%-per-layer bug
  could hide until the full-model run, where sharper logits would expose it. The
  card's published scores remain for the unquantized model, not this quantization. **At
  22 layers (3a, §16.6)** SGLang-vs-oracle is top-1 0.794 / mean |Δlp| 0.331, **rejected**
  against the bar and **marginally above** the one-prompt noise floor (0.809 / 0.302 vs
  SGLang's 0.792 / 0.316 on the same prompt). The 40-layer oracle does not fit (dense bf16
  > 31.4 GiB), so **full-depth quality stays open**.
- The Engram corpus is narrow (repetitive financial text), so general traffic will
  reuse less than the measured curve in §5.
- The global-singleton cross-layer KV reuse needs a batched design. Upstream presumably
  solved it; confirm during the `dsv41` merge.
- The 8B/16B activated-param split is unexplained.
- Host memory is shared with co-tenants, and swap use varied between 30 and 60 GB in
  one day.

---

## 14. Phase 0 findings

Task 6. Worktree `dsv41-worktrees/dsv41` @ `03c1617238`, merged `upstream-scope/dsv4.1`
tip `a5b84f11e5`; exllamav3 cloned `--depth 1` at research time (commit not pinned by
upstream, MIT). Line numbers below are from that state; the brief's line numbers had
drifted from a prior snapshot, so symbols were located by name.

### 14.1 exllamav3 alignment requirements

**(a) Minimum alignment.** `trellis` needs **16 bytes**; `suh`/`svh` need **8 bytes**.
- The fused MoE kernel's B pointer (`trellis`) is `const uint16_t*`
  (`EXL3_MOE_KERNEL_ARGS`, `exllamav3_ext/quant/exl3_moe_common.cuh:31-39`), threaded
  through `moe_gemm_tile` (`exl3_moe_kernel.cuh:25-52`) into
  `exl3_gemm_kernel_inner` (`exl3_gemm_inner.cuh:26-27`, parameter `B`). There, the B
  tile is cast `(const int4*) gl_b_ptr` and copied with `cp_async`
  (`exl3_gemm_inner.cuh:264-267`), which lowers to `cp.async.cg.shared.global` with a
  hard-coded 16-byte transfer size (`ptx.cuh:159-168`, `cp_async`: `const int bytes =
  16`). `cp.async.cg` requires both its global and shared addresses to be 16-byte
  aligned (CUDA `cp.async` semantics); nothing in the kernel checks or corrects for
  misalignment, so an unaligned trellis pointer is undefined behavior, not a slow
  path.
- `suh`/`svh` are `const half*` arrays (`exl3_moe_common.cuh:33,35,37`) offset in
  128-element (256 B) strides (`exl3_moe_kernel.cuh:143,150,212-214,260-271`;
  `exl3_gemm_kernel.cuh:25,66,74,163,202,276,284`) and loaded as `half4`
  (`hadamard_inner.cuh:106,112`, `had_hf_r_128_inner`). `half4` is declared
  `__align__(8)` (`exllamav3_ext/util.cuh:8`), so the vectorized cast needs only
  8-byte alignment — 256 B stride from an 8-byte-aligned base always lands 8-byte
  aligned, so in practice any tensor placed on an 8 B (or coarser) boundary works.
- No file in exllamav3 asserts or requires alignment wider than 16 B anywhere
  (checked `exllamav3_ext/quant/*.cu*`, `doc/env_vars.md`); the closest textual
  mentions of "128" are tile shapes (`MOE_TILESIZE_N`, `N_TILE=128`), not pointer
  alignment.

**(b) 128 B vs 16 B default.** **16 B is correct; 128 B was an unsupported guess.**
Evidence above shows the hard floor is 16 B (trellis) / 8 B (suh, svh); no exllamav3
code path needs more. The size cost of the wider default was also negligible either
way — 128 B padding adds at most 112 B/tensor x 12 tensors ≈ 1.3 KB per 13.3 MB
expert, so this was purely a correctness-of-requirement question, not a
size-budget one. **Changed**: `build_exl3_slot_layout`'s `alignment` default in
`python/sglang/srt/layers/moe/exl3_slot_layout.py` from `128` to `16`, in commit
`5322cb4847` (separate from this docs commit), with a new test
`test_default_alignment_is_16_bytes` in
`test/registered/unit/layers/moe/test_exl3_slot_layout.py` asserting the default
matches the explicit `alignment=16` layout. Run on divix01 (`taskset -c 0-63`):
`6 passed` (see task-6-report.md for full output). This also answers the doc-figure
follow-up: DSV41_REFERENCE.md never quoted a padded/aligned slot-byte total (only the
raw 13,315,596 B row and the raw trellis offset 14,852), so no other §3/§4/§9.2
figure needed a change.

**(c) Padding site: RAM→VRAM segment copy, not a host memcpy.** Recommend padding
inside the existing `copy_expert_row_segments_gpu` device-side gather
(`python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh:200-213`), not as a
separate NVMe→RAM host memcpy pass.
- That kernel already takes a `{src, dst, bytes}` segment list per call
  (`copy_expert_row_segments_gpu_kernel`, used by `:200-213`) — exactly the shape of
  the 12-segment plan `Exl3SlotLayout.segments` already produces
  (`exl3_slot_layout.py:17-21`).
- Its per-row copy (`copy_expert_row_lane`, `:53-78`) checks `src`/`dst` alignment at
  runtime and **degrades gracefully**: 16-byte vectorized loads
  (`copy_expert_host_unit16`, `:35-44`, `ld.global.nc.v2.b64`) when both pointers are
  16-byte aligned, else 4-byte scalar words via `load_expert_host_word_noncoherent`
  (`:17`, `ld.global.nc.u32`) down to a final byte tail (`:74-77`). So even if the
  RAM-side source offset is not 16-byte aligned (which it usually will not be — the
  raw row sits at whatever offset the O_DIRECT superset read left it inside its RAM
  buffer, §9.2), the copy is still correct; it only loses the vectorized fast path
  for that one gather, while its VRAM destination — the only address `exl3_moe`'s
  `cp.async` cares about — is written at the slot's aligned offset regardless.
- A host memcpy at NVMe→RAM admission would need its own ~13 MB, ~1-2 ms/admission
  CPU copy (§9.2's existing estimate) on top of the link/NVMe time, and would still
  need the RAM buffer's own base address to be 16-byte aligned for the *host*
  memcpy's writes to be simply offset-correct. Padding at RAM→VRAM instead reuses an
  existing, already-tested kernel and avoids adding a new CPU-bound step to the
  admission path.
- Consequence for Phase 1-3: keep RAM-cache rows in their raw (unpadded, byte-8
  contiguous) 13,315,596 B layout — this also keeps the RAM tier's ~75 GB / ~5,630-
  expert budget (§9.1) exactly as measured, with no RAM-side padding tax — and do the
  per-tensor re-placement only in the RAM→VRAM `copy_expert_row_segments_gpu` call,
  using `Exl3SlotLayout` segments (source = raw row offsets, destination = the VRAM
  slot's 16-byte-aligned offsets).
- **Closed: the batched/coop variant needs no stricter alignment.**
  `exllamav3_ext/quant/exl3_moe_coop.cu` and `exl3_moe_coop_kernel.cuh` were checked
  directly (`grep -n "cp.async\|int4\|uint4\|__align__"` over both files: zero
  matches). The coop kernel loads the trellis through plain 32-bit word pointers —
  `const uint32_t* B32 = (const uint32_t*) (is_gate ? p.g_trellis : p.u_trellis)[local];`
  (`exl3_moe_coop_kernel.cuh:746`) and the same pattern for `d_trellis` (`:849`) — and
  `suh`/`svh` through `scale_h4`/`load_h4`, which cast to `const uint2*`
  (`unpack_h4`/`load_h4`, `exl3_moe_coop_kernel.cuh:148-166`; called at `:625,
  781, 789, 798, 916`). `uint2` is 8 bytes, matching the main kernel's `half4`
  requirement exactly — no wider alignment anywhere in the coop path. §13's matching
  risk entry is closed by this finding.

**Phase 3a update: what the RAM tier holds.** (c)'s "raw, unpadded row in the RAM tier" was
the plan before the expert framework landed; 3a built something different.
- The RAM tier is the **framework's per-name, inclusive pinned tier** (§16.2). It holds the
  six streamed names (`w13_trellis`, `w13_suh`, `w13_svh`, `w2_trellis`, `w2_suh`, `w2_svh`,
  13,315,584 B per row) as separate tensors, **split from the shards** on each RAM miss
  (or read from a repack, had Task 16 run; it did not).
- The split costs **~1.43–1.48 ms per row on the model thread** (split share 0.16 of a
  read + split, §16.4), and **2.14–2.18 ms per decode RAM miss** on the live run (§16.8).
- The padding site is unchanged from (c): the RAM→VRAM segment copy, with the slot layout
  supplying the aligned destinations.
- **`Exl3RamExpertCache`** (the bounded LRU host-RAM tier of raw rows, `e8a17adf6a`) was
  **deleted**; `git show e8a17adf6a` keeps it.
- **The split numbers are the input for any future BLOB host layout**: `superset_raw` 7.55 ms/row, split 1.44–1.48 ms/row, repacked 7.66 ms/row at batch
  8 on nvme2 (§16.4). On nvme2 the repack saves 16% of the read + split time (ratio 0.839), short of
  the plan's 0.8 threshold, so 3a did not build it.

### 14.2 Can upstream's `_HostTable` back a partial Engram RAM tier?

**The gather takes a row index, not a raw table offset**, but two things still have
to change for a genuine N-slot cache — the current design is a *shard* of the full
table, not a *cache* of it.

- `EngramEmbedding.forward` for the host-table (shared) layout calls `engram_gather`
  directly with `self.weight.data_ptr()`/`self.scale.data_ptr()` and the raw ids
  (`python/sglang/srt/layers/engram.py:738-757`). The Triton kernel
  (`python/sglang/kernels/ops/embeddings/engram_gather.py:17-42`) computes
  `local = idx - row_lo` and addresses `w_ptr + local * DIM + offs` — an index, not a
  byte offset the caller precomputes. So a slot map (id → slot number) can be
  inserted **before** the kernel call, by translating `ids` into slot ids (or by
  adding a `slot_map_ptr` argument the kernel dereferences before the existing
  `local` arithmetic), without changing the pointer-arithmetic style of the kernel.
- **What has to change to size the table at N slots instead of all rows:**
  1. **Allocation.** `EngramEmbedding._init_host_table`
     (`engram.py:703-716`) sizes the table at `n = num_embeddings` (shared layout) or
     `self.rows` (per-rank shard) — the full compressed-vocab row count (99,092 x
     24 rows/layer, §5), not a cache budget. This has to become a chosen N (e.g. the
     5 GB / ~19M-row budget from §5) independent of `num_embeddings`.
  2. **Addressing and miss semantics.** `_engram_gather_kernel`'s `owned = (idx >=
     row_lo) & (idx < row_hi)` (`engram_gather.py:33-34`) is a *contiguous shard
     range* test, correct for "this TP rank's slice of the full table," not "this id
     is resident in the cache." A slot-indexed cache needs a real lookup (e.g. a
     hash table or direct-mapped array keyed by `idx`) that returns either a slot
     number or a miss. Critically, today's "not owned" path **zero-fills and relies
     on the sharded all-reduce to sum in the owning rank's contribution**
     (`engram.py:773-787`, `_lookup`/`_owned_rows`); that trick is semantically wrong
     for a genuine cache miss (there is no other rank holding the row — it has to be
     fetched from RAM/NVMe, per §9.3), so the miss path itself, not just the
     addressing, needs a redesign.
- Net: `_HostTable`'s mechanics (memfd/anon mmap, `cudaHostRegister`, huge pages,
  `class _HostTable` at `engram.py:549-651`) are reusable as-is for a reduced-size
  backing buffer; the row-index gather convention is reusable as the calling
  convention. What is not reusable without changes is the allocation-size
  computation and the shard-membership test that currently stands in for "hit."

### 14.3 DSpark target layers and planner

- **Confirmed: `dspark_target_layer_ids = [37, 38, 39]`, and there is no separate
  draft checkpoint** — the draft is bundled inside the same EXL3/official checkpoint
  as prefixed `dspark_*` keys on the *target* model's own `config.json`. Verified by
  a read-only `config.json` read on divix01:
  ```
  ssh divix01 'grep -n "dspark_target_layer_ids\|num_nextn_predict_layers" \
    /mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/config.json \
    /mnt/nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/config.json'
  ```
  Both files agree:
  `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/config.json:158-167` has
  `"num_nextn_predict_layers": 3` and `"dspark_target_layer_ids": [37, 38, 39]`; the
  official snapshot's
  `.../snapshots/dba1be0a40aa45a94ad051997016db3960a90277/config.json:145,148` has
  the same two keys with the same values. This matches the "37-39" figure already in
  §10 from the DeepSeek reference code (`inference/model.py:1089-1157`) exactly.
- **Draft weights are the `mtp.*` tensors inside the main EXL3 checkpoint** — 3
  stages x 128 experts, verified by Task 2's
  `test/manual/dsv41/test_exl3_checkpoint_layout.py:27` (`test_draft_experts`):
  `build_exl3_expert_layout(EXL3_DIR, prefix="mtp")` asserts
  `(layout.num_layers, layout.num_experts) == (3, 128)` and `row_bytes ==
  17,739,276` — the same "one DSpark draft expert is 17,739,276 B" figure already in
  §3.
- **How our fork reads this from the config**: `checkpoint_bundles_dspark_draft`
  (`python/sglang/srt/speculative/dspark_components/dspark_config.py:164-176`)
  detects a bundled draft by checking for any of the prefixed `dspark_*` keys
  (including `dspark_target_layer_ids`) directly on the *target* hf config — no
  separate draft checkpoint path is consulted when they're present.
  `parse_dspark_draft_config`
  (`dspark_config.py:208-221`) then reads `dspark_target_layer_ids` off that same
  config object (`_cfg_get(draft_hf_config, "dspark_target_layer_ids", None)`,
  `:221`) into `DSparkDraftConfig.target_layer_ids`, which
  `resolve_spec_aux_hidden_state_config`
  (`model_executor/model_runner_components/spec_aux_hidden_state.py:198-208`) then
  assigns to `config.dflash_target_layer_ids`
  (`spec_aux_hidden_state.py:199-201`) — the value `set_dspark_layers_to_capture`
  (`models/deepseek_v4.py:4767-4775`, called from
  `attention_backend_setup.py:57-58`) ultimately installs as
  `self.dspark_layers_to_capture` (`deepseek_v4.py:4076`). So for the DSV4.1-Flash
  checkpoint this resolves to `[37, 38, 39]` with no separate draft checkpoint or
  extra download required — the values above are already fully determined, not
  merely a code path that will consume them once something else is downloaded.
- **Maximum verify tokens for graph capture** is set by `resolved_max_verify_len()`
  (`python/sglang/srt/speculative/dspark_components/dspark_planner.py:912-913`:
  `self.max_verify_len or (self.gamma + 1)`), where `gamma =
  speculative_num_draft_tokens - 1`
  (`dspark_gamma_from_num_draft_tokens`, `dspark_config.py:51-58`). That resolves to
  the same `speculative_num_draft_tokens` CLI value that
  `max_speculative_num_draft_tokens()` (`python/sglang/srt/runtime_context.py:2020`)
  reports, which is what actually sizes CUDA-graph capture:
  `model_runner.py:744-748` multiplies `graph_gather_batch_size` by
  `decode_num_tokens_per_req(num_draft_tokens=max_speculative_num_draft_tokens())`
  (matching §8's existing citation `model_runner.py:740-750`). So
  `DSparkVerifyPlanner`'s per-request dynamic verify length
  (`_dynamic_graph_tier`, `dspark_planner.py:135`) is bounded above by the same
  static `resolved_max_verify_len()` the graph was captured for — the planner picks
  a length at or under the captured maximum; it does not itself resize the graph.

### 14.4 Find the vLLM EXL3 implementation

**No official or single canonical implementation exists.** Searches run (no `gh`
binary on the laptop or divix01; used WebSearch/WebFetch instead, recorded per the
brief's fallback):
- WebSearch: `vllm-project vllm exl3 quantization github`
- WebSearch: `vllm EXL3 quantization pull request exllamav3`
- WebFetch: `https://github.com/vllm-project/vllm/issues/19896`
- WebFetch: `https://github.com/vcruz305/vllm-exl3`

Findings:
- **vLLM upstream**: [Issue #19896](https://github.com/vllm-project/vllm/issues/19896)
  ("[Feature]: EXL3 support"), opened 2025-06-20, **closed as not planned** (stale,
  90+ days inactive). No in-tree `--quantization exl3` exists in
  `vllm/model_executor/layers/quantization`.
- **Out-of-tree plugins**: several near-identically-described repos —
  `vcruz305/vllm-exl3`, `Blackwellboy/vllm-exl3`, `fattchris/vllm-exl3`,
  `lna-lab/vllm-exl3`, `joeynyc/vllm-exl3` — each described as "An out-of-tree vLLM
  plugin registering `--quantization exl3` for EXL3 (ExLlamaV3 trellis) packs,"
  serving "routed MoE experts and declared dense EXL3 tensors through ExLlamaV3 and
  optional native CUDA kernels." **Does support MoE** per that description. One
  fetched repo (`vcruz305/vllm-exl3`) mentions a compiled
  `exllamav3_ext.ngram_dequant` kernel (used for the n-gram/Engram-adjacent
  embedding path) but the README text pulled did not enumerate `exl3_moe`/
  `exl3_gemm`/`exl3_gemv` by name; it describes calling into "ExLlamaV3 and optional
  native CUDA kernels" generically. **Not confirmed**: which exact exllamav3 kernel
  entry points (e.g. `exl3_moe_kernel` vs a Python-level `nn.Module` call into
  exllamav3's own `linear.py`/`block_sparse_mlp.py`) these plugins call — the repos'
  source was not read past the fetched README summary.
- The repeated near-identical descriptions across five differently-named accounts
  suggest a template or mirrored project rather than five independent
  implementations; treat all of them as one unverified community source, not five
  corroborating ones.
- **Consequence for Phase 1-3**: there is no vetted prior-art integration to port
  from. The nearest genuine prior art remains exllamav3 itself (§7); any of these
  vLLM plugins is, at best, a reference for how someone else wired the same kernels
  into a serving loop, not a dependency or a source of ported code (license/
  provenance unverified, and vLLM's own maintainers declined the feature).

### 14.5 Earlier proxies for routing skew

Ran (on divix01, one small PDF, `pdftotext` present at `/usr/bin/pdftotext`):
```
ssh divix01 'pdftotext -layout /mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/DeepSeek_V41_Tech_Report.pdf - | grep -n -i -B2 -A6 "load balanc\|expert load\|routing\|utilization"'
```
**No per-expert load statistics or plots exist in the tech report.** The matches are
all qualitative/architectural, not measured skew data:
- §2.2 (line ~336-342, ~367-376): describes "modality-specific load balancing" —
  separate auxiliary-loss-free correction biases for text vs. image tokens, updated
  from "their respective expert loads" — a training-time mechanism description, no
  numbers.
- Later matches (routing-replay for RL, image-sharding load balancing, "expert
  routing" persisted for rollout resumption) are all training/infra sections
  unrelated to inference-time per-expert load distribution.
- No table, histogram, or per-layer/per-expert utilization figure was found anywhere
  in the report's text extraction.
- **The EXL3 calibration trace `cal_trace_dsv41_flash_workload.json` is confirmed
  not present** in the downloaded files at `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/`
  (51 files enumerated in §1; no file by that name or pattern). No expert or Engram
  data was read beyond this file-listing check, per the task's constraint.
- **Consequence for Phase 1-3**: `G` and `f` (§9.4) have **no proxy signal at all**
  from the tech report or the EXL3 export. The only remaining earlier proxy is
  Phase 1's truncated-model router statistics (§11 Phase 1, "Record router
  statistics from the truncated model as a first, partial skew signal") — that
  remains the first real data point, not confirmatory of anything found here.

---

## 15. Phase 1 findings

All runs on divix01's RTX 5090 (sm_120) on 2026-09-18, production down, under
`cc-gpu.lock`. Artifacts: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/`
(`ANA` below).

### 15.1 Window A (kernels on sm_120)

- **exllamav3 extension build:** `/usr/local/cuda-13.2/bin/nvcc` (13.2), 5 min 50 s
  wall for the first JIT build.
- **EXL3 GPU tests (`test_exl3_{ops,method,moe}_gpu.py`):** 31/31 pass after
  `aeff379ef7`. The first run was 20 pass / 11 fail: 5 were a test-harness bug (a CPU
  generator under a leaked `set_default_device("cuda")`); 6 were `rows=1` GEMM cases at
  0.66–0.79% relative error against a 5e-3 bound. That is exllamav3's regular kernel at
  m=1 (its GEMV path is off on Blackwell; `EXL3_GEMV=0` changes nothing); m≥8 is
  0.03–0.05%. Ruling: `rows=1` tolerance 1.2e-2 with the measured values recorded; the
  m=1 precision is parked for Phase 2b (shape-index sweep or upstream report).
- **Reference `inference/kernel.py` under tilelang (`ANA/windowA-smoke.log`):** ok
  `act_quant`, `fp4_act_quant`, `hc_split_sinkhorn`. `fp8_gemm` and `fp4_gemm` fail in
  the smoke harness only (wrong C dtype; `fill_cuda` unimplemented for
  `float4_e2m1fn_x2`); the oracle never calls them on this path. `sparse_attn` fails
  for real at the model's shape: 64 heads × 512 needs 141,312 B of dynamic shared
  memory, above sm_120's 99 KB. Fix: `ref_oracle.make_head_split_sparse_attn` runs it
  on 16-head groups (`d375a74e9a`, test `test/manual/dsv41/test_ref_sparse_attn_head_split.py`).
- **DeepGEMM on sm_120:** `sgl-deep-gemm` has no sm_120 paged MQA logits; the
  owner installed the `lucifer1004/DeepGEMM-sm120` fork (2.8.0+b6acafe) into the
  shared venv (production loads it on its next restart). Against an fp64 reference
  (`ANA/dg_check.py`): page 64 dense fp32 7.9e-8, sparse bf16 4.3e-3 (bf16 floor
  2.3e-3); page 128 dense exact, sparse mean 4.8e-4 with 2/1006 columns ~2 bf16 ulps at
  the row max (accepted: the indexer only ranks). On sm_120 the dense path needs fp32
  weights, the sparse path bf16, and `num_heads` must be 32 (V4.1's
  `index_n_heads`). Wired in `8709dde032`.

### 15.2 Window B (truncated bring-up, oracle, router skew)

The truncated model is `dsv41-trunc3`: layers 0–2 (compress ratios 0, 0, 2; layers 0–1
pure SWA), EXL3 experts, Engram from shard 47, `candidate_source_layer_id=-1`.

- **Phase 0 MoE suite (Phase 0 Task 1 Step 8):** same 41 files on both worktrees. Baseline `wt-stageb` (`797be6f678`): 861 passed.
  `wt-dsv41` (`bacaf8c63b`, post-merge): 860 passed, 1 failed —
  `test_expert_doorbell_copier.py::test_a_second_exhausted_drain_whose_copy_never_lands_aborts_after_the_fatal_wait`,
  a known timing-flaky test (owner: fixed on the main production branch, not yet in
  `dsv41`'s fork point `b59edf2dc4`); not a merge regression. No other difference.
  Resolves §6's and §11's "GPU suite pending".
- **EXL3 vs official FP8 `layers.1.engram.wkv`**
  (`test/manual/dsv41/test_exl3_vs_fp8_engram_wkv.py`): relative RMS 0.0392, cosine
  0.99923; the transposed orientation gives 1.414, which pins the orientation.
- **Launch smoke:** PASS at `124f0db954` (4 tokens, 107 s end to end including JIT).
  Needs `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` (shared venv has sglang-kernel
  0.4.6.post1; the branch wants 0.4.7). Load memory (`ANA/memtrace.log`): 3.4 GiB init
  + 3 × 4.74 GiB routed experts = 18.5 GiB. Fixes found on the way:

  | Commit | Fix |
  |---|---|
  | `e7fb0e29e2` | shared expert: probe `gate_up_proj` for `trellis` (EXL3 has no `weight`) |
  | `8709dde032` | candidate indexer: skip when `candidate_source_layer_id < 0`; allow DeepGEMM paged sparse logits on sm_120 |
  | `9050e4f52e` | EXL3 MoE: read `topk_weights`/`topk_ids` by name (the fused gate returns 4 fields) |
  | `a98ecbe65e` | force DeepGEMM indexer metadata on sm_120 for c1/c2 pools (decode) and prefill |
  | `124f0db954` | split the c2 extra KV pool (128-token pages) to 64 for FlashInfer's sm_120 sparse MLA |

- **sm_120 paths taken** (truncated model): attention decode `flash_mla_sm120`
  (sparse MLA, FlashInfer, 64-token pages); prefill sparse through the same backend
  with the torch prefill indexer (`SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`); c2 indexer
  logits via the DeepGEMM sm_120 fork; routed experts through `Exl3MoEMethod`
  (trellis weights resident, 3 × 4.74 GiB); the shared expert keeps a dense copy; other
  EXL3 linears use the native kernel and fall back to a per-call dense reconstruction
  above 144 rows (the 1.3 GB head among them, hence `mem_fraction_static` 0.7);
  candidate indexer off (no source layer below layer 20).
- **Oracle vs SGLang** (4 prompts × 256 tokens, `ANA/prompts.jsonl`). The plan's
  bar (top-1 ≥ 0.98, mean |Δlogprob| ≤ 0.05) was not met; per-layer bisection
  (`ANA/bisect/`) found:

  | Iteration | top-1 | mean \|Δlp\| | Finding |
  |---|---:|---:|---|
  | first run | OOM | — | `exl3._materialize` thread race: concurrent loads zeroed experts and double-allocated (29 GiB). Fixed `ca5c3e0ef8` (lock) |
  | vs `oracle-p4` | 0.66 | 0.43 | `routed_scaling_factor` (1.5) never applied to EXL3 MoE output on CUDA. Fixed `b45b6c3971` |
  | vs `oracle-p4` | 0.891 | 0.116 | layer-0 attention error fully explained by the window-KV storage format (reference fp8/32 over 512 dims vs FlashMLA fp8/64 over 448 + bf16 RoPE): predicted 0.0237, measured 0.0237 |
  | vs `oracle-p4fm` (FlashMLA KV everywhere) | 0.908 | 0.109 | compressed KV already rounds like the reference fp4; quantizing it the FlashMLA way over-corrects |
  | vs `oracle-p4fw` (FlashMLA window KV) | **0.921** | **0.084** | per-layer hidden error 0.63% / 1.08% / 2.65%; router overlap 0.994 / 0.986 / 0.969 |

  **Noise floor:** two legitimate oracles that differ only in KV rounding disagree
  more than SGLang does: `oracle-p4` vs `oracle-p4fw` top-1 0.887, mean |Δlp| 0.111.
  The truncated model's logits are flat (oracle top-1 probability median 0.092), so
  tiny numeric differences flip the argmax. **Ruling:** Step 5 is accepted on a
  noise-floor criterion (SGLang-vs-oracle disagreement ≤ oracle-vs-oracle, with no
  unexplained module in the bisect), since the absolute bar sits below the floor for a
  3-layer model. Cost if wrong: a sub-1%-per-layer bug could hide until the full-model
  run, which has sharper logits. The merged tree (`c55f1572b0`, §6.1) reproduces the
  final row exactly.
- **Router skew** (`ANA/router-corpus200.json`; 200 first turns, ≤1,024 prefill tokens
  each, reference-oracle routing):

  | Layer | Gini | unused experts | top 5% mass | top 20% | static hit 7.9% | 11.5% | 36.7% |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 0 | 0.603 | 1 | 0.307 | 0.619 | 0.389 | 0.471 | 0.806 |
  | 1 | 0.649 | 5 | 0.362 | 0.673 | 0.444 | 0.530 | 0.836 |
  | 2 | 0.754 | 2 | 0.502 | 0.787 | 0.585 | 0.659 | 0.899 |

  Hit rates are for an in-sample, per-layer static top-k cache (optimistic for a
  static cache; blind to within-session locality, which a dynamic cache can exploit).
  **Implication for §9.4:** with a static cache the mean VRAM hit is 0.553 at 11.5%
  residency, so `G` ≈ 240 × 0.447 ≈ **107** (the 45% rows), and ≈ **127** at 7.9%
  with DSpark resident. The RAM tier (36.7%) misses 15.3% of rows, so `f` ≈
  0.153 / 0.447 ≈ **34%** (29% at 7.9%). That is above the 25% line where §9.4 says the
  system is NVMe-bound: ≈ 243 ms/token (4.1 tok/s) on an idle x4 drive, ≈ 374 ms (2.7
  tok/s) on nvme2, compute excluded. This moves §9.4 toward its pessimistic rows. It
  is a first, partial signal: three shallow layers only (skew rises with depth here,
  Gini 0.60 → 0.75), prefill routing rather than decode, and static rather than
  dynamic caching. The full-model decode trace in Phase 3 is the real test.

## 16. Phase 3a findings

Phase 3a is EXL3 streaming on the expert framework, checked against the reference oracle
and traced for `G` and `f` on the full 40-layer model. Window C ran on divix01's RTX 5090
(sm_120) on 2026-09-19, production down, under `cc-gpu.lock`. Code state: the smoke (Step
5) ran at `a0c5a6b81d` and Step 6b at `3eac82a412`, with Step 7's arms right after in the
same window; divix01's `wt-dsv41` stayed at `3eac82a412` from 02:28:14 until after the
window, so Step 7 ran at `3eac82a412` too. The prefill-indexer gate `f80100db71` was
authored later (02:35:53) and **never ran on the GPU**; `env.sh` still sets
`SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`, so its absence does not change those runs. Artifacts:
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a/`
(`ANA` below; `ANA/window-c.log` is the chronological record).

**Everything below is eager mode** (`disable_cuda_graph=True`): no CUDA graphs, no graph
gather, and no in-graph RAM tier. That is Phase 3b (§11).

### 16.1 Deviations from the plan

1. **Owner scope cuts** (owner's own message, 2026-09-19 02:00):
   - full-reference oracle noise floor on **1 prompt** only;
   - corpus **cold 8 sessions** (not 16), **seeded 4 sessions** (`--skip 8 --n 4`, not 8);
   - `--prompt-tokens 256` (not 512), 128 new tokens;
   - **the seeded static arm was dropped.** So there is **no stall-free tok/s**, and
     promotion stalls cannot be isolated from tok/s (§16.12);
   - `tier_sim` is **in-sample**: it replays the cold-arm trace (sessions 0–7) whose routing
     it models, so it is fit and evaluated on the same 8 sessions. Its simulated hot cache
     was not seeded (`seeded: false`); the seed was only emitted from that trace afterwards.
     The seeded arm ran on **different sessions** (8–11) from the cold arm, so cold-vs-seeded is
     **not a paired comparison**.
2. **Hot budget:** `SGLANG_MOE_HOT_GPU_MB` 16,384 → **14,336** after the first smoke
   failed KV-pool sizing at load (minimum viable `mem_fraction` 0.8534 > 0.85).
3. **Oracle depth:** the 40-layer dense oracle OOMs at load (dense bf16 > 31.4 GiB;
   `ANA/oracle-full40-fw-oom.log`), so the comparison ran at **22 layers** (§16.6).
   Full-depth comparison stays **open**.
4. **Three sm_120 fixes were found in the window**, each red then green and reviewed
   (§16.7 for the first):
   - candidate indexer: `01d1f88c5b` (red) + `a0c5a6b81d`;
   - Engram `_fetch` allocated without `device="cpu"`: `0efe2df293` + `3eac82a412`;
   - the dense fp4 prefill indexer now requires `topk_v2`: `9cea40d588` + `f80100db71`, so
     `SGLANG_DSV41_TORCH_PREFILL_INDEXER` is no longer required on sm_120.
5. **Engram hit rate: unavailable for the arms** (§16.11).
6. **nvme1 fio: pending**, so the repack ruling is provisional (§16.4). Task 16 (the repack)
   did not run, because it is conditional on that ruling.

### 16.2 How 3a streams

- **Framework plugins.** The EXL3 layout plugs into the expert framework as an
  `Exl3ExpertFormat` (`layers/moe/exl3_expert_format.py`) and an `Exl3ShardRowSource`
  (`layers/moe/exl3_shard_row_source.py`). Rows are read from the original EXL3 shards with
  `SGLANG_MOE_EXPERT_ROW_SOURCE=shards` and `SGLANG_MOE_EXPERT_FILE_READER=uring_direct`.
- **Schema.** Six streamed names: `w13_trellis`, `w13_suh`, `w13_svh`, `w2_trellis`,
  `w2_suh`, `w2_svh`, **13,315,584 B per row**. §3's raw row is 13,315,596 B; the 12 B
  difference was not investigated here.
- **Inclusive pinned tier (§9.1).** `inclusive_pinned_tier = True`. The pinned tier holds
  every hot-resident row too, and the per-layer hot-slot count is clamped to what the
  pinned tier can hold inclusively (Task 3). **No layer was clamped** at the budgets that ran (zero
  `Expert hot cache clamps layer` lines in the smoke log; the arms' logs run below the
  level that prints them, so the arms carry no such line either way).
- **Gate.** `arg_groups/expert_stream_requirements_exl3.py` registers the EXL3 server-args
  gate. On the real full-model directory it resolves to `exl3 EXL3`
  (`ANA/gate-check.txt`); the host arena is unset.
- **Budgets that ran** (`ANA/env.sh`):

  | Setting | Value |
  |---|---|
  | Pinned host tier | `SGLANG_MOE_PINNED_HOST_MB=71680` → 5,644 rows (141–142 per layer, 141 × 40 + 4 = 5,644; 36.7% of 15,360), requested 75,161,927,680 B, resident 75,153,156,096 B |
  | Hot cache | `SGLANG_MOE_HOT_GPU_MB=14336` → **1,128 slots**, 15,019,978,752 B (7.3% of 15,360) |
  | Dynamic residency | `HOT_DYNAMIC=1`, update after 256 prefill tokens and every 32 decode forwards, min residence 8 forwards, no async promotions |
  | Off | graph gather, GPU residency update, doorbell, prefetch candidates |
  | Engram RAM | `SGLANG_DSV41_ENGRAM_RAM_GIB=5` |
  | Engine (smoke script) | `disable_cuda_graph=True`, `disable_shared_experts_fusion=True`, `context_length=4096`, `mem_fraction_static=0.85`, `chunked_prefill_size=512`, `max_running_requests=4`. The corpus arms use `trace_corpus`'s own `engine_kwargs` (same eager mode, `mem_fraction_static` default 0.85) |

  The live hot cache (1,128 slots) is **below the smallest simulated size** (1,220,
  §16.10): the 16,384 MiB first tried failed KV-pool sizing at load (§16.1).

### 16.3 Drive table

nvme2, `ANA/bench-nvme2.jsonl` (256 direct reads, queue depth 128, payload bytes):

| Batch | GB/s | ms per expert |
|---:|---:|---:|
| 1 | 1.637 | 8.13 |
| 4 | 1.749 | 7.61 |
| 8 | 1.759 | 7.57 |
| 16 | 1.736 | 7.67 |
| 32 | 1.766 | 7.54 |

nvme1: **pending**. No fio ran (`ANA/fio-nvme1-pending.txt`: a 20 GiB QLC write was not
requested). §9.4's 3.4 ms/expert for an idle x4 drive is still an estimate.

### 16.4 Split versus repack (Task 5; **provisional, nvme2 only**)

Layer 0, 16 rows, direct reads for the shards, buffered for the repack.
`ANA/split-bench-nvme2-qd{32,128}.jsonl`. ms per row, medians:

| QD | Batch | Raw superset read | Read + split | of which split (share) | Repacked | Repacked / split |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 7.864 | 9.311 | 1.453 (0.157) | 8.077 | 0.868 |
| 128 | 8 | 7.553 | 9.127 | 1.482 (0.163) | 7.662 | **0.839** |
| 32 | 1 | 7.733 | 9.220 | 1.429 (0.155) | 8.023 | 0.870 |
| 32 | 8 | 7.556 | 9.089 | 1.443 (0.159) | 7.656 | **0.842** |

Batch 1 is a single repeat, and so is the raw read at batch 8 (7.55 ms, the BLOB baseline);
only the split and repacked rows at batch 8 are five repeats. The repack keeps the small names
(`w13_svh`, `w2_suh`, `w2_svh`) buffered.

- **Ruling: skip the repack.** The plan's single threshold is 0.8 at batch 8, and both
  queue depths read above it (0.839, 0.842). Task 16 did not run, so there is **no repack
  wall time**.
- **Provisional.** The ratio is measured on nvme2 (~1.76 GB/s). If nvme1 reads faster and
  the ~1.5 ms/row CPU split stays fixed, the ratio could fall toward ~0.70 and flip the
  ruling; that estimate is arithmetic, not a measurement. The nvme1 re-run needs the
  owner's OK after the copy exists.
- **Kept as input for a future BLOB host tier:** `superset_raw` (7.55 ms/row) and
  `split_share` (0.16, ~1.5 ms/row on the model thread).

### 16.5 Load footprint, pinned-tier registration and smoke

Smoke (Step 5), attempt 4, `ANA/smoke.log`: a 640-token prefill plus 8 decode tokens.
- Load: **9.35 s**, **9.90 GB** (`Load weight end`).
- Pinned-tier registration for ~70 GiB (75,161,927,680 B): **not logged as a duration.**
  The log timestamps bound it: `Load weight end` 02:28:26 → `Pinned host expert cache
  startup` 02:28:52, so **at most 26 s** (that window also holds any other post-load
  setup).
- Hot cache startup line at 02:29:04; `cuda_allocated_bytes` 25,543,779,840;
  `max_total_num_tokens=1,000,704`; `available_gpu_mem=2.62 GB`.
- End to end **129.7 s**; peak GPU **30,097 MiB** (`ANA/smoke-gpu-mem.txt`).
- **Zero** `weights not found` warnings (non-expert or otherwise), zero clamp lines, zero
  tier-setup warnings, zero `DeepGemmCandidateIndexer` mentions. Task 17 Step 5's
  non-expert-weights check is clean. One related warning does appear: `Some weights are not
  initialized from checkpoints` lists 306 names in `smoke.log`, namely 259 `vision.*`, 7
  aligner/`image_*` names (a Qwen-VL wrapper artefact, presumably benign) and 40
  `model.layers.N.mlp.gate.e_score_correction_bias_vl` entries. The 40 gate-bias entries are
  the only ones that are not vision or aligner names; their `_vl` suffix suggests a
  multimodal-path parameter, but that was not checked.
- Attempts 1–3 failed: 1 on KV-pool sizing at hot 16,384 MiB; 2 and 3 in the candidate
  indexer (§16.7).
- Step 3's tests ran before the smoke: 21 passed (`ANA/wc-steps35.out`).

### 16.6 Oracle comparison (22 layers)

`ANA/compare-full22-fw.json`, four prompts:

| | top-1 agree | mean overlap | mean \|Δlogprob\| | max \|Δlogprob\| |
|---|---:|---:|---:|---:|
| SGLang-streamed vs oracle-fw, mean of 4 prompts | **0.794** | 0.829 | **0.331** | 5.973 |
| vs the bar | ≥ 0.98 | | ≤ 0.05 | |
| SGLang prompt 0 | 0.792 | 0.843 | 0.316 | 7.127 |
| Noise floor, prompt 0: oracle-ref vs oracle-fw | 0.809 | | 0.302 | |

`accept: false`. Per prompt top-1: 0.792, 0.816, 0.753, 0.816.
- **Against the noise floor,** SGLang's prompt 0 (0.792 / 0.316) sits **marginally above**
  it (0.809 / 0.302): 1.7 points of top-1 and 0.014 of |Δlp|. §15.2's fallback criterion
  (SGLang-vs-oracle disagreement ≤ oracle-vs-oracle) is **marginally missed**.
- The noise floor is **one prompt only** (owner scope cut), and its source is the
  `step6b RESULT` line in `ANA/window-c.log`; the reference-vs-oracle-fw comparison is not
  a file in `ANA`. The 4-prompt mean has no floor of its own.
- **Phase 1 at 3 layers (`dsv41-trunc3`) was 0.92 / 0.084** (§15.2). Disagreement grows
  with depth (22 layers: 0.794 / 0.331); these are different models, so this is a comparison of
  the two runs, not a per-layer measurement.
- **No bisect ran** in this window, so the disagreement is not attributed to a module.
- **Open:** the full-depth (40-layer) comparison, because the dense oracle does not fit.
  Quality at full depth is therefore unverified.

### 16.7 Candidate indexer on sm_120

**It does not run on sm_120.**
- `model_hook` forces `SGLANG_OPT_USE_TOPK_V2=False` on sm_120 (the kernel needs more than
  the 99 KB shared memory that sm_120 has).
- `make_candidate_indexer` gated only on `sm >= 100`, and `publish_decode` called
  `topk_transform_paged_v2` with a placeholder CPU `topk_metadata`, which crashed at
  `topk_v2.cuh:591` in layer 20 decode (`ANA/smoke-diag.log`).
- **Fix:** `01d1f88c5b` (red) + `a0c5a6b81d`. With `topk_v2` off there is no candidate
  indexer, and layers 20 and above take the mask decode path (Hopper-style, reviewed
  against the reference two-level selection; `candidate_block_size` 8).
- So §13's "candidate indexer not exercised" is closed as **"not applicable on sm_120"**.
  The mask path ran through both arms' decode (layers 20–39). CUDA-graph capture on
  sm_120 is **untested** (eager only).

### 16.8 Measured `G` and `f`

From the traces (`ANA/trace-{cold,seeded}.jsonl`; `ANA/boundary-stalls.json`,
`ANA/tier-sim-cold.json` `live` block). `G` is VRAM misses per decode token (of 240 routed
rows: 40 layers × 6), `f` the fraction of those that also miss the RAM tier.
`f_after_warmup` leaves out each session's first 16 decode tokens.

| | Cold dynamic | Seeded dynamic |
|---|---:|---:|
| Sessions (decode tokens) | 8 (1,024) | 4 (512), sessions 8–11 |
| `G` | **115.03** (48% miss) | **111.73** (47%) |
| `f` | **0.1853** | **0.1844** |
| `f_after_warmup` | 0.1749 | 0.1754 |
| RAM misses per decode token (`f`·`G`) | 21.3 | 20.6 |
| Decode RAM misses (trace) | 21,823 | 10,547 |
| `decode_read_ms_per_ram_miss` | 8.1005 | 8.1217 |
| `decode_split_ms_per_ram_miss` | 2.1759 | 2.1353 |
| `prefill_read_ms_per_ram_miss` | 7.6079 | 7.6014 |
| `prefill_split_ms_per_ram_miss` | 2.1607 | 2.1356 |
| Prefill RAM misses (trace) | 39,598 | 20,592 |

- The per-miss pairs are per RAM miss, decode calls only; the `prefill_` pair sits beside
  them because the periodic log line mixes phases. A decode miss costs **~10.3 ms**
  (8.10 read + 2.18 split), against §9.4's 7.0 ms for nvme2.
- **In-sample or not.** The cold arm needs no seed, so its `G` and `f` are plain
  measurements on 8 sessions. The seeded arm's seed was built from those 8 sessions and
  the arm ran on 4 other sessions, so it is **out of sample for the seed**, and not
  paired with the cold arm.
- **Framework cross-check: available, and it agrees.** The manager counters
  (`ANA/hot-metrics-{cold,seeded}.jsonl`, final line, decode phase, summed over 40 layers):

  | | Manager | Trace |
  |---|---:|---:|
  | Cold decode VRAM misses (`miss_rows`) | 116,875 | 117,794 (`G` × 1,024) |
  | Cold decode pinned-tier misses | 21,663 | 21,823 |
  | Cold decode promotions | 2,690 | |
  | Seeded decode VRAM misses | 56,578 | 57,204 (`G` × 512) |
  | Seeded decode pinned-tier misses | 10,460 | 10,547 |
  | Seeded decode promotions | 625 | |

  So dynamic residency ran on DSV4. Counters and trace agree to about 1% (0.7–1.1%).

### 16.9 Admission policy shapes `f`

The framework admits **every prefill miss** into the per-layer RAM tier (141–142 rows per
layer), in ascending expert order. A long prefill can therefore flush the tier before
decode. `tier_sim` models both cases; `prefill_admits=False` is **simulation only; not a
framework policy**.

Gap between the `prefill_admits` true and false rows (`ANA/tier-sim-cold.json`, RAM 5,644,
8 cold sessions of 256-token prompts, **in-sample**):

| VRAM slots | Boundary | `f`, admits | `f`, no admit | ms/token (nvme2), admits | no admit |
|---:|---:|---:|---:|---:|---:|
| 1,220 | 0 | 0.1699 | 0.1829 | 289.4 | 300.9 |
| 1,220 | 32 | 0.1897 | 0.2048 | 278.7 | 291.8 |
| 1,770 | 0 | 0.1921 | 0.2057 | 269.1 | 279.6 |
| 1,770 | 32 | 0.2198 | 0.2360 | 260.2 | 272.9 |

Beside the live `f` **0.1853** (`f_after_warmup` 0.1749).
- **At 256-token prompts the gap has the opposite sign to the flush hypothesis.** Not
  admitting prefill misses makes `f` **worse** by 0.013–0.016 (0.002 at 3,000 RAM rows),
  and modelled ms/token worse by 10–13 ms. Admitting them helps decode, presumably because
  a session's decode reuses its own prefill's experts (an observation, not tested).
- So the simulation gives **no evidence that a prefill admission policy is a lever** at
  256 tokens. It was **not run at 512 tokens or more**, where the flush is larger; that
  stays unknown.
- The live `f_after_warmup` is 0.0103 below `f` (cold; 0.0090 seeded), so the first 16
  decode tokens of each session carry a slightly higher miss rate. That warm-up effect is
  small.

### 16.10 `tier_sim` table

`ANA/tier-sim-cold.json`, replaying the cold trace, unseeded (**in-sample**: it replays the
same 8 sessions, 0–7, whose routing it models; `seed-counts.json` was emitted from this
trace afterwards for the seeded arm, which ran on sessions 8–11), 41,320 calls, `--update-prefill-tokens 256`. `G` and `f` are decode
misses per token and RAM-miss fraction; ms/token is §9.4's model (`G` × 1.11 ms + `f` × `G`
× `t_nvme`, plus decode-boundary promotions amortized per token), link plus NVMe only,
**compute excluded**. Rows marked ✗ are `prefill_admits=False`: **simulation only; not a
framework policy**.

**RAM 5,644 rows** (the budget that ran). Boundary is decode forwards between residency updates.

| VRAM | Boundary | Prefill admit | `G` | `f` | ms/token nvme2 | ms/token x4 |
|---:|---:|:---:|---:|---:|---:|---:|
| 1,220 | 0 | ✓ | 125.87 | 0.1699 | 289.4 | 212.4 |
| 1,220 | 0 | ✗ | 125.87 | 0.1829 | 300.9 | 218.0 |
| 1,220 | 32 | ✓ | 112.10 | 0.1897 | 278.7 | 201.0 |
| 1,220 | 32 | ✗ | 112.10 | 0.2048 | 291.8 | 207.4 |
| 1,290 | 0 | ✓ | 123.67 | 0.1727 | 286.8 | 209.9 |
| 1,290 | 0 | ✗ | 123.67 | 0.1857 | 298.1 | 215.4 |
| 1,290 | 32 | ✓ | 109.65 | 0.1942 | 276.4 | 198.6 |
| 1,290 | 32 | ✗ | 109.65 | 0.2089 | 289.2 | 204.8 |
| 1,540 | 0 | ✓ | 116.07 | 0.1826 | 277.2 | 200.9 |
| 1,540 | 0 | ✗ | 116.07 | 0.1959 | 288.0 | 206.1 |
| 1,540 | 32 | ✓ | 101.78 | 0.2069 | 267.6 | 190.2 |
| 1,540 | 32 | ✗ | 101.78 | 0.2230 | 280.9 | 196.6 |
| 1,770 | 0 | ✓ | 109.64 | 0.1921 | 269.1 | 193.3 |
| 1,770 | 0 | ✗ | 109.64 | 0.2057 | 279.6 | 198.4 |
| 1,770 | 32 | ✓ | 95.06 | 0.2198 | 260.2 | 183.0 |
| 1,770 | 32 | ✗ | 95.06 | 0.2360 | 272.9 | 189.1 |

**RAM 3,000 rows.** All four VRAM sizes give identical rows: the inclusive clamp cuts the
hot cache to **440 slots with all 40 layers clamped**, so VRAM size stops mattering.

| Boundary | Prefill admit | `G` | `f` | ms/token nvme2 | ms/token x4 |
|---:|:---:|---:|---:|---:|---:|
| 0 | ✓ | 162.70 | 0.2682 | 486.1 | 329.0 |
| 0 | ✗ | 162.70 | 0.2705 | 488.7 | 330.2 |
| 32 | ✓ | 150.90 | 0.2860 | 471.6 | 315.7 |
| 32 | ✗ | 150.90 | 0.2885 | 474.2 | 317.0 |

- The live run (1,128 slots, boundary 32, RAM 5,644) measured `G` 115.03, `f` 0.1853. The
  nearest simulated row (1,220 / 32 / ✓) gives 112.10 / 0.1897, so the simulation brackets
  the live run.
- A bigger VRAM cache lowers `G` but raises `f` (the misses left are harder), so modelled
  ms/token falls only from 278.7 to 260.2 ms over 1,220 → 1,770 slots (nvme2).
- **The RAM tier size is the sensitive knob:** 3,000 → 5,644 rows takes modelled ms/token
  from 472–486 to 260–289 ms on nvme2 (prefill admitted).
- The model is **I/O only**: for the live `G` and `f` it gives **276.9 ms** on nvme2
  (3.6 tok/s) and **200.1 ms** on an x4 drive (5.0 tok/s). Measured tok/s is in §16.12.

### 16.11 Engram hit rate

**Unavailable for the arms.** The corpus logs carry no `engram row cache:` line
(`trace_corpus`'s log level suppresses it). The only lines in `ANA` are from the two
crashed smoke attempts (`smoke-attempt3-noautotune.log`, `smoke-diag.log`): 6 lookups,
30,768 accesses, **1 hit** (rate 3.3e-5). That is a synthetic 640-token prompt that
crashed in decode, on a cold table, so it is **not representative**. §5's simulated 71.7%
ceiling stays untested on the full model.

### 16.12 Measured eager tok/s and TTFT

`ANA/corpus-{cold,seeded}.json`. tok/s is the mean of per-session decode tok/s. TTFT is
the **median over sessions, plus session 0 separately, never the mean**: with the overlap
schedule a session's TTFT can include one decode step from the previous session's
overshoot forward. Session 0 is 35–38 s slower than the median in both arms (a first-use
cost that was not isolated).

| | Cold dynamic | Seeded dynamic |
|---|---:|---:|
| Sessions | 8 | 4 (sessions 8–11) |
| Decode tok/s, mean | **1.604** (per session 1.43–1.95) | **1.628** (1.47–1.84) |
| ms/token (1000 / mean) | 623 | 614 |
| TTFT, median | 56.6 s | 59.8 s |
| TTFT, session 0 | 94.8 s | 94.9 s |
| Median decode forward | 0.602 s | 0.601 s |
| Mid-session boundary promotions | 8 forwards | 4 forwards |
| Promotion stall (lower bound) | 5.38 s | 0.99 s |
| Post-prefill promotion (part of TTFT) | 7 forwards, 87 background rows | 3 forwards, 37 background rows |

- **Seeded is not measurably faster than cold** (1.628 vs 1.604 tok/s), on different
  sessions and with per-session tok/s spanning 1.43–1.95, so this is not a result either
  way.
- **Stall is a lower bound.** `stall_s` sums each promotion forward's excess over the
  median plain forward, over mid-session boundaries only (the promotion after a session's
  prefill is `after_prefill_*`, part of TTFT). A forward counts as promoted only when it
  read from disk, so promotions served from the inclusive RAM tier (~1.1 ms of H2D per
  row) are classed as plain forwards. Even so it is 5.3 ms per decode token cold (1.9
  seeded) against ~620 ms per token.
- **No stall-free number exists:** the seeded static arm, which the plan used for the clean
  comparison, was dropped. Both arms include promotion stalls, so **dynamic-versus-static
  is not measured.**
- **Against Qwen's 19.36 tok/s** (MOE_EXPERT_TRANSFER) these arms are ~12× slower, at
  1.6 tok/s and ~0.6 s/token, well below §9.4's ~3–8 tok/s envelope and below the modelled
  3.6 tok/s (§16.10).
- **TTFT is ~57 s for a 256-token prompt**, and it is mostly RAM-miss I/O: 39,598 prefill
  RAM misses over 8 sessions at 9.77 ms (7.61 read + 2.16 split) is ~48 s per session.
- **Observation (controller's arithmetic, not a profile):** a decode step has ~21 RAM
  misses × (8.10 read + 2.18 split) ms = **~0.22 s** of I/O, against a median forward of
  0.60 s. So **~0.38 s per token is not RAM-miss I/O.** This is eager mode with no CUDA
  graphs, and the model's own compute is not costed. It is the next thing to profile, and
  it caps what any 3b I/O work can win.

### 16.13 Open

- Owner decision on 3b (§12.1 go/no-go bar).
- nvme1 fio and the split-versus-repack re-run on nvme1 (§16.4).
- The full-depth oracle comparison (§16.6), the Engram hit rate on the full model (§16.11),
  and a static-arm run to isolate stalls (§16.12).

## 17. Phase 3b findings (option C)

Phase 3b built §9.3's option C: a CPU io_uring thread serves RAM misses, and the device
posts the missed ids and waits for them. Decode runs under breakable CUDA graphs with
the routed MoE inside the graph. The GPU window (Task 16) ran on divix01's RTX 5090 on
2026-09-19 from 08:52 to 10:16 CDT, production down, under `gpu-run.sh`'s lock. Artifacts:
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/`
(`ANA` below; `ANA/window.log` is the chronological record). The ledger
(`.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md`) holds every ruling cited
here.

### 17.1 Scope and code state

- **Phases (R1'):** P1, decode under breakable graphs with the EXL3 MoE as an eager break;
  P2, the `exl3_moe` probe (the gate); P3, the pinned-tier graph gather and the fused
  `exl3_moe` in the graph; P4, the RAM-miss service (C++ io_uring thread, device post and
  wait kernels, fail-stop); P5, next-layer advisory prefetch; P6, the window and this
  record.
- **Code state.** `window.log`'s first line is `window start at 475e67af663`. Five bugs
  found in the window moved the tree during it (§17.8); each run's commit is in
  `window.log`:

  | Run | Commit |
  |---|---|
  | GPU regression (152 passed) | `9554a8ece1` |
  | §9.3 component acceptance | `9554a8ece1` |
  | T3 option C parity | `dad6262508` |
  | Full-model smoke, Engine fail-stop, corpus arm `p1` | `51e3eeb457` |
  | Corpus arms `c`, `cpf` (rerun) | `4b906b3d4f` |

  The Nsight capture (§17.7) started at 09:52:03, before the `c`/`cpf` rerun, so it ran at
  `51e3eeb457`. `parity-t3-p1.{json,log}` is from Task 5 (07:12), before the window.

  After the window, the final whole-branch review's fix round hardened the service up to
  `c1970d371f`. Those commits are not re-measured here:
  - demand records carry an `armed` flag, and an unarmed record only refreshes recency;
  - the hot set is pushed to C++ at each outermost host use;
  - the watchdog covers hung advisories, with limit max(30 s, 3 × the RAM-miss timeout);
  - the slot-map cache is rebuilt only when the map version bumps;
  - a failed evicting request now bumps the map version.

  They are correctness fixes on paths the corpus arms rarely or never take. The CPU suite
  and the option C GPU tests pass at `c1970d371f`
  (`.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/final-fix-report.md`).
- **Budgets.** `ANA/env-full.sh` sources 3a's `env.sh` (§16.2: pinned tier
  `SGLANG_MOE_PINNED_HOST_MB=71680`, 5,644 rows; hot `SGLANG_MOE_HOT_GPU_MB=14336`) and adds
  `SGLANG_MOE_EXPERT_GRAPH_GATHER=1`, `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS=2000`,
  `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=0` (`cpf` sets it to 1). `ANA/env-full-p1.sh` sets
  graph gather to 0. The Step 6 setting (`window.log`) was **`MEM_FRACTION=0.80`,
  `HOT_MB=14336`**, with no ladder rung needed. The `p1` arm ran with a hot budget of
  **11,288 MiB** = `HOT_MB` − 3,048 (option C's scratch, 3,195,740,160 B). Source:
  `corpus-p1.log`'s `Expert hot cache startup` line, `requested_bytes` 11,836,325,888 =
  11,288 MiB.
- **The DSV4 multi-stream overlap is off** for EXL3 breakable decode (the EXL3 gate,
  Task 5). The root cause of the first defect was found and fixed: `hc_stats_stream` was
  forked before the MoE, the EXL3 MoE break ended the segment, and `ffn_stats` launched
  off-capture (NaNs). The fix re-forks the stats stream after a break (`6f0ad28e16`). A
  second defect, the MQA alt-stream prepare in segment 0, has an unknown cause, and the
  two-attempt cap was reached. The gate keeps the overlap off, so its speed-up is
  forgone in eager and graph modes alike. **Its size was not measured.** Overlap
  *scheduling* (the scheduler's) stays on: `disable_overlap_schedule': False` in
  `smoke-full.log` and `corpus-c.log`.

### 17.2 P2 probe (gate)

`ANA/probe-exl3-moe.json` (Task 1, 05:56; layer 3, experts 0, 32, …, 352, 3/3/3 bits).
Task 16 Step 2 reran the probe test inside the 152-passed regression; the JSON on disk is
Task 1's. `rel_*` is relative error against the fp32 reference; `max_abs` is fused vs loop.

| `num_active` | Route set (remap) | `rel_fused` | `rel_loop` | `max_abs_fused_vs_loop` |
|---:|---|---:|---:|---:|
| 6 | 3,1,8,5,2,6 | 1.031e-3 | 1.038e-2 | 0.0095 |
| 6 | 11,4,8,0,7,2 | 9.750e-4 | 1.103e-2 | 0.0094 |
| 6 | 4,5,10,9,8,1 | **1.155e-3** | 1.108e-2 | 0.0160 |
| 6 | 10,4,11,9,1,6 | 9.670e-4 | 1.071e-2 | 0.0102 |
| 6 | 6,8,7,5,11,2 | 1.020e-3 | 1.094e-2 | 0.0096 |
| 6 | 1,10,2,6,11,0 | 1.045e-3 | **1.149e-2** | 0.0108 |
| 6 | 9,1,0,8,4,7 | 9.958e-4 | 1.096e-2 | 0.0100 |
| 6 | 0,10,4,8,2,3 | 1.032e-3 | 1.109e-2 | 0.0113 |
| −1 | 3,1,8,5,2,6 | 9.202e-4 | 1.038e-2 | 0.0096 |
| −1 | 11,4,8,0,7,2 | 8.814e-4 | 1.103e-2 | 0.0096 |
| −1 | 4,5,10,9,8,1 | 9.361e-4 | 1.108e-2 | 0.0156 |
| −1 | 10,4,11,9,1,6 | 8.635e-4 | 1.071e-2 | 0.0101 |
| −1 | 6,8,7,5,11,2 | 9.476e-4 | 1.094e-2 | 0.0096 |
| −1 | 1,10,2,6,11,0 | 9.204e-4 | 1.149e-2 | 0.0108 |
| −1 | 9,1,0,8,4,7 | 8.532e-4 | 1.096e-2 | 0.0099 |
| −1 | 0,10,4,8,2,3 | 9.457e-4 | 1.109e-2 | 0.0112 |

| `num_active` | Replay vs eager | `eager_us` | `replay_us` |
|---:|---|---:|---:|
| **6** | bitwise, 4/4 steps | 242.4 | **110.1** |
| −1 | bitwise, 4/4 steps | 265.7 | 260.3 |

- **Verdict: PASS**, `num_active` **6** (chosen for its 110.1 µs replay; −1 replays no
  faster than it runs eager).
- The fused kernel is **~10x closer to fp32** than `exl3_moe_loop`: max `rel_fused`
  1.16e-3 against max `rel_loop` 1.15e-2 (`num_active` 6).

### 17.3 Graph decode

Breaks per capture, from the `Breakable CUDA graph captured` lines:

| Run | Model | Arm | Capture |
|---|---|---|---|
| `parity-t3-p1.log` (Task 5) | `T3` (3 layers) | P1: EXL3 MoE as an eager break | `segments=5 breaks=4` |
| `parity-t3-c.log` | `T3` | option C | `segments=2 breaks=1` (graph and debug Engines) |
| `smoke-full.log`, `corpus-c.log` | full, 40 layers | option C | `segments=3 breaks=2` in each of 4 attention variants (`candidate_all`, `candidate_c2_all`, `candidate_unfiltered`, `candidate_filtered`) |
| `corpus-p1.log` | full | P1 | `segments=43 breaks=42` in each of the 4 variants |

- Option C leaves only the Engram lookups as breaks: 1 on `T3`, **2 on the full model**
  (the two Engram layers). P1 adds one break per MoE layer (3 + 1 on `T3`, 40 + 2 on the
  full model).
- **DSV4 decode had zero breaks before 3b; the ~40-breaks claim was prefill-only**
  (corrected in §8 and §9.3). §9.3's option A would therefore *add* ~40 breaks per decode
  forward. `corpus-p1.log` is that shape (42 breaks), and it ran at 1.973 tok/s against
  option C's 2.781 (§17.6).

### 17.4 R4 numerics

`ANA/parity-t3-p1.json` and `ANA/parity-t3-c.json`, `T3`, 32 decode tokens:

| Report | Comparison | `first_token_mismatch` | `max_abs_dlogprob` | Pass |
|---|---|---:|---:|---|
| `parity-t3-p1` | `eager_vs_graph` | None | 0.0 | ✓ |
| `parity-t3-p1` | `eager_vs_eager` (control) | None | 0.0 | ✓ |
| `parity-t3-c` | **`graph_vs_debug`** (the bitwise capture gate) | None | **0.0** | **✓** |
| `parity-t3-c` | `eager_vs_graph` | 18 | 1.653 | ✗ |
| `parity-t3-c` | `debug_vs_eager` | 18 | 1.653 | ✗ |
| `parity-t3-c` | `eager_vs_eager` (control) | None | 0.0 | ✓ |

**`R4: FAIL (fused-kernel numerics)` — first_token_mismatch 18, max_abs_dlogprob 1.653;
probe rel_fused/rel_loop 1.16e-3/1.15e-2** (`window.log`, Step 5).
- Before the divergence (steps 0–17) max |dlogprob| was 0.413, median 0.057. At step 18
  the tokens differ: eager 77062 at −1.064, graph 56642 at −2.718 (task-16 report).
- **The capture gate passes bitwise.** Graph and debug-eager run the same fused kernel,
  and they agree exactly, so capture introduces no error. The eager control is bitwise,
  which rules out run-to-run nondeterminism. What remains is the fused `exl3_moe` against
  the eager `exl3_moe_loop`.
- **Observation:**
  - Capture is not the cause: `graph_vs_debug` is bitwise.
  - Per layer, the fused kernel is **~7–10x closer to fp32** than `exl3_moe_loop` in
    every measurement below.
  - So the end-to-end mismatch at step 18 is a numeric difference between two kernels.
    These data do not attribute it to either one.
  - **No end-to-end fp32 logprob comparison was run on `T3`** (open, §17.8).

  Per-layer MoE error against the fp32 reference:

  | Source | Fused / graph vs fp32 | `exl3_moe_loop` vs fp32 |
  |---|---:|---:|
  | Task 1 probe (`probe-exl3-moe.json`, max over 8 route sets) | 1.16e-3 | 1.15e-2 |
  | Task 9 GPU test, 3 routes, synthetic rows (`task-9-report.md`) | 2.13–2.18e-3 (graph replay) | 1.50–1.57e-2 |
  | Task 14 GPU test, 2 routes with served misses, synthetic rows (`task-14-report.md`) | 2.148e-3, 2.292e-3 | 1.752e-2, 1.478e-2 |

  That is ~10x in the probe and ~7x in the Task 9 and 14 tests. The step-18 divergence
  was not bisected further.
- **By ruling this does not block P3** (ledger: "R4 fused-kernel mismatch does not block
  P3 done if graph vs debug-eager is bitwise identical; record R4 FAIL with numbers"). The
  fused kernel is a deliberate numeric change. Whether R4's bar should instead be set
  against the fp32 reference is for the owner.

### 17.5 §9.3 acceptance

| # | Criterion | Result | Source |
|---:|---|---|---|
| 1 | Overhead per layer on the RAM-hit path | **18.24 µs/layer**, ×40 = **0.73 ms/token** (test bar < 100 µs). The wait kernel's extra fatal-word read is included | `ram-miss-overheads.json` `hit_path_us_per_layer` |
| 2 | Stream idle on a forced NVMe miss | 1 row: **11.40 ms** (expected ≈ 8–11, just above). 6 rows: **61.69 ms** (expected ≈ 25–65). Thread: served 3, rows_read 13, read_errors 0 | `ram-miss-overheads.json` |
| 3 | A timeout fails stop without hanging | **Component:** `test_a_hung_read_times_out_and_everything_after_is_fast` and `test_a_forced_timeout_fails_stop_without_hanging` passed in Step 2 (`cuda-tests.log`). **Engine (`T3`):** fault `demand reads sleep 20.0 s after 50 demands` (09:20:16), capture `segments=2 breaks=1` (09:20:23), prefill (09:20:33), then `ERROR exl3 RAM miss: request 59 timed out or failed; the process must stop` and `RuntimeError: ... fail-stop` through `Scheduler._expert_doorbell_fail_stop_check` (09:20:37; decode, after capture). **rc 137**, not 124 (the Engine's own `kill_process_tree`, not `timeout`). Error to scheduler exit ~19 s; **error to process exit 65 s**. No process left on the GPU | `failstop-t3.log` |
| 4 | Capture under breakable with overlap scheduling on | `Breakable CUDA graph captured: shape=ShapeKey(size=1, stream_idx=None, variant_label=None, attention_variant='candidate_all') segments=3 breaks=2`, one per variant; `disable_overlap_schedule': False` | `smoke-full.log` |

- **The 65 s is sglang's crash path, not the fail-stop.** The fatal was raised at the
  first batch check after the 2 s wait timeout. The rest is the parent's SIGQUIT handler:
  `Sleeping 5 seconds before crash diagnostics` (09:20:37), then `Waiting 60.0 seconds for
  CUDA coredumps before exiting` (09:20:42), then `kill_process_tree` (09:21:42).
- The Engine fail-stop ran at `--mem-fraction-static 0.5`. Run 1 at `trace_corpus`'s
  default 0.85 OOMed in the sm_120 FlashMLA page-split buffer (`_split_kv_pages_to_64`,
  19.26 GiB; `failstop-t3-run1-oom.log`), the known `T3` issue (§17.8).

### 17.6 Corpus arms (R5)

`ANA/corpus-summary.txt` (from `corpus-{p1,c,cpf}.json`, `trace-{p1,c,cpf}.jsonl`); sessions
0–3, 256-token prompt, 128 new tokens, `mem_fraction_static` 0.80. tok/s is the mean of
per-session decode tok/s; ms/token is 1000 / mean. TTFT is the median plus session 0, as
in §16.12. `G` and `f` are as in §16.8; `f` counts **demand rows only**.

| Arm | Hot slots | tok/s mean | Per session | ms/token | TTFT median / s0 (s) | `G` | `f` | `f_after_warmup` |
|---|---:|---:|---|---:|---|---:|---:|---:|
| `p1`: graphs, EXL3 MoE as eager break, hot 11,288 MiB | 888 | **1.973** | 1.663 / 2.121 / 1.758 / 2.350 | 507 | 56.0 / 109.1 | 126.14 | 0.1518 | 0.1400 |
| **`c`**: option C, prefetch off | 888 | **2.781** | 2.180 / 3.144 / 2.269 / 3.530 | 360 | 56.7 / 98.6 | 126.92 | 0.1460 | 0.1353 |
| `cpf`: option C, prefetch on | 888 | **2.780** | 2.172 / 3.178 / 2.274 / 3.497 | 360 | 57.2 / 96.1 | 126.92 | 0.1463 | 0.1354 |
| Window C cold, eager (3a `corpus-cold.json`, sessions 0–3) | 1,128 | **1.664** | 1.433 / 1.768 / 1.510 / 1.947 | 601 | 55.6 / 94.8 | — | — | — |

- **`c` is +67% over Window C** on the same four sessions, and +41% over `p1`. `p1` is +19%
  over Window C.
- **Hot capacity.** Graph gather's scratch is 3,195,740,160 B (3,048 MiB) and comes out of
  the 14,336 MB hot budget, so `c` and `cpf` get **888 slots**
  (`Expert hot cache startup`: residency 11,824,238,592 B, scratch 3,195,740,160 B, in
  `corpus-c.log`). `p1` ran at 14,336 − 3,048 = 11,288 MiB with no scratch
  (`corpus-p1.log`: requested 11,836,325,888 B) and got the same 888 slots and the same
  residency bytes. **All three arms share one hot capacity, with no slot difference
  left**, and it is below Window C's 1,128 slots. So the `G` of 126–127 is not comparable
  with §16.8's 115.03 (1,128 slots, 8 sessions); the `p1`–`c` `G` gap (0.78) has no slot
  difference behind it.
- **Prefetch is inert under this predictor.** `cpf`: `advisories` 132,
  `advisories_skipped` 0, `advisory_rows` **30** over 511 decode tokens (`c`: 0 / 0 / 0).
  **`cpf − c` = −0.001 tok/s**, inside the per-session spread (per-session differences
  −0.008 / +0.034 / +0.005 / −0.033, against a 2.180–3.530 range within `c`).
- Thread counters (cumulative, from the last `graph_step` line; the atexit log line is
  never written, §17.8 bug 3): `c` served 7,212, touch_only 13,588, rows_read 9,663,
  evictions 26,202; `cpf` served 7,225, touch_only 13,575, rows_read 9,718 (demand +
  advisory), evictions 26,255. Both: read_errors 0, overruns 0, late_after_fatal 0,
  no_victim 0.
- `decode_tokens`: `p1` 512, `c` 511, `cpf` 511. `c` and `cpf` have 495 `graph_step` lines,
  16 of which hold 2 steps (§17.8 bug 5).
  - **Why 511, not 512.** Under overlap scheduling the per-batch register read lags a step
    now and then, and the next check writes the lagged step. After the run's last replay
    that next check never comes, because `Engine.shutdown` SIGKILLs the scheduler. So the
    final lagged step is never recorded. Per-session `routed_rows / 240` gives 127 or 128
    steps, so nothing else was lost (task-16 report, bug 5; Task 16 review M1).
  - A lagged step at a session boundary is written after the next session's prefill, so it
    counts as that session's first warmup step.
  - `p1` (eager MoE break, per-layer lines, no lag) counts all 512.
  - The effect is one step in 512 (~0.2%) on `G` and `f`.

### 17.7 Per-token breakdown

`ANA/prof-graph.nsys-rep` (6.4 MB, kept): option C, prefetch off, one unseen session
(`--skip 12`), 64 new tokens, `--cuda-graph-trace=graph`. Capture from decode step 16 to
61 (`prof-graph.log`); the window of 14.80 s holds **35 decode steps** (summed from
`trace-prof-graph.jsonl`'s `routed_rows / 240`). The session gave 2.102 tok/s, TTFT
96.0 s. Analysis: `prof-graph-analysis.log`, `prof-graph-stats_{cuda_api_sum,osrt_sum}.csv`,
`prof-graph-union.py`, `prof-graph-sched.py`. Eager column: `prof-summary.md` (Window C,
`prof-decode.nsys-rep`, 3a).

| Per token | Option C graph | Eager (Window C) |
|---|---:|---:|
| ms/token in capture | **423** (14.80 s / 35) | 790 |
| ms/token just after capture | **261** (12 steps) | 690 |
| Profiler overhead | **~62%** | ~15% |
| GPU busy | **367 ms** (union of graph, kernel and memcpy intervals, 87%; includes the in-graph RAM-miss waits, so it is not compute) | 238 ms (kernels) |
| GPU idle | **56 ms** | ~550 ms |
| `cudaGraphLaunch` | 3.1 calls, **203 ms** (~65 ms each, blocking; 3 segments per step) | — |
| `cudaMemcpyAsync` (D2H copies) | 26 calls, **138 ms** | ~835 calls, 168 ms |
| `cudaStreamSynchronize` | 14 calls, **26 ms** | ~784 calls, 51 ms |
| Kernel launches (`cudaLaunchKernel` + ExC) | **52 calls** (24 ms over the window) | ~7.1k calls, 57 ms |
| Scheduler thread off-CPU | **26 ms** | ~175 ms |
| `G` / RAM misses per token in the window | 147 / 16.9 | 122 / 16.9 |

**Prefer corpus tok/s (§17.6) over traced ms/token.** Graph-mode profiler overhead is ~62%
(423 vs 261 ms), and the after-capture figure rests on 12 steps. A graph-mode trace has
no graph-body kernel table, so **no kernel ranking** was made.

Which eager buckets moved:
- **Host work, ~340 ms/token** (Python, torch CPU ops, bookkeeping) plus ~57 ms of launch
  API: launches fell from ~7.1k to 52 per token. The scheduler now spends its time
  blocked inside `cudaGraphLaunch` (203 ms/token) and `cudaMemcpyAsync` (138 ms/token),
  each waiting for the previous segment, in-graph waits included.
  - **Residual host work: ~28–54 ms/token.** This is span, less CUDA runtime time on the
    scheduler thread, less the part of off-CPU time that falls outside API calls
    (`prof-graph-analysis.log`):
    - span 14.798 s;
    - runtime 12.902 s (`cudaGraphLaunch` 7.105 + `cudaMemcpyAsync` 4.842 +
      `cudaStreamSynchronize` 0.925 + launches 0.024 + event/capture queries 0.005);
    - off-CPU 0.924 s.
  - The split of off-CPU time between inside and outside the API calls was not measured.
    So the residual lies between 14.798 − 12.902 − 0.924 = 0.972 s and
    14.798 − 12.902 = 1.896 s, i.e. **27.8–54.2 ms/token** over 35 steps.
  - Eager: ~340 ms/token plus ~57 ms of launch API.
- **Syncs, ~220 ms/token** (168 `cudaMemcpyAsync` + 51 `cudaStreamSynchronize`): now 164 ms
  (138 + 26) in ~40 calls instead of ~1,600. The remaining copies are the Engram break's
  syncing copies, and their time is mostly waiting on the GPU.
- **Read + split, ~182 ms/token** on the scheduler thread (off-CPU ~175 ms): moved to the
  io_uring thread. Scheduler off-CPU is 26 ms/token. RAM misses per token are unchanged
  (16.9), so the read time now sits inside the graph as the wait kernels' time, within
  the 367 ms GPU busy. It was **not measured separately**.
- **Zero-copy gather, ~172 ms/token** (`_gather_host_rows_kernel`): **not measured.** The
  graph-mode trace has no graph-body kernel table.
- **GPU idle** fell from ~550 to 56 ms/token.

### 17.8 Known properties and open items

**Known properties:**
- **Fail-stop latency (D15).** The forward that timed out finishes with that layer's MoE
  output dropped (later layers take the sticky fast path). Without overlap scheduling its
  tokens are the last emitted; with overlap one more forward may run before the
  after-batch check raises. The Engine run took 65 s from the error to process exit, all
  of it sglang's crash path (§17.5).
- **Predictor limitation.** The advisory for layer L+1 uses the previous token's routes,
  and those rows are usually already in RAM. A low `advisory_rows` (30 over 511 tokens)
  measures the predictor, not the mechanism.
- **Scratch VRAM.** Graph gather's scratch is 3,195,740,160 B (≈3.0 GiB) out of the hot
  budget: 888 slots instead of 1,128 at 14,336 MB.
- **Profiler overhead** in graph mode is ~62% (§17.7).
- **`f` and `rows_read`.** The trace's `f` counts demand rows only; the thread's
  `rows_read` includes advisory rows (demand = `rows_read − advisory_rows`).

**Bugs found and fixed in the window** (red / green, all pushed):

| # | Found at | Bug | Fix |
|---:|---|---|---|
| 1 | Step 2 | `env-full.sh`'s `SGLANG_MOE_EXPERT_ROW_SOURCE=shards` leaked into `test_expert_pinned_graph_gather_cuda.py` (4 failures; test isolation, not a product regression) | FW `8a52ba1aa2` (fixture clears the var), merged at `9554a8ece1` |
| 2 | Step 4 | Every option C Engine was refused at argument resolution: the first offload pass runs before `parse_cuda_graph_config`, and the EXL3 gate read the raw `None` as eager decode | `468888abe4` / `dfdd3f798f`, test narrowed at `dad6262508` |
| 3 | Step 8 prep | `Engine.shutdown` SIGKILLs the scheduler, so the atexit thread-counter line is never written | `e407139c06` / `e5ea4aaeee`: `graph_step` lines carry the cumulative counters |
| 4 | Step 6 | `trace_corpus.py` left the Engine at log level `error`, so capture, thread-start and hot-cache lines never appeared | `55f2541d53` / `51e3eeb457`: `--graphs` sets `log_level="info"` |
| 5 | Step 8 | Under overlap scheduling a per-batch register read lags one step now and then; `live_summary` counted lines as tokens (495 instead of 511; `G` 131.0 instead of 126.9) | `8344f30e61` / `4b906b3d4f`: lines carry `steps`; `c` and `cpf` rerun |

**Open:**
- **The MQA alt-stream defect** in segment 0 (§17.1). The DSV4 multi-stream overlap stays
  off for EXL3 breakable decode until it is found; its speed-up is unmeasured.
- **Raw JSON `--cuda-graph-config` crash** at two framework sites in
  `arg_groups/memory_hook.py` (`cc/moe-expert-plugins`):
  - `:174-176`, the graph-gather check, reads `.decode` off any non-None pre-parse value;
  - `:208-211`, the Qwen4 PLE staging check, reads `.prefill` the same way.

  An explicit JSON dict with either feature enabled would raise `AttributeError` in the
  pre-parse pass. Flag-only
  launches pass `None` and are unaffected. Not fixed (framework branch, R3 scope).
- **Trace lines written before `4b906b3d4f` undercount steps under overlap.** For such a
  trace use `routed_rows / routed_rows_per_step` (240 on the full model).
- **P1 `T3` FlashMLA page-split OOM at high mem fractions.** The sm_120 page-split buffer
  scales with the KV pool: `T3` needs `mem_fraction_static` 0.5, or 0.8 with a
  `max_total_tokens` cap (`graph_parity.py`). The full model at 0.80 was unaffected.
- **R4** stays FAIL against the eager loop (§17.4).
- **No end-to-end fp32 reference for R4.** Neither path's `T3` logprobs were compared with
  an fp32 model, so the step-18 divergence is not attributed to either kernel (§17.4).
- Parked minors from the ledger that a reader may hit:
  - the RAM-miss thread is not stopped at a clean scheduler shutdown (the exit hook
    covers it; Task 14 M3);
  - a timeout during warmup surfaces only at the first batch check (Task 14 M5);
  - `pool_host/common.py:171` raises a `TypeError` that masks a `cudaHostRegister` error
    (Task 13, outside scope);
  - the slot table's `contains()` is true at slot claim, before the read completes; the
    device reads the published map and eager paths pause the thread first, so production
    is unaffected (Task 15 ruling);
  - the dense NVFP4 path submits its promotion copy after host use closes (default no-op
    table only; Task 7);
  - a TODO to upstream an `allow_graph` option for `_EagerGraphView` (Task 3).
- **Engram hit rate, first reading on the full model.** Bug 4's fix put the
  `engram row cache:` line into the corpus logs. The last line per arm reads: `p1` 17,889
  hits of 73,680 accesses (**0.243**), `c` and `cpf` 17,409 of 73,680 (**0.236**), 1,024
  lookups each (`corpus-{p1,c,cpf}.log`). These are 4 cold sessions against §5's 71.7%
  exact-LRU ceiling over 1.5M tokens. The line's scope (which phases it counts, whether it
  is cumulative) was not checked, so §16.11 stays open.
- Not run in 3b: graph decode at bs > 1 or with DSpark, and nvme1.

## 18. After Phase 3b: where the step goes, and what can hide it (2026-09-19)

Artifacts: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-overlap/`
(`OVL` below). `OVL/SUMMARY.md` has the same numbers; scripts are named where used.

### 18.1 Code state and a fixed regression

- `dsv41` merged `cc/moe-expert-plugins` (`ad4998c0fe`, bringing mainline's offload
  presets, cuda_graph_config normalization and the stage-2 shortlist fix), then was merged
  with `origin/main` 993d1fccba into `master` (`afca79bc89`,
  `39362680bb`, pushed to `shared`). That candidate was validated against the Qwen4
  production config before main was fast-forwarded (decode tok/s and tail NLL at 2.5k and
  23k tokens matched main within run-to-run spread).
- **Regression, found by the first DSV4.1 launch after that merge:** every EXL3 launch
  with a hot cache was refused at argument resolution with `--moe-offload-preset off:
  SGLANG_MOE_HOT_GPU_MB requires SGLANG_MOE_EXPERT_STREAM=1`. The preset checks
  (`offload_presets.check_offload_config`, `needs_overlap_off`) applied NVFP4's hot-cache
  rules to every format; EXL3 streams under `SGLANG_DSV41_EXPERT_STREAM` and keeps overlap
  scheduling on, and `needs_overlap_off` would also have turned overlap scheduling off.
  The 146 GPU tests passed because none launches a full DSV4.1 Engine.
  **Fix `76829dff55`** (on `dsv41`; merged into main at `69c3ca4ce2`): both rules apply
  only when the launch's quantization method is NVFP4 or not yet known, as
  `memory_hook`'s per-format requirements already decide; the doorbell's overlap rule
  still applies to every format. Tests: `test_offload_presets.py` (+4 cases).
- One CPU test fails only in directory order and predates the fix:
  `test_server_args.py::TestMultimodalFeatureTransport::test_default_transport_is_cpu_for_unsupported_multinode_model`
  (passes alone; fails in `test/registered/unit/server_args` at `cef0875dac` too).
- The shared venv has `sglang-kernel` 0.4.6.post1; upstream now asserts 0.4.7, so every
  run here sets `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` (as `env.sh` always did).

### 18.2 Per-step breakdown (node-mode trace)

`OVL/prof-node.sh` is §17.7's `prof-graph.sh` with `--cuda-graph-trace=node`, 96 new
tokens and a 20 s capture; option C, prefetch off, session `--skip 12`, `wt-dsv41` at
`76829dff55`. Capture `segments=3 breaks=2` as in §17.3. The window holds 2,022 MoE layer
calls = **50.6 decode steps**. Session decode 2.289 tok/s (traced). Report
`OVL/prof-node.nsys-rep` (17 MB); analysis `layer_breakdown.py`, `align.py`, `gaps.py`.

| Per decode step | ms | Share of 391 ms |
|---|---:|---:|
| `exl3_ram_miss_wait_kernel`: the GPU waits for NVMe reads | **190** | 48.6% |
| `copy_expert_row_segments_gpu_kernel`: pinned RAM to VRAM gather | **128** | 32.7% |
| all compute (EXL3 GEMVs, `exl3_moe`, attention, mHC, router, …) | **16.8** | 4.3% |
| one residency-boundary stall (2.1 s, an outlier), averaged over the window (§18.6) | ~41 | ~10.6% |
| the two Engram breaks (graph → eager → graph) | ~11 | ~2.9% |

- **Everything is serialized.** The union of wait, gather and compute intervals equals
  their sum. Streams 141–143 are successive graph launches, never concurrent.
- Kernel totals over the window: wait 9,603 ms (2,022 calls), gather 6,464 ms (2,022),
  `exl3_gemv_int8_sq_kernel` 238 ms, `exl3_moe_kernel` 195 ms (96 µs/layer),
  `exl3_ram_miss_post_kernel` 15 ms (7.5 µs/layer). Stream 37 ran 390 off-graph
  `copy_expert_rows_gpu_kernel` calls (0.97 s) outside the decode graph.
- **Per layer:** wall 9.77 ms mean (median 4.77); wait 4.75 ms mean, **median 5 µs**, p90
  11.6 ms, so only **676 of 2,022** layer calls wait on NVMe; gather 3.20 ms mean; compute
  0.42 ms.
- **Per row** (window aligned to `trace-prof-node.jsonl`'s `graph_step` lines at 74.5%
  per-layer agreement): G = 121.2 VRAM misses and 18.8 RAM misses per step. Gather
  **1.055 ms/row**, at the PCIe line rate for a 13.3 MB row. NVMe wait **10.16 ms per
  RAM-miss row** (single-row median 10.55 ms; §17.5 measured 11.40 ms).
- **Tracing cost.** Node mode charges ~0.77 µs of `cudaGraphLaunch` host time per node
  (project CLAUDE.md), but the same session untraced ran 390 ms/step (py-spy run, §18.6)
  against 391 traced, so the cost is negligible here. GPU kernel durations are unaffected.

### 18.3 Upper bounds on overlap without prediction

Per step, from the per-layer records (`layer_breakdown.py`, `align.py`):

| Change | Bound | Share |
|---|---:|---:|
| Copy ∥ compute inside a layer (the MoE for experts already on the GPU, the shared expert and bookkeeping run under the gather) | 15.5 ms | ≤ 4.0% |
| Gather a layer's RAM-resident missed rows while its NVMe read runs | 20.8 ms | ≤ 5.3% (approximate: rests on the 74.5% alignment) |
| Engram lookups in the graph (removes both breaks) | ~11 ms | ≤ 2.9% |

Compute is too small to hide the copies behind, and across layers nothing can overlap
without knowing the next layer's routes, since layer L+1's attention needs layer L's MoE
output.

### 18.4 Prefetch: can a predictor hide the NVMe reads?

The NVMe wait is 49% of the step, and the drive is idle ~200 ms of each 391 ms step, so
reads started early, even some wrong ones, have room. The 3b advisory prefetch (`cpf`)
was inert (−0.001 tok/s, §17.6).

**Probe.** `OVL/probe-run.sh`: arm P1 (graph decode, EXL3 MoE as an eager break, hot
11,288 MiB = 888 slots, as §17.6's `p1`), sessions 0–3, 256-token prompts, 128 new tokens:
**1.982 tok/s** (`p1`: 1.973). `OVL/probe-site/sitecustomize.py` (analysis only, on
`PYTHONPATH`; no product code) wraps `Exl3MoEMethod._apply_streamed` and records, per
decode MoE call, the MoE input (= the gate's input), the routed top-6, and every expert's
tier (VRAM / RAM / NVMe) before the gather: 20,480 records = 512 tokens × 40 layers.
Analysis `lookahead.py` and `lookahead_rank.py` (outputs `lookahead.out`,
`lookahead_rank.out`).

- **Self-check:** each layer's gate from the checkpoint (`layers.N.ffn.gate.weight` and
  `.bias`; top-6 of sqrt(softplus(Wx)) + b) on the recorded input reproduces the recorded
  routes **20,480 of 20,480**. DSV4.1 has no hash-routed layers.
- NVMe rows: **19.15 per step** (18.26 over layers 1–39, the ones a lookahead can target).
- **The previous token's routes catch 0.000 of NVMe rows**: those experts were just read
  into RAM. That is why `cpf` was inert (`advisory_rows` 30 over 511 tokens).
- **Lookahead:** layer L+d's gate applied to layer L's MoE input, per step:

| Predictor | Recall, all routed | Recall, NVMe rows | Useful NVMe reads | Wasted NVMe reads |
|---|---:|---:|---:|---:|
| L+1 top-6 | 0.645 | **0.481** | 8.79 | 22.25 |
| L+1 top-8 | 0.719 | 0.572 | 10.44 | 41.04 |
| L+1 top-12 | 0.794 | 0.683 | 12.47 | 90.53 |
| L+1 top-24 | 0.876 | 0.810 | 14.79 | 284.44 |
| L+2 top-6 | 0.571 | 0.411 | 7.16 | 26.91 |
| L+2 top-12 | 0.716 | 0.591 | 10.31 | 98.39 |
| previous token | 0.376 | 0.000 | 0.00 | 0.42 |

"Wasted" counts predicted experts in the NVMe tier that the token does not route to;
predictions already in RAM or VRAM cost no drive time.

- **Precision falls steeply with rank.** NVMe-tier candidates at L+1: rank 1 0.681,
  rank 2 0.528, rank 3 0.375, ranks 4–6 0.193, ranks 7–8 0.081. By confidence
  sigmoid(20 × (score − 6th score)), the top bin [0.9, 1.0) holds **7.31 candidates per
  step, 4.52 useful (0.617)**; at L+2, 8.00 and 4.05 (0.506).
- **What it buys [estimate, not measured].** One layer of lead is ~4.8 ms (median layer
  without a wait), two ~10 ms, and a read takes ~10.2 ms. The top-confidence L+1 set
  hides ~4.5 × ~4.8 ≈ **22 ms/step (~6%)**; the L+2 set ~4 × ~10 ≈ **40 ms/step (~10%)**.
  Each spends ~30–40 ms of idle drive time on wasted reads, which is affordable only if
  prefetch reads never queue ahead of demand reads (reads serialize on nvme2: 6 rows took
  61.7 ms, §17.5).
- A wasted read still lands in the RAM tier and may evict a row a later token needs;
  neither that nor any later reuse is modelled.
- **The large levers remain the drive and the RAM tier**: every NVMe row costs ~10 ms on
  nvme2 (§1: Gen3 x2), and `f` falls with a larger pinned tier (§9.4). Neither needs a
  predictor.

### 18.5 DSpark does not run on the EXL3 stack yet

Asked to try `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`. Found by
reading code and checkpoints, not by a launch:

- **The full-model dir has no draft.** `dsv41-full40` (made by
  `make_truncated_model.py`) sets `num_nextn_predict_layers: 0` and drops every `mtp.*`
  key (187,350 keys). The original `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw` has 4,836
  `mtp.*` keys and `num_nextn_predict_layers: 3`. Both keep the `dspark_*` config keys, so
  `_handle_dspark` would point the draft at the target dir.
- **The draft loader has no EXL3 path.** `DeepseekV4ForCausalLMDSpark.load_weights` maps
  only upstream FP8/FP4 names; only the target calls `adapt_exl3_weights`.
- **The streamer refuses the draft's MoE.** `exl3_streamed` is process-wide
  (`SGLANG_DSV41_EXPERT_STREAM`), and `build_exl3_expert_streamer` raises when the layer's
  expert count (128 for the draft) differs from the checkpoint layout's 384.
- **Verify shape.** Block size 5 gives a 6-token verify; the EXL3 gate allows breakable
  decode graphs at max batch size 1 only, and option C's scratch and RAM-miss posting are
  sized for one token per step.
- Draft experts resident would cost 3 × 128 × 17.7 MB ≈ 6.8 GB, about half of the 888-slot
  hot cache. §10's rule stands: α must be measured before DSpark ships. **Parked by the
  owner (2026-09-19) in favour of the prefetch measurement.**

### 18.6 The "host gap" is the residency boundary

The ~41 ms/step of GPU idle between steps in §18.2 is not a per-step cost.

- **All 40 idle gaps over 20 ms (2,089 ms) fall inside one 2.1 s stretch** of the 19.8 s
  window (5.46–7.51 s; the post-kernel interval spanning it is 2,105 ms). A second,
  smaller stretch sits at 15.7 s (`gapclusters.py`).
- **Inside it:** every gap holds 6 off-graph `copy_expert_rows_gpu_kernel` launches on
  stream 37 (931 ms of that stream's 967 ms). The scheduler waits for them in
  `cudaStreamSynchronize` (923 ms). CPU samples on the scheduler thread: 73% spinning in
  that sync, 17% in a CPU-side ATen elementwise loop (`hostgap*.py`).
- **py-spy** (`spy-run.sh`: the same option C session without nsys, 30 s = 77 decode
  steps, 390 ms/step, as traced) attributes, per step averaged:
  - `graph replay`, 338 ms, mostly blocked in the Engram break's `lookup` waiting on the GPU;
  - `on_expert_distribution` → `_update_residency`, **45 ms**:
    - `_prepare_promotion` 19.5 ms, with synchronous NVMe reads of promoted rows under it
      (io_uring 15.9 ms, shard reads 6.5 ms as leaf frames);
    - `synchronize` 13.8 ms, waiting on the promotion copies;
    - `decide_residency_policies` 9.6 ms of CPU;
  - everything else outside replay (batch scheduling, result processing, sampling): **~3–4 ms**.
- **Cause.** `_update_residency` runs only at a residency boundary. Phase 3a's `env.sh` sets
  `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=32` and `SGLANG_MOE_HOT_ASYNC_PROMOTIONS=0`, so
  every 32nd decode forward promotes experts synchronously, reading from NVMe the rows not
  in RAM.
- **`SGLANG_MOE_HOT_ASYNC_PROMOTIONS=1` does nothing for EXL3.** EXL3 streams six tensors per
  expert (`EXL3_STREAMED_NAMES`), none with a dense source, so `stage_reassign` takes
  `_load_reserved` and returns no promotion to defer. That path promotes through the
  pinned tier in chunks (`ExpertHotCache._load_reserved_in_chunks`), and each chunk:
  1. pauses the RAM-miss thread (`host_use`);
  2. admits its rows into the pinned tier, reading NVMe for rows not in RAM;
  3. copies them to the GPU, one kernel per tensor (the 6 stream-37 kernels);
  4. waits in `current_stream.synchronize()` before the next chunk may evict pinned rows.

  Dropping the sync alone would let the next chunk or the resumed thread overwrite pinned
  rows a copy still reads. A real async path needs those rows protected until the copy
  completes, slots published on completion, and the NVMe reads off the scheduler thread.
  **Since 2026-09-19 the EXL3 requirements refuse the flag** instead of ignoring it.

**Paired arms** (`OVL/boundary/`, `boundary-arms.sh`): Phase 3b's `c` shape (option C,
sessions 0–3, 256-token prompts, 128 new tokens, 888 slots), `wt-dsv41` at `76829dff55`,
run back to back on 2026-09-19.

| Arm | Change | tok/s mean | Per session | G | RAM misses/token |
|---|---|---:|---|---:|---:|
| `c32` | none | **2.823** | 2.246 / 3.222 / 2.289 / 3.535 | 126.9 | 18.52 |
| `casync` | `SGLANG_MOE_HOT_ASYNC_PROMOTIONS=1` (ignored, above) | 2.817 | 2.231 / 3.220 / 2.285 / 3.534 | 126.9 | 18.52 |
| `c0` | `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=0` | 2.729 | 1.844 / 3.252 / 2.317 / 3.502 | 151.8 | 18.98 |

- `c32` reproduces 3b's `c` (2.781). `casync` equals `c32` to within 0.3%, with identical
  G and RAM misses: the flag is inert.
- **Decode-time promotions pay for themselves:** they cut G by 16% (24.9 rows/token, ~26 ms
  of gather at 1.055 ms/row). Net +3.4% tok/s, carried by session 0 (+22%); sessions 1–3
  are within ±1.3%. Median step 331 ms (`c32`) against 374 ms (`c0`).
- **A typical boundary is cheap.** Steps at the 32-forward boundaries (and the step after,
  for overlap lag) average 400 ms against 359 ms for other steps in `c32` (~82 ms of
  excess per boundary). In `c0`, which has no boundaries, the same positions show ~41 ms,
  so ~40–80 ms per boundary, **~1–2.5 ms/token (<1%)**. `c32`'s slowest step in 494 was
  1.6 s; p99 804 ms against `c0`'s 811.
- So the 2.1 s stall of §18.2's window, and py-spy's 45 ms/step average on the same
  `--skip 12` session, are a promotion burst on that session, not the typical cost.
  Extrapolating it (40–65 ms/token) was wrong.
- **Ruling:** keep `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=32`; defer a real async promotion
  path (under 1% on these sessions). The burst frequency below settles it.

**Burst frequency** (`OVL/burst-run.sh`, `burst_analysis.py`; the `c32` arm again, sessions
4–19, 16 sessions × 124 graph steps = 1,983 steps, 2.831 tok/s mean — the same as `c32`'s
2.823 on sessions 0–3). Bursts are rare, and the slow steps are not at boundaries:

| Quantity | Value |
|---|---|
| Steps over 1 s | **6 of 1,983 (0.3%)**, worst 1.40 s |
| Their positions | steps 30, 82, 110, 110, 124, 125 — none with k%32 in (0,1) |
| Boundary steps | 112, mean 417 ms, against 359 ms for the other 1,855 (**~115 ms excess per boundary, ~3.6 ms/token**) |
| Step ms | p50 344, p90 540, p99 776, p99.9 1,267, max 1,403 |
| Promotions per 32-forward window | median 25, max 832 over 48 windows |
| Windows with ≥200 promotions | 2; mean window 12.6 s against 10.5–12.0 s for the smaller buckets |

- **No boundary stall recurred.** Nothing approached the 2.1 s of the `--skip 12` session
  in 16 further sessions; the worst step is 1.40 s, in line with `c32`'s 1.6 s on sessions
  0–3. The six slow steps sit away from the boundaries, so they are NVMe tail latency on
  RAM misses, not promotions.
- **Promotion count barely moves the window.** The two ≥200-promotion windows (832 at the
  top) run 12.6 s against 10.5–12.0 s elsewhere: ~1–2 s spread over 32 forwards, which is
  the same ~3.6 ms/token the boundary mean shows.
- The per-boundary excess is higher than sessions 0–3 suggested (~115 ms against 40–80 ms),
  still **~1% of throughput**.
- **Ruling (2026-09-19):** no async promotion path, and no cheaper substitute either (a
  per-boundary promotion cap, or promoting only experts already in RAM). Both target ~1%,
  and the cap would give back some of the 16% G reduction promotions buy. The tail belongs
  to NVMe reads on misses, so prediction and prefetch (§18.4) are where the step time is.

### 18.7 Open

- ~~How common are promotion bursts?~~ **Settled** (§18.6, burst frequency): 6 steps over
  1 s in 1,983, none at a boundary, ~115 ms per boundary. No async promotion path.
- **Stream 37's copies are the boundary's promotions** (§18.6); they do not run during
  ordinary steps.
- **What is the NVMe tail?** The six slow steps are RAM misses whose reads ran long; the
  per-row 10.16 ms of §18.2 is a mean. Their distribution is unmeasured, and it caps what
  prefetch can hide.
- A prefetch prototype (confidence-gated L+1 or L+2 lookahead, prefetch reads queued
  behind demand reads) would check §18.4's estimate.
- Everything in §17.8's open list stands, including the raw-JSON `--cuda-graph-config`
  crash in `memory_hook.py`.

## 19. Expert-row mirroring: end-to-end arms (2026-09-19)

Two 205 GB byte-identical copies of the EXL3 expert checkpoint, on
`/mnt/nvme0/dsv41_flash` and `/mnt/nvme4/dsv41_flash`, selected with
`SGLANG_MOE_EXPERT_MIRROR_DIRS`. Four arms, sessions 0-3, 256-token prompts,
128 new tokens, `c32` settings. Each arm's per-drive read volume comes from
`/proc/diskstats` sectors-read deltas across the whole arm, which is what makes
the routing question answerable rather than inferred.
Script `analysis/dsv41-drive/run-mirror-arms.sh`; raw output in
`analysis/dsv41-drive/e2e/`.

| arm | graphs | GRAPH_GATHER | mirrors | mean decode tok/s |
|---|---|---|---|---|
| g-base | yes | 1 | none | 2.8286 |
| g-mirror | yes | 1 | nvme0+nvme4 | 2.8277 |
| e-base | no | 0 | none | 2.6947 |
| e-mirror | no | 0 | nvme0+nvme4 | 2.0126 |

`g-base` reproduces the recorded c32 baseline (2.823 tok/s) to 0.2%, so the
harness is measuring the same thing as before.

### Decode is untouched by mirroring; prefill is 1.84x faster

> **Superseded as a general claim, 2026-09-20: decode was untouched here because
> mirroring never reached it.** The arms below ran while `exl3_ram_miss_tables` built
> its path table from the source checkpoint and never consulted the row source, so
> every in-graph decode miss was served from nvme2 alone -- diagnosed two subsections
> down, in "Where the bytes went". Once that bypass was fixed (`099eadba33`), the same
> comparison gave **1.3482x on graph decode**, and Task 1's repeated arms gave 1.347x;
> both are in "Follow-up: the native reader now reaches the mirrors", below. Read the
> heading as "decode is untouched while the reader bypasses the mirrors", which is a
> statement about that bug, not about mirroring.

Per session, graph arms:

| session | g-base tok/s | g-mirror tok/s | g-base TTFT s | g-mirror TTFT s |
|---|---|---|---|---|
| 0 | 2.2540 | 2.2528 | 121.11 | 78.32 |
| 1 | 3.2269 | 3.2320 | 55.26 | 30.13 |
| 2 | 2.2923 | 2.2904 | 53.56 | 29.15 |
| 3 | 3.5411 | 3.5357 | 56.41 | 30.58 |

Decode throughput pairs to within 0.2% on every session: mirroring changes it
by -0.03% overall, which is nothing. TTFT falls by **1.84x** in steady state
(sessions 1-3).

### Where the bytes went, which is the whole explanation

| arm | nvme0 | nvme2 (source) | nvme4 | total |
|---|---:|---:|---:|---:|
| g-base | 0.00 | 401.30 | 0.10 | 401.40 |
| g-mirror | 137.59 | 127.21 | 137.89 | 402.69 |
| e-base | 0.00 | 257.93 | 0.03 | 257.96 |
| e-mirror | 198.52 | 0.38 | 198.51 | 397.41 |

All figures GiB.

In `g-mirror` the mirrors carry 275.48 GiB and nvme2 still carries 127.21 GiB.
Subtracting that residual from `g-base`'s single-drive total gives
401.30 - 127.21 = 274.09 GiB, within 0.5% of the mirrored 275.48. The two
arms move the same bytes; mirroring is byte-neutral end to end, which
independently confirms the split is correct at scale. nvme0 and nvme4 differ
by 0.2% (137.59 vs 137.89), so the static 1:1 policy holds across 275 GiB of
real traffic.

That residual 127 GiB is the finding. **`exl3_ram_miss_tables` builds its path
table from `layout.records[(layer, expert)].path` - the source checkpoint - and
never consults the row source** (`exl3_ram_miss.py:52-63`). When
`SGLANG_MOE_EXPERT_GRAPH_GATHER` is set, `pinned_tier_options` installs the
native `Exl3RamMissService` to own the tier's slots
(`exl3_expert_format.py:176-186`), and every in-graph decode miss is served by
the C++ reader from nvme2 alone. Only the eager row-source path - prefill -
reaches the mirror source.

So the 2.31x per-row gain measured in
`analysis/dsv41-drive/MIRROR_ROWS.md` lands entirely on prefill and **does
not reach graph decode at all**. Wiring a mirror-aware extent plan into the
native reader is the prerequisite for any decode benefit; until then
`SGLANG_MOE_EXPERT_MIRROR_DIRS` is a TTFT optimisation.

> **That prerequisite was met the next day** (`099eadba33`, the follow-up below), so the
> closing sentence no longer describes the code: `SGLANG_MOE_EXPERT_MIRROR_DIRS` is not a
> TTFT-only optimisation any more. The diagnosis above stands and is what the fix acted on;
> the 127 GiB nvme2 residual it explains is also the number to compare a later arm's residual
> against -- 2026-09-22's HTTP arm left 9.3 GiB there.

### Unexplained: eager + mirrors is slower

> **Superseded, 2026-09-20: `e-base` below did not reproduce, and the tier-warming
> explanation is refuted.** A counterbalanced re-run (B M M B, six arms, ~1.9 TiB,
> cache counters added for the purpose) reproduced `e-mirror` closely - TTFT
> 65.3/29.9/29.0/30.1 s against 59.2/29.6/28.9/30.0, decode 2.01-2.03 against
> 2.0126 - while `e-base` did not: it read 380.9 GiB of session bytes, not 257.96,
> and held TTFT at ~55 s instead of warming to 5.6 s. Both arms then read the SAME
> bytes, made identical cache decisions, and mirroring was faster in every session
> (1.85x steady TTFT, 1.19x decode) with byte-identical greedy output.
>
> Neither tier warms: occupancy hits capacity (5644/5644) inside session 0 in both
> arms and admissions equal evictions thereafter, so the "`e-base`'s tier warms and
> `e-mirror`'s does not" reading is refuted rather than merely unconfirmed.
>
> What is now unexplained is not the mirror arm's extra bytes - there are none -
> but `e-base`'s MISSING reads and its 5.6 s prefill. Ordering, tracing, the
> counters and the environment are ruled out by measurement; system state that day
> and an undiagnosed build difference are not.
>
> **Retired, same day: the code is exonerated.** The eager base arm was re-run at
> §19's own commit `1525e43ab9` and reads 380.7 GiB with TTFT 95.4/54.9/53.6/55.4 -
> indistinguishable from HEAD - with greedy output sha1s identical to HEAD's for all
> four sessions. The old code reads the same rows and computes the same answer, so
> nothing between `1525e43ab9` and HEAD caused it and there is nothing to bisect.
> `e-base`'s 257.96 GiB and 5.6 s prefill are not reproducible from their own commit
> and are retired, not merely unconfirmed: do not cite them, nor the 1.54x byte ratio
> or the 25% regression derived from them.
>
> By elimination the cause was machine state that day, and the cause is unknown.
>
> **The page-cache candidate is strongly weakened, and excluded for today's arms only.**
> It was attractive - the `g-mirror` arm ran immediately before `e-base` and read 127 GiB
> from the same drive, against the 123 GiB by which `e-base` undershoots today - but that
> coincidence is a mixed-basis artefact: 381 GiB is today's SESSION bytes and 258 GiB is
> §19's WHOLE-ARM total including ~24 GiB of startup. Whole-arm to whole-arm the gap is
> ~147 GiB against `g-mirror`'s 127.2 GiB, and they do not match. Three further facts cut
> against it: 127 GiB of cached rows would not fit beside the 70 GiB pinned tier in 188 GiB
> of RAM; `e-mirror` ran right after `g-mirror` had read ~137 GiB from each mirror and did
> NOT benefit, reading its full 397 GiB with TTFTs within 1-3% of today's; and §19's decode
> had already diverged in session 1 (2.16 vs 1.80 tok/s at an identical 55 s TTFT), which
> no "the cache warms from session 2" story fits.
>
> For TODAY's arms it is excluded by measurement: after ~2.3 TiB of reads, `fincore` shows
> 15.6 GiB of the 204.1 GiB source expert files resident, which is startup-scale. For §19
> itself there is no process-level evidence - that run wrote no env dump and its logs carry
> no reader line - so "it ran direct" rests on the config alone:
> `dsv41-phase3a/env.sh:14` sets
> `SGLANG_MOE_EXPERT_FILE_READER=uring_direct`; its mtime precedes the §19 run and
> neither `run-mirror-arms.sh` nor `eager-cache-arms.sh` overrides it or drops caches.
> The fd is opened `O_RDONLY | O_DIRECT` unconditionally when direct is set
> (`io/uring_file_reader.cpp:115`), and an unaligned destination is served through a
> page-aligned bounce the reader owns rather than by falling back to buffered reads
> (`:249-252`). The comment at `expert_file_reader.py:47`, "O_DIRECT is used when a
> destination is page-aligned, buffered reads otherwise", describes that bounce and
> reads as if there were a fallback; there is not. So the only way §19 could have read
> buffered is if that env var was not in force in its process, which nothing records.
>
> So §19's arm most likely requested ~147 GiB fewer bytes, which means it read fewer ROWS,
> which means its pinned tier was hitting where today's misses. Why is not recoverable:
> §19 set no trace path and had no per-row counters, and those counters exist only
> because this investigation added them.
>
> Detail and full per-session tables: `analysis/dsv41-drive/EAGER_ANOMALY.md`,
> commit `cd14545797`.

`e-mirror` is 25% slower than `e-base` (2.0126 vs 2.6947) while reading 1.54x
more bytes (397.41 vs 257.96 GiB). Its first two prefills are faster than
`e-base`'s (59.2 vs 85.4 s, 29.6 vs 55.1 s), as mirroring predicts, but then it
plateaus at ~30 s while `e-base` warms to 24.8 and 5.6 s. `e-base`'s pinned
host tier is warming across sessions and `e-mirror`'s is not.

Byte-neutrality is established by the graph arms above, so this is not each
root reading a full row. The remaining candidates are a tier-population
difference tied to the row source, or queue-depth behaviour: production reads
many rows per call, and mirroring doubles the extents per batch, whereas the
per-row bench measured one row per call. nvme4's fio QD6 result was
catastrophic (p50 255 ms) on an untrimmed file; that measurement was withdrawn
at QD1 on fresh data but **was never repeated at QD6 on fresh data**.

Recorded as an open anomaly. Not explained, and no conclusion about eager
mirroring should be drawn from this arm until miss counts are collected with
`SGLANG_DSV41_EXPERT_TRACE_PATH` and the arms are repeated interleaved to rule
out ordering.

### Follow-up: the native reader now reaches the mirrors (2026-09-20)

The bypass above is fixed. `exl3_ram_miss_tables` now builds a per-(row, expert,
part) extent table `[file, offset, length, dest_offset]` from the row source, and
the C++ reader submits one SQE per extent, so in-graph decode misses are served
from the mirrors. Arms re-run with graphs on and `GRAPH_GATHER=1`, same corpus
and settings as the table above. Script
`analysis/dsv41-drive/run-native-mirror-arm.sh`, report
`analysis/dsv41-drive/native_mirror_report.py`, raw output in
`analysis/dsv41-drive/native-mirror/`.

| arm | mean decode tok/s |
|---|---|
| base (mirrors off) | 2.8198 |
| mirror (nvme0+nvme4) | 3.8016 |

**1.3482x on graph decode**, against a ceiling of 1.38x recorded in the script
before the run so it could be falsified. Coming in just under a ceiling is the
expected shape; a result above it would have indicated a broken measurement.

Per session, paired:

| session | base tok/s | mirror tok/s | ratio | base TTFT s | mirror TTFT s |
|---|---|---|---|---|---|
| 0 | 2.2286 | 3.0651 | 1.3753 | 108.32 | 69.08 |
| 1 | 3.2245 | 4.2881 | 1.3298 | 55.12 | 30.07 |
| 2 | 2.2906 | 3.1942 | 1.3945 | 53.64 | 29.05 |
| 3 | 3.5354 | 4.6589 | 1.3178 | 56.22 | 30.58 |

Every session improves, in a 1.32-1.39x band. This matters more than the mean:
base's own sessions span 2.23-3.54 tok/s, a 1.59x spread wider than the effect
being measured, so a single session proves nothing and only the paired result
carries the claim.

> **What code these numbers measure, and one place they are misattributed.**
> The arms above ran at `099eadba33`, the commit that made the native reader
> build a per-extent table and reach the mirrors. That is what they establish,
> and the section is scoped to it correctly.
>
> They do **not** measure the two-bank pipeline. `ddcb0d55ff`, which introduced
> two-bank, quotes these same figures in its commit message as "Measured: mean
> decode 2.8198 to 3.8016 tok/s, 1.3482x". That attribution is wrong: the
> figures already existed in this file at `ddcb0d55ff`'s parent. The commit
> message cannot be amended, so the correction lives here. The two-bank
> pipeline's own effect is measured separately below, and it is **about +3%,
> not 1.35x**.
>
> Both arms are also n=1 per cell and carry no provenance. Against a base spread
> of 1.59x across sessions, wider than the 1.35x effect, the paired per-session
> result is what carries the claim and the mean does not. Treat these as the
> bypass fix landing, not as a baseline: Task 1's matched baselines are being
> collected separately, repeated and interleaved, with
> `scripts/dsv41/provenance.py` recording the resolved environment, the imported
> tree's HEAD and dirtiness, the reader mode actually in force, and a drive-idle
> check.


#### Task 1 matched baselines, and what two-bank is actually worth (2026-09-21)

The arms above were single shots without provenance. These are their replacement:
repeated, interleaved, each arm refusing to start unless its worktree is clean at
an expected sha, and each result carrying `scripts/dsv41/provenance.py`'s record
of the resolved environment, the imported tree, the reader mode in force and a
drive-idle check. Scripts `analysis/dsv41-drive/task1-baseline-arms.sh` and
`task1_arm_verdict.py`; raw output in `analysis/dsv41-drive/task1-results/`.
To run an arm, set `REFERENCE=<clean-reference.json>` (and `EXPECT_NEW`): the script refuses to start, exit 5, if it is unset or if an arm's
`git rev-parse <sha>:python` is not registered there. See `analysis/dsv41-drive/PIPELINE_BASELINE.md` section 2 and `task1-results/GENERATIONS.txt`.

Graph decode, `GRAPH_GATHER=1`, 4 sessions, 256 prompt / 128 new, 70 GiB pinned
tier. `multi_token_chunks` was 0 in every arm of the series, so the step-latency
percentiles are exact rather than smoothed.

| arm | code | mirrors | mean decode tok/s | n |
|---|---|---|---:|---:|
| new, mirrors off | `f6608901a3` | off | 2.903, 2.905, 2.907, 2.917, 2.919 | 5 |
| new, mirrors on | `f6608901a3` | on | 3.905, 3.927, 3.933 | 3 |
| old, mirrors on | `099eadba33` | on | 3.798 | 1 |

**The mirror effect is 1.347x** (3.92 / 2.91), reproducing the single-shot
1.3482x above with repeats and provenance. Run-to-run spread within a cell is
0.5-0.6%, far below the effect, and the four off arms span 2.903-2.919.

**New code beats old by about +3.2%** (3.92 clean new / 3.798 clean old; +3.5%
if `task1b-0` is included as the manifest does, giving n=4). The sign is
consistent across all four sessions individually (+3-5%, +3%, +4-5%, +2%) and
across arms (+2.8% to +4.2%).

**This is NOT two-bank's effect alone, and the distinction is the same one this
section corrects elsewhere.** The `099eadba33` to `f6608901a3` delta is four
commits touching `python/`, two of them behavioural: `ddcb0d55ff` (the two-bank
pipeline) and `cd14545797` (pinned-tier and Engram traffic counters, which touch
`engram_row_cache.py` and `expert_host_tier.py`). The other two, `be76ba501f`
and `6a606e2b33`, are comment-only and can be excluded by inspection. So +3.2%
is what those two behavioural commits are worth together. Attributing it to
two-bank alone would repeat, in this file, exactly the error this section
corrects in `ddcb0d55ff`'s commit message.

**The old arm's n is 1.** Five old arms have run; four were disturbed by machine
contention and are excluded, leaving a single clean measurement at 3.798 against
three clean new arms. Their undisturbed sessions repeat the clean arm's
per-session values to 1-2%, which corroborates the ~3.8 level without supplying
a second clean measurement. The honest phrasing is n=1 clean, corroborated by
undisturbed sessions of disturbed arms; the figure should be read as "about 3%,
sign-consistent", not as a precise value.

Corroboration worth noting: that clean old arm reads 3.798 against this
section's single-shot 3.8016, measured on the same reader about 21 hours
earlier (2026-09-20 02:56 CDT against 23:43 CDT).
The old number was right; only the label attached to it was wrong.

##### What these arms do not settle

- **The harness of the old arms was not pinned, and one script sha is blank.** The harness always runs from `wt-task1-new`, but only the code under test is
  recorded, so for an old arm neither the harness commit nor its cleanliness was recorded or checked. Reconstructed from that worktree's HEAD reflog: `task1c-1` (the
  one clean old arm) and `task1c-4` ran with harness `f6608901a3`, `task1d-0/2/3` with `4626789547`, `task1e-0/2` with `87417376f5`; dirtiness is unrecoverable. Separately,
  `task1e`'s run.out has a blank script sha (a relative `$0` after `cd`). Expected effect: on what is recorded and gated (harness changes since gen2 touch drive counters and the
  idle check, not the timed loop), not on decode tok/s; this does not invalidate the cells above, and it is why the old cell stays "n=1, harness by reconstruction". See
  `analysis/dsv41-drive/PIPELINE_BASELINE.md` section 7.4.
- **Session-0 TTFT varies from 49.8 to 60.3 s across mirrors-on arms**, with no
  explanation. A pre-registered prediction (`task1c-PREDICTIONS.txt`, sha256
  recorded before the run) that this tracked boot-phase page-cache growth was
  **falsified**: a boot-warm arm came in at 50.8 s, inside the stated
  falsification condition. Sessions 1-3 are stable at 29-30 s (on) and 53-56 s
  (off) in every clean arm, so whatever this is, it is confined to the first
  session. Do not pool session-0 TTFT across arms without saying so.
- **Two arms had a disturbed session** that the verdict could not see, because it
  checks start state only: one session's prefill ran 13 s and 30 s slow while
  bytes read, residency and start-state idleness were all normal. Those arms are
  recorded VALID-but-disturbed and are excluded from the means above. Box load
  average was 2.5-2.9 at the time against 0.7 earlier, which is a hint and not a
  cause.
- **Page-cache independence rests on O_DIRECT, not on dropped caches**, because
  no one here can drop them. It is supported rather than assumed: across six
  arms reading ~400 GiB each, expert-shard residency moved by less than 0.09
  GiB, and a direct test (`dd iflag=direct` against a cold shard on each mount,
  with a buffered control) showed O_DIRECT populating **zero** bytes of page
  cache on both xfs and ext4 while the control populated 70 MiB. Note that the
  two mirrors are on different filesystems: `/mnt/nvme0` is xfs, `/mnt/nvme4` is
  ext4 and is physically `nvme3n1`.
- **Boot-phase page-cache behaviour is not understood.** Residency of the source
  directory changes during engine startup by anywhere from -3.21 to +5.31 GiB, and
  these changes do not track device reads. An explanation in terms of eviction
  and re-reading was proposed and **withdrawn** when diskstats contradicted it.
  A second hypothesis, that the 70 GiB pinned allocation reclaims page cache
  concurrently with the buffered weight load, is pre-registered and **untested**.
- **Spans are not compared across schema 1 and 2** (see the schema note above),
  so the old-versus-new comparison here is on tok/s and bytes only.
- **An attempt to raise the old cell above n=1 failed, and the reason is
  recorded.** A six-arm interleaved series was run on 2026-09-21 to measure the
  same comparison at n=3 per cell. It returned **UNRESOLVED**: three arms ran,
  two were valid, and the third was refused because a mirror drive was reading
  1.90 MB/s at its idle probe against a 1.05 MB/s limit. Its single pair gives
  1.047, which is **not a result** -- one pair, both arms contended, and the two
  cells displaced from their references by different amounts. Full write-up and
  the pre-registration in `analysis/dsv41-drive/task1-results/`
  (`task1e-RESULT.txt`). The useful output was that **two of the verdict's gates
  are calibrated for a quiet machine**: the contention gate, which disqualifies
  every arm on a box where contention is the norm, and the cross-arm check,
  which compares against a single historical reference arm. A third gate, the
  drive-idle probe, correctly caught a transient. Also recorded: foreign
  processes do not respect their nominal CPU affinities, so no core range avoids
  them. **Resolving a ~3% effect here needs a quiet machine**, and that is now a
  prerequisite rather than a detail.
- **Nothing recorded whether the box was quiet during the arms above.** Load and
  foreign-process sampling was only added afterwards, and a later series caught
  three unrelated jobs starting mid-run on cores overlapping the arms', which
  cost those arms 17 to 31 per cent. The arms in the table were very probably
  quiet -- their tok/s repeats to 0.5-0.6 per cent within a cell, which
  contention does not usually permit -- but that is inferred from the outcome,
  not observed in the condition. Treat "matched" here as matched in workload,
  seed, capacity, policy and code, not in machine load.
- **The old-barrier figure is one clean arm.** Four further old arms ran while
  the box was contended and are excluded. Their least-disturbed sessions repeat
  the clean arm's per-session values to 1-2 per cent, which corroborates the
  ~3.8 level without supplying a second clean measurement. The honest phrasing
  is n=1 clean, corroborated by undisturbed sessions of disturbed arms.

#### The gate: service-attributed expert bytes

Per-drive bytes come from the RAM-miss service's own accounting, carried on each
`ram_miss_request` trace line as `drives: [{dev, bytes, extents}]`. This
attributes expert bytes from inside the service; aggregate `/proc/diskstats`
cannot, because it sees every read on the device whoever caused it. Device ids
are `st_dev` resolved against the mount points, not assumed.

| arm | nvme0 | nvme2 (source) | nvme4 | total | extents |
|---|---:|---:|---:|---:|---:|
| base | 0.00 | 119.77 | 0.00 | 119.77 | 9,655 |
| mirror | 59.89 | 0.00 | 59.88 | 119.77 | 19,310 |

All figures GiB, 20,800 requests per arm (13,589 touch, 7,211 demand), zero
failed.

nvme2 serves **0.00 GiB, 0.00%** of the mirror arm's expert bytes. A small
residual was allowed for, since the layout is still built from nvme2; there is
none. Byte parity is exact at 100.0%, the mirrors split 50.0/50.0, and the
extent count doubles 9,655 -> 19,310 exactly as within-row splitting predicts.
Diskstats agrees independently: nvme2 402.60 -> 7.66 GiB, mirrors 0 -> 197.48
and 198.19 GiB. The 7.66 GiB residual is layout and metadata outside the
service, which is why the service-attributed figure is the gate and diskstats
is only corroboration.

#### Where the time went

| span | base | mirror |
|---|---:|---:|
| submit -> first cqe | 7.887 ms | 2.192 ms |
| first -> last cqe | 9.664 ms (n=1,888) | 1.170 ms (n=7,211) |
| pack | 3.656 ms | 3.669 ms |

Queue wait falls 3.6x. `pack` is CPU work and is unchanged at 3.66 ms, which
acts as a control: it says the gain is I/O and not measurement drift between
arms. The `n` on first-to-last cqe rising to 7,211 is simply every request now
spanning more than one extent.

**These two spans are measured on the pre-Task-4 reader and do not carry
forward.** There, each io_uring batch had its own submit-to-first and
first-to-last span and the record summed them over batches, so packing never
fell inside first-to-last. Task 4 makes each span cover the whole read, and
first-to-last can then include the packing of early rows whenever completions
return in more than one reap - which is the overlap itself, not a regression.
The two definitions coincide only for a single-batch read whose completions all
return in one reap. So this table may be compared with other pre-Task-4 runs and
with nothing after it. How far the definitions diverge on this workload is
unknown, because the split of these 1,888 and 7,211 requests between advisory,
single-batch and multi-batch was not recorded.

The per-extent `extent_cqe` stamps did not change meaning, so a comparison
across that boundary should be built from those, or from a fresh baseline taken
on the new reader.

#### What the gain is measured against

The mirrors-off arm reads from `/mnt/nvme2`, which is **Gen3 x2** (about 1.9 GB/s;
see the drive table in section 2), while both mirrors are on Gen3 x4 drives. So
the comparison is one half-width link against two full-width ones: roughly 4x the
aggregate ceiling, not the 2x that "two drives instead of one" suggests.

That is the right control, because nvme2 is what the system actually read from
before mirroring, so 1.348x is the real gain from enabling it. But the mechanism
should not be misattributed, and the per-row bench already separates it.
`MIRROR_ROWS.md` records mirrored at **2.31x the nvme2 baseline and 1.42x nvme0
alone**, so nvme0 alone was 2.31 / 1.42 = **1.63x** the source. At the per-row
level, most of the gain came from leaving the half-width link, not from using
two drives: 1.63x from the link and 1.42x from the split.

That decomposition is per-row, not end to end; the decode arm's 1.348x has not
been split the same way and one x4 mirror has never been run end to end. The
practical reading is that a single x4 mirror would likely retain most of the
benefit if a drive ever has to be freed. It does not qualify the measured
result.

Link state confirmed from PCI sysfs on 2026-09-20 (`lspci -vv` shows no LnkSta
without root): 0000:88:00.0 (nvme2) `current_link_width` 2 against
`max_link_width` 4, the other three at 4; its root port 0000:85:02.0 likewise
negotiated x2 of 4. AER correctable counters 0. One snapshot, so a transient
downtrain is not excluded. The 1.9 GB/s figure is the Gen3 x2 spec ceiling;
nvme2's throughput was not measured here.

#### Three defects found on the way

The extent work surfaced bugs that the previous arms could not have exposed:

1. **`ensure_started` built its tables with no roots**, so the env never reached
   in-graph reads and the feature was inert. Task 4 would have measured nothing
   for a second time.
2. **A latent io_uring crash** at three or more roots: a fixed ring of 16 SQEs
   against 3 roots x 8 rows returned a null `get_sqe`. The original plan
   asserted this could not happen.
3. **A past-EOF extent clamped to nothing returned SUCCESS** while publishing
   stale bounce-buffer bytes. The `max(0, ...)` in the clamp is load-bearing.

## 20. The HTTP serving harness: mirrors, leases, and what the arms compare (2026-09-22)

`benchmarks/dsv41_baseline/run_arm.sh` drives the **served** path -- a real
`sglang.launch_server` over `/v1/chat/completions` -- where sections 18-19 drove the
offline `Engine`. It is a different workload on the same machine, so its numbers are
comparable *within* the harness and not against sections 19's cells. Four arms, each
n=1, 8 sessions from the cfq PDF corpus, 485 completion tokens, `--max-running-requests 1`,
breakable decode graphs. Raw output under
`cc-expert-prediction/dsv41-baseline/servers/<arm>/run-<stamp>/`.

| arm | leases | mirrors | mean | median | token-weighted | mean TTFT s |
|---|---|---|---:|---:|---:|---:|
| `phase1-leases` | on | off | 2.037 | 1.997 | 2.141 | - |
| `phase1-nolease` | off | off | 2.003 | 1.953 | 2.102 | 68.0 |
| `phase1-nolease-mirror` | off | **on** | 2.775 | 2.851 | 2.741 | **38.9** |

**Mirroring reproduces on the served path: 1.385x on the mean** (2.775 / 2.003),
against 1.347x from Task 1's repeated Engine arms and 1.3482x from the paired
single-shot pair. TTFT falls 1.75x (68.0 -> 38.9 s), against section 19's 1.84x.

Per-drive reads, from `/proc/diskstats` sector deltas across the whole arm:

| arm | nvme0 | nvme2 (source) | nvme4 | total |
|---|---:|---:|---:|---:|
| mirrors off | - | 1112.0 | 9.4 | 1121.4 |
| mirrors on | 600.7 | 9.3 | 610.8 | 1220.8 |

All figures GiB. The source drive falls to 9.3 GiB and the two mirrors split 49.6/50.4,
so the native reader is reaching them: contrast section 19's pre-fix `g-mirror`, which
left a 127.21 GiB residual on nvme2 because in-graph decode misses bypassed the mirrors.

**Open, and weakly evidenced: the mirrored arm read 8.9% more bytes** (1220.8 vs 1121.4),
where section 19's pre-fix pair was byte-neutral to 0.5%. One arm per cell, run half an
hour apart with no page-cache control between them, so this is an observation to check on
a repeat, not a finding.

### Leases are not visible in throughput, and this is the wrong instrument for them

`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` on vs off differs by **1.8%** (2.141 vs 2.102
token-weighted), with per-session values spanning 1.67-2.28 within a single arm. At n=1
against a spread that wide, the comparison resolves nothing except that lease mode plus
the phase-1 progress callback costs no measurable throughput. Distinguishing an effect
of that size needs roughly three arms per cell, about 2.5 h of GPU.

This is also the wrong metric to look for phase 1 in: it retires leases *during* a read
rather than only at the top of `pump()`, which is a stall/progress property. Its mechanism
is established by a mutation-killed unit test
(`test_a_lease_acknowledged_mid_read_retires_before_that_read_returns`), not by these arms.

### Do not compare these to 2.91 / 3.93 without the offset

Both arms sit at a consistent fraction of Task 1's matched Engine cells: 2.003 / 2.910 =
**0.69** (mirrors off) and 2.775 / 3.930 = **0.71** (mirrors on). The same offset in both
cells, with the mirror ratio reproducing between them, points at workload shape -- this
corpus generates ~60 tokens per session behind a 39-68 s TTFT, against section 19's
256-prompt / 128-new sessions -- rather than at anything mirror- or lease-related. The
offset is unexplained and nobody has run the two corpora against each other.

### The verdicts this harness printed before 2026-09-22 were graded on the wrong process

`report_builder.build_report` copied `sglang_env` / `sglang_env_resolved` / `sglang_file`
out of `provenance.capture()`, which samples the **harness**, into the keys
`task1_arm_verdict.check_arm` reads as facts about the process that ran the arm. The
harness holds none of the server's `SGLANG_*` vars, so every arm it graded was judged on
the wrong process: it reported the reader as `mmap` and, on the mirrored arm,
`SGLANG_MOE_EXPERT_MIRROR_DIRS` unset -- for a server whose `/proc/<pid>/environ` had it
set and whose drive counters are the table above. Fixed at `a50d7683cb`; the server's own
environ now populates those keys, and the checks that genuinely cannot be answered from
outside the process are declared unavailable and re-asked of the measured environment.
**No measurement was affected** -- the verdict runs after the timed set -- but every
verdict printed before that commit should be re-read, not trusted.


## 21. Nsight trace of the served engram arm: where the time goes (2026-09-22)

> **Historical interpretation.** The later sampled and graph-node captures in §23
> separate prefill from decode. The long visible `.tolist()` readback is in prefill;
> decode graph time is dominated by MoE expert service and pinned-row copy. The
> `_gather_host_rows_kernel` discussed below is a **MoE expert** gather, not an
> Engram table gather. Keep the timings below as trace observations, not the current
> decode optimization ranking.

Arm `engram-on-trace` at `6205f090ea`, generation `engram-host-node-uring-chunked`,
`SGLANG_MOE_PINNED_HOST_MB=51200`, uring reader, leases off. Report:
`/mnt/nvme1/dsv41-nsys/engram-on-trace-20260922-221602.nsys-rep` (42 MB), exported
alongside as `.sqlite`. Run dir
`cc-expert-prediction/dsv41-baseline/servers/engram-on-trace/run-20260922-221546`.

**This arm's verdict is void and its throughput must not be quoted.** It ran with
`DSV41_MAX_SESSIONS=2` to bound the report, so the result gate refused it
(`2 records (expected 8)`), and the capture is traced. What survives is per-unit cost,
which does not depend on how many sessions ran. The *ratio* of prefill to decode below
is an artifact of the two-session cap and is not a property of the workload.

138.1 s captured, one CUDA-calling thread. Two regimes that overlap:

| | wall | GPU kernel busy | shape |
|---|---:|---:|---|
| eager / prefill | 1.3-92.4 s | ~18% | 308k eager launches, 19.8k host syncs |
| decode | 42.3-138.2 s, 220 steps | n/a (in graph) | 203 graph launches after 95 s |

### The pinned-host gather sits on the Gen3 wall, on a second kernel

`_gather_host_rows_kernel` (the Triton byte-copy in `moe/expert_stream.py`) is **15.19 s
of the 16.62 s** of all eager GPU kernel time -- 91% -- across 1,680 launches moving
**187.3 GB** out of the pinned host buffer. Bucketed by row size, with bytes taken from
the launch geometry (`gridX` rows x `gridY` x `BLOCK`=1024 B):

| row bytes | launches | avg ms | GB/s |
|---:|---:|---:|---:|
| 5,120 | 280 | 0.02 | 11.9 |
| 9,216 | 280 | 0.04 | 11.5 |
| 10,240 | 280 | 0.04 | 11.6 |
| 20,480 | 280 | 0.09 | 11.9 |
| 4,423,680 | 280 | 18.02 | 12.3 |
| 8,847,360 | 280 | 36.04 | 12.3 |

Flat at 11.5-12.3 GB/s across a 1,700x range of transfer sizes, with no per-call fixed
cost. **This is not a new finding.** It is the wall already recorded in
[`MOE_EXPERT_TRANSFER.md`](MOE_EXPERT_TRANSFER.md), "The link is at line rate, and the
host caps it at Gen3": `nvidia-smi` reports `gpumax=5, hostmax=3, current=3` on this
RTX 5090, and Gen3 x16 is ~12.3 GB/s practical. What is new is only that a *second,
independent* kernel on a *different* code path reproduces the same constant. Record it
so the hope is not re-derived on this path either: this **eager MoE expert gather**
already reaches the host-link limit. Row packing before the copy is a separate
service stage and later improved throughput (§23).

The GPU also sits on NUMA node 0 (`local_cpulist=0-17,36-53`), which is where
`arm_env.SERVER_CORES` now pins the server. The node-0 pinning from `86b4bc19df` was
made to stop the THP direct-compaction hang; it happens to also be the correct side of
the board for the PCIe transfers.

### The costly "memcpy" is not a transfer, it is a sync

`cudaMemcpyAsync` is the single largest CUDA API cost in the trace (32.54 s over 23,832
calls). Joining each call to its GPU-side memcpy record by `correlationId` shows what it
actually is:

| phase | n | avg bytes | GPU transfer | host blocked |
|---|---:|---:|---:|---:|
| prefill | 19,761 D2H | 31 B | negligible | 16.63 s (841 us each) |
| decode | 256 D2H | 2 KB | 0.7 us each | 15.26 s (105 calls exceed 50 ms, averaging 145 ms) |

A 31-byte readback that blocks the host for 841 microseconds is a stream
synchronization wearing a transfer's name. `gather_rows` documents the mechanism in its
own docstring -- "Each chunk's hit count (`.item()`) syncs the stream on the host, so the
previous chunk's copy has run before its slots can be" reused. The cost of that
`.item()` is 16.6 s of the 91 s prefill window.

This is the one number in the trace that looks directly actionable: the sync is there to
order chunk N+1's admission against chunk N's copy, which an event wait on the stream
could do without returning to the host. Not attempted, and not costed.

### Half the trace is host-side Python that this capture cannot name

In the 91.1 s prefill window the CUDA thread spends 21.76 s inside CUDA API calls, and
OSRT shows it blocking on OS primitives for 0.81 s. The remaining **~69 s -- 76% of
prefill and 50% of the whole trace -- is that thread running host code with the GPU
idle.** 307,995 eager kernel launches in that window is ~3,350 launches/s, which is a
host-bound issue rate, not a GPU-bound one.

The arm ran `--sample=none`, so there are no callstacks and **this 69 s is unattributed**.
Re-running with `NSYS_SAMPLE=cpu` is the next step and has not been done. Do not assume
it is the engram path; the eager expert gather is only 15.19 s of GPU time inside it.

### Two measurement traps in this report

**`CUPTI_ACTIVITY_KIND_GRAPH_TRACE` here describes the warm-up, not the capture.** All
1,298 rows carry negative timestamps (-467.4 s to -0.8 s) and correlation IDs from
157,965 to 1,090,103, strictly below the captured window's minimum of 1,090,415. They are
flushed pre-capture events on a different time base. Read naively they yield a confident
"1,298 graph executions averaging 181.7 ms", which describes nothing in the traced
region and sums to 236 s inside a 138 s trace -- the impossibility is the tell.

**The graph-mode kernel-table trap held exactly as `CLAUDE.md` describes it.** No kernel
row has `graphId > 0`, so all 342,564 kernels and all 16.62 s of kernel time are eager
work; the decode graph body is absent. Nothing above ranks a kernel inside decode. The
memcpy table is unaffected, but see the previous section -- its *totals* were fine while
being badly misleading about what the cost was.

### Decode, to the extent this trace can see it

203 graph launches after 95 s against 220 `alloc_decode_kernel` launches over the whole
trace, i.e. two graph launches per decode step: the `segments=2 breaks=1` shape of the
breakable decode graph with layer 1 captured and layer 14 eager. The host thread is
blocked 98% of decode wall, 63% in `cudaGraphLaunch` and 35% in the 2 KB D2H sync above.
Because the graph body is invisible, this trace cannot say how that 63% divides between
genuine GPU work and queueing. This is the measurement that
`analysis/dsv41-drive/ENGRAM_LAYER14_GRAPH_PLAN.md` should move: with layer 14 captured,
the step should fall to one graph launch.


## 22. Layer 14 in the decode graph: the break is gone, the time is not (2026-09-22)

> **Updated diagnosis (2026-09-23):** the structural result below still holds: both
> Engram layers run inside one decode graph. The recommendation below to attack the
> visible `.item()` sync as the next *decode* fix was superseded by sampled stacks and
> graph-node attribution (§23). Those syncs belong to prefill or instrumentation;
> decode is chiefly MoE wait and expert-row copy.

`d337301dd1` extends the captured host-node gate from `layer_id == 1` to `1, 14`. Two
arms at that commit, both at `SGLANG_MOE_PINNED_HOST_MB=51200`, generation
`engram-host-node-layer14`, uring reader, leases off, 8 sessions each.

| arm | traced | mean | median | token-weighted | mean TTFT | verdict |
|---|---|---:|---:|---:|---:|---|
| `engram-on-chunked2` (layer 1 only) | no | 2.705 | - | **2.777** | - | failed (residency 5.53 GiB) |
| `layer14-graph` | no | 2.625 | 2.613 | **2.724** | 40.5 s | failed (residency 3.03 GiB) |
| `layer14-trace2` | yes | 2.608 | 2.621 | **2.720** | 39.5 s | **passed** (bar the step-latency gap) |

### The structural change works, exactly

`segments=1 breaks=0` on all three decode graph variants in the real server log, against
`segments=2 breaks=1` before. Measured against generated tokens rather than inferred, the
trace is unambiguous:

| arm | completion tokens | cudaGraphLaunch | launches per token |
|---|---:|---:|---:|
| `engram-on-trace` (layer 1 only) | 110 | 220 | **2.000** |
| `layer14-trace2` (layers 1 and 14) | 485 | 485 | **1.000** |

One graph launch per decode token. The break is removed.

### It buys nothing measurable, and section 21 says why

2.777 -> 2.724 token-weighted is a 1.9% *decrease*, and it does not resolve: per-session
values span 2.157-2.891 inside a single arm, every cell is n=1, and `layer14-graph` ran
with `nimbus_beacon_node`, `tmux` and `htop` contending on server cores. Treat the three
cells above as indistinguishable.

That null is the predicted one. Section 21 measured decode as 98% host-blocked, 63% of it
inside `cudaGraphLaunch` waiting on the GPU and 35% inside a 2 KB device-to-host sync.
Removing a break removes *host-side launch overhead*, which was not the binding term, so
merging two segments into one simply means the single launch blocks for what two used to.

The following all-process trace totals reproduce at 4x the sample. They mix prefill
and decode and cannot by themselves rank costs inside decode:

| | section 21 (110 tokens) | this arm (485 tokens) |
|---|---|---|
| gather kernel | 1,680 calls, 187.3 GB, **12.3 GB/s**, 91% of eager GPU time | 6,660 calls, 736.3 GB, **12.4 GB/s**, 91% of eager GPU time |
| D2H sync readbacks | 20,017 calls, 31-41 B, 31.9 s blocked | 78,647 calls, 41 B avg, **64.1 s blocked** |
| eager GPU busy | ~18% of wall | 65.3 s of 502.4 s = **13%** of wall |

**Do not read this as "graphing layer 14 was not worth doing."** It removes a real graph
break and is structurally correct; it is simply not where the time is. The original
next-step suggestion was to remove the `.item()` sync (64 s in this whole-process
trace). Subsequent sampled stacks placed the long readbacks in prefill, and §23's
graph-node trace made MoE service and pinned-row copy the decode targets.

### Operational: divix01's /tmp is too small for a full-length capture

The first traced attempt (`layer14-trace`) died mid-run. nsys warned that its temporary
directory had 3 MiB free against the 200 MiB it wants, the driver then saw
`RemoteDisconnected`, and the report came out at 361 KB. divix01's `/tmp` is on the root
xfs volume, which was 88% full in this historical run. Its two-session trace fitted at
the time, while the then-full eight-session capture did not; the failure killed the
server rather than erroring cleanly. The current benchmark has **two** timed sessions.
Set `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp` for traced arms rather than relying on `/tmp`;
the historical relaunch with it set produced a 164 MB report and a passing verdict.


## 23. Current serving state and next decode work (2026-09-23)

### 23.1 Checkout, launch, and cache budgets

**Updated 2026-09-25 (§25).** Production is launched by the checked-in
[`benchmarks/dsv41_baseline/launch_prod.sh`](benchmarks/dsv41_baseline/launch_prod.sh), which
serves `arm_env.base_env()` unchanged with `arm_env.ServerArgs.prod()` (port 7867, host
`0.0.0.0`). It has no overrides of its own, so production and every benchmark arm run the same
recipe. The divix01 wrapper `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live/launch.sh`
only calls it from the `dsv41-direct-prod` checkout. Port 7867 is **stopped** at the owner's
request. The following are **DSV4.1 recipe defaults**, not general SGLang defaults:

| Setting | Current value | Status |
|---|---:|---|
| EXL3 expert source | `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw` | `uring_direct`; mirrors on `/mnt/nvme0` and `/mnt/nvme4` |
| MoE pinned host tier | 100 GiB (`SGLANG_MOE_PINNED_HOST_MB=102400`, `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:61440,1:40960`) | Since 2026-09-24 (§24.6): 60 GiB bound to node 0, 40 GiB to node 1; earlier 50 GiB unplaced |
| Native Engram row cache | 5 GiB | Ordinary `mmap` slab; pinned transfer buffers are separate. A proposed 20 GiB pinned row cache is **not implemented**. `SGLANG_DSV41_ENGRAM_RAM_GIB` sizes the Python cache, not this native slab |
| GPU hot expert budget | 14 GiB (`SGLANG_MOE_HOT_GPU_MB=14336`) | DIRECT mode observed 1,128 resident slots and zero graph-gather scratch bytes |
| MoE miss path | leases=1, GPU residency update=1, DIRECT insert stage=2, decode update interval=1, two-phase=1, hit-wait 100 µs, piece streaming=1 | Two-phase and piece streaming default since 2026-09-24 (§24); prefetch and doorbell off |
| Row packing | 8 workers | Current default; eight versus four has not had a matched served-path comparison |
| Engram lookup | host-node cache with io_uring=1, device wait=1 | Since 2026-09-25 the decode lookup is a device post plus a device wait served by a polling thread; the decode graph has no host nodes (§25.2) |
| Layer fusion | `SGLANG_DSV41_ENABLE_LAYER_FUSION=1` | Since 2026-09-25: three JIT kernels replace 89 small kernels per layer, byte-identical (§25.2) |
| Copy engine | `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1` with `CUDA_MODULE_LOADING=EAGER` | Since 2026-09-25: RAM-hit rows move by DMA, not the SM kernel C1; requires EAGER (§25.3) |
| Memory fraction | 0.83 | Since 2026-09-25, for EAGER's ~1 GiB; KV 204,288 tokens. A ~30k-token prompt at 0.83 is **untested** (§25.3) |
| Decoder SWA bounded replay | `--enable-decoder-swa-bounded-replay` | Since 2026-09-25; prompt-token logprob requests are refused, not crashed (§25.5) |
| Context and prefix | 32,768 tokens; prefix caching enabled | Production and current benchmark match |
| Async CPU residency scores | 0 | Conflicts with the selected GPU residency update mode; the older async-score win was measured in a different mode |
| Expert fused plan | 1 by default | Enabled in `arm_env.base_env()` after the one-arm opt-in measurement below; compatible with DIRECT stage 2 |

The exact option ledger, including incompatible modes, is
[`analysis/dsv41-drive/DSV41_LAUNCH_OPTIONS_20260923.md`](analysis/dsv41-drive/DSV41_LAUNCH_OPTIONS_20260923.md).
The old §9.1 allocation, the proposed `/mnt/nvme1` expert move, and §17's 888-slot
scratch budget are historical configurations.

### 23.2 What decode currently costs

The [110-replay graph-node capture](analysis/dsv41-drive/ENGRAM_DECODE_GRAPH_NODE_MEASUREMENT_20260923.md)
measured **19.418 s** in the MoE RAM-miss wait kernel and **18.213 s** in the pinned
expert-row copy kernel. Together they were 95.3% of summed graph-kernel duration.
Both Engram host callbacks took **0.687 s** in total, about 1.7% of summed replay
spans. This is an attribution trace with node-level tracing overhead, not an
unprofiled throughput number; it used the earlier leases-off, pre-DIRECT recipe, so
its percentages are not a measurement of the current launch. The graph has **one
segment and zero Engram breaks**;
the hash-ID readback is a graph D2H node feeding the native host callback, not a
per-token Python `.cpu()` call. The
[sampled capture](analysis/dsv41-drive/ENGRAM_DECODE_SAMPLED_MEASUREMENT_20260923.md)
located the long visible `.tolist()` readbacks in **prefill**, not decode.

A [matched native service trace](analysis/dsv41-drive/MOE_SERVICE_TRACE_MEASUREMENT_20260923.md)
measured 19.550 s of GPU wait against 19.482 s of CPU service over 2,598 demand
layers. It split service into an 11.893 s read-completion window and a 7.560 s
exposed row-packing tail. A separate [four-worker measurement](analysis/dsv41-drive/MOE_PACK4_SERVING_MEASUREMENT_20260923.md)
cut the exposed tail from 7.453 s to 2.574 s and improved the median paired
decode rate by **16.6% across eight sessions**. That comparison used an older
leases-off, pre-DIRECT recipe; its original formal verdict failed a page-cache gate
later corrected in the harness. It supports row packing as a useful lever, but does
not measure the current eight-worker default against four.

The [DIRECT insertion served comparison](analysis/dsv41-drive/DSV41_DIRECT_INSERT_MEASUREMENT_20260923.md)
found a **1.1631 median paired rate ratio, eight wins in eight sessions** against
its then-current control and raised the resident set from 888 to 1,128 slots by
removing scratch. It was measured before the current eight-worker and production
context/prefix recipe. The [live 100-token trace](analysis/dsv41-drive/DECODE_LIVE_NSYS_20260923.md)
under DIRECT still measured 18.155 s MoE wait and 9.826 s pinned-row copy. Its
native records had 2,314 NVMe-read demands (54.810 GB), an 11.190 s read window,
and 6.866 s exposed packing tail, with inline packing at that time. These traces
show why the next decode work is MoE demand count, read/pack service, and pinned
row-copy throughput; they do not establish the current unprofiled rate.

### 23.3 Latest benchmark and its limits

Commit `e36fa2530c` changed the HTTP benchmark default from eight to **two timed
sessions** plus warm-up. The result gate, expected IDs, manifest, and paired-run
checks use that shape; pairing a two-session arm with an older eight-session arm is
rejected. Two sessions make a quick diagnostic, but a clean sweep in a paired sign
test only reaches p=0.25. Historical eight-session results in §§20–22 retain their
original meaning.

On divix01, a **single fused-plan-on arm** ran at that commit with an explicit
`SGLANG_MOE_EXPERT_FUSED_PLAN=1` override and every other recipe value unchanged.
The flag was then promoted into the DSV4.1 defaults:

| Timed session | Generated tokens | Decode rate | TTFT |
|---|---:|---:|---:|
| CDW | 7 | 2.862 tok/s | 37.67 s |
| ETR | 103 | 4.297 tok/s | 37.81 s |

The two-rate median is **3.579 tok/s**. The run passed preflight, live environment
verification (35 variables), warm-up stability, and the two-record/no-error result
gate. The verdict reported **no unacknowledged problems** and
`valid_except_acknowledged_gaps=True`. Its strict `valid=False` reflects missing
engine-side step latency and unavailable internal server provenance: the imported
`sglang` path and resolved `uring_direct`/graph-gather settings cannot be observed
inside the HTTP server by this harness. The actual server environment was checked
through `/proc`, and the gaps are labeled explicitly. CPU contention from `tmux` and
`htop` was noted. This is one arm with no same-recipe fused-plan-off control, so
**no speedup is established**.
The run directory is
`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/fused-plan-on-2timed-20260923/run-20260923-224033`.
The harness verified the flag in the server environment, but this run did not
independently trace whether every layer selected the fused route rather than its
supported generic fallback.

The near-term decode measurement is a same-recipe comparison of expert demand,
io_uring completion, row-packing tail, doorbell signal, and GPU row-copy duration,
followed by a pinned-copy bandwidth experiment. Keep the current 50 GiB MoE tier,
DIRECT mode, and eight workers fixed while isolating those costs. A larger pinned
Engram cache requires allocator and NUMA-budget work and should be evaluated
separately from decode MoE service.

## 24. RAM-miss piece streaming (2026-09-24)

### 24.1 What it is and where it lives

Plan: [`docs/superpowers/plans/2026-09-24-dsv41-piece-streaming.md`](docs/superpowers/plans/2026-09-24-dsv41-piece-streaming.md).
Protocol: [`analysis/dsv41-drive/LEASE_PROTOCOL.md`](analysis/dsv41-drive/LEASE_PROTOCOL.md),
area P and the E1, §2, §4.3 and §4.4 amendments.

- **Flag:** `SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM`, default off.
  - Startup refuses it unless two-phase is on, leases are on, and there are one or more
    pack workers.
- **What changes:** each RAM-miss row (13,320,192 B) is read as two mirror halves of four
  sub-reads each: 8 pieces.
  - Each piece has its own pack job.
  - The owner publishes each packed piece to the lease block's per-lane mask words
    (area P) with a generation-checked CAS.
  - A new stream kernel **S** replaces the two-phase W2 + C2 pair. It copies each piece
    to the GPU as soon as the piece is published.
  - A slot whose stream was refused goes to quarantine.
- **Flag off:** the two-phase chain is unchanged (10 nodes, 9 edges). Two-phase GPU tests:
  67/67.
- **Branch history:** built on `cc/dsv41-piece-stream` (tasks 0b and 1–6, each
  task-reviewed, plus a final whole-branch review). It was fast-forwarded into
  `cc/dsv41-direct-two-phase` and then into `master` on `shared`.
  The merged tree is the tree benchmarked below.
- **Tests at merge:** kernels plus MoE RAM-miss 1,474 passed and 0 failed; piece-stream
  CPU tests 82/82; piece-stream GPU tests 22/22.
- **Correction:** two-phase **does** run under EXL3 DIRECT stage 2, since `9c64bac755`.
  The "DIRECT insertion rejects two-phase" line in the launch-options ledger was stale.
  Both arms below ran the DIRECT recipe.

### 24.2 Benchmark: 8 of 8 paired sessions

Setup:
- Graph mode, one cold server per invocation, all at `a082278187`.
- Order A B B A A B B A, two sessions per invocation, covering all eight corpus sessions.
  This uses the new `DSV41_SESSION_INDICES` override; `concat_arms.py` joins the four
  runs of each arm for `paired.py`.
- **A** = `arm_env` base + `TWO_PHASE=1` + `HIT_WAIT_US=100`. **B** = A + the piece flag.
- The environment diff between the arms is exactly the flag.

| Measure | Result |
|---|---|
| Wins, one-sided sign test | **8/8, p = 0.0039** (the pre-registered bar was 7 of 8) |
| Paired median gain | 55.7 ms/token (mean 91.6); rate ratio median 1.328 |
| Own medians | A 4.278 tok/s, B 5.588 tok/s |
| Robust gain estimate | **about 40 ms/token** |
| Output | byte-identical, 12/12 greedy responses across two smoke pairs |
| Error counters | 0 refusals, 0 quarantined slots, 0 fatals |

- **Why the robust estimate is lower than the median:**
  - The median is inflated by short sessions: one decodes only 6 intervals.
  - It is also inflated by one flag-off outlier, a session at 516.5 ms/token.
  - The 40 ms/token figure comes from two same-prompt, stage-traced smokes (about 40 each)
    and from the warm-up request, which was the same in all 8 servers (37.6).
- **Why the gain beats Stage 0b's 28.9 ms/token ceiling:**
  - The ceiling modelled only the GPU copy overlapping the read.
  - Per-piece packing also removes the host pack tail, the time from the last read
    completion to `done`. Per step it drops from **24–25 ms to 4.3 ms**.
- **Node-mode GPU chain:** 150.6 → 118.7 ms/token, a saving of 31.9.
  - S starts 3.1 µs (p50) after C1.
  - From the last piece publish to the end of S: 186 µs at p50.
- **Scope limit:** this is not a comparison with the saved production recipe. Production
  runs with two-phase **off**, and both arms had it on. No same-session result against the
  production recipe exists yet.

**Read-wall regression (open):**
- At credit 32, reads are four times smaller, so the pure disk window per demand grows.
- **6-row demands:** the median rises 3.5%, but the mean rises **15.7%**. The plan's 10%
  kill line did not name a statistic; it was judged on the median.
- **1-row demands:** the median rises 12.5%.
- Read plus pack still finishes earlier at the median: −2.9% at 6 rows, −16.9% at 1 row.

**Other measured costs:**
- **W1 always runs out its budget:** on every read layer W1 spends its full 100 µs budget
  (104 µs p50), about 1.9 ms/token.

Evidence on divix01:
- Scripts and logs: `/data/models/slang/nvfp4-work/direct-two-phase-tests/piece-stream/t6/`.
  The paired verdict is in `abba/paired.json`; smoke checks are in `smoke_both.log` and
  `smoke2_check.out`.
- Node trace: `/mnt/nvme1/dsv41-nsys/ps-on-t100-node-20260924-105604.nsys-rep`.

### 24.3 Nsight graph-mode trace of the flag-on arm

Report: `/mnt/nvme1/dsv41-nsys/ps-B-nsys-graph-20260924-105920.nsys-rep` (46 MB), with a
`.sqlite` export alongside. Tracing cost nothing in steady state: 158.4 ms/token traced
against 158.8 untraced.

Decode steps below are the 106 launch intervals under 1 s. The capture has 110 launches:
the 1.4 s first step after capture start and the 44 s interval holding session 2's
prefill are excluded.

- **Step time:** launch to launch, p50 **155.8 ms** (p10 108.6, p90 225.6).
- **Inside the step:**
  - The host spends p50 **151.6 ms inside `cudaGraphLaunch`**.
  - It spends **4.06 ms** in the tail: sampling and the scheduler. Of that, 3.46 ms is
    Python outside any CUDA call.
- **The GPU does not wait for the host tail:**
  - The host runs about one step ahead: `cudaGraphLaunch` returns while the GPU still has
    about 148 ms of that step's graph to run.
  - The tail's sampling kernels start on the GPU **148 ms** (p50) after the host launches
    them, in all 106 steps.
  - Each tail's 20 eager kernels (0.08 ms of GPU time) run during the next launch's host
    span. The 4 ms host tail is therefore hidden.
- **No per-step blocking sync in decode:**
  - An earlier summary of this trace reported "65.8 ms per step of blocking 256-byte
    readbacks". That figure spread the whole window's `cudaMemcpyAsync` time over the
    decode launches.
  - All 134 blocking readbacks (256 B and smaller, stream 13, about 59 ms each) fall in
    the 44 s interval between the sessions. That interval also holds 164,462 eager
    kernels: it is session 2's 32k-token **prefill**, not decode.
  - 106 of the 106 decode steps have none.
- **What this means:** decode is GPU-bound. The node trace (§24.2) puts the RAM-miss chain
  at W1 1.9 + C1 23.9 + S 60.7 ms per token over read layers, and S is mostly waiting on
  the NVMe read.
- **Not measurable from this report:** GPU idle inside a step.
  - The graph body is hidden in graph mode.
  - `GRAPH_TRACE` holds only pre-capture warm-up rows, per the project notes.
- **Scripts:** `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/sync-attr/`:
  - `step_timeline.py`: where the blocking copies fall.
  - `tail.py`: step and tail split.
  - `idle.py`: tail kernel queue delay.

  Each has a matching `.out` file.

### 24.4 The read wall: not the credit, not the mirror split (2026-09-24)

**Credit cannot bind in most decode demands.** Credit is `kQueueDepth` (16) × parts
(2) = 32 SQEs. With pieces, a demand for m rows issues 8m SQEs, so the credit binds only at
m ≥ 5. That is 16 of the node trace's 1,155 read layers.

**The read-window growth is on one drive.** From the task 6 smokes (per-sub-read
completion stamps in the stage trace):
- The faster half is unchanged: 2.07 → 2.14 ms p50 at m=1, with the mean flat. So one
  6.66 MB read and four 1.66 MB reads cost the same on one drive.
- Half 1, from the second mirror root `/mnt/nvme4`, finishes last in about 99% of demands.
- The gap between the halves grew from p50 0.20 → 0.53 ms at m=1.
- It is not a reap delay: in decode, the four sub-reads of a half carry four distinct reap
  stamps in 99% of demands.

**Why nvme4 is slower.** It is an SPCC drive with a 256 KB maximum transfer, against the
Samsung's 512 KB on nvme0. Live `iostat` showed:
- equal 910 MB/s on each drive;
- 3,896 reads/s at 239 KB on nvme4, against 2,059 reads/s at 453 KB on nvme0;
- similar average wait on both: 8.0 vs 7.8 ms.

**Swap is not the cause.** The swapfile on nvme4 held 9.5 GB but was idle during decode:
swap-in 0–16 KB/s, swap-out 0, and 0.05 MB/s of writes to the drive.

**Mirror-weight sweep.** Four flag-on stage-traced smokes, in order, at `a082278187`.
Scripts are in `divix01:.../direct-two-phase-tests/mirror-weights/`. Every run had
byte-identical responses and 0 refusals, quarantines, read errors or voided leases.

| nvme0:nvme4 weight | m=1 gap (nvme4 − nvme0) p50 | m=4+ gap p50 | all-m pack_end mean | Σ decode pack_end per step | step wall p50 |
|---|---:|---:|---:|---:|---:|
| 1:1 (first) | +0.42 ms | +0.67 | 4.10 ms | 96.3 ms | 153.3 ms |
| 55:45 | +0.16 | −0.94 | 4.24 | 99.6 | 156.2 |
| 60:40 | −0.35 | −2.48 | 4.22 | 99.1 | 156.2 |
| 1:1 (repeat) | +0.58 | +0.73 | 4.23 | 99.2 | 154.8 |

- **No weight is better than 1:1.** The spread between weights is inside the spread
  between the two 1:1 runs.
- **Why a fixed weight can't help:** nvme4's disadvantage behaves like a fixed extra
  **~0.4–0.7 ms per demand**, not a lower bandwidth. At 55:45, m=1 balances but nvme0
  becomes the late drive from m=2 up.
- **Upper bound for a split that varies with m:** half the gap, about 0.2–0.35 ms per
  demand. It has not been shown to move pack_end.
- **The tail was an environment effect.** Today's nvme4 tail is much smaller than in the
  task 6 smokes: the m=1 slow-half mean is 2.58–2.72 ms, against 3.32 then. That day's
  +15.7% mean read-wall regression was therefore partly the environment of that run.

### 24.5 Tier simulator at today's recipe: RAM tier size (2026-09-24)

**Input.** Graph-mode decode traces carry no routed expert IDs (`graph_step` has
`"experts": []`, and RAM-miss request records carry no IDs either), so `tier_sim.py`
cannot replay them. The simulation instead replays the eager 8-session corpus trace
`analysis/dsv41-phase3a/trace-cold.jsonl` (41,320 calls, 1,024 decode tokens). Routing
depends only on the model and the prompts, so this is valid input.

**Settings.** Today's VRAM is 1,128 slots and today's RAM tier is 4,031 rows (50 GiB, from
the server log). Residency updates every decode forward; prefill updates every 256 tokens.

**Fidelity.** The native RAM tier is a per-layer LRU that skips VRAM-resident experts,
which matches the simulator's inclusive per-layer LRU. The simulator does **not** model
DIRECT insert-on-miss into VRAM.

| Source, at 4,031 RAM rows | G | f | NVMe rows/token |
|---|---:|---:|---:|
| Simulator, corpus trace | 93.6 | 0.344 | 32.2 |
| Measured today, 50:50 smoke (graph decode) | 91.2 | 0.405 | 36.9 |
| Measured, task 6 stage trace | — | — | 36.8 |

G agrees closely. The simulator puts about 13% fewer rows on NVMe than measured, so read
the curve as somewhat optimistic.

| RAM rows (≈ GiB) | f | NVMe rows/token | vs today |
|---:|---:|---:|---:|
| 3,000 (37) | 0.332 | 43.7 | +36% |
| **4,031 (50, today)** | 0.344 | **32.2** | — |
| 5,000 (62) | 0.275 | 25.7 | −20% |
| 6,000 (74) | 0.215 | 20.2 | −37% |
| 7,000 (87) | 0.160 | 14.9 | −54% |
| 8,000 (99) | 0.114 | 10.7 | −67% |
| 10,000 (124) | 0.053 | 5.0 | −85% |

The 3,000-row row is not comparable to the rest: at that size the inclusive clamp limits
all 40 layers, so the VRAM set also changes and G rises to 131.7.

- **The curve does not flatten** up to 8,000 rows. Each extra 1,000 rows (13.3 GB)
  removes about 5.5–6.5 NVMe rows per token.
- **Rough value** [estimate]: 1.2–1.6 ms saved per row converted from an NVMe read to a
  pinned-memory hit. That is about **8–10 ms/token per extra 13 GB**.
  - Growing to about 65 GiB on node 0 would save about 10–13 ms/token.
  - About 100 GiB across both nodes would save about 25–35 ms/token.
  - Not measured. The corpus prompts are shorter than the benchmark's 32k-token
    prefills, which likely flush the tier more.
- **Prefill admission:** letting prefill admit rows into RAM helps at every size from
  5,000 rows up (the ✓ rows beat the ✗ rows).
- **Output:** `divix01:.../direct-two-phase-tests/tier-sim/cold.json` and `cold.tab`.

### 24.6 NUMA-placed pinned tier, and the link becomes the wall (2026-09-24)

**Code.** Branch `cc/dsv41-pinned-numa` at `e0ce02579f`, not yet merged.
- `SGLANG_MOE_PINNED_HOST_NUMA_MB="0:61440,1:40960"` binds each layer's slab rows to NUMA
  nodes in proportion, with `mbind` applied before first touch.
- Startup refuses a node that cannot hold its share: free memory plus page cache, less
  4 GiB of headroom.
- Readers are unchanged, because each slab stays one contiguous range.
- Placed memory is still 100% on 2 MB transparent huge pages. On one thread, memcpy into
  it runs at 7.2 GB/s, against 6.1–6.3 GB/s on 4 KB pages.

**Smokes** (flag on, stage-traced, 584 decode tokens each). All four had byte-identical
responses and 0 refusals or quarantined slots.

| Tier | NVMe rows/token | f | Σ read+pack per step | Step p50 / mean |
|---|---:|---:|---:|---:|
| 50 GiB, unplaced | 36.9 | 0.405 | 97.3 ms | 153.5 / 160.6 ms |
| 50 GiB on node 0 | 36.7 | 0.404 | 98.9 ms | 155.0 / 161.5 ms |
| 60 GiB on node 0 | 31.9 | 0.352 | 85.5 ms | 147.7 / 156.1 ms |
| 90 GiB (60 + 30 on node 1) | 20.8 | 0.230 | 57.0 ms | 134.8 / 139.3 ms |

- Binding the 50 GiB tier to node 0 made no difference; the two 50 GiB rows are within
  noise.
- At 90 GiB, placing the tier took 57 s against about 20 s for the node-0 tiers, because
  node 1 reclaims page cache first.

**Node-mode trace at 100 GiB** (60 + 40, 8,063 rows).
- Report: `/mnt/nvme1/dsv41-nsys/numa100-node-20260924-144026.nsys-rep`.
- Same prompt and token count as §24.2's node trace.

| GPU time per token | 50 GiB | 100 GiB |
|---|---:|---:|
| Pinned-row copy C1 (`copy_expert_row_segments`) | 51.2 ms | **75.0 ms** |
| Stream kernel S (NVMe wait plus copy) | 64.8 ms | 16.4 ms |
| Read layers per token | 18.0 | 4.2 |
| RAM-miss chain W1..F | 118.9 ms | 92.7 ms |
| All graph kernels | 136.0 ms | 110.1 ms |
| Expert GEMMs, attention, norms, other | about 17 ms | about 17 ms |

**The host-to-GPU link is now the wall.**
- **C1 runs at line rate.** It moves 49.1 → 74.1 rows per token at **12.8 → 13.2 GB/s**,
  which is Gen3 x16 line rate. Rows on node 1 do not slow it.
- **The floor is fixed by the rows per token.** Each token needs G ≈ 79 VRAM-miss rows
  over the link: 1.06 GB, about **84.5 ms at 12.5 GB/s**, whatever the RAM tier size.
- **What can still move:**
  - fewer VRAM misses per token (G);
  - the link's idle time: about 85 ms busy in a step of about 135 ms, so copies could
    overlap compute if the rows were known a layer ahead.

### 24.7 Next decode work, in order

1. **Fewer VRAM misses per token (G ≈ 79).** Each row avoided saves about 1.07 ms of link
   time.
   - The hot cache is 14 GiB, 1,128 slots of 15,360. Look for VRAM to return to it: the
     2.8 GiB of dense dequantized modules (§4) and the 1.23 GiB embedding.
   - Retune the residency policy for the DIRECT recipe. This needs routing traces with
     expert IDs, which today only eager decode records.
2. **Closed 2026-09-25 (§25.4): no prefetch placement pays.** Originally: overlap the link
   with compute, fetching the next layer's likely experts during the current layer. The link idles about 40% of the step, so it has spare capacity, but a
   wrong guess costs link time. Prefetch is currently refused under DIRECT.
3. **Done 2026-09-24 (§24.8):** W1 now exits once every lane is claimed or LOADING. On
   read layers its p90 fell from 101 to 21 µs, saving 0.29 ms/token at 100 GiB.
4. **Done 2026-09-24, on the owner's instruction:** the NUMA placement is merged, and the
   recipe runs at 100 GiB with two-phase and piece streaming on, matching §24.6's trace.
   No same-session comparison against the old 50 GiB recipe with two-phase off was run.
5. **Dropped:**
   - Removing CPU–GPU syncs from decode (§24.3).
   - Retuning credit, and static mirror weights (§24.4).
   - Moving the swapfile off nvme4. It is idle during decode, so this is housekeeping at
     most.

### 24.8 RAM-miss synchronization pass and host copy bandwidth (2026-09-24)

**Code.**
- Branch `cc/dsv41-pinned-numa`, commits `195dba659b` through `92e588b20e`, on top of
  `9183637f51`. Not merged.
- The starting point was an audit of every wait and fence in the RAM-miss path
  (`RAM_MISS_SYNC_HANDOFF.md` in the main checkout).
- Verification:
  - CPU: `test/registered/unit/kernels -k "ram_miss or lease or piece"` under
    `taskset -c 0-31`. 980 passed, against 979 at the base. The one skip, the sibling test,
    passes under `taskset -c 0-63`.
  - GPU: the six manual RAM-miss files. 74 passed, against 72 at the base; the difference
    is T12.
  - Every 100 GiB smoke gave byte-identical responses.

| Change | Commit | Measured (100 GiB tier) |
|---|---|---|
| **W1 early exit:** under piece streaming a LOADING lane never turns READY, so W1 now stops once every lane is claimed or LOADING | `195dba659b` | Node trace: W1 p90 101 → 21 µs; 0.84 → 0.55 ms/token |
| **The RowResult re-read is now a real seqlock** (LEASE_PROTOCOL.md 11.4). The host clears the ready word before it rewrites a payload, and the device fences between the payload loads and the re-read. Before, neither half was in place | `028210c696` | A correctness fix, with no visible cost |
| **The post kernel drops four `membar.sys`** that sat right before a release store | `3ca3327c01` | post p50 20.7 → 20.0 µs. A fence straight before a release store is nearly free, so the post's 20 µs lies elsewhere (unexplained) |
| **The waits drop the `membar.sys` after an acquire of `kDemandDone`**, S included | `c96a51d91d` | S 16.4 → 15.7 ms/token, W1's effect included |
| **Packing workers are pinned one per physical core, and the service thread is kept off their CPUs** | `8f64630b03` … `92e588b20e` | Pack tail (last piece landing → last chunk end) p50 212 / 236 µs (two baseline runs) → 184 µs |

**Why a piece takes ~200 µs to pack: the socket's DRAM, not the pool.**

`pack_pool_bench` stamps every chunk. `memscale`, a scratch probe on divix01, measures
cold copies with the source on node 0, where the bounce slots live:

| Copy, 208 KB chunks | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| Node-0 cores, node-0 destination | 4.9 GB/s | 13.4 GB/s | 13.7 GB/s |
| Node-0 cores, node-1 destination | — | 7.8 GB/s | 7.3 GB/s |
| Node-1 cores, node-1 destination | — | 15.7 GB/s | 24.6 GB/s |

- **Node 0 saturates.** Its read ceiling is about 30 GB/s at 8 and at 16 threads. Each
  socket holds 4 RDIMMs (32 + 16 + 32 + 16 GB) on a 6-channel CPU.
- **Four workers already reach the copy ceiling.** A 1.66 MB piece at 13.7 GB/s is
  ≥ 120 µs, which is the bench's pack tail.
- **In serving the same DRAM is busier still.** It also carries C1's 13 GB/s of zero-copy
  reads and the NVMe DMA, and node-1 rows copy across the link.
- **The per-worker rate falls, the total doesn't.** Eight workers at 2–2.5 GB/s each is
  the plateau split eight ways.

**Spinning workers were built, measured and removed.**
- **In the bench, spinning cut only the first piece of each read:** 163 → 120 µs, from
  removing the ~100 µs C6 wake after the read gap. Later pieces were copy-bound either way.
  With a node-1 destination, spinning was worse (157 → 220 µs).
- **In serving, spinning plus the narrowed service thread stalled the SPCC mirror.** In
  303 of 1,982 multi-row demands, its sub-reads completed 10–25 ms late while the Samsung
  mirror's were on time.
- **Each change alone was clean:** 3 and 1 stalls respectively.
- **Likely mechanism:** the busy-polling service thread was pushed onto 16-17 and 36-53, the
  same CPUs as the SPCC's completion interrupts (36-71). The spinners on 0-7 crowded them.
- The shipped arm, no spin with the narrowed service thread, had 0 stalls in 1,999 at
  `92e588b20e`.

**Next, if packing is worth more:**
- **Pack node-1 rows on node-1 cores.** That copy is 3× faster. It needs pack workers
  outside the server's node-0 CPU set.
- **Read straight into slab rows.** This would remove the copy, and its DRAM traffic,
  altogether. Rows start mid-page and the segment layout differs from the file's, so it is
  a layout change.
- Either would be worth about 0.1–0.3 ms/token at 4.2 read layers per token. C1's 75 ms
  on the link (§24.6) is still the wall. (That estimate undercounted: §24.9 measured the
  second option at about 4 ms/token.)

### 24.9 Row images: reads land straight in the pinned slabs (2026-09-24)

**What changed.** With `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1`, the RAM-miss reader
reads each sub-read with one O_DIRECT `readv` whose iovecs are the pinned slab rows it
fills. There is no bounce buffer and no pack workers, and each piece is published as soon
as its read is checked. Plan: `docs/superpowers/plans/2026-09-24-dsv41-row-images.md`.

- **Why the files had to change.** Checkpoint rows start at odd file offsets, and the
  4-byte `mul1`s shift each later tensor, so no tensor boundary is 512-aligned. O_DIRECT
  needs 512-aligned file offsets and segment lengths; memory alignment is 4 bytes.
- **Row images.** A row image stores each row as its six slab rows back to back
  (`exl3_row_image.py`). Every slab row is a multiple of 512 bytes, so a `readv` can
  scatter any 512-aligned range of an image into the slabs.
- **Build.** `scripts/dsv41/build_row_images.py` writes them under
  `<mirror root>/exl3_row_images/`, 204.5 GB per root, in 7.5 min for both drives. It
  checks each row against its digest and writes the manifest last. The reader refuses a
  root whose manifest doesn't match the live checkpoint.
- **What stays the same.** The slabs, copy tables, device kernels and lease protocol are
  unchanged. The mode requires mirror dirs, `uring_direct` and leases.
- **Why leases are required.** Leases keep a GPU copy's slot from being chosen as a
  victim. A direct read overwrites its slot from the moment it is submitted, not after the
  read as packing did.
- **Probe:** `analysis/dsv41-drive/direct-read-probe/`. XFS and ext4 both report
  `dio_mem_align 4` / `dio_offset_align 512`. `readv` into mbind'd, registered slab rows
  matched the bounce path's throughput within 1%.

**100 GiB smoke at `e543f74c80`.** Script: `analysis/dsv41-drive/row-images/compare_arms.py`,
with `--include-warmup`. Runs: `divix01:.../direct-two-phase-tests/row-images/`.

| | bounce: fresh | bounce: base2 | row images 1 | row images 2 |
|---|---|---|---|---|
| pack tail p50 / mean (µs) | 179.3 / 208.4 | 185.8 / 220.3 | 0.3 / 0.3 | 0.3 / 0.3 |
| host tail p50 (µs) | 182.1 | 189.3 | 2.0 | 1.8 |
| submit→done p50, 1 / 2 / 3+ rows (µs) | 2905 / 5205 / 7305 | 2944 / – / – | 2605 / 4731 / 6714 | 2610 / 4721 / 6744 |
| stalls (multi-row demands > 10 ms) | 0 | 4 | 0 | 0 |
| ms/token, trace (checked against wall) | 136.1 (138.8) | 137.0 (139.8) | **132.4 (135.2)** | **132.1 (134.9)** |

- **Responses** are byte-identical in all five runs (6 of 6 each).
- **Gain:** about 4 ms/token. Every request was faster, by 1.6–7.0 ms/token.
- **Where the gain comes from.** Removing the pack tail explains about 13.3 demands per
  step × ~0.2 ms. The rest is reads finishing sooner: single-row submit→done fell 300 µs.
- **The first bounce run on the merged head (not shown).** One request hit the known SPCC
  late-sub-read pattern: 254 stalls, and part-1 sub-reads 10–35 ms late. It also had a
  ~38 µs higher tail. A rerun (base2) was clean, so it was the drive, not the merge.

**Traces of the default recipe at `2a3523e76e`** (row images on). Reports:
`/mnt/nvme1/dsv41-nsys/default-{graph-20260924-193419,node-20260924-193728}`. Script:
`divix01:.../row-images/trace_default.sh`. Same prompt as §24.2/§24.6: 2 warm-ups, then one
traced 96-token request.

- **Graph mode, 96 launches.**
  - Step p50 116.1 ms and mean 115.6 (p10 87.7, p90 147.5).
  - The host is in `cudaGraphLaunch` for 104 ms p50. The host tail is 12.3 ms, all hidden:
    the tail's kernels start 111 ms after their launch, so the host is a step ahead.
  - There are 0 blocking copies of 5 ms or more in decode.
- **Node mode:** 112.1 ms of GPU time per token, so the GPU is busy almost the whole step.

  | GPU time per token | ms |
  |---|---:|
  | C1 `copy_expert_row_segments` | 72.3 |
  | S `lease_stream_kernel` | 21.3 |
  | Expert GEMMs (`exl3_moe`, `exl3_gemv`) | 8.8 |
  | Everything else | 9.7 |

- **C1 still runs at Gen3 x16 line rate.** Each token has 79.9 VRAM misses, of which 9.8
  are NVMe reads. C1's 70.1 rows are 0.93 GB at **12.9 GB/s**.
- **The whole step is the link.** All of the ~80 VRAM misses cross it: 1.06 GB, about
  82 ms/token at 13 GB/s. That leaves about 34 ms/token when the link is idle: compute
  (~17), S's wait on the disk (~11), and the rest. Those periods run one after another, not
  overlapped.
- **What moves decode now:**
  - fewer VRAM misses: each row is about 1.02 ms;
  - overlapping the link with compute (§24.7 item 2);
  - a faster link: the RTX 5090 is Gen5 but sits in a Gen3 slot.

**Suites at `e543f74c80`:**
- CPU `test/registered/unit/kernels`: 1665 passed, 1 skipped (the sibling test, under
  `taskset -c 0-31`).
- GPU, six manual files plus the row-image file: 99 passed.

## 25. Copy engine, fusion, and the closed overlap studies (2026-09-25)

Everything below is merged into `master` (then named `codex/nvfp4-expert-stream-main`). Plans and raw numbers live in
`docs/superpowers/plans/2026-09-25-dsv41-*.md`; runs live under
`divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/` and `/mnt/nvme1/`.

### 25.1 Where decode stands

**Full arms** (`run_arm.sh`, A then B once each, same commit; `2026-09-25-dsv41-final-arms.md`):

| | A: recipe, copy engine off | B: recipe + copy engine |
|---|---:|---:|
| ms/token, pooled | 119.3 | **112.4** |
| ms/token, 103-token session | 117.8 | 110.8 |
| client step p50 / p90 (ms) | 117.6 / 153.9 | 110.7 / 148.0 |
| TTFT (s) | 21.4 / 17.7 | 20.9 / 17.7 |
| stalls | 0 | 0 |

- Outputs are byte-identical between A and B.
- B won both sessions, but two sessions only reach p = 0.25, so the result is directional.
- Both arms ran with questdb's `java` at ~90% on the server cores; the verdict flagged contention,
  mostly in A.
- These arms ran under LAZY module loading at memory fraction 0.80, before §25.3's EAGER requirement.
  EAGER's cost in ms/token is **not yet measured**.

**Node-mode trace of B, per step** (attribution only; not a ms/token source):

| Group | Kernels before today | Kernels now | ms before | ms now (p50) |
|---|---:|---:|---:|---:|
| attention | 1,485 | 1,489 | 7.50 | 7.18 |
| shared expert | 440 | 440 | 1.67 | 1.62 |
| routing | 1,254 | 254 | 1.39 | 0.45 |
| chain (waits, copies, S, F) | 280 | 320 | 95.15 | 96.9 |
| bookkeeping | 2,560 | 160 | 2.00 | 0.35 |
| MoE | 80 | 80 | 4.02 | 4.11 |
| **total kernels** | **6,112** | **2,756** | | |

- C1, the SM copy kernel, fell from 72.26 to 0.03 ms. The copy engine moves 458 copies (1.02 GB)
  per step at 12.57 GB/s, and 99.98% of that time sits under the chain's own wait kernels.
- **Where ~111 ms/token goes:** ~78–81 ms is the link moving ~76 RAM-hit rows of 13.3 MB;
  ~16–20 ms is the rest of S (streamed rows and NVMe waits); ~13.7 ms is all compute; ~2 ms is
  small chain kernels.
- **What that implies:** there is at most 13.7 ms of compute to hide copies behind, and the link is
  ~70% busy with demand rows. Further gains need fewer bytes per token or a faster link, not more
  overlap (§25.4).

### 25.2 Layer fusion and the Engram device wait

Both are on in `arm_env`, and both are byte-identical to the unfused recipe.

- **Layer fusion** (`SGLANG_DSV41_ENABLE_LAYER_FUSION`): three JIT kernels
  (`dsv41_layer_fusion.cuh`) replace 89 small torch kernels per layer (152 to 66): the gather
  destinations, `commit_gather`, and the fused MoE's route tables. **−3.2 ms/token.**
- **Engram device wait** (`SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT`): the layer-1 and layer-14
  lookups are a device post plus a device wait. A polling thread serves them and makes no CUDA calls.
  The post goes out early, and the wait sits at the layer. **−2.6 ms/token.**
  - The decode graph now has **zero host nodes**, which is asserted at capture. That was the
    precondition for the copy engine: with host nodes, `cudaGraphLaunch` held the driver lock that
    the copy thread needs, and the graph spun on a copy that could never be issued.
  - The 104 ms `cudaGraphLaunch` block seen in graph-mode nsys is an nsys artefact, not host-node
    cost.
- **Combined:** 125.5 ms/token against 132.1–132.4 for the row-image recipe of §24.9. Output was
  6/6 byte-identical and there were 0 stalls.

### 25.3 The copy engine

`SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1` is on in `arm_env`. Protocol: `LEASE_PROTOCOL.md` §7.6.

- **What it does:**
  - The post kernel publishes each lane's destination slot.
  - The RAM-miss service issues the six segment copies of each RAM-hit row with `cuMemcpyAsync` on
    its own thread and top-priority stream, then writes a completion word.
  - The graph's W1 and C1 become one wait kernel on that word (CW).
  - The service holds each lease until an event after its copies completes.
- **Why it's faster:** at the bench, the DMA engine gets 13.5–13.7 GB/s against the SM copy's
  12.1–12.3, and it doesn't slow concurrent compute. In practice the gain came from overlapping
  the copy with S.
- **Deadlock 1, startup:** a CUDA module loaded on the scheduler thread while an armed step spins.
  - The load waits for the device, the device waits in CW, and the copy thread waits for the load.
  - **Fixed:** the engine arms only after 16 decode forwards since capture. Once armed, Triton and
    `tvm_ffi.load_module` loads first wait for device idle.
- **Deadlock 2, the soak:** a seeded soak (`analysis/dsv41-drive/ce-soak/`, seed 20260925)
  fail-stopped deterministically at decode step 14 of item 15, with items 11 and 15 both using
  min_p sampling.
  - At the stall no copy on any stream completed, including fresh ones.
  - It passes under `CUDA_MODULE_LOADING=EAGER` and fails under LAZY, 2 of 2 each in a reduced
    GPU scenario.
  - **Fixed:** the service refuses the copy engine unless `CUDA_MODULE_LOADING=EAGER`. `arm_env`
    sets EAGER, and raises `MEM_FRACTION_STATIC` from 0.80 to 0.83 for EAGER's ~1 GiB. The KV pool
    is 204,288 tokens (209,408 before).
  - The lazily loaded kernel is **not identified**. Finding it would allow a targeted warm-up
    instead of EAGER.
- **Soak status: incomplete, stopped by the owner.**
  - `s4`, with the fix, ran 91 requests over 43 minutes, including items 11 and 15, with no
    fail-stop and a largest inter-token gap of ~0.2 s.
  - The full ~2 h soak, a second seed, and mutants of the two copy-engine changes have not run.
  - **A ~30k-token prompt at 0.83 + EAGER is untested.** §25.4's Track A saw a 30k prompt peak at
    31.0 of 31.8 GiB at 0.80, so this is the most likely out-of-memory case.
  - To resume: the "Handoff (mid-task)" section of `2026-09-25-dsv41-copy-engine-soak.md`.
- **Operational rules:**
  - Never run graph-mode nsys with the copy engine on. `run_arm.sh` refuses it
    (`NSYS_CUDA_GRAPH_TRACE=node` is required for the default arm).
  - A first-time kernel on a path no earlier request took is the residual risk class. It
    fail-stops after the 2 s deadline and never produces a wrong answer.

### 25.4 Studies that closed

| Study | Result | Plan |
|---|---|---|
| Link and NUMA | HPE DL380 Gen10: the GPU is on a Gen3 x16 slot, the only kind the board has. The copy engine gets 13.67 GB/s and an SM zero-copy 12.23 GB/s, the same from either NUMA node. CPU memory load on the GPU's node cuts H2D 27–43%; load on the other node has no effect. All NVMe is on socket 1; nvme2 (the Engram table) runs at x2 | `analysis/dsv41-drive/numa-h2d/` |
| VRAM headroom (Track A) | None to spare. A 30k prompt peaked at 31.0 of 31.8 GiB, with allocator retries driven by the torch prefill indexer's score tensor. Keep the hot cache at 14336 MiB. Candidates, not built: cap the score tensor (~2 GiB), keep the dense modules quantized (2.8 GiB), move the embedding to host (1.23 GiB) | `analysis/dsv41-drive/hot-cache-size/` |
| Residency policy (Track B) | An exact replay matches measurement (G 78.783). The current policy is the best online policy found. Each +1 GiB of hot cache saves 2–2.7 misses per token. Belady's bound is 31 misses per token lower | `2026-09-25-dsv41-prefetch-study.md` |
| Side stream for the shared expert and `commit_gather` (1a) | Overlaps correctly, saves nothing at the wall; the flag stays off | `…-copy-compute-overlap.md` |
| Route-only predictors | Useless. Prefetch breaks even at a precision of ~0.25 and is worth building at ≥0.4 | `…-prefetch-study.md` |
| Native-gate lookahead | Layer T's own gate on layer T−1's input gives 0.68 rank-1 precision on non-resident rows | `…-router-capture.md` |
| Native prefetch, built (`SGLANG_DSV41_ENABLE_NATIVE_PREFETCH`, off) | Precision 0.73 live, byte-identical, but **123.2 vs 120.6 ms/token**. Only ~0.36 ms of compute sits between the post and the next gather, so ~0.62 ms of each ~1 ms copy is exposed | `…-native-prefetch.md` |
| Early-post prefetch | Replay on real per-layer windows: at best 117.8 vs 120.6, below the 8 ms bar. 68% of layers have no NVMe wait to hide under. Not built | `…-native-prefetch.md` |
| Resident-first MoE split | Byte-identical is possible (112/112 parity cases), but one `exl3_moe` launch costs ~92 µs whether it runs 1 or 6 experts. A second launch adds ~90 µs per layer | `…-per-expert-compute.md` |

### 25.5 Correctness fixes found on the way

- **Prompt-token logprobs under `--enable-decoder-swa-bounded-replay` crashed the server.** Rows
  outside the replay tail have no late-layer logits, so the error came from
  `_check_late_layer_tail_readers` during prefill. The scheduler now refuses such a request at
  admission with a message, and the TokenizerManager no longer crashes on the refusal
  (`c2f69ce864`, `d46a5f8e91`). Output-token logprobs are unaffected.
- **`Dsv41Config` had fallen six knobs behind `Envs`,** so `test_one_field_per_knob` failed. It now
  carries every `SGLANG_*DSV41*` knob again.
- **The reversed-CQE fault in `exl3_ram_miss_host.cpp` waits for every read in flight,** so a test
  that used it is no longer flaky.
- **Known test exceptions:**
  `test_graph_routes_are_logged_only_when_the_stage_trace_is_on[trace_on]` errors on the GPU at
  base too; `test_work_queued_behind_the_graph_on_other_streams_does_not_hold_the_copy_back[kernel]`
  and `test_a_hit_lanes_slot_is_never_its_own_requests_victim` (`lease_double_signal`) are rare
  intermittents.

### 25.6 Next decode work

Superseded by §26.2, which orders and expands this list.

1. **Finish the copy-engine soak** (§25.3): the full soak, a second seed, the 30k-prompt memory
   test at 0.83, the mutants, and EAGER's ms/token cost. If 0.83 runs out of memory, find the
   highest fraction that survives, or identify the lazily loaded kernel and warm it up instead of
   using EAGER.
2. **Fewer bytes per token.** The link carries ~1 GB per token. Return VRAM to the hot cache
   (Track A's candidates, ~6 GiB together). Each GiB saves ~2–2.7 rows per token at ~1 ms each.
3. **Why one H2D stream stops at ~13.7 GB/s** on Gen3 x16. The platform cannot go faster, but the
   gap to line rate has not been explained.

## 26. Decode performance roadmap and plan status (2026-09-25)

This section expands §25.6. It lists the next decode performance steps in order, then every DSV4.1
plan file with its status. Numbers are measured unless marked *estimate*.

### 26.1 How decode got here

| Date | Change | ms/token | Where |
|---|---|---:|---|
| 2026-09-24 | Row-image recipe (piece streaming + row images) | 132.1–132.4 | §24.9 |
| 2026-09-25 | + layer fusion (−3.2) and Engram device wait (−2.6) | 125.5 | §25.2 |
| 2026-09-25 | + copy engine (A 119.3 → B 112.4, pooled, same session) | **112.4** | §25.1 |

Production has run this recipe since 2026-09-25 16:31: `master` `b9620985ed`,
`CUDA_MODULE_LOADING=EAGER`, memory fraction 0.83, copy engine armed after 16 decode forwards.
The 112.4 arms ran under LAZY at 0.80, so production's own ms/token is not yet measured (item 1).

Where the ~111 ms/token goes (§25.1, node trace):

- ~78–81 ms: the PCIe link moving ~76 RAM-hit rows of 13.3 MB each (~1.02 GB per token at
  12.57 GB/s).
- ~16–20 ms: the rest of S (streamed rows and NVMe waits).
- ~13.7 ms: all compute.
- ~2 ms: small chain kernels.

**The link is the bottleneck.** Overlap has run out: there is at most 13.7 ms of compute to hide
copies behind, and three overlap studies have closed (§25.4). The levers left are fewer bytes per
token, a faster link, and less NVMe exposure.

### 26.2 Next steps, in order

GPU work cannot run while production holds `cc-gpu.lock`. Items 1, 2, 4, 5 and 6 need production
stopped (`run_server.md` D7).

1. **Finish the copy-engine soak.** This gates trusting production.
   - Remaining: the full ~2 h soak, a second seed, mutants of the two copy-engine changes, and a
     ~30k-token prompt at 0.83 + EAGER. The 30k prompt is the likeliest out-of-memory case: it
     peaked at 31.0 of 31.8 GiB at 0.80 (§25.4).
   - To resume: "Handoff (mid-task)" in `2026-09-25-dsv41-copy-engine-soak.md`.
   - **Also measure EAGER's cost** with the copy engine off, A = LAZY then B = EAGER. The copy
     engine refuses LAZY, so the two cannot be compared with it on.
2. **Replace EAGER with a targeted warm-up.** The kernel that loads lazily and hangs the copy engine
   (§25.3) is not identified. Once it is, warming it up at startup would return EAGER's ~1 GiB, to
   the hot cache or to prefill headroom.
3. **Fewer bytes per token: return VRAM to the hot cache.** This is the largest lever.
   - Each +1 GiB of hot cache saves 2–2.7 misses per token (Track B), at ~1 ms of link per miss.
   - Track A's three candidates, none built, total ~6 GiB:
     - cap the torch prefill indexer's score tensor (~2 GiB);
     - keep the dense modules quantized (2.8 GiB);
     - move the embedding to host (1.23 GiB).
   - Together, *estimate* 12–16 ms/token.
   - **Constraint:** Track A found no spare VRAM at 0.80. Each freed GiB must first cover the
     30k-prompt peak from item 1 before it can go to the hot cache.
4. **Faster link: close the gap to line rate.**
   - One H2D stream reaches 13.67 GB/s. Gen3 x16's theoretical rate is 15.75 GB/s, so at most 13%
     of link time (~10 ms/token, *estimate*, an upper bound) is unexplained.
   - Things to try:
     - **Coalesce copies.** Each RAM-hit row is six segment copies today; try fewer, larger ones.
     - **Split the copies** across two streams or copy engines.
     - **Keep the GPU's NUMA node quiet.** CPU memory load there cuts H2D by 27–43% (§25.4), and
       the final arms ran with questdb's `java` at ~90% on the server cores. Move such load off the
       GPU's node before trusting any arm, and possibly in production too.
5. **Re-test the MoE side stream under the copy engine.** This is cheap: an environment flag and
   A-then-B arms.
   - `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM` (1a) was measured only before the copy engine existed.
     It overlapped correctly, but 131.8 vs 131.9 ms/token showed no gain. Part of its saving came
     back as a longer C1, because the SM copy kernel shared SMs with the shared-expert GEMVs
     (`2026-09-25-dsv41-copy-compute-overlap.md` §7).
   - With the copy engine, C1 is DMA and uses no SMs, so that interference is gone.
   - At stake: the shared expert (1.62 ms/step) plus `commit_gather` (0.35 ms of bookkeeping,
     after fusion). *Estimate* ≤2 ms/token.
   - Not yet validated: the side stream together with the copy engine's wait kernel in the same
     graph. Run the side-stream GPU tests and a soak item before any arm.
6. **Less NVMe exposure (the ~16–20 ms of S).** The session-aware RAM cache plan targets this and
   is not started. The Belady bound shows 31 misses per token between today's policy and the
   optimum (Track B). Today's policy is the best online one found, so closing that gap needs
   prediction, not a better recency rule.
7. **Parked:**
   - **Prefetch** (native gate, early post): precision is good enough (0.68–0.73), but there is
     no idle link or compute to hide under. Revisit only if items 3–4 leave the link well under
     70% busy.
   - **Resident-first MoE split:** each extra `exl3_moe` launch costs ~90 µs per layer.
   - **DSpark:** blocked on four items (§18.5), and a 6-token verify routes more experts per step,
     which means more link bytes.

### 26.3 Plan files and progress

All under `docs/superpowers/plans/`. The plans do not tick their checkboxes. Status comes from the
sections cited and the merged code.

| Plan | What | Status | Results |
|---|---|---|---|
| `2026-09-18-dsv41-phase0.md` | Base branch, Engram parity, quant-file scope, `_HostTable` | Done | §11, §14 |
| `2026-09-18-dsv41-phase1.md` | EXL3-first bring-up | Done | §15 |
| `2026-09-18-dsv41-phase3a.md` | Eager three-tier streaming, `G`/`f` | Replaced by revision 2 | §16 |
| `2026-09-18-dsv41-phase3a-r2.md` | 3a on the MoE expert framework | Done | §16 |
| `2026-09-19-dsv41-phase3b-optionC.md` | MoE inside the decode graph, io_uring RAM-miss thread, device wait | Done; the base of today's path | §17 |
| `2026-09-19-dsv41-phase3b1.md` | "Graphs first": breakable graphs, one readback per layer | Superseded by option C, which met its goals | §17, §18 |
| `2026-09-19-dsv41-dspark.md` | DSpark on the EXL3 stack, phase D1 | Parked by the owner 2026-09-19; not started | §18.5 |
| `2026-09-19-dsv41-session-aware-ram-cache.md` | Session-aware RAM admission and replacement | Not started (route-log tooling only) | item 6 |
| `2026-09-24-dsv41-piece-streaming.md` | Stream NVMe rows to the GPU piece by piece | Done; in the recipe | §24 |
| `2026-09-24-dsv41-row-images.md` | Read rows straight into the pinned slabs | Done; in the recipe | §24.9 |
| `2026-09-25-dsv41-engram-no-hostnode.md` | Engram lookups without host nodes | Done; in the recipe (−2.6) | §25.2 |
| `2026-09-25-dsv41-layer-fusion.md` | Fuse the per-layer bookkeeping chains | Done; in the recipe (−3.2) | §25.2 |
| `2026-09-25-dsv41-copy-compute-overlap.md` | 1a side stream, 1b copy engine, item 3 Python gaps | 1a closed (flag off; see item 5); 1b became the copy engine; item 3 has no decode value | §25.3, §25.4 |
| `2026-09-25-dsv41-copy-engine-soak.md` | Soak the copy engine | **In progress:** the LAZY hang is fixed with EAGER; `s4` ran 91 requests clean; full soak, 2nd seed, mutants and the 30k prompt remain | §25.3, item 1 |
| `2026-09-25-dsv41-final-arms.md` | Full A/B arms and node trace | Done: 119.3 → 112.4, byte-identical | §25.1 |
| `2026-09-25-dsv41-prefetch-study.md` | Offline prefetch study (route-only, residency replay) | Done: route-only predictors useless | §25.4 |
| `2026-09-25-dsv41-router-capture.md` | Router-input capture, native-gate lookahead | Done: 0.68 rank-1 precision | §25.4 |
| `2026-09-25-dsv41-native-prefetch.md` | Native next-layer prefetch, plus the early-post estimate | Built, flag off (123.2 vs 120.6); early post not built (misses the 8 ms bar) | §25.4 |
| `2026-09-25-dsv41-per-expert-compute.md` | Compute experts as they land (resident-first split) | Closed: step 2 fails the gate | §25.4 |

## 27. The transfer path under a traced arm, with PCIe RX/TX (2026-09-25)

A node-mode arm at `master` `69df716412` was traced together with GPU-metrics sampling (PCIe RX/TX, §26, `run_arm.sh`
`NSYS_GPU_METRICS`):

- `NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node run_arm.sh pcie-node 30021`: exit 0, result gate OK.
- The run has two timed sessions, each a 260-token prompt: TTFT 23.6 s with 7 output tokens, and 19.8 s with 103.
- Reports on divix01: `/mnt/nvme1/dsv41-nsys/pcie-node-20260925-170510{,-pcie}.nsys-rep`. Laptop copies and their
  sqlite exports: `/home/dimitri/data/divix/nsys-reports/`.
- Scripts: `analysis/dsv41-drive/pcie-trace/`. Each script reads the laptop sqlite paths; run them under
  `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`.
- The PCIe report starts 530.79 ms before the trace (compare the two `TARGET_INFO_SESSION_START_TIME`): trace time
  t is PCIe time t + 0.531 s.

Node mode inflates small-kernel and graph-launch cost, so no ms/token below comes from this trace.
Traced decode steps average 118.4 ms, against ~111 ms/token untraced (§25.1).

### 27.1 PCIe metrics on this machine

- **The link is Gen3 x16.** `nvidia-smi -q` reports Host Max 3 and Device Max 5. An idle GPU drops the link to Gen1,
  so read the link speed under load.
- **nsys's "PCIe RX Throughput %" is not a fraction of that link.** During copy-engine bursts measured at
  12.17 GB/s, RX reads 42.8%, so 100% ≈ 28.4 GB/s and 1 point ≈ 0.284 GB/s. The highest RX observed, 46%, is
  ≈ 13.1 GB/s, the practical Gen3 ceiling.
- **Counter access:** GPU counters are admin-only on divix01 (`RmProfilingAdminOnly: 1`). `run_arm.sh` therefore
  samples them in a second root session through `sudo -n /usr/local/sbin/nsys-profile` (CLAUDE.md).

### 27.2 Prefill: the transfer path is serial, and it dominates TTFT

Session 1's prefill ran from 1.18 to 24.56 s in the trace, 23.4 s for 260 tokens:

| Component | Time | What it is |
|---|---:|---|
| `_gather_host_rows_kernel` | 6.5 s | An SM zero-copy gather from pinned RAM at 12.33 GB/s, which is link-bound. There are 129 calls; each is 6 segment launches over ≤64 experts (the staging buffer holds 64), ~47 on average. That moves ~78 GB for 260 tokens: ~146 non-VRAM experts × 13.3 MB per layer. |
| GPU idle, host busy | ~12 s | 128 gaps of 10–250 ms, about 3 per layer, one per gather chunk. During them the scheduler thread makes no CUDA call and no traced syscall: it is filling the pinned tier (below). |
| GPU idle, launch gaps | 3.9 s | ~157,000 gaps under 0.1 ms between ~149,000 eager kernels, mostly torch glue: ~65,000 elementwise, ~10,000 index kernels, and ~23,000 cub select/reduce/compact kernels from per-chunk routing bookkeeping. Prefill runs without CUDA graphs (`--cuda-graph-backend-prefill disabled`). |
| Other kernels | ~0.5 s | All other prefill compute. |

**How the pinned-tier fill works:** `_gather_cached` → `ExpertPinnedHostCache.gather_rows` →
`ensure_rows` → `Exl3ShardRowSource.read`. Each chunk's fill runs synchronously on the scheduler thread:

- **Read:** an io_uring read (liburing's raw syscalls, which nsys does not see) into a **bounce buffer**.
- **Copy:** a **single-threaded CPU `copy_`**, one segment at a time, into the pinned slot.

Decode dropped this double handling with row images (§24.9); the prefill path still has it. Each chunk runs
fill → gather → `.item()`, with no overlap between NVMe, CPU and GPU.

**The pageable copies in the trace are sync points, not a pinning problem.**

- **Readbacks:** there are 1,325 device-to-pageable readbacks of ≤256 B during prefill (`hit_mask.sum().item()`,
  `chunk.tolist()`, `torch.unique(...).numel()`). Their GPU time totals 0.87 ms, but the host is blocked in them for
  12.5 s across both prefills, waiting for queued gathers to finish.
- **Larger copies:** the 1.52 MiB and 48.75 KiB pageable copies total 0.65 ms.
- **Decode:** has only 10 pageable copies.

**Prefill evicts decode's RAM set.** The RAM-miss service reads the same `pinned_host_cache` that prefill fills.
After session 2's prefill:

| Decode steps | NVMe wait (S) per step | Step wall |
|---|---:|---:|
| 0–4 | 64 ms | 149 ms |
| 70–102 | 36 ms | 107 ms |

One session, so this is directional (`s_decay.py`).

**Long prompts, *estimate*.** With `--chunked-prefill-size 512`, every chunk routes 3,072 lanes per layer, which
touches nearly all 384 experts. That streams ~356 non-VRAM rows per layer, ~190 GB per chunk: ≥15 s per chunk at the
link rate. A 30k-token prompt (59 chunks) would take on the order of 15 minutes to first token. Not measured.

### 27.3 Decode: the link is well used while copying; the idle link is compute and NVMe waits

- **Copy engine:** per step, 458 copies (~2.2 MB each, six segments per row), 1.02 GB in total, on stream 141.
  - Within a layer the copies run back to back: 46,500 of 50,412 gaps are under 10 µs.
  - Busy throughput is 12.4–12.5 GB/s, ~91% of the 13.7 GB/s bench.
  - A layer's first copy starts a median 20 µs after its post kernel ends (`ce_latency.py`).
- **Link use per step, from RX samples joined to kernel state (`pcie_decode.py`):**

| State | ms/step (traced) | Mean link rate |
|---|---:|---:|
| Copy engine running, GPU in CW | 54.1 | 12.5 GB/s |
| Copy engine running, GPU in S | 27.0 | 11.5 GB/s |
| No copy, GPU in S (NVMe waits and SM pulls of streamed pieces) | 17.4 | 7.4 GB/s; RX < 5% in 25% of samples |
| No copy, compute | 19.3 | 0.9 GB/s; RX < 5% in 85% of samples |

  The link carries ~1.14 GB per step in all. Only prefetch could use the idle link during compute (§25.4).
- **Chain order:** post → W1 → S → CW → MoE. S runs before CW, and the RAM-hit copies overlap S (`ce_order.py`).
- **After a layer's RAM-hit copies land, the layer waits for NVMe: 13.3 ms per step.** The time from a layer's last
  copy to the end of its CW is S still waiting for NVMe-streamed rows, not a CW wait (`tail_one.py`).
  - The chain order is S → CW, so CW starts only when S ends, and it then runs for ~5 µs.
  - An earlier reading of this trace called the 13.3 ms a "CW tail caused by the copy thread not running". That was
    wrong. `exl3-copy-eng`'s 1,250 silences ≥ 0.5 ms (23% of 47–58.6 s, median 1.5 ms, `ce_silence.py`) are the
    thread idle: with nothing queued or in flight, it waits on its condition variable for up to 1 ms.
  - This time is part of the "no copy, GPU in S" state above: the link is partly idle while NVMe serves the
    layer's remaining rows.
- **The link carries no duplicate or padded bytes.** 111.88 GB in 50,412 copies averages 2.219 MB, and six
  segments make 13.32 MB, exactly one row (13,315,584 B). Each RAM-hit row crosses once per step.
  - The link carries ~1.14 GB per step, against 1.02 GB of copy-engine traffic plus a few streamed rows (~0.04 GB).
    The remainder is within the RX calibration's error.
  - Rows re-copied on consecutive steps are a residency-policy cost (Track B: Belady's bound is 31 misses per token
    lower), not a lease-protocol cost. Hot-cache insertions copy device to device (7.6 GB on stream 13), off PCIe.
- **NVMe waits are concentrated in a few layers** (`more.py`). S time per step: layer 0 3.10 ms, layer 19 2.43,
  layer 1 2.11, layer 23 2.10, layer 39 2.09, 47.0 ms in total. Only layers 0, 1 and 19 have a median S above 1 ms
  in every step.
  - **Not hash routing:** DSV4.1's layer 0 has a learned gate (`layers.0.ffn.gate.weight`, `.bias`, no `tid2eid`),
    and the config sets no hash layers, so these routes cannot be known before the layer runs.
  - **The pinned tier is split evenly:** `ExpertPinnedHostCacheManager.from_model` deals the 100 GiB budget out
    round-robin, one row per layer per pass. Every layer gets 201–202 of the 8,063 rows, whatever its miss rate.
  - **Layer 0 has least cover:** it runs first in the step, so no earlier compute hides its NVMe reads.
- **Small kernels are on the critical path** (node mode, so inflated). Per decode step: 15.1 ms of compute kernels,
  ~1,800 of them under 3 µs, and 3.0 ms of gaps between kernels.
  - The link is idle ~85% of the compute time, so compute is serial with the transfers and every ms removed is a ms
    per token.
  - Measure it in graph mode with the copy engine off; graph mode with the copy engine on is refused.
- **The first decode step after a prefill is slow:** the two sessions' first graphs ran 240.8 and 198.9 ms on the
  GPU, against ~110 ms in steady state. This is consistent with prefill evicting decode's RAM rows plus a stale hot
  cache.

### 27.4 Next steps

1. **Prefill fills: done (§27.6), behind `SGLANG_DSV41_ENABLE_PREFILL_FILLS`.** TTFT fell from 21.1/17.8 s to
   12.1/11.1 s.
   - Done: native direct reads through the service's reader; a layer's reads issued once it has routed; chunk k+1 read
     while chunk k gathers.
   - Not done: the per-chunk readbacks and the per-expert `torch.where` syncs stay (item 7).
   - Not done: the copy-engine gather. The SM gather already runs at the link rate, so DMA would only overlap it with
     the ~0.5 s of MoE compute, and that needs a second ~852 MB staging set.
2. **Keep prefill from evicting decode's RAM set:** stage prefill misses without admitting them, or protect
   decode-hot rows.
3. **NVMe waits in decode:** ~13 ms per step of S outlasts the layer's RAM-hit copies, with the link partly idle.
   Fewer NVMe misses (the RAM tier, and item 2) or faster streaming are the levers. The earlier "copy-thread" reading
   of this time was wrong (§27.3).
4. **Long prompts:** larger prefill chunks, which trades against Track A's VRAM headroom (§25.4).
5. **Per-layer RAM split: tried, no gain; not merged.** Branch `cc/pinned-layer-weights`: `SGLANG_MOE_PINNED_HOST_LAYER_WEIGHTS`
   plus `scripts/dsv41/ram_split.py`.
   - **Replay:** an LRU stack-distance replay of the varied24 route log reproduces the measured per-layer ranking, but
     runs 12% high (12.66 against 11.26 RAM misses/token). A greedy split fitted on even-numbered requests cut
     held-out RAM misses from 11.99 to 11.16 per token (−7%).
   - **Arms** (`pinw-A` even, then `pinw-B` weighted, same commit, byte-identical outputs): decode NVMe `rows_read`
     went **up**, 4,387 → 4,554 (+3.8%). Prefill RAM misses were flat (10,967 → 10,926). TTFT and tok/s moved
     within noise.
   - **Likely cause, not verified:** the replay models decode only. Prefill admissions churn the same tier, and
     layers cut to ~116 rows lose more of their rows to them. Revisit after item 2 stops prefill evictions.
6. **Decode small kernels:** another fusion round over the ~1,800 sub-3 µs kernels per step. Worth a few ms/token,
   *estimate*.
7. **Prefill glue:** the ~149,000 eager kernels per 260-token prefill, once item 1 has removed the serial fills.
8. **Environment:** the spinning tmux server and questdb's `java` share the server's cores and contaminate every arm.

### 27.5 Copy-thread scheduling, untraced

An untraced arm (`sched-probe`, same commit and recipe) sampled `/proc/<pid>/task/<tid>/schedstat` and the context
switches once a second (`divix01:/mnt/nvme1/sched-probe/`). It confirms the corrected reading in §27.3: nothing
preempts the copy thread.

| Thread | On CPU | Run-queue wait | Voluntary / involuntary switches |
|---|---:|---:|---:|
| `exl3-copy-eng` | 35.8% | 11 ms of 175 s (0.01%) | 106,834 / 132 |
| `exl3-ram-miss` | 34.0% | 52 ms (0.03%) | 1,311,302 / 322 |
| scheduler | 84.8% | 12 ms (0.01%) | 24,870 / 302 |

### 27.6 Prefill fills through the native reader (result)

**Flag.** `SGLANG_DSV41_ENABLE_PREFILL_FILLS` (default off). It needs row images and option C's native slot table.
Plan: `docs/superpowers/plans/2026-09-25-dsv41-prefill-fills.md`.

**How it works.** Eager pinned-tier misses are now read by the RAM-miss service's own `RowReader`, straight into the
slabs, on a helper thread (`RamTier::fill_begin/fill_wait/fill_end`).
- `_apply_streamed` holds one host use per layer, which syncs the stream and pauses the service thread once. Inside
  it, the layer claims slots for every expert that misses both VRAM and RAM, in chunk order.
- Each gather chunk waits only for its own rows.
- A claimed slot is flagged `filling` until its read ends. A filling slot is never a victim and cannot be released.
- The service thread's `resume` joins any fill still running.

**Arms.** Master `b1441f9b0b` on port 30021: A (flag off) at 19:17, then B (flag on) at 19:37, once each. Both
returned rc 0 with every harness gate passed.

| | A (off) | B (on) |
|---|---:|---:|
| TTFT, session 0 / 1 (s) | 21.06 / 17.79 | **12.05 / 11.14** |
| ms/token, session 0 / 1 | 138.8 / 110.5 | 139.5 / 110.9 |
| pooled ms/token | 112.1 | 112.5 |
| outputs vs A | | **2 of 2 byte-identical** |
| eager `read_ms` / `split_ms` (whole run) | 22,732 / 24,974 | 1,836 / 0 |

- **`read_ms` means something different in B.** With the flag on it is only the time the scheduler thread was blocked
  waiting for a fill.
- **`ram_misses` also changes meaning: 10,948 in A, 923 in B.** B's prefetched rows are already claimed when the
  chunk looks them up, so they count as hits. Only overflow admissions still count as misses.
- **Decode is unchanged.** ms/token matches within 0.4 ms. That is a single run, so treat it as directional.

Evidence is on divix01:
- `cc-expert-prediction/dsv41-baseline/servers/prefill-fills-{A,B}/`.
- `/mnt/nvme1/prefill-fills/arm_metrics_AB.json`.

**What remains of the 11-12 s.**
- The link-bound SM gather: ~6.5 s.
- The eager launch gaps and per-expert syncs (item 7).
- The first chunk's reads of each layer, which nothing overlaps.

## Sources

- Official repo snapshot and tech report (paths in §1).
- EXL3 model card: <https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw>.
- exllamav3: <https://github.com/turboderp-org/exllamav3>.
- SGLang `dsv4.1`: <https://github.com/sgl-project/sglang/tree/dsv4.1> (local
  `upstream-scope/dsv4.1` @ `85e8eddc54`).
- CUDA runtime docs for `cudaLaunchHostFunc` (host-function restrictions and stream
  semantics).
- divix01 analysis: `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/`
  (`cross-token/spec_window_summary.txt`, `strategy/static_curves.json`).
- Our fork: §8 line references; [`MOE_EXPERT_TRANSFER.md`](MOE_EXPERT_TRANSFER.md).
- vLLM EXL3 (§14.4): <https://github.com/vllm-project/vllm/issues/19896>,
  <https://github.com/vcruz305/vllm-exl3>.
