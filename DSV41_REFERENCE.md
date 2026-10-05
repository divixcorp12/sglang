# DeepSeek V4.1 Flash — scoping reference


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

**Update (2026-09-29):** §29 records the merges of 2026-09-27 to 2026-09-29. The recipe now reads expert rows from
three mirror roots, always uses row images and lease mode (the packed path, its pack workers and the
`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` knob are deleted), launches the lease chain with PDL, and runs at
`MEM_FRACTION_STATIC` 0.885, hot cache 15400 MB and context 131072 on NVIDIA driver 615.71.09 (§29.11). The last
merge, the copy thread's completion word (§29.9), went in without review or a decode A/B.

**Update (2026-09-29, evening):** the copy wait is stream-ordered and no longer spins (§29.14). The recipe runs at
`MEM_FRACTION_STATIC` 0.875 and hot cache 16080 MB (§29.15). It warms the prefill Triton variants at startup (§29.16),
serves `--language-model-only` without the empty vision tower (§29.18), and backs HiCache with a ~9 GB host pool on
the direct IO backend (§29.19). A grammar request no longer crashes the single-GPU server (§29.17).

**Environment variables:** §32 lists the DeepSeek V4.1 and expert-streaming variables, their defaults and the
production values.

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
  `/mnt/nvme2` and uses expert-row mirrors on `/mnt/nvme0` and `/mnt/nvme4` (and on `/mnt/nvme2` as a third root since
  2026-09-28, §29.3).

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
`base_env()`. (Since 2026-09-29 there are no pack workers and no lease knob: row images and lease mode are
unconditional, §29.8.) Note that mirroring
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
| Doorbell (`expert_doorbell.py/.cuh`) | **Removed 2026-09-27 (8ac64c9c99).** Refuses overlap scheduling (`model_runner.py:797-801`), all speculative decoding (`:769-772`) and decode graph bs > 1 (`:773-775`). The overlap refusal is a *placement* constraint (fail-stop check after results), not physics |
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
- The tech report gives **no acceptance-rate numbers**. Measured eager on the EXL3 stack (§33.2): mean accept
  length 2.88 (α ≈ 0.69) at 128 tokens, ~3.4 at 64k context.

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

**Superseded (2026-10-02):** every blocker below was closed on 2026-09-19 and DSpark runs eager; results and
the graphed-verify analysis are in §33.

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
negotiated x2 of 4. PCIe Advanced Error Reporting correctable counters 0. One snapshot, so a transient
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

*Note 2026-09-29: the knob below and the packed path are deleted (merge `2810a7ad48`, §29.8); lease mode is always on,
and a set value warns and is ignored (`python/sglang/srt/environ.py`). The measurement stands as history.*

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
  - *Since 2026-09-29 (§29.8):* pack workers and the lease knob are gone; the service always reads row images in
    lease mode.
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

*Note 2026-09-29: row images are now the only reader; `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES` and
`SGLANG_DSV41_RAM_MISS_PACK_WORKERS` are deprecated (warn, ignored) and a root without row images refuses at startup
(§29.8, `python/sglang/srt/environ.py`).*

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
- **Current state (2026-10-02):**
  - With `SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES=1`, in the recipe, the engine copies only each RAM-hit row's two
    trellis tensors and the copy wait reads the four small ones (§27.14, `analysis/dsv41-drive/LEASE_PROTOCOL.md`), so a row is
    not six DMA copies.
  - The "lease held until an event" and the W1 wait described above are gone: the slot-map protocol (§31) has no
    leases, and the copy wait is one stream-ordered gate on the completion word that the copy thread's
    `cuStreamWriteValue32_v2` writes after each job's copies (`host/copy_engine.h`).
  - `SGLANG_DSV41_RAM_HIT_COPY=sm` replaces the engine with the in-graph SM copy (§31.1).
  - The engine arms after `COPY_ENGINE_ARM_DECODES` = 16 captured decode forwards. With CPU experts on, the CPU/DMA
    split is calibrated at that moment (§30.1).
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
   - **DSpark:** runs eager at 1.4-2.2 tok/s (§33.2). A graphed verify needs the record, route plan and in-graph
     MoE widened past one token, and a capped width with an overflow path (§33.3); the offline union curve
     decides whether that is worth building.

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
| `2026-09-19-dsv41-dspark.md` | DSpark on the EXL3 stack, phase D1 | D1 Tasks 1-7 done; Task 8 partial (parity, α; no rows/token or break-even). D2 not started | §33 |
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
2. **Keep prefill from evicting decode's RAM set: done, small; default off (§27.9),** behind
   `SGLANG_DSV41_ENABLE_PREFILL_SHARE`. A bounded 64-row prefill share cut S by ~3 ms per step and cost 0.5–1.2 s
   of TTFT. Staging without admission is worse. Most of early decode's excess misses are a cold start.
3. **NVMe waits in decode:** ~13 ms per step of S outlasts the layer's RAM-hit copies, with the link partly idle.
   Fewer NVMe misses (the RAM tier, and item 2) or faster streaming are the levers. The earlier "copy-thread" reading
   of this time was wrong (§27.3).
4. **Long prompts: done (§27.17).** 4096-token prefill chunks, with the hot cache cut by 1 GiB and the static fraction
   lowered to 0.90: a 16k prompt's TTFT fell 444 -> 107 s. (Superseded 2026-09-29: 0.885 / 15400 / context 131072,
   §29.11.)
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
6. **Decode small kernels: done for the EXL3 cast glue (§27.8), behind `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION`.**
   -358 kernels per step, byte-identical, -0.6 ms/token (within noise). Not "a few ms": most of the remaining small
   kernels are attention, mHC reductions and the RAM-miss chain, not glue.
7. **Prefill glue: partly done (§27.11), behind `SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN`.** The host no longer waits
   for each chunk's gather before launching its compute: TTFT 12.6 → 9.9 s and 11.5 → 9.2 s, outputs identical,
   prefill GPU idle 49% → 29%. Then the split fill gather (§27.12, behind
   `SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER`): TTFT 9.9 → 8.6 s and 9.2 → 8.2 s, outputs identical, prefill GPU
   idle 29% → 21%. In the recipe since `e0bd452bf6`. Left: 1.76 s of GPU idle, of which ~0.99 s is demand reads
   for chunks the fill did not claim (§27.13), ~0.5 s Python and launches, and ~0.2 s `pthread_cond_wait`.
   Copying landed rows first (§27.13) is correct but moved nothing.
8. **Environment:** the spinning tmux server and questdb's `java` share the server's cores and contaminate every arm.
9. **Prefill indexer score cap: done, in the recipe since `24bbd3baab` (§27.7).** A 128 MB cap frees ~3.0 GiB at
   30k/32k and removes the OOM retries. Spent on 3 GiB more hot cache (mem-fraction 0.925): 110.1 -> 103.0 ms/token,
   output identical, 116 MB/step fewer RAM-hit copies. Open: a 30k-32k prompt at these settings (~0.5 GiB headroom).
10. **Decode RAM-miss frontend (W1/C1/A1): sized, no-go (§27.15).** At most 0.41 ms/step to gain; neither
    `HIT_WAIT_US=0` nor a reset-only frontend was built or run. Resident-first stays shelved: with real copies,
    overlap saves 10-13 us/layer against a 25 us bar,
    ~1 us of it the resident launch and the rest a same-stream copy residual.
11. **Multi-turn prefix reuse: done, HiCache in the recipe (§27.16).** The 3584-slot SWA pool lost a 4k conversation's
    tail to the next prefill, so revisits reused nothing (TTFT 130 s). HiCache reloads the tail from host (2.3 s), and
    cache hits matched cold output within noise. Two defects found and fixed: HiCache staging buffers ate the runtime
    slack (CUDA OOM; capped and reserved), and decoder bounded replay let a mid-chunk hit read unwritten late-layer
    SWA slots (upstream bug; a hit now re-prefills the window, `c515312414`).

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

### 27.7 Prefill indexer score cap (measured; payoff -7 ms/token, 2026-09-26)

**Why.** Track A (§25.4) found no VRAM headroom: a 30k prompt peaked at 31.0 of 31.8 GiB, with allocator retries from
the torch prefill indexer's score tensor. Freed VRAM can go to the hot cache, at 2–2.7 fewer misses per token per GiB
(Track B).

**The path, corrected.** Production runs `SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`, so SM120 prefill scores in
`DeepseekV4AttnBackend._low_ratio_index_topk_torch` (`deepseek_v4_backend.py`), not in `indexer.py`'s
`fp8_paged_mqa_logits_torch_sm120`.
- It already shares one key copy per request: an einsum over `[rows, 32 heads, lc]` in bf16.
- It already chunks rows, but under a hard-coded 1 GiB budget (`_TORCH_INDEXER_SCORE_BUDGET_BYTES`). With 512-row
  prefill chunks and at most 32k context, that budget never splits: one tensor is ~0.98 GB at 30k, and the einsum,
  `relu` and `*weights` keep ~3x that live.
- Track A's failed allocations grow 32 MiB per 512-token chunk, which is this einsum at the ratio-1 layers, so the
  attribution holds.

**What was built** (branch `cc/indexer-cap`, head `b9ae131af0`, on `01e0a6ea7f`; merged to master at `c08f5484c9`, where `test/registered/unit/kernels` gives 1740 passed, 1 skipped against 1733 at the parent):
- `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB = EnvInt(0)`. 0 keeps the built-in 1 GiB, today's behaviour.
  Otherwise rows per chunk = `max(1, budget // (heads * lc * 2))` (`_torch_indexer_rows_per_chunk`).
- Only the chunking changes: no truncation of the scored context, same precision, same top-k. Rows are scored and
  reduced independently, so the output should be unchanged.
- Tests: `test/registered/unit/kernels/test_dsv41_torch_indexer_chunking.py` compares page indices, raw indices and
  candidate masks bitwise, chunked against one pass, through the real `_low_ratio_index_topk_torch` and
  `DeepseekV41Indexer.scores`. It covers ties, a request shorter than `index_topk`, chunk sizes that don't divide the
  row count, and on CUDA the production shape (512 rows, 32 heads, 30k ratio-1 context).
  - CPU 4 passed, 3 skipped; GPU 7 passed; a mutant (the chunked path drops the last score column) fails 2.
  - `test/registered/unit/kernels` with CUDA hidden: 1290 passed, 413 skipped, EXIT=0. No merge-base comparison.

**Peaks and parity (measured 2026-09-26, `divix01:/mnt/nvme1/indexer-cap/`).** Run by `drive_peaks.sh` at 128 MB,
plus a repeat of budget 0 at 30k (`b0-30k-r2`). All smokes rc=0, on the branch's own recipe (`b9ae131af0`, on
`01e0a6ea7f`, older than today's master recipe), so the TTFTs do not carry over to production.

| Long prompt | Peak VRAM | Headroom | OOM retries | TTFT |
|---|---:|---:|---:|---:|
| 30k, budget 0 | 31.38 GiB | 0.47 GiB | 28 | 2,098 s |
| 30k, budget 0, repeat | 31.36 GiB | 0.49 GiB | 28 | 1,869 s |
| 30k, 128 MB | 28.33 GiB | 3.51 GiB | 0 | 1,851 s |
| 32k, budget 0 | 31.38 GiB | 0.47 GiB | 32 | 1,949 s |
| 32k, 128 MB | 28.35 GiB | 3.49 GiB | 0 | 1,958 s |

- **VRAM:** at 128 MB the long prompt adds nothing measurable over idle (~29.0 GiB). **~3.0 GiB freed** against the
  2.5 GB estimate, and the allocator retries go to zero.
- **TTFT:** no measurable change. The two budget-0 runs at 30k differ by 229 s, which covers the cap's -247 s at 30k
  and its +9 s at 32k.
- **Parity:** the six short greedy responses are byte-identical across all five runs. The 64-token long-prompt output
  is not reproducible even at a fixed budget: the two budget-0 runs diverge at character 75, earlier than budget 0 vs
  128 MB (character 109). So the long-output difference is run-to-run nondeterminism in this build's long prefill, not
  the cap; the chunking tests above remain the bitwise evidence.

**Payoff arm (2026-09-26, `f62f7af0b6`; driver `analysis/dsv41-drive/indexer-cap/drive_payoff.sh`).** The recipe
plus `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB=128` and `SGLANG_MOE_HOT_GPU_MB=17408` (+3 GiB, 232 more
slots).
- **The hot cache counts against `--mem-fraction-static`.** At 0.83, 17408 MB left no KV ("minimum viable 0.911") and
  both arms refused to start. The arm runs at 0.925 (+3072 MiB of the card's 32607), which keeps today's KV pool; the
  cap's freed prefill transient is what makes that fit. `run_arm.sh` takes it per arm as
  `DSV41_MEM_FRACTION_STATIC`; `arm_env.MEM_FRACTION_STATIC` stays 0.83.
- **Untraced** (`indexer-payoff`, one arm, compared with `sm-small-B` from 04:22 the same day, not an interleaved
  A/B): output byte-identical. The 103-token session decoded at 9.08 -> 9.71 tok/s, **110.1 -> 103.0 ms/token
  (-7.1)**; the other session is 7 tokens. TTFT 8.46/8.16 -> 8.41/8.07 s. Median ratio 1.047, 2 of 2 sessions faster
  (p = 0.25). The prediction was 6-8 ms/token (3 GiB x 2-2.7 misses per GiB x ~1 ms per miss).
- **Traced** (`indexer-payoff-node-20260926-143541`, node mode, against `sm-small-B-node-20260926-034122`, 90 steps
  each; `ce_trace.py` and `frontend_bound.py`):

| Per decode step | sm-small-B-node | payoff-node |
|---|---:|---:|
| RAM-hit copies (copy engine) | 994.6 MB, 149.9 copies | **878.4 MB**, 132.4 copies |
| Copy-engine busy | 78.0 ms | 69.2 ms |
| CW | 54.6 ms | **47.9 ms** |
| S (NVMe pieces) | 40.8 ms | 40.3 ms |
| W1 | 1.06 ms | 0.89 ms |
| Step span (node mode, inflated) | 113.7 ms | **106.5 ms** |

  The gain is fewer RAM hits to copy: -116 MB/step, ~8.7 rows per token at 13.3 MB. NVMe streaming is unchanged.
- **PCIe RX** (the root session's `-pcie` report; clock offset 482,882,340 ns from the session starts; calibration
  12.43 GB/s = 43.3% RX, so 100% ~ 28.7 GB/s; 108 steps, 122.1 ms/step traced): 1.05 GB/step, mean 8.6 GB/s.

| Link state | ms/step | Mean RX | ~GB/s |
|---|---:|---:|---:|
| Copying, GPU in CW | 46.7 | 44.5% | 12.77 |
| No copy, GPU in S | 32.4 | 16.2% | 4.67 |
| Copying, GPU in S | 23.6 | 41.2% | 11.82 |
| No copy, other | 18.4 | 3.1% | 0.89 |

  Decoded with a divix01 copy of `pcie-trace/pcie_decode.py` that takes its reports as arguments
  (`/mnt/nvme1/indexer-cap/pcie_decode_args.py`, output `payoff-pcie.txt`). Its step windows differ from §27.14's
  steady-step selection, so compare states within this table, not against §27.14's.
- **In the recipe since `24bbd3baab`:** `base_env()` carries both flags and `arm_env.MEM_FRACTION_STATIC` is 0.925,
  so `launch_prod.sh` and every arm use them. Not checked at these settings: a 30k-32k prompt, where the smokes
  (at 0.83) left ~3.5 GiB and this leaves ~0.5 GiB.

**To resume.**
- Driver: `analysis/dsv41-drive/indexer-cap/drive_peaks.sh <worktree> <budget_mb>` runs 30000 then 32000 tokens, each
  at budget 0 then `budget_mb` (tags `b<budget>-<N>k`), under `/mnt/nvme1/indexer-cap`:
  ```bash
  cd /mnt/nvme1/indexer-cap && setsid nohup bash \
    /data/models/slang/nvfp4-work/wt-indexer-cap/analysis/dsv41-drive/indexer-cap/drive_peaks.sh \
    /data/models/slang/nvfp4-work/wt-indexer-cap 128 > drive_peaks.log 2>&1 < /dev/null &
  ```
- One smoke is `analysis/dsv41-drive/indexer-cap/smoke.sh <tag> <worktree> 14336 <budget_mb> <long_tokens>`:
  Track A's smoke plus the budget override. It refuses to run unless the budget and the torch prefill indexer are in
  the live server's environment, uses `arm_env`'s mem-fraction (0.83), takes `rowimg-disk.lock` before
  `cc-gpu.lock`, and counts OOM retries into `retries.txt`.
- Peaks: `analysis/dsv41-drive/hot-cache-size/vram_peaks.py /mnt/nvme1/indexer-cap/<tag>`. Parity: diff
  `responses.jsonl` and the `long.json` text between budgets; TTFT is in `long.json`.
- Before the payoff arm, re-check the lock order (`rowimg-disk.lock`, then `cc-gpu.lock`) against the other drivers.
- `wt-indexer-cap` on divix01 is clean at `b9ae131af0`.

### 27.8 Decode cast fusion (result)

Plan: `docs/superpowers/plans/2026-09-25-dsv41-decode-fusion-2.md`. Flag: `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION`,
default off, not yet in `arm_env`.

**Where the small kernels were.** In the traced decode step (§27.3), 566 of the ~1,800 sub-3 µs kernels are the fp16/bf16
casts around the seven dense EXL3 gemvs per layer: `exl3_gemm` takes and returns fp16, and the model runs in bf16. Three
of those casts per layer redo the same conversion: wq_a and wkv share an input, and so do the shared expert's gate and
up. The rest of the small kernels are not glue:

- attention metadata and norms, 537;
- the mHC reductions, 320, which cannot be fused bit-exactly;
- the RAM-miss chain, 320-400.

**What was fused (F1-F4), all at BS1:**

- **Sublayer input.** `hc_combine_norm` also writes the fp16 copy of the sublayer input and publishes it (a one-slot
  registry keyed on the tensor object, its version and the stream). wq_a, wkv and the shared expert take the copy
  instead of casting.
- **Merged linears.** A merged linear writes every part into one fp16 row, with one output cast and no `cat`.
- **Shared expert.** It stays in fp16 between its gemvs: `silu_mul_clamp` reads the fp16 gate/up and writes the fp16
  down input, keeping both bf16 roundings.
- **Routed output.** The cast to bf16 and `* routed_scaling_factor` are one kernel.

Every kernel is compared as bits against the chain it replaces, and the chain captures with 0 host nodes.

| | Flag off | Flag on |
|---|---:|---:|
| kernels per decode step (node mode, copy engine off) | 2,676 | **2,318** |
| pooled ms/token (arms A then B, same commit `cc7620a247`) | 112.4 | **111.8** |
| 103-token session ms/token | 110.9 | 110.2 |
| outputs | | 2 of 2 byte-identical |

The gain is ~1.7 µs per removed kernel. It matches the round-1 rate (§25.2) but is within two-session noise
(p = 0.75). Left for a later round: folding wq_b's and wkv's output casts into the q rope and k-norm-rope, and wo_b's
into the mHC post (3 kernels per layer, ~0.1 ms/token each).

### 27.9 Keeping prefill from evicting decode's RAM set (result: small, default off)

**Flag.** `SGLANG_DSV41_ENABLE_PREFILL_SHARE` (default off; `Dsv41Config.enable_prefill_share`). Plan and replay:
`docs/superpowers/plans/2026-09-25-dsv41-prefill-eviction.md`.

**Option chosen: a bounded prefill share, not staging.** A replay of the C++ tier's victim rule over three route
captures (`analysis/dsv41-drive/prefill-evict/ram_replay.py`; its base arm gives 11.33 RAM misses per token against
11.26 measured) decided it:
- **(a) Staging without admission is worst.** A prompt's experts are what its first decode tokens route to, so not
  admitting them raised decode steps 0–4 from 19.5 to 49.8 RAM misses per token on varied24. It would also need an
  ~850 MB pinned staging set outside the service's slab tables.
- **Most of the early-decode excess is cold start, not eviction.** Even a tier where prefill evicts nothing leaves
  steps 0–4 at 16.9 misses per token, against 9.3 at steps 70+.
- **(b) A 64-row share** (one gather chunk, `EXL3_MAX_GATHER_ROWS`) was never worse than base for decode, and on
  the long-prompt soaks it cut decode RAM misses by 6–7%. The cost is 28–32% more prefill row reads, because a long
  prompt's chunks reuse each other's experts. Cold admission (stamp 0) saved slightly more decode misses but cost
  ~40% more prefill reads.

**How it works.**
- `RamTier` marks the rows a prefill admits as prefill-owned.
- Once a layer holds 64 of them, and no slot is free, a prefill admission evicts the LRU owned row instead of one of
  decode's. The victim is bound by every exclusion of `take_slot_locked`: READY, unleased, not hot, not protected,
  not filling.
- Decode ends a row's ownership by using it: a served demand, an unarmed touch, a prefetch lease, or `set_hot`.
- A pre-forward observer sets the share for prefill forwards (`is_extend_without_speculative`) and clears it for
  every other forward. It needs the expert-distribution recorder, and startup refuses the no-op one.
- **With `SGLANG_DSV41_ENABLE_PREFILL_FILLS` (§27.6).** `fill_begin` claims through the same rule. A layer's
  prefetch stops once the share is full; the chunk admissions after it evict the rows earlier chunks have gathered.

**Arms.**
- A (off) and B (on) ran at `b4e660e996` (before the rebase onto §27.6). Both were node-mode traced on port 30021,
  once each, in that order.
- C ran after the rebase, at `48186189c1`, with both flags on. It was untraced.
- Every arm returned rc 0. The session with 103 output tokens carries the comparison.

| | A (off) | B (share) | C (share + fills) | §27.6 B (fills) |
|---|---:|---:|---:|---:|
| S per step, decode steps 0–4 / 5–14 / 70+ (ms, traced) | 63.6 / 39.5 / 35.4 | 60.5 / 37.0 / 32.6 | | |
| step wall, steps 0–4 / 5–14 / 70+ (ms, traced) | 148.3 / 110.2 / 106.3 | 145.5 / 109.8 / 105.9 | | |
| ms/token, session 1 (pooled) | 114.2 (115.8) | 113.6 (115.4) | 110.1 (111.8) | 110.9 (112.5) |
| TTFT, session 0 / 1 (s) | 22.87 / 19.47 | 22.93 / **20.63** | 12.54 / 11.88 | 12.05 / 11.14 |
| decode NVMe rows (service `rows_read`, whole server) | 4,358 | 4,154 | 4,465 | 4,453 |
| outputs vs A | | 2 of 2 byte-identical | 2 of 2 byte-identical | 2 of 2 byte-identical |

- **Decode gains are small.** B cut S by 3 ms per step at decode steps 0–4 and by 2.8 ms at steps 70+, and the
  service read 4.7% fewer rows. Step wall moved by 0.4–2.8 ms. Each arm is one session, so all of this is
  directional.
- **TTFT got worse:** +1.2 s on session 1 with the share alone. With fills on, the share cost +0.5 / +0.7 s,
  because chunks after the first are no longer prefetched.
- **So the flag stays off.** The replay and the arms agree: after a short prefill, decode's early misses are mostly
  a cold start that no admission policy removes. Only a predictor could remove them (Track B, §25.4).

**Tests.**
- `test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py` has 16 tests. Among them: decode's rows survive
  a prefill that holds its share, and prefill fills claim through the share. Three C++ mutants were each caught.
- The service tests are at the end of `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`.

**Evidence** (divix01):
- Arm runs: `cc-expert-prediction/dsv41-baseline/servers/pevict-{A,B,C}/`.
- Traces: `/mnt/nvme1/dsv41-nsys/pevict-{A,B}-*.{nsys-rep,sqlite}`.
- Replays: `/mnt/nvme1/prefill-evict/replay3_*.json`.
- Command: `analysis/dsv41-drive/pcie-trace/s_decay.py <sqlite>`, which now finds the last session's decode itself.

### 27.10 The production recipe traced: prefill fills and cast fusion on, with NVMe load (2026-09-25)

**Recipe.** `003d82fb77` turns on `SGLANG_DSV41_ENABLE_PREFILL_FILLS` and `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION` in
`arm_env.base_env()`, which is both the arms' base and production's launch env.

**Arm.** `prod-flags-node`: node-mode trace plus the root PCIe session, port 30021, once, rc 0, outputs normal.
- Driver: `analysis/dsv41-drive/nvme-load/drive_traced_arm.sh`.
- Session 1 (103 tokens): TTFT 13.63 / 12.90 s and 113.3 ms/token, traced; §27.6's untraced fills arm had
  12.05 / 11.14 s.
- The comparison below is against `pevict-A` (§27.9): the same recipe with both flags off, traced the same way.
  `analysis/dsv41-drive/pcie-trace/compare_arms.py` measures the last session in both.

**Prefill (260 tokens, last session).**

| | Flags off | Both on |
|---|---:|---:|
| Wall | 19,454 ms | 12,882 ms |
| GPU busy (kernels and copies) | 6,588 ms | 6,571 ms |
| of which `_gather_host_rows_kernel` (726 launches) | 6,083 ms | 6,068 ms |
| GPU idle | 66% | 49% |
| D2H readbacks ≤256 B / host time blocked in them | 8,380 / 6,114 ms | 8,174 / 6,102 ms |
| PCIe RX mean | 13.7% | 20.6% (~5.9 GB/s, ~75 GB in all) |

- **The gather is capped by the link.** A 260-token prompt touches ~75 GB of expert rows, and the SM gather moves them
  over PCIe Gen3 in ~6.1 s. Only gathering fewer rows shortens it.
- **The other ~6.3 s is the GPU waiting for the host.** `prefill_idle_vs_nvme.py` splits the idle time by NVMe
  activity in 100 ms bins:
  - 1.4 s with a mirror ≥50% busy (fill waits);
  - 2.7 s with 10–50% busy;
  - 2.2 s with the mirrors idle, i.e. pure host overhead: ~141,600 eager launches, Python, the syncs.
- **The host and GPU take turns.** The host is blocked ~6.1 s in the readbacks waiting for the GPU, and the GPU is idle
  ~6.3 s waiting for the host; together they are nearly the whole prefill. Removing the syncs (§27.4 item 7) would
  let the next chunk's launches and reads overlap the current gather. The floor is then the ~6.1 s link time.

**Decode.**

| Steps | Flags off: step / S | Both on: step / S |
|---|---:|---:|
| 0 | 198 / 103 ms | 201 / 108 ms |
| 1–4 | 125 / 54 ms | 126 / 56 ms |
| 5–14 | 106 / 41 ms | 105 / 41 ms |
| 15–102 | 111.6 / 37.3 ms | 111.2 / 37.2 ms |

- **Cast fusion removes kernels, not time.** Kernels per step 2,788 → 2,430 and sub-3 µs kernels 1,851 → 1,493, but
  GPU busy per step is 108.6 ms in both arms.
- **The cold start after a prefill is unchanged by either flag:** ~150 ms of extra S per request, mostly in step 0
  (§27.9). In steady state S is still a third of every step.
- PCIe RX over steps 15+: 34.1% / 34.4%.

**NVMe load.** `analysis/dsv41-drive/nvme-load/` samples `/proc/diskstats` every 100 ms (plus `iostat -x` at 1 s)
alongside an arm, with wall and monotonic clocks for alignment with the trace. The two mirrors split every read in
half: nvme0n1 (`/mnt/nvme0`) at 416 kB per request, nvme3n1 (`/mnt/nvme4`, SPCC, 256 KB maximum transfer) at 212 kB.
nvme2n1 serves only 4 kB Engram lookups.

| Session 1 | nvme0n1: mean / while busy / busy | nvme3n1: mean / while busy / busy |
|---|---:|---:|
| Prefill (12.9 s) | 873 MB/s / 3.84 GB/s / 23% | 872 MB/s / 3.80 GB/s / 23% |
| Decode (11.9 s) | 767 MB/s / 3.33 GB/s / 23% | 765 MB/s / 2.67 GB/s / 29% |

- **The mirrors saturate in bursts.** Whenever a mirror is reading, it runs at the Gen3 x4 link rate (~3.5–3.9 GB/s).
  The low averages mean no read is known yet, not a shallow queue: a layer's misses run both drives flat out, and a
  13.3 MB row takes ~2 ms at best across the two. Deeper queues would not shorten S. The levers are:
  - fewer NVMe misses;
  - more read bandwidth, e.g. a third mirror;
  - reading earlier, which needs a predictor (Track B, §25.4).
- **Another tenant writes to the SPCC mirror.** Bursts of 200–576 MB/s of writes hit nvme3n1 during decode, with queue
  depth up to ~920, and its busy read rate is the lowest (2.67 GB/s). The bursts continued after the arm ended.
  The likely writer is a Ray cluster (running since 2026-09-25 00:34) whose temp and spill directory is
  `/mnt/nvme4/ray_tmp`; not proven. The 64 GB swapfile on nvme4 was moved to `/mnt/nvme1` on 2026-09-25.

**Evidence** (divix01):
- Run: `cc-expert-prediction/dsv41-baseline/servers/prod-flags-node/run-20260925-211029/`.
- Traces: `/mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044{.nsys-rep,-pcie.nsys-rep,.sqlite}`; the PCIe
  exports are under `/mnt/nvme1/prod-flags/`.
- NVMe samples: `/mnt/nvme1/prod-flags/{nvme_diskstats.jsonl,iostat.log}`.
- Timed sessions, wall clock: session 0 prefill 21:14:10–21:14:23.3; session 1 prefill 21:14:25.5–21:14:38.4, decode to
  21:14:50.4.

### 27.11 Prefill route plan: the host runs ahead of the gather (result, 2026-09-25)

Item 7 of §27.4, steps 1, 2 and 4 of `MOE_PREFILL_OPT.md`. Plan: `docs/superpowers/plans/2026-09-25-dsv41-prefill-route-plan.md`.
Flag: **`SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN`** (`EnvBool`, default off; `Dsv41Config.enable_prefill_route_plan`).

**Attribution** (`prod-flags-node`, 260-token prefill, 12,882 ms; scripts in `analysis/dsv41-drive/pcie-trace/`:
`prefill_sync_sites.py` keys each readback by the kernels before it, `chunk_host_time.py` times each chunk):

| Where the host blocked | Readbacks | Host blocked |
|---|---:|---:|
| `chunk.tolist()` right after a chunk's gather was queued (`exl3.py`, `_apply_streamed`) | 115 | 5,970 ms (median 57.9 ms) |
| `row_of_source.tolist()` right after it | 116 | ~2 ms |
| Per-expert `torch.where` count (`exl3_moe_accumulate`) | ~6,970 | 72 ms |
| Everything else | ~1,050 | ~60 ms |

- The ~8,000 small syncs were cheap: they ran with the GPU queue empty.
- All of the cost was the host waiting for each chunk's ~58 ms gather before launching that chunk's compute. After
  each wait, the host spent a median 38 ms (same layer) or 63 ms (next layer, with attention) before the next gather.
- The host and GPU strictly took turns, ~6.0 s each. The §27.10 reading ("8,174 small readbacks") was right on the
  count and wrong on the cause: one readback per chunk carried almost all of it.

**What changed.** With the flag on, `_apply_streamed` reads the layer's `topk_ids` to the host once, before any of its
gathers is queued.
- `Exl3RoutePlan` groups the routes by expert: `torch.where`'s own (token, slot) indices, row-major within each
  expert.
- `ExpertStreamer.iter_gather_experts_host` yields each chunk's expert ids and `row_of_source` as host lists. The hot
  slots and hit mask come back in the one readback `_gather_cached` already made before the gather, in place of its
  hit-count `.item()`.
- `exl3_moe_accumulate_planned` runs the same per-expert body (`_accumulate_expert`, shared with the flag-off loop)
  with no readback.
- Stream order still makes chunk k+1's gather wait for chunk k's consumers.
- With the flag off, the ops and their order are unchanged.
- Commits `d1031bcb2a`..`469aafec49` on master.

**Tests** (divix01, private worktree; commands and logs in `/mnt/nvme1/prefill-opt/`):
- CPU: `test_exl3_route_plan.py`, `test_exl3_moe_stream_mode.py` (parity over the flag with a fake and a real
  pinned-tier streamer, three chunks, eviction, all routes dropped, capture), plus `test_exl3_ops_cpu.py`,
  `test_exl3_moe_method.py`, `test_expert_gather_experts.py` and `test_dsv41_config.py`: 70 passed.
- GPU, real EXL3 kernels, under `cc-gpu.lock`: `test/manual/dsv41/test_exl3_stream_apply_gpu.py` (bitwise over the
  flag; 80 experts in two 64-expert chunks with a repeated expert and dropped routes) and `test_expert_plugins_cuda.py`
  (host lists equal the device gather through mixed, all-hot and all-cold chunks): 24 passed, 0 skipped.
- **No-sync proof:** `test_planned_chunk_body_never_syncs` runs the planned loop under
  `torch.cuda.set_sync_debug_mode("error")`. Its control, `test_the_where_loop_does_sync`, shows the mode catches the
  old loop.
- Registered suite, `pytest -q -p no:randomly test/registered/unit/kernels` under the GPU lock:
  - base `4831d251bc`: 1726 passed, 1 skipped;
  - tip `e6a01c33ac`: 1725 passed, 1 skipped, 1 failed. The failure,
    `test_exl3_ram_miss_pack_workers::test_no_worker_thread_exists_unless_asked_for_and_close_joins_them`, counts
    threads process-wide. It passed 5 of 5 alone and in its whole file (549 passed), and nothing here touches pack
    workers.
- Pre-existing breakage fixed on the way: three stream-mode tests failed on master because their fake streamers
  predated `prefill_fills`.

**Mutants** (each reverted; the suite was green again after):

| Mutant | Caught by |
|---|---|
| Plan order reversed within an expert | `test_plan_matches_torch_where_row_major`, `test_planned_accumulate_is_bitwise_...` |
| `host_row_of_source` puts hits first | CPU row test; GPU streamed-apply and many-experts parity; the CUDA host-rows test |
| The planned loop walks a chunk's experts in reverse | `test_streamed_apply_accumulates_in_ascending_expert_order` |
| `hot_out` slots rotated by one | GPU all-hot streamed-apply; the CUDA host-rows test |

The reversed-order mutant first escaped every parity test, CPU and GPU: the bf16 output rounds away a change in fp32
accumulation order. The ascending-order test makes experts add 2^24, 1 and -2^24, which give 0 only in ascending
order. The flag-off loop had the same blind spot, and the test covers both.

**Arms** (A = production recipe, B = + the flag; A then B, once each, port 30021, commit `469aafec49`):

| | A | B |
|---|---:|---:|
| TTFT, session 0 (260-token prompt) | 12.61 s | **9.91 s** |
| TTFT, session 1 | 11.48 s | **9.16 s** |
| Decode, pooled client ms/token | 116.4 | 114.3 |
| Output | | identical to A (both sessions) |

**Trace** (node-mode traced B, last timed session's prefill, against `prod-flags-node`):

| | Before (§27.10) | Route plan |
|---|---:|---:|
| Prefill wall | 12,882 ms | **9,296 ms** |
| GPU busy (kernels + copies) | 6,571 ms | 6,562 ms |
| of which the gather | 6,068 ms | 6,095 ms |
| GPU idle | 49% | **29%** |
| D2H readbacks / host blocked in them | 8,256 / 6,103 ms | 1,385 / 2,828 ms |
| Eager kernels | 141,593 | 108,452 |
| Decode steps 15+, ms / GPU busy per step | 111.2 / 108.6 | 111.3 / 108.7 |

**Review fix** (`7d05dc2bc1`).
- **The problem:** the flag-off path's `chunk.tolist()` also guaranteed that a layer's pinned-tier host use
  (`prefill_fills`) ended only after its last gather had read the slabs. Running ahead, the host could end the host use,
  and so resume the RAM-miss service thread, with that copy still reading. This is the hazard of §18.6's `host_use`
  contract. There is no known trigger today: the thread's current work is ordered on the same stream.
- **The fix:** each chunk now records an event after its gather and waits on it after queuing the chunk's compute.
  The launches still overlap the gather.
- **Test:** `test_route_plan_leaves_the_host_use_only_after_the_last_gather_lands` failed before the fix, 25/25 GPU
  tests pass after.
- **Cost:** none. Arm B2 at the fix measured TTFT **9.91 / 9.21 s**, 114.0 ms/token, output identical to A.

- **Where the host waits now:** one ~1 KB readback per chunk, `_gather_cached`'s pre-gather lookup, blocked 2.8 s in
  total (median 35.5 ms). It waits for the previous chunk's gather, which the host has already queued compute
  behind: this is the GPU being the bottleneck, as intended.
- Within a layer, the host then needs a median 2.2 ms (914 ms total) to launch the next gather.
- **What is left above the link floor:** ~2.7 s of GPU idle. The 0.9 s of within-layer host gaps accounts for part of
  it. The remaining ~1.8 s is inferred to sit at the 40 layer boundaries, not measured per site. At a boundary, the
  next layer's attention, route readback, fill start and first-chunk fill wait all run with the gather queue empty.
- The PCIe metrics session failed to stop (`nsys stop failed`), so this arm has no PCIe RX figures.

**Where the remaining GPU idle goes, by site** (traced arm at `035ce226d3`, route plan plus the review fix; 9,280 ms
prefill; TTFT 10.04 / 9.30 s traced):

| Host state while the GPU idles | GPU idle |
|---|---:|
| Fill wait before a layer's **first** chunk's gather (40 chunks, median 26.9 ms, max 97.5 ms) | **1,296 ms** |
| Fill wait before a later chunk's gather (13 of 81 chunks, median 51 ms, max 106 ms) | 637 ms |
| Blocking CUDA calls | 30 ms |
| Everything else: Python and launches | 765 ms |
| Total | 2,728 ms (29%) |

- **Method.** A fill wait is a CUDA-free stretch of at least 1 ms that ends within 1 ms of a chunk's first gather
  launch. OS-runtime tracing shows the thread there in back-to-back ~70 µs `nanosleep`s, which is `RamTier::fill_wait`
  polling every 20 µs from `_await_fills`. Scripts: `analysis/dsv41-drive/pcie-trace/idle_by_host_state.py` and
  `boundary_timeline.py`.
- **Correction.** The "~1.8 s at layer boundaries" read above was mostly right about place and wrong about cause. It
  is not attention or syncs but NVMe: a layer's fills can start only once its routing is known, and `gather_rows`
  copies nothing of a chunk until every row of it has landed.
- The fill reader works in claim order (ascending, batches of 8 rows, progress every 200 µs), so a first chunk
  waits for its own misses, not the layer's.

**Next** (§27.4 item 7 remainder):
- **Split each chunk's gather around its fills.** Copy the chunk's rows already in the pinned tier first, then wait
  for the fills, then copy the filled rows. The same bytes land in the same staging rows, so outputs stay bitwise
  identical. This needs:
  - an output-row index on `_gather_host_rows_kernel`;
  - a second launch for chunks with fills.

  The pre-gather `.item()` has already drained the stream, so the index copies cost no wait. At ~1.08 ms a row
  gathered against ~1.75 ms a row read from NVMe, a first chunk's ~49 resident rows cover its ~15 fills in most
  layers. Estimated gain: most of the 1.3 s, and part of the 0.64 s.
- **Starting a layer's fills before its routing is known** needs a predictor (Track B).
- **Grouped expert compute** does not pay yet. The host now waits on the GPU within a layer, so fewer launches would
  not shorten it.
- **Production** still runs without the flag. Adding it to `arm_env.base_env()` is a separate, asked-for change.

**Evidence** (divix01):
- Runs: `cc-expert-prediction/dsv41-baseline/servers/route-plan-{A,B,B-node}/run-20260925-23*/`.
- Trace: `/mnt/nvme1/dsv41-nsys/route-plan-B-node-20260925-233535{.nsys-rep,.sqlite}`.
- Analyses: `/mnt/nvme1/prefill-opt/{compare.txt,sites-after.txt,chunks-after.txt,arm-table.txt,mutants.txt}`.

### 27.12 Split fill gather: a chunk's resident rows go first (result, 2026-09-26)

The first item of §27.11's Next. Plan: `docs/superpowers/plans/2026-09-26-dsv41-split-fill-gather.md`.
Flag: **`SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER`** (`EnvBool`, default off; `Dsv41Config.enable_prefill_split_gather`).

**What changed.** With the flag on, `ExpertPinnedHostCache.gather_rows` splits a chunk that has rows still filling:
- It copies the chunk's rows already resident in the pinned tier first, through `copy_rows(..., rows=...)`.
  `_gather_host_rows_to_kernel` is `_gather_host_rows_kernel` with an output-row index.
- It then waits for the chunk's fills (`_await_fills`), and copies the filled rows.
- The same bytes land in the same staging rows, so outputs are bitwise identical.
- A chunk with no fills, or with only filling rows, takes the unsplit path.
- The split is used only when every output is on the CPU, or the source and outputs are contiguous (`_splits`). A split
  copy to named rows refuses outputs the unsplit path would send to its fallback, rather than silently taking it.
- Commits `0bb3ffbe55`..`5fe531c64b` on master; each implementation commit follows its red test.

**Tests** (divix01, private worktree `wt-route-plan`; runners `/mnt/nvme1/prefill-opt/{rpt,gpt}.sh`):
- CPU, `test_expert_pinned_row_fills.py`: 14 passed. Resident rows are copied before the fill wait. Every chunk's rows
  land where the unsplit gather puts them (chunks [0,2,5] and [7,5,3]). A chunk of only filling rows waits, then
  copies once. A failed fill still raises. The overflow test runs with and without the split.
- GPU, under `cc-gpu.lock`: `test_expert_plugins_cuda.py`, `test_expert_pinned_row_fills.py` and
  `test/manual/dsv41/test_exl3_stream_apply_gpu.py`: 41 passed at `5fe531c64b`, both with the flag unset and with
  `SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER=1` in the environment. The CUDA tests cover:
  - a copy to named rows leaves the other rows untouched;
  - a split copy refuses outputs the fallback would take.
- Registered suite, `pytest -q -p no:randomly test/registered/unit/kernels` under the GPU lock: 1726 passed, 1 skipped
  at both base `eb70bd3413` and tip `5fe531c64b`.
- Focused set (`test_exl3_ops_cpu.py`, `test_exl3_moe_stream_mode.py`, `test_exl3_moe_method.py`,
  `test_expert_gather_experts.py`, `test_expert_plugins_cuda.py`, `test_dsv41_config.py`): 70 → 72 passed. The two new
  tests are the difference.

**Mutants** (`/mnt/nvme1/prefill-opt/mutants_split.py`; each reverted, and the files were green again after):

| Mutant | Caught by |
|---|---|
| Fill wait moved before the resident copy | `test_split_gather_copies_resident_rows_before_waiting_for_the_fills` |
| Ready and filling rows swapped | the same, and `test_split_gather_places_every_chunks_rows_like_the_unsplit_gather` |
| The indexed kernel stores to its program row, not the named row | `test_copy_rows_to_named_rows_on_cuda_leaves_the_rest` |
| `_filling_positions` always empty | `test_split_gather_copies_resident_rows_before_waiting_for_the_fills` |
| `_splits` always true | `test_a_split_copy_refuses_outputs_the_fallback_would_take` |

The `_splits` mutant first survived every test. The CUDA refusal test (`9a26ec89a5`) was added for it.

**Arms** (A = production recipe with the route plan, B = + the flag; A then B, once each, port 30021, `5fe531c64b`):

| | A | B | B, traced |
|---|---:|---:|---:|
| TTFT, session 0 (260-token prompt) | 9.94 s | **8.59 s** | 8.78 s |
| TTFT, session 1 | 9.19 s | **8.20 s** | 8.32 s |
| Decode, pooled client ms/token | 114.1 | 114.1 | 116.6 |
| Output | | identical to A | identical to A |

**Trace** (node-mode traced B, last session's prefill, against §27.11's `route-plan-fix-node`):

| | Route plan (§27.11) | + split |
|---|---:|---:|
| Prefill wall | 9,280 ms | **8,300 ms** |
| GPU busy (kernels + copies) | 6,552 ms | 6,543 ms |
| of which the gather | 6,087 ms (726 launches) | 6,076 ms (1,236 indexed + 108 plain) |
| GPU idle | 2,728 ms (29%) | **1,756 ms (21%)** |
| Decode steps 15+, ms / GPU busy per step | 110.6 / 108.1 | 110.5 / 107.9 |

| Host state while the GPU idles | Route plan | + split |
|---|---:|---:|
| Fill wait before a layer's first chunk's gather | 1,296 ms (40 chunks, median 26.9 ms) | **286 ms** (5 chunks, median 74.1 ms) |
| Fill wait before a later chunk's gather | 637 ms (13 chunks, median 51 ms) | 714 ms (12 chunks, median 60.4 ms) |
| Blocking CUDA calls | 30 ms | 28 ms |
| Everything else: Python and launches | 765 ms | 729 ms |

- The first-chunk wait fell by 1.0 s, as §27.11 estimated. Only 5 of 40 first chunks still wait: those whose resident
  rows gather faster than their fills land.
- The later-chunk waits did not move. **Correction (§27.13):** they are not fill waits at all. Each is a demand read
  (`ensure_rows` → `_fill_rows`, one `pthread_create` + `pthread_join`) for a chunk the layer's fill did not claim.
  `idle_by_host_state.py` matches only the plain gather kernel, so it misreads split traces;
  `analysis/dsv41-drive/split-gather/fill_waits.py` replaces it for them.
- The gather's own time is unchanged, which is the link floor. The extra launches add ~0.5% of eager kernels
  (108,450 → 110,508).

**The traced arm needs 4 GiB of the pinned tier moved off node 0.** Its first two attempts refused to start:
- `check_capacity` found node 0 ~0.5 GB short: 54.7 GB free plus 10.3 GB of page cache, against 60 GiB plus 4 GiB of
  headroom.
- A 1 s sample of node 0 during a traced start showed anonymous memory growing 1 → 17.4 GB before the check. The
  scheduler held 9.0 GB, the main process 4.9 GB and the detokenizer 4.8 GB, all under `nsys launch
  --trace=cuda,nvtx,osrt`.
- Untraced arms start: the recipe's 60 GiB on node 0 is sized to leave exactly 4 GiB spare.
- The third attempt ran with `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:45056`: the same 100 GiB and rows, 4 GiB of
  them on node 1. §25.4 measured the same H2D rate from either node.
- `analysis/dsv41-drive/split-gather/drive_arms.sh` now passes that placement to its traced arm.

**What remains** (§27.4 item 7):
- **Later-chunk waits, 714 ms:** demand reads, not fills (§27.13).
- **Python and launches, 729 ms.** Not yet attributed.
- **Starting a layer's fills before its routing is known** needs a predictor (Track B, §25.4).
- **Production:** the flag is in `arm_env.base_env()` since `e0bd452bf6`.

**Evidence** (divix01):
- Runs: `cc-expert-prediction/dsv41-baseline/servers/split-gather-{A,B}/run-20260926-01*/` and
  `split-gather-B-node/run-20260926-014232/`.
- Trace: `/mnt/nvme1/dsv41-nsys/split-gather-B-node-20260926-014247{.nsys-rep,.sqlite}`.
- Analyses: `/mnt/nvme1/prefill-opt/{idle-split.txt,compare-split.txt,arm-table-split.txt,mutants-split.txt,
  mutants-split-4.txt,node0-sample.log,suite-split{tip,base}.log}`.

### 27.13 Split gather: landed rows first; the later-chunk waits are demand reads (result, 2026-09-26)

Review minors 4 and 5 of §27.12. Same flag, `SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER`; no new knob.

**What changed** (`a8e0fb6bc2` red tests, `ec9dbda80f` implementation, `96ef745220` and `35083933b5` analysis):
- `RamTier::fill_landed()` returns how many claimed rows have landed, without blocking. It is exposed through the
  binding, `NativePinnedSlotTable` and the `PinnedRowFills` protocol.
- A chunk copies its resident and already-landed rows first. It then waits for its filling rows eight at a time (the
  reader's batch) and copies each batch as it lands.
- One host-to-device index tensor per chunk (`ready + filling`), sliced for each copy. `slots[rows]` is computed once
  per copy, not per tensor.
- Failures still surface at the wait: the real `fill_wait` binding raises on failure or timeout.
- **Pre-existing, unchanged:** the landed count is published during a read without the `_mm_sfence()` the end of
  the fill uses. `fill_wait` has always relied on this; `fill_landed` inherits it.

**Tests** (divix01, private worktree; logs in `/mnt/nvme1/split-landed/`):
- CPU, `test_expert_pinned_row_fills.py` + `test_exl3_ram_miss_prefill_fills.py`: 5 failed / 30 passed at the red
  commit, 35 passed after.
- GPU: `test_expert_plugins_cuda.py`, `test_expert_pinned_row_fills.py`, `test/manual/dsv41/test_exl3_stream_apply_gpu.py`:
  47 passed, flag unset and set. The CUDA one-index-copy test was red first (waits `[8, 16, 18]` against `[18]`).
- Registered suite, `pytest -q -p no:randomly test/registered/unit/kernels` under the GPU lock: base `ca74e5391a`
  1725 passed, 1 skipped, 2 failed (the two red kernel tests); tip 1727 passed, 1 skipped.

**Mutants** (all caught; worktree clean after; baseline green again, CPU 35 and GPU 15):

| Mutant | Caught by |
|---|---|
| Landed rows count as filling (`fill_landed` ignored) | `test_a_row_that_landed_in_an_earlier_chunks_wait_is_copied_before_the_next_wait`, `test_rows_the_reader_landed_before_the_chunk_are_copied_before_its_wait` |
| Wait for every filling row before the first filled copy | `test_a_chunks_filling_rows_are_copied_batch_by_batch_as_they_land` |
| Copy the whole filling rest after the first batch wait | the same |
| Filling rows left in chunk order, not claim order | `test_split_gather_places_every_chunks_rows_like_the_unsplit_gather` |
| Filling rows copied from a second host-to-device index | `test_a_split_chunk_copies_its_row_index_to_the_device_once` |
| `fill_landed` reports nothing landed | `test_fill_landed_reports_the_landed_prefix_without_blocking` |

**Arms** (A = base `ca74e5391a`, B = tip `ec9dbda80f`, same recipe, once each):

| | A | B | B, traced |
|---|---:|---:|---:|
| TTFT, session 0 | 8.61 s | 8.52 s | 8.69 s |
| TTFT, session 1 | 8.21 s | 8.23 s | 8.43 s |
| Decode, pooled client ms/token | 113.9 | 114.2 | 116.8 |
| Output | | identical to A | identical to A |

**Why nothing moved: the 714 ms was never a fill wait.** `fill_waits.py` classifies every wait before a gather copy,
indexed copies included, and joins the host thread's OS-runtime calls:

| GPU idle by host OS call | Split (§27.12 trace) | + landed first |
|---|---:|---:|
| `pthread_join` (demand reads: `ensure_rows` → `_fill_rows`) | 989 ms | 990 ms |
| No OS call (Python, launches) | 500 ms | 634 ms |
| `pthread_cond_wait` | 216 ms | 217 ms |
| `nanosleep` (`fill_wait` polling) | 47 ms | 13 ms |
| Total GPU idle | 1,756 ms | 1,859 ms |
| Prefill wall | 8,300 ms | 8,413 ms |

- **Split trace, by wait** (121 chunks, 40 first-of-layer):
  - first chunks: 3 unsplit waits cost 265 ms of GPU idle; 37 split waits total 985 ms of waiting but only 43 ms of
    GPU idle, which the split hides;
  - later chunks: 14 unsplit waits, 735 ms, all GPU idle.
- The later-chunk waits are whole chunks (41–60 rows each) read on demand. They come from layers cold in the pinned
  tier: the layer's fill stops claiming at the prefill share, so each later chunk's misses are read in one blocking
  `_fill_rows` with nothing queued on the GPU. Examples: layer 0's chunks 1–4, 6–8 and chunks 50–52, 63–66.
- The review's "count already waited" fix could not help. Chunks and claims both run in ascending expert order, so a
  later chunk never holds rows an earlier wait covered.
- The after-trace's 454 ms of later-chunk waits (against 735) is run-to-run: the same cold chunks take the same time,
  and that run had 3 fewer of them. The rise in Python and launches is unexplained in one traced run.

**Next prefill lever: ~1 s of demand-read joins.** Either claim past the prefill share, or start chunk k+1's demand
reads while chunk k computes. Per-layer pinned room limits both.

**Evidence** (divix01):
- Trace: `/mnt/nvme1/dsv41-nsys/split-landed-B-node-20260926-025931.sqlite`.
- Analyses: `/mnt/nvme1/split-landed/{fill-waits-before.txt,fill-waits-after.txt,idle-osrt.txt,compare.txt,
  arm-table.txt,mutants-landed.txt,suite-base.log,suite-tip.log}`.

### 27.14 Decode link losses, async promotions and pinned-tier prefetch (studies, 2026-09-26)

**Link losses in decode** (trace `prod-flags-node`, steady steps 22–109; `divix01:/mnt/nvme1/prefill-opt/link-idle/`):

| Link state | ms/step | Link rate |
|---|---:|---:|
| Copying, GPU in CW | 55.0 | 13.62 GB/s |
| Copying, GPU in S | 24.2 | 12.79 GB/s |
| No copy, GPU in S (NVMe waits) | 16.9 | 7.9 GB/s |
| No copy, compute | 14.6 (+2.6 idle, +0.65 other waits) | 0.9 GB/s |

- **Per copy on stream 141**, 50,448 copies:
  - 16,816 large ones (8.8 and 4.4 MB) carry 99.7% of the bytes; 73% of them run at 12–13.75 GB/s.
  - The other 27% run at 10.8–12 GB/s while `exl3_ram_miss_lease_stream_kernel` runs. The link is then ~96% full
    (13.2 GB/s, 2.5 GB/s of it the kernel's SM reads), so the true loss is ~0.8 ms/step.
  - 33,632 small ones (4.6–20 KB, four per row) run at 1.5–5 GB/s.
- **Small segments:** 1.06 ms/step of copy time plus 0.62 ms/step of gaps inside rows, an upper bound of
  1.44 ms/token. Every layer's last copy is a small one and CW ends ~4.5 µs after it, so 0.66 ms/token is on the
  critical path.
- **Idle gaps:** gaps of 1 ms or more total 20.0 ms/step, all between layers: 69% in S, 21% in compute.
  S totals 41.1 ms/step (steady). By layer: L0 2.70 (median 1.33), L19 2.11, L39 2.05, L23 1.96, L13 1.67; the others
  are rare stalls with medians of 0.08–0.09 ms.

**Merging a row's copies** (investigation; `divix01:/mnt/nvme1/coalesce/`):
- The six copies per row are per tensor, not deliberate chunking. `CopyEngine::issue`
  (`exl3_ram_miss_host.cpp:3398-3405`) makes one `cuMemcpyAsync` per `ExpertRowSegments` entry, and both the pinned
  slabs and the hot cache store rows name-major.
- `cuMemcpyBatchAsync` gains nothing at 1–3 rows per layer.
- One copy per row saves ~7–8 µs/row alone (~0.55 ms/token) but needs a slab layout redesign across the C++ reader,
  two CUDA kernels and several Python consumers.
- **Built: `SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES`, in the recipe since `c515e0c153`.** CW reads each
  RAM-hit row's four small tensors (44.5 KB) from the pinned slab with SM loads, then publishes a device-written
  `SmAck`. The copy engine sends only the two large tensors. A copy-engine lease is released only after both its DMA
  (CopyDone) and its `SmAck`. Protocol: `LEASE_PROTOCOL.md` §7.6. Commits `62527a1444`..`150d7212c7`.
  - Tests: a 60-step byte-for-byte parity run against the six-copy path, and a sentinel test that overwrites a slab
    row the instant its lease drops. After review: CW commits only the lanes its SM phase read (else Identity), 16 B
    alignment is enforced, and there are tests for late reads (threads 32+ delayed 100 ms) and a failed request.
    Mutants caught: release on DMA alone, stale SmAck accepted, no barrier after the reads, SmAck above the reads,
    SmAck only when something was read, commit without the mask check.
  - Registered suite 1727 → 1733 passed. CPU set 68 passed, GPU set 48 passed plus the known
    `other_streams[kernel]` flake (2 of 8 at base too), which does not use this path.
  - Traced: copies per row 6 → 2, copies per step 450.7 → 150.2, copy busy 79.1 → 78.1 ms/step, CW 55.2 → 54.7 ms/step,
    median step 111.0 → 109.9 ms. Untraced A/B, outputs identical: 111.5 → 111.0 ms/token, within noise.
  - Deferred: the SM mask comes from `streamer._graph_sources`, not the copy table's own names; the copy thread spins
    on jobs awaiting an SmAck that a dead chain will never write, until the drain deadline.

**Async hot-cache promotions: no-go.**
- The synchronous path the EXL3 gate named (`ExpertHotCache._load_reserved_in_chunks`) never runs in the recipe.
  `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` with insert-on-miss stage 2 ranks victims in-graph, and the gather writes missed
  rows straight into them.
- The whole cost is ~0.35–0.4 ms/step of in-graph kernels (`direct_commit_gather_kernel` 235 µs,
  `direct_gather_destinations_kernel` 81 µs), with no host syncs. These stay on the critical path whenever promotion
  runs.
- §18.6's old ~3.6 ms/token was removed by the in-graph path. `6512709f1a` corrects the gate's refusal message.

**NVMe → pinned-tier prefetch: no-go.** Full write-up: `NVME_PINNED_PREFETCH_HANDOFF.md`; code:
`analysis/dsv41-drive/prefetch-replay/`.
- A whole next layer is 384 × 13.3 MB = 5.11 GB, 0.68–0.85 s at the mirrors' 6–7.5 GB/s, against ~2.75 ms per layer:
  250–310× too slow.
- A missed row still crosses the link. Prefetch removes only the NVMe wait beyond the layer's link-bound time,
  ~6.2 ms/token in a replay calibrated to 11.33 RAM misses/token (11.265 measured).
- Best realistic predictor, layer T+1's gate on layer T's input: precision 0.40 on not-in-RAM rows, **~2 ms/token**.
- Perfect prediction saves 6.2 ms/token. Next-token and frequency predictors, and anything for layer 0, gain nothing.
- Demand reads must outrank speculative ones; FIFO turns most arms negative.

### 27.15 Decode RAM-miss frontend: sized, no-go; resident-first after real copies (study, 2026-09-26)

Plan: `docs/superpowers/plans/2026-09-26-dsv41-ram-miss-frontend.md`. The question was whether the stage-1 frontend
(`post -> W1 -> C1 -> A1` before `S -> A2 -> CW -> F`) still costs decode time now that the copy engine carries the RAM
hits, and whether resident-first compute pays once real transfers are in the picture.

**The bound** (`analysis/dsv41-drive/frontend/frontend_bound.py`, CPU-tested, 4 tests). The host starts the copy-engine
DMA after post, so CW ends no earlier than the layer's last copy. A cut of `c` before CW, on a layer where CW spun `s`
past its floor (p5 of CW), saves at most `max(0, c - s)`. For a reset-only frontend `c = S.start - post.end`; for
`HIT_WAIT_US=0`, `c = W1 - p10(W1)`. S's own NVMe wait can hide more of `c`, so both are upper bounds.

**Result on the recipe with SM small copies** (node-mode trace `sm-small-B-node-20260926-034122`, 90 steady steps x 40
layers; W1 1.055 ms/step and 994.6 MB/step of copies match `ce_trace.py` on the same trace):

| Per step | Value |
|---|---:|
| Frontend span (post end to S start) | 1.13 ms |
| W1 | 1.05 ms; p50 11.3 us, p90 78.1 us; 7.7% at the 100 us budget |
| CW floor / CW spin | 0.86 us / 54.6 ms |
| Copy-engine H2D | 994.6 MB, 78.0 ms busy |
| Last copy lands after post (p50 per layer) | 1,979 us |
| CW ends after the last copy (p50 per layer) | 4.96 us |
| **Bound, reset-only frontend** | **0.41 ms** |
| **Bound, `HIT_WAIT_US=0`** | **0.32 ms** |

Both are under the 1.0 ms/step gate (untraced A/Bs move ~0.5 ms/token between identical arms), so neither Task 2's
env-var A/B nor Task 3's reset chain was built or run. The frontend is almost wholly hidden: each layer waits ~2 ms for
its copies, and CW finishes ~5 us after the last one lands. The lever is the transfer, not the chain in front of it.
`frontend_bound.py` and `ce_trace.py` share the step grouping and memcpy filter, so their agreement checks the
arithmetic, not the grouping; `check_steps` confirms every step has exactly 40 layers.

**Resident-first after real copies** (`analysis/dsv41-drive/resident-first/split_launch_bench.py`, commit `f04a78f3d3`;
`divix01:/mnt/nvme1/frontend/resident-first/`). New arms copy each layer's missed rows H2D from pinned memory inside
the 40-layer graph. `test_the_bench_copy_lands_in_the_rows_the_missed_launch_reads` proves the copies reach the launch:
poisoned bytes change the output, the true bytes restore it bitwise (parity file 5 passed). `one` = 101.43 us/layer
(bench-run2: 101.13), 788 GB/s implied.

| us/layer | 5+1 | 4+2 | 3+3 |
|---|---:|---:|---:|
| `miss_only` (no copy) | 100.38 | 100.87 | 100.86 |
| `copy_only` | 976.19 | 1952.47 | 2926.98 |
| `copy_then_one` (today's serial order) | 1086.92 | 2066.18 | 3037.52 |
| `copy_then_miss` | 1084.03 | 2063.73 | 3036.31 |
| `overlap` (copy beside the resident launch, join, missed launch, one gather) | 1076.53 | 2053.37 | 3027.75 |
| **Serial minus overlap** | **10.39** | **12.81** | **9.77** |

- Copies run at 13.6 GB/s (26.6 MB in 1.95 ms at 4+2) and dominate every copy arm.
- The arms decompose as `overlap = copy_only + miss_only` (within 0.1 us at every split) and
  `copy_then_one = copy_only + one + R`, with `R` = 9.3 / 12.3 / 9.1 us/layer. The same `R` (~10 us) appears in
  `copy_then_miss`: it is a residual whenever a kernel follows the copies on the same stream in the graph, not a cache
  effect.
- So the serial-minus-overlap figure is `(one - miss_only) + R`: about 1 us/layer of real resident-first gain plus
  `R`. Production copies run off-graph on the copy engine and CW waits on a flag, so `R` may not exist there at all.
- **Gate C (25 us/layer, 1 ms/token) fails** at every split, even crediting all of `R`. Resident-first stays shelved.

### 27.16 Multi-turn prefix reuse and the hierarchical cache (result, in the recipe, 2026-09-26)

**The production recipe reused no prefix for a returning 4k-token conversation.** The hierarchical cache (HiCache)
restores it, at about 50x lower revisit TTFT, and is now in the recipe (`arm_env.ServerArgs.argv`):
`--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 --hicache-write-policy write_through`.
(Superseded 2026-09-29: ratio 14 with `page_first_direct`, which resolves to the direct IO backend, §29.19.)

**Single-turn benchmark: no effect.** The standard arm with the HiCache flags ran at ratio 1.004 against the recipe,
byte-identical. It is compatible with the DSV4 pools, adding ~0.7 GB of host pools. A single-turn arm never revisits a
prefix, so this measures only overhead.

**Multi-turn.** `analysis/dsv41-drive/hicache/multiturn.py` runs two conversations, X and Y, over 4096-token slices
of this file, interleaved X1 Y1 X2 Y2 through `/generate` (greedy, 32 new tokens). The driver is `drive_multiturn.sh`.
Output is on divix01 under `/mnt/nvme1/hicache/mt-<arm>/`.

| Arm | Prompt | Revisit cached (X2 / Y2) | Revisit TTFT (s) | Full KV pool (tokens) |
|---|---|---|---|---|
| `big`: the recipe | 4.1k | 0 / 0 | 132 / 126 | 172,544 |
| `big-noreplay`: without `--enable-decoder-swa-bounded-replay` | 4.1k | 0 / 0 | 163 / 146 | |
| `big-2k` | 2.1k | 2048 / 2048 | 3.6 / 2.0 | 172,544 |
| `big-tails16`: `--swa-prefix-tails 16` (SWA pool 3584 -> 8192) | 4.1k | 4096 / 4096 | 2.3 / 3.8 | 101,376 |
| `big-hicache`: the recipe plus HiCache | 4.1k | 4096 / 4096 | 2.3 / 3.8 | 166,144 |

- **Cause: the capped SWA pool.** In cap mode it holds 3584 slots: the request cap plus four prefix tails. Admitting
  a 4k chunked prefill evicts the other conversation's SWA tail from the tree. A match needs a live
  `sliding_window` of SWA ending at the match boundary (`SWAComponent.create_match_validator`), so the evicted
  prefix cannot match, although its full KV is still cached.
  - At 2k both tails fit, which is why earlier soaks (`ce-soak/s4`, 2.3-2.6k conversation turns) did hit.
  - Bounded replay is not involved: turning it off changes nothing.
- **Two fixes.**
  - A larger pool (`--swa-prefix-tails 16`) costs 71k tokens of full KV and still caps the number of live tails.
  - HiCache costs 6k tokens. Its write-through host copy of the SWA tail is a valid match boundary, so an evicted
    tail is reloaded from host memory. The host SWA pool is 28 pages, 2x the device's.
- **Revisit TTFT with HiCache equals the on-GPU hit's.** The host reload is not visible: 2.3 / 3.8 s, the same as
  `big-tails16`, against 125-160 s for a cold 4k prefill at ~30 tok/s.

**Equivalence (`prefix_equiv.py`, arm `equiv-hicache`).** Each case compares a warm request against the same request
cold after `/flush_cache`, with a second cold run as the noise floor. All runs are greedy, 64 tokens, `ignore_eos`.

| Case | Cached | Warm vs cold: first diff, max abs dlogprob before it | Cold vs cold |
|---|---|---|---|
| `aligned`: 4106-token prompt re-sent; boundary 4096 is a chunk end | 4096 | token 1, 0.0 | token 9, 0.70 |
| `midchunk`: 3900-token seed, then its first 3840 + 4 new tokens | 3840 | token 41, 0.215 | token 44, 0.205 |
| `reload`: X after Y evicted its tail; X reloaded from host | 4096 | token 15, 0.225 | token 15, 0.173 |

- **No hit is distinguishable from run-to-run noise**, which is large on this stack: two cold runs of one prompt split
  by token 9 with logprob differences up to 0.70. All nine texts are coherent and on-document.
  - The `aligned` warm run diverges at token 1, while the cold runs agree to token 9. The token before the split has
    identical logprobs, and the warm text is as grounded as the cold ones. Read it as noise, one sample.
- **`midchunk` was the suspect.** Under bounded replay a prefill writes late-layer SWA only for the last
  `min(128, extend)` tokens of each extend (`late_layer_tail_layout`). A match boundary inside a chunk therefore
  leaves late-layer slots, which a short suffix's first decode windows read, that no forward wrote.
  - The outputs show no damage: warm and cold diverge at the same depth as cold and cold.
  - This check was blind. After startup and a flush, those slots held zeros or same-document KV, and one greedy
    sample cannot see a sub-noise bias. The defect is real: see "Decoder bounded replay: stale late-layer SWA" below.
- **One sample per case.** This rules out gross corruption, not a subtle bias.
- **Headroom: HiCache's staging buffers caused an OOM (fixed).**
  - Every page_first host mirror allocates a device write-back staging buffer after the KV pool is sized, out of
    the runtime slack. On DSV4 that was ~0.18 GiB, 160 MiB of it the SWA mirror (28 pages x 40 layers x 149,760 B).
  - Free memory fell to 0.02-0.03 GiB. A poisoned-pool arm then died of a CUDA OOM: a 300 MiB Engram EXL3 dequant
    transient failed with 276 MiB free.
  - The `_gather_host_rows*` Triton kernels logged near those lows belong to the MoE expert stream
    (`expert_stream.py`), not HiCache. The load watch only reports under 1 GiB free.
  - **Fix (`cc/hicache-reserve`).**
    - `SGLANG_HICACHE_WRITE_BACK_STAGING_MAX_MB` (32) caps each staging buffer in bytes, so the SWA mirror stages
      5 pages.
    - `SGLANG_HICACHE_DEVICE_RESERVE_MB` (64) comes out of the KV budget at both sizing sites, like the multimodal
      reservation.
    - The DSV4 HiCache stack refuses to start if its staging exceeds the reserve (~54 MiB here).
  - **GPU check.** The same arm then ran all six trials with no OOM, and its lowest logged free memory was 0.19 GiB.
    That run's KV pool was 122,368 tokens, against 166-186k in earlier HiCache runs. Sizing varies run to run, and
    this drop is not yet explained.
- **Operational note.** Under HiCache, `/flush_cache` returns 400 ("pending requests", with none queued or running)
  straight after a request, until write-back drains. Retry it.

#### Decoder bounded replay: stale late-layer SWA on a mid-chunk prefix hit (bug, fixed in `c515312414`)

**The defect, which is also in upstream sglang.** `--enable-decoder-swa-bounded-replay` comes from upstream PR #38798
and is still on `upstream/main` (`fc9e1c8d29`, 2026-09-26).
- Layers past the last `kv_source` layer (21-39 of 40; `kv_source_layer_ids` [2, 8, 14, 20]) run a prefill only over
  each extend's last `min(128, extend)` tokens. So they write SWA KV only for those positions
  (`late_layer_tail_layout`, `deepseek_v4_backend.py` "window KV before it is never written here").
- Decode reads the full 128-token window with no floor (`make_forward_metadata_from_raw_decode` passes no
  `swa_replay_start`).
- The radix match checks only that the window's SWA slots are allocated, not that each layer wrote them
  (`SWAComponent.create_match_validator`).
- **Trigger.** A request reuses a cached prefix whose page-aligned boundary lies inside an earlier extend, then adds
  fewer than ~128 tokens. Its first decode steps then read late-layer slots no forward wrote: stale KV of whatever last
  used them.
  - Examples: the same long document with different short questions, regenerating or editing a long last message, or
    a shared system prompt followed by a short query.
- **Safe case.** Append-only multi-turn chat is safe: the revisit boundary is a prior extend end or past it, where the
  prior tail and decode wrote the window.
- **HiCache is not involved.** The host copy mirrors the same unwritten bytes, and a no-HiCache arm showed the defect.

**Output evidence (`prefix_poison.py`, arms `poison`, `poison-nohicache`).**
- Setup: 20 random-token requests of 255 tokens write late-layer KV into every SWA page, then `/flush_cache` frees the
  pages but keeps the bytes. A 900-token seed prefills [0, 512) and [512, 900), so the late layers write only
  [384, 512) and [772, 900).
- A warm request of its first 768 tokens plus 4 hits 768. Its first decode window reads 123 unwritten slots in
  [645, 768).
- A 136-token-suffix control reads none.
- Result:
  - Short trials: first-decode logprob off by 0.11-0.50 (one token flipped).
  - Controls: 0.008-0.088 (one flipped).
  - Cold vs cold: 0.002-0.12.
- Suggestive, not decisive: one control also flipped.

**Direct evidence (`swa_window_probe.py`, arms `swaprobe-a`/`-b`, probe hook on branch `cc/swa-window-probe` only).**
- Mechanism: `SGLANG_DEBUG_SWA_WINDOW_DUMP_DIR` makes the TP worker dump, after each bs=1 extend, the 576 data bytes
  (fp8 nope + bf16 rope) of every layer's SWA window rows.
- Sequence: same poisoned pool and seed, then warm, cold and cold2 of the 772-token prompt. The rows are compared by
  cosine similarity, over two trials each.

| Rows | master: warm vs cold | Fix: warm vs cold | cold vs cold |
|---|---|---|---|
| Late layers, 645..767 | **0.05, 0.08** | 1.0000 | 1.0000 |
| Late layers, 768..771 (the suffix) | 0.85 | 1.0000 | 1.0000 |
| Early layers, 645..767 | 0.99 | 1.0000 | 1.0000 |
| Early layers, 768..771 | 0.98 | 1.0000 | 1.0000 |

- On master the rows the first decode reads are uncorrelated with the right ones, a cosine of 0.05.
- The early layers are the extraction check: the same positions match at 0.99.
- Two cold runs are byte-identical, so the gap is not noise.
- **The suffix's own late rows (0.85).** A 4-token extend's late layers attend only those 4 tokens, floored at the
  tail start. That is bounded replay's approximation on a short extend, not the stale slots.

**The fix (`c515312414`).** `UnifiedRadixCache.swa_reprefill_tail_tokens` returns the sliding window whenever decoder
bounded replay is on, with or without a host SWA pool.
- The scheduler already caps a match at `input_len - swa_reprefill_tail_tokens()`; unified_kv uses the same hook for
  its per-request ring. So a hit now leaves at least one window to prefill, and the request's own late-layer tail
  covers every slot its decode reads.
- The tree still returns the deepest valid boundary under the cap: 512 here, extend 260, the same extend a cold run's
  second chunk has.
- Test: `test/registered/unit/mem_cache/test_decoder_replay_reprefill.py`, red before and green after.
- Suite: `pytest test/registered/unit/kernels` plus the two new files gives 1339 passed / 412 skipped. That is
  master's 1329/412 plus the 10 new cases (divix01, `CUDA_VISIBLE_DEVICES=` so the GPU tests skip, `-p no:randomly`).
- **Cost.**
  - A hit whose suffix is under 128 tokens now prefills up to ~383 tokens: a 128-token window plus up to 255 of page
    alignment.
  - That includes hits that were safe, such as a revisit whose boundary is a prior chunk end. At this box's ~30 tok/s
    prefill that is up to ~9-13 s of TTFT on such a hit, against ~2 s before.
  - Hits with a suffix of 128 tokens or more pay nothing.
  - Recovering the safe cases would need the tree to record which page boundaries have late-layer coverage.

### 27.17 Prefill chunk size: 4096-token chunks for 250k contexts (result, in the recipe, 2026-09-26)

**Why prefill was slow.** A chunk's cost is streaming the experts it routes to, not its tokens or its context.
- In the 30k-prompt smoke (`/mnt/nvme1/indexer-cap/b128-30k`, §27.7's older recipe), every 512-token chunk took
  ~31 s, flat from 0 to 30k of context.
- Per chunk, layers 0-20 touch ~250 of 384 experts: ~230 miss VRAM and ~190 are read from NVMe (`stages.jsonl`).
- Layers 21-39 see only the chunk's last 128 tokens (decoder bounded replay), so a larger chunk adds tokens to the
  first 21 layers only.
- The payoff trace's short prefill agrees: 14.7 s held ~0.3 s of expert GEMMs, 6.3 s of pinned-RAM gathers and
  ~6 s of GPU idle between gathers.
- At 512 tokens a 250k prompt was ~490 chunks, about 4 h on that recipe and ~2 h on today's.

**The Engram dense transient.** Above 144 rows `exl3_linear` rebuilt the whole fp16 weight and, for a weight
narrower than 32768 columns, a second full-size reconstruct buffer. That is 600 MiB for the Engram wkv
(6144 x 25600).
- 2048- and 4096-token chunks ran out of memory on it.
- Fixed in `d7f6ded58d`: `_exl3_dense_matmul` reconstructs and multiplies one 4096-column slice at a time.
- `test_dense_path_never_holds_the_whole_weight` (`test/manual/dsv41/test_exl3_ops_gpu.py`) measured a 606 MiB peak
  before the fix against a 150 MiB bound. After it, the file passes 27/27 on the GPU.

**The hot cache counts against `--mem-fraction-static`.** Cutting it at 0.925 only grows the KV pool: at 16100 MB
the pool went from ~256k to 934k tokens, and 2048- and 4096-token chunks still ran out of memory. The cut needs
the fraction lowered with it.

**Sweep** (`analysis/dsv41-drive/prefill-chunk/`, `divix01:/mnt/nvme1/prefill-chunk/`). One cold server per arm,
a 256-token warm-up, then one 16,000-token prompt, context 32768, one run each:

| Chunk | Hot MB | Fraction | TTFT | Prefill | Median chunk | KV pool | Long prompt adds | OOM retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 17120 | 0.925 | 444 s | 36 tok/s | 14 s | 304k | ~0 | 0 |
| 2048 | 17120 | 0.925 | 204 s | 78.5 tok/s | 23 s | 256k | 764 MiB | 1 |
| 2048 | 16100 | 0.90 | 186 s | 85.9 tok/s | 23 s | 431k | 948 MiB | 0 |
| **4096** | **16100** | **0.90** | **107 s** | **150 tok/s** | 17 s | 387k | 1,416 MiB | 0 |
| 4096 | 16100 | 0.925 | OOM | | | 877k | | 3 |
| 2048 | 16100 | 0.925 | OOM | | | 934k | | 3 |

- Headroom at the 4096 arm's peak was recorded as ~470 MiB, but that is nvidia-smi's 32,607 MiB total minus the peak
  (32,134 MiB, `c4096-16k-h16100-m090/vram.csv`). CUDA can reach 32,202 MiB, so the real margin was ~68 MiB (§27.18).
  *Corrected 2026-09-29:* this line first said "CUDA can use only 32,150 MiB ... ~16 MiB"; the measured ceiling is
  32,202 MiB (torch's total is 33,766,572,032 B, and a chunked 16k prompt peaked at 32,201 MiB `memory.used`).
  Evidence: `analysis/dsv41-drive/recipe-mem/results.md`, "Verification at 0.885 / 15400".
- Decode after the long prompt read 109-121 ms/token across arms. These are 64 tokens right after a prefill that
  evicts decode's RAM set (§27.2), and the 512 -> 2048 change at the same hot size moved it as much as the hot-cache
  cut did, so this is not a measure of the cut. §27.7's slope suggests the 1 GiB cut costs ~2-3 ms/token of steady
  decode. Not measured.
- The earlier 2048 arm's warm-up read 20.1 s TTFT and 307 ms/token against ~12.3 s / ~168 ms in every other arm.
  Not investigated; its 16k prompt was in line.

**In the recipe:** `CHUNKED_PREFILL_SIZE = 4096`, `MEM_FRACTION_STATIC = 0.90`, `SGLANG_MOE_HOT_GPU_MB = 16100` and
`CONTEXT_LENGTH = 262144`.

**Superseded 2026-09-29:** the recipe is now `MEM_FRACTION_STATIC = 0.885`, `SGLANG_MOE_HOT_GPU_MB = 15400` and
`CONTEXT_LENGTH = 131072` (`CHUNKED_PREFILL_SIZE` stays 4096). NVIDIA driver 615.71.09 (from 610.57.04) grew the EAGER
CUDA context, and 0.90 / 16100 no longer fit the KV pool. Merge `cfe4d822da`; `benchmarks/dsv41_baseline/arm_env.py`;
§29.11.

### 27.18 128k prefill: the layer-20 candidate mask OOM (result, 2026-09-27)

- **The failure (phase 0b Task 0, `divix01:/mnt/nvme1/prefill-chunk/phase0-128k/`):** OOM'd in `flash_mla_sm120.py:251`
  needing 256 MiB with 519.5 MiB reserved and free but no contiguous block that size, after 57,344 prompt tokens
  processed plus 256 cached (chunk 15 per the stage trace; `server.log` logs only 14 long-prompt chunk lines before
  the OOM, so the log alone does not settle 14th vs 15th).
- **The cause:** layer 20 is the unique layer that runs the full 4096-row chunk, scores with indexer ratio 1 (so
  `lc` equals the whole prefix), and is `candidate_source_layer_id`. It publishes a `[T, P]` bool candidate mask that
  stays alive through `flash_mla` and layers 21-39, though only its last 128 rows are ever read
  (`deepseek_v4_backend.py`, `enter_late_layer_tail`). Mask size scales with the prefix: 62.5 MiB at 16k, 240 MiB at the 61k
  failure, 512 MiB at 131k, and 1 GiB at 262k (build peak is 2x that). This is the diagnosis's top-ranked hypothesis
  (`oom-diagnosis.md` Q3.1): the failing run sat 61-96 MiB below the 32,202 MiB CUDA limit (`phase0-128k/vram.csv`:
  32,106-32,141 MiB used) from chunk 1, and the mask's growth alone exceeds that margin, with allocator
  fragmentation as the proximate trigger. The fixed 128k run passing (below) supports the hypothesis but does not
  prove it: the confirming memory snapshot (Q4) was not run, and an unattributed ~258 MiB request (Q3.3) remains
  unexplained.
- **The fix (Task 1, this branch):** publish only the tail rows' candidate mask, instead of the full `[T, P]` mask.
  Tail-only publishing applies when the config puts every candidate consumer in the late-layer tail
  (`candidate_tail_only_of`) **and** a tail runs this forward (`candidate_publish_rows`); otherwise the full mask is
  still published. Where it applies it is exact, not an approximation: `uses_candidates` is true only for layers
  above 20, and with bounded replay every one of them reads `mask[-t:]` — layer 20's own top-k never reads the mask
  it publishes. Row independence means the tail rows' bits are unchanged by not building the rest.
- **Scope:** every GPU run below had `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0` (layer-major off). Task 1's other
  change, `run_layer`'s tail install for the layer-major path, is exercised only by CPU unit tests here, not by any
  GPU run.
- **Step 2 equality (16k, greedy, `temperature=0`):** base (`290b39fa85`, pre-fix) and fix (`cee8138fc4`) each ran
  `long rc=0`, 0 OOM retries. Base: TTFT 118.3 s, decode 127.0 ms/token. Fix: TTFT 106.3 s, decode 130.4 ms/token.
  The 32-token greedy completions are byte-identical (`DIFF_EXIT=0`).
- **Step 3, the 128k run with the fix:** `long rc=0`, 0 OOM retries, 0 "memory allocation failed with OOM" lines.
  TTFT 931.5 s (8-token completion, decode 168.9 ms/token). 256 cached warm-up tokens, then 31 full 4,096-token
  chunks plus a 3,840-token remainder. Chunk intervals 25-44 s, median 26 s, last full-size interval 25 s (the final
  two chunks log at the same second and cannot be timed individually). Peak VRAM 32,136 MiB, ~66 MiB below the
  32,202 MiB CUDA limit — thin, matching §27.17's corrected margin. Minimum NUMA node-1 free: 44,568 (MB).
- **Gate:** `long rc=0`, 0 OOM retries, last full-size chunk (25 s) within 1.5x the first (27 s) → **Done. Resume
  layer-major Task 12.** 128k passes with only ~66 MiB of real headroom; 262k was not run (the fix shrinks its mask
  from 1 GiB to ~32 MiB, but other T- and P-scaled items are untested there).
- Tree tested: `cee8138fc4` (`f22e3607f8` after the rebase onto the expert-stream master).
- *Corrected 2026-09-29:* the margins above first read 9-44 MiB (failing run) and ~14 MiB (128k peak), against a
  "32,150 MiB CUDA limit". CUDA reaches 32,202 MiB (`analysis/dsv41-drive/recipe-mem/results.md`), so the margins are
  recomputed from the same `vram.csv` peaks: 32,202 − 32,141..32,106 = 61-96 MiB, and 32,202 − 32,136 = 66 MiB.
- Evidence: `divix01:/mnt/nvme1/prefill-chunk/oom0b-base-16k/`, `oom0b-fix-16k/`, `oom0b-fix-128k/`, `phase0-128k/`,
  `c4096-16k-h16100-m090/` (`driver.log`, `server.log`, `vram.csv`, `numa.log`, `long.json`); diagnosis at
  `.superpowers/sdd/2026-09-27-dsv41-layer-major-prefill-phase1/oom-diagnosis.md`.

**Traced 4096-token chunk** (`trace-c4096-12k`, today's recipe at context 262144; a 12,288-token prompt, node mode;
`prefill_trace.py`).
- The server started at context 262144 with a 374,016-token KV pool and 2.43 GB free, as at 32768. (Driver 610.57.04,
  0.90 / 16100. Superseded 2026-09-29: at 0.885 / 15400 on driver 615.71.09 the pool is 150,784-232,448 tokens and the
  recipe's context is 131072; `analysis/dsv41-drive/recipe-mem/results.md`, §29.11.)
- TTFT was 81.6 s traced, ~27 s per chunk, against ~17 s untraced; node mode's per-launch cost on ~170k eager kernels
  per chunk accounts for the difference. Proportions below are per chunk and traced.

| Per 4096-token chunk | Value |
|---|---:|
| Rows crossing PCIe (VRAM misses): layers 0-20 / 21-39 | ~6,400 / ~2,200 |
| Gather kernels, at 115 GB / 9.3 s = 12.3 GB/s (the link) | 9.3 s |
| NVMe rows read (all in layers 0-20) | ~6,000 (82 GB) |
| Host blocked on NVMe reads, 82 GB / 11.8 s = 7 GB/s | 11.8 s |
| GPU idle, ~125 gaps of ~100 ms after a routing index kernel | 12.4 s |
| EXL3 GEMMs / other kernels / attention | 0.8 / 0.6 / 0.06 s |

- **Both transfers run at their ceilings, one after the other.**
  - 7 GB/s is the two expert mirrors' line rate: nvme0 and nvme4 are Gen3 x4 links, ~3.5 GB/s each.
  - Of every chunk, 94% of layers 0-20's VRAM misses are also RAM misses. The pinned tier's ~200 rows per layer cannot
    hold the ~340 experts per layer that each chunk routes to, so every chunk re-reads them from NVMe.
- **Late layers:** ~26% of the link traffic and none of the NVMe reads, so skipping them on non-final chunks is worth
  at most ~2.4 s of link time per chunk.

**Not run at this recipe:**
- Any prompt past 16k. The 250k estimate is ~61 chunks at 17-27 s, ~20-30 min, if chunk time stays flat with
  context as it did to 30k.
- Steady decode ms/token at the smaller hot cache.

### 27.19 Task 12 fix round: layer-major vs chunked GPU equivalence, radix crash closed (2026-09-27, round 3)

Phase 0's numbers (TTFT, chunk times at 128k, peak VRAM, indexer path, node-1 free) are §27.18; not repeated here.

**The radix crash from the first Task 12 run (`new_prefix_len=32512, len(new_indices)=16384` in
`unified_radix_cache.cache_unfinished_req`) is fixed, in the layer-major adapter only.** In brief: bounded-replay
admission can cap a match at a device-window boundary that lies inside a *tombstoned* SWA span the tree still
holds live full KV for (`swa_branching_seqlen`), and the post-prefill radix insert then needs live window KV up to
that branch point, not just up to `seq_len`'s last window. `DeepseekV4LayerMajorAdapter._finalize_ring` kept only
the latter, so the insert key ended on an unmatchable tombstone. The fix (`_insert_keep_from`) keeps from
`max(prefix_len, page_floor(end - 1 - max(window, page)))`, where `end` is the branch point when the ring still
covers it, else `page_floor(seq_len)`; it falls back to the seq_len floor when the ring has already overwritten the
branch point's window. `unified_radix_cache.py` and `components/swa.py` are unchanged.

**The pass criterion is token-0 id equality, not full 64-token completion identity.**
- Why: cross-server decode drift. In the run this criterion was adopted from, 5 of 7 chunked-vs-chunked cases
  (both arms running the *same* strategy) diverged within 64 decode tokens, so comparing the full completion would
  fail a correct layer-major implementation as often as a real bug; DSV4.1's decode kernels are documented as not
  bitwise-stable across batch composition, and `--enable-deterministic-inference` is refused on this backend.
- Who: the change was agreed in the user's session with the Task 12 agent, and ratified by controller ruling.
- `equiv.py`'s module docstring and `compare()`'s docstring both state this; `compare()` gates on `ids[0]` equality
  and prints the full-completion divergence informationally only.

**Check (a).** `test_window_ring.py::TestWindowRingCudaFreePath` reaches `_free_swa_pages_cuda` for real: it frees
the kept tail through `free_segment`/`free_swa_segment` (wrapped with `mock.patch.object(..., wraps=...)`, asserted
called), then frees the already-released prefix through `free_full_segment` -- which itself asserts
`finalize_ring` cleared that prefix's SWA mapping (`_SWA_PEER_RELEASED`), rather than `free()`'s
`finalize_ring`-only path, which never dispatches to `_free_swa_pages_cuda` at all. GPU run (CPU-affine pytest
under `flock -s` on `cc-gpu.lock`, per today's shared-lock convention): `1 passed`.

**Radix I1 (unflushed chain): closed as CPU-covered, not GPU-covered, per controller ruling.** `equiv.py` includes
`chain-32768` (flush), `chain-32868` (no flush, shares the 32768-token prefix), `chain-32868-again` (no flush,
re-sent). In the `layer-major-8k` arm's `server.log`:

```
[2026-09-27 18:30:24] layer-major prefill: 32768 tokens in 8 chunks
[2026-09-27 18:30:24] Prefill batch, #new-seq: 1, #new-token: 32768, #cached-token: 0, ...
[2026-09-27 18:30:42] Prefill batch, #new-seq: 1, #new-token: 356, #cached-token: 32512, ...
[2026-09-27 18:30:59] Prefill batch, #new-seq: 1, #new-token: 356, #cached-token: 32512, ...
```
`#cached-token: 32512` is `page_floor(32868 - 128)`, the bounded-replay cap -- **an ordinary live-window hit, not a
branch point.** With the fix, `chain-32768`'s finalize keeps window KV live from `page_floor(32768 - 1 - 256) =
32256` onward, and 32512 falls inside that live span (`32256 <= 32512 < 32768`). `chain-32868`'s own extend (356
tokens) is below `min_tokens` (8192) and runs chunked on both arms, so no layer-major pass ever sees this
admission's branch logic. The chunked arm hits the identical `#cached-token: 32512` at 17:48:36/17:48:53 for its
own `chain-32868`/`chain-32868-again`, consistent with an ordinary radix hit, not a fix-specific code path.

**No GPU case in this run exercises the branch-kept path.** It is constructible on GPU without a race: an
unflushed follow-up like `ids[:20000] + <2000 fresh tokens>` after the unflushed `ids[:32768]` would get a device
match of 0 (below `chain-32768`'s live floor of 32256, since this key diverges earlier) and a branch point at the
tombstone boundary 19968, and at seq_len 22000 (>= 8192) it would run layer-major with keep 19456 >= `oldest_intact`
17664 -- the round-1 reviewer's construction, the shape `test_prefix_hit_capped_at_a_tombstone_branch_point`
already covers on CPU. Not run in this pass; deferred per controller ruling. Coverage of that path is the CPU test
`test_layer_major_radix_insert.py::test_prefix_hit_capped_at_a_tombstone_branch_point` (a pre-seeded tombstoned
tree, `prefix_len=0`, branch point 1792) and the new
`test_live_prefix_with_a_branch_clamps_to_the_prefix` (a pre-seeded live prefix of 1024 with a tombstoned branch at
1280, spy-verified: `prefix_len=1024 > 0` and the clamp overrides the branch-derived floor of 768). Per controller
ruling, this is accepted as CPU-only coverage; the chain case stays in `equiv.py` as a real regression case for the
crash's *admission* shape (pre-fix, `chain-32868`'s match came up short and it was misrouted to layer-major with
an unrecoverable branch; post-fix it is correctly recognized as a live hit and runs chunked), not as branch-path
GPU coverage.

**GPU equivalence table** (`/mnt/nvme1/layer-major/equiv-t12fix/`, both arms at commit `08da71777d`, `dirty=0`;
"chunked" column re-read from `server.log`'s own `Prefill batch` lines per case):

| case | prompt | chunked #new/#cached | layer-major-8k #new/#cached | ran layer-major? | token 0 | e2e chunked / lm-8k (s) |
|---|---:|---|---|---|---|---|
| len8192 | 8192 | 4096x2/0 | 8192/0 | yes, 2 chunks | IDENTICAL | 69.7 / 59.2 |
| len16384 | 16384 | 4096x4/0 | 16384/0 | yes, 4 chunks | IDENTICAL | 112.0 / 102.3 |
| len32768 | 32768 | 4096x8/0 | 32768/0 | yes, 8 chunks | IDENTICAL | 219.5 / 193.1 |
| len33000 | 33000 | 4096x8+232/0 | 33000/0 | yes, 9 chunks | IDENTICAL | 221.7 / 192.3 |
| len32868 | 32868 | 4096x8+100/0 | 32868/0 | yes, 9 chunks | IDENTICAL | 202.1 / 187.5 |
| prefix-warm | 1024 | 1024/0 | 1024/0 | no (below 8192) | IDENTICAL | 23.2 / 22.9 |
| prefix | 33792 | 4096x8/1024 | 32768/1024 | yes, 8 chunks, radix prefix hit | IDENTICAL | 198.1 / 186.8 |
| after | 256 | 256/0 | 256/0 | no | IDENTICAL | 16.1 / 15.8 |
| chain-32768 | 32768 | 4096x8/0 | 32768/0 | yes, 8 chunks | IDENTICAL | 196.9 / 201.4 |
| chain-32868 | 32868 (unflushed) | 356/32512 | 356/32512 | no on either arm (live-window hit, see above) | IDENTICAL | 17.2 / 17.5 |
| chain-32868-again | 32868 (unflushed, resend) | 356/32512 | 356/32512 | no on either arm, same hit, resend | IDENTICAL | 17.9 / 17.4 |

`compare chunked.jsonl layer-major-8k.jsonl` (`--allow-head-mismatch` not needed, both arms share a head):
`EXIT=0`, every case `IDENTICAL (token 0)`, top-5 logprob max |delta| `0` on every case (T12 Minor 1: the server
accepted `return_logprob`/`top_logprobs_num=5` without incident, so this ran; informational only, the gate stays
token-0 id equality). **8 of 11 cases'** full 64-token completions differ from decode step 1-62 onward
(`chain-32768`, `chain-32868`, `chain-32868-again`, `len16384`, `len32768`, `len8192`, `prefix`, `prefix-warm`;
documented decode drift, not gating -- `len33000`, `len32868`, `after` are byte-identical). `grep -c "memory
allocation failed with OOM"`: chunked 1 (recovered, between `prefix-warm` and `prefix`, unrelated to layer-major),
layer-major-8k 0.

The e2e figures above include the 64 decode tokens and are single-run wall-clock, not averaged. Of the 7 rows where
layer-major actually ran (`len8192`, `len16384`, `len32768`, `len33000`, `len32868`, `prefix`, `chain-32768`), 6 ran
faster than chunked and 1 (`chain-32768`, 196.9 s chunked vs 201.4 s layer-major) ran slower. The other 4 rows
(`prefix-warm`, `after`, `chain-32868`, `chain-32868-again`) ran chunked on both arms and are not a layer-major
comparison. No causal claim is made about the 6/7 split (both paths still stream experts per chunk in phase 1), and
this is one run on one box, not a claim that generalizes.

**Check (d).** The `262144 x 4 x 5120` bf16 `hidden` StateStore field is `262144 * 4 * 5120 * 2 bytes = 10.0 GiB`.
Measured NUMA node-1 free minimum, with that store already resident: chunked arm 26.2 GiB, layer-major-8k arm
19.5 GiB. That free minimum *is* the headroom past the store (it was sampled while the store was allocated), not a
quantity to further subtract the store from.

**Radix M5 (held window after finalize).** A layer-major request with a branch can hold up to about one ring
(~17 pages at production geometry) of window KV briefly after `_finalize_ring`, not "1-2 pages"; it is freed in the
same scheduler step by `cleanup_after_caching_req` while
`SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS` is on (the production default). Not a correctness issue.

**Corpus (T12-I4, remaining half).** `drive_equiv.sh` now reads prompts from a pinned snapshot,
`/mnt/nvme1/layer-major/equiv-corpus/corpus.txt` (`DSV41_REFERENCE.md` at `29c7d4e2b1`, sha256
`59abe27a89c6915935e6ce1cc79acf3d2acf393067b5609d753b8ff0c9fc70c3`), and refuses if the file's hash differs, instead
of re-tokenizing the worktree head's own (edited-since) `DSV41_REFERENCE.md`.

**Controller additions, CPU-tested only (no GPU re-run):**
- `equiv.py run --cases quick` runs a fast subset for iteration (`len8192`, `len32768`, `len33000`,
  `prefix-warm`/`prefix`, the unflushed chain); the header records `cases` (`all`/`quick`).
- `drive_equiv.sh <arm> <wt> <min_tokens> <out_root> <baseline_jsonl>` skips that arm's server run and copies
  `baseline_jsonl` to `<out_root>/<arm>.jsonl` instead, for reusing an existing chunked baseline.
- `compare()` now supports a `quick`-vs-`all` pairing directly: it compares the smaller file's cases, provided
  every one is present in the larger file (the larger file's extra cases are not an error); it refuses when either
  arm's header is `dirty` unless `--allow-dirty` is passed, and refuses when both arms have `min_tokens=0` (nothing
  ran layer-major on either side). `--allow-head-mismatch` is still required separately for a different-head
  baseline reuse; the per-case `prompt_hash` check is unconditional either way.

**Open question, not blocking (from `review-radix-fix.md`).** Late-layer SWA in the tree-adopted window below a
branch point is never written under bounded replay: `swa_reprefill_tail` re-prefills one window from the match end,
but the first replayed token still attends one window further back, and nothing writes those positions in late
layers. Chunked prefill has the same property (this predates layer-major), so token-0 equivalence above holds
regardless; whether either is numerically correct is outside this diff. Owner: bounded replay.

**Commands:**
```bash
# divix01, wt-lm-t12fix at cc/lm-t12-fix (08da71777d)
cd /mnt/nvme1/layer-major && bash <wt>/analysis/dsv41-drive/layer-major/drive_equiv.sh chunked <wt> 0 /mnt/nvme1/layer-major/equiv-t12fix
bash <wt>/analysis/dsv41-drive/layer-major/drive_equiv.sh layer-major-8k <wt> 8192 /mnt/nvme1/layer-major/equiv-t12fix
cd /mnt/nvme1/layer-major/equiv-t12fix
python equiv.py compare chunked.jsonl layer-major-8k.jsonl     # EXIT=0
```
Evidence: `divix01:/mnt/nvme1/layer-major/equiv-t12fix/{chunked,layer-major-8k}/` (`driver.log`, `server.log`,
`vram.csv`, `numa.log`, `phases.txt`) and `equiv-t12fix/{chunked,layer-major-8k}.jsonl`.

### 27.20 Final-review fix pass: C1 late-layer SWA tail coverage, I1 launch refusal (2026-09-27)

**C1 mechanism.** `finish_pass` (`models/deepseek_v4_layer_major.py`) ran the late layers
(`late_layer_start..end`) once, over the final span's own tail only. When the final span has
`r < SWA_WINDOW` rows -- e.g. an 8200-token suffix at `chunked_prefill_size=4096` splits into spans
of 4096, 4096, 8 -- the late layers never ran over the penultimate span's positions
`[s-window, s-r)`. Their layer-21..39 SWA slots for those positions kept whatever an earlier
position or request had left there. Decode then attends, for up to `window - r` steps, to KV that
was never written in this pass. Token 0 cannot see this: the tail is floored at the tail start, so
it is exactly the rows the (buggy) pass did write. Chunked prefill does not have this failure mode,
because every extend under bounded replay runs its own tail (`min(window, rows)`), never skipping
one.

**Fix.** `tail_run_spans(spans, window, prefix_len)` (pure) walks back from the final span and
returns every span, oldest first, whose own last `min(window, rows)` rows fall inside decode's
reach `[max(prefix_len, s-window), s)` -- in practice the final span plus, only when it is short,
the one before it (guaranteed sufficient by the existing ring invariant `chunk >= 2*page`,
`page >= window`, so every non-final span has `>= window` rows). `begin_pass` captures each needed
span's own tail metadata (already built by `init_forward_metadata` for every span, previously kept
only for the final one) and gates candidate-mask publishing on the same needed-span set, so the
candidate-source layer (20) also publishes tail-only masks for the penultimate span when required,
via the existing `layer_major_skip_candidates`/`candidate_tail_only`/`candidate_publish_rows`
mechanism, unchanged. `finish_pass` then runs the late layers over each needed span in order
(penultimate before final), discarding every output but the true final span's.

**CPU tests (`test/registered/unit/layer_major/`, all `register_cpu_ci`):**
- The C1 proof is `test_c1_late_layer_write_coverage.py`, RED at `43d7813a41` and GREEN at the fix
  head. It calls the unbound `finish_pass` with stand-ins for `causal_lm`/backend/store, recording
  which positions the late layers ran over (not a GPU-verified write), using no symbol the fix
  adds. RED at `43d7813a41` for suffix lengths `chunk*2+8` and `chunk*2+127` (assertion failure:
  positions `[s-window, s-r)` never covered, not an ImportError); GREEN at the fix head for those
  two plus the `chunk*2` and `chunk*2+window` boundary cases, which were already correct and stay
  so. A separate case asserts the call order and the exact per-span metadata pairing.
- `test_c1_late_layer_tail_coverage.py`: pure `tail_run_spans` unit tests for the same four cases.
- `test_dsv4_backend_install.py::TestRunLayerPenultimateTailMetadata`: `run_layer` installs each
  needed span's own tail metadata (penultimate distinct from final), not None and not the final
  one's -- the generic "publish exactly the tail's rows" mechanism itself is pinned generically in
  `test_dsv4_candidate_indexer.py`'s existing tail-publish tests, reused unchanged here.
- Registered suites unaffected by the rename (`_Pass.final_tail_metadata` -> `_Pass.tail_by_span`)
  were updated to the new field name; behavior of those tests is otherwise unchanged.
- On divix01 at `1e8d22ed39` (`PYTHONPATH=$WT/python CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
  taskset -c 18-35,54-63 .venv/bin/python -m pytest <target> -q -p no:randomly
  --basetemp=/mnt/nvme1/pytest-tmp/...; echo EXIT=${PIPESTATUS[0]}`): `unit/layer_major` 78
  passed, 2 skipped, `EXIT=0`; `test_dsv4_candidate_indexer.py` 43 passed, `EXIT=0`;
  `test_dsv41_torch_indexer_chunking.py` 4 passed, 3 skipped, `EXIT=0`.

**GPU verification.** `equiv.py` gained a `len8200` case, a `--cases c1` subset (just that case),
and `compare()` accepts a `c1` run against a full baseline the same way it already accepted
`quick`. Corpus, locks and `/mnt/nvme1` scratch per the run protocol; production untouched
throughout.

| arm | worktree (head) | min_tokens | path (server.log) | token 0 | first differing decode index | mean output-token logprob |
|---|---|---:|---|---|---:|---:|
| chunked | `wt-lm-final-fix` (`0a96af96df`) | 0 | chunked (no layer-major) | -- | -- | -0.3103 |
| (a) pre-fix | `wt-lm-final-prefix` (`43d7813a41`, dirty: `equiv.py` copied in) | 8192 | `layer-major prefill: 8200 tokens in 3 chunks` | IDENTICAL vs chunked | 3 / 64 | -0.3434 |
| (b) fix head | `wt-lm-final-fix` (`0a96af96df`) | 8192 | `layer-major prefill: 8200 tokens in 3 chunks` | IDENTICAL vs chunked | 45 / 64 | -0.4156 |

First 20 decoded tokens:
- chunked / (b): `[63, 7640, 94, 2619, 39981, 23809, 14, 418, 420, 119683, 666, 369, 2619, 96, 856, 16, 20, 666, 369, 223]` (identical)
- (a): `[63, 7640, 94, 420, 119683, 25137, 369, 5420, 24, 16, 2402, 369, 223, 21, 13656, 14, 223, 7833, 13523, 14]` (diverges at index 3)

Reading: token 0 is identical on both arms, as expected (C1 cannot move it). (a) diverges at decode
index 3, inside the 27.19 drift envelope; with one sample it neither shows nor rules out C1. (b)
matches chunked's first 20 tokens exactly and diverges only at index 45/64, consistent with drift.
Its mean logprob (-0.416) is further from chunked's (-0.310) than (a)'s (-0.343); mean self-logprob
is not comparable across arms once they diverge. The GPU result is inconclusive. The proof of C1
and its fix is the CPU write-coverage RED/GREEN. A decisive GPU check remains open: a NaN-sentinel
assert on late-layer SWA slots for `[s-128, s)`, or >=5 repeats per arm against the
chunked-vs-chunked spread.

**I1 fix.** `scheduler_layer_major_refusal` (`layer_major/gate.py`) refuses a launch when
`SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS < chunked_prefill_size + page_size` (the ring size), naming
both values and the env var; before this, such a launch passed and the first eligible request hit
`alloc_extend_swa_tail`'s assertion inside scheduling (outside `run_batch`'s exception containment),
SIGQUITing the server. CPU test: refusal at 4096, acceptance at 4352 (chunk 4096, page 256).

**Commands:** `drive_equiv.sh` itself has no `--cases` flag; a scratch copy
(`/mnt/nvme1/lm-final-fix/drive_equiv_c1.sh`, not committed) added `--cases c1` to its `equiv.py run`
line. (a)'s worktree is pre-fix, so its own `equiv.py` (no `len8200`/`c1` case) was overwritten with
the fix head's copy (scratch only, not committed) before the run.
```bash
# divix01, wt-lm-final-fix / wt-lm-final-prefix at cc/lm-final-fix (0a96af96df) / 43d7813a41
bash /mnt/nvme1/lm-final-fix/drive_equiv_c1.sh chunked <wt-fix> 0 /mnt/nvme1/layer-major/equiv-final
bash /mnt/nvme1/lm-final-fix/drive_equiv_c1.sh layer-major-a <wt-prefix> 8192 /mnt/nvme1/layer-major/equiv-final
bash /mnt/nvme1/lm-final-fix/drive_equiv_c1.sh layer-major-b <wt-fix> 8192 /mnt/nvme1/layer-major/equiv-final
cd /mnt/nvme1/layer-major/equiv-final
python equiv.py compare chunked.jsonl layer-major-a.jsonl --allow-head-mismatch --allow-dirty   # EXIT=0
python equiv.py compare chunked.jsonl layer-major-b.jsonl --allow-head-mismatch --allow-dirty   # EXIT=0
```
Evidence: `divix01:/mnt/nvme1/layer-major/equiv-final/{chunked,layer-major-a,layer-major-b}/`
(`driver.log`, `server.log`) and `equiv-final/{chunked,layer-major-a,layer-major-b}.jsonl`.

## 28. Computing RAM-resident experts on the CPU: kernel and handoff microbenchmarks (2026-09-29)

**Why.** Decode spends ~67 ms/token moving ~68 RAM-hit experts of 13.3 MB over the Gen3 link (§§25-27); the lease
protocol, service thread and copy thread together cost ~2-4 ms/token. The alternative is to leave RAM-hit experts in
RAM and compute them on the CPU, sending only the hidden state (10 KB) down and the expert output (20 KB) back. These
two microbenchmarks measure both halves. No server ran; nothing here is a decode arm.

**Kernel.** exllamav3's own CPU MoE kernel, `exllamav3_ext/cpu/moe_mul1.cpp` at `02aef45` (turboderp, upstream):
the mul1 codebook fused into integer dot products, with scalar / AVX2 / AVX-512BW / VNNI / VBMI tiers. The Xeon 6154
runs the AVX-512BW tier (no VNNI). It quantizes activations to int8, so it is not bit-identical to the GPU path.

**Harness.** `analysis/dsv41-drive/cpu-experts/`: `bench.py` + `bench_ext.cpp` (kernel and DRAM-read probe),
`handoff.py` + `handoff_ext.cu` (40-layer CUDA graph with a GPU->CPU->GPU handoff per layer), the three drivers
`run.sh`/`run2.sh`/`run3.sh` as run. The raw `results.jsonl` stays on divix01 beside them (the repo ignores
`*.jsonl`). exllamav3 is cloned beside the scripts, not vendored.
- DSV4.1 geometry (5120 x 2304, 3 bpw, 13,315,584 B per expert, gated SiLU), 384 random-trellis experts (5.1 GB)
  per run. Calls rotate through them so no expert repeats within 384/topk calls: every read is cold (L3 24.75 MB).
  Throughput does not depend on the trellis values.
- `EXL3_MOE_CPU_PIN=0` in every run: the kernel's pool pins workers to the machine's first physical cores and ignores
  `taskset`. Cores were physical only (node 0: 0-17, node 1: 18-35) with `numactl --membind` to the same node.
- All runs held `cc-gpu.lock`. Another session's CPU unit tests (`pytest test/registered/unit/kernels`) ran during
  r2 and the handoff runs, so those numbers lean pessimistic.

**Accuracy (DSV4.1 shapes, 2 experts, 1 thread).** The BW tier is 1.4% relative L2 from the tier-scalar fp32
reference (the int8 activation quantization; upstream's tolerance is 5%). The band-contiguous ("swizzled") layout is
bit-identical to the native one.

### 28.1 CPU expert throughput

| | node 1, r1 (pre-reboot) | node 1, r2 | node 0, r2 |
|---|---:|---:|---:|
| DRAM streaming read, 8-12 threads | 32.3 GB/s | 34.9 GB/s | **62.0 GB/s** |
| ms per expert, 1 thread | 3.92 | 4.04 | - |
| ms per expert, 4 threads | 1.02 | 1.02 | - |
| ms per expert, 8 threads | 0.57 | 0.55 | 0.56 |
| ms per expert, 12 threads | 0.45 | 0.40 | 0.39 |
| ms per expert, 18 threads | 0.46 | 0.37 | **0.28** |

Per-expert figures are 6 experts per call (p50). One expert per call costs about the same per expert (node 1,
12 threads: 0.46 ms at 1, 0.42 at 2, 0.40 at 6), which matters because decode has 1-2 RAM hits per layer.

- **One core decodes ~3.3 GB/s of EXL3**, so the kernel is compute-bound below ~8 threads and memory-bound above
  it on node 1, where 12 threads reach ~92% of the socket's read ceiling.
- **Node 0's memory reads 1.8x faster than node 1's**, so the two sockets' DIMM population evidently differs.
  On node 0 the kernel stays core-bound at 18 threads (47 GB/s effective).
- **Cross-socket costs nothing here:** node-0 cores reading node-1 memory ran 0.44 ms per expert at 16 threads (r1),
  the same as local. Node 1's DRAM is the limit, not the socket link.
- **Against the link:** the copy engine moves one expert in ~0.98 ms (13.6 GB/s). The CPU is 2.1-3.5x faster per
  expert, depending on the socket.

### 28.2 Per-layer GPU->CPU->GPU handoff

Per layer, the GPU publishes the fp16 hidden state and a ready flag to pinned host memory. A spinning worker on node 1
clears the flag, runs the kernel (12 threads, cold experts) and publishes a fp32 output and a done flag. The GPU waits,
pulls the output into VRAM, and a consumer kernel makes the next layer's input depend on it. Two GPU sides were
captured and replayed: **kernel** (zero-copy stores + a spin-wait kernel with a 2 s abort) and **memop**
(`cudaMemcpyAsync` + `cuStreamWriteValue32`/`cuStreamWaitValue32`, no SM spinning). Overhead is per-layer time minus
the worker's own measured work (p50). No replay aborted.

| Experts per layer | kernel: per layer | kernel: overhead | memop: per layer | memop: overhead |
|---:|---:|---:|---:|---:|
| 0 (worker copies the output only) | 18.5 us | 8.6 us | 25.8 us | 16.0 us |
| 1 | 473 us | 12 us | 481 us | 19 us |
| 2 | 853 us | 16 us | 863 us | 23 us |
| 4 | 1,647 us | 19 us | 1,671 us | 23 us |

- **The handoff costs 9-23 us per layer, 0.4-0.9 ms per token.** Today's path spends ~45 us per layer between a
  layer's post and its first copy (§27.3) before any bytes move.
- **Compute inside the loop matches standalone:** 461 us for one expert against 0.40-0.46 ms in §28.1.
- **16 threads:** within noise at 1 expert per layer (475-478 us), 5-15% faster at 2-4, with more jitter.

### 28.3 What it implies [estimate]

- ~68 RAM-hit experts per token at ~0.42 ms is ~28 ms on node 1 alone, against ~67 ms on the link today. Using node 0
  as well gets ~15-20 ms if each socket computes the rows in its own memory, or ~21 ms if node 1 shares bands of each
  expert with the link.
- With NVMe exposure (~17 ms) and GPU compute (~14.6 ms) unchanged, decode lands near **50-60 ms/token (17-20
  tok/s)** against ~100 today.

**Not established:**
- Quality: a logprob/KL comparison of the int8-activation CPU path on real DSV4.1 weights.
- DSV4.1's clamped SwiGLU (limit 10) is not one of the kernel's activations; SiLU was measured (same cost).
- Behaviour under a running server: service threads, NVMe DMA into node-1 memory, co-tenants.
- CPU load on node 0 cuts H2D by 27-43% (§25.4); that matters less once the link carries little.
- An expert computed on the CPU is not promoted into VRAM; the idle link could do that in the background.

**Operational traps:**
- The kernel's pool pins its own workers unless `EXL3_MOE_CPU_PIN=0` (above).
- Pinning the handoff worker to one core made the pool it spawns inherit that core. Twelve spinning threads on one
  core livelocked and held `cc-gpu.lock` for ~12 minutes; `handoff.py` now refuses a pinned worker.

**Commands** (divix01, `/data/models/slang/nvfp4-work/cc-exl3-cpu-bench`, which also holds `results.jsonl` and the logs;
`EXL3_MOE_CPU_PIN=0`, `.venv` python):
```bash
numactl --membind=1 taskset -c 18-21 python bench.py tiers
numactl --membind=1 taskset -c 18-35 python bench.py bw 4 1,4,8,12,18 node1-local-r2
numactl --membind=0 taskset -c 0-17  python bench.py bw 4 1,4,8,12,18 node0-local-r2
numactl --membind=1 taskset -c 18-35 python bench.py perf 384 1,4,8,12,16,18 1,2,6 node1-local-r2
numactl --membind=0 taskset -c 0-17  python bench.py perf 384 8,12,18 1,2,6 node0-local-r2
numactl --membind=1 taskset -c 0-15  python bench.py perf 384 8,16 2,6 node0cores-node1mem       # r1
numactl --membind=1 taskset -c 18-35 python handoff.py run -1 12    # and 16
```
Each ran under `flock cc-gpu.lock` and exited 0.

## 29. Expert-stream I/O, hot path and recipe changes (2026-09-27 to 2026-09-29)

This section records the merges to `master` from 2026-09-27 to 2026-09-29, one subsection per merge, roughly in merge order.
Every figure comes from the file or merge commit message cited next to it (`git show <merge>` prints the message).
Decode figures are pooled ms/token over `run_arm.sh`'s 2 timed sessions (90 decode tokens), one pass per arm. The
campaign's noise bar is ±1.5 ms/token, so sub-millisecond deltas below are noise. The arms ran at different tiers and
memory fractions, so compare figures only within the pair or set that produced them.

**What the recipe runs now** (`benchmarks/dsv41_baseline/arm_env.py` at `1b8380eb6c`):

| Setting | Value | Since | Where |
|---|---|---|---|
| Expert-row mirror roots | 3: `/mnt/nvme0`, `/mnt/nvme4`, `/mnt/nvme2` (`*/dsv41_flash`) | 2026-09-28 | §29.3 |
| Mirror weights (`SGLANG_MOE_EXPERT_MIRROR_WEIGHTS`) | unset, i.e. 1:1:1 | unchanged | §29.10 |
| RAM-miss reader | row images, lease mode, unconditional; no pack workers | 2026-09-29 | §29.8 |
| `SGLANG_DSV41_ENABLE_LEASE_PDL` | 1 | 2026-09-28 | §29.2 |
| `SGLANG_EXPERT_STREAM_URING_*` | not set, so defaults: `MODE=default`, `READ_MODE=normal`, `FIXED_FILES=0`, `SLAB_ARENA=0`, `READ_CUTS=auto` (on only under IOPOLL) | unchanged | §29.4, §29.5 |
| Copy-thread completion | stream-written completion word, no `cuEventQuery`; no flag, no fallback | 2026-09-29 | §29.9 (**unverified**) |
| `MEM_FRACTION_STATIC` / `SGLANG_MOE_HOT_GPU_MB` / `CONTEXT_LENGTH` | 0.875 / 16080 / 131072 (0.885 / 15400 until the evening) | 2026-09-29 | §29.11, §29.15 |
| Copy wait (CW) | stream-ordered `cuStreamWaitValue32_v2` on a gate word; lease ABI 4 | 2026-09-29 | §29.14 |
| `--warmups dsv41_prefill_shapes` | on | 2026-09-29 | §29.16 |
| `--language-model-only` | on (no vision tower) | 2026-09-29 | §29.18 |
| HiCache | ratio 14 (~9 GB host), `page_first_direct`, direct IO, write_through | 2026-09-29 | §29.19 |
| `CUDA_MODULE_LOADING` | EAGER | unchanged | §29.11 |

**Also merged in the window, covered elsewhere or without measurements:** `e10d2ca067` and `43d7813a41` (layer-major
prefill, §27.19-§27.20); `6474fe3d36` (CPU experts, §28); `b5d50d0734` (the layer-fusion and cast-fusion kernels split
into generic expert-residency and EXL3 files, with byte-reproduction proofs in `analysis/layer-fusion-split/proof/`);
`4d8a9444cd` (expert-stream follow-ups: exact out-buffer checks, the double-signal counter, U7 and pack-thread flake
fixes); `b9d7b7f2f1` (prefill-OOM phase 0b minors).

### 29.1 Native sync primitives, lease kernel fixes, doorbell removal (2026-09-27/28)

**Native sync primitives (`7da74eb569`).** The expert-stream device code uses `__ldcv`/`__stcg` and `globaltimer` intrinsics,
named relaxed/acquire/release helpers, and acquire/release seqlock fences.
- **SASS:** identical to base except 16 `MEMBAR.SC.SYS` → `MEMBAR.ALL.SYS`. The copy wait's SmAck fence stays
  `MEMBAR.SC.SYS`.
- **`cuda::atomic_ref` was tried and reverted.** libcu++ adds a local-pointer check to every access through a
  `__grid_constant__` parameter, and it merged and reordered relaxed accesses.
- **Tests:** `unit/kernels` 1778 → 1784 passed on divix01; manual GPU tests 26 passed.
- Evidence: merge message; plan `docs/superpowers/plans/2026-09-27-expert-stream-native-sync.md`.

**Lease kernel fixes (`48467f2ade`),** from an external review of `lease_kernels.cuh`:
- The ack kernels' racy shared flag becomes `__syncthreads_or` (one `BAR.RED.OR`; the fence before the fatal release
  is kept).
- `rest_wait` no longer reads `claimed[]` past its 8 entries for a count above the bound (compute-sanitizer: 1
  invalid read at base, 0 on the branch), and counts every unserved lane once.
- The plain wait now ends on a fatal raised mid-poll instead of sitting out its timeout.
- Only `wait`, `stage_ack`, `lease_ack` and `rest_wait` change in SASS. Each fix's mutant turns its own tests red.
- The seqlock stress floor miss seen once is a load-dependent flake: 20/20 green alternating base and branch, torn 0.
- Evidence: merge message.

**Doorbell removal (`7f982d6be6`, leftovers `50bba481f1`).** The doorbell side-thread copier, superseded by the lease
protocol and unused in production, is gone with its ops module, tests, benchmarks and wiring.
- The eight `SGLANG_MOE_EXPERT_DOORBELL*` variables are deprecated: a set variable warns and does not fail.
  `--moe-offload-preset doorbell` is refused.
- The per-batch fail-stop hook, also used by the EXL3 RAM-miss path, is now `ExpertHotCacheManager.run_fail_stop_checks`.
- Cores 64-71 stay reserved, but the reason is now NVMe completion interrupts, not the
  doorbell's spin core.
- The offline prefetch pricing model loses its doorbell arms, and `price_prefetch.py`'s result JSON keys changed.
- Evidence: both merge messages. §8's doorbell row already carries the removal.

### 29.2 Lease-chain PDL, on in the recipe (`da243bbbfa`, 2026-09-28)

**What changed.** The lease chain (W1 → S → copy kernels) launches with programmatic dependent launch behind
`SGLANG_DSV41_ENABLE_LEASE_PDL`. W1 and S take the row-table capacity as a kernel argument.

**Result.** Decode A/B at `a2f0ae97b0` on driver 615.71.09: outputs byte-identical over 2/2 sessions, 101.5 → 101.6
ms/token on the 85-token session (+0.1, noise). `arm_env.py` summarizes it as "at most ~1 us/layer".

**In the recipe:** on (`SGLANG_DSV41_ENABLE_LEASE_PDL=1`).

**Evidence:** merge message. The results file `analysis/dsv41-drive/chain-pdl/results.md` is on branch
`expert-stream-transfer-measurement` (`6bba593810`) and **is not on `master`**; `arm_env.py` cites it there.

### 29.3 Three mirror roots via a per-row piece cut (`9428a6812a`, 2026-09-28)

**What changed.** Piece streaming cuts each reading part into its share of a row's 8 pieces, so a row can be read
from up to 8 mirror parts (tests cover 3 to 8); it refuses only more than 8 (`dc94f6724d`, `edfa9326e0`). The recipe adds `/mnt/nvme2` as a
third mirror root (`4fe0c37a41`).

**Result.** 2 vs 3 roots at one commit, both on the same reduced pinned tier: **109.9 → 101.7 ms/token** pooled,
byte-identical, reads ~33% per drive.
- The reduced tier is `0:57344,1:40960` / 98304 MiB, 4096 MiB smaller on node 0 than the recipe: the ZFS ARC holds
  node-0 memory that `host_numa.check_capacity` does not count as reclaimable, and the recipe's 0:61440 was refused
  1463 MiB short (`analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh`, header).

**In the recipe:** 3 roots (`EXPERT_MIRROR_DIRS`), equal weights.

**Known gaps.** No `results.md` for this pair is committed; the figures above are in the recipe commit's message
(`4fe0c37a41`) and in `arm_env.py`'s comment, with raw data at `divix01:/mnt/nvme1/mirror3-20260928-121453`.
`arm_env.py` says `/mnt/nvme2` is "now x4"; §1 lists it as Gen3 x2, and no link-width reading is recorded in the
analysis files.

**Evidence:** `4fe0c37a41`; `analysis/dsv41-drive/mirror3/` (driver, `mirror3_report.py`); plan
`docs/superpowers/plans/2026-09-28-mirror3-piece-stream.md`.

### 29.4 Reader CRTP split, io_uring registered buffers and files, 2 MiB NUMA splits (`e1559b948b`, 2026-09-28)

**Reader split.** `ReaderCore` plus `PackReader` / `RowReader` / `AnyReader`; the default path is unchanged (golden
unedited). Refactor pair on the reduced 98304 MiB tier: **102.40 (base) vs 102.56 (split) ms/token**, byte-identical
2/2, `read_errors` 0 in both. (`PackReader` was later deleted with the packed path, §29.8.) Evidence:
`analysis/dsv41-drive/reader-crtp/results.md`.

**Registered buffers and fixed files work, and stay off.** Registration uses row-aligned ≤1 GiB chunks; a fixed read
whose iovecs meet several chunks fans out into one SQE per chunk.
- R0 (defaults) vs R3 (`SLAB_ARENA=1 READ_MODE=readv_fixed FIXED_FILES=1`), full tier `0:61440,1:40960` / 102400:
  **101.46 vs 101.65 ms/token**, byte-identical. The rule needed R3 to win by ≥1.5 ms/token, so the defaults stay.
- R3 registered 280 chunks, 107,363,553,792 B, in **59,231 ms**. First log line to "fired up": 113 s (R0) vs 172 s
  (R3). 33,736 of 101,208 fixed reads (33%) fanned out across chunks.
- Evidence: `analysis/dsv41-drive/uring-reg/results.md`.

**2 MiB-aligned NUMA splits (Task 10, `40452dacc2`).** The first full-tier R3 hung in registration for 13+ minutes.
- **Cause:** quadratic pin accounting in the 6.12 kernel's buffer registration (`io_buffer_account_pin` →
  `headpage_already_acct`). `host_numa.allocate_bound` placed its per-node `mbind` splits on page-rounded row
  boundaries, so chunks spanning them mixed folio sizes and did not coalesce.
- **Fix:** 2 MiB-aligned mapping bases and node-change boundaries (`host_numa.plan_bindings`). Harness registration at
  the 90 GiB layout fell from an extrapolated hour to 58.4 s.
- **Check on the placement change:** R0 before vs after, full tier, 101.89 → 101.46 ms/token, byte-identical.
- The kernel frame is inferred from the v6.12 source and the linear-per-chunk growth; kernel stacks are root-only.
- Evidence: `analysis/dsv41-drive/uring-reg/results.md`, "How we got here".

**Configuration surface.** The reader's ring is configured by ten `SGLANG_EXPERT_STREAM_URING_*` variables (`MODE`,
`QUEUE_DEPTH`, `FIXED_FILES`, `READ_MODE`, `WAIT_MODE`, `SQ_THREAD_IDLE_MS`, `SQ_THREAD_CPU`, `DIAGNOSTICS`,
`READ_CUTS` (added in §29.5), `SLAB_ARENA`), all defaulting to the previous behavior. Evidence: `analysis/dsv41-drive/uring-config/HANDOFF.md`
(its "2026-09-28 update" paragraph is the corrected account of registration cost; §29.6).

**Merge-time suite:** kernels 1911 passed / 23 skipped (merge message).

### 29.5 IOPOLL wait fix, read cuts, 10 s SQPOLL idle (`ba01695c35`, 2026-09-28)

**Why IOPOLL was slower** (`analysis/dsv41-drive/iopoll/diagnosis.md`). Two effects that only hurt together:
1. **Every production read punted to io-wq.** One row-image sub-read is a ~2.2 MB `READV`; the drives take at most
   512 KiB (Samsung 990 EVO Plus) or 256 KiB (SPCC) per request, and a non-page-aligned iovec join crosses the NVMe
   `virt_boundary`. The block layer will not split a `REQ_NOWAIT` polled bio, so it returns `-EAGAIN` and the read
   is re-issued from io-wq.
2. **The punted reads queue behind the waiter's lock.** `WAIT_MODE=block` on an IOPOLL ring polls inside
   `io_uring_enter(GETEVENTS, min_complete=n)` holding `uring_lock`, and each io-wq worker spins for it. Row p50 went
   1.70 → 3.2 ms in the microbenchmark.
The kernel paths are from upstream source and fit every measurement; they are not confirmed by kernel stacks
(root-only).

**What changed.**
- An IOPOLL ring without SQPOLL now reaps (`effective_wait=reap`) instead of block-waiting.
- `READ_CUTS=auto` cuts READV legs at each drive's `max_sectors_kb` and at `virt_boundary` gaps, on only under IOPOLL,
  so nothing punts.
- `SQ_THREAD_IDLE_MS` defaults to 10000 (1000 before 2026-09-28; user decision, `uring-config/HANDOFF.md`). It
  applies only to SQPOLL modes, which the recipe does not use.

**Result** (full tier, all byte-identical, 0 io-wq workers over every whole run; `analysis/dsv41-drive/iopoll-cuts/results.md`):

| Arm | Setting | ms/token | Δ vs mean(A, A2) = 100.61 |
|---|---|---:|---:|
| A | default, cuts off | 100.95 | +0.34 |
| B | default, cuts on | 100.75 | +0.14 |
| C | `iopoll`, cuts on, reap wait | 101.23 | +0.62 (was +10.2 ms/token in the Codex S4 arm, with 12-14 io-wq workers) |
| D | `sqpoll_iopoll`, cuts on | 100.79 | +0.18; the SQ thread used 24.34 CPU-s in a ~24.5 s window |
| A2 | default, cuts off | 100.27 | −0.34 |

- **In the recipe:** unchanged, `MODE=default`, `READ_CUTS=auto`. Nothing reached the −1.5 ms/token promotion bar.
  What holds is "C removed the +10 ms regression", not "C is 0.6 ms slower".
- Mutants M1-M9 killed; kernels suite 1933 / 23 (merge message; `iopoll-cuts/mutants.md`).
- **Open:** follow-ups #7 (fold the IOPOLL reap loop's peek and `get_events` into one call; cold path), #8 (test gaps:
  EOF-clamped cut leg, files on two devices with different limits, fallback wording), #9 (the `read cuts:` line prints
  at every reader `open()`). The SPCC's slow episodes, below in §29.13.

### 29.6 THP fallback and the clone-buffers pin leak on kernel 6.12.0-211 (`af765201a3`, 2026-09-29; write-up only)

**Do not use `IORING_REGISTER_CLONE_BUFFERS` on this kernel (6.12.0-211.60.1.el10_2). Fixed buffers are off by
default, and registration cost is paid only by an arm that turns them on.**

**Root cause, refined.** Registration is slow because the kernel's pin accounting walks, for every huge page of a new
chunk, every bvec of every buffer registered before it. A chunk holding any 4 KiB page does not coalesce and keeps
262,144 bvecs per GiB.
- What costs time is *where* the 4 KiB pages sit relative to the huge pages registered after them, not how many: 0.18%
  of a 16 GiB tier, in 15 chunks, takes registration from 0.52 s to 68.75 s.
- Each mixed 1 GiB chunk costs ~0.85 s per GiB of THP registered after it.
- In production 8-22% of the 100 GiB tier lands on 4 KiB pages, depending on fragmentation at launch; direct
  registration took 18.9-103.5 s across these launches.
- Even an all-THP 100 GiB tier registers quadratically, 12-21 s by the per-chunk model (~19 s).

**Mitigations that failed at full size:** `MADV_HUGEPAGE` (4× slower registration: 103.5 s), re-faulting (0 of 4,440
frames recovered), `MADV_COLLAPSE` (3,839 of 3,982 calls failed), reordering (18.6 s vs 18.9 s), 64 MiB chunks
(18.9 s). Node 0 has too few order-9 blocks.

**The clone fix and the leak.** Registering each chunk in a scratch ring and cloning it in cut 18.9 s to 1.1 s, but
the pinned pages stay allocated after both rings and the process are gone.
- Controlled repro, 2 GiB tier: direct −2 MiB orphaned, clone **+1025 MiB**.
- Five 100 GiB clone registrations stranded **~149.5 GiB**, node-0 MemFree fell to 670 MiB with 41 GiB of swap in use,
  and only the 2026-09-29 02:02 reboot freed it (§29.12).
- The fix was reverted (`f82f61f3d7`). No code change lands; the merge also corrects the overclaim in
  `uring-config/HANDOFF.md`.
- The kernel source was not read, so which reference leaks is not established.

**Options not tried, predicted:** split 4 KiB rows into their own chunks registered last (8-13 s), leave mixed chunks
unregistered (7-12 s), 4 KiB-only tier (`MADV_NOHUGEPAGE`, ~0 walk; decode cost unmeasured), 1 GiB hugetlbfs (root).
Registering mixed chunks first is the worst order (900-2,700 s).

**Evidence:** `analysis/dsv41-drive/thp-fallback/results.md`.

### 29.7 Ring reset drains unconsumed SQEs as NOPs (`65754399e3`, 2026-09-29)

**The defect.** When a failed submit left SQEs the kernel never consumed, `UringReader::drain` closed and re-created
the ring and re-registered every fixed buffer and file: 31-53 s at a 90 GiB tier by extrapolation, past the 30 s
`fatal_wait`, so the process would abort with "a request stayed in service".

**The fix.** Without SQPOLL the unconsumed SQEs still belong to userspace, so `drain()` zeroes them, rewrites them as
`IORING_OP_NOP`, and submits and retires them on the same ring. In a fixed read mode a refused NOP drain throws with the
cause instead of the slow reset; other modes keep the cheap reset.

**Result** (divix01 after the reboot, 1,613,631,488 B slab in 3 chunks):

| MODE / READ_MODE | drain_ms before → after | registrations |
|---|---|---|
| default / fixed | 595.5 → 0.011 | 2 → 1 |
| default / readv_fixed | 898.6 → 0.017 | 2 → 1 |
| iopoll / readv_fixed | 780.1 → 0.012 | 2 → 1 |

- Kernels suite 1933 → 1949 passed / 23 skipped (+16 new). Mutants M1, M2, M4-M8 killed; M3 (no `memset`) survives on
  the real kernel and is caught by the fake-liburing contract.
- **Open, pre-existing:** `test_big_fixed_slab_real_kernel` with `MODE=iopoll` times out at 300 s at master too.
- **In the recipe:** it matters only when fixed buffers are on, which they are not.
- Evidence: `analysis/dsv41-drive/ringreset/results.md`; merge message.

### 29.8 Zero-overhead hot path (`2810a7ad48`, merged at `b35dc14de4`, 2026-09-29)

**What changed** (plan `docs/superpowers/plans/2026-09-29-hotpath-zero-overhead.md`):
- **The packed path is deleted.** Row images are the only reader, and lease mode is unconditional.
  `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES`, `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES` and
  `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` are deprecated: a set value warns and is ignored, and a checkpoint without row
  images refuses at startup (`python/sglang/srt/environ.py`).
- **Metrics compile out.** `ProdBuild` removes stats, trace and fault state from the production instantiation;
  `InstrBuild` is picked when a trace or fault env var is set.
- **No tier mutex.** The service thread owns the tier state. Copy completions and unpaused Python calls reach it
  through SPSC rings, and the copy engine's queue is an SPSC ring with a futex wake only when its thread sleeps.
- Clockless pacing, and `FixedVec` in place of per-request containers.

**Result.**
- **Decode A/B/A2** (master `65754399e3` / branch / master): **100.52 / 99.99 / 100.10 ms/token**, byte-identical,
  `read_errors` 0. A later pass of the branch at `1d522c5207`: 99.93. Evidence: `analysis/dsv41-drive/hotpath/results.md`
  §8a, §8e.
- **Service thread:** ~16-19% fewer instructions per served request (33.5 M vs 41.5 M / 40.0 M; 17.8% vs mean(A, A2);
  the merge message says −18%) over the same cycles, which are its idle loop (§8b). TSan clean (merge message).
- **Counts per request, CPU shim, prod build** (§5a): service malloc 14.00 → 0, mutex 1758.29 → 0, clock 4958.40 → 0;
  copy malloc 2.65 → 0, mutex 2.65 → 0.
- **Whole run, production server under the shim** (§8e, branch CS2 vs master CM): service malloc 159,085 → 0 and
  mutex 114,655,774 → 0; copy malloc 136,131,275 → 72 (start-up only). Copy mutexes 110,094,979 → 36,594,783, all of
  them libcuda's.

**P1 attribution (`hotpath/results.md` §9a, arm CS3).** The copy thread's remaining mutexes are libcuda's:
- **98.5% are `cuEventQuery` polling:** 35,082,931 [95% CI 34.99 M, 35.17 M] of 35,603,559.
- Per job that is **~2,280-3,215**, over two estimates of the job count (15.4 k and 10.9 k), since production does not
  count jobs. The merge message quotes ~3,215, the 10.9 k end.
- Submission (`cuMemcpyAsync`, 8 mutexes per call; `cuEventRecord`, 4) is ~1.5% of the total.

**Suite counts are not comparable across this merge.** The kernels suite collects 1956 → 1140 IDs (1119 passed / 21
skipped at `2a3aaee887`, 1124 / 21 at `c40e834e96`): 944 packed-path IDs were deleted and 128 added (`hotpath/results.md`
§1a, §8f).

**Open:** the copy thread's libcuda traffic, which §29.9 addresses; `test_the_seqlock_reader_never_accepts_a_torn_record`
misses its throughput floor 10/200 at head vs 9/200 at base, with 0 torn records (§6).

### 29.9 Completion word, P2 (`688cc004d8`, 2026-09-29): merged unverified

> **Open risk. This merge went in at the user's request without review, mutants, TSan or a decode A/B** (merge
> message). It is unconditional: whenever the copy engine is on, as in the recipe, the copy thread uses it, and there
> is no event fallback. Any production checkout updated past `688cc004d8` runs it.

**What changed** (tests `65bf4610f6`, implementation `fcdaff8230`):
- `CudaCopyBackend::mark` writes the job's 32-bit sequence into a host-mapped pinned word with
  `cuStreamWriteValue32_v2` (default flags: after the stream's prior copies, fenced), resolved by `dlsym`.
- `query()` is one acquire load of the word with a wrap-safe compare and no driver call. The event pool and
  `cuEventRecord`/`cuEventQuery` are gone, and the head is polled every turn (the hot path's 1-in-8 cadence existed only
  to ration `cuEventQuery`).
- Liveness without a clock: one `cuStreamQuery` per 2^16 consecutive pending polls. An error fails stop; an idle
  stream fails stop only if a re-load of the word is still short.
- `init()` refuses the start if the v2 op cannot be resolved, errors, or its first write does not land.
- The word's final value, the exact job count, is logged at shutdown.

**What supports the design** (probe on divix01's RTX 5090, driver 615.71.09; `hotpath/results.md` §9b-§9d):
- The v2 write-value ops are supported (`CAN_USE_64_BIT_STREAM_MEM_OPS` = 1; the v1 attributes read 0).
- Stream order: 0 violations in 2,500 jobs, where an unordered control fails 500/500.
- Submit to seen: 459.8 µs vs 460.2 µs for an every-turn `cuEventQuery` poller. A word load costs 0.27 ns; a
  `cuEventQuery` on a completed event 118.6 ns and 1 mutex.
- Predicted copy-thread mutexes per job: ~2,320-3,260 → ~37-49 [estimate]. `cuMemcpyAsync`'s 8 mutexes per call remain.

**Not tested before the merge,** named in §9b-§9c as P2's gates: the `dlsym` path (the probe linked `-lcuda`), a
copy-thread clock count of 0 for the write and `cuStreamQuery`, the forced interleaving where the word lands between the
load and a `SUCCESS` query, and any decode arm.

### 29.10 SPCC mirror weight 1:0.9:1: no gain (`7d4c0eeb69`, 2026-09-29; analysis only)

**Question.** Does down-weighting the DRAM-less SPCC mirror (`/mnt/nvme4`) improve decode?

**Result.** W (1:0.9:1) is **+0.38 ms/token** against U (1:1:1) over 3 alternating pairs (sd 0.46, range +0.01 to
+0.89); means 100.77 vs 100.39. Output byte-identical in every arm.
- The weights take effect: the SPCC's share falls 33.2% → 30.9%. RAM misses per token are unchanged (8.94-9.03).
- The SPCC's device time per read request falls 7.0 → 5.6 ms, while the Samsungs rise (nvme0 7.5 → 8.2, nvme2 4.3 →
  4.8 ms). Under decode's concurrent reads the SPCC is not the straggler the QD1 bench measured.
- The predicted −1.9 ms/token gain is excluded by all three pairs.

**In the recipe:** equal weights (unset).

**Deviation.** Every arm ran at `DSV41_MEM_FRACTION_STATIC=0.91`, because 0.90 was refused at KV sizing before and
after the reboot (available 0.14 / 0.17 GB against a 0.23 GB SWA floor). This is the problem §29.11 fixes.

**Evidence:** `analysis/dsv41-drive/mirror-scaling/weight-pair.md`.

### 29.11 Recipe GPU memory for driver 615.71.09 (`cfe4d822da`, 2026-09-29)

**What changed.** `MEM_FRACTION_STATIC` 0.90 → **0.885**, `SGLANG_MOE_HOT_GPU_MB` 16100 → **15400**, `CONTEXT_LENGTH`
262144 → **131072** (`benchmarks/dsv41_baseline/arm_env.py`). §27.17's recipe line carries a superseded note.
(Superseded the same evening: 0.875 / 16080 after a 419-token prompt OOMed at 0.885, §29.15.)

**Cause: the driver, not the code** (`analysis/dsv41-drive/recipe-mem/diagnosis.md`).
- NVIDIA driver 610.57.04 → 615.71.09 (dnf transaction 477, 2026-09-27 23:11, live after the 09-28 00:30 reboot) made
  the pre-load footprint larger and jittery. "Load weight begin" fell from 29.90-29.91 GB to 29.21-29.45 GB.
- Bare-context probe under 615.71.09: EAGER 1961 / 1973 / 1903 MiB, LAZY 697 / 697 MiB.
- Sources differ on the size: the diagnosis and `arm_env.py` say ~0.5 GiB; the merge message says ~0.5-0.7 GB. The
  diagnosis's own footprint figures, 2.0-2.24 GiB against a steady 1.54 GiB before, span both.
- At 0.90 all of it came out of the KV pool: available 0.80 → 0.14-0.38 GB against the 0.23 GB SWA floor, so some
  launches were refused. The 09-27 code on today's driver behaves like master, which clears the code.
- 0.91 / 16100 restored the pool but a 16k layer-major prompt OOMed; 0.895 / 15400 put the cut into the KV pool and a
  chunked 16k prompt needed 2 allocator retries with 1 MiB free at peak.

**Verification at 0.885 / 15400** (`analysis/dsv41-drive/recipe-mem/results.md`). 9 launches: KV available 0.46-0.59
GB, KV pool **150,784-232,448 tokens** (against ~387k at 0.90 on the old driver), 2.67-2.72 GB free after decode graph
capture. Prefill, 4096-token chunks, two runs each (free = 32,202 MiB − peak `memory.used`):

| prompt | run 1: TTFT s, peak MiB, free MiB, retries | run 2: TTFT s, peak MiB, free MiB, retries |
|---|---|---|
| layer-major 16,384 | 83.9, 31,589, 613, 0 | 83.9, 31,607, 595, 0 |
| layer-major 65,536 | 339.9, 31,649, 553, 0 | 322.9, 31,665, 537, 0 |
| chunked 16,384 | 88.9, 31,973, **229**, 0 | 89.2, 31,969, **233**, 0 |
| chunked 65,536 | 352.8, 31,949, 253, 0 | 352.6, 31,945, 257, 0 |

- No OOM and no allocator retry. The chunked path is the tightest, and it is the path every prompt under 8,192
  uncached tokens takes.
- **CUDA reaches 32,202 MiB, not 32,150.** §27.17 and §27.18 are corrected.

**Decode cost:** 102.44 ms/token at 0.885 / 15400 against 100.58 at 0.91 / 16100, **~+1.9 ms/token**, one arm each,
byte-identical. Treat it as an estimate against the weight pairs' ±0.3 spread.

**Open.**
- **The startup pool check at 131072 was cancelled and not run.** No launch has shown that the server accepts context
  131072 or that the pool at the merged commit is at or above it; the margin rests on the 9 launches above, smallest
  pool 150,784 tokens.
- A startup eager-prefill warm-up before KV sizing, so the profile sees what serving uses, is recommended and not
  implemented (diagnosis, Q2).
- Keep `CUDA_MODULE_LOADING=EAGER`: LAZY would return ~1.3 GiB but reopens LEASE_PROTOCOL 7.6's unguarded fail-stop.

### 29.12 divix01 environment changes (2026-09-27 to 2026-09-29)

- **NVIDIA driver 610.57.04 → 615.71.09** (dnf transaction 477, 2026-09-27 23:11), with the kernel moving to
  `6.12.0-211.60.1.el10_2` and system cuDNN to 9.26 (torch loads the venv's cuDNN 9.20). Live after the 2026-09-28 00:30
  reboot. The driver cannot be rolled back without root. Evidence: `recipe-mem/diagnosis.md`.
- **Reboot 2026-09-29 02:02 CDT** to free the ~149-150 GiB of page pins the clone-buffers experiment stranded (§29.6).
  Evidence: `mirror-scaling/weight-pair.md` (timeline), `thp-fallback/results.md` (incident), `ringreset/results.md`
  (timings re-taken after it).
- **Device names moved at that reboot; the mount-to-drive mapping did not.** Before it (2026-09-28): `/mnt/nvme0` =
  nvme0n1 (Samsung 990 EVO Plus, xfs, 512 KiB max request), `/mnt/nvme4` = nvme2n1 (SPCC, ext4, 256 KiB), `/mnt/nvme2`
  = nvme3n1 (Samsung 990 EVO Plus, xfs, 512 KiB) (`iopoll/diagnosis.md`, "Host facts"). After it, `/mnt/nvme2` is
  nvme1n1 and the SPCC is still nvme2n1 (`mirror-scaling/weight-pair.md`, "How it ran"). Resolve drives from the mounts,
  never from a remembered `/dev` name. The post-reboot name of `/mnt/nvme0`'s drive is not recorded.
- **Host I/O facts (2026-09-28):** liburing 2.12, `nvme.poll_queues=1`, `io_poll=1` on all four NVMe namespaces, THP
  `enabled=always`, `defrag=madvise`, `perf_event_paranoid=2` (no kernel profiling), `bpftrace` absent
  (`iopoll/diagnosis.md`, `thp-fallback/results.md`, `uring-config/HANDOFF.md`).
- **Node-0 memory:** the ZFS ARC holds node-0 memory that `check_capacity` does not count as reclaimable, and some arms
  ran a 98304 MiB tier for it (§29.3).

### 29.13 Open follow-ups from these results

1. **Verify P2** (§29.9): review, mutants, TSan, the `dlsym` path, the clock-count gate, the forced interleaving, and a
   decode A/B.
2. **Guard the pool against the context.** Every launch since has accepted context 131072, but at 0.875 / 15400 the
   server also started with a 7,936-token pool, so nothing refuses a pool below the context (§29.15).
3. **A startup warm-up before KV sizing** (§29.11) is still not built. The post-sizing `--warmups` (§29.16) moves the
   kernel loads before traffic but does not change what KV sizing sees.
4. **Fixed buffers stay off.** Before re-enabling: never the clone path on 6.12.0-211; the untried options are in
   §29.6. Any future clone-based fix must check the system-wide orphan count (`thp-fallback/results.md`).
5. **IOPOLL follow-ups #7-#9** and the pre-existing IOPOLL `real_kernel` big-slab 300 s timeout (§29.5, §29.7).
6. **The SPCC's slow episodes:** in 7 of 24 multi-row microbenchmark runs its per-SQE p50 was 10-35 ms; it is
   DRAM-less with a 64 MiB HMB over a 191 GB span, and smart-log reads healthy (`iopoll-cuts/results.md`). Weighting it
   down does not help (§29.10).
7. **nvme0 sees half nvme2's IOPS at the same MB/s** (requests twice the size); not investigated
   (`mirror-scaling/weight-pair.md`).
8. **The copy thread's submission floor:** `cuMemcpyAsync` takes 8 libcuda mutexes per call; fewer calls per job
   (`cuMemcpyBatchAsync`) is not measured (`hotpath/results.md` §9d).
9. **The chain-PDL results file** is not on `master` (§29.2).
10. **Validate the stream-ordered copy wait** (§29.14): run its tests, a decode A/B against the spinning wait, and a
    node-mode trace re-run. Also check the head-of-queue risk.
11. **Measure the hot cache at 16080** against 15080 (decode ms/token), and the prefill headroom (0.16 GiB minimum in
    the warmup) on a 16k chunked prompt (§29.15).
12. **HiCache on the direct backend** (§29.19): measure revisit TTFT and whether its backups slow prefill or decode
    alongside the RAM-miss copy engine.
13. **`get_and_clear_swa_pages_kernel` still loads after startup** (§29.16).

### 29.14 Stream-ordered copy wait (`c7ce4aa337`, `264ddece72`, `d835f1b045`, 2026-09-29)

**The problem: node-mode Nsight traces of the prod recipe hung** (`divix01:/mnt/nvme1/dsv41-nsys/prod-node-trace-20260929/results.md`).
- With `--cuda-graph-trace=node`, the RAM-miss request never completed once capture started. The server fail-stopped at
  the RAM-miss timeout: 2 s at the recipe value, and exactly 30 s with `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS=30000`.
- **Not P2:** the bisect at `7d4c0eeb69`, before the completion word (§29.9), hangs the same way.
- **The copy engine is required:** the same capture with `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=0` (and
  `SM_SMALL_COPIES=0`, which the startup check then requires) ran 1,110 decode steps cleanly.
- **Hang shape (Case A):**
  - The GPU sat at 100% SM, 0% memory utilisation, ~176 W.
  - `exl3-copy-eng` was live, polling in its own loop, with no libcuda, CUPTI or ioctl frames.
  - The scheduler was blocked in `copy_done.synchronize()`.
  - No thread was blocked on a CUDA/CUPTI mutex.
- **Reading:** under CUPTI node tracing, the copy stream's DMA and its stream-written completion word never ran
  behind CW's spinning wait kernel. In production, the same spin cost one SM per waiting layer.

**The fix: CW no longer spins.** It is three stream-ordered steps:
1. **Arm kernel (PDL).** It closes a u32 gate word in lease area C and publishes `CopyArm = tagged(G)`. After a
   system fence it re-reads `CopyDone`; if that already carries G with the exact lane mask, it opens the gate itself.
2. **`cuStreamWaitValue32_v2(gate, GEQ open)`.** This holds the decode stream; it is a memop node in the graph with
   no programmatic edge.
3. **Plain commit kernel.** It checks the gate outcome, `CopyDone == G` and the lane mask, never waits, and fails
   closed on the fatal word.

**Gate word:** `seq << 2 | state` (29-bit seq), with bit 31 set when closed. Read as an int32, a closed word is negative,
so the GEQ wait holds.

**Releasers.** Each host releaser opens the gate only by a CAS from G's exact closed word:
- the copy thread, after `CopyDone`;
- the service, on every `pump_demand`;
- `close_admission`;
- the RamThread watchdog, which owns the copy-wait deadline: fatal word first, then gate = timeout.

A releaser stalled past CW's own open of G, and past G + 1's close, therefore changes nothing.

**Shutdown and startup.**
- `stop_thread` and `close` raise the shutdown word and open an armed gate (`abort_copy_waits`) before joining.
- `start_thread` is refused after an abort.
- The copy thread refuses to start on a driver or device where a v2 stream wait on host-mapped memory does not
  both hold and release. The probe uses the production closed word, with bit 31 set.

**ABI and docs.** Lease block ABI 3 → 4 (gate and `CopyArm` lines in area C). `LEASE_PROTOCOL.md` 7.6 documents the
new wait interval: it is watchdog-timed from the arm, so post-to-fail-stop can take ~2x the timeout plus 20 ms.

**Evidence.**
- Every launch since `d835f1b045` passed the init probe and captured `segments=1 breaks=0`.
- RAM misses were served through the new wait with `copy_errors: 0`. The counters at the 16:43 OOM read `served` 616
  and `rows_read` 1,324; this is `logs/prod-server-20260929-163831-streamwait.log`.
- A user session on `1b8380eb6c` decoded 121 log intervals at a mean 10.35 gen tok/s (max 13.03) with no RAM-miss
  timeout (`logs/prod-server-20260929-180120-lmonly-hicache14.log`).

**Not done.**
- The commits' tests were updated but not run.
- No decode A/B against the spinning wait.
- No node-mode trace re-run to confirm the hang is gone.
- Open risk (`LEASE_PROTOCOL.md`): a wait node at the head of a hardware queue may block other streams on that queue.

### 29.15 Recipe memory after a 419-token OOM (`08afb8561e`, `89ef36ede0`, then `3f6154c8a5`, 2026-09-29)

**At 0.885 / 15400, a 419-token prompt OOMed in prefill.**
- Free memory fell from 2.70 GB after capture to 37 MB, and a 256 MiB `exl3_linear` output could not be allocated.
- Late Triton kernel loads came first. Earlier launches had bottomed out at 0.12 GB.
- Log: `logs/prod-server-20260929-163831-streamwait.log`.

**The KV pool is only the remainder of the fraction.**
- `MEM_FRACTION_STATIC` 0.875 alone shrank the pool from 161,536 to **7,936 tokens**, with 3.12 GB free after capture
  (`...-165242-mf0875.log`).
- The server still started with `context_len=131072`, so **nothing refuses a pool smaller than the context**.
- The ~320 MiB was taken from the hot cache instead, 15400 → 15080 (`89ef36ede0`). Pool 225,792, 2.96 GB free
  (`...-165638-hot15080.log`).

**Hot cache 15080 → 16080 with `--language-model-only`** (§29.18). The freed vision memory went to the hot cache, and
the KV pool held.

Launches on 2026-09-29, 16:38-18:01:

| Log (`logs/prod-server-20260929-*`) | Recipe | KV pool, tokens | Free after capture | Outcome |
|---|---|---|---|---|
| `163831-streamwait` | 0.885 / 15400 | 161,536 | 2.70 GB | OOM on a 419-token prompt |
| `165242-mf0875` | 0.875 / 15400 | 7,936 | 3.12 GB | pool collapsed |
| `165638-hot15080` | 0.875 / 15080 | 225,792 | 2.96 GB | ok |
| `172700-warmup` | + warmup | 177,408 | 3.00 GB | grammar request crashed (§29.17) |
| `173859-grammarsync` | + sampler fix | 195,840 | 2.99 GB | ok |
| `174941-lmonly` | + language-model-only, 16080 | - | - | assert at decode capture (§29.18) |
| `175813-lmonly-hicache` | + `--hicache-size 10` | 272,128 | 2.86 GB | refused at tree-cache init (§29.19) |
| `180120-lmonly-hicache14` | + ratio 14 | 238,080 | 2.85 GB | ok, current |

**Headroom is thinner at 16080.** The lowest free VRAM during the warmup was 0.16 GiB, against 0.27 GiB at 15080. The
4096-token chunk and the grammar request both passed. If a long prompt OOMs, move MiB back from the hot cache.

### 29.16 Startup warmup of prefill Triton variants (`2cc4313906`, grammar request `5c4725dc90`)

**Why kernels loaded late.** Prefill Triton kernels compile one variant per token-count class, and a class first seen
mid-serving loads its cubin with ~0.2 GiB free.
- The class comes from `_block_m_for` in `mhc.py`, the M thresholds in `hc_combine_norm`, and Triton's own
  specialization on M == 1 and M % 16.
- The default 193-token server warmup covers one class.
- The load watcher (`utils/triton_load_watch.py`) warns only below 1 GiB free.

**What it does.** `--warmups dsv41_prefill_shapes` (`entrypoints/warmup.py`) sends one `max_new_tokens=1` prompt per
class before the server listens: sizes 1, 5, 8, 16, 33, 64, 257, 400, 2048, 2049 and 4096. It then sends a 16-token
regex-constrained request, which loads xgrammar's bitmask kernel and exercises the grammar token sync (§29.17).

**Cost and effect.**
- The 11 prompts take ~71 s of startup, and the prompts of 256 tokens or more seed the hot cache.
- Their loads now land during startup. `_fwd_kernel` loads during the built-in 193-token warmup, still before
  "fired up".
- The recipe passes the flag to prod and to the arms alike, keeping their argv identical.

**Still late:** `get_and_clear_swa_pages_kernel` loaded after "fired up" at 0.16 GiB free
(`...-180120-lmonly-hicache14.log`, 18:04:55).

### 29.17 Grammar requests crashed the single-GPU server (`5c4725dc90`, `1cfdf9cc59`)

**What happened.** A structured-output request died in sampling with `NCCL ... Failed to CUDA calloc 536870912 bytes`
(`...-172700-warmup.log`).

**Cause.** `Sampler._sync_token_ids_across_tp` all-reduces the next token ids whenever a batch has grammars.
- On one TP rank that all-reduce is an identity.
- Its first call creates the NCCL communicator, which allocated 512 MiB with 0.27 GiB free.
- None of the ungrammared warmups reached it.

**Fix** (`layers/sampler.py`). The sync runs only at a TP world size above 1, the guard the status sync in the same
file already had. The grammar manager's own all-gather already skipped size 1.

**Test.** `test/registered/unit/sampling/test_sampler_token_sync.py`: 3 passed on divix01. With the guard removed it
fails 1 of 3; restored, it passes 3 of 3.

### 29.18 The vision tower is not built (`3f6154c8a5`, `dde35cecd9`)

**What was wasted.** The checkpoint keeps `vision_config` (32 blocks, dim 1024) but ships no vision weights.
- The ViT (~411M params) and aligner (~73M) were built empty: ~0.97 GB in bf16. Startup logged "Some weights are not
  initialized: vision.blocks...".
- The multimodal path also reserved 0.10 GB of the KV budget.

**A correctness bug with it.** The 40 per-layer `e_score_correction_bias_vl` parameters were left as `torch.empty`,
because the loader keeps them whenever the tower exists. The fused gate applies them to any token equal to
`image_token_id` 129264 (`multimodal/dsv41/vl_routing.py`).

**Fix.**
- `--language-model-only` now sets `vision_n_layers = 0` for a V4.1 config in `ModelConfig`. Every vision site keys
  off that value: the ViT and aligner, the VL bias, and the Engram image-token bypass.
- `DeepseekV4ForCausalLM` joins the flag's allowlist (`server_args.py`).
- The recipe passes the flag.
- **Result:** "Load weight end" `mem usage` fell 9.98 → **9.04 GB**, and the uninitialized-weights warning is gone.

**What it exposed.** V4.1 had only ever routed through `vision_topk`.
- Without the VL bias, the MoE fell back to `self.topk`. Under `moe_runner_backend=flashinfer_mxfp4` with non-FP4
  (EXL3) experts, that emits the BYPASSED format.
- The expert layer's `format_is_standard` assert then failed at decode capture (`...-174941-lmonly.log`).
- `dde35cecd9` keeps every V4.1 layer on `vision_topk`, with no `input_ids` when there is no VL bias. That turns the
  fused gate's image-token switch off, so text-token routing is the same kernel call as before.
- V4.1 has no hash layers, which `vision_topk` does not implement.
- DSpark's draft layers, previously on `self.topk`, now take `vision_topk` too. That path is not exercised in
  production.

### 29.19 HiCache host pool ~9 GB, direct IO (`ea4c01b618`, `1b8380eb6c`)

**Size.** DeepSeek V4 HiCache refuses `--hicache-size` at tree-cache init (`_deepseek_v4_num_host_pages`), so the host
pool is sized by ratio instead.
- Each sub-pool's host pages are ratio x device pages.
- Ratio 2 held 1.14 GB at a 195,840-token pool.
- Each unit is ~0.25 GB of SWA plus ~1.6 KB per GPU pool token.
- **`--hicache-ratio 14`** gives ~6.7-10.6 GB over the 140k-311k-token pools seen, and **8.96 GB** at 238,080. With it,
  available system RAM went 45 → 38 GB.

**Layout.** `--hicache-mem-layout page_first_direct` with `--hicache-io-backend kernel` is rewritten to the direct
backend (`hicache_hook.resolve_layout_io_compatibility`).
- DSV4 backups then use `transfer_kv_all_layer_direct_lf_pf`, which issues cudaMemcpy DMA on the copy engines the
  RAM-miss path also uses.
- This drops the staged write-back kernel that the page_first/kernel pair used.
- **Not measured:** prefill or revisit TTFT against §27.16's ratio 2, page_first/kernel.

### 29.20 Operations

- **`/data/models/slang/nvfp4-work/server.log`** always names the running production log. After taking `cc-gpu.lock`,
  `launch_prod.sh` renames a fresh symlink over it to its stdout's file (`d804ce7a5f`, `SERVER_LOG_LINK` overrides the
  path). A refused start, a terminal or a pipe leaves the link alone.
- **Restart only an idle server.** A SIGTERM during a request drains it and holds `cc-gpu.lock`, so the next launch
  is refused. Check the log's last `#running-req` first.

## 30. CPU experts, the minimal lease protocol, and the host test exports (`4e07d261bd`, 2026-09-30)

One merge of `dsv41-cpu-experts` at `1ff55fb37b`. It carries three things:
- CPU experts, off by default (§30.1-§30.4);
- the minimal lease protocol, which is now the default path (§30.5);
- the move of the host test and tool exports into their own header (§30.6).

Design detail and the P0-P2 records are in `docs/superpowers/plans/2026-09-29-dsv41-cpu-experts.md`. Every figure
below comes from that plan, a commit message, or the divix01 file cited next to it.

**Changes to §29's recipe** (the table's copy-wait row, and `arm_env.py`).
- **Copy wait:** still the stream-ordered gate wait. `CopyArm` folds into the gate, the fatal word, status words and
  lease-block header are gone, and the lease block's offsets are compile-time constants.
  `analysis/dsv41-drive/LEASE_PROTOCOL.md` describes the protocol as it now is, so §29.14's area-C and ABI-4
  description is superseded.
- **Recipe flags:** `SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE`, `_RAM_MISS_PIECE_STREAM` and `_EXPERT_PREFETCH`
  (with `_NATIVE_PREFETCH`) are gone.
  - The chain is always two-phase with piece streaming, and native prefetch is deleted.
  - A set value warns and is ignored (`environ.py`, `_LEASE_CHAIN_MINIMAL_NOTE`), so archived arms still launch.
  - `arm_env.py` no longer sets them.
- **CPU experts:** `SGLANG_DSV41_CPU_EXPERTS` is not in the recipe.

### 30.1 What CPU experts do

A captured decode step's pinned-tier hits (RAM hits) are split: `split[n]` of a request's n copy-engine lanes are
computed on the CPU, and the rest are copied over the link as before. The CPU lanes run inside the lease chain, so
CopyDone, the leases, the gate and the fail-stop keep one publisher. The plan's "Step B design" section has the device,
host and Python sides.
- **Kernel:** exllamav3's `cpu/moe_mul1.cpp` at `02aef45`, vendored verbatim (`e371887e1e`).
  - Built with residual and block-128 int8 activations (`SGLANG_EXL3_CPU_ACT_RESIDUAL`, `_BLOCK`; `d2b154cb13`).
  - Rows are read in the checkpoint's native layout. The loader copies the EXL3 safetensors byte for byte, and
    nothing swizzles them (§30.4).
- **Which lanes go to the CPU:** the lowest-scored ones (§30.2).
- **Launch requirements**, refused by the gate (`expert_stream_requirements_exl3.py`) and again at startup:
  - a breakable BS1 decode graph without speculative decoding, refused first and on its own;
  - graph gather and the fused plan (`SGLANG_MOE_EXPERT_FUSED_PLAN=1`);
  - DIRECT residency (`SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`, `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`);
  - the copy engine, layer fusion, and `SGLANG_DSV41_CPU_EXPERTS_CORES`;
  - `SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE` off.
  - The env prerequisites are listed at once, and no refusal suggests a disabled or full decode graph.
  - `EXL3_MOE_CPU_PIN=0` is checked at startup only (`srt/layers/moe/cpu_experts/exl3.py`), not by the gate.
- **A failed CPU forward** calls fail_stop in the CPU expert thread (`a799c9bb5a`).
- **Knobs:**
  - `SGLANG_DSV41_CPU_EXPERTS_SPLIT`, an explicit 9-entry table. Otherwise `k*(n)` comes from `_CPU_MS` (0.52),
    `_LINK_MS` (1.0) and `_HANDOFF_MS` (0.02).
  - `_THREADS`.
  - `_RETUNE_BATCHES`, where 0 keeps the startup table.
  - `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION` (default on) and `_CALIBRATION_REPS` (default 10): the startup
    calibration below. `SGLANG_DSV41_CPU_EXPERTS_MISSES` is §31.1.

**Startup split calibration (2026-10-02; `cpu_experts/service.py`, `cpu_experts/policy.py`,
`expert_stream/host/split_calibration.h`).** The configured costs are only the starting table. When the copy engine
arms (after 16 captured decode forwards, §25.3), the service measures the split on the loaded model:
- **When:** once, after `torch.cuda.synchronize()`, with the RAM thread paused and the copy engine not yet armed, so no
  CPU or copy lane is typed and nothing competes. The drain matters under the overlap scheduler: a replaying decode
  would otherwise be left with an unserved miss lane while the thread is paused.
- **What:** on one registered row with at least 8 RAM slots, with one discarded warm-up and `_CALIBRATION_REPS` timed
  runs per cell, it times k CPU lanes alone (`cpu`), m DMA'd experts alone (`link`) and k CPU lanes with n - k DMA'd
  experts started together (`both[n][k]`, for n and k up to 8). The combined cell decides the split, since it includes
  the CPU kernel and the DMA contending for host memory bandwidth.
- **Choice:** `split[n]` is the `k` with the lowest `both[n][k]`; a larger `k` within 2% of the best wins, because the
  layer time is equal and the link stays free for the GPU's own misses (`split_from_grid`).
- **Output:** a `CPU experts calibration:` report on stdout and in the log: the `cpu` and `link` times, the layer time
  at the chosen `k` and the split.
- **Afterwards:** the calibrated split is final, so `retune` returns without changing it, and calibration's own CPU jobs
  are left out of `log_stats` and out of `retune`'s baseline.
- **Skipped, keeping the previous split:** `SGLANG_DSV41_CPU_EXPERTS_SPLIT` is set, calibration is off, no row has 8
  slots, or a measurement fails or times out (a warning says which). After a failed run the scratch block stays
  allocated, because a timed-out DMA may still write into it.

**P2 results, partial** (2026-09-29, `d87a3ff24f`, before the lane order of §30.2; the plan's "P2 results").
- **Arms:** production recipe plus `SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_DSV41_CPU_EXPERTS_CORES=18-29`, split
  `[0, 1, 1, 2, 3, 3, 4, 5, 5]`.
- **Throughput:** median decode 9.51 → 13.48 tok/s, a **median paired ratio of 1.40x** (per session 1.34-1.47x).
  The on arm won 8 of 8 paired sessions (p = 0.0039), and TTFT was unchanged.
- **Hot hit rate:** 0.644 → 0.593 over one traced off/on pair (~820 steps each). CPU lanes were never inserted into
  VRAM.
- **CPU cost:** 0.61-0.66 ms per lane under load, against the configured 0.52.
- **Quality:**
  - Greedy match 11/16 against a noise floor of 13/16.
  - KL mean 0.0018 against 0.0012.
  - **E31 fails on one flip** (prompt 3, position 31). The verdict is open.
- **Raw data:** `divix01:cc-expert-prediction/dsv41-cpu-experts/kstar/` and `/mnt/nvme1/cpu-b/p2/`.

### 30.2 CPU lanes take the lowest-scored RAM hits (`c3cd6f1b19`, `35298c6a4d`)

**The loss.** Under DIRECT insert-on-miss, lane j of the plan pairs with victim `usable[j]`, and the commit skips CPU
lanes, so a CPU-computed expert never enters VRAM. That cost P2 about 5 points of hot hit (§30.1).

**The fix.** The fused route plan sorts each layer's residual miss lanes by DIRECT's victim-ranking key, highest
first, and the host gives the CPU the last `split[n]` job lanes. So the CPU computes the RAM hits the ranking values
least, and the high-scored ones are still copied and inserted.
- **The key** is `(routed in the closed window, insert score, -expert)`. `GpuResidencyUpdater.enable_miss_order()`
  keeps it as `miss_keys` (int64 `[layers, E]`), rewritten by every `_rank_victims`.
- **The sort** is a warp-uniform rank in `plan_unique_routes_kernel`, with ties broken by lane. Non-residual
  positions and every remap consumer keep indexing by plan row.
- **Host:** `choose_cpu_lanes_locked` takes the last `split[n]` pinned-tier hits, excluding LOADING lanes. The earlier
  NUMA local-first pass and its slot-node plumbing are deleted.
- **CPU experts off:** the keys pointer is null and the plan is bit-identical. This is gated because sorting changes
  the fused MoE's fp32 summation order.
- **Wiring:** `Exl3RamMissService.attach` enables the miss order and points each pinned-tier streamer at its row of
  `miss_keys`. A capture with CPU experts but no miss order raises.

**Chosen by replay, not measured in a server.** No decode A/B or quality run exists for the sorted order.

### 30.3 Insertion policies replayed (`scripts/dsv41/cpu_expert_sim.py`, `e225a7a94b`..`1bcbde08e2`)

The simulator replays a router-capture trace through `tier_sim.py`'s VRAM residency with the CPU lanes chosen per
policy. It uses flat `c_cpu` 0.63 ms and NVMe 1.5 ms. The off baseline is 109.76 ms/token.

| Policy | Hot hit | ms/token |
|---|---:|---:|
| Insert every lane, CPU lanes included (upper bound) | 0.672 | 73.03 |
| Lane order (P2's behaviour) | 0.606 | 79.48 |
| NUMA-local first | 0.601 | 80.05 |
| By score | 0.643 | 76.00 |
| By score, device ascending, CPU takes the head | 0.641 | 76.13 |
| **By score, device descending, CPU takes the tail (merged)** | **0.643** | **75.94** |

**Deferred inserts** push the CPU lanes' rows over the idle link afterwards. None beats the merged policy once the
link cost is counted:

| Link cost of the deferred inserts | ms/token |
|---|---:|
| Not costed | 75.02 |
| Worst case | 110.08 |
| Amortised | 107.97 |
| Per-token, idle-optimistic | 81.64 |

- **Periodic promotion** every N forwards (N in {4, 8, 16, 32}, P in {1, 2, 4} per layer) gains at most 0.2%
  amortised and loses in every worst case. It is not built.
- **Calibration:** the sim drops 0.066 of hit rate for lane order, against the 0.051 P2 served (§30.1), so it
  overstates the loss by ~30%. Its insert-all hit rate, 0.672, matches the served-hit formula.
- **Outputs:** `divix01:/mnt/nvme1/cpu-p1/insert-policies{,-v2,-promote,-sorted}.{json,txt}`.
- **Tests:** `test/registered/unit/kernels/test_cpu_expert_sim.py`, `test/manual/dsv41/test_tier_sim.py`.

### 30.4 NUMA and layout (microbenchmarks: NUMA 2026-09-30, swizzle P0 2026-09-29)

**NUMA matters little to the kernel.** It ran 12 threads on cores 18-29 (node 1) with the `resid_b128` flavor, in the
native layout.

| Experts per call | 1 | 2 | 6 |
|---|---:|---:|---:|
| Rows on node 1 (local), ms per expert | 0.578 | 0.525 | 0.490 |
| Rows on node 0 (remote), ms per expert | 0.625 | 0.538 | 0.500 |

- Remote costs 2-2.5% at 2-6 experts per call, and 8% at 1 (p90 0.88 against 0.55 ms). The kernel reads an effective
  21-27 GB/s, well under node 0's 62 GB/s (§28.1).
- **Under node-0 memory load** (a numpy `copyto` hog on node 0, not PCIe or NVMe DMA; 2 experts per call), remote rows
  slow down against remote unloaded (0.538 ms):
  - +0.009 ms at ~11 GB/s of load, one run;
  - +0.031 ms at ~21 GB/s, one run;
  - +0.116 ms at ~38 GB/s, two runs.
  - The 11 and 21 GB/s loads are per-hog rates (~10.6 GB/s) times the number of hogs.
- The recipe's pinned tier is `PINNED_HOST_NUMA_MB="0:61440,1:40960"`, so most RAM hits are remote to the CPU-expert
  cores. Placement was left as it is. Measuring node-0 traffic under serving comes first, and needs root.
- **Logs:** `divix01:/data/models/slang/nvfp4-work/cpu-numa/`: `matrix.log`, `gated*.log` and per-run
  `numa-g-*.log`. They were gated on quiet windows, because another session's CPU work shared cores 18-29.

**Swizzling does not pay with the chosen flavor.** In ms per expert at 1 / 2 / 6 per call (P0 bench, `p0_real.py`):

| Flavor | Native | Swizzled |
|---|---|---|
| Upstream | 0.558 / 0.511 / 0.475 | 0.459 / 0.416 / 0.395 |
| Residual + block 128 (merged) | 0.575 / 0.518 / 0.475 | 0.735 / 0.670 / 0.631 |

Block-128 already gives the locality a swizzle would. Upstream exllamav3 swizzles its own CPU copy and unswizzles on
the GPU (`moe_unswizzle_trellis`). The only swizzle here is the P0 bench's.

### 30.5 The minimal lease protocol (`dsv41-lease-minimal`, merged into the branch at `4fe14b9f2b`)

**What changed** (`1060a85239`). There is one fail-stop chain per layer: post → W1 → C1 → S → CW → stream wait on
the gate → CC.
- **Removed:**
  - The error, diagnostic and self-description words: the fatal word, status, Terminal, LaneAck, SmAck, StreamProbe,
    CopyArm (folded into the gate), the header, the row table and SlotGen.
  - The advisory ring, native prefetch, the batched and non-streaming chains, and their env vars.
- **CW** writes one Done word per request.
- **Failures:** host failures `std::abort()` and device failures `__trap()`; S keeps its deadline as a trap.
- **Shutdown** synchronises the device before it closes admission.
- **Protocol reference:** `analysis/dsv41-drive/LEASE_PROTOCOL.md`.

**Served smoke** (`analysis/dsv41-drive/lease-minimal/smoke.sh`; `divix01:/mnt/nvme1/lease-minimal/smoke1/`).
- Setup: `644c3bedf4` in `wt-lease-gpu`, before the merge into this branch, so CPU experts and the lane order were not
  exercised. The production recipe with no overrides, and six requests (three prompts, twice).
- Repeats identical; copy engine armed.
- Server exit 0 with an orderly service stop; 0 FATAL, quarantine or traceback lines.
- **No decode A/B** against the previous protocol was run.

**Test surface.** The lease rewrite deleted these test files, among others:
- `test_exl3_ram_miss_two_phase*.py`, `test_exl3_ram_miss_advisory.py`, `test_exl3_native_prefetch*.py`;
- the manual `test_exl3_two_phase_*_cuda.py`, `test_exl3_piece_stream*_cuda.py` and `test_exl3_ram_miss_cuda.py`.

That is why the branch's diff against `567337b395` removes more lines (19,885) than it adds (12,925).

### 30.6 Host test and tool exports in their own header (`1fb0743d30`..`1ff55fb37b`)

`host/ffi_test_exports.h` holds `HostTestExports<HostExports<...>>`, a partial specialization that derives from
`HostExports`. It carries 22 exports plus 2 helpers moved out of `ffi_exports.h`, byte-identical.
- **What moved:** the exports no server path calls, which tests and analysis tools use.
  - Only 9 of them are refused on the production build (`TEST_ONLY_EXPORTS`, `expert_stream_transport.py`).
  - Tools call others against a live module: `copy_engine_idle` (`copy-engine/module_load_probe.py`,
    `ce-soak/hazard_probe.py`) and `read_rows_traced` (`bench_pack_workers.py`).
- **Every build must still expand `EXPERT_STREAM_HOST_TEST_EXPORTS`, production included.** Dropping it from prod
  would break those tools with a missing symbol.
- **Symbol check:** `nm -D --defined-only` on prod, instr and two_name modules built through `load_jit` gives the same
  61 `expert_stream_*` exports before and after. instr_tsan was checked on its object with `readelf`, because divix01
  has no gcc TSan runtime to link.
- **Move proof:** `.omc/artifacts/ffi-test-split/verify.py` in the author's laptop tree, not in the repo.

### 30.7 Verification and gaps

**Suites** (divix01; `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-17,30-63`;
EXIT read from pytest):

| Run | Commit | Result |
|---|---|---|
| `test/registered/unit/kernels test/registered/unit/scripts/test_cpu_experts_quality.py` | `1ff55fb37b` | 775 passed / 456 skipped |
| `test/registered/unit/test_expert_stream_requirements_exl3.py` | `02d343733c` | 39 passed |
| `test_exl3_ram_miss_attach_lanes.py` | `02d343733c` | 12 passed |
| GPU: `test_expert_route_plan_fused.py`, `test_expert_residency_gpu.py`, `test_exl3_cpu_lane_order_cuda.py` | `fad844b678` | 382 passed / 1 failed |
| GPU manual: split parity, lease kernels, copy engine, lease ordering (`SGLANG_EXL3_SRC` set) | `fad844b678` | 37 passed |

- **The GPU failure** is `test_exl3_direct_ack_violation_cannot_publish_a_resident`. It also fails at `1bcbde08e2`,
  before the lane-order work. Its SimpleNamespace backend lacks `cpu_experts`.
- **Mutants caught:**
  - The plan kernel ignoring the keys: 97 failed.
  - The host taking the head instead of the tail: 2 failed, plus the end-to-end test.
  - Attach skipping `enable_miss_order`, or not assigning a streamer's row: the attach test.
  - Attach passing `cpu_experts=False`: the attach test.
- **Commit `3937a8833a` alone is red** on two gate cases, fixed in `02d343733c`. Do not quote gate counts at it.

**Not done:**
1. **No GPU run** of `3937a8833a` (the capture-time streamer check) or `1ff55fb37b`. GPU runs were held for the merge.
2. **No server run** of the lane order (§30.2): neither decode A/B nor quality gate. P2's figures predate it.
3. **The E31 verdict** is open (§30.1).
4. **The split retune** from the measured 0.61-0.66 ms per lane. (2026-10-02: the startup calibration, §30.1, now
   measures the split by default; no served run of it is recorded here.)
5. **NUMA placement** of the pinned tier against the CPU-expert cores (§30.4).
6. **Silent no-op.** With `SGLANG_MOE_HOT_GPU_MB=0`, `SGLANG_MOE_PINNED_HOST_MB=0`, prefetch off and graph gather
   off, `memory_hook.py` returns before the launch gate, so `SGLANG_DSV41_CPU_EXPERTS=1` silently does nothing. With
   graph gather on, the launch is refused, but the message doesn't name CPU experts.
7. **Test gaps:**
   - No CPU test covers a captured post with CPU experts off (`cpu_input` must stay None).
   - The attach test uses a stand-in residency updater.
8. **No decode A/B** of the minimal lease protocol against its predecessor (§30.5). §29.13 items 1 and 10 still stand.

## 31. The slot-map protocol: device-held expert map, staging slots, CPU-computed misses (`dsv41-device-slot-map`, 2026-09-30)

Branch `dsv41-device-slot-map` from `59cfb07c99`. Plan: `docs/superpowers/plans/2026-09-30-dsv41-device-slot-map.md`
(Revision 1 is binding); ledger of rulings: `.superpowers/sdd/2026-09-30-dsv41-device-slot-map/progress.md`
(untracked; the rulings exist only there and in the branch's final summary). The protocol as built: `analysis/dsv41-drive/LEASE_PROTOCOL.md`,
which supersedes §30.5's lease description. Merged to master as `27a6195235` (2026-09-30); §31.5 covers the
host simplifications after the merge. Nothing in §31 has run in a server.

### 31.1 What changed

- **The device holds the map.** Each layer's expert -> pinned-slot map lives on the GPU (`map_bank`) and is updated
  only by deltas: one per decode record with a miss, numbered by a per-row map chain, and one bulk delta after each
  eager host use. The post types every lane from it: `HIT_COPY`, `HIT_SM`, `HIT_CPU`, `MISS_GPU`, `MISS_CPU`.
- **No leases.** RowResult, W1 (the hit wait), Done, the lease counters, deferral and acknowledgements are gone. A RAM
  hit is copied the moment the post types it; the host only checks the lane. Safety comes from victims being chosen at
  record time, never among the record's routes, and from one record in flight (LEASE_PROTOCOL.md, "Why it is safe").
- **Staging slots.** Each row reserves K = min(gather width, capacity - 1) slots (at service start since §31.5;
  the branch reserved them at attach). A miss reads into a
  staging slot, which becomes the expert's RAM slot; its victim becomes staging (a swap, not a copy). With no victim the
  RAM insert is skipped and the miss is still served.
- **CPU experts take misses too** with `SGLANG_DSV41_CPU_EXPERTS_MISSES=1` (default off): the miss is read into
  staging, then computed by the CPU. The CPU output is two parts per row (hits' and misses'), and CC flags which parts
  hold this record's partial; the route tables sum the flagged parts.
- **The hit copy is switchable:** `SGLANG_DSV41_RAM_HIT_COPY=ce` (default, the copy engine) or `sm` (C1).
- **Env:** `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US` is deprecated and ignored. The gate refuses `CPU_EXPERTS_MISSES`
  without `SGLANG_DSV41_CPU_EXPERTS=1` and an unknown `RAM_HIT_COPY`.
- **Wire v2:** request page 4160 B (128-byte records, `kRecordBytes` in `lease_layout.h`, with chain, epoch, per-lane slot, dst, weight and kind), completion
  block 20480 B plus a 256-byte delta per row.

Found on the way, fixed on the branch: the post stored every lane-kind byte as 0 (an `st_relaxed_sys<uint8_t>` call
resolved to a self-recursive overload), and the host counted that malformed record as an overrun instead of failing;
it now fails stop. Attach could find no slot to stage when every slot held a VRAM-hot expert; it now evicts those last.
Attach also ran unpaused, so on a full tier its evictions reached the device only at the next eager host use; it now
runs inside the pause. Two failures predate the branch: `test_exl3_direct_ack_violation_cannot_publish_a_resident`
(its stub backend lacked `cpu_experts` since the CPU-experts merge; stub fixed) and the shutdown test in item 5.

### 31.2 The replay (Task 0)

`scripts/dsv41/cpu_expert_sim.py --slot-map` on the router capture
`direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl` (6153 decode tokens; 11.32 NVMe misses/token
at K = 0), deferred RAM inserts, flat CPU cost 0.63 ms/lane, NVMe 1.5 ms/miss, split [0, 1, 1, 2, 3, 3, 4]. Output:
`divix01:/mnt/nvme1/cpu-p1/slot-map-policies.{json,txt}`. ms/token; "pess" runs a CPU miss after the layer's link and
hit work, "opt" overlaps them:

| K (staging) | CPU off | CPU hits | CPU hits + misses (pess / opt) |
|---|---|---|---|
| 0 | 109.76 | 75.94 | 76.29 / 70.15 |
| 6 | 110.73 | 77.18 | 77.52 / 71.05 |
| 8 | 111.05 | 77.58 | 77.92 / 71.34 |

- Staging costs +1.24 ms/token at K = 6 and +1.64 at K = 8 (CPU hits), from the slots it takes out of the tier
  (NVMe misses/token 11.32 -> 11.97 -> 12.18).
- Protect reads off against on: at most 0.01 ms/token at every K and CPU setting.
- **Decision:** `CPU_EXPERTS_MISSES` defaults off. On the pessimistic bound it costs 0.35 ms/token at K = 0 (75.94 ->
  76.29); only the optimistic bound gains (5.8 ms/token). A served A/B decides.

### 31.3 Suites

All on divix01 in private worktrees at the pushed branch head, `PYTHONPATH` at the tree under test, pytest's own
status read.

- **CPU:** `test/registered/unit/kernels`, `test/registered/unit/scripts/test_cpu_experts_quality.py` and
  `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`, with `-q -p no:randomly`, `-n 2`,
  `taskset -c 0-17,30-63`, `OMP_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=`:
  - merge base `59cfb07c99`: 826 passed, 458 skipped (1284 collected);
  - branch, at `1db383e34b` (the run's `HEAD=`): 830 passed, 458 skipped (1288 collected, before the attach test
    below).
  - The +4: the seven deleted lease test files held 37 tests (`lease_defer` 7, `lease_publication` 5,
    `lease_service` 9, `leases` 8, `lease_thread` 4, `lease_wrap` 2, `task5_item5_ack_independence` 2); the two new
    ones hold 27 (`test_exl3_ram_miss_slot_map.py` 19, `test_ram_slot_map.py` 8); modified files net +14
    (`cpu_expert_sim` +11, `cpu_experts` +3, `lease_block` +1, `copy_engine` +1, `ram_miss_service` +1,
    `prefill_share` -1, `build_variants` -2).
  - Re-run at `0dc3ecd672`: 830 passed, 458 skipped. One later commit adds a test
    (`test_attach_unmaps_on_a_full_tier_reach_the_device_before_any_post`).
- **GPU** (RTX 5090, `flock cc-gpu.lock taskset -c 32-63`, `CUDA_MODULE_LOADING=EAGER`, `SGLANG_EXL3_SRC` set, from
  `test/manual/dsv41`, `-k 'not ends_a_gpu_reader'`): every `*_cuda.py` and `*_gpu.py` the branch touched
  (`copy_engine`, `cpu_lane_order`, `lease_kernels`, `lease_ordering`, `moe_split_parity`, `ram_miss_graph`,
  `slot_map_kernels`, `task5_item6_shutdown`, `layer_fusion_launcher_checks`), plus `test_dsv41_layer_fusion_gpu.py`,
  `test/registered/unit/kernels/test_expert_route_plan_fused.py` and
  `test/registered/unit/layers/moe/test_expert_residency_gpu.py`, at `cc19d4a8d2`: **585 passed, 1 deselected**.
  The deselected test is item 5 below.

### 31.4 Not done

1. **No server run** of any kind: no smoke, no decode A/B against master, no quality gate (E31) for CPU-computed
   misses.
2. **`SGLANG_DSV41_RAM_HIT_COPY=sm` has never been measured**, nor `CPU_EXPERTS_MISSES=1` beyond the replay.
3. **No mutation runs.** The plan listed five mutants in Task 7, plus a delta-ordering one; they were not run, at the
   user's direction ("no mutants needed for tests"). The tests' sensitivity is shown only by the self-recursive store
   the GPU rig caught, and by the red tests written for the attach fixes.
4. **Duplicate experts in a plan trap.** The fused planner emits unique experts; the non-fused `route_plan` path is
   not exercised by graph gather, and was not checked to be duplicate-free.
5. **`test_shutdown_ends_a_gpu_reader_waiting_on_the_service_without_waiting_out_its_timeout` fails**, identically at
   the merge base: shutdown runs its barrier before admission closes, by design, so a chain waiting on a paused
   service waits for its deadline. The test predates that design.
6. **The "Lease Chain Map" artifact** still shows the lease protocol.
7. **The served runs are deferred on purpose** (2026-10-01): an optimized exllamav3 CPU-expert function is in progress,
   and the A/B (master vs before, `RAM_HIT_COPY` ce/sm, `CPU_EXPERTS_MISSES` on/off, CPU experts on cores 18-29 as in
   P2) and the E31 gate wait for it.
8. **No ThreadSanitizer run** after §31.5's queue removal (`test_expert_stream_hotpath_tsan.py` is manual).

### 31.5 After the merge: the host tier simplified (2026-10-01, `6cf91a8e17`..`d3228133a0`)

Each step on master, test-first, CPU and GPU suites on divix01 before the push.

- **`kLoading` is gone** (`6cf91a8e17`). A miss's slot stays `kStaging` during its read and is mapped and made `kReady`
  when the bytes land; the service owns the tier for the whole read, so nothing else could observe the state.
  `release()` loses its "while it is loading" refusal; the other states keep their numbers (0 FREE, 2 READY, 3 STAGING).
  `serve_record` walks the routed experts once, and `touch_request` shares its stamping.
- **Comments condensed** in `ram_tier.h` and `ram_thread.h` (`a53a3534fa`): plan and task citations dropped, the
  `RamThread` preamble split so each fact sits at the function it constrains. No code change.
- **Staging reserved by the host at start** (`70fd826fe6`, `7a77475a20`). `RamTier::reserve_staging(k)` takes the first
  K FREE slots of every row and publishes the tag-1 delta before any slot is filled, so nothing is evicted. `from_model`
  hands the graph-gather width to the service (`plan_gather_width`, widest layer, at most `MAX_IDS`) before the startup
  `reassign` starts it; 8 when graph gather is off. Gone: `attach_row`, its pause, eviction at attach and the hot-slot
  fallback (`take_hot`).
- **DIRECT residency required** (`d833748d7c`). The RAM-miss service refuses at attach, and the launch gate refuses,
  unless `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` and `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2` (the arm recipe sets both).
  Gone: the `on_residency` listener, `enable_gpu_hot`/`set_gpu_hot` and the C++ hot-mode switch; a record's hot bitmap
  is applied whenever a hot page is given. The residency-listener machinery in `expert_hot_cache.py`
  (`add_residency_listener`, `_notify_residency_listeners`) had no other caller and is gone too (`7b444830c8`).
- **The hot set is still pushed from Python at each pause.** Moving it into C++ was dropped: the post writes the
  record's hot bitmap before the same gather's `commit_gather` rewrites `slot_to_expert` (`lease_kernels.cuh:144-156`,
  `expert_residency_gpu.py:1001`), so the last record's bitmap can miss that gather's inserts, and a fresh copy needs
  either a device read at pause or a per-step cost.
- **`expert_to_slot` from the published map** (`2b45888b89`). `NativePinnedSlotTable.expert_to_slot` is built from the
  lock-free `mapping()` (version-cached); no reader used `lru_order`'s order. `slot_to_expert` is its inverse.
- **No command queue** (`d1b1ef8670`). Every Python entry point that touches the tier now needs the service paused
  (`require_owner`): `assign`, `release`, `contains`, `touch`, `fill_begin`, `reserve_staging`, `take_bulk_delta`, `set_hot` and the snapshots
  (`slot_info`, `victim_census`, `lru_order`, `slot_to_expert`); everything else reads lock-free (`mapping`, `version`,
  `handled_through`, counters). Gone: the command ring, `run_as_owner`, the drains, mid-read snapshot answering and the
  stop path's drain. The service loop serves records, parks on pause (after serving every posted record) and sleeps
  when idle. The one-time hot seed at attach runs paused. The watchdog stays: it is the only bound on the copy wait
  (`cuStreamWaitValue32` has no timeout) and on a stop that joins a service hung in a read.
- **Size:** `ram_tier.h` 1738 -> 1525 lines, `ram_thread.h` 250 -> 232.
- **Suites at the end** (`d3228133a0` and its parent; commands as in §31.3 with `-k "ram_miss or ram_slot or tier or
  lease or copy_engine or cpu_experts or expert_stream"` for the kernels): CPU kernels 674 -> 673 passed (the queued
  `set_hot` burst test deleted), the layers/moe RAM-miss, host-tier, format and gate files 195 -> 201, the hot-cache
  and residency files 28 -> 25 (three listener tests deleted); GPU (the §31.3 manual files) 56 passed before and after.


## 32. Environment variables (2026-10-02)

The variables a DeepSeek V4.1 EXL3 deployment reads, grouped by what they control. Defaults are the code's;
"Production" is the value `benchmarks/dsv41_baseline/arm_env.py` (`base_env()`) sets when it differs from the default, and `-` means
the recipe leaves the variable alone. Most `SGLANG_DSV41_*` flags are parsed once by `Dsv41Config.from_envs`
(`python/sglang/srt/dsv41_config.py`). The full descriptions are in
[`docs/docs/references/environment_variables.mdx`](docs/docs/references/environment_variables.mdx); declarations are in
`python/sglang/srt/environ.py`. An explicitly set variable always wins over a `--moe-offload-preset` fill.

### 32.1 Streaming and tiers

| Variable | Default | Production | What it does |
|---|---|---|---|
| `SGLANG_DSV41_EXPERT_STREAM` | `False` | `1` | Streams the EXL3 routed experts from disk; the loader skips them. Eager or breakable graph only. |
| `SGLANG_DSV41_EXPERT_DIR` | `""` | the EXL3 shard dir | Shards the experts are read from; required when streaming. |
| `SGLANG_MOE_EXPERT_ROW_SOURCE` | `auto` | `shards` | Where host rows are read from (`auto`, `files`, `tensor`, or a format kind). |
| `SGLANG_MOE_EXPERT_FILE_READER` | `mmap` | `uring_direct` | `mmap`, `uring` or `uring_direct`. |
| `SGLANG_MOE_EXPERT_MIRROR_DIRS` | `""` | three roots (nvme0, nvme4, nvme2) | `os.pathsep`-separated byte-identical checkpoint copies; non-empty reads every row from all roots at once. |
| `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS` | `""` | - | Colon-separated read shares per root (`0` drops one); empty is equal. |
| `SGLANG_MOE_PINNED_HOST_MB` | `0` | `81920` | Pinned host tier of expert rows, in MiB. Mutually exclusive with `SGLANG_MOE_EXPERT_HOST_ARENA`. |
| `SGLANG_MOE_PINNED_HOST_NUMA_MB` | `""` | `0:40960,1:40960` | Per-NUMA-node split of the pinned tier; must sum to `SGLANG_MOE_PINNED_HOST_MB`. |
| `SGLANG_MOE_HOT_GPU_MB` | `0` | `16080` | GPU hot expert cache in MiB; counts against `--mem-fraction-static`. |
| `SGLANG_MOE_EXPERT_GRAPH_GATHER` | `False` | `1` | Decode-sized gathers without host syncs, so CUDA graphs capture them. |
| `SGLANG_MOE_EXPERT_FUSED_PLAN` | `False` | `1` | One fused kernel plans BS1 graph-gather routes. |
| `SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS` | `0` | - | Speculative decoding only: caps graph-gather scratch rows. |
| `SGLANG_MOE_PREFETCH_MAX_CANDIDATES` | `0` | `0` | NVFP4 prefetch candidates; nonzero is refused with graph gather and the EXL3 gate. |
| `SGLANG_URING_FILE_READER_QUEUE_DEPTH` | `128` | - | Queue depth of the shared io_uring file reader. |
| `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS` | `0` | `8192` | Uncached suffixes this long prefill layer-major; `0` disables. |
| `SGLANG_LAYER_MAJOR_STATE_NUMA_NODE` | `1` | - | NUMA node of the layer-major host state store. |
| `SGLANG_FILE_CACHE_MODEL_PATH` | `""` | - | Model path recorded in file-cache identities after a checkpoint move. |

### 32.2 GPU hot cache and residency

| Variable | Default | Production | What it does |
|---|---|---|---|
| `SGLANG_MOE_GPU_RESIDENCY_UPDATE` | `False` | `1` | Decode residency update as device ops inside the captured graph. Required by the RAM-miss service. |
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` | `0` | `2` | `1` SCRATCH, `2` DIRECT (copies misses into victim slots). The RAM-miss service requires `2`. Deprecated alias: `SGLANG_MOE_HOT_INSERT_ON_MISS`. |
| `SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY` | `0.98` | - | Per-token decay of the victim scores. |
| `SGLANG_MOE_HOT_FUSED_INSERT` | `False` | - | Stage 1 boundary inserts through a fused Triton kernel. |
| `SGLANG_MOE_HOT_DYNAMIC` | `False` | `1` | Updates residency while serving. |
| `SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS` | `1024` | `256` | Minimum prefill tokens that trigger an update. |
| `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` | `0` | `1` | Update every N decode forwards; insert-on-miss needs `1`. |
| `SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS` | `8` | `8` | Forwards a promoted expert stays resident. |
| `SGLANG_MOE_HOT_BENEFIT_RATIO` | `1.0` | - | Score ratio a candidate must exceed to replace an expert. |
| `SGLANG_MOE_HOT_DECAY_TOKENS` | `0` | - | Decay scores per N routed tokens; `0` is per boundary. |
| `SGLANG_MOE_HOT_PROMOTION_SIGMAS` | `0.0` | - | Extra noise margin a candidate must lead by. |
| `SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS` | `64` | - | Promotions per layer per decode boundary. |
| `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` | `False` | `0` | Refused by the EXL3 gate. |
| `SGLANG_MOE_ASYNC_RESIDENCY_SCORES` | `False` | `0` | Asynchronous score copies at boundaries. |
| `SGLANG_MOE_HOT_SEED` | `""` | - | Startup residency seed JSON. |
| `SGLANG_MOE_HOT_LOG_INTERVAL` | `100` | `64` | Residency log interval, in boundaries. |
| `SGLANG_MOE_HOT_METRICS_FILE` | `""` | - | Residency metrics file; empty disables. |

### 32.3 RAM-miss service, copy engine and fusion

| Variable | Default | Production | What it does |
|---|---|---|---|
| `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS` | `2000` | `2000` | Per-layer in-graph wait bound before fail-stop; the thread watchdog aborts after `max(30 s, 3x)`. |
| `SGLANG_DSV41_RAM_MISS_SPIN_CORE` | unset | `17` | Core the service thread busy-polls on; its physical core must be its own. |
| `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE` | `False` | `1` | DMA copies of RAM hits on their own thread (§25.3). Requires `CUDA_MODULE_LOADING=EAGER`. |
| `CUDA_MODULE_LOADING` | torch's `LAZY` | `EAGER` | Startup raises unless `EAGER` when the copy engine is on (~1 GiB of device memory). |
| `SGLANG_DSV41_RAM_HIT_COPY` | `ce` | - | `ce` (DMA) or `sm` (in-graph SM copy). |
| `SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES` | `False` | `1` | Copy engine moves only the two trellis tensors; the wait reads the four small ones. |
| `SGLANG_DSV41_ENABLE_LAYER_FUSION` | `False` | `1` | JIT kernels replace about 89 small torch kernels per layer. |
| `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION` | `False` | `1` | BS1 decode cast fusion. |
| `SGLANG_DSV41_ENABLE_LEASE_PDL` | `False` | `1` | Programmatic dependent launch for the chain kernels. |
| `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM` | `False` | - | Shared expert and DIRECT commit on a side stream. |
| `SGLANG_DSV41_ENABLE_PREFILL_FILLS` | `False` | `1` | Eager pinned-tier misses read by the native reader into the slabs. |
| `SGLANG_DSV41_ENABLE_PREFILL_SHARE` | `False` | - | Bounds a prefill's pinned-tier admissions per layer. |
| `SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN` | `False` | `1` | Eager MoE plans routes on the host. |
| `SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER` | `False` | `1` | Prefill chunk copies resident rows before the filled ones. |

### 32.4 CPU experts (off in the recipe)

| Variable | Default | What it does |
|---|---|---|
| `SGLANG_DSV41_CPU_EXPERTS` | `False` | Computes a layer's RAM-tier experts on the CPU. Batch-1 decode only. |
| `SGLANG_DSV41_CPU_EXPERTS_CORES` | `""` | Optional override of the pool's cores as a taskset list (at least two); unset, derived from the node's free physical cores. `SGLANG_EXPERT_NUMA_CORES` sets per-node plans. |
| `SGLANG_DSV41_CPU_EXPERTS_THREADS` | `0` | Worker threads, at most one per core; `0` is one per core. |
| `SGLANG_DSV41_CPU_EXPERTS_SPLIT` | `""` | Nine counts, CPU lanes per n lanes; set, it disables calibration. |
| `SGLANG_DSV41_CPU_EXPERTS_CPU_MS` / `_LINK_MS` / `_HANDOFF_MS` | `0.52` / `1.0` / `0.02` | Cost model that builds the starting split. |
| `SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES` | `0` | Batches between re-tunes from measured CPU cost; `0` keeps the table. |
| `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION` | `True` | Measures the split once when the copy engine arms (§30.1). |
| `SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS` | `10` | Timed runs per calibration cell. |
| `SGLANG_DSV41_CPU_EXPERTS_MISSES` | `False` | CPU also computes NVMe misses; needs `SGLANG_DSV41_CPU_EXPERTS`. |
| `EXL3_MOE_CPU_PIN` | unset | Must be `0` or startup raises. |
| `SGLANG_EXL3_CPU_ACT_RESIDUAL` / `_ACT_BLOCK` | `False` / `0` | Build-time options of the CPU kernel's int8 activations. |
| `SGLANG_EXL3_CPU_CXX` | `""` | GCC 15 `g++` for the kernel build (the recipe sets the gcc-toolset-15 one). |

### 32.5 EXL3 build, Engram and DeepSeek V4.1 model options

| Variable | Default | Production | What it does |
|---|---|---|---|
| `SGLANG_EXL3_SRC` | `""` | set | Pinned exllamav3 checkout JIT-built into the quant method; required for EXL3. |
| `SGLANG_EXL3_BUILD_DIR` | `~/.cache/sglang/exl3_ext` | set | Build directory of that extension. |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | `False` | - | Engram tables in host memory. |
| `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | `shared` | - | `shared` or `per_rank`. |
| `SGLANG_DSV41_ENGRAM_TABLE_DIR` | `""` | Flash checkpoint dir | Serve Engram rows from safetensors shards via `np.memmap`. |
| `SGLANG_DSV41_ENGRAM_RAM_GIB` | `0.0` | `5` | Engram RAM row cache in GiB; misses use O_DIRECT. |
| `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING` | `False` | `1` | Layer-1 lookup as a CUDA host node with an io_uring worker. |
| `SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT` | `False` | `1` | Device post and wait kernels, no host nodes in the graph. |
| `SGLANG_DSV41_TORCH_PREFILL_INDEXER` | `False` | `1` | Torch prefill indexer instead of the DeepGEMM kernel. |
| `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB` | `0` | `128` | Cap on one bf16 score chunk; `0` is 1 GiB. |
| `SGLANG_DSV41_FUSED_WO_A` | `True` | - | SM100/SM103 `wo_a` verify megakernel. |
| `SGLANG_DSV41_REASONING_EFFORT` / `SGLANG_DSV4_REASONING_EFFORT` | unset / `""` | - | Default reasoning effort for requests that carry none. |
| `SGLANG_DSV4_KV_LAYOUT` | `v4` | - | `v4`, `v41` or `auto`. |
| `SGLANG_DSV4_COMPRESSED_KV_LAYOUT` | `auto` | - | Compressed-cache layout under `v41`. |
| `SGLANG_DSV4_FP4_EXPERTS` / `SGLANG_DSV4_FP4_DEQUANT` | `True` / `False` | - | FP4 checkpoint handling. |
| `SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE` | `False` | - | SWA KV quantized from bf16-rounded values. |
| `SGLANG_DSV4_UNIFIED_KV_FP8` | `False` | - | `unified_kv` only: fp8 nope pool plus bf16 rope pool. |
| `SGLANG_DSV4_COMPRESS_STATE_DTYPE` | `float32` | - | Compressor state dtype. |
| `SGLANG_OPT_DSV4_NONPAGED_INDEXER` / `_MIN_QUERY_TOKENS` | `True` / `8192` | - | Non-paged indexer and its minimum per-rank query rows. |

### 32.6 Debug, trace and test

| Variable | What it does |
|---|---|
| `SGLANG_DSV41_EXPERT_TRACE_PATH` | One JSON line per streamed MoE layer call, for `scripts/dsv41/tier_sim.py`. |
| `SGLANG_DSV41_ROUTER_CAPTURE_PATH` | Per-layer router inputs and top-k weights of every graph forward; requires the trace. |
| `SGLANG_DSV41_SYNC_WAIT_NVTX` | NVTX ranges around host synchronous waits (raw `== "1"`). |
| `SGLANG_TEST_DSV41_RAM_MISS_FAULT` | Test only: `<demands>:<seconds>` delay before demand reads. |
| `SGLANG_MOE_ROUTE_TRACE_DIR` / `_MAX_TOKENS` / `_SPECULATIVE` | Debug only: eager decode routing tensors. |
| `SGLANG_MOE_EXPERT_PREFETCH_*`, `SGLANG_MOE_EXPERT_PREDICTOR*` | Shadow prefetch and predictor studies; the EXL3 gate refuses any pull mode. |
| `SGLANG_EXPERT_STREAM_URING_*` | io_uring reader options (`QUEUE_DEPTH`, `MODE`, `FIXED_FILES`, `READ_MODE`, `WAIT_MODE`, `SQ_THREAD_IDLE_MS`, `SQ_THREAD_CPU`, `DIAGNOSTICS`, `READ_CUTS`, `SLAB_ARENA`), read by `UringOptions::from_env`, not declared in `environ.py`. |

### 32.7 Deprecated and removed

Set values warn and are ignored: `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES`, `SGLANG_DSV41_RAM_MISS_PACK_WORKERS`,
`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES`, `_TWO_PHASE`, `_PIECE_STREAM`, `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`,
`SGLANG_DSV41_ENABLE_NATIVE_PREFETCH`, `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US` (`arm_env.py` still sets it to `100`, so
production logs the warning), and the `SGLANG_MOE_EXPERT_DOORBELL*` family. `SGLANG_MOE_EXPERT_PREFETCH_PULL` and
`SGLANG_MOE_HOT_INSERT_ON_MISS` are legacy aliases of their `_MODE` and `_STAGE` replacements. `SGLANG_DSV41_EXPERT_RAM_GIB`
is retired with no deprecation entry and no read site: a launch script that sets it is ignored and gets no RAM tier,
with only the generic one-time warning about a missing `SGLANG_MOE_PINNED_HOST_MB` (which does not name the old
variable); use `SGLANG_MOE_PINNED_HOST_MB`.

## 33. DSpark: the Phase D1 results, and what a graphed verify needs (2026-10-02)

Phase D1 (`docs/superpowers/plans/2026-09-19-dsv41-dspark.md`) was built and run on 2026-09-19..24, after §18.5 was
written; none of it reached this doc until now. DSpark runs eager only. Run files are in
`divix01:cc-expert-prediction/analysis/dsv41-dspark/`; the draft dir is `cc-expert-prediction/dsv41-dspark-draft`.

### 33.1 What was built (all on `master`, 2026-09-19)

§18.5's blockers, each closed by a commit:
- **Draft dir:** `scripts/dsv41/make_dspark_draft_dir.py` (`8f1755193f`). A truncated checkpoint with
  `num_nextn_predict_layers: 0` no longer counts as bundling a draft (`474640ac70`).
- **Streamer:** EXL3 streams only the target's routed experts, not DSpark's stages (`519a903754`).
- **Loader:** the DSpark loader reads EXL3 drafts (`b2b30b592e`, `01db7a2886`). Weightless (EXL3) `wkv` linears and
  `lm_head` fall back correctly, and a quantized target `lm_head` is accepted (`2b35dd4560`, `2c1e3e0442`).
- **Residency clock:** accepted tokens reach `on_speculative_commit` (`c6daeb9eb1`). The draft-block forward is no
  longer counted as a verify (`f6beec8bc7`).
- **Gate:** speculation is refused with any EXL3 decode graph (`57578836f1`), so DSpark is eager-only. CPU experts
  are refused with speculation too (§30.1).
- **Harness:** `trace_corpus.py --dspark` captures `spec_verify_ct` (`2d25b78087`).

### 33.2 Results (eager, not traced)

| Run | Files | Result |
|---|---|---|
| Smoke, 1 session, 32 tokens | `d1-smoke.json`, `run-d1.log` | Runs; 12 verifies for 32 tokens (accept length 2.67); 0.94 tok/s; TTFT 109 s |
| Greedy parity, sessions 0-7, 128 tokens | `parity-{base,dspark}.json`, `parity-run.log` | **7 of 8 differ.** Mean accept length 2.884 (α ≈ 0.69 from 1 + α + ... + α^5) |
| Determinism control | `parity-base2.json`, `determinism-run.log` | Base vs base: 8 of 8 byte-identical |
| Margin probe, sessions 1-2, 32 tokens | `MARGIN_PROBE_FINDINGS.md` | First divergence is an exact bf16 tie (margin 0 in both arms); DSpark's token is in base's own tied-max set |
| Production checkout, 64k context, one long session (2026-09-24) | `current-prod-64k.log`, `launch-current-smoke.sh` | Accept length 2.73-4.22 per logged interval (mean of 11 ≈ 3.4; accept rate 0.34-0.65); **1.4-2.2 tok/s** |

**Parity verdict (margin probe): not an acceptance defect.** Both argmaxes break ties to the lowest index. A tie can
only reject a legitimate draft, so α is not inflated. But the 6-token verify forward and the 1-token decode forward
disagree by up to 1.4 logprob on near-top tokens (top-5 set differs at ~28% of positions). That is what moves ties.
The source is not isolated. Candidates: the sm120 sparse attention at 6 query rows, M-dependent GEMM kernels, or the
Engram verify context. Exact-text parity is the wrong test here. The probe proposes a teacher-forced tolerance test
(eps two bf16 quanta), not yet built.

**Not done:** Task 8's rows per accepted token, the union curve, and the break-even. The 2026-09-24 launch script
sets flags that §30-§32 have since removed (`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES`, `_RAM_MISS_PACK_WORKERS`; §32.7),
so it needs updating before a relaunch.

### 33.3 Running verify in the breakable decode graph: what blocks it

Read-only code audit, 2026-10-02, at `master` `7f22a6bcbb`. Target: DSpark verify (M ≤ 6 tokens, bs 1) under
`--cuda-graph-backend-decode breakable` with option C.

**Already works for M tokens at bs 1:**
- **Capture.** The breakable backend uses the same runner as the full graph. Under speculation it captures
  `TARGET_VERIFY`, sized `max_bs × verify width` (`decode_cuda_graph_runner.py:291-348`). Max bs 1 gives one 6-token
  graph. A shorter verify replays eager unless ragged mode is on (`:660-668`).
- **Scratch and accounting.** Gather scratch is multiplied by the verify width (`model_runner.py:747-752`). The host
  clock counts `bs × draft_token_num` (`expert_residency_clock.py:45-47`), and the device counter counts
  `topk_ids.shape[0]` (`expert_stream.py:1451`). §10's `is_speculative()` guards are gone.
- **Attention.** Verify is `is_extend()`, so attention runs as an eager break (`deepseek_v4.py:2063-2068`).
- **One record per layer for the union of the M tokens' experts, on the GPU side.** Since §31 the pinned RAM map
  changes only at record time. Victims are never among the record's routes, and one record is in flight, so the RAM
  set is fixed for the layer. A lane is one distinct expert (`lease_device.cuh:379-380` traps on a repeat).
  `slot`/`dst`/`kind` are per expert. Per-token weights are applied only in the GPU combine
  (`exl3_route_tables.cuh:108-118`).

**Single-token assumptions, hardest first:**
1. **Width against VRAM capacity (redesign).** M = 6 at top-6 is up to 36 distinct experts per layer.
   - DIRECT insert-on-miss needs `capacity ≥ 2 × width` (`expert_residency_gpu.py:378-383`): 72 slots/layer at full
     width, against ≈30/layer with the draft resident (§4).
   - Staging at width 36 is up to 1,440 rows ≈ 17.9 GiB of pinned RAM (§10, Stage A).
   - So the gather width W must be capped below 36. A device-side overflow check must then send larger unions to an
     eager or second-pass verify; `model_runner.py:759-764` already names the missing "overflow re-verify".
   - W comes from the union curve (§10 gate 1).
2. **The record is 8 lanes** (`kMaxIds = 8`, static-asserted, `lease_layout.h:24,84`; `MAX_IDS` at
   `expert_stream_transport.py:1035`). Attach refuses `graph_gather_rows > MAX_IDS` (`exl3_ram_miss.py:1134-1140`),
   so a lifted gate still fails at startup. Two ways out:
   - a wider wire (W lanes, 64-bit kind/CE/CPU masks, PieceMask and delta blocks resized; `row_copy_kernels.cuh`
     packs 8-bit masks);
   - ⌈U/8⌉ records per layer, which keeps the wire but lengthens each layer's miss wait.
3. **Fused route plan:** one warp, ≤32 routes, unique ids only, and `topk_ids.shape[0] == 1` is required
   (`expert_route_plan.py:126-134`). It needs cross-token dedup; `plan_graph_routes` already has the logic.
4. **In-graph EXL3 MoE:** raises on `x.shape[0] != 1` (`exl3_fused_moe.py:171-174`), with buffers sized for one
   token's top-k. It needs `[M, H]` in and out and a token index per route.
   - Unverified: whether exllamav3's `exl3_moe` takes several rows per slot under deterministic mode. The P2 probe
     (`test/manual/dsv41/test_exl3_moe_probe_gpu.py`) can check.
5. **CPU experts are one token end to end.**
   - The C ABI calls `forward_raw(..., 1, k, ...)` (`moe_mul1.cpp:3250`).
   - The pool needs `x.shape[0] == 1`. Input is one row, output is one `[hidden]` per part.
   - The record's lane `weight` sums the matching routes into one scalar (`lease_kernels.cuh:204-211`).
   - The tuned path turns off for chunks with more than one token (`moe_mul1.cpp:3073`).
   - Cost: link cost is flat per expert, while CPU cost grows with the tokens routed to it (≈ ⌈t/2⌉ weight passes,
     inferred). So verify favours the link anyway.
   - **v1 runs CPU experts off for verify.** Every DSpark target step is a verify, so that gives up §30.1's 1.40×.
6. **Small items:**
   - The copy-engine barrier tests `is_decode()`, which excludes `TARGET_VERIFY` (`exl3_ram_miss.py:1364-1367`).
     So an armed engine forces `torch.cuda.synchronize()` on every captured verify, and verifies never count toward
     arming.
   - The Engram native lookup needs one token in decode mode (`engram.py:130-137`): two eager breaks per verify.
   - The protect list truncates silently at 8 (`lease_kernels.cuh:164`). Only recency stamps are lost.
7. **DSpark side:**
   - The resident draft MoE runs `exl3_moe_loop`, which asserts it is not capturing and reads `bincount().tolist()`
     (`exl3_ops.py:191-211`). So draft capture must be skipped for an EXL3 draft (`dspark_worker_v2.py:531-566`)
     while the target keeps its graph.
   - Use static verify mode: compact mode adds host syncs (`dspark_planner.py:347-352`, `:501`).
   - Folding accept/commit into the verify graph currently also requires a folded draft proposal
     (`dspark_worker_v2.py:886-895`).
   - Graphing the draft needs a capturable multi-token EXL3 MoE for its 128 resident experts. None exists.
8. **The gate** (`expert_stream_requirements_exl3.py:100-109`) is lifted last. Its "scratch sized for one token"
   comment is stale; the record width and the MoE are the real limits.

**Why it may not pay [estimate].**
- The link (~1 ms per RAM-hit row) is the bottleneck. A verify moves the union for up to 6 tokens and yields ~2.9-3.4
  accepted tokens.
- At the Qwen stand-in's reuse (§10), a 6-token union near 4× one token's rows moves ~1.3× more bytes per accepted
  token than plain decode, before draft time.
- The baseline is now ~13.5 tok/s with CPU experts (§30.1), which v1 must give up for verify.
- The resident draft's 6.75 GiB comes out of the hot cache.

**Next step if resumed:** measure the union offline before any D2 code. Replay the router capture
(`direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl`, 6,153 consecutive decode tokens) in windows of
N = 2/4/6. That gives:
- per-layer union size and its p95/p99, which sets W;
- RAM-hit and miss rows per accepted token at accept length ~3, through `cpu_expert_sim.py --slot-map`;
- a projected tok/s against 13.5.

Build items 1-5 only if the projection beats it.

### 33.5 The verify union curve and the graphed-verify projection (2026-10-05)

Plan `docs/superpowers/plans/2026-10-05-dsv41-dspark-graph-verify-gate.md`, branch `dsv41-dspark-graph`, run at
`00cff9c620`. This is §33.3's "next step" and §10's measurement gate 1. The run is offline, on the CPU only.

**Method.** `scripts/dsv41/verify_union.py` replays the DSV4.1 router capture:
`direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl`, with 40 layers, 384 experts, top-6, 6,153
decode tokens and 26 requests.
- **Windows.** Each verify is a window of `width` consecutive decode tokens of one request. The next window starts
  `stride` tokens later, so stride stands for the accept length. A window never crosses a request or a prefill.
- **Draft tokens.** The true next tokens' routes stand in for the draft tokens' (teacher forcing).
- **Replay.** Windows go through `cpu_expert_sim.replay_nm`, which now takes the DIRECT shortlist width (`miss_rows`
  = W). The settings are the slot-map recipe's: deferred RAM inserts, K = 0, and CPU experts off, as §33.3 v1 requires.
- **Cost.** A verify costs `(n + m) × 1.0 ms` of link plus `m × 1.5 ms` of NVMe wait plus 14 ms of GPU.
- **Baseline.** The baseline is the same simulator's plain-decode arm with CPU hits: 75.94 ms/token = 13.17 tok/s,
  exactly §31.2's figure.
- **Command.** `taskset -c 0-63 python scripts/dsv41/verify_union.py <stages.jsonl> --out $G/projection.json --jobs 8`,
  with `$G = cc-expert-prediction/analysis/dsv41-dspark/graph-verify/` (`projection.{md,json}`). It ran in 2 minutes
  with exit 0.

**Union curve** (distinct experts per verify and layer):

| width:stride | mean | p95 | p99 | max |
|---|---|---|---|---|
| 4:2 / 4:3 | 16.0 / 15.9 | 21 | 23 | 24 |
| 6:2 / 6:3 / 6:4 | 21.2 / 21.1 / 21.2 | 29 | 32 | 36 |

- A 6-token verify touches 3.5× one token's 6 experts per layer.
- The p99 of 32 sits exactly at the wire cap, and the maximum of 36 exceeds it.

**Projection** (draft slots = hot slots per layer given to the draft: hybrid ≈ 4, resident ≈ 13; tok/s assumes no
draft time):

| width:stride | W | draft slots | overflow | VRAM ok (cap ≥ 2W) | ms per accepted token | tok/s |
|---|---|---|---|---|---|---|
| 6:3 | 8 | 4 | 0.502 | yes | 149.0 | 6.71 |
| 6:3 | 24 | 4 | 0.002 | no | 115.1 | 8.69 |
| 6:3 | 32 | 0 | 0.000 | no | 101.8 | 9.83 |
| 6:4 | 32 | 0 | 0.000 | no | 99.6 | 10.04 |
| 4:3 | 24 | 0 | 0.000 | no | 99.6 | 10.04 |
| plain decode, CPU experts off (§31.2) | | | | | 109.76 | 9.11 |
| **plain decode, CPU hits (baseline)** | | | | | **75.94** | **13.17** |

- **NVMe floor.** NVMe reads per verify are 11.2-11.3 × stride in every row. That is the same per accepted token as
  plain decode (§31.2's 11.32). A verify cannot avoid reading each accepted token's new experts, so both paths pay
  the same ≈28 ms/token NVMe floor.
- **The union saves no link time.** At 6:3, W = 32, a verify moves 69.2 RAM-hit rows per accepted token against
  plain decode's 67.5. Its whole 9% lead over plain decode without CPU experts (99.6 vs 109.76 ms) is the 14 ms of
  GPU shared by 3 tokens, the one term this model understates.
- **What it gives up.** Plain decode with CPU experts saves 31%.
- **The VRAM base.** The trace ran at `SGLANG_MOE_HOT_GPU_MB=14336` (`router-capture/env.txt`): 28-29 hot slots per
  layer. The current recipe without a draft runs at 16080 MB, about 31.6 per layer. The A/B arms of §33.4 run at
  12040 (hybrid) and 7168 MB (resident), about 23.7 and 14 per layer. That is the trace's capacity minus ≈4.5 and
  minus ≈14, which `draft_slots` 4 and 13 model.
- **The W that keeps overflow low does not fit.** It needs W ≥ 24, and DIRECT's `capacity ≥ 2W` rule needs ≥ 48 hot
  slots per layer.
- **The W that fits overflows.** At about 24 slots, W ≤ 11. W = 8 overflows half the layers (0.50).

**Verdict for v1 (CPU experts off in verify): NO-GO.** The gate (6:3, hybrid draft slots, overflow ≤ 2%,
VRAM-admissible W) finds no admissible lane count. Every row loses, even with the gate's limits dropped:
- every one of the 60 rows is below the baseline with zero draft time;
- the best is 10.04 tok/s against 13.17;
- the draft budget at parity (`stride × 75.94 − verify ms`) is negative in all 60.

Every modelling simplification favours verify:
- 14 ms of GPU for a 6-token verify leaves out the attention and Engram breaks (§33.3 item 6);
- rejected draft tokens route at least as diversely as the true tokens that stand in for them;
- residency decays once per verify, not once per token.

**v2, multi-token CPU experts in verify (§33.3 item 5, size L): undecided.** A second run at `9bd866391b`
(`projection2.{md,json}`, same command) costs the same misses with the simulator's own best per-layer CPU split
(`verify_cpu_ms`, `CostModel.best_k`). It sweeps two unmeasured inputs: CPU cost per row as a multiple of the
calibrated 0.63 ms (a verify's expert can serve several tokens), and GPU ms per verify.
- **The CPU amortizes better in a verify than in plain decode.** A verify has 5.2 RAM-hit rows per layer against
  1.7 for one token, so it pays the handoff once and overlaps more of the CPU with the link. That is why scaling
  plain decode's 31% saving understated it.

  | arm (no draft time) | 1.0× CPU, 14 ms | 1.5×, 28 ms | 2.0×, 42 ms |
  |---|---|---|---|
  | plain decode, ideal split (1.0×, 14 ms only) | 14.16 | | |
  | 6:3, W 32, draft slots 0 | 17.93 | 14.57 | 12.58 |
  | 6:3, W 32, draft slots 4 (hybrid) | 16.47 | 13.34 | 11.52 |
  | 6:4, W 32, draft slots 4 | 17.37 | 14.23 | 12.41 |

- **The band.** Against the served 13.17 tok/s it runs from +25% to −13% for the hybrid draft at 6:3. A graphed
  draft's time per verify comes off the top. Tok/s rises mildly with stride (6:2 → 6:4 with the hybrid's slots: 14.8 → 17.4 at 1.0×, 14 ms).
- **It needs item 1 solved first.** Every W ≥ 24 row fails `capacity ≥ 2W`. At W ≤ 11, half the layers overflow, and
  the model does not cost the overflow path (⌈U/W⌉ records per layer, or an eager re-verify).
- **Three measurements decide v2 before any build:**
  1. GPU ms of a 6-token target verify forward with the attention and Engram breaks;
  2. CPU ms per expert row when an expert serves 1-6 tokens. §33.4's kernel bench suggests ≈1.0-1.3×: 21 experts per
     layer cover 36 routes, so about 1.7 tokens each at ≈⌈t/2⌉ weight passes. That is an inference, not a measurement;
  3. what the `capacity ≥ 2W` rule (`expert_residency_gpu.py:377-383`) really requires at W ≈ 21.

  If 1 and 2 land near 1.0-1.5× and ≤ 28 ms, and item 1 has a cheap answer, v2 is worth a build plan. Otherwise
  graphed DSpark stays shelved, and §33.4's eager hybrid draft remains the only DSpark path.

### 33.6 D2-1: the route plan and the in-graph MoE over M tokens (2026-10-05)

Plan `docs/superpowers/plans/2026-10-05-dsv41-dspark-graph-d2-1-multitoken-moe.md`, branch `dsv41-dspark-graph`.
This is the first of four v2 plans. It covers §33.3 items 3 and 4.

**What changed.**
- **Route plan (item 3), `93213724ca`.**
  - A second fused planner kernel, `plan_dedup_routes_kernel` (`expert_route_plan.cuh`), plans up to 64 routes of
    several tokens in one launch. It uses one block of 64 threads and dedups by first occurrence, so a slot shared
    across tokens is planned once.
  - The one-warp BS1 kernel is unchanged and still serves `topk_ids.shape[0] == 1`.
  - `supports_fused_graph_routes` admits multi-token calls up to 64 routes.
- **Route tables, `bc832c0a88`.**
  - `route_tables` and the layer-fusion kernel `exl3_moe_route_tables` now take `[M, H]` and up to 64 routes.
  - Both rank routes stably (torch: `argsort(stable=True)`), so they agree bit for bit when tokens share a slot.
  - Both write each rank's token (`route // top_k`) as exllamav3's `token_sorted`.
  - The launcher refuses CPU experts for M > 1.
- **The MoE (item 4), `aeafeac263`.**
  - `Exl3FusedMoE(tokens=)` sizes its route buffers to `tokens × top_k` and runs any 1 ≤ M ≤ tokens by slicing them.
  - `exl3_fused_moe_for` sets `tokens = graph_gather_rows // top_k` and refuses more than `ROW_TILE` = 16 tokens.

**exllamav3 needed no change.** This closes item 4's "Unverified" line. `exl3_moe` is multi-token already:
- `token_sorted` maps each sorted route to its input row;
- experts are handed out by ticket, so a slot carries as many rows as `expert_count` says, up to its 16-row tile;
- `num_active` only sizes the launch;
- `exl3_moe_gather` sums per token.

**Gate.** `test/manual/dsv41/test_exl3_fused_moe_multitoken_gpu.py` (`919f726c79`) ran on real layer-3 rows: 16
experts in 16 slots, top-6, 8 route sets per M, with tokens sharing slots.
- It passed every bar: rel ≤ 1.2e-2 and ≤ 2× the per-expert loop's error; layer fusion equal to the torch chain bitwise;
  eager reruns and graph replays with rewritten inputs bitwise equal; M = 1 after M = 6 equal to a fresh object.
- The P2 probe passed in the same run, so BS1 is intact.
- Raw data: `cc-expert-prediction/analysis/dsv41-dspark/graph-verify/d2-1-multitoken.{json,log}`.

| M | max rel_fused | max rel_loop | eager µs | replay µs | num_active |
|---|---|---|---|---|---|
| 1 | 1.10e-3 | 1.15e-2 | 97.0 | 96.6 | 6 |
| 2 | 1.00e-3 | 1.21e-2 | 267.4 | 266.5 | -1 |
| 4 | 1.00e-3 | 1.12e-2 | 272.3 | 271.4 | -1 |
| 6 | 1.01e-3 | 1.08e-2 | 274.2 | 274.3 | -1 |

**The first measured input to §33.5's "verify GPU ms".** This covers the MoE kernels only, for one layer of 16 slots.
- Replay at M = 6 costs 2.84× M = 1 (274 vs 97 µs).
- Almost all of that is the launch size, not the tokens. M = 2 already costs 267 µs, and the probe's own run gives
  `num_active = -1` 279 µs of replay at M = 1, against 113 µs at 6.
- **`num_active = -1` at M > 1 is untuned.** A launch sized to the distinct-slot bound (≤ min(M × top_k, slots)) is
  the obvious next measurement. At 40 layers, today's figure adds ≈7 ms per verify over BS1's MoE.

**What D2-1 does not do:**
- the record and wire at width W (D2-2, item 2);
- DIRECT's `capacity ≥ 2W` and the overflow path (D2-2);
- the end-to-end graphed verify with CPU experts off, the small items and the gate (D2-3, items 6-8);
- multi-token CPU experts (D2-4, item 5).

### 33.7 D2-2: a miss width below the routes, and the overflow flag (2026-10-05)

Plan `docs/superpowers/plans/2026-10-05-dsv41-dspark-graph-d2-2-miss-lanes.md`, branch `dsv41-dspark-graph`.
This is the second of four v2 plans. It covers §33.3 items 1 and 2.

**§33.3 item 2 is out of date.**
- The record is no longer 8 lanes. `LeaseLayout<NumLanes, NumNodes>` takes 1-32 lanes, one JIT build per lane count
  (`0b82fc37c5`, `5813a147d5`, `7a77475a20`).
- The real one-token assumption was the unit the width counted: `graph_gather_rows` counts routes (36 for a 6-token
  verify at top-6). From it came the lane build (`plan_gather_width` raises above 32), DIRECT's victim shortlist and
  `capacity ≥ 2 × rows` floor, the allocator's floor, and the attach check `graph_gather_rows > lanes`.

**What changed.**
- **The fused DIRECT gather takes up to 64 routes, `3ad17dfb8a`.**
  - The kernel is one warp. It translated a route's remap only where `lane < top_k`, and its launcher refused more
    than 32 routes. The remap loop now strides by the warp.
  - The launcher checks the shortlist (1-32) and the routes (1-64) separately.
  - Layer fusion's per-layer remap rows are as wide as the routes, not the shortlist (`f503e4caa3`). BS1 is unchanged.
- **A miss width W separate from the routes, `1136ec5f0e`.**
  - `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES` (an `EnvInt(0)`), or `from_model(graph_gather_miss_lanes=)`, caps each
    layer's distinct misses per gather. It is capped at the routes, and 0 keeps one lane per route.
  - `ExpertStreamer.graph_miss_lanes` and `graph_miss_width` carry it.
  - Routes keep their width: planner, remap and protect list. Misses take W: the lane build, staging, DIRECT's
    shortlist, the allocator floor (`2W`) and the attach check.
  - A width below the routes needs DIRECT (`SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`) and refuses CPU experts.
  - `exl3_fused_moe_for` needs `top_k` resident slots, not one per route.
  - The RAM-miss attach checks a row's staging against W, not the routes (`37cd93df93`).
- **Clamp and flag, `7d06ed4782`.**
  - `GpuResidencyUpdater.clamp_gather_misses` runs after the DIRECT destinations, only when W is below the routes.
    It sets the shared miss count to the lanes that found a victim.
  - `gather_overflow` (int64 per layer) counts the gathers that had more misses than that.
  - `overflow_flag` (int32 [1], sticky) says this forward's output is not a verify result.
  - The metrics snapshot gains `gather_overflow` only when some layer is narrowed.

**Why residency stays exact.**
- The live lanes are a prefix: usable shortlist entries come first, then `lane < count`. So `live.sum()` is the
  number served.
- The post kernel, S, the copy wait and the commit all read one count, `_graph_miss_count`. After the clamp, every row
  copied is committed, and nothing else is.
- Without the clamp, a counted lane that is not live copies into slot 0. Reusing `keep = 0` would be wrong as well:
  the copies have already been issued, and the commit would then skip rows it had overwritten.
- An overflowed forward reads the wrong rows. An unserved miss's route reads slot 0, because a lane that is not live
  has destination 0 (a rank past the shortlist reads its last lane). Slot 0 may be free and hold any bytes, so the
  output can be NaN or Inf, not merely the wrong experts. The flag marks the output, and D2-3 must discard
  everything the flagged forward wrote, not only its tokens.
- A mutant that drops the clamp fails all three narrow-gather tests: the flag stays 0, and served rows are wrong.

**The `capacity ≥ 2W` floor stays, though its guarantee no longer holds for a verify.**
- At one token, `H + M ≤ rows` guarantees every miss a victim. For a verify, `H + M` may exceed W, and the hits may
  disqualify more shortlist entries than there are to spare.
- The flag covers that shortfall. Whether a floor below `2W` is safe is §33.5's measurement 3, still open.

**Gate.** `test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py` (`7d66c99a1c`, `8cc3050e44`) captures a 6-token verify
at top-6 through the real EXL3 lease chain, with the service built for 8 lanes. It passed both arms, generic and
layer fusion.
- One step has 6 shared misses (copied once), then a step of all hits (no rows read), then a union of 8 that fills
  the lanes. These are served exactly, and every token's output is within the probe's bar.
- A union of 12 serves 8 and flags the forward, with `gather_overflow` + 1 and no trap or fail-stop. Every mapped
  slot still holds its checkpoint bytes, and `insertion_truncated` stays 0.
- The next step is served exactly again.
- The BS1 lease-chain, lease-kernel and layer-fusion files stayed green in the same run.
- The NVFP4 residency tests cover the rest:
  - a narrow gather equals the one-lane-per-route gather bit for bit until its first overflow;
  - hits can empty the shortlist even when the misses fit, and the gather is flagged with nothing copied;
  - a one-token gather never calls the clamp.

**What D2-2 does not do:**
- D2-3 owns the rest of the graphed verify:
  - reading and clearing the flag, and re-verifying an overflowed verify;
  - the gate lift;
  - the protect list, which still truncates silently at the lane count (recency stamps only, §33.3 item 6).
- W is not chosen. That waits for D2-3's overflow rate on real routes, against §33.5's projection; in the offline
  model W = 8 overflows half the layers.
- D2-4 owns CPU experts at W < routes.

### 33.8 D2-3: the DSpark verify in the decode graph, end to end (2026-10-05)

Plan `docs/superpowers/plans/2026-10-05-dsv41-dspark-graph-d2-3-graphed-verify.md`, branch `dsv41-dspark-graph`.
This is the third of four v2 plans. It covers §33.3 items 6-8, and measures the overflow rate and verify time that choose W.

**What changed.**
- **The manager reads and clears the flag, suspends the graph gather, and traces both, `bea76848d7`.**
  - `take_verify_overflow()` does one `.item()` on the sticky `overflow_flag`, zeroes it when it is set, and counts
    `graphed_verify_ct` and `verify_overflow_ct`.
  - `suspend_graph_gather()` turns the gather off on every streamer and restores it in `finally`.
  - The metrics trace gains `graphed_verify` and, when narrowed, `gpu_residency:gather_overflow`.
- **The DSpark re-verify, `1237b9e487`.** `forward_verify_with_reverify` (`dspark_graphed_verify.py`) re-runs a flagged
  graphed verify with `DecodeCudaGraphRunner.eager_only()` and the gather suspended. `SGLANG_TEST_DSPARK_FORCE_REVERIFY`
  re-runs every verify.
- **No epilogue while the gather is narrowed, and no decode graphs for an EXL3 draft**, also `1237b9e487`.
- **The copy-engine barrier, `9f6dcb2407`.** A graphed verify counts toward arming like a graphed decode. Its eager
  re-run is not graphed, so it drains the device first.
- **The gate, `5c7173519b`.** DSpark verify may run in the breakable decode graph when it is a static verify
  (`SGLANG_RAGGED_VERIFY_MODE=static`) on DIRECT residency (graph gather, GPU residency update, insert-on-miss stage 2)
  with `MISS_LANES` 1-32. This replaces the old refusal, "runs DSpark verify eagerly only".
- **Three defects the run found, each fixed test-first:**
  - **A pinned gather's room counted the row's staging slots, `218d31446a`.** `evictable_rows` counted the slots the
    service's lanes own, which `assign` never hands out. The eager re-run then evicted pinned rows before their copy
    ("pinned host rows of experts [24, 25] were evicted"). `NativePinnedSlotTable.reserved_rows` is now subtracted.
  - **The draft's CPU-expert cores were outside the core plan, `ceec0db386`.** The first arm run was refused at start
    (`core 17 shares a physical core with the server's affinity`). The recipe hand-pins the RAM thread to 17, the plan's
    `taskset 0-5,18-63` held its sibling 53, and the hybrid draft's hand-named 6-17 overlapped the copy and RAM threads
    (16, 17), which `ThreadingConfig` never saw. The draft is now a role of `ThreadingConfig`:
    - unset `SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES` derives its cores on the GPU's node from what the copy thread and
      every node's plan leave, capped by `_THREADS`;
    - a named list overrides, is kept out of every derived role, and a given role on its physical core is refused.

    The D2-3 driver names no cores.
  - **A captured verify took the prefill graph's break-points, `3c9d547cbd`.** `ForwardMode.is_extend()` counts
    `TARGET_VERIFY`, so four break-points meant for the breakable prefill graph fired while the decode graph captured
    a verify: the Engram hash ids, low-ratio sources, MQA attention, and the backend's low-ratio projections. Each
    reads the prefill runner's piecewise forward context, which the decode runner never sets, and capture died in
    `deepseek_v4_engram_hash_ids`. `is_in_breakable_prefill_graph(mode)` names the condition once, and a verify now
    takes the in-graph paths, as decode does.

**Why a re-run, and why eager.**
- Re-running in the graph would overflow again: the narrowed gather is chosen by route count alone.
- The eager MoE (`_apply_streamed`) is the path DSpark verify ran on before D2, so a re-run is the old verify.
- Everything the flagged forward wrote is overwritten: its KV and compressed-cache writes go to the same
  `out_cache_loc`, and its logits are discarded.
- **Known bias:** the residency and recorder counters count an overflowed verify twice, once per forward.

**The protect list needs no wire change.**
- The post kernel keeps the first `Wire::kLanes` distinct routes (`lease_kernels.cuh:164-169`).
- With no overflow, every route is either a VRAM hit or a lane, and lanes enter `wanted` on their own.
- So the routes the truncation drops are exactly an overflowed verify's clamped misses, and that verify is re-run.

**The run.** Arms: `eager` (§33.4's hybrid draft, eager verify), `graphed` (W = 8), and `reverify` (graphed, every verify
re-run eagerly). Each runs 8 sessions of 256 prompt tokens and 128 new tokens, under the recipe's server cores 0-5,36-41.
Core plan, from the logs: copy thread 17, RAM threads 16 and 35, draft CPU experts 6-15 (10 workers; §33.4 had 12).

```bash
flock rowimg-disk.lock flock cc-gpu.lock taskset -c 0-5,36-41 \
  python analysis/dsv41-drive/dspark/graphed_verify.py $G eager graphed reverify
```

The eager arm ran at `ceec0db386`; `graphed` and `reverify` ran at `3c9d547cbd`. The last fix changed only paths an
eager arm never enters. Raw data is in `cc-expert-prediction/analysis/dsv41-dspark/graph-verify/d2-3/`
(`summary.json`). The earlier refused and capture-failed runs are kept in its subdirectories.

| arm | tok/s | accept length | verify ms mean / p50 / p95 (n) | re-verify rate | layer overflow mean / max | text = eager |
|---|---|---|---|---|---|---|
| eager | 2.22 | 2.37 | 862 / 872 / 989 (183) | - | - | - |
| graphed, W 8 | 2.84 | 2.37 | 831 / 828 / 967 (354) | 1.00 | 0.75 / 1.00 | 8 / 8 |
| reverify | 2.75 | 2.37 | 826 / 825 / 973 (334) | 1.00 (forced) | 0.75 / 1.00 | 8 / 8 |

- **Bars: all met.**
  - The re-verify-all arm's text equals eager's in 8 of 8 sessions, so the graphed forward leaves no state the re-run
    reads.
  - `insertion_truncated` is 0 in both graphed arms.
  - Every arm finishes its 8 sessions with identical token counts.
  - The verify graph is captured (breakable, 3 segments, 2 breaks), with no `RuntimeError`, trap or fail-stop.
- **At W = 8, every graphed verify overflows.** 380 of 380 were re-run, and 75% of layers overflow per verify (max
  100%). §33.5 projected 0.50 of layers at W = 8; real routes are worse.
- **The graphed arm's tok/s gain is not a graph gain.** Every verify was re-run eagerly, so each graphed verify paid a
  replay and an eager forward. The +28% over `eager` comes with the graphed arms' expert-stream configuration (DIRECT
  in-graph residency, the fused plan, prefill fills), which the eager arm turns off. Its re-runs run on that residency.
- **Verify ms is GPU-event time over the `TARGET_VERIFY` segment.**
  - It includes waiting on RAM and NVMe misses. In the graphed arms it covers the replay plus the re-run.
  - Not every verify has a record (n < 380), so the n column is reported.

**What it decides, against §33.5.**
- **Measurement 1 is not answered by this run.** The graphed verify ms (≈830 ms) is an upper bound on replay + eager
  re-run + miss waits, not on the GPU compute of a graphed verify. §33.5's ≤ 28 ms test needs a verify that does not
  overflow, which W = 8 never produced.
- **The overflow path is not cheap at an admissible W.** At W = 8 it is taken by every verify. W ≥ 24 is what §33.5
  needed for low overflow, and DIRECT's `capacity ≥ 2W` floor cannot fit that in today's hot slots.
- **So by §33.5's rule, v2 (D2-4, multi-token CPU experts) is not worth building on this evidence.** Graphed DSpark
  stays shelved, and §33.4's eager hybrid draft remains the DSpark path.
- **What would reopen it:**
  - a W sweep (`D23_MISS_LANES` 16, 24, 32) to find where the re-verify rate falls;
  - more hot slots for the `2W` floor, freed from VRAM (the Qwen NextN lever, +6% there);
  - §33.5's measurement 3 (what `capacity ≥ 2W` really requires).

**What D2-3 does not do:**
- multi-token CPU experts (D2-4);
- the epilogue under a narrowed gather;
- decode graphs for an EXL3 draft;
- the num_active launch sizing (§33.6);
- choosing W, which the sweep above would do.

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
