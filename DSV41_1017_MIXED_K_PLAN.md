# DeepSeek-V4.1-Flash-1017 (EXL3, mixed K): plan and open work

Started 2026-10-23. Production is down until this serves. Update the checkboxes as phases land.

## The checkpoint

- `divix01:/mnt/nvme0/dsv41-1071/dsv41-1017`: EXL3 1.6.2, `avg_bits` 2.9 (no `bits` key), 43 shards, 197 GB.
- Routed experts: K3, except layers 18-22 at K2 (rows 8,891,916 B on disk against 13,315,596). Shared: K5 or K4;
  layer 29's shared w1/w3 are K4 and its w2 K5. Engram wkv: layer 1 K5, layer 14 K4. Inline `mtp.*` (3 layers):
  experts K4, attention and shared K6. `exl3_k_map.json` describes this, but it is not part of the EXL3 format.
- Engram tables: `/mnt/nvme0/dsv41-1071/engram` (`model-00047` layer 1, `model-00048` layer 14; the owner is
  downloading 00047). The new config's `engram_table_dir` is null.
- Mirrors: byte copies of the shards on `/mnt/nvme{0,4,2}/dsv41_flash` (done 2026-10-23, 43 shards each). Row
  images are not built yet.
- Removed 2026-10-23 (owner): the old mirrors and `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw`. Kept: the FP8
  original `/mnt/nvme2/DeepSeek-V4.1-Flash`.

## Design rule (owner, 2026-10-23)

The stack serves **any** assignment of EXL3 bit widths, per layer and per tensor, read from the checkpoint. Nothing
assumes a K or a row size. Specialized C++ kernels stay templated, but Python decides once at startup which
specialization each layer uses; the kernel never probes per call.

- **Source of truth:** the safetensors headers. A trellis `[k/16, n/16, 16*K]` gives K per tensor.
  `quantization_config.bits`/`avg_bits` are informational. `exl3_k_map.json`, when present, is cross-checked
  against the headers and a disagreement refuses the load.
- **Row geometry:** the experts of one layer share a geometry (spans, row bytes, K of gate/up and of down); layers
  may differ. w1 and w3 must match (they share the `w13` slabs); w2 may differ. The layout holds the distinct
  geometries and each layer's index into them.
- **Kernel plans:** the CPU extension lists its compiled specializations (name, hidden, intermediate, K of w13 and
  w2, activation limit, ISA floor). Python matches each layer's facts against that list at registration, logs the
  choice per layer, and passes the plan id in the layer's params. C++ refuses a plan that does not accept the layer
  at registration, then dispatches on the id.

## Phases

### Phase 0: unblock
- [ ] `Exl3Config.from_config` (`quantization/exl3/exl3.py:87`): accept a config without `bits` (`avg_bits` only).
- [ ] Engram: once 00047 lands, open both tables through `EngramFileTable` and point production at the new dir.
- [ ] Draft dir: extract the inline `mtp.*` layers (`scripts/dsv41/make_dspark_draft_dir.py`, check it carries
      K6 attention and shared) and load the drafter.

### Phase 1: per-layer expert layout (Python)
- [ ] `exl3_expert_layout.py`: geometries plus each layer's geometry index. Uniformity is checked within a layer,
      not across layers. Cross-check `exl3_k_map.json` if present.
- [ ] `expert_engine/formats/exl3.py`: `_row_schema` per geometry; each layer's format uses its own.
- [ ] `exl3_row_image.py` `source_fingerprint`: covers every geometry and the layer map.
- [ ] Tests: a synthetic checkpoint with K3, K2 and a w2-differs layer; the real headers on divix01 (CPU only).

### Phase 2: row images (after 1)
- [ ] `exl3_row_image.py`: the manifest records each geometry's image layout and the layer map. Bump `VERSION`.
- [ ] `scripts/dsv41/build_row_images.py`, `verify_expert_mirror.py`: per-layer geometry.
- [ ] Build and fully verify the images on the three roots (~196 GB per root).

### Phase 3: native reader tables (after 1, alongside 2)
- [ ] `expert_engine/formats/exl3_tables.py` and `row_tables.h`/`row_reader.h`/`piece_geometry.h`: `row_bytes`,
      segments and `need_end` per layer; `slot_bytes` is the largest layer's; the slab check is per layer.
- [ ] Mutant: a K3-sized run aimed at a K2 slab must go red.
- [ ] C++ change: full JIT rebuild and the expert-stream/RAM-miss GPU groups (`-n 8` after a serial warm-up).

### Phase 4: CPU expert kernel configured at startup (alongside 1-3)
- [ ] `SglangExl3CpuParams`: K of w13 and of w2, and the plan id. `row_bytes` (quant.hpp) per projection.
- [ ] An op listing the compiled plans; `schemes/exl3_cpu_experts.py` picks one per layer at registration.
- [ ] `run_plan`/`dispatch`: dispatch on the id, no per-call `accepts`. Registration refuses a plan whose facts do
      not match the layer.
- [ ] Tests: K2 and a mixed w13/w2 layer against the reference on the generic plan; the K3 layer still takes the
      fixed plan (`exl3_cpu_plan_calls`); a forced mismatch is refused.

### Phase 5: production settings (after 2-4)
- [ ] `arm_env.py`: MODEL/EXPERT/ENGRAM/draft paths; the pinned size from the checkpoint's per-layer row bytes
      instead of `PROD_PINNED_ROW_BYTES`. `run_arm.sh` paths. Settings tests.
- [ ] Route-fitted artifacts are not used for the first serve (see below).

### Phase 6: first serve
- [ ] Start production: KV pin 381,184 (watch the second set of eager staging buffers, ~570 MB, for the K2
      layers), per-layer geometry and CPU plan choices in the log, the startup calibration.
- [ ] Quality: the smoke prompt.
- [ ] Throughput and acceptance length against the last recorded numbers (no A/B: the old model is deleted).

## Follow-ups (not in this push)

- **Route artifacts, after the first serve:** recapture and refit the hot-cache seed, `resident-top32.json`,
  `ram_prefetch_margin_floors.json`, the prefetch heads and the confidence-cut STS/T(w) fits on the new model.
- **Hot-cache calibration:** price each layer's slots at its own row size (today the first cache's, which is
  conservative); the seed score is frequency times bytes, which biases against the K2 layers.
- **Split-model calibration:** per-layer (or per-geometry) CPU and copy costs; today one global set misprices
  layers 18-22.
- **CPU kernel:** a fixed fast plan for K2 (layers 18-22 run the generic plan).
- **Pinned tier:** layers 18-22 free about 8 GiB of pinned RAM; decide whether it goes to the hot cache or KV.
- **Comment fix:** `ram_thread.h:24-25` says a busy-poll thread spins with no PAUSE; `park()` pauses anyway.
