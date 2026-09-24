# DSV41 row images: read expert rows straight into the pinned slabs

Goal: remove the RAM-miss reader's bounce buffer and pack copy. io_uring reads each sub-read with one O_DIRECT
`readv` whose iovecs are the pinned slab rows it fills, and a piece is published as soon as its sub-read lands.

Why: the final 100 GiB smoke at `92e588b20e` averaged 13.6 read demands per decode step, each with a pack tail
(last CQE to last piece published) of p50 184 us, mean 220 us, while the stream kernel waits for that last piece.
Ceiling about 2.5-3 ms/token. It also frees the 8 packing cores and node-0 copy bandwidth (13.7 GB/s ceiling).

Feasibility (probe `b199a39d47`, `analysis/dsv41-drive/direct-read-probe/`): XFS (nvme0) and ext4 (nvme4) report
`dio_mem_align 4`, `dio_offset_align 512`; readv into mbind'd, cudaHostRegister'd slab rows at non-page offsets
lands every byte correctly, from both drives, slabs on either node; throughput equals the bounce path
(3.55 vs 3.56 GB/s, 3.46 vs 3.50 GB/s). A segment length or file offset off 512 is refused (EINVAL).

Why a new file: in the checkpoint every row starts at an odd offset and the 4-byte `mul1`s shift later tensors, so
a tensor boundary is never 512-aligned and no readv can split a checkpoint row into the slabs. A **row image**
stores each row as its six slab rows back to back; every slab row is a multiple of 512 bytes
(20480, 9216, 8847360, 4608, 10240, 4423680), so every name starts 512-aligned.

## Contract (done, owned by the controller; do not edit, report needed changes)

`python/sglang/srt/layers/moe/exl3_row_image.py`, tests `test/registered/unit/kernels/test_exl3_row_image.py`:

- Directory `<mirror root>/exl3_row_images/`: `layer-LLL.rows` (expert `e` at `e * row_stride`, `image_bytes` of
  image then zero padding to `row_stride` = image rounded up to 4096) and `manifest.json` (written last, atomic).
- `row_image_layout(segments)` -> `RowImageLayout` (row_bytes / name_offsets in `EXL3_STREAMED_NAMES` order,
  image_bytes, row_stride). Byte mapping: `image_of_row(layout, raw_row)`. Digest: `row_digest(image)`.
- `source_fingerprint(layout, source_root)`, `manifest_json(...)`, `write_manifest(root, m)`, `read_manifest(root)`.
- `open_row_images(roots, layout, segments, source_root, layer_ids)` -> `RowImageSet` (validated paths per layer
  per root), refusing incomplete, mismatched, short or cross-root-inconsistent sets.

dsv41-full40: image_bytes 13,315,584 (26,007 x 512), row_stride 13,316,096, 40 layers x 384 experts,
~205 GB per root. Space: nvme0 757 GB free, nvme4 1.4 TB free.

## Shared decisions

- Flag: `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES` (`EnvBool(False)`, in `environ.py` next to the piece-stream flag).
  On: the native RAM-miss reader reads `<root>/exl3_row_images` of every root in `SGLANG_MOE_EXPERT_MIRROR_DIRS`,
  split by `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS` exactly as the mirrors are. Refused without mirror dirs or without
  `uring_direct`. The eager row source (tier fill, eager path) keeps reading the mirrored shards, unchanged.
- The slabs, the copy tables, the device kernels and the lease protocol do not change. With images the reader's
  segment table is six identity segments `[name, dst 0, src name_offset, row_bytes]` and `starts` is all 0, so
  `row_geometry` gives piece `j` = exactly sub-read `j`'s bytes (cuts are 512-aligned, so already 128-aligned).
- Branches off `cc/dsv41-pinned-numa` at the contract commit: `cc/rowimg-converter`, `cc/rowimg-reader`,
  `cc/rowimg-harness`, each pushed to `shared`. divix01 private worktrees `/data/models/slang/nvfp4-work/wt-rowimg-<part>`.
  GPU work only under `cc-gpu.lock`; production stays stopped. The controller merges.

## Part A: converter (`cc/rowimg-converter`)

Owns `scripts/dsv41/build_row_images.py`, `test/registered/unit/kernels/test_build_row_images.py`.

1. CLI: `--model-dir` (the source, as `SGLANG_DSV41_EXPERT_DIR`), `--roots` (os.pathsep list), `--layers`
   (default all), `--verify` (re-check an existing set without writing), `--threads`.
2. Layout from `build_exl3_expert_layout` and the format's segment map (`_row_schema`); image layout, mapping,
   digests and manifest only through `exl3_row_image`.
3. Read each source row once; write its image to every root. Per layer: write `layer-LLL.rows.tmp`, fsync, rename.
   Keep the page cache out of it (O_DIRECT or `POSIX_FADV_DONTNEED`). Resumable: a finished layer leaves a digest
   sidecar; a rerun skips a layer whose file and sidecar agree, after re-reading it.
4. Verify every written layer by reading it back with O_DIRECT and comparing digests; only then write the manifest
   on each root. Refuse a root that is the source dir, lacks space, or already holds a manifest for another source.
5. Tests: synthetic checkpoints (odd row offsets, the real 12-tensor shape), resume, interrupted run leaves no
   manifest, a corrupted byte fails `--verify`, `open_row_images` accepts the output.
6. On divix01: build all 40 layers on `/mnt/nvme0/dsv41_flash` and `/mnt/nvme4/dsv41_flash`, then `--verify`.
   Under `taskset -c 0-63`, `ionice -c3`. Record wall time and the command.

## Part B: reader (`cc/rowimg-reader`)

Owns `exl3_ram_miss_host.cpp`, `exl3_ram_miss_pack_pool.h`, `kernels/ops/moe/exl3_ram_miss.py`,
`srt/layers/moe/exl3_ram_miss.py`, `exl3_expert_format.py` (mirror/table args), `environ.py`, the DSV41 config,
`python/sglang/test/dsv41_ram_miss_fixtures.py`, the reader's tests, `LEASE_PROTOCOL.md` if a rule changes.

1. `exl3_ram_miss_tables(..., row_images=RowImageSet)`: files per layer per root; extents
   `(file, e * row_stride + part_start, part_bytes, part_start)` with the policy's page-aligned cuts over
   `image_bytes` (the last part ends at `image_bytes`: never read the padding, it would overrun the last slab row);
   `starts` 0; identity segments; a flag telling the host the tables are direct.
2. C++ direct mode: no bounce, no pack pool. Each descriptor carries a preallocated iovec array (at most 6);
   `io_uring_prep_readv`; a short read resumes by advancing the iovecs past `done`. Refuse at open any slab row,
   extent offset or length not 512-aligned.
3. Piece streaming: sub-read `j` vetted -> publish piece `j` at once. Flag off: a row is done when its parts land.
   Keep the trace meaningful (pack stamps = publish time in this mode; say so next to `STAGE_FIELDS`).
4. Failure: I/O lands in unpublished slots; the ring is drained before `read()` returns, as today; the caller's
   quarantine/release is unchanged. The poison fault poisons the slot's slab rows at admission.
5. Wiring: `Exl3RamMissService.ensure_started` opens the images when the flag is on (layers = streamed layers).
6. Tests: run the split/thread/pack-worker suites' promises in direct mode (parametrize the mode as the
   pack-worker file does); byte-identical slabs vs bounce mode; short read across an iovec boundary; EIO; refusals.
   GPU: the piece-stream CUDA tests in direct mode. A reference image builder for fixtures lives in the fixtures
   file, built from `exl3_row_image` primitives.

## Part C: integration and measurement (`cc/rowimg-harness`)

Owns `analysis/dsv41-drive/row-images/` and divix01 scripts under
`/data/models/slang/nvfp4-work/direct-two-phase-tests/row-images/`.

Phase 1 (parallel with A and B):
1. Smoke arms from `pinned-numa/smoke_100_audit.sh`: `base` (current settings) and `rowimg` (+ the flag).
2. `publish_latency.py`: per piece `publish - cqe`, per demand host tail (`done - last cqe`), rows/demand, per arm.
   ms/token from `graph_step` records; responses compared across arms.
3. A fresh `base` 100 GiB smoke at the contract commit for same-day numbers.

Phase 2 (after A and B are merged, on the controller's signal):
4. Images built and verified (Part A's run). CPU suite (`test/registered/unit/kernels`) and GPU manual suite on
   the merged head, counts against the merge base.
5. Two `rowimg` smokes: responses byte-identical to `base`, pack tail gone, ms/token delta, stalls.
6. Write-up: `DSV41_REFERENCE.md` new section, `PACK_WORKERS.md` note.
