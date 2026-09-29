# THP fallback and io_uring buffer registration on divix01

2026-09-29, branch `cc/thp-fallback` (cut from `ba01695c35`). Kernel `6.12.0-211.60.1.el10_2`, THP `enabled=always`,
`defrag=madvise`, only 2 MiB THP (every mTHP size `never`), liburing 2.12.

## Verdict

- **Root cause confirmed, and refined.** Registration is slow because of the kernel's pin accounting
  (`io_buffer_account_pin` → `headpage_already_acct`). It walks, for every huge page of a new chunk, the chunk's own
  page array and every bvec of every buffer registered before it. A chunk containing even one 4 KiB page does not
  coalesce, so it keeps 262,144 bvecs per GiB. What costs time is **where** the 4 KiB pages sit relative to the huge
  pages registered after them, not what fraction of the tier they are. 0.18 % of a 16 GiB tier, placed in 15 chunks,
  takes registration from 0.5 s to 69 s.
- **No mitigation is shipped.** One fix worked: registering each chunk in a scratch ring and cloning it into the table
  with `IORING_REGISTER_CLONE_BUFFERS`. It cut a 100 GiB registration on the same page layout from 18.9 s to 1.1 s.
  But on this kernel it **leaks page pins**: the pinned pages stay allocated after both rings and the process are
  gone. It was reverted (`f82f61f3d7`). The leak stranded ~149 GiB on divix01, which only a reboot frees
  (see "Incident").
- The mitigations that stay at user level and on the allocation side all failed at full size (see the table below):
  - `MADV_HUGEPAGE`;
  - re-faulting the non-THP ranges;
  - `MADV_COLLAPSE`;
  - reordering the registrations;
  - registering smaller chunks directly.

  The node has too few order-9 blocks, and compaction cannot make more. What remains needs root or a kernel change
  (see "Options that remain").
- The branch carries **no code change**: `git diff origin/master -- python test benchmarks` is empty. It holds only
  this analysis.

## Method

`thp_probe.py` builds the tier the way production does. It uses `build_tier` from
`test/manual/dsv41/test_numa_aligned_registration_growth.py`: one `allocate_host_slab_arena` per layer, the dsv41 EXL3
row sizes, and rows bound by `host_numa.allocate_bound`. It then faults the tier with `fill_`. Next it classifies every
present page with the **unprivileged `PAGEMAP_SCAN` ioctl** (`PAGE_IS_HUGE` = PMD-mapped THP; no root needed). It
registers every named slab with the production `RegisteredBufferTable` in the planned row-aligned ≤1 GiB chunks,
timing each chunk.

For each chunk it predicts the struct-page visits `headpage_already_acct` makes:

```
visits(c) = Σ_{new head page h in c} index(h in c's page array)     (own array)
          + H(c) · Σ_{c' registered before c} bvecs(c')                 (earlier buffers)
bvecs = one per folio if every page of the chunk is THP, else one per 4 KiB page
```

Chunk time is then fitted against visits by least squares. `summarize.py` prints the tables. The runs used
`sweep.sh natural`, `sweep.sh strategies` and `sweep.sh inject`, each under `rowimg-disk.lock` → `cc-gpu.lock`, with
the probe on `taskset -c 32-63`. Every run printed `sglang.__file__` from the worktree. Raw JSONL is in
`divix01:/mnt/nvme1/thp-fallback/`.

## Measured 4 KiB fraction (full size: the production split 0:61440,1:40960, 100 GiB)

Registration here is direct, i.e. production's path at master. Runs are in time order: each run changes the
fragmentation the next one finds.

| run | 4 KiB after fault | chunks with 4 KiB pages | `thp_fault_fallback` | register s | model fit r² (ns/visit) |
|---|---|---|---|---|---|
| no madvise (production) | **21.6 GiB (21.6 %)** | 71 | 11,157 | 23.5 | 0.981 (11.6) |
| `MADV_HUGEPAGE` before the fault | 9.5 GiB (9.5 %) | 43 | 4,858 | **103.5** | 0.991 (8.9) |
| no madvise, again | 9.3 GiB (9.3 %) | 33 | 4,789 | 37.9 | 0.994 (12.4) |
| no madvise + re-fault non-THP 2 MiB frames | 8.7 → 8.7 GiB (4,440 frames re-faulted, none became THP) | 31 | 4,472 | 23.0 | 0.993 (8.7) |
| no madvise + `MADV_COLLAPSE` non-THP frames | 7.8 → 7.5 GiB (3,839 of 3,982 calls failed) | 28 | 3,982 | 72.7 | 0.827 (8.0) |
| no madvise (strategies run) | 13.5 GiB (13.5 %) | 45 | 6,929 | **18.9** (direct) | — |

- In production, 8–22 % of the tier lands on 4 KiB pages, depending on fragmentation at launch. The earlier estimate
  in `uring-reg/results.md`, 14.6 GiB, falls inside this range.
- Before these runs, node 0 had 85–88 GiB free+cache, but only ~45 GiB of it was in order ≥ 9 buddy blocks. So a
  61 GiB node-0 share must fall back somewhere.
- Registration time does **not** track the fraction (21.6 % → 23.5 s, 9.5 % → 103.5 s). It tracks the visit count,
  which the per-chunk model predicts with r² 0.98–0.99.

## Scaling law (16 GiB, controlled)

For these runs the tier was all THP (`MADV_HUGEPAGE`, 0 bytes of fallback). The probe then forced 2 MiB onto 4 KiB
pages in each of K evenly spaced big chunks, using `MADV_NOHUGEPAGE` on that piece before the fault.

| K mixed chunks | 4 KiB share | register s |
|---|---|---|
| 0 | 0 % | 0.52 |
| 1 | 0.01 % | 5.52 |
| 2 | 0.02 % | 11.48 |
| 4 | 0.05 % | 24.39 |
| 8 | 0.10 % | 35.25 (fit r² 0.998, 6.4 ns/visit) |
| 15 | 0.18 % | 68.75 (fit r² 0.999, 6.2 ns/visit) |

The mixed-chunk counts for K = 2 and 4 are missing: those runs predate a sort fix in the model (`620d26eebe`).
Their times are valid.

The law:
- Each mixed 1 GiB chunk costs about **0.85 s per GiB of THP registered after it**:
  512 head pages/GiB × 262,144 bvecs × 6–12 ns.
- It also costs up to ~0.4 s for its own array.
- Coalesced chunks cost ~3 ms per earlier GiB, which is the floor in `uring-reg/results.md`.

So registration grows with (mixed chunks) × (tier size). Pure-4 KiB regions registered last cost almost nothing,
because they have no head pages to trigger the walk. That is why the fraction alone predicts so little.

## Mitigations tried

All of the following were measured on the same faulted 100 GiB layout (the strategies run, 13.5 % 4 KiB, 45 mixed
chunks). Each strategy re-registers the tier on a fresh ring.

| strategy | items | register s | shippable? |
|---|---|---|---|
| production, direct (master) | 315 | **18.9** | — (baseline) |
| order: coalesced first, then by head-page count (Smith's rule) | 315 | 18.6 | no gain |
| cap64: 64 MiB row-aligned chunks, direct | 1,935 | 18.9 | no gain |
| **clone**: each chunk pinned in a 1-slot scratch ring, cloned into its slot | 315 | **1.24** | **no: leaks pins** |
| production table with the clone fix (`5ba8ea0400`) | 315 | 1.12 | no: leaks pins |
| clone + 64 MiB chunks | 1,935 | 0.53 | no: leaks pins |
| clone + 256 MiB chunks | 630 | 0.64 | no: leaks pins |
| clone + per-row split of mixed chunks | 7,193 | 0.87 | no: leaks pins |

Allocation-side results (previous table):
- **`MADV_HUGEPAGE`** halves the 4 KiB bytes: direct compaction is allowed under `defrag=madvise`, with 14,826
  compaction stalls and 12,259 compaction failures. But what fallback remains is scattered across the tier, so
  registration got 4× slower. Faulting also took 14.9 s against 8.5 s.
- **Re-fault** (`MADV_DONTNEED` plus a touch under `MADV_HUGEPAGE`) recovered 0 of 4,440 frames.
- **`MADV_COLLAPSE`** failed 96 % of its calls, the same finding as in `uring-reg`.
- The node has no compactable order-9 memory left for these, so no user-level allocation change reaches 0 %.
- **Explicit hugetlbfs / `MAP_HUGETLB`** needs reserved hugepages, a root change, so it was not tried.
- **`MADV_NOHUGEPAGE`** on the whole tier gives no head pages, so no walk (`uring-reg` measured 0.6 s per 32 GiB).
  The full-size run was started but killed during the incident, and its chunk times are contaminated by swap. Its
  decode cost is unmeasured:
  - NVMe reads should not change: `max_segments` × 4 KiB ≥ `max_sectors_kb` on every drive (65 × 4 = 260 ≥ 256;
    128 × 4 = 512 ≥ 512), so the request count is unchanged.
  - The GPU H2D path from 4 KiB-backed pinned memory has not been measured.

### The clone fix, its tests, and why it was reverted

Commits `8ecd187a4b` (tests, RED) and `5ba8ea0400` (fix), both reverted by `f82f61f3d7`.

What the fix did:
- `RegisteredBufferTable` opened a one-slot scratch ring per table.
- Each chunk was pinned there with `update_tag`, then cloned into its slot with
  `io_uring_clone_buffers_offset(..., IORING_REGISTER_DST_REPLACE)`, after which the scratch slot was emptied.
- Headers without the clone call (the fake liburing of `test_expert_stream_uring_options.py`), or a refused clone,
  kept the direct path.

Tests:
- Unit, `test_registered_buffer_table.py`:
  - reads through cloned and direct registrations;
  - no scratch-ring descriptor leaks across re-init or destruction.
- Manual, `test_numa_aligned_registration_growth.py::test_mixed_folio_chunks_register_without_the_quadratic_walk`:
  4 GiB, 3 mixed chunks, cloned **0.49 s** against direct **4.12 s**.
- Mutants, each killed by the unit test (laptop, then restored green):
  - no clone;
  - clone into the wrong slot (the read returns an error);
  - scratch ring never closed (descriptor count grows).

The leak, measured on divix01 after the full-size runs:
- Leaked pages show up as `Active(anon) + Inactive(anon) − AnonPages`: anonymous pages still on the LRU that no
  process maps.
- That figure was flat at **149.5 GiB** with nothing of this work running.
- Controlled repro (`thp_probe.py --gib 2 --madvise hugepage`, then 5 s after exit):
  - direct registration: −2 MiB;
  - production table with the clone: **+1,025 MiB**.
- The clone in this backport keeps page references after the source slot is emptied and both rings have exited.

The unit test did not catch the leak. Its buffers were a few pages, and it did not check the system-wide orphan
count. Any retry of cloning must include the orphan check above as a test.

## Full-size before / after

- **Registration (same layout, 100 GiB):** direct 18.9 s. Clone 1.1–1.2 s, but not shippable (it leaks pins).
  Across launches, direct registration ranged 18.9–103.5 s in these runs. The server itself registered in 59.2 s in
  `uring-reg`.
- **Startup time:** not measured. `startup_arm.sh` and `chain.sh` queued startup pairs (master, branch, master,
  branch) behind the probe sweeps. They were stopped at the incident before any arm launched.
- **Kernels suite at master and at the branch:** not run. After the revert the branch's `python/` and `test/` trees
  equal master's (`git diff origin/master -- python test` is empty), so a branch run would re-run master's code. The
  only suite evidence for the reverted fix is the targeted subset on divix01 at `5ba8ea0400`. Command:
  `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 python -m pytest -q -p no:randomly
  test/registered/unit/kernels/{test_registered_buffer_table,test_expert_stream_uring_options,test_uring_file_reader,test_expert_stream_uring_native,test_expert_stream_uring_integration}.py`.
  Result: 38 passed, 22 skipped, exit 0 (`PIPESTATUS` read).

## Incident: ~149 GiB of pinned memory stranded on divix01

- It happened during the strategies run, which registered the 100 GiB tier through a clone path 5 times,
  and the production-table registration with the fix was one of them.
- The following `--madvise nohugepage` run then found node 0 nearly empty and swapped: 41 GiB of swap, node-0
  MemFree 670 MiB.
- Every job of this work was stopped at ~01:45 (divix01 clock). The orphan count did not fall over the following minutes.
- Freeing it needs a reboot (root).
- Scratch reruns of the clone strategies are disabled in `sweep.sh`, and `thp_probe.py` warns against them.

## Options that remain (all need root or a kernel change, or are unmeasured)

1. **Reserved hugetlbfs pages + `MAP_HUGETLB`** for the tier. Every folio is 2 MiB, so every chunk coalesces and
   registration becomes linear and fragmentation-proof. This needs `vm.nr_hugepages` per node, which is root.
2. **Compact before launch** (`echo 1 > /proc/sys/vm/compact_memory`, root), or change `defrag`. This is best effort
   only: `MADV_HUGEPAGE`'s direct compaction already failed 12,259 times here.
3. **A kernel whose clone does not leak.** Then re-enable `5ba8ea0400`, with an orphan-count test (register, exit,
   check `Active+Inactive(anon) − AnonPages`).
4. **`MADV_NOHUGEPAGE` on the tier** (user level). Registration becomes linear. The decode cost of 4 KiB host pages
   (GPU H2D) must be measured before adopting it.
5. **Leave fixed buffers off**, which is today's default: `uring-reg` kept R0. The cost exists only when
   `SGLANG_EXPERT_STREAM_URING_READ_MODE=fixed|readv_fixed` is turned on, and on drain's ring reset in that mode.

## File overlap with hotpath (`cc/hotpath-zero-overhead`)

None. The reverted fix touched `python/sglang/kernels/jit/csrc/io/registered_buffers.h`,
`test/registered/unit/kernels/test_registered_buffer_table.py` and
`test/manual/dsv41/test_numa_aligned_registration_growth.py`. None of them is in the plan's Files lists or in the
branch's diff. `generations.json` (which hotpath lists) was avoided by using `startup_arm.sh` in place of
`run_arm.sh`.
